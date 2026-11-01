"""A realistic *simulated* LLM provider.

Why simulate?
-------------
1. The project must run on any teammate's laptop with no API keys and no GPU.
2. Benchmarks need *controlled* conditions: we can inject a 2-minute outage,
   a 5 % error rate or a heavy latency tail on purpose and check that the
   gateway handles them. With a real API you can't schedule an outage.
3. Tests become fast and deterministic.

The latency model is log-normal, which matches real network/LLM latency
well: most calls are near the median, and a long right tail of slow calls
(the "p99 problem") is what hedged requests are designed to fix.
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from typing import Any, Iterator

from ..errors import ProviderTimeout, ProviderUnavailable, RateLimitError
from ..types import ChatRequest, estimate_tokens
from .base import LLMProvider, LoggableMixin, MetricsMixin

_VOCAB = (
    "the model considers context tokens attention layer output gradient system request cache "
    "latency provider answer result value function data python thread lock memory vector "
    "matrix cost route fallback stream user prompt response example step first then finally "
    "because therefore however simple clear efficient reliable concurrent design pattern"
).split()


class SimulatedProvider(LoggableMixin, MetricsMixin, LLMProvider, kind="simulated"):
    def __init__(
        self,
        name: str,
        model: str = "sim-model",
        latency_median_ms: float = 400.0,
        latency_sigma: float = 0.45,
        failure_rate: float = 0.0,
        rate_limit_rate: float = 0.0,
        mid_stream_failure_rate: float = 0.0,
        stall_rate: float = 0.0,
        stall_ms: float = 2500.0,
        outages: list[list[float]] | None = None,
        quality: float = 0.8,
        seed: int | None = None,
        time_scale: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(name=name, model=model, **kwargs)  # cooperative init through the MRO
        for label, p in (("failure_rate", failure_rate), ("rate_limit_rate", rate_limit_rate),
                         ("mid_stream_failure_rate", mid_stream_failure_rate), ("stall_rate", stall_rate)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{label} must be a probability in [0, 1]")
        self.latency_median_ms = latency_median_ms
        self.latency_sigma = latency_sigma
        self.failure_rate = failure_rate
        self.rate_limit_rate = rate_limit_rate
        self.mid_stream_failure_rate = mid_stream_failure_rate
        self.stall_rate = stall_rate  # occasional long stalls (queueing, cold starts): the p99 problem
        self.stall_ms = stall_ms
        self.outages = [tuple(w) for w in (outages or [])]  # [(start_s, end_s), ...] since start
        self.quality = quality
        self.time_scale = time_scale
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()  # random.Random is not safe to share across threads
        self._t0 = time.monotonic()
        self.forced_down = False  # tests/demos can flip this to simulate an outage

    # ----- helpers -----------------------------------------------------------
    def _draw(self) -> tuple[float, float, float, float]:
        with self._rng_lock:
            z = self._rng.gauss(0.0, 1.0)
            return self._rng.random(), self._rng.random(), z, self._rng.random()

    def in_outage(self) -> bool:
        if self.forced_down:
            return True
        elapsed = time.monotonic() - self._t0
        return any(start <= elapsed < end for start, end in self.outages)

    def reset_clock(self) -> None:
        self._t0 = time.monotonic()

    def _sleep(self, ms: float) -> None:
        time.sleep(max(0.0, ms) * self.time_scale / 1000)

    @staticmethod
    def _deterministic_text(prompt: str, model: str, n_tokens: int) -> str:
        """Same prompt + model -> same answer, so caching is observable."""
        seed = int.from_bytes(hashlib.sha256(f"{model}|{prompt}".encode()).digest()[:8], "little")
        rng = random.Random(seed)
        n_words = max(3, int(n_tokens * 0.75))
        words = [rng.choice(_VOCAB) for _ in range(n_words)]
        return f"[{model}] " + " ".join(words).capitalize() + "."

    def _completion_tokens(self, request: ChatRequest) -> int:
        # Longer / harder prompts get longer answers, capped by max_tokens.
        base = 40 + 2 * estimate_tokens(request.prompt_text)
        return min(request.max_tokens, base)

    def _latency_ms(self, z: float, completion_tokens: int, u_stall: float = 1.0) -> float:
        # log-normal time-to-first-token plus a small per-output-token cost
        latency = self.latency_median_ms * math.exp(self.latency_sigma * z) + completion_tokens * 0.5
        if u_stall < self.stall_rate:
            # a stall of 1x-2x stall_ms (reuse u_stall to stay deterministic)
            latency += self.stall_ms * (1 + u_stall / max(self.stall_rate, 1e-9))
        return latency

    def _maybe_fail(self, u_fail: float, u_rl: float) -> None:
        if self.in_outage():
            self._sleep(30)
            raise ProviderUnavailable("simulated outage (HTTP 503)", self.name)
        if u_rl < self.rate_limit_rate:
            self._sleep(15)
            raise RateLimitError("simulated 429 Too Many Requests", self.name, retry_after=0.05)
        if u_fail < self.failure_rate:
            self._sleep(50)
            raise ProviderUnavailable("simulated HTTP 500", self.name)

    # ----- primitives --------------------------------------------------------
    def _complete(self, request: ChatRequest) -> tuple[str, int, int]:
        u_fail, u_rl, z, u_stall = self._draw()
        self._maybe_fail(u_fail, u_rl)
        c_tok = self._completion_tokens(request)
        latency = self._latency_ms(z, c_tok, u_stall)
        if latency / 1000 > self.timeout_s:
            self._sleep(self.timeout_s * 1000)
            raise ProviderTimeout(f"no response within {self.timeout_s}s", self.name)
        self._sleep(latency)
        text = self._deterministic_text(request.prompt_text, self.model, c_tok)
        return text, estimate_tokens(request.all_text), c_tok

    def _stream(self, request: ChatRequest) -> Iterator[str]:
        u_fail, u_rl, z, u_stall = self._draw()
        self._maybe_fail(u_fail, u_rl)  # failures before the first token
        c_tok = self._completion_tokens(request)
        latency = self._latency_ms(z, c_tok, u_stall)
        text = self._deterministic_text(request.prompt_text, self.model, c_tok)
        words = text.split(" ")
        chunks = [" ".join(words[i:i + 4]) + (" " if i + 4 < len(words) else "") for i in range(0, len(words), 4)]
        ttft = latency * 0.6
        per_chunk = (latency - ttft) / max(1, len(chunks))
        self._sleep(ttft)
        for i, chunk in enumerate(chunks):
            u_drop = self._draw()[0]
            if i > 0 and u_drop < self.mid_stream_failure_rate / max(1, len(chunks)):
                raise ProviderUnavailable("connection dropped mid-stream", self.name)
            yield chunk
            self._sleep(per_chunk)
