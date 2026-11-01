"""Provider abstraction.

Syllabus link
-------------
* Experiment 4 (Abstract classes + multiple inheritance + MRO):
  ``LLMProvider`` is an ABC. Concrete providers inherit from two *mixins*
  (``LoggableMixin``, ``MetricsMixin``) plus the ABC. Every ``__init__``
  calls ``super().__init__(**kwargs)`` - "cooperative multiple inheritance" -
  so each class in the MRO initialises exactly once. Print
  ``SimulatedProvider.__mro__`` to see the order.
* Experiment 3 (Inheritance + polymorphism + super()): the gateway calls
  ``provider.complete()`` and ``provider.calculate_cost()`` without knowing
  which subclass it has. ``LocalProvider`` overrides ``calculate_cost``.
* Experiment 1 (instance / class / static methods):
  - instance:  ``complete()``                      (needs one provider's state)
  - class:     ``from_config()`` alternate constructor, ``registry`` lookup
  - static:    ``validate_name()``                 (pure check, needs no state)
  ``__init_subclass__`` auto-registers each subclass under a ``kind`` name,
  so adding a provider type needs zero changes to the factory.
* Experiment 6 (Generators): ``stream()`` is a generator that yields text
  chunks as they arrive.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterator

from ..errors import ConfigError
from ..types import ChatRequest, ChatResponse, Usage, estimate_tokens


# ---------------------------------------------------------------------------
# Mixins: small, focused bits of behaviour that can be "mixed into" any class.
# ---------------------------------------------------------------------------
class LoggableMixin:
    """Gives a class a named logger and a ``log()`` helper."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)  # cooperative: pass along the MRO
        self._logger = logging.getLogger(f"conduit.{type(self).__name__}")

    def log(self, level: int, msg: str, *args: Any) -> None:
        self._logger.log(level, f"[{getattr(self, 'name', '?')}] " + msg, *args)


class MetricsMixin:
    """Thread-safe call counters plus an exponentially weighted moving
    average (EWMA) of latency, which the router uses to prefer fast providers.

    EWMA: ``avg = alpha * new + (1 - alpha) * avg``. Recent samples matter
    more than old ones, it needs O(1) memory, and it reacts to a provider
    slowing down within a handful of requests.
    """

    EWMA_ALPHA: ClassVar[float] = 0.2

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._metrics_lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        self.ewma_latency_ms: float | None = None

    def record_call(self, latency_ms: float, ok: bool) -> None:
        with self._metrics_lock:
            self.calls += 1
            if not ok:
                self.failures += 1
                return
            if self.ewma_latency_ms is None:
                self.ewma_latency_ms = latency_ms
            else:
                a = self.EWMA_ALPHA
                self.ewma_latency_ms = a * latency_ms + (1 - a) * self.ewma_latency_ms

    @property
    def error_rate(self) -> float:
        with self._metrics_lock:
            return self.failures / self.calls if self.calls else 0.0

    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                "calls": self.calls,
                "failures": self.failures,
                "ewma_latency_ms": None if self.ewma_latency_ms is None else round(self.ewma_latency_ms, 1),
            }


# ---------------------------------------------------------------------------
# The abstract base class
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """Interface every provider implements.

    Subclasses implement two primitives - ``_complete`` and ``_stream`` -
    and inherit timing, cost calculation and response packaging for free
    (the *Template Method* pattern).
    """

    registry: ClassVar[dict[str, type["LLMProvider"]]] = {}
    kind: ClassVar[str] = "abstract"

    _NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-.]{0,63}$")
    VALID_TIERS = ("small", "medium", "large")

    def __init_subclass__(cls, kind: str | None = None, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if kind:
            cls.kind = kind
            LLMProvider.registry[kind] = cls

    def __init__(
        self,
        name: str,
        model: str,
        input_price_per_m: float = 0.0,
        output_price_per_m: float = 0.0,
        tier: str = "medium",
        timeout_s: float = 30.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not self.validate_name(name):
            raise ConfigError(f"invalid provider name {name!r} (lowercase letters, digits, - _ . only)")
        if tier not in self.VALID_TIERS:
            raise ConfigError(f"tier must be one of {self.VALID_TIERS}, got {tier!r}")
        if input_price_per_m < 0 or output_price_per_m < 0:
            raise ConfigError("prices cannot be negative")
        self.name = name
        self.model = model
        self.input_price_per_m = float(input_price_per_m)
        self.output_price_per_m = float(output_price_per_m)
        self.tier = tier
        self.timeout_s = float(timeout_s)

    # ----- static & class methods ------------------------------------------
    @staticmethod
    def validate_name(name: str) -> bool:
        """Pure validation - needs neither the instance nor the class."""
        return isinstance(name, str) and bool(LLMProvider._NAME_RE.match(name))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LLMProvider":
        """Alternate constructor / factory: ``{"type": "simulated", ...}`` -> instance.

        Called on the base class it dispatches through the registry; called
        on a subclass it builds that subclass directly.
        """
        cfg = dict(cfg)
        kind = cfg.pop("type", None)
        target = cls
        if cls is LLMProvider or kind is not None:
            if kind not in LLMProvider.registry:
                known = ", ".join(sorted(LLMProvider.registry))
                raise ConfigError(f"unknown provider type {kind!r}; known types: {known}")
            target = LLMProvider.registry[kind]
        try:
            return target(**cfg)
        except TypeError as exc:
            raise ConfigError(f"bad config for provider {cfg.get('name')!r}: {exc}") from exc

    # ----- polymorphic hooks --------------------------------------------------
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost of one call. Subclasses override for special pricing."""
        return (prompt_tokens * self.input_price_per_m + completion_tokens * self.output_price_per_m) / 1_000_000

    @property
    def blended_price(self) -> float:
        """Rough $/1M tokens assuming a 3:1 input:output ratio (used for routing)."""
        return (3 * self.input_price_per_m + self.output_price_per_m) / 4

    # ----- primitives each subclass must implement --------------------------
    @abstractmethod
    def _complete(self, request: ChatRequest) -> tuple[str, int, int]:
        """Return ``(text, prompt_tokens, completion_tokens)`` or raise a ProviderError."""

    @abstractmethod
    def _stream(self, request: ChatRequest) -> Iterator[str]:
        """Yield text chunks. Must raise before the first yield if the call fails to start."""

    # ----- template methods (shared behaviour) ------------------------------
    def complete(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        try:
            text, p_tok, c_tok = self._complete(request)
        except Exception:
            self._record((time.perf_counter() - start) * 1000, ok=False)
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        self._record(latency_ms, ok=True)
        return ChatResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage=Usage(p_tok, c_tok, self.calculate_cost(p_tok, c_tok)),
            latency_ms=latency_ms,
            request_id=request.request_id,
        )

    def stream(self, request: ChatRequest) -> Iterator[str]:
        """Generator wrapper that records metrics once the stream finishes."""
        start = time.perf_counter()
        ok = False
        try:
            yield from self._stream(request)
            ok = True
        finally:
            self._record((time.perf_counter() - start) * 1000, ok=ok)

    def stream_usage(self, request: ChatRequest, text: str) -> Usage:
        p_tok = estimate_tokens(request.all_text)
        c_tok = estimate_tokens(text)
        return Usage(p_tok, c_tok, self.calculate_cost(p_tok, c_tok))

    def _record(self, latency_ms: float, ok: bool) -> None:
        record = getattr(self, "record_call", None)  # present when MetricsMixin is mixed in
        if record:
            record(latency_ms, ok)

    def describe(self) -> dict[str, Any]:
        info = {
            "name": self.name,
            "type": self.kind,
            "model": self.model,
            "tier": self.tier,
            "input_price_per_m": self.input_price_per_m,
            "output_price_per_m": self.output_price_per_m,
        }
        if hasattr(self, "metrics"):
            info.update(self.metrics())
        return info

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r}, tier={self.tier!r})"
