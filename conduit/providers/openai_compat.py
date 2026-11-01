"""Real providers over HTTP, using only the standard library.

Most LLM vendors now expose an *OpenAI-compatible* endpoint
(``POST {base_url}/chat/completions``): OpenAI, Groq, Google Gemini,
OpenRouter, Together, Mistral, DeepSeek, and local servers like Ollama,
LM Studio and vLLM. So one class covers all of them - only ``base_url``,
``model`` and the API-key environment variable change.

Syllabus link
-------------
* Experiment 3 (Inheritance + super()): ``LocalProvider`` extends
  ``OpenAICompatibleProvider``, calls ``super().__init__`` with a default
  base URL, and overrides ``calculate_cost`` (local inference has no per-token bill).
* Experiment 6 (Generators): ``_stream`` parses Server-Sent Events line by
  line and ``yield``s each text delta as soon as it arrives.
* Experiment 7 (JSON + exception handling): every network/JSON failure is
  translated into one of Conduit's typed errors.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any, Iterator

from ..errors import (
    ConfigError,
    ProviderBadRequest,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitError,
)
from ..types import ChatRequest, estimate_tokens
from .base import LLMProvider, LoggableMixin, MetricsMixin


class OpenAICompatibleProvider(LoggableMixin, MetricsMixin, LLMProvider, kind="openai_compatible"):
    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key_env: str | None = None,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, model=model, **kwargs)
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"base_url for {name!r} must start with http:// or https://")
        self.base_url = base_url.rstrip("/")
        self.api_key = None
        if api_key_env:
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key:
                raise ConfigError(f"provider {name!r} needs the environment variable {api_key_env}")
        self.extra_headers = extra_headers or {}

    # ----- HTTP plumbing -----------------------------------------------------
    def _build_request(self, request: ChatRequest, stream: bool) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": [m.to_dict() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )

    def _translate_http_error(self, exc: urllib.error.HTTPError) -> ProviderError:
        try:
            detail = exc.read().decode(errors="replace")[:300]
        except Exception:  # noqa: BLE001 - best effort only
            detail = ""
        if exc.code == 429:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                ra = float(retry_after) if retry_after else None
            except ValueError:
                ra = None
            return RateLimitError(f"HTTP 429 {detail}", self.name, retry_after=ra)
        if exc.code >= 500:
            return ProviderUnavailable(f"HTTP {exc.code} {detail}", self.name)
        return ProviderBadRequest(f"HTTP {exc.code} {detail}", self.name)

    def _open(self, req: urllib.request.Request):
        try:
            return urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            raise self._translate_http_error(exc) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise ProviderTimeout(f"timed out after {self.timeout_s}s", self.name) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise ProviderTimeout(f"timed out after {self.timeout_s}s", self.name) from exc
            raise ProviderUnavailable(f"connection failed: {exc.reason}", self.name) from exc

    # ----- primitives --------------------------------------------------------
    def _complete(self, request: ChatRequest) -> tuple[str, int, int]:
        with self._open(self._build_request(request, stream=False)) as resp:
            raw = resp.read()
        try:
            data = json.loads(raw)
            text = data["choices"][0]["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailable(f"unparseable response: {raw[:200]!r}", self.name) from exc
        usage = data.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or estimate_tokens(request.all_text))
        c_tok = int(usage.get("completion_tokens") or estimate_tokens(text))
        return text, p_tok, c_tok

    def _stream(self, request: ChatRequest) -> Iterator[str]:
        resp = self._open(self._build_request(request, stream=True))  # errors raise before first yield
        with resp:
            for raw_line in resp:  # an HTTP response is an iterator over lines
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue  # SSE comments / keep-alives / blank separators
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta


class LocalProvider(OpenAICompatibleProvider, kind="local"):
    """A model served on your own machine (Ollama, LM Studio, llama.cpp, vLLM).

    Inherits all the HTTP logic; only the defaults and the pricing differ.
    """

    def __init__(self, name: str, model: str, base_url: str = "http://localhost:11434/v1", **kwargs: Any):
        kwargs.setdefault("tier", "small")
        kwargs.setdefault("timeout_s", 120.0)  # CPU inference can be slow
        super().__init__(name=name, model=model, base_url=base_url, **kwargs)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # No per-token bill. (A fuller model could charge electricity/GPU-hours.)
        return 0.0
