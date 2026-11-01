"""Gateway configuration.

Syllabus link
-------------
* Experiment 2 (Encapsulation + properties): every numeric setting is a
  private attribute behind a ``@property`` whose setter validates the range,
  exactly like the Student-marks experiment validates 0-100. An invalid
  config fails loudly at start-up instead of misbehaving at 3 a.m.
* Experiment 1 (class/static methods): ``from_json`` and ``from_dict`` are
  alternate constructors (``@classmethod``); ``_require_range`` is a pure
  helper (``@staticmethod``).
* Experiment 7 (JSON + exception handling): missing file, invalid JSON and
  wrong types are each turned into a clear ``ConfigError``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import ConfigError


class GatewayConfig:
    def __init__(self, **kwargs: Any):
        # routing
        self.strategy: str = kwargs.pop("strategy", "smart")
        self.routing_weights: dict[str, float] = kwargs.pop("routing_weights", {})
        # cache
        self.cache_enabled: bool = kwargs.pop("cache_enabled", True)
        self.semantic_cache: bool = kwargs.pop("semantic_cache", True)
        self.similarity_threshold = kwargs.pop("similarity_threshold", 0.85)
        self.cache_ttl_s = kwargs.pop("cache_ttl_s", 3600.0)
        self.cache_capacity = kwargs.pop("cache_capacity", 10_000)
        self.cache_max_temperature = kwargs.pop("cache_max_temperature", 0.3)
        self.entity_guard: bool = kwargs.pop("entity_guard", True)
        self.idf_weighting: bool = kwargs.pop("idf_weighting", True)
        self.embedder: str = kwargs.pop("embedder", "hashing")
        # resilience
        self.max_attempts_per_provider = kwargs.pop("max_attempts_per_provider", 2)
        self.backoff_base_s = kwargs.pop("backoff_base_s", 0.05)
        self.max_providers_per_request = kwargs.pop("max_providers_per_request", 3)
        self.breaker_failure_threshold = kwargs.pop("breaker_failure_threshold", 5)
        self.breaker_recovery_s = kwargs.pop("breaker_recovery_s", 10.0)
        self.hedge_after_ms = kwargs.pop("hedge_after_ms", None)
        # admission control
        self.rate_limit_rps = kwargs.pop("rate_limit_rps", None)
        self.rate_limit_burst = kwargs.pop("rate_limit_burst", None)
        self.queue_timeout_s = kwargs.pop("queue_timeout_s", 5.0)
        self.budget_usd = kwargs.pop("budget_usd", None)
        # misc
        self.log_path: str | None = kwargs.pop("log_path", None)
        self.providers: list[dict[str, Any]] = kwargs.pop("providers", [])
        if kwargs:
            raise ConfigError(f"unknown config keys: {sorted(kwargs)}")

    # ----- helpers ---------------------------------------------------------------
    @staticmethod
    def _require_range(name: str, value: Any, lo: float | None = None, hi: float | None = None,
                       allow_none: bool = False) -> Any:
        if value is None and allow_none:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name} must be a number, got {type(value).__name__}")
        if lo is not None and value < lo:
            raise ConfigError(f"{name} must be >= {lo}, got {value}")
        if hi is not None and value > hi:
            raise ConfigError(f"{name} must be <= {hi}, got {value}")
        return value

    # ----- validated properties ------------------------------------------------
    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, v: float) -> None:
        self._similarity_threshold = float(self._require_range("similarity_threshold", v, 0.01, 1.0))

    @property
    def cache_ttl_s(self) -> float:
        return self._cache_ttl_s

    @cache_ttl_s.setter
    def cache_ttl_s(self, v: float) -> None:
        self._cache_ttl_s = float(self._require_range("cache_ttl_s", v, 0))

    @property
    def cache_capacity(self) -> int:
        return self._cache_capacity

    @cache_capacity.setter
    def cache_capacity(self, v: int) -> None:
        self._cache_capacity = int(self._require_range("cache_capacity", v, 1, 5_000_000))

    @property
    def cache_max_temperature(self) -> float:
        return self._cache_max_temperature

    @cache_max_temperature.setter
    def cache_max_temperature(self, v: float) -> None:
        self._cache_max_temperature = float(self._require_range("cache_max_temperature", v, 0, 2))

    @property
    def max_attempts_per_provider(self) -> int:
        return self._max_attempts

    @max_attempts_per_provider.setter
    def max_attempts_per_provider(self, v: int) -> None:
        self._max_attempts = int(self._require_range("max_attempts_per_provider", v, 1, 10))

    @property
    def backoff_base_s(self) -> float:
        return self._backoff_base_s

    @backoff_base_s.setter
    def backoff_base_s(self, v: float) -> None:
        self._backoff_base_s = float(self._require_range("backoff_base_s", v, 0, 10))

    @property
    def max_providers_per_request(self) -> int:
        return self._max_providers

    @max_providers_per_request.setter
    def max_providers_per_request(self, v: int) -> None:
        self._max_providers = int(self._require_range("max_providers_per_request", v, 1, 20))

    @property
    def breaker_failure_threshold(self) -> int:
        return self._breaker_threshold

    @breaker_failure_threshold.setter
    def breaker_failure_threshold(self, v: int) -> None:
        self._breaker_threshold = int(self._require_range("breaker_failure_threshold", v, 1, 1000))

    @property
    def breaker_recovery_s(self) -> float:
        return self._breaker_recovery_s

    @breaker_recovery_s.setter
    def breaker_recovery_s(self, v: float) -> None:
        self._breaker_recovery_s = float(self._require_range("breaker_recovery_s", v, 0))

    @property
    def hedge_after_ms(self) -> float | None:
        return self._hedge_after_ms

    @hedge_after_ms.setter
    def hedge_after_ms(self, v: float | None) -> None:
        self._hedge_after_ms = self._require_range("hedge_after_ms", v, 1, allow_none=True)

    @property
    def rate_limit_rps(self) -> float | None:
        return self._rate_limit_rps

    @rate_limit_rps.setter
    def rate_limit_rps(self, v: float | None) -> None:
        self._rate_limit_rps = self._require_range("rate_limit_rps", v, 0.001, allow_none=True)

    @property
    def rate_limit_burst(self) -> float | None:
        return self._rate_limit_burst

    @rate_limit_burst.setter
    def rate_limit_burst(self, v: float | None) -> None:
        self._rate_limit_burst = self._require_range("rate_limit_burst", v, 1, allow_none=True)

    @property
    def queue_timeout_s(self) -> float:
        return self._queue_timeout_s

    @queue_timeout_s.setter
    def queue_timeout_s(self, v: float) -> None:
        self._queue_timeout_s = float(self._require_range("queue_timeout_s", v, 0))

    @property
    def budget_usd(self) -> float | None:
        return self._budget_usd

    @budget_usd.setter
    def budget_usd(self, v: float | None) -> None:
        self._budget_usd = self._require_range("budget_usd", v, 0, allow_none=True)

    # ----- alternate constructors ----------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GatewayConfig":
        if not isinstance(data, dict):
            raise ConfigError("config root must be a JSON object")
        flat: dict[str, Any] = {}
        # Accept either flat keys or the nested sections used in config.example.json.
        for key, value in data.items():
            if key in ("routing", "cache", "resilience", "limits") and isinstance(value, dict):
                flat.update(value)
            elif key.startswith("_"):
                continue  # "_comment" keys are allowed for documentation
            else:
                flat[key] = value
        return cls(**flat)

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "GatewayConfig":
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ConfigError(f"config file not found: {p}") from None
        except OSError as exc:
            raise ConfigError(f"cannot read config {p}: {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {p} at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "routing_weights": self.routing_weights,
            "cache_enabled": self.cache_enabled,
            "semantic_cache": self.semantic_cache,
            "similarity_threshold": self.similarity_threshold,
            "cache_ttl_s": self.cache_ttl_s,
            "cache_capacity": self.cache_capacity,
            "cache_max_temperature": self.cache_max_temperature,
            "entity_guard": self.entity_guard,
            "idf_weighting": self.idf_weighting,
            "embedder": self.embedder,
            "max_attempts_per_provider": self.max_attempts_per_provider,
            "backoff_base_s": self.backoff_base_s,
            "max_providers_per_request": self.max_providers_per_request,
            "breaker_failure_threshold": self.breaker_failure_threshold,
            "breaker_recovery_s": self.breaker_recovery_s,
            "hedge_after_ms": self.hedge_after_ms,
            "rate_limit_rps": self.rate_limit_rps,
            "rate_limit_burst": self.rate_limit_burst,
            "queue_timeout_s": self.queue_timeout_s,
            "budget_usd": self.budget_usd,
            "log_path": self.log_path,
            "providers": self.providers,
        }

    def __repr__(self) -> str:
        return f"GatewayConfig(strategy={self.strategy!r}, providers={len(self.providers)})"
