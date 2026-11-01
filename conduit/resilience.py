"""Resilience primitives: decorators and the circuit breaker.

Syllabus link
-------------
* Experiment 6 (Decorators): ``log_execution`` and ``retry`` are decorators;
  ``retry`` is a *decorator factory* (a function that returns a decorator)
  because it takes arguments.
* Experiment 8 (Multithreading): ``CircuitBreaker`` is shared by every worker
  thread, so its state transitions are protected by a ``threading.Lock``.
* Experiment 2 (Properties): ``CircuitBreaker.state`` is a read-only property
  that also performs the lazy OPEN -> HALF_OPEN transition.
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from enum import Enum
from typing import Callable, TypeVar

from .errors import CircuitOpenError, RateLimitError, RetryableError

logger = logging.getLogger("conduit")

F = TypeVar("F", bound=Callable)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def log_execution(func: F) -> F:
    """Log entry, exit and duration of the wrapped function.

    ``functools.wraps`` copies ``__name__``/``__doc__`` so the decorated
    function still looks like the original (important for debugging,
    introspection and documentation tools).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        logger.debug("-> %s", func.__qualname__)
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.debug("x  %s raised %s", func.__qualname__, type(exc).__name__)
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug("<- %s (%.2f ms)", func.__qualname__, elapsed)

    return wrapper  # type: ignore[return-value]


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random | None = None) -> float:
    """Exponential backoff with *full jitter* (AWS Architecture Blog, 2015).

    attempt 1 -> U(0, base), attempt 2 -> U(0, 2*base), attempt 3 -> U(0, 4*base) ...
    Jitter spreads retries out so that many clients failing at the same moment
    do not all retry at the same moment ("thundering herd").
    """
    rng = rng or random
    return rng.uniform(0, min(cap, base * (2 ** (attempt - 1))))


def retry(
    max_attempts: int = 3,
    base_delay: float = 0.05,
    max_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (RetryableError,),
    on_retry: Callable[[int, BaseException], None] | None = None,
):
    """Decorator factory: retry the wrapped call on transient errors.

    Only exceptions listed in ``retry_on`` are retried; anything else (e.g. a
    400 Bad Request) propagates immediately, because retrying a request that
    is wrong will never make it right.

    If a ``RateLimitError`` carries a ``retry_after`` hint we honour it
    instead of guessing.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:  # type: ignore[misc]
                    if attempt == max_attempts:
                        raise
                    if on_retry:
                        on_retry(attempt, exc)
                    delay = backoff_delay(attempt, base_delay, max_delay)
                    if isinstance(exc, RateLimitError) and exc.retry_after:
                        delay = min(max_delay, exc.retry_after)
                    time.sleep(delay)
            raise AssertionError("unreachable")  # pragma: no cover

        wrapper.max_attempts = max_attempts  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class BreakerState(str, Enum):
    CLOSED = "closed"        # normal: requests flow
    OPEN = "open"            # tripped: fail fast, don't even call the provider
    HALF_OPEN = "half_open"  # probing: let ONE trial request through


class CircuitBreaker:
    """Per-provider circuit breaker (pattern popularised by Michael Nygard's
    *Release It!* and Netflix Hystrix).

    Why: if a provider is down, every request to it waits for a timeout
    before falling back. A breaker notices consecutive failures and skips
    the provider entirely for ``recovery_timeout`` seconds, turning a slow
    failure into a fast one. After the cool-down it lets one probe through
    (HALF_OPEN); success closes the circuit, failure re-opens it.

    ``clock`` is injectable so tests can move time forward without sleeping.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self.times_opened = 0

    # Read-only property with a side effect-free *view* of the lazy transition.
    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._current_state_locked()

    def _current_state_locked(self) -> BreakerState:
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self.recovery_timeout
        ):
            self._state = BreakerState.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def allow_request(self) -> bool:
        """Should we call this provider right now?"""
        with self._lock:
            state = self._current_state_locked()
            if state is BreakerState.CLOSED:
                return True
            if state is BreakerState.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True  # exactly one probe at a time
                return True
            return False

    def check(self) -> None:
        if not self.allow_request():
            raise CircuitOpenError("circuit open, skipping", self.name)

    def record_success(self) -> None:
        with self._lock:
            self._state = BreakerState.CLOSED
            self._consecutive_failures = 0
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            state = self._current_state_locked()
            if state is BreakerState.HALF_OPEN or self._consecutive_failures >= self.failure_threshold:
                if self._state is not BreakerState.OPEN:
                    self.times_opened += 1
                self._state = BreakerState.OPEN
                self._opened_at = self._clock()
                self._probe_in_flight = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self._current_state_locked().value,
                "consecutive_failures": self._consecutive_failures,
                "times_opened": self.times_opened,
            }

    def __repr__(self) -> str:
        return f"CircuitBreaker({self.name!r}, state={self.state.value})"
