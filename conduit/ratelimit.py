"""Thread-safe token-bucket rate limiter.

Syllabus link
-------------
* Experiment 8 (Multithreading): a ``threading.Condition`` (a lock plus a
  wait/notify queue) lets many request threads share one bucket safely.
* Experiment 2 (Properties): ``rate`` and ``capacity`` validate on assignment.

How a token bucket works
------------------------
A bucket holds up to ``capacity`` tokens and refills continuously at ``rate``
tokens per second. Each request takes one token. If the bucket is empty the
request waits (or is rejected). This allows short *bursts* up to
``capacity`` while enforcing the *average* rate - which is exactly how most
LLM providers define their limits (RPM/TPM).
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class TokenBucket:
    def __init__(self, rate: float, capacity: float, clock: Callable[[], float] = time.monotonic):
        self._lock = threading.Condition()
        self._clock = clock
        self.rate = rate          # goes through the property setter
        self.capacity = capacity  # goes through the property setter
        self._tokens = float(capacity)
        self._last = clock()

    # ----- validated properties -------------------------------------------
    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        if value <= 0:
            raise ValueError("rate must be > 0 tokens/second")
        self._rate = float(value)

    @property
    def capacity(self) -> float:
        return self._capacity

    @capacity.setter
    def capacity(self, value: float) -> None:
        if value < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = float(value)

    @property
    def available(self) -> float:
        with self._lock:
            self._refill_locked()
            return self._tokens

    # ----- core logic ------------------------------------------------------
    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take ``tokens`` if available right now; never blocks."""
        with self._lock:
            self._refill_locked()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> bool:
        """Block until ``tokens`` are available or ``timeout`` expires.

        Returns True if acquired, False on timeout. The wait time is computed
        exactly (deficit / rate) instead of busy-polling.
        """
        if tokens > self._capacity:
            raise ValueError("cannot request more tokens than the bucket capacity")
        deadline = None if timeout is None else self._clock() + timeout
        with self._lock:
            while True:
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                wait = (tokens - self._tokens) / self._rate
                if deadline is not None:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        return False
                    wait = min(wait, remaining)
                self._lock.wait(wait)  # releases the lock while sleeping

    def __repr__(self) -> str:
        return f"TokenBucket(rate={self._rate}/s, capacity={self._capacity:g})"
