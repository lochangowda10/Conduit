import threading
import time

import pytest

from conduit.errors import ProviderBadRequest, ProviderUnavailable, RateLimitError
from conduit.ledger import CostLedger
from conduit.ratelimit import TokenBucket
from conduit.resilience import BreakerState, CircuitBreaker, backoff_delay, retry
from conduit.types import Usage


class TestRetry:
    def test_retries_transient_then_succeeds(self):
        calls = []

        @retry(max_attempts=3, base_delay=0.001)
        def f():
            calls.append(1)
            if len(calls) < 3:
                raise ProviderUnavailable("503")
            return "ok"

        assert f() == "ok" and len(calls) == 3

    def test_gives_up_after_max_attempts(self):
        @retry(max_attempts=2, base_delay=0.001)
        def f():
            raise RateLimitError("429")

        with pytest.raises(RateLimitError):
            f()

    def test_does_not_retry_bad_request(self):
        calls = []

        @retry(max_attempts=5, base_delay=0.001)
        def f():
            calls.append(1)
            raise ProviderBadRequest("400")

        with pytest.raises(ProviderBadRequest):
            f()
        assert len(calls) == 1

    def test_wraps_preserves_name(self):
        @retry()
        def my_function():
            """doc"""

        assert my_function.__name__ == "my_function" and my_function.__doc__ == "doc"

    def test_backoff_bounds(self):
        for attempt in range(1, 8):
            assert 0 <= backoff_delay(attempt, 0.1, 1.0) <= min(1.0, 0.1 * 2 ** (attempt - 1))


class TestCircuitBreaker:
    def test_opens_after_threshold_and_recovers(self, clock):
        b = CircuitBreaker("p", failure_threshold=3, recovery_timeout=10, clock=clock)
        for _ in range(3):
            assert b.allow_request()
            b.record_failure()
        assert b.state is BreakerState.OPEN and not b.allow_request()
        clock.advance(10)
        assert b.state is BreakerState.HALF_OPEN
        assert b.allow_request()        # the single probe
        assert not b.allow_request()    # no second probe while the first is in flight
        b.record_success()
        assert b.state is BreakerState.CLOSED and b.times_opened == 1

    def test_failed_probe_reopens(self, clock):
        b = CircuitBreaker("p", failure_threshold=1, recovery_timeout=5, clock=clock)
        b.record_failure()
        clock.advance(5)
        assert b.allow_request()
        b.record_failure()
        assert b.state is BreakerState.OPEN

    def test_success_resets_count(self, clock):
        b = CircuitBreaker("p", failure_threshold=3, clock=clock)
        b.record_failure()
        b.record_failure()
        b.record_success()
        b.record_failure()
        assert b.state is BreakerState.CLOSED


class TestTokenBucket:
    def test_burst_then_refill(self, clock):
        bucket = TokenBucket(rate=2, capacity=3, clock=clock)
        assert [bucket.try_acquire() for _ in range(4)] == [True, True, True, False]
        clock.advance(0.5)              # +1 token
        assert bucket.try_acquire() and not bucket.try_acquire()
        clock.advance(100)
        assert bucket.available == 3    # capped at capacity

    def test_blocking_acquire_waits(self):
        bucket = TokenBucket(rate=50, capacity=1)
        bucket.acquire()
        t0 = time.perf_counter()
        assert bucket.acquire(timeout=1)
        assert 0.01 < time.perf_counter() - t0 < 0.2

    def test_acquire_timeout(self):
        bucket = TokenBucket(rate=0.1, capacity=1)
        bucket.acquire()
        assert bucket.acquire(timeout=0.05) is False

    def test_invalid(self):
        with pytest.raises(ValueError):
            TokenBucket(rate=0, capacity=1)


class TestLedger:
    def test_concurrent_adds_are_not_lost(self):
        ledger = CostLedger()
        u = Usage(1, 1, 0.001)

        def work():
            for _ in range(1000):
                ledger.add("p", u)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(ledger) == 8000 and ledger.total.prompt_tokens == 8000

    def test_unsafe_add_loses_updates(self):
        ledger = CostLedger()
        u = Usage(1, 0, 0)
        threads = [threading.Thread(target=lambda: [ledger.unsafe_add("p", u) for _ in range(500)]) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert ledger._by_provider["p"].prompt_tokens < 4000  # the race really happens

    def test_budget(self):
        ledger = CostLedger(budget_usd=0.01)
        ledger.add("p", Usage(1, 1, 0.006))
        assert not ledger.over_budget and ledger.remaining_budget == pytest.approx(0.004)
        ledger.add("p", Usage(1, 1, 0.005), wasted=True)  # hedge waste counts toward the budget
        assert ledger.over_budget
