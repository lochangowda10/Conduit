import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conduit.providers import SimulatedProvider  # noqa: E402


@pytest.fixture
def fast_provider():
    """Factory for quick simulated providers (latencies scaled down 100x)."""

    def make(name="sim", tier="small", **kw):
        kw.setdefault("latency_median_ms", 100)
        kw.setdefault("time_scale", 0.01)
        kw.setdefault("seed", 1)
        return SimulatedProvider(name, tier=tier, **kw)

    return make


class FakeClock:
    """Manually advanced clock for time-dependent logic (no sleeping in tests)."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()
