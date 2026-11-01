"""Thread-safe cost ledger.

Syllabus link
-------------
* Experiment 8 (Race conditions + Lock): ``add`` is protected by a
  ``threading.Lock``; ``unsafe_add`` is the *same* code without the lock and
  is used by ``demos/race_condition_demo.py`` to show lost updates.
* Experiment 5 (Magic methods): totals are built with ``Usage.__add__``.
* Experiment 2 (Properties): ``budget_usd`` validates; ``remaining_budget``
  and ``over_budget`` are computed properties.

Why a read-modify-write is a race even with the GIL
---------------------------------------------------
``total = total + usage`` is several bytecodes: LOAD total, LOAD usage,
BINARY_ADD, STORE total. The GIL can switch threads between any two
bytecodes, so two threads can both LOAD the old total and the second STORE
overwrites the first: one update is lost. The GIL protects the interpreter's
internals, not your program's invariants.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from .types import Usage


class CostLedger:
    def __init__(self, budget_usd: float | None = None):
        self._lock = threading.Lock()
        self._by_provider: dict[str, Usage] = defaultdict(Usage)
        self._waste_by_provider: dict[str, Usage] = defaultdict(Usage)
        self._requests = 0
        self.budget_usd = budget_usd

    # ----- validated property -------------------------------------------------
    @property
    def budget_usd(self) -> float | None:
        return self._budget

    @budget_usd.setter
    def budget_usd(self, value: float | None) -> None:
        if value is not None and value < 0:
            raise ValueError("budget cannot be negative")
        self._budget = value

    # ----- recording ------------------------------------------------------------
    def add(self, provider: str, usage: Usage, wasted: bool = False) -> None:
        """Record usage. ``wasted=True`` marks the losing call of a hedged pair."""
        with self._lock:
            target = self._waste_by_provider if wasted else self._by_provider
            target[provider] = target[provider] + usage
            if not wasted:
                self._requests += 1

    def unsafe_add(self, provider: str, usage: Usage, widen_race_window: bool = True) -> None:
        """Deliberately broken version for the race-condition demo. DO NOT USE."""
        current = self._by_provider[provider]          # 1. read
        if widen_race_window:
            time.sleep(0)                              # yield the GIL -> another thread runs here
        self._by_provider[provider] = current + usage  # 2. write (may overwrite another thread's write)
        self._requests += 1

    # ----- reading ---------------------------------------------------------------
    @property
    def total(self) -> Usage:
        with self._lock:
            return sum(self._by_provider.values(), Usage())

    @property
    def wasted(self) -> Usage:
        with self._lock:
            return sum(self._waste_by_provider.values(), Usage())

    @property
    def spent_usd(self) -> float:
        return self.total.cost_usd + self.wasted.cost_usd

    @property
    def remaining_budget(self) -> float | None:
        return None if self._budget is None else max(0.0, self._budget - self.spent_usd)

    @property
    def over_budget(self) -> bool:
        return self._budget is not None and self.spent_usd >= self._budget

    def snapshot(self) -> dict:
        with self._lock:
            by_provider = {k: v.to_dict() for k, v in self._by_provider.items()}
            waste = {k: v.to_dict() for k, v in self._waste_by_provider.items()}
            requests = self._requests
        return {
            "requests": requests,
            "total": self.total.to_dict(),
            "hedge_waste": self.wasted.to_dict(),
            "by_provider": by_provider,
            "waste_by_provider": waste,
            "budget_usd": self._budget,
            "remaining_budget_usd": self.remaining_budget,
        }

    def __len__(self) -> int:
        return self._requests

    def __repr__(self) -> str:
        return f"CostLedger(requests={self._requests}, spent=${self.spent_usd:.6f})"
