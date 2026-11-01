"""Routing: decide which providers to try, and in what order.

Syllabus link
-------------
* Experiment 6 (Iterators): ``FallbackChain`` is a custom iterator class
  (``__iter__`` + ``__next__``) that lazily yields the next *eligible*
  provider, skipping ones whose circuit breaker is open.
* Experiment 4 (Abstract classes): ``RoutingStrategy`` is an ABC; the
  strategies are interchangeable (the *Strategy* pattern).
* Experiment 3 (Polymorphism): the gateway calls ``strategy.rank(...)``
  without caring which strategy it holds.

The idea behind complexity routing
----------------------------------
Most production traffic is easy ("summarise this", "what's the capital of
X"). Sending everything to the biggest model wastes money; sending
everything to the smallest hurts quality on hard prompts. Research such as
RouteLLM (LMSYS, 2024) shows a router can cut cost substantially while
keeping quality. We use a transparent heuristic score instead of a trained
classifier: it adds microseconds, needs no training data, and every
decision can be explained. Logged outcomes could later train a real
classifier - that is the natural "v2".
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from .errors import ConfigError
from .providers.base import LLMProvider
from .resilience import CircuitBreaker
from .types import ChatRequest, estimate_tokens

TIER_RANK = {"small": 0, "medium": 1, "large": 2}

_CODE_RE = re.compile(r"```|\bdef |\bclass |\bfunction\b|\bSELECT\b|\breturn\b|=>|\{.*\}", re.I)
_REASONING_WORDS = (
    "prove", "derive", "design", "architect", "trade-off", "tradeoff", "optimi", "step by step",
    "analy", "debug", "compare", "evaluate", "critique", "why does", "explain in detail",
    "algorithm", "complexity", "distributed", "concurren", "scal",
)


def complexity_score(prompt: str) -> float:
    """A 0..1 estimate of how hard a prompt is. Transparent on purpose."""
    text = prompt.lower()
    tokens = estimate_tokens(prompt)
    score = min(tokens / 400, 0.4)                                            # length: up to 0.4
    score += 0.25 if _CODE_RE.search(prompt) else 0.0                        # code present
    hits = sum(1 for w in _REASONING_WORDS if w in text)
    score += min(hits * 0.12, 0.48)                                          # reasoning verbs
    score += min(max(prompt.count("?") - 1, 0) * 0.05, 0.1)                  # multi-part question
    return round(min(score, 1.0), 3)


def tier_for(prompt: str) -> str:
    s = complexity_score(prompt)
    if s < 0.3:
        return "small"
    if s < 0.55:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
class RoutingStrategy(ABC):
    name = "abstract"

    @abstractmethod
    def rank(self, request: ChatRequest, providers: list[LLMProvider]) -> list[LLMProvider]:
        """Return providers in the order they should be tried."""

    def explain(self, request: ChatRequest) -> dict:
        return {"strategy": self.name}

    @staticmethod
    def pinned(request: ChatRequest, providers: list[LLMProvider]) -> list[LLMProvider] | None:
        """If the caller asked for a concrete provider name, honour it first."""
        match = [p for p in providers if p.name == request.model]
        if match:
            return match + [p for p in providers if p.name != request.model]
        return None


class SingleProvider(RoutingStrategy):
    """Baseline: always the first configured provider, no fallback."""

    name = "single"

    def rank(self, request, providers):
        return providers[:1]


class PriorityOrder(RoutingStrategy):
    """Try providers in the order they are configured (classic primary/backup)."""

    name = "priority"

    def rank(self, request, providers):
        return self.pinned(request, providers) or list(providers)


class CheapestFirst(RoutingStrategy):
    name = "cheapest"

    def rank(self, request, providers):
        return self.pinned(request, providers) or sorted(providers, key=lambda p: p.blended_price)


class FastestFirst(RoutingStrategy):
    name = "fastest"

    def rank(self, request, providers):
        def latency(p):
            ewma = getattr(p, "ewma_latency_ms", None)
            return ewma if ewma is not None else 0.0  # untried providers get explored first
        return self.pinned(request, providers) or sorted(providers, key=latency)


class SmartRouter(RoutingStrategy):
    """Complexity-aware, cost/latency/reliability-weighted routing.

    1. Estimate the tier the prompt needs (small/medium/large).
    2. *Capable* providers (tier >= needed) are ranked by a weighted score of
       normalised price, EWMA latency and observed error rate.
    3. Less capable providers are appended as a last resort (graceful
       degradation: a decent answer beats an error page).
    """

    name = "smart"

    def __init__(self, w_cost: float = 0.6, w_latency: float = 0.3, w_errors: float = 0.1):
        if min(w_cost, w_latency, w_errors) < 0:
            raise ConfigError("routing weights must be non-negative")
        total = w_cost + w_latency + w_errors
        if total == 0:
            raise ConfigError("at least one routing weight must be positive")
        self.w_cost, self.w_latency, self.w_errors = w_cost / total, w_latency / total, w_errors / total

    def _scores(self, providers: list[LLMProvider]) -> dict[str, float]:
        prices = [p.blended_price for p in providers]
        max_price = max(prices) or 1.0
        lats = [getattr(p, "ewma_latency_ms", None) for p in providers]
        known = [x for x in lats if x is not None]
        default_lat = sum(known) / len(known) if known else 1.0
        max_lat = max(known) if known else 1.0
        scores = {}
        for p, lat in zip(providers, lats):
            lat = default_lat if lat is None else lat
            err = getattr(p, "error_rate", 0.0)
            scores[p.name] = (
                self.w_cost * p.blended_price / max_price
                + self.w_latency * lat / max_lat
                + self.w_errors * err
            )
        return scores

    def rank(self, request, providers):
        pinned = self.pinned(request, providers)
        if pinned:
            return pinned
        needed = request.model if request.model in TIER_RANK else tier_for(request.prompt_text)
        scores = self._scores(providers)
        capable = [p for p in providers if TIER_RANK[p.tier] >= TIER_RANK[needed]]
        weaker = [p for p in providers if TIER_RANK[p.tier] < TIER_RANK[needed]]
        capable.sort(key=lambda p: scores[p.name])
        weaker.sort(key=lambda p: -TIER_RANK[p.tier])  # strongest of the weaker ones first
        return capable + weaker

    def explain(self, request):
        return {
            "strategy": self.name,
            "complexity": complexity_score(request.prompt_text),
            "tier": request.model if request.model in TIER_RANK else tier_for(request.prompt_text),
        }


STRATEGIES: dict[str, type[RoutingStrategy]] = {
    cls.name: cls for cls in (SingleProvider, PriorityOrder, CheapestFirst, FastestFirst, SmartRouter)
}


def make_strategy(name: str, **kwargs) -> RoutingStrategy:
    try:
        return STRATEGIES[name](**kwargs)
    except KeyError:
        raise ConfigError(f"unknown routing strategy {name!r}; choose from {sorted(STRATEGIES)}") from None


# ---------------------------------------------------------------------------
# The custom iterator
# ---------------------------------------------------------------------------
class FallbackChain:
    """Iterate over ranked providers, lazily skipping open circuits.

    Lazy matters: the breaker state is checked at the moment we are about to
    use a provider, not when the list was built, so a provider that recovers
    (or trips) mid-request is handled correctly.

    >>> chain = FallbackChain(ranked, breakers)
    >>> for provider in chain:      # __iter__ then repeated __next__
    ...     try: return provider.complete(req)
    ...     except ProviderError: continue
    """

    def __init__(self, ranked: Iterable[LLMProvider], breakers: dict[str, CircuitBreaker], max_providers: int | None = None):
        self._providers = list(ranked)
        self._breakers = breakers
        self._index = 0
        self._yielded = 0
        self._max = max_providers if max_providers is not None else len(self._providers)
        self.skipped: list[str] = []

    def __iter__(self) -> Iterator[LLMProvider]:
        return self

    def __next__(self) -> LLMProvider:
        while self._index < len(self._providers) and self._yielded < self._max:
            provider = self._providers[self._index]
            self._index += 1
            breaker = self._breakers.get(provider.name)
            if breaker is not None and not breaker.allow_request():
                self.skipped.append(provider.name)
                continue
            self._yielded += 1
            return provider
        raise StopIteration

    def next_if(self, predicate) -> LLMProvider | None:
        """Take the first remaining eligible provider that satisfies ``predicate``.

        Used for hedging: the backup must be at least as capable as the
        primary, otherwise a fast-but-weaker model would "win" the race and
        quietly lower answer quality.
        """
        if self._yielded >= self._max:
            return None
        for pos in range(self._index, len(self._providers)):
            provider = self._providers[pos]
            if not predicate(provider):
                continue
            breaker = self._breakers.get(provider.name)
            if breaker is not None and not breaker.allow_request():
                continue
            del self._providers[pos]  # consumed: it must not be yielded again later
            self._yielded += 1
            return provider
        return None

    def peek_remaining(self) -> list[str]:
        return [p.name for p in self._providers[self._index:]]

    def __len__(self) -> int:
        return len(self._providers)
