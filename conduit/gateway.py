"""The Gateway: ties cache, admission control, routing, resilience and telemetry together.

Request lifecycle (``Gateway.complete``)
---------------------------------------
1. **Cache**     exact hash lookup -> semantic (embedding) lookup. A hit returns
                 immediately at ~zero cost.
2. **Admission** gateway-wide token bucket (protects your provider quota) and
                 a hard spending cap.
3. **Route**     the strategy ranks providers; ``FallbackChain`` lazily yields
                 the next one whose circuit breaker allows traffic.
4. **Call**      each provider call is wrapped with ``@retry`` (exponential
                 backoff + jitter) and reports success/failure to its breaker.
                 Optionally *hedged*: if the first provider hasn't answered
                 after ``hedge_after_ms``, a second provider is raced against it.
5. **Record**    store in cache, add cost to the ledger, append a log record.

Syllabus link
-------------
* Experiment 8: ``ThreadPoolExecutor`` + ``concurrent.futures.wait`` for
  hedged requests; every shared component is lock-protected.
* Experiment 6: ``stream()`` is a generator; ``@retry`` is a decorator.
* Experiment 5: ``__enter__``/``__exit__`` make the gateway a context manager.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Iterator

from .cache import ResponseCache
from .config import GatewayConfig
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .errors import (
    AllProvidersFailed,
    BudgetExceeded,
    ConduitError,
    ConfigError,
    ProviderError,
    RateLimitedByGateway,
    RetryableError,
)
from .ledger import CostLedger
from .providers import LLMProvider
from .ratelimit import TokenBucket
from .resilience import CircuitBreaker, retry
from .router import TIER_RANK, FallbackChain, RoutingStrategy, make_strategy
from .telemetry import RequestLogger, RequestRecord
from .types import ChatRequest, ChatResponse, Usage, estimate_tokens

log = logging.getLogger("conduit.gateway")


class Gateway:
    def __init__(
        self,
        providers: list[LLMProvider],
        config: GatewayConfig | None = None,
        strategy: RoutingStrategy | None = None,
        logger: RequestLogger | None = None,
        scenario: str = "",
    ):
        if not providers:
            raise ConfigError("the gateway needs at least one provider")
        names = [p.name for p in providers]
        if len(names) != len(set(names)):
            raise ConfigError(f"provider names must be unique: {names}")
        self.config = config or GatewayConfig()
        self.providers = providers
        self.strategy = strategy or make_strategy(self.config.strategy, **self.config.routing_weights)
        self.logger = logger or RequestLogger(self.config.log_path)
        self.scenario = scenario
        self.ledger = CostLedger(self.config.budget_usd)
        self.breakers = {
            p.name: CircuitBreaker(p.name, self.config.breaker_failure_threshold, self.config.breaker_recovery_s)
            for p in providers
        }
        self.cache = self._build_cache() if self.config.cache_enabled else None
        self.bucket = (
            TokenBucket(self.config.rate_limit_rps, self.config.rate_limit_burst or max(1.0, self.config.rate_limit_rps))
            if self.config.rate_limit_rps
            else None
        )
        self._pool = ThreadPoolExecutor(max_workers=128, thread_name_prefix="conduit-call")
        self._started = time.time()

    # ------------------------------------------------------------------ setup
    def _build_cache(self) -> ResponseCache:
        cfg = self.config
        if cfg.embedder == "hashing":
            embedder = HashingEmbedder()
        elif cfg.embedder in ("sentence-transformer", "sentence_transformer"):
            embedder = SentenceTransformerEmbedder()
        else:
            raise ConfigError(f"unknown embedder {cfg.embedder!r}")
        return ResponseCache(
            embedder=embedder,
            capacity=cfg.cache_capacity,
            similarity_threshold=cfg.similarity_threshold,
            ttl_seconds=cfg.cache_ttl_s,
            max_temperature=cfg.cache_max_temperature,
            semantic_enabled=cfg.semantic_cache,
            entity_guard=cfg.entity_guard,
            idf_weighting=cfg.idf_weighting,
        )

    @classmethod
    def from_config(cls, config: GatewayConfig, **kwargs: Any) -> "Gateway":
        providers: list[LLMProvider] = []
        for raw in config.providers:
            raw = dict(raw)
            optional = raw.pop("optional", False)
            if not raw.pop("enabled", True):
                continue
            try:
                providers.append(LLMProvider.from_config(raw))
            except ConfigError as exc:
                if optional:
                    log.warning("skipping optional provider %s: %s", raw.get("name"), exc)
                    continue
                raise
        return cls(providers, config, **kwargs)

    @classmethod
    def from_json(cls, path: str, **kwargs: Any) -> "Gateway":
        return cls.from_config(GatewayConfig.from_json(path), **kwargs)

    # --------------------------------------------------------- context manager
    def __enter__(self) -> "Gateway":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------------- helpers
    def _new_record(self, request: ChatRequest) -> RequestRecord:
        info = self.strategy.explain(request)
        return RequestRecord(
            request_id=request.request_id,
            scenario=self.scenario,
            tier=info.get("tier"),
            complexity=info.get("complexity"),
            label=request.metadata.get("label"),
        )

    def _admit(self) -> None:
        if self.bucket and not self.bucket.acquire(timeout=self.config.queue_timeout_s):
            raise RateLimitedByGateway("gateway rate limit exceeded; try again later")
        if self.ledger.over_budget:
            raise BudgetExceeded(f"spending cap of ${self.config.budget_usd} reached")

    def _call_provider(self, provider: LLMProvider, request: ChatRequest, attempts: list[str]) -> ChatResponse:
        """One provider, with retries; every attempt reports to the breaker."""
        breaker = self.breakers[provider.name]

        @retry(max_attempts=self.config.max_attempts_per_provider, base_delay=self.config.backoff_base_s,
               retry_on=(RetryableError,))
        def attempt() -> ChatResponse:
            attempts.append(provider.name)  # list.append is atomic -> safe across threads
            try:
                resp = provider.complete(request)
            except RetryableError:
                breaker.record_failure()  # provider-health failure
                raise
            breaker.record_success()
            return resp

        return attempt()

    def _account_loser(self, fut: Future) -> None:
        """Callback for the hedge that lost the race: its tokens were still billed."""
        if fut.cancelled() or fut.exception() is not None:
            return
        resp: ChatResponse = fut.result()
        self.ledger.add(resp.provider, resp.usage, wasted=True)

    def _run_chain(self, request: ChatRequest, chain: FallbackChain, attempts: list[str]) -> tuple[ChatResponse, bool]:
        errors: list[Exception] = []
        hedge_s = self.config.hedge_after_ms / 1000 if self.config.hedge_after_ms else None
        hedged = False
        while True:
            primary = next(chain, None)
            if primary is None:
                raise AllProvidersFailed(errors)

            if hedge_s is None:  # ---- plain sequential fallback --------------
                try:
                    return self._call_provider(primary, request, attempts), False
                except ProviderError as exc:
                    errors.append(exc)
                    continue

            # ---- hedged: race a backup if the primary is slow ----------------
            futures: dict[Future, LLMProvider] = {
                self._pool.submit(self._call_provider, primary, request, attempts): primary
            }
            done, _ = wait(futures, timeout=hedge_s)
            if not done:
                # Race a backup that is at least as capable as the primary.
                backup = chain.next_if(lambda p: TIER_RANK[p.tier] >= TIER_RANK[primary.tier])
                if backup is not None:
                    futures[self._pool.submit(self._call_provider, backup, request, attempts)] = backup
                    hedged = True
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    exc = fut.exception()
                    if exc is not None:
                        errors.append(exc)
                        if not isinstance(exc, ProviderError):
                            raise exc
                        continue
                    for loser in pending:  # winner found: deal with the other runner
                        if not loser.cancel():
                            loser.add_done_callback(self._account_loser)
                    return fut.result(), hedged
            # both failed -> loop and try the next provider(s) in the chain

    # ----------------------------------------------------------- public API
    def complete(self, request: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        record = self._new_record(request)
        attempts: list[str] = []
        try:
            if self.cache is None or not self.cache.cacheable(request):
                record.cache = "bypass"
            else:
                hit = self.cache.get(request)
                if hit is not None:
                    entry, kind, sim = hit
                    latency = (time.perf_counter() - t0) * 1000
                    usage = Usage(estimate_tokens(request.all_text), estimate_tokens(entry.response_text), 0.0)
                    response = ChatResponse(
                        text=entry.response_text, provider=entry.provider, model=entry.model, usage=usage,
                        latency_ms=latency, cache=kind, attempts=0, request_id=request.request_id,
                    )
                    record.cache, record.similarity = kind, round(sim, 4)
                    record.cached_label = entry.meta.get("label")
                    self._finish(record, response)
                    return response

            self._admit()
            ranked = self.strategy.rank(request, self.providers)
            chain = FallbackChain(ranked, self.breakers, self.config.max_providers_per_request)
            response, hedged = self._run_chain(request, chain, attempts)
            response.latency_ms = (time.perf_counter() - t0) * 1000  # end-to-end, incl. retries
            response.attempts = len(attempts)
            response.hedged = hedged
            self.ledger.add(response.provider, response.usage)
            if self.cache is not None:
                self.cache.put(request, response, meta={"label": request.metadata.get("label")})
            self._finish(record, response)
            return response
        except ConduitError as exc:
            record.status = "error"
            record.error_type = type(exc).__name__
            record.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            record.attempts = len(attempts)
            self.logger.log(record)
            raise

    def _finish(self, record: RequestRecord, response: ChatResponse) -> None:
        record.provider = response.provider
        record.model = response.model
        record.latency_ms = round(response.latency_ms, 3)
        record.prompt_tokens = response.usage.prompt_tokens
        record.completion_tokens = response.usage.completion_tokens
        record.cost_usd = response.usage.cost_usd
        record.attempts = response.attempts
        record.hedged = response.hedged
        self.logger.log(record)

    def stream(self, request: ChatRequest) -> Iterator[str]:
        """Generator of text chunks with the same cache/route/fallback logic.

        Important limitation (and a classic interview question): we can only
        fall back to another provider *before the first chunk is sent*. Once
        bytes have reached the client, switching providers would splice two
        different answers together, so a mid-stream failure is surfaced as an
        error instead.
        """
        t0 = time.perf_counter()
        record = self._new_record(request)
        attempts: list[str] = []
        try:
            if self.cache is not None:
                hit = self.cache.get(request)
                if hit is not None:
                    entry, kind, sim = hit
                    words = entry.response_text.split(" ")
                    for i in range(0, len(words), 4):
                        yield " ".join(words[i:i + 4]) + (" " if i + 4 < len(words) else "")
                    usage = Usage(estimate_tokens(request.all_text), estimate_tokens(entry.response_text), 0.0)
                    record.cache, record.similarity = kind, round(sim, 4)
                    record.cached_label = entry.meta.get("label")
                    self._finish(record, ChatResponse(entry.response_text, entry.provider, entry.model, usage,
                                                      (time.perf_counter() - t0) * 1000, kind, 0,
                                                      request_id=request.request_id))
                    return

            self._admit()
            ranked = self.strategy.rank(request, self.providers)
            chain = FallbackChain(ranked, self.breakers, self.config.max_providers_per_request)
            errors: list[Exception] = []
            for provider in chain:
                attempts.append(provider.name)
                breaker = self.breakers[provider.name]
                gen = provider.stream(request)
                try:
                    first = next(gen)               # the call really starts here
                except StopIteration:
                    first = ""
                except ProviderError as exc:        # failed before any output -> safe to fall back
                    if isinstance(exc, RetryableError):
                        breaker.record_failure()
                    errors.append(exc)
                    continue
                breaker.record_success()
                parts = [first]
                if first:
                    yield first
                for chunk in gen:                   # mid-stream errors propagate to the caller
                    parts.append(chunk)
                    yield chunk
                text = "".join(parts)
                usage = provider.stream_usage(request, text)
                response = ChatResponse(text, provider.name, provider.model, usage,
                                        (time.perf_counter() - t0) * 1000, attempts=len(attempts),
                                        request_id=request.request_id)
                self.ledger.add(provider.name, usage)
                if self.cache is not None:
                    self.cache.put(request, response, meta={"label": request.metadata.get("label")})
                self._finish(record, response)
                return
            raise AllProvidersFailed(errors)
        except ConduitError as exc:
            record.status = "error"
            record.error_type = type(exc).__name__
            record.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            record.attempts = len(attempts)
            self.logger.log(record)
            raise

    # --------------------------------------------------------- observability
    def stats(self) -> dict[str, Any]:
        records = list(self.logger)
        ok = sum(r.status == "ok" for r in records)
        return {
            "uptime_s": round(time.time() - self._started, 1),
            "requests": {"total": len(records), "ok": ok, "error": len(records) - ok},
            "strategy": self.strategy.name,
            "ledger": self.ledger.snapshot(),
            "cache": self.cache.stats.to_dict() if self.cache else None,
            "breakers": {name: b.snapshot() for name, b in self.breakers.items()},
            "providers": [p.describe() for p in self.providers],
        }

    def __repr__(self) -> str:
        return f"Gateway(providers={[p.name for p in self.providers]}, strategy={self.strategy.name!r})"
