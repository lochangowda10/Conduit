import pytest

from conduit import ChatRequest, Gateway, GatewayConfig
from conduit.errors import AllProvidersFailed, BudgetExceeded, ConfigError, ProviderUnavailable
from conduit.providers import LLMProvider, SimulatedProvider
from conduit.router import FallbackChain, SmartRouter, complexity_score, tier_for
from conduit.resilience import CircuitBreaker

HARD = ("Design a distributed job scheduler. Compare the trade-offs of two approaches, analyze failure "
        "modes step by step and explain how it scales.")


class TestRouter:
    def test_complexity_ordering(self):
        assert complexity_score("What is 2+2?") < complexity_score(HARD)
        assert tier_for("What is the capital of France?") == "small"
        assert tier_for(HARD) == "large"

    def test_smart_router_prefers_cheap_capable(self, fast_provider):
        small = fast_provider("s", "small", input_price_per_m=0.1, output_price_per_m=0.4)
        large = fast_provider("l", "large", input_price_per_m=3, output_price_per_m=15)
        router = SmartRouter()
        assert [p.name for p in router.rank(ChatRequest.from_prompt("hi there"), [large, small])] == ["s", "l"]
        assert [p.name for p in router.rank(ChatRequest.from_prompt(HARD), [large, small])] == ["l", "s"]
        pinned = router.rank(ChatRequest.from_prompt("hi", model="l"), [small, large])
        assert pinned[0].name == "l"

    def test_fallback_chain_skips_open_breakers(self, fast_provider):
        ps = [fast_provider(n) for n in "abc"]
        breakers = {p.name: CircuitBreaker(p.name, failure_threshold=1) for p in ps}
        breakers["a"].record_failure()
        chain = FallbackChain(ps, breakers)
        assert iter(chain) is chain
        assert [p.name for p in chain] == ["b", "c"] and chain.skipped == ["a"]

    def test_next_if(self, fast_provider):
        ps = [fast_provider("a", "small"), fast_provider("b", "large")]
        chain = FallbackChain(ps, {})
        assert chain.next_if(lambda p: p.tier == "large").name == "b"
        assert [p.name for p in chain] == ["a"]


class TestProviders:
    def test_registry_factory(self):
        p = LLMProvider.from_config({"type": "simulated", "name": "x1", "time_scale": 0.01})
        assert isinstance(p, SimulatedProvider)
        with pytest.raises(ConfigError, match="unknown provider type"):
            LLMProvider.from_config({"type": "nope", "name": "x"})
        with pytest.raises(ConfigError):
            LLMProvider.from_config({"type": "simulated", "name": "BAD NAME"})

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        with pytest.raises(ConfigError, match="NOPE_KEY"):
            LLMProvider.from_config({"type": "openai_compatible", "name": "r", "model": "m",
                                     "base_url": "https://example.com/v1", "api_key_env": "NOPE_KEY"})

    def test_optional_provider_skipped(self, monkeypatch):
        monkeypatch.delenv("NOPE_KEY", raising=False)
        cfg = GatewayConfig(providers=[
            {"type": "openai_compatible", "name": "r", "model": "m", "base_url": "https://e.com/v1",
             "api_key_env": "NOPE_KEY", "optional": True},
            {"type": "simulated", "name": "s", "time_scale": 0.01},
        ])
        with Gateway.from_config(cfg) as gw:
            assert [p.name for p in gw.providers] == ["s"]

    def test_mro_and_mixins(self, fast_provider):
        p = fast_provider()
        names = [c.__name__ for c in type(p).__mro__]
        assert names[:4] == ["SimulatedProvider", "LoggableMixin", "MetricsMixin", "LLMProvider"]
        p.complete(ChatRequest.from_prompt("hi"))
        assert p.calls == 1 and p.ewma_latency_ms is not None


class TestGateway:
    def test_basic_and_cache(self, fast_provider):
        with Gateway([fast_provider()]) as gw:
            r1 = gw.complete(ChatRequest.from_prompt("What is the capital of France?"))
            r2 = gw.complete(ChatRequest.from_prompt("What is the capital of France?"))
            r3 = gw.complete(ChatRequest.from_prompt("Tell me the capital of France."))
            assert (r1.cache, r2.cache, r3.cache) == ("miss", "exact", "semantic")
            assert r2.text == r1.text and r2.usage.cost_usd == 0
            assert len(gw.ledger) == 1

    def test_fallback_on_outage(self, fast_provider):
        a, b = fast_provider("a"), fast_provider("b")
        a.forced_down = True
        cfg = GatewayConfig(strategy="priority", cache_enabled=False, max_attempts_per_provider=1)
        with Gateway([a, b], cfg) as gw:
            r = gw.complete(ChatRequest.from_prompt("hello"))
            assert r.provider == "b" and r.attempts == 2

    def test_breaker_trips_and_skips(self, fast_provider):
        a, b = fast_provider("a"), fast_provider("b")
        a.forced_down = True
        cfg = GatewayConfig(strategy="priority", cache_enabled=False, max_attempts_per_provider=1,
                            breaker_failure_threshold=2, breaker_recovery_s=60)
        with Gateway([a, b], cfg) as gw:
            for i in range(5):
                gw.complete(ChatRequest.from_prompt(f"hello {i}"))
            assert a.calls == 2  # after 2 failures the circuit opened; a was never called again
            assert gw.breakers["a"].state.value == "open"

    def test_all_fail(self, fast_provider):
        a = fast_provider("a")
        a.forced_down = True
        with Gateway([a], GatewayConfig(cache_enabled=False, max_attempts_per_provider=1)) as gw:
            with pytest.raises(AllProvidersFailed):
                gw.complete(ChatRequest.from_prompt("x"))
            assert gw.logger.records[-1].status == "error"

    def test_hedging_beats_a_stall(self, fast_provider):
        slow = fast_provider("slow", "small", latency_median_ms=3000, latency_sigma=0, time_scale=0.1)  # 300 ms
        fast = fast_provider("fast", "medium", latency_median_ms=200, latency_sigma=0, time_scale=0.1)   # 20 ms
        cfg = GatewayConfig(strategy="priority", cache_enabled=False, hedge_after_ms=50)
        with Gateway([slow, fast], cfg) as gw:
            r = gw.complete(ChatRequest.from_prompt("hedge me"))
            assert r.provider == "fast" and r.hedged and r.latency_ms < 250

    def test_hedge_never_downgrades_tier(self, fast_provider):
        large = fast_provider("large", "large", latency_median_ms=2000, latency_sigma=0, time_scale=0.1)
        small = fast_provider("small", "small", latency_median_ms=100, latency_sigma=0, time_scale=0.1)
        cfg = GatewayConfig(strategy="priority", cache_enabled=False, hedge_after_ms=20)
        with Gateway([large, small], cfg) as gw:
            r = gw.complete(ChatRequest.from_prompt("hard question"))
            assert r.provider == "large" and not r.hedged

    def test_stream_falls_back_before_first_token(self, fast_provider):
        a, b = fast_provider("a"), fast_provider("b")
        a.forced_down = True
        cfg = GatewayConfig(strategy="priority", cache_enabled=False)
        with Gateway([a, b], cfg) as gw:
            text = "".join(gw.stream(ChatRequest.from_prompt("stream please")))
            assert text and gw.logger.records[-1].provider == "b"

    def test_mid_stream_failure_is_surfaced(self, fast_provider):
        p = fast_provider("a", mid_stream_failure_rate=1.0)
        with Gateway([p], GatewayConfig(cache_enabled=False)) as gw:
            with pytest.raises(ProviderUnavailable):
                list(gw.stream(ChatRequest.from_prompt("a long enough prompt to produce several chunks " * 3)))

    def test_budget_cap_still_serves_cache(self, fast_provider):
        with Gateway([fast_provider(input_price_per_m=1000, output_price_per_m=1000)],
                     GatewayConfig(budget_usd=0.001)) as gw:
            gw.complete(ChatRequest.from_prompt("expensive question"))
            assert gw.complete(ChatRequest.from_prompt("expensive question")).cache == "exact"
            with pytest.raises(BudgetExceeded):
                gw.complete(ChatRequest.from_prompt("a brand new question"))

    def test_duplicate_names_rejected(self, fast_provider):
        with pytest.raises(ConfigError):
            Gateway([fast_provider("x"), fast_provider("x")])
