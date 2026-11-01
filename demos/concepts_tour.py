"""A guided tour: every syllabus experiment, demonstrated live inside Conduit.

Run:  python -m conduit tour      (or: python -m demos.concepts_tour)

Each section prints which file implements the concept, then runs real code.
Use it in the viva: "here is experiment N, and here is where it lives in the project".
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from conduit import ChatRequest, Gateway, GatewayConfig, Usage
from conduit.analytics import summarize, to_frame
from conduit.embeddings import HashingEmbedder, cosine_similarity_matrix
from conduit.errors import ConfigError, ProviderUnavailable, RetryableError
from conduit.providers import LLMProvider, LocalProvider, SimulatedProvider
from conduit.ratelimit import TokenBucket
from conduit.resilience import CircuitBreaker, log_execution, retry
from conduit.router import FallbackChain, SmartRouter, complexity_score
from conduit.telemetry import RequestRecord, export_csv, load_jsonl


def section(n: int, title: str, where: str) -> None:
    print("\n" + "=" * 78)
    print(f"EXPERIMENT {n}: {title}")
    print(f"  lives in: {where}")
    print("=" * 78)


def exp1() -> None:
    section(1, "Instance, class and static methods", "providers/base.py, types.py")
    print("static  LLMProvider.validate_name('groq-small')  ->", LLMProvider.validate_name("groq-small"))
    print("static  LLMProvider.validate_name('Bad Name!')   ->", LLMProvider.validate_name("Bad Name!"))
    p = LLMProvider.from_config({"type": "simulated", "name": "sim-a", "tier": "small", "time_scale": 0.01})
    print("class   LLMProvider.from_config({...})          ->", p)
    print("        registry filled by __init_subclass__    ->", sorted(LLMProvider.registry))
    req = ChatRequest.from_prompt("Hello there")
    print("class   ChatRequest.from_prompt(...)            ->", req.messages)
    r = p.complete(req)
    print("instance p.complete(req)                        ->", r.text[:60], "...")


def exp2() -> None:
    section(2, "Encapsulation and properties", "config.py, ratelimit.py, ledger.py")
    cfg = GatewayConfig()
    print("cfg.similarity_threshold =", cfg.similarity_threshold)
    for bad in (1.7, -3, "high"):
        try:
            cfg.similarity_threshold = bad
        except ConfigError as exc:
            print(f"cfg.similarity_threshold = {bad!r:<7} -> rejected: {exc}")
    bucket = TokenBucket(rate=5, capacity=10)
    print("bucket.available (computed property) =", round(bucket.available, 2))
    try:
        bucket.rate = 0
    except ValueError as exc:
        print("bucket.rate = 0 -> rejected:", exc)


def exp3() -> None:
    section(3, "Inheritance, polymorphism and super()", "providers/*.py")
    providers = [
        SimulatedProvider("sim-cheap", input_price_per_m=0.1, output_price_per_m=0.4, time_scale=0.01),
        SimulatedProvider("sim-premium", input_price_per_m=3, output_price_per_m=15, time_scale=0.01),
        LocalProvider("ollama", model="llama3.2:3b"),
    ]
    for p in providers:  # same call, different behaviour: polymorphism
        print(f"  {type(p).__name__:<20} calculate_cost(1000, 500) = ${p.calculate_cost(1000, 500):.6f}")
    print("LocalProvider.__init__ calls super().__init__ with base_url =", providers[2].base_url)


def exp4() -> None:
    section(4, "Abstract classes, multiple inheritance (mixins) and MRO", "providers/base.py")
    try:
        LLMProvider("x", "y")  # type: ignore[abstract]
    except TypeError as exc:
        print("LLMProvider('x','y') ->", exc)
    print("SimulatedProvider.__mro__:")
    for i, cls in enumerate(SimulatedProvider.__mro__):
        print(f"  {i}. {cls.__module__}.{cls.__qualname__}")
    p = SimulatedProvider("sim-mro", time_scale=0.01)
    p.complete(ChatRequest.from_prompt("hi"))
    print("mixin state after one call ->", p.metrics(), "| has log():", hasattr(p, "log"))


def exp5() -> None:
    section(5, "Magic methods", "types.py (Usage), ledger.py, gateway.py")
    a, b = Usage(100, 50, 0.002), Usage(40, 10, 0.0005)
    print("a       ->", repr(a))
    print("str(a)  ->", str(a))
    print("a + b   ->", a + b)
    print("len(a)  ->", len(a), "(total tokens)")
    print("sum([a, b, a]) ->", sum([a, b, a]), "(needs __radd__ because sum starts at 0)")
    print("a == Usage(100, 50, 0.002) ->", a == Usage(100, 50, 0.002), "| b < a ->", b < a)
    print("bool(Usage()) ->", bool(Usage()))
    print("Gateway supports `with Gateway(...) as gw:` via __enter__/__exit__")


def exp6() -> None:
    section(6, "Decorators, iterators and generators", "resilience.py, router.py, gateway.py, workload.py")
    calls = {"n": 0}

    @log_execution
    @retry(max_attempts=4, base_delay=0.01, retry_on=(RetryableError,),
           on_retry=lambda i, e: print(f"  attempt {i} failed ({e}); backing off..."))
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderUnavailable("HTTP 503", "demo")
        return "success on attempt 3"

    print("@retry ->", flaky(), "| name preserved by functools.wraps:", flaky.__name__)

    providers = [SimulatedProvider(n, time_scale=0.01) for n in ("p1", "p2", "p3")]
    breakers = {p.name: CircuitBreaker(p.name, failure_threshold=1) for p in providers}
    breakers["p2"].record_failure()  # p2's circuit is now open
    chain = FallbackChain(providers, breakers)
    print("FallbackChain yields ->", [p.name for p in chain], "| skipped:", chain.skipped)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmarks.workload import generate_workload

    gen = generate_workload(1_000_000)
    as_list = list(generate_workload(10_000))
    print(f"generator for 1,000,000 prompts: {sys.getsizeof(gen)} bytes | "
          f"list of only 10,000: {sys.getsizeof(as_list) + sum(sys.getsizeof(x.prompt) for x in as_list):,} bytes")
    print("next(gen) ->", next(gen).prompt)

    with Gateway([SimulatedProvider("sim-stream", time_scale=0.05)]) as gw:
        print("gw.stream(...) yields:", end=" ")
        for chunk in gw.stream(ChatRequest.from_prompt("Explain generators")):
            print(repr(chunk), end=" ")
        print()


def exp7() -> None:
    section(7, "CSV and JSON processing with exception handling", "telemetry.py, config.py")
    tmp = Path(tempfile.mkdtemp())
    log = tmp / "requests.jsonl"
    records = [RequestRecord(f"r{i}", provider="sim", latency_ms=100 + i, cost_usd=0.001 * i) for i in range(3)]
    log.write_text("\n".join(json.dumps(r.__dict__) for r in records) + "\n{this line is corrupted\n\n[1,2]\n")
    loaded, stats = load_jsonl(log)
    print(f"load_jsonl -> {len(loaded)} good records, stats = {stats}")
    print("export_csv ->", export_csv(records, tmp / "requests.csv"), "rows ->", (tmp / "requests.csv").read_text().splitlines()[0][:70], "...")
    for path, content in (("missing.json", None), ("broken.json", "{ \"strategy\": smart }")):
        if content:
            (tmp / path).write_text(content)
        try:
            GatewayConfig.from_json(tmp / path)
        except ConfigError as exc:
            print("GatewayConfig.from_json ->", exc)


def exp8() -> None:
    section(8, "Multithreading and concurrency", "ledger.py, ratelimit.py, resilience.py, gateway.py, server.py")
    from demos.race_condition_demo import demo_race

    demo_race()
    bucket = TokenBucket(rate=20, capacity=5)
    t0 = time.perf_counter()
    for _ in range(15):
        bucket.acquire()
    print(f"TokenBucket(rate=20/s, burst=5): 15 acquires took {time.perf_counter() - t0:.2f}s "
          "(5 instant, then 10 at 20/s = ~0.5s)")
    print("Hedged requests use ThreadPoolExecutor + concurrent.futures.wait(FIRST_COMPLETED) - see gateway._run_chain")


def exp9() -> None:
    section(9, "NumPy", "embeddings.py, cache.py")
    prompts = ["What is the capital of France?", "Tell me France's capital.", "What is the capital of Spain?",
               "Explain recursion simply."]
    emb = HashingEmbedder(dim=256)
    m = emb.embed_many(prompts)
    print("embedding matrix shape:", m.shape, "| row norms:", np.round(np.linalg.norm(m, axis=1), 3))
    sim = cosine_similarity_matrix(m, m)
    print("cosine similarity matrix (one matrix multiply):")
    print(pd.DataFrame(np.round(sim, 2), index=[p[:22] for p in prompts], columns=range(4)).to_string())
    print("slicing sim[0, 1:]:", np.round(sim[0, 1:], 2), "| argmax excluding self:", int(np.argmax(sim[0, 1:]) + 1))
    print("axis-wise stats: mean per row", np.round(sim.mean(axis=1), 2), "max per column", np.round(sim.max(axis=0), 2))
    print("reshape 256-d vector -> (16, 16):", m[0].reshape(16, 16).shape)


def exp10() -> None:
    section(10, "Pandas", "analytics.py")
    rng = np.random.default_rng(0)
    recs = []
    for i in range(12):
        ok = i % 5 != 4
        recs.append(RequestRecord(f"r{i}", scenario="demo", status="ok" if ok else "error",
                                  provider=["sim-small", "sim-large"][i % 2] if ok else None,
                                  latency_ms=float(rng.integers(100, 900)) if ok else None,
                                  cost_usd=round(float(rng.random()) / 1000, 6) if ok else None,
                                  cache=["miss", "exact", "semantic"][i % 3]))
    df = to_frame(recs)
    print(f"failed rows: latency stays NaN ({df['latency_ms'].isna().sum()} rows, dropped with dropna() for percentiles); "
          f"cost NaN -> fillna(0.0) (total cost ${df['cost_usd'].sum():.6f})")
    print(df[["request_id", "status", "provider", "cache", "latency_ms", "cost_usd"]].sort_values("latency_ms").head(5).to_string())
    print("\ngroupby provider:")
    print(df[df["ok"]].groupby("provider").agg(requests=("request_id", "count"), mean_ms=("latency_ms", "mean"),
                                                cost=("cost_usd", "sum")).round(4).to_string())
    print("\nsummarize():")
    print(summarize(df)[["requests", "success_rate", "cache_hit_rate", "p50", "p95"]].round(2).to_string())


def exp11() -> None:
    section(11, "Matplotlib", "benchmarks/report.py")
    print("python -m conduit bench   # run the benchmark (writes results/*.csv)")
    print("python -m conduit report  # draws line plot, bar charts, histogram -> results/*.png")
    pngs = sorted(Path("results").glob("*.png"))
    print("charts already in results/:", [p.name for p in pngs] or "none yet")


def exp12() -> None:
    section(12, "Mini project: everything together", "gateway.py, server.py")
    cfg = GatewayConfig(similarity_threshold=0.85)
    providers = [
        SimulatedProvider("sim-small", tier="small", input_price_per_m=0.1, output_price_per_m=0.4,
                          latency_median_ms=300, time_scale=0.2),
        SimulatedProvider("sim-large", tier="large", input_price_per_m=3, output_price_per_m=15,
                          latency_median_ms=900, time_scale=0.2),
    ]
    with Gateway(providers, cfg, strategy=SmartRouter()) as gw:
        for prompt in ["What is the capital of France?", "Tell me the capital of France.",
                       "Design a distributed job scheduler. Compare the trade-offs, analyze failure modes "
                       "step by step and explain how it scales."]:
            r = gw.complete(ChatRequest.from_prompt(prompt))
            print(f"  complexity={complexity_score(prompt):.2f} -> {r.provider:<10} cache={r.cache:<9} "
                  f"${r.usage.cost_usd:.6f}  {prompt[:45]}")
        providers[1].forced_down = True
        r = gw.complete(ChatRequest.from_prompt("Design a URL shortener. Compare trade-offs and analyze scaling step by step."))
        print(f"  sim-large DOWN -> served by {r.provider} after {r.attempts} attempts (graceful degradation)")
        print("  ledger:", gw.ledger)


def main() -> None:
    for fn in (exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10, exp11, exp12):
        fn()
    print("\nTour complete.")


if __name__ == "__main__":
    main()
