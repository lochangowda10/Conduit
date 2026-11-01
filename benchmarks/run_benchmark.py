"""End-to-end benchmark: the same workload through five gateway configurations.

Each scenario adds one capability, so the effect of each is visible:

  1. baseline       one premium provider, no retry, no breaker, no fallback, no cache
  2. +resilience    retries with backoff + circuit breakers + fallback chain
  3. +cache         2 + exact & semantic cache
  4. +smart routing 3 + complexity-aware, cost/latency-weighted routing
  5. +hedging       4 + hedged requests to cut tail latency

Load model: OPEN loop. Requests arrive on a fixed Poisson schedule (``--rps``)
no matter how fast earlier ones finished, like real users. A closed loop
("N workers, each sends the next request when the last returns") is easier
to write but lies: when the system fails fast, the workers race ahead and
the benchmark accidentally sends more traffic during an outage; when it is
slow, they send less and hide queueing delay ("coordinated omission").

Conditions are identical across scenarios: same seeds, same arrival times,
and the premium provider ("sim-large") is down for the same time window
[OUTAGE_START, OUTAGE_END) of the run.

All providers are *simulated* (latency, errors and prices are modelled, not
real API calls). Prices are illustrative tiers, not any vendor's price list.

Run:  python -m benchmarks.run_benchmark --n 1000 --rps 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from conduit import ChatRequest, Gateway, GatewayConfig
from conduit.analytics import provider_share, summarize, to_frame, underserved_rate
from conduit.errors import ConduitError
from conduit.providers import SimulatedProvider
from conduit.telemetry import export_csv

from .workload import REQUIRED_TIER, generate_workload

OUTAGE_START, OUTAGE_END = 0.30, 0.50  # fraction of the run during which "sim-large" is down

# Every provider has: log-normal latency, ~2 % HTTP 5xx, ~1 % HTTP 429, and 4 % "stalls"
# (an extra 2.5-5 s, modelling queueing / cold starts - the source of bad p99s).
_SHARED = dict(failure_rate=0.02, rate_limit_rate=0.01, stall_rate=0.04, stall_ms=2500, timeout_s=10.0)
PROVIDER_SPECS = [
    dict(name="sim-large", tier="large", model="sim-frontier", input_price_per_m=3.00, output_price_per_m=15.00,
         latency_median_ms=1100, latency_sigma=0.5, seed=101, **_SHARED),
    dict(name="sim-medium", tier="medium", model="sim-mid", input_price_per_m=0.50, output_price_per_m=2.00,
         latency_median_ms=650, latency_sigma=0.45, seed=202, **_SHARED),
    dict(name="sim-small", tier="small", model="sim-mini", input_price_per_m=0.10, output_price_per_m=0.40,
         latency_median_ms=350, latency_sigma=0.45, seed=303, **_SHARED),
]

COMMON = dict(breaker_failure_threshold=5, breaker_recovery_s=2.0, backoff_base_s=0.05,
              similarity_threshold=0.85)

SCENARIOS: dict[str, dict] = {
    "1 baseline": dict(strategy="single", cache_enabled=False, max_attempts_per_provider=1,
                       max_providers_per_request=1, breaker_failure_threshold=1000),  # i.e. no breaker
    "2 +resilience": dict(strategy="priority", cache_enabled=False, max_attempts_per_provider=2,
                          max_providers_per_request=3),
    "3 +cache": dict(strategy="priority", cache_enabled=True, max_attempts_per_provider=2,
                     max_providers_per_request=3),
    "4 +smart routing": dict(strategy="smart", cache_enabled=True, max_attempts_per_provider=2,
                             max_providers_per_request=3),
    "5 +hedging": dict(strategy="smart", cache_enabled=True, max_attempts_per_provider=2,
                       max_providers_per_request=3, hedge_after_ms=1000),
}


def build_gateway(name: str, overrides: dict, duration_s: float) -> Gateway:
    window = [[OUTAGE_START * duration_s, OUTAGE_END * duration_s]]
    providers = [
        SimulatedProvider(outages=window if spec["name"] == "sim-large" else None, **spec)
        for spec in PROVIDER_SPECS
    ]
    return Gateway(providers, GatewayConfig(**{**COMMON, **overrides}), scenario=name)


def arrival_times(n: int, rps: float, seed: int) -> list[float]:
    """Poisson process: exponential gaps with mean 1/rps."""
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rps)
        out.append(t)
    return out


def run_scenario(name: str, overrides: dict, items: list, arrivals: list[float]) -> tuple[pd.DataFrame, float, Gateway]:
    duration = arrivals[-1]
    gw = build_gateway(name, overrides, duration)

    def work(i: int) -> None:
        item = items[i]
        req = ChatRequest.from_prompt(item.prompt, max_tokens=300, metadata={"label": item.label})
        try:
            gw.complete(req)
        except ConduitError:
            pass  # already logged as an error record

    with ThreadPoolExecutor(max_workers=512) as pool:
        for p in gw.providers:
            p.reset_clock()  # outage window is relative to this moment
        t0 = time.perf_counter()
        for i, at in enumerate(arrivals):
            delay = t0 + at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pool.submit(work, i)
    wall = time.perf_counter() - t0
    gw.close()
    return to_frame(gw.logger.records), wall, gw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="requests per scenario")
    ap.add_argument("--rps", type=float, default=40.0, help="mean arrival rate (requests/second)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", help="run only scenarios whose name contains one of these words")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = list(generate_workload(args.n, seed=args.seed))
    arrivals = arrival_times(args.n, args.rps, args.seed)
    frames, waste = [], {}
    for name, overrides in SCENARIOS.items():
        if args.only and not any(w in name for w in args.only):
            continue
        print(f"running {name:<18} ...", end="", flush=True)
        df, wall, gw = run_scenario(name, overrides, items, arrivals)
        frames.append(df)
        waste[name] = gw.ledger.wasted.cost_usd
        print(f" {wall:5.1f}s")

    logs = pd.concat(frames, ignore_index=True)
    records = logs.drop(columns=["ok", "is_hit", "false_hit"]).to_dict("records")
    export_csv(records, out / "benchmark_logs.csv")

    summary = summarize(logs)
    tiers = {s["name"]: s["tier"] for s in PROVIDER_SPECS}
    summary["underserved_rate"] = underserved_rate(logs, tiers, REQUIRED_TIER)
    # Hedging pays for the losing call too, so it must be included in the cost.
    summary["hedge_waste_usd"] = pd.Series(waste)
    summary["total_cost_usd"] += summary["hedge_waste_usd"]
    summary["cost_per_1k_usd"] = 1000 * summary["total_cost_usd"] / summary["requests"]
    summary.to_csv(out / "benchmark_summary.csv")
    provider_share(logs).to_csv(out / "provider_share.csv")

    base, full = summary.iloc[0], summary.iloc[-1]
    headline = {
        "requests_per_scenario": args.n,
        "arrival_rate_rps": args.rps,
        "outage_window_fraction": [OUTAGE_START, OUTAGE_END],
        "baseline_success_rate": base["success_rate"],
        "full_success_rate": full["success_rate"],
        "baseline_cost_per_1k_usd": base["cost_per_1k_usd"],
        "full_cost_per_1k_usd": full["cost_per_1k_usd"],
        "cost_reduction_pct": 100 * (1 - full["cost_per_1k_usd"] / base["cost_per_1k_usd"]),
        "baseline_p50_ms": base["p50"],
        "full_p50_ms": full["p50"],
        "baseline_p95_ms": base["p95"],
        "full_p95_ms": full["p95"],
        "baseline_p99_ms": base["p99"],
        "full_p99_ms": full["p99"],
        "full_cache_hit_rate": full["cache_hit_rate"],
        "full_false_hits": int(full["false_hits"]),
        "full_underserved_rate": full["underserved_rate"],
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2, default=float))

    pd.set_option("display.width", 200)
    cols = ["success_rate", "cache_hit_rate", "false_hits", "p50", "p95", "p99", "cost_per_1k_usd",
            "hedge_waste_usd", "underserved_rate", "hedged_rate", "mean_attempts"]
    print()
    print(summary[cols].round(4).to_string())
    print()
    print(json.dumps(headline, indent=2, default=float))


if __name__ == "__main__":
    main()
