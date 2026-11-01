"""Measure semantic-cache quality: hit rate vs. false-hit rate across thresholds.

A semantic cache has a precision/recall trade-off, just like a classifier:
* lower threshold  -> more hits (cheaper) but more *false hits* (wrong answers)
* higher threshold -> safer but fewer savings

We replay the labelled workload through ``ResponseCache`` directly (no
provider calls needed) and, for every threshold, count:
* hit_rate        - fraction of requests answered from cache
* false_hit_rate  - fraction of hits that returned another question's answer
* recall          - correct hits / requests that *could* have hit (an
                    equivalent prompt was already cached)

Run:  python -m benchmarks.cache_eval --n 3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conduit.cache import ResponseCache
from conduit.embeddings import HashingEmbedder
from conduit.types import ChatRequest, ChatResponse, Usage

from .workload import generate_workload

CONFIGS = {
    "exact only": dict(semantic_enabled=False, idf_weighting=False, entity_guard=False),
    "hashing": dict(semantic_enabled=True, idf_weighting=False, entity_guard=False),
    "hashing + IDF": dict(semantic_enabled=True, idf_weighting=True, entity_guard=False),
    "hashing + IDF + guard": dict(semantic_enabled=True, idf_weighting=True, entity_guard=True),
}


def replay(items, threshold: float, **cache_kwargs) -> dict:
    cache = ResponseCache(embedder=HashingEmbedder(), capacity=len(items) + 1,
                          similarity_threshold=threshold, **cache_kwargs)
    seen: set[str] = set()
    hits = correct = opportunities = 0
    for item in items:
        req = ChatRequest.from_prompt(item.prompt)
        could_hit = item.label in seen
        opportunities += could_hit
        found = cache.get(req)
        if found is not None:
            hits += 1
            correct += found[0].meta["label"] == item.label
        else:
            cache.put(req, ChatResponse(item.label, "eval", "eval", Usage(), 0.0), meta={"label": item.label})
        seen.add(item.label)
    n = len(items)
    return {
        "threshold": threshold,
        "hit_rate": hits / n,
        "false_hit_rate": (hits - correct) / hits if hits else 0.0,
        "wrong_answers_per_1k": 1000 * (hits - correct) / n,
        "recall": correct / opportunities if opportunities else 0.0,
        "oracle_hit_rate": opportunities / n,
        "guard_rejections": cache.stats.guard_rejections,
    }


def run(n: int = 3000, seed: int = 11) -> pd.DataFrame:
    items = list(generate_workload(n, seed=seed))
    rows = []
    for name, kwargs in CONFIGS.items():
        thresholds = [1.0] if name == "exact only" else np.round(np.arange(0.50, 0.96, 0.05), 2)
        for t in thresholds:
            row = replay(items, float(t), **kwargs)
            row["config"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    df = run(args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cache_threshold_sweep.csv", index=False)
    pd.set_option("display.width", 140)
    print(df.round(3).to_string(index=False))

    # Pick the operating point: the highest hit rate with <= 1 wrong answer per 1,000 requests.
    best = df[(df.config == "hashing + IDF + guard") & (df.wrong_answers_per_1k <= 1.0)]
    best = best.sort_values("hit_rate", ascending=False).head(1)
    if not best.empty:
        choice = best.iloc[0].to_dict()
        (out / "cache_operating_point.json").write_text(json.dumps(choice, indent=2, default=float))
        print("\nRecommended operating point:", json.dumps(choice, default=float))


if __name__ == "__main__":
    main()
