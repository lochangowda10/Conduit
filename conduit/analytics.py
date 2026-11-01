"""Turn request logs into numbers with Pandas.

Syllabus link
-------------
* Experiment 10 (Pandas): ``DataFrame`` creation from records, boolean
  filtering, ``sort_values``, ``groupby().agg()`` with named aggregations,
  ``pivot_table``, and explicit missing-value handling (``fillna`` for cost,
  ``dropna`` for latency percentiles).
* Experiment 9 (NumPy): percentiles via ``np.percentile`` inside the
  aggregation.

Missing values are meaningful here, not noise:
* a failed request has no tokens and no cost -> cost is filled with 0
  (we really did pay nothing),
* a failed request's latency is not comparable to a successful one, so
  latency percentiles are computed on successful rows only (``dropna``).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .telemetry import RequestRecord, load_jsonl

TIER_RANK = {"small": 0, "medium": 1, "large": 2}


def to_frame(records: Iterable[RequestRecord] | Iterable[dict]) -> pd.DataFrame:
    rows = [asdict(r) if isinstance(r, RequestRecord) else r for r in records]
    df = pd.DataFrame(rows, columns=RequestRecord.columns())
    return clean(df)


def load_logs(path: str) -> tuple[pd.DataFrame, dict]:
    records, stats = load_jsonl(path)
    return to_frame(records), stats


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    for col in ("prompt_tokens", "completion_tokens", "attempts"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["cache"] = df["cache"].fillna("miss")
    df["hedged"] = df["hedged"].fillna(False).astype(bool)
    df["ok"] = df["status"].eq("ok")
    df["is_hit"] = df["cache"].isin(["exact", "semantic"])
    # A false hit: answered from cache with an entry that belongs to another question.
    df["false_hit"] = df["is_hit"] & df["cached_label"].notna() & df["label"].notna() & (df["cached_label"] != df["label"])
    return df


def _p(q: float):
    def f(s: pd.Series) -> float:
        s = s.dropna()
        return float(np.percentile(s, q)) if len(s) else float("nan")
    f.__name__ = f"p{q:g}"
    return f


def summarize(df: pd.DataFrame, by: str = "scenario") -> pd.DataFrame:
    ok = df[df["ok"]]
    lat = ok.groupby(by, sort=False)["latency_ms"].agg(p50=_p(50), p95=_p(95), p99=_p(99), mean="mean")
    base = df.groupby(by, sort=False).agg(
        requests=("request_id", "count"),
        success_rate=("ok", "mean"),
        cache_hit_rate=("is_hit", "mean"),
        false_hits=("false_hit", "sum"),
        hedged_rate=("hedged", "mean"),
        mean_attempts=("attempts", "mean"),
        total_cost_usd=("cost_usd", "sum"),
    )
    out = base.join(lat)
    out["cost_per_1k_usd"] = 1000 * out["total_cost_usd"] / out["requests"]
    return out


def provider_share(df: pd.DataFrame, by: str = "scenario") -> pd.DataFrame:
    """Share of successful *provider-served* requests (cache hits excluded) per provider."""
    served = df[df["ok"] & ~df["is_hit"]]
    table = pd.pivot_table(served, index=by, columns="provider", values="request_id", aggfunc="count", fill_value=0)
    return table.div(table.sum(axis=1), axis=0)


def underserved_rate(df: pd.DataFrame, provider_tiers: dict[str, str], required_tier: dict[str, str],
                     by: str = "scenario") -> pd.Series:
    """Fraction of provider-served requests answered by a weaker tier than the task needs.

    ``required_tier`` maps an intent (the part of the label before ':') to a tier.
    A quality *proxy*: it can't judge answers, but it shows whether cheap
    routing is sending hard prompts to small models.
    """
    served = df[df["ok"] & ~df["is_hit"] & df["label"].notna()].copy()
    intent = served["label"].str.split(":").str[0]
    need = intent.map(required_tier).fillna("small").map(TIER_RANK)
    got = served["provider"].map(provider_tiers).map(TIER_RANK)
    served["under"] = got < need
    return served.groupby(by, sort=False)["under"].mean()


def cumulative_hit_rate(df: pd.DataFrame, by: str = "scenario") -> pd.DataFrame:
    df = df.sort_values("ts")
    df["n"] = df.groupby(by).cumcount() + 1
    df["cum_hit_rate"] = df.groupby(by)["is_hit"].cumsum() / df["n"]
    return df[[by, "n", "cum_hit_rate"]]
