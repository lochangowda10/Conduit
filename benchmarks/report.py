"""Draw the benchmark charts with Matplotlib and save them as PNG files.

Syllabus link - Experiment 11 (Matplotlib): line plots, bar charts and a
histogram, each with a title, axis labels and a legend where there is more
than one series, saved to files with ``savefig``. ``plt.subplots`` is used for
the two-panel chart (two measures with different units get two panels,
never one chart with two y-axes).

Run:  python -m benchmarks.report --results results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed (works on servers/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conduit.analytics import clean

# Validated categorical slots (fixed order) + chart chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REFERENCE_GRAY = "#898781"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
        "font.family": "DejaVu Sans", "figure.dpi": 110,
    })


def _save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def chart_cost(summary: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    s = summary["cost_per_1k_usd"][::-1]
    bars = ax.barh(s.index, s.values, color=BLUE, height=0.55)
    ax.bar_label(bars, labels=[f"${v:.3f}" for v in s.values], padding=4, color=INK2, fontsize=9)
    ax.set_title("Cost per 1,000 requests")
    ax.set_xlabel("USD (simulated prices, includes hedge waste)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, s.max() * 1.18)
    return _save(fig, out, "cost_per_1k.png")


def chart_latency_percentiles(summary: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(summary))
    w = 0.26
    for i, (col, color, label) in enumerate([("p50", BLUE, "p50"), ("p95", ORANGE, "p95"), ("p99", AQUA, "p99")]):
        bars = ax.bar(x + (i - 1) * w, summary[col] / 1000, width=w - 0.03, color=color, label=label)
        if col == "p99":
            ax.bar_label(bars, labels=[f"{v / 1000:.1f}s" for v in summary[col]], padding=3, color=INK2, fontsize=8)
    ax.set_xticks(x, [n.split(" ", 1)[1] for n in summary.index])
    ax.set_title("End-to-end latency percentiles (successful requests)")
    ax.set_ylabel("seconds")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=3, loc="upper right")
    return _save(fig, out, "latency_percentiles.png")


def chart_latency_histogram(logs: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ok = logs[logs["ok"]]
    first, last = logs["scenario"].iloc[0], logs["scenario"].iloc[-1]
    bins = np.logspace(np.log10(0.05), np.log10(12_000), 60)
    for scen, color in ((first, BLUE), (last, ORANGE)):
        data = ok.loc[ok["scenario"] == scen, "latency_ms"].clip(lower=0.06)
        ax.hist(data, bins=bins, histtype="step", linewidth=2, color=color, label=scen.split(" ", 1)[1])
    ax.set_xscale("log")
    ax.set_title("Latency distribution: baseline vs full gateway")
    ax.set_xlabel("latency (ms, log scale) - left peaks are cache hits: exact ~0.1 ms, semantic ~2-3 ms")
    ax.set_ylabel("requests")
    ax.legend()
    return _save(fig, out, "latency_histogram.png")


def chart_success_over_time(logs: pd.DataFrame, out: Path, outage: tuple[float, float] = (0.30, 0.50)) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    scenarios = list(dict.fromkeys(logs["scenario"]))
    picks = [(scenarios[0], BLUE), (scenarios[1], ORANGE)]
    duration = None
    for scen, color in picks:
        d = logs[logs["scenario"] == scen].sort_values("ts").copy()
        d["t"] = d["ts"] - d["ts"].min()
        duration = d["t"].max() if duration is None else max(duration, d["t"].max())
        d["sec"] = d["t"].astype(int)
        per_sec = d.groupby("sec")["ok"].mean()
        # plot each 1-second bucket at its centre so the dip lines up with the outage band
        ax.plot(per_sec.index + 0.5, 100 * per_sec.values, color=color, linewidth=2, label=scen.split(" ", 1)[1])
    ax.axvspan(outage[0] * duration, outage[1] * duration, color=GRID, alpha=0.7, linewidth=0)
    ax.text(np.mean(outage) * duration, 8, "premium provider\noutage", ha="center", color=INK2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Success rate during a provider outage")
    ax.set_xlabel("seconds since start of run")
    ax.set_ylabel("% of requests succeeding (per second)")
    ax.legend(loc="lower right")
    return _save(fig, out, "success_over_time.png")


def chart_provider_share(share: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    order = [c for c in ("sim-large", "sim-medium", "sim-small") if c in share.columns]
    colors = {"sim-large": BLUE, "sim-medium": ORANGE, "sim-small": AQUA}
    share = share[order][::-1] * 100
    left = np.zeros(len(share))
    for col in order:
        ax.barh(share.index, share[col], left=left, color=colors[col], label=col, height=0.55,
                edgecolor=SURFACE, linewidth=1.5)
        for y, (l, v) in enumerate(zip(left, share[col])):
            if v >= 9:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="#ffffff", fontsize=8.5)
        left += share[col].values
    ax.set_xlim(0, 100)
    ax.set_title("Which provider served each request (cache hits excluded)")
    ax.set_xlabel("% of provider-served requests")
    ax.grid(axis="y", visible=False)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return _save(fig, out, "provider_share.png")


def chart_cache_sweep(sweep: pd.DataFrame, out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    configs = [("hashing", BLUE), ("hashing + IDF", ORANGE), ("hashing + IDF + guard", AQUA)]
    exact = sweep[sweep["config"] == "exact only"]
    for name, color in configs:
        d = sweep[sweep["config"] == name]
        ax1.plot(d["threshold"], 100 * d["hit_rate"], color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax2.plot(d["threshold"], d["wrong_answers_per_1k"], color=color, linewidth=2, marker="o", markersize=4, label=name)
    if not exact.empty:
        ax1.axhline(100 * exact["hit_rate"].iloc[0], color=REFERENCE_GRAY, linestyle="--", linewidth=1.2)
        ax1.text(0.95, 100 * exact["hit_rate"].iloc[0] + 1.5, "exact-match cache only", color=INK2, fontsize=8.5,
                 ha="right")
    for ax in (ax1, ax2):
        ax.axvline(0.85, color=AXIS, linewidth=1)
        ax.set_xlabel("similarity threshold")
    ax1.set_title("Cache hit rate")
    ax1.set_ylabel("% of requests served from cache")
    ax2.set_title("Wrong answers from cache")
    ax2.set_ylabel("false hits per 1,000 requests")
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_ylim(bottom=0)
    ax1.set_ylim(bottom=25)
    ax2.legend(loc="upper right")
    return _save(fig, out, "cache_threshold_sweep.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out = Path(args.results)
    _style()
    made = []
    summary_path = out / "benchmark_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, index_col=0)
        logs = clean(pd.read_csv(out / "benchmark_logs.csv"))
        made += [chart_cost(summary, out), chart_latency_percentiles(summary, out),
                 chart_latency_histogram(logs, out), chart_success_over_time(logs, out)]
        share_path = out / "provider_share.csv"
        if share_path.exists():
            made.append(chart_provider_share(pd.read_csv(share_path, index_col=0), out))
    else:
        print(f"no {summary_path}; run `python -m benchmarks.run_benchmark` first")
    sweep_path = out / "cache_threshold_sweep.csv"
    if sweep_path.exists():
        made.append(chart_cache_sweep(pd.read_csv(sweep_path), out))
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
