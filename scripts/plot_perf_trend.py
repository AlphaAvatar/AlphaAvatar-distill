"""Render the README performance-trend figure from assets/perf_trend.json.

Usage:
    uv run python scripts/plot_perf_trend.py

Every plotted point must be backed by an experiment log (AGENTS.md P7); the
data file records the log path for each point. Append new student attempts to
`attempts` as stages progress and re-run this script.

Output: assets/performance_trend.svg (committed; small, reviewable).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    data = json.loads((REPO_ROOT / "assets/perf_trend.json").read_text())
    attempts = data["attempts"]
    xs = list(range(1, len(attempts) + 1))
    ys = [a["nll"] for a in attempts]
    best = [min(ys[: i + 1]) for i in range(len(ys))]

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ref_styles = [("#2a9d2a", "--"), ("#888888", ":")]
    for ref, (color, ls) in zip(data["references"], ref_styles):
        ax.axhline(ref["nll"], color=color, linestyle=ls, linewidth=1.4, zorder=1)
        ax.annotate(f"{ref['label']}  ({ref['nll']:.2f})",
                    xy=(0.01, ref["nll"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    ha="left", fontsize=8.5, color=color)

    ax.plot(xs, best, color="#d62728", linewidth=1.6, drawstyle="steps-post",
            label="best student so far", zorder=2)
    ax.scatter(xs, ys, s=42, color="#1f77b4", zorder=3, label="student attempt")
    for x, a in zip(xs, attempts):
        dx, dy = a.get("label_offset", [6, 6])
        ax.annotate(a["label"], xy=(x, a["nll"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color="#333333")

    ax.set_xlabel("student checkpoint attempts (chronological)")
    ax.set_ylabel("held-out NLL (nats/token) — lower is better")
    ax.set_title("Qwen3-4B-Thinking-2507 → 0.6B student: held-out NLL by attempt")
    ax.set_xticks(xs)
    ax.set_xlim(0.5, len(xs) + 1.5)
    ax.set_ylim(0, max(ys) * 1.15)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.text(0.995, 0.01, "every point is backed by a log in logs/experiments/",
             ha="right", fontsize=7, color="#999999")
    fig.tight_layout()

    out = REPO_ROOT / "assets/performance_trend.svg"
    fig.savefig(out, format="svg", facecolor="white")
    print(f"Wrote {out} ({out.stat().st_size} bytes, {len(attempts)} attempts)")


if __name__ == "__main__":
    main()
