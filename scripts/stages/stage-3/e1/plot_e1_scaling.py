"""Render the Experiment 1 scaling curves.

    uv run python scripts/evaluation/plot_e1_scaling.py

Two panels, not one chart with two y-axes: teacher-native CE is in nats and
natural termination is a rate, so a shared axis would be meaningless. Both share
the x-axis (supervised tokens, log-spaced because the rungs are geometric), so
the eye reads them as one story.

Colors are the repo's existing categorical slots, validated with the dataviz
palette validator on the light surface (CVD ΔE 24.7 worst adjacent pair, normal
vision 33.6, contrast >= 3:1). Identity is carried by a legend AND by direct
labels at the last point, never by color alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "artifacts/stage3/e1_consolidated.json"
OUT = REPO_ROOT / "assets/e1_scaling.svg"

PCA = "#2a78d6"       # categorical slot 1
RAND = "#eb6834"      # categorical slot 2
CONTROL = "#52514e"   # the compute control is chrome, not a third identity
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"


def human_tokens(v: float, _pos: int | None = None) -> str:
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def series(rows, init, key):
    """Seed-averaged curve plus the individual seed points."""
    pts, seeds = [], []
    for toks in sorted({r["tokens"] for r in rows if not r["control"]}):
        vals = [r[key] for r in rows
                if not r["control"] and r["init"] == init
                and r["tokens"] == toks and isinstance(r.get(key), float)]
        if vals:
            pts.append((toks, sum(vals) / len(vals)))
            seeds += [(toks, v) for v in vals]
    return pts, seeds


def draw(ax, rows, key, title, ylabel, label_fmt, ctl_offset=(12, 14)):
    for init, color, name in ((("pca"), PCA, "PCA/sandwich init"),
                              (("rand"), RAND, "random init")):
        pts, seeds = series(rows, init, key)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # individual seeds, so the between-seed spread is visible rather than
        # hidden behind the mean
        ax.scatter([s[0] for s in seeds], [s[1] for s in seeds], s=18,
                   color=color, alpha=0.35, edgecolors="none", zorder=3)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=4, label=name)
        ax.annotate(label_fmt(ys[-1]), (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=8, color=INK, fontweight="bold")

    ctl = next((r for r in rows if r["control"] and isinstance(r.get(key), float)), None)
    if ctl:
        ax.scatter([ctl["tokens"]], [ctl[key]], s=52, marker="D",
                   color=SURFACE, edgecolors=CONTROL, linewidths=1.8, zorder=5)
        ax.annotate("step-matched\ncontrol", (ctl["tokens"], ctl[key]),
                    textcoords="offset points", xytext=ctl_offset, fontsize=7,
                    color=INK_SOFT, linespacing=1.3)

    ax.set_xscale("log")
    # Tick at the rungs themselves. Left to itself a log axis labels its minor
    # ticks in scientific notation, which collided with the rung labels.
    rungs = sorted({r["tokens"] for r in rows if not r["control"]})
    ax.set_xticks(rungs)
    ax.xaxis.set_major_formatter(FuncFormatter(human_tokens))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_xlabel("supervised tokens trained on (log)", fontsize=8, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_SOFT)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    style(ax)


def main() -> None:
    rows = json.loads(DATA.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=SURFACE)

    draw(axes[0], rows, "val_ce",
         "Teacher-native held-out CE", "cross-entropy (nats)",
         lambda v: f"{v:.2f}", ctl_offset=(14, 16))
    draw(axes[1], rows, "nat_term",
         "Natural termination (uncapped generation)", "rate",
         lambda v: f"{v:.2f}", ctl_offset=(14, -20))
    axes[1].set_ylim(-0.03, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8, labelcolor=INK_SOFT, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Experiment 1 — recovery-data scaling, 6 rungs x 2 seeds x 2 initializations",
                 fontsize=11, color=INK, x=0.005, ha="left", y=1.0)
    fig.text(0.005, 0.925,
             "Uncapped generation within an effective context of 8,192 derived from the trained block_len. "
             "Faint dots are individual seeds.",
             fontsize=7.5, color=INK_SOFT, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
