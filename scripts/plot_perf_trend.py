"""Render the README performance-trend figure from `assets/perf_trend.json`.

Usage:
    uv run python scripts/plot_perf_trend.py                 # write the SVG
    uv run python scripts/plot_perf_trend.py --print-table   # markdown run table

Data in, layout out. `perf_trend.json` holds only facts — date, run name,
one-line summary, metric value, and the experiment log that backs it
(AGENTS.md P7). Everything about placement is computed here: attempts are
numbered in chronological order, the detail panel picks its own window from the
data, and no point carries a hand-tuned label offset. Adding an attempt is a
pure data edit.

The figure deliberately carries no per-point prose: the numbered markers key
into the run table in `README.md`, which `--print-table` regenerates from the
same file so the two cannot drift.

Output: `assets/performance_trend.svg` (committed; small, reviewable, opaque
light surface so it reads the same in GitHub's light and dark themes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "assets/perf_trend.json"
OUT = REPO_ROOT / "assets/performance_trend.svg"

# Validated with the dataviz palette validator (light surface #fcfcfb):
# the two identity hues pass every check; the grays below are chrome/ink, not
# categorical slots.
STUDENT = "#2a78d6"  # categorical slot 1 — the student series
TARGET = "#008300"  # the teacher line: a target threshold, not a series
BASELINE = "#8b8a85"  # random-init reference: recessive chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"

# An attempt joins the detail panel when it is within this factor of the best
# result so far — i.e. when the overview panel can no longer separate it.
DETAIL_FACTOR = 1.35


def load(path: Path = DATA) -> dict:
    return json.loads(path.read_text())


def detail_window(attempts: list[dict], refs: list[dict]) -> tuple[list[int], float, float] | None:
    """Indices (0-based) of the clustered attempts plus the y-window to show.

    Returns None when a detail panel would not add anything: fewer than two
    clustered attempts, or every attempt already inside the cluster.
    """
    values = [a["nll"] for a in attempts]
    best = min(values)
    idx = [i for i, v in enumerate(values) if v <= best * DETAIL_FACTOR]
    if len(idx) < 2 or len(idx) == len(attempts):
        return None
    floor = min([r["nll"] for r in refs if r.get("role") == "target"] or [best])
    top = max(values[i] for i in idx)
    span = top - floor
    return idx, floor - span * 0.06, top + span * 0.15


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=3, width=0.9)


def draw_attempts(ax, xs: list[int], ys: list[float], size: float = 150,
                  best: list[float] | None = None) -> None:
    """Numbered markers + the best-so-far staircase, the only two series.

    `best` is passed in for the detail panel so its staircase stays the *global*
    running minimum rather than the minimum of the shown subset.
    """
    best = best or [min(ys[: i + 1]) for i in range(len(ys))]
    ax.plot(xs, best, color=STUDENT, linewidth=1.8, drawstyle="steps-post",
            alpha=0.45, zorder=2, label="best so far")
    ax.scatter(xs, ys, s=size, color=STUDENT, zorder=3, linewidths=1.6,
               edgecolors=SURFACE, label="student attempt")
    for x, y in zip(xs, ys):
        ax.annotate(str(x), xy=(x, y), ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=4)


def draw_reference(ax, ref: dict, *, label: bool) -> None:
    color = TARGET if ref.get("role") == "target" else BASELINE
    ax.axhline(ref["nll"], color=color, linestyle=(0, (6, 4)), linewidth=1.3, zorder=1)
    if label:
        ax.annotate(f"{ref['label']}  ({ref['nll']:.2f})",
                    xy=(0.015, ref["nll"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=8.5, color=color)


def render(data: dict, out: Path = OUT) -> Path:
    attempts, refs = data["attempts"], data["references"]
    xs = list(range(1, len(attempts) + 1))
    ys = [a["nll"] for a in attempts]
    detail = detail_window(attempts, refs)

    fig = plt.figure(figsize=(10.0, 4.4), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(1, 2 if detail else 1, width_ratios=[1.85, 1] if detail else [1],
                            left=0.065, right=0.985, top=0.84, bottom=0.135, wspace=0.16)
    ax = fig.add_subplot(grid[0, 0])
    style_axes(ax)

    for ref in refs:
        draw_reference(ax, ref, label=True)
    draw_attempts(ax, xs, ys)

    ax.set_xlabel("student checkpoint attempts (chronological)", fontsize=9, color=INK_SOFT)
    ax.set_ylabel(data.get("metric_axis", "held-out NLL"), fontsize=9, color=INK_SOFT)
    ax.set_xticks(xs)
    ax.set_xlim(0.4, len(xs) + 0.6)
    ax.set_ylim(0, max(ys) * 1.1)
    legend = ax.legend(loc="upper right", fontsize=8.5, framealpha=1.0,
                       facecolor=SURFACE, edgecolor=GRID)
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    if detail:
        idx, lo, hi = detail
        ax.axhspan(lo, hi, color=STUDENT, alpha=0.07, zorder=0)

        ax_d = fig.add_subplot(grid[0, 1])
        style_axes(ax_d)
        dxs = [xs[i] for i in idx]
        dys = [ys[i] for i in idx]
        for ref in refs:
            if lo <= ref["nll"] <= hi:
                draw_reference(ax_d, ref, label=False)
                ax_d.annotate(f"teacher {ref['nll']:.2f}" if ref.get("role") == "target"
                              else f"{ref['label']} {ref['nll']:.2f}",
                              xy=(0.02, ref["nll"]), xycoords=("axes fraction", "data"),
                              xytext=(0, 4), textcoords="offset points", fontsize=8,
                              color=TARGET if ref.get("role") == "target" else BASELINE)
        running = [min(ys[: i + 1]) for i in range(len(ys))]
        draw_attempts(ax_d, dxs, dys, size=140, best=[running[i] for i in idx])
        # Values only in the detail panel, where there is room for all of them.
        for x, y in zip(dxs, dys):
            ax_d.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 13),
                          textcoords="offset points", ha="center", fontsize=8.5,
                          color=INK if y == min(dys) else INK_SOFT,
                          fontweight="bold" if y == min(dys) else "normal")
        ax_d.set_xticks(dxs)
        ax_d.set_xlim(min(dxs) - 0.6, max(dxs) + 0.6)
        ax_d.set_ylim(lo, hi)
        ax_d.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax_d.set_title(f"detail: attempts {min(dxs)}–{max(dxs)} vs the teacher line",
                       fontsize=9, color=INK_SOFT, pad=6)

        # Tie the shaded band to the detail panel so the zoom is unambiguous.
        for y_band in (lo, hi):
            fig.add_artist(ConnectionPatch(
                xyA=(ax.get_xlim()[1], y_band), coordsA=ax.transData,
                xyB=(ax_d.get_xlim()[0], y_band), coordsB=ax_d.transData,
                color=STUDENT, alpha=0.25, linewidth=0.8))

    fig.suptitle(data.get("title", "held-out NLL by attempt"), fontsize=12,
                 color=INK, x=0.065, ha="left", y=0.955)
    fig.text(0.065, 0.9, data["metric"], fontsize=8.5, color=INK_SOFT, ha="left")
    fig.text(0.985, 0.012,
             "numbers key into the run table in README.md · every point is backed "
             "by a log in logs/experiments/",
             ha="right", fontsize=7, color="#9b9a95")

    fig.savefig(out, facecolor=SURFACE)  # format follows the suffix (.svg / .png preview)
    plt.close(fig)
    return out


def markdown_table(data: dict) -> str:
    """The README run table, generated from the same facts as the figure."""
    best = min(a["nll"] for a in data["attempts"])
    rows = ["| # | date | run | what changed | held-out NLL |",
            "| ---: | --- | --- | --- | ---: |"]
    for n, a in enumerate(data["attempts"], start=1):
        nll = f"**{a['nll']:.4f}**" if a["nll"] == best else f"{a['nll']:.4f}"
        rows.append(f"| {n} | {a['date']} | [{a['run']}]({a['log'].replace('logs/', './logs/')}) "
                    f"| {a['summary']} | {nll} |")
    refs = " · ".join(f"{r['label']} {r['nll']:.4f}" for r in data["references"])
    rows.append("")
    rows.append(f"Reference points on the same set: {refs}.")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-table", action="store_true",
                        help="print the README run table instead of rendering")
    args = parser.parse_args()

    data = load()
    if args.print_table:
        print(markdown_table(data))
        return
    out = render(data)
    print(f"Wrote {out} ({out.stat().st_size} bytes, {len(data['attempts'])} attempts)")


if __name__ == "__main__":
    main()
