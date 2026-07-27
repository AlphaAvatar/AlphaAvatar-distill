"""Render the README performance-trend figure from `assets/perf_trend.json`.

Usage:
    uv run python scripts/plot_perf_trend.py                 # write the SVG
    uv run python scripts/plot_perf_trend.py --print-table   # markdown run table

Data in, layout out. `perf_trend.json` holds only facts — date, run name, the
checkpoint the run started from, the optimizer steps that run added, a one-line
summary, the metric value, and the experiment log that backs it (AGENTS.md P7).
Everything about placement is computed here. Adding an attempt is a pure data
edit, and the figure grows a row rather than getting more crowded.

**One row per run.** The runs form a tree, not a queue, and the metric spans 14
nats end to end while the interesting differences are 0.03 — no single scatter
can hold both, which is what made the earlier steps-vs-NLL version unreadable.
Rows fix it: every run gets its own line, so nothing overlaps and nothing needs a
zoom panel.

* left — **lineage and cost**: x is the cumulative optimizer steps of the run's
  whole lineage, and each run hangs off its start checkpoint by a git-graph
  elbow. Sibling branches leave a shared parent instead of queueing up behind
  each other.
* right — **the metric**: an arrow per run, from its start checkpoint's held-out
  NLL to what the run reached, against the teacher and random-init rules. The
  arrow is the run's effect; the tail is where it started. Log axis, because the
  init points are ~4× the recovered ones; exact values are in the aligned column
  on the right, so the axis carries magnitude and the column carries precision.

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
from matplotlib.ticker import ScalarFormatter

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

ROW_IN = 0.40  # figure inches per attempt row
CHROME_IN = 1.75  # title, panel headers, axis band


def load(path: Path = DATA) -> dict:
    return json.loads(path.read_text())


def lineage(attempts: list[dict]) -> list[dict]:
    """Resolve `parent` ids into indices and accumulate optimizer steps.

    A parent must be an attempt defined earlier in the file, which makes the
    result a tree by construction. Fails loudly on an unknown or duplicate id
    rather than silently dropping an edge.
    """
    nodes: list[dict] = []
    index: dict[str, int] = {}
    for i, a in enumerate(attempts):
        if a["id"] in index:
            raise ValueError(f"duplicate attempt id {a['id']!r}")
        parent = a.get("parent")
        if parent is None:
            parent_index, base = None, 0
        elif parent in index:
            parent_index = index[parent]
            base = nodes[parent_index]["total_steps"]
        else:
            raise ValueError(f"attempt {a['id']!r} starts from {parent!r}, which is "
                             "not an earlier attempt in the file")
        index[a["id"]] = i
        nodes.append({**a, "n": i + 1, "parent_index": parent_index,
                      "total_steps": base + a["steps"]})
    return nodes


def log_ticks(lo: float, hi: float) -> list[float]:
    """Readable ticks for a log axis over a small range: 1-2-3-5 per decade."""
    candidates = [c * m for m in (0.1, 1, 10, 100) for c in (1, 2, 3, 4, 5, 7)]
    return [c for c in sorted(candidates) if lo <= c <= hi]


def style_row_axes(ax, n_rows: int) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylim(n_rows - 0.45, -1.25)  # inverted: attempt 1 on top; headroom on top
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([])
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=8.5, length=3, width=0.9)
    ax.tick_params(axis="y", length=0)


def draw_lineage_panel(ax, nodes: list[dict]) -> None:
    """Git-graph elbows: a run drops out of its start checkpoint, then extends
    right by what its own leg cost."""
    xs = [node["total_steps"] for node in nodes]
    span = max(xs) or 1
    ax.set_xlim(-span * 0.09, span * 1.09)
    ax.set_xticks(sorted(set(xs)))

    for i, node in enumerate(nodes):
        parent = node["parent_index"]
        if parent is None:
            continue
        px = nodes[parent]["total_steps"]
        ax.plot([px, px, node["total_steps"]], [parent + 0.24, i, i],
                color=STUDENT, alpha=0.45, linewidth=1.6, zorder=2,
                solid_joinstyle="round", solid_capstyle="round")

    ax.scatter(xs, range(len(nodes)), s=150, color=STUDENT, zorder=3,
               linewidths=1.6, edgecolors=SURFACE)
    for i, x in enumerate(xs):
        ax.annotate(str(i + 1), xy=(x, i), ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=4)


def draw_metric_panel(ax, nodes: list[dict], refs: list[dict]) -> None:
    """One arrow per run: start checkpoint's metric → what the run reached."""
    values = [node["nll"] for node in nodes] + [ref["nll"] for ref in refs]
    lo, hi = min(values) * 0.86, max(values) * 1.10
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(log_ticks(lo, hi))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()

    for ref in refs:
        color = TARGET if ref.get("role") == "target" else BASELINE
        ax.axvline(ref["nll"], color=color, linestyle=(0, (6, 4)), linewidth=1.3, zorder=1)
        # Reference lines sit near the panel edges, so labels hang inwards —
        # centering them would push text outside the panel.
        inward = ref["nll"] < (lo * hi) ** 0.5
        ax.annotate(f"{ref.get('short', ref['label'])}  {ref['nll']:.2f}",
                    xy=(ref["nll"], -1.15),
                    xytext=(4 if inward else -4, 0), textcoords="offset points",
                    ha="left" if inward else "right", va="bottom",
                    fontsize=8, color=color)

    best = min(node["nll"] for node in nodes)
    for i, node in enumerate(nodes):
        parent = node["parent_index"]
        if parent is not None:
            ax.annotate("", xy=(node["nll"], i), xytext=(nodes[parent]["nll"], i),
                        arrowprops=dict(arrowstyle="-|>", color=STUDENT, alpha=0.55,
                                        linewidth=1.5, shrinkA=2, shrinkB=8,
                                        mutation_scale=11), zorder=2)
        ax.scatter([node["nll"]], [i], s=110, color=STUDENT, zorder=3,
                   linewidths=1.6, edgecolors=SURFACE)
        # Values in an aligned column outside the plot: the axis carries
        # magnitude, the column carries the precision the near-ties need.
        is_best = node["nll"] == best
        ax.annotate(f"{node['nll']:.3f}", xy=(1.0, i), xycoords=("axes fraction", "data"),
                    xytext=(48, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8.5, color=INK if is_best else INK_SOFT,
                    fontweight="bold" if is_best else "normal", annotation_clip=False)
    ax.annotate("NLL", xy=(1.0, -1.15), xycoords=("axes fraction", "data"),
                xytext=(48, 0), textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=INK_SOFT, annotation_clip=False)


def render(data: dict, out: Path = OUT) -> Path:
    nodes, refs = lineage(data["attempts"]), data["references"]
    n = len(nodes)

    height = CHROME_IN + ROW_IN * n
    fig = plt.figure(figsize=(10.0, height), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.075,
                            left=0.035, right=0.925,
                            top=1 - 0.92 / height, bottom=0.62 / height)
    ax_tree = fig.add_subplot(grid[0, 0])
    ax_metric = fig.add_subplot(grid[0, 1], sharey=ax_tree)
    style_row_axes(ax_tree, n)
    style_row_axes(ax_metric, n)

    draw_lineage_panel(ax_tree, nodes)
    draw_metric_panel(ax_metric, nodes, refs)

    ax_tree.set_xlabel(data.get("steps_axis", "cumulative optimizer steps"),
                       fontsize=8.5, color=INK_SOFT)
    ax_metric.set_xlabel(data.get("metric_axis", "held-out NLL"),
                         fontsize=8.5, color=INK_SOFT)

    fig.suptitle(data.get("title", "held-out NLL by lineage"), fontsize=12,
                 color=INK, x=0.035, ha="left", y=1 - 0.24 / height)
    fig.text(0.035, 1 - 0.52 / height, data["metric"], fontsize=8.5, color=INK_SOFT,
             ha="left", va="top")
    fig.text(0.985, 0.055 / height,
             "each run hangs off the checkpoint it started from · arrows run from "
             "that checkpoint's score to the run's own · numbers key into the run "
             "table in README.md",
             ha="right", fontsize=7, color="#9b9a95")

    fig.savefig(out, facecolor=SURFACE)  # format follows the suffix (.svg / .png preview)
    plt.close(fig)
    return out


def markdown_table(data: dict) -> str:
    """The README run table, generated from the same facts as the figure."""
    nodes = lineage(data["attempts"])
    best = min(n["nll"] for n in nodes)
    rows = ["| # | date | run | starts from | what changed | total steps | held-out NLL |",
            "| ---: | --- | --- | :---: | --- | ---: | ---: |"]
    for node in nodes:
        nll = f"**{node['nll']:.4f}**" if node["nll"] == best else f"{node['nll']:.4f}"
        parent = "—" if node["parent_index"] is None else f"#{nodes[node['parent_index']]['n']}"
        rows.append(f"| {node['n']} | {node['date']} "
                    f"| [{node['run']}]({node['log'].replace('logs/', './logs/')}) "
                    f"| {parent} | {node['summary']} | {node['total_steps']} | {nll} |")
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
