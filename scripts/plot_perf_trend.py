"""Render the README performance-trend figure from `assets/perf_trend.json`.

Usage:
    uv run python scripts/plot_perf_trend.py                 # write the SVG
    uv run python scripts/plot_perf_trend.py --print-table   # markdown run table

Data in, layout out. `perf_trend.json` holds only facts — date, run name, the
checkpoint the run started from, the optimizer steps that run added, a one-line
summary, the metric value, and the experiment log that backs it (AGENTS.md P7).
Everything about placement is computed here, and no point carries a hand-tuned
label offset. Adding an attempt is a pure data edit.

The figure is a **lineage** plot, not a chronological one. A run sits at the
cumulative optimizer steps of its whole lineage (its start checkpoint's total
plus its own leg), and a line joins it to the checkpoint it started from.
Sibling branches — the two A/B arms, the two start-point ablation arms — leave
their common parent instead of queueing up behind each other, which is what the
runs actually did; reading them as one sequence would imply progress that was
never measured. The lines are lineage edges, not training curves: only the
endpoints are evaluated.

Two consequences of plotting real lineage, both handled by rule rather than by
hand: attempts can share an x (same total steps), so markers that land within a
marker width of each other are drawn once and share a joined label, and value
labels in the detail panel flip below their marker when a neighbour occupies the
space above.

The figure deliberately carries no per-point prose: the numbered markers key
into the run table in `README.md`, which `--print-table` regenerates from the
same file so the two cannot drift.

Output: `assets/performance_trend.svg` (committed; small, reviewable, opaque
light surface so it reads the same in GitHub's light and dark themes).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def detail_window(nodes: list[dict]) -> tuple[list[int], float, float] | None:
    """Indices (0-based) of the clustered attempts plus the y-window to show.

    The window brackets the cluster itself rather than stretching down to the
    teacher line: the runs it has to separate differ by well under 1%, and a
    window tall enough to reach the teacher squeezes sibling branches into one
    blob. The overview keeps the teacher in view.

    Returns None when a detail panel would not add anything: fewer than two
    clustered attempts, or every attempt already inside the cluster.
    """
    values = [n["nll"] for n in nodes]
    best = min(values)
    idx = [i for i, v in enumerate(values) if v <= best * DETAIL_FACTOR]
    if len(idx) < 2 or len(idx) == len(nodes):
        return None
    top = max(values[i] for i in idx)
    span = (top - best) or max(best * 0.02, 0.01)
    return idx, best - span * 0.35, top + span * 0.18


def clusters(ax, nodes: list[dict], idx: list[int], min_px: float) -> list[list[int]]:
    """Group the given attempts whose markers land within `min_px` on screen.

    Lineage puts sibling runs at the same x, so overlap is a property of the
    data, not of a bad hand-placement. Both panels resolve it from this
    grouping; it needs the axes limits to be final.
    """
    pts = dict(zip(idx, ax.transData.transform(
        [(nodes[i]["total_steps"], nodes[i]["nll"]) for i in idx])))
    groups: list[list[int]] = []
    for i in idx:
        x, y = pts[i]
        for group in groups:
            gx, gy = pts[group[0]]
            if math.hypot(x - gx, y - gy) < min_px:
                group.append(i)
                break
        else:
            groups.append([i])
    return groups


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


def draw_lineage(ax, nodes: list[dict], idx: list[int], *, size: float = 150,
                 merge_px: float = 0.0) -> None:
    """Lineage edges plus one numbered marker per checkpoint.

    An edge is drawn whenever the *child* is shown, even if its parent is not:
    in the detail panel that leaves a line entering from off-panel, which is the
    honest picture of a run that branched from a much worse checkpoint. Those
    edges recede, since only one of their endpoints is on screen.
    """
    for i in idx:
        parent = nodes[i]["parent_index"]
        if parent is None:
            continue
        ax.plot([nodes[parent]["total_steps"], nodes[i]["total_steps"]],
                [nodes[parent]["nll"], nodes[i]["nll"]],
                color=STUDENT, linewidth=1.6, zorder=2, solid_capstyle="round",
                alpha=0.5 if parent in idx else 0.25)

    groups = clusters(ax, nodes, idx, merge_px) if merge_px else [[i] for i in idx]
    for group in groups:
        x = sum(nodes[i]["total_steps"] for i in group) / len(group)
        y = sum(nodes[i]["nll"] for i in group) / len(group)
        ax.scatter([x], [y], s=size * (1 + 0.22 * (len(group) - 1)), color=STUDENT,
                   zorder=3, linewidths=1.6, edgecolors=SURFACE)
        ax.annotate("·".join(str(nodes[i]["n"]) for i in group), xy=(x, y),
                    ha="center", va="center", fontsize=7.5 if len(group) == 1 else 6.2,
                    color="white", fontweight="bold", zorder=4)


def draw_reference(ax, ref: dict, *, label: bool) -> None:
    color = TARGET if ref.get("role") == "target" else BASELINE
    ax.axhline(ref["nll"], color=color, linestyle=(0, (6, 4)), linewidth=1.3, zorder=1)
    if label:
        ax.annotate(f"{ref['label']}  ({ref['nll']:.2f})",
                    xy=(0.015, ref["nll"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=8.5, color=color)


def span_label(numbers: list[int]) -> str:
    """`3–8` when the shown attempts are contiguous, `3, 5, 8` when they are not."""
    if numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"{numbers[0]}–{numbers[-1]}"
    return ", ".join(str(n) for n in numbers)


def axis_window(values: list[float], pad_fraction: float) -> tuple[float, float]:
    pad = (max(values) - min(values)) * pad_fraction or 1.0
    return min(values) - pad, max(values) + pad


def render(data: dict, out: Path = OUT) -> Path:
    nodes, refs = lineage(data["attempts"]), data["references"]
    xs = [n["total_steps"] for n in nodes]
    ys = [n["nll"] for n in nodes]
    detail = detail_window(nodes)

    fig = plt.figure(figsize=(10.0, 4.4), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(1, 2 if detail else 1, width_ratios=[1.85, 1] if detail else [1],
                            left=0.065, right=0.985, top=0.84, bottom=0.135, wspace=0.16)
    ax = fig.add_subplot(grid[0, 0])
    style_axes(ax)

    # Limits first: the marker grouping is computed in screen space.
    ax.set_xticks(sorted(set(xs)))
    ax.set_xlim(*axis_window(xs, 0.06))
    ax.set_ylim(0, max(ys) * 1.1)
    for ref in refs:
        draw_reference(ax, ref, label=True)
    draw_lineage(ax, nodes, list(range(len(nodes))), merge_px=15.0)

    ax.set_xlabel(data.get("steps_axis", "cumulative optimizer steps"),
                  fontsize=9, color=INK_SOFT)
    ax.set_ylabel(data.get("metric_axis", "held-out NLL"), fontsize=9, color=INK_SOFT)
    legend = ax.legend(handles=[
        Line2D([], [], color=STUDENT, linewidth=1.6, alpha=0.6,
               label="training leg, from its start checkpoint"),
        Line2D([], [], color=STUDENT, marker="o", markersize=9, linestyle="none",
               markeredgecolor=SURFACE, markeredgewidth=1.4, label="student checkpoint"),
    ], loc="upper right", fontsize=8.5, framealpha=1.0, facecolor=SURFACE, edgecolor=GRID)
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
        ax_d.set_xticks(sorted(set(dxs)))
        ax_d.set_xlim(*axis_window(dxs, 0.08))
        ax_d.set_ylim(lo, hi)
        ax_d.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        for ref in refs:
            if lo <= ref["nll"] <= hi:
                draw_reference(ax_d, ref, label=False)
                ax_d.annotate(f"teacher {ref['nll']:.2f}" if ref.get("role") == "target"
                              else f"{ref['label']} {ref['nll']:.2f}",
                              xy=(0.02, ref["nll"]), xycoords=("axes fraction", "data"),
                              xytext=(0, 4), textcoords="offset points", fontsize=8,
                              color=TARGET if ref.get("role") == "target" else BASELINE)
        draw_lineage(ax_d, nodes, idx, size=120)

        # Values only in the detail panel, where there is room for all of them.
        # Within a crowded group only the topmost keeps the space above its
        # marker; the rest label downwards.
        best = min(dys)
        for group in clusters(ax_d, nodes, idx, 36.0):
            for rank, i in enumerate(sorted(group, key=lambda j: -nodes[j]["nll"])):
                y = nodes[i]["nll"]
                ax_d.annotate(f"{y:.3f}", xy=(nodes[i]["total_steps"], y),
                              xytext=(0, 9 if rank == 0 else -9), textcoords="offset points",
                              ha="center", va="bottom" if rank == 0 else "top",
                              fontsize=8.5, color=INK if y == best else INK_SOFT,
                              fontweight="bold" if y == best else "normal")
        target = next((r for r in refs if r.get("role") == "target"), None)
        off_window = target and not lo <= target["nll"] <= hi
        ax_d.set_title(f"detail: attempts {span_label([nodes[i]['n'] for i in idx])}"
                       + (f"\n(teacher {target['nll']:.2f} sits below this window)"
                          if off_window else " vs the teacher line"),
                       fontsize=9, color=INK_SOFT, pad=6, linespacing=1.5)

        # Tie the shaded band to the detail panel so the zoom is unambiguous.
        for y_band in (lo, hi):
            fig.add_artist(ConnectionPatch(
                xyA=(ax.get_xlim()[1], y_band), coordsA=ax.transData,
                xyB=(ax_d.get_xlim()[0], y_band), coordsB=ax_d.transData,
                color=STUDENT, alpha=0.25, linewidth=0.8))

    fig.suptitle(data.get("title", "held-out NLL by lineage"), fontsize=12,
                 color=INK, x=0.065, ha="left", y=0.955)
    fig.text(0.065, 0.9, data["metric"], fontsize=8.5, color=INK_SOFT, ha="left")
    fig.text(0.985, 0.012,
             "lines are lineage, not training curves · numbers key into the run table "
             "in README.md · every point is backed by a log in logs/experiments/",
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
