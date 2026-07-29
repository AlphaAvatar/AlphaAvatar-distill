"""Render the README performance figure from `assets/perf_trend.json`.

Usage:
    uv run python scripts/evaluation/plot_perf_trend.py                 # write the SVG
    uv run python scripts/evaluation/plot_perf_trend.py --print-table   # markdown run table

Data in, layout out. `perf_trend.json` holds only facts — measured scores, the
checkpoint and log behind each, parameter counts, and the run history
(AGENTS.md P7). Everything about placement is computed here.

**One point per student, at its current best** (quality vs size, the shape the
ARC-AGI leaderboard uses). Listing every checkpoint was the wrong frame: what a
reader wants is where each distilled model stands now and how far it is from its
teacher. Earlier checkpoints stay in the README run table; only the previous
*best* appears here, as a faded dot with an arrow to the current one, so the
figure shows direction without turning into a run log.

The y metric is whatever `headline` in the data file names — today
`behavior_score_v0`, six mechanical axes over 76 held-out prompts, because a
0.6B student this early cannot meaningfully attempt real test suites. When it
can, the headline block changes and the same code plots the new metric; held-out
NLL is demoted to the guard rail it always was.

The x axis is parameters, standing in for inference cost until Stage 6 measures
latency and memory. A reference with no score yet (the teacher has never been
run on this eval) is drawn as its size line only — never as a guessed y.

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
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "assets/perf_trend.json"
OUT = REPO_ROOT / "assets/performance_trend.svg"

# Validated with the dataviz palette validator (light surface #fcfcfb):
# the two identity hues pass every check; the grays below are chrome/ink, not
# categorical slots.
STUDENT = "#2a78d6"  # categorical slot 1 — the student series
TARGET = "#008300"  # the teacher line: a target threshold, not a series
BASELINE = "#8b8a85"  # recessive chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"


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


def human_params(value: float, _pos: int | None = None) -> str:
    """0.6B / 1B / 4B — the units people actually compare models in."""
    billions = value / 1e9
    return f"{billions:.1f}B".replace(".0B", "B")


def size_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 style ticks per decade, in parameters."""
    candidates = [c * 10 ** e for e in range(6, 13) for c in (1, 2, 5)]
    return [c for c in candidates if lo <= c <= hi]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=0.9)


def draw_reference(ax, ref: dict, hi: float) -> None:
    """A reference model is a size line; its score appears only once measured."""
    color = TARGET if ref.get("role") == "target" else BASELINE
    ax.axvline(ref["params"], color=color, linestyle=(0, (6, 4)), linewidth=1.3, zorder=1)
    if ref.get("score") is not None:
        ax.scatter([ref["params"]], [ref["score"]], s=200, color=color, zorder=3,
                   linewidths=1.6, edgecolors=SURFACE)
    label = f"{ref['label']}  ({human_params(ref['params'])})"
    if ref.get("note"):
        label += f"\n{ref['note']}"
    ax.annotate(label, xy=(ref["params"], hi), xytext=(-8, -6), textcoords="offset points",
                ha="right", va="top", fontsize=8.5, color=color, linespacing=1.5)


def draw_system(ax, system: dict) -> None:
    """Current best, plus an arrow from the previous best — direction, not a log."""
    x, best = system["params"], system["best"]
    previous = system.get("previous_best")

    if previous:
        ax.scatter([x], [previous["score"]], s=110, color=STUDENT, alpha=0.32,
                   zorder=2, linewidths=1.4, edgecolors=SURFACE)
        ax.annotate(f"{previous['score']:.1%} · {previous['checkpoint']} · {previous['date']}",
                    xy=(x, previous["score"]), xytext=(15, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=INK_SOFT)
        ax.annotate("", xy=(x, best["score"]), xytext=(x, previous["score"]),
                    arrowprops=dict(arrowstyle="-|>", color=STUDENT, alpha=0.5,
                                    linewidth=1.6, shrinkA=9, shrinkB=13,
                                    mutation_scale=13), zorder=2)

    ax.scatter([x], [best["score"]], s=260, color=STUDENT, zorder=4,
               linewidths=1.8, edgecolors=SURFACE)
    ax.annotate(f"{system['label']} · {system['stage']}", xy=(x, best["score"]),
                xytext=(17, 5), textcoords="offset points", ha="left", va="bottom",
                fontsize=10.5, color=INK, fontweight="bold")
    ax.annotate(f"{best['score']:.1%} · {best['checkpoint']} · {best['date']}",
                xy=(x, best["score"]), xytext=(17, -7), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=INK_SOFT)


def draw_compression_bracket(ax, system: dict, ref: dict, y: float) -> None:
    """The distance between the two size lines is the whole point of the project."""
    ax.annotate("", xy=(ref["params"], y), xytext=(system["params"], y),
                arrowprops=dict(arrowstyle="<|-|>", color=BASELINE, linewidth=1.0,
                                shrinkA=1, shrinkB=1, mutation_scale=9), zorder=2)
    ax.annotate(f"{ref['params'] / system['params']:.1f}× fewer parameters",
                xy=((system["params"] * ref["params"]) ** 0.5, y), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.5, color=INK_SOFT)


def render(data: dict, out: Path = OUT) -> Path:
    systems, refs = data["systems"], data["references"]
    headline, size_axis = data["headline"], data["size_axis"]

    scored = [s["best"]["score"] for s in systems]
    scored += [s["previous_best"]["score"] for s in systems if s.get("previous_best")]
    scored += [r["score"] for r in refs if r.get("score") is not None]
    # Headroom for the top label, but never past 100% — these are rates.
    hi = min(1.0, max(scored) * 1.3)
    sizes = [s["params"] for s in systems] + [r["params"] for r in refs]

    fig = plt.figure(figsize=(10.0, 5.0), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0.088, 0.145, 0.899, 0.665))
    style_axes(ax)

    ax.set_xscale("log")
    ax.set_xlim(min(sizes) / 1.4, max(sizes) * 1.4)
    ax.set_xticks(size_ticks(*ax.get_xlim()))
    ax.xaxis.set_major_formatter(FuncFormatter(human_params))
    ax.minorticks_off()
    ax.set_ylim(0, hi)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    for ref in refs:
        draw_reference(ax, ref, hi)
    for system in systems:
        draw_system(ax, system)
    target = next((r for r in refs if r.get("role") == "target"), None)
    if target and systems:
        # Low enough to clear the lowest system label above it.
        draw_compression_bracket(ax, systems[0], target, hi * 0.035)

    ax.set_xlabel(f"{size_axis['label']} — {size_axis['note']}", fontsize=9, color=INK_SOFT)
    ax.set_ylabel(headline["axis"], fontsize=9, color=INK_SOFT)

    fig.suptitle(data.get("title", "behavior score vs model size"), fontsize=12.5,
                 color=INK, x=0.088, ha="left", y=0.955)
    fig.text(0.088, 0.895, headline["summary"], fontsize=8.5, color=INK_SOFT,
             ha="left", va="top")
    fig.text(0.088, 0.855, headline["note"], fontsize=8.5, color=INK_SOFT,
             ha="left", va="top", style="italic")
    fig.text(0.987, 0.03,
             "one point per student at its current best · every run, including the "
             "ones that did not improve, is in the README table",
             ha="right", fontsize=7, color="#9b9a95")

    fig.savefig(out, facecolor=SURFACE)  # format follows the suffix (.svg / .png preview)
    plt.close(fig)
    return out


def markdown_table(data: dict) -> str:
    """The README run table, generated from the same facts as the figure."""
    nodes = lineage(data["attempts"])
    best_nll = min(n["nll"] for n in nodes)
    best_behavior = max((n["behavior"] for n in nodes if "behavior" in n), default=None)
    rows = ["| # | date | run | starts from | what changed | total steps | behavior | held-out NLL |",
            "| ---: | --- | --- | :---: | --- | ---: | ---: | ---: |"]
    for node in nodes:
        nll = f"**{node['nll']:.4f}**" if node["nll"] == best_nll else f"{node['nll']:.4f}"
        behavior = "–"
        if "behavior" in node:
            behavior = (f"**{node['behavior']:.1%}**" if node["behavior"] == best_behavior
                        else f"{node['behavior']:.1%}")
        parent = "—" if node["parent_index"] is None else f"#{nodes[node['parent_index']]['n']}"
        rows.append(f"| {node['n']} | {node['date']} "
                    f"| [{node['run']}]({node['log'].replace('logs/', './logs/')}) "
                    f"| {parent} | {node['summary']} | {node['total_steps']} "
                    f"| {behavior} | {nll} |")
    guard = " · ".join(f"{r['label']} {r['nll']:.4f}"
                       for r in data["guard"].get("references", []))
    pending = " · ".join(r["label"] for r in data["references"]
                         if r.get("score") is None)
    rows.append("")
    caption = (f"Behavior score is the headline metric. **Held-out NLL is now a guard rail "
               f"({data['guard']['band']} band), not the target** — {guard}.")
    # Only mention unscored references when there are some. The teacher was the
    # last one outstanding and was scored on 2026-07-28, which left this trailing
    # an empty list.
    if pending:
        caption += f" Not scored on the behavior eval yet: {pending}."
    rows.append(caption)
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
    print(f"Wrote {out} ({out.stat().st_size} bytes, {len(data['systems'])} systems)")


if __name__ == "__main__":
    main()
