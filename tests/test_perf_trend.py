"""Tests for the README trend figure (scripts/plot_perf_trend.py).

The figure is a public artifact (AGENTS.md P7), so the parts that decide what it
claims are tested: the committed data file stays loadable and log-backed, the
lineage it draws resolves to a real tree with accumulated step counts, the detail
panel picks its window from the data, and the generated run table matches the
plotted points. Rendering is exercised once to catch matplotlib breakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import plot_perf_trend as pt


def attempt(nll: float, run: str = "r", *, id: str = "", parent: str | None = None,
            steps: int = 0) -> dict:
    return {"id": id or run, "parent": parent, "steps": steps,
            "date": "2026-01-01", "stage": "stage3", "run": run,
            "summary": "s", "nll": nll, "log": "logs/experiments/x.md"}


def test_committed_data_is_loadable_and_log_backed():
    data = pt.load()
    assert data["attempts"], "trend figure must have at least one attempt"
    for entry in data["attempts"] + data["references"]:
        assert (REPO_ROOT / entry["log"]).exists(), f"missing log for {entry}"
        assert isinstance(entry["nll"], float)
    # Layout must not leak back into the data file.
    assert not any("label_offset" in a for a in data["attempts"])


def test_committed_lineage_resolves_and_accumulates_steps():
    """Every committed attempt names a real start checkpoint (or none)."""
    nodes = pt.lineage(pt.load()["attempts"])
    for node in nodes:
        parent = node["parent_index"]
        if parent is None:
            assert node["total_steps"] == node["steps"]
        else:
            assert parent < nodes.index(node), "a start checkpoint must exist first"
            assert node["total_steps"] == nodes[parent]["total_steps"] + node["steps"]


def test_lineage_numbers_attempts_and_sums_the_branch():
    nodes = pt.lineage([
        attempt(9.0, "init", id="init"),
        attempt(4.0, "leg a", id="a", parent="init", steps=660),
        attempt(3.5, "leg b", id="b", parent="a", steps=2700),
        attempt(3.6, "branch", id="c", parent="init", steps=2700),
    ])
    assert [n["n"] for n in nodes] == [1, 2, 3, 4]
    assert [n["total_steps"] for n in nodes] == [0, 660, 3360, 2700]
    assert [n["parent_index"] for n in nodes] == [None, 0, 1, 0]


def test_lineage_fails_loudly_on_a_broken_start_checkpoint():
    with pytest.raises(ValueError, match="not an earlier attempt"):
        pt.lineage([attempt(4.0, "child", id="c", parent="ghost")])
    # A parent defined later would allow a cycle; only earlier ids resolve.
    with pytest.raises(ValueError, match="not an earlier attempt"):
        pt.lineage([attempt(4.0, "child", id="c", parent="p"), attempt(9.0, "p", id="p")])
    with pytest.raises(ValueError, match="duplicate"):
        pt.lineage([attempt(9.0, "x", id="x"), attempt(4.0, "y", id="x")])


def test_detail_window_brackets_the_cluster():
    nodes = pt.lineage([attempt(17.8, "a", id="a"), attempt(11.7, "b", id="b"),
                        attempt(4.21, "c", id="c"), attempt(3.80, "d", id="d")])
    idx, lo, hi = pt.detail_window(nodes)
    assert idx == [2, 3]
    # Tight around the cluster — a window stretched to the teacher line would
    # squeeze sibling branches into one blob.
    assert lo < 3.80 and hi > 4.21
    assert hi - lo < 2 * (4.21 - 3.80)


def test_detail_window_is_skipped_when_it_would_add_nothing():
    # All attempts already inside the cluster -> the overview shows everything.
    assert pt.detail_window(pt.lineage([attempt(4.0, "a", id="a"),
                                        attempt(3.9, "b", id="b")])) is None
    # Only one clustered attempt -> nothing to compare in a detail panel.
    assert pt.detail_window(pt.lineage([attempt(17.8, "a", id="a"), attempt(11.7, "b", id="b"),
                                        attempt(3.8, "c", id="c")])) is None


def test_span_label_collapses_only_contiguous_runs():
    assert pt.span_label([3, 4, 5]) == "3–5"
    assert pt.span_label([3, 5, 8]) == "3, 5, 8"


def test_markdown_table_matches_the_plotted_points():
    data = pt.load()
    nodes = pt.lineage(data["attempts"])
    table = pt.markdown_table(data)
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(rows) == len(data["attempts"]) + 2  # header + separator
    best = min(a["nll"] for a in data["attempts"])
    assert f"**{best:.4f}**" in table
    for node in nodes:
        assert node["run"] in table and node["log"].replace("logs/", "./logs/") in table
        # The lineage the figure draws is legible in the table too.
        parent = node["parent_index"]
        expected = "—" if parent is None else f"#{nodes[parent]['n']}"
        assert f"| {expected} | {node['summary']} | {node['total_steps']} |" in table


def test_readme_table_is_the_generated_one():
    """The README table is generated output — regenerate it, don't hand-edit."""
    readme = (REPO_ROOT / "README.md").read_text()
    for row in pt.markdown_table(pt.load()).splitlines():
        if row.startswith("| ") and not row.startswith("| ---"):
            assert row in readme, f"README is out of date, missing row:\n{row}"


def test_render_writes_an_svg(tmp_path):
    out = pt.render(pt.load(), tmp_path / "trend.svg")
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_text().lstrip().startswith("<?xml")
