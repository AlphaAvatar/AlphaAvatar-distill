"""Tests for the README trend figure (scripts/plot_perf_trend.py).

The figure is a public artifact (AGENTS.md P7), so the parts that decide what it
claims are tested: the committed data file stays loadable and log-backed, the
detail panel picks its window from the data, and the generated run table matches
the plotted points. Rendering is exercised once to catch matplotlib breakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import plot_perf_trend as pt


def attempt(nll: float, run: str = "r") -> dict:
    return {"date": "2026-01-01", "stage": "stage3", "run": run,
            "summary": "s", "nll": nll, "log": "logs/experiments/x.md"}


TEACHER = [{"label": "teacher", "role": "target", "nll": 2.0, "log": "logs/experiments/x.md"}]


def test_committed_data_is_loadable_and_log_backed():
    data = pt.load()
    assert data["attempts"], "trend figure must have at least one attempt"
    for entry in data["attempts"] + data["references"]:
        assert (REPO_ROOT / entry["log"]).exists(), f"missing log for {entry}"
        assert isinstance(entry["nll"], float)
    # Layout must not leak back into the data file.
    assert not any("label_offset" in a for a in data["attempts"])


def test_detail_window_selects_the_cluster():
    attempts = [attempt(17.8), attempt(11.7), attempt(4.21), attempt(3.80)]
    idx, lo, hi = pt.detail_window(attempts, TEACHER)
    assert idx == [2, 3]
    assert lo < 2.0 < hi and hi > 4.21


def test_detail_window_is_skipped_when_it_would_add_nothing():
    # All attempts already inside the cluster -> the overview shows everything.
    assert pt.detail_window([attempt(4.0), attempt(3.9)], TEACHER) is None
    # Only one clustered attempt -> nothing to compare in a detail panel.
    assert pt.detail_window([attempt(17.8), attempt(11.7), attempt(3.8)], TEACHER) is None


def test_markdown_table_matches_the_plotted_points():
    data = pt.load()
    table = pt.markdown_table(data)
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(rows) == len(data["attempts"]) + 2  # header + separator
    best = min(a["nll"] for a in data["attempts"])
    assert f"**{best:.4f}**" in table
    for a in data["attempts"]:
        assert a["run"] in table and a["log"].replace("logs/", "./logs/") in table


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
