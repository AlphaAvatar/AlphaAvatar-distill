"""Tests for the README performance figure (scripts/plot_perf_trend.py).

The figure is a public artifact (AGENTS.md P7), so the parts that decide what it
claims are tested: the committed data file stays loadable and log-backed, its
recorded behavior scores still match the scorecards they came from, a reference
with no measurement never acquires one, the run history resolves to a real
lineage tree, and the generated run table matches the data. Rendering is
exercised to catch matplotlib breakage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import plot_perf_trend as pt
from aadistill.behavior import behavior_score


def attempt(nll: float, run: str = "r", *, id: str = "", parent: str | None = None,
            steps: int = 0) -> dict:
    return {"id": id or run, "parent": parent, "steps": steps,
            "date": "2026-01-01", "stage": "stage3", "run": run,
            "summary": "s", "nll": nll, "log": "logs/experiments/x.md"}


def test_committed_data_is_loadable_and_log_backed():
    data = pt.load()
    assert data["systems"], "the figure must have at least one system"
    logged = data["attempts"] + [s["best"] for s in data["systems"]] + data["references"]
    for entry in logged:
        assert (REPO_ROOT / entry["log"]).exists(), f"missing log for {entry}"
    # Layout must not leak back into the data file.
    assert not any("label_offset" in a for a in data["attempts"])


def test_a_system_point_is_the_best_of_the_recorded_attempts():
    """The figure's headline point must not beat anything the history records."""
    data = pt.load()
    measured = [a["behavior"] for a in data["attempts"] if "behavior" in a]
    for system in data["systems"]:
        assert system["best"]["score"] == max(measured)
        if system.get("previous_best"):
            assert system["previous_best"]["score"] < system["best"]["score"]


def test_recorded_scores_match_their_scorecards():
    """Scores in the data file are recomputed from the stored per-sample rows.

    Scorecards are gitignored artifacts, so this checks whatever is present and
    skips when the machine has none.
    """
    cards = {
        "s1_ffn_norm_v0": "artifacts/stage3/reference_scorecards/s1_ffn_norm_v0_step660_behavior_v0.json",
        "s2_blocks_v1": "artifacts/stage3/reference_scorecards/s2_blocks_v1_step2700_behavior_v0.json",
        "s2v1_from_s1": "artifacts/stage3/s2v1_from_s1/eval_behavior_v0.json",
        "s2v1_from_init": "artifacts/stage3/s2v1_from_init/eval_behavior_v0.json",
    }
    recorded = {a["id"]: a["behavior"] for a in pt.load()["attempts"] if "behavior" in a}
    checked = 0
    for run_id, relative in cards.items():
        card = REPO_ROOT / relative
        if not card.exists():
            continue
        rows = json.loads(card.read_text())["per_sample"]
        assert behavior_score(rows)["score"] == recorded[run_id], run_id
        checked += 1
    if not checked:
        pytest.skip("no behavior scorecards on this machine (gitignored artifacts)")


def test_an_unmeasured_reference_is_never_given_a_score():
    """The teacher has not been run on this eval; nothing may invent a y for it."""
    for ref in pt.load()["references"]:
        assert "score" in ref, "a reference states its score, even when it is null"
        if ref["score"] is None:
            assert ref.get("note"), "an unmeasured reference must say so in the figure"


def test_committed_lineage_resolves_and_accumulates_steps():
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


def test_size_ticks_and_param_formatting():
    assert pt.size_ticks(4e8, 6e9) == [5e8, 1e9, 2e9, 5e9]
    assert pt.human_params(596049920) == "0.6B"
    assert pt.human_params(4022468096) == "4B"


def test_markdown_table_matches_the_data():
    data = pt.load()
    nodes = pt.lineage(data["attempts"])
    table = pt.markdown_table(data)
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(rows) == len(data["attempts"]) + 2  # header + separator
    assert f"**{min(a['nll'] for a in data['attempts']):.4f}**" in table
    for node in nodes:
        assert node["run"] in table and node["log"].replace("logs/", "./logs/") in table
        parent = "—" if node["parent_index"] is None else f"#{nodes[node['parent_index']]['n']}"
        assert f"| {parent} | {node['summary']} | {node['total_steps']} |" in table
        # Unmeasured behavior stays blank rather than becoming a zero.
        if "behavior" not in node:
            assert f"| {node['total_steps']} | – |" in table


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


def test_render_survives_a_system_with_no_history(tmp_path):
    """A newly added student has a current best and nothing before it."""
    data = pt.load()
    fresh = {**data["systems"][0]}
    fresh.pop("previous_best", None)
    out = pt.render({**data, "systems": [fresh]}, tmp_path / "fresh.svg")
    assert out.stat().st_size > 1000
