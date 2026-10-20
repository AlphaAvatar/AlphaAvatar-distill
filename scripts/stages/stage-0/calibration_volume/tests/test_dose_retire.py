"""A later dose frees the volume by retiring a COMPLETE earlier dose's states.

The step deletes bytes on the study's only durable store, so what it may touch
is checked at $0: everything but the committed Top-1 of a COMPLETE dose, with
a tombstone naming each retired state; never an unfinished dose.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stages.calibration_volume import governance as G
from stages.calibration_volume.dose_driver import retire_completed_states


def _dose(study: Path, t: str, status: str, states=("top", "a", "b")) -> Path:
    unit = study / t / "search_unit"
    for s in states:
        (unit / "search" / "states" / s / "ck").mkdir(parents=True)
        (unit / "search" / "states" / s / "ck" / "model.safetensors").write_bytes(b"x" * 10)
    (unit / "search" / "states.jsonl").write_text("".join(
        json.dumps({"state_id": s, "artifact_digest": f"d_{s}", "checkpoint_sha256": f"c_{s}"})
        + "\n" for s in states))
    (unit / "stage1_selection.json").write_text(json.dumps({"selected": [{"state_id": "top"}]}))
    (study / t / "dose_out").mkdir(parents=True)
    (study / t / "dose_out" / "dose_record.json").write_text(json.dumps({"status": status}))
    return unit / "search" / "states"


def test_a_complete_dose_keeps_only_its_top1_and_leaves_a_tombstone(tmp_path):
    states = _dose(tmp_path, "d1", "COMPLETE")
    doc = retire_completed_states(tmp_path, "d1")
    assert sorted(p.name for p in states.iterdir()) == ["top"]
    assert doc["status"] == "retired" and doc["kept"] == ["top"]
    assert [(r["state_id"], r["artifact_digest"], r["bytes"]) for r in doc["retired"]] == [
        ("a", "d_a", 10), ("b", "d_b", 10)]
    assert json.loads((tmp_path / "d1" / "retired_states.json").read_text()) == doc
    assert (tmp_path / "d1" / "search_unit" / "search" / "states.jsonl").is_file()
    assert retire_completed_states(tmp_path, "d1") == doc          # idempotent


@pytest.mark.parametrize("status", ["STARTED", "FAILED"])
def test_an_unfinished_dose_is_refused_and_untouched(tmp_path, status):
    states = _dose(tmp_path, "d1", status)
    with pytest.raises(RuntimeError, match="only a COMPLETE dose"):
        retire_completed_states(tmp_path, "d1")
    assert sorted(p.name for p in states.iterdir()) == ["a", "b", "top"]


def test_each_dose_retires_exactly_the_doses_before_it():
    order = json.loads((G.REPO / "logs/stages/stage-0/calibration_volume/plans/"
                        "volume_preregistration.json").read_text())["execution_order"]
    for i, t in enumerate(order):
        assert G.plan_doc(f"dose_{t}").get("retire_states_of", []) == order[:i], t


def test_with_the_top4_extension_a_dose_keeps_its_distinct_top4(tmp_path):
    """2026-10-20: the extension recovers each dose's distinct Top-4, so retirement
    keeps them; a byte-identical leaf is represented by its better-ranked twin."""
    states = _dose(tmp_path, "d2", "COMPLETE", states=("top", "twin", "q3", "q4", "q5", "q6"))
    unit = tmp_path / "d2" / "search_unit"
    shas = {"top": "c1", "twin": "c1", "q3": "c3", "q4": "c4", "q5": "c4", "q6": "c6"}
    (unit / "search" / "states.jsonl").write_text("".join(
        json.dumps({"state_id": s, "artifact_digest": f"d_{s}", "checkpoint_sha256": c}) + "\n"
        for s, c in shas.items()))
    (unit / "stage1_selection.json").write_text(json.dumps({
        "selected": [{"state_id": "top"}],
        "decisions": [{"state_id": s, "quality_order": i + 1} for i, s in enumerate(shas)]}))
    doc = retire_completed_states(tmp_path, "d2", keep_k=4)
    assert sorted(p.name for p in states.iterdir()) == ["q3", "q4", "q6", "top"]
    assert sorted(r["state_id"] for r in doc["retired"]) == ["q5", "twin"]
