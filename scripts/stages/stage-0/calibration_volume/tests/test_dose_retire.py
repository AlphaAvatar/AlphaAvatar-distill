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


# --- releasing finalists the relay holds (2026-10-20: d4 filled the volume) ----

from types import SimpleNamespace

from stages.calibration_volume.dose_driver import (read_products, release_relayed_finalists,
                                                   require_pushed, volume_room)
from stages.calibration_volume.finalists import _git_blob_sha1, _sha256


def _finalists(study: Path):
    """d1-shaped: its Top-1 is a search state, its Top-2 a rebuilt copy."""
    unit = study / "d1" / "search_unit"
    ck = unit / "search" / "states" / "top" / "ck"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"w-top")
    (ck / "config.json").write_text("{}")
    (unit / "search" / "states" / "top" / "pruned.safetensors").write_bytes(b"p" * 7)
    (unit / "search" / "states" / "other" / "ck").mkdir(parents=True)
    (unit / "search" / "states.jsonl").write_text(json.dumps(
        {"state_id": "top", "checkpoint_path": "/pod/states/top/ck"}) + "\n")
    rebuilt = study / "d1" / "finalists" / "r2"
    rebuilt.mkdir(parents=True)
    (rebuilt / "model.safetensors").write_bytes(b"w-r2")
    (rebuilt / "config.json").write_text('{"r": 2}')
    files, cands = [], []
    for sid, d in (("top", ck), ("r2", rebuilt)):
        cands.append({"state_id": sid, "rank": len(cands) + 1,
                      "checkpoint_sha256": _sha256(d / "model.safetensors")})
        for f in sorted(d.iterdir()):
            files.append({"treatment": "d1", "state_id": sid, "name": f.name,
                          "path": f"relay/d1/{sid}/{f.name}", "sha256": _sha256(f),
                          "is_shard": f.suffix == ".safetensors"})
    hub = {}
    for r in files:
        local = (ck if r["state_id"] == "top" else rebuilt) / r["name"]
        hub[r["path"]] = SimpleNamespace(
            lfs=SimpleNamespace(sha256=r["sha256"]) if r["is_shard"] else None,
            blob_id=_git_blob_sha1(local))
    api = SimpleNamespace(get_paths_info=lambda repo, paths, repo_type: [
        hub[p] for p in paths if p in hub])
    return {"candidates": {"d1": cands}, "files": files}, hub, api


def test_finalists_the_relay_holds_are_released_with_a_tombstone(tmp_path):
    manifest, _, api = _finalists(tmp_path)
    doc = release_relayed_finalists(tmp_path, "d1", manifest, api=api, repo="r")
    states = tmp_path / "d1" / "search_unit" / "search" / "states"
    assert sorted(p.name for p in states.iterdir()) == ["other"]       # not a candidate
    assert not (tmp_path / "d1" / "finalists" / "r2").exists()
    assert doc["status"] == "released" and doc["released_bytes"] == 5 + 2 + 7 + 4 + 8
    assert [r["dirs"] for r in doc["released"]] == [["d1/search_unit/search/states/top"],
                                                    ["d1/finalists/r2"]]
    assert json.loads((tmp_path / "d1" / "released_finalists.json").read_text()) == doc
    assert release_relayed_finalists(tmp_path, "d1", manifest, api=api, repo="r") == doc


@pytest.mark.parametrize("what", ["shard", "small", "missing"])
def test_a_candidate_the_relay_does_not_hold_releases_nothing(tmp_path, what):
    manifest, hub, api = _finalists(tmp_path)
    if what == "shard":
        hub["relay/d1/r2/model.safetensors"].lfs.sha256 = "0" * 64
    elif what == "small":
        hub["relay/d1/r2/config.json"].blob_id = "0" * 40
    else:
        del hub["relay/d1/r2/config.json"]
    with pytest.raises(RuntimeError, match="nothing of d1 was released"):
        release_relayed_finalists(tmp_path, "d1", manifest, api=api, repo="r")
    assert (tmp_path / "d1" / "search_unit" / "search" / "states" / "top").is_dir()
    assert (tmp_path / "d1" / "finalists" / "r2").is_dir()


def test_a_torn_product_record_is_set_aside_and_the_dose_resumes_unrecorded(tmp_path):
    p = tmp_path / "products.json"
    assert read_products(p) == ({}, None)
    p.write_text('{"search": {"complete": true}}')
    assert read_products(p) == ({"search": {"complete": True}}, None)
    p.write_text('{"search": {"compl')
    prod, aside = read_products(p)
    assert prod == {} and not p.exists() and (tmp_path / aside).read_text() == '{"search": {"compl'


@pytest.mark.parametrize("res", [{"mode": "push", "error": "OSError(122, 'Disk quota exceeded')"},
                                 {"rc": 1, "stderr": "boom"}])
def test_a_failed_push_is_never_recorded_as_durable(res):
    with pytest.raises(RuntimeError, match="push of x"):
        require_pushed(res, "x")
    ok = {"mode": "push", "copied": 1, "skipped": 0, "bytes": 3}
    assert require_pushed(ok, "x") is ok


def test_room_is_the_sold_size_less_what_du_finds_after_stale_partials_go(tmp_path):
    dose = tmp_path / "study" / "d4"
    (dose / "a").mkdir(parents=True)
    (dose / "a" / "s.safetensors").write_bytes(b"x" * 4096)
    (dose / "a" / ".s2.safetensors.partial").write_bytes(b"y" * 4096)
    room = volume_room(tmp_path, 1.0, dose)
    assert room["stale_partials_removed"] == ["a/.s2.safetensors.partial"]
    assert not (dose / "a" / ".s2.safetensors.partial").exists()
    assert room["free_bytes"] == int(1e9) - room["used_bytes"] and room["used_bytes"] >= 4096
