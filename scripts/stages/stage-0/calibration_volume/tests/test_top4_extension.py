"""The Top-4 extension's $0 pieces: the candidate rule, where each candidate's
bytes are, the units and their seeds, and the relay export's stored-identity check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stages.calibration_volume import finalists as F
from stages.calibration_volume import launch
from stages.calibration_volume.dose_driver import candidate_dirs

SEEDS = {"top1": [20260727], "others": [20260726, 20260727]}


def _selection(order):
    return {"decisions": [{"state_id": s, "quality_order": q, "path": f"P({s})"}
                          for s, q in order]}


def test_byte_identical_leaves_count_once_in_quality_order():
    states = {s: {"checkpoint_sha256": c} for s, c in
              [("a", "x"), ("b", "x"), ("c", "y"), ("d", "z"), ("e", "z"), ("f", "w")]}
    got = F.distinct_finalists(_selection([("b", 2), ("a", 1), ("c", 3), ("d", 4),
                                           ("e", 5), ("f", 6)]), states, 4)
    assert [(g["rank"], g["state_id"], g["quality_order"]) for g in got] == [
        (1, "a", 1), (2, "c", 3), (3, "d", 4), (4, "f", 6)]
    assert got[0]["identical_leaves"] == [{"state_id": "b", "quality_order": 2}]
    with pytest.raises(F.FinalistError, match="only 3 distinct"):
        F.distinct_finalists(_selection([("a", 1), ("b", 2), ("c", 3), ("d", 4)]), states, 4)


def test_a_resumed_journal_row_never_blanks_a_recorded_value(tmp_path):
    j = tmp_path / "states.jsonl"
    j.write_text(json.dumps({"state_id": "a", "checkpoint_sha256": "x", "steps": [{"trace": 1}]})
                 + "\n" + json.dumps({"state_id": "a", "checkpoint_sha256": "x", "steps": []}) + "\n")
    assert F.journal_index(j)["a"]["steps"] == [{"trace": 1}]


def test_a_rebuilt_candidate_is_read_from_finalists_and_the_rest_from_the_search(tmp_path):
    unit, fin = tmp_path / "search_unit", tmp_path / "finalists"
    rows = [{"state_id": s, "checkpoint_sha256": f"c_{s}",
             "checkpoint_path": f"/workspace/aad/artifacts/audit/x/d1/search/states/{s}/h{s}"}
            for s in "abcd"]
    (unit / "search").mkdir(parents=True)
    (unit / "search" / "states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (unit / "stage1_selection.json").write_text(json.dumps(_selection(
        [(s, i + 1) for i, s in enumerate("abcd")])))
    (fin / "b").mkdir(parents=True)
    got = {c["state_id"]: c["dir"] for c in candidate_dirs(unit, fin, 4)}
    assert got["b"] == str(fin / "b")
    assert got["a"] == str(unit / "search" / "states" / "a" / "ha")


def _manifest():
    cands = [{"rank": r, "state_id": s, "quality_order": q, "checkpoint_sha256": f"sh_{s}"}
             for r, s, q in [(1, "aaaaaaaa1", 1), (2, "bbbbbbbb2", 3), (3, "cccccccc3", 4),
                             (4, "dddddddd4", 6)]]
    files = [{"treatment": "d2", "state_id": c["state_id"], "path": f"p/{c['state_id']}/{n}",
              "name": n, "sha256": (c["checkpoint_sha256"] if n.endswith(".safetensors") else "cfg"),
              "bytes": 1}
             for c in cands for n in ("config.json", "model.safetensors")]
    return {"candidates": {"d2": cands}, "files": files}


def test_each_dose_has_seven_units_and_top1_only_takes_the_new_seed():
    units = F.probe_units(_manifest(), "d2", SEEDS)
    assert [(u["rank"], u["seed"]) for u in units] == [
        (1, 20260727), (2, 20260726), (2, 20260727), (3, 20260726), (3, 20260727),
        (4, 20260726), (4, 20260727)]
    assert len({u["label"] for u in units}) == 7
    # two seeds of one candidate stage its files once
    plan = {"units": units}
    assert len(launch.probe_relay_inputs(plan)) == 8


def test_a_unit_without_its_shard_on_the_relay_is_refused():
    m = _manifest()
    m["files"] = [f for f in m["files"] if f["state_id"] != "cccccccc3" or f["name"] != "model.safetensors"]
    with pytest.raises(F.FinalistError, match="no relay copy"):
        F.probe_units(m, "d2", SEEDS)


class _Hub:
    """A relay that stores what it is sent, and answers with what it stores."""

    def __init__(self, corrupt=False):
        self.files, self.uploads, self.corrupt = {}, 0, corrupt

    def get_paths_info(self, repo, paths, repo_type):
        out = []
        for p in paths:
            if p in self.files:
                data = self.files[p]
                lfs = (SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
                       if p.endswith(".safetensors") else None)
                blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
                out.append(SimpleNamespace(lfs=lfs, blob_id=blob, size=len(data)))
        return out

    def upload_file(self, path_or_fileobj, path_in_repo, **_k):
        self.uploads += 1
        data = Path(path_or_fileobj).read_bytes()
        self.files[path_in_repo] = data + (b"!" if self.corrupt else b"")


def test_the_export_verifies_what_the_relay_stored_and_is_idempotent(tmp_path):
    d = tmp_path / "ck"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"weights")
    (d / "config.json").write_text("{}")
    c = [{"state_id": "s1", "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(), "dir": d}]
    hub = _Hub()
    rows = F.export_candidates(c, study="formal_v1", treatment="d1", repo="r", token="t", api=hub)
    assert hub.uploads == 2 and {r["name"] for r in rows} == {"config.json", "model.safetensors"}
    F.export_candidates(c, study="formal_v1", treatment="d1", repo="r", token="t", api=hub)
    assert hub.uploads == 2                                   # nothing re-sent
    with pytest.raises(F.FinalistError, match="the relay holds"):
        F.export_candidates(c, study="formal_v1", treatment="d1", repo="r", token="t",
                            api=_Hub(corrupt=True))
    c[0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(F.FinalistError, match="does not hold the shard"):
        F.export_candidates(c, study="formal_v1", treatment="d1", repo="r", token="t", api=hub)


def test_the_shared_recovery_writes_the_config_d1s_session_wrote(tmp_path, monkeypatch):
    """`train_recovery` is now the dose's R stage and every probe's: lifting it out
    must not move a byte of the config a dose's Top-1 is trained under."""
    from stages.calibration_volume import dose_driver as DD
    from stages.phase_c1 import autoinit_c1_driver as C1
    recorded = (DD.REPO / "logs/stages/stage-0/calibration_volume/runs/dose_d1_20261019_055515/"
                "evidence/d1/recovery/recovery_config.json")
    monkeypatch.setattr(DD.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(C1, "trained_model_dir", lambda out: out / "model")
    cfg = tmp_path / "recovery_config.json"
    rec_out = DD.REPO / "artifacts/stages/stage-0/calibration_volume/formal/d1/recovery"
    student = ("/workspace/aad/artifacts/audit/stage0_calvol_search/d1/search/states/"
               "72b76a1ca4300f1fc3b341c28738f425/1a35fed31c0bff69c42b39262449adf4")
    monkeypatch.setattr(DD.shutil, "rmtree", lambda *a, **k: None)
    DD.train_recovery(Path(student), rec_out, cfg, tmp_path / "recovery.log", env={},
                      run_name="s0v_formal_d1", seed=20260726,
                      purpose=("Stage-0 volume formal dose d1: the committed Top-1 recovered by "
                               "the frozen recipe at its own seed; only run identity, pack path "
                               "and student_path differ from it"))
    assert cfg.read_bytes() == recorded.read_bytes()


def test_a_probe_may_change_the_seed_and_nothing_outside_the_override_set(tmp_path, monkeypatch):
    from stages.calibration_volume import dose_driver as DD
    from stages.phase_c1 import autoinit_c1_driver as C1
    monkeypatch.setattr(DD.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(C1, "trained_model_dir", lambda out: out / "model")
    monkeypatch.setattr(DD.shutil, "rmtree", lambda *a, **k: None)
    cfg = tmp_path / "c.json"
    DD.train_recovery(tmp_path / "student", DD.REPO / "artifacts/x/recovery", cfg,
                      tmp_path / "l.log", env={}, run_name="u", seed=20260727, purpose="p")
    frozen = json.loads(C1.FROZEN_RECIPE.read_text())
    got = json.loads(cfg.read_text())
    assert got["seed"] == 20260727
    assert {k for k in frozen if frozen[k] != got[k]} <= set(C1.C1_PROBE_OVERRIDES)
