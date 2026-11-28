"""The round's own wiring: the frozen selection, the DEPTH record check, the init record's provenance shape, the
training-plan derivation, the evaluation protocol, the save-point mirror and the kinds' bound records. Seconds, CPU,
no GPU and no network (the relay is a stand-in).

    pytest scripts/stages/stage-1/teacher_parent_initialization/tests
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from stages.teacher_parent_initialization import init_driver as ID
from stages.teacher_parent_initialization import launch as L
from stages.teacher_parent_initialization import plan as PL
from stages.teacher_parent_initialization import selection as SEL
from stages.teacher_parent_initialization import train_driver as TD

REPO = Path(__file__).resolve().parents[5]
JOURNAL = REPO / json.loads(SEL.DESIGN.read_text())["source_search"]["journal"]["path"]
needs_journal = pytest.mark.skipif(not JOURNAL.is_file(), reason="the packed search's journal is not on this host")


@needs_journal
def test_the_committed_selection_is_the_journals_and_its_width_points_follow_the_rule():
    doc = SEL.extract()
    assert SEL.render(doc) == SEL.OUT.read_text()
    k = doc["kept_blocks"]
    assert doc["width"]["points"] == sorted({0, *k}) + [doc["n_root_blocks"]]
    w = doc["width"]["point_weights"]
    assert len(w) == len(doc["width"]["points"]) and w[0] == 9.0 and w[-1] == 8.0 and set(w[1:-1]) == {1.0}
    assert doc["ffn_is_b_first_step"] and len(doc["query_heads"]) == doc["n_root_blocks"]


@needs_journal
def test_a_depth_record_that_did_not_choose_the_lowest_score_is_refused():
    rows = SEL.last_rows(JOURNAL)
    step = copy.deepcopy(rows[json.loads(SEL.DESIGN.read_text())["arms"]["A_teacher_parent"]["selection"]["DEPTH"]
                              ["state_id"]]["steps"][-1])
    assert SEL.depth_rounds_problems(step, 36, 28) == []
    t = step["artifacts"]["search_rounds"][3]["table"]
    worst = max(t, key=lambda x: x["score"])
    worst["score"] = -1.0                       #: another layer now had the lowest score in round 3
    assert any("round 3 chose" in p for p in SEL.depth_rounds_problems(step, 36, 28))
    step2 = copy.deepcopy(rows[json.loads(SEL.DESIGN.read_text())["arms"]["A_teacher_parent"]["selection"]["DEPTH"]
                               ["state_id"]]["steps"][-1])
    step2["local_metrics"]["reference"] = "root_reference/v1"
    assert any("parent_state" in p for p in SEL.depth_rounds_problems(step2, 36, 28))


def _init_build_record() -> dict:
    return {"run_id": "r", "root": {"root_teacher_id": "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c",
                                    "root_teacher_sha256": "a8b9dec375a8f2c5238cac67e7542b3cbaed02d02385e08b05806a75be911192"},
            "checkpoint": {"artifact_digest": "ad" * 32, "shards": [{"filename": "model.safetensors", "sha256": "5e" * 32,
                                                                     "size_bytes": 1}]},
            "mapping": {"content_digest": "c0" * 32}, "gates": {}, "statistics_reproduction": {}, "statistics": {},
            "width_transform": {}, "design_sha256": "d", "selection_sha256": "s"}


def test_the_init_record_is_what_the_lineage_gate_reads(tmp_path):
    from shared.training.initialization_provenance import read
    relay = {k: {"verified": True, "prefix": f"p/{k}", "files_sha256": {}, "relay": "r"} for k in ("init", "mapping",
                                                                                                    "statistics")}
    doc = ID.init_record(_init_build_record(), relay)
    p = tmp_path / "init_record.json"
    p.write_text(json.dumps(doc))
    got = read(p, hashlib.sha256(p.read_bytes()).hexdigest(), "5e" * 32)
    assert got["root_teacher_sha256"].startswith("a8b9dec3") and got["checkpoint_sha256"] == "5e" * 32


def test_the_training_plan_changes_only_the_declared_fields(tmp_path):
    d = json.loads(PL.DESIGN.read_text())
    src = d["training"]["derived_from"]["path"]
    (tmp_path / src).parent.mkdir(parents=True)
    shutil.copyfile(REPO / src, tmp_path / src)
    relay = {k: {"verified": True, "prefix": f"durable/x/{k}", "relay": "r",
                 "files_sha256": {"init/model.safetensors": "5e" * 32} if k == "init" else {}}
             for k in ("init", "mapping", "statistics")}
    rec = {**ID.init_record(_init_build_record(), relay), "arm": PL.ARM}
    rp = tmp_path / "logs/x/init_record.json"
    rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps(rec))
    plan = PL.derive(rp, root=tmp_path)
    assert [u["unit"] for u in plan["units"]] == ["A_teacher_parent-s20260726", "A_teacher_parent-s20260801"]
    b = {u["seed"]: u for u in json.loads((REPO / src).read_text())["units"] if u["arm"] == PL.B_ARM}
    for u in plan["units"]:
        diff = PL.differences(b[u["seed"]]["config"], u["config"])
        assert diff and all(x.split(".")[0] in ("run_name", "student_path", "out_dir", "lineage") for x in diff)
        assert all(x.startswith("lineage.initialization") for x in diff if x.startswith("lineage"))
        assert u["config"]["stage"] == "stage3_recovery" and u["evaluations"] == b[u["seed"]]["evaluations"]
        assert u["config"]["lineage"]["initialization"]["model_safetensors_sha256"] == "5e" * 32
    bad = {**rec, "relay": {**relay, "init": {**relay["init"], "verified": False}}}
    rp.write_text(json.dumps(bad))
    with pytest.raises(PL.PlanError):
        PL.derive(rp, root=tmp_path)


def test_the_evaluation_protocol_is_b_s():
    ev = TD.protocol()
    amix = json.loads((REPO / "logs/stages/stage-2/teacher_modes/plans/token_matched_preregistration.json").read_text())
    assert ev["context"]["tokens"] == 8192                          #: A-MIX's evaluation_spec: trained_context 8192
    assert ev["decoding"] == amix["evaluation"]["decoding"]["path"]
    assert ev["action_protocol"] == amix["evaluation"]["set"]["action_protocol"]
    assert ev["modes"]["H-T"]["template_kwargs"] == {"enable_thinking": True}
    assert ev["modes"]["H-N"]["template_kwargs"] == {"enable_thinking": False}
    #: the runners' defaults, which A-MIX's evaluate_all relied on, are the values passed explicitly here
    import argparse  # noqa: F401
    src = (REPO / "scripts/stages/stage-0/calibration_volume/run_battery.py").read_text()
    assert f'"--check-every", type=int, default={ev["degeneration_check_every"]}' in src
    assert f'"--system", default="{ev["system_message"]}"' in src


def test_the_mirror_keeps_one_save_point_and_never_the_final(tmp_path, monkeypatch):
    from stages.teacher_parent_initialization import relay as R
    from .test_takeover import FakeRelay
    fake = FakeRelay()
    calls = fake.calls
    monkeypatch.setattr(R, "run_online", fake)
    out, ev = tmp_path / "train", tmp_path / "ev"
    m = TD.SavepointMirror("u", out, ev, total_steps=3000, prefix="durable/u")
    for step in (1104, 2208):
        ck = out / "checkpoints" / f"step_{step:06d}"
        (ck / "model").mkdir(parents=True)
        (ck / "model" / "model.safetensors").write_bytes(b"w")
        (ck / "trainer_state.pt").write_bytes(b"s")
        (out / "checkpoints" / "latest.txt").write_text(f"step_{step:06d}\n")
        assert m.mirror_once()["verified"]
        assert m.mirror_once() is None                              #: a mirrored save point is not copied again
    recs = m.records()
    assert [r["tag"] for r in recs] == ["step_001104", "step_002208"]
    assert recs[0]["removed"]["verified"] and "removed" not in recs[1]
    assert ("remove", "durable/u/savepoints/step_001104") in calls
    (out / "checkpoints" / "step_003000" / "model").mkdir(parents=True)
    (out / "checkpoints" / "latest.txt").write_text("step_003000\n")
    assert m.mirror_once() is None                                  #: the final step goes out as final.json


def test_a_single_copy_mirror_removes_the_previous_save_point_before_the_next_upload(tmp_path, monkeypatch):
    """``durability.savepoint_copies: 1``: the relay never holds two save points of the unit -- the previous one is
    removed (confirmed absent) BEFORE the next is uploaded; a removal the relay does not confirm holds the upload back
    for the next round. The default (2) is the registered rotation above."""
    from stages.teacher_parent_initialization import relay as R
    from .test_takeover import FakeRelay
    fake = FakeRelay()
    monkeypatch.setattr(R, "run_online", fake)
    out, ev = tmp_path / "train", tmp_path / "ev"
    m = TD.SavepointMirror("u", out, ev, total_steps=4000, prefix="durable/u", d={**TD.DURABILITY, "savepoint_copies": 1})

    def save(step):
        ck = out / "checkpoints" / f"step_{step:06d}"
        (ck / "model").mkdir(parents=True)
        (ck / "model" / "model.safetensors").write_bytes(b"w")
        (ck / "trainer_state.pt").write_bytes(b"s")
        (out / "checkpoints" / "latest.txt").write_text(f"step_{step:06d}\n")
    save(1104)
    assert m.mirror_once()["verified"]
    save(2208)
    assert m.mirror_once()["verified"]
    ops = [c for c in fake.calls if c[0] in ("put", "remove")]
    i_rm = ops.index(("remove", "durable/u/savepoints/step_001104"))
    i_up = ops.index(("put", "durable/u/savepoints/step_002208"))
    assert i_rm < i_up                                              #: removed first, then uploaded
    real = R.run_online
    monkeypatch.setattr(R, "run_online", lambda op, payload, timeout_s=3600: (
        {"verified_absent": False, "error": "refused"} if op == "remove" else real(op, payload, timeout_s)))
    save(3312)
    assert m.mirror_once() is None                                  #: not confirmed absent: nothing uploaded
    assert [r["tag"] for r in m.records()] == ["step_001104", "step_002208"]
    with pytest.raises(TD.DriverError):
        TD.savepoint_copies({"savepoint_copies": 3})


def test_a_replacement_pod_resumes_the_latest_save_point_with_the_units_whole_log(tmp_path, monkeypatch):
    got = []

    def fake(op, payload, timeout_s=3600):
        got.append(payload["record"]["tag"])
        Path(payload["dest"]).mkdir(parents=True)                 #: `relay.get` writes the files under dest
        return {"verified": True, "from": "r"}
    from stages.teacher_parent_initialization import relay as R
    monkeypatch.setattr(R, "run_online", fake)
    ev, out = tmp_path / "ev", tmp_path / "train"
    (ev / "train").mkdir(parents=True)
    (ev / "train" / "train_log.jsonl").write_text('{"event": "train_step", "step": 2540}\n')
    (ev / "savepoints.json").write_text(json.dumps([
        {"tag": "step_001104", "step": 1104, "verified": True, "removed": {"verified": True}},
        {"tag": "step_002208", "step": 2208, "verified": True}]))
    rec = TD.restore_savepoint(ev, out)
    assert got == ["step_002208"] and rec["tag"] == "step_002208"
    assert (out / "checkpoints" / "latest.txt").read_text().strip() == "step_002208"
    #: the trainer appends to the secured log (`resume_loaded`, then step 2209 on), never starts a tail-only one
    assert (out / "train_log.jsonl").read_text() == (ev / "train" / "train_log.jsonl").read_text()


def test_the_train_kind_binds_the_plan_and_its_init_record(tmp_path):
    for rel in (L.DESIGN, L.SELECTION):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, tmp_path / rel)
    (tmp_path / "logs/x").mkdir(parents=True)
    (tmp_path / "logs/x/init_record.json").write_text("{}")
    (tmp_path / L.PLAN).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / L.PLAN).write_text(json.dumps({"init_record": {"path": "logs/x/init_record.json"}}))
    init = L.kind_conditions(L.INIT_KIND, tmp_path)
    train = L.kind_conditions(L.TRAIN_KIND, tmp_path)
    assert set(init) == {"design_sha256", "selection_sha256"}
    assert set(train) == set(init) | {"plan_sha256", "init_record_sha256"}
    assert L.STUDY.kinds == (L.INIT_KIND, L.TRAIN_KIND) and L.PLAN not in L.STUDY.identity_files
