"""A-MIX-6cat-sr (maintainer 2026-11-09, 'Research A 新增 Hybrid Mixed Stage 3 与 Checkpoint 精简'): the hybrid's thinking
and non-thinking documents admitted as frozen, in one manifest and one sequential pack, each span in its mode's think
state, training and validation isolated across the modes; two fresh units from the packed search's Top-1 under one
multi-mode lineage, each checkpoint evaluated in both modes; kind train_sr_mix refuses without its own grant block;
and its polls carry the Stage-3 students' final weights to the dev box, received only by sha256."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from aadistill.data.token_documents import MaskedDocument
from aadistill.training import lineage as LIN
from stages.teacher_modes import a1_units as AU
from stages.teacher_modes import mixed_packs as MP
from stages.teacher_modes import mixed_training_plan as MTP
from stages.teacher_modes import packed_training_plan as PTP
from stages.teacher_modes import stage3_backup as BK
from stages.teacher_modes import token_matched_phase as TM

PLAN = json.loads(MTP.BOUND.read_text())
BUDGET = json.loads(MTP.BUDGET.read_text())
FREEZE = json.loads(MP.FREEZE.read_text())["arms"][MP.ARM]
SELF_OPEN = [9, 9, MP.IM_START, MP.ASSISTANT, MP.NL]
PRECLOSED = [9, 9, *MP.OPENS["preclosed"]]


# ---- data ---------------------------------------------------------------------------------------------------------

def test_each_supervised_span_is_read_in_the_think_state_its_tokens_carry():
    assert MP.span_state(SELF_OPEN + [MP.THINK, 5], len(SELF_OPEN)) == "self_opening"
    assert MP.span_state(PRECLOSED + [5], len(PRECLOSED)) == "preclosed"
    assert MP.span_state(SELF_OPEN + [5], len(SELF_OPEN)) is None          #: an assistant turn not opening <think>
    assert MP.spans([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]


def _src(train_a1, val_a1, train_a2, val_a2):
    """Two source arms in `sources`' shape: each row (doc_id, data_type, category, unit, supervised, sha256)."""
    def arm(code, train, val):
        man = {"documents": {}, "manifest": None}
        docs = {}
        for split, units in (("training", train), ("validation", val)):
            ds = [MaskedDocument(f"{u}|{code}|t0", (MP.IM_START, 872, MP.NL, hash(u) % 1000, MP.IM_END, 5),
                                 (False,) * 5 + (True,)) for u in units]
            man["documents"][split] = [[d.doc_id, "primary.x", None, u, 1, "s"] for d, u in zip(ds, units)]
            docs[split] = ds
        return {"manifest": man, "docs": docs}
    return {"A1-6cat-sr": arm("H-T", train_a1, val_a1), "A2-6cat-sr": arm("H-N", train_a2, val_a2)}


def test_a_validation_task_of_either_mode_never_trains_in_the_other():
    ok = MP.isolation(_src(["t1", "t2"], ["v1"], ["t1", "t3"], ["v2"]), catalogue={}, eval_set=False)
    assert ok["passed"] and ok["units_in_both_modes"]["training"] == 1     #: one task, two legitimate trajectories
    leak = MP.isolation(_src(["t1"], ["v1"], ["t1", "v1"], []), catalogue={}, eval_set=False)
    assert not leak["passed"] and leak["crossing"]["units"]["n"] == 1
    #: another id, the same task content in the catalogue
    cat = {"t9": {"content_sha256": "c", "family_id": "f"}, "v1": {"content_sha256": "c", "family_id": "f"}}
    alias = MP.isolation(_src(["t9"], ["v1"], [], []), catalogue=cat, eval_set=False)
    assert not alias["passed"] and alias["crossing"]["task_content"]["n"] == 1


def test_document_ids_stay_the_sources_and_are_namespaced_by_mode_only_on_a_collision():
    src = _src(["t1"], [], ["t1"], [])
    assert set(MP.mixed_ids(src).values()) == {"t1|H-T|t0", "t1|H-N|t0"}
    for x in src.values():                      #: the same source id in both modes
        x["manifest"]["documents"]["training"][0][0] = "t1"
    assert set(MP.mixed_ids(src).values()) == {"H-T::t1", "H-N::t1"}


def test_the_frozen_mixed_data_is_the_two_admitted_sets_unchanged_and_50_50():
    fr = json.loads(MP.SOURCE_FREEZE.read_text())["arms"]
    bp = FREEZE["before_packing"]["training"]
    for arm, spec in MP.SOURCES.items():
        assert FREEZE["sources"][arm]["manifest_sha256"] == fr[arm]["identity"]["data_manifest_sha256"]
        assert bp["by_mode"][spec["mode"]] == fr[arm]["before_packing"]["training"]
        assert FREEZE["think_states"][f"{arm}/training"]["all_declared"]
    assert bp["unique_supervised_tokens"] == 5_935_809 + 5_926_755 == 11_862_564
    assert abs(FREEZE["proportions"]["mode_shares"]["thinking"] - 0.5) < 0.001
    pk = FREEZE["packing"]["train"]
    assert pk["unique_supervised_tokens"] == bp["unique_supervised_tokens"] and pk["truncated_tokens"] == 0
    assert sum(m["supervised_tokens_placed"] for m in FREEZE["packing"]["by_mode"]["by_mode"].values()) \
        == pk["supervised_tokens"]
    assert FREEZE["packing"]["spec"]["packing"] == "sequential_truncate_v1" and FREEZE["packing"]["by_mode"][
        "sequences_with_both_modes"] > 0                                   #: one stream, not two packs side by side
    iso = json.loads(MP.ISOLATION.read_text())
    assert iso["passed"] and all(v["n"] == 0 for v in iso["crossing"].values())
    assert all(not f["trained_only_in_the_other_mode"] for f in iso["families"].values())


@pytest.mark.skipif(not (MP.OUT / MP.ARM / "blocks.npz").is_file(), reason="the mixed pack is not on this machine")
def test_the_mixed_pack_passes_the_formal_consumers_check_as_frozen():
    data = PTP.bind_pack(MP.ARM, MP.OUT / MP.ARM, json.loads(MP.FREEZE.read_text()))
    assert data["files_sha256"] == FREEZE["identity"]["pack_files_sha256"]


# ---- plan ---------------------------------------------------------------------------------------------------------

def test_two_fresh_units_from_the_packed_top1_under_one_multi_mode_lineage():
    a1 = json.loads(PTP.BOUND.read_text())["units"][0]
    assert [(u["unit"], u["seed"]) for u in PLAN["units"]] == [(f"{MP.ARM}-s{s}", s) for s in PTP.SEEDS]
    for u in PLAN["units"]:
        cfg, lin = u["config"], u["config"]["lineage"]
        assert cfg["student_path"] == a1["config"]["student_path"]                    #: the Top-1, never A1/A2's
        assert lin["initialization"] == a1["config"]["lineage"]["initialization"]
        assert "mode" not in lin and lin["modes"] == ["thinking", "non_thinking"] and LIN.declared(lin) == []
        assert {s["mode"]: s["source_arm"] for s in lin["data"][0]["sources"]} == {
            "thinking": "A1-6cat-sr", "non_thinking": "A2-6cat-sr"}
        for k in ("loss", "optim", "block_len"):
            assert cfg[k] == a1["config"][k]                                          #: A1/A2's recipe unchanged
        assert cfg["batch"] == a1["config"]["batch"]
        es = u["effective_schedule"]
        assert es["optimizer_steps"] == math.ceil(3 * FREEZE["compute"]["packed_sequences"] / 2) == \
            cfg["schedule"]["total_steps"] != a1["config"]["schedule"]["total_steps"]
        assert {e["mode"]: e["template"] for e in u["evaluations"]} == {
            "thinking": {"enable_thinking": True}, "non_thinking": {"enable_thinking": False}}
    assert all(not v["problems"] for v in PLAN["validation"].values())


def test_the_budget_prices_this_packs_steps_and_both_evaluations():
    arm = BUDGET["arms"][MP.ARM]
    u = PLAN["units"][0]
    s = MTP.seconds_per_step(u["effective_schedule"]["kd_positions_per_step"], "expected")
    row = BUDGET["per_unit"][u["unit"]]["expected"]
    assert row["seconds_per_step"] == s and set(row["evaluation_s"]) == {"thinking", "non_thinking"}
    assert arm["unit_deadline_hours"] * 3600 >= BUDGET["per_unit"][u["unit"]]["upper"]["train_s"]
    assert arm["kind_cumulative_cap_usd"] == round(arm["session_hard_cap"]["hard_cap_usd"]
                                                   + MTP.RECOVERIES * arm["recovery"]["usd_each"], 2)
    assert TM.unit_deadline_hours("train_sr_mix") == arm["unit_deadline_hours"]


@pytest.mark.skipif(not all(Path(p).is_file() for p in MTP.STEP_LOGS), reason="the -sr train logs are not here")
def test_the_step_model_is_the_six_units_measured_steps():
    got = MTP.fit_logs()
    assert {k: got[k] for k in ("a", "b", "steps")} == {k: MTP.STEP_FIT[k] for k in ("a", "b", "steps")}


# ---- driver -------------------------------------------------------------------------------------------------------

def test_a_mixed_unit_is_done_only_when_both_modes_are_scored(tmp_path):
    u = PLAN["units"][0]
    ev = AU.evaluations_of(u, MP.ARM, tmp_path)
    assert [e["path"].name for e in ev] == ["evaluation_thinking", "evaluation_non_thinking"]
    out = tmp_path / "train"
    (out).mkdir()
    (out / "run_completion.json").write_text("{}")
    ev[0]["path"].mkdir()
    (ev[0]["path"] / "formal_report.json").write_text("{}")
    assert AU.state(out, [e["path"] for e in ev]) == "trained"                       #: one mode scored: evaluate on
    ev[1]["path"].mkdir()
    (ev[1]["path"] / "formal_report.json").write_text("{}")
    assert AU.state(out, [e["path"] for e in ev]) == "done"
    with pytest.raises(RuntimeError, match="per mode"):
        AU.evaluations_of({"unit": "x"}, MP.ARM, tmp_path)                           #: never one arm template
    assert AU.evaluations_of({"unit": "x"}, "A1-6cat-sr", tmp_path)[0]["template"] == {"enable_thinking": True}


# ---- kind ---------------------------------------------------------------------------------------------------------

def _grant(tmp_path, block: bool):
    g = json.loads((TM.REPO_ROOT / TM.STUDY.grant).read_text())
    cap = BUDGET["arms"][MP.ARM]["kind_cumulative_cap_usd"]
    g.pop(TM.STAGE3_MIX_BLOCK, None)
    if block:
        g["session_hard_caps_usd"] = {**g["session_hard_caps_usd"], "train_sr_mix": cap}
        g[TM.STAGE3_MIX_BLOCK] = {"approves": TM.STAGE3_MIX_APPROVAL, "granted_by": "maintainer", "decision": "test",
                                  "date": "2026-11-09", "cap_is_cumulative": True, "gpu": TM.GPU,
                                  "plan_sha256": TM.P1._sha(TM.KIND_RECORDS["train_sr_mix"][0], TM.REPO_ROOT),
                                  "budget_sha256": TM.P1._sha(TM.KIND_RECORDS["train_sr_mix"][1], TM.REPO_ROOT),
                                  "kinds": {"train_sr_mix": {"arm": MP.ARM, "hard_cap_usd": cap}},
                                  "network_volume": {"id": "thprcgh7s9"},
                                  "protected_volumes": ["4rhubprswo", "am5m373kuq"]}
    (tmp_path / TM.STUDY.grant).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / TM.STUDY.grant).write_text(json.dumps(g))
    for rel in TM.KIND_RECORDS["train_sr_mix"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((TM.REPO_ROOT / rel).read_bytes())
    return tmp_path, cap


def test_train_sr_mix_needs_its_own_grant_block_and_the_six_units_grant_never_covers_it(tmp_path):
    cap = BUDGET["arms"][MP.ARM]["kind_cumulative_cap_usd"]                           #: the committed grant (2026-11-09)
    first = BUDGET["arms"][MP.ARM]["session_hard_cap"]["hard_cap_usd"]
    assert TM.kind_cap_problems("train_sr_mix", first) == []
    assert TM.kind_cap_problems("train_sr_mix", round(cap + 0.01, 2))
    root, cap = _grant(tmp_path / "a", block=False)
    assert TM.kind_cap_problems("train_sr_mix", 1.0, root)
    root, cap = _grant(tmp_path / "b", block=True)
    assert TM.kind_cap_problems("train_sr_mix", cap, root) == []
    assert TM.kind_cap_problems("train_sr_mix", round(cap + 0.01, 2), root)
    g = json.loads((root / TM.STUDY.grant).read_text())
    g[TM.STAGE3_MIX_BLOCK]["approves"] = TM.STAGE3_APPROVAL                             #: the six units' scope
    (root / TM.STUDY.grant).write_text(json.dumps(g))
    assert any("approves" in b for b in TM.kind_cap_problems("train_sr_mix", cap, root))


def test_the_mixed_kind_relays_its_pack_reads_the_top1_on_the_volume_and_binds_its_records():
    ri = TM.stage3_relay_inputs("train_sr_mix")
    assert {Path(r.path).name: r.sha256 for r in ri} == FREEZE["identity"]["pack_files_sha256"]
    assert all(r.dest == f"{PTP.PACKS}/{MP.ARM}" for r in ri)
    cond = TM.kind_conditions("train_sr_mix")
    assert {k.split(":", 1)[1] for k in cond if k.startswith("record:")} == set(TM.KIND_RECORDS["train_sr_mix"])
    assert "--arm A-MIX-6cat-sr" in TM.train_sr_command(SimpleNamespace(args=SimpleNamespace(
        run_id="r", volume_mount_path="/runpod-volume", network_volume_id="thprcgh7s9", volume_gb=350)), None,
        "train_sr_mix")


# ---- dev-box backup -----------------------------------------------------------------------------------------------

def _pod(tmp_path, files: dict[str, bytes]):
    """A pod whose `sha256sum` lists ``files`` (remote path -> bytes) and whose scp copies them."""
    def run(cmd, timeout=None):
        paths = cmd.split("sha256sum ", 1)[1].split("; echo")[0].split()
        out = "".join(f"{hashlib.sha256(files[p]).hexdigest()}  {p}\n" for p in paths if p in files)
        return SimpleNamespace(returncode=0, stdout=out + f"{TM.P1.LISTED}\n")
    return SimpleNamespace(target=SimpleNamespace(run=run, pod_id="pod1"), evidence={})


def test_a_units_weights_are_received_only_when_every_file_matches_the_pod_and_the_record(tmp_path, monkeypatch):
    model_dir = "/runpod-volume/teacher_modes/stage3_sr/U/train/checkpoints/step_000010/model"
    files = {f"{model_dir}/{n}": n.encode() for n in BK.MODEL_FILES}
    files[f"{str(Path(model_dir).parent)}/resume_identity.json"] = b"{}"
    monkeypatch.setattr(TM.P1.SE, "_scp", lambda ctx, remote, dest, limit_s=None: Path(dest).write_bytes(files[remote]))
    row = {"model_dir": model_dir, "model_safetensors_sha256": hashlib.sha256(b"model.safetensors").hexdigest()}
    r = BK.pull_unit(_pod(tmp_path, files), "U", row, tmp_path)
    assert r["verified"] and BK.received(tmp_path, "U", row["model_safetensors_sha256"])
    assert (tmp_path / "U/step_000010/model/model.safetensors").read_bytes() == b"model.safetensors"
    assert (tmp_path / "U/step_000010/resume_identity.json").is_file()
    wrong = {**row, "model_safetensors_sha256": "0" * 64}                               #: not the committed weights
    r = BK.pull_unit(_pod(tmp_path, files), "V", wrong, tmp_path)
    assert not r["verified"] and not BK.received(tmp_path, "V", wrong["model_safetensors_sha256"])
    record = {"destination_root": str(tmp_path), "units": {"U": row, "V": wrong}}
    session = {"units": {"M": {"final_model_path": model_dir, "final_model_safetensors_sha256": "1" * 64}}}
    assert sorted(BK.owed(record, session)) == ["M", "V"]                              #: U is received


def test_the_backup_record_names_the_six_units_committed_weights():
    rec = json.loads(BK.RECORD.read_text())
    assert sorted(rec["units"]) == sorted(f"{a}-s{s}" for a in PTP.ARMS for s in PTP.SEEDS)
    assert rec == {**BK.plan(Path(rec["destination_root"]))}
    assert "trainer_state.pt" in rec["not_copied"]
