"""DMR1 variant 2 as revised 2026-11-28 (recurrent depth, both initializations x seed 20260726), at $0.

* the schedule is the committed mapping's (16 blocks, 36 logical layers) and the recurrent config is the base config
  through `recurrent_config_from`, byte for byte;
* each loop arm's recurrent init record is over the base the design FIXES for it (V2-D: SOB1's depth init; V2-M:
  variant 1's COMPLETE build record), with that base's weights and relay copy and the committed config; a record over
  the other base, another arm, config or digest is refused; variant 3 has no base;
* the two loop arms' plans are SOB1's depth unit outside the declared fields, seed 20260726 only, the eval-only state
  probe alike, and differ from each other only in the initialization and the arm's names; the engine validation is
  bound into V2-D's plan only;
* the real training driver's entry for depth_merge and both loop arms, with SOB1's stand-in trainer and engine: the
  loop arms' student path holds the recurrent config when the trainer starts, the units end durable and their sets
  complete against their finals, the closeout's completion is COMPLETE, and a takeover trains and generates nothing;
  V2-D's engine check runs before any training, a FAIL trains nothing and a PASS is not repeated by a takeover;
* the engine check's gate and its HF cache check on a tiny recurrent model on the CPU: the real cache passes against
  the control, the shared-cache fault fails;
* R1-R4: the interaction is (V2-M - V2-D) - (M - D) item by item with all four records of an item resampled
  together; a condition that misses items computes nothing that needs it;
* the real trainer on a tiny recurrent student and teacher with the hidden-state term: a resumed run continues
  exactly as the uninterrupted one, and every save point reloads as the recurrent model on its schedule.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from stages.depth_merge import common as DC
from stages.depth_merge.tests.test_depth_merge import link_into, merge_init_record
from stages.depth_recurrence import schedule as SCH

REPO = Path(__file__).resolve().parents[5]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- schedule, config ---------------------------------------------------------------------------------------------

def test_the_schedule_is_the_mappings_and_the_config_is_the_derivation():
    s = SCH.schedule()
    assert s["n_logical_layers"] == 36 and s["n_parameter_blocks"] == 16
    assert s["root_layer_of_layer"] == list(range(36))
    assert SCH.config_problems() == []
    cfg = json.loads(SCH.RECURRENT_CONFIG.read_text())
    base = json.loads(SCH.BASE_CONFIG.read_text())
    assert sha(SCH.BASE_CONFIG) == DC.recurrence_design()["base"]["kept_init"]["config_json_sha256"]
    assert cfg["layer_to_block"] == s["layer_to_block"] and cfg["num_parameter_blocks"] == base["num_hidden_layers"] == 16
    for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
              "vocab_size", "rope_parameters", "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps"):
        assert cfg[k] == base[k], k
    d = DC.recurrence_design()
    assert d["hidden_state"]["config"]["pairs"] == [[p, p] for p in range(36)]
    assert d["hidden_state"]["config"]["weight"] == d["hidden_state"]["lambda_hidden"] == 1.0


# ---- each arm's fixed base --------------------------------------------------------------------------------------------

def _kept_root(tmp: Path) -> tuple[Path, dict]:
    k = DC.recurrence_design()["base"]["kept_init"]
    link_into(tmp, k["path"])
    link_into(tmp, DC.design()["baseline"]["training_plan"]["path"])
    link_into(tmp, "configs/stages/stage-1/depth_recurrence/recurrent_config.json")
    rec = SCH.init_record("depth_loop_kept", root=tmp)
    p = tmp / "logs/stages/stage-1/depth_recurrence/inits/KEPT/init_record.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    return p, rec


def _merged_root(tmp: Path, monkeypatch) -> tuple[Path, dict]:
    """The MERGED base: variant 1's (synthetic) COMPLETE init record bound by its training plan."""
    from stages.depth_merge import plan as PL
    link_into(tmp, DC.design()["baseline"]["training_plan"]["path"])
    link_into(tmp, "configs/stages/stage-1/depth_recurrence/recurrent_config.json")
    ir = merge_init_record(tmp)
    rel = str(ir.relative_to(tmp))
    plan_p = tmp / "logs/stages/stage-1/depth_merge/plans/training_plan_depth_merge.json"
    plan_p.parent.mkdir(parents=True, exist_ok=True)
    plan_p.write_text(json.dumps({"init_record": {"path": rel, "sha256": sha(ir)}}))
    real_out = PL.out_path
    monkeypatch.setattr(PL, "out_path", lambda arm: plan_p if arm == "depth_merge" else real_out(arm))
    rec = SCH.init_record("depth_loop_merged", root=tmp)
    p = tmp / "logs/stages/stage-1/depth_recurrence/inits/MERGED/init_record.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    return p, rec


def test_each_loop_arm_has_a_fixed_base_and_variant_3_has_none():
    d = DC.recurrence_design()
    assert SCH.loop_arms(d) == ("depth_loop_kept", "depth_loop_merged")
    assert SCH.base_of("depth_loop_kept") == "KEPT" and SCH.base_of("depth_loop_merged") == "MERGED"
    with pytest.raises(SCH.ScheduleError):
        SCH.base_of("depth_loop_hidden")
    assert "initialization_selection" not in d and "initialization_selection" in d["superseded_2026_11_28"]
    assert d["arms"]["depth_loop_kept"]["seeds"] == d["arms"]["depth_loop_merged"]["seeds"] == [20260726]


def test_the_kept_arms_recurrent_init_is_sob1s_weights_on_the_36_layer_schedule(tmp_path):
    p, rec = _kept_root(tmp_path)
    base = json.loads((REPO / DC.recurrence_design()["base"]["kept_init"]["path"]).read_text())
    assert rec["arms"] == ["depth_loop_kept"] and rec["path"] == "KEPT"
    assert rec["checkpoint"]["files_sha256"]["model.safetensors"] == base["checkpoint"]["shards"][0]["sha256"]
    assert rec["checkpoint"]["files_sha256"]["config.json"] == sha(SCH.RECURRENT_CONFIG)
    assert rec["checkpoint"]["artifact_digest"] == SCH.execution_digest(
        base["checkpoint"]["shards"][0]["sha256"], sha(SCH.RECURRENT_CONFIG))
    assert rec["checkpoint"]["artifact_digest"] != base["checkpoint"]["artifact_digest"]     #: weights alone are not it
    assert rec["relay"]["init"] == base["relay"]["init"] and rec["execution"]["n_logical_layers"] == 36
    assert SCH.init_record_problems(rec, arm="depth_loop_kept", root=tmp_path) == []
    #: refusals: another base, another config, another arm, another digest, a KEPT record named for the merged arm
    bad = json.loads(p.read_text())
    bad["base"]["init_record"] = {"path": "logs/x.json", "sha256": "0" * 64}
    assert SCH.init_record_problems(bad, arm="depth_loop_kept", root=tmp_path)
    bad = json.loads(p.read_text())
    bad["checkpoint"]["files_sha256"]["config.json"] = "0" * 64
    assert SCH.init_record_problems(bad, arm="depth_loop_kept", root=tmp_path)
    assert SCH.init_record_problems(rec, arm="depth_merge", root=tmp_path)
    bad = json.loads(p.read_text())
    bad["checkpoint"]["artifact_digest"] = bad["base"]["artifact_digest"]
    assert SCH.init_record_problems(bad, arm="depth_loop_kept", root=tmp_path)
    bad = {**json.loads(p.read_text()), "arms": ["depth_loop_merged"]}
    assert SCH.init_record_problems(bad, arm="depth_loop_merged", root=tmp_path)
    assert SCH.init_record_problems(rec, arm="depth_loop_hidden", root=tmp_path)


def test_the_merged_arms_recurrent_init_is_variant_1s_weights_and_its_relay_copy(tmp_path, monkeypatch):
    p, rec = _merged_root(tmp_path, monkeypatch)
    assert rec["path"] == "MERGED" and rec["arms"] == ["depth_loop_merged"]
    assert rec["checkpoint"]["files_sha256"]["model.safetensors"] == "b" * 64
    assert rec["relay"]["init"]["prefix"].startswith("durable/stage1_depth_merge/")
    assert SCH.init_record_problems(rec, arm="depth_loop_merged", root=tmp_path) == []
    relabeled = {**rec, "arms": ["depth_loop_kept"]}             #: a MERGED record presented for the kept arm
    assert SCH.init_record_problems(relabeled, arm="depth_loop_kept", root=tmp_path)


def test_the_committed_init_records_and_plans_are_the_derivations():
    from stages.depth_merge import plan as PL
    for arm, path in (("depth_loop_kept", "KEPT"), ("depth_loop_merged", "MERGED")):
        p = SCH.init_record_path(path)
        assert json.loads(p.read_text()) == SCH.init_record(arm)
        assert SCH.init_record_problems(json.loads(p.read_text()), arm=arm) == []
        cur = PL.out_path(arm)
        assert cur.read_text() == PL.render(PL.derive(arm, REPO / json.loads(cur.read_text())["init_record"]["path"]))


# ---- the loop arms' plans -------------------------------------------------------------------------------------------

def test_the_loop_arms_plans_differ_from_sob1s_depth_unit_only_where_declared_and_from_each_other_only_in_the_init(
        tmp_path, monkeypatch):
    from stages.depth_merge import plan as PL
    from stages.teacher_parent_initialization.plan import differences
    ik, _ = _kept_root(tmp_path)
    im, _ = _merged_root(tmp_path, monkeypatch)
    d = DC.recurrence_design()
    docs = {"depth_loop_kept": PL.derive("depth_loop_kept", ik, root=tmp_path),
            "depth_loop_merged": PL.derive("depth_loop_merged", im, root=tmp_path)}
    for arm, doc in docs.items():
        assert [u["unit"] for u in doc["units"]] == [f"{arm}-s20260726"] and doc["unit_order"] == [f"{arm}-s20260726"]
        for u in doc["units"]:
            c = u["config"]
            assert c["stage"] == "stage3_recovery" and "hidden_state" not in c["loss"]
            assert c["state_probe"] == d["diagnostics"]["state_probe"]["config"]
            assert u["initialization"]["recurrent_config"]["text"] == SCH.RECURRENT_CONFIG.read_text()
            assert c["schedule"]["total_steps"] == 9701 and c["batch"]["blocks_per_step"] == 2
            assert c["batch"]["micro_blocks"] == 1 and c["block_len"] == 8192 and c["seed"] == 20260726
            assert c["loss"]["ce_weight"] == 0.25 and c["loss"]["kd_weight"] == 1.0
            assert doc["derivation_checks"][u["unit"]]["outside_declared"] == []
        assert doc["relay_prefix_root"] == "durable/stage1_depth_recurrence"
        assert doc["durability"]["savepoint_copies"] == 1
    assert docs["depth_loop_kept"]["engine_validation"]["required"] is True
    assert docs["depth_loop_merged"]["engine_validation"]["required"] is False
    #: the two conditions differ in the initialization and the arm's names only
    k, m = docs["depth_loop_kept"]["units"][0], docs["depth_loop_merged"]["units"][0]
    assert set(differences(k["config"], m["config"])) <= {
        "run_name", "student_path", "out_dir", "lineage.initialization.model_safetensors_sha256",
        "lineage.initialization.provenance.record", "lineage.initialization.provenance.record_sha256"}
    assert k["initialization"]["recurrent_config"] == m["initialization"]["recurrent_config"]
    assert k["initialization"]["execution"] == m["initialization"]["execution"]
    assert k["initialization"]["artifact_digest"] != m["initialization"]["artifact_digest"]
    with pytest.raises(Exception):
        PL.derive("depth_loop_hidden", ik, root=tmp_path)


# ---- the real driver entry, end to end, for the three arms ----------------------------------------------------------

def _eval_set_or_skip():
    from stages.single_operator_budget.tests.test_single_operator_budget import _eval_set_or_skip as f
    f()


def _plan_for(tmp: Path, arm: str, monkeypatch) -> tuple[Path, dict]:
    from stages.depth_merge import plan as PL
    root = tmp / "root"
    if arm == "depth_merge":
        link_into(root, DC.design()["baseline"]["training_plan"]["path"])
        ip = merge_init_record(root)
    elif arm == "depth_loop_merged":
        ip, _ = _merged_root(root, monkeypatch)
    else:
        ip, _ = _kept_root(root)
    plan = PL.derive(arm, ip, root=root)
    for u in plan["units"]:
        u["config"]["out_dir"] = str(tmp / "pod" / u["unit"] / "train")
        u["config"]["student_path"] = str(tmp / "pod" / "init" / arm)
    p = tmp / f"plan_{arm}.json"
    p.write_text(json.dumps(plan))
    return p, plan


def _run(tmp: Path, monkeypatch, arm: str, out: Path, resume: Path | None = None, engine_status=("PASS",)) -> dict:
    """One `depth_merge.train_driver.main` attempt; the trainer, the engine, the package, the pod's environment
    and the relay's transport stood in for (SOB1's harness), everything else the real chain."""
    from shared import study_engineering as SE
    import shared.pod.teacher_cache as TC
    from stages.depth_merge import train_driver as DTD
    from stages.reasoning_inheritance import driver as RID
    from stages.single_operator_budget import common as C
    from stages.single_operator_budget import train_driver as SD
    from stages.single_operator_budget.tests import test_single_operator_budget as T
    from stages.teacher_modes import a1_units as AU
    from stages.teacher_modes import packed_training_plan as PTP
    from stages.teacher_modes import training_plan as TP
    from stages.teacher_parent_initialization import train_driver as TD
    import os
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))     #: the driver sets it for a loop arm
    L, kind = DTD.launcher_of(arm)
    plan_p = tmp / f"plan_{arm}.json"
    plan = json.loads(plan_p.read_text())
    trained, asked, seen = [], {}, []
    monkeypatch.setattr(L.Authorization, "load", classmethod(lambda cls, p: type("A", (), {
        "stage_conditions": {"kind": kind, "arm": arm}})()))
    monkeypatch.setattr(SE, "require_on_pod", lambda *a, **k: {"authorization_id": "test"})
    monkeypatch.setattr(SE, "identity_hash", lambda *a, **k: "h")
    monkeypatch.setattr(SD, "authorized_plan_problems", lambda *a, **k: [])     #: the binding is the launcher's test
    monkeypatch.setattr(TC, "local_snapshot", lambda *a, **k: str(tmp))
    monkeypatch.setattr(TP, "stage_tokenizers", lambda snap: {})
    monkeypatch.setattr(TP, "admit", lambda *a, **k: None)
    monkeypatch.setattr(PTP, "attention_problems", lambda cfg, repo: [])
    monkeypatch.setattr(AU, "train_telemetry", lambda d: {"context_strategies": [PTP.CONTEXT_STRATEGY]})

    def fetch(u, d=None):
        sp = Path(u["config"]["student_path"])
        sp.mkdir(parents=True, exist_ok=True)
        (sp / "model.safetensors").write_bytes(b"base weights")
        shutil.copyfile(SCH.BASE_CONFIG, sp / "config.json")
        return {"stub": True}
    monkeypatch.setattr(TD, "fetch_init", fetch)
    monkeypatch.setattr(TD, "_sleep", lambda s: None)
    inner = T._trainer(trained)

    def train(cfg_path, *, resume, deadline_s, log):
        cfg = json.loads(Path(cfg_path).read_text())
        seen.append({"config_json": json.loads((Path(cfg["student_path"]) / "config.json").read_text()),
                     "hidden": cfg["loss"].get("hidden_state"), "stage": cfg["stage"]})
        return inner(cfg_path, resume=resume, deadline_s=deadline_s, log=log)
    monkeypatch.setattr(AU, "train", train)
    from stages.depth_recurrence import engine_check as EC

    def engine(plan, student, work, vllm_python, **k):
        seen.append({"engine_check": str(student), "student_has_weights": (Path(student) / "model.safetensors").is_file()})
        doc = {"status": engine_status[0], "checks": {"stub": engine_status[0] == "PASS"}}
        Path(work).mkdir(parents=True, exist_ok=True)
        (Path(work) / "engine_check.json").write_text(json.dumps(doc))
        return doc
    monkeypatch.setattr(EC, "run", engine)
    monkeypatch.setattr(RID, "_run_bounded", T._mode_engine(asked))
    monkeypatch.setattr(C, "package", lambda model_dir, dest: {
        "weights_sha256": {"model.safetensors": sha(Path(model_dir) / "model.safetensors")},
        "linked": ["model.safetensors"]})
    before = {k: getattr(C, k) for k in ("DESIGN", "SELECTION", "STRUCTURES", "LOGS")}
    try:
        argv = ["--authorization", "a.json", "--out", str(out), "--status", str(tmp / f"status_{arm}.log"), "--plan",
                str(plan_p), "--unit-deadline-hours", "1", "--attempt", f"dmr1_{arm}_t1"]
        rc = DTD.main(argv + (["--resume", str(resume)] if resume else []))
    finally:
        C.restore(before)
    return {"rc": rc, "units": json.loads((out / "units.json").read_text()), "trained": trained, "seen": seen,
            "plan": plan}


@pytest.mark.parametrize("arm", ["depth_merge", "depth_loop_kept", "depth_loop_merged"])
def test_each_arm_runs_through_the_real_driver_to_a_complete_session_and_a_takeover_does_nothing(tmp_path, monkeypatch,
                                                                                               arm):
    _eval_set_or_skip()
    from stages.depth_merge import session as S
    from stages.single_operator_budget import common as C
    from stages.single_operator_budget import completion as CP
    from stages.single_operator_budget import launch as SL
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization.tests.test_takeover import FakeRelay
    relay = FakeRelay()
    monkeypatch.setattr(R, "run_online", relay)
    _plan_for(tmp_path, arm, monkeypatch)
    out = tmp_path / "attempt1/store/out"
    got = _run(tmp_path, monkeypatch, arm, out)
    assert got["rc"] == 0 and got["units"]["terminal"] == "ALL_DONE", got["units"].get("error")
    assert got["trained"] == ([20260726, 20260801] if arm == "depth_merge" else [20260726])
    checks = [s for s in got["seen"] if "engine_check" in s]
    assert len(checks) == (1 if arm == "depth_loop_kept" else 0)
    if checks:                         #: before any training, on the staged init
        assert "engine_check" in got["seen"][0] and checks[0]["student_has_weights"]
    assert got["units"]["vllm_plugin_on_pythonpath"] is DC.is_recurrent(arm)   #: a stock Qwen3 arm keeps SOB1's env
    for s in (x for x in got["seen"] if "engine_check" not in x):
        if DC.is_recurrent(arm):
            assert s["config_json"] == json.loads(SCH.RECURRENT_CONFIG.read_text())
            assert s["config_json"]["architectures"] == ["RecurrentDepthForCausalLM"]
            assert s["stage"] == "stage3_recovery"
        else:
            assert s["config_json"]["architectures"] == ["Qwen3ForCausalLM"] and s["stage"] == "stage3_recovery"
        assert s["hidden"] is None
    before = DC.bind_for(arm)
    try:
        ses = CP.session(out, arm, plan=got["plan"])
        assert ses["complete"] and ses["n_complete"] == (4 if arm == "depth_merge" else 2)
        for k, v in ses["sets"].items():
            assert v["binding"]["arm"] == arm
    finally:
        C.restore(before)
    #: the closeout's completion on the secured copy: both finals at their placement (the relay), four sets
    L, kind = __import__("stages.depth_merge.train_driver", fromlist=["x"]).launcher_of(arm)
    root = tmp_path / "root"
    pp = root / S.plan_path(arm)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps(got["plan"]))
    monkeypatch.setattr(SL, "_lfs_sha", lambda paths: {q: sha(relay.files[q]) if isinstance(relay.files[q], Path)
                                                       else hashlib.sha256(relay.files[q]).hexdigest()
                                                       for q in paths if q in relay.files})
    res = S.completion(L.VARIANT, kind, arm, f"dmr1_{arm}_t1", tmp_path / "attempt1", root=root,
                       lfs_sha=lambda paths: {q: (hashlib.sha256(relay.files[q]).hexdigest()
                                                  if isinstance(relay.files[q], (bytes, bytearray)) else sha(relay.files[q]))
                                              for q in paths if q in relay.files})
    assert res["status"] == "COMPLETE", res
    #: a takeover of the secured evidence: nothing trained, nothing generated
    again = _run(tmp_path, monkeypatch, arm, tmp_path / "attempt2/store/out", resume=out)
    assert again["rc"] == 0 and again["trained"] == []
    assert all(r["state_at_start"] == "done" for r in again["units"]["units"].values())
    assert not [s for s in again["seen"] if "engine_check" in s]         #: a PASS taken over is not repeated


def test_a_failed_engine_check_trains_nothing(tmp_path, monkeypatch):
    _eval_set_or_skip()
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization.tests.test_takeover import FakeRelay
    monkeypatch.setattr(R, "run_online", FakeRelay())
    _plan_for(tmp_path, "depth_loop_kept", monkeypatch)
    got = _run(tmp_path, monkeypatch, "depth_loop_kept", tmp_path / "a/store/out", engine_status=("FAIL",))
    assert got["rc"] != 0 and got["units"]["terminal"] == "RUN_FAILED" and got["trained"] == []
    assert "engine validation failed" in got["units"]["error"]["error"]
    status = (tmp_path / "status_depth_loop_kept.log").read_text()
    assert "ENGINE_CHECK_FAILED" in status and "TRAIN_START" not in status


# ---- the engine check's gate and its HF part on the CPU ---------------------------------------------------------------

def test_the_engine_gate_reads_each_check_against_the_control_and_needs_the_fault_to_fail():
    from stages.depth_recurrence import engine_check as EC
    ok = {"top1_agreement": 1.0, "mean_abs_dlogp": 0.01}
    m = {"hf_cache": {"recurrent": {**ok, "cache_entries": 36, "logical_layers": 36}, "control": ok,
                      "shared_cache_fault": {"top1_agreement": 0.4, "mean_abs_dlogp": 2.0}},
         "vllm_generation": {"recurrent": {"top1_agreement": 0.99, "mean_abs_dlogp": 0.04}, "control": ok}}
    assert EC.verdict(m)["status"] == "PASS"
    worse = {**m, "vllm_generation": {**m["vllm_generation"], "recurrent": {"top1_agreement": 0.9, "mean_abs_dlogp": 0.04}}}
    assert EC.verdict(worse)["status"] == "FAIL"
    blind = {**m, "hf_cache": {**m["hf_cache"], "shared_cache_fault": ok}}       #: the gate cannot see a cache bug
    assert EC.verdict(blind)["status"] == "FAIL" and not EC.verdict(blind)["checks"]["sensitivity_fault_fails"]
    short = {**m, "hf_cache": {**m["hf_cache"], "recurrent": {**ok, "cache_entries": 16, "logical_layers": 36}}}
    assert EC.verdict(short)["status"] == "FAIL"


def test_the_hf_cache_check_passes_the_real_cache_and_fails_the_shared_one_on_a_tiny_model():
    import sys
    sys.path.insert(0, str(REPO / "tests/training"))
    from test_hidden_state import student
    from stages.depth_recurrence import engine_check as EC
    torch.manual_seed(3)
    m = student(sched=[0, 1, 1, 1, 2]).eval()
    seqs = [torch.randint(0, m.config.vocab_size, (40,)).tolist() for _ in range(2)]
    good = EC.hf_cache_check(m, seqs, 24)
    bad = EC.hf_cache_check(m, seqs, 24, fault=True)
    assert good["positions"] == bad["positions"] == 2 * 16
    assert good["top1_agreement"] == 1.0 and good["max_abs_dlogp"] < 1e-4
    assert good["cache_entries"] == good["logical_layers"] == 5
    assert bad["top1_agreement"] < 1.0 or bad["max_abs_dlogp"] > 1e-2    #: another iteration's keys read
    assert not EC.passes(bad, good)                                   #: the gate sees it against a clean control
    from aadistill.initialization.adapters import recurrent_depth as RD
    assert RD.LogicalLayerCache.__name__ == "LogicalLayerCache"       #: the fault is undone


# ---- R1-R4 ----------------------------------------------------------------------------------------------------------------

def test_the_interaction_is_formed_item_by_item_and_resampled_with_all_four_records():
    from stages.depth_merge import analysis as A
    import random
    rnd = random.Random(7)
    items = [f"i{j}" for j in range(60)]
    groups = {f"g{j}": items[3 * j:3 * j + 3] for j in range(20)}
    y = {c: {i: rnd.random() < p for i in items} for c, p in (("V2-D", .3), ("V2-M", .4), ("D", .2), ("M", .1))}
    row = A.recurrence_modes({"H-T": y}, groups, 1)["H-T"]
    e = lambda a, b: 100 * sum(int(y[a][i]) - int(y[b][i]) for i in items) / len(items)  # noqa: E731
    assert row["R1"]["estimate_pp"] == round(e("V2-D", "D"), 2) and row["R2"]["estimate_pp"] == round(e("V2-M", "M"), 2)
    assert row["R3"]["estimate_pp"] == round(e("V2-M", "V2-D"), 2)
    assert row["R4"]["estimate_pp"] == round(e("V2-M", "V2-D") - e("M", "D"), 2)
    lo, hi = row["R4"]["ci95_pp"]
    assert lo <= row["R4"]["estimate_pp"] <= hi
    #: recurrence changes nothing in either condition: the interaction is 0 in every resample
    same = {"V2-D": y["D"], "V2-M": y["M"], "D": y["D"], "M": y["M"]}
    r0 = A.recurrence_modes({"H-T": same}, groups, 1)["H-T"]["R4"]
    assert r0["estimate_pp"] == 0 and r0["ci95_pp"] == [0.0, 0.0]
    #: a condition missing an item: nothing that needs it is computed
    part = {**y, "V2-M": {i: v for i, v in y["V2-M"].items() if i != "i0"}}
    rp = A.recurrence_modes({"H-T": part}, groups, 1)["H-T"]
    assert rp["R2"]["status"] == rp["R3"]["status"] == rp["R4"]["status"] == "not_computed"
    assert "estimate_pp" in rp["R1"] and rp["accuracy_pct"]["V2-M"] is None


def test_a_recurrent_config_that_does_not_hash_to_the_plans_is_refused(tmp_path):
    from stages.depth_merge import train_driver as DTD
    u = {"unit": "depth_loop_kept-s1", "initialization": {"recurrent_config": {"text": "{}\n", "sha256": "0" * 64,
                                                                         "path": "x"}}}
    with pytest.raises(DTD.DriverError):
        DTD.write_recurrent_config(u, tmp_path)


# ---- the real trainer on a tiny recurrent student: resume and save points ------------------------------------------

def test_a_resumed_recurrent_hidden_run_continues_exactly_and_saves_reload_on_the_schedule(tmp_path):
    import sys
    sys.path.insert(0, str(REPO / "tests/training"))
    from test_hidden_state import cfg as hcfg, hidden_cfg, student, teacher
    from test_train import toy_blocks
    from transformers import AutoModelForCausalLM
    from aadistill.initialization.adapters.families import register_custom_model_families
    from aadistill.training.train import Trainer
    register_custom_model_families()
    blocks = toy_blocks()

    def make(out):
        c = hcfg(out, hidden=hidden_cfg(1.0), gradient_checkpointing=True)
        c["checkpoint"] = {"save_every": 2, "keep_last": 3}
        c["schedule"]["total_steps"] = 4
        return c
    whole = Trainer(make(tmp_path / "w"), student(), blocks, teacher=teacher(), device="cpu")
    log_w = [whole.step_once() for _ in range(4)]
    part = Trainer(make(tmp_path / "p"), student(), blocks, teacher=teacher(), device="cpu")
    for _ in range(2):
        part.step_once()
    ck = part.save_checkpoint()
    back = AutoModelForCausalLM.from_pretrained(ck / "model", dtype=torch.float32)
    assert type(back).__name__ == "RecurrentDepthForCausalLM" and back.config.layer_to_block == [0, 1, 1, 2]
    resumed = Trainer(make(tmp_path / "p"), back, blocks, teacher=teacher(), device="cpu")
    resumed.restore(ck)
    log_r = [resumed.step_once() for _ in range(2)]
    for a, b in zip(log_w[2:], log_r):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-6) and a["hidden"] == pytest.approx(b["hidden"], rel=1e-6)
    for (n, p), (_, q) in zip(whole.student.named_parameters(), resumed.student.named_parameters()):
        assert torch.allclose(p, q, atol=1e-6), n
