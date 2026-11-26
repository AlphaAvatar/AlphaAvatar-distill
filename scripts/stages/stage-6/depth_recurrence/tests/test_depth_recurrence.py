"""DMR1 variants 2 and 3 (recurrent depth, with and without per-iteration hidden-state distillation), at $0.

* the schedule is the committed mapping's (16 blocks, 36 logical layers) and the recurrent config is the base config
  through `recurrent_config_from`, byte for byte;
* the selection rule: MERGED only when comparison 1 is complete and the pooled and every seed's delta exceed 0;
  ties and disagreement select KEPT; an incomplete comparison selects nothing;
* the recurrent init record binds only the two legal paths, the selection that chose it, the base's weights and
  relay copy and the committed config; anything else is refused;
* the loop arms' plans are SOB1's depth unit outside the declared fields: the stage, and for variant 3 exactly the
  registered hidden-state configuration;
* the real training driver's entry, for all three arms, with SOB1's stand-in trainer and engine: the recurrent arms'
  student path holds the recurrent config when the trainer starts, both seeds end durable and four sets complete
  against their finals, the closeout's completion is COMPLETE, and a takeover trains and generates nothing;
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
    assert sha(SCH.BASE_CONFIG) == DC.stage6_design()["base"]["kept_init"]["config_json_sha256"]
    assert cfg["layer_to_block"] == s["layer_to_block"] and cfg["num_parameter_blocks"] == base["num_hidden_layers"] == 16
    for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
              "vocab_size", "rope_parameters", "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps"):
        assert cfg[k] == base[k], k
    d = DC.stage6_design()
    assert d["hidden_state"]["config"]["pairs"] == [[p, p] for p in range(36)]
    assert d["hidden_state"]["config"]["weight"] == d["hidden_state"]["lambda_hidden"] == 1.0


# ---- the selection --------------------------------------------------------------------------------------------------

def _c1(pooled, s1, s2, complete=True):
    return {"modes": {"H-T": {"complete": complete, "delta": pooled,
                              "per_seed": {"20260726": {"delta": s1}, "20260801": {"delta": s2}}}}}


def test_the_selection_rule_prefers_kept_unless_both_seeds_and_the_pool_favour_merged():
    assert SCH.decide(_c1(0.02, 0.03, 0.01))["choice"] == "MERGED"
    assert SCH.decide(_c1(0.01, 0.03, -0.01))["choice"] == "KEPT"        #: the seeds disagree
    assert SCH.decide(_c1(0.0, 0.0, 0.0))["choice"] == "KEPT"            #: a tie
    assert SCH.decide(_c1(-0.02, -0.01, -0.03))["choice"] == "KEPT"
    assert SCH.decide(_c1(0.05, 0.05, 0.05, complete=False))["status"] == "REFUSED"
    #: the p-value is never read
    sig = _c1(0.001, 0.001, 0.001)
    sig["modes"]["H-T"]["p_two_sided"] = 0.9
    assert SCH.decide(sig)["choice"] == "MERGED"


def _selection(root: Path, choice: str, base_ref: dict) -> Path:
    p = root / "logs/stages/stage-6/depth_recurrence/plans/init_selection.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": SCH.SELECTION_SCHEMA, "status": "COMPLETE", "choice": choice,
                             "path": {"base_init_record": base_ref}}))
    return p


def _kept_root(tmp: Path) -> tuple[Path, dict]:
    k = DC.stage6_design()["base"]["kept_init"]
    link_into(tmp, k["path"])
    link_into(tmp, DC.design()["baseline"]["training_plan"]["path"])
    link_into(tmp, "configs/stages/stage-6/depth_recurrence/recurrent_config.json")
    sel = _selection(tmp, "KEPT", {"path": k["path"], "sha256": k["sha256"]})
    rec = SCH.init_record(sel, root=tmp)
    p = tmp / "logs/stages/stage-6/depth_recurrence/inits/KEPT/init_record.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    return p, rec


def test_the_kept_paths_recurrent_init_is_sob1s_weights_on_the_36_layer_schedule(tmp_path):
    p, rec = _kept_root(tmp_path)
    base = json.loads((REPO / DC.stage6_design()["base"]["kept_init"]["path"]).read_text())
    assert rec["checkpoint"]["files_sha256"]["model.safetensors"] == base["checkpoint"]["shards"][0]["sha256"]
    assert rec["checkpoint"]["files_sha256"]["config.json"] == sha(SCH.RECURRENT_CONFIG)
    assert rec["checkpoint"]["artifact_digest"] == SCH.execution_digest(
        base["checkpoint"]["shards"][0]["sha256"], sha(SCH.RECURRENT_CONFIG))
    assert rec["checkpoint"]["artifact_digest"] != base["checkpoint"]["artifact_digest"]     #: weights alone are not it
    assert rec["relay"]["init"] == base["relay"]["init"] and rec["execution"]["n_logical_layers"] == 36
    assert SCH.init_record_problems(rec, arm="depth_loop", root=tmp_path) == []
    #: refusals: another base, another config, another arm, another digest
    bad = json.loads(p.read_text())
    bad["base"]["init_record"] = {"path": "logs/x.json", "sha256": "0" * 64}
    assert SCH.init_record_problems(bad, arm="depth_loop", root=tmp_path)
    bad = json.loads(p.read_text())
    bad["checkpoint"]["files_sha256"]["config.json"] = "0" * 64
    assert SCH.init_record_problems(bad, arm="depth_loop", root=tmp_path)
    assert SCH.init_record_problems(rec, arm="depth_merge", root=tmp_path)
    bad = json.loads(p.read_text())
    bad["checkpoint"]["artifact_digest"] = bad["base"]["artifact_digest"]
    assert SCH.init_record_problems(bad, arm="depth_loop", root=tmp_path)


def test_a_selection_naming_the_other_paths_base_is_refused(tmp_path, monkeypatch):
    k = DC.stage6_design()["base"]["kept_init"]
    link_into(tmp_path, k["path"])
    #: MERGED, but the record it names is SOB1's depth init: not variant 1's
    sel = _selection(tmp_path, "MERGED", {"path": k["path"], "sha256": k["sha256"]})
    from stages.depth_merge import plan as PL
    monkeypatch.setattr(PL, "out_path", lambda arm: tmp_path / "no_plan.json")
    with pytest.raises(SCH.ScheduleError):
        SCH.init_record(sel, root=tmp_path)


def _merged_root(tmp: Path, monkeypatch) -> tuple[Path, dict]:
    """The MERGED path: variant 1's (synthetic) COMPLETE init record bound by its training plan."""
    from stages.depth_merge import plan as PL
    link_into(tmp, DC.design()["baseline"]["training_plan"]["path"])
    link_into(tmp, "configs/stages/stage-6/depth_recurrence/recurrent_config.json")
    ir = merge_init_record(tmp)
    rel = str(ir.relative_to(tmp))
    plan_p = tmp / "logs/stages/stage-1/depth_merge/plans/training_plan_depth_merge.json"
    plan_p.parent.mkdir(parents=True, exist_ok=True)
    plan_p.write_text(json.dumps({"init_record": {"path": rel, "sha256": sha(ir)}}))
    real_out = PL.out_path
    monkeypatch.setattr(PL, "out_path", lambda arm: plan_p if arm == "depth_merge" else real_out(arm))
    sel = _selection(tmp, "MERGED", {"path": rel, "sha256": sha(ir)})
    rec = SCH.init_record(sel, root=tmp)
    p = tmp / "logs/stages/stage-6/depth_recurrence/inits/MERGED/init_record.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    return p, rec


def test_the_merged_paths_recurrent_init_is_variant_1s_weights_and_its_relay_copy(tmp_path, monkeypatch):
    p, rec = _merged_root(tmp_path, monkeypatch)
    assert rec["path"] == "MERGED" and rec["checkpoint"]["files_sha256"]["model.safetensors"] == "b" * 64
    assert rec["relay"]["init"]["prefix"].startswith("durable/stage1_depth_merge/")
    assert SCH.init_record_problems(rec, arm="depth_loop_hidden", root=tmp_path) == []


# ---- the loop arms' plans -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["depth_loop", "depth_loop_hidden"])
def test_the_loop_arms_plans_differ_from_sob1s_depth_unit_only_where_declared(tmp_path, arm):
    from stages.depth_merge import plan as PL
    ip, _rec = _kept_root(tmp_path)
    doc = PL.derive(arm, ip, root=tmp_path)
    hidden = DC.stage6_design()["hidden_state"]["config"]
    for u in doc["units"]:
        c = u["config"]
        assert c["stage"] == "stage6_recurrent_depth"
        assert (c["loss"].get("hidden_state") == hidden) == (arm == "depth_loop_hidden")
        assert ("hidden_state" in c["loss"]) == (arm == "depth_loop_hidden")
        assert u["initialization"]["recurrent_config"]["text"] == SCH.RECURRENT_CONFIG.read_text()
        assert u["initialization"]["recurrent_config"]["sha256"] == sha(SCH.RECURRENT_CONFIG)
        assert c["schedule"]["total_steps"] == 9701 and c["batch"]["blocks_per_step"] == 2
        assert c["loss"]["ce_weight"] == 0.25 and c["loss"]["kd_weight"] == 1.0
    assert doc["relay_prefix_root"] == "durable/stage6_depth_recurrence"
    #: depth_loop's units and depth_loop_hidden's differ only by the hidden-state term (and their names)
    if arm == "depth_loop_hidden":
        other = PL.derive("depth_loop", ip, root=tmp_path)
        from stages.teacher_parent_initialization.plan import differences
        for a, b in zip(doc["units"], other["units"]):
            assert set(differences(b["config"], a["config"])) == {"run_name", "student_path", "out_dir",
                                                                  "loss.hidden_state"}


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
    else:
        ip, _ = _kept_root(root)
    plan = PL.derive(arm, ip, root=root)
    for u in plan["units"]:
        u["config"]["out_dir"] = str(tmp / "pod" / u["unit"] / "train")
        u["config"]["student_path"] = str(tmp / "pod" / "init" / arm)
    p = tmp / f"plan_{arm}.json"
    p.write_text(json.dumps(plan))
    return p, plan


def _run(tmp: Path, monkeypatch, arm: str, out: Path, resume: Path | None = None) -> dict:
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


@pytest.mark.parametrize("arm", ["depth_merge", "depth_loop", "depth_loop_hidden"])
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
    assert got["trained"] == [20260726, 20260801]
    assert got["units"]["vllm_plugin_on_pythonpath"] is DC.is_recurrent(arm)   #: a stock Qwen3 arm keeps SOB1's env
    for s in got["seen"]:
        if DC.is_recurrent(arm):
            assert s["config_json"] == json.loads(SCH.RECURRENT_CONFIG.read_text())
            assert s["config_json"]["architectures"] == ["RecurrentDepthForCausalLM"]
            assert s["stage"] == "stage6_recurrent_depth"
        else:
            assert s["config_json"]["architectures"] == ["Qwen3ForCausalLM"] and s["stage"] == "stage3_recovery"
        assert (s["hidden"] is not None) == (arm == "depth_loop_hidden")
    before = DC.bind_for(arm)
    try:
        ses = CP.session(out, arm, plan=got["plan"])
        assert ses["complete"] and ses["n_complete"] == 4
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


def test_a_recurrent_config_that_does_not_hash_to_the_plans_is_refused(tmp_path):
    from stages.depth_merge import train_driver as DTD
    u = {"unit": "depth_loop-s1", "initialization": {"recurrent_config": {"text": "{}\n", "sha256": "0" * 64,
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
