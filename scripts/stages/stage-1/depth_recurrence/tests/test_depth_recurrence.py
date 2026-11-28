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


ENV = {"train": {"torch": "t", "transformers": "x", "cuda": "c", "gpu": "NVIDIA L40S"},
       "vllm": {"vllm": "0.27.1", "torch": "t2"}}


def good_record(plan: dict, env: dict = ENV, **edit) -> dict:
    """An engine-check record as `engine_check.run` writes it for ``plan``: complete, PASS, bound to the plan."""
    from stages.depth_recurrence import engine_check as EC
    n = EC.INPUTS["n_prompts"] * EC.INPUTS["new_tokens"]
    ex = plan["units"][0]["initialization"]["execution"]
    ok = {"positions": n, "top1_agreement": 0.99, "mean_abs_dlogp": 0.01, "max_abs_dlogp": 0.2}
    m = {"hf_cache": {"recurrent": {**ok, "cache_entries": ex["n_logical_layers"], "logical_layers": ex["n_logical_layers"]},
                      "control": {**ok, "cache_entries": ex["n_parameter_blocks"],
                                  "logical_layers": ex["n_parameter_blocks"]},
                      "shared_cache_fault": {**ok, "top1_agreement": 0.31, "mean_abs_dlogp": 2.4,
                                             "cache_entries": ex["n_logical_layers"],
                                             "logical_layers": ex["n_logical_layers"]}},
         "vllm_generation": {"recurrent": dict(ok), "control": dict(ok)}}
    outs = {k: {"n_outputs": EC.INPUTS["n_prompts"], "lengths": [EC.INPUTS["new_tokens"]] * EC.INPUTS["n_prompts"],
                "logprob_lengths": [EC.INPUTS["new_tokens"]] * EC.INPUTS["n_prompts"],
                "finish_reasons": ["length"] * EC.INPUTS["n_prompts"], "all_logprobs_finite": True,
                "prompt_lengths": [EC.INPUTS["prompt_tokens"]] * EC.INPUTS["n_prompts"]} for k in ("recurrent", "control")}
    for path, v in edit.items():                      #: "a.b.c" -> value, applied before the verdict
        node, keys = {"metrics": m, "vllm_outputs": outs}, path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = v
    ll = {"recurrent": ex["n_logical_layers"], "control": ex["n_parameter_blocks"]}
    src = EC.registered_source(plan)
    return {"schema": EC.SCHEMA, "arm": src["arm"], "source": src,
            "binding": {**EC.expected_binding(plan), "environment": env, "prompts_sha256": "p" * 64},
            "validated_weights_sha256": {"model.safetensors": src["registered_weights_sha256"]},
            "metrics": m, "vllm_outputs": outs, **EC.verdict(m, outs, EC.GATE, EC.INPUTS, ll)}


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
        assert "savepoint_copies" not in doc["durability"]
    ek, em = docs["depth_loop_kept"]["engine_validation"], docs["depth_loop_merged"]["engine_validation"]
    assert ek["required"] is em["required"] is True and ek["record"] == em["record"]
    assert ek["mode"] == "run" and em["mode"] == "reuse" and em["from_arm"] == "depth_loop_kept"
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
    if arm == "depth_loop_merged":
        #: the record V2-D's session would leave: made for the KEPT plan, adopted where V2-M's plan looks
        rec = tmp / "adopted_engine_check.json"            #: as V2-D's real session would have written it
        rec.write_text(json.dumps(good_record(json.loads(PL.out_path("depth_loop_kept").read_text()))))
        plan["engine_validation"]["record"] = str(rec)
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
        latest = Path(cfg["out_dir"]) / "checkpoints" / "latest.txt"
        seen.append({"config_json": json.loads((Path(cfg["student_path"]) / "config.json").read_text()),
                     "hidden": cfg["loss"].get("hidden_state"), "stage": cfg["stage"], "resume": resume,
                     "resumed_from": latest.read_text().strip() if latest.is_file() else None})
        return inner(cfg_path, resume=resume, deadline_s=deadline_s, log=log)
    monkeypatch.setattr(AU, "train", train)
    from stages.depth_recurrence import engine_check as EC

    def engine(plan, student, work, vllm_python, **k):
        seen.append({"engine_check": str(student), "student_has_weights": (Path(student) / "model.safetensors").is_file()})
        doc = (good_record(plan) if engine_status[0] == "PASS" else
               good_record(plan, **{"metrics.vllm_generation.recurrent.positions": 0}))
        Path(work).mkdir(parents=True, exist_ok=True)
        (Path(work) / "engine_check.json").write_text(json.dumps(doc))
        return doc
    monkeypatch.setattr(EC, "run", engine)
    monkeypatch.setattr(EC, "environment", lambda vllm_python=None: ENV)
    #: the pod's input re-derivation (the pack re-hashed, the prompts rebuilt) is tested on its own below
    monkeypatch.setattr(EC, "input_problems", lambda rec, plan, root=None: [])
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


# ---- the save-point rotation, interrupted, through the real takeover entry ------------------------------------------

def _cut(relay, when: str, tag: str = "step_002208"):
    """Attempt 1's pod lost at one point of the rotation that publishes ``tag`` (the second save point) over the first:
    the new save point uploading, being verified (`relay.put` re-downloads and re-hashes: it answers unverified), its
    handoff unfinished (pointer written, the listing never confirmed), or the old one's removal refused."""
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization.tests.test_takeover import PodLost
    real_put, real_remove = relay.put, relay.remove
    if when == "uploading the new save point":
        relay.crash = lambda op, p, ph: op == "put" and tag in (p.get("prefix") or "") and ph == "before"
    elif when == "verifying the new save point":
        def put(p):
            if tag in p["prefix"] and set(p["files"]) != {R.POINTER}:
                real_put(p)                                     #: the bytes arrived; the verifying download failed
                return {"verified": False, "error": "the verifying download did not re-hash"}
            return real_put(p)
        relay.put = put
        relay.crash = lambda op, p, ph: op == "records" and ph == "before"      #: then the pod goes
    elif when == "handoff not finished":
        relay.crash = lambda op, p, ph: op == "put" and set(p.get("files") or {}) == {R.POINTER} and ph == "after"
    elif when == "the old save point's removal failed":
        def remove(p):
            return {"verified_absent": False, "error": "refused"}
        relay.remove = remove
        relay.crash = lambda op, p, ph: op == "remove" and ph == "after"
    else:
        raise AssertionError(when)
    return PodLost, (real_put, real_remove)


@pytest.mark.parametrize("when", ["uploading the new save point", "verifying the new save point",
                                  "handoff not finished", "the old save point's removal failed"])
def test_an_interrupted_rotation_is_taken_over_from_a_durable_save_point_never_from_step_0(tmp_path, monkeypatch,
                                                                                         when):
    """The registered two-copy rotation (the review of 2026-11-28): from the first durable save point on, a valid and
    discoverable copy exists at every interruption. Attempt 1's mirror publishes step 1104, then is cut while publishing
    step 2208; the dev box holds only what its last poll pulled (step 1104's record). Attempt 2 enters through the real
    driver (`train_driver.main --resume`): it restores the latest save point the relay lists and verifies, and the
    trainer resumes from it -- never from the initialization."""
    _eval_set_or_skip()
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization import train_driver as TD
    from stages.teacher_parent_initialization.tests.test_takeover import FakeRelay, save_point
    relay = FakeRelay()
    monkeypatch.setattr(R, "run_online", relay)
    arm = "depth_loop_kept"
    _, plan = _plan_for(tmp_path, arm, monkeypatch)
    u = plan["units"][0]
    out1 = tmp_path / "attempt1/store/out"
    out_dir, ev_unit = Path(u["config"]["out_dir"]), out1 / u["unit"]
    prefix = f"{plan['relay_prefix_root']}/{arm}/{u['unit']}"
    m = TD.SavepointMirror(u["unit"], out_dir, ev_unit, int(u["config"]["schedule"]["total_steps"]), prefix,
                           d=TD.durability(plan))
    save_point(out_dir, 1104)
    assert m.mirror_once()["handoff"]["verified"]
    devbox = tmp_path / "devbox/store/out"
    shutil.copytree(out1, devbox)                                  #: the poll's last copy: step 1104 only
    save_point(out_dir, 2208)
    PodLost, _ = _cut(relay, when)
    try:
        m.mirror_once()
        m.mirror_once()                                            #: a second round: removal / listing / handoff
    except PodLost:
        pass
    relay.crash = lambda op, p, ph: False
    shutil.rmtree(tmp_path / "pod")                                #: the pod and its local checkpoints are gone
    listed = {t for t in ("step_001104", "step_002208")
              if f"{prefix}/savepoints/{t}/{R.POINTER}" in relay.files
              and f"{prefix}/savepoints/{t}/model/model.safetensors" in relay.files}
    assert "step_001104" in listed or "step_002208" in listed      #: never zero durable copies
    got = _run(tmp_path, monkeypatch, arm, tmp_path / "attempt2/store/out", resume=devbox)
    assert got["rc"] == 0, got["units"].get("error")
    train = [s for s in got["seen"] if "engine_check" not in s]
    assert len(train) == 1 and train[0]["resume"] is True, train
    assert train[0]["resumed_from"] in listed
    row = got["units"]["units"][u["unit"]]
    assert row["restored_savepoint"]["tag"] == train[0]["resumed_from"]


def test_the_rotation_never_leaves_the_relay_without_a_durable_save_point():
    """The relay arithmetic the launch gate asks for is the two-copy rotation's: each new save point beside the
    previous one, the final beside the last; no single-copy branch exists."""
    from stages.depth_merge import plan as PL
    from stages.single_operator_budget import launch as SL
    from stages.teacher_parent_initialization import train_driver as TD
    assert not hasattr(TD, "savepoint_copies") and not hasattr(TD, "remove_before_upload")
    sizes = __import__("stages.depth_merge.session", fromlist=["x"]).checkpoint_bytes()
    plan = json.loads(PL.out_path("depth_loop_kept").read_text())
    assert "savepoint_copies" not in plan["durability"]
    need = SL.rotation_need(plan, {}, sizes)
    assert need["peak_bytes"] == 2 * sizes["save_point"]
    trace = need["trace"]
    for i, row in enumerate(trace):
        if "removed" in row["step"]:
            assert "uploaded" in trace[i - 1]["step"] or "removed" in trace[i - 1]["step"]


@pytest.mark.parametrize("arm", ["depth_loop_kept", "depth_loop_merged"])
def test_the_launch_gate_asks_the_relay_for_the_storage_plans_need_margin_included(monkeypatch, arm):
    """The launch gate (`session.relay_room`) and the storage plan share `session.training_relay_need`: a first run
    needs two overlapping save points + the registered 1 GiB margin (39.99 GiB), so 39.50 GiB of headroom is REFUSED
    and 40.5 GiB accepted; a takeover holding a valid save point needs its remaining rotation + the same margin."""
    from types import SimpleNamespace
    import aadistill.runtime.hub_capacity as HC
    from stages.depth_merge import plan as PL
    from stages.depth_merge import session as S
    from stages.depth_merge import storage as ST
    from stages.depth_recurrence import launch as L
    from stages.single_operator_budget import launch as SL
    G = 2 ** 30
    if not Path("~/.cache/huggingface/token").expanduser().is_file():
        pytest.skip("no hub token file to read (the gate reads it before asking)")
    plan = json.loads(PL.out_path(arm).read_text())
    unit = plan["unit_order"][0]
    state = {"now": {}}
    monkeypatch.setattr(SL, "reclaim_superseded", lambda arm_, plan=None, record=None: {"stub": True})
    monkeypatch.setattr(SL, "relay_state", lambda arm_, plan=None: state["now"])
    room = {"bytes": 0}
    asked = []
    monkeypatch.setattr(HC, "would_accept", lambda repo, sizes, tok: asked.append(sum(sizes)) or SimpleNamespace(
        accepted=sum(sizes) <= room["bytes"], __str__=lambda self: "answer"))
    first = S.training_relay_need(arm, plan, {})
    assert first["need_gib"] == 39.99 and first["margin_bytes"] == G
    assert first["need_bytes"] == ST.v2_sequence()["sessions"][arm]["need_bytes"]       #: the plan's own figure
    for gib, ok in ((39.50, False), (40.5, True)):
        room["bytes"] = int(gib * G)
        got, msg = S.relay_room(L.VARIANT, L.LOOP_KIND, arm, {})
        assert got is ok, (gib, msg)
        assert asked[-1] == first["need_bytes"]                    #: margin included, never the bare peak
    #: a takeover that holds a valid save point: the rest of the rotation + the margin
    state["now"] = {unit: {"save_points": [1104]}}
    take = S.training_relay_need(arm, plan, state["now"])
    sp = S.checkpoint_bytes()["save_point"]
    assert take["need_bytes"] == sp + G                            #: the next save point beside the held one
    room["bytes"] = take["need_bytes"] - 1
    assert S.relay_room(L.VARIANT, L.LOOP_KIND, arm, {})[0] is False
    room["bytes"] = take["need_bytes"]
    assert S.relay_room(L.VARIANT, L.LOOP_KIND, arm, {})[0] is True


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

def _plans():
    from stages.depth_merge import plan as PL
    return {a: json.loads(PL.out_path(a).read_text()) for a in ("depth_loop_kept", "depth_loop_merged")}


def test_the_engine_gate_reads_each_check_against_the_control_and_needs_the_fault_to_fail():
    from stages.depth_recurrence import engine_check as EC
    kept = _plans()["depth_loop_kept"]
    assert good_record(kept)["status"] == "PASS"
    worse = good_record(kept, **{"metrics.vllm_generation.recurrent.top1_agreement": 0.9})
    assert worse["status"] == "FAIL" and not worse["checks"]["vllm_generation"]
    blind = good_record(kept, **{"metrics.hf_cache.shared_cache_fault.top1_agreement": 0.99,
                                 "metrics.hf_cache.shared_cache_fault.mean_abs_dlogp": 0.01})
    assert blind["status"] == "FAIL" and not blind["checks"]["sensitivity_fault_fails"]
    short = good_record(kept, **{"metrics.hf_cache.recurrent.cache_entries": 16})
    assert short["status"] == "FAIL"
    assert EC.record_problems(good_record(kept), kept, env=ENV) == []


@pytest.mark.parametrize("why,edit", [
    ("vLLM scored no position", {"metrics.vllm_generation.recurrent.positions": 0,
                                 "metrics.vllm_generation.recurrent.top1_agreement": None,
                                 "metrics.vllm_generation.recurrent.mean_abs_dlogp": None,
                                 "metrics.vllm_generation.recurrent.max_abs_dlogp": None}),
    ("both paths zero, which a relative gate would pass", {
        "metrics.vllm_generation.recurrent.positions": 0, "metrics.vllm_generation.control.positions": 0,
        "metrics.vllm_generation.recurrent.mean_abs_dlogp": 0.0, "metrics.vllm_generation.control.mean_abs_dlogp": 0.0}),
    ("a short vLLM output", {"vllm_outputs.recurrent.lengths": [64, 64, 64, 12]}),
    ("a missing output", {"vllm_outputs.control.n_outputs": 3}),
    ("a stop that is not the length rule", {"vllm_outputs.recurrent.finish_reasons": ["length", "stop", "length",
                                                                                      "length"]}),
    ("log-probabilities that do not match the tokens", {"vllm_outputs.recurrent.logprob_lengths": [64, 64, 63, 64]}),
    ("a non-finite log-probability", {"vllm_outputs.control.all_logprobs_finite": False}),
    ("a non-finite metric", {"metrics.hf_cache.control.mean_abs_dlogp": float("nan")}),
    ("a missing check", {"metrics.hf_cache.shared_cache_fault": None}),
])
def test_an_incomplete_or_empty_engine_result_is_never_a_pass(why, edit):
    from stages.depth_recurrence import engine_check as EC
    kept = _plans()["depth_loop_kept"]
    rec = good_record(kept, **edit)
    assert rec["status"] == "FAIL" and rec["checks"] == {"complete": False}, why
    forged = {**rec, "status": "PASS"}                          #: a PASS string over incomplete metrics
    assert EC.record_problems(forged, kept, env=ENV), why


@pytest.mark.parametrize("what", ["recurrent_config_sha256", "execution", "control_config_sha256", "inputs", "pack",
                                  "gate", "implementation", "check_code"])
def test_a_pass_bound_to_another_model_config_input_threshold_or_code_is_refused(what):
    from stages.depth_recurrence import engine_check as EC
    kept = _plans()["depth_loop_kept"]
    rec = good_record(kept)
    rec["binding"][what] = "another"
    assert any(what in x for x in EC.record_problems(rec, kept, env=ENV))
    #: the thresholds in force are the registered ones: a record whose gate differs is not this protocol's
    assert EC.record_problems(good_record(kept), kept, env={**ENV, "vllm": {"vllm": "0.28.0", "torch": "t2"}})


def test_one_pass_covers_both_initializations_but_not_another_schedule(tmp_path):
    """D's and M's weights differ by design: a PASS made on V2-D's init stands for V2-M's plan (same schedule, control,
    inputs, gate, implementation), while a plan with another schedule is refused."""
    from stages.depth_recurrence import engine_check as EC
    plans = _plans()
    rec = good_record(plans["depth_loop_kept"])
    assert rec["validated_weights_sha256"]["model.safetensors"] == \
        plans["depth_loop_kept"]["init_record"]["model_safetensors_sha256"]
    assert EC.record_problems(rec, plans["depth_loop_merged"], env=ENV) == []
    assert (plans["depth_loop_kept"]["init_record"]["model_safetensors_sha256"]
            != plans["depth_loop_merged"]["init_record"]["model_safetensors_sha256"])
    other = json.loads(json.dumps(plans["depth_loop_merged"]))
    other["units"][0]["initialization"]["recurrent_config"]["sha256"] = "1" * 64
    other["units"][0]["initialization"]["execution"]["layer_to_block"] = list(range(36))
    assert EC.record_problems(rec, other, env=ENV)
    #: a docstring edit leaves the implementation's identity alone; a statement edit does not
    src = REPO / EC.IMPLEMENTATION[0]
    root = tmp_path / "root"
    for rel in EC.IMPLEMENTATION:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    base = EC.source_identity(root=root)[EC.IMPLEMENTATION[0]]
    text = src.read_text()
    (root / EC.IMPLEMENTATION[0]).write_text(text.replace('"""', '"""Reworded. ', 1))
    assert EC.source_identity(root=root)[EC.IMPLEMENTATION[0]] == base
    (root / EC.IMPLEMENTATION[0]).write_text(text + "\nX_CHANGED = 1\n")
    assert EC.source_identity(root=root)[EC.IMPLEMENTATION[0]] != base


@pytest.mark.parametrize("why,edit", [
    ("a missing source", lambda r, P: r.pop("source")),
    ("another arm's session", lambda r, P: r["source"].update(arm="depth_loop_merged")),
    ("another unit", lambda r, P: r["source"].update(unit="depth_loop_kept-s20260801")),
    ("another init record", lambda r, P: r["source"].update(init_record_sha256="0" * 64)),
    ("another execution", lambda r, P: r["source"].update(execution_digest="0" * 64)),
    ("M's weights validated, not D's", lambda r, P: r.update(validated_weights_sha256={
        "model.safetensors": P["depth_loop_merged"]["init_record"]["model_safetensors_sha256"]})),
    ("no weights recorded", lambda r, P: r.pop("validated_weights_sha256")),
])
def test_a_record_not_made_in_the_registered_v2d_session_on_ds_init_is_refused(why, edit):
    from stages.depth_recurrence import engine_check as EC
    plans = _plans()
    for asked in ("depth_loop_kept", "depth_loop_merged"):                 #: the source decides for both uses
        rec = good_record(plans["depth_loop_kept"])
        edit(rec, plans)
        assert EC.source_problems(rec, plans[asked]) and EC.record_problems(rec, plans[asked], env=ENV), (why, asked)
    assert EC.source_problems(good_record(plans["depth_loop_kept"]), plans["depth_loop_merged"]) == []


def test_the_inputs_are_the_packs_content_and_the_prompts_are_rebuilt_from_it(tmp_path, monkeypatch):
    """The same data_dir with other content is another input; the prompts are not merely recorded: re-derived."""
    from stages.depth_recurrence import engine_check as EC
    plans = _plans()
    rec = good_record(plans["depth_loop_kept"])
    other = json.loads(json.dumps(plans["depth_loop_merged"]))
    other["units"][0]["data"]["files_sha256"]["blocks.npz"] = "1" * 64          #: same path, other bytes
    assert any("binding.pack" in x for x in EC.record_problems(rec, other, env=ENV))
    #: the pack on disk is re-hashed: a directory with the plan's file names but other bytes is refused
    fake = tmp_path / "pack"
    fake.mkdir()
    for n in plans["depth_loop_kept"]["units"][0]["data"]["files_sha256"]:
        (fake / n).write_bytes(b"other bytes")
    moved = json.loads(json.dumps(plans["depth_loop_kept"]))
    moved["units"][0]["config"]["data_dir"] = str(fake)
    assert len(EC.pack_problems(moved)) == 3
    #: the prompts: rebuilt and compared (the pack check stubbed here; the real one just above)
    seqs = [[1, 2, 3], [4, 5, 6]]
    monkeypatch.setattr(EC, "pack_problems", lambda plan, root=None: [])
    monkeypatch.setattr(EC, "pack_prompts", lambda u: seqs)
    rec = good_record(plans["depth_loop_kept"])        #: made under the patched probe (its code identity included)
    rec["binding"]["prompts_sha256"] = EC.prompts_digest(seqs)
    assert EC.input_problems(rec, plans["depth_loop_kept"]) == []
    assert EC.record_problems(rec, plans["depth_loop_merged"], env=ENV, verify_inputs=True) == []   #: D -> M
    rec["binding"]["prompts_sha256"] = EC.prompts_digest([[1, 2, 3], [4, 5, 7]])
    assert EC.input_problems(rec, plans["depth_loop_kept"])
    assert EC.record_problems(rec, plans["depth_loop_merged"], env=ENV, verify_inputs=True)


def test_the_real_pack_rehashes_to_the_plan_and_gives_the_registered_prompts():
    from stages.depth_recurrence import engine_check as EC
    kept = _plans()["depth_loop_kept"]
    if EC.pack_problems(kept):
        pytest.skip("the training pack is not on this machine")
    seqs = EC.pack_prompts(kept["units"][0])
    assert len(seqs) == EC.INPUTS["n_prompts"] and all(len(s) == EC.INPUTS["prompt_tokens"] + EC.INPUTS["new_tokens"]
                                                       for s in seqs)
    rec = good_record(kept)
    rec["binding"]["prompts_sha256"] = EC.prompts_digest(seqs)
    assert EC.input_problems(rec, kept) == []


def test_adopt_takes_only_the_registered_v2d_record(tmp_path):
    from stages.depth_recurrence import engine_check as EC
    plans = _plans()
    bad = good_record(plans["depth_loop_kept"])
    bad.pop("source")
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(bad))
    out = tmp_path / "adopted.json"
    with pytest.raises(EC.EngineCheckError, match="no source"):
        EC.adopt(p, "depth_loop_kept", dest=out)
    with pytest.raises(EC.EngineCheckError, match="first loop arm"):
        EC.adopt(p, "depth_loop_merged", dest=out)
    assert not out.exists()


def test_v2m_cannot_train_without_a_pass_that_stands_for_its_plan(tmp_path, monkeypatch):
    """On the dev box: the bound conditions and the launch gate refuse; on the pod: the driver trains nothing."""
    _eval_set_or_skip()
    from stages.depth_merge import plan as PL
    from stages.depth_merge import session as S
    from stages.depth_recurrence import launch as L
    from stages.teacher_parent_initialization import relay as R
    from stages.teacher_parent_initialization.tests.test_takeover import FakeRelay
    merged = json.loads(PL.out_path("depth_loop_merged").read_text())
    assert merged["engine_validation"]["mode"] == "reuse" and merged["engine_validation"]["from_arm"] == "depth_loop_kept"
    canon = REPO / merged["engine_validation"]["record"]
    if not canon.is_file():                                      #: today: V2-D has not run
        with pytest.raises(S.SE.AuthorizationError, match="does not exist"):
            L.kind_conditions(L.LOOP_KIND, arm="depth_loop_merged")
        assert S.engine_validation_problems("depth_loop_merged")
    assert S.engine_validation_problems("depth_loop_kept") == []        #: V2-D runs its own
    monkeypatch.setattr(R, "run_online", FakeRelay())
    for case in ("absent", "bound to another schedule", "incomplete"):
        sub = tmp_path / case.replace(" ", "_")
        _, plan = _plan_for(sub, "depth_loop_merged", monkeypatch)
        rec_p = Path(plan["engine_validation"]["record"])
        if case == "absent":
            rec_p.unlink()
        elif case == "bound to another schedule":
            rec = json.loads(rec_p.read_text())
            rec["binding"]["recurrent_config_sha256"] = "1" * 64
            rec_p.write_text(json.dumps(rec))
        else:
            rec = json.loads(rec_p.read_text())
            rec["metrics"]["vllm_generation"]["recurrent"]["positions"] = 0
            rec_p.write_text(json.dumps(rec))
        (sub / "plan_depth_loop_merged.json").write_text(json.dumps(plan))
        got = _run(sub, monkeypatch, "depth_loop_merged", sub / "a/store/out")
        assert got["rc"] != 0 and got["trained"] == [] and "no engine validation stands" in got["units"]["error"]["error"]
        assert "TRAIN_START" not in (sub / "status_depth_loop_merged.log").read_text()


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


def test_the_d_and_m_controls_are_the_registered_weights_whatever_path_they_are_read_from(monkeypatch):
    """The real D and M evidence passes; a replacement that is complete and self-consistent but bound to other weights
    -- through the default path or an explicit d_out / m_out -- is refused before anything is computed."""
    _eval_set_or_skip()
    from stages.depth_merge import analysis as A
    reg = A.registered_controls()
    assert reg["D"]["final_sha256"].startswith("5635c33e") and reg["M"]["final_sha256"].startswith("ed0ee3e1")
    real = {c: A.unit_sets_complete(o, reg[c]["arm"], 20260726)
            for c, o in (("D", DC.sob1_depth_evidence()), ("M", DC.merged_control_evidence()))}
    assert A.control_problems("D", real["D"]) == [] and A.control_problems("M", real["M"]) == []
    other = {**real["D"], "final_sha256": "e4159f1f" + "0" * 56,               #: SOB1 depth's seed-2 final, say
             "set_weights_sha256": {m: "e4159f1f" + "0" * 56 for m in A.MODES}}
    assert len(A.control_problems("D", other)) == 3
    half = {**real["M"], "set_weights_sha256": {**real["M"]["set_weights_sha256"], "H-N": "1" * 64}}
    assert A.control_problems("M", half) and "H-N" in A.control_problems("M", half)[0]
    real_usc = A.unit_sets_complete
    monkeypatch.setattr(A, "unit_sets_complete", lambda out, arm, seed: (
        other if arm == DC.SOB1_DEPTH else real_usc(out, arm, seed)))
    with pytest.raises(SystemExit, match="the D control"):
        A.recurrence(Path("/nonexistent/kept"), Path("/nonexistent/merged"), d_out=Path("/elsewhere/depth_only_copy"))
    with pytest.raises(SystemExit, match="the D control"):
        A.recurrence(Path("/nonexistent/kept"), Path("/nonexistent/merged"))


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
