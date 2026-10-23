"""H1's unit sessions at $0: the typed grant per kind, the cumulative envelope, the launcher, setup's branch, P1g's rules."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from stages.reasoning_data import assets as A
from stages.reasoning_data import p1g_driver as P
from stages.reasoning_data import unit_driver as UD
from stages.reasoning_data import unit_governance as UG

from aadistill.governance.authorization import AuthorizationError
from support.session_specs import load_session_launcher, session_args

L = load_session_launcher("h1_unit_launch")
VOLUME = ("--network-volume-id", "v", "--volume-mount-path", "/runpod-volume")


def test_each_kind_loads_only_under_its_own_policy(tmp_path):
    v = UG.issue("validation", "t_v", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "v.json")
    f = UG.issue("formal", "t_f", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "f.json")
    assert UG.load_for("validation", v).plan_hash == UG.units_hash() == UG.load_for("formal", f).plan_hash
    with pytest.raises(AuthorizationError):
        UG.load_for("formal", v)
    doc = json.loads(v.read_text())
    doc["stage_conditions"]["kind"] = "formal"          # a validation grant relabelled as formal
    (tmp_path / "forged.json").write_text(json.dumps(doc))
    with pytest.raises(AuthorizationError):
        UG.H1UnitAuthorization.load(tmp_path / "forged.json")
    a = UG.load_for("validation", v)
    assert a.allows("training") and a.allows("evaluation") and not a.allows("teacher_generation")
    assert not a.allows("formal")
    assert a.harness_source_digest and "src/aadistill/training/train.py" in a.harness_source_files


def test_the_envelope_is_cumulative_and_an_unknown_run_refuses(tmp_path, monkeypatch):
    env = UG.remaining_envelope()
    assert env["ceiling_usd"] == 50.0 and env["project_cap_usd"] == 712.17
    with pytest.raises(AuthorizationError, match="does not fit the envelope"):
        UG.issue("validation", "t_big", expected_usd=1.0, hard_cap_usd=env["remaining_usd"] + 0.01,
                 out=tmp_path / "big.json")
    root = tmp_path / "repo"
    run = root / UG.run_dir("u1")
    (run / "governance").mkdir(parents=True)
    (run / "runtime").mkdir()
    (run / "governance/authorization.json").write_text(json.dumps({"plan_id": "stage2.reasoning_data.h1_units"}))
    (run / "runtime/session.json").write_text(json.dumps({"provider_resource_created": True}))
    (root / UG.UNITS_GRANT).parent.mkdir(parents=True)
    (root / UG.UNITS_GRANT).write_text((A.REPO / UG.UNITS_GRANT).read_text())
    with pytest.raises(AuthorizationError, match="record no closeout cost"):
        UG.remaining_envelope(root)
    (run / "closeout").mkdir()
    (run / "closeout/outcome.json").write_text(json.dumps({"money": {"all_in_usd": 3.25}}))
    (root / UG.STORAGE).mkdir(parents=True)
    (root / UG.STORAGE / "vol.json").write_text(json.dumps({"volume_id": "vol", "reserved_usd": 4.2}))
    env = UG.remaining_envelope(root)
    assert env["remaining_usd"] == round(50.0 - 3.25 - 4.2, 4) and env["storage_usd"] == 4.2


def test_the_validation_spec_builds_and_formal_is_priced_only_from_a_passed_p1g(tmp_path):
    a = session_args(L, ("--phase", "P1g", *VOLUME))
    s = L.spec(a)
    s.validate()
    assert s.plan_hash == UG.units_hash() and s.budget.accounted_provider_resources == ("v",)
    f_auth = UG.load_for("formal", UG.issue("formal", "t_k", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "k.json"))
    ok, why = L.the_artifact_funds_this_phase(type("C", (), {"auth": f_auth, "args": a})())
    assert not ok and "cannot fund a P1g" in why
    assert "p1g_driver.py" in s.driver_command(type("C", (), {"args": a})(), None)
    with pytest.raises(ValueError, match="needs --measured"):
        L.spec(session_args(L, ("--phase", "P2", *VOLUME)))
    rep = {"passed": True, "checks": {
        "throughput_A": {"median_step_seconds": 10.5}, "throughput_B": {"median_step_seconds": 3.0},
        "evaluation_path": {"fluid": {"seconds": 60.0, "items": 20}, "capability": {"seconds": 40.0, "items": 8}}}}
    path = tmp_path / "p1g_report.json"
    path.write_text(json.dumps(rep))
    plan = UD.load_plan()
    t, e = L.unit_minutes(rep, plan, "A")
    fc = A.read_json(A.REPO / L.GL.FORECAST)
    assert t == round(plan["schedule"]["steps"] * 10.5 / 60, 1)
    # the slowest recorded full evaluation x the labelled reasoning factor, never the subset extrapolated
    assert e == round(fc["measured"]["eval_seconds_per_unit_L40S"]["max"]
                      * fc["contingency_factors_not_measurements"]["eval_floor_to_reasoning"] / 60, 1)
    f = L.spec(session_args(L, ("--phase", "P2", *VOLUME, "--measured", str(path), "--unit", "B-s20260726")))
    names = [ph.name for ph in f.budget.other_phases]
    assert names == ["B-s20260726_train", "B-s20260726_evaluate"]
    cmd = f.driver_command(type("C", (), {"args": f_args(path)})(), None)
    assert "unit_driver.py" in cmd and "MARKER:$m" in cmd and "--capacity-bytes" in cmd
    rep["passed"] = False
    path.write_text(json.dumps(rep))
    with pytest.raises(ValueError, match="did not pass"):
        L.spec(session_args(L, ("--phase", "P2", *VOLUME, "--measured", str(path))))


def f_args(path):
    return session_args(L, ("--phase", "P2", *VOLUME, "--measured", str(path), "--unit", "B-s20260726"))


def test_setups_branch_binds_the_artifact_on_the_pods_own_files(tmp_path):
    """The branch body, executed as setup executes it, against a real artifact."""
    text = (A.REPO / "scripts/shared/pod/autoinit_preflight_setup.sh").read_text()
    body = text[text.index('elif [ "$SESSION_KIND" = "stage2_h1_units" ]'):]
    code = re.search(r'/opt/train/bin/python -c "\n(.*?)\n" \|\|', body, re.S).group(1)
    code = code.replace('\\"', '"').replace("\\$", "$")
    v = UG.issue("validation", "t_s", expected_usd=1.0, hard_cap_usd=2.0, out=tmp_path / "a.json")
    env = {**os.environ, "PYTHONPATH": "src:scripts", "SESSION_AUTH_PATH": str(v), "SESSION_PLAN_HASH": UG.units_hash()}
    r = subprocess.run([sys.executable, "-c", code], cwd=A.REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "validation" in r.stdout, r.stderr[-800:]
    env["SESSION_PLAN_HASH"] = "0" * 64
    assert subprocess.run([sys.executable, "-c", code], cwd=A.REPO, env=env, capture_output=True).returncode != 0


def test_p1g_runs_the_frozen_recipe_with_only_the_log_interval_added(tmp_path):
    plan = UD.load_plan()
    d = P.recipe(plan, "A", steps=12, save_every=0, out_dir=tmp_path, pack_root=tmp_path, init=tmp_path, seed=1,
                 name="n")
    assert d["batch"]["blocks_per_step"] == 7 and d["loss"] == plan["recipe"]["loss"]
    assert d["optim"] == plan["recipe"]["optim"] and d["intervals"]["log_every"] == 1


def test_p1g_resume_rule_needs_identical_data_and_lr_and_a_loss_inside_the_gpu_floor():
    ref = {s: {"lr": 1e-5 * s, "executed_positions": 100, "executed_nonpad_tokens": 90, "supervised_tokens": 80,
               "kd_positions": 85, "loss": 2.0, "ce": 1.0, "kd": 1.5} for s in range(13, 25)}
    same = P.compare(ref, json.loads(json.dumps(ref, default=str).replace('"13"', '"13"')) and
                     {int(k): v for k, v in ref.items()}, range(13, 25))
    assert same["exact_mismatch"] == [] and same["max_rel_loss"] == 0.0
    moved = {s: dict(r) for s, r in ref.items()}
    moved[20]["executed_positions"] = 99                       # another block came next: not a resume
    moved[21]["loss"] = 2.002
    got = P.compare(ref, moved, range(13, 25))
    assert got["exact_mismatch"][0]["step"] == 20 and got["max_rel_loss"] == pytest.approx(0.001)
    assert P.compare(ref, {13: ref[13]}, range(13, 15))["exact_mismatch"][0]["missing"] == ["other"]
