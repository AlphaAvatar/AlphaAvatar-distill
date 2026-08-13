"""Zero-cost rehearsal of the micro-preflight harness: every failure path.

The preflight spends money on exactly one thing — two permanent control runs —
and the whole point of the staging is that a bad machine never reaches them. That
property is worth nothing unless it has been *executed*, so this drives the real
`Driver` through every scenario with the expensive calls stubbed, and asserts, for
each blocking failure:

    later stages are not entered
    permanent controls are not trained when Stage 0/1 fails
    the evidence file is still written
    a terminal marker is emitted so the launcher tears down

plus the one property the success path must have: it cannot fall through into
Phase A.

Scenarios (the maintainer's list):

    stage-0 attestation failure          stage-1 repeatability failure
    stage-1 memory failure               stage-1 storage failure
    control training failure             protocol fingerprint mismatch
    generation/scoring failure           missing required artifact
    driver crash/hang                    watchdog hard-cap path
    successful stage 0 -> 3
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.authorization import (  # noqa: E402
    AuthorizationError, SpendAuthorization,
)
from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1  # noqa: E402

DRIVER_PATH = REPO / "scripts/pod/autoinit_preflight_driver.py"
LAUNCH_PATH = REPO / "scripts/pod/autoinit_preflight_launch.py"
AUTH_PATH = REPO / "logs/autoinit_micro_preflight_authorization.json"

pytestmark = pytest.mark.skipif(
    not (REPO / "artifacts/stage3/recovery_search_v1/manifest.json").is_file(),
    reason="frozen assets are local artifacts, not tracked in git")


def load_driver_module(tmp_path: Path):
    """Import the driver with its pod paths redirected into a temp tree."""
    spec = importlib.util.spec_from_file_location("preflight_driver", DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["preflight_driver"] = mod
    spec.loader.exec_module(mod)
    mod.WS = tmp_path
    mod.STATUS = tmp_path / "preflight.status"
    mod.AUDIT = tmp_path / "audit"
    mod.AUDIT.mkdir(parents=True, exist_ok=True)
    return mod


class Args:
    stage = "all"
    image_digest = "sha256:rehearsal"
    rate = 0.99
    spent_usd = 0.30
    soft_stop_usd = 6.00
    authorized_usd = 8.60
    repeats = 10
    control_minutes = 85.0
    characterization_minutes = 18.0
    disk_probe_gib = 0.01


CAP = {"gsm8k": 30, "math_verified": 30, "multihop": 30, "rag": 30,
       "knowledge": 30, "tool": 20}


def fake_result(usable_rate: float, correct_rate: float, seed: int) -> dict:
    """A scored recovery-search result of the shape Stage 3 consumes."""
    per_cap = {}
    for name, n in CAP.items():
        usable = int(round(n * usable_rate))
        per_cap[name] = {"n": n, "usable": usable,
                         "correct": int(round(usable * correct_rate)),
                         "usable_rollout_rate": round(usable / n, 4),
                         "correct_overall": correct_rate}
    n = 190
    usable = int(round(n * usable_rate))
    return {"n": n, "usable": usable, "n_scorable": 170,
            "correct": int(round(170 * usable_rate * correct_rate)),
            "usable_rollout_rate": usable_rate,
            "correct_overall": round(usable_rate * correct_rate, 4),
            "correct_given_usable": correct_rate,
            "per_capability": per_cap, "seed": seed,
            "scoring_contract": {"digest": "DIGEST"}}


def build(tmp_path, *, stage0=True, gates=None, controls_ok=True,
          protocol_ok=True, stage3="ok"):
    """A driver with every paid call replaced by a scripted outcome."""
    mod = load_driver_module(tmp_path)
    d = mod.Driver.__new__(mod.Driver)
    d.a = Args()
    d.t0 = __import__("time").time()
    d.results = {}
    d.ev = {"stages": {}}
    d.auth = SpendAuthorization.load(AUTH_PATH)
    d.attested = None
    d.evaluation_protocol = None
    d.trained = []
    d.mod = mod

    gates = gates or {}

    def stage0_impl():
        d.enter(0)
        if not stage0:
            return d.record(0, False, "pinned input does not match its hash")

        class P:
            fingerprint = "ATTESTED"
        d.attested = P()

        class E:
            scoring_digest = "DIGEST"
            evaluation_protocol_hash = "EVAL"
        d.evaluation_protocol = E()
        return d.record(0, True, attested_protocol_fingerprint="ATTESTED",
                        evaluation_protocol_hash="EVAL")

    def stage1_impl():
        d.enter(1)
        rep = gates.get("repeatability_range", 1e-6)
        peak = gates.get("peak_gib", 14.3)
        write = gates.get("write_mb_s", 900.0)
        failures = []
        if rep >= 1e-4:
            failures.append(f"evaluator repeatability range {rep:.3g} >= epsilon")
        if peak > 40.0:
            failures.append(f"peak resident {peak:.1f} GiB > 40 GiB")
        if write < 50:
            failures.append(f"disk write {write:.0f} MB/s infeasible")
        return d.record(1, not failures, "; ".join(failures),
                        gates={"evaluator_repeatability": {"max_objective_range": rep},
                               "peak_memory": {"peak_gib": peak},
                               "disk_throughput": {"write_mb_s": write}})

    def stage2_impl():
        d.enter(2)
        arms = {}
        for name, seed, _ in mod.CONTROLS:
            if not d.afford(d.a.control_minutes, f"control {name}"):
                return d.record(2, False, f"insufficient budget for {name}", arms=arms)
            d.trained.append(name)
            if not controls_ok:
                arms[name] = {"trained": False, "rc": 1}
                return d.record(2, False, f"{name} failed rc=1", arms=arms)
            problem = "" if protocol_ok else "ran a different protocol"
            arms[name] = {"trained": True, "seed": seed,
                          "checkpoint": f"artifacts/stage3/{name}/checkpoints/step_001023",
                          "weights_sha256": "a" * 64, "probe_id": "p" * 64,
                          "required_artifacts": {"run_manifest.json": True,
                                                 "train_log.jsonl": True},
                          "protocol_verified": protocol_ok, "problem": problem}
            if problem:
                return d.record(2, False, f"{name}: {problem}", arms=arms)
        return d.record(2, True, arms=arms)

    def stage3_impl():
        d.enter(3)
        if stage3 == "generation_failed":
            return d.record(3, False, "preflight_ctl_r0860k_sa generation rc=1")
        if stage3 == "contract_drift":
            return d.record(3, False, "scored under a different scoring contract")
        from aadistill.autoinit.recovery import (
            CATASTROPHIC_V1, POOLED_COUNTS_V1, EquivalenceRule, FeasibilityRule)
        sa, sb = fake_result(0.62, 0.31, mod.SEED_SA), fake_result(0.58, 0.29, mod.SEED_SB)
        pooled = POOLED_COUNTS_V1.pool([
            {"seed": mod.SEED_SA, "n": sa["n"], "usable": sa["usable"],
             "correct": sa["correct"]},
            {"seed": mod.SEED_SB, "n": sb["n"], "usable": sb["usable"],
             "correct": sb["correct"]}])
        eq = EquivalenceRule(n_pooled=340).materialize(
            p_pool=pooled["correct_overall"], p_sa=sa["correct_overall"],
            p_sb=sb["correct_overall"]).as_dict()
        fl = FeasibilityRule(n_pooled=380).materialize(
            u_pool=pooled["usable_rollout_rate"], u_sa=sa["usable_rollout_rate"],
            u_sb=sb["usable_rollout_rate"]).as_dict()
        thresholds = {"pooled": pooled, "equivalence_interval": eq,
                      "feasibility_floor": fl,
                      "catastrophic_rule": CATASTROPHIC_V1.as_dict(),
                      "per_capability_control_baseline": {
                          c: {"pooled_usable_rate": round(
                              (sa["per_capability"][c]["usable"]
                               + sb["per_capability"][c]["usable"])
                              / (2 * CAP[c]), 4)} for c in CAP}}
        (mod.AUDIT / "materialized_thresholds.json").write_text(
            json.dumps(thresholds, indent=2))
        return d.record(3, True, thresholds=thresholds)

    d.stage0, d.stage1, d.stage2, d.stage3 = (
        stage0_impl, stage1_impl, stage2_impl, stage3_impl)
    d.enter = lambda s: mod.Driver.enter(d, s)
    d.record = lambda *args, **kw: mod.Driver.record(d, *args, **kw)
    d.usd = lambda: mod.Driver.usd(d)
    d.afford = lambda m, w: mod.Driver.afford(d, m, w)
    d.save = lambda: mod.Driver.save(d)
    d.run = lambda: mod.Driver.run(d)
    return d, mod


def markers(mod) -> list[str]:
    if not mod.STATUS.is_file():
        return []
    return [line.split("MARKER:")[1] for line in
            mod.STATUS.read_text().splitlines() if "MARKER:" in line]


# --- blocking failures ------------------------------------------------------


@pytest.mark.parametrize("name,kwargs,expect_stage", [
    ("stage0_attestation_failure", dict(stage0=False), 0),
    ("stage1_repeatability_failure", dict(gates={"repeatability_range": 3e-4}), 1),
    ("stage1_memory_failure", dict(gates={"peak_gib": 46.0}), 1),
    ("stage1_storage_failure", dict(gates={"write_mb_s": 12.0}), 1),
])
def test_a_blocking_gate_never_reaches_the_permanent_controls(
        tmp_path, name, kwargs, expect_stage):
    d, mod = build(tmp_path, **kwargs)
    rc = d.run()

    assert rc == 20 + expect_stage, name
    assert d.trained == [], (
        f"{name}: permanent controls were trained after a blocking failure")
    assert 2 not in d.results and 3 not in d.results, f"{name}: a later stage ran"
    assert f"STAGE_FAILED:{expect_stage}" in markers(mod)
    assert "PREFLIGHT_FAILED" in markers(mod), (
        f"{name}: no terminal marker, so the launcher would poll until its limit")
    evidence = json.loads((mod.AUDIT / "preflight_evidence.json").read_text())
    assert evidence["stages"][str(expect_stage)]["passed"] is False
    assert evidence["stages"][str(expect_stage)]["reason"], "no reason recorded"


def test_the_plan_itself_refuses_stage_2_after_a_failed_gate(tmp_path):
    """Belt and braces: even if a caller skipped the check, the plan refuses."""
    from aadistill.autoinit.recovery import RecoveryAdmissionError

    with pytest.raises(RecoveryAdmissionError, match="did not pass"):
        PREFLIGHT_PLAN_V1.advance_to(
            2, {0: {"passed": True}, 1: {"passed": False, "reason": "peak memory"}})
    with pytest.raises(RecoveryAdmissionError, match="no recorded result"):
        PREFLIGHT_PLAN_V1.advance_to(2, {0: {"passed": True}})


# --- stage 2 and 3 failures -------------------------------------------------


def test_a_control_training_failure_stops_before_the_second_arm(tmp_path):
    d, mod = build(tmp_path, controls_ok=False)
    assert d.run() == 22
    assert d.trained == ["preflight_ctl_r0860k_sa"], (
        "the second control was started after the first failed")
    assert 3 not in d.results
    assert "PREFLIGHT_FAILED" in markers(mod)


def test_a_protocol_fingerprint_mismatch_fails_the_control(tmp_path):
    d, mod = build(tmp_path, protocol_ok=False)
    assert d.run() == 22
    arms = d.ev["stages"]["2"]["arms"]
    assert arms["preflight_ctl_r0860k_sa"]["protocol_verified"] is False
    assert 3 not in d.results, (
        "a control that did not run the attested protocol was characterized anyway")


@pytest.mark.parametrize("mode", ["generation_failed", "contract_drift"])
def test_a_generation_or_scoring_failure_is_recorded_and_stops(tmp_path, mode):
    d, mod = build(tmp_path, stage3=mode)
    rc = d.run()
    assert rc == 23
    assert d.results[3]["passed"] is False
    assert d.results[2]["passed"] is True, "the controls still exist and are kept"
    assert (mod.AUDIT / "preflight_evidence.json").is_file()
    # Stage 3 is non-blocking, so nothing earlier is invalidated -- but the
    # session is not complete and must not claim it is.
    assert "ALL_DONE" not in markers(mod), (
        "a failed characterization emitted the success marker; the launcher "
        "would gate on the full artifact spec and report a complete preflight")
    assert "PREFLIGHT_INCOMPLETE" in markers(mod)


def test_a_missing_required_artifact_blocks_teardown_not_the_watchdog():
    """The gate refuses; the watchdog is unaffected, which is the whole design."""
    from aadistill.infrastructure.artifact_gate import evaluate_teardown

    state = {"training_complete": True, "evaluation_complete": True,
             "artifact_manifest_created": True, "required_files_present": False,
             "final_streams_quiescent": True}
    decision = evaluate_teardown(state)
    assert not decision.allowed
    assert decision.failed_check
    # And the emergency path still allows teardown, with the loss recorded.
    emergency = evaluate_teardown(
        state, emergency_budget=True, emergency_reason="hard threshold",
        incomplete_event_streams=("artifacts/stage3/x/train_log.jsonl",))
    assert emergency.allowed
    assert "hard threshold" in json.dumps(emergency.as_dict())


# --- budget and authorization ----------------------------------------------


def test_the_soft_stop_refuses_a_control_it_cannot_finish(tmp_path):
    d, mod = build(tmp_path)
    d.a.spent_usd = 5.60          # a control is ~85 min = ~$1.40
    rc = d.run()
    assert rc == 22
    assert d.trained == [], "a control was started that could not finish in budget"
    assert "insufficient budget" in d.results[2]["reason"]


def test_the_authorization_bounds_the_session(tmp_path):
    auth = SpendAuthorization.load(AUTH_PATH)
    auth.require_plan(PREFLIGHT_PLAN_V1.plan_hash)
    assert auth.hard_cap_usd == 8.60 and auth.expected_usd == 4.20
    assert auth.authorized_stages == (0, 1, 2, 3)
    with pytest.raises(AuthorizationError):
        auth.require_stage(4)
    with pytest.raises(AuthorizationError):
        auth.require_within_cap(8.61, what="session")
    with pytest.raises(AuthorizationError, match="separately unauthorized"):
        auth.refuse_phase_a()
    assert auth.allows_phase_a is False
    assert auth.automatic_phase_a_start is False


def test_an_unrehearsed_harness_cannot_consume_the_authorization(tmp_path):
    """The executable identity is enforced, not merely recorded."""
    from aadistill.autoinit.authorization import harness_source_digest

    auth = SpendAuthorization.load(AUTH_PATH)
    observed = auth.require_harness(REPO)          # the committed harness passes
    assert observed["digest"] == auth.harness_source_digest
    assert auth.authorized_session_commit

    # An edited harness is a different harness.
    from dataclasses import replace
    edited = replace(auth, harness_source_digest="0" * 64)
    with pytest.raises(AuthorizationError, match="differ"):
        edited.require_harness(REPO)
    # An authorization that names no harness cannot authorize an executable.
    with pytest.raises(AuthorizationError, match="no harness_source_digest"):
        replace(auth, harness_source_digest=None).require_harness(REPO)
    # A missing declared file raises rather than shrinking the digest.
    with pytest.raises(AuthorizationError, match="is missing"):
        harness_source_digest(REPO, files=("scripts/pod/watchdog.py",
                                           "scripts/pod/does_not_exist.py"))


def test_the_launcher_checks_the_harness_before_a_pod_can_exist():
    source = LAUNCH_PATH.read_text()
    assert "self.auth.require_harness(REPO_ROOT)" in source
    # In __init__, i.e. before make_plan/create are ever called.
    assert source.index("require_harness") < source.index("def make_plan")


def test_every_engine_observed_generation_field_is_required():
    """A field cannot be part of the comparison and allowed to stay null."""
    from aadistill.autoinit.generation import declared_generation_protocol

    required = declared_generation_protocol().MATERIALIZATION_REQUIRED
    for field in ("max_num_seqs", "max_num_batched_tokens", "enforce_eager",
                  "context_source", "vllm_version", "stop_token_ids"):
        assert field in required, field


def test_an_authorization_bound_to_a_different_plan_is_refused():
    auth = SpendAuthorization.load(AUTH_PATH)
    with pytest.raises(AuthorizationError, match="does not transfer"):
        auth.require_plan("0" * 64)


def test_a_tampered_authorization_does_not_load(tmp_path):
    raw = json.loads(AUTH_PATH.read_text())
    raw["hard_cap_usd"] = 100.0
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(AuthorizationError, match="authorization_sha256"):
        SpendAuthorization.load(path)
    # And one that claims Phase A is refused even if its hash is consistent.
    from aadistill.infrastructure.manifest import sha256_json
    raw = json.loads(AUTH_PATH.read_text())
    raw["hard_cap_usd"] = 8.60
    raw["phase_a_authorized"] = True
    raw.pop("authorization_sha256")
    raw["authorization_sha256"] = sha256_json(raw)
    path.write_text(json.dumps(raw))
    with pytest.raises(AuthorizationError, match="cannot grant"):
        SpendAuthorization.load(path)


# --- driver liveness and the watchdog ---------------------------------------


def test_the_launcher_treats_a_dead_or_hung_driver_as_terminal():
    source = LAUNCH_PATH.read_text()
    assert 'live, _ = probe(target, job)' in source
    assert 'terminal = f"DRIVER_{live}"' in source
    # It polls the PROVIDER, not the log: a wedged driver writes no lines and
    # silence must not look like an idle session.
    assert "state = self.provider.get(self.pod_id)" in source
    assert 'terminal = "POD_GONE"' in source
    # The watchdog starts immediately after creation, before setup.
    create_at = source.index("outcome = self.setup_on_draw(draw)")
    watchdog_at = source.index("self.launch_watchdog()")
    assert watchdog_at < create_at, (
        "the watchdog must be running before the long setup, or a wedged setup "
        "has no cost backstop")


def test_the_watchdog_hard_cap_path_terminates_and_verifies(tmp_path):
    """Rehearse the watchdog's own thresholds with no pod (its --simulate path)."""
    import subprocess
    rc = subprocess.run(
        [sys.executable, str(REPO / "scripts/pod/watchdog.py"), "--simulate",
         "--pod-id", "rehearsal", "--session-start-epoch", "0",
         "--price-per-hour", "0.99", "--hard-minutes", "300",
         "--authorized-usd", "8.60", "--journal", str(tmp_path / "wd.jsonl")],
        capture_output=True, text=True, cwd=REPO, timeout=300,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert rc.returncode == 0, rc.stdout + rc.stderr
    # Either verdict proves the hard-cap path executed: it either terminated the
    # pod or found it already gone. What must never happen is a verdict of
    # "idle" -- `SessionWatcher.assess` has no such value by design.
    verdict = (rc.stdout + rc.stderr).upper()
    assert "TERMINATE" in verdict or "POD_GONE" in verdict, verdict
    assert "IDLE" not in verdict


# --- the success path -------------------------------------------------------


def test_the_success_path_completes_and_cannot_reach_phase_a(tmp_path):
    d, mod = build(tmp_path)
    assert d.run() == 0
    assert sorted(d.results) == [0, 1, 2, 3]
    assert all(r["passed"] for r in d.results.values())
    assert d.trained == list(mod.CONTROLS[i][0] for i in (0, 1))
    assert "ALL_DONE" in markers(mod)

    evidence = json.loads((mod.AUDIT / "preflight_evidence.json").read_text())
    assert evidence["phase_a_started"] is False
    assert evidence["phase_a_reachable_from_this_driver"] is False

    thresholds = json.loads((mod.AUDIT / "materialized_thresholds.json").read_text())
    assert thresholds["equivalence_interval"]["value"] > 0
    assert thresholds["feasibility_floor"]["value"] >= 0.30
    assert set(thresholds["per_capability_control_baseline"]) == set(CAP)
    assert thresholds["pooled"]["aggregation"] == "pooled_counts@v1"


def test_phase_a_is_not_expressible_in_the_driver_or_the_launcher():
    driver = DRIVER_PATH.read_text()
    launch = LAUNCH_PATH.read_text()
    assert 'choices=("all", "0", "1", "2", "3")' in driver, (
        "the driver accepts a stage outside the authorized set")
    # Nothing in either file can start a search: no beam, no leaf admission, no
    # probe scheduling. `phase_a_protocol` is a *shape* the control also runs and
    # is not a launch path.
    for source, who in ((driver, "driver"), (launch, "launcher")):
        for forbidden in ("BeamSearch", "admit_leaves", "probe_configs",
                          "SuccessiveHalvingPlan", "run_phase_a", "start_phase_a"):
            assert forbidden not in source, f"the {who} can reach {forbidden}"
    assert "PARETO_V1" not in launch
