"""Exactly one observation may be purchased, and it is `fe9683e6a9c7/sc`.

After Attempt 4 the scientific inventory is:

* every `sa` — retained, all six evidence candidates;
* every `sb` — retained, including the one Attempt 4 paid for;
* `85bde4ded2c3/sc` — retained from the Phase-A continuation;
* `fe9683e6a9c7/sc` — **missing**, and the only thing still owed.

A dollar ceiling does not encode that. `$5.4784` funds one probe of any kind, so
a session that bought a replacement `sb` instead of the owed `sc` would stay
inside budget and report success. Three independent statements of the scope have
to agree instead — the pricing artifact, the launcher's booked probes, and the
driver's purchase whitelist — and the driver refuses at the purchase seam itself.

A retained observation that has gone missing is a **corrupted-evidence**
condition, not a reason to regenerate it: a replacement probe is a different
measurement from the one the corrected rung-2 decision was computed over.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

PROBES = REPO / "logs/autoinit_continuation_b_attempt4/probes"
PRICING = REPO / "logs/autoinit_behavioural_continuation_pricing.json"
ATTEMPT4_REUSE = REPO / "logs/autoinit_attempt4_probe_reuse.json"

FE, BD = "fe9683e6a9c7", "85bde4ded2c3"
SEED_SC = 20260813


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/pod" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drv():
    return load("continuation_b_driver_1p", "autoinit_continuation_b_driver.py")


@pytest.fixture(scope="module")
def launcher():
    return load("continuation_b_launch_1p", "autoinit_continuation_b_launch.py")


def descriptor(candidate: str, rung: int, seed: int):
    suffix = {1: "sa", 2: "sb", 3: "sc"}[rung]
    return {"probe_id": f"autoinit.v1.phase_a.rung{rung}.{candidate}.{suffix}",
            "rung": rung, "seed": seed,
            "student_artifact_digest": "0" * 64}


# --- 1. the attempt-4 reuse digest is AUTHORIZATION-bound -------------------

def test_the_attempt4_reuse_digest_is_bound_evidence(drv):
    from aadistill.autoinit.phase_b_continuation import BOUND_EVIDENCE

    assert "attempt4_reuse_probes_dir_digest" in BOUND_EVIDENCE, (
        "Attempt 4's probe is a necessary citation — without it the session has "
        "no complete sa+sb — and an unbound record can be edited between "
        "issuance and execution")
    observed = drv.ContinuationDriver.observed_evidence()
    assert set(observed) == set(BOUND_EVIDENCE)
    assert observed["attempt4_reuse_probes_dir_digest"] == json.loads(
        ATTEMPT4_REUSE.read_text())["probes_dir_digest"]


def test_a_moved_attempt4_digest_fails_before_any_probe(drv, monkeypatch, tmp_path):
    """`require_evidence` is what stage 0 and the launcher's evidence gate call.
    A moved record must be refused there, not discovered on a pod."""
    from aadistill.autoinit.authorization import AuthorizationError
    from aadistill.autoinit.phase_b_continuation import ContinuationAuthorization

    observed = dict(drv.ContinuationDriver.observed_evidence())
    auth = ContinuationAuthorization(
        authorization_id="t", granted_utc="2026-08-31T00:00:00Z", granted_by="t",
        plan_id="p", plan_hash="h", science_plan_hash="s",
        calibration_profile_hashes={}, calibration_content_hashes={},
        bound_evidence=dict(observed), planning_floor_usd=4.183,
        hard_cap_usd=5.4784, per_launch_hard_usd=5.4784,
        authorized_stages=(0, 1, 3, 4, 5), stage_conditions={}, scope_note="t",
        source_digest="0" * 64)
    auth.require_evidence(observed)                      # unchanged -> accepted

    moved = dict(observed)
    moved["attempt4_reuse_probes_dir_digest"] = "f" * 64
    with pytest.raises(AuthorizationError) as exc:
        auth.require_evidence(moved)
    assert "attempt4_reuse_probes_dir_digest" in str(exc.value)


def test_the_issuer_and_preregistration_both_carry_it():
    prereg = json.loads(
        (REPO / "logs/autoinit_continuation_b_preregistration.json").read_text())
    record = prereg["reuse_rule"]["attempt4_record"]
    assert record["probes_dir_digest"] == json.loads(
        ATTEMPT4_REUSE.read_text())["probes_dir_digest"]
    assert record["admitted"] == ["fe9683e6a9c7/sb"]


# --- 2 & 3. retained evidence is reuse-only, and fails closed ---------------

@pytest.mark.parametrize("candidate,rung,seed,what", [
    (FE, 2, 20260801, "the sb Attempt 4 purchased"),
    (BD, 2, 20260801, "a retained sb"),
    (FE, 1, 20260726, "a retained sa"),
    (BD, 3, SEED_SC, "the retained 85bde sc"),
])
def test_a_missing_retained_observation_fails_closed(drv, candidate, rung, seed,
                                                     what):
    """Never a replacement purchase. `require_purchasable` is reached only when
    nothing could be cited — which is exactly the missing/corrupt case."""
    from aadistill.autoinit.recovery import RecoveryAdmissionError

    d = drv.ContinuationDriver.__new__(drv.ContinuationDriver)
    with pytest.raises(RecoveryAdmissionError) as exc:
        drv.ContinuationDriver.require_purchasable(
            d, descriptor(candidate, rung, seed))
    message = str(exc.value)
    assert "may not buy it" in message, what
    assert "fe9683e6a9c7/rung3" in message


# --- 4. the one owed observation IS purchasable ----------------------------

def test_the_owed_sc_is_the_only_purchasable_descriptor(drv):
    d = drv.ContinuationDriver.__new__(drv.ContinuationDriver)
    # Permitted, and returns without raising.
    drv.ContinuationDriver.require_purchasable(d, descriptor(FE, 3, SEED_SC))

    assert drv.ContinuationDriver.PURCHASABLE == ((FE, 3),), (
        "the purchase whitelist is not exactly the one owed observation")


def test_the_purchase_seam_is_the_only_route_to_training(drv):
    """`probe_config` is called by the inherited `run_probe` on the line after it
    decides nothing could be restored, and nothing else calls it. Checking there
    binds the scope to the act of BUYING rather than to a count or a budget."""
    import inspect

    import autoinit_phase_a_driver as parent

    run_probe = inspect.getsource(parent.PhaseADriver.run_probe)
    assert "restored = self.restore_probe(descriptor)" in run_probe
    assert "if restored is not None:\n            return restored" in run_probe
    assert "config = self.probe_config(descriptor)" in run_probe

    override = inspect.getsource(drv.ContinuationDriver.probe_config)
    assert "require_purchasable" in override

    # And no other caller exists anywhere in the pod scripts.
    callers = [p for p in (REPO / "scripts/pod").glob("*.py")
               if "self.probe_config(" in p.read_text()]
    assert [p.name for p in callers] == ["autoinit_phase_a_driver.py"], callers


# --- 5. widening the runtime scope fails the pre-provider gate --------------

def test_the_workload_scope_gate_accepts_the_one_probe_contract(launcher):
    from aadistill.infrastructure.session import SessionContext

    args = launcher.build_parser().parse_args(
        ["--session-commit", "0" * 40, "--bundle", "b"])
    assert args.rung2_probes == 0 and args.tie_break_probes == 1
    assert launcher.workload_scope_gate in launcher.spec(args).precheck

    ctx = SessionContext(scr=Path("/tmp/x"), args=args, auth=None, evidence={},
                         say=lambda m: None)
    ok, why = launcher.workload_scope_gate(ctx)
    assert ok, why
    scope = ctx.evidence["precheck"]["workload_scope"]
    assert scope["priced_hard_probes"] == 1 and scope["booked_probes"] == 1
    assert scope["missing_sb"] == [] and scope["missing_sc"] == [FE]


@pytest.mark.parametrize("field,value", [("rung2_probes", 1),
                                         ("tie_break_probes", 2),
                                         ("tie_break_probes", 0)])
def test_widening_or_narrowing_the_booked_scope_is_refused(launcher, field, value):
    from aadistill.infrastructure.session import SessionContext

    args = launcher.build_parser().parse_args(
        ["--session-commit", "0" * 40, "--bundle", "b"])
    setattr(args, field, value)
    ctx = SessionContext(scr=Path("/tmp/x"), args=args, auth=None, evidence={},
                         say=lambda m: None)
    ok, why = launcher.workload_scope_gate(ctx)
    assert not ok, f"{field}={value} was accepted"
    assert "probe" in why


def test_a_whitelist_that_names_the_wrong_probe_is_refused(launcher, drv,
                                                           monkeypatch):
    """The count can be right and the science still wrong."""
    from aadistill.infrastructure.session import SessionContext

    import autoinit_continuation_b_driver as live

    monkeypatch.setattr(live.ContinuationDriver, "PURCHASABLE", ((BD, 3),))
    args = launcher.build_parser().parse_args(
        ["--session-commit", "0" * 40, "--bundle", "b"])
    ctx = SessionContext(scr=Path("/tmp/x"), args=args, auth=None, evidence={},
                         say=lambda m: None)
    ok, why = launcher.workload_scope_gate(ctx)
    assert not ok and "different observation" in why


# --- the session plan says what the session does ---------------------------

def test_the_session_plan_describes_the_one_probe_scope():
    from aadistill.autoinit.phase_b_continuation import CONTINUATION_PLAN_V1

    assert CONTINUATION_PLAN_V1.version == 3, (
        "version 2 described 'one missing sb and at most two conditional sc', "
        "which is no longer the scientific state")
    stages = {s.stage: s for s in CONTINUATION_PLAN_V1.stages}
    assert "REUSE ONLY" in stages[3].name
    assert "fe9683e6a9c7/sc" in stages[4].name
    conditions = " ".join(stages[4].stop_conditions)
    assert "at most ONE descriptor" in conditions
    assert "no fourth seed" in conditions


# --- the committed artifacts must describe the committed source -------------
#
# The metadata-coherence repair found that the committed preregistration bound
# executable digest `20c37deb…` while the source tree produced `a5ce6311…`. The
# cause was mundane: the preregistration was regenerated, two further comment
# edits were made to source files in the set, and everything was committed
# together. Nothing checked the two against each other.
#
# `continuation_source_gate` would not have caught it — it compares the GRANT's
# digest to the live source, and the issuer computes the grant's digest fresh.
# The stale value was only ever in the preregistration.

def test_the_preregistration_binds_the_live_executable_digest():
    from aadistill.autoinit.phase_b_continuation import continuation_source_digest

    prereg = json.loads(
        (REPO / "logs/autoinit_continuation_b_preregistration.json").read_text())
    recorded = prereg["executable_source"]["digest"]
    live = continuation_source_digest(REPO)["digest"]
    assert recorded == live, (
        f"the preregistration binds {recorded[:12]}… but the source tree digests "
        f"to {live[:12]}…. Regenerate it: an edit to a file in the source set "
        "after the last regeneration leaves the document describing code that no "
        "longer exists.")
    assert prereg["executable_source"]["n_files"] == len(
        continuation_source_digest(REPO)["files"])


def test_the_preregistration_binds_the_live_session_plan_and_pricing():
    from aadistill.autoinit.phase_b_continuation import CONTINUATION_PLAN_V1

    prereg = json.loads(
        (REPO / "logs/autoinit_continuation_b_preregistration.json").read_text())
    priced = json.loads(PRICING.read_text())

    assert prereg["session_plan"]["plan_hash"] == CONTINUATION_PLAN_V1.plan_hash
    assert prereg["session_plan"]["version"] == CONTINUATION_PLAN_V1.version == 3
    assert prereg["budget"]["floor_usd"] == priced["total"]["low_usd"] == 4.1830
    assert prereg["budget"]["hard_ceiling_usd"] == priced["total"]["hard_usd"] \
        == 5.4784
    assert prereg["probe_inventory"]["new_probes_max"] == \
        priced["total"]["hard_probes"] == 1


def test_the_preregistration_states_the_current_scientific_state():
    """No stale V2 narrative alongside V3 fields."""
    prereg = json.loads(
        (REPO / "logs/autoinit_continuation_b_preregistration.json").read_text())
    blob = json.dumps(prereg)

    assert "at most two conditional sc" not in blob, (
        "the preregistration still describes the pre-Attempt-4 scope")
    assert "one missing sb" not in blob

    corrected = prereg["corrected_rung2"]
    assert corrected["decision_status"] == "tie_pending"
    assert corrected["winner"] is None
    assert [c[:12] for c in corrected["tie_break_candidates"]] == [
        "fe9683e6a9c7", "85bde4ded2c3"]
    assert [s[:12] for s in corrected["sc_still_owed"]] == ["fe9683e6a9c7"]
    assert corrected["control_is_outside_the_interval"] is True
    assert corrected["admitted_rungs"] == [1, 2]

    inv = prereg["probe_inventory"]
    assert inv["missing_sb"] == []
    assert sorted(inv["reused_sb"]) == sorted(
        ["fe9683e6a9c7", "85bde4ded2c3", "control-qwen"])
    assert inv["purchasable"] == ["fe9683e6a9c7/sc"]

    # Three reuse records, not two.
    assert set(prereg["reuse_rule"]) >= {
        "historical_record", "attempt5_record", "attempt4_record"}


def test_the_pricing_cites_the_attempt4_reuse_that_makes_missing_sb_empty():
    priced = json.loads(PRICING.read_text())
    prov = priced["reuse_provenance"]
    records = {r["record"]: r for r in prov["records"]}
    a4 = records["logs/autoinit_attempt4_probe_reuse.json"]
    assert a4["admits"] == ["fe9683e6a9c7/sb"]
    assert a4["probes_dir_digest"] == json.loads(
        ATTEMPT4_REUSE.read_text())["probes_dir_digest"]
    assert priced["evidence"]["missing_sb"] == []


def test_the_live_snapshot_records_the_terminal_phase_b_state():
    """Phase B is CLOSED and the snapshot must say so.

    This test previously pinned the snapshot to `TIE_PENDING`, which was correct
    while one observation was still owed and became wrong the moment
    `fe9683e6a9c7/sc` resolved it. It now pins the terminal state instead, and
    keeps the parts that never depended on the phase being open: that nothing is
    authorized, nothing is running, and no superseded ceiling is described as
    current.
    """
    state = json.loads((REPO / "logs/current_state.json").read_text())
    blob = json.dumps(state)

    # Terminal, and resolved.
    assert "COMPLETE" in state["phase_b_state"]
    assert state["phase_b_result"]["status"] == "RESOLVED"
    assert state["phase_b_result"]["winner"] == (
        "fe9683e6a9c783bbc6fe276a78c851c6")
    assert state["phase_b_result"]["winner_is_control"] is False
    assert state["phase_b_result"]["tie_break_ran"] is True

    # The caveat travels with the result, always.
    assert state["phase_b_result"]["clears_by"] == 0.00007
    assert "not comfortable" in state["phase_b_result"]["read_with_care"]
    assert "SELECTION evidence" in state["phase_b_result"]["not_capability"]
    assert state["phase_b_result"]["authorizes"].startswith("nothing")

    # Nothing is live.
    assert state["authorized"]["any"] is False
    assert state["running"]["pods"] == 0 and state["running"]["launchers"] == 0
    assert state["prepared_launch"]["any"] is False
    assert state["budget"]["planning_floor_usd"] is None, (
        "a planning floor implies a priced next session; none is planned")

    # Phase C execution is not started.
    #
    # This used to require the four words "NOT STARTED / NOT DESIGNED / NOT
    # PRICED / NOT AUTHORIZED" in a single `phase_c.status`. That premise
    # expired on 2026-09-01: Phase C0 is COMPLETE / APPROVED / FROZEN, and its
    # whole purpose was to *design* C1. Keeping "NOT DESIGNED" would have forced
    # the snapshot to state something false, so the assertion moved to the part
    # that is still true and still worth protecting — that no Phase-C *execution*
    # has started, been priced, or been authorized.
    assert "FROZEN" in state["phase_c"]["c0"]["status"]
    assert state["phase_c"]["c0"]["authorizes"] == "nothing"
    for word in ("NOT STARTED", "NOT IMPLEMENTED", "NOT PRICED", "NOT AUTHORIZED"):
        assert word in state["phase_c"]["c1"]["status"], word
    assert "NOT STARTED" in state["phase_c"]["c2"]["status"]

    # No superseded scope or ceiling described as current.
    assert "at most 2 conditional sc" not in blob
    assert "HARD CEILING $8.0691" not in blob
    # The roadmap survives.
    assert "PHASE C" in blob or "phase_c" in blob
    assert "ATTENTION" in blob


def test_the_handoff_and_phase_index_exist_and_are_linked():
    """A new reviewer must be able to reconstruct the history without knowing
    filenames. These two are the entry points the snapshot promises."""
    for name in ("PHASE_INDEX.md", "phase_a_vs_phase_b_comparison.md",
                 "phase_c_roadmap.md", "HANDOFF_next_session.md"):
        assert (REPO / "logs" / name).is_file(), name
    state = json.loads((REPO / "logs/current_state.json").read_text())
    assert state["read_order"][0].startswith("logs/PHASE_INDEX.md")
    assert "HANDOFF_next_session.md" in state["handoff"]
