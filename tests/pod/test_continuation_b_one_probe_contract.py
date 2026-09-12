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
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

PROBES = REPO / "logs/stages/stage-1/continuation_b/runs/attempt4/probes"
PRICING = REPO / "logs/shared/analyses/autoinit_behavioural_continuation_pricing.json"
ATTEMPT4_REUSE = REPO / "logs/shared/analyses/autoinit_attempt4_probe_reuse.json"

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
    from experiments.phase_b.continuation import BOUND_EVIDENCE

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
    from aadistill.governance.authorization import AuthorizationError
    from experiments.phase_b.continuation import ContinuationAuthorization

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
        (REPO / "logs/stages/stage-1/continuation_b/plans/autoinit_continuation_b_preregistration.json").read_text())
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
    from aadistill.initialization.planning.recovery import RecoveryAdmissionError

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

    # And no other caller exists in this driver's LINEAGE. Scoped to scripts
    # that import the Phase-A driver, because that is what the invariant is
    # about: a session inheriting `run_probe` must not gain a second route to
    # buying a probe. The standalone C1 driver has its own `probe_config` and its
    # own training loop, imports none of this, and is not in scope — an
    # unqualified scan over `scripts/pod` would make every future standalone
    # driver fail a continuation invariant it cannot violate.
    lineage = [p for p in (REPO / "scripts/pod").glob("*.py")
               if "autoinit_phase_a_driver" in p.read_text()
               or p.name == "autoinit_phase_a_driver.py"]
    callers = [p for p in lineage if "self.probe_config(" in p.read_text()]
    assert [p.name for p in callers] == ["autoinit_phase_a_driver.py"], callers
    assert "autoinit_c1_driver.py" not in [p.name for p in lineage]


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
    from experiments.phase_b.continuation import CONTINUATION_PLAN_V1

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
    from experiments.phase_b.continuation import continuation_source_digest

    prereg = json.loads(
        (REPO / "logs/stages/stage-1/continuation_b/plans/autoinit_continuation_b_preregistration.json").read_text())
    recorded = prereg["executable_source"]["digest"]
    live = continuation_source_digest(REPO)["digest"]

    if recorded != live:
        # A drift is allowed ONLY when it is declared. The continuation is closed
        # and consumed — its authorization binds a non-HEAD commit, so no
        # continuation-B session can launch again — but the SINGLE SESSION_KIND
        # dispatcher it measures is shared by every launchable session, and a new
        # session needs a branch in it. Rewriting the preregistration would
        # destroy the evidence of what attempt 5 executed; deleting this check
        # would destroy its meaning. Declaring the change is the remedy this
        # project already chose once, for the same file, when the continuation
        # itself needed its branch.
        record = REPO / "logs/stages/stage-1/continuation_b/analyses/autoinit_continuation_b_post_freeze_changes.json"
        assert record.is_file(), (
            f"the preregistration binds {recorded[:12]}… but the source tree "
            f"digests to {live[:12]}… and nothing declares the change. Either "
            "revert the edit or record it, as "
            "logs/stages/stage-1/phase_b/analyses/autoinit_phase_b_post_freeze_changes.json does.")
        declared = json.loads(record.read_text())
        assert declared["frozen_digest"] == recorded
        assert declared["post_freeze_digest"] == live, (
            "the declaration is itself stale: it records "
            f"{declared['post_freeze_digest'][:12]}… but the tree is now "
            f"{live[:12]}…")
        assert declared["preregistration_rewritten"] is False
        assert declared["scientific_impact"].startswith("NONE")
    else:
        assert prereg["executable_source"]["n_files"] == len(
            continuation_source_digest(REPO)["files"])


def test_the_preregistration_binds_the_live_session_plan_and_pricing():
    from experiments.phase_b.continuation import CONTINUATION_PLAN_V1

    prereg = json.loads(
        (REPO / "logs/stages/stage-1/continuation_b/plans/autoinit_continuation_b_preregistration.json").read_text())
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
        (REPO / "logs/stages/stage-1/continuation_b/plans/autoinit_continuation_b_preregistration.json").read_text())
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
    #: The pricing record is EVIDENCE and names the path this file had when it
    #: was priced. log-layout-v1 moved the file and did not rewrite the record,
    #: so the lookup goes through the migration rather than assuming either
    #: address.
    import json as _json

    forward = {}
    for m in sorted((REPO / "logs/migrations").glob("*/manifest.json")):
        try:
            doc = _json.loads(m.read_text())
        except (OSError, _json.JSONDecodeError):
            continue
        forward.update({e["old_path"]: e["new_path"]
                        for e in doc.get("entries", [])})
    records = {}
    for r in prov["records"]:
        records[r["record"]] = r
        if r["record"] in forward:
            records[forward[r["record"]]] = r
    a4 = records["logs/shared/analyses/autoinit_attempt4_probe_reuse.json"]
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
    state = json.loads((REPO / "logs/state/current.json").read_text())
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

    # Nothing PHASE-B is live, which is what this module is about.
    #
    # This asserted `authorized.any is False` outright until 2026-09-11, when a
    # Phase-C1 execution package was approved and the assertion started failing
    # in a module that has nothing to do with Phase C. "Nothing anywhere is
    # authorized" was never the claim worth protecting here — "Phase B is closed
    # and authorizes nothing further" is — so whatever IS authorized must be
    # named, and must not be Phase B.
    if state["authorized"]["any"]:
        who = json.dumps(state["authorized"]).lower()
        assert "phase_c1" in who, (
            "something is authorized and the snapshot does not say what")
        assert "phase_b" not in who, (
            "Phase B is CLOSED and its result authorizes nothing; an "
            "authorization naming it would be reopening a resolved phase")
    #: Not "nothing is running anywhere" — that was the same sentence as "Phase
    #: B is closed" only while no other phase could run, and it stopped being so
    #: the moment a Phase-C1 attempt went live. Whether anything is running at
    #: all is compared against the prose view in
    #: `test_the_two_state_views_agree_on_what_is_running_and_authorized`, in
    #: whichever direction is true. Here: whatever runs must not be Phase B.
    for block in ("running", "prepared_launch"):
        assert "phase_b" not in json.dumps(state[block]).lower(), (
            f"{block} names Phase B, which is CLOSED and authorizes nothing")
    if state["running"]["pods"]:
        assert state["running"].get("pod_id"), (
            "a pod is recorded as running and the snapshot does not name it")
    # This used to require `planning_floor_usd is None` on the reasoning that a
    # floor implies a priced next session and none was planned. C1 is now priced,
    # so the assertion moved to what still protects the boundary: a floor may
    # exist, but it must be the accepted pricing record's, and priced must not be
    # mistaken for funded.
    floor = state["budget"]["planning_floor_usd"]
    if floor is not None:
        pricing = json.loads((REPO / "logs/stages/stage-1/phase_c1/plans/phase_c1_pricing.json").read_text())
        assert floor == pricing["totals"]["floor_usd"], (
            f"the snapshot's planning floor {floor} is not the accepted pricing "
            f"record's {pricing['totals']['floor_usd']}")
        assert pricing["authorizes"] == "nothing"
        #: Priced is not funded. The pricing record cannot authorize, and any
        #: funding that does exist has to come from a named maintainer package
        #: rather than from the existence of a price.
        if state["authorized"]["any"]:
            assert state["authorized"].get("package_id"), (
                "funding is claimed with no package named; a price is not a "
                "permission and neither is an unattributed 'yes'")

    # Phase C *execution* has not started.
    #
    # This assertion has now outlived two premises. It first required the four
    # words "NOT STARTED / NOT DESIGNED / NOT PRICED / NOT AUTHORIZED" in a
    # single `phase_c.status`; C0 made "NOT DESIGNED" false. It was then rewritten
    # to require "NOT IMPLEMENTED"; the CPU machinery made that false too. Both
    # times the *protective* content was the same and both times a wording change
    # broke it, so it is stated once here in terms that only become false when
    # something real happens:
    #
    #   design and implementation may proceed at $0 — execution may not.
    #
    # The live-fact fields asserted above (authorized.any, prepared_launch.any,
    # running.pods, budget.planning_floor_usd) already carry "nothing was bought".
    assert "FROZEN" in state["phase_c"]["c0"]["status"]
    assert state["phase_c"]["c0"]["authorizes"] == "nothing"
    # THE THIRD wording break. The comment above already records two: "NOT
    # DESIGNED" then "NOT IMPLEMENTED", each made false by ordinary progress
    # while the protective content was unchanged. The Milestone-A repair of the
    # snapshot's self-contradiction made it happen again -- "NOT SCIENTIFICALLY
    # EXECUTED" became "TREATMENT AND ENDPOINT UNMEASURED", which says the same
    # thing in fewer words.
    #
    # So this stops matching a PHRASE and asserts the content, which is what the
    # comment above said to do and what the phrase kept standing in for:
    # attempt 9 measured the replay and nothing else, and nothing is live.
    measured = state["phase_c"]["c1"]["measured"]
    assert "UNMEASURED" in measured, measured
    for axis in ("treatment", "endpoint"):
        assert axis in measured.lower(), axis
    assert re.search(r"\bno decision\b", measured, re.I), measured
    # A grant WAS issued and consumed, so "never authorized" would be false.
    #
    # Nor is "nothing is authorized" the invariant: on 2026-09-11 a Phase-C1
    # execution package was approved, and later a bundle will be staged for it —
    # both legitimate, and both of which used to fail here. Whether anything is
    # LIVE is checked against the prose view in
    # `test_the_two_state_views_agree_on_what_is_running_and_authorized`, in
    # whichever direction is true. What this module owns is narrower and does not
    # move: **C1 execution has produced no science**, whatever is authorized.
    c1 = state["phase_c"]["c1"]
    assert re.search(r"not currently authorized|no current grant|"
                     r"treatment and endpoint unmeasured", c1["status"], re.I), (
        c1["status"])
    if state["authorized"]["any"]:
        #: `formal_sessions_used` since 2026-09-11, when the attempt CAP was
        #: withdrawn and the counted thing was renamed to what it always was --
        #: a launcher session. The older key is still accepted so this reads a
        #: snapshot from either side of that change; what it will not accept is
        #: NEITHER, which is a package whose usage is untracked.
        used = (state["authorized"].get("formal_sessions_used")
                if "formal_sessions_used" in state["authorized"]
                else state["authorized"].get("formal_attempts_used"))
        assert isinstance(used, int), (
            "an authorization exists and the snapshot does not count what it "
            "has been used for")
        assert "UNMEASURED" in c1["measured"], (
            "an authorization exists AND C1 claims a measured endpoint; one of "
            "the two is wrong and neither may be assumed")
    assert "NOT STARTED" in state["phase_c"]["c2"]["status"]
    # Nothing that needs a GPU may be claimed as built.
    assert "pre-ATTENTION parent" in state["phase_c"]["c1"]["not_built"]

    # No superseded scope or ceiling described as current.
    assert "at most 2 conditional sc" not in blob
    assert "HARD CEILING $8.0691" not in blob
    # The roadmap survives.
    assert "PHASE C" in blob or "phase_c" in blob
    assert "ATTENTION" in blob


def test_the_handoff_and_phase_index_exist_and_are_linked():
    """A new reviewer must be able to reconstruct the history without knowing
    filenames. These are the entry points the snapshot promises."""
    #: By ROUTE, not by root filename. Two of these moved into the experiment
    #: that owns them on 2026-09-12, and pinning their directory would have
    #: forced the documents back to the root to keep a test green. What the
    #: reviewer needs is unchanged: each document exists, and each is reachable
    #: by following links from the entry point without knowing where it lives.
    import re as _re

    def reachable_from(start: Path, hops: int = 3) -> set[str]:
        seen, frontier = {start.resolve()}, [start]
        for _ in range(hops):
            nxt = []
            for doc in frontier:
                if not doc.is_file() or doc.suffix != ".md":
                    continue
                for t in _re.findall(r"\]\(([^)#\s]+)\)", doc.read_text()):
                    if t.startswith(("http", "mailto:")):
                        continue
                    q = (doc.parent / t).resolve()
                    #: A directory link is followed to its README: that is how
                    #: the canonical layout indexes a group, and a walker that
                    #: only followed files stopped at every directory.
                    if q.is_dir() and (q / "README.md").is_file():
                        q = q / "README.md"
                    if q.exists() and q not in seen:
                        seen.add(q)
                        nxt.append(q)
            frontier = nxt
        return {q.name for q in seen}

    found = reachable_from(REPO / "logs/README.md")
    #: `PHASE_INDEX.md` became `state/phase_index.md` in log-layout-v1. The
    #: requirement is that a reviewer can REACH each document from the entry
    #: point, which is what the loop below checks -- the basename is incidental.
    for name in ("phase_index.md", "phase_a_vs_phase_b_comparison.md",
                 "phase_c_roadmap.md", "HANDOFF_next_session.md"):
        hits = list(REPO.glob(f"logs/**/{name}"))
        assert hits, f"{name} exists nowhere"
        assert name in found, (
            f"{name} exists at {hits[0].relative_to(REPO)} but cannot be "
            "reached by following links from logs/README.md")

    state = json.loads((REPO / "logs/state/current.json").read_text())
    first = state["read_order"][0].split()[0]
    assert (REPO / first).is_file(), f"read_order starts at {first}, which is not a file"


# --- where the CURRENT handoff route actually lands -------------------------
#
# This asked only that `state["handoff"]` contained the string
# "HANDOFF_next_session.md". It did, and that file had said since 2026-09-02
# that C1 was implemented-but-not-executed, with `$19.9003` of headroom against
# a `$13.7578` ceiling -- so the route was green while pointing a new reviewer
# at a state nine paid attempts out of date.
#
# So these resolve the route and read what is on the other end.

def handoff_targets() -> list[Path]:
    """Every logs/ file the current handoff entry names, resolved to a path."""
    import re

    state = json.loads((REPO / "logs/state/current.json").read_text())
    entry = state["handoff"]
    #: `/` included: the canonical layout is nested, so a pattern that stopped
    #: at the first slash matched only `logs/README.md` and the route was read
    #: as landing on the index -- which states no figures by design.
    names = re.findall(r"logs/[A-Za-z0-9_./\-]+\.(?:md|json)", entry)
    assert names, f"the handoff entry names no file at all: {entry!r}"
    return [REPO / n for n in names]


def test_the_handoff_route_resolves_to_files_that_exist():
    for target in handoff_targets():
        assert target.is_file(), f"the handoff routes to a missing file: {target}"


def test_the_route_does_not_land_on_the_superseded_handoff():
    """Not a string check on the snapshot: the resolved TARGET must not be it."""
    for target in handoff_targets():
        assert target.name != "HANDOFF_next_session.md", (
            "current_state.json still routes to the 2026-09-02 handoff, which "
            "says C1 is unexecuted and quotes a superseded budget")


def current_region(text: str) -> str:
    """A living snapshot's CURRENT claims: everything before its history.

    `STATE.md` is a snapshot followed by dated session blocks (AGENTS.md 3.3),
    and those blocks legitimately quote figures that were true when written --
    a `$13.7578` cap that really did apply to attempt 3, for instance. Scanning
    the whole file would make every honest historical record look like a stale
    current claim, and the fix for that would be deleting history.

    So the boundary is the first dated block, `^> **`.
    """
    import re

    lines = text.splitlines()
    end = next((i for i, l in enumerate(lines) if re.match(r"^> \*\*", l)),
               len(lines))
    return "\n".join(lines[:end])


def test_what_the_route_lands_on_is_actually_current():
    """Read the destination's CONTENT. A route that resolves to a real file
    saying the wrong thing is the failure this replaces."""
    state = json.loads((REPO / "logs/state/current.json").read_text())
    text = "\n".join(t.read_text() for t in handoff_targets())
    current = "\n".join(current_region(t.read_text()) for t in handoff_targets())

    # The superseded handoff's headline claims, which must not be asserted as
    # current anywhere the route lands.
    for stale in ("$19.9003", "$13.7578"):
        assert stale not in current, (
            f"the current handoff destination asserts {stale} as current")
    assert "not executed" not in current.lower(), (
        "the destination still describes C1 as unexecuted")

    # And the facts a reviewer arriving there must find, taken from the snapshot
    # rather than transcribed -- a hardcoded figure here would go stale exactly
    # the way the old handoff did.
    spend = f"${state['budget']['cumulative_spend_usd']:.4f}"
    remaining = f"${state['budget']['remaining_usd']:.4f}"
    for fact in (spend, remaining):
        assert fact in current, (
            f"the handoff destination does not state {fact!r} as current")
    for fact in ("MEASURED", "NO DECISION", "CONFIRMED ON REAL CUDA"):
        assert fact in text, f"the handoff destination does not state {fact!r}"


def test_the_current_region_is_not_the_whole_file():
    """Guards `current_region`: if the boundary vanished, the check above would
    silently start scanning the history and could only be made green by
    deleting it."""
    for target in handoff_targets():
        whole = target.read_text()
        assert len(current_region(whole)) < len(whole), (
            f"{target.name} has no dated history block; the current-region "
            "boundary is not doing anything")


def test_the_superseded_handoff_is_registered_as_historical():
    """Kept as evidence, and labelled -- in the file that owns ownership."""
    assert (REPO / "logs/archive/repository/handoffs/HANDOFF_next_session.md").is_file(), (
        "the historical handoff was deleted rather than superseded")
    state = json.loads((REPO / "logs/state/current.json").read_text())
    assert "HANDOFF_next_session.md" in state["superseded_handoff"]
    assert "HISTORICAL" in state["superseded_handoff"]

    #: The superseding statement moved to the document that OWNS the archive.
    #: `ownership.md` classifies `logs/` top-level entries, and after
    #: log-layout-v1 the handoff is not one -- it is inside `archive/handoffs/`,
    #: whose README says what it holds and what replaced it. Asking the catalog
    #: for a row about a file two levels down would be asking the wrong owner.
    archive = (REPO / "logs/archive/README.md").read_text()
    assert "handoffs/" in archive, "the archive does not index its groups"
    row = next(l for l in archive.splitlines() if "handoffs/" in l)
    assert "superseded" in row.lower(), row
    assert "state/current.md" in row, (
        "the archive index does not say what replaced the handoff")
    #: and the snapshot still routes a reader away from it
    assert "HISTORICAL" in state["superseded_handoff"]


def test_the_old_handoffs_own_text_is_not_rewritten():
    """Superseding is a routing change. Editing the historical document to make
    it look current would destroy the evidence of what was believed then."""
    for stale in ("$19.9003", "$13.7578"):
        assert stale in (REPO / "logs/archive/repository/handoffs/HANDOFF_next_session.md").read_text(), (
            f"{stale} was edited out of the historical handoff")
