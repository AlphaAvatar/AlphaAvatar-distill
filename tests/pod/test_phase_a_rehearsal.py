"""Rehearse the Phase-A harness before it can cost anything.

Four paid pods in this project have died in lines that had never been executed —
a `KeyError: 'metrics'` after both models loaded, a launcher misreading a clean
setup, a config gate reading an unstaged battery, a control that passed every
identity check and had no tokenizer. Phase A is the most expensive session the
project has ever planned (12-17 GPU-hours), so its harness is executed here at
toy scale rather than inspected.

What is rehearsed:

* the session plan's ordering and its blocking gates;
* the driver's full lifecycle, success and failure, with every paid call scripted;
* that a blocking failure stops before the probes;
* that cleanup is not success;
* the budget, priced through the real `plan_session`;
* the artifact specs, against the paths the driver really writes;
* the marker vocabulary the launcher polls for;
* the probe-config override set and the probe-resume binding;
* the authorization type's refusals, in both directions;
* that no follow-on experiment is reachable.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from aadistill.autoinit.authorization import (  # noqa: E402
    AuthorizationError, MICRO_PREFLIGHT_AUTHORIZATION, SpendAuthorization,
)
from aadistill.autoinit.phase_a import (  # noqa: E402
    PHASE_A_HARNESS_SOURCE_FILES_V1, PHASE_A_PLAN_V1, PHASE_A_SCOPE,
    PhaseAAuthorization, phase_a_harness_digest,
)
from aadistill.autoinit.recovery import RecoveryAdmissionError  # noqa: E402


# --- the plan ---------------------------------------------------------------


def test_the_plan_orders_the_stages_and_blocks_on_them():
    stages = {s.stage: s for s in PHASE_A_PLAN_V1.stages}
    assert sorted(stages) == [0, 1, 2, 3, 4, 5]
    # Attestation, search and both rungs block. Selection and the conditional
    # tie-break do not: a tie that survives seed sc is a RESULT, and a
    # non-blocking stage still fails the session without blocking teardown.
    assert [s for s in sorted(stages) if stages[s].blocking] == [0, 1, 2, 3]
    assert not stages[4].blocking and not stages[5].blocking


def test_a_rung_cannot_start_before_the_search_passes():
    with pytest.raises(RecoveryAdmissionError):
        PHASE_A_PLAN_V1.advance_to(2, {0: {"passed": True},
                                       1: {"passed": False, "reason": "3 leaves"}})
    with pytest.raises(RecoveryAdmissionError):
        PHASE_A_PLAN_V1.advance_to(1, {0: {"passed": False, "reason": "thresholds"}})
    # And the happy path is reachable, or the test above proves nothing.
    PHASE_A_PLAN_V1.advance_to(2, {0: {"passed": True}, 1: {"passed": True}})


def test_there_is_no_stage_six():
    assert max(s.stage for s in PHASE_A_PLAN_V1.stages) == 5
    driver = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    assert 'choices=("all",)' in driver, "a stage outside the plan is expressible"


def test_the_scope_says_what_phase_a_does_and_does_not_do():
    scope = PHASE_A_SCOPE.as_dict()
    # It DOES train: nine recovery probes. Claiming otherwise would be the
    # continuation's scope pasted into a session that trains.
    assert scope["trains_anything"] is True
    assert scope["retrains_permanent_controls"] is False
    assert scope["reaches_any_followon"] is False
    assert scope["control_is_injected_by_hash"] is True
    assert scope["searched_leaves"] == 5 and scope["survivors"] == 2
    assert scope["seeds"] == [20260726, 20260801]
    assert scope["conditional_tie_break_seed"] == 20260813


# --- the authorization ------------------------------------------------------


def _auth(**kw):
    base = dict(
        authorization_id="rehearsal", granted_utc="2026-08-15T00:00:00Z",
        granted_by="rehearsal", plan_id=PHASE_A_PLAN_V1.plan_id,
        plan_hash=PHASE_A_PLAN_V1.plan_hash, science_plan_hash="SCIENCE",
        expected_usd=17.9, hard_cap_usd=20.02,
        authorized_stages=(0, 1, 2, 3, 4, 5),
        stage_conditions={"0": "attestation"}, scope_note="rehearsal")
    base.update(kw)
    return PhaseAAuthorization(**base)


def test_a_spend_authorization_cannot_be_loaded_as_a_phase_a_grant(tmp_path):
    """Refused by SCHEMA, so adding a key to the narrow artifact cannot work."""
    p = tmp_path / "spend.json"
    p.write_text(json.dumps(MICRO_PREFLIGHT_AUTHORIZATION.as_dict()))
    with pytest.raises(AuthorizationError, match="schema"):
        PhaseAAuthorization.load(p)


def test_a_phase_a_grant_cannot_be_loaded_by_the_narrow_type(tmp_path):
    """The gate that guards the preflight and the continuation, unchanged."""
    p = tmp_path / "phase_a.json"
    p.write_text(json.dumps(_auth().as_dict()))
    with pytest.raises(AuthorizationError, match="Phase A authorization"):
        SpendAuthorization.load(p)


def test_the_narrow_type_still_answers_no(tmp_path):
    assert MICRO_PREFLIGHT_AUTHORIZATION.allows_phase_a is False
    assert MICRO_PREFLIGHT_AUTHORIZATION.automatic_phase_a_start is False


def test_phase_a_cannot_authorize_a_follow_on(tmp_path):
    a = _auth()
    assert a.allows_phase_a is True
    assert a.automatic_followon_start is False
    with pytest.raises(AuthorizationError):
        a.refuse_followon("full recovery of the winner")
    # And an artifact that claims one is refused on load.
    payload = a.as_dict()
    payload["automatic_followon_start"] = True
    from aadistill.infrastructure.manifest import sha256_json
    payload.pop("authorization_sha256")
    payload["authorization_sha256"] = sha256_json(payload)
    p = tmp_path / "chained.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(AuthorizationError, match="follow-on"):
        PhaseAAuthorization.load(p)


def test_both_plans_are_bound_independently():
    a = _auth()
    a.require_plan(PHASE_A_PLAN_V1.plan_hash)
    a.require_science_plan("SCIENCE")
    with pytest.raises(AuthorizationError, match="session plan"):
        a.require_plan("moved")
    # The science plan moves on its own — a threshold changing after the grant
    # must not be waved through because the session plan still matches.
    with pytest.raises(AuthorizationError, match="science plan"):
        a.require_science_plan("moved")


def test_an_edited_harness_is_refused():
    observed = phase_a_harness_digest(REPO)
    _auth(harness_source_digest=observed["digest"]).require_harness(REPO)
    with pytest.raises(AuthorizationError, match="digests to"):
        _auth(harness_source_digest="0" * 64).require_harness(REPO)
    with pytest.raises(AuthorizationError, match="no harness_source_digest"):
        _auth().require_harness(REPO)


def test_the_harness_set_covers_the_code_that_actually_runs():
    """An authorization digesting the wrong files admits an edited driver."""
    for required in ("scripts/pod/autoinit_phase_a_driver.py",
                     "scripts/pod/autoinit_phase_a_launch.py",
                     # Imported by the driver, so just as much the executable.
                     "scripts/autoinit/phase_a_search.py",
                     "scripts/autoinit/write_preregistration.py",
                     "src/aadistill/autoinit/phase_a.py"):
        assert required in PHASE_A_HARNESS_SOURCE_FILES_V1, required
    for rel in PHASE_A_HARNESS_SOURCE_FILES_V1:
        assert (REPO / rel).is_file(), f"declared harness source {rel} is missing"


def test_a_missing_harness_file_refuses_rather_than_digesting_a_smaller_set():
    with pytest.raises(AuthorizationError, match="missing"):
        phase_a_harness_digest(REPO, files=("scripts/pod/does_not_exist.py",))


# --- the frozen science plan ------------------------------------------------


def test_the_executing_plan_reproduces_the_frozen_one():
    """The driver rebuilds the plan; drift between the two is the failure mode.

    `assert_preregistered` compares hashes, so a builder that produced a
    near-copy would fail the gate in a way that looks like tampering.
    """
    from aadistill.autoinit.recovery import assert_preregistered
    from write_preregistration import build_frozen_plan

    frozen = REPO / "logs/autoinit_phase_a_recovery_plan_frozen.json"
    assert frozen.is_file(), "no frozen science plan; Stage 0 has nothing to bind"
    plan = build_frozen_plan(REPO)
    assert assert_preregistered(plan, frozen)["plan_hash"] == plan.plan_hash


def test_the_frozen_plan_carries_materialized_thresholds():
    from write_preregistration import build_frozen_plan
    plan = build_frozen_plan(REPO)
    # Both must be real numbers: a selector with a pending rule raises by design,
    # and discovering that after nine probes would waste the session.
    assert plan.equivalence.require_value() == pytest.approx(0.011695296982299022)
    assert plan.feasibility_floor() == pytest.approx(0.30)
    assert plan.searched_leaves == 5 and plan.survivors == 2
    assert plan.tie_break_seed == 20260813


# --- the driver lifecycle ---------------------------------------------------


def load_driver(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "phase_a_driver", REPO / "scripts/pod/autoinit_phase_a_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_a_driver"] = mod
    spec.loader.exec_module(mod)
    mod.WS = tmp_path
    mod.STATUS = tmp_path / "phase_a.status"
    mod.AUDIT = tmp_path / "audit"
    (mod.AUDIT / "probes").mkdir(parents=True, exist_ok=True)
    return mod


class Args:
    stage = "all"
    image_digest = "sha256:rehearsal"
    rate = 0.99
    spent_usd = 0.20
    soft_stop_usd = 19.68
    authorized_usd = 20.02
    search_minutes = 180.0
    probe_train_minutes = 61.55
    probe_battery_minutes = 9.82


def build_driver(tmp_path, *, attest_ok=True, search_ok=True, rung1_ok=True,
                 rung2_ok=True, tie_break=False, select_ok=True):
    mod = load_driver(tmp_path)
    d = mod.PhaseADriver.__new__(mod.PhaseADriver)
    d.a = Args()
    d.t0 = __import__("time").time()
    d.results, d.ev = {}, {"stages": {}}
    d.evaluation_protocol = None
    d.plan = None
    d.leaves, d.control_state, d.rung1, d.rung2 = [], None, None, None
    d.mod = mod

    class Auth:
        hard_cap_usd = 20.02
        def require_stage(self, s): pass
        def require_within_cap(self, usd, what=""): pass
        def as_dict(self): return {"authorization_id": "rehearsal"}
    d.auth = Auth()

    def stage0():
        d.enter(0)
        if not attest_ok:
            return d.record(0, False, "the equivalence interval is not materialized")
        class E:
            evaluation_protocol_hash = "EVAL"
        d.evaluation_protocol = E()
        return d.record(0, True, attested={"equivalence_interval": 0.0117})

    def stage1():
        d.enter(1)
        if not search_ok:
            return d.record(1, False, "3 admissible leaves; the plan asks for 5")
        d.leaves = [f"leaf{i}" for i in range(5)]
        return d.record(1, True, n_leaves=5)

    def stage2():
        d.enter(2)
        if not rung1_ok:
            return d.record(2, False, "insufficient budget for probe 4")
        d.rung1 = {"advancing": ["control", "leaf0", "leaf1"]}
        return d.record(2, True, n_probes=6)

    def stage3():
        d.enter(3)
        if not rung2_ok:
            return d.record(3, False, "the canonical control did not advance")
        d.rung2 = {"decision_status": "tie_pending" if tie_break else "resolved",
                   "needs_tie_break_seed": tie_break,
                   "tie_break_candidates": ["leaf0", "control"] if tie_break else []}
        return d.record(3, True, n_probes=3)

    def stage4():
        d.enter(4)
        if not d.rung2.get("needs_tie_break_seed"):
            return d.record(4, True, ran=False)
        return d.record(4, True, ran=True, n_probes=2)

    def stage5():
        d.enter(5)
        if not select_ok:
            return d.record(5, False, "selection raised")
        return d.record(5, True, result={"decision_status": "resolved"})

    d.stage0, d.stage1, d.stage2 = stage0, stage1, stage2
    d.stage3, d.stage4, d.stage5 = stage3, stage4, stage5
    for name in ("enter", "record", "usd", "afford", "save", "run", "finish"):
        setattr(d, name, getattr(mod.PhaseADriver, name).__get__(d))
    return d, mod


def markers_of(mod) -> list[str]:
    if not mod.STATUS.is_file():
        return []
    return [ln.split("MARKER:")[1] for ln in mod.STATUS.read_text().splitlines()
            if "MARKER:" in ln]


def test_the_success_lifecycle_completes_and_starts_nothing_after_it(tmp_path):
    d, mod = build_driver(tmp_path)
    assert d.run() == 0
    assert sorted(d.results) == [0, 1, 2, 3, 4, 5]
    assert "ALL_DONE" in markers_of(mod)
    ev = json.loads((mod.AUDIT / "phase_a_evidence.json").read_text())
    assert ev["phase_a_successful"] is True and ev["outcome"] == "SUCCESS"
    assert ev["retrains_permanent_controls"] is False
    assert ev["followon_started"] is False
    assert ev["followon_reachable_from_this_driver"] is False


def test_the_tie_break_rung_runs_only_when_it_is_owed(tmp_path):
    d, _ = build_driver(tmp_path, tie_break=False)
    assert d.run() == 0
    assert d.ev["stages"]["4"]["ran"] is False

    d2, _ = build_driver(tmp_path / "b", tie_break=True)
    assert d2.run() == 0
    assert d2.ev["stages"]["4"]["ran"] is True


@pytest.mark.parametrize("kwargs,stage", [
    ({"attest_ok": False}, 0),
    ({"search_ok": False}, 1),
    ({"rung1_ok": False}, 2),
    ({"rung2_ok": False}, 3),
])
def test_a_blocking_failure_stops_before_any_later_stage(tmp_path, kwargs, stage):
    d, mod = build_driver(tmp_path, **kwargs)
    rc = d.run()
    assert rc == 20 + stage
    assert max(d.results) == stage
    assert "PHASE_A_FAILED" in markers_of(mod)
    assert "ALL_DONE" not in markers_of(mod)


def test_a_selection_failure_is_incomplete_not_success(tmp_path):
    """Cleanup is not success: stage 5 is non-blocking for TEARDOWN only."""
    d, mod = build_driver(tmp_path, select_ok=False)
    rc = d.run()
    assert rc == 25
    assert "PHASE_A_INCOMPLETE" in markers_of(mod)
    assert "ALL_DONE" not in markers_of(mod)
    ev = json.loads((mod.AUDIT / "phase_a_evidence.json").read_text())
    assert ev["phase_a_successful"] is False
    assert ev["outcome"] == "INCOMPLETE"


# --- probe machinery --------------------------------------------------------


#: Stated literally, NOT read from the driver. Asserting `changed <=
#: mod.PROBE_OVERRIDES` would make the constant its own oracle: widening it to
#: admit `loss` would keep the test green, which is precisely the change that
#: must never pass. Verified by mutation — the literal form catches it, the
#: self-referential form did not.
EXPECTED_PROBE_OVERRIDES = {
    "run_name", "_purpose", "out_dir", "data_dir", "seed", "student_path"}


def test_the_probe_override_set_is_exactly_the_declared_one(tmp_path):
    mod = load_driver(tmp_path)
    assert set(mod.PROBE_OVERRIDES) == EXPECTED_PROBE_OVERRIDES, (
        "the set of fields a probe may change relative to the frozen recipe has "
        "moved. Only run identity, output path, pack path, seed and the "
        "initialization may differ; anything else makes the arms incomparable.")


def test_a_probe_config_may_differ_only_in_the_override_set(tmp_path):
    mod = load_driver(tmp_path)
    d = mod.PhaseADriver.__new__(mod.PhaseADriver)
    d.a = Args()
    descriptor = {"probe_id": "p1", "rung": 1, "seed": 20260726,
                  "student_checkpoint": "artifacts/whatever/model"}
    path = mod.PhaseADriver.probe_config(d, descriptor)
    derived = json.loads(path.read_text())
    frozen = json.loads(mod.FROZEN_RECIPE.read_text())
    changed = {k for k in set(frozen) | set(derived) if frozen.get(k) != derived.get(k)}
    assert changed <= EXPECTED_PROBE_OVERRIDES
    # The two that carry the experiment must actually be applied.
    assert derived["seed"] == 20260726
    assert derived["student_path"] == "artifacts/whatever/model"
    # And the ones that make arms comparable must not move.
    for untouched in ("loss", "optim", "schedule", "batch", "trainable_patterns",
                      "block_len", "dtype", "packing", "teacher"):
        assert derived[untouched] == frozen[untouched], untouched


def test_a_probe_config_that_would_change_the_recipe_is_refused(tmp_path):
    mod = load_driver(tmp_path)
    d = mod.PhaseADriver.__new__(mod.PhaseADriver)
    d.a = Args()
    original = mod.PROBE_OVERRIDES
    try:
        # Simulate an override set that let the learning rate move: the guard is
        # the set membership, so shrink it and confirm the check bites.
        mod.PROBE_OVERRIDES = frozenset({"run_name"})
        with pytest.raises(RecoveryAdmissionError, match="outside the allowed"):
            mod.PhaseADriver.probe_config(
                d, {"probe_id": "p2", "rung": 1, "seed": 1,
                    "student_checkpoint": "x"})
    finally:
        mod.PROBE_OVERRIDES = original


def _descriptor():
    return {"probe_id": "p1", "rung": 1, "state_id": "leaf0", "is_control": False,
            "seed": 20260726, "student_artifact_digest": "DIGEST",
            "student_checkpoint": "x"}


def _driver_for_restore(tmp_path):
    mod = load_driver(tmp_path)
    d = mod.PhaseADriver.__new__(mod.PhaseADriver)
    class E:
        evaluation_protocol_hash = "EVAL"
    d.evaluation_protocol = E()
    return d, mod


def test_a_journalled_probe_is_restored_when_it_still_binds(tmp_path):
    d, mod = _driver_for_restore(tmp_path)
    (mod.AUDIT / "probes" / "p1.json").write_text(json.dumps({
        "student_artifact_digest": "DIGEST", "seed": 20260726,
        "evaluation_protocol_hash": "EVAL", "complete": True, "result": {}}))
    restored = mod.PhaseADriver.restore_probe(d, _descriptor())
    assert restored is not None and restored["resumed"] is True


@pytest.mark.parametrize("field,value", [
    ("student_artifact_digest", "OTHER"),   # a different initialization
    ("seed", 20260801),                     # a different replicate
    ("evaluation_protocol_hash", "OTHER"),  # measured under a different protocol
])
def test_a_journalled_probe_that_no_longer_binds_is_re_run(tmp_path, field, value):
    """Identity is not the probe id. This is the rule `_restore` applies to
    search states, and a resume that ignored it would silently mix a previous
    run's measurements into this one's selection."""
    d, mod = _driver_for_restore(tmp_path)
    record = {"student_artifact_digest": "DIGEST", "seed": 20260726,
              "evaluation_protocol_hash": "EVAL", "complete": True, "result": {}}
    record[field] = value
    (mod.AUDIT / "probes" / "p1.json").write_text(json.dumps(record))
    assert mod.PhaseADriver.restore_probe(d, _descriptor()) is None


def test_an_incomplete_journal_entry_is_not_restored(tmp_path):
    d, mod = _driver_for_restore(tmp_path)
    (mod.AUDIT / "probes" / "p1.json").write_text(json.dumps({
        "student_artifact_digest": "DIGEST", "seed": 20260726,
        "evaluation_protocol_hash": "EVAL", "complete": False}))
    assert mod.PhaseADriver.restore_probe(d, _descriptor()) is None


def test_a_corrupt_journal_entry_is_not_restored(tmp_path):
    """A pod killed mid-write leaves a truncated file; it must re-run, not raise."""
    d, mod = _driver_for_restore(tmp_path)
    (mod.AUDIT / "probes" / "p1.json").write_text('{"student_artifact_dig')
    assert mod.PhaseADriver.restore_probe(d, _descriptor()) is None


def test_capability_pooling_sums_counts_rather_than_averaging_rates(tmp_path):
    """Averaging two seeds' rates over unequal denominators is the pooled_counts
    v1 defect in a different place."""
    mod = load_driver(tmp_path)
    pooled = mod.PhaseADriver.pool_capabilities([
        {"result": {"per_capability": {"gsm8k": {"n": 30, "usable": 17}}}},
        {"result": {"per_capability": {"gsm8k": {"n": 10, "usable": 1}}}},
    ])
    assert pooled["gsm8k"]["n"] == 40 and pooled["gsm8k"]["usable"] == 18
    assert pooled["gsm8k"]["usable_rollout_rate"] == pytest.approx(18 / 40)


def test_a_pooled_row_satisfies_the_real_capability_schema(tmp_path):
    """The gate the rungs call is `CapabilitySchema.validate_all`, and it fails
    closed on a missing capability, a non-integer count or an out-of-range rate.
    A row that does not satisfy it would raise at rung 1 — after six probes and
    roughly seven hours of paid compute. Checked here against the REAL frozen
    schema rather than a hand-written shape.
    """
    from aadistill.autoinit.recovery import CAPABILITY_SCHEMA_V1

    mod = load_driver(tmp_path)
    # The capability names and counts the real scorer emits, per the Stage-3
    # characterization of both controls.
    per_seed = {cap: {"n": 30, "usable": u} for cap, u in
                (("gsm8k", 17), ("math_verified", 12), ("multihop", 15),
                 ("rag", 21), ("knowledge", 5), ("tool", 14))}
    records = [
        {"probe_id": "p1", "state_id": "leaf0", "is_control": False,
         "seed": 20260726,
         "result": {"n": 190, "usable": 74, "n_scorable": 170,
                    "usable_scorable": 74, "correct": 2,
                    "per_capability": per_seed}},
        {"probe_id": "p2", "state_id": "leaf0", "is_control": False,
         "seed": 20260801,
         "result": {"n": 190, "usable": 67, "n_scorable": 170,
                    "usable_scorable": 66, "correct": 2,
                    "per_capability": per_seed}},
    ]
    rows = mod.PhaseADriver.selection_row(
        mod.PhaseADriver.__new__(mod.PhaseADriver), records)
    assert len(rows) == 1
    CAPABILITY_SCHEMA_V1.validate_all(rows)

    row = rows[0]
    # Pooled over both seeds, and the counts the aggregation needs are present.
    assert row["seeds"] == [20260726, 20260801]
    assert row["per_capability"]["gsm8k"]["n"] == 60
    for count in ("n", "usable"):
        assert isinstance(row["per_capability"]["gsm8k"][count], int)


def test_a_pooled_row_missing_a_capability_is_refused(tmp_path):
    """Proves the check above is load-bearing rather than decorative."""
    from aadistill.autoinit.recovery import (
        CAPABILITY_SCHEMA_V1, CapabilitySchemaError,
    )

    mod = load_driver(tmp_path)
    short = {cap: {"n": 30, "usable": 10} for cap in
             ("gsm8k", "math_verified", "multihop", "rag", "knowledge")}  # no tool
    rows = mod.PhaseADriver.selection_row(
        mod.PhaseADriver.__new__(mod.PhaseADriver),
        [{"probe_id": "p1", "state_id": "leaf0", "is_control": False,
          "seed": 20260726,
          "result": {"n": 190, "usable": 74, "n_scorable": 170,
                     "usable_scorable": 74, "correct": 2,
                     "per_capability": short}}])
    with pytest.raises(CapabilitySchemaError, match="tool"):
        CAPABILITY_SCHEMA_V1.validate_all(rows)


# --- leaf retention ---------------------------------------------------------


class _Leaf:
    """Minimal stand-in carrying the fields the retention record reads."""

    def __init__(self, state_id, control=False):
        self.state_id = state_id
        self.provenance = "retained_canonical" if control else "search"
        self.artifact_digest = f"digest-{state_id}"
        self.checkpoint_sha256 = f"sha-{state_id}"
        self.path_label = f"depth>width>ffn>attn::{state_id}"
        self.num_parameters = 596_049_920


def _retention_driver(tmp_path):
    mod = load_driver(tmp_path)
    d = mod.PhaseADriver.__new__(mod.PhaseADriver)
    d.leaves = [_Leaf(f"leaf{i}") for i in range(5)]
    d.control_state = _Leaf("control-qwen3_0p6b_init_v0", control=True)
    d.rung1 = {
        "advancing": ["control-qwen3_0p6b_init_v0", "leaf0", "leaf1"],
        "rule": "top 2 by correct_overall among feasible searched leaves",
        "all_exclusions": [{"state_id": "leaf4",
                            "reason": "usable_rollout_rate=0.1000 below the "
                                      "preregistered feasibility floor 0.3000"}],
    }
    records = [{"probe_id": f"p.{s.state_id}", "state_id": s.state_id,
                "evaluation_protocol_hash": "EVAL",
                "result": {"usable_rollout_rate": 0.4, "correct_overall": 0.02,
                           "correct_given_usable": 0.05, "n": 190,
                           "n_scorable": 170}}
               for s in (*d.leaves, d.control_state)]
    return d, mod, records


def test_all_five_leaves_are_recorded_at_the_sa_decision(tmp_path):
    d, mod, records = _retention_driver(tmp_path)
    retention = mod.PhaseADriver.emit_leaf_retention(d, records)

    assert retention["n_leaves"] == 5
    ids = [e["canonical_id"] for e in retention["entries"]]
    assert len([i for i in ids if i.startswith("leaf")]) == 5, (
        "every searched leaf must appear, advanced or not — a leaf that is "
        "simply absent from the record is unaccountable")
    assert "control-qwen3_0p6b_init_v0" in ids


def test_rejected_leaves_keep_their_evidence_and_lose_only_their_bytes(tmp_path):
    d, mod, records = _retention_driver(tmp_path)
    retention = mod.PhaseADriver.emit_leaf_retention(d, records)
    rejected = [e for e in retention["entries"]
                if not e["is_control"] and not e["advanced_to_rung2"]]
    assert len(rejected) == 3, "5 searched leaves, 2 survivors -> 3 rejected"
    for e in rejected:
        assert e["permanent_checkpoint_retained"] is False
        assert e["retention_tier"] == "TIER_4_DISPOSABLE"
        # ...but everything that makes it accountable is present.
        assert e["artifact_digest"] and e["weights_sha256"]
        assert e["search_lineage"]
        assert e["sa_probe_id"] and e["sa_evaluation_protocol_hash"]
        assert e["sa_result"]["usable_rollout_rate"] is not None
        assert e["selection_rule"]
        # And it is still physically on the pod until teardown — not deleted.
        assert e["physically_present_on_pod_until_teardown"] is True


def test_survivors_and_control_are_retained(tmp_path):
    d, mod, records = _retention_driver(tmp_path)
    retention = mod.PhaseADriver.emit_leaf_retention(d, records)
    kept = {e["canonical_id"] for e in retention["entries"]
            if e["permanent_checkpoint_retained"]}
    assert kept == {"leaf0", "leaf1", "control-qwen3_0p6b_init_v0"}
    control = next(e for e in retention["entries"] if e["is_control"])
    assert control["retention_tier"] == "TIER_1_ACTIVE_CANONICAL"


def test_the_rejection_reason_is_carried_from_the_selection(tmp_path):
    d, mod, records = _retention_driver(tmp_path)
    retention = mod.PhaseADriver.emit_leaf_retention(d, records)
    leaf4 = next(e for e in retention["entries"] if e["canonical_id"] == "leaf4")
    assert "feasibility floor" in leaf4["rejected_reason"]


def test_the_launcher_fetches_finalists_not_only_a_winner(tmp_path):
    """`unresolved_equivalence` has no winner and BOTH tied candidates are the
    result; fetching only a winner would discard the finding."""
    mod = load_launcher()
    session = mod.PhaseA.__new__(mod.PhaseA)
    import argparse
    session.a = argparse.Namespace(fetch_finalists=True,
                                   ckpt_store=str(tmp_path / "store"))
    session.scr = tmp_path
    session.say = lambda m: None
    store = tmp_path / "store"
    store.mkdir()
    (store / "leaf_retention.json").write_text(json.dumps({"entries": [
        {"canonical_id": "leaf0", "is_control": False,
         "permanent_checkpoint_retained": True},
        {"canonical_id": "leaf1", "is_control": False,
         "permanent_checkpoint_retained": True},
        {"canonical_id": "control-x", "is_control": True,
         "permanent_checkpoint_retained": True},
        {"canonical_id": "leaf2", "is_control": False,
         "permanent_checkpoint_retained": False},
    ]}))
    assert mod.PhaseA.finalists_to_fetch(session) == ["leaf0", "leaf1"], (
        "the control is already held locally and a rejected leaf is not fetched")


def test_relay_staging_is_off_by_default():
    """Measured 2026-08-15: 1.60 GiB of relay headroom against 5.61 GiB of
    leaves. A default that tried would fail on quota partway through."""
    src = (REPO / "scripts/pod/autoinit_phase_a_launch.py").read_text()
    assert 'ap.add_argument("--stage-leaves-to-relay", action="store_true", ' \
           'default=False)' in src


# --- the launcher -----------------------------------------------------------


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "phase_a_launch", REPO / "scripts/pod/autoinit_phase_a_launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_a_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


def launcher_defaults(mod):
    import argparse
    ap = argparse.ArgumentParser()
    src = (REPO / "scripts/pod/autoinit_phase_a_launch.py").read_text()
    ns = argparse.Namespace()
    # Parse the real `main()` defaults by running its parser construction.
    import re
    for m in re.finditer(r'ap\.add_argument\("(--[a-z0-9-]+)"(.*?)\)\n', src, re.S):
        flag, rest = m.group(1), m.group(2)
        name = flag[2:].replace("-", "_")
        d = re.search(r"default=([^,)]+)", rest)
        if d:
            try:
                setattr(ns, name, eval(d.group(1), vars(mod)))  # noqa: S307
            except Exception:
                setattr(ns, name, None)
        else:
            setattr(ns, name, None)
    return ns


def test_make_plan_prices_and_stays_inside_the_authorization():
    """It raised `BudgetError` before a pod could exist, in the continuation.

    The step-time model has a measured 4.15 s/step floor and refuses anything
    below it without a stated reason; a Phase-A plan priced at the measured
    end-to-end rate must supply one, or `make_plan` throws where nothing catches.
    """
    mod = load_launcher()
    a = launcher_defaults(mod)
    a.max_price = 0.99

    session = mod.PhaseA.__new__(mod.PhaseA)
    session.a = a
    session.ev = {}
    session.plan = None
    session.say = lambda msg: None
    session.check_gpu_offered = lambda: True

    class Auth:
        hard_cap_usd = 21.4538
        per_launch_hard_usd = 21.4538
        def require_within_cap(self, usd, what=""):
            if usd > self.hard_cap_usd:
                raise AuthorizationError(f"{usd} over {self.hard_cap_usd}")
        def require_within_launch_limit(self, usd, what=""):
            pass
    session.auth = Auth()

    assert mod.PhaseA.make_plan(session) is True
    plan = session.plan
    # 12 priced probes: rung 1's 6, rung 2's 3, and headroom for the conditional
    # tie-break so the watchdog cannot kill a legitimately triggered one.
    assert session.ev["priced_probes"]["total_priced"] == 12
    assert plan.hard_terminate_usd <= 21.4538, "does not fit the raised cap"
    assert plan.expected_usd < plan.soft_stop_usd < plan.hard_terminate_usd


def test_the_authorization_constant_matches_what_make_plan_prices():
    """`make_plan` calls `require_within_cap(plan.hard_terminate_usd)`. If the
    granted cap is below the priced threshold the launcher aborts at $0 — safe,
    but only discovered at launch. Checked here instead."""
    from aadistill.autoinit.phase_a import PHASE_A_AUTHORIZATION as A

    mod = load_launcher()
    a = launcher_defaults(mod)
    a.max_price = 0.99
    session = mod.PhaseA.__new__(mod.PhaseA)
    session.a, session.ev, session.plan = a, {}, None
    session.say = lambda m: None
    session.check_gpu_offered = lambda: True
    session.auth = A
    assert mod.PhaseA.make_plan(session) is True, (
        "the granted cap does not cover the plan the launcher prices")
    assert session.plan.hard_terminate_usd <= A.hard_cap_usd
    assert session.plan.hard_terminate_usd <= A.per_launch_hard_usd
    # And it fits the raised project cap with real margin.
    assert A.hard_cap_usd <= 213.00 - 191.5462


def test_the_authorization_constant_still_carries_its_placeholders():
    """The constant is a template; the issuer fills identity and time. If these
    were pre-filled, an artifact could be issued with a back-dated grant or a
    stale science-plan hash."""
    from aadistill.autoinit.phase_a import PHASE_A_AUTHORIZATION as A

    assert A.granted_utc == "PLACEHOLDER"
    assert A.science_plan_hash == "PLACEHOLDER"
    assert A.authorized_session_commit is None
    assert A.harness_source_digest is None


def test_a_non_auth_path_changed_after_the_authorized_base_is_refused(tmp_path):
    """The gap the harness digest and the auth-blob check both miss.

    `authorized_session_commit` is necessarily the clean PRE-authorization HEAD,
    because the artifact cannot be committed before it exists — so the commit the
    pod checks out is always later. Both existing checks pass on such a commit
    even if arbitrary other paths changed in the gap: the declared harness files
    are untouched, and the auth blob is exact. Only lineage catches it.

    Built on a real git repository, not a mock, because the predicate IS git.
    """
    import subprocess

    # Loaded through the same helper the other launcher tests use, not by a bare
    # `from phase_a_launch import ...` — that only resolves if some earlier test
    # happened to put the module in sys.modules, so the test passed in the full
    # suite and failed in isolation.
    mod = load_launcher()
    lineage_from_authorized_base = mod.lineage_from_authorized_base

    AUTH = "logs/autoinit_phase_a_authorization.json"
    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True)
    (repo / "scripts").mkdir()

    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (repo / "scripts" / "harness.py").write_text("harness v1\n")
    (repo / "unrelated.txt").write_text("before\n")
    git("add", "-A"); git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD").stdout.strip()

    # (a) the legitimate shape: only the authorization artifact is added.
    (repo / AUTH).write_text('{"grant": 1}\n')
    git("add", "-A"); git("commit", "-q", "-m", "authorization")
    good = git("rev-parse", "HEAD").stdout.strip()
    ok = lineage_from_authorized_base(repo, base, good, AUTH)
    assert ok["ok"] is True, ok["reason"]
    assert ok["changed_paths"] == [AUTH]

    # (b) the gap: same harness, exact auth blob, but something else moved too.
    (repo / "unrelated.txt").write_text("after\n")
    git("add", "-A"); git("commit", "-q", "-m", "a path that was never authorized")
    bad = git("rev-parse", "HEAD").stdout.strip()
    # The harness file is byte-identical and the auth blob is still exact...
    assert git("show", f"{bad}:scripts/harness.py").stdout == "harness v1\n"
    assert git("show", f"{bad}:{AUTH}").stdout == '{"grant": 1}\n'
    # ...and it is refused anyway, naming the path.
    refused = lineage_from_authorized_base(repo, base, bad, AUTH)
    assert refused["ok"] is False
    assert refused["unexpected_paths"] == ["unrelated.txt"]
    assert "unrelated.txt" in refused["reason"]

    # (c) an unrelated line of history is refused as not descending.
    git("checkout", "-q", "--orphan", "other")
    (repo / "scripts" / "harness.py").write_text("harness v1\n")
    (repo / AUTH).write_text('{"grant": 1}\n')
    git("add", "-A"); git("commit", "-q", "-m", "orphan")
    orphan = git("rev-parse", "HEAD").stdout.strip()
    off = lineage_from_authorized_base(repo, base, orphan, AUTH)
    assert off["ok"] is False and off["descends_from_base"] is False

    # (d) no declared base cannot silently mean "anything goes".
    none = lineage_from_authorized_base(repo, None, good, AUTH)
    assert none["ok"] is False and "no authorized_session_commit" in none["reason"]

    # (e) THE WIRING. The helper being correct is worthless if the launcher does
    # not consult it — verified by mutation: deleting the refusal from
    # `verify_session_commit` left every other test green. So drive the real
    # method against this repository, with the harness digest and the auth blob
    # both deliberately VALID, and confirm the commit is still refused.
    import argparse
    import hashlib

    harness_files = ("scripts/harness.py",)

    def digest_at(ref):
        rows = []
        for rel in sorted(harness_files):
            blob = subprocess.run(["git", "show", f"{ref}:{rel}"],
                                  cwd=repo, capture_output=True).stdout
            rows.append(f"{rel}:{hashlib.sha256(blob).hexdigest()}\n")
        return hashlib.sha256("".join(rows).encode()).hexdigest()

    class _Auth:
        harness_source_files = harness_files
        harness_source_digest = digest_at(good)     # valid at BOTH commits
        authorized_session_commit = base

    saved_root, saved_auth = mod.REPO_ROOT, mod.AUTH_PATH
    try:
        mod.REPO_ROOT, mod.AUTH_PATH = repo, AUTH
        session = mod.PhaseA.__new__(mod.PhaseA)
        session.auth = _Auth()
        session.ev = {}
        session.say = lambda m: None

        session.a = argparse.Namespace(session_commit=good)
        assert mod.PhaseA.verify_session_commit(session) is True, (
            "the legitimate shape must still pass")

        session.a = argparse.Namespace(session_commit=bad)
        session.ev = {}
        assert mod.PhaseA.verify_session_commit(session) is False, (
            "a commit whose harness digest and auth blob are both valid, but "
            "which changed a non-auth path after the authorized base, must be "
            "refused before a pod can exist")
        check = session.ev["session_commit_check"]
        # ...and refused for the RIGHT reason: the other two gates passed.
        assert check["harness_matches"] is True
        assert check["commit_carries_this_authorization"] is True
        assert check["lineage"]["unexpected_paths"] == ["unrelated.txt"]
    finally:
        mod.REPO_ROOT, mod.AUTH_PATH = saved_root, saved_auth


def test_the_launcher_polls_for_the_markers_its_own_driver_emits():
    """The poll loop watched for PREFLIGHT_* while the driver emitted its own."""
    mod = load_launcher()
    driver = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    for marker in mod.PhaseA.failure_markers:
        assert f'mark("{marker}")' in driver, (
            f"the launcher polls for {marker} but the driver never emits it")
    assert 'mark("ALL_DONE")' in driver
    for marker in mod.PhaseA.incomplete_markers:
        assert marker in mod.PhaseA.failure_markers


def test_the_launcher_and_driver_agree_on_where_artifacts_go():
    mod = load_launcher()
    driver = (REPO / "scripts/pod/autoinit_phase_a_driver.py").read_text()
    assert f'artifacts/audit/{mod.PhaseA.audit_dirname}' in driver
    assert mod.PhaseA.evidence_filename in driver
    for name in mod.PhaseA.report_names:
        assert name in driver, f"the launcher fetches {name}; the driver never writes it"


def test_the_launcher_declares_a_phase_a_authorization_not_a_spend_one():
    src = (REPO / "scripts/pod/autoinit_phase_a_launch.py").read_text()
    assert "PhaseAAuthorization" in src
    assert "SESSION_KIND" in src, "setup would load the type that always says no"
    # The retarget must actually be there: the inherited `Preflight.__init__`
    # resolves `SpendAuthorization` from the base module's globals at call time,
    # so this line is what makes a Phase-A session load the type that can say yes.
    assert "_preflight.SpendAuthorization = PhaseAAuthorization" in src
    assert "_preflight.PREFLIGHT_PLAN_V1 = PHASE_A_PLAN_V1" in src


def test_phase_a_authorization_satisfies_everything_the_launcher_calls_on_auth():
    """The base launcher is inherited wholesale, so the substituted type must
    answer every call it makes. A missing method would surface as an
    `AttributeError` inside `__init__` — on the dev box, before a pod, but only
    if something exercises it. This does."""
    import inspect
    import re

    base = (REPO / "scripts/pod/autoinit_preflight_launch.py").read_text()
    phase = (REPO / "scripts/pod/autoinit_phase_a_launch.py").read_text()
    used = set(re.findall(r"self\.auth\.([a-z_]+)", base + phase))
    assert used, "the detector found no auth usage; it is broken"

    a = _auth()
    missing = [name for name in sorted(used) if not hasattr(a, name)]
    assert not missing, (
        f"PhaseAAuthorization is substituted for SpendAuthorization in the "
        f"inherited launcher but lacks {missing}")
    # And the attributes it reads, not just the methods it calls.
    for attr in ("hard_cap_usd", "harness_source_files", "harness_source_digest",
                 "authorized_session_commit", "science_plan_hash"):
        assert hasattr(a, attr), attr
    # `as_dict` must be serializable: the launcher writes it into the evidence.
    json.dumps(a.as_dict())
    assert inspect.signature(a.require_within_cap).parameters.keys() >= {
        "projected_usd", "what"}


def test_setup_routes_by_session_kind_and_defaults_to_the_narrow_type():
    setup = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    assert 'SESSION_KIND="${SESSION_KIND:-spend}"' in setup, (
        "an unset SESSION_KIND must mean the narrow type, so the preflight and "
        "the continuation are unaffected")
    # The assertion guarding every non-Phase-A session must still be there,
    # literally.
    assert "assert a.allows_phase_a is False" in setup
    assert "assert a.allows_phase_a is True" in setup


# --- the artifact specs -----------------------------------------------------


def _load_specs(path):
    sys.path.insert(0, str(REPO / "scripts/pod"))
    from collect_artifacts import load_specs
    return load_specs(str(path))


@pytest.mark.parametrize("spec_name", ["phase_a_artifacts.json",
                                       "phase_a_artifacts_failed.json"])
def test_the_specs_load_with_the_real_parser(spec_name):
    """`ArtifactSpec(**item)` rejects an unknown key; collection died of that
    once already, on the pod, after the run had finished."""
    specs = _load_specs(REPO / "configs/autoinit" / spec_name)
    assert specs


def test_the_success_spec_requires_what_the_driver_actually_writes(tmp_path):
    """A successful Phase A must not fail its own artifact manifest."""
    from aadistill.infrastructure.artifact_gate import build_manifest

    root = tmp_path / "artifacts"
    audit = root / "audit/autoinit_phase_a"
    (audit / "probes").mkdir(parents=True)
    (audit / "configs").mkdir(parents=True)
    for name in ("phase_a_evidence.json", "attested_evaluation_protocol.json",
                 "search_result.json", "phase_a_result.json",
                 "rung1_selection.json", "rung2_selection.json",
                 "leaf_retention.json", "engine_probe.json"):
        (audit / name).write_text("{}")
    for i in range(9):
        (audit / "probes" / f"p{i}.json").write_text("{}")
        (audit / "configs" / f"p{i}.json").write_text("{}")
        (audit / f"p{i}_recovery_search.json").write_text("{}")
        (audit / f"p{i}_per_sample.jsonl").write_text("{}\n")
        gen = root / "eval/phase_a" / f"p{i}"
        gen.mkdir(parents=True)
        (gen / f"p{i}.generations.jsonl").write_text("{}\n")
    search = root / "autoinit/phase_a_search"
    search.mkdir(parents=True)
    (search / "states.jsonl").write_text("{}\n")

    manifest = build_manifest(str(root), _load_specs(
        REPO / "configs/autoinit/phase_a_artifacts.json"),
        created_utc="2026-08-15T00:00:00Z", settle_seconds=0)
    assert manifest.ok, f"missing {manifest.missing}"


def test_the_failed_spec_requires_only_the_evidence(tmp_path):
    """A gate that demands artifacts a failed run refused to produce keeps the
    most expensive pod in the project billing."""
    from aadistill.infrastructure.artifact_gate import build_manifest

    root = tmp_path / "artifacts"
    audit = root / "audit/autoinit_phase_a"
    audit.mkdir(parents=True)
    (audit / "phase_a_evidence.json").write_text("{}")
    manifest = build_manifest(str(root), _load_specs(
        REPO / "configs/autoinit/phase_a_artifacts_failed.json"),
        created_utc="2026-08-15T00:00:00Z", settle_seconds=0)
    assert manifest.ok, f"missing {manifest.missing}"
