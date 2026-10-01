"""The shortened A-bsz3 design: its derivations, its gates, and its price.

Three things are checked here and they are different in kind.

**The gates are mutated, not merely exercised.** Every predicate the
structural comparison can refuse on is driven BOTH ways. A gate asserted only
on inputs that pass it is a gate nobody has tested; this repository has
shipped one of those before and it could not have passed for any record.

**The design's numbers are checked against their sources.** The control values
must be attempt75's, the identities the preregistration's, and the fail-fast
threshold a function of the per-seed spread C3 actually observed -- so the
threshold test feeds a DIFFERENT spread and requires the answer to move. A
derivation that ignores its input is a constant with extra steps.

**The price is checked where it differs from C3's.** The overrun factor reaches
the parent replay here and did not there, because on a step-1 session the
replay is more than half the work; the ceiling rounds up; and the book a shape
is charged to follows what the session DOES, not what its scientific claim is
called.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "autoinit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from compare_a_bsz3 import (  # noqa: E402
    _counter_comparison, _incumbent_gate, _score_comparison, _timing,
    pair_rounds, void_reasons,
)
from experiments.phase_c3 import a_bsz3_pricing as pricing  # noqa: E402
from experiments.phase_c3.a_bsz3 import ABsz3Error, frozen_identities  # noqa: E402

import write_a_bsz3_adoption as adoption  # noqa: E402

ADOPTION_PLAN = REPO / "logs/stages/stage-1/phase_c3/plans/a_bsz3_adoption.json"
WITHDRAWN_PLAN = REPO / "logs/stages/stage-1/phase_c3/plans/a_bsz3_noninferiority.json"
PRICING_RECORD = REPO / "logs/stages/stage-1/phase_c3/plans/a_bsz3_pricing.json"


# --- 1. the structural comparison's gates, driven both ways ----------------


def _record(*, digest="aa", tokens=100, executed=110, padded=10,
            repeatable=True, forwards=4):
    return {
        "artifact_digest": digest,
        "calibration_tokens": tokens,
        "digest_repeatable_within_session": repeatable,
        "observed_execution_counters": {
            "physical_forward_invocations": forwards,
            "executed_positions": executed,
            "valid_positions": executed - padded,
            "padded_positions": padded},
    }


def test_the_masking_invariant_is_checked_and_can_fail():
    """`executed - padded == calibration_tokens`, both ways round."""
    good = _counter_comparison(_record(), _record(), None)
    assert good["masking_invariant_holds"] is True

    #: One position unaccounted for: the loop says 110 ran and 10 were
    #: padding, while the collector accumulated 99.
    bad = _counter_comparison(_record(), _record(tokens=99), None)
    assert bad["masking_invariant_holds"] is False
    assert "executed_positions - padded_positions" in bad["_masking_invariant_is"]


def test_an_absent_counter_leaves_the_invariant_unknown_not_true():
    """A missing field must not read as agreement."""
    rec = _record()
    rec["observed_execution_counters"]["padded_positions"] = None
    assert _counter_comparison(rec, rec, None)["masking_invariant_holds"] is None


def test_the_zero_cost_prediction_is_compared_and_can_disagree():
    """The $0 model predicted these counters before any pod existed.

    The comparison is only worth making because the two sides are computed by
    different code: `padding_profile` from the item lengths, the observation by
    the operator's own loop. So the disagreeing case must be reachable.
    """
    zero = {"protocols": {
        "A_bsz1": {"padding": {"n_groups": 4, "executed_positions": 110,
                               "padded_positions": 10}},
        "A_bsz3": {"padding": {"n_groups": 2, "executed_positions": 110,
                               "padded_positions": 10}}}}
    held = _counter_comparison(_record(forwards=4), _record(forwards=2), zero)
    assert held["prediction_held"] is True

    #: bsz3 was predicted to need 2 forwards and ran 3.
    missed = _counter_comparison(_record(forwards=4), _record(forwards=3), zero)
    assert missed["prediction_held"] is False
    assert missed["prediction_vs_observation"]["A_bsz3"]["forwards_agree"] is False


def test_no_prediction_is_reported_as_unavailable_not_as_agreement():
    out = _counter_comparison(_record(), _record(), None)
    assert "prediction_held" not in out
    assert "_prediction_unavailable" in out


def test_the_incumbent_gate_has_three_answers():
    """Match, mismatch, and not-asked -- and not-asked is not a pass."""
    assert _incumbent_gate("abc", "abc")["matches"] is True
    assert _incumbent_gate("abc", "def")["matches"] is False
    unchecked = _incumbent_gate("abc", None)
    assert unchecked["checked"] is False
    assert "matches" not in unchecked


@pytest.mark.parametrize(
    "mutate,expect",
    [
        (lambda r, c: None, 0),
        #: a masking defect
        (lambda r, c: c["execution_counters"].__setitem__(
            "masking_invariant_holds", False), 1),
        #: the two protocols averaged over different denominators
        (lambda r, c: c.__setitem__("calibration_tokens_identical", False), 1),
        #: one protocol does not reproduce its own digest
        (lambda r, c: r["A_bsz3"].__setitem__(
            "digest_repeatable_within_session", False), 1),
        #: the reference protocol is not the incumbent
        (lambda r, c: c["incumbent_digest_gate"].update(
            {"checked": True, "matches": False, "observed": "x" * 64,
             "expected": "y" * 64}), 1),
        #: everything at once
        (lambda r, c: (c.__setitem__("calibration_tokens_identical", False),
                       r["A_bsz1"].__setitem__(
                           "digest_repeatable_within_session", False)), 2),
    ],
    ids=["clean", "masking", "denominators", "repeatability", "incumbent",
         "two-at-once"])
def test_every_void_reason_is_reachable(mutate, expect):
    """A void comparison is not a result, so each way of becoming one must be
    detectable -- including when several hold together."""
    results = {"A_bsz1": _record(), "A_bsz3": _record(digest="bb")}
    comparison = {
        "calibration_tokens_identical": True,
        "execution_counters": {"masking_invariant_holds": True},
        "incumbent_digest_gate": {"checked": False},
    }
    mutate(results, comparison)
    assert len(void_reasons(results, comparison)) == expect


# --- 2. the runtime measurement protocol -----------------------------------


def _round(i, warm, scorer):
    return {"round": i, "warm_up": warm, "scorer_seconds": scorer,
            "suffix_seconds": scorer + 1.0, "peak_vram_gib": None,
            "artifact_digest": "aa"}


def test_the_warm_up_round_is_excluded_from_the_timing():
    """It is excluded by NAME, not by being first.

    The first round of each protocol pays for allocator growth, autotuning and
    kernel selection. Including it would report the arms' order as part of the
    protocol difference.
    """
    t = _timing([_round(0, True, 99.0), _round(1, False, 10.0),
                 _round(2, False, 12.0)])
    assert t["n_timed_rounds"] == 2
    assert t["warm_up_rounds_excluded"] == 1
    assert t["_fell_back_to_the_warm_up"] is False
    assert t["scorer_seconds"]["values"] == [10.0, 12.0]
    assert t["scorer_seconds"]["mean"] == 11.0


def test_a_single_round_falls_back_and_says_so():
    """Reporting nothing would be worse than reporting a warm number that
    declares itself warm."""
    t = _timing([_round(0, True, 10.0)])
    assert t["n_timed_rounds"] == 1
    assert t["_fell_back_to_the_warm_up"] is True
    assert t["warm_up_rounds_excluded"] == 0


def test_an_unpaired_round_is_dropped_rather_than_widening_one_sample():
    """Interleaving only means something while the rounds stay paired.

    A cleanup failure can stop the loop between the two protocols. The odd
    round out is precisely the one a drifting device moves, so keeping it
    would give one protocol an extra sample taken at a moment the other never
    saw.
    """
    paired, common, dropped = pair_rounds({"a": [1, 2, 3], "b": [1, 2, 3]})
    assert common == 3 and dropped == {}
    assert paired["a"] == [1, 2, 3]

    paired, common, dropped = pair_rounds({"a": [1, 2, 3], "b": [1, 2]})
    assert common == 2
    assert dropped == {"a": 1}
    assert paired["a"] == [1, 2] and paired["b"] == [1, 2]


def test_rank_correlation_separates_scores_from_selections():
    """Two protocols can differ in every score and agree on every selection.

    The counts cannot see that and this can, so identical and perturbed
    inputs must give different answers.
    """
    base = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
    same = _score_comparison(base, [row[:] for row in base])
    assert same["identical"] is True
    assert same["rank_correlation"]["overall"] == pytest.approx(1.0)
    assert same["score_drift"]["abs_max"] == 0.0

    #: Order preserved, values moved: a correlation of 1.0 with real drift.
    scaled = _score_comparison(base, [[v * 1.01 for v in row] for row in base])
    assert scaled["identical"] is False
    assert scaled["rank_correlation"]["overall"] == pytest.approx(1.0)
    assert scaled["score_drift"]["abs_max"] > 0.0

    #: Order broken: the correlation must fall.
    swapped = _score_comparison(base, [[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    assert swapped["rank_correlation"]["overall"] < 1.0


def test_mismatched_score_traces_are_unavailable_rather_than_zero():
    assert "_unavailable" in _score_comparison(None, [[1.0]])
    assert "_unavailable" in _score_comparison([[1.0, 2.0]], [[1.0], [2.0]])


# --- 3. the design's numbers come from their sources -----------------------


def test_the_frozen_identities_are_read_and_refuse_to_be_invented(tmp_path):
    ids = frozen_identities()
    prereg = json.loads(
        (REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json")
        .read_text())
    assert ids["incumbent_artifact_digest"] == \
        prereg["arms"]["A_incumbent"]["artifact_digest"]
    assert ids["shared_parent_artifact_digest"] == \
        prereg["shared_parent"]["artifact_digest"]
    assert ids["recovery_seeds"] == prereg["seeds"]["recovery"]

    missing = tmp_path / "nope.json"
    with pytest.raises(ABsz3Error):
        frozen_identities(missing)

    hollow = tmp_path / "hollow.json"
    hollow.write_text(json.dumps({"seeds": {"recovery": [1]}}))
    with pytest.raises(ABsz3Error):
        frozen_identities(hollow)


def test_the_fail_fast_threshold_is_derived_from_the_observed_spread():
    """Feed it a different spread and the answer must move.

    A gate threshold that is the same whatever the instrument did is a written
    constant wearing a gate's clothing. The real derivation: 1.25x the largest
    single-seed excursion C3 observed between arms whose POOLED differences
    were within +/-0.002, rounded up to the next 0.005.
    """
    real = json.loads(
        (REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i/"
         "c3_decision.json").read_text())
    got = adoption.seed1_stop_threshold(real)
    assert got["threshold_delta"] == -0.030
    assert got["derivation"]["observed_max_abs_per_seed_delta"] == \
        pytest.approx(0.022353, abs=1e-6)

    #: A quieter instrument earns a tighter stop.
    quiet = {"contrasts": {"primary": {
        "delta": 0.0, "per_seed_delta": [0.001, -0.002, 0.0015]}}}
    assert adoption.seed1_stop_threshold(quiet)["threshold_delta"] == -0.005

    #: A noisier one earns a looser one.
    noisy = {"contrasts": {"primary": {
        "delta": 0.0, "per_seed_delta": [0.05, -0.04, 0.01]}}}
    assert adoption.seed1_stop_threshold(noisy)["threshold_delta"] == -0.065


def test_the_seed_one_stop_sits_above_the_measured_seed_level_noise():
    """The point of the threshold, stated as an assertion.

    If the stop were at or below the largest same-protocol excursion this
    instrument has produced, it would fire on protocols that are behaving
    identically -- and a single seed would veto the study for noise.
    """
    plan = json.loads(ADOPTION_PLAN.read_text())
    stop = plan["step_2"]["fail_fast"]["after_seed_1"]
    observed = stop["derivation"]["observed_max_abs_per_seed_delta"]
    assert abs(stop["threshold_delta"]) > observed
    assert abs(stop["threshold_delta"]) >= 3 * adoption.SESOI


def test_the_controls_are_attempt75s_values_and_not_restated():
    """Every control number must equal the probe record it came from."""
    plan = json.loads(ADOPTION_PLAN.read_text())
    probes = json.loads(
        (REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i/"
         "c3_probe_results.json").read_text())
    source = {p["seed"]: p for p in probes["probes"]
              if p["arm"] == "A_incumbent"}
    control = plan["step_2"]["control"]
    assert control["source_run"] == "attempt75"
    assert len(control["probes"]) == 3
    for row in control["probes"]:
        src = source[row["seed"]]
        assert row["correct_overall"] == src["rates"]["correct_overall"]
        assert row["usable_rollout_rate"] == src["rates"]["usable_rollout_rate"]
        assert row["correct"] == src["counts"]["correct"]
        assert row["per_sample_sha256"] == src["per_sample_sha256"]
        assert row["initialization_artifact_digest"] == \
            src["initialization_artifact_digest"]


def test_the_design_is_the_shortened_one_and_claims_no_noninferiority():
    plan = json.loads(ADOPTION_PLAN.read_text())
    assert plan["step_2"]["treatment"]["max_new_probes"] == 3
    assert plan["step_2"]["treatment"]["seeds_in_fail_fast_order"] == \
        frozen_identities()["recovery_seeds"]
    #: The arms are not re-measured: only the treatment is trained.
    assert plan["step_2"]["control"]["arm"] == "A_incumbent"
    #: And the claim is bounded in the document itself.
    forbidden = " ".join(plan["_what_may_not_be_claimed_from_this"]).lower()
    assert "non-inferiority" in forbidden
    assert plan["step_1"]["runtime_threshold"]["none"] is True
    #: The cross-session price is stated, not implied.
    assert "DIFFERENT SESSIONS" in \
        plan["claim_boundary"]["the_price_of_the_shortened_design"]


def test_the_design_blocks_adoption_while_one_id_serves_both_purposes():
    """Scientific identity != materialization identity, and the plan says so.

    `compute_state_id` binds neither the `ExecutionConfig` nor the artifact
    digest, so two executions that differ in BYTES collide on one resumable,
    deduplicable id. While that holds, a differing digest may not be adopted
    into D1/D2/D3 on the strength of a behavioural result -- and the plan must
    carry that precondition explicitly rather than leaving it to be noticed.

    A disjunction, so it survives the mechanism being built: if the ids ever
    differ, the precondition is satisfied by construction and this test says
    so instead of failing.
    """
    from aadistill.initialization.specs.state import (
        OperatorStep, compute_state_id,
    )

    common = dict(index=0, kind="ATTENTION",
                  impl_id="attention.activation_importance_v1",
                  impl_signature_hash="sig", profile_id="p", profile_hash="ph",
                  config_hash="ch", seed=0, result_spec_hash="rs")
    collide = (compute_state_id("r", "t", (OperatorStep(
                   **common, trace={"micro_batch_size": 1}),))
               == compute_state_id("r", "t", (OperatorStep(
                   **common, trace={"micro_batch_size": 3}),)))

    plan = json.loads(ADOPTION_PLAN.read_text())
    sem = plan["identity_semantics"]
    if not collide:
        return  # the mechanism exists; the precondition is moot

    assert "ExecutionConfig" in sem["what_it_does_NOT_bind"]
    assert "the resulting artifact digest" in sem["what_it_does_NOT_bind"]

    differ = sem["adoption_logic"]["digests_differ"]
    assert "MAY NOT be adopted into D1/D2/D3" in differ, (
        "the plan does not block automatic adoption of a differing artifact")
    assert "materialization/resume identity" in differ
    #: And a behavioural pass must not be mistaken for clearing it. Asserted
    #: on substance with an accepted SET of wordings, not one phrase: a lock
    #: on a single sentence expires the first time the sentence improves.
    lift = sem["adoption_logic"][
        "_the_behavioural_result_does_not_lift_this"].lower()
    assert "resume" in lift
    assert any(v in lift for v in ("says nothing", "does not", "cannot"))
    assert "_what_passing_step_2_does_NOT_authorize" in plan["step_2"]
    #: The forbidden shortcut is named, because it is the obvious one.
    assert "activation_importance_bsz3" in sem["_forbidden_resolution"]
    #: And the mechanism is not built speculatively.
    assert "NOT BUILT" in sem["_required_shape_if_it_is_ever_needed"]


def test_the_fail_fast_stop_is_never_called_an_equivalence_margin():
    """It is a catastrophic-regression stop. Clearing it bounds nothing."""
    plan = json.loads(ADOPTION_PLAN.read_text())
    stop = plan["step_2"]["fail_fast"]["after_seed_1"]
    text = stop["_it_is_not_an_equivalence_margin"]
    assert "never be reported as an equivalence margin" in text
    assert "does not bound the difference" in text


def test_the_structural_classification_names_the_materialization_consequence():
    """The record a reader sees must not stop at "the digests differ"."""
    import compare_a_bsz3 as drv

    src = Path(drv.__file__).read_text()
    assert "DISTINCT_NUMERICAL_MATERIALIZATION_PROTOCOL" in src
    assert "TRANSPARENT_EXECUTION_OPTIMIZATION" in src
    assert "_what_a_differing_digest_implies" in src


def test_the_protocol_identity_requirements_bind_attempt75s_actual_values():
    """Reusing another session's controls is sound only while the two fields
    are one field. C2's confirmation set carried three generation fingerprints
    and produced no verdict."""
    plan = json.loads(ADOPTION_PLAN.read_text())
    probes = json.loads(
        (REPO / "logs/stages/stage-1/phase_c3/analyses/attempt75_stage_i/"
         "c3_probe_results.json").read_text())
    req = plan["step_2"]["protocol_identity_requirements"]["must_equal_attempt75"]
    assert req["generation_protocol_fingerprint"] == \
        probes["observed_generation_fingerprint"]
    assert req["scoring_contract_digest"] == \
        probes["scoring_contract"]["digest"]
    assert req["battery_manifest_sha256"] == \
        probes["battery"]["manifest_sha256"]
    assert req["n_scorable"] == probes["decision_inputs_audit"]["n_scorable"]


def test_the_withdrawn_design_is_marked_and_carries_no_live_hash():
    """A withdrawn preregistration must not be loadable as a live one."""
    doc = json.loads(WITHDRAWN_PLAN.read_text())
    assert doc["status"] == "WITHDRAWN"
    assert doc["superseded_by"] == "a_bsz3_adoption.json"
    assert "preregistration_sha256" not in doc, (
        "a live preregistration hash is a thing a launcher binds to")
    assert doc["withdrawn_preregistration_sha256"] == \
        adoption.WITHDRAWN["preregistration_sha256"]
    #: The findings that outlived the design are still here.
    assert doc["what_survives_the_withdrawal"]["the_power_finding"][
        "power_at_true_equivalence"]["8_seeds"] == 0.929


def test_the_committed_plan_matches_what_the_writer_produces_today():
    """A committed derived record expires silently. Regenerate and compare."""
    fresh = adoption.build(rate_note=json.loads(
        ADOPTION_PLAN.read_text())["pricing"]["_rate_note"])
    committed = json.loads(ADOPTION_PLAN.read_text())
    committed.pop("design_sha256")
    assert fresh == committed


def test_the_plans_self_hash_is_over_its_own_content():
    committed = json.loads(ADOPTION_PLAN.read_text())
    claimed = committed.pop("design_sha256")
    import hashlib
    recomputed = hashlib.sha256(json.dumps(
        committed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert recomputed == claimed


# --- 4. the price, where it differs from C3's -----------------------------


def test_the_overrun_factor_reaches_the_parent_replay():
    """C3 applied it to training and evaluation only, because everything else
    was rounding error at nine probes. Here the replay is more than half the
    work, and a ceiling that excluded it would not bound anything."""
    hard = pricing._minutes("step1_structural", worst=True)
    expected = pricing._minutes("step1_structural", worst=False)
    assert hard["parent_replay"] == pytest.approx(
        expected["parent_replay"] * pricing.OVERRUN_FACTOR)
    #: setup has its own worst case rather than a factor
    assert hard["setup"] == pricing.SETUP_WORST_CASE_MINUTES
    #: and the pre-provider gates run before the meter
    assert hard["pre_provider_gates"] == expected["pre_provider_gates"]


def test_the_hard_ceiling_rounds_up_and_the_expected_does_not():
    """A limit rounds down; a ceiling rounds up. A budget planner already
    refused once to build a plan terminating $0.00003 above its grant."""
    p = pricing.price("step2_three_seeds", 1.09)
    exact = p.hard_minutes / 60.0 * p.billed_rate_usd_per_hour
    assert p.hard_usd >= exact
    assert p.hard_usd - exact < 1e-4
    assert p.hard_usd > p.expected_usd


def test_three_seeds_cost_exactly_two_more_probes_than_the_fail_fast_shape():
    one = pricing._minutes("step2_fail_fast", worst=False)
    three = pricing._minutes("step2_three_seeds", worst=False)
    per_probe = (pricing.COMPONENTS["recovery_probe"]["minutes"]
                 + pricing.COMPONENTS["evaluation_probe"]["minutes"])
    delta = (sum(three.values()) - sum(one.values()))
    assert delta == pytest.approx(2 * per_probe, abs=1e-6)


def test_the_book_follows_what_the_session_does_not_what_it_is_called():
    """The study's scientific claim is 'engineering'. That does not move it
    between allowances: step 2 trains probes and consumes the frozen
    confirmation battery, and the package says the engineering allowance
    authorizes neither."""
    assert pricing.SHAPES["step1_structural"]["book"] == \
        "gpu_engineering_allowance"
    assert pricing.SHAPES["step1_structural"]["probes"] == 0
    for shape in ("step2_fail_fast", "step2_three_seeds"):
        assert pricing.SHAPES[shape]["book"] == "formal_allowance"
        assert pricing.SHAPES[shape]["probes"] >= 1

    terms = json.loads(
        (REPO / "configs/experiments/phase_c1/authorization.json").read_text())
    forbidden = " ".join(terms["execution_package"][
        "engineering_allowance_does_not_authorize"]).lower()
    assert "confirmation battery" in forbidden


def test_fundability_is_judged_against_the_booked_allowance_only():
    """An unspent balance in the OTHER book must not fund a shape.

    The limits bind separately, which is the whole reason the package states
    them separately -- and with the formal allowance overspent, a pricer that
    looked at the project cap alone would have called step 2 affordable.
    """
    env = {"per_session_envelope_usd": 30.0, "project_cap_usd": 400.0,
           "cumulative_spend_usd": 383.3623,
           "formal_remaining_usd": 25.0,
           "engineering_remaining_usd": 0.10}
    step1 = pricing.fundability("step1_structural", 1.09, env)
    assert step1["FUNDABLE"] is False
    assert "gpu_engineering_allowance" in step1["shortfalls_usd"]

    #: and the reverse: an empty engineering allowance does not block step 2
    env2 = dict(env, engineering_remaining_usd=0.0, formal_remaining_usd=25.0)
    assert pricing.fundability("step2_three_seeds", 1.09, env2)["FUNDABLE"]


def test_the_live_position_is_what_the_record_reports():
    """The committed pricing record must agree with the derived balances."""
    doc = json.loads(PRICING_RECORD.read_text())
    assert doc["shapes"]["step1_structural"]["book"] == \
        "gpu_engineering_allowance"
    for shape in ("step2_fail_fast", "step2_three_seeds"):
        row = doc["shapes"][shape]
        assert row["FUNDABLE"] is False, (
            "the formal allowance is overspent; a record saying otherwise "
            "has gone stale")
        assert "formal_allowance" in row["shortfalls_usd"]
