"""The D1 derivations: the selection-noise arithmetic, the space, and the record.

Three things are checked, and the first is the one that caught a real bug.

**The quadrature.** `expected_max_of_standard_normals` integrates a survival
function, and the first version wrote the negative tail as the positive tail
again. The two cancelled, every selection bias came out exactly `0.0`, and the
design table read "no design has a selection problem". Two closed forms pin it.

**The space.** Enumerated from the registry through the same functions the
search itself calls, so a changed registry moves the number rather than leaving
a stale literal in a plan.

**The record.** Generated, at a fixed point, and stating both blockers. A design
document that read as ready when it is not is how a grant gets requested for an
experiment that could not have produced a valid result.

No `pytest.skip` anywhere in this file, deliberately: every input is either in
git or derived, so there is no filesystem premise for the skip-predicate audit
to resolve.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/data", "scripts/autoinit"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.phase_d1 import selection_noise as noise  # noqa: E402

DESIGN = REPO / "logs/stages/stage-1/phase_d1/plans/d1_design.json"
CAPACITY = REPO / "logs/stages/stage-1/phase_d1/analyses/d1_evidence_capacity.json"


class TestTheQuadrature:
    """`E[max of k standard normals]`, against the values that are known."""

    def test_k_one_is_zero(self):
        assert noise.expected_max_of_standard_normals(1) == 0.0

    def test_k_two_is_one_over_root_pi(self):
        got = noise.expected_max_of_standard_normals(2)
        assert got == pytest.approx(1.0 / math.sqrt(math.pi), abs=1e-6)

    def test_the_self_check_runs_and_would_raise(self):
        assert set(noise.self_check()) == set(noise.CLOSED_FORMS)
        #: And it really would refuse. Driven by tightening the tolerance past
        #: what any quadrature achieves, because a self-check that cannot fail
        #: is the same as no self-check — which is exactly the state the cancelled
        #: tails left this function in.
        with pytest.raises(ValueError, match="quadrature is wrong"):
            noise.self_check(tolerance=1e-18)

    def test_it_is_monotone_in_k(self):
        values = [noise.expected_max_of_standard_normals(k) for k in range(1, 8)]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    def test_k_below_one_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            noise.expected_max_of_standard_normals(0)


class TestTheMeasuredNoiseInput:
    def test_the_seed_sd_is_derived_from_the_published_deltas(self):
        """Not a transcribed constant: a reader can check it against A3's
        closeout, and a corrected delta moves it automatically."""
        deltas = noise.A3_SEED_DELTAS
        mean = sum(deltas) / len(deltas)
        expected = math.sqrt(sum((d - mean) ** 2 for d in deltas)
                             / (len(deltas) - 1))
        assert noise.SEED_SD == pytest.approx(expected, rel=1e-12)

    def test_it_agrees_in_magnitude_with_the_independent_cross_check(self):
        """C3 measured a seed-level spread as a RANGE on three seeds. A range and
        a standard deviation are different statistics, so this is a magnitude
        check and not an equality — but an order-of-magnitude disagreement would
        mean one of the two measurements describes something else."""
        assert 0.3 < noise.SEED_SD / noise.C3_SEED_LEVEL_SPREAD_RANGE < 1.2


class TestTheDesignTrade:
    def test_more_screening_seeds_always_reduce_the_inflation(self):
        for k in (2, 3, 5):
            values = [noise.screening_estimate_inflation(k, m)
                      for m in (1, 2, 3)]
            assert values == sorted(values, reverse=True)

    def test_more_candidates_always_increase_the_inflation(self):
        for m in (1, 2, 3):
            values = [noise.screening_estimate_inflation(k, m)
                      for k in (2, 3, 4, 5)]
            assert values == sorted(values)

    def test_no_screening_is_an_honest_zero(self):
        assert noise.screening_estimate_inflation(5, 0) == 0.0
        assert noise.advance_probability(1, 2) == 1.0

    def test_discrimination_falls_as_the_field_widens(self):
        """The half of the trade an inflation figure alone would hide: a design
        can have a low inflation and still be a coin flip."""
        probabilities = [noise.advance_probability(k, 1) for k in (2, 3, 4, 5)]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_the_report_is_deterministic(self):
        """It uses a seeded generator, so the design table is a property of the
        arithmetic rather than of when it ran."""
        assert noise.report()["rows"] == noise.report()["rows"]


class TestTheClaimBoundaryOfTheNoiseModel:
    """The correction of 2026-10-03, asserted so it cannot drift back.

    This class replaces a test named
    `test_the_c2_design_exceeds_the_sesoi_and_the_d1_design_does_not`, which
    asserted `c2 > SESOI > d1` and so encoded the wrong argument as a
    requirement. The max-of-K inflation is the winner's curse on the SCREENING
    estimate; it is not a validity condition for a confirmation rung measured on
    fresh disjoint prompts and fresh seeds, and no design may be admitted or
    rejected by comparing it to the SESOI.
    """

    def test_no_design_row_carries_a_sesoi_comparison(self):
        """The structural guard. The filter that encoded the wrong rule read a
        boolean column off these rows, so the regression is that no such column
        exists — not merely that nothing currently reads one."""
        from experiments.phase_d1 import search_space as d1

        for row in d1.designs():
            for key in row:
                assert "sesoi" not in key.lower(), (
                    f"{key!r} invites a design to be admitted or rejected by "
                    "comparing a screening-estimate property to the SESOI")

    def test_the_three_design_numbers_are_stated_not_selected(self):
        """K, screening seeds and confirmation seeds are a recorded judgement.
        Mutating the grid's probabilities must not move them."""
        from autoinit.write_d1_design import (
            D1_CONFIRMATION_SEEDS,
            D1_SCREENING_SEEDS,
            D1_TOP_K,
            behavioural_design,
        )

        design = behavioural_design()
        #: Top-K widened 2 -> 4 on 2026-10-07 (downstream policy only); the two
        #: seed counts are unchanged. The point of this test is that the three
        #: numbers come from the DECLARED constants and that perturbing the
        #: grid's probabilities cannot move them -- not that any particular
        #: width is frozen forever.
        assert (design["top_k"], design["screening_seeds"],
                design["confirmation_seeds"]) == \
            (D1_TOP_K, D1_SCREENING_SEEDS, D1_CONFIRMATION_SEEDS) == (4, 2, 3)

    def test_the_design_records_how_the_numbers_were_chosen(self):
        from autoinit.write_d1_design import behavioural_design

        design = behavioural_design()
        why = design["_how_these_three_numbers_were_chosen"]
        assert "pragmatic balance" in why
        assert "NOT claimed" in why and "optimum" in why

    def test_the_design_carries_the_claim_boundary(self):
        from autoinit.write_d1_design import behavioural_design

        boundary = behavioural_design()["_claim_boundary_of_the_noise_model"]
        assert "not a validity condition" in boundary
        assert "UNKNOWN" in boundary

    def test_the_unknown_factor_is_named_and_not_assumed_to_be_one(self):
        report = noise.report()
        unknown = report["unknown_factor"]
        assert unknown["status"] == "UNMEASURED"
        assert "Top-K" in unknown["name"]
        #: Reported across assumed values, never as one number.
        assert len(unknown["pipeline_probability_at_assumed_values"]) >= 3
        at_one = unknown["pipeline_probability_at_assumed_values"]["1.0"]
        assert at_one == noise.advance_probability(2, 2)
        assert unknown["pipeline_probability_at_assumed_values"]["0.5"] \
            == pytest.approx(at_one / 2, abs=1e-4)

    def test_the_pipeline_probability_has_no_default_for_the_unknown(self):
        """A caller cannot quote one without stating what it assumed."""
        with pytest.raises(TypeError):
            noise.pipeline_detection_probability(2, 2)
        with pytest.raises(ValueError):
            noise.pipeline_detection_probability(2, 2, p_in_top_k=1.5)

    def test_the_sesoi_role_is_recorded_as_an_effect_size_not_a_threshold(self):
        role = noise.report()["sesoi_role"]
        assert "NOT a threshold" in role


class TestTheSpreadIsThreeObservations:
    """Why `0.7808` is planning analysis and not a power claim."""

    def test_the_interval_is_derived_from_the_three_deltas(self):
        low, high = noise.seed_sd_interval()
        assert low < noise.SEED_SD < high
        #: At n=3 the chi-square interval spans a factor of about twelve. A
        #: narrower claim would need more than three paired seeds.
        assert 10.0 < high / low < 14.0

    def test_two_observations_still_give_an_interval(self):
        low, high = noise.seed_sd_interval((0.01, -0.01))
        assert 0 < low < high

    def test_one_observation_is_refused(self):
        with pytest.raises(ValueError, match="two observations"):
            noise.seed_sd_interval((0.01,))

    def test_the_advance_probability_spans_the_interval(self):
        sens = noise.advance_probability_sensitivity(2, 2)
        assert sens["p_at_sd_high"] < sens["p_at_sd_point"] < sens["p_at_sd_low"]
        #: The width that makes the point estimate a planning figure: the
        #: interval is wider than the differences between candidate designs.
        spread = sens["p_at_sd_low"] - sens["p_at_sd_high"]
        between_designs = abs(noise.advance_probability(2, 2)
                              - noise.advance_probability(3, 2))
        assert spread > between_designs

    def test_the_report_states_how_many_observations_it_rests_on(self):
        report = noise.report()
        assert report["seed_sd_n"] == 3
        assert len(report["seed_sd_interval_95"]) == 2


@pytest.fixture(scope="module", autouse=True)
def _registered():
    """The shipped operators plus the promoted ATTENTION one.

    Module-scoped and a plain function: a class-scoped fixture defined as an
    instance method is deprecated, and the registration is process-global
    anyway, so the scope it belongs to is the module's. `conftest` removes
    whatever a module adds, which is what keeps this from leaking into an
    unrelated search's enumeration.
    """
    from experiments.phase_c2.search_space import register_c2_operators
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance,
    )
    register_c2_operators()
    activation_importance.register()
    yield


class TestTheSpace:
    def test_the_frozen_set_is_one_implementation_per_structural_kind(self):
        from aadistill.initialization.operators.base import get_implementation
        from experiments.phase_d1.search_space import FROZEN_IMPLEMENTATIONS

        for kind, impl_id in FROZEN_IMPLEMENTATIONS.items():
            assert get_implementation(impl_id).kind == kind
        assert len(set(FROZEN_IMPLEMENTATIONS.values())) == \
            len(FROZEN_IMPLEMENTATIONS)

    def test_every_frozen_implementation_consumes_calibration(self):
        """Which is what makes the scoring policy reach all four of them. A
        weight-only operator has no mechanism by which a position policy could
        change its output, so one in the frozen set would be a structural kind
        D1's variable cannot touch."""
        from aadistill.initialization.calibration.profiles import (
            consumes_calibration,
        )
        from aadistill.initialization.operators.base import get_implementation
        from experiments.phase_d1.search_space import FROZEN_IMPLEMENTATIONS

        for impl_id in FROZEN_IMPLEMENTATIONS.values():
            assert consumes_calibration(get_implementation(impl_id)), impl_id

    def test_every_exclusion_states_a_reason_and_is_really_excluded(self):
        """An exclusion is a DECISION, so it owes a reason and it has to bite.

        Membership in the live registry is deliberately NOT asserted:
        `attention.causal_kl_v1` is registered by the C3 launcher's own `spec()`
        rather than as a shipped default — `register.py` warns that an
        import-time registration silently adds a branch to an unrelated search —
        so in a clean process it is absent. An exclusion naming it is still
        meaningful, because the id is what a search would have to allow.
        """
        from experiments.phase_d1.search_space import EXCLUSIONS, d1_space

        allowed = set(d1_space().allowed_impls)
        for impl_id, why in EXCLUSIONS.items():
            assert impl_id not in allowed, (
                f"{impl_id} is listed as excluded and is in the space anyway")
            assert len(why) > 40, f"{impl_id}'s exclusion has no stated reason"

    def test_the_excluded_operators_that_ARE_registered_would_be_applicable(self):
        """So the exclusions are doing work rather than restating a precondition.

        An operator the geometry would reject anyway needs no exclusion, and
        listing one would make the decision look larger than it is.
        """
        from aadistill.initialization.operators.base import (
            applicable_implementations, get_implementation,
            registered_implementations,
        )
        from experiments.phase_d1.search_space import EXCLUSIONS, d1_space

        space = d1_space()
        registered = set(registered_implementations())
        candidates = sorted(set(EXCLUSIONS) & registered)
        assert candidates, "no excluded operator is registered to check"
        applicable = {impl.impl_id for impl, _ in applicable_implementations(
            space.adapter, space.teacher, space.target,
            allow_impls=candidates)}
        assert applicable == set(candidates), (
            f"{sorted(set(candidates) - applicable)} are excluded but the "
            "teacher geometry would have rejected them anyway, so the "
            "exclusion claims more than it decides")
        for impl_id in candidates:
            assert get_implementation(impl_id).kind

    def test_the_leaf_count_is_enumerated_not_written_down(self):
        """Four kinds in any order, each over two mixtures: 4! * 2^4 = 384. The
        assertion is on the arithmetic the registry produces, so a registry
        change moves it rather than leaving a stale literal in a plan."""
        from experiments.phase_d1.search_space import d1_space, size_report

        report = size_report()
        assert report["d1_frozen_set"]["total_leaves"] == 384
        assert math.factorial(4) * 2 ** 4 == 384
        #: And admitting the excluded operators really does change it, so the
        #: exclusions are load-bearing rather than decorative.
        assert (report["if_no_operator_were_excluded"]["total_leaves"]
                > report["d1_frozen_set"]["total_leaves"])
        assert len(d1_space().allowed_impls) == 4

    def test_the_search_is_priced_from_measured_telemetry(self):
        from experiments.phase_d1.search_space import search_cost

        cost = search_cost()
        assert cost["unmeasured_inputs"] == [], (
            "the D1 space contains an implementation the cost model has never "
            f"measured: {cost['unmeasured_inputs']}. A ceiling resting on a "
            "proxy is a margin, not a measurement.")
        assert cost["hard_ceiling_minutes"] > cost["expected_minutes"] > 0
        assert cost["gpu_usd"] > 0 and cost["container_disk_usd"] > 0

    def test_the_coverage_is_stated_and_the_warmup_protects_every_hypothesis(self):
        """A reviewer should not have to derive that a 384-leaf space is searched
        by visiting about a dozen leaves.

        The number that makes the pruning defensible is not the fraction: it is
        that the warm-up level keeps every root child, so each structural
        kind / mixture pair is measured once before anything is eliminated.
        """
        from experiments.phase_d1.search_space import coverage

        cov = coverage()
        assert cov["complete_leaves_visited"] > 0
        assert cov["reachable_leaves"] == 384
        assert 0 < cov["fraction_of_leaves_visited"] < 0.1
        levels = cov["levels"]
        #: Level 0 produces 4 kinds x 2 mixtures and level 1 expands all eight,
        #: which is what "no hypothesis is pruned unmeasured" means in numbers.
        assert levels[0]["generated"] == 8
        assert levels[1]["parents"] == 8
        #: And pruning does bite after that, or the beam width would be doing
        #: nothing and the cost would be the full enumeration's.
        assert levels[2]["parents"] < levels[1]["generated"]

    def test_the_coverage_and_the_price_walk_the_same_beam(self):
        """Two numbers from one walk. A coverage figure derived from a different
        beam than the price would describe a search nobody is paying for."""
        from experiments.phase_d1.search_space import coverage, search_cost

        assert coverage()["states_produced"] == \
            search_cost()["expected_trajectory_expansions"]

    def test_a_behavioural_session_needs_probes(self):
        from experiments.phase_d1.search_space import D1SpaceError, behavioural_cost

        with pytest.raises(D1SpaceError):
            behavioural_cost(n_probes=0)

    def test_the_hard_ceiling_bounds_the_expected_cost(self):
        from experiments.phase_d1.search_space import behavioural_cost

        cost = behavioural_cost(n_probes=12)
        assert cost["hard_ceiling_minutes"] > cost["expected_minutes"]
        assert cost["overrun_factor"] > 1.0


class TestTheCommittedRecords:
    def test_both_records_exist_and_are_json(self):
        for path in (DESIGN, CAPACITY):
            assert path.is_file(), f"{path} is missing"
            json.loads(path.read_text())

    def test_the_design_authorizes_nothing_and_says_so(self):
        doc = json.loads(DESIGN.read_text())
        assert doc["_authorizes"] == "nothing"
        assert "NOT AUTHORIZED" in doc["status"]
        assert doc["search_stage"]["cost"]["hard_ceiling_usd"] > 0

    def test_every_open_blocker_is_backed_by_the_field_it_derives_from(self):
        """Not a count, and not a membership list.

        **The count has been wrong in both directions and has now moved a third
        time** -- TWO, then THREE, then TWO again, now ONE after the production
        Top-K measurement resolved the envelope. So neither the number nor the
        membership is written here. Each POSSIBLE entry is checked against the
        field it derives from, in both directions: present when the field says
        open, absent when the field says closed.
        """
        doc = json.loads(DESIGN.read_text())
        assert "BLOCKER" in doc["budget"]
        open_ = set(doc["open_blockers"])
        budget = doc["budget"]

        #: funding -- open iff D1 is outside the funded list OR the provisional
        #: shortfall is positive.
        funding_open = (not budget["d1_is_in_the_funded_list"]
                        or budget["provisional_shortfall_usd"] > 0)
        assert ("funding authorization" in open_) is funding_open

        #: envelope -- open iff measured compatibility is not RESOLVED_FITS. The
        #: provisional basis does NOT enter this, deliberately.
        envelope_open = (budget["per_session_envelope_compatibility"]
                         != "RESOLVED_FITS")
        assert ("per-session envelope" in open_) is envelope_open

        #: and nothing else may appear.
        assert open_ <= {"evidence", "funding authorization",
                         "per-session envelope"}, open_

        #: evidence — CLOSED, and the design says so from the realized family
        evidence = doc["evidence"]
        assert evidence["status"] == "CLOSED"
        assert set(("d1_screening", "d1_confirmation")) <= set(
            evidence["roles_available"])
        assert len(evidence["family_content_id"]) == 64
        assert evidence["_authorizes"].startswith("nothing")
        #: funding -- whichever side of the funded list D1 is on, the predicate
        #: above checks the blocker against it. This used to pin `is False`, which
        #: was the state for every round until the 2026-10-05 amendment added
        #: `phase_d1` to `funds_formal_sessions_of` -- and then the test asserted
        #: that the maintainer decision had not been taken.
        assert isinstance(doc["budget"]["d1_is_in_the_funded_list"], bool)
        #: per-session envelope -- whichever state it is in, the test above
        #: checks the blocker against it rather than pinning one value.
        assert doc["budget"]["per_session_envelope_compatibility"] in (
            "UNRESOLVED", "RESOLVED_FITS", "RESOLVED_NEEDS_RAISE")
        #: AND THE TWO PREDICATES AGREE WITH THE WRITER'S OWN, which is the thing
        #: that must not drift: this test and `open_blockers()` must compute the
        #: same answer from the same fields.
        import write_d1_design as w

        assert set(w.open_blockers(doc["budget"],
                                   doc["contamination_protection"])) == open_

    def test_the_old_capacity_analysis_no_longer_drives_a_blocker(self):
        """It is kept as the reasoning that prompted the source decision, and it
        must not reopen a closed problem on every regeneration."""
        doc = json.loads(DESIGN.read_text())
        contamination = doc["contamination_protection"]
        assert "HISTORICAL REASONING" in contamination["_status"]
        assert "no longer a live blocker" in contamination["_status"]
        #: the figures survive -- it is history, not a deletion
        assert contamination["batteries_available"] == 0
        assert "evidence" not in doc["open_blockers"]

    def test_the_two_cost_derived_blocker_figures_are_named_provisional(self):
        """A planning figure must not be readable as finalized authorization
        pricing, and the name is where that is enforced.

        Both come from FULL-VOCABULARY telemetry for a protocol D1 will not run,
        so neither bounds the real cost. `shortfall_usd` and
        `fits_per_session_envelope` were exactly the names a later reader would
        have taken for settled figures.
        """
        budget = json.loads(DESIGN.read_text())["budget"]
        for field in ("provisional_shortfall_usd",
                      "provisional_per_session_excess_usd",
                      "provisional_basis_fits_per_session_envelope"):
            assert field in budget, f"{field} is missing"
        for retired in ("shortfall_usd", "per_session_envelope_excess_usd",
                        "fits_per_session_envelope"):
            assert retired not in budget, (
                f"{retired} is back; a cost-derived field whose direction is "
                "unknown must say so in its name")
        #: AND THE CLAIM BOUNDARY IS IN THE PROSE, not only the field names --
        #: asserted as the boundary rather than as one sentence. It used to demand
        #: the exact phrase "NOT the finalized amount", which held only while the
        #: shortfall was a planning figure of unknown direction; the measured
        #: DEPTH cell made the direction known and rewrote the sentence around a
        #: claim boundary that had not changed at all.
        note = budget["BLOCKER"]
        assert "PROVISIONAL" in note, (
            "the shortfall must say it is not a settled figure; three of four "
            "cost cells are still unmeasured")
        assert "does not claim what the cap should become" in note \
            or "does not claim the cap must move" in note, (
            "the design must not name the amount the project cap should become; "
            "that is a maintainer decision and this field is where it is "
            "disclaimed")

    def test_the_second_blocker_note_is_a_function_of_the_measured_status(self):
        """The ceiling note must say what the DERIVED status says, in all three
        directions.

        It used to be locked to the phrase `NOT ESTABLISHED`, which was true only
        while no production Top-K timing existed. A measurement then made that
        prose false, and the lock made the measurement look like the regression.
        So this exercises the writer's own branch function over every state
        instead: the note is derived, and no state may leave prose that
        contradicts it.
        """
        import write_d1_design as w

        budget = json.loads(DESIGN.read_text())["budget"]
        topk = budget["topk_production_basis"]
        envelope = budget["per_session_envelope_usd"]
        status = budget["per_session_envelope_compatibility"]
        notes = {s: w._envelope_blocker_note(budget_section_compatibility=s,
                                            envelope_usd=envelope, topk=topk)
                 for s in (("UNRESOLVED",) if topk is None else
                           ("UNRESOLVED", "RESOLVED_FITS",
                            "RESOLVED_NEEDS_RAISE"))}
        #: The committed note is the one its own status produces.
        assert budget["_SECOND_BLOCKER_THE_PER_SESSION_CEILING"] == notes[status]
        if topk is None:
            #: No valid production measurement exists, so UNRESOLVED is the ONLY
            #: reachable state and there is no price for a note to name. The
            #: resolved branches are exercised whenever a basis is present, which
            #: is the state this file's `TestThePricingGate...` sibling in the
            #: D-series suite drives through all three with synthetic records.
            assert status == "UNRESOLVED", (
                f"the design says {status} with no valid production basis, so a "
                "sentence is resolving the envelope that no measurement can")
            assert "SEPARATE constraint from the project-level cap" in \
                notes["UNRESOLVED"]
            return
        #: Each note names its own state and no other, and only a resolved one
        #: may name a price -- keyed on the state tokens the writer must use,
        #: not on a phrase it happens to use today.
        priced = f"${topk['search_session']['hard_ceiling_usd']:.4f}"
        assert "UNRESOLVED" in notes["UNRESOLVED"]
        assert priced not in notes["UNRESOLVED"], (
            "an unresolved ceiling must not quote a price as though it were "
            "bound")
        for resolved in ("RESOLVED_FITS", "RESOLVED_NEEDS_RAISE"):
            assert "UNRESOLVED" not in notes[resolved], (
                f"{resolved} prose still calls the ceiling unresolved, which "
                "would let a sentence outvote a measurement")
            assert priced in notes[resolved], (
                f"{resolved} must name the price it was resolved at")
        assert "no longer a blocker" in notes["RESOLVED_FITS"]
        assert "would have to move" in notes["RESOLVED_NEEDS_RAISE"]
        #: And every state keeps the distinction the field exists to make: this
        #: ceiling binds one session, the project cap is cumulative.
        for note in notes.values():
            assert "SEPARATE constraint from the project-level cap" in note

    def test_the_budget_position_is_derived_not_restated(self):
        """The writer calls `derive_budget.derive()`; a hand-copied balance
        expires the next time anything spends."""
        doc = json.loads(DESIGN.read_text())["budget"]["position"]
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_derive_budget", REPO / "scripts/consolidate/derive_budget.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        live = module.derive()
        assert doc["project_remaining_usd"] == \
            pytest.approx(float(live["project"]["remaining_usd"]))
        assert doc["formal_remaining_usd"] == \
            pytest.approx(float(live["formal"]["remaining_usd"]))

    def test_the_capacity_record_states_zero_and_names_the_binding_stratum(self):
        doc = json.loads(CAPACITY.read_text())
        binding = doc["binding_stratum"]
        assert doc["batteries_remaining"] == 0
        assert doc["strata"][binding]["batteries_remaining"] == 0
        assert doc["strata"][binding]["short_by"] > 0
        #: Every stratum's remaining count is at least the binding one's, which
        #: is what "binding" means.
        assert all(row["batteries_remaining"] >= 0
                   for row in doc["strata"].values())
        assert binding == min(doc["strata"],
                              key=lambda s: doc["strata"][s]["batteries_remaining"])

    def test_the_design_and_the_capacity_record_agree(self):
        """Two documents, one fact. The design restates the count, so it has to
        match the record that measured it."""
        design = json.loads(DESIGN.read_text())["contamination_protection"]
        capacity = json.loads(CAPACITY.read_text())
        assert design["batteries_available"] == capacity["batteries_remaining"]
        assert design["binding_stratum"] == capacity["binding_stratum"]

    def test_the_design_names_the_scoring_policies_it_compares(self):
        from aadistill.initialization.scoring.positions import (
            ALL_POSITIONS_V1, SUPERVISED_TARGET_V1,
        )

        policy = json.loads(DESIGN.read_text())["scoring_policy"]
        assert policy["treatment"]["policy_hash"] == \
            SUPERVISED_TARGET_V1.policy_hash
        assert policy["control"]["policy_hash"] == ALL_POSITIONS_V1.policy_hash

    def test_the_design_regenerates_byte_identically(self):
        """A committed verdict expires silently; this is how a reader knows the
        record still describes the tree."""
        import write_d1_design

        assert (json.dumps(write_d1_design.build(), indent=1, sort_keys=True)
                + "\n") == DESIGN.read_text()


class TestTheOwedGpuValidationStatusIsDerived:
    """The status of the owed GPU validation is a function of the qualification's
    closeout, not a sentence.

    A typed status is wrong exactly when the work it describes completes, which
    is the moment a reader is most likely to trust it. `_qualification_state`
    reads the closeout, so writing or removing one moves the design with no edit
    to the writer -- and these assert BOTH directions, because a derivation that
    has only ever been exercised on its absent branch is not known to derive.
    """

    def test_absent_closeout_reads_owed(self, monkeypatch, tmp_path):
        import write_d1_design as w

        monkeypatch.setattr(w, "REPO", tmp_path)
        state = w._qualification_state()
        assert state["status"].startswith("OWED, NOT RUN")
        assert "no closeout at" in state["_status_owner"]

    def test_a_closeout_flips_it_and_still_authorizes_nothing(
            self, monkeypatch, tmp_path):
        import write_d1_design as w

        closeout = tmp_path / w.QUALIFICATION_CLOSEOUT
        closeout.parent.mkdir(parents=True)
        #: The REAL answer shape, not a stand-in. The first version put a
        #: string where the closeout has an object, and when the writer started
        #: reading named fields out of it the test failed on its own fixture
        #: rather than on the code.
        closeout.write_text(json.dumps({
            "verdict": "PASSED", "gpu": "NVIDIA L40S", "cost_usd": 1.23,
            "price_per_hour_usd": 1.09, "paid_subruns": 2,
            "answers": {
                "incumbent_reconstruction": {
                    "matched_the_frozen_incumbent": True,
                    "artifact_digest": "a" * 64},
                "discrete_decisions": {
                    "n_moved": 3,
                    "steps_that_moved": [
                        {"impl_id": "ffn.activation_importance_v0",
                         "attribution": "THE CALIBRATION BATCH SIZE moved it: "
                                        "holding the policy reproduced it"}]},
            },
        }))
        monkeypatch.setattr(w, "REPO", tmp_path)
        state = w._qualification_state()

        #: TWO ROUNDS now, and the status names each: a bare "RUN -- PASSED" was
        #: true of the full-vocab qualification and would read as true of the
        #: Top-K adoption too.
        assert "full-vocab qualification RUN -- PASSED" in state["status"]
        assert "Top-K adoption" in state["status"]
        assert state["_status_owner"] == w.QUALIFICATION_CLOSEOUT
        assert state["ran"]["cost_usd"] == 1.23
        #: The design carries the closeout's PATH and CONTENT HASH plus the
        #: conclusions it consumes -- never a copy of `answers`, which made this
        #: one key 1.48 MB of a 3.9 MB design.
        assert state["ran"]["closeout"] == w.QUALIFICATION_CLOSEOUT
        assert len(state["ran"]["closeout_sha256"]) == 64
        assert "answers" not in state["ran"]
        assert state["ran"]["reconstructed_the_frozen_incumbent"] is True
        assert state["ran"]["n_operator_selections_moved"] == 3
        assert state["ran"]["what_moved_them"] == [
            "THE CALIBRATION BATCH SIZE moved it"]
        #: A passed qualification is an ENGINEERING result. It must not read as
        #: though it had funded D1 or closed one of D1's blockers.
        assert state["ran"]["_authorizes"].startswith("nothing")

    def test_the_retired_phrasings_cannot_come_back(self):
        """Pin the wording that went stale, not the wording that is current."""
        source = (REPO / "scripts/autoinit/write_d1_design.py").read_text()
        for retired in ("the validation is owed at authorization time rather "
                        "than now",
                        "It is not requested here"):
            assert retired not in source, retired


class TestMeasuredEnvelopeCompatibilityIsAuthoritative:
    """A superseded planning figure must not veto a measurement.

    `open_blockers` used to require BOTH `RESOLVED_FITS` and that the provisional
    FULL-VOCABULARY basis fit. So a future Top-K measurement proving D1 fits
    inside the envelope would have stayed blocked by an estimate it supersedes --
    an estimate outvoting a measurement.
    """

    FUNDED = {"d1_is_in_the_funded_list": True,
              "provisional_shortfall_usd": 0.0}

    def _blockers(self, compatibility, provisional_fits):
        import write_d1_design

        return write_d1_design.open_blockers(
            {**self.FUNDED,
             "per_session_envelope_compatibility": compatibility,
             "provisional_basis_fits_per_session_envelope": provisional_fits},
            {})

    def test_resolved_fits_closes_it_even_when_the_provisional_basis_does_not(self):
        """THE REGRESSION."""
        assert self._blockers("RESOLVED_FITS", False) == ()

    def test_resolved_needs_raise_keeps_it_open(self):
        assert "per-session envelope" in self._blockers("RESOLVED_NEEDS_RAISE",
                                                       True)

    def test_unresolved_keeps_it_open_whatever_the_provisional_basis_says(self):
        for provisional in (True, False):
            assert "per-session envelope" in self._blockers("UNRESOLVED",
                                                            provisional)

    def test_an_unknown_compatibility_value_does_not_silently_close_it(self):
        """A typo must fail closed."""
        for value in ("RESOLVED", "FITS", "", None, "resolved_fits"):
            assert "per-session envelope" in self._blockers(value, True), value


class TestNoLiveFieldClaimsTheTopKCostIsOwed:
    """The sweep an independent review had to run by hand, twice.

    The writer kept emitting live fields that said the production Top-K cost was
    unmeasured, under corrective review or being re-measured — once in the SAME
    `budget` object as a `price_status` reading "MEASURED PRODUCTION TOP-K BASIS".
    One object, two answers, and the detailed field was the stale one.

    Keyed on the production-basis predicate, so it holds in both directions: while
    no basis exists these phrases are the honest state and are REQUIRED somewhere;
    once one exists no live field may carry them.
    """

    #: Phrases that assert the Top-K cost is still owed. A negation containing one
    #: trips this too, deliberately: prose that happens to contain the stale claim
    #: cannot be distinguished from the claim by any sweep, including a reviewer's.
    OWED_PHRASES = (
        "STILL NOT MEASURED",
        "being re-measured",
        "UNDER CORRECTIVE REVIEW",
        "stays UNRESOLVED",
        "representative timing exists",
        "representative expansion timing",
        "the production Top-K cost is not",
        "owed GPU qualification",
    )

    @staticmethod
    def _strings(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield f"{path}.{k}", str(k)
                yield from TestNoLiveFieldClaimsTheTopKCostIsOwed._strings(
                    v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from TestNoLiveFieldClaimsTheTopKCostIsOwed._strings(
                    v, f"{path}[{i}]")
        elif isinstance(obj, str):
            yield path, obj

    @staticmethod
    def _basis():
        import write_d1_design as w

        return w._topk_production_basis()

    def test_the_live_design_carries_none_of_them_once_the_cost_is_measured(self):
        doc = json.loads(DESIGN.read_text())
        if self._basis() is None:
            pytest.skip("no production basis; the next test owns that state")
        hits = [(path, phrase)
                for path, text in self._strings(doc)
                for phrase in self.OWED_PHRASES if phrase in text]
        assert not hits, (
            "the production Top-K cost is MEASURED and these live fields say "
            f"otherwise: {hits}. A reader taking any one of them concludes the "
            "evidence is in flight.")

    def test_they_would_be_required_if_no_basis_existed(self, monkeypatch):
        """The other direction, so this is not a test that only ever passes.

        With the basis removed, the writer must go back to saying the cost is
        unmeasured -- in the detailed status, in the envelope note and in what a
        GPU still owes. A writer that said "measured" either way would pass the
        test above for the wrong reason.
        """
        import write_d1_design as w

        monkeypatch.setattr(w, "TOPK_PRODUCTION", "logs/does/not/exist.json")
        assert w._topk_production_basis() is None
        rebuilt = w.build()
        blob = json.dumps(rebuilt)
        assert any(p in blob for p in self.OWED_PHRASES), (
            "with no production basis the design claims nothing is owed; the "
            "narratives are not derived from the basis at all")
        assert rebuilt["budget"]["per_session_envelope_compatibility"] == \
            "UNRESOLVED"
        assert "per-session envelope" in rebuilt["open_blockers"]

    def test_the_four_fields_that_must_agree_do(self):
        """`price_status`, `_price_status`, the envelope state and the blockers."""
        doc = json.loads(DESIGN.read_text())
        budget = doc["budget"]
        measured = self._basis() is not None
        assert ("MEASURED PRODUCTION TOP-K BASIS" in budget["price_status"]) \
            is measured
        assert ("the dominant cell is MEASURED" in budget["_price_status"]) \
            is measured
        assert (budget["per_session_envelope_compatibility"] != "UNRESOLVED") \
            is measured
        assert ("per-session envelope" not in doc["open_blockers"]) is measured


class TestOneCurrentSearchCost:
    """`search_stage.cost` and `budget.chain.sessions.search` are one session.

    They disagreed in the same document: `search_stage.cost.hard_ceiling_usd` was
    $31.1577, the superseded full-vocabulary planning figure, while the budget
    chain carried the measured $21.4897. Two generic current-cost fields for one
    session, and the one a reader reaches first held the stale number.
    """

    @staticmethod
    def _parts():
        import write_d1_design as w

        doc = json.loads(DESIGN.read_text())
        return doc, w._topk_production_basis()

    def test_both_fields_price_the_same_session(self):
        doc, _ = self._parts()
        stage = doc["search_stage"]["cost"]
        chain = doc["budget"]["chain"]["sessions"]["search"]
        shared = sorted(set(stage) & set(chain) - {"_cost_basis"})
        assert len(shared) >= 8, f"only {shared} are comparable; expected the "\
            "whole priced session"
        differing = {k: (stage[k], chain[k]) for k in shared
                     if stage[k] != chain[k]}
        assert not differing, (
            f"the same session is priced two ways: {differing}. One of them is "
            "what a reader takes as the current cost.")

    def test_with_a_measured_basis_both_are_the_measured_figure(self):
        doc, basis = self._parts()
        if basis is None:
            pytest.skip("no production basis; the frozen figure is correct then")
        measured_minutes = basis["root_max_minutes"]
        stage = doc["search_stage"]["cost"]
        #: The measured DEPTH cell has to be IN it: equality with the chain is not
        #: enough on its own, since two stale figures would also be equal.
        assert stage["hard_ceiling_minutes"] < \
            doc["search_stage"]["superseded_full_vocab_planning_cost"][
                "hard_ceiling_minutes"], (
            "the live cost is not below the full-vocab planning basis, so the "
            "measured cell did not reach it")
        assert str(round(measured_minutes, 2))[:4] in stage["_cost_basis"] \
            or "MEASURED" in stage["_cost_basis"], stage["_cost_basis"][:200]

    def test_the_superseded_figure_is_named_historical_not_generic(self):
        doc, basis = self._parts()
        if basis is None:
            pytest.skip("nothing is superseded while nothing is measured")
        stage = doc["search_stage"]
        old = stage["superseded_full_vocab_planning_cost"]
        assert "HISTORICAL" in json.dumps(old), (
            "a retained superseded price must say so in its own content")
        assert old["hard_ceiling_usd"] != stage["cost"]["hard_ceiling_usd"], (
            "the superseded entry equals the live one, so it proves nothing and "
            "should be removed rather than kept as provenance")
        #: And no OTHER generic cost key may appear beside `cost`.
        generic = [k for k in stage
                   if k.endswith("_cost") and not k.startswith("superseded")]
        assert not generic, f"{generic} are second generic cost fields"


class TestThePhaseFundingAmendment:
    """The 2026-10-05 maintainer decision, asserted as a decision not a balance.

    A test on a remaining balance expires the next time anything spends. These
    assert the GRANTED figures and the funded list, which only a maintainer moves.
    """

    AUTH = Path("configs/experiments/phase_c1/authorization.json")

    @staticmethod
    def _terms():
        doc = json.loads((REPO / "configs/experiments/phase_c1"
                                 "/authorization.json").read_text())
        return doc["execution_package"], doc["accepted_pricing"]

    def test_the_five_figures_are_what_was_decided(self):
        """TWO decisions now, and the second one moved a figure the first had
        deliberately left alone -- so this asserts the current grant and the
        comment records which decision set which number.

        2026-10-07, +$25.0000 formal, after the formal search was stopped
        mid-beam by an exhausted RunPod account balance and its $8.1716 bought
        no endpoint. That amendment explicitly left the per-session and
        engineering limits untouched.

        2026-10-08, the GPU engineering allowance only, raised by exactly the
        shortfall between it and the rematerialization campaign's own approved
        remaining dollars. NOT a new grant: the hard limit on that work is the
        campaign's $10.0000 ceiling, and this removed an older accounting bound
        that would otherwise have refused the campaign's approved spend. The
        derivation is in the authorization's own amendment block.
        """
        ep, ap = self._terms()
        #: 2026-10-07.
        assert ep["formal_allowance_usd"] == 156.6523
        assert ap["cumulative_cap_usd"] == 490.0
        #: 2026-10-08: +$5.0741, the derived shortfall and nothing more.
        assert ep["gpu_engineering_allowance_usd"] == 25.0741
        #: Derived, not independently granted -- see the test below.
        assert ep["package_total_usd"] == 181.7264
        #: UNCHANGED by BOTH amendments.
        assert ep["per_attempt_hard_ceiling_usd"] == 30.0

    def test_the_engineering_raise_is_exactly_the_derived_shortfall(self):
        """A raise bigger than the shortfall would be a new grant wearing a
        bookkeeping amendment's name. The authorization carries its own
        derivation; this checks the arithmetic closes."""
        ep, _ = self._terms()
        a = ep["_amendment_2026_10_08_engineering_allowance"]
        d = a["derivation"]
        assert d["required_engineering_remaining_usd"] == round(
            d["campaign_remaining_usd"] + d["campaign_teardown_reserve_usd"], 4)
        assert d["shortfall_usd"] == round(
            d["required_engineering_remaining_usd"]
            - d["engineering_remaining_before_usd"], 4)
        assert ep["gpu_engineering_allowance_usd"] == round(
            a["gpu_engineering_allowance_usd"]["from"] + d["shortfall_usd"], 4)
        #: And the campaign ceiling it serves is NOT raised by it.
        assert d["campaign_ceiling_usd"] == 10.0

    def test_the_project_cap_was_verified_rather_than_raised(self):
        """The instruction was to derive whether existing headroom covers the
        campaign, not to raise the cap by default."""
        ep, ap = self._terms()
        cap = ep["_amendment_2026_10_08_engineering_allowance"]["project_cap"]
        assert cap["cap_usd"] == ap["cumulative_cap_usd"] == 490.0
        assert cap["remaining_usd"] >= cap["cap_usd"] - cap[
            "cumulative_spend_usd"] - 1e-9
        assert cap["remaining_usd"] > ep[
            "_amendment_2026_10_08_engineering_allowance"][
            "derivation"]["campaign_remaining_usd"], (
            "the amendment claims existing headroom covers the campaign; it "
            "does not")

    def test_the_package_total_is_still_the_sum_of_its_parts(self):
        """It is not an independent number; a drift here hides a real raise."""
        ep, _ = self._terms()
        assert ep["package_total_usd"] == pytest.approx(
            ep["formal_allowance_usd"] + ep["gpu_engineering_allowance_usd"])

    def test_d1_is_funded_and_says_by_what_authority(self):
        ep, _ = self._terms()
        funded = ep["funds_formal_sessions_of"]
        assert "phase_d1" in funded["experiment_ids"]
        assert "2026-10-05" in funded["phase_d1"]
        #: Without this line D1's formal spend reaches the project cumulative while
        #: the formal book reports it as never having happened.
        assert "formal allowance" in funded["phase_d1"]

    def test_the_amendment_states_what_it_does_not_authorize(self):
        ep, _ = self._terms()
        note = ep["_amendment_2026_10_05"]
        for required in ("PROSPECTIVE AND CURRENT-ONLY", "scientific repetition",
                         "D2 and D3 are NOT authorized", "K sweep",
                         "UNCHANGED", "no historical grant"):
            assert required in note, f"the amendment does not say {required!r}"

    def test_the_earlier_amendments_keep_their_figures(self):
        """Not retrospective: a reader checks what each session ran under."""
        ep, _ = self._terms()
        assert "76.6523" in ep["_amendment_2026_10_05"], (
            "the amendment must name the figure it raised FROM")
        for older, figure in (("_amendment_2026_10_03", "96.6523"),
                              ("_amendment_2026_10_02", None),
                              ("_amendment_2026_10_01", None),
                              ("_engineering_allowance_amendment_2026_09_27",
                               "55.4425")):
            assert older in ep, f"{older} was removed"
            if figure:
                assert figure in ep[older], (
                    f"{older} no longer names {figure}; a historical amendment "
                    "was rewritten")

    def test_the_cap_amendment_history_is_append_only(self):
        _, ap = self._terms()
        history = ap["_cap_amendment"]
        for cap in ("283.76", "320.00", "370.00", "400.00", "465.00"):
            assert cap in history, f"the cap history dropped {cap}"


class TestTheScientificDesignHashIsNotAFunctionOfMoney:
    """`design_hash` must move on protocol and never on the ledger.

    It was a sha over every non-underscore key, which swept in `budget`. So
    booking a session's spend moved `budget.position` and
    `provisional_shortfall_usd`, moved the hash, invalidated the committed
    record, reddened two tests in D1's own pod gate and blocked every launch
    until the record was regenerated and the authorization re-issued against a
    new hash. That cycle was paid THREE TIMES in one round before the preimage
    was narrowed. A scientific design does not change because money was spent.
    """

    @staticmethod
    def _hash(doc):
        from aadistill.infrastructure.manifest import sha256_json
        from autoinit.write_d1_design import scientific_preimage

        return sha256_json(scientific_preimage(doc))

    @pytest.fixture(scope="class")
    def doc(self):
        from autoinit.write_d1_design import build

        return build()

    def test_the_committed_hash_is_the_scientific_preimage_hash(self, doc):
        assert doc["design_hash"] == self._hash(doc)

    @pytest.mark.parametrize("label,mutate", [
        ("remaining formal balance",
         lambda d: d["budget"]["position"].update(formal_remaining_usd=1.0)),
        ("provisional shortfall",
         lambda d: d["budget"].update(provisional_shortfall_usd=-999.0)),
        ("chain hard ceiling",
         lambda d: d["budget"]["chain"].update(hard_ceiling_usd=123.0)),
        ("live GPU price",
         lambda d: d["search_stage"]["cost"].update(price_per_hour=9.99)),
        ("search session ceiling",
         lambda d: d["search_stage"]["cost"].update(hard_ceiling_usd=99.0)),
        ("the priced design grid",
         lambda d: d["behavioural_design"]["priced_grid"].clear()),
        ("run status", lambda d: d.update(status="anything")),
        ("open blockers", lambda d: d.update(open_blockers=["x"])),
        ("owed GPU validation state",
         lambda d: d.update(gpu_validation_owed={"status": "x"})),
    ])
    def test_money_and_run_state_do_not_move_it(self, doc, label, mutate):
        import copy

        moved = copy.deepcopy(doc)
        mutate(moved)
        assert self._hash(moved) == self._hash(doc), (
            f"{label} moved the scientific design hash; money and run state "
            "are bound by the run's authorization, not by the science")

    @pytest.mark.parametrize("label,mutate", [
        ("Top-K", lambda d: d["behavioural_design"].update(top_k=5)),
        ("the finalist-retention rule",
         lambda d: d["behavioural_design"].update(
             finalist_retention="quality_with_lineage_diversity")),
        ("screening seeds",
         lambda d: d["behavioural_design"].update(screening_seeds=3)),
        ("confirmation seeds",
         lambda d: d["behavioural_design"].update(confirmation_seeds=5)),
        ("the decision rule",
         lambda d: d["behavioural_design"].update(decision_rule="other")),
        ("the recovery recipe", lambda d: d.update(recovery_recipe={"r": 1})),
        ("the scoring protocol", lambda d: d.update(scoring_policy={"s": 1})),
        ("the hypothesis", lambda d: d.update(hypothesis="other")),
        ("beam width", lambda d: d["search_stage"]["beam"].update(width=8)),
        ("the ranking policy",
         lambda d: d["search_stage"]["ranking_policy"].update(id="other")),
        ("the contamination protocol",
         lambda d: d.update(contamination_protection={"c": 1})),
    ])
    def test_protocol_changes_do_move_it(self, doc, label, mutate):
        import copy

        moved = copy.deepcopy(doc)
        mutate(moved)
        assert self._hash(moved) != self._hash(doc), (
            f"{label} did NOT move the scientific design hash; a protocol "
            "change that leaves the identity alone is a change nothing records")

    def test_a_missing_scientific_key_is_refused_not_skipped(self, doc):
        """A preimage that drops an absent field hashes a smaller document and
        still reads as a valid identity."""
        import copy

        from autoinit.write_d1_design import scientific_preimage

        broken = copy.deepcopy(doc)
        del broken["recovery_recipe"]
        with pytest.raises(KeyError, match="recovery_recipe"):
            scientific_preimage(broken)
        broken = copy.deepcopy(doc)
        del broken["behavioural_design"]["top_k"]
        with pytest.raises(KeyError, match="top_k"):
            scientific_preimage(broken)

    def test_the_preimage_is_a_positive_list_not_an_exclusion(self):
        """A blacklist would silently admit the next money-bearing field added
        anywhere in the document.

        Asserted on the AST rather than by grepping for the old expression's
        absence: the comment that explains WHY the exclusion was wrong quotes
        it, and a text probe matched its own documentation.
        """
        import ast

        from autoinit import write_d1_design as W

        tree = ast.parse(Path(W.__file__).read_text())
        assigned = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "design_hash"):
                    assigned.append(node.value)
        assert len(assigned) == 1, (
            f"expected exactly one assignment to doc['design_hash'], found "
            f"{len(assigned)}")
        call = assigned[0]
        assert isinstance(call, ast.Call)
        assert getattr(call.func, "id", None) == "sha256_json"
        inner = call.args[0]
        assert isinstance(inner, ast.Call), (
            "design_hash is computed from an expression rather than from the "
            "declared preimage")
        assert getattr(inner.func, "id", None) == "scientific_preimage"
        #: And the positive lists exclude every money-bearing block.
        assert "budget" not in W.SCIENTIFIC_TOP_LEVEL
        assert "cost" not in W.SCIENTIFIC_SEARCH_STAGE
        assert "priced_grid" not in W.SCIENTIFIC_BEHAVIOURAL

    def test_the_finalist_retention_rule_is_quality_only_and_hash_bound(self, doc):
        from aadistill.initialization.planning.ranking import PARETO_V1

        from autoinit.write_d1_design import SCIENTIFIC_BEHAVIOURAL

        assert doc["behavioural_design"]["finalist_retention"] == \
            PARETO_V1.RETENTION_QUALITY_ONLY
        assert "finalist_retention" in SCIENTIFIC_BEHAVIOURAL


class TestTheLiveEvidenceOwnerCannotBeMisreadAsExhausted:
    """A reader followed `inputs.evidence_capacity`, found
    `batteries_remaining: 0`, and reported D1 as blocked on a battery it
    already has.

    Every fact needed to avoid that was in the same document: `evidence.status`
    is CLOSED, `open_blockers` is empty, and the prose directly above `inputs`
    says the realized family superseded the capacity record. None of that
    helped, because a key named like a live input sent the reader to the
    historical record first.

    So the live owners are named as such and the superseded one is
    underscore-prefixed with its reason. The capacity record is NOT deleted --
    it is the provenance showing why the D-series family was necessary.
    """

    @staticmethod
    def _doc():
        return json.loads((REPO / "logs/stages/stage-1/phase_d1/plans/"
                                  "d1_design.json").read_text())

    def test_the_live_evidence_owners_are_named_in_inputs(self):
        inputs = self._doc()["inputs"]
        assert "behavioural_evidence_family" in inputs
        assert "behavioural_evidence_realized" in inputs
        for key in ("behavioural_evidence_family",
                    "behavioural_evidence_realized"):
            assert (REPO / inputs[key]).is_file(), inputs[key]

    def test_the_superseded_record_is_marked_superseded(self):
        inputs = self._doc()["inputs"]
        assert "evidence_capacity" not in inputs, (
            "the historical capacity record is listed as a peer of the live "
            "inputs again; that is what sent a reader to a closed blocker")
        assert "_superseded_capacity_record" in inputs
        assert "NOT the live evidence owner" in inputs[
            "_why_that_key_is_underscored"]

    def test_it_is_kept_rather_than_deleted(self):
        """It explains why the family exists. Deleting it would lose the
        reason and leave the family looking arbitrary."""
        inputs = self._doc()["inputs"]
        assert (REPO / inputs["_superseded_capacity_record"]).is_file()

    def test_the_evidence_block_is_the_owner_and_says_closed(self):
        ev = self._doc()["evidence"]
        assert ev["status"] == "CLOSED"
        assert ev["owner"].endswith("autoinit_d_series_family_manifest.json")
        assert ev["roles_required"] == ["d1_screening", "d1_confirmation"]

    def test_the_family_the_design_names_is_built_and_verified(self):
        """Not a claim in the design -- read from the family record itself."""
        ev = self._doc()["evidence"]
        fam = json.loads((REPO / "logs/shared/analyses/"
                                 "autoinit_d_series_battery_family.json").read_text())
        assert fam["status"] == "BUILT / VERIFIED"
        assert fam["capacity_source_blocker"] == "CLOSED"
        assert fam["family_content_id"] == ev["family_content_id"]
        assert fam["allocation_rule_id"] == ev["allocation_rule_id"]

    def test_both_d1_roles_are_realized_on_disk_and_match_the_manifest(self):
        """The one check that would have answered the question directly."""
        import hashlib

        man = json.loads((REPO / "logs/shared/analyses/"
                                 "autoinit_d_series_family_manifest.json").read_text())
        base = REPO / "artifacts/stage3/d_series_behavioural_v1"
        checked = 0
        for rel, rec in man["output_files"].items():
            if not rel.startswith(("d1_screening/", "d1_confirmation/")):
                continue
            f = base / rel
            assert f.is_file(), f"{rel} is not realized"
            assert hashlib.sha256(f.read_bytes()).hexdigest() == rec["sha256"], rel
            checked += 1
        assert checked == 14, f"expected 14 D1 role files, checked {checked}"

    def test_each_d1_role_preserves_the_frozen_stratum_balance(self):
        man = json.loads((REPO / "logs/shared/analyses/"
                                 "autoinit_d_series_family_manifest.json").read_text())
        frozen = {"code": 100, "gsm8k": 150, "knowledge": 150,
                  "math_verified": 150, "multihop": 150, "rag": 150,
                  "tool": 100}
        for role in ("d1_screening", "d1_confirmation"):
            r = man["roles"][role]
            assert r["per_stratum"] == frozen, role
            assert r["n_prompts"] == 950 and r["n_scorable"] == 850, role

    def test_the_two_d1_roles_are_disjoint(self):
        """The validity condition for the two-rung design: an advancing
        candidate is selected on prompts the confirmation does not reuse."""
        man = json.loads((REPO / "logs/shared/analyses/"
                                 "autoinit_d_series_family_manifest.json").read_text())
        a = man["roles"]["d1_screening"]["item_ids_sha256"]
        b = man["roles"]["d1_confirmation"]["item_ids_sha256"]
        assert a != b
        base = REPO / "artifacts/stage3/d_series_behavioural_v1"
        for stratum in ("math_verified", "gsm8k", "code"):
            def ids(role):
                f = base / role / f"{stratum}.jsonl"
                return {json.loads(line)["id"] for line in
                        f.read_text().splitlines() if line.strip()}
            overlap = ids("d1_screening") & ids("d1_confirmation")
            assert not overlap, f"{stratum}: {len(overlap)} shared prompts"
