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
        assert (design["top_k"], design["screening_seeds"],
                design["confirmation_seeds"]) == \
            (D1_TOP_K, D1_SCREENING_SEEDS, D1_CONFIRMATION_SEEDS) == (2, 2, 3)

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

    def test_the_open_blockers_are_exactly_the_two_that_remain(self):
        """Not as caveats. Either alone prevents execution.

        **The count has now been wrong in both directions**, which is why neither
        the number nor the membership is written here as a constant: it read TWO
        after the per-session ceiling became the third, then THREE after the
        realized D-series family closed the evidence one. Each entry is checked
        against the field it is derived from, and the evidence entry's ABSENCE is
        checked against the realized family.
        """
        doc = json.loads(DESIGN.read_text())
        assert "BLOCKER" in doc["budget"]
        assert set(doc["open_blockers"]) == {
            "funding authorization", "per-session envelope"}

        #: evidence — CLOSED, and the design says so from the realized family
        evidence = doc["evidence"]
        assert evidence["status"] == "CLOSED"
        assert set(("d1_screening", "d1_confirmation")) <= set(
            evidence["roles_available"])
        assert len(evidence["family_content_id"]) == 64
        assert evidence["_authorizes"].startswith("nothing")
        #: funding — definite for a reason needing no cost estimate.
        assert doc["budget"]["d1_is_in_the_funded_list"] is False
        #: per-session envelope — open because UNRESOLVED, not proven to fail.
        assert doc["budget"]["per_session_envelope_compatibility"] == "UNRESOLVED"

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

        Both come from UNBATCHED telemetry whose direction relative to batched
        D1 is unknown, so neither bounds the real cost. `shortfall_usd` and
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
        #: and the claim boundary is in the prose, not only the field names
        assert "NOT the finalized amount" in budget["BLOCKER"]
        assert "NOT ESTABLISHED" in budget[
            "_SECOND_BLOCKER_THE_PER_SESSION_CEILING"]

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

        assert state["status"] == "RUN -- PASSED"
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
