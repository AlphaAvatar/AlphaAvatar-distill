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

REPO = Path(__file__).resolve().parents[2]
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
    def test_more_screening_seeds_always_reduce_the_bias(self):
        for k in (2, 3, 5):
            biases = [noise.expected_max_bias(k, m) for m in (1, 2, 3)]
            assert biases == sorted(biases, reverse=True)

    def test_more_candidates_always_increase_the_bias(self):
        for m in (1, 2, 3):
            biases = [noise.expected_max_bias(k, m) for k in (2, 3, 4, 5)]
            assert biases == sorted(biases)

    def test_no_screening_is_an_honest_zero(self):
        assert noise.expected_max_bias(5, 0) == 0.0
        assert noise.advance_probability(1, 2) == 1.0

    def test_discrimination_falls_as_the_field_widens(self):
        """The half of the trade a bias figure alone would hide: a design can
        have a low bias and still be a coin flip."""
        probabilities = [noise.advance_probability(k, 1) for k in (2, 3, 4, 5)]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_the_c2_design_exceeds_the_sesoi_and_the_d1_design_does_not(self):
        """The comparison the D1 design is built on, asserted rather than
        narrated. Five candidates on one seed against two on two."""
        c2 = noise.expected_max_bias(5, 1)
        d1 = noise.expected_max_bias(2, 2)
        assert c2 > noise.SESOI > d1
        assert noise.advance_probability(2, 2) > noise.advance_probability(5, 1)
        #: And the probe counts are equal, which is what makes it a strict
        #: improvement rather than a trade: (5+1)*1 + 6 == (2+1)*2 + 6.
        assert (5 + 1) * 1 + 6 == (2 + 1) * 2 + 6

    def test_the_report_is_deterministic(self):
        """It uses a seeded generator, so the design table is a property of the
        arithmetic rather than of when it ran."""
        assert noise.report()["rows"] == noise.report()["rows"]


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

    def test_both_blockers_are_recorded_as_blockers(self):
        """Not as caveats. Either alone prevents execution, and a design that
        buried them in prose would read as ready."""
        doc = json.loads(DESIGN.read_text())
        assert "BLOCKER" in doc["contamination_protection"]
        assert "BLOCKER" in doc["budget"]
        assert doc["contamination_protection"]["batteries_available"] < \
            doc["contamination_protection"]["batteries_D1_requires"]
        assert doc["budget"]["shortfall_usd"] > 0
        assert doc["budget"]["d1_is_in_the_funded_list"] is False

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
