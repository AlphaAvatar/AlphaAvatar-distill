"""Beam ranking: multi-objective, deterministic, auditable, and not NLL alone."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.initialization  # noqa: F401,E402
from aadistill.initialization.specs.metrics import MetricNamespaceError, StateEvaluation
from aadistill.initialization.planning.ranking import (  # noqa: E402
    PARETO_V1,
    BeamRankingPolicy,
    Objective,
    RankingError,
)
from aadistill.initialization.specs.artifact import (  # noqa: E402
    CheckpointIdentity,
    ShardRecord,
)
from aadistill.initialization.specs.state import (  # noqa: E402
    OperatorStep,
    child_state,
    make_root_state,
)

KL = "state.teacher_kl.equal_domain_mean"
WORST = "state.teacher_kl.worst_domain"
CRIT = "state.critical_token_kl"
NLL = "state.nll.general"


def make_state(teacher_spec, target_spec, name, kl, crit, nll, worst=None,
               parent_impl=None):
    parent = make_root_state(root_teacher_id="t", root_teacher_sha256="root",
                             spec=teacher_spec, target_spec=target_spec,
                             num_parameters=1, seed=1)
    steps = []
    if parent_impl:
        steps.append(OperatorStep(
            index=0, kind="DEPTH", impl_id=parent_impl, impl_signature_hash="s",
            profile_id="p@v1", profile_hash="p", config_hash="c", seed=1,
            result_spec_hash="r"))
        parent = child_state(parent, steps[0], target_spec, 1, 1)
    state = child_state(parent, OperatorStep(
        index=len(steps), kind="FFN", impl_id="ffn.activation_importance_v0",
        impl_signature_hash="s", profile_id=f"{name}@v1", profile_hash=name,
        config_hash="c", seed=1, result_spec_hash="r"), target_spec, 1, 1)
    art = CheckpointIdentity(
        path=f"/tmp/{name}",
        shards=(ShardRecord("model.safetensors", f"sha-{name}", 10),),
        config_sha256="cfg", arch_signature="arch", num_parameters=1)
    state.mark_materialized(art)
    state.mark_validated()
    state.attach_evaluation(StateEvaluation(
        artifact_digest=art.artifact_digest, suite_id="t@v1", suite_hash="h",
        reference="root_teacher", positions=10,
        values={KL: kl, WORST: worst if worst is not None else kl,
                CRIT: crit, NLL: nll}))
    return state


def test_the_policy_hashes_and_is_versioned():
    assert PARETO_V1.qualified_id == "beam.pareto_multi_objective@v2"
    assert len(PARETO_V1.policy_hash) == 64
    moved = BeamRankingPolicy(
        policy_id=PARETO_V1.policy_id, version=PARETO_V1.version,
        description=PARETO_V1.description,
        objectives=PARETO_V1.objectives[:2], tie_break=PARETO_V1.tie_break,
        guardrails=PARETO_V1.guardrails, epsilon={}, diversity_key="parent_path")
    assert moved.policy_hash != PARETO_V1.policy_hash


def test_objectives_must_be_state_metrics():
    with pytest.raises(MetricNamespaceError, match="operator_local"):
        Objective("op.depth.causal_kl.final", "minimize")
    with pytest.raises(MetricNamespaceError, match="no level namespace"):
        Objective("teacher_kl", "minimize")


def test_a_single_objective_beam_needs_an_explicit_acknowledgement():
    """E7 is the reason: a -5.22 nat NLL swing moved behaviour by +0.0000."""
    with pytest.raises(RankingError, match="single-objective"):
        BeamRankingPolicy(policy_id="p", version=1, description="",
                          objectives=(Objective(NLL),), tie_break=(NLL, "state_id"))
    ok = BeamRankingPolicy(policy_id="p", version=1, description="",
                           objectives=(Objective(NLL),), tie_break=(NLL, "state_id"),
                           metadata={"single_objective_acknowledged": True})
    assert ok.required_metrics() == (NLL,)


def test_the_tie_break_must_be_total():
    with pytest.raises(RankingError, match="tie_break must end"):
        BeamRankingPolicy(policy_id="p", version=1, description="",
                          objectives=PARETO_V1.objectives, tie_break=(KL,))


def test_nll_alone_cannot_prune_a_state_that_leads_on_fidelity(teacher_spec, target_spec):
    """The concrete E7 protection.

    ``worst_nll`` has by far the worst general NLL and the best teacher KL. Under
    a minimum-NLL beam of 1 it is gone; under the Pareto policy it is
    non-dominated and survives.
    """
    states = [
        make_state(teacher_spec, target_spec, "worst_nll", kl=0.10, crit=0.90, nll=9.9),
        make_state(teacher_spec, target_spec, "best_nll", kl=0.90, crit=0.10, nll=2.0),
    ]
    result = PARETO_V1.rank(states, beam_width=1)
    assert len(result.fronts[0]) == 2, "neither dominates the other"
    # Beam of 1 forces a tie-break, and the configured first key is teacher KL.
    assert result.selected[0].profile_ids == ("worst_nll@v1",)

    nll_only = BeamRankingPolicy(
        policy_id="nll_only", version=1, description="",
        objectives=(Objective(NLL),), tie_break=(NLL, "state_id"),
        metadata={"single_objective_acknowledged": True})
    assert nll_only.rank(states, beam_width=1).selected[0].profile_ids == ("best_nll@v1",)
    # ... and NLL is not an objective of the shipped v1 policy at all.
    assert NLL not in {o.key for o in PARETO_V1.objectives}
    assert NLL not in PARETO_V1.tie_break


def test_dominated_states_are_pruned_and_non_dominated_ones_are_kept(
        teacher_spec, target_spec):
    states = [
        make_state(teacher_spec, target_spec, "a", 0.1, 0.1, 1.0),   # dominates c
        make_state(teacher_spec, target_spec, "b", 0.9, 0.05, 1.0),  # trades with a
        make_state(teacher_spec, target_spec, "c", 0.2, 0.2, 2.0),   # dominated by a
    ]
    result = PARETO_V1.rank(states, beam_width=3)
    assert set(result.fronts[0]) == {states[0].state_id, states[1].state_id}
    assert result.fronts[1] == (states[2].state_id,)


def test_ranking_is_deterministic_under_input_reordering(teacher_spec, target_spec):
    states = [make_state(teacher_spec, target_spec, n, kl, crit, nll)
              for n, kl, crit, nll in [("a", 0.5, 0.5, 3.0), ("b", 0.5, 0.5, 3.0),
                                       ("c", 0.4, 0.7, 3.0), ("d", 0.7, 0.4, 3.0)]]
    first = PARETO_V1.rank(states, beam_width=2).selected_ids
    for order in ([3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
        shuffled = [states[i] for i in order]
        assert PARETO_V1.rank(shuffled, beam_width=2).selected_ids == first
    # `a` and `b` are identical on every objective; only the state-id tie-break
    # separates them, and it must do so the same way every time.
    assert len(set(first)) == 2


def test_every_pruned_state_carries_a_reason(teacher_spec, target_spec):
    states = [make_state(teacher_spec, target_spec, n, kl, 0.5, 3.0)
              for n, kl in [("a", 0.1), ("b", 0.2), ("c", 0.3)]]
    result = PARETO_V1.rank(states, beam_width=1)
    assert len(result.pruned) == 2
    for decision in result.pruned:
        assert decision["reason"].startswith("pruned:")
        assert decision["state_id"]
        assert decision["front"] is not None
    assert all(d["objectives"] for d in result.decisions if d["front"] is not None)


def test_an_unmeasured_state_is_rejected_with_a_reason(teacher_spec, target_spec):
    good = make_state(teacher_spec, target_spec, "good", 0.1, 0.1, 1.0)
    parent = make_root_state(root_teacher_id="t", root_teacher_sha256="root",
                             spec=teacher_spec, target_spec=target_spec,
                             num_parameters=1, seed=1)
    unmeasured = child_state(parent, OperatorStep(
        index=0, kind="FFN", impl_id="ffn.activation_importance_v0",
        impl_signature_hash="s", profile_id="p@v1", profile_hash="p", config_hash="c",
        seed=1, result_spec_hash="r"), target_spec, 1, 1)

    result = PARETO_V1.rank([good, unmeasured], beam_width=5)
    assert result.selected_ids == (good.state_id,)
    reason = next(d["reason"] for d in result.decisions
                  if d["state_id"] == unmeasured.state_id)
    assert "not measured" in reason


def test_a_state_missing_a_required_metric_is_rejected_not_defaulted(
        teacher_spec, target_spec):
    partial = make_state(teacher_spec, target_spec, "partial", 0.1, 0.1, 1.0)
    partial.attach_evaluation(StateEvaluation(
        artifact_digest=partial.artifact_digest, suite_id="t@v1", suite_hash="h",
        reference="root_teacher", positions=10, values={KL: 0.1}))
    result = PARETO_V1.rank([partial], beam_width=1)
    assert result.selected_ids == ()
    assert "missing required metrics" in result.decisions[0]["reason"]


# ---------------------------------------------------------------------------
# diversity is an EXPLORATION mechanism, not a winner-selection mechanism
#
# Standing maintainer policy, 2026-10-07, for every full-search experiment:
# during intermediate beam levels, retain with lineage diversity so one early
# proxy measurement cannot extinguish a structural family; once complete leaves
# exist, retain by the scientific quality ordering ALONE.
#
# Measured, not theoretical. A completed 12-leaf search committed a finalist at
# quality position 11 of 12 -- over twice the best leaf's objective value, and
# worse than seven leaves it excluded -- because that leaf was the sole member of
# its lineage, while the candidates at quality positions 2 and 4 were excluded
# for sharing one. The widening meant to admit near-misses excluded exactly them.
# ---------------------------------------------------------------------------

def _lineaged(teacher_spec, target_spec):
    """Two lineages, where one dominates the other at every position.

    `a1` and `a2` are the two best states overall and share a lineage; `b1` is
    worse than both and is the only member of its own. This is the shape that
    makes the two retention rules disagree, and it is the shape the real search
    produced.
    """
    a1 = make_state(teacher_spec, target_spec, "a1", kl=1.0, crit=1.0, nll=9.0,
                    parent_impl="depth.causal_kl_greedy_v1")
    a2 = make_state(teacher_spec, target_spec, "a2", kl=2.0, crit=2.0, nll=9.0,
                    parent_impl="depth.causal_kl_greedy_v1")
    b1 = make_state(teacher_spec, target_spec, "b1", kl=9.0, crit=9.0, nll=9.0,
                    parent_impl="width.global_pca_v0")
    return a1, a2, b1


class TestQualityOrderIsOneSharedOrdering:

    def test_it_is_fronts_best_first_with_the_tie_break_inside(
            self, teacher_spec, target_spec):
        a1, a2, b1 = _lineaged(teacher_spec, target_spec)
        order = PARETO_V1.quality_order([b1, a2, a1])
        assert [s.state_id for s in order.ordered] == \
            [a1.state_id, a2.state_id, b1.state_id]
        assert order.front_of(a1.state_id) == 0

    def test_it_carries_no_width_and_no_selection(self, teacher_spec, target_spec):
        """K is the caller's. A quality order that embedded one would be a
        retention rule wearing an ordering's name."""
        order = PARETO_V1.quality_order(list(_lineaged(teacher_spec, target_spec)))
        assert len(order.ordered) == 3
        assert [s.state_id for s in order.take(2)] == list(order.ordered_ids[:2])
        with pytest.raises(RankingError):
            order.take(0)

    def test_it_is_deterministic_across_input_orderings(
            self, teacher_spec, target_spec):
        a1, a2, b1 = _lineaged(teacher_spec, target_spec)
        first = PARETO_V1.quality_order([a1, a2, b1]).ordered_ids
        second = PARETO_V1.quality_order([b1, a1, a2]).ordered_ids
        third = PARETO_V1.quality_order([a2, b1, a1]).ordered_ids
        assert first == second == third

    def test_it_is_not_a_sort_by_any_single_objective(
            self, teacher_spec, target_spec):
        """Collapsing a multi-objective search into one scalar is the failure the
        fronts exist to avoid, so a state that leads on one axis and trails on
        another must share a front rather than be ordered by either."""
        x = make_state(teacher_spec, target_spec, "x", kl=1.0, crit=9.0, nll=9.0)
        y = make_state(teacher_spec, target_spec, "y", kl=9.0, crit=1.0, nll=9.0)
        order = PARETO_V1.quality_order([x, y])
        assert len(order.fronts) == 1 and len(order.fronts[0]) == 2


class TestTheTwoRetentionRulesDiffer:

    def test_beam_retention_may_take_a_worse_state_to_keep_a_lineage(
            self, teacher_spec, target_spec):
        a1, a2, b1 = _lineaged(teacher_spec, target_spec)
        beam = PARETO_V1.rank([a1, a2, b1], 2)
        assert set(beam.selected_ids) == {a1.state_id, b1.state_id}
        assert a2.state_id not in beam.selected_ids, (
            "the beam must give the second lineage a slot before the first "
            "lineage gets two -- that is what preserves exploration")
        assert beam.retention == PARETO_V1.RETENTION_WITH_DIVERSITY

    def test_finalist_retention_never_skips_a_better_state_for_diversity(
            self, teacher_spec, target_spec):
        a1, a2, b1 = _lineaged(teacher_spec, target_spec)
        final = PARETO_V1.rank([a1, a2, b1], 2, diversity=False)
        assert list(final.selected_ids) == [a1.state_id, a2.state_id]
        assert b1.state_id not in final.selected_ids
        assert final.retention == PARETO_V1.RETENTION_QUALITY_ONLY

    def test_the_same_states_and_k_legitimately_give_different_sets(
            self, teacher_spec, target_spec):
        """Not a contradiction: the two rules answer different questions."""
        states = list(_lineaged(teacher_spec, target_spec))
        beam = PARETO_V1.rank(states, 2)
        final = PARETO_V1.rank(states, 2, diversity=False)
        assert set(beam.selected_ids) != set(final.selected_ids)
        assert beam.selected_ids[0] == final.selected_ids[0]

    def test_the_ordering_underneath_is_identical(self, teacher_spec, target_spec):
        """Only retention differs. If the fronts differed, a finalist selection
        would be ranking on a different notion of quality than the beam did."""
        states = list(_lineaged(teacher_spec, target_spec))
        beam = PARETO_V1.rank(states, 2)
        final = PARETO_V1.rank(states, 2, diversity=False)
        assert beam.fronts == final.fronts

    def test_beam_retention_is_the_default(self, teacher_spec, target_spec):
        """Every search written before this policy keeps its behaviour."""
        states = list(_lineaged(teacher_spec, target_spec))
        assert PARETO_V1.rank(states, 2).selected_ids == \
            PARETO_V1.rank(states, 2, diversity=True).selected_ids

    def test_k_is_supplied_by_the_caller(self, teacher_spec, target_spec):
        states = list(_lineaged(teacher_spec, target_spec))
        for k in (1, 2, 3):
            assert len(PARETO_V1.rank(states, k, diversity=False).selected) == k

    def test_a_record_says_which_rule_produced_it(self, teacher_spec, target_spec):
        states = list(_lineaged(teacher_spec, target_spec))
        assert PARETO_V1.rank(states, 2).as_dict()["retention"] == \
            "quality_with_lineage_diversity"
        assert PARETO_V1.rank(states, 2, diversity=False).as_dict()["retention"] == \
            "quality_only"

    def test_every_decision_records_its_quality_position(
            self, teacher_spec, target_spec):
        """A finalist record's whole claim is about this number, so it is written
        rather than left for a reader to reconstruct from (front, position)."""
        states = list(_lineaged(teacher_spec, target_spec))
        final = PARETO_V1.rank(states, 2, diversity=False)
        ranked = {d["state_id"]: d["quality_order"]
                  for d in final.decisions if d["front"] is not None}
        assert sorted(ranked.values()) == [1, 2, 3]
        kept = [d for d in final.decisions if d["selected"]]
        assert all("lineage diversity NOT applied" in d["reason"] for d in kept)
