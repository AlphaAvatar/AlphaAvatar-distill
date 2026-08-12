"""Beam ranking: multi-objective, deterministic, auditable, and not NLL alone."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.metrics import MetricNamespaceError, StateEvaluation  # noqa: E402
from aadistill.autoinit.ranking import (  # noqa: E402
    PARETO_V1,
    BeamRankingPolicy,
    Objective,
    RankingError,
)
from aadistill.autoinit.state import OperatorStep, child_state, make_root_state  # noqa: E402

KL = "state.teacher_kl.equal_domain_mean"
CRIT = "state.critical_token_kl"
NLL = "state.nll.general"


def make_state(teacher_spec, target_spec, name, kl, crit, nll):
    parent = make_root_state(root_teacher_id="t", root_teacher_sha256="root",
                             spec=teacher_spec, target_spec=target_spec,
                             num_parameters=1, seed=1)
    state = child_state(parent, OperatorStep(
        index=0, kind="DEPTH", impl_id="depth.positional_v0", impl_signature_hash="s",
        profile_id=f"{name}@v1", profile_hash=name, config_hash="c", seed=1,
        result_spec_hash="r"), target_spec, 1, 1)
    sha = f"sha-{name}"
    state.mark_materialized(f"/tmp/{name}", sha, "cfg")
    state.mark_validated()
    state.attach_evaluation(StateEvaluation(
        checkpoint_sha256=sha, suite_id="t@v1", suite_hash="h",
        reference="root_teacher", positions=10,
        values={KL: kl, CRIT: crit, NLL: nll}))
    return state


def test_the_policy_hashes_and_is_versioned():
    assert PARETO_V1.qualified_id == "beam.pareto_multi_objective@v1"
    assert len(PARETO_V1.policy_hash) == 64
    moved = BeamRankingPolicy(
        policy_id=PARETO_V1.policy_id, version=PARETO_V1.version,
        description=PARETO_V1.description,
        objectives=PARETO_V1.objectives[:2], tie_break=PARETO_V1.tie_break,
        guardrails=PARETO_V1.guardrails)
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
        make_state(teacher_spec, target_spec, "worst_nll", kl=0.10, crit=0.10, nll=9.9),
        make_state(teacher_spec, target_spec, "best_nll", kl=0.90, crit=0.90, nll=2.0),
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
        checkpoint_sha256=partial.checkpoint_sha256, suite_id="t@v1", suite_hash="h",
        reference="root_teacher", positions=10, values={KL: 0.1}))
    result = PARETO_V1.rank([partial], beam_width=1)
    assert result.selected_ids == ()
    assert "missing required metrics" in result.decisions[0]["reason"]
