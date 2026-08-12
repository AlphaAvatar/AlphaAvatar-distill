"""Search state: hash-bound metrics, no inheritance, and the recovery gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.metrics import MeasurementError, StateEvaluation  # noqa: E402
from aadistill.autoinit.state import (  # noqa: E402
    InitializationState,
    OperatorStep,
    StateError,
    StateStore,
    StateValidity,
    child_state,
    compute_state_id,
    make_root_state,
)

VALUES = {
    "state.teacher_kl.equal_domain_mean": 0.5,
    "state.critical_token_kl": 0.6,
    "state.nll.general": 3.0,
}


def evaluation(sha: str, values=None) -> StateEvaluation:
    return StateEvaluation(checkpoint_sha256=sha, suite_id="t@v1", suite_hash="h",
                           reference="root_teacher", values=values or VALUES,
                           positions=100)


def step(index=0, kind="DEPTH", impl="depth.positional_v0", profile="p@v1",
         profile_hash="ph", spec_hash="sh") -> OperatorStep:
    return OperatorStep(index=index, kind=kind, impl_id=impl, impl_signature_hash="sig",
                        profile_id=profile, profile_hash=profile_hash, config_hash="c",
                        seed=1, result_spec_hash=spec_hash)


def root(teacher_spec, target_spec) -> InitializationState:
    return make_root_state(root_teacher_id="t", root_teacher_sha256="rootsha",
                           spec=teacher_spec, target_spec=target_spec,
                           num_parameters=1, seed=1)


def measured_child(teacher_spec, target_spec, sha="childsha") -> InitializationState:
    parent = root(teacher_spec, target_spec)
    state = child_state(parent, step(), target_spec, 1, 1)
    state.mark_materialized("/tmp/x", sha, "cfg")
    state.mark_validated()
    state.attach_evaluation(evaluation(sha))
    return state


def test_a_child_starts_with_no_metrics_at_all(teacher_spec, target_spec):
    parent = measured_child(teacher_spec, target_spec)
    grandchild = child_state(parent, step(index=1, kind="FFN"), target_spec, 1, 1)
    assert grandchild.evaluation is None
    assert grandchild.checkpoint_sha256 is None
    assert grandchild.validity is StateValidity.PLANNED


def test_a_parents_evaluation_cannot_be_attached_to_its_child(teacher_spec, target_spec):
    """'Inherit the parent's NLL to save a forward pass' has no code path."""
    parent = measured_child(teacher_spec, target_spec, sha="parentsha")
    grandchild = child_state(parent, step(index=1, kind="FFN"), target_spec, 1, 1)
    grandchild.mark_materialized("/tmp/y", "grandchildsha", "cfg")
    grandchild.mark_validated()
    with pytest.raises(MeasurementError, match="not inherited"):
        grandchild.attach_evaluation(parent.evaluation)
    grandchild.attach_evaluation(evaluation("grandchildsha"))
    assert grandchild.evaluation.checkpoint_sha256 == "grandchildsha"


def test_measuring_before_materializing_is_refused(teacher_spec, target_spec):
    state = child_state(root(teacher_spec, target_spec), step(), target_spec, 1, 1)
    with pytest.raises(StateError, match="before the checkpoint"):
        state.attach_evaluation(evaluation("anything"))


def test_ranking_requires_a_complete_measurement(teacher_spec, target_spec):
    state = measured_child(teacher_spec, target_spec)
    state.ready_for_ranking(list(VALUES))
    with pytest.raises(MeasurementError, match="missing required metrics"):
        state.ready_for_ranking([*VALUES, "state.teacher_kl.tool"])


def test_an_unmeasured_state_cannot_be_ranked(teacher_spec, target_spec):
    state = child_state(root(teacher_spec, target_spec), step(), target_spec, 1, 1)
    state.mark_materialized("/tmp/z", "sha", "cfg")
    with pytest.raises(StateError, match="no hash-bound evaluation"):
        state.ready_for_ranking(list(VALUES))


# --- the load-bearing invariant --------------------------------------------


def test_an_intermediate_state_cannot_enter_recovery(teacher_spec, target_spec):
    """Intermediate checkpoints are search states only."""
    intermediate_spec = teacher_spec.replace(
        num_hidden_layers=target_spec["num_hidden_layers"])
    parent = root(teacher_spec, target_spec)
    state = child_state(parent, step(), intermediate_spec, 999, 1)
    state.mark_materialized("/tmp/i", "sha", "cfg")
    state.mark_validated()
    state.attach_evaluation(evaluation("sha"))

    assert not state.is_complete_leaf()
    assert state.remaining_differences() == {
        "hidden_size", "intermediate_size", "num_attention_heads"}
    with pytest.raises(StateError, match="intermediate search state"):
        state.require_recovery_admissible()


def test_a_complete_leaf_must_still_be_measured(teacher_spec, target_spec):
    parent = root(teacher_spec, target_spec)
    leaf = child_state(parent, step(), target_spec, 1, 1)
    leaf.mark_materialized("/tmp/l", "sha", "cfg")
    leaf.mark_validated()
    assert leaf.is_complete_leaf()
    with pytest.raises(StateError, match="hash-bound measurements"):
        leaf.require_recovery_admissible()
    leaf.attach_evaluation(evaluation("sha"))
    leaf.require_recovery_admissible()


def test_a_leaf_matches_the_target_field_for_field(teacher_spec, target_spec):
    """"Close enough" is not a category. One differing field disqualifies."""
    parent = root(teacher_spec, target_spec)
    almost = target_spec.replace(intermediate_size=target_spec["intermediate_size"] + 8)
    leaf = child_state(parent, step(), almost, 1, 1)
    leaf.mark_materialized("/tmp/a", "sha", "cfg")
    leaf.mark_validated()
    leaf.attach_evaluation(evaluation("sha"))
    assert not leaf.is_complete_leaf()
    with pytest.raises(StateError, match="intermediate search state"):
        leaf.require_recovery_admissible()


# --- identity and the journal ----------------------------------------------


def test_state_ids_are_content_derived_and_order_sensitive(teacher_spec, target_spec):
    depth = step(0, "DEPTH", "depth.positional_v0")
    ffn = step(0, "FFN", "ffn.activation_importance_v0")
    a = compute_state_id("root", target_spec.spec_hash, [depth, ffn])
    b = compute_state_id("root", target_spec.spec_hash, [ffn, depth])
    assert a != b, "operator order must change the state identity"
    assert a == compute_state_id("root", target_spec.spec_hash, [depth, ffn])
    # A different calibration profile is a different state, not a relabelling.
    other_profile = step(0, "DEPTH", "depth.positional_v0", profile="q@v1",
                         profile_hash="qh")
    assert compute_state_id("root", target_spec.spec_hash, [other_profile, ffn]) != a
    # And so is a different target.
    assert compute_state_id("root", "otherhash", [depth, ffn]) != a


def test_operator_order_is_preserved_in_the_record(teacher_spec, target_spec):
    parent = root(teacher_spec, target_spec)
    s1 = child_state(parent, step(0, "ATTENTION", "attention.weight_proxy_v0"),
                     teacher_spec.replace(num_attention_heads=2), 1, 1)
    s2 = child_state(s1, step(1, "DEPTH", "depth.positional_v0"),
                     teacher_spec.replace(num_attention_heads=2, num_hidden_layers=4), 1, 1)
    assert s2.applied_kinds == ("ATTENTION", "DEPTH")
    assert [st.index for st in s2.steps] == [0, 1]
    assert s2.as_dict()["applied_kinds"] == ["ATTENTION", "DEPTH"]


def test_per_operator_calibration_profiles_reach_the_record(teacher_spec, target_spec):
    parent = root(teacher_spec, target_spec)
    s1 = child_state(parent, step(0, "DEPTH", profile="reasoning@v1",
                                  profile_hash="r"),
                     teacher_spec.replace(num_hidden_layers=4), 1, 1)
    s2 = child_state(s1, step(1, "FFN", "ffn.activation_importance_v0",
                              profile="balanced@v1", profile_hash="b"),
                     teacher_spec.replace(num_hidden_layers=4, intermediate_size=24), 1, 1)
    assert s2.profile_ids == ("reasoning@v1", "balanced@v1")
    record = s2.as_dict()
    assert record["calibration_profiles"] == ["reasoning@v1", "balanced@v1"]
    assert [st["profile_hash"] for st in record["steps"]] == ["r", "b"]
    assert "DEPTH(reasoning@v1)->FFN(balanced@v1)" == s2.path_label


def test_pruned_states_stay_auditable(tmp_path, teacher_spec, target_spec):
    store = StateStore(tmp_path / "states.jsonl")
    state = measured_child(teacher_spec, target_spec)
    store.append(state)
    state.mark_pruned("pruned: dominated (front 1) and the beam was full")
    store.append(state)

    latest = store.latest_by_state_id()[state.state_id]
    assert latest["validity"] == "pruned"
    assert "dominated" in latest["prune_reason"]
    # Its metrics and hash survive the pruning: the record still supports
    # re-deriving why it lost.
    assert latest["checkpoint_sha256"] == "childsha"
    assert latest["evaluation"]["values"] == VALUES
    assert len(store.records()) == 2
