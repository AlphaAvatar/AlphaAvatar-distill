"""Search state: hash-bound metrics, no inheritance, and the recovery gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.artifact import CheckpointIdentity, ShardRecord  # noqa: E402
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


def artifact(digest_seed: str, path="/tmp/x", n_shards: int = 1) -> CheckpointIdentity:
    return CheckpointIdentity(
        path=path,
        shards=tuple(ShardRecord(f"model-{i:05d}.safetensors", f"{digest_seed}{i}", 10)
                     for i in range(n_shards)),
        config_sha256="cfg", arch_signature="arch", num_parameters=1)


def evaluation(art: CheckpointIdentity, values=None) -> StateEvaluation:
    return StateEvaluation(artifact_digest=art.artifact_digest, suite_id="t@v1",
                           suite_hash="h", reference="root_teacher",
                           values=values or VALUES, positions=100)


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
    art = artifact(sha)
    state.mark_materialized(art)
    state.mark_validated()
    state.attach_evaluation(evaluation(art))
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
    gart = artifact("grandchildsha", path="/tmp/y")
    grandchild.mark_materialized(gart)
    grandchild.mark_validated()
    with pytest.raises(MeasurementError, match="not inherited"):
        grandchild.attach_evaluation(parent.evaluation)
    grandchild.attach_evaluation(evaluation(gart))
    assert grandchild.evaluation.artifact_digest == gart.artifact_digest


def test_measuring_before_materializing_is_refused(teacher_spec, target_spec):
    state = child_state(root(teacher_spec, target_spec), step(), target_spec, 1, 1)
    with pytest.raises(StateError, match="before the checkpoint"):
        state.attach_evaluation(evaluation(artifact("anything")))


def test_ranking_requires_a_complete_measurement(teacher_spec, target_spec):
    state = measured_child(teacher_spec, target_spec)
    state.ready_for_ranking(list(VALUES))
    with pytest.raises(MeasurementError, match="missing required metrics"):
        state.ready_for_ranking([*VALUES, "state.teacher_kl.tool"])


def test_an_unmeasured_state_cannot_be_ranked(teacher_spec, target_spec):
    state = child_state(root(teacher_spec, target_spec), step(), target_spec, 1, 1)
    state.mark_materialized(artifact("sha", path="/tmp/z"))
    with pytest.raises(StateError, match="no hash-bound evaluation"):
        state.ready_for_ranking(list(VALUES))


# --- the load-bearing invariant --------------------------------------------


def test_an_intermediate_state_cannot_enter_recovery(teacher_spec, target_spec):
    """Intermediate checkpoints are search states only."""
    intermediate_spec = teacher_spec.replace(
        num_hidden_layers=target_spec["num_hidden_layers"])
    parent = root(teacher_spec, target_spec)
    state = child_state(parent, step(), intermediate_spec, 999, 1)
    art = artifact("sha", path="/tmp/i")
    state.mark_materialized(art)
    state.mark_validated()
    state.attach_evaluation(evaluation(art))

    assert not state.is_complete_leaf()
    assert state.remaining_differences() == {
        "hidden_size", "intermediate_size", "num_attention_heads"}
    with pytest.raises(StateError, match="intermediate search state"):
        state.require_recovery_admissible()


def test_a_complete_leaf_must_still_be_measured(teacher_spec, target_spec):
    parent = root(teacher_spec, target_spec)
    leaf = child_state(parent, step(), target_spec, 1, 1)
    art = artifact("sha", path="/tmp/l")
    leaf.mark_materialized(art)
    leaf.mark_validated()
    assert leaf.is_complete_leaf()
    with pytest.raises(StateError, match="hash-bound measurements"):
        leaf.require_recovery_admissible()
    leaf.attach_evaluation(evaluation(art))
    leaf.require_recovery_admissible()


def test_a_leaf_matches_the_target_field_for_field(teacher_spec, target_spec):
    """"Close enough" is not a category. One differing field disqualifies."""
    parent = root(teacher_spec, target_spec)
    almost = target_spec.replace(intermediate_size=target_spec["intermediate_size"] + 8)
    leaf = child_state(parent, step(), almost, 1, 1)
    art = artifact("sha", path="/tmp/a")
    leaf.mark_materialized(art)
    leaf.mark_validated()
    leaf.attach_evaluation(evaluation(art))
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
    assert latest["artifact_digest"] == state.artifact_digest
    assert latest["evaluation"]["values"] == VALUES
    assert len(store.records()) == 2


# --- the live snapshot must not deny a grant it also declares ---------------
#
# On 2026-09-07 `logs/current_state.json` said BOTH that the Attempt-9 grant was
# present and one-use, and — in `blocker` and `phase_c.c1.not_built` — that "No
# C1 grant exists" / "no grant". Three fields had simply not been updated when
# the grant landed. A handoff document that contradicts itself about whether a
# grant exists is worse than a stale one, because both halves look authoritative.
#
# Deliberately NOT a natural-language framework. It covers the owned live fields
# that produced this contradiction and nothing else.

SNAPSHOT = Path(__file__).resolve().parents[2] / "logs/current_state.json"

#: Phrases that DENY a grant. "the grant exists" must not match, so each is a
#: negation, not a keyword.
GRANT_DENIALS = (
    "no c1 grant",
    "no grant exists",
    "no grant is live",
    "no grant;",
    "no pod, grant or authorization",
    "grant: absent",
    "no grant or authorization",
)

#: The live fields that must agree. Named, because these are the ones that
#: actually went stale — `authorized.note`, `phase_c.c1.status`,
#: `next_starting_point.the_ask`, `phase_c.c1.not_built` and `blocker`.
LIVE_GRANT_FIELDS = (
    ("blocker",),
    ("authorized", "note"),
    ("phase_c", "c1", "status"),
    ("phase_c", "c1", "not_built"),
    ("phase_c", "c1", "needs"),
    ("next_starting_point", "the_ask"),
    ("next_starting_point", "status"),
)


def _snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text())


def _at(doc: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc if isinstance(doc, str) else None


def _declares_a_grant(doc: dict) -> bool:
    """Does the snapshot say an Attempt-9 grant is PRESENT?"""
    blob = json.dumps(doc).lower()
    return "attempt-9 grant" in blob or "autoinit_c1_attempt9_grant.json" in blob


def test_a_declared_grant_is_not_denied_by_any_live_field():
    #: Conditional by nature, and expressed WITHOUT `pytest.skip`: a skip
    #: predicate keyed on repository content is one more thing the pod/sweep
    #: skip-set comparison has to account for, and the audit correctly refused to
    #: resolve it. When no grant is declared the rule is vacuous, so the
    #: offender list is simply empty and the assertion still runs.
    doc = _snapshot()
    offenders = []
    for path in (LIVE_GRANT_FIELDS if _declares_a_grant(doc) else ()):
        text = _at(doc, path)
        if text is None:
            continue
        for denial in GRANT_DENIALS:
            if denial in text.lower():
                offenders.append(f"{'.'.join(path)}: {denial!r} in {text[:90]!r}")
    assert not offenders, (
        "the snapshot declares an Attempt-9 grant AND denies one:\n  "
        + "\n  ".join(offenders))


def test_the_grant_file_the_snapshot_names_actually_exists():
    doc = _snapshot()
    grant = SNAPSHOT.parent / "autoinit_c1_attempt9_grant.json"
    if not _declares_a_grant(doc):
        assert not grant.is_file() or True      # nothing is claimed, nothing owed
        return
    assert grant.is_file(), f"{grant} is named by the snapshot and absent"
    assert json.loads(grant.read_text())["schema"] == "aadistill.autoinit.c1_grant/v1"


def test_a_declared_grant_is_never_called_an_authorization():
    """A grant permits an ISSUANCE. The two must stay distinguishable."""
    doc = _snapshot()
    if not _declares_a_grant(doc):
        return
    assert doc["authorized"]["any"] is False
    assert doc["running"]["paid_compute"] is False
    assert doc["running"]["pods"] == 0
    assert doc["prepared_launch"]["any"] is False
