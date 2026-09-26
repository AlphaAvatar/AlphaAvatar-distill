"""The pilot may bind a step's numerics without moving a historical hash.

Two separate guarantees, both `$0`, both blockers before paid work.

**The prefix must replay at B=1.** The frozen pre-ATTENTION parent
`eea90c91…` was produced by the historical one-item path. `materialize_fixed_path`
defaults to `DEFAULT_EXECUTION`, and on this branch that is
`micro_batch_size = 4` -- and B4 activation statistics are now known to change
FFN and WIDTH structural decisions. Reconstructing the prefix under the default
would therefore build a DIFFERENT parent. The digest gate would catch it, but
only after a pod had been paid for; these tests catch it here.

**A new step config must not move an old hash.** `FixedPathSpec.spec_hash` is
`sha256_json(as_dict())`, so a key added unconditionally to `FixedPathStep`
would change every historical fixed-path hash, including the C1
preregistration's. The field is omitted from the serialization when empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.execution import (  # noqa: E402
    DEFAULT_MICRO_BATCH_SIZE, ExecutionConfig,
)
from aadistill.initialization.planning.fixed_path import FixedPathStep  # noqa: E402


# --- the serialization must not move ---------------------------------------


def test_a_step_without_config_serializes_exactly_as_before():
    """The historical four keys, and nothing else."""
    step = FixedPathStep(impl_id="ffn.activation_importance_v0",
                         profile_id="calib.domain_balanced@v1",
                         expected_artifact_digest=None, label="FFN")
    assert step.as_dict() == {
        "impl_id": "ffn.activation_importance_v0",
        "profile_id": "calib.domain_balanced@v1",
        "expected_artifact_digest": None,
        "label": "FFN",
    }


@pytest.mark.parametrize("empty", [None, {}])
def test_an_empty_config_is_absent_from_the_serialization(empty):
    """`{}` must serialize like `None`, or the default form has two spellings."""
    step = FixedPathStep(impl_id="x", profile_id="p", config=empty)
    assert "config" not in step.as_dict()


def test_a_declared_config_appears_and_is_key_sorted():
    """Compared as SERIALIZED bytes, not as dicts.

    `a.as_dict() == b.as_dict()` cannot detect key order at all -- Python dict
    equality ignores it -- so that assertion passed against a version that did
    not sort, and the hash it is protecting is `sha256_json`, which does not.
    """
    import json

    from aadistill.infrastructure.manifest import sha256_json

    a = FixedPathStep(impl_id="x", profile_id="p", config={"b": 2, "a": 1})
    b = FixedPathStep(impl_id="x", profile_id="p", config={"a": 1, "b": 2})
    assert list(a.as_dict()["config"]) == ["a", "b"]
    assert json.dumps(a.as_dict()) == json.dumps(b.as_dict())
    assert sha256_json(a.as_dict()) == sha256_json(b.as_dict())


def test_the_executor_merges_a_step_config_over_the_derived_one():
    """The binding is worthless if the executor never reads it."""
    from aadistill.initialization.planning.fixed_path import step_operator_config

    plain = FixedPathStep(impl_id="x", profile_id="p")
    assert step_operator_config(plain, 67) == {"n_calibration_items": 67}

    bound = FixedPathStep(impl_id="x", profile_id="p",
                          config={"calibration_forward_batch_size": 4})
    assert step_operator_config(bound, 67) == {
        "n_calibration_items": 67, "calibration_forward_batch_size": 4}


def test_a_step_may_not_override_an_executor_derived_key():
    """REVERSED. The first version asserted "last writer wins", which is unsafe.

    `n_calibration_items` is not operator policy -- it is
    `len(ctx.calibration_items)`. A step allowed to override it would PLAN as
    if there were 8 items while EXECUTING against 67, and the plan is what the
    cost model and the reachability check read. It fails closed.
    """
    from aadistill.initialization.planning.fixed_path import (
        FixedPathError, step_operator_config,
    )

    step = FixedPathStep(impl_id="x", profile_id="p",
                         config={"n_calibration_items": 8})
    with pytest.raises(FixedPathError, match="executor-owned"):
        step_operator_config(step, 67)


def test_the_refusal_names_the_offending_key_and_the_step():
    from aadistill.initialization.planning.fixed_path import (
        FixedPathError, step_operator_config,
    )

    step = FixedPathStep(impl_id="attention.causal_kl_v1", profile_id="p",
                         config={"n_calibration_items": 8,
                                 "calibration_forward_batch_size": 4})
    with pytest.raises(FixedPathError) as excinfo:
        step_operator_config(step, 67)
    assert "n_calibration_items" in str(excinfo.value)
    assert "attention.causal_kl_v1" in str(excinfo.value)


def test_the_pilot_causal_step_survives_the_collision_check():
    """The real arms must pass the guard that the unsafe case fails."""
    from aadistill.initialization.planning.fixed_path import step_operator_config

    from experiments.phase_c3.pilot import causal_step

    for bs in (1, 4):
        merged = step_operator_config(causal_step(bs), 67)
        assert merged == {"n_calibration_items": 67,
                          "calibration_forward_batch_size": bs}


def test_the_real_item_count_reaches_the_plan():
    """The operator must be PLANNED against the corpus it will execute on."""
    from aadistill.initialization.planning.fixed_path import step_operator_config

    from experiments.phase_c3.pilot import causal_step

    assert step_operator_config(causal_step(4), 67)["n_calibration_items"] == 67
    assert step_operator_config(causal_step(4), 8)["n_calibration_items"] == 8


# --- the identity-bearing config must be immutable -------------------------


def test_mutating_the_input_dict_afterwards_does_not_move_the_hash():
    """A frozen dataclass holding a live dict is frozen in name only."""
    from aadistill.infrastructure.manifest import sha256_json

    source = {"calibration_forward_batch_size": 4}
    step = FixedPathStep(impl_id="x", profile_id="p", config=source)
    before = sha256_json(step.as_dict())

    source["calibration_forward_batch_size"] = 1
    source["injected"] = True
    assert sha256_json(step.as_dict()) == before


def test_the_stored_config_cannot_be_mutated_through_the_step():
    step = FixedPathStep(impl_id="x", profile_id="p", config={"a": 1})
    with pytest.raises(TypeError):
        step.config["b"] = 2


def test_the_serialization_is_a_copy_not_the_stored_mapping():
    """Mutating what `as_dict()` returns must not reach the step."""
    from aadistill.infrastructure.manifest import sha256_json

    step = FixedPathStep(impl_id="x", profile_id="p", config={"a": 1})
    before = sha256_json(step.as_dict())
    step.as_dict()["config"]["a"] = 99
    assert sha256_json(step.as_dict()) == before


# --- the prefix must match the FROZEN C1 path, not a pilot constant --------


def _committed_c1_path():
    """The four steps as the committed C1 replay record declares them."""
    import json

    from experiments.phase_c3.pilot import C1_PATH_RECORD

    doc = json.loads((REPO / C1_PATH_RECORD).read_text())
    return doc["path"]["steps"]


def test_the_pilot_prefix_matches_the_committed_c1_path():
    """Read from the RECORD, so the pilot constant cannot drift from the path.

    The first version of the pilot gave all three prefix steps
    `calib.domain_balanced@v1`. WIDTH actually ran against
    `calib.reasoning_heavy@v2`, so it would have replayed the wrong mixture
    and reconstructed a different parent -- refused by the digest gate, but
    only after an expensive replay.
    """
    from experiments.phase_c3.pilot import prefix_steps

    committed = _committed_c1_path()[:3]
    pilot = prefix_steps()
    assert len(pilot) == 3
    for step, record in zip(pilot, committed):
        assert step.impl_id == record["impl_id"]
        assert step.profile_id == record["profile_id"]


def test_the_frozen_profiles_are_not_uniform():
    """States the trap explicitly, so a future uniform refactor fails here."""
    from experiments.phase_c3.pilot import prefix_steps

    by_impl = {s.impl_id: s.profile_id for s in prefix_steps()}
    assert by_impl["depth.causal_kl_greedy_v1"] == "calib.domain_balanced@v1"
    assert by_impl["ffn.activation_importance_v0"] == "calib.domain_balanced@v1"
    assert by_impl["width.global_pca_v0"] == "calib.reasoning_heavy@v2"
    assert len(set(by_impl.values())) == 2, "the prefix uses TWO profiles"


def test_only_the_width_step_is_pinned_to_the_frozen_parent_digest():
    from experiments.phase_c3.pilot import FROZEN_PARENT_DIGEST, prefix_steps

    steps = prefix_steps(pin_parent=True)
    pinned = [s for s in steps if s.expected_artifact_digest]
    assert len(pinned) == 1
    assert pinned[0].impl_id == "width.global_pca_v0"
    assert pinned[0].expected_artifact_digest == FROZEN_PARENT_DIGEST


def test_the_frozen_digest_matches_the_committed_arm_identity():
    """The digest is evidence, cross-checked against a second committed file."""
    import json

    from experiments.phase_c3.pilot import C1_ARM_IDENTITIES, FROZEN_PARENT_DIGEST

    arms = json.loads((REPO / C1_ARM_IDENTITIES).read_text())
    assert arms["parent"]["artifact_digest"] == FROZEN_PARENT_DIGEST
    assert arms["parent"]["impl_id"] == "width.global_pca_v0"
    assert arms["parent"]["profile_id"] == "calib.reasoning_heavy@v2"


def test_both_arms_differ_only_by_the_protocol_config():
    """Same impl, same profile, same label. Only the batch size separates them."""
    from experiments.phase_c3.pilot import BATCH_SIZE_CONFIG_KEY, causal_step

    b1, b4 = causal_step(1), causal_step(4)
    assert b1.impl_id == b4.impl_id
    assert b1.profile_id == b4.profile_id
    assert b1.label == b4.label, (
        "a differing label is a SECOND identity difference; the arm names "
        "belong in the pilot record, not the step")
    assert set(b1.config) == set(b4.config) == {BATCH_SIZE_CONFIG_KEY}


def test_the_committed_c1_fixed_path_hash_is_unchanged():
    """The real historical spec, rehashed through the modified type.

    This is the guarantee that matters: the C1 preregistration pins a
    fixed-path hash, and an identity migration would invalidate a frozen
    scientific record.
    """
    from aadistill.infrastructure.manifest import sha256_json

    #: The four historical keys, exactly as they were serialized before the
    #: `config` field existed. Hashing this literal and the live `as_dict()`
    #: must agree -- if they ever diverge, the serialization moved.
    historical = {"impl_id": "attention.activation_importance_v1",
                  "profile_id": "calib.domain_balanced@v1",
                  "expected_artifact_digest": None, "label": "ATTENTION"}
    live = FixedPathStep(**historical).as_dict()
    assert sha256_json(live) == sha256_json(historical)


def test_two_batch_sizes_give_the_causal_step_different_identities():
    """B1 and B4 can materialize different head maps, so they must not collide."""
    from aadistill.infrastructure.manifest import sha256_json

    from experiments.phase_c3.pilot import causal_step

    #: Through the PILOT's own constructor, not a hand-built step -- otherwise
    #: `causal_step` could stop binding the config and this would still pass.
    from experiments.phase_c3.pilot import BATCH_SIZE_CONFIG_KEY

    b1, b4 = causal_step(1), causal_step(4)
    assert b1.as_dict()["impl_id"] == b4.as_dict()["impl_id"], (
        "the same implementation is under test; only the protocol differs")

    #: The CONFIG must carry the argument. Asserting only that the two hashes
    #: differ is not enough: the labels differ too ("causal ATTENTION B1" vs
    #: "B4"), so a `causal_step` that hardcoded the batch size would still
    #: produce two different hashes -- for the wrong reason. A mutation showed
    #: exactly that.
    assert b1.config[BATCH_SIZE_CONFIG_KEY] == 1
    assert b4.config[BATCH_SIZE_CONFIG_KEY] == 4

    #: And they must differ with the LABELS EQUALISED, so the protocol is what
    #: separates them.
    same_label = [FixedPathStep(impl_id=s.impl_id, profile_id=s.profile_id,
                                label="fixed", config=s.config)
                  for s in (b1, b4)]
    assert sha256_json(same_label[0].as_dict()) != sha256_json(
        same_label[1].as_dict())
    assert sha256_json(b1.as_dict()) != sha256_json(b4.as_dict())


def test_binding_the_causal_step_leaves_the_other_steps_untouched():
    """Only the causal step carries a config; the prefix hashes do not move."""
    from aadistill.infrastructure.manifest import sha256_json

    prefix = [FixedPathStep(impl_id=i, profile_id="calib.domain_balanced@v1")
              for i in ("depth.causal_kl_greedy_v1",
                        "ffn.activation_importance_v0",
                        "width.global_pca_v0")]
    before = [sha256_json(s.as_dict()) for s in prefix]
    causal = FixedPathStep(impl_id="attention.causal_kl_v1",
                           profile_id="calib.domain_balanced@v1",
                           config={"calibration_forward_batch_size": 4})
    assert sha256_json(causal.as_dict()) not in before
    after = [sha256_json(s.as_dict()) for s in prefix]
    assert before == after


# --- the prefix must replay at B=1 ------------------------------------------


def test_the_branch_default_is_not_one_so_the_pilot_must_be_explicit():
    """States the hazard as a fact, so the next test has a reason to exist.

    If this ever becomes 1, the explicit binding below is still required --
    relying on a default that happens to be right is how the prefix would
    silently change the next time someone tunes it.
    """
    assert DEFAULT_MICRO_BATCH_SIZE != 1, (
        "if the default becomes 1, keep the explicit pilot binding anyway")


def test_the_pilot_prefix_execution_is_b1_and_not_the_default():
    from experiments.phase_c3.pilot import PREFIX_EXECUTION

    assert isinstance(PREFIX_EXECUTION, ExecutionConfig)
    assert PREFIX_EXECUTION.micro_batch_size == 1


def test_the_pilot_prefix_execution_is_a_literal_not_a_reference():
    """`PREFIX_EXECUTION` must not be `DEFAULT_EXECUTION` under another name.

    An alias would track the default, which is exactly the coupling the frozen
    prefix must not have.
    """
    import ast

    from experiments.phase_c3 import pilot

    src = Path(pilot.__file__).read_text()
    tree = ast.parse(src)
    found = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "PREFIX_EXECUTION"
                        for t in node.targets)):
            found = node.value
    assert found is not None, "PREFIX_EXECUTION is not assigned in the module"
    assert isinstance(found, ast.Call), (
        "PREFIX_EXECUTION must CONSTRUCT an ExecutionConfig, not alias one")
    kwargs = {k.arg: k.value for k in found.keywords}
    assert "micro_batch_size" in kwargs, (
        "the batch size must be passed explicitly, not defaulted")
    assert isinstance(kwargs["micro_batch_size"], ast.Constant)
    assert kwargs["micro_batch_size"].value == 1
    assert "DEFAULT_EXECUTION" not in src.split("PREFIX_EXECUTION")[1][:200]


def test_every_prefix_step_the_pilot_declares_carries_no_config():
    """The B1/B4 variation begins at ATTENTION and nowhere earlier."""
    from experiments.phase_c3.pilot import prefix_steps

    for step in prefix_steps():
        assert not step.config, (
            f"{step.impl_id} carries a config; the prefix must be identical "
            "for both pilot arms")
        assert step.impl_id != "attention.causal_kl_v1"
