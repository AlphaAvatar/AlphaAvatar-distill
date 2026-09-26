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


def test_a_step_config_can_override_the_derived_key():
    """Last writer wins, and it is the step. Stated because it is a choice."""
    from aadistill.initialization.planning.fixed_path import step_operator_config

    step = FixedPathStep(impl_id="x", profile_id="p",
                         config={"n_calibration_items": 8})
    assert step_operator_config(step, 67)["n_calibration_items"] == 8


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
