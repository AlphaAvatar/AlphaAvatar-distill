"""OperatorKind vs OperatorImplementation, id immutability, capability dispatch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.autoinit  # noqa: F401,E402  (registers the v1 library)
from aadistill.autoinit.arch import ArchSpec, Capability  # noqa: E402
from aadistill.autoinit.operators.base import (  # noqa: E402
    CalibrationNeed,
    OperatorError,
    OperatorImplementation,
    OperatorKindSpec,
    OperatorPlan,
    applicable_implementations,
    get_implementation,
    get_kind,
    implementations_for_kind,
    register_implementation,
    register_kind,
    registered_implementations,
    registered_kinds,
    unregister_implementation,
    unregister_kind,
    verify_ledger,
)


def test_a_kind_is_not_an_implementation():
    """One structural dimension, two algorithms that disagree about it.

    E8a is the reason this distinction is structural rather than cosmetic: the
    positional map and the causal search share exactly one removed layer out of
    eight for 36 -> 28.
    """
    depth_impls = implementations_for_kind("DEPTH")
    assert {i.impl_id for i in depth_impls} == {
        "depth.positional_v0", "depth.causal_kl_greedy_v1"}
    assert all(i.kind == "DEPTH" for i in depth_impls)
    # The kind carries no algorithm; the implementations carry no dimension name
    # of their own beyond the kind they declare.
    kind = get_kind("DEPTH")
    assert kind.kind_id == "DEPTH"
    assert not hasattr(kind, "apply")
    # ... and their objectives genuinely differ.
    assert len({i.objective for i in depth_impls}) == 2


def test_the_kind_set_is_open():
    """A new structural dimension registers without touching the core."""
    before = set(registered_kinds())
    assert "MOE_EXPERT_SET" not in before
    register_kind(OperatorKindSpec("MOE_EXPERT_SET", "expert count",
                                   "how many experts survive"))
    try:
        assert "MOE_EXPERT_SET" in registered_kinds()
        # COMPOSITE_STAGE1 already proves the same thing in shipped code: it is
        # registered from an operator module, not from base.py's four.
        assert "COMPOSITE_STAGE1" in before
    finally:
        unregister_kind("MOE_EXPERT_SET")


def test_implementation_ids_are_immutable():
    """Rebinding an id to different declared semantics is refused."""

    class Sneaky(OperatorImplementation):
        impl_id = "depth.positional_v0"       # an existing, historical id
        kind = "DEPTH"
        version = 0
        modifies = frozenset({"num_hidden_layers"})
        objective = "something else entirely"  # <- the semantic change
        calibration = CalibrationNeed.FORWARD_LOGITS

        def plan(self, spec, target, adapter, config=None):  # pragma: no cover
            raise NotImplementedError

        def apply(self, ctx):  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(OperatorError, match="already bound to signature"):
        register_implementation(Sneaky())
    # The historical one is untouched.
    assert get_implementation("depth.positional_v0").objective.startswith("none")


def test_registering_the_identical_declaration_is_idempotent():
    existing = get_implementation("width.global_pca_v0")
    assert register_implementation(existing) is existing


def test_signature_hash_covers_the_declaration_not_the_docstring():
    impl = get_implementation("ffn.activation_importance_v0")
    signature = impl.signature()
    assert "description" not in signature
    assert set(signature) == {
        "impl_id", "kind", "version", "required_capabilities", "modifies",
        "preserves", "calibration", "objective", "deterministic", "requires_seed",
        "produces", "target_validation"}


def test_the_committed_ledger_matches_the_live_registry():
    """The ledger is the cross-session guarantee, not just an in-process one."""
    report = verify_ledger(repo_root=Path(__file__).resolve().parents[2])
    assert report["ok"], report
    assert report["changed"] == []
    assert report["removed"] == []


def test_dispatch_is_by_capability_not_by_name(teacher_spec, target_spec):
    from aadistill.autoinit.arch import get_adapter

    adapter = get_adapter("qwen3")
    options = applicable_implementations(adapter, teacher_spec, target_spec)
    ids = {i.impl_id for i, _ in options}
    # All five decomposed operators plus the composite apply at the root.
    assert ids == set(registered_implementations())

    class NarrowAdapter:
        """Only the capabilities a block-list depth operator needs."""

        family = "qwen3"
        adapter_version = "test"
        capabilities = frozenset({Capability.BLOCK_LIST})
        structural_fields = adapter.structural_fields

    narrow = applicable_implementations(NarrowAdapter(), teacher_spec, target_spec)
    assert {i.impl_id for i, _ in narrow} == {"depth.positional_v0"}


def test_an_operator_declaring_unmanaged_fields_is_not_applicable(target_spec):
    """Declaration is checked against the adapter, not assumed."""
    from aadistill.autoinit.arch import get_adapter

    adapter = get_adapter("qwen3")

    class Bogus(OperatorImplementation):
        impl_id = "test.bogus_v0"
        kind = "DEPTH"
        version = 0
        required_capabilities = frozenset()
        modifies = frozenset({"n_experts"})   # qwen3's adapter manages no such field

        def plan(self, spec, target, adapter, config=None):  # pragma: no cover
            raise NotImplementedError

        def apply(self, ctx):  # pragma: no cover
            raise NotImplementedError

    impl = Bogus()
    spec = ArchSpec.of("qwen3", {**dict(target_spec.fields)})
    ok, reason = impl.applicable(spec, spec, adapter)
    assert not ok and "does not manage" in reason


def test_an_operator_whose_field_is_already_at_target_is_not_offered(target_spec):
    from aadistill.autoinit.arch import get_adapter

    adapter = get_adapter("qwen3")
    at_target_depth = target_spec.replace(hidden_size=64, intermediate_size=48,
                                          num_attention_heads=2)
    options = applicable_implementations(adapter, at_target_depth, target_spec)
    kinds = {i.kind for i, _ in options}
    assert "DEPTH" not in kinds       # num_hidden_layers already matches
    assert "RESIDUAL_WIDTH" in kinds
    unregister_implementation("test.bogus_v0")
