"""What each v1 operator does, and what the framework refuses to let it do."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.metrics import (  # noqa: E402
    MetricLevel,
    MetricNamespaceError,
    OperatorLocalMetrics,
    metric_level,
)
from aadistill.autoinit.operators._common import SurgeryError  # noqa: E402
from aadistill.autoinit.operators.base import (  # noqa: E402
    CalibrationNeed,
    ContractViolation,
    OperatorContext,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
    get_implementation,
)

ADAPTER = get_adapter("qwen3")

DECOMPOSED = [
    ("depth.positional_v0", "num_hidden_layers"),
    ("depth.causal_kl_greedy_v1", "num_hidden_layers"),
    ("width.global_pca_v0", "hidden_size"),
    ("ffn.activation_importance_v0", "intermediate_size"),
    ("attention.weight_proxy_v0", "num_attention_heads"),
]


def run(impl_id, model, parent_spec, target_spec, items, profile, **config):
    impl = get_implementation(impl_id)
    ctx = OperatorContext(
        adapter=ADAPTER, model=model, parent_spec=parent_spec,
        target_spec=target_spec, profile=profile, calibration_items=items,
        seed=1234, config=config)
    return impl, impl.execute(ctx)


@pytest.mark.parametrize("impl_id,field", DECOMPOSED)
def test_operator_changes_exactly_its_declared_field(
        impl_id, field, teacher, teacher_spec, target_spec, calibration_items, profile):
    impl, outcome = run(impl_id, teacher, teacher_spec, target_spec,
                        calibration_items, profile)
    child_spec = ADAPTER.spec_of(outcome.model)
    changed = teacher_spec.diff(child_spec)
    assert changed == {field}
    assert child_spec[field] == target_spec[field]
    assert impl.modifies == {field}
    # And everything it promised to preserve really is untouched.
    for preserved in impl.preserves:
        assert child_spec[preserved] == teacher_spec[preserved], preserved


@pytest.mark.parametrize("impl_id,_field", DECOMPOSED)
def test_operator_does_not_consume_its_parent(
        impl_id, _field, teacher, teacher_spec, target_spec, calibration_items, profile):
    before = {n: p.clone() for n, p in teacher.named_parameters()}
    _, outcome = run(impl_id, teacher, teacher_spec, target_spec,
                     calibration_items, profile)
    assert outcome.model is not teacher
    assert ADAPTER.spec_of(teacher).matches(teacher_spec)
    for name, p in teacher.named_parameters():
        assert torch.equal(p, before[name]), f"parent tensor {name} was mutated"


@pytest.mark.parametrize("impl_id,_field", DECOMPOSED)
def test_operator_is_deterministic(
        impl_id, _field, teacher, teacher_spec, target_spec, calibration_items, profile):
    _, a = run(impl_id, teacher, teacher_spec, target_spec, calibration_items, profile)
    _, b = run(impl_id, teacher, teacher_spec, target_spec, calibration_items, profile)
    for (name, pa), (_, pb) in zip(a.model.named_parameters(),
                                   b.model.named_parameters()):
        assert torch.equal(pa, pb), name
    assert a.trace == b.trace


@pytest.mark.parametrize("impl_id,_field", DECOMPOSED)
def test_operator_local_metrics_stay_in_their_namespace(
        impl_id, _field, teacher, teacher_spec, target_spec, calibration_items, profile):
    _, outcome = run(impl_id, teacher, teacher_spec, target_spec,
                     calibration_items, profile)
    metrics = outcome.local_metrics
    assert metrics.impl_id == impl_id
    for key in metrics.values:
        assert metric_level(key) is MetricLevel.OPERATOR_LOCAL
    # Its reference is the parent, never the original teacher — recorded, because
    # a parent-referenced number and a root-referenced number are not comparable.
    assert metrics.reference in ("parent_state", "none")


def test_operator_local_metrics_cannot_masquerade_as_state_metrics():
    with pytest.raises(MetricNamespaceError):
        OperatorLocalMetrics(impl_id="x", objective="y", reference="parent_state",
                             values={"state.teacher_kl.equal_domain_mean": 0.1})


def test_depth_positional_reproduces_the_incumbent_map(teacher, teacher_spec,
                                                       target_spec, calibration_items,
                                                       profile):
    """The 6 -> 4 map here is the same rule that gives 36 -> 28 in production."""
    from aadistill.init.sandwich import depth_span_map

    _, outcome = run("depth.positional_v0", teacher, teacher_spec, target_spec,
                     calibration_items, profile)
    expected = [s["representative"] for s in depth_span_map(6, 4)]
    assert outcome.trace["kept_layers"] == expected
    # Blocks are carried verbatim, so the child's block k is the parent's block
    # kept[k], tensor for tensor.
    parent_blocks = ADAPTER.blocks(teacher)
    for child_block, src in zip(ADAPTER.blocks(outcome.model), expected):
        for (name, a), (_, b) in zip(child_block.named_parameters(),
                                     parent_blocks[src].named_parameters()):
            assert torch.equal(a, b), name


def test_causal_depth_search_reports_its_rounds(teacher, teacher_spec, target_spec,
                                                calibration_items, profile):
    _, outcome = run("depth.causal_kl_greedy_v1", teacher, teacher_spec, target_spec,
                     calibration_items, profile)
    rounds = outcome.artifacts["search_rounds"]
    assert len(rounds) == 2                       # 6 -> 4 removes two blocks
    assert [r["n_candidates"] for r in rounds] == [6, 5]
    # Every candidate's score is recorded, not just the winner's: a table that
    # only kept the argmin could not be re-audited.
    assert all(len(r["table"]) == r["n_candidates"] for r in rounds)
    assert outcome.trace["removal_order"] == [r["chosen"] for r in rounds]


def test_ffn_selection_keeps_a_whole_circuit(teacher, teacher_spec, target_spec,
                                             calibration_items, profile):
    _, outcome = run("ffn.activation_importance_v0", teacher, teacher_spec,
                     target_spec, calibration_items, profile)
    kept = outcome.artifacts["kept_neurons"]
    assert len(kept) == teacher_spec["num_hidden_layers"]
    for layer_idx, neurons in enumerate(kept):
        assert len(neurons) == target_spec["intermediate_size"]
        assert neurons == sorted(neurons)
        src, dst = ADAPTER.blocks(teacher)[layer_idx], ADAPTER.blocks(outcome.model)[layer_idx]
        rows = torch.tensor(neurons)
        assert torch.equal(ADAPTER.ffn(dst).gate_proj.weight,
                           ADAPTER.ffn(src).gate_proj.weight[rows])
        assert torch.equal(ADAPTER.ffn(dst).down_proj.weight,
                           ADAPTER.ffn(src).down_proj.weight[:, rows])


def test_attention_selection_preserves_gqa_grouping(teacher, teacher_spec, target_spec,
                                                    calibration_items, profile):
    _, outcome = run("attention.weight_proxy_v0", teacher, teacher_spec, target_spec,
                     calibration_items, profile)
    n_kv = teacher_spec["num_key_value_heads"]
    per_group = target_spec["num_attention_heads"] // n_kv
    for heads in outcome.artifacts["kept_heads"]:
        groups = [sum(1 for h in heads if h // (teacher_spec["num_attention_heads"] // n_kv) == g)
                  for g in range(n_kv)]
        assert groups == [per_group] * n_kv, "selection must not empty a KV group"
    # KV projections are untouched, so the RoPE basis and the cache layout hold.
    for src, dst in zip(ADAPTER.blocks(teacher), ADAPTER.blocks(outcome.model)):
        assert torch.equal(ADAPTER.attention(dst).k_proj.weight,
                           ADAPTER.attention(src).k_proj.weight)


def test_width_projection_is_orthonormal_and_folds_the_norms(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    _, outcome = run("width.global_pca_v0", teacher, teacher_spec, target_spec,
                     calibration_items, profile)
    diag = outcome.artifacts["projection_diagnostics"]
    assert diag["orthonormality_error"] < 1e-9
    assert 0.0 < diag["energy_captured_frac"] <= 1.0 + 1e-12
    for block in ADAPTER.blocks(outcome.model):
        assert torch.allclose(ADAPTER.attn_norm(block).weight,
                              torch.ones_like(ADAPTER.attn_norm(block).weight))
        assert torch.allclose(ADAPTER.ffn_norm(block).weight,
                              torch.ones_like(ADAPTER.ffn_norm(block).weight))


def test_a_child_with_an_unassigned_parameter_is_refused(teacher, teacher_spec,
                                                         target_spec):
    """The guard that stops a random tensor shipping inside a real checkpoint."""
    from aadistill.autoinit.operators._common import ChildBuilder

    builder = ChildBuilder(ADAPTER, teacher, teacher_spec.replace(num_hidden_layers=4),
                           seed=3)
    with pytest.raises(SurgeryError, match="never assigned"):
        builder.finish()


# --- contract enforcement ---------------------------------------------------


class _Liar(OperatorImplementation):
    """Declares DEPTH, also shrinks the FFN."""

    impl_id = "test.liar_v0"
    kind = "DEPTH"
    version = 0
    required_capabilities = frozenset()
    modifies = frozenset({"num_hidden_layers"})
    preserves = frozenset({"intermediate_size"})
    calibration = CalibrationNeed.NONE
    objective = "none"

    def plan(self, spec, target, adapter, config=None):
        return OperatorPlan(self.impl_id, spec.replace(num_hidden_layers=4), 0, 0)

    def apply(self, ctx):
        from aadistill.autoinit.operators._common import (
            ChildBuilder, copy_embeddings_and_final_norm, copy_module_except)

        spec = ctx.parent_spec.replace(num_hidden_layers=4, intermediate_size=24)
        builder = ChildBuilder(ctx.adapter, ctx.model, spec, seed=1)
        for dst, src in zip(ctx.adapter.blocks(builder.model),
                            ctx.adapter.blocks(ctx.model)[:4]):
            for name, p in dst.named_parameters():
                src_p = dict(src.named_parameters())[name]
                sliced = src_p
                for dim, (a, b) in enumerate(zip(p.shape, src_p.shape)):
                    if a != b:
                        sliced = sliced.narrow(dim, 0, a)
                builder.assign(p, sliced)
        copy_embeddings_and_final_norm(builder, ctx.adapter, ctx.model)
        return OperatorOutcome(
            model=builder.finish(),
            local_metrics=OperatorLocalMetrics(self.impl_id, "none", "parent_state", {}))


class _Consumer(OperatorImplementation):
    """Returns the parent itself."""

    impl_id = "test.consumer_v0"
    kind = "DEPTH"
    version = 0
    required_capabilities = frozenset()
    modifies = frozenset({"num_hidden_layers"})
    calibration = CalibrationNeed.NONE
    objective = "none"

    def plan(self, spec, target, adapter, config=None):
        return OperatorPlan(self.impl_id, spec, 0, 0)

    def apply(self, ctx):
        return OperatorOutcome(
            model=ctx.model,
            local_metrics=OperatorLocalMetrics(self.impl_id, "none", "parent_state", {}))


def test_an_operator_that_touches_an_undeclared_field_is_caught(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    ctx = OperatorContext(adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
                          target_spec=target_spec, profile=profile,
                          calibration_items=calibration_items, seed=1)
    with pytest.raises(ContractViolation, match="intermediate_size"):
        _Liar().execute(ctx)


def test_an_operator_that_returns_its_parent_is_caught(
        teacher, teacher_spec, target_spec, calibration_items, profile):
    ctx = OperatorContext(adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
                          target_spec=target_spec, profile=profile,
                          calibration_items=calibration_items, seed=1)
    with pytest.raises(ContractViolation, match="parent model itself"):
        _Consumer().execute(ctx)


def test_a_calibrated_operator_refuses_to_run_on_nothing(teacher, teacher_spec,
                                                         target_spec, profile):
    from aadistill.autoinit.operators.base import OperatorError

    impl = get_implementation("width.global_pca_v0")
    ctx = OperatorContext(adapter=ADAPTER, model=teacher, parent_spec=teacher_spec,
                          target_spec=target_spec, profile=profile,
                          calibration_items=[], seed=1)
    with pytest.raises(OperatorError, match="no calibration items"):
        impl.execute(ctx)
