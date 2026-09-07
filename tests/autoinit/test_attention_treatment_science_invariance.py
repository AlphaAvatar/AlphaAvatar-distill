"""What the device repair was NOT allowed to move.

The repair to `attention.activation_importance_v1` changes *where* tensors live.
The claim that has to survive independent review is that it changed nothing
about *what is computed* — because the treatment formula, the impl id, the arm
specs and both path hashes are frozen, and a silent numerical drift would
invalidate an Attempt-10 comparison without failing anything.

So this module does not compare the repaired operator against a stored number
from before the repair (that would only prove the repair reproduces itself). It
recomputes the treatment from its definition:

    score_h = mean_t || W_o,h a_h(t) ||^2

directly, over the same calibration items, by capturing every `a_h(t)` — the
brute-force form the streamed second moment exists to avoid — and then requires
the streamed path, the selection, the tie-break, the GQA grouping, the child
weight slices and the local metrics to agree with it.

`test_attention_operator_device_split.py` holds the device claims; this holds
the science.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.autoinit.device import model_device  # noqa: E402
from aadistill.autoinit.operators import attention_activation  # noqa: E402
from aadistill.autoinit.operators.attention_activation import (  # noqa: E402
    ATTENTION_STATS_SPEC, select_q_heads_by_score,
)
from aadistill.autoinit.operators.base import (  # noqa: E402
    OperatorContext, get_implementation,
)
from aadistill.init.attention_stats import (  # noqa: E402
    AttentionHeadStatsCollector, head_write_energy,
)

ADAPTER = get_adapter("qwen3")
IMPL = "attention.activation_importance_v1"


@pytest.fixture(autouse=True)
def registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


@pytest.fixture
def geo():
    from conftest import TEACHER_GEOMETRY
    return dict(TEACHER_GEOMETRY)


@pytest.fixture
def target(geo):
    return dict(geo, num_attention_heads=2)


def _ctx(teacher, geo, target, items, profile):
    return OperatorContext(
        adapter=ADAPTER, model=teacher,
        parent_spec=ArchSpec.of("qwen3", geo),
        target_spec=ArchSpec.of("qwen3", target),
        profile=profile, calibration_items=items, seed=1234,
        device=str(model_device(teacher)))


def _brute_force_scores(model, items, n_heads, head_dim):
    """`mean_t ||W_o,h a_h(t)||^2` with every `a_h(t)` retained.

    The definition, computed the expensive way: hook each `o_proj`, keep the
    whole activation, project it head by head and average the squared norms.
    This is what the streamed second moment claims to equal exactly.
    """
    layers = model.model.layers
    captured: list[list[torch.Tensor]] = [[] for _ in layers]
    hooks = []

    def make(idx):
        def hook(_m, args):
            captured[idx].append(args[0].detach().reshape(-1, args[0].shape[-1])
                                 .to(torch.float64))
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make(i)))
    try:
        with torch.no_grad():
            for item in items:
                ids = item["input_ids"]
                model(ids.unsqueeze(0) if ids.dim() == 1 else ids)
    finally:
        for h in hooks:
            h.remove()

    out = []
    for i, layer in enumerate(layers):
        a = torch.cat(captured[i], dim=0)                      # (tokens, h*d)
        w = layer.self_attn.o_proj.weight.to(torch.float64)    # (hidden, h*d)
        per_head = []
        for h in range(n_heads):
            ah = a[:, h * head_dim:(h + 1) * head_dim]         # (tokens, d)
            z = ah @ w[:, h * head_dim:(h + 1) * head_dim].T   # (tokens, hidden)
            per_head.append((z * z).sum(dim=1).mean().detach())
        out.append(torch.stack(per_head))
    return out


# --- C1. the streamed statistic is the definition, not an approximation ------

def test_C_the_streamed_second_moment_equals_the_direct_write_energy(
        teacher, geo, calibration_items):
    n_heads, head_dim = geo["num_attention_heads"], geo["head_dim"]
    direct = _brute_force_scores(teacher, calibration_items, n_heads, head_dim)

    collector = AttentionHeadStatsCollector(teacher, num_heads=n_heads,
                                            head_dim=head_dim)
    try:
        for item in calibration_items:
            collector.process(item["input_ids"])
    finally:
        collector.close()
    stats = collector.state()

    for idx, layer in enumerate(teacher.model.layers):
        streamed = head_write_energy(stats, idx, layer.self_attn.o_proj.weight,
                                     n_heads, head_dim)
        assert torch.allclose(streamed, direct[idx], rtol=1e-10, atol=1e-12), (
            f"layer {idx}: streamed {streamed} != direct {direct[idx]}")


def test_C2_the_accumulation_dtype_is_float64_end_to_end(teacher, geo,
                                                         calibration_items):
    """A repair that dropped to float32 would still pass a loose comparison."""
    assert ATTENTION_STATS_SPEC.accumulation_dtype == "float64"
    c = AttentionHeadStatsCollector(teacher, num_heads=geo["num_attention_heads"],
                                    head_dim=geo["head_dim"])
    try:
        c.process(calibration_items[0]["input_ids"])
    finally:
        c.close()
    st = c.state()
    assert st["attn_head_sqsum"].dtype is torch.float64
    scores = head_write_energy(st, 0, teacher.model.layers[0].self_attn.o_proj.weight,
                               geo["num_attention_heads"], geo["head_dim"])
    assert scores.dtype is torch.float64


# --- C2. the whole operator agrees with the definition ----------------------

def test_C3_selection_metrics_and_child_slices_match_the_definition(
        teacher, geo, target, calibration_items, profile):
    """Everything the operator emits, recomputed from the brute-force scores."""
    n_q, n_kv = geo["num_attention_heads"], geo["num_key_value_heads"]
    head_dim, keep_q = geo["head_dim"], target["num_attention_heads"]

    direct = _brute_force_scores(teacher, calibration_items, n_q, head_dim)
    expected_kept = [select_q_heads_by_score(s, n_q, n_kv, keep_q) for s in direct]

    outcome = get_implementation(IMPL).execute(
        _ctx(teacher, geo, target, calibration_items, profile))

    assert outcome.artifacts["kept_heads"] == expected_kept, (
        "the operator selected different heads than the definition does")

    # GQA grouping: KV heads untouched, retention spread evenly across groups.
    child_spec = ADAPTER.spec_of(outcome.model)
    assert child_spec["num_key_value_heads"] == n_kv
    assert child_spec["num_attention_heads"] == keep_q
    per_group = keep_q // n_kv
    for kept in expected_kept:
        for g in range(n_kv):
            lo, hi = g * (n_q // n_kv), (g + 1) * (n_q // n_kv)
            assert sum(lo <= h < hi for h in kept) == per_group

    # Local metrics: retained share of realized write energy, per the definition.
    exp_share = [float(s[k].sum() / s.sum()) for s, k in zip(direct, expected_kept)]
    vals = outcome.local_metrics.values
    assert vals["op.attention.retained_write_energy_mean"] == pytest.approx(
        sum(exp_share) / len(exp_share), rel=1e-9)
    assert vals["op.attention.retained_write_energy_min"] == pytest.approx(
        min(exp_share), rel=1e-9)
    assert outcome.local_metrics.detail["per_layer_retained_share"] == pytest.approx(
        exp_share, rel=1e-9)

    # Child weights are the parent's rows/columns for the kept heads, exactly.
    for idx, (src, dst) in enumerate(zip(ADAPTER.blocks(teacher),
                                         ADAPTER.blocks(outcome.model))):
        s_attn, d_attn = ADAPTER.attention(src), ADAPTER.attention(dst)
        rows = [h * head_dim + i for h in expected_kept[idx] for i in range(head_dim)]
        assert torch.equal(d_attn.q_proj.weight, s_attn.q_proj.weight[rows])
        assert torch.equal(d_attn.o_proj.weight, s_attn.o_proj.weight[:, rows])
        assert torch.equal(d_attn.k_proj.weight, s_attn.k_proj.weight)
        assert torch.equal(d_attn.v_proj.weight, s_attn.v_proj.weight)


def test_C4_the_trace_still_reports_the_frozen_treatment_identity(
        teacher, geo, target, calibration_items, profile):
    """The fields an Attempt-10 record would be read back against."""
    outcome = get_implementation(IMPL).execute(
        _ctx(teacher, geo, target, calibration_items, profile))
    tr = outcome.trace
    assert tr["score"] == "mean_t ||W_o,h a_h(t)||^2"
    assert tr["source"] == "activation_write_energy_per_group_topk"
    assert tr["stats_spec"] == ATTENTION_STATS_SPEC.spec_hash
    assert tr["q_heads"] == [geo["num_attention_heads"], target["num_attention_heads"]]
    assert tr["kv_heads"] == geo["num_key_value_heads"]
    assert isinstance(tr["calibration_tokens"], int) and tr["calibration_tokens"] > 0


def test_C5_the_impl_identity_the_arms_name_is_unmoved():
    impl = get_implementation(IMPL)
    assert impl.impl_id == "attention.activation_importance_v1"
    assert impl.version == 1
    assert impl.kind == "ATTENTION"
    assert impl.objective == "retained share of realized attention-output write energy"
    assert impl.modifies == frozenset({"num_attention_heads"})
    assert impl.deterministic and not impl.requires_seed


def test_C6_the_tie_break_is_ascending_index_under_exactly_equal_scores():
    """Stated, not inherited: the case a float comparison cannot separate."""
    scores = torch.ones(8, dtype=torch.float64)
    assert select_q_heads_by_score(scores, 8, 2, 4) == [0, 1, 4, 5]
    mixed = torch.tensor([1.0, 5.0, 5.0, 1.0, 9.0, 2.0, 9.0, 2.0], dtype=torch.float64)
    assert select_q_heads_by_score(mixed, 8, 2, 4) == [1, 2, 4, 6]


def test_C7_two_runs_of_the_repaired_operator_select_identically(
        teacher, geo, target, calibration_items, profile):
    """Determinism, after the repair introduced a transfer into the path."""
    first = get_implementation(IMPL).execute(
        _ctx(teacher, geo, target, calibration_items, profile))
    second = get_implementation(IMPL).execute(
        _ctx(teacher, geo, target, calibration_items, profile))
    assert first.artifacts["kept_heads"] == second.artifacts["kept_heads"]
    assert (first.local_metrics.detail["per_layer_retained_share"]
            == second.local_metrics.detail["per_layer_retained_share"])
