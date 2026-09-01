"""`attention.activation_importance_v1` — the Phase-C1 replacement operator.

The point of C1 is that the two ATTENTION implementations differ in **exactly
one** thing: the importance signal. So these tests check both halves — that the
structural contract is identical to `weight_proxy_v0`, and that the signal is
genuinely different when the evidence says it should be.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.autoinit.adapters.qwen3 import QWEN3_ADAPTER  # noqa: E402
from aadistill.autoinit.arch import ArchSpec  # noqa: E402
from aadistill.autoinit.calibration import NO_CALIBRATION  # noqa: E402
from aadistill.autoinit.operators import get_implementation  # noqa: E402
from aadistill.autoinit.operators import attention_activation  # noqa: E402
from aadistill.autoinit.operators.attention_activation import (  # noqa: E402
    ATTENTION_STATS_SPEC,
    select_q_heads_by_score,
)
from aadistill.autoinit.operators.base import OperatorContext  # noqa: E402
from aadistill.autoinit.stats import DEFAULT_STATS_SPEC  # noqa: E402
from aadistill.init.attention_stats import (  # noqa: E402
    AttentionHeadStatsCollector,
    head_write_energy,
)

from conftest import build_tiny_model  # noqa: E402

#: 4 query heads over 2 KV groups, so each group keeps 2 of 2 -> use 8/2 to get a
#: real 4-of-2 choice per group. Mirrors the C1 geometry's 4-per-group -> keep 2.
GEOMETRY = dict(hidden_size=32, num_hidden_layers=3, intermediate_size=48,
                num_attention_heads=8, num_key_value_heads=2, head_dim=8,
                vocab_size=128, tie_word_embeddings=True)
KEEP_HEADS = 4          # 4 of 8 query heads: 2 of 4 within each of 2 GQA groups


def items(n=4, seq=16, vocab=128, seed=11):
    g = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randint(0, vocab, (1, seq), generator=g)}
            for _ in range(n)]


def context(model, parent_spec, target_spec, calib):
    return OperatorContext(
        adapter=QWEN3_ADAPTER, model=model, parent_spec=parent_spec,
        target_spec=target_spec, profile=NO_CALIBRATION, calibration_items=calib,
        seed=0, device="cpu", config={"n_calibration_items": len(calib)})


@pytest.fixture(autouse=True)
def registered():
    """Registration is explicit and must not leak: an unrestricted `BeamSearch`
    enumerates the whole registry, so a stray registration changes other
    searches. Register for the test, leave afterwards."""
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


@pytest.fixture
def geo():
    return ArchSpec.of("qwen3", GEOMETRY)


@pytest.fixture
def target(geo):
    return geo.replace(num_attention_heads=KEEP_HEADS)


def run(model, geo, target, calib):
    impl = get_implementation("attention.activation_importance_v1")
    return impl.execute(context(model, geo, target, calib))


# --- structural contract ----------------------------------------------------

def test_it_changes_only_the_query_head_count(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    after = QWEN3_ADAPTER.spec_of(out.model)
    assert geo.diff(after) == frozenset({"num_attention_heads"})
    assert after["num_attention_heads"] == KEEP_HEADS


def test_gqa_grouping_kv_heads_head_dim_and_rope_are_preserved(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    a, b = model.config, out.model.config
    assert b.num_key_value_heads == a.num_key_value_heads == 2
    assert b.head_dim == a.head_dim == 8
    assert b.num_attention_heads % b.num_key_value_heads == 0
    # RoPE basis untouched: same theta, same scaling, same max positions.
    assert b.rope_parameters == a.rope_parameters
    assert b.max_position_embeddings == a.max_position_embeddings
    # KV projections are copied verbatim, never sliced.
    for src, dst in zip(QWEN3_ADAPTER.blocks(model), QWEN3_ADAPTER.blocks(out.model)):
        s, d = QWEN3_ADAPTER.attention(src), QWEN3_ADAPTER.attention(dst)
        assert torch.equal(s.k_proj.weight, d.k_proj.weight)
        assert torch.equal(s.v_proj.weight, d.v_proj.weight)


def test_exactly_two_of_four_heads_are_kept_in_each_gqa_group(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    for kept in out.artifacts["kept_heads"]:
        assert len(kept) == KEEP_HEADS
        assert kept == sorted(kept)
        for g in range(2):                       # per GQA group of 4 query heads
            in_group = [h for h in kept if g * 4 <= h < (g + 1) * 4]
            assert len(in_group) == 2, f"group {g} kept {in_group}, expected 2 of 4"


def test_selected_rows_and_columns_are_copied_verbatim(geo, target):
    model = build_tiny_model(GEOMETRY)
    out = run(model, geo, target, items())
    hd = GEOMETRY["head_dim"]
    for layer, (src, dst) in enumerate(zip(QWEN3_ADAPTER.blocks(model),
                                           QWEN3_ADAPTER.blocks(out.model))):
        kept = out.artifacts["kept_heads"][layer]
        s, d = QWEN3_ADAPTER.attention(src), QWEN3_ADAPTER.attention(dst)
        rows = [h * hd + i for h in kept for i in range(hd)]
        assert torch.equal(d.q_proj.weight, s.q_proj.weight[rows])
        assert torch.equal(d.o_proj.weight, s.o_proj.weight[:, rows])


# --- the selection rule -----------------------------------------------------

def test_tie_breaking_is_by_ascending_head_index_and_is_stated_not_inherited():
    scores = torch.ones(8)                       # every head identical
    assert select_q_heads_by_score(scores, 8, 2, 4) == [0, 1, 4, 5]
    # A single winner in the second group must still pull its group's other slot
    # from the lowest-indexed tied head.
    scores = torch.tensor([1., 1., 1., 1., 1., 1., 9., 1.])
    assert select_q_heads_by_score(scores, 8, 2, 4) == [0, 1, 4, 6]


def test_it_refuses_a_retention_that_would_break_gqa_grouping():
    scores = torch.arange(8, dtype=torch.float64)
    with pytest.raises(ValueError):
        select_q_heads_by_score(scores, 8, 3, 4)      # 8 not divisible by 3
    with pytest.raises(ValueError):
        select_q_heads_by_score(scores, 8, 2, 5)      # 5 not divisible by 2


# --- the signal actually differs from the weight proxy ----------------------

def test_activation_evidence_overrides_the_weight_proxy_when_they_disagree(geo, target):
    """The whole scientific claim of C1, made falsifiable on a toy model.

    The two signals differ in *where* they look. `weight_proxy_v0` uses
    ||W_o,h||_F, which weights every input coordinate equally; the activation
    score uses only the coordinates the head's output actually occupies. So the
    honest disagreement is a head whose large output weights sit in directions
    its activation never visits.

    Construction: KV head 0 is made to emit only coordinate 0, so every query
    head in GQA group 0 has an output confined to that one coordinate. Head 0
    then gets enormous `W_o` columns *everywhere except* coordinate 0 — a huge
    Frobenius norm that multiplies nothing — while head 1 gets a modest column
    exactly on coordinate 0. The proxy must prefer head 0; the activation
    operator must drop it.
    """
    model = build_tiny_model(GEOMETRY, seed=3)
    hd = GEOMETRY["head_dim"]
    block = QWEN3_ADAPTER.blocks(model)[0]
    attn = QWEN3_ADAPTER.attention(block)
    with torch.no_grad():
        attn.v_proj.weight[1:hd, :] = 0.0          # kv head 0 emits coord 0 only
        attn.o_proj.weight[:, 0:hd] = 0.0
        attn.o_proj.weight[:, 1:hd] = 20.0         # head 0: norm without write
        attn.o_proj.weight[:, hd:2 * hd] = 0.0
        attn.o_proj.weight[:, hd] = 3.0            # head 1: write without norm
        attn.q_proj.weight[0:hd, :] *= 10.0        # and push the proxy harder

    calib = items()
    proxy = get_implementation("attention.weight_proxy_v0").execute(
        context(model, geo, target, ()))
    activation = run(model, geo, target, calib)

    assert 0 in proxy.artifacts["kept_heads"][0], (
        "the weight proxy should be captured by the inflated W_q/W_o norms")
    assert 0 not in activation.artifacts["kept_heads"][0], (
        "the activation operator followed the weight proxy instead of the "
        "realized write energy")


def test_the_score_is_the_exact_mean_squared_residual_write():
    """`mean_t ||W_o,h a_h(t)||^2`, checked against a direct computation.

    The streamed second moment is a sufficient statistic, not an approximation,
    so this must agree to float tolerance rather than merely correlate.
    """
    model = build_tiny_model(GEOMETRY, seed=5)
    n_q, n_kv, hd = QWEN3_ADAPTER.head_groups(ArchSpec.of("qwen3", GEOMETRY))
    calib = items(n=3, seq=12)

    captured: list[torch.Tensor] = []
    o_proj = QWEN3_ADAPTER.attention(QWEN3_ADAPTER.blocks(model)[0]).o_proj
    h = o_proj.register_forward_pre_hook(lambda _m, a: captured.append(a[0].detach()))
    collector = AttentionHeadStatsCollector(model, num_heads=n_q, head_dim=hd)
    try:
        for it in calib:
            collector.process(it["input_ids"])
    finally:
        collector.close()
        h.remove()
    state = collector.state()

    got = head_write_energy(state, 0, o_proj.weight, n_q, hd)

    a = torch.cat([c.reshape(-1, c.shape[-1]) for c in captured]).to(torch.float64)
    w = o_proj.weight.to(torch.float64)
    want = torch.stack([
        (a[:, i * hd:(i + 1) * hd] @ w[:, i * hd:(i + 1) * hd].T).pow(2).sum(-1).mean()
        for i in range(n_q)])
    assert torch.allclose(got, want, rtol=1e-9, atol=1e-12)


# --- statistics identity ----------------------------------------------------

#: The signature hashes the Phase-A/B search ran under. Adding an ATTENTION
#: implementation must not move any of them: every frozen `impl_signature_hash`
#: in the Phase-A/B state journal is one of these, so a change would invalidate
#: the identity of evidence that is closed and cannot be re-measured.
FROZEN_SIGNATURES = {
    "depth.positional_v0": "d97fb631c37f",
    "depth.causal_kl_greedy_v1": "810660469462",
    "width.global_pca_v0": "acb56886de90",
    "ffn.activation_importance_v0": "accb89629456",
    "attention.weight_proxy_v0": "479ff7b3ef15",
    "composite.stage1_sandwich_v0": "7ddeb8cfa81f",
}


@pytest.mark.parametrize("impl_id,prefix", sorted(FROZEN_SIGNATURES.items()))
def test_the_phase_a_b_operator_identities_did_not_move(impl_id, prefix):
    assert get_implementation(impl_id).signature_hash.startswith(prefix)


def test_the_new_operator_is_not_in_the_frozen_search_library():
    """Absent from `V1_IMPLEMENTATIONS`, so it cannot enter the frozen library."""
    from aadistill.autoinit.operators import V1_IMPLEMENTATIONS

    ids = [i.impl_id for i in V1_IMPLEMENTATIONS]
    assert "attention.activation_importance_v1" not in ids
    assert "attention.weight_proxy_v0" in ids
    assert get_implementation("attention.activation_importance_v1") is not None


def test_importing_the_module_does_not_register_anything():
    """`BeamSearch._allowed_impl_ids` falls back to the ENTIRE registry when
    `allowed_impls` is None, so registering at import would add a calibrated
    ATTENTION branch to every search in the process. Staying out of
    `V1_IMPLEMENTATIONS` is not enough on its own — the full suite caught exactly
    this. Import must therefore be inert."""
    from aadistill.autoinit.operators import registered_implementations

    attention_activation.unregister()
    try:
        import importlib

        importlib.reload(attention_activation)
        assert "attention.activation_importance_v1" not in registered_implementations()
    finally:
        attention_activation.register(replace=True)


def test_the_attention_stats_spec_cannot_collide_with_the_ffn_width_cache():
    assert ATTENTION_STATS_SPEC.spec_hash != DEFAULT_STATS_SPEC.spec_hash
    assert ATTENTION_STATS_SPEC.spec_id != DEFAULT_STATS_SPEC.spec_id
    assert set(ATTENTION_STATS_SPEC.quantities).isdisjoint(DEFAULT_STATS_SPEC.quantities)


def test_the_collector_refuses_to_report_zero_tokens():
    model = build_tiny_model(GEOMETRY)
    n_q, _, hd = QWEN3_ADAPTER.head_groups(ArchSpec.of("qwen3", GEOMETRY))
    c = AttentionHeadStatsCollector(model, num_heads=n_q, head_dim=hd)
    c.close()
    with pytest.raises(ValueError, match="no tokens"):
        c.state()


def test_it_declares_calibration_and_refuses_to_run_without_items(geo, target):
    impl = get_implementation("attention.activation_importance_v1")
    assert impl.calibration.value == "activation_stats"
    model = build_tiny_model(GEOMETRY)
    with pytest.raises(Exception):
        impl.execute(context(model, geo, target, ()))


def test_it_is_deterministic_across_repeated_application(geo, target):
    calib = items()
    a = run(build_tiny_model(GEOMETRY), geo, target, calib)
    b = run(build_tiny_model(GEOMETRY), geo, target, calib)
    assert a.artifacts["kept_heads"] == b.artifacts["kept_heads"]
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.equal(pa, pb)
