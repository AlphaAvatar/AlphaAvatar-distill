"""The treatment path knows no family module tree, and works across geometries.

Two claims, and the second is only interesting because of the first.

**The boundary.** `AttentionHeadStatsCollector` used to walk `model.model.layers`
and read `layer.self_attn.o_proj`, and the operator used to reach through
`adapter.attention(block).q_proj`. Those are Qwen/Llama member names living in
files that are supposed to be family-agnostic — a second place to update when an
architecture is added, and a silently wrong answer when only one of them is.
Everything now resolves through `ArchitectureAdapter`: `blocks()`,
`stream_out_projections()["attn_out"]`, `stream_in_projections()["q"]`,
`head_groups()`. `test_the_collector_hooks_modules_it_is_handed` proves it by
handing the collector a model whose attention modules are *not* reachable by any
Qwen path at all.

**The geometry matrix.** Selection is per-GQA-group top-k, so the arithmetic that
decides how many heads survive a group is where an off-by-one would hide. It is
exercised across layer counts, hidden sizes, head dims and GQA ratios rather than
at the single 4Q/2KV shape the earlier tests used — and, just as importantly, the
two shapes that CANNOT be served are required to say so instead of approximating:
a non-divisible target, and any MHA reduction, which cannot preserve the KV heads
this operator is contracted to leave alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.initialization.specs.arch import ArchSpec, get_adapter  # noqa: E402
from aadistill.initialization.device import model_device  # noqa: E402
from aadistill.initialization.operators import attention_activation  # noqa: E402
from aadistill.initialization.operators.attention_activation import (  # noqa: E402
    ATTN_OUT_ROLE,
    QUERY_ROLE,
    attention_out_projection,
    query_projection,
    select_q_heads_by_score,
)
from aadistill.initialization.operators.base import (  # noqa: E402
    OperatorContext,
    get_implementation,
)
from aadistill.initialization.statistics.attention import AttentionHeadStatsCollector  # noqa: E402

ADAPTER = get_adapter("qwen3")
IMPL = "attention.activation_importance_v1"


@pytest.fixture(autouse=True)
def registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


def _ctx(model, geo, target, items, profile):
    return OperatorContext(
        adapter=ADAPTER, model=model,
        parent_spec=ArchSpec.of("qwen3", geo),
        target_spec=ArchSpec.of("qwen3", target),
        profile=profile, calibration_items=items, seed=1234,
        device=str(model_device(model)))


def _geo(**over):
    base = dict(hidden_size=32, num_hidden_layers=3, intermediate_size=48,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                vocab_size=128, tie_word_embeddings=True)
    base.update(over)
    return base


# --- the collector inspects nothing -----------------------------------------

class _NotAQwenModel(torch.nn.Module):
    """A model whose attention lives nowhere a Qwen path would look.

    No `.model`, no `.layers`, no `.self_attn`, no `.o_proj`. If the collector
    still walks a family tree, it cannot construct against this at all; if it
    hooks what it is handed, the statistics come out correct.
    """

    def __init__(self, n_blocks: int, n_heads: int, head_dim: int, hidden: int):
        super().__init__()
        width = n_heads * head_dim
        self.stack = torch.nn.ModuleList(
            [torch.nn.Linear(width, hidden, bias=False) for _ in range(n_blocks)])
        self.n_heads, self.head_dim, self.width = n_heads, head_dim, width

    def forward(self, input_ids):
        x = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.width)
        x = x + input_ids.unsqueeze(-1).to(torch.float32) * 0.01
        for block in self.stack:
            block(x)
        return x

    #: What the ADAPTER would provide. Named differently on purpose — the point
    #: is that the collector takes a sequence, not that it finds this.
    def attention_writers(self):
        return list(self.stack)


def test_the_collector_hooks_modules_it_is_handed(monkeypatch):
    """The boundary proof. No Qwen attribute exists on this model."""
    torch.manual_seed(3)
    model = _NotAQwenModel(n_blocks=2, n_heads=3, head_dim=4, hidden=6)
    assert not hasattr(model, "model")
    assert not any(hasattr(b, "self_attn") for b in model.stack)

    collector = AttentionHeadStatsCollector(
        model, model.attention_writers(), num_heads=3, head_dim=4)
    try:
        collector.process(torch.arange(5).reshape(1, 5))
    finally:
        collector.close()
    state = collector.state()

    assert state["attn_head_sqsum"].shape == (2, 3, 4, 4)
    assert int(state["attn_token_count"]) == 5
    assert state["attn_head_sqsum"].dtype is torch.float64


def test_the_collector_refuses_an_empty_projection_sequence():
    model = _NotAQwenModel(n_blocks=1, n_heads=2, head_dim=2, hidden=4)
    with pytest.raises(ValueError, match="no attention-output projections"):
        AttentionHeadStatsCollector(model, [], num_heads=2, head_dim=2)


def test_the_collector_refuses_something_it_cannot_hook():
    model = _NotAQwenModel(n_blocks=1, n_heads=2, head_dim=2, hidden=4)
    with pytest.raises(ValueError, match="cannot be hooked"):
        AttentionHeadStatsCollector(model, ["not a module"], num_heads=2, head_dim=2)


def test_mutation_the_collector_source_names_no_family_attribute():
    """Reintroducing the walk must be visible, not merely unlikely."""
    import aadistill.initialization.statistics.attention as AS

    src = Path(AS.__file__).read_text()
    code = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[-1]          # past the module docstring
    for forbidden in (".self_attn", "model.layers", 'getattr(model, "model"'):
        assert forbidden not in body, (
            f"{forbidden!r} is back in attention_stats.py; family module-tree "
            "knowledge belongs to the adapter")


def test_the_operator_resolves_projections_through_role_maps(teacher,
                                                             calibration_items,
                                                             profile):
    """Both roles come from the adapter, and both are the modules it names."""
    block = ADAPTER.blocks(teacher)[0]
    assert attention_out_projection(ADAPTER, block) is \
        ADAPTER.stream_out_projections(block)[ATTN_OUT_ROLE]
    assert query_projection(ADAPTER, block) is \
        ADAPTER.stream_in_projections(block)[QUERY_ROLE][0]


def test_a_family_without_the_roles_fails_closed():
    """An adapter that cannot name these roles is refused, not guessed around."""
    from aadistill.initialization.specs.arch import UnsupportedCapability

    class _RolelessAdapter:
        family = "roleless"

        def stream_out_projections(self, block):
            return {"ffn_out": object()}

        def stream_in_projections(self, block):
            return {"gate": (object(), object())}

    with pytest.raises(UnsupportedCapability, match="attn_out"):
        attention_out_projection(_RolelessAdapter(), object())
    with pytest.raises(UnsupportedCapability, match="'q'"):
        query_projection(_RolelessAdapter(), object())


# --- the geometry matrix ----------------------------------------------------

#: (label, n_layers, hidden, n_q, n_kv, head_dim, keep_q)
SUPPORTED = [
    ("gqa 4Q/2KV -> 2Q",      3, 32,  4, 2,  8, 2),
    ("gqa 8Q/2KV -> 4Q",      2, 64,  8, 2,  8, 4),
    ("gqa 12Q/3KV -> 6Q",     2, 48, 12, 3,  4, 6),
    ("mqa 8Q/1KV -> 3Q",      2, 32,  8, 1,  4, 3),
    ("mqa 6Q/1KV -> 1Q",      1, 24,  6, 1, 16, 1),
    ("head_dim 16, 8Q/4KV",   2, 64,  8, 4, 16, 4),
    ("deep, 6 layers",        6, 32,  4, 2,  8, 2),
    ("no reduction, 4Q -> 4Q", 2, 32, 4, 2,  8, 4),
]


@pytest.mark.parametrize("label,n_layers,hidden,n_q,n_kv,hd,keep_q", SUPPORTED,
                         ids=[s[0] for s in SUPPORTED])
def test_supported_geometries_execute_and_preserve_the_contract(
        label, n_layers, hidden, n_q, n_kv, hd, keep_q,
        calibration_items, profile):
    from conftest import build_tiny_model

    geo = _geo(num_hidden_layers=n_layers, hidden_size=hidden,
               num_attention_heads=n_q, num_key_value_heads=n_kv, head_dim=hd,
               intermediate_size=max(16, hidden + 16))
    target = dict(geo, num_attention_heads=keep_q)
    model = build_tiny_model(geo)

    outcome = get_implementation(IMPL).execute(
        _ctx(model, geo, target, calibration_items, profile))
    child = ADAPTER.spec_of(outcome.model)

    assert child["num_attention_heads"] == keep_q
    assert child["num_key_value_heads"] == n_kv, "KV heads must be preserved"
    assert child["head_dim"] == hd
    assert child["num_hidden_layers"] == n_layers
    assert child["hidden_size"] == hidden

    kept = outcome.artifacts["kept_heads"]
    assert len(kept) == n_layers
    per_group = keep_q // n_kv
    for layer_kept in kept:
        assert len(layer_kept) == keep_q
        assert layer_kept == sorted(layer_kept), "kept heads must be ascending"
        for g in range(n_kv):
            lo, hi = g * (n_q // n_kv), (g + 1) * (n_q // n_kv)
            assert sum(lo <= h < hi for h in layer_kept) == per_group

    # The child's weights really are the parent's selected slices.
    for idx, (src, dst) in enumerate(zip(ADAPTER.blocks(model),
                                         ADAPTER.blocks(outcome.model))):
        rows = [h * hd + i for h in kept[idx] for i in range(hd)]
        assert torch.equal(query_projection(ADAPTER, dst).weight,
                           query_projection(ADAPTER, src).weight[rows])
        assert torch.equal(attention_out_projection(ADAPTER, dst).weight,
                           attention_out_projection(ADAPTER, src).weight[:, rows])


#: (label, n_q, n_kv, keep_q, expected message fragment)
UNSUPPORTED = [
    ("mha 8Q/8KV -> 4Q",  8, 8, 4, "multi-head attention"),
    ("mha 4Q/4KV -> 2Q",  4, 4, 2, "multi-head attention"),
    ("non-divisible 8Q/3KV -> 4Q", 9, 3, 4, "not divisible"),
    ("non-divisible 12Q/4KV -> 6Q", 12, 4, 6, "not divisible"),
    ("adding heads 4Q -> 8Q", 4, 2, 8, "cannot add query heads"),
]


@pytest.mark.parametrize("label,n_q,n_kv,keep_q,fragment", UNSUPPORTED,
                         ids=[u[0] for u in UNSUPPORTED])
def test_unsupported_geometries_refuse_explicitly(label, n_q, n_kv, keep_q,
                                                  fragment):
    """Refused with a reason, never approximated.

    The MHA cases are the ones worth naming. Divisibility alone would already
    reject them, with a message about arithmetic that hides the actual
    obstruction: under MHA each query head owns its KV head, so there is no way
    to drop query heads while preserving the KV heads this operator is
    contracted to leave unchanged.
    """
    geo = _geo(num_attention_heads=n_q, num_key_value_heads=n_kv,
               hidden_size=max(32, n_q * 8))
    target = dict(geo, num_attention_heads=keep_q)
    impl = get_implementation(IMPL)

    ok, reason = impl.applicable(ArchSpec.of("qwen3", geo),
                                 ArchSpec.of("qwen3", target), ADAPTER)
    assert not ok, f"{label} was accepted"
    assert fragment in reason, f"{label}: {reason!r} does not say {fragment!r}"


def test_an_mha_refusal_names_kv_preservation_not_just_arithmetic():
    """The specific improvement, pinned so it cannot regress to a modulo test."""
    geo = _geo(num_attention_heads=8, num_key_value_heads=8, hidden_size=64)
    target = dict(geo, num_attention_heads=4)
    ok, reason = get_implementation(IMPL).applicable(
        ArchSpec.of("qwen3", geo), ArchSpec.of("qwen3", target), ADAPTER)
    assert not ok
    assert "KV heads" in reason and "preserve" in reason
    assert "different operator" in reason, (
        "a refusal should say what would serve the request, not only that this "
        "does not")


def test_selection_is_per_group_across_the_matrix():
    """The pure selector, over the same ratios, independent of any model."""
    for n_q, n_kv, keep_q in [(4, 2, 2), (8, 2, 4), (12, 3, 6), (8, 1, 3),
                              (6, 1, 1), (8, 4, 4)]:
        scores = torch.arange(n_q, dtype=torch.float64)
        kept = select_q_heads_by_score(scores, n_q, n_kv, keep_q)
        assert len(kept) == keep_q
        per_group = keep_q // n_kv
        for g in range(n_kv):
            lo, hi = g * (n_q // n_kv), (g + 1) * (n_q // n_kv)
            group = [h for h in kept if lo <= h < hi]
            assert len(group) == per_group
            # ascending scores => the top of each group is its tail
            assert group == sorted(range(lo, hi))[-per_group:]
