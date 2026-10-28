"""Full-width selection, per-layer bases and the projected materialization.

The two exactness anchors:

* the projected student with ``r = d`` and ``P_l = I`` computes exactly ``W0``;
* when every sub-layer input really lies in an ``r``-dimensional subspace, the
  bases the collector finds span it and the rank-``r`` student computes exactly
  ``W0`` again -- the projection then loses nothing, by construction.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from aadistill.initialization.adapters import build_registry
from aadistill.initialization.transforms.projected_residual import (
    INDEPENDENT_BASES,
    STATISTICS,
    SelectionError,
    SublayerStatsCollector,
    independent_bases,
    materialize_projected,
    projection_basis,
    select_full_width,
    selection_problems,
)

ADAPTER = build_registry(["qwen3"]).get("qwen3")
TEACHER = dict(vocab_size=61, hidden_size=32, num_hidden_layers=4, intermediate_size=24,
               num_attention_heads=8, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
               max_position_embeddings=128, rope_theta=10000.0)


def teacher(seed=0):
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(Qwen3Config(**TEACHER)).double().eval()
    with torch.no_grad():
        for b in m.model.layers:            # non-trivial norms, so a norm copy is visible
            b.input_layernorm.weight.uniform_(0.5, 1.5)
            b.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    return m


SEL = {"kept_layers": [0, 2, 3],
       "kept_heads": [[0, 1, 4, 7], [1, 2, 5, 6], [0, 3, 4, 5]],
       "kept_neurons": [list(range(0, 24, 2)), list(range(12)), list(range(12, 24))]}


def w0_spec():
    spec = ADAPTER.spec_of(teacher())
    return spec.replace(num_hidden_layers=3, num_attention_heads=4, intermediate_size=12)


def test_selection_problems_name_each_defect():
    parent, target = TEACHER, w0_spec().as_dict()
    assert selection_problems(SEL, parent, target) == []
    bad = dict(SEL, kept_heads=[[0, 1, 2, 3], *SEL["kept_heads"][1:]])
    assert any("balanced" in p for p in selection_problems(bad, parent, target))
    assert selection_problems(dict(SEL, kept_layers=[0, 2]), parent, target)
    assert selection_problems(dict(SEL, kept_neurons=[[0] * 12] * 3), parent, target)
    with pytest.raises(SelectionError):
        select_full_width(ADAPTER, teacher(), bad, w0_spec())


def _ablated_teacher(t):
    """The teacher with unkept heads/neurons zeroed and unkept layers removed."""
    keep = torch.nn.ModuleList([t.model.layers[i] for i in SEL["kept_layers"]])
    with torch.no_grad():
        for b, heads, neurons in zip(keep, SEL["kept_heads"], SEL["kept_neurons"]):
            for h in set(range(8)) - set(heads):
                b.self_attn.o_proj.weight[:, h * 8:(h + 1) * 8] = 0
            drop = sorted(set(range(24)) - set(neurons))
            b.mlp.down_proj.weight[:, drop] = 0
    t.model.layers = keep
    t.config.num_hidden_layers = 3
    t.config.layer_types = t.config.layer_types[:3]
    return t


def test_full_width_selection_equals_deleting_heads_neurons_and_layers():
    t = teacher(1)
    w0 = select_full_width(ADAPTER, t, SEL, w0_spec())
    assert w0.config.hidden_size == 32 and w0.config.num_attention_heads == 4
    ids = torch.randint(0, 61, (2, 9))
    ref = _ablated_teacher(teacher(1))
    assert torch.allclose(w0(ids, use_cache=False).logits, ref(ids, use_cache=False).logits, atol=1e-10)


def test_identity_projection_reproduces_w0_exactly():
    w0 = select_full_width(ADAPTER, teacher(2), SEL, w0_spec())
    student = materialize_projected(ADAPTER, w0, [torch.eye(32, dtype=torch.float64)] * 3, dtype=torch.float64)
    assert student.config.internal_size == 32
    ids = torch.randint(0, 61, (2, 10))
    assert torch.allclose(student(ids, use_cache=False).logits, w0(ids, use_cache=False).logits, atol=1e-12)


def _subspace_w0(r=12):
    """A W0 whose residual stream and every sub-layer input live in the first r axes."""
    w0 = select_full_width(ADAPTER, teacher(3), SEL, w0_spec())
    with torch.no_grad():
        w0.model.embed_tokens.weight[:, r:] = 0
        for b in w0.model.layers:
            b.self_attn.o_proj.weight[r:] = 0
            b.mlp.down_proj.weight[r:] = 0
    return w0


def test_collector_moments_match_a_direct_computation_and_respect_the_mask():
    w0 = _subspace_w0()
    ids = torch.randint(0, 61, (2, 8))
    active = torch.zeros(2, 8, dtype=torch.bool)
    active[0, 2:6] = True
    active[1, :3] = True
    col = SublayerStatsCollector(ADAPTER, w0, ["attn_in", "residual_in"])
    col.process(w0, ids, active)
    col.close()
    out = w0(ids, use_cache=False, output_hidden_states=True)
    x = out.hidden_states[1]                       # input of block 1
    u = w0.model.layers[1].input_layernorm(x)
    st = col.state()
    rows = u[active]
    assert st["attn_in"][1][1] == int(active.sum())
    assert torch.allclose(st["attn_in"][1][0], rows.T @ rows, atol=1e-10)
    assert torch.allclose(st["residual_in"][1][0], x[active].T @ x[active], atol=1e-10)


def test_bases_from_the_collector_lose_nothing_when_inputs_lie_in_a_subspace():
    r = 12
    w0 = _subspace_w0(r)
    col = SublayerStatsCollector(ADAPTER, w0, STATISTICS["sublayer_inputs"])
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        col.process(w0, torch.randint(0, 61, (2, 16), generator=g))
    col.close()
    st = col.state()
    bases, diags = [], []
    for i in range(3):
        p, diag = projection_basis({role: st[role][i] for role in STATISTICS["sublayer_inputs"]}, r)
        bases.append(p)
        diags.append(diag)
        assert diag["orthonormality_error"] < 1e-10
        assert diag["attn_in_energy_captured"] > 1 - 1e-10
    student = materialize_projected(ADAPTER, w0, bases, dtype=torch.float64)
    ids = torch.randint(0, 61, (2, 11))
    assert student.config.internal_size == r
    assert torch.allclose(student(ids, use_cache=False).logits, w0(ids, use_cache=False).logits, atol=1e-9)
    # and the basis is a deterministic function of the moments
    again, _ = projection_basis({role: st[role][0] for role in STATISTICS["sublayer_inputs"]}, r)
    assert torch.equal(again, bases[0])


def test_a_truncated_basis_is_an_approximation_not_an_identity():
    w0 = select_full_width(ADAPTER, teacher(4), SEL, w0_spec())
    col = SublayerStatsCollector(ADAPTER, w0, STATISTICS["sublayer_inputs"])
    col.process(w0, torch.randint(0, 61, (4, 16)))
    col.close()
    st = col.state()
    bases = [projection_basis({k: st[k][i] for k in STATISTICS["sublayer_inputs"]}, 8)[0] for i in range(3)]
    student = materialize_projected(ADAPTER, w0, bases, dtype=torch.float64)
    ids = torch.randint(0, 61, (1, 7))
    assert not torch.allclose(student(ids, use_cache=False).logits, w0(ids, use_cache=False).logits, atol=1e-6)


def test_projection_basis_refuses_degenerate_input():
    with pytest.raises(ValueError):
        projection_basis({}, 2)
    with pytest.raises(ValueError):
        projection_basis({"attn_in": (torch.zeros(4, 4, dtype=torch.float64), 3)}, 2)
    with pytest.raises(ValueError):
        projection_basis({"attn_in": (torch.eye(4, dtype=torch.float64), 3)}, 5)


# --- independent projections: one basis per site -------------------------------------------

GATED = "learnable_silu_residual"


@pytest.mark.parametrize("activation", [None, GATED])
def test_identity_independent_bases_reproduce_w0_exactly(activation):
    w0 = select_full_width(ADAPTER, teacher(2), SEL, w0_spec())
    eye = torch.eye(32, dtype=torch.float64)
    student = materialize_projected(ADAPTER, w0, [{k: eye for k in INDEPENDENT_BASES}] * 3,
                                    dtype=torch.float64, projection_activation=activation)
    assert student.config.projection_sharing == "independent"
    assert student.config.projection_activation == activation
    ids = torch.randint(0, 61, (2, 10))
    assert torch.allclose(student(ids, use_cache=False).logits, w0(ids, use_cache=False).logits, atol=1e-12)


def test_independent_bases_lose_nothing_when_inputs_and_outputs_lie_in_a_subspace():
    from aadistill.evaluation.function_error import function_error

    r = 12
    w0 = _subspace_w0(r)
    col = SublayerStatsCollector(ADAPTER, w0, sorted(set(INDEPENDENT_BASES.values())))
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        col.process(w0, torch.randint(0, 61, (2, 16), generator=g))
    col.close()
    st = col.state()
    bases = []
    for i in range(3):
        b, diags = independent_bases({role: st[role][i] for role in INDEPENDENT_BASES.values()}, r)
        assert set(b) == set(INDEPENDENT_BASES)
        for site, role in INDEPENDENT_BASES.items():
            assert diags[site][f"{role}_energy_captured"] > 1 - 1e-10
        bases.append(b)
    student = materialize_projected(ADAPTER, w0, bases, dtype=torch.float64, projection_activation=GATED)
    ids = torch.randint(0, 61, (2, 11))
    assert torch.allclose(student(ids, use_cache=False).logits, w0(ids, use_cache=False).logits, atol=1e-9)
    rows = torch.ones(2, 10, dtype=torch.bool)
    err = function_error(w0, student, [(ids, rows)])
    assert err["positions"] == 20 and err["kl"]["max"] < 1e-9 and err["top1_agreement"] == 1.0


def test_a_truncated_independent_init_is_measured_as_an_approximation():
    from aadistill.evaluation.function_error import function_error

    w0 = select_full_width(ADAPTER, teacher(4), SEL, w0_spec())
    col = SublayerStatsCollector(ADAPTER, w0, sorted(set(INDEPENDENT_BASES.values())))
    col.process(w0, torch.randint(0, 61, (4, 16)))
    col.close()
    st = col.state()
    bases = [independent_bases({k: st[k][i] for k in INDEPENDENT_BASES.values()}, 8)[0] for i in range(3)]
    student = materialize_projected(ADAPTER, w0, bases, dtype=torch.float64, projection_activation=GATED)
    ids = torch.randint(0, 61, (2, 9))
    rows = torch.rand(2, 8) > 0.3
    err = function_error(w0, student, [(ids, rows)], chunk=3)
    # the definition, computed directly over the full vocabulary
    lr = torch.log_softmax(w0(ids, use_cache=False).logits[:, :-1][rows].float(), -1)
    lc = torch.log_softmax(student(ids, use_cache=False).logits[:, :-1][rows].float(), -1)
    kl = (lr.exp() * (lr - lc)).sum(-1)
    assert err["positions"] == int(rows.sum())
    assert err["kl"]["mean"] == pytest.approx(float(kl.mean()), rel=1e-5) and err["kl"]["mean"] > 1e-6
    with pytest.raises(ValueError, match="same form"):
        materialize_projected(ADAPTER, w0, [bases[0], torch.eye(32, dtype=torch.float64)[:, :8], bases[2]])
