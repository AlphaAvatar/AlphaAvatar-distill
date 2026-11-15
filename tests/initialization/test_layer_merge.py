"""Layer merging's arithmetic: the exact assignment, the streaming moments, folding, permuting and merging.

The pod builds two 3.2B candidates from these functions and nothing re-derives them there, so each is pinned on the
tiny real Qwen3 (randomized norms, so a folding bug cannot hide behind unit gains): the solver against brute force,
the moments against numpy, folding and permutation against the unmodified block's forward, and a merge of a block
with its own permuted copy, aligned back, against the block itself.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import aadistill.initialization  # noqa: F401,E402
from aadistill.initialization.planning import layer_merge as LM  # noqa: E402
from aadistill.initialization.specs.arch import get_adapter  # noqa: E402

ADAPTER = get_adapter("qwen3")


def test_spans_join_each_removed_block_to_the_kept_block_before_it():
    kept = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 27, 28, 29, 31, 33, 34, 35]
    spans = [s for s in LM.spans_from_kept_blocks(36, kept) if len(s) > 1]
    assert spans == [(2, 3), (10, 11), (16, 17, 18), (24, 25, 26), (29, 30), (31, 32)]
    with pytest.raises(LM.LayerMergeError):
        LM.spans_from_kept_blocks(6, [1, 3, 5])            #: block 0 has no kept block before it


def _brute(score):
    n = score.shape[0]
    return max(sum(float(score[i, p[i]]) for i in range(n)) for p in itertools.permutations(range(n)))


def test_linear_assignment_is_exact_and_its_ties_are_fixed():
    g = torch.Generator().manual_seed(0)
    for n in (2, 3, 4, 5, 6):
        for _ in range(20):
            s = torch.randn(n, n, generator=g, dtype=torch.float64)
            col = LM.linear_assignment(s)
            assert sorted(col.tolist()) == list(range(n))
            assert float(s[torch.arange(n), col].sum()) == pytest.approx(_brute(s), abs=1e-12)
    #: every assignment ties: the rule (an unassigned column first, then the lowest index) gives the identity
    assert LM.linear_assignment(torch.zeros(5, 5)).tolist() == [0, 1, 2, 3, 4]
    #: a planted permutation is recovered
    p = torch.randperm(40, generator=g)
    s = torch.eye(40, dtype=torch.float64)[:, p] + 0.01 * torch.randn(40, 40, generator=g, dtype=torch.float64)
    assert torch.equal(LM.linear_assignment(s), torch.argsort(p))
    with pytest.raises(LM.LayerMergeError):
        LM.linear_assignment(torch.tensor([[0.0, float("nan")], [0.0, 0.0]]))


def test_pearson_moments_stream_to_numpys_correlation_and_zero_variance_is_zero():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((300, 5)), rng.standard_normal((300, 4))
    y[:, 1] += 2 * x[:, 0]
    x[:, 3] = 7.0                                     #: a constant feature: its correlation is undefined
    m = LM.PearsonMoments(5, 4)
    for lo in range(0, 300, 64):
        m.add(torch.from_numpy(x[lo:lo + 64]), torch.from_numpy(y[lo:lo + 64]))
    c = m.correlation().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):   #: numpy's own NaN for the constant column
        full = np.corrcoef(np.concatenate([x, y], 1).T)[:5, 5:]
    keep = [0, 1, 2, 4]
    np.testing.assert_allclose(c[keep], full[keep], atol=1e-6)
    assert (c[3] == 0).all() and m.zero_variance()[0].tolist() == [False, False, False, True, False]


def _naive_float32_correlation(x: torch.Tensor) -> torch.Tensor:
    """The first version's arithmetic: E[xy] - E[x]E[y] with the cross products summed in float32."""
    n = x.shape[0]
    x64 = x.to(torch.float64)
    mx = x64.mean(0)
    vx = ((x64 * x64).sum(0) / n - mx * mx).clamp_min(0.0)
    cov = (x.float().T @ x.float()).to(torch.float64) / n - mx[:, None] * mx[None, :]
    return cov / (vx.sqrt()[:, None] * vx.sqrt()[None, :])


def test_pearson_keeps_a_near_constant_bf16_feature():
    """A feature exactly representable in bf16 whose spread is small against its mean (1000 + 4 b, b ~ Bernoulli(0.01),
    the next bf16 value after 1000 is 1004): its autocorrelation is 1. The uncentered float32 sums read 0.14 or 1.10;
    the centered, pairwise-merged moments match numpy's float64 to rounding, streamed in uneven chunks, one of them
    constant (a chunk with zero within-chunk variance must still merge exactly)."""
    g = torch.Generator().manual_seed(0)
    n = 8192
    b1 = (torch.rand(n, generator=g) < 0.01).double()
    b2 = (torch.rand(n, generator=g) < 0.01).double()
    b1[:600] = 0.0                                    #: the first chunk below is constant in feature 0
    cols = [1000.0 + 4.0 * b1, 1000.0 + 4.0 * b2, torch.randn(n, generator=g, dtype=torch.float64)]
    x = torch.stack(cols, 1).to(torch.bfloat16)
    assert torch.equal(x[:, :2].double(), torch.stack(cols[:2], 1))      #: bf16-exact
    ref = np.corrcoef(x.double().numpy().T)
    naive = _naive_float32_correlation(x)
    assert abs(float(naive[0, 0]) - 1.0) > 0.05       #: the counterexample bites the old arithmetic (1.10 here)
    #: the merge arithmetic with a float64 product: numpy to rounding; with the pod's float32 product: the float32
    #: accumulation of centered products, about 1e-6
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        m = LM.PearsonMoments(3, 3, matmul_dtype=dtype)
        for lo, hi in ((0, 600), (600, 601), (601, 4000), (4000, n)):
            m.add(x[lo:hi], x[lo:hi])
        c = m.correlation().numpy()
        np.testing.assert_allclose(c, ref, atol=tol)
        assert not m.zero_variance()[0].any()
    with pytest.raises(LM.LayerMergeError, match="non-finite"):
        LM.PearsonMoments(1, 1).add(torch.tensor([[float("nan")]]), torch.tensor([[1.0]]))


def test_gram_cosine_and_group_sums():
    a = torch.randn(50, 4, 6, dtype=torch.float64)
    m = LM.GramMoments(4, 4)
    m.add(a, a)
    cos = m.cosine()
    assert torch.allclose(torch.diagonal(cos), torch.ones(4, dtype=torch.float64), atol=1e-6)
    z = LM.GramMoments(2, 2)
    z.add(torch.zeros(3, 2, 6), torch.randn(3, 2, 6))
    assert (z.cosine()[0] == 0).all()                 #: a zero-energy write has cosine 0, never NaN
    assert LM.group_sum(torch.ones(4, 4), 2, (0, 1)).tolist() == [[4.0, 4.0], [4.0, 4.0]]


def test_convex_weights_fall_back_to_uniform_where_every_score_is_zero():
    w, zero = LM.convex_weights(torch.tensor([[1.0, 0.0, 3.0], [1.0, 0.0, 1.0]], dtype=torch.float64))
    assert zero == 1 and w[:, 1].tolist() == [0.5, 0.5] and w[:, 2].tolist() == [0.75, 0.25]
    with pytest.raises(LM.LayerMergeError):
        LM.convex_weights(torch.tensor([[-1.0], [2.0]]))


def _block_io(teacher, b=2, t=12):
    """A real block input (and the kwargs the model passes) from one forward of the toy teacher."""
    store = {}

    def pre(_m, args, kwargs):
        store["x"] = args[0] if args else kwargs["hidden_states"]
        store["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
    h = ADAPTER.blocks(teacher)[b].register_forward_pre_hook(pre, with_kwargs=True)
    teacher(torch.randint(0, 128, (1, t)), use_cache=False)
    h.remove()
    return store["x"], store["kw"]


def _run(block, x, kw):
    out = block(x, **kw)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def test_folding_and_permuting_preserve_the_blocks_forward(teacher):
    teacher = teacher.double()
    n_q, n_kv, hd = ADAPTER.head_groups(ADAPTER.spec_of(teacher))
    block = ADAPTER.blocks(teacher)[2]
    x, kw = _block_io(teacher)
    y = _run(block, x, kw)
    t = LM.block_tensors(ADAPTER, block)
    folded = LM.fold_norm_gains(ADAPTER, t)
    assert (folded["attn_norm"] == 1).all() and (folded["ffn_norm"] == 1).all()
    assert not torch.equal(folded["q"], t["q"])        #: the toy's gains are not 1, so folding moved the weights
    yf = _run(LM.block_like(ADAPTER, block, folded), x, kw)
    assert float((yf - y).norm() / (y - x).norm()) < 1e-10
    g = torch.Generator().manual_seed(3)
    ffn = torch.randperm(t["gate"].shape[0], generator=g).tolist()
    kv = torch.randperm(n_kv, generator=g).tolist()
    within = [torch.randperm(n_q // n_kv, generator=g).tolist() for _ in range(n_kv)]
    perm = LM.permute_block(t, ffn_perm=ffn, kv_perm=kv, q_perm_within=within, n_q=n_q, n_kv=n_kv, head_dim=hd)
    assert not torch.equal(perm["gate"], t["gate"])
    yp = _run(LM.block_like(ADAPTER, block, perm), x, kw)
    assert float((yp - y).norm() / (y - x).norm()) < 1e-10
    with pytest.raises(LM.LayerMergeError):
        LM.permute_block(t, ffn_perm=[0] * len(ffn), kv_perm=None, q_perm_within=None, n_q=n_q, n_kv=n_kv,
                         head_dim=hd)


@torch.no_grad()
def test_a_block_merged_with_its_own_permuted_copy_aligned_back_is_the_block(teacher):
    """Alignment recovers a permutation exactly, so averaging a block with its realigned copy gives the block back;
    an unaligned average does not (it mixes unrelated neurons and heads)."""
    n_q, n_kv, hd = ADAPTER.head_groups(ADAPTER.spec_of(teacher))
    block = ADAPTER.blocks(teacher)[3]
    t = LM.block_tensors(ADAPTER, block)
    g = torch.Generator().manual_seed(4)
    ffn = torch.randperm(t["gate"].shape[0], generator=g)
    kv = torch.randperm(n_kv, generator=g)
    within = [torch.randperm(n_q // n_kv, generator=g) for _ in range(n_kv)]
    copy = LM.permute_block(t, ffn_perm=ffn.tolist(), kv_perm=kv.tolist(), q_perm_within=[w.tolist() for w in within],
                            n_q=n_q, n_kv=n_kv, head_dim=hd)
    #: the inverse: slot i of the anchor holds what slot inv[i] of the copy holds
    inv_ffn = torch.argsort(ffn)
    inv_kv = torch.argsort(kv)
    inv_within = [torch.argsort(within[int(inv_kv[j])]).tolist() for j in range(n_kv)]
    back = LM.permute_block(copy, ffn_perm=inv_ffn.tolist(), kv_perm=inv_kv.tolist(), q_perm_within=inv_within,
                            n_q=n_q, n_kv=n_kv, head_dim=hd)
    w = LM.MergeWeights.uniform(2, t["gate"].shape[0], n_q, n_kv)
    merged = LM.merge_blocks([t, back], w, head_dim=hd)
    assert all(torch.allclose(merged[k], t[k], atol=1e-12) for k in t)
    naive = LM.merge_blocks([t, copy], w, head_dim=hd)
    assert not torch.allclose(naive["gate"], t["gate"])


@torch.no_grad()
def test_merge_weights_are_per_slot_and_write_block_round_trips(teacher):
    n_q, n_kv, hd = ADAPTER.head_groups(ADAPTER.spec_of(teacher))
    a, b = (LM.block_tensors(ADAPTER, ADAPTER.blocks(teacher)[i]) for i in (1, 2))
    n_ffn = a["gate"].shape[0]
    ffn = torch.zeros(2, n_ffn, dtype=torch.float64)
    ffn[0, 0], ffn[1, 0], ffn[0, 1:], ffn[1, 1:] = 1.0, 0.0, 0.0, 1.0
    w = LM.MergeWeights(ffn=ffn, query_head=torch.full((2, n_q), 0.5, dtype=torch.float64),
                        kv_group=torch.full((2, n_kv), 0.5, dtype=torch.float64),
                        block=torch.tensor([0.5, 0.5], dtype=torch.float64))
    m = LM.merge_blocks([a, b], w, head_dim=hd)
    assert torch.equal(m["gate"][0], a["gate"][0]) and torch.equal(m["gate"][1:], b["gate"][1:])
    assert torch.equal(m["ffn_out"][:, 0], a["ffn_out"][:, 0])
    bad = LM.MergeWeights(ffn=ffn * 2, query_head=w.query_head, kv_group=w.kv_group, block=w.block)
    with pytest.raises(LM.LayerMergeError):
        LM.merge_blocks([a, b], bad, head_dim=hd)
    blk = LM.block_like(ADAPTER, ADAPTER.blocks(teacher)[1], m)
    back = LM.block_tensors(ADAPTER, blk)
    assert all(torch.allclose(back[k], m[k].to(torch.float32).to(torch.float64)) for k in m)


def test_update_moments_summarize_norms_cosine_and_error():
    r = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    m = LM.UpdateMoments()
    m.add(0.5 * r, r)
    s = m.summary()
    assert s["norm_ratio"] == pytest.approx(0.5) and s["cosine"] == pytest.approx(1.0)
    assert s["normalized_error"] == pytest.approx(0.25)
