"""Grouped-head selection topology, shared by every GQA ATTENTION algorithm.

Everything here encodes one structural fact: query heads are partitioned into
groups that each share a KV head, and an operator of this topology reduces the
query heads **within** each group while leaving the KV heads, ``head_dim``, the
grouping and the RoPE basis alone. `weight_proxy`, `activation_importance` and
any future causal-KL algorithm differ only in the *score* they rank heads by;
the grouping arithmetic, the tie-break and the weight slicing below are common
to all of them, which is what makes a comparison between two of them a
comparison of the importance signal.

That is also why this is `attention/gqa/_common.py` and not
`attention/_common.py`. None of it survives a change of topology: MLA has no
per-head output slice to select columns of, and linear attention has no
softmax-normalised head to drop independently. A helper that would be wrong for
the sibling topologies does not belong at their shared level.

Family knowledge stays in the adapter. This module asks for roles — ``q``,
``attn_out`` — and never for ``.q_proj`` or ``.o_proj``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from aadistill.initialization.specs.arch import (
    ArchitectureAdapter,
    UnsupportedCapability,
)

#: The adapter role names this topology consumes. These two constants are the
#: whole coupling between a GQA operator and a model family.
ATTN_OUT_ROLE = "attn_out"
QUERY_ROLE = "q"


def head_rows(heads: list[int], head_dim: int, device: Any = None) -> torch.Tensor:
    """Row indices for a set of attention heads, on the device that will be
    indexed.

    An index built from a Python list lands on the host whatever it is about to
    slice. Some torch ops accept that and some raise; relying on which is worse
    than either. `device` is not optional in practice — every caller in the
    search passes the weight's device — but it defaults to None so a caller that
    only wants the arithmetic is not forced to invent one.

    Concatenated-per-head layout is a GQA assumption, which is why this lives
    here rather than in the kind-neutral `operators/_common.py` it used to share
    with FFN and DEPTH surgery. `transforms/sandwich.py` keeps its own older
    sibling `_head_rows`; the two are deliberately not merged, because that one
    belongs to the frozen monolithic recipe.
    """
    rows = torch.tensor([h * head_dim + i for h in heads for i in range(head_dim)])
    return rows if device is None else rows.to(device)


def attention_out_projection(adapter: ArchitectureAdapter, block: Any) -> Any:
    """The linear that writes attention's result into the residual stream."""
    roles = adapter.stream_out_projections(block)
    if ATTN_OUT_ROLE not in roles:
        raise UnsupportedCapability(
            f"the {adapter.family} adapter exposes stream-out roles "
            f"{sorted(roles)} and not {ATTN_OUT_ROLE!r}; a grouped-head operator "
            "scores heads by what they write through that projection and cannot "
            "proceed without it")
    return roles[ATTN_OUT_ROLE]


def query_projection(adapter: ArchitectureAdapter, block: Any) -> Any:
    """The linear that reads the residual stream into query space."""
    roles = adapter.stream_in_projections(block)
    if QUERY_ROLE not in roles:
        raise UnsupportedCapability(
            f"the {adapter.family} adapter exposes stream-in roles "
            f"{sorted(roles)} and not {QUERY_ROLE!r}; a grouped-head operator "
            "selects query heads and cannot proceed without it")
    #: role -> (linear, preceding norm). Only the linear is returned; the norm is
    #: untouched, because selecting query heads changes the output width of this
    #: projection and not the residual width it reads.
    return roles[QUERY_ROLE][0]


def select_q_heads_by_score(scores: Sequence[float] | torch.Tensor, n_q_heads: int,
                            n_kv_heads: int, keep_q: int) -> list[int]:
    """Per-GQA-group top-k by a precomputed score. Deterministic.

    Same grouping and same retention arithmetic as
    ``init.sandwich.select_q_heads`` — only the score differs — so every
    algorithm of this topology shares one selection topology and a contrast
    between two of them is the importance signal alone.

    Ties break by **ascending head index**, stated rather than inherited from
    sort stability, because a silent tie-break is exactly what makes a replay
    irreproducible.
    """
    if n_q_heads % n_kv_heads or keep_q % n_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads (GQA grouping)")
    per_g_t, per_g_s = n_q_heads // n_kv_heads, keep_q // n_kv_heads
    if per_g_s > per_g_t:
        raise ValueError(f"cannot keep {per_g_s} of {per_g_t} Q heads per group")
    kept: list[int] = []
    for g in range(n_kv_heads):
        group = range(g * per_g_t, (g + 1) * per_g_t)
        top = sorted(group, key=lambda h: (-float(scores[h]), h))[:per_g_s]
        kept.extend(sorted(top))
    return kept


def refuse_unless_reducible_within_groups(n_q: int, n_kv: int,
                                          keep_q: int) -> tuple[bool, str]:
    """The applicability rule every grouped-head reduction shares.

    Named before the divisibility test, which would otherwise refuse the MHA
    case with an arithmetic message that hides the real reason. Under MHA every
    query head owns its KV head, so dropping a query head necessarily drops a KV
    head — and this topology's contract is that KV heads, ``head_dim`` and the
    grouping are PRESERVED. There is no approximation to fall back on, so it
    refuses rather than silently redefining the transformation.
    """
    if keep_q > n_q:
        return False, "cannot add query heads"
    if n_kv == n_q and keep_q != n_q:
        return False, (
            f"multi-head attention ({n_q}Q/{n_kv}KV): reducing to {keep_q} "
            f"query heads cannot preserve {n_kv} KV heads, because under MHA "
            "each query head has its own. This operator preserves KV heads by "
            "contract; reducing them is a different transformation and needs "
            "a different operator.")
    if keep_q % n_kv:
        return False, (f"target {keep_q} query heads is not divisible "
                       f"by {n_kv} KV heads")
    return True, "ok"
