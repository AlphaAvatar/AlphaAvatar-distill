"""Initialize a projected-residual student from a full-width parent.

The student (`aadistill.initialization.adapters.projected_residual`) keeps the parent's residual
width and gives every layer ``l`` one projection ``P_l`` through which its
attention and FFN read and write. This module builds it in three explicit steps:

1. ``select_full_width`` -- the **full-width parent** ``W0``: the teacher with a
   given set of layers, query heads and FFN neurons kept and nothing else
   changed. It is a stock model of the teacher's width; it is also exactly the
   projected student with ``internal_size == hidden_size`` and ``P_l = I``,
   which is what makes the next two steps checkable.
2. ``SublayerStatsCollector`` + ``projection_basis`` -- per-layer uncentered
   second moments of a declared activation, and the top-``r`` eigenbasis of
   them as ``P_l``. Which activation is a *declared choice*
   (``STATISTICS``), never a default buried here:

   * ``sublayer_inputs`` -- the two normalized sub-layer inputs, the vectors
     ``P_l`` actually multiplies, each trace-normalized so neither dominates;
   * ``residual_input`` -- the un-normalized stream entering the block;
   * ``read_write`` -- the two normalized inputs plus the two sub-layer outputs
     ``P_l^T`` must be able to write.

3. ``materialize_projected`` -- fold ``P_l`` into ``W0``'s sub-layer weights:
   readers ``W P_l``, writers ``P_l^T W``. Norms, q/k norms, embedding, output
   head and RoPE are ``W0``'s, untouched.

Module access goes through the architecture adapter's roles (``q``, ``gate``,
``attn_out``, the two norms ...), so nothing here names a teacher, a width or a
selection; those arrive as arguments (AGENTS.md P3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

#: Which second moments a basis is computed from: name -> the roles pooled.
STATISTICS = {
    "sublayer_inputs": ("attn_in", "ffn_in"),
    "residual_input": ("residual_in",),
    "read_write": ("attn_in", "ffn_in", "attn_out", "ffn_out"),
}
ROLES = ("residual_in", "attn_in", "ffn_in", "attn_out", "ffn_out")
READERS = ("q", "k", "v", "gate", "up")
WRITERS = ("attn_out", "ffn_out")

__all__ = [
    "STATISTICS",
    "SelectionError",
    "selection_problems",
    "select_full_width",
    "SublayerStatsCollector",
    "projection_basis",
    "materialize_projected",
]


class SelectionError(ValueError):
    """A structural selection that does not describe the requested target."""


def _head_rows(heads: Sequence[int], head_dim: int, device: Any = None) -> torch.Tensor:
    """Row indices of the given query heads in a concatenated-per-head projection.

    A local sibling of the GQA operators' helper, as `transforms/sandwich.py`
    keeps its own: a transform importing operator code would make the two
    packages depend on each other.
    """
    rows = torch.tensor([h * head_dim + i for h in heads for i in range(head_dim)])
    return rows if device is None else rows.to(device)


def _gqa_reduction_problem(n_q: int, n_kv: int, keep_q: int) -> str | None:
    """Why ``n_q -> keep_q`` query heads cannot keep all ``n_kv`` KV heads and the grouping."""
    if keep_q > n_q:
        return "cannot add query heads"
    if n_q % n_kv:
        return f"{n_q} query heads are not divisible by {n_kv} KV heads"
    if n_kv == n_q and keep_q != n_q:
        return "multi-head attention: dropping a query head drops its KV head"
    if keep_q % n_kv:
        return f"target {keep_q} query heads is not divisible by {n_kv} KV heads"
    return None


def selection_problems(selection: Mapping[str, Any], parent: Mapping[str, Any],
                       target: Mapping[str, Any]) -> list[str]:
    """Why ``selection`` cannot turn ``parent`` geometry into ``target``; empty when it can.

    ``selection`` holds ``kept_layers`` (parent layer ids, ascending),
    ``kept_heads`` (per kept layer, parent query-head ids) and
    ``kept_neurons`` (per kept layer, parent FFN neuron ids).
    """
    out = []
    layers = list(selection.get("kept_layers") or [])
    heads = list(selection.get("kept_heads") or [])
    neurons = list(selection.get("kept_neurons") or [])
    n = int(target["num_hidden_layers"])
    if len(layers) != n or layers != sorted(set(layers)):
        out.append(f"kept_layers must be {n} distinct ascending ids, got {len(layers)}")
    if layers and not 0 <= min(layers) <= max(layers) < int(parent["num_hidden_layers"]):
        out.append("kept_layers names a layer the parent does not have")
    n_q, n_kv = int(parent["num_attention_heads"]), int(parent["num_key_value_heads"])
    keep_q = int(target["num_attention_heads"])
    why = _gqa_reduction_problem(n_q, n_kv, keep_q)
    if why:
        out.append(why)
    if int(target["num_key_value_heads"]) != n_kv or int(target["head_dim"]) != int(parent["head_dim"]):
        out.append("KV heads and head_dim are preserved by this transform")
    if len(heads) != n:
        out.append(f"kept_heads must have one entry per kept layer ({n}), got {len(heads)}")
    group = n_q // n_kv
    for i, hs in enumerate(heads):
        if len(hs) != keep_q or list(hs) != sorted(set(hs)) or not all(0 <= h < n_q for h in hs):
            out.append(f"layer {i}: kept_heads must be {keep_q} distinct ascending parent heads")
            continue
        per_group = [sum(1 for h in hs if h // group == g) for g in range(n_kv)]
        if len(set(per_group)) != 1:
            out.append(f"layer {i}: kept heads are not balanced over the {n_kv} KV groups {per_group}")
    inter = int(target["intermediate_size"])
    if len(neurons) != n:
        out.append(f"kept_neurons must have one entry per kept layer ({n}), got {len(neurons)}")
    for i, ns in enumerate(neurons):
        if len(ns) != inter or list(ns) != sorted(set(ns)) or not all(
                0 <= x < int(parent["intermediate_size"]) for x in ns):
            out.append(f"layer {i}: kept_neurons must be {inter} distinct ascending parent neurons")
    return out


@torch.no_grad()
def select_full_width(adapter: Any, parent: Any, selection: Mapping[str, Any],
                      target_spec: Any, *, dtype: Any = None) -> Any:
    """``W0``: ``parent`` with the selected layers, heads and neurons kept, width unchanged.

    ``target_spec`` is the ``ArchSpec`` of ``W0`` (the parent's width and the
    student's layer, head and FFN counts). Refuses a selection that does not
    describe it.
    """
    pspec = adapter.spec_of(parent)
    problems = selection_problems(selection, pspec.as_dict(), target_spec.as_dict())
    if target_spec["hidden_size"] != pspec["hidden_size"]:
        problems.append("W0 keeps the parent's width")
    if problems:
        raise SelectionError("; ".join(problems))
    dtype = dtype or next(parent.parameters()).dtype
    child = adapter.build_model(adapter.build_config(parent.config, target_spec), dtype, seed=0)
    hd = int(pspec["head_dim"])
    src_blocks = adapter.blocks(parent)
    adapter.embedding(child).weight.copy_(adapter.embedding(parent).weight)
    adapter.final_norm(child).weight.copy_(adapter.final_norm(parent).weight)
    for i, (layer, heads, neurons) in enumerate(zip(selection["kept_layers"], selection["kept_heads"],
                                                    selection["kept_neurons"])):
        s, c = src_blocks[layer], adapter.blocks(child)[i]
        dev = adapter.attn_norm(s).weight.device
        rows = _head_rows(list(heads), hd, device=dev)
        idx = torch.as_tensor(list(neurons), device=dev)
        s_in, c_in = adapter.stream_in_projections(s), adapter.stream_in_projections(c)
        s_out, c_out = adapter.stream_out_projections(s), adapter.stream_out_projections(c)
        c_in["q"][0].weight.copy_(s_in["q"][0].weight[rows])
        for role in ("k", "v"):
            c_in[role][0].weight.copy_(s_in[role][0].weight)
        c_out["attn_out"].weight.copy_(s_out["attn_out"].weight[:, rows])
        for role in ("gate", "up"):
            c_in[role][0].weight.copy_(s_in[role][0].weight[idx])
        c_out["ffn_out"].weight.copy_(s_out["ffn_out"].weight[:, idx])
        adapter.attn_norm(c).weight.copy_(adapter.attn_norm(s).weight)
        adapter.ffn_norm(c).weight.copy_(adapter.ffn_norm(s).weight)
        for role, norm in adapter.attention_subnorms(s).items():
            adapter.attention_subnorms(c)[role].weight.copy_(norm.weight)
    if getattr(child.config, "tie_word_embeddings", False):
        child.tie_weights()
    return child.eval()


class SublayerStatsCollector:
    """Per-layer float64 uncentered second moments of the requested roles.

    ``attn_in`` / ``ffn_in`` are the two norms' outputs (what the sub-layers
    read), ``residual_in`` the block input, ``attn_out`` / ``ffn_out`` the
    sub-layer outputs (what they write). Only positions marked in ``active``
    accumulate, so padding and unscored positions never reach a moment.
    Accumulating in float64 follows `ActivationStatsCollector`: the residual
    stream carries outlier dimensions float32 would lose precision on.
    """

    def __init__(self, adapter: Any, model: Any, roles: Sequence[str]):
        bad = [r for r in roles if r not in ROLES]
        if bad:
            raise ValueError(f"unknown roles {bad}; known {ROLES}")
        self.roles = tuple(roles)
        blocks = adapter.blocks(model)
        d = int(adapter.spec_of(model)["hidden_size"])
        self.moments = {r: [torch.zeros(d, d, dtype=torch.float64) for _ in blocks] for r in self.roles}
        self.counts = {r: [0 for _ in blocks] for r in self.roles}
        self._mask = None
        self._handles = []
        for i, b in enumerate(blocks):
            taps = {"attn_in": adapter.attn_norm(b), "ffn_in": adapter.ffn_norm(b),
                    "attn_out": adapter.stream_out_projections(b)["attn_out"],
                    "ffn_out": adapter.stream_out_projections(b)["ffn_out"]}
            for role in self.roles:
                if role == "residual_in":
                    self._handles.append(b.register_forward_pre_hook(self._pre(role, i)))
                else:
                    self._handles.append(taps[role].register_forward_hook(self._post(role, i)))

    def _add(self, role: str, i: int, x: torch.Tensor) -> None:
        x = x.detach()
        rows = x[self._mask] if self._mask is not None else x.reshape(-1, x.shape[-1])
        rows = rows.double()
        m = self.moments[role][i]
        if m.device != rows.device:
            self.moments[role][i] = m = m.to(rows.device)
        m.add_(rows.T @ rows)
        self.counts[role][i] += int(rows.shape[0])

    def _pre(self, role, i):
        def hook(_module, args, kwargs=None):
            self._add(role, i, args[0])
        return hook

    def _post(self, role, i):
        def hook(_module, _args, output):
            self._add(role, i, output[0] if isinstance(output, tuple) else output)
        return hook

    @torch.no_grad()
    def process(self, model: Any, input_ids: torch.Tensor, active: torch.Tensor | None = None,
                **forward_kwargs) -> None:
        """Forward one batch; ``active`` is a ``[B, T]`` bool mask of the positions that count."""
        self._mask = None if active is None else active.to(input_ids.device).bool()
        try:
            model(input_ids=input_ids, use_cache=False, **forward_kwargs)
        finally:
            self._mask = None

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def state(self) -> dict:
        """Host copies: ``{role: [(moment, count) per layer]}``."""
        return {r: [(m.cpu(), c) for m, c in zip(self.moments[r], self.counts[r])] for r in self.roles}


def projection_basis(moments: Mapping[str, tuple[torch.Tensor, int]], rank: int) -> tuple[torch.Tensor, dict]:
    """The top-``rank`` eigenbasis of the pooled, trace-normalized moments of one layer.

    ``moments`` maps role -> (``sum x x^T``, count) for that layer. Each role is
    divided by its trace before averaging, so a role with larger activations
    does not decide the basis on its own. Columns are ordered by eigenvalue and
    signed so the largest-magnitude entry of each is positive: the basis is a
    deterministic function of the moments. Returns ``(P [d, rank], diagnostics)``
    where the diagnostics give each role's captured energy fraction.
    """
    if not moments:
        raise ValueError("no moments to compute a basis from")
    pooled = None
    for role, (m, n) in moments.items():
        if n <= 0:
            raise ValueError(f"role {role} accumulated no positions")
        tr = float(torch.trace(m))
        if not tr > 0:
            raise ValueError(f"role {role} has a non-positive trace")
        term = m.double() / tr
        pooled = term if pooled is None else pooled + term
    pooled = pooled / len(moments)
    d = pooled.shape[0]
    if not 0 < rank <= d:
        raise ValueError(f"rank {rank} outside (0, {d}]")
    evals, evecs = torch.linalg.eigh(pooled)
    order = torch.argsort(evals, descending=True)[:rank]
    basis = evecs[:, order]
    pivot = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivot, torch.arange(rank)])
    signs[signs == 0] = 1.0
    basis = basis * signs
    diag = {"rank": rank, "dim": d,
            "pooled_energy_captured": float(evals[order].sum() / evals.sum()),
            "orthonormality_error": float((basis.T @ basis - torch.eye(rank, dtype=basis.dtype)).abs().max())}
    for role, (m, _n) in moments.items():
        m = m.double()
        diag[f"{role}_energy_captured"] = float(torch.trace(basis.T @ m @ basis) / torch.trace(m))
    return basis, diag


@torch.no_grad()
def materialize_projected(adapter: Any, w0: Any, bases: Sequence[torch.Tensor], *,
                          dtype: Any = None) -> Any:
    """The projected student: ``W0``'s weights with each layer's ``P_l`` folded in.

    Readers ``W[out, d] @ P_l -> [out, r]``; writers ``P_l^T @ W[d, in] -> [r, in]``.
    With ``r == d`` and ``P_l = I`` the result computes exactly ``W0``.
    """
    from aadistill.initialization.adapters.projected_residual import (
        ProjectedResidualConfig,
        ProjectedResidualForCausalLM,
    )

    blocks = adapter.blocks(w0)
    if len(bases) != len(blocks):
        raise ValueError(f"{len(bases)} bases for {len(blocks)} layers")
    d = int(adapter.spec_of(w0)["hidden_size"])
    ranks = {int(b.shape[1]) for b in bases}
    if len(ranks) != 1 or any(b.shape[0] != d for b in bases):
        raise ValueError("every basis must be [hidden_size, r] with one common r")
    raw = w0.config.to_dict()
    for key in ("model_type", "architectures", "transformers_version"):
        raw.pop(key, None)
    cfg = ProjectedResidualConfig(internal_size=ranks.pop(), **raw)
    cfg.architectures = ["ProjectedResidualForCausalLM"]
    dtype = dtype or next(w0.parameters()).dtype
    student = ProjectedResidualForCausalLM(cfg).to(dtype)
    adapter.embedding(student).weight.copy_(adapter.embedding(w0).weight)
    adapter.final_norm(student).weight.copy_(adapter.final_norm(w0).weight)
    for p, s, c in zip(bases, blocks, adapter.blocks(student)):
        p = p.double().to(adapter.attn_norm(s).weight.device)
        c.projection.copy_(p)
        s_in, c_in = adapter.stream_in_projections(s), adapter.stream_in_projections(c)
        s_out, c_out = adapter.stream_out_projections(s), adapter.stream_out_projections(c)
        for role in READERS:
            c_in[role][0].weight.copy_(s_in[role][0].weight.double() @ p)
        for role in WRITERS:
            c_out[role].weight.copy_(p.T @ s_out[role].weight.double())
        adapter.attn_norm(c).weight.copy_(adapter.attn_norm(s).weight)
        adapter.ffn_norm(c).weight.copy_(adapter.ffn_norm(s).weight)
        for role, norm in adapter.attention_subnorms(s).items():
            adapter.attention_subnorms(c)[role].weight.copy_(norm.weight)
    student.tie_weights()
    return student.eval()
