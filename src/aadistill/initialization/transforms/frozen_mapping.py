"""A frozen structural mapping: one set of selections and transforms, applied to any subset of a root's blocks.

A search journal records WHAT the selecting operators chose -- the FFN neurons of every block, the blocks DEPTH kept,
the query heads of every block -- but an operator that COMPUTES a transform (the residual-width projection) keeps it
only inside the child it builds. To compare structural factors under one mapping (the same selections with and without
depth compression, say), the selections and the transform must exist as data, independent of the operator invocation
that produced them. This module is that data contract and its materialization, and nothing else:

* `FrozenMapping`: the root's identity, the number of root blocks, the kept blocks, the FFN neurons and query heads of
  each block keyed by its ROOT index, and an optional `WidthTransform` (the projection ``P``, the scale compensation and
  the final-norm vector). `save` writes JSON plus one safetensors file; every tensor's sha256 and one content digest
  over the whole mapping are in the JSON, and `load` refuses bytes that do not hash to them.
* builders that repeat the operators' own surgery (`ChildBuilder`, copying by parameter identity) on given selections:
  `select_ffn_neurons` (``ffn.activation_importance_v0``'s child), `keep_blocks` (DEPTH's), `project_residual_width`
  (``width.global_pca_v0``'s) and `select_query_heads` (``attention.activation_importance_v1``'s). Given the same parent
  and the same selection or transform they write the same tensors (tests/initialization/test_frozen_mapping.py).
* `width_transform_from_statistics`: the width operator's arithmetic on a statistics state, returned instead of
  applied, so ``P`` can be kept.
* `materialize`: a mapping onto a root, on all of its blocks or only the kept ones, with or without the non-depth
  transforms. The operations run in the order FFN -> blocks -> width -> heads, the order of the path a mapping is
  normally taken from; a different order can change float rounding (a GEMM's kernel depends on its shape), so the
  order is fixed rather than left to the caller.
* `block_differences`: which parameters of two models' corresponding blocks are not bit-identical.

It names no family, geometry, experiment or path: every instance fact (the root, the selections, the point weights the
width statistics were averaged with) arrives as an argument (AGENTS.md P3).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from aadistill.initialization.operators._common import (
    ChildBuilder,
    copy_embeddings_and_final_norm,
    copy_module_except,
)
from aadistill.initialization.statistics.spec import StatsCache
from aadistill.initialization.operators.attention.gqa._common import (
    attention_out_projection,
    head_rows,
    query_projection,
)
from aadistill.initialization.transforms.project import final_norm_weights, stream_projection
from aadistill.initialization.transforms.sandwich import _in_proj as fold_and_project

SCHEMA = "aadistill.frozen_structural_mapping/v1"
TENSOR_FILE = "tensors.safetensors"
MAPPING_FILE = "mapping.json"
DEPTH_FIELD, FFN_FIELD, WIDTH_FIELD, HEADS_FIELD = (
    "num_hidden_layers", "intermediate_size", "hidden_size", "num_attention_heads")


class FrozenMappingError(RuntimeError):
    """A mapping is malformed, does not fit its root, or its bytes are not the recorded ones."""


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of a tensor's dtype, shape and raw little-endian bytes (contiguous, on the host)."""
    t = t.detach().to("cpu").contiguous()
    head = f"{t.dtype}|{tuple(t.shape)}|".encode()
    return _sha256(head + t.view(torch.uint8).numpy().tobytes() if t.numel() else head)


@dataclass(frozen=True)
class WidthTransform:
    """The residual-width map: stream readers become ``scale * W diag(norm) P``, stream writers ``P^T W``, the embedding
    ``E P`` and the final norm ``final_norm``. ``projection`` is ``(d_parent, d_child)`` with orthonormal columns."""
    projection: torch.Tensor
    scale: float
    final_norm: torch.Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def d_parent(self) -> int:
        return int(self.projection.shape[0])

    @property
    def d_child(self) -> int:
        return int(self.projection.shape[1])


@dataclass(frozen=True)
class FrozenMapping:
    """One structural mapping of a root, every selection keyed by the ROOT's block index."""
    root: Mapping[str, Any]
    n_root_blocks: int
    kept_blocks: tuple[int, ...]
    ffn_neurons: Mapping[int, tuple[int, ...]] | None = None
    query_heads: Mapping[int, tuple[int, ...]] | None = None
    width: WidthTransform | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """Everything but the tensors' bytes, which are named by sha256."""
        w = self.width
        return {
            "schema": SCHEMA, "root": dict(self.root), "n_root_blocks": int(self.n_root_blocks),
            "kept_blocks": [int(b) for b in self.kept_blocks],
            "removed_blocks": sorted(set(range(self.n_root_blocks)) - set(self.kept_blocks)),
            "ffn_neurons": (None if self.ffn_neurons is None else
                            {str(b): [int(i) for i in v] for b, v in sorted(self.ffn_neurons.items())}),
            "query_heads": (None if self.query_heads is None else
                            {str(b): [int(h) for h in v] for b, v in sorted(self.query_heads.items())}),
            "width": (None if w is None else {
                "d_parent": w.d_parent, "d_child": w.d_child, "scale": float(w.scale),
                "projection": {"key": "width.projection", "dtype": str(w.projection.dtype),
                               "shape": list(w.projection.shape), "sha256": tensor_sha256(w.projection)},
                "final_norm": {"key": "width.final_norm", "dtype": str(w.final_norm.dtype),
                               "shape": list(w.final_norm.shape), "sha256": tensor_sha256(w.final_norm)},
                "diagnostics": dict(w.diagnostics)}),
            "provenance": dict(self.provenance)}

    @property
    def content_digest(self) -> str:
        return _sha256(_canonical(self.record()))


def problems(mapping: FrozenMapping, *, root_spec: Mapping[str, Any] | None = None) -> list[str]:
    """Why ``mapping`` is malformed (or does not fit ``root_spec``); empty when it is sound."""
    bad: list[str] = []
    n = int(mapping.n_root_blocks)
    kept = list(mapping.kept_blocks)
    if kept != sorted(set(kept)) or any(not 0 <= b < n for b in kept) or not kept:
        bad.append(f"kept_blocks {kept[:8]}... are not a non-empty increasing subset of range({n})")
    if root_spec is not None and int(root_spec[DEPTH_FIELD]) != n:
        bad.append(f"the root has {root_spec[DEPTH_FIELD]} blocks, the mapping {n}")
    for name, sel, width_key in (("ffn_neurons", mapping.ffn_neurons, FFN_FIELD),
                                 ("query_heads", mapping.query_heads, HEADS_FIELD)):
        if sel is None:
            continue
        sizes = {len(v) for v in sel.values()}
        if len(sizes) != 1:
            bad.append(f"{name}: blocks keep different counts {sorted(sizes)}")
        for b, v in sel.items():
            if not 0 <= int(b) < n:
                bad.append(f"{name}: block {b} is not a root block")
            if list(v) != sorted(set(v)):
                bad.append(f"{name}: block {b}'s selection is not strictly increasing")
            if root_spec is not None and (min(v) < 0 or max(v) >= int(root_spec[width_key])):
                bad.append(f"{name}: block {b} selects outside range({root_spec[width_key]})")
    if mapping.query_heads is not None and root_spec is not None:
        n_q, n_kv = int(root_spec[HEADS_FIELD]), int(root_spec["num_key_value_heads"])
        per = n_q // n_kv
        for b, v in mapping.query_heads.items():
            groups = [sum(1 for h in v if h // per == g) for g in range(n_kv)]
            if len(set(groups)) != 1:
                bad.append(f"query_heads: block {b} keeps {groups} heads per KV group; GQA needs an equal count")
    w = mapping.width
    if w is not None:
        if w.projection.dim() != 2 or w.final_norm.shape != (w.d_child,):
            bad.append(f"width: projection {tuple(w.projection.shape)} / final norm {tuple(w.final_norm.shape)}")
        elif root_spec is not None and w.d_parent != int(root_spec[WIDTH_FIELD]):
            bad.append(f"width: P maps {w.d_parent} dimensions, the root has {root_spec[WIDTH_FIELD]}")
        else:
            p = w.projection.to(torch.float64)
            err = float((p.T @ p - torch.eye(w.d_child, dtype=torch.float64, device=p.device)).abs().max())
            if err > 1e-8:
                bad.append(f"width: P's columns are not orthonormal (max error {err:.2e})")
    return bad


def save(mapping: FrozenMapping, directory: str | Path) -> dict[str, Any]:
    """Write ``mapping.json`` and ``tensors.safetensors`` into ``directory``; return the record with its digests."""
    from safetensors.torch import save_file
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    rec = mapping.record()
    if mapping.width is not None:
        save_file({"width.projection": mapping.width.projection.detach().to("cpu").contiguous(),
                   "width.final_norm": mapping.width.final_norm.detach().to("cpu").contiguous()},
                  str(d / TENSOR_FILE))
        rec["tensor_file"] = {"name": TENSOR_FILE, "sha256": _sha256((d / TENSOR_FILE).read_bytes())}
    rec["content_digest"] = mapping.content_digest
    (d / MAPPING_FILE).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def load(directory: str | Path, *, expected_content_digest: str | None = None) -> FrozenMapping:
    """A saved mapping, every tensor re-hashed; refuses a digest that is not the recorded (or expected) one."""
    from safetensors.torch import load_file
    d = Path(directory)
    rec = json.loads((d / MAPPING_FILE).read_text())
    if rec.get("schema") != SCHEMA:
        raise FrozenMappingError(f"{d / MAPPING_FILE} is {rec.get('schema')!r}, not {SCHEMA}")
    width = None
    if rec.get("width"):
        tf = d / TENSOR_FILE
        if _sha256(tf.read_bytes()) != rec["tensor_file"]["sha256"]:
            raise FrozenMappingError(f"{tf} does not hash to the recorded {rec['tensor_file']['sha256'][:12]}")
        t = load_file(str(tf))
        w = rec["width"]
        width = WidthTransform(projection=t[w["projection"]["key"]], scale=float(w["scale"]),
                               final_norm=t[w["final_norm"]["key"]], diagnostics=w.get("diagnostics") or {})
    sel = lambda key: (None if rec.get(key) is None else  # noqa: E731
                       {int(b): tuple(int(i) for i in v) for b, v in rec[key].items()})
    m = FrozenMapping(root=rec["root"], n_root_blocks=int(rec["n_root_blocks"]),
                      kept_blocks=tuple(int(b) for b in rec["kept_blocks"]), ffn_neurons=sel("ffn_neurons"),
                      query_heads=sel("query_heads"), width=width, provenance=rec.get("provenance") or {})
    got = m.content_digest
    for want, what in ((rec.get("content_digest"), "recorded"), (expected_content_digest, "expected")):
        if want is not None and got != want:
            raise FrozenMappingError(f"{d}: the mapping's content digest is {got[:12]}, the {what} one {want[:12]}")
    return m


# ---- the builders: the operators' surgery on given selections ----------------------------------------------------

@torch.no_grad()
def select_ffn_neurons(adapter, parent, kept: Sequence[Sequence[int]], *, seed: int = 0):
    """Each block keeps the given intermediate neurons: gate/up rows and down columns, everything else copied."""
    blocks = adapter.blocks(parent)
    sizes = {len(k) for k in kept}
    if len(kept) != len(blocks) or len(sizes) != 1:
        raise FrozenMappingError(f"{len(kept)} FFN selections of sizes {sorted(sizes)} for {len(blocks)} blocks")
    builder = ChildBuilder(adapter, parent, adapter.spec_of(parent).replace(**{FFN_FIELD: sizes.pop()}), seed=seed)
    for src, dst, k in zip(blocks, adapter.blocks(builder.model), kept):
        src_in, dst_in = adapter.stream_in_projections(src), adapter.stream_in_projections(dst)
        down_src = adapter.stream_out_projections(src)["ffn_out"]
        down_dst = adapter.stream_out_projections(dst)["ffn_out"]
        idx = torch.tensor(list(k), dtype=torch.long, device=down_src.weight.device)
        transformed = set()
        for role in ("gate", "up"):
            target = dst_in[role][0].weight
            builder.assign(target, src_in[role][0].weight[idx])
            transformed.add(id(target))
        builder.assign(down_dst.weight, down_src.weight[:, idx])
        transformed.add(id(down_dst.weight))
        copy_module_except(builder, src, dst, skip=transformed)
    copy_embeddings_and_final_norm(builder, adapter, parent)
    return builder.finish()


@torch.no_grad()
def keep_blocks(adapter, parent, kept: Sequence[int], *, seed: int = 0):
    """A child holding exactly the parent blocks ``kept`` (parent indices), in order, verbatim."""
    builder = ChildBuilder(adapter, parent, adapter.spec_of(parent).replace(**{DEPTH_FIELD: len(kept)}), seed=seed)
    parent_blocks = adapter.blocks(parent)
    for dst, i in zip(adapter.blocks(builder.model), kept):
        copy_module_except(builder, parent_blocks[i], dst)
    copy_embeddings_and_final_norm(builder, adapter, parent)
    return builder.finish()


@torch.no_grad()
def project_residual_width(adapter, parent, transform: WidthTransform, *, seed: int = 0):
    """The width map on every block: readers folded with their norm and projected, writers projected, the folded norms
    set to ones, the attention sub-norms copied, the embedding projected and the final norm assigned."""
    spec = adapter.spec_of(parent)
    if int(spec[WIDTH_FIELD]) != transform.d_parent:
        raise FrozenMappingError(f"the parent's width is {spec[WIDTH_FIELD]}, P maps {transform.d_parent}")
    device = adapter.embedding(parent).weight.device
    proj = transform.projection.to(device=device, dtype=torch.float64)
    scale = float(transform.scale)
    builder = ChildBuilder(adapter, parent, spec.replace(**{WIDTH_FIELD: transform.d_child}), seed=seed)
    for src, dst in zip(adapter.blocks(parent), adapter.blocks(builder.model)):
        src_in, dst_in = adapter.stream_in_projections(src), adapter.stream_in_projections(dst)
        for role, (linear, norm) in src_in.items():
            builder.assign(dst_in[role][0].weight, fold_and_project(linear.weight, norm.weight, proj, scale))
        src_out, dst_out = adapter.stream_out_projections(src), adapter.stream_out_projections(dst)
        for role, linear in src_out.items():
            builder.assign(dst_out[role].weight, proj.T @ linear.weight.to(torch.float64))
        builder.assign(adapter.attn_norm(dst).weight, torch.ones_like(adapter.attn_norm(dst).weight))
        builder.assign(adapter.ffn_norm(dst).weight, torch.ones_like(adapter.ffn_norm(dst).weight))
        s_attn, d_attn = adapter.attention(src), adapter.attention(dst)
        builder.assign(d_attn.q_norm.weight, s_attn.q_norm.weight)
        builder.assign(d_attn.k_norm.weight, s_attn.k_norm.weight)
    builder.assign(adapter.embedding(builder.model).weight, adapter.embedding(parent).weight.to(torch.float64) @ proj)
    builder.assign(adapter.final_norm(builder.model).weight, transform.final_norm.to(device=device))
    return builder.finish()


@torch.no_grad()
def select_query_heads(adapter, parent, kept: Sequence[Sequence[int]], *, seed: int = 0):
    """Each block keeps the given query heads: their q_proj rows and o_proj columns; KV heads, head_dim and every other
    parameter copied."""
    blocks = adapter.blocks(parent)
    spec = adapter.spec_of(parent)
    n_q, n_kv, head_dim = adapter.head_groups(spec)
    sizes = {len(k) for k in kept}
    if len(kept) != len(blocks) or len(sizes) != 1:
        raise FrozenMappingError(f"{len(kept)} head selections of sizes {sorted(sizes)} for {len(blocks)} blocks")
    keep_q = sizes.pop()
    if keep_q % n_kv:
        raise FrozenMappingError(f"{keep_q} query heads do not divide over {n_kv} KV heads")
    builder = ChildBuilder(adapter, parent, spec.replace(**{HEADS_FIELD: keep_q}), seed=seed)
    for src, dst, k in zip(blocks, adapter.blocks(builder.model), kept):
        s_out, d_out = attention_out_projection(adapter, src), attention_out_projection(adapter, dst)
        s_q, d_q = query_projection(adapter, src), query_projection(adapter, dst)
        rows = head_rows([int(h) for h in k], head_dim, device=s_q.weight.device)
        builder.assign(d_q.weight, s_q.weight[rows])
        builder.assign(d_out.weight, s_out.weight[:, rows])
        copy_module_except(builder, src, dst, skip={id(d_q.weight), id(d_out.weight)})
    copy_embeddings_and_final_norm(builder, adapter, parent)
    return builder.finish()


@dataclass
class RetainingStatsCache(StatsCache):
    """A `StatsCache` that also keeps every state it hands out (``retained``, by key), so the statistics an operator
    collected for itself -- the width operator's residual moments -- can be read back after it ran. Host-resident,
    like the cache it extends."""
    retained: dict = field(default_factory=dict)

    def get_or_collect(self, key: str, collect):
        state = super().get_or_collect(key, collect)
        self.retained[key] = state
        return state


@torch.no_grad()
def width_transform_from_statistics(state: Mapping[str, torch.Tensor], *, d_child: int, points: Sequence[int],
                                    point_weights: Sequence[float], final_norm_weight: torch.Tensor,
                                    post_norm_point: int, device: Any = None) -> WidthTransform:
    """``width.global_pca_v0``'s arithmetic on a residual-statistics state: ``P`` from the weighted trace-normalized
    second moments at ``points``, scale ``sqrt(d_parent / d_child)``, and the final norm ``scale *`` the least-squares
    diagonal at ``post_norm_point``. ``device`` is where it is computed (the parent's, to repeat the operator)."""
    from aadistill.initialization.device import stats_to
    if device is not None:
        state = stats_to(dict(state), device)
    proj, diag = stream_projection(dict(state), int(d_child), list(points), [float(w) for w in point_weights])
    d_parent = int(proj.shape[0])
    scale = (d_parent / int(d_child)) ** 0.5
    w_final = final_norm_weights(dict(state), proj, final_norm_weight.to(proj.device), post_norm_point=post_norm_point)
    keep = {k: (float(v) if isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1) else None)
            for k, v in diag.items() if k != "points"}
    return WidthTransform(projection=proj, scale=scale, final_norm=scale * w_final,
                          diagnostics={**{k: v for k, v in keep.items() if v is not None},
                                       "point_weights": [float(w) for w in point_weights],
                                       "points": [int(p) for p in points]})


def materialize(adapter, root, mapping: FrozenMapping, *, blocks: str = "all", non_depth: bool = True,
                seed: int = 0):
    """``mapping`` onto ``root``. ``blocks``: ``"all"`` (every root block) or ``"kept"`` (the kept blocks);
    ``non_depth``: apply the FFN, width and head transforms, or none of them. Returns the model and the root index of
    each of its blocks. ``root`` itself is never modified."""
    if blocks not in ("all", "kept"):
        raise FrozenMappingError(f"blocks must be 'all' or 'kept', not {blocks!r}")
    bad = problems(mapping, root_spec=adapter.spec_of(root))
    if bad:
        raise FrozenMappingError("; ".join(bad))
    roots = list(range(mapping.n_root_blocks))
    model = root
    if non_depth and mapping.ffn_neurons is not None:
        model = select_ffn_neurons(adapter, model, [mapping.ffn_neurons[r] for r in roots], seed=seed)
    if blocks == "kept":
        nxt = keep_blocks(adapter, model, [roots.index(k) for k in mapping.kept_blocks], seed=seed)
        model, roots = nxt, list(mapping.kept_blocks)
    if non_depth and mapping.width is not None:
        model = project_residual_width(adapter, model, mapping.width, seed=seed)
    if non_depth and mapping.query_heads is not None:
        missing = [r for r in roots if r not in mapping.query_heads]
        if missing:
            raise FrozenMappingError(f"no query-head selection for root blocks {missing}")
        model = select_query_heads(adapter, model, [mapping.query_heads[r] for r in roots], seed=seed)
    if model is root:
        raise FrozenMappingError("this materialization changes nothing; use the root itself")
    return model, roots


def block_differences(adapter, a, b, pairs: Sequence[tuple[int, int]]) -> list[str]:
    """For each ``(i, j)``, the parameters of ``a``'s block i and ``b``'s block j that are not bit-identical."""
    out = []
    ba, bb = adapter.blocks(a), adapter.blocks(b)
    for i, j in pairs:
        pa, pb = dict(ba[i].named_parameters()), dict(bb[j].named_parameters())
        if pa.keys() != pb.keys():
            out.append(f"blocks {i}/{j}: different parameter names")
            continue
        out += [f"blocks {i}/{j}: {n}" for n in pa
                if pa[n].shape != pb[n].shape or pa[n].dtype != pb[n].dtype
                or not torch.equal(pa[n].detach().cpu(), pb[n].detach().cpu())]
    return out


def shared_parameter_differences(adapter, a, b) -> list[str]:
    """The embedding and final norm of two models, compared bit for bit (a tied output head is the embedding)."""
    out = []
    for name, get in (("embedding", adapter.embedding), ("final_norm", adapter.final_norm)):
        x, y = get(a).weight.detach().cpu(), get(b).weight.detach().cpu()
        if x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            out.append(name)
    return out
