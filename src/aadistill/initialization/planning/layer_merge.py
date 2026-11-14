"""Layer merging: one block written from a span of consecutive root blocks, as data and as arithmetic.

A depth-reduced child that keeps root block ``l0`` and drops ``l0+1 .. lk`` executes the span ``[l0 .. lk]`` as ONE
block, at ``l0``'s place, reading the span's input. Direct deletion keeps ``l0`` as it is; a merge writes that one
block from all of the span's blocks. This module is the reusable part of doing so, and it names no family, geometry,
experiment or path (AGENTS.md P3): blocks are reached through the architecture adapter's roles (``stream_in_projections``
q/k/v/gate/up with their preceding norms, ``stream_out_projections`` attn_out/ffn_out, ``attention_subnorms``,
``head_groups``), and every instance fact -- the spans, the statistics, the positions they were taken at -- is an
argument.

* `spans_from_kept_blocks`: each removed block joins the nearest kept block BEFORE it (the merged block stands where its
  span starts and receives the span's input); a run of consecutive removed blocks joins as a whole.
* `linear_assignment`: an exact maximum-weight perfect matching on a square score matrix -- the shortest augmenting
  path algorithm with dual variables (Crouse 2016, "On implementing 2D rectangular assignment algorithms", the method
  `scipy.optimize.linear_sum_assignment` implements), in float64 torch so it runs where the statistics are. Tie rule:
  among columns of equal reduced cost an unassigned column first, then the lowest column index -- deterministic for a
  given matrix.
* `PearsonMoments`: streaming float64 sums for the Pearson cross-correlation between two feature sets on the same
  positions. A feature whose variance is zero on those positions (relative variance at or below `ZERO_VARIANCE_REL`)
  has correlation 0 with every feature: its correlation is undefined, and 0 is "no evidence", never a match.
* `fold_norm_gains`: an RMSNorm gain moved into the linears that read its output (``W (g * x_hat) = (W diag g) x_hat``)
  and the gain set to one -- exact in real arithmetic; the per-head q/k sub-norms act after the projection and are not
  folded.
* `permute_block`: an exact re-parameterization -- FFN neurons (gate rows, up rows, down columns together) and GQA
  heads (whole KV groups with their query heads; query heads within a group), head dimensions never moved.
* `merge_blocks`: one block from several aligned blocks, per tensor slot a convex combination whose weights are given
  (uniform, or per FFN neuron / query head / KV group / block); `convex_weights` turns non-negative scores into such
  weights and falls back to uniform where every score of a slot is zero.
* `UpdateMoments`: streaming sums for how a candidate block's residual update compares with a reference update
  (norms, inner product, squared error), per component.

Numerics: every combination is computed in float64 from the source tensors and cast to the destination dtype once.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

#: a feature counts as having zero variance when its variance is at or below this fraction of its mean square (float64
#: cancellation in ``E[x^2] - E[x]^2`` leaves about 1e-16 relative; 1e-12 is far above it and far below any feature a
#: correlation could rest on)
ZERO_VARIANCE_REL = 1e-12

FFN_IN_ROLES = ("gate", "up")
ATTN_IN_ROLES = ("q", "k", "v")


class LayerMergeError(RuntimeError):
    pass


# ---- spans --------------------------------------------------------------------------------------------------------

def spans_from_kept_blocks(n_root: int, kept: Sequence[int]) -> list[tuple[int, ...]]:
    """One span per kept block: the kept block and the removed blocks after it, up to the next kept block. A removed
    block with no kept block before it has nowhere to go and is refused."""
    kept = [int(k) for k in kept]
    if kept != sorted(set(kept)) or not kept or kept[0] < 0 or kept[-1] >= n_root:
        raise LayerMergeError(f"kept blocks {kept} are not increasing indices into {n_root} root blocks")
    if kept[0] != 0:
        raise LayerMergeError(f"blocks {list(range(kept[0]))} precede every kept block; no kept block absorbs them")
    bounds = kept[1:] + [n_root]
    return [tuple(range(k, b)) for k, b in zip(kept, bounds)]


# ---- exact linear assignment ----------------------------------------------------------------------------------------

@torch.no_grad()
def linear_assignment(score: torch.Tensor) -> torch.Tensor:
    """``col[i]``: the column matched to row ``i`` in a permutation maximizing ``sum_i score[i, col[i]]`` (square
    score). Exact; ties resolved as the module docstring states."""
    if score.dim() != 2 or score.shape[0] != score.shape[1]:
        raise LayerMergeError(f"linear_assignment needs a square matrix, got {tuple(score.shape)}")
    if not bool(torch.isfinite(score).all()):
        raise LayerMergeError("linear_assignment: the score matrix holds a non-finite value")
    cost = (-score).to(torch.float64)
    n = cost.shape[0]
    dev = cost.device
    u = torch.zeros(n, dtype=torch.float64, device=dev)
    v = torch.zeros(n, dtype=torch.float64, device=dev)
    col4row = torch.full((n,), -1, dtype=torch.long, device=dev)
    row4col = [-1] * n
    inf = torch.tensor(float("inf"), dtype=torch.float64, device=dev)
    index = torch.arange(n, device=dev)
    for cur in range(n):
        spc = torch.full((n,), float("inf"), dtype=torch.float64, device=dev)
        path = torch.full((n,), -1, dtype=torch.long, device=dev)
        scanned = torch.zeros(n, dtype=torch.bool, device=dev)
        free = torch.tensor([r == -1 for r in row4col], dtype=torch.bool, device=dev)
        rows_seen, i, min_val, sink = [], cur, 0.0, -1
        while sink == -1:
            rows_seen.append(i)
            reduced = min_val + cost[i] - u[i] - v
            better = (~scanned) & (reduced < spc)
            spc = torch.where(better, reduced, spc)
            path = torch.where(better, torch.full_like(path, i), path)
            candidates = torch.where(scanned, inf, spc)
            low = candidates.min()
            ties = candidates == low
            pick = ties & free
            j = int(index[pick][0]) if bool(pick.any()) else int(index[ties][0])
            min_val = float(low)
            scanned[j] = True
            if row4col[j] == -1:
                sink = j
            else:
                i = row4col[j]
        u[cur] += min_val
        others = torch.tensor([r for r in rows_seen if r != cur], dtype=torch.long, device=dev)
        if len(others):
            u[others] += min_val - spc[col4row[others]]
        v[scanned] -= min_val - spc[scanned]
        j, trail = sink, path.tolist()
        while True:
            i = trail[j]
            row4col[j] = i
            prev = int(col4row[i])
            col4row[i] = j
            j = prev
            if i == cur:
                break
    if sorted(col4row.tolist()) != list(range(n)):
        raise LayerMergeError("linear_assignment did not produce a permutation")
    return col4row


# ---- streaming statistics -----------------------------------------------------------------------------------------

@dataclass
class PearsonMoments:
    """Sums for ``corr(x_i, y_j)`` over positions, x: ``[T, n_x]``, y: ``[T, n_y]`` on the SAME positions.

    ``matmul_dtype`` is the dtype of the per-call cross product (float32 by default: inputs that are bf16 activations
    are represented exactly, and the products accumulate in float32 within one call); every sum across calls is
    float64."""
    n_x: int
    n_y: int
    device: str | torch.device = "cpu"
    matmul_dtype: torch.dtype = torch.float32
    count: int = 0
    sx: torch.Tensor = field(init=False)
    sy: torch.Tensor = field(init=False)
    sxx: torch.Tensor = field(init=False)
    syy: torch.Tensor = field(init=False)
    sxy: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=self.device)  # noqa: E731
        self.sx, self.sy, self.sxx, self.syy = z(self.n_x), z(self.n_y), z(self.n_x), z(self.n_y)
        self.sxy = z(self.n_x, self.n_y)

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[0] != y.shape[0] or x.shape[1] != self.n_x or y.shape[1] != self.n_y:
            raise LayerMergeError(f"PearsonMoments: shapes {tuple(x.shape)} / {tuple(y.shape)} for {self.n_x} x "
                                  f"{self.n_y} on shared positions")
        if x.shape[0] == 0:
            return
        x64, y64 = x.to(self.device, torch.float64), y.to(self.device, torch.float64)
        self.sx += x64.sum(0)
        self.sy += y64.sum(0)
        self.sxx += (x64 * x64).sum(0)
        self.syy += (y64 * y64).sum(0)
        xm, ym = x.to(self.device, self.matmul_dtype), y.to(self.device, self.matmul_dtype)
        self.sxy += (xm.T @ ym).to(torch.float64)
        self.count += int(x.shape[0])

    def zero_variance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Boolean masks of the x and y features with zero variance (module docstring)."""
        out = []
        for s, ss in ((self.sx, self.sxx), (self.sy, self.syy)):
            mean, msq = s / self.count, ss / self.count
            var = (msq - mean * mean).clamp_min(0.0)
            out.append((var <= ZERO_VARIANCE_REL * msq) | (msq == 0))
        return out[0], out[1]

    def correlation(self) -> torch.Tensor:
        """``[n_x, n_y]`` Pearson correlations; a zero-variance feature's row or column is 0."""
        if self.count < 2:
            raise LayerMergeError("PearsonMoments: fewer than two positions")
        n = float(self.count)
        mx, my = self.sx / n, self.sy / n
        vx = (self.sxx / n - mx * mx).clamp_min(0.0)
        vy = (self.syy / n - my * my).clamp_min(0.0)
        cov = self.sxy / n - mx[:, None] * my[None, :]
        zx, zy = self.zero_variance()
        sx = torch.where(zx, torch.ones_like(vx), vx.sqrt())
        sy = torch.where(zy, torch.ones_like(vy), vy.sqrt())
        c = cov / (sx[:, None] * sy[None, :])
        c[zx, :] = 0.0
        c[:, zy] = 0.0
        return c.clamp(-1.0, 1.0)


@dataclass
class GramMoments:
    """``sum_t <a_i(t), b_j(t)>`` between two sets of vectors over positions (a: ``[T, n_a, d]``, b: ``[T, n_b, d]``),
    and each set's own energies ``sum_t |a_i|^2``: the cosine similarity of two sets of residual writes."""
    n_a: int
    n_b: int
    device: str | torch.device = "cpu"
    count: int = 0
    cross: torch.Tensor = field(init=False)
    energy_a: torch.Tensor = field(init=False)
    energy_b: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.cross = torch.zeros(self.n_a, self.n_b, dtype=torch.float64, device=self.device)
        self.energy_a = torch.zeros(self.n_a, dtype=torch.float64, device=self.device)
        self.energy_b = torch.zeros(self.n_b, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def add(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
            raise LayerMergeError(f"GramMoments: shapes {tuple(a.shape)} / {tuple(b.shape)}")
        a32, b32 = a.to(self.device, torch.float32), b.to(self.device, torch.float32)
        self.cross += torch.einsum("tid,tjd->ij", a32, b32).to(torch.float64)
        self.energy_a += (a32 * a32).sum((0, 2)).to(torch.float64)
        self.energy_b += (b32 * b32).sum((0, 2)).to(torch.float64)
        self.count += int(a.shape[0])

    def cosine(self, cross: torch.Tensor | None = None, ea: torch.Tensor | None = None,
               eb: torch.Tensor | None = None) -> torch.Tensor:
        """Cosine of the concatenated (over positions) writes; a zero-energy vector set has cosine 0."""
        cross = self.cross if cross is None else cross
        ea = self.energy_a if ea is None else ea
        eb = self.energy_b if eb is None else eb
        denom = (ea[:, None] * eb[None, :]).sqrt()
        return torch.where(denom > 0, cross / torch.where(denom > 0, denom, torch.ones_like(denom)),
                           torch.zeros_like(cross))


def group_sum(m: torch.Tensor, group: int, dims: Sequence[int]) -> torch.Tensor:
    """Sum consecutive blocks of ``group`` entries along each of ``dims`` (heads -> their KV groups)."""
    for d in dims:
        shape = list(m.shape)
        if shape[d] % group:
            raise LayerMergeError(f"{shape[d]} entries do not divide into groups of {group}")
        shape[d:d + 1] = [shape[d] // group, group]
        m = m.reshape(shape).sum(d + 1)
    return m


# ---- block surgery (through the adapter) ---------------------------------------------------------------------------

def _params(adapter, block) -> dict:
    """The block's tensors by adapter role (the unit of folding, permuting and merging)."""
    ins, outs = adapter.stream_in_projections(block), adapter.stream_out_projections(block)
    p = {r: ins[r][0].weight for r in (*ATTN_IN_ROLES, *FFN_IN_ROLES)}
    p["attn_out"], p["ffn_out"] = outs["attn_out"].weight, outs["ffn_out"].weight
    p["attn_norm"], p["ffn_norm"] = adapter.attn_norm(block).weight, adapter.ffn_norm(block).weight
    for k, m in adapter.attention_subnorms(block).items():
        p[f"subnorm_{k}"] = m.weight
    for name, t in block.named_parameters():
        if all(t is not q for q in p.values()):
            raise LayerMergeError(f"block parameter {name} is not reached through the adapter's roles; merging would "
                                  "leave it unhandled")
    return p


def block_tensors(adapter, block) -> dict[str, torch.Tensor]:
    """The block's tensors by role, as float64 copies on their device."""
    return {k: v.detach().to(torch.float64).clone() for k, v in _params(adapter, block).items()}


def fold_norm_gains(adapter, t: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``t`` (role -> float64 tensor) with the attention and FFN norm gains folded into their readers and set to one.
    Which norm precedes which reader is the adapter's (`stream_in_projections`); this uses only the roles."""
    out = {k: v.clone() for k, v in t.items()}
    for r in ATTN_IN_ROLES:
        out[r] = t[r] * t["attn_norm"][None, :]
    for r in FFN_IN_ROLES:
        out[r] = t[r] * t["ffn_norm"][None, :]
    out["attn_norm"] = torch.ones_like(t["attn_norm"])
    out["ffn_norm"] = torch.ones_like(t["ffn_norm"])
    return out


def head_rows(head: int, head_dim: int) -> slice:
    return slice(head * head_dim, (head + 1) * head_dim)


def query_head_order(kv_perm: Sequence[int], q_perm_within: Sequence[Sequence[int]], group: int) -> list[int]:
    """Slot ``j*group + r`` takes source query head ``kv_perm[j]*group + q_perm_within[j][r]``."""
    return [int(kv_perm[j]) * group + int(q_perm_within[j][r]) for j in range(len(kv_perm)) for r in range(group)]


def permute_block(t: Mapping[str, torch.Tensor], *, ffn_perm: Sequence[int] | None, kv_perm: Sequence[int] | None,
                  q_perm_within: Sequence[Sequence[int]] | None, n_q: int, n_kv: int,
                  head_dim: int) -> dict[str, torch.Tensor]:
    """``t`` re-parameterized: FFN slot i takes neuron ``ffn_perm[i]``; KV slot j takes group ``kv_perm[j]`` and its
    query slot r the group's head ``q_perm_within[j][r]``. A function-preserving permutation (exact in real
    arithmetic). ``None`` keeps that part in place."""
    out = {k: v.clone() for k, v in t.items()}
    if ffn_perm is not None:
        idx = torch.as_tensor(list(ffn_perm), dtype=torch.long, device=t["gate"].device)
        if sorted(idx.tolist()) != list(range(t["gate"].shape[0])):
            raise LayerMergeError("ffn_perm is not a permutation of the FFN neurons")
        out["gate"], out["up"], out["ffn_out"] = t["gate"][idx], t["up"][idx], t["ffn_out"][:, idx]
    if kv_perm is not None:
        group = n_q // n_kv
        if sorted(int(x) for x in kv_perm) != list(range(n_kv)) or len(q_perm_within) != n_kv or any(
                sorted(int(x) for x in q) != list(range(group)) for q in q_perm_within):
            raise LayerMergeError("kv_perm / q_perm_within are not permutations of the KV groups / their heads")
        q_src = query_head_order(kv_perm, q_perm_within, group)
        dev = t["q"].device
        q_rows = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim, device=dev) for h in q_src])
        kv_rows = torch.cat([torch.arange(int(g) * head_dim, (int(g) + 1) * head_dim, device=dev) for g in kv_perm])
        out["q"] = t["q"][q_rows]
        out["k"], out["v"] = t["k"][kv_rows], t["v"][kv_rows]
        out["attn_out"] = t["attn_out"][:, q_rows]
    return out


def convex_weights(scores: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``scores`` ``[n_blocks, ...]`` (non-negative) to weights summing to one over blocks; a slot whose scores are all
    zero gets the uniform weights. Returns (weights, number of slots that fell back to uniform)."""
    if bool((scores < 0).any()) or not bool(torch.isfinite(scores).all()):
        raise LayerMergeError("convex_weights: scores must be finite and non-negative")
    total = scores.sum(0, keepdim=True)
    zero = total == 0
    w = torch.where(zero, torch.full_like(scores, 1.0 / scores.shape[0]),
                    scores / torch.where(zero, torch.ones_like(total), total))
    return w, int(zero.sum())


@dataclass(frozen=True)
class MergeWeights:
    """Per-slot convex weights over the span's blocks (first axis = block): ``ffn`` ``[n, I]``, ``query_head``
    ``[n, n_q]``, ``kv_group`` ``[n, n_kv]``, ``block`` ``[n]`` (the q/k sub-norms and any unfolded norm)."""
    ffn: torch.Tensor
    query_head: torch.Tensor
    kv_group: torch.Tensor
    block: torch.Tensor

    @staticmethod
    def uniform(n: int, n_ffn: int, n_q: int, n_kv: int, device=None) -> "MergeWeights":
        f = lambda m: torch.full((n, m), 1.0 / n, dtype=torch.float64, device=device)  # noqa: E731
        return MergeWeights(ffn=f(n_ffn), query_head=f(n_q), kv_group=f(n_kv),
                            block=torch.full((n,), 1.0 / n, dtype=torch.float64, device=device))


def merge_blocks(tensors: Sequence[Mapping[str, torch.Tensor]], w: MergeWeights, *, head_dim: int) -> dict:
    """One block's tensors (float64) from several aligned blocks: each slot the convex combination its weights give.
    FFN: gate/up rows and down columns by neuron; attention: q rows and o_proj columns by query head, k/v rows by KV
    group; norms and q/k sub-norms by block."""
    n = len(tensors)
    if any(x.shape[0] != n for x in (w.ffn, w.query_head, w.kv_group, w.block)):
        raise LayerMergeError(f"weights for {w.ffn.shape[0]} blocks, {n} blocks given")
    for x in (w.ffn, w.query_head, w.kv_group, w.block):
        if not torch.allclose(x.sum(0), torch.ones_like(x.sum(0)), atol=1e-12):
            raise LayerMergeError("merge weights do not sum to one over the blocks")
    rows = lambda m: m.repeat_interleave(head_dim, dim=-1)  # noqa: E731  head -> its head_dim rows
    out = {}
    for r in ("gate", "up"):
        out[r] = sum(w.ffn[i][:, None] * tensors[i][r] for i in range(n))
    out["ffn_out"] = sum(w.ffn[i][None, :] * tensors[i]["ffn_out"] for i in range(n))
    out["q"] = sum(rows(w.query_head[i])[:, None] * tensors[i]["q"] for i in range(n))
    out["attn_out"] = sum(rows(w.query_head[i])[None, :] * tensors[i]["attn_out"] for i in range(n))
    for r in ("k", "v"):
        out[r] = sum(rows(w.kv_group[i])[:, None] * tensors[i][r] for i in range(n))
    for k in tensors[0]:
        if k not in out:
            out[k] = sum(w.block[i] * tensors[i][k] for i in range(n))
    return out


@torch.no_grad()
def write_block(adapter, block, tensors: Mapping[str, torch.Tensor]) -> None:
    """Assign role tensors into ``block`` in place, cast once to each parameter's dtype and device."""
    p = _params(adapter, block)
    if set(p) != set(tensors):
        raise LayerMergeError(f"roles {sorted(set(p) ^ set(tensors))} differ between the block and the tensors")
    for k, param in p.items():
        if tuple(param.shape) != tuple(tensors[k].shape):
            raise LayerMergeError(f"{k}: {tuple(tensors[k].shape)} for a parameter of {tuple(param.shape)}")
        param.copy_(tensors[k].to(device=param.device, dtype=param.dtype))


def block_like(adapter, block, tensors: Mapping[str, torch.Tensor], *, dtype: torch.dtype | None = None):
    """A deep copy of ``block`` holding ``tensors`` (optionally in another dtype, e.g. float32 for a numerical check)."""
    b = copy.deepcopy(block)
    if dtype is not None:
        b = b.to(dtype)
    write_block(adapter, b, tensors)
    return b


# ---- update diagnostics --------------------------------------------------------------------------------------------

@dataclass
class UpdateMoments:
    """For a candidate update ``c`` against a reference update ``r`` at the same positions (``[T, d]``): sums of
    ``|r|^2``, ``|c|^2``, ``<c, r>`` and ``|c - r|^2``. `summary` gives the RMS norms, their ratio, the cosine and the
    normalized squared error ``E|c - r|^2 / E|r|^2``."""
    count: int = 0
    ref_sq: float = 0.0
    cand_sq: float = 0.0
    inner: float = 0.0
    err_sq: float = 0.0

    @torch.no_grad()
    def add(self, cand: torch.Tensor, ref: torch.Tensor) -> None:
        c, r = cand.to(torch.float64), ref.to(torch.float64)
        self.ref_sq += float((r * r).sum())
        self.cand_sq += float((c * c).sum())
        self.inner += float((c * r).sum())
        self.err_sq += float(((c - r) ** 2).sum())
        self.count += int(c.shape[0])

    def summary(self) -> dict:
        n = max(self.count, 1)
        rms_r, rms_c = (self.ref_sq / n) ** 0.5, (self.cand_sq / n) ** 0.5
        return {"positions": self.count, "rms_reference": rms_r, "rms_candidate": rms_c,
                "norm_ratio": (rms_c / rms_r) if rms_r > 0 else None,
                "cosine": (self.inner / (self.ref_sq * self.cand_sq) ** 0.5) if self.ref_sq * self.cand_sq > 0 else None,
                "normalized_error": (self.err_sq / self.ref_sq) if self.ref_sq > 0 else None}
