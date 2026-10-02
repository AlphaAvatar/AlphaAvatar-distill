"""Contribution-guided teacher depth selection.

The Stage 1 depth map decides which teacher blocks survive compression. The
canonical map (`sandwich.depth_span_map`) decides that by **position**: merge a
middle band pairwise, keep the ends 1:1. For 36 -> 28 it drops
`{5, 7, 9, 11, 13, 15, 17, 19}` — a choice justified by a single-axis ablation of
*where* the band sits, never by what the individual blocks compute.

This module replaces position with a **causal** measure: bypass a candidate set
of blocks through the residual path and ask how far the teacher's own output
distribution moves. Nothing here looks at hidden-state magnitude, activation
norm or layer index; those are properties of the representation, and a block can
carry a large residual delta while contributing almost nothing to the next-token
distribution (and the reverse).

Three deliberate design commitments
-----------------------------------

**Distributional distortion, not reconstruction error.** The objective is
`KL(teacher || teacher-with-S-bypassed)` over real prediction positions, in the
forward direction so that positions the intact teacher is confident about
dominate. Hidden-state distance would let a block that only rescales an
unread subspace look important.

**Iterative greedy, not one-shot Top-N.** Redundancy is conditional: two blocks
can each be individually removable because the other compensates, and removing
both is fatal. One-shot ranking cannot see that. `greedy_removal` re-scores every
surviving candidate against the *current* removal set in every round, which for
36 -> 28 is 36+35+...+29 = 260 subset evaluations. The full per-round table is
returned, not just the winners.

**Domain-balanced aggregation.** A token-weighted mean over a mixed calibration
corpus is a mean over whichever domain tokenizes longest. The primary score is
therefore the unweighted mean over domains of the unweighted mean over each
domain's sub-types of that sub-type's token-mean KL, so a domain's influence is
set by the design and not by its token count.

Everything in this module is pure or model-generic and is exercised on CPU with a
tiny random model; the expensive part is the caller's forward passes.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def _adapter_for(model):
    """The registered adapter for this model's family. Refuses if there is none.

    `bypassed_blocks` is called from four places, none of which hands in an
    adapter, so this resolves one rather than changing four signatures for a
    swap that is purely structural.

    There is deliberately no fallback to `model.model.layers`. A guess about
    where a family keeps its blocks is family knowledge living outside an
    adapter, and one that happened to be wrong would bypass the wrong modules
    and report a contribution measurement for a model nobody built.
    """
    from aadistill.initialization.specs.arch import adapter_for_config
    return adapter_for_config(model.config)


def _decoder_layers(model):
    return _adapter_for(model).blocks(model)


def _set_decoder_layers(model, blocks) -> None:
    """Replace the block list WITHOUT touching config.num_hidden_layers.

    Rewriting `num_hidden_layers` would change what the config hash and any
    downstream mask construction describe, for a swap that lasts only as long as
    a `with` block -- so the adapter is asked not to.
    """
    _adapter_for(model).set_blocks(model, blocks, update_config=False)


@contextmanager
def bypassed_blocks(model, skip: Collection[int]):
    """Temporarily bypass decoder blocks so the residual stream passes through.

    A decoder block is `x -> x + attn(norm(x)) + mlp(norm(x))`, so *removing* it
    from the module list is exactly bypassing it: the residual carries `x`
    unchanged to the next surviving block. That is the same operation the depth
    map performs at initialization, which is why the search and the construction
    agree by construction rather than by comment.

    Implemented by swapping `model.model.layers` for a filtered `ModuleList`.
    `Qwen3Model.forward` iterates `self.layers[: config.num_hidden_layers]` and
    reads `decoder_layer.attention_type` per layer, so a shorter list is honoured
    and each surviving block keeps its own attention type. `config` is left
    untouched: rewriting `num_hidden_layers` would also change what the config
    hash and any downstream mask construction describe.

    Requires `use_cache=False` (the caller's job): a KV cache is indexed by
    `layer_idx`, which no longer matches a filtered list.
    """
    layers = _decoder_layers(model)
    n = len(layers)
    skip = {int(i) for i in skip}
    bad = sorted(i for i in skip if not 0 <= i < n)
    if bad:
        raise ValueError(f"skip indices out of range for {n} layers: {bad}")
    if len(skip) >= n:
        raise ValueError(f"cannot bypass all {n} layers")
    if getattr(model.config, "use_cache", False):
        raise ValueError("bypassed_blocks requires config.use_cache=False")
    kept = [layers[i] for i in range(n) if i not in skip]
    _set_decoder_layers(model, kept)
    try:
        yield model
    finally:
        _set_decoder_layers(model, layers)


# --- distributional distortion -------------------------------------------------


@dataclass
class DistortionSums:
    """Unreduced accumulators, so a caller can pool positions its own way.

    Sums rather than means: the aggregation weights are a design decision made
    once at the top (`domain_balanced_score`), and a partially-averaged
    intermediate would quietly bake in token weighting.

    **`positions` counts, `weight` divides.** Under the incumbent all-positions
    policy the two are equal and either would do; under a position-weighted
    policy they are not, and only one of them is the denominator the estimand is
    defined with. So they are separate fields with separate jobs: `positions` is
    evidence about how many predictions were looked at, `weight` is `sum_t w_t`
    and is what every mean below is formed over.

    `weight = None` means "unweighted: the denominator is the position count".
    That is a STATED default rather than a coincidence, and it is what lets a
    hand-built oracle — the host-path copies in the device-equivalence test and
    the performance check — keep constructing these sums the way they always did.
    :func:`distortion` itself always sets it.
    """

    positions: int = 0
    #: `sum_t w_t`, or `None` for "the position count". The denominator.
    weight: float | None = None
    kl: float = 0.0
    reverse_kl: float = 0.0
    ref_ce: float = 0.0
    abl_ce: float = 0.0
    #: Weighted when the caller weights: `sum_t w_t * 1[argmax agrees]`. Float
    #: rather than int for that reason; it is exactly integral when unweighted.
    top1_agree: float = 0.0
    #: tag -> [kl_sum, weight, positions]. Three numbers because a weighted
    #: tagged mean divides by the tagged WEIGHT while a record still wants to
    #: state how many positions carried the tag.
    tagged: dict[str, list[float]] = field(default_factory=dict)

    @property
    def denominator(self) -> float:
        return float(self.positions) if self.weight is None else float(self.weight)

    def add_tagged(self, tag: str, kl_sum: float, weight: float,
                   positions: float | None = None) -> None:
        cur = self.tagged.setdefault(tag, [0.0, 0.0, 0.0])
        cur[0] += float(kl_sum)
        cur[1] += float(weight)
        #: Defaults to the weight, which is the unweighted case and is what the
        #: three-argument callers mean. A weighted caller passes both.
        cur[2] += float(weight if positions is None else positions)

    def merge(self, other: DistortionSums) -> None:
        self.positions += other.positions
        if other.weight is not None or self.weight is not None:
            #: Mixing a weighted part into an unweighted accumulator (or the
            #: reverse) must not silently drop either denominator, so an absent
            #: weight contributes its own position count — which is exactly what
            #: `None` means.
            self.weight = ((0.0 if self.weight is None else self.weight)
                           + (float(other.positions) if other.weight is None
                              else other.weight))
        self.kl += other.kl
        self.reverse_kl += other.reverse_kl
        self.ref_ce += other.ref_ce
        self.abl_ce += other.abl_ce
        self.top1_agree += other.top1_agree
        for tag, entry in other.tagged.items():
            kl_sum, weight = entry[0], entry[1]
            count = entry[2] if len(entry) > 2 else weight
            self.add_tagged(tag, kl_sum, weight, count)

    def as_dict(self) -> dict:
        n = self.positions
        if n == 0:
            raise ValueError("no positions accumulated")
        w = self.denominator
        if w <= 0:
            raise ValueError(
                f"{n} positions accumulated but their weights sum to {w}; a "
                "position set with no weight has no mean, and dividing by it "
                "would report one")
        out = {
            "positions": n,
            "weight": w,
            "kl": self.kl / w,
            "reverse_kl": self.reverse_kl / w,
            "ref_ce": self.ref_ce / w,
            "abl_ce": self.abl_ce / w,
            "ce_delta": (self.abl_ce - self.ref_ce) / w,
            "top1_agreement": self.top1_agree / w,
        }
        out["tagged"] = {
            #: Rounded rather than truncated: the count is an exact integer in
            #: float64 at these magnitudes and `int()` on `3.9999…` would
            #: silently lose one.
            tag: {"positions": int(round(entry[2] if len(entry) > 2
                                        else entry[1])),
                  "weight": entry[1],
                  "kl": (entry[0] / entry[1]) if entry[1] else None}
            for tag, entry in sorted(self.tagged.items())
        }
        return out


@torch.no_grad()
def forward_kl_mean(
    ref_logits: torch.Tensor,
    abl_logits: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    chunk: int = 512,
) -> float:
    """Mean forward KL(reference || ablated) over positions. Nothing else.

    ``weights`` is an optional ``[T_pred]`` non-negative vector; the result is
    then ``sum_t w_t k_t / sum_t w_t`` instead of the unweighted mean. ``None``
    takes the path this function has always taken, operation for operation —
    not a multiply by ``1.0``, because the frozen DEPTH decisions were produced
    by these exact lines.

    `distortion` computes six quantities for every (candidate, item) pair:
    forward KL, reverse KL, reference CE, ablated CE, top-1 agreement and the
    tagged KL sums. `depth.causal_kl_greedy_v1` reads exactly one of them --
    `sums["kl"]` -- and discards the rest, 260 candidate subsets x 67 items per
    expansion.

    What this drops, relative to `distortion`:

    * `q_log.exp()` and its reduction, the reverse-KL term;
    * two `gather`s over the vocabulary, the two cross-entropies. With no CE to
      compute, the TARGETS are not needed at all and are not taken;
    * two `argmax`es over the vocabulary, the top-1 agreement.

    What it preserves EXACTLY, because these are what make the result the same
    number rather than a similar one:

    * float32 `log_softmax`, computed on the same upcast inputs;
    * the same chunk loop and the same chunk boundaries -- measured to matter
      at ~9e-8 relative, so chunking is a real constraint and not a formality;
    * the same per-chunk float32 reduction accumulated in float64;
    * device residency, with one host transfer at the end.

    It reports a MEAN rather than sums because that is what the caller reads;
    `distortion` remains the function for anything that needs more than KL.
    """
    if ref_logits.shape != abl_logits.shape:
        raise ValueError(f"logit shape mismatch: {tuple(ref_logits.shape)} vs "
                         f"{tuple(abl_logits.shape)}")
    positions = int(ref_logits.shape[0])
    if positions == 0:
        raise ValueError("no positions to reduce")
    w = _position_weights(weights, positions, ref_logits.device, "forward_kl_mean")
    denominator = positions if w is None else float(w.sum())
    if denominator <= 0:
        raise ValueError(
            "every position has weight zero, so these logits have no weighted "
            "mean KL; an item the scoring policy cannot score must not reach a "
            "reducer")
    resident = _reduce_on_device(ref_logits.device)
    total = (torch.zeros((), dtype=torch.float64, device=ref_logits.device)
             if resident else 0.0)
    for a in range(0, positions, chunk):
        b = min(a + chunk, positions)
        p_log = F.log_softmax(ref_logits[a:b].float(), dim=-1)
        q_log = F.log_softmax(abl_logits[a:b].float(), dim=-1)
        per_pos = (p_log.exp() * (p_log - q_log)).sum(-1)
        #: Weighted BEFORE the chunk sum, so a zero-weight position contributes
        #: exactly zero rather than a value scaled afterwards. `w is None` skips
        #: the multiply entirely — see the docstring.
        if w is not None:
            per_pos = per_pos * w[a:b].to(per_pos.dtype)
        if resident:
            total += per_pos.sum().double()
        else:
            total += float(per_pos.sum())
    return (float(total.item()) if resident else float(total)) / denominator


@torch.no_grad()
def forward_kl_mean_batch(
    ref_logits: torch.Tensor,
    abl_logits: torch.Tensor,
    prediction_mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    chunk: int = 512,
) -> torch.Tensor:
    """Per-item mean forward KL over a padded batch. One scalar per ROW.

    ``ref_logits`` / ``abl_logits`` are ``[B, T_pred, V]`` and
    ``prediction_mask`` is ``[B, T_pred]`` bool over the positions each item
    really predicts. Returns ``[B]``, where ``out[i]`` is the mean forward KL of
    item ``i`` over **item i's own** valid positions — the same quantity
    :func:`forward_kl_mean` returns for that item alone.

    ``weights`` is an optional ``[B, T_pred]`` non-negative array of scoring
    weights. It is multiplied INTO the validity mask rather than replacing it:
    the mask says which positions exist, the weights say which of them the
    objective cares about, and a padded position must be zero under both. Each
    row is then ``sum_t m_t w_t k_t / sum_t m_t w_t`` — per row, so an item's
    weighting still cannot leak into a neighbour that shared its forward.

    **This is a per-item mean, not a pooled one, and the difference is the
    objective.** Reducing as ``(kl * mask).sum() / mask.sum()`` over the whole
    batch would be a token-weighted batch mean: an item with 1000 positions at
    KL 0.1 beside one with 100 at 0.5 would give ~0.136 instead of
    ``[0.1, 0.5]``, silently handing the long item ten times the influence. The
    callers here weight items by *subtype and domain*, deliberately and equally,
    so the per-row denominator is the whole point of the mask.

    **The numerical contract is :func:`forward_kl_mean`'s, unchanged:** float32
    ``log_softmax`` on upcast inputs, chunks along the SEQUENCE-POSITION axis at
    the same boundaries (0:512, 512:1024, …), a float32 per-chunk reduction,
    float64 accumulation, and accumulators kept where the logits are. Chunking
    is not a formality — it was measured to matter at ~9e-8 relative — and it
    walks the original position axis rather than a flattened list of valid
    positions, so every item keeps the boundaries it would have had alone
    regardless of how long its neighbours are.

    **And it bounds memory.** A single ``[B, T, V]`` softmax at a ~152k
    vocabulary is a large transient; chunking to ``[B, chunk, V]`` keeps the
    peak proportional to ``chunk`` rather than to the longest item in the batch.
    Batching exists to use the accelerator better, not to trade utilisation for
    an uncontrolled memory spike.

    The result stays on the logits' device. A caller that needs host floats
    converts once for the whole batch — one synchronisation instead of B.
    """
    if ref_logits.shape != abl_logits.shape:
        raise ValueError(f"logit shape mismatch: {tuple(ref_logits.shape)} vs "
                         f"{tuple(abl_logits.shape)}")
    if ref_logits.dim() != 3:
        raise ValueError(
            f"expected [B, T_pred, V] logits, got {tuple(ref_logits.shape)}; "
            "the scalar oracle is `forward_kl_mean`")
    rows, positions = int(ref_logits.shape[0]), int(ref_logits.shape[1])
    if prediction_mask.shape != (rows, positions):
        raise ValueError(
            f"prediction mask {tuple(prediction_mask.shape)} does not describe "
            f"logits {tuple(ref_logits.shape)[:2]}")
    if positions == 0:
        raise ValueError("no positions to reduce")
    mask = prediction_mask.to(ref_logits.device).bool()
    w = _position_weights(weights, positions, ref_logits.device,
                          "forward_kl_mean_batch", rows=rows)
    if w is None:
        #: The reference path, and `effective` is the BOOLEAN MASK ITSELF — so
        #: the per-chunk `effective[:, a:b].to(per_pos.dtype)` below is
        #: character for character the operation this loop always performed.
        effective = mask
        counts = mask.sum(dim=1).double()
    else:
        #: Masked AND weighted. The product is formed once here rather than per
        #: chunk, so the denominator and the numerator are provably the same
        #: vector — a row whose weights are zeroed by padding cannot end up with
        #: a count that padding did not zero.
        effective = (mask.to(w.dtype) * w)
        counts = effective.sum(dim=1).double()
    empty = (counts <= 0).nonzero().flatten().tolist()
    if empty:
        raise ValueError(
            f"rows {empty} carry no scoring weight over any valid prediction "
            "position; an item that predicts nothing the policy scores has no "
            "mean KL and must not reach this reducer")

    #: Same two accumulation paths the scalar oracle has, and for the same
    #: reason: the device branch is one no CPU-only test would otherwise reach,
    #: so `_reduce_on_device` stays a named seam a test can override to drive it
    #: with host tensors.
    resident = _reduce_on_device(ref_logits.device)
    #: float64 per ROW. The accumulator is [B] rather than a scalar precisely so
    #: that no two items' KL ever meet before their own means are formed.
    total = torch.zeros(rows, dtype=torch.float64,
                        device=ref_logits.device if resident else "cpu")
    for a in range(0, positions, chunk):
        b = min(a + chunk, positions)
        p_log = F.log_softmax(ref_logits[:, a:b].float(), dim=-1)
        q_log = F.log_softmax(abl_logits[:, a:b].float(), dim=-1)
        per_pos = (p_log.exp() * (p_log - q_log)).sum(-1)          # [B, chunk]
        #: Masked BEFORE the row sum, so a padded position contributes exactly
        #: zero rather than a garbage KL scaled by a garbage denominator. At
        #: `weights=None` `effective` IS the boolean mask cast to the logits'
        #: float dtype, so this is the multiply this line always performed.
        per_pos = per_pos * effective[:, a:b].to(per_pos.dtype)
        chunk_sum = per_pos.sum(dim=1).double()                    # [B]
        total += chunk_sum if resident else chunk_sum.cpu()
    return total / counts.to(total.device)


def _position_weights(weights: torch.Tensor | None, positions: int,
                      device: torch.device, where: str,
                      *, rows: int | None = None) -> torch.Tensor | None:
    """Validate a position-weight vector, or pass ``None`` straight through.

    One validator for all three reducers, because the failure it catches is the
    same in each: a weight vector of the wrong length would be broadcast or
    truncated by the arithmetic below and would silently reweight the objective
    rather than raise. The expected shape is ``[positions]`` for the scalar
    reducers and ``[rows, positions]`` for the batched one.
    """
    if weights is None:
        return None
    expected = (positions,) if rows is None else (rows, positions)
    if tuple(weights.shape) != expected:
        raise ValueError(
            f"{where}: position weights have shape {tuple(weights.shape)}, not "
            f"{expected}. A mis-shaped weight vector reweights the objective "
            "instead of failing, which is why this is checked rather than "
            "broadcast")
    if bool((weights < 0).any()):
        raise ValueError(f"{where}: negative position weight")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError(f"{where}: non-finite position weight")
    return weights.to(device)


def _reduce_on_device(device: torch.device) -> bool:
    """Whether to keep the accumulators where the logits are.

    A named function rather than an inline `device.type != "cpu"` for one
    reason: it is the only thing separating the two accumulation paths, and a
    branch that can only be taken on a machine with an accelerator is a branch
    no `$0` test executes. Four paid pods in this project have died inside lines
    no test had reached. A test overrides this to drive the device path with
    host tensors, which exercises the real accumulator code -- the arithmetic --
    while leaving the kernel question to the GPU validation that owns it.
    """
    return device.type != "cpu"


@torch.no_grad()
def distortion(
    ref_logits: torch.Tensor,
    abl_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tags: Mapping[str, torch.Tensor] | None = None,
    weights: torch.Tensor | None = None,
    chunk: int = 512,
) -> DistortionSums:
    """Accumulate teacher -> ablated-teacher distortion over prediction positions.

    `ref_logits` / `abl_logits` are `[T_pred, V]` already aligned to the
    positions being scored, and `targets` is `[T_pred]`. `tags` maps a diagnostic
    name to a boolean `[T_pred]` mask; tagged KL is reported alongside but has no
    standing to change a selection (see `greedy_removal`).

    `weights` is an optional `[T_pred]` non-negative scoring-weight vector. Every
    one of the six quantities is then weighted and the returned sums carry
    `sum_t w_t` as their denominator, including each tag's own. `None` performs
    the operations this function has always performed — the state-eval drift
    certification measured this path to 9.032e-06 of a 1e-5 budget, and a
    multiply by 1.0 is not something to spend the remainder on.

    **Tags and weights are different things and both apply.** A tag selects a
    diagnostic subset of positions; the weights say how much each position counts
    to the objective. A `think_close` position the policy does not supervise
    contributes to neither, which is why the tag accumulators take the product.

    Reduced in float32 chunks: the vocabulary is ~152k, and a full-sequence
    float32 softmax of both models at once is a needless memory spike on the one
    device that also has to hold the teacher.
    """
    if ref_logits.shape != abl_logits.shape:
        raise ValueError(f"logit shape mismatch: {tuple(ref_logits.shape)} vs "
                         f"{tuple(abl_logits.shape)}")
    if ref_logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets disagree on the position count")
    tags = dict(tags or {})
    for name, mask in tags.items():
        if mask.shape[0] != targets.shape[0]:
            raise ValueError(f"tag {name!r} mask has the wrong length")

    out = DistortionSums()
    device = ref_logits.device
    w = _position_weights(weights, int(targets.shape[0]), device, "distortion")
    #: Set unconditionally, so the denominator is never inferred. At
    #: `weights=None` it equals the position count, which is what the unweighted
    #: estimand divides by; a weighted call sets the real total. `DistortionSums`
    #: would fall back to the count on `None`, and that fallback exists only for
    #: the hand-built host oracles — this function does not rely on it.
    out.weight = float(targets.shape[0]) if w is None else float(w.sum())
    resident = _reduce_on_device(device)
    #: DEVICE-RESIDENT ACCUMULATORS when the logits are not on the host.
    #:
    #: The arithmetic is unchanged -- each chunk is still reduced in float32 and
    #: accumulated in float64, in the same order -- but on an accelerator the
    #: per-chunk `float(...)` calls were SYNCHRONISATION POINTS. Six of them per
    #: chunk, each stalling the pipeline until the kernel finished, on a loop
    #: that runs once per (candidate, item) pair. Keeping the accumulators on
    #: the device lets the chunks pipeline and returns one host transfer per
    #: call instead of `6 * ceil(T/chunk)`.
    #:
    #: float64 deliberately: the current path adds a float32 chunk sum into a
    #: Python float, which is a float64 accumulator. Accumulating in float32
    #: here would be a different -- and worse -- summation than the one this
    #: replaces, so the dtype follows the behaviour rather than the tensors.
    if resident:
        acc = torch.zeros(5, dtype=torch.float64, device=device)
        #: THREE slots per tag once weights exist: KL sum, weight, position
        #: count. The third used to be derivable from the second and is not any
        #: more, which is precisely the distinction a weighted tagged mean needs.
        tag_acc = {name: torch.zeros(3, dtype=torch.float64, device=device)
                   for name in tags}
    for a in range(0, ref_logits.shape[0], chunk):
        b = min(a + chunk, ref_logits.shape[0])
        p_log = F.log_softmax(ref_logits[a:b].float(), dim=-1)
        q_log = F.log_softmax(abl_logits[a:b].float(), dim=-1)
        p = p_log.exp()
        per_pos = (p * (p_log - q_log)).sum(-1)
        tg = targets[a:b]
        out.positions += int(b - a)
        #: The chunk's weights, cast to the reduction dtype once. `None` leaves
        #: every expression below exactly as it was written — that is the whole
        #: reason this is a branch and not a universal multiply.
        wc = None if w is None else w[a:b].to(per_pos.dtype)
        if resident:
            if wc is None:
                acc[0] += per_pos.sum().double()
                acc[1] += (q_log.exp() * (q_log - p_log)).sum(-1).sum().double()
                acc[2] += (-p_log.gather(1, tg[:, None]).sum()).double()
                acc[3] += (-q_log.gather(1, tg[:, None]).sum()).double()
                acc[4] += (p_log.argmax(-1) == q_log.argmax(-1)).sum().double()
            else:
                acc[0] += (per_pos * wc).sum().double()
                acc[1] += ((q_log.exp() * (q_log - p_log)).sum(-1)
                           * wc).sum().double()
                acc[2] += (-(p_log.gather(1, tg[:, None]).squeeze(1)
                             * wc).sum()).double()
                acc[3] += (-(q_log.gather(1, tg[:, None]).squeeze(1)
                             * wc).sum()).double()
                acc[4] += ((p_log.argmax(-1) == q_log.argmax(-1)).to(wc.dtype)
                           * wc).sum().double()
            for name, mask in tags.items():
                m = mask[a:b]
                if wc is None:
                    tag_acc[name][0] += per_pos[m].sum().double()
                    tag_acc[name][1] += m.sum().double()
                    tag_acc[name][2] += m.sum().double()
                else:
                    mw = m.to(wc.dtype) * wc
                    tag_acc[name][0] += (per_pos * mw).sum().double()
                    tag_acc[name][1] += mw.sum().double()
                    tag_acc[name][2] += m.sum().double()
        else:
            if wc is None:
                out.kl += float(per_pos.sum())
                out.reverse_kl += float((q_log.exp() * (q_log - p_log)).sum(-1).sum())
                out.ref_ce += float(-p_log.gather(1, tg[:, None]).sum())
                out.abl_ce += float(-q_log.gather(1, tg[:, None]).sum())
                out.top1_agree += int((p_log.argmax(-1) == q_log.argmax(-1)).sum())
            else:
                out.kl += float((per_pos * wc).sum())
                out.reverse_kl += float(((q_log.exp() * (q_log - p_log)).sum(-1)
                                         * wc).sum())
                out.ref_ce += float(-(p_log.gather(1, tg[:, None]).squeeze(1)
                                      * wc).sum())
                out.abl_ce += float(-(q_log.gather(1, tg[:, None]).squeeze(1)
                                      * wc).sum())
                out.top1_agree += float(
                    ((p_log.argmax(-1) == q_log.argmax(-1)).to(wc.dtype)
                     * wc).sum())
            for name, mask in tags.items():
                m = mask[a:b]
                k = int(m.sum())
                if not k:
                    continue
                if wc is None:
                    out.add_tagged(name, float(per_pos[m].sum()), k)
                else:
                    mw = m.to(wc.dtype) * wc
                    out.add_tagged(name, float((per_pos * mw).sum()),
                                   float(mw.sum()), k)
    if resident:
        #: ONE transfer, at the end. The only values that cross to the host are
        #: these reduced scalars -- never a `[T, V]` logit tensor.
        kl, rkl, ref_ce, abl_ce, top1 = acc.tolist()
        out.kl += kl
        out.reverse_kl += rkl
        out.ref_ce += ref_ce
        out.abl_ce += abl_ce
        #: Rounded to an integer ONLY when it is one. Under weights the agreement
        #: sum is a weighted count and rounding it would quantise the metric.
        out.top1_agree += int(round(top1)) if w is None else top1
        for name, triple in tag_acc.items():
            kl_sum, weight, count = triple.tolist()
            #: A tag with no matching position is OMITTED, exactly as the host
            #: path's `if k:` omits it -- an entry with zero positions would
            #: make `as_dict` report a tag the suite never saw. The test is the
            #: POSITION count, not the weight: a tag whose every position the
            #: policy zeroed was still present in the suite, and reporting it
            #: with `kl: None` says so where silence would not.
            if count:
                out.add_tagged(name, kl_sum,
                               weight if w is not None else int(round(count)),
                               int(round(count)))
    return out


# --- domain-balanced aggregation ----------------------------------------------


def domain_balanced_score(
    subtype_scores: Mapping[str, float],
    domains: Mapping[str, Sequence[str]],
) -> tuple[float, dict[str, float]]:
    """Mean over domains of the mean over each domain's sub-types.

    Two levels, both unweighted, so neither a long-tokenizing sub-type nor a
    domain that happens to own more sub-types can dominate. Every declared
    sub-type must be present: a silently missing sub-type would reweight the
    domain it belongs to.
    """
    if not domains:
        raise ValueError("no domains declared")
    per_domain: dict[str, float] = {}
    for domain, subtypes in domains.items():
        if not subtypes:
            raise ValueError(f"domain {domain!r} declares no sub-types")
        missing = [s for s in subtypes if s not in subtype_scores]
        if missing:
            raise ValueError(f"domain {domain!r} is missing sub-types {missing}")
        per_domain[domain] = sum(float(subtype_scores[s]) for s in subtypes) / len(subtypes)
    primary = sum(per_domain.values()) / len(per_domain)
    return primary, per_domain


# --- iterative greedy removal -------------------------------------------------


def greedy_removal(
    score_fn,
    n_layers: int,
    n_remove: int,
    *,
    protect: Collection[int] = (),
    completed_rounds: Iterable[dict] | None = None,
    on_round=None,
    on_candidate=None,
) -> dict:
    """Remove `n_remove` blocks one at a time, re-scoring survivors every round.

    `score_fn(frozenset_of_skipped) -> float` is the preregistered objective;
    lower is less damaging. Each round evaluates every surviving candidate
    against the *current* removal set and commits the argmin. Ties are broken by
    the smaller layer index — stated here because a tie-break invented after
    seeing a table is a selection rule chosen on the outcome.

    `protect` excludes layers from removal. It defaults to empty: constraining
    the search by position is the assumption this module exists to test.

    `completed_rounds` resumes a partially finished search from previously
    written round records (the search is ~260 model evaluations; losing it to a
    pod restart is avoidable). Resumed rounds are replayed, not re-scored, and
    their recorded choice is trusted — the caller is responsible for only
    passing records produced by the same objective.

    `on_round(record)` fires when a round commits; `on_candidate(progress)` fires
    after every candidate is scored. **Neither can change a decision** — they are
    called with what has already been computed and their return value is
    discarded — but `on_candidate` may *raise*, which is how a wall-clock
    deadline stops the search inside an expansion rather than after it. Both
    exist because attempt 10 spent 10 h 47 m in one expansion emitting nothing:
    a round record is written only when a round completes, and a round is 29-36
    model evaluations, so between rounds the search is silent for hours.
    """
    if not 0 <= n_remove < n_layers:
        raise ValueError(f"cannot remove {n_remove} of {n_layers} layers")
    protect = {int(i) for i in protect}
    if len(protect) > n_layers - n_remove:
        raise ValueError("protect set leaves too few removable layers")

    skipped: list[int] = []
    rounds: list[dict] = []
    evaluations = 0

    for record in completed_rounds or ():
        chosen = int(record["chosen"])
        if chosen in skipped:
            raise ValueError(f"resumed round re-removes layer {chosen}")
        if chosen in protect:
            raise ValueError(f"resumed round removes protected layer {chosen}")
        skipped.append(chosen)
        rounds.append(dict(record, resumed=True))
        if len(skipped) > n_remove:
            raise ValueError("resumed more rounds than the search asks for")

    while len(skipped) < n_remove:
        candidates = [i for i in range(n_layers)
                      if i not in skipped and i not in protect]
        table = []
        for c in candidates:
            score = float(score_fn(frozenset(skipped + [c])))
            if not math.isfinite(score):
                raise ValueError(f"objective returned {score} for candidate {c}")
            evaluations += 1
            table.append({"candidate": c, "score": score})
            if on_candidate is not None:
                # After the append, so the observer sees the same table the
                # decision below will see. Its return value is discarded; only an
                # exception it raises can affect control flow, and that stops the
                # search rather than steering it.
                on_candidate({"round": len(skipped), "candidate": c,
                              "score": score, "index": len(table),
                              "of": len(candidates),
                              "evaluations": evaluations})
        best = min(table, key=lambda r: (r["score"], r["candidate"]))
        record = {
            "round": len(skipped),
            "removed_before": list(skipped),
            "chosen": best["candidate"],
            "chosen_score": best["score"],
            "n_candidates": len(table),
            # Ordered by index, not by score: a table sorted by the outcome
            # invites reading a ranking that the greedy rule never used.
            "table": table,
        }
        skipped.append(best["candidate"])
        rounds.append(record)
        if on_round is not None:
            on_round(record)

    kept = [i for i in range(n_layers) if i not in skipped]
    return {
        "n_layers": n_layers,
        "n_remove": n_remove,
        "protect": sorted(protect),
        "removed": sorted(skipped),
        "removal_order": list(skipped),
        "kept": kept,
        "rounds": rounds,
        "evaluations": evaluations,
    }


def expected_evaluations(n_layers: int, n_remove: int, n_protected: int = 0) -> int:
    """Subset evaluations a full greedy search performs — 260 for 36 -> 28."""
    return sum(n_layers - n_protected - r for r in range(n_remove))
