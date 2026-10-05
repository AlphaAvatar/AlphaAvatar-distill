"""Which vocabulary entries a divergence is reduced over.

Two supports, and the choice is part of what a measurement MEANS rather than how
fast it runs:

``full_vocab_v1``
    Every vocabulary entry. :func:`aadistill.initialization.statistics.contribution.distortion`
    implements it and remains the ORACLE — it is not superseded and not deleted.

``reference_topk_tail_v1(top_k=N)``
    The Top-N entries of the **reference** distribution at each prediction
    position, plus one aggregate ``K+1`` bucket holding every entry outside that
    support.

For reference ``p`` and compared distribution ``q``, with ``S = TopK(p, K)``::

    p_tail = 1 - sum_{i in S} p_i
    q_tail = 1 - sum_{i in S} q_i

    KL = sum_{i in S} p_i * (log p_i - log q_i)
         + p_tail * (log p_tail - log q_tail)

**The support is REFERENCE-defined, and that is a semantic commitment, not an
implementation detail.** The candidate never chooses its own Top-K. Forward KL
``KL(p || q)`` weights each term by ``p_i``, so the entries that matter are the
ones ``p`` puts mass on; letting ``q`` pick the support would weight terms by
mass ``p`` may not have and would make two candidates incomparable, because each
would be scored on its own partition. The consequence to accept is that a
candidate which moves mass onto entries outside ``S`` has that movement counted
only through the single tail bucket.

**This is not mathematically identical to full-vocabulary KL.** It is a coarsening
of the partition, so by the data-processing inequality it is a LOWER bound on the
full KL for the same pair of distributions. Anything that needs to tell the two
apart therefore has to carry the support in its identity — see
:class:`aadistill.initialization.scoring.protocol_identity.ReductionSemantics`,
whose ``as_dict`` omits the field entirely under the historical full-vocab
contract so that no existing ``measurement_protocol_id`` recomputes differently.

``top_k`` is always supplied by the caller. It is an experiment-policy parameter,
not a framework default, and this module contains no value for it.

**The tail is computed from the complement's own logits, never by subtracting
the support's mass from one.** An independent review caught the subtraction as a
real numerical-semantic defect, and the evidence was already in our own report: on
real logits the measured Top-200 support mass reached ``1.000001``, a probability
above one, so the complement was below float32 resolution and one per-item score
had the coarse KL EXCEED the full-vocabulary KL by 2.2e-03 relative. A coarsening
cannot do that. Masking the support and reducing the rest has no cancellation, and
the partition's two masses then sum to one by construction.

**A zero candidate tail against a non-zero reference tail is ``+inf``, and stays
``+inf``.** Only zero REFERENCE mass makes a forward-KL term vanish. The earlier
code dropped the term whenever either mass rounded to zero, which silently turned
an infinite divergence into a finite and favourable score.

Implementation notes taken from ``lasgroup/SDPO`` @
``7c457fc1b1f636ae794eb0362ba37d4743b06fbc``: normalized top-k log
probabilities, the optional aggregate tail bucket, and gathering the compared
distribution on one fixed support. Its support policy — the *student's* top-k —
is deliberately NOT adopted, for the reason above, and its stable-complement
formulation is superseded here by the complement logsumexp, which needs no
stabilisation because it never cancels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from aadistill.initialization.statistics.contribution import (
    DistortionSums, _position_weights, _reduce_on_device,
)

SUPPORT_FULL_VOCAB_V1 = "full_vocab_v1"
SUPPORT_REFERENCE_TOPK_TAIL_V1 = "reference_topk_tail_v1"


@dataclass(frozen=True)
class DistributionSupport:
    """The entries a divergence is reduced over, as an identity-bearing value.

    ``as_dict`` is what goes into a hash, so it is deliberately minimal and it
    OMITS ``top_k`` under the full-vocabulary contract: a key present with a
    ``None`` value would still change the serialization, and the whole point of
    the optional field is that historical identities are byte-identical.
    """

    support_id: str
    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.support_id == SUPPORT_FULL_VOCAB_V1:
            if self.top_k is not None:
                raise ValueError(
                    f"{SUPPORT_FULL_VOCAB_V1} reduces over every entry and takes "
                    f"no top_k; got {self.top_k!r}")
            return
        if self.support_id != SUPPORT_REFERENCE_TOPK_TAIL_V1:
            raise ValueError(
                f"unknown distribution support {self.support_id!r}; known: "
                f"{SUPPORT_FULL_VOCAB_V1!r}, "
                f"{SUPPORT_REFERENCE_TOPK_TAIL_V1!r}")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise ValueError(
                f"{SUPPORT_REFERENCE_TOPK_TAIL_V1} needs an integer top_k, got "
                f"{self.top_k!r}. It is an experiment-policy parameter and this "
                "module has no default for it.")
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")

    @property
    def is_full_vocab(self) -> bool:
        return self.support_id == SUPPORT_FULL_VOCAB_V1

    def as_dict(self) -> dict[str, Any]:
        if self.is_full_vocab:
            return {"support_id": self.support_id}
        return {"support_id": self.support_id, "top_k": int(self.top_k)}

    def __str__(self) -> str:
        return (self.support_id if self.is_full_vocab
                else f"{self.support_id}(top_k={self.top_k})")


#: The historical contract. Named so callers can be explicit rather than passing
#: `None` and relying on a default that may change meaning later.
FULL_VOCAB_V1 = DistributionSupport(SUPPORT_FULL_VOCAB_V1)


def reference_topk_tail(top_k: int) -> DistributionSupport:
    """``reference_topk_tail_v1`` at a caller-supplied ``top_k``."""
    return DistributionSupport(SUPPORT_REFERENCE_TOPK_TAIL_V1, top_k=int(top_k))


# --- numerics ---------------------------------------------------------------

def owned_float32(logits: torch.Tensor) -> torch.Tensor:
    """An fp32 copy this module may destroy.

    `.float()` RETURNS THE SAME TENSOR when the input is already fp32, so masking
    the support in place would mutate the caller's logits -- the reference block a
    search has cached, or the candidate block the operator is still holding. The
    copy is explicit for that reason, not defensively.
    """
    return logits.to(torch.float32, copy=True)


def complement_log_mass(z_owned: torch.Tensor, support: torch.Tensor,
                        log_z: torch.Tensor) -> torch.Tensor:
    """``log P(outside support)``, from the complement's own logits.

    DESTROYS ``z_owned``: the support entries are set to ``-inf`` and the
    remaining row is reduced. The caller must own it.

    This replaces ``log(1 - sum of the support's probabilities)``, which is the
    defect an independent review caught. Reconstructing the tail by subtraction
    crosses float32 resolution whenever the support holds nearly all the mass:
    the measured Top-200 support mass reached **1.000001** on real logits, a
    probability above one, and from there the complement is noise. One per-item
    score consequently had the coarse KL EXCEED the full-vocabulary KL by
    2.2e-03 relative, which a coarsening cannot do.

    The complement logits already exist in the row. Reducing them directly has no
    cancellation at all, and it is exact in the only sense that matters here: the
    K+1 partition's two masses are computed by the same ``logsumexp`` against the
    same normalizer, so they sum to one by construction rather than by luck.
    """
    z_owned.scatter_(-1, support, float("-inf"))
    return torch.logsumexp(z_owned, dim=-1) - log_z


# --- the reference sketch ---------------------------------------------------

@dataclass(frozen=True)
class ReferenceDistributionSketch:
    """Everything a ``K+1`` reduction needs from the reference, and nothing more.

    The persistent reference state is ``O(T*K)`` rather than ``O(T*V)``. Shapes,
    with ``T`` prediction positions:

    ======================  ==========  ==================================
    ``support_indices``     ``[T, K]``  reference Top-K entries, descending
    ``support_log_probs``   ``[T, K]``  their normalized log probabilities
    ``tail_log_prob``       ``[T]``     ``log(1 - sum of the above)``
    ``top1_token``          ``[T]``     reference argmax, for exact top-1
    ``target_log_prob``     ``[T]``     exact reference CE at the gold token
    ======================  ==========  ==================================

    ``top1_token`` and ``target_log_prob`` are carried so that top-1 agreement and
    CE stay EXACT under the coarsened partition: a gold token does not have to be
    inside the Top-K, and a metric that silently became approximate because the
    KL partition changed would be a second, unannounced protocol change.
    """

    support_indices: torch.Tensor
    support_log_probs: torch.Tensor
    tail_log_prob: torch.Tensor
    top1_token: torch.Tensor
    target_log_prob: torch.Tensor
    top_k: int
    vocab_size: int

    def __post_init__(self) -> None:
        t, k = self.support_indices.shape
        if k != self.top_k:
            raise ValueError(
                f"support_indices has width {k} but top_k is {self.top_k}")
        for name, want in (("support_log_probs", (t, k)),
                           ("tail_log_prob", (t,)),
                           ("top1_token", (t,)),
                           ("target_log_prob", (t,))):
            got = tuple(getattr(self, name).shape)
            if got != want:
                raise ValueError(f"{name} is {got}, expected {want}")

    @property
    def positions(self) -> int:
        return int(self.support_indices.shape[0])

    def bytes_held(self) -> int:
        return sum(t.numel() * t.element_size() for t in (
            self.support_indices, self.support_log_probs, self.tail_log_prob,
            self.top1_token, self.target_log_prob))


def sketch_reference(logits: torch.Tensor, targets: torch.Tensor, *,
                     top_k: int, chunk: int = 512,
                     ) -> ReferenceDistributionSketch:
    """Reduce ``[T, V]`` reference logits to an ``O(T*K)`` sketch.

    ``logits`` is ``[T_pred, V]`` already aligned to the positions being scored
    and ``targets`` is ``[T_pred]``. The vocabulary comes from the tensor, never
    from a constant: this must work for any model this framework loads.

    Chunked in float32 for the same reason the full reducer is — a full-sequence
    float32 softmax at a six-figure vocabulary is a needless spike on the one
    device that also holds the model.

    ``top_k`` larger than the vocabulary is clamped to the vocabulary, which makes
    the sketch lossless and the resulting KL equal to the full-vocabulary KL. That
    is the identity the correctness tests pin, so it is a supported input rather
    than an error.
    """
    if logits.ndim != 2:
        raise ValueError(f"reference logits must be [T, V], got "
                         f"{tuple(logits.shape)}")
    t_pred, vocab = int(logits.shape[0]), int(logits.shape[1])
    if int(targets.shape[0]) != t_pred:
        raise ValueError("logits and targets disagree on the position count")
    if t_pred == 0:
        raise ValueError("a sketch over zero positions measures nothing")
    k = max(1, min(int(top_k), vocab))

    idx = torch.empty((t_pred, k), dtype=torch.long, device=logits.device)
    lp = torch.empty((t_pred, k), dtype=torch.float32, device=logits.device)
    tail = torch.empty((t_pred,), dtype=torch.float32, device=logits.device)
    top1 = torch.empty((t_pred,), dtype=torch.long, device=logits.device)
    tgt = torch.empty((t_pred,), dtype=torch.float32, device=logits.device)

    has_tail = k < vocab
    for start in range(0, t_pred, max(1, int(chunk))):
        stop = min(start + max(1, int(chunk)), t_pred)
        #: OWNED, because the tail is computed by masking this tensor below.
        block = owned_float32(logits[start:stop])
        #: ONE log-normalizer over the FULL vocabulary. Every log probability
        #: below is normalized against it, including the gold token's -- so CE
        #: stays exact whether or not the gold token is in the Top-K.
        log_z = torch.logsumexp(block, dim=-1, keepdim=True)
        top = torch.topk(block, k, dim=-1, largest=True, sorted=True)
        idx[start:stop] = top.indices
        lp[start:stop] = top.values - log_z
        top1[start:stop] = top.indices[:, 0]
        #: EVERYTHING THAT NEEDS THE UNMODIFIED ROW IS CAPTURED FIRST. The gold
        #: token may be inside the support, so its logit must be read before the
        #: support is masked out.
        tgt[start:stop] = (
            block.gather(1, targets[start:stop].view(-1, 1).to(block.device))
            .squeeze(1) - log_z.squeeze(-1))
        #: Only now: the tail from the complement's own logits. `block` is
        #: destroyed by this and is not read again.
        if has_tail:
            tail[start:stop] = complement_log_mass(
                block, top.indices, log_z.squeeze(-1))
        else:
            #: K >= V: the support IS the vocabulary and there is structurally no
            #: tail bucket. `-inf` records an empty one; no consumer reads it,
            #: because `has_tail` is false everywhere downstream.
            tail[start:stop] = float("-inf")
        del block, log_z, top

    return ReferenceDistributionSketch(
        support_indices=idx, support_log_probs=lp, tail_log_prob=tail,
        top1_token=top1, target_log_prob=tgt, top_k=k, vocab_size=vocab)


# --- the K+1 reduction ------------------------------------------------------

def distortion_on_reference_support(
    sketch: ReferenceDistributionSketch,
    cand_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tags: Mapping[str, torch.Tensor] | None = None,
    weights: torch.Tensor | None = None,
    chunk: int = 512,
) -> DistortionSums:
    """The six quantities, reduced on the reference's ``K+1`` partition.

    Same output contract as the full-vocabulary
    :func:`~aadistill.initialization.statistics.contribution.distortion`, so every
    consumer — ``greedy_removal``, the state evaluator, the domain-balanced
    aggregation — works unchanged. Which quantities change meaning and which do
    not is the whole point:

    ``kl``, ``reverse_kl``, ``tagged``
        on the ``K+1`` partition. **Both directions on the SAME partition**: one
        state evaluation carrying a reference-defined forward KL and a
        candidate-defined reverse KL would be two incompatible partitions in one
        number.
    ``ref_ce``, ``abl_ce``, ``top1_agree``
        EXACT. CE uses the exact target logit against the full log-normalizer,
        and top-1 compares true argmaxes. A gold token outside the Top-K is
        ordinary, not a special case.

    ``weights`` and ``tags`` keep the semantics they have in the full reducer: a
    tag selects a diagnostic subset of positions, the weights say how much each
    position counts, and a tagged accumulator takes the product.
    """
    if cand_logits.ndim != 2:
        raise ValueError(f"candidate logits must be [T, V], got "
                         f"{tuple(cand_logits.shape)}")
    t_pred = sketch.positions
    if int(cand_logits.shape[0]) != t_pred:
        raise ValueError(
            f"the sketch covers {t_pred} positions and the candidate logits "
            f"{int(cand_logits.shape[0])}; they do not describe the same "
            "positions, so no divergence between them exists")
    if int(cand_logits.shape[1]) != sketch.vocab_size:
        raise ValueError(
            f"the sketch was built at vocabulary {sketch.vocab_size} and the "
            f"candidate has {int(cand_logits.shape[1])}; the two models are not "
            "logit-comparable")
    if int(targets.shape[0]) != t_pred:
        raise ValueError("targets and the sketch disagree on the position count")

    tags = dict(tags or {})
    for name, mask in tags.items():
        if int(mask.shape[0]) != t_pred:
            raise ValueError(f"tag {name!r} mask has the wrong length")

    device = cand_logits.device
    out = DistortionSums()
    w = _position_weights(weights, t_pred, device, "distortion_topk")
    out.weight = float(t_pred) if w is None else float(w.sum())
    resident = _reduce_on_device(device)
    acc = torch.zeros(5, dtype=torch.float64, device=device)
    tag_acc = {name: torch.zeros(3, dtype=torch.float64, device=device)
               for name in tags}

    has_tail = sketch.top_k < sketch.vocab_size
    step = max(1, int(chunk))
    for start in range(0, t_pred, step):
        stop = min(start + step, t_pred)
        #: OWNED: the candidate's tail is computed by masking this tensor.
        cand = owned_float32(cand_logits[start:stop])
        #: The candidate's OWN full-vocabulary normalizer. Not optional: its
        #: probabilities on the reference support only mean anything relative to
        #: its whole distribution.
        cand_log_z = torch.logsumexp(cand, dim=-1, keepdim=True)

        p_log = sketch.support_log_probs[start:stop].to(device).float()
        p_tail_log = sketch.tail_log_prob[start:stop].to(device).float()
        support = sketch.support_indices[start:stop].to(device)
        #: GATHERED ON THE REFERENCE SUPPORT. The candidate does not choose it.
        q_log = cand.gather(1, support) - cand_log_z
        tgt = targets[start:stop].to(device)
        #: Captured before the mask, since the gold token may be in the support.
        abl_ce = -(cand.gather(1, tgt.view(-1, 1)).squeeze(1)
                   - cand_log_z.squeeze(-1))
        cand_argmax = cand.argmax(dim=-1)
        #: The candidate's tail from ITS OWN complement logits, against the
        #: reference support. `cand` is destroyed here and not read again.
        q_tail_log = (complement_log_mass(cand, support,
                                          cand_log_z.squeeze(-1))
                      if has_tail else None)

        p = p_log.exp()
        q = q_log.exp()

        #: Forward KL on the K+1 partition. `_term` is the convention that makes
        #: the empty-bucket limits right: a bucket with zero reference mass
        #: contributes zero however small the candidate's mass is, which is
        #: `0 * log(0/q) = 0`; a bucket with reference mass and ZERO candidate
        #: mass is a genuine infinity and is reported as one rather than hidden
        #: behind a floor, because silently clamping it would make an impossible
        #: candidate look merely bad.
        kl_support = _term(p, p_log, q_log).sum(dim=-1)
        rkl_support = _term(q, q_log, p_log).sum(dim=-1)
        if not has_tail:
            #: K >= V: the partition is the vocabulary and has no tail bucket.
            kl, rkl = kl_support, rkl_support
        else:
            #: THE TAIL TERM IS NOT CONDITIONAL. For finite logits and K < V a
            #: mathematical complement exists, and the only thing that makes the
            #: forward term vanish is zero REFERENCE mass -- which `_term`
            #: handles. If the candidate has zero complement mass while the
            #: reference does not, forward KL is `+inf` and is PRESERVED: that is
            #: a candidate assigning zero probability to something the reference
            #: does, and reporting it as finite would be a false measurement.
            p_tail, q_tail = p_tail_log.exp(), q_tail_log.exp()
            kl = kl_support + _term(p_tail, p_tail_log, q_tail_log)
            #: Reverse KL on the SAME partition, roles swapped.
            rkl = rkl_support + _term(q_tail, q_tail_log, p_tail_log)

        ref_ce = -sketch.target_log_prob[start:stop].to(device).float()
        agree = (cand_argmax
                 == sketch.top1_token[start:stop].to(device)).float()

        block_w = None if w is None else w[start:stop]
        if block_w is None:
            acc[0] += kl.sum().double()
            acc[1] += rkl.sum().double()
            acc[2] += ref_ce.sum().double()
            acc[3] += abl_ce.sum().double()
            acc[4] += agree.sum().double()
        else:
            acc[0] += (kl * block_w).sum().double()
            acc[1] += (rkl * block_w).sum().double()
            acc[2] += (ref_ce * block_w).sum().double()
            acc[3] += (abl_ce * block_w).sum().double()
            acc[4] += (agree * block_w).sum().double()

        for name, mask in tags.items():
            m = mask[start:stop].to(device)
            if block_w is None:
                tag_acc[name][0] += (kl * m).sum().double()
                tag_acc[name][1] += m.sum().double()
            else:
                tag_acc[name][0] += (kl * m * block_w).sum().double()
                tag_acc[name][1] += (m * block_w).sum().double()
            tag_acc[name][2] += m.sum().double()

        del cand, cand_log_z, p_log, p_tail_log, support, q_log, kl, rkl

    host = acc.cpu() if resident else acc
    out.positions = t_pred
    out.kl = float(host[0])
    out.reverse_kl = float(host[1])
    out.ref_ce = float(host[2])
    out.abl_ce = float(host[3])
    out.top1_agree = float(host[4])
    for name, a in tag_acc.items():
        h = a.cpu() if resident else a
        out.tagged[name] = [float(h[0]), float(h[1]), float(h[2])]
    return out


def _term(p: torch.Tensor, p_log: torch.Tensor,
          q_log: torch.Tensor) -> torch.Tensor:
    """``p * (log p - log q)``, with ``p == 0`` contributing exactly zero.

    Needed because an empty bucket gives ``0 * (-inf - x)``, which is NaN in
    floating point and zero in the limit. Masking the multiply rather than
    flooring ``p`` keeps a bucket with real reference mass and no candidate mass
    as ``+inf``, which is the truthful answer for a candidate that assigns zero
    probability to something the reference does.
    """
    empty = p <= 0
    diff = torch.where(empty, torch.zeros_like(p_log), p_log - q_log)
    return torch.where(empty, torch.zeros_like(p), p * diff)


# --- DEPTH's reducer: forward KL only, on the reference support --------------

def sketch_forward_kl(support_indices: torch.Tensor,
                      support_log_probs: torch.Tensor,
                      tail_log_prob: torch.Tensor,
                      cand_logits: torch.Tensor,
                      *, has_tail: bool) -> torch.Tensor:
    """Per-position forward KL on the reference's ``K+1`` partition.

    Generic over leading dimensions: ``support_indices`` and
    ``support_log_probs`` are ``[..., K]``, ``tail_log_prob`` is ``[...]`` and
    ``cand_logits`` is ``[..., V]``; the result is ``[...]``. ONE implementation
    serves the per-item and the batched DEPTH paths, so the two cannot drift.

    ``has_tail`` is the caller's and is not inferred from the sketch. A tail value
    exists in the sketch either way, and only the caller knows whether
    ``top_k >= vocab_size`` — trusting the stored value is how a partition that
    should have no tail acquires a spurious one.

    Forward KL ONLY. DEPTH reads exactly this quantity and discards the other
    five, 260 candidate subsets x 67 items per expansion, which is why it does
    not come through the six-quantity reducer.
    """
    #: OWNED: the tail is computed by masking this tensor, and `.float()` would
    #: alias a caller-owned fp32 block -- the reference a search has cached or the
    #: candidate the operator still holds.
    cand = owned_float32(cand_logits)
    log_z = torch.logsumexp(cand, dim=-1, keepdim=True)
    q_log = cand.gather(-1, support_indices) - log_z
    p_log = support_log_probs.float()

    kl = _term(p_log.exp(), p_log, q_log).sum(dim=-1)
    if not has_tail:
        #: K >= V: no tail bucket exists.
        return kl
    p_tail_log = tail_log_prob.float()
    #: The candidate's tail from ITS OWN complement logits on the reference
    #: support. NOT `1 - sum(support)`, which crosses float32 resolution once the
    #: support holds nearly all the mass and is what let a coarse KL exceed the
    #: full one. `cand` is destroyed here.
    q_tail_log = complement_log_mass(cand, support_indices, log_z.squeeze(-1))
    #: UNCONDITIONAL. Only zero REFERENCE mass makes the forward term vanish, and
    #: `_term` is what decides that. A candidate with zero complement mass against
    #: a non-zero reference tail is `+inf`, and that is preserved.
    return kl + _term(p_tail_log.exp(), p_tail_log, q_tail_log)


def sketch_forward_kl_mean(sketch: ReferenceDistributionSketch,
                           cand_logits: torch.Tensor,
                           *, weights: torch.Tensor | None = None) -> float:
    """``forward_kl_mean``'s quantity, on the reference support. One item."""
    kl = sketch_forward_kl(
        sketch.support_indices.to(cand_logits.device),
        sketch.support_log_probs.to(cand_logits.device),
        sketch.tail_log_prob.to(cand_logits.device),
        cand_logits, has_tail=sketch.top_k < sketch.vocab_size)
    if weights is None:
        return float(kl.mean())
    w = weights.to(kl.device).float()
    total = float(w.sum())
    if total <= 0:
        raise ValueError(
            "every scoring weight is zero, so the weighted mean has no "
            "denominator; a zero-weight item cannot contribute a score")
    return float((kl * w).sum() / total)


def sketch_forward_kl_mean_batch(support_indices: torch.Tensor,
                                 support_log_probs: torch.Tensor,
                                 tail_log_prob: torch.Tensor,
                                 cand_logits: torch.Tensor,
                                 prediction_mask: torch.Tensor,
                                 *, has_tail: bool,
                                 weights: torch.Tensor | None = None,
                                 ) -> torch.Tensor:
    """``forward_kl_mean_batch``'s quantity, on the reference support.

    ``[B]`` out, where ``out[i]`` is item ``i``'s mean over **item i's own** valid
    positions — a PER-ITEM mean, never a pooled one. Pooling would hand a
    1000-position item ten times the influence of a 100-position one and change
    the objective, which is the same reason the full-vocabulary batched reducer
    reduces per row.

    ``weights`` multiplies INTO the validity mask rather than replacing it: the
    mask says which positions exist, the weights say which the objective cares
    about, and a padded position must be zero under both.
    """
    kl = sketch_forward_kl(support_indices, support_log_probs, tail_log_prob,
                           cand_logits, has_tail=has_tail)
    m = prediction_mask.to(kl.device).float()
    if weights is not None:
        m = m * weights.to(kl.device).float()
    denom = m.sum(dim=-1)
    if bool((denom <= 0).any()):
        raise ValueError(
            "a row has no weighted valid position, so its mean has no "
            "denominator; an item that contributes nothing must not be scored")
    #: MASK BEFORE MULTIPLY, and never `nan_to_num`. A padded or zero-weight
    #: position holds whatever the pad token produced and must contribute zero; a
    #: VALID position holding `+inf` must stay `+inf`, because that is a candidate
    #: assigning zero probability to something the reference does. The previous
    #: `nan_to_num(posinf=0.0)` erased exactly that distinction and could turn a
    #: genuine infinite divergence into a finite -- and favourable -- score.
    effective = m
    safe = torch.where(effective > 0, kl, torch.zeros_like(kl))
    #: float64 to match `forward_kl_mean_batch`, which this stands in for; the
    #: dtype is part of the contract its caller consumes.
    return (safe * effective).double().sum(dim=-1) / denom.double()
