"""Reference-Top-K + tail forward KL as a differentiable TRAINING objective.

The divergence is the one :mod:`aadistill.initialization.scoring.support`
defines as ``reference_topk_tail_v1``: at each KD prediction position the
TEACHER's Top-K entries ``S`` plus one aggregate bucket holding every entry
outside ``S``::

    p_tail = P_teacher(not S),   q_tail = P_student(not S)
    KL = sum_{i in S} p_i (log p_i - log q_i) + p_tail (log p_tail - log q_tail)

It is NOT the full-vocabulary KL. The partition is coarser, so by the
data-processing inequality it is a lower bound on it; with ``top_k >= |V|``
the support is the whole vocabulary, there is no tail, and the two are equal
(the identity the tests pin). The probabilities are the ABSOLUTE ones against
each model's full-vocabulary normalizer, never renormalized inside ``S``, so
every position still pays the full ``O(V)`` log-normalizer: what K bounds is
the teacher state carried per position (``O(K)``), not the softmax cost.

**Rows are chosen before the output head.** The backbone runs the complete
sequence (every token is context); only the prediction positions the loss
reads are projected to the vocabulary. :func:`head_rows` takes the final hidden
states at those positions, so no ``[B, T, V]`` logits tensor exists for either
model.

**The tail is the complement's own log-sum-exp**, never ``log(1 - sum p_S)``:
masking ``S`` and reducing the rest has no cancellation, and the two masses
sum to one by construction (see ``complement_log_mass``). For the student this
is differentiable: the gradient of ``q_tail`` reaches exactly the entries
outside ``S``, through the softmax of the complement.

**Numerics.** The output head runs under the caller's autocast, exactly as in
the model's own forward; each block of rows is then widened to float32 and
divided by the temperature before any normalizer is taken. The student's block
runs under activation checkpointing, so the float32 ``[rows, V]`` transient
exists for one block at a time in the forward and again in the backward; its
peak is ``O(chunk * V)`` whatever the sequence length.

**The teacher's Top-K is ``torch.topk``** of its float32-widened logits, the
selection ``sketch_reference`` makes for ``reference_topk_tail_v1``.

Nothing here names a model, a vocabulary or a ``top_k``: the support is the
caller's (an experiment-policy parameter), the head is the model's own module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.checkpoint import checkpoint

from aadistill.initialization.scoring.support import (
    SUPPORT_REFERENCE_TOPK_TAIL_V1,
    DistributionSupport,
)

#: Identity strings for the objective, emitted into every run manifest by the
#: trainer (``Trainer.execution_record``). A description that stops matching the
#: code is a change to this file, visible in the trainer source digest.
OBJECTIVE_ID = "target_rows_reference_topk_tail_forward_kl/v1"
TAIL_ALGORITHM_ID = "complement_logsumexp/v1"
SUPPORT_SELECTION_ID = "torch.topk of the teacher's float32-widened logits / T"
ROW_SELECTION_ID = "final hidden states at the loss's prediction positions, before the output head"
REDUCTION_ID = ("sum over the step's KD positions of tau^2 * KL, divided by that step's KD position "
                "count (the trainer's normalizer); CE likewise over its own positions")
#: Training supports this module implements.
TRAINING_SUPPORTS = (SUPPORT_REFERENCE_TOPK_TAIL_V1,)

__all__ = [
    "OBJECTIVE_ID", "TAIL_ALGORITHM_ID", "SUPPORT_SELECTION_ID", "ROW_SELECTION_ID",
    "REDUCTION_ID", "TRAINING_SUPPORTS", "TeacherSketch", "training_support",
    "objective_identity", "head_rows", "output_head", "teacher_sketch",
    "topk_tail_losses",
]


def training_support(spec: Any) -> DistributionSupport:
    """The support a ``loss.kd_support`` config entry names, validated."""
    if not isinstance(spec, dict) or "support_id" not in spec:
        raise ValueError("loss.kd_support must be an object with a support_id")
    extra = sorted(set(spec) - {"support_id", "top_k"})
    if extra:
        raise ValueError(f"loss.kd_support has unknown keys {extra}")
    support = DistributionSupport(spec["support_id"], spec.get("top_k"))
    if support.support_id not in TRAINING_SUPPORTS:
        raise ValueError(f"loss.kd_support {support} is not a training support; "
                         f"known: {TRAINING_SUPPORTS}")
    return support


def objective_identity(support: DistributionSupport, kd_scope: str, temperature: float) -> dict:
    """What the KD term IS, as the run manifest records it."""
    return {"objective": OBJECTIVE_ID, "support": support.as_dict(),
            "support_selection": SUPPORT_SELECTION_ID, "tail": TAIL_ALGORITHM_ID,
            "rows": ROW_SELECTION_ID, "position_policy": f"kd_scope {kd_scope!r}",
            "temperature": float(temperature), "reduction": REDUCTION_ID,
            "is_full_vocabulary_kl": False}


def output_head(model) -> Callable[[torch.Tensor], torch.Tensor]:
    """The model's own output head as a row-wise function of final hidden states.

    Refuses a model whose forward post-processes the head's output (a logit
    soft-cap or scale): applying the head module alone would then compute
    different logits from the model's own forward.
    """
    cfg = getattr(model, "config", None)
    for field in ("final_logit_softcapping", "logit_scale", "logits_scaling"):
        value = getattr(cfg, field, None)
        if value not in (None, 1, 1.0):
            raise ValueError(f"{type(model).__name__} sets {field}={value!r}; its logits are not "
                             "the output head alone, which head-row selection would need")
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError(f"{type(model).__name__} has no output head")
    return head


def head_rows(model, input_ids: torch.Tensor, rows: torch.Tensor, **forward_kwargs) -> torch.Tensor:
    """``[N, H]`` final hidden states at the prediction positions ``rows`` marks.

    ``rows`` is a ``[B, T-1]`` boolean mask over prediction positions (position
    ``t`` predicts token ``t + 1``), the layout of the trainer's CE and KD
    masks. The decoder runs the whole ``[B, T]`` sequence.
    """
    hidden = model.get_decoder()(input_ids=input_ids, use_cache=False,
                                 **forward_kwargs).last_hidden_state
    return hidden[:, :-1][rows]


@dataclass(frozen=True)
class TeacherSketch:
    """The teacher's state at ``N`` rows: ``O(N*K)``, held without a graph."""

    support_indices: torch.Tensor      # [N, K] long
    support_log_probs: torch.Tensor    # [N, K] float32, against the full normalizer
    tail_log_prob: torch.Tensor        # [N] float32; -inf when there is no tail
    has_tail: bool


@torch.no_grad()
def teacher_sketch(head: Callable, hidden: torch.Tensor, *, top_k: int, temperature: float = 1.0,
                   chunk: int = 512) -> TeacherSketch:
    """The teacher's Top-``top_k`` entries, their log probabilities and its tail, per row."""
    n = int(hidden.shape[0])
    if n == 0:
        raise ValueError("a teacher sketch over zero rows")
    step = max(1, int(chunk))
    idx = lp = tail = None
    for start in range(0, n, step):
        z = head(hidden[start:start + step]).float() / float(temperature)
        vocab = int(z.shape[-1])
        k = max(1, min(int(top_k), vocab))
        if idx is None:
            idx = torch.empty(n, k, dtype=torch.long, device=z.device)
            lp = torch.empty(n, k, dtype=torch.float32, device=z.device)
            tail = torch.full((n,), float("-inf"), dtype=torch.float32, device=z.device)
        log_z = torch.logsumexp(z, dim=-1)
        top = torch.topk(z, k, dim=-1, largest=True, sorted=True)
        idx[start:start + step] = top.indices
        lp[start:start + step] = top.values - log_z[:, None]
        if k < vocab:
            #: `z` is this block's own float32 tensor, destroyed here.
            z.scatter_(-1, top.indices, float("-inf"))
            tail[start:start + step] = torch.logsumexp(z, dim=-1) - log_z
        del z, log_z, top
    return TeacherSketch(idx, lp, tail, has_tail=bool(k < vocab))


def _xlogx_from_log(log_p: torch.Tensor) -> torch.Tensor:
    """``p log p`` from ``log p``, with ``0 log 0 = 0``."""
    return torch.where(torch.isneginf(log_p), torch.zeros_like(log_p), log_p.exp() * log_p)


def _row_losses(head, hidden, idx, lp, tail, targets, ce_w, kd_w, temperature, has_tail):
    """Per-block CE and KL sums; differentiable in ``hidden`` and the head."""
    z = head(hidden).float()
    log_z = torch.logsumexp(z, dim=-1)
    ce = (log_z - z.gather(-1, targets[:, None]).squeeze(-1)) * ce_w
    if temperature != 1.0:
        z = z / temperature
        log_z = torch.logsumexp(z, dim=-1)
    q_s = z.gather(-1, idx) - log_z[:, None]
    #: KL = sum p log p - sum p log q over the K+1 categories. The first part is
    #: the teacher's alone (no gradient); writing it this way keeps a zero-mass
    #: bucket at an exact zero in value and in gradient.
    cross = (lp.exp() * q_s).sum(-1)
    const = _xlogx_from_log(lp).sum(-1)
    if has_tail:
        q_tail = torch.logsumexp(z.scatter(-1, idx, float("-inf")), dim=-1) - log_z
        p_tail = tail.exp()
        cross = cross + p_tail * q_tail
        const = const + _xlogx_from_log(tail)
    kl = (const - cross) * kd_w
    return ce.sum(), kl.sum()


def topk_tail_losses(head: Callable, student_hidden: torch.Tensor, sketch: TeacherSketch,
                     targets: torch.Tensor, ce_rows: torch.Tensor, kd_rows: torch.Tensor, *,
                     temperature: float = 1.0, chunk: int = 512,
                     recompute: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ce_sum, kd_sum)`` over ``N`` rows: CE at ``ce_rows``, ``tau^2 * KL`` at ``kd_rows``.

    ``student_hidden`` ``[N, H]`` carries the graph; ``targets`` ``[N]`` are the
    next tokens; ``ce_rows`` / ``kd_rows`` ``[N]`` bool say which loss each row
    enters (the trainer forms the rows as their union). ``recompute`` checkpoints
    each block so its ``[chunk, V]`` logits are not kept for the backward.
    """
    n = int(student_hidden.shape[0])
    if not (int(targets.shape[0]) == int(ce_rows.shape[0]) == int(kd_rows.shape[0])
            == int(sketch.support_indices.shape[0]) == n):
        raise ValueError("rows, targets, masks and the teacher sketch disagree on the row count")
    zero = student_hidden.sum() * 0.0
    ce_total, kd_total = zero, zero
    t = float(temperature)
    step = max(1, int(chunk))
    for start in range(0, n, step):
        s = slice(start, start + step)
        args = (student_hidden[s], sketch.support_indices[s], sketch.support_log_probs[s],
                sketch.tail_log_prob[s], targets[s], ce_rows[s].float(), kd_rows[s].float())
        fn = lambda h, i, l, tl, tg, cw, kw: _row_losses(head, h, i, l, tl, tg, cw, kw, t,  # noqa: E731
                                                         sketch.has_tail)
        if recompute and torch.is_grad_enabled():
            ce, kd = checkpoint(fn, *args, use_reentrant=False)
        else:
            ce, kd = fn(*args)
        ce_total = ce_total + ce
        kd_total = kd_total + kd
    return ce_total, kd_total * (t * t)
