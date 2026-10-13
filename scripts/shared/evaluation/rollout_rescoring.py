"""Per-position diagnostics of a student's OWN rollout, against the teacher.

The structural search scores students on TEACHER-NATIVE prefixes. A student's
behaviour is what it does on ITS OWN prefixes. These are different quantities
and this module measures the second one, after generation, without touching
how anything was generated:

``student_confidence``
    the student's own Top-K confidence at each position of its own rollout --
    the same definition the D-series objectives use, on a different prefix;
``student_token_logprob``
    the log-probability the student gave the token it emitted;
``teacher_surprisal``
    ``-log p_teacher(token)``: how unexpected the student's trajectory is to
    the teacher, token by token;
``kl_teacher_student``
    the reference Top-K + tail KL from teacher to student at the student's
    own prefixes -- distribution fit where the student actually goes.

It needs the STUDENT'S checkpoint (for its distributions) and the teacher. A
unit whose checkpoint is gone cannot be rescored, and nothing here estimates
what it would have shown.

The trajectory is the token sequence the caller passes. When only decoded
text survives, re-tokenizing it is an APPROXIMATION of the generated ids; the
caller records whether the count matched the engine's.

Thresholds for "high confidence" and "high KL" are the caller's, registered
before any result: this module holds none.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from aadistill.initialization.scoring.objective import (
    confidence_from_entropy, student_topk_terms,
)
from aadistill.initialization.scoring.support import (
    sketch_forward_kl_terms, sketch_reference,
)
from aadistill.initialization.specs.arch import adapter_for_config

#: Rows projected to the vocabulary at once. Memory bound, not semantics.
DEFAULT_CHUNK = 1024


@torch.no_grad()
def _completion_logits(model, ids: torch.Tensor, start: int, stop: int):
    """Logits predicting tokens ``start..stop-1`` of ``ids``, in row chunks."""
    adapter = adapter_for_config(model.config)
    hidden = adapter.final_hidden_states(model, ids, None)
    #: The causal shift: token j is predicted from position j - 1.
    for a in range(start, stop, DEFAULT_CHUNK):
        b = min(a + DEFAULT_CHUNK, stop)
        yield a, b, adapter.output_logits(model, hidden[0, a - 1:b - 1])


@torch.no_grad()
def rescore_trajectory(student, teacher, prompt_ids: torch.Tensor,
                       completion_ids: torch.Tensor, *, support_top_k: int,
                       entropy_top_k: int) -> dict[str, torch.Tensor]:
    """Per completion position: the four quantities above, float32, on host."""
    if prompt_ids.dim() != 2 or completion_ids.dim() != 2:
        raise ValueError("expected [1, P] prompt ids and [1, C] completion ids")
    p = int(prompt_ids.shape[1])
    c = int(completion_ids.shape[1])
    if p < 1 or c < 1:
        raise ValueError("an empty prompt or completion has no prediction")
    device = next(student.parameters()).device
    ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)
    out: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("student_confidence", "student_entropy",
                        "student_token_logprob", "teacher_surprisal",
                        "kl_teacher_student", "student_topk_mass")}
    student_chunks = _completion_logits(student, ids, p, p + c)
    teacher_chunks = _completion_logits(teacher, ids.to(
        next(teacher.parameters()).device), p, p + c)
    for (a, b, s_logits), (_, _, t_logits) in zip(student_chunks,
                                                   teacher_chunks):
        targets = ids[0, a:b].to(t_logits.device)
        vocab = int(t_logits.shape[-1])
        sketch = sketch_reference(t_logits.float(), targets,
                                  top_k=int(support_top_k))
        s_logits = s_logits.to(t_logits.device)
        kl, log_z = sketch_forward_kl_terms(
            sketch.support_indices, sketch.support_log_probs,
            sketch.tail_log_prob, s_logits,
            has_tail=int(support_top_k) < vocab)
        entropy, mass = student_topk_terms(s_logits, log_z,
                                           top_k=int(entropy_top_k))
        token_lp = (s_logits.float().gather(1, targets.view(-1, 1)).squeeze(1)
                    - log_z)
        out["student_confidence"].append(
            confidence_from_entropy(entropy, int(entropy_top_k)).cpu())
        out["student_entropy"].append(entropy.cpu())
        out["student_token_logprob"].append(token_lp.float().cpu())
        out["teacher_surprisal"].append((-sketch.target_log_prob).float().cpu())
        out["kl_teacher_student"].append(kl.float().cpu())
        out["student_topk_mass"].append(mass.float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def summarize_trajectory(terms: Mapping[str, torch.Tensor], *,
                         reasoning_positions: int | None,
                         high_confidence: float,
                         high_kl: float) -> dict[str, Any]:
    """Per-sample means, split into reasoning and answer, and the quadrant.

    ``reasoning_positions`` is how many leading completion positions are the
    thinking span (``None`` when it never closed). ``high_confidence`` and
    ``high_kl`` are the caller's registered thresholds; the quadrant is the
    fraction of positions that are confident AND far from the teacher -- the
    positions where an alternative trajectory and overconfidence look alike
    and have to be told apart by correctness, not by this number.
    """
    c = terms["student_confidence"]
    kl = terms["kl_teacher_student"]
    n = int(c.numel())

    def means(sl: slice) -> dict[str, float] | None:
        if sl.stop is not None and sl.stop <= (sl.start or 0):
            return None
        return {k: float(v[sl].double().mean()) for k, v in terms.items()
                if int(v[sl].numel())}

    split = None if reasoning_positions is None else int(reasoning_positions)
    quadrant = ((c >= float(high_confidence)) & (kl >= float(high_kl)))
    return {
        "positions": n,
        "all": means(slice(0, n)),
        "reasoning": None if split is None else means(slice(0, split)),
        "answer": None if split is None else means(slice(split, n)),
        "high_confidence_high_kl_fraction": float(quadrant.double().mean()),
        "thresholds": {"high_confidence": float(high_confidence),
                       "high_kl": float(high_kl)},
    }


__all__ = ["rescore_trajectory", "summarize_trajectory"]
