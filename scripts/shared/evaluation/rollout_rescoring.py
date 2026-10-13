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

The trajectory is the token sequence the caller passes, and every result
says WHICH: ``ids_source="engine"`` (the generator's own ids, from the
read-only ``uncapped_eval_token_ids`` sidecar, checked here against the
record's count and decoded text, and refused if either differs)
or ``ids_source="retokenized_text"`` (an APPROXIMATION of the generated ids,
labelled as such, with whether its count matched the engine's).

Failure isolation: :func:`rescore_unit` writes NOTHING into a unit's
evaluation outputs and never raises for one sample -- a failed sample is
recorded as failed, the rest continue, and a completed per-sample score is
untouched whatever happens here.

Thresholds for "high confidence" and "high KL" are the caller's, registered
before any result: this module holds none.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

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


class RescoringError(ValueError):
    """A trajectory that is not the one the record describes."""


def _sha(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.reshape(-1).to(torch.int64).cpu().numpy()
                          .tobytes()).hexdigest()


def alignment_evidence(record: Mapping[str, Any], prompt_ids: torch.Tensor,
                       completion_ids: torch.Tensor, tokenizer, *,
                       ids_source: str, tokenizer_sha256: str,
                       generation_fingerprint: str) -> dict[str, Any]:
    """What was rescored, and whether it is the sequence that was generated."""
    if ids_source not in ("engine", "retokenized_text"):
        raise RescoringError(f"unknown ids_source {ids_source!r}")
    decoded = tokenizer.decode(completion_ids[0].tolist(),
                               skip_special_tokens=False)
    count_ok = int(completion_ids.shape[1]) == int(record["generated_tokens"])
    text_ok = decoded == record.get("raw")
    if ids_source == "engine" and not (count_ok and text_ok):
        raise RescoringError(
            f"{record.get('id')}: the engine ids do not reproduce the record "
            f"(count {int(completion_ids.shape[1])} vs "
            f"{record['generated_tokens']}, text match {text_ok}); they are "
            "not the generated trajectory")
    return {"id": record.get("id"), "ids_source": ids_source,
            "exact": ids_source == "engine",
            "prompt_tokens": int(prompt_ids.shape[1]),
            "completion_tokens": int(completion_ids.shape[1]),
            "engine_generated_tokens": int(record["generated_tokens"]),
            "count_matches_engine": count_ok,
            "decodes_to_recorded_text": text_ok,
            "prompt_ids_sha256": _sha(prompt_ids),
            "completion_ids_sha256": _sha(completion_ids),
            "tokenizer_sha256": tokenizer_sha256,
            "generation_fingerprint": generation_fingerprint}


def trajectory_sample(record: Mapping[str, Any],
                      sidecar: tuple[Mapping[str, Any], Mapping[str, Any]] | None,
                      retokenize) -> dict[str, Any]:
    """One rescoring sample: the engine's ids when the sidecar is exact.

    ``sidecar`` is ``(header, id -> row)`` from
    ``uncapped_eval_token_ids.load_sidecar`` or ``None``. Engine ids are used
    only when the stratum's header says ``exact`` and the record has a row;
    otherwise ``retokenize(record)`` supplies ``(prompt_ids, completion_ids)``
    and the sample is labelled ``retokenized_text``. Either way
    :func:`alignment_evidence` checks it against the record again.
    """
    header, rows = sidecar if sidecar else ({}, {})
    row = rows.get(str(record.get("id"))) if header.get("exact") else None
    #: A row that failed its own alignment is never treated as exact.
    if row is not None and (row.get("alignment") or {}).get("aligned") is False:
        row = None
    if row is not None:
        return {"record": record, "ids_source": "engine",
                "prompt_ids": torch.tensor([row["prompt_token_ids"]]),
                "completion_ids": torch.tensor([row["output_token_ids"]])}
    prompt_ids, completion_ids = retokenize(record)
    return {"record": record, "ids_source": "retokenized_text",
            "prompt_ids": prompt_ids, "completion_ids": completion_ids}


def summarize_trajectory(terms: Mapping[str, torch.Tensor], *,
                         reasoning_positions: int | None,
                         high_confidence: float,
                         high_kl: float,
                         low_topk_mass: float | None = None) -> dict[str, Any]:
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
    out = {
        "positions": n,
        "all": means(slice(0, n)),
        "reasoning": None if split is None else means(slice(0, split)),
        "answer": None if split is None else means(slice(split, n)),
        "high_confidence_high_kl_fraction": float(quadrant.double().mean()),
        "thresholds": {"high_confidence": float(high_confidence),
                       "high_kl": float(high_kl),
                       "low_topk_mass": (None if low_topk_mass is None
                                         else float(low_topk_mass))},
    }
    if low_topk_mass is not None and "student_topk_mass" in terms:
        #: High CONDITIONAL confidence on a Top-K that holds little of the
        #: student's real probability: the confidence is about a small slice.
        thin = ((c >= float(high_confidence))
                & (terms["student_topk_mass"] < float(low_topk_mass)))
        out["high_confidence_low_topk_mass_fraction"] = float(
            thin.double().mean())
    return out


def rescore_unit(student, teacher, samples: Iterable[Mapping[str, Any]], *,
                 tokenizer, think_close_id: int | None, tokenizer_sha256: str,
                 generation_fingerprint: str, support_top_k: int,
                 entropy_top_k: int, thresholds: Mapping[str, float]
                 ) -> dict[str, Any]:
    """Rescore every sample of a unit; one failure never stops the others.

    ``samples`` yields ``{"record", "prompt_ids", "completion_ids",
    "ids_source"}``. The return value is the unit's diagnostic evidence, to be
    persisted by the caller BEFORE the checkpoint is released; it is a
    separate artifact from the unit's evaluation outputs.
    """
    rows, failures = [], []
    for sample in samples:
        record = sample["record"]
        try:
            evidence = alignment_evidence(
                record, sample["prompt_ids"], sample["completion_ids"],
                tokenizer, ids_source=sample["ids_source"],
                tokenizer_sha256=tokenizer_sha256,
                generation_fingerprint=generation_fingerprint)
            terms = rescore_trajectory(
                student, teacher, sample["prompt_ids"],
                sample["completion_ids"], support_top_k=support_top_k,
                entropy_top_k=entropy_top_k)
            completion = sample["completion_ids"][0].tolist()
            split = (completion.index(think_close_id)
                     if think_close_id is not None
                     and think_close_id in completion else None)
            rows.append({"alignment": evidence,
                         "summary": summarize_trajectory(
                             terms, reasoning_positions=split,
                             high_confidence=thresholds["high_confidence"],
                             high_kl=thresholds["high_kl"],
                             low_topk_mass=thresholds.get("low_topk_mass"))})
        except Exception as exc:                    # noqa: BLE001
            failures.append({"id": record.get("id"),
                             "error": f"{type(exc).__name__}: {exc}"})
    return {"rows": rows, "failures": failures,
            "n_exact": sum(r["alignment"]["exact"] for r in rows),
            "thresholds": dict(thresholds),
            "complete": not failures}


__all__ = ["RescoringError", "alignment_evidence", "rescore_trajectory",
           "rescore_unit", "summarize_trajectory", "trajectory_sample"]
