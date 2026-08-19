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


def _decoder_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError(f"Cannot locate decoder layers on {type(model).__name__}")
    return layers


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
    model.model.layers = torch.nn.ModuleList(kept)
    try:
        yield model
    finally:
        model.model.layers = layers


# --- distributional distortion -------------------------------------------------


@dataclass
class DistortionSums:
    """Unreduced accumulators, so a caller can pool positions its own way.

    Sums rather than means: the aggregation weights are a design decision made
    once at the top (`domain_balanced_score`), and a partially-averaged
    intermediate would quietly bake in token weighting.
    """

    positions: int = 0
    kl: float = 0.0
    reverse_kl: float = 0.0
    ref_ce: float = 0.0
    abl_ce: float = 0.0
    top1_agree: int = 0
    tagged: dict[str, list[float]] = field(default_factory=dict)

    def add_tagged(self, tag: str, kl_sum: float, count: int) -> None:
        cur = self.tagged.setdefault(tag, [0.0, 0.0])
        cur[0] += float(kl_sum)
        cur[1] += float(count)

    def merge(self, other: DistortionSums) -> None:
        self.positions += other.positions
        self.kl += other.kl
        self.reverse_kl += other.reverse_kl
        self.ref_ce += other.ref_ce
        self.abl_ce += other.abl_ce
        self.top1_agree += other.top1_agree
        for tag, (s, c) in other.tagged.items():
            self.add_tagged(tag, s, c)

    def as_dict(self) -> dict:
        n = self.positions
        if n == 0:
            raise ValueError("no positions accumulated")
        out = {
            "positions": n,
            "kl": self.kl / n,
            "reverse_kl": self.reverse_kl / n,
            "ref_ce": self.ref_ce / n,
            "abl_ce": self.abl_ce / n,
            "ce_delta": (self.abl_ce - self.ref_ce) / n,
            "top1_agreement": self.top1_agree / n,
        }
        out["tagged"] = {
            tag: {"positions": int(c), "kl": (s / c) if c else None}
            for tag, (s, c) in sorted(self.tagged.items())
        }
        return out


@torch.no_grad()
def distortion(
    ref_logits: torch.Tensor,
    abl_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tags: Mapping[str, torch.Tensor] | None = None,
    chunk: int = 512,
) -> DistortionSums:
    """Accumulate teacher -> ablated-teacher distortion over prediction positions.

    `ref_logits` / `abl_logits` are `[T_pred, V]` already aligned to the
    positions being scored, and `targets` is `[T_pred]`. `tags` maps a diagnostic
    name to a boolean `[T_pred]` mask; tagged KL is reported alongside but has no
    standing to change a selection (see `greedy_removal`).

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
    for a in range(0, ref_logits.shape[0], chunk):
        b = min(a + chunk, ref_logits.shape[0])
        p_log = F.log_softmax(ref_logits[a:b].float(), dim=-1)
        q_log = F.log_softmax(abl_logits[a:b].float(), dim=-1)
        p = p_log.exp()
        per_pos = (p * (p_log - q_log)).sum(-1)
        tg = targets[a:b]
        out.positions += int(b - a)
        out.kl += float(per_pos.sum())
        out.reverse_kl += float((q_log.exp() * (q_log - p_log)).sum(-1).sum())
        out.ref_ce += float(-p_log.gather(1, tg[:, None]).sum())
        out.abl_ce += float(-q_log.gather(1, tg[:, None]).sum())
        out.top1_agree += int((p_log.argmax(-1) == q_log.argmax(-1)).sum())
        for name, mask in tags.items():
            m = mask[a:b]
            k = int(m.sum())
            if k:
                out.add_tagged(name, float(per_pos[m].sum()), k)
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
