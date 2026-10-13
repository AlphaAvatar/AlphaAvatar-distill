"""Per-position scoring objectives that combine KL with the student's own confidence.

The historical objective of every KL-scored decision is a per-item mean of the
forward KL over the item's scored prediction positions. This module adds a
small family of objectives that keep that KL and add one more per-position
quantity: how confident the COMPARED (student) distribution is in itself.

**Student confidence, exactly.** For the compared distribution ``q_t`` at a
prediction position, take the ``K`` entries of ``q_t``'s OWN logits that are
largest, renormalize inside them, and measure their entropy::

    S_t      = TopK(z_t, K)                       the student's own Top-K
    q~_t,i   = exp(z_t,i) / sum_{j in S_t} exp(z_t,j)          i in S_t
    H_K(t)   = - sum_{i in S_t} q~_t,i log q~_t,i              in [0, log K]
    c(t)     = 1 - H_K(t) / log K                              in [0, 1]

This is a conditional entropy on the student's own support. It is NOT the
reference's entropy, NOT the full-vocabulary entropy, NOT the student's entropy
on the reference's support, and NOT a ``K+1``-bucket entropy: the mass outside
``S_t`` is reported beside it (``student_topk_mass``) and never folded in.
``K`` is the caller's and this module holds no value for it.

**Forms.** Each is a per-position integrand ``g_t`` of the KL and ``c``:

``student_uncertainty_weighted_kl``
    ``g_t = (1 - c_t) * KL_t``. A position the student is confident about is
    penalized less for diverging from the reference; an uncertain one more.
``kl_minus_student_confidence_reward``
    ``g_t = KL_t - lambda * c_t``. Confidence is a separate reward, never a
    weight on the KL; ``lambda > 0`` is part of the identity.

Both are MINIMIZED, and both are averaged per item over a FIXED denominator,
the number of scored positions ``T_x``::

    L(x) = (1 / T_x) * sum_{t in T_x} g_t

and not over a dynamic weight sum. ``sum w KL / sum w`` would cancel a uniform
rise in confidence, which is exactly the effect the uncertainty-weighted form
exists to let through. Items are then aggregated by the caller's existing
sub-type and domain rule; nothing here pools positions across items.

**What a confident student cannot buy.** ``KL_t = +inf`` stays ``+inf`` under
every form, including at ``c_t = 1``: a candidate assigning zero probability to
reference mass is impossible, and confidence does not rescue it (``0 * inf`` is
undefined, and resolving it to zero would turn an impossible candidate into the
best one). A finite KL at ``c_t = 1`` contributes zero to the weighted form.

**Identity.** An objective's ``as_dict`` is part of what a measurement means,
so it is bound into the operator config and the measurement protocol
identity -- and absent, by construction, when no objective is declared, so the
historical KL-only identities are unchanged.

Nothing here names an experiment, a model family or a size.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from aadistill.infrastructure.manifest import sha256_json

SCHEMA = "aadistill.scoring_objective/v1"

FORM_UNCERTAINTY_WEIGHTED_KL = "student_uncertainty_weighted_kl"
FORM_KL_MINUS_CONFIDENCE_REWARD = "kl_minus_student_confidence_reward"
FORMS = (FORM_UNCERTAINTY_WEIGHTED_KL, FORM_KL_MINUS_CONFIDENCE_REWARD)

#: Every definitional choice, named so it is part of the identity rather than
#: of the reader's assumptions. Changing any of them is a new objective.
ENTROPY_DEFINITION = "student_own_topk_conditional_entropy/v1"
CONFIDENCE_MAPPING = "one_minus_entropy_over_log_k/v1"
DENOMINATOR = "scored_position_count/v1"
ITEM_AGGREGATION = "per_item_fixed_denominator_then_caller_rule/v1"
DIRECTION = "minimize"

#: Rows per reduction block. A transient bound, not semantics: Top-K over a
#: ``[rows, V]`` block allocates proportionally to it.
DEFAULT_ROW_CHUNK = 512


class ObjectiveError(ValueError):
    """An objective that cannot be evaluated as defined."""


@dataclass(frozen=True)
class ScoringObjective:
    """One per-position KL + student-confidence objective. Hashable identity."""

    objective_id: str
    version: int
    form: str
    entropy_top_k: int
    #: ``lambda`` of the reward form; must be absent for the weighted form.
    reward_scale: float | None = None

    def __post_init__(self) -> None:
        if not self.objective_id:
            raise ObjectiveError("an objective needs an id")
        if self.form not in FORMS:
            raise ObjectiveError(f"unknown objective form {self.form!r}; "
                                 f"known: {FORMS}")
        if int(self.entropy_top_k) < 2:
            raise ObjectiveError(
                "entropy_top_k must be at least 2: log K normalizes the "
                "entropy, and log 1 = 0")
        if self.form == FORM_KL_MINUS_CONFIDENCE_REWARD:
            lam = self.reward_scale
            if lam is None or not math.isfinite(float(lam)) or float(lam) <= 0:
                raise ObjectiveError(
                    f"{self.objective_id}: the reward form needs a finite "
                    f"reward_scale > 0, got {lam!r}")
        elif self.reward_scale is not None:
            raise ObjectiveError(
                f"{self.objective_id}: {self.form} has no reward term, so a "
                "reward_scale would be an identity field that changes nothing")

    @property
    def qualified_id(self) -> str:
        return f"{self.objective_id}@v{self.version}"

    def as_dict(self) -> dict[str, Any]:
        out = {
            "schema": SCHEMA,
            "objective_id": self.objective_id,
            "version": int(self.version),
            "form": self.form,
            "entropy_definition": ENTROPY_DEFINITION,
            "entropy_top_k": int(self.entropy_top_k),
            "confidence_mapping": CONFIDENCE_MAPPING,
            "denominator": DENOMINATOR,
            "item_aggregation": ITEM_AGGREGATION,
            "direction": DIRECTION,
        }
        if self.reward_scale is not None:
            out["reward_scale"] = float(self.reward_scale)
        return out

    @property
    def objective_hash(self) -> str:
        return sha256_json(self.as_dict())

    def component_name(self) -> str:
        """What the second component is called in records: weight or reward."""
        return ("uncertainty_weight" if self.form == FORM_UNCERTAINTY_WEIGHTED_KL
                else "confidence_reward")

    def integrand(self, kl: torch.Tensor, confidence: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(g, component)`` per position. ``+inf`` KL stays ``+inf``."""
        kl = kl.float()
        conf = confidence.float()
        if self.form == FORM_UNCERTAINTY_WEIGHTED_KL:
            weight = 1.0 - conf
            #: `weight * kl` is NaN at weight 0 and kl +inf; resolved to +inf
            #: on purpose (see the module docstring).
            g = torch.where(torch.isposinf(kl), kl, weight * kl)
            return g, weight
        reward = float(self.reward_scale) * conf
        return kl - reward, reward


def objective_config(objective: ScoringObjective | None) -> dict[str, Any]:
    """An objective as hashed config, or ``{}`` when none is declared."""
    if objective is None:
        return {}
    return {"scoring_objective": objective.as_dict()}


@torch.no_grad()
def student_topk_terms(cand_logits: torch.Tensor, log_z: torch.Tensor, *,
                       top_k: int, chunk: int = DEFAULT_ROW_CHUNK
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(entropy, topk_mass)`` per row of ``cand_logits`` ``[..., V]``.

    ``entropy`` is ``H_K``, the conditional entropy on the row's own Top-K;
    ``topk_mass`` is the full-softmax mass of that Top-K, ``exp(lse_K - log_z)``,
    reported beside it and never folded in. ``log_z`` ``[...]`` is the row's
    full-vocabulary log-normalizer, which the caller's KL reduction has already
    computed -- so the only new vocabulary-wide work here is the selection.

    The selection reads the logits in their own dtype and only the ``K``
    selected values are promoted to float32: promotion is exact, so the set and
    the values are those a float32 copy would give, without a second
    ``[rows, V]`` float32 buffer.
    """
    vocab = int(cand_logits.shape[-1])
    k = int(top_k)
    if k > vocab:
        raise ObjectiveError(
            f"entropy_top_k={k} exceeds the vocabulary ({vocab}); the student "
            "has no Top-K of that size and log K would not normalize it")
    lead = cand_logits.shape[:-1]
    flat = cand_logits.reshape(-1, vocab)
    lz = log_z.reshape(-1).float()
    if int(lz.numel()) != int(flat.shape[0]):
        raise ObjectiveError(
            f"{int(lz.numel())} log-normalizers for {int(flat.shape[0])} rows")
    entropy = torch.empty(flat.shape[0], dtype=torch.float32,
                          device=cand_logits.device)
    mass = torch.empty_like(entropy)
    step = max(1, int(chunk))
    for start in range(0, int(flat.shape[0]), step):
        stop = min(start + step, int(flat.shape[0]))
        values = torch.topk(flat[start:stop], k, dim=-1, sorted=False).values
        values = values.float()
        lse = torch.logsumexp(values, dim=-1, keepdim=True)
        log_q = values - lse
        entropy[start:stop] = -(log_q.exp() * log_q).sum(dim=-1)
        mass[start:stop] = (lse.squeeze(-1) - lz[start:stop].to(values.device)
                            ).exp()
    return entropy.reshape(lead), mass.reshape(lead)


def confidence_from_entropy(entropy: torch.Tensor, top_k: int) -> torch.Tensor:
    """``c = 1 - H_K / log K``, clamped to ``[0, 1]`` against rounding only.

    ``H_K`` lies in ``[0, log K]`` mathematically; float32 can step a uniform
    row's entropy a few ulps past ``log K``. The clamp removes that and nothing
    else -- a value far outside the interval would be a defect upstream, and the
    tests bound the unclamped value before relying on this.
    """
    return (1.0 - entropy.float() / math.log(int(top_k))).clamp(0.0, 1.0)


def require_binary_scoring_weights(weights: torch.Tensor | None) -> None:
    """The fixed denominator counts scored positions, so weights must be 0/1.

    A continuous position weight would leave ``T_x`` undefined -- is it the
    count or the weight sum? -- and the objective's definition answers neither.
    Refused rather than rounded.
    """
    if weights is None:
        return
    w = weights.float()
    if bool(((w != 0) & (w != 1)).any()):
        raise ObjectiveError(
            "a student-confidence objective averages over a fixed count of "
            "scored positions, which needs a binary position policy; these "
            "weights are continuous")


def fixed_denominator_item_means(values: torch.Tensor,
                                 prediction_mask: torch.Tensor,
                                 weights: torch.Tensor | None = None,
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(means [B], counts [B])``: ``sum over scored positions / T_x``.

    ``values`` ``[B, T]``. A position is scored when it is valid in
    ``prediction_mask`` and, if ``weights`` are given, has weight 1. Positions
    outside are never read, so the caller may leave anything there. float64,
    like the KL-only per-item reducer this sits beside.
    """
    require_binary_scoring_weights(weights)
    scored = prediction_mask.to(values.device).bool()
    if weights is not None:
        scored = scored & (weights.to(values.device) > 0)
    counts = scored.sum(dim=-1)
    if bool((counts <= 0).any()):
        raise ObjectiveError(
            "an item has no scored position, so T_x is zero and its mean has no "
            "denominator; an item that contributes nothing must not be scored")
    safe = torch.where(scored, values.double(), torch.zeros((), dtype=torch.float64,
                                                            device=values.device))
    return safe.sum(dim=-1) / counts.double(), counts


#: The per-position quantities every record carries, in a fixed order.
COMPONENTS = ("objective", "raw_kl", "student_entropy", "student_confidence",
              "component", "student_topk_mass", "reference_tail_mass")

#: Confidence histogram edges: ten equal bins on [0, 1].
CONFIDENCE_BINS = 10


@torch.no_grad()
def position_terms(objective: ScoringObjective, kl: torch.Tensor,
                   cand_logits: torch.Tensor, log_z: torch.Tensor,
                   reference_tail_log: torch.Tensor | None,
                   ) -> dict[str, torch.Tensor]:
    """Every per-position component, same leading shape as ``kl``."""
    entropy, mass = student_topk_terms(cand_logits, log_z,
                                       top_k=objective.entropy_top_k)
    entropy = entropy.to(kl.device)
    conf = confidence_from_entropy(entropy, objective.entropy_top_k)
    g, component = objective.integrand(kl, conf)
    tail = (torch.zeros_like(kl, dtype=torch.float32)
            if reference_tail_log is None
            else reference_tail_log.to(kl.device).float().exp())
    return {"objective": g, "raw_kl": kl.float(), "student_entropy": entropy,
            "student_confidence": conf, "component": component,
            "student_topk_mass": mass.to(kl.device), "reference_tail_mass": tail}


def item_component_means(terms: Mapping[str, torch.Tensor],
                         prediction_mask: torch.Tensor,
                         weights: torch.Tensor | None = None
                         ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Each component's per-item fixed-denominator mean, and ``T_x``."""
    out: dict[str, torch.Tensor] = {}
    counts = None
    for name in COMPONENTS:
        out[name], counts = fixed_denominator_item_means(
            terms[name], prediction_mask, weights)
    return out, counts


def confidence_histogram(confidence: torch.Tensor,
                         scored: torch.Tensor) -> list[int]:
    """Counts of scored positions per confidence decile, low to high."""
    values = confidence[scored.to(confidence.device).bool()].float()
    if values.numel() == 0:
        return [0] * CONFIDENCE_BINS
    edges = torch.clamp((values * CONFIDENCE_BINS).long(), 0,
                        CONFIDENCE_BINS - 1)
    return [int(x) for x in torch.bincount(edges,
                                           minlength=CONFIDENCE_BINS).tolist()]


__all__ = ["COMPONENTS", "CONFIDENCE_BINS", "CONFIDENCE_MAPPING",
           "DENOMINATOR", "DIRECTION", "ENTROPY_DEFINITION",
           "FORM_KL_MINUS_CONFIDENCE_REWARD", "FORM_UNCERTAINTY_WEIGHTED_KL",
           "FORMS", "ITEM_AGGREGATION", "ObjectiveError", "ScoringObjective",
           "confidence_from_entropy", "confidence_histogram",
           "fixed_denominator_item_means", "item_component_means",
           "objective_config", "position_terms",
           "require_binary_scoring_weights", "student_topk_terms"]
