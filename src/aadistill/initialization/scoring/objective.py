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
``reference_confidence_weighted_kl``
    ``g_t = c^ref_t * KL_t``, where ``c^ref_t`` is the REFERENCE's own
    confidence, ``1 - H_ref(t) / log |V|`` from the reference's full-vocabulary
    Shannon entropy. The candidate's logits never enter the weight. The KL's
    reference is declared in the identity (``kl_reference``); this form
    requires the ROOT reference -- the frozen original model at the root of
    the lineage -- for every consumer, including DEPTH, whose historical KL is
    taken against the parent. A position the reference is unsure of is
    penalized less; one it is sure of more. Optionally the same weights also
    weight the activation statistics the structural operators read
    (``statistics_weighting``).

``reference_topk_confidence_weighted_union_kl``
    ``g_t = c^ref_t * KL^U_t``. Same shape as the form above, with BOTH
    ingredients redefined, and declared as a separate form rather than a
    parameter of it:

    * ``c^ref_t = 1 - H^{K+1}_ref(t) / log(K+1)``, where ``H^{K+1}`` is the
      Shannon entropy of the reference's own Top-K entries and ONE tail bucket,
      at their ABSOLUTE probabilities (never renormalized inside the Top-K;
      ``0 log 0 = 0``). The number of categories is derived from the support,
      ``K + 1`` when ``K < |V|``; still candidate-independent, so it is
      computed once and stored with the reference;
    * ``KL^U_t`` is the forward KL on the reference ∪ candidate Top-K support
      plus one tail (``reference_student_union_topk_tail_v1``), whose top_k
      equals the confidence's ``K``. See
      :mod:`aadistill.initialization.scoring.support`.

    Its KL's reference is the ROOT for every consumer, and it may weight the
    activation statistics, exactly as above. ``exp(-H)`` and every other
    transform of the entropy are NOT this form.

The reference's entropy is a FULL-vocabulary quantity under the first
reference form and a ``K+1``-category one under the second, while the KL stays
the declared support's: different reductions over the same positions, never to
be read as one another.

Both are MINIMIZED, and both are averaged per item over a FIXED denominator,
the number of scored positions ``T_x``::

    L(x) = (1 / T_x) * sum_{t in T_x} g_t

and not over a dynamic weight sum. ``sum w KL / sum w`` would cancel a uniform
rise in confidence, which is exactly the effect the uncertainty-weighted form
exists to let through. Across items each consumer keeps the weighting its
KL-only path already uses (see :data:`AGGREGATION`), so that replacing the KL
by the integrand is the only change a consumer makes.

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
from aadistill.initialization.scoring.support import (
    SUPPORT_REFERENCE_STUDENT_UNION_TOPK_TAIL_V1, topk_tail_categories,
    topk_tail_entropy,
)

SCHEMA = "aadistill.scoring_objective/v1"

FORM_UNCERTAINTY_WEIGHTED_KL = "student_uncertainty_weighted_kl"
FORM_KL_MINUS_CONFIDENCE_REWARD = "kl_minus_student_confidence_reward"
FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL = "reference_confidence_weighted_kl"
FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL = (
    "reference_topk_confidence_weighted_union_kl")
#: The forms that read the CANDIDATE's own confidence.
STUDENT_FORMS = (FORM_UNCERTAINTY_WEIGHTED_KL, FORM_KL_MINUS_CONFIDENCE_REWARD)
#: The forms whose weight is the REFERENCE's own confidence.
REFERENCE_FORMS = (FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL,
                   FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL)
FORMS = (*STUDENT_FORMS, *REFERENCE_FORMS)

#: Every definitional choice, named so it is part of the identity rather than
#: of the reader's assumptions. Changing any of them is a new objective.
ENTROPY_DEFINITION = "student_own_topk_conditional_entropy/v1"
CONFIDENCE_MAPPING = "one_minus_entropy_over_log_k/v1"
DENOMINATOR = "scored_position_count/v1"
#: How per-item values are combined ACROSS items: by the same weighting each
#: consumer's KL-only path uses, so the integrand is the only difference. DEPTH
#: averages items equally within a sub-type (as its KL-only path does); the
#: state evaluator weights items by T_x within a sub-type and pools each
#: critical class over items (as its KL-only path does). v1 averaged items
#: equally in the state evaluator too, which changed the length weighting
#: against D1 and was replaced before any run used it.
AGGREGATION = "kl_path_weighting_per_consumer/v2"
DIRECTION = "minimize"

#: The reference-confidence form's definitional choices, serialized ONLY for
#: that form so every student-confidence identity stays byte-identical.
REFERENCE_ENTROPY_DEFINITION = "reference_full_vocabulary_shannon_entropy/v1"
REFERENCE_CONFIDENCE_MAPPING = "one_minus_entropy_over_log_vocabulary/v1"
#: The KL's reference for every consumer: the frozen model at the root of the
#: lineage. The only value this form accepts; a parent-referenced variant
#: would be a different, undeclared objective.
KL_REFERENCE_ROOT = "root_reference/v1"
KL_REFERENCES = (KL_REFERENCE_ROOT,)
#: Activation statistics weighted by the same reference confidence, on the
#: token axis: token ``j`` feeds prediction ``j`` and carries its weight; the
#: last token feeds no prediction inside the item and carries zero.
STATISTICS_WEIGHTING = "reference_confidence_token_weights/v1"

#: The union-KL form's definitional choices, serialized ONLY for that form.
REFERENCE_TOPK_ENTROPY_DEFINITION = (
    "reference_topk_plus_tail_absolute_shannon_entropy/v1")
REFERENCE_TOPK_CATEGORIES = "top_k entries + 1 tail bucket when top_k < |V|"
REFERENCE_TOPK_CONFIDENCE_MAPPING = "one_minus_entropy_over_log_categories/v1"
REFERENCE_TOPK_STATISTICS_WEIGHTING = (
    "reference_topk_confidence_token_weights/v1")
#: The KL support the union form is defined on, by id; its top_k is the
#: confidence's K.
UNION_KL_SUPPORT = SUPPORT_REFERENCE_STUDENT_UNION_TOPK_TAIL_V1

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
    #: The student's Top-K for the student-confidence forms; absent (None) for
    #: the reference-confidence form, which reads no student entropy.
    entropy_top_k: int | None = None
    #: ``lambda`` of the reward form; must be absent for the weighted form.
    reward_scale: float | None = None
    #: The KL's reference, declared. Required by the reference-confidence form
    #: and absent from the student forms (their consumers keep their
    #: historical references), so their identities do not move.
    kl_reference: str | None = None
    #: Whether the reference confidence also weights the activation statistics
    #: of the structural operators. Reference-confidence form only.
    weights_activation_statistics: bool = False

    def __post_init__(self) -> None:
        if not self.objective_id:
            raise ObjectiveError("an objective needs an id")
        if self.form not in FORMS:
            raise ObjectiveError(f"unknown objective form {self.form!r}; "
                                 f"known: {FORMS}")
        if self.form == FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL:
            k = self.entropy_top_k
            if (not isinstance(k, int) or isinstance(k, bool) or k < 1):
                raise ObjectiveError(
                    f"{self.objective_id}: the union form needs the "
                    f"reference's confidence Top-K as an integer >= 1, got "
                    f"{k!r}; it is the experiment's and has no default")
            if self.reward_scale is not None:
                raise ObjectiveError(
                    f"{self.objective_id}: the union form has no reward term")
            if self.kl_reference not in KL_REFERENCES:
                raise ObjectiveError(
                    f"{self.objective_id}: the union form needs kl_reference "
                    f"in {KL_REFERENCES}, got {self.kl_reference!r}")
            return
        if self.form == FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL:
            if self.entropy_top_k is not None or self.reward_scale is not None:
                raise ObjectiveError(
                    f"{self.objective_id}: the reference-confidence form reads "
                    "no student Top-K and has no reward; entropy_top_k and "
                    "reward_scale would be identity fields that change nothing")
            if self.kl_reference not in KL_REFERENCES:
                raise ObjectiveError(
                    f"{self.objective_id}: the reference-confidence form needs "
                    f"kl_reference in {KL_REFERENCES}, got "
                    f"{self.kl_reference!r}")
            return
        if self.kl_reference is not None or self.weights_activation_statistics:
            raise ObjectiveError(
                f"{self.objective_id}: kl_reference and "
                "weights_activation_statistics belong to the "
                "reference-confidence form")
        if self.entropy_top_k is None or int(self.entropy_top_k) < 2:
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

    @property
    def uses_reference_confidence(self) -> bool:
        return self.form in REFERENCE_FORMS

    @property
    def uses_union_support(self) -> bool:
        """The KL is the reference ∪ candidate Top-K + tail KL, and the weight
        the reference's Top-K + tail confidence."""
        return self.form == FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL

    @property
    def reference_confidence_definition(self) -> tuple[str, str] | None:
        """``(entropy definition, mapping)`` of the reference confidence this
        form weights by, or ``None`` for a student form."""
        if self.form == FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL:
            return REFERENCE_ENTROPY_DEFINITION, REFERENCE_CONFIDENCE_MAPPING
        if self.uses_union_support:
            return (REFERENCE_TOPK_ENTROPY_DEFINITION,
                    REFERENCE_TOPK_CONFIDENCE_MAPPING)
        return None

    @property
    def uses_student_confidence(self) -> bool:
        return self.form in STUDENT_FORMS

    @property
    def requires_root_reference(self) -> bool:
        """The KL's reference is the lineage root for EVERY consumer."""
        return self.kl_reference == KL_REFERENCE_ROOT

    def components(self) -> tuple[str, ...]:
        """The per-position quantities this objective's records carry."""
        if self.uses_union_support:
            return UNION_COMPONENTS
        return (REFERENCE_COMPONENTS if self.uses_reference_confidence
                else COMPONENTS)

    def confidence_key(self) -> str:
        return ("reference_confidence" if self.uses_reference_confidence
                else "student_confidence")

    def as_dict(self) -> dict[str, Any]:
        if self.uses_union_support:
            return {
                "schema": SCHEMA,
                "objective_id": self.objective_id,
                "version": int(self.version),
                "form": self.form,
                "confidence_source": REFERENCE_TOPK_ENTROPY_DEFINITION,
                "confidence_top_k": int(self.entropy_top_k),
                "confidence_categories": REFERENCE_TOPK_CATEGORIES,
                "confidence_mapping": REFERENCE_TOPK_CONFIDENCE_MAPPING,
                "kl_support": UNION_KL_SUPPORT,
                "kl_reference": self.kl_reference,
                "statistics_weighting": (REFERENCE_TOPK_STATISTICS_WEIGHTING
                                         if self.weights_activation_statistics
                                         else "none"),
                "denominator": DENOMINATOR,
                "aggregation": AGGREGATION,
                "direction": DIRECTION,
            }
        if self.uses_reference_confidence:
            return {
                "schema": SCHEMA,
                "objective_id": self.objective_id,
                "version": int(self.version),
                "form": self.form,
                "confidence_source": REFERENCE_ENTROPY_DEFINITION,
                "confidence_mapping": REFERENCE_CONFIDENCE_MAPPING,
                "kl_reference": self.kl_reference,
                "statistics_weighting": (STATISTICS_WEIGHTING
                                         if self.weights_activation_statistics
                                         else "none"),
                "denominator": DENOMINATOR,
                "aggregation": AGGREGATION,
                "direction": DIRECTION,
            }
        out = {
            "schema": SCHEMA,
            "objective_id": self.objective_id,
            "version": int(self.version),
            "form": self.form,
            "entropy_definition": ENTROPY_DEFINITION,
            "entropy_top_k": int(self.entropy_top_k),
            "confidence_mapping": CONFIDENCE_MAPPING,
            "denominator": DENOMINATOR,
            "aggregation": AGGREGATION,
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
        if self.form in REFERENCE_FORMS:
            return "reference_confidence_weight"
        return ("uncertainty_weight" if self.form == FORM_UNCERTAINTY_WEIGHTED_KL
                else "confidence_reward")

    def integrand(self, kl: torch.Tensor, confidence: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(g, component)`` per position. ``+inf`` KL stays ``+inf``.

        ``confidence`` is the student's for the student forms and the
        reference's for the reference-confidence form; the caller supplies the
        one the form declares.
        """
        kl = kl.float()
        conf = confidence.float()
        if self.form in REFERENCE_FORMS:
            weight = conf
            g = torch.where(torch.isposinf(kl), kl, weight * kl)
            return g, weight
        if self.form == FORM_UNCERTAINTY_WEIGHTED_KL:
            weight = 1.0 - conf
            #: `weight * kl` is NaN at weight 0 and kl +inf; resolved to +inf
            #: on purpose (see the module docstring).
            g = torch.where(torch.isposinf(kl), kl, weight * kl)
            return g, weight
        reward = float(self.reward_scale) * conf
        return kl - reward, reward


#: WHICH IMPLEMENTATION READS AN OBJECTIVE, AND WHICH NEEDS THE ROOT. Here,
#: beside the objective, because both the operators and the planning layer
#: ask -- and an operators -> planning import would be a package cycle.
def objective_consumed_by(implementation: Any, objective: Any) -> bool:
    """Whether `implementation`'s SELECTION reads `objective`. One rule.

    A KL-scored operator (`consumes_scoring_objective`) reads any objective.
    An activation-statistics operator (`consumes_reference_weighted_statistics`)
    reads one only when that objective weights the activation statistics by
    the root reference's confidence -- under every other objective its
    statistics, and so its child, are the historical ones, and its config must
    not move.
    """
    if objective is None:
        return False
    if getattr(implementation, "consumes_scoring_objective", False):
        return True
    return bool(getattr(implementation, "consumes_reference_weighted_statistics",
                        False)
                and getattr(objective, "weights_activation_statistics", False))


def statistics_weighted_by_root(implementation: Any, objective: Any) -> bool:
    """Whether `implementation`'s activation statistics are weighted by the
    root reference's confidence under `objective`."""
    return bool(objective is not None
                and getattr(objective, "weights_activation_statistics", False)
                and getattr(implementation,
                            "consumes_reference_weighted_statistics", False))


def root_reference_needed(implementation: Any, objective: Any) -> bool:
    """Whether an invocation of `implementation` must be handed the ROOT
    reference store: a KL-scored operator under an objective whose reference
    is the root, or a weighted-statistics operator under an objective that
    weights them. One rule for the search, the fixed-path replay and
    `OperatorImplementation.execute`."""
    if objective is None:
        return False
    return bool((getattr(implementation, "consumes_scoring_objective", False)
                 and getattr(objective, "requires_root_reference", False))
                or statistics_weighted_by_root(implementation, objective))



def require_objective_support(objective: Any, support: Any) -> None:
    """An objective and a run's distribution support that belong together.

    The union support exists for the union-KL form and the union-KL form is
    defined on it, at the same ``K``: the support's retrieval needs the root
    store only that form requests, and the form's confidence ``K`` is the KL's.
    Every other pairing keeps its historical rule (checked where it always
    was). Raises :class:`ObjectiveError`.
    """
    union_support = bool(getattr(support, "is_union", False))
    union_form = bool(getattr(objective, "uses_union_support", False))
    if union_support and not union_form:
        raise ObjectiveError(
            f"distribution support {support} reads the reference's "
            "probability of candidate-only entries from the ROOT reference "
            "store, which only the union-KL objective form requests; "
            f"declared objective: "
            f"{None if objective is None else objective.qualified_id}")
    if union_form:
        if not union_support:
            raise ObjectiveError(
                f"{objective.qualified_id} is defined on the "
                f"{UNION_KL_SUPPORT} support; this run declares {support}")
        if int(support.top_k) != int(objective.entropy_top_k):
            raise ObjectiveError(
                f"{objective.qualified_id} weights by the reference's "
                f"Top-{objective.entropy_top_k} confidence and its KL is on "
                f"the union of Top-{objective.entropy_top_k} sets; the "
                f"support declares top_k={support.top_k}")


def reference_topk_confidence(support_log_probs: torch.Tensor,
                              tail_log_prob: torch.Tensor, *, top_k: int,
                              vocab_size: int
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(H^{K+1}, c)`` of the union form, from the reference's own sketch.

    ``c = 1 - H^{K+1} / log(categories)`` with the categories derived from
    ``top_k`` and the vocabulary (:func:`topk_tail_categories`); clamped to
    ``[0, 1]`` against rounding only. Candidate-independent: the sketch is the
    reference's.
    """
    categories = topk_tail_categories(top_k, vocab_size)
    if categories < 2:
        raise ObjectiveError(f"{categories} categories have no entropy scale")
    entropy = topk_tail_entropy(support_log_probs, tail_log_prob,
                                has_tail=int(top_k) < int(vocab_size))
    conf = (1.0 - entropy.double() / math.log(categories)).clamp(0.0, 1.0)
    return entropy, conf.float()


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


@torch.no_grad()
def full_vocabulary_entropy(logits: torch.Tensor, *,
                            chunk: int = DEFAULT_ROW_CHUNK) -> torch.Tensor:
    """Shannon entropy of ``softmax(logits)`` over the FULL last dimension.

    ``logits`` ``[..., V]``; returns float32 ``[...]``. Each block of rows is
    promoted to float32 once and reduced there (``-sum p log p`` with
    ``0 log 0 = 0``); no ``[rows, V]`` float32 copy outlives its block. ``V``
    is read from the tensor, never assumed.
    """
    vocab = int(logits.shape[-1])
    lead = logits.shape[:-1]
    flat = logits.reshape(-1, vocab)
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
    step = max(1, int(chunk))
    for start in range(0, int(flat.shape[0]), step):
        stop = min(start + step, int(flat.shape[0]))
        p = torch.softmax(flat[start:stop].float(), dim=-1)
        out[start:stop] = torch.special.entr(p).sum(dim=-1)
        del p
    return out.reshape(lead)


def confidence_from_vocabulary_entropy(entropy: torch.Tensor,
                                       vocab_size: int) -> torch.Tensor:
    """``c = 1 - H / log |V|``, clamped to ``[0, 1]`` against rounding only."""
    v = int(vocab_size)
    if v < 2:
        raise ObjectiveError(f"a vocabulary of {v} has no entropy scale")
    return (1.0 - entropy.float() / math.log(v)).clamp(0.0, 1.0)


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
#: The reference-confidence form's: no student quantity at all.
REFERENCE_COMPONENTS = ("objective", "raw_kl", "reference_entropy",
                        "reference_confidence", "component",
                        "reference_tail_mass")
#: The union-KL form's: the reference-confidence components, with ``raw_kl``
#: the UNWEIGHTED union KL, plus the union reduction's own evidence -- the
#: reference-Top-K + tail KL of the same pair (its lower bound), the union's
#: size and overlap with the reference's Top-K, and both masses outside it.
UNION_DIAGNOSTICS = ("kl_reference_topk", "union_size", "union_overlap",
                     "reference_union_tail_mass", "candidate_union_tail_mass")
UNION_COMPONENTS = (*REFERENCE_COMPONENTS, *UNION_DIAGNOSTICS)

#: Confidence histogram edges: ten equal bins on [0, 1].
CONFIDENCE_BINS = 10

#: The KL axis of the joint diagnostic histograms: a fixed, log-spaced
#: RECORDING GRID in nats, not a threshold. Flags are computed offline against
#: thresholds registered elsewhere; no decision reads this grid.
KL_GRID_EDGES = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0)


@torch.no_grad()
def position_terms(objective: ScoringObjective, kl: torch.Tensor,
                   cand_logits: torch.Tensor, log_z: torch.Tensor,
                   reference_tail_log: torch.Tensor | None, *,
                   reference_entropy: torch.Tensor | None = None,
                   reference_confidence: torch.Tensor | None = None,
                   union: Mapping[str, torch.Tensor] | None = None,
                   ) -> dict[str, torch.Tensor]:
    """Every per-position component, same leading shape as ``kl``.

    The reference-confidence forms take the reference's entropy and confidence
    from the caller -- aligned to ``kl`` position for position -- and read
    nothing of the candidate but the KL already computed from it. The union
    form also takes ``union``, the union reduction's per-position evidence
    (:meth:`~aadistill.initialization.scoring.support.UnionKLTerms.diagnostics`)
    aligned the same way, and refuses to run without it: a union-form record
    missing its lower bound and union size would not say what it measured.
    """
    tail = (torch.zeros_like(kl, dtype=torch.float32)
            if reference_tail_log is None
            else reference_tail_log.to(kl.device).float().exp())
    if objective.uses_reference_confidence:
        if reference_confidence is None or reference_entropy is None:
            raise ObjectiveError(
                f"{objective.qualified_id} weights by the reference's "
                "confidence and none was supplied")
        conf = reference_confidence.to(kl.device).float()
        if tuple(conf.shape) != tuple(kl.shape):
            raise ObjectiveError(
                f"reference confidence {tuple(conf.shape)} is not aligned with "
                f"the KL {tuple(kl.shape)}")
        g, component = objective.integrand(kl, conf)
        out = {"objective": g, "raw_kl": kl.float(),
               "reference_entropy": reference_entropy.to(kl.device).float(),
               "reference_confidence": conf, "component": component,
               "reference_tail_mass": tail}
        if objective.uses_union_support:
            if union is None:
                raise ObjectiveError(
                    f"{objective.qualified_id} needs the union reduction's "
                    "per-position evidence and none was supplied")
            for name in UNION_DIAGNOSTICS:
                value = union[name].to(kl.device).float()
                if tuple(value.shape) != tuple(kl.shape):
                    raise ObjectiveError(
                        f"union evidence {name!r} {tuple(value.shape)} is not "
                        f"aligned with the KL {tuple(kl.shape)}")
                out[name] = value
        return out
    entropy, mass = student_topk_terms(cand_logits, log_z,
                                       top_k=objective.entropy_top_k)
    entropy = entropy.to(kl.device)
    conf = confidence_from_entropy(entropy, objective.entropy_top_k)
    g, component = objective.integrand(kl, conf)
    return {"objective": g, "raw_kl": kl.float(), "student_entropy": entropy,
            "student_confidence": conf, "component": component,
            "student_topk_mass": mass.to(kl.device), "reference_tail_mass": tail}


def item_component_means(terms: Mapping[str, torch.Tensor],
                         prediction_mask: torch.Tensor,
                         weights: torch.Tensor | None = None,
                         names: tuple[str, ...] = COMPONENTS,
                         ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Each component's per-item fixed-denominator mean, and ``T_x``."""
    out: dict[str, torch.Tensor] = {}
    counts = None
    for name in names:
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


def add_joint_histograms(acc: dict[str, Any] | None,
                         terms: Mapping[str, torch.Tensor],
                         scored: torch.Tensor) -> dict[str, Any]:
    """Accumulate two joint histograms over scored positions.

    ``confidence_x_kl``: confidence decile x :data:`KL_GRID_EDGES` bin.
    ``confidence_x_topk_mass``: confidence decile x student Top-K mass decile
    -- where a high CONDITIONAL confidence sits on a Top-K that holds little of
    the student's actual probability. Threshold-free by construction: the
    records carry the grid, and any registered threshold is applied afterwards.
    """
    conf_key = ("student_confidence" if "student_confidence" in terms
                else "reference_confidence")
    keep = scored.to(terms[conf_key].device).bool()
    conf = terms[conf_key][keep].float()
    kl = terms["raw_kl"][keep].float()
    #: The reference-confidence form has no student Top-K mass; its second
    #: grid is then recorded empty rather than invented.
    mass = (terms["student_topk_mass"][keep].float()
            if "student_topk_mass" in terms else None)
    n_kl = len(KL_GRID_EDGES) + 1
    if acc is None:
        acc = {"confidence_edges": [i / CONFIDENCE_BINS
                                    for i in range(CONFIDENCE_BINS + 1)],
               "kl_edges": list(KL_GRID_EDGES),
               "topk_mass_edges": [i / CONFIDENCE_BINS
                                   for i in range(CONFIDENCE_BINS + 1)],
               "confidence_x_kl": [[0] * n_kl for _ in range(CONFIDENCE_BINS)],
               "confidence_x_topk_mass": [[0] * CONFIDENCE_BINS
                                          for _ in range(CONFIDENCE_BINS)]}
    if conf.numel() == 0:
        return acc
    c_bin = torch.clamp((conf * CONFIDENCE_BINS).long(), 0, CONFIDENCE_BINS - 1)
    edges = torch.tensor(KL_GRID_EDGES, dtype=kl.dtype, device=kl.device)
    k_bin = torch.bucketize(kl, edges, right=True)
    ck = torch.bincount(c_bin * n_kl + k_bin,
                        minlength=CONFIDENCE_BINS * n_kl).tolist()
    if mass is None:
        cm = [0] * (CONFIDENCE_BINS ** 2)
    else:
        m_bin = torch.clamp((mass * CONFIDENCE_BINS).long(), 0,
                            CONFIDENCE_BINS - 1)
        cm = torch.bincount(c_bin * CONFIDENCE_BINS + m_bin,
                            minlength=CONFIDENCE_BINS ** 2).tolist()
    for i in range(CONFIDENCE_BINS):
        for j in range(n_kl):
            acc["confidence_x_kl"][i][j] += int(ck[i * n_kl + j])
        for j in range(CONFIDENCE_BINS):
            acc["confidence_x_topk_mass"][i][j] += int(
                cm[i * CONFIDENCE_BINS + j])
    return acc


__all__ = ["AGGREGATION", "COMPONENTS", "CONFIDENCE_BINS",
           "CONFIDENCE_MAPPING", "DENOMINATOR", "DIRECTION",
           "ENTROPY_DEFINITION", "FORM_KL_MINUS_CONFIDENCE_REWARD",
           "FORM_REFERENCE_CONFIDENCE_WEIGHTED_KL",
           "FORM_REFERENCE_TOPK_CONFIDENCE_WEIGHTED_UNION_KL",
           "FORM_UNCERTAINTY_WEIGHTED_KL", "FORMS", "KL_GRID_EDGES",
           "KL_REFERENCE_ROOT", "KL_REFERENCES", "REFERENCE_COMPONENTS",
           "REFERENCE_CONFIDENCE_MAPPING", "REFERENCE_ENTROPY_DEFINITION",
           "REFERENCE_FORMS", "REFERENCE_TOPK_CATEGORIES",
           "REFERENCE_TOPK_CONFIDENCE_MAPPING",
           "REFERENCE_TOPK_ENTROPY_DEFINITION",
           "REFERENCE_TOPK_STATISTICS_WEIGHTING", "UNION_COMPONENTS",
           "UNION_DIAGNOSTICS", "UNION_KL_SUPPORT",
           "STATISTICS_WEIGHTING", "STUDENT_FORMS",
           "reference_topk_confidence", "require_objective_support",
           "ObjectiveError", "ScoringObjective", "add_joint_histograms",
           "confidence_from_entropy", "confidence_from_vocabulary_entropy",
           "confidence_histogram", "fixed_denominator_item_means",
           "full_vocabulary_entropy", "item_component_means",
           "objective_config", "objective_consumed_by", "position_terms",
           "require_binary_scoring_weights", "root_reference_needed",
           "statistics_weighted_by_root", "student_topk_terms"]
