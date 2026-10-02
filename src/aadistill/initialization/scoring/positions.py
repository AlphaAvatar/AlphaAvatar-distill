"""Position weighting as a first-class, hash-bound policy.

Four calibrated operators and one global evaluator all reduce *something* over
*some set of positions*, and until now each of them decided that set for itself
— implicitly, as "every real token". That is a scientific choice, and the
project has never been able to vary it, record it, or compare two answers to it,
because it was not a value anywhere.

This module makes it one. A :class:`ScoringPositionPolicy` turns an item into a
non-negative weight per position, and **every** reduction that a structural
decision reads is then the same shape:

    score = sum_t w_t * value_t / sum_t w_t

with ``w_t = 0`` for a padded position. Three consequences are mechanical:

* a padding weight of zero can never enter a denominator;
* batch composition cannot change a domain weighting, because an item's own
  weights travel with the item and not with the forward it shared;
* an operator and the beam metric that prunes it consume the *same* policy
  object, so "target-aware operator, full-sequence beam" is not expressible.

**Two axes, deliberately named.** A transformer item of ``L`` tokens has ``L``
token positions and ``L - 1`` prediction positions, and the two are not
interchangeable:

* :data:`PREDICTION_AXIS` — what a KL, a cross-entropy or a top-1 agreement
  reduces over. Prediction position ``j`` predicts token ``j + 1``.
* :data:`TOKEN_AXIS` — what an activation statistic accumulates over: a residual
  second moment, an FFN activation magnitude, an attention-output second moment.

A policy answers for both, and the answers are allowed to differ. That is not an
inconsistency; it is the one place the relationship between an activation and
the prediction it feeds is written down.

**The default is exactly today's behaviour, and that is load-bearing.**
:data:`ALL_POSITIONS_V1` returns weight ``1`` for every real position on either
axis, which is what every committed operator and every frozen artifact in this
repository was computed under. The reducers below detect that case and skip the
weighting arithmetic entirely rather than multiplying by ``1.0``, so the
reference path performs the operations it always performed — the same argument
``ActivationStatsCollector._keep_valid`` already makes about padding, and the
reason a historical checkpoint digest stays reproducible after this module
exists.

**The supervised-target policy reads the mixture, not a hardcoded rule.**
:data:`SUPERVISED_TARGET_V1` takes the item's own ``assistant`` tag — the
prediction positions whose target token belongs to the supervised assistant
turn, computed when the mixture was built and hashed with it — and falls back to
every real prediction position when an item carries no such tag at all. That
fallback is not a convenience: an untemplated raw-LM document has no assistant
turn, and every one of its next-token predictions is a legitimate supervised
position. Which items are which is a property of the frozen mixture, so this
policy introduces no data change.

A policy id is immutable and its declared semantics hash to
:attr:`ScoringPositionPolicy.policy_hash`. Changing what a policy computes means
a new id, never a redefinition — the rule the operator registry already
enforces, for the same reason: a manifest that recorded a policy id has to stay
interpretable.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

#: Positions a next-token objective reduces over: ``L - 1`` of them, where
#: position ``j`` predicts token ``j + 1``.
PREDICTION_AXIS = "prediction"

#: Positions an activation statistic accumulates over: ``L`` of them, one per
#: real token of the item.
TOKEN_AXIS = "token"

POSITION_AXES = (PREDICTION_AXIS, TOKEN_AXIS)

#: Which tag names a policy may consult. Not a closed world — a policy declares
#: what it reads and the declaration is hashed — but naming the ones the frozen
#: mixtures actually carry keeps a typo from silently selecting nothing.
#: Produced by `scripts/data/build_e8_calibration.py :: tag_positions` and by
#: the state-eval suite builder, both as PREDICTION-position indices.
KNOWN_TAGS = ("assistant", "reasoning", "final_answer", "think_close", "eos",
              "tool_close")

#: The accumulation dtype for every weight vector. float64 because the weights
#: enter a denominator and the project's statistics accumulate in float64; a
#: float32 weight sum over 60k positions would lose bits the divisor needs.
WEIGHT_DTYPE = torch.float64


class ScoringPositionError(RuntimeError):
    """A position policy could not be applied to an item."""


class PositionAxis:
    """Namespace for the two axis constants, for callers that prefer a dotted
    name over a bare string. The strings are the canonical values: they are what
    gets serialized into evidence."""

    PREDICTION = PREDICTION_AXIS
    TOKEN = TOKEN_AXIS


# --- the value a policy returns ---------------------------------------------

#: Every real position carries weight 1. The reference arithmetic applies.
FORM_ALL = "all"
#: Weights are 0 or 1. A reducer may implement this by selecting rows, which is
#: both cheaper and numerically cleaner than multiplying by a 0/1 float.
FORM_SELECT = "select"
#: General non-negative weights. Requires a multiply.
FORM_WEIGHTED = "weighted"


@dataclass(frozen=True)
class PositionWeights:
    """One item's weights on one axis, plus who the item is.

    ``item_id``, ``domain`` and ``subtype`` travel with the weights because the
    reductions above this layer are per-item and then per-subtype and then
    per-domain, all deliberately unweighted at those levels
    (:func:`~aadistill.initialization.statistics.contribution.domain_balanced_score`
    owns that decision and this module does not reopen it). Carrying the
    ownership here is what lets a batched reducer attribute a row to the right
    subtype without the batcher knowing that subtypes exist.
    """

    axis: str
    weights: torch.Tensor
    policy_id: str
    item_id: str | None = None
    domain: str | None = None
    subtype: str | None = None

    def __post_init__(self) -> None:
        if self.axis not in POSITION_AXES:
            raise ScoringPositionError(
                f"unknown position axis {self.axis!r}; known: {list(POSITION_AXES)}")
        w = self.weights
        if not isinstance(w, torch.Tensor):
            raise ScoringPositionError(
                f"{self.where()}: weights are {type(w).__name__}, not a torch.Tensor")
        if w.dim() != 1:
            raise ScoringPositionError(
                f"{self.where()}: weights have shape {tuple(w.shape)}; one axis, "
                "one dimension")
        if w.numel() == 0:
            raise ScoringPositionError(f"{self.where()}: no positions")
        if bool((w < 0).any()):
            raise ScoringPositionError(
                f"{self.where()}: negative position weight. A negative weight is "
                "not a weighting, it is a sign flip in the objective")
        if not bool(torch.isfinite(w).all()):
            raise ScoringPositionError(f"{self.where()}: non-finite position weight")

    def where(self) -> str:
        return f"{self.policy_id}[{self.axis}]" + (
            f" item {self.item_id!r}" if self.item_id else "")

    @property
    def n_positions(self) -> int:
        return int(self.weights.numel())

    @property
    def total(self) -> float:
        """``sum_t w_t`` — the denominator of every reduction under this policy."""
        return float(self.weights.sum())

    @property
    def form(self) -> str:
        """Which of the three reduction strategies this vector permits.

        Classified from the values rather than declared by the policy, so a
        policy cannot claim a cheaper form than its numbers support.
        """
        w = self.weights
        if bool((w == 1).all()):
            return FORM_ALL
        if bool(((w == 0) | (w == 1)).all()):
            return FORM_SELECT
        return FORM_WEIGHTED

    @property
    def n_active(self) -> int:
        return int((self.weights > 0).sum())

    def active_mask(self) -> torch.Tensor:
        """``[n]`` bool over positions with non-zero weight."""
        return self.weights > 0

    def require_nonempty(self) -> None:
        if self.total <= 0.0:
            raise ScoringPositionError(
                f"{self.where()}: every position has weight zero, so this item "
                "has no score under this policy. An item the policy cannot "
                "score must be excluded from the mixture, not divided by zero")

    def as_dict(self) -> dict[str, Any]:
        """Evidence, not the vector. The weights themselves are per-position and
        belong in an operator's own trace if anywhere; what a record needs is
        what they summed to and how many positions survived."""
        return {"axis": self.axis, "policy_id": self.policy_id,
                "item_id": self.item_id, "domain": self.domain,
                "subtype": self.subtype, "n_positions": self.n_positions,
                "n_active": self.n_active, "weight_total": self.total,
                "form": self.form}


# --- the policy contract ----------------------------------------------------


class ScoringPositionPolicy(ABC):
    """One versioned rule for which positions a structural decision may read."""

    policy_id: str = ""
    version: int = 0
    description: str = ""
    #: Tag names this policy consults. Declared so the hash covers it and so a
    #: reader can tell a policy that reads the mixture's supervision structure
    #: from one that ignores it.
    reads_tags: tuple[str, ...] = ()
    #: What this policy does with an item that carries none of `reads_tags`.
    #: Stated rather than left to be read out of the code, because it is the
    #: whole treatment of the raw-LM part of the current frozen mixtures.
    untagged_behaviour: str = ""

    def signature(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "version": self.version,
                "reads_tags": sorted(self.reads_tags),
                "untagged_behaviour": self.untagged_behaviour}

    @property
    def policy_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.signature(), sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()

    @property
    def qualified_id(self) -> str:
        return f"{self.policy_id}@v{self.version}"

    def declare(self) -> dict[str, Any]:
        return {**self.signature(), "description": self.description,
                "qualified_id": self.qualified_id,
                "policy_hash": self.policy_hash}

    # --- behaviour ---------------------------------------------------------

    @abstractmethod
    def prediction_weights(self, *, n_predictions: int,
                           tags: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[n_predictions]`` non-negative weights over prediction positions."""

    @abstractmethod
    def token_weights(self, *, n_tokens: int,
                      tags: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``[n_tokens]`` non-negative weights over token positions."""

    def weights(self, item: Any, *, axis: str) -> PositionWeights:
        """The weights for one item on one axis, with the item's ownership.

        ``item`` is either a calibration-mixture mapping (tags stored as
        prediction-position index lists) or a
        :class:`~aadistill.initialization.specs.metrics.SuiteItem` (tags stored
        as boolean masks). :func:`normalized_prediction_tags` reconciles the two
        so a policy never has to know which it was handed.
        """
        n_tokens, tags, meta = _item_view(item)
        if axis == PREDICTION_AXIS:
            n = n_tokens - 1
            if n < 1:
                raise ScoringPositionError(
                    f"{self.policy_id}: item {meta.get('item_id')!r} holds "
                    f"{n_tokens} token(s) and therefore predicts nothing")
            w = self.prediction_weights(n_predictions=n, tags=tags)
        elif axis == TOKEN_AXIS:
            w = self.token_weights(n_tokens=n_tokens, tags=tags)
            n = n_tokens
        else:
            raise ScoringPositionError(
                f"unknown position axis {axis!r}; known: {list(POSITION_AXES)}")
        if not isinstance(w, torch.Tensor) or w.dim() != 1 or int(w.numel()) != n:
            got = tuple(w.shape) if isinstance(w, torch.Tensor) else type(w).__name__
            raise ScoringPositionError(
                f"{self.policy_id}: returned {got} weights for {n} positions on "
                f"the {axis} axis")
        return PositionWeights(axis=axis, weights=w.to(WEIGHT_DTYPE),
                               policy_id=self.qualified_id,
                               item_id=meta.get("item_id"),
                               domain=meta.get("domain"),
                               subtype=meta.get("subtype"))


class AllPositionsV1(ScoringPositionPolicy):
    """Every real position, weight one. The incumbent semantics.

    Registered, named and hashed rather than left implicit, because "the
    objective was computed over all positions" is a claim a record should be
    able to state. Its numerical contract is that it changes nothing: a reducer
    sees :data:`FORM_ALL` and takes the path it took before this module existed.
    """

    policy_id = "positions.all_v1"
    version = 1
    description = ("every real position carries weight one, on both axes; the "
                   "semantics every committed operator result was computed "
                   "under")
    reads_tags = ()
    untagged_behaviour = "irrelevant: no tag is consulted"

    def prediction_weights(self, *, n_predictions: int,
                           tags: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.ones(n_predictions, dtype=WEIGHT_DTYPE)

    def token_weights(self, *, n_tokens: int,
                      tags: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.ones(n_tokens, dtype=WEIGHT_DTYPE)


class SupervisedTargetV1(ScoringPositionPolicy):
    """Only the positions the item's own objective contract supervises.

    On the prediction axis this is the item's ``assistant`` tag: the positions
    whose *target* token belongs to the supervised assistant turn. The tag is
    computed when a mixture is built and hashed with it, so this policy reads
    the mixture's recorded supervision structure rather than re-deriving a
    chat-template rule at scoring time.

    On the token axis, a position is active exactly when the prediction it feeds
    is active. Activation at token position ``j`` is what produces the logits at
    prediction position ``j``, so the token-axis vector is the prediction-axis
    vector with a zero appended: the final token of an item feeds no prediction
    *inside* the item and therefore carries no supervised signal.

    **That last zero is a real change from the incumbent and is deliberate.**
    :data:`ALL_POSITIONS_V1` accumulates activation statistics over all ``L``
    token positions; this policy accumulates over at most ``L - 1``. For an
    untagged raw-LM item — where every prediction stays active — the difference
    is exactly the item's final token. It follows from defining the statistic
    over *positions whose prediction is supervised* rather than over *tokens*,
    which is the hypothesis, not an accident of implementation.

    An item carrying no ``assistant`` tag at all is an untemplated document with
    no assistant turn, and every one of its next-token predictions is a
    legitimate supervised position. It is NOT given zero weight: that would
    silently drop the general-language domain out of the mixture and change the
    data, which this policy must not do.
    """

    policy_id = "positions.supervised_target_v1"
    version = 1
    description = (
        "prediction positions whose target token is supervised by the item's "
        "own objective contract, read from the mixture's `assistant` tag; every "
        "real prediction position for an item that carries no such tag; on the "
        "token axis, the position that feeds each active prediction")
    reads_tags = ("assistant",)
    untagged_behaviour = (
        "an item with no `assistant` tag is untemplated raw text with no "
        "assistant turn; all of its real prediction positions stay active")

    #: The tag that defines supervision. A single name, declared, so the policy
    #: hash covers it and a future policy that supervised `final_answer` only
    #: would be a different id rather than a different constant.
    SUPERVISION_TAG = "assistant"

    def _prediction_mask(self, n_predictions: int,
                         tags: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = tags.get(self.SUPERVISION_TAG)
        if mask is None or not bool(mask.any()):
            return torch.ones(n_predictions, dtype=torch.bool)
        if int(mask.numel()) != n_predictions:
            raise ScoringPositionError(
                f"{self.policy_id}: the {self.SUPERVISION_TAG!r} tag covers "
                f"{int(mask.numel())} positions but the item predicts "
                f"{n_predictions}")
        return mask.bool()

    def prediction_weights(self, *, n_predictions: int,
                           tags: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self._prediction_mask(n_predictions, tags).to(WEIGHT_DTYPE)

    def token_weights(self, *, n_tokens: int,
                      tags: Mapping[str, torch.Tensor]) -> torch.Tensor:
        pred = self._prediction_mask(n_tokens - 1, tags).to(WEIGHT_DTYPE)
        #: The appended zero is the final token, which feeds no prediction in
        #: this item. See the class docstring: this is the definition, not an
        #: off-by-one.
        return torch.cat([pred, torch.zeros(1, dtype=WEIGHT_DTYPE)])


ALL_POSITIONS_V1 = AllPositionsV1()
SUPERVISED_TARGET_V1 = SupervisedTargetV1()

#: The policies this module ships, resolvable WITHOUT registration.
#:
#: Deliberately not a registration side effect at import. The operator registry
#: makes that mistake expensive — `BeamSearch._allowed_impl_ids` falls back to
#: every registered implementation, so a leaked import-time registration
#: silently adds a branch to an unrelated search — and a core module whose
#: contents depend on who imported what first is a module nobody can reason
#: about. `scripts/architecture/core_boundaries.py` enforces the rule.
#:
#: But resolution must not depend on an application remembering to call a
#: registrar either: `resolve_position_policy` runs deep inside a fixed-path
#: execution, replaying a plan that may have been frozen months earlier, and
#: "the id is unknown because nobody registered it" would be a failure of
#: bookkeeping dressed up as a failure of identity. So the shipped ids are
#: ALWAYS resolvable, from this immutable mapping, and the mutable registry
#: below exists for policies defined elsewhere.
SHIPPED_POLICIES: Mapping[str, ScoringPositionPolicy] = MappingProxyType({
    ALL_POSITIONS_V1.qualified_id: ALL_POSITIONS_V1,
    SUPERVISED_TARGET_V1.qualified_id: SUPERVISED_TARGET_V1,
})


# --- registry ---------------------------------------------------------------
#
# Same shape and same reason as the operator registry: an id is permanently
# bound to a declared semantics, so a manifest that recorded one stays
# interpretable. For policies defined OUTSIDE this module — a future
# experiment's own weighting — registration is explicit and never at import.

_POLICIES: dict[str, ScoringPositionPolicy] = {}


def register_position_policy(policy: ScoringPositionPolicy, *,
                             replace: bool = False) -> ScoringPositionPolicy:
    if not policy.policy_id:
        raise ScoringPositionError("a position policy must declare a policy_id")
    key = policy.qualified_id
    existing = _POLICIES.get(key) or SHIPPED_POLICIES.get(key)
    if existing is not None and not replace:
        if existing.policy_hash != policy.policy_hash:
            raise ScoringPositionError(
                f"{key} is already registered with a different declaration "
                f"({existing.policy_hash[:12]} vs {policy.policy_hash[:12]}); a "
                "changed rule needs a new version, not a redefinition — every "
                "record that cited this id would otherwise mean something else")
        return existing
    _POLICIES[key] = policy
    return policy


def get_position_policy(qualified_id: str) -> ScoringPositionPolicy:
    """A policy by qualified id: a shipped one, or one a caller registered.

    Shipped first, so an id this module defines cannot be shadowed by a
    registration — a frozen plan citing `positions.all_v1@v1` has to resolve to
    the semantics that id was recorded under, whatever else is in the process.
    """
    policy = SHIPPED_POLICIES.get(qualified_id) or _POLICIES.get(qualified_id)
    if policy is None:
        raise KeyError(
            f"no scoring position policy {qualified_id!r}; shipped: "
            f"{sorted(SHIPPED_POLICIES)}; registered: {sorted(_POLICIES)}")
    return policy


def registered_position_policies() -> list[str]:
    """Every resolvable id, shipped and registered."""
    return sorted(set(SHIPPED_POLICIES) | set(_POLICIES))


def unregister_position_policy(qualified_id: str) -> None:
    """Test-only, and it cannot remove a shipped policy.

    A shipped id is part of this module rather than of the process's state, so
    there is nothing to unregister; letting a test remove one would let it
    construct a process in which a frozen plan no longer resolves.
    """
    _POLICIES.pop(qualified_id, None)


# --- item normalization -----------------------------------------------------


def normalized_prediction_tags(raw: Mapping[str, Any] | None,
                              n_predictions: int,
                              *, where: str = "") -> dict[str, torch.Tensor]:
    """Tags as boolean masks over prediction positions, from either stored form.

    A frozen calibration mixture stores each tag as a **list of
    prediction-position indices** (``tag_positions`` in the mixture builder);
    a loaded :class:`SuiteItem` carries the same information as a **boolean
    mask**. Both reach a policy through here, so a policy never branches on the
    form and the two can never be read with different conventions — which is
    exactly the mistake the state-eval loader's own comment warns about ("read
    as masks they would be the wrong length and silently reweight").

    Out-of-range indices and wrong-length masks raise. A tag that silently
    selected nothing would make a target-aware objective quietly
    full-sequence.
    """
    out: dict[str, torch.Tensor] = {}
    for name, value in dict(raw or {}).items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.bool:
            if int(value.numel()) != n_predictions:
                raise ScoringPositionError(
                    f"{where}tag {name!r} is a mask over {int(value.numel())} "
                    f"positions but the item predicts {n_predictions}")
            out[name] = value
            continue
        mask = torch.zeros(n_predictions, dtype=torch.bool)
        index = torch.as_tensor(list(value), dtype=torch.long)
        if index.numel():
            if int(index.min()) < 0 or int(index.max()) >= n_predictions:
                raise ScoringPositionError(
                    f"{where}tag {name!r} indexes prediction position "
                    f"[{int(index.min())}, {int(index.max())}] outside "
                    f"[0, {n_predictions - 1}]")
            mask[index] = True
        out[name] = mask
    return out


def _item_view(item: Any) -> tuple[int, dict[str, torch.Tensor], dict[str, Any]]:
    """``(n_tokens, prediction tags, ownership)`` for either item form."""
    if isinstance(item, Mapping):
        ids = item.get("input_ids")
        if isinstance(ids, torch.Tensor):
            n_tokens = int(ids.shape[-1])
        elif isinstance(item.get("ids"), Sequence):
            n_tokens = len(item["ids"])
        else:
            raise ScoringPositionError(
                "a calibration item needs 'input_ids' (a [1, T] tensor) or 'ids' "
                "(a token list) before a position policy can weight it; it "
                "reaches an operator through `prepare_calibration_items`, which "
                "guarantees the former")
        raw_tags = item.get("tags")
        meta = {k: item.get(k) for k in ("item_id", "domain", "subtype")}
    else:
        ids = getattr(item, "input_ids", None)
        if not isinstance(ids, torch.Tensor):
            raise ScoringPositionError(
                f"{type(item).__name__} carries no `input_ids` tensor, so a "
                "position policy cannot tell how many positions it has")
        n_tokens = int(ids.shape[-1])
        raw_tags = getattr(item, "tags", None)
        meta = {k: getattr(item, k, None)
                for k in ("item_id", "domain", "subtype")}
    if n_tokens < 1:
        raise ScoringPositionError(
            f"item {meta.get('item_id')!r} holds no tokens")
    where = f"item {meta['item_id']!r}: " if meta.get("item_id") else ""
    tags = normalized_prediction_tags(raw_tags, n_tokens - 1, where=where)
    return n_tokens, tags, meta


# --- reading a policy out of a declared operator config ---------------------

#: The two keys an operator config uses to NAME its policy. They reach
#: `OperatorStep.config_hash` and therefore the state id, which is why the id and
#: the hash are both carried: the id is what a reader looks up, the hash is what
#: makes a silently-redefined policy impossible to pass off as the recorded one.
POLICY_CONFIG_KEY = "position_policy"
POLICY_HASH_CONFIG_KEY = "position_policy_hash"


def resolve_position_policy(config: Mapping[str, Any] | None, *,
                            where: str = "") -> ScoringPositionPolicy:
    """The policy a config names, from the registry — never from a caller's object.

    A plan DECLARES a policy id; the executor RESOLVES it here. Those are
    deliberately not the same step: if the executor also accepted a policy
    object it would have two sources for one fact, and the one that reached the
    arithmetic would not be the one that reached the state id.

    A declared hash that disagrees with the registered policy raises. That is the
    case where an id has been rebound to different semantics between recording
    and replay — the failure the registry's own refusal exists to prevent, caught
    again here because a frozen plan outlives the process that wrote it.
    """
    config = dict(config or {})
    named = config.get(POLICY_CONFIG_KEY)
    declared_hash = config.get(POLICY_HASH_CONFIG_KEY)
    if named is None:
        if declared_hash is not None:
            raise ScoringPositionError(
                f"{where}declares {POLICY_HASH_CONFIG_KEY}="
                f"{str(declared_hash)[:12]} with no {POLICY_CONFIG_KEY}; a hash "
                "with no id names nothing a reader can resolve")
        return ALL_POSITIONS_V1
    policy = get_position_policy(str(named))
    if declared_hash is not None and str(declared_hash) != policy.policy_hash:
        raise ScoringPositionError(
            f"{where}declares position policy {named!r} at hash "
            f"{str(declared_hash)[:12]} but the registered {policy.qualified_id} "
            f"hashes to {policy.policy_hash[:12]}. An id rebound to different "
            "semantics cannot replay a record that cited it")
    return policy


def policy_config(policy: ScoringPositionPolicy) -> dict[str, str]:
    """The two config keys that name ``policy``, or ``{}`` at the incumbent.

    Empty at :data:`ALL_POSITIONS_V1` so that a run under the incumbent
    semantics hashes an operator config identical to every committed one. The
    omission is the compatibility guarantee, not an oversight: a historical state
    id must stay derivable from the code that is live now.
    """
    if policy.policy_hash == ALL_POSITIONS_V1.policy_hash:
        return {}
    return {POLICY_CONFIG_KEY: policy.qualified_id,
            POLICY_HASH_CONFIG_KEY: policy.policy_hash}


def weights_for_items(items: Sequence[Any], policy: ScoringPositionPolicy, *,
                      axis: str) -> list[PositionWeights]:
    """One :class:`PositionWeights` per item, in the items' own order.

    Every item is required to carry positive weight: a mixture containing an
    item this policy cannot score is a mixture/policy mismatch, and the two
    places that would otherwise absorb it — a zero denominator and a silently
    dropped domain — are both worse than a refusal.
    """
    out = []
    for item in items:
        w = policy.weights(item, axis=axis)
        w.require_nonempty()
        out.append(w)
    return out
