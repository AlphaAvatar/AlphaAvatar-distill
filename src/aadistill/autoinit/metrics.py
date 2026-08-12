"""The metric taxonomy, kept separate in code and not only in prose.

Five levels, in strict order, none of them substitutable for the one below:

1. **operator-local objective** — namespace ``op.``. Algorithm-specific: the
   causal depth search's parent->candidate KL, the PCA's captured energy, the
   FFN's activation importance. Its reference is the operator's *parent* state.
2. **global state evaluation** — namespace ``state.``. Every produced checkpoint,
   measured against the **original teacher** on a frozen suite, hash-bound to the
   weights it was measured on.
3. **beam ranking policy** — ``ranking.py``. Consumes level 2 only.
4. **recovery-probe selection** — ``recovery.py``. Consumes autonomous behaviour,
   not state NLL.
5. **final promotion** — isolated from the search entirely (``datasets.py``).

Level 1 becoming level 3 is the specific failure mode this file is built to
block. E8a is the reason: its operator-local objective (full-width ablation KL)
was 3.11x better for the contribution map, and the resulting fully-compressed
initializer was 2.8 nats *worse*. An operator-local win is evidence about the
operator, not about the state. So the namespaces are enforced: ``metric_level``
raises on an unnamespaced key, and ``BeamRankingPolicy`` refuses any objective
outside ``state.``.

The second rule with teeth is hash binding. ``StateEvaluation`` carries the
sha256 of the weights it was measured on, and ``InitializationState`` refuses an
evaluation whose hash is not its own — which is what makes "no inherited NLL"
mechanical rather than aspirational (E8's ``init/nll_gate.py`` established the
same rule for initialization NLL).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from ..init.contribution import DistortionSums, distortion, domain_balanced_score


class MetricLevel(Enum):
    OPERATOR_LOCAL = "operator_local"
    STATE_EVALUATION = "state_evaluation"
    BEAM_RANKING = "beam_ranking"
    RECOVERY_SELECTION = "recovery_selection"
    FINAL_PROMOTION = "final_promotion"


OP_PREFIX = "op."
STATE_PREFIX = "state."


class MetricNamespaceError(ValueError):
    """A metric key was used at a level it does not belong to."""


class MeasurementError(RuntimeError):
    """A state could not be measured, or a measurement does not bind."""


def metric_level(key: str) -> MetricLevel:
    if key.startswith(OP_PREFIX):
        return MetricLevel.OPERATOR_LOCAL
    if key.startswith(STATE_PREFIX):
        return MetricLevel.STATE_EVALUATION
    raise MetricNamespaceError(
        f"metric key {key!r} carries no level namespace; use {OP_PREFIX!r} for an "
        f"operator-local objective or {STATE_PREFIX!r} for a global state metric")


def require_state_metric(key: str) -> str:
    level = metric_level(key)
    if level is not MetricLevel.STATE_EVALUATION:
        raise MetricNamespaceError(
            f"{key!r} is a {level.value} metric. An operator's own objective "
            "cannot rank the beam: E8a's operator-local KL was 3.11x better for "
            "the map that initialized 2.8 nats worse.")
    return key


# --- level 1: operator-local ------------------------------------------------


@dataclass(frozen=True)
class OperatorLocalMetrics:
    """What an operator measured about its own decision.

    ``reference`` is almost always ``"parent_state"``: the operator compares the
    checkpoint it was handed to the candidates it could produce from it. It is
    recorded explicitly because a global-reference operator is legitimate too,
    and the two cannot be compared.
    """

    impl_id: str
    objective: str
    reference: str
    values: Mapping[str, float]
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in self.values:
            if metric_level(key) is not MetricLevel.OPERATOR_LOCAL:
                raise MetricNamespaceError(
                    f"{self.impl_id} reported {key!r} as an operator-local metric; "
                    f"operator-local keys must start with {OP_PREFIX!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "impl_id": self.impl_id,
            "objective": self.objective,
            "reference": self.reference,
            "values": dict(sorted(self.values.items())),
            "detail": _jsonable(self.detail),
        }


# --- level 2: global state evaluation ---------------------------------------


@dataclass(frozen=True)
class StateEvalSuite:
    """The frozen held-out suite every produced checkpoint is scored on.

    Domains are declared rather than inferred so the equal-domain aggregate is a
    design decision, not a property of whichever domain happened to tokenize
    longest — the same rule E8a used (``domain_balanced_score``).
    """

    suite_id: str
    version: int
    domains: tuple[str, ...]
    subtypes: Mapping[str, tuple[str, ...]]
    critical_tags: tuple[str, ...]
    items_path: str | None = None
    content_sha256: str | None = None
    n_items: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.domains:
            raise MeasurementError(f"{self.suite_id}: no domains declared")
        missing = [d for d in self.domains if d not in self.subtypes]
        if missing:
            raise MeasurementError(f"{self.suite_id}: domains without sub-types {missing}")

    @property
    def qualified_id(self) -> str:
        return f"{self.suite_id}@v{self.version}"

    @property
    def suite_hash(self) -> str:
        return hashlib.sha256(json.dumps({
            "suite_id": self.suite_id,
            "version": self.version,
            "domains": list(self.domains),
            "subtypes": {k: list(v) for k, v in sorted(self.subtypes.items())},
            "critical_tags": list(self.critical_tags),
            "content_sha256": self.content_sha256,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def required_metrics(self) -> tuple[str, ...]:
        """Keys a complete evaluation of this suite must carry.

        A state missing any of these cannot be ranked. "Report what we managed
        to measure" is how a checkpoint with a missing domain quietly wins on
        the domains it did produce.
        """
        return (
            "state.teacher_kl.equal_domain_mean",
            *(f"state.teacher_kl.{d}" for d in self.domains),
            "state.critical_token_kl",
            "state.nll.general",
            "state.top1_agreement",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id, "version": self.version,
            "domains": list(self.domains),
            "subtypes": {k: list(v) for k, v in sorted(self.subtypes.items())},
            "critical_tags": list(self.critical_tags),
            "items_path": self.items_path, "content_sha256": self.content_sha256,
            "n_items": self.n_items, "description": self.description,
            "suite_hash": self.suite_hash,
        }


@dataclass(frozen=True)
class StateEvaluation:
    """Global metrics for one checkpoint, bound to that checkpoint's hash."""

    checkpoint_sha256: str
    suite_id: str
    suite_hash: str
    reference: str
    values: Mapping[str, float]
    positions: int
    detail: Mapping[str, Any] = field(default_factory=dict)
    measured_utc: str | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_sha256:
            raise MeasurementError("a state evaluation must name the weights it measured")
        for key in self.values:
            require_state_metric(key)

    def require(self, keys: Sequence[str]) -> None:
        missing = [k for k in keys if k not in self.values]
        if missing:
            raise MeasurementError(
                f"evaluation of {self.checkpoint_sha256[:12]} is missing required "
                f"metrics {missing}; an incomplete evaluation may not be ranked")

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "suite_id": self.suite_id, "suite_hash": self.suite_hash,
            "reference": self.reference,
            "values": dict(sorted(self.values.items())),
            "positions": self.positions,
            "detail": _jsonable(self.detail),
            "measured_utc": self.measured_utc,
            "runtime": dict(self.runtime),
        }


# --- the measurement itself -------------------------------------------------


@dataclass
class SuiteItem:
    """One scored item: token ids, the domain/sub-type it counts under, and tags.

    ``tags`` map a critical-token name (``think_close``, ``eos``, ``final_answer``,
    ``tool_close``, ...) to a boolean mask over prediction positions. They are
    reported separately and, at level 2, aggregated into
    ``state.critical_token_kl`` — the fidelity metric that a plain token-mean KL
    washes out, because the tokens that decide whether a rollout terminates are a
    vanishing fraction of positions.
    """

    item_id: str
    input_ids: torch.Tensor
    domain: str
    subtype: str
    tags: Mapping[str, torch.Tensor] = field(default_factory=dict)


class StateEvaluator:
    """Scores a candidate checkpoint against the **original teacher**.

    The root teacher's logits are computed once per item and reused for every
    state in the search — they cannot change, and re-running a 4B forward for
    every one of ~140 candidates would dominate the search cost. What is *never*
    reused is the candidate's own forward: every checkpoint is measured on its
    own weights, and the result is stamped with those weights' hash.
    """

    def __init__(
        self,
        suite: StateEvalSuite,
        items: Sequence[SuiteItem],
        *,
        device: str = "cpu",
        chunk: int = 512,
    ) -> None:
        if not items:
            raise MeasurementError(f"{suite.qualified_id}: no items to score")
        declared = {(d, s) for d, subs in suite.subtypes.items() for s in subs}
        seen = {(i.domain, i.subtype) for i in items}
        unknown = sorted(seen - declared)
        if unknown:
            raise MeasurementError(
                f"{suite.qualified_id}: items carry undeclared (domain, subtype) {unknown}")
        missing = sorted(declared - seen)
        if missing:
            raise MeasurementError(
                f"{suite.qualified_id}: declared sub-types with no items {missing}; "
                "a silently absent sub-type reweights its domain")
        self.suite = suite
        self.items = list(items)
        self.device = device
        self.chunk = chunk
        self._ref_logits: dict[str, torch.Tensor] = {}
        self._ref_ready = False

    @torch.no_grad()
    def prime_reference(self, teacher) -> None:
        """Cache the original teacher's logits for every suite item."""
        for item in self.items:
            ids = item.input_ids.to(self.device)
            logits = teacher(ids).logits[0, :-1].float().cpu()
            self._ref_logits[item.item_id] = logits
        self._ref_ready = True

    @torch.no_grad()
    def evaluate(self, model, checkpoint_sha256: str, *, reference: str = "root_teacher",
                 runtime: Mapping[str, Any] | None = None) -> StateEvaluation:
        if not self._ref_ready:
            raise MeasurementError(
                "reference logits were never primed; a candidate cannot be scored "
                "against a teacher that was not run")
        if not checkpoint_sha256:
            raise MeasurementError("refusing to measure without the checkpoint hash")

        per_subtype: dict[str, DistortionSums] = {}
        totals = DistortionSums()
        for item in self.items:
            ids = item.input_ids.to(self.device)
            cand = model(ids).logits[0, :-1].float().cpu()
            ref = self._ref_logits[item.item_id]
            if cand.shape != ref.shape:
                raise MeasurementError(
                    f"item {item.item_id}: candidate logits {tuple(cand.shape)} do not "
                    f"match the reference {tuple(ref.shape)}; the two models are not "
                    "logit-comparable, so no KL against the original teacher exists")
            targets = item.input_ids[0, 1:].cpu()
            sums = distortion(ref, cand, targets, tags=item.tags, chunk=self.chunk)
            per_subtype.setdefault(item.subtype, DistortionSums()).merge(sums)
            totals.merge(sums)

        subtype_kl = {k: v.as_dict()["kl"] for k, v in per_subtype.items()}
        primary, per_domain = domain_balanced_score(
            subtype_kl, {d: list(self.suite.subtypes[d]) for d in self.suite.domains})

        agg = totals.as_dict()
        values: dict[str, float] = {
            "state.teacher_kl.equal_domain_mean": float(primary),
            "state.teacher_kl.token_mean": float(agg["kl"]),
            "state.reverse_kl.token_mean": float(agg["reverse_kl"]),
            "state.nll.general": float(agg["abl_ce"]),
            "state.nll.teacher_reference": float(agg["ref_ce"]),
            "state.nll_delta_vs_teacher": float(agg["ce_delta"]),
            "state.top1_agreement": float(agg["top1_agreement"]),
        }
        for domain, score in per_domain.items():
            values[f"state.teacher_kl.{domain}"] = float(score)

        tagged = agg["tagged"]
        for tag, entry in tagged.items():
            if entry["kl"] is not None:
                values[f"state.critical_token_kl.{tag}"] = float(entry["kl"])
        present = [tagged[t]["kl"] for t in self.suite.critical_tags
                   if t in tagged and tagged[t]["kl"] is not None]
        if present:
            # Unweighted mean over declared critical-token classes: a token-count
            # mean would be dominated by whichever class is common, and the rare
            # ones (think_close, eos) are the ones that decide termination.
            values["state.critical_token_kl"] = float(sum(present) / len(present))

        return StateEvaluation(
            checkpoint_sha256=checkpoint_sha256,
            suite_id=self.suite.qualified_id,
            suite_hash=self.suite.suite_hash,
            reference=reference,
            values=values,
            positions=int(agg["positions"]),
            detail={"per_subtype_kl": subtype_kl, "per_domain_kl": per_domain,
                    "tagged": tagged},
            runtime=dict(runtime or {}),
        )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)
