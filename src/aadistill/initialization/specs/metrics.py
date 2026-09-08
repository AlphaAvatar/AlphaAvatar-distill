"""Measurement contracts: what a metric is, and what a suite and a result are.

These types are shared by the operators, by the state spec and by the evaluator
that produces them, so they sit at the bottom of the initialization layering
rather than beside the evaluator. They were in `planning.metrics`, which made
`operators -> planning` and `specs -> planning` edges and closed two dependency
cycles: seven operator modules imported this package for `OperatorLocalMetrics`
alone, and `specs.state` for three exception and result types.

Nothing here imports another initialization layer. The evaluator that consumes
these contracts, and the distortion statistics it needs, stay in
`planning.metrics` -- the direction the layering intends.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


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
    #: Which declared domain, if any, is general language. `state.nll.general` is
    #: computed from this domain alone; when it is absent the metric is not
    #: emitted at all, rather than being silently backed by a pooled average over
    #: reasoning, code and tool text that is not "general" by any reading.
    general_domain: str | None = "general"

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
            "state.teacher_kl.worst_domain",
            *(f"state.teacher_kl.{d}" for d in self.domains),
            "state.critical_token_kl",
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
            "general_domain": self.general_domain,
            "suite_hash": self.suite_hash,
        }


@dataclass(frozen=True)
class StateEvaluation:
    """Global metrics for one checkpoint, bound to its artifact digest.

    The digest, not a single file's sha256: a sharded checkpoint has no single
    file, and binding to one would either fail or — worse — bind to whichever
    shard happened to be named first.
    """

    artifact_digest: str
    suite_id: str
    suite_hash: str
    reference: str
    values: Mapping[str, float]
    positions: int
    detail: Mapping[str, Any] = field(default_factory=dict)
    measured_utc: str | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_digest:
            raise MeasurementError("a state evaluation must name the artifact it measured")
        for key in self.values:
            require_state_metric(key)

    def require(self, keys: Sequence[str]) -> None:
        missing = [k for k in keys if k not in self.values]
        if missing:
            raise MeasurementError(
                f"evaluation of {self.artifact_digest[:12]} is missing required "
                f"metrics {missing}; an incomplete evaluation may not be ranked")

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_digest": self.artifact_digest,
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


class ReferenceStrategy(Enum):
    """How the original teacher's reference logits are obtained per candidate.

    ``RECOMPUTE`` is the default, and the arithmetic is not close. Caching the
    reference for the intended 59,763-position suite at a 151,936 vocabulary is
    **33.8 GiB** in float32 (16.9 GiB in float16, still with a numerical
    tolerance to justify). Recomputing it costs one teacher forward over the
    suite per candidate: 5.6 s on an L40S, or **3.9 minutes across a whole
    42-candidate search**. Thirty-four gigabytes of RAM to save four minutes is
    not a trade; it is a way to make the pilot fail on a memory limit that the
    tiny dry run would never expose.

    ``CACHE_IN_MEMORY`` remains available for small suites and is what the dry
    run uses, but it refuses to allocate past an explicit byte budget instead of
    discovering the limit at runtime.
    """

    RECOMPUTE = "recompute"
    CACHE_IN_MEMORY = "cache_in_memory"


#: Default ceiling for `CACHE_IN_MEMORY`. Deliberately small: anything that
#: wants more should be recomputing.
DEFAULT_REFERENCE_CACHE_BUDGET_BYTES = 2 * 2**30


def reference_cache_bytes(items: Sequence["SuiteItem"], vocab_size: int,
                          bytes_per_value: int = 4) -> int:
    """What caching the reference logits for these items would cost."""
    positions = sum(int(i.input_ids.shape[1]) - 1 for i in items)
    return positions * vocab_size * bytes_per_value

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
