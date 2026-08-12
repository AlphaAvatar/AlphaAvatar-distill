"""``InitializationState`` — a node in the initialization search.

The load-bearing distinction in this file is **search state != recovery
candidate**, and it is mechanical rather than documented:

* a state carries its own ``ArchSpec``, which may be an intermediate geometry
  with an intermediate parameter count. That is legal and expected;
* ``is_complete_leaf`` is true only when the spec matches the requested target
  *exactly, field for field*;
* ``require_recovery_admissible`` raises otherwise, and the recovery orchestrator
  calls it before admitting anything to Top-N.

The second load-bearing rule is that **metrics never travel between
checkpoints**. A state is created with no evaluation at all; ``attach_evaluation``
refuses any evaluation whose ``artifact_digest`` is not this state's own, and
``ready_for_ranking`` requires a present, matching, complete evaluation. A child
cannot inherit its parent's NLL because there is no code path that would let it —
the same guarantee ``init/nll_gate.py`` established for initialization NLL after
E8, generalized to every node of the search. The binding is to the *artifact*
(every shard, the index, the config, the architecture signature, the tokenizer)
rather than to one filename, so it survives a checkpoint being sharded.

A third distinction: ``provenance``. Most states are ``search`` — an operator
produced them. A ``retained_canonical`` state is a frozen checkpoint injected as a
control (see ``make_control_state``); it has no operator steps because this search
did not build it, and re-running its recipe would not reproduce it.

State ids are content-derived: the sha256 of (root teacher, ordered
(implementation, signature, calibration profile, operator config)) along the path.
Two runs of the same path produce the same id, which is what makes resume exact
and deduplication free.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .arch import ArchSpec
from .artifact import CheckpointIdentity
from .metrics import MeasurementError, OperatorLocalMetrics, StateEvaluation

SCHEMA = "aadistill.autoinit.state/v1"


class StateValidity(Enum):
    """Where a node is in the mandatory materialize -> measure lifecycle."""

    PLANNED = "planned"
    MATERIALIZED = "materialized"
    VALIDATED = "validated"
    MEASURED = "measured"
    INVALID = "invalid"
    PRUNED = "pruned"


class StateError(RuntimeError):
    """A state was used at a lifecycle point it has not reached."""


@dataclass(frozen=True)
class OperatorStep:
    """One applied operator, in path order.

    Records the implementation's *signature hash* alongside its id so a manifest
    stays interpretable even if the registry later gains a ``_v2``: the id says
    which algorithm, the signature says which declared semantics.
    """

    index: int
    kind: str
    impl_id: str
    impl_signature_hash: str
    profile_id: str
    profile_hash: str
    config_hash: str
    seed: int
    result_spec_hash: str
    local_metrics: OperatorLocalMetrics | None = None
    trace: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    wall_seconds: float | None = None

    def identity(self) -> dict[str, str]:
        """The part of a step that determines the resulting state's id."""
        return {"impl_id": self.impl_id, "impl_signature_hash": self.impl_signature_hash,
                "profile_hash": self.profile_hash, "config_hash": self.config_hash,
                "seed": str(self.seed)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "kind": self.kind, "impl_id": self.impl_id,
            "impl_signature_hash": self.impl_signature_hash,
            "profile_id": self.profile_id, "profile_hash": self.profile_hash,
            "config_hash": self.config_hash, "seed": self.seed,
            "result_spec_hash": self.result_spec_hash,
            "local_metrics": self.local_metrics.as_dict() if self.local_metrics else None,
            "trace": _jsonable(self.trace), "artifacts": _jsonable(self.artifacts),
            "wall_seconds": self.wall_seconds,
        }


def compute_state_id(root_teacher_hash: str, target_spec_hash: str,
                     steps: Sequence[OperatorStep]) -> str:
    """Content-derived id: the same path under the same target is the same state.

    The target participates because a path is only meaningful relative to what it
    is compressing toward — the same three operators aimed at a 596M target and a
    4.xB target are different states, and sharing an id would let one run's
    journal resume the other's.
    """
    payload = {"root": root_teacher_hash, "target": target_spec_hash,
               "steps": [s.identity() for s in steps]}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]


@dataclass
class InitializationState:
    """One node: a checkpoint, the path that produced it, and what is known of it."""

    state_id: str
    parent_id: str | None
    root_teacher_id: str
    root_teacher_sha256: str
    spec: ArchSpec
    target_spec: ArchSpec
    steps: tuple[OperatorStep, ...]
    num_parameters: int
    depth: int
    seed: int
    checkpoint_path: str | None = None
    artifact: CheckpointIdentity | None = None
    validity: StateValidity = StateValidity.PLANNED
    #: Where this state came from. `search` for anything an operator produced;
    #: `retained_canonical` for a frozen checkpoint injected as a control.
    provenance: str = "search"
    evaluation: StateEvaluation | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    prune_reason: str | None = None
    invalid_reason: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    # --- identity ----------------------------------------------------------

    @property
    def applied_kinds(self) -> tuple[str, ...]:
        return tuple(s.kind for s in self.steps)

    @property
    def impl_ids(self) -> tuple[str, ...]:
        return tuple(s.impl_id for s in self.steps)

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(s.profile_id for s in self.steps)

    @property
    def path_label(self) -> str:
        """Human-readable path, e.g. ``DEPTH(calib.domain_balanced@v1)->FFN(...)``."""
        return "->".join(f"{s.kind}({s.profile_id})" for s in self.steps) or "ROOT"

    def remaining_differences(self) -> frozenset[str]:
        return self.spec.diff(self.target_spec)

    def is_complete_leaf(self) -> bool:
        """Exactly at the requested target architecture. Nothing weaker counts."""
        return self.spec.matches(self.target_spec)

    # --- lifecycle ---------------------------------------------------------

    @property
    def artifact_digest(self) -> str | None:
        """The identity metrics bind to. None until the state is materialized."""
        return self.artifact.artifact_digest if self.artifact else None

    @property
    def checkpoint_sha256(self) -> str | None:
        """The lone shard's own sha256, when the checkpoint is unsharded.

        Kept so a frozen single-file hash from the historical record stays
        directly checkable. It is deliberately **not** what metrics bind to: a
        sharded checkpoint has no such value, and a binding that silently became
        None once a 30B teacher sharded would be worse than no binding at all.
        """
        return self.artifact.single_shard_sha256 if self.artifact else None

    def mark_materialized(self, artifact: CheckpointIdentity) -> None:
        self.checkpoint_path = artifact.path
        self.artifact = artifact
        self.validity = StateValidity.MATERIALIZED

    def mark_validated(self) -> None:
        if self.validity is not StateValidity.MATERIALIZED:
            raise StateError(
                f"{self.state_id}: validation requires a materialized checkpoint, "
                f"not {self.validity.value}")
        self.validity = StateValidity.VALIDATED

    def mark_invalid(self, reason: str) -> None:
        self.validity = StateValidity.INVALID
        self.invalid_reason = reason

    def mark_pruned(self, reason: str) -> None:
        """Pruned states stay in the manifest with their reason. Nothing vanishes."""
        self.prune_reason = reason
        if self.validity not in (StateValidity.INVALID,):
            self.validity = StateValidity.PRUNED

    def attach_evaluation(self, evaluation: StateEvaluation) -> None:
        """Bind a global measurement, or refuse it.

        The refusal is the point. An evaluation produced from the parent's
        weights carries the parent's hash and cannot be attached here, so
        "inherit the parent's NLL to save a forward pass" is not an available
        shortcut at 2am on a paid pod.
        """
        if self.artifact is None:
            raise StateError(
                f"{self.state_id}: cannot attach an evaluation before the checkpoint "
                "is materialized and identified")
        if evaluation.artifact_digest != self.artifact_digest:
            raise MeasurementError(
                f"{self.state_id}: evaluation was measured on artifact "
                f"{evaluation.artifact_digest[:12]} but this state's artifact digests "
                f"to {self.artifact_digest[:12]}. Metrics bind to artifacts; they are "
                "not inherited, copied or interpolated.")
        if self.validity not in (StateValidity.VALIDATED, StateValidity.MEASURED):
            raise StateError(
                f"{self.state_id}: measure only after structural validation, not from "
                f"{self.validity.value}")
        self.evaluation = evaluation
        self.validity = StateValidity.MEASURED

    # --- gates -------------------------------------------------------------

    def ready_for_ranking(self, required_metrics: Sequence[str]) -> None:
        if self.validity is not StateValidity.MEASURED or self.evaluation is None:
            raise StateError(
                f"{self.state_id} is {self.validity.value} and has no hash-bound "
                "evaluation; an unmeasured state cannot enter beam ranking")
        if self.evaluation.artifact_digest != self.artifact_digest:
            raise MeasurementError(
                f"{self.state_id}: attached evaluation no longer matches the artifact")
        self.evaluation.require(required_metrics)

    def require_recovery_admissible(self) -> None:
        """The gate between the search and any recovery probe."""
        if not self.is_complete_leaf():
            missing = sorted(self.remaining_differences())
            raise StateError(
                f"{self.state_id} is an intermediate search state: it still differs "
                f"from the target in {missing} ({self.spec.describe()} vs "
                f"{self.target_spec.describe()}). Intermediate checkpoints are search "
                "states only and never enter recovery Top-N.")
        if self.validity is not StateValidity.MEASURED:
            raise StateError(
                f"{self.state_id} is {self.validity.value}; a recovery candidate must "
                "carry its own hash-bound measurements")

    # --- serialization -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "state_id": self.state_id,
            "parent_id": self.parent_id,
            "root_teacher_id": self.root_teacher_id,
            "root_teacher_sha256": self.root_teacher_sha256,
            "arch_spec": self.spec.as_dict(),
            "arch_spec_hash": self.spec.spec_hash,
            "target_spec": self.target_spec.as_dict(),
            "target_spec_hash": self.target_spec.spec_hash,
            "family": self.spec.family,
            "steps": [s.as_dict() for s in self.steps],
            "applied_kinds": list(self.applied_kinds),
            "impl_ids": list(self.impl_ids),
            "calibration_profiles": list(self.profile_ids),
            "path_label": self.path_label,
            "remaining_differences": sorted(self.remaining_differences()),
            "is_complete_leaf": self.is_complete_leaf(),
            "num_parameters": self.num_parameters,
            "depth": self.depth,
            "seed": self.seed,
            "checkpoint_path": self.checkpoint_path,
            "artifact": self.artifact.as_dict() if self.artifact else None,
            "artifact_digest": self.artifact_digest,
            "checkpoint_sha256": self.checkpoint_sha256,
            "provenance": self.provenance,
            "validity": self.validity.value,
            "evaluation": self.evaluation.as_dict() if self.evaluation else None,
            "artifacts": dict(self.artifacts),
            "prune_reason": self.prune_reason,
            "invalid_reason": self.invalid_reason,
            "notes": _jsonable(self.notes),
        }


def make_root_state(*, root_teacher_id: str, root_teacher_sha256: str, spec: ArchSpec,
                    target_spec: ArchSpec, num_parameters: int, seed: int,
                    checkpoint_path: str | None = None,
                    artifact: CheckpointIdentity | None = None) -> InitializationState:
    state = InitializationState(
        state_id=compute_state_id(root_teacher_sha256, target_spec.spec_hash, ()),
        parent_id=None,
        root_teacher_id=root_teacher_id,
        root_teacher_sha256=root_teacher_sha256,
        spec=spec, target_spec=target_spec, steps=(),
        num_parameters=num_parameters, depth=0, seed=seed,
        checkpoint_path=checkpoint_path, artifact=artifact,
        provenance="root_teacher",
    )
    # The root is the teacher: not produced by the search, identified by its own
    # published revision hash. It is never ranked and never a recovery candidate,
    # so it needs no artifact identity of its own unless a caller supplies one.
    state.validity = StateValidity.VALIDATED
    return state


def make_control_state(*, control_id: str, artifact: CheckpointIdentity,
                       spec: ArchSpec, target_spec: ArchSpec, num_parameters: int,
                       root_teacher_id: str, root_teacher_sha256: str,
                       description: str, expected_single_file_sha256: str | None = None,
                       ) -> InitializationState:
    """Inject a retained, frozen checkpoint as a recovery candidate.

    This exists because **a re-executed composite is not the historical
    incumbent.** ``composite.stage1_sandwich_v0`` running inside the search on
    ``calib.domain_balanced@v1`` produces a checkpoint built from *that* mixture's
    activation statistics; the canonical ``qwen3_0p6b_init_v0`` was built from the
    original Stage-0 statistics over the 949,859-token warm-up mixture. Same
    algorithm, different input, therefore different weights — and every existing
    behaviour number in this project belongs to the latter.

    So the recovery control is injected by artifact rather than regenerated. It
    carries no operator steps (it was not produced by this search), it is marked
    ``retained_canonical``, and it is measured on the same suite as everything
    else so its step-0 metrics are comparable.
    """
    if expected_single_file_sha256 is not None:
        actual = artifact.single_shard_sha256
        if actual != expected_single_file_sha256:
            raise StateError(
                f"{control_id}: expected the frozen checkpoint {expected_single_file_sha256[:12]} "
                f"but the artifact at {artifact.path} has "
                f"{'shards' if artifact.is_sharded else actual[:12] if actual else 'no shard'}. "
                "A control that is not the retained checkpoint is not a control.")
    if not spec.matches(target_spec):
        raise StateError(
            f"{control_id}: a recovery control must already be at the target "
            f"architecture; it is {spec.describe()}")
    state = InitializationState(
        state_id=f"control-{control_id}",
        parent_id=None,
        root_teacher_id=root_teacher_id,
        root_teacher_sha256=root_teacher_sha256,
        spec=spec, target_spec=target_spec, steps=(),
        num_parameters=num_parameters, depth=0, seed=0,
        checkpoint_path=artifact.path, artifact=artifact,
        provenance="retained_canonical",
        notes={"description": description,
               "frozen_single_file_sha256": expected_single_file_sha256},
    )
    state.validity = StateValidity.VALIDATED
    return state


def child_state(parent: InitializationState, step: OperatorStep, spec: ArchSpec,
                num_parameters: int, seed: int) -> InitializationState:
    """A planned child. Deliberately carries **no** metrics from ``parent``."""
    steps = (*parent.steps, step)
    return InitializationState(
        state_id=compute_state_id(parent.root_teacher_sha256,
                                  parent.target_spec.spec_hash, steps),
        parent_id=parent.state_id,
        root_teacher_id=parent.root_teacher_id,
        root_teacher_sha256=parent.root_teacher_sha256,
        spec=spec, target_spec=parent.target_spec, steps=steps,
        num_parameters=num_parameters, depth=parent.depth + 1, seed=seed,
    )


class StateStore:
    """Append-only JSONL journal of every state the search touched.

    Append-only because a pruned state is evidence: the manifest has to be able
    to say which alternatives existed and why each was dropped. Rewriting the
    journal in place would make a search that pruned the eventual winner
    indistinguishable from one that never generated it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, state: InitializationState) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(state.as_dict(), sort_keys=True) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()
                if line.strip()]

    def latest_by_state_id(self) -> dict[str, dict[str, Any]]:
        """Last written record per state id — the journal's current view."""
        out: dict[str, dict[str, Any]] = {}
        for record in self.records():
            out[record["state_id"]] = record
        return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
