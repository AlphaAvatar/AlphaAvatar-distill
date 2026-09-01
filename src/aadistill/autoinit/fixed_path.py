"""Replay ONE frozen operator sequence. Deliberately not a search.

Phase C1 varies exactly one operator on the frozen `fe9683e6a9c7` path, so it
needs to reproduce that path's prefix and then branch. `BeamSearch` cannot do
that job safely: `_candidate_expansions` enumerates and sorts implementations,
`SearchConfig` *requires* a schedule, a ranking policy and an eval suite, and
`allowed_impls` restricts the library without forcing an order — a depth-4 beam
would still explore permutations. Even at width 1 it makes ranking decisions.
An isolation experiment whose executor can reorder operators is not an isolation
experiment.

So this module does one thing: given an ordered `(impl_id, profile_id)` list, a
target spec and a root, apply exactly those steps in exactly that order.

    no candidate enumeration · no ranking · no pruning · no profile branching
    no tie-breaking · no StateStore · no beam

What it *does* reuse is everything that already carries a contract:
`OperatorImplementation.execute` (which verifies the operator touched only the
structural fields it declared, and did not mutate its parent), `OperatorContext`,
`ChildBuilder` inside the operators, `StatsCache` keyed on the parent's artifact
digest, `calibration.get_profile().resolve()` with its fail-closed content check,
and `identify_checkpoint`.

**Digest gating is the point, not a nicety.** Phase C0 binds the pre-ATTENTION
parent `b8820f41d062…` to artifact digest `eea90c91346a…`, and applying the
current ATTENTION to it must reproduce the retained incumbent `c313d1b4081b…`.
`expected_digests` turns those into a fail-stop: a step whose realized digest
differs raises `FixedPathDigestMismatch` carrying the full evidence, and nothing
downstream runs. A mismatch is a finding to be reviewed, never a thing to waive
in-flight.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infrastructure.manifest import sha256_json
from .arch import ArchitectureAdapter, ArchSpec
from .artifact import CheckpointIdentity, identify_checkpoint
from .calibration import (
    NO_CALIBRATION,
    consumes_calibration,
    get_profile,
    profile_for,
)
from .operators.base import OperatorContext, get_implementation
from .stats import DEFAULT_STATS_SPEC, StatsCache, StatsSpec, stats_cache_key

SCHEMA = "aadistill.autoinit.fixed_path/v1"


class FixedPathError(RuntimeError):
    """The path cannot be executed as specified."""


class FixedPathDigestMismatch(FixedPathError):
    """A realized artifact digest is not the one the path was pinned to.

    Carries both digests and the step that produced them, because the whole
    value of the gate is the evidence it hands to a reviewer.
    """

    def __init__(self, step_index: int, label: str, expected: str, actual: str,
                 evidence: Mapping[str, Any]):
        self.step_index = step_index
        self.label = label
        self.expected = expected
        self.actual = actual
        self.evidence = dict(evidence)
        super().__init__(
            f"step {step_index} ({label}): artifact digest {actual} does not match "
            f"the pinned {expected}. STOP: this is a replay mismatch, not a "
            "recoverable condition. Record the evidence and refer it to review; "
            "do not continue to any recovery or behavioural measurement.")


@dataclass(frozen=True)
class FixedPathStep:
    """One operator application. Both ids are resolved, never inferred."""

    impl_id: str
    profile_id: str
    #: Optional artifact digest this step's output is pinned to.
    expected_artifact_digest: str | None = None
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"impl_id": self.impl_id, "profile_id": self.profile_id,
                "expected_artifact_digest": self.expected_artifact_digest,
                "label": self.label}


@dataclass(frozen=True)
class FixedPathSpec:
    """An ordered operator sequence, its target and its root. Hashable.

    `steps` is a tuple and the executor walks it by index. There is no field by
    which an ordering could be reconsidered, which is the property C1 needs.
    """

    path_id: str
    family: str
    target_spec: ArchSpec
    steps: tuple[FixedPathStep, ...]
    root_repo_id: str
    root_revision: str
    stats_spec: StatsSpec = DEFAULT_STATS_SPEC
    device: str = "cpu"
    seed: int = 0
    max_shard_size: str | int | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise FixedPathError(f"{self.path_id}: a fixed path needs at least one step")
        kinds: list[str] = []
        for i, s in enumerate(self.steps):
            impl = get_implementation(s.impl_id)  # raises on an unregistered id
            if impl.kind in kinds:
                raise FixedPathError(
                    f"{self.path_id}: step {i} repeats kind {impl.kind!r}, already "
                    f"applied at step {kinds.index(impl.kind)}. A fixed path applies "
                    "each structural kind at most once.")
            kinds.append(impl.kind)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(get_implementation(s.impl_id).kind for s in self.steps)

    @property
    def path_label(self) -> str:
        return "->".join(f"{k}({s.profile_id})"
                         for k, s in zip(self.kinds, self.steps))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "path_id": self.path_id,
            "family": self.family,
            "target_spec": self.target_spec.as_dict(),
            "target_spec_hash": self.target_spec.spec_hash,
            "steps": [s.as_dict() for s in self.steps],
            "kinds": list(self.kinds),
            "path_label": self.path_label,
            "root_repo_id": self.root_repo_id,
            "root_revision": self.root_revision,
            "stats_spec": self.stats_spec.as_dict(),
            "device": self.device,
            "seed": self.seed,
            "max_shard_size": self.max_shard_size,
        }

    @property
    def spec_hash(self) -> str:
        return sha256_json(self.as_dict())

    def replace_tail(self, index: int, step: FixedPathStep,
                     *, path_id: str) -> "FixedPathSpec":
        """The same path with one step substituted. The C1 arm constructor.

        Both C1 arms are this call on the same prefix, which is what makes the
        two arms differ by exactly one operator *by construction* rather than by
        inspection.
        """
        if not 0 <= index < len(self.steps):
            raise FixedPathError(f"step index {index} out of range")
        steps = list(self.steps)
        steps[index] = step
        return FixedPathSpec(
            path_id=path_id, family=self.family, target_spec=self.target_spec,
            steps=tuple(steps), root_repo_id=self.root_repo_id,
            root_revision=self.root_revision, stats_spec=self.stats_spec,
            device=self.device, seed=self.seed,
            max_shard_size=self.max_shard_size)


@dataclass
class StepResult:
    index: int
    impl_id: str
    profile_id: str
    kind: str
    result_spec_hash: str
    identity: CheckpointIdentity
    checkpoint_path: str
    seconds: float
    local_metrics: Mapping[str, Any] = field(default_factory=dict)
    trace: Mapping[str, Any] = field(default_factory=dict)
    selection: Mapping[str, Any] = field(default_factory=dict)
    digest_expected: str | None = None
    digest_matches: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "impl_id": self.impl_id,
            "profile_id": self.profile_id, "kind": self.kind,
            "result_spec_hash": self.result_spec_hash,
            "artifact_digest": self.identity.artifact_digest,
            "weights_digest": self.identity.weights_digest,
            "config_sha256": self.identity.config_sha256,
            "arch_signature": self.identity.arch_signature,
            "tokenizer_sha256": self.identity.tokenizer_sha256,
            "single_shard_sha256": self.identity.single_shard_sha256,
            "num_parameters": self.identity.num_parameters,
            "checkpoint_path": self.checkpoint_path,
            "seconds": round(self.seconds, 4),
            "local_metrics": dict(self.local_metrics),
            "trace": dict(self.trace),
            "selection": dict(self.selection),
            "digest_expected": self.digest_expected,
            "digest_matches": self.digest_matches,
        }


#: What each operator kind must record about the choice it made, so a replay
#: mismatch can be diagnosed rather than merely observed. Keys are the operator
#: `produces` names; DEPTH additionally carries its removal order in `trace`.
SELECTION_ARTIFACTS = ("kept_blocks", "removed_blocks", "kept_neurons",
                       "kept_heads", "projection_diagnostics")


def _selection_evidence(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    return {k: artifacts[k] for k in SELECTION_ARTIFACTS if k in artifacts}


def materialize_fixed_path(
    spec: FixedPathSpec,
    *,
    adapter: ArchitectureAdapter,
    root_loader: Callable[[], Any],
    workdir: str | Path,
    repo_root: str | Path = ".",
    stats_cache: StatsCache | None = None,
    calibration_items: Mapping[str, Sequence[Any]] | None = None,
    on_step: Callable[[StepResult], None] | None = None,
) -> list[StepResult]:
    """Apply every step in order, identifying and gating each intermediate.

    `calibration_items` may be supplied to avoid re-resolving (the resolver
    verifies a content hash on every call); when omitted the profiles are
    resolved from `repo_root`, fail-closed.
    """
    import time

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    cache = stats_cache if stats_cache is not None else StatsCache(
        stats_spec=spec.stats_spec)
    resolved: dict[str, Sequence[Any]] = dict(calibration_items or {})

    model = root_loader()
    parent_spec = adapter.spec_of(model)
    parent_digest: str | None = None
    results: list[StepResult] = []

    for i, step in enumerate(spec.steps):
        impl = get_implementation(step.impl_id)
        # One place decides what an operator is actually invoked with, exactly as
        # the search does: a CalibrationNeed.NONE implementation gets the
        # canonical sentinel, which is not a registry entry and must not be
        # looked up as one. A step that names anything else for such an operator
        # is a specification error, not something to quietly substitute.
        if not consumes_calibration(impl):
            if step.profile_id != NO_CALIBRATION.qualified_id:
                raise FixedPathError(
                    f"{spec.path_id} step {i} ({impl.impl_id}) consumes no "
                    f"calibration, so its profile must be "
                    f"{NO_CALIBRATION.qualified_id!r}, not {step.profile_id!r}")
            profile = NO_CALIBRATION
        else:
            profile = profile_for(impl, get_profile(step.profile_id))
        if profile.is_null:
            items: Sequence[Any] = ()
        else:
            if profile.qualified_id not in resolved:
                resolved[profile.qualified_id] = profile.resolve(repo_root)
            items = resolved[profile.qualified_id]

        ok, reason = impl.applicable(parent_spec, spec.target_spec, adapter)
        if not ok:
            raise FixedPathError(
                f"{spec.path_id} step {i} ({impl.impl_id}): not applicable to "
                f"{parent_spec.describe()} — {reason}")

        operator_config = {"n_calibration_items": len(items)}
        plan = impl.plan(parent_spec, spec.target_spec, adapter, operator_config)

        ctx = OperatorContext(
            adapter=adapter, model=model, parent_spec=parent_spec,
            target_spec=spec.target_spec, profile=profile,
            calibration_items=items, seed=spec.seed, device=spec.device,
            workdir=work, config=dict(operator_config),
            stats_cache=cache,
            stats_cache_key=(
                None if parent_digest is None else stats_cache_key(
                    parent_artifact_digest=parent_digest,
                    profile_hash=profile.profile_hash,
                    stats_spec=spec.stats_spec,
                    adapter_version=adapter.adapter_version,
                    numerical_config={
                        "device": spec.device,
                        "accumulation": spec.stats_spec.accumulation_dtype})),
        )

        started = time.time()
        outcome = impl.execute(ctx)          # contract-verified application
        seconds = time.time() - started

        realized = adapter.spec_of(outcome.model)
        if not realized.matches(plan.result_spec):
            raise FixedPathError(
                f"{spec.path_id} step {i} ({impl.impl_id}): realized spec "
                f"{realized.describe()} is not the planned {plan.result_spec.describe()}")

        ckpt = work / "steps" / f"{i:02d}_{impl.kind.lower()}"
        ckpt.mkdir(parents=True, exist_ok=True)
        adapter.save(outcome.model, str(ckpt), max_shard_size=spec.max_shard_size)
        identity = identify_checkpoint(
            ckpt, adapter=adapter, spec=plan.result_spec,
            num_parameters=adapter.param_count(plan.result_spec))

        lm = outcome.local_metrics
        result = StepResult(
            index=i, impl_id=impl.impl_id, profile_id=profile.qualified_id,
            kind=impl.kind, result_spec_hash=plan.result_spec.spec_hash,
            identity=identity, checkpoint_path=str(ckpt), seconds=seconds,
            local_metrics=dict(getattr(lm, "values", {}) or {}),
            trace=dict(outcome.trace),
            selection=_selection_evidence(outcome.artifacts),
            digest_expected=step.expected_artifact_digest,
            digest_matches=(None if step.expected_artifact_digest is None
                            else identity.artifact_digest == step.expected_artifact_digest),
        )
        results.append(result)
        if on_step is not None:
            on_step(result)

        if result.digest_matches is False:
            raise FixedPathDigestMismatch(
                i, step.label or impl.impl_id, step.expected_artifact_digest,
                identity.artifact_digest,
                {"steps": [r.as_dict() for r in results],
                 "path": spec.as_dict()})

        # The next step expands the checkpoint on disk, not the in-memory child:
        # the file is what every downstream stage loads, so the file is what the
        # rest of the path is built from.
        del outcome, model
        model = adapter.load(str(ckpt), device=spec.device)
        parent_spec = plan.result_spec
        parent_digest = identity.artifact_digest

    return results


def write_replay_record(spec: FixedPathSpec, results: Sequence[StepResult],
                        path: str | Path, *, runtime: Mapping[str, Any],
                        root_binding: Mapping[str, Any] | None = None) -> Path:
    """The auditable record of one replay: identities, choices and runtime.

    Written whether or not the digests matched — a mismatch is exactly when this
    file matters most.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "aadistill.autoinit.fixed_path_replay/v1",
        "path": spec.as_dict(),
        "path_hash": spec.spec_hash,
        "runtime": dict(runtime),
        "root_binding": dict(root_binding or {}),
        "steps": [r.as_dict() for r in results],
        "all_pinned_digests_matched": all(
            r.digest_matches for r in results if r.digest_expected is not None),
        "n_pinned": sum(1 for r in results if r.digest_expected is not None),
    }
    p.write_text(json.dumps(record, indent=1) + "\n")
    return p
