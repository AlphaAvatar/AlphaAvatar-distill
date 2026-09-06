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
from .calibration_items import prepare_calibration_items
from .device import model_device
from .operators.base import OperatorContext, get_implementation
from .stats import DEFAULT_STATS_SPEC, StatsCache, StatsSpec, stats_cache_key

SCHEMA = "aadistill.autoinit.fixed_path/v1"


class FixedPathError(RuntimeError):
    """The path cannot be executed as specified."""


class FixedPathRootDeviceMismatch(FixedPathError):
    """The root model's weights are not where the path says the path executes.

    `FixedPathSpec.device` is carried into `OperatorContext.device` and into the
    stats cache key, and `adapter.load` binds every *later* parent to it. The
    root is the one model the executor does not place itself — it comes from
    `root_loader` — so it is the one place where declared and actual can differ.

    They differed. C1 stage D declared `cuda` and loaded its root through a raw
    `AutoModelForCausalLM.from_pretrained(...).eval()` with no transfer, and
    `depth.apply` reads `model_device(model)` — the fact, not the intent — so the
    entire parent replay would have run on the host CPU inside a paid GPU hour.
    """

    def __init__(self, declared: str, actual: str,
                 evidence: Mapping[str, Any] | None = None):
        self.declared = declared
        self.actual = actual
        self.evidence = dict(evidence or {})
        super().__init__(
            f"the root model's weights are on {actual}, but this path declares "
            f"device {declared!r}. STOP: every operator reads the weights' real "
            "device, so the path would execute somewhere other than where it is "
            "specified to. Load the root on the declared device — do not relax "
            f"the declaration to match a misplaced model. evidence={self.evidence}")


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


def verify_root_placement(model: Any, device: str) -> dict[str, Any]:
    """Every parameter and every buffer, not just the first one. Returns evidence.

    `model_device()` returns `next(model.parameters()).device` and stops there.
    That answers "where did this model start" and not "where is this model",
    which are different questions the moment anything is partially moved — a
    `device_map` shard, a buffer left behind by a hand-written `.to()` on
    submodules, or a `meta` tensor from a skipped materialization. Each of those
    presents as a working model whose first parameter is in the right place, and
    each would make an operator read or write on the wrong device mid-path.

    Four refusals, all fail-closed:

    * a **meta** parameter or buffer — no storage at all, so any read is a lie;
    * **mixed types**, e.g. some CPU and some CUDA;
    * **mixed CUDA ordinals**, even when every tensor is on a GPU;
    * the **wrong ordinal** when the declaration names one (`"cuda:1"`).

    An unindexed `"cuda"` accepts any single ordinal — that is what
    `Tensor.to("cuda")` means — but still requires exactly one.

    Deliberately does **not** move anything. Silently relocating a misplaced
    model would destroy the evidence that it was misplaced, and the caller
    declared a device precisely so that it would be honoured rather than
    approximated.
    """
    import torch

    declared = torch.device(device)
    named: list[tuple[str, Any]] = []
    n_params = 0
    for name, t in model.named_parameters():
        named.append((f"parameter {name}", t))
        n_params += 1
    n_buffers = 0
    for name, t in model.named_buffers():
        if t is None:                      # an unset optional buffer holds nothing
            continue
        named.append((f"buffer {name}", t))
        n_buffers += 1

    if not named:
        raise FixedPathRootDeviceMismatch(
            device, "<no tensors>",
            {"reason": "the model exposes no parameters or buffers, so its "
                       "placement cannot be verified at all",
             "declared": device, "n_parameters": 0, "n_buffers": 0})

    meta = [n for n, t in named if t.device.type == "meta"]
    if meta:
        raise FixedPathRootDeviceMismatch(
            device, "meta",
            {"reason": "meta tensors have no storage; this model was never "
                       "materialized", "declared": device,
             "meta_tensors": meta[:20], "n_meta": len(meta),
             "n_parameters": n_params, "n_buffers": n_buffers})

    seen: dict[str, list[str]] = {}
    for n, t in named:
        seen.setdefault(str(t.device), []).append(n)
    places = sorted(seen)

    evidence = {"declared": device, "n_parameters": n_params,
                "n_buffers": n_buffers, "devices": places,
                "n_tensors_checked": len(named)}

    if len(places) > 1:
        raise FixedPathRootDeviceMismatch(
            device, "+".join(places),
            {**evidence,
             "reason": "the model's tensors are split across devices",
             "examples": {p: seen[p][:5] for p in places}})

    actual = torch.device(places[0])
    if actual.type != declared.type or (
            declared.index is not None and actual.index != declared.index):
        raise FixedPathRootDeviceMismatch(device, places[0], evidence)

    return {**evidence, "resolved": places[0]}


def require_root_on_declared_device(model: Any, spec: FixedPathSpec) -> str:
    """Refuse a root whose weights are not on `spec.device`. Returns the device.

    Read from the tensors, never from `ctx.device`: the context field is the
    declaration, and a declaration cannot be evidence about itself.
    """
    return verify_root_placement(model, spec.device)["resolved"]


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
    deadline: Any = None,
) -> list[StepResult]:
    """Apply every step in order, identifying and gating each intermediate.

    `calibration_items` may be supplied to avoid re-resolving (the resolver
    verifies a content hash on every call); when omitted the profiles are
    resolved from `repo_root`, fail-closed.

    Either way the items pass through `prepare_calibration_items` before they
    reach an `OperatorContext`, so a caller-supplied mixture and a
    profile-resolved one arrive at DEPTH, FFN, WIDTH and ATTENTION in exactly one
    shape. Supplying items skips the resolver's content hash — it does not skip
    the item contract.

    `deadline` is handed straight to `OperatorContext.deadline`. It is the
    caller's budget object, not one this module invents: the search has passed
    one for a long time and the fixed path did not, so an operator whose work is
    measured in hours (`depth.causal_kl_greedy_v1` checks it per candidate) had
    nothing to consult here.
    """
    model = root_loader()
    # Before the first operator, and from the tensors rather than the
    # declaration. `root_loader` is the caller's, so this is the only step whose
    # placement the executor did not perform itself.
    require_root_on_declared_device(model, spec)
    return _run_steps(
        spec, adapter=adapter, model=model, indices=range(len(spec.steps)),
        parent_spec=adapter.spec_of(model), parent_digest=None,
        workdir=workdir, repo_root=repo_root, stats_cache=stats_cache,
        calibration_items=calibration_items, on_step=on_step,
        deadline=deadline)


def _run_steps(
    spec: FixedPathSpec,
    *,
    adapter: ArchitectureAdapter,
    model: Any,
    indices: Sequence[int] | range,
    parent_spec: ArchSpec,
    parent_digest: str | None,
    workdir: str | Path,
    repo_root: str | Path,
    stats_cache: StatsCache | None,
    calibration_items: Mapping[str, Sequence[Any]] | None,
    on_step: Callable[[StepResult], None] | None,
    deadline: Any,
) -> list[StepResult]:
    """The step loop, walked by ORIGINAL index.

    Shared by the whole-path and verified-suffix entry points so there is one
    implementation of "apply a step of this spec". `indices` are indices into
    `spec.steps`; a `StepResult.index` and its checkpoint directory are always
    the step's index in the FULL spec, never its position in the slice, because
    the identity of a step is where it sits on the frozen path.
    """
    import time

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    cache = stats_cache if stats_cache is not None else StatsCache(
        stats_spec=spec.stats_spec)
    resolved: dict[str, Sequence[Any]] = {
        key: prepare_calibration_items(items, profile_id=key)
        for key, items in dict(calibration_items or {}).items()}
    results: list[StepResult] = []

    for i in indices:
        step = spec.steps[i]
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
                # `resolve()` hands back the RAW frozen evidence — tokens under
                # `ids`, because that is what the pinned content hash is defined
                # over. The operators read `input_ids`. One boundary converts.
                resolved[profile.qualified_id] = prepare_calibration_items(
                    profile.resolve(repo_root),
                    profile_id=profile.qualified_id)
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
            deadline=deadline,
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


class FixedPathSuffixRefused(FixedPathError):
    """A verified-suffix execution was refused before any operator ran."""

    def __init__(self, reason: str, evidence: Mapping[str, Any]):
        self.reason = reason
        self.evidence = dict(evidence)
        super().__init__(
            f"verified-suffix execution refused: {reason}. STOP: nothing was "
            "executed. The suffix entry point exists to skip a prefix that has "
            "ALREADY been verified, so an unverified premise is not a slow path "
            f"to fall back to — it is the whole risk. evidence={self.evidence}")


@dataclass(frozen=True)
class VerifiedSuffix:
    """What the caller asserts, so this module can refuse without knowing C1.

    Every expectation is supplied by the caller. No digest, path hash, operator
    id or step index of any particular experiment appears in this file — the
    C1 session owns those constants and hands them in, exactly as it hands in
    the spec itself.
    """

    #: First ORIGINAL index to execute. The prefix below it is taken as given.
    start_index: int
    #: The StepResult that produced the parent, from the verified prefix run.
    parent: StepResult
    #: The frozen artifact digest the parent was pinned to.
    expected_parent_artifact_digest: str
    #: The frozen hash of the FULL path this suffix belongs to.
    expected_path_hash: str
    #: The prefix as executed elsewhere — normally the other arm's steps 0..n-1.
    prefix_reference_steps: tuple[FixedPathStep, ...]
    #: `(impl_id, profile_id)` for each step from `start_index` onward.
    expected_suffix_steps: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "parent_step_index": self.parent.index,
            "parent_checkpoint": self.parent.checkpoint_path,
            "parent_artifact_digest": self.parent.identity.artifact_digest,
            "parent_result_spec_hash": self.parent.result_spec_hash,
            "expected_parent_artifact_digest": self.expected_parent_artifact_digest,
            "expected_path_hash": self.expected_path_hash,
            "prefix_reference_steps": [s.as_dict()
                                       for s in self.prefix_reference_steps],
            "expected_suffix_steps": [list(s) for s in self.expected_suffix_steps],
        }


def verify_suffix_premise(spec: FixedPathSpec, vs: VerifiedSuffix) -> dict[str, Any]:
    """Every check that needs no model and no disk. Runs first, and cheaply.

    Ordered so the most structural failure is reported rather than a downstream
    symptom of it: a path that is not the frozen path makes every other question
    meaningless.
    """
    ev: dict[str, Any] = {"path_id": spec.path_id, **vs.as_dict()}

    if spec.spec_hash != vs.expected_path_hash:
        raise FixedPathSuffixRefused(
            "the spec is not the frozen path it claims to be",
            {**ev, "actual_path_hash": spec.spec_hash})

    n = len(spec.steps)
    if not 0 < vs.start_index < n:
        raise FixedPathSuffixRefused(
            f"start_index {vs.start_index} is not a proper suffix of {n} steps",
            ev)

    if vs.parent.index != vs.start_index - 1:
        raise FixedPathSuffixRefused(
            f"the supplied parent is step {vs.parent.index}, but a suffix "
            f"starting at {vs.start_index} must continue from step "
            f"{vs.start_index - 1}", ev)

    prefix = tuple(spec.steps[:vs.start_index])
    if prefix != tuple(vs.prefix_reference_steps):
        raise FixedPathSuffixRefused(
            "this path's prefix is not the prefix that was actually executed, "
            "so the parent on disk is not this path's parent",
            {**ev, "this_prefix": [s.as_dict() for s in prefix]})

    actual_suffix = tuple((s.impl_id, s.profile_id)
                          for s in spec.steps[vs.start_index:])
    if actual_suffix != tuple(tuple(s) for s in vs.expected_suffix_steps):
        raise FixedPathSuffixRefused(
            "the steps to execute are not the frozen ones",
            {**ev, "actual_suffix_steps": [list(s) for s in actual_suffix]})

    if vs.parent.identity.artifact_digest != vs.expected_parent_artifact_digest:
        raise FixedPathSuffixRefused(
            "the parent's realized artifact digest is not the frozen one", ev)
    if vs.parent.digest_expected != vs.expected_parent_artifact_digest:
        raise FixedPathSuffixRefused(
            "the parent step was not pinned to the frozen parent digest, so it "
            "was never gated against it", ev)
    if vs.parent.digest_matches is not True:
        raise FixedPathSuffixRefused(
            "the parent step did not record a digest MATCH; an unverified "
            "parent cannot be a verified prefix", ev)

    return {**ev, "premise": "verified", "actual_path_hash": spec.spec_hash}


def materialize_fixed_path_suffix(
    spec: FixedPathSpec,
    *,
    adapter: ArchitectureAdapter,
    root_loader: Callable[[], Any],
    workdir: str | Path,
    verified: VerifiedSuffix,
    repo_root: str | Path = ".",
    stats_cache: StatsCache | None = None,
    calibration_items: Mapping[str, Sequence[Any]] | None = None,
    on_step: Callable[[StepResult], None] | None = None,
    deadline: Any = None,
) -> tuple[list[StepResult], dict[str, Any]]:
    """Execute the tail of a frozen path from an ALREADY-VERIFIED parent.

    Two arms that share a prefix should not both compute it. The incumbent arm's
    replay already produced, gated and wrote the shared parent; the treatment arm
    differs only in its last step, so recomputing DEPTH, FFN and WIDTH would burn
    GPU hours to reproduce a checkpoint that is sitting on disk — and, worse, the
    first of those operators is **not applicable** to it, because that parent is
    already at the target depth. Handing the full spec and the step-2 parent to
    `materialize_fixed_path` therefore does not merely waste time; it raises.

    The alternative that must NOT be taken is building a one-step
    `FixedPathSpec` for the tail. That spec would have a different `path_hash`,
    and the arm's identity is the full frozen path — a result bound to a
    synthesized one-step path is a result for a different experiment. So the full
    spec is passed unchanged, its hash is checked against the frozen one, and
    only `indices` narrows. `StepResult.index` and the checkpoint directory keep
    their ORIGINAL numbering.

    Everything the caller asserts is checked before a single operator runs, in
    two rounds: `verify_suffix_premise` for the structural claims, then the
    physical ones — the checkpoint re-identified from disk, the loaded model's
    placement, and its ArchSpec against the parent's recorded `result_spec_hash`.

    Returns `(results, evidence)`; the evidence is what the caller writes into
    the record, so that a reader can check the premise without rerunning it.
    """
    premise = verify_suffix_premise(spec, verified)

    model = root_loader()
    placement = verify_root_placement(model, spec.device)

    parent_spec = adapter.spec_of(model)
    if parent_spec.spec_hash != verified.parent.result_spec_hash:
        raise FixedPathSuffixRefused(
            "the loaded parent's architecture is not the one the verified step "
            "recorded",
            {**premise, "loaded_spec_hash": parent_spec.spec_hash,
             "loaded_spec": parent_spec.describe()})

    # Re-identified from the FILES, not trusted from the StepResult: the record
    # says what was written, and this asks what is there now.
    reident = identify_checkpoint(
        verified.parent.checkpoint_path, adapter=adapter, spec=parent_spec,
        num_parameters=adapter.param_count(parent_spec))
    if reident.artifact_digest != verified.expected_parent_artifact_digest:
        raise FixedPathSuffixRefused(
            "the parent checkpoint on disk no longer identifies to the frozen "
            "parent digest",
            {**premise, "reidentified_artifact_digest": reident.artifact_digest})

    results = _run_steps(
        spec, adapter=adapter, model=model,
        indices=range(verified.start_index, len(spec.steps)),
        parent_spec=parent_spec,
        parent_digest=reident.artifact_digest,
        workdir=workdir, repo_root=repo_root, stats_cache=stats_cache,
        calibration_items=calibration_items, on_step=on_step,
        deadline=deadline)

    evidence = {
        **premise,
        "executed_step_indices": list(range(verified.start_index, len(spec.steps))),
        "prefix_step_indices_not_executed": list(range(verified.start_index)),
        "root_placement": placement,
        "parent_reidentified_artifact_digest": reident.artifact_digest,
        "parent_reidentified_from": str(verified.parent.checkpoint_path),
        "loaded_parent_spec_hash": parent_spec.spec_hash,
    }
    return results, evidence


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


def write_suffix_execution_record(
    spec: FixedPathSpec,
    results: Sequence[StepResult],
    path: str | Path,
    *,
    runtime: Mapping[str, Any],
    suffix_evidence: Mapping[str, Any],
    calibration: Mapping[str, Any] | None = None,
) -> Path:
    """The auditable record of a verified-suffix execution.

    Deliberately a DIFFERENT schema from the replay record, and it says so in
    two places. A replay reproduces a checkpoint whose digest was frozen in
    advance, and "matched" is the finding; this executes a step whose output has
    never existed before, so there is nothing to match and claiming otherwise
    would manufacture a reproducibility result. `output_digest_was_pre_pinned`
    is written as an explicit `false` rather than left absent, because an absent
    field reads as an oversight and a present `false` reads as a decision.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "aadistill.autoinit.fixed_path_suffix_execution/v1",
        "_what_this_is": (
            "one arm's tail, executed from a parent that ANOTHER run produced "
            "and gated. It is not a replay: the executed step's output digest "
            "was never pinned, so this record reports an identity, not a match."),
        "path": spec.as_dict(),
        "path_hash": spec.spec_hash,
        "is_replay": False,
        "output_digest_was_pre_pinned": False,
        "output_digest_claim": (
            "NONE. No expected digest exists for this output and none is "
            "asserted. Do not read this record as a replay match."),
        "verified_prefix": dict(suffix_evidence),
        "calibration": dict(calibration or {}),
        "runtime": dict(runtime),
        "steps": [r.as_dict() for r in results],
        "executed_step_indices": [r.index for r in results],
        "output": results[-1].as_dict() if results else None,
        "n_pinned": sum(1 for r in results if r.digest_expected is not None),
    }
    p.write_text(json.dumps(record, indent=1) + "\n")
    return p
