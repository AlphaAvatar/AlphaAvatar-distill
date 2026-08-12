"""Operator kinds, operator implementations, and the registry between them.

**A kind is not an implementation.** ``DEPTH`` is a structural dimension — which
blocks survive. ``depth.positional_v0`` and ``depth.causal_kl_greedy_v1`` are two
different algorithms for deciding it, and E8a showed they disagree: the
positional heuristic and the causal search share exactly one removed layer out of
eight. Collapsing the two concepts would make "we tried DEPTH" mean nothing, and
would make a later ``depth.xxx_v2`` either overwrite history or need a new kind.

Four consequences are enforced here rather than documented:

*Implementation ids are immutable.* An implementation's *declared semantics* —
kind, capabilities, the structural fields it modifies and preserves, its
calibration need, its objective, its determinism — hash to a signature. The
registry refuses to bind an existing id to a different signature, and
``verify_ledger`` compares the live registry against a committed ledger file. An
algorithm change is a new id (``..._v2``), never a redefinition of ``..._v1``.

*Dispatch is by capability, not by name.* ``applicable_implementations`` filters
on the adapter's declared capabilities and the implementation's own precondition
check. The search engine never branches on a kind, which is what lets a
``MOE_EXPERT_SET`` kind or an MLA-specific ``ATTENTION`` implementation register
from outside this package with no core edit.

*Declarations are checked against reality.* An operator declares the structural
fields it modifies and the ones it guarantees untouched; ``execute`` diffs the
before/after ``ArchSpec`` and raises if the operator touched anything else. A
comment promising "attention only" is not a guarantee; this is.

*The kind set is open.* ``register_kind`` accepts ids this module has never heard
of. The four v1 kinds are registered at the bottom as data, not as an enum.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..arch import ArchitectureAdapter, ArchSpec
from ..calibration import CalibrationProfile
from ..metrics import OperatorLocalMetrics


class OperatorError(RuntimeError):
    """An operator could not be registered, planned or applied."""


class ContractViolation(OperatorError):
    """An operator did something other than what it declared."""


class CalibrationNeed(Enum):
    """What an implementation must be fed to make its decision."""

    NONE = "none"
    ACTIVATION_STATS = "activation_stats"
    FORWARD_LOGITS = "forward_logits"


# --- operator kinds ---------------------------------------------------------


@dataclass(frozen=True)
class OperatorKindSpec:
    kind_id: str
    dimension: str
    description: str

    def as_dict(self) -> dict[str, str]:
        return {"kind_id": self.kind_id, "dimension": self.dimension,
                "description": self.description}


_KINDS: dict[str, OperatorKindSpec] = {}


def register_kind(spec: OperatorKindSpec, *, replace: bool = False) -> OperatorKindSpec:
    existing = _KINDS.get(spec.kind_id)
    if existing is not None and not replace:
        if existing != spec:
            raise OperatorError(
                f"operator kind {spec.kind_id!r} is already registered with a "
                "different definition")
        return existing
    _KINDS[spec.kind_id] = spec
    return spec


def get_kind(kind_id: str) -> OperatorKindSpec:
    if kind_id not in _KINDS:
        raise KeyError(f"no operator kind {kind_id!r}; registered: {sorted(_KINDS)}")
    return _KINDS[kind_id]


def registered_kinds() -> list[str]:
    return sorted(_KINDS)


def unregister_kind(kind_id: str) -> None:
    """Test-only."""
    _KINDS.pop(kind_id, None)


# --- execution plumbing -----------------------------------------------------


@dataclass
class OperatorContext:
    """Everything an operator is allowed to see when it runs.

    Note what is *not* here: any other search state, the beam, the ranking
    policy, or the final promotion battery. An operator sees its parent, the
    original teacher (as a loader, because most operators never need it), the
    target it is working toward, and its own calibration data.
    """

    adapter: ArchitectureAdapter
    model: Any
    parent_spec: ArchSpec
    target_spec: ArchSpec
    profile: CalibrationProfile
    calibration_items: Sequence[Any]
    seed: int
    device: str = "cpu"
    dtype: Any = None
    workdir: Path | None = None
    root_teacher_loader: Any = None
    config: Mapping[str, Any] = field(default_factory=dict)
    tokenizer: Any = None
    #: Shared activation-statistics cache and this invocation's key. Two operators
    #: on the *same* parent under the *same* profile share one pass; the key
    #: includes the parent's artifact digest, so reuse across parents is
    #: impossible rather than merely discouraged. `None` disables sharing.
    stats_cache: Any = None
    stats_cache_key: str | None = None

    def cached_stats(self, collect):
        """Statistics for this parent under this profile, collected at most once."""
        if self.stats_cache is None or self.stats_cache_key is None:
            return collect()
        return self.stats_cache.get_or_collect(self.stats_cache_key, collect)


@dataclass(frozen=True)
class OperatorPlan:
    """What an operator will do, computed without doing it.

    The cost model and the search's reachability check both need the resulting
    spec before any weights move; a plan that cannot be produced is a
    precondition failure, not a runtime crash halfway through a 4B checkpoint.
    """

    impl_id: str
    result_spec: ArchSpec
    forward_passes: int
    stats_passes: int
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"impl_id": self.impl_id, "result_spec": self.result_spec.as_dict(),
                "result_spec_hash": self.result_spec.spec_hash,
                "forward_passes": self.forward_passes,
                "stats_passes": self.stats_passes, "notes": self.notes}


@dataclass
class OperatorOutcome:
    model: Any
    local_metrics: OperatorLocalMetrics
    trace: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)


# --- the implementation contract -------------------------------------------


class OperatorImplementation(ABC):
    """One versioned algorithm for one operator kind."""

    impl_id: str = ""
    kind: str = ""
    version: int = 0
    description: str = ""
    required_capabilities: frozenset[str] = frozenset()
    #: Structural fields this implementation may change.
    modifies: frozenset[str] = frozenset()
    #: Structural fields it guarantees it will not change. Checked, not trusted.
    preserves: frozenset[str] = frozenset()
    calibration: CalibrationNeed = CalibrationNeed.NONE
    objective: str = ""
    deterministic: bool = True
    requires_seed: bool = False
    produces: tuple[str, ...] = ()
    target_validation: str = "result spec must move the modified fields toward the target"

    # --- declared semantics ------------------------------------------------

    def signature(self) -> dict[str, Any]:
        """The declaration that an id is permanently bound to."""
        return {
            "impl_id": self.impl_id,
            "kind": self.kind,
            "version": self.version,
            "required_capabilities": sorted(self.required_capabilities),
            "modifies": sorted(self.modifies),
            "preserves": sorted(self.preserves),
            "calibration": self.calibration.value,
            "objective": self.objective,
            "deterministic": self.deterministic,
            "requires_seed": self.requires_seed,
            "produces": list(self.produces),
            "target_validation": self.target_validation,
        }

    @property
    def signature_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.signature(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def declare(self) -> dict[str, Any]:
        return {**self.signature(), "description": self.description,
                "signature_hash": self.signature_hash}

    # --- behaviour ----------------------------------------------------------

    def supported_by(self, adapter: ArchitectureAdapter) -> bool:
        return self.required_capabilities <= adapter.capabilities

    def applicable(self, spec: ArchSpec, target: ArchSpec,
                   adapter: ArchitectureAdapter) -> tuple[bool, str]:
        """Preconditions. Returns (ok, reason) so a rejection is auditable."""
        if not self.supported_by(adapter):
            missing = sorted(self.required_capabilities - adapter.capabilities)
            return False, f"adapter lacks {missing}"
        undeclared = sorted((self.modifies | self.preserves)
                            - set(adapter.structural_fields))
        if undeclared:
            return False, f"declares fields the adapter does not manage: {undeclared}"
        for f in sorted(self.modifies):
            if spec[f] == target[f]:
                return False, f"{f} already at target ({spec[f]})"
        return True, "ok"

    @abstractmethod
    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        ...

    @abstractmethod
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        ...

    # --- the enforced wrapper ----------------------------------------------

    def execute(self, ctx: OperatorContext) -> OperatorOutcome:
        """Apply, then verify the operator did only what it declared.

        This is the guarantee behind "operators only mutate declared structural
        fields". It runs on every application, including inside the dry run, so
        a contract break cannot survive to a paid pod.
        """
        before = ctx.adapter.spec_of(ctx.model)
        if not before.matches(ctx.parent_spec):
            raise ContractViolation(
                f"{self.impl_id}: the model handed in is {before.describe()} but the "
                f"state claims {ctx.parent_spec.describe()}")
        if self.calibration is not CalibrationNeed.NONE and not ctx.calibration_items:
            raise OperatorError(
                f"{self.impl_id} declares calibration={self.calibration.value} and was "
                "given no calibration items")
        outcome = self.apply(ctx)
        if outcome.model is ctx.model:
            raise ContractViolation(
                f"{self.impl_id} returned the parent model itself. An operator builds a "
                "child; consuming the parent in place makes the node unre-expandable "
                "and resume non-deterministic.")
        still = ctx.adapter.spec_of(ctx.model)
        if not still.matches(before):
            raise ContractViolation(
                f"{self.impl_id} mutated the parent model in place: it was "
                f"{before.describe()} and is now {still.describe()}")
        after = ctx.adapter.spec_of(outcome.model)
        changed = before.diff(after)
        illegal = sorted(changed - self.modifies)
        if illegal:
            raise ContractViolation(
                f"{self.impl_id} changed structural fields {illegal}, which it did not "
                f"declare; it declares modifies={sorted(self.modifies)}")
        broken = sorted(changed & self.preserves)
        if broken:
            raise ContractViolation(
                f"{self.impl_id} changed {broken} after declaring them preserved")
        if not outcome.local_metrics.impl_id == self.impl_id:
            raise ContractViolation(
                f"{self.impl_id} returned metrics labelled "
                f"{outcome.local_metrics.impl_id!r}")
        return outcome


# --- implementation registry -----------------------------------------------


_IMPLEMENTATIONS: dict[str, OperatorImplementation] = {}


def register_implementation(impl: OperatorImplementation, *,
                            replace: bool = False) -> OperatorImplementation:
    if not impl.impl_id:
        raise OperatorError("implementation must declare an impl_id")
    if impl.kind not in _KINDS:
        raise OperatorError(
            f"{impl.impl_id} declares kind {impl.kind!r}, which is not registered; "
            "register the kind first (a new kind needs no core edit)")
    if not impl.modifies:
        raise OperatorError(f"{impl.impl_id} declares no structural fields to modify")
    overlap = sorted(impl.modifies & impl.preserves)
    if overlap:
        raise OperatorError(
            f"{impl.impl_id} declares {overlap} as both modified and preserved")
    existing = _IMPLEMENTATIONS.get(impl.impl_id)
    if existing is not None and not replace:
        if existing.signature_hash != impl.signature_hash:
            raise OperatorError(
                f"implementation id {impl.impl_id!r} is already bound to signature "
                f"{existing.signature_hash[:12]}; re-binding it to "
                f"{impl.signature_hash[:12]} would rewrite the meaning of every "
                "manifest that recorded it. Register a new version instead.")
        return existing
    _IMPLEMENTATIONS[impl.impl_id] = impl
    return impl


def get_implementation(impl_id: str) -> OperatorImplementation:
    if impl_id not in _IMPLEMENTATIONS:
        raise KeyError(
            f"no operator implementation {impl_id!r}; registered: "
            f"{sorted(_IMPLEMENTATIONS)}")
    return _IMPLEMENTATIONS[impl_id]


def registered_implementations() -> list[str]:
    return sorted(_IMPLEMENTATIONS)


def implementations_for_kind(kind_id: str) -> list[OperatorImplementation]:
    return [_IMPLEMENTATIONS[k] for k in sorted(_IMPLEMENTATIONS)
            if _IMPLEMENTATIONS[k].kind == kind_id]


def unregister_implementation(impl_id: str) -> None:
    """Test-only."""
    _IMPLEMENTATIONS.pop(impl_id, None)


def applicable_implementations(
    adapter: ArchitectureAdapter,
    spec: ArchSpec,
    target: ArchSpec,
    *,
    exclude_kinds: Sequence[str] = (),
    allow_impls: Sequence[str] | None = None,
) -> list[tuple[OperatorImplementation, str]]:
    """Every implementation that could legally run on ``spec``, with its reason.

    Capability dispatch: the engine calls this and never inspects a kind name.
    Rejections are returned alongside acceptances by ``rejected_implementations``
    so a pruned branch can say *why* nothing expanded it.
    """
    out = []
    excluded = set(exclude_kinds)
    for impl_id in sorted(_IMPLEMENTATIONS):
        impl = _IMPLEMENTATIONS[impl_id]
        if impl.kind in excluded:
            continue
        if allow_impls is not None and impl_id not in allow_impls:
            continue
        ok, reason = impl.applicable(spec, target, adapter)
        if ok:
            out.append((impl, reason))
    return out


def rejected_implementations(
    adapter: ArchitectureAdapter,
    spec: ArchSpec,
    target: ArchSpec,
    *,
    exclude_kinds: Sequence[str] = (),
) -> list[dict[str, str]]:
    out = []
    excluded = set(exclude_kinds)
    for impl_id in sorted(_IMPLEMENTATIONS):
        impl = _IMPLEMENTATIONS[impl_id]
        if impl.kind in excluded:
            out.append({"impl_id": impl_id, "reason": "kind already applied on this path"})
            continue
        ok, reason = impl.applicable(spec, target, adapter)
        if not ok:
            out.append({"impl_id": impl_id, "reason": reason})
    return out


# --- the immutability ledger ------------------------------------------------

LEDGER_PATH = Path("configs/autoinit/operator_ledger.json")


def registry_ledger() -> dict[str, Any]:
    return {
        "schema": "aadistill.autoinit.operator_ledger/v1",
        "kinds": {k: _KINDS[k].as_dict() for k in sorted(_KINDS)},
        "implementations": {
            k: _IMPLEMENTATIONS[k].declare() for k in sorted(_IMPLEMENTATIONS)
        },
    }


def verify_ledger(path: str | Path = LEDGER_PATH, *,
                  repo_root: str | Path = ".") -> dict[str, Any]:
    """Compare the live registry against the committed ledger.

    Drift is reported in three buckets, and only one of them is benign:
    ``added`` (a new implementation — expected; re-freeze the ledger),
    ``removed`` (a historical id disappeared — a manifest that cites it can no
    longer be interpreted), and ``changed`` (an id's declared semantics moved —
    the failure this whole mechanism exists to catch).
    """
    p = Path(repo_root) / path
    if not p.is_file():
        raise OperatorError(f"operator ledger {p} is missing; freeze it before a run")
    frozen = json.loads(p.read_text())
    live = registry_ledger()
    frozen_impls = frozen.get("implementations", {})
    live_impls = live["implementations"]

    changed = sorted(
        k for k in set(frozen_impls) & set(live_impls)
        if frozen_impls[k].get("signature_hash") != live_impls[k]["signature_hash"])
    removed = sorted(set(frozen_impls) - set(live_impls))
    added = sorted(set(live_impls) - set(frozen_impls))
    report = {"changed": changed, "removed": removed, "added": added,
              "ledger_path": str(p), "ok": not changed and not removed}
    if changed:
        raise OperatorError(
            f"operator implementations {changed} changed their declared semantics "
            "while keeping their ids. A historical id is immutable — register a new "
            "version instead.")
    if removed:
        raise OperatorError(
            f"operator implementations {removed} are in the ledger but not in the "
            "registry; manifests citing them can no longer be interpreted")
    return report


def write_ledger(path: str | Path = LEDGER_PATH, *, repo_root: str | Path = ".") -> Path:
    p = Path(repo_root) / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(registry_ledger(), indent=2, sort_keys=True) + "\n")
    return p


# --- the four v1 kinds, as data --------------------------------------------

DEPTH = register_kind(OperatorKindSpec(
    "DEPTH", "block count",
    "Which teacher blocks survive. Changes the block list only; every surviving "
    "block keeps its own widths."))

RESIDUAL_WIDTH = register_kind(OperatorKindSpec(
    "RESIDUAL_WIDTH", "residual stream dimension",
    "The dimension of the residual stream every block reads and writes. Touches "
    "every stream-facing projection, the embedding and the final norm."))

FFN = register_kind(OperatorKindSpec(
    "FFN", "feed-forward intermediate width",
    "The per-block feed-forward intermediate dimension."))

ATTENTION = register_kind(OperatorKindSpec(
    "ATTENTION", "attention head structure",
    "Head counts and head grouping. Family-specific: an MHA, GQA, MLA or linear "
    "attention implementation registers separately under this kind."))
