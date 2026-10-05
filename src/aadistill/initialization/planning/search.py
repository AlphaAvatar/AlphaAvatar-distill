"""Bounded, deterministic, resumable Beam Search over initialization paths.

The engine's whole job is to apply this cycle and refuse to skip any part of it:

    apply operator -> materialize -> canonical reload -> hash -> validate
      -> measure that exact checkpoint -> bind metrics to the hash
      -> only then rank or expand

Nothing in this file knows what a Qwen3 block looks like, which structural fields
exist, or what ``DEPTH`` means. It asks the adapter for spec algebra and model
lifecycle, the registry for whatever implementations are applicable *by
capability*, and the policy for the beam. A new operator kind, a new attention
family or an MoE adapter therefore needs no edit here — proven by test rather
than asserted.

Order is searched, not assumed. A kind is applied at most once per path (v1), so
the reachable leaves are the permutations of the kinds the target requires, times
the implementations for each kind, times the calibration profile chosen at each
invocation. Every operator runs against the checkpoint the previous operators
produced, and its local reference is that parent; the global reference for state
evaluation stays the original teacher. Both are recorded explicitly.

Resume is exact because state ids are content-derived: the journal is replayed,
any state already carrying a hash-bound measurement is restored rather than
recomputed, and the first uncompleted expansion runs live. Same config, same
seed, same tree.
"""

from __future__ import annotations

import json
import time
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from aadistill.infrastructure.manifest import sha256_file, sha256_json
from aadistill.initialization.specs.arch import ArchitectureAdapter, ArchSpec
from aadistill.initialization.specs.artifact import (
    CheckpointIdentity,
    identify_checkpoint,
)
from aadistill.initialization.calibration.profiles import (
    CalibrationProfile,
    consumes_calibration,
    profile_for,
)
from aadistill.initialization.specs.materialization import (
    MaterializationIdentity,
    NumericalEnvironment,
)
from aadistill.initialization.specs.metrics import StateEvalSuite, StateEvaluation
from aadistill.initialization.device import model_device
from aadistill.initialization.execution import DEFAULT_EXECUTION, ExecutionConfig
from aadistill.initialization.scoring.content import scoring_content_config
from aadistill.initialization.scoring.protocol_identity import (
    PROTOCOL_FIELD,
    measurement_is_comparable,
)
from aadistill.initialization.scoring.support import (
    FULL_VOCAB_V1, DistributionSupport,
)
from aadistill.initialization.scoring.positions import (
    ALL_POSITIONS_V1,
    ScoringPositionPolicy,
    policy_config,
)
from aadistill.initialization.statistics.spec import (
    DEFAULT_STATS_SPEC,
    StatsCache,
    StatsSpec,
    stats_cache_key,
)
from aadistill.runtime.telemetry import TelemetrySink
from aadistill.initialization.operators.base import (
    OperatorContext,
    OperatorImplementation,
    applicable_implementations,
    get_implementation,
    rejected_implementations,
)
from aadistill.initialization.planning.ranking import (
    BeamRankingPolicy,
    BeamSchedule,
    RankingResult,
)
from aadistill.initialization.specs.state import (
    InitializationState,
    OperatorStep,
    StateStore,
    StateValidity,
    child_state,
    compute_state_id,
    make_root_state,
)


class SearchError(RuntimeError):
    """The search cannot proceed as configured."""


class SearchDeadlineExceeded(SearchError):
    """The search ran past its wall-clock budget and stopped, fail-closed."""


@dataclass
class Deadline:
    """A real runtime budget, checked *inside* expensive work.

    Added 2026-08-19. ``--search-minutes 180.0`` already existed and was already
    priced, but it reached only ``self.afford(...)`` in the driver — an
    affordability check *before* the search starts. Nothing consulted a clock
    afterwards: this module recorded ``elapsed`` and ``wall_seconds`` and never
    compared them to anything, and ``_expand_one`` had no clock at all. So one
    expansion ran 10.78 h against a 3.0 h budget for the whole search and would
    have continued to the watchdog's ceiling.

    Deliberately **not** a :class:`SearchConfig` field: that dataclass "fixes a
    search run, and therefore everything that hashes". A wall-clock budget is an
    operational limit, not part of the search's identity, and putting it there
    would make every re-pricing a different search.

    ``check()`` is cheap enough to call per candidate — one ``time.monotonic``
    and a comparison — which is the granularity that matters, because the
    expensive thing here is a single candidate evaluation.
    """

    seconds: float
    started: float = field(default_factory=time.monotonic)
    #: Set when the deadline fires, so a caller can report where it stopped.
    fired_at: str = ""

    @classmethod
    def from_minutes(cls, minutes: float | None) -> "Deadline | None":
        return None if minutes is None else cls(seconds=float(minutes) * 60.0)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return self.seconds - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def check(self, where: str = "") -> None:
        """Raise if the budget is spent. Fail closed: no partial credit."""
        if self.expired():
            self.fired_at = where or self.fired_at or "unspecified"
            raise SearchDeadlineExceeded(
                f"search deadline exceeded after {self.elapsed() / 60:.1f} min "
                f"(budget {self.seconds / 60:.1f} min) at {self.fired_at}. The "
                "search stopped rather than continuing to the cost backstop.")

    def as_dict(self) -> dict[str, Any]:
        return {"budget_minutes": round(self.seconds / 60, 4),
                "elapsed_minutes": round(self.elapsed() / 60, 4),
                "expired": self.expired(),
                "fired_at": self.fired_at or None}


CalibrationLoader = Callable[[CalibrationProfile], Sequence[Mapping[str, Any]]]
Measurer = Callable[[Any, str], StateEvaluation]


def expansion_profiles(
    implementation,
    profiles: Sequence[CalibrationProfile],
    impl_profiles: Mapping[str, Sequence[str]] | None = None,
) -> list[CalibrationProfile]:
    """Every profile one implementation is offered for ONE parent, in order.

    Module-level, and the ONE definition of the branching factor. `BeamSearch`
    calls it to expand; a cost model calls it to count. Two implementations of
    this rule disagree immediately — the second one branched a
    `CalibrationNeed.NONE` operator over every active profile and over-counted
    the root's children — and a price for a space the search does not run is
    worse than no price. See `docs/core-provenance.md`.

    Both rules live here:

    * an implementation declaring ``CalibrationNeed.NONE`` is offered **once**,
      against the canonical sentinel, however many profiles are active;
    * anything else is offered every active profile, unless ``impl_profiles``
      restricts it. An implementation absent from that mapping branches over
      everything, which is what every search before the field existed did.
    """
    active = sorted(profiles, key=lambda p: p.qualified_id)
    if not consumes_calibration(implementation):
        #: `profile_for` returns the sentinel and ignores the argument; passing
        #: the first active profile only keeps the call total.
        return [profile_for(implementation, active[0])]
    allowed = (impl_profiles or {}).get(implementation.impl_id)
    if allowed is None:
        return active
    keep = set(allowed)
    return [p for p in active if p.qualified_id in keep]


@dataclass
class SearchConfig:
    """Everything that fixes a search run, and therefore everything that hashes."""

    run_id: str
    target_spec: ArchSpec
    schedule: BeamSchedule
    seed: int
    workdir: Path
    profiles: tuple[CalibrationProfile, ...]
    policy: BeamRankingPolicy
    suite: StateEvalSuite
    stats_spec: StatsSpec = DEFAULT_STATS_SPEC
    max_shard_size: str | int | None = None
    allowed_impls: tuple[str, ...] | None = None
    #: Which of `profiles` a given implementation may branch over, by impl_id.
    #: An implementation with no entry branches over all of them, which is what
    #: every search before this one did and remains the default.
    #:
    #: This exists because `profiles` is a property of the RUN and the question
    #: is sometimes a property of ONE operator. A search that has already fixed
    #: three operators' mixtures and wants to vary the fourth's had no way to
    #: say so: adding the second profile to `profiles` branched all four, which
    #: is a full factorial and a different, much more expensive experiment.
    #: Restricting the space is not the same as restricting the run's identity —
    #: this is part of `as_dict`, so two searches that reach different leaves
    #: cannot share a `config_hash`.
    impl_profiles: Mapping[str, tuple[str, ...]] | None = None
    #: WHICH POSITIONS every calibrated operator objective and the global state
    #: metrics are allowed to care about. A `SearchConfig` field — not an
    #: `ExecutionConfig` one — because it changes the estimand rather than the
    #: schedule: the same path scored over supervised assistant positions and
    #: over every position are two different hypotheses and must not share a
    #: `config_hash`.
    #:
    #: It lives on the RUN rather than per operator because of what the
    #: alternative permits: a candidate chosen by target-aware operators must not
    #: then be pruned by a full-sequence beam metric. One policy, consumed by the
    #: operators and by the measurer, makes that disagreement unexpressible.
    position_policy: ScoringPositionPolicy = ALL_POSITIONS_V1
    #: WHICH VOCABULARY PARTITION every calibrated KL objective and the global
    #: state metrics are reduced over. A `SearchConfig` field for exactly the
    #: reasons `position_policy` is one: it changes the ESTIMAND rather than the
    #: schedule, so the same path scored over the full vocabulary and over a
    #: reference-defined Top-K plus a tail bucket are two different hypotheses
    #: and must not share a `config_hash`.
    #:
    #: It lives on the RUN rather than per operator because of what the
    #: alternative permits, which is the same hazard again: a candidate chosen by
    #: Top-K operators must not then be pruned by a full-vocabulary beam metric.
    #: One support, consumed by the operators AND by the measurer, makes that
    #: disagreement unexpressible -- and the measurer is asked for its own support
    #: at construction, below, rather than trusted to match.
    #:
    #: `FULL_VOCAB_V1` is the historical default and is OMITTED from `as_dict`, so
    #: every committed search keeps the `config_hash` its own record carries.
    #: `src/aadistill` holds no `top_k` value: a campaign's K is the experiment
    #: layer's, supplied as a `DistributionSupport`.
    distribution_support: DistributionSupport = FULL_VOCAB_V1
    #: The measurement protocol the injected measurer takes its numbers under,
    #: as declared by the driver that built it — `StateEvaluator` exposes it as
    #: `measurement_protocol_id`. ONE field covering the suite's content, the
    #: scoring content, the policy, the reduction semantics and the execution,
    #: so resume and the measurer-agreement check ask one question instead of a
    #: list that grows until something is forgotten.
    #:
    #: `None` is the historical default: a run that declares none is judged by
    #: the two fields a pre-identity record carries. See
    #: `aadistill.initialization.scoring.protocol_identity`.
    measurement_protocol_id: str | None = None
    max_depth: int | None = None
    allow_kind_repeat: bool = False
    device: str = "cpu"
    #: Drop the weights of pruned states once their metrics are hash-bound. The
    #: metrics, hashes, traces and prune reasons stay in the journal; only the
    #: bytes go. At 4B-class intermediates this is the difference between ~1 TB
    #: and ~100 GB of working storage.
    prune_weights: bool = True
    keep_leaf_weights: bool = True
    notes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target_spec": self.target_spec.as_dict(),
            "target_spec_hash": self.target_spec.spec_hash,
            "schedule": self.schedule.as_dict(),
            "seed": self.seed,
            "profiles": [p.qualified_id for p in self.profiles],
            "profile_hashes": {p.qualified_id: p.profile_hash for p in self.profiles},
            "policy": self.policy.qualified_id,
            "policy_hash": self.policy.policy_hash,
            "suite": self.suite.qualified_id,
            "suite_hash": self.suite.suite_hash,
            "allowed_impls": list(self.allowed_impls) if self.allowed_impls else None,
            #: Normalised when present, because the reachable leaf set is what
            #: this describes and dict order is not part of it. ABSENT when
            #: unset — see `config_hash`.
            **({"impl_profiles": {k: sorted(v) for k, v
                                  in sorted(self.impl_profiles.items())}}
               if self.impl_profiles else {}),
            #: ABSENT at the incumbent policy, present otherwise — the same rule
            #: and the same reason as `impl_profiles` above. Every committed
            #: search was run over all positions, so a search recorded before
            #: this field existed must still hash to the value its own record
            #: carries; a search that changes the scoring positions gets a
            #: different `config_hash`, which is the point.
            **({"position_policy": self.position_policy.qualified_id,
                "position_policy_hash": self.position_policy.policy_hash}
               if self.position_policy.policy_hash
               != ALL_POSITIONS_V1.policy_hash else {}),
            #: ABSENT at the full vocabulary, present otherwise -- the same rule
            #: and the same reason as `position_policy` above. A search recorded
            #: before this field existed reduced over the whole vocabulary, so it
            #: must still hash to the value its own record carries; a search that
            #: coarsens the partition gets a different `config_hash`, which is the
            #: point of putting it here.
            **({} if self.distribution_support.is_full_vocab else
               {"distribution_support": self.distribution_support.as_dict()}),
            #: ABSENT when undeclared, for the same compatibility reason: a
            #: search recorded before measurement-protocol identity existed must
            #: still hash to the value its own record carries.
            **({"measurement_protocol_id": self.measurement_protocol_id}
               if self.measurement_protocol_id else {}),
            "max_depth": self.max_depth,
            "allow_kind_repeat": self.allow_kind_repeat,
            "device": self.device,
            "prune_weights": self.prune_weights,
            "keep_leaf_weights": self.keep_leaf_weights,
            "stats_spec": self.stats_spec.as_dict(),
            "max_shard_size": self.max_shard_size,
            "notes": dict(self.notes),
        }

    @property
    def config_hash(self) -> str:
        #: `as_dict` OMITS `impl_profiles` when it is unset rather than emitting
        #: a null, so a search recorded before the field existed still hashes to
        #: the value its own record carries and stays verifiable against this
        #: code. A search that DOES restrict gets a different hash, which is the
        #: point: two searches reaching different leaves must not share an
        #: identity. See `docs/core-provenance.md`.
        return sha256_json(self.as_dict())


@dataclass
class LevelRecord:
    level: int
    expanded_from: tuple[str, ...]
    generated: tuple[str, ...]
    ranking: RankingResult | None
    leaves: tuple[str, ...]
    dead_ends: tuple[dict[str, str], ...]
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "expanded_from": list(self.expanded_from),
            "generated": list(self.generated),
            "ranking": self.ranking.as_dict() if self.ranking else None,
            "leaves": list(self.leaves),
            "dead_ends": [dict(d) for d in self.dead_ends],
            "seconds": self.seconds,
        }


class BeamSearch:
    """The engine. Family-agnostic by construction."""

    def __init__(
        self,
        *,
        adapter: ArchitectureAdapter,
        config: SearchConfig,
        root_teacher_id: str,
        root_teacher_sha256: str,
        root_loader: Callable[[], Any],
        calibration_loader: CalibrationLoader,
        measurer: Measurer,
        root_spec: ArchSpec | None = None,
        deadline: "Deadline | None" = None,
        execution: ExecutionConfig = DEFAULT_EXECUTION,
        numerics: NumericalEnvironment | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        #: Runtime only. Never hashed — see `Deadline`.
        self.deadline = deadline
        #: The declared numerical conditions, which together with `execution`
        #: form each state's materialization identity. DECLARED BY THE CALLER and
        #: not sniffed: a fingerprint resolved from whatever model happened to be
        #: loaded would be a fingerprint of an accident. `None` keeps the
        #: historical behaviour — no materialization identity, resume keyed on
        #: the state id alone — which is what every committed search ran under
        #: and what the toy fixtures construct.
        self.numerics = numerics
        #: Runtime only, for the same reason and by the same rule: HOW the
        #: operators run. Deliberately NOT a `SearchConfig` field, because that
        #: dataclass "fixes a search run, and therefore everything that hashes".
        self.execution = execution
        self.root_teacher_id = root_teacher_id
        self.root_teacher_sha256 = root_teacher_sha256
        self.root_loader = root_loader
        self.calibration_loader = calibration_loader
        self.measurer = measurer
        self.workdir = Path(config.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.store = StateStore(self.workdir / "states.jsonl")

        self._root_model: Any | None = None
        self._root_spec = root_spec
        self.states: dict[str, InitializationState] = {}
        self.levels: list[LevelRecord] = []
        self.leaves: list[InitializationState] = []
        self.resumed_ids: set[str] = set()
        self._calibration_cache: dict[str, Sequence[Mapping[str, Any]]] = {}
        self._journal: dict[str, dict[str, Any]] = {}
        # ONE ENTRY PER ACTIVE PROFILE, not one entry.
        #
        # `_candidate_expansions` orders by impl_id, so a P=2 parent is expanded
        # as ... ffn(db), ffn(rh), width(db), width(rh). With a single entry that
        # sequence thrashes: every one of the four misses, evicting the entry the
        # next one wants, and the parent pays FOUR statistics passes where two
        # would do. At P=1 this is `max_entries=1` exactly as before, so no
        # single-profile search changes.
        #
        # This does not widen the reuse boundary by one inch. The key still
        # carries the parent's artifact digest, so cross-parent reuse remains
        # impossible by construction — and `run()` additionally clears the cache
        # at each parent boundary, so the resident set is bounded by the profiles
        # of the parent being expanded rather than by eviction order.
        self.stats_cache = StatsCache(stats_spec=config.stats_spec,
                                      max_entries=max(1, len(config.profiles)))
        #: Operational timings. Never hashed, never returned into a state.
        self.telemetry = TelemetrySink(self.workdir / "telemetry.jsonl")

        #: WHICH MEASUREMENT PROTOCOL THIS SEARCH IS RESUMABLE UNDER.
        #:
        #: Taken from the config when a driver declares it — which also puts it
        #: in `config_hash` — and otherwise asked of the measurer, which knows
        #: its own protocol. Asking matters: `StateEvaluator` stamps every
        #: measurement with its protocol id, so a search that neither declared
        #: nor asked would write records it could never adopt again and would
        #: silently re-measure every state on resume. Resume must not depend on
        #: a driver remembering to pass a field.
        #:
        #: `None` only when the measurer declares none, which is the historical
        #: case; `measurement_is_comparable` then applies the historical rule.
        measurer_protocol = getattr(measurer, "measurement_protocol_id", None)
        declared = self.config.measurement_protocol_id
        if declared and measurer_protocol and str(measurer_protocol) != declared:
            raise SearchError(
                f"this search declares measurement protocol {declared[:12]} and "
                f"its measurer reports {str(measurer_protocol)[:12]}. Refused at "
                "construction rather than at the first measurement, because "
                "every state measured in between would have to be discarded.")
        #: ONE PARTITION FOR THE OPERATORS AND THE MEASURER, checked rather than
        #: assumed. The protocol id above is a hash and cannot be decoded, so
        #: agreeing on it proves nothing about the support when a driver declares
        #: neither. This is the hazard `position_policy` was given a run-level home
        #: to prevent, in its other half: candidates chosen by a Top-K objective
        #: and then pruned by a full-vocabulary beam metric would be one
        #: experiment in name and two in fact.
        #:
        #: A COURTESY, NOT THE GUARANTEE. Every driver in this repository wraps
        #: its evaluator in a `lambda model, digest: ...`, so this attribute is
        #: invisible on the usual path and this check usually does nothing. The
        #: load-bearing one is per measurement, in `_materialize_and_measure`,
        #: which reads what the measurer actually REDUCED OVER out of its own
        #: detail and therefore cannot be hidden by wrapping. This one exists
        #: because when it CAN see the measurer it fails before the first
        #: expensive measurement rather than after it.
        #:
        #: A measurer that exposes no support is historical and is taken to be
        #: full-vocabulary, which is what it was.
        measurer_support = getattr(measurer, "distribution_support", None)
        if measurer_support is not None:
            mine = self.config.distribution_support
            if measurer_support.support_id != mine.support_id:
                raise SearchError(
                    f"this search reduces over {mine.support_id} and its measurer "
                    f"reduces over {measurer_support.support_id}. The operators "
                    "would choose candidates under one partition and the beam "
                    "would prune them under another. Refused at construction.")
        self.measurement_protocol_id = (
            declared or (str(measurer_protocol) if measurer_protocol else None))

        adapter.validate_target(config.target_spec)
        self._validate_impl_profiles()

    # --- setup -------------------------------------------------------------

    def _validate_impl_profiles(self) -> None:
        """Refuse a restriction that does not describe this search.

        Checked here, before the teacher is loaded, because every way of getting
        this wrong is silent at runtime: a typo'd impl_id restricts nothing and
        the search quietly runs the factorial it was configured to avoid; a
        profile id that is not active names a mixture no expansion could pick;
        an empty list removes a kind from the space without removing it from
        `allowed_impls`, so the target becomes unreachable several expensive
        levels later instead of now.
        """
        if not self.config.impl_profiles:
            return
        active = {p.qualified_id for p in self.config.profiles}
        allowed = set(self._allowed_impl_ids())
        for impl_id, ids in sorted(self.config.impl_profiles.items()):
            if impl_id not in allowed:
                raise SearchError(
                    f"impl_profiles restricts {impl_id!r}, which this search "
                    f"cannot run; allowed implementations are {sorted(allowed)}")
            impl = get_implementation(impl_id)
            if not consumes_calibration(impl):
                raise SearchError(
                    f"impl_profiles restricts {impl_id!r}, which declares "
                    f"{impl.calibration.value!r} and is offered exactly once "
                    "against the no-calibration sentinel however many profiles "
                    "are active. A restriction on it would be ignored, and a "
                    "configuration whose stated space is not its real space is "
                    "the thing this check exists to refuse.")
            if not ids:
                raise SearchError(
                    f"impl_profiles gives {impl_id!r} no profile at all. That "
                    "silently removes a kind from the space while it stays in "
                    "allowed_impls; drop it from allowed_impls instead, so the "
                    "target-reachability check can see it.")
            unknown = sorted(set(ids) - active)
            if unknown:
                raise SearchError(
                    f"impl_profiles gives {impl_id!r} profiles {unknown} that "
                    f"this search does not branch over; active are "
                    f"{sorted(active)}")

    def root_model(self) -> Any:
        if self._root_model is None:
            self._root_model = self.root_loader()
        return self._root_model

    def root_state(self) -> InitializationState:
        if self._root_spec is None:
            self._root_spec = self.adapter.spec_of(self.root_model())
        spec = self._root_spec
        if spec.family != self.config.target_spec.family:
            raise SearchError(
                f"teacher family {spec.family!r} and target family "
                f"{self.config.target_spec.family!r} differ; one adapter cannot span them")
        self._assert_target_reachable(spec)
        #: STAMPED HERE, not in `run()`, so every caller of `root_state()` gets
        #: a root that carries its lineage. A root built without one would make
        #: `_materialization_for` refuse its first child, which is the correct
        #: failure but a confusing place to meet it.
        root = make_root_state(
            root_teacher_id=self.root_teacher_id,
            root_teacher_sha256=self.root_teacher_sha256,
            spec=spec, target_spec=self.config.target_spec,
            num_parameters=self.adapter.param_count(spec), seed=self.config.seed)
        root.materialization = self._root_materialization()
        return root

    def _assert_target_reachable(self, root_spec: ArchSpec) -> None:
        """Every field the target changes must be some implementation's business.

        Checked once, before anything expensive: a target differing in
        ``vocab_size`` has no operator that could ever close the gap, and the
        useful moment to learn that is now, not after a beam of dead ends.
        """
        differing = root_spec.diff(self.config.target_spec)
        coverable: set[str] = set()
        for impl_id in self._allowed_impl_ids():
            coverable |= get_implementation(impl_id).modifies
        orphan = sorted(differing - coverable)
        if orphan:
            raise SearchError(
                f"target differs from the teacher in {orphan}, and no registered "
                "implementation modifies those fields; the target is unreachable")

    def _allowed_impl_ids(self) -> list[str]:
        from aadistill.initialization.operators.base import registered_implementations

        if self.config.allowed_impls is None:
            return registered_implementations()
        unknown = sorted(set(self.config.allowed_impls) - set(registered_implementations()))
        if unknown:
            raise SearchError(f"allowed_impls names unregistered implementations {unknown}")
        return sorted(self.config.allowed_impls)

    def calibration_for(self, profile: CalibrationProfile) -> Sequence[Mapping[str, Any]]:
        # The sentinel describes no mixture — `NO_CALIBRATION.resolve()` raises by
        # design — so asking the loader for its items is asking for something that
        # cannot exist. Every loader written so far has been `lambda profile:
        # items`, which ignores its argument and therefore answered anyway; the
        # first loader that actually dispatched on the profile raised KeyError
        # inside the beam. A correct loader delegating to
        # `profile.resolve()` would have raised CalibrationError there instead —
        # on a paid pod, mid-search.
        #
        # Identity is untouched: `n_calibration_items` is the only thing derived
        # from this list and `_expand_one` explicitly excludes it from
        # `config_hash`, while both `CalibrationNeed.NONE` implementations ignore
        # `config` entirely. So no state id, and no recorded state, moves.
        if profile.is_no_calibration:
            return ()
        key = profile.qualified_id
        if key not in self._calibration_cache:
            self._calibration_cache[key] = list(self.calibration_loader(profile))
        return self._calibration_cache[key]

    # --- the mandatory cycle ----------------------------------------------

    def _materialize_and_measure(self, state: InitializationState, model: Any,
                                 planned_spec: ArchSpec) -> None:
        """materialize -> canonical reload -> hash -> validate -> measure.

        Every step is required and none may be inherited. The reload is not
        ceremony: the thing that trains, and the thing every downstream stage
        loads, is the file — so the file is what gets hashed and what gets
        measured, not the in-memory object that wrote it.
        """
        ckpt_dir = self.checkpoint_dir(state)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with self.telemetry.timed("materialize_seconds"):
            self.adapter.save(model, str(ckpt_dir),
                              max_shard_size=self.config.max_shard_size)

        with self.telemetry.timed("identify_seconds"):
            artifact = identify_checkpoint(
                ckpt_dir, adapter=self.adapter, spec=planned_spec,
                num_parameters=self.adapter.param_count(planned_spec))
        state.mark_materialized(artifact)

        # The reload is validated on the PRODUCED model's device, then moved to
        # the search device to be measured.
        #
        # A paid search died here. `_validate` forwards both models through
        # one input, and that input used to be built on `config.device`. The
        # reload is placed there too, but the produced child is whatever the
        # operator built — `ChildBuilder` calls `build_student`, which sets the
        # dtype and deliberately does NOT place the model, so on a GPU run the
        # child is on the host and the probe indexed CPU embedding weights with
        # a CUDA index. Every zero-cost run passes `device="cpu"`, where the two
        # coincide and the mismatch cannot appear.
        #
        # Validating on the produced model's device rather than moving the child
        # keeps the comparison on ONE numerical backend — a save/reload check
        # that compared a host forward against a device forward would be
        # measuring the backend, not the serialization — and it leaves the
        # memory model alone: the child is not duplicated into VRAM to be
        # checked, only the canonical reload goes there, and only after it has
        # been validated.
        validation_device = model_device(model)
        with self.telemetry.timed("canonical_reload_seconds"):
            reloaded = self.adapter.load(str(ckpt_dir), device=validation_device)
        with self.telemetry.timed("validation_seconds"):
            checks = self._validate(state, model, reloaded, planned_spec,
                                    device=validation_device)
        checks["n_shards"] = len(artifact.shards)
        checks["sharded"] = artifact.is_sharded
        checks["validation_device"] = str(validation_device)
        checks["measurement_device"] = str(self.config.device)
        state.notes["validation"] = checks
        state.mark_validated()

        # The canonical reload — the file, not the in-memory object that wrote
        # it — is what gets measured, on the search device.
        reloaded = reloaded.to(self.config.device)
        with self.telemetry.timed("state_evaluation_seconds"):
            evaluation = self.measurer(reloaded, artifact.artifact_digest)
        #: THE MEASURER IS INJECTED, so this is the only place the search can
        #: check that it scores the positions the search's operators protected.
        #: `measurer` is an opaque callable — the application builds the
        #: evaluator — and the whole point of a run-level policy is that an
        #: operator and the beam metric cannot disagree. Without this they could:
        #: a driver that passed the policy to `SearchConfig` and forgot the
        #: evaluator would run target-aware operators and prune them on a
        #: full-sequence KL — selecting on one objective and pruning on
        #: another.
        #:
        #: Absent means the incumbent policy, for the same reason `_restore`
        #: reads it that way: every evaluation written before the policy existed
        #: measured every position.
        measured_policy = (evaluation.detail or {}).get(
            "position_policy_hash", ALL_POSITIONS_V1.policy_hash)
        if measured_policy != self.config.position_policy.policy_hash:
            state.mark_invalid(
                f"the measurer scored position policy {str(measured_policy)[:12]} "
                f"but this search runs {self.config.position_policy.qualified_id} "
                f"({self.config.position_policy.policy_hash[:12]})")
            raise SearchError(
                f"{state.state_id}: {state.invalid_reason}. The operators and the "
                "beam metric must consume ONE policy; a candidate selected on "
                "supervised positions and pruned on all of them is two "
                "experiments reported as one.")
        #: AND THE SAME VOCABULARY PARTITION, per measurement. The construction
        #: check above asks the measurer what support it holds; this reads what it
        #: actually REDUCED OVER, which is the thing a candidate was pruned on. A
        #: measurer whose support changed mid-run -- a second evaluator, a rebuilt
        #: one, a cache restored from another protocol -- passes construction and
        #: fails here.
        #:
        #: Absent means the full vocabulary, for the same reason the policy check
        #: reads an absent policy as the incumbent: every evaluation written before
        #: this field existed reduced over every entry. `ReductionSemantics`
        #: serializes the key only under a coarsened partition.
        measured_support = ((evaluation.detail or {}).get("reduction") or {}).get(
            "distribution_support") or {"support_id": FULL_VOCAB_V1.support_id}
        expected_support = self.config.distribution_support.as_dict()
        if measured_support != expected_support:
            state.mark_invalid(
                f"the measurer reduced over {measured_support} but this search "
                f"runs {expected_support}")
            raise SearchError(
                f"{state.state_id}: {state.invalid_reason}. The operators and the "
                "beam metric must reduce over ONE partition; a candidate selected "
                "on a Top-K objective and pruned on a full-vocabulary metric is "
                "two experiments reported as one.")
        #: AND THE WHOLE PROTOCOL, not only the policy. The check above catches
        #: a measurer scoring other POSITIONS; this catches the rest of the list
        #: — a measurer whose suite content, scoring content, reduction semantics
        #: or execution is not the one the rest of this search measured under.
        self._bind_measurement_protocol(state,
                                        (evaluation.detail or {}).get(PROTOCOL_FIELD))
        state.attach_evaluation(evaluation)
        del reloaded

    def _validate(self, state: InitializationState, produced: Any, reloaded: Any,
                  planned_spec: ArchSpec, *, device: Any = None) -> dict[str, Any]:
        actual = self.adapter.spec_of(reloaded)
        if not actual.matches(planned_spec):
            state.mark_invalid(
                f"reloaded checkpoint is {actual.describe()}, planned {planned_spec.describe()}")
            raise SearchError(state.invalid_reason)
        expected_params = self.adapter.param_count(actual)
        real_params = sum(p.numel() for p in reloaded.parameters())
        if real_params != expected_params:
            state.mark_invalid(
                f"parameter count {real_params:,} != {expected_params:,} implied by the spec")
            raise SearchError(state.invalid_reason)

        with torch.no_grad():
            # On the device the two models being compared are actually on, which
            # is the produced model's. NOT `config.device`: see the note in
            # `_materialize_and_measure`.
            ids = torch.tensor([[1, 2, 3, 4, 5]],
                               device=device if device is not None
                               else model_device(produced))
            a = produced(ids).logits
            b = reloaded(ids).logits
        if not torch.isfinite(b).all():
            state.mark_invalid("reloaded checkpoint produced non-finite logits")
            raise SearchError(state.invalid_reason)
        max_diff = float((a.float() - b.float()).abs().max())
        return {"spec_hash": actual.spec_hash, "num_parameters": real_params,
                "reload_max_logit_diff": max_diff, "finite": True}

    # --- expansion ---------------------------------------------------------

    def _candidate_expansions(self, parent: InitializationState):
        """(implementation, profile) pairs for a parent, in deterministic order.

        An implementation declaring ``CalibrationNeed.NONE`` is offered **once**,
        against the canonical no-calibration sentinel, however many profiles are
        active. ``depth.positional_v0`` is a fixed positional heuristic and
        ``attention.weight_proxy_v0`` scores weights; neither has a mechanism by
        which a mixture could change its output, so branching them over profiles
        would manufacture byte-identical states, occupy beam slots that distinct
        hypotheses should hold, and inflate the search-space count by a factor
        that means nothing.

        An implementation that DOES consume calibration is offered every active
        profile, unless ``config.impl_profiles`` restricts it — see that field.
        """
        exclude = () if self.config.allow_kind_repeat else tuple(sorted(set(parent.applied_kinds)))
        options = applicable_implementations(
            self.adapter, parent.spec, self.config.target_spec,
            exclude_kinds=exclude, allow_impls=self._allowed_impl_ids())
        for impl, _ in sorted(options, key=lambda pair: pair[0].impl_id):
            for profile in self.profiles_for(impl):
                yield impl, profile

    def profiles_for(self, impl: OperatorImplementation) -> list[CalibrationProfile]:
        """This search's branching for one implementation."""
        return expansion_profiles(
            impl, self.config.profiles, self.config.impl_profiles)

    def _expand_one(self, parent: InitializationState, impl: OperatorImplementation,
                    profile: CalibrationProfile) -> InitializationState:
        operator_config = {"n_calibration_items": len(self.calibration_for(profile)),
                           **self._position_policy_config(impl, profile),
                           **self._distribution_support_config()}
        plan = impl.plan(parent.spec, self.config.target_spec, self.adapter, operator_config)
        config_hash = sha256_json(
            {k: v for k, v in operator_config.items() if k != "n_calibration_items"})

        step = OperatorStep(
            index=len(parent.steps), kind=impl.kind, impl_id=impl.impl_id,
            impl_signature_hash=impl.signature_hash, profile_id=profile.qualified_id,
            profile_hash=profile.profile_hash, config_hash=config_hash,
            seed=self.config.seed, result_spec_hash=plan.result_spec.spec_hash)
        state = child_state(parent, step, plan.result_spec,
                            self.adapter.param_count(plan.result_spec), self.config.seed)
        state.materialization = self._materialization_for(state.state_id, parent)

        restored = self._restore(state)
        if restored is not None:
            return restored

        load_started = time.perf_counter()
        parent_model = self._load_state_model(parent)
        parent_load_seconds = time.perf_counter() - load_started
        stats_before = (self.stats_cache.hits, self.stats_cache.misses)
        ctx = OperatorContext(
            adapter=self.adapter, model=parent_model, parent_spec=parent.spec,
            target_spec=self.config.target_spec, profile=profile,
            calibration_items=self.calibration_for(profile), seed=self.config.seed,
            device=self.config.device, workdir=self.workdir,
            config=dict(operator_config),
            stats_cache=self.stats_cache,
            stats_cache_key=self._stats_key(parent, profile),
            execution=self.execution,
            position_policy=self.config.position_policy,
            #: THE OBJECT, whose declaration went into `operator_config` above.
            #: `OperatorImplementation.execute` refuses a disagreement between the
            #: two, so passing one without the other fails closed.
            distribution_support=self.config.distribution_support,
            deadline=self.deadline)

        # Before the expansion, so a budget already spent does not buy one more
        # hour-long operator; and the operator itself checks *inside* its own
        # loop, which is the granularity that actually bounds the cost.
        if self.deadline is not None:
            self.deadline.check(f"before {impl.impl_id} on {parent.spec.spec_hash[:12]}")

        started = time.time()
        outcome = impl.execute(ctx)
        elapsed = time.time() - started

        state.steps = (*parent.steps, replace(
            step, local_metrics=outcome.local_metrics, trace=dict(outcome.trace),
            artifacts=dict(outcome.artifacts), wall_seconds=elapsed))
        self._materialize_and_measure(state, outcome.model, plan.result_spec)
        # Emitted AFTER materialization so one record covers the whole expansion.
        # `drain_phases` empties the accumulators the materialize path filled, so
        # each record describes its own expansion and nothing accumulates across
        # them. None of this reaches `state`.
        self.telemetry.record(
            "expansion",
            state_id=state.state_id, parent_id=parent.state_id,
            impl_id=impl.impl_id, profile=profile.qualified_id,
            parent_load_seconds=round(parent_load_seconds, 4),
            operator_seconds=round(elapsed, 4),
            stats_cache={"hits": self.stats_cache.hits - stats_before[0],
                         "misses": self.stats_cache.misses - stats_before[1],
                         "profile": profile.qualified_id},
            operator_timing=dict(outcome.artifacts.get("timing", {})),
            reference_cache=dict(outcome.artifacts.get("reference_cache", {})),
            reference_counters=dict(outcome.artifacts.get("reference_counters", {})),
            **self.telemetry.drain_phases())
        del outcome
        self.store.append(state)
        return state

    def _position_policy_config(self, impl: OperatorImplementation,
                                profile: CalibrationProfile) -> dict[str, Any]:
        """The scoring-position policy AND its content, as HASHED operator config.

        In `operator_config` rather than in `ctx.execution` because it changes
        what is computed: two searches whose operators protect different
        positions reach different leaves and must not share a state id. It is
        therefore read by `config_hash`, which is what forks the whole subtree.

        **The policy id alone was not enough**, and that is the gap this closes.
        A policy that reads positions consumes metadata no other identity
        covers: `profile_hash` pins the profile's SPEC, which pins one
        `content_sha256`, which hashes only item ids and token ids. Two assets
        with identical tokens and different supervised masks therefore agreed on
        every term above — the policy hash included — while producing different
        operator decisions. `scoring_content_config` binds what the policy
        actually reads, so the mask is part of the scientific path identity.

        **Omitted at the incumbent policy**, and omitted for an implementation
        that consumes no calibration data. Both omissions preserve a recorded
        identity rather than tidying one away: every committed state hashed an
        operator config of `{}`, and a weight-only operator has no mechanism by
        which a position policy could change its output — branching it would
        manufacture byte-identical states, which is the same argument
        `profile_for` already makes about the no-calibration sentinel.
        """
        if not consumes_calibration(impl):
            return {}
        policy = self.config.position_policy
        named = policy_config(policy)
        if not named:
            return {}
        items = self.calibration_for(profile)
        return {**named, **scoring_content_config(items, policy)}

    def _distribution_support_config(self) -> dict[str, Any]:
        """The vocabulary partition as HASHED operator config, or `{}`.

        In `operator_config` rather than in `ctx.execution` for the same reason as
        the position policy: it changes what is computed, so two searches whose
        operators reduce over different partitions reach different leaves and must
        not share a `config_hash`.

        **Omitted at the full vocabulary**, which preserves a recorded identity
        rather than tidying one away: every committed state hashed an operator
        config without this key, and emitting it now -- even as
        `{"support": "full_vocab_v1"}` -- would change 785 historical
        `measurement_protocol_id`s and every `config_hash` beside them.

        Not restricted to calibrated operators, unlike the position policy. A
        support reaches an operator through `OperatorContext` whether or not that
        operator consumes calibration items, and an operator that ignores it is
        free to; what must not happen is a declaration that disagrees with the
        object, which `execute` refuses.
        """
        support = self.config.distribution_support
        if support.is_full_vocab:
            return {}
        return {"distribution_support": support.as_dict()}

    def _materialization_for(self, semantic_state_id: str,
                             parent: InitializationState,
                             ) -> MaterializationIdentity | None:
        """The state's identity, when the caller declared its numerics.

        `None` otherwise, which is the historical behaviour and is why this is
        not a hard requirement: a toy construction that has not stated a device
        class or a compute dtype has nothing honest to fingerprint, and inventing
        one would be worse than having none.

        **The parent's materialization is REQUIRED when numerics are declared**,
        because the child's bytes are a function of the bytes it consumed. A
        parent that somehow lacks one under a fingerprinted run is a gap in the
        lineage, not something to paper over with a placeholder: every state in
        such a run gets its identity here or at the root, so the only way to
        reach this refusal is a bug.
        """
        if self.numerics is None:
            return None
        if parent.materialization is None:
            raise SearchError(
                f"{parent.state_id} carries no materialization identity, so the "
                f"child of it cannot bind the bytes it consumed. This run "
                "declared its numerics, so every state in it should have one — "
                "including the root, which takes the teacher's pinned identity")
        return MaterializationIdentity.build(
            semantic_state_id=semantic_state_id, execution=self.execution,
            environment=self.numerics,
            parent_materialization_id=parent.materialization.materialization_id)

    def _root_materialization(self) -> MaterializationIdentity | None:
        """The root's identity: the teacher's published revision, not an
        execution. See `MaterializationIdentity.root`."""
        if self.numerics is None:
            return None
        return MaterializationIdentity.root(
            semantic_state_id=compute_state_id(
                self.root_teacher_sha256, self.config.target_spec.spec_hash, ()),
            root_teacher_id=self.root_teacher_id,
            root_teacher_sha256=self.root_teacher_sha256)

    def checkpoint_dir(self, state: InitializationState) -> Path:
        """Where a state's weights live. ONE owner for the layout.

        Materialization-keyed, because two materializations of one semantic
        state are two different sets of bytes and a shared destination means
        the second silently overwrites the first:

            states/<semantic_state_id>/<materialization_id>/

        The semantic id stays the outer level deliberately. It is still the
        scientific coordinate — every materialization of one hypothesis sits
        together, and a reader looking for "that path's checkpoints" finds them
        in one place rather than scattered by protocol.

        A run that declared no numerics keeps the historical flat layout. It has
        no materialization to key on, and relocating its checkpoints would move
        paths that committed records already name.
        """
        base = Path(self.workdir) / "states" / state.state_id
        if state.materialization is None:
            return base
        return base / state.materialization.materialization_id

    def _stats_key(self, parent: InitializationState,
                   profile: CalibrationProfile) -> str | None:
        """Cache key for an activation pass on ``parent`` under ``profile``.

        Returns ``None`` for the root, whose identity is a published revision
        rather than an artifact this search computed — sharing a pass across the
        root's children is legitimate, but it needs a digest to key on, and
        inventing one would be exactly the cross-parent reuse the key exists to
        prevent. The root simply pays for its own passes.
        """
        if parent.artifact_digest is None:
            return None
        return stats_cache_key(
            parent_artifact_digest=parent.artifact_digest,
            profile_hash=profile.profile_hash,
            stats_spec=self.config.stats_spec,
            adapter_version=self.adapter.adapter_version,
            #: The key's own docstring promises "numerical configuration (device,
            #: accumulation dtype, BATCH RULE)" and the batch rule was missing.
            #: It matters for exactly the reason A3 measured: a statistics pass
            #: taken at one batch size and packing is not the pass another would
            #: have produced, so handing one to the other is the same collision
            #: as resuming across protocols. And the scoring positions decide
            #: which tokens entered the accumulators at all, so a cache shared
            #: across policies would hand a target-aware operator the
            #: full-sequence statistic.
            numerical_config={"device": self.config.device,
                              "accumulation": self.config.stats_spec.accumulation_dtype,
                              **self.execution.as_fingerprint(),
                              "position_policy_hash":
                                  self.config.position_policy.policy_hash})

    def _load_state_model(self, state: InitializationState) -> Any:
        if state.parent_id is None:
            return self.root_model()
        if not state.checkpoint_path:
            raise SearchError(
                f"{state.state_id} has no materialized checkpoint to expand from; its "
                "weights were pruned before it was expanded")
        return self.adapter.load(state.checkpoint_path, device=self.config.device)

    def _restore(self, state: InitializationState) -> InitializationState | None:
        """Rehydrate a state the journal already carries a full measurement for."""
        #: LOOKED UP BY MATERIALIZATION when this run has one, and that is the
        #: whole fix rather than a refinement of the refusal below.
        #:
        #: Refusing a mismatch is necessary and was not sufficient: keyed on the
        #: semantic state, the journal can only ever offer the NEWEST record for
        #: a path, so a run whose own materialization was journalled before
        #: another protocol's would be told "that record is not yours" and
        #: rebuild work it already had. Keyed on the materialization, an earlier
        #: record is as findable as a later one and a hit matches by
        #: construction.
        if state.materialization is not None:
            record = self._journal_by_materialization.get(
                state.materialization.materialization_id)
        else:
            record = self._journal.get(state.state_id)
        if record is None or record.get("validity") != StateValidity.MEASURED.value:
            return None
        #: THE A3 COLLISION, still refused — now as a structural assertion
        #: rather than the mechanism. A keyed hit cannot mismatch, so this fires
        #: only if the index and the record disagree, which would be a defect in
        #: `latest_by_materialization_id`. Kept because a lookup whose key is
        #: trusted silently is a lookup nobody checks: one semantic id covered
        #: two artifacts in this project, and the cost of re-asserting it here
        #: is one comparison per restored state.
        if state.materialization is not None:
            state.materialization.require_same_materialization(
                record.get("materialization") or {},
                what=f"journal record for {state.state_id}")
        elif record.get("materialization"):
            #: The journal declared a materialization and this run has not. The
            #: run cannot show the record describes its own execution, so it does
            #: not get to inherit it. Declining rather than raising: a protocol
            #: change is a legitimate new run and it should simply rebuild.
            return None
        path = record.get("checkpoint_path")
        if not path or not Path(path).is_dir():
            return None
        artifact_record = record.get("artifact")
        if not artifact_record:
            return None
        # Re-derive the identity from the files on disk rather than trusting the
        # journal's copy: resume is exactly the moment a stale or truncated
        # checkpoint would otherwise be adopted along with its old metrics.
        try:
            artifact = identify_checkpoint(
                path, adapter=self.adapter, spec=state.spec,
                num_parameters=self.adapter.param_count(state.spec))
        except Exception:
            return None
        if artifact.artifact_digest != artifact_record.get("artifact_digest"):
            return None
        eval_record = record.get("evaluation") or {}
        #: ONE QUESTION, and it replaced two. A state's identity is its path,
        #: which does not include the suite, the scoring positions, the
        #: reduction or the execution — so a journal written under any of those
        #: differing would otherwise be adopted wholesale and the beam would
        #: rank this run's states on another run's questions.
        #:
        #: This used to be an inline suite-hash comparison, then an inline
        #: policy-hash comparison beside it, and the next three terms would have
        #: been three more. `measurement_is_comparable` owns the rule, including
        #: how a record that predates the identity is judged, and it returns the
        #: reason so a declined resume is distinguishable from an empty journal.
        comparable, why = measurement_is_comparable(
            {"detail": eval_record.get("detail") or {},
             "suite_hash": eval_record.get("suite_hash")},
            protocol_id=self.measurement_protocol_id,
            historical_suite_hash=self.config.suite.suite_hash,
            historical_policy_hash=self.config.position_policy.policy_hash)
        if not comparable:
            self.telemetry.record("restore_declined", state_id=state.state_id,
                                  reason=why)
            return None
        state.mark_materialized(artifact)
        state.validity = StateValidity.VALIDATED
        state.attach_evaluation(StateEvaluation(
            artifact_digest=eval_record["artifact_digest"],
            suite_id=eval_record["suite_id"], suite_hash=eval_record["suite_hash"],
            reference=eval_record["reference"], values=eval_record["values"],
            positions=eval_record["positions"], detail=eval_record.get("detail", {}),
            measured_utc=eval_record.get("measured_utc"),
            runtime=eval_record.get("runtime", {})))
        state.notes["resumed"] = True
        self.resumed_ids.add(state.state_id)
        return state

    # --- the loop ----------------------------------------------------------

    def _bind_measurement_protocol(self, state: InitializationState,
                                   measured: str | None) -> None:
        """Hold every measurement in this run to ONE protocol.

        Two directions, because the protocol can be known in either order:

        * the search already knows one — from `SearchConfig`, or adopted below —
          and the measurement must agree, or the run is mixing measurements of
          different quantities and stops.
        * the search knows none and the measurement carries one: ADOPT it.

        Adoption is what makes the mechanism work without a driver remembering
        anything. Every driver wraps its evaluator in a lambda, so the attribute
        the constructor looks for is invisible, and a search that asked for a
        declaration and got none would write records stamped with a protocol it
        could never match again — resume would decline every state, silently,
        and a nine-hour search would re-measure everything it had already done.
        Learning from the first measurement costs at most ONE re-measured state
        on a resumed run that declared nothing, and that one is recorded.

        Declaring `measurement_protocol_id` on the config is still better: it
        enters `config_hash`, so the run's own identity states what it measured
        under, and it is checked before the first expensive measurement rather
        than at it.
        """
        measured = str(measured) if measured else None
        declared = self.measurement_protocol_id
        if declared and measured != declared:
            state.mark_invalid(
                f"the measurer reports protocol {str(measured)[:12]} and the "
                f"rest of this search measured under {declared[:12]}")
            raise SearchError(
                f"{state.state_id}: {state.invalid_reason}. A measurement taken "
                "under another protocol is a measurement of another quantity; "
                "ranking them together would compare two experiments as one.")
        if declared is None and measured:
            self.measurement_protocol_id = measured
            self.telemetry.record("measurement_protocol_adopted",
                                  state_id=state.state_id, protocol_id=measured)
            print(f"measurement protocol {measured[:12]} adopted from the first "
                  f"measurement ({state.state_id[:12]}); declare it on "
                  "SearchConfig to have it checked before the first expansion "
                  "and recorded in config_hash")

    def run(self) -> "SearchResult":
        self._journal = self.store.latest_by_state_id()
        #: THE RESUME VIEW, keyed on the materialization. The semantic view
        #: above is kept for records written before the field existed and for
        #: the frozen canonical-record rule that reads it; neither can find an
        #: EARLIER materialization of a path whose newest record belongs to
        #: another protocol, which is what this one is for.
        self._journal_by_materialization = (
            self.store.latest_by_materialization_id())
        root = self.root_state()
        self.states[root.state_id] = root
        beam: list[InitializationState] = [root]
        level = 0
        # One operator closes one structural difference, so the number of
        # differing fields *is* the path length. Deriving it beats a constant:
        # a 30B -> 4.xB target that also changes KV heads simply needs one more
        # level, with no config edit and no silently truncated search.
        max_depth = self.config.max_depth or len(root.remaining_differences())

        while beam and level < max_depth:
            started = time.time()
            parents = list(beam)
            generated: list[InitializationState] = []
            dead_ends: list[dict[str, str]] = []

            for parent in parents:
                # The resident statistics belong to the PREVIOUS parent and can
                # never be hit again — the key carries the parent's artifact
                # digest — so holding them only costs ~1.8 GiB each. Dropping
                # them here bounds the resident set to the parent being expanded.
                self.stats_cache.clear()
                expansions = list(self._candidate_expansions(parent))
                if not expansions and not parent.is_complete_leaf():
                    exclude = () if self.config.allow_kind_repeat else tuple(
                        sorted(set(parent.applied_kinds)))
                    dead_ends.append({
                        "state_id": parent.state_id, "path": parent.path_label,
                        "remaining": ",".join(sorted(parent.remaining_differences())),
                        "rejections": json.dumps(rejected_implementations(
                            self.adapter, parent.spec, self.config.target_spec,
                            exclude_kinds=exclude)),
                    })
                for impl, profile in expansions:
                    child = self._expand_one(parent, impl, profile)
                    self.states[child.state_id] = child
                    generated.append(child)

            complete = [s for s in generated if s.is_complete_leaf()]
            partial = [s for s in generated if not s.is_complete_leaf()]
            for leaf in complete:
                self.leaves.append(leaf)

            ranking = None
            if partial:
                # `None` width means a warmup level: retain every child of the
                # root so that no structural hypothesis dies on one step-0
                # measurement. The distinction is carried into the record, so a
                # level that pruned nothing by design is not confused with one
                # that happened not to.
                width = self.config.schedule.width_at(level)
                ranking = self.config.policy.rank(partial, width)
                kept = set(ranking.selected_ids)
                for state in partial:
                    if state.state_id not in kept:
                        reason = next((d["reason"] for d in ranking.decisions
                                       if d["state_id"] == state.state_id), "pruned")
                        state.mark_pruned(reason)
                        self._release_weights(state)
                    self.store.append(state)
                beam = list(ranking.selected)
            else:
                beam = []

            self.levels.append(LevelRecord(
                level=level,
                expanded_from=tuple(s.state_id for s in parents),
                generated=tuple(s.state_id for s in generated),
                ranking=ranking, leaves=tuple(s.state_id for s in complete),
                dead_ends=tuple(dead_ends), seconds=time.time() - started))
            level += 1

        return SearchResult(
            config=self.config, states=dict(self.states), levels=list(self.levels),
            leaves=list(self.leaves), resumed=sorted(self.resumed_ids),
            finished_utc=datetime.now(timezone.utc).isoformat())

    def _release_weights(self, state: InitializationState) -> None:
        """Drop a pruned state's bytes; keep everything that makes it auditable."""
        if not self.config.prune_weights or not state.checkpoint_path:
            return
        path = Path(state.checkpoint_path)
        if not path.is_dir():
            return
        freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        shutil.rmtree(path)
        state.notes["weights_released"] = {
            "bytes": freed, "artifact_digest": state.artifact_digest,
            "single_shard_sha256": state.checkpoint_sha256,
            "n_shards": len(state.artifact.shards) if state.artifact else 0,
            "reason": "pruned from the beam; metrics and artifact identity retained",
        }
        state.checkpoint_path = None


@dataclass
class SearchResult:
    config: SearchConfig
    states: dict[str, InitializationState]
    levels: list[LevelRecord]
    leaves: list[InitializationState]
    resumed: list[str]
    finished_utc: str

    @property
    def complete_leaves(self) -> list[InitializationState]:
        return [s for s in self.leaves if s.is_complete_leaf()]

    def top_n(self, policy: BeamRankingPolicy, n: int) -> RankingResult:
        """Rank the complete target-size leaves. Intermediates cannot appear here.

        The guard is not decorative: ``require_recovery_admissible`` is what stops
        a 3.2B depth-only intermediate — which will often score *better* on
        teacher KL than any fully compressed leaf — from being promoted into a
        recovery probe it could never be a candidate for.
        """
        # Deliberately iterates `self.leaves` rather than the filtered
        # `complete_leaves`: silently dropping an inadmissible candidate would
        # hide the upstream bug that put it there, and the whole point is that
        # this boundary is loud.
        for leaf in self.leaves:
            leaf.require_recovery_admissible()
        return policy.rank(self.leaves, n)

    def summary(self) -> dict[str, Any]:
        pruned = [s for s in self.states.values() if s.validity is StateValidity.PRUNED]
        return {
            "run_id": self.config.run_id,
            "config_hash": self.config.config_hash,
            "n_states": len(self.states),
            "n_levels": len(self.levels),
            "n_complete_leaves": len(self.complete_leaves),
            "n_pruned": len(pruned),
            "n_resumed": len(self.resumed),
            "finished_utc": self.finished_utc,
        }
