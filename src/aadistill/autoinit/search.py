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

from ..infrastructure.manifest import sha256_file, sha256_json
from .arch import ArchitectureAdapter, ArchSpec
from .calibration import CalibrationProfile
from .metrics import StateEvalSuite, StateEvaluation
from .operators.base import (
    OperatorContext,
    OperatorImplementation,
    applicable_implementations,
    get_implementation,
    rejected_implementations,
)
from .ranking import BeamRankingPolicy, RankingResult
from .state import (
    InitializationState,
    OperatorStep,
    StateStore,
    StateValidity,
    child_state,
    make_root_state,
)


class SearchError(RuntimeError):
    """The search cannot proceed as configured."""


CalibrationLoader = Callable[[CalibrationProfile], Sequence[Mapping[str, Any]]]
Measurer = Callable[[Any, str], StateEvaluation]


@dataclass
class SearchConfig:
    """Everything that fixes a search run, and therefore everything that hashes."""

    run_id: str
    target_spec: ArchSpec
    beam_width: int
    seed: int
    workdir: Path
    profiles: tuple[CalibrationProfile, ...]
    policy: BeamRankingPolicy
    suite: StateEvalSuite
    allowed_impls: tuple[str, ...] | None = None
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
            "beam_width": self.beam_width,
            "seed": self.seed,
            "profiles": [p.qualified_id for p in self.profiles],
            "profile_hashes": {p.qualified_id: p.profile_hash for p in self.profiles},
            "policy": self.policy.qualified_id,
            "policy_hash": self.policy.policy_hash,
            "suite": self.suite.qualified_id,
            "suite_hash": self.suite.suite_hash,
            "allowed_impls": list(self.allowed_impls) if self.allowed_impls else None,
            "max_depth": self.max_depth,
            "allow_kind_repeat": self.allow_kind_repeat,
            "device": self.device,
            "prune_weights": self.prune_weights,
            "keep_leaf_weights": self.keep_leaf_weights,
            "notes": dict(self.notes),
        }

    @property
    def config_hash(self) -> str:
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
    ) -> None:
        self.adapter = adapter
        self.config = config
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

        adapter.validate_target(config.target_spec)

    # --- setup -------------------------------------------------------------

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
        return make_root_state(
            root_teacher_id=self.root_teacher_id,
            root_teacher_sha256=self.root_teacher_sha256,
            spec=spec, target_spec=self.config.target_spec,
            num_parameters=self.adapter.param_count(spec), seed=self.config.seed)

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
        from .operators.base import registered_implementations

        if self.config.allowed_impls is None:
            return registered_implementations()
        unknown = sorted(set(self.config.allowed_impls) - set(registered_implementations()))
        if unknown:
            raise SearchError(f"allowed_impls names unregistered implementations {unknown}")
        return sorted(self.config.allowed_impls)

    def calibration_for(self, profile: CalibrationProfile) -> Sequence[Mapping[str, Any]]:
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
        ckpt_dir = self.workdir / "states" / state.state_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.adapter.save(model, str(ckpt_dir))

        weight_path = ckpt_dir / self.adapter.weight_file(str(ckpt_dir))
        if not weight_path.is_file():
            raise SearchError(f"{state.state_id}: {weight_path} was not written")
        digest = sha256_file(weight_path)
        config_path = ckpt_dir / "config.json"
        config_hash = sha256_json(json.loads(config_path.read_text())) if config_path.is_file() else ""
        state.mark_materialized(str(ckpt_dir), digest, config_hash)

        reloaded = self.adapter.load(str(ckpt_dir), device=self.config.device)
        checks = self._validate(state, model, reloaded, planned_spec)
        state.notes["validation"] = checks
        state.mark_validated()

        evaluation = self.measurer(reloaded, digest)
        state.attach_evaluation(evaluation)
        del reloaded

    def _validate(self, state: InitializationState, produced: Any, reloaded: Any,
                  planned_spec: ArchSpec) -> dict[str, Any]:
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
            ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.config.device)
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
        """(implementation, profile) pairs for a parent, in deterministic order."""
        exclude = () if self.config.allow_kind_repeat else tuple(sorted(set(parent.applied_kinds)))
        options = applicable_implementations(
            self.adapter, parent.spec, self.config.target_spec,
            exclude_kinds=exclude, allow_impls=self._allowed_impl_ids())
        for impl, _ in sorted(options, key=lambda pair: pair[0].impl_id):
            for profile in sorted(self.config.profiles, key=lambda p: p.qualified_id):
                yield impl, profile

    def _expand_one(self, parent: InitializationState, impl: OperatorImplementation,
                    profile: CalibrationProfile) -> InitializationState:
        operator_config = {"n_calibration_items": len(self.calibration_for(profile))}
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

        restored = self._restore(state)
        if restored is not None:
            return restored

        parent_model = self._load_state_model(parent)
        ctx = OperatorContext(
            adapter=self.adapter, model=parent_model, parent_spec=parent.spec,
            target_spec=self.config.target_spec, profile=profile,
            calibration_items=self.calibration_for(profile), seed=self.config.seed,
            device=self.config.device, workdir=self.workdir,
            config=dict(operator_config))

        started = time.time()
        outcome = impl.execute(ctx)
        elapsed = time.time() - started

        state.steps = (*parent.steps, replace(
            step, local_metrics=outcome.local_metrics, trace=dict(outcome.trace),
            artifacts=dict(outcome.artifacts), wall_seconds=elapsed))
        self._materialize_and_measure(state, outcome.model, plan.result_spec)
        del outcome
        self.store.append(state)
        return state

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
        record = self._journal.get(state.state_id)
        if record is None or record.get("validity") != StateValidity.MEASURED.value:
            return None
        path = record.get("checkpoint_path")
        if not path or not Path(path).is_dir():
            return None
        state.checkpoint_path = path
        state.checkpoint_sha256 = record["checkpoint_sha256"]
        state.config_sha256 = record.get("config_sha256")
        state.validity = StateValidity.VALIDATED
        eval_record = record.get("evaluation") or {}
        state.attach_evaluation(StateEvaluation(
            checkpoint_sha256=eval_record["checkpoint_sha256"],
            suite_id=eval_record["suite_id"], suite_hash=eval_record["suite_hash"],
            reference=eval_record["reference"], values=eval_record["values"],
            positions=eval_record["positions"], detail=eval_record.get("detail", {}),
            measured_utc=eval_record.get("measured_utc"),
            runtime=eval_record.get("runtime", {})))
        state.notes["resumed"] = True
        self.resumed_ids.add(state.state_id)
        return state

    # --- the loop ----------------------------------------------------------

    def run(self) -> "SearchResult":
        self._journal = self.store.latest_by_state_id()
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
                ranking = self.config.policy.rank(partial, self.config.beam_width)
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
            "bytes": freed, "sha256": state.checkpoint_sha256,
            "reason": "pruned from the beam; metrics and hash retained",
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
