"""A SELECTED FORWARD BASELINE: the method new executions run by default.

A project accumulates methods, and at some point a maintainer selects one of
them as the default for forward work. This module turns such a selection into
the objects a session is built from, and refuses a record the core cannot
honour. It decides nothing: which method is selected, and why, is the record's
content and the caller's file.

Two records, two facts, both supplied by the caller:

* the BASELINE RECORD -- the method. One implementation per structural kind,
  the scoring-position policy, the distribution support, the operator-local and
  state-evaluation KL references, the execution profile and its declared
  numerical environment, the search schedule and ranking, and the identities of
  any artifact that instantiates the method;
* the SELECTION -- which record is in force for a role, bound by the record's
  sha256, so editing a record without a deliberate re-selection fails rather
  than silently changing what every forward session runs.

Every term is checked against what the core actually implements: an id the core
does not ship, or a hash that disagrees with the shipped object, is refused at
load rather than discovered inside a paid session.

**What a record cannot do is lend an artifact identity across execution
profiles.** A materialization's bytes are a function of its numerical execution
fingerprint (`specs.materialization`), not only of its semantic state. An
artifact produced under one profile is never the expected artifact of another,
even when the semantic state id is shared, so `reference_artifact_for` returns a
recorded artifact only for the fingerprint it was materialized under.

Nothing here names an experiment, a stage, a model, a checkpoint or a path.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aadistill.initialization.execution import ExecutionConfig, ExecutionError
from aadistill.initialization.planning.ranking import (
    PARETO_V1, SCHEDULE_V1, BeamRankingPolicy, BeamSchedule,
)
from aadistill.initialization.scoring.kl_reference import (
    DECLARABLE, KL_REFERENCE_KEY, PARENT_STATE,
)
from aadistill.initialization.scoring.positions import (
    ScoringPositionPolicy, get_position_policy,
)
from aadistill.initialization.scoring.support import DistributionSupport
from aadistill.initialization.specs.materialization import (
    NumericalEnvironment, numerical_execution_fingerprint,
)

SCHEMA = "aadistill.initialization.forward_baseline/v1"
SELECTION_SCHEMA = "aadistill.initialization.baseline_selection/v1"

#: The reference global state evaluation compares every state against: the
#: lineage root, i.e. the original model before any compression. It is the only
#: one `planning.metrics.StateEvaluator` implements, so it is the only one a
#: record may name -- a record claiming another would describe a computation no
#: session can perform.
STATE_EVALUATION_REFERENCE = "lineage_root"

#: What an operator-local KL reference may be: the historical parent (expressed
#: by omitting the key) or one of the declarable references.
OPERATOR_REFERENCES = frozenset({PARENT_STATE}) | DECLARABLE

#: The search schedules and ranking policies the core ships, by qualified id.
SHIPPED_SCHEDULES: Mapping[str, BeamSchedule] = {
    SCHEDULE_V1.qualified_id: SCHEDULE_V1}
SHIPPED_RANKING_POLICIES: Mapping[str, BeamRankingPolicy] = {
    PARETO_V1.qualified_id: PARETO_V1}

#: The identities a recorded artifact is bound by. All four, for the reason
#: `specs.artifact` gives: a consumer comparing fewer would accept bytes that
#: differ under a colliding construction identity.
ARTIFACT_IDENTITY_FIELDS: tuple[str, ...] = (
    "artifact_digest", "weights_digest", "single_shard_sha256",
    "arch_signature",
)


class BaselineError(ValueError):
    """A baseline record or selection the core cannot honour as written."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise BaselineError(f"{where} has no {key!r}")
    return mapping[key]


def _execution(raw: Mapping[str, Any], where: str) -> ExecutionConfig:
    try:
        return ExecutionConfig(
            micro_batch_size=_require(raw, "micro_batch_size", where),
            calibration_batch_packing=_require(
                raw, "calibration_batch_packing", where),
            prediction_rows=_require(raw, "prediction_rows", where))
    except ExecutionError as exc:
        raise BaselineError(f"{where}: {exc}") from exc


def _numerics(raw: Mapping[str, Any], where: str) -> NumericalEnvironment:
    try:
        return NumericalEnvironment(
            device_type=_require(raw, "device_type", where),
            compute_dtype=_require(raw, "compute_dtype", where),
            accumulation_dtype=_require(raw, "accumulation_dtype", where))
    except Exception as exc:  # the environment's own refusal, re-labelled
        raise BaselineError(f"{where}: {exc}") from exc


@dataclass(frozen=True)
class RecordedArtifact:
    """An artifact that instantiates the method under one execution profile."""

    label: str
    semantic_state_id: str
    identities: Mapping[str, str]
    execution: ExecutionConfig
    numerical_execution_fingerprint: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label,
                "semantic_state_id": self.semantic_state_id,
                **dict(self.identities),
                "execution": self.execution.as_trace() | {
                    "prediction_rows": self.execution.prediction_rows},
                "numerical_execution_fingerprint":
                    self.numerical_execution_fingerprint}


@dataclass(frozen=True)
class ForwardBaseline:
    """A loaded, checked baseline record. Built by `load_baseline`, never by hand."""

    baseline_id: str
    version: int
    record_sha256: str
    execution: ExecutionConfig
    numerics: NumericalEnvironment
    distribution_support: DistributionSupport
    position_policy: ScoringPositionPolicy
    schedule: BeamSchedule
    ranking_policy: BeamRankingPolicy
    #: KIND -> {"impl_id", "signature_hash"}
    operators: Mapping[str, Mapping[str, str]]
    #: KIND -> operator-local KL reference (PARENT_STATE when not declared)
    operator_kl_references: Mapping[str, str]
    state_evaluation_reference: str
    reference_artifacts: tuple[RecordedArtifact, ...]
    forward_materializations: tuple[RecordedArtifact, ...]
    record: Mapping[str, Any]

    @property
    def qualified_id(self) -> str:
        return f"{self.baseline_id}@v{self.version}"

    @property
    def execution_fingerprint(self) -> str:
        """The forward profile's numerical execution fingerprint."""
        return numerical_execution_fingerprint(self.execution, self.numerics)

    def implementation_ids(self) -> dict[str, str]:
        """KIND -> impl_id."""
        return {k: str(v["impl_id"]) for k, v in self.operators.items()}

    def kl_reference_config(self, kind: str) -> dict[str, str]:
        """The operator-config fragment that declares `kind`'s KL reference.

        Empty for the parent: `scoring.kl_reference` expresses the historical
        default by OMITTING the key, and refuses it spelled out.
        """
        if kind not in self.operators:
            raise BaselineError(f"{self.qualified_id} names no {kind!r} operator")
        ref = self.operator_kl_references.get(kind, PARENT_STATE)
        return {} if ref == PARENT_STATE else {KL_REFERENCE_KEY: ref}

    def reference_artifact_for(self, execution: ExecutionConfig,
                               numerics: NumericalEnvironment | None = None
                               ) -> RecordedArtifact | None:
        """The recorded artifact materialized under EXACTLY this execution.

        None when no recorded artifact shares the fingerprint. A shared semantic
        state id is not enough: the bytes are a function of the fingerprint, so
        an artifact recorded under another profile is not what this execution
        would produce, and handing out its digest as an expectation would bind a
        future materialization to bytes it cannot reproduce.
        """
        want = numerical_execution_fingerprint(execution,
                                               numerics or self.numerics)
        for art in (*self.forward_materializations, *self.reference_artifacts):
            if art.numerical_execution_fingerprint == want:
                return art
        return None

    def forward_materialization(self) -> RecordedArtifact | None:
        """An artifact materialized under the forward profile, if one exists."""
        return self.reference_artifact_for(self.execution)

    def require_registered_operators(self, get_implementation) -> dict[str, str]:
        """Check each named implementation against the live registry.

        `get_implementation` is the caller's registry lookup (registration is
        explicit in this codebase, so the caller registers first). Raises when
        an implementation is missing, is of another kind, has a different
        signature, or is handed a KL reference it does not accept.
        """
        out: dict[str, str] = {}
        for kind, spec in self.operators.items():
            impl_id = str(spec["impl_id"])
            try:
                impl = get_implementation(impl_id)
            except Exception as exc:
                raise BaselineError(
                    f"{self.qualified_id}: {impl_id} is not registered ({exc}); "
                    "register the implementations before resolving a session"
                ) from exc
            if getattr(impl, "kind", kind) != kind:
                raise BaselineError(
                    f"{self.qualified_id}: {impl_id} implements "
                    f"{getattr(impl, 'kind', None)!r}, not {kind!r}")
            got = impl.signature_hash
            if got != spec["signature_hash"]:
                raise BaselineError(
                    f"{self.qualified_id}: {impl_id} has signature "
                    f"{got[:12]} and the record names "
                    f"{str(spec['signature_hash'])[:12]}. A changed algorithm "
                    "is a new baseline, not this one")
            ref = self.operator_kl_references.get(kind, PARENT_STATE)
            if ref != PARENT_STATE and not getattr(
                    impl, "accepts_kl_reference", False):
                raise BaselineError(
                    f"{self.qualified_id}: {impl_id} does not accept a KL "
                    f"reference and the record declares {ref!r} for {kind}")
            out[kind] = impl_id
        return out

    def as_record(self) -> dict[str, Any]:
        """What a session's evidence states about the baseline it ran."""
        return {
            "baseline": self.qualified_id,
            "record_sha256": self.record_sha256,
            "numerical_execution_fingerprint": self.execution_fingerprint,
            "execution": self.execution.as_trace() | {
                "prediction_rows": self.execution.prediction_rows},
            "implementations": self.implementation_ids(),
            "position_policy": self.position_policy.qualified_id,
            "distribution_support": self.distribution_support.as_dict(),
        }


def _artifact(raw: Mapping[str, Any], numerics: NumericalEnvironment,
              where: str) -> RecordedArtifact:
    label = str(_require(raw, "label", where))
    where = f"{where} {label!r}"
    execution = _execution(_require(raw, "execution", where), where)
    stated = str(_require(raw, "numerical_execution_fingerprint", where))
    derived = numerical_execution_fingerprint(execution, numerics)
    if stated != derived:
        raise BaselineError(
            f"{where} states fingerprint {stated[:12]} and its execution "
            f"derives {derived[:12]}; an artifact's profile and its fingerprint "
            "are one fact and must agree")
    identities = {}
    for name in ARTIFACT_IDENTITY_FIELDS:
        value = raw.get(name)
        if not value:
            raise BaselineError(
                f"{where} records no {name}; an artifact bound by fewer than its "
                "four identities can be satisfied by bytes that are not it")
        identities[name] = str(value)
    return RecordedArtifact(
        label=label,
        semantic_state_id=str(_require(raw, "semantic_state_id", where)),
        identities=identities, execution=execution,
        numerical_execution_fingerprint=stated,
        detail={k: v for k, v in raw.items()
                if k not in ARTIFACT_IDENTITY_FIELDS})


def parse_baseline(doc: Mapping[str, Any], *, record_sha256: str) -> ForwardBaseline:
    """Check a parsed record against the core and build the baseline."""
    if doc.get("schema") != SCHEMA:
        raise BaselineError(
            f"schema {doc.get('schema')!r} is not {SCHEMA!r}")
    baseline_id = str(_require(doc, "baseline_id", "the record"))
    version = _require(doc, "version", "the record")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise BaselineError(f"version must be a positive int, got {version!r}")
    where = f"{baseline_id}@v{version}"
    method = _require(doc, "method", where)

    execution = _execution(_require(doc, "execution", where), f"{where} execution")
    numerics = _numerics(_require(doc, "numerics", where), f"{where} numerics")
    stated_fp = str(_require(doc, "numerical_execution_fingerprint", where))
    derived_fp = numerical_execution_fingerprint(execution, numerics)
    if stated_fp != derived_fp:
        raise BaselineError(
            f"{where} states execution fingerprint {stated_fp[:12]} and its "
            f"execution profile derives {derived_fp[:12]}")

    sup = _require(method, "distribution_support", where)
    try:
        support = DistributionSupport(
            str(_require(sup, "support_id", where)), sup.get("top_k"))
    except Exception as exc:
        raise BaselineError(f"{where} distribution_support: {exc}") from exc

    pos = _require(method, "position_policy", where)
    try:
        policy = get_position_policy(str(_require(pos, "qualified_id", where)))
    except KeyError as exc:
        raise BaselineError(f"{where}: {exc}") from exc
    if policy.policy_hash != _require(pos, "policy_hash", where):
        raise BaselineError(
            f"{where}: {policy.qualified_id} hashes to {policy.policy_hash[:12]} "
            f"and the record names {str(pos['policy_hash'])[:12]}")

    search = _require(method, "search", where)
    sched = _require(search, "schedule", where)
    schedule = SHIPPED_SCHEDULES.get(str(_require(sched, "qualified_id", where)))
    if schedule is None or schedule.schedule_hash != sched.get("schedule_hash"):
        raise BaselineError(
            f"{where}: schedule {sched.get('qualified_id')!r} with hash "
            f"{str(sched.get('schedule_hash'))[:12]} is not one the core ships; "
            f"shipped: {sorted(SHIPPED_SCHEDULES)}")
    rank = _require(search, "ranking", where)
    ranking = SHIPPED_RANKING_POLICIES.get(str(_require(rank, "qualified_id", where)))
    if ranking is None or ranking.policy_hash != rank.get("policy_hash"):
        raise BaselineError(
            f"{where}: ranking {rank.get('qualified_id')!r} with hash "
            f"{str(rank.get('policy_hash'))[:12]} is not one the core ships; "
            f"shipped: {sorted(SHIPPED_RANKING_POLICIES)}")

    ops = _require(method, "operators", where)
    if not isinstance(ops, Mapping) or not ops:
        raise BaselineError(f"{where} names no operators")
    operators: dict[str, dict[str, str]] = {}
    for kind, spec in ops.items():
        operators[str(kind)] = {
            "impl_id": str(_require(spec, "impl_id", f"{where} {kind}")),
            "signature_hash": str(_require(spec, "signature_hash",
                                           f"{where} {kind}"))}

    refs = _require(method, "kl_reference", where)
    local = dict(_require(refs, "operator_local", where) or {})
    for kind, ref in local.items():
        if kind not in operators:
            raise BaselineError(
                f"{where} declares a KL reference for {kind!r}, which it names "
                "no operator for")
        if ref not in OPERATOR_REFERENCES:
            raise BaselineError(
                f"{where}: {kind} KL reference {ref!r} is not one of "
                f"{sorted(OPERATOR_REFERENCES)}")
    state_ref = _require(refs, "state_evaluation", where)
    if state_ref != STATE_EVALUATION_REFERENCE:
        raise BaselineError(
            f"{where}: state evaluation reference {state_ref!r} is not "
            f"{STATE_EVALUATION_REFERENCE!r}, the only one the state evaluator "
            "implements")

    references = tuple(_artifact(a, numerics, f"{where} reference artifact")
                       for a in doc.get("reference_artifacts") or ())
    forward = tuple(_artifact(a, numerics, f"{where} forward materialization")
                    for a in doc.get("forward_materializations") or ())
    for art in forward:
        if art.numerical_execution_fingerprint != derived_fp:
            raise BaselineError(
                f"{where}: forward materialization {art.label!r} was produced "
                f"under fingerprint {art.numerical_execution_fingerprint[:12]}, "
                f"not the forward profile's {derived_fp[:12]}")

    return ForwardBaseline(
        baseline_id=baseline_id, version=version, record_sha256=record_sha256,
        execution=execution, numerics=numerics, distribution_support=support,
        position_policy=policy, schedule=schedule, ranking_policy=ranking,
        operators=operators, operator_kl_references=local,
        state_evaluation_reference=state_ref, reference_artifacts=references,
        forward_materializations=forward, record=doc)


def load_baseline(path: str | Path, *,
                  expect_sha256: str | None = None) -> ForwardBaseline:
    """Load and check one baseline record from `path`."""
    p = Path(path)
    if not p.is_file():
        raise BaselineError(f"no baseline record at {p}")
    raw = p.read_bytes()
    digest = _sha256_bytes(raw)
    if expect_sha256 is not None and digest != expect_sha256:
        raise BaselineError(
            f"{p} hashes to {digest[:12]} and its selection binds "
            f"{expect_sha256[:12]}: the record was edited after it was "
            "selected. A changed baseline is re-selected deliberately")
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise BaselineError(f"{p} is not JSON: {exc}") from exc
    return parse_baseline(doc, record_sha256=digest)


def resolve_selection(selection_path: str | Path, role: str) -> ForwardBaseline:
    """The baseline in force for `role`, through its hash-bound selection.

    The record path in a selection is relative to the selection file's own
    directory, so a selection resolves the same wherever the tree is checked
    out.
    """
    sp = Path(selection_path)
    if not sp.is_file():
        raise BaselineError(f"no baseline selection at {sp}")
    try:
        doc = json.loads(sp.read_text())
    except ValueError as exc:
        raise BaselineError(f"{sp} is not JSON: {exc}") from exc
    if doc.get("schema") != SELECTION_SCHEMA:
        raise BaselineError(
            f"{sp}: schema {doc.get('schema')!r} is not {SELECTION_SCHEMA!r}")
    selections = doc.get("selections") or {}
    if role not in selections:
        raise BaselineError(
            f"{sp} selects nothing for {role!r}; roles: {sorted(selections)}")
    entry = selections[role]
    record = sp.parent / str(_require(entry, "record", f"{sp} {role}"))
    return load_baseline(record,
                         expect_sha256=str(_require(entry, "sha256",
                                                    f"{sp} {role}")))


__all__ = ["ARTIFACT_IDENTITY_FIELDS", "BaselineError", "ForwardBaseline",
           "OPERATOR_REFERENCES", "RecordedArtifact", "SCHEMA",
           "SELECTION_SCHEMA", "SHIPPED_RANKING_POLICIES", "SHIPPED_SCHEDULES",
           "STATE_EVALUATION_REFERENCE", "load_baseline", "parse_baseline",
           "resolve_selection"]
