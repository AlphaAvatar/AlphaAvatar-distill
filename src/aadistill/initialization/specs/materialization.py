"""Four identities, because a search state is not a checkpoint's bytes.

**The measured fact this exists for.** One calibration-consuming operator, one
path, one hashed config and one ``result_spec_hash`` can produce two different
checkpoints, differing only in which calibration items shared a forward pass.
That has been measured on real hardware in this project, reproducibly and across
machines: the difference is deterministic, not noise. Same science, different
bytes. (The run that measured it, the two digests and the replicate structure
are in ``docs/core-provenance.md``, which is where a campaign's instance data
belongs.)

``compute_state_id`` reads the root teacher, the target spec and each step's
``identity()`` — implementation, signature, profile, config hash, seed. It binds
neither the :class:`~aadistill.initialization.execution.ExecutionConfig` nor the
artifact digest, by design: an execution knob must not fork a scientific state.
But resume, deduplication and checkpoint ownership are keyed on that same id, so
two genuinely different artifacts would collide on one resumable, deduplicable
state. That is an engineering correctness defect, and **a passing behavioural
result does not clear it**: a measurement that happens to agree says nothing
about whether two artifacts can be told apart.

So the single id becomes four, each with one job:

``semantic_state_id``
    The scientific/path identity. ``compute_state_id``, unchanged, still
    *deliberately* blind to execution. Two batching protocols of the same path
    are the same hypothesis and must keep sharing this.

``numerical_execution_fingerprint``
    The byte-affecting execution semantics: which items share a forward, how
    they are grouped, the compute dtype, the device class, the accumulation
    dtype. Nothing else — see :data:`FINGERPRINT_FIELDS` and the refusal in
    :class:`NumericalEnvironment`. A fingerprint that absorbed every runtime
    detail would fork on a log level.

``materialization_id``
    ``semantic_state_id`` + ``numerical_execution_fingerprint`` + **the parent
    materialization actually consumed**. **This** is what resume, dedup and
    checkpoint ownership may key on. Two protocols of one path have one semantic
    id and two materialization ids, which is exactly the distinction the
    measurement above found and the single id could not express.

    **The parent term is not decoration.** A child's bytes are a function of the
    bytes it was handed, not only of its own operator and execution protocol.
    Without it, two children of two *differently materialized* parents — same
    path, same fingerprint — would share one materialization id and could
    therefore resume each other, which is the original collision one level up
    the tree. A root has no parent and takes
    :func:`root_materialization_id` instead.

``artifact_digest``
    What was actually built. Observed, not predicted. The materialization id says
    which materialization a file is *supposed* to be; the digest says what it
    *is*, and :meth:`MaterializationIdentity.bind` records the pair so a later
    reader can check one against the other.

**What this module must not become.** A fake operator id. Registering a
second implementation whose id names a batch size was explicitly forbidden, and
for the right reason: the operator's semantics did not change, so an id change
would lie about the science and multiply the registry by every execution knob
forever. The fingerprint is a second coordinate, not a renaming of the first.

Nothing here knows about an experiment, a stage, a model family, a geometry or a
provider. It takes an execution config and a declared numerical environment and
returns hashes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aadistill.infrastructure.manifest import sha256_json

SCHEMA = "aadistill.autoinit.materialization_identity/v1"

FINGERPRINT_SCHEMA = "aadistill.autoinit.numerical_execution_fingerprint/v1"

#: The ROOT's materialization schema, separate from the child one because the
#: two are computed from different things: a root from its published revision,
#: a child from its path, its execution and its parent.
ROOT_SCHEMA = "aadistill.autoinit.root_materialization/v1"

#: Every field the fingerprint covers, enumerated POSITIVELY. The same
#: discipline `aadistill.initialization.execution` adopted for the
#: science/execution boundary, and for the same reason: an exclusion list spelled
#: as a negative makes the next person's omission silent, while a positive list
#: makes it visible. Two groups, because they have two owners —
#: ``ExecutionConfig`` owns how calibration items reach a forward, and the
#: application owns what numerical environment the forward runs in.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "micro_batch_size",
    "calibration_batch_packing",
    "device_type",
    "compute_dtype",
    "accumulation_dtype",
)

#: WHAT BELONGS IN THE FINGERPRINT, AND WHAT DOES NOT. The list above is v1 and
#: will grow; this is the rule for growing it, so that a future field is added
#: deliberately rather than by whoever hits the problem first.
#:
#: **In** — anything that selects a different ARITHMETIC PATH over the same
#: inputs, because two arithmetic paths can produce two sets of bytes:
#:
#: * which items share a forward, and how they are grouped (both v1 fields);
#: * the device CLASS, the compute dtype, the accumulation dtype (v1);
#: * an explicit attention/kernel/backend selection — an SDPA-vs-eager choice,
#:   a flash/math/mem-efficient kernel preference, a deterministic-algorithms
#:   flag, a TF32 or reduced-precision-reduction setting. **None of these is a
#:   field yet, because nothing in this project selects them explicitly.** The
#:   moment something does, it belongs here: this project has already measured
#:   an attention output reducing shape-dependently on one card, so a backend
#:   choice is exactly the kind of thing that moves bytes under an unchanged
#:   operator.
#:
#: **Out** — anything that identifies WHERE or WHEN the work ran rather than
#: what arithmetic it performed. A provider or pod id, a CUDA ordinal, a host
#: name, a driver patch number, a log level, a telemetry flag, a wall clock, a
#: run id, a workdir path. A fingerprint that absorbed these would fork on
#: scheduling, and resume would never match anything.
#:
#: **The boundary case, stated because it is the one that will come up.** A
#: driver or image version is OUT: `generation_compat` already classifies a
#: driver patch as provenance rather than a runtime event, and the project has
#: reproduced one artifact digest byte-identically across three separately
#: rented cards of one model. A MAJOR version change that alters kernel
#: selection would be a real arithmetic-path change — and the way to express it
#: is an explicit field for the thing that changed, not a version string that
#: forks the identity on every patch.
#:
#: The objective is narrow: a future explicit backend choice must not be able
#: to produce different bytes under one `materialization_id`. Nothing here asks
#: for a runtime framework.
FINGERPRINT_GROWTH_RULE = (
    "in: anything selecting a different arithmetic path over the same inputs "
    "(grouping, device class, dtypes, and an explicit attention/kernel/backend "
    "or determinism/TF32 selection once one exists). out: anything naming "
    "where or when the work ran (provider or pod id, CUDA ordinal, host, "
    "driver patch, log level, wall clock, run id, workdir)."
)


class MaterializationError(RuntimeError):
    """A materialization identity could not be formed, or does not match."""


@dataclass(frozen=True)
class NumericalEnvironment:
    """The numerical conditions a materialization was produced under.

    Declared by the application rather than sniffed from a live model, because
    the core must not guess: a fingerprint derived from whatever happened to be
    loaded would be a fingerprint of an accident, and a dry run that resolved
    ``cpu``/``float32`` would then claim to describe a bf16 CUDA artifact.

    ``device_type`` is the device *class*, never an ordinal. One digest has
    been rebuilt byte-identically on three different rented cards of the same
    model, so the particular card is not part of the identity — but ``cpu`` and
    ``cuda`` are different kernels and different reductions, and a CPU dry run
    must not be able to satisfy a GPU resume. See AGENTS.md P3: a CUDA ordinal is
    exactly the kind of instance fact that may not reach reusable core.
    """

    device_type: str
    compute_dtype: str
    accumulation_dtype: str = "float64"

    def __post_init__(self) -> None:
        for name in ("device_type", "compute_dtype", "accumulation_dtype"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MaterializationError(
                    f"NumericalEnvironment.{name} must be a non-empty string, got "
                    f"{value!r}; an unstated numerical condition is not a default, "
                    "it is a fingerprint that does not describe the run")
        if any(ch.isdigit() for ch in self.device_type):
            raise MaterializationError(
                f"device_type {self.device_type!r} carries an ordinal. The device "
                "CLASS is byte-affecting; which card of that class is not — "
                "one digest has reproduced on three cards of a model — and an "
                "ordinal in an identity makes a resume depend on scheduling")

    def as_dict(self) -> dict[str, str]:
        return {"device_type": self.device_type,
                "compute_dtype": self.compute_dtype,
                "accumulation_dtype": self.accumulation_dtype}


def numerical_execution_fingerprint(execution: Any,
                                   environment: NumericalEnvironment) -> str:
    """The byte-affecting execution semantics of one materialization.

    ``execution`` is an :class:`~aadistill.initialization.execution.ExecutionConfig`
    (anything exposing ``as_fingerprint()``). Only the fields in
    :data:`FINGERPRINT_FIELDS` participate, and a field appearing on neither side
    raises rather than being skipped: a fingerprint silently missing a
    byte-affecting knob is the defect, not the protection.
    """
    fields = dict(_execution_fingerprint_fields(execution))
    fields.update(environment.as_dict())
    missing = [f for f in FINGERPRINT_FIELDS if f not in fields]
    if missing:
        raise MaterializationError(
            f"the numerical execution fingerprint is missing {missing}. Every "
            f"field in FINGERPRINT_FIELDS must be supplied by the execution "
            "config or the numerical environment; an absent one would make two "
            "differing materializations hash alike")
    extra = sorted(set(fields) - set(FINGERPRINT_FIELDS))
    if extra:
        raise MaterializationError(
            f"{extra} were offered to the fingerprint but are not declared in "
            "FINGERPRINT_FIELDS. Add the field to that tuple deliberately — a "
            "fingerprint that absorbs whatever it is handed forks on runtime "
            "detail and stops being an identity")
    return sha256_json({"schema": FINGERPRINT_SCHEMA,
                        **{f: fields[f] for f in FINGERPRINT_FIELDS}})


def _execution_fingerprint_fields(execution: Any) -> Mapping[str, Any]:
    view = getattr(execution, "as_fingerprint", None)
    if view is None:
        raise MaterializationError(
            f"{type(execution).__name__} exposes no `as_fingerprint()`. The "
            "execution config owns which of its knobs affect bytes; a caller "
            "that enumerated them here would be a second opinion about the same "
            "question")
    return view()


def root_materialization_id(*, root_teacher_id: str,
                            root_teacher_sha256: str) -> str:
    """The root's materialization: the teacher's own published identity.

    A root is not something this project built. Its bytes are a published
    revision, so its materialization is that revision's identity and **nothing
    else** — in particular it is deliberately NOT fingerprinted, because the
    teacher's weights do not depend on how this project groups calibration
    items or which dtype a later operator accumulates in.

    Generic by construction: it takes the two fields that pin any teacher,
    whatever family, stage or scale, rather than naming an experiment's
    checkpoint. `make_root_state` already requires both.
    """
    if not root_teacher_id or not root_teacher_sha256:
        raise MaterializationError(
            "a root materialization needs the teacher's id AND its revision "
            f"hash; got {root_teacher_id!r} / {root_teacher_sha256!r}. An "
            "unpinned root is a lineage that starts from nothing in "
            "particular, which every child would then inherit")
    return sha256_json({"schema": ROOT_SCHEMA,
                        "root_teacher_id": root_teacher_id,
                        "root_teacher_sha256": root_teacher_sha256})[:32]


def materialization_id(semantic_state_id: str, execution_fingerprint: str,
                       parent_materialization_id: str) -> str:
    """The identity resume, dedup and checkpoint ownership may key on.

    All three terms are REQUIRED and none has a default. An optional parent
    would reintroduce the collision one level up the tree — two children of
    differently materialized parents sharing an id — and the whole point of
    this function is that such a collision cannot be formed by omission.
    """
    if not semantic_state_id:
        raise MaterializationError(
            "refusing to form a materialization id without a semantic state id")
    if not execution_fingerprint:
        raise MaterializationError(
            "refusing to form a materialization id without an execution "
            "fingerprint: its absence is precisely the measured collision "
            "this type exists to prevent")
    if not parent_materialization_id:
        raise MaterializationError(
            "refusing to form a materialization id without the parent "
            "materialization actually consumed: a child's bytes are a function "
            "of the bytes handed to it, so two children of differently "
            "materialized parents would otherwise share one id. A root has no "
            "parent and uses `root_materialization_id`")
    return sha256_json({"schema": SCHEMA,
                        "semantic_state_id": semantic_state_id,
                        "numerical_execution_fingerprint": execution_fingerprint,
                        "parent_materialization_id": parent_materialization_id,
                        })[:32]


@dataclass(frozen=True)
class MaterializationIdentity:
    """The four identities of one produced checkpoint, together.

    ``artifact_digest`` is ``None`` until something has actually been built. That
    is not laxity: the identity is formed *before* the bytes exist — it is what
    decides whether to build them at all — and :meth:`bind` is the moment the
    prediction meets the observation.
    """

    semantic_state_id: str
    numerical_execution_fingerprint: str
    materialization_id: str
    artifact_digest: str | None = None
    #: The parent materialization this one was built FROM. `None` only on a
    #: root, which has no parent by construction. Carried as a field rather
    #: than left inside the hash so lineage is readable: a record can say which
    #: bytes it consumed, not merely that it consumed some.
    parent_materialization_id: str | None = None

    @classmethod
    def root(cls, *, semantic_state_id: str, root_teacher_id: str,
             root_teacher_sha256: str,
             artifact_digest: str | None = None) -> "MaterializationIdentity":
        """The root's identity, from the pinned teacher and nothing else.

        Its `numerical_execution_fingerprint` is the root materialization id
        itself, because there is no execution to fingerprint — this project did
        not run anything to produce the teacher. Stated rather than left as a
        surprising value: the alternative was an empty string, which
        `require_same_materialization` would then have to special-case, or a
        fingerprint of the *current* run's execution, which would falsely claim
        the teacher's bytes depend on how we batch.
        """
        root_id = root_materialization_id(
            root_teacher_id=root_teacher_id,
            root_teacher_sha256=root_teacher_sha256)
        return cls(semantic_state_id=semantic_state_id,
                   numerical_execution_fingerprint=root_id,
                   materialization_id=root_id,
                   artifact_digest=artifact_digest,
                   parent_materialization_id=None)

    @classmethod
    def build(cls, *, semantic_state_id: str, execution: Any,
              environment: NumericalEnvironment,
              parent_materialization_id: str,
              artifact_digest: str | None = None) -> "MaterializationIdentity":
        fingerprint = numerical_execution_fingerprint(execution, environment)
        return cls(semantic_state_id=semantic_state_id,
                   numerical_execution_fingerprint=fingerprint,
                   materialization_id=materialization_id(
                       semantic_state_id, fingerprint,
                       parent_materialization_id),
                   artifact_digest=artifact_digest,
                   parent_materialization_id=parent_materialization_id)

    def bind(self, artifact_digest: str) -> "MaterializationIdentity":
        if not artifact_digest:
            raise MaterializationError(
                f"{self.materialization_id[:12]}: refusing to bind an empty "
                "artifact digest")
        if self.artifact_digest is not None and \
                self.artifact_digest != artifact_digest:
            raise MaterializationError(
                f"{self.materialization_id[:12]} already names artifact "
                f"{self.artifact_digest[:12]} and cannot be rebound to "
                f"{artifact_digest[:12]}. One materialization, one set of bytes")
        return MaterializationIdentity(
            semantic_state_id=self.semantic_state_id,
            numerical_execution_fingerprint=self.numerical_execution_fingerprint,
            materialization_id=self.materialization_id,
            artifact_digest=artifact_digest,
            parent_materialization_id=self.parent_materialization_id)

    def same_materialization(self, other: "MaterializationIdentity | Mapping[str, Any]",
                             ) -> bool:
        return _materialization_of(other) == self.materialization_id

    def require_same_materialization(
            self, other: "MaterializationIdentity | Mapping[str, Any]", *,
            what: str = "record") -> None:
        """Refuse a record produced under different byte-affecting execution.

        The one call that makes this module load-bearing rather than
        descriptive. A journal entry, a cached statistic or a restored
        checkpoint whose semantic id matches but whose fingerprint does not is
        **not** this materialization, and admitting it is how a bsz=1 artifact
        comes to stand in for a bsz=3 one.
        """
        theirs = _materialization_of(other)
        if theirs == self.materialization_id:
            return
        their_fp = _fingerprint_of(other)
        raise MaterializationError(
            f"{what} is materialization {str(theirs)[:12]} and this run is "
            f"{self.materialization_id[:12]}: the semantic state may be the same "
            f"path, but the byte-affecting execution differs "
            f"({str(their_fp)[:12]} vs {self.numerical_execution_fingerprint[:12]}). "
            "Two different artifacts under one semantic id have been measured; "
            "reusing one for the other is that collision")

    def as_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA,
                "semantic_state_id": self.semantic_state_id,
                "numerical_execution_fingerprint":
                    self.numerical_execution_fingerprint,
                "materialization_id": self.materialization_id,
                "artifact_digest": self.artifact_digest,
                "parent_materialization_id": self.parent_materialization_id}


def _materialization_of(other: Any) -> str | None:
    if isinstance(other, MaterializationIdentity):
        return other.materialization_id
    if isinstance(other, Mapping):
        return other.get("materialization_id")
    return getattr(other, "materialization_id", None)


def _fingerprint_of(other: Any) -> str | None:
    if isinstance(other, MaterializationIdentity):
        return other.numerical_execution_fingerprint
    if isinstance(other, Mapping):
        return other.get("numerical_execution_fingerprint")
    return getattr(other, "numerical_execution_fingerprint", None)
