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
    ``semantic_state_id`` + ``numerical_execution_fingerprint``. **This** is what
    resume, dedup and checkpoint ownership may key on. Two protocols of one path
    have one semantic id and two materialization ids, which is exactly the
    distinction the measurement above found and the single id could not express.

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


def materialization_id(semantic_state_id: str, execution_fingerprint: str) -> str:
    """The identity resume, dedup and checkpoint ownership may key on."""
    if not semantic_state_id:
        raise MaterializationError(
            "refusing to form a materialization id without a semantic state id")
    if not execution_fingerprint:
        raise MaterializationError(
            "refusing to form a materialization id without an execution "
            "fingerprint: its absence is precisely the measured collision "
            "this type exists to prevent")
    return sha256_json({"schema": SCHEMA,
                        "semantic_state_id": semantic_state_id,
                        "numerical_execution_fingerprint": execution_fingerprint,
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

    @classmethod
    def build(cls, *, semantic_state_id: str, execution: Any,
              environment: NumericalEnvironment,
              artifact_digest: str | None = None) -> "MaterializationIdentity":
        fingerprint = numerical_execution_fingerprint(execution, environment)
        return cls(semantic_state_id=semantic_state_id,
                   numerical_execution_fingerprint=fingerprint,
                   materialization_id=materialization_id(semantic_state_id,
                                                         fingerprint),
                   artifact_digest=artifact_digest)

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
            artifact_digest=artifact_digest)

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
                "artifact_digest": self.artifact_digest}


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
