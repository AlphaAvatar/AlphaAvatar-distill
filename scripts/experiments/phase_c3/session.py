"""The Phase-C3 session: ordered stages, fail-closed gates, and arm construction.

One session, ten stages, and two of them are scientific gates that stop it dead.

    A  provider / session setup
    B  fetch the pinned teacher revision, verify EVERY file against the binding
    C  register EVERY non-builtin implementation the three arms name
    D  replay DEPTH -> FFN -> RESIDUAL_WIDTH   GATE: parent  == eea90c91...
    E  apply attention.activation_importance_v1  GATE: incumbent == 53e30566...
    F  materialize ALL THREE arms from the SAME verified parent
    G  9 recovery probes: 3 arms x 3 fresh seeds, no elimination
    H  evaluate every completed probe once on c1_confirmation_v1
    I  apply the frozen paired decision, only after all nine results exist
    J  collect evidence, then teardown

**D and E are stop conditions, not warnings.** If either digest mismatches, the
session must end *before* any 0.86M recovery training and preserve the evidence:
every intermediate identity, the DEPTH/FFN selections, the WIDTH projection
diagnostics and the runtime triple. There is no automatic waiver, and a later
functional-equivalence amendment is a decision to be made from the actual
mismatch evidence rather than pre-authorized here.

**C must precede D and F.** Neither experimental ATTENTION implementation is a
builtin, and neither registers on import, because `BeamSearch._allowed_impl_ids`
falls back to the entire registry when `allowed_impls` is None — so registering
at import would add a calibrated ATTENTION branch to any search in the process.
`build_arm_specs` therefore *refuses* to construct an arm whose implementation
is unregistered, which makes the ordering a property of the code rather than a
step in a runbook. The set it registers is **derived from the arms**, so a
fourth arm naming a fourth implementation cannot be silently unregistered.

**The preregistration owns the experimental instance; this module owns the
session's shape.** Arm ids, their implementations and calibration profiles,
their identity-bearing configs, the digest gates, the seeds and the probe count
are all READ from the frozen plan and never re-declared here. At `b937aebb` the
plan said three arms while this module built two, and nothing compared them;
`tests/pod/test_c3_session_contract.py` now proves the two agree field by
field.

This module deliberately contains no search, ranking, successive halving or
tie-breaking, and no provider or transport code: it is the session's shape and
its scientific gates. The pod wiring consumes it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aadistill.infrastructure.manifest import sha256_json
from aadistill.initialization.operators.base import get_implementation
from aadistill.initialization.operators.register import register_builtin_operators
from aadistill.initialization.planning.fixed_path import FixedPathSpec, FixedPathStep
from aadistill.initialization.specs.arch import ArchSpec

#: Explicit: importing an operator module no longer registers it. This call
#: covers the BUILTINS only -- the two experimental ATTENTION implementations
#: are registered by `register_experimental_operators()` at stage C.
register_builtin_operators()

SCHEMA = "aadistill.autoinit.c3_session/v2"

REPO = Path(__file__).resolve().parents[3]
PREREGISTRATION_PATH = REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json"

TEACHER_REPO = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: The frozen student geometry. Held fixed across every arm.
TARGET_GEOMETRY: dict[str, Any] = {
    "head_dim": 128, "hidden_size": 1024, "intermediate_size": 3072,
    "num_attention_heads": 16, "num_hidden_layers": 28,
    "num_key_value_heads": 8, "tie_word_embeddings": True, "vocab_size": 151936,
}

#: The prefix every arm shares. DERIVED from `shared_parent.prefix` in the
#: frozen plan, not restated: the plan already owns it, and a second copy is a
#: second thing to drift. `prefix_steps()` below is the accessor; the two
#: module constants are computed once at import from the same source.

#: The search seed the Phase-B run used. Every operator on this path declares
#: `requires_seed=False` and `ChildBuilder` overwrites every parameter, so it
#: cannot affect the output — it is carried so the FixedPathSpec hash describes
#: the same configuration the journal recorded.
SEARCH_SEED = 20260815

#: Which implementations are NOT builtins and must be registered explicitly.
#: Mapping impl_id -> the module exposing `register()`. An arm naming an id
#: that is neither a builtin nor in here is refused rather than assumed.
EXPERIMENTAL_REGISTRARS: Mapping[str, str] = {
    "attention.activation_importance_v1":
        "aadistill.initialization.operators.attention.gqa.activation_importance",
    "attention.causal_kl_v1":
        "aadistill.initialization.operators.attention.gqa.causal_kl",
}


class C3SessionError(RuntimeError):
    """The session cannot be constructed or ordered as declared."""


# ---------------------------------------------------------------------------
# The frozen preregistration is the instance owner
# ---------------------------------------------------------------------------

def load_preregistration(path: Path | None = None) -> dict[str, Any]:
    """Read the frozen plan and REFUSE it if its stamp does not bind it.

    A plan whose stamp does not verify is not a frozen plan, and building an
    experiment from one would reproduce exactly the failure this module was
    rewritten to prevent: a document that looks authoritative and is not.
    """
    p = Path(path or PREREGISTRATION_PATH)
    if not p.is_file():
        raise C3SessionError(f"no C3 preregistration at {p}")
    doc = json.loads(p.read_text())
    stated = doc.get("preregistration_sha256")
    if not stated:
        raise C3SessionError(f"{p} declares no preregistration_sha256")
    got = sha256_json({k: v for k, v in doc.items()
                       if k != "preregistration_sha256"})
    if got != stated:
        raise C3SessionError(
            f"{p} does not bind itself: stamp {stated[:16]}, body hashes "
            f"{got[:16]}. Refusing to build a formal experiment from an "
            f"unverified plan.")
    return doc


_PREREG: dict[str, Any] | None = None


def preregistration() -> dict[str, Any]:
    global _PREREG
    if _PREREG is None:
        _PREREG = load_preregistration()
    return _PREREG


def arm_ids() -> tuple[str, ...]:
    """The arm ids, in the plan's own order.

    Keys beginning `_` are the plan's commentary (`_operator_semantics`,
    `_only_intended_difference`), not arms. Order is the plan's, because the
    incumbent must be first for the digest-gated replay at stage E.
    """
    return tuple(k for k in preregistration()["arms"] if not k.startswith("_"))


def arm(arm_id: str) -> dict[str, Any]:
    arms = preregistration()["arms"]
    if arm_id not in arms:
        raise C3SessionError(f"no arm {arm_id!r}; the plan declares "
                             f"{list(arm_ids())}")
    return arms[arm_id]


def recovery_seeds() -> tuple[int, ...]:
    return tuple(preregistration()["seeds"]["recovery"])


def bootstrap_seed() -> int:
    return int(preregistration()["seeds"]["bootstrap"])


def prefix_steps() -> tuple[tuple[str, str], ...]:
    """The shared prefix, from the plan. (impl_id, calibration_profile) pairs."""
    return tuple((impl, prof)
                 for impl, prof in preregistration()["shared_parent"]["prefix"])


def prefix_execution() -> dict[str, Any]:
    """The replay's execution config, pinned by the plan rather than defaulted.

    `micro_batch_size` is a RUNTIME knob and is deliberately not hashed into
    any state id -- but a formal replay must still not depend on whatever
    `DEFAULT_MICRO_BATCH_SIZE` happens to be, so the plan pins it and this
    reads it.
    """
    return dict(preregistration()["shared_parent"].get("execution") or {})


def expected_parent_digest() -> str:
    return preregistration()["shared_parent"]["artifact_digest"]


def expected_incumbent_digest() -> str:
    return arm("A_incumbent")["artifact_digest"]


def battery_asset_id() -> str:
    return preregistration()["evaluation"]["battery"]


# ---------------------------------------------------------------------------
# Stage C: register exactly what the arms name
# ---------------------------------------------------------------------------

def required_implementations() -> tuple[str, ...]:
    """Every implementation the arms and the shared prefix name, deduplicated."""
    ids = [impl for impl, _ in prefix_steps()]
    ids += [arm(a)["attention"][0] for a in arm_ids()]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return tuple(out)


def register_experimental_operators() -> tuple[str, ...]:
    """Stage C. Register every non-builtin implementation the arms name.

    Derived from the arms rather than listed, so an arm added to the plan
    cannot reach `build_arm_specs` with its implementation unregistered. An
    implementation that is neither a builtin nor a known experimental one is
    an error here, not a mystery at materialization time.
    """
    import importlib

    registered = []
    for impl_id in required_implementations():
        try:
            get_implementation(impl_id)
            continue                       # already registered (builtin or prior call)
        except Exception:                  # noqa: BLE001 - absence, not failure
            pass
        module = EXPERIMENTAL_REGISTRARS.get(impl_id)
        if module is None:
            raise C3SessionError(
                f"{impl_id} is named by the plan, is not registered, and has "
                f"no known registrar. Add it to EXPERIMENTAL_REGISTRARS or "
                f"to BUILTIN_OPERATORS -- do not let it resolve by accident.")
        importlib.import_module(module).register()
        registered.append(impl_id)
    #: Fail closed: after this call every named implementation must resolve.
    missing = []
    for impl_id in required_implementations():
        try:
            get_implementation(impl_id)
        except Exception:                  # noqa: BLE001
            missing.append(impl_id)
    if missing:
        raise C3SessionError(f"still unregistered after stage C: {missing}")
    return tuple(registered)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class C3Stage:
    letter: str
    stage_id: str
    description: str
    #: What makes this stage stop the session rather than continue.
    fail_closed_on: str
    #: Evidence this stage must have produced before the next may start.
    produces: tuple[str, ...] = ()
    #: True when a failure here must prevent ANY paid recovery training.
    blocks_training: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"letter": self.letter, "stage_id": self.stage_id,
                "description": self.description,
                "fail_closed_on": self.fail_closed_on,
                "produces": list(self.produces),
                "blocks_training": self.blocks_training}


def _stages() -> tuple[C3Stage, ...]:
    n_arms = len(arm_ids())
    n_seeds = len(recovery_seeds())
    n_probes = n_arms * n_seeds
    return (
        C3Stage("A", "session_setup",
                "provider, pod, scratch, bundle, prechecks, watchdog armed",
                "any precheck failing, before a pod exists where possible",
                ("session_record", "watchdog_journal")),
        C3Stage("B", "teacher_fetch_verify",
                f"fetch {TEACHER_REPO}@{TEACHER_REVISION[:12]} and verify every "
                "file against logs/stages/stage-1/phase_c1/plans/teacher_binding.json "
                "-- a shared frozen identity, referenced not copied",
                "ANY shard or config file whose hash differs from the binding",
                ("teacher_verification",), blocks_training=True),
        C3Stage("C", "register_operator",
                "explicitly register EVERY non-builtin implementation the arms "
                f"name ({', '.join(sorted(EXPERIMENTAL_REGISTRARS))}); import "
                "alone registers nothing",
                "any named implementation unregistered before a FixedPathSpec "
                "names it",
                ("operator_registration",), blocks_training=True),
        C3Stage("D", "replay_parent",
                "replay DEPTH -> FFN -> RESIDUAL_WIDTH from the verified teacher",
                f"pre-ATTENTION artifact_digest != {expected_parent_digest()[:12]}...",
                ("parent_identity", "replay_record"), blocks_training=True),
        C3Stage("E", "replay_incumbent",
                "apply attention.activation_importance_v1 to the verified parent",
                f"incumbent artifact_digest != {expected_incumbent_digest()[:12]}...",
                ("incumbent_identity", "replay_record"), blocks_training=True),
        C3Stage("F", "materialize_arms",
                f"materialize all {n_arms} arms from the SAME verified parent "
                "and bind their identities; each initialization is built ONCE",
                "any arm failing to materialize, or the arms not sharing a parent",
                ("arm_identities",), blocks_training=True),
        C3Stage("G", "recovery_probes",
                f"{n_probes} probes: {n_arms} arms x {n_seeds} fresh seeds, "
                "every arm runs every seed",
                "any probe failing; no arm is ever eliminated to make progress",
                ("probe_results", "train_logs")),
        C3Stage("H", "evaluate",
                f"evaluate every completed probe exactly once on {battery_asset_id()}",
                "an incomplete or duplicated evaluation",
                ("per_sample_rows", "generations", "probe_aggregates")),
        C3Stage("I", "decide",
                "apply the frozen paired decision to the PRIMARY contrast, only "
                f"after all {n_probes} results exist; report the secondary and "
                "practical contrasts completely",
                f"fewer than {n_probes} valid probe results",
                ("decision_record",)),
        C3Stage("J", "collect_teardown",
                "collect raw evidence and transfer it, then confirm teardown "
                "with the provider",
                "a required artifact missing from the manifest gate",
                ("evidence_archive", "teardown_confirmation")),
    )


C3_STAGES: tuple[C3Stage, ...] = _stages()

#: Stages whose failure must stop the session before any paid recovery training.
GATE_STAGES: tuple[str, ...] = tuple(
    s.stage_id for s in C3_STAGES if s.blocks_training)

#: What a mismatch at D or E must preserve. A stop that keeps no evidence turns
#: a scientific finding into an outage.
MISMATCH_EVIDENCE: tuple[str, ...] = (
    "every intermediate artifact_digest, in order",
    "DEPTH selected and removed blocks",
    "FFN selected neurons per layer",
    "WIDTH projection diagnostics",
    "ATTENTION kept heads per layer",
    "runtime: image digest, torch, transformers, CUDA runtime, driver, GPU",
    "the expected and realized digests, side by side",
)


def stage(letter: str) -> C3Stage:
    for s in C3_STAGES:
        if s.letter == letter:
            return s
    raise C3SessionError(f"no stage {letter!r}")


def assert_stage_order(completed: Sequence[str]) -> None:
    """Refuse an out-of-order or gapped execution.

    The ordering is the science: evaluating before all the probes exist, or
    deciding before evaluating, would each produce a number that looks like a
    result. A prefix is fine — a session may stop early, and at a gate it must.
    """
    expected = [s.stage_id for s in C3_STAGES]
    if list(completed) != expected[:len(completed)]:
        raise C3SessionError(
            f"stages ran out of order: got {list(completed)}, "
            f"which is not a prefix of {expected}")


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _target_spec() -> ArchSpec:
    return ArchSpec.of("qwen3", TARGET_GEOMETRY)


def build_arm_specs(*, workdir_device: str = "cuda") -> dict[str, FixedPathSpec]:
    """Every arm the frozen plan declares, sharing one pinned prefix.

    Three arms, derived from the preregistration and never listed here:

        A_incumbent   activation_importance_v1, digest-gated
        B_causal_b1   causal_kl_v1  @ batch 1, original_order_v1
        C_causal_b3   causal_kl_v1  @ batch 3, length_sorted_v1

    The incumbent arm carries BOTH digest pins — the parent on its third step
    and the incumbent on its fourth — so a single replay of that arm exercises
    the whole end-to-end gate. Every other arm shares the identical prefix and
    the same parent pin, so they cannot silently diverge before ATTENTION.

    Refuses if stage C has not registered the implementations the arms name.
    """
    for impl_id in required_implementations():
        try:
            get_implementation(impl_id)
        except Exception as exc:               # noqa: BLE001 - re-raised typed
            raise C3SessionError(
                f"{impl_id} is not registered. Stage C registers it "
                "explicitly; importing its module does not, because an "
                "unrestricted BeamSearch enumerates the whole registry. Call "
                "register_experimental_operators() first."
            ) from exc

    parent_digest = expected_parent_digest()
    steps = prefix_steps()
    prefix = [FixedPathStep(impl, prof) for impl, prof in steps]
    prefix[-1] = FixedPathStep(
        steps[-1][0], steps[-1][1],
        expected_artifact_digest=parent_digest,
        label=f"pre-ATTENTION parent {parent_digest[:12]}")

    common = dict(family="qwen3", target_spec=_target_spec(),
                  root_repo_id=TEACHER_REPO, root_revision=TEACHER_REVISION,
                  device=workdir_device, seed=SEARCH_SEED)

    specs: dict[str, FixedPathSpec] = {}
    for arm_id in arm_ids():
        a = arm(arm_id)
        impl_id, profile = a["attention"]
        config = a.get("config")
        digest = a.get("artifact_digest")
        step = FixedPathStep(
            impl_id, profile,
            config=dict(config) if config else None,
            expected_artifact_digest=digest,
            label=(f"{arm_id} {digest[:12]}" if digest else f"{arm_id} ATTENTION"))
        specs[arm_id] = FixedPathSpec(
            path_id=f"autoinit.v1.phase_c3.{arm_id}",
            steps=(*prefix, step), **common)
    return specs


def arms_share_the_prefix(arms: Mapping[str, FixedPathSpec]) -> bool:
    """Every arm must differ in the last step and nowhere else.

    Generalized from a two-key unpack. The old form read
    `a, b = arms["incumbent"], arms["treatment"]`, which would have raised a
    KeyError on the real three-arm matrix -- and which, worse, could only ever
    have compared two of however many arms existed.
    """
    specs = list(arms.values())
    if len(specs) < 2:
        return False
    first = specs[0]
    if any(s.steps[:-1] != first.steps[:-1] for s in specs[1:]):
        return False
    if any(s.target_spec.spec_hash != first.target_spec.spec_hash
           or s.root_revision != first.root_revision for s in specs[1:]):
        return False
    #: Distinct arms must be distinguishable at the last step -- by
    #: implementation OR by identity-bearing config, since B1 and B3 share an
    #: implementation and differ only in the hashed protocol.
    tails = {(s.steps[-1].impl_id, sha256_json(s.steps[-1].as_dict()))
             for s in specs}
    return len(tails) == len(specs)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class C3SessionContract:
    """The session's declared shape, hashable for the preregistration.

    `n_arms`, `n_seeds` and `n_probes` are DERIVED from the frozen plan. They
    were literals `2`, `3`, `6` at `b937aebb` while the plan said three arms
    and nine probes, and the `__post_init__` consistency check passed happily
    because 2 x 3 really is 6 — an internally consistent statement of the
    wrong experiment. A derived value cannot drift from its source.
    """

    session_id: str = "autoinit.v1.phase_c3"
    stages: tuple[C3Stage, ...] = C3_STAGES
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_arms(self) -> int:
        return len(arm_ids())

    @property
    def n_seeds(self) -> int:
        return len(recovery_seeds())

    @property
    def n_probes(self) -> int:
        return self.n_arms * self.n_seeds

    def __post_init__(self) -> None:
        if self.n_arms < 2:
            raise C3SessionError(
                f"the plan declares {self.n_arms} arm(s); a contrast needs two")
        if self.n_probes != self.n_arms * self.n_seeds:
            raise C3SessionError(
                f"{self.n_arms} arms x {self.n_seeds} seeds is not "
                f"{self.n_probes} probes; every arm runs every seed and none "
                "is eliminated")

    def as_dict(self) -> dict[str, Any]:
        prereg = preregistration()
        return {
            "schema": SCHEMA,
            "session_id": self.session_id,
            "stages": [s.as_dict() for s in self.stages],
            "gate_stages": list(GATE_STAGES),
            "mismatch_evidence": list(MISMATCH_EVIDENCE),
            "arm_ids": list(arm_ids()),
            "n_arms": self.n_arms, "n_seeds": self.n_seeds,
            "n_probes": self.n_probes,
            "recovery_seeds": list(recovery_seeds()),
            "bootstrap_seed": bootstrap_seed(),
            "battery_asset_id": battery_asset_id(),
            "expected_parent_digest": expected_parent_digest(),
            "expected_incumbent_digest": expected_incumbent_digest(),
            "preregistration_sha256": prereg["preregistration_sha256"],
            "primary_contrast": prereg["claim_boundary"]["primary_contrast"],
            "teacher": {"repo_id": TEACHER_REPO, "revision": TEACHER_REVISION},
            "target_geometry": dict(TARGET_GEOMETRY),
            "contains": {"search": False, "ranking": False,
                         "successive_halving": False, "tie_breaking": False,
                         "arm_elimination": False},
            "notes": dict(self.notes),
        }

    @property
    def contract_hash(self) -> str:
        return sha256_json(self.as_dict())


C3_SESSION_CONTRACT = C3SessionContract()
