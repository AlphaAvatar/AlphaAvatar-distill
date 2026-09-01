"""The Phase-C1 session: ordered stages, fail-closed gates, and arm construction.

One session, ten stages, and two of them are scientific gates that stop it dead.

    A  provider / session setup
    B  fetch the pinned teacher revision, verify EVERY file against the binding
    C  register attention.activation_importance_v1  (explicit; import is inert)
    D  replay DEPTH -> FFN -> RESIDUAL_WIDTH   GATE: parent  == eea90c91...
    E  apply attention.weight_proxy_v0          GATE: incumbent == c313d1b4...
    F  materialize both arms from the SAME verified parent
    G  6 recovery probes: 2 arms x 3 fresh seeds, no elimination
    H  evaluate every completed probe once on c1_confirmation_v1
    I  apply the frozen paired decision, only after all six results exist
    J  collect evidence, then teardown

**D and E are stop conditions, not warnings.** If either digest mismatches, the
session must end *before* any 0.86M recovery training and preserve the evidence:
every intermediate identity, the DEPTH/FFN selections, the WIDTH projection
diagnostics and the runtime triple. There is no automatic waiver, and a later
functional-equivalence amendment is a decision to be made from the actual
mismatch evidence rather than pre-authorized here.

**C must precede D.** `attention.activation_importance_v1` is registered by an
explicit call, never by import, because `BeamSearch._allowed_impl_ids` falls back
to the entire registry when `allowed_impls` is None — so registering at import
would add a calibrated ATTENTION branch to any search in the process.
`build_arm_specs` therefore *refuses* to construct the treatment arm until the
operator is registered, which makes the ordering a property of the code rather
than a step in a runbook.

This module deliberately contains no search, ranking, successive halving or
tie-breaking, and no provider or transport code: it is the session's shape and
its scientific gates. The pod wiring consumes it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..infrastructure.manifest import sha256_json
from .arch import ArchSpec
from .fixed_path import FixedPathSpec, FixedPathStep

SCHEMA = "aadistill.autoinit.c1_session/v1"

#: The pre-ATTENTION parent of `fe9683e6a9c7`, recorded in the Phase-B search
#: journal. Reproducing it is the first scientific gate.
EXPECTED_PARENT_DIGEST = (
    "eea90c91346a0745b8b1b847503b48fe73c33bb9d75d92c196dc43598e91e722")
EXPECTED_PARENT_STATE_ID = "b8820f41d062bffffca4b99148602136"

#: The retained Phase-B winner. Applying the CURRENT ATTENTION to the verified
#: parent must reproduce it — an end-to-end replay check of the whole fixed path.
EXPECTED_INCUMBENT_DIGEST = (
    "c313d1b4081b9a3b410dddf7a29ebcaad8dd0759179d51e1d761238c1743a2a6")
EXPECTED_INCUMBENT_STATE_ID = "fe9683e6a9c783bbc6fe276a78c851c6"

TEACHER_REPO = "Qwen/Qwen3-4B-Thinking-2507"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: The frozen student geometry. Held fixed across both arms.
TARGET_GEOMETRY: dict[str, Any] = {
    "head_dim": 128, "hidden_size": 1024, "intermediate_size": 3072,
    "num_attention_heads": 16, "num_hidden_layers": 28,
    "num_key_value_heads": 8, "tie_word_embeddings": True, "vocab_size": 151936,
}

#: The prefix every arm shares, at the incumbent's implementations and profiles.
PREFIX_STEPS: tuple[tuple[str, str], ...] = (
    ("depth.causal_kl_greedy_v1", "calib.domain_balanced@v1"),
    ("ffn.activation_importance_v0", "calib.domain_balanced@v1"),
    ("width.global_pca_v0", "calib.reasoning_heavy@v2"),
)

INCUMBENT_ATTENTION = ("attention.weight_proxy_v0", "calib.none@v1")
TREATMENT_ATTENTION = ("attention.activation_importance_v1",
                       "calib.domain_balanced@v1")

#: The search seed the Phase-B run used. Every operator on this path declares
#: `requires_seed=False` and `ChildBuilder` overwrites every parameter, so it
#: cannot affect the output — it is carried so the FixedPathSpec hash describes
#: the same configuration the journal recorded.
SEARCH_SEED = 20260815


class C1SessionError(RuntimeError):
    """The session cannot be constructed or ordered as declared."""


@dataclass(frozen=True)
class C1Stage:
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


C1_STAGES: tuple[C1Stage, ...] = (
    C1Stage("A", "session_setup",
            "provider, pod, scratch, bundle, prechecks, watchdog armed",
            "any precheck failing, before a pod exists where possible",
            ("session_record", "watchdog_journal")),
    C1Stage("B", "teacher_fetch_verify",
            f"fetch {TEACHER_REPO}@{TEACHER_REVISION[:12]} and verify every file "
            "against logs/phase_c1_teacher_binding.json",
            "ANY shard or config file whose hash differs from the binding",
            ("teacher_verification",), blocks_training=True),
    C1Stage("C", "register_operator",
            "explicitly register attention.activation_importance_v1; import alone "
            "does not register it",
            "the operator not being registered before any FixedPathSpec names it",
            ("operator_registration",), blocks_training=True),
    C1Stage("D", "replay_parent",
            "replay DEPTH -> FFN -> RESIDUAL_WIDTH from the verified teacher",
            f"pre-ATTENTION artifact_digest != {EXPECTED_PARENT_DIGEST[:12]}...",
            ("parent_identity", "replay_record"), blocks_training=True),
    C1Stage("E", "replay_incumbent",
            "apply the CURRENT attention.weight_proxy_v0 to the verified parent",
            f"incumbent artifact_digest != {EXPECTED_INCUMBENT_DIGEST[:12]}...",
            ("incumbent_identity", "replay_record"), blocks_training=True),
    C1Stage("F", "materialize_arms",
            "materialize both arms from the SAME verified parent and bind their "
            "identities",
            "either arm failing to materialize, or the two not sharing a parent",
            ("arm_identities",), blocks_training=True),
    C1Stage("G", "recovery_probes",
            "6 probes: 2 arms x 3 fresh seeds, every arm runs every seed",
            "any probe failing; no arm is ever eliminated to make progress",
            ("probe_results", "train_logs")),
    C1Stage("H", "evaluate",
            "evaluate every completed probe exactly once on c1_confirmation_v1",
            "an incomplete or duplicated evaluation",
            ("per_sample_rows", "generations", "probe_aggregates")),
    C1Stage("I", "decide",
            "apply the frozen paired decision, only after all six results exist",
            "fewer than six valid probe results",
            ("decision_record",)),
    C1Stage("J", "collect_teardown",
            "collect raw evidence and transfer it, then confirm teardown with the "
            "provider",
            "a required artifact missing from the manifest gate",
            ("evidence_archive", "teardown_confirmation")),
)

#: Stages whose failure must stop the session before any paid recovery training.
GATE_STAGES: tuple[str, ...] = tuple(
    s.stage_id for s in C1_STAGES if s.blocks_training)

#: What a mismatch at D or E must preserve. A stop that keeps no evidence turns a
#: scientific finding into an outage.
MISMATCH_EVIDENCE: tuple[str, ...] = (
    "every intermediate artifact_digest, in order",
    "DEPTH selected and removed blocks",
    "FFN selected neurons per layer",
    "WIDTH projection diagnostics",
    "ATTENTION kept heads per layer",
    "runtime: image digest, torch, transformers, CUDA runtime, driver, GPU",
    "the expected and realized digests, side by side",
)


def stage(letter: str) -> C1Stage:
    for s in C1_STAGES:
        if s.letter == letter:
            return s
    raise C1SessionError(f"no stage {letter!r}")


def assert_stage_order(completed: Sequence[str]) -> None:
    """Refuse an out-of-order or gapped execution.

    The ordering is the science: evaluating before all six probes exist, or
    deciding before evaluating, would each produce a number that looks like a
    result. A prefix is fine — a session may stop early, and at a gate it must.
    """
    expected = [s.stage_id for s in C1_STAGES]
    if list(completed) != expected[:len(completed)]:
        raise C1SessionError(
            f"stages ran out of order: got {list(completed)}, "
            f"which is not a prefix of {expected}")


def _target_spec() -> ArchSpec:
    return ArchSpec.of("qwen3", TARGET_GEOMETRY)


def build_arm_specs(*, workdir_device: str = "cuda") -> dict[str, FixedPathSpec]:
    """The two arms, sharing a pinned prefix. Refuses if stage C has not run.

    The incumbent arm carries BOTH digest pins — the parent on its third step and
    the incumbent on its fourth — so a single replay of that arm exercises the
    whole end-to-end gate. The treatment arm shares the identical prefix, and its
    parent pin is the same, so the two cannot silently diverge before ATTENTION.
    """
    from .operators.base import get_implementation

    try:
        get_implementation(TREATMENT_ATTENTION[0])
    except Exception as exc:                       # noqa: BLE001 - re-raised typed
        raise C1SessionError(
            f"{TREATMENT_ATTENTION[0]} is not registered. Stage C registers it "
            "explicitly; importing its module does not, because an unrestricted "
            "BeamSearch enumerates the whole registry. Run stage C first."
        ) from exc

    prefix = [FixedPathStep(impl, prof) for impl, prof in PREFIX_STEPS]
    prefix[-1] = FixedPathStep(
        PREFIX_STEPS[-1][0], PREFIX_STEPS[-1][1],
        expected_artifact_digest=EXPECTED_PARENT_DIGEST,
        label=f"pre-ATTENTION parent {EXPECTED_PARENT_STATE_ID[:12]}")

    common = dict(family="qwen3", target_spec=_target_spec(),
                  root_repo_id=TEACHER_REPO, root_revision=TEACHER_REVISION,
                  device=workdir_device, seed=SEARCH_SEED)
    incumbent = FixedPathSpec(
        path_id="autoinit.v1.phase_c1.incumbent",
        steps=(*prefix, FixedPathStep(
            *INCUMBENT_ATTENTION,
            expected_artifact_digest=EXPECTED_INCUMBENT_DIGEST,
            label=f"incumbent {EXPECTED_INCUMBENT_STATE_ID[:12]}")),
        **common)
    treatment = FixedPathSpec(
        path_id="autoinit.v1.phase_c1.treatment",
        steps=(*prefix, FixedPathStep(*TREATMENT_ATTENTION,
                                      label="treatment ATTENTION")),
        **common)
    return {"incumbent": incumbent, "treatment": treatment}


def arm_prefix_is_shared(arms: Mapping[str, FixedPathSpec]) -> bool:
    """Both arms must differ in the last step and nowhere else."""
    a, b = arms["incumbent"], arms["treatment"]
    return (a.steps[:-1] == b.steps[:-1]
            and a.steps[-1].impl_id != b.steps[-1].impl_id
            and a.target_spec.spec_hash == b.target_spec.spec_hash
            and a.root_revision == b.root_revision)


@dataclass(frozen=True)
class C1SessionContract:
    """The session's declared shape, hashable for the preregistration."""

    session_id: str = "autoinit.v1.phase_c1"
    stages: tuple[C1Stage, ...] = C1_STAGES
    n_arms: int = 2
    n_seeds: int = 3
    n_probes: int = 6
    battery_asset_id: str = "c1_confirmation_v1"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_arms * self.n_seeds != self.n_probes:
            raise C1SessionError(
                f"{self.n_arms} arms x {self.n_seeds} seeds is not {self.n_probes} "
                "probes; every arm runs every seed and none is eliminated")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "session_id": self.session_id,
            "stages": [s.as_dict() for s in self.stages],
            "gate_stages": list(GATE_STAGES),
            "mismatch_evidence": list(MISMATCH_EVIDENCE),
            "n_arms": self.n_arms, "n_seeds": self.n_seeds,
            "n_probes": self.n_probes,
            "battery_asset_id": self.battery_asset_id,
            "expected_parent_digest": EXPECTED_PARENT_DIGEST,
            "expected_incumbent_digest": EXPECTED_INCUMBENT_DIGEST,
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


C1_SESSION_CONTRACT = C1SessionContract()
