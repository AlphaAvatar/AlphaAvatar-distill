"""A3's frozen session identity: one path, two executions, three probes.

**One FixedPathSpec, and that is the whole point.** A-bsz1 and A-bsz3 are the
same operator at the same step with the same hashed config, because both
batching knobs live on `ExecutionConfig` and neither enters a hash. So A3 does
not build two arm specs — it builds ONE and materializes it twice under two
`ExecutionConfig`s:

    A_bsz1   micro_batch_size=1, original_order_v1   GATE: 53e30566… (incumbent)
    A_bsz3   micro_batch_size=3, length_sorted_v1    NO GATE: this is measured

Pinning a digest on the A-bsz3 run would answer the experiment's question by
assertion. Pinning one on the A-bsz1 run is mandatory: A-bsz1 **is** canonical
A, and if it does not rebuild the incumbent then the attempt75 controls A3
reuses do not describe what this session built.

**The teacher pin, the prefix and the target geometry are IMPORTED from C3's
session module, never restated.** They are the same frozen objects; a second
copy is a second thing to keep in step, and this repository has already had a
plan assert a seed the computation did not use.

**Stage I is not here, and that is deliberate.** attempt75 trained, preserved
and scored all nine of its probes and then lost its decision artifact to a
stage-I crash on the pod. A3's comparison runs OFF POD at `$0` from preserved
evidence, in `scripts/autoinit/aggregate_a3.py`. The driver's job ends at
evaluation and preservation, which removes that failure class instead of
guarding against it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from aadistill.initialization.planning.fixed_path import (  # noqa: E402
    FixedPathSpec, FixedPathStep,
)
from experiments.phase_c3 import session as CS  # noqa: E402
from experiments.phase_a3.a_bsz3 import (  # noqa: E402
    A_BSZ1, A_BSZ3, ATTENTION_IMPL_ID, frozen_identities,
)

EXPERIMENT_ID = "phase_a3"
DESIGN_PATH = "logs/stages/stage-1/phase_c3/plans/a3_design.json"
PRICING_PATH = "logs/stages/stage-1/phase_c3/plans/a3_pricing.json"

#: The pod's own paths. ONE status path, named once. The C3 driver wrote its
#: markers to `autoinit_c3.status` while the C3 launcher polled
#: `autoinit_c1.status`, so every driver marker -- `ALL_DONE` included -- was
#: invisible, and the acquisition loop would have read a completed formal run
#: as "no measurement began" and launched a second one.
POD_WORKSPACE = "/workspace"
STATUS_PATH = f"{POD_WORKSPACE}/autoinit_a3.status"
RUN_LOG_PATH = f"{POD_WORKSPACE}/autoinit_a3_run.log"
DRIVER_JOB_ID = "autoinit_a3"

#: The two executions under comparison, and which one carries the gate.
PROTOCOL_EXECUTIONS = {"A_bsz1": A_BSZ1, "A_bsz3": A_BSZ3}
REFERENCE_PROTOCOL = "A_bsz1"
TREATMENT_PROTOCOL = "A_bsz3"

#: Interleaved rounds per protocol, one of them a declared warm-up. The
#: operator measures 11.2732 s on this parent geometry, so a single sample run
#: after the other protocol reports the arms' order as much as the protocols.
DIAGNOSTIC_ROUNDS = 4

#: The treatment arm trains at these seeds and the controls are NOT retrained.
TREATMENT_ARM = "A_bsz3"


class A3SessionError(RuntimeError):
    """A3's frozen session identity cannot be established as declared."""


# --- the frozen design ----------------------------------------------------


def design(repo_root: str | Path = REPO) -> dict[str, Any]:
    """The frozen A3 design. READ and hash-verified, never assumed."""
    p = Path(repo_root) / DESIGN_PATH
    if not p.is_file():
        raise A3SessionError(
            f"no A3 design at {DESIGN_PATH}; run "
            "scripts/autoinit/write_a3_design.py --write")
    doc = json.loads(p.read_text())
    claimed = doc.get("design_sha256")
    if not claimed:
        raise A3SessionError(f"{DESIGN_PATH} carries no design_sha256")
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual != claimed:
        raise A3SessionError(
            f"{DESIGN_PATH} does not hash to its own design_sha256 "
            f"({actual[:12]} vs {claimed[:12]}); it has been edited since it "
            "was derived and a session may not bind an edited design")
    return doc


def design_hash(repo_root: str | Path = REPO) -> str:
    return design(repo_root)["design_sha256"]


def recovery_seeds(repo_root: str | Path = REPO) -> tuple[int, ...]:
    """The three frozen C3 recovery seeds, from the design that binds them."""
    seeds = tuple(int(s) for s in design(repo_root)["recovery"]["seeds"])
    frozen = tuple(frozen_identities()["recovery_seeds"])
    if seeds != frozen:
        raise A3SessionError(
            f"the A3 design declares seeds {seeds} and the C3 "
            f"preregistration declares {frozen}; A3 reuses attempt75's "
            "controls and the seeds must be the same three")
    return seeds


def expected_incumbent_digest() -> str:
    return frozen_identities()["incumbent_artifact_digest"]


def expected_parent_digest() -> str:
    return frozen_identities()["shared_parent_artifact_digest"]


# --- the one path, and the two executions ---------------------------------


def incumbent_attention_step() -> tuple[str, str]:
    """`(impl_id, profile_id)` of the FROZEN incumbent's ATTENTION step.

    From `arms.A_incumbent.attention` in C3's preregistration, which is the
    document that froze what the incumbent IS. A3 reuses attempt75's controls
    as evidence, so the initialization its reference protocol rebuilds has to
    be the one those controls were trained from -- and the only safe way to
    say that is to read it from the same place attempt75 did.
    """
    arm = (CS.preregistration().get("arms") or {}).get("A_incumbent") or {}
    pair = arm.get("attention")
    if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
        raise A3SessionError(
            "the frozen C3 preregistration's A_incumbent arm does not name an "
            f"(impl_id, profile_id) ATTENTION pair; got {pair!r}")
    return str(pair[0]), str(pair[1])


def path_spec(*, workdir_device: str = "cuda") -> FixedPathSpec:
    """A3's single fixed path: the frozen prefix, then ATTENTION.

    Built from C3's own prefix and target geometry so the parent this session
    replays is the parent attempt75 replayed. The ATTENTION step carries NO
    expected digest: the gate belongs to the A-bsz1 RUN, not to the path,
    because the same path produces both protocols' artifacts.
    """
    steps = CS.prefix_steps()
    parent = expected_parent_digest()
    prefix = [FixedPathStep(impl, prof) for impl, prof in steps]
    prefix[-1] = FixedPathStep(
        steps[-1][0], steps[-1][1], expected_artifact_digest=parent,
        label=f"pre-ATTENTION parent {parent[:12]}")
    #: THE ATTENTION STEP'S PROFILE COMES FROM THE FROZEN ARM, not from the
    #: prefix. It was `steps[-1][1]` -- the profile of the step BEFORE it,
    #: `width.global_pca_v0`'s `calib.reasoning_heavy@v2` -- while the
    #: incumbent's ATTENTION step uses `calib.domain_balanced@v1`. A different
    #: calibration mixture gives different activation statistics, a different
    #: head map and different weights, so A-bsz1 built `7fbfadd0f0f6` instead
    #: of the frozen `53e30566c5f7` and stage D stopped the chain at `$0.65`.
    #:
    #: The asymmetric digest gate caught it, which is exactly what it is for:
    #: without it this session would have trained three probes against an
    #: initialization the attempt75 controls do not describe, evaluated them,
    #: and reported a clean paired comparison of two things that were never
    #: comparable.
    #:
    #: Derived from `arms.A_incumbent.attention` in the frozen C3
    #: preregistration -- one fact, one owner -- so it cannot drift from the
    #: thing it has to equal.
    attention_impl, attention_profile = incumbent_attention_step()
    if attention_impl != ATTENTION_IMPL_ID:
        raise A3SessionError(
            f"the frozen incumbent arm names {attention_impl!r} and this "
            f"session is built on {ATTENTION_IMPL_ID!r}")
    return FixedPathSpec(
        path_id=f"autoinit.v1.{EXPERIMENT_ID}.{TREATMENT_ARM}",
        steps=(*prefix, FixedPathStep(
            attention_impl, attention_profile,
            label=f"ATTENTION {attention_impl}")),
        family="qwen3", target_spec=CS._target_spec(),
        root_repo_id=CS.TEACHER_REPO, root_revision=CS.TEACHER_REVISION,
        device=workdir_device, seed=CS.SEARCH_SEED)


def protocol_gate(protocol: str) -> str | None:
    """The digest a protocol's run must reproduce, or None if it is measured.

    The asymmetry is the experiment. A-bsz1 must rebuild the incumbent; what
    A-bsz3 builds is the finding, and gating it would decide the question by
    assertion.
    """
    if protocol not in PROTOCOL_EXECUTIONS:
        raise A3SessionError(f"unknown protocol {protocol!r}")
    return expected_incumbent_digest() if protocol == REFERENCE_PROTOCOL else None


# --- the stage ladder -----------------------------------------------------


@dataclass(frozen=True)
class A3Stage:
    letter: str
    stage_id: str
    what: str
    blocking: bool


STAGES: tuple[A3Stage, ...] = (
    A3Stage("B", "teacher_fetch_verify",
            "fetch the pinned teacher and verify every shard", True),
    A3Stage("C", "register_operator",
            "register the incumbent ATTENTION implementation", True),
    A3Stage("D", "parent_replay",
            "replay DEPTH -> FFN -> WIDTH under the parent digest gate", True),
    A3Stage("E", "diagnostics",
            "interleaved A_bsz1/A_bsz3 rounds; A_bsz1 gated on the incumbent",
            True),
    A3Stage("F", "recovery_probes",
            "three A_bsz3 recovery trainings; nothing is evaluated here", True),
    A3Stage("G", "evaluations",
            "three confirmation evaluations, once each", True),
    A3Stage("H", "preserve_and_package",
            "durable preservation and the evidence manifest", True),
)
#: Stage I is ABSENT ON PURPOSE -- see the module docstring. The comparison is
#: `scripts/autoinit/aggregate_a3.py`, off pod, at $0.
AGGREGATION_IS_OFF_POD = True

STAGE_LETTERS = tuple(s.letter for s in STAGES)


def stage(letter: str) -> A3Stage:
    for s in STAGES:
        if s.letter == letter:
            return s
    raise A3SessionError(f"no A3 stage {letter!r}; have {STAGE_LETTERS}")


#: The ladder a RESUME-AT-SCORING session executes. DECLARED, not derived by
#: subtraction, so the shorter sequence is as reviewable as the full one and
#: an arbitrary reordering is still refused.
#:
#: D replays the parent and E rebuilds both initialization protocols. Their
#: only consumer is the training in F, so a session that restores three
#: already-trained, already-durable probes reads neither -- which is what
#: AGENTS.md P8.4 state 2 prescribes and what `a3_attempt36` spent $3.39 not
#: doing. B, C, G and H are unchanged: the teacher and battery are still
#: verified, the operator still registered, the probes still attested,
#: admitted, scored and packaged.
RESUME_STAGE_LETTERS: tuple[str, ...] = ("B", "C", "F", "G", "H")


def assert_stage_order(completed: list[str],
                       ladder: tuple[str, ...] | None = None) -> None:
    """Stages run in order and none is skipped, within the DECLARED ladder.

    `ladder` names which of the two declared sequences is executing. It is a
    parameter rather than an inference because "which stages may be absent"
    is a scientific statement: the full chain owes a parent replay and both
    initializations, and the resume chain cites them from the attempt whose
    probes it restores.
    """
    letters = tuple(ladder) if ladder is not None else STAGE_LETTERS
    if letters not in (STAGE_LETTERS, RESUME_STAGE_LETTERS):
        raise A3SessionError(
            f"{letters} is not a declared A3 ladder; the declared ones are "
            f"{STAGE_LETTERS} and {RESUME_STAGE_LETTERS}")
    want = list(letters)[:len(completed)]
    if completed != want:
        raise A3SessionError(
            f"stages ran out of order: completed {completed}, expected {want}")


# --- what the session promises --------------------------------------------


@dataclass(frozen=True)
class A3SessionContract:
    """The identity a launcher binds and an authorization certifies."""

    experiment_id: str = EXPERIMENT_ID
    treatment_arm: str = TREATMENT_ARM
    n_probes: int = 3
    n_arms: int = 1
    diagnostic_rounds: int = DIAGNOSTIC_ROUNDS
    aggregation_off_pod: bool = AGGREGATION_IS_OFF_POD

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "treatment_arm": self.treatment_arm,
            "n_arms": self.n_arms, "n_probes": self.n_probes,
            "seeds": list(recovery_seeds()),
            "diagnostic_rounds": self.diagnostic_rounds,
            "protocols": sorted(PROTOCOL_EXECUTIONS),
            "reference_protocol": REFERENCE_PROTOCOL,
            "treatment_protocol": TREATMENT_PROTOCOL,
            "gated_protocol": REFERENCE_PROTOCOL,
            "measured_protocol": TREATMENT_PROTOCOL,
            "expected_parent_digest": expected_parent_digest(),
            "expected_incumbent_digest": expected_incumbent_digest(),
            "controls_retrained": False,
            "aggregation_off_pod": self.aggregation_off_pod,
            "stages": [{"letter": s.letter, "stage_id": s.stage_id,
                        "what": s.what, "blocking": s.blocking}
                       for s in STAGES],
            "design_sha256": design_hash(),
            "status_path": STATUS_PATH,
            "_one_status_path": (
                "named once. A driver writing markers to a path the launcher "
                "does not poll made every marker invisible, and the "
                "acquisition loop would have read a completed formal run as "
                "'no measurement began'."),
        }

    @property
    def contract_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()


A3_SESSION_CONTRACT = A3SessionContract()


def probe_id(seed: int) -> str:
    return f"autoinit.v1.{EXPERIMENT_ID}.{TREATMENT_ARM}.{seed}"


def probe_ids() -> tuple[str, ...]:
    return tuple(probe_id(s) for s in recovery_seeds())


# --- host comparability ---------------------------------------------------
#
# A3 reuses attempt75's controls, so a host property that makes this session's
# generations incomparable to theirs decides the experiment's validity. The
# repository already declares the rule: `generation_compat` treats an NVIDIA
# driver PATCH within a branch as provenance and a BRANCH change as a real
# runtime event. attempt75 ran on 580.159.03; a3_attempt35 landed on 595.91.07
# and the protocol admission refused it -- correctly, at `$4.33`, after three
# probes had trained and the first had generated.
#
# The required branch is DERIVED from the controls' own attested protocol, not
# written here, so it cannot drift from the thing it has to match.

CONTROL_ATTESTATION = (
    "/home/ecs-user/aad-artifacts/phase_c3/attempt75/audit/autoinit_c3/"
    "c3_attested_evaluation_protocol.json")


def control_driver_branch(attestation: str | Path = CONTROL_ATTESTATION
                          ) -> str | None:
    """The NVIDIA driver branch attempt75's controls were measured on."""
    from aadistill.initialization.planning.generation_compat import (
        driver_branch, split_image_identity,
    )

    p = Path(attestation)
    if not p.is_file():
        return None
    runtime = json.loads(p.read_text()).get("runtime") or {}
    return driver_branch(
        split_image_identity(runtime.get("image_digest")).get(
            "nvidia_driver_version"))


def host_admission(image_digest: str) -> tuple[bool, str]:
    """May A3 run on the host behind `image_digest`?

    Admits only a host whose driver branch matches the controls'. A patch
    within the branch is admitted, because that is exactly what
    `generation_compat` demotes to provenance -- and requiring the exact
    patch would make this unsatisfiable on nearly every host.

    Fails CLOSED when the controls' branch cannot be read: an unknown
    comparability premise is not a satisfied one.
    """
    from aadistill.initialization.planning.generation_compat import (
        driver_branch, split_image_identity,
    )

    want = control_driver_branch()
    if want is None:
        return False, (f"{CONTROL_ATTESTATION} is unreadable, so the driver "
                       "branch attempt75's controls were measured on cannot "
                       "be established; A3 will not generate against an "
                       "unknown comparability premise")
    split = split_image_identity(image_digest)
    got_version = split.get("nvidia_driver_version")
    got = driver_branch(got_version)
    if got is None:
        return False, (f"the image identity {image_digest!r} carries no "
                       "NVIDIA driver version, so this host's comparability "
                       "to the controls cannot be established")
    if got != want:
        return False, (
            f"driver branch {got} ({got_version}); attempt75's controls were "
            f"measured on branch {want}. `generation_compat` treats a branch "
            "change as a real runtime event rather than provenance, so this "
            "host's generations would not be comparable to the controls A3 "
            "reuses -- and the protocol admission would refuse them after "
            "three trainings. Redraw.")
    return True, (f"driver branch {want} ({got_version}) matches the "
                  "controls'; a patch within the branch is provenance")
