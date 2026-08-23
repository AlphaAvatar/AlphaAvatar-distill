#!/usr/bin/env python3
"""Pod-side driver for AutoInitializer Phase A. Six stages, then STOP.

    /opt/train/bin/python scripts/pod/autoinit_phase_a_driver.py \
        --image-digest <digest> --rate 0.99 --spent-usd 0.20 \
        --soft-stop-usd 18.00 --authorized-usd 20.13

    0  attestation and preregistration binding   CPU + engine probe, fails closed
    1  beam search over initialization paths     GPU; produces the searched leaves
    2  recovery rung 1 on seed sa                6 probes: 5 leaves + the control
    3  recovery rung 2 on seed sb                3 probes: 2 survivors + control
    4  conditional tie-break on seed sc          only for finalists inside the interval
    5  selection and report                      CPU

Four semantics are load-bearing and are implemented here rather than described.

**Only the initialization differs.** Every probe derives its config from the same
frozen recipe and may override exactly `PROBE_OVERRIDES` — run identity, output
path, the staged pack path, the seed, and `student_path`. Anything else differing
raises, because a probe whose learning rate moved is not measuring its
initialization.

**Resume is per probe, not per session.** Nine probes at ~71 min each is a long
run on hardware that has failed this project repeatedly. Each probe's result is
journalled the moment it is scored and is restored on resume only when the
student digest, the seed and the evaluation protocol hash all still match — the
same binding rule `BeamSearch._restore` applies to search states.

**The permanent controls are inputs.** This driver trains recovery probes; it
never retrains `preflight_ctl_r0860k_{sa,sb}`. The canonical control enters the
comparison as a frozen checkpoint injected by hash, so "the incumbent won" stays
a reachable conclusion.

**Phase A is a terminus.** There is no stage 6, `--stage` cannot name one, and
the authorization refuses a follow-on. A tie surviving seed sc is reported as
`unresolved_equivalence` and is a RESULT, not a condition to be resolved by a
fourth seed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from aadistill.autoinit.authorization import AuthorizationError  # noqa: E402
from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.device import apply_cpu_budget  # noqa: E402
from aadistill.autoinit.device_handoff import (  # noqa: E402
    DeviceHandoffError, complete_release, cuda_memory, require_headroom,
    require_released,
)
from aadistill.autoinit.leaf_durability import (  # noqa: E402
    LeafDurabilityError, persist_selected_leaves,
)
from aadistill.autoinit.generation import (  # noqa: E402
    RecoveryEvaluationProtocol, declared_generation_protocol,
    generation_source_digest, observe_generation_protocol,
)
from aadistill.autoinit.generation_compat import (  # noqa: E402
    ComparabilityError, comparable_generation_identity, require_comparable,
)
from aadistill.autoinit.phase_a import (  # noqa: E402
    PHASE_A_PLAN_V1, PHASE_A_SCOPE, PhaseAAuthorization, phase_a_manifest,
)
from aadistill.autoinit.recovery import (  # noqa: E402
    POOLED_COUNTS_V2, RecoveryAdmissionError, RuntimeEnvironmentFingerprint,
    admit_leaves, assert_preregistered, probe_configs, recovery_scoring_contract,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

WS = Path("/workspace")
STATUS = WS / "autoinit_phase_a.status"
AUDIT = REPO / "artifacts/audit/autoinit_phase_a"
SEARCH_WORKDIR = REPO / "artifacts/autoinit/phase_a_search"


def trained_model_dir(out_dir: Path) -> Path:
    """The model a just-finished probe wrote, resolved the way the trainer writes it.

    `Trainer.save_checkpoint` (`src/aadistill/training/train.py`) writes
    `out_dir/checkpoints/latest.txt` holding a tag, and the checkpoint itself at
    `out_dir/checkpoints/<tag>/model`. The trainer entrypoint's own resume path
    reads exactly that and always has.

    (That entrypoint is deliberately not named in full here: a rehearsal test
    scans this file for the first occurrence of each invoked script path and
    reads the flags that follow it, so a mention in prose above the real call
    site would hide the arguments it checks.)

    This consumer did not: until 2026-08-23 it read `out_dir/latest.txt` and
    `out_dir/<tag>/model`, dropping the `checkpoints/` component in both.
    Recovery continuation attempt 5 trained a rung-1 probe for 61.7 minutes and
    then died here, and that training was lost with the pod.

    A `$0` path **does** execute this line — `test_phase_a_stages1_5_execute.py`
    drives the whole stage end to end — and it certified the defect anyway,
    because its fake trainer wrote the layout this consumer expected rather than
    the one the trainer writes. The fake now emits the real tree, so the harness
    is sensitive to this class; attempt 6 then confirmed the fix on hardware.

    Both failures are named rather than left to a bare `FileNotFoundError`,
    because at this point a probe has already been paid for and the distinction
    between "the trainer wrote nothing" and "the tag points nowhere" is the
    whole diagnosis.
    """
    ckpt_root = out_dir / "checkpoints"
    latest = ckpt_root / "latest.txt"
    if not latest.is_file():
        raise RecoveryAdmissionError(
            f"the probe finished but {latest} does not exist; the trainer writes "
            f"its checkpoint index under {ckpt_root}, so either training wrote no "
            "checkpoint or the output directory is not the one it was given")
    tag = latest.read_text().strip()
    model_dir = ckpt_root / tag / "model"
    if not model_dir.is_dir():
        raise RecoveryAdmissionError(
            f"{latest} names tag {tag!r} but {model_dir} is not a directory; the "
            "checkpoint index and the checkpoint tree disagree")
    return model_dir


def selected_leaf_dir() -> Path:
    """Where the five SELECTED leaves are preserved at the stage-1/2 boundary.

    Under `AUDIT` because collection walks that tree on EVERY path including the
    failure path — precisely what attempt 11 needed and the stage-5 fetch route
    could not give it.

    DERIVED at call time, not a module constant. As a constant computed from
    `REPO` it ignored the rehearsal's `mod.AUDIT = tmp_path` redirection and
    wrote seven real leaf directories into the repository's own artifact tree,
    193 MB, on a box that was already out of disk. A test that redirects the
    audit root must redirect everything written under it.
    """
    return AUDIT / "selected_leaves"
BATTERY = REPO / "artifacts/stage3/recovery_search_v2"
STATE_EVAL = REPO / "artifacts/stage1/state_eval_v1"
#: The engine probe's target at stage 0. The canonical initialization is the only
#: target-architecture checkpoint that exists before the search runs, and it is
#: what the continuation probed, so the generation identity stays comparable.
CANONICAL_INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
FROZEN_PLAN = REPO / "logs/autoinit_phase_a_recovery_plan_frozen.json"
#: The Stage-3 controls materialized this run's equivalence interval and
#: feasibility floor under ONE evaluation protocol. Phase A must measure
#: under the same one or the thresholds do not describe its candidates.
STAGE3_THRESHOLDS = REPO / "logs/autoinit_stage3_complete/materialized_thresholds.json"
STAGE3_EVALUATION_PROTOCOL_HASH = (
    "250f72efbd43b86a475e8dda293b45f07ee61a4d858e147f4a5bd7681c32c2e4")
STAGE3_ATTESTATION = REPO / "logs/autoinit_stage3_complete/attested_evaluation_protocol.json"
STAGE3_PROBE = REPO / "logs/autoinit_stage3_complete/engine_probe.json"
#: What the stage-2 recovery trainer needs on the device. DERIVED FROM A
#: MEASUREMENT, not chosen: the full basis is recorded in
#: `logs/autoinit_recovery_trainer_memory_basis.json` and a test pins these terms
#: against it.
#:
#: The old value was `22 * 2**30`, attributed to attempt 12's OOM — a subprocess
#: that "had 17.97 GiB in use when it asked for 3.58 more", rounded up. That is a
#: **lower bound observed mid-failure**, not a requirement, and it was 17.79 GiB
#: below the trainer's real peak. So `require_headroom` passed on recovery
#: continuation attempt 4 with 12.32 GiB of apparent slack and the probe OOM'd.
#:
#: The peak comes from `preflight_ctl_r0860k_{sa,sb}`, the two permanent
#: controls: the same frozen recipe in every memory-relevant field, on an L40S,
#: each reporting `torch.cuda.max_memory_allocated()` of **39.79 GiB** over a
#: COMPLETED 1023-step run — so gradients and both AdamW moments are included,
#: which attempt 4's pre-backward footprint could not be.
RECOVERY_TRAINER_PEAK_ALLOCATED_GIB = 39.79
#: `max_memory_allocated` counts PyTorch's live tensors; `require_headroom`
#: compares against the driver's free bytes. These two convert one into the
#: other, both read off attempt 4's own OOM message, which decomposes the
#: trainer process exactly: 34.44 allocated + 1.35 reserved-but-unallocated =
#: 35.79 reserved, against 36.30 total including non-PyTorch memory.
RECOVERY_TRAINER_RESERVED_SLACK_GIB = 1.35
RECOVERY_TRAINER_NON_TORCH_GIB = 0.51
RECOVERY_TRAINER_BYTES = int(
    (RECOVERY_TRAINER_PEAK_ALLOCATED_GIB
     + RECOVERY_TRAINER_RESERVED_SLACK_GIB
     + RECOVERY_TRAINER_NON_TORCH_GIB) * 2**30)
#: The versioned comparability relation the live protocol is judged under.
COMPAT_V2 = REPO / "logs/autoinit_phase_a_protocol_compat_v2.json"
FROZEN_RECIPE = REPO / "configs/stage3/e1/e1_r0860k_sa_pca.json"
#: The preregistered `state_eval@v1` identity, as two hashes that bind two
#: different things. The manifest carries only the first.
#:
#:   * `content_sha256` — the ITEMS. This is what the preregistration pins under
#:     `data_hashes.initializer_state_eval`, and `verify_frozen_assets.py`
#:     re-derives it from the loaded items at stage 0.
#:   * the STRUCTURAL suite hash — suite_id/version/domains/subtypes/critical_tags,
#:     which is what `StateEvalSuite.required_metrics()` and therefore the beam
#:     ranking read. This is the value `run_phase_a_search` reports.
#:
#: They are pinned rather than re-read from the staged manifest because a check
#: that loads both sides from the same directory verifies nothing.
STATE_EVAL_CONTENT_SHA256 = (
    "a1197205e43aad0e71c0e1bb436ee7babba3b5d8bb25b9c4d5c464f659db20fc")
STATE_EVAL_SUITE_HASH = (
    "6421fa4cf12ee2a16f452557c486aa95beb37e4aac4f7c7fd72d380993b39833")
PACK_DIR = "artifacts/stage3/ladder_uniform_probe"
BATTERY_CONTENT = "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323"

# The teacher id, revision, target geometry and canonical-control hash live in
# `scripts/autoinit/phase_a_search.py`, which is the only module that needs them.
# Restating them here would be a second definition of the search's identity, and
# the two would eventually disagree about which teacher was measured.

#: The ONLY fields a probe may change relative to the frozen recipe. `seed` and
#: `student_path` are the two that carry the experiment: the seed is the
#: replicate, the initialization is the treatment. Everything else identical is
#: what makes the arms comparable at all.
PROBE_OVERRIDES = frozenset(
    {"run_name", "_purpose", "out_dir", "data_dir", "seed", "student_path"})


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


class PhaseADriver:
    #: WHICH artifact governs this run, and what type may govern it. Class
    #: attributes rather than a hard-coded path because the recovery
    #: continuation subclasses this driver: with the path fixed here, the
    #: continuation would have loaded attempt 12's CONSUMED Phase-A
    #: authorization on the pod — a real, committed file — and run its
    #: `require_within_cap` against $23.0484 instead of the $16.7456 its own
    #: budget derives. Not a crash; a silently wrong ceiling, and evidence
    #: naming the wrong grant.
    AUTHORIZATION_TYPE = PhaseAAuthorization
    AUTHORIZATION_PATH = "logs/autoinit_phase_a_authorization.json"

    def __init__(self, a):
        self.a = a
        self.t0 = time.time()
        self.results: dict[int, dict] = {}
        self.evaluation_protocol = None
        self.plan = None            # the frozen SuccessiveHalvingPlan
        self.search_result = None
        self.leaves: list = []
        self.control_state = None
        self.rung1 = None
        self.rung2 = None
        self.ev: dict = {
            "schema": "aadistill.autoinit.phase_a_evidence/v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "phase_a": phase_a_manifest(),
            "scope": PHASE_A_SCOPE.as_dict(),
            "retrains_permanent_controls": False,
            "followon_started": False,
            "followon_reachable_from_this_driver": False,
            "stages": {}}
        AUDIT.mkdir(parents=True, exist_ok=True)
        (AUDIT / "probes").mkdir(parents=True, exist_ok=True)
        self.auth = self.AUTHORIZATION_TYPE.load(REPO / self.AUTHORIZATION_PATH)
        self.auth.require_plan(PHASE_A_PLAN_V1.plan_hash)
        self.ev["authorization"] = self.auth.as_dict()

    # -- budget -----------------------------------------------------------
    def usd(self) -> float:
        return self.a.spent_usd + (time.time() - self.t0) / 3600 * self.a.rate

    def afford(self, minutes: float, what: str) -> bool:
        projected = self.usd() + minutes / 60 * self.a.rate
        if projected > self.a.soft_stop_usd:
            say(f"SOFT STOP: {what} needs ~{minutes:.0f} min "
                f"(${projected:.2f} > ${self.a.soft_stop_usd:.2f}) — not starting")
            return False
        try:
            self.auth.require_within_cap(projected, what=what)
        except AuthorizationError as exc:
            say(f"AUTHORIZATION: {exc}")
            return False
        return True

    def child_env(self) -> dict:
        return {**os.environ, "PYTHONPATH": f"{REPO}/src",
                "AADISTILL_IMAGE_DIGEST": self.a.image_digest}

    def save(self) -> None:
        self.ev["elapsed_min"] = round((time.time() - self.t0) / 60, 2)
        self.ev["spend_usd"] = round(self.usd(), 4)
        (AUDIT / "phase_a_evidence.json").write_text(
            json.dumps(self.ev, indent=2, default=str) + "\n")

    def record(self, stage: int, passed: bool, reason: str = "", **payload) -> bool:
        self.results[stage] = {"passed": passed, "reason": reason}
        self.ev["stages"][str(stage)] = {
            "passed": passed, "reason": reason,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "spend_usd": round(self.usd(), 4), **payload}
        mark(f"STAGE_{'PASSED' if passed else 'FAILED'}:{stage}")
        self.save()
        if not passed:
            say(f"STAGE {stage} FAILED: {reason}")
        return passed

    def preserve_traceback(self, stage: int, exc: BaseException) -> dict:
        """Keep the frame an UNEXPECTED in-process exception died in.

        Attempt 6 lost stage 1's frame. Stages that shell out keep a log tail;
        stages 1 and 5 run in-process, so all that survived was
        `f"{type(exc).__name__}: {exc}"`. The message happened to name
        `index_select`, which is how the call site was recovered — luck, not
        evidence.

        The short reason is unchanged and still what `record()` reports. The
        full traceback goes to `AUDIT/stage{n}_traceback.log`, which the session
        tars up with the rest of the audit directory, AND into the evidence JSON
        so it survives even if the archive does not.
        """
        text = traceback.format_exc()
        path = AUDIT / f"stage{stage}_traceback.log"
        try:
            path.write_text(
                f"stage {stage}: {type(exc).__name__}: {exc}\n"
                f"{datetime.now(timezone.utc):%FT%TZ}\n\n{text}")
        except OSError as write_failed:                           # noqa: BLE001
            # Never let evidence collection be the thing that fails the stage.
            return {"traceback": text[-6000:],
                    "traceback_file_error": str(write_failed)}
        return {"traceback": text[-6000:], "traceback_file": path.name}

    def enter(self, stage: int) -> None:
        self.auth.require_stage(stage)
        PHASE_A_PLAN_V1.advance_to(stage, self.results)
        mark(f"STAGE_START:{stage}")

    def gate(self, name: str, argv: list[str], *, timeout: float,
             python: str = "/opt/train/bin/python") -> subprocess.CompletedProcess:
        out = subprocess.run([python, *argv], capture_output=True, text=True,
                             timeout=timeout, env=self.child_env())
        (AUDIT / f"{name}.log").write_text(
            f"$ {' '.join(argv)}\nrc={out.returncode}\n--- stdout ---\n"
            f"{out.stdout}\n--- stderr ---\n{out.stderr}\n")
        return out

    # -- stage 0: attestation and preregistration binding -----------------
    def stage0(self) -> bool:
        self.enter(0)
        runtime = RuntimeEnvironmentFingerprint.observe(
            image_digest=self.a.image_digest)
        scoring = recovery_scoring_contract(REPO)

        assets = self.gate("frozen_assets",
                           [str(REPO / "scripts/autoinit/verify_frozen_assets.py")],
                           timeout=900)
        if assets.returncode != 0:
            return self.record(0, False,
                               f"frozen assets rc={assets.returncode}; tail: "
                               f"...{(assets.stdout + assets.stderr)[-1200:]}")

        # The science plan this run will select against must BE the frozen one.
        # `assert_preregistered` compares hashes, so a threshold that moved after
        # freezing is caught here rather than in the selection at hour eleven.
        try:
            from write_preregistration import build_frozen_plan
            self.plan = build_frozen_plan(REPO)
            frozen = assert_preregistered(self.plan, FROZEN_PLAN)
        except Exception as exc:                                  # noqa: BLE001
            return self.record(0, False,
                               f"preregistration binding: {type(exc).__name__}: "
                               f"{exc}"[-1500:],
                               **self.preserve_traceback(0, exc))
        try:
            self.auth.require_science_plan(self.plan.plan_hash)
        except AuthorizationError as exc:
            return self.record(0, False, str(exc)[-1500:])

        # Both thresholds must be materialized. A selection with a pending rule
        # cannot run, and discovering that after nine probes would waste the
        # whole session.
        try:
            interval = self.plan.equivalence.require_value()
            floor = self.plan.feasibility_floor()
        except RecoveryAdmissionError as exc:
            return self.record(0, False, f"thresholds not materialized: {exc}")

        # `--model` and `--image-digest` are NOT optional: the probe requires the
        # first and records the second into the generation identity. Omitting
        # `--model` cost attempt 2 a $0.47 pod, dying one second after the driver
        # detached, because the rehearsal scripted stage 0 instead of building
        # this argv.
        engine = self.gate("engine_probe",
                           [str(REPO / "scripts/pod/autoinit_engine_probe.py"),
                            "--model", str(CANONICAL_INIT),
                            "--out", str(AUDIT / "engine_probe.json"),
                            "--image-digest", self.a.image_digest],
                           timeout=1800, python="/opt/vllm/bin/python")
        if engine.returncode != 0:
            return self.record(0, False,
                               f"engine probe rc={engine.returncode}; tail: "
                               f"...{(engine.stdout + engine.stderr)[-1200:]}")

        # Ported verbatim in shape from the continuation's stage 1, which has run
        # green on hardware. Attempt 3 died here because this driver invented a
        # signature: `declared_generation_protocol()` takes no arguments, and the
        # protocol is built by materializing twice — first the source digests,
        # then the engine-observed fields — before `require_materialized`.
        observed = json.loads((AUDIT / "engine_probe.json").read_text())
        try:
            gen = declared_generation_protocol().materialized(
                generation_source_digest=generation_source_digest(REPO)["digest"],
                degeneration_source_digest=sha256_file(
                    REPO / "src/aadistill/evaluation/degeneration.py"))
            gen = gen.materialized(
                vllm_version=observed["vllm_version"],
                transformers_version=observed["transformers_version"],
                torch_version=observed["torch_version"],
                runtime_digest=observed["runtime_digest"],
                dtype=observed["dtype"],
                gpu_memory_utilization=observed["gpu_memory_utilization"],
                max_num_seqs=observed["max_num_seqs"],
                max_num_batched_tokens=observed["max_num_batched_tokens"],
                enforce_eager=observed["enforce_eager"],
                tokenizer_sha256=observed["tokenizer_sha256"],
                chat_template_sha256=observed["chat_template_sha256"],
                resolved_context=observed["resolved_context"],
                context_source=observed["context_source"],
                stop_token_ids=tuple(observed["stop_token_ids"]))
            gen.require_materialized(context="phase A stage 0")
        except Exception as exc:                                  # noqa: BLE001
            return self.record(0, False,
                               f"generation protocol: {type(exc).__name__}: "
                               f"{exc}"[-1500:],
                               **self.preserve_traceback(0, exc))

        battery_manifest = json.loads((BATTERY / "manifest.json").read_text())
        if battery_manifest.get("content_sha256") != BATTERY_CONTENT:
            return self.record(0, False, "the battery does not verify")

        self.evaluation_protocol = RecoveryEvaluationProtocol(
            generation=gen,
            scoring_contract=scoring["contract"], scoring_digest=scoring["digest"],
            battery_artifact=battery_manifest["artifact"],
            battery_manifest_sha256=battery_manifest["manifest_sha256"],
            battery_content_sha256=battery_manifest["content_sha256"])

        # THE SCIENTIFIC BINDING. The selection thresholds this run applies were
        # materialized by the Stage-3 controls under one specific evaluation
        # protocol. Candidates being mutually consistent with *this* session's
        # attestation is NOT sufficient: the equivalence interval and the
        # feasibility floor are numbers that came from those controls, so a
        # candidate measured under any other protocol is being judged against
        # thresholds that do not describe it.
        stage3 = json.loads(STAGE3_THRESHOLDS.read_text())
        stage3_hash = stage3.get("evaluation_protocol_hash")
        if stage3_hash != STAGE3_EVALUATION_PROTOCOL_HASH:
            return self.record(
                0, False,
                f"the Stage-3 thresholds artifact declares evaluation protocol "
                f"{stage3_hash}, not the pinned "
                f"{STAGE3_EVALUATION_PROTOCOL_HASH}; the thresholds this run "
                "would select against are not the ones that were characterized")
        observed_hash = self.evaluation_protocol.evaluation_protocol_hash

        # Comparability is decided under generation_runtime_comparability@v2,
        # not by exact equality of the v1 protocol hash. v1 fused the container
        # image with the HOST NVIDIA DRIVER inside `runtime_digest`, so exact
        # equality made this a host lottery: attempt 4 was refused at $0.2052
        # with every generation-semantic field identical and only the driver
        # patch different. v2 keeps every one of those fields material, demotes
        # the driver patch to recorded provenance, and still fails closed on a
        # driver BRANCH change. The historical protocol is NOT rewritten — it is
        # loaded from the untouched Stage-3 attestation and compared against.
        compat = json.loads(COMPAT_V2.read_text())
        if compat["bound_to_historical_protocol"]["evaluation_protocol_hash"] \
                != stage3_hash:
            return self.record(
                0, False,
                "the v2 compatibility artifact is bound to "
                f"{compat['bound_to_historical_protocol']['evaluation_protocol_hash']}, "
                f"not to the Stage-3 protocol {stage3_hash} the thresholds came "
                "from")
        s3_att = json.loads(STAGE3_ATTESTATION.read_text())
        if s3_att["evaluation_protocol_hash"] != stage3_hash:
            return self.record(
                0, False,
                f"the Stage-3 attestation declares "
                f"{s3_att['evaluation_protocol_hash']}, not {stage3_hash}")
        try:
            historical = comparable_generation_identity(
                protocol=s3_att["evaluation_protocol"],
                runtime=json.loads(STAGE3_PROBE.read_text())["runtime"])
            live = comparable_generation_identity(
                protocol=self.evaluation_protocol.as_dict(),
                runtime=observed["runtime"],
                host_provenance={"image_digest_arg": self.a.image_digest})
            comparison = require_comparable(
                live, historical, context="phase A stage 0")
        except ComparabilityError as exc:
            return self.record(
                0, False,
                f"{exc} The equivalence interval {interval!r} and feasibility "
                f"floor {floor!r} were materialized under the Stage-3 protocol "
                f"{stage3_hash}; candidates measured under a protocol that is "
                "not comparable to it cannot be judged against them."[-1500:])

        attested = {
            "schema": "aadistill.autoinit.phase_a_attested_protocol/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime.as_dict(),
            "generation_source_digest": generation_source_digest(REPO),
            "scoring_contract": scoring,
            "evaluation_protocol": self.evaluation_protocol.as_dict(),
            "evaluation_protocol_hash": observed_hash,
            "stage3_evaluation_protocol_hash": stage3_hash,
            "bound_to_stage3_thresholds": True,
            "comparability": comparison,
            "comparable_identity": live["comparable_identity"],
            "science_plan_hash": self.plan.plan_hash,
            "equivalence_interval": interval,
            "feasibility_floor": floor,
        }
        attested["report_sha256"] = sha256_json(attested)
        (AUDIT / "attested_evaluation_protocol.json").write_text(
            json.dumps(attested, indent=2) + "\n")
        say(f"attested: interval {interval:.6f}, floor {floor:.4f}, plan "
            f"{self.plan.plan_hash[:12]}…")
        return self.record(0, True, attested=attested,
                           frozen_plan_hash=frozen["plan_hash"])

    # -- stage 1: beam search ---------------------------------------------
    def stage1(self) -> bool:
        self.enter(1)
        if not self.afford(self.a.search_minutes, "beam search"):
            return self.record(1, False, "insufficient budget for the search")

        # In-process, not a subprocess. The rungs need the live
        # `InitializationState` objects: the journal is append-only evidence with
        # no `from_dict`, so a driver that reconstructed candidates from JSON
        # would have to skip `admit_leaves` — the one gate that refuses an
        # intermediate which cannot be a recovery candidate at all.
        from phase_a_search import run_phase_a_search

        # Hold torch to the CPUs the cgroup actually granted, before any heavy
        # work. Attempt 10 ran 192 threads on a 13-vCPU grant because torch sized
        # its pools from the 128 the container could see; the setup script has
        # computed this correctly since E8b and applied it to the test suite
        # only. Recorded, not just done, so the run says which limit bound.
        budget = apply_cpu_budget()
        say(f"cpu budget: {budget['threads']} threads "
            f"({budget['source']}, {budget['visible_cpus']} visible)")
        self.ev.setdefault("runtime", {})["cpu_budget"] = budget

        # The DERIVED deadline, not the base allowance. `--search-minutes` is
        # what stage 1 is expected to cost and is what `afford()` above checks;
        # `--search-deadline-minutes` is what stage 1 is PAID for — the base plus
        # both soft-stop reserves, computed by the launcher from the same
        # BudgetPlan that produces the dollar thresholds.
        #
        # These were the same number until 2026-08-20, which meant a search that
        # legitimately took the reference-cache fallback path — the risk the
        # 147.7683-minute reserve was bought for — would have been killed at 180
        # minutes with the reserve unspent, and the kill would have read as a
        # failed search rather than a deadline disagreeing with its own price.
        found = run_phase_a_search(
            workdir=SEARCH_WORKDIR, state_eval=STATE_EVAL,
            top_n=self.plan.searched_leaves, device="cuda", repo_root=REPO,
            search_minutes=self.a.search_deadline_minutes)
        (AUDIT / "search_result.json").write_text(
            json.dumps(found.summary, indent=2, default=str) + "\n")

        # This read used to be `manifest["suite_hash"]`. That key does not exist
        # in `state_eval_v1/manifest.json` and never has, so the line was a
        # guaranteed `KeyError` — raised AFTER the whole GPU beam search had
        # completed, which is the most expensive place in the session to fail.
        # It had never executed: every rehearsal scripted stage 1.
        staged = json.loads((STATE_EVAL / "manifest.json").read_text())
        if staged.get("content_sha256") != STATE_EVAL_CONTENT_SHA256:
            return self.record(
                1, False,
                f"the staged state_eval declares content "
                f"{str(staged.get('content_sha256'))[:12]}… but the "
                f"preregistration pins {STATE_EVAL_CONTENT_SHA256[:12]}…; the "
                "beam would be ranking on a different suite's questions")
        if found.summary["suite_hash"] != STATE_EVAL_SUITE_HASH:
            return self.record(
                1, False,
                f"the search ranked on suite structure "
                f"{found.summary['suite_hash'][:12]}… but the preregistered "
                f"state_eval@v1 is {STATE_EVAL_SUITE_HASH[:12]}…; the domains, "
                "sub-types or critical tags the ranking reads are not the "
                "attested ones")

        self.leaves = found.leaves
        self.control_state = found.control
        self.search_result = found.result
        if len(self.leaves) < self.plan.searched_leaves:
            return self.record(
                1, False,
                f"the search produced {len(self.leaves)} admissible leaves but "
                f"the plan asks for {self.plan.searched_leaves}. Reporting the "
                "shortfall rather than shrinking N.",
                n_admissible=len(self.leaves))
        mark(f"SEARCH_DONE:{len(self.leaves)}")

        # THE DURABILITY BOUNDARY. Attempt 11 spent 180.3 min producing five
        # valid selected leaves and then lost every checkpoint, because
        # persistence happened only after stage-5 selection and stage 2 failed
        # six seconds after stage 1 passed. Collection DOES run on the failure
        # path — attempt 11's manifest came home with rc=0 — so a leaf that is a
        # collected ARTIFACT survives a stage-2 failure, while a leaf that is
        # only a stage-5 fetch product does not.
        #
        # Weight-only and byte-identical: `artifact_digest` folds in
        # `tokenizer_sha256`, so adding tokenizer files here would move the
        # identity the search metrics hang on. The tokenizer is the separate
        # consumer dependency resolved in `train_stage3.py`.
        try:
            durability = persist_selected_leaves(
                leaves=[{"state_id": s.state_id,
                         "checkpoint_path": s.checkpoint_path,
                         "artifact_digest": s.artifact["artifact_digest"]
                         if isinstance(s.artifact, dict)
                         else s.artifact.artifact_digest,
                         "total_bytes": s.artifact["total_bytes"]
                         if isinstance(s.artifact, dict)
                         else s.artifact.total_bytes,
                         "num_parameters": s.num_parameters}
                        for s in self.leaves],
                destination=selected_leaf_dir(),
                adapter=get_adapter("qwen3"), spec=self.leaves[0].spec)
        except LeafDurabilityError as exc:
            return self.record(
                1, False,
                f"the search succeeded but its result could not be preserved: "
                f"{exc}")
        (AUDIT / "selected_leaf_durability.json").write_text(
            json.dumps(durability, indent=2, default=str) + "\n")
        say(f"selected leaves persisted: {durability['n_leaves']} at "
            f"{durability['required_bytes'] / 2**30:.2f} GiB, digests re-verified")

        # THE DEVICE HANDOFF. Attempt 12 died six seconds after this point:
        # the driver runs the search IN-PROCESS and still held 24.05 GiB when
        # stage 2 spawned train_stage3.py needing 17.97 GiB on a 44.39 GiB card.
        #
        # Deliberately AFTER durability: if the release or the headroom contract
        # fails, the five leaves are already off the pod and the search is not
        # lost. Ordering the other way would trade a completed 203-minute search
        # for a memory diagnostic.
        #
        # `found` is dropped here rather than left to fall out of scope, because
        # BeamSearch's closures capture the teacher, the primed evaluator and
        # the calibration; anything holding the result holds them too. The
        # states themselves carry metadata and checkpoint identities, not CUDA
        # models, so the record below is what says which of the two the 24 GiB
        # actually was.
        # Everything still needed from `found` is taken FIRST. The whole point
        # is that nothing survives this line holding the search's device state,
        # so a later read of `found` would either resurrect it or -- as the
        # first version of this block did -- raise NameError after a 203-minute
        # search had already succeeded.
        summary = found.summary
        self.search_result = summary                 # the serializable part only
        # The BEFORE snapshot is taken here, by the frame that owns `found`, and
        # the `del` happens BEFORE `complete_release` measures the result. A
        # callee cannot rebind this name, which is why it no longer pretends to.
        before = cuda_memory()
        del found
        handoff = complete_release(before)
        (AUDIT / "device_handoff.json").write_text(
            json.dumps(handoff, indent=2, default=str) + "\n")
        self.ev.setdefault("runtime", {})["device_handoff"] = handoff
        say(f"device handoff: {handoff.get('verdict', 'n/a')}")
        try:
            require_released(handoff, what="the stage-2 recovery trainer")
            require_headroom(handoff["after"],
                             need_bytes=RECOVERY_TRAINER_BYTES,
                             what="the stage-2 recovery trainer")
        except DeviceHandoffError as exc:
            return self.record(
                1, False,
                f"stage 1 succeeded and its leaves are preserved, but the card "
                f"cannot carry the recovery trainer: {exc}")

        return self.record(1, True,
                           durability=durability,
                           n_states=summary["summary"]["n_states"],
                           n_leaves=summary["summary"]["n_complete_leaves"],
                           n_resumed=len(summary["resumed_state_ids"]),
                           selected=[s.state_id for s in self.leaves],
                           control=summary["control"],
                           levels=summary["levels"])

    # -- probe machinery, shared by every rung ----------------------------
    def probe_config(self, descriptor: dict) -> Path:
        """Derive a probe config that differs ONLY in `PROBE_OVERRIDES`."""
        frozen = json.loads(FROZEN_RECIPE.read_text())
        name = descriptor["probe_id"]
        derived = {**frozen,
                   "run_name": name,
                   "out_dir": f"artifacts/stage3/phase_a/{name}",
                   "data_dir": PACK_DIR,
                   "seed": descriptor["seed"],
                   "student_path": descriptor["student_checkpoint"],
                   "_purpose": (
                       f"AutoInitializer Phase A rung {descriptor['rung']} probe. "
                       "Identical recovery; the only intended difference between "
                       f"probes is the initialization. Derived from {FROZEN_RECIPE.name} "
                       "by overriding run identity, pack path, seed and student_path.")}
        diff = sorted(k for k in set(frozen) | set(derived)
                      if frozen.get(k) != derived.get(k))
        if not set(diff) <= PROBE_OVERRIDES:
            raise RecoveryAdmissionError(
                f"{name}: the derived probe config differs from the frozen recipe "
                f"in {sorted(set(diff) - PROBE_OVERRIDES)}, outside the allowed "
                f"override set {sorted(PROBE_OVERRIDES)}")
        path = AUDIT / "configs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(derived, indent=2) + "\n")
        return path

    def restore_probe(self, descriptor: dict) -> dict | None:
        """Adopt a journalled probe only when it still describes THIS run.

        Same binding rule the search applies to a state: identity is not the
        probe id alone. A result measured against a different student, seed or
        evaluation protocol is not this probe's result.
        """
        path = AUDIT / "probes" / f"{descriptor['probe_id']}.json"
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        expected = {
            "student_artifact_digest": descriptor["student_artifact_digest"],
            "seed": descriptor["seed"],
            "evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
        }
        if any(record.get(k) != v for k, v in expected.items()):
            say(f"  {descriptor['probe_id']}: journal does not bind to this run "
                "(student/seed/protocol moved) — re-running")
            return None
        if not record.get("complete"):
            return None
        say(f"  {descriptor['probe_id']}: restored from the journal")
        record["resumed"] = True
        return record

    def run_probe(self, descriptor: dict) -> dict:
        """Train one probe, then evaluate it on the frozen battery."""
        name = descriptor["probe_id"]
        restored = self.restore_probe(descriptor)
        if restored is not None:
            return restored

        config = self.probe_config(descriptor)
        t = time.time()
        rc = subprocess.run(
            ["/opt/train/bin/python", str(REPO / "scripts/training/train_stage3.py"),
             "--config", str(config)],
            capture_output=True, text=True,
            timeout=int(self.a.probe_train_minutes * 60 * 2), env=self.child_env())
        train_minutes = (time.time() - t) / 60
        (AUDIT / f"{name}_train_tail.log").write_text((rc.stdout + rc.stderr)[-1500:])
        if rc.returncode != 0:
            raise RecoveryAdmissionError(
                f"{name}: training failed rc={rc.returncode}; tail: "
                f"...{(rc.stdout + rc.stderr)[-1200:]}")
        mark(f"PROBE_TRAINED:{name}")

        model_dir = trained_model_dir(REPO / f"artifacts/stage3/phase_a/{name}")
        result = self.battery(name, model_dir, descriptor["seed"])

        record = {
            **{k: descriptor[k] for k in
               ("probe_id", "rung", "state_id", "is_control", "seed",
                "student_artifact_digest")},
            "evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
            "train_minutes": round(train_minutes, 2),
            "battery_minutes": result.pop("_battery_minutes"),
            "result": result,
            "complete": True,
            "resumed": False,
        }
        (AUDIT / "probes" / f"{name}.json").write_text(
            json.dumps(record, indent=2) + "\n")
        mark(f"PROBE_SCORED:{name}")
        return record

    def battery(self, label: str, model_dir: Path, seed: int) -> dict:
        """Generation + scoring on recovery_search_v2, protocol-checked."""
        sets = json.loads((BATTERY / "manifest.json").read_text())["sets"]
        gen_dir = REPO / f"artifacts/eval/phase_a/{label}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        t = time.time()
        out = self.gate(f"{label}_generation",
                        [str(REPO / "scripts/evaluation/uncapped_eval.py"),
                         "--model", str(model_dir), "--label", label,
                         "--prompts", *[str(BATTERY / f"{s}.jsonl") for s in sets],
                         "--out-dir", str(gen_dir), "--diagnostics"],
                        timeout=int(self.a.probe_battery_minutes * 60 * 3),
                        python="/opt/vllm/bin/python")
        minutes = (time.time() - t) / 60
        if out.returncode != 0:
            raise RecoveryAdmissionError(
                f"{label}: generation rc={out.returncode}; tail: "
                f"...{(out.stdout + out.stderr)[-1200:]}")

        scored = AUDIT / f"{label}_recovery_search.json"
        rc = self.gate(f"{label}_scoring",
                       [str(REPO / "scripts/autoinit/score_recovery_search.py"),
                        "--generations", str(gen_dir), "--label", label,
                        "--seed", str(seed), "--out", str(scored),
                        "--per-sample", str(AUDIT / f"{label}_per_sample.jsonl")],
                       timeout=1800)
        if rc.returncode != 0:
            raise RecoveryAdmissionError(
                f"{label}: scoring rc={rc.returncode}; tail: "
                f"...{(rc.stdout + rc.stderr)[-1200:]}")
        result = json.loads(scored.read_text())

        observed = RecoveryEvaluationProtocol(
            generation=observe_generation_protocol(
                [json.loads(p.read_text()) for p in sorted(gen_dir.glob("*.json"))
                 if not p.name.endswith(".generations.jsonl")]).protocol,
            scoring_contract=result["scoring_contract"]["contract"],
            scoring_digest=result["scoring_contract"]["digest"],
            battery_artifact=result["battery"]["artifact"],
            battery_manifest_sha256=result["battery"]["manifest_sha256"],
            battery_content_sha256=result["battery"]["content_sha256"])
        observed.require_comparable(self.evaluation_protocol, context=label)
        result["evaluation_protocol_hash"] = observed.evaluation_protocol_hash
        result["_battery_minutes"] = round(minutes, 2)
        return result

    def selection_row(self, records: list[dict]) -> list[dict]:
        """Pooled rows, one per candidate, in the shape the plan's gates read."""
        by_state: dict[str, list[dict]] = {}
        for record in records:
            by_state.setdefault(record["state_id"], []).append(record)
        rows = []
        for state_id, group in by_state.items():
            per_seed = [{"seed": r["seed"],
                         **{k: r["result"][k] for k in
                            POOLED_COUNTS_V2.required_counts}} for r in group]
            pooled = POOLED_COUNTS_V2.pool(per_seed)
            rows.append({
                "state_id": state_id,
                "is_control": group[0]["is_control"],
                "seeds": [r["seed"] for r in group],
                "probe_ids": [r["probe_id"] for r in group],
                **pooled,
                # The catastrophic rule reads per-capability rates; they pool the
                # same way and must be present or the schema validation refuses.
                "per_capability": self.pool_capabilities(group),
            })
        return rows

    @staticmethod
    def pool_capabilities(group: list[dict]) -> dict:
        caps: dict[str, dict[str, int]] = {}
        for record in group:
            for cap, values in record["result"]["per_capability"].items():
                acc = caps.setdefault(cap, {"n": 0, "usable": 0})
                acc["n"] += values["n"]
                acc["usable"] += values["usable"]
        return {cap: {**v,
                      "usable_rollout_rate": (v["usable"] / v["n"]) if v["n"] else 0.0}
                for cap, v in caps.items()}

    def run_rung(self, stage: int, descriptors: list[dict], label: str) -> list[dict]:
        """Run every descriptor, journalling as it goes. Raises to fail the stage."""
        records = []
        for descriptor in descriptors:
            need = self.a.probe_train_minutes + self.a.probe_battery_minutes
            if self.restore_probe(descriptor) is None and not self.afford(
                    need, f"{label} probe {descriptor['probe_id']}"):
                raise RecoveryAdmissionError(
                    f"insufficient budget for {descriptor['probe_id']}; "
                    f"{len(records)}/{len(descriptors)} probes completed and "
                    "journalled")
            records.append(self.run_probe(descriptor))
            self.ev.setdefault("probes", []).append(
                {k: records[-1][k] for k in
                 ("probe_id", "rung", "seed", "is_control", "resumed")})
            self.save()
        return records

    # -- stage 2: rung 1 on seed sa ---------------------------------------
    def stage2(self) -> bool:
        self.enter(2)
        candidates = [*self.leaves, self.control_state]
        admit_leaves(candidates, self.plan)
        descriptors = probe_configs(candidates, self.plan, rung=1)
        try:
            records = self.run_rung(2, descriptors, "rung1")
        except RecoveryAdmissionError as exc:
            return self.record(2, False, str(exc)[-1500:])

        rows = self.selection_row(records)
        self.rung1 = self.plan.select_rung1_survivors(rows)
        (AUDIT / "rung1_selection.json").write_text(
            json.dumps(self.rung1, indent=2, default=str) + "\n")
        retention = self.emit_leaf_retention(records)
        say(f"rung 1: advancing {self.rung1['advancing']}")
        say(f"retention: {retention['n_advancing']}/{retention['n_leaves']} leaves "
            f"advance plus the control, {retention['n_rejected']} tombstoned")
        return self.record(2, True, selection=self.rung1,
                           n_probes=len(records), retention=retention)

    # -- leaf retention: what survives rung 1, and what is only recorded ----
    def emit_leaf_retention(self, records: list[dict]) -> dict:
        """The durability record for all five searched leaves, written the moment
        the sa survivor decision is materialized.

        The five leaves are **already** safe through this point: `BeamSearch`
        releases weights only for PRUNED INTERMEDIATE states, never for a
        complete leaf, so every leaf sits on the pod's container disk from the
        moment it is materialized until teardown. Nothing here deletes anything.

        What this decides is **permanent** retention. A rejected leaf does not
        earn permanent checkpoint storage — but it must never become
        unaccountable, so its artifact digest, its search lineage, its sa
        recovery evidence and the selection result that rejected it are recorded
        here in the shape `logs/checkpoint_tombstones.json` already uses. The
        checkpoint is expendable; the evidence is not.
        """
        advancing = set(self.rung1["advancing"])
        by_state = {r["state_id"]: r for r in records}
        entries = []
        for state in (*self.leaves, self.control_state):
            is_control = state.provenance == "retained_canonical"
            probe = by_state.get(state.state_id, {})
            keeps = is_control or state.state_id in advancing
            entries.append({
                "canonical_id": state.state_id,
                "provenance": state.provenance,
                "is_control": is_control,
                "advanced_to_rung2": state.state_id in advancing,
                # Identity and lineage: enough to say exactly what this was.
                "artifact_digest": state.artifact_digest,
                "weights_sha256": state.checkpoint_sha256,
                "search_lineage": state.path_label,
                "num_parameters": state.num_parameters,
                # The sa evidence that justified the decision.
                "sa_probe_id": probe.get("probe_id"),
                "sa_result": {k: probe.get("result", {}).get(k) for k in
                              ("usable_rollout_rate", "correct_overall",
                               "correct_given_usable", "n", "n_scorable")},
                "sa_evaluation_protocol_hash": probe.get(
                    "evaluation_protocol_hash"),
                # The decision itself.
                "selection_rule": self.rung1["rule"],
                "rejected_reason": next(
                    (e.get("reason") for e in self.rung1["all_exclusions"]
                     if e.get("state_id") == state.state_id), None),
                "retention_tier": ("TIER_1_ACTIVE_CANONICAL" if is_control
                                   else "TIER_2_RETAINED" if keeps
                                   else "TIER_4_DISPOSABLE"),
                "permanent_checkpoint_retained": keeps,
                "physically_present_on_pod_until_teardown": True,
                "size_human": f"{(state.num_parameters * 2) / 2**30:.2f} GiB",
            })
        retention = {
            "schema": "aadistill.autoinit.phase_a_leaf_retention/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "rung1_selection_materialized": True,
            "n_leaves": len(self.leaves),
            # Counted over the SEARCHED leaves only, so that
            # n_advancing + n_rejected == n_leaves. The control advances
            # unconditionally and is not one of the five; folding it in made
            # "3 leaves advance" the stage-2 log line for two leaves.
            "n_advancing": sum(1 for e in entries if not e["is_control"]
                               and e["advanced_to_rung2"]),
            "n_control_advancing": sum(1 for e in entries if e["is_control"]
                                       and e["advanced_to_rung2"]),
            "n_rejected": sum(1 for e in entries if not e["is_control"]
                              and not e["advanced_to_rung2"]),
            "entries": entries,
            "policy": (
                "all five searched leaves stay materialized on the pod through "
                "rung 1 and the materialization of the sa survivor decision; "
                "the two survivors and the control then stay through rung 2 and "
                "any conditional rung 3. Only the finalists earn permanent "
                "off-pod retention. A rejected leaf keeps its digest, lineage, "
                "sa evidence and rejection reason, and not its bytes."),
            "no_scientific_artifact_deleted": (
                "nothing is deleted here. Rejected leaves are simply not "
                "fetched off the pod, and the pod is destroyed at teardown "
                "either way."),
        }
        retention["report_sha256"] = sha256_json(retention)
        (AUDIT / "leaf_retention.json").write_text(
            json.dumps(retention, indent=2, default=str) + "\n")
        mark(f"LEAF_RETENTION:{retention['n_advancing']}+control")
        return retention

    # -- stage 3: rung 2 on seed sb ---------------------------------------
    def stage3(self) -> bool:
        self.enter(3)
        advancing = set(self.rung1["advancing"])
        candidates = [s for s in (*self.leaves, self.control_state)
                      if s.state_id in advancing]
        if not any(s.provenance == "retained_canonical" for s in candidates):
            return self.record(3, False,
                               "the canonical control did not advance to rung 2; "
                               "it advances unconditionally by construction, so "
                               "this is a defect rather than a result")
        descriptors = probe_configs(candidates, self.plan, rung=2)
        try:
            records = self.run_rung(3, descriptors, "rung2")
        except RecoveryAdmissionError as exc:
            return self.record(3, False, str(exc)[-1500:])

        rows = self.pooled_over_rungs()
        self.rung2 = self.plan.select_final_winner(rows)
        (AUDIT / "rung2_selection.json").write_text(
            json.dumps(self.rung2, indent=2, default=str) + "\n")
        say(f"after two seeds: {self.rung2['decision_status']}")
        return self.record(3, True, selection=self.rung2, n_probes=len(records))

    def pooled_over_rungs(self) -> list[dict]:
        """Pool every probe a finalist has, across all completed rungs."""
        finalists = set(self.rung1["advancing"])
        records = [json.loads(p.read_text())
                   for p in sorted((AUDIT / "probes").glob("*.json"))]
        return self.selection_row([r for r in records
                                   if r.get("complete")
                                   and r["state_id"] in finalists])

    # -- stage 4: conditional tie-break on seed sc ------------------------
    def stage4(self) -> bool:
        self.enter(4)
        if not self.rung2.get("needs_tie_break_seed"):
            say("no tie-break owed — finalists are separated by more than the "
                "equivalence interval, or already resolved")
            return self.record(4, True, ran=False,
                               reason_not_run=self.rung2["decision_status"])

        tied = set(self.rung2["tie_break_candidates"])
        candidates = [s for s in (*self.leaves, self.control_state)
                      if s.state_id in tied]
        # `probe_configs` indexes `plan.seeds`, which holds sa and sb only, and
        # refuses rung 3. Extending it would mean editing recovery.py, which is
        # inside the frozen `recovery_search_scoring@v2` source set — changing it
        # moves a digest pinned in the preregistration and in the attested
        # protocol. So the tie-break descriptors are built here, from the plan's
        # own `tie_break_seed`, and carry the same fields.
        descriptors = [
            {**d, "rung": 3, "seed": self.plan.tie_break_seed,
             "probe_id": d["probe_id"].replace(".rung2.", ".rung3.")
                          .replace(".sb", ".sc")}
            for d in probe_configs(candidates, self.plan, rung=2)]
        try:
            records = self.run_rung(4, descriptors, "rung3")
        except RecoveryAdmissionError as exc:
            return self.record(4, False, str(exc)[-1500:])
        say(f"tie-break: {len(records)} probes on seed {self.plan.tie_break_seed}")
        return self.record(4, True, ran=True, n_probes=len(records))

    # -- stage 5: selection and report ------------------------------------
    def stage5(self) -> bool:
        self.enter(5)
        rows = self.pooled_over_rungs()
        final = self.plan.select_final_winner(rows)
        report = {
            "schema": "aadistill.autoinit.phase_a_result/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "science_plan_hash": self.plan.plan_hash,
            "evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
            "equivalence_interval": self.plan.equivalence.require_value(),
            "feasibility_floor": self.plan.feasibility_floor(),
            "rung1_selection": self.rung1,
            "final_selection": final,
            "decision_status": final["decision_status"],
            "winner": final["winner"],
            "winner_is_control": final["winner_is_control"],
            "tie_break_ran": bool(self.results.get(4, {}).get("passed")
                                  and self.ev["stages"].get("4", {}).get("ran")),
            "pooled_rows": rows,
            # Reported on separate axes, never combined. `usable_rollout` is blind
            # to correctness by construction, which is why it gates and does not
            # rank.
            "axes": {
                "behaviour": "usable_rollout_rate, with every component rate",
                "capability": "correct_overall over scorable prompts",
                "diagnostic": "correct_given_usable; reported, never reorders",
            },
            "capability_schema_enforced": final["capability_schema_enforced"],
            "no_followon": ("Phase A stops here. A winner does not authorize full "
                            "recovery, and unresolved_equivalence does not "
                            "authorize a fourth seed."),
        }
        report["report_sha256"] = sha256_json(report)
        (AUDIT / "phase_a_result.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")
        say(f"RESULT: {final['decision_status']} · winner={final['winner']} · "
            f"is_control={final['winner_is_control']}")
        return self.record(5, True, result=report)

    # -- run --------------------------------------------------------------
    def run(self) -> int:
        mark("DRIVER_START")
        stages = {0: self.stage0, 1: self.stage1, 2: self.stage2,
                  3: self.stage3, 4: self.stage4, 5: self.stage5}
        blocking = {s.stage for s in PHASE_A_PLAN_V1.stages if s.blocking}
        failed: list[int] = []
        for stage in sorted(stages):
            try:
                ok = stages[stage]()
            except (RecoveryAdmissionError, AuthorizationError) as exc:
                self.record(stage, False, f"refused: {exc}"[-1500:])
                ok = False
            except Exception as exc:                              # noqa: BLE001
                # Unexpected. `RecoveryAdmissionError`/`AuthorizationError`
                # above are refusals whose message IS the explanation; this
                # branch is a defect, and a defect without a frame cost this
                # project a paid session.
                self.record(stage, False, f"{type(exc).__name__}: {exc}"[-1500:],
                            **self.preserve_traceback(stage, exc))
                ok = False
            if ok:
                continue
            if stage in blocking:
                mark("PHASE_A_FAILED")
                say("stopping before any later stage; completed probes are "
                    "journalled and the permanent controls are untouched")
                self.finish(False, failed=[stage])
                return 20 + stage
            mark("STAGE_NONBLOCKING_FAIL")
            failed.append(stage)
        if failed:
            mark("PHASE_A_INCOMPLETE")
            say(f"Phase A INCOMPLETE: stage(s) {failed} failed — ${self.usd():.2f}")
            self.finish(False, failed=failed)
            return 20 + failed[0]
        mark("ALL_DONE")
        say(f"Phase A complete — ${self.usd():.2f}. STOP for review; no follow-on "
            "experiment is reachable from this driver.")
        self.finish(True, failed=[])
        return 0

    def finish(self, success: bool, *, failed: list[int]) -> None:
        self.ev["phase_a_successful"] = success
        self.ev["failed_stages"] = failed
        self.ev["outcome"] = "SUCCESS" if success else (
            "INCOMPLETE" if failed and all(s not in
                                           {x.stage for x in PHASE_A_PLAN_V1.stages
                                            if x.blocking} for s in failed)
            else "FAILED")
        self.ev["cleanup_is_not_success"] = (
            "collection and teardown run regardless of the outcome; a clean "
            "cleanup does not make a failed search a successful Phase A")
        self.ev["retrains_permanent_controls"] = False
        self.ev["followon_started"] = False
        self.ev["followon_reachable_from_this_driver"] = False
        self.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    # There is no stage 6. Phase A is a terminus.
    ap.add_argument("--stage", default="all", choices=("all",))
    ap.add_argument("--image-digest", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    # Both REQUIRED, with no defaults. The launcher is the single owner of
    # both numbers; a default here would be a second copy of a pricing constant
    # that could drift from the one that actually books the money.
    ap.add_argument("--search-minutes", type=float, required=True,
                    help="base search allowance; funds the affordability check")
    ap.add_argument("--search-deadline-minutes", type=float, required=True,
                    help="base allowance PLUS the stage-1 soft-stop reserves, "
                         "derived by the launcher from the priced envelope; "
                         "this is what actually bounds the search at runtime")
    ap.add_argument("--probe-train-minutes", type=float, default=62.0)
    ap.add_argument("--probe-battery-minutes", type=float, default=10.0)
    args = ap.parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    driver = PhaseADriver(args)
    driver.auth.require_within_cap(args.authorized_usd, what="session backstop")
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
