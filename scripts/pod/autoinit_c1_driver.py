#!/usr/bin/env python3
"""Phase C1: replay one frozen path under two digest gates, train six, then score.

    /opt/train/bin/python scripts/pod/autoinit_c1_driver.py \
        --image-digest <digest> --rate 0.99 --spent-usd 0.20 \
        --soft-stop-usd 13.4277 --authorized-usd 13.7578

    B  fetch the pinned teacher and verify every shard against the binding
    C  register attention.activation_importance_v1, and pre-flight the scorer
    D  replay DEPTH -> FFN -> RESIDUAL_WIDTH     GATE: parent    == eea90c91...
    E  apply attention.weight_proxy_v0           GATE: incumbent == c313d1b4...
    F  materialize both arms from the SAME verified parent, then release the card
    G  six 0.86M recovery trainings. NOTHING is evaluated here
    H  six confirmation evaluations, once each, on the C1 battery only
    I  the frozen paired decision, only after all six results exist

A and J belong to the session runner.

**This driver is C1's own.** It does not subclass the Phase-A driver and does not
import it. That inheritance was not a shortcut, it was a defect: `run()`
dispatched Phase A's numeric stages 0..5, so every C1 method was dead code; the
constructor bound Phase A's plan hash while the launcher bound the
C1IsolationPlan's; the inherited battery method read the wrong asset; and the
probe machinery required descriptor keys C1 has no meaning for. Three helpers
worth reusing — `mark`, `say`, `trained_model_dir` — are ~30 lines and are
restated here rather than importing a 1200-line operational driver to obtain them.

**G and H are physically separate, and that is load-bearing.** Phase A's probe
routine trains and immediately evaluates. If C1 did that, the first arm would
meet the confirmation battery before the last arm was trained, and "each probe is
evaluated exactly once on a battery no arm has seen" would stop being true of the
session as a whole. So stage G starts no evaluator and no scorer, and stage H
refuses to begin until six training completions exist.

**Two seams, and only two.** `train_one` and `generate_one` are the only
hardware-bound steps; the `$0` execution regression replaces exactly those and
runs everything else — the loops, the guards, the packaging, the real scorer and
the real decision — as production code.
"""

from __future__ import annotations

import argparse
import gc
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
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))

from aadistill.autoinit import c1_session as CS  # noqa: E402
from aadistill.autoinit.authorization import AuthorizationError  # noqa: E402
from aadistill.autoinit.c1_authorization import C1Authorization  # noqa: E402
from aadistill.autoinit.c1_isolation import (  # noqa: E402
    C0_PREREGISTRATION_SHA256, C1Arm, C1IsolationPlan, decide,
    derive_recovery_seeds, paired_differences, stratified_cluster_bootstrap,
)
from aadistill.autoinit.c1_packaging import build_evaluation_package  # noqa: E402
from aadistill.autoinit.c1_probe_results import (  # noqa: E402
    ARMS, C1ProbeRecord, build_probe_results, decision_inputs,
)
from aadistill.autoinit.c1_scoring import (  # noqa: E402
    C1_BATTERY_CONTENT_SHA256, C1_METRIC_CONTRACT, c1_scoring_contract,
)
from aadistill.autoinit.device_handoff import (  # noqa: E402
    DeviceHandoffError, complete_release, cuda_memory, require_headroom,
    require_released,
)
from aadistill.autoinit.fixed_path import (  # noqa: E402
    FixedPathDigestMismatch, materialize_fixed_path, write_replay_record,
)
from aadistill.autoinit.generation import (  # noqa: E402
    RecoveryEvaluationProtocol, declared_generation_protocol,
    generation_source_digest, observe_generation_protocol,
)
from aadistill.autoinit.operators import attention_activation  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

WS = Path("/workspace")
STATUS = WS / "autoinit_c1.status"

#: C1 owns its own roots. Nothing scientific is written under another phase's
#: tree: the launcher's ArtifactPolicy collects `audit/autoinit_c1`, and evidence
#: scattered across a directory the collector does not walk is how a session
#: loses it.
AUDIT = REPO / "artifacts/audit/autoinit_c1"
TRAIN = REPO / "artifacts/stage3/c1"
EVAL = REPO / "artifacts/eval/c1"
WORK = REPO / "artifacts/autoinit/c1_arms"

BATTERY = REPO / "artifacts/stage3/c1_confirmation_v1"
BATTERY_IDENTITY = REPO / "logs/phase_c1_battery.json"
TEACHER_BINDING = REPO / "logs/phase_c1_teacher_binding.json"
MEMORY_BASIS = REPO / "logs/autoinit_recovery_trainer_memory_basis.json"
FROZEN_RECIPE = REPO / "configs/stage3/e1/e1_r0860k_sa_pca.json"
PACK_DIR = "artifacts/stage3/ladder_uniform_probe"
C1_SCORER = REPO / "scripts/autoinit/score_c1_confirmation.py"
UNCAPPED_EVAL = REPO / "scripts/evaluation/uncapped_eval.py"
TRAINER = REPO / "scripts/training/train_stage3.py"
ENGINE_PROBE = REPO / "scripts/pod/autoinit_engine_probe.py"

#: The frozen evaluation tokenizer, staged from the relay. Pinned by file hash:
#: these exact bytes are what `chat_template_sha256 = 3802169b…` in the Stage-3
#: attestation was observed over. The teacher's own tokenizer is a DIFFERENT
#: artifact — 4 bytes larger, a different hash, a 10,834-byte config and no
#: chat template — and cannot substitute.
TOKENIZER_SOURCE = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
TOKENIZER_SIDECAR_SHA256 = {
    "tokenizer.json":
        "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
    "tokenizer_config.json":
        "8fa82a4ba512c8bee7c1c5e82b9a71ddbef362e4665be5c8f7ce0afd78af129a",
    "chat_template.jinja":
        "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7",
}

#: The ONLY fields a C1 probe may change relative to the frozen recipe. The seed
#: is the replicate and the initialization is the treatment; everything else
#: identical is what makes the two arms comparable at the same seed.
C1_PROBE_OVERRIDES = frozenset(
    {"run_name", "_purpose", "out_dir", "data_dir", "seed", "student_path"})


def _trainer_bytes() -> int:
    """Derived from the committed measurement, never typed in.

    The peak allocated over two COMPLETED 1023-step runs of this exact recipe,
    plus the two observed overheads that convert a PyTorch figure into the
    driver's free-bytes figure. A written constant is a guess wearing a gate's
    clothing; this reads the basis.
    """
    terms = json.loads(
        MEMORY_BASIS.read_text())["conversion_to_device_bytes"]["terms_gib"]
    return int((terms["peak_allocated"] + terms["allocator_reserved_slack"]
                + terms["non_pytorch_overhead"]) * 2**30)


class C1DriverError(RuntimeError):
    """A C1 stage refused. The message is the explanation."""


class C1ReplayMismatch(RuntimeError):
    """A frozen digest did not reproduce. The session stops here, by design."""


def _rel(p) -> str:
    """Repo-relative when it can be, absolute otherwise.

    `Path.relative_to` RAISES on a path outside the root, and the audit root is
    redirectable — the `$0` execution regression points it at a tmp tree. A path
    recorded for provenance must never be the thing that fails a stage.
    """
    try:
        return str(Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def trained_model_dir(out_dir: Path) -> Path:
    """The model a finished probe wrote, resolved the way the trainer writes it.

    `out_dir/checkpoints/latest.txt` holds a tag; the checkpoint is at
    `out_dir/checkpoints/<tag>/model`. Reading `out_dir/latest.txt` instead cost
    recovery continuation attempt 5 a 61.7-minute probe, so both failures are
    named rather than left to a bare FileNotFoundError.
    """
    ckpt_root = out_dir / "checkpoints"
    latest = ckpt_root / "latest.txt"
    if not latest.is_file():
        raise C1DriverError(
            f"the probe finished but {latest} does not exist; either training "
            f"wrote no checkpoint or {out_dir} is not the directory it was given")
    tag = latest.read_text().strip()
    model_dir = ckpt_root / tag / "model"
    if not model_dir.is_dir():
        raise C1DriverError(
            f"{latest} names tag {tag!r} but {model_dir} is not a directory")
    return model_dir


class C1Driver:
    """Stages B-I. No search, no rungs, no ranking, no elimination."""

    AUTHORIZATION_TYPE = C1Authorization
    AUTHORIZATION_PATH = "logs/autoinit_c1_authorization.json"

    def __init__(self, a):
        self.a = a
        self.t0 = time.time()
        self.completed: list[str] = []
        self.arms: dict = {}
        self.parent = None
        self.incumbent_step = None
        self.arm_init: dict[str, tuple[str, str]] = {}
        self.teacher_path: str | None = None
        self.training: dict[tuple[str, int], dict] = {}
        self.scored: dict[tuple[str, int], dict] = {}
        self.evaluation_protocol = None
        self.seeds = derive_recovery_seeds()

        for d in (AUDIT, AUDIT / "probes", AUDIT / "configs", TRAIN, EVAL, WORK):
            d.mkdir(parents=True, exist_ok=True)

        #: C1's OWN authorization and C1's OWN plan. A Phase-A grant carries a
        #: different plan hash and a ceiling derived for a beam search; it is
        #: refused by type at load and by hash immediately after.
        self.auth = self.AUTHORIZATION_TYPE.load(REPO / self.AUTHORIZATION_PATH)
        self.plan = self.frozen_plan()
        self.auth.require_plan(self.plan.plan_hash)
        self.auth.require_science_plan(C0_PREREGISTRATION_SHA256)

        self.ev: dict = {
            "schema": "aadistill.autoinit.c1_evidence/v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "session": "autoinit.v1.phase_c1",
            "session_contract_hash": CS.C1_SESSION_CONTRACT.contract_hash,
            "plan_hash": self.plan.plan_hash,
            "science_plan_hash": C0_PREREGISTRATION_SHA256,
            "seeds": list(self.seeds),
            "authorization": self.auth.as_dict(),
            "stages_completed": [],
            "training_started": False,
            "probes_trained": 0,
            "probes_evaluated": 0,
            "decision_ran": False,
            "runs_a_search": False,
            "eliminates_arms": False,
            "formal_recovery_evidence": "OUT OF SCOPE",
            "followon_started": False,
            "followon_reachable_from_this_driver": False,
            "stages": {},
        }
        self.save()

    # -- the frozen plan, rebuilt rather than transcribed -------------------
    def frozen_plan(self) -> C1IsolationPlan:
        battery = json.loads(BATTERY_IDENTITY.read_text())
        return C1IsolationPlan(
            plan_id="autoinit.v1.phase_c1",
            arms=(C1Arm("c1.incumbent", "incumbent", *CS.INCUMBENT_ATTENTION),
                  C1Arm("c1.treatment", "treatment", *CS.TREATMENT_ATTENTION)),
            seeds=tuple(self.seeds),
            battery_asset_id=battery["asset_id"],
            battery_content_sha256=battery["content_sha256"])

    # -- budget, evidence, subprocesses ------------------------------------
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
        self.ev["stages_completed"] = list(self.completed)
        (AUDIT / "c1_evidence.json").write_text(
            json.dumps(self.ev, indent=2, default=str) + "\n")

    def gate(self, name: str, argv: list[str], *, timeout: float,
             python: str = "/opt/train/bin/python") -> subprocess.CompletedProcess:
        out = subprocess.run([python, *argv], capture_output=True, text=True,
                             timeout=timeout, env=self.child_env())
        (AUDIT / f"{name}.log").write_text(
            f"$ {' '.join(argv)}\nrc={out.returncode}\n--- stdout ---\n"
            f"{out.stdout}\n--- stderr ---\n{out.stderr}\n")
        return out

    def complete(self, letter: str, **payload) -> None:
        """Record a finished stage and refuse an out-of-order execution."""
        stage = CS.stage(letter)
        self.completed.append(stage.stage_id)
        CS.assert_stage_order(["session_setup", *self.completed])
        self.ev["stages"][letter] = {
            "stage_id": stage.stage_id, "passed": True,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "spend_usd": round(self.usd(), 4), **payload}
        mark(f"STAGE_PASSED:{letter}")
        self.save()

    def fail(self, letter: str, reason: str, **payload) -> None:
        self.ev["stages"][letter] = {
            "stage_id": CS.stage(letter).stage_id, "passed": False,
            "reason": reason[-1500:],
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "spend_usd": round(self.usd(), 4), **payload}
        mark(f"STAGE_FAILED:{letter}")
        self.save()
        say(f"STAGE {letter} FAILED: {reason}")

    def runtime_identity(self) -> dict:
        import torch
        import transformers

        return {"image_digest": self.a.image_digest,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
                "driver": os.environ.get("NVIDIA_DRIVER_VERSION")}

    # -- B: teacher --------------------------------------------------------
    def stage_b(self) -> None:
        mark("STAGE_START:B")
        binding = json.loads(TEACHER_BINDING.read_text())
        if binding["revision"] != CS.TEACHER_REVISION:
            raise C1DriverError(
                f"teacher binding pins {binding['revision']} but the session "
                f"declares {CS.TEACHER_REVISION}")
        import hashlib

        from huggingface_hub import snapshot_download

        local = snapshot_download(CS.TEACHER_REPO, revision=CS.TEACHER_REVISION)
        bad = []
        for name, want in binding["expected_shard_sha256"].items():
            p = Path(local) / name
            if not p.is_file():
                bad.append(f"{name}: absent after fetch")
                continue
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != want:
                bad.append(f"{name}: {got} != {want}")
        if bad:
            raise C1DriverError("teacher verification FAILED: " + "; ".join(bad))
        self.teacher_path = local
        say(f"teacher {CS.TEACHER_REPO}@{CS.TEACHER_REVISION[:12]} verified, "
            f"{len(binding['expected_shard_sha256'])} shards")
        self.complete("B", repo_id=CS.TEACHER_REPO,
                      revision=CS.TEACHER_REVISION, local_path=local,
                      shards_verified=len(binding["expected_shard_sha256"]))

    # -- C: operator, and the CPU pre-flight of everything stage H needs ----
    def stage_c(self) -> None:
        """Register the operator, then refuse now for anything H would die on.

        The scorer pre-flight is here rather than in H on purpose: a battery the
        scorer cannot read is a `$0.30` stop at stage C and a `~$10` one after
        six trainings. It costs seconds and touches no GPU.
        """
        mark("STAGE_START:C")
        impl = attention_activation.register(replace=True)
        say(f"registered {impl.impl_id} ({impl.signature_hash[:12]})")

        contract = c1_scoring_contract(REPO)
        probe = self.gate(
            "scorer_preflight",
            [str(C1_SCORER), "--generations", str(AUDIT / "_preflight_absent"),
             "--label", "preflight", "--seed", "0",
             "--out", str(AUDIT / "_preflight.json")],
            timeout=300, python=sys.executable)
        text = probe.stdout + probe.stderr
        if "no generations for" not in text:
            raise C1DriverError(
                "the C1 scorer did not reach its generation check on the frozen "
                f"battery; it would fail after six trainings. tail: ...{text[-800:]}")

        for name, want in TOKENIZER_SIDECAR_SHA256.items():
            p = TOKENIZER_SOURCE / name
            if not p.is_file():
                raise C1DriverError(
                    f"the frozen evaluation tokenizer is not staged: {p} is "
                    "missing. Stage H cannot package a probe without it.")
            got = sha256_file(p)
            if got != want:
                raise C1DriverError(f"{p} hashes to {got}, pinned {want}")

        manifest = json.loads((BATTERY / "manifest.json").read_text())
        if manifest["content_sha256"] != C1_BATTERY_CONTENT_SHA256:
            raise C1DriverError("the staged battery is not the frozen one")

        self.complete("C", impl_id=impl.impl_id,
                      signature_hash=impl.signature_hash,
                      scoring_contract=contract["contract"],
                      scoring_digest=contract["digest"],
                      tokenizer_sidecars=sorted(TOKENIZER_SIDECAR_SHA256),
                      battery_content_sha256=manifest["content_sha256"])

    # -- D and E: one replay, two individually observable gates -------------
    def stage_de(self) -> None:
        mark("STAGE_START:D")
        from aadistill.autoinit.adapters.qwen3 import QWEN3_ADAPTER
        from transformers import AutoModelForCausalLM

        self.arms = CS.build_arm_specs(workdir_device="cuda")
        if not CS.arm_prefix_is_shared(self.arms):
            raise C1DriverError("the two arms do not share their prefix")
        runtime = self.runtime_identity()
        seen: list = []

        def on_step(result) -> None:
            #: `materialize_fixed_path` calls this BEFORE it raises on a
            #: mismatch, so a gate is only passed when its digest actually
            #: matched. Keying on `digest_expected` alone would mark stage D
            #: PASSED and then stop the session for D failing.
            seen.append(result)
            if result.digest_expected is None or result.digest_matches is not True:
                return
            if result.index == 2:
                say(f"D: parent {result.identity.artifact_digest[:12]} matches")
                self.complete("D", step_index=result.index,
                              artifact_digest=result.identity.artifact_digest,
                              expected=result.digest_expected,
                              selection=result.selection, runtime=runtime)
                mark("STAGE_START:E")
            elif result.index == 3:
                say(f"E: incumbent {result.identity.artifact_digest[:12]} matches")
                self.complete("E", step_index=result.index,
                              artifact_digest=result.identity.artifact_digest,
                              expected=result.digest_expected,
                              selection=result.selection, runtime=runtime)

        try:
            steps = materialize_fixed_path(
                self.arms["incumbent"], adapter=QWEN3_ADAPTER,
                root_loader=lambda: AutoModelForCausalLM.from_pretrained(
                    self.teacher_path, dtype="bfloat16").eval(),
                workdir=WORK / "incumbent", repo_root=str(REPO), on_step=on_step)
        except FixedPathDigestMismatch as exc:
            self.replay_mismatch(exc, runtime, seen)
            raise C1ReplayMismatch(str(exc)) from exc

        write_replay_record(self.arms["incumbent"], steps,
                            AUDIT / "c1_replay_record.json", runtime=runtime,
                            root_binding=json.loads(TEACHER_BINDING.read_text()))
        self.parent = steps[2]
        self.incumbent_step = steps[3]

    def replay_mismatch(self, exc, runtime: dict, seen: list) -> None:
        """Write the evidence, PROVE it landed, and only then mark the outcome.

        The marker is what the session runner reads to choose the reduced failure
        artifact spec. Emitting it before the record is on disk and re-readable
        would let a write failure turn a scientific finding into an empty
        teardown — and the generic two-spec policy cannot require this file,
        because a stage-B failure never wrote it. So the requirement is enforced
        here, where it can be.
        """
        letter = "D" if exc.step_index < 3 else "E"
        record = {
            "schema": "aadistill.autoinit.c1_replay_mismatch/v1",
            "stage": letter, "step_index": exc.step_index, "label": exc.label,
            "expected": exc.expected, "actual": exc.actual, "runtime": runtime,
            "evidence": exc.evidence,
            "steps_observed": [r.as_dict() for r in seen],
            "training_started": False,
            "meaning": (
                "the frozen path did not reproduce its recorded digest under this "
                "runtime. NO recovery training was started. This is the session's "
                "product: refer it to review with the evidence attached. Do not "
                "retry, and do not substitute a rebuilt parent for the historical "
                "one without a reviewed amendment."),
        }
        path = AUDIT / "c1_replay_record.json"
        path.write_text(json.dumps(record, indent=1) + "\n")
        back = json.loads(path.read_text())
        if back.get("stage") != letter or back.get("training_started") is not False:
            raise C1DriverError(
                f"{path} did not read back as the mismatch record it was just "
                "given; refusing to emit C1_REPLAY_MISMATCH without its evidence")
        self.fail(letter, f"replay mismatch at step {exc.step_index}: "
                          f"{exc.expected} != {exc.actual}",
                  mismatch_record=_rel(path),
                  mismatch_record_sha256=sha256_file(path))
        mark("C1_REPLAY_MISMATCH")

    # -- F: the treatment arm, from the SAME verified parent ----------------
    def stage_f(self) -> None:
        mark("STAGE_START:F")
        from aadistill.autoinit.adapters.qwen3 import QWEN3_ADAPTER

        treatment = materialize_fixed_path(
            self.arms["treatment"], adapter=QWEN3_ADAPTER,
            root_loader=lambda: QWEN3_ADAPTER.load(self.parent.checkpoint_path,
                                                   device="cuda"),
            workdir=WORK / "treatment", repo_root=str(REPO))
        identities = {
            "schema": "aadistill.autoinit.c1_arm_identities/v1",
            "parent": self.parent.as_dict(),
            "incumbent": self.incumbent_step.as_dict(),
            "treatment": treatment[-1].as_dict(),
            "shared_parent": True,
            "runtime": self.runtime_identity(),
        }
        (AUDIT / "c1_arm_identities.json").write_text(
            json.dumps(identities, indent=1) + "\n")
        self.arm_init = {
            "incumbent": (self.incumbent_step.checkpoint_path,
                          self.incumbent_step.identity.artifact_digest),
            "treatment": (treatment[-1].checkpoint_path,
                          treatment[-1].identity.artifact_digest),
        }
        self.complete("F", **{a: d for a, (_, d) in self.arm_init.items()})

    def release_device(self) -> dict:
        """Hand the card to the trainer, and prove the handoff before training.

        Not scope exit: attempt 4 read a verdict saying 7.55 GiB was still
        allocated, started the trainer anyway and lost the probe. Both conditions
        are enforced — the release actually worked, and the card has room for the
        measured peak plus its observed overheads.
        """
        before = cuda_memory()
        self.arms = {}
        self.parent = None
        self.incumbent_step = None
        gc.collect()
        handoff = complete_release(before)
        (AUDIT / "c1_device_handoff.json").write_text(
            json.dumps(handoff, indent=2, default=str) + "\n")
        need = _trainer_bytes()
        require_released(handoff, what="the C1 recovery trainer")
        require_headroom(handoff["after"], need_bytes=need,
                         what="the C1 recovery trainer")
        say(f"device handoff: {handoff.get('verdict', 'n/a')}")
        return {"handoff": handoff, "need_bytes": need}

    # -- G: six trainings. NOTHING is evaluated here ------------------------
    def descriptors(self) -> list[dict]:
        """Every arm on every seed. No rung, no control, no elimination."""
        out = []
        for arm in ARMS:
            path, digest = self.arm_init[arm]
            for seed in self.seeds:
                out.append({
                    "probe_id": f"autoinit.v1.phase_c1.{arm}.{seed}",
                    "arm": arm, "seed": seed,
                    "student_path": path,
                    "initialization_artifact_digest": digest,
                })
        assert len(out) == 6, "C1 is exactly six probes"
        return out

    def probe_config(self, d: dict) -> Path:
        frozen = json.loads(FROZEN_RECIPE.read_text())
        name = d["probe_id"]
        derived = {**frozen, "run_name": name,
                   "out_dir": f"artifacts/stage3/c1/{name}",
                   "data_dir": PACK_DIR, "seed": d["seed"],
                   "student_path": d["student_path"],
                   "_purpose": (
                       f"Phase C1 confirmation probe, arm {d['arm']}, seed "
                       f"{d['seed']}. Identical recovery; the only intended "
                       "difference between the paired arms is the ATTENTION "
                       f"operator. Derived from {FROZEN_RECIPE.name} by overriding "
                       "run identity, pack path, seed and student_path.")}
        diff = sorted(k for k in set(frozen) | set(derived)
                      if frozen.get(k) != derived.get(k))
        if not set(diff) <= C1_PROBE_OVERRIDES:
            raise C1DriverError(
                f"{name}: the derived probe config differs from the frozen recipe "
                f"in {sorted(set(diff) - C1_PROBE_OVERRIDES)}, outside the allowed "
                f"override set {sorted(C1_PROBE_OVERRIDES)}")
        path = AUDIT / "configs" / f"{name}.json"
        path.write_text(json.dumps(derived, indent=2) + "\n")
        return path

    def train_one(self, name: str, config: Path) -> Path:
        """Spawn the recovery trainer. THE ONLY hardware-bound step of stage G.

        A seam, not a configuration surface. Everything around it — the budget
        check, the override check, the journal, the completion count — is the
        loop's business, and the `$0` execution regression replaces exactly this
        method. A harness that reimplemented the loop could not notice the loop
        being broken, which is the failure mode that certified a defective line
        once already.
        """
        rc = subprocess.run(
            ["/opt/train/bin/python", str(TRAINER), "--config", str(config)],
            capture_output=True, text=True,
            timeout=int(self.a.probe_train_minutes * 60 * 2), env=self.child_env())
        (AUDIT / f"{name}_train_tail.log").write_text(
            (rc.stdout + rc.stderr)[-1500:])
        if rc.returncode != 0:
            raise C1DriverError(
                f"{name}: training failed rc={rc.returncode}; tail: "
                f"...{(rc.stdout + rc.stderr)[-1200:]}")
        return TRAIN / name

    def require_all_trained(self) -> None:
        """Stage H's precondition, in one place the driver and its harness share."""
        if len(self.training) != 6:
            raise C1DriverError(
                f"stage H requires six training completions, found "
                f"{len(self.training)}; the confirmation battery is evaluated "
                "once per fully trained probe and never before")

    def stage_g(self) -> None:
        mark("STAGE_START:G")
        self.release_device()
        self.ev["training_started"] = True
        for d in self.descriptors():
            name = d["probe_id"]
            journal = AUDIT / "probes" / f"{name}.training.json"
            if journal.is_file():
                record = json.loads(journal.read_text())
                if (record.get("complete")
                        and record.get("initialization_artifact_digest")
                        == d["initialization_artifact_digest"]):
                    say(f"  {name}: training restored from the journal")
                    self.training[(d["arm"], d["seed"])] = record
                    continue
            if not self.afford(self.a.probe_train_minutes, name):
                raise C1DriverError(f"budget refuses {name}; no probe is skipped "
                                    "to make progress")
            config = self.probe_config(d)
            t = time.time()
            out_dir = self.train_one(name, config)
            model_dir = trained_model_dir(out_dir)
            record = {
                "schema": "aadistill.autoinit.c1_training_completion/v1",
                **{k: d[k] for k in ("probe_id", "arm", "seed",
                                     "initialization_artifact_digest")},
                "config": _rel(config),
                "config_sha256": sha256_file(config),
                "out_dir": _rel(out_dir),
                "model_dir": _rel(model_dir),
                "train_minutes": round((time.time() - t) / 60, 2),
                "run_completion": _rel(out_dir / "run_completion.json"),
                "evaluated": False,
                "complete": True,
            }
            journal.write_text(json.dumps(record, indent=2) + "\n")
            self.training[(d["arm"], d["seed"])] = record
            self.ev["probes_trained"] = len(self.training)
            self.save()
            mark(f"PROBE_TRAINED:{name}")
            say(f"  {name}: trained in {record['train_minutes']:.1f} min")

        self.require_all_trained()
        self.complete("G", probes_trained=6,
                      completions=sorted(r["probe_id"]
                                         for r in self.training.values()))

    # -- H: six evaluations, once each, on the C1 battery only --------------
    def attest(self) -> dict:
        """C1's own evaluation-protocol attestation. No Phase-A threshold is read.

        Phase A binds its candidates to the Stage-3 equivalence interval and
        feasibility floor, because those are absolute thresholds materialized by
        other checkpoints. C1's estimand is a within-session paired DIFFERENCE, so
        it verifies only what it needs: the generation fingerprint, the C1 scoring
        contract, the battery identity, the tokenizer identity and semantics, and
        the runtime fields the execution preregistration requires be recorded.
        """
        sample = next(iter(self.training.values()))
        package = EVAL / "_attestation_package"
        report = build_evaluation_package(
            REPO / sample["model_dir"], tokenizer_source=TOKENIZER_SOURCE,
            dest=package, expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
        engine = self.gate(
            "engine_probe",
            [str(ENGINE_PROBE), "--model", str(package),
             "--out", str(AUDIT / "engine_probe.json"),
             "--image-digest", self.a.image_digest],
            timeout=1800, python="/opt/vllm/bin/python")
        if engine.returncode != 0:
            raise C1DriverError(
                f"engine probe rc={engine.returncode}; tail: "
                f"...{(engine.stdout + engine.stderr)[-1200:]}")
        observed = json.loads((AUDIT / "engine_probe.json").read_text())

        gen = declared_generation_protocol().materialized(
            generation_source_digest=generation_source_digest(REPO)["digest"],
            degeneration_source_digest=sha256_file(
                REPO / "src/aadistill/evaluation/degeneration.py"))
        gen = gen.materialized(
            vllm_version=observed["vllm_version"],
            transformers_version=observed["transformers_version"],
            torch_version=observed["torch_version"],
            runtime_digest=observed["runtime_digest"], dtype=observed["dtype"],
            gpu_memory_utilization=observed["gpu_memory_utilization"],
            max_num_seqs=observed["max_num_seqs"],
            max_num_batched_tokens=observed["max_num_batched_tokens"],
            enforce_eager=observed["enforce_eager"],
            tokenizer_sha256=observed["tokenizer_sha256"],
            chat_template_sha256=observed["chat_template_sha256"],
            resolved_context=observed["resolved_context"],
            context_source=observed["context_source"],
            stop_token_ids=tuple(observed["stop_token_ids"]))
        gen.require_materialized(context="phase C1 stage H")

        manifest = json.loads((BATTERY / "manifest.json").read_text())
        contract = c1_scoring_contract(REPO)
        self.evaluation_protocol = RecoveryEvaluationProtocol(
            generation=gen, scoring_contract=contract["contract"],
            scoring_digest=contract["digest"],
            battery_artifact=manifest["artifact"],
            battery_manifest_sha256=manifest["manifest_sha256"],
            battery_content_sha256=manifest["content_sha256"])
        attested = {
            "schema": "aadistill.autoinit.c1_attested_protocol/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": self.runtime_identity(),
            "generation_source_digest": generation_source_digest(REPO),
            "generation_protocol_fingerprint": gen.fingerprint,
            "scoring_contract": contract,
            "metric_contract": C1_METRIC_CONTRACT,
            "evaluation_protocol": self.evaluation_protocol.as_dict(),
            "evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
            "battery": {"artifact": manifest["artifact"],
                        "content_sha256": manifest["content_sha256"],
                        "manifest_sha256": manifest["manifest_sha256"],
                        "n_prompts": manifest["n_prompts"],
                        "n_scorable_prompts": manifest["n_scorable_prompts"]},
            "tokenizer": {
                "source_rule": "the evaluated checkpoint",
                "packaged_from": _rel(TOKENIZER_SOURCE),
                "sidecar_sha256": dict(TOKENIZER_SIDECAR_SHA256),
                "observed_sha256": observed["tokenizer_sha256"],
                "observed_chat_template_sha256": observed["chat_template_sha256"],
                "packaging": report["tokenizer_source_rule"],
            },
            "no_phase_a_thresholds": (
                "C1's estimand is a within-session paired difference; the Phase-A "
                "equivalence interval and feasibility floor are absolute "
                "thresholds materialized by other checkpoints and are not read"),
        }
        attested["report_sha256"] = sha256_json(attested)
        (AUDIT / "c1_attested_evaluation_protocol.json").write_text(
            json.dumps(attested, indent=2) + "\n")
        say(f"attested: protocol {attested['evaluation_protocol_hash'][:12]}…, "
            f"tokenizer {observed['tokenizer_sha256'][:12]}…")
        return attested

    def generate_one(self, name: str, package: Path, gen_dir: Path, sets) -> None:
        """Run the evaluator. THE ONLY hardware-bound step of stage H."""
        out = self.gate(
            f"{name}_generation",
            [str(UNCAPPED_EVAL), "--model", str(package), "--label", name,
             "--prompts", *[str(BATTERY / f"{s}.jsonl") for s in sets],
             "--out-dir", str(gen_dir), "--diagnostics"],
            timeout=int(self.a.probe_battery_minutes * 60 * 3),
            python="/opt/vllm/bin/python")
        if out.returncode != 0:
            raise C1DriverError(
                f"{name}: generation rc={out.returncode}; tail: "
                f"...{(out.stdout + out.stderr)[-1200:]}")

    def admit_generation(self, name: str, gen_dir: Path) -> dict:
        """Refuse a probe whose generations were not produced under the protocol.

        The attestation is a statement about what the runtime *should* do, made
        once, from an engine probe. It is not evidence about any particular
        probe's rollouts. This reconstructs the protocol from THIS probe's raw
        per-set summaries and requires the resulting evaluation protocol to be
        comparable to the attested one, under the same versioned relation the
        rest of the project uses.

        Fail-closed and BEFORE the scorer runs: a probe generated under a drifted
        protocol must not be scored, must not be followed by another probe, and
        must not reach the decision. The summaries stay on disk either way — the
        artifact spec collects them — so a refusal is a diagnosis, not a loss.
        """
        summaries = [json.loads(p.read_text())
                     for p in sorted(gen_dir.glob("*.json"))
                     if not p.name.endswith(".generations.jsonl")]
        observed_gen = observe_generation_protocol(summaries).protocol
        manifest = json.loads((BATTERY / "manifest.json").read_text())
        contract = c1_scoring_contract(REPO)
        observed = RecoveryEvaluationProtocol(
            generation=observed_gen,
            scoring_contract=contract["contract"],
            scoring_digest=contract["digest"],
            battery_artifact=manifest["artifact"],
            battery_manifest_sha256=manifest["manifest_sha256"],
            battery_content_sha256=manifest["content_sha256"])
        record = {
            "probe_id": name,
            "generation_fingerprint": observed_gen.fingerprint,
            "evaluation_protocol_hash": observed.evaluation_protocol_hash,
            "attested_evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
            "n_summaries": len(summaries),
        }
        try:
            observed.require_comparable(self.evaluation_protocol, context=name)
        except Exception as exc:                                  # noqa: BLE001
            record["comparable"] = False
            record["reason"] = str(exc)[-1500:]
            (AUDIT / f"{name}_generation_admission.json").write_text(
                json.dumps(record, indent=2) + "\n")
            raise C1DriverError(
                f"{name}: the generations were not produced under the attested "
                f"evaluation protocol, so this probe cannot be scored and no "
                f"later probe may be evaluated. {exc}") from exc
        record["comparable"] = True
        (AUDIT / f"{name}_generation_admission.json").write_text(
            json.dumps(record, indent=2) + "\n")
        say(f"  {name}: generation protocol admitted "
            f"({record['evaluation_protocol_hash'][:12]}…)")
        return record

    def stage_h(self) -> None:
        mark("STAGE_START:H")
        self.require_all_trained()
        attested = self.attest()
        for (arm, seed), record in sorted(self.training.items()):
            name = record["probe_id"]
            if not self.afford(self.a.probe_battery_minutes, name):
                raise C1DriverError(f"budget refuses evaluation of {name}")
            package = EVAL / name / "package"
            build_evaluation_package(
                REPO / record["model_dir"], tokenizer_source=TOKENIZER_SOURCE,
                dest=package, expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
            gen_dir = EVAL / name
            sets = json.loads((BATTERY / "manifest.json").read_text())["sets"]
            self.generate_one(name, package, gen_dir, sets)

            #: ADMISSION, before scoring. The attestation says what the protocol
            #: is expected to be; this says what it actually WAS, reconstructed
            #: from this probe's own raw summaries. Scoring first and recording
            #: the observed fingerprint afterwards — which is what this did — let
            #: a result claim the expected identity before the evidence for it had
            #: been admitted, and never compared the two at all.
            observed = self.admit_generation(name, gen_dir)
            scored = AUDIT / f"{name}_c1_confirmation.json"
            per_sample = AUDIT / f"{name}_per_sample.jsonl"
            rc = self.gate(
                f"{name}_scoring",
                [str(C1_SCORER), "--generations", str(gen_dir), "--label", name,
                 "--seed", str(seed), "--out", str(scored),
                 "--per-sample", str(per_sample), "--arm", arm,
                 "--init-digest", record["initialization_artifact_digest"],
                 "--trained-run", str(REPO / record["run_completion"]),
                 #: The OBSERVED fingerprint, not the attested one. They are equal
                 #: by the check above; the direction of provenance is the point.
                 "--generation-fingerprint", observed["generation_fingerprint"]],
                timeout=1800, python=sys.executable)
            if rc.returncode != 0:
                raise C1DriverError(
                    f"{name}: scoring rc={rc.returncode}; tail: "
                    f"...{(rc.stdout + rc.stderr)[-1200:]}")
            result = json.loads(scored.read_text())
            self.scored[(arm, seed)] = {
                "record": record, "result": result,
                "result_path": _rel(scored),
                "per_sample_path": _rel(per_sample),
                "observed_generation": observed["generation_fingerprint"],
                "observed_evaluation_protocol_hash":
                    observed["evaluation_protocol_hash"],
            }
            self.ev["probes_evaluated"] = len(self.scored)
            self.save()
            mark(f"PROBE_SCORED:{name}")
            say(f"  {name}: usable {result['usable_rollout_rate']}, "
                f"correct {result['correct_overall']}")
        self.complete("H", probes_evaluated=len(self.scored),
                      evaluation_protocol_hash=attested["evaluation_protocol_hash"],
                      battery=attested["battery"]["content_sha256"])

    # -- I: the frozen paired decision --------------------------------------
    def stage_i(self) -> None:
        mark("STAGE_START:I")
        if len(self.scored) != 6:
            raise C1DriverError(
                f"{len(self.scored)} probe results, not 6; the decision rule may "
                "not run on a partial design")
        per_sample = {}
        records = []
        for (arm, seed), s in self.scored.items():
            rows = [json.loads(x) for x in
                    (REPO / s["per_sample_path"]).open() if x.strip()]
            per_sample[(arm, seed)] = rows
            r = s["result"]
            records.append(C1ProbeRecord(
                probe_id=s["record"]["probe_id"], arm=arm, seed=seed,
                initialization_artifact_digest=s["record"][
                    "initialization_artifact_digest"],
                trained_run=r.get("trained_run") or {},
                result_path=s["result_path"],
                result_sha256=sha256_file(REPO / s["result_path"]),
                per_sample_path=s["per_sample_path"],
                per_sample_sha256=sha256_file(REPO / s["per_sample_path"]),
                generations=r["generations"],
                counts={k: r[k] for k in ("n", "usable", "correct", "n_scorable",
                                          "usable_scorable")},
                rates={k: r[k] for k in ("usable_rollout_rate", "correct_overall",
                                         "correct_given_usable")},
                per_capability=r["per_capability"],
                scoring_contract=r["scoring_contract"], battery=r["battery"],
                observed_generation_fingerprint=s["observed_generation"],
                observed_evaluation_protocol_hash=s[
                    "observed_evaluation_protocol_hash"]))

        inputs = decision_inputs(per_sample, seeds=self.seeds)
        results = build_probe_results(
            records, plan_hash=self.plan.plan_hash, seeds=self.seeds,
            inputs=inputs,
            attested_evaluation_protocol_hash=(
                self.evaluation_protocol.evaluation_protocol_hash))
        (AUDIT / "c1_probe_results.json").write_text(
            json.dumps(results, indent=2) + "\n")

        d = paired_differences(inputs.arm("incumbent"), inputs.arm("treatment"))
        boot = stratified_cluster_bootstrap(d, inputs.strata)
        per_seed = [
            sum(bool(inputs.correct["treatment"][s][j])
                - bool(inputs.correct["incumbent"][s][j]) for j in d) / len(d)
            for s in self.seeds]
        decision = decide(
            self.plan, boot=boot, per_seed_delta=per_seed,
            usable_pooled_delta=inputs.usable_pooled_delta,
            usable_per_seed_delta=list(inputs.usable_per_seed_delta),
            catastrophic_violations=inputs.catastrophic_violations)
        decision["plan_hash"] = self.plan.plan_hash
        decision["probe_results_sha256"] = results["results_sha256"]
        decision["mcnemar"] = {
            "note": ("per-seed discordant pairs over the 850 scorable prompts; "
                     "counts, not rates"),
            "per_seed": [
                {"seed": s,
                 "treatment_only": sum(
                     1 for j in d if inputs.correct["treatment"][s][j]
                     and not inputs.correct["incumbent"][s][j]),
                 "incumbent_only": sum(
                     1 for j in d if inputs.correct["incumbent"][s][j]
                     and not inputs.correct["treatment"][s][j])}
                for s in self.seeds],
        }
        decision["diagnostics"] = {
            "per_seed_correct": {
                arm: {s: sum(1 for j in d if inputs.correct[arm][s][j])
                      for s in self.seeds} for arm in ARMS},
            "usable_counts": inputs.audit["usable_counts"],
            "strata_sizes": inputs.audit["strata_sizes"],
        }
        (AUDIT / "c1_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
        self.ev["decision_ran"] = True
        say(f"DECISION: {decision['verdict']} · delta {decision['delta']:+.4f} "
            f"· LCB {decision['lcb_one_sided']:+.4f}")
        self.complete("I", verdict=decision["verdict"], delta=decision["delta"],
                      probe_results_sha256=results["results_sha256"])

    # -- run ---------------------------------------------------------------
    def run(self) -> int:
        mark("DRIVER_START")
        stages = (("B", self.stage_b), ("C", self.stage_c), ("DE", self.stage_de),
                  ("F", self.stage_f), ("G", self.stage_g), ("H", self.stage_h),
                  ("I", self.stage_i))
        for letter, fn in stages:
            try:
                fn()
            except C1ReplayMismatch:
                self.finish("C1_REPLAY_MISMATCH")
                return 30
            except Exception as exc:                              # noqa: BLE001
                self.fail(letter[0], f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc()[-6000:])
                blocking = CS.stage(letter[0]).blocks_training
                outcome = ("C1_FAILED" if blocking or not self.ev["training_started"]
                           else "C1_INCOMPLETE")
                mark(outcome)
                self.finish(outcome)
                return 40
        mark("ALL_DONE")
        say(f"C1 complete — ${self.usd():.2f}. STOP for review.")
        self.finish("ALL_DONE")
        return 0

    def finish(self, outcome: str) -> None:
        self.ev["outcome"] = outcome
        self.ev["cleanup_is_not_success"] = (
            "collection and teardown run regardless of the outcome; a clean "
            "cleanup does not make a stopped session a successful C1")
        self.save()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase C1 fixed-path ATTENTION isolation")
    ap.add_argument("--image-digest", default="")
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--probe-train-minutes", type=float, default=70.0)
    ap.add_argument("--probe-battery-minutes", type=float, default=25.0)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    say(f"C1: {CS.C1_SESSION_CONTRACT.n_probes} probes, ceiling "
        f"${args.authorized_usd:.4f}, soft stop ${args.soft_stop_usd:.4f}")
    for s in CS.C1_STAGES:
        say(f"  {s.letter}: {s.stage_id}"
            + ("   [blocks training on failure]" if s.blocks_training else ""))
    driver = C1Driver(args)
    driver.auth.require_within_cap(args.authorized_usd, what="session backstop")
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
