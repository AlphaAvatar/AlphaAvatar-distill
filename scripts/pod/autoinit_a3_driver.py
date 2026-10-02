#!/usr/bin/env python3
"""A3: replay the frozen parent, diagnose two protocols, train three, score three.

    /opt/train/bin/python scripts/pod/autoinit_a3_driver.py \
        --image-digest <digest> --rate 1.09 --spent-usd 0.20 \
        --soft-stop-usd 7.95 --authorized-usd 8.3047

    B  fetch the pinned teacher and verify every shard against the binding
    C  register attention.activation_importance_v1, and pre-flight the scorer
    D  replay DEPTH -> FFN -> RESIDUAL_WIDTH        GATE: parent eea90c91...
    E  interleaved A_bsz1/A_bsz3 diagnostics         GATE: A_bsz1 == 53e30566...
    F  three 0.86M recovery trainings from the A_bsz3 initialization
    G  three confirmation evaluations, once each
    H  preserve and package

A and I belong to the session runner. **There is no stage for the decision.**
attempt75 trained, preserved and scored nine probes over 919 minutes and
`$16.71`, then raised in its on-pod aggregation before writing a verdict. A3's
comparison is `scripts/autoinit/aggregate_a3.py`, off pod, at `$0`, so no
scientific product depends on the pod surviving one more stage.

**The gate is asymmetric and that is the experiment.** A-bsz1 must rebuild the
frozen incumbent; A-bsz3 carries no digest pin, because pinning what the
experiment measures would answer it by assertion. **A differing A-bsz3 digest
is a FINDING and stage E records it and continues.** The integrity stops are in
the frozen design and each one is repaired rather than interpreted.

**This driver is A3's own.** It does not subclass or import the C3 driver.
That inheritance is not a shortcut: the C3 driver binds C3's preregistration,
its three-arm invariant, its seeds, its status path and its audit roots, and
every one of those is wrong here. The small helpers worth reusing — `mark`,
`say`, `trained_model_dir` — are about thirty lines and are restated rather
than importing a 1651-line operational driver to obtain them.

**Two seams, and only two.** `train_one` and `generate_one` are the only
hardware-bound steps; the `$0` production rehearsal replaces exactly those and
runs everything else — the loops, the gates, the packaging, the real scorer —
as production code.

**Every arm-shaped default is passed explicitly.** `C1ProbeRecord.allowed_arms`
and `build_probe_results(arms=...)` both default to C1's two ROLES. attempt75
passed C3's arm ids and died on `unknown arm 'A_incumbent'` after a complete
measurement. A3's vocabulary is one arm and it is stated at every layer that
has a default.
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
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))

from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.initialization.adapters import register_builtin_adapters  # noqa: E402
from aadistill.initialization.calibration.profiles import get_profile  # noqa: E402
from aadistill.initialization.operators.attention.gqa import (  # noqa: E402
    activation_importance as attention_activation,
)
from aadistill.initialization.planning.fixed_path import (  # noqa: E402
    FixedPathDigestMismatch, VerifiedSuffix, materialize_fixed_path,
    write_replay_record,
)
from aadistill.initialization.planning.generation import (  # noqa: E402
    RecoveryEvaluationProtocol, declared_generation_protocol,
    observe_generation_protocol,
)
from aadistill.runtime.device_handoff import (  # noqa: E402
    complete_release, cuda_memory, require_headroom, require_released,
)
from experiments.calibration import register_builtin_profiles  # noqa: E402
from experiments.phase_c1.packaging import build_evaluation_package  # noqa: E402
from experiments.phase_c1.probe_results import C1ProbeRecord  # noqa: E402
from experiments.phase_c1.scoring import (  # noqa: E402
    C1_METRIC_CONTRACT, c1_scoring_contract,
)
from experiments.phase_c3 import a3_session as A3S  # noqa: E402
from experiments.source_sets import generation_source_digest  # noqa: E402

#: Explicit. Importing the core registers neither a mixture nor an adapter, and
#: a driver that forgot the adapter call reached stage D, loaded 398 tensors,
#: asked for `qwen3` and got `registered: []` at $0.43.
register_builtin_profiles()
register_builtin_adapters()

WS = Path(A3S.POD_WORKSPACE)
#: From the ONE owner. Hardcoding it is how a launcher came to poll a
#: different file and see none of the driver's markers, `ALL_DONE` included.
STATUS = Path(A3S.STATUS_PATH)

#: A3 owns its own roots. A probe landing under another phase's tree is
#: collected by that phase's artifact spec and attributed to a closed
#: experiment.
AUDIT = REPO / "artifacts/audit/autoinit_a3"
TRAIN = REPO / "artifacts/stage3/a3"
EVAL = REPO / "artifacts/eval/a3"
WORK = REPO / "artifacts/autoinit/a3_arms"

#: Shared FROZEN identities, referenced rather than copied. A second copy under
#: a phase_a3 directory would be a second owner free to drift from the battery
#: the probes are actually evaluated on.
BATTERY = REPO / "artifacts/stage3/c1_confirmation_v1"
BATTERY_IDENTITY = REPO / "logs/stages/stage-1/phase_c1/plans/battery.json"
TEACHER_BINDING = REPO / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
MEMORY_BASIS = (REPO / "logs/stages/stage-1/recovery_continuation/analyses"
                       "/autoinit_recovery_trainer_memory_basis.json")
FROZEN_RECIPE = REPO / "configs/stage3/e1/e1_r0860k_sa_pca.json"
PACK_DIR = "artifacts/stage3/ladder_uniform_probe"
C1_SCORER = REPO / "scripts/autoinit/score_c1_confirmation.py"
UNCAPPED_EVAL = REPO / "scripts/evaluation/uncapped_eval.py"
TRAINER = REPO / "scripts/training/train_stage3.py"
ENGINE_PROBE = REPO / "scripts/pod/autoinit_engine_probe.py"

TOKENIZER_SOURCE = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
TOKENIZER_SIDECAR_SHA256 = {
    "tokenizer.json":
        "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
    "tokenizer_config.json":
        "8fa82a4ba512c8bee7c1c5e82b9a71ddbef362e4665be5c8f7ce0afd78af129a",
    "chat_template.jinja":
        "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7",
}

#: DURABLE STORAGE for completed probes, pushed the moment a probe exists.
#: C1 attempt 17 trained six over ten hours and lost all six when a later
#: stage failed and the pod was deleted.
RELAY = "AlphaAvatar/aadistill-artifacts"
PRESERVED_PREFIX = "a3_preserved_probes"
TOKEN_FILE = Path("/workspace/hf/token")

#: The ONLY fields an A3 probe may change relative to the frozen recipe.
A3_PROBE_OVERRIDES = frozenset(
    {"run_name", "_purpose", "out_dir", "data_dir", "seed", "student_path"})


class A3DriverError(RuntimeError):
    """An A3 stage refused. The message is the explanation."""


def relay_token() -> str:
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    if TOKEN_FILE.is_file() and TOKEN_FILE.read_text().strip():
        return TOKEN_FILE.read_text().strip()
    raise A3DriverError(
        f"no relay token: HF_TOKEN unset and {TOKEN_FILE} absent or empty. "
        "The driver runs detached with a minimal environment, so the "
        "credential is staged on disk rather than inherited.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark(name: str) -> None:
    """One line to the ONE status file the launcher polls."""
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a") as fh:
        fh.write(f"{name}\n")
        fh.flush()
    print(f"MARKER:{name}", flush=True)


def say(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def trained_model_dir(out_dir: Path) -> Path:
    """The final model directory the trainer wrote, by highest step."""
    steps = sorted((out_dir / "checkpoints").glob("step_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    if not steps:
        raise A3DriverError(f"{out_dir} holds no checkpoints/step_* directory")
    model = steps[-1] / "model"
    if not model.is_dir():
        raise A3DriverError(f"{steps[-1]} holds no model/ directory")
    return model


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _trainer_bytes() -> int:
    """Derived from the committed measurement, never typed in."""
    terms = json.loads(
        MEMORY_BASIS.read_text())["conversion_to_device_bytes"]["terms_gib"]
    return int((terms["peak_allocated"] + terms["allocator_reserved_slack"]
                + terms["non_pytorch_overhead"]) * 2 ** 30)


class A3Deadline:
    """The session's soft-stop budget, exposed as an operator deadline.

    Invents no timeout: it reads `driver.usd()` and `driver.a.soft_stop_usd`,
    the two numbers every other decision is priced against, so the deadline
    and the spend cannot disagree about what is affordable.
    """

    def __init__(self, driver: "A3Driver", what: str) -> None:
        self.driver, self.what = driver, what

    def check(self, where: str = "") -> None:
        spent = self.driver.usd()
        if spent >= self.driver.a.soft_stop_usd:
            raise A3DriverError(
                f"{self.what}: soft stop reached at ${spent:.4f} of "
                f"${self.driver.a.soft_stop_usd:.4f}{' at ' + where if where else ''}")


class A3Driver:
    """The whole chain, stage by stage, with its evidence written as it goes."""

    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.t0 = time.time()
        self.completed: list[str] = []
        self.ev: dict[str, Any] = {
            "schema": "aadistill.autoinit.a3_evidence/v1",
            "experiment_id": A3S.EXPERIMENT_ID,
            "started_utc": _now(),
            "image_digest": a.image_digest,
            "design_sha256": A3S.design_hash(),
            "session_contract_hash": A3S.A3_SESSION_CONTRACT.contract_hash,
            "stages": {},
        }
        self.seeds = list(A3S.recovery_seeds())
        self.training: dict[int, dict[str, Any]] = {}
        self.scored: dict[int, dict[str, Any]] = {}
        self.parent_step = None
        self.diagnostics: dict[str, Any] = {}
        self._handoffs: list[dict[str, Any]] = []
        self.a3_init_dir: Path | None = None
        self.a3_init_digest: str | None = None
        for d in (AUDIT, TRAIN, EVAL, WORK, AUDIT / "probes", AUDIT / "configs"):
            d.mkdir(parents=True, exist_ok=True)

    # -- budget ------------------------------------------------------------

    def usd(self) -> float:
        elapsed_h = (time.time() - self.t0) / 3600.0
        return self.a.spent_usd + elapsed_h * self.a.rate

    def afford(self, minutes: float, what: str) -> bool:
        projected = self.usd() + (minutes / 60.0) * self.a.rate
        ok = projected <= self.a.soft_stop_usd
        if not ok:
            say(f"  budget refuses {what}: ${projected:.4f} projected against "
                f"a ${self.a.soft_stop_usd:.4f} soft stop")
        return ok

    # -- evidence ----------------------------------------------------------

    def save(self) -> None:
        self.ev["updated_utc"] = _now()
        self.ev["spent_usd"] = round(self.usd(), 4)
        (AUDIT / "a3_evidence.json").write_text(
            json.dumps(self.ev, indent=2) + "\n")

    def complete(self, letter: str, **payload: Any) -> None:
        stage = A3S.stage(letter)
        A3S.assert_stage_order([*self.completed, letter])
        self.completed.append(letter)
        self.ev["stages"][stage.stage_id] = {
            "letter": letter, "stage_id": stage.stage_id, "passed": True,
            "completed_utc": _now(), **payload}
        self.save()
        mark(f"STAGE_PASSED:{letter}")

    def child_env(self) -> dict[str, str]:
        """A detached job inherits only what it is given."""
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/root"),
            "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}",
            "HF_HOME": os.environ.get("HF_HOME", "/workspace/hf"),
            "TOKENIZERS_PARALLELISM": "false",
        }
        for k in ("CUDA_VISIBLE_DEVICES", "HF_TOKEN", "HF_HUB_CACHE"):
            if k in os.environ:
                env[k] = os.environ[k]
        return env

    def gate(self, name: str, cmd: list[str], *, timeout: int,
             python: str | None = None) -> subprocess.CompletedProcess:
        argv = ([python, *cmd[1:]] if python and cmd[0].endswith("python")
                else cmd)
        say(f"  gate {name}: {' '.join(argv[:3])} …")
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, env=self.child_env())

    def runtime_identity(self) -> dict[str, Any]:
        """What this session ran on. Recorded, never assumed."""
        import torch
        import transformers

        return {"image_digest": self.a.image_digest,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
                "driver": os.environ.get("NVIDIA_DRIVER_VERSION")}

    def release_device(self) -> None:
        """Hand the card back between stages, and PROVE it.

        `complete_release` takes the before-snapshot and returns a record;
        `require_released` reads that record. Calling either without the
        snapshot is the signature mismatch a $0 check catches and a pod does
        not.
        """
        gc.collect()
        before = cuda_memory()
        record = complete_release(before)
        require_released(record, what="A3 stage boundary")
        #: WRITTEN, because the artifact spec names it and because "the card
        #: was handed back" is a claim a reader should be able to check.
        self._handoffs.append({"at": _now(), "before": before,
                               "after": record})
        (AUDIT / "a3_device_handoff.json").write_text(
            json.dumps({"schema": "aadistill.autoinit.a3_device_handoff/v1",
                        "releases": self._handoffs}, indent=2) + "\n")

    # -- B: the teacher ----------------------------------------------------

    def stage_b(self) -> None:
        mark("STAGE_START:B")
        import hashlib

        from huggingface_hub import snapshot_download

        binding = json.loads(TEACHER_BINDING.read_text())
        from experiments.phase_c3 import session as CS
        if binding["revision"] != CS.TEACHER_REVISION:
            raise A3DriverError(
                f"teacher binding pins {binding['revision']} and the session "
                f"declares {CS.TEACHER_REVISION}")
        local = snapshot_download(CS.TEACHER_REPO,
                                  revision=CS.TEACHER_REVISION)
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
            raise A3DriverError("teacher verification FAILED: " + "; ".join(bad))
        self.teacher_path = local
        say(f"teacher {CS.TEACHER_REPO}@{CS.TEACHER_REVISION[:12]} verified, "
            f"{len(binding['expected_shard_sha256'])} shards")
        self.complete("B", repo_id=CS.TEACHER_REPO,
                      revision=CS.TEACHER_REVISION, local_path=local,
                      shards_verified=len(binding["expected_shard_sha256"]))

    # -- C: the operator, and everything stage G would die on --------------

    def stage_c(self) -> None:
        """Register, then refuse NOW for anything the scorer would die on.

        The pre-flight is here rather than in stage G on purpose: a battery
        the scorer cannot read is a cheap stop at stage C and an expensive one
        after three trainings. It costs seconds and touches no GPU.
        """
        mark("STAGE_START:C")
        impl = attention_activation.register(replace=True)
        if impl.impl_id != A3S.ATTENTION_IMPL_ID:
            raise A3DriverError(
                f"registered {impl.impl_id}, expected {A3S.ATTENTION_IMPL_ID}")
        #: The calibration profile the path names must resolve now too.
        from experiments.phase_c3 import session as CS
        profile_id = CS.prefix_steps()[-1][1]
        get_profile(profile_id)

        if not BATTERY.is_dir():
            raise A3DriverError(f"the confirmation battery is not staged at {BATTERY}")
        identity = json.loads(BATTERY_IDENTITY.read_text())
        pre = self.gate("scorer_preflight",
                        ["/opt/train/bin/python", str(C1_SCORER), "--help"],
                        timeout=300)
        if pre.returncode != 0:
            raise A3DriverError(
                f"the C1 scorer cannot even print help (rc={pre.returncode}); "
                f"tail: ...{(pre.stdout + pre.stderr)[-800:]}")
        contract = c1_scoring_contract(REPO)
        say(f"operator {impl.impl_id} registered; battery "
            f"{identity.get('artifact')} staged; scoring contract "
            f"{contract['digest'][:12]}")
        self.complete("C", impl_id=impl.impl_id, profile_id=profile_id,
                      battery=identity.get("artifact"),
                      scoring_contract=contract["digest"])

    # -- D: the parent, under its frozen gate ------------------------------

    def stage_d(self) -> None:
        mark("STAGE_START:D")
        if not self.afford(28.0, "parent replay"):
            raise A3DriverError("budget refuses the parent replay")
        from aadistill.initialization.specs.arch import get_adapter
        from experiments.phase_c3.a_bsz3 import A_BSZ1

        spec = A3S.path_spec(workdir_device="cuda")
        adapter = get_adapter("qwen3")
        require_headroom(cuda_memory(), need_bytes=_trainer_bytes(),
                         what="parent replay")
        #: ONE replay of the whole path under A-bsz1's execution exercises
        #: BOTH frozen gates -- the parent on step 3 and the incumbent on
        #: step 4. `materialize_fixed_path` has no index window, so a
        #: prefix-only replay would need a second spec and a second parent
        #: identity to keep in step; this keeps one.
        #:
        #: A_BSZ1 is passed explicitly. The module default is
        #: `micro_batch_size=4`, the frozen parent is pinned at 1, and a
        #: default-executed prefix has already moved a DEPTH decision once.
        try:
            steps = materialize_fixed_path(
                spec, adapter=adapter,
                root_loader=lambda: adapter.load(
                    self.teacher_path, dtype="bfloat16", device="cuda"),
                workdir=WORK / "replay", repo_root=REPO, execution=A_BSZ1,
                deadline=A3Deadline(self, "parent replay"))
        except FixedPathDigestMismatch as exc:
            mark("A3_REPLAY_MISMATCH")
            raise A3DriverError(
                f"the frozen path did not reproduce: {exc}. Every arm is "
                "defined relative to that parent, so this is an integrity "
                "failure to repair and never a result.") from exc

        self.parent_step = steps[-2]
        parent_got = self.parent_step.identity.artifact_digest
        parent_want = A3S.expected_parent_digest()
        if parent_got != parent_want:
            mark("A3_REPLAY_MISMATCH")
            raise A3DriverError(
                f"parent digest {parent_got[:12]} != frozen "
                f"{parent_want[:12]}")
        #: THE ASYMMETRIC GATE, enforced by the RUN and not by the path: the
        #: spec carries no ATTENTION pin, because the same spec produces
        #: A-bsz3's artifact and pinning it would decide the experiment.
        incumbent_got = steps[-1].identity.artifact_digest
        incumbent_want = A3S.protocol_gate(A3S.REFERENCE_PROTOCOL)
        if incumbent_got != incumbent_want:
            mark("A3_REPLAY_MISMATCH")
            raise A3DriverError(
                f"A_bsz1 built {incumbent_got[:12]} and the frozen incumbent "
                f"is {incumbent_want[:12]}. A-bsz1 IS canonical A, so the "
                "attempt75 controls A3 reuses do not describe what this "
                "session built. Repair; never interpret.")
        write_replay_record(
            spec, steps, AUDIT / "a3_replay.json",
            runtime=self.runtime_identity(),
            root_binding=json.loads(TEACHER_BINDING.read_text()))
        say(f"parent {parent_got[:12]} and incumbent {incumbent_got[:12]} both "
            "reproduced under their frozen gates")
        self.complete("D", parent_artifact_digest=parent_got,
                      incumbent_artifact_digest=incumbent_got,
                      path_hash=spec.spec_hash,
                      checkpoint=_rel(Path(self.parent_step.checkpoint_path)))

    # -- E: the diagnostics, and the asymmetric gate -----------------------

    def stage_e(self) -> None:
        """Both protocols from the ALREADY-VERIFIED parent, interleaved.

        A-bsz1 is gated on the frozen incumbent. A-bsz3 is NOT gated: what it
        builds is the finding. A differing digest is recorded and the chain
        continues; only the declared integrity stops end it.
        """
        mark("STAGE_START:E")
        from compare_a_bsz3 import structural_half
        from experiments.phase_c3.a_bsz3 import (
            execution_comparison, item_token_counts,
        )

        if self.parent_step is None:
            raise A3DriverError("stage E has no verified parent")
        spec = A3S.path_spec(workdir_device="cuda")
        start = len(spec.steps) - 1
        verified = VerifiedSuffix(
            start_index=start, parent=self.parent_step,
            expected_parent_artifact_digest=A3S.expected_parent_digest(),
            expected_path_hash=spec.spec_hash,
            prefix_reference_steps=tuple(spec.steps[:start]),
            expected_suffix_steps=((A3S.ATTENTION_IMPL_ID,
                                    spec.steps[-1].profile_id),))

        from aadistill.initialization.specs.arch import get_adapter

        adapter = get_adapter("qwen3")
        rounds = A3S.DIAGNOSTIC_ROUNDS
        if not self.afford(0.25 * 2 * rounds * 1.5, "diagnostics"):
            raise A3DriverError("budget refuses the diagnostic rounds")

        lengths = item_token_counts()
        out = structural_half(
            spec, adapter=adapter,
            root_loader=lambda: adapter.load(
                self.parent_step.checkpoint_path, dtype="bfloat16",
                device="cuda"),
            verified=verified, workdir=WORK / "diagnostics", repo_root=REPO,
            device="cuda", repeats=rounds,
            expected_incumbent_digest=A3S.expected_incumbent_digest(),
            zero_cost=execution_comparison(lengths))

        comparison = out["_comparison"]
        if comparison.get("INVALID"):
            mark("A3_INTEGRITY_FAILURE")
            raise A3DriverError(
                "the diagnostics are VOID, not a result: "
                + "; ".join(comparison["INVALID"]))

        self.diagnostics = out
        (AUDIT / "a3_diagnostics.json").write_text(
            json.dumps(out, indent=2, default=str) + "\n")
        #: THE ARM IDENTITIES, as their own record. The maintainer's report
        #: asks for artifact identities and the artifact spec names this file;
        #: burying them inside the diagnostics blob would make a reader parse
        #: a comparison to learn what was built.
        (AUDIT / "a3_arm_identities.json").write_text(json.dumps({
            "schema": "aadistill.autoinit.a3_arm_identities/v1",
            "experiment_id": A3S.EXPERIMENT_ID,
            "shared_parent": {
                "artifact_digest": A3S.expected_parent_digest(),
                "gated": True},
            "protocols": {
                name: {
                    "execution": out[name]["execution"],
                    "artifact_digest": out[name]["artifact_digest"],
                    "result_spec_hash": out[name]["result_spec_hash"],
                    "expected_artifact_digest": A3S.protocol_gate(name),
                    "gated": A3S.protocol_gate(name) is not None,
                    "calibration_tokens": out[name]["calibration_tokens"],
                    "peak_vram_gib": out[name]["peak_vram_gib"],
                } for name in A3S.PROTOCOL_EXECUTIONS},
            "artifact_digest_identical":
                comparison["artifact_digest_identical"],
            "classification": comparison["classification"],
            "_the_gate_is_asymmetric": (
                "A_bsz1 must rebuild the frozen incumbent; A_bsz3 is pinned "
                "nowhere, because what it builds is the finding."),
        }, indent=2) + "\n")

        #: Round 0 of A_bsz3 IS the initialization the probes train from.
        a3 = out[A3S.TREATMENT_PROTOCOL]
        self.a3_init_digest = a3["artifact_digest"]
        self.a3_init_dir = WORK / "diagnostics" / A3S.TREATMENT_PROTOCOL / "rep0"
        identical = comparison["artifact_digest_identical"]
        say(f"A_bsz1 {out['A_bsz1']['artifact_digest'][:12]} == frozen "
            f"incumbent; A_bsz3 {self.a3_init_digest[:12]}; "
            f"{'IDENTICAL' if identical else 'DIFFERENT'} -> "
            f"{comparison['classification']}")
        mark(f"A3_DIGESTS:{'IDENTICAL' if identical else 'DIFFERENT'}")
        self.complete(
            "E", classification=comparison["classification"],
            artifact_digest_identical=identical,
            a_bsz1_artifact_digest=out["A_bsz1"]["artifact_digest"],
            a_bsz3_artifact_digest=self.a3_init_digest,
            incumbent_gate=comparison["incumbent_digest_gate"],
            rounds_completed=comparison["rounds_completed"],
            runtime=comparison["runtime"],
            kept_head_selection=comparison["kept_head_selection"],
            head_scores=comparison["head_scores"],
            execution_counters=comparison["execution_counters"],
            peak_vram_gib={p: out[p]["peak_vram_gib"]
                           for p in A3S.PROTOCOL_EXECUTIONS},
            _a_differing_digest_is_a_finding=(
                "recorded and continued. Only the declared integrity stops "
                "end the chain."))

    # -- F: three recovery trainings ---------------------------------------

    def descriptors(self) -> list[dict[str, Any]]:
        if self.a3_init_dir is None or self.a3_init_digest is None:
            raise A3DriverError("stage F has no A_bsz3 initialization")
        return [{
            "probe_id": A3S.probe_id(seed),
            "arm": A3S.TREATMENT_ARM,
            "seed": seed,
            "initialization_artifact_digest": self.a3_init_digest,
            "student_path": _rel(self.a3_init_dir),
        } for seed in self.seeds]

    def probe_config(self, d: dict[str, Any]) -> Path:
        frozen = json.loads(FROZEN_RECIPE.read_text())
        name = d["probe_id"]
        derived = {**frozen, "run_name": name,
                   "out_dir": f"artifacts/stage3/a3/{name}",
                   "data_dir": PACK_DIR, "seed": d["seed"],
                   "student_path": d["student_path"],
                   "_purpose": (
                       f"A3 recovery probe, arm {d['arm']}, seed {d['seed']}. "
                       "Identical recovery to the attempt75 controls; the only "
                       "intended difference is the ATTENTION step's execution "
                       f"protocol. Derived from {FROZEN_RECIPE.name} by "
                       "overriding run identity, pack path, seed and "
                       "student_path.")}
        diff = sorted(k for k in set(frozen) | set(derived)
                      if frozen.get(k) != derived.get(k))
        if not set(diff) <= A3_PROBE_OVERRIDES:
            raise A3DriverError(
                f"{name}: the derived probe config differs from the frozen "
                f"recipe in {sorted(set(diff) - A3_PROBE_OVERRIDES)}, outside "
                f"the allowed override set {sorted(A3_PROBE_OVERRIDES)}. The "
                "recovery recipe is frozen science and a drifting probe "
                "config is not comparable to the controls.")
        path = AUDIT / "configs" / f"{name}.json"
        path.write_text(json.dumps(derived, indent=2) + "\n")
        return path

    def train_one(self, name: str, config: Path) -> Path:
        """Spawn the recovery trainer. ONE of the two hardware seams."""
        rc = subprocess.run(
            ["/opt/train/bin/python", str(TRAINER), "--config", str(config)],
            capture_output=True, text=True,
            timeout=int(self.a.probe_train_minutes * 60 * 2),
            env=self.child_env())
        (AUDIT / f"{name}_train_tail.log").write_text(
            (rc.stdout + rc.stderr)[-1500:])
        if rc.returncode != 0:
            raise A3DriverError(
                f"{name}: training failed rc={rc.returncode}; tail: "
                f"...{(rc.stdout + rc.stderr)[-1200:]}")
        return TRAIN / name

    def preserve_probe(self, name: str, model_dir: Path,
                       record: dict[str, Any]) -> dict[str, Any]:
        """Push a finished probe's weights to durable storage immediately.

        Not at closeout: C1 attempt 17 trained six probes over ten hours and
        lost every one because a LATER stage failed and the pod was deleted.
        PRESERVATION IS NOT PERMISSION -- it authorizes no pooling, resuming or
        reuse across attempts.
        """
        files = {}
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=relay_token())
            prefix = f"{PRESERVED_PREFIX}/{self.a.run_id}/{name}"
            total = 0
            for p in sorted(model_dir.iterdir()):
                if not p.is_file():
                    continue
                api.upload_file(path_or_fileobj=str(p), path_in_repo=f"{prefix}/{p.name}",
                                repo_id=RELAY, repo_type="model")
                files[p.name] = sha256_file(p)
                total += p.stat().st_size
            return {"preserved": True, "relay_repo": RELAY,
                    "relay_prefix": prefix, "files": files, "bytes": total,
                    "authorizes": ("nothing. Preservation and reuse are "
                                   "separate decisions.")}
        except Exception as exc:                       # noqa: BLE001
            #: NON-RAISING by design: a preservation failure must not destroy a
            #: completed, verified probe. It is recorded with its exact reason
            #: so a reader can see that the mechanism ran and what refused it --
            #: C1 attempt 18 exercised this path on six probes and preserved
            #: none, every upload refused for private-storage quota.
            say(f"  {name}: preservation FAILED and is recorded: {exc}")
            return {"preserved": False, "reason": f"{type(exc).__name__}: {exc}",
                    "files": files}

    def stage_f(self) -> None:
        mark("STAGE_START:F")
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
                    self.training[d["seed"]] = record
                    continue
            if not self.afford(self.a.probe_train_minutes, name):
                raise A3DriverError(
                    f"budget refuses {name}; no probe is skipped to make "
                    "progress -- the estimand is a paired difference over "
                    "three seeds and two is not that quantity")
            config = self.probe_config(d)
            t = time.time()
            out_dir = self.train_one(name, config)
            model_dir = trained_model_dir(out_dir)
            record = {
                "schema": "aadistill.autoinit.a3_training_completion/v1",
                **{k: d[k] for k in ("probe_id", "arm", "seed",
                                     "initialization_artifact_digest")},
                "config": _rel(config), "config_sha256": sha256_file(config),
                "out_dir": _rel(out_dir), "model_dir": _rel(model_dir),
                "train_minutes": round((time.time() - t) / 60, 2),
                "run_completion": _rel(out_dir / "run_completion.json"),
                "evaluated": False, "complete": True,
            }
            record["preserved"] = self.preserve_probe(name, model_dir, record)
            journal.write_text(json.dumps(record, indent=2) + "\n")
            self.training[d["seed"]] = record
            self.ev["probes_trained"] = len(self.training)
            self.save()
            mark(f"PROBE_TRAINED:{name}")
            say(f"  {name}: trained in {record['train_minutes']:.1f} min")

        self.require_all_trained()
        self.complete("F", probes_trained=len(self.training),
                      completions=sorted(r["probe_id"]
                                         for r in self.training.values()))

    def require_all_trained(self) -> None:
        """Stage G's precondition, in one place the driver and harness share."""
        if len(self.training) != len(self.seeds):
            raise A3DriverError(
                f"stage G requires {len(self.seeds)} training completions and "
                f"found {len(self.training)}; the confirmation battery is "
                "evaluated once per fully trained probe and never before")

    # -- G: three evaluations ----------------------------------------------

    def generate_one(self, name: str, package: Path, gen_dir: Path,
                     sets: list[str]) -> None:
        """Run the uncapped evaluator. THE OTHER hardware seam.

        The flag set is the evaluator's REAL one -- `--prompts` takes the
        battery's per-set jsonl files and the output flag is `--out-dir`. A
        plausible-looking `--battery/--out/--sets` would have been an argparse
        failure after three trainings, which is the class of defect that has
        ended a paid attempt here before.
        """
        rc = subprocess.run(
            ["/opt/vllm/bin/python", str(UNCAPPED_EVAL),
             "--model", str(package), "--label", name,
             "--prompts", *[str(BATTERY / f"{s}.jsonl") for s in sets],
             "--out-dir", str(gen_dir), "--diagnostics"],
            capture_output=True, text=True,
            timeout=int(self.a.probe_eval_minutes * 60 * 3),
            env=self.child_env())
        (AUDIT / f"{name}_eval_tail.log").write_text(
            (rc.stdout + rc.stderr)[-1500:])
        if rc.returncode != 0:
            raise A3DriverError(
                f"{name}: generation failed rc={rc.returncode}; tail: "
                f"...{(rc.stdout + rc.stderr)[-1200:]}")

    def stage_g(self) -> None:
        mark("STAGE_START:G")
        self.require_all_trained()
        identity = json.loads(BATTERY_IDENTITY.read_text())
        sets = sorted(identity["sets"]) if "sets" in identity else [
            "gsm8k", "math_verified", "multihop", "rag", "knowledge", "tool",
            "code"]
        attested = self.attest()
        self.ev["attested_evaluation_protocol_hash"] = \
            attested["evaluation_protocol_hash"]
        for seed in self.seeds:
            record = self.training[seed]
            name = record["probe_id"]
            result = AUDIT / f"{name}_c1_confirmation.json"
            if result.is_file():
                say(f"  {name}: already scored")
                self.scored[seed] = json.loads(result.read_text())
                continue
            if not self.afford(self.a.probe_eval_minutes, f"{name} eval"):
                raise A3DriverError(f"budget refuses the evaluation of {name}")
            package = EVAL / f"{name}_package"
            build_evaluation_package(
                REPO / record["model_dir"], tokenizer_source=TOKENIZER_SOURCE,
                dest=package, expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
            gen_dir = EVAL / name
            gen_dir.mkdir(parents=True, exist_ok=True)
            self.generate_one(name, package, gen_dir, sets)
            #: BEFORE the scorer, fail-closed.
            admission = self.admit_generation(name, gen_dir)
            #: `--arm` is NOT passed. Its choices are C1's two ROLES, and
            #: passing an experiment's arm id there is what ended attempt66
            #: one stage from a verdict.
            rc = self.gate(
                f"score:{name}",
                ["/opt/train/bin/python", str(C1_SCORER),
                 "--generations", str(gen_dir), "--battery", str(BATTERY),
                 "--label", name, "--seed", str(seed),
                 "--out", str(result),
                 "--per-sample", str(AUDIT / f"{name}_per_sample.jsonl"),
                 "--init-digest", record["initialization_artifact_digest"],
                 "--trained-run", str(REPO / record["run_completion"]),
                 "--generation-fingerprint",
                 admission["generation_fingerprint"]],
                timeout=3600)
            if rc.returncode != 0:
                raise A3DriverError(
                    f"{name}: scoring failed rc={rc.returncode}; tail: "
                    f"...{(rc.stdout + rc.stderr)[-1200:]}")
            self.scored[seed] = json.loads(result.read_text())
            record["evaluated"] = True
            (AUDIT / "probes" / f"{name}.training.json").write_text(
                json.dumps(record, indent=2) + "\n")
            self.ev["probes_scored"] = len(self.scored)
            self.save()
            mark(f"PROBE_SCORED:{name}")
            say(f"  {name}: scored, correct_overall "
                f"{self.scored[seed].get('correct_overall')}")
        self.complete("G", probes_scored=len(self.scored),
                      attested_evaluation_protocol_hash=attested[
                          "evaluation_protocol_hash"],
                      generation_protocol_fingerprint=attested[
                          "generation_protocol_fingerprint"])

    def attest(self) -> dict[str, Any]:
        """The protocol A3's generations must be comparable to.

        A statement about the RUNTIME, made once from an engine probe. It is
        not evidence about any particular probe's rollouts -- that is
        `admit_generation`, which reconstructs the protocol from each probe's
        own summaries before the scorer runs.
        """
        sample = next(iter(self.training.values()))
        package = EVAL / "_attestation_package"
        report = build_evaluation_package(
            REPO / sample["model_dir"], tokenizer_source=TOKENIZER_SOURCE,
            dest=package, expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
        engine = self.gate(
            "engine_probe",
            ["/opt/vllm/bin/python", str(ENGINE_PROBE), "--model", str(package),
             "--out", str(AUDIT / "engine_probe.json"),
             "--image-digest", self.a.image_digest],
            timeout=1800)
        if engine.returncode != 0:
            raise A3DriverError(
                f"engine probe rc={engine.returncode}; tail: "
                f"...{(engine.stdout + engine.stderr)[-1200:]}")
        observed = json.loads((AUDIT / "engine_probe.json").read_text())

        gen = declared_generation_protocol().materialized(
            generation_source_digest=generation_source_digest(REPO)["digest"],
            degeneration_source_digest=sha256_file(
                REPO / "src/aadistill/evaluation/degeneration.py"),
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
        gen.require_materialized(context="A3 stage G")

        manifest = json.loads((BATTERY / "manifest.json").read_text())
        contract = c1_scoring_contract(REPO)
        self.evaluation_protocol = RecoveryEvaluationProtocol(
            generation=gen, scoring_contract=contract["contract"],
            scoring_digest=contract["digest"],
            battery_artifact=manifest["artifact"],
            battery_manifest_sha256=manifest["manifest_sha256"],
            battery_content_sha256=manifest["content_sha256"])
        attested = {
            "schema": "aadistill.autoinit.a3_attested_protocol/v1",
            "generated_utc": _now(),
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
                "observed_chat_template_sha256":
                    observed["chat_template_sha256"],
                "packaging": report["tokenizer_source_rule"],
            },
            "_must_equal_attempt75": (
                "A3 reuses attempt75's controls, so this fingerprint is "
                "checked against theirs OFF POD before any delta is "
                "reported. C2's confirmation field carried three "
                "fingerprints and produced no canonical verdict."),
        }
        attested["report_sha256"] = sha256_json(attested)
        (AUDIT / "a3_attested_evaluation_protocol.json").write_text(
            json.dumps(attested, indent=2) + "\n")
        say(f"attested: protocol {attested['evaluation_protocol_hash'][:12]}…, "
            f"tokenizer {observed['tokenizer_sha256'][:12]}…")
        return attested

    def admit_generation(self, name: str, gen_dir: Path) -> dict[str, Any]:
        """Refuse a probe whose generations were not produced under the protocol.

        Fail-closed and BEFORE the scorer runs. A probe generated under a
        drifted protocol must not be scored, must not be followed by another
        probe, and must not reach the comparison. The summaries stay on disk
        either way, so a refusal is a diagnosis rather than a loss.
        """
        summaries = [json.loads(q.read_text())
                     for q in sorted(gen_dir.glob("*.json"))
                     if not q.name.endswith(".generations.jsonl")]
        if not summaries:
            raise A3DriverError(
                f"{name}: the generation directory holds no per-set summaries, "
                "so the protocol it ran under cannot be reconstructed")
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
        except Exception as exc:                            # noqa: BLE001
            record["comparable"] = False
            record["reason"] = str(exc)[-1500:]
            (AUDIT / f"{name}_generation_admission.json").write_text(
                json.dumps(record, indent=2) + "\n")
            raise A3DriverError(
                f"{name}: the generations were not produced under the attested "
                "evaluation protocol, so this probe cannot be scored and no "
                f"later probe may be evaluated. {exc}") from exc
        record["comparable"] = True
        (AUDIT / f"{name}_generation_admission.json").write_text(
            json.dumps(record, indent=2) + "\n")
        say(f"  {name}: generation protocol admitted "
            f"({record['evaluation_protocol_hash'][:12]}…)")
        return record

    # -- H: preserve and package -------------------------------------------

    def stage_h(self) -> None:
        """The evidence the OFF-POD comparison will read, and its manifest.

        No decision is computed here. attempt75's lesson is that a complete
        measurement must not depend on the pod surviving an aggregation.
        """
        mark("STAGE_START:H")
        records = []
        for seed in self.seeds:
            t, s = self.training[seed], self.scored[seed]
            name = t["probe_id"]
            result = AUDIT / f"{name}_c1_confirmation.json"
            per_sample = AUDIT / f"{name}_per_sample.jsonl"
            #: ARM VOCABULARY PASSED EXPLICITLY. `allowed_arms` defaults to
            #: C1's two ROLES, and attempt75 died on `unknown arm` AFTER a
            #: complete nine-probe measurement.
            #: COUNTS AND RATES FROM THE SCORED RECORD, not empty maps.
            #: `C1ProbeRecord.__post_init__` requires five counts and refuses
            #: a row total that is not 950 or a scorable total that is not
            #: 850 -- so an empty map does not merely omit evidence, it
            #: REFUSES, and it would have refused here after three trainings
            #: and three evaluations. The $0 rehearsal is what caught it.
            counts = {k: s[k] for k in ("n", "usable", "correct",
                                        "n_scorable", "usable_scorable")}
            rates = {k: s.get(k) for k in ("usable_rollout_rate",
                                           "correct_overall",
                                           "correct_given_usable")}
            records.append(C1ProbeRecord(
                probe_id=name, arm=t["arm"], seed=seed,
                initialization_artifact_digest=t["initialization_artifact_digest"],
                trained_run={"run_completion": t["run_completion"],
                             "config_sha256": t["config_sha256"]},
                result_path=str(result), result_sha256=sha256_file(result),
                per_sample_path=str(per_sample),
                per_sample_sha256=sha256_file(per_sample),
                generations=s.get("generations", {}),
                counts=counts, rates=rates,
                per_capability=s.get("per_capability", {}),
                scoring_contract=s.get("scoring_contract", {}),
                battery=s.get("battery", {}),
                observed_generation_fingerprint=s.get(
                    "generation_protocol_fingerprint", ""),
                observed_evaluation_protocol_hash=self.ev.get(
                    "attested_evaluation_protocol_hash", ""),
                #: PASSED EXPLICITLY. The default is C1's two ROLES and
                #: attempt75 died on it after a complete measurement.
                allowed_arms=(A3S.TREATMENT_ARM,),
            ))
        inventory = {
            "schema": "aadistill.phase_a3.probe_inventory/v1",
            "run_id": self.a.run_id,
            "experiment_id": A3S.EXPERIMENT_ID,
            "arm": A3S.TREATMENT_ARM,
            "seeds": self.seeds,
            "n_probes": len(records),
            "a_bsz3_initialization_digest": self.a3_init_digest,
            "probes": {r.probe_id: {
                "seed": r.seed, "arm": r.arm,
                "initialization_artifact_digest":
                    r.initialization_artifact_digest,
                "result_sha256": r.result_sha256,
                "per_sample_sha256": r.per_sample_sha256,
                "counts": dict(r.counts), "rates": dict(r.rates),
                "preserved": self.training[r.seed].get("preserved", {}),
            } for r in records},
            "_the_comparison_runs_off_pod": (
                "scripts/autoinit/aggregate_a3.py reads this evidence at $0. "
                "No verdict is computed on the meter."),
        }
        (AUDIT / "a3_probe_inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n")
        preserved = sum(1 for r in self.training.values()
                        if (r.get("preserved") or {}).get("preserved"))
        say(f"packaged {len(records)} probes; {preserved} preserved durably")
        self.complete("H", probes=len(records), preserved=preserved,
                      inventory=_rel(AUDIT / "a3_probe_inventory.json"))

    # -- the chain ---------------------------------------------------------

    def run(self) -> int:
        stages = {"B": self.stage_b, "C": self.stage_c, "D": self.stage_d,
                  "E": self.stage_e, "F": self.stage_f, "G": self.stage_g,
                  "H": self.stage_h}
        try:
            for letter in A3S.STAGE_LETTERS:
                stages[letter]()
            mark("ALL_DONE")
            say("A3 complete on pod. The comparison runs off pod at $0.")
            return 0
        except Exception as exc:                        # noqa: BLE001
            self.ev["failure"] = {
                "stage": (self.completed[-1] + "->next" if self.completed
                          else "before B"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
            self.save()
            say(f"A3 FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            mark("A3_FAILED")
            return 40


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image-digest", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--run-id", default="a3_attempt1")
    ap.add_argument("--probe-train-minutes", type=float, default=61.76)
    ap.add_argument("--probe-eval-minutes", type=float, default=26.85)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.soft_stop_usd > a.authorized_usd:
        raise SystemExit(
            f"soft stop ${a.soft_stop_usd} exceeds the authorized "
            f"${a.authorized_usd}; a session may not plan past its grant")
    return A3Driver(a).run()


if __name__ == "__main__":
    raise SystemExit(main())
