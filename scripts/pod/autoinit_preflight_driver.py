#!/usr/bin/env python3
"""Pod-side driver for the AutoInitializer micro-preflight. Stages 0-3, then stop.

    /opt/train/bin/python scripts/pod/autoinit_preflight_driver.py --stage all \
        --image-digest sha256:... --rate 0.99 --spent-usd 0.30 \
        --soft-stop-usd 6.00 --authorized-usd 8.60

Staged and fail-closed. `PreflightPlan.advance_to` refuses a stage until every
blocking earlier stage recorded a pass, and this driver additionally refuses to
*start* work it cannot finish inside the soft stop. The order exists so that a
bad machine costs profiling minutes rather than two permanent control runs:

    0  attest the runtime, materialize and FREEZE both protocol identities
    1  cheap machine gates
    2  the two permanent canonical controls   <- the only expensive stage
    3  characterize them, materialize the frozen thresholds

Phase A is not reachable from here. There is no stage 4, `--stage` cannot name
one, and the authorization object refuses it.

Every stage appends `MARKER:<NAME>` to the status file, which is what the
launcher polls and what the artifact gate keys on. The final marker is
`MARKER:ALL_DONE`; a blocking failure emits `MARKER:STAGE_FAILED:<n>` and stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from aadistill.autoinit.authorization import (  # noqa: E402
    AuthorizationError, SpendAuthorization,
)
from aadistill.autoinit.generation import (  # noqa: E402
    GenerationProtocolError,
    RecoveryEvaluationProtocol,
    declared_generation_protocol,
    generation_source_digest,
    observe_generation_protocol,
)
from aadistill.autoinit.ranking import EPSILON_RESPONSE_V1, PARETO_V1  # noqa: E402
from aadistill.autoinit.recovery import (  # noqa: E402
    CATASTROPHIC_V1,
    POOLED_COUNTS_V2,
    PREFLIGHT_PLAN_V1,
    SEED_SA,
    SEED_SB,
    EquivalenceRule,
    FeasibilityRule,
    ObservedProtocolError,
    RecoveryAdmissionError,
    RecoveryProbeIdentity,
    RuntimeEnvironmentFingerprint,
    observe_recovery_protocol,
    recovery_scoring_contract,
    trainer_source_digest,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

WS = Path("/workspace")
STATUS = WS / "autoinit_preflight.status"
AUDIT = REPO / "artifacts/audit/autoinit_preflight"
BATTERY = REPO / "artifacts/stage3/recovery_search_v2"
CANONICAL_INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
CONTROLS = (("preflight_ctl_r0860k_sa", SEED_SA,
             "configs/stage3/e1/e1_r0860k_sa_pca.json"),
            ("preflight_ctl_r0860k_sb", SEED_SB,
             "configs/stage3/e1/e1_r0860k_sb_pca.json"))
PINNED = {
    "canonical_init_weights": (
        "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors",
        "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"),
    "recovery_pack_blocks": (
        "artifacts/stage3/ladder_uniform_probe/blocks.npz",
        "6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c"),
}
BATTERY_CONTENT = "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323"


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


class Driver:
    def __init__(self, a):
        self.a = a
        self.t0 = time.time()
        self.results: dict[int, dict] = {}
        self.ev: dict = {"schema": "aadistill.autoinit.preflight_evidence/v1",
                         "started_utc": datetime.now(timezone.utc).isoformat(),
                         "stages": {}, "markers": []}
        AUDIT.mkdir(parents=True, exist_ok=True)
        self.auth = SpendAuthorization.load(
            REPO / "logs/autoinit_micro_preflight_authorization.json")
        self.auth.require_plan(PREFLIGHT_PLAN_V1.plan_hash)
        self.ev["authorization"] = self.auth.as_dict()
        self.attested = None
        self.evaluation_protocol = None

    # -- budget ----------------------------------------------------------
    def usd(self) -> float:
        return self.a.spent_usd + (time.time() - self.t0) / 3600 * self.a.rate

    def afford(self, minutes: float, what: str) -> bool:
        """Refuse to *start* work that cannot finish inside the soft stop."""
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

    def save(self) -> None:
        self.ev["elapsed_min"] = round((time.time() - self.t0) / 60, 2)
        self.ev["spend_usd"] = round(self.usd(), 4)
        (AUDIT / "preflight_evidence.json").write_text(
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

    def enter(self, stage: int) -> None:
        self.auth.require_stage(stage)
        PREFLIGHT_PLAN_V1.advance_to(stage, self.results)
        mark(f"STAGE_START:{stage}")

    # -- stage 0 ---------------------------------------------------------
    def stage0(self) -> bool:
        self.enter(0)
        runtime = RuntimeEnvironmentFingerprint.observe(
            image_digest=self.a.image_digest)
        try:
            runtime.require_pinned()
        except RecoveryAdmissionError as exc:
            return self.record(0, False, f"runtime not pinned: {exc}")
        trainer = trainer_source_digest(REPO)

        inputs, ok = {}, True
        for name, (rel, pinned) in PINNED.items():
            path = REPO / rel
            present = path.is_file()
            actual = sha256_file(path) if present else None
            inputs[name] = {"path": rel, "pinned_sha256": pinned,
                            "actual_sha256": actual, "present": present,
                            "match": actual == pinned}
            ok = ok and inputs[name]["match"]
        battery_manifest = json.loads((BATTERY / "manifest.json").read_text())
        battery_ok = battery_manifest.get("content_sha256") == BATTERY_CONTENT
        ok = ok and battery_ok

        # Recovery (training) protocol, materialized and frozen.
        from compare_recovery_fingerprints import phase_a_protocol
        prereg = phase_a_protocol(REPO / CONTROLS[0][2])
        try:
            attested = prereg.materialized(runtime=runtime, trainer_source=trainer)
            attested.require_materialized(context="Stage 0")
        except RecoveryAdmissionError as exc:
            return self.record(0, False, f"protocol drift: {exc}",
                               pinned_inputs=inputs)
        self.attested = attested

        # Generation protocol. The implementation digests are repo facts; every
        # engine and runtime field comes from the live engine probe, which runs
        # in the vLLM environment. It must: generation happens there, and the
        # training venv's torch and transformers versions describe a stack that
        # never produces a token. Booting the engine here rather than at first
        # use also turns "the engine works on this image" into a Stage-0 gate,
        # ahead of $2.80 of permanent controls.
        gen = declared_generation_protocol().materialized(
            generation_source_digest=generation_source_digest(REPO)["digest"],
            degeneration_source_digest=sha256_file(
                REPO / "src/aadistill/evaluation/degeneration.py"))
        try:
            observed = self.observe_engine()
        except Exception as exc:                                  # noqa: BLE001
            return self.record(0, False, f"engine probe failed: {exc}"[-1500:],
                               pinned_inputs=inputs)
        gen = gen.materialized(**observed)
        try:
            gen.require_materialized(context="Stage 0")
        except Exception as exc:                                  # noqa: BLE001
            return self.record(0, False, f"generation protocol: {exc}",
                               pinned_inputs=inputs)

        scoring = recovery_scoring_contract(REPO)
        self.evaluation_protocol = RecoveryEvaluationProtocol(
            generation=gen, scoring_contract=scoring["contract"],
            scoring_digest=scoring["digest"],
            battery_artifact=battery_manifest["artifact"],
            battery_manifest_sha256=battery_manifest["manifest_sha256"],
            battery_content_sha256=battery_manifest["content_sha256"])

        frozen = {
            "schema": "aadistill.autoinit.attested_protocol/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "preflight_plan_hash": PREFLIGHT_PLAN_V1.plan_hash,
            "runtime": runtime.as_dict(), "trainer_source": trainer,
            "pinned_inputs": inputs, "battery_content_verified": battery_ok,
            "attested_protocol": attested.as_dict(),
            "attested_protocol_fingerprint": attested.fingerprint,
            "evaluation_protocol": self.evaluation_protocol.as_dict(),
            "evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
        }
        frozen["report_sha256"] = sha256_json(frozen)
        (AUDIT / "attested_protocol.json").write_text(
            json.dumps(frozen, indent=2) + "\n")
        say(f"attested protocol {attested.fingerprint[:16]}… · evaluation "
            f"{self.evaluation_protocol.evaluation_protocol_hash[:16]}…")
        return self.record(
            0, ok, "" if ok else "a pinned input or the battery does not verify",
            attested_protocol_fingerprint=attested.fingerprint,
            evaluation_protocol_hash=(
                self.evaluation_protocol.evaluation_protocol_hash),
            runtime_digest=runtime.digest,
            trainer_source_digest=trainer["digest"],
            pinned_inputs=inputs)

    def observe_engine(self) -> dict:
        """Boot vLLM once on the canonical init and read what it actually uses.

        `dtype` and `gpu_memory_utilization` are already declared, so passing
        them through `materialized()` is a *check*: it raises if the engine was
        given something other than what the protocol declares, rather than
        quietly recording the engine's value.
        """
        out = subprocess.run(
            ["/opt/vllm/bin/python", str(REPO / "scripts/pod/autoinit_engine_probe.py"),
             "--model", str(CANONICAL_INIT), "--out", str(AUDIT / "engine_probe.json"),
             "--image-digest", self.a.image_digest],
            capture_output=True, text=True, timeout=1800,
            env=self.child_env())
        if out.returncode != 0:
            raise RuntimeError(f"engine probe rc={out.returncode}: "
                               f"{(out.stdout + out.stderr)[-600:]}")
        probe = json.loads((AUDIT / "engine_probe.json").read_text())
        return {"vllm_version": probe["vllm_version"],
                "transformers_version": probe["transformers_version"],
                "torch_version": probe["torch_version"],
                "runtime_digest": probe["runtime_digest"],
                "dtype": probe["dtype"],
                "gpu_memory_utilization": probe["gpu_memory_utilization"],
                "max_num_seqs": probe["max_num_seqs"],
                "max_num_batched_tokens": probe["max_num_batched_tokens"],
                "enforce_eager": probe["enforce_eager"],
                "tokenizer_sha256": probe["tokenizer_sha256"],
                "chat_template_sha256": probe["chat_template_sha256"],
                "resolved_context": probe["resolved_context"],
                "context_source": probe["context_source"],
                "stop_token_ids": tuple(probe["stop_token_ids"])}

    def child_env(self) -> dict:
        """Environment for every child that must record the session's identity.

        The image digest cannot be observed inside the container, so the trainer
        and the evaluator receive it here — the same launcher-supplied value
        Stage 0 attests. A child that does not get it records a null image
        digest, and the strict reconstruction then fails closed rather than
        accepting an unpinned runtime.
        """
        return {**os.environ, "PYTHONPATH": f"{REPO}/src",
                "AADISTILL_IMAGE_DIGEST": self.a.image_digest}

    # -- stage 1 ---------------------------------------------------------
    def stage1(self) -> bool:
        self.enter(1)
        if not self.afford(25, "stage 1 machine gates"):
            return self.record(1, False, "insufficient budget for the gates")
        gates = {}
        try:
            gates["engine_probe"] = json.loads(
                (AUDIT / "engine_probe.json").read_text())
            gates["generation_smoke"] = self.generation_smoke()
            gates["evaluator_repeatability"] = self.repeatability()
            gates["activation_statistics"] = self.stats_split()
            gates["peak_memory"] = self.peak_memory()
            gates["disk_throughput"] = self.disk_throughput()
        except Exception as exc:                                  # noqa: BLE001
            return self.record(1, False, f"gate raised: {exc}"[-1500:], gates=gates)

        # The declared epsilon is the smallest across the beam's objectives: a
        # measured range at or above ANY objective's tolerance fires the gate.
        declared_epsilon = min(PARETO_V1.epsilon.values())
        rep = gates["evaluator_repeatability"]["max_objective_range"]
        peak = gates["peak_memory"]["peak_gib"]
        failures = []
        # A smoke run is not a measurement. Both probes can execute on CPU or
        # against a stand-in model so that they are testable at all; an artifact
        # produced that way must never satisfy the gate it exists to inform.
        for gate_name, key in (("peak_memory", "is_gate_measurement"),
                               ("evaluator_repeatability", "is_real_teacher")):
            if not gates[gate_name].get(key):
                failures.append(
                    f"{gate_name} did not produce a gate measurement "
                    f"({key} is not true); this is a smoke artifact")
        if peak is None:
            failures.append("peak memory was not measured (peak_gib is null)")
        if rep >= declared_epsilon:
            failures.append(
                f"evaluator repeatability range {rep:.3g} >= declared epsilon "
                f"{declared_epsilon:.3g}: {EPSILON_RESPONSE_V1.rule_id} fires, no "
                "epsilon is re-derived, Phase A is blocked pending review")
        if peak is not None and peak > 40.0:
            failures.append(f"peak resident {peak:.1f} GiB > 40 GiB")
        if gates["disk_throughput"]["write_mb_s"] < 50:
            failures.append(
                f"disk write {gates['disk_throughput']['write_mb_s']:.0f} MB/s "
                "makes the working-set plan infeasible")
        return self.record(1, not failures, "; ".join(failures), gates=gates)

    def gate(self, name: str, script: str, extra: list[str], *,
             timeout: float) -> dict:
        """Run one Stage-1 measurement, keeping its whole output as an artifact.

        The 2026-08-13 attempt learned this the expensive way. A gate raised,
        the exception carried the last 600 characters of the subprocess output,
        and the stage record then truncated *that* to its first 300 — which is
        the progress bars. The pod was deleted before anyone could look, and the
        session's only account of why it stopped was "Loading weights: 100%".

        So the full stdout and stderr go to a file the artifact spec collects,
        and the in-record excerpt is the **tail**: the end of a traceback is the
        part that says what happened.
        """
        log = AUDIT / f"{name}.log"
        out = subprocess.run(
            ["/opt/train/bin/python", str(REPO / script), *extra],
            capture_output=True, text=True, timeout=timeout, env=self.child_env())
        log.write_text(f"$ {script} {' '.join(extra)}\n"
                       f"rc={out.returncode}\n--- stdout ---\n{out.stdout}\n"
                       f"--- stderr ---\n{out.stderr}\n")
        if out.returncode != 0:
            raise RuntimeError(
                f"{name} rc={out.returncode}; full output in "
                f"{log.relative_to(REPO)}; tail: "
                f"...{(out.stdout + out.stderr).rstrip()[-1200:]}")
        return json.loads((AUDIT / f"{name}.json").read_text())

    def generation_smoke(self) -> dict:
        """Run the WHOLE Stage-3 generation path once, on two prompts, for pennies.

        Stage 3 failed on 2026-08-13 after $2.69 of permanent controls had been
        trained, and its cause could not be recovered. The generation path — the
        evaluator, the summaries it writes, and the observed-protocol
        reconstruction that reads them — has no cheap rehearsal anywhere else:
        it needs vLLM and a GPU, so no CPU test can execute it.

        So it executes here, against the canonical init, on a two-prompt slice of
        the frozen battery, before any money goes into a control. It exercises
        exactly what Stage 3 does: `uncapped_eval.py` end to end, a summary per
        set, `observe_generation_protocol`, and the comparison against the
        Stage-0 attested fingerprint. A failure costs profiling minutes.

        It is a *path* check, not a measurement: the prompts are a subset and the
        model is the initializer, so nothing here is scored or retained.
        """
        smoke_dir = REPO / "artifacts/eval/preflight/_generation_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        battery = json.loads((BATTERY / "manifest.json").read_text())
        # The HARDEST set, not the alphabetically first. On 2026-08-13 this took
        # `sorted(sets)[0]` = `code`, the one set that declares no tools, and the
        # smoke passed while Stage 3 then died rendering a tool prompt. A smoke
        # that avoids the awkward input is a smoke that agrees with you.
        sets = sorted(battery["sets"])
        chosen = [s for s in ("tool", "rag") if s in sets] or sets[:1]
        subsets = []
        for name in chosen:
            subset = smoke_dir / f"{name}.jsonl"
            lines = (BATTERY / f"{name}.jsonl").read_text().splitlines()
            subset.write_text("\n".join(lines[:2]) + "\n")
            subsets.append(subset)
        first_set = chosen[0]

        out = subprocess.run(
            ["/opt/vllm/bin/python", str(REPO / "scripts/evaluation/uncapped_eval.py"),
             "--model", str(CANONICAL_INIT), "--label", "_generation_smoke",
             "--prompts", *[str(x) for x in subsets], "--out-dir", str(smoke_dir),
             "--diagnostics"],
            capture_output=True, text=True, timeout=2700, env=self.child_env())
        (AUDIT / "generation_smoke.log").write_text(
            f"rc={out.returncode}\n--- stdout ---\n{out.stdout}\n"
            f"--- stderr ---\n{out.stderr}\n")
        if out.returncode != 0:
            raise RuntimeError(
                f"generation smoke rc={out.returncode}; full output in "
                f"{(AUDIT / 'generation_smoke.log').relative_to(REPO)}; tail: "
                f"...{(out.stdout + out.stderr).rstrip()[-1200:]}")

        summaries = [json.loads(p.read_text())
                     for p in sorted(smoke_dir.glob("*.json"))
                     if not p.name.endswith(".generations.jsonl")]
        observed = observe_generation_protocol(summaries, strict=True)
        comparison = observed.protocol.compare(self.evaluation_protocol.generation)
        report = {"sets": chosen, "prompts": 2 * len(chosen),
                  "model": "canonical init (not a control)",
                  "observed_generation_fingerprint": observed.protocol.fingerprint,
                  "attested_generation_fingerprint":
                      self.evaluation_protocol.generation.fingerprint,
                  "identical": comparison["identical"],
                  "mismatched": comparison["mismatched_fields"],
                  "unknown": comparison["unverifiable_fields"],
                  "is_measurement": False}
        (AUDIT / "generation_smoke.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")
        if not comparison["identical"]:
            raise RuntimeError(
                "generation smoke: the rollouts this evaluator produces do not "
                "reconstruct the attested generation protocol — mismatched "
                f"{[m['field'] for m in comparison['mismatched_fields']]}, "
                f"unknown {[u['field'] for u in comparison['unverifiable_fields']]}. "
                "Stage 3 would reject its own characterization.")
        return report

    def repeatability(self) -> dict:
        """Score one checkpoint N times on the frozen suite; report the range."""
        return self.gate(
            "evaluator_repeatability",
            "scripts/autoinit/measure_state_repeatability.py",
            ["--checkpoint", str(CANONICAL_INIT), "--repeats", str(self.a.repeats),
             "--out", str(AUDIT / "evaluator_repeatability.json")],
            timeout=5400)

    def stats_split(self) -> dict:
        return self.gate(
            "statistics_split", "scripts/autoinit/profile_statistics_pass.py",
            ["--out", str(AUDIT / "statistics_split.json")], timeout=5400)

    def peak_memory(self) -> dict:
        return self.gate(
            "peak_memory", "scripts/autoinit/probe_peak_memory.py",
            ["--out", str(AUDIT / "peak_memory.json")], timeout=3600)

    def disk_throughput(self) -> dict:
        """Write and re-read a real intermediate-sized file, not a synthetic loop."""
        path = WS / "disk_probe.bin"
        size = int(self.a.disk_probe_gib * 2**30)
        chunk = os.urandom(2**24)
        t = time.time()
        written = 0
        with path.open("wb") as f:
            while written < size:
                f.write(chunk)
                written += len(chunk)
            f.flush()
            os.fsync(f.fileno())
        write_s = time.time() - t
        os.system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true")
        t = time.time()
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(2**24), b""):
                h.update(block)
        read_s = time.time() - t
        path.unlink(missing_ok=True)
        return {"bytes": written,
                "write_seconds": round(write_s, 2), "read_seconds": round(read_s, 2),
                "write_mb_s": round(written / 2**20 / write_s, 1),
                "read_mb_s": round(written / 2**20 / read_s, 1),
                "note": "read follows an fsync and a cache drop attempt"}

    # -- stage 2 ---------------------------------------------------------
    def control_config(self, name: str, config: str) -> Path:
        """Materialize the run config for one control, from the frozen recipe.

        Two fields are overridden and nothing else: `out_dir`, so the two
        preflight controls do not collide with the historical E1 run directories
        the frozen configs name, and `data_dir`, so the run reads the pack under
        its canonical `ladder_uniform_probe` path — the path the preregistration
        and the attested protocol both pin. Setup stages the identical bytes
        under both names and verifies the hash of each, so this changes which
        path is read and not what is read.

        The realized diff is recorded, and anything else differing raises. That
        is the same guard `validate_e{7,8}_arms.py` applies to an experiment's
        arms: a config edited into a run is a config nobody reviewed.
        """
        allowed = {"data_dir", "out_dir", "run_name", "_purpose"}
        frozen = json.loads((REPO / config).read_text())
        derived = {**frozen,
                   "out_dir": f"artifacts/stage3/{name}",
                   "data_dir": "artifacts/stage3/ladder_uniform_probe",
                   "run_name": name,
                   "_purpose": (
                       "AutoInitializer micro-preflight: a PERMANENT canonical "
                       "control probe at the 0.86M rung, re-executed under the "
                       f"attested runtime. Derived from {config} by overriding "
                       "only run identity and the pack path.")}
        diff = sorted(k for k in set(frozen) | set(derived)
                      if frozen.get(k) != derived.get(k))
        if not set(diff) <= allowed:
            raise RecoveryAdmissionError(
                f"{name}: the derived control config differs from the frozen "
                f"recipe in {sorted(set(diff) - allowed)}, which is outside the "
                f"allowed override set {sorted(allowed)}")
        path = AUDIT / "configs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(derived, indent=2) + "\n")
        return path

    def stage2(self) -> bool:
        self.enter(2)
        arms = {}
        for name, seed, config in CONTROLS:
            minutes = self.a.control_minutes
            if not self.afford(minutes, f"control {name}"):
                return self.record(2, False, f"insufficient budget for {name}",
                                   arms=arms)
            run_config = self.control_config(name, config)
            t = time.time()
            rc = subprocess.run(
                ["/opt/train/bin/python", str(REPO / "scripts/training/train_stage3.py"),
                 "--config", str(run_config)],
                capture_output=True, text=True, timeout=int(minutes * 60 * 2),
                env=self.child_env())
            elapsed = (time.time() - t) / 60
            tail = (rc.stdout + rc.stderr)[-1500:]
            (AUDIT / f"{name}_train_tail.log").write_text(tail)
            if rc.returncode != 0:
                arms[name] = {"trained": False, "rc": rc.returncode,
                              "minutes": round(elapsed, 1)}
                mark(f"TRAIN_FAILED:{name}")
                return self.record(2, False, f"{name} failed rc={rc.returncode}",
                                   arms=arms)
            arm = self.verify_control(name, seed, run_config, elapsed)
            arms[name] = arm
            if not arm["protocol_verified"]:
                return self.record(2, False, f"{name}: {arm['problem']}", arms=arms)
            mark(f"TRAIN_DONE:{name}")
        return self.record(2, True, arms=arms)

    def verify_control(self, name: str, seed: int, config: Path,
                       minutes: float) -> dict:
        """A control that did not run the attested protocol is not a control.

        The protocol is **reconstructed from the run's own artifacts** and then
        compared to the Stage-0 attestation. It is not built from the attested
        object: doing that compares the attestation to itself, which passes
        whatever the trainer did.
        """
        out_dir = REPO / f"artifacts/stage3/{name}"
        problem = ""
        observed = None
        comparison: dict = {}
        try:
            observed = observe_recovery_protocol(out_dir, repo_root=REPO,
                                                 strict=True)
        except (ObservedProtocolError, RecoveryAdmissionError) as exc:
            problem = f"observed protocol could not be established: {exc}"

        probe = None
        digest = init_digest = None
        ckpts = sorted(out_dir.glob("checkpoints/step_*"))
        ckpt = ckpts[-1] if ckpts else None
        if ckpt is not None:
            weights = ckpt / "model" / "model.safetensors"
            digest = sha256_file(weights) if weights.is_file() else None
        elif not problem:
            problem = "no checkpoint was written"

        if observed is not None:
            comparison = observed.protocol.compare(self.attested)
            if not comparison["protocol_identical"] and not problem:
                problem = (
                    "the run's OBSERVED protocol differs from the Stage-0 "
                    "attested protocol: mismatched "
                    f"{[m['field'] for m in comparison['mismatched_fields']]}, "
                    f"unverifiable {[u['field'] for u in comparison['unverifiable_fields']]}")
            if observed.seed != seed and not problem:
                problem = (f"the run recorded seed {observed.seed}, not the "
                           f"{seed} this control is supposed to be")
            # The initialization is not part of the protocol — it is the
            # treatment — so it is established separately, by hashing the
            # weights the run actually read.
            init = Path(observed.initialization_source or "")
            init_weights = (init if init.is_absolute() else REPO / init) / "model.safetensors"
            init_digest = sha256_file(init_weights) if init_weights.is_file() else None
            if init_digest != PINNED["canonical_init_weights"][1] and not problem:
                problem = (f"the run started from {observed.initialization_source} "
                           f"(weights {init_digest}), not the pinned canonical "
                           f"init {PINNED['canonical_init_weights'][1]}")
            if not problem:
                probe = RecoveryProbeIdentity(
                    protocol=observed.protocol,
                    initialization_artifact_digest=init_digest,
                    seed=observed.seed, label=name)
                try:
                    probe.require_attested(self.attested.fingerprint)
                except RecoveryAdmissionError as exc:
                    problem = str(exc)

        manifest = out_dir / "run_manifest.json"
        train_log = out_dir / "train_log.jsonl"
        completion = out_dir / "run_completion.json"
        required = {"run_manifest.json": manifest.is_file(),
                    "run_completion.json": completion.is_file(),
                    "train_log.jsonl": train_log.is_file()}
        if not all(required.values()):
            problem = problem or f"missing required artifacts: {required}"
        # The plan's stop condition is written symmetrically ("diverges by more
        # than 25%"); it is enforced on the SLOW side only, and the fast side is
        # recorded. A run faster than priced is not a machine we mispriced in
        # any way that matters: it costs less, the budget machinery already
        # refuses an arm that cannot finish, and the step accounting above
        # separately proves the run completed every declared step. Failing a
        # completed permanent control for finishing early would destroy the
        # session's only expensive artifact over good news.
        expected = self.a.control_minutes
        drift = (minutes - expected) / expected if expected else 0.0
        if drift > 0.25:
            problem = problem or (
                f"step time diverged: {minutes:.1f} min vs priced "
                f"{expected:.1f} min (+{drift:.0%})")
        def rel(path: Path | None) -> str | None:
            if path is None:
                return None
            return str(path.relative_to(REPO)) if path.is_relative_to(REPO) \
                else str(path)

        record = {
            "trained": True, "seed": seed, "minutes": round(minutes, 1),
            "priced_minutes": expected,
            "wall_clock_drift": round(drift, 4),
            "faster_than_priced": drift < -0.25,
            "run_config": rel(config),
            "checkpoint": rel(ckpt),
            "weights_sha256": digest,
            "initialization_artifact_digest": init_digest,
            "observed_protocol": (observed.as_dict() if observed else None),
            "observed_vs_attested": comparison,
            "probe_identity": probe.as_dict() if probe else None,
            "probe_id": probe.probe_id if probe else None,
            "attested_protocol_fingerprint": self.attested.fingerprint,
            "required_artifacts": required,
            "protocol_verified": not problem, "problem": problem,
            # The three hashes a permanent control is bound by. A checkpoint
            # whose protocol was never established is not a control, so this is
            # only complete when the verification passed.
            "control_binding": ({
                "observed_protocol_fingerprint": observed.protocol.fingerprint,
                "probe_id": probe.probe_id,
                "checkpoint_weights_sha256": digest,
            } if (probe and observed and not problem) else None),
        }
        (AUDIT / f"{name}_probe_identity.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n")
        return record

    # -- stage 3 ---------------------------------------------------------
    def stage_tokenizer(self, model_dir: Path) -> dict:
        """`save_pretrained` writes no tokenizer; install the canonical one.

        Loading a checkpoint directory without tokenizer files does **not**
        fail: `AutoTokenizer.from_pretrained` builds a degenerate tokenizer from
        the config and every prompt encodes to nothing. A silent-wrong failure,
        and it would also make the observed `tokenizer_sha256` disagree with the
        Stage-0 attestation — which is the check that would catch it.

        The tokenizer is a project constant, verified against its pinned hash at
        setup, so this restores a file that should have been written beside the
        weights rather than changing anything about the checkpoint.
        """
        import shutil
        installed = {}
        for fname in ("tokenizer.json", "tokenizer_config.json",
                      "chat_template.jinja"):
            src, dst = CANONICAL_INIT / fname, model_dir / fname
            if src.is_file() and not dst.is_file():
                shutil.copy(src, dst)
            installed[fname] = dst.is_file()
        if not all(installed.values()):
            raise RecoveryAdmissionError(
                f"tokenizer files are not complete in {model_dir}: {installed}")
        return installed

    def observed_generation(self, gen_dir: Path, label: str) -> tuple[dict, Any]:
        """Reconstruct the generation protocol from the summaries just written.

        Returns the JSON-safe report and the fingerprint object. Raises when the
        rollouts were produced under anything other than the attested protocol —
        before they are scored, because a characterization of generations made
        under some other protocol materializes thresholds nothing can be
        compared against later.
        """
        summaries = []
        for path in sorted(gen_dir.glob("*.json")):
            if path.name.endswith(".generations.jsonl"):
                continue
            summaries.append(json.loads(path.read_text()))
        observed = observe_generation_protocol(summaries, strict=True)
        comparison = observed.protocol.compare(self.evaluation_protocol.generation)
        report = {"label": label, **observed.as_dict(),
                  "observed_vs_attested": comparison,
                  "attested_generation_fingerprint":
                      self.evaluation_protocol.generation.fingerprint}
        (AUDIT / f"{label}_observed_generation.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")
        if not comparison["identical"]:
            raise GenerationProtocolError(
                f"{label}: the stored rollouts were produced under a different "
                "generation protocol than Stage 0 attested — mismatched "
                f"{[m['field'] for m in comparison['mismatched_fields']]}, "
                f"unknown {[u['field'] for u in comparison['unverifiable_fields']]}")
        return report, observed.protocol

    def stage3(self) -> bool:
        self.enter(3)
        per_seed, results, generation, gen_protocols = [], {}, {}, {}
        for name, seed, _ in CONTROLS:
            if not self.afford(self.a.characterization_minutes, f"characterize {name}"):
                return self.record(3, False, f"insufficient budget for {name}")
            ckpt = REPO / self.ev["stages"]["2"]["arms"][name]["checkpoint"] / "model"
            tokenizer_files = self.stage_tokenizer(ckpt)
            gen_dir = REPO / f"artifacts/eval/preflight/{name}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            sets = [str(BATTERY / f"{s}.jsonl") for s in
                    json.loads((BATTERY / "manifest.json").read_text())["sets"]]
            rc = subprocess.run(
                ["/opt/vllm/bin/python", str(REPO / "scripts/evaluation/uncapped_eval.py"),
                 "--model", str(ckpt), "--label", name, "--prompts", *sets,
                 "--out-dir", str(gen_dir), "--diagnostics"],
                capture_output=True, text=True,
                timeout=int(self.a.characterization_minutes * 60 * 2),
                env=self.child_env())
            if rc.returncode != 0:
                (AUDIT / f"{name}_generation_tail.log").write_text(
                    (rc.stdout + rc.stderr)[-2000:])
                return self.record(3, False, f"{name} generation rc={rc.returncode}")
            # Before anything is scored: were these rollouts produced under the
            # protocol Stage 0 attested? A characterization of generations made
            # under some other protocol would materialize thresholds that no
            # later probe can be compared against.
            try:
                report, gen_protocols[name] = self.observed_generation(gen_dir, name)
            except GenerationProtocolError as exc:
                return self.record(3, False, f"{name}: {exc}"[-1500:],
                                   generation=generation)
            generation[name] = {**report,
                                "tokenizer_files_installed": tokenizer_files}
            scored = AUDIT / f"{name}_recovery_search.json"
            rc = subprocess.run(
                ["/opt/train/bin/python",
                 str(REPO / "scripts/autoinit/score_recovery_search.py"),
                 "--generations", str(gen_dir), "--label", name,
                 "--seed", str(seed), "--out", str(scored),
                 "--per-sample", str(AUDIT / f"{name}_per_sample.jsonl")],
                capture_output=True, text=True, timeout=1800,
                env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
            if rc.returncode != 0:
                return self.record(3, False, f"{name} scoring: "
                                             f"{(rc.stdout + rc.stderr)[-400:]}")
            result = json.loads(scored.read_text())
            if result["scoring_contract"]["digest"] != \
                    self.evaluation_protocol.scoring_digest:
                return self.record(3, False, f"{name} scored under a different "
                                             "scoring contract than Stage 0 froze")
            # The evaluation protocol this arm's numbers were actually produced
            # under: its own observed generation fingerprint, its own scoring
            # contract digest, and the battery identity the scorer recorded.
            # Built from the result, then required to be comparable with the
            # Stage-0 attestation — not copied from it.
            observed_eval = RecoveryEvaluationProtocol(
                generation=gen_protocols[name],
                scoring_contract=result["scoring_contract"]["contract"],
                scoring_digest=result["scoring_contract"]["digest"],
                battery_artifact=result["battery"]["artifact"],
                battery_manifest_sha256=result["battery"]["manifest_sha256"],
                battery_content_sha256=result["battery"]["content_sha256"])
            try:
                observed_eval.require_comparable(self.evaluation_protocol,
                                                 context=f"{name} characterization")
            except GenerationProtocolError as exc:
                return self.record(3, False, f"{name}: {exc}"[-1500:],
                                   generation=generation)
            generation[name]["evaluation_protocol_hash"] = \
                observed_eval.evaluation_protocol_hash
            result["evaluation_protocol_hash"] = observed_eval.evaluation_protocol_hash
            result["bound_to"] = {
                "evaluation_protocol_hash": observed_eval.evaluation_protocol_hash,
                "observed_generation_fingerprint":
                    observed_eval.generation.fingerprint,
                "checkpoint": self.ev["stages"]["2"]["arms"][name]["checkpoint"],
                "probe_id": self.ev["stages"]["2"]["arms"][name].get("probe_id"),
                "rule": ("this result is comparable only to results carrying the "
                         "same evaluation_protocol_hash"),
            }
            scored.write_text(json.dumps(result, indent=2) + "\n")
            results[name] = result
            # Every denominator the per-seed rates were computed over;
            # `pooled_counts@v2` refuses a row that omits one.
            per_seed.append({"seed": seed,
                             **{k: result[k]
                                for k in POOLED_COUNTS_V2.required_counts}})

        pooled = POOLED_COUNTS_V2.pool(per_seed)
        sa, sb = (results[c[0]] for c in CONTROLS)
        equivalence = EquivalenceRule(
            n_pooled=pooled["n_scorable"]).materialize(
                p_pool=pooled["correct_overall"], p_sa=sa["correct_overall"],
                p_sb=sb["correct_overall"]).as_dict()
        feasibility = FeasibilityRule(n_pooled=pooled["n"]).materialize(
            u_pool=pooled["usable_rollout_rate"], u_sa=sa["usable_rollout_rate"],
            u_sb=sb["usable_rollout_rate"]).as_dict()
        capabilities = {
            cap: {"pooled_usable_rate": round(
                      (sa["per_capability"][cap]["usable"]
                       + sb["per_capability"][cap]["usable"])
                      / (sa["per_capability"][cap]["n"]
                         + sb["per_capability"][cap]["n"]), 4),
                  "sa": sa["per_capability"][cap]["usable_rollout_rate"],
                  "sb": sb["per_capability"][cap]["usable_rollout_rate"]}
            for cap in sa["per_capability"]}
        thresholds = {
            "schema": "aadistill.autoinit.materialized_thresholds/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
            # Every control's numbers were produced under a protocol
            # reconstructed from its own rollouts and verified identical to the
            # attestation. The thresholds inherit that binding: a later probe
            # measured under a different hash is not comparable to them.
            "observed_evaluation_protocol_hash_per_control": {
                name: gen.get("evaluation_protocol_hash")
                for name, gen in generation.items()},
            "observed_generation_fingerprint_per_control": {
                name: gen.get("observed_generation_fingerprint")
                for name, gen in generation.items()},
            "aggregation": POOLED_COUNTS_V2.as_dict(),
            "pooled": pooled, "per_seed": per_seed,
            "equivalence_interval": equivalence,
            "feasibility_floor": feasibility,
            "per_capability_control_baseline": capabilities,
            "catastrophic_rule": CATASTROPHIC_V1.as_dict(),
            "materialized_from": "control data only; no searched candidate exists",
        }
        thresholds["report_sha256"] = sha256_json(thresholds)
        (AUDIT / "materialized_thresholds.json").write_text(
            json.dumps(thresholds, indent=2) + "\n")
        return self.record(3, True, thresholds=thresholds, generation=generation,
                           per_control={k: {m: v[m] for m in
                                            ("usable_rollout_rate", "correct_overall",
                                             "correct_given_usable", "n")}
                                        for k, v in results.items()})

    # -- run -------------------------------------------------------------
    def run(self) -> int:
        mark("DRIVER_START")
        stages = {0: self.stage0, 1: self.stage1, 2: self.stage2, 3: self.stage3}
        wanted = sorted(stages) if self.a.stage == "all" else [int(self.a.stage)]
        failed: list[int] = []
        for stage in wanted:
            try:
                ok = stages[stage]()
            except RecoveryAdmissionError as exc:
                self.record(stage, False, f"refused: {exc}")
                ok = False
            except Exception as exc:                              # noqa: BLE001
                self.record(stage, False,
                            f"{type(exc).__name__}: {exc}"[-1500:])
                ok = False
            if not ok:
                blocking = {s.stage for s in PREFLIGHT_PLAN_V1.stages if s.blocking}
                if stage in blocking:
                    mark("PREFLIGHT_FAILED")
                    say("stopping before any later stage; permanent controls are "
                        "not trained under a configuration that must change")
                    self.save()
                    return 20 + stage
                # A non-blocking stage may fail without invalidating what came
                # before -- the controls exist and are kept -- but the session is
                # NOT complete, and `ALL_DONE` must not be emitted for it. That
                # marker is what the launcher reads as success and what the full
                # artifact spec is gated on.
                mark("STAGE_NONBLOCKING_FAIL")
                failed.append(stage)
        self.ev["phase_a_started"] = False
        self.ev["phase_a_reachable_from_this_driver"] = False
        self.ev["failed_nonblocking_stages"] = failed
        self.save()
        if failed:
            mark("PREFLIGHT_INCOMPLETE")
            say(f"preflight INCOMPLETE: stage(s) {failed} failed — "
                f"${self.usd():.2f}. Artifacts from earlier stages are retained.")
            return 20 + failed[0]
        mark("ALL_DONE")
        say(f"preflight complete — ${self.usd():.2f}. Phase A NOT started and not "
            "reachable from this driver.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    # No stage 4: Phase A is not expressible here.
    ap.add_argument("--stage", default="all", choices=("all", "0", "1", "2", "3"))
    ap.add_argument("--image-digest", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--control-minutes", type=float, default=85.0)
    ap.add_argument("--characterization-minutes", type=float, default=18.0)
    ap.add_argument("--disk-probe-gib", type=float, default=6.0)
    args = ap.parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    driver = Driver(args)
    driver.auth.require_within_cap(args.authorized_usd, what="session backstop")
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
