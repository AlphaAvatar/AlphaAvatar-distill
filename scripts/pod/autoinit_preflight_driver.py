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
    RecoveryEvaluationProtocol,
    declared_generation_protocol,
    generation_source_digest,
)
from aadistill.autoinit.ranking import EPSILON_RESPONSE_V1, PARETO_V1  # noqa: E402
from aadistill.autoinit.recovery import (  # noqa: E402
    CATASTROPHIC_V1,
    POOLED_COUNTS_V1,
    PREFLIGHT_PLAN_V1,
    SEED_SA,
    SEED_SB,
    EquivalenceRule,
    FeasibilityRule,
    RecoveryAdmissionError,
    RecoveryProbeIdentity,
    RuntimeEnvironmentFingerprint,
    recovery_scoring_contract,
    trainer_source_digest,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

WS = Path("/workspace")
STATUS = WS / "autoinit_preflight.status"
AUDIT = REPO / "artifacts/audit/autoinit_preflight"
BATTERY = REPO / "artifacts/stage3/recovery_search_v1"
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

        # Generation protocol: everything except the engine's own scheduler
        # defaults can be read without a GPU; those need a live engine, so the
        # engine is booted here rather than discovered after two control runs.
        gen = declared_generation_protocol().materialized(
            generation_source_digest=generation_source_digest(REPO)["digest"],
            transformers_version=runtime.transformers_version,
            torch_version=runtime.torch_version,
            runtime_digest=runtime.digest,
            degeneration_source_digest=sha256_file(
                REPO / "src/aadistill/evaluation/degeneration.py"))
        try:
            observed = self.observe_engine()
        except Exception as exc:                                  # noqa: BLE001
            return self.record(0, False, f"engine probe failed: {exc!r}"[:300],
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
        """Boot vLLM once on the canonical init and read what it actually uses."""
        out = subprocess.run(
            ["/opt/vllm/bin/python", str(REPO / "scripts/pod/autoinit_engine_probe.py"),
             "--model", str(CANONICAL_INIT), "--out", str(AUDIT / "engine_probe.json")],
            capture_output=True, text=True, timeout=1800,
            env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
        if out.returncode != 0:
            raise RuntimeError(f"engine probe rc={out.returncode}: "
                               f"{(out.stdout + out.stderr)[-600:]}")
        probe = json.loads((AUDIT / "engine_probe.json").read_text())
        return {"vllm_version": probe["vllm_version"],
                "max_num_seqs": probe["max_num_seqs"],
                "max_num_batched_tokens": probe["max_num_batched_tokens"],
                "enforce_eager": probe["enforce_eager"],
                "tokenizer_sha256": probe["tokenizer_sha256"],
                "chat_template_sha256": probe["chat_template_sha256"],
                "resolved_context": probe["resolved_context"],
                "context_source": probe["context_source"],
                "stop_token_ids": tuple(probe["stop_token_ids"])}

    # -- stage 1 ---------------------------------------------------------
    def stage1(self) -> bool:
        self.enter(1)
        if not self.afford(25, "stage 1 machine gates"):
            return self.record(1, False, "insufficient budget for the gates")
        gates = {}
        try:
            gates["engine_probe"] = json.loads(
                (AUDIT / "engine_probe.json").read_text())
            gates["evaluator_repeatability"] = self.repeatability()
            gates["activation_statistics"] = self.stats_split()
            gates["peak_memory"] = self.peak_memory()
            gates["disk_throughput"] = self.disk_throughput()
        except Exception as exc:                                  # noqa: BLE001
            return self.record(1, False, f"gate raised: {exc!r}"[:300], gates=gates)

        # The declared epsilon is the smallest across the beam's objectives: a
        # measured range at or above ANY objective's tolerance fires the gate.
        declared_epsilon = min(PARETO_V1.epsilon.values())
        rep = gates["evaluator_repeatability"]["max_objective_range"]
        peak = gates["peak_memory"]["peak_gib"]
        failures = []
        if rep >= declared_epsilon:
            failures.append(
                f"evaluator repeatability range {rep:.3g} >= declared epsilon "
                f"{declared_epsilon:.3g}: {EPSILON_RESPONSE_V1.rule_id} fires, no "
                "epsilon is re-derived, Phase A is blocked pending review")
        if peak > 40.0:
            failures.append(f"peak resident {peak:.1f} GiB > 40 GiB")
        if gates["disk_throughput"]["write_mb_s"] < 50:
            failures.append(
                f"disk write {gates['disk_throughput']['write_mb_s']:.0f} MB/s "
                "makes the working-set plan infeasible")
        return self.record(1, not failures, "; ".join(failures), gates=gates)

    def repeatability(self) -> dict:
        """Score one checkpoint N times on the frozen suite; report the range."""
        from aadistill.autoinit.metrics import StateEvaluation  # noqa: F401
        out = subprocess.run(
            ["/opt/train/bin/python",
             str(REPO / "scripts/autoinit/measure_state_repeatability.py"),
             "--checkpoint", str(CANONICAL_INIT), "--repeats", str(self.a.repeats),
             "--out", str(AUDIT / "evaluator_repeatability.json")],
            capture_output=True, text=True, timeout=5400,
            env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
        if out.returncode != 0:
            raise RuntimeError(f"repeatability rc={out.returncode}: "
                               f"{(out.stdout + out.stderr)[-600:]}")
        return json.loads((AUDIT / "evaluator_repeatability.json").read_text())

    def stats_split(self) -> dict:
        out = subprocess.run(
            ["/opt/train/bin/python",
             str(REPO / "scripts/autoinit/profile_statistics_pass.py"),
             "--out", str(AUDIT / "statistics_split.json")],
            capture_output=True, text=True, timeout=5400,
            env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
        if out.returncode != 0:
            raise RuntimeError(f"statistics profile rc={out.returncode}: "
                               f"{(out.stdout + out.stderr)[-600:]}")
        return json.loads((AUDIT / "statistics_split.json").read_text())

    def peak_memory(self) -> dict:
        out = subprocess.run(
            ["/opt/train/bin/python",
             str(REPO / "scripts/autoinit/probe_peak_memory.py"),
             "--out", str(AUDIT / "peak_memory.json")],
            capture_output=True, text=True, timeout=3600,
            env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
        if out.returncode != 0:
            raise RuntimeError(f"peak memory rc={out.returncode}: "
                               f"{(out.stdout + out.stderr)[-600:]}")
        return json.loads((AUDIT / "peak_memory.json").read_text())

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
    def stage2(self) -> bool:
        self.enter(2)
        arms = {}
        for name, seed, config in CONTROLS:
            minutes = self.a.control_minutes
            if not self.afford(minutes, f"control {name}"):
                return self.record(2, False, f"insufficient budget for {name}",
                                   arms=arms)
            t = time.time()
            rc = subprocess.run(
                ["/opt/train/bin/python", str(REPO / "scripts/training/train_stage3.py"),
                 "--config", str(REPO / config), "--out-dir",
                 str(REPO / f"artifacts/stage3/{name}")],
                capture_output=True, text=True, timeout=int(minutes * 60 * 2),
                env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
            elapsed = (time.time() - t) / 60
            tail = (rc.stdout + rc.stderr)[-1500:]
            (AUDIT / f"{name}_train_tail.log").write_text(tail)
            if rc.returncode != 0:
                arms[name] = {"trained": False, "rc": rc.returncode,
                              "minutes": round(elapsed, 1)}
                mark(f"TRAIN_FAILED:{name}")
                return self.record(2, False, f"{name} failed rc={rc.returncode}",
                                   arms=arms)
            arm = self.verify_control(name, seed, config, elapsed)
            arms[name] = arm
            if not arm["protocol_verified"]:
                return self.record(2, False, f"{name}: {arm['problem']}", arms=arms)
            mark(f"TRAIN_DONE:{name}")
        return self.record(2, True, arms=arms)

    def verify_control(self, name: str, seed: int, config: str,
                       minutes: float) -> dict:
        """A control that did not run the attested protocol is not a control."""
        out_dir = REPO / f"artifacts/stage3/{name}"
        ckpt = sorted(out_dir.glob("checkpoints/step_*"))[-1]
        weights = ckpt / "model" / "model.safetensors"
        digest = sha256_file(weights)
        probe = RecoveryProbeIdentity(
            protocol=self.attested,
            initialization_artifact_digest=PINNED["canonical_init_weights"][1],
            seed=seed, label=name)
        problem = ""
        try:
            probe.require_attested(self.attested.fingerprint)
        except RecoveryAdmissionError as exc:
            problem = str(exc)
        manifest = out_dir / "run_manifest.json"
        train_log = out_dir / "train_log.jsonl"
        required = {"run_manifest.json": manifest.is_file(),
                    "train_log.jsonl": train_log.is_file()}
        if not all(required.values()):
            problem = problem or f"missing required artifacts: {required}"
        expected = self.a.control_minutes
        drift = abs(minutes - expected) / expected if expected else 0.0
        if drift > 0.25:
            problem = problem or (
                f"step time diverged: {minutes:.1f} min vs priced "
                f"{expected:.1f} min ({drift:.0%})")
        record = {
            "trained": True, "seed": seed, "minutes": round(minutes, 1),
            "checkpoint": str(ckpt.relative_to(REPO)),
            "weights_sha256": digest,
            "probe_identity": probe.as_dict(),
            "probe_id": probe.probe_id,
            "attested_protocol_fingerprint": self.attested.fingerprint,
            "required_artifacts": required,
            "protocol_verified": not problem, "problem": problem,
        }
        (AUDIT / f"{name}_probe_identity.json").write_text(
            json.dumps(record, indent=2) + "\n")
        return record

    # -- stage 3 ---------------------------------------------------------
    def stage3(self) -> bool:
        self.enter(3)
        per_seed, results = [], {}
        for name, seed, _ in CONTROLS:
            if not self.afford(self.a.characterization_minutes, f"characterize {name}"):
                return self.record(3, False, f"insufficient budget for {name}")
            ckpt = REPO / self.ev["stages"]["2"]["arms"][name]["checkpoint"] / "model"
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
                env={**os.environ, "PYTHONPATH": f"{REPO}/src"})
            if rc.returncode != 0:
                (AUDIT / f"{name}_generation_tail.log").write_text(
                    (rc.stdout + rc.stderr)[-2000:])
                return self.record(3, False, f"{name} generation rc={rc.returncode}")
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
            results[name] = result
            per_seed.append({"seed": seed, "n": result["n"],
                             "usable": result["usable"], "correct": result["correct"]})

        pooled = POOLED_COUNTS_V1.pool(per_seed)
        sa, sb = (results[c[0]] for c in CONTROLS)
        equivalence = EquivalenceRule(
            n_pooled=sa["n_scorable"] + sb["n_scorable"]).materialize(
                p_pool=pooled["correct_overall"], p_sa=sa["correct_overall"],
                p_sb=sb["correct_overall"]).as_dict()
        feasibility = FeasibilityRule(n_pooled=sa["n"] + sb["n"]).materialize(
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
            "aggregation": POOLED_COUNTS_V1.as_dict(),
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
        return self.record(3, True, thresholds=thresholds,
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
                self.record(stage, False, f"{type(exc).__name__}: {exc}"[:400])
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
