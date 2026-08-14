#!/usr/bin/env python3
"""Pod-side driver for the characterization continuation. Four stages, then stop.

    /opt/train/bin/python scripts/pod/autoinit_continuation_driver.py \
        --image-digest <digest> --rate 0.99 --spent-usd 0.15 \
        --soft-stop-usd 1.50 --authorized-usd 1.75

Characterizes two permanent controls this session did **not** train:

    0  import the controls           strict, CPU, fails closed
    1  current evaluation attestation  runtime + engine + recovery_search_v2
    2  real v2 tool and RAG smoke      the path that broke the preflight
    3  characterize sa and sb          the only paid measurement

Two semantics are load-bearing and are implemented here rather than described:

**Nothing is trained.** There is no training stage and no `--stage` that names
one. The controls arrive as a local artifact; how they got there is the
transport layer's business and not this driver's, so the same strict import gate
runs whether they came from the relay, over scp, or were already on the disk.

**Characterization is non-blocking for CLEANUP ONLY.** A scoring failure must
not prevent evidence collection or teardown — but it must not be able to turn
into a successful continuation either. Stage 3 failing emits
`CONTINUATION_INCOMPLETE`, never `ALL_DONE`, and the recorded outcome stays
failed no matter how cleanly the artifacts came home.
"""

from __future__ import annotations

import argparse
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
from aadistill.autoinit.continuation import (  # noqa: E402
    CONTINUATION_PLAN_V1, CONTINUATION_SCOPE, ControlImportError,
    continuation_manifest, import_permanent_control,
)
from aadistill.autoinit.generation import (  # noqa: E402
    GenerationProtocolError, RecoveryEvaluationProtocol,
    declared_generation_protocol, generation_source_digest,
    observe_generation_protocol,
)
from aadistill.autoinit.recovery import (  # noqa: E402
    CATASTROPHIC_V1, POOLED_COUNTS_V1, EquivalenceRule, FeasibilityRule,
    RecoveryAdmissionError, RuntimeEnvironmentFingerprint,
    recovery_scoring_contract,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

WS = Path("/workspace")
STATUS = WS / "autoinit_continuation.status"
AUDIT = REPO / "artifacts/audit/autoinit_continuation"
BATTERY = REPO / "artifacts/stage3/recovery_search_v2"
CANONICAL_INIT = REPO / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
#: Where the transport layer materializes the controls. The driver does not care
#: how they arrived; it only cares that the import gate accepts them.
CONTROL_ROOT = REPO / "artifacts/controls"
CONTROLS = (("preflight_ctl_r0860k_sa", 20260726),
            ("preflight_ctl_r0860k_sb", 20260801))
RECORDS = REPO / "logs/autoinit_permanent_controls"
BATTERY_CONTENT = "a1b22778b00d95b6aba358c14a5af5b559fd807bb371c92131eacca59479f323"
SMOKE_SETS = ("tool", "rag")


def mark(name: str) -> None:
    line = f"{datetime.now(timezone.utc):%FT%TZ} MARKER:{name}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def say(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


class ContinuationDriver:
    def __init__(self, a):
        self.a = a
        self.t0 = time.time()
        self.results: dict[int, dict] = {}
        self.imported: dict = {}
        self.evaluation_protocol = None
        self.ev: dict = {
            "schema": "aadistill.autoinit.continuation_evidence/v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "continuation": continuation_manifest(),
            "scope": CONTINUATION_SCOPE.as_dict(),
            "trains_anything": False,
            "phase_a_started": False,
            "phase_a_reachable_from_this_driver": False,
            "stages": {}}
        AUDIT.mkdir(parents=True, exist_ok=True)
        self.auth = SpendAuthorization.load(
            REPO / "logs/autoinit_continuation_authorization.json")
        self.auth.require_plan(CONTINUATION_PLAN_V1.plan_hash)
        self.ev["authorization"] = self.auth.as_dict()

    # -- budget ----------------------------------------------------------
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
        (AUDIT / "continuation_evidence.json").write_text(
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
        CONTINUATION_PLAN_V1.advance_to(stage, self.results)
        mark(f"STAGE_START:{stage}")

    def gate(self, name: str, argv: list[str], *, timeout: float,
             python: str = "/opt/train/bin/python") -> subprocess.CompletedProcess:
        """Run a child, keep its whole output, and keep the TAIL of a failure."""
        out = subprocess.run([python, *argv], capture_output=True, text=True,
                             timeout=timeout, env=self.child_env())
        (AUDIT / f"{name}.log").write_text(
            f"$ {' '.join(argv)}\nrc={out.returncode}\n--- stdout ---\n"
            f"{out.stdout}\n--- stderr ---\n{out.stderr}\n")
        return out

    # -- stage 0: import -------------------------------------------------
    def stage0(self) -> bool:
        self.enter(0)
        imported, problem = {}, ""
        for name, seed in CONTROLS:
            try:
                control = import_permanent_control(
                    name,
                    record_path=RECORDS / f"{name}_probe_identity.json",
                    checkpoint_dir=CONTROL_ROOT / name / "model",
                    run_evidence_dir=CONTROL_ROOT / name,
                    repo_root=REPO, strict=True)
            except (ControlImportError, RecoveryAdmissionError) as exc:
                problem = f"{name}: {exc}"[-1500:]
                break
            if control.seed != seed:
                problem = (f"{name}: imported seed {control.seed} is not the "
                           f"{seed} this control is supposed to be")
                break
            imported[name] = control
            say(f"imported {name}: weights {control.weights_sha256[:16]}… "
                f"protocol {control.observed_protocol_fingerprint[:16]}… "
                f"reconstructed={control.reconstructed_from_run_evidence}")
        if problem:
            return self.record(0, False, problem,
                               imported={k: v.as_dict() for k, v in imported.items()})
        fingerprints = {c.observed_protocol_fingerprint for c in imported.values()}
        if len(fingerprints) != 1:
            return self.record(0, False,
                               "the imported controls do not share one protocol "
                               f"fingerprint: {sorted(fingerprints)}")
        self.imported = imported
        report = {"controls": {k: v.as_dict() for k, v in imported.items()},
                  "shared_protocol_fingerprint": fingerprints.pop(),
                  "seeds": sorted(c.seed for c in imported.values()),
                  "trained_this_session": False}
        (AUDIT / "imported_controls.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")
        return self.record(0, True, **{"import": report})

    # -- stage 1: current evaluation attestation --------------------------
    def stage1(self) -> bool:
        self.enter(1)
        runtime = RuntimeEnvironmentFingerprint.observe(
            image_digest=self.a.image_digest)
        try:
            runtime.require_pinned()
        except RecoveryAdmissionError as exc:
            return self.record(1, False, f"runtime not pinned: {exc}")

        battery_manifest = json.loads((BATTERY / "manifest.json").read_text())
        if battery_manifest.get("content_sha256") != BATTERY_CONTENT:
            return self.record(1, False, "the battery does not verify")

        gen = declared_generation_protocol().materialized(
            generation_source_digest=generation_source_digest(REPO)["digest"],
            degeneration_source_digest=sha256_file(
                REPO / "src/aadistill/evaluation/degeneration.py"))
        probe = self.gate("engine_probe",
                          [str(REPO / "scripts/pod/autoinit_engine_probe.py"),
                           "--model", str(CANONICAL_INIT),
                           "--out", str(AUDIT / "engine_probe.json"),
                           "--image-digest", self.a.image_digest],
                          timeout=1800, python="/opt/vllm/bin/python")
        if probe.returncode != 0:
            return self.record(1, False, f"engine probe rc={probe.returncode}; "
                               f"tail: ...{(probe.stdout + probe.stderr)[-1200:]}")
        observed = json.loads((AUDIT / "engine_probe.json").read_text())
        try:
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
            gen.require_materialized(context="continuation stage 1")
        except (GenerationProtocolError, Exception) as exc:      # noqa: BLE001
            return self.record(1, False, f"generation protocol: {exc}"[-1500:])

        scoring = recovery_scoring_contract(REPO)
        self.evaluation_protocol = RecoveryEvaluationProtocol(
            generation=gen, scoring_contract=scoring["contract"],
            scoring_digest=scoring["digest"],
            battery_artifact=battery_manifest["artifact"],
            battery_manifest_sha256=battery_manifest["manifest_sha256"],
            battery_content_sha256=battery_manifest["content_sha256"])
        frozen = {
            "schema": "aadistill.autoinit.continuation_attestation/v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "continuation_plan_hash": CONTINUATION_PLAN_V1.plan_hash,
            "runtime": runtime.as_dict(),
            "evaluation_protocol": self.evaluation_protocol.as_dict(),
            "evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
            "battery": {"artifact": battery_manifest["artifact"],
                        "manifest_sha256": battery_manifest["manifest_sha256"],
                        "content_sha256": battery_manifest["content_sha256"],
                        "tools_materialization_sha256":
                            battery_manifest.get("tools_materialization_sha256")},
            "note": ("the controls' RECOVERY identity is imported and is not "
                     "re-attested here; this attests the EVALUATION identity, "
                     "which has moved since they were trained"),
        }
        frozen["report_sha256"] = sha256_json(frozen)
        (AUDIT / "attested_evaluation_protocol.json").write_text(
            json.dumps(frozen, indent=2) + "\n")
        say(f"evaluation protocol "
            f"{self.evaluation_protocol.evaluation_protocol_hash[:16]}…")
        return self.record(
            1, True,
            evaluation_protocol_hash=self.evaluation_protocol.evaluation_protocol_hash,
            generation_fingerprint=gen.fingerprint, runtime_digest=runtime.digest)

    # -- stage 2: the v2 tool + RAG smoke ---------------------------------
    def stage2(self) -> bool:
        self.enter(2)
        if not self.afford(10, "v2 generation smoke"):
            return self.record(2, False, "insufficient budget for the smoke")
        smoke_dir = REPO / "artifacts/eval/continuation/_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        sets = json.loads((BATTERY / "manifest.json").read_text())["sets"]
        chosen = [s for s in SMOKE_SETS if s in sets]
        if "tool" not in chosen:
            return self.record(2, False, "the battery has no tool set; a smoke "
                                         "that renders no tool prompt is the "
                                         "failure this stage exists for")
        subsets = []
        for name in chosen:
            path = smoke_dir / f"{name}.jsonl"
            path.write_text("\n".join(
                (BATTERY / f"{name}.jsonl").read_text().splitlines()[:2]) + "\n")
            subsets.append(str(path))
        out = self.gate("generation_smoke",
                        [str(REPO / "scripts/evaluation/uncapped_eval.py"),
                         "--model", str(CANONICAL_INIT), "--label", "_smoke",
                         "--prompts", *subsets, "--out-dir", str(smoke_dir),
                         "--diagnostics"],
                        timeout=2700, python="/opt/vllm/bin/python")
        if out.returncode != 0:
            return self.record(2, False, f"smoke generation rc={out.returncode}; "
                               f"tail: ...{(out.stdout + out.stderr)[-1200:]}")
        try:
            report = self.compare_generation(smoke_dir, "_smoke")
        except GenerationProtocolError as exc:
            return self.record(2, False, f"smoke: {exc}"[-1500:])
        report["sets"] = chosen
        report["covers_tool_set"] = True
        (AUDIT / "generation_smoke.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n")
        return self.record(2, True, smoke=report)

    def compare_generation(self, gen_dir: Path, label: str) -> dict:
        """Reconstruct the generation protocol from summaries and compare it."""
        summaries = [json.loads(p.read_text()) for p in sorted(gen_dir.glob("*.json"))
                     if not p.name.endswith(".generations.jsonl")]
        observed = observe_generation_protocol(summaries, strict=True)
        comparison = observed.protocol.compare(self.evaluation_protocol.generation)
        report = {"label": label, **observed.as_dict(),
                  "observed_vs_attested": comparison,
                  "attested_generation_fingerprint":
                      self.evaluation_protocol.generation.fingerprint}
        if not comparison["identical"]:
            raise GenerationProtocolError(
                f"{label}: the rollouts were produced under a different "
                "generation protocol than this session attested — mismatched "
                f"{[m['field'] for m in comparison['mismatched_fields']]}, "
                f"unknown {[u['field'] for u in comparison['unverifiable_fields']]}")
        return report

    # -- stage 3: characterize --------------------------------------------
    def stage3(self) -> bool:
        self.enter(3)
        per_seed, results, generation = [], {}, {}
        sets = json.loads((BATTERY / "manifest.json").read_text())["sets"]
        for name, _ in CONTROLS:
            control = self.imported[name]
            if not self.afford(self.a.characterization_minutes, f"characterize {name}"):
                return self.record(3, False, f"insufficient budget for {name}",
                                   characterized=list(results))
            gen_dir = REPO / f"artifacts/eval/continuation/{name}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            t = time.time()
            out = self.gate(f"{name}_generation",
                            [str(REPO / "scripts/evaluation/uncapped_eval.py"),
                             "--model", str(control.checkpoint_dir),
                             "--label", name,
                             "--prompts", *[str(BATTERY / f"{s}.jsonl") for s in sets],
                             "--out-dir", str(gen_dir), "--diagnostics"],
                            timeout=int(self.a.characterization_minutes * 60 * 2),
                            python="/opt/vllm/bin/python")
            minutes = (time.time() - t) / 60
            if out.returncode != 0:
                return self.record(3, False,
                                   f"{name} generation rc={out.returncode}; tail: "
                                   f"...{(out.stdout + out.stderr)[-1200:]}",
                                   characterized=list(results))
            try:
                generation[name] = {**self.compare_generation(gen_dir, name),
                                    "minutes": round(minutes, 2)}
            except GenerationProtocolError as exc:
                return self.record(3, False, f"{name}: {exc}"[-1500:],
                                   characterized=list(results))
            scored = AUDIT / f"{name}_recovery_search.json"
            rc = self.gate(f"{name}_scoring",
                           [str(REPO / "scripts/autoinit/score_recovery_search.py"),
                            "--generations", str(gen_dir), "--label", name,
                            "--seed", str(control.seed), "--out", str(scored),
                            "--per-sample", str(AUDIT / f"{name}_per_sample.jsonl")],
                           timeout=1800)
            if rc.returncode != 0:
                return self.record(3, False, f"{name} scoring rc={rc.returncode}; "
                                   f"tail: ...{(rc.stdout + rc.stderr)[-1200:]}",
                                   characterized=list(results))
            result = json.loads(scored.read_text())
            observed_eval = RecoveryEvaluationProtocol(
                generation=observe_generation_protocol(
                    [json.loads(p.read_text()) for p in sorted(gen_dir.glob("*.json"))
                     if not p.name.endswith(".generations.jsonl")]).protocol,
                scoring_contract=result["scoring_contract"]["contract"],
                scoring_digest=result["scoring_contract"]["digest"],
                battery_artifact=result["battery"]["artifact"],
                battery_manifest_sha256=result["battery"]["manifest_sha256"],
                battery_content_sha256=result["battery"]["content_sha256"])
            try:
                observed_eval.require_comparable(self.evaluation_protocol,
                                                 context=f"{name}")
            except GenerationProtocolError as exc:
                return self.record(3, False, f"{name}: {exc}"[-1500:])
            result["evaluation_protocol_hash"] = \
                observed_eval.evaluation_protocol_hash
            result["bound_to"] = {
                "evaluation_protocol_hash": observed_eval.evaluation_protocol_hash,
                "imported_control": control.as_dict()["control_binding"],
                "probe_id": control.probe_id,
                "weights_sha256": control.weights_sha256,
            }
            scored.write_text(json.dumps(result, indent=2) + "\n")
            results[name] = result
            per_seed.append({"seed": control.seed, "n": result["n"],
                             "usable": result["usable"],
                             "correct": result["correct"]})
            mark(f"CHARACTERIZED:{name}")

        pooled = POOLED_COUNTS_V1.pool(per_seed)
        sa, sb = (results[c[0]] for c in CONTROLS)
        thresholds = {
            "schema": "aadistill.autoinit.materialized_thresholds/v2",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_protocol_hash":
                self.evaluation_protocol.evaluation_protocol_hash,
            "battery": "recovery_search_v2",
            "from_imported_controls": {n: c.probe_id
                                       for n, c in self.imported.items()},
            "aggregation": POOLED_COUNTS_V1.as_dict(),
            "pooled": pooled, "per_seed": per_seed,
            "equivalence_interval": EquivalenceRule(
                n_pooled=sa["n_scorable"] + sb["n_scorable"]).materialize(
                    p_pool=pooled["correct_overall"], p_sa=sa["correct_overall"],
                    p_sb=sb["correct_overall"]).as_dict(),
            "feasibility_floor": FeasibilityRule(
                n_pooled=sa["n"] + sb["n"]).materialize(
                    u_pool=pooled["usable_rollout_rate"],
                    u_sa=sa["usable_rollout_rate"],
                    u_sb=sb["usable_rollout_rate"]).as_dict(),
            "per_capability_control_baseline": {
                cap: {"pooled_usable_rate": round(
                          (sa["per_capability"][cap]["usable"]
                           + sb["per_capability"][cap]["usable"])
                          / (sa["per_capability"][cap]["n"]
                             + sb["per_capability"][cap]["n"]), 4),
                      "sa": sa["per_capability"][cap]["usable_rollout_rate"],
                      "sb": sb["per_capability"][cap]["usable_rollout_rate"]}
                for cap in sa["per_capability"]},
            "catastrophic_rule": CATASTROPHIC_V1.as_dict(),
            "measured_cost": {
                "generation_minutes_per_control":
                    {n: g["minutes"] for n, g in generation.items()},
                "note": ("the number Phase A must be repriced from; it had never "
                         "been measured for this battery"),
            },
            "materialized_from": "two imported permanent controls; nothing trained",
        }
        thresholds["report_sha256"] = sha256_json(thresholds)
        (AUDIT / "materialized_thresholds.json").write_text(
            json.dumps(thresholds, indent=2) + "\n")
        return self.record(3, True, thresholds=thresholds, generation=generation,
                           per_control={k: {m: v[m] for m in
                                            ("usable_rollout_rate", "correct_overall",
                                             "correct_given_usable", "n")}
                                        for k, v in results.items()})

    # -- run ---------------------------------------------------------------
    def run(self) -> int:
        mark("DRIVER_START")
        stages = {0: self.stage0, 1: self.stage1, 2: self.stage2, 3: self.stage3}
        failed: list[int] = []
        for stage in sorted(stages):
            try:
                ok = stages[stage]()
            except (RecoveryAdmissionError, ControlImportError) as exc:
                self.record(stage, False, f"refused: {exc}"[-1500:])
                ok = False
            except Exception as exc:                              # noqa: BLE001
                self.record(stage, False,
                            f"{type(exc).__name__}: {exc}"[-1500:])
                ok = False
            if ok:
                continue
            blocking = {s.stage for s in CONTINUATION_PLAN_V1.stages if s.blocking}
            if stage in blocking:
                mark("CONTINUATION_FAILED")
                say("stopping before any later stage; the imported controls are "
                    "untouched and nothing was trained")
                self.finish(False, failed=[stage])
                return 20 + stage
            # Non-blocking — for CLEANUP ONLY. Evidence and teardown still
            # happen, and the continuation is still a failure.
            mark("STAGE_NONBLOCKING_FAIL")
            failed.append(stage)
        if failed:
            mark("CONTINUATION_INCOMPLETE")
            say(f"continuation INCOMPLETE: stage(s) {failed} failed — "
                f"${self.usd():.2f}. Artifacts are collected and the pod is torn "
                "down, and the continuation is NOT successful.")
            self.finish(False, failed=failed)
            return 20 + failed[0]
        mark("ALL_DONE")
        say(f"continuation complete — ${self.usd():.2f}. Nothing was trained and "
            "Phase A is not reachable from this driver.")
        self.finish(True, failed=[])
        return 0

    def finish(self, success: bool, *, failed: list[int]) -> None:
        self.ev["continuation_successful"] = success
        self.ev["failed_stages"] = failed
        self.ev["outcome"] = "SUCCESS" if success else (
            "INCOMPLETE" if failed and all(
                s not in {x.stage for x in CONTINUATION_PLAN_V1.stages if x.blocking}
                for s in failed) else "FAILED")
        self.ev["cleanup_is_not_success"] = (
            "collection and teardown run regardless of the characterization "
            "outcome; they are cleanup, and a clean cleanup does not make a "
            "failed characterization a successful continuation")
        # Restated in the OUTCOME record, not only at construction: this is the
        # part a reader checks, and it must be true of the session that ran.
        self.ev["trains_anything"] = False
        self.ev["retrained_controls"] = False
        self.ev["phase_a_started"] = False
        self.ev["phase_a_reachable_from_this_driver"] = False
        self.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    # No stage names a training step, and there is no stage 4.
    ap.add_argument("--stage", default="all", choices=("all",))
    ap.add_argument("--image-digest", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--spent-usd", type=float, default=0.0)
    ap.add_argument("--soft-stop-usd", type=float, required=True)
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--characterization-minutes", type=float, default=18.0)
    args = ap.parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    driver = ContinuationDriver(args)
    driver.auth.require_within_cap(args.authorized_usd, what="session backstop")
    return driver.run()


if __name__ == "__main__":
    raise SystemExit(main())
