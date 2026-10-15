#!/usr/bin/env python3
"""Train and score D2's eight screening probes; stop at preservation.

    /opt/train/bin/python scripts/stages/stage-1/phase_d2/autoinit_d2_screening_driver.py \
        --out artifacts/audit/autoinit_d2_screening --run-id <id> \
        --authorization <governance/authorization.json> --arm-root <staged arms>

Four committed D2 initializations x the two shared screening seeds. The seed is
the REPLICATE and the initialization the TREATMENT; the comparator is B's two
IMPORTED units, so nothing about B is trained or generated here.

EVERY PROBE GOES THROUGH D1'S EXACT PATH. The per-probe steps -- the config
derived from the frozen recipe with only C1's override set, the device
handoff, the trainer, the evaluation package, the attestation from the first
real package, the admission of each probe's generations against it, the pinned
scorer, the headroom check, the acked workdir release and the resume of a
restored trained checkpoint -- are `autoinit_d1_behavioural_driver`'s functions
called with D2's paths. The trainer, evaluator and scorer are the shared
protocol's pinned bytes (the contract refuses a tree whose source digests
differ). Three things are D2's own and each is a refusal or a diagnostic:

* BEFORE ANY PROBE TRAINS, the staged arms are verified at all four
  identities, the contract is re-derived and must hash to the authorized one,
  and the TRAINING RUNTIME fingerprint this interpreter observes must equal
  the protocol's -- a probe trained under another runtime is not paired with
  B's units;
* AFTER THE ATTESTATION, its evaluation-protocol hash must equal the shared
  rung's, before any generation; and generation runs through the read-only
  token-id sidecar wrapper around the UNCHANGED generator -- if the wrapper
  itself fails, its partial output is preserved aside and the plain generator
  runs, so the diagnostic can never cost the endpoint;
* AFTER EACH PROBE IS SCORED, the rollout rescoring diagnostic runs as its own
  subprocess; its failure is recorded and never touches the scored probe.

The ranking is NOT computed here: `rank_d2_screening.py` applies the frozen
rule off-pod, from the secured rows, at `$0`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/pod", "scripts/autoinit",
               "scripts/stages/stage-1"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

SUCCESS_MARKER = "ALL_DONE"
FAILURE_MARKER = "RUN_FAILED"
CHECK_ONLY_MARKER = "CHECK_ONLY_OK"

EVIDENCE_FILENAME = "d2_screening.json"
ATTESTED_NAME = "d2_screening_attested_protocol.json"
ATTESTED_SCHEMA = "aadistill.phase_d2.screening_attested_protocol/v1"
PROBE_ROOT_REL = "artifacts/stages/stage-3/d2_screening"
EVAL_PACKAGES_REL = "artifacts/stages/stage-3/d2_screening_eval"
ACK_REL = "artifacts/audit/autoinit_d2_screening/release_acks"
SCORER = REPO_ROOT / "scripts/stages/stage-1/phase_d1/score_d1_screening.py"
TOKEN_ID_WRAPPER = REPO_ROOT / "scripts/shared/evaluation/uncapped_eval_token_ids.py"
RESCORER = REPO_ROOT / "scripts/stages/stage-1/phase_d2/rescore_d2_probe.py"


class D2ScreeningDriverError(RuntimeError):
    """This session cannot proceed on the evidence it has."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--authorization", default=None)
    ap.add_argument("--arm-root", default=None,
                    help="where the relay staged the arms; None reads the "
                         "dev host's durable products (check-only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-digest", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--probe-train-minutes", type=float, default=60.0)
    ap.add_argument("--probe-eval-minutes", type=float, default=40.0)
    ap.add_argument("--rescoring-minutes", type=float, default=20.0)
    ap.add_argument("--check-only", action="store_true")
    return ap


def training_runtime(image_digest: str | None) -> dict[str, Any]:
    """The runtime fingerprint the trainer will observe, observed HERE first.

    The driver runs under the trainer's interpreter, so this is the same
    observation `train_stage3.py` makes; refusing on it costs seconds, where
    discovering it in the first run manifest costs a probe.
    """
    from aadistill.initialization.planning.recovery import (
        RuntimeEnvironmentFingerprint,
    )
    from stages.phase_d2 import d2_screening as SC

    observed = RuntimeEnvironmentFingerprint.observe(image_digest=image_digest)
    want = SC.protocol(REPO_ROOT)["recovery"]["training_runtime_digest"]
    row = {"observed": observed.as_dict(), "digest": observed.digest,
           "protocol_training_runtime_digest": want,
           "matches": observed.digest == want}
    if not row["matches"]:
        raise D2ScreeningDriverError(
            f"the training runtime digests to {observed.digest[:12]} and the "
            f"shared protocol pins {want[:12]} ({observed.as_dict()}); a probe "
            "trained here would not be paired with B's imported units")
    return row


def generate(package_dir: Path, prompt_files: list[str], gen_dir: Path, *,
             label: str, image_digest: str | None, timeout_s: int,
             out: Path) -> dict[str, Any]:
    """The UNCHANGED generator through the token-id wrapper; plain on failure."""
    from shared.source_sets import generation_source_digest
    from stages.phase_c1.autoinit_c1_driver import UNCAPPED_EVAL
    from stages.phase_d1 import autoinit_d1_behavioural_driver as D1D

    argv = ["--model", str(package_dir), "--label", label,
            "--prompts", *prompt_files, "--out-dir", str(gen_dir),
            "--diagnostics"]
    digest = generation_source_digest(REPO_ROOT)["digest"]
    started = time.time()
    run = subprocess.run(
        ["/opt/vllm/bin/python", str(TOKEN_ID_WRAPPER),
         "--expect-generation-source-digest", digest, *argv],
        capture_output=True, text=True, timeout=timeout_s,
        env=D1D.child_env(image_digest))
    (out / f"{label}_eval_tail.log").write_text(
        (run.stdout + run.stderr)[-2000:])
    row: dict[str, Any] = {"wrapper_rc": run.returncode,
                           "generate_seconds": round(time.time() - started, 1),
                           "via": "token_id_wrapper"}
    if run.returncode == 0:
        return row
    #: THE DIAGNOSTIC MUST NOT COST THE ENDPOINT. The wrapper's partial output
    #: is preserved aside, and the plain generator -- the protocol's own entry
    #: point, the same argv -- produces the probe's generations.
    if gen_dir.exists():
        aside = gen_dir.with_name(f"{gen_dir.name}.wrapper_failed")
        shutil.move(str(gen_dir), str(aside))
        row["wrapper_partial_preserved_at"] = str(aside)
    started = time.time()
    plain = subprocess.run(
        ["/opt/vllm/bin/python", str(UNCAPPED_EVAL), *argv],
        capture_output=True, text=True, timeout=timeout_s,
        env=D1D.child_env(image_digest))
    (out / f"{label}_eval_plain_tail.log").write_text(
        (plain.stdout + plain.stderr)[-2000:])
    row.update({"via": "plain_generator_after_wrapper_failure",
                "plain_rc": plain.returncode,
                "generate_seconds": round(time.time() - started, 1)})
    return row


def sidecar_alignment(gen_dir: Path) -> dict[str, Any]:
    """Per stratum, what the token-id sidecar says about its own rows."""
    from shared.evaluation.uncapped_eval_token_ids import load_sidecar

    out = {}
    for summary in sorted(gen_dir.glob("*.json")):
        loaded = load_sidecar(summary)
        if loaded is None:
            out[summary.stem] = {"available": False}
            continue
        header, rows = loaded
        out[summary.stem] = {
            "available": bool(header.get("available")),
            "exact": bool(header.get("exact")),
            "rows": len(rows),
            "rows_aligned": sum(1 for r in rows.values()
                                if (r.get("alignment") or {}).get("aligned")),
            "reason_unavailable": header.get("reason_unavailable")}
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from stages.phase_d1 import autoinit_d1_behavioural_driver as D1D

    D1D.RUN_ID_FOR_PROVENANCE[0] = str(args.run_id or "")
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    status = args.status or "/workspace/autoinit_d2_screening.status"
    record: dict[str, Any] = {
        "schema": "aadistill.phase_d2.screening_session/v1",
        "run_id": args.run_id, "rung": "screening",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "RUNNING", "probes": []}
    evidence = out / EVIDENCE_FILENAME

    def save() -> None:
        evidence.write_text(json.dumps(record, indent=2, default=str) + "\n")

    D1D.mark(status, "DRIVER_START")
    try:
        from aadistill.infrastructure.manifest import sha256_json
        from stages.phase_d2 import d2_screening as SC

        #: Cheap refusals first: battery bytes, role disjointness, pack digest.
        record["cheap_preflight"] = D1D.cheap_preflight("screening", REPO_ROOT)
        save()
        #: The CONTRACT, re-derived on THIS tree with THESE staged arms (four
        #: identities hashed) and the protocol's source pins compared.
        contract = SC.session_contract(REPO_ROOT, arm_root=args.arm_root)
        record["contract"] = contract
        record["contract_hash"] = SC.contract_hash(contract)
        record["arm_verification"] = SC.require_arms_present(
            REPO_ROOT, arm_root=args.arm_root)
        save()
        if args.authorization:
            from stages.phase_d2.d2_screening_authorization import (
                D2ScreeningAuthorization,
            )

            auth = D2ScreeningAuthorization.load(args.authorization)
            auth.require_run_id(args.run_id)
            auth.require_plan(contract["preregistration_hash"])
            auth.require_contract(record["contract_hash"])
            record["authorization"] = {
                "path": str(args.authorization),
                "authorization_sha256": json.loads(Path(
                    args.authorization).read_text())["authorization_sha256"],
                "contract_hash": auth.contract_hash,
                "hard_cap_usd": auth.hard_cap_usd}
            save()
        elif not args.check_only:
            raise D2ScreeningDriverError(
                "a formal screening session runs under its one-use "
                "authorization; --authorization may be omitted only with "
                "--check-only")

        from stages.phase_c1.autoinit_c1_driver import (
            C1_PROBE_OVERRIDES, FROZEN_RECIPE, PACK_DIR,
            TOKENIZER_SIDECAR_SHA256, TOKENIZER_SOURCE, TRAINER, _trainer_bytes,
            trained_model_dir,
        )
        from stages.phase_c1.packaging import build_evaluation_package
        from aadistill.runtime.device_handoff import (
            complete_release, cuda_memory, require_headroom, require_released,
        )

        probes = SC.probes(REPO_ROOT, arm_root=args.arm_root)
        record["schedule"] = [p.probe_id for p in probes]
        arms = {a.arm_id: a for a in SC.arms(REPO_ROOT, arm_root=args.arm_root)}
        for path in (SCORER, TOKEN_ID_WRAPPER, RESCORER):
            if not path.is_file():
                raise D2ScreeningDriverError(f"{path} is missing")
        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            save()
            return 0

        record["training_runtime"] = training_runtime(args.image_digest)
        scoring = D1D.d1_scoring_contract("screening", REPO_ROOT)
        battery_fields = D1D.battery_protocol_fields(contract)
        record["scoring_contract"] = {k: scoring[k]
                                      for k in ("contract", "digest")}
        if scoring["digest"] != contract["evaluation"]["scoring_digest"]:
            raise D2ScreeningDriverError(
                f"the scoring digest {scoring['digest'][:12]} is not the "
                f"shared rung's {contract['evaluation']['scoring_digest'][:12]}")
        save()
        attested = None
        num_parameters = int(SC.selection(REPO_ROOT)["selected"][0][
            "num_parameters"])
        from stages.phase_d1 import behavioural as D1B

        battery_root = REPO_ROOT / contract["battery"]["root"]
        #: D1's construction, in D1's order: the generator runs one wave per
        #: file, so the file order is part of what generation did.
        prompt_files = [str(battery_root / f"{name}.jsonl")
                        for name in sorted(D1B.FROZEN_STRATA)]

        for index, probe in enumerate(probes):
            entry: dict[str, Any] = {
                "probe_id": probe.probe_id, "arm": probe.arm_id,
                "seed": probe.seed, "index": index,
                "state_id": arms[probe.arm_id].state_id,
                "trained": False, "scored": False}
            record["probes"].append(entry)
            save()
            print(f"[{index + 1}/{len(probes)}] {probe.probe_id}", flush=True)
            D1D.release_acked_probe_workdirs(
                record, save, ack_rel=ACK_REL,
                roots_rel=(PROBE_ROOT_REL, EVAL_PACKAGES_REL))
            model_root = REPO_ROOT / PROBE_ROOT_REL / probe.probe_id
            resumed = D1D.trained_checkpoint_state(model_root)
            if resumed is not None:
                entry.update({"resumed_from_trained": resumed, "trained": True,
                              "model_dir": resumed["model_dir"]})
                model_dir = Path(resumed["model_dir"])
                save()
            else:
                D1D.require_probe_headroom(probe.probe_id, record, save,
                                           num_parameters=num_parameters)
                config = D1D.probe_config(
                    probe, audit=out, frozen_recipe=FROZEN_RECIPE,
                    pack_dir=PACK_DIR, overrides=C1_PROBE_OVERRIDES,
                    out_root=PROBE_ROOT_REL, experiment="D2")
                entry["config"] = str(config.relative_to(REPO_ROOT))
                import gc

                before = cuda_memory()
                gc.collect()
                handoff = complete_release(before)
                require_released(handoff, what="the D2 recovery trainer")
                require_headroom(handoff["after"], need_bytes=_trainer_bytes(),
                                 what="the D2 recovery trainer")
                entry["device_handoff"] = handoff.get("verdict")
                save()
                started = time.time()
                train = subprocess.run(
                    ["/opt/train/bin/python", str(TRAINER), "--config",
                     str(config)], capture_output=True, text=True,
                    timeout=int(args.probe_train_minutes * 60 * 2),
                    env=D1D.child_env(args.image_digest))
                (out / f"{probe.probe_id}_train_tail.log").write_text(
                    (train.stdout + train.stderr)[-2000:])
                entry["train_seconds"] = round(time.time() - started, 1)
                entry["train_rc"] = train.returncode
                if train.returncode != 0:
                    entry["failed"] = "training"
                    save()
                    raise D2ScreeningDriverError(
                        f"{probe.probe_id}: the trainer exited "
                        f"{train.returncode}")
                entry["trained"] = True
                model_dir = trained_model_dir(model_root)
                entry["model_dir"] = str(model_dir)
            import hashlib as _hashlib

            shard = Path(model_dir) / "model.safetensors"
            if shard.is_file():
                entry["trained_sha256"] = _hashlib.sha256(
                    shard.read_bytes()).hexdigest()
                entry["trained_bytes"] = shard.stat().st_size
            #: The records the off-pod ranking reads, digested HERE so every
            #: off-pod copy is verified against what this pod holds. A resumed
            #: probe's were restored with its preserved checkpoint.
            entry["training_records_sha256"] = {
                name: _hashlib.sha256((model_root / name).read_bytes()
                                      ).hexdigest()
                for name in SC.TRAINING_RECORDS
                if (model_root / name).is_file()}
            #: Announced BEFORE evaluation: the launcher preserves trained
            #: weights off-pod from this flag.
            save()

            package_dir = REPO_ROOT / EVAL_PACKAGES_REL / probe.probe_id
            entry["package_dir"] = str(package_dir)
            build_evaluation_package(
                model_dir, tokenizer_source=TOKENIZER_SOURCE, dest=package_dir,
                expected_sidecar_sha256=TOKENIZER_SIDECAR_SHA256)
            if attested is None:
                attested = D1D.build_attested_protocol(
                    package_dir, image_digest=args.image_digest,
                    scoring=scoring, battery_fields=battery_fields, out=out,
                    report_name=ATTESTED_NAME, schema=ATTESTED_SCHEMA)
                record["attested_evaluation_protocol_hash"] = \
                    attested.evaluation_protocol_hash
                want = contract["evaluation"]["evaluation_protocol_hash"]
                record["attested_matches_the_shared_rung"] = (
                    attested.evaluation_protocol_hash == want)
                save()
                if attested.evaluation_protocol_hash != want:
                    raise D2ScreeningDriverError(
                        f"this session attests evaluation protocol "
                        f"{attested.evaluation_protocol_hash[:12]} and the "
                        f"shared rung pins {want[:12]}; generations here "
                        "would not be paired with B's imported units. "
                        "Stopping BEFORE any generation; the trained probe "
                        "is preserved and resumes at evaluation.")

            gen_dir = out / "generations" / probe.probe_id
            gen = generate(package_dir, prompt_files, gen_dir,
                           label=probe.probe_id, image_digest=args.image_digest,
                           timeout_s=int(args.probe_eval_minutes * 60 * 2),
                           out=out)
            entry["generation"] = gen
            entry["generate_seconds"] = gen["generate_seconds"]
            if gen.get("plain_rc", gen["wrapper_rc"]) != 0:
                entry["failed"] = "generation"
                save()
                raise D2ScreeningDriverError(
                    f"{probe.probe_id}: generation failed ({gen})")
            entry["token_id_sidecar"] = sidecar_alignment(gen_dir)
            save()

            admission = D1D.admit_generation(
                probe.probe_id, gen_dir, attested=attested, scoring=scoring,
                battery_fields=battery_fields, out=out)
            entry["generation_admission"] = {
                "comparable": True,
                "generation_fingerprint": admission["generation_fingerprint"],
                "evaluation_protocol_hash":
                    admission["evaluation_protocol_hash"]}
            save()

            result_path = out / "results" / f"{probe.probe_id}.json"
            per_sample = out / "per_sample" / f"{probe.probe_id}.jsonl"
            score = subprocess.run(
                ["/opt/train/bin/python", str(SCORER),
                 "--generations", str(gen_dir), "--label", probe.probe_id,
                 "--seed", str(probe.seed), "--out", str(result_path),
                 "--per-sample", str(per_sample),
                 "--screening-arm", probe.arm_id,
                 "--init-digest", arms[probe.arm_id].artifact_digest,
                 "--trained-run", str(model_root / "run_completion.json"),
                 "--generation-fingerprint",
                 admission["generation_fingerprint"]],
                capture_output=True, text=True, timeout=1800,
                env=D1D.child_env(args.image_digest))
            (out / f"{probe.probe_id}_score_tail.log").write_text(
                (score.stdout + score.stderr)[-2000:])
            if score.returncode != 0 or not result_path.is_file():
                entry["failed"] = "scoring"
                save()
                raise D2ScreeningDriverError(
                    f"{probe.probe_id}: scoring exited {score.returncode}")
            scored = json.loads(result_path.read_text())
            entry.update({
                "scored": True,
                "result": str(result_path.relative_to(REPO_ROOT)),
                "correct_overall": scored["correct_overall"],
                "usable_rollout_rate": scored["usable_rollout_rate"],
                "n": scored["n"], "n_scorable": scored["n_scorable"],
                "result_sha256": scored["result_sha256"]})
            save()
            print(f"  correct_overall {scored['correct_overall']:.4f} usable "
                  f"{scored['usable_rollout_rate']:.4f}", flush=True)

            #: THE DIAGNOSTIC, isolated: a failure here is recorded and the
            #: scored probe above is never read back or re-run because of it.
            rescoring_out = out / "rescoring" / f"{probe.probe_id}.json"
            started = time.time()
            try:
                resc = subprocess.run(
                    ["/opt/train/bin/python", str(RESCORER),
                     "--student", str(model_dir),
                     "--package", str(package_dir),
                     "--generations", str(gen_dir),
                     "--label", probe.probe_id,
                     "--generation-fingerprint",
                     admission["generation_fingerprint"],
                     "--battery-root", str(battery_root),
                     "--out", str(rescoring_out),
                     "--device", args.device],
                    capture_output=True, text=True,
                    timeout=int(args.rescoring_minutes * 60),
                    env=D1D.child_env(args.image_digest))
                (out / f"{probe.probe_id}_rescoring_tail.log").write_text(
                    (resc.stdout + resc.stderr)[-2000:])
                entry["rescoring"] = {
                    "rc": resc.returncode,
                    "written": rescoring_out.is_file(),
                    "seconds": round(time.time() - started, 1)}
            except Exception as exc:                              # noqa: BLE001
                entry["rescoring"] = {
                    "rc": None, "written": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.time() - started, 1)}
            save()

        record["_ranking"] = (
            "NOT COMPUTED HERE. The frozen screening rule is applied off-pod "
            "at $0 by rank_d2_screening.py, from the secured rows and B's "
            "imported units.")
        record["status"] = "COMPLETE"
        save()
    except BaseException as exc:                                  # noqa: BLE001
        import traceback

        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        raise
    finally:
        record["ended_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        written = False
        try:
            save()
            written = True
        except Exception as exc:                                  # noqa: BLE001
            print(f"could not write {evidence}: {exc}", flush=True)
        print(f"\nwrote {evidence}  status={record['status']}", flush=True)
        D1D.mark(status, D1D.terminal_marker(
            check_only=bool(args.check_only), status=record["status"],
            evidence_written=written))
    return 0 if record["status"] in ("COMPLETE", "CHECK_ONLY_OK") else 11


if __name__ == "__main__":
    raise SystemExit(main())
