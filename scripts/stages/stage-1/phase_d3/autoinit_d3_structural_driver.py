#!/usr/bin/env python3
"""The D3 STRUCTURAL session driver: q2's path with the teacher-referenced DEPTH.

    /opt/train/bin/python scripts/stages/stage-1/phase_d3/autoinit_d3_structural_driver.py \
        --out artifacts/audit/autoinit_d3_structural --run-id d3_structural_<utc> \
        --authorization <governance/authorization.json> --deadline-s N

    A  setup: the registration, the registered treatment spec and its
       single-factor contract against the control; the one-use authorization;
    B  the historical root state re-derived from the control plan's own
       identities, the execution agreement, the staged teacher verified;
    C  `materialize_fixed_path` on the TREATMENT spec, from the teacher:
         - the teacher reference over calib.reasoning_heavy@v2 is built from
           the ORIGINAL teacher before step 0 (the generic mechanism, bound to
           the teacher's id, revision and shard hash, the mixture, the Top-200
           support and the execution protocol);
         - step 0, q2's FFN prefix, must reproduce q2's pinned digest;
         - step 1, DEPTH 36 -> 28 under KL(T || C(S)), recording every round,
           every candidate, both references' KL and the cache behaviour;
         - steps 2-3, WIDTH and ATTENTION, recomputed on the actual new parent;
       every step's record is persisted the moment it completes;
    D  the structural comparison with the control and the outcome
       (NO_STRUCTURAL_CHANGE / STRUCTURAL_CHANGE / UNEXPLAINED_BYTE_DIFFERENCE);
    E  the product: the final initial checkpoint's four identities and shard
       sha256; intermediates released.

Trains nothing, evaluates nothing, selects nothing beyond DEPTH's own registered
greedy rule. AUTHORIZES NOTHING: a paid run needs `--authorization`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stages.phase_d1.autoinit_d1_driver import (  # noqa: E402
    CHECK_ONLY_MARKER, FAILURE_MARKER, SUCCESS_MARKER, Journal, mark,
    probe_environment, say, terminal_marker,
)

__all__ = ["CHECK_ONLY_MARKER", "FAILURE_MARKER", "SUCCESS_MARKER", "main"]

EVIDENCE_FILENAME = "d3_structural.json"
STEPS_JOURNAL = "steps.jsonl"
PRODUCT_FILENAME = "product.json"
STATUS_PATH = "/workspace/autoinit_d3_structural.status"
#: Where the path is materialized on the pod. Execution configuration.
POD_WORKDIR_REL = "artifacts/stages/stage-1/d3_structural"


class D3StructuralDriverError(RuntimeError):
    pass


def load_authorization(path: str, *, run_id: str, registration_hash: str,
                       treatment_path_hash: str) -> dict[str, Any]:
    from stages.phase_d3.d3_authorization import D3StructuralAuthorization

    auth = D3StructuralAuthorization.load(path)
    auth.require_run_id(run_id)
    auth.require_plan(registration_hash)
    auth.require_contract(treatment_path_hash)
    if auth.allows_recovery_training or auth.authorizes_d3_probes:
        raise D3StructuralDriverError(
            "a structural grant may not claim recovery training or the probes")
    return {"authorization_id": auth.authorization_id,
            "hard_cap_usd": auth.hard_cap_usd,
            "authorized_session_commit": auth.authorized_session_commit,
            "registration_hash": auth.plan_hash,
            "treatment_path_hash": auth.contract_hash}


def resource_usage() -> dict[str, Any]:
    """Peak device memory over the whole path (teacher, store, parent,
    candidates), and the per-step wall time is in each step record. Never
    raises: evidence about resources must not cost the result."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"device": "cpu"}
        return {"device": torch.cuda.get_device_name(0),
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / 2**30, 3),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / 2**30, 3),
                "total_gib": round(torch.cuda.get_device_properties(0)
                                   .total_memory / 2**30, 3)}
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--authorization", default=None)
    ap.add_argument("--deadline-s", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--status", default=STATUS_PATH)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    from stages.phase_d3 import d3_design as D

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    status = Path(args.status)
    journal = Journal(out / "journal.jsonl")
    record: dict[str, Any] = {
        "schema": "aadistill.phase_d3.structural_session/v1",
        "experiment_id": D.EXPERIMENT_ID, "stage_id": D.STAGE_ID,
        "run_id": args.run_id,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "STARTED", "steps": []}
    evidence_written = False

    def save() -> None:
        (out / EVIDENCE_FILENAME).write_text(
            json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")

    mark(status, "DRIVER_START")
    try:
        with journal.stage("A", "registration_and_contract") as st:
            if not args.check_only:
                record["environment"] = probe_environment()
            reg = D.registration(REPO)
            leaf = D.control_leaf(REPO)
            spec = D.registered_treatment_spec(REPO, device=args.device)
            control = json.loads((REPO / D.CONTROL_RECORD).read_text())
            contract = D.treatment_contract(
                D.control_spec(leaf, device=args.device), spec)
            record.update({
                "registration_hash": reg["registration_hash"],
                "treatment_path_hash": spec.spec_hash,
                "control_path_hash": contract["control_path_hash"],
                "contract": contract, "root_identity": D.root_identity(leaf)})
            st.result = {"registration": reg["registration_hash"][:16],
                         "treatment": spec.spec_hash[:16]}
            save()
        if args.authorization:
            with journal.stage("A", "authorization") as st:
                record["authorization"] = load_authorization(
                    args.authorization, run_id=args.run_id,
                    registration_hash=reg["registration_hash"],
                    treatment_path_hash=spec.spec_hash)
                st.result = {"hard_cap_usd":
                             record["authorization"]["hard_cap_usd"]}
                save()
        elif not args.check_only:
            raise D3StructuralDriverError(
                "--authorization is required for a paid session")
        if args.check_only:
            record["status"] = "CHECK_ONLY_OK"
            return 0

        with journal.stage("B", "root_state_and_teacher") as st:
            from stages.phase_d1 import d1_session as D1S
            from stages.phase_d1 import replay_specs as R
            from stages.phase_d1.autoinit_d1_driver import (
                resolve_root_teacher_path,
            )

            plan, plan_sha = D.control_plan(REPO)
            teacher = D1S.root_teacher_identity(REPO)
            root_state = R.root_state_from_plan(
                plan, [leaf], base_config=R.teacher_config(
                    teacher["repo_id"], teacher["revision"]))
            execution = D.execution()
            record["root_state"] = root_state
            record["execution_agreement"] = R.verify_execution_config(
                [leaf], execution=execution)
            record["execution"] = {
                "micro_batch_size": execution.micro_batch_size,
                "calibration_batch_packing": execution.calibration_batch_packing,
                "prediction_rows": execution.prediction_rows}
            record["root_teacher"] = D1S.verify_staged_teacher(
                resolve_root_teacher_path()["path"])
            record["control_plan_sha256"] = plan_sha
            st.result = {"root_state": root_state["chosen_candidate"]}
            save()

        with journal.stage("C", "materialize_treatment") as st:
            from aadistill.initialization.planning.fixed_path import (
                FixedPathDigestMismatch, materialize_fixed_path,
            )
            from aadistill.initialization.planning.search import Deadline
            from aadistill.initialization.specs.arch import get_adapter

            workdir = Path(args.workdir or (REPO / POD_WORKDIR_REL))
            steps_journal = out / STEPS_JOURNAL
            overrides = dict(root_state["config_overrides"])

            def on_step(result) -> None:
                row = result.as_dict()
                with steps_journal.open("a") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
                record["steps"].append(row)
                save()
                say(f"step {row['index']} {row['impl_id']}: "
                    f"{row['artifact_digest'][:16]} in {row['seconds']:.1f} s")

            deadline = (Deadline(seconds=float(args.deadline_s))
                        if args.deadline_s > 0 else None)
            started = time.time()
            try:
                results = materialize_fixed_path(
                    spec, adapter=get_adapter("qwen3"),
                    root_loader=lambda: R.load_root_model(
                        spec, config_overrides=overrides, device=args.device),
                    workdir=workdir, repo_root=REPO, on_step=on_step,
                    deadline=deadline, execution=execution,
                    root_identity=D.root_identity(leaf))
            except FixedPathDigestMismatch as exc:
                #: Only step 0 is output-pinned: q2's FFN prefix did not
                #: reproduce. The conditions are the control's own, so this
                #: is diagnosed, never worked around.
                record["prefix_not_reproduced"] = {
                    "step": exc.step_index, "expected": exc.expected,
                    "actual": exc.actual}
                raise
            record["materialize_seconds"] = round(time.time() - started, 1)
            record["resources"] = resource_usage()
            st.result = {"steps": len(results)}
            save()

        with journal.stage("D", "structural_comparison") as st:
            steps = record["steps"]
            at = D.treated_index(leaf)
            depth = D.depth_evidence(steps[at], control)
            outcome = D.classify_structure(
                control=control, depth_kept=depth["kept_layers"],
                step_identities=steps)
            record["depth"] = depth
            record["structural"] = outcome
            record["teacher_reference"] = (steps[at].get("operations") or {}).get(
                "reference_cache")
            st.result = {"outcome": outcome["outcome"]}
            save()

        with journal.stage("E", "product") as st:
            from stages.phase_d1.behavioural_materialize import (
                release_intermediate_steps,
            )

            final = Path(results[-1].checkpoint_path)
            product = {
                "outcome": outcome["outcome"],
                "checkpoint_path": str(final),
                "identities": {k: steps[-1][k] for k in (
                    "artifact_digest", "weights_digest",
                    "single_shard_sha256", "arch_signature",
                    "config_sha256", "num_parameters")},
                "files": {p.name: {"sha256": _sha256(p),
                                   "bytes": p.stat().st_size}
                          for p in sorted(final.iterdir()) if p.is_file()},
                "treatment_path_hash": spec.spec_hash,
                "registration_hash": reg["registration_hash"]}
            if product["files"]["model.safetensors"]["sha256"] != \
                    product["identities"]["single_shard_sha256"]:
                raise D3StructuralDriverError(
                    "the final shard does not hash to its recorded identity")
            (out / PRODUCT_FILENAME).write_text(
                json.dumps(product, indent=1, sort_keys=True) + "\n")
            record["product"] = product
            record["intermediates_released"] = release_intermediate_steps(
                D.TREATMENT_PATH_ID, results, workdir)
            if record["intermediates_released"]["failed"]:
                raise D3StructuralDriverError(
                    f"intermediates not released: "
                    f"{record['intermediates_released']['failed']}")
            st.result = {"artifact_digest":
                         product["identities"]["artifact_digest"][:16]}
            save()
        record["status"] = "COMPLETE"
    except BaseException as exc:                                  # noqa: BLE001
        record["status"] = "FAILED"
        record["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        journal.event(stage="driver", status="failed", error=str(exc)[:400])
    finally:
        record["stages"] = journal.stages
        record["elapsed_seconds"] = round(time.time() - journal.started, 3)
        record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            save()
            evidence_written = True
            say(f"\nwrote {out}/{EVIDENCE_FILENAME}  status={record['status']}")
        except OSError as exc:
            say(f"\nCOULD NOT WRITE {out}/{EVIDENCE_FILENAME}: {exc}")
        mark(status, terminal_marker(check_only=bool(args.check_only),
                                     status=str(record["status"]),
                                     evidence_written=evidence_written))
    if args.check_only:
        return 0 if record["status"] == "CHECK_ONLY_OK" else 1
    return 0 if record["status"] == "COMPLETE" and evidence_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
