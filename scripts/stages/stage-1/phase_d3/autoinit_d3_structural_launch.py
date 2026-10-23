#!/usr/bin/env python3
"""The D3 STRUCTURAL session launcher. One session, declared completely.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/autoinit_d3_structural_launch.py \
        --scr <durable store> --session-commit <sha> --bundle <canonical> \
        --run-id d3_structural_<utc> --country-code AU --create-attempts 144

The shared `SessionRunner` owns acquisition, the independent watchdog at the
plan's hard-terminate minutes, teardown on every exit path and collection
before teardown. This file owns what makes the session D3-structural's:

* the typed one-use grant (`D3StructuralAuthorization`) and its `$0` gates --
  the run id and registration, the registered treatment spec re-derived and
  equal to the bound path hash, the session commit, the same-failure rule and
  the bundle round trip;
* HOST ADMISSION to the shared protocol's image identity -- the host the
  D-series probes train on, and one on which q2's own path reproduced
  byte-exactly -- so the product a probe later trains from was built under the
  numerics the comparison assumes;
* DURABILITY: the driver's evidence and its per-step journal relay off the pod
  on every poll; at collection the product (the new initial checkpoint) comes
  home and is re-identified at all four identities BEFORE the pod may go,
  whenever the outcome is a new initialization or a difference to diagnose.
  A NO_STRUCTURAL_CHANGE product is q2's own bytes, already durable, and is
  not moved (P8.4).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, LocalAsset, MarkerPolicy,
    ProductFetchResult, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    same_failure_gate, session_commit_gate,
)
from shared.deployment import MAIN_RELAY, deployment_commands  # noqa: E402
from shared.run_layout import rel_run_dir  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from stages.phase_d1.autoinit_d1_launch import SCIENCE_ASSETS as D1_ASSETS  # noqa: E402
from stages.phase_d3 import d3_design as D  # noqa: E402
from stages.phase_d3.autoinit_d3_structural_driver import (  # noqa: E402
    EVIDENCE_FILENAME, POD_WORKDIR_REL, PRODUCT_FILENAME, STATUS_PATH,
    STEPS_JOURNAL,
)
from stages.phase_d3.d3_authorization import (  # noqa: E402
    D3StructuralAuthorization, governance,
)

#: DECLARED, and `autoinit_preflight_setup.sh` has the matching branch.
SESSION_KIND = "d3_structural"
RUNG = "structural"
ACCOUNT_OPERATIONAL_RESERVE_USD = 5.0
TEST_PATHS = ("scripts/stages/stage-1/phase_d3/tests/structural_pod_preflight",)
REPO = "/workspace/aad"
AUDIT_DIRNAME = "autoinit_d3_structural"
EVIDENCE_DIR = f"artifacts/audit/{AUDIT_DIRNAME}"
STORE_SUBDIR = "store"
RUN_LOG_PATH = "/workspace/autoinit_d3_structural_run.log"
DRIVER = "scripts/stages/stage-1/phase_d3/autoinit_d3_structural_driver.py"
#: The two calibration mixtures q2's path reads. The state-eval asset is not
#: needed -- nothing is evaluated -- but the pod's test gate imports D1's
#: session, which loads it, so the setup stages D1's proven set.
SCIENCE_ASSETS: tuple[LocalAsset, ...] = tuple(D1_ASSETS)
#: The product files: weight-only, exactly as every searched leaf is saved.
PRODUCT_FILES = ("model.safetensors", "config.json", "generation_config.json")
#: Outcomes whose product must be secured before the pod is deleted.
PRODUCT_REQUIRED_FOR = (D.STRUCTURAL_CHANGE, D.UNEXPLAINED_BYTE_DIFFERENCE)


def run_dir_for(run_id: str | None) -> str:
    return rel_run_dir(D.EXPERIMENT_ID, run_id or "unrecorded", D.STAGE_ID)


def auth_path_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/authorization.json"


def bundle_record_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/bundle.json"


def driver_deadline_seconds(ctx: Any, plan: Any) -> int:
    """Seconds from the driver's start to the plan's soft stop (floored)."""
    soft = float(plan.soft_stop_minutes)
    elapsed = float(getattr(ctx, "elapsed_minutes", 0.0) or 0.0)
    return int(max(60.0, (soft - elapsed) * 60.0))


def driver_command(ctx: Any, plan: Any) -> str:
    run_id = getattr(ctx.args, "run_id", "unrecorded")
    return (f"/opt/train/bin/python {REPO}/{DRIVER} "
            f"--out '{REPO}/{EVIDENCE_DIR}' --run-id '{run_id}' "
            f"--authorization '{REPO}/{auth_path_for(run_id)}' "
            f"--workdir '{REPO}/{POD_WORKDIR_REL}' --status '{STATUS_PATH}' "
            f"--device cuda --deadline-s {driver_deadline_seconds(ctx, plan)}")


#: The driver's append-only streams, relayed off the pod on every poll: a pod
#: lost after DEPTH still leaves every completed step's identity and DEPTH's
#: whole round record.
RELAY_STREAMS = ((STEPS_JOURNAL, STEPS_JOURNAL), ("journal.jsonl",
                                                  "journal.jsonl"))


def relay_streams(ctx: Any) -> tuple[tuple[str, str], ...]:
    return tuple((f"{REPO}/{EVIDENCE_DIR}/{remote}", local)
                 for remote, local in RELAY_STREAMS)


# --- host admission -------------------------------------------------------------

def host_admission(image_digest: str) -> tuple[bool, str]:
    """Admit only the shared protocol's training image identity."""
    from stages.d_series import shared_evaluation as SE

    want = SE._read(SE.RECORD, REPO_ROOT)["recovery"]["training_image_digest"]
    if image_digest == want:
        return True, f"image identity {image_digest} is the protocol's"
    return False, (f"image identity {image_digest!r} is not the shared "
                   f"protocol's {want!r}; the D3 product would not be built "
                   "under the numerics its probes and comparators assume. "
                   "Redraw.")


# --- products ---------------------------------------------------------------------

class ProductUnreadable(RuntimeError):
    """The structural evidence should exist and cannot be read."""


def structural_outcome(ctx) -> tuple[str | None, dict[str, Any] | None]:
    """The driver's outcome and product record, from the fetched reports.

    ``(None, None)`` when the driver's own evidence shows it never reached the
    comparison; unknown is RAISED, never read as absence.
    """
    store = Path(ctx.args.scr) / STORE_SUBDIR
    evidence, product = store / EVIDENCE_FILENAME, store / PRODUCT_FILENAME
    if not evidence.is_file():
        raise ProductUnreadable("the structural evidence did not come home")
    doc = json.loads(evidence.read_text())
    outcome = (doc.get("structural") or {}).get("outcome")
    reached = any(s.get("stage") == "E" for s in doc.get("stages") or [])
    if outcome is None and not reached:
        return None, None
    if not product.is_file():
        raise ProductUnreadable(
            f"the session reached outcome {outcome!r} and no product record "
            "came home; the initialization may be on the pod")
    return outcome, json.loads(product.read_text())


def fetch_product(ctx) -> list:
    """The new initial checkpoint, when its outcome requires it, verified."""
    from stages.phase_d1.autoinit_d1_behavioural_launch import _scp_from_pod
    from stages.phase_d1.behavioural_materialize import (
        D1MaterializeError, _verify_bytes_at,
    )

    if not ctx.products_eligible:
        return []
    try:
        outcome, product = structural_outcome(ctx)
    except ProductUnreadable as exc:
        ctx.say(f"  CANNOT READ THE STRUCTURAL OUTCOME: {exc}")
        return [ProductFetchResult(kind="transfer", rc=1, detail=str(exc))]
    if outcome not in PRODUCT_REQUIRED_FOR:
        return []
    store = Path(getattr(ctx.args, "ckpt_store", None)
                 or Path(ctx.args.scr) / "products")
    dest = store / product["identities"]["artifact_digest"]
    dest.mkdir(parents=True, exist_ok=True)
    rcs = {}
    for name in PRODUCT_FILES:
        rcs[name] = _scp_from_pod(
            ctx, f"{product['checkpoint_path']}/{name}", dest / name,
            limit_min=int(getattr(ctx.args, "ckpt_fetch_limit_min", 45)))
    want = {k: product["identities"][k] for k in (
        "artifact_digest", "weights_digest", "single_shard_sha256",
        "arch_signature")}
    try:
        got = _verify_bytes_at(dest, want, what="the D3 product")
        verified, why = True, f"re-identified at {got['verified']}"
    except (D1MaterializeError, OSError, KeyError, ValueError) as exc:
        verified, why = False, str(exc)
    ok = verified and all(rc == 0 for rc in rcs.values())
    ctx.evidence["product_secured"] = {"dest": str(dest), "rcs": rcs,
                                       "verified": verified, "why": why,
                                       "outcome": outcome}
    return [ProductFetchResult(
        kind="transfer", rc=0 if ok else 1,
        detail=f"{want['artifact_digest'][:16]}: rcs={rcs}, {why}")]


def product_secured(ctx, fetched) -> tuple[bool, str]:
    try:
        outcome, _ = structural_outcome(ctx)
    except ProductUnreadable as exc:
        return False, f"PRODUCT GATE FAILURE: {exc}"
    if outcome is None:
        return True, ("the driver did not reach the comparison; its evidence "
                      "is the product")
    if outcome not in PRODUCT_REQUIRED_FOR:
        return True, (f"{outcome}: the treatment reproduced q2's bytes, which "
                      "are already durable; nothing new to secure")
    ok = [f for f in fetched if getattr(f, "ok", False)]
    if len(ok) != 1:
        return False, (f"{outcome}: the new initial checkpoint is not secured "
                       "and re-identified off-pod; the pod may not be deleted")
    return True, f"{outcome}: the product is secured and re-identified"


# --- the $0 gates ---------------------------------------------------------------------

def run_identity_gate(ctx) -> tuple[bool, str]:
    try:
        ctx.auth.require_run_id(getattr(ctx.args, "run_id", ""))
        ctx.auth.require_plan(D.registration_hash(REPO_ROOT))
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, (f"run {ctx.auth.run_id}, registration "
                  f"{ctx.auth.plan_hash[:12]}")


def contract_gate(ctx) -> tuple[bool, str]:
    """The registered treatment spec, re-derived HERE, is the bound one."""
    try:
        spec = D.registered_treatment_spec(REPO_ROOT)
        contract = D.treatment_contract(
            D.control_spec(D.control_leaf(REPO_ROOT)), spec)
        ctx.auth.require_contract(spec.spec_hash)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    ctx.evidence["d3_structural_contract_check"] = {
        "treatment_path_hash": spec.spec_hash,
        "control_path_hash": contract["control_path_hash"],
        "treated_step": contract["treated_step"]}
    return True, f"treatment verified at $0: {spec.spec_hash[:16]}"


def bundle_staged_gate(ctx) -> tuple[bool, str]:
    import tempfile

    from aadistill.infrastructure.bundle_transport import hf_download, roundtrip

    commit = getattr(ctx.args, "session_commit", "")
    try:
        D1S.require_canonical_bundle(getattr(ctx.args, "bundle", ""), commit)
        record = json.loads((REPO_ROOT / bundle_record_for(
            getattr(ctx.args, "run_id", None))).read_text())
        auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
        with tempfile.TemporaryDirectory(prefix="d3s-bundle-") as tmp:
            evidence = roundtrip(
                D1S.transport(), session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=(REPO_ROOT / auth_rel).read_bytes(),
                authorization_path=auth_rel,
                expected_harness_digest=ctx.auth.harness_source_digest,
                harness_files=tuple(ctx.auth.harness_source_files),
                download=hf_download, workdir=Path(tmp))
    except Exception as exc:                                      # noqa: BLE001
        return False, f"the pod could not obtain the granted commit: {exc}"
    ctx.evidence["bundle_staged_check"] = evidence
    return True, f"{evidence['canonical_bundle_name']} round-trips"


RUNS_ROOT = REPO_ROOT / "logs/stages/stage-1/phase_d3/runs"


def _prior_attempt_logs() -> list[tuple[str, str]]:
    if not RUNS_ROOT.is_dir():
        return []
    prefix = governance(RUNG).run_prefix
    out = []
    for d in sorted(p for p in RUNS_ROOT.iterdir()
                    if p.is_dir() and p.name.startswith(prefix)):
        log = d / "runtime" / "launcher.log"
        if log.is_file():
            out.append((d.name, log.read_text(errors="replace")))
    return out


def _corrective_change_since() -> tuple[bool, str]:
    """Anything tracked outside the run directories changed since the
    previous attempt's session commit."""
    logs = _prior_attempt_logs()
    if not logs:
        return True, "no prior structural attempt exists"
    for name, _ in reversed(logs):
        record = REPO_ROOT / run_dir_for(name) / "runtime" / "session.json"
        if not record.is_file():
            continue
        commit = (json.loads(record.read_text()).get("session_commit_check")
                  or {}).get("session_commit")
        if not commit:
            continue
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO_ROOT).stdout.strip()
        diff = subprocess.run(["git", "diff", "--name-only", f"{commit}..{head}"],
                              capture_output=True, text=True,
                              cwd=REPO_ROOT).stdout.splitlines()
        outside = [p for p in diff
                   if not p.startswith("logs/stages/stage-1/phase_d3/runs/")]
        return (bool(outside), f"{len(outside)} tracked path(s) changed since "
                f"{name}" if outside else f"only run records changed since {name}")
    return False, ("a prior structural attempt exists and its session commit "
                   "cannot be read")


def prechecks(args) -> tuple:
    return (run_identity_gate, contract_gate,
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", None)),
                                check_lineage=True),
            same_failure_gate(_prior_attempt_logs,
                              corrective_change=_corrective_change_since),
            bundle_staged_gate)


def budget_spec(repo_root: Path, *, authorized_minutes: float | None = None
                ) -> BudgetSpec:
    """DECOMPOSES the priced bound so the plan lands exactly on it."""
    from aadistill.infrastructure.budget import Phase
    from shared.step_times import MEASURED_STEP_SECONDS

    from stages.phase_d1.d1_authorization import live_money
    from stages.phase_d3.d3_authorization import session_ceiling

    priced = session_ceiling(repo_root, RUNG)
    overhead = float(priced["session_overhead_minutes"])
    accepted_hard = float(priced["hard_ceiling_minutes"])
    if authorized_minutes is not None:
        accepted_hard = min(accepted_hard, float(authorized_minutes))
    expected_total = float(priced["expected_minutes"])
    contingency, recovery_reserve = 0.10, 10.0
    reserve = round(accepted_hard - expected_total * (1.0 + contingency)
                    - recovery_reserve, 2)
    if reserve < 0:
        raise SystemExit(
            f"the accepted {accepted_hard:.2f}-minute bound cannot hold the "
            f"expected {expected_total:.2f} min plus contingency and reserve")
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=MEASURED_STEP_SECONDS,
        step_source="unused: the structural session trains nothing",
        setup_minutes=round(min(overhead, expected_total) * 0.6, 2),
        transfer_minutes=round(min(overhead, expected_total) * 0.15, 2),
        other_phases=(Phase("fixed_path_expected", round(max(
            0.0, expected_total - min(overhead, expected_total) * 0.75), 2)),),
        contingency_fraction=contingency,
        soft_stop_reserves=(Phase("depth_and_host_variability", reserve),),
        artifact_recovery_reserve_minutes=recovery_reserve,
        below_floor_reason=(f"priced by {priced['_basis']}; this spec "
                            "decomposes that bound and adds nothing to it"),
        account_balance_required_usd=round(max(
            float(live_money(repo_root)["per_session_envelope_usd"]),
            float(priced["hard_ceiling_usd"]) + ACCOUNT_OPERATIONAL_RESERVE_USD),
            4))


def _authorized_minutes(args) -> float | None:
    path = REPO_ROOT / auth_path_for(getattr(args, "run_id", None))
    if not path.is_file():
        return None
    return D3StructuralAuthorization.load(path).hard_runtime_minutes


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-d3-structural",
        schema="aadistill.autoinit.d3_structural_session/v1",
        description=(
            "D3 STRUCTURAL: D1-q2's frozen fixed path from the original "
            "teacher with ONE declared change -- DEPTH's KL measured against "
            "the stored teacher reference instead of the FFN-compressed "
            "parent. The teacher reference first; q2's FFN prefix pinned to "
            "its digest; DEPTH's every round and both references' KL "
            "recorded; WIDTH and ATTENTION recomputed on the new parent; the "
            "structural outcome against q2; the product secured."),
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        authorization_loader=D3StructuralAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=f"autoinit.v1.{D.EXPERIMENT_ID}.{RUNG}",
        plan_hash=D.registration_hash(REPO_ROOT),
        budget=budget_spec(REPO_ROOT,
                           authorized_minutes=_authorized_minutes(args)),
        setup=SetupManifest(
            env={"SESSION_KIND": SESSION_KIND},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_KIND", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "TEACHER_READY", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            local_assets=SCIENCE_ASSETS,
            teacher_revision=D1S.root_teacher_identity(REPO_ROOT)["revision"],
            test_paths=TEST_PATHS,
            uv_max_seconds=getattr(args, "uv_max_s", 1500),
            tests_max_seconds=getattr(args, "tests_max_s", 1200)),
        driver_command=driver_command,
        driver_job_id="autoinit_d3_structural",
        status_path=STATUS_PATH,
        run_log_path=RUN_LOG_PATH,
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note=("the structural session stopped -- collecting its "
                          "evidence and every completed step's record")),
        host_admission=host_admission,
        precheck=prechecks(args),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename=EVIDENCE_FILENAME,
            archive_basename="d3_structural_artifacts.tar.gz",
            spec_success=("configs/stages/stage-1/phase_d3/"
                          "d3_structural_artifacts.json"),
            spec_failed=("configs/stages/stage-1/phase_d3/"
                         "d3_structural_artifacts_failed.json"),
            report_names=(EVIDENCE_FILENAME, PRODUCT_FILENAME, STEPS_JOURNAL,
                          "journal.jsonl"),
            fetch_products=fetch_product,
            products_secured=product_secured,
            extra_relay_streams=relay_streams),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Module scope; every required argument a STRING (the dispatch probe)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--runpod-config",
                    default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--relay-repo", default=MAIN_RELAY)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=None)
    ap.add_argument("--ckpt-store", default=None)
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=45)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--token-src",
                    default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=5)
    ap.add_argument("--country-code", default=None)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=None)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=1200)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def resolve_operational_defaults(args) -> dict[str, Any]:
    from stages.phase_d3.d3_authorization import session_ceiling

    priced = session_ceiling(REPO_ROOT, RUNG)
    resolved: dict[str, Any] = {}
    if getattr(args, "max_price", None) is None:
        args.max_price = resolved["max_price"] = float(priced["price_per_hour"])
    if getattr(args, "disk_gb", None) is None:
        args.disk_gb = resolved["disk_gb"] = int(priced["container_disk_gb"])
    if getattr(args, "out", None) is None:
        args.out = resolved["out"] = (
            f"{run_dir_for(getattr(args, 'run_id', None))}/runtime/session.json")
    if getattr(args, "poll_limit_min", None) is None:
        args.poll_limit_min = resolved["poll_limit_min"] = round(
            float(priced["hard_ceiling_minutes"]) + 120.0, 1)
    return resolved


def main(argv: list[str] | None = None) -> int:
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.session_runner import SessionRunner, run_session

    args = build_parser().parse_args(argv)
    resolved = resolve_operational_defaults(args)
    try:
        session = spec(args)
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11
    if args.dry_run:
        print(f"session_id {session.session_id}  plan_hash "
              f"{session.plan_hash[:24]}  resolved {resolved}")
        try:
            runner = SessionRunner(session, args, REPO_ROOT)
        except AuthorizationError as exc:
            print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
            return 11
        runner.run()
        passed = runner.ev.get("terminal") == "DRY_RUN_GATES_PASSED"
        runner.ev["passed"] = False
        runner.save()
        print(f"terminal {runner.ev.get('terminal')}; CREATED NOTHING")
        return 0 if passed else 11
    try:
        return run_session(session, args, REPO_ROOT,
                           summary=("D3 structural: the step records, DEPTH's "
                                    "round evidence, the structural outcome "
                                    "and -- for a new initialization -- the "
                                    "product are what it owes."))
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11


if __name__ == "__main__":
    raise SystemExit(main())
