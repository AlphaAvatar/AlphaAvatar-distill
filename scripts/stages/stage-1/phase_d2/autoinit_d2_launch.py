#!/usr/bin/env python3
"""The D2 formal SEARCH launcher. One session, declared completely.

    PYTHONPATH=src:scripts python scripts/stages/stage-1/phase_d2/autoinit_d2_launch.py \
        --scr <dir> --session-commit <sha> --bundle <name> --run-id d2_search_<utc>

The shared `SessionRunner` owns acquisition, the independent watchdog at the
plan's hard-terminate minutes, teardown on every exit path and collection
before teardown. This file owns what makes the session D2's: the typed grant,
the driver command, the budget decomposition, the `$0` gates and the products.

Differences from D1's launcher, each for a recorded reason:

* the driver's soft stop is `plan.soft_stop_minutes` minus the session minutes
  already elapsed when the driver starts. D1's launcher read a
  `plan.soft_stop_seconds` that `BudgetPlan` does not have, so D1's driver ran
  with no deadline at all and only the watchdog bounded it;
* no launch-readiness record or gate: every fact it re-checked is already
  bound by the grant, the session-commit gate and the bundle round trip
  (AGENTS.md P8.2.1);
* the price is `d2_budget.py`'s search price in force, which switches from
  D1's recompute-mode maxima to the cached cell only on booked, hash-verified
  stage-L evidence. When the cached price is in force, the driver's
  reference-cache regression guard stops the search safely on the first
  recomputing DEPTH expansion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, MarkerPolicy,
    ProductFetchResult, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate,
)
from shared.deployment import deployment_commands  # noqa: E402
from shared.run_layout import rel_run_dir  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from stages.phase_d1.autoinit_d1_launch import (  # noqa: E402
    SCIENCE_ASSETS, _reidentify,
)
from stages.phase_d2 import d2_session as S  # noqa: E402
from stages.phase_d2.d2_authorization import D2Authorization  # noqa: E402

#: DECLARED, and `autoinit_preflight_setup.sh` has the matching branch.
SESSION_KIND = "d2"
ACCOUNT_OPERATIONAL_RESERVE_USD = 5.0
TEST_PATHS = ("scripts/stages/stage-1/phase_d2/tests",)
REPO = "/workspace/aad"
AUDIT_DIRNAME = "autoinit_d2"
EVIDENCE_DIR = f"artifacts/audit/{AUDIT_DIRNAME}"
STORE_SUBDIR = "store"
RUN_LOG_PATH = f"{S.POD_WORKSPACE}/autoinit_d2_run.log"


def run_dir_for(run_id: str | None) -> str:
    return rel_run_dir(S.EXPERIMENT_ID, run_id or "unrecorded", S.STAGE_ID)


def auth_path_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/authorization.json"


def bundle_record_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/bundle.json"


def driver_deadline_seconds(ctx: Any, plan: Any) -> int:
    """Seconds from the DRIVER's start to the plan's soft stop.

    `soft_stop_minutes` is counted from the session's first create; the
    driver starts after acquisition and setup, so what it has left is the soft
    stop minus the minutes already elapsed. Floored at 60 s: a driver that
    starts past the soft stop is stopped by its first check, not given a
    negative budget that a falsy test would read as "no deadline".
    """
    soft = float(plan.soft_stop_minutes)
    elapsed = float(getattr(ctx, "elapsed_minutes", 0.0) or 0.0)
    return int(max(60.0, (soft - elapsed) * 60.0))


def driver_command(ctx: Any, plan: Any) -> str:
    run_id = getattr(ctx.args, "run_id", "unrecorded")
    rows = getattr(ctx.args, "prediction_rows", "all_v1")
    out = f"{REPO}/{EVIDENCE_DIR}"
    return (f"/opt/train/bin/python {REPO}/scripts/stages/stage-1/phase_d2/"
            f"autoinit_d2_driver.py --out '{out}' --run-id '{run_id}' "
            f"--authorization '{REPO}/{auth_path_for(run_id)}' "
            f"--prediction-rows '{rows}' --status '{S.STATUS_PATH}' "
            f"--device cuda --deadline-s {driver_deadline_seconds(ctx, plan)}")


# --- products: the committed Top-K -----------------------------------------

class SelectionUnreadable(RuntimeError):
    """A selection should exist and cannot be read. Never read as absence."""


def committed_selection(ctx) -> list[dict]:
    """The selected leaves from the committed record, or [] when the driver's
    own evidence shows it never reached commit. Unknown is raised."""
    from aadistill.initialization.planning import stage1_selection

    store = Path(ctx.args.scr) / STORE_SUBDIR
    selection, evidence = store / "stage1_selection.json", store / "d2_search.json"
    reached = False
    summary: list[dict] = []
    if evidence.is_file():
        doc = json.loads(evidence.read_text())
        commit = doc.get("commit") or {}
        summary = list(commit.get("selected_rows") or [])
        reached = bool(commit) or any(
            s.get("stage") == "D" and s.get("status") == "ok"
            for s in (doc.get("stages") or []))
    if selection.is_file():
        rows = list(stage1_selection.load(selection).get("selected") or [])
        if not rows:
            raise SelectionUnreadable(f"{selection} names no selected rows")
        if summary and {r.get("state_id") for r in rows} != \
                {r.get("state_id") for r in summary}:
            raise SelectionUnreadable("the committed selection and the driver's "
                                      "summary disagree about which leaves")
        return rows
    if reached or summary or not evidence.is_file():
        raise SelectionUnreadable(
            "the session reached commit or its evidence did not come home, and "
            "no committed selection was fetched; the leaves may be on the pod")
    return []


def fetch_selected_checkpoints(ctx) -> list:
    import subprocess

    if not ctx.products_eligible:
        return []
    try:
        rows = committed_selection(ctx)
    except SelectionUnreadable as exc:
        ctx.say(f"  CANNOT READ THE COMMITTED SELECTION: {exc}")
        return [ProductFetchResult(kind="transfer", rc=1, detail=str(exc))]
    store = Path(getattr(ctx.args, "ckpt_store", None)
                 or Path(ctx.args.scr) / "products")
    store.mkdir(parents=True, exist_ok=True)
    fetched = []
    for row in rows:
        state_id, remote = row.get("state_id") or "unknown", row.get("checkpoint_path")
        if not remote:
            fetched.append(ProductFetchResult(kind="transfer", rc=1,
                                              detail=f"{state_id}: no path"))
            continue
        dest = store / state_id
        rc = subprocess.run(
            ["timeout", f"{getattr(ctx.args, 'ckpt_fetch_limit_min', 45)}m",
             "scp", "-r", "-P", str(ctx.target.port),
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             f"root@{ctx.host}:{remote}", str(dest)],
            capture_output=True).returncode
        verified, why = _reidentify(dest, row)
        fetched.append(ProductFetchResult(
            kind="transfer", rc=0 if rc == 0 and verified else 1,
            detail=f"{state_id}: rc={rc}, identity "
                   f"{'MATCHED' if verified else 'NOT MATCHED'} ({why})"))
    return fetched


def all_selected_leaves_secured(ctx, fetched) -> tuple[bool, str]:
    try:
        rows = committed_selection(ctx)
    except SelectionUnreadable as exc:
        return False, f"PRODUCT GATE FAILURE: {exc}"
    if not rows:
        return True, "the search did not reach commit; its evidence is the product"
    want = int(S.preregistration(REPO_ROOT)["retention"]["k"])
    if len(rows) != want:
        return False, (f"PRODUCT GATE FAILURE: {len(rows)} committed leaves, the "
                       f"preregistration commits {want}")
    ok = [f for f in fetched if getattr(f, "ok", False)]
    if len(ok) != want:
        return False, (f"{len(ok)} of {want} selected checkpoints secured and "
                       "identity-verified; the pod may not be deleted")
    return True, f"all {want} selected checkpoints secured and identity-verified"


# --- the $0 gates ------------------------------------------------------------

def run_identity_gate(ctx) -> tuple[bool, str]:
    try:
        ctx.auth.require_run_id(getattr(ctx.args, "run_id", ""))
        ctx.auth.require_plan(S.preregistration_hash(REPO_ROOT))
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    rows = getattr(ctx.args, "prediction_rows", "all_v1")
    if rows != ctx.auth.prediction_rows:
        return False, (f"the grant binds prediction_rows "
                       f"{ctx.auth.prediction_rows!r}, the launch declares {rows!r}")
    return True, (f"run {ctx.auth.run_id}, preregistration "
                  f"{ctx.auth.plan_hash[:12]}, rows {rows}")


def session_contract_gate(ctx) -> tuple[bool, str]:
    """Build the REAL session here and check it binds the granted identities."""
    import tempfile

    from stages.phase_d2.d2_authorization import FORMAL_DEVICE

    try:
        with tempfile.TemporaryDirectory(prefix="d2-gate-") as tmp:
            session = S.build_session(
                workdir=Path(tmp), run_id=ctx.auth.run_id, device=FORMAL_DEVICE,
                prediction_rows=ctx.auth.prediction_rows, repo_root=REPO_ROOT)
            contract = S.assert_session_contract(session, REPO_ROOT)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"the D2 session does not build or bind: {exc}"
    problems = [k for k in ("config_hash", "measurement_protocol_id",
                            "objective_hash", "ranking_policy_hash")
                if contract[k] != getattr(ctx.auth, k)]
    ctx.evidence["d2_session_contract_check"] = {
        **{k: contract[k] for k in ("config_hash", "measurement_protocol_id",
                                    "objective_hash", "ranking_policy_hash")},
        "problems": problems}
    if problems:
        return False, f"the tree's session is not the granted one: {problems}"
    return True, f"session contract verified at $0: config {contract['config_hash'][:12]}"


def staged_science_inputs_gate(ctx) -> tuple[bool, str]:
    """Each declared local asset exists and matches the pin it ships."""
    try:
        from aadistill.initialization.calibration.profiles import get_profile
        from shared.calibration import register_builtin_profiles

        D1S.verify_state_eval_bytes(REPO_ROOT)
        register_builtin_profiles()
        for pid in S.preregistration(REPO_ROOT)["search"]["profiles"]:
            get_profile(pid).resolve(REPO_ROOT)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"a science input does not verify: {exc}"
    missing = [a.repo_path for a in SCIENCE_ASSETS
               if not (REPO_ROOT / a.repo_path).exists()]
    if missing:
        return False, f"declared local assets absent: {missing}"
    return True, "state-eval asset and both calibration mixtures verified"


def bundle_staged_gate(ctx) -> tuple[bool, str]:
    """Can a pod obtain the exact granted code right now? Read-only."""
    import tempfile

    from aadistill.infrastructure.bundle_transport import hf_download, roundtrip

    commit = getattr(ctx.args, "session_commit", "")
    try:
        D1S.require_canonical_bundle(getattr(ctx.args, "bundle", ""), commit)
        rel = bundle_record_for(getattr(ctx.args, "run_id", None))
        record = json.loads((REPO_ROOT / rel).read_text())
        auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
        with tempfile.TemporaryDirectory(prefix="d2-bundle-") as tmp:
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


def prechecks(args) -> tuple:
    return (run_identity_gate, session_contract_gate, staged_science_inputs_gate,
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", None)),
                                check_lineage=True),
            bundle_staged_gate)


def budget_spec(repo_root: Path) -> BudgetSpec:
    """DECOMPOSES the priced bound so the plan lands exactly on it."""
    from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase

    from stages.phase_d1.d1_authorization import live_money
    from stages.phase_d2.d2_authorization import session_ceiling

    priced = session_ceiling(repo_root)
    overhead = float(priced["session_overhead_minutes"])
    accepted_hard = float(priced["hard_ceiling_minutes"])
    expected_total = float(priced["expected_minutes"])
    contingency, recovery_reserve = 0.10, 30.0
    reserve = round(accepted_hard - expected_total * (1.0 + contingency)
                    - recovery_reserve, 2)
    if reserve < 0:
        raise SystemExit(
            f"the priced {accepted_hard:.2f}-minute bound cannot hold the "
            f"expected {expected_total:.2f} min plus contingency and reserve")
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=MEASURED_STEP_SECONDS,
        step_source="unused: a D2 search trains nothing",
        setup_minutes=round(overhead * 0.75, 2),
        transfer_minutes=round(overhead * 0.25, 2),
        other_phases=(Phase("beam_search_expected",
                            round(max(0.0, expected_total - overhead), 2)),),
        contingency_fraction=contingency,
        soft_stop_reserves=(Phase("beam_composition_risk", reserve),),
        artifact_recovery_reserve_minutes=recovery_reserve,
        below_floor_reason=(f"priced by {priced['_basis']}; this spec "
                            "decomposes that bound and adds nothing to it"),
        account_balance_required_usd=round(max(
            float(live_money(repo_root)["per_session_envelope_usd"]),
            float(priced["hard_ceiling_usd"]) + ACCOUNT_OPERATIONAL_RESERVE_USD),
            4))


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-d2-search",
        schema="aadistill.autoinit.d2_search_session/v1",
        description=("D2: ONE formal beam search over D1's frozen space "
                     "minimizing the frozen-teacher-confidence-weighted root "
                     "KL (preregistration v2). Commits "
                     "the preregistered Top-K and stops."),
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        authorization_loader=D2Authorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=f"autoinit.v1.{S.EXPERIMENT_ID}",
        #: THE PREREGISTRATION HASH is the plan hash: the dispatch branch calls
        #: `require_plan` with it.
        plan_hash=S.preregistration_hash(REPO_ROOT),
        budget=budget_spec(REPO_ROOT),
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
            tests_max_seconds=getattr(args, "tests_max_s", 2700)),
        driver_command=driver_command,
        driver_job_id=S.DRIVER_JOB_ID,
        status_path=S.STATUS_PATH,
        run_log_path=RUN_LOG_PATH,
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note="the search stopped -- collecting its evidence"),
        precheck=prechecks(args),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="d2_search.json",
            archive_basename="d2_search_artifacts.tar.gz",
            spec_success="configs/stages/stage-1/phase_d2/d2_search_artifacts.json",
            spec_failed="configs/stages/stage-1/phase_d2/d2_search_artifacts_failed.json",
            report_names=("d2_search.json", "journal.jsonl",
                          "stage1_selection.json"),
            fetch_products=fetch_selected_checkpoints,
            products_secured=all_selected_leaves_secured),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Module scope, every required argument a STRING: the dispatch probe
    fills them with one."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--runpod-config",
                    default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--prediction-rows", default="all_v1",
                    choices=list(S.PREDICTION_ROWS))
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
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    #: MUST OUTLAST THE HARD THRESHOLD, or the launcher stops watching a pod
    #: still billing. Derived from the priced bound in `resolve_operational_
    #: defaults` when left at None.
    ap.add_argument("--poll-limit-min", type=float, default=None)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def resolve_operational_defaults(args) -> dict[str, Any]:
    """Price, disk, record path and poll limit from the priced session."""
    from stages.phase_d2.d2_authorization import session_ceiling

    priced = session_ceiling(REPO_ROOT)
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
                           summary=("D2 formal search: the committed Top-K and "
                                    "the search evidence are what it owes."))
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11


if __name__ == "__main__":
    raise SystemExit(main())
