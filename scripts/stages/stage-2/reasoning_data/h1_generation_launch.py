#!/usr/bin/env python3
"""Launch H1's P1 teacher-generation session on one provider GPU (needs an issued authorization).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/reasoning_data/h1_generation_launch.py \
        --run-id <run_id> --scr <scratch> --session-commit <sha> --bundle <name> \
        --max-price <live securePrice> [--resume-store <dir>] [--dry-run]

Reuses `aadistill.infrastructure.session_runner`: before any pod exists, the
argument contract, the authorization loaded under `governance.POLICY` and bound
to this LAUNCH IDENTITY (plan, protocol, inputs and pools: `governance.
launch_hash`) and to the harness on disk, the budget priced from the forecast
inside the authorized cap, the provider-resource reconciliation and the
prechecks below; then the detached watchdog (the hard deadline derived from the
cap and the live price), setup, the driver, the poll loop, collection, teardown
and the provider's confirmation that the pod is gone. This module supplies only
what is H1's: the plan identity, the budget rows, the driver command and where
finished work goes.

Finished work leaves the pod while the session runs: every shard the driver
seals (``pass*.partNNNN.jsonl`` with its ``.done`` sha256) and every group's
``.complete`` marker is pulled on the next poll and hash-checked into
``<scr>/store``, so a later failure, a lost pod or a teardown cannot take a
finished candidate with it. ``--resume-store`` stages a previous session's
secured shards so the driver generates only what is missing.

The pod runs no repository test suite (AGENTS.md P8.2.1): the driver's own
contract -- launch identity, harness and sampling, re-checked on the pod -- is
this experiment's preflight. `closeout` writes the run's outcome on the dev box
after the session, including a fresh provider reconciliation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (ArtifactPolicy, BudgetSpec,  # noqa: E402
                                              ExecutionCommands, LocalAsset, MarkerPolicy,
                                              SessionContext, SessionSpec, SetupManifest,
                                              TeardownPolicy)
from shared.deployment import deployment_commands  # noqa: E402
from stages.reasoning_data import governance as GOV  # noqa: E402

REPO = "/workspace/aad"
AUDIT = "stage2_h1_generation"
EVIDENCE = "generation_session.json"
SESSION_KIND = "stage2_h1_generation"
STATUS = "/workspace/stage2_h1_generation.status"
FORECAST = GOV.EXPERIMENT_LOGS / "plans/cost_forecast.json"
RESUME_INSTALL = "artifacts/stages/stage-2/reasoning_data"


def out_dir() -> str:
    return f"{REPO}/artifacts/audit/{AUDIT}"


def _scp(ctx: SessionContext, remote: str, dest: Path, *, limit_s: float = 1200) -> int:
    """One bounded copy: `timeout` kills scp at `limit_s`, and the wait itself
    is bounded too, so no copy outlives the time it was given."""
    limit = max(1, int(limit_s))
    cmd = ["timeout", f"{limit}s", "scp", "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", f"root@{ctx.host}:{remote}", str(dest)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=limit + 30).returncode
    except subprocess.TimeoutExpired:
        return 124


def store(ctx: SessionContext) -> Path:
    return Path(ctx.args.scr) / "store"


#: The listing proves itself: a sentinel the shell prints only after the loop
#: finished, so an empty stdout from a failed or cut-off ssh is never read as
#: "the pod holds no shards".
LISTED = "__H1_SHARD_LISTING_COMPLETE__"
#: Bounded retries of the secured check, well inside the budget's artifact
#: recovery reserve (`budget().artifact_recovery_reserve_minutes`); after them
#: an unverifiable state is RECORDED and returned as not secured -- it never
#: holds the session open: the runner records PRODUCTS_UNSECURED and the
#: watchdog's hard deadline still applies.
SECURE_ATTEMPTS = 3
SECURE_RETRY_SECONDS = 30.0
#: The whole secured check, every fetch included, ends within this (inside the
#: 15-minute artifact recovery reserve the budget prices).
SECURE_BUDGET_SECONDS = 600.0
#: below this many seconds left, an operation is not started (recorded as skipped)
MIN_OPERATION_SECONDS = 2.0
_now = time.monotonic


class ListingUnknown(RuntimeError):
    """The pod's shard listing could not be obtained: the state is UNKNOWN."""


def remote_markers(ctx: SessionContext, *, timeout: float = 120) -> dict[str, str]:
    """Every sealed shard's marker on the pod: shard name -> its sha256.

    Returns ``{}`` only when the listing SUCCEEDED and found none (the output
    directory does not exist yet or holds no marker). Raises `ListingUnknown`
    when there is no target, the command timed out or failed, the sentinel is
    missing, or a line is not ``<name>.done <sha256>``."""
    if ctx.target is None:
        raise ListingUnknown("no target to ask")
    res = ctx.target.run(f"if [ -d {out_dir()} ]; then cd {out_dir()} && for f in pass*.part*.done; do "
                         f"[ -e \"$f\" ] && echo \"$f $(cat \"$f\")\"; done; fi; echo {LISTED}",
                         timeout=max(1.0, timeout))
    if getattr(res, "timed_out", False):
        raise ListingUnknown("the listing timed out")
    if getattr(res, "returncode", 1) != 0:
        raise ListingUnknown(f"the listing exited {getattr(res, 'returncode', None)}")
    lines = (getattr(res, "stdout", "") or "").splitlines()
    if not lines or lines[-1].strip() != LISTED:
        raise ListingUnknown("the listing did not complete (no sentinel)")
    out = {}
    for line in lines[:-1]:
        parts = line.split()
        if (len(parts) != 2 or not parts[0].endswith(".done") or len(parts[1]) != 64
                or any(ch not in "0123456789abcdef" for ch in parts[1])):
            raise ListingUnknown(f"unreadable listing line {line[:80]!r}")
        out[parts[0][:-5] + ".jsonl"] = parts[1]
    return out


def _local_ok(path: Path, digest: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest


def secure_shards(ctx: SessionContext, remote: dict[str, str] | None = None, *,
                  deadline: float | None = None) -> None:
    """Pull each sealed shard (and each group's .complete and the driver's
    record) off the pod and check the LOCAL bytes against the marker.
    Idempotent; never raises; an unobtainable listing is recorded, not read as
    an empty one. EVERY remote operation -- listings and copies alike -- is
    bounded by the time left to `deadline` (its own `SECURE_BUDGET_SECONDS`
    when called from the poll loop); what the deadline leaves undone is
    recorded in ``secure_deadline_skipped`` and verified shards are kept."""
    if deadline is None:
        deadline = _now() + SECURE_BUDGET_SECONDS
    skipped = ctx.evidence.setdefault("secure_deadline_skipped", [])

    def left() -> float:
        return deadline - _now()
    try:
        if remote is None:
            if left() < MIN_OPERATION_SECONDS:
                skipped.append("listing")
                return
            try:
                remote = remote_markers(ctx, timeout=min(120.0, left()))
            except ListingUnknown as exc:
                ctx.evidence.setdefault("listing_unknown", []).append(str(exc))
                return
        dest = store(ctx)
        dest.mkdir(parents=True, exist_ok=True)
        secured = ctx.evidence.setdefault("shards_secured", {})
        for name, digest in remote.items():
            if _local_ok(dest / name, digest):
                secured[name] = {"ok": True, "sha256": digest}
                continue
            if left() < MIN_OPERATION_SECONDS:
                secured[name] = {"ok": False, "sha256": digest, "rc": None, "why": "check deadline passed"}
                skipped.append(name)
                continue
            rc = _scp(ctx, f"{out_dir()}/{name}", dest / name, limit_s=min(1200.0, left()))
            ok = _local_ok(dest / name, digest)
            if ok:
                (dest / name).with_suffix(".done").write_text(digest + "\n")
            secured[name] = {"ok": ok, "sha256": digest, "rc": rc}
            ctx.say(f"shard {name}: {'SECURED' if ok else 'fetch incomplete'}")
        if left() < MIN_OPERATION_SECONDS:
            skipped.append("complete-listing")
        else:
            res = ctx.target.run(f"cd {out_dir()} 2>/dev/null && ls *.complete 2>/dev/null || true",
                                 timeout=min(120.0, left()))
            for name in (getattr(res, "stdout", "") or "").split():
                if left() < MIN_OPERATION_SECONDS:
                    skipped.append(name)
                    continue
                _scp(ctx, f"{out_dir()}/{name}", dest / name, limit_s=min(1200.0, left()))
        if left() < MIN_OPERATION_SECONDS:
            skipped.append(EVIDENCE)
        else:
            _scp(ctx, f"{out_dir()}/{EVIDENCE}", dest / EVIDENCE, limit_s=min(1200.0, left()))
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def shards_secured(ctx: SessionContext, *_a, sleep=time.sleep, **_k) -> tuple[bool, str]:
    """Three answers, never a fourth: the listing succeeded and the pod holds
    no shard; every listed shard is in the local store with the bytes its
    marker names (re-hashed now, not taken from an in-memory flag); or the state
    is not established -- the listing failed or a listed shard is not verified
    locally -- after `SECURE_ATTEMPTS` bounded tries, which is NOT secured."""
    why = "not checked"
    deadline = _now() + SECURE_BUDGET_SECONDS
    for attempt in range(SECURE_ATTEMPTS):
        if deadline - _now() < MIN_OPERATION_SECONDS:
            why = f"UNVERIFIED: the {SECURE_BUDGET_SECONDS:.0f} s check deadline passed ({why})"
            break
        try:
            remote = remote_markers(ctx, timeout=min(120.0, deadline - _now()))
        except ListingUnknown as exc:
            why = f"UNKNOWN: {exc}"
        else:
            secure_shards(ctx, remote, deadline=deadline)
            missing = sorted(n for n, d in remote.items() if not _local_ok(store(ctx) / n, d))
            if not missing:
                return True, ("the listing succeeded and the pod holds no sealed shard" if not remote else
                              f"{len(remote)} sealed shard(s) secured off-pod, each re-hashed in the local store")
            why = f"UNVERIFIED: {len(missing)} listed shard(s) not in the local store as marked: {missing[:5]}"
        if attempt + 1 < SECURE_ATTEMPTS and _now() + SECURE_RETRY_SECONDS < deadline:
            sleep(SECURE_RETRY_SECONDS)
        else:
            break
    ctx.evidence["shards_unverified"] = {"why": why, "attempts": SECURE_ATTEMPTS,
                                         "note": "recorded as not secured; teardown is not held for it"}
    return False, why


def budget(repo_root: Path = REPO_ROOT) -> BudgetSpec:
    """Priced from the forecast's generation rows (measured throughput); the
    minutes are bounds, and the hard deadline follows from the cap and the
    live price (the watchdog)."""
    fc = json.loads((repo_root / FORECAST).read_text())
    gen, factor = fc["generation"], fc["contingency_factors_not_measurements"]["generation"]
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=1.0,
        step_source="unused: no training in this session (arms=0)",
        below_floor_reason="no training steps; every phase is priced below",
        setup_minutes=25.0, transfer_minutes=10.0,
        other_phases=(Phase("teacher_generation_two_passes", round(gen["gpu_hours"] * 60, 1)),),
        contingency_fraction=round(factor - 1.0, 4),
        soft_stop_reserves=(Phase("long_tail_generation", 20.0),),
        artifact_recovery_reserve_minutes=15.0, account_balance_required_usd=30.0)


def driver_command(ctx: Any, plan: Any) -> str:
    py = f"PYTHONPATH={REPO}/src:{REPO}/scripts /opt/train/bin/python"
    resume = (f" --resume-from '{REPO}/{RESUME_INSTALL}/resume_store'"
              if getattr(ctx.args, "resume_store", None) else "")
    return (f"cd {REPO} && {py} -m stages.reasoning_data.generation_driver "
            f"--authorization '{REPO}/{GOV.authorization_path(ctx.args.run_id)}' "
            f"--status '{STATUS}' --out '{out_dir()}'{resume}")


def plan_is_the_chosen_admitted_one(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    plan = json.loads((repo_root / GOV.PLAN).read_text())
    if not (plan.get("admitted") and plan.get("corpus_id") == json.loads(
            (repo_root / "configs/stages/stage-2/reasoning_data/corpus_spec.json").read_text())["chosen_variant"]["id"]):
        return False, f"{GOV.PLAN} is not the chosen, admitted variant's plan"
    if plan.get("shortfalls"):
        return False, f"the plan has shortfalls {plan['shortfalls']}"
    return True, f"{plan['corpus_id']}: {plan['totals']['tasks']} tasks, no shortfall"


def bound_files_are_in_the_session_commit(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    """The pod checks out the session commit: every bound file must be in it
    with the bytes the launch identity hashed."""
    ident = GOV.launch_identity(repo_root)
    files = [ident["generation_plan"], ident["generation_protocol"], *ident["inputs"],
             *[{"path": p["path"], "sha256": p["sha256"]} for p in ident["pools"]]]
    bad = []
    for f in files:
        path = f.get("path") or f["file"]
        blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{path}"], cwd=repo_root,
                              capture_output=True)
        if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != f["sha256"]:
            bad.append(path)
    return (not bad, "every bound file is in the session commit" if not bad else
            f"not in the session commit as bound: {bad[:4]}")


def spec(args) -> SessionSpec:
    proto = json.loads((REPO_ROOT / GOV.PROTOCOL).read_text())
    #: The run inputs reach the pod by a DECLARED route (the core rule: a pod
    #: has only its bundle plus what the session names). They are also
    #: committed, for review; setup installs identical bytes over them, and the
    #: driver re-hashes them into the launch identity before reading them.
    assets = (LocalAsset(repo_path=str(GOV.INPUTS_DIR), dest_name=GOV.INPUTS_DIR.name,
                         install_to=str(GOV.INPUTS_DIR.parent)),)
    if getattr(args, "resume_store", None):
        assets += (LocalAsset(repo_path=str(args.resume_store), dest_name="resume_store",
                              install_to=RESUME_INSTALL),)
    return SessionSpec(
        session_id="stage2-h1-generation",
        schema="aadistill.stage2.reasoning_data.h1_generation_session/v1",
        description="H1 P1: teacher-native candidates for arm B's generated cells (text only)",
        authorization_path=str(GOV.authorization_path(args.run_id)),
        authorization_loader=GOV.H1GenerationAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id="stage2.reasoning_data.h1_generation",
        plan_hash=GOV.launch_hash(REPO_ROOT),
        budget=budget(),
        setup=SetupManifest(
            env={"SESSION_KIND": SESSION_KIND},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS", "SESSION_KIND", "TEACHER_REVISION"),
            #: no TESTS_OK: no repository suite on the paid pod (P8.2.1).
            #: AUTHORIZATION_OK: setup's stage2_h1_generation branch loads the
            #: artifact by its own type and binds it to the launch identity and
            #: harness recomputed on the pod, before the driver (which checks again)
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED", "TRAIN_ENV", "VLLM_READY",
                           "TEACHER_READY", "AUTHORIZATION_OK", "SETUP_DONE"),
            local_assets=assets, teacher_revision=proto["teacher"]["revision"],
            uv_max_seconds=1500, tests_max_seconds=600),
        driver_command=driver_command,
        driver_job_id="stage2_h1_generation_driver",
        status_path=STATUS,
        run_log_path="/workspace/stage2_h1_generation_run.log",
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note="every sealed shard is secured off-pod before teardown"),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT, evidence_filename=EVIDENCE, archive_basename="stage2_h1_generation",
            spec_success="configs/stages/stage-2/reasoning_data/h1_generation_artifacts.json",
            spec_failed="configs/stages/stage-2/reasoning_data/h1_generation_artifacts.json",
            report_names=(EVIDENCE,), on_poll=secure_shards, products_secured=shards_secured),
        teardown=TeardownPolicy(note="delete the pod, verify from the provider that it is gone, STOP"),
        precheck=(plan_is_the_chosen_admitted_one, bound_files_are_in_the_session_commit),
    )


#: The container-disk basis every Stage-0 closeout priced with
#: (`stages.calibration_volume.closeout.DISK_USD_PER_GB_MONTH`).
DISK_USD_PER_GB_MONTH = 0.10


def closeout(run_id: str, scr: Path, *, repo_root: Path = REPO_ROOT, reconcile=None, disk_gb: float = 150.0) -> dict:
    """The run's outcome on the dev box: every secured shard re-hashed against
    its marker, group completeness from the .complete markers, money and the
    provider's teardown confirmation from the session record, and a fresh
    provider reconciliation. Writes ``runs/<run_id>/closeout/outcome.json``."""
    store_dir = Path(scr) / "store"
    shards = sorted(store_dir.glob("pass*.part*.jsonl"))
    bad = [s.name for s in shards
           if not s.with_suffix(".done").is_file()
           or hashlib.sha256(s.read_bytes()).hexdigest() != s.with_suffix(".done").read_text().strip()]
    complete = {}
    for c in sorted(store_dir.glob("*.complete")):
        doc = json.loads(c.read_text())
        complete[doc["label"]] = all((store_dir / s["name"]).is_file() and hashlib.sha256(
            (store_dir / s["name"]).read_bytes()).hexdigest() == s["sha256"] for s in doc["shards"])
    session = repo_root / GOV.run_dir(run_id) / "runtime/session.json"
    sess = json.loads(session.read_text()) if session.is_file() else {}
    if reconcile is None:
        from aadistill.infrastructure.provider import RunPodProvider, read_api_key, reconcile_account
        reconcile = lambda: reconcile_account(  # noqa: E731
            RunPodProvider(read_api_key(str(Path("~/.runpod/config.toml").expanduser()))), ())
    rec = reconcile()
    driver = json.loads((store_dir / EVIDENCE).read_text()) if (store_dir / EVIDENCE).is_file() else {}
    cost = sess.get("cost") or {}
    gpu = float(cost.get("actual_usd") or 0.0)
    minutes = float(cost.get("elapsed_minutes") or 0.0)
    disk = disk_gb * DISK_USD_PER_GB_MONTH * minutes / (60 * 24 * 30.4375)
    complete_run = (bool(complete) and all(complete.values()) and not bad and driver.get("terminal") == "ALL_DONE")
    outcome = {"schema": "aadistill.stage2.reasoning_data.h1_generation_outcome/v1", "run_id": run_id,
               "status": "COMPLETE" if complete_run else "INCOMPLETE", "terminal": driver.get("terminal"),
               "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "driver_terminal": driver.get("terminal"), "session_passed": sess.get("passed"),
               "shards": {"secured": len(shards), "unverified": bad,
                          "records": sum(sum(1 for _ in open(s)) for s in shards if s.name not in bad)},
               "groups_complete": complete,
               "complete": complete_run,
               "money": {"all_in_usd": math.ceil((gpu + disk) * 1e4) / 1e4, "gpu_usd": gpu, "disk_usd": round(disk, 4),
                         "elapsed_minutes": minutes, "price_per_hour": cost.get("price_per_hour"),
                         "authorized_usd": cost.get("authorized_usd"),
                         "_basis": ("GPU actual from runtime/session.json :: cost plus container disk at "
                                    f"${DISK_USD_PER_GB_MONTH}/GB-month over the elapsed minutes ({disk_gb:g} GB), "
                                    "ceiled to the 4-decimal quantum -- the Stage-0 closeouts' basis")},
               "cost": cost, "provider_confirms_gone": sess.get("provider_confirms_gone"),
               "provider_reconciliation_after": rec,
               "authorizes": "nothing"}
    out = repo_root / GOV.run_dir(run_id) / "closeout/outcome.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outcome, indent=1) + "\n")
    return outcome


def bundle_on_relay(args, api=None) -> tuple[bool, str]:
    """Can the pod obtain the session commit? Asked at $0 before any pod: the
    setup clones the bundle from the relay (`stages.phase_c1.bundle`)."""
    from stages.phase_c1 import bundle as B
    try:
        B.require_canonical_bundle_arg(args.bundle, args.session_commit)
    except B.C1BundleError as exc:
        return False, str(exc)
    path = B.canonical_repo_path(args.session_commit)
    try:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        found = list(api.get_paths_info(B.RELAY_REPO, [path], repo_type="model"))
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot ask the relay whether it holds {path}: {exc!r}"
    return (bool(found), f"{path} on the relay" if found else f"the relay holds no {path}; stage it first")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--resume-store", default=None, help="a previous session's secured shards to stage")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA A100-SXM4-80GB")
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=150)
    ap.add_argument("--token-src", default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--runpod-config", default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--country-code", default=None)
    ap.add_argument("--network-volume-id", default="", help="none for P1: its records fit existing storage")
    ap.add_argument("--volume-mount-path", default="")
    ap.add_argument("--data-center-ids", default="")
    ap.add_argument("--image-digest", default=None)
    ap.add_argument("--setup-timeout-s", type=float, default=3600.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=420.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.session_runner import SessionRunner, run_session

    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = str(GOV.run_dir(args.run_id) / "runtime/session.json")
    if args.max_price is None:
        raise SystemExit("--max-price is required: read the live securePrice first")
    if args.network_volume_id:
        raise SystemExit("P1 attaches no network volume: its records fit existing storage")
    try:
        session = spec(args)
        runner = SessionRunner(session, args, REPO_ROOT) if args.dry_run else None
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11
    ok, why = bundle_on_relay(args)
    print(f"{'bundle OK' if ok else 'REFUSED at $0'}: {why}")
    if not ok and not args.dry_run:
        print("No provider resource was created.")
        return 13
    if args.dry_run:
        ok = runner.make_plan()
        print(json.dumps({"plan_ok": ok, "plan": runner.ev.get("budget_plan"),
                          "harness": runner.harness["digest"], "launch_identity": session.plan_hash}, indent=1)[:4000])
        return 0 if ok else 12
    rc = run_session(session, args, REPO_ROOT)
    print(json.dumps(closeout(args.run_id, Path(args.scr), disk_gb=args.disk_gb)["shards"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
