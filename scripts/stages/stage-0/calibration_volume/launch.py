#!/usr/bin/env python3
"""Launch one Stage-0 calibration-volume session on a provider GPU.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/launch.py \
        --session battery --run-id <run_id> --scr <scratch> \
        --session-commit <sha> --bundle <name> --gpu "NVIDIA A100-SXM4-80GB" [--dry-run]

Reuses `aadistill.infrastructure.session_runner`: price before creation, the
authorization's plan / harness / cap checks before any pod exists, the detached
watchdog, the poll loop, artifact collection, teardown and provider
confirmation. This module supplies only what is this experiment's: the plan,
the inputs, the driver command and where completed work goes.

Completed work leaves the pod as it finishes: every `<label>.done` unit is
pulled on the next poll and hash-checked (P8.2.1), so a later failure, a lost
pod or a teardown cannot take a finished measurement with it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (ArtifactPolicy, BudgetSpec,  # noqa: E402
                                              ExecutionCommands, MarkerPolicy,
                                              RelayInput, SessionContext,
                                              SessionSpec, SetupManifest,
                                              TeardownPolicy)
from shared.deployment import MAIN_RELAY, deployment_commands  # noqa: E402
from stages.calibration_volume import governance as G  # noqa: E402

REPO = "/workspace/aad"
AUDIT = {"battery": "stage0_calvol_battery"}
EVIDENCE = {"battery": "battery.json"}


def _scp(ctx: SessionContext, remote: str, dest: Path, *, recursive=False,
         limit_min=10) -> int:
    cmd = ["timeout", f"{limit_min}m", "scp", *(["-r"] if recursive else []),
           "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", f"root@{ctx.host}:{remote}", str(dest)]
    return subprocess.run(cmd, capture_output=True).returncode


def store(ctx: SessionContext) -> Path:
    return Path(ctx.args.scr) / "store"


def _remote_done(ctx: SessionContext, kind: str) -> dict[str, str]:
    audit = f"{REPO}/artifacts/audit/{AUDIT[kind]}"
    res = ctx.target.run(f"cd {audit}/records 2>/dev/null && for f in *.done; do "
                         f"[ -e \"$f\" ] && echo \"$f $(cat $f)\"; done", timeout=120)
    out = {}
    for line in (getattr(res, "stdout", "") or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].endswith(".done"):
            out[parts[0][:-5]] = parts[1]
    return out


def secure_units(ctx: SessionContext, kind: str) -> None:
    """Pull each finished unit off the pod. Idempotent; never raises."""
    try:
        if ctx.target is None:
            return
        audit = f"{REPO}/artifacts/audit/{AUDIT[kind]}"
        secured = ctx.evidence.setdefault("units_secured", {})
        for label, digest in _remote_done(ctx, kind).items():
            if secured.get(label, {}).get("ok"):
                continue
            dest = store(ctx) / label
            dest.mkdir(parents=True, exist_ok=True)
            rcs = {}
            for name in (f"{label}.jsonl", f"{label}.summary.json", f"{label}.log"):
                rcs[name] = _scp(ctx, f"{audit}/records/{name}", dest / name,
                                 limit_min=30)
            for suffix in (".rows.jsonl", ".summary.json"):
                rcs[f"score{suffix}"] = _scp(ctx, f"{audit}/scores/{label}{suffix}",
                                             dest / f"score{suffix}")
            rec = dest / f"{label}.jsonl"
            ok = rec.is_file() and hashlib.sha256(rec.read_bytes()).hexdigest() == digest
            secured[label] = {"ok": ok, "sha256": digest, "rc": rcs}
            ctx.say(f"unit {label}: {'SECURED' if ok else 'fetch incomplete'}")
        _scp(ctx, f"{audit}/{EVIDENCE[kind]}", store(ctx) / EVIDENCE[kind])
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def units_secured(ctx: SessionContext, *_a, kind: str, **_k) -> tuple[bool, str]:
    secure_units(ctx, kind)
    secured = ctx.evidence.get("units_secured", {})
    remote = _remote_done(ctx, kind) if ctx.target is not None else {}
    missing = [l for l in remote if not secured.get(l, {}).get("ok")]
    if missing:
        return False, f"finished units not secured off-pod: {missing}"
    return True, f"{len(secured)} finished unit(s) secured off-pod"


def battery_relay_inputs(repo_root: Path = REPO_ROOT) -> tuple[RelayInput, ...]:
    plan = G.plan_doc("battery", repo_root)
    return tuple(RelayInput(path=f["path"], repo=MAIN_RELAY, dest=s["dir"],
                            sha256=f["sha256"])
                 for s in plan["students"] for f in s["relay_files"])


def battery_budget(args) -> BudgetSpec:
    """Priced from the plan's work, phase by phase; minutes are bounds, not hopes."""
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=0.0,
        step_source="unused: no training in this session",
        below_floor_reason="no training steps; every phase is priced below",
        setup_minutes=25.0, transfer_minutes=10.0,
        other_phases=(
            Phase("students_dev_3x", 30.0),
            Phase("teacher_dev_262k", 35.0),
            Phase("teacher_calibration_generation", 75.0),
            Phase("scoring", 5.0),
        ),
        contingency_fraction=0.25,
        soft_stop_reserves=(Phase("long_tail_generation", 20.0),),
        artifact_recovery_reserve_minutes=15.0,
        account_balance_required_usd=30.0)


def driver_command(ctx: Any, plan: Any, *, kind: str) -> str:
    run_id = ctx.args.run_id
    return (f"PYTHONPATH={REPO}/src:{REPO}/scripts /opt/train/bin/python "
            f"{REPO}/scripts/stages/stage-0/calibration_volume/{kind}_driver.py "
            f"--auth '{REPO}/{G.authorization_path(run_id)}' "
            f"--plan-hash '{G.plan_hash(kind)}' --status '{status_path(kind)}'")


def status_path(kind: str) -> str:
    return f"/workspace/stage0_calvol_{kind}.status"


def spec(args) -> SessionSpec:
    kind = args.session
    if kind != "battery":
        raise SystemExit(f"session kind {kind!r} is not built yet")
    return SessionSpec(
        session_id=f"stage0-calvol-{kind}",
        schema=f"aadistill.stage0.calibration_volume.{kind}_session/v1",
        description="Stage-0 calibration volume: dev-battery calibration and "
                    "teacher-native calibration-session generation",
        authorization_path=str(G.authorization_path(args.run_id)),
        authorization_loader=partial(G.load_for, kind),
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=f"stage0.calibration_volume.{kind}",
        plan_hash=G.plan_hash(kind),
        budget=battery_budget(args),
        setup=SetupManifest(
            env={"SESSION_KIND": "stage0_calvol"},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_KIND", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED", "TRAIN_ENV",
                           "VLLM_READY", "TEACHER_READY", "SETUP_DONE"),
            relay_inputs=battery_relay_inputs(),
            teacher_revision=G.plan_doc(kind)["teacher"]["revision"],
            uv_max_seconds=1500, tests_max_seconds=600),
        driver_command=partial(driver_command, kind=kind),
        driver_job_id=f"stage0_calvol_{kind}_driver",
        status_path=status_path(kind),
        run_log_path=f"/workspace/stage0_calvol_{kind}_run.log",
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note="every finished unit is secured off-pod before teardown"),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT[kind], evidence_filename=EVIDENCE[kind],
            archive_basename=f"stage0_calvol_{kind}",
            spec_success=f"configs/stages/stage-0/calibration_volume/{kind}_artifacts.json",
            spec_failed=f"configs/stages/stage-0/calibration_volume/{kind}_artifacts.json",
            report_names=(EVIDENCE[kind],),
            on_poll=partial(secure_units, kind=kind),
            products_secured=partial(units_secured, kind=kind)),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True, choices=G.SESSION_KINDS)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA A100-SXM4-80GB")
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=150)
    ap.add_argument("--token-src",
                    default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--runpod-config",
                    default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--country-code", default=None)
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
        args.out = str(REPO_ROOT / G.run_dir(args.run_id) / "runtime/session.json")
    if args.max_price is None:
        raise SystemExit("--max-price is required: read the live securePrice first")
    try:
        session = spec(args)
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11
    if args.dry_run:
        runner = SessionRunner(session, args, REPO_ROOT)
        ok = runner.make_plan()
        print(json.dumps({"plan_ok": ok, "plan": runner.ev.get("budget_plan"),
                          "harness": runner.harness["digest"],
                          "plan_hash": session.plan_hash}, indent=1)[:4000])
        return 0 if ok else 12
    return run_session(session, args, REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
