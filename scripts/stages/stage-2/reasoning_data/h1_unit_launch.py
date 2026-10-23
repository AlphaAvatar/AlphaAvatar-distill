#!/usr/bin/env python3
"""Launch one H1 unit session -- P1g's GPU validation or P2's formal units -- on one provider GPU.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/reasoning_data/h1_unit_launch.py \
        --phase {P1g,P2} --run-id <run_id> --scr <scratch> --session-commit <sha> --bundle <name> \
        --max-price <live price> --network-volume-id <id> --volume-mount-path /runpod-volume \
        [--unit A-s20260726 ...] [--measured <P1g report>] [--dry-run]

Reuses `aadistill.infrastructure.session_runner` exactly as the generation
launcher does: before any pod, the argument contract, the authorization loaded
under its kind's policy (`unit_governance`) and bound to the UNITS IDENTITY and
the harness on disk, the budget inside the authorized cap, the GPU + disk +
termination bound (`h1_generation_launch.all_in_within_cap`), the provider
reconciliation (the session's network volume is the one resource it expects)
and the prechecks; then the watchdog, setup, the driver, the poll loop,
collection, teardown and the provider's confirmation.

The durable root is the network volume (``<mount>/h1``): the driver persists
every checkpoint, save point and evaluation there as it completes, so a lost
pod costs no finished work. Each unit whose report is complete has its
EVIDENCE -- descriptors, report, evaluation generations and rows, training
records, attempts; never weights or save points (P8.4) -- tarred on the pod,
pulled on the next poll and hash-checked into ``<scr>/store``.

``--phase P1g`` (the ``validation`` kind) runs `p1g_driver.py`; ``--phase P2``
(``formal``) runs `unit_driver.py` for the named units (default: all four). The pod runs no repository suite (P8.2.1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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
from stages.reasoning_data import h1_generation_launch as GL  # noqa: E402
from stages.reasoning_data import unit_governance as UG  # noqa: E402

REPO = "/workspace/aad"
AUDIT = "stage2_h1_units"
SESSION_KIND = "stage2_h1_units"
STATUS = "/workspace/stage2_h1_units.status"
PACKS = "artifacts/stages/stage-2/reasoning_data/packs"
UNITS = ("A-s20260726", "A-s20260727", "B-s20260726", "B-s20260727")
#: what a unit's evidence leaves out: weights, save points, in-flight generation progress
EVIDENCE_EXCLUDES = ("./checkpoint", "./train", "./eval_progress")
#: P1g's measured basis for the formal budget, when no --measured record is given
P1G_DEFAULT_MINUTES = 45.0
#: setup on an A100 image with the teacher, vLLM and the initialization (P1: 4-6 min); a bound
SETUP_MINUTES = 20.0


#: phase -> the authorization kind that funds it (`unit_governance.KINDS`, inverted)
KIND_OF = {phase: kind for kind, phase in UG.KINDS.items()}


def kind(args) -> str:
    return KIND_OF[args.phase]


def store(ctx: SessionContext) -> Path:
    return Path(ctx.args.scr) / "store"


def durable_root(args) -> str:
    return f"{args.volume_mount_path.rstrip('/')}/h1"


def _scp(ctx: SessionContext, remote: str, dest: Path, *, limit_s: float = 1800) -> int:
    cmd = ["timeout", f"{int(limit_s)}s", "scp", "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
           "-o", "UserKnownHostsFile=/dev/null", f"root@{ctx.host}:{remote}", str(dest)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=limit_s + 30).returncode
    except subprocess.TimeoutExpired:
        return 124


def secure_units(ctx: SessionContext) -> None:
    """Each unit whose report is complete on the volume: its evidence tarred on
    the pod, pulled and hash-checked. Idempotent; never raises."""
    try:
        if ctx.target is None:
            return
        root, aud = durable_root(ctx.args), f"{REPO}/artifacts/audit/{AUDIT}"
        secured = ctx.evidence.setdefault("units_secured", {})
        names = list(UNITS) + (["validation/p1g", "validation/p1g-B"] if kind(ctx.args) == "validation" else [])
        excl = " ".join(f"--exclude={e}" for e in EVIDENCE_EXCLUDES)
        for name in names:
            if secured.get(name, {}).get("ok"):
                continue
            ready = "p1g_report.json" if name == "validation/p1g" else "report.descriptor.json"
            flat = name.replace("/", "__")
            res = ctx.target.run(
                f"test -f {root}/{name}/{ready} && mkdir -p {aud} && tar czf {aud}/{flat}.evidence.tgz {excl} "
                f"-C {root}/{name} . && sha256sum {aud}/{flat}.evidence.tgz | cut -d' ' -f1", timeout=600)
            digest = (getattr(res, "stdout", "") or "").strip().splitlines()
            if getattr(res, "returncode", 1) != 0 or not digest or len(digest[-1]) != 64:
                continue
            dest = store(ctx) / f"{flat}.evidence.tgz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            rc = _scp(ctx, f"{aud}/{flat}.evidence.tgz", dest)
            ok = dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest[-1]
            if ok:
                dest.with_suffix(".sha256").write_text(digest[-1] + "\n")
            secured[name] = {"ok": ok, "sha256": digest[-1], "rc": rc}
            ctx.say(f"unit {name}: evidence {'SECURED' if ok else 'fetch incomplete'}")
        _scp(ctx, STATUS, store(ctx) / "driver.status", limit_s=120)
        if kind(ctx.args) == "formal":
            _scp(ctx, f"{aud}/driver_status.json", store(ctx) / "driver_status.json", limit_s=120)
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def units_secured(ctx: SessionContext, *_a, **_k) -> tuple[bool, str]:
    secure_units(ctx)
    secured = ctx.evidence.get("units_secured", {})
    bad = [k for k, v in secured.items() if not v.get("ok")]
    if bad:
        return False, f"finished units whose evidence is not secured off-pod: {bad}"
    return True, f"{len(secured)} finished unit(s) secured off-pod; checkpoints are on the volume"


def measured_basis(path: str | None, repo_root: Path = REPO_ROOT) -> dict:
    """The formal budget's rates: P1g's measured step seconds and evaluation
    seconds per item, read from its report (committed under the P1g run)."""
    if not path:
        raise ValueError("--phase P2 needs --measured <P1g report>: the formal budget is priced from P1g")
    rep = json.loads((repo_root / path).read_text())
    if not rep.get("passed"):
        raise ValueError(f"{path}: P1g did not pass; no formal unit is priced from it")
    return rep


def unit_minutes(rep: dict, plan: dict, arm: str) -> tuple[float, float]:
    """(training, evaluation) minutes for one unit at the frozen schedule."""
    s = plan["schedule"]
    tp = rep["checks"][f"throughput_{arm}"]
    train = s["steps"] * tp["median_step_seconds"] / 60
    ev = rep["checks"]["evaluation_path"]
    per_item = max(ev["fluid"]["seconds"] / max(1, ev["fluid"]["items"]),
                   ev["capability"]["seconds"] / max(1, ev["capability"]["items"]))
    items = 720 + 846 + 150
    return round(train, 1), round(per_item * items / 60, 1)


def budget(args, repo_root: Path = REPO_ROOT) -> BudgetSpec:
    """P1g: a bounded validation session. Formal: the named units at P1g's
    measured rates. The project's hard-cap factor is the contingency (P1/P1t's
    rule); setup and teardown sit in the soft-stop reserve."""
    f = json.loads((repo_root / GL.FORECAST).read_text())["contingency_factors_not_measurements"]["hard_cap"]
    if kind(args) == "validation":
        phases = (Phase("p1g_validation", P1G_DEFAULT_MINUTES),)
    else:
        from stages.reasoning_data import unit_driver as UD
        plan = UD.load_plan()
        rep = measured_basis(args.measured, repo_root)
        phases = []
        for u in (args.unit or UNITS):
            t, e = unit_minutes(rep, plan, u.split("-")[0])
            phases += [Phase(f"{u}_train", t), Phase(f"{u}_evaluate", e)]
        phases = tuple(phases)
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=1.0,
        step_source="unused: training is priced per unit below from P1g's measured step seconds (arms=0)",
        below_floor_reason="no arm-level steps; every phase is priced below",
        setup_minutes=SETUP_MINUTES, transfer_minutes=10.0, other_phases=phases,
        contingency_fraction=round(f - 1.0, 4),
        soft_stop_reserves=(Phase("closeout_setup_and_teardown", round(f * (SETUP_MINUTES + 10.0), 2)),),
        artifact_recovery_reserve_minutes=round(f * GL.ARTIFACT_RECOVERY_MINUTES - GL.TERMINATION_MINUTES, 2),
        account_balance_required_usd=30.0,
        accounted_provider_resources=(args.network_volume_id,))


def driver_command(ctx: Any, plan: Any) -> str:
    a = ctx.args
    py = f"cd {REPO} && PYTHONPATH={REPO}/src:{REPO}/scripts:{REPO}/scripts/stages/stage-1 /opt/train/bin/python"
    root = durable_root(a)
    if kind(a) == "validation":
        return (f"mkdir -p {root} && {py} scripts/stages/stage-2/reasoning_data/p1g_driver.py --run-id {a.run_id} "
                f"--durable-root {root} --status {STATUS} --pack-root {REPO}/{PACKS}")
    units = " ".join(f"--unit {u}" for u in (a.unit or UNITS))
    aud = f"{REPO}/artifacts/audit/{AUDIT}"
    cap = f"$(( {int(a.volume_gb) * 10**9} - $(du -sb {a.volume_mount_path} | cut -f1) ))"
    return (f"mkdir -p {root} {aud} && {py} scripts/stages/stage-2/reasoning_data/unit_driver.py "
            f"--run-id {a.run_id} --durable-root {root} --pack-root {REPO}/{PACKS} {units} "
            f"--capacity-bytes {cap} --status {aud}/driver_status.json; rc=$?; "
            f"if [ $rc = 0 ]; then m=ALL_DONE; else m=RUN_FAILED; fi; "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:$m\" >> {STATUS}")


def the_artifact_funds_this_phase(ctx: SessionContext) -> tuple[bool, str]:
    """The type loads any unit artifact under ITS declared kind's policy; this
    launch runs one phase, and only that phase's kind may fund it."""
    got = (ctx.auth.stage_conditions or {}).get("kind")
    if got != kind(ctx.args):
        return False, f"a {got} authorization cannot fund a {ctx.args.phase} session"
    return True, f"a {got} authorization for {ctx.args.phase}"


def units_identity_is_the_authorized_one(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    got = UG.units_hash(repo_root)
    if got != ctx.auth.plan_hash:
        return False, f"the units identity on disk is {got[:12]}, not the authorized {ctx.auth.plan_hash[:12]}"
    return True, f"units identity {got[:12]} is the authorized one"


def bound_files_are_in_the_session_commit(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    """The pod checks out the session commit: every bound file must be in it with
    the bytes the units identity hashed (the packs travel as local assets)."""
    bad = []
    for f in UG.units_identity(repo_root)["files"]:
        blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{f['path']}"], cwd=repo_root,
                              capture_output=True)
        if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != f["sha256"]:
            bad.append(f["path"])
    return (not bad, "every bound file is in the session commit" if not bad else
            f"not in the session commit as bound: {bad[:4]}")


def spec(args) -> SessionSpec:
    k = kind(args)
    proto = json.loads((REPO_ROOT / "configs/stages/stage-3/e1/e1_r0860k_sa_pca.json").read_text())
    from stages.phase_c1.autoinit_c1_launch import C1_EVAL_TOKENIZER, C1_ROPE_INPUT
    assets = (LocalAsset(repo_path=f"{PACKS}/A", dest_name="A", install_to=PACKS),
              LocalAsset(repo_path=f"{PACKS}/B_v3", dest_name="B", install_to=PACKS),
              LocalAsset(repo_path="artifacts/stages/stage-3/eval/battery_v2", dest_name="battery_v2",
                         install_to="artifacts/stages/stage-3/eval"))
    return SessionSpec(
        session_id=f"stage2-h1-units-{k}",
        schema="aadistill.stage2.reasoning_data.h1_unit_session/v1",
        description=f"H1 {'P1g GPU validation' if k == 'validation' else 'P2 formal units'}",
        authorization_path=str(UG.authorization_path(args.run_id)),
        authorization_loader=UG.H1UnitAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id="stage2.reasoning_data.h1_units",
        plan_hash=UG.units_hash(REPO_ROOT),
        budget=budget(args),
        setup=SetupManifest(
            env={"SESSION_KIND": SESSION_KIND},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS", "SESSION_KIND", "TEACHER_REVISION"),
            #: no TESTS_OK: no repository suite on the paid pod (P8.2.1)
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED", "TRAIN_ENV", "VLLM_READY",
                           "TEACHER_READY", "AUTHORIZATION_OK", "SETUP_DONE"),
            local_assets=assets, relay_inputs=(*C1_EVAL_TOKENIZER, *C1_ROPE_INPUT),
            teacher_revision=proto["teacher"]["revision"],
            uv_max_seconds=1500, tests_max_seconds=600),
        driver_command=driver_command,
        driver_job_id=f"stage2_h1_units_{k}_driver",
        status_path=STATUS,
        run_log_path="/workspace/stage2_h1_units_run.log",
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note="every finished unit is on the network volume and its evidence off-pod"),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT, evidence_filename="driver_status.json", archive_basename=f"stage2_h1_units_{k}",
            spec_success="configs/stages/stage-2/reasoning_data/h1_unit_artifacts.json",
            spec_failed="configs/stages/stage-2/reasoning_data/h1_unit_artifacts.json",
            report_names=(), on_poll=secure_units, products_secured=units_secured),
        teardown=TeardownPolicy(note="delete the pod, verify from the provider that it is gone, STOP; "
                                     "the network volume is KEPT (it holds the units' checkpoints)"),
        precheck=(the_artifact_funds_this_phase, GL.all_in_within_cap, units_identity_is_the_authorized_one,
                  bound_files_are_in_the_session_commit),
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=sorted(KIND_OF),
                    help="P1g (the validation kind) or P2 (formal units): the grant's phase")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--unit", action="append", default=None, choices=UNITS)
    ap.add_argument("--measured", default=None, help="P1g's report (formal: the budget's rates)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA A100-SXM4-80GB")
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=150)
    ap.add_argument("--volume-gb", type=int, default=30)
    ap.add_argument("--token-src", default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--runpod-config", default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--country-code", default=None)
    #: required: a unit session's checkpoints, save points and evaluations must survive the pod
    ap.add_argument("--network-volume-id", required=True)
    ap.add_argument("--volume-mount-path", required=True)
    ap.add_argument("--data-center-ids", default="")
    ap.add_argument("--image-digest", default=None)
    ap.add_argument("--setup-timeout-s", type=float, default=3600.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=1440.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.session_runner import SessionRunner, run_session

    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = str(UG.run_dir(args.run_id) / "runtime/session.json")
    if args.max_price is None:
        raise SystemExit("--max-price is required: read the live price first")
    try:
        session = spec(args)
        runner = SessionRunner(session, args, REPO_ROOT) if args.dry_run else None
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11
    ok, why = GL.bundle_on_relay(args)
    print(f"{'bundle OK' if ok else 'REFUSED at $0'}: {why}")
    if not ok and not args.dry_run:
        print("No provider resource was created.")
        return 13
    if args.dry_run:
        ok = runner.make_plan()
        bound = GL.all_in(runner.plan, args.max_price, args.disk_gb, runner.auth.hard_cap_usd) if ok else None
        ok = ok and bound["all_in_usd"] <= runner.auth.hard_cap_usd
        print(json.dumps({"plan_ok": ok, "plan": runner.ev.get("budget_plan"), "all_in": bound,
                          "harness": runner.harness["digest"], "units_identity": session.plan_hash}, indent=1)[:5000])
        return 0 if ok else 12
    return run_session(session, args, REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
