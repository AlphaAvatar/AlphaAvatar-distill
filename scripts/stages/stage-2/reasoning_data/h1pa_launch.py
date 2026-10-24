#!/usr/bin/env python3
"""Launch one H1-PA student session -- PA2v's GPU validation or PA2's formal units -- on one provider GPU.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/reasoning_data/h1pa_launch.py \
        --phase {PA2v,PA2} --run-id <run_id> --scr <scratch> --session-commit <sha> --bundle <name> \
        --max-price <live price> --network-volume-id <id> --volume-mount-path /runpod-volume \
        --data-center-ids <the volume's datacenter> [--unit D4-Original-s20260726 ...] \
        [--step-seconds <PA2v's measured>] [--dry-run]

`h1_unit_launch`'s session, with H1-PA's governance, driver and inputs:
`aadistill.infrastructure.session_runner` runs the argument contract, the
authorization loaded under its kind's policy (`h1pa_governance`) and bound to
the STUDY IDENTITY and the harness on disk, the budget inside the authorized
cap, the GPU + disk + termination bound (`h1_generation_launch.all_in_within_cap`:
the watchdog's deadline follows the accepted price), the provider
reconciliation (the study's network volume is the one resource it expects)
and the prechecks; then the watchdog, setup, the driver, the poll loop,
collection, teardown and the provider's confirmation.

Inputs follow their consumers (P8.4): the pod pulls, hash-pinned from the
relay, only what the session's units read -- E1's pack for D4-Original, the
merged set for D4-Augmented, E1's checkpoint for each PCA-Original seed it
evaluates. D4 Top-1 is restored by the driver itself (`restore_initialization`).
Nothing heavy crosses the dev box's uplink.

The durable root is the network volume (``<mount>/h1pa``): every checkpoint,
save point and evaluation is persisted there as it completes, so a lost pod
costs no finished work; each unit whose report is complete has its EVIDENCE
(never weights or save points) tarred, pulled on the next poll and
hash-checked into ``<scr>/store``. The pod runs no repository suite (P8.2.1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (ArtifactPolicy, BudgetSpec,  # noqa: E402
                                              ExecutionCommands, LocalAsset, MarkerPolicy, RelayInput,
                                              SessionContext, SessionSpec, SetupManifest, TeardownPolicy)
from shared.deployment import MAIN_RELAY, deployment_commands  # noqa: E402
from stages.reasoning_data import h1_generation_launch as GL  # noqa: E402
from stages.reasoning_data import h1_unit_launch as HUL  # noqa: E402
from stages.reasoning_data import h1pa_governance as HG  # noqa: E402

REPO = "/workspace/aad"
AUDIT = "stage2_h1pa_units"
SESSION_KIND = "stage2_h1pa"       #: setup's authorization branch (autoinit_preflight_setup.sh)
STATUS = "/workspace/stage2_h1pa_units.status"
PACKS = "artifacts/stages/stage-2/reasoning_data/h1pa/packs"
CHECKPOINTS = "artifacts/stages/stage-2/reasoning_data/h1pa/pca_original"
STUDY = REPO_ROOT / HG.STUDY
KIND_OF = {phase: kind for kind, phase in HG.KINDS.items()}
#: the GPU validation: one training unit for a few steps, one evaluation-only unit, each on a subset
VALIDATION_UNITS = ("PCA-Original-s20260726", "D4-Augmented-s20260726")
VALIDATION_STEPS = 40
VALIDATION_ITEMS = 24
VALIDATION_MINUTES = 60.0
#: setup on an L40S image with the teacher, vLLM and the inputs (H1 P1g/P2: under 20 min); a bound
SETUP_MINUTES = 20.0
CHECKPOINT_FILES = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
                    "tokenizer_config.json", "chat_template.jinja")


def study() -> dict:
    return json.loads(STUDY.read_text())


def all_units() -> tuple[str, ...]:
    st = study()
    return tuple(f"{arm}-s{seed}" for arm in st["arms"] for seed in st["seeds"])


def kind(args) -> str:
    return KIND_OF[args.phase]


def units_of(args) -> tuple[str, ...]:
    return tuple(args.unit or (VALIDATION_UNITS if kind(args) == "validation" else all_units()))


def durable_root(args) -> str:
    return f"{args.volume_mount_path.rstrip('/')}/h1pa"


def relay_inputs(units: tuple[str, ...], st: dict, relay_files: dict) -> tuple[RelayInput, ...]:
    """What these units read, and nothing else: each trained arm's pack, each
    evaluated PCA-Original seed's checkpoint. `relay_files` is the hash-pinned
    file list of each source (`h1pa.json :: relay_inputs`), never re-typed here."""
    arms = {u.rsplit("-s", 1)[0] for u in units}
    seeds = sorted({u.rsplit("-s", 1)[1] for u in units if u.startswith("PCA-Original")})
    out = []
    for arm in ("D4-Original", "D4-Augmented"):
        if arm in arms:
            key = st["arms"][arm]["dataset"]
            out += [RelayInput(f"{relay_files['packs'][key]['prefix']}/{n}", repo=MAIN_RELAY,
                               dest=f"{PACKS}/{key}", sha256=sha)
                    for n, sha in sorted(relay_files["packs"][key]["files"].items())]
    for seed in seeds:
        src = relay_files["pca_original"][seed]
        out += [RelayInput(f"{src['prefix']}/{n}", repo=MAIN_RELAY, dest=f"{CHECKPOINTS}/{seed}", sha256=sha)
                for n, sha in sorted(src["files"].items())]
    return tuple(out)


def secure_units(ctx: SessionContext) -> None:
    """Each unit whose report is complete on the volume: its evidence tarred on
    the pod, pulled and hash-checked. Idempotent; never raises."""
    import hashlib
    try:
        if ctx.target is None:
            return
        root = durable_root(ctx.args) + ("/validation" if kind(ctx.args) == "validation" else "")
        aud = f"{REPO}/artifacts/audit/{AUDIT}"
        secured = ctx.evidence.setdefault("units_secured", {})
        excl = " ".join(f"--exclude={e}" for e in HUL.EVIDENCE_EXCLUDES)
        for name in units_of(ctx.args):
            if secured.get(name, {}).get("ok"):
                continue
            res = ctx.target.run(
                f"test -f {root}/{name}/report.descriptor.json && mkdir -p {aud} && tar czf {aud}/{name}.evidence.tgz "
                f"{excl} -C {root}/{name} . && sha256sum {aud}/{name}.evidence.tgz | cut -d' ' -f1", timeout=600)
            digest = (getattr(res, "stdout", "") or "").strip().splitlines()
            if getattr(res, "returncode", 1) != 0 or not digest or len(digest[-1]) != 64:
                continue
            dest = HUL.store(ctx) / f"{name}.evidence.tgz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            rc = HUL._scp(ctx, f"{aud}/{name}.evidence.tgz", dest)
            ok = dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest[-1]
            if ok:
                dest.with_suffix(".sha256").write_text(digest[-1] + "\n")
            secured[name] = {"ok": ok, "sha256": digest[-1], "rc": rc}
            ctx.say(f"unit {name}: evidence {'SECURED' if ok else 'fetch incomplete'}")
        HUL._scp(ctx, STATUS, HUL.store(ctx) / "driver.status", limit_s=120)
        HUL._scp(ctx, f"{aud}/driver_status.json", HUL.store(ctx) / "driver_status.json", limit_s=120)
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def units_secured(ctx: SessionContext, *_a, **_k) -> tuple[bool, str]:
    secure_units(ctx)
    secured = ctx.evidence.get("units_secured", {})
    bad = [k for k, v in secured.items() if not v.get("ok")]
    if bad:
        return False, f"finished units whose evidence is not secured off-pod: {bad}"
    return True, f"{len(secured)} finished unit(s) secured off-pod; checkpoints are on the volume"


def step_seconds(args, repo_root: Path = REPO_ROOT) -> float:
    """Seconds per optimizer step: the committed L40S measurement
    (`cost_forecast.json :: measured.recovery_seconds_per_block_pass.L40S.max`,
    8,192 positions per sequence pass, teacher online, padding forwarded) times
    E1's batch size, or PA2v's measurement (`--step-seconds`) when slower."""
    fc = json.loads((repo_root / GL.FORECAST).read_text())
    e1 = json.loads((repo_root / study()["recipe"]["e1_configs"]["20260726"]["path"]).read_text())
    base = fc["measured"]["recovery_seconds_per_block_pass"]["L40S"]["max"] * int(e1["batch"]["blocks_per_step"])
    return round(max(base, float(args.step_seconds or 0.0)), 4)


def unit_minutes(unit: str, *, step_s: float, repo_root: Path = REPO_ROOT) -> tuple[float, float]:
    """(training, evaluation) minutes for one unit. Training: its optimizer
    steps at `step_s`. Evaluation: the slowest full eval-set-1 evaluation
    recorded on L40S (students at the floor) times the forecast's labelled
    factor for a student that reasons at length."""
    st = study()
    arm = unit.rsplit("-s", 1)[0]
    steps = 0 if arm == "PCA-Original" else int(st["arms"][arm]["schedule"]["optimizer_steps"])
    fc = json.loads((repo_root / GL.FORECAST).read_text())
    ev = (fc["measured"]["eval_seconds_per_unit_L40S"]["max"]
          * fc["contingency_factors_not_measurements"]["eval_floor_to_reasoning"] / 60)
    return round(steps * step_s / 60, 1), round(ev, 1)


def budget(args, repo_root: Path = REPO_ROOT) -> BudgetSpec:
    """PA2v: a bounded validation session. PA2: the named units. The project's
    hard-cap factor is the contingency (P1/P1t's rule); setup and teardown sit
    in the soft-stop reserve."""
    f = json.loads((repo_root / GL.FORECAST).read_text())["contingency_factors_not_measurements"]["hard_cap"]
    if kind(args) == "validation":
        phases = (Phase("pa2v_validation", VALIDATION_MINUTES),)
    else:
        phases, step_s = [], step_seconds(args, repo_root)
        for u in units_of(args):
            t, e = unit_minutes(u, step_s=step_s, repo_root=repo_root)
            phases += ([Phase(f"{u}_train", t)] if t else []) + [Phase(f"{u}_evaluate", e)]
        phases = tuple(phases)
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=1.0,
        step_source="unused: training is priced per unit below (arms=0)",
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
    aud = f"{REPO}/artifacts/audit/{AUDIT}"
    units = " ".join(f"--unit {u}" for u in units_of(a))
    val = (f" --validation --validation-steps {VALIDATION_STEPS} --validation-items {VALIDATION_ITEMS}"
           if kind(a) == "validation" else "")
    cap = f"$(( {int(a.volume_gb) * 10**9} - $(du -sb {a.volume_mount_path} | cut -f1) ))"
    return (f"mkdir -p {durable_root(a)} {aud} && {py} scripts/stages/stage-2/reasoning_data/h1pa_driver.py "
            f"--run-id {a.run_id} --durable-root {durable_root(a)} --pack-root {REPO}/{PACKS} "
            f"--checkpoint-root {REPO}/{CHECKPOINTS} {units}{val} --capacity-bytes {cap} "
            f"--status {aud}/driver_status.json; rc=$?; "
            f"if [ $rc = 0 ]; then m=ALL_DONE; else m=RUN_FAILED; fi; "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:$m\" >> {STATUS}")


def the_artifact_funds_this_phase(ctx: SessionContext) -> tuple[bool, str]:
    got = (ctx.auth.stage_conditions or {}).get("kind")
    if got != kind(ctx.args):
        return False, f"a {got} authorization cannot fund a {ctx.args.phase} session"
    return True, f"a {got} authorization for {ctx.args.phase}"


def study_identity_is_the_authorized_one(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    got = HG.study_hash(repo_root)
    if got != ctx.auth.plan_hash:
        return False, f"the study identity on disk is {got[:12]}, not the authorized {ctx.auth.plan_hash[:12]}"
    return True, f"study identity {got[:12]} is the authorized one"


def bound_files_are_in_the_session_commit(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    import hashlib
    import subprocess
    bad = []
    for f in HG.study_identity(repo_root)["files"]:
        blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{f['path']}"], cwd=repo_root,
                              capture_output=True)
        if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != f["sha256"]:
            bad.append(f["path"])
    return (not bad, "every bound file is in the session commit" if not bad else
            f"not in the session commit as bound: {bad[:4]}")


def spec(args) -> SessionSpec:
    k = kind(args)
    st = study()
    proto = json.loads((REPO_ROOT / st["recipe"]["e1_configs"]["20260726"]["path"]).read_text())
    from stages.phase_c1.autoinit_c1_launch import C1_EVAL_TOKENIZER, C1_ROPE_INPUT
    assets = (LocalAsset(repo_path="artifacts/stages/stage-3/eval/battery_v2", dest_name="battery_v2",
                         install_to="artifacts/stages/stage-3/eval"),)
    return SessionSpec(
        session_id=f"stage2-h1pa-{k}",
        schema="aadistill.stage2.reasoning_data.h1pa_session/v1",
        description=f"H1-PA {'PA2v GPU validation' if k == 'validation' else 'PA2 formal units'}",
        authorization_path=str(HG.authorization_path(args.run_id)),
        authorization_loader=HG.H1PAAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=HG.PLAN_ID,
        plan_hash=HG.study_hash(REPO_ROOT),
        budget=budget(args),
        setup=SetupManifest(
            env={"SESSION_KIND": SESSION_KIND},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS", "SESSION_KIND", "TEACHER_REVISION"),
            #: no TESTS_OK: no repository suite on the paid pod (P8.2.1)
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED", "TRAIN_ENV", "VLLM_READY",
                           "TEACHER_READY", "AUTHORIZATION_OK", "SETUP_DONE"),
            local_assets=assets,
            relay_inputs=(*C1_EVAL_TOKENIZER, *C1_ROPE_INPUT,
                          *relay_inputs(units_of(args), st, st["relay_inputs"])),
            teacher_revision=proto["teacher"]["revision"],
            uv_max_seconds=1500, tests_max_seconds=600),
        driver_command=driver_command,
        driver_job_id=f"stage2_h1pa_{k}_driver",
        status_path=STATUS,
        run_log_path="/workspace/stage2_h1pa_run.log",
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note="every finished unit is on the network volume and its evidence off-pod"),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT, evidence_filename="driver_status.json", archive_basename=f"stage2_h1pa_{k}",
            spec_success="configs/stages/stage-2/reasoning_data/h1_unit_artifacts.json",
            spec_failed="configs/stages/stage-2/reasoning_data/h1_unit_artifacts.json",
            report_names=(), on_poll=secure_units, products_secured=units_secured),
        teardown=TeardownPolicy(note="delete the pod, verify from the provider that it is gone, STOP; "
                                     "the network volume is KEPT (it holds the trained checkpoints)"),
        precheck=(the_artifact_funds_this_phase, GL.all_in_within_cap, study_identity_is_the_authorized_one,
                  bound_files_are_in_the_session_commit),
    )


def closeout(run_id: str, scr: Path, *, repo_root: Path = REPO_ROOT, reconcile=None, disk_gb: float = 150.0) -> dict:
    """`h1_unit_launch.closeout`'s outcome, with this study's storage as the
    expected resources: the driver's terminal marker, every secured evidence
    archive re-hashed, money (GPU + container disk; the volume is booked in its
    storage record) and a fresh provider reconciliation."""
    if reconcile is None:
        from aadistill.infrastructure.provider import RunPodProvider, read_api_key, reconcile_account
        owned = tuple(h["volume"] for h in HG.spent(repo_root)["storage"])
        reconcile = lambda: reconcile_account(  # noqa: E731
            RunPodProvider(read_api_key(str(Path("~/.runpod/config.toml").expanduser()))), owned)
    out = HUL.closeout(run_id, scr, repo_root=repo_root, reconcile=reconcile, disk_gb=disk_gb)
    out["schema"] = "aadistill.stage2.reasoning_data.h1pa_outcome/v1"
    (repo_root / HG.run_dir(run_id) / "closeout/outcome.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=sorted(KIND_OF),
                    help="PA2v (the validation kind) or PA2 (formal units): the grant's phase")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--unit", action="append", default=None, choices=all_units())
    ap.add_argument("--step-seconds", type=float, default=None,
                    help="PA2v's measured seconds per optimizer step; used when slower than the committed L40S rate")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=150)
    ap.add_argument("--volume-gb", type=int, default=50)
    ap.add_argument("--token-src", default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--runpod-config", default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--country-code", default=None)
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
        args.out = str(HG.run_dir(args.run_id) / "runtime/session.json")
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
                          "harness": runner.harness["digest"], "study_identity": session.plan_hash}, indent=1)[:5000])
        return 0 if ok else 12
    rc = run_session(session, args, REPO_ROOT)
    print(json.dumps(closeout(args.run_id, Path(args.scr), disk_gb=args.disk_gb)["money"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
