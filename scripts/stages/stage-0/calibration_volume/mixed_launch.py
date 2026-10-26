#!/usr/bin/env python3
"""MC1 and MC2 of the mixed-calibration 2x study: the $0 steps around them and the two paid sessions.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/mixed_launch.py <command> ...

    position                                   the project position as the provider bills it now (writes nothing)
    issue {search,recovery} --run-id R --expected-usd X --hard-cap-usd Y
    create-volume --data-center-id DC          the MC1 network volume (runs/mixvol_<utc>), after the MC1 issue
    launch {MC1,MC2} --run-id R --scr S --session-commit C --bundle B --max-price P [...] [--dry-run]
    closeout {MC1,MC2} --run-id R --scr S
    read-back --store S                        the MC1 Top-1 from the relay to the dev box, re-identified
    retire-volume --run-id mixvol_<utc>        after the read-back: tombstone, delete, provider check, actual cost

Both sessions are `aadistill.infrastructure.session_runner` sessions: the
argument contract, the authorization (`mixed_governance`) bound to the
session identity and the harness, the price, GPU + disk + termination inside
the cap (`h1_generation_launch.all_in_within_cap`), the account's resources
exactly the declared ones, the project position against the provider's bill
now (`the_project_cap_holds_now`), then the watchdog, setup, the driver, the
poll loop, collection, teardown and the provider's confirmation.

Completed work leaves the pod as it finishes (P8.2.1): MC1's search records and
Top-1 manifest as each `records/<stage>.done` appears; MC2's per-unit evidence
archive (never weights or save points) as each unit's report completes. The
weights stay on the network volumes: MC1's states on its own volume until the
read-back, MC2's checkpoints on `am5m373kuq`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from aadistill.infrastructure.session import (ArtifactPolicy, BudgetSpec, ExecutionCommands,  # noqa: E402
                                              LocalAsset, MarkerPolicy, RelayInput, SessionContext,
                                              SessionSpec, SetupManifest, TeardownPolicy)
from aadistill.infrastructure.budget import Phase  # noqa: E402
from shared.deployment import MAIN_RELAY, deployment_commands  # noqa: E402
from stages.calibration_volume import governance as G  # noqa: E402
from stages.calibration_volume import launch as L  # noqa: E402
from stages.calibration_volume import mixed_governance as MG  # noqa: E402

REPO = "/workspace/aad"
MC1_AUDIT = "stage0_calvol_mixed_search"
MC2_AUDIT = "stage0_calvol_mixed_recovery"
STATUS = {"MC1": "/workspace/stage0_mixed_mc1.status", "MC2": "/workspace/stage0_mixed_mc2.status"}
SEARCH_UNIT = "artifacts/audit/stage0_calvol_search/2x"
PACKS = "artifacts/stages/stage-2/reasoning_data/h1pa/packs"
WORK = "artifacts/stages/stage-0/calibration_volume/mixed/units"
EVIDENCE_EXCLUDES = ("./checkpoint", "./train", "./eval_progress")
#: H1-PA's measured D4-Augmented units (analyses/h1pa_result.json :: units.<u>.seconds), the
#: bound factor the application priced MC2 with (mixed_application.json :: phases.MC2)
RECOVERY_MEASURED = "logs/stages/stage-2/reasoning_data/analyses/h1pa_result.json"
RECOVERY_BOUND_FACTOR = 1.25
KIND = {"MC1": "search", "MC2": "recovery"}


def mixed_study() -> dict:
    return json.loads((REPO_ROOT / MG.RECOVERY_STUDY).read_text())


def init_dir() -> str:
    return mixed_study()["initialization"]["dir"]


# ---- MC1 ------------------------------------------------------------------------------------------------

def mc1_budget(plan: dict, volume_id: str) -> BudgetSpec:
    """The application's bound (DEPTH in all 16 positions) is the contingency: none is added on top."""
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=1.0,
        step_source="unused: no training in this session (arms=0)",
        below_floor_reason="no training steps; every phase is priced below",
        setup_minutes=float(plan["setup_minutes"]), transfer_minutes=0.0,
        other_phases=tuple(Phase(k, float(v)) for k, v in plan["budget_phases"].items()),
        contingency_fraction=float(plan["contingency_fraction"]), soft_stop_reserves=(),
        artifact_recovery_reserve_minutes=float(plan["artifact_recovery_reserve_minutes"]),
        account_balance_required_usd=20.0,
        accounted_provider_resources=tuple(v for v in (volume_id, *retained_volumes()) if v))


def retained_volumes(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    """The recorded volumes still alive (H1-PA's am5m373kuq; MC1's while it exists)."""
    return tuple(v["volume"] for v in MG.volume_records(repo_root) if v["deleted"] is None)


def mc1_driver_command(ctx: Any, plan: Any) -> str:
    a = ctx.args
    py = f"cd {REPO} && PYTHONPATH={REPO}/src:{REPO}/scripts /opt/train/bin/python"
    return (f"{py} scripts/stages/stage-0/calibration_volume/mixed_search_driver.py --run-id {a.run_id} "
            f"--auth '{REPO}/{MG.authorization_path(a.run_id)}' --status {STATUS['MC1']} "
            f"--volume {a.volume_mount_path}; "
            f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {STATUS['MC1']} || "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {STATUS['MC1']}")


def _done(ctx: SessionContext, audit: str) -> dict[str, str]:
    res = ctx.target.run(f"cd {audit}/records 2>/dev/null && for f in *.done; do "
                         f"[ -e \"$f\" ] && echo \"$f $(cat $f)\"; done", timeout=120)
    out = {}
    for line in (getattr(res, "stdout", "") or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].endswith(".done"):
            out[parts[0][:-5]] = parts[1]
    return out


def secure_mc1(ctx: SessionContext) -> None:
    """Each finished MC1 stage's records, pulled and hash-checked. Idempotent; never raises."""
    try:
        if ctx.target is None:
            return
        aud, unit = f"{REPO}/artifacts/audit/{MC1_AUDIT}", f"{REPO}/{SEARCH_UNIT}"
        secured = ctx.evidence.setdefault("units_secured", {})
        dest = L.store(ctx) / "2x"
        (dest / "search").mkdir(parents=True, exist_ok=True)
        for name, digest in _done(ctx, aud).items():
            if secured.get(name, {}).get("sha256") == digest and secured[name].get("ok"):
                continue
            files = list(L.SEARCH_UNIT_FILES) if name == "search" else ["top1_manifest.json"]
            if name == "search":
                ls = ctx.target.run(f"ls {unit}/ | grep -E '^width_projection_' || true", timeout=120)
                files += (getattr(ls, "stdout", "") or "").split()
            rcs = {f: L._scp(ctx, f"{unit}/{f}", dest / f, limit_min=20) for f in files}
            key = dest / ("search_record.json" if name == "search" else "top1_manifest.json")
            ok = key.is_file() and hashlib.sha256(key.read_bytes()).hexdigest() == digest and \
                all(rc == 0 for rc in rcs.values())
            secured[name] = {"ok": ok, "sha256": digest, "rc": rcs}
            ctx.say(f"MC1 {name}: {'SECURED' if ok else 'fetch incomplete'}")
        (L.store(ctx) / "mc1").mkdir(parents=True, exist_ok=True)
        L._scp(ctx, f"{aud}/mixed_search_record.json", L.store(ctx) / "mc1" / "mixed_search_record.json")
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def mc1_secured(ctx: SessionContext, *_a, **_k) -> tuple[bool, str]:
    secure_mc1(ctx)
    secured = ctx.evidence.get("units_secured", {})
    bad = [k for k, v in secured.items() if not v.get("ok")]
    if bad:
        return False, f"finished MC1 stages not secured off-pod: {bad}"
    return True, f"{len(secured)} finished MC1 stage(s) secured off-pod; the states are on the MC1 volume"


def the_volume_is_the_mc1_volume(ctx: SessionContext, provider=None) -> tuple[bool, str]:
    """The MC1 volume as the PROVIDER reports it: this study's live record names
    it, its size is the plan's, and it sits in the session's datacenter."""
    plan = G.plan_doc(MG.SEARCH_PLAN)
    rec = next((v for v in MG.volume_records(REPO_ROOT) if v["volume"] == ctx.args.network_volume_id
                and v["record"].startswith(str(MG.EXPERIMENT_LOGS / "runs" / "mixvol_"))), None)
    if rec is None or rec["deleted"] is not None:
        return False, f"{ctx.args.network_volume_id} is not a live MC1 volume record (runs/mixvol_*)"
    res = (provider or _provider(ctx.args)).account_resources()
    if not res.known:
        return False, f"the provider's resources are unknown ({res.error}); nothing is created"
    v = next((x for x in res.network_volumes if x.get("id") == ctx.args.network_volume_id), None)
    want = int(plan["hardware"]["network_volume"]["size_gb"])
    dcs = [d for d in str(ctx.args.data_center_ids or "").split(",") if d]
    if v is None or int(v.get("size") or 0) != want or v.get("dataCenterId") not in dcs:
        return False, f"the provider reports {v} for the MC1 volume; want {want} GB in {dcs}"
    ctx.evidence["volume"] = {"id": v["id"], "size_gb": v.get("size"), "data_center_id": v.get("dataCenterId")}
    return True, f"MC1 volume {v['id']}: {v['size']} GB in {v['dataCenterId']}"


def mc1_spec(args) -> SessionSpec:
    plan = G.plan_doc(MG.SEARCH_PLAN)
    if not args.network_volume_id or not args.volume_mount_path:
        raise SystemExit("MC1 needs --network-volume-id and --volume-mount-path: its journal and states must "
                         "survive the pod")
    if args.gpu != plan["hardware"]["gpu"]:
        raise SystemExit(f"--gpu {args.gpu!r} is not the plan's {plan['hardware']['gpu']!r}")
    return SessionSpec(
        session_id="stage0-mixed-mc1",
        schema="aadistill.stage0.calibration_volume.mixed_search_session/v1",
        description="Mixed calibration 2x MC1: D1 full search on calib.s0m_mix_2x@v2 and its Top-1",
        authorization_path=str(MG.authorization_path(args.run_id)),
        authorization_loader=partial(MG.load_for, "search"),
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=MG.PLAN_IDS["search"],
        plan_hash=MG.identity_hash("search"),
        budget=mc1_budget(plan, args.network_volume_id),
        setup=SetupManifest(
            env={"SESSION_KIND": "stage0_calvol"},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS", "SESSION_KIND", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED", "TRAIN_ENV", "TEACHER_READY", "SETUP_DONE"),
            local_assets=L.search_assets(plan),
            teacher_revision=plan["teacher_revision"],
            uv_max_seconds=1500, tests_max_seconds=600),
        driver_command=mc1_driver_command,
        driver_job_id="stage0_mixed_mc1_driver",
        status_path=STATUS["MC1"],
        run_log_path="/workspace/stage0_mixed_mc1_run.log",
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note="the journal and every state are on the MC1 volume; finished records are off-pod"),
        artifacts=ArtifactPolicy(
            audit_dirname=MC1_AUDIT, evidence_filename="mixed_search_record.json",
            archive_basename="stage0_mixed_mc1",
            spec_success="configs/stages/stage-0/calibration_volume/mixed_search_artifacts.json",
            spec_failed="configs/stages/stage-0/calibration_volume/mixed_search_artifacts.json",
            report_names=(), on_poll=secure_mc1, products_secured=mc1_secured),
        teardown=TeardownPolicy(note="delete the pod, verify from the provider that it is gone, STOP; the MC1 "
                                     "volume is KEPT until the Top-1 is read back and the evidence is durable"),
        precheck=(the_artifact_is_this_kind, the_authorization_is_in_the_session_commit, _all_in_within_cap,
                  identity_is_the_authorized_one, the_volume_is_the_mc1_volume, the_project_cap_holds_now),
    )


# ---- MC2 ------------------------------------------------------------------------------------------------

def mc2_units(args) -> tuple[str, ...]:
    st = mixed_study()
    seeds = json.loads((REPO_ROOT / MG.PREREG).read_text())["seeds"]
    return tuple(args.unit or [f"{st['arm']}-s{s}" for s in seeds])


def measured_recovery_minutes(repo_root: Path = REPO_ROOT) -> dict:
    """The slowest measured D4-Augmented training and evaluation (H1-PA), in minutes."""
    units = json.loads((repo_root / RECOVERY_MEASURED).read_text())["units"]
    d4a = {n: u["seconds"] for n, u in units.items() if n.startswith("D4-Augmented")}
    return {"train": max(s["train"] for s in d4a.values()) / 60,
            "evaluate": max(s["evaluate"] for s in d4a.values()) / 60, "units": sorted(d4a)}


def mc2_budget(args) -> BudgetSpec:
    m = measured_recovery_minutes()
    phases = []
    for u in mc2_units(args):
        phases += [Phase(f"{u}_train", round(m["train"] * RECOVERY_BOUND_FACTOR, 1)),
                   Phase(f"{u}_evaluate", round(m["evaluate"] * RECOVERY_BOUND_FACTOR, 1))]
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=1.0,
        step_source="unused: training is priced per unit below (arms=0)",
        below_floor_reason="no arm-level steps; every phase is priced below",
        setup_minutes=20.0, transfer_minutes=0.0, other_phases=tuple(phases),
        contingency_fraction=0.0, soft_stop_reserves=(),
        artifact_recovery_reserve_minutes=5.0, account_balance_required_usd=20.0,
        accounted_provider_resources=retained_volumes())


def mc2_relay_inputs() -> tuple[RelayInput, ...]:
    """What the two units read and nothing else: the merged set (D4-Augmented's pack, from H1-PA's
    hash-pinned list) and the MC1 Top-1 (from its read-back record)."""
    from stages.phase_c1.autoinit_c1_launch import C1_EVAL_TOKENIZER, C1_ROPE_INPUT
    st = mixed_study()
    base = json.loads((REPO_ROOT / st["base_study"]["path"]).read_text())
    key = base["arms"][st["control_arm"]]["dataset"]
    src = base["relay_inputs"]["packs"][key]
    out = [RelayInput(f"{src['prefix']}/{n}", repo=MAIN_RELAY, dest=f"{PACKS}/{key}", sha256=s)
           for n, s in sorted(src["files"].items())]
    rec = json.loads(MG.mc1_top1_record(REPO_ROOT).read_text())
    out += [RelayInput(f["path"], repo=MAIN_RELAY, dest=init_dir(), sha256=f["sha256"]) for f in rec["relay_files"]]
    return (*C1_EVAL_TOKENIZER, *C1_ROPE_INPUT, *out)


def mc2_driver_command(ctx: Any, plan: Any) -> str:
    from stages.reasoning_data import h1pa_launch as HPL
    a = ctx.args
    root = f"{a.volume_mount_path.rstrip('/')}/{mixed_study()['durable_subdir']}"
    aud = f"{REPO}/artifacts/audit/{MC2_AUDIT}"
    py = f"cd {REPO} && PYTHONPATH={REPO}/src:{REPO}/scripts:{REPO}/scripts/stages/stage-1 /opt/train/bin/python"
    units = " ".join(f"--unit {u}" for u in mc2_units(a))
    return (f"mkdir -p {root} {aud} && {py} scripts/stages/stage-0/calibration_volume/mixed_recovery_driver.py "
            f"--auth '{REPO}/{MG.authorization_path(a.run_id)}' --pack-root {REPO}/{PACKS} "
            f"--run-id {a.run_id} --durable-root {root} --init-dir {REPO}/{init_dir()} "
            f"--work {REPO}/{WORK}/{a.run_id} {units} --volume-mount {a.volume_mount_path} "
            f"--volume-bytes {HPL.volume_bytes(ctx)} --status {aud}/driver_status.json; rc=$?; "
            f"if [ $rc = 0 ]; then m=ALL_DONE; else m=RUN_FAILED; fi; "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:$m\" >> {STATUS['MC2']}")


def secure_mc2(ctx: SessionContext) -> None:
    """Each unit whose report is complete on the volume: its evidence tarred on the pod
    (never weights, save points or evaluation progress), pulled and hash-checked."""
    try:
        if ctx.target is None:
            return
        root = f"{ctx.args.volume_mount_path.rstrip('/')}/{mixed_study()['durable_subdir']}"
        aud = f"{REPO}/artifacts/audit/{MC2_AUDIT}"
        secured = ctx.evidence.setdefault("units_secured", {})
        excl = " ".join(f"--exclude={e}" for e in EVIDENCE_EXCLUDES)
        for name in mc2_units(ctx.args):
            if secured.get(name, {}).get("ok"):
                continue
            res = ctx.target.run(
                f"test -f {root}/{name}/report.descriptor.json && mkdir -p {aud} && tar czf {aud}/{name}.evidence.tgz "
                f"{excl} -C {root}/{name} . && sha256sum {aud}/{name}.evidence.tgz | cut -d' ' -f1", timeout=600)
            digest = (getattr(res, "stdout", "") or "").strip().splitlines()
            if getattr(res, "returncode", 1) != 0 or not digest or len(digest[-1]) != 64:
                continue
            dest = L.store(ctx) / f"{name}.evidence.tgz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            rc = L._scp(ctx, f"{aud}/{name}.evidence.tgz", dest, limit_min=30)
            ok = dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest[-1]
            if ok:
                dest.with_suffix(".sha256").write_text(digest[-1] + "\n")
            secured[name] = {"ok": ok, "sha256": digest[-1], "rc": rc}
            ctx.say(f"MC2 unit {name}: evidence {'SECURED' if ok else 'fetch incomplete'}")
        L._scp(ctx, STATUS["MC2"], L.store(ctx) / "driver.status")
        L._scp(ctx, f"{aud}/driver_status.json", L.store(ctx) / "driver_status.json")
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def mc2_secured(ctx: SessionContext, *_a, **_k) -> tuple[bool, str]:
    secure_mc2(ctx)
    secured = ctx.evidence.get("units_secured", {})
    bad = [k for k, v in secured.items() if not v.get("ok")]
    if bad:
        return False, f"finished units whose evidence is not secured off-pod: {bad}"
    return True, f"{len(secured)} finished unit(s) secured off-pod; checkpoints are on am5m373kuq"


def bound_files_are_in_the_session_commit(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    bad = []
    for f in MG.recovery_identity(repo_root)["files"]:
        blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{f['path']}"], cwd=repo_root,
                              capture_output=True)
        if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != f["sha256"]:
            bad.append(f["path"])
    return (not bad, "every bound file is in the session commit" if not bad else
            f"not in the session commit as bound: {bad[:4]}")


def mc2_spec(args) -> SessionSpec:
    from stages.reasoning_data import h1pa_launch as HPL
    proto = json.loads((REPO_ROOT / "configs/stages/stage-2/reasoning_data/h1pa.json").read_text())
    e1 = json.loads((REPO_ROOT / proto["recipe"]["e1_configs"]["20260726"]["path"]).read_text())
    return SessionSpec(
        session_id="stage0-mixed-mc2",
        schema="aadistill.stage0.calibration_volume.mixed_recovery_session/v1",
        description="Mixed calibration 2x MC2: Mix-Augmented recovery and eval-set-1, both seeds",
        authorization_path=str(MG.authorization_path(args.run_id)),
        authorization_loader=partial(MG.load_for, "recovery"),
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=MG.PLAN_IDS["recovery"],
        plan_hash=MG.identity_hash("recovery"),
        budget=mc2_budget(args),
        setup=SetupManifest(
            env={"SESSION_KIND": "stage0_calvol"},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS", "SESSION_KIND", "TEACHER_REVISION"),
            #: no TESTS_OK: no repository suite on the paid pod (P8.2.1); the driver checks the authorization
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED", "TRAIN_ENV", "VLLM_READY",
                           "TEACHER_READY", "SETUP_DONE"),
            local_assets=(LocalAsset(repo_path="artifacts/stages/stage-3/eval/battery_v2", dest_name="battery_v2",
                                     install_to="artifacts/stages/stage-3/eval"),),
            relay_inputs=mc2_relay_inputs(),
            teacher_revision=e1["teacher"]["revision"],
            uv_max_seconds=1500, tests_max_seconds=600),
        driver_command=mc2_driver_command,
        driver_job_id="stage0_mixed_mc2_driver",
        status_path=STATUS["MC2"],
        run_log_path="/workspace/stage0_mixed_mc2_run.log",
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note="every finished unit is on am5m373kuq and its evidence off-pod"),
        artifacts=ArtifactPolicy(
            audit_dirname=MC2_AUDIT, evidence_filename="driver_status.json", archive_basename="stage0_mixed_mc2",
            spec_success="configs/stages/stage-0/calibration_volume/mixed_recovery_artifacts.json",
            spec_failed="configs/stages/stage-0/calibration_volume/mixed_recovery_artifacts.json",
            report_names=(), on_poll=secure_mc2, products_secured=mc2_secured),
        teardown=TeardownPolicy(note="delete the pod, verify from the provider that it is gone, STOP; "
                                     "am5m373kuq is KEPT (it holds the control and the new checkpoints)"),
        precheck=(the_artifact_is_this_kind, the_authorization_is_in_the_session_commit, _all_in_within_cap,
                  identity_is_the_authorized_one, bound_files_are_in_the_session_commit,
                  HPL.the_volume_is_the_planned_one, the_project_cap_holds_now),
    )


# ---- prechecks shared by both -----------------------------------------------------------------------------

def _provider(args):
    from aadistill.infrastructure.provider import RunPodProvider, read_api_key
    return RunPodProvider(read_api_key(str(Path(args.runpod_config).expanduser())))


def _all_in_within_cap(ctx: SessionContext) -> tuple[bool, str]:
    from stages.reasoning_data import h1_generation_launch as GL
    return GL.all_in_within_cap(ctx)


def the_artifact_is_this_kind(ctx: SessionContext) -> tuple[bool, str]:
    got = (ctx.auth.stage_conditions or {}).get("kind")
    if got != KIND[ctx.args.phase]:
        return False, f"a {got} authorization cannot fund {ctx.args.phase}"
    return True, f"a {got} authorization for {ctx.args.phase}"


def identity_is_the_authorized_one(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    got = MG.identity_hash(KIND[ctx.args.phase], repo_root)
    if got != ctx.auth.plan_hash:
        return False, f"the session identity on disk is {got[:12]}, not the authorized {ctx.auth.plan_hash[:12]}"
    return True, f"session identity {got[:12]} is the authorized one"


def the_authorization_is_in_the_session_commit(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    """The pod checks out the session commit, so the authorization its driver loads must be IN
    that commit, byte for byte (mixs_20261026_102914 died at A_contract: issued after the commit,
    it never reached the pod)."""
    rel = MG.authorization_path(ctx.args.run_id)
    blob = subprocess.run(["git", "show", f"{ctx.args.session_commit}:{rel}"], cwd=repo_root, capture_output=True)
    here = (repo_root / rel).read_bytes() if (repo_root / rel).is_file() else None
    if blob.returncode != 0 or blob.stdout != here:
        return False, f"{rel} is not in session commit {ctx.args.session_commit[:12]} as issued; commit it first"
    return True, f"the authorization is in session commit {ctx.args.session_commit[:12]}"


def the_project_cap_holds_now(ctx: SessionContext, snapshot: dict | None = None) -> tuple[bool, str]:
    """The project cap against the provider's bill NOW (`mixed_governance.position`): this
    session's cap must fit what is left; a position that cannot be confirmed refuses."""
    from aadistill.governance.authorization import AuthorizationError
    try:
        pos = MG.position(REPO_ROOT, snapshot=snapshot)
    except AuthorizationError as exc:
        return False, str(exc)
    ctx.evidence["position"] = pos
    if ctx.auth.hard_cap_usd > pos["project_remaining_usd"]:
        return False, (f"HARD BOUNDARY: this session's ${ctx.auth.hard_cap_usd:.2f} does not fit the "
                       f"${pos['project_remaining_usd']:.2f} left under the ${pos['project_cap_usd']:.2f} project cap")
    return True, f"${ctx.auth.hard_cap_usd:.2f} inside the ${pos['project_remaining_usd']:.2f} the provider leaves now"


# ---- the MC1 volume ---------------------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _volume_listing() -> list[dict]:
    r = subprocess.run(["runpodctl", "network-volume", "list"], capture_output=True, text=True, timeout=120,
                       check=True)
    return json.loads(r.stdout)


def create_volume(data_center_id: str, repo_root: Path = REPO_ROOT) -> Path:
    """The MC1 volume, after its reservation fits the envelope and the project position."""
    from aadistill.governance.authorization import AuthorizationError
    g = MG.grant(repo_root)
    spec = g["phases"]["MC1"]["network_volume"]
    env = MG.remaining_envelope(repo_root)
    if env["storage"]:
        raise AuthorizationError(f"the study already holds volume records {env['storage']}")
    pos = MG.position(repo_root)
    if pos["committed_usd"] + float(spec["reserved_usd"]) > pos["project_cap_usd"]:
        raise AuthorizationError(f"HARD BOUNDARY: the volume's ${spec['reserved_usd']} does not fit {pos}")
    plan = G.plan_doc(MG.SEARCH_PLAN, repo_root)
    size = int(plan["hardware"]["network_volume"]["size_gb"])
    if size > int(spec["max_gb"]):
        raise AuthorizationError(f"{size} GB exceeds the grant's {spec['max_gb']} GB")
    run_id = f"mixvol_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    name = "aad-mixed-mc1"
    cmd = ["runpodctl", "network-volume", "create", "--name", name, "--size", str(size),
           "--data-center-id", data_center_id]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    try:
        vid = json.loads(r.stdout).get("id")
    except json.JSONDecodeError:
        vid = None
    rec = {"schema": "aadistill.stage0.calibration_volume.network_volume/v1", "run_id": run_id,
           "kind": "network_volume", "phase": "MC1", "study_id": "stage0.calibration_volume.mixed_v2",
           "provider_resource_created": bool(vid), "volume_id": vid, "name": name, "data_center_id": data_center_id,
           "size_gb": size, "created_utc": _now(), "created_by": " ".join(cmd),
           "reserved_usd": float(spec["reserved_usd"]), "create_output": (r.stdout + r.stderr)[-1500:],
           "why": "MC1's durable store: the search journal and every materialized state, mirrored while the search "
                  "runs, so a lost pod resumes rather than recomputes; deleted after the Top-1 and the evidence are "
                  "durable and read back",
           "price_basis": {"usd_per_gb_month": 0.07, "hours_per_month": 720, "provider_api_exposes_this": False}}
    p = repo_root / MG.run_dir(run_id) / "runtime/session.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1) + "\n")
    if not vid:
        raise SystemExit(f"the create returned no volume id (rc {r.returncode}); recorded in {p}")
    return p


def retire_volume(run_id: str, why: str, evidence: list[str], repo_root: Path = REPO_ROOT) -> dict:
    """Tombstone first, then delete, then the provider's listing, then the actual cost."""
    run = repo_root / MG.run_dir(run_id)
    sess = json.loads((run / "runtime/session.json").read_text())
    vid = sess["volume_id"]
    refs = sorted(subprocess.run(["git", "grep", "-l", vid], cwd=repo_root, capture_output=True,
                                 text=True).stdout.split())
    tomb = {"schema": "aadistill.stage0.calibration_volume.volume_tombstone/v1", "volume_id": vid,
            "name": sess["name"], "data_center_id": sess["data_center_id"], "size_gb": sess["size_gb"],
            "created_utc": sess["created_utc"], "status": "INTENT", "intent_utc": _now(), "deleted_utc": None,
            "deletion_criterion": "the grant's MC1 term: deleted after the Top-1 and the search evidence are durable "
                                  "and read back (AGENTS.md 2.5: the states have no consumer once the Top-1 is on "
                                  "the relay and re-identified on the dev box)",
            "why": why, "canonical_surviving_evidence": evidence, "references_at_deletion": refs}
    out = run / "closeout/tombstone.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tomb, indent=1) + "\n")
    if not any(v.get("id") == vid for v in _volume_listing()):
        raise SystemExit(f"the provider does not list {vid}; reconcile before closing")
    r = subprocess.run(["runpodctl", "network-volume", "delete", vid], capture_output=True, text=True, timeout=300)
    if any(v.get("id") == vid for v in _volume_listing()):
        raise SystemExit(f"{vid} is still listed after the delete (rc {r.returncode}); the tombstone stays INTENT")
    deleted = datetime.now(timezone.utc)
    tomb.update(status="DELETED", deleted_utc=deleted.isoformat(timespec="seconds"),
                delete_command=f"runpodctl network-volume delete {vid}", provider_confirms_gone=True)
    out.write_text(json.dumps(tomb, indent=1) + "\n")
    pb = sess["price_basis"]
    hours = (deleted - datetime.fromisoformat(sess["created_utc"].replace("Z", "+00:00"))).total_seconds() / 3600
    usd = math.ceil(sess["size_gb"] * pb["usd_per_gb_month"] * hours / pb["hours_per_month"] * 1e4) / 1e4
    oc = {"schema": "aadistill.stage0.calibration_volume.closeout/v1", "run_id": run_id,
          "experiment_id": G.EXPERIMENT_ID, "stage_id": G.STAGE_ID, "phase": "MC1",
          "recorded_utc": deleted.isoformat(timespec="seconds"), "status": "VOLUME_DELETED",
          "provider_resource_created": True,
          "money": {"all_in_usd": usd, "hours": round(hours, 3), "reserved_usd_replaced": sess["reserved_usd"],
                    "_basis": f"ACTUAL: {sess['size_gb']} GB x ${pb['usd_per_gb_month']}/GB-month x {hours:.3f} h / "
                              f"{pb['hours_per_month']} h, ceiled to the 4-decimal quantum"},
          "tombstone": "closeout/tombstone.json", "authorizes": "nothing"}
    (run / "closeout/outcome.json").write_text(json.dumps(oc, indent=1) + "\n")
    return oc


# ---- the MC1 Top-1, read back -------------------------------------------------------------------------------

def read_back_top1(store: Path, repo_root: Path = REPO_ROOT, token: str | None = None) -> Path:
    """Download the Top-1 from the relay by the pod's manifest, check every file's sha256 and
    size, rebuild the checkpoint identity from the bytes on the dev box, and write the
    initialization record MC2 binds (`mixed_recovery.json :: initialization.record`)."""
    import shutil
    from huggingface_hub import HfApi, hf_hub_download
    from aadistill.initialization.adapters import register_builtin_adapters
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.runtime.leaf_durability import verify_transferred_leaf
    from stages.reasoning_data.unit_driver import hf_token
    token = token or hf_token()
    man_p = Path(store) / "2x" / "top1_manifest.json"
    man = json.loads(man_p.read_text())
    srec = Path(store) / "2x" / "search_record.json"
    if hashlib.sha256(srec.read_bytes()).hexdigest() != man["search_record_sha256"]:
        raise SystemExit("the secured search record is not the one the Top-1 manifest names")
    dest = repo_root / init_dir()
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    files = []
    for f in man["files"]:
        got = Path(hf_hub_download(man["relay_repo"], f["path"], token=token, local_dir=str(dest / ".relay")))
        target = dest / f["name"]
        shutil.move(str(got), target)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        files.append({"name": f["name"], "bytes": target.stat().st_size, "sha256": sha,
                      "matched": sha == f["sha256"] and target.stat().st_size == f["bytes"]})
    shutil.rmtree(dest / ".relay", ignore_errors=True)
    info = HfApi(token=token).get_paths_info(man["relay_repo"], [f["path"] for f in man["files"]], expand=True)
    meta = {i.path: {"size": i.size, "lfs_sha256": i.lfs.sha256 if i.lfs else None, "blob_id": i.blob_id}
            for i in info}
    ident = man["identity"]
    register_builtin_adapters()
    v = verify_transferred_leaf(dest, ident, adapter=get_adapter("qwen3"))
    ok = all(f["matched"] for f in files) and v["matched"] and v["weights_digest_matched"] \
        and v["config_matched"] is not False
    rec = {"schema": "aadistill.stage2.reasoning_data.initialization_materialization/v1",
           "written_utc": _now(), "study_id": man["study_id"],
           "decision": "logs/budget/approvals/mixed_calibration_grant.json (MC1: the committed Top-1 of the "
                       "single-profile D1 full search on calib.s0m_mix_2x@v2)",
           "method_identity": {"baseline": "stage1_init.d1_target_aware.scored_v1@v1",
                               "search_run_id": man["search_run_id"], "search_config_hash": man["search_config_hash"],
                               "mc1_run_id": man["run_id"],
                               "search_evidence": {"search_record_sha256": man["search_record_sha256"],
                                                   "selection_sha256": man["selection_sha256"],
                                                   "manifest_sha256": hashlib.sha256(man_p.read_bytes()).hexdigest()}},
           "materialization_identity": {
               "state_id": man["selected"]["state_id"], "operator_path": man["selected"].get("path"),
               "path": ident.get("materialization"), "checkpoint_sha256": ident.get("checkpoint_sha256"),
               "artifact_digest": ident["artifact_digest"], "weights_digest": ident["weights_digest"],
               "config_sha256": ident.get("config_sha256"), "arch_signature": ident["arch_signature"],
               "tokenizer_sha256": ident.get("tokenizer_sha256"), "num_parameters": ident["num_parameters"]},
           "tokenizer_binding": {"stored_with_checkpoint": False,
                                 "source": "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint (the tokenizer "
                                           "source of D4-Augmented's recovery, h1pa_driver.TOKENIZER_SOURCE)"},
           "relay_files": [{k: f[k] for k in ("path", "name", "sha256", "bytes")} for f in man["files"]],
           "relay_metadata": meta,
           "read_back": {"dest": str(dest.relative_to(repo_root)), "files": files, "identity": v, "verified": ok,
                         "_": "downloaded from the relay to the dev box and re-identified from the bytes that "
                              "arrived (verify_transferred_leaf), separately from the relay's own metadata"},
           "authorizes": "nothing"}
    out = MG.mc1_top1_record(repo_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    if not ok:
        raise SystemExit(f"the relay copy does not re-identify: {files} {v}; MC2 is not authorized")
    return out


# ---- closeouts ---------------------------------------------------------------------------------------------

def closeout(phase: str, run_id: str, scr: Path, *, repo_root: Path = REPO_ROOT, disk_gb: float,
             reconcile=None) -> dict:
    """The run's outcome on the dev box: the driver's terminal marker, every secured file
    re-hashed, money (GPU from the session record plus container disk; volumes are booked
    in their own records) and the provider's account after. Writes closeout/outcome.json,
    whose money.all_in_usd the envelope reads."""
    from aadistill.infrastructure.provider import RunPodProvider, read_api_key, reconcile_account
    from stages.reasoning_data import h1_generation_launch as GL
    run = repo_root / MG.run_dir(run_id)
    st = Path(scr) / "store"
    sess = json.loads((run / "runtime/session.json").read_text())
    created = bool(sess.get("provider_resource_created"))
    cost = sess.get("cost") or {}
    if created and cost.get("actual_usd") is None:
        raise SystemExit("the session created a provider resource and records no cost; reconcile first")
    gpu, minutes = float(cost.get("actual_usd") or 0.0), float(cost.get("elapsed_minutes") or 0.0)
    disk = disk_gb * GL.DISK_USD_PER_GB_MONTH * minutes / (60 * 24 * 30.4375)
    files = {str(p.relative_to(st)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in sorted(st.rglob("*")) if p.is_file()} if st.is_dir() else {}
    if phase == "MC1":
        rec_p = st / "mc1" / "mixed_search_record.json"
        terminal = json.loads(rec_p.read_text()).get("status") if rec_p.is_file() else None
        complete = terminal == "COMPLETE" and (sess.get("required_products_secured") or {}).get("ok")
    else:
        status = (st / "driver.status").read_text() if (st / "driver.status").is_file() else ""
        terminal = "ALL_DONE" if "MARKER:ALL_DONE" in status else "RUN_FAILED" if "MARKER:RUN_FAILED" in status else None
        complete = terminal == "ALL_DONE" and (sess.get("required_products_secured") or {}).get("ok")
        for t in sorted(st.glob("*.evidence.tgz")):
            want = t.with_suffix(".sha256")
            files[t.name]["verified"] = want.is_file() and want.read_text().strip() == files[t.name]["sha256"]
    rec = (reconcile or (lambda: reconcile_account(
        RunPodProvider(read_api_key(str(Path("~/.runpod/config.toml").expanduser()))), retained_volumes(repo_root))))()
    out = {"schema": "aadistill.stage0.calibration_volume.mixed_outcome/v1", "run_id": run_id, "phase": phase,
           "experiment_id": G.EXPERIMENT_ID, "stage_id": G.STAGE_ID, "recorded_utc": _now(),
           "status": "COMPLETE" if complete and sess.get("provider_confirms_gone") else "INCOMPLETE",
           "terminal": terminal, "runner_terminal": sess.get("terminal"), "session_passed": sess.get("passed"),
           "provider_resource_created": created, "pod_id": sess.get("pod_id"),
           "provider_confirms_gone": sess.get("provider_confirms_gone"), "store": str(st), "files": files,
           "money": {"all_in_usd": math.ceil((gpu + disk) * 1e4) / 1e4 if created else 0.0, "gpu_usd": gpu,
                     "disk_usd": round(disk, 4), "elapsed_minutes": minutes,
                     "price_per_hour": cost.get("price_per_hour"), "authorized_usd": cost.get("authorized_usd"),
                     "_basis": f"GPU actual from runtime/session.json :: cost plus container disk at "
                               f"${GL.DISK_USD_PER_GB_MONTH}/GB-month over the elapsed minutes ({disk_gb:g} GB); "
                               "volumes are booked in their own records"},
           "provider_reconciliation_after": rec, "authorizes": "nothing"}
    dest = run / "closeout/outcome.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


UNITS_LOG = Path("logs/stages/stage-0/calibration_volume/units")
EVIDENCE_MIX = Path("artifacts/stages/stage-0/calibration_volume/mixed/units/evidence")
#: small, reviewable records that go into git; journals, telemetry and generations stay in the store
SMALL = 2_000_000


def land_evidence(phase: str, run_id: str, scr: Path, repo_root: Path = REPO_ROOT) -> dict:
    """MC1: the search records and the Top-1 manifest into the run's evidence/ (the journal
    and telemetry stay in the store, hashed in the closeout). MC2: each verified unit archive
    extracted to the artifacts root the analysis reads, and its report, descriptors, attempts,
    formal report and per-item rows into units/<unit>/ -- never a generation, which may
    complete an unsafe prompt."""
    import shutil
    import tarfile
    st, landed = Path(scr) / "store", []
    if phase == "MC1":
        ev = repo_root / MG.run_dir(run_id) / "evidence"
        for f in sorted(st.rglob("*.json")):
            if f.stat().st_size <= SMALL:
                (ev / f.relative_to(st)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, ev / f.relative_to(st))
                landed.append(str(f.relative_to(st)))
        return {"landed": landed}
    out = json.loads((repo_root / MG.run_dir(run_id) / "closeout/outcome.json").read_text())
    for t in sorted(st.glob("*.evidence.tgz")):
        if not out["files"].get(t.name, {}).get("verified"):
            continue
        unit = t.name.removesuffix(".evidence.tgz")
        dest = repo_root / EVIDENCE_MIX / unit
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        with tarfile.open(t) as tf:
            tf.extractall(dest, filter="data")
        log = repo_root / UNITS_LOG / unit
        shutil.rmtree(log, ignore_errors=True)
        keep = ["report/report.json", "attempts.jsonl", "checkpoint.descriptor.json", "evaluation.descriptor.json",
                "report.descriptor.json", "training.descriptor.json", "evaluation/formal_report.json"]
        keep += [str(p.relative_to(dest)) for p in sorted((dest / "evaluation/rows").glob("*.rows.jsonl"))]
        for k in keep:
            if (dest / k).is_file():
                target = log / (Path(k).name if k == "report/report.json" else k)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest / k, target)
        landed.append(unit)
    return {"landed": landed}


# ---- the command line -----------------------------------------------------------------------------------------

def build_launch_parser(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("phase", choices=sorted(KIND))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--unit", action="append", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--max-price", type=float, required=True)
    ap.add_argument("--disk-gb", type=int, default=None)
    ap.add_argument("--volume-gb", type=int, default=30)
    ap.add_argument("--token-src", default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--runpod-config", default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--country-code", default=None)
    ap.add_argument("--network-volume-id", required=True)
    ap.add_argument("--volume-mount-path", default="/runpod-volume")
    ap.add_argument("--data-center-ids", required=True)
    ap.add_argument("--image-digest", default=None)
    ap.add_argument("--setup-timeout-s", type=float, default=3600.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=1440.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")


DEFAULTS = {"MC1": {"gpu": "NVIDIA A100-SXM4-80GB", "disk_gb": 160},
            "MC2": {"gpu": "NVIDIA L40S", "disk_gb": 150}}


def launch(args) -> int:
    from aadistill.governance.authorization import AuthorizationError
    from aadistill.infrastructure.session_runner import run_session
    from stages.reasoning_data import h1_generation_launch as GL
    for k, v in DEFAULTS[args.phase].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.out is None:
        args.out = str(MG.run_dir(args.run_id) / "runtime/session.json")
    try:
        session = (mc1_spec if args.phase == "MC1" else mc2_spec)(args)
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11
    ok, why = GL.bundle_on_relay(args)
    print(f"{'bundle OK' if ok else 'REFUSED at $0'}: {why}")
    if not ok and not args.dry_run:
        print("No provider resource was created.")
        return 13
    return run_session(session, args, REPO_ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("position")
    i = sub.add_parser("issue")
    i.add_argument("kind", choices=sorted(MG.KINDS))
    i.add_argument("--run-id", required=True)
    i.add_argument("--expected-usd", type=float, required=True)
    i.add_argument("--hard-cap-usd", type=float, required=True)
    i.add_argument("--note", default="")
    c = sub.add_parser("create-volume")
    c.add_argument("--data-center-id", required=True)
    build_launch_parser(sub.add_parser("launch"))
    o = sub.add_parser("closeout")
    o.add_argument("phase", choices=sorted(KIND))
    o.add_argument("--run-id", required=True)
    o.add_argument("--scr", required=True)
    o.add_argument("--disk-gb", type=float, required=True)
    r = sub.add_parser("read-back")
    r.add_argument("--store", required=True)
    v = sub.add_parser("retire-volume")
    v.add_argument("--run-id", required=True)
    v.add_argument("--why", required=True)
    v.add_argument("--evidence", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "position":
        print(json.dumps(MG.position(), indent=1, default=str))
    elif a.cmd == "issue":
        print(MG.issue(a.kind, a.run_id, expected_usd=a.expected_usd, hard_cap_usd=a.hard_cap_usd, note=a.note))
    elif a.cmd == "create-volume":
        print(create_volume(a.data_center_id))
    elif a.cmd == "launch":
        return launch(a)
    elif a.cmd == "closeout":
        print(json.dumps(closeout(a.phase, a.run_id, Path(a.scr), disk_gb=a.disk_gb)["money"]))
        print(json.dumps(land_evidence(a.phase, a.run_id, Path(a.scr))))
    elif a.cmd == "read-back":
        print(read_back_top1(Path(a.store)))
    elif a.cmd == "retire-volume":
        print(json.dumps(retire_volume(a.run_id, a.why, a.evidence)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
