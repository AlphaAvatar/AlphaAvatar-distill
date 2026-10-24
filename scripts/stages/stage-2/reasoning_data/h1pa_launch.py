#!/usr/bin/env python3
"""Launch one H1-PA student session -- PA2v's GPU validation or PA2's formal units -- on one provider GPU.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/reasoning_data/h1pa_launch.py \
        --phase {PA2v,PA2} --run-id <run_id> --scr <scratch> --session-commit <sha> --bundle <name> \
        --max-price <live price> --network-volume-id <id> --volume-mount-path /runpod-volume \
        --data-center-ids <the volume's datacenter> [--unit D4-Original-s20260726 ...] \
        [--pa2v-gate runs/<PA2v run>/closeout/pa2v_gate.json] [--dry-run]

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

PA2v runs `h1pa_validation.py`; a formal PA2 session runs only on the passed
PA2v gate its authorization names (`the_pa2v_gate_holds`, re-derived from the
evidence at launch), with the volume the provider reports at the planned size
(`the_volume_is_the_planned_one`) and the project cap holding against the
provider's billing at that moment (`the_project_cap_holds_now`).

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
#: PA2v's bound: three 24-step runs and one 12-step run (each loading teacher and student), the
#: persistence check and a 24-item evaluation of each kind; priced at x1.5 like every session
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
    return tuple(args.unit or (study()["gpu_validation"]["units"] if kind(args) == "validation" else all_units()))


def input_units(args) -> tuple[str, ...]:
    """The units whose inputs the pod must hold: a formal session's own; for
    PA2v also every arm `h1pa_validation.RUNS` trains (O trains D4-Original on
    E1's pack, which `gpu_validation.units` does not name), at the validation seed."""
    units = units_of(args)
    if kind(args) != "validation":
        return units
    from stages.reasoning_data import h1pa_validation as V
    seed = study()["gpu_validation"]["training"]["seed"]
    return units + tuple(sorted({f"{arm}-s{seed}" for arm, _ in V.RUNS.values()} - set(units)))


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
        aud = f"{REPO}/artifacts/audit/{AUDIT}"
        secured = ctx.evidence.setdefault("units_secured", {})
        if kind(ctx.args) == "validation":
            #: PA2v's report and evidence (never its training directories or the landed copy)
            root, names, ready = f"{durable_root(ctx.args)}/validation/pa2v", [ctx.args.run_id], "pa2v_report.json"
            excl = "--exclude=./durable --exclude=./train_Y"
            flat = {ctx.args.run_id: HG.PA2V_EVIDENCE.removesuffix(".evidence.tgz")}
        else:
            root, names, ready = durable_root(ctx.args), list(units_of(ctx.args)), "report.descriptor.json"
            excl = " ".join(f"--exclude={e}" for e in HUL.EVIDENCE_EXCLUDES)
            flat = {}
        for name in names:
            if secured.get(name, {}).get("ok"):
                continue
            tag = flat.get(name, name)
            res = ctx.target.run(
                f"test -f {root}/{name}/{ready} && mkdir -p {aud} && tar czf {aud}/{tag}.evidence.tgz "
                f"{excl} -C {root}/{name} . && sha256sum {aud}/{tag}.evidence.tgz | cut -d' ' -f1", timeout=600)
            digest = (getattr(res, "stdout", "") or "").strip().splitlines()
            if getattr(res, "returncode", 1) != 0 or not digest or len(digest[-1]) != 64:
                continue
            dest = HUL.store(ctx) / f"{tag}.evidence.tgz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            rc = HUL._scp(ctx, f"{aud}/{tag}.evidence.tgz", dest)
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


def committed_step_seconds(repo_root: Path = REPO_ROOT) -> float:
    """The committed L40S rate: `cost_forecast.json ::
    measured.recovery_seconds_per_block_pass.L40S.max` (8,192 positions per
    sequence pass, teacher online, padding forwarded) times E1's batch size."""
    fc = json.loads((repo_root / GL.FORECAST).read_text())
    e1 = json.loads((repo_root / study()["recipe"]["e1_configs"]["20260726"]["path"]).read_text())
    return round(fc["measured"]["recovery_seconds_per_block_pass"]["L40S"]["max"] * int(e1["batch"]["blocks_per_step"]), 4)


def formal_unit_seconds(repo_root: Path = REPO_ROOT) -> list[tuple[str, dict]]:
    """(unit, seconds) for every unit a closed formal session of this study
    records (`closeout`: ``units.<u>.seconds``)."""
    out = []
    for d in sorted((repo_root / HG.EXPERIMENT_LOGS / "runs").glob("*")):
        a, o = d / "governance/authorization.json", d / "closeout/outcome.json"
        if a.is_file() and o.is_file() and json.loads(a.read_text()).get("plan_id") == HG.PLAN_ID \
                and (json.loads(a.read_text()).get("stage_conditions") or {}).get("kind") == "formal":
            out += [(u, r.get("seconds") or {}) for u, r in (json.loads(o.read_text()).get("units") or {}).items()]
    return out


def measured_evaluation_seconds(repo_root: Path = REPO_ROOT) -> float | None:
    """The slowest full formal evaluation this phase has measured, or None before any."""
    secs = [s["evaluate"] for _, s in formal_unit_seconds(repo_root) if s.get("evaluate")]
    return max(secs) if secs else None


def measured_training_seconds(repo_root: Path = REPO_ROOT) -> dict[str, float]:
    """arm -> the slowest full formal training of that arm measured so far."""
    out: dict[str, float] = {}
    for u, s in formal_unit_seconds(repo_root):
        if s.get("train"):
            arm = u.rsplit("-s", 1)[0]
            out[arm] = max(out.get(arm, 0.0), float(s["train"]))
    return out


def pricing(args, repo_root: Path = REPO_ROOT) -> dict:
    """What a session's phases are priced from (`h1pa.json :: pricing`).

    Training: max(the committed L40S rate, PA2v's measured median) per step
    (maintainer, 2026-10-25: a 24-step measurement never lowers the historical
    rate, so a hard deadline cannot end a valid training early), +
    ``training_fixed_minutes`` once the gate exists; and never less than a full
    training of the same arm a formal session has already measured.
    Evaluation: the slowest measured full formal evaluation x the margin once
    one exists, else the conservative forecast."""
    pr, fc = study()["pricing"], json.loads((repo_root / GL.FORECAST).read_text())
    gate = json.loads(Path(args.pa2v_gate).read_text()) if getattr(args, "pa2v_gate", None) else None
    measured = (gate or {}).get("measured", {}).get("median_step_seconds")
    committed = committed_step_seconds(repo_root)
    step = max(committed, float(measured)) if measured else committed
    ev_s = measured_evaluation_seconds(repo_root)
    forecast = (fc["measured"]["eval_seconds_per_unit_L40S"]["max"]
                * fc["contingency_factors_not_measurements"]["eval_floor_to_reasoning"] / 60)
    return {"step_seconds": step, "committed_step_seconds": committed,
            "pa2v_median_step_seconds": float(measured) if measured else None,
            "step_source": ("max(the committed L40S rate, PA2v's measured median)" if measured
                            else "the committed L40S rate (no PA2v gate given)"),
            "training_fixed_minutes": float(pr["training_fixed_minutes"]) if measured else 0.0,
            "training_measured_minutes": {a: round(s / 60, 1) for a, s in measured_training_seconds(repo_root).items()},
            "evaluation_minutes": round(ev_s * pr["evaluation_margin_after_measurement"] / 60, 1) if ev_s
            else round(forecast, 1),
            "evaluation_source": "the slowest measured formal evaluation x the margin" if ev_s
            else "the conservative forecast (no formal evaluation measured yet)"}


def unit_minutes(unit: str, price: dict) -> tuple[float, float]:
    """(training, evaluation) minutes for one unit under `pricing`."""
    arm = unit.rsplit("-s", 1)[0]
    steps = 0 if arm == "PCA-Original" else int(study()["arms"][arm]["schedule"]["optimizer_steps"])
    train = steps * price["step_seconds"] / 60 + (price["training_fixed_minutes"] if steps else 0.0)
    if steps:
        train = max(train, (price.get("training_measured_minutes") or {}).get(arm, 0.0))
    return round(train, 1), price["evaluation_minutes"]


def budget(args, repo_root: Path = REPO_ROOT) -> BudgetSpec:
    """PA2v: a bounded validation session. PA2: the named units, priced by
    `pricing`. The project's hard-cap factor is the contingency (P1/P1t's
    rule); setup and teardown sit in the soft-stop reserve."""
    f = json.loads((repo_root / GL.FORECAST).read_text())["contingency_factors_not_measurements"]["hard_cap"]
    if kind(args) == "validation":
        phases = (Phase("pa2v_validation", VALIDATION_MINUTES),)
    else:
        phases, price = [], pricing(args, repo_root)
        for u in units_of(args):
            t, e = unit_minutes(u, price)
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


def volume_bytes(ctx: Any) -> int:
    """The network volume's size as the PROVIDER reported it to the precheck."""
    v = ctx.evidence.get("volume") or {}
    if not v.get("size_gb"):
        raise RuntimeError("no provider-reported volume size: the_volume_is_the_planned_one did not run")
    return int(v["size_gb"]) * 10 ** 9


def driver_command(ctx: Any, plan: Any) -> str:
    a = ctx.args
    py = f"cd {REPO} && PYTHONPATH={REPO}/src:{REPO}/scripts:{REPO}/scripts/stages/stage-1 /opt/train/bin/python"
    aud = f"{REPO}/artifacts/audit/{AUDIT}"
    vol = f"--volume-mount {a.volume_mount_path} --volume-bytes {volume_bytes(ctx)}"
    if kind(a) == "validation":
        cmd = (f"scripts/stages/stage-2/reasoning_data/h1pa_validation.py --run-id {a.run_id} "
               f"--durable-root {durable_root(a)} {vol} --pack-root {REPO}/{PACKS} "
               f"--checkpoint-root {REPO}/{CHECKPOINTS} --status {STATUS}")
        #: the validation driver writes its own terminal marker; one that dies before it gets RUN_FAILED
        return (f"mkdir -p {durable_root(a)} {aud} && {py} {cmd}; "
                f"grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {STATUS} || "
                f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {STATUS}")
    units = " ".join(f"--unit {u}" for u in units_of(a))
    return (f"mkdir -p {durable_root(a)} {aud} && {py} scripts/stages/stage-2/reasoning_data/h1pa_driver.py "
            f"--run-id {a.run_id} --durable-root {durable_root(a)} --pack-root {REPO}/{PACKS} "
            f"--checkpoint-root {REPO}/{CHECKPOINTS} {units} {vol} --status {aud}/driver_status.json; rc=$?; "
            f"if [ $rc = 0 ]; then m=ALL_DONE; else m=RUN_FAILED; fi; "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:$m\" >> {STATUS}")


def _provider(args):
    from aadistill.infrastructure.provider import RunPodProvider, read_api_key
    return RunPodProvider(read_api_key(str(Path(args.runpod_config).expanduser())))


def the_volume_is_the_planned_one(ctx: SessionContext, provider=None) -> tuple[bool, str]:
    """The network volume as the PROVIDER reports it: it exists, its size is the
    plan's (`h1pa.json :: storage.volume_gb`, also --volume-gb), and it sits in
    the session's datacenter. Its size is what the driver counts free space
    from; nothing is inferred from a command-line value."""
    planned = int(study()["storage"]["volume_gb"])
    if int(ctx.args.volume_gb) != planned:
        return False, f"--volume-gb {ctx.args.volume_gb} is not the planned {planned}"
    res = (provider or _provider(ctx.args)).account_resources()
    if not res.known:
        return False, f"the provider's resources are unknown ({res.error}); nothing is created"
    v = next((x for x in res.network_volumes if x.get("id") == ctx.args.network_volume_id), None)
    if v is None:
        return False, f"the provider has no network volume {ctx.args.network_volume_id}"
    dcs = [d for d in str(ctx.args.data_center_ids or "").split(",") if d]
    ctx.evidence["volume"] = {"id": v["id"], "size_gb": v.get("size"), "data_center_id": v.get("dataCenterId")}
    if int(v.get("size") or 0) != planned:
        return False, f"the provider reports {v.get('size')} GB for {v['id']}, not the planned {planned}"
    if not dcs or v.get("dataCenterId") not in dcs:
        return False, f"the volume is in {v.get('dataCenterId')}; --data-center-ids {dcs} must name it"
    return True, f"volume {v['id']}: {v['size']} GB in {v['dataCenterId']}, as planned"


def the_project_cap_holds_now(ctx: SessionContext, snapshot: dict | None = None) -> tuple[bool, str]:
    """The project cap against the provider's billing NOW, not the
    application's snapshot: everything billed, closed sessions not yet posted,
    the study's storage; this session's cap must fit what is left."""
    from aadistill.governance.authorization import AuthorizationError
    try:
        live = HG.live_position(REPO_ROOT, snapshot=snapshot)
    except AuthorizationError as exc:
        return False, str(exc)
    ctx.evidence["live_position"] = live
    if ctx.auth.hard_cap_usd > live["project_remaining_usd"]:
        return False, (f"this session's ${ctx.auth.hard_cap_usd:.2f} does not fit the ${live['project_remaining_usd']:.2f} "
                       f"left under the ${live['project_cap_usd']:.2f} project cap as the provider bills it now")
    return True, f"${ctx.auth.hard_cap_usd:.2f} inside the ${live['project_remaining_usd']:.2f} the provider leaves now"


def the_pa2v_gate_holds(ctx: SessionContext, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    """A formal session runs only on the passed PA2v gate its authorization
    names, re-derived now, with this session's study identity and harness."""
    from aadistill.governance.authorization import AuthorizationError
    if kind(ctx.args) != "formal":
        return True, "a validation session needs no gate"
    named = json.loads((ctx.auth.stage_conditions or {}).get("pa2v_gate") or "null")
    if not named:
        return False, "the formal authorization names no PA2v gate"
    try:
        gate = HG.verify_gate(Path(named["path"]), harness=ctx.auth.harness_source_digest,
                              study=HG.study_hash(repo_root))
    except AuthorizationError as exc:
        return False, str(exc)
    if gate["sha256"] != named["sha256"]:
        return False, "the PA2v gate record changed after the authorization was issued"
    return True, f"PA2v {gate['run_id']} passed; its gate is the one the authorization names"


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
                          *relay_inputs(input_units(args), st, st["relay_inputs"])),
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
                  bound_files_are_in_the_session_commit, the_volume_is_the_planned_one, the_project_cap_holds_now,
                  the_pa2v_gate_holds),
    )


def closeout(run_id: str, scr: Path, *, repo_root: Path = REPO_ROOT, reconcile=None, disk_gb: float = 150.0) -> dict:
    """`h1_unit_launch.closeout`'s outcome, with this study's storage as the
    expected resources, plus where the evidence is (``store``) and, per secured
    unit, its report's stage seconds and its persisted checkpoint's identity
    (the four trained checkpoints are verified here and KEPT: `h1pa.json ::
    retention`)."""
    import tarfile
    if reconcile is None:
        from aadistill.infrastructure.provider import RunPodProvider, read_api_key, reconcile_account
        owned = tuple(h["volume"] for h in HG.spent(repo_root)["storage"])
        reconcile = lambda: reconcile_account(  # noqa: E731
            RunPodProvider(read_api_key(str(Path("~/.runpod/config.toml").expanduser()))), owned)
    out = HUL.closeout(run_id, scr, repo_root=repo_root, reconcile=reconcile, disk_gb=disk_gb)
    store = Path(scr) / "store"
    units = {}
    for t in sorted(store.glob("*.evidence.tgz")):
        name = t.name.removesuffix(".evidence.tgz")
        if name == HG.PA2V_EVIDENCE.removesuffix(".evidence.tgz") or not out["evidence"].get(t.name, {}).get("verified"):
            continue
        with tarfile.open(t) as tf:
            rep = json.loads(tf.extractfile("./report/report.json").read())
            ck = json.loads(tf.extractfile("./checkpoint.descriptor.json").read()) if rep.get("execution") == "train" else None
        units[name] = {"seconds": rep.get("seconds"), "execution": rep.get("execution"),
                       "checkpoint": ({k: ck.get(k) for k in ("artifact_digest", "weights_digest", "destination", "persisted")}
                                      if ck else rep.get("checkpoint"))}
    out.update(schema="aadistill.stage2.reasoning_data.h1pa_outcome/v1", store=str(store), units=units)
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
    ap.add_argument("--pa2v-gate", default=None,
                    help="formal: PA2v's gate record (runs/<PA2v run>/closeout/pa2v_gate.json); prices the training")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=150)
    ap.add_argument("--volume-gb", type=int, default=30, help="must equal h1pa.json :: storage.volume_gb and the provider's size")
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
