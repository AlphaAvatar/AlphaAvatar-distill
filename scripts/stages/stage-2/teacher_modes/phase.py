#!/usr/bin/env python3
"""Research A's D4-scale phase: one grant, three session kinds, every session issued and launched here ($0 until granted).

    # the maintainer's grant from the phase proposal (only on the maintainer's explicit approval)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/phase.py grant \\
        --decision "<the maintainer's words>" --granted-by maintainer
    # a one-use authorization for one session, --kind check | teacher | recovery (--dry-run reads the
    # proposal and writes outside the run dir)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/phase.py issue --kind check \\
        --run-id <run> --expected-usd <x> --hard-cap-usd <cap> [--dry-run --out <scratch>/auth.json]
    # the session (every $0 gate for real first; --dry-run --authorization <scratch>/auth.json stops before a create)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/phase.py launch --kind check \\
        --run-id <run> --scr <scratch> --session-commit <sha> --bundle <name> --max-price <live pod price> \\
        --network-volume-id <id> --volume-mount-path /runpod-volume --data-center-ids <its datacenter> --volume-gb <size>

`shared.study_engineering` with this phase's instance, a FORMAL study: the
teacher and recovery artifacts claim formal, teacher generation, training and
evaluation, the check's engineering validation only (``kind_claims``); each
names one session kind, and setup's ``stage2_teacher_modes_phase`` branch and
each driver re-check identity, harness and kind on the pod (`require_on_pod`).

HARDWARE (d4_scale_design.json :: execution.hardware): every kind runs on the
NVIDIA L40S (``GPU``), the only card the proposal lists. An A100 is never a
default: only an operation measured not to fit an L40S after memory management
that changes no protocol is evidence for one, for that operation alone, under a
separate funding application.

* ``check`` (one L40S, first): `l40s_check.py` -- the phase's formal paths
  on diagnostic inputs (generation, the evaluation engine at 40,960, the formal
  training step, D4 Top-1's path on the hybrid), measured on the card. Its
  evidence directory is the shared relay's, so it is secured, re-hashed and its
  verdict re-derived like a validation's; the dev box's `l40s_check.py record`
  commits its verdict inputs under the run and writes ``CHECK_RECORD``.
* ``teacher`` (one L40S): `teacher_session.py` -- the calibration
  trajectories, the common initialization, T0/V and E's waves in both modes,
  X in H-N, the teacher's own evaluation in both modes. Sealed shards and
  finished evidence are pulled on every poll; everything is on the volume.
  It is priced from the check's measurement, and its artifact is issued only
  once the committed check record admits generation, the 40,960 engine and
  the initialization (`l40s_check.gate`, fail-closed), binding it by sha256.
* ``recovery`` (one or two L40S on ONE pod): `recovery_units.py` -- the six
  units. Its artifact needs the check record to admit the training step, and
  binds it with the committed admission and initialization records
  (``conditions``), which exist only after the teacher session and the dev
  box's admission.

Between them, at $0 on the dev box: `admission.py` on the secured shards
(the exact stopping rule, X, every pack's sha256), committed with the
initialization record the teacher session wrote. Nothing scientific is
decided anywhere else, and nothing here is a pilot or a promotion gate.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from shared import study_engineering as SE  # noqa: E402

DESIGN = "configs/stages/stage-2/teacher_modes/d4_scale_design.json"
PREREG = "logs/stages/stage-2/teacher_modes/plans/d4_scale_preregistration.json"
PROTOCOL = "configs/stages/stage-2/teacher_modes/generation_protocol.json"
H1PA = "configs/stages/stage-2/reasoning_data/h1pa.json"
ORDER = "data/stages/stage-2/teacher_modes/tasks/order_v1.jsonl"
ADMISSION = "logs/stages/stage-2/teacher_modes/data/admission.json"
INIT_RECORD = "logs/stages/stage-2/teacher_modes/assets/common_initialization.json"
CHECK_CONFIG = "configs/stages/stage-2/teacher_modes/l40s_check.json"
CHECK_RECORD = "logs/stages/stage-2/teacher_modes/assets/l40s_check.json"
DIAGNOSTIC = "data/stages/stage-2/teacher_modes/tasks/diagnostic_v1.jsonl"
#: the phase's card, every kind (d4_scale_design.json :: execution.hardware)
GPU = "NVIDIA L40S"
SESSION_KIND = "stage2_teacher_modes_phase"
DURABLE = "teacher_modes"


def _e1_configs() -> tuple[str, ...]:
    return tuple(c["path"] for c in json.loads((REPO_ROOT / H1PA).read_text())["recipe"]["e1_configs"].values())


def _t0_lists() -> tuple[str, ...]:
    return tuple(l["path"] for l in json.loads((REPO_ROOT / PROTOCOL).read_text())["inputs"]["t0"])


STUDY = SE.Study(
    key="A-D4", plan_id="stage2.teacher_modes.d4_scale",
    logs=Path("logs/stages/stage-2/teacher_modes"),
    grant=Path("logs/budget/approvals/teacher_modes_d4_scale_phase_grant.json"),
    proposal=Path("logs/budget/applications/research_a_2026-10-29/proposal.json"),
    identity_files=(DESIGN, PREREG, PROTOCOL, "configs/stages/stage-2/teacher_modes/teacher_identity.json",
                    "configs/stages/stage-2/teacher_modes/evaluation_identity.json", ORDER,
                    "data/stages/stage-2/teacher_modes/tasks/order_v1.manifest.json",
                    "logs/stages/stage-2/teacher_modes/plans/calibration_inputs/calibration_input_g0.jsonl",
                    "configs/shared/evaluation/decoding/nucleus_v1.json", "configs/shared/baselines/selection.json",
                    CHECK_CONFIG, DIAGNOSTIC, "configs/stages/stage-0/calibration_volume/profiles_formal.json",
                    H1PA, *_e1_configs(), *_t0_lists()),
    scripts=tuple(f"scripts/stages/stage-2/teacher_modes/{n}.py" for n in
                  ("phase", "teacher_session", "initialization", "hybrid_calibration", "admission", "recovery_units",
                   "mode_statistics", "analysis", "l40s_check")) + (
        "scripts/shared/study_engineering.py", "scripts/shared/project_position.py", "scripts/shared/eval_sets.py",
        "scripts/shared/forward_baseline.py", "scripts/shared/pod/teacher_cache.py",
        "scripts/shared/evaluation/eval_set_scoring.py", "scripts/shared/evaluation/uncapped_eval.py",
        "scripts/shared/training/train_stage3.py",
        "scripts/stages/stage-0/calibration_volume/run_battery.py",
        "scripts/stages/stage-0/calibration_volume/score_battery.py",
        "scripts/stages/stage-0/calibration_volume/dose_driver.py",
        "scripts/stages/stage-0/calibration_volume/mixed_governance.py",
        "scripts/stages/stage-1/phase_c1/packaging.py", "scripts/stages/stage-1/phase_c1/autoinit_c1_driver.py",
        "scripts/stages/stage-1/phase_d1/d1_session.py", "scripts/stages/stage-1/phase_d1/replay_specs.py",
        "scripts/stages/stage-1/phase_a3/a3_session.py",
        "scripts/stages/stage-1/projected_residual/recovery_driver.py",
        "scripts/stages/stage-2/reasoning_data/generation_driver.py",
        "scripts/stages/stage-2/reasoning_data/generation_adapter.py",
        "scripts/stages/stage-2/reasoning_data/generation_plan.py",
        "scripts/stages/stage-2/reasoning_data/h1pa_driver.py", "scripts/stages/stage-2/reasoning_data/unit_driver.py",
        "scripts/stages/stage-2/reasoning_data/persist_unit.py",
        "scripts/stages/stage-2/reasoning_data/h1_generation_launch.py",
        "scripts/maintenance/consolidation/derive_budget.py", "scripts/maintenance/consolidation/provider_billing.py",
        "scripts/shared/deployment.py", "scripts/shared/pod/watchdog.py",
        "scripts/shared/pod/autoinit_preflight_setup.sh", "scripts/shared/pod/collect_artifacts.py"),
    wire_schema="aadistill.stage2.teacher_modes.phase_authorization/v1",
    claims=frozenset({"formal", "teacher_generation", "training", "evaluation"}),
    kinds=("check", "teacher", "recovery"),
    kind_claims=(("check", frozenset({"engineering_validation"})),))


class PhaseAuthorization(SE.EngineeringAuthorization):
    """The phase's artifact type: setup's ``stage2_teacher_modes_phase`` branch loads exactly this one."""
    STUDY = STUDY


def _sha(rel: str, root: Path = REPO_ROOT) -> str | None:
    p = Path(root) / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def kind_conditions(kind: str, root: Path = REPO_ROOT) -> dict:
    """What a formal kind's artifact binds beyond the phase identity, refusing it until each holds: the
    committed L40S check record, once it ADMITS the operations that kind runs (`l40s_check.gate`, fail-closed:
    identities, evidence, verdict, per-operation status and memory margin; the config's ``gates``), and for
    recovery the committed admission and initialization records too. Issue checks them, and so does the
    driver on the pod (`require_on_pod`)."""
    if kind == "check":
        return {}
    got = {}
    if kind == "recovery":
        got = {"admission_sha256": _sha(ADMISSION, root), "initialization_record_sha256": _sha(INIT_RECORD, root)}
        missing = [k for k, v in got.items() if v is None]
        if missing:
            raise SE.AuthorizationError(f"a recovery session reads {ADMISSION} and {INIT_RECORD}; {missing} do not "
                                        "exist yet: the teacher session and the dev box's admission come first")
    from stages.teacher_modes import l40s_check as LC
    bad = LC.gate(LC.load_config(Path(root) / CHECK_CONFIG)["gates"][kind], root)
    if bad:
        raise SE.AuthorizationError(f"the L40S check record does not admit a {kind} session: " + "; ".join(bad))
    return {**got, "l40s_check_record_sha256": _sha(CHECK_RECORD, root)}


def require_on_pod(path: Path, kind: str, root: Path = REPO_ROOT) -> dict:
    """A driver's own contract on the pod: the artifact, this phase's identity and harness on the pod's
    files, exactly this kind, never a dry run, and the kind's bound records unchanged."""
    auth = PhaseAuthorization.load(path)
    got = SE.require_on_pod(auth, STUDY, auth.plan_hash, str(root), kind=kind)
    bound = kind_conditions(kind, Path(root))
    sc = auth.stage_conditions or {}
    bad = {k: (sc.get(k), v) for k, v in bound.items() if sc.get(k) != v}
    if bad:
        raise SE.AuthorizationError(f"the {kind} artifact binds other records than the pod holds: {bad}")
    return {**got, "harness_digest": got["harness"]}


# ---- securing finished work, on every poll ------------------------------------------------------------------

LISTED = "__A_PHASE_LISTING_COMPLETE__"


def durable(args) -> str:
    return f"{args.volume_mount_path.rstrip('/')}/{DURABLE}"


def _listing(ctx, cmd: str) -> list[str] | None:
    res = ctx.target.run(f"{cmd}; echo {LISTED}", timeout=180)
    lines = (getattr(res, "stdout", "") or "").splitlines()
    if getattr(res, "returncode", 1) != 0 or not lines or lines[-1].strip() != LISTED:
        return None
    return lines[:-1]


def _pull(ctx, remote: str, dest: Path, digest: str | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    SE._scp(ctx, remote, dest, limit_s=1200)
    return dest.is_file() and (digest is None or hashlib.sha256(dest.read_bytes()).hexdigest() == digest)


def _tar(ctx, src: str, name: str, excludes: tuple[str, ...] = ()) -> tuple[bool, str | None]:
    """`src` tarred on the pod, pulled into the store and re-hashed."""
    ex = " ".join(f"--exclude={e}" for e in excludes)
    tgz = f"/workspace/{name}.tgz"
    res = ctx.target.run(f"tar czf {tgz} {ex} -C {src} . && sha256sum {tgz} | cut -d' ' -f1", timeout=900)
    digest = (getattr(res, "stdout", "") or "").strip().splitlines()
    if getattr(res, "returncode", 1) != 0 or not digest or len(digest[-1]) != 64:
        return False, None
    ok = _pull(ctx, tgz, SE.store(ctx) / f"{name}.tgz", digest[-1])
    if ok:
        (SE.store(ctx) / f"{name}.tgz.sha256").write_text(digest[-1] + "\n")
    return ok, digest[-1]


def secure(kind: str):
    """on_poll: every sealed shard by its marker (teacher), every finished unit's evidence and the
    small records (both kinds), into ``<scr>/store``. Idempotent; never raises. (The check is secured
    by the shared evidence relay: its report and evidence directory.)"""
    def on_poll(ctx) -> None:
        try:
            if ctx.target is None:
                return
            root, st = durable(ctx.args), SE.store(ctx)
            got = ctx.evidence.setdefault("secured", {})
            if kind == "teacher":
                lines = _listing(ctx, f"cd {root}/generation 2>/dev/null && for f in *.done; do [ -e \"$f\" ] && "
                                      f"echo \"$f $(cat \"$f\")\"; done; ls *.complete 2>/dev/null || true")
                if lines is None:
                    ctx.evidence.setdefault("listing_unknown", []).append("generation")
                    return
                for line in lines:
                    parts = line.split()
                    if len(parts) == 2 and parts[0].endswith(".done") and len(parts[1]) == 64:
                        name = parts[0][:-5] + ".jsonl"
                        if got.get(name, {}).get("ok"):
                            continue
                        ok = _pull(ctx, f"{root}/generation/{name}", st / "generation" / name, parts[1])
                        if ok:
                            (st / "generation" / parts[0]).write_text(parts[1] + "\n")
                        got[name] = {"ok": ok, "sha256": parts[1]}
                    elif len(parts) == 1 and parts[0].endswith(".complete"):
                        _pull(ctx, f"{root}/generation/{parts[0]}", st / "generation" / parts[0], None)
                for small in ("teacher_session.json", "initialization/initialization.json",
                              "initialization/common_initialization/checkpoint.descriptor.json"):
                    _pull(ctx, f"{root}/{small}", st / small, None)
                units = [f"teacher_eval/teacher_{m}" for m in ("H-T", "H-N")]
            else:
                lines = _listing(ctx, f"cd {root}/units 2>/dev/null && ls -d */ 2>/dev/null | tr -d / || true")
                units = [f"units/{u}" for u in (lines or [])]
                _pull(ctx, f"{root}/recovery_{ctx.args.run_id}.json", st / f"recovery_{ctx.args.run_id}.json", None)
            for u in units:
                tag = u.replace("/", "__")
                if got.get(tag, {}).get("ok"):
                    continue
                done = "evaluation.descriptor.json" if u.startswith("teacher_eval") else "report.descriptor.json"
                ready = ctx.target.run(f"test -f {root}/{u}/{done}", timeout=60)
                if getattr(ready, "returncode", 1) == 0:
                    ok, digest = _tar(ctx, f"{root}/{u}", tag, ("./checkpoint", "./train", "./eval_progress",
                                                                "./progress"))
                    got[tag] = {"ok": ok, "sha256": digest}
            SE._scp(ctx, f"/workspace/{SESSION_KIND}.status", st / "driver.status", limit_s=120)
        except Exception as exc:  # noqa: BLE001
            ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])
    return on_poll


def secured(kind: str):
    def products_secured(ctx, *_a, **_k) -> tuple[bool, str]:
        secure(kind)(ctx)
        got = ctx.evidence.get("secured", {})
        bad = sorted(k for k, v in got.items() if not v.get("ok"))
        if bad or ctx.evidence.get("listing_unknown"):
            return False, f"not secured off-pod: {bad[:6]} {ctx.evidence.get('listing_unknown') or ''}"
        return True, f"{len(got)} finished item(s) secured off-pod; everything is on the network volume"
    return products_secured


# ---- prechecks of a formal session ----------------------------------------------------------------------------

def the_volume_is_the_planned_one(ctx, provider=None) -> tuple[bool, str]:
    """The network volume as the PROVIDER reports it: it exists at the planned size in the session's datacenter."""
    from aadistill.infrastructure.provider import RunPodProvider, read_api_key
    a = ctx.args
    if not a.network_volume_id or not a.volume_mount_path or a.volume_gb is None:
        return False, "a formal session of this phase runs on its network volume (--network-volume-id, --volume-gb)"
    res = (provider or RunPodProvider(read_api_key(str(Path(a.runpod_config).expanduser())))).account_resources()
    if not res.known:
        return False, f"the provider's resources are unknown ({res.error}); nothing is created"
    v = next((x for x in res.network_volumes if x.get("id") == a.network_volume_id), None)
    if v is None:
        return False, f"the provider has no network volume {a.network_volume_id}"
    ctx.evidence["volume"] = {"id": v["id"], "size_gb": v.get("size"), "data_center_id": v.get("dataCenterId")}
    if int(v.get("size") or 0) != int(a.volume_gb):
        return False, f"the provider reports {v.get('size')} GB for {v['id']}, not --volume-gb {a.volume_gb}"
    if v.get("dataCenterId") not in [d for d in str(a.data_center_ids or "").split(",") if d]:
        return False, f"the volume is in {v.get('dataCenterId')}; --data-center-ids must name it"
    return True, f"volume {v['id']}: {v['size']} GB in {v['dataCenterId']}"


def the_kind_records_are_the_bound_ones(ctx) -> tuple[bool, str]:
    sc = ctx.auth.stage_conditions or {}
    try:
        want = kind_conditions(sc.get("kind"))
    except SE.AuthorizationError as exc:
        return False, str(exc)
    bad = {k: v for k, v in want.items() if sc.get(k) != v}
    return (not bad, "the kind's bound records are the committed ones" if not bad else f"records changed: {bad}")


# ---- the sessions ------------------------------------------------------------------------------------------------

def _py() -> str:
    return (f"cd {SE.POD_REPO} && HF_HUB_OFFLINE=1 PYTHONPATH={SE.POD_REPO}/src:{SE.POD_REPO}/scripts:"
            f"{SE.POD_REPO}/scripts/stages/stage-1 /opt/train/bin/python")


def _tail(status: str) -> str:
    return (f"; grep -qE 'MARKER:(ALL_DONE|RUN_FAILED)' {status} || "
            f"echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) MARKER:RUN_FAILED\" >> {status}")


def teacher_command(ctx, plan) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    phases = f" --phases {a.phases}" if getattr(a, "phases", None) else ""
    return (f"mkdir -p {durable(a)} && {_py()} scripts/stages/stage-2/teacher_modes/teacher_session.py "
            f"--authorization {auth} --durable-root {durable(a)} --status {status}{phases}" + _tail(status))


def check_evidence_dir(run_id: str) -> str:
    """The check's evidence directory: the shared relay's (`SE.evidence_dir`), so the shared securing and
    closeout verify it and re-derive its verdict."""
    return f"{SE.POD_REPO}/artifacts/audit/{SESSION_KIND}/{run_id}"


def check_verdict(evidence: Path) -> dict:
    from stages.teacher_modes import l40s_check as LC
    return LC.verdict(Path(evidence), LC.load_config())


def check_command(ctx, plan) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    from stages.teacher_modes import ae_launch as AE
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    out = check_evidence_dir(a.run_id)
    return (f"mkdir -p {out} && {_py()} scripts/stages/stage-2/teacher_modes/l40s_check.py run "
            f"--authorization {auth} --out {out} --work /workspace/a_check --status {status} "
            f"--student {SE.POD_REPO}/{AE.STAND_IN}" + _tail(status))


def recovery_command(ctx, plan) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    from stages.reasoning_data import h1pa_launch as HPL
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    units = "".join(f" --unit {u}" for u in (getattr(a, "unit", None) or []))
    return (f"mkdir -p {durable(a)} && {_py()} scripts/stages/stage-2/teacher_modes/recovery_units.py "
            f"--authorization {auth} --run-id {a.run_id} --durable-root {durable(a)} --status {status} "
            f"--volume-mount {a.volume_mount_path} --volume-bytes {HPL.volume_bytes(ctx)} "
            f"--gpus {a.gpu_count}{units}" + _tail(status))


def launch_spec(kind: str, minutes: float = 0.0) -> SE.Launch:
    from aadistill.infrastructure.session import LocalAsset
    from stages.phase_c1.autoinit_c1_launch import C1_EVAL_TOKENIZER
    t = json.loads((REPO_ROOT / PROTOCOL).read_text())["teacher"]
    battery = LocalAsset(repo_path="artifacts/stages/stage-3/eval/battery_v2", dest_name="battery_v2",
                         install_to="artifacts/stages/stage-3/eval")
    stage0_items = LocalAsset(repo_path="artifacts/stages/stage-0/calibration_volume/profiles_formal_v1",
                              dest_name="profiles_formal_v1", install_to="artifacts/stages/stage-0/calibration_volume")
    common = dict(session_kind=SESSION_KIND, authorization=PhaseAuthorization, teacher_model=t["model"],
                  teacher_revision=t["revision"], kind=kind,
                  local_assets=(stage0_items,) if kind == "check" else (battery,), disk_gb=150,
                  contingency_fraction=0.25, setup_env={"PHASE_KIND": kind},
                  extra_prechecks=(the_volume_is_the_planned_one, the_kind_records_are_the_bound_ones),
                  on_poll=secure(kind), products_secured=secured(kind),
                  artifact_spec="configs/stages/stage-2/teacher_modes/engineering_artifacts.json",
                  teardown_note=("delete the pod, verify from the provider that it is gone, STOP; the network volume "
                                 "is KEPT: it holds every shard, the initialization, save points and checkpoints"))
    if kind == "check":
        from stages.teacher_modes import ae_launch as AE
        common.update(on_poll=None, products_secured=None, verdict=check_verdict,
                      teardown_note=("delete the pod, verify from the provider that it is gone, STOP; the check keeps "
                                     "nothing: its evidence is relayed, its diagnostic weights are scratch"))
        return SE.Launch(**common, driver="scripts/stages/stage-2/teacher_modes/l40s_check.py",
                         validation_minutes=minutes, setup_minutes=20.0, command=check_command,
                         relay_inputs=AE.relay_inputs(),
                         description="Research A: the L40S engineering check (diagnostic inputs, never formal)")
    if kind == "teacher":
        return SE.Launch(**common, driver="scripts/stages/stage-2/teacher_modes/teacher_session.py",
                         validation_minutes=minutes, setup_minutes=20.0, command=teacher_command,
                         description="Research A: the teacher session (calibration, initialization, waves, X, "
                                     "the teacher's evaluation)")
    return SE.Launch(**common, driver="scripts/stages/stage-2/teacher_modes/recovery_units.py",
                     validation_minutes=minutes, setup_minutes=20.0, command=recovery_command,
                     relay_inputs=tuple(C1_EVAL_TOKENIZER),
                     description="Research A: the recovery units (train, persist, evaluate in their own mode)")


def main(argv=None) -> int:
    """``grant | issue --kind K | launch --kind K --minutes M | closeout --kind K | position``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("issue", "launch", "closeout") and "--kind" in argv:
        i = argv.index("--kind")
        kind = argv[i + 1]
        rest = argv[:i] + argv[i + 2:]
        minutes = 0.0
        if "--minutes" in rest:
            j = rest.index("--minutes")
            minutes = float(rest[j + 1])
            rest = rest[:j] + rest[j + 2:]
        if argv[0] == "issue":
            import argparse
            ap = argparse.ArgumentParser(prog=f"{STUDY.key} issue")
            ap.add_argument("--run-id", required=True)
            ap.add_argument("--expected-usd", type=float, required=True)
            ap.add_argument("--hard-cap-usd", type=float, required=True)
            ap.add_argument("--dry-run", action="store_true")
            ap.add_argument("--out", default=None)
            ap.add_argument("--note", default="")
            x = ap.parse_args(rest[1:])
            try:
                path = SE.issue(STUDY, x.run_id, expected_usd=x.expected_usd, hard_cap_usd=x.hard_cap_usd,
                                dry_run=x.dry_run, out=Path(x.out) if x.out else None, note=x.note, kind=kind,
                                conditions=kind_conditions(kind))
            except SE.AuthorizationError as exc:
                print(f"REFUSED at $0: {exc}")
                return 11
            print(path)
            return 0
        if argv[0] == "launch":
            return SE.launch_main(STUDY, launch_spec(kind, minutes), rest[1:], description=__doc__.split("\n")[0],
                                  gpu=GPU)
        return SE.cli(STUDY, lambda: launch_spec(kind, minutes), rest, gpu=GPU,
                      description=__doc__.split("\n")[0])
    return SE.cli(STUDY, lambda: launch_spec("teacher"), argv, gpu=GPU, description=__doc__.split("\n")[0])


def build_parser():
    """The launch parser, for the shared launcher-contract tests."""
    return SE.build_parser(__doc__.split("\n")[0], gpu=GPU)


def spec(args):
    """The session a launch would run, for the shared launcher-contract tests."""
    return SE.session_spec(STUDY, launch_spec(getattr(args, "kind", "teacher") or "teacher", 600.0), args)


if __name__ == "__main__":
    raise SystemExit(main())
