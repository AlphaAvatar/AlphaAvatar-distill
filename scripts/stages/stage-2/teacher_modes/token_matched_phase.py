#!/usr/bin/env python3
"""Research A v2 (token-matched): its own study identity, one session kind -- the supplementary teacher generation.

    # a one-use authorization for one supplement session (--dry-run writes outside the run dir)
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/token_matched_phase.py issue \\
        --kind supplement --run-id <run> --expected-usd <x> --hard-cap-usd <cap> [--dry-run --out <scratch>/auth.json]
    # the session
    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/token_matched_phase.py launch \\
        --kind supplement --minutes <M> --run-id <run> --scr <scratch> --session-commit <sha> --bundle <name> \\
        --max-price <live price> --network-volume-id am5m373kuq --volume-mount-path /runpod-volume \\
        --data-center-ids EUR-IS-2 --volume-gb 55

The study binds token_matched_design.json, its preregistration, the supplement protocol and plan, order_v2 and
the source catalogue extension (`STUDY.identity_files`): Research A v1's frozen identity (phase.py, 848f6508...)
never funds or authorizes anything here, and its grant is closed for new work. The grant
(``logs/budget/approvals/teacher_modes_token_matched_grant.json``) is the maintainer's 2026-10-31 order to
complete the data inside Research A's and the project's existing caps: its ceiling is Research A's $142.19 minus
everything v1 spent (sessions and storage), so the cumulative phase cap is unchanged.

KIND ``supplement`` (one L40S, on am5m373kuq): `supplement_session.py` -- order_v2's rounds in both modes until
each (mode, cell) need is met. Sealed shards are pulled on every poll, like the teacher session's; the volume
keeps everything. There is no other kind: the D4 full search, the initialization, recovery and evaluation are
proposals awaiting the maintainer's review, and this module cannot issue them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

from shared import study_engineering as SE  # noqa: E402
from stages.teacher_modes import phase as P1  # noqa: E402

DESIGN = "configs/stages/stage-2/teacher_modes/token_matched_design.json"
PREREG = "logs/stages/stage-2/teacher_modes/plans/token_matched_preregistration.json"
SUPPLEMENT_PROTOCOL = "configs/stages/stage-2/teacher_modes/supplement_protocol.json"
PLAN = "logs/stages/stage-2/teacher_modes/data/supplement_plan.json"
#: outside git (heavy): bound by its sha256 in order_v2.manifest.json and supplement_protocol.json, staged as an asset
ORDER = "artifacts/stages/stage-2/teacher_modes/tasks/order_v2.jsonl.gz"
GPU = P1.GPU
SESSION_KIND = "stage2_teacher_modes_tm"

STUDY = SE.Study(
    key="A-TM", plan_id="stage2.teacher_modes.token_matched",
    logs=Path("logs/stages/stage-2/teacher_modes"),
    grant=Path("logs/budget/approvals/teacher_modes_token_matched_grant.json"),
    proposal=Path("logs/budget/applications/research_a_tm_2026-10-31/proposal.json"),
    identity_files=(DESIGN, PREREG, SUPPLEMENT_PROTOCOL, PLAN,
                    "data/stages/stage-2/teacher_modes/tasks/order_v2.manifest.json",
                    "data/stages/stage-2/teacher_modes/tasks/calibration_v2.jsonl",
                    "logs/stages/stage-2/teacher_modes/audit/calibration_coverage.json",
                    "configs/stages/stage-2/teacher_modes/token_matched_sources.json", P1.PROTOCOL,
                    "configs/stages/stage-2/teacher_modes/teacher_identity.json"),
    scripts=tuple(f"scripts/stages/stage-2/teacher_modes/{n}.py" for n in
                  ("token_matched_phase", "phase", "supplement_session", "teacher_session", "admission",
                   "supply_audit", "build_supplement_tasks", "build_tasks", "build_calibration_supplement",
                   "hybrid_calibration")) + (
        "scripts/shared/study_engineering.py", "scripts/shared/project_position.py", "scripts/shared/eval_sets.py",
        "scripts/shared/pod/teacher_cache.py", "scripts/stages/stage-0/calibration_volume/run_battery.py",
        "scripts/stages/stage-0/calibration_volume/score_battery.py",
        "scripts/stages/stage-0/calibration_volume/mixed_governance.py",
        "scripts/stages/stage-2/reasoning_data/generation_driver.py",
        "scripts/stages/stage-2/reasoning_data/generation_adapter.py",
        "scripts/stages/stage-2/reasoning_data/generation_plan.py",
        "scripts/stages/stage-2/reasoning_data/supporting_verify.py",
        "scripts/maintenance/consolidation/derive_budget.py", "scripts/maintenance/consolidation/provider_billing.py",
        "scripts/shared/deployment.py", "scripts/shared/pod/watchdog.py",
        "scripts/shared/pod/autoinit_preflight_setup.sh", "scripts/shared/pod/collect_artifacts.py"),
    wire_schema="aadistill.stage2.teacher_modes.tm_authorization/v1",
    claims=frozenset({"formal", "teacher_generation"}),
    kinds=("supplement",))


class TMAuthorization(SE.EngineeringAuthorization):
    """The study's artifact type: setup's ``stage2_teacher_modes_tm`` branch loads exactly this one."""
    STUDY = STUDY


def kind_conditions(kind: str, root: Path = REPO_ROOT) -> dict:
    """The supplement runs the generation path the committed L40S check admitted (the same engine, card and
    protocol as the v1 teacher session); its artifact binds that record."""
    from stages.teacher_modes import l40s_check as LC
    bad = LC.gate(LC.load_config(Path(root) / P1.CHECK_CONFIG)["gates"]["teacher"], root)
    if bad:
        raise SE.AuthorizationError("the L40S check record does not admit generation: " + "; ".join(bad))
    return {"l40s_check_record_sha256": P1._sha(P1.CHECK_RECORD, root)}


def require_on_pod(path: Path, kind: str, root: Path = REPO_ROOT) -> dict:
    auth = TMAuthorization.load(path)
    got = SE.require_on_pod(auth, STUDY, auth.plan_hash, str(root), kind=kind)
    sc = auth.stage_conditions or {}
    bad = {k: (sc.get(k), v) for k, v in kind_conditions(kind, Path(root)).items() if sc.get(k) != v}
    if bad:
        raise SE.AuthorizationError(f"the {kind} artifact binds other records than the pod holds: {bad}")
    return {**got, "harness_digest": got["harness"]}


#: small v1 records left on the volume when the recovery was stopped (unit 6 trained, not scored)
V1_SMALL = tuple(f"units/H-T-C-s20260801/{n}" for n in ("checkpoint.descriptor.json", "training.descriptor.json",
                                                         "attempts.jsonl", "train/a1/train_log.jsonl",
                                                         "train/a1/run_manifest.json", "train/a1/train_inputs.json",
                                                         "train/a1/checkpoints/latest.txt")) + (
    "recovery_a_recovery_20261030_211948.json",)


#: this study's stage labels (supplement_session.py: s<round><t|n>_<mode>_pass<k>_...)
OWN_SHARD = re.compile(r"^s\d\d[tn]_")


def secure(ctx) -> None:
    """on_poll: every sealed shard by its marker, the session record and v1's small records. Never raises."""
    try:
        if ctx.target is None:
            return
        root, st = P1.durable(ctx.args), SE.store(ctx)
        got = ctx.evidence.setdefault("secured", {})
        lines = P1._listing(ctx, f"cd {root}/generation 2>/dev/null && for f in *.done; do [ -e \"$f\" ] && "
                                 f"echo \"$f $(cat \"$f\")\"; done; ls *.complete 2>/dev/null || true")
        if lines is None:
            ctx.evidence.setdefault("listing_unknown", []).append("generation")
            return
        for line in lines:
            parts = line.split()
            if not parts or not OWN_SHARD.match(parts[0]):
                continue                    #: v1's shards are on the dev box already: artifacts follow consumers
            if len(parts) == 2 and parts[0].endswith(".done") and len(parts[1]) == 64:
                name = parts[0][:-5] + ".jsonl"
                if got.get(name, {}).get("ok"):
                    continue
                ok = P1._pull(ctx, f"{root}/generation/{name}", st / "generation" / name, parts[1])
                if ok:
                    (st / "generation" / parts[0]).write_text(parts[1] + "\n")
                got[name] = {"ok": ok, "sha256": parts[1]}
            elif len(parts) == 1 and parts[0].endswith(".complete"):
                P1._pull(ctx, f"{root}/generation/{parts[0]}", st / "generation" / parts[0], None)
        P1._pull(ctx, f"{root}/supplement_session.json", st / "supplement_session.json", None)
        if not ctx.evidence.get("v1_small"):
            ctx.evidence["v1_small"] = {f: P1._pull(ctx, f"{root}/{f}", st / "v1_small" / f, None) for f in V1_SMALL}
        SE._scp(ctx, f"/workspace/{SESSION_KIND}.status", st / "driver.status", limit_s=120)
    except Exception as exc:  # noqa: BLE001
        ctx.evidence.setdefault("secure_errors", []).append(repr(exc)[:300])


def products_secured(ctx, *_a, **_k) -> tuple[bool, str]:
    secure(ctx)
    got = ctx.evidence.get("secured", {})
    bad = sorted(k for k, v in got.items() if not v.get("ok"))
    if bad or ctx.evidence.get("listing_unknown"):
        return False, f"not secured off-pod: {bad[:6]} {ctx.evidence.get('listing_unknown') or ''}"
    return True, f"{len(got)} sealed shard(s) secured off-pod; everything is on the network volume"


def the_kind_records_are_the_bound_ones(ctx) -> tuple[bool, str]:
    sc = ctx.auth.stage_conditions or {}
    try:
        want = kind_conditions(sc.get("kind"))
    except SE.AuthorizationError as exc:
        return False, str(exc)
    bad = {k: v for k, v in want.items() if sc.get(k) != v}
    return (not bad, "the kind's bound records are the committed ones" if not bad else f"records changed: {bad}")


def supplement_command(ctx, plan) -> str:
    a, status = ctx.args, f"/workspace/{SESSION_KIND}.status"
    auth = f"{SE.POD_REPO}/{STUDY.authorization_path(a.run_id)}"
    return (f"mkdir -p {P1.durable(a)} && {P1._py()} scripts/stages/stage-2/teacher_modes/supplement_session.py "
            f"--authorization {auth} --durable-root {P1.durable(a)} --status {status}" + P1._tail(status))


def launch_spec(kind: str = "supplement", minutes: float = 0.0) -> SE.Launch:
    import json
    from aadistill.infrastructure.session import LocalAsset
    t = json.loads((REPO_ROOT / P1.PROTOCOL).read_text())["teacher"]
    battery = LocalAsset(repo_path="artifacts/stages/stage-3/eval/battery_v2", dest_name="battery_v2",
                         install_to="artifacts/stages/stage-3/eval")         #: the v1 teacher kind's assets, unchanged
    order = LocalAsset(repo_path=ORDER, dest_name=Path(ORDER).name, install_to=str(Path(ORDER).parent))
    return SE.Launch(
        session_kind=SESSION_KIND, authorization=TMAuthorization, teacher_model=t["model"],
        teacher_revision=t["revision"], kind=kind, local_assets=(battery, order), disk_gb=150,
        contingency_fraction=0.25,
        setup_env={"PHASE_KIND": kind},
        extra_prechecks=(P1.the_volume_is_the_planned_one, the_kind_records_are_the_bound_ones),
        on_poll=secure, products_secured=products_secured,
        artifact_spec="configs/stages/stage-2/teacher_modes/engineering_artifacts.json",
        teardown_note=("delete the pod, verify from the provider that it is gone, STOP; the network volume is KEPT: it "
                       "holds every shard, v1's checkpoints and the historical ones"),
        driver="scripts/stages/stage-2/teacher_modes/supplement_session.py", validation_minutes=minutes,
        setup_minutes=20.0, command=supplement_command,
        description="Research A v2: the supplementary teacher generation (order_v2, both modes, token quotas)")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("issue", "launch", "closeout") and "--kind" in argv:
        i = argv.index("--kind")
        kind = argv[i + 1]
        if kind not in STUDY.kinds:
            print(f"REFUSED at $0: {kind!r} is not a kind of {STUDY.key} ({list(STUDY.kinds)})")
            return 11
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
        return SE.cli(STUDY, lambda: launch_spec(kind, minutes), rest, gpu=GPU, description=__doc__.split("\n")[0])
    return SE.cli(STUDY, launch_spec, argv, gpu=GPU, description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())
