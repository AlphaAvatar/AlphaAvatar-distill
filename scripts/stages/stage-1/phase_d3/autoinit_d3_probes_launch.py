#!/usr/bin/env python3
"""The D3 PROBES launcher: three recovery probes from the D3 initialization.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
        scripts/stages/stage-1/phase_d3/autoinit_d3_probes_launch.py \
        --scr <durable store> --session-commit <sha> --bundle <canonical> \
        --run-id d3_probes_<utc> --country-code AU --create-attempts 144

CONDITIONAL and separately authorized (`D3ProbesAuthorization`): it exists
only once the structural session completed with STRUCTURAL_CHANGE and its
product is durable on the relay. The session IS D2's shared-confirmation
session with another arm and another experiment's records -- the same driver
(`autoinit_d2_screening_driver.py --rung d3_probes`), the same host admission
to the protocol's image identity, the same durability (D1's functions under
this rung's layout: every scored probe's evidence and every trained
checkpoint off-pod, verified, before the pod may release them). What is D3's
own: the run directories, the grant, the registration, the arm's relay
inputs, the price, the `$0` gates. The comparison with q2 and B is computed
off-pod (`analyze_d3_probes.py`), exploratory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, MarkerPolicy, RelayInput,
    SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    same_failure_gate, session_commit_gate,
)
from shared.deployment import MAIN_RELAY, deployment_commands  # noqa: E402
from shared.pod.autoinit_science_inputs import RECOVERY_LADDER  # noqa: E402
from shared.run_layout import rel_run_dir  # noqa: E402
from stages.phase_c1.autoinit_c1_launch import C1_EVAL_TOKENIZER  # noqa: E402
from stages.phase_d1 import autoinit_d1_behavioural_launch as D1L  # noqa: E402
from stages.phase_d1 import d1_session as D1S  # noqa: E402
from stages.phase_d2 import autoinit_d2_screening_launch as D2L  # noqa: E402
from stages.phase_d2.autoinit_d2_screening_driver import rung_layout  # noqa: E402
from stages.phase_d3 import d3_design as D  # noqa: E402
from stages.phase_d3 import d3_probes as P  # noqa: E402
from stages.phase_d3.d3_authorization import (  # noqa: E402
    D3ProbesAuthorization, governance,
)

SESSION_KIND = "d3_probes"
RUNG = "probes"
REPO = "/workspace/aad"
DRIVER = "scripts/stages/stage-1/phase_d2/autoinit_d2_screening_driver.py"
ACCOUNT_OPERATIONAL_RESERVE_USD = 5.0
TEST_PATHS = ("scripts/stages/stage-1/phase_d3/tests/probes_pod_preflight",)
SCIENCE_ASSETS = D2L.SCIENCE_ASSETS


def make_rung() -> D2L.Rung:
    """The shared launcher core's `Rung`, carrying THIS rung's layout, so the
    durability functions secure under D3's paths and never D2's."""
    d = rung_layout(P.RUNG)
    evidence_dir = f"artifacts/audit/{d['audit_dirname']}"
    layout = D1L.BehaviouralLayout(
        evidence_dir=evidence_dir, evidence_filename=d["evidence_filename"],
        attested_name=d["attested_name"],
        pod_probe_root=f"{REPO}/{d['probe_root_rel']}",
        release_ack_rel=d["ack_rel"],
        extra_probe_files=(("rescoring.json", "rescoring/{probe_id}.json",
                            False),),
        required_training_records=P.TRAINING_RECORDS)
    return D2L.Rung(
        name=P.RUNG, session_kind=SESSION_KIND, contract=P,
        authorization_type=D3ProbesAuthorization, test_paths=TEST_PATHS,
        audit_dirname=d["audit_dirname"], evidence_dir=evidence_dir,
        evidence_filename=d["evidence_filename"],
        attested_name=d["attested_name"], status=d["status"],
        run_log="/workspace/autoinit_d3_probes_run.log",
        probe_root_rel=d["probe_root_rel"],
        eval_packages_rel=d["eval_packages_rel"], release_ack_rel=d["ack_rel"],
        run_prefix=governance(RUNG).run_prefix, layout=layout,
        description=(
            "D3 PROBES on d_series_shared_eval@v1: the D3 initialization (the "
            "structural session's product), staged from the relay and "
            "verified at all four identities, trained under the frozen "
            "recovery recipe at the three D1 confirmation seeds, generated "
            "through the unchanged generator, admitted against the shared "
            "confirmation rung's evaluation protocol, scored by its pinned "
            "scorer and rollout-rescored as a diagnostic. Stops at "
            "preservation; the exploratory comparison with q2 and B is "
            "computed off-pod."))


PROBES = make_rung()


def run_dir_for(run_id: str | None) -> str:
    return rel_run_dir(D.EXPERIMENT_ID, run_id or "unrecorded", D.STAGE_ID)


def auth_path_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/authorization.json"


def bundle_record_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/bundle.json"


def arm_relay_inputs(repo_root: Path = REPO_ROOT) -> tuple[RelayInput, ...]:
    """The D3 product's three files, pinned to the bytes the structural
    closeout verified locally AND after a relay round trip."""
    product = P.structural_product(repo_root)
    digest = product["identities"]["artifact_digest"]
    relay = product["relay"]
    out = []
    for name in P.ARM_FILES:
        row = relay["files"][name]
        if row["sha256"] != product["files"][name]["sha256"]:
            raise P.D3ProbesError(f"{name}: the relay pin is not the product's")
        out.append(RelayInput(row["path"], repo=relay["repo"],
                              dest=f"{P.POD_ARMS_REL}/{digest}",
                              sha256=row["sha256"]))
    return tuple(out)


def relay_inputs(repo_root: Path = REPO_ROOT) -> tuple[RelayInput, ...]:
    return (*RECOVERY_LADDER, *C1_EVAL_TOKENIZER, *arm_relay_inputs(repo_root))


def driver_command(ctx: Any, plan: Any) -> str:
    run_id = getattr(ctx.args, "run_id", "unrecorded")
    return (f"/opt/train/bin/python {REPO}/{DRIVER} --rung {P.RUNG} "
            f"--out '{REPO}/{PROBES.evidence_dir}' --run-id '{run_id}' "
            f"--authorization '{REPO}/{auth_path_for(run_id)}' "
            f"--arm-root '{REPO}/{P.POD_ARMS_REL}' "
            f"--status '{PROBES.status}' --device cuda "
            f"--image-digest '{ctx.image_digest}'")


# --- the $0 gates ----------------------------------------------------------------------

def run_identity_gate(ctx) -> tuple[bool, str]:
    try:
        ctx.auth.require_run_id(getattr(ctx.args, "run_id", ""))
        ctx.auth.require_plan(D.registration_hash(REPO_ROOT))
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, (f"run {ctx.auth.run_id}, registration "
                  f"{ctx.auth.plan_hash[:12]}")


def relay_arm_gate(ctx) -> tuple[bool, str]:
    """The D3 product is on the relay at its pinned sizes and LFS sha256."""
    from stages.phase_d3.closeout_d3_structural import remote_listing

    try:
        inputs = arm_relay_inputs(REPO_ROOT)
        prefix = inputs[0].path.rsplit("/", 1)[0]
        listing = remote_listing(prefix)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    problems = []
    for item in inputs:
        have = listing.get(item.path)
        if have is None:
            problems.append(f"{item.path}: absent")
        elif have["lfs_sha256"] not in (None, item.sha256):
            problems.append(f"{item.path}: relay oid {have['lfs_sha256'][:12]}")
    if problems:
        return False, f"the D3 product is not obtainable: {problems[:3]}"
    ctx.evidence["d3_probes_relay_arm"] = [r.as_record() for r in inputs]
    return True, f"{len(inputs)} product files on the relay at their pins"


def bundle_staged_gate(ctx) -> tuple[bool, str]:
    import tempfile

    from aadistill.infrastructure.bundle_transport import hf_download, roundtrip

    commit = getattr(ctx.args, "session_commit", "")
    try:
        D1S.require_canonical_bundle(getattr(ctx.args, "bundle", ""), commit)
        record = json.loads((REPO_ROOT / bundle_record_for(
            getattr(ctx.args, "run_id", None))).read_text())
        auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
        with tempfile.TemporaryDirectory(prefix="d3p-bundle-") as tmp:
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
    out = []
    for d in sorted(p for p in RUNS_ROOT.iterdir()
                    if p.is_dir() and p.name.startswith(PROBES.run_prefix)):
        log = d / "runtime" / "launcher.log"
        if log.is_file():
            out.append((d.name, log.read_text(errors="replace")))
    return out


def _corrective_change_since() -> tuple[bool, str]:
    logs = _prior_attempt_logs()
    if not logs:
        return True, "no prior probes attempt exists"
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
    return False, ("a prior probes attempt exists and its session commit "
                   "cannot be read")


def prechecks(args) -> tuple:
    return (run_identity_gate, partial(D2L.contract_gate, rung=PROBES),
            relay_arm_gate,
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", None)),
                                check_lineage=True),
            same_failure_gate(_prior_attempt_logs,
                              corrective_change=_corrective_change_since),
            bundle_staged_gate)


def budget_spec(repo_root: Path, *, authorized_minutes: float | None = None
                ) -> BudgetSpec:
    """D2's confirmation decomposition, at D3's priced bound."""
    from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase

    from stages.phase_d1.d1_authorization import live_money
    from stages.phase_d3.d3_authorization import session_ceiling

    priced = session_ceiling(repo_root, RUNG)
    overhead = float(priced["session_overhead_minutes"])
    accepted_hard = float(priced["hard_ceiling_minutes"])
    if authorized_minutes is not None:
        accepted_hard = min(accepted_hard, float(authorized_minutes))
    expected_total = float(priced["expected_minutes"])
    contingency, recovery_reserve = 0.10, 30.0
    reserve = round(accepted_hard - expected_total * (1.0 + contingency)
                    - recovery_reserve, 2)
    if reserve < 0:
        raise SystemExit(
            f"the accepted {accepted_hard:.2f}-minute bound cannot hold the "
            f"expected {expected_total:.2f} min plus contingency and reserve")
    return BudgetSpec(
        arms=0, steps_per_arm=0, step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: three probes from one relay-staged arm, every "
                     "phase named below from the priced step"),
        setup_minutes=round(overhead * 0.75, 2),
        transfer_minutes=round(overhead * 0.25, 2),
        other_phases=(Phase("probes_train_eval_score_rescore_expected",
                            round(max(0.0, expected_total - overhead), 2)),),
        contingency_fraction=contingency,
        soft_stop_reserves=(Phase("probe_overrun_risk", reserve),),
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
    return D3ProbesAuthorization.load(path).hard_runtime_minutes


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-d3-probes",
        schema="aadistill.autoinit.d3_probes_session/v1",
        description=PROBES.description,
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        authorization_loader=D3ProbesAuthorization.load,
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
                           "TRAIN_ENV", "TEACHER_READY", "VLLM_READY",
                           "TESTS_OK", "AUTHORIZATION_OK", "SETUP_DONE"),
            local_assets=SCIENCE_ASSETS,
            relay_inputs=relay_inputs(REPO_ROOT),
            teacher_revision=D1S.root_teacher_identity(REPO_ROOT)["revision"],
            test_paths=TEST_PATHS,
            uv_max_seconds=getattr(args, "uv_max_s", 1500),
            tests_max_seconds=getattr(args, "tests_max_s", 2700)),
        driver_command=driver_command,
        materialize_inputs=partial(D2L.restore_trained_probes, rung=PROBES),
        driver_job_id="autoinit_d3_probes_driver",
        status_path=PROBES.status,
        run_log_path=PROBES.run_log,
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note=("the session stopped before its rung completed; "
                          "every scored probe and trained checkpoint is "
                          "secured first")),
        host_admission=D2L.host_admission,
        precheck=prechecks(args),
        artifacts=ArtifactPolicy(
            audit_dirname=PROBES.audit_dirname,
            evidence_filename=PROBES.evidence_filename,
            archive_basename="d3_probes_artifacts.tar.gz",
            spec_success="configs/stages/stage-1/phase_d3/d3_probes_artifacts.json",
            spec_failed=("configs/stages/stage-1/phase_d3/"
                         "d3_probes_artifacts_failed.json"),
            report_names=(PROBES.evidence_filename,),
            on_poll=partial(D2L.secure_probe_evidence, rung=PROBES),
            fetch_products=partial(D2L.fetch_probe_evidence, rung=PROBES),
            fetch_result_ok=D1L.probe_fetch_ok,
            products_secured=partial(D2L.probes_evidence_secured, rung=PROBES)),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
    )


build_parser = D2L.build_parser


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
    except (AuthorizationError, P.D3ProbesError) as exc:
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
                           summary=("D3 probes: every scored probe's evidence "
                                    "and the session record are what it owes; "
                                    "the exploratory comparison is computed "
                                    "off-pod."))
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11


if __name__ == "__main__":
    raise SystemExit(main())
