#!/usr/bin/env python3
"""The D2 behavioural SCREENING launcher. One session, declared completely.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/autoinit_d2_screening_launch.py \
        --scr <scr> --session-commit <sha> --bundle <canonical> \
        --run-id d2_screening_<utc>

The shared `SessionRunner` owns acquisition, the watchdog, teardown on every
exit path and collection before teardown. This file owns what makes the
session D2-screening's, and REUSES D1's behavioural machinery wherever the
question is the same:

* DURABILITY is D1's: every scored probe's result, per-sample rows, admission
  and generations come off the pod on every poll; every TRAINED checkpoint is
  preserved off-pod and identity-verified before the pod may release it; a
  replacement pod restores preserved probes and resumes them at evaluation
  (P8.4 state 2); preserved weights are released only at a successful
  terminal. The functions are `autoinit_d1_behavioural_launch`'s, called with
  D2's `BehaviouralLayout`, so the two sessions never share a path.
* THE ARMS ARE NOT REBUILT. Each is a committed search product persisted on
  the relay (`d2_product_durability.py`), staged during setup as a pinned
  relay input and verified at all four identities by the driver.
* THE HOST must be the one B's imported units were trained and generated on:
  the shared protocol pins the training image digest -- which carries the host
  driver version -- so `host_admission` admits exactly that identity and the
  runner redraws otherwise. A probe trained elsewhere would not be paired with
  B's rows.

The ranking is NOT computed on the pod: the driver stops at preservation and
`rank_d2_screening.py` applies the frozen rule off-pod at `$0`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for extra in ("src", "scripts", "scripts/stages/stage-1"):
    p = str(REPO_ROOT / extra)
    if p not in sys.path:
        sys.path.insert(0, p)

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, LocalAsset, MarkerPolicy,
    RelayInput, SessionSpec, SetupManifest, TeardownPolicy,
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
from stages.phase_d2 import d2_screening as SC  # noqa: E402
from stages.phase_d2.d2_screening_authorization import (  # noqa: E402
    D2ScreeningAuthorization,
)

#: DECLARED, and `autoinit_preflight_setup.sh` has the matching branch.
SESSION_KIND = "d2_screening"
ACCOUNT_OPERATIONAL_RESERVE_USD = 5.0
#: The pod's blocking gate runs THIS preflight, positively declared: the rest
#: of `phase_d2/tests` covers the search chain and its staged assets.
TEST_PATHS = ("scripts/stages/stage-1/phase_d2/tests/screening_pod_preflight",)
REPO = "/workspace/aad"
AUDIT_DIRNAME = "autoinit_d2_screening"
EVIDENCE_DIR = f"artifacts/audit/{AUDIT_DIRNAME}"
EVIDENCE_FILENAME = "d2_screening.json"
ATTESTED_NAME = "d2_screening_attested_protocol.json"
STATUS = "/workspace/autoinit_d2_screening.status"
RUN_LOG = "/workspace/autoinit_d2_screening_run.log"
#: Where probes train and where their evaluation packages go, on the pod.
PROBE_ROOT_REL = "artifacts/stages/stage-3/d2_screening"
EVAL_PACKAGES_REL = "artifacts/stages/stage-3/d2_screening_eval"
RELEASE_ACK_REL = f"{EVIDENCE_DIR}/release_acks"
DRIVER = "scripts/stages/stage-1/phase_d2/autoinit_d2_screening_driver.py"
STORE_SUBDIR = D1L.STORE_SUBDIR

#: D2's layout for D1's durability functions. The rescoring diagnostic is
#: secured beside the strict evidence, never as a gate; the training records
#: the off-pod ranking reads are STRICT (secured, verified, and required before
#: a release, a prune or "evidence complete").
LAYOUT = D1L.BehaviouralLayout(
    evidence_dir=EVIDENCE_DIR, evidence_filename=EVIDENCE_FILENAME,
    attested_name=ATTESTED_NAME,
    pod_probe_root=f"{REPO}/{PROBE_ROOT_REL}",
    release_ack_rel=RELEASE_ACK_REL,
    extra_probe_files=(("rescoring.json", "rescoring/{probe_id}.json",
                        False),),
    required_training_records=SC.TRAINING_RECORDS)

#: Both battery roles travel (~11 MiB), because the cheap preflight re-hashes
#: the screening role and shows the two roles disjoint on the pod.
BATTERY_INSTALL = ("artifacts/stages/stage-1/families/d_series/batteries/"
                   "d_series_behavioural_v1")
SCIENCE_ASSETS: tuple[LocalAsset, ...] = (
    LocalAsset(repo_path=f"{BATTERY_INSTALL}/d1_screening",
               dest_name="d1_screening", install_to=BATTERY_INSTALL),
    LocalAsset(repo_path=f"{BATTERY_INSTALL}/d1_confirmation",
               dest_name="d1_confirmation", install_to=BATTERY_INSTALL),
)


def arm_relay_inputs(repo_root: Path = REPO_ROOT) -> tuple[RelayInput, ...]:
    """Each arm's three files, pinned to the bytes the durability record
    verified locally AND after a relay round trip. No record, no arms."""
    from stages.phase_d2.d2_product_durability import RECORD_REL, RELAY_PREFIX

    path = repo_root / RECORD_REL
    if not path.is_file():
        raise SC.D2ScreeningError(
            f"{RECORD_REL} is missing: the arms have not been shown durable "
            "on the relay, so a pod has nothing verified to stage")
    record = json.loads(path.read_text())
    local = {p["state_id"]: p for p in
             record["local_verification"]["products"]}
    restored = {r["state_id"] for r in
                record["relay_verification"]["restored_checkpoints"]
                if r.get("matched")}
    out = []
    for arm in SC.arms(repo_root):
        if arm.state_id not in local or arm.state_id not in restored:
            raise SC.D2ScreeningError(
                f"{arm.arm_id} ({arm.state_id}) is not in the durability "
                "record as both locally verified and relay round-tripped")
        files = local[arm.state_id]["files"]
        if files["model.safetensors"]["sha256"] != \
                arm.identities["single_shard_sha256"]:
            raise SC.D2ScreeningError(
                f"{arm.arm_id}: the durability record's shard is not the "
                "selection's single_shard_sha256")
        for name in SC.ARM_FILES:
            out.append(RelayInput(
                f"{RELAY_PREFIX}/products/{arm.state_id}/{name}",
                repo=record["locations"]["relay"]["repo"],
                dest=f"{SC.POD_ARMS_REL}/{arm.state_id}",
                sha256=files[name]["sha256"]))
    return tuple(out)


def relay_inputs(repo_root: Path = REPO_ROOT) -> tuple[RelayInput, ...]:
    return (*RECOVERY_LADDER, *C1_EVAL_TOKENIZER, *arm_relay_inputs(repo_root))


def run_dir_for(run_id: str | None) -> str:
    return rel_run_dir(SC.EXPERIMENT_ID, run_id or "unrecorded", SC.STAGE_ID)


def auth_path_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/authorization.json"


def bundle_record_for(run_id: str | None) -> str:
    return f"{run_dir_for(run_id)}/governance/bundle.json"


def driver_command(ctx: Any, plan: Any) -> str:
    run_id = getattr(ctx.args, "run_id", "unrecorded")
    return (f"/opt/train/bin/python {REPO}/{DRIVER} "
            f"--out '{REPO}/{EVIDENCE_DIR}' --run-id '{run_id}' "
            f"--authorization '{REPO}/{auth_path_for(run_id)}' "
            f"--arm-root '{REPO}/{SC.POD_ARMS_REL}' "
            f"--status '{STATUS}' --device cuda "
            f"--image-digest '{ctx.image_digest}'")


# --- host admission -----------------------------------------------------------

def host_admission(image_digest: str) -> tuple[bool, str]:
    """Admit only the image identity B's imported units were trained under.

    The shared protocol pins `training_image_digest` -- image AND host driver
    version, fused -- inside the training runtime digest every new candidate
    must equal, and the generation runtime digest of the rung likewise. A host
    on another driver would train probes that are not paired with B's rows.
    Exact equality, read from the protocol record, never typed here.
    """
    want = SC.protocol(REPO_ROOT)["recovery"]["training_image_digest"]
    if image_digest == want:
        return True, f"image identity {image_digest} is the protocol's"
    return False, (f"image identity {image_digest!r} is not the shared "
                   f"protocol's {want!r}; probes trained here would not be "
                   "paired with B's imported units. Redraw.")


# --- durability: D1's functions under D2's layout ----------------------------

def secure_probe_evidence(ctx) -> None:
    D1L.secure_probe_evidence(ctx, LAYOUT)


def fetch_probe_evidence(ctx) -> list:
    return D1L.fetch_probe_evidence(ctx, LAYOUT)


def probes_evidence_secured(ctx, fetched) -> tuple[bool, str]:
    return D1L.probes_evidence_secured(ctx, fetched, LAYOUT)


def restore_trained_probes(ctx) -> bool:
    return D1L.restore_trained_probes(ctx, LAYOUT)


# --- the $0 gates ---------------------------------------------------------------

def run_identity_gate(ctx) -> tuple[bool, str]:
    try:
        ctx.auth.require_run_id(getattr(ctx.args, "run_id", ""))
        ctx.auth.require_plan(SC.preregistration(REPO_ROOT)[
            "preregistration_hash"])
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, (f"run {ctx.auth.run_id}, preregistration "
                  f"{ctx.auth.plan_hash[:12]}")


def contract_gate(ctx) -> tuple[bool, str]:
    """Derive the REAL contract here -- the arms' bytes hashed, the protocol's
    trainer/generator/scorer pins compared -- and require the bound hash."""
    try:
        contract = SC.session_contract(REPO_ROOT)
        got = SC.contract_hash(contract)
        ctx.auth.require_contract(got)
    except Exception as exc:                                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    ctx.evidence["d2_screening_contract_check"] = {
        "contract_hash": got, "n_probes": contract["n_probes"],
        "seeds": contract["seeds"],
        "evaluation_protocol_hash": contract["evaluation"][
            "evaluation_protocol_hash"]}
    return True, f"contract verified at $0: {got[:16]}, {contract['n_probes']} probes"


def relay_arms_gate(ctx) -> tuple[bool, str]:
    """Every arm file is on the relay at its pinned size and LFS sha256. Read
    from the hub's own listing; nothing is downloaded."""
    from stages.phase_d2.d2_product_durability import remote_listing

    try:
        inputs = arm_relay_inputs(REPO_ROOT)
        listing = remote_listing()
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
        return False, f"arms not obtainable from the relay: {problems[:3]}"
    ctx.evidence["d2_screening_relay_arms"] = [r.as_record() for r in inputs]
    return True, f"{len(inputs)} arm files on the relay at their pins"


def bundle_staged_gate(ctx) -> tuple[bool, str]:
    import tempfile

    from aadistill.infrastructure.bundle_transport import hf_download, roundtrip

    commit = getattr(ctx.args, "session_commit", "")
    try:
        D1S.require_canonical_bundle(getattr(ctx.args, "bundle", ""), commit)
        record = json.loads((REPO_ROOT / bundle_record_for(
            getattr(ctx.args, "run_id", None))).read_text())
        auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
        with tempfile.TemporaryDirectory(prefix="d2s-bundle-") as tmp:
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


RUNS_ROOT = REPO_ROOT / "logs/stages/stage-1/phase_d2/runs"


def _prior_attempt_logs() -> list[tuple[str, str]]:
    if not RUNS_ROOT.is_dir():
        return []
    out = []
    for d in sorted(p for p in RUNS_ROOT.iterdir()
                    if p.is_dir() and p.name.startswith("d2_screening")):
        log = d / "runtime" / "launcher.log"
        if log.is_file():
            out.append((d.name, log.read_text(errors="replace")))
    return out


def _corrective_change_since() -> tuple[bool, str]:
    """D1's rule: anything tracked outside the run directories changed since
    the previous attempt's session commit."""
    logs = _prior_attempt_logs()
    if not logs:
        return True, "no prior screening attempt exists"
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
                   if not p.startswith("logs/stages/stage-1/phase_d2/runs/")]
        return (bool(outside), f"{len(outside)} tracked path(s) changed since "
                f"{name}" if outside else
                f"only run records changed since {name}")
    return False, ("a prior screening attempt exists and its session commit "
                   "cannot be read")


def prechecks(args) -> tuple:
    return (run_identity_gate, contract_gate, relay_arms_gate,
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", None)),
                                check_lineage=True),
            same_failure_gate(_prior_attempt_logs,
                              corrective_change=_corrective_change_since),
            bundle_staged_gate)


def budget_spec(repo_root: Path, *,
                authorized_minutes: float | None = None) -> BudgetSpec:
    """DECOMPOSES the priced bound (capped by the authorized minutes) so the
    plan lands exactly on it; refuses rather than thinning a reserve."""
    from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase

    from stages.phase_d1.d1_authorization import live_money
    from stages.phase_d2.d2_screening_authorization import session_ceiling

    priced = session_ceiling(repo_root)
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
        step_source=("unused: eight probes from relay-staged arms, every "
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
    return D2ScreeningAuthorization.load(path).hard_runtime_minutes


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-d2-screening",
        schema="aadistill.autoinit.d2_screening_session/v1",
        description=(
            "D2 behavioural SCREENING on d_series_shared_eval@v1: the four "
            "committed D2 initializations, staged from the relay and verified "
            "at all four identities, each trained under the frozen recovery "
            "recipe at the two shared screening seeds, generated through the "
            "unchanged generator, admitted against the shared rung's "
            "evaluation protocol, scored by the pinned scorer, and "
            "rollout-rescored as a diagnostic. Stops at preservation; the "
            "ranking against B's imported units is computed off-pod."),
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        authorization_loader=D2ScreeningAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/shared/pod/watchdog.py",
            setup_script="scripts/shared/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/shared/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=f"autoinit.v1.{SC.EXPERIMENT_ID}.screening",
        plan_hash=SC.preregistration(REPO_ROOT)["preregistration_hash"],
        budget=budget_spec(REPO_ROOT,
                           authorized_minutes=_authorized_minutes(args)),
        setup=SetupManifest(
            env={"SESSION_KIND": SESSION_KIND},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_KIND", "TEACHER_REVISION"),
            #: VLLM_READY: probes generate through vLLM. TEACHER_READY: the KD
            #: trainer and the rescoring read the pinned teacher.
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
        materialize_inputs=restore_trained_probes,
        driver_job_id="autoinit_d2_screening_driver",
        status_path=STATUS,
        run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE", failure=("RUN_FAILED",),
            products_eligible=lambda terminal, stages: True,
            failure_note=("the session stopped before its rung completed; "
                          "every scored probe and trained checkpoint is "
                          "secured first")),
        host_admission=host_admission,
        precheck=prechecks(args),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename=EVIDENCE_FILENAME,
            archive_basename="d2_screening_artifacts.tar.gz",
            spec_success="configs/stages/stage-1/phase_d2/d2_screening_artifacts.json",
            spec_failed=("configs/stages/stage-1/phase_d2/"
                         "d2_screening_artifacts_failed.json"),
            report_names=(EVIDENCE_FILENAME,),
            on_poll=secure_probe_evidence,
            fetch_products=fetch_probe_evidence,
            fetch_result_ok=D1L.probe_fetch_ok,
            products_secured=probes_evidence_secured),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Module scope; every required argument a STRING (the dispatch probe)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--runpod-config",
                    default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--relay-repo", default=MAIN_RELAY)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--token-src",
                    default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    #: FIVE draws: admission wants one exact host driver, which the D1 and D2
    #: sessions drew on 6 of 10 hosts; a refused draw costs one startup.
    ap.add_argument("--host-draws", type=int, default=5)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=None)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--ckpt-store", default=None,
                    help="where trained probes are preserved; <scr>/products")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=45)
    ap.add_argument("--restore-trained", default=None,
                    help="preserved trained probes to push to a REPLACEMENT "
                         "pod; the driver resumes each at evaluation")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def resolve_operational_defaults(args) -> dict[str, Any]:
    from stages.phase_d2.d2_screening_authorization import session_ceiling

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
                           summary=("D2 screening: every scored probe's "
                                    "evidence and the session record are what "
                                    "it owes; the ranking is computed "
                                    "off-pod."))
    except AuthorizationError as exc:
        print(f"REFUSED at $0: {exc}\nNo provider resource was created.")
        return 11


if __name__ == "__main__":
    raise SystemExit(main())
