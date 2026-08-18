#!/usr/bin/env python3
"""The AutoInitializer micro-preflight, as a session specification.

    PYTHONPATH=src setsid nohup python -u scripts/pod/autoinit_preflight_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

This file declares WHAT the session is. How a session is run — detached start
with a durable descriptor, an independent provider-side watchdog, continuous log
relay, four budget thresholds, the artifact gate and provider-confirmed teardown
— lives once, in `aadistill.infrastructure.session_runner`, and is not
inherited, subclassed or retargeted by anybody.

Until 2026-08-18 this file WAS the machinery, and three other sessions were
subclasses of it that mutated its module globals to point it somewhere else.
That cost three paid pods, every one to the same defect: a session inheriting a
requirement it never declared. Nothing here is inherited now, so nothing can be
inherited silently.

Phase A is not reachable from this session. It names `SpendAuthorization`, whose
`allows_phase_a` is a hard `False`, so that is a property of the declaration
rather than a promise about the code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.authorization import SpendAuthorization  # noqa: E402
from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1  # noqa: E402
from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, LocalAsset, MarkerPolicy, RelayInput,
    SessionContext, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402

STATUS = f"{WS}/autoinit_preflight.status"
RUN_LOG = f"{WS}/autoinit_preflight_run.log"
AUTH_PATH = "logs/autoinit_micro_preflight_authorization.json"
CONTROLS = ("preflight_ctl_r0860k_sa", "preflight_ctl_r0860k_sb")

#: Dev-box-only artifacts the pod cannot fetch from the relay (~1.6 MB). Each one
#: says where it is installed, so the shared setup script does not have to know
#: their names — which is the change that makes a session declaring no assets
#: actually get none.
LOCAL_ASSETS = (
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage3/recovery_search_v2", "recovery_search_v2",
               "artifacts/stage3"),
)
#: Ignored by the pod's blocking test gate. Must stay equal to the pod
#: simulator's list, and a test pins them equal.
TEST_IGNORES = ("tests/data/test_recovery_corpus_pipeline.py",
                "tests/pod/test_phase_a_stages1_5_execute.py")
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


def fetch_controls(ctx: SessionContext) -> list:
    """Fetch what this session PRODUCED and cannot regenerate for free.

    The preflight trains two permanent controls; they are the only such artifact,
    and they are fetched whenever they EXIST rather than only on a fully
    successful session. `if terminal == "ALL_DONE"` deleted $2.82 of verified
    controls on 2026-08-13 for want of that distinction.
    """
    fetched: list = []
    if not ctx.stage2_passed:
        return fetched
    for name in CONTROLS:
        dest = Path(ctx.args.ckpt_store) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            ["timeout", f"{ctx.args.ckpt_fetch_limit_min}m", "scp", "-r",
             "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"root@{ctx.host}:{REPO}/artifacts/stage3/{name}/checkpoints",
             str(dest)], capture_output=True, timeout=None)
        size = sum(f.stat().st_size for f in dest.rglob("*")
                   if f.is_file()) if dest.exists() else 0
        fetched.append({"control": name, "rc": rc.returncode,
                        "bytes": size, "dest": str(dest)})
        ctx.say(f"  checkpoint {name}: rc={rc.returncode}, "
                f"{size / 2**30:.2f} GiB -> {dest}")
    return fetched


def control_streams(ctx: SessionContext) -> tuple[str, ...]:
    """Append-only streams a torn-down session may have left mid-write."""
    return tuple(f"artifacts/stage3/{c}/train_log.jsonl" for c in CONTROLS)


def control_relay(ctx: SessionContext) -> tuple[tuple[str, str], ...]:
    return tuple((f"{REPO}/artifacts/stage3/{c}/train_log.jsonl",
                  f"{c}.train_log.jsonl") for c in CONTROLS)


def driver_command(ctx: SessionContext, plan) -> str:
    return (f"/opt/train/bin/python scripts/pod/autoinit_preflight_driver.py "
            f"--stage all --image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price} --spent-usd {ctx.spent_usd:.3f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.2f} "
            f"--authorized-usd {plan.hard_terminate_usd:.2f}")


def spec(args) -> SessionSpec:
    """The whole session, in one object. Nothing about it lives anywhere else."""
    return SessionSpec(
        session_id="autoinit-preflight",
        schema="aadistill.autoinit.preflight_session/v1",
        description=("the AutoInitializer micro-preflight: attestation, machine "
                     "gates, two permanent controls, and their characterization"),
        authorization_path=AUTH_PATH,
        authorization_loader=SpendAuthorization.load,
        plan_id=PREFLIGHT_PLAN_V1.plan_id,
        plan_hash=PREFLIGHT_PLAN_V1.plan_hash,
        budget=BudgetSpec(
            arms=2, steps_per_arm=1023,
            step_seconds=4.15,
            step_source=("E6b measured 4.15 s/step for this exact model, rung "
                         "and card"),
            setup_minutes=45.0, transfer_minutes=6.0,
            other_phases=(
                Phase("stage0_attestation_and_engine_probe", 8.0),
                Phase("stage1_machine_gates", 22.0),
                Phase("stage3_characterization_two_seeds", 36.0),
                Phase("artifact_manifest_and_verify", 8.0),
                Phase("artifact_synchronization", 6.0)),
            eval_minutes_per_arm=0.0, contingency_fraction=0.10,
            artifact_recovery_reserve_minutes=30.0),
        setup=SetupManifest(
            relay_inputs=(
                RelayInput("stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors"),
                RelayInput("stage3_recovery_corpus_v2/ladder_uniform/blocks.npz"),
            ),
            local_assets=LOCAL_ASSETS,
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "VLLM_READY",
                           "TEACHER_READY", "ROPE_OK", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_preflight_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("PREFLIGHT_FAILED", "PREFLIGHT_INCOMPLETE"),
            incomplete=("PREFLIGHT_INCOMPLETE",),
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. Permanent controls were not trained "
                          "under a configuration that has to change.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_preflight",
            evidence_filename="preflight_evidence.json",
            archive_basename="preflight_artifacts.tar.gz",
            spec_success="configs/autoinit/preflight_artifacts.json",
            spec_failed="configs/autoinit/preflight_artifacts_failed.json",
            report_names=("preflight_evidence.json", "attested_protocol.json",
                          "materialized_thresholds.json"),
            event_streams=control_streams,
            fetch_products=fetch_controls,
            extra_relay_streams=control_relay),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
        #: No commit gate. The preflight has never had one — the continuation and
        #: Phase A each grew their own after attempt 5 — and this declaration is
        #: where that asymmetry becomes visible instead of being spread over
        #: three files. Adding one is a behaviour change and belongs in its own
        #: decision, not in a refactor that must reproduce the existing prices.
        precheck=(),
        evidence_fields={"preflight_plan_hash": PREFLIGHT_PLAN_V1.plan_hash,
                         "phase_a_launched": False,
                         "phase_a_reachable_from_this_launcher": False})


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test can assert on the namespace it
    produces rather than on a transcription of it."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=0.99)
    ap.add_argument("--disk-gb", type=int, default=150)
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--ckpt-store", default="/home/ecs-user/aad-artifacts/autoinit")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=25)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=420.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/autoinit_preflight_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary="Phase A NOT launched.")


if __name__ == "__main__":
    raise SystemExit(main())
