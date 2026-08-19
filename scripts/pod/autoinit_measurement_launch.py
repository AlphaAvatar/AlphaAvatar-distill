#!/usr/bin/env python3
"""The bounded causal-depth runtime and backend measurement, as a session spec.

    PYTHONPATH=src setsid nohup python -u \
        scripts/pod/autoinit_measurement_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

**This is not Phase-A attempt 11.** It runs no greedy search, selects no depth
map, writes no checkpoint and has no follow-on. It exists to answer three
questions a CPU box cannot:

1. does the repaired port reach E8a's measured 12.0 evaluations/min now that the
   causal-depth reduction is back on the accelerator?
2. what is the real peak VRAM, and which way does the production reference-cache
   gate decide at the frozen mixture?
3. does the repaired port compute what E8a computes, per item, on one GPU?

It names `SpendAuthorization`, whose `allows_phase_a` is a hard `False`, so this
session **cannot** start Phase A whatever artifact it is pointed at — a property
of the type, not a promise in a comment. `fetch_products` is empty: no weights
come home because none are produced.

Attempt 10 is why this session exists at all. It spent $11.43 discovering that
the port scored on the host; the repair put the reduction back where E8a keeps
it, and the equivalence was proved at $0. What remains is whether the ported code
achieves the ancestor's throughput on real hardware — which is exactly the class
of thing attempt 10 proved a CPU run cannot tell you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
# The sibling science-input declarations. Present when this file is run
# directly; absent when a test loads it by path, which is how the structural
# checks load every launcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.autoinit.authorization import SpendAuthorization  # noqa: E402
from aadistill.autoinit.measurement import MEASUREMENT_PLAN_V1  # noqa: E402
from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, MarkerPolicy, SessionContext, SessionSpec,
    SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
from autoinit_science_inputs import (  # noqa: E402
    CALIBRATION_V1, CANONICAL_INIT, RECOVERY_LADDER,
)

STATUS = f"{WS}/autoinit_measurement.status"
RUN_LOG = f"{WS}/autoinit_measurement_run.log"
AUTH_PATH = "logs/autoinit_measurement_authorization.json"
#: Nothing is scp'd. The measurement reads the frozen calibration and the teacher,
#: both from the relay, so it declares no dev-box asset and — since 2026-08-18 —
#: is therefore given none.
LOCAL_ASSETS: tuple = ()
TEST_IGNORES = ("tests/data/test_recovery_corpus_pipeline.py",
                "tests/pod/test_phase_a_stages1_5_execute.py")
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: From `logs/autoinit_causal_depth_pricing_bound.json`. The measurement itself is
#: ~2.4 min of evaluations at E8a's rate; everything else is session overhead,
#: which is why the ceiling is dominated by setup and load rather than by work.
SETUP_MINUTES = 12.0
TEACHER_LOAD_MINUTES = 8.0
MEASURE_MINUTES = 8.0
MANIFEST_MINUTES = 4.0
SYNC_MINUTES = 3.0
TRANSFER_MINUTES = 5.0


def driver_command(ctx: SessionContext, plan) -> str:
    """The corrected measurement job, invoked with its defaults.

    The defaults ARE the reviewed design: 3 samples at each cardinality 1-8,
    weighted extrapolation, the production `_ReferenceLogits` policy, the frozen
    teacher revision, exactly two E8a pairs with `cache_reference=False`. Nothing
    here overrides them, so the reviewed job and the executed job are the same
    job.
    """
    return (f"/opt/train/bin/python "
            f"{REPO}/scripts/autoinit/measure_causal_depth_runtime.py "
            f"--out artifacts/audit/autoinit_measurement/result.json")


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-measurement",
        schema="aadistill.autoinit.measurement_session/v1",
        description=("bounded causal-depth runtime and backend measurement: a "
                     "rate and a per-item comparison against E8a. No search, no "
                     "depth map, no checkpoint, no follow-on"),
        authorization_path=AUTH_PATH,
        #: The ordinary type. `allows_phase_a` is a hard False, so a measurement
        #: pointed at a Phase-A artifact refuses it rather than running it.
        authorization_loader=SpendAuthorization.load,
        plan_id=MEASUREMENT_PLAN_V1.plan_id,
        plan_hash=MEASUREMENT_PLAN_V1.plan_hash,
        budget=BudgetSpec(
            arms=0, steps_per_arm=0,          # nothing is trained
            step_seconds=4.15,
            step_source="unused; the measurement trains nothing",
            setup_minutes=args.setup_minutes,
            transfer_minutes=args.transfer_minutes,
            other_phases=(
                Phase("teacher_load", TEACHER_LOAD_MINUTES),
                Phase("timed_evaluations_and_e8a_pairs", MEASURE_MINUTES),
                Phase("artifact_manifest_and_verify", MANIFEST_MINUTES),
                Phase("artifact_synchronization", SYNC_MINUTES)),
            eval_minutes_per_arm=0.0, contingency_fraction=0.10,
            artifact_recovery_reserve_minutes=10.0),
        setup=SetupManifest(
            #: The same ten science inputs every session stages. The measurement
            #: reads the calibration and the teacher; the canonical init and the
            #: recovery pack are staged because the shared setup stages what a
            #: session declares and this session declares what setup will do.
            relay_inputs=(*CANONICAL_INIT, *RECOVERY_LADDER, *CALIBRATION_V1),
            local_assets=LOCAL_ASSETS,
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "VLLM_READY",
                           "TEACHER_READY", "ROPE_OK", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_measurement",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            #: Two distinct failures, because they mean different things.
            #: MEASUREMENT_FAILED is a backend disagreement or a missing device:
            #: the finding. MEASUREMENT_FALLBACK is the reference cache
            #: recomputing, which roughly doubles the forwards and invalidates
            #: the cost basis this session exists to establish.
            failure=("MEASUREMENT_FAILED", "MEASUREMENT_FALLBACK"),
            #: Both write the report before they stop, and the report is the
            #: whole artifact — so both are "incomplete" in the sense that
            #: matters: what they produced must still come home.
            incomplete=("MEASUREMENT_FAILED", "MEASUREMENT_FALLBACK"),
            failure_note=("the measurement stopped on one of its stated "
                          "conditions — collecting the report, which carries the "
                          "timings, the cache decision and the per-item deltas, "
                          "then tearing down. Nothing is retried.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_measurement",
            evidence_filename="result.json",
            archive_basename="measurement_artifacts.tar.gz",
            spec_success="configs/autoinit/measurement_artifacts.json",
            spec_failed="configs/autoinit/measurement_artifacts.json",
            report_names=("result.json",),
            event_streams=lambda ctx: (),
            #: Empty. The measurement produces no weights, by design.
            fetch_products=lambda ctx: []),
        teardown=TeardownPolicy(note="nothing chains off the measurement"),
        precheck=(),
        evidence_fields={"scientific_use": False,
                         "trains_anything": False,
                         "runs_greedy_search": False,
                         "selects_a_depth_map": False,
                         "writes_a_checkpoint": False,
                         "retrains_permanent_controls": False,
                         "phase_a_launched": False,
                         "phase_a_reachable_from_this_launcher": False,
                         "followon_started": False,
                         "purpose": ("measure the repaired port's evaluations/min, "
                                     "peak VRAM and cache decision, and compare it "
                                     "to E8a per item on one accelerator")})


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test asserts on the namespace it produces
    rather than on a transcription of it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=0.99)
    ap.add_argument("--disk-gb", type=int, default=120)
    ap.add_argument("--setup-minutes", type=float, default=SETUP_MINUTES)
    ap.add_argument("--transfer-minutes", type=float, default=TRANSFER_MINUTES)
    ap.add_argument("--token-src",
                    default=str(Path.home() / ".cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--poll-limit-min", type=float, default=90.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=str(Path.home() / ".runpod/config.toml"))
    ap.add_argument("--out", default="logs/autoinit_measurement_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("STOP for review. The measured values are inputs "
                                "to a repricing and a separate budget decision; "
                                "they authorize nothing."))


if __name__ == "__main__":
    raise SystemExit(main())
