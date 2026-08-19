#!/usr/bin/env python3
"""The Stage-1 device canary, as a session specification. TERMINATED.

    PYTHONPATH=src setsid nohup python -u \
        scripts/pod/autoinit_device_canary_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

**The paid device-canary session path is TERMINATED (2026-08-18.)** Two
authorized sessions, $0.1240, **zero canary runs**: attempt 1 died before setup
on three inherited `self.a` attributes this launcher's parser had never defined,
and the retry died inside setup because it had honestly declared
`LOCAL_ASSETS = ()` and the shared setup script copied two assets anyway. Neither
reached the canary script, so neither says anything about device placement on
CUDA. No further canary is prepared, and converting this file does not prepare
one: it has no authorization artifact to load.

It is kept, converted, for two reasons. It is the workload description — one
invocation of each frozen operator on real CUDA, through the production
materialize -> reload -> validate -> measure lifecycle — and it is the smallest
session in the repository, which makes it the one that shows what the
specification form actually removed:

* the argument contract is checked before a pod exists, so the three attributes
  that killed attempt 1 cannot be missing silently;
* `local_assets` is a manifest field the setup script READS, so declaring none
  now means none are copied, which is exactly what the retry needed;
* `SESSION_KIND` cannot leak in from another session, because there is no module
  global for it to leak through.

It names `SpendAuthorization`, whose `allows_phase_a` is a hard `False`, so this
session cannot start Phase A whatever it is pointed at. Nothing it produces may
enter scientific selection: there is no `fetch_products` and no checkpoint comes
home. Its children are compressions of the canonical student toward a geometry
picked to make every operator do work, and they die with the pod.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
# The sibling science-input declarations. Present when this file is run
# directly; absent when a test loads it by path, which is how the
# structural checks load every launcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.autoinit.authorization import SpendAuthorization  # noqa: E402
from aadistill.autoinit.recovery import PreflightPlan, PreflightStage  # noqa: E402
from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, LocalAsset, MarkerPolicy, RelayInput,
    SessionContext, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
from autoinit_science_inputs import (  # noqa: E402
    CALIBRATION_V1, CANONICAL_INIT, RECOVERY_LADDER,
)

STATUS = f"{WS}/autoinit_device_canary.status"
RUN_LOG = f"{WS}/autoinit_device_canary_run.log"
AUTH_PATH = "logs/autoinit_device_canary_authorization.json"
#: The canary reads the frozen calibration mixture and the canonical student.
#: Both come from the relay; neither is a dev-box-only asset, so nothing is
#: scp'd and no asset is installed. Under the old shared setup that declaration
#: was ignored and cost $0.0637; `SESSION_ASSETS` is now what setup reads.
#: The two frozen assets the SHARED SETUP verifies. The canary reads neither —
#: which is exactly what it declared in 2026-08-18, and exactly why it died in
#: setup at $0.0637. That fix stopped the setup COPYING undeclared assets; it did
#: not correct this declaration, so this session would still fail the
#: unconditional `verify_frozen_assets.py` gate. TERMINATED means no further
#: canary is prepared, not that its specification may misdescribe the run it
#: would perform.
LOCAL_ASSETS = (
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage3/recovery_search_v2", "recovery_search_v2",
               "artifacts/stage3"),
)
TEST_IGNORES = ("tests/data/test_recovery_corpus_pipeline.py",
                "tests/pod/test_phase_a_stages1_5_execute.py")
#: Forwarded to the shared setup script. The canary needs no teacher, but setup
#: is shared and its behaviour must not change because a canary is driving it,
#: so this is the same frozen revision every other session pins.
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: One stage. The plan exists because the authorization binds to a plan hash, and
#: a canary that borrowed the preflight's hash would be claiming to be a
#: preflight.
CANARY_PLAN_V1 = PreflightPlan(
    plan_id="autoinit.device_canary",
    version=1,
    stages=(
        PreflightStage(
            stage=0, name="operator device canary", blocking=True,
            purpose=("run each frozen operator once on real CUDA, through the "
                     "production materialize -> canonical reload -> validate -> "
                     "measure lifecycle, and report where every tensor was"),
            produces=("per-operator pass/fail with the traceback on failure",
                      "the validation and measurement device of each lifecycle",
                      "the resident statistics-cache device and size",
                      "peak VRAM per operator"),
            stop_conditions=(
                "any operator raises -> STOP: that is the finding",
                "no CUDA device -> STOP: a CPU run certifies nothing")),),
)

#: Priced from the project's own cost model: the compute is ~1 s, so this is
#: session overhead and nothing else. See `logs/autoinit_stage1_device_audit.json`.
SETUP_MINUTES = 11.0
PARENT_LOAD_MINUTES = 3.0
CANARY_MINUTES = 8.0
MANIFEST_MINUTES = 5.0
SYNC_MINUTES = 5.0
TRANSFER_MINUTES = 6.0


def driver_command(ctx: SessionContext, plan) -> str:
    return (f"/opt/train/bin/python "
            f"{REPO}/scripts/pod/autoinit_device_canary.py "
            f"--out {REPO}/artifacts/audit/autoinit_device_canary/result.json "
            f"--workdir artifacts/autoinit/device_canary --device cuda")


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-device-canary",
        schema="aadistill.autoinit.device_canary_session/v1",
        description=("one invocation of each frozen operator on real CUDA, "
                     "through the production lifecycle. TERMINATED path"),
        authorization_path=AUTH_PATH,
        #: The ordinary type, deliberately. Its `allows_phase_a` is a hard False,
        #: so this session cannot start Phase A even if pointed at the wrong
        #: artifact. The canary is infrastructure, not science.
        authorization_loader=SpendAuthorization.load,
        plan_id=CANARY_PLAN_V1.plan_id,
        plan_hash=CANARY_PLAN_V1.plan_hash,
        budget=BudgetSpec(
            arms=0, steps_per_arm=0,          # nothing is trained
            step_seconds=4.15,
            step_source="unused; the canary trains nothing",
            setup_minutes=args.setup_minutes,
            transfer_minutes=args.transfer_minutes,
            other_phases=(
                Phase("parent_load_and_calibration", PARENT_LOAD_MINUTES),
                Phase("six_operator_invocations_and_lifecycles", CANARY_MINUTES),
                Phase("artifact_manifest_and_verify", MANIFEST_MINUTES),
                Phase("artifact_synchronization", SYNC_MINUTES)),
            eval_minutes_per_arm=0.0, contingency_fraction=0.10,
            artifact_recovery_reserve_minutes=20.0),
        setup=SetupManifest(
            #: This session declared the init and the calibration and NOT the
            #: recovery pack, while the shared setup staged the pack for it
            #: anyway and hash-verified it twice — the same shape of undeclared
            #: inheritance that cost this session $0.0637 on local assets, one
            #: layer down. The pack is declared now because it is what setup
            #: stages; TERMINATED means no further canary is prepared, not that
            #: its specification may misdescribe the run it would perform.
            relay_inputs=(*CANONICAL_INIT, *RECOVERY_LADDER, *CALIBRATION_V1),
            #: Empty, and now honoured. `SESSION_KIND` is absent too, which
            #: routes setup to `SpendAuthorization` — whose
            #: `assert a.allows_phase_a is False` is exactly the assertion this
            #: session wants to pass.
            local_assets=LOCAL_ASSETS,
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "TEACHER_REVISION"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_device_canary",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("CANARY_FAILED",),
            #: Empty on purpose. There is no partial success to come home for:
            #: the report is written either way and is the only artifact.
            incomplete=(),
            failure_note=("an operator failed on CUDA — collecting the report, "
                          "which carries the traceback, then tearing down. That "
                          "failure IS the finding; nothing is retried.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_device_canary",
            evidence_filename="result.json",
            archive_basename="device_canary_artifacts.tar.gz",
            spec_success="configs/autoinit/device_canary_artifacts.json",
            spec_failed="configs/autoinit/device_canary_artifacts.json",
            report_names=("result.json",),
            event_streams=lambda ctx: (),
            #: Nothing. The canary produces no artifact worth keeping beyond its
            #: report, which the normal report fetch already brings home.
            fetch_products=lambda ctx: []),
        teardown=TeardownPolicy(note="nothing chains off the canary"),
        precheck=(),
        evidence_fields={"scientific_use": False,
                         "trains_anything": False,
                         "retrains_permanent_controls": False,
                         "phase_a_launched": False,
                         "phase_a_reachable_from_this_launcher": False,
                         "session_path_status": "TERMINATED 2026-08-18",
                         "priced_note": ("the compute is ~1 s by the project's "
                                         "own cost model; this price is session "
                                         "overhead")})


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test can assert on the namespace it
    produces rather than on a transcription of it.

    Attempt 1 was lost at $0.0603 because the inherited base read attributes off
    `self.a` that this parser did not define. There is no base to inherit from
    now, and `SessionRunner.__init__` checks the namespace against
    `RUNNER_ARGUMENT_CONTRACT` before a pod exists — but the parser is still
    extracted, because a contract checked at runtime and a contract checked by a
    test are worth having both.
    """
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
    ap.add_argument("--out", default="logs/autoinit_device_canary_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary="STOP for review; nothing else was started.")


if __name__ == "__main__":
    raise SystemExit(main())
