#!/usr/bin/env python3
"""One-use launcher for the Stage-1 device canary. Nothing general.

    PYTHONPATH=src setsid nohup python -u \
        scripts/pod/autoinit_device_canary_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

A **subclass** of the micro-preflight launcher, exactly as the Phase-A launcher
is. Everything that has been proven on hardware is inherited untouched: the
bundle identity check, the spend authorization, the detached start with a
durable descriptor, the independent watchdog, the log relay, the four-threshold
budget, the artifact gate and the provider-confirmed teardown. What is overridden
is only what this session *is* — nine short declarations and three methods.

It loads a **`SpendAuthorization`**, not a `PhaseAAuthorization`, and that is a
property rather than an accident: the ordinary type's `allows_phase_a` is a hard
`False`, so this launcher cannot start Phase A even if someone points it at the
wrong artifact. The canary is infrastructure, not science.

Nothing it produces may enter scientific selection. There is no `fetch_products`
and no checkpoint comes home: the canary's children are compressions of the
canonical student toward a geometry picked to make every operator do work, and
they die with the pod.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.recovery import PreflightPlan, PreflightStage  # noqa: E402
from aadistill.infrastructure.budget import Phase, StepTime, plan_session  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "autoinit_preflight_launch",
    REPO_ROOT / "scripts/pod/autoinit_preflight_launch.py")
_preflight = importlib.util.module_from_spec(_spec)
sys.modules["autoinit_preflight_launch"] = _preflight
_spec.loader.exec_module(_preflight)

WS = _preflight.WS
REPO = _preflight.REPO
STATUS = f"{WS}/autoinit_device_canary.status"
RUN_LOG = f"{WS}/autoinit_device_canary_run.log"
AUTH_PATH = "logs/autoinit_device_canary_authorization.json"
#: The canary reads the frozen calibration mixture and the canonical student.
#: Both come from the relay; neither is a dev-box-only asset, so nothing is
#: scp'd and `LOCAL_ASSETS` is empty.
LOCAL_ASSETS: tuple[str, ...] = ()

#: One stage. The plan exists because the authorization binds to a plan hash,
#: and a canary that borrowed the preflight's hash would be claiming to be a
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


class DeviceCanary(_preflight.Preflight):
    """The micro-preflight machinery, pointed at one script."""

    def __init__(self, a):
        super().__init__(a)
        self.ev["schema"] = "aadistill.autoinit.device_canary_session/v1"
        self.ev["scientific_use"] = False
        self.ev["trains_anything"] = False
        self.ev["retrains_permanent_controls"] = False
        self.ev["phase_a_launched"] = False
        self.ev["phase_a_reachable_from_this_launcher"] = False

    audit_dirname = "autoinit_device_canary"
    evidence_filename = "result.json"
    archive_basename = "device_canary_artifacts.tar.gz"
    spec_success = "configs/autoinit/device_canary_artifacts.json"
    spec_failed = "configs/autoinit/device_canary_artifacts.json"
    failure_markers = ("CANARY_FAILED",)
    #: Empty on purpose. There is no partial success to come home for: the
    #: report is written either way and is the only artifact.
    incomplete_markers: tuple[str, ...] = ()
    failure_note = ("an operator failed on CUDA — collecting the report, which "
                    "carries the traceback, then tearing down. That failure IS "
                    "the finding; nothing is retried.")
    report_names = ("result.json",)
    job_id = "autoinit_device_canary"

    def event_streams(self) -> tuple[str, ...]:
        return ()

    def session_auth_path(self) -> str:
        return AUTH_PATH

    def session_plan_hash(self) -> str:
        return CANARY_PLAN_V1.plan_hash

    def setup_env(self) -> dict[str, str]:
        """`SESSION_KIND` is left at its default, `spend`.

        That routes setup to `SpendAuthorization`, whose
        `assert a.allows_phase_a is False` is exactly the assertion this session
        wants to pass.
        """
        return {}

    def make_plan(self) -> bool:
        self.plan = plan_session(
            price_per_hour=self.a.max_price,
            authorized_usd=self.auth.hard_cap_usd,
            arms=0, steps_per_arm=0,          # nothing is trained
            step_time=StepTime(4.15, "unused; the canary trains nothing"),
            setup_minutes=self.a.setup_minutes,
            other_phases=(
                Phase("parent_load_and_calibration", PARENT_LOAD_MINUTES),
                Phase("six_operator_invocations_and_lifecycles", CANARY_MINUTES),
                Phase("artifact_manifest_and_verify", MANIFEST_MINUTES),
                Phase("artifact_synchronization", SYNC_MINUTES)),
            eval_minutes_per_arm=0.0, transfer_minutes=self.a.transfer_minutes,
            contingency_fraction=0.10, artifact_recovery_reserve_minutes=20.0)
        self.ev["budget_plan"] = self.plan.as_dict()
        self.ev["priced_note"] = (
            "the compute is ~1 s by the project's own cost model; this price is "
            "session overhead")
        try:
            self.auth.require_within_cap(self.plan.hard_terminate_usd,
                                         what="planned hard threshold")
            self.auth.require_within_launch_limit(self.plan.hard_terminate_usd,
                                                  what="planned hard threshold")
        except Exception as exc:                                  # noqa: BLE001
            self.say(f"AUTHORIZATION: {exc}")
            return False
        self.say(f"budget: expected {self.plan.expected_minutes:.0f} min "
                 f"${self.plan.expected_usd:.2f} · soft "
                 f"${self.plan.soft_stop_usd:.2f} · hard "
                 f"${self.plan.hard_terminate_usd:.2f} "
                 f"(authorized ${self.auth.hard_cap_usd:.2f})")
        return True

    def relay_precheck(self) -> bool:
        """Fail at $0 rather than after a 45-minute setup.

        Same shape as the base's, with this session's two inputs: the canonical
        student it compresses and the frozen calibration mixture it reads. The
        base hardcodes its own list, so this is an override rather than a
        parameterization — one list, not a new abstraction.
        """
        need = ["stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors",
                "e8_inputs_20260810/calibration_v1/items.jsonl"]
        try:
            from huggingface_hub import HfApi
            present = set(HfApi().list_repo_files(
                self.a.relay_repo, repo_type="model"))
        except Exception as exc:                                  # noqa: BLE001
            self.say(f"ABORT: cannot list the relay: {exc!r}"[:200])
            return False
        missing = [f for f in need if f not in present]
        self.ev["precheck"] = {"relay_needed": need, "relay_missing": missing,
                               "local_assets": list(LOCAL_ASSETS),
                               "local_missing": []}
        if missing:
            self.say(f"ABORT: relay missing {missing}")
            return False
        self.say(f"precheck OK: {len(need)} relay inputs, no local assets")
        return True

    def fetch_products(self, host: str, target, stage2_passed: bool) -> list:
        """Nothing. The canary produces no artifact worth keeping beyond its
        report, which the normal report fetch already brings home."""
        return []

    def driver_command(self) -> str:
        return (f"/opt/train/bin/python "
                f"{REPO}/scripts/pod/autoinit_device_canary.py "
                f"--out {REPO}/artifacts/audit/autoinit_device_canary/result.json "
                f"--workdir artifacts/autoinit/device_canary --device cuda")


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test can assert on the namespace it
    produces rather than on a transcription of it.

    Attempt 1 was lost at $0.0603 because the inherited base reads attributes
    off `self.a` that this parser did not define. Subclassing a launcher means
    inheriting its ARGUMENT contract as well as its methods, and the contract is
    invisible from the subclass.
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
    # --- read by the INHERITED base, not by anything written here ----------
    #
    # `TEACHER_REVISION` is forwarded to the shared setup script. The canary
    # needs no teacher, but setup is shared and its behaviour must not change
    # because a canary is driving it, so this is the same frozen revision every
    # other session pins. Nothing here downloads or uses a teacher.
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    # Read only by the base's `fetch_products`, which this session OVERRIDES to
    # return nothing. The value is therefore never used; it is canary-scoped
    # rather than the real checkpoint store so that if some future edit ever did
    # fetch, it could not land among real checkpoints.
    ap.add_argument("--ckpt-store",
                    default="artifacts/audit/autoinit_device_canary/never_fetched")
    # Same: unused. One minute rather than a generous window, because a timeout
    # that would mask a fetch this session must never perform is worse than one
    # that fails.
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=1)
    return ap


def main() -> int:
    args = build_parser().parse_args()

    _preflight.AUTH_PATH = AUTH_PATH
    _preflight.STATUS = STATUS
    _preflight.RUN_LOG = RUN_LOG
    _preflight.LOCAL_ASSETS = LOCAL_ASSETS
    _preflight.PREFLIGHT_PLAN_V1 = CANARY_PLAN_V1

    session = DeviceCanary(args)
    ok = False
    try:
        ok = session.run()
    except Exception as exc:                                      # noqa: BLE001
        session.ev["launcher_error"] = f"{type(exc).__name__}: {exc}"
        session.say(f"LAUNCHER ERROR: {type(exc).__name__}: {exc}")
        if session.pod_id:
            session.teardown_now("launcher error")
    session.ev["passed"] = bool(ok)
    session.ev["scientific_use"] = False
    session.ev["followon_started"] = False
    session.save()
    print(f"\nDevice canary {'PASSED' if ok else 'FAILED'} — "
          f"{REPO_ROOT / args.out}. STOP for review; nothing else was started.")
    return 0 if ok else 11


if __name__ == "__main__":
    raise SystemExit(main())
