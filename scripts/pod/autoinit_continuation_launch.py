#!/usr/bin/env python3
"""Dev-box orchestrator for the characterization continuation.

    PYTHONPATH=src setsid nohup python -u \
        scripts/pod/autoinit_continuation_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> \
        --transport relay < /dev/null &

A **subclass** of the micro-preflight launcher, not a copy of it. Everything that
has been verified live — the detached start, the independent watchdog, the log
relay, the four-threshold budget, the artifact gate and the provider-confirmed
teardown — is inherited unchanged. What is overridden is what actually differs:
the plan, the authorization, the driver, the artifact spec, and the fact that
this session **materializes two existing controls instead of training anything**.

Two things are deliberate:

**Transport is separate from identity.** `materialize_controls` puts the control
artifacts under `artifacts/controls/<name>/` on the pod, by whichever route
`--transport` names. The driver then runs the same strict import gate on whatever
it finds there. A relay outage or a quota decision changes the transport; it
cannot change what counts as the control.

**Cleanup is not success.** `collect_and_teardown` returns the inherited `done`,
which is true only for `ALL_DONE`. A characterization failure still collects and
still tears down — and still reports a failed continuation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.continuation import (  # noqa: E402
    CONTINUATION_PLAN_V1, CONTINUATION_SCOPE,
)
from aadistill.infrastructure.budget import Phase, StepTime, plan_session  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "autoinit_preflight_launch",
    REPO_ROOT / "scripts/pod/autoinit_preflight_launch.py")
_preflight = importlib.util.module_from_spec(_spec)
sys.modules["autoinit_preflight_launch"] = _preflight
_spec.loader.exec_module(_preflight)

WS = _preflight.WS
REPO = _preflight.REPO
STATUS = f"{WS}/autoinit_continuation.status"
RUN_LOG = f"{WS}/autoinit_continuation_run.log"
AUTH_PATH = "logs/autoinit_continuation_authorization.json"
CONTROLS = ("preflight_ctl_r0860k_sa", "preflight_ctl_r0860k_sb")
#: Dev-box-only inputs the pod cannot fetch from git: the frozen battery, and the
#: permanent controls' own records. Small; the weights travel by `--transport`.
LOCAL_ASSETS = ("artifacts/stage1/state_eval_v1",
                "artifacts/stage3/recovery_search_v2",
                "logs/autoinit_permanent_controls")
CKPT_STORE = "/home/ecs-user/aad-artifacts/autoinit"


class Continuation(_preflight.Preflight):
    """The preflight launcher, retargeted at a session that trains nothing."""

    def __init__(self, a):
        super().__init__(a)
        self.ev["schema"] = "aadistill.autoinit.continuation_session/v1"
        self.ev["continuation_plan_hash"] = CONTINUATION_PLAN_V1.plan_hash
        self.ev["scope"] = CONTINUATION_SCOPE.as_dict()
        self.ev["trains_anything"] = False
        self.ev["transport"] = a.transport

    # -- budget: no training phase, so the shape is different -------------
    def make_plan(self) -> bool:
        phases = (Phase("import_verification", 2.0),
                  Phase("evaluation_attestation", 3.0),
                  Phase("v2_tool_rag_smoke", 6.0),
                  Phase("characterization_two_controls",
                        2 * self.a.characterization_minutes),
                  Phase("artifact_manifest_and_verify", 6.0),
                  Phase("artifact_synchronization", 4.0))
        self.plan = plan_session(
            price_per_hour=self.a.max_price,
            authorized_usd=self.auth.hard_cap_usd,
            # Nothing is trained: zero arms, and the step-time model is unused.
            arms=0, steps_per_arm=0,
            step_time=StepTime(3.15, "measured, and not used here: this session "
                                     "trains nothing"),
            setup_minutes=self.a.setup_minutes, other_phases=phases,
            eval_minutes_per_arm=0.0, transfer_minutes=self.a.transfer_minutes,
            contingency_fraction=0.10, artifact_recovery_reserve_minutes=10.0)
        self.ev["budget_plan"] = self.plan.as_dict()
        try:
            self.auth.require_within_cap(self.plan.hard_terminate_usd,
                                         what="planned hard threshold")
        except _preflight.AuthorizationError as exc:
            self.say(f"ABORT: {exc}")
            return False
        self.say(f"budget: expected {self.plan.expected_minutes:.0f} min "
                 f"${self.plan.expected_usd:.2f} · soft "
                 f"${self.plan.soft_stop_usd:.2f} · hard "
                 f"${self.plan.hard_terminate_usd:.2f} "
                 f"(authorized ${self.auth.hard_cap_usd:.2f})")
        return self.check_gpu_offered()

    # -- transport: the only part that knows how bytes arrive -------------
    def materialize_controls(self, target, host: str, scp: list) -> bool:
        """Put the control artifacts where the driver's import gate looks.

        The driver does not learn which branch ran. It sees
        `artifacts/controls/<name>/{model/,run_manifest.json,run_completion.json}`
        and applies the same strict gate either way.
        """
        route = self.a.transport
        self.ev["transport_detail"] = {"route": route, "controls": {}}
        for name in CONTROLS:
            dest = f"{REPO}/artifacts/controls/{name}"
            target.run(f"mkdir -p {dest}", timeout=60)
            if route == "relay":
                rc = target.run(
                    f"cd {REPO} && PYTHONPATH=src python3 -c \""
                    "import os;from huggingface_hub import snapshot_download;"
                    f"snapshot_download('{self.a.relay_repo}', repo_type='model',"
                    f" allow_patterns=['permanent_controls/{name}/*'],"
                    " local_dir='/workspace/controls',"
                    " token=os.environ['HF_TOKEN'])\" && "
                    f"cp -r /workspace/controls/permanent_controls/{name}/* {dest}/",
                    timeout=1800)
                ok = rc.returncode == 0
            elif route == "scp":
                local = Path(CKPT_STORE) / name / "step_001023"
                r1 = subprocess.run(scp + ["-r", str(local / "model"),
                                           f"root@{host}:{dest}/"],
                                    capture_output=True, timeout=7200)
                ok = r1.returncode == 0
            else:
                self.say(f"ABORT: unknown transport {route!r}")
                return False
            # The records travel with the assets either way; they are tiny.
            for suffix in ("run_manifest", "run_completion"):
                subprocess.run(
                    scp + [str(REPO_ROOT / "logs/autoinit_permanent_controls"
                               / f"{name}_{suffix}.json"),
                           f"root@{host}:{dest}/{suffix}.json"],
                    capture_output=True, timeout=600)
            probe = target.run(
                f"test -s {dest}/model/model.safetensors && "
                f"test -s {dest}/run_manifest.json && echo PRESENT=1 || echo PRESENT=0",
                timeout=120)
            present = "PRESENT=1" in probe.stdout
            self.ev["transport_detail"]["controls"][name] = {
                "route": route, "materialized": bool(ok and present)}
            self.say(f"  {name}: {route} rc_ok={ok} present={present}")
            if not (ok and present):
                self.say(f"ABORT: {name} did not materialize; the driver would "
                         "have nothing to import")
                return False
        self.save()
        return True

    job_id = "autoinit_continuation_driver"

    def materialize_inputs(self, target, host: str, scp: list) -> bool:
        return self.materialize_controls(target, host, scp)

    # -- the driver this session runs -------------------------------------
    def driver_command(self) -> str:
        return (f"/opt/train/bin/python "
                f"{REPO}/scripts/pod/autoinit_continuation_driver.py "
                f"--stage all --image-digest '{self.image_digest}' "
                f"--rate {self.price or self.a.max_price} "
                f"--spent-usd {self.usd():.4f} "
                f"--soft-stop-usd {self.plan.soft_stop_usd:.4f} "
                f"--authorized-usd {self.auth.hard_cap_usd:.4f} "
                f"--characterization-minutes {self.a.characterization_minutes}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--transport", choices=("relay", "scp"), required=True,
                    help="how the control weights reach the pod. Identity is "
                         "verified by the driver either way.")
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=0.99)
    ap.add_argument("--disk-gb", type=int, default=120)
    ap.add_argument("--characterization-minutes", type=float, default=18.0)
    ap.add_argument("--setup-minutes", type=float, default=8.0)
    ap.add_argument("--transfer-minutes", type=float, default=4.0)
    ap.add_argument("--teacher-revision",
                    default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--ckpt-store", default=CKPT_STORE)
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=25)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=180.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/autoinit_continuation_session.json")
    args = ap.parse_args()

    # Retarget the inherited module-level constants before anything runs.
    _preflight.AUTH_PATH = AUTH_PATH
    _preflight.STATUS = STATUS
    _preflight.RUN_LOG = RUN_LOG
    _preflight.LOCAL_ASSETS = LOCAL_ASSETS
    _preflight.CONTROLS = CONTROLS
    _preflight.PREFLIGHT_PLAN_V1 = CONTINUATION_PLAN_V1

    session = Continuation(args)
    ok = False
    try:
        ok = session.run()
    except Exception as exc:                                      # noqa: BLE001
        session.ev["launcher_error"] = f"{type(exc).__name__}: {exc}"
        session.say(f"LAUNCHER ERROR: {type(exc).__name__}: {exc}")
        if session.pod_id:
            session.teardown_now("launcher error")
    # `ok` is the inherited `done`, true only for ALL_DONE. Collection and
    # teardown having succeeded does not make a failed characterization a
    # successful continuation.
    session.ev["passed"] = bool(ok)
    session.ev["continuation_successful"] = bool(ok)
    session.ev["cleanup_is_not_success"] = (
        "artifacts are collected and the pod is torn down on every path; the "
        "session outcome is decided by the driver's terminal marker alone")
    session.ev["phase_a_launched"] = False
    session.ev["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session.save()
    print(f"\ncontinuation {'COMPLETE' if ok else 'INCOMPLETE'} — "
          f"{REPO_ROOT / args.out}. Nothing trained. Phase A NOT launched.")
    return 0 if ok else 11


if __name__ == "__main__":
    raise SystemExit(main())
