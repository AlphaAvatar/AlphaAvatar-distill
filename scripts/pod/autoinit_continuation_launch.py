#!/usr/bin/env python3
"""The Stage-3 characterization continuation, as a session specification.

    PYTHONPATH=src setsid nohup python -u \
        scripts/pod/autoinit_continuation_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> \
        --transport relay < /dev/null &

Declares WHAT the session is. How it is run lives once, in
`aadistill.infrastructure.session_runner`. This was a **subclass** of the
micro-preflight launcher until 2026-08-18; attempt 5 died at $0.1369 asserting an
unrelated session's authorization, and attempt 6 at $0.1324 because the shared
setup wrote its markers to the preflight's status filename. Both are things a
session can only get wrong when its own contract is spread across two files.

Two things are deliberate:

**Transport is separate from identity.** `materialize_controls` puts the control
artifacts under `artifacts/controls/<name>/` on the pod, by whichever route
`--transport` names. The driver then runs the same strict import gate on whatever
it finds there. A relay outage or a quota decision changes the transport; it
cannot change what counts as the control.

**Cleanup is not success.** The runner returns `done`, which is true only for the
success marker. A characterization failure still collects and still tears down —
and still reports a failed continuation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
# The sibling science-input declarations. Present when this file is run
# directly; absent when a test loads it by path, which is how the
# structural checks load every launcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.autoinit.authorization import SpendAuthorization  # noqa: E402
from aadistill.autoinit.continuation import (  # noqa: E402
    CONTINUATION_PLAN_V1, CONTINUATION_SCOPE,
)
from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, LocalAsset, MarkerPolicy, RelayInput,
    SessionContext, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    local_files_gate, session_commit_gate,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
from autoinit_science_inputs import (  # noqa: E402
    CALIBRATION_V1, CANONICAL_INIT, RECOVERY_LADDER,
)

STATUS = f"{WS}/autoinit_continuation.status"
RUN_LOG = f"{WS}/autoinit_continuation_run.log"
AUTH_PATH = "logs/autoinit_continuation_authorization.json"
CONTROLS = ("preflight_ctl_r0860k_sa", "preflight_ctl_r0860k_sb")
CKPT_STORE = "/home/ecs-user/aad-artifacts/autoinit"
#: Dev-box-only inputs the pod cannot fetch from git: the frozen battery, and the
#: permanent controls' own records. Small; the weights travel by `--transport`.
LOCAL_ASSETS = (
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage3/recovery_search_v2", "recovery_search_v2",
               "artifacts/stage3"),
    LocalAsset("logs/autoinit_permanent_controls", "autoinit_permanent_controls",
               "logs"),
)
TEST_IGNORES = ("tests/data/test_recovery_corpus_pipeline.py",
                "tests/pod/test_phase_a_stages1_5_execute.py")
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
#: The three record files each control travels with.
CONTROL_RECORDS = tuple(
    f"logs/autoinit_permanent_controls/{c}_{suffix}.json"
    for c in CONTROLS
    for suffix in ("probe_identity", "run_manifest", "run_completion"))

#: Per-control characterization budget. 18 min was the historical GUESS;
#: characterizing one checkpoint on this battery has never been measured, and
#: measuring it is why this session exists. 24 keeps BOTH controls affordable if
#: the true cost is a third over the guess. Losing sb to a tight soft stop would
#: cost a whole second paid session; an unused leash costs nothing, because the
#: pod is torn down on completion rather than at the threshold.
CHARACTERIZATION_MINUTES = 24.0
#: Setup allowance, derived from MEASUREMENT rather than estimate: create->ssh
#: 2.0 min, ENV_READY..ASSETS_READY 3.2 min, teacher+rope 0.7 min, CPU suite 2.4
#: min, plus the vLLM wheelhouse fetch (~1.5 min for 3.62 GiB) and its 33-second
#: offline install. ~10.4 min, priced at 11 so the number is not the optimistic
#: one.
SETUP_MINUTES = 11.0


def materialize_controls(ctx: SessionContext) -> bool:
    """Put the control artifacts where the driver's import gate looks.

    The driver does not learn which branch ran. It sees
    `artifacts/controls/<name>/{model/,run_manifest.json,run_completion.json}`
    and applies the same strict gate either way.
    """
    route = ctx.args.transport
    scp = list(ctx.scp)
    ctx.evidence["transport_detail"] = {"route": route, "controls": {}}
    for name in CONTROLS:
        dest = f"{REPO}/artifacts/controls/{name}"
        ctx.target.run(f"mkdir -p {dest}", timeout=60)
        if route == "relay":
            # `HF_TOKEN` is exported inside setup.sh and dies with it. This runs
            # in a FRESH ssh session, which inherits none of that, so the token
            # is read from the file the launcher already staged — reading
            # `os.environ['HF_TOKEN']` here raises KeyError, after the setup it
            # would have paid for.
            rc = ctx.target.run(
                f"cd {REPO} && HF_TOKEN=\"$(cat {WS}/hf/token)\" "
                "PYTHONPATH=src python3 -c \""
                "import os;from huggingface_hub import snapshot_download;"
                f"snapshot_download('{ctx.args.relay_repo}', repo_type='model',"
                f" allow_patterns=['permanent_controls/{name}/*'],"
                " local_dir='/workspace/controls',"
                " token=os.environ['HF_TOKEN'])\" && "
                f"cp -r /workspace/controls/permanent_controls/{name}/* {dest}/",
                timeout=1800)
            ok = rc.returncode == 0
        elif route == "scp":
            local = Path(ctx.args.ckpt_store) / name / "step_001023"
            r1 = subprocess.run(scp + ["-r", str(local / "model"),
                                       f"root@{ctx.host}:{dest}/"],
                                capture_output=True, timeout=7200)
            ok = r1.returncode == 0
        else:
            ctx.say(f"ABORT: unknown transport {route!r}")
            return False
        # The records travel with the assets either way; they are tiny.
        for suffix in ("run_manifest", "run_completion"):
            subprocess.run(
                scp + [str(REPO_ROOT / "logs/autoinit_permanent_controls"
                           / f"{name}_{suffix}.json"),
                       f"root@{ctx.host}:{dest}/{suffix}.json"],
                capture_output=True, timeout=600)
        probe = ctx.target.run(
            f"test -s {dest}/model/model.safetensors && "
            f"test -s {dest}/run_manifest.json && echo PRESENT=1 || echo PRESENT=0",
            timeout=120)
        present = "PRESENT=1" in probe.stdout
        ctx.evidence["transport_detail"]["controls"][name] = {
            "route": route, "materialized": bool(ok and present)}
        ctx.say(f"  {name}: {route} rc_ok={ok} present={present}")
        if not (ok and present):
            ctx.say(f"ABORT: {name} did not materialize; the driver would "
                    "have nothing to import")
            return False
    return True


def transport_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Under `--transport scp`, the local control weights must be there.

    The relay branch's requirement is a `RelayInput` in the manifest instead, so
    both routes fail at $0 rather than at materialization.
    """
    if ctx.args.transport != "scp":
        return True, ""
    missing = [str(Path(ctx.args.ckpt_store) / c / "step_001023/model")
               for c in CONTROLS
               if not (Path(ctx.args.ckpt_store) / c / "step_001023/model").is_dir()]
    if missing:
        return False, f"--transport scp but the local control weights are missing: {missing}"
    return True, f"transport scp: both control weight directories present"


transport_gate.__name__ = "transport_inputs"


def driver_command(ctx: SessionContext, plan) -> str:
    return (f"/opt/train/bin/python "
            f"{REPO}/scripts/pod/autoinit_continuation_driver.py "
            f"--stage all --image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f} "
            f"--characterization-minutes {ctx.args.characterization_minutes}")


def spec(args) -> SessionSpec:
    #: The pack is not training input here — nothing trains — but the strict
    #: import gate RECOMPUTES each control's pack hash from `blocks.npz` rather
    #: than trusting the value its run recorded, so Stage 0 reads it. The
    #: calibration is staged too: this session never declared it and the shared
    #: shell staged it regardless, which is the asymmetry that made the
    #: declaration decorative.
    relay_inputs = [*CANONICAL_INIT, *RECOVERY_LADDER, *CALIBRATION_V1]
    if args.transport == "relay":
        #: No `dest`, and that now means one thing only: setup does not stage
        #: these. `materialize_inputs` does, by the route `--transport` names,
        #: and the declaration buys them the $0 relay precheck. It can no longer
        #: mean "the shell knows where this goes" — the shell knows nothing.
        relay_inputs += [RelayInput(f"permanent_controls/{c}/model/model.safetensors")
                         for c in CONTROLS]
    return SessionSpec(
        session_id="autoinit-continuation",
        schema="aadistill.autoinit.continuation_session/v1",
        description=("the Stage-3 control characterization: imports two existing "
                     "permanent controls, trains nothing, materializes the "
                     "frozen thresholds"),
        authorization_path=AUTH_PATH,
        authorization_loader=SpendAuthorization.load,
        plan_id=CONTINUATION_PLAN_V1.plan_id,
        plan_hash=CONTINUATION_PLAN_V1.plan_hash,
        budget=BudgetSpec(
            # Nothing is trained: zero arms, and the step-time model is unused.
            arms=0, steps_per_arm=0,
            step_seconds=3.15,
            step_source="measured, and not used here: this session trains nothing",
            below_floor_reason=(
                "no step is taken: arms=0 and steps_per_arm=0, so the step-time "
                "model contributes zero minutes to this plan. The floor exists "
                "to stop a TRAINING session being priced from an optimistic "
                "step time; there is no training here to misprice."),
            setup_minutes=args.setup_minutes,
            transfer_minutes=args.transfer_minutes,
            other_phases=(
                Phase("import_verification", 2.0),
                Phase("evaluation_attestation", 3.0),
                Phase("v2_tool_rag_smoke", 6.0),
                Phase("characterization_two_controls",
                      2 * args.characterization_minutes),
                Phase("artifact_manifest_and_verify", 6.0),
                Phase("artifact_synchronization", 4.0)),
            eval_minutes_per_arm=0.0, contingency_fraction=0.10,
            artifact_recovery_reserve_minutes=10.0),
        setup=SetupManifest(
            relay_inputs=tuple(relay_inputs),
            local_assets=LOCAL_ASSETS,
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "TEACHER_REVISION"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_continuation_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("CONTINUATION_FAILED", "CONTINUATION_INCOMPLETE"),
            incomplete=("CONTINUATION_INCOMPLETE",),
            failure_note=("the continuation did not complete — collecting "
                          "evidence, then tearing down. The permanent controls "
                          "are inputs here and are untouched; nothing was "
                          "trained.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_continuation",
            evidence_filename="continuation_evidence.json",
            archive_basename="continuation_artifacts.tar.gz",
            spec_success="configs/autoinit/continuation_artifacts.json",
            spec_failed="configs/autoinit/continuation_artifacts_failed.json",
            report_names=("continuation_evidence.json", "imported_controls.json",
                          "attested_evaluation_protocol.json",
                          "materialized_thresholds.json"),
            #: None: this session appends to no training stream because it trains
            #: nothing. Naming the preflight's train logs here would make every
            #: teardown wait on files that cannot exist.
            event_streams=lambda ctx: (),
            #: Nothing to fetch back: the controls are INPUTS, and the
            #: measurements travel in the artifact archive. The preflight fetches
            #: checkpoints because it created them; creating nothing, this
            #: fetches nothing.
            fetch_products=lambda ctx: []),
        teardown=TeardownPolicy(
            note="cleanup is not success; the terminal marker decides"),
        precheck=(
            session_commit_gate(REPO_ROOT, AUTH_PATH, check_lineage=False),
            local_files_gate(REPO_ROOT, CONTROL_RECORDS,
                             what="permanent_control_records"),
            transport_gate,
        ),
        materialize_inputs=materialize_controls,
        evidence_fields={
            "continuation_plan_hash": CONTINUATION_PLAN_V1.plan_hash,
            "scope": CONTINUATION_SCOPE.as_dict(),
            "trains_anything": False,
            "transport": args.transport,
            "phase_a_launched": False,
            "phase_a_reachable_from_this_launcher": False})


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--characterization-minutes", type=float,
                    default=CHARACTERIZATION_MINUTES)
    ap.add_argument("--setup-minutes", type=float, default=SETUP_MINUTES)
    ap.add_argument("--transfer-minutes", type=float, default=4.0)
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
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary="Nothing trained. Phase A NOT launched.")


if __name__ == "__main__":
    raise SystemExit(main())
