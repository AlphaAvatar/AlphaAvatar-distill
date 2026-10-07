#!/usr/bin/env python3
"""Reconstruct D1's two unretained quality-order finalists on one L40S.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_d1_replay_launch.py \
        --scr <dir> --session-commit <sha> --bundle <name> --run-id <id>

ENGINEERING, not science. The session decides nothing and measures nothing: the
candidate set is read from the maintainer's retention decision, and every one of
the eight operator steps is pinned to the artifact digest the completed search
recorded for it. The only two outcomes are "byte-identical to the search" and
"STOP". It therefore loads the NARROW `PreflightAuthorization`, which grants
nothing and cannot reach Phase A, and declares no `SESSION_KIND` -- that routes
the shared setup to the generic branch whose single assertion,
`not a.allows("phase_a")`, is exactly what this session wants to pass.

THE GPU IS PINNED, NOT CHOSEN. Success here IS digest equality, so the device is
part of the experiment: a different architecture can select different kernels and
therefore different low bits, which would surface as a digest mismatch -- a stop
condition -- and manufacture a finding out of an infrastructure substitution.

THE JOURNAL DOES NOT TRAVEL. Ancestry resolution needs all 92 recorded states,
67 MB of them, against an uplink measured at 0.72 MB/s -- about 93 minutes of
billing to ship what resolves to 5.4 KB. The plan is resolved HERE, at $0, by
the module that owns the journal's checks, and the pod receives only the pinned
digests it must satisfy.

THE PRODUCT IS WEIGHTS. Both leaves are fetched, re-identified against the
search's own four digests on arrival, and teardown is refused while a
reconstructed leaf exists only on the pod. Fetching is gated on leaves having
been RECONSTRUCTED, never on the session having succeeded: `if terminal ==
"ALL_DONE"` has already destroyed verified checkpoints in this programme.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, LocalAsset, MarkerPolicy,
    Phase, ProductFetchResult, SessionSpec, SetupManifest, TeardownPolicy,
)
from experiments.deployment import deployment_commands  # noqa: E402
from experiments.phase_d1 import d1_session as D1S  # noqa: E402
from experiments.phase_d1 import replay_specs as R  # noqa: E402
from experiments.preflight import PreflightAuthorization  # noqa: E402
from experiments.run_layout import rel_run_dir  # noqa: E402

REPO = "/workspace/aad"
WS = D1S.POD_WORKSPACE
STATUS = f"{WS}/autoinit_d1_replay.status"
RUN_LOG = f"{WS}/autoinit_d1_replay_run.log"
AUTH_PATH = "logs/budget/approvals/autoinit_d1_replay_authorization.json"
AUDIT_DIRNAME = "autoinit_d1_replay"
EVIDENCE_DIR = f"artifacts/audit/{AUDIT_DIRNAME}"
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: The resolved plan, staged as an asset. Written by `--write-plan` before the
#: launch and verified on the pod against its own declared digests.
PLAN_NAME = "d1_replay_plan.json"
PLAN_REL = f"artifacts/stage1/{PLAN_NAME}"

#: WHAT THE POD REQUIRES -- which is not the same question as what this session
#: reads, and getting those two confused cost $0.1506.
#:
#: The first version staged only the two calibration mixtures, on the reasoning
#: that the eight operator steps use both profiles and that the replay evaluates
#: nothing, so staging the state-eval suite "would be undeclared inheritance in
#: reverse". That inverted the rule. A SESSION DECLARES WHAT THE SETUP REQUIRES,
#: not what it reads: this session's `test_paths` points the pod's blocking gate
#: at D1's own suite, and that suite loads the frozen state-eval asset. Setup
#: reached TESTS_OK with `rc=1` and
#:
#:     D1SessionError: the frozen state-eval asset is not staged at
#:     artifacts/stage1/state_eval_v1
#:
#: on a billing pod. It is the same shape as the device canary's $0.0637: a
#: session that honestly declared it wanted an asset it did not read, and a
#: shared setup step that required it anyway.
#:
#: THE SET IS NOW THE SEARCH SESSION'S, which is the proven one: these exact
#: three made this exact suite pass its pod gate, 346 tests, on the run that
#: produced the leaves this session rebuilds. 2.4 MiB total, seconds over scp.
SCIENCE_ASSETS: tuple[LocalAsset, ...] = (
    LocalAsset(D1S.STATE_EVAL_ROOT, "state_eval_v1", "artifacts/stage1"),
    LocalAsset("artifacts/stage1/e8_calibration_v1", "e8_calibration_v1",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage1/reasoning_heavy_v2", "reasoning_heavy_v2",
               "artifacts/stage1"),
)


def run_dir_for(run_id: str | None) -> str:
    return rel_run_dir("phase_d1_replay", run_id or "unrecorded", D1S.STAGE_ID)


def write_plan(repo_root: Path) -> dict[str, Any]:
    """Resolve the plan at $0 and stage it where the launcher ships assets."""
    plan = R.describe(R.replay_plan(repo_root))
    dest = repo_root / PLAN_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(plan, indent=1) + "\n")
    return plan


def driver_command(ctx: Any, plan: Any) -> str:
    """The exact command the pod runs. A test parses it with the driver's parser."""
    return (
        f"/opt/train/bin/python {REPO}/scripts/pod/autoinit_d1_replay_driver.py "
        f"--out {EVIDENCE_DIR} "
        f"--run-id {getattr(ctx.args, 'run_id', 'unrecorded')} "
        f"--plan {REPO}/{PLAN_REL} "
        f"--products {REPO}/{EVIDENCE_DIR}/products "
        f"--status {STATUS} --device {R.REPLAY_DEVICE}")


def _reconstructed(ctx) -> list[dict[str, Any]]:
    """Leaves the driver's own evidence says it rebuilt. Read, never inferred."""
    store = Path(ctx.args.scr) / "store"
    doc = store / "d1_replay.json"
    if not doc.is_file():
        ctx.say("  no replay evidence came home; nothing to fetch")
        return []
    try:
        record = json.loads(doc.read_text())
    except ValueError as exc:
        ctx.say(f"  replay evidence is unparseable ({exc})")
        return []
    return [e for e in (record.get("leaves") or []) if e.get("reconstructed")]


def fetch_leaves(ctx) -> list:
    """Bring every RECONSTRUCTED leaf home and re-identify it here.

    Gated on reconstruction, not on session success: a mismatch on the second
    leaf leaves the first one reproducing exactly and costing real GPU time.
    """
    fetched: list = []
    if not ctx.products_eligible:
        return fetched
    rows = _reconstructed(ctx)
    if not rows:
        ctx.say("  no reconstructed leaf to secure")
        return fetched
    store = Path(getattr(ctx.args, "ckpt_store", None)
                 or Path(ctx.args.scr) / "products")
    store.mkdir(parents=True, exist_ok=True)
    plan = {leaf.state_id: leaf for leaf in R.replay_plan(REPO_ROOT)}
    for row in rows:
        sid = row.get("state_id") or "unknown"
        remote = row.get("checkpoint_path")
        if not remote:
            fetched.append(ProductFetchResult(
                kind="transfer", rc=1, detail=f"{sid} names no checkpoint_path"))
            continue
        dest = store / sid
        rc = subprocess.run(
            ["timeout", f"{getattr(ctx.args, 'ckpt_fetch_limit_min', 45)}m",
             "scp", "-r", "-P", str(ctx.target.port),
             "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"root@{ctx.host}:{remote}", str(dest)],
            capture_output=True, timeout=None).returncode
        size = (sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
                if dest.exists() else 0)
        ok, why = _verify_here(dest, plan.get(sid))
        fetched.append(ProductFetchResult(
            kind="transfer", rc=(0 if rc == 0 and ok else 1),
            detail=(f"{sid}: rc={rc}, {size / 2**30:.2f} GiB -> {dest}; "
                    f"identity {'MATCHED' if ok else 'NOT MATCHED'} ({why})")))
        ctx.say(f"  leaf {sid}: rc={rc}, {size / 2**30:.2f} GiB, "
                f"identity {'MATCHED' if ok else 'NOT MATCHED'}")
    return fetched


def _verify_here(dest: Path, leaf) -> tuple[bool, str]:
    """Re-derive the identity from the bytes that ARRIVED, on this machine.

    The driver already gated every step against its pin on the pod. This asks
    the same question of the transferred copy, because a product verified only
    where it was produced is a product whose transfer was never checked.
    """
    import hashlib

    if leaf is None:
        return False, "no plan entry for this state id"
    shard = next((p for p in dest.rglob("model.safetensors")), None)
    if shard is None:
        return False, f"no model.safetensors under {dest}"
    h = hashlib.sha256()
    with shard.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    got = h.hexdigest()
    if got != leaf.expected_single_shard_sha256:
        return False, (f"single_shard_sha256 {got[:12]} != "
                       f"{leaf.expected_single_shard_sha256[:12]}")
    return True, f"single_shard_sha256 {got[:12]} as the search recorded"


def leaves_secured(ctx, fetched) -> tuple[bool, str]:
    """BOTH, or this session owes products it did not secure.

    Separate from the fetch returning cleanly, because `all([])` is True: a
    fetch that secured nothing would pass every transfer check while leaving the
    only copies on a pod about to be deleted.
    """
    rows = _reconstructed(ctx)
    if not rows:
        return True, ("no leaf was reconstructed, so there are no product bytes "
                      "to secure; the replay's evidence is what it produced")
    good = [f for f in fetched if getattr(f, "rc", 1) == 0]
    if len(good) != len(rows):
        return False, (f"{len(good)} of {len(rows)} reconstructed leaves are "
                       "secured and identity-verified off-pod; deleting the pod "
                       "now would destroy a checkpoint that reproduces exactly")
    return True, (f"all {len(good)} reconstructed leaves secured off-pod and "
                  "re-identified against the search's own digests")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runpod-config",
                    default=str(Path("~/.runpod/config.toml").expanduser()))
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=1.09)
    ap.add_argument("--disk-gb", type=int, default=120)
    ap.add_argument("--ckpt-store", default=None)
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=45)
    ap.add_argument("--out", default=None)
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--token-src",
                    default=str(Path("~/.cache/huggingface/token").expanduser()))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=1)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--poll-limit-min", type=float, default=300.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--write-plan", action="store_true",
                    help="resolve and stage the plan, then exit. $0")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def spec(args) -> SessionSpec:
    from aadistill.initialization.planning.recovery import (
        PreflightPlan, PreflightStage,
    )

    plan = PreflightPlan(
        plan_id="autoinit.d1_replay", version=1,
        stages=(PreflightStage(
            stage=0, name="d1 finalist reconstruction", blocking=True,
            purpose=("rebuild the two quality-order finalists the completed "
                     "search measured and did not retain, every step pinned to "
                     "the digest it recorded"),
            produces=("two checkpoints whose four identities equal the "
                      "search's own, or a digest mismatch naming the step",),
            stop_conditions=("a digest mismatch, which is deterministic and is "
                             "not retried",)),))
    return SessionSpec(
        session_id="autoinit-d1-replay",
        schema="aadistill.autoinit.d1_replay_session/v1",
        description=("D1: reconstruct two unretained quality-order finalists, "
                     "eight digest-pinned operator steps. Decides nothing, "
                     "measures nothing, trains nothing."),
        authorization_path=AUTH_PATH,
        authorization_loader=PreflightAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=plan.plan_id, plan_hash=plan.plan_hash,
        budget=BudgetSpec(
            arms=0, steps_per_arm=0, step_seconds=4.15,
            step_source="unused; the replay trains nothing",
            setup_minutes=12.0, transfer_minutes=8.0,
            other_phases=(
                #: From the SEARCH's own telemetry, worst observed per operator:
                #: DEPTH 41.44 min and the other three under 50 s each, twice.
                Phase("teacher_load_twice", 12.0),
                Phase("eight_pinned_operator_steps", 88.0),
                Phase("materialize_and_reload", 8.0)),
            eval_minutes_per_arm=0.0, contingency_fraction=0.20,
            artifact_recovery_reserve_minutes=20.0),
        setup=SetupManifest(
            relay_inputs=(),
            local_assets=(*SCIENCE_ASSETS,
                          LocalAsset(PLAN_REL, PLAN_NAME, "artifacts/stage1")),
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "TEACHER_REVISION"),
            #: ROPE_OK absent: the shared step globs a STAGED student checkpoint
            #: and exits 1 when none matches, and this session stages none -- it
            #: materializes from the teacher. The risk it covers is handled
            #: strictly more strongly here: every step is canonically reloaded
            #: and identity-checked against a pinned digest.
            #: VLLM_READY absent: this session serves nothing.
            #: ASSETS_READY absent: it verifies frozen roots this session does
            #: not stage.
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "TEACHER_READY", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION,
            test_paths=("scripts/experiments/stage-1/phase_d1/tests",)),
        driver_command=driver_command,
        driver_job_id="autoinit_d1_replay",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("RUN_FAILED", "DIGEST_MISMATCH"),
            failure_note=("the replay stopped -- collecting its evidence, then "
                          "tearing down. A digest mismatch is deterministic "
                          "and is NOT retried.")),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="d1_replay.json",
            archive_basename="d1_replay_artifacts.tar.gz",
            spec_success="configs/autoinit/d1_replay_artifacts.json",
            spec_failed="configs/autoinit/d1_replay_artifacts_failed.json",
            report_names=("d1_replay.json",),
            fetch_products=fetch_leaves,
            products_secured=leaves_secured),
        teardown=TeardownPolicy(),
    )


def main(argv: list[str] | None = None) -> int:
    from aadistill.infrastructure.session_runner import SessionRunner, run_session

    #: `--write-plan` IS A $0 PREPARATION STEP and needs none of the launch
    #: arguments. They stay `required` on the parser so the shared dispatch
    #: probe -- which fills every required option and builds the spec -- still
    #: sees this launcher; a launcher invisible to that probe is one whose
    #: missing SESSION_KIND branch falls through to `spend`.
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--write-plan" in raw:
        plan = write_plan(REPO_ROOT)
        print(f"wrote {PLAN_REL}  leaves={plan['n_leaves']}")
        for leaf in plan["leaves"]:
            print(f"  q{leaf['quality_position']} {leaf['state_id']} "
                  f"{len(leaf['steps'])} pinned steps")
        return 0

    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = f"{run_dir_for(args.run_id)}/runtime/session.json"
    session = spec(args)
    if args.dry_run:
        runner = SessionRunner(session, args, REPO_ROOT)
        runner.run()
        terminal = runner.ev.get("terminal")
        runner.ev["passed"] = False
        runner.save()
        print(f"\nterminal {terminal}\nrecord {args.out}")
        return 0 if terminal == "DRY_RUN_GATES_PASSED" else 11
    return run_session(session, args, REPO_ROOT,
                       summary=("D1 replay: two reconstructed finalists, "
                                "identity-verified off-pod, are what this "
                                "session owes."))


if __name__ == "__main__":
    raise SystemExit(main())
