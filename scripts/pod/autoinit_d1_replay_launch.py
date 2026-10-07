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
    """Resolve the plan at $0 and stage it where the launcher ships assets.

    THE ROOT STATE IS DERIVED HERE, on the host that has the journal, and it is
    derived rather than declared: every candidate's reconstructed step-0 config
    hash goes into the plan next to the recorded one. The pod re-derives it from
    the plan's own step-0 identities before it loads any weights.
    """
    leaves = R.replay_plan(repo_root)
    root = D1S.root_teacher_identity(repo_root)
    root_state = R.derive_root_state(
        leaves, base_config=R.teacher_config(root["repo_id"], root["revision"]))
    plan = R.describe(leaves, root_state)
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


def _reconstructed(ctx, *, quiet: bool = False) -> list[dict[str, Any]]:
    """Leaves the driver's own evidence says it rebuilt. Read, never inferred.

    TWO PLACES, because this is asked at two times. During the run the runner
    relays the evidence document whole-file into `scr/relay/` on every poll --
    which is the only copy that exists while the driver is still working, and
    therefore the only one `on_poll` can act on. After collection the extracted
    archive puts it in `scr/store/`. `store` wins when both exist: it is the
    collected artifact rather than a snapshot of a run in progress.

    Looking only in `store` is what would have made the poll hook dead code:
    it would have found no evidence on every poll, said so, and secured
    nothing until closeout -- the behaviour it exists to replace.
    """
    scr = Path(ctx.args.scr)
    for where in (scr / "store", scr / "relay"):
        doc = where / "d1_replay.json"
        if not doc.is_file():
            continue
        try:
            record = json.loads(doc.read_text())
        except ValueError as exc:
            #: A relayed snapshot can be mid-write. Not an error during the
            #: run; the next poll reads a complete one.
            if not quiet:
                ctx.say(f"  replay evidence at {doc.name} is unparseable ({exc})")
            continue
        return [e for e in (record.get("leaves") or [])
                if e.get("reconstructed")]
    if not quiet:
        ctx.say("  no replay evidence came home; nothing to fetch")
    return []


#: Files a qwen3 checkpoint directory carries beside its weights. Named here
#: because which files a format writes is this session's knowledge, not the
#: transport's.
WEIGHTS_NAME = "model.safetensors"
SIDECARS = ("config.json", "generation_config.json")


#: Leaves already brought home in THIS process, keyed by state id. The fetch is
#: called from two places -- every poll, and once at collection -- and must do
#: the work once. A re-fetch would be correct and would cost another 1.2 GiB of
#: billed transfer per poll.
_SECURED: dict[str, dict] = {}


def secure_finished_leaves(ctx) -> None:
    """`on_poll`: bring a leaf home THE MOMENT the driver says it reconstructed.

    WHY NOT AT COLLECTION. q2 finishes around minute 60 of a session whose
    deadline is at 147; q4 finishes around 110. Everything that can go wrong in
    those 50 minutes -- a digest mismatch on q4, a provider stop, the deadline
    firing, this session's cumulative cap -- destroys a checkpoint that already
    reproduces exactly and already cost its GPU time. C1 attempt 17 lost six
    probes and ten hours that way, which is why AGENTS.md asks for durability
    at the moment of completion rather than collection at closeout.

    The transfer overlaps the next leaf's compute on a pod that is billing
    regardless, so it is close to free in dollars; what it buys is that the
    expensive completed unit is off the pod before anything else can fail.

    Never allowed to disturb the run: the runner already swallows and records
    an exception from this hook, and a leaf that fails here is simply retried
    at collection.
    """
    fetch_leaves(ctx, quiet=True)


def fetch_leaves(ctx, *, quiet: bool = False) -> list:
    """Bring every RECONSTRUCTED leaf home and re-identify it here.

    Gated on reconstruction, not on session success: a mismatch on the second
    leaf leaves the first one reproducing exactly and costing real GPU time.

    MULTI-STREAM, BECAUSE THE BUDGET REQUIRES IT. A single scp connection to a
    pod in this programme measured 0.486 MB/s, so a 1.2 GiB leaf is 44 minutes
    of billing -- two of them is $1.60 of transfer against this session's
    remaining cap, and the previous single-stream path would have hit its own
    45-minute limit on the first one. `infrastructure.transfer` fetches the
    weights in parallel byte ranges at a measured 4.5-8 MB/s; the same bytes,
    an eighth of the clock.

    The digest is re-derived from the bytes that ARRIVED. A product verified
    only where it was produced is a product whose transfer was never checked.
    """
    from aadistill.infrastructure.transfer import (
        TransferError, fetch_checkpoint_dir,
    )

    fetched: list = []
    if not ctx.products_eligible:
        return fetched
    rows = _reconstructed(ctx, quiet=quiet)
    if not rows:
        return list(_SECURED.values())
    store = Path(getattr(ctx.args, "ckpt_store", None)
                 or Path(ctx.args.scr) / "products")
    store.mkdir(parents=True, exist_ok=True)
    plan = {leaf.state_id: leaf for leaf in R.replay_plan(REPO_ROOT)}
    for row in rows:
        sid = row.get("state_id") or "unknown"
        if sid in _SECURED:
            fetched.append(_SECURED[sid])
            continue
        remote = row.get("checkpoint_path")
        leaf = plan.get(sid)
        if not remote or leaf is None:
            fetched.append(ProductFetchResult(
                kind="transfer", rc=1,
                detail=(f"{sid}: " + ("names no checkpoint_path" if not remote
                                      else "has no plan entry to verify against"))))
            continue
        dest = store / sid
        try:
            record = fetch_checkpoint_dir(
                f"root@{ctx.host}", ctx.target.port, remote, dest,
                weights_name=WEIGHTS_NAME, sidecars=SIDECARS,
                expect_sha256=leaf.expected_single_shard_sha256,
                on_log=lambda line: ctx.say(f"    {line}"))
        except TransferError as exc:
            fetched.append(ProductFetchResult(
                kind="transfer", rc=1, detail=f"{sid}: {exc}"))
            ctx.say(f"  leaf {sid}: TRANSFER FAILED — {exc}")
            continue
        ok = bool(record["verified"])
        result = ProductFetchResult(
            kind="transfer", rc=(0 if ok else 1),
            detail=(f"{sid}: {record['bytes'] / 2**30:.2f} GiB -> {dest} in "
                    f"{record['seconds'] / 60:.1f} min "
                    f"({record['mb_per_s']} MB/s, {record['streams']} streams); "
                    f"single_shard_sha256 {record['sha256'][:12]} "
                    f"{'MATCHED' if ok else 'DID NOT MATCH '}"
                    f"{leaf.expected_single_shard_sha256[:12]}"))
        fetched.append(result)
        if ok:
            #: Cached only on SUCCESS, so a failed transfer is retried at
            #: collection instead of being remembered as done.
            _SECURED[sid] = result
        ctx.say(f"  leaf {sid}: {record['bytes'] / 2**30:.2f} GiB, "
                f"{record['mb_per_s']} MB/s, identity "
                f"{'MATCHED' if ok else 'NOT MATCHED'}")
    return fetched


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
        #: RE-PRICED AGAINST THE CAMPAIGN REMAINDER, not re-estimated to fit.
        #: Four paid subruns have settled $0.8144 of the $3.5000 ceiling, and
        #: P12.1 makes the budget cumulative, so $2.5856 remains -- 142.3 min
        #: at $1.09/h. The first version priced at 174 min / $3.15, which was
        #: correct when nothing had been spent and is now unauthorizable.
        #:
        #: Every phase below is MEASURED, and two of them moved because four
        #: attempts produced observations the first estimate did not have:
        budget=BudgetSpec(
            arms=0, steps_per_arm=0, step_seconds=4.15,
            step_source="unused; the replay trains nothing",
            #: Four observed setups: 7.7, 9.0, 8.4 and 10.1 min. 11 bounds them.
            setup_minutes=11.0,
            #: 2.4 GiB over the MULTI-STREAM transport at a measured 4.5-8
            #: MB/s is 5-9 min, and one of the two leaves moves during the
            #: next leaf's compute (see `secure_finished_leaves`), so only the
            #: last one is serial. Was 8.0 for single-stream scp, which could
            #: not have moved either leaf inside its own 45-min cap.
            transfer_minutes=6.0,
            other_phases=(
                #: Observed at about 2 min per load in subrun r4, not 6.
                Phase("teacher_load_twice", 5.0),
                #: From the SEARCH's own telemetry, worst observed per
                #: operator: DEPTH 41.44 min twice -- both of these paths run
                #: it once, and the closeout's n=9 non-root sample maxes at
                #: 41.41, so the root figure bounds both -- plus six steps at
                #: the 0.78 min worst of the other three operators.
                Phase("eight_pinned_operator_steps", 88.0),
                #: 1.02 min per expansion, measured, times eight.
                Phase("materialize_and_reload", 8.0)),
            eval_minutes_per_arm=0.0,
            #: 0.08, not 0.20. The phases above are worst-observed rather than
            #: means, so the contingency is absorbing variance that is already
            #: priced at its maximum; and the campaign remainder will not fund
            #: a 20% band on top of a worst-case bound.
            contingency_fraction=0.08,
            #: 12, not 20. The reserve exists so products can be recovered
            #: after the soft stop, and `on_poll` now secures each leaf the
            #: moment it reconstructs -- so the exposure this covers is one
            #: leaf at 8 streams (about 4 min) plus collection, not two.
            artifact_recovery_reserve_minutes=12.0),
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
                          "and is NOT retried."),
            #: A LEAF'S PRODUCTS EXIST OR THEY DO NOT, and the terminal does
            #: not decide it. The generic rule is `terminal == success or
            #: is_incomplete(terminal)`, and this session declares no
            #: incomplete terminals -- so a DIGEST_MISMATCH on the second leaf
            #: would have fetched NOTHING, deleting a first leaf that
            #: reproduced exactly and cost its GPU time. `fetch_leaves` reads
            #: the driver's own per-leaf `reconstructed` flag and returns
            #: nothing when none did, so the gate belongs there and not here.
            products_eligible=lambda terminal, stages: True),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="d1_replay.json",
            archive_basename="d1_replay_artifacts.tar.gz",
            spec_success="configs/autoinit/d1_replay_artifacts.json",
            spec_failed="configs/autoinit/d1_replay_artifacts_failed.json",
            report_names=("d1_replay.json",),
            on_poll=secure_finished_leaves,
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
