#!/usr/bin/env python3
"""Run the replay-only artifact reconstruction on one L40S, and bring the leaves home.

    PYTHONPATH=src:scripts python scripts/pod/autoinit_c2_replay_launch.py \
        --run-id attempt1 --max-price 1.09

One pod, five digest-pinned paths, five checkpoints off the pod and verified on
arrival. The session decides nothing: it restores the artifacts behind a Top-5
that is already frozen and accepted.

The product here is WEIGHTS, and that changes what this launcher must get right.
The full-search launcher declared no `fetch_products` at all, so even a fully
successful search would have collected its evidence and left five 1.11 GiB
checkpoints to die with the pod. This one fetches them, verifies each against
attempt 3's recorded identity **on the destination**, and refuses teardown while
any reconstructed leaf is still only on the pod.

Fetching is gated on leaves having been RECONSTRUCTED, never on the session
having succeeded. A failure at leaf 4 leaves three leaves that cost real GPU
time and reproduce exactly; `if terminal == "ALL_DONE"` has already destroyed
$2.82 of verified checkpoints in this programme.

Destination is the development host's out-of-tree store, not the hub: the hub's
private tier admits one 1.11 GiB leaf and refuses the second. Measured, not
assumed — see `logs/stages/stage-1/phase_c2_replay/plans/leaf_destination_decision.md`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, MarkerPolicy, SessionContext,
    SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate)
from aadistill.infrastructure.session_runner import run_session  # noqa: E402

from autoinit_science_inputs import CALIBRATION_V1  # noqa: E402
from experiments.deployment import (  # noqa: E402
    POD_IMAGE, deployment_commands)
from experiments.phase_c2 import replay as RG  # noqa: E402
from experiments.phase_c2 import replay_bundle as RT  # noqa: E402
from experiments.phase_c2 import replay_specs as RS  # noqa: E402
from experiments.run_layout import rel_run_dir  # noqa: E402
from phase_a_frozen import TEACHER_REVISION  # noqa: E402

EXPERIMENT_ID = "phase_c2_replay"
STAGE_ID = "1"

REPO = "/workspace/aad/repo"
WS = "/workspace/aad"
WORKDIR = f"{REPO}/artifacts/autoinit/c2_replay"
LEAF_DIR = f"{WORKDIR}/leaves"
AUDIT_DIRNAME = "autoinit_c2_replay"
STATUS = f"{WS}/scratch/autoinit_c2_replay.status"
RUN_LOG = f"{WS}/scratch/autoinit_c2_replay_run.log"

#: The relay inputs the search itself used. Both calibration mixtures are
#: needed: the five paths between them name `calib.domain_balanced@v1` and
#: `calib.reasoning_heavy@v2`, and an operator resolves its profile from disk,
#: hash-verified.

#: Disk. Derived from the five paths this session actually runs, NOT inherited
#: from the full search's 400 GB: the worst single path's intermediates measured
#: 16.12 GiB, the teacher is 7.29 GiB, the five retained leaves are 5.55 GiB, and
#: the image and environment want room of their own. 120 GB is comfortable for
#: all of it and costs about $0.04 over a session of this length.
CONTAINER_DISK_GB = 120

#: The image attempt 3 ran. The replay reproduces artifacts that
#: image produced, so it is pinned rather than defaulted.
BOUND_IMAGE = POD_IMAGE


def governance_path(run_id: str, name: str) -> str:
    return f"{rel_run_dir(EXPERIMENT_ID, run_id, STAGE_ID)}/governance/{name}"


def auth_path_for(run_id: str) -> str:
    return governance_path(run_id, "authorization.json")


def bundle_record_for(run_id: str) -> str:
    return governance_path(run_id, "bundle.json")


# -- gates ------------------------------------------------------------------
def replay_scope_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The authorization must permit this session and nothing adjacent to it."""
    a = ctx.auth
    forbidden = {
        "authorizes_c2_full_search": a.authorizes_c2_full_search,
        "authorizes_c2_search1": a.authorizes_c2_search1,
        "authorizes_c2_baseline_completion": a.authorizes_c2_baseline_completion,
        "authorizes_behavioural_selection": a.authorizes_behavioural_selection,
        "allows_phase_a": a.allows_phase_a,
        "allows_recovery_training": a.allows_recovery_training,
        "automatic_followon_start": a.automatic_followon_start,
    }
    claimed = [k for k, v in forbidden.items() if v]
    if claimed:
        return False, (
            f"this artifact claims {claimed}. A replay reconstructs artifacts "
            "behind a frozen selection; it may not authorize a beam, a rebuild, "
            "behavioural work or a follow-on.")
    if not a.authorizes_c2_replay:
        return False, "this artifact does not authorize a replay"
    if tuple(a.authorized_stages) != RG.AUTHORIZED_STAGES:
        return False, (f"authorized stages {tuple(a.authorized_stages)} are not "
                       f"{RG.AUTHORIZED_STAGES}")
    return True, (f"replay-only scope confirmed; stages "
                  f"{list(a.authorized_stages)}, hard ${a.hard_cap_usd:.4f}")


def source_binding_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The pins must still describe the tree this session will run.

    `assert_operators_unmoved` is the $0 half of the digest gate. If an operator
    moved since attempt 3, every path diverges on the pod and the session pays to
    learn what a `git ls-tree` says for free.
    """
    try:
        report = RS.assert_operators_unmoved(REPO_ROOT)
    except RS.ReplaySourceError as exc:
        return False, str(exc)
    binding = RS.source_binding(REPO_ROOT)
    if binding["selection_sha256"] != RS.SELECTION_SHA256:
        return False, "the committed selection is not the one being replayed"
    if len(binding["leaves"]) != 5:
        return False, f"{len(binding['leaves'])} leaves to replay, expected 5"
    unpinned = [leaf["state_id"] for leaf in binding["leaves"]
                if len(leaf["step_digests"]) != 4]
    if unpinned:
        return False, f"leaves without a complete four-step pin: {unpinned}"
    return True, (f"{report['verdict']} since "
                  f"{binding['source_session_commit'][:12]}…; 5 leaves, "
                  f"20 pinned steps, selection "
                  f"{binding['selection_sha256'][:12]}…")


def plan_binding_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The authorization's plan hash IS the source binding's hash.

    So an authorization issued against a different set of pinned paths cannot
    run this session, and this session cannot silently change what it replays.
    """
    expected = RG.plan_hash(REPO_ROOT)
    try:
        ctx.auth.require_plan(expected)
    except Exception as exc:                                    # noqa: BLE001
        return False, (f"the authorization does not bind to this session's "
                       f"source binding ({expected[:12]}…): {exc}")
    return True, f"authorization binds to source binding {expected[:12]}…"


def destination_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Room for what this session is protecting, before it is created.

    A durability mechanism whose backend cannot hold its artifacts preserves
    nothing, correctly, and reports success. Six 2.22 GiB probes were lost that
    way in C1 attempt 18. Five leaves are 5.55 GiB.
    """
    import shutil

    store = Path(ctx.args.ckpt_store)
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    need = 5 * RS_LEAF_BYTES
    if free < need * 2:
        return False, (
            f"{store} has {free / 2**30:.1f} GiB free and the five leaves are "
            f"{need / 2**30:.2f} GiB. Refusing to start a session whose product "
            "has nowhere to land.")
    return True, (f"destination {store}: {free / 2**30:.1f} GiB free for "
                  f"{need / 2**30:.2f} GiB of leaves")


def readiness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A launch-bound sweep of THIS tree, recorded and committed."""
    path = REPO_ROOT / governance_path(ctx.args.run_id, "readiness.json")
    if not path.is_file():
        return False, f"no readiness record at {path.relative_to(REPO_ROOT)}"
    record = json.loads(path.read_text())
    if record.get("record_kind") != "launch_bound":
        return False, (f"readiness record is {record.get('record_kind')!r}, "
                       "not 'launch_bound'")
    swept = record.get("swept_base_commit")
    if swept != ctx.auth.authorized_session_commit:
        return False, (f"the sweep describes {str(swept)[:12]}… and the "
                       f"authorization is for "
                       f"{ctx.auth.authorized_session_commit[:12]}…")
    if not record.get("ok"):
        return False, "the readiness sweep did not pass"
    return True, f"launch-bound sweep of {str(swept)[:12]}… passed"


def bundle_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Can a pod, RIGHT NOW, obtain the exact authorized code?

    Every other gate verifies the CONTENTS of a commit; this one asks whether
    the pod can reach it at all. C1 attempt 1 answered every other question
    correctly and died at `SETUP_RC=1` fetching an alias for nothing — eight
    gates verified the commit and none that a bundle for it could be fetched.

    Read-only: it uploads nothing. LAST, because it is the only gate that
    touches the network and everything it verifies against must already be
    checked.
    """
    run_id = getattr(ctx.args, "run_id", None)
    commit = ctx.args.session_commit
    try:
        RT.require_canonical_bundle_arg(ctx.args.bundle, commit)
    except RT.BundleTransportError as exc:
        return False, str(exc)

    bundle_rel = bundle_record_for(run_id)
    staged = REPO_ROOT / bundle_rel
    if not staged.is_file():
        return False, (f"{bundle_rel} is missing; stage the canonical bundle "
                       f"for {commit[:12]}… first")
    record = json.loads(staged.read_text())
    if record.get("session_commit") != commit:
        return False, (f"{bundle_rel} describes a bundle for "
                       f"{str(record.get('session_commit'))[:12]}…, not the "
                       f"session commit {commit[:12]}…")

    auth_rel = auth_path_for(run_id)
    auth_file = REPO_ROOT / auth_rel
    if not auth_file.is_file():
        return False, (f"{auth_rel} does not exist, so there is no "
                       "authorization for the round-trip to find in the bundle")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = RT.roundtrip(
                session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=auth_file.read_bytes(),
                authorization_path=auth_rel,
                #: The AUTHORIZED pair, not the live one. Asking the round-trip
                #: about the live digest would make a stale authorization
                #: unfalsifiable here.
                expected_harness_digest=ctx.auth.harness_source_digest,
                harness_files=tuple(ctx.auth.harness_source_files),
                workdir=Path(tmp))
    except Exception as exc:                                    # noqa: BLE001
        return False, f"the pod could not obtain the authorized commit: {exc}"

    ctx.evidence["bundle_staged_check"] = evidence
    return True, (f"{evidence['canonical_bundle_name']} "
                  f"({evidence['bytes']} bytes, "
                  f"{evidence['remote_sha256'][:12]}…) round-trips to "
                  f"{evidence['roundtrip_head'][:12]}… carrying this "
                  f"authorization and executable set "
                  f"{evidence['roundtrip_harness_digest'][:12]}…")


RS_LEAF_BYTES = 1_192_135_096


# -- products ---------------------------------------------------------------
def reconstructed_leaves(ctx: SessionContext) -> list[dict]:
    """What the driver says it reconstructed, read from the evidence it wrote."""
    evidence = (Path(ctx.args.scr) / "autoinit_c2_replay"
                / "c2_replay_evidence.json")
    if not evidence.is_file():
        return []
    try:
        record = json.loads(evidence.read_text())
    except json.JSONDecodeError:
        return []
    return [leaf for leaf in record.get("leaves", [])
            if leaf.get("identity_matches_attempt3")]


def fetch_leaves(ctx: SessionContext) -> list:
    """Pull every reconstructed leaf and re-identify it from the bytes that land.

    Verification happens HERE, on the destination, not on the pod: staging a
    leaf beside the process that made it is a copy that dies with the pod, and a
    transfer that truncated a shard is exactly what a pod-side check cannot see.
    """
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.runtime.leaf_durability import (
        LeafDurabilityError, verify_transferred_leaf,
    )

    fetched: list = []
    if not ctx.products_eligible:
        ctx.say("  no leaf was reconstructed; nothing to fetch")
        return fetched

    adapter = get_adapter("qwen3")
    store = Path(ctx.args.ckpt_store) / "phase_c2_full_search" / "attempt3_replay"
    for leaf in reconstructed_leaves(ctx):
        state_id = leaf["state_id"]
        dest = store / state_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            ["timeout", f"{ctx.args.ckpt_fetch_limit_min}m", "scp", "-r",
             "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"root@{ctx.host}:{LEAF_DIR}/{state_id}", str(dest)],
            capture_output=True, timeout=None)
        size = (sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
                if dest.exists() else 0)

        matched, why = False, "not verified"
        if rc.returncode == 0:
            sidecar = dest / "replay_leaf.json"
            try:
                record = json.loads(sidecar.read_text())["identity"]
                verify_transferred_leaf(dest, record, adapter=adapter)
                matched, why = True, "re-identified from the delivered bytes"
            except (LeafDurabilityError, OSError, KeyError,
                    json.JSONDecodeError) as exc:
                why = f"{type(exc).__name__}: {exc}"
        else:
            why = f"scp rc={rc.returncode}: {(rc.stderr or b'')[-200:]!r}"

        fetched.append({"artifact": "c2_replay_leaf", "state_id": state_id,
                        "rc": rc.returncode, "bytes": size, "dest": str(dest),
                        "matched": matched, "why": why})
        ctx.say(f"  leaf {state_id[:12]}…: rc={rc.returncode}, "
                f"{size / 2**30:.2f} GiB -> {dest} [{why}]")
    return fetched


def leaves_secured(ctx: SessionContext, fetched: list) -> tuple[bool, str]:
    """Teardown may not proceed while a reconstructed leaf is only on the pod."""
    want = {leaf["state_id"] for leaf in reconstructed_leaves(ctx)}
    if not want:
        return True, "no leaf was reconstructed, so none is owed off-pod"
    got = {f["state_id"] for f in fetched
           if isinstance(f, Mapping) and f.get("artifact") == "c2_replay_leaf"
           and f.get("rc") == 0 and f.get("matched")}
    missing = sorted(want - got)
    if missing:
        return False, (
            f"{len(want)} leaves were reconstructed and {len(got)} are verified "
            f"off-pod; missing {missing}. Deleting the pod now would destroy "
            "the only copies of work this session already paid for — which is "
            "the exact failure it exists to repair.")
    return True, f"all {len(want)} reconstructed leaves verified off-pod"


def driver_command(ctx: SessionContext, plan) -> str:
    return (f"/opt/train/bin/python {REPO}/scripts/pod/autoinit_c2_replay_driver.py "
            f"--workdir {WORKDIR} --leaf-dir {LEAF_DIR} "
            f"--image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--already-spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f}")


def budget(args) -> BudgetSpec:
    """Built from the five paths this session actually runs.

    `arms=0`: a replay trains nothing, so the step term multiplies out.

    The reconstruction phase is the sum of the per-path BOUNDS, each derived
    from the worst observation of its operator kind anywhere in attempt 3's
    telemetry — not from the measured per-step values, which include cache hits
    a fresh replay will not get. One selected path recorded DEPTH at 0.6 minutes
    where the same operator took 25.7 minutes elsewhere; pricing from that would
    underestimate the path tenfold.

    **This plan does not fit the $5.00 ceiling, and it is not what bounds the
    session.** The phase sum is deliberately conservative; what keeps the money
    inside the authorization is the driver's admission control, which starts a
    path only when the remaining soft-stop budget can fund that path's full
    bound. The consequence is explicit and acceptable: under worst-case timings
    the session secures fewer than five leaves rather than overspending, and
    every leaf it does secure is verified off-pod. Under attempt 3's measured
    timings all five fit comfortably.
    """
    from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase

    from experiments.phase_c2.full_search_space import SESSION_PHASE_MINUTES

    phases = dict(SESSION_PHASE_MINUTES)
    leaves = RS.build_replay_leaves(REPO_ROOT, device="cuda")
    reconstruction = sum(leaf.bounded_minutes for leaf in leaves)

    #: 5.55 GiB at the measured 11.5 MB/s dev-box downlink is 8.6 minutes; the
    #: reserve is that, with margin, rather than the generic 30.
    leaf_transfer = 15.0

    return BudgetSpec(
        arms=0, steps_per_arm=0,
        #: `arms=0` multiplies the step term out entirely. The measured floor is
        #: passed because the below-floor guard refuses a zero, not because a
        #: step time means anything to a session that trains nothing.
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: a replay trains nothing, so arms=0. The bound is "
                     "the sum of the five paths' worst-case operator timings "
                     "from attempt 3's telemetry"),
        setup_minutes=phases["setup_and_asset_staging"],
        transfer_minutes=phases["bundle_transfer"],
        other_phases=(
            *(Phase(name, minutes) for name, minutes in SESSION_PHASE_MINUTES
              if name not in ("setup_and_asset_staging", "bundle_transfer")),
            Phase("reconstruct_five_pinned_paths", round(reconstruction, 2)),
        ),
        contingency_fraction=0.10,
        artifact_recovery_reserve_minutes=leaf_transfer,
    )


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id=RG.SESSION_ID,
        schema="aadistill.autoinit.c2_replay_session/v1",
        description=(
            "Replay-only reconstruction of the five checkpoints behind Phase-C2 "
            "Full Joint Search attempt 3's frozen Top-5. Every path is pinned at "
            "every step to the artifact digest attempt 3 recorded. It runs no "
            "beam, ranks nothing, produces no selection, evaluates nothing that "
            "bears on a decision, injects no control and trains nothing. A "
            "digest mismatch is a scientific finding and stops the session."),
        authorization_path=auth_path_for(getattr(args, "run_id", "")),
        authorization_loader=RG.ReplayAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=RG.PLAN_ID,
        plan_hash=RG.plan_hash(REPO_ROOT),
        budget=budget(args),
        setup=SetupManifest(
            relay_inputs=CALIBRATION_V1,
            local_assets=(),
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "SESSION_SETUP_MARKERS",
                          "TEACHER_REVISION"),
            #: ROPE_OK is ABSENT for the reason two sessions have already paid
            #: to establish: the shared step globs a STAGED student checkpoint
            #: and exits 1 when none matches. This session stages none — it
            #: materializes every state on the pod from the teacher — so the
            #: step would refuse. The risk it covers is handled where the
            #: artifact exists: every materialized state is canonically
            #: reloaded and identity-checked inside `materialize_fixed_path`,
            #: against a pinned digest, which is strictly stronger here.
            #: VLLM_READY is ABSENT: this session serves nothing.
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "TEACHER_READY",
                           "TESTS_OK", "AUTHORIZATION_OK", "SETUP_DONE"),
            env={"SESSION_KIND": "c2_replay"},
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION),
        driver_command=driver_command,
        driver_job_id="autoinit_c2_replay_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="C2_REPLAY_ALL_DONE",
            failure=("C2_REPLAY_FAILED",),
            incomplete=(),
            #: Gated on leaves EXISTING, never on the session succeeding. A run
            #: that reconstructed three leaves and then hit a mismatch has three
            #: checkpoints that cost real GPU time and reproduce exactly.
            products_eligible=lambda terminal, stages: True,
            failure_note=(
                "the replay stopped before reconstructing all five leaves. "
                "Whatever WAS reconstructed is fetched and verified first; a "
                "digest mismatch is a scientific finding, not a retryable "
                "engineering failure, and must go to review unchanged.")),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="c2_replay_evidence.json",
            archive_basename="c2_replay_artifacts.tar.gz",
            spec_success="configs/autoinit/c2_replay_artifacts.json",
            spec_failed="configs/autoinit/c2_replay_artifacts_failed.json",
            report_names=("c2_replay_evidence.json",),
            fetch_products=fetch_leaves,
            products_secured=leaves_secured),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
        precheck=(
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", "")),
                                check_lineage=True),
            replay_scope_gate,
            plan_binding_gate,
            source_binding_gate,
            destination_gate,
            readiness_gate,
            #: LAST, because it is the only gate that touches the network and
            #: everything it verifies against must already be checked.
            bundle_staged_gate,
        ),
    )


RUN_OUTPUTS: tuple[str, ...] = (
    "autoinit_c2_replay", "c2_replay_artifacts.tar.gz", "watchdog_*.jsonl")


def build_parser() -> argparse.ArgumentParser:
    """The real parser.

    It must supply every name in the runner's argument contract:
    `SessionRunner.__init__` reads those attributes and refuses a namespace
    missing any of them, before provider creation. The first version of this
    file omitted `--scr`, `--session-commit` and `--bundle`, which made the
    launcher unparseable and dropped it silently out of the dispatch enumeration
    — the exact failure `test_no_launcher_was_dropped_because_it_would_not_parse`
    exists to name.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--run-id", required=True,
                    help="the attempt this session runs as, e.g. attempt1")
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-transport")
    ap.add_argument("--image", default=BOUND_IMAGE)
    #: L40S and not a cheaper 48 GB card. This session's whole content is
    #: byte-exact agreement with artifacts produced on an L40S; a different
    #: architecture can select different kernels and present as a replay
    #: mismatch for a purely infrastructural reason.
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=None,
                    help="defaults to the authorization's own accepted rate")
    ap.add_argument("--disk-gb", type=int, default=CONTAINER_DISK_GB)
    ap.add_argument("--ckpt-store", default="/home/ecs-user/aad-artifacts",
                    help="the development host's out-of-tree artifact store")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=30)
    ap.add_argument("--token-src",
                    default=str(Path.home() / ".cache/huggingface/token"))
    ap.add_argument("--runpod-config",
                    default=str(Path.home() / ".runpod/config.toml"))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--host-draws", type=int, default=2)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    #: Sized to the derived hard window, not to a generic hour.
    ap.add_argument("--poll-limit-min", type=float, default=330.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every $0 gate and stop before provider creation")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
