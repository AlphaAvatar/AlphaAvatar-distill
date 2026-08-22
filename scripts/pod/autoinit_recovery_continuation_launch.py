#!/usr/bin/env python3
"""The Phase-A recovery continuation, as a session specification.

    PYTHONPATH=src setsid nohup python -u \
        scripts/pod/autoinit_recovery_continuation_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

**This session does not search.** Attempts 11 and 12 produced byte-identical
Stage-1 results and attempt 12 preserved the five selected checkpoints off-pod;
this consumes that result and runs Stages 2-5. Two consequences are declared
rather than assumed:

*It is priced without the search.* `continuation_budget` removes the
`stage1_beam_search` phase and both Stage-1-only reserves — the beam-6 pricing
correction and the reference-cache fallback — from the full Phase-A `BudgetSpec`.
Nothing here writes a dollar figure; the numbers come from the same `plan()`
that prices the full session, so the two cannot drift.

*It is authorized against its own harness.* The continuation loads a
`RecoveryContinuationAuthorization`, not a Phase-A one, and the two are refused
across by schema. A Phase-A artifact measures the search harness — a file set
containing neither this launcher nor its driver — and carries the search's
$23.0484 ceiling; accepting one here would certify code this session does not
run, leave the code it does run unmeasured, and price it for work it does not do.
`continuation_harness_gate` recomputes the continuation file set at $0, before a
pod exists, so an artifact that declares the wrong list cannot verify itself.

*The five leaves are session inputs.* They are declared by state id, in the
selected order, with the artifact and shard digests attempt 12 recorded, staged
from the canonical local checkpoint store through the ordinary local-asset
contract, and **re-identified from bytes again on the pod** before the strict
importer will use them. A continuation must not depend on an undeclared host
path or silently accept a different leaf.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.autoinit.phase_a import (  # noqa: E402
    PHASE_A_PLAN_V1, PHASE_A_SCOPE,
)
from aadistill.autoinit.recovery_continuation import (  # noqa: E402
    RECOVERY_CONTINUATION_HARNESS_FILES_V1, RecoveryContinuationAuthorization,
    SEARCH_ONLY_HARNESS_FILES, recovery_continuation_harness_digest,
)
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, MarkerPolicy, RelayInput, SessionContext, SessionSpec,
    SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    frozen_science_plan_gate, session_commit_gate,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
from autoinit_science_inputs import (  # noqa: E402
    CALIBRATION_V1, CANONICAL_INIT, RECOVERY_LADDER,
)
#: The launcher owns the pricing; the continuation derivation lives beside the
#: full one so the two are visibly the same arithmetic minus the search.
from autoinit_phase_a_launch import (  # noqa: E402
    LOCAL_ASSETS as PHASE_A_LOCAL_ASSETS, TEST_IGNORES, TEACHER_REVISION,
    continuation_budget, finalists_to_fetch, probe_streams,
    selected_leaves_secured,
)

STATUS = f"{WS}/autoinit_recovery_continuation.status"
RUN_LOG = f"{WS}/autoinit_recovery_continuation_run.log"
AUTH_PATH = "logs/autoinit_recovery_continuation_authorization.json"
FROZEN_SCIENCE_PLAN = "logs/autoinit_phase_a_recovery_plan_frozen.json"
#: Attempt 12's committed durability record — the five ids, in order, with the
#: digests the bytes must reproduce.
STAGE1_EVIDENCE = REPO_ROOT / "logs/autoinit_phase_a_attempt12"
#: The canonical local checkpoint store. Still the scientific owner: the
#: transport repo is a delivery path and nothing more.
CKPT_STORE = Path("/home/ecs-user/aad-artifacts/autoinit/phase_a")
#: Transport only, private, and verified at $0 before any paid session may use
#: it -- see logs/autoinit_selected_leaf_transport_manifest.json.
TRANSPORT_REPO = "AlphaAvatar/aadistill-transport"
TRANSPORT_MANIFEST = REPO_ROOT / "logs/autoinit_selected_leaf_transport_manifest.json"
#: Where they land in the pod's repository, and where the driver reads them.
STAGED_INTO = "artifacts/autoinit/phase_a_selected"


def selected_leaf_identities() -> list[dict]:
    """The five leaves this session is bound to, in the selected order.

    Read from the committed record rather than restated here: a second copy of
    five state ids is a second thing to keep in step, and the order is the
    ranking.
    """
    dur = json.loads((STAGE1_EVIDENCE / "selected_leaf_durability.json").read_text())
    return [{"state_id": r["state_id"],
             "artifact_digest": r["artifact_digest"],
             "single_shard_sha256": r.get("single_shard_sha256")}
            for r in dur["leaves"]]


def transport_is_verified() -> bool:
    """Has the transport manifest been written AND round-trip verified?

    Publishing the leaves is a separate `$0` step
    (`scripts/autoinit/publish_selected_leaves.py`). Until it has verified all
    five remote copies against attempt 12's identities, this session has no
    usable transport and must not launch.
    """
    if not TRANSPORT_MANIFEST.is_file():
        return False
    try:
        man = json.loads(TRANSPORT_MANIFEST.read_text())
    except json.JSONDecodeError:
        return False
    return bool(man.get("verified")) and man.get("repo") == TRANSPORT_REPO


def selected_leaf_inputs() -> tuple[RelayInput, ...]:
    """The five preserved checkpoints, PULLED from the transport repo.

    They used to be `LocalAsset`s, pushed by scp — and that is what ended
    continuation attempt 2. The launcher stages each local asset with a
    hard-coded 600 s timeout, and one 1.110 GiB leaf needs **1.99 MB/s** to fit
    it against a dev box observed at 0.44-0.72 MB/s. The first leaf could not
    have arrived, and four more would have followed.

    So the slow half moved off the paid path entirely: the bytes were published
    to a private transport repo at `$0` on the dev box, and the pod now pulls
    them at hub speed through the ordinary declared-manifest contract. The
    per-file digests come from the transport manifest, which itself only
    reproduces attempt 12's committed identities — nothing here is a new
    scientific claim, and `artifacts/autoinit/phase_a_selected/<state_id>` is
    still where the strict importer looks.

    One `RelayInput` per FILE, because that is what the relay contract stages;
    the directory is reassembled by every file naming the same `dest`.
    """
    if not transport_is_verified():
        # NOT a spec-build failure: whether bytes are published is runtime state,
        # and a session that cannot be *described* cannot be structurally
        # checked either. Declaring nothing here means the $0 leaf gate refuses
        # with the reason, which is where every other "this cannot run" lives —
        # and it is still strictly before a pod exists.
        return ()
    man = json.loads(TRANSPORT_MANIFEST.read_text())
    out = []
    for rec in sorted(man["leaves"], key=lambda r: r["selected_order"]):
        for f in rec["files"]:
            out.append(RelayInput(
                path=f["remote_path"], repo=TRANSPORT_REPO,
                dest=f"{STAGED_INTO}/{rec['state_id']}", sha256=f["sha256"]))
    return tuple(out)


#: Only genuinely small dev-box artifacts stay on the scp path. The leaves left
#: it; nothing else joined it.
LOCAL_ASSETS = PHASE_A_LOCAL_ASSETS


def driver_command(ctx: SessionContext, plan) -> str:
    """The continuation driver. There is no `--stage` value that searches."""
    return (f"/opt/train/bin/python "
            f"{REPO}/scripts/pod/autoinit_recovery_continuation_driver.py "
            f"--stage all --image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f} "
            f"--probe-train-minutes {ctx.args.probe_train_minutes} "
            f"--probe-battery-minutes {ctx.args.probe_battery_minutes}")


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-recovery-continuation",
        schema="aadistill.autoinit.recovery_continuation_session/v1",
        description=("Phase-A recovery continuation: Stage 1 imported from the "
                     "verified attempt-12 result, then Stages 2-5. Runs no "
                     "search"),
        authorization_path=AUTH_PATH,
        #: The CONTINUATION type, refused by schema. A full-Phase-A artifact
        #: measures the search harness and carries the search's $23.0484
        #: ceiling; loading one here would authorize this session against a
        #: digest that never read its launcher, driver or importer, at a price
        #: derived for work it does not do.
        authorization_loader=RecoveryContinuationAuthorization.load,
        #: The SAME frozen plan. This session is a different operational
        #: identity, not a different science: nothing here rewrites 9377a2dc to
        #: pretend Phase A always began at stage 2.
        plan_id=PHASE_A_PLAN_V1.plan_id,
        plan_hash=PHASE_A_PLAN_V1.plan_hash,
        #: PRICED WITHOUT THE SEARCH. 904.44 expected minutes, $14.9233 expected,
        #: $16.7456 hard — derived, never written.
        budget=continuation_budget(args),
        setup=SetupManifest(
            relay_inputs=(*CANONICAL_INIT, *RECOVERY_LADDER, *CALIBRATION_V1,
                          *selected_leaf_inputs()),
            local_assets=LOCAL_ASSETS,
            #: Declared, not defaulted. Without it the setup script falls to
            #: `SESSION_KIND=spend` and loads a `SpendAuthorization`, which
            #: refuses this artifact — the session would have died at setup,
            #: exit 98, before any work.
            env={"SESSION_KIND": "recovery_continuation"},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "VLLM_READY",
                           "TEACHER_READY", "ROPE_OK", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_recovery_continuation",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("PHASE_A_FAILED", "PHASE_A_INCOMPLETE"),
            incomplete=("PHASE_A_INCOMPLETE",),
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. The imported stage-1 leaves are inputs "
                          "here and are not re-derived or modified.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_phase_a",
            evidence_filename="phase_a_evidence.json",
            archive_basename="phase_a_artifacts.tar.gz",
            spec_success="configs/autoinit/phase_a_artifacts.json",
            spec_failed="configs/autoinit/phase_a_artifacts_failed.json",
            report_names=("phase_a_evidence.json",
                          "attested_evaluation_protocol.json",
                          "stage1_import.json", "control_measurement.json",
                          "device_handoff.json", "rung1_selection.json",
                          "leaf_retention.json",
                          "rung2_selection.json", "phase_a_result.json"),
            event_streams=probe_streams,
            #: No stage-1 leaves to bring home: they arrived as inputs. Only the
            #: finalists this session produces travel back.
            fetch_products=finalists_to_fetch,
            products_secured=selected_leaves_secured),
        teardown=TeardownPolicy(
            note="nothing chains off the recovery continuation"),
        precheck=(
            session_commit_gate(REPO_ROOT, AUTH_PATH, check_lineage=True),
            frozen_science_plan_gate(REPO_ROOT, FROZEN_SCIENCE_PLAN),
            continuation_harness_gate,
            selected_leaves_present_gate,
        ),
        evidence_fields={
            "phase_a_session_plan_hash": PHASE_A_PLAN_V1.plan_hash,
            "scope": PHASE_A_SCOPE.as_dict(),
            "runs_a_search": False,
            "stage1_source": "imported from phase-a attempt 12, verified from bytes",
            "selected_state_ids": [l["state_id"] for l in selected_leaf_identities()],
            "retrains_permanent_controls": False,
            "followon_started": False,
            "followon_reachable_from_this_launcher": False},
    )


def continuation_harness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Recompute the continuation harness set here, independently of the artifact.

    `require_harness()` digests whatever file list the authorization *stores*,
    which is what lets a session declare its own executable — and is also the one
    thing an artifact could get wrong in its own favour. An authorization
    carrying the Phase-A file list would verify perfectly against the Phase-A
    files while this launcher, this driver and the strict importer went
    unmeasured.

    So this asks the question from the other side, at $0, before a pod exists:
    is the set the artifact declares *the set this session runs*, and does the
    code on disk digest to what was authorized?
    """
    declared = tuple(ctx.auth.harness_source_files)
    expected = RECOVERY_CONTINUATION_HARNESS_FILES_V1
    ev = ctx.evidence.setdefault("precheck", {})
    ev["continuation_harness"] = {
        "n_files": len(expected),
        "covers_search": bool(set(declared) & set(SEARCH_ONLY_HARNESS_FILES)),
    }
    if set(declared) != set(expected):
        missing = sorted(set(expected) - set(declared))
        extra = sorted(set(declared) - set(expected))
        return False, (
            f"the authorization declares a different harness set: missing "
            f"{missing}, unexpected {extra}. A digest over the wrong file list "
            "certifies code this session does not run and leaves code it does "
            "run unmeasured.")

    observed = recovery_continuation_harness_digest(REPO_ROOT)
    ev["continuation_harness"]["digest"] = observed["digest"]
    if observed["digest"] != ctx.auth.harness_source_digest:
        return False, (
            f"the continuation harness on disk digests to {observed['digest']} "
            f"but the authorization was granted against "
            f"{ctx.auth.harness_source_digest}. Re-rehearse, re-commit, re-issue.")
    if ctx.auth.allows_beam_search:
        return False, ("this artifact claims to allow a beam search; the "
                       "continuation imports Stage 1 and cannot reach one")
    return True, (f"continuation harness {observed['digest'][:12]}… over "
                  f"{len(expected)} files, search excluded")


def selected_leaves_present_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Are the five canonical leaves intact, and does the transport declare them?

    Asked at $0, before a pod exists. Two questions, deliberately separate now
    that the bytes travel by relay:

    * the **canonical** copies under `CKPT_STORE` still reproduce attempt 12's
      identities — they remain the scientific owner, and the transport manifest
      is only meaningful if what it was built from is still right;
    * the session's declared relay inputs cover **exactly** those five state ids,
      in the frozen selected order, with the digests the manifest recorded.

    Whether those paths actually EXIST in the transport repo is the runner's
    multi-repo relay precheck, which lists every declared repository. This gate
    does not duplicate it.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from aadistill.autoinit.arch import get_adapter
    from aadistill.autoinit.leaf_durability import verify_transferred_leaf

    dur = json.loads((STAGE1_EVIDENCE / "selected_leaf_durability.json").read_text())
    adapter = get_adapter("qwen3")
    bad = []
    for rec in dur["leaves"]:
        path = CKPT_STORE / rec["state_id"]
        if not path.is_dir():
            bad.append(f"{rec['state_id'][:12]}: absent"); continue
        try:
            v = verify_transferred_leaf(path, rec, adapter=adapter)
            if not (v["matched"] and v["shard_matched"]):
                bad.append(f"{rec['state_id'][:12]}: digest mismatch")
        except Exception as exc:                       # noqa: BLE001
            bad.append(f"{rec['state_id'][:12]}: {type(exc).__name__}")
    # The declared transport must cover exactly these five, in this order.
    if not transport_is_verified():
        ctx.evidence.setdefault("precheck", {})["selected_leaves"] = {
            "store": str(CKPT_STORE), "transport_repo": TRANSPORT_REPO,
            "transport_verified": False}
        return False, (
            f"the five leaves have no verified transport: {TRANSPORT_MANIFEST} "
            "is absent or not marked verified. Publish and verify them with "
            "scripts/autoinit/publish_selected_leaves.py before launching; the "
            "canonical copies alone cannot reach a pod.")
    inputs = selected_leaf_inputs()
    man = json.loads(TRANSPORT_MANIFEST.read_text())
    want_order = [r["state_id"] for r in dur["leaves"]]
    got_order = [r["state_id"] for r in sorted(man["leaves"],
                                               key=lambda x: x["selected_order"])]
    if got_order != want_order:
        bad.append(f"transport order {got_order} != selected order {want_order}")
    by_state: dict[str, set] = {}
    for r in inputs:
        by_state.setdefault(r.dest.rsplit("/", 1)[-1], set()).add(r.repo)
    if set(by_state) != set(want_order):
        bad.append(f"transport covers {sorted(by_state)}, expected {want_order}")
    off = {s: sorted(v) for s, v in by_state.items() if v != {TRANSPORT_REPO}}
    if off:
        bad.append(f"leaves declared from an unexpected repo: {off}")
    # Every declared digest must be one the manifest recorded for that leaf.
    recorded = {f["sha256"] for rec in man["leaves"] for f in rec["files"]}
    stray = [r.path for r in inputs if r.sha256 not in recorded]
    if stray:
        bad.append(f"relay inputs with digests absent from the manifest: {stray}")

    ctx.evidence.setdefault("precheck", {})["selected_leaves"] = {
        "store": str(CKPT_STORE), "n": len(dur["leaves"]),
        "transport_repo": TRANSPORT_REPO, "transport_files": len(inputs),
        "transport_verified": bool(man.get("verified")), "problems": bad}
    if bad:
        return False, (
            f"the preserved stage-1 leaves are not usable: {bad}. This session "
            "imports them rather than searching; without them there is nothing "
            "to continue from.")
    return True, (f"all {len(dur['leaves'])} canonical stage-1 leaves verify "
                  f"locally; {len(inputs)} transport files declared from "
                  f"{TRANSPORT_REPO} in selected order")


def build_parser() -> argparse.ArgumentParser:
    from autoinit_phase_a_launch import build_parser as phase_a_parser

    ap = phase_a_parser()
    ap.set_defaults(out="logs/autoinit_recovery_continuation_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("STOP for review. The continuation consumed the "
                                "verified attempt-12 Stage-1 result and ran no "
                                "search."))


if __name__ == "__main__":
    raise SystemExit(main())
