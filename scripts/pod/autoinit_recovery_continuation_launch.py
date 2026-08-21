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
    PHASE_A_PLAN_V1, PHASE_A_SCOPE, PhaseAAuthorization,
)
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, LocalAsset, MarkerPolicy, SessionContext, SessionSpec,
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
#: The canonical local checkpoint store the leaves were preserved into.
CKPT_STORE = Path("/home/ecs-user/aad-artifacts/autoinit/phase_a")
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


def selected_leaf_assets() -> tuple[LocalAsset, ...]:
    """The five preserved checkpoints, declared as ordinary session inputs.

    Through the same `SESSION_ASSETS` contract every other local asset uses, so
    the setup script stages them without knowing what they are — and so a
    session that failed to declare them gets none rather than silently finding
    them on a host path.
    """
    return tuple(LocalAsset(str(CKPT_STORE / leaf["state_id"]),
                            leaf["state_id"], STAGED_INTO)
                 for leaf in selected_leaf_identities())


LOCAL_ASSETS = (*PHASE_A_LOCAL_ASSETS, *selected_leaf_assets())


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
        authorization_loader=PhaseAAuthorization.load,
        #: The SAME frozen plan. This session is a different operational
        #: identity, not a different science: nothing here rewrites 9377a2dc to
        #: pretend Phase A always began at stage 2.
        plan_id=PHASE_A_PLAN_V1.plan_id,
        plan_hash=PHASE_A_PLAN_V1.plan_hash,
        #: PRICED WITHOUT THE SEARCH. 904.44 expected minutes, $14.9233 expected,
        #: $16.7456 hard — derived, never written.
        budget=continuation_budget(args),
        setup=SetupManifest(
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


def selected_leaves_present_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Are the five preserved leaves on this host, with the right bytes?

    Asked at $0, before a pod exists. Staging is what puts them on the pod; this
    asks whether there is anything to stage, and whether it is the right thing.
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
    ctx.evidence.setdefault("precheck", {})["selected_leaves"] = {
        "store": str(CKPT_STORE), "n": len(dur["leaves"]), "problems": bad}
    if bad:
        return False, (
            f"the preserved stage-1 leaves are not usable: {bad}. This session "
            "imports them rather than searching; without them there is nothing "
            "to continue from.")
    return True, f"all {len(dur['leaves'])} preserved stage-1 leaves verify locally"


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
