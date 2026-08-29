#!/usr/bin/env python3
"""Launch the Phase-B BEHAVIOURAL CONTINUATION. It cannot buy the search.

    python3 scripts/pod/autoinit_continuation_b_launch.py --dry-run

Phase-B Stage 1 is complete: attempt 5 emitted an authoritative Top-5, a durable
Stage-1 selection artifact and a retained journal, and rung 1 was finished at `$0`
under the identity-collapse amendment. This session buys **one missing `sb`** and
**at most two conditional `sc`** — one to three probes against the ten a full
Phase B books, and no search at all.

Three things keep that honest.

**A different authorization type.** `ContinuationAuthorization.runs_search` is
`False` by type — there is no field to set — so a continuation grant cannot
authorize a search and a full Phase-B grant cannot govern this session. Passing
either to the other is a schema refusal.

**A different ceiling.** The `$35.6660` full-Phase-B figure prices a 16.5 h search
that has already been bought. This session is priced from what is actually
missing, and the launcher refuses if the two are confused.

**A different executable.** `CONTINUATION_SOURCE_FILES_V2` covers what a paid
continuation runs and deliberately excludes the search, the operators and the
ranking policy: this session runs none of them, and a digest covering them would
imply it might.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V2,
)
from aadistill.autoinit.phase_b_continuation import (  # noqa: E402
    CONTINUATION_PLAN_V1, ContinuationAuthorization, continuation_source_digest,
)
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, MarkerPolicy, RelayInput, SessionContext, SessionSpec,
    SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    frozen_science_plan_gate, session_commit_gate,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
from autoinit_science_inputs import CANONICAL_INIT, RECOVERY_LADDER  # noqa: E402
from autoinit_phase_a_launch import (  # noqa: E402
    LOCAL_ASSETS as PHASE_A_LOCAL_ASSETS, TEACHER_REVISION, TEST_IGNORES,
    budget as phase_a_budget, probe_streams,
)
from autoinit_recovery_continuation_launch import (  # noqa: E402
    STAGED_INTO, TRANSPORT_MANIFEST, TRANSPORT_REPO, transport_is_verified,
)

STATUS = f"{WS}/autoinit_continuation_b.status"
RUN_LOG = f"{WS}/autoinit_continuation_b_run.log"
AUTH_PATH = "logs/autoinit_continuation_b_authorization.json"
FROZEN_SCIENCE_PLAN = "logs/autoinit_phase_a_recovery_plan_frozen.json"
PREREGISTRATION = REPO_ROOT / "logs/autoinit_continuation_b_preregistration.json"
PRICING = REPO_ROOT / "logs/autoinit_behavioural_continuation_pricing.json"
AMENDMENT = REPO_ROOT / "logs/autoinit_phase_b_identity_collapse_amendment.json"

#: Same exclusions as Phase B, plus this session's own whole-function test, which
#: runs the continuation end to end on CPU and belongs on the dev box, plus the
#: two modules that exercise the Phase-B SEARCH this session cannot run.
#:
#: **Bind is not consume.** Attempt 1 died here for `$0.2513`. Both calibration
#: mixtures are bound into this session's authorization and preregistration as
#: provenance for the imported Stage-1 result — and neither is a runtime input.
#: The paid behavioural probes train from `artifacts/stage3/ladder_uniform_probe`
#: and are scored on the `artifacts/stage3/recovery_search_v2` battery; the
#: continuation driver contains no calibration reference at all, and the one
#: mention in `PhaseADriver` is a comment inside the `stage1` search this
#: session overrides with a raise and never binds into its stage map.
#:
#: `calib.domain_balanced@v1` and `calib.reasoning_heavy@v2` are therefore
#: Phase-B search inputs, and the right repair is NOT to stage them here. Making
#: a session depend on bytes it never reads, so that an unrelated inherited test
#: can find them, is the inheritance defect one layer further out.
#:
#: What the two modules below actually assert:
#:
#: * `test_causal_depth_measurement_job` drives the causal-depth/calibration
#:   measurement job through the real resolver, which loads
#:   `artifacts/stage1/e8_calibration_v1/items.jsonl`. Diagnostic search
#:   machinery; nothing in the continuation's stage map reaches it.
#: * `test_phase_b_driver_and_launcher` builds the **Phase-B** spec and requires
#:   both calibration profiles and the `reasoning_heavy_v2` staging that Phase B
#:   declares and this session does not. It is a correct test of a session this
#:   one is structurally unable to be.
#:
#: Both keep running in full on the dev box, where the bytes live. The check
#: moves; it does not disappear — exactly as `test_phase_b_reuse_hostlocal` is
#: excluded here and checked before a pod exists.
CONTINUATION_TEST_IGNORES = (
    *TEST_IGNORES,
    "tests/autoinit/test_phase_b_reuse_hostlocal.py",
    "tests/pod/test_phase_b_stage1_executes.py",
    "tests/pod/test_continuation_b_executes.py",
    "tests/autoinit/test_causal_depth_measurement_job.py",
    "tests/pod/test_phase_b_driver_and_launcher.py",
)

#: The advancing candidates whose bytes must be on the pod. `fe9683e6a9c7` is the
#: one Attempt 5 produced and the only one not already staged from a prior
#: session; it is uploaded to the relay at `$0` before any pod exists.
CONTINUATION_LEAF_PREFIX = "phase_b_attempt5/selected_leaves"
ADVANCING_NEW_LEAF = "fe9683e6a9c783bbc6fe276a78c851c6"
ADVANCING_RETAINED = "85bde4ded2c31953f802e39cf2252c87"

CONTINUATION_ASSET_MANIFEST = REPO_ROOT / "logs/autoinit_continuation_b_assets.json"

#: New scratch goes under the project's scratch root, one directory per session
#: — NOT another `phase_b_*_scr` beside it in `$HOME`. Five of those already
#: exist from earlier attempts; a sixth would be the sixth place a reader has to
#: look for a session's launch log. `/home/ecs-user/aad-artifacts` is untouched:
#: it is the canonical scientific artifact store, and this is scratch.
DEFAULT_SCRATCH_ROOT = Path("/home/ecs-user/aad-scratch/sessions")

# --- this session's artifact-store capacity contract ------------------------
#
# **What this session actually produces.** Not checkpoints. Its three finalists
# already exist, on this box and on the relay, and it runs no search: its only
# irreplaceable product is the artifact archive carrying the newly bought probe
# journal, generations, per-sample rows, the pooled decision and the report.
#
# **Measured, not estimated.** `SessionRunner.collect_and_teardown` writes the
# downloaded archive AND its extracted copy under `<args.scr>/store`, so the
# bound is the whole store directory rather than the tarball. `du -sb` over
# every retained session scratch on this box, restricted to SEARCH-FREE
# sessions — the only comparable class, since this one cannot run a search:
#
#     recovery continuation attempt 7   12,094,788 B   11 probes, ALL_DONE
#     E7                                13,641,956 B   largest search-free store
#
# Every store above that on this box (33.8 MB, 70.3 MB, 74.1 MB) belongs to a
# session that ran the Phase-A or P=2 search and is dominated by a `states.jsonl`
# search journal of 26-55 MB, which this session is mechanically unable to write.
# Derivation and the full measurement table:
# `logs/autoinit_continuation_b_capacity.json`.
CONTINUATION_STORE_MEASURED_BYTES = 13_641_956
#: Applied to the largest measured comparable rather than to the closest one. A
#: continuation that runs its conditional `sc` writes two probes' worth of
#: generations that no measured comparable contained, and a factor is the honest
#: way to carry that rather than predicting the number.
CONTINUATION_STORE_SAFETY_FACTOR = 4
#: Working room on the same filesystem for the manifest, the relay journals, the
#: launch log and the watchdog stream. Deliberately far above the measured need:
#: the cost of over-reserving a GiB is nothing, and the cost of under-reserving
#: is a successful session whose evidence cannot land.
CONTINUATION_SCRATCH_MARGIN_BYTES = 1 * 2**30


def continuation_artifact_bytes_required() -> int:
    return (CONTINUATION_STORE_MEASURED_BYTES * CONTINUATION_STORE_SAFETY_FACTOR
            + CONTINUATION_SCRATCH_MARGIN_BYTES)


def continuation_artifact_capacity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Can the SCRATCH filesystem hold what this session brings home?

    **Deliberately not `--ckpt-store`.** Phase A's `ckpt_store_capacity_gate`
    asks whether `/home/ecs-user/aad-artifacts/autoinit` can hold five bf16
    stage-1 leaves, because a Phase-A or Phase-B session that completes a search
    and cannot land its leaves has lost the search — the attempt-11 failure. This
    session produces no leaves. Inheriting that gate made it demand 11.55 GiB, a
    thousand times its need, on the wrong filesystem for the wrong product, and
    it refused every launch on that basis.

    The filesystem checked here is the one the collector actually writes:
    `<args.scr>/store`, where both the archive and its extracted copy land.
    """
    dest = Path(ctx.args.scr) / "store"
    probe = dest
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    need = continuation_artifact_bytes_required()
    ctx.evidence.setdefault("precheck", {})["continuation_artifact_store"] = {
        "destination": str(dest), "filesystem_probed": str(probe),
        "free_bytes": free, "required_bytes": need,
        "measured_comparable_bytes": CONTINUATION_STORE_MEASURED_BYTES,
        "safety_factor": CONTINUATION_STORE_SAFETY_FACTOR,
        "working_margin_bytes": CONTINUATION_SCRATCH_MARGIN_BYTES,
        "produces_checkpoint_products": False}
    if free < need:
        return False, (
            f"{probe} has {free / 2**20:.1f} MiB free; this session's artifact "
            f"archive and its extracted copy need {need / 2**20:.1f} MiB "
            f"({CONTINUATION_STORE_MEASURED_BYTES / 2**20:.1f} MiB measured from "
            f"the largest comparable search-free session, x"
            f"{CONTINUATION_STORE_SAFETY_FACTOR}, plus "
            f"{CONTINUATION_SCRATCH_MARGIN_BYTES / 2**30:.0f} GiB of working "
            "room). Free space before launching: the probe journal is the only "
            "thing here that cannot be re-derived for free.")
    return True, (f"{probe}: {free / 2**30:.2f} GiB free for "
                  f"{need / 2**20:.0f} MiB of archive and working room")


def continuation_inputs() -> tuple:
    """Relay inputs for exactly the advancing candidates, and nothing else."""
    out: list[RelayInput] = []
    if CONTINUATION_ASSET_MANIFEST.is_file():
        manifest = json.loads(CONTINUATION_ASSET_MANIFEST.read_text())
        for f in manifest.get("files", []):
            out.append(RelayInput(path=f["remote_path"], repo=manifest["repo"],
                                  dest=f"{STAGED_INTO}/{manifest['state_id']}",
                                  sha256=f["sha256"]))
    if transport_is_verified():
        transport = json.loads(TRANSPORT_MANIFEST.read_text())
        for record in transport["leaves"]:
            if record["state_id"] != ADVANCING_RETAINED:
                continue
            for f in record["files"]:
                out.append(RelayInput(path=f["remote_path"], repo=TRANSPORT_REPO,
                                      dest=f"{STAGED_INTO}/{record['state_id']}",
                                      sha256=f["sha256"]))
    return tuple(out)


def continuation_budget(args):
    """Phase A's arithmetic with the search removed and the probes reduced.

    Derived rather than rewritten, so a change to the measured step time reaches
    this session too. What is removed is the stage-1 search phase and both of its
    soft-stop reserves: this session has no search, so a reserve bought to cover
    one would be funding work it cannot perform.
    """
    base = phase_a_budget(args)
    kept = tuple(p for p in base.other_phases
                 if "search" not in p.name and "beam" not in p.name)
    return replace(base, arms=args.rung2_probes + args.tie_break_probes,
                   other_phases=kept, soft_stop_reserves=())


def continuation_source_gate(ctx: SessionContext) -> tuple[bool, str]:
    observed = continuation_source_digest(REPO_ROOT)
    expected = getattr(ctx.auth, "source_digest", None)
    if expected is None:
        return False, "the authorization declares no source_digest"
    if observed["digest"] != expected:
        return False, (f"the continuation executable digests to "
                       f"{observed['digest']} but the grant named {expected}")
    return True, f"continuation executable {observed['digest'][:12]}… matches the grant"


def evidence_binding_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Every cited identity, re-derived here, before a pod exists.

    The driver checks this again at stage 0 on the pod. Doing it first means a
    session whose evidence moved costs nothing rather than a setup.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))
    try:
        from autoinit_continuation_b_driver import ContinuationDriver
    except Exception as exc:                       # noqa: BLE001
        return False, f"the continuation driver does not import: {exc}"
    try:
        observed = ContinuationDriver.observed_evidence()
    except Exception as exc:                       # noqa: BLE001
        return False, f"could not re-derive the cited evidence: {exc}"
    try:
        ctx.auth.require_evidence(observed)
    except Exception as exc:                       # noqa: BLE001
        return False, str(exc)
    return True, (f"cited evidence matches the grant: universe "
                  f"{observed['collapsed_universe_identity'][:12]}…")


def no_search_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The session must be structurally unable to buy Stage 1 again.

    Two separate claims, checked separately because conflating them is how a
    source digest starts lying:

    1. The continuation's OWN path contains no search call site at all.
    2. Across everything the session loads, the files that DO contain one are
       exactly the known-neutralized set — today, `PhaseADriver`, whose
       `stage1` the continuation overrides with a raise and never binds into
       its stage map. A call site appearing anywhere else fails here.
    """
    from aadistill.autoinit.phase_b_continuation import (
        CONTINUATION_OWN_PATH_FILES, FORBIDDEN_CALLS,
        KNOWN_NEUTRALIZED_SEARCH_CALL_SITES, search_call_site_owners,
    )

    own = search_call_site_owners(REPO_ROOT, files=CONTINUATION_OWN_PATH_FILES)
    if own:
        return False, (f"the continuation's own path {list(own)} contains a "
                       "search call site; this session must not be able to "
                       "purchase Stage 1 again")
    owners = search_call_site_owners(REPO_ROOT)
    unexpected = sorted(set(owners) - set(KNOWN_NEUTRALIZED_SEARCH_CALL_SITES))
    if unexpected:
        return False, (f"{unexpected} contain a search call site and are not in "
                       "the known-neutralized set; a loaded file gained a way to "
                       "reach a search")
    if getattr(ctx.auth, "runs_search", False):
        return False, "the authorization claims runs_search"
    return True, (f"no search call site on the continuation's own path; "
                  f"{len(owners)} known-neutralized elsewhere; runs_search False")


def continuation_price_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The ceiling must be the continuation's, not the full session's."""
    if not PRICING.is_file():
        return False, f"{PRICING.name} is missing"
    priced = json.loads(PRICING.read_text())["total"]["hard_usd"]
    granted = ctx.auth.hard_cap_usd
    full_phase_b = 35.6660
    if granted >= full_phase_b:
        return False, (f"the grant carries ${granted:.4f}, at or above the full "
                       f"Phase-B ceiling ${full_phase_b:.4f}. Stage 1 is bought; "
                       "a continuation authorized at the full figure could fund a "
                       "session this one is not")
    if granted < priced:
        return False, (f"the grant carries ${granted:.4f} against a priced "
                       f"continuation ceiling of ${priced:.4f}")
    return True, (f"continuation ceiling ${granted:.4f} covers the priced "
                  f"${priced:.4f} and is below the full-session ${full_phase_b:.4f}")


def driver_command(ctx: SessionContext, plan) -> str:
    return (f"/opt/train/bin/python "
            f"{REPO}/scripts/pod/autoinit_continuation_b_driver.py "
            f"--stage all --image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f} "
            f"--probe-train-minutes {ctx.args.probe_train_minutes} "
            f"--probe-battery-minutes {ctx.args.probe_battery_minutes}")


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-continuation-b",
        schema="aadistill.autoinit.continuation_b_session/v1",
        description=("Phase-B behavioural continuation: import the completed "
                     "Stage-1 and rung-1 state, buy only the missing sb and "
                     "conditional sc, then the frozen final selection"),
        authorization_path=AUTH_PATH,
        #: Refused by schema if a full Phase-B artifact is passed: that grant
        #: authorizes a search this session must not run.
        authorization_loader=ContinuationAuthorization.load,
        plan_id=CONTINUATION_PLAN_V1.plan_id,
        plan_hash=CONTINUATION_PLAN_V1.plan_hash,
        budget=continuation_budget(args),
        setup=SetupManifest(
            relay_inputs=(*CANONICAL_INIT, *RECOVERY_LADDER, *continuation_inputs()),
            local_assets=PHASE_A_LOCAL_ASSETS,
            env={"SESSION_KIND": "continuation_b"},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "VLLM_READY",
                           "TEACHER_READY", "ROPE_OK", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION,
            test_ignores=CONTINUATION_TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_continuation_b",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("PHASE_A_FAILED", "PHASE_A_INCOMPLETE"),
            incomplete=("PHASE_A_INCOMPLETE",),
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. A stage-0 comparability failure is a "
                          "TERMINATE: every cited observation would be lost at "
                          "once and re-buying them is a larger session.")),
        # `fetch_products` and `products_secured` are left at their DEFAULTS —
        # "fetch nothing" and "this session owes no off-pod products". That is
        # the honest contract for this session, not a skipped gate.
        #
        # They were Phase B's `fetch_selected_leaves` / `selected_leaves_secured`,
        # inherited along with the launcher. Both exist to rescue SEARCH OUTPUT:
        # attempt 11 produced five measured stage-1 leaves and lost every one
        # because the only product fetch returned early. This session runs no
        # search. Its three finalists are INPUTS, already durable on this box and
        # on the relay, and at teardown the inherited pair would have gone
        # looking on the pod for five leaf directories nothing ever created
        # there — then judged a scientifically complete session unsecured.
        #
        # What this session must not lose is its probe journal, generations,
        # per-sample rows, pooled decision and report. None of those are weights;
        # all of them travel in the artifact archive described by
        # `configs/autoinit/continuation_b_artifacts.json`, whose entries were
        # enumerated from a real end-to-end run. The temporary probe training
        # checkpoints are deliberately NOT promoted to permanent products: no
        # downstream evidence contract reads them, and the frozen final selection
        # is computed from the journal.
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_phase_a",
            evidence_filename="phase_a_evidence.json",
            archive_basename="continuation_b_artifacts.tar.gz",
            spec_success="configs/autoinit/continuation_b_artifacts.json",
            spec_failed="configs/autoinit/continuation_b_artifacts_failed.json",
            report_names=("phase_a_evidence.json",
                          "attested_evaluation_protocol.json",
                          "rung2_selection.json", "phase_a_result.json"),
            event_streams=probe_streams),
        teardown=TeardownPolicy(
            note="nothing chains off the continuation; it is a terminus"),
        precheck=(
            session_commit_gate(REPO_ROOT, AUTH_PATH, check_lineage=True),
            frozen_science_plan_gate(REPO_ROOT, FROZEN_SCIENCE_PLAN),
            continuation_source_gate,
            no_search_gate,
            evidence_binding_gate,
            continuation_price_gate,
            continuation_artifact_capacity_gate,
        ),
        evidence_fields={
            "phase": "B-continuation",
            "continuation_plan_hash": CONTINUATION_PLAN_V1.plan_hash,
            "runs_search": False,
            "stage1_imported_not_recomputed": True,
            "calibration_profiles": {
                p.qualified_id: {"profile_hash": p.profile_hash,
                                 "content_sha256": p.content_sha256}
                for p in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)},
            "retrains_permanent_controls": False,
            "redefines_thresholds": False,
            "followon_started": False,
            "followon_reachable_from_this_launcher": False},
    )


def build_parser() -> argparse.ArgumentParser:
    from autoinit_phase_a_launch import build_parser as phase_a_parser

    ap = phase_a_parser()
    ap.set_defaults(out="logs/autoinit_continuation_b_session.json",
                    disk_gb=120,
                    rung1_probes=0, rung2_probes=1, tie_break_probes=2,
                    poll_limit_min=None,
                    scr=str(DEFAULT_SCRATCH_ROOT / "autoinit-continuation-b"))
    # `set_defaults` does not clear `required`, and Phase A declares `--scr`
    # required precisely because it has no sensible default. This session does:
    # one directory per session under the project scratch root. Passing `--scr`
    # explicitly still wins, and the capacity gate checks whatever is resolved.
    for action in ap._actions:                     # no public argparse accessor
        if action.dest == "scr":
            action.required = False
    return ap


def continuation_poll_limit_minutes(args) -> float:
    """How long to keep polling. Derived from THIS session's priced plan.

    The shared helper is Phase B's, and it takes the plan as an argument for
    exactly this reason. Calling it without one would price the polling lifetime
    off `phase_b_budget`, whose envelope contains the 16.5 h P=2 search this
    session does not run: the launcher would poll for roughly 32 h over a session
    bounded at ~5 h.

    That is not a harmless over-estimate. The launcher's polling lifetime is what
    bounds how long a HUNG pod bills before anything gives up, so inheriting the
    wrong session's number silently converts an $8.07 ceiling into a much larger
    exposure. The relational contract itself — hard-terminate plus one poll
    interval plus the fetch bound plus measured teardown — is reused unchanged.
    """
    from autoinit_phase_b_launch import phase_b_poll_limit_minutes

    plan = continuation_budget(args).plan(price_per_hour=args.max_price,
                                          authorized_usd=float("inf"))
    return phase_b_poll_limit_minutes(args, plan=plan)


def main() -> int:                                             # pragma: no cover
    args = build_parser().parse_args()
    if args.poll_limit_min is None:
        args.poll_limit_min = continuation_poll_limit_minutes(args)
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("STOP for review. The behavioural continuation "
                                "finished Phase B's rungs from completed "
                                "evidence; nothing chains off it."))


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
