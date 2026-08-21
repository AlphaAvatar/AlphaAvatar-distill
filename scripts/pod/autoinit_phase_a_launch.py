#!/usr/bin/env python3
"""AutoInitializer Phase A, as a session specification.

    PYTHONPATH=src setsid nohup python -u \
        scripts/pod/autoinit_phase_a_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

Declares WHAT the session is. How it is run lives once, in
`aadistill.infrastructure.session_runner`. This was a **subclass** of the
micro-preflight launcher until 2026-08-18, retargeted by mutating that module's
globals; attempt 1 died at $0.1075 because `SESSION_KIND` leaked between two
sessions sharing one setup script, which is a failure only a module global can
have.

**This is the long one.** Nine recovery probes at ~62 min each, plus a beam
search, is 12-17 GPU-hours against a project whose longest successful session so
far was 3.6 hours and four of whose last eight attempts hit an infrastructure
event. Two consequences are designed in rather than hoped for:

*Resume is per probe.* The driver journals each probe the moment it is scored and
restores it only when the student digest, seed and evaluation protocol still
match. A redrawn pod re-runs the search cheaply from its own journal and repeats
no completed probe.

*Retention is decided by measurement, not by preference.* The relay reports
`usedStorage` **91.54 GiB** against an inferred 100 GB (93.13 GiB) limit — 1.60
GiB of headroom — and five bf16 leaves are 5.61 GiB, so relay staging of the
searched leaves is **off**. Only the finalists are pulled to the dev box.
Rejected leaves are neither fetched nor deleted: `leaf_retention.json` carries
their digest, lineage, sa evidence and rejection reason. Full accounting in
`logs/autoinit_phase_a_storage.md`.

Phase A is a terminus. It starts one driver, that driver has no stage 6, and the
authorization has no code path to a follow-on experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
# The sibling science-input declarations. Present when this file is run
# directly; absent when a test loads it by path, which is how the
# structural checks load every launcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.autoinit.phase_a import (  # noqa: E402
    PHASE_A_PLAN_V1, PHASE_A_SCOPE, PhaseAAuthorization,
)
from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, LocalAsset, MarkerPolicy, RelayInput,
    SessionContext, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    frozen_science_plan_gate, session_commit_gate,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
from autoinit_science_inputs import (  # noqa: E402
    CALIBRATION_V1, CANONICAL_INIT, RECOVERY_LADDER,
)

STATUS = f"{WS}/autoinit_phase_a.status"
RUN_LOG = f"{WS}/autoinit_phase_a_run.log"
AUTH_PATH = "logs/autoinit_phase_a_authorization.json"
FROZEN_SCIENCE_PLAN = "logs/autoinit_phase_a_recovery_plan_frozen.json"
#: Dev-box-only inputs the pod cannot fetch from git: the two frozen assets.
LOCAL_ASSETS = (
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage3/recovery_search_v2", "recovery_search_v2",
               "artifacts/stage3"),
)
TEST_IGNORES = ("tests/data/test_recovery_corpus_pipeline.py",
                "tests/pod/test_phase_a_stages1_5_execute.py")
TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: Measured, not estimated. 61.55 min end-to-end for an 0.86M probe (attempt 4,
#: two arms) and 9.82 min conservative for one battery (half of the 19.65 min
#: Stage 3 took for two, so it carries engine load and scoring).
PROBE_TRAIN_MINUTES = 61.55
#: The CONSERVATIVE battery figure. The marginal 8.43 min is what one more
#: battery costs once the engine is warm, and pricing from the marginal figure
#: would under-book the first one.
PROBE_BATTERY_MINUTES = 9.82
#: The search: 39-56 states at ~2.6 min evaluation + ~8 s statistics each, plus
#: the operator build. This is the CACHED-path expected figure and it is left
#: alone, because the expected figure is what an authorization request quotes.
#: The two risks it does not cover are carried as named soft-stop reserves below
#: rather than folded in here, where the 10% contingency would multiply them.
SEARCH_MINUTES = 180.0
#: Reserve 1 — the reference-cache fallback.
#:
#: `depth.causal_kl_greedy_v1` caches the unbypassed parent's logits for the
#: whole calibration mixture; at 59,763 positions x 151,936 vocabulary that is
#: 16.91 GiB in bf16. When the cgroup cannot take it the operator recomputes the
#: reference per candidate — numerically identical, ~2.10x the forward work.
#: Worst case the frozen schedule reaches 16 invocations (1+3+6+6) and the extra
#: is 8,866.1 s. Derived in `logs/autoinit_phase_a_fallback_audit.json`.
#:
#: It is a SOFT-STOP reserve, not a hard-only one. The fallback is consumed
#: entirely inside stage 1, so a reserve placed after the soft stop would leave
#: `PhaseADriver.afford()` refusing later probes — including a legitimately
#: triggered conditional seed-sc rung. That would truncate the frozen design to
#: pay for an infrastructure risk.
#:
#: **MEASURED 2026-08-20, attempt 11: the fallback fired 4/4.** Inside the search
#: only ~20.3 GiB is free — the beam holds the parent teacher and candidate state
#: — against the 36.42 GiB a standalone measurement saw, so the 16.9 GiB cache
#: never fits. Four causal-depth invocations ran at 6.96–10.79 eval/min against
#: the standalone 12.07, taking 122.1 min, 68% of the search.
#:
#: This is recorded and **not** repriced. The complete Stage-1 wall time was
#: 180.283 min, which essentially reproduces the 180.0 base allowance above, so
#: the dollar semantics of these reserves are unchanged and they remain
#: contingency. If a future cost derivation shows the fallback is the expected
#: path rather than the risk — which this single observation suggests but does
#: not establish — that derivation, not this note, is what may move the price.
FALLBACK_RESERVE_MINUTES = 8866.1 / 60.0                      # 147.7683
#: Reserve 2 — the beam-6 search pricing correction. `SEARCH_MINUTES` was taken
#: from the cost model's beam-4 row (182.07 min); the frozen schedule is beam 6,
#: whose row is 12,972.95 s = 216.22 min. A known pricing defect, carried
#: explicitly rather than left to the gap between expected and hard.
BEAM6_SEARCH_CORRECTION_MINUTES = 12972.947682669545 / 60.0 - SEARCH_MINUTES
#: Setup allowance, measured across the 2026-08-14/15 runs at ~10.4 min and
#: priced at 11 so the number is not the optimistic one.
SETUP_MINUTES = 11.0
#: The Stage-1 phase name. ONE string with two readers: `budget()` prices stage 1
#: under it, and `stage1_deadline_minutes()` reads it back out to bound stage 1
#: at runtime. Neither restates the other's number.
STAGE1_SEARCH_PHASE = "stage1_beam_search"


def stage1_deadline_minutes(plan) -> float:
    """The Stage-1 runtime deadline, READ OFF the priced envelope.

    `SEARCH_MINUTES` is the base search allowance and stays exactly that. What
    stage 1 is *paid for*, however, is that base plus both soft-stop reserves,
    and both are consumed inside stage 1 by construction: the reference-cache
    fallback happens during the search, and the beam-6 correction exists only
    because the base was taken from the cost model's beam-4 row while the frozen
    schedule is beam 6.

    Until 2026-08-20 the runtime `Deadline` was built from the base alone. A
    search that legitimately entered the fallback path — the exact risk the
    147.7683-minute reserve was bought for — would have been killed at 180
    minutes with that reserve unspent, and the kill would have looked like a
    failed search rather than a deadline that disagreed with its own price by
    183.98 minutes.

    **Derived, not restated.** This reads the same `BudgetPlan` the dollar
    figures come from, so a reserve added to the pricing extends this bound
    automatically and cannot be added to one without the other. Every soft-stop
    reserve is a Stage-1 risk by construction — `soft_stop_reserves` is folded in
    *before* the soft stop precisely so stage 1 can consume it, per the field's
    own contract in `budget.py` — and a test pins the current set, so a future
    reserve that is NOT a Stage-1 risk has to be classified rather than silently
    inflating this deadline.
    """
    base = [p.minutes for p in plan.breakdown if p.name == STAGE1_SEARCH_PHASE]
    if len(base) != 1:
        raise ValueError(
            f"expected exactly one {STAGE1_SEARCH_PHASE!r} phase in the priced "
            f"envelope, found {len(base)}. The Stage-1 deadline is derived from "
            "the price and cannot be derived from a missing or ambiguous phase; "
            "fix the pricing rather than hard-coding a deadline here.")
    return base[0] + sum(r.minutes for r in plan.soft_stop_reserves)


#: What five bf16 leaves at the frozen student size actually weigh, measured
#: from attempt 11's own search record: 5 x 1.110 GiB. Not an estimate.
SELECTED_LEAF_BYTES = 5 * 1_192_099_840
#: Room to land them AND for the box to keep working. The dev box ran out of
#: disk mid-suite on 2026-08-20 and produced fourteen errors that were not code
#: failures; the suite alone needs roughly 5 GB of scratch to complete.
CKPT_STORE_MARGIN_BYTES = 6 * 2**30


def ckpt_store_capacity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Can the DEV BOX hold the five selected leaves? Asked before a pod exists.

    The driver's own headroom check runs on the pod and proves only that the pod
    can stage them. That is not durability, and a session that discovers at
    teardown that the leaves have nowhere to land has already paid for the
    search it is about to lose.
    """
    dest = Path(ctx.args.ckpt_store) / "phase_a"
    probe = dest
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    need = SELECTED_LEAF_BYTES + CKPT_STORE_MARGIN_BYTES
    ctx.evidence.setdefault("precheck", {})["ckpt_store"] = {
        "destination": str(dest), "free_bytes": free,
        "required_bytes": need, "leaf_bytes": SELECTED_LEAF_BYTES}
    if free < need:
        return False, (
            f"{probe} has {free / 2**30:.2f} GiB free; the five stage-1 selected "
            f"leaves are {SELECTED_LEAF_BYTES / 2**30:.2f} GiB and landing them "
            f"needs {need / 2**30:.2f} GiB with working room. Free space before "
            "launching: a search that succeeds and cannot come home is the "
            "attempt-11 loss repeated at full price.")
    return True, (f"{probe}: {free / 2**30:.1f} GiB free for "
                  f"{need / 2**30:.1f} GiB of leaves and working room")


def probe_streams(ctx: SessionContext) -> tuple[str, ...]:
    """The probe training streams a torn-down session may have left mid-write.

    Nine probes, so this is derived rather than listed: naming a fixed set would
    either miss a rung-2 probe or demand a rung-3 one that correctly never ran.
    """
    journalled = sorted((ctx.scr / "relay").glob("*.train_log.jsonl"))
    return tuple(f"artifacts/stage3/phase_a/{p.name.split('.')[0]}/train_log.jsonl"
                 for p in journalled)


def finalists_to_fetch(ctx: SessionContext) -> list[str]:
    """Which initializations earn permanent off-pod retention.

    **The finalists, not the winner.** If the run ends in
    `unresolved_equivalence` there is no winner and BOTH tied candidates are the
    result; fetching only a winner would throw away the finding.

    The control is excluded: it is the retained canonical checkpoint, it already
    exists at `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, and re-fetching
    it would spend transfer time on a byte we already hold.
    """
    if not ctx.args.fetch_finalists:
        return []
    store = ctx.scr / "store"
    retention = store / "leaf_retention.json"
    if retention.is_file():
        entries = json.loads(retention.read_text())["entries"]
        return [e["canonical_id"] for e in entries
                if e.get("permanent_checkpoint_retained")
                and not e.get("is_control")]
    result = store / "phase_a_result.json"
    if result.is_file():
        winner = json.loads(result.read_text()).get("winner")
        if winner:
            ctx.say("  no retention record; falling back to the winner alone")
            return [winner]
    return []


#: Where the Stage-1 selected leaves land off-pod. The existing dev-box
#: checkpoint store, not a new one: `--ckpt-store` already defaults to
#: `/home/ecs-user/aad-artifacts/autoinit` and `fetch_finalists` already writes
#: `<store>/phase_a/<state_id>` through the product transfer path.
SELECTED_LEAF_REPORT = "selected_leaf_durability.json"


def selected_leaf_records(ctx: SessionContext) -> list[dict]:
    """The five leaves Stage 1 staged, or [] if Stage 1 did not get that far.

    Read from the durability report, which is fetched with the other reports
    BEFORE products. Its existence IS the "stage 1 completed" signal: the driver
    writes it only after persisting and re-verifying every selected leaf.
    """
    report = ctx.scr / "store" / SELECTED_LEAF_REPORT
    if not report.is_file():
        return []
    return list(json.loads(report.read_text()).get("leaves", []))


def fetch_selected_leaves(ctx: SessionContext) -> list:
    """Bring the Stage-1 selected leaves off the pod, whatever Stage 2 did.

    **This is the attempt-11 fix.** That session produced five valid, measured,
    selected leaves in 180.3 minutes and lost every one, because the only
    product fetch returns early when stage 2 did not pass — and stage 2 failed
    six seconds after stage 1 succeeded. Staging them on the pod, which the
    driver now does, is not durability; it is a copy that dies with the pod.

    Deliberately NOT routed through the artifact tarball. The collector keeps
    both the downloaded archive and its extracted copy while verifying, so five
    incompressible 1.11 GiB safetensors would roughly double the temporary local
    footprint on a box that is already short of disk. This uses the same
    `scp -r` product path `fetch_finalists` uses.

    Each leaf is re-identified ON THE DEV BOX after transfer and required to
    equal the digest Stage 1 recorded. A transfer that silently truncated would
    otherwise be indistinguishable from one that worked.
    """
    records = selected_leaf_records(ctx)
    if not records:
        return []

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from aadistill.autoinit.arch import get_adapter
    from aadistill.autoinit.leaf_durability import verify_transferred_leaf

    adapter = get_adapter("qwen3")
    staged = f"{REPO}/artifacts/audit/autoinit_phase_a/selected_leaves"
    out: list = []
    for rec in records:
        state_id = rec["state_id"]
        dest = Path(ctx.args.ckpt_store) / "phase_a" / state_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            ["timeout", f"{ctx.args.ckpt_fetch_limit_min}m", "scp", "-r",
             *ctx.scp[1:], f"root@{ctx.host}:{staged}/{state_id}", str(dest)],
            capture_output=True, timeout=ctx.args.ckpt_fetch_limit_min * 60 + 120)
        entry = {"artifact": "stage1_selected_leaf", "state_id": state_id,
                 "route": "scp", "dest": str(dest), "rc": rc.returncode}
        if rc.returncode == 0:
            try:
                entry.update(verify_transferred_leaf(dest, rec, adapter=adapter))
                if not entry.get("matched"):
                    entry["rc"] = 1
            except Exception as exc:                       # noqa: BLE001
                entry["rc"] = 1
                entry["verify_error"] = f"{type(exc).__name__}: {exc}"
        else:
            entry["scp_tail"] = (rc.stderr or b"").decode(errors="replace")[-300:]
        out.append(entry)
        ctx.say(f"  stage-1 leaf {state_id[:12]} -> {dest}: rc={entry['rc']} "
                f"digest={'MATCHED' if entry.get('matched') else 'NOT MATCHED'}")
    return out


def selected_leaves_secured(ctx: SessionContext, fetched: list) -> tuple[bool, str]:
    """Teardown may not proceed on a Stage-1 success whose leaves are still only
    on the pod."""
    records = selected_leaf_records(ctx)
    if not records:
        return True, "stage 1 did not stage any selected leaves"
    want = {r["state_id"] for r in records}
    got = {f["state_id"] for f in fetched
           if f.get("artifact") == "stage1_selected_leaf" and f.get("rc") == 0
           and f.get("matched")}
    missing = sorted(want - got)
    if missing:
        return False, (
            f"stage 1 staged {len(want)} selected leaves and only {len(got)} are "
            f"verified off-pod; missing {missing}. Deleting the pod now would "
            "destroy a search that already succeeded, which is exactly what "
            "attempt 11 did.")
    return True, f"all {len(want)} stage-1 selected leaves verified off-pod"


def fetch_finalists(ctx: SessionContext) -> list:
    """Get the irreplaceable artifacts off the pod before it is deleted.

    Rejected leaves are not fetched, and not deleted either. They stay on the pod
    until it is destroyed, and what survives them is `leaf_retention.json`.
    """
    fetched: list = []
    if not ctx.stage2_passed:
        return fetched
    if ctx.args.stage_leaves_to_relay:
        rc = ctx.target.run(
            f"cd {REPO} && HF_TOKEN=\"$(cat {WS}/hf/token)\" "
            f"PYTHONPATH={REPO}/src /opt/train/bin/python "
            f"{REPO}/scripts/pod/collect_artifacts.py stage-leaves "
            f"--search-dir {REPO}/artifacts/autoinit/phase_a_search "
            f"--repo {ctx.args.relay_repo} --prefix phase_a_leaves",
            timeout=5400)
        fetched.append({"artifact": "searched_leaves", "route": "relay",
                        "rc": rc.returncode, "tail": (rc.stdout or "")[-400:]})
        ctx.say(f"  searched leaves -> relay: rc={rc.returncode}")

    for state_id in finalists_to_fetch(ctx):
        dest = Path(ctx.args.ckpt_store) / "phase_a" / state_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            ["timeout", f"{ctx.args.ckpt_fetch_limit_min}m", "scp", "-r",
             "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"root@{ctx.host}:{REPO}/artifacts/autoinit/phase_a_search/"
             f"states/{state_id}", str(dest)], capture_output=True, timeout=None)
        size = sum(f.stat().st_size for f in dest.rglob("*")
                   if f.is_file()) if dest.exists() else 0
        fetched.append({"artifact": "finalist_initialization",
                        "state_id": state_id, "rc": rc.returncode,
                        "bytes": size, "dest": str(dest)})
        ctx.say(f"  finalist {state_id}: rc={rc.returncode}, "
                f"{size / 2**30:.2f} GiB -> {dest}")
    if not fetched:
        ctx.say("  nothing to fetch: rung 1 produced no retention record "
                "and no winner")
    return fetched


def driver_command(ctx: SessionContext, plan) -> str:
    return (f"/opt/train/bin/python "
            f"{REPO}/scripts/pod/autoinit_phase_a_driver.py "
            f"--stage all --image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f} "
            # The base allowance funds the affordability check; the DERIVED
            # deadline bounds the search itself. They are different questions
            # and were the same number until 2026-08-20, which would have
            # killed a paid, valid fallback search at 180 min.
            f"--search-minutes {ctx.args.search_minutes} "
            f"--search-deadline-minutes {stage1_deadline_minutes(plan):.4f} "
            f"--probe-train-minutes {ctx.args.probe_train_minutes} "
            f"--probe-battery-minutes {ctx.args.probe_battery_minutes}")


def budget(args) -> BudgetSpec:
    """Priced separately so the price can be reproduced without a pod.

    The conditional tie-break rung IS given headroom, even though it usually does
    not run. If it were left out, a legitimately-triggered seed-sc rung would be
    killed by the watchdog at the threshold — and an unused leash costs nothing,
    because the pod is torn down on completion rather than at the threshold.
    """
    priced = args.rung1_probes + args.rung2_probes + args.tie_break_probes
    return BudgetSpec(
        # The probes ARE the training: each is one arm of 1023 steps. Priced
        # through the step-time model so the budget moves with the measured step
        # time rather than with a transcribed total.
        arms=priced, steps_per_arm=1023,
        step_seconds=args.probe_train_minutes * 60 / 1023,
        step_source=("measured end-to-end: 61.55 min per 1023-step probe, on "
                     "two arms"),
        below_floor_reason=(
            "the 4.15 s/step floor is E6b's IN-LOOP figure. Attempt 4 measured "
            "3.15 s/step in-loop for this exact model, rung and card, and 61.55 "
            "min end-to-end for the whole 1023-step probe including its periodic "
            "evaluation and checkpointing, on two arms. The rate used here is "
            "that end-to-end figure divided by the step count, so it is slower "
            "than the in-loop measurement rather than an optimistic version of "
            "it."),
        setup_minutes=args.setup_minutes,
        transfer_minutes=args.transfer_minutes,
        other_phases=(
            Phase("stage0_attestation_and_binding", 8.0),
            Phase(STAGE1_SEARCH_PHASE, args.search_minutes),
            Phase("stage5_selection_and_report", 5.0),
            Phase("artifact_manifest_and_verify", 8.0),
            Phase("artifact_synchronization", 10.0)),
        eval_minutes_per_arm=args.probe_battery_minutes,
        contingency_fraction=0.10,
        # Before the soft stop, so the conditional seed-sc rung survives a risk
        # that materializes in stage 1. See the constants' provenance.
        soft_stop_reserves=(
            Phase("stage1_reference_cache_fallback", args.fallback_reserve_minutes),
            Phase("beam6_search_pricing_correction", args.beam6_correction_minutes)),
        artifact_recovery_reserve_minutes=20.0)


def continuation_budget(args) -> BudgetSpec:
    """The Stage-2-to-5 budget, DERIVED from the full Phase-A one.

    A recovery continuation starting from a verified Stage-1 result no longer
    owes the search, so three things come out and nothing else moves:

    * the `stage1_beam_search` phase — 180 min of work already done and paid for
      by attempts 11 and 12, which produced byte-identical results;
    * the beam-6 pricing correction, which exists only because `SEARCH_MINUTES`
      was taken from the cost model's beam-4 row;
    * the reference-cache fallback reserve, which the audit places **entirely
      inside stage 1**.

    Everything else is retained exactly: the probe training and battery work, the
    same contingency fraction, the same artifact-recovery reserve, the same step
    time and its provenance. **No dollar figure is written here** — the numbers
    come from the same `BudgetSpec.plan()` that prices the full session, so the
    continuation cannot drift from the thing it is derived from.
    """
    full = budget(args)
    return replace(
        full,
        other_phases=tuple(p for p in full.other_phases
                           if p.name != STAGE1_SEARCH_PHASE),
        #: Both reserves are Stage-1 risks by construction, and stage 1 is not
        #: run. Carrying them would price a risk this session cannot take.
        soft_stop_reserves=())


def spec(args) -> SessionSpec:
    priced = args.rung1_probes + args.rung2_probes + args.tie_break_probes
    return SessionSpec(
        session_id="autoinit-phase-a",
        schema="aadistill.autoinit.phase_a_session/v1",
        description=("AutoInitializer Phase A: a beam search over initialization "
                     "operators, then the recovery probes that rank its leaves"),
        authorization_path=AUTH_PATH,
        #: The one artifact type that CAN grant Phase A. Naming it is what makes
        #: the permission a property of the declaration.
        authorization_loader=PhaseAAuthorization.load,
        plan_id=PHASE_A_PLAN_V1.plan_id,
        plan_hash=PHASE_A_PLAN_V1.plan_hash,
        budget=budget(args),
        setup=SetupManifest(
            #: Every relay object a Phase-A session reads — and, since
            #: 2026-08-18, every one it STAGES, with its destination and its
            #: digest. The calibration joined this list after attempt 5 died on
            #: it at $0.6426: stage 1 calls `DOMAIN_BALANCED_V1.resolve()`, and
            #: nothing staged or checked the file it reads. Naming it here bought
            #: the $0 precheck; it did not make the staging visible, because the
            #: shared shell staged it either way. Now this IS the staging.
            relay_inputs=(*CANONICAL_INIT, *RECOVERY_LADDER, *CALIBRATION_V1),
            local_assets=LOCAL_ASSETS,
            #: Setup must load the Phase-A authorization TYPE, not the narrow
            #: one. It arrives here, in this session's own manifest, rather than
            #: in a module global — which is the defect that cost attempt 1.
            env={"SESSION_KIND": "phase_a"},
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "TEACHER_REVISION"),
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_phase_a_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("PHASE_A_FAILED", "PHASE_A_INCOMPLETE"),
            incomplete=("PHASE_A_INCOMPLETE",),
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. Completed probes are journalled and the "
                          "permanent controls are inputs here and untouched.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_phase_a",
            evidence_filename="phase_a_evidence.json",
            archive_basename="phase_a_artifacts.tar.gz",
            spec_success="configs/autoinit/phase_a_artifacts.json",
            spec_failed="configs/autoinit/phase_a_artifacts_failed.json",
            #: `leaf_retention.json` is fetched BEFORE fetch_products runs,
            #: because `finalists_to_fetch` reads it to decide which
            #: initializations come home. The runner fetches reports first for
            #: exactly that reason.
            report_names=("phase_a_evidence.json",
                          "attested_evaluation_protocol.json",
                          "search_result.json", "rung1_selection.json",
                          "leaf_retention.json",
                          #: Fetched BEFORE products, because
                          #: `fetch_selected_leaves` reads it to learn which five
                          #: leaves stage 1 staged and what they must hash to.
                          #: Its presence is also the "stage 1 completed" signal.
                          SELECTED_LEAF_REPORT,
                          "rung2_selection.json", "phase_a_result.json"),
            event_streams=probe_streams,
            #: BOTH, and in this order. `fetch_selected_leaves` runs whenever
            #: stage 1 staged leaves — it is NOT gated on stage 2, which is the
            #: gate that lost attempt 11's five leaves six seconds after the
            #: search succeeded.
            fetch_products=lambda ctx: [*fetch_selected_leaves(ctx),
                                        *fetch_finalists(ctx)],
            products_secured=selected_leaves_secured),
        teardown=TeardownPolicy(
            note="Phase A is a terminus; nothing chains off it"),
        precheck=(
            session_commit_gate(REPO_ROOT, AUTH_PATH, check_lineage=True),
            frozen_science_plan_gate(REPO_ROOT, FROZEN_SCIENCE_PLAN),
            ckpt_store_capacity_gate,
        ),
        evidence_fields={
            "phase_a_session_plan_hash": PHASE_A_PLAN_V1.plan_hash,
            "scope": PHASE_A_SCOPE.as_dict(),
            "retrains_permanent_controls": False,
            "followon_started": False,
            "followon_reachable_from_this_launcher": False,
            "priced_probes": {
                "rung1": args.rung1_probes, "rung2": args.rung2_probes,
                "tie_break_conditional": args.tie_break_probes,
                "total_priced": priced,
                "note": ("the tie-break rung is priced so it CAN run, not "
                         "because it is expected to. It runs only for finalists "
                         "inside the preregistered equivalence interval."),
                "why_this_exceeds_the_repricing_doc": (
                    "logs/autoinit_phase_a_repricing.md priced search + probes "
                    "only. This plan additionally carries setup, Stage-0 "
                    "attestation, selection, artifact manifest/verify, "
                    "synchronization, transfer, a 10% contingency and a "
                    "20-minute artifact-recovery reserve — all of which are "
                    "session time that is really spent."),
            }})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=0.99)
    #: 106 GiB peak working storage from the priced search, plus the probe
    #: checkpoints. The preregistration says provision at least 150.
    ap.add_argument("--disk-gb", type=int, default=200)
    ap.add_argument("--rung1-probes", type=int, default=6)
    ap.add_argument("--rung2-probes", type=int, default=3)
    #: Conditional by construction; priced so it can run, not because it will.
    ap.add_argument("--tie-break-probes", type=int, default=3)
    ap.add_argument("--search-minutes", type=float, default=SEARCH_MINUTES)
    ap.add_argument("--probe-train-minutes", type=float, default=PROBE_TRAIN_MINUTES)
    ap.add_argument("--fallback-reserve-minutes", type=float,
                    default=FALLBACK_RESERVE_MINUTES,
                    help="soft-stop reserve for the reference-cache fallback")
    ap.add_argument("--beam6-correction-minutes", type=float,
                    default=BEAM6_SEARCH_CORRECTION_MINUTES,
                    help="soft-stop reserve for the beam-6 search pricing defect")
    ap.add_argument("--probe-battery-minutes", type=float,
                    default=PROBE_BATTERY_MINUTES)
    ap.add_argument("--setup-minutes", type=float, default=SETUP_MINUTES)
    ap.add_argument("--transfer-minutes", type=float, default=6.0)
    #: OFF, and measured rather than assumed. The relay reports usedStorage
    #: 91.54 GiB against an inferred 100 GB (93.13 GiB) limit — 1.60 GiB of
    #: headroom — and five bf16 leaves are 5.61 GiB. Staging them would fail on
    #: quota partway through, and deletion on this relay is a maintainer
    #: decision, not a launcher's. See logs/autoinit_phase_a_storage.md.
    ap.add_argument("--stage-leaves-to-relay", action="store_true", default=False)
    ap.add_argument("--no-stage-leaves-to-relay", dest="stage_leaves_to_relay",
                    action="store_false")
    ap.add_argument("--fetch-finalists", action="store_true", default=True)
    ap.add_argument("--no-fetch-finalists", dest="fetch_finalists",
                    action="store_false")
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--ckpt-store", default="/home/ecs-user/aad-artifacts/autoinit")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=30)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    #: The session is priced at 12-17 GPU-hours; the poll limit must outlast the
    #: hard threshold or the launcher would stop watching a pod that is still
    #: billing and still working.
    ap.add_argument("--poll-limit-min", type=float, default=1320.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/autoinit_phase_a_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("STOP for review; no follow-on experiment was "
                                "started."))


if __name__ == "__main__":
    raise SystemExit(main())
