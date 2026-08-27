#!/usr/bin/env python3
"""Launch the Phase-B session: joint P=2 search, then the cross-phase rungs.

    python3 scripts/pod/autoinit_phase_b_launch.py --dry-run

A `SessionSpec` declaration, like every launcher in this directory. What is
Phase-B-specific is the authorization type, the session plan, the calibration
inputs and the probe count — and the probe count is the whole point of the
cross-phase design: **Phase B prices ten probes, not twelve**, because three of
the eight candidates arrive with verified Phase-A evidence and are cited rather
than re-bought.

Two things this launcher will not do.

It will not run under a Phase-A grant. `authorization_loader` is
`PhaseBAuthorization.load`, which refuses a Phase-A artifact by schema — the
Phase-A authorization measures a different harness and carries a ceiling derived
for work Phase B does not do.

It will not proceed past a comparability failure. That is decided by the driver's
stage 0, which is blocking, so the session tears down with evidence rather than
falling back to a larger run. The 14-probe no-reuse path is a **rejected
counterfactual**, priced in `logs/autoinit_phase_b_pricing.json` only so the
rejection is on the record; nothing here can select it.
"""

from __future__ import annotations

import argparse
import json
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
from aadistill.autoinit.cost import L40S_MEASURED, price_search  # noqa: E402
from aadistill.autoinit.ranking import SCHEDULE_V1  # noqa: E402
from aadistill.autoinit.phase_b import (  # noqa: E402
    CANONICAL_CONTROL, PHASE_A_IMPORTED_FINALISTS, PHASE_B_PLAN_V1,
    PHASE_B_SEARCHED_LEAVES, PhaseBAuthorization, phase_b_source_digest,
)
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, LocalAsset, MarkerPolicy, RelayInput, SessionContext,
    SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    frozen_science_plan_gate, session_commit_gate,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
from autoinit_science_inputs import (  # noqa: E402
    CALIBRATION_V1, CANONICAL_INIT, RECOVERY_LADDER,
)
from autoinit_phase_a_launch import (  # noqa: E402
    BEAM6_SEARCH_CORRECTION_MINUTES, FALLBACK_RESERVE_MINUTES,
    LOCAL_ASSETS as PHASE_A_LOCAL_ASSETS, STAGE1_SEARCH_PHASE, TEACHER_REVISION,
    TEST_IGNORES, budget as phase_a_budget, ckpt_store_capacity_gate,
    fetch_selected_leaves, probe_streams, selected_leaves_secured,
    stage1_deadline_minutes,
)
#: The frozen Phase-B geometry and mixture sizes the cost model prices against.
#: Imported rather than restated so the deadline and the dollar figures cannot
#: describe different searches.
import price_phase_b as _pricing  # noqa: E402
#: The transport staging Phase A's continuation proved on hardware. Phase B
#: needs TWO of those five leaves on the pod — not to train them, but so the
#: search can measure them on the same state-evaluation suite as everything else.
from autoinit_recovery_continuation_launch import (  # noqa: E402
    STAGED_INTO, TRANSPORT_MANIFEST, TRANSPORT_REPO, transport_is_verified,
)
from verify_historical_probe_reuse import (  # noqa: E402
    ADMITTED as REUSE_ADMITTED, verify as verify_historical_reuse,
)

STATUS = f"{WS}/autoinit_phase_b.status"
RUN_LOG = f"{WS}/autoinit_phase_b_run.log"
AUTH_PATH = "logs/autoinit_phase_b_authorization.json"
FROZEN_SCIENCE_PLAN = "logs/autoinit_phase_a_recovery_plan_frozen.json"
PREREGISTRATION = REPO_ROOT / "logs/autoinit_phase_b_preregistration.json"
REUSE_RECORD = REPO_ROOT / "logs/autoinit_historical_probe_reuse.json"

#: Phase-B-specific, and deliberately built by extension rather than by editing
#: Phase A's tuple: `TEST_IGNORES` is the historical contract several completed
#: sessions ran under, and rewriting it would retroactively change what they
#: mean. The one addition is the strict-reconstruction module, whose tests read
#: the dev-box retained checkpoint store that a pod does not have and does not
#: need. That store is checked on the machine that owns it, before a pod exists,
#: by `historical_reuse_reconstruction_gate` below — the check moves, it does not
#: disappear.
#:
#: Attempt 1 aborted at `$0.15` because those tests ran on the pod. Ignoring the
#: whole pricing module would have been the wrong repair: everything else in it
#: reads committed records and is exactly the kind of thing a pod should re-check.
PHASE_B_TEST_IGNORES = (*TEST_IGNORES,
                        "tests/autoinit/test_phase_b_reuse_hostlocal.py")

#: The citations Phase B's ten-probe budget spends nothing on, derived from the
#: candidate rules rather than pasted: sa/sb/sc for each imported finalist, and
#: sa/sb for the control, which has no sc on record. If any one of them stops
#: reconstructing, the session would need a probe it has not booked.
REQUIRED_CITATIONS = frozenset(
    [f"{c}/{s}" for c in PHASE_A_IMPORTED_FINALISTS for s in ("sa", "sb", "sc")]
    + [f"{CANONICAL_CONTROL}/sa", f"{CANONICAL_CONTROL}/sb"])

#: The measured admission share of the intact reference on the hardware Phase B
#: actually ran on: 16.9 GiB of reference against 66% of 20.3 GiB free. Used ONLY
#: for the `high` projection. The `hard` bound assumes nothing is cached, because
#: the bounded cache guarantees a fraction and a ceiling may not rest on one.
DEPTH_CACHED_FRACTION = 13.4 / 16.9

#: The P=2 search envelope, DERIVED from the corrected cost model rather than
#: doubled from the P=1 figure.
#:
#: The old constant was 360.0 min, reached by doubling the P=1 base because the
#: model's P=2 range doubled with it. Attempt 3 falsified that: the search ran
#: 544.7 min — the full derived deadline — and had not finished, because the model
#: priced the intact reference as ONE pass while the run recomputed it per
#: candidate. The corrected model prices all three reference modes and bounds the
#: beam by its most expensive admissible parents instead of by an average.
PHASE_B_SEARCH_ESTIMATE = price_search(
    _pricing.TEACHER, _pricing.TARGET, _pricing.ADAPTER, _pricing.DECOMPOSED,
    calibration_tokens=_pricing.CALIBRATION_TOKENS,
    suite_tokens=_pricing.CALIBRATION_TOKENS,
    seq_len=_pricing.CALIBRATION_SEQ_LEN, n_profiles=2,
    beam_width=SCHEDULE_V1.width, warmup_levels=SCHEDULE_V1.warmup_levels,
    hardware=L40S_MEASURED, composite=_pricing.COMPOSITE,
    depth_cached_fraction=DEPTH_CACHED_FRACTION)

#: What Stage 1 must be allowed, end to end. This IS the deadline.
SEARCH_HARD_MINUTES_P2 = PHASE_B_SEARCH_ESTIMATE.seconds_hard / 60.0

#: And the base, so that `stage1_deadline_minutes` — which is `base + every
#: soft-stop reserve`, because every reserve is consumed inside stage 1 — lands
#: exactly on the hard bound.
#:
#: Both inherited reserves are SUBSUMED by the corrected model and must not be
#: added on top of it: `stage1_reference_cache_fallback` bought the difference
#: between a cached and a recomputed reference, which the `hard` mode now prices
#: directly, and `beam6_search_pricing_correction` corrected a beam-4 row to beam
#: 6, which `conservative_hard_seconds` now does structurally from `parents_max`.
#: Subtracting them here keeps them in the dollar envelope while making the
#: deadline exactly the derived bound rather than the bound plus its own
#: corrections counted twice.
SUBSUMED_RESERVE_MINUTES = FALLBACK_RESERVE_MINUTES + BEAM6_SEARCH_CORRECTION_MINUTES
SEARCH_MINUTES_P2 = SEARCH_HARD_MINUTES_P2 - SUBSUMED_RESERVE_MINUTES
if SEARCH_MINUTES_P2 <= 0:                       # pragma: no cover - defensive
    raise SystemExit(
        f"the derived P=2 search bound ({SEARCH_HARD_MINUTES_P2:.1f} min) is "
        f"smaller than the reserves it subsumes ({SUBSUMED_RESERVE_MINUTES:.1f} "
        "min); the envelope shape changed and Phase B must be re-derived")

#: Ten, not twelve. Five Phase-B leaves at sa; up to two survivors at sb; up to
#: three at the conditional sc rung. The three imported candidates are cited from
#: verified Phase-A evidence and cost nothing — that reduction IS the cross-phase
#: design, and pricing twelve would quietly fund re-buying them.
RUNG1_PROBES_P2 = PHASE_B_SEARCHED_LEAVES          # the 5 new leaves only
RUNG2_PROBES_P2 = 2                                # survivors; control is cited
TIE_BREAK_PROBES_P2 = 3

#: The materialized reasoning-heavy mixture travels the DEV-BOX path, not the
#: relay: it was built here at `$0` and has never been uploaded, so declaring it
#: as a `RelayInput` would name an object the pod cannot fetch. That is exactly
#: what killed Phase-A attempt 5 at $0.6426 — a calibration file the session
#: assumed was staged and was not.
#:
#: 780 KB over a measured 0.72 MB/s uplink is about a second, so the local path
#: costs nothing here. `calib.domain_balanced@v1` keeps travelling by relay as
#: CALIBRATION_V1, because it is already there.
CALIBRATION_V2_LOCAL = (
    LocalAsset("artifacts/stage1/reasoning_heavy_v2", "reasoning_heavy_v2",
               "artifacts/stage1"),
)


#: The two retained Phase-A finalists Phase B admits to the cross-phase
#: comparison. Their behaviour is CITED from verified evidence — no probe is
#: bought for them — but the bytes must be here for their step-0 measurement to
#: be comparable with the Phase-B leaves', which is why they are staged.
IMPORTED_FINALIST_PREFIXES = ("cca699c93f34", "85bde4ded2c3")


def imported_finalist_inputs() -> tuple:
    """Relay inputs for exactly the two admitted finalists, and no others.

    Filtered rather than staging all five: the other three are excluded from the
    candidate set, and staging 3.3 GiB of checkpoints a session may not compare
    would spend pod disk and transfer time on bytes it must not use.
    """
    if not transport_is_verified():
        return ()
    manifest = json.loads(TRANSPORT_MANIFEST.read_text())
    out = []
    for record in sorted(manifest["leaves"], key=lambda r: r["selected_order"]):
        if not record["state_id"].startswith(IMPORTED_FINALIST_PREFIXES):
            continue
        for f in record["files"]:
            out.append(RelayInput(
                path=f["remote_path"], repo=TRANSPORT_REPO,
                dest=f"{STAGED_INTO}/{record['state_id']}", sha256=f["sha256"]))
    return tuple(out)


def imported_finalists_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Refuse before a pod exists if the two finalists cannot be delivered.

    Without them the search cannot measure them, `candidate_universe` would be
    six rather than the preregistered eight, and the run would compare a
    different set from the one it froze.
    """
    inputs = imported_finalist_inputs()
    if not inputs:
        return False, ("the two retained Phase-A finalists are not deliverable: "
                       "the transport manifest is missing or unverified, so the "
                       "cross-phase candidate set could not be assembled")
    staged = {i.dest for i in inputs}
    if len(staged) != len(IMPORTED_FINALIST_PREFIXES):
        return False, (f"expected {len(IMPORTED_FINALIST_PREFIXES)} finalist "
                       f"destinations, found {sorted(staged)}")
    return True, (f"{len(inputs)} files stage the {len(staged)} admitted Phase-A "
                  "finalists for measurement")


#: The frozen storage contract: 244.87 GiB peak working set, provision >= 300.
PHASE_B_PEAK_WORKING_GIB = 244.87
PHASE_B_PROVISION_GIB = 300


def disk_provision_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A pod smaller than the frozen provision cannot finish the search.

    Mechanically known, not a thing to discover: the P=2 working set is 244.87
    GiB and the cost model says provision >= 300. A 200 GiB pod — the Phase-A
    default this launcher would otherwise inherit — runs out mid-beam, after the
    search has been paid for.
    """
    requested = int(getattr(ctx.args, "disk_gb", 0) or 0)
    if requested < PHASE_B_PROVISION_GIB:
        return False, (
            f"--disk-gb {requested} is below the frozen Phase-B provision of "
            f"{PHASE_B_PROVISION_GIB} GiB (peak working set "
            f"{PHASE_B_PEAK_WORKING_GIB} GiB). Refusing before the pod exists.")
    return True, (f"{requested} GiB container disk against a "
                  f"{PHASE_B_PEAK_WORKING_GIB} GiB peak working set")


def phase_b_budget(args):
    """Phase A's arithmetic, with the search widened and the probes reduced.

    Derived rather than rewritten, so a change to the step-time model or the
    contingency fraction reaches Phase B automatically instead of leaving two
    pricing implementations to drift.
    """
    base = phase_a_budget(args)
    widened = tuple(
        replace(phase, minutes=SEARCH_MINUTES_P2) if phase.name == STAGE1_SEARCH_PHASE
        else phase
        for phase in base.other_phases)
    if not any(p.name == STAGE1_SEARCH_PHASE for p in widened):
        raise SystemExit(
            f"expected a {STAGE1_SEARCH_PHASE!r} phase to widen; the Phase-A "
            "budget shape changed and Phase B must be re-derived, not guessed")
    return replace(
        base,
        arms=RUNG1_PROBES_P2 + RUNG2_PROBES_P2 + TIE_BREAK_PROBES_P2,
        other_phases=widened)


#: Measured, not chosen. Attempt 3 saw its terminal marker at 02:40:27 and had
#: the pod deleted and provider-confirmed at 02:43:15 — **2.8 min** — for a run
#: that FAILED and therefore had no products to transfer. Ten minutes carries
#: that with margin and covers the artifact manifest, the archive build and the
#: provider confirmation on a run that succeeded.
TEARDOWN_OBSERVED_MINUTES = 10.0


def phase_b_poll_limit_minutes(args, plan=None) -> float:
    """How long the launcher must keep polling. DERIVED from the priced plan.

    Phase B inherited Phase A's `--poll-limit-min 1320`, which is 22 h. The
    corrected Phase-B envelope terminates at ~1926 min (~32.1 h), so the launcher
    would have stopped polling **ten hours before** the session reached its own
    operational hard bound — and a launcher that stops polling leaves a pod alive
    and billing. That is the most expensive failure mode this project has.

    The contract is relational rather than another constant: the polling lifetime
    must exceed the plan's hard-terminate lifetime by enough to observe the end,
    collect, transfer and tear down. Every term is an existing bounded knob:

    * one poll interval of granularity, because the terminal marker can be seen
      up to `--poll-seconds` late;
    * `--ckpt-fetch-limit-min`, the bound already placed on fetching the products
      home, which is the one post-terminal step that moves real bytes;
    * the measured teardown above.

    `authorized_usd` is deliberately unbounded here: this reads MINUTES, which do
    not depend on it, and the authorization check belongs to the runner's
    `make_plan`, where it already happens against the real grant.
    """
    if plan is None:
        plan = phase_b_budget(args).plan(price_per_hour=args.max_price,
                                         authorized_usd=float("inf"))
    slack = (args.poll_seconds / 60.0
             + args.ckpt_fetch_limit_min
             + TEARDOWN_OBSERVED_MINUTES)
    return plan.hard_terminate_minutes + slack


def poll_lifetime_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Refuse, before a pod exists, to poll for less than the session can run.

    The derived default makes this pass; the gate is what makes it a contract
    rather than a default, because `--poll-limit-min` is still a flag a caller
    can set to anything.
    """
    limit = getattr(ctx.args, "poll_limit_min", None)
    if limit is None:
        return False, ("--poll-limit-min was never resolved; the Phase-B default "
                       "is derived from the priced plan and main() must set it")
    if ctx.plan is None:                          # pragma: no cover - runner sets it
        return False, "the budget plan is not available, so the poll lifetime is unbound"
    hard = ctx.plan.hard_terminate_minutes
    # Subsumed by the slack check below — anything failing this fails that too —
    # and kept only because "the launcher would stop polling while the pod is
    # still alive" is the sentence someone reading a refusal at 3 a.m. needs, and
    # "less than the 42 min needed to tear down" is not. A mutation weakening
    # `<=` to `<` changes the message and not the verdict; that is expected.
    if limit <= hard:
        return False, (f"--poll-limit-min {limit:.1f} does not exceed the plan's "
                       f"hard-terminate lifetime {hard:.1f} min; the launcher would "
                       "stop polling while the pod is still alive and billing")
    required = phase_b_poll_limit_minutes(ctx.args, plan=ctx.plan)
    if limit < required:
        return False, (f"--poll-limit-min {limit:.1f} exceeds the hard bound "
                       f"{hard:.1f} but leaves less than the {required - hard:.1f} min "
                       "needed to observe the end, fetch the products and tear down")
    return True, (f"polling {limit:.1f} min against a {hard:.1f} min hard bound "
                  f"({limit - hard:.1f} min of teardown slack)")


def driver_command(ctx: SessionContext, plan) -> str:
    return (f"/opt/train/bin/python "
            f"{REPO}/scripts/pod/autoinit_phase_b_driver.py "
            f"--stage all --image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f} "
            f"--search-minutes {SEARCH_MINUTES_P2} "
            f"--search-deadline-minutes {stage1_deadline_minutes(plan):.4f} "
            f"--probe-train-minutes {ctx.args.probe_train_minutes} "
            f"--probe-battery-minutes {ctx.args.probe_battery_minutes}")


def phase_b_source_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The executable about to run must be the one the grant was issued against.

    Checked before the pod exists. `require_source` performs the same comparison
    on the pod; doing it here means an unrehearsed executable costs nothing
    rather than a setup.
    """
    observed = phase_b_source_digest(REPO_ROOT)
    expected = getattr(ctx.auth, "source_digest", None)
    if expected is None:
        return False, ("the Phase-B authorization declares no source_digest, so "
                       f"it authorizes no executable (observed {observed['digest']})")
    if observed["digest"] != expected:
        return False, (f"the Phase-B executable digests to {observed['digest']} "
                       f"but the authorization was granted against {expected}")
    if observed["not_yet_covered"]:
        return False, ("the Phase-B executable-source set still declares "
                       f"uncovered files: {observed['not_yet_covered']}")
    return True, f"phase-B executable {observed['digest'][:12]}… matches the grant"


def preregistration_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The frozen preregistration must exist, verify, and bind this executable.

    A preregistration that no longer describes the code about to run is not a
    preregistration; catching that here is the difference between a documented
    experiment and one whose record was written against something else.
    """
    if not PREREGISTRATION.is_file():
        return False, f"{PREREGISTRATION.name} is missing"
    from aadistill.infrastructure.manifest import sha256_json

    prereg = json.loads(PREREGISTRATION.read_text())
    stated = prereg.get("preregistration_sha256")
    # Same rule the writer uses: `generated_utc` is provenance, not commitment,
    # and hashing it would make the id churn on every regeneration.
    material = {k: v for k, v in prereg.items()
                if k not in ("preregistration_sha256", "generated_utc")}
    if sha256_json(material) != stated:
        return False, f"{PREREGISTRATION.name} has been edited since it was frozen"
    if prereg["session_plan"]["plan_hash"] != PHASE_B_PLAN_V1.plan_hash:
        return False, ("the preregistration binds session plan "
                       f"{prereg['session_plan']['plan_hash'][:12]}… but the plan "
                       f"about to run is {PHASE_B_PLAN_V1.plan_hash[:12]}…")
    observed = phase_b_source_digest(REPO_ROOT)["digest"]
    if prereg["executable_source"]["digest"] != observed:
        return False, ("the preregistration was frozen against executable "
                       f"{prereg['executable_source']['digest'][:12]}… but the "
                       f"executable is {observed[:12]}…; re-freeze it rather than "
                       "running code the record does not describe")
    by_id = {p["qualified_id"]: p for p in prereg["calibration_profiles"]}
    for profile in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2):
        entry = by_id.get(profile.qualified_id)
        if entry is None:
            return False, f"{profile.qualified_id} is not in the preregistration"
        if entry["spec_identity"]["profile_hash"] != profile.profile_hash:
            return False, f"{profile.qualified_id}: spec hash moved"
        if entry["materialized_identity"]["content_sha256"] != profile.content_sha256:
            return False, (f"{profile.qualified_id}: materialized content moved; "
                           "profile_hash does not identify the sampled bytes")
    return True, f"preregistration {stated[:12]}… binds this executable and both mixtures"


def reuse_record_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Phase B prices ten probes because three are cited. Prove it before paying.

    If the reuse record is missing or unverified the session would run twelve
    probes against a ten-probe budget and be killed mid-rung by the watchdog.
    """
    if not REUSE_RECORD.is_file():
        return False, (f"{REUSE_RECORD.name} is missing; the ten-probe budget "
                       "assumes three cited probes")
    record = json.loads(REUSE_RECORD.read_text())
    if not record.get("reuse_verified"):
        return False, (f"{REUSE_RECORD.name} reports reuse_verified=false: "
                       f"{record.get('failures')}")
    admitted = record.get("admitted_reusable_probes") or []
    if len(admitted) < 3:
        return False, (f"only {len(admitted)} probes are admitted-and-reusable; "
                       "the budget assumes at least the three sa citations")
    return True, f"{len(admitted)} verified probes citable; pricing ten, not twelve"


def historical_reuse_reconstruction_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Re-run the STRICT reconstruction here, on the machine that has the bytes.

    `reuse_record_gate` above reads the committed record. That is the artifact
    the pricing consumes, so it must be checked — but a record is a claim about
    bytes, and this gate is the claim being re-earned: it re-derives every
    probe's `student_artifact_digest` from the retained Phase-A checkpoints under
    the dev-box artifact store and refuses to create a pod if any citation no
    longer reconstructs.

    It lives here rather than in the pod's setup gate because the retained store
    is a dev-box artifact store and is intentionally not transported. Phase-B
    attempt 1 spent `$0.15` learning that a pod cannot answer this question. The
    answer is to ask the right machine, at the right moment — before the provider
    is touched — not to lower the standard.

    Checked: 11 probes; every artifact digest re-derived from bytes;
    `reuse_verified`; no failures; the evidence set still digests to what the
    record describes; and the eight citations the ten-probe budget depends on.
    """
    expected = {*PHASE_A_IMPORTED_FINALISTS, CANONICAL_CONTROL}
    if set(REUSE_ADMITTED) != expected:
        return False, ("the verifier admits candidates "
                       f"{sorted(REUSE_ADMITTED)} but the frozen Phase-B rule "
                       f"admits {sorted(expected)}")
    try:
        live = verify_historical_reuse()
    except Exception as exc:                     # noqa: BLE001 - refuse, never crash
        return False, (f"the strict historical-reuse verifier raised "
                       f"{type(exc).__name__}: {exc}")

    if live["n_probes"] != 11:
        return False, (f"the verifier found {live['n_probes']} historical probes, "
                       "not the 11 the cross-phase procedure was frozen against")
    unreconstructed = [p["probe_id"] for p in live["probes"]
                       if not p["checks"]["artifact_digest_re_derives_from_bytes"]]
    if unreconstructed:
        return False, (f"{len(unreconstructed)} probe(s) no longer re-derive their "
                       f"artifact digest from the retained checkpoint bytes: "
                       f"{unreconstructed[:3]}")
    if not live["reuse_verified"] or live["failures"]:
        return False, (f"strict reconstruction failed: {live['failures']}")

    if not REUSE_RECORD.is_file():
        return False, f"{REUSE_RECORD.name} is missing; nothing to reconcile against"
    record = json.loads(REUSE_RECORD.read_text())
    if live["probes_dir_digest"] != record.get("probes_dir_digest"):
        return False, ("the evidence set has drifted since the record was "
                       f"written: live {live['probes_dir_digest'][:12]}… vs "
                       f"recorded {str(record.get('probes_dir_digest'))[:12]}…; "
                       "re-run the verifier rather than citing a stale record")

    admitted = set(live["admitted_reusable_probes"])
    missing = sorted(REQUIRED_CITATIONS - admitted)
    if missing:
        return False, (f"the citations Phase B books no probe for are not all "
                       f"reusable: missing {missing}")
    if admitted != set(record.get("admitted_reusable_probes") or []):
        return False, ("the live admitted set disagrees with the committed "
                       "record, so the pricing was computed over a different "
                       "evidence set than the one on disk")
    return True, (f"{len(admitted)} citations re-derived from retained bytes; "
                  f"evidence set {live['probes_dir_digest'][:12]}… matches the record")


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id="autoinit-phase-b",
        schema="aadistill.autoinit.phase_b_session/v1",
        description=("Phase B: joint P=2 search over calib.domain_balanced@v1 and "
                     "calib.reasoning_heavy@v2, then cross-phase rungs citing "
                     "verified Phase-A evidence"),
        authorization_path=AUTH_PATH,
        #: Refused by schema if a Phase-A artifact is passed. Phase A is
        #: complete; its grant measures a different harness at a different price.
        authorization_loader=PhaseBAuthorization.load,
        plan_id=PHASE_B_PLAN_V1.plan_id,
        plan_hash=PHASE_B_PLAN_V1.plan_hash,
        budget=phase_b_budget(args),
        setup=SetupManifest(
            relay_inputs=(*CANONICAL_INIT, *RECOVERY_LADDER, *CALIBRATION_V1,
                          *imported_finalist_inputs()),
            local_assets=(*PHASE_A_LOCAL_ASSETS, *CALIBRATION_V2_LOCAL),
            env={"SESSION_KIND": "phase_b"},
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
            test_ignores=PHASE_B_TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_phase_b",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("PHASE_A_FAILED", "PHASE_A_INCOMPLETE"),
            incomplete=("PHASE_A_INCOMPLETE",),
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. A stage-0 comparability failure is a "
                          "TERMINATE, not a trigger for a larger run.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_phase_a",
            evidence_filename="phase_a_evidence.json",
            archive_basename="phase_a_artifacts.tar.gz",
            #: Phase-B-specific, because the Phase-A specs collect the Phase-A
            #: search journal. Attempt 3 hit its search deadline and the journal
            #: — per-state timings, beam contents, how close it came — was
            #: deleted with the pod because the collector looked in
            #: `phase_a_search`. The probe minimums here are also Phase B's own.
            spec_success="configs/autoinit/phase_b_artifacts.json",
            spec_failed="configs/autoinit/phase_b_artifacts_failed.json",
            report_names=("phase_a_evidence.json",
                          "attested_evaluation_protocol.json",
                          "phase_b_stage0_binding.json", "search_result.json",
                          "device_handoff.json", "rung1_selection.json",
                          "leaf_retention.json", "rung2_selection.json",
                          "phase_a_result.json"),
            event_streams=probe_streams,
            #: `fetch_selected_leaves` returns TRANSFER RESULTS — dicts carrying
            #: `state_id`, `rc` and `matched` — which is exactly what
            #: `selected_leaves_secured` reads. `finalists_to_fetch` returns
            #: canonical-id STRINGS, and pairing the two would have failed only
            #: after a scientifically successful run, or left the P=2 Top-5
            #: bytes unsecured. That mismatch is the same shape as the defect
            #: that mislabelled a successful Phase A as INCOMPLETE.
            #:
            #: `fetch_finalists` is deliberately NOT chained here: for Phase B its
            #: retention record also names the two imported Phase-A finalists,
            #: which are already retained off-pod, and re-fetching them would
            #: spend transfer time on bytes the dev box already holds.
            fetch_products=fetch_selected_leaves,
            products_secured=selected_leaves_secured),
        teardown=TeardownPolicy(
            note="nothing chains off Phase B; it is a terminus like Phase A"),
        precheck=(
            session_commit_gate(REPO_ROOT, AUTH_PATH, check_lineage=True),
            frozen_science_plan_gate(REPO_ROOT, FROZEN_SCIENCE_PLAN),
            phase_b_source_gate,
            preregistration_gate,
            reuse_record_gate,
            historical_reuse_reconstruction_gate,
            disk_provision_gate,
            imported_finalists_staged_gate,
            ckpt_store_capacity_gate,
            poll_lifetime_gate,
        ),
        evidence_fields={
            "phase": "B",
            "phase_b_session_plan_hash": PHASE_B_PLAN_V1.plan_hash,
            "calibration_profiles": {
                p.qualified_id: {"profile_hash": p.profile_hash,
                                 "content_sha256": p.content_sha256}
                for p in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)},
            "runs_a_search": True,
            "search_profiles": 2,
            "priced_probes": RUNG1_PROBES_P2 + RUNG2_PROBES_P2 + TIE_BREAK_PROBES_P2,
            "cited_probes": "three, from verified Phase-A evidence",
            "retrains_permanent_controls": False,
            "redefines_thresholds": False,
            "followon_started": False,
            "followon_reachable_from_this_launcher": False},
    )


def build_parser() -> argparse.ArgumentParser:
    from autoinit_phase_a_launch import build_parser as phase_a_parser

    ap = phase_a_parser()
    ap.set_defaults(out="logs/autoinit_phase_b_session.json",
                    disk_gb=PHASE_B_PROVISION_GIB,
                    search_minutes=SEARCH_MINUTES_P2,
                    rung1_probes=RUNG1_PROBES_P2,
                    rung2_probes=RUNG2_PROBES_P2,
                    tie_break_probes=TIE_BREAK_PROBES_P2,
                    #: Resolved in `main()` from the priced plan. NOT a Phase-B
                    #: constant: Phase A's historical 1320 stays exactly where it
                    #: is, and a number here would go stale the next time the
                    #: envelope moves — which is precisely how 1320 became wrong.
                    poll_limit_min=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_limit_min is None:
        args.poll_limit_min = phase_b_poll_limit_minutes(args)
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("STOP for review. Phase B searched two "
                                "calibration distributions jointly and cited "
                                "verified Phase-A evidence; nothing chains off it."))


if __name__ == "__main__":
    raise SystemExit(main())
