#!/usr/bin/env python3
"""The Phase-C2 Search-1 session, as a specification. NOT AUTHORIZED.

    PYTHONPATH=src setsid nohup python -u scripts/pod/autoinit_phase_c2_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

This file declares WHAT the session is. How a session is run — detached start
with a durable descriptor, an independent provider-side watchdog, continuous log
relay, four budget thresholds, the artifact gate and provider-confirmed teardown
— lives once, in `aadistill.infrastructure.session_runner`, and is not
inherited, subclassed or retargeted by anybody.

**It cannot run today, and not because of a missing flag.** `SESSION_AUTH_PATH`
names an artifact that does not exist, and `C2Authorization.load` refuses
anything that is not a C2 grant under its own schema. There is no grant, no
readiness record, no bundle and no provider resource, so the only thing this
launcher can currently do is refuse.

What makes it a search session rather than a copy of another one:

* **Two stages and no training.** `arms=0`; there is no probe, no battery, no
  rung and no elimination, so none of that is declared and none of it can be
  reached.
* **Its own audit root and its own search workdir.** `audit/autoinit_phase_c2`
  and `autoinit/phase_c2_search`, named once here and once in the artifact spec,
  with a test holding the two equal — because Phase-B attempt 3 wrote
  `phase_b_search` while its specs named `phase_a_search`, the collector matched
  nothing, and a deadline failure came home with no per-state timings.
* **The evidence document is a `whole_file` relay spec** by virtue of being the
  `ArtifactPolicy` evidence file: the driver rewrites it on every state change,
  and the runner declares exactly that spec as rewritten-in-place.
* **A disk gate before the provider is contacted.** The search's peak working
  set is derived from the real state geometries, not guessed, and a session
  asking for less disk than the search needs is refused at `$0` rather than
  filling the volume at level 1 with 87 GiB of intermediates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
#: `scripts` too: the experiment instances live under `experiments.` since the
#: core/application separation, and this file is also run as a subprocess with a
#: caller-set PYTHONPATH.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
#: The sibling science-input declarations. Present when this file is run
#: directly; absent when a test loads it by path, which is how the structural
#: checks load every launcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.infrastructure.budget import Phase  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, ExecutionCommands, LocalAsset, MarkerPolicy,
    SessionContext, SessionSpec, SessionSpecError, SetupManifest,
    TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate)
from aadistill.infrastructure.session_runner import run_session  # noqa: E402
from aadistill.runtime.pod_environment import LAUNCH_BOUND  # noqa: E402
from aadistill.runtime.staging_contract import (  # noqa: E402
    derive_contract, ignores_for_selection)
from experiments.deployment import POD_IMAGE, deployment_commands  # noqa: E402
from experiments.phase_c2 import bundle as BUNDLE  # noqa: E402
from experiments.phase_c2 import pod_environment as PE  # noqa: E402
from experiments.phase_c2.session import (  # noqa: E402
    C2_PLAN_ID, C2_RUN_EXPERIMENT_ID, C2_RUN_ROLES, C2Authorization,
    c2_authorization_path, c2_budget_spec, c2_current_executable,
    c2_hard_ceiling_usd, c2_plan_hash, c2_price_per_hour_usd, c2_run_path,
)

WS = POD_IMAGE["workspace_root"]
REPO = POD_IMAGE["checkout_root"]

from autoinit_science_inputs import CALIBRATION_V1, CANONICAL_INIT  # noqa: E402

STATUS = f"{WS}/autoinit_phase_c2.status"
RUN_LOG = f"{WS}/autoinit_phase_c2_run.log"

#: The stage this experiment's runs are placed under, read from the experiment's
#: own configuration rather than decided here.
RUN_STAGE_ID = json.loads(
    (REPO_ROOT / "configs/experiments/phase_c2/authorization.json").read_text()
)["stage_id"]


def auth_path_for(run_id: str) -> str:
    """Where THIS run's authorization lives, repository-relative.

    Run-owned, with no repository-level fallback. C1's authorization was one
    root file that every issuance overwrote, so the artifact a session ran under
    sat where the next issuance would replace it; unwinding that took a pointer,
    a history file and a migration. C2 starts where C1 arrived.
    """
    return c2_authorization_path(run_id, RUN_STAGE_ID)


def bundle_record_for(run_id: str) -> str:
    """Where this run's staged-bundle record lives.

    The one artifact that must stay UNCOMMITTED inside a launch window:
    committing it would add a third path to the session lineage diff and
    `session_commit_gate` would refuse.
    """
    return c2_run_path(run_id, "bundle_record", RUN_STAGE_ID)

TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

#: Dev-box-only assets the pod cannot fetch from the relay.
#:
#: `reasoning_heavy_v2` travels this path and not the relay for the reason
#: Phase B recorded: it was built on the dev box at `$0` and has never been
#: uploaded, so declaring it as a `RelayInput` would name an object the pod
#: cannot fetch — which is exactly what killed Phase-A attempt 5 at $0.6426.
#: 780 KB over a measured 0.72 MB/s uplink is about a second.
#:
#: `state_eval_v1` is the suite every candidate is measured on. Without it there
#: is no ranking metric at all.
LOCAL_ASSETS = (
    LocalAsset("artifacts/stage1/reasoning_heavy_v2", "reasoning_heavy_v2",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
)

#: The directory the paid pod runs, named so the contract is greppable from the
#: launcher rather than only inferable from what is missing.
POD_TEST_SELECTION = "tests/c2_preflight"

#: Ignored by the pod's blocking test gate: everything that is not the
#: selection, DERIVED from the tree at launch time.
#:
#: This declared two file ignores until 2026-09-15 and therefore sent the pod the
#: entire repository. A $0 probe measured what that costs — 3976 passed, 65
#: skipped, 907 s, 33.6% of the gate's 2700 s timeout — for a suite that
#: regression-tests other experiments' history, this project's documentation and
#: budget arithmetic on a machine billing $1.09/h. AGENTS.md P8.2.1: a paid
#: experiment machine runs only what that experiment needs.
#:
#: Derived rather than listed, unlike C1's, because a complement fails unsafely
#: when it is hand-maintained: a new top-level test directory joins the paid
#: suite silently. Derivation makes the default EXCLUDED, and the 65 skips the
#: probe recorded stop being a launch problem — a readiness sweep over this
#: selection has no expected-skip map to populate because there are no skips.
TEST_IGNORES = ignores_for_selection(POD_TEST_SELECTION, REPO_ROOT)

#: Derived, not guessed. Peak resident search states = the kept parents plus the
#: level being generated plus the leaves accumulated so far, over the real
#: geometries: 87.4 GiB, at level 1, where 5 retained level-0 states (still
#: teacher-width, ~6.8 GiB each) are expanded into 18 children. Plus the teacher
#: (7.5 GiB), the repo, the venv and the conditional baseline rebuild's four
#: intermediates. Provision covers that with headroom, because disk is cents and
#: a search that fills the volume at level 1 loses every state it has measured.
C2_PEAK_WORKING_GIB = 87.4
C2_PROVISION_GIB = 200


def storage_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Refuse an under-provisioned volume before a pod exists.

    Mechanically known rather than something to discover at level 1: the peak
    working set is a function of the configured space and the target geometry,
    both frozen.
    """
    requested = int(getattr(ctx.args, "disk_gb", 0) or 0)
    if requested < C2_PROVISION_GIB:
        return False, (
            f"--disk-gb {requested} is below the C2 provision of "
            f"{C2_PROVISION_GIB} GiB. The search's peak working set is "
            f"{C2_PEAK_WORKING_GIB:.1f} GiB of intermediate states, reached at "
            "level 1, plus the teacher, the checkout and the venv. A volume "
            "that fills there loses every state measured up to that point.")
    return True, (f"volume {requested} GiB >= {C2_PROVISION_GIB} for a "
                  f"{C2_PEAK_WORKING_GIB:.1f} GiB peak working set")


def pricing_identity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The grant's ceiling must be the one the pricing record derives.

    `load_pricing` already refuses a record that does not match its own
    `pricing_sha256`. This is the other half: a grant written against a figure
    the record does not produce is a grant for a different session.
    """
    ceiling = c2_hard_ceiling_usd(REPO_ROOT)
    if abs(float(ctx.auth.hard_cap_usd) - ceiling) > 1e-9:
        return False, (
            f"the grant caps at ${ctx.auth.hard_cap_usd:.4f} and the pricing "
            f"record derives ${ceiling:.4f}. One of them is describing a "
            "different session; refusing to launch either.")
    return True, (f"grant cap ${ceiling:.4f} matches the pricing record, "
                  f"whose own sha256 verified")


def plan_identity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The grant must bind the space this launcher is about to search.

    `c2_plan_hash` is derived from the session contract AND the configured
    space, so a grant issued against a different order policy, a different
    profile restriction or a different implementation set cannot authorize this
    run.
    """
    live = c2_plan_hash()
    if ctx.auth.plan_hash != live:
        return False, (
            f"the grant binds plan {str(ctx.auth.plan_hash)[:12]}… and the "
            f"configured space hashes to {live[:12]}…. The searched space has "
            "moved since the grant was issued; re-issue it or revert the space.")
    return True, f"grant binds the live plan {live[:12]}…"


def c2_executable_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Re-derive the C2 executable set here, independently of the artifact.

    `session_commit_gate` and the bundle round-trip both digest whatever file
    list the authorization *stores*. That is what lets a session declare its own
    executable, and it is also the one thing an artifact could get wrong in its
    own favour: a grant carrying another phase's list, or the superseded
    eighteen-path declaration, would verify perfectly against those files while
    this launcher, this driver, the search seam, the baseline rebuild and the
    comparison record went unmeasured.

    So the live closure is derived and the artifact is required to declare
    exactly it.
    """
    try:
        live = c2_current_executable(REPO_ROOT)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot derive the C2 executable set: {exc}"
    expected = tuple(f["path"] for f in live["files"])
    declared = tuple(getattr(ctx.auth, "harness_source_files", ()) or ())
    if declared != expected:
        from experiments.phase_c2.session import C2_HARNESS_SOURCE_FILES_V1

        if declared == C2_HARNESS_SOURCE_FILES_V1:
            return False, (
                "the authorization declares the SUPERSEDED eighteen-path C2 "
                "harness set, which was hand-maintained and named none of the "
                f"{len(expected) - len(declared)} further files the live walk "
                "finds. Its digest cannot describe what this session executes. "
                "Re-issue against the derived closure.")
        if not declared:
            return False, (
                "the authorization declares NO harness file set, so there is "
                "nothing for the commit gate or the bundle round-trip to "
                "re-digest. An empty set matches nothing, by design.")
        return False, (f"the authorization declares a different executable set "
                       f"({len(declared)} paths) than this session derives "
                       f"({len(expected)})")
    stored = getattr(ctx.auth, "harness_source_digest", None)
    if stored and stored != live["digest"]:
        return False, (f"executable digest {stored[:12]}… in the authorization "
                       f"does not match the live tree {live['digest'][:12]}…")
    ctx.evidence["c2_executable_closure"] = {
        "digest": live["digest"], "n_files": live["n_files"],
        "entry_points": list(live["entry_points"]),
        "declared_non_python_inputs": list(live["declared_non_python_inputs"]),
        "rule": live.get("rule"),
    }
    return True, (f"C2 executable closure {live['digest'][:12]}… over "
                  f"{live['n_files']} derived files")


def pod_environment_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Has this exact executable been proved to survive a pod's test gate?

    C1 attempt 3R is the reason the mechanism exists: it cleared every readiness
    marker and then died on a CPU test suite that had never been run under the
    conditions a fresh pod is in — empty `$HOME`, its own `HF_HOME`, no dataset
    cache, no credential file.

    The sweep that answers it cannot run while a pod bills, so it is run once
    against a committed tree and recorded, and this gate checks in milliseconds
    that the recording still describes the code that would run: the live C2
    executable digest and the pod test-environment digest must both still match,
    the staged view the sweep ran under must be the one this session stages, and
    the session commit must descend from `swept_base_commit` with no tracked path
    changed beyond the record itself and this session's authorization.

    **And the record must be `launch_bound`.** A `diagnostic` record proves the
    machinery works on a tree and stays valid as that; it is not the sweep a paid
    launch rests on. C1's gate accepted either for a month and merely copied the
    kind into evidence, so the promised launch-bound sweep need never have
    happened.
    """
    run_id = getattr(ctx.args, "run_id", None)
    if not run_id:
        return False, "this session has no run_id, so it owns no readiness record"
    record_rel = PE.record_path_for(run_id, RUN_STAGE_ID)
    try:
        record = PE.load_record(REPO_ROOT, run_id=run_id, stage_id=RUN_STAGE_ID)
    except FileNotFoundError:
        return False, (f"{record_rel} does not exist: no pod-like sweep has been "
                       "recorded for this run. A sweep is taken on the clean "
                       "pre-authorization tree and written into the run it is "
                       "for — `record_pod_environment.py --experiment phase_c2 "
                       f"--kind launch_bound --run-id {run_id} --stage-id "
                       f"{RUN_STAGE_ID}`.")
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot read {record_rel}: {exc}"

    try:
        live_staging = derive_contract(
            spec(ctx.args).setup, session_id="autoinit-phase-c2-search1")["digest"]
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot derive this session's staging contract: {exc}"

    ok, reason = PE.verify_record(
        record, REPO_ROOT, run_id=run_id, stage_id=RUN_STAGE_ID,
        session_commit=getattr(ctx.args, "session_commit", None),
        authorization_path=auth_path_for(run_id),
        required_kind=LAUNCH_BOUND,
        staging_contract_digest=live_staging)
    ctx.evidence["pod_environment_verification"] = {
        "verdict": "PASS" if ok else "FAIL",
        "record": record_rel,
        "record_self_sha256": record.get("self_sha256"),
        "record_kind": record.get("record_kind"),
        "required_record_kind": LAUNCH_BOUND,
        "swept_base_commit": record.get("swept_base_commit"),
        "session_commit": getattr(ctx.args, "session_commit", None),
        "permitted_post_sweep_paths": [record_rel, auth_path_for(run_id)],
        "live_staging_contract_digest": live_staging,
        "recorded_staging_contract_digest": record.get("staging_contract_digest"),
        "counts": record.get("counts"),
        "pod_test_selection": PE.POD_TEST_SELECTION,
        "unexpected_environment_skips": record.get("unexpected_environment_skips"),
        "reason": reason,
    }
    return ok, reason


def bundle_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Can a pod, RIGHT NOW, obtain the exact authorized code?

    C1 attempt 1 answered every other question correctly and died at
    `SETUP_RC=1` fetching an alias for nothing. Eight gates verified the
    *contents* of the session commit; none asked whether the pod could reach it.

    So this one operates on the object the pod would actually fetch: it derives
    the canonical name from `--session-commit`, downloads that exact relay
    object, hashes it against the staged bundle, `git bundle verify`s the
    round-tripped bytes, clones them, and requires the checkout to be the exact
    session commit, to carry the exact authorization this launcher is loading,
    and to digest to the authorized executable value.

    Read-only: it uploads nothing and mutates nothing. Preparation is
    `scripts/autoinit/stage_c2_bundle.py`, deliberately a separate command, so
    what this verifies is the relay's state rather than a side effect of the
    verification.
    """
    run_id = getattr(ctx.args, "run_id", None)
    commit = ctx.args.session_commit
    try:
        BUNDLE.require_canonical_bundle_arg(ctx.args.bundle, commit)
    except BUNDLE.BundleTransportError as exc:
        return False, str(exc)

    bundle_rel = bundle_record_for(run_id)
    staged = REPO_ROOT / bundle_rel
    if not staged.is_file():
        return False, (f"{bundle_rel} is missing; run "
                       f"scripts/autoinit/stage_c2_bundle.py --run-id {run_id} "
                       f"--session-commit {commit} first")
    record = json.loads(staged.read_text())
    if record.get("session_commit") != commit:
        return False, (f"{bundle_rel} describes a bundle for "
                       f"{str(record.get('session_commit'))[:12]}…, not the "
                       f"session commit {commit[:12]}…")

    auth_rel = auth_path_for(run_id)
    auth_file = REPO_ROOT / auth_rel
    if not auth_file.is_file():
        return False, (f"{auth_rel} does not exist, so there is no authorization "
                       "for the round-trip to find inside the bundle")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = BUNDLE.roundtrip(
                session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=auth_file.read_bytes(),
                authorization_path=auth_rel,
                #: The AUTHORIZED pair, not the live one. `c2_executable_gate`
                #: has already required the two to agree; asking the round-trip
                #: about the live digest instead would make a stale
                #: authorization unfalsifiable here.
                expected_harness_digest=ctx.auth.harness_source_digest,
                harness_files=tuple(ctx.auth.harness_source_files),
                workdir=Path(tmp))
    except Exception as exc:                                   # noqa: BLE001
        return False, f"the pod could not obtain the authorized commit: {exc}"

    ctx.evidence["bundle_staged_check"] = evidence
    return True, (f"{evidence['canonical_bundle_name']} ({evidence['bytes']} "
                  f"bytes, {evidence['remote_sha256'][:12]}…) round-trips to "
                  f"{evidence['roundtrip_head'][:12]}… carrying this "
                  f"authorization and executable set "
                  f"{evidence['roundtrip_harness_digest'][:12]}…")


def reserve_minutes(plan, name: str) -> float:
    """One named reserve from the plan. By NAME, because the names are the
    partition: summing them would hand the beam a clock the accounting says
    belongs to the baseline rebuild."""
    for reserve in plan.soft_stop_reserves:
        if reserve.name == name:
            return float(reserve.minutes)
    raise SessionSpecError(
        f"the plan declares no {name!r} reserve; the runtime partition cannot "
        f"be derived from it. Declared: "
        f"{[r.name for r in plan.soft_stop_reserves]}")


def beam_envelope_minutes(plan) -> float:
    """What the beam may spend: the DEPTH-early base plus its own risk reserve.

    **Not** the sum of all reserves. `baseline_rebuild_reserve` is excluded on
    purpose: the pricing record separates the two because they buy different
    things, and a deadline of `base + every reserve` let the beam run into the
    27.665 minutes held for a missing-B rebuild. Enforcement now matches the
    accounting, which is a partition repair and adds nothing to the ceiling.
    """
    base = next(p.minutes for p in plan.breakdown
                if p.name == "beam_search_depth_early")
    return base + reserve_minutes(plan, "beam_composition_risk")


def driver_command(ctx: SessionContext, plan) -> str:
    """The pod's command. Every figure comes from the plan, once, by name.

    Three minute numbers with three meanings:

    * `--search-deadline-minutes` bounds the beam at runtime, and is the beam's
      whole envelope — base plus `beam_composition_risk` and nothing else.
    * `--search-minutes` funds the affordability check taken before the beam
      starts. It is the SAME envelope, because approving the beam against its
      expected 300.16-minute trajectory would approve work its own deadline
      permits and the soft stop may not fund.
    * `--baseline-rebuild-minutes` is the conditional reserve, spent only if the
      deterministic rule finds B absent, on a clock that starts then.
    """
    beam = beam_envelope_minutes(plan)
    rebuild = reserve_minutes(plan, "baseline_rebuild_reserve")
    #: The interpreter is a DEPLOYMENT fact, from the same declaration the
    #: runner reads for the collector. Not `ctx.args`, which has no such field:
    #: the first version of this line said `ctx.args.remote_python` and would
    #: have raised AttributeError at driver start, on a billing pod, after setup
    #: had already succeeded.
    #: FLOORED, not rounded. `:.2f` rounds to nearest, so the plan's
    #: $14.499561 reached the driver as `14.50` — three hundredths of a cent
    #: MORE than the plan funds. A limit handed downward rounds DOWN or it is
    #: not a limit, which is the mirror of the rule a ceiling obeys when it is
    #: written: C1's record rounds its ceiling UP for the same reason.
    def floor2(value: float) -> str:
        return f"{math.floor(value * 100) / 100:.2f}"

    return (f"{POD_IMAGE['remote_python']} "
            f"scripts/pod/autoinit_phase_c2_driver.py --stage all "
            f"--image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price} --spent-usd {ctx.spent_usd:.3f} "
            f"--soft-stop-usd {floor2(plan.soft_stop_usd)} "
            f"--authorized-usd {floor2(plan.hard_terminate_usd)} "
            f"--search-minutes {beam:.2f} "
            f"--search-deadline-minutes {beam:.2f} "
            f"--baseline-rebuild-minutes {rebuild:.3f} "
            f"--authorization-path {auth_path_for(ctx.args.run_id)} "
            f"--device cuda")


def spec(args) -> SessionSpec:
    """The whole session, in one object. Nothing about it lives anywhere else."""
    budget = c2_budget_spec(REPO_ROOT)
    return SessionSpec(
        session_id="autoinit-phase-c2-search1",
        schema="aadistill.autoinit.c2_session/v1",
        description=(
            "Phase C2 Search-1: one beam search over four operator kinds, order "
            "free, ATTENTION fixed to activation_importance_v1 and branching "
            "over two calibration mixtures; plus, conditionally, one rebuild of "
            "the frozen C1 baseline B. Trains nothing."),
        authorization_path=auth_path_for(args.run_id),
        authorization_loader=C2Authorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=C2_PLAN_ID,
        plan_hash=c2_plan_hash(),
        budget=budget,
        setup=SetupManifest(
            #: The canonical init is staged because `run_phase_a_search`
            #: injects it as the measured control, verified against its frozen
            #: single-file sha256. The domain-balanced mixture comes by relay
            #: because it is already there; the reasoning-heavy one is a local
            #: asset above.
            relay_inputs=(*CANONICAL_INIT, *CALIBRATION_V1),
            local_assets=LOCAL_ASSETS,
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "TEACHER_REVISION"),
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "TEACHER_READY",
                           "ROPE_OK", "TESTS_OK", "AUTHORIZATION_OK",
                           "SETUP_DONE"),
            #: DECLARED. An undeclared kind falls through to
            #: `SESSION_KIND=spend`, whose branch loads a
            #: `PreflightAuthorization` — and a branch that reads the artifact
            #: through the wrong type is worse than a missing one, because it
            #: can SUCCEED while binding this session to another phase's file
            #: list and price. Phase-B attempt 2 proved that at $0.2300, a
            #: KeyError one step after the pod's test gate passed.
            env={"SESSION_KIND": "c2"},
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_phase_c2_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("PHASE_C2_FAILED",),
            incomplete=(),
            #: Search-1 produces no checkpoint anybody fetches: its leaves are
            #: measured on the pod and their identities come home in the
            #: journal. So there are no products, and saying so is what stops a
            #: collector looking for weights this session never exports.
            products_eligible=lambda terminal, stages: False,
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. Nothing was trained and no permanent "
                          "artifact was replaced.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_phase_c2",
            evidence_filename="c2_evidence.json",
            archive_basename="c2_search1_artifacts.tar.gz",
            spec_success="configs/autoinit/c2_artifacts.json",
            spec_failed="configs/autoinit/c2_artifacts_failed.json",
            report_names=("c2_evidence.json", "c2_search_summary.json")),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
        #: EVERY gate runs before a pod exists, and each one names a failure a
        #: paid session has actually had. Nothing here is defence in depth:
        #:
        #: * the commit gate asks whether the tree the pod checks out is the
        #:   authorized one, carries this exact authorization, and differs from
        #:   the authorized base in nothing else;
        #: * the executable gate re-derives the closure independently, because
        #:   every other check digests the set the ARTIFACT declares;
        #: * storage, pricing and plan refuse an under-provisioned volume, a
        #:   mis-priced grant and a grant bound to a moved search space;
        #: * the readiness gate requires a launch-bound sweep that still
        #:   describes this executable, this staged view and this commit;
        #: * the bundle gate asks whether a pod could obtain that code at all,
        #:   which is the question C1 attempt 1 paid $0.0786 to discover nobody
        #:   was asking.
        precheck=(
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", "")),
                                check_lineage=True),
            c2_executable_gate,
            storage_gate,
            pricing_identity_gate,
            plan_identity_gate,
            pod_environment_gate,
            bundle_staged_gate,
        ),
        evidence_fields={
            "c2_plan_hash": c2_plan_hash(),
            "trains_anything": False,
            "peak_working_gib": C2_PEAK_WORKING_GIB,
            "search_is_not_a_recovery_result": (
                "the state_eval ranking is a hypothesis generator. A search "
                "winner is not a demonstrated initialization improvement, and "
                "behavioural confirmation is a separately authorized paid "
                "experiment that this launcher cannot reach."),
            "search2_reachable_from_this_launcher": False,
        })


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test can assert on the namespace it
    produces rather than on a transcription of it."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    #: REQUIRED. Every governance artifact this session consumes is owned by its
    #: run — grant, authorization, readiness record, bundle record — and each is
    #: resolved from this id. A session with no run id has no grant that can
    #: belong to it, no readiness record of its own, and nowhere for its
    #: authorization to live that the next issuance would not overwrite.
    ap.add_argument("--run-id", required=True,
                    help="the attempt this session runs as, e.g. attempt2")
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    #: From the pricing record, so there is one hand-maintained price in the
    #: repository and it is the one `pricing_identity_gate` verifies. A stale
    #: value here can only ever refuse a launch, never buy one.
    ap.add_argument("--max-price", type=float,
                    default=c2_price_per_hour_usd(REPO_ROOT))
    ap.add_argument("--disk-gb", type=int, default=C2_PROVISION_GIB)
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--host-draws", type=int, default=3)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    #: The hard-terminate envelope plus slack. A poll limit below the priced
    #: ceiling would stop watching a session that is still billing.
    ap.add_argument("--poll-limit-min", type=float, default=900.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out",
                    default="logs/stages/stage-1/phase_c2/runs/"
                            "autoinit_phase_c2_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("Search-1 is a terminus: Search-2 is "
                                "conditional on this evidence and behavioural "
                                "confirmation is separately authorized."))


if __name__ == "__main__":
    raise SystemExit(main())
