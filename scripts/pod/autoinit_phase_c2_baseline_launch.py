#!/usr/bin/env python3
"""Launch ONE Phase-C2 baseline-completion session through the generic runner.

    PYTHONPATH=src python scripts/pod/autoinit_phase_c2_baseline_launch.py \
        --scr /path/to/scratch --session-commit <sha> --bundle <name> \
        --run-id attempt5

**Thin on purpose.** Search-1's launcher is large because a ten-hour beam over
87 GiB of intermediates has a lot of ways to go wrong. This session rebuilds one
checkpoint, measures it once and writes one record, so it declares less: no
canonical control to stage, no vLLM environment, no beam envelope, no
conditional reserve, and a working set an order of magnitude smaller. Everything
it does use is the machinery every other session uses -- `SessionSpec`, the run
layout, the grant/authorization primitives, the resource scope, the setup
declaration, the bundle transport, the readiness record, the artifact collector
and the teardown policy.

**A grant for this session cannot buy a beam.** The authorization is a distinct
type with a distinct schema reporting `authorizes_c2_search1 = False`, and
`BaselineCompletionAuthorization.load` refuses an artifact that claims
otherwise. The Search-1 launcher's own loader refuses this schema symmetrically.
Nothing here imports the beam runner, and
`tests/pod/test_phase_c2_baseline_completion.py` asserts that over the import
graph.

**What it stages, and nothing else.** The pinned teacher revision, the two
calibration mixtures the frozen B path consumes, and the frozen `state_eval_v1`
suite. The protocol, the frozen candidate record and the beam ranking arrive
with the bundle because they are committed files. Search-1 additionally staged
the canonical 0.6B init as its measured control; this session has no control and
does not stage it.

THIS FILE AUTHORIZES NOTHING. It refuses to run without a committed
authorization, and no baseline-completion authorization has been issued.
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
for _extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, ExecutionCommands, LocalAsset, MarkerPolicy, SessionContext,
    SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate)
from aadistill.infrastructure.session_runner import run_session  # noqa: E402
from aadistill.runtime.staging_contract import (  # noqa: E402
    derive_contract, ignores_for_selection)
from autoinit_science_inputs import CALIBRATION_V1  # noqa: E402
from experiments.deployment import POD_IMAGE, deployment_commands  # noqa: E402
from experiments.phase_c2 import baseline_completion as BC  # noqa: E402
from experiments.phase_c2 import baseline_completion_bundle as BCT  # noqa: E402
#: The COMPLETION's readiness instance, not the generic runtime module. The
#: generic module owns the mechanism; which experiment, which schema, which
#: harness and which staging contract are this experiment's own facts.
from experiments.phase_c2 import (  # noqa: E402
    baseline_completion_pod_environment as CPE)
from experiments.phase_c2.baseline_completion_pod_environment import (  # noqa: E402
    LAUNCH_BOUND)
from experiments.phase_c2.frozen_assets import STATE_EVAL_ASSET  # noqa: E402
from experiments.run_layout import (  # noqa: E402
    ArtifactSpec as RunArtifactSpec, claim_output_root, open_run, record_run,
    rel_run_dir, write_run_readmes,
)
from phase_a_frozen import TEACHER_REVISION  # noqa: E402

WS = "/workspace"
STATUS = f"{WS}/autoinit_phase_c2_baseline.status"
RUN_LOG = f"{WS}/autoinit_phase_c2_baseline_run.log"

AUDIT_DIRNAME = "autoinit_phase_c2_baseline"

RUN_EXPERIMENT_ID = "phase_c2_baseline_completion"
RUN_STAGE_ID = json.loads(
    (REPO_ROOT / "configs/experiments/phase_c2/authorization.json").read_text()
)["stage_id"]

#: The frozen-asset expectation. The same document Search-1 named, because the
#: asset is the same one and there is exactly one declaration of it.
FROZEN_EXPECT = "configs/experiments/phase_c2/frozen_assets.json"

#: `calib.reasoning_heavy@v2` and the frozen suite come from the dev box;
#: `calib.domain_balanced@v1` is already on the relay. Both mixtures are needed
#: because the frozen B path consumes domain-balanced at DEPTH, FFN and
#: ATTENTION and reasoning-heavy at RESIDUAL_WIDTH.
LOCAL_ASSETS = (
    LocalAsset("artifacts/stage1/reasoning_heavy_v2", "reasoning_heavy_v2",
               "artifacts/stage1"),
    LocalAsset(f"artifacts/stage1/{STATE_EVAL_ASSET}", STATE_EVAL_ASSET,
               "artifacts/stage1"),
)

#: This session's OWN selection. Search-1's asserts that every path SEARCH-1
#: stages is present, including the canonical control it injects as its measured
#: baseline -- which this session has no use for and does not stage. Running
#: Search-1's preflight under this staged view failed two tests that are
#: entirely correct about Search-1, and narrowing a consumed experiment's gate
#: to fit this one would stop it guarding the session it was written for.
POD_TEST_SELECTION = CPE.POD_TEST_SELECTION
TEST_IGNORES = ignores_for_selection(POD_TEST_SELECTION, REPO_ROOT)

#: Derived from what this session actually holds at once: the teacher in bf16
#: (7.5 GiB), the four materialized steps of the B path (the widest is the
#: pre-DEPTH teacher-width intermediate at ~6.8 GiB, and the final B is 2.22
#: GiB), the repository and the venv. No beam, so no 87 GiB of retained search
#: states -- which is why this asks for a fraction of Search-1's volume.
PEAK_WORKING_GIB = 32.0
#: The provision, and the ONE storage value in this session. The `$0` gate and
#: provider creation both read `args.disk_gb`; there is no second knob that
#: could pass the gate and provision something else.
COMPLETION_PROVISION_GIB = 60


def session_record_path(run_id: str) -> str:
    """Where THIS run's session record goes, repository-relative.

    ONE rule, called by the parser's `--run-id` action and again by `main`, so
    the path the runner writes and the directory the run was opened in cannot
    disagree.
    """
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
            f"/{COMPLETION_RUN_ROLES['session_record']}")


class _RunIdSetsOut(argparse.Action):
    """`--run-id` also produces `out`, because the RUNNER reads `out`.

    `SessionRunner.save()` writes `args.out`, and the argument contract requires
    every attribute the runner reads to come from the REAL parser -- device
    canary attempt 1 died at $0.0603 on an attribute a hand-written namespace
    had and the parser did not, after the pod was billing.

    Deriving it here keeps both properties at once: the namespace is complete,
    and there is no `--out` flag that could point this attempt's session record
    at another run or at an arbitrary repository path.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, value)
        namespace.out = session_record_path(value)


def auth_path_for(run_id: str) -> str:
    """Where THIS run's authorization lives, repository-relative."""
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
            "/governance/authorization.json")


# --- the $0 gates -----------------------------------------------------------
#
# Fewer than Search-1's, and each one names a failure THIS session can have.
# The ones deliberately absent: no beam envelope to price-check, no search-space
# identity to pin, and no 87 GiB volume to defend.


def completion_executable_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Re-derive the completion closure, independently of the artifact.

    Every other check digests the file list the authorization STORES. This one
    derives the live set and requires the artifact to declare exactly it, so a
    grant carrying Search-1's list -- which would verify perfectly against
    Search-1's files -- cannot leave this driver, this comparison path and the
    frozen-input loader unmeasured.
    """
    try:
        live = BC.current_executable(REPO_ROOT)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"cannot derive the completion executable set: {exc}"
    expected = tuple(row["path"] for row in live["files"])
    declared = tuple(getattr(ctx.auth, "harness_source_files", ()) or ())
    if declared != expected:
        only_declared = sorted(set(declared) - set(expected))
        only_live = sorted(set(expected) - set(declared))
        return False, (
            f"the authorization declares {len(declared)} executable file(s) and "
            f"the live closure derives {len(expected)}: "
            f"declared-only {only_declared[:5]}, live-only {only_live[:5]}")
    return True, (f"completion closure {live['digest'][:12]}… over "
                  f"{len(expected)} derived files")


def completion_scope_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The authorization must permit completion and REFUSE a beam."""
    auth = ctx.auth
    if not getattr(auth, "authorizes_c2_baseline_completion", False):
        return False, "this authorization does not permit baseline completion"
    if getattr(auth, "authorizes_c2_search1", False):
        return False, (
            "this authorization claims it can authorize the Search-1 beam. A "
            "completion grant that could buy a ten-hour search is a budget and "
            "a scientific expansion at once.")
    scope = getattr(auth, "resource_scope", None)
    if scope is None:
        return False, (
            "the authorization states no resource scope; one that cannot say "
            "how many provider resources it permits does not permit an unknown "
            "number of them")

    #: Through the scope's OWN methods, not a re-implementation of them. The
    #: gate previously compared `scope.run_id` by hand and never asked about the
    #: draw count at all -- so an operator could pass `--host-draws 8` under an
    #: authorization permitting two, and the refusal would have come from the
    #: provider or from nowhere.
    for permitted, reason in (scope.permits_run(ctx.args.run_id),
                              scope.permits_draws(int(ctx.args.host_draws))):
        if not permitted:
            return False, reason

    #: Stated as a requirement rather than assumed from the type's validator: an
    #: authorization reaching this gate with it false would mean two resources
    #: could bill at once, which is the failure the runner's confirmed-release
    #: rule exists to prevent.
    if scope.one_billing_resource_at_a_time is not True:
        return False, (
            "the authorization does not require one billing resource at a time; "
            "two resources billing at once is not something a grant may waive")

    return True, (
        f"authorized for baseline completion of run {scope.run_id!r}: "
        f"{ctx.args.host_draws} draw(s) within the "
        f"{scope.provider_resources_permitted} permitted, "
        f"{scope.issuances_permitted} issuance(s), "
        f"{scope.launch_attempts_permitted} launch attempt(s), one billing "
        "resource at a time, beam NOT authorized")


def frozen_runtime_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The runtime the comparability contract BINDS, not merely its defaults.

    The parser defaults to the protocol-bound GPU, image and price basis -- but a
    default is only what happens when nobody passes a flag. The frozen C
    measurements were taken on this GPU class and this image, and the formal
    price basis is the accepted boundary, so a launch that overrode any of them
    would be measuring B under a runtime the comparability contract does not
    cover and spending at a rate nobody approved.

    A LOWER max price is safe and permitted: it can only ever refuse a launch.
    """
    expected_gpu = BC.gpu_class(REPO_ROOT)
    expected_image = BC.image_name(REPO_ROOT)
    basis = BC.price_per_hour_usd(REPO_ROOT)

    if ctx.args.gpu != expected_gpu:
        return False, (f"--gpu {ctx.args.gpu!r} is not the bound {expected_gpu!r}. "
                       "The frozen candidate measurements were taken on that "
                       "class and the comparability contract names it.")
    if ctx.args.image != expected_image:
        return False, (f"--image {ctx.args.image!r} is not the bound "
                       f"{expected_image!r}. A different image is a different "
                       "numerical runtime for the one measurement this session "
                       "exists to take.")
    requested = float(ctx.args.max_price)
    if requested > basis:
        return False, (f"--max-price ${requested:.4f}/h exceeds the accepted "
                       f"formal basis ${basis:.4f}/h. A higher ceiling on the "
                       "rate is a budget decision, not a launcher flag.")
    return True, (f"runtime bound: {expected_gpu}, {expected_image}, "
                  f"max price ${requested:.4f}/h <= ${basis:.4f}/h")


def frozen_inputs_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The scientific input, verified before a pod exists.

    The candidate side of the comparison is a committed document. If it does not
    match its own hash, or was not extracted from the ranking this session
    cites, the session has nothing valid to compare a measured B against -- and
    learning that after B has been rebuilt costs the whole rebuild.
    """
    from aadistill.initialization.planning import stage1_selection
    from experiments.phase_c2.frozen_inputs import load_frozen_candidates, load_record
    try:
        record = load_record(REPO_ROOT / BC.FROZEN_INPUTS)
        candidates = load_frozen_candidates(
            REPO_ROOT / BC.FROZEN_INPUTS,
            expect_suite_hash=record["suite"]["hash"],
            expect_policy_hash=record["policy"]["hash"])
        selection = stage1_selection.load(REPO_ROOT / BC.SELECTION_RECORD)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"the frozen comparison inputs are unusable: {exc}"
    committed = record["sources"]["selection_commitment_sha256"]
    if selection["selection_sha256"] != committed:
        return False, (f"the cited ranking commits {selection['selection_sha256']} "
                       f"and the frozen extraction came from {committed}")
    return True, (f"{len(candidates)} frozen candidate measurement(s), record "
                  f"{record['self_sha256'][:12]}…, extracted from the ranking "
                  f"this session cites")


def frozen_assets_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The suite this session measures B on, checked by the real verifier."""
    import subprocess
    #: This gate runs on the DEV BOX, before a pod exists, so it uses this
    #: interpreter. The pod re-asks the same question through the setup script's
    #: ASSETS_READY step against the same expectation document.
    result = subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "scripts/autoinit/verify_frozen_assets.py"),
         "--expect", FROZEN_EXPECT],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        env={"PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/scripts", "PATH": "/usr/bin:/bin"})
    if result.returncode != 0:
        return False, (f"the frozen-asset expectation does not verify: "
                       f"{(result.stdout + result.stderr).strip()[-400:]}")
    return True, f"frozen-asset expectation {FROZEN_EXPECT} verifies"


def pricing_and_plan_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The grant's ceiling and plan must be the documents' own."""
    try:
        ceiling = BC.hard_ceiling_usd(REPO_ROOT)
        plan = BC.plan_hash(REPO_ROOT)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"the completion pricing or protocol does not verify: {exc}"
    declared_cap = float(getattr(ctx.auth, "hard_cap_usd", 0.0) or 0.0)
    if abs(declared_cap - ceiling) > 1e-9:
        return False, (f"the authorization caps ${declared_cap:.4f} and the "
                       f"pricing record's ceiling is ${ceiling:.4f}")
    declared_plan = getattr(ctx.auth, "plan_hash", None)
    if declared_plan and declared_plan != plan:
        return False, (f"the authorization binds plan {declared_plan} and the "
                       f"live protocol hashes to {plan}")
    return True, (f"ceiling ${ceiling:.4f} matches the pricing record, whose own "
                  f"sha256 verified; plan {plan[:12]}…")


def storage_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A volume that cannot hold the B path is refused before it is paid for.

    Reads `disk_gb` -- the same attribute provider creation reads. A gate that
    checked its own flag would be a gate that can pass while the pod is
    provisioned from a different number.
    """
    requested = int(getattr(ctx.args, "disk_gb", 0) or 0)
    if requested < COMPLETION_PROVISION_GIB:
        return False, (
            f"--disk-gb {requested} is below the completion provision of "
            f"{COMPLETION_PROVISION_GIB} GiB. This session holds the teacher in "
            f"bf16, the four materialized steps of the B path and one "
            f"measurement at once -- {PEAK_WORKING_GIB:.1f} GiB -- plus the "
            "checkout and the venv. A volume that fills mid-rebuild loses the "
            "rebuild.")
    return True, (f"volume {requested} GiB >= {COMPLETION_PROVISION_GIB} for a "
                  f"{PEAK_WORKING_GIB:.1f} GiB peak working set")


def readiness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A launch-bound sweep taken for THIS run, of THIS session.

    Asked through the completion's own readiness instance. The launcher used to
    call the GENERIC runtime module's `record_path_for` and `load_record`, which
    resolve under whatever experiment the generic default names -- so a record
    written for Search-1 would have been found, verified against Search-1's
    harness and Search-1's staging contract, and reported as this session's
    readiness. The schema alone now refuses that.
    """
    run_id = getattr(ctx.args, "run_id", None)
    if not run_id:
        return False, "this session has no run_id, so it owns no readiness record"
    try:
        record_rel = CPE.record_path_for(run_id, RUN_STAGE_ID)
    except CPE.ReadinessError as exc:
        return False, str(exc)
    try:
        record = CPE.load_record(REPO_ROOT, run_id=run_id, stage_id=RUN_STAGE_ID)
    except FileNotFoundError:
        return False, (
            f"{record_rel} does not exist: no pod-like sweep has been recorded "
            "for this run. Take one on the clean grant-containing tree with "
            "`record_pod_environment.py --experiment "
            f"{CPE.EXPERIMENT_ID} --kind {LAUNCH_BOUND} --run-id {run_id} "
            f"--stage-id {RUN_STAGE_ID}`.")
    except Exception as exc:                                    # noqa: BLE001
        return False, f"cannot read {record_rel}: {exc}"

    try:
        #: Derived from THIS session's own manifest, under THIS session's id.
        live_staging = derive_contract(
            spec(ctx.args).setup, session_id=BC.SESSION_ID)["digest"]
    except Exception as exc:                                    # noqa: BLE001
        return False, f"cannot derive this session's staging contract: {exc}"

    ok, reason = CPE.verify_record(
        record, REPO_ROOT, run_id=run_id, stage_id=RUN_STAGE_ID,
        session_commit=getattr(ctx.args, "session_commit", None),
        authorization_path=auth_path_for(run_id),
        required_kind=LAUNCH_BOUND,
        staging_contract_digest=live_staging)
    ctx.evidence["pod_environment_verification"] = {
        "verdict": "PASS" if ok else "FAIL",
        "record": record_rel,
        "record_self_sha256": record.get("self_sha256"),
        "record_schema": record.get("schema"),
        "required_schema": CPE.SCHEMA,
        "record_kind": record.get("record_kind"),
        "required_record_kind": LAUNCH_BOUND,
        "experiment": CPE.EXPERIMENT_ID,
        "session_commit": getattr(ctx.args, "session_commit", None),
        "permitted_post_sweep_paths": [record_rel, auth_path_for(run_id)],
        "live_staging_contract_digest": live_staging,
        "recorded_staging_contract_digest": record.get("staging_contract_digest"),
        "live_completion_harness_digest": CPE.harness_digest(REPO_ROOT),
        "recorded_completion_harness_digest": record.get(
            "completion_harness_digest"),
    }
    return ok, reason


def bundle_record_for(run_id: str) -> str:
    """Where this run's staged-bundle record lives."""
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
            f"/{COMPLETION_RUN_ROLES['bundle_record']}")


def bundle_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Can a pod, RIGHT NOW, obtain the exact authorized code?

    The pod's setup downloads the relay object and checks out from it before any
    scientific stage, so every other gate verifies the CONTENTS of a commit and
    this one asks whether the pod can reach it at all. C1 attempt 1 answered
    every other question correctly and died at `SETUP_RC=1` fetching an alias
    for nothing.

    Read-only: it uploads nothing. Preparation is a separate command, so what
    this verifies is the relay's state rather than a side effect of its own
    verification. The round-trip is the generic one, driven by the COMPLETION's
    transport spec -- so the authorization it looks for inside the bundle is
    this session's, and the digest it requires is the completion closure's.
    """
    run_id = getattr(ctx.args, "run_id", None)
    commit = ctx.args.session_commit
    try:
        BCT.require_canonical_bundle_arg(ctx.args.bundle, commit)
    except BCT.BundleTransportError as exc:
        return False, str(exc)

    bundle_rel = bundle_record_for(run_id)
    staged = REPO_ROOT / bundle_rel
    if not staged.is_file():
        return False, (f"{bundle_rel} is missing; stage the canonical bundle for "
                       f"{commit[:12]}… first")
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
            evidence = BCT.roundtrip(
                session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=auth_file.read_bytes(),
                authorization_path=auth_rel,
                #: The AUTHORIZED pair, not the live one.
                #: `completion_executable_gate` has already required the two to
                #: agree; asking the round-trip about the live digest instead
                #: would make a stale authorization unfalsifiable here.
                expected_harness_digest=ctx.auth.harness_source_digest,
                harness_files=tuple(ctx.auth.harness_source_files),
                workdir=Path(tmp))
    except Exception as exc:                                    # noqa: BLE001
        return False, f"the pod could not obtain the authorized commit: {exc}"

    ctx.evidence["bundle_staged_check"] = evidence
    return True, (f"{evidence['canonical_bundle_name']} ({evidence['bytes']} "
                  f"bytes, {evidence['remote_sha256'][:12]}…) round-trips to "
                  f"{evidence['roundtrip_head'][:12]}… carrying this "
                  f"authorization and executable set "
                  f"{evidence['roundtrip_harness_digest'][:12]}…")


def driver_command(ctx: SessionContext, plan) -> str:
    """The driver invocation. Every identity it needs, it BINDS itself."""
    def floor2(value: float) -> str:
        #: FLOORED. A limit handed downward rounds DOWN or it is not a limit.
        return f"{math.floor(value * 100) / 100:.2f}"

    return (f"{POD_IMAGE['remote_python']} "
            f"scripts/pod/autoinit_phase_c2_baseline_driver.py "
            f"--protocol {BC.PROTOCOL} "
            f"--frozen-inputs {BC.FROZEN_INPUTS} "
            f"--selection-record {BC.SELECTION_RECORD} "
            f"--rebuild-minutes {BC.rebuild_minutes(REPO_ROOT):.3f} "
            f"--rate {ctx.price} --spent-usd {ctx.spent_usd:.3f} "
            f"--soft-stop-usd {floor2(plan.soft_stop_usd)} "
            f"--device cuda")


def spec(args) -> SessionSpec:
    """The whole session, in one object."""
    return SessionSpec(
        session_id=BC.SESSION_ID,
        schema="aadistill.autoinit.c2_baseline_completion_session/v1",
        description=(
            "Phase C2 baseline completion: rebuild the frozen C1 treatment "
            "baseline B once through its complete deterministic fixed path, "
            "measure it once on the frozen state_eval suite, and compute the "
            "preregistered B->C comparison against the five candidate "
            "measurements Attempt 4 froze. No beam search, no new candidate, no "
            "candidate remeasured. Trains nothing."),
        authorization_path=auth_path_for(getattr(args, "run_id", "")),
        authorization_loader=BC.BaselineCompletionAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=BC.PLAN_ID,
        plan_hash=BC.plan_hash(REPO_ROOT),
        budget=BC.budget_spec(REPO_ROOT),
        setup=SetupManifest(
            #: `calib.domain_balanced@v1`'s items. NOT `CANONICAL_INIT`: that is
            #: Search-1's measured control and this session has no control.
            relay_inputs=CALIBRATION_V1,
            local_assets=LOCAL_ASSETS,
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "SESSION_FROZEN_EXPECT",
                          "SESSION_SETUP_MARKERS", "TEACHER_REVISION"),
            #: VLLM_READY is ABSENT: this session serves nothing and builds no
            #: inference environment. The declaration is what the shared setup
            #: script reads to decide which optional sections run, so omitting
            #: the marker is what stops the work rather than a comment saying it
            #: is unnecessary.
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "TEACHER_READY",
                           "ROPE_OK", "TESTS_OK", "AUTHORIZATION_OK",
                           "SETUP_DONE"),
            env={"SESSION_KIND": "c2_baseline_completion",
                 "SESSION_FROZEN_EXPECT": FROZEN_EXPECT},
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_phase_c2_baseline_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="BASELINE_COMPLETION_ALL_DONE",
            failure=("BASELINE_COMPLETION_FAILED",),
            incomplete=(),
            #: One 2.22 GiB checkpoint is rebuilt and it is re-derivable from
            #: the frozen path; what must survive is its MEASUREMENT, which the
            #: comparison record carries. So there are no products to fetch.
            products_eligible=lambda terminal, stages: False,
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. Nothing was trained, no candidate was "
                          "remeasured and no committed record was replaced.")),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="c2_baseline_completion_evidence.json",
            archive_basename="c2_baseline_completion_artifacts.tar.gz",
            spec_success="configs/autoinit/c2_baseline_completion_artifacts.json",
            spec_failed=("configs/autoinit/"
                         "c2_baseline_completion_artifacts_failed.json"),
            report_names=("c2_baseline_completion_evidence.json",
                          "c2_baseline_comparison.json")),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
        precheck=(
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", "")),
                                check_lineage=True),
            completion_executable_gate,
            completion_scope_gate,
            frozen_runtime_gate,
            frozen_inputs_gate,
            frozen_assets_gate,
            pricing_and_plan_gate,
            storage_gate,
            readiness_gate,
            #: LAST, because it is the only gate that touches the network, and
            #: everything it would verify against must already be checked.
            bundle_staged_gate,
        ),
    )


#: role -> path within the run. The same five areas every run uses.
COMPLETION_RUN_ROLES: dict[str, str] = {
    #: --- governance: inputs, prepared before the run opens -----------------
    "grant": "governance/grant.json",
    "readiness_record": "governance/readiness.json",
    "authorization": "governance/authorization.json",
    "bundle_record": "governance/bundle.json",
    #: --- runtime: how it executed -------------------------------------------
    #: Written on EVERY path including a $0 pre-provider refusal, which is why
    #: it is the one role the manifest requires.
    "session_record": "runtime/session.json",
    "launcher_log": "runtime/launcher.log",
    "watchdog_journal": "runtime/watchdog",
    #: --- evidence: what it produced ------------------------------------------
    "driver_log": "evidence/driver_run.log",
    "driver_status": "evidence/driver_status.txt",
    "session_evidence": "evidence/c2_baseline_completion_evidence.json",
    #: The point of the session: B's measurement and the comparison against the
    #: five frozen candidates.
    "baseline_comparison": "evidence/c2_baseline_comparison.json",
    #: --- artifacts / closeout -------------------------------------------------
    "artifact_manifest": "artifacts/manifest.json",
    "outcome": "closeout/outcome.json",
}

COMPLETION_RUN_SPEC = RunArtifactSpec(
    spec_id="phase_c2_baseline_completion_session_v1",
    required=("session_record",),
    optional=tuple(r for r in COMPLETION_RUN_ROLES if r != "session_record"))

#: Exempt from `open_run`'s occupancy rule and from nothing else: these four are
#: committed BEFORE the run opens, in the order grant -> readiness ->
#: authorization -> bundle, and a launcher that refused to open a run because
#: its own inputs were already there could never start.
_RUN_PREPARED: tuple[str, ...] = ("grant", "readiness_record", "authorization",
                                  "bundle_record")


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test can assert on the namespace it
    produces rather than on a transcription of it.

    It must supply every name in `RUNNER_ARGUMENT_CONTRACT`:
    `SessionRunner.__init__` reads those eighteen attributes and refuses a
    namespace missing any of them -- deterministically, before provider
    creation, which is the cheap place but still a wasted invocation.

    The operational defaults are Search-1's, because they are the ones this
    image and this provider have been observed under. Three differ, and each
    for a stated reason about THIS workload: a smaller volume, one host draw,
    and a poll limit sized to a one-hour session rather than a ten-hour one.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    #: REQUIRED, and it is what produces `out`. Every artifact this session
    #: consumes or produces -- grant, readiness record, authorization, bundle
    #: record, session record, evidence, comparison, manifest -- is resolved
    #: from this id.
    ap.add_argument("--run-id", required=True, action=_RunIdSetsOut,
                    help="the attempt this session runs as, e.g. attempt5")
    ap.add_argument("--relay-repo", default=BCT.RELAY_REPO)
    #: The accepted C2 image. Same image the frozen candidates were measured
    #: under, which is part of what makes the two halves comparable.
    ap.add_argument("--image", default=BC.image_name(REPO_ROOT))
    ap.add_argument("--gpu", default=BC.gpu_class(REPO_ROOT))
    #: From the completion pricing record, so there is one hand-maintained
    #: price and it is the one the pricing gate verifies. A stale value here can
    #: only ever refuse a launch, never buy one.
    ap.add_argument("--max-price", type=float,
                    default=BC.price_per_hour_usd(REPO_ROOT))
    #: THE storage value. The `$0` gate reads the same attribute.
    ap.add_argument("--disk-gb", type=int, default=COMPLETION_PROVISION_GIB)
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    ap.add_argument("--create-attempts", type=int, default=8)
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    #: ONE. Search-1 permitted three host draws because losing a ten-hour beam
    #: to an unusable host is expensive; a one-hour session that loses its host
    #: is cheap to relaunch, and the grant owns the real limit either way --
    #: `completion_scope_gate` enforces what the authorization permits, not this
    #: default.
    ap.add_argument("--host-draws", type=int, default=1)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    #: Sized to THIS session's envelope plus slack, not Search-1's. A poll limit
    #: below the priced ceiling would stop watching a pod that is still billing;
    #: one far above it just costs nothing.
    ap.add_argument("--poll-limit-min", type=float, default=180.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    #: No `--out`. It is `run_id` and the layout, or it is nothing: a flag that
    #: could redirect one attempt's session record into another run, or into an
    #: arbitrary repository path, is a flag that can destroy the evidence of
    #: what happened.
    return ap


def close_completion_run(layout, args):
    """Copy what the session produced into the run, then record it.

    Small because the session is small: one evidence document, one comparison
    record, the two logs and whatever watchdog journals its resources wrote.
    """
    import shutil

    scr = Path(args.scr)
    for source, role in (("launch.log", "launcher_log"),
                         (f"relay/{Path(RUN_LOG).name}", "driver_log"),
                         (f"relay/{Path(STATUS).name}", "driver_status"),
                         ("relay/c2_baseline_completion_evidence.json",
                          "session_evidence"),
                         ("store/manifest.json", "artifact_manifest")):
        candidate = scr / source
        if candidate.is_file():
            destination = layout.path(COMPLETION_RUN_ROLES[role])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)

    #: The comparison, wherever the collector left it in the extracted archive.
    for found in sorted(scr.glob("store/extracted/**/c2_baseline_comparison.json")):
        destination = layout.path(COMPLETION_RUN_ROLES["baseline_comparison"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, destination)
        break

    journals = layout.path(COMPLETION_RUN_ROLES["watchdog_journal"])
    journals.mkdir(parents=True, exist_ok=True)
    for found in (sorted(scr.glob("watchdog_*.jsonl"))
                  + sorted(scr.glob("watchdog_*.out"))):
        shutil.copy2(found, journals / found.name)

    #: `args.out` is derived from `--run-id` by the parser's action and is
    #: always inside this run. Resolving it again from the layout would be a
    #: second derivation of one path.
    record_path = REPO_ROOT / args.out
    session = json.loads(record_path.read_text()) if record_path.is_file() else {}
    return record_run(
        layout, spec=COMPLETION_RUN_SPEC,
        plan={"session_id": session.get("session_id"),
              "plan_hash": session.get("session_plan_hash"),
              "session_commit": getattr(args, "session_commit", None),
              "bundle": getattr(args, "bundle", None),
              "scratch_root": str(scr)},
        implementation={
            "launcher": "scripts/pod/autoinit_phase_c2_baseline_launch.py",
            "driver": "scripts/pod/autoinit_phase_c2_baseline_driver.py",
            "harness_source_digest": session.get("harness_source_digest"),
            "authorization": auth_path_for(args.run_id)},
        status={"passed": session.get("passed"),
                "terminal": session.get("terminal"),
                "measures_b_once": True,
                "runs_a_beam_search": False,
                "remeasures_any_candidate": False})


def main() -> int:
    args = build_parser().parse_args()
    if args.max_price is None:
        args.max_price = BC.price_per_hour_usd(REPO_ROOT)
    #: BEFORE anything is priced or created: a colliding run id or a foreign
    #: scratch root costs $0 here.
    claim_output_root(args.scr, RUN_EXPERIMENT_ID, args.run_id, RUN_STAGE_ID)
    layout = open_run(REPO_ROOT, RUN_EXPERIMENT_ID, args.run_id,
                      roles=COMPLETION_RUN_ROLES, prepared=_RUN_PREPARED,
                      stage_id=RUN_STAGE_ID)
    write_run_readmes(layout, RUN_EXPERIMENT_ID, args.run_id, RUN_STAGE_ID)
    #: `args.out` was set by `--run-id`'s action, so the run the layout opened
    #: and the path the runner writes cannot disagree. Asserted rather than
    #: assigned: filling it in here would mean the parser's namespace was
    #: incomplete, which is the state the argument contract exists to refuse.
    assert args.out == session_record_path(args.run_id), (args.out, args.run_id)
    rc = run_session(spec(args), args, REPO_ROOT,
                     summary=("baseline completion is a terminus: it measures B "
                              "once and computes the comparison. Search-2 and "
                              "behavioural confirmation are separately "
                              "authorized and unreachable from here."))
    try:
        doc = close_completion_run(layout, args)
    except Exception as exc:                                      # noqa: BLE001
        print(f"\nRUN NOT RECORDED: {type(exc).__name__}: {exc}\n"
              f"  the run directory is "
              f"{rel_run_dir(RUN_EXPERIMENT_ID, args.run_id, RUN_STAGE_ID)}; it "
              "holds whatever the session produced and has no manifest. Do not "
              "reuse this run id.")
        return rc or 1
    print(f"run {doc['experiment_id']}/{doc['run_id']} recorded — "
          f"{len(doc['roles'])} role(s) under "
          f"{rel_run_dir(RUN_EXPERIMENT_ID, args.run_id, RUN_STAGE_ID)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
