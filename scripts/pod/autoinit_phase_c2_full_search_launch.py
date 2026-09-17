#!/usr/bin/env python3
"""Launch ONE Phase-C2 full-joint-re-search session through the generic runner.

    PYTHONPATH=src python scripts/pod/autoinit_phase_c2_full_search_launch.py \
        --scr /path/to/scratch --session-commit <sha> --bundle <name> \
        --run-id attempt1

**A beam, so it declares what a beam needs.** Search-1's launcher is the closest
precedent and this follows it: a re-quoted rate, a storage provision sized to
resident search states, a long poll limit and the same staged assets. What
differs is the SPACE — Search-1 fixed three operators and varied one, this
searches all four jointly with calibration unpinned — and the storage, which is
derived for this space rather than inherited from that one.

**The storage number is the one that can lose everything.** Search-1 provisions
for a 87.4 GiB peak: five retained level-0 states expanded into eighteen
children. The joint space is a different shape, because every operator kind may
go first: level 0 generates eleven children of which nine continue, and level 1
expands those nine into SIXTY. `full_search.peak_resident_gib` derives 243.4
GiB, at level 1 — nearly three times Search-1's. Carrying 87.4 across would
under-provision a search that then fills its volume mid-level and loses every
state it has measured. The derivation cross-checks against reality: it puts a
finished leaf at 1.11 GiB, and the CUDA engineering validation weighed the real
596M student at exactly that.

**A grant for this session cannot buy anything else.** The authorization is a
distinct type with a distinct schema reporting `authorizes_c2_search1 = False`,
`authorizes_c2_baseline_completion = False` and
`authorizes_behavioural_selection = False`, and `FullSearchAuthorization.load`
refuses an artifact claiming otherwise. The other C2 loaders refuse this schema
symmetrically.

**IT ENDS AT A COMMITTED TOP-5.** `commit_top_k` is the last authorized stage.
Nothing here imports a recovery or scoring path, and the behavioural session's
candidate identities do not exist until this search succeeds.

THIS FILE AUTHORIZES NOTHING. It refuses to run without a committed
authorization, and no full-search authorization has been issued.
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
    ArtifactPolicy, ExecutionCommands, MarkerPolicy, SessionContext,
    SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate)
from aadistill.infrastructure.session_runner import run_session  # noqa: E402
from aadistill.runtime.staging_contract import (  # noqa: E402
    derive_contract, ignores_for_selection)
from autoinit_science_inputs import CALIBRATION_V1  # noqa: E402
from experiments.deployment import POD_IMAGE, deployment_commands  # noqa: E402
from experiments.phase_c2 import full_search as FSG  # noqa: E402
from experiments.phase_c2 import full_search_bundle as FST  # noqa: E402
#: The FULL SEARCH's readiness instance, not the generic runtime module. The
#: generic module owns the mechanism; which experiment, which schema, which
#: harness and which staging contract are this experiment's own facts.
from experiments.phase_c2 import (  # noqa: E402
    full_search_pod_environment as FPE)
from experiments.run_layout import (  # noqa: E402
    ArtifactSpec as RunArtifactSpec, claim_output_root, open_run,
    record_run, rel_run_dir, write_run_readmes,
)
from phase_a_frozen import TEACHER_REVISION  # noqa: E402

WS = "/workspace"
STATUS = f"{WS}/autoinit_phase_c2_full_search.status"
RUN_LOG = f"{WS}/autoinit_phase_c2_full_search_run.log"

AUDIT_DIRNAME = "autoinit_phase_c2_full_search"

RUN_EXPERIMENT_ID = "phase_c2_full_search"
RUN_STAGE_ID = json.loads(
    (REPO_ROOT / "configs/experiments/phase_c2/full_search_authorization.json"
     ).read_text())["stage_id"]

#: The frozen-asset expectation. The same document Search-1 and the completion
#: named, because the asset is the same one and there is exactly one declaration.
FROZEN_EXPECT = "configs/experiments/phase_c2/frozen_assets.json"

#: The image both halves of the pooled cost table were measured under --
#: Phase-B attempt 5 and C2 attempt 4, whose recorded digest was
#: `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@580.126.09`. The tag is
#: what a launch can request; the driver digest is what the pod reports back.
BOUND_IMAGE = "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404"

#: DERIVED, never listed. Two paid subruns of the CUDA engineering validation
#: died one per producer of non-source input because a hand-maintained shipping
#: list named only the producer its author had thought of. `staged_assets` asks
#: each calibration profile where its items are and the frozen-asset
#: expectation where the suite is, so a reweighted mixture or a third profile is
#: staged without anyone remembering to add it.
LOCAL_ASSETS = FSG.staged_assets(REPO_ROOT)

#: Search-1's selection, deliberately: both sessions stage the same assets and
#: run the same beam machinery against the same teacher, so what a pod can fail
#: before the search starts is the same set of things. See
#: `full_search_pod_environment` for why a second selection would be two
#: answers to one question.
POD_TEST_SELECTION = FPE.POD_TEST_SELECTION
TEST_IGNORES = ignores_for_selection(POD_TEST_SELECTION, REPO_ROOT)

#: DERIVED from the real space at the standing width, not inherited. See the
#: module docstring for why Search-1's 87.4 GiB is the wrong number here.
_STORAGE = FSG.peak_resident_gib(REPO_ROOT)
PEAK_WORKING_GIB = float(_STORAGE["peak_resident_gib"])
#: The teacher in bf16, the checkout, the venv and the staged assets sit on top
#: of the resident search states, and a beam that fills its volume mid-level
#: loses every state it has measured. Disk is cents; the provision is generous
#: on purpose.
TEACHER_AND_ENVIRONMENT_GIB = 25.0
FULL_SEARCH_PROVISION_GIB = int(
    math.ceil((PEAK_WORKING_GIB + TEACHER_AND_ENVIRONMENT_GIB) * 1.25 / 50) * 50)


def session_record_path(run_id: str) -> str:
    """Where THIS run's session record goes, repository-relative.

    ONE rule, called by the parser's `--run-id` action and again by `main`, so
    the path the runner writes and the directory the run was opened in cannot
    disagree.
    """
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
            f"/{FULL_SEARCH_RUN_ROLES['session_record']}")


class _RunIdSetsOut(argparse.Action):
    """`--run-id` also produces `out`, because the RUNNER reads `out`.

    `SessionRunner.save()` writes `args.out`, and the argument contract requires
    every attribute the runner reads to come from the REAL parser -- device
    canary attempt 1 died at $0.0603 on an attribute a hand-written namespace
    had and the parser did not, after the pod was billing.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, value)
        namespace.out = session_record_path(value)


def auth_path_for(run_id: str) -> str:
    """Where THIS run's authorization lives, repository-relative."""
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
            "/governance/authorization.json")


def bundle_record_for(run_id: str) -> str:
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
            "/governance/bundle.json")


# --- the $0 gates -----------------------------------------------------------
#
# Each one names a failure THIS session can have, before a pod exists.


def full_search_executable_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Re-derive the full-search closure, independently of the artifact.

    Every other check digests the file list the authorization STORES. This one
    derives the live set and requires the artifact to declare exactly it, so an
    authorization carrying Search-1's or the completion's list -- which would
    verify perfectly against their files -- cannot leave this driver and this
    space derivation unmeasured.
    """
    try:
        live = FSG.current_executable(REPO_ROOT)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"cannot derive the full-search executable set: {exc}"
    expected = tuple(row["path"] for row in live["files"])
    declared = tuple(getattr(ctx.auth, "harness_source_files", ()) or ())
    if declared != expected:
        only_declared = sorted(set(declared) - set(expected))
        only_live = sorted(set(expected) - set(declared))
        return False, (
            f"the authorization declares {len(declared)} executable file(s) and "
            f"the live closure derives {len(expected)}: "
            f"declared-only {only_declared[:5]}, live-only {only_live[:5]}")
    return True, (f"full-search closure {live['digest'][:12]}… over "
                  f"{len(expected)} derived files")


def full_search_scope_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The authorization must permit THIS search and refuse everything else."""
    auth = ctx.auth
    if not getattr(auth, "authorizes_c2_full_search", False):
        return False, "this authorization does not permit the full joint search"
    for forbidden, what in (
            ("authorizes_c2_search1", "Search-1's consumed beam"),
            ("authorizes_c2_baseline_completion", "a baseline rebuild"),
            ("authorizes_behavioural_selection", "behavioural selection"),
            ("allows_recovery_training", "recovery training")):
        if getattr(auth, forbidden, False):
            return False, (f"this authorization claims {forbidden}, i.e. that it "
                           f"can buy {what}. A full-search artifact cannot.")
    stages = tuple(getattr(auth, "authorized_stages", ()) or ())
    if stages != FSG.AUTHORIZED_STAGES:
        return False, (f"the authorization authorizes {stages}, not "
                       f"{FSG.AUTHORIZED_STAGES}. The sequence is the scope, and "
                       "a stage after commit_top_k is the behavioural session.")
    scope = getattr(auth, "resource_scope", None)
    if scope is None:
        return False, ("the authorization carries no resource scope, so how many "
                       "provider resources it permits is stated nowhere a gate "
                       "can read")
    run_id = getattr(ctx.args, "run_id", None)
    if getattr(scope, "run_id", None) != run_id:
        return False, (f"the resource scope names run {getattr(scope, 'run_id', None)!r} "
                       f"and this launch is {run_id!r}. A one-use authorization "
                       "is one attempt's.")
    return True, (f"authorizes the full joint search for {run_id}, stages "
                  f"{' -> '.join(stages)}, and refuses Search-1, a rebuild and "
                  "behavioural selection")


def frozen_space_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The SPACE the authorization was issued against is the live one.

    This is the gate that has no counterpart in the completion session, and it
    is the one that distinguishes this experiment. A newly registered operator,
    a withdrawn exclusion or a third calibration mixture changes the space the
    beam explores -- so a session whose authorization bound 578 leaves must not
    quietly search 866.
    """
    from experiments.phase_c2 import full_search_space as FS

    bound = getattr(ctx.auth, "bound", None)
    if not isinstance(bound, dict):
        #: `bound` lives on the artifact rather than the dataclass; read it from
        #: the committed JSON so the gate does not depend on loader internals.
        auth_file = REPO_ROOT / auth_path_for(getattr(ctx.args, "run_id", ""))
        if not auth_file.is_file():
            return False, f"{auth_path_for(getattr(ctx.args, 'run_id', ''))} is missing"
        bound = (json.loads(auth_file.read_text()).get("bound") or {})
    try:
        FS.register_c2_operators()
        report = FS.size_report(REPO_ROOT)["full_joint"]
        space = FS.full_joint_space(REPO_ROOT)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"cannot derive the live joint space: {exc}"

    checks = {
        "joint_space_total_leaves": int(report["total_leaves"]),
        "joint_space_decomposed_leaves": int(report["decomposed_leaves"]),
        "joint_space_allowed_impls": sorted(space.allowed_impls),
        "joint_space_impl_profiles_are_unpinned": space.impl_profiles is None,
        "excluded_implementations": sorted(FS.EXCLUSIONS),
        "beam_width": FSG.standing_beam_width(),
        "beam_warmup_levels": int(FS.SCHEDULE_V1.warmup_levels),
    }
    wrong = {k: {"authorization": bound.get(k), "live": v}
             for k, v in checks.items() if bound.get(k) != v}
    if wrong:
        return False, ("the authorization was issued against a different search "
                       "space than this tree derives: " + json.dumps(wrong))
    if not checks["joint_space_impl_profiles_are_unpinned"]:
        return False, ("the live space PINS implementations to profiles, which "
                       "is Search-1's restriction. The joint re-search leaves "
                       "calibration free to compete.")
    return True, (f"joint space {checks['joint_space_total_leaves']} leaves "
                  f"({checks['joint_space_decomposed_leaves']} decomposed) over "
                  f"{len(checks['joint_space_allowed_impls'])} impls, profiles "
                  f"unpinned, beam {checks['beam_width']}/"
                  f"{checks['beam_warmup_levels']}")


def frozen_runtime_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The runtime the authorization was priced against, not merely defaults.

    The rate is the part that moves. The authorization's ceiling was DERIVED
    from the pricing record's minutes at the rate its grant quoted, so a launch
    that raised `--max-price` above that rate would be spending at a rate the
    ceiling does not cover. A LOWER max price is safe and permitted: it can only
    ever refuse a launch.
    """
    auth_file = REPO_ROOT / auth_path_for(getattr(ctx.args, "run_id", ""))
    if not auth_file.is_file():
        return False, "the authorization is missing, so its rate is unknown"
    money = json.loads(auth_file.read_text()).get("approved_money") or {}
    approved = money.get("price_basis_usd_per_hour")
    if approved is None:
        return False, ("the authorization states no approved rate, so the "
                       "ceiling cannot be checked against what a launch pays")
    approved = round(float(approved), 4)
    requested = round(float(ctx.args.max_price), 4)
    if requested > approved + 1e-9:
        return False, (f"--max-price ${requested:.4f}/h exceeds the ${approved:.4f}/h "
                       "the authorization's ceiling was derived at. A higher "
                       "rate needs the ceiling re-derived and a new grant; it "
                       "is not a launcher flag, and beam width 6 is not "
                       "narrowed to absorb it.")
    if ctx.args.gpu != "NVIDIA L40S":
        return False, (f"--gpu {ctx.args.gpu!r} is not the priced NVIDIA L40S. "
                       "The cost table was pooled from L40S expansions and the "
                       "ceiling is minutes on that card.")
    if ctx.args.image != BOUND_IMAGE:
        return False, (f"--image {ctx.args.image!r} is not {BOUND_IMAGE!r}, "
                       "the image both halves of the pooled cost table were "
                       "measured under. A different image is a different "
                       "numerical runtime, and the ceiling is minutes "
                       "observed on this one.")
    return True, (f"runtime bound: {ctx.args.gpu}, max price "
                  f"${requested:.4f}/h <= the authorized ${approved:.4f}/h")


def frozen_assets_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The suite every candidate is ranked on, checked by the real verifier."""
    import subprocess
    #: This gate runs on the DEV BOX, before a pod exists, so it uses this
    #: interpreter. The pod re-asks the same question through the setup script's
    #: ASSETS_READY step against the same expectation document.
    result = subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "scripts/autoinit/verify_frozen_assets.py"),
         "--expect", FROZEN_EXPECT],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        env={"PYTHONPATH": f"{REPO_ROOT}/src:{REPO_ROOT}/scripts",
             "PATH": "/usr/bin:/bin"})
    if result.returncode != 0:
        return False, (f"the frozen-asset expectation does not verify: "
                       f"{(result.stdout + result.stderr).strip()[-400:]}")
    return True, f"frozen-asset expectation {FROZEN_EXPECT} verifies"


def staged_inputs_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Every non-source input the search reads is obtainable. $0.

    The gate the CUDA engineering validation paid twice to learn. Its first
    subrun reached a real L40S, passed every capability check and died in stage
    A because nothing under `logs/` had been shipped; its second got stage A to
    pass and died in stage B because nothing under `artifacts/` had. Both are
    absence-of-input failures that cost real money to discover on a pod, and
    both are answerable here for nothing.

    Derived on both halves, so a new telemetry source or a reweighted mixture is
    covered without anyone remembering to add it.
    """
    missing_tracked = [rel for rel in FSG.tracked_non_source_inputs()
                       if not (REPO_ROOT / rel).is_file()]
    if missing_tracked:
        return False, (
            f"{len(missing_tracked)} tracked input(s) the cost model pools over "
            f"are absent: {missing_tracked}. The search refuses to price from a "
            "partial history, so the session would die in stage A.")
    missing_assets = [a.repo_path for a in LOCAL_ASSETS
                      if not (REPO_ROOT / a.repo_path).is_dir()]
    if missing_assets:
        return False, (
            f"{len(missing_assets)} staged asset(s) are absent from this dev "
            f"box: {missing_assets}. They are gitignored, so the bundle cannot "
            "carry them and the pod would die resolving the calibration "
            "mixtures in stage B.")
    return True, (f"{len(FSG.tracked_non_source_inputs())} tracked input(s) "
                  f"present and {len(LOCAL_ASSETS)} asset(s) stageable: "
                  f"{', '.join(a.dest_name for a in LOCAL_ASSETS)}")


def pricing_and_plan_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The authorization's ceiling and plan must be the documents' own.

    The ceiling is checked against the DERIVATION at the authorization's own
    rate, not against the pricing record's printed figure: the record was priced
    at a basis and the artifact was issued at a re-quoted rate, so comparing to
    the printed number would refuse every launch whose rate had legitimately
    moved.
    """
    auth_file = REPO_ROOT / auth_path_for(getattr(ctx.args, "run_id", ""))
    if not auth_file.is_file():
        return False, "the authorization is missing"
    doc = json.loads(auth_file.read_text())
    money = doc.get("approved_money") or {}
    try:
        plan = FSG.plan_hash(REPO_ROOT)
        rate = round(float(money["price_basis_usd_per_hour"]), 4)
        expected_ceiling = FSG.derive_ceiling_usd(rate, REPO_ROOT)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"the full-search pricing or protocol does not verify: {exc}"
    declared_cap = float(getattr(ctx.auth, "hard_cap_usd", 0.0) or 0.0)
    if abs(declared_cap - expected_ceiling) > 1e-9:
        return False, (f"the authorization caps ${declared_cap:.4f} and the "
                       f"pricing record's minutes at its own ${rate}/h derive "
                       f"${expected_ceiling:.4f}")
    declared_plan = getattr(ctx.auth, "plan_hash", None)
    if declared_plan and declared_plan != plan:
        return False, (f"the authorization binds plan {declared_plan} and the "
                       f"live protocol hashes to {plan}")
    return True, (f"ceiling ${expected_ceiling:.4f} is the pricing record's "
                  f"minutes at ${rate}/h; plan {plan[:12]}…")


def storage_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Refuse an under-provisioned volume before a pod exists.

    Reads `disk_gb` -- the same attribute provider creation reads. A gate that
    checked its own flag would be a gate that can pass while the pod is
    provisioned from a different number.
    """
    requested = int(getattr(ctx.args, "disk_gb", 0) or 0)
    if requested < FULL_SEARCH_PROVISION_GIB:
        return False, (
            f"--disk-gb {requested} is below the full-search provision of "
            f"{FULL_SEARCH_PROVISION_GIB} GiB. The beam's peak working set is "
            f"{PEAK_WORKING_GIB:.1f} GiB of resident states, reached at level "
            f"{_STORAGE['peak_at_level']} where "
            f"{_STORAGE['levels'][_STORAGE['peak_at_level']]['parents']} "
            "retained parents are expanded into "
            f"{_STORAGE['levels'][_STORAGE['peak_at_level']]['generated']} "
            "children, plus the teacher, the checkout and the venv. A volume "
            "that fills there loses every state measured up to that point -- "
            "and this space is nearly three times Search-1's peak, so its "
            "200 GiB is not enough.")
    return True, (f"volume {requested} GiB >= {FULL_SEARCH_PROVISION_GIB} for a "
                  f"{PEAK_WORKING_GIB:.1f} GiB peak working set at level "
                  f"{_STORAGE['peak_at_level']}")


def readiness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A launch-bound sweep taken for THIS run, of THIS session."""
    run_id = getattr(ctx.args, "run_id", None)
    try:
        record = FPE.load_record(REPO_ROOT, run_id=run_id, stage_id=RUN_STAGE_ID)
    except FileNotFoundError:
        return False, (f"{FPE.record_path_for(run_id, RUN_STAGE_ID)} does not "
                       "exist; a launch-bound sweep is owed for this run")
    if record.get("kind") != FPE.LAUNCH_BOUND:
        return False, (f"the readiness record is {record.get('kind')!r}, not "
                       f"{FPE.LAUNCH_BOUND!r}. Only a launch-bound record "
                       "describes the tree a launch will use.")
    try:
        FPE.verify_record(record, REPO_ROOT, run_id=run_id,
                          stage_id=RUN_STAGE_ID,
                          session_commit=ctx.args.session_commit)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"the readiness record does not describe this tree: {exc}"
    return True, (f"launch-bound readiness at {record.get('commit', '?')[:12]}… "
                  f"binds harness {str(record.get('full_search_harness_digest'))[:12]}…")


def bundle_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Can a pod, RIGHT NOW, obtain the exact authorized code?

    The pod's setup downloads the relay object and checks out from it before any
    scientific stage, so every other gate verifies the CONTENTS of a commit and
    this one asks whether the pod can reach it at all. C1 attempt 1 answered
    every other question correctly and died at `SETUP_RC=1` fetching an alias
    for nothing.

    Read-only: it uploads nothing. LAST, because it is the only gate that
    touches the network and everything it verifies against must already be
    checked.
    """
    run_id = getattr(ctx.args, "run_id", None)
    commit = ctx.args.session_commit
    try:
        FST.require_canonical_bundle_arg(ctx.args.bundle, commit)
    except FST.BundleTransportError as exc:
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
            evidence = FST.roundtrip(
                session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=auth_file.read_bytes(),
                authorization_path=auth_rel,
                #: The AUTHORIZED pair, not the live one.
                #: `full_search_executable_gate` has already required the two to
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

    row = FSG._standing_row(REPO_ROOT)
    return (f"{POD_IMAGE['remote_python']} "
            f"scripts/pod/autoinit_phase_c2_full_search_driver.py "
            f"--authorization-path {auth_path_for(ctx.args.run_id)} "
            f"--image-digest {getattr(ctx.args, 'image', '')} "
            f"--rate {ctx.price} --spent-usd {ctx.spent_usd:.3f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f} "
            f"--soft-stop-usd {floor2(plan.soft_stop_usd)} "
            #: The beam's ENVELOPE and its CLOCK, both from the pricing record's
            #: row for the standing width. The driver defaults neither: a driver
            #: that invented its own budget is one that can outrun the plan that
            #: funded it.
            f"--search-minutes {float(row['hard_ceiling_minutes']):.2f} "
            f"--search-deadline-minutes {float(row['hard_ceiling_minutes']):.2f} "
            f"--top-n 5 "
            f"--device cuda")


#: role -> path within the run. The same five areas every run uses.
FULL_SEARCH_RUN_ROLES: dict[str, str] = {
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
    "session_evidence": "evidence/c2_full_search_evidence.json",
    #: The point of the session: the frozen Top-5 and the search journal that
    #: shows how the beam reached it.
    "stage1_selection": "evidence/stage1_selection.json",
    "search_journal": "evidence/search_telemetry.jsonl",
    "search_result": "evidence/search_result.json",
    #: --- artifacts / closeout -------------------------------------------------
    "artifact_manifest": "artifacts/manifest.json",
    "outcome": "closeout/outcome.json",
}

FULL_SEARCH_RUN_SPEC = RunArtifactSpec(
    spec_id="phase_c2_full_search_session_v1",
    required=("session_record",),
    optional=tuple(r for r in FULL_SEARCH_RUN_ROLES if r != "session_record"))

#: Exempt from `open_run`'s occupancy rule and from nothing else: these four are
#: committed BEFORE the run opens, in the order grant -> readiness ->
#: authorization -> bundle, and a launcher that refused to open a run because
#: its own inputs were already there could never start.
_RUN_PREPARED: tuple[str, ...] = ("grant", "readiness_record", "authorization",
                                  "bundle_record")


def spec(args) -> SessionSpec:
    """The whole session, in one object."""
    return SessionSpec(
        session_id=FSG.SESSION_ID,
        schema="aadistill.autoinit.c2_full_search_session/v1",
        description=(
            "Phase C2 full joint re-search: ONE ε-Pareto beam at the standing "
            "width over the DERIVED joint space of all four required operator "
            "kinds, with calibration unpinned, ending at a committed Top-5 "
            "candidate set. It trains nothing, scores no battery, produces no "
            "correct_overall and cannot name an incumbent. Screening and "
            "confirmation are a separate session under a separate "
            "authorization."),
        authorization_path=auth_path_for(getattr(args, "run_id", "")),
        authorization_loader=FSG.FullSearchAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=FSG.PLAN_ID,
        plan_hash=FSG.plan_hash(REPO_ROOT),
        #: The SAME plan the pricing record states -- `budget_spec` is
        #: built from the ingredients `full_search_space.price` used, so
        #: the runner and the record cannot disagree about what this
        #: session costs.
        budget=FSG.budget_spec(REPO_ROOT),
        setup=SetupManifest(
            #: `calib.domain_balanced@v1`'s items, already on the relay. The
            #: other mixture and the suite come from the dev box as assets.
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
            #: ROPE_OK is ABSENT for the same reason it is absent from the
            #: completion: the shared step globs a STAGED student checkpoint's
            #: config and exits 1 when none matches. This session stages no
            #: checkpoint -- it materializes every state on the pod from the
            #: teacher -- so at setup time there is nothing to check and the step
            #: would refuse. Two sessions have paid to establish that
            #: ($0.0412 and $0.1013). The risk is real and it is covered where
            #: the artifact exists: every materialized state is canonically
            #: reloaded and identity-checked inside the search itself, which the
            #: CUDA engineering validation confirmed on a real L40S over 35
            #: states.
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "TEACHER_READY",
                           "TESTS_OK", "AUTHORIZATION_OK", "SETUP_DONE"),
            env={"SESSION_KIND": "c2_full_search",
                 "SESSION_FROZEN_EXPECT": FROZEN_EXPECT},
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION, test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        driver_job_id="autoinit_phase_c2_full_search_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="C2_FULL_SEARCH_ALL_DONE",
            failure=("C2_FULL_SEARCH_FAILED",),
            incomplete=(),
            #: The WEIGHTS of the Top-5 are what a later behavioural session
            #: needs, and they are expensive: five 1.11 GiB checkpoints that
            #: took a multi-hour beam to find. A completed unit of work must
            #: survive a later stage's failure, so they are fetched whenever the
            #: selection stage completed -- gated on the PRODUCING stage, never
            #: on session success, which is the rule $2.82 of verified
            #: checkpoints was lost to.
            products_eligible=lambda terminal, stages: (
                "commit_top_k" in (stages or ())),
            failure_note=("a blocking stage failed — collecting evidence, then "
                          "tearing down. Nothing was trained, no behaviour was "
                          "measured and no committed record was replaced.")),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="c2_full_search_evidence.json",
            archive_basename="c2_full_search_artifacts.tar.gz",
            spec_success="configs/autoinit/c2_full_search_artifacts.json",
            spec_failed="configs/autoinit/c2_full_search_artifacts_failed.json",
            report_names=("c2_full_search_evidence.json",
                          "stage1_selection.json")),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
        precheck=(
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", "")),
                                check_lineage=True),
            full_search_executable_gate,
            full_search_scope_gate,
            frozen_space_gate,
            frozen_runtime_gate,
            frozen_assets_gate,
            staged_inputs_gate,
            pricing_and_plan_gate,
            storage_gate,
            readiness_gate,
            #: LAST, because it is the only gate that touches the network and
            #: everything it would verify against must already be checked.
            bundle_staged_gate,
        ),
    )


#: The scratch-relative paths this run writes and later collects.
RUN_OUTPUTS: tuple[str, ...] = (
    "autoinit_phase_c2_full_search", "c2_full_search_artifacts.tar.gz",
    "watchdog_*.jsonl")


def build_parser() -> argparse.ArgumentParser:
    """The real parser, extracted so a test can assert on the namespace it
    produces rather than on a transcription of it.

    It must supply every name in `RUNNER_ARGUMENT_CONTRACT`:
    `SessionRunner.__init__` reads those attributes and refuses a namespace
    missing any of them -- deterministically, before provider creation, which is
    the cheap place but still a wasted invocation.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    #: REQUIRED, and it is what produces `out`.
    ap.add_argument("--run-id", required=True, action=_RunIdSetsOut,
                    help="the attempt this session runs as, e.g. attempt1")
    ap.add_argument("--relay-repo", default=FST.RELAY_REPO)
    #: The image C2 attempt 4 ran, whose telemetry is half the pooled cost
    #: table. The protocol binds no image because this search compares
    #: against no external measurement -- the ranking is internal to it --
    #: but the cost model's MINUTES are this image's, so a different one
    #: would price the session from another runtime's observations.
    ap.add_argument("--image", default=BOUND_IMAGE)
    ap.add_argument("--gpu", default="NVIDIA L40S")
    #: DEFAULTS TO NONE, resolved in `main` from the AUTHORIZATION's own rate.
    #: Not from the pricing record: the record is a priced basis and the
    #: artifact was issued at a re-quoted rate, so defaulting to the record
    #: would refuse every launch whose rate had legitimately moved.
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--disk-gb", type=int, default=FULL_SEARCH_PROVISION_GIB)
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
    #: Sized to a multi-hour beam, not a one-hour session: the standing width's
    #: ceiling is over thirty hours of minutes and the poll must outlast it.
    ap.add_argument("--poll-limit-min", type=float, default=2100.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every $0 gate and stop before provider creation")
    return ap


def close_full_search_run(layout, args):
    """Record the run, whatever happened. Called on every path."""
    from experiments.run_layout import present_roles

    return record_run(
        layout, spec=FULL_SEARCH_RUN_SPEC,
        plan={"session": FSG.SESSION_ID, "plan_id": FSG.PLAN_ID,
              "session_commit": args.session_commit, "bundle": args.bundle},
        implementation={"launcher":
                        "scripts/pod/autoinit_phase_c2_full_search_launch.py",
                        "driver":
                        "scripts/pod/autoinit_phase_c2_full_search_driver.py"},
        status={"authorizes": "nothing",
                "terminates_at": "commit_top_k"},
        roles=present_roles(layout, FULL_SEARCH_RUN_ROLES))


def main() -> int:
    args = build_parser().parse_args()
    if args.max_price is None:
        #: From the AUTHORIZATION, which was issued at a re-quoted rate.
        auth_file = REPO_ROOT / auth_path_for(args.run_id)
        if not auth_file.is_file():
            raise SystemExit(
                f"{auth_path_for(args.run_id)} does not exist. The max price "
                "defaults to the rate the authorization's ceiling was derived "
                "at, so there is nothing to default to and nothing to launch.")
        money = json.loads(auth_file.read_text()).get("approved_money") or {}
        if "price_basis_usd_per_hour" not in money:
            raise SystemExit(
                "the authorization states no approved rate, so --max-price has "
                "no safe default. Pass one explicitly at or below the rate the "
                "ceiling was derived at.")
        args.max_price = float(money["price_basis_usd_per_hour"])

    #: BEFORE anything is priced or created: a colliding run id or a foreign
    #: scratch root costs $0 here.
    claim_output_root(args.scr, RUN_EXPERIMENT_ID, args.run_id,
                      outputs=RUN_OUTPUTS)
    layout = open_run(REPO_ROOT, RUN_EXPERIMENT_ID, args.run_id,
                      roles=FULL_SEARCH_RUN_ROLES, prepared=_RUN_PREPARED,
                      stage_id=RUN_STAGE_ID)
    write_run_readmes(layout, experiment_id=RUN_EXPERIMENT_ID,
                      run_id=args.run_id, stage_id=RUN_STAGE_ID,
                      roles=FULL_SEARCH_RUN_ROLES)
    assert args.out == session_record_path(args.run_id), (args.out, args.run_id)
    rc = run_session(spec(args), args, REPO_ROOT,
                     summary=("the full joint re-search is a TERMINUS: it "
                              "commits a Top-5 and stops. Behavioural screening "
                              "and confirmation are separately authorized and "
                              "unreachable from here."))
    try:
        doc = close_full_search_run(layout, args)
    except Exception as exc:                                      # noqa: BLE001
        print(f"\nRUN NOT RECORDED: {type(exc).__name__}: {exc}\n"
              f"  the run directory is "
              f"{rel_run_dir(RUN_EXPERIMENT_ID, args.run_id, RUN_STAGE_ID)}; it "
              "holds whatever the session produced and has no manifest. Do not "
              "reuse this run id.")
        return rc or 1
    print(f"run {doc['experiment_id']}/{doc['run_id']} recorded — "
          f"{len(doc['roles'])} role(s)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
