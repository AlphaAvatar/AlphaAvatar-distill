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
import shutil
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
#: The GENERIC run-layout primitives. `scripts/experiments/run_layout.py` owns
#: the five-area convention, the occupancy rule, the output claim and the
#: manifest; what is C2's is the role vocabulary in `phase_c2.session` and the
#: two compositions below. There is no C2 run-layout framework.
from experiments.run_layout import (  # noqa: E402
    ArtifactSpec as RunArtifactSpec, claim_output_root, layout_for, open_run,
    present_roles, record_run, rel_run_dir, require_output_claim,
    write_run_readmes,
)
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

#: What the pod's frozen-asset gate checks this tree against. C2 consumes
#: exactly ONE frozen asset — `state_eval_v1`, the suite every candidate and the
#: baseline are ranked on — and no scoring contract, so the document declares
#: that subset and omits `scoring_contract` entirely.
#:
#: Attempt 2 declared nothing here. The shared setup script then asked the
#: verifier its HISTORICAL question against compiled-in Phase-A/C1 constants,
#: which demanded `artifacts/stage3/recovery_search_v2` — an asset C2 neither
#: stages nor needs — and a scoring digest from a source set C2 does not
#: execute. `SETUP_RC=91`, no driver stage, nothing measured, `$0.0552`. The
#: script now refuses a session that declares `ASSETS_READY` without naming its
#: own expectation, so the fallback that produced that abort is unreachable.
FROZEN_EXPECT = "configs/experiments/phase_c2/frozen_assets.json"

#: The audit root the driver writes into and the collector walks, named ONCE.
#: `ArtifactPolicy` books it and `close_c2_run` looks inside the extracted
#: archive under it; two spellings of this string is how Phase-B attempt 3 came
#: to write `phase_b_search` while its specs named `phase_a_search`, collected
#: nothing, and reported `missing: 0`.
AUDIT_DIRNAME = "autoinit_phase_c2"

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


def session_record_path(run_id: str) -> str:
    """Where THIS run's session record goes, repository-relative.

    ONE rule, called by the parser's `--run-id` action and again by
    `open_c2_run`, so the path the runner writes to and the directory the run
    was created in cannot disagree. Two derivations of one path is how they
    drift.
    """
    return c2_run_path(run_id, "session_record", RUN_STAGE_ID)


#: The scratch-relative paths this run WRITES and later collects. Ownership of
#: the scratch root is decided by these alone: a shared model cache or a staged
#: input living beside them neither claims the directory nor blocks it. The
#: watchdog journals are matched by PATTERN, because each is named after a pod
#: that does not exist when the claim is made — and they are exactly the
#: evidence that a scratch root belonged to a run.
RUN_OUTPUTS: tuple[str, ...] = (
    "launch.log",
    f"relay/{Path(RUN_LOG).name}",
    f"relay/{Path(STATUS).name}",
    "relay/c2_evidence.json",
    "store/manifest.json",
    "store/c2_search_summary.json",
    "store/c2_evidence.json",
    "watchdog_*.jsonl",
)

#: Scratch-relative source -> role, for the small text records the runner leaves
#: beside the pod. Copied into the run AFTER the session, because the scratch
#: directory is outside the repository by design and does not survive as
#: evidence.
#:
#: The relay names are DERIVED from the same constants the relay is built from
#: — `LogRelay` names each local copy `Path(remote).name` — so renaming the
#: status file cannot leave this list quietly pointing at a path that stopped
#: existing. The collection runs once, after teardown, where a wrong name loses
#: the evidence instead of failing.
#:
#: `store/` holds what `ArtifactPolicy.report_names` fetched by scp; `relay/`
#: holds what was streamed while the pod was alive. Both are named because a
#: session that ends badly may have one and not the other.
_RUN_COLLECT: tuple[tuple[str, str], ...] = (
    ("launch.log", "launcher_log"),
    (f"relay/{Path(RUN_LOG).name}", "driver_log"),
    (f"relay/{Path(STATUS).name}", "driver_status"),
    ("relay/c2_evidence.json", "driver_evidence"),
    ("store/c2_search_summary.json", "search_summary"),
    ("store/manifest.json", "artifact_manifest"),
)

#: Roles written before the run opens, by someone other than the launcher.
#: Exempt from `open_run`'s occupancy rule and from NOTHING else: prepared
#: grants no trust and bypasses no gate, and each of these is still validated
#: independently by the gate that owns it.
#:
#: All four are produced by the documented pre-launch sequence, in this order:
#:
#:     grant            a maintainer input, authored before anything else
#:     readiness_record written by the launch-bound sweep, which by contract
#:                      runs before the authorization is issued
#:     authorization    written by `issue_c2_authorization.py --run-id`, and
#:                      read back from this exact path by `session_commit_gate`
#:     bundle_record    written by `stage_c2_bundle.py --run-id`, and read back
#:                      from the working tree by `bundle_staged_gate`
#:
#: C1 shipped without the last two and the chain became unsatisfiable rather
#: than merely strict: the issuer writes the authorization into the run, the
#: launcher reads it from there, nothing copies it in after `open_run` — and
#: `open_run` then refused the run as occupied by an undeclared file. Attempt 13
#: died on it at $0. The exemption is per role, BY NAME: a declared role absent
#: from this tuple is still refused, and an undeclared governance file is still
#: refused.
_RUN_PREPARED: tuple[str, ...] = ("grant", "readiness_record",
                                  "authorization", "bundle_record")

#: What a recorded C2 run must and may contain. `session_record` is the only
#: requirement, because a session refused at a $0 pre-provider gate produced
#: exactly that and nothing else — and it must still be able to record itself,
#: owned, rather than leaving an unowned file in a shared location.
C2_RUN_SPEC = RunArtifactSpec(
    spec_id="phase_c2_session_v1",
    required=("session_record",),
    optional=tuple(r for r in C2_RUN_ROLES if r != "session_record"))


def layout_for_run(repo_root: Path | str, run_id: str):
    """This attempt's layout, without touching the filesystem."""
    return layout_for(repo_root, C2_RUN_EXPERIMENT_ID, run_id,
                      stage_id=RUN_STAGE_ID)


def open_c2_run(args, repo_root: Path | None = None):
    """Claim this attempt's outputs, create its run directory, point `out` at it.

    Runs BEFORE `SessionSpec` construction and therefore before any provider
    call, so a run id that collides with a recorded run — or a scratch root that
    belongs to a different attempt — costs `$0` rather than being discovered
    after a pod exists. `SessionRunner.save()` writes `args.out` without
    creating its parent, which is the other reason this happens first.

    The scratch claim comes FIRST. The run directory and `--scr` are two
    independent output locations, and checking only the first is how a fresh run
    id aimed at a previous attempt's scratch passed the run-directory rule and
    then collected that attempt's evidence as its own.
    """
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    claim_output_root(args.scr, C2_RUN_EXPERIMENT_ID, args.run_id,
                      outputs=RUN_OUTPUTS)
    layout = open_run(repo_root, C2_RUN_EXPERIMENT_ID, args.run_id,
                      roles=C2_RUN_ROLES, prepared=_RUN_PREPARED,
                      stage_id=RUN_STAGE_ID)
    #: Describe the directories as they are created. Documentation only: the
    #: manifest stays the canonical index, and a README neither counts as a
    #: produced role nor makes an unexecuted run look like a failed one.
    write_run_readmes(layout, experiment_id=C2_RUN_EXPERIMENT_ID,
                      run_id=args.run_id, stage_id=RUN_STAGE_ID,
                      roles=C2_RUN_ROLES)
    #: Idempotent for a parser-built namespace, and the whole answer for a
    #: hand-built one. Same rule either way — see `session_record_path`.
    args.out = session_record_path(args.run_id)
    return layout


def close_c2_run(layout, args, repo_root: Path | None = None) -> dict:
    """Collect the small records beside the pod, then write the run manifest.

    Runs after the session on every normal return path, including a launcher
    error, because the runner already caught that and saved. What it cannot
    cover is the launcher PROCESS dying: then the run directory exists with no
    manifest, and that is exactly the state `open_run` refuses to reopen.

    Large working state stays out. The search's intermediate checkpoints are
    tens of GiB and the collected archive is in scratch; `artifacts/manifest.json`
    carries their hashes, and nothing here copies a model directory into git to
    satisfy the layout.
    """
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    scr = Path(args.scr)
    #: Asked AGAIN here, not assumed from the open. The two happen at opposite
    #: ends of a session, and what is collected has to be what THIS execution
    #: produced — otherwise a foreign scratch turns a run that failed before its
    #: driver started into a manifest full of somebody else's evidence.
    require_output_claim(scr, C2_RUN_EXPERIMENT_ID, args.run_id)
    for source, role in _RUN_COLLECT:
        src = scr / source
        if src.is_file():
            shutil.copy2(src, layout.path(C2_RUN_ROLES[role]))
    #: The comparison record, which the artifact spec marks REQUIRED on a
    #: successful run. It is fetched into `store/` only if the archive was
    #: extracted, so it is looked for in both places rather than assumed.
    for candidate in (scr / "store/c2_baseline_comparison.json",
                      scr / f"store/extracted/audit/{AUDIT_DIRNAME}/"
                            "c2_baseline_comparison.json"):
        if candidate.is_file():
            shutil.copy2(candidate,
                         layout.path(C2_RUN_ROLES["baseline_comparison"]))
            break
    #: Every resource's watchdog evidence, by the pod id in its name. Resolved
    #: by glob rather than listed, because how many resources a session held is
    #: only known once it has ended.
    wd = layout.path(C2_RUN_ROLES["watchdog_journal"])
    wd.mkdir(parents=True, exist_ok=True)
    for src in (sorted(scr.glob("watchdog_*.jsonl"))
                + sorted(scr.glob("watchdog_*.out"))):
        shutil.copy2(src, wd / src.name)

    session = json.loads((repo_root / args.out).read_text())
    return record_run(
        layout, spec=C2_RUN_SPEC,
        plan={"session_id": session.get("session_id"),
              "plan_hash": session.get("session_plan_hash"),
              "session_commit": getattr(args, "session_commit", None),
              "bundle": getattr(args, "bundle", None),
              "scratch_root": str(scr),
              "scratch_note": ("the artifact archive, the extracted tree and "
                               "every intermediate search state stay here; "
                               "artifacts/manifest.json carries their hashes. "
                               "Large artifacts are not moved into git")},
        implementation={"launcher": "scripts/pod/autoinit_phase_c2_launch.py",
                        "driver": "scripts/pod/autoinit_phase_c2_driver.py",
                        "harness_source_digest": session.get(
                            "harness_source_digest"),
                        "authorization": auth_path_for(args.run_id)},
        status={"passed": session.get("passed"),
                #: `terminal`, spelled the way the runner writes it. A key the
                #: record does not have would read as `None` and look like a
                #: session that produced no marker.
                "terminal": session.get("terminal"),
                "pod_id": session.get("pod_id") or None,
                "cost": session.get("cost"),
                "provider_confirms_gone": session.get("provider_confirms_gone"),
                "trains_anything": False,
                "authorizes": "nothing"},
        roles=present_roles(layout, C2_RUN_ROLES))


#: Exit code for "the session finished, the run did not get recorded".
#:
#: Distinct from the session's own codes, and it never overwrites one: a session
#: that already failed keeps its result, because the pod outcome is what an
#: operator acts on. But an unrecorded run is not a silent condition either — it
#: is exactly the state `open_run` refuses to reopen, so a successful session
#: that could not record itself must not exit 0.
RUN_NOT_RECORDED = 12


class _RunIdSetsOut(argparse.Action):
    """`--run-id` also produces `out`, because the RUNNER reads `out`.

    `SessionRunner.save()` writes `args.out`, and the structural check
    `test_every_session_namespace_carries_what_the_runner_reads` requires every
    such attribute to come from the REAL parser: device-canary attempt 1 died at
    `$0.0603` on an attribute a hand-written namespace had and the parser did
    not, after the pod was created and billing. Filling `out` in later would
    leave the parser's namespace incomplete and that gate red.

    Deriving it here keeps both properties: the namespace is complete, and there
    is no `--out` flag that could point one attempt's session record at another.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, value)
        namespace.out = session_record_path(value)

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


def frozen_assets_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The expectation the pod's setup will check, verified here first, at $0.

    Not defence in depth: it is the check whose absence cost attempt 2. The
    shared setup script runs the verifier on the pod, after the checkout, the
    staging and the training environment have all been paid for — so a document
    that is missing, malformed, or simply wrong about a staged asset is
    discovered at roughly `$0.05` and one consumed one-use chain. Run here, the
    same verifier answers the same question for nothing.

    It is honest about what it can and cannot see: `state_eval_v1` is staged as
    a WHOLE TREE, so the bytes this checks are the bytes the pod gets. An
    expectation naming something C2 stages at file granularity would need the
    staging contract to say so, and this gate would then be checking a more
    generous tree than the pod's.
    """
    import subprocess

    rel = FROZEN_EXPECT
    expect = REPO_ROOT / rel
    if not expect.is_file():
        return False, (f"{rel} is missing, and the setup script refuses a "
                       "session that declares ASSETS_READY without one rather "
                       "than falling back to another experiment's constants")
    try:
        doc = json.loads(expect.read_text())
    except json.JSONDecodeError as exc:
        return False, f"{rel} is not readable as JSON: {exc}"
    assets = doc.get("assets")
    if not isinstance(assets, dict) or not assets:
        return False, (f"{rel} declares no assets; an empty expectation passes "
                       "vacuously and would verify nothing")
    for forbidden in ("recovery_search_v2", "recovery_search_v1"):
        if forbidden in assets:
            return False, (f"{rel} requires {forbidden}, which this session "
                           "does not stage. That is the attempt-2 abort.")
    if "scoring_contract" in doc:
        return False, (f"{rel} declares a scoring_contract. C2 trains nothing "
                       "and scores no battery; it consumes none.")

    with tempfile.TemporaryDirectory() as tmp:
        done = subprocess.run(
            #: The DEV-BOX interpreter: this is the $0 pre-provider check. The
            #: pod runs the same script under /opt/train, which is where the
            #: setup script invokes it.
            [sys.executable, "scripts/autoinit/verify_frozen_assets.py",
             "--expect", rel, "--out", f"{tmp}/frozen_check.json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": "src:scripts"})
    if done.returncode != 0:
        return False, (f"the frozen-asset expectation {rel} does not verify "
                       f"against this tree: {(done.stdout + done.stderr)[-500:]}")
    ctx.evidence["frozen_assets_expectation"] = {
        "document": rel,
        "assets": sorted(assets),
        "scoring_contract_requested": False,
        "verified_at_zero_cost": True,
        "rule": ("the pod runs the same verifier with --expect on this "
                 "document; a session declaring ASSETS_READY and naming no "
                 "expectation is refused by the setup script"),
    }
    return True, (f"frozen-asset expectation {rel} verifies: "
                  f"{sorted(assets)}, no scoring contract requested")


def resource_scope_gate(ctx: SessionContext) -> tuple[bool, str]:
    """This run, and no more provider resources than the grant permitted.

    Three refusals, all at `$0` and all before `create()`:

    * the launcher is running as a run the authorization was not issued for —
      an authorization belongs to one attempt, whose grant, readiness record
      and bundle all live in that run;
    * more host draws are requested than the authorization permits;
    * the authorization carries no machine-readable scope at all, which is a
      refusal rather than a permission: an unknown limit is not an unlimited
      one.

    It adds no mechanism. The two things that actually keep a session safe once
    it is running already live in the shared runner and are untouched: a second
    resource is not created until the provider confirms the first is not
    billing, and every draw shares this session's single dollar ceiling. What
    was missing was anybody checking the COUNT the maintainer wrote down.
    """
    run_id = getattr(ctx.args, "run_id", None)
    scope = getattr(ctx.auth, "resource_scope", None)
    if scope is None:
        return False, (
            "the authorization carries no resource scope, so the number of "
            "provider resources it permits is unknown. An unknown limit is not "
            "an unlimited one; re-issue from a grant that states `one_use` with "
            "issuances_permitted, launch_attempts_permitted, "
            "provider_resources_permitted and one_billing_resource_at_a_time.")
    ok, why = scope.permits_run(run_id)
    if not ok:
        return False, why
    requested = int(getattr(ctx.args, "host_draws", 0) or 0)
    ok, draws_why = scope.permits_draws(requested)
    if not ok:
        return False, draws_why
    ctx.evidence["resource_scope"] = {
        **scope.as_dict(),
        "requested_host_draws": requested,
        "create_attempts_per_draw": int(getattr(ctx.args, "create_attempts", 0)
                                        or 0),
        "enforced_before": "provider creation",
        "enforced_by": ("this gate for the COUNT; SessionRunner for the "
                        "confirmed-release rule and the dollar ceiling"),
    }
    return True, f"{why}; {draws_why}"


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
                          "SESSION_KIND", "SESSION_FROZEN_EXPECT",
                          "SESSION_SETUP_MARKERS", "TEACHER_REVISION"),
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
            env={"SESSION_KIND": "c2",
                 #: NAMED, not inherited. See FROZEN_EXPECT above.
                 "SESSION_FROZEN_EXPECT": FROZEN_EXPECT},
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
            audit_dirname=AUDIT_DIRNAME,
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
        #: * the resource-scope gate checks the run identity and the number of
        #:   provider resources the grant actually permitted — the count was
        #:   prose in the grant and enforced by nothing;
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
            resource_scope_gate,
            frozen_assets_gate,
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
    #: REQUIRED, and it is what produces `out`. Every artifact this session
    #: consumes or produces is owned by its run — grant, authorization,
    #: readiness record, bundle record, session record, evidence, manifest —
    #: and each is resolved from this id. A session with no run id has no grant
    #: that can belong to it and nowhere for its record to live that the next
    #: attempt would not overwrite.
    ap.add_argument("--run-id", required=True, action=_RunIdSetsOut,
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
    #: No `--out`. It is `run_id` and the layout, or it is nothing.
    #:
    #: It was `--out` with a default of
    #: `logs/stages/stage-1/phase_c2/runs/autoinit_phase_c2_session.json` — a
    #: file directly in the runs root, belonging to no run, which every attempt
    #: would have written in turn. `SessionRunner.save()` writes exactly
    #: `args.out`, so an operator could also point the record of one attempt at
    #: another. `--run-id` fills `out` through `_RunIdSetsOut`, so the namespace
    #: the runner reads is complete and there is no flag that could aim it
    #: anywhere but at its own run.
    return ap


def main() -> int:
    args = build_parser().parse_args()
    #: BEFORE `spec(args)` and therefore before anything is priced or created:
    #: a colliding run id or a foreign scratch root costs $0 here.
    layout = open_c2_run(args)
    rc = run_session(spec(args), args, REPO_ROOT,
                     summary=("Search-1 is a terminus: Search-2 is "
                              "conditional on this evidence and behavioural "
                              "confirmation is separately authorized."))
    try:
        doc = close_c2_run(layout, args)
    except Exception as exc:                                      # noqa: BLE001
        print(f"\nRUN NOT RECORDED: {type(exc).__name__}: {exc}\n"
              f"  the run directory is "
              f"{rel_run_dir(C2_RUN_EXPERIMENT_ID, args.run_id, RUN_STAGE_ID)}; "
              "it holds whatever the session produced and has no manifest. Do "
              "not reuse this run id.")
        return rc or RUN_NOT_RECORDED
    print(f"run {doc['experiment_id']}/{doc['run_id']} recorded — "
          f"{len(doc['roles'])} role(s) under "
          f"{rel_run_dir(doc['experiment_id'], doc['run_id'], RUN_STAGE_ID)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
