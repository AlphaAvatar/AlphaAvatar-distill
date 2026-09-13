#!/usr/bin/env python3
"""Phase C1 — fixed-path ATTENTION isolation, as a session specification.

    PYTHONPATH=src setsid nohup python -u scripts/pod/autoinit_c1_launch.py \
        --scr <scratch> --run-id <attemptN> \
        --session-commit <sha> --bundle <name> < /dev/null &

**This session does not search.** It replays one frozen operator sequence, gates
it against two recorded artifact digests, then runs six fixed probes. There is no
beam, no ranking, no successive halving, no tie-breaking and no arm elimination —
and none of those is a flag to be turned off: `C1Authorization.allows_beam_search`
is a hard `False`, `C1IsolationPlan` has no `survivors` or `tie_break_seed` field
to set, and `c1_session.assert_stage_order` refuses a permuted run.

Three properties are declared here rather than assumed.

*The science lives elsewhere.* This launcher builds a `SessionSpec` and nothing
else. The stage order, the two replay gates, the arm construction and the
decision rule are `aadistill.autoinit.c1_session` and `c1_isolation`; duplicating
any of them here would create a second copy to keep in step.

*The ceiling is derived, once.* `c1_budget_spec()` reads
`logs/stages/stage-1/phase_c1/plans/phase_c1_pricing.json` and back-derives the step time from its measured
per-probe minutes, so the enforceable ceiling exists in exactly one place. A
second hand-maintained figure is how a session comes to be authorized for one
number and priced at another.

*The authorization is a distinct type with its own setup branch.*
`SESSION_KIND=c1` is declared, because an undeclared kind falls through to
`spend` and loads a `SpendAuthorization` — which is what killed Phase-B attempt 2
at `$0.2300`, one step after its test gate passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))   # experiments.* live here
sys.path.insert(0, str(Path(__file__).resolve().parent))
# `renderer_parity_gate` lives with the other dev-box verifiers, and the eleventh
# pre-provider gate executes it directly rather than trusting a transcript of it.
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from experiments.deployment import MAIN_RELAY, POD_IMAGE, deployment_commands  # noqa: E402
from experiments.run_layout import (  # noqa: E402
    ArtifactSpec as RunArtifactSpec, claim_output_root, layout_for,
    rel_run_dir,
    open_run, present_roles, record_run, require_output_claim, write_run_readmes,
)
from experiments.phase_c1 import session as CS
from experiments.phase_c1.authorization import C1_HARNESS_SOURCE_FILES_V1, C1Authorization, c1_budget_spec, c1_hard_ceiling_usd, c1_harness_digest, c1_price_per_hour_usd  # noqa: E402
from experiments.phase_c1.bundle import RELAY_REPO as RELAY_REPO_ID, C1BundleError, canonical_bundle_name, hf_download, require_canonical_bundle_arg, roundtrip  # noqa: E402
from experiments.phase_c1.isolation import derive_recovery_seeds  # noqa: E402
from experiments.phase_c1.authorization_payload import load_config  # noqa: E402
from aadistill.runtime.staging_contract import derive_contract  # noqa: E402
from experiments.phase_c1.pod_environment import (  # noqa: E402
    LAUNCH_BOUND,
    RECORD_POINTER as POD_ENV_RECORD,
    c1_record_contract,
    record_path_for as pod_env_record_for,
    load_record as load_pod_env_record,
    verify_record as verify_pod_env_record,
)
from aadistill.infrastructure.manifest import (  # noqa: E402
    sha256_file, sha256_json,
)
from aadistill.infrastructure.session import (
    ExecutionCommands,  # noqa: E402
    ArtifactPolicy, LocalAsset, MarkerPolicy, SessionContext, SessionSpec,
    SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate,
)
from aadistill.infrastructure.session_runner import run_session  # noqa: E402
#: The image layout THIS session declares, and the same values it hands
#: the runner. They were module constants in the runner, so a second
#: image could only be supported by patching the framework's globals.
WS = POD_IMAGE["workspace_root"]
REPO = POD_IMAGE["checkout_root"]
from aadistill.infrastructure.session import RelayInput  # noqa: E402
from autoinit_science_inputs import CALIBRATION_V1, RECOVERY_LADDER  # noqa: E402

#: The frozen EVALUATION tokenizer, and nothing else from that checkpoint.
#:
#: Declared HERE rather than beside the other frozen science inputs, which is
#: where it belongs by topic: `scripts/pod/autoinit_science_inputs.py` is a member
#: of FIVE hash-bound executable sets (Phase A, Phase B, both continuations and
#: the measurement authorization), so adding a C1-only group to it moved five
#: frozen digests for a group only C1 reads. The launcher is already inside the
#: C1 harness, so the pins stay measured and no other phase's identity moves.
#:
#: Stage H evaluates each probe through a PACKAGE — the trained model files plus
#: these three sidecars — so the frozen generation protocol's
#: `tokenizer_source = "the evaluated checkpoint"` stays literally true without
#: mutating the scientific checkpoint. Only the sidecars are needed: 10.9 MiB, not
#: the 1.19 GiB of weights. The teacher's own tokenizer CANNOT substitute —
#: `tokenizer.json` is 11,422,654 bytes at `aeb13307…` against `be756060…` here,
#: `tokenizer_config.json` is 10,834 bytes against 694, and the teacher ships no
#: `chat_template.jinja` at all.
C1_EVAL_TOKENIZER: tuple[RelayInput, ...] = tuple(
    RelayInput(f"stage1/qwen3_0p6b_init_v0/checkpoint/{name}",
               dest="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint", sha256=sha, repo=MAIN_RELAY)
    for name, sha in (
        ("tokenizer.json",
         "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
        ("tokenizer_config.json",
         "8fa82a4ba512c8bee7c1c5e82b9a71ddbef362e4665be5c8f7ce0afd78af129a"),
        ("chat_template.jinja",
         "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"),
    )
)

#: SETUP READINESS, not a C1 measurement input. The shared setup's `ROPE_OK` step
#: globs `artifacts/stage1/*/checkpoint/config.json` and loads each match through
#: `AutoConfig.from_pretrained` in BOTH venvs, requiring a stored RoPE base of
#: 5,000,000. It reads no weights. C1 attempt 2 staged the three sidecars above
#: and nothing else, so the glob was empty and setup exited `no staged checkpoint
#: to check` after the teacher had already been fetched and verified — $0.1013.
#:
#: 1,418 bytes. The alternative was the full CANONICAL_INIT group, which would
#: pull 1.19 GiB of weights this session never opens to satisfy a check that
#: never reads them.
#:
#: The hash is INDEPENDENTLY VERIFIED, not transcribed: the relay object was
#: downloaded read-only and hashed, and `rope_input_gate` re-derives it before
#: every launch.
C1_ROPE_INPUT: tuple[RelayInput, ...] = (
    RelayInput("stage1/qwen3_0p6b_init_v0/checkpoint/config.json",
               dest="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
               sha256="a7131bb092b38a078edc213961f0eb57eaead24f1396e25741f4887b1a694054", repo=MAIN_RELAY),
)
#: What `stored_rope_base` must report for the staged config, in both venvs.
C1_ROPE_BASE = 5_000_000
#: The directory the shared setup globs. Named once so the gate and the
#: RelayInput cannot drift apart.
C1_ROPE_CHECKPOINT_DIR = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"

#: C1's own, not Phase A's. The Phase-A launcher was imported for three things:
#: the teacher revision (which `c1_session` already declares), a two-entry test
#: ignore list, and its `build_parser` -- whose flags are `--rung1-probes`,
#: `--rung2-probes`, `--tie-break-probes`, `--search-minutes`,
#: `--stage-leaves-to-relay` and `--fetch-finalists`. Inheriting that parser gave
#: this session a command line for a search it structurally cannot run, which is
#: the same class of defect as inheriting its driver. So C1 declares its own.
TEACHER_REVISION = CS.TEACHER_REVISION
#: Two Phase-A rehearsals C1 does not exercise. C1's own execution regression is
#: deliberately NOT ignored: it is ~60 seconds against a 2700 s gate, and running
#: it on the pod proves the driver's control flow in the real environment before
#: any stage spends money.
#: `test_phase_b_reuse_hostlocal.py` joined them after attempt 4, which it cost
#: three of its six failures. The module's premise is a retained probe byte store
#: that is deliberately HOST-LOCAL — the file says so in its name — and Phase B's
#: own session already excludes it from its pod test gate for exactly that reason.
#: C1 re-ran it on a host where its premises are absent. The WHOLE module is
#: ignored, not the three nodeids that happened to fail: the other cases in it
#: rest on the same absent store and would fail the moment they were reached.
#: Staging the store instead was rejected — it is Phase-B reuse evidence, not a C1
#: runtime input, and no frozen Phase-B science is touched here.
#: `test_stage1_import.py` joined on 2026-09-05. The WHOLE module runs against
#: the real Attempt-12 retained checkpoints at
#: `/home/ecs-user/aad-artifacts/autoinit/phase_a`, and module-skips on any
#: machine without that store — which is every pod. It is host-local Phase-A
#: evidence, not a declared C1 staged input, and copying it onto a billing GPU to
#: satisfy pytest was rejected. This changes C1 TEST SELECTION only; the module's
#: Phase-A and continuation semantics are untouched.
TEST_IGNORES = ("tests/data/test_recovery_corpus_pipeline.py",
                "tests/pod/test_phase_a_stages1_5_execute.py",
                "tests/autoinit/test_phase_b_reuse_hostlocal.py",
                "tests/autoinit/test_stage1_import.py")

STATUS = f"{WS}/autoinit_c1.status"
RUN_LOG = f"{WS}/autoinit_c1_run.log"
#: The GLOBAL entry point for the issued authorization. Like the readiness
#: record, this was the canonical artifact: every issuance overwrote one
#: repository-root file, and each closeout copied it into the run afterwards to
#: keep a copy. The authorization a session ran under therefore lived at a path
#: the next issuance would replace.
AUTH_POINTER = "logs/budget/approvals/autoinit_c1_authorization.json"

#: Back-compatible name. Every authorization issued before 2026-09-12 is here,
#: and a session with no run id still reads it.
AUTH_PATH = AUTH_POINTER


def auth_path_for(run_id: str | None) -> str:
    """Where THIS run's issued authorization lives, repository-relative.

    Resolved through the same convention as every other role, so the issuer, the
    gates, the lineage rule and the run's own manifest cannot disagree about
    where it is.
    """
    if not run_id:
        return AUTH_POINTER
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
            f"/{C1_RUN_ROLES['authorization']}")

PRICING = "logs/stages/stage-1/phase_c1/plans/phase_c1_pricing.json"
PREREG = "logs/stages/stage-1/phase_c1/plans/execution_preregistration.json"
#: The expectation the frozen-asset gate checks this tree against, on the pod
#: and — since 2026-09-11 — at $0 before a pod exists.
FROZEN_EXPECT = "configs/experiments/phase_c1/frozen_assets.json"
#: Declared once. `artifact_spec_gate` reads these and `ArtifactPolicy` books
#: them, so the gate cannot end up validating a different file than the one the
#: pod is handed.
SPEC_SUCCESS = "configs/autoinit/c1_artifacts.json"
SPEC_FAILED = "configs/autoinit/c1_artifacts_failed.json"
BATTERY_MANIFEST = "artifacts/stage3/c1_confirmation_v1/manifest.json"
BATTERY_IDENTITY = "logs/stages/stage-1/phase_c1/plans/battery.json"
TEACHER_BINDING = "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
#: Written by scripts/autoinit/stage_c1_bundle.py; the local half of the
#: transport check. The gate verifies the REMOTE object against it.
#: The GLOBAL entry point for the staged-bundle record. Like the readiness
#: record and the authorization, this was one repository-root file that every
#: session staged over -- and it is the one artifact that MUST stay uncommitted
#: inside a launch window, because committing it adds a third path to the
#: session lineage diff and `session_commit_gate` refuses.
BUNDLE_POINTER = "logs/stages/stage-1/phase_c1/analyses/autoinit_c1_bundle.json"
BUNDLE_RECORD = BUNDLE_POINTER


def bundle_record_for(run_id: str | None) -> str:
    """Where THIS run's bundle record lives, repository-relative."""
    if not run_id:
        return BUNDLE_POINTER
    return (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
            f"/{C1_RUN_ROLES['bundle_record']}")

# ---------------------------------------------------------------------------
# where this run's files go
#
# Every C1 attempt so far wrote its session record to ONE flat path,
# `logs/stages/stage-1/phase_c1/analyses/autoinit_c1_session.json`, which the next attempt overwrote; the evidence
# directory `logs/stages/stage-1/phase_c1/runs/attempt9/` was then assembled by hand afterwards,
# and the run index found it by matching the directory's NAME. Three
# consequences, all of them real: the live record and the preserved copy are
# byte-identical duplicates of one fact, `logs/state/ownership.md` described the live
# file as attempt 5's when it held attempt 9's, and a launcher that died before
# the manual step left evidence with no owner at all.
#
# So the run declares its identity BEFORE it runs, and the launcher writes into
# it. `experiments.run_layout` owns the five-area convention; the roles below are
# C1's own vocabulary, which is why a Stage-0 collection run or a rollout
# benchmark can use the same mechanism without inheriting `replay_record`.
# ---------------------------------------------------------------------------

#: This experiment's key in `logs/runs/` and in the run index. `phase_c1` is
#: already the index's experiment id for attempts 1-9, so a tenth attempt joins
#: the same series instead of starting a parallel one.
RUN_EXPERIMENT_ID = "phase_c1"

#: The pipeline stage this experiment's runs EXECUTE, read from the experiment's
#: own configuration rather than decided here. C1 trains and evaluates Stage-3
#: recovery probes to answer a Stage-1 question, under Phase C; the config
#: carries that distinction and this module does not restate it.
RUN_STAGE_ID = load_config(REPO_ROOT)["stage_id"]

#: role -> path inside this run. Small, reviewable text only: the artifact
#: TARBALL and the extracted tree stay in the scratch directory, and
#: `artifacts/manifest.json` carries their hashes. A run manifest holds a
#: verifiable reference to a large artifact; it never holds the artifact.
C1_RUN_ROLES: dict[str, str] = {
    #: Written by `SessionRunner.save()` on every path, including a launcher
    #: error, so it is the one role that is always present.
    "session_record": "runtime/session.json",
    "launcher_log": "runtime/launcher.log",
    #: A DIRECTORY, because a session may hold more than one resource in turn
    #: and each watchdog writes its own journal from its first tick. One file
    #: here would mean one resource's evidence collected and the rest left in
    #: scratch -- or, before the isolation repair, two resources' events in one
    #: file.
    "watchdog_journal": "runtime/watchdog/",
    #: The maintainer decision this attempt runs under. UNLIKE every other role
    #: here it is an INPUT: it is authored and committed before the launch-bound
    #: sweep, because the authorization is issued from it and the sweep must see
    #: the final clean tree. The launcher neither writes nor copies it -- it is
    #: already in place, exempted from `open_run`'s occupancy rule by name, and
    #: verified against the authorization that records its hash.
    #:
    #: Its predecessors were `logs/autoinit_c1_attempt<N>_grant.json`: nine flat
    #: files in the log root, each one a per-attempt fact with no run to belong
    #: to. This is the same fact with an owner.
    "grant": "governance/grant.json",
    #: Snapshots of the ONE-USE artifacts this attempt consumed. Their live
    #: paths are rewritten by the next issuance, so the snapshot is a fact about
    #: this run that has no other owner -- not a second copy of a current one.
    "authorization": "governance/authorization.json",
    "bundle_record": "governance/bundle.json",
    "readiness_record": "governance/readiness.json",
    "driver_evidence": "evidence/c1_evidence.json",
    "driver_log": "evidence/driver_run.log",
    #: The COMPLETE marker sequence. The session record echoes only the last
    #: status it saw into its timeline, so the stream is the one place the whole
    #: ordering survives -- and stage ordering is what `assert_stage_order`
    #: exists to police.
    "driver_status": "evidence/driver_status.txt",
    "artifact_manifest": "artifacts/manifest.json",
    #: NOT written by the launcher. A maintainer's post-review classification
    #: outlives the process that ran the session; declaring the role says where
    #: it goes and lets a later `record_run` name it without widening the spec.
    "outcome": "closeout/outcome.json",
}

C1_RUN_SPEC = RunArtifactSpec(
    spec_id="phase_c1_session_v1",
    required=("session_record",),
    optional=tuple(r for r in C1_RUN_ROLES if r != "session_record"))

#: Scratch-relative source -> role, for the small text files the runner leaves
#: beside the pod. Copied into the run after the session, because the scratch
#: directory is outside the repository by design and does not survive as
#: evidence.
#:
#: The three relay names are DERIVED from the same constants the relay is built
#: from — `LogRelay` names each local copy `Path(remote).name` — so renaming the
#: status file cannot leave this list quietly pointing at a path that stopped
#: existing. That is a transcription this file would otherwise have to keep in
#: step by hand, and the collection runs once, after teardown, where a wrong
#: name loses the evidence instead of failing.
_RUN_COLLECT: tuple[tuple[str, str], ...] = (
    ("launch.log", "launcher_log"),
    (f"relay/{Path(RUN_LOG).name}", "driver_log"),
    (f"relay/{Path(STATUS).name}", "driver_status"),
    ("relay/c1_evidence.json", "driver_evidence"),
    ("store/manifest.json", "artifact_manifest"),
)

#: Repository-relative source -> role, snapshotted when the run opens, while the
#: artifacts still describe THIS attempt.
#:
#: `grant` is deliberately absent: it is not copied from a live repository path,
#: because it has no live repository path. It is authored directly at its role
#: location and is already there when the run opens.
#:
#: `readiness_record` left for the same reason on 2026-09-12. The sweep writes
#: it into this run's governance area directly, so there is nothing to copy --
#: and copying it was the workaround for its living at a repository-root path
#: that the next run would overwrite. The root file is now a pointer and is
#: never snapshotted: a pointer is not evidence.
#: `authorization` left on 2026-09-12 for the same reason as the readiness
#: record: the issuer writes it into this run's governance area, so there is
#: nothing to copy. The repository-root file is a pointer and is never
#: snapshotted -- a pointer is not evidence.
#: EMPTY since 2026-09-12. Every governance artifact -- grant, readiness
#: record, authorization, bundle record -- is now produced into the run that
#: owns it, so there is nothing to copy in at closeout. Copying was the
#: workaround for artifacts living at repository-root paths the next session
#: would overwrite. Kept as an empty declaration rather than deleted: a future
#: artifact that genuinely arrives from outside the run belongs here.
_RUN_GOVERNANCE: tuple[tuple[str, str], ...] = ()

#: Roles written before the run opens, by someone other than the launcher.
#: Exempt from `open_run`'s occupancy rule and from NOTHING else: prepared
#: grants no trust, bypasses no provenance gate, and each of these four is still
#: validated independently by the gate that owns it.
#:
#: All four are produced by the documented pre-launch sequence, in this order:
#:
#:     grant            a maintainer input, authored before anything else
#:     readiness_record written by the launch-bound sweep, which by contract
#:                      runs before the authorization is issued
#:     authorization    written by `issue_c1_authorization.py --run-id`, and
#:                      read back from this exact path by `session_commit_gate`
#:     bundle_record    written by `stage_c1_bundle.py --run-id`, and read back
#:                      from the working tree by `bundle_staged_gate`
#:
#: The last two were missing, and the omission made the chain unsatisfiable
#: rather than merely strict: the issuer writes the authorization into the run,
#: the launcher reads it from there, `_RUN_GOVERNANCE` is empty so nothing
#: copies it in after `open_run` -- and `open_run` then refused the run as
#: occupied by an undeclared file. There was no ordering that satisfied all
#: three. Attempt 13 died on it at $0, before pricing.
#:
#: The exemption is per role, BY NAME. Not `governance/`, not a glob, not "any
#: declared role": a declared role absent from this tuple is still refused, and
#: an undeclared governance file is still refused.
_RUN_PREPARED: tuple[str, ...] = ("grant", "readiness_record",
                                  "authorization", "bundle_record")


def session_record_path(run_id: str) -> str:
    """Where THIS run's session record goes, repository-relative.

    ONE rule, called by the parser and by `open_c1_run`, so the path the runner
    writes to and the directory the run was created in cannot disagree. Two
    derivations of one path is how they drift.
    """
    return str(layout_for(REPO_ROOT, RUN_EXPERIMENT_ID, run_id,
                          stage_id=RUN_STAGE_ID).root.relative_to(REPO_ROOT)
               / C1_RUN_ROLES["session_record"])


class _RunIdSetsOut(argparse.Action):
    """`--run-id` also produces `out`, because the RUNNER reads `out`.

    `SessionRunner.save()` writes `args.out`, and
    `test_every_session_namespace_carries_what_the_runner_reads` requires every
    such attribute to come from the REAL parser: device-canary attempt 1 died at
    `$0.0603` on an attribute a hand-written namespace had and the parser did
    not, *after* the pod was created and billing. Filling `out` in later would
    have left the parser's namespace incomplete and that gate red, and narrowing
    the gate to suit this session is the move that cost attempt 2 `$0.1013`.

    Deriving it here keeps both properties: the namespace is complete, and there
    is still no `--out` flag that could point the session record somewhere other
    than its own run.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, value)
        namespace.out = session_record_path(value)

#: EXACTLY ONE provider resource, for the whole session.
#:
#: The C1 grant permits one issuance, one launch attempt and one provider
#: resource, and says in as many words that after consumption there is no retry
#: and no replacement pod. The launcher did not enforce any of that: it defaulted
#: to `--create-attempts 8` and `--host-draws 3`, so a cold or endpoint-less
#: first pod would be deleted and a SECOND one drawn — a replacement pod the
#: grant forbids — and a create failure would sleep 300 s and try again, up to
#: seven times, waiting on stock the grant says not to chase.
#:
#: `host_draws = 1` makes `if outcome in ("cold","no_endpoint") and draw <
#: host_draws` false by construction, so the redraw branch is unreachable and
#: every abort falls through to `teardown_now`. `create_attempts = 1` makes
#: `if attempt < create_attempts` false, so there is exactly one provider-create
#: invocation and no sleep. The same field also bounds `check_gpu_offered`'s
#: zero-provider price READ, which therefore becomes a single `$0` read that
#: returns for review — the simpler of the two options the review allowed, and
#: it cannot create a resource or wait for stock.
#:
#: SUPERSEDED 2026-09-11 by a maintainer decision, prospectively. The paragraph
#: above is why the limits existed and what they protected against; it is kept
#: because attempts 1-11 ran under it. What changed is only the FIRST of the two
#: rules it enforced.
#:
#: Attempt 11 showed what coupling them costs. Its pod was created, billed and
#: never became reachable -- a provider cold host, which every earlier session in
#: this project handled by deleting the draw and taking another, and which this
#: launcher could only handle by consuming an entire formal attempt. Under
#: `host_draws = 1` a condition every other session treats as a retryable DRAW
#: ends the ATTEMPT.
#:
#: So draws are permitted again, up to a batch of three. What is NOT relaxed is
#: the property the old rule was really protecting: **at most one billing
#: resource at any instant**. That is no longer enforced by making the redraw
#: branch unreachable -- it is enforced inside that branch, by
#: `SessionRunner.release_and_confirm`, which refuses to create the next
#: resource until the provider itself reports the abandoned one not billing. A
#: rule that holds by construction is better than a rule that holds because a
#: code path is dead, and the dead path was hiding a real defect: it cleared
#: `pod_id` locally and continued without ever asking the provider.
#:
#: `create_attempts` stays at 1. Draws replace an unusable HOST; create-attempts
#: sleep 300 s and ask the same market again, which is stock chasing, and that
#: is still not authorized. The per-session ceiling is unchanged and covers ALL
#: draws together: `start_epoch` is set once, so `elapsed()` and `usd()` span the
#: session and three draws do not buy three ceilings.
C1_CREATE_ATTEMPTS = 1
C1_MAX_HOST_DRAWS = 3


def _bounded(flag: str, lo: int, hi: int):
    """An argparse type that accepts a range, and says why the range is that."""
    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} must be an integer") from None
        if not lo <= value <= hi:
            raise argparse.ArgumentTypeError(
                f"{flag}={value} is outside {lo}..{hi}. Draws replace an "
                "unreachable host inside ONE session and share its ceiling; "
                "they are not extra attempts and not a way to chase stock.")
        return value
    return parse


def require_bounded_acquisition(args) -> None:
    """The same rule where a hand-built namespace cannot slip past the parser.

    Device-canary attempt 1 died at `$0.0603` on an attribute a real parser
    defined and a hand-written namespace did not, so an argparse type alone is
    not the whole guard. This runs during spec construction, before
    `run_session` and therefore before anything can be created.
    """
    attempts = getattr(args, "create_attempts", None)
    if attempts != C1_CREATE_ATTEMPTS:
        raise SystemExit(
            f"refusing to build the C1 session: create_attempts={attempts!r}. "
            "A create-attempt sleeps and asks the same market again, which is "
            f"stock chasing; --create-attempts is fixed at {C1_CREATE_ATTEMPTS}.")
    draws = getattr(args, "host_draws", None)
    if not isinstance(draws, int) or not 1 <= draws <= C1_MAX_HOST_DRAWS:
        raise SystemExit(
            f"refusing to build the C1 session: host_draws={draws!r}. A draw "
            "replaces a host that never became usable, inside one session and "
            f"inside its one ceiling; the batch is capped at {C1_MAX_HOST_DRAWS}.")

#: Dev-box-only assets the launcher scp's. The battery is 3.26 MiB and the
#: reasoning-heavy mixture 0.76 MiB, so both fit the observed 0.44-0.72 MB/s
#: uplink comfortably — unlike a 1.1 GiB checkpoint, which is why the selected
#: leaves became relay pulls after continuation attempt 2.
LOCAL_ASSETS = (
    LocalAsset("artifacts/stage3/c1_confirmation_v1", "c1_confirmation_v1",
               "artifacts/stage3"),
    LocalAsset("artifacts/stage1/reasoning_heavy_v2", "reasoning_heavy_v2",
               "artifacts/stage1"),
    #: C1 reads NEITHER of these. They are staged because the SHARED setup runs
    #: `verify_frozen_assets.py` unconditionally at its ASSETS_READY gate, and
    #: that script checks both. A session declares what the SETUP requires, not
    #: what the session reads — declaring only what it needs is what cost the
    #: device-canary retry $0.0637 and the measurement session $0.0700.
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage3/recovery_search_v2", "recovery_search_v2",
               "artifacts/stage3"),
)


# ---------------------------------------------------------------------------
# prechecks — everything that can refuse before a pod exists
# ---------------------------------------------------------------------------

def c1_harness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Recompute the C1 harness set here, independently of the artifact.

    `require_harness()` digests whatever file list the authorization *stores*.
    That is what lets a session declare its own executable, and it is also the
    one thing an artifact could get wrong in its own favour: a grant carrying
    another phase's file list would verify perfectly against those files while
    this launcher, this driver, the fixed-path replayer and the new ATTENTION
    operator went unmeasured.
    """
    try:
        current = c1_harness_digest(REPO_ROOT)
    except Exception as exc:                       # noqa: BLE001
        return False, f"cannot compute the C1 harness digest: {exc}"
    live = current["digest"]
    expected = tuple(f["path"] for f in current["files"])
    declared = tuple(getattr(ctx.auth, "harness_source_files", ()) or ())
    if declared != expected:
        #: Named separately, because it is the failure an authorization issued
        #: after the initialization cutover actually lands on, and "a different
        #: file set" would send the reader looking for another phase's grant.
        if declared == C1_HARNESS_SOURCE_FILES_V1:
            return False, (
                "the authorization declares the PRE-MIGRATION harness set: "
                f"{len(declared)} paths under src/aadistill/autoinit/, which the "
                "initialization cutover moved. Its digest was computed over the "
                f"{len(expected)} current paths, so the two describe different "
                "sets and session_commit_gate can never pass. Re-issue.")
        return False, ("the authorization declares a different harness file set "
                       f"({len(declared)} paths) than this session executes "
                       f"({len(expected)})")
    stored = getattr(ctx.auth, "harness_source_digest", None)
    if stored and stored != live:
        return False, (f"harness digest {stored[:12]}… in the authorization does "
                       f"not match the live tree {live[:12]}…")
    return True, f"C1 harness {live[:12]}… over {len(expected)} declared files"


def grant_provenance_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The maintainer decision this session runs under must belong to THIS run.

    The authorization records the grant it was issued from — its path and the
    hash of its contents — and until now nothing checked that reference again.
    Three things could go wrong silently, and each of them is a different way of
    running under a decision that was made about something else:

    * the grant could have been edited after issuance, since the authorization's
      own self-hash covers the reference, not the file it points at;
    * it could have been deleted, leaving an authorization whose stated
      provenance cannot be produced on request;
    * it could belong to a *different attempt*. This is the live one. Nine
      grants exist as `logs/autoinit_c1_attempt<N>_grant.json`, all structurally
      valid, and a launch that read one of those would be running attempt 10
      under attempt 6's permission.

    So the grant is required at this run's own `governance/grant.json` — the
    location `open_run` exempts by name and `record_run` gives an owner. That is
    also what makes the flat per-attempt file unnecessary rather than merely
    discouraged: there is nowhere else a grant can be and still pass here.
    """
    run_id = getattr(ctx.args, "run_id", None)
    if not run_id:
        return False, "this session has no run_id, so no grant can belong to it"
    #: Through the helper, so this gate and `open_run` cannot disagree about
    #: where the run is. They did: this line named the pre-stage location while
    #: the run moved under its declared stage.
    rel = (f"{rel_run_dir(RUN_EXPERIMENT_ID, run_id, RUN_STAGE_ID)}"
           f"/{C1_RUN_ROLES['grant']}")
    want = (REPO_ROOT / rel).resolve()
    auth_rel = auth_path_for(run_id)
    try:
        raw = json.loads((REPO_ROOT / auth_rel).read_text())
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot read {auth_rel}: {exc}"
    ref = raw.get("grant") or {}
    stated_path, stated_sha = ref.get("path"), ref.get("sha256")
    if not stated_path or not stated_sha:
        return False, (f"{auth_rel} records no grant path and hash, so the "
                       "decision it was issued from cannot be identified")
    #: Resolved, not string-compared: the issuer stores whatever `--grant` was
    #: typed, and `./logs/...` is the same file as `logs/...`.
    got = Path(stated_path)
    got = (got if got.is_absolute() else REPO_ROOT / got).resolve()
    if got != want:
        return False, (f"the authorization was issued from {stated_path}, which "
                       f"is not this run's grant at {rel}. A grant belongs to "
                       "one attempt; using another attempt's is running under a "
                       "decision made about a different session")
    if not want.is_file():
        return False, (f"{rel} does not exist, so the authorization's stated "
                       "provenance cannot be produced")
    try:
        live = sha256_json(json.loads(want.read_text()))
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot read {rel} as a grant: {exc}"
    if live != stated_sha:
        return False, (f"{rel} hashes to {live[:12]}… but the authorization was "
                       f"issued from {str(stated_sha)[:12]}…; the grant was "
                       "edited after it was used")
    ctx.evidence["grant_provenance"] = {
        "role": "grant", "path": rel, "sha256": live,
        "recorded_by": auth_rel, "run_id": run_id,
        "rule": ("the authorization's grant reference must resolve to THIS "
                 "run's governance/grant.json and hash to the recorded value"),
    }
    return True, f"grant {live[:12]}… at {rel}, as recorded by the authorization"


def frozen_assets_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Run the pod's frozen-asset check HERE, before a pod exists.

    Attempt 10 is the reason, and it cost `$0.1177` plus one of three attempts.
    The setup script has verified the frozen assets since long before C1, and
    there was no dev-box counterpart — so a condition fully decidable on this
    machine was decided on a billing pod instead. The initialization cutover
    relocated two of the scoring contract's six declared files, the contract
    legitimately became `@v3`, the verifier's compiled-in constants still said
    `@v2`, and nothing passed the `--expect` flag that exists for exactly that.
    `SETUP_RC=91`, no driver stage, no probe trained.

    This is the same script, the same expectation document and the same exit
    convention as the pod runs — not a reimplementation of it, which would be a
    second thing to keep in step and would agree with the pod right up until it
    mattered.
    """
    import subprocess

    expect = REPO_ROOT / FROZEN_EXPECT
    if not expect.is_file():
        return False, (f"{FROZEN_EXPECT} is missing; the setup would refuse at "
                       "the frozen-asset gate after a pod exists")
    proc = subprocess.run(
        [sys.executable, "scripts/autoinit/verify_frozen_assets.py",
         "--expect", str(expect)],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "src:scripts"})
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    ctx.evidence["frozen_assets"] = {
        "expect": FROZEN_EXPECT, "returncode": proc.returncode,
        "tail": tail[-12:],
    }
    if proc.returncode != 0:
        #: Verbatim, and for the same reason the setup script says so: "the
        #: verifier could not run" and "these are not the expected assets" are
        #: different findings and must not be reported as one.
        return False, ("the frozen-asset check refuses this tree: "
                       + " ".join(tail[-6:])[:400])
    return True, f"frozen assets verified against {FROZEN_EXPECT}"


def pricing_identity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The grant's ceiling must be the accepted pricing record's, exactly.

    A session whose authorization was written against a different price is not a
    cheaper session — it is an unpriced one.
    """
    try:
        ceiling = c1_hard_ceiling_usd(REPO_ROOT)
    except Exception as exc:                       # noqa: BLE001
        return False, f"cannot read {PRICING}: {exc}"
    granted = float(getattr(ctx.auth, "hard_cap_usd", 0.0) or 0.0)
    if abs(granted - ceiling) > 1e-9:
        return False, (f"the authorization caps at ${granted:.4f} but the accepted "
                       f"pricing record says ${ceiling:.4f}")
    return True, f"ceiling ${ceiling:.4f}, derived from {PRICING}"


def preregistration_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The execution preregistration must exist and describe this tree.

    It is the document a grant binds. If its executable-source digest no longer
    matches the live tree, the session about to run is not the one that was
    registered.
    """
    p = REPO_ROOT / PREREG
    if not p.is_file():
        return False, f"{PREREG} is missing; nothing describes what would run"
    doc = json.loads(p.read_text())

    #: The document's own declared hash, recomputed under the writer's exact
    #: convention: `sha256_json` over the document with `preregistration_sha256`
    #: removed. Without this, only the harness block was checked — so every other
    #: field, including the stage order, the decision rule and the admission rule,
    #: could be edited after freezing and the gate would still pass. The commit
    #: binding makes that hard to do unnoticed; it does not make it impossible.
    stated = doc.get("preregistration_sha256")
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "preregistration_sha256"})
    if not stated:
        return False, "the preregistration declares no preregistration_sha256"
    if stated != recomputed:
        return False, (f"the preregistration declares {stated[:12]}… but its "
                       f"contents hash to {recomputed[:12]}…; it was edited after "
                       "it was written")

    live = c1_harness_digest(REPO_ROOT)["digest"]
    recorded = (doc.get("c1_harness") or {}).get("digest")
    if recorded != live:
        return False, (f"the preregistration records harness "
                       f"{str(recorded)[:12]}… but the tree digests to {live[:12]}…")
    if doc.get("authorizes") != "nothing":
        return False, "the preregistration claims to authorize something"
    return True, (f"preregistration {stated[:12]}… (self-hash verified), harness "
                  f"{live[:12]}…")


def frozen_c1_science_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The scientific constants this session must not have drifted from.

    Cheap, and it has an exact precedent: Phase A's frozen-plan gate exists
    because a plan that changed after it was frozen is not the plan that was
    reviewed.
    """
    problems = []
    seeds = derive_recovery_seeds()
    if seeds != [1635674081, 1656475568, 696460635]:
        problems.append(f"derived seeds moved: {seeds}")
    if CS.EXPECTED_PARENT_DIGEST != (
            "eea90c91346a0745b8b1b847503b48fe73c33bb9d75d92c196dc43598e91e722"):
        problems.append("the parent replay digest moved")
    if CS.EXPECTED_INCUMBENT_DIGEST != (
            "c313d1b4081b9a3b410dddf7a29ebcaad8dd0759179d51e1d761238c1743a2a6"):
        problems.append("the incumbent replay digest moved")
    battery = json.loads((REPO_ROOT / BATTERY_IDENTITY).read_text())
    manifest = json.loads((REPO_ROOT / BATTERY_MANIFEST).read_text())
    if manifest["content_sha256"] != battery["content_sha256"]:
        problems.append("the staged battery is not the frozen one")
    if (manifest["n_prompts"], manifest["n_scorable_prompts"]) != (950, 850):
        problems.append(f"battery is {manifest['n_prompts']}/"
                        f"{manifest['n_scorable_prompts']}, want 950/850")
    if problems:
        return False, "; ".join(problems)
    return True, ("seeds, both replay digests and the 950/850 battery "
                  f"{battery['content_sha256'][:12]}… are unchanged")


def teacher_binding_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The teacher's expected shard hashes must be on hand before the fetch.

    Checked WITHOUT fetching: the point is that the session knows what it will
    demand before it spends anything demanding it. Phase A/B never bound the
    root teacher at all — `root_teacher_sha256` is all zeros in the whole search
    journal — so this gate is closing real inherited debt.
    """
    p = REPO_ROOT / TEACHER_BINDING
    if not p.is_file():
        return False, f"{TEACHER_BINDING} is missing"
    b = json.loads(p.read_text())
    if b["revision"] != CS.TEACHER_REVISION:
        return False, (f"the binding pins {b['revision'][:12]}… but the session "
                       f"declares {CS.TEACHER_REVISION[:12]}…")
    shards = b.get("expected_shard_sha256") or {}
    if len(shards) != b.get("n_shards") or not shards:
        return False, "the binding does not carry a hash for every shard"
    bad = [k for k, v in shards.items() if not (isinstance(v, str) and len(v) == 64)]
    if bad:
        return False, f"shards without a usable sha256: {bad}"
    return True, (f"{len(shards)} teacher shards bound at "
                  f"{b['revision'][:12]}…, none fetched")


def battery_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The battery must be present locally, and be the canonical bytes."""
    canonical = Path(json.loads(
        (REPO_ROOT / BATTERY_IDENTITY).read_text())["canonical_path"])
    local = REPO_ROOT / "artifacts/stage3/c1_confirmation_v1"
    if not local.is_dir():
        return False, f"{local} is missing; the launcher has nothing to stage"
    if not canonical.is_dir():
        return False, f"the canonical copy {canonical} is missing"
    for f in sorted(local.glob("*.jsonl")):
        if sha256_file(f) != sha256_file(canonical / f.name):
            return False, f"{f.name} differs from the canonical copy"
    return True, f"battery staged from bytes identical to {canonical}"


#: Manifest-root-relative first path components a C1 artifact pattern may name.
#: The manifest root is `{REPO}/artifacts`, so anything outside these is either a
#: typo or an attempt to archive something this session does not own.
ARTIFACT_ROOTS = ("audit", "eval", "stage3", "stage1", "autoinit")

#: Classes that cannot exist until recovery training has produced them. A
#: FAILED spec that *requires* any of these blocks teardown on a pod that
#: correctly never trained -- the single most expensive way to be wrong here.
POST_TRAINING_CLASSES = (
    "probe_event_stream", "probe_run_manifest", "probe_run_completion",
    "probe_journal", "probe_config", "per_sample", "scored_probe_aggregate",
    "generations", "generation_summary", "probe_train_tail", "decision",
)


def rope_input_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Does the shared setup have the input its `ROPE_OK` step requires?

    Attempt 2 passed all nine gates, reached `TEACHER_READY` on the pod, and died
    at `ROPE_OK` with `no staged checkpoint to check` — the setup globs
    `artifacts/stage1/*/checkpoint/config.json` and C1 staged only tokenizer
    sidecars there. `$0.1013` for a missing 1,418-byte file.

    This is NOT a replacement for that pod-side check, which is the thing that
    actually proves the RoPE base resolves under both runtimes. It proves only
    that the input exists, is the canonical object, and carries the right base —
    before a pod exists.

    Read-only, and it downloads no weights.
    """
    import tempfile

    if not C1_ROPE_INPUT:
        return False, "the session declares no RoPE config input"
    entry = C1_ROPE_INPUT[0]
    if entry.dest != C1_ROPE_CHECKPOINT_DIR:
        return False, (f"the RoPE config stages to {entry.dest!r}, not the "
                       f"{C1_ROPE_CHECKPOINT_DIR!r} the shared setup globs")
    if not (entry.sha256 or "").strip():
        return False, "the RoPE config input carries no pinned sha256"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            got = hf_download(RELAY_REPO_ID, entry.path, Path(tmp))
            data = Path(got).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry.sha256:
                return False, (f"the relay's {entry.path} hashes to {digest}, "
                               f"pinned {entry.sha256}")
            cfg = json.loads(data)
            from transformers import AutoConfig

            from aadistill.models.student import stored_rope_base
            loaded = AutoConfig.from_pretrained(str(Path(got).parent))
            base = stored_rope_base(loaded)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"the RoPE config input is not usable: {exc}"

    if abs(base - C1_ROPE_BASE) > 1:
        return False, (f"the staged config records RoPE base {base:,.0f}, not "
                       f"{C1_ROPE_BASE:,.0f}; the pod's ROPE_OK step would refuse it")
    ctx.evidence["rope_input_check"] = {
        "relay_path": entry.path, "dest": entry.dest, "bytes": len(data),
        "sha256": digest, "stored_rope_base": base,
        "model_type": cfg.get("model_type"),
        "weights_downloaded": False,
        "note": ("proves the shared setup HAS its input; the pod-side ROPE_OK "
                 "check still runs it through both venvs"),
    }
    return True, (f"{entry.path} ({len(data)} bytes, {digest[:12]}…) stages to "
                  f"{entry.dest} with stored RoPE base {base:,.0f}")


def renderer_parity_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Is the C1 battery still rendered exactly as every historical measurement?

    The seven parametrized parity cases in `tests/data/test_c1_battery.py` used to
    carry this guarantee alone. They need the pinned Hugging Face source snapshots
    — a dev-box readiness input, never a C1 runtime or scientific one — so on a
    pod they could only ever fail, and on 2026-09-04 fourteen of them did, at the
    setup test gate, for `$0.3482` with no scientific stage run.

    Making them skip where the sources are absent would retire the guarantee if
    nothing replaced it. This replaces it, on the one host that can prove it and
    before a provider exists: all seven snapshots present, all seven groups
    re-rendered byte for byte, zero skips. A skip is a refusal here.

    Executes the check live rather than reading a stored verdict — it costs a few
    seconds, and a recorded parity result is exactly as stale as the last time
    somebody remembered to regenerate it.
    """
    try:
        from renderer_parity_gate import gate_verdict, run_parity

        record = run_parity()
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot run the renderer parity check: {exc}"
    ok, reason = gate_verdict(record)
    ctx.evidence["renderer_parity"] = {
        "verdict": "PASS" if ok else "FAIL",
        "counts": record["counts"],
        "resolved_hub_cache": record["resolved_hub_cache"],
        "groups": [{k: g[k] for k in ("group", "status", "repo_id", "revision",
                                      "file", "resolved_snapshot", "n_frozen",
                                      "n_checked")}
                   for g in record["groups"]],
        "note": ("the seven pinned dataset snapshots are a dev-box readiness "
                 "input; they are NOT staged to the pod and no C1 number reads "
                 "them"),
    }
    return ok, reason


def pod_environment_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Has this exact executable been proved to survive a pod's test gate?

    Attempt 3R is the reason. It cleared `VLLM_READY → TEACHER_READY → ROPE_OK`
    and then died on a CPU test suite that had never been run under the conditions
    a fresh pod is in — empty `$HOME`, its own `HF_HOME`, no dataset cache, no
    credential file. Of the fourteen, 7 renderer-parity and 2 repository-state
    failures were identified; the other 5 remain UNEXPLAINED. They were once
    attributed to leaf transport, but that $0 reproduction ran with no
    HF_TOKEN — a state no pod is in — and does not reproduce.

    The sweep that answers this takes about thirteen minutes, far too slow to run
    while a pod bills. So it is run once against a committed tree and recorded,
    and this gate checks that the recording still describes the code that would
    run: the C1 harness digest and the pod test-environment digest must both still
    match, and the session commit must descend from `swept_base_commit` with no
    tracked path changed beyond the record itself and this session's
    authorization.

    **And the record must be `launch_bound`.** A `diagnostic` record proves the
    machinery works on some tree; it is genuine readiness evidence and stays valid
    as such, but it is not the sweep a paid launch rests on. Until 2026-09-05 this
    gate accepted either and merely copied the kind into evidence, which meant the
    promised launch-bound sweep need never have happened — the distinction existed
    only in prose. Requiring it here is the whole enforcement: `verify_record`
    takes `required_kind`, so a diagnostic record still verifies for anyone asking
    whether the tree is sound, and refuses only the thing that creates a provider
    resource.
    """
    #: THIS run's readiness record. It was a single repository-root file that
    #: each run in turn overwrote, so the evidence a session launched under sat
    #: at a path the next session would replace, and each closeout copied it
    #: away afterwards to keep a copy. It is now produced into the run that owns
    #: it, and the root file is a pointer.
    run_id = getattr(ctx.args, "run_id", None)
    record_rel = pod_env_record_for(run_id, RUN_STAGE_ID)
    try:
        record = load_pod_env_record(REPO_ROOT, run_id=run_id,
                                     stage_id=RUN_STAGE_ID)
    except FileNotFoundError:
        return False, (f"{record_rel} does not exist: no pod-like sweep has "
                       "been recorded for this run. A sweep is taken on the "
                       "clean pre-authorization tree and written into the run "
                       "it is for.")
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot read {record_rel}: {exc}"

    # The session commit is what the pod checks out, so it — not this working
    # tree — is what must descend from the swept base. This run's authorization
    # is permitted
    # because an issued session commits its authorization after the sweep by
    # construction; it is the same single path the session-lineage gate allows.
    # The staged view the sweep ran under must be the one THIS session stages.
    # Attempt 4's sweep used the simulator's generic default and modelled a
    # machine 55 tests more generous than the pod.
    try:
        live_staging = derive_contract(spec(ctx.args).setup,
                                       session_id="autoinit-c1")["digest"]
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot derive this session's staging contract: {exc}"

    ok, reason = verify_pod_env_record(
        record, REPO_ROOT,
        contract=c1_record_contract(run_id, RUN_STAGE_ID),
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
        "permitted_post_sweep_paths": [record_rel,
                                       auth_path_for(run_id)],
        "live_staging_contract_digest": live_staging,
        "recorded_staging_contract_digest": record.get("staging_contract_digest"),
        "counts": record.get("counts"),
        "renderer_parity_skips": record.get("renderer_parity_skipped_as_expected"),
        "leaf_transport_all_passed": record.get("leaf_transport_all_passed"),
        "reason": reason,
    }
    return ok, reason


def bundle_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Can a pod, RIGHT NOW, obtain the exact authorized code?

    Attempt 1 answered every other question correctly and died at `SETUP_RC=1`
    fetching `transfer/c1`, an alias for nothing. Eight gates verified the
    *contents* of the session commit; none asked whether the pod could reach it.

    So this one operates on the object the pod would actually fetch. It derives
    the canonical name from `--session-commit`, downloads that exact relay
    object, hashes it against the staged bundle, `git bundle verify`s the
    round-tripped bytes, clones them, and requires the checkout to be the exact
    session commit, to carry the exact authorization the launcher is loading, and
    to digest to the authorized harness value.

    Read-only: it uploads nothing and mutates nothing. Preparation is
    `scripts/autoinit/stage_c1_bundle.py`, deliberately a separate command, so
    that what this verifies is the relay's state rather than a side effect of the
    verification.
    """
    import tempfile

    commit = ctx.args.session_commit
    try:
        require_canonical_bundle_arg(ctx.args.bundle, commit)
    except C1BundleError as exc:
        return False, str(exc)

    bundle_rel = bundle_record_for(getattr(ctx.args, "run_id", None))
    staged = REPO_ROOT / bundle_rel
    if not staged.is_file():
        return False, (f"{bundle_rel} is missing; run "
                       f"scripts/autoinit/stage_c1_bundle.py --session-commit "
                       f"{commit} first")
    record = json.loads(staged.read_text())
    if record.get("session_commit") != commit:
        return False, (f"{bundle_rel} describes a bundle for "
                       f"{str(record.get('session_commit'))[:12]}…, not the session "
                       f"commit {commit[:12]}…")

    auth_rel = auth_path_for(getattr(ctx.args, "run_id", None))
    auth_bytes = (REPO_ROOT / auth_rel).read_bytes()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = roundtrip(
                session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=auth_bytes,
                authorization_path=auth_rel,
                expected_harness_digest=ctx.auth.harness_source_digest,
                harness_files=tuple(ctx.auth.harness_source_files),
                download=hf_download, workdir=Path(tmp))
    except Exception as exc:                                   # noqa: BLE001
        return False, f"the pod could not obtain the authorized commit: {exc}"

    ctx.evidence["bundle_staged_check"] = evidence
    return True, (f"{evidence['canonical_bundle_name']} ({evidence['bytes']} bytes, "
                  f"{evidence['remote_sha256'][:12]}…) round-trips to "
                  f"{evidence['roundtrip_head'][:12]}… carrying this authorization "
                  f"and harness {evidence['roundtrip_harness_digest'][:12]}…")


def artifact_spec_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Both artifact specs must exist, load, stay in-tree and cover the contract.

    `SessionSpec.validate()` only checks that the two spec *path strings* are
    non-empty; the files themselves are first read by `collect_artifacts.py` on
    the pod, at teardown, after the money is spent. Both C1 spec files were in
    fact absent from the tree that passed every other precheck. This gate closes
    that: it is the one place a missing, unparseable, out-of-tree or
    contract-violating evidence declaration can still be free.

    The success minimums are DERIVED -- probes from the session contract, sets
    from the staged battery manifest -- so a battery that gained or lost a set
    moves the required generation count instead of silently accepting a spec
    that would now archive six sevenths of the evidence.
    """
    from collect_artifacts import load_specs

    paths = (SPEC_SUCCESS, SPEC_FAILED)
    #: Requirement and invariant in one line: these files decide what evidence
    #: survives, so they must be inside the set the authorization measures.
    #: Without this, editing an evidence declaration would not move the harness
    #: digest, and a grant would certify a collection policy it never saw.
    unmeasured = [p for p in paths if p not in C1_HARNESS_SOURCE_FILES_V1]
    if unmeasured:
        return False, (f"{unmeasured} decide what evidence survives teardown but "
                       "are outside the measured C1 harness set")
    loaded = {}
    for rel in paths:
        p = REPO_ROOT / rel
        if not p.is_file():
            return False, (f"{rel} is missing; collection would die on the pod "
                           "at teardown, with the evidence still on it")
        try:
            specs = load_specs(str(p))
        except Exception as exc:                       # noqa: BLE001
            return False, f"{rel} is not a spec collect_artifacts accepts: {exc}"
        if not specs:
            return False, f"{rel} declares no entries"
        for s in specs:
            head = s.pattern.split("/", 1)[0]
            if (s.pattern.startswith("/") or ".." in s.pattern.split("/")
                    or head not in ARTIFACT_ROOTS):
                return False, (f"{rel}: pattern {s.pattern!r} leaves the "
                               f"artifact roots {ARTIFACT_ROOTS}")
        loaded[rel] = specs

    n_probes = CS.C1_SESSION_CONTRACT.n_probes
    n_sets = len(json.loads((REPO_ROOT / BATTERY_MANIFEST).read_text())["sets"])
    #: The frozen `evidence_manifest_contract`, as enforceable minimums.
    required_minimums = {
        "probe_event_stream": n_probes,
        "per_sample": n_probes,
        "scored_probe_aggregate": n_probes,
        "training_completion": n_probes,
        "generations": n_probes * n_sets,
        "generation_summary": n_probes * n_sets,
        "arm_identities": 1,
        "replay_record": 1,
        "decision": 1,
        "attested_protocol": 1,
        "session_evidence": 1,
        "probe_results": 1,
        "device_handoff": 1,
        "engine_probe": 1,
    }
    have = {}
    for s in loaded[paths[0]]:
        if s.required:
            have[s.artifact_class] = max(have.get(s.artifact_class, 0),
                                         s.min_matches)
    gaps = [f"{k} requires {have.get(k, 0)}, contract needs {v}"
            for k, v in required_minimums.items() if have.get(k, 0) < v]
    if gaps:
        return False, f"{paths[0]} does not cover the evidence contract: " \
                      + "; ".join(gaps)

    presupposed = sorted({s.artifact_class for s in loaded[paths[1]]
                          if s.required and s.min_matches > 0
                          and s.artifact_class in POST_TRAINING_CLASSES})
    if presupposed:
        return False, (f"{paths[1]} requires {presupposed}, which cannot exist "
                       "before training; a replay mismatch would be unable to "
                       "tear down")
    return True, (f"success spec covers the contract ({n_probes} probes x "
                  f"{n_sets} sets = {n_probes * n_sets} generation files); "
                  f"failure spec presupposes no training")


# ---------------------------------------------------------------------------

def driver_command(ctx: SessionContext, plan) -> str:
    """The C1 driver. There is no argument that searches, ranks or eliminates.

    And no `--stage` either. C1 attempt 7 cleared the pod CPU test gate -- the
    first attempt ever to -- and then died at `$0.4231` because this function
    emitted `--stage all` and the driver's parser has no such option: argparse
    exited 2 before a line of the driver ran.

    The flag is REMOVED rather than accepted, because `run()` already executes
    the complete fixed sequence B -> C -> DE -> F -> G -> H -> I. Adding
    `--stage` to the parser, even as a no-op, would create a stage-selection
    surface, and partial C1 execution is not an authorized control.

    `test_the_launcher_driver_cli_seam.py` now parses this exact string with the
    driver's OWN parser, which is the authority on what it accepts.
    """
    return (f"/opt/train/bin/python {REPO}/scripts/pod/autoinit_c1_driver.py "
            f"--image-digest '{ctx.image_digest}' "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f}")


def probe_streams(ctx: SessionContext) -> tuple[str, ...]:
    """Every probe's training event stream must come home before teardown."""
    return tuple(f"artifacts/stage3/c1/{arm}_{seed}/train_log.jsonl"
                 for arm in ("incumbent", "treatment")
                 for seed in derive_recovery_seeds())


def spec(args) -> SessionSpec:
    #: Before anything can be created. A namespace that would permit a second
    #: provider resource never becomes a SessionSpec.
    require_bounded_acquisition(args)
    return SessionSpec(
        session_id="autoinit-c1",
        schema="aadistill.autoinit.c1_session/v1",
        description=("Phase C1: fixed-path ATTENTION isolation. Replays the frozen "
                     "fe9683 path under two digest gates, then runs 2 arms x 3 "
                     "fresh seeds. Runs no search"),
        authorization_path=auth_path_for(getattr(args, "run_id", None)),
        #: The C1 type. A Phase-A, Phase-B or continuation artifact is refused by
        #: schema at load: each measures a different harness and carries a
        #: ceiling derived for different work.
        authorization_loader=C1Authorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id="autoinit.v1.phase_c1",
        #: The isolation plan's own hash, not Phase A's. C1 is different science,
        #: not a different operational identity for the same science.
        plan_hash=_plan_hash(),
        #: DERIVED from logs/stages/stage-1/phase_c1/plans/phase_c1_pricing.json. Never written here.
        budget=c1_budget_spec(REPO_ROOT),
        setup=SetupManifest(
            relay_inputs=(*RECOVERY_LADDER, *CALIBRATION_V1,
                           *C1_EVAL_TOKENIZER, *C1_ROPE_INPUT),
            local_assets=LOCAL_ASSETS,
            #: Declared. Without it setup falls to SESSION_KIND=spend and loads a
            #: SpendAuthorization, which refuses this artifact — the session
            #: would die at setup, exit 98, before any work.
            #:
            #: `SESSION_FROZEN_EXPECT` names the document the frozen-asset gate
            #: checks against. Empty for every other session, which keeps asking
            #: the historical question with the verifier's compiled-in
            #: constants; C1 runs on the migrated tree, where the scoring
            #: contract legitimately reads `@v3` because the cutover relocated
            #: two of its six declared files. Attempt 10 died on exactly that
            #: for `$0.1177` — `SETUP_RC=91`, no driver stage, no probe trained
            #: — because the `--expect` mechanism existed and nothing passed it.
            env={"SESSION_KIND": "c1",
                 "SESSION_FROZEN_EXPECT": FROZEN_EXPECT},
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
        driver_job_id="autoinit_c1",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="ALL_DONE",
            failure=("C1_FAILED", "C1_REPLAY_MISMATCH", "C1_INCOMPLETE"),
            incomplete=("C1_INCOMPLETE",),
            #: NEUTRAL, deliberately. `SessionRunner` prints this for ANY marker
            #: in `failure`, so the previous text described a replay mismatch
            #: whatever had actually happened — and on attempt 8 it did: stage D
            #: raised `KeyError: 'input_ids'` before a single digest was
            #: computed, and the launcher announced that the frozen path had not
            #: reproduced its recorded digest. A false claim about reproducibility
            #: for exactly the property C1 exists to test.
            #:
            #: A replay mismatch is asserted in ONE place, by the code that
            #: observed one: `C1Driver.replay_mismatch` writes
            #: `c1_replay_record.json` with the expected and actual digests, reads
            #: it back, and only then emits `MARKER:C1_REPLAY_MISMATCH`. That path
            #: is unchanged. This line must never anticipate it.
            failure_note=(
                "a blocking C1 stage failed — collecting evidence, then tearing "
                "down. Classify from the explicit terminal marker and the "
                "collected driver evidence; this line asserts nothing about "
                "which stage failed or why.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_c1",
            evidence_filename="c1_evidence.json",
            archive_basename="c1_artifacts.tar.gz",
            spec_success=SPEC_SUCCESS,
            spec_failed=SPEC_FAILED,
            report_names=("c1_evidence.json",
                          "c1_attested_evaluation_protocol.json",
                          "c1_replay_record.json", "c1_arm_identities.json",
                          "c1_probe_results.json", "c1_decision.json"),
            event_streams=probe_streams),
        teardown=TeardownPolicy(note="nothing chains off C1"),
        # A setup abort deletes the pod before artifact collection and shows only
        # `tail -40`, so attempt 5's 99 skip identities died with the pod and one
        # sweep/pod divergence is still unexplained. This is small and is read
        # while the pod still exists.
        setup_failure_files=("/workspace/pytest_outcomes.json",
                             "/workspace/pytest_junit.xml",
                             "/workspace/pytest.log"),
        precheck=(
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", None)),
                                check_lineage=True),
            grant_provenance_gate,
            c1_harness_gate,
            pricing_identity_gate,
            preregistration_gate,
            frozen_c1_science_gate,
            frozen_assets_gate,
            teacher_binding_gate,
            battery_staged_gate,
            artifact_spec_gate,
            rope_input_gate,
            bundle_staged_gate,
            renderer_parity_gate,
            pod_environment_gate,
        ),
        evidence_fields={
            "c1_session_contract_hash": CS.C1_SESSION_CONTRACT.contract_hash,
            "runs_a_search": False,
            "eliminates_arms": False,
            "arms": 2, "seeds": 3, "probes": 6,
            "expected_parent_digest": CS.EXPECTED_PARENT_DIGEST,
            "expected_incumbent_digest": CS.EXPECTED_INCUMBENT_DIGEST,
            "formal_recovery_evidence": "OUT OF SCOPE",
            "followon_started": False,
            "followon_reachable_from_this_launcher": False},
    )


def _plan_hash() -> str:
    """The frozen C1IsolationPlan's hash, rebuilt rather than transcribed."""
    from experiments.phase_c1.isolation import C1Arm, C1IsolationPlan
    from aadistill.initialization.operators import attention_activation

    attention_activation.register(replace=True)
    battery = json.loads((REPO_ROOT / BATTERY_IDENTITY).read_text())
    return C1IsolationPlan(
        plan_id="autoinit.v1.phase_c1",
        arms=(C1Arm("c1.incumbent", "incumbent", *CS.INCUMBENT_ATTENTION),
              C1Arm("c1.treatment", "treatment", *CS.TREATMENT_ATTENTION)),
        seeds=tuple(derive_recovery_seeds()),
        battery_asset_id=battery["asset_id"],
        battery_content_sha256=battery["content_sha256"]).plan_hash


def build_parser():
    """C1's session command line. No search flag exists to be set."""
    import os

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    #: REQUIRED, and there is no `--out` to point somewhere else. A default
    #: would only help if every future launch command remembered to override it,
    #: which is how nine attempts came to share one session-record path; naming
    #: the run is now the same act as launching it.
    ap.add_argument("--run-id", required=True, action=_RunIdSetsOut,
                    help="this attempt's id under logs/runs/"
                         f"{RUN_EXPERIMENT_ID}/, e.g. attempt10. Lowercase "
                         "letters, digits and underscores. Refused if that run "
                         "already has a manifest. Also derives the session "
                         "record's path; there is no --out")
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True,
                    help="must be the canonical name derived from\n"
                         "--session-commit; an alias fails at $0")
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    # Derived from the hash-verified pricing record, not re-declared. The
    # literal 0.99 here survived the reprice to the accepted secure L40S
    # rate of 1.09; it was never a spend risk, because the runner refuses a
    # live quote ABOVE --max-price before any provider resource exists, so a
    # stale low value can only refuse a launch. But it meant the operator had
    # to remember the real rate on the command line, and the price lived in
    # two places. A quote above the accepted rate still aborts at $0 and
    # returns for review: this default does not chase the market upward.
    ap.add_argument("--max-price", type=float,
                    default=c1_price_per_hour_usd(REPO_ROOT))
    #: Two arm materializations plus six probe checkpoints and their generations.
    ap.add_argument("--disk-gb", type=int, default=200)
    ap.add_argument("--probe-train-minutes", type=float, default=70.0)
    ap.add_argument("--probe-battery-minutes", type=float, default=25.0)
    ap.add_argument("--token-src",
                    default=os.path.expanduser("~/.cache/huggingface/token"))
    ap.add_argument("--uv-max-s", type=int, default=1500)
    ap.add_argument("--tests-max-s", type=int, default=2700)
    ap.add_argument("--startup-limit-min", type=float, default=15.0)
    #: ONE. Not a default a launch command has to remember to pass — the type
    #: itself refuses anything else. See `C1_CREATE_ATTEMPTS` / `C1_MAX_HOST_DRAWS`.
    ap.add_argument("--create-attempts",
                    type=_bounded("--create-attempts", 1, C1_CREATE_ATTEMPTS),
                    default=C1_CREATE_ATTEMPTS)
    #: Defaults to the batch cap. A cold host is common enough on this provider
    #: that defaulting to 1 is what made attempt 11 cost an attempt rather than
    #: a draw; all draws share this session's single ceiling.
    ap.add_argument("--host-draws",
                    type=_bounded("--host-draws", 1, C1_MAX_HOST_DRAWS),
                    default=C1_MAX_HOST_DRAWS)
    #: Unreachable with one attempt (`if attempt < create_attempts` is never
    #: true), and kept only because the runner argument contract requires the
    #: field. C1 never sleeps against changing stock.
    ap.add_argument("--create-retry-seconds", type=float, default=300.0)
    ap.add_argument("--setup-timeout-s", type=float, default=5400.0)
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    #: The session is priced at ~14 GPU-hours; the poll limit must outlast the
    #: hard threshold or the launcher would stop watching a billing pod.
    ap.add_argument("--poll-limit-min", type=float, default=1320.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    #: No `--out`. It is `run_id` and the layout, or it is nothing: see
    #: `C1_RUN_ROLES`. `SessionRunner` reads `args.out`, so `open_c1_run` sets it
    #: to the run's own session-record path before the runner is constructed.
    return ap


#: The scratch-relative paths this run WRITES and later collects. Ownership of
#: the scratch root is decided by these alone: a shared model cache or a staged
#: input living beside them neither claims the directory nor blocks it.
#: Plus the per-resource watchdog journals, by pattern: they are named after a
#: pod that does not exist when the claim is made, and they are exactly the
#: evidence that a scratch root belonged to a run.
RUN_OUTPUTS: tuple[str, ...] = (*(source for source, _ in _RUN_COLLECT),
                                "watchdog_*.jsonl")


def layout_for_run(repo_root: Path | str, run_id: str):
    """This attempt's layout, without touching the filesystem."""
    return layout_for(repo_root, RUN_EXPERIMENT_ID, run_id,
                      stage_id=RUN_STAGE_ID)


def open_c1_run(args, repo_root: Path | None = None):
    """Claim this attempt's outputs, create its run directory, point `out` at it.

    Runs BEFORE `SessionSpec` construction and therefore before any provider
    call, so a run id that collides with a recorded run — or a scratch root that
    belongs to a different attempt — costs `$0` rather than being discovered
    after a pod exists. `SessionRunner.save()` writes `args.out` without
    creating its parent, which is the other reason this happens first.

    The scratch claim comes FIRST. `logs/runs/<run_id>/` and `--scr` are two
    independent output locations, and only the first was ever checked: a fresh
    run id aimed at attempt 9's scratch passed the run-directory rule and then
    collected attempt 9's driver evidence as its own.
    """
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    claim_output_root(args.scr, RUN_EXPERIMENT_ID, args.run_id,
                      outputs=RUN_OUTPUTS)
    layout = open_run(repo_root, RUN_EXPERIMENT_ID, args.run_id,
                      roles=C1_RUN_ROLES, prepared=_RUN_PREPARED,
                      stage_id=RUN_STAGE_ID)
    for source, role in _RUN_GOVERNANCE:
        src = repo_root / source
        if src.is_file():
            shutil.copy2(src, layout.path(C1_RUN_ROLES[role]))
    #: Describe the directories as they are created. Documentation only: the
    #: manifest stays the canonical index, and a README neither counts as a
    #: produced role nor makes an unexecuted run look like a failed one.
    write_run_readmes(layout, experiment_id=RUN_EXPERIMENT_ID, run_id=args.run_id,
                      stage_id=RUN_STAGE_ID, roles=C1_RUN_ROLES)
    #: Idempotent for a parser-built namespace, and the whole answer for a
    #: hand-built one. Same rule either way -- see `session_record_path`.
    args.out = session_record_path(args.run_id)
    return layout


def close_c1_run(layout, args, repo_root: Path | None = None) -> dict:
    """Collect the small evidence beside the pod, then write the run manifest.

    Runs after the session on every path, including a launcher error, because
    the runner already caught that and saved. What it cannot cover is the
    launcher process itself dying: then the run directory exists with no
    manifest, which `record_run_index.py` reports as an unrecorded run rather
    than silently omitting.
    """
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    scr = Path(args.scr)
    #: Asked again here, not assumed from the open. The two happen at opposite
    #: ends of a session, and what is collected has to be what THIS execution
    #: produced -- otherwise a foreign scratch turns a run that failed before
    #: its driver started into a manifest full of someone else's evidence.
    require_output_claim(scr, RUN_EXPERIMENT_ID, args.run_id)
    for source, role in _RUN_COLLECT:
        src = scr / source
        if src.is_file():
            shutil.copy2(src, layout.path(C1_RUN_ROLES[role]))
    #: Every resource's watchdog evidence, by the pod id in its name. Resolved
    #: by glob rather than listed, because how many resources a session held is
    #: only known once it has ended.
    wd = layout.path(C1_RUN_ROLES["watchdog_journal"])
    wd.mkdir(parents=True, exist_ok=True)
    for src in sorted(scr.glob("watchdog_*.jsonl")) + sorted(scr.glob("watchdog_*.out")):
        shutil.copy2(src, wd / src.name)
    session = json.loads((repo_root / args.out).read_text())
    return record_run(
        layout, spec=C1_RUN_SPEC,
        plan={"session_id": session.get("session_id"),
              "plan_hash": session.get("session_plan_hash"),
              "session_commit": args.session_commit,
              "bundle": args.bundle,
              "preregistration": PREREG,
              "scratch_root": str(scr),
              "scratch_note": ("the artifact archive and the extracted tree stay "
                               "here; artifacts/manifest.json carries their "
                               "hashes. Large artifacts are not moved into git")},
        implementation={"launcher": "scripts/pod/autoinit_c1_launch.py",
                        "harness_source_digest": session.get(
                            "harness_source_digest"),
                        "authorization": auth_path_for(
                            getattr(args, "run_id", None))},
        status={"passed": session.get("passed"),
                #: `terminal`, spelled the way the runner writes it. A key the
                #: record does not have would read as `None` and look like a
                #: session that produced no marker.
                "terminal": session.get("terminal"),
                "pod_id": session.get("pod_id") or None,
                "cost": session.get("cost"),
                "provider_confirms_gone": session.get("provider_confirms_gone"),
                "authorizes": "nothing"},
        roles=present_roles(layout, C1_RUN_ROLES))


#: Exit code for "the session finished, the run did not get recorded".
#:
#: Distinct from the session's own codes, and it never overwrites one: a session
#: that already failed keeps its result, because the pod outcome is what an
#: operator acts on. But an unrecorded run is not a silent condition either --
#: it is exactly the state `open_run` refuses to reopen, so a successful session
#: that could not record itself must not exit 0.
RUN_NOT_RECORDED = 12


def main() -> int:
    args = build_parser().parse_args()
    layout = open_c1_run(args)
    rc = run_session(spec(args), args, REPO_ROOT,
                     summary=("STOP for review. C1 replayed one frozen path "
                              "under two digest gates and ran no search."))
    try:
        doc = close_c1_run(layout, args)
    except Exception as exc:                                      # noqa: BLE001
        print(f"\nRUN NOT RECORDED: {type(exc).__name__}: {exc}\n"
              f"  the run directory is "
              f"{rel_run_dir(RUN_EXPERIMENT_ID, args.run_id, RUN_STAGE_ID)}; "
              "it holds "
              "whatever the session produced and has no manifest. Do not reuse "
              "this run id.")
        return rc or RUN_NOT_RECORDED
    print(f"run {doc['experiment_id']}/{doc['run_id']} recorded — "
          f"{len(doc['roles'])} role(s) under "
          f"{rel_run_dir(doc['experiment_id'], doc['run_id'], RUN_STAGE_ID)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
