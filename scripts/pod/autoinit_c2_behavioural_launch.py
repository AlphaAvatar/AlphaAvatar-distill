#!/usr/bin/env python3
"""Launch ONE Phase-C2 behavioural selection session. Twelve probes, one verdict.

    python scripts/pod/autoinit_c2_behavioural_launch.py --run-id <id> \
        --bundle <name> --max-price <usd/h> [--dry-run]

What it runs, and the only thing it can run: six arms built from the teacher
along digest-pinned paths, six screening probes on one preregistered seed, a
mechanical ranking that advances exactly one candidate, six confirmation probes
on three disjoint preregistered seeds, and one verdict under the frozen Phase-C
rule.

**There is no code path to anything else.** Not to a Full Search — the search is
complete and its Top-5 frozen. Not to Search-1 — its beam is a consumed
measurement. Not to a re-measurement of B's state evaluation — that result
stands; this session rebuilds B's BYTES and gates them on the identity that
measurement already froze. Not to the replay — it is closed and its output is an
input here. Not to C3 or C4 — they challenge whatever incumbent this session
leaves and cannot be priced before it has one. Each of those is a property on
`BehaviouralAuthorization` that returns False, and `scope_gate` below refuses an
artifact claiming any of them before a pod exists.

**Durability runs during the session, not at closeout.** Each probe is announced
by the driver the moment it finishes; the poll hook pulls it off-pod and
re-identifies it from the bytes that land, against all six identity fields. C1
attempt 17 trained six probes over ten hours and lost every one because nothing
left the pod until a closeout that never came. Teardown is refused while a
finished probe exists only on the pod — and refused, too, when the evidence
cannot be READ, because "I found no probes" and "no probes were trained" are
different findings and treating the first as the second is how a pod holding
finished work gets deleted with every check green.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("src", "scripts", "scripts/autoinit"):
    if str(REPO_ROOT / _extra) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _extra))

from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, BudgetSpec, ExecutionCommands, LocalAsset, MarkerPolicy,
    SessionContext, SessionSpec, SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate)
from aadistill.infrastructure.session_runner import run_session  # noqa: E402
from aadistill.runtime.cpu_test_env import (  # noqa: E402
    host_local_store,
)
from aadistill.runtime.staging_contract import (  # noqa: E402
    ignores_for_selection)

from autoinit_science_inputs import CALIBRATION_V1, RECOVERY_LADDER  # noqa: E402
from autoinit_c1_launch import C1_EVAL_TOKENIZER, C1_ROPE_INPUT  # noqa: E402
from experiments.deployment import deployment_commands  # noqa: E402
from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_bundle as BT  # noqa: E402
from experiments.phase_c2 import behavioural_continuation as BC  # noqa: E402
from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402
from experiments.phase_c2 import behavioural_pod_environment as BPE  # noqa: E402
from experiments.run_layout import (  # noqa: E402
    ArtifactSpec as RunArtifactSpec, claim_output_root, open_run,
    present_roles, record_run, rel_run_dir, write_run_readmes,
)
from phase_a_frozen import TEACHER_REVISION  # noqa: E402

EXPERIMENT_ID = BPE.EXPERIMENT_ID
STAGE_ID = "1"

#: The POD's layout, and it is the SETUP SCRIPT's, not one invented here.
#: `autoinit_preflight_setup.sh` declares `WS=/workspace` and `REPO=$WS/aad`.
#: A sibling launcher invented `/workspace/aad` and `/workspace/aad/repo` and
#: put its status file under a subdirectory nothing creates, so the script's
#: very first `mark()` failed on a missing directory: setup died at ENV_READY,
#: 1.2 min and $0.02 into a billing pod.
WS = "/workspace"
REPO = f"{WS}/aad"
WORKDIR = f"{REPO}/artifacts/autoinit/c2_behavioural"
ARM_DIR = f"{WORKDIR}/arms"
EVAL_DIR = f"{REPO}/artifacts/eval/c2_behavioural"

AUDIT_DIRNAME = "autoinit_c2_behavioural"
#: Where the driver writes its evidence and where the collector looks. ONE
#: constant, so the two cannot disagree.
AUDIT_DIR = f"{REPO}/artifacts/audit/{AUDIT_DIRNAME}"
STATUS = f"{WS}/autoinit_c2_behavioural.status"
RUN_LOG = f"{WS}/autoinit_c2_behavioural_run.log"

#: This session declares ASSETS_READY, and the setup script refuses a session
#: that declares it and names no expectation — explicit or refused, never
#: inherited. Names the assets THIS session reads: both calibration mixtures,
#: the recovery pack, the evaluation tokenizer and both batteries.
FROZEN_EXPECT = "configs/experiments/phase_c2/behavioural_frozen_assets.json"

#: Where finished probes live off-pod. ONE constant: the fetcher writes here and
#: the preflight reads here, so "is this probe already durable" has one answer.
#:
#: LOCATED THROUGH `$HOME` rather than written absolute. On this dev box the two
#: resolve identically, so nothing about a real launch changes — but an absolute
#: path survives the simulator's fresh empty HOME, and that is what let the
#: launch-bound sweep certify a machine the pod could not reproduce. See
#: `aadistill.runtime.cpu_test_env.host_local_store`, which exists for exactly
#: this and which this module was not using.
DURABLE_STORE = str(host_local_store() / "phase_c2_behavioural")

#: THE PRE-STAGED BACKEND. A provider network volume holding this campaign's
#: completed probes, written by `scripts/autoinit/stage_c2_probes_to_volume.py`
#: while nothing expensive was billing, and attached to every later pod of the
#: campaign.
#:
#: It exists because the alternative is worse in every direction. The bytes are
#: on the launcher host, the host's uplink is ~0.72 MB/s, and 22.2 GiB is
#: therefore ~9 hours — of L40S time, if the transfer happens during the
#: session. Paying the most expensive machine in the budget to watch a slow
#: upload is how a continuation stops fitting its own ceiling. Attaching a
#: volume removes the transfer from the experiment instead of budgeting for it:
#: the probes are simply present when the pod boots.
#:
#: A volume lives in ONE datacenter and a pod can attach it only from there, so
#: naming one CONSTRAINS THE DRAW to that datacenter. That is a real constraint
#: on acquisition, this comment predicted it would bite, and it did: attempt8
#: spent forty minutes being told "no longer any instances available with the
#: requested specifications" by EU-NL-1 and never created a pod.
#:
#: So the attachment is DERIVED FROM NEED, by `volume_attachment` — it is worth
#: a one-datacenter pool exactly when some remaining operation reads the bytes
#: on it, and worth nothing otherwise. A session that attaches it for no
#: consumer pays the whole constraint for none of the benefit, which is the
#: same defect as moving the bytes themselves (AGENTS.md P8.4) one level up: the
#: CONSTRAINT was following the campaign rather than a consumer.
CAMPAIGN_VOLUME_ID = "59qt99zeg5"
CAMPAIGN_VOLUME_GB = 40
VOLUME_DATACENTER = "EU-NL-1"

#: Where the volume is mounted on the pod. NOT `/workspace`, which is the
#: provider's default and this project's checkout root: mounting shared
#: network storage over the working tree would put a session's repository on
#: the volume and let two sessions share it.
VOLUME_MOUNT = "/durable"


def campaign_store(campaign_id: str, store: str | Path = DURABLE_STORE) -> Path:
    """This CAMPAIGN's durable root. The campaign owns its probes, not a run.

    A probe belongs to the twelve-probe experiment, and a replacement resource
    continuing that experiment has to be able to see what the previous one
    produced. Keyed under the run attempt alone — which is what the destination
    used to be — a continuation could not find its own campaign's work.

    Each run attempt still gets its own subdirectory inside. Two attempts of one
    campaign must never write the same bytes: a deterministic pipeline
    reproduces identical unit ids across sessions, so a flat campaign directory
    would let a second attempt silently overwrite the first attempt's verified
    checkpoint with an unverified partial copy.
    """
    return Path(store) / campaign_id


def probe_destination(ctx: SessionContext, unit_id: str) -> Path:
    """Where ONE probe's bytes land. Read by the fetcher and by the gates.

    From `--ckpt-store` rather than from the constant, so the flag that says
    where probes are secured is the flag every consumer reads. `destination_gate`
    checking one volume while the fetcher wrote to another would verify capacity
    on a disk nothing uses.
    """
    return (campaign_store(ctx.auth.campaign_id, ctx.args.ckpt_store)
            / ctx.args.run_id / unit_id)


#: The suffix a dry run's OUTPUT locations take. Owned by
#: `behavioural_governance`, because two separate enumerations of this
#: campaign's run directories have to agree about what is NOT a run.
DRY_RUN_SUFFIX = BG.DRY_RUN_SUFFIX

CONTAINER_DISK_GB = BG.PROVISION_GB

#: The image, pinned. Same family the replay and the search ran on.
BOUND_IMAGE = "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404"

#: SMALL things only. The six arms are NOT here: each is a 1.19 GB checkpoint,
#: the shared runner gives every local asset a hardcoded 600-second scp timeout
#: against a dev-box uplink measured at 0.44-0.79 MB/s, and continuation attempt
#: 2 died staging exactly this size of artifact — "arithmetic rather than luck".
#: They are built on the pod instead; see `BG.CANDIDATE_TRANSPORT`.
LOCAL_ASSETS = (
    *BG.staged_assets(REPO_ROOT),
    #: Both batteries. 3.26 MiB each, comfortably inside the scp timeout.
    LocalAsset("artifacts/stage3/c1_confirmation_v1", "c1_confirmation_v1",
               "artifacts/stage3"),
    LocalAsset("artifacts/stage3/c2_screening_v1", "c2_screening_v1",
               "artifacts/stage3"),
    #: Read by NEITHER rung. Staged because the SHARED setup runs
    #: `verify_frozen_assets.py` unconditionally at ASSETS_READY and that script
    #: checks both. A session declares what the SETUP requires, not only what it
    #: reads — declaring only what it needed cost two sibling sessions a pod
    #: each.
    LocalAsset("artifacts/stage1/state_eval_v1", "state_eval_v1",
               "artifacts/stage1"),
    LocalAsset("artifacts/stage3/recovery_search_v2", "recovery_search_v2",
               "artifacts/stage3"),
)

POD_TEST_SELECTION = BPE.POD_TEST_SELECTION
TEST_IGNORES = ignores_for_selection(POD_TEST_SELECTION, REPO_ROOT)

#: Every path this run writes, by role. ONE mapping, so the launcher, the
#: collector and the closeout cannot disagree about where a thing lives.
BEHAVIOURAL_RUN_ROLES: dict[str, str] = {
    #: --- governance: inputs, prepared before the run opens -----------------
    "grant": "governance/grant.json",
    "readiness_record": "governance/readiness.json",
    "authorization": "governance/authorization.json",
    "bundle_record": "governance/bundle.json",
    #: --- runtime: how it executed ------------------------------------------
    #: Written on EVERY path including a $0 pre-provider refusal, which is why
    #: it is the one role the manifest requires.
    "session_record": "runtime/session.json",
    "launcher_log": "runtime/launcher.log",
    "watchdog_journal": "runtime/watchdog",
    #: --- evidence: what it produced -----------------------------------------
    "driver_log": "evidence/driver_run.log",
    "driver_status": "evidence/driver_status.txt",
    "session_evidence": "evidence/c2_behavioural_evidence.json",
    "screening_ranking": "evidence/c2_screening_ranking.json",
    "decision": "evidence/c2_decision.json",
    #: --- artifacts / closeout ------------------------------------------------
    "artifact_manifest": "artifacts/manifest.json",
    "outcome": "closeout/outcome.json",
}

BEHAVIOURAL_RUN_SPEC = RunArtifactSpec(
    spec_id="phase_c2_behavioural_session_v1",
    required=("session_record",),
    optional=tuple(r for r in BEHAVIOURAL_RUN_ROLES if r != "session_record"))

_RUN_PREPARED = ("grant", "readiness_record", "authorization", "bundle_record")


class _RunIdSetsOut(argparse.Action):
    """`--run-id` also produces `out`, because the RUNNER reads `out`.

    `SessionRunner.save()` writes `args.out`, and the argument contract requires
    every attribute the runner reads to come from the REAL parser — another
    session died at `$0.0603` on an attribute a hand-written namespace had and
    the parser did not, after the pod was billing.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, value)
        namespace.out = session_record_path(value)


def governance_path(run_id: str, name: str) -> str:
    return f"{rel_run_dir(EXPERIMENT_ID, run_id, STAGE_ID)}/governance/{name}"


def auth_path_for(run_id: str) -> str:
    return governance_path(run_id, "authorization.json")


def bundle_record_for(run_id: str) -> str:
    return governance_path(run_id, "bundle.json")


def session_record_path(run_id: str) -> str:
    return f"{rel_run_dir(EXPERIMENT_ID, run_id, STAGE_ID)}/runtime/session.json"


def runs_root_rel() -> str:
    """Where THIS experiment's run directories live, repo-relative.

    Derived from `rel_run_dir`, the one function that knows the convention.
    `runs_root_for` is the LEGACY root (`logs/runs/stage-1`) and is not where
    `open_run` writes; a gate that enumerated attempts there would find none
    and report a campaign's paid predecessors as absent — the exact shape of
    the bug this enumeration exists to fix.
    """
    return str(Path(rel_run_dir(EXPERIMENT_ID, "_", STAGE_ID)).parent)


# ---------------------------------------------------------------------------
# prechecks — everything that can refuse before a pod exists
# ---------------------------------------------------------------------------

def scope_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The artifact must permit THIS work and nothing adjacent to it.

    Read from the loaded authorization's properties rather than from the
    document's text: a document that disagreed with the object that wrote it
    would be worse than no document.
    """
    auth = ctx.auth
    must_be_true = ("authorizes_behavioural_selection", "allows_recovery_training")
    must_be_false = ("authorizes_c2_full_search", "authorizes_c2_search1",
                     "authorizes_c2_baseline_completion", "authorizes_c2_replay",
                     "authorizes_later_cycles", "allows_phase_a")
    wrong = [f for f in must_be_true if not getattr(auth, f, False)]
    wrong += [f"not-{f}" for f in must_be_false if getattr(auth, f, False)]
    if wrong:
        return False, (
            f"the authorization's scope is wrong for this session: {wrong}. It "
            "must permit the behavioural selection and its recovery training, "
            "and permit no search, no re-measurement of B, no replay and no "
            "later cycle.")
    stages = tuple(getattr(auth, "authorized_stages", ()) or ())
    if stages != BG.AUTHORIZED_STAGES:
        return False, (f"authorized stages {stages} != {BG.AUTHORIZED_STAGES}")
    return True, ("scope OK: behavioural selection with recovery training, and "
                  "nothing else reachable")


def plan_binding_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The authorization's plan hash must equal the live derived plan."""
    live = BG.plan_hash(REPO_ROOT)
    declared = getattr(ctx.auth, "plan_hash", None)
    if declared != live:
        return False, (
            f"the authorization binds plan {declared} but this tree derives "
            f"{live}. The protocol, the six arms' identities or the batteries "
            "have moved since it was issued.")
    return True, f"plan binding OK ({live[:12]}…)"


def source_binding_gate(ctx: SessionContext) -> tuple[bool, str]:
    """Recompute the executable closure here, independently of the artifact.

    `require_harness()` digests whatever file list the authorization STORES.
    That is what lets a session declare its own executable, and it is also the
    one thing an artifact could get wrong in its own favour: a grant carrying
    another phase's file list would verify perfectly against those files while
    this launcher, this driver, the two scorers and the decision module went
    unmeasured.
    """
    try:
        live = BG.current_executable(REPO_ROOT)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"cannot derive the behavioural closure: {exc}"
    expected = tuple(row["path"] for row in live["files"])
    declared = tuple(getattr(ctx.auth, "harness_source_files", ()) or ())
    if declared != expected:
        extra = sorted(set(declared) - set(expected))[:4]
        missing = sorted(set(expected) - set(declared))[:4]
        return False, (
            f"the authorization declares {len(declared)} harness files and this "
            f"tree derives {len(expected)}; missing e.g. {missing}, extra e.g. "
            f"{extra}")
    if getattr(ctx.auth, "harness_source_digest", None) != live["digest"]:
        return False, (
            f"harness digest {getattr(ctx.auth, 'harness_source_digest', None)} "
            f"!= live {live['digest']}")
    return True, f"source binding OK ({live['n_files']} files, {live['digest'][:12]}…)"


def destination_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The durable store must exist and have room BEFORE a probe is trained.

    AGENTS.md: "a durability mechanism needs a backend with the capacity to hold
    what it is protecting." C1 attempt 18 exercised a correct preservation
    mechanism on six 2.22 GiB probes and preserved none of them, because every
    upload was refused for storage quota. The mechanism behaved perfectly and
    nothing survived.
    """
    import shutil

    #: From the FLAG, and the campaign's own root, because that is where
    #: `probe_destination` writes. Checking `DURABLE_STORE` while the fetcher
    #: honoured `--ckpt-store` would verify capacity on a volume nothing uses.
    store = campaign_store(ctx.auth.campaign_id, ctx.args.ckpt_store)
    try:
        store.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(store).free
    except OSError as exc:
        return False, f"the durable store {store} is unusable: {exc}"
    #: THE DURABLE REQUIREMENT, DERIVED. This charged
    #: `total_probes * int(1.11 * 2**30)` -- a hardcoded 1.11 GiB that is the
    #: bf16 size of the INITIALIZATION LEAF, not the fp32 size of a trained
    #: probe. It approved 37.0 GiB of free space against a claimed 13.3 GiB
    #: need whose real value is 26.7 GiB: it passed for the wrong reason, and
    #: with 15 GiB free it would also have passed and then failed mid-campaign
    #: with every earlier probe already durable and irreplaceable.
    #:
    #: Now read from `storage_requirement`'s `durable_backend`, which derives
    #: the SAVE footprint from the recipe's own dtype -- so a recipe that saves
    #: in another precision, or a model family with a different parameter
    #: count, moves this without an edit here.
    sched = BH.schedule(REPO_ROOT)
    req = BH.storage_requirement(BH.candidate_manifest(REPO_ROOT), sched,
                                 REPO_ROOT)
    #: THE PROBES THIS SESSION WILL PRODUCE, not the whole protocol's.
    #:
    #: This charged the full twelve-probe requirement — 26.654 GiB — against
    #: free space, every time. That is right for a fresh campaign and wrong for
    #: a continuation, because the probes already in the store are already
    #: OCCUPYING it: their bytes are counted as a need while being subtracted
    #: from the supply. This campaign holds ten of twelve, `free` is 16.4 GiB,
    #: and the gate refused a session whose real appetite is two probes and
    #: 4.4 GiB.
    #:
    #: The remaining probes come from the same derivation the budget and the
    #: continuation gate use, so a session cannot be admitted for work it was
    #: not funded for or refused for work it does not owe.
    per_probe = int(float(
        req["components_gib"]["_probe_checkpoint_footprint"]["gib"]) * 2**30)
    work = campaign_remaining_work(ctx)
    owed = int(work["n_probes_remaining"])
    need = per_probe * owed
    total = int(float(req["durable_backend"]["gib"]) * 2**30)
    ctx.evidence["durable_destination"] = {
        "store": str(store), "free_bytes": free,
        "probes_owed": owed, "bytes_per_probe": per_probe,
        "need_bytes": need, "whole_protocol_need_bytes": total,
        "_need_is_the_remainder": (
            "the probes this session will produce. A probe already in the "
            "store occupies it; charging its bytes as a need while also "
            "subtracting them from the supply counts them twice and refuses a "
            "continuation whose real appetite is what it still owes."),
    }
    if free < need:
        return False, (
            f"{store} has {free / 2**30:.1f} GiB free and the {owed} probe(s) "
            f"this session still owes need {need / 2**30:.1f} GiB "
            f"({per_probe / 2**30:.3f} GiB each, the save footprint of this "
            "recipe's dtype). A run that trains work it cannot preserve is a "
            "run that will lose it.")
    return True, (f"destination OK: {free / 2**30:.1f} GiB free at {store} for "
                  f"the {owed} probe(s) owed, {need / 2**30:.1f} GiB "
                  f"(of {total / 2**30:.1f} GiB for the whole protocol; the "
                  "rest is already there)")


def container_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The PROVISIONED container disk must hold the peak LOCAL residency.

    The other half of `destination_gate`, and it did not exist. The durable
    requirement was checked against the dev-box store while nothing checked
    the pod's own disk against the work it would do there -- so attempt5 was
    provisioned 120 GB by a model that charged trained probes at half their
    size, and the trainer hit `No space left on device` on probe 11 of 12.

    Two resources, each against its owner: this one is container storage, and
    the probes that must merely SURVIVE teardown are the destination gate's.
    """
    sched = BH.schedule(REPO_ROOT)
    req = BH.storage_requirement(BH.candidate_manifest(REPO_ROOT), sched,
                                 REPO_ROOT)
    res = req["container_residency"]
    peak_gib = float(res["peak_gib"])
    #: The FLAG the pod will actually be created with, not the constant: a
    #: session that overrode it must be checked against what it asked for.
    gb = int(getattr(ctx.args, "disk_gb", CONTAINER_DISK_GB))
    #: GB -> GiB through the repository's recorded conversion, the same one the
    #: provision was derived with. Treating the provider's GB flag as GiB
    #: over-states capacity by 7%, which is the direction that hurts.
    conv = BH.storage_pricing(REPO_ROOT)["gb_versus_gib"]
    have_gib = gb / float(conv["gb_per_gib"])
    ctx.evidence["container_storage"] = {
        "provisioned_gb": gb, "provisioned_gib": round(have_gib, 3),
        "peak_local_residency_gib": peak_gib,
        "peak_at_unit": res["peak_at_unit"],
        "headroom_gib": round(have_gib - peak_gib, 3),
        "retained_bytes_per_probe": res["retained_bytes_per_probe"],
        "release_is_enforced": res["released_on_completion"],
        "_two_resources": (
            "this is CONTAINER storage. The bytes that must outlive the pod "
            "are the destination gate's and are not charged here; charging "
            "them to both is what produced a 140 GB provision request from a "
            "model that had already double-counted them."),
    }
    if have_gib < peak_gib:
        return False, (
            f"the session would provision {gb} GB ({have_gib:.1f} GiB) and its "
            f"peak local residency is {peak_gib:.1f} GiB, at "
            f"{res['peak_at_unit']}. A pod that cannot hold the work is a pod "
            "that fails partway through it, which is how attempt5 lost probe "
            "11 of 12 and the campaign's verdict.")
    return True, (f"container OK: {gb} GB ({have_gib:.1f} GiB) provisioned for "
                  f"a {peak_gib:.1f} GiB peak local residency, "
                  f"{have_gib - peak_gib:.1f} GiB spare")


#: Where the staging tool records what it put on the volume, and the ONLY
#: evidence available at `$0` that the volume holds anything. The gate below
#: reads these; the pod re-checks the bytes themselves.
#:
#: Under the experiment's own `validations/`, beside the CUDA campaigns, and
#: deliberately NOT under `runs/`: pre-staging is engineering infrastructure,
#: not a run attempt, and a directory under `runs/` would be summed into this
#: campaign's settled spend by `campaign_attempts` and
#: `settled_campaign_all_in`. Its cost belongs to the stage envelope, not to
#: the campaign's scientific ceiling.
#:
#: The ROOT is the fact; the glob is derived from it, so the fixture that has
#: to make this directory writable and the gate that reads it cannot end up
#: with two spellings of one path.
#: DERIVED from `rel_run_dir`, the one function that knows the log convention,
#: rather than written out: relocating `logs/` has silently repointed declared
#: paths in this repository before, and a literal here would make the gate read
#: an empty directory and report a staged volume as unstaged.
STAGING_RECORD_ROOT = str(
    Path(rel_run_dir(EXPERIMENT_ID, "_", STAGE_ID)).parent.parent
    / "validations" / "durable-staging")
STAGING_RECORD_GLOB = f"{STAGING_RECORD_ROOT}/*/staging_record.json"


def staged_probe_index(repo_root: str | Path = REPO_ROOT,
                       *, campaign_id: str, volume_id: str) -> dict:
    """What the launcher host believes is on the volume, from staging records.

    Every completed staging run for THIS campaign onto THIS volume, merged by
    probe id with the most recent staging winning. A record that did not reach
    `ALL_STAGED` is ignored: it may have verified some probes, but a run that
    could not confirm its own resource was released is not evidence about what
    survived it.
    """
    out: dict[str, dict] = {}
    records: list[tuple[str, dict]] = []
    for path in sorted(Path(repo_root).glob(STAGING_RECORD_GLOB)):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (rec.get("campaign_id") != campaign_id
                or rec.get("volume_id") != volume_id
                or rec.get("terminal") != "ALL_STAGED"):
            continue
        records.append((str(rec.get("finished_utc") or ""), rec))
    #: BY THE KEY ONLY. `sorted` on the pairs falls through to comparing the
    #: dicts whenever two runs share a timestamp, which raises — and two
    #: staging runs finishing in the same second is ordinary, not exotic.
    for _, rec in sorted(records, key=lambda pair: pair[0]):
        for probe in rec.get("probes") or []:
            if probe.get("verified"):
                out[probe["probe_id"]] = probe
    return out


def volume_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The pre-staged probes must exist BEFORE a pod is drawn. `$0`.

    A continuation's completed probes now arrive by having been written to an
    attached network volume in advance, rather than by being copied onto the
    pod while it bills. That removes ~9 hours of L40S time from every attempt
    and it introduces exactly one new way to fail: launching against a volume
    that does not hold what the campaign needs.

    That failure is cheap here and expensive anywhere else. On the pod it
    surfaces after setup, after the image pull, after the teacher is
    materialized — and it is unrecoverable, because the protocol forbids
    retraining a completed probe, so the session can only abort.

    A campaign that owes no restore passes: a fresh campaign has no probe to
    pre-stage, and demanding a staging record from it would refuse the first
    attempt of every future campaign.
    """
    work = campaign_remaining_work(ctx)
    #: THE PROBES WHOSE WEIGHTS A REMAINING OPERATION READS, and nobody else.
    #: This asked for every probe the campaign held, which made the gate demand
    #: a 22.21 GiB pre-stage to satisfy no consumer: a completed and validly
    #: scored probe's checkpoint is archival evidence, and its rows travel on
    #: the evidence leg. Artifacts follow consumers, not campaigns.
    restorable = list(work["transfer"]["weights"]["probes"])
    if not restorable:
        return True, (
            "volume OK: no probe needs its weights on the pod — "
            f"{work['transfer']['evidence']['n']} completed probe(s) "
            f"contribute {work['transfer']['evidence']['mib']} MiB of "
            f"evidence and their "
            f"{work['transfer']['skipped_weights']['gib']} GiB of checkpoints "
            "have no remaining consumer")

    volume = str(getattr(ctx.args, "network_volume_id", "") or "").strip()
    mount = str(getattr(ctx.args, "volume_mount_path", "") or "").strip()
    centre = str(getattr(ctx.args, "data_center_ids", "") or "").strip()
    if not (volume and mount and centre):
        return False, (
            f"this campaign owes a restore of {len(restorable)} probe(s) and "
            f"the session names volume={volume!r} mount={mount!r} "
            f"datacenter={centre!r}. All three are required: a volume with no "
            "datacenter is drawn where it cannot attach, and a volume with no "
            "mount path defaults to /workspace, the checkout root.")
    if mount == "/workspace":
        return False, (
            "the volume mount path is /workspace, which is this project's "
            "checkout root. Mounting shared network storage over the working "
            "tree would put the session's repository on the volume.")

    staged = staged_probe_index(REPO_ROOT, campaign_id=ctx.auth.campaign_id,
                                volume_id=volume)
    missing = [p for p in restorable if p not in staged]
    if missing:
        return False, (
            f"{len(missing)} of {len(restorable)} probe(s) this campaign must "
            f"restore were never verified onto volume {volume}: "
            f"{missing[:4]}{'…' if len(missing) > 4 else ''}. Stage them with "
            "scripts/autoinit/stage_c2_probes_to_volume.py before launching; "
            "a completed probe may not be retrained, so a pod that finds them "
            "absent can only abort.")
    gib = sum(int(staged[p].get("bytes") or 0) for p in restorable) / 2**30
    return True, (f"volume OK: {len(restorable)} probe(s), {gib:.2f} GiB "
                  f"pre-staged on {volume} in {centre}, mounted at {mount}")


def volume_attachment(args) -> dict:
    """Resolve whether this session attaches the campaign volume. `$0`.

    DERIVED FROM NEED, and it **clears `args`** when the answer is no, rather
    than returning a verdict for a caller to apply. The clearing was a single
    `if` in `main`, which is the one place no test reaches: the gates and
    `SessionRunner.create` both read these args, so the normalization and the
    decision have to be the same step or a mutation to either is invisible.

    Attaching a volume pins the draw to ONE datacenter, and the campaign's
    volume lives in `VOLUME_DATACENTER`. That is worth paying when some
    remaining operation reads the bytes on it, and it is worth nothing at all
    otherwise — attempt8 paid the entire constraint for no consumer and failed
    eight consecutive create calls over forty minutes because EU-NL-1 had no
    L40S. Nothing was created and nothing billed, but the chain was consumed.

    The volume is a RESTORE SOURCE and only that. It is not where a newly
    trained probe is preserved: `_fetch_and_verify` pulls each announced unit
    to `--ckpt-store` on the launcher host and re-identifies it there, so a
    session that trains two fresh probes still needs no volume. Nor does the
    evidence leg, which is copied from the host's durable store. So when no
    probe's weights have a consumer, all three of volume, mount and datacenter
    fall away together and the draw may take an L40S anywhere.

    Reads the durable store a second time — `campaign_remaining_work` caches
    per-context and the context does not exist yet. A probe arriving between
    the two reads would make this skip a volume that `volume_gate` then
    demands, and the gate REFUSES rather than launching without it, which is
    the safe direction.
    """
    state = BC.campaign_state(campaign_store(BG.CAMPAIGN_ID, args.ckpt_store),
                              exclude_attempt=args.run_id)
    work = BC.remaining_work(REPO_ROOT, state=state)
    t = work["transfer"]
    probes = list(t["weights"]["probes"])
    if probes:
        return {
            "attach": True, "probes": probes,
            "why": (f"{len(probes)} probe(s) need their weights on the pod "
                    f"({t['weights']['gib']} GiB), so the session attaches "
                    f"{args.network_volume_id} and accepts the "
                    f"{args.data_center_ids} draw"),
        }
    args.network_volume_id = ""
    args.volume_mount_path = ""
    args.data_center_ids = ""
    return {
        "attach": False, "probes": [],
        "why": ("no remaining operation reads a pre-staged checkpoint — "
                f"{t['evidence']['n']} completed probe(s) contribute "
                f"{t['evidence']['mib']} MiB of evidence and "
                f"{t['skipped_weights']['gib']} GiB of checkpoints have no "
                "consumer — so no volume is attached and the draw is not "
                "pinned to one datacenter"),
    }


def campaign_attempts(campaign_id: str, *, exclude: str = "",
                      store: str | Path = DURABLE_STORE,
                      repo_root: str | Path = REPO_ROOT) -> list[str]:
    """Every prior run attempt of this campaign. THE UNION OF TWO SOURCES.

    Name order, not chronological — `attempt10` sorts before `attempt2` — and
    the gate does not care, because it reconciles every prior resource and sums
    every prior spend rather than looking at the latest one.

    **The durable store is not a resource ledger, and using it as one missed a
    paid pod.** Derived from durable probes alone, this returned `[]` for the
    most ordinary failure there is: attempt 1 creates a pod, bills, and dies in
    setup or arm materialization before the first probe finishes. The next
    attempt then called itself the campaign's *first resource*, neither summing
    attempt 1's spend nor checking whether attempt 1 was provider-confirmed
    released — in direct conflict with R9, whose whole content is that the
    ceiling is cumulative across every resource and subrun and that a previous
    resource must be confirmed non-billing.

    So the authority for *a resource existed* is the campaign's RUN RECORDS,
    and the durable store contributes only the attempts that left science.
    Neither source subsumes the other: a launcher that died before writing its
    manifest can still have left probes, and a pod that died before its first
    probe leaves a record and no probes.

    A run whose manifest names a different campaign is a different experiment
    and is excluded. A run with no readable manifest is INCLUDED, because
    "which campaign was that?" is a question the gate must answer with a
    refusal rather than by omitting the attempt.
    """
    names: set[str] = set()
    root = campaign_store(campaign_id, store)
    if root.is_dir():
        names.update(d.name for d in root.iterdir()
                     if d.is_dir() and any(d.iterdir()))

    runs = Path(repo_root) / runs_root_rel()
    if runs.is_dir():
        for d in runs.iterdir():
            if not d.is_dir():
                continue
            declared = None
            manifest = d / "manifest.json"
            if manifest.is_file():
                try:
                    declared = (json.loads(manifest.read_text()).get("plan")
                                or {}).get("campaign_id")
                except json.JSONDecodeError:
                    declared = None
            if declared is None or declared == campaign_id:
                names.add(d.name)
    names.discard(exclude)
    #: A DRY RUN IS NOT A RESOURCE. `--dry-run` writes to its own run id so it
    #: cannot consume the chain, which puts that directory under `runs/` beside
    #: the real attempts -- and this function reads it as a prior resource whose
    #: spend could not be established, then refuses the launch it had just
    #: rehearsed at $0. A dry run contacts no provider by construction.
    return sorted(n for n in names if not BG.is_dry_run_id(n))


def campaign_remaining_work(ctx: SessionContext) -> dict:
    """What this campaign still owes, from the durable destination. `$0`.

    Cached on the context: the gate, the budget and the restore step must all
    price the SAME state, and re-reading the store between them would let a
    probe arriving mid-launch change the answer under one of them.
    """
    cached = ctx.evidence.get("_remaining_work")
    if cached is None:
        state = BC.campaign_state(
            campaign_store(ctx.auth.campaign_id, ctx.args.ckpt_store),
            exclude_attempt=ctx.args.run_id)
        cached = BC.remaining_work(REPO_ROOT, state=state)
        cached["_state"] = state
        ctx.evidence["_remaining_work"] = cached
    return cached


def continuation_all_in_usd(ctx: SessionContext, work: dict) -> float:
    """All-in dollars for the REMAINING work at the authorized rate.

    GPU and separately billed container disk, summed at the end, over the
    remaining-work hard window rather than a fresh full session's. Reserving a
    whole session per attempt — which this gate did — refused every
    continuation by construction: with `approved` sized for one session, any
    prior spend above `$0` made `settled + approved > approved`.
    """
    minutes = float(work["decomposition"]["hard_minutes"])
    gpu = minutes / 60.0 * float(ctx.auth.rate_usd_per_hour)
    #: The disk rate the authorization was derived at, re-derived from its own
    #: figures rather than re-quoted: `disk_hard_usd` is what the full window
    #: costs, so the per-minute rate is that over the full window.
    disk_per_minute = (float(ctx.auth.disk_hard_usd)
                       / float(ctx.auth.hard_runtime_minutes))
    return gpu + disk_per_minute * minutes


def prior_attempt_actual(ctx: SessionContext, attempt: str) -> dict:
    """One predecessor's ACTUAL all-in spend, GPU and disk. Or UNKNOWN.

    `SessionRunner` records `cost.actual_usd` from `self.usd()`, which is GPU
    only — container disk is billed separately by the provider and the runner
    never sees it. Summing that field alone made the campaign check

        prior GPU + future GPU + future disk <= all-in ceiling

    which silently omits every predecessor's disk. R9 says the ceiling is
    cumulative across every resource and subrun, and the ceiling it is checked
    against — `campaign_all_in_hard_usd` — means all-in, so the predecessor's
    disk has to be in it.

    It is DERIVED rather than looked up, because nothing records it: the
    provisioned volume is billed for the pod's whole lifetime, so it is the
    authorization's own disk rate per minute times the minutes that pod ran.
    Generic core is not changed for this — the arithmetic belongs to whoever
    holds the all-in ceiling.

    A resource that was CREATED but whose cost or elapsed minutes cannot be
    read is `UNKNOWN`, never `$0`. Defaulting a paid predecessor to zero is
    how a cumulative ceiling comes to be checked against a fraction of what
    was spent.
    """
    record = REPO_ROOT / session_record_path(attempt)
    if not record.is_file():
        #: No session record means `run_session` never wrote one, and that has
        #: TWO causes which must not be conflated: a launcher that died
        #: mid-flight (UNKNOWN — it may have created a resource), and a chain
        #: RETIRED before the launcher ran at all (nothing was ever created).
        #:
        #: Only an affirmative statement can tell them apart, so only an
        #: affirmative statement is accepted: a closeout that explicitly
        #: records `provider_resource_created: false`. Absence stays UNKNOWN.
        #: A retired chain is not rare — a repair inside the executable closure
        #: forces one every time — and without this the campaign's first
        #: retirement would block every later attempt forever.
        outcome = REPO_ROOT / (
            f"{rel_run_dir(EXPERIMENT_ID, attempt, STAGE_ID)}/closeout/"
            "outcome.json")
        if outcome.is_file():
            try:
                closed = json.loads(outcome.read_text())
            except json.JSONDecodeError as exc:
                return {"attempt": attempt,
                        "unknown": f"unreadable closeout: {exc}"}
            if closed.get("provider_resource_created") is False:
                return {
                    "attempt": attempt, "provider_resource_created": False,
                    "gpu_usd": 0.0, "disk_usd": 0.0, "all_in_usd": 0.0,
                    "elapsed_minutes": 0.0,
                    "basis": "closeout", "terminal": closed.get("terminal"),
                    "_why_zero": (
                        "this run has no session record because its chain was "
                        "retired before the launcher ran. Its closeout states "
                        "affirmatively that no provider resource was created, "
                        "so nothing was billed. Absence of a closeout would "
                        "still be UNKNOWN."),
                }
        return {"attempt": attempt, "unknown": "no session record",
                "path": session_record_path(attempt)}
    try:
        ev = json.loads(record.read_text())
    except json.JSONDecodeError as exc:
        return {"attempt": attempt, "unknown": f"unreadable record: {exc}"}

    created = bool(ev.get("provider_resource_created"))
    cost = ev.get("cost") or {}
    if not created:
        #: A `$0` pre-provider refusal. No resource existed, so nothing was
        #: billed and there is nothing to derive.
        return {"attempt": attempt, "provider_resource_created": False,
                "gpu_usd": 0.0, "disk_usd": 0.0, "all_in_usd": 0.0,
                "elapsed_minutes": 0.0,
                "_why_zero": ("no provider resource was created, so neither "
                              "GPU nor disk was billed")}
    gpu, minutes = cost.get("actual_usd"), cost.get("elapsed_minutes")
    if gpu is None or minutes is None:
        return {"attempt": attempt, "provider_resource_created": True,
                "unknown": ("the record states "
                            f"actual_usd={gpu!r} and "
                            f"elapsed_minutes={minutes!r}; a resource that "
                            "billed cannot be accounted from either alone")}
    #: The authorization's own disk price per minute: `disk_hard_usd` is what
    #: the provisioned volume costs over `hard_runtime_minutes`, so the rate is
    #: that quotient. One basis for the ceiling and for the predecessor.
    disk_per_minute = (float(ctx.auth.disk_hard_usd)
                       / float(ctx.auth.hard_runtime_minutes))
    #: Rounded UP to the 4-decimal quantum every amount in this programme
    #: uses. A derived SPEND accumulating against a ceiling rounds up, the way
    #: `money()` ceils its amounts: rounding a predecessor's cost to nearest
    #: lets a campaign creep past its ceiling a hundredth of a cent at a time.
    disk = math.ceil(disk_per_minute * float(minutes) * 10_000) / 10_000
    return {"attempt": attempt, "provider_resource_created": True,
            "gpu_usd": float(gpu),
            "elapsed_minutes": float(minutes),
            "disk_usd": disk,
            "all_in_usd": math.ceil((float(gpu) + disk) * 10_000) / 10_000,
            "disk_usd_per_minute": round(disk_per_minute, 8),
            "provisioned_disk_gb": int(getattr(ctx.args, "disk_gb",
                                               CONTAINER_DISK_GB)),
            "_disk_is_derived": (
                "the provider bills the provisioned container disk for the "
                "pod's whole lifetime and the runner never sees it, so it is "
                "derived from the authorization's own disk rate times this "
                "resource's elapsed minutes"),
            "provider_confirms_gone": bool(ev.get("provider_confirms_gone"))}


def campaign_continuation_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A replacement RESOURCE may continue this campaign. It may not restart it,
    and it may not spend past the campaign's ceiling.

    The preregistered policy (R1-R6) governs which probes a continuation may
    consume, and two of its conditions are not the driver's to check because
    they are about resources and money rather than bytes:

    * **the previous resource must be provider-confirmed non-billing.** At most
      one resource of a campaign may bill at a time. A continuation launched
      while an earlier pod's release was never confirmed would put two on the
      meter, and "the remove call returned zero" is not confirmation — only the
      provider reporting the resource not billing is.
    * **cumulative campaign spend must stay inside the campaign's approved
      all-in ceiling.** The ceiling is cumulative across every resource and
      subrun: a rerun does not reset it and a replacement resource does not
      receive a fresh allocation. So the check is settled campaign spend plus
      this session's planned all-in against `campaign_all_in_hard_usd` — the
      CAMPAIGN's ceiling, never `all_in_hard_usd`, which bounds one session.

    This gate fails CLOSED and says why. While a campaign ceiling is sized for
    exactly one full session, a continuation after a resource that already
    spent real money is REFUSED here rather than permitted to overspend —
    funding a campaign for more than one full session is a maintainer decision,
    and this gate is where that need becomes visible instead of becoming an
    overrun. It is what happened after attempt3: the gate refused, the
    maintainer raised the CAMPAIGN ceiling by what attempt3 had spent, and the
    session ceiling did not move. Raising it is not this gate's to do, and a
    larger campaign ceiling reaches nothing in this file except this comparison.
    """
    campaign = ctx.auth.campaign_id
    prior = campaign_attempts(campaign, exclude=ctx.args.run_id,
                              store=ctx.args.ckpt_store, repo_root=REPO_ROOT)
    work = campaign_remaining_work(ctx)
    planned = continuation_all_in_usd(ctx, work)
    #: THE CAMPAIGN CEILING, never the session's. They are the same number only
    #: until the first attempt spends something; after that a session ceiling
    #: would refuse every continuation by exactly what was already spent, which
    #: is what it did to attempt3's successor.
    approved = float(ctx.auth.campaign_all_in_hard_usd)
    ctx.evidence["campaign"] = {
        "campaign_id": campaign, "run_attempt": ctx.args.run_id,
        "prior_attempts": prior,
        "remaining_work": {k: v for k, v in work.items()
                           if not k.startswith("_")},
        "this_session_planned_all_in_usd": round(planned, 4),
        "campaign_approved_all_in_usd": approved,
        "session_all_in_hard_usd": float(ctx.auth.all_in_hard_usd),
        "_two_ceilings": (
            "the session ceiling bounds ONE attempt and equals its GPU plus "
            "its disk. The campaign ceiling bounds the campaign cumulatively "
            "and is the only one prior spend is charged against. A larger "
            "campaign ceiling buys room beside settled spend, never a longer "
            "or dearer session: the window comes from gpu_hard_usd and "
            "hard_runtime_minutes, and the work from the frozen "
            "decomposition."),
        "_continuation_is_not_pooling": (
            "a replacement resource is a new RESOURCE and a new run attempt "
            "inside the SAME scientific campaign. Consuming its predecessor's "
            "destination-verified probes is continuation of one preregistered "
            "experiment, not pooling across experiments."),
        "_prior_attempts_source": (
            "the union of this campaign's run records and its durable store. "
            "Derived from durable probes alone this missed a pod that billed "
            "and died before its first probe, and the next attempt then called "
            "itself the campaign's first resource."),
    }
    if not prior:
        return True, (f"campaign {campaign} has no prior run attempt; this is "
                      f"its first resource, owing {work['n_probes_remaining']} "
                      f"probes and ${planned:.4f} all-in against the "
                      f"${approved:.4f} ceiling")

    from experiments.run_layout import rel_run_dir

    actuals = [prior_attempt_actual(ctx, a) for a in prior]
    unreadable = [a["attempt"] for a in actuals if a.get("unknown")]
    unconfirmed = [a["attempt"] for a in actuals
                   if a.get("provider_resource_created")
                   and not a.get("unknown")
                   and not a.get("provider_confirms_gone")]
    #: ALL-IN, not GPU. Every predecessor's derived disk is in here; see
    #: `prior_attempt_actual` for why it has to be derived at all.
    settled_gpu = sum(float(a.get("gpu_usd") or 0.0) for a in actuals
                      if not a.get("unknown"))
    settled_disk = sum(float(a.get("disk_usd") or 0.0) for a in actuals
                       if not a.get("unknown"))
    settled = settled_gpu + settled_disk
    ctx.evidence["campaign"].update({
        "prior_attempt_actuals": actuals,
        "settled_campaign_gpu_usd": round(settled_gpu, 4),
        "settled_campaign_disk_usd": round(settled_disk, 4),
        "settled_campaign_spend_usd": round(settled, 4),
        "_settled_is_all_in": (
            "GPU actual from each predecessor's session record plus its "
            "container disk derived from the authorization's own disk rate. "
            "The runner records GPU only, and comparing GPU-only prior spend "
            "against an all-in ceiling omitted every predecessor's disk."),
    })
    if unreadable:
        return False, (
            f"campaign {campaign} has prior run attempt(s) {unreadable} whose "
            f"actual spend could not be established: "
            f"{[a.get('unknown') for a in actuals if a.get('unknown')]}. A "
            "resource that may have billed is UNKNOWN, not zero, and an "
            "unknown billing state is a stop condition. Reconcile them before "
            "launching another. Records: "
            f"{[session_record_path(a) for a in unreadable]}")
    if unconfirmed:
        return False, (
            f"run attempt(s) {unconfirmed} of campaign {campaign} created a "
            "provider resource that was never confirmed released. At most one "
            "resource of a campaign may bill at a time, and a zero return code "
            "on a remove call is not evidence of release. Reconcile and tear "
            "down before creating another. Run dir(s): "
            f"{[rel_run_dir(EXPERIMENT_ID, a, STAGE_ID) for a in unconfirmed]}")

    if work["n_probes_remaining"] == 0:
        return False, (
            f"campaign {campaign} owes no probe: all "
            f"{work['probes_expected']} are complete and verified off-pod. A "
            "complete behavioural selection is TERMINAL — GO, NO_GO and "
            "INCONCLUSIVE are all complete results, and none of them is a "
            "reason to start another attempt. There is nothing here to "
            "continue.")

    #: TWO READERS OF ONE QUANTITY, RECONCILED. This gate reconciles each
    #: predecessor RESOURCE under R9 — GPU actual plus a disk term derived from
    #: the authorization's own rate — while the issuer caps this session's
    #: ceiling using the figure each attempt PUBLISHED in its closeout. Both
    #: are needed and they answer slightly different questions, which is
    #: exactly why they must not be allowed to drift: a session capped against
    #: one total and charged against the other can exceed its campaign's
    #: ceiling with every gate it passed saying yes.
    published = BG.settled_campaign_all_in(REPO_ROOT)
    ctx.evidence["campaign"]["settled_campaign_published_usd"] = published
    #: DIRECTIONAL. Only one of the two disagreements is dangerous. If the
    #: published total is SMALLER than what this gate reconciles, the issuer
    #: capped the session against too little settled spend and the session has
    #: room the campaign does not have. If it is larger, the cap was stricter
    #: than necessary — safe, and refusing it would block a launch over an
    #: over-conservative number.
    if published["total_usd"] < settled - BG.DOLLAR_QUANTUM_USD:
        return False, (
            f"campaign {campaign} reconciles to ${settled:.4f} settled from "
            f"its predecessors' session records and to "
            f"${published['total_usd']:.4f} from the closeouts those attempts "
            f"published ({published['per_attempt']}). The issuer capped this "
            "session's ceiling against the second figure and this gate charges "
            "the first, so a disagreement means the session could be "
            "authorized for more than the campaign has left. Reconcile the "
            "closeouts before launching.")

    if settled + planned > approved + BG.DOLLAR_QUANTUM_USD:
        return False, (
            f"campaign {campaign} has settled ${settled:.4f} all-in "
            f"(${settled_gpu:.4f} GPU + ${settled_disk:.4f} disk) across "
            f"{len(prior)} prior run attempt(s) and the work it still owes — "
            f"{work['n_probes_remaining']} probes, "
            f"{len(work['arms_needed'])} arm rebuild(s) and "
            f"{work['transfer']['weights']['gib']} GiB of probe restore — "
            "bounds at "
            f"${planned:.4f} all-in, which is ${settled + planned:.4f} against "
            f"an approved CAMPAIGN ceiling of ${approved:.4f} (one session's "
            f"all-in is ${float(ctx.auth.all_in_hard_usd):.4f}). The campaign "
            "ceiling is cumulative across every resource and subrun; a "
            "replacement resource does not receive a fresh allocation. This is "
            "a maintainer decision about funding the campaign — the experiment "
            "is NOT shortened to fit, and this gate may not raise the "
            "ceiling.")
    return True, (
        f"campaign continuation OK: ${settled:.4f} all-in settled "
        f"(${settled_gpu:.4f} GPU + ${settled_disk:.4f} disk) across "
        f"{len(prior)} prior attempt(s) plus ${planned:.4f} for the remaining "
        f"{work['n_probes_remaining']} probes is inside the ${approved:.4f} "
        f"campaign ceiling, and every prior resource is provider-confirmed "
        "released")


def readiness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A launch-bound readiness record for THIS run, on THIS tree, that PASSED.

    Three questions, and this gate used to get two of them wrong.

    It read `record.get("kind")`. **No readiness record carries `kind`** — the
    field is `record_kind`, and every sibling launcher reads that. So the
    comparison was `None != "launch_bound"` on every correct record and the
    gate COULD NOT PASS. The replay launcher's own docstring names this exact
    shape as the defect the full-search launcher was repaired for: "a gate that
    could not pass". Found here at `$0` by running the gates before creating a
    provider resource; it would otherwise have refused the launch at gate seven
    of eight and consumed the chain.

    And it never asked the verdict. A sweep that FAILED would have satisfied a
    kind check, which is the more dangerous half: the first behavioural sweep
    DID fail, and a gate that only asks "is this launch-bound" would have let a
    failing tree through.

    `verify_record` stays: a record that passed on another tree is evidence
    about that tree, not this one.
    """
    run_id = getattr(ctx.args, "run_id", "")
    try:
        record = BPE.load_record(REPO_ROOT, run_id=run_id, stage_id=STAGE_ID)
    except FileNotFoundError as exc:
        return False, f"no behavioural readiness record for this run: {exc}"
    try:
        BPE.verify_record(record, REPO_ROOT, run_id=run_id, stage_id=STAGE_ID)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"the readiness record does not describe this tree: {exc}"
    kind = record.get("record_kind")
    if kind != BPE.LAUNCH_BOUND:
        return False, (
            f"the readiness record is {kind!r}, not {BPE.LAUNCH_BOUND!r}. Only "
            "a launch-bound sweep describes the tree a launch will use.")
    if record.get("verdict") != "PASS":
        return False, (
            f"the launch-bound sweep recorded {record.get('verdict')!r} with "
            f"problems {record.get('problems')}. A launch does not rest on a "
            "sweep that failed.")
    return True, (f"readiness OK ({kind}, {record.get('verdict')}, "
                  f"{record.get('counts')})")


def bundle_staged_gate(ctx: SessionContext) -> tuple[bool, str]:
    """The pod's bundle must exist on the relay and contain this executable."""
    import tempfile

    run_id = getattr(ctx.args, "run_id", "")
    record_path = REPO_ROOT / bundle_record_for(run_id)
    if not record_path.is_file():
        return False, f"no bundle record at {bundle_record_for(run_id)}"
    record = json.loads(record_path.read_text())
    auth_rel = auth_path_for(run_id)
    auth_file = REPO_ROOT / auth_rel
    if not auth_file.is_file():
        return False, f"no authorization at {auth_rel}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = BT.roundtrip(
                session_commit=record["session_commit"],
                local_bundle_sha256=record["bundle_sha256"],
                authorization_bytes=auth_file.read_bytes(),
                authorization_path=auth_rel,
                workdir=Path(tmp), repo_root=REPO_ROOT)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"bundle roundtrip failed: {type(exc).__name__}: {exc}"
    if not out.get("ok", True):
        return False, f"bundle roundtrip refused: {out}"
    return True, f"bundle OK ({record['session_commit'][:12]}…)"


# ---------------------------------------------------------------------------
# products — durability, during the run
# ---------------------------------------------------------------------------

def evidence_locations(ctx: SessionContext) -> tuple[Path, ...]:
    """Where the fetched driver evidence can be, DERIVED from the spec.

    The runner extracts the verified archive to `<scr>/store/extracted/` before
    it calls `fetch_products`, and mirrors the driver's live evidence to
    `<scr>/relay/` during the run. A sibling launcher GUESSED this path, found
    no evidence, reported no finished work, and tore down a pod holding two
    completed leaves with every check green.
    """
    from collect_artifacts import load_specs

    scr = Path(ctx.args.scr)
    out: list[Path] = []
    spec_file = REPO_ROOT / "configs/autoinit/c2_behavioural_artifacts.json"
    if spec_file.is_file():
        for entry in load_specs(str(spec_file)):
            if entry.artifact_class == "session_evidence":
                out.append(scr / "store" / "extracted" / entry.pattern)
    out.append(scr / "relay" / "c2_behavioural_evidence.json")
    out.append(scr / "store" / "c2_behavioural_evidence.json")
    return tuple(out)


def finished_probes(ctx: SessionContext) -> list[dict] | None:
    """What the driver says it finished, or None when that is UNKNOWN.

    The distinction is the whole point. `[]` means the driver ran and finished
    nothing; `None` means its evidence could not be read, which is not the same
    claim and must never be reported as one.

    Only PROBES are owed off-pod. The six arm initializations are announced too,
    but they reconstruct deterministically from the teacher — that is why they
    are built rather than shipped — so losing one costs minutes, not an
    experiment. A probe is thirty minutes of training that the protocol forbids
    repeating for a different outcome.
    """
    for path in evidence_locations(ctx):
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        ctx.evidence["probe_evidence_read_from"] = str(path)
        #: EVERY announced probe, including one whose identity could not be
        #: computed. `announce_durable` never raises — a durability failure must
        #: not destroy the training it protects — so it records `identity: null`
        #: and continues. Filtering those out here would drop thirty minutes of
        #: completed training from the set teardown waits for, silently, which
        #: is precisely the failure this whole mechanism exists to prevent. They
        #: are fetched like any other and reported as unverifiable.
        return [u for u in record.get("durable_units", [])
                if u.get("kind") == "probe"]
    ctx.evidence["probe_evidence_unreadable"] = [
        str(p) for p in evidence_locations(ctx)]
    return None


def _scp_from_pod(ctx: SessionContext, remote: str, dest: Path, *,
                  limit_min: int = 5) -> int:
    """One small file off the pod. Returns the return code; never raises."""
    return subprocess.run(
        ["timeout", f"{limit_min}m", "scp",
         "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null",
         f"root@{ctx.host}:{remote}", str(dest)],
        capture_output=True, timeout=None).returncode


def secure_probe_evidence(ctx: SessionContext, units: list) -> list:
    """Pull each probe's SCIENCE evidence beside its bytes, during the run.

    The weights alone cannot continue a campaign. What the remaining stages
    read is the score and the per-sample rows: ranking needs
    `correct_overall`, and the verdict reads the per-sample JSONL. Those live
    in the pod's audit directory, and on the path where continuation is
    actually needed — a session that FAILED — the failed artifact spec collects
    only the session evidence. So the ranking, the probe records and the
    per-sample rows would not have come home at all, and a replacement resource
    would have had verified checkpoints it could not score against and a
    commitment it could not honour.

    They are tiny — a 950-row per-sample file is well under a megabyte — so
    they travel on every poll until they are complete, beside the probe they
    describe, under the destination's own names.

    A probe is announced when it finishes TRAINING, before it is scored, so the
    record arrives without a score first and is re-fetched until it has one.
    That ordering is the reason this is idempotent rather than once-only.

    MUST NOT raise: a durability helper that throws into a paid session's poll
    loop is a defect regardless of who catches it.
    """
    import shutil

    secured: list[dict] = []
    for unit in units:
        unit_id = unit["unit_id"]
        dest = probe_destination(ctx, unit_id)
        if not dest.is_dir():
            #: The bytes have not landed yet; the evidence follows them.
            continue
        record_dest = dest / BC.RECORD_NAME
        if record_dest.is_file():
            try:
                if (json.loads(record_dest.read_text()).get("score")
                        and (dest / BC.RESULT_NAME).is_file()
                        and (dest / BC.PER_SAMPLE_NAME).is_file()):
                    continue          # complete; nothing more to pull
            except json.JSONDecodeError:
                pass
        want = {
            BC.RECORD_NAME: f"{AUDIT_DIR}/probes/{unit_id}.json",
            BC.RESULT_NAME: f"{AUDIT_DIR}/{unit_id}_result.json",
            BC.PER_SAMPLE_NAME: f"{AUDIT_DIR}/{unit_id}_per_sample.jsonl",
        }
        got = {}
        for name, remote in want.items():
            tmp = dest / f".{name}.part"
            rc = _scp_from_pod(ctx, remote, tmp)
            if rc == 0 and tmp.is_file() and tmp.stat().st_size > 0:
                shutil.move(str(tmp), str(dest / name))
                got[name] = True
            else:
                tmp.unlink(missing_ok=True)
                got[name] = False
        secured.append({"unit_id": unit_id, "evidence": got})
        ctx.say(f"  probe {unit_id} evidence: "
                f"{sorted(n for n, ok in got.items() if ok)}")
    return secured


def secure_campaign_ranking(ctx: SessionContext) -> dict:
    """Pull the campaign's screening commitment to the CAMPAIGN root.

    Screening commits once per campaign, so the commitment belongs to the
    experiment rather than to the attempt that happened to compute it. A
    replacement resource must confirm the candidate its campaign advanced, and
    it can only do that if the record outlived the pod — which, on the failed
    path, the artifact spec does not collect.

    Written atomically and never overwritten once present: a second attempt
    must not be able to replace the commitment it is supposed to honour.
    """
    import shutil

    root = campaign_store(ctx.auth.campaign_id, ctx.args.ckpt_store)
    dest = root / BC.RANKING_NAME
    if dest.is_file():
        return {"already_present": True, "path": str(dest)}
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{BC.RANKING_NAME}.part"
    rc = _scp_from_pod(ctx, f"{AUDIT_DIR}/{BC.RANKING_NAME}", tmp)
    if rc != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        #: Not an error: before stage R there is no ranking to secure.
        return {"secured": False, "rc": rc}
    try:
        record = json.loads(tmp.read_text())
    except json.JSONDecodeError as exc:
        tmp.unlink(missing_ok=True)
        return {"secured": False, "why": f"unreadable: {exc}"}
    if record.get("campaign") != ctx.auth.campaign_id:
        tmp.unlink(missing_ok=True)
        return {"secured": False, "why": (
            f"the ranking names campaign {record.get('campaign')!r}, not "
            f"{ctx.auth.campaign_id!r}; it is not this campaign's commitment")}
    shutil.move(str(tmp), str(dest))
    ctx.say(f"  campaign ranking secured: advanced "
            f"{(record.get('advanced') or {}).get('state_id', '?')[:12]}…")
    return {"secured": True, "path": str(dest),
            "advanced": (record.get("advanced") or {}).get("state_id")}


def _fetch_and_verify(ctx: SessionContext, units: list) -> list:
    """Pull each named probe and RE-IDENTIFY it from the bytes that land.

    ONE implementation, used by the poll hook and by the closeout, so a probe
    secured mid-run and a probe secured at the end are verified identically.

    Verification reads the VERDICT, not merely the absence of an exception:
    `verify_transferred_leaf` raises only on a structurally unreadable arrival
    and reports a digest mismatch in its return value. A sibling launcher called
    it and kept only "it did not raise", which accepted any arrival that parsed.
    """
    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.runtime.leaf_durability import (
        LeafDurabilityError, verify_transferred_leaf,
    )

    fetched: list = []
    adapter = get_adapter("qwen3")
    for unit in units:
        unit_id = unit["unit_id"]
        dest = probe_destination(ctx, unit_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            ["timeout", f"{ctx.args.ckpt_fetch_limit_min}m", "scp", "-r",
             "-P", str(ctx.target.port), "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null",
             f"root@{ctx.host}:{unit['path']}", str(dest)],
            capture_output=True, timeout=None)
        size = (sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
                if dest.exists() else 0)

        matched, why = False, "not verified"
        if rc.returncode == 0 and not unit.get("identity"):
            #: The bytes are off-pod, which is the point, but nothing can say
            #: they are the right bytes. Recorded loudly rather than counted as
            #: a clean save, and it does not strand the pod: a unit with no
            #: announced identity can never be verified, so waiting for it
            #: would mean never tearing down.
            why = ("PRESERVED BUT UNVERIFIABLE: the driver announced no "
                   f"identity for this probe ({unit.get('identity_error')}), "
                   "so the delivered bytes cannot be re-identified")
        elif rc.returncode == 0:
            try:
                v = verify_transferred_leaf(dest, unit["identity"], adapter=adapter)
                #: Every field the announcement carries, compared. A check that
                #: only compares the fields it happens to find is not a check.
                matched = bool(v["matched"] and v["weights_digest_matched"]
                               and v["shard_matched"]
                               and v["config_matched"] is not False)
                why = ("re-identified from the delivered bytes" if matched else
                       f"DIGEST MISMATCH: artifact={v['matched']}, "
                       f"weights={v['weights_digest_matched']}, "
                       f"shard={v['shard_matched']}, config={v['config_matched']}")
            except (LeafDurabilityError, OSError, KeyError,
                    json.JSONDecodeError) as exc:
                why = f"{type(exc).__name__}: {exc}"
        else:
            why = f"scp rc={rc.returncode}: {(rc.stderr or b'')[-200:]!r}"

        fetched.append({"artifact": "c2_behavioural_probe", "unit_id": unit_id,
                        "rc": rc.returncode, "bytes": size, "dest": str(dest),
                        "matched": matched,
                        "identity_announced": bool(unit.get("identity")),
                        "why": why})
        ctx.say(f"  probe {unit_id}: rc={rc.returncode}, "
                f"{size / 2**30:.2f} GiB -> {dest} [{why}]")
        if matched:
            #: The durable ACK, beside the bytes, so a reader of the destination
            #: can tell a verified arrival from a partial copy. It is NOT
            #: permission: whether a preserved probe may be reused is a separate
            #: scientific decision.
            (dest / "durable_ack.json").write_text(json.dumps({
                "schema": "aadistill.autoinit.c2_behavioural_probe_ack/v1",
                "unit_id": unit_id, "campaign": unit.get("campaign"),
                #: BOTH identities. The campaign says which experiment this
                #: probe belongs to and therefore which later resource may
                #: continue with it; the run attempt says which invocation and
                #: which provider resource produced it.
                "authorized_campaign": ctx.auth.campaign_id,
                "run_attempt": ctx.args.run_id,
                "run_id": ctx.args.run_id,
                "bytes": size,
                "identity": unit["identity"],
                "re_identified_from_delivered_bytes": True,
                "_what_this_means": (
                    "the bytes at this path were re-identified after transfer "
                    "and match what the driver announced. It is a claim about "
                    "this probe, not about the session."),
                "authorizes": ("nothing. Preservation and reuse are separate "
                               "decisions."),
            }, indent=1) + "\n")
            #: AND TELL THE POD. This is the acknowledgement boundary the
            #: probe-local lifecycle needs: until the bytes are independently
            #: durable AND re-identified HERE, the pod's copy is the only one
            #: and deleting it would be a durability race. After this file
            #: exists the pod may release that probe's training workdir.
            #:
            #: attempt5 had no such boundary, so nothing was ever released and
            #: twelve probes' workdirs accumulated until the trainer could not
            #: write probe 11. `announce_durable` deliberately only announces;
            #: it cannot know that the transfer succeeded, because it runs
            #: before the transfer does.
            release_ack(ctx, unit_id, size)
    return fetched


#: Where the pod looks for the launcher's release acknowledgements. One small
#: file per probe, written only after destination re-identification matched.
#: Composed from the SHARED repo-relative constant, so the launcher that writes
#: it and the driver that reads it cannot disagree about where it is.
RELEASE_ACK_DIR = f"{REPO}/{BC.RELEASE_ACK_REL}"


def release_ack(ctx: SessionContext, unit_id: str, size: int) -> bool:
    """Tell the pod that this probe's bytes are durable, so it may release.

    MUST NOT raise: it runs inside the runner's poll loop, where a durability
    helper that throws into a paid session is a defect whoever swallows it.
    A failure here costs disk on the pod, which the fail-closed caller in the
    driver then reports; it must never cost the session.

    Returns whether the acknowledgement landed, so the evidence can say.
    """
    payload = json.dumps({
        "schema": "aadistill.autoinit.c2_behavioural_release_ack/v1",
        "unit_id": unit_id, "bytes": size,
        "destination_re_identified": True,
        "_what_this_permits": (
            "releasing this probe's LOCAL training workdir on the pod. It "
            "permits nothing about reuse: whether a preserved probe may be "
            "consumed by a later attempt is R1-R10's decision, not this "
            "file's."),
    })
    #: Copied as a FILE rather than echoed through a shell: the payload is
    #: JSON with quotes and braces, and a heredoc that a remote shell decides
    #: to expand writes something the driver cannot parse. `run` is the pod
    #: exec this launcher already uses; `ssh` is not a method on the target.
    import tempfile

    try:
        ctx.target.run(f"mkdir -p {RELEASE_ACK_DIR}", timeout=60)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / f"{unit_id}.json"
            local.write_text(payload + "\n")
            rc = subprocess.run(
                [*ctx.scp, str(local),
                 f"root@{ctx.host}:{RELEASE_ACK_DIR}/{unit_id}.json"],
                capture_output=True, text=True, timeout=120)
        ok = rc.returncode == 0
    except Exception as exc:                                    # noqa: BLE001
        ctx.evidence.setdefault("release_ack_errors", []).append(
            f"{unit_id}: {type(exc).__name__}: {exc}")
        return False
    ctx.evidence.setdefault("release_acks", []).append(
        {"unit_id": unit_id, "delivered": ok})
    if not ok:
        ctx.say(f"  probe {unit_id}: release ack NOT delivered; the pod keeps "
                f"its local copy and its storage bound no longer holds")
    return ok


def secure_finished_probes(ctx: SessionContext) -> None:
    """Fetch and verify every probe that has FINISHED, while the driver runs.

    Called from the runner's poll loop, which relays the driver's evidence
    live, so a probe finished at minute 40 is off-pod by minute 42 rather than
    at closeout — or never. Idempotent: probes already secured are skipped.

    MUST NOT raise. The runner swallows exceptions, but a durability helper
    that throws into a paid session's poll loop is a defect regardless of who
    catches it.
    """
    try:
        units = finished_probes(ctx)
        if not units:
            return
        already = {r["unit_id"] for r in ctx.evidence.get("probes_secured", [])}
        pending = [u for u in units if u["unit_id"] not in already]
        if pending:
            fetched = _fetch_and_verify(ctx, pending)
            #: Anything that ARRIVED is done, verified or not. Keying only on
            #: `matched` would re-scp an unverifiable probe on every poll for
            #: the rest of the session, and a retry cannot make an identity the
            #: driver never computed appear.
            ctx.evidence.setdefault("probes_secured", []).extend(
                r for r in fetched
                if r.get("rc") == 0 and (r.get("matched")
                                         or not r.get("identity_announced")))
            ctx.evidence.setdefault("probe_fetch_attempts", []).extend(fetched)
        #: The science evidence and the campaign's commitment, EVERY poll and
        #: not only for newly arrived probes: a probe is announced when its
        #: training finishes and scored afterwards, so its score lands on a
        #: later poll than its bytes.
        ctx.evidence["probe_evidence_secured"] = secure_probe_evidence(
            ctx, units)
        ctx.evidence["campaign_ranking_secured"] = secure_campaign_ranking(ctx)
    except Exception as exc:                                    # noqa: BLE001
        ctx.evidence.setdefault("on_poll_errors", []).append(
            f"secure_finished_probes: {type(exc).__name__}: {exc}")


def fetch_probes(ctx: SessionContext) -> list:
    """Closeout sweep: anything the poll loop did not already secure."""
    units = finished_probes(ctx)
    if not units:
        return list(ctx.evidence.get("probe_fetch_attempts", []))
    already = {r["unit_id"] for r in ctx.evidence.get("probes_secured", [])}
    pending = [u for u in units if u["unit_id"] not in already]
    fetched = _fetch_and_verify(ctx, pending) if pending else []
    ctx.evidence.setdefault("probes_secured", []).extend(
        r for r in fetched
        if r.get("rc") == 0 and (r.get("matched")
                                 or not r.get("identity_announced")))
    #: The LAST chance for the evidence, on the same terms as the bytes. A
    #: probe whose score landed between the final poll and teardown would
    #: otherwise be a verified checkpoint nothing can rank.
    try:
        ctx.evidence["probe_evidence_secured"] = secure_probe_evidence(
            ctx, units)
        ctx.evidence["campaign_ranking_secured"] = secure_campaign_ranking(ctx)
    except Exception as exc:                                    # noqa: BLE001
        ctx.evidence.setdefault("closeout_errors", []).append(
            f"secure_probe_evidence: {type(exc).__name__}: {exc}")
    return list(ctx.evidence.get("probe_fetch_attempts", [])) + fetched


def probes_secured(ctx: SessionContext, fetched: list) -> tuple[bool, str]:
    """Teardown may not proceed while a finished probe is only on the pod."""
    units = finished_probes(ctx)
    if units is None:
        return False, (
            "the driver evidence could not be read, so what it finished is "
            "UNKNOWN. Refusing teardown: 'I found no probes' and 'no probes "
            "were trained' are different findings, and treating the first as "
            "the second is how a pod holding finished work gets deleted with "
            f"every check green. Looked in: "
            f"{[str(p) for p in evidence_locations(ctx)]}")
    want = {u["unit_id"] for u in units}
    if not want:
        return True, "the driver finished no probe, so none is owed off-pod"
    rows = [f for f in fetched if isinstance(f, Mapping)
            and f.get("artifact") == "c2_behavioural_probe"]
    #: A probe is SECURED when its bytes arrived and were re-identified. One
    #: whose driver could not compute an identity can never be re-identified,
    #: so requiring it would mean never tearing down; it counts as secured once
    #: the bytes arrive, and is reported separately so nobody reads it as a
    #: clean save.
    verified = {f["unit_id"] for f in rows if f.get("rc") == 0 and f.get("matched")}
    unverifiable = {f["unit_id"] for f in rows
                    if f.get("rc") == 0 and not f.get("identity_announced")}
    got = verified | unverifiable
    missing = sorted(want - got)
    if missing:
        return False, (
            f"{len(want)} probes finished and {len(got)} are off-pod; missing "
            f"{missing}. Deleting the pod now would destroy work this session "
            "already paid for, which is the exact failure this mechanism "
            "exists to prevent.")
    note = f"all {len(want)} finished probes off-pod ({len(verified)} re-identified"
    if unverifiable:
        note += (f", {len(unverifiable)} PRESERVED BUT UNVERIFIABLE: "
                 f"{sorted(unverifiable)}")
    return True, note + ")"


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------

#: Where a replacement pod's restored probes land, and where the manifest that
#: names them lives. NEW paths: the producing pod's absolute `model_dir` does
#: not exist on a replacement resource and must not be the truth about one.
RESTORE_MANIFEST = f"{WORKDIR}/campaign_continuation.json"

#: Where a completed probe's LIGHTWEIGHT EVIDENCE lands. On the container disk,
#: not the volume: it is 8.1 MiB for this whole campaign, it is copied fresh
#: each session from the launcher host's durable store, and the verdict reads
#: it. Putting it on the shared volume would make ten sessions write the same
#: small files to one place for no gain.
EVIDENCE_DIR = f"{WORKDIR}/evidence"


def volume_probe_root(ctx: SessionContext) -> str:
    """Where this campaign's pre-staged probes are, on the attached volume.

    The layout is the staging tool's and is FLAT — one directory per probe id,
    mirroring how `campaign_state` keys its probes. The producing attempt is
    recorded per probe inside the staged index rather than in the path, because
    a continuation restores by probe id and a path segment naming an attempt
    would have to be guessed by whoever builds the manifest.

    EMPTY when no volume is attached, rather than a path rooted at `/`. A
    session whose probes all have their weights already, or need none, mounts
    nothing (`volume_attachment`), and recording `/<campaign>/probes` as though
    it were a real location would tell a later reader the bytes were somewhere
    they have never been.
    """
    mount = str(getattr(ctx.args, "volume_mount_path", "") or "").strip()
    if not mount:
        return ""
    return f"{mount}/{ctx.auth.campaign_id}/probes"


def restore_campaign_probes(ctx: SessionContext) -> bool:
    """Put this campaign's verified probes on the replacement pod. `False` aborts.

    THE PRODUCTION HANDOFF, and the whole reason it has to exist: a replacement
    resource has a FRESH FILESYSTEM. The previous pod's `audit/probes/*.json`
    is gone, every `model_dir` it recorded points at nothing, and the driver's
    `load_campaign_journal` reads exactly those two things. Without this step
    the preregistered continuation policy is unreachable in production no
    matter how correct the campaign id is — which is what it was.

    Runs from `materialize_inputs`: after setup, before the driver starts, and
    the runner tears the pod down if it returns `False`. That placement is the
    point — a probe restored after the driver had begun would be a probe the
    campaign journal had already decided was absent.

    What is here, and from where:

    * the BYTES are ALREADY ON THE POD, on the attached network volume, put
      there by `stage_c2_probes_to_volume.py` while nothing expensive was
      billing. Only probes whose `durable_ack.json` records a destination-side
      re-identification that MATCHED were eligible to be staged, and only they
      are named here;
    * their science evidence travelled with them, because the remaining stages
      read the score and the per-sample rows, not the weights;
    * the campaign's screening commitment, if it has one;
    * a small manifest naming each probe's pod path and the identity the driver
      must reproduce there.

    **What changed, and why it is not a weakening.** This step used to `scp`
    22.2 GiB from the launcher host onto a billing pod, and was priced as the
    expensive phase of a continuation. The bytes still have to be on the pod
    and still have to re-identify there — R2 and R6 are untouched, and the
    driver's check is the same check against the same announced identity. What
    moved is only *when* the copy was paid for: once, onto a volume, at CPU
    prices, instead of once per attempt at L40S prices. The launcher-host
    re-identification below still happens, so the three-point check the resume
    policy describes — at the destination, on the host, and on the consuming
    pod — is intact.
    """
    work = campaign_remaining_work(ctx)
    state = work["_state"]
    pod_root = volume_probe_root(ctx)
    manifest = BC.build_manifest(
        REPO_ROOT, campaign_id=ctx.auth.campaign_id,
        run_attempt=ctx.args.run_id, state=state, work=work,
        pod_root=pod_root, evidence_root=EVIDENCE_DIR)

    #: THE MANDATORY TRANSFER PREFLIGHT, derived and printed before anything
    #: moves. An execution sanity check, not an approval gate: it does not stop
    #: to ask, and a set smaller than an earlier conservative implementation
    #: produced is not a reason to return.
    plan = BC.transfer_plan(state, work)
    ctx.say(f"transfer preflight: {plan['summary']}")
    for row in plan["rows"]:
        ctx.say(f"  {row['artifact'][:46]:48s} {row['state'][:34]:36s} "
                f"{row['decision']}")
    t = work["transfer"]
    ctx.evidence["campaign_restore"] = {
        "n_weights_probes": t["weights"]["n"],
        "weights_gib": t["weights"]["gib"],
        "n_evidence_probes": t["evidence"]["n"],
        "evidence_mib": t["evidence"]["mib"],
        "skipped_checkpoints": t["skipped_weights"]["n"],
        "avoided_gib": t["skipped_weights"]["gib"],
        "bounded_minutes": t["minutes"],
        "transfer_plan": plan,
        "committed_candidate": manifest["committed_candidate"],
        "pod_root": pod_root,
        "evidence_root": EVIDENCE_DIR,
        "transport": (
            "weights come from the attached network volume when a remaining "
            "operation reads them; a completed probe's checkpoint is archival "
            "and is not moved. Evidence is copied fresh from the launcher "
            "host's durable store."),
        "probes": [],
        "evidence": [],
    }

    #: The manifest always travels, even empty: the driver must be able to tell
    #: "this campaign restored nothing" from "the restore step never ran", and
    #: an absent file cannot say which.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "campaign_continuation.json"
        local.write_text(json.dumps(manifest, indent=1) + "\n")
        rc = subprocess.run(
            list(ctx.scp) + [str(local),
                             f"root@{ctx.host}:{RESTORE_MANIFEST}"],
            capture_output=True, timeout=600)
    if rc.returncode != 0:
        ctx.say(f"ABORT: the continuation manifest did not reach the pod: "
                f"{(rc.stderr or b'')[-200:]!r}")
        return False

    #: THE EVIDENCE LEG. A completed and validly scored probe contributes
    #: per-sample rows, a score, a descriptor, seeds, battery identities and
    #: hashes -- 8.1 MiB for this whole campaign against 22.21 GiB of weights
    #: nothing remaining reads. It is copied fresh from the launcher host's
    #: durable store rather than staged, because at this size staging it would
    #: be more machinery than transfer.
    for entry in manifest["evidence"]:
        pid, source = entry["probe_id"], Path(entry["durable_path"])
        dest = entry["pod_path"]
        missing = [n for n in entry["files"] if not (source / n).is_file()]
        if missing:
            ctx.say(f"ABORT: {pid}'s evidence is incomplete on this host: "
                    f"{missing} absent. The verdict reads these rows; a score "
                    "without them is not a usable score.")
            return False
        ctx.target.run(f"mkdir -p {dest}", timeout=60)
        out = subprocess.run(
            list(ctx.scp) + [str(source / n) for n in entry["files"]]
            + [f"root@{ctx.host}:{dest}/"],
            capture_output=True, timeout=600)
        check = ctx.target.run(
            " && ".join(f"test -s {dest}/{n}" for n in entry["files"])
            + " && echo PRESENT=1 || echo PRESENT=0", timeout=120)
        present = "PRESENT=1" in check.stdout
        ctx.evidence["campaign_restore"]["evidence"].append({
            "probe_id": pid, "sent": out.returncode == 0, "present": present,
            "pod_path": dest, "files": entry["files"],
            "weights_transferred": False,
            "_why_no_weights": (
                "completed and validly scored: no remaining authorized "
                "operation reads this checkpoint")})
        if out.returncode != 0 or not present:
            ctx.say(f"ABORT: {pid}'s evidence did not reach the pod "
                    f"(rc={out.returncode}, present={present}). The verdict "
                    "reads these rows.")
            return False
    if manifest["evidence"]:
        ctx.say(f"campaign evidence: {len(manifest['evidence'])} scored "
                f"probe(s), {t['evidence']['mib']} MiB at {EVIDENCE_DIR}; "
                f"their {t['skipped_weights']['gib']} GiB of checkpoints were "
                "not moved and have no remaining consumer")

    if not manifest["probes"]:
        ctx.say("campaign restore: no probe needs its WEIGHTS on this pod — "
                "every held probe is completed and validly scored, so nothing "
                "remaining reads a checkpoint")
        return True

    from aadistill.initialization.specs.arch import get_adapter
    from aadistill.runtime.leaf_durability import (
        LeafDurabilityError, verify_transferred_leaf,
    )

    #: THE VOLUME'S OWN INDEX, read once. It records which attempt each staged
    #: copy came from. Probe ids are unique within a campaign and
    #: `campaign_state` collapses them across attempts, so without this a stale
    #: copy from an earlier attempt could sit under exactly the right name and
    #: fail re-identification only on the paid pod — a correct refusal arriving
    #: at the most expensive possible moment.
    index_path = f"{pod_root}/staged_index.json"
    raw = ctx.target.run(f"cat {index_path} 2>/dev/null", timeout=120)
    try:
        staged = json.loads(raw.stdout or "{}")
    except json.JSONDecodeError as exc:
        ctx.say(f"ABORT: {index_path} on the volume is unreadable ({exc}). "
                "The pre-staged probes cannot be shown to belong to this "
                "campaign.")
        return False
    if staged.get("campaign_id") != ctx.auth.campaign_id:
        ctx.say(f"ABORT: the volume holds probes staged for campaign "
                f"{staged.get('campaign_id')!r}, not {ctx.auth.campaign_id!r}. "
                "One experiment's probes may never be pooled into another.")
        return False
    ctx.evidence["campaign_restore"]["staged_index"] = {
        "campaign_id": staged.get("campaign_id"),
        "staged_utc": staged.get("staged_utc"),
        "n_staged": len(staged.get("probes") or {}),
    }

    adapter = get_adapter("qwen3")
    for entry in manifest["probes"]:
        pid, source = entry["probe_id"], Path(entry["durable_path"])
        staged_here = (staged.get("probes") or {}).get(pid)
        if not staged_here:
            ctx.say(f"ABORT: {pid} is named for restore and was never staged "
                    "onto the volume. The protocol forbids retraining a "
                    "completed probe, so there is nothing safe to do here.")
            return False
        if staged_here.get("source_attempt") != entry["source_attempt"]:
            ctx.say(f"ABORT: {pid} was staged from "
                    f"{staged_here.get('source_attempt')!r} and this "
                    f"campaign's state names {entry['source_attempt']!r}. The "
                    "copy on the volume is not the measurement the manifest "
                    "identifies.")
            return False
        #: RE-IDENTIFIED HERE TOO, before a byte is sent. The ack says these
        #: bytes matched when they landed; it does not say they still do. An
        #: ack is a record, and a record is not a checkpoint.
        try:
            v = verify_transferred_leaf(source, entry["identity"],
                                        adapter=adapter)
            local_ok = bool(v["matched"] and v["weights_digest_matched"]
                            and v["shard_matched"]
                            and v["config_matched"] is not False)
            why = "re-identified at the destination" if local_ok else (
                f"DESTINATION MISMATCH: artifact={v['matched']}, "
                f"weights={v['weights_digest_matched']}, "
                f"shard={v['shard_matched']}, config={v['config_matched']}")
        except (LeafDurabilityError, OSError, KeyError,
                json.JSONDecodeError) as exc:
            local_ok, why = False, f"{type(exc).__name__}: {exc}"
        if not local_ok:
            ctx.say(f"ABORT: {pid} is named for restore and no longer "
                    f"re-identifies on this host — {why}. A continuation that "
                    "shipped unverifiable bytes would be substituting for a "
                    "measurement.")
            ctx.evidence["campaign_restore"]["probes"].append(
                {"probe_id": pid, "sent": False, "why": why})
            return False

        #: PRESENCE ON THE VOLUME. Not a transfer: the bytes were put here
        #: before this pod existed. The files checked are the ones the driver
        #: and the verdict actually read — the weights, the config that
        #: identifies them, and, for a scored probe, the per-sample rows the
        #: decision rule consumes. A score whose rows did not survive is not a
        #: usable score, and finding that out in stage D would waste the run.
        #: Only probes whose WEIGHTS a remaining operation reads reach here,
        #: so the files checked are the ones the scorer needs. A scored probe's
        #: rows travel on the evidence leg above and are checked there.
        dest = entry["pod_path"]
        needed = ["model.safetensors", "config.json", "probe_record.json"]
        test = " && ".join(f"test -s {dest}/{n}" for n in needed)
        probe = ctx.target.run(f"{test} && echo PRESENT=1 || echo PRESENT=0",
                               timeout=120)
        present = "PRESENT=1" in probe.stdout
        ctx.evidence["campaign_restore"]["probes"].append({
            "probe_id": pid, "pre_staged": True, "present": present,
            "bytes": entry["bytes"], "scored": entry["scored"],
            "source_attempt": entry["source_attempt"],
            "destination_reverified": True, "pod_path": dest,
            "files_checked": needed})
        ctx.say(f"  pre-staged {pid}: present={present} "
                f"({entry['bytes'] / 2**30:.2f} GiB) at {dest}")
        if not present:
            ctx.say(f"ABORT: {pid} is not readable on the volume at {dest}. "
                    "The driver would treat a completed probe as absent and "
                    "the protocol forbids retraining it, so there is nothing "
                    "safe to do here but stop.")
            return False

    ctx.say(f"campaign restore: {len(manifest['probes'])} probe(s) whose "
            f"weights a remaining operation reads are pre-staged on the volume "
            f"at {pod_root}; the driver re-identifies each one THERE before "
            "admitting it")
    return True


def driver_command(ctx: SessionContext, plan) -> str:
    """The campaign is the SCIENCE; the run attempt is this invocation.

    `--campaign` used to be `ctx.args.run_id`, which made the two the same
    string and the preregistered continuation policy unreachable: R1 permits
    reuse only inside one campaign, so a replacement provider resource became a
    different campaign and had to refuse every probe the previous resource had
    trained and verified off-pod.

    The campaign comes from the AUTHORIZATION rather than from a constant read
    here, so an artifact that does not name this campaign cannot permit work
    under it.
    """
    return (f"/opt/train/bin/python "
            f"{REPO}/scripts/pod/autoinit_c2_behavioural_driver.py "
            f"--campaign {ctx.auth.campaign_id} "
            f"--run-attempt {ctx.args.run_id} "
            f"--continuation-manifest {RESTORE_MANIFEST} "
            f"--audit-dir {AUDIT_DIR} --eval-dir {EVAL_DIR} "
            f"--b-workdir {ARM_DIR} --status-path {STATUS} "
            f"--image-digest '{ctx.image_digest}' "
            f"--b-build-minutes {ctx.args.b_build_minutes:.2f} "
            f"--probe-train-minutes {ctx.args.probe_train_minutes:.2f} "
            f"--probe-battery-minutes {ctx.args.probe_battery_minutes:.2f} "
            f"--rate {ctx.price or ctx.args.max_price} "
            f"--spent-usd {ctx.spent_usd:.4f} "
            f"--soft-stop-usd {plan.soft_stop_usd:.4f} "
            f"--authorized-usd {ctx.auth.hard_cap_usd:.4f}")


def budget_work(args) -> dict:
    """The decomposition arguments for THIS attempt: full session or remainder.

    Read from the durable destination, at `$0`, before a pod exists — the same
    source and the same derivation the continuation gate uses, so the plan the
    driver spends admission control against and the gate that permitted the
    launch cannot describe different work.

    A campaign with nothing verified off-pod owes the whole protocol, which is
    what makes the default path and the continuation path one code path rather
    than two.
    """
    campaign = BG.CAMPAIGN_ID
    store = getattr(args, "ckpt_store", DURABLE_STORE)
    state = BC.campaign_state(campaign_store(campaign, store),
                              exclude_attempt=getattr(args, "run_id", ""))
    work = BC.remaining_work(REPO_ROOT, state=state)
    return {"materialization_minutes": work["materialization_minutes"],
            #: THREE states, priced as the driver executes them: an untrained
            #: probe is trained and scored, a restored trained-but-unscored
            #: probe owes the battery only.
            "train_and_score_probes": work["n_train_and_score"],
            "score_only_probes": work["n_score_only"],
            "restore_minutes": work["transfer"]["minutes"]}


def budget(args) -> BudgetSpec:
    """THE canonical decomposition, carried verbatim. Nothing is re-derived here.

    This function used to build a SECOND budget model on top of the proposal's
    FINAL one. `BG.ceiling()` reports a hard window that already contains the
    frozen probe model's named reserves, its 10% contingency and its
    artifact-recovery reserve; this function subtracted the materialization term
    back out, fed the remainder into a fresh `BudgetSpec` beside its own
    `setup`, `transfer` and `materialize` phases, and applied a SECOND
    contingency and a SECOND recovery reserve. `plan_session` then answered
    2036.62 hard minutes — about `$36.9987` of GPU at `$1.09/h`, bigger than the
    proposal's entire `$33.2099` all-in ceiling. A correct authorization derived
    from the proposal would have refused this launch at the gate, for reserves
    nobody granted twice.

    So the phases, the named reserves and the recovery reserve all come from
    `BH.session_decomposition`, and the contingency fraction is ZERO because the
    contingency is already one of those named reserves in minutes.
    `test_the_launcher_and_the_proposal_derive_one_hard_window` asserts the two
    agree at the same quoted rate.

    What keeps the money inside the authorization is not this estimate but the
    driver's admission control, which starts a unit only when the remaining
    soft-stop budget can fund that unit's full bound. The consequence is
    explicit: under worst-case timings the session stops with a rung
    incomplete — and a partial rung decides nothing — rather than overspending.
    """
    from aadistill.infrastructure.budget import MEASURED_STEP_SECONDS, Phase

    #: REMAINING work, and a fresh campaign's remaining work is all of it. The
    #: same decomposition either way — a continuation budget derived beside
    #: this one would be the duplicate-budget defect again — and it is the same
    #: figures the continuation gate checked, because both read the campaign
    #: state from the durable destination on this host at `$0`.
    d = BH.session_decomposition(
        REPO_ROOT, **budget_work(args))

    return BudgetSpec(
        #: `arms=0` and both generic phases at zero: the generic
        #: arms x steps + setup + transfer shape does not describe arm
        #: materializations followed by probes, and every phase this session
        #: has is named in the decomposition instead.
        arms=0, steps_per_arm=0,
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: the generic arms x steps term does not describe "
                     "arm materializations followed by probes. Every phase is "
                     "named in BH.session_decomposition, which is derived from "
                     "committed records and reconciled against them."),
        setup_minutes=0.0,
        transfer_minutes=0.0,
        other_phases=tuple(Phase(name, minutes)
                           for name, minutes in d["expected_phases"]),
        #: ZERO. The frozen model's 10% contingency is a named reserve below.
        contingency_fraction=d["contingency_fraction"],
        soft_stop_reserves=tuple(Phase(name, minutes)
                                 for name, minutes in d["soft_stop_reserves"]),
        artifact_recovery_reserve_minutes=(
            d["artifact_recovery_reserve_minutes"]),
    )


def spec(args) -> SessionSpec:
    return SessionSpec(
        session_id=BG.SESSION_ID,
        schema="aadistill.autoinit.c2_behavioural_session/v1",
        description=(
            "The Phase-C2 behavioural selection. Six arms materialized from the "
            "teacher along digest-pinned paths, six screening probes on one "
            "preregistered seed, a mechanical ranking that advances exactly one "
            "candidate, six confirmation probes on three disjoint preregistered "
            "seeds, and one verdict under the frozen Phase-C decision rule. It "
            "runs no beam, re-ranks nothing, re-measures no frozen result, and "
            "reaches no later cycle. GO, NO_GO and INCONCLUSIVE are all "
            "complete results."),
        authorization_path=auth_path_for(getattr(args, "run_id", "")),
        authorization_loader=BG.BehaviouralAuthorization.load,
        commands=ExecutionCommands(
            watchdog="scripts/pod/watchdog.py",
            setup_script="scripts/pod/autoinit_preflight_setup.sh",
            artifact_collector="scripts/pod/collect_artifacts.py",
            **deployment_commands()),
        plan_id=BG.PLAN_ID,
        plan_hash=BG.plan_hash(REPO_ROOT),
        budget=budget(args),
        setup=SetupManifest(
            relay_inputs=(*RECOVERY_LADDER, *CALIBRATION_V1,
                          *C1_EVAL_TOKENIZER, *C1_ROPE_INPUT),
            local_assets=LOCAL_ASSETS,
            required_env=("SESSION_COMMIT", "BUNDLE_NAME", "SESSION_STATUS",
                          "SESSION_AUTH_PATH", "SESSION_PLAN_HASH",
                          "SESSION_ASSETS", "SESSION_RELAY_INPUTS",
                          "SESSION_KIND", "SESSION_SETUP_MARKERS",
                          "TEACHER_REVISION"),
            #: ROPE_OK is PRESENT: this session stages the 0.6B config the
            #: shared step globs, and every probe is trained and evaluated
            #: through that tokenizer/config family, so the check is live here.
            #: VLLM_READY is PRESENT: both rungs generate through vLLM.
            setup_markers=("ENV_READY", "REPO_READY", "ASSETS_STAGED",
                           "TRAIN_ENV", "ASSETS_READY", "TEACHER_READY",
                           "ROPE_OK", "VLLM_READY", "TESTS_OK",
                           "AUTHORIZATION_OK", "SETUP_DONE"),
            env={"SESSION_KIND": "c2_behavioural",
                 "SESSION_FROZEN_EXPECT": FROZEN_EXPECT},
            uv_max_seconds=args.uv_max_s, tests_max_seconds=args.tests_max_s,
            teacher_revision=TEACHER_REVISION,
            test_ignores=TEST_IGNORES),
        driver_command=driver_command,
        #: The replacement-resource handoff, AFTER setup and BEFORE the driver
        #: starts. The runner tears the pod down if it returns False.
        materialize_inputs=restore_campaign_probes,
        driver_job_id="autoinit_c2_behavioural_driver",
        status_path=STATUS, run_log_path=RUN_LOG,
        markers=MarkerPolicy(
            success="C2_BEHAVIOURAL_ALL_DONE",
            failure=("C2_BEHAVIOURAL_FAILED",),
            incomplete=(),
            #: Gated on probes EXISTING, never on the session succeeding. A run
            #: that trained nine probes and then failed has nine checkpoints
            #: that cost real GPU time and may not be retrained for a different
            #: outcome. `if terminal == "ALL_DONE"` has already deleted $2.82 of
            #: verified checkpoints in this programme.
            products_eligible=lambda terminal, stages: True,
            failure_note=(
                "the session stopped before its verdict. Whatever probes "
                "FINISHED are fetched and verified first. A provenance "
                "mismatch on an arm is a scientific finding, not a retryable "
                "engineering failure, and goes to review unchanged.")),
        artifacts=ArtifactPolicy(
            audit_dirname=AUDIT_DIRNAME,
            evidence_filename="c2_behavioural_evidence.json",
            archive_basename="c2_behavioural_artifacts.tar.gz",
            spec_success="configs/autoinit/c2_behavioural_artifacts.json",
            spec_failed="configs/autoinit/c2_behavioural_artifacts_failed.json",
            report_names=("c2_behavioural_evidence.json",),
            on_poll=secure_finished_probes,
            fetch_products=fetch_probes,
            products_secured=probes_secured),
        teardown=TeardownPolicy(
            note="delete the pod, verify from the provider that it is gone, STOP"),
        precheck=(
            session_commit_gate(REPO_ROOT,
                                auth_path_for(getattr(args, "run_id", "")),
                                check_lineage=True),
            scope_gate,
            plan_binding_gate,
            source_binding_gate,
            destination_gate,
            container_gate,
            #: BEFORE the money gate, because "the bytes this continuation
            #: needs are not staged" is a cheaper and more actionable refusal
            #: than "the campaign cannot afford the work those bytes feed".
            volume_gate,
            campaign_continuation_gate,
            readiness_gate,
            #: LAST, because it is the only gate that touches the network.
            bundle_staged_gate,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Every attribute the RUNNER reads must come from THIS parser.

    Not from a hand-written namespace: another session died at `$0.0603` on an
    attribute a namespace had and the parser did not, after the pod was
    billing. And every required option must be fillable by a string, because
    `test_no_launcher_was_dropped_because_it_would_not_parse` probes each
    launcher with `"kind_probe"` for unconstrained required flags — a required
    `type=float` flag makes the launcher unparseable and drops it silently out
    of the dispatch enumeration, which is what the first version of this parser
    did with `--max-price`.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--run-id", required=True, action=_RunIdSetsOut,
                    help="the attempt this session runs as, e.g. attempt1")
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-transport")
    ap.add_argument("--image", default=BOUND_IMAGE)
    #: L40S. The probe minutes this session is priced from were measured on one,
    #: and the six arms must reproduce digests that were produced on one.
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=None,
                    help="defaults to the authorization's own accepted rate")
    ap.add_argument("--disk-gb", type=int, default=CONTAINER_DISK_GB)
    ap.add_argument("--ckpt-store", default=DURABLE_STORE,
                    help="where finished probes are secured off-pod")
    ap.add_argument("--ckpt-fetch-limit-min", type=int, default=20)
    #: THE PRE-STAGED BACKEND. A continuation's probes are already on this
    #: volume when the pod boots, so the restore direction — which used to be
    #: the slow one, ~9 hours of L40S time against a 0.72 MB/s uplink — is a
    #: presence and identity check rather than a transfer. `--restore-limit-min`
    #: is gone with the transfer it bounded.
    ap.add_argument("--network-volume-id", default=CAMPAIGN_VOLUME_ID,
                    help="provider network volume holding this campaign's "
                         "pre-staged probes")
    ap.add_argument("--volume-mount-path", default=VOLUME_MOUNT,
                    help="where that volume is mounted on the pod; never "
                         "/workspace, which is the checkout root")
    ap.add_argument("--volume-gb", type=int, default=CAMPAIGN_VOLUME_GB,
                    help="the volume's provisioned size; df at the mount "
                         "reports the backing cluster, not the quota")
    ap.add_argument("--data-center-ids", default=VOLUME_DATACENTER,
                    help="a volume can only be attached from its own "
                         "datacenter, so the draw is constrained to it. "
                         "CLEARED, with the volume, when no remaining "
                         "operation reads a pre-staged checkpoint")
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
    #: DERIVED, not defaulted. `main` sets it from the authorization's own
    #: window: the shorter of what its GPU dollars buy at the accepted rate and
    #: its authorized runtime. A literal default here was 1800.0 against a
    #: 1800.53-minute authorized runtime, so the generic number silently
    #: truncated the authorized experiment by half a minute; a larger literal
    #: would have been worse in the other direction.
    ap.add_argument("--poll-limit-min", type=float, default=None)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--uv-max-s", type=int, default=2400)
    ap.add_argument("--tests-max-s", type=int, default=1800)
    #: The probe-cost bounds the driver spends admission control against.
    ap.add_argument("--b-build-minutes", type=float, default=31.0)
    ap.add_argument("--probe-train-minutes", type=float, default=75.0)
    ap.add_argument("--probe-battery-minutes", type=float, default=45.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every $0 gate and stop before provider creation")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    #: THROUGH ITS REAL LOADER, before anything is priced. `load` verifies the
    #: document against its own hash, its schema, its scope, its campaign and
    #: the internal consistency of its five amounts; reading the rate out with
    #: `json.loads(...).get(...)` — which this did — would accept a document
    #: whose dollars and deadline described different experiments.
    auth_file = REPO_ROOT / auth_path_for(args.run_id)
    if not auth_file.is_file():
        raise SystemExit(
            f"{auth_path_for(args.run_id)} does not exist. Every amount this "
            "launcher derives — the rate, the window, the campaign ceiling — "
            "comes from the authorization, so there is nothing to derive them "
            "from and nothing to launch.")
    auth = BG.BehaviouralAuthorization.load(auth_file)

    if args.max_price is None:
        #: The rate the authorization's own ceiling was derived at.
        args.max_price = float(auth.rate_usd_per_hour)
    elif args.max_price > float(auth.rate_usd_per_hour) + BG.DOLLAR_QUANTUM_USD:
        #: A `$0` refusal, before `check_gpu_offered` and long before `create`.
        #: The runner already refuses a live quote above `--max-price`, at `$0`
        #: pre-provider and by confirmed teardown post-provider; that protection
        #: is only worth the authorized rate if `--max-price` cannot be raised
        #: above it here.
        raise SystemExit(
            f"--max-price ${args.max_price}/h is above the ${auth.rate_usd_per_hour}"
            "/h this authorization's ceiling was derived at. A price increase is "
            "not solved by paying more per hour, and it is not solved by "
            "shortening the scientific experiment either: re-quote, regenerate "
            "the proposal, and take a materially changed dollar authorization "
            "back to the maintainer.")

    #: The deadline is the SHORTER of what the authorized dollars buy at the
    #: live rate and the authorized runtime, and both bounds come from the
    #: authorization. Deriving it from `QUOTED_RATE_USD_PER_HOUR` instead —
    #: which this did — meant a valid authorization re-derived at a different
    #: live quote still inherited the old `$1.09/h` dollar window and could
    #: truncate the experiment; and a card cheaper than the authorized rate
    #: funds more minutes than the experiment is authorized to use.
    window = BG.window_minutes(
        args.max_price, gpu_hard_usd=float(auth.gpu_hard_usd),
        hard_runtime_minutes=float(auth.hard_runtime_minutes))
    args.poll_limit_min = (window if args.poll_limit_min is None
                           else min(args.poll_limit_min, window))

    #: BEFORE anything is priced or created: a colliding run id or a foreign
    #: scratch root costs $0 here.
    #: A DRY RUN MUST NOT CONSUME THE CHAIN IT EXISTS TO DE-RISK. Every
    #: invocation records the run — deliberately, so a launcher that dies
    #: mid-flight cannot be silently re-invoked against one authorization — and
    #: `open_run` then refuses a recorded or occupied directory. A dry run
    #: inherited both: it advertises "run every $0 gate and stop before
    #: provider creation", consumed attempt4's one-use chain at $0, and the
    #: real launch that followed seconds later was refused by the occupancy
    #: rule. The flag was a trap, and the trap fired.
    #:
    #: So a dry run writes its evidence to its OWN run directory and leaves the
    #: real one pristine. `args.run_id` is NOT redirected: every gate resolves
    #: the grant, readiness record, authorization, bundle and the campaign's
    #: prior attempts from it, so a dry run under a different id would check a
    #: different chain and prove nothing about this one. Only the OUTPUT
    #: locations move.
    #: UNDERSCORE, not a hyphen. `run_layout` validates a run id against
    #: `^[a-z0-9][a-z0-9_]*$` and refuses anything else rather than resolving
    #: it, so `attempt6-dryrun` raised inside `open_run` -- and the repair that
    #: introduced it was itself a repair for the dry run CONSUMING the chain.
    #: It crashed at $0 before any output was claimed, so it consumed nothing;
    #: but a rehearsal that cannot run is a rehearsal nobody gets.
    layout_run_id = (f"{args.run_id}{DRY_RUN_SUFFIX}" if args.dry_run
                     else args.run_id)
    if args.dry_run:
        args.out = session_record_path(layout_run_id)
    claim_output_root(args.scr, EXPERIMENT_ID, layout_run_id,
                      outputs=("store", "relay"))
    layout = open_run(REPO_ROOT, EXPERIMENT_ID, layout_run_id,
                      roles=BEHAVIOURAL_RUN_ROLES, prepared=_RUN_PREPARED,
                      stage_id=STAGE_ID)
    write_run_readmes(layout, experiment_id=EXPERIMENT_ID,
                      run_id=layout_run_id, stage_id=STAGE_ID,
                      roles=BEHAVIOURAL_RUN_ROLES)
    assert args.out == session_record_path(layout_run_id), (
        args.out, layout_run_id)

    #: THE VOLUME IS ATTACHED ONLY IF SOMETHING READS IT, and this must happen
    #: before `run_session`: the gates and `SessionRunner.create` both read
    #: these args, so deciding here is what makes them agree. Attaching pins
    #: the draw to one datacenter, which is why it is not a harmless default —
    #: see `volume_attachment`.
    attachment = volume_attachment(args)
    print(f"volume: {'attached' if attachment['attach'] else 'not attached'} — "
          f"{attachment['why']}\n")

    #: `record_run` in a `finally`, not after a successful return. A run_session
    #: that RAISES leaves the run directory populated and unrecorded, and the
    #: next invocation is then refused by the occupancy rule — correctly, since
    #: an unrecorded directory is indistinguishable from a launcher that died
    #: mid-flight. The chain is consumed by the invocation whether or not a
    #: provider resource followed.
    rc = 1
    try:
        rc = run_session(spec(args), args, REPO_ROOT,
                         summary=("the behavioural selection is a TERMINUS: it "
                                  "names a C2 incumbent or it does not, and "
                                  "NO_GO and INCONCLUSIVE are complete "
                                  "results. C3 is separately authorized and "
                                  "unreachable from here."))
    finally:
        try:
            record_run(
                layout, spec=BEHAVIOURAL_RUN_SPEC,
                plan={"session": BG.SESSION_ID, "plan_id": BG.PLAN_ID,
                      #: The scientific campaign, recorded beside the run
                      #: attempt. The continuation gate reads prior attempts of
                      #: THIS campaign, so a run that does not say which
                      #: campaign it belonged to is a run a later attempt
                      #: cannot reconcile.
                      "campaign_id": BG.CAMPAIGN_ID,
                      "run_attempt": args.run_id,
                      "session_commit": args.session_commit,
                      "bundle": args.bundle,
                      "plan_hash": BG.plan_hash(REPO_ROOT),
                      "protocol_sha256":
                          BH.protocol(REPO_ROOT)["protocol_sha256"]},
                implementation={
                    "launcher": "scripts/pod/autoinit_c2_behavioural_launch.py",
                    "driver": "scripts/pod/autoinit_c2_behavioural_driver.py"},
                status={"authorizes": "nothing",
                        "terminates_at": "decide",
                        "decides": "the C2 incumbent, under the frozen "
                                   "Phase-C rule, or no incumbent at all"},
                #: PRESENT roles only, under the parameter name `record_run`
                #: actually takes. This call passed `present=` and `stage_id=`,
                #: neither of which exists in that signature, so every
                #: invocation raised `TypeError` into the `except` below and
                #: printed a warning — the run manifest was never written, on
                #: any path, including the `$0` refusals whose only evidence it
                #: is. A swallowed exception in a `finally` is exactly where a
                #: signature mismatch can hide forever.
                roles=present_roles(layout, BEHAVIOURAL_RUN_ROLES))
        except Exception as exc:                                # noqa: BLE001
            print(f"\nRUN NOT RECORDED: {type(exc).__name__}: {exc}\n"
                  f"  the run directory is "
                  f"{rel_run_dir(EXPERIMENT_ID, args.run_id, STAGE_ID)}; it "
                  f"holds whatever this invocation produced and the chain is "
                  f"consumed either way.\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
