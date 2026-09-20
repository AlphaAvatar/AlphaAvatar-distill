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
from aadistill.runtime.staging_contract import (  # noqa: E402
    ignores_for_selection)

from autoinit_science_inputs import CALIBRATION_V1, RECOVERY_LADDER  # noqa: E402
from autoinit_c1_launch import C1_EVAL_TOKENIZER, C1_ROPE_INPUT  # noqa: E402
from experiments.deployment import deployment_commands  # noqa: E402
from experiments.phase_c2 import behavioural as BH  # noqa: E402
from experiments.phase_c2 import behavioural_bundle as BT  # noqa: E402
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
DURABLE_STORE = "/home/ecs-user/aad-artifacts/phase_c2_behavioural"


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
    sched = BH.schedule(REPO_ROOT)
    need = int(sched["total_probes"]) * int(1.11 * 2**30)
    if free < need:
        return False, (
            f"{store} has {free / 2**30:.1f} GiB free and the twelve probes it "
            f"must hold need {need / 2**30:.1f} GiB. A run that trains work it "
            "cannot preserve is a run that will lose it.")
    return True, (f"destination OK: {free / 2**30:.1f} GiB free at {store} for "
                  f"{need / 2**30:.1f} GiB of probes")


def campaign_attempts(campaign_id: str, *, exclude: str = "",
                      store: str | Path = DURABLE_STORE) -> list[str]:
    """Run attempts of this campaign that left durable probes, in name order.

    Name order, not chronological — `attempt10` sorts before `attempt2` — and
    the gate does not care, because it sums every prior attempt's spend and
    reconciles every prior resource rather than looking at the latest one.

    The durable store is the authority, not the run log: a run attempt whose
    launcher died before it recorded itself still produced probes, and those
    probes are the thing a continuation would consume.
    """
    root = campaign_store(campaign_id, store)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and d.name != exclude
                  and any(d.iterdir()))


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
      this session's planned all-in against `all_in_hard_usd`.

    This gate fails CLOSED and says why. With a ceiling sized for one full
    session, a continuation after a resource that already spent real money will
    be REFUSED here rather than permitted to overspend — funding a campaign for
    more than one full session is a maintainer decision, and this gate is where
    that need becomes visible instead of becoming an overrun.
    """
    campaign = ctx.auth.campaign_id
    prior = campaign_attempts(campaign, exclude=ctx.args.run_id,
                              store=ctx.args.ckpt_store)
    ctx.evidence["campaign"] = {
        "campaign_id": campaign, "run_attempt": ctx.args.run_id,
        "prior_attempts_with_durable_probes": prior,
        "_continuation_is_not_pooling": (
            "a replacement resource is a new RESOURCE and a new run attempt "
            "inside the SAME scientific campaign. Consuming its predecessor's "
            "destination-verified probes is continuation of one preregistered "
            "experiment, not pooling across experiments."),
    }
    if not prior:
        return True, (f"campaign {campaign} has no prior run attempt holding "
                      "durable probes; this is its first resource")

    from experiments.run_layout import rel_run_dir

    settled, unconfirmed, unreadable = 0.0, [], []
    for attempt in prior:
        record = REPO_ROOT / session_record_path(attempt)
        if not record.is_file():
            unreadable.append(attempt)
            continue
        try:
            ev = json.loads(record.read_text())
        except json.JSONDecodeError:
            unreadable.append(attempt)
            continue
        settled += float((ev.get("cost") or {}).get("actual_usd") or 0.0)
        if ev.get("provider_resource_created") and not ev.get(
                "provider_confirms_gone"):
            unconfirmed.append(attempt)
    if unreadable:
        return False, (
            f"campaign {campaign} has durable probes from run attempt(s) "
            f"{unreadable} whose session record could not be read at "
            f"{[session_record_path(a) for a in unreadable]}. Whether those "
            "resources are still billing is therefore UNKNOWN, and an unknown "
            "billing state is a stop condition, not a clear one. Reconcile "
            "them before launching another.")
    if unconfirmed:
        return False, (
            f"run attempt(s) {unconfirmed} of campaign {campaign} created a "
            "provider resource that was never confirmed released. At most one "
            "resource of a campaign may bill at a time, and a zero return code "
            "on a remove call is not evidence of release. Reconcile and tear "
            "down before creating another. Run dir(s): "
            f"{[rel_run_dir(EXPERIMENT_ID, a, STAGE_ID) for a in unconfirmed]}")

    planned = float(ctx.auth.gpu_hard_usd) + float(ctx.auth.disk_hard_usd)
    approved = float(ctx.auth.all_in_hard_usd)
    ctx.evidence["campaign"].update({
        "settled_campaign_spend_usd": round(settled, 4),
        "this_session_planned_all_in_usd": round(planned, 4),
        "campaign_approved_all_in_usd": approved,
    })
    if settled + planned > approved + BG.DOLLAR_QUANTUM_USD:
        return False, (
            f"campaign {campaign} has settled ${settled:.4f} across "
            f"{len(prior)} prior run attempt(s) and this session plans "
            f"${planned:.4f} all-in, which is ${settled + planned:.4f} against "
            f"an approved campaign ceiling of ${approved:.4f}. The ceiling is "
            "cumulative across every resource and subrun; a replacement "
            "resource does not receive a fresh allocation. Continuing would "
            "need a maintainer decision to fund the campaign for more than one "
            "full session — it is not something this gate may grant.")
    return True, (
        f"campaign continuation OK: ${settled:.4f} settled across {len(prior)} "
        f"prior attempt(s) plus ${planned:.4f} planned is inside the "
        f"${approved:.4f} campaign ceiling, and every prior resource is "
        "provider-confirmed released")


def readiness_gate(ctx: SessionContext) -> tuple[bool, str]:
    """A launch-bound readiness record for THIS run, on THIS tree."""
    run_id = getattr(ctx.args, "run_id", "")
    try:
        record = BPE.load_record(REPO_ROOT, run_id=run_id, stage_id=STAGE_ID)
    except FileNotFoundError as exc:
        return False, f"no behavioural readiness record for this run: {exc}"
    try:
        BPE.verify_record(record, REPO_ROOT, run_id=run_id, stage_id=STAGE_ID)
    except Exception as exc:                                    # noqa: BLE001
        return False, f"the readiness record does not describe this tree: {exc}"
    if record.get("kind") != BPE.LAUNCH_BOUND:
        return False, (
            f"the readiness record is {record.get('kind')!r}, not "
            f"{BPE.LAUNCH_BOUND!r}. Only a launch-bound sweep describes the tree "
            "a launch will use.")
    return True, f"readiness OK ({record.get('kind')})"


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
    return fetched


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
        if not pending:
            return
        fetched = _fetch_and_verify(ctx, pending)
        #: Anything that ARRIVED is done, verified or not. Keying only on
        #: `matched` would re-scp an unverifiable probe on every poll for the
        #: rest of the session, and a retry cannot make an identity the driver
        #: never computed appear.
        ctx.evidence.setdefault("probes_secured", []).extend(
            r for r in fetched
            if r.get("rc") == 0 and (r.get("matched")
                                     or not r.get("identity_announced")))
        ctx.evidence.setdefault("probe_fetch_attempts", []).extend(fetched)
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

    prep = BG.materialization_minutes(REPO_ROOT)
    d = BH.session_decomposition(
        REPO_ROOT, materialization_minutes=prep["total_minutes"])

    return BudgetSpec(
        #: `arms=0` and both generic phases at zero: the generic
        #: arms x steps + setup + transfer shape does not describe six
        #: materializations followed by twelve probes, and every phase this
        #: session has is named in the decomposition instead.
        arms=0, steps_per_arm=0,
        step_seconds=MEASURED_STEP_SECONDS,
        step_source=("unused: the generic arms x steps term does not describe "
                     "six materializations followed by twelve probes. Every "
                     "phase is named in BH.session_decomposition, which is "
                     "derived from committed records and reconciled against "
                     "them."),
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
    claim_output_root(args.scr, EXPERIMENT_ID, args.run_id,
                      outputs=("store", "relay"))
    layout = open_run(REPO_ROOT, EXPERIMENT_ID, args.run_id,
                      roles=BEHAVIOURAL_RUN_ROLES, prepared=_RUN_PREPARED,
                      stage_id=STAGE_ID)
    write_run_readmes(layout, experiment_id=EXPERIMENT_ID,
                      run_id=args.run_id, stage_id=STAGE_ID,
                      roles=BEHAVIOURAL_RUN_ROLES)
    assert args.out == session_record_path(args.run_id), (args.out, args.run_id)

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
