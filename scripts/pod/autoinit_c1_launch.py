#!/usr/bin/env python3
"""Phase C1 — fixed-path ATTENTION isolation, as a session specification.

    PYTHONPATH=src setsid nohup python -u scripts/pod/autoinit_c1_launch.py \
        --scr <scratch> --session-commit <sha> --bundle <name> < /dev/null &

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
`logs/phase_c1_pricing.json` and back-derives the step time from its measured
per-probe minutes, so the enforceable ceiling exists in exactly one place. A
second hand-maintained figure is how a session comes to be authorized for one
number and priced at another.

*The authorization is a distinct type with its own setup branch.*
`SESSION_KIND=c1` is declared, because an undeclared kind falls through to
`spend` and loads a `SpendAuthorization` — which is what killed Phase-B attempt 2
at `$0.2300`, one step after its test gate passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aadistill.autoinit import c1_session as CS  # noqa: E402
from aadistill.autoinit.c1_authorization import (  # noqa: E402
    C1_HARNESS_SOURCE_FILES_V1,
    C1Authorization,
    c1_budget_spec,
    c1_hard_ceiling_usd,
    c1_harness_digest,
)
from aadistill.autoinit.c1_isolation import derive_recovery_seeds  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, LocalAsset, MarkerPolicy, SessionContext, SessionSpec,
    SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
from autoinit_phase_a_launch import (  # noqa: E402
    TEACHER_REVISION, TEST_IGNORES, build_parser as phase_a_parser,
)
from autoinit_science_inputs import CALIBRATION_V1, RECOVERY_LADDER  # noqa: E402

STATUS = f"{WS}/autoinit_c1.status"
RUN_LOG = f"{WS}/autoinit_c1_run.log"
AUTH_PATH = "logs/autoinit_c1_authorization.json"

PRICING = "logs/phase_c1_pricing.json"
PREREG = "logs/phase_c1_execution_preregistration.json"
#: Declared once. `artifact_spec_gate` reads these and `ArtifactPolicy` books
#: them, so the gate cannot end up validating a different file than the one the
#: pod is handed.
SPEC_SUCCESS = "configs/autoinit/c1_artifacts.json"
SPEC_FAILED = "configs/autoinit/c1_artifacts_failed.json"
BATTERY_MANIFEST = "artifacts/stage3/c1_confirmation_v1/manifest.json"
BATTERY_IDENTITY = "logs/phase_c1_battery.json"
TEACHER_BINDING = "logs/phase_c1_teacher_binding.json"

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
        live = c1_harness_digest(REPO_ROOT)["digest"]
    except Exception as exc:                       # noqa: BLE001
        return False, f"cannot compute the C1 harness digest: {exc}"
    declared = tuple(getattr(ctx.auth, "harness_source_files", ()) or ())
    if declared != C1_HARNESS_SOURCE_FILES_V1:
        return False, ("the authorization declares a different harness file set "
                       "than this session executes")
    stored = getattr(ctx.auth, "harness_source_digest", None)
    if stored and stored != live:
        return False, (f"harness digest {stored[:12]}… in the authorization does "
                       f"not match the live tree {live[:12]}…")
    return True, f"C1 harness {live[:12]}… over {len(C1_HARNESS_SOURCE_FILES_V1)} files"


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
    live = c1_harness_digest(REPO_ROOT)["digest"]
    recorded = (doc.get("c1_harness") or {}).get("digest")
    if recorded != live:
        return False, (f"the preregistration records harness "
                       f"{str(recorded)[:12]}… but the tree digests to {live[:12]}…")
    if doc.get("authorizes") != "nothing":
        return False, "the preregistration claims to authorize something"
    return True, f"preregistration {doc.get('preregistration_sha256', '?')[:12]}…"


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
        "probe_journal": n_probes,
        "generations": n_probes * n_sets,
        "generation_summary": n_probes * n_sets,
        "arm_identities": 1,
        "replay_record": 1,
        "decision": 1,
        "attested_protocol": 1,
        "session_evidence": 1,
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
    """The C1 driver. There is no argument that searches, ranks or eliminates."""
    return (f"/opt/train/bin/python {REPO}/scripts/pod/autoinit_c1_driver.py "
            f"--stage all --image-digest '{ctx.image_digest}' "
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
    return SessionSpec(
        session_id="autoinit-c1",
        schema="aadistill.autoinit.c1_session/v1",
        description=("Phase C1: fixed-path ATTENTION isolation. Replays the frozen "
                     "fe9683 path under two digest gates, then runs 2 arms x 3 "
                     "fresh seeds. Runs no search"),
        authorization_path=AUTH_PATH,
        #: The C1 type. A Phase-A, Phase-B or continuation artifact is refused by
        #: schema at load: each measures a different harness and carries a
        #: ceiling derived for different work.
        authorization_loader=C1Authorization.load,
        plan_id="autoinit.v1.phase_c1",
        #: The isolation plan's own hash, not Phase A's. C1 is different science,
        #: not a different operational identity for the same science.
        plan_hash=_plan_hash(),
        #: DERIVED from logs/phase_c1_pricing.json. Never written here.
        budget=c1_budget_spec(REPO_ROOT),
        setup=SetupManifest(
            relay_inputs=(*RECOVERY_LADDER, *CALIBRATION_V1),
            local_assets=LOCAL_ASSETS,
            #: Declared. Without it setup falls to SESSION_KIND=spend and loads a
            #: SpendAuthorization, which refuses this artifact — the session
            #: would die at setup, exit 98, before any work.
            env={"SESSION_KIND": "c1"},
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
            failure_note=(
                "a blocking stage failed — collecting evidence, then tearing "
                "down. C1_REPLAY_MISMATCH is the scientific stop: the frozen path "
                "did not reproduce its recorded digest, no recovery training was "
                "started, and the mismatch evidence is the session's product.")),
        artifacts=ArtifactPolicy(
            audit_dirname="autoinit_c1",
            evidence_filename="c1_evidence.json",
            archive_basename="c1_artifacts.tar.gz",
            spec_success=SPEC_SUCCESS,
            spec_failed=SPEC_FAILED,
            report_names=("c1_evidence.json", "attested_evaluation_protocol.json",
                          "c1_replay_record.json", "c1_arm_identities.json",
                          "c1_probe_results.json", "c1_decision.json"),
            event_streams=probe_streams),
        teardown=TeardownPolicy(note="nothing chains off C1"),
        precheck=(
            session_commit_gate(REPO_ROOT, AUTH_PATH, check_lineage=True),
            c1_harness_gate,
            pricing_identity_gate,
            preregistration_gate,
            frozen_c1_science_gate,
            teacher_binding_gate,
            battery_staged_gate,
            artifact_spec_gate,
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
    from aadistill.autoinit.c1_isolation import C1Arm, C1IsolationPlan
    from aadistill.autoinit.operators import attention_activation

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
    ap = phase_a_parser()
    ap.set_defaults(out="logs/autoinit_c1_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("STOP for review. C1 replayed one frozen path "
                                "under two digest gates and ran no search."))


if __name__ == "__main__":
    raise SystemExit(main())
