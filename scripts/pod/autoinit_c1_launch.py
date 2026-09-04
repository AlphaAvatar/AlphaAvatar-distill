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

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# `renderer_parity_gate` lives with the other dev-box verifiers, and the eleventh
# pre-provider gate executes it directly rather than trusting a transcript of it.
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

from aadistill.autoinit import c1_session as CS  # noqa: E402
from aadistill.autoinit.c1_authorization import (  # noqa: E402
    C1_HARNESS_SOURCE_FILES_V1,
    C1Authorization,
    c1_budget_spec,
    c1_hard_ceiling_usd,
    c1_harness_digest,
)
from aadistill.autoinit.c1_bundle import (  # noqa: E402
    RELAY_REPO as RELAY_REPO_ID, C1BundleError, canonical_bundle_name,
    hf_download, require_canonical_bundle_arg, roundtrip,
)
from aadistill.autoinit.c1_isolation import derive_recovery_seeds  # noqa: E402
from aadistill.autoinit.pod_environment import (  # noqa: E402
    RECORD_PATH as POD_ENV_RECORD, load_record as load_pod_env_record,
    verify_record as verify_pod_env_record,
)
from aadistill.infrastructure.manifest import (  # noqa: E402
    sha256_file, sha256_json,
)
from aadistill.infrastructure.session import (  # noqa: E402
    ArtifactPolicy, LocalAsset, MarkerPolicy, SessionContext, SessionSpec,
    SetupManifest, TeardownPolicy,
)
from aadistill.infrastructure.session_prechecks import (  # noqa: E402
    session_commit_gate,
)
from aadistill.infrastructure.session_runner import REPO, WS, run_session  # noqa: E402
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
               dest="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint", sha256=sha)
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
               sha256="a7131bb092b38a078edc213961f0eb57eaead24f1396e25741f4887b1a694054"),
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
TEST_IGNORES = ("tests/data/test_recovery_corpus_pipeline.py",
                "tests/pod/test_phase_a_stages1_5_execute.py")

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
#: Written by scripts/autoinit/stage_c1_bundle.py; the local half of the
#: transport check. The gate verifies the REMOTE object against it.
BUNDLE_RECORD = "logs/autoinit_c1_bundle.json"

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
    credential file. Twelve of the fourteen failures were environment.

    The sweep that answers this takes about three quarters of an hour, far too
    slow to run while a pod bills. So it is run once against a committed tree and
    recorded, and this gate checks only that the recording still describes the
    code that would run: the C1 harness digest and the pod test-environment digest
    must both still match. Documentation and preregistration commits do not
    invalidate it; an edit to the setup script, the simulator, the test tree or
    the harness does.
    """
    try:
        record = load_pod_env_record(REPO_ROOT)
    except FileNotFoundError:
        return False, (f"{POD_ENV_RECORD} does not exist: no pod-like sweep has "
                       "been recorded for this executable")
    except Exception as exc:                                   # noqa: BLE001
        return False, f"cannot read {POD_ENV_RECORD}: {exc}"

    ok, reason = verify_pod_env_record(record, REPO_ROOT)
    ctx.evidence["pod_environment_verification"] = {
        "verdict": "PASS" if ok else "FAIL",
        "record": POD_ENV_RECORD,
        "record_self_sha256": record.get("self_sha256"),
        "swept_at_head": record.get("executable_head"),
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

    staged = REPO_ROOT / BUNDLE_RECORD
    if not staged.is_file():
        return False, (f"{BUNDLE_RECORD} is missing; run "
                       f"scripts/autoinit/stage_c1_bundle.py --session-commit "
                       f"{commit} first")
    record = json.loads(staged.read_text())
    if record.get("session_commit") != commit:
        return False, (f"{BUNDLE_RECORD} describes a bundle for "
                       f"{str(record.get('session_commit'))[:12]}…, not the session "
                       f"commit {commit[:12]}…")

    auth_bytes = (REPO_ROOT / AUTH_PATH).read_bytes()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = roundtrip(
                session_commit=commit,
                local_bundle_sha256=record["sha256"],
                authorization_bytes=auth_bytes,
                authorization_path=AUTH_PATH,
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
            relay_inputs=(*RECOVERY_LADDER, *CALIBRATION_V1,
                           *C1_EVAL_TOKENIZER, *C1_ROPE_INPUT),
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
            report_names=("c1_evidence.json",
                          "c1_attested_evaluation_protocol.json",
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
    """C1's session command line. No search flag exists to be set."""
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scr", required=True)
    ap.add_argument("--session-commit", required=True)
    ap.add_argument("--bundle", required=True,
                    help="must be the canonical name derived from\n"
                         "--session-commit; an alias fails at $0")
    ap.add_argument("--relay-repo", default="AlphaAvatar/aadistill-artifacts")
    ap.add_argument("--image",
                    default="runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404")
    ap.add_argument("--gpu", default="NVIDIA L40S")
    ap.add_argument("--max-price", type=float, default=0.99)
    #: Two arm materializations plus six probe checkpoints and their generations.
    ap.add_argument("--disk-gb", type=int, default=200)
    ap.add_argument("--probe-train-minutes", type=float, default=70.0)
    ap.add_argument("--probe-battery-minutes", type=float, default=25.0)
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
    #: The session is priced at ~14 GPU-hours; the poll limit must outlast the
    #: hard threshold or the launcher would stop watching a billing pod.
    ap.add_argument("--poll-limit-min", type=float, default=1320.0)
    ap.add_argument("--settle-seconds", type=float, default=20.0)
    ap.add_argument("--runpod-config",
                    default=os.path.expanduser("~/.runpod/config.toml"))
    ap.add_argument("--out", default="logs/autoinit_c1_session.json")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    return run_session(spec(args), args, REPO_ROOT,
                       summary=("STOP for review. C1 replayed one frozen path "
                                "under two digest gates and ran no search."))


if __name__ == "__main__":
    raise SystemExit(main())
