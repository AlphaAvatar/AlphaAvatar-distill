"""Rehearsal of the characterization continuation: every way it must refuse.

The continuation session spends money on one thing — characterizing two controls
it did not train — so the load-bearing property is that an imported artifact
which is not the permanent control is rejected *before* any measurement is bound
to it.

    wrong imported weights hash          -> import refused
    wrong control_binding                -> import refused
    missing material evidence            -> import refused, field named
    attestation mismatch                 -> characterization refused
    v2 tool smoke failure                -> stop before the controls
    normal success path                  -> through collection and teardown

The happy path runs against the **real** permanent controls where they are
staged, so the gate is exercised on the artifacts it will actually admit rather
than on a fixture built to satisfy it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from aadistill.autoinit.continuation import (  # noqa: E402
    CONTINUATION_AUTHORIZATION,
    CONTINUATION_PLAN_V1,
    CONTINUATION_SCOPE,
    IMPORT_REQUIRED_FIELDS,
    ControlImportError,
    continuation_manifest,
    import_permanent_control,
)
from aadistill.autoinit.recovery import RecoveryAdmissionError  # noqa: E402

RECORDS = REPO / "logs/autoinit_permanent_controls"
STAGED = Path("/home/ecs-user/aad-artifacts/autoinit")
CONTROLS = ("preflight_ctl_r0860k_sa", "preflight_ctl_r0860k_sb")

have_controls = pytest.mark.skipif(
    not (STAGED / CONTROLS[0] / "step_001023/model/model.safetensors").is_file(),
    reason="the permanent controls are local artifacts, not tracked in git")


def staged(name: str) -> Path:
    return STAGED / name / "step_001023/model"


def evidence_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / f"evidence_{name}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(RECORDS / f"{name}_run_manifest.json", d / "run_manifest.json")
    shutil.copy(RECORDS / f"{name}_run_completion.json", d / "run_completion.json")
    return d


def record_copy(tmp_path: Path, name: str, **edits) -> Path:
    record = json.loads((RECORDS / f"{name}_probe_identity.json").read_text())
    for dotted, value in edits.items():
        target, _, leaf = dotted.rpartition(".")
        node = record
        for part in filter(None, target.split(".")):
            node = node[part]
        if value is _DELETE:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    path = tmp_path / f"{name}_record.json"
    path.write_text(json.dumps(record, indent=2))
    return path


class _Delete:
    pass


_DELETE = _Delete()


# --- the success path -------------------------------------------------------


@have_controls
def test_the_real_permanent_controls_import(tmp_path):
    """The gate admits exactly what it should, on the real artifacts."""
    for name, seed in zip(CONTROLS, (20260726, 20260801)):
        imported = import_permanent_control(
            name, record_path=RECORDS / f"{name}_probe_identity.json",
            checkpoint_dir=staged(name),
            run_evidence_dir=evidence_dir(tmp_path, name),
            repo_root=REPO, strict=True)
        assert imported.seed == seed
        assert imported.reconstructed_from_run_evidence, (
            "the protocol must be re-derived from the run's own evidence, not "
            "taken from the record")
        reconstruction = imported.evidence["reconstruction"]
        assert reconstruction["fingerprint"] == imported.observed_protocol_fingerprint
        assert reconstruction["missing_fields"] == []
        assert reconstruction["step_accounting"]["completed_all_steps"] is True
        assert reconstruction["pack_recomputed"].startswith("6f324cb0")
        assert imported.weights_sha256 == \
            imported.control_binding["checkpoint_weights_sha256"]
    # Both controls share one protocol and differ only by seed — the
    # single-variable structure Phase A needs, re-established on import.
    a, b = (import_permanent_control(
        n, record_path=RECORDS / f"{n}_probe_identity.json",
        checkpoint_dir=staged(n), run_evidence_dir=evidence_dir(tmp_path, n),
        repo_root=REPO, strict=True) for n in CONTROLS)
    assert a.observed_protocol_fingerprint == b.observed_protocol_fingerprint
    assert a.seed != b.seed and a.probe_id != b.probe_id
    assert a.initialization_artifact_digest == b.initialization_artifact_digest


# --- 1. wrong imported weights ----------------------------------------------


@have_controls
def test_wrong_imported_weights_hash_is_refused(tmp_path):
    name = CONTROLS[0]
    record = record_copy(tmp_path, name, weights_sha256="0" * 64)
    with pytest.raises(ControlImportError, match="not the same checkpoint"):
        import_permanent_control(name, record_path=record,
                                 checkpoint_dir=staged(name),
                                 run_evidence_dir=evidence_dir(tmp_path, name),
                                 repo_root=REPO, strict=True)


@have_controls
def test_a_checkpoint_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(ControlImportError, match="nothing was staged"):
        import_permanent_control(
            CONTROLS[0], record_path=RECORDS / f"{CONTROLS[0]}_probe_identity.json",
            checkpoint_dir=tmp_path / "absent", repo_root=REPO, strict=True)


# --- 2. wrong control_binding -----------------------------------------------


@have_controls
def test_a_control_binding_that_disagrees_is_refused(tmp_path):
    name = CONTROLS[0]
    for edit, expect in (
        ({"control_binding.checkpoint_weights_sha256": "1" * 64},
         "not the ones the binding was issued for"),
        ({"control_binding.probe_id": "2" * 64},
         "disagrees with the recorded probe id"),
    ):
        record = record_copy(tmp_path, name, **edit)
        with pytest.raises(ControlImportError, match=expect):
            import_permanent_control(name, record_path=record,
                                     checkpoint_dir=staged(name),
                                     run_evidence_dir=evidence_dir(tmp_path, name),
                                     repo_root=REPO, strict=True)


@have_controls
def test_two_copies_of_the_fingerprint_must_agree(tmp_path):
    """The record states it twice; a record that contradicts itself is not one."""
    name = CONTROLS[0]
    record = record_copy(tmp_path, name,
                         **{"control_binding.observed_protocol_fingerprint": "3" * 64})
    with pytest.raises(ControlImportError, match="disagree"):
        import_permanent_control(name, record_path=record,
                                 checkpoint_dir=staged(name),
                                 run_evidence_dir=evidence_dir(tmp_path, name),
                                 repo_root=REPO, strict=True)


# --- 3. missing material evidence -------------------------------------------


@have_controls
def test_missing_material_evidence_is_refused_and_named(tmp_path):
    name = CONTROLS[0]
    for dotted, field in (("weights_sha256", "weights_sha256"),
                          ("probe_id", "probe_id"),
                          ("seed", "seed"),
                          ("initialization_artifact_digest",
                           "initialization_artifact_digest"),
                          ("control_binding", "control_binding")):
        record = record_copy(tmp_path, name, **{dotted: _DELETE})
        with pytest.raises(ControlImportError, match=field):
            import_permanent_control(name, record_path=record,
                                     checkpoint_dir=staged(name),
                                     run_evidence_dir=evidence_dir(tmp_path, name),
                                     repo_root=REPO, strict=True)


@have_controls
def test_no_run_evidence_is_refused_in_strict_mode(tmp_path):
    """A sidecar summary is not evidence that a protocol was executed."""
    name = CONTROLS[0]
    with pytest.raises(ControlImportError, match="could not be re-derived"):
        import_permanent_control(
            name, record_path=RECORDS / f"{name}_probe_identity.json",
            checkpoint_dir=staged(name), run_evidence_dir=None,
            repo_root=REPO, strict=True)


@have_controls
def test_run_evidence_that_reconstructs_a_different_protocol_is_refused(tmp_path):
    name = CONTROLS[0]
    evidence = evidence_dir(tmp_path, name)
    manifest = json.loads((evidence / "run_manifest.json").read_text())
    manifest["execution"]["optimizer_defaults"]["lr"] = 9e-9
    manifest["config"]["optim"]["lr"] = 9e-9
    (evidence / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ControlImportError, match="reconstructs protocol"):
        import_permanent_control(
            name, record_path=RECORDS / f"{name}_probe_identity.json",
            checkpoint_dir=staged(name), run_evidence_dir=evidence,
            repo_root=REPO, strict=True)


@have_controls
def test_a_control_that_never_verified_cannot_be_imported(tmp_path):
    name = CONTROLS[0]
    record = record_copy(tmp_path, name, protocol_verified=False)
    with pytest.raises(ControlImportError, match="protocol_verified is false"):
        import_permanent_control(name, record_path=record,
                                 checkpoint_dir=staged(name),
                                 run_evidence_dir=evidence_dir(tmp_path, name),
                                 repo_root=REPO, strict=True)


# --- 4/5. the plan's own gating ---------------------------------------------


def test_the_continuation_plan_orders_the_gates_and_blocks_on_them():
    stages = {s.stage: s for s in CONTINUATION_PLAN_V1.stages}
    assert [stages[i].name for i in range(4)] == [
        "import the permanent controls",
        "current evaluation attestation",
        "v2 tool and RAG generation smoke",
        "characterize the imported controls"]
    # Import, attestation and smoke all block; characterization does not, so a
    # scoring failure cannot retroactively invalidate what came before it.
    assert all(stages[i].blocking for i in (0, 1, 2))
    assert stages[3].blocking is False

    # An attestation failure stops before the smoke and before characterization.
    with pytest.raises(RecoveryAdmissionError, match="did not pass"):
        CONTINUATION_PLAN_V1.advance_to(
            2, {0: {"passed": True}, 1: {"passed": False, "reason": "drift"}})
    # A smoke failure stops before characterization.
    with pytest.raises(RecoveryAdmissionError, match="did not pass"):
        CONTINUATION_PLAN_V1.advance_to(
            3, {0: {"passed": True}, 1: {"passed": True},
                2: {"passed": False, "reason": "tool prompt did not render"}})
    # And a missing predecessor is refused outright, not assumed.
    with pytest.raises(RecoveryAdmissionError, match="no recorded result"):
        CONTINUATION_PLAN_V1.advance_to(3, {0: {"passed": True}, 1: {"passed": True}})


def test_the_smoke_stage_names_the_tool_set_as_its_reason_to_exist():
    smoke = next(s for s in CONTINUATION_PLAN_V1.stages if s.stage == 2)
    joined = " ".join(smoke.stop_conditions) + smoke.purpose + " ".join(smoke.produces)
    assert "tool" in joined and "rag" in joined
    assert "renders no tool prompt" in joined, (
        "the stage must say that a smoke which never renders a tool prompt is "
        "the failure it exists to prevent")


def test_the_continuation_does_not_train_and_cannot_reach_phase_a():
    scope = CONTINUATION_SCOPE.as_dict()
    assert scope["trains_anything"] is False
    assert scope["retrains_controls"] is False
    assert scope["reaches_phase_a"] is False
    assert scope["battery"] == "recovery_search_v2"
    assert sorted(scope["controls"]) == sorted(CONTROLS)
    manifest = continuation_manifest()
    assert manifest["manifest_sha256"]
    assert manifest["import_required_fields"] == list(IMPORT_REQUIRED_FIELDS)
    # It is a different plan from the preflight, not a mutation of it.
    from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1
    assert CONTINUATION_PLAN_V1.plan_hash != PREFLIGHT_PLAN_V1.plan_hash
    assert CONTINUATION_PLAN_V1.plan_id != PREFLIGHT_PLAN_V1.plan_id


def test_advance_to_was_not_weakened():
    """The preflight gate must still refuse what it refused before."""
    from aadistill.autoinit.recovery import PREFLIGHT_PLAN_V1

    with pytest.raises(RecoveryAdmissionError, match="no recorded result"):
        PREFLIGHT_PLAN_V1.advance_to(3, {0: {"passed": True}, 1: {"passed": True}})
    with pytest.raises(RecoveryAdmissionError, match="did not pass"):
        PREFLIGHT_PLAN_V1.advance_to(
            2, {0: {"passed": True}, 1: {"passed": False, "reason": "peak memory"}})


# --- full lifecycle ---------------------------------------------------------


def load_continuation_driver(tmp_path: Path):
    """Import the continuation driver with its pod paths redirected."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "continuation_driver", REPO / "scripts/pod/autoinit_continuation_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continuation_driver"] = mod
    spec.loader.exec_module(mod)
    mod.WS = tmp_path
    mod.STATUS = tmp_path / "continuation.status"
    mod.AUDIT = tmp_path / "audit"
    mod.AUDIT.mkdir(parents=True, exist_ok=True)
    return mod


class ContArgs:
    stage = "all"
    image_digest = "sha256:rehearsal"
    rate = 0.99
    spent_usd = 0.15
    soft_stop_usd = 1.50
    authorized_usd = 1.75
    characterization_minutes = 18.0


def build_continuation(tmp_path, *, import_ok=True, attest_ok=True,
                       smoke_ok=True, characterize_ok=True):
    """The real driver with every paid call replaced by a scripted outcome."""
    mod = load_continuation_driver(tmp_path)
    d = mod.ContinuationDriver.__new__(mod.ContinuationDriver)
    d.a = ContArgs()
    d.t0 = __import__("time").time()
    d.results, d.imported, d.ev = {}, {}, {"stages": {}}
    d.evaluation_protocol = None
    d.mod = mod

    class Auth:
        hard_cap_usd = 1.75
        def require_stage(self, s): pass
        def require_within_cap(self, usd, what=""): pass
        def as_dict(self): return {"authorization_id": "rehearsal"}
    d.auth = Auth()

    def stage0():
        d.enter(0)
        if not import_ok:
            return d.record(0, False, "weights sha256 does not match the record")
        d.imported = {n: object() for n, _ in mod.CONTROLS}
        return d.record(0, True, **{"import": {"controls": list(d.imported)}})

    def stage1():
        d.enter(1)
        if not attest_ok:
            return d.record(1, False, "the battery does not verify")
        class E:
            evaluation_protocol_hash = "EVAL"
        d.evaluation_protocol = E()
        return d.record(1, True, evaluation_protocol_hash="EVAL")

    def stage2():
        d.enter(2)
        if not smoke_ok:
            return d.record(2, False, "smoke: the tool prompt did not render")
        return d.record(2, True, smoke={"sets": ["tool", "rag"],
                                        "covers_tool_set": True})

    def stage3():
        d.enter(3)
        if not characterize_ok:
            return d.record(3, False, "preflight_ctl_r0860k_sa scoring rc=1")
        return d.record(3, True, thresholds={"battery": "recovery_search_v2"})

    d.stage0, d.stage1, d.stage2, d.stage3 = stage0, stage1, stage2, stage3
    for name in ("enter", "record", "usd", "afford", "save", "run", "finish"):
        setattr(d, name, getattr(mod.ContinuationDriver, name).__get__(d))
    return d, mod


def markers_of(mod) -> list[str]:
    if not mod.STATUS.is_file():
        return []
    return [ln.split("MARKER:")[1] for ln in mod.STATUS.read_text().splitlines()
            if "MARKER:" in ln]


def test_the_success_lifecycle_completes_and_trains_nothing(tmp_path):
    d, mod = build_continuation(tmp_path)
    assert d.run() == 0
    assert sorted(d.results) == [0, 1, 2, 3]
    assert all(r["passed"] for r in d.results.values())
    assert "ALL_DONE" in markers_of(mod)
    ev = json.loads((mod.AUDIT / "continuation_evidence.json").read_text())
    assert ev["continuation_successful"] is True
    assert ev["outcome"] == "SUCCESS"
    assert ev["trains_anything"] is False
    assert ev["phase_a_started"] is False
    assert ev["phase_a_reachable_from_this_driver"] is False


def test_characterization_failure_still_collects_and_tears_down_but_FAILS(tmp_path):
    """The semantics the maintainer locked: cleanup is not success.

        import PASS -> attestation PASS -> smoke PASS -> characterization FAIL
        -> evidence collected -> teardown -> outcome FAILED
    """
    d, mod = build_continuation(tmp_path, characterize_ok=False)
    rc = d.run()

    # Everything before characterization passed, and characterization did not.
    assert [d.results[i]["passed"] for i in (0, 1, 2)] == [True, True, True]
    assert d.results[3]["passed"] is False
    assert rc == 23 and rc != 0

    # The evidence exists — collection is not blocked by the failure.
    evidence = mod.AUDIT / "continuation_evidence.json"
    assert evidence.is_file(), "a failed characterization suppressed the evidence"
    ev = json.loads(evidence.read_text())
    assert ev["stages"]["3"]["passed"] is False
    assert ev["stages"]["3"]["reason"]

    # And the outcome is a failure, whatever cleanup did afterwards.
    assert ev["continuation_successful"] is False
    assert ev["outcome"] in ("FAILED", "INCOMPLETE")
    assert ev["failed_stages"] == [3]
    assert "cleanup" in ev["cleanup_is_not_success"]

    # The marker the launcher reads is NOT the success marker.
    marks = markers_of(mod)
    assert "CONTINUATION_INCOMPLETE" in marks
    assert "ALL_DONE" not in marks, (
        "a failed characterization emitted the success marker; the launcher "
        "gates teardown and the session verdict on it")
    # Non-blocking means cleanup proceeds, not that the stage passed.
    assert "STAGE_NONBLOCKING_FAIL" in marks
    assert ev["phase_a_started"] is False


@pytest.mark.parametrize("kwargs,stage", [
    (dict(import_ok=False), 0),
    (dict(attest_ok=False), 1),
    (dict(smoke_ok=False), 2),
])
def test_a_blocking_failure_stops_before_characterization(tmp_path, kwargs, stage):
    d, mod = build_continuation(tmp_path, **kwargs)
    assert d.run() == 20 + stage
    assert 3 not in d.results, "the controls were characterized after a blocking failure"
    marks = markers_of(mod)
    assert "CONTINUATION_FAILED" in marks and "ALL_DONE" not in marks
    ev = json.loads((mod.AUDIT / "continuation_evidence.json").read_text())
    assert ev["continuation_successful"] is False
    assert ev["stages"][str(stage)]["reason"]


def test_the_launcher_reports_failure_when_the_driver_does():
    """`collect PASS + teardown PASS` must not become a successful session."""
    launch = (REPO / "scripts/pod/autoinit_continuation_launch.py").read_text()
    assert 'session.ev["continuation_successful"] = bool(ok)' in launch
    assert "cleanup_is_not_success" in launch
    assert "return 0 if ok else 11" in launch
    # `ok` comes from the inherited collect_and_teardown, which returns `done`.
    preflight = (REPO / "scripts/pod/autoinit_preflight_launch.py").read_text()
    tail = preflight[preflight.index("def collect_and_teardown"):]
    assert "return done" in tail
    assert 'done = terminal == "ALL_DONE"' in tail or "done" in tail


def test_transport_is_separate_from_identity():
    """The driver imports a local artifact; how it arrived is not its business."""
    driver = (REPO / "scripts/pod/autoinit_continuation_driver.py").read_text()
    for forbidden in ("snapshot_download", "hf_hub_download", "scp", "relay"):
        assert forbidden not in driver.split('"""')[2], (
            f"the driver mentions {forbidden}: transport has leaked into the "
            "component that decides what a control IS")
    assert "CONTROL_ROOT" in driver
    launch = (REPO / "scripts/pod/autoinit_continuation_launch.py").read_text()
    assert "def materialize_controls" in launch
    assert '"relay"' in launch and '"scp"' in launch
    assert "--transport" in launch


def test_the_continuation_driver_cannot_train_or_reach_phase_a():
    driver = (REPO / "scripts/pod/autoinit_continuation_driver.py").read_text()
    for forbidden in ("train_stage3.py", "Trainer(", "BeamSearch", "admit_leaves",
                      "probe_configs", "SuccessiveHalvingPlan", "run_phase_a"):
        assert forbidden not in driver, f"the continuation driver can reach {forbidden}"
    assert 'choices=("all",)' in driver, "a stage outside the plan is expressible"


# --- the launcher paths that had never been executed ------------------------
#
# Everything below was found by *running* the launcher's own methods rather than
# reading them. Each test corresponds to a defect that a source-level review had
# already passed over, and that would have surfaced only on a paid pod.


def load_continuation_launcher():
    """Import the real launcher module (it subclasses the preflight one)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "continuation_launch", REPO / "scripts/pod/autoinit_continuation_launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continuation_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


class LaunchArgs:
    """The launcher's real defaults, as `main()` would parse them."""

    scr = "/tmp/unused"
    session_commit = "HEAD"
    transport = "relay"
    relay_repo = "AlphaAvatar/aadistill-artifacts"
    max_price = 0.99
    characterization_minutes = None      # filled from the launcher's own default
    setup_minutes = 8.0
    transfer_minutes = 4.0
    ckpt_store = "/home/ecs-user/aad-artifacts/autoinit"
    ckpt_fetch_limit_min = 25


def bare_launcher(mod, **overrides):
    """A launcher instance with no provider, no ssh and no pod — real methods."""
    obj = mod.Continuation.__new__(mod.Continuation)
    args = LaunchArgs()
    # Take the budget from the launcher itself, so a change there cannot pass
    # these tests by leaving a copied constant behind.
    args.characterization_minutes = mod.CHARACTERIZATION_MINUTES
    for k, v in overrides.items():
        setattr(args, k, v)
    obj.a = args
    obj.ev = {}
    obj.plan = None
    obj.said: list[str] = []
    obj.say = obj.said.append
    from aadistill.autoinit.continuation import CONTINUATION_AUTHORIZATION
    obj.auth = CONTINUATION_AUTHORIZATION
    return obj


def test_make_plan_actually_prices_and_stays_inside_the_authorization():
    """It raised `BudgetError` before a pod could exist and nothing caught it.

    `make_plan` passed a 3.15 s/step `StepTime` with no `below_floor_reason`,
    which `plan_session` refuses. The continuation could never have launched:
    `run()` calls this first, and the exception is not a `BudgetError` the
    launcher handles. Nothing executed it, because every rehearsal stopped at
    the driver.
    """
    mod = load_continuation_launcher()
    obj = bare_launcher(mod)
    obj.check_gpu_offered = lambda: True          # the one real provider query
    assert obj.make_plan() is True
    plan = obj.plan
    # The authorized bound is the HARD threshold; that is what the launcher
    # enforces before a pod can exist, and what the maintainer set at $1.75.
    assert plan.hard_terminate_usd <= 1.75 + 1e-9, plan.as_dict()
    # The expected figure is the harness's own conservative ceiling — $1.34 at
    # the 24 min/control budget — not the $0.88 sketch in the repricing note,
    # which left out the transfer phase and used the optimistic column for four
    # others. Asserted so the two numbers cannot drift apart again unnoticed.
    assert 1.30 <= plan.expected_usd <= 1.40, plan.as_dict()
    # And the leash is long enough to characterize BOTH controls even if the
    # unmeasured per-control cost is a third over the historical guess.
    assert obj.a.characterization_minutes >= 24.0
    assert plan.expected_usd <= CONTINUATION_AUTHORIZATION.expected_usd + 1e-9
    assert obj.ev["budget_plan"]["expected_usd"] == pytest.approx(
        plan.expected_usd, abs=1e-4)
    # The floor waiver is recorded in the plan, not hidden in a comment.
    assert any("below-floor" in n for n in plan.as_dict()["notes"])


def test_the_precheck_fails_at_zero_when_a_control_is_not_staged():
    """The staged controls are this session's whole input, so $0 is where an
    unstaged one must be discovered — not after a paid setup."""
    mod = load_continuation_launcher()
    staged = {f"permanent_controls/{c}/model/model.safetensors" for c in mod.CONTROLS}
    init = "stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors"
    pack = "stage3_recovery_corpus_v2/ladder_uniform/blocks.npz"

    class FakeApi:
        def __init__(self, files):
            self.files = files

        def list_repo_files(self, repo, repo_type=None):
            return sorted(self.files)

    import huggingface_hub

    def with_files(files, **overrides):
        obj = bare_launcher(mod, **overrides)
        # The commit gate runs first and has its own test; this one is about
        # the inputs.
        obj.verify_session_commit = lambda: True
        original = huggingface_hub.HfApi
        huggingface_hub.HfApi = lambda *a, **k: FakeApi(files)
        try:
            return obj.relay_precheck(), obj
        finally:
            huggingface_hub.HfApi = original

    ok, obj = with_files({init, pack, *staged})
    assert ok is True
    assert all(f"permanent_controls/{c}/model/model.safetensors"
               in obj.ev["precheck"]["relay_needed"] for c in mod.CONTROLS)
    # The pack stays required: Stage 0 recomputes each control's pack hash from
    # it rather than trusting what the training run recorded.
    assert pack in obj.ev["precheck"]["relay_needed"]

    # One control staged, the other not: the old precheck said OK.
    ok, obj = with_files({init, pack, sorted(staged)[0]})
    assert ok is False
    assert obj.ev["precheck"]["relay_missing"]
    assert any("ABORT at $0" in s for s in obj.said)

    # Over scp the relay need not hold the controls at all.
    ok, obj = with_files({init, pack}, transport="scp")
    assert obj.ev["precheck"]["transport"] == "scp"
    assert not [m for m in obj.ev["precheck"]["relay_missing"]
                if "permanent_controls" in m]


def continuation_output_tree(root: Path) -> None:
    """Exactly what the continuation driver writes on a successful session."""
    audit = root / "audit/autoinit_continuation"
    (audit / "session").mkdir(parents=True)
    for name in ("continuation_evidence.json", "imported_controls.json",
                 "attested_evaluation_protocol.json",
                 "materialized_thresholds.json", "engine_probe.json",
                 "generation_smoke.json"):
        (audit / name).write_text('{"rehearsal": true}\n')
    (audit / "engine_probe.log").write_text("rc=0\n")
    for ctl in ("preflight_ctl_r0860k_sa", "preflight_ctl_r0860k_sb"):
        (audit / f"{ctl}_recovery_search.json").write_text('{"n": 190}\n')
        (audit / f"{ctl}_per_sample.jsonl").write_text('{"id": "x"}\n')
        gen = root / "eval/continuation" / ctl
        gen.mkdir(parents=True)
        (gen / f"{ctl}.generations.jsonl").write_text('{"text": "y"}\n')
        (gen / f"{ctl}.json").write_text('{"summary": true}\n')


def test_the_success_spec_accepts_what_the_driver_writes(tmp_path):
    """A successful continuation must not fail its own artifact manifest.

    It would have: the inherited spec required `preflight_evidence.json`,
    per-control `train_log.jsonl`, `run_manifest.json` and checkpoints — none of
    which a session that trains nothing produces. `required_files_present` is a
    teardown-gate check, so the failure mode was: characterize both controls,
    fail the manifest, BLOCK teardown, and keep billing to the hard threshold.
    """
    sys.path.insert(0, str(REPO / "scripts/pod"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "collect_artifacts", REPO / "scripts/pod/collect_artifacts.py")
    collect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect)
    from aadistill.infrastructure.artifact_gate import build_manifest

    root = tmp_path / "artifacts"
    root.mkdir()
    continuation_output_tree(root)

    good = build_manifest(root, collect.load_specs(
        str(REPO / "configs/autoinit/continuation_artifacts.json")))
    assert good.ok, good.missing

    # The same tree under the preflight's success spec: this is the bug.
    bad = build_manifest(root, collect.load_specs(
        str(REPO / "configs/autoinit/preflight_artifacts.json")))
    assert not bad.ok
    assert any(m["artifact_class"] == "preflight_evidence" for m in bad.missing)
    assert any("train_log" in m["pattern"] for m in bad.missing)

    # And the reduced spec still accepts a session that produced almost nothing.
    sparse = tmp_path / "sparse"
    (sparse / "audit/autoinit_continuation").mkdir(parents=True)
    (sparse / "audit/autoinit_continuation/continuation_evidence.json").write_text("{}\n")
    reduced = build_manifest(sparse, collect.load_specs(
        str(REPO / "configs/autoinit/continuation_artifacts_failed.json")))
    assert reduced.ok, reduced.missing


def test_the_launcher_recognises_the_markers_its_own_driver_emits():
    """The poll loop watched for PREFLIGHT_* while the driver emits CONTINUATION_*."""
    mod = load_continuation_launcher()
    driver = (REPO / "scripts/pod/autoinit_continuation_driver.py").read_text()
    import re

    emitted = set(re.findall(r'mark\("([A-Z_]+)"\)', driver))
    terminal = {m for m in emitted if m in {"ALL_DONE", "CONTINUATION_FAILED",
                                            "CONTINUATION_INCOMPLETE"}}
    assert terminal == {"ALL_DONE", "CONTINUATION_FAILED", "CONTINUATION_INCOMPLETE"}
    recognised = {"ALL_DONE", *mod.Continuation.failure_markers}
    assert terminal <= recognised, terminal - recognised
    # INCOMPLETE means the blocking stages passed, so products still exist.
    assert mod.Continuation.incomplete_markers == ("CONTINUATION_INCOMPLETE",)


def test_the_continuation_fetches_no_checkpoints_and_waits_on_no_train_log():
    """It creates neither, and demanding either would block its teardown."""
    mod = load_continuation_launcher()
    obj = bare_launcher(mod)
    assert obj.fetch_products("host", None, stage2_passed=True) == []
    assert obj.event_streams() == ()
    assert obj.audit_dirname == "autoinit_continuation"
    assert obj.spec_success.endswith("continuation_artifacts.json")

    # The preflight's own defaults are untouched by the hooks it gained.
    pre = mod._preflight.Preflight
    assert pre.audit_dirname == "autoinit_preflight"
    assert pre.failure_markers == ("PREFLIGHT_FAILED", "PREFLIGHT_INCOMPLETE")
    assert pre.spec_success == "configs/autoinit/preflight_artifacts.json"
    assert pre.report_names[0] == "preflight_evidence.json"


def test_the_authorization_binds_the_code_that_actually_runs():
    """It digested the preflight's files, so an edited continuation driver —
    the executable that spends the money — passed the gate unnoticed."""
    from aadistill.autoinit.authorization import (
        HARNESS_SOURCE_FILES_V1, harness_source_digest,
    )
    from aadistill.autoinit.continuation import (
        CONTINUATION_AUTHORIZATION, CONTINUATION_HARNESS_SOURCE_FILES_V1,
    )

    files = set(CONTINUATION_HARNESS_SOURCE_FILES_V1)
    for executable in ("scripts/pod/autoinit_continuation_launch.py",
                       "scripts/pod/autoinit_continuation_driver.py",
                       "src/aadistill/autoinit/continuation.py"):
        assert executable in files
        assert executable not in HARNESS_SOURCE_FILES_V1   # the gap that existed
    assert CONTINUATION_AUTHORIZATION.harness_source_files == \
        CONTINUATION_HARNESS_SOURCE_FILES_V1
    assert CONTINUATION_AUTHORIZATION.as_dict()["harness_source_files"] == \
        list(CONTINUATION_HARNESS_SOURCE_FILES_V1)

    # The digest is over the continuation set, and it differs from the
    # preflight's — the two authorizations cannot be confused for each other.
    observed = harness_source_digest(REPO, files=CONTINUATION_HARNESS_SOURCE_FILES_V1)
    assert observed["digest"] != harness_source_digest(REPO)["digest"]


def test_the_continuation_authorization_is_narrow_and_cannot_train():
    from aadistill.autoinit.authorization import AuthorizationError
    from aadistill.autoinit.continuation import CONTINUATION_AUTHORIZATION as auth

    assert (auth.expected_usd, auth.hard_cap_usd) == (1.34, 1.75)
    assert auth.plan_hash == CONTINUATION_PLAN_V1.plan_hash
    assert auth.allows_phase_a is False and auth.automatic_phase_a_start is False
    with pytest.raises(AuthorizationError):
        auth.require_within_cap(1.76, what="session")
    with pytest.raises(AuthorizationError):
        auth.require_stage(4)
    with pytest.raises(AuthorizationError, match="separately unauthorized"):
        auth.refuse_phase_a()
    assert "does not permit training" in auth.scope_note


def test_the_session_commit_is_verified_against_the_authorization():
    """`authorized_session_commit` was recorded and never enforced.

    The pod clones a bundle and checks out `--session-commit`; the harness gate
    reads the dev box's working tree. Nothing connected the two, so the pod
    could have run a commit the authorization never covered.
    """
    import subprocess

    mod = load_continuation_launcher()
    auth_path = REPO / "logs/autoinit_continuation_authorization.json"
    if not auth_path.is_file():
        pytest.skip("no continuation authorization has been issued yet")

    from aadistill.autoinit.authorization import SpendAuthorization
    auth = SpendAuthorization.load(auth_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO).stdout.strip()

    obj = bare_launcher(mod, session_commit=head)
    obj.auth = auth
    verified = obj.verify_session_commit()
    check = obj.ev["session_commit_check"]
    if verified:
        assert check["harness_matches"] is True
        assert check["commit_carries_this_authorization"] is True
    else:
        # Before the authorization is committed, HEAD cannot carry it — but the
        # harness at HEAD must still be the authorized one, or the launch would
        # be running unauthorized code.
        assert check["harness_matches"] is True, check

    # An old commit's harness is not this one, and is refused by digest.
    old = subprocess.run(["git", "rev-parse", "HEAD~3"], capture_output=True,
                         text=True, cwd=REPO).stdout.strip()
    stale = bare_launcher(mod, session_commit=old)
    stale.auth = auth
    assert stale.verify_session_commit() is False
    assert any("ABORT at $0" in s for s in stale.said)


def test_the_relay_transport_does_not_depend_on_an_unexported_env_var(tmp_path):
    """`HF_TOKEN` is exported inside setup.sh and dies with that shell.

    `materialize_controls` runs over a FRESH ssh session, which inherits none of
    setup's environment, so `os.environ['HF_TOKEN']` raises KeyError — after the
    session has already paid for setup, and for the only transport this
    continuation actually uses. The token has to come from the staged file.
    """
    mod = load_continuation_launcher()
    obj = bare_launcher(mod)
    obj.save = lambda: None

    class FakeResult:
        returncode = 0
        stdout = "PRESENT=1"

    class FakeTarget:
        def __init__(self):
            self.commands: list[str] = []

        def run(self, cmd, timeout=None):
            self.commands.append(cmd)
            return FakeResult()

    target = FakeTarget()
    # scp of the two record files is a real subprocess; point it at /bin/true.
    import subprocess as sp
    original = sp.run
    sp.run = lambda *a, **k: FakeResult()
    try:
        assert obj.materialize_controls(target, "1.2.3.4", ["scp"]) is True
    finally:
        sp.run = original

    relay = [c for c in target.commands if "snapshot_download" in c]
    assert len(relay) == len(mod.CONTROLS)
    for cmd in relay:
        assert 'HF_TOKEN="$(cat /workspace/hf/token)"' in cmd, cmd
        # The token must reach the child through the environment it sets here,
        # not through one it hopes to inherit.
        assert cmd.index("HF_TOKEN=") < cmd.index("python3")
    assert obj.ev["transport_detail"]["route"] == "relay"
    assert all(c["materialized"] for c in
               obj.ev["transport_detail"]["controls"].values())
