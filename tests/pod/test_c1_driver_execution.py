"""The real `C1Driver.run` control flow, end to end, at `$0`.

Not a canary. This drives the **actual** driver — its own `run`, its own stage
methods, its own authorization and plan binding — with only the hardware-bound
operations replaced by deterministic local fakes that emit the real filesystem
layout and the real file contracts. Everything between them is production code.

What it is here to prevent, in order of what has already cost this project money:

* the driver silently inheriting Phase-A operational machinery again;
* an evaluation starting before all six trainings finish, which would let the
  first arm meet the confirmation battery before the last arm was trained;
* a probe reaching `recovery_search_v2` instead of `c1_confirmation_v1`;
* a replay mismatch emitting its marker without its evidence;
* the CUDA handoff gate being skipped before the trainer is spawned;
* a decision built from five results, or from unpaired prompts.

Every load-bearing failure is mutation-checked: the test asserts that breaking
the rule makes the test fail, because a suite that only ever passes is unverified.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "pod"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))

import autoinit_c1_driver as D  # noqa: E402

from aadistill.autoinit.c1_isolation import derive_recovery_seeds  # noqa: E402
from aadistill.autoinit.c1_probe_results import (  # noqa: E402
    C1ResultsError, decision_inputs,
)
from aadistill.autoinit.c1_scoring import C1_BATTERY_SETS  # noqa: E402

SEEDS = derive_recovery_seeds()
BATTERY = REPO / "artifacts/stage3/c1_confirmation_v1"


# --- the driver is C1's own -------------------------------------------------

def test_c1driver_is_not_a_phase_a_subclass():
    """Mutation target: restoring the inheritance must fail this."""
    names = [c.__name__ for c in D.C1Driver.__mro__]
    assert "PhaseADriver" not in names, names
    assert D.C1Driver.__mro__[1] is object


DRIVER_SRC = REPO / "scripts/pod/autoinit_c1_driver.py"


def _executable_text(path: Path) -> str:
    """Everything the module can actually DO: identifiers plus live strings.

    Prose is excluded deliberately. The first version of these checks grepped the
    raw file and failed on this driver's own docstring, which names
    `score_recovery_search.py` and `recovery_search_v2` precisely to explain why
    they are not used. A check that cannot tell an explanation from a call site
    would force the explanation to be deleted.
    """
    import ast

    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                docstrings.add(d)
    parts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.alias):
            parts.append(node.name)
            parts.append(node.asname or "")
        elif isinstance(node, ast.ImportFrom):
            parts.append(node.module or "")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                parts.append(node.value)
    return "\n".join(parts)


def test_the_driver_owns_run_and_imports_no_phase_a_operational_module():
    import ast

    assert "run" in vars(D.C1Driver)
    tree = ast.parse(DRIVER_SRC.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for forbidden in ("autoinit_phase_a_driver", "autoinit_phase_a_launch",
                      "phase_a_search", "aadistill.autoinit.phase_a"):
        assert forbidden not in imported, forbidden
    assert "PHASE_A_PLAN_V1" not in _executable_text(DRIVER_SRC)


def test_stage_h_can_only_reach_the_c1_scorer_and_the_c1_battery():
    code = _executable_text(DRIVER_SRC)
    assert "score_recovery_search" not in code
    assert "recovery_search_v2" not in code
    assert "c1_confirmation_v1" in code
    assert "score_c1_confirmation.py" in code


def test_all_c1_paths_are_c1_owned():
    assert D.AUDIT == REPO / "artifacts/audit/autoinit_c1"
    assert D.TRAIN == REPO / "artifacts/stage3/c1"
    assert D.EVAL == REPO / "artifacts/eval/c1"
    code = _executable_text(DRIVER_SRC)
    for bad in ("audit/autoinit_phase_a", "stage3/phase_a", "eval/phase_a"):
        assert bad not in code, bad


# --- the roots the driver loads, and where they land ------------------------
#
# `_fake_hardware` replaces `stage_de` WHOLE, so the harness below never executes
# the real one — and the real one is where attempt 8's second defect lived. Stage
# D declared `cuda` through `build_arm_specs(workdir_device="cuda")` and loaded
# its root with a bare `AutoModelForCausalLM.from_pretrained(...).eval()`. Every
# operator reads `model_device(model)`, the fact rather than the intent, so the
# whole parent replay would have executed on the host CPU inside a paid GPU hour.
#
# These run the REAL `stage_de` and `stage_f`, faking only `materialize_fixed_path`
# and the adapter's loader, so the root_loader closure itself is executed.

@pytest.fixture
def treatment_registered():
    from aadistill.autoinit.operators import attention_activation

    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


#: The two frozen pins a SUCCESSFUL incumbent replay carries. Both, not just the
#: parent: stage E now requires its own step to have matched before it passes,
#: so a fake that pins only step 2 no longer models a successful replay.
_PINS = {2: D.CS.EXPECTED_PARENT_DIGEST, 3: D.CS.EXPECTED_INCUMBENT_DIGEST}


class _FakeStep:
    def __init__(self, i):
        self.index = i
        self.digest_expected = _PINS.get(i)
        self.digest_matches = True if i in _PINS else None
        self.identity = types.SimpleNamespace(
            artifact_digest=_PINS.get(i, "d" * 64))
        self.checkpoint_path = f"/tmp/ckpt{i}"
        self.result_spec_hash = "s" * 64
        self.selection = {"step": i}

    def as_dict(self):
        return {"index": self.index,
                "artifact_digest": self.identity.artifact_digest,
                "digest_expected": self.digest_expected,
                "digest_matches": self.digest_matches}


def _write_valid_replay_record(spec, results, path, **kw):
    """A stub that writes what the REAL writer would, so the readback is real.

    Stubbing this to `/dev/null` would make `require_replay_record` fail on a
    missing file — which is the check working, and useless as a fixture.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema": "aadistill.autoinit.fixed_path_replay/v1",
        "path_hash": spec.spec_hash,
        "steps": [r.as_dict() for r in results],
        "all_pinned_digests_matched": True,
        "n_pinned": sum(1 for r in results if r.digest_expected is not None),
    }))
    return p


def _capture_root(monkeypatch, driver):
    """Run the real stage, capture what the root loader was asked for.

    Both materializers are faked, because stage D uses the whole-path one and
    stage F uses the verified-suffix one — the point of these cases is the
    root_loader closure and the device it names, which is real either way.
    """
    from aadistill.autoinit.adapters.qwen3 import QWEN3_ADAPTER

    seen: dict = {}

    def fake_load(path, dtype=None, device="cpu"):
        seen.update(path=str(path), dtype=dtype, device=device)
        return types.SimpleNamespace(config=types.SimpleNamespace())

    monkeypatch.setattr(QWEN3_ADAPTER, "load", fake_load)

    def fake_materialize(spec, *, adapter, root_loader, workdir, repo_root,
                         on_step=None, **kw):
        seen["spec_device"] = spec.device
        seen["deadline"] = kw.get("deadline")
        root_loader()                       # the closure under test
        steps = [_FakeStep(i) for i in range(4)]
        for s in steps:                     # the real materializer calls this
            if on_step is not None:
                on_step(s)
        return steps

    def fake_suffix(spec, *, adapter, root_loader, workdir, verified,
                    repo_root=".", on_step=None, **kw):
        seen["spec_device"] = spec.device
        seen["deadline"] = kw.get("deadline")
        seen["start_index"] = verified.start_index
        seen["expected_path_hash"] = verified.expected_path_hash
        seen["prefix_reference_steps"] = tuple(verified.prefix_reference_steps)
        seen["expected_suffix_steps"] = tuple(verified.expected_suffix_steps)
        seen["expected_parent_digest"] = verified.expected_parent_artifact_digest
        root_loader()
        return [_FakeStep(verified.start_index)], {"premise": "faked"}

    monkeypatch.setattr(D, "materialize_fixed_path", fake_materialize)
    monkeypatch.setattr(D, "materialize_fixed_path_suffix", fake_suffix)
    monkeypatch.setattr(D, "write_replay_record", _write_valid_replay_record)
    return seen


def test_stage_d_loads_its_root_on_the_arm_declared_device(
        harness, monkeypatch, treatment_registered):
    driver = D.C1Driver(_args())
    driver.teacher_path = str(harness.tmp / "teacher")
    #: stage_de now completes BOTH D and E through the real stage-order
    #: contract, so the two stages before them must have happened. Declared,
    #: not faked away — an ordering regression must still be visible here.
    driver.completed = [D.CS.stage(x).stage_id for x in ("B", "C")]
    seen = _capture_root(monkeypatch, driver)

    driver.stage_de()

    assert seen["spec_device"] == "cuda", "the arm no longer declares cuda"
    assert seen["device"] == seen["spec_device"], (
        "stage D asked for a root on "
        f"{seen['device']!r} while its arm declares {seen['spec_device']!r}")
    assert seen["device"] == driver.arms["incumbent"].device
    assert seen["path"] == driver.teacher_path
    assert seen["dtype"] == "bfloat16"


def test_stage_f_loads_its_root_on_its_own_arm_device(
        harness, monkeypatch, treatment_registered):
    driver = D.C1Driver(_args())
    driver.teacher_path = str(harness.tmp / "teacher")
    #: stage_de now completes BOTH D and E through the real stage-order
    #: contract, so the two stages before them must have happened. Declared,
    #: not faked away — an ordering regression must still be visible here.
    driver.completed = [D.CS.stage(x).stage_id for x in ("B", "C")]
    _capture_root(monkeypatch, driver)
    driver.stage_de()
    driver.parent = _FakeStep(2)
    assert driver.completed == [D.CS.stage(x).stage_id
                                for x in ("B", "C", "D", "E")], (
        "stage_de must have completed both replay gates by now")

    seen = _capture_root(monkeypatch, driver)
    driver.stage_f()

    assert seen["device"] == driver.arms["treatment"].device
    assert seen["path"] == driver.parent.checkpoint_path


def test_stage_f_asks_for_the_frozen_suffix_and_nothing_else(
        harness, monkeypatch, treatment_registered):
    """What the driver ASSERTS to the executor, checked at the seam.

    The refusals live in `fixed_path`; this is the other half — that the driver
    hands it the frozen path hash, the frozen parent digest, the incumbent's own
    prefix objects, and a start index derived from the prefix length rather than
    written as `3`.
    """
    driver = D.C1Driver(_args())
    driver.teacher_path = str(harness.tmp / "teacher")
    #: stage_de now completes BOTH D and E through the real stage-order
    #: contract, so the two stages before them must have happened. Declared,
    #: not faked away — an ordering regression must still be visible here.
    driver.completed = [D.CS.stage(x).stage_id for x in ("B", "C")]
    _capture_root(monkeypatch, driver)
    driver.stage_de()
    driver.parent = _FakeStep(2)
    assert driver.completed == [D.CS.stage(x).stage_id
                                for x in ("B", "C", "D", "E")]

    seen = _capture_root(monkeypatch, driver)
    driver.stage_f()

    treatment = driver.arms["treatment"]
    incumbent = driver.arms["incumbent"]
    assert seen["start_index"] == len(D.CS.PREFIX_STEPS) == 3
    assert seen["expected_path_hash"] == treatment.spec_hash
    assert seen["expected_parent_digest"] == D.CS.EXPECTED_PARENT_DIGEST
    assert seen["prefix_reference_steps"] == tuple(incumbent.steps[:3])
    assert seen["expected_suffix_steps"] == (D.CS.TREATMENT_ATTENTION,)
    # the full four-step arm, unchanged — not a synthesized one-step spec
    assert len(treatment.steps) == 4


def test_stage_f_writes_a_treatment_record_that_claims_no_replay(
        harness, monkeypatch, treatment_registered):
    driver = D.C1Driver(_args())
    driver.teacher_path = str(harness.tmp / "teacher")
    #: stage_de now completes BOTH D and E through the real stage-order
    #: contract, so the two stages before them must have happened. Declared,
    #: not faked away — an ordering regression must still be visible here.
    driver.completed = [D.CS.stage(x).stage_id for x in ("B", "C")]
    _capture_root(monkeypatch, driver)
    driver.stage_de()
    driver.parent = _FakeStep(2)
    assert driver.completed == [D.CS.stage(x).stage_id
                                for x in ("B", "C", "D", "E")]
    _capture_root(monkeypatch, driver)

    driver.stage_f()

    rec = json.loads((D.AUDIT / "c1_treatment_record.json").read_text())
    assert rec["schema"] == "aadistill.autoinit.fixed_path_suffix_execution/v1"
    assert rec["is_replay"] is False
    assert rec["output_digest_was_pre_pinned"] is False
    assert rec["executed_step_indices"] == [3]
    assert rec["path_hash"] == driver.arms["treatment"].spec_hash
    assert "NONE" in rec["output_digest_claim"]

    ids = json.loads((D.AUDIT / "c1_arm_identities.json").read_text())
    assert ids["treatment_executed_step_indices"] == [3]
    assert ids["treatment_output_digest_was_pre_pinned"] is False
    assert ids["treatment_record"].endswith("c1_treatment_record.json")


def test_stage_f_refuses_to_report_success_if_more_than_the_suffix_ran(
        harness, monkeypatch, treatment_registered):
    """The driver re-checks the executor's answer instead of assuming it."""
    driver = D.C1Driver(_args())
    driver.teacher_path = str(harness.tmp / "teacher")
    #: stage_de now completes BOTH D and E through the real stage-order
    #: contract, so the two stages before them must have happened. Declared,
    #: not faked away — an ordering regression must still be visible here.
    driver.completed = [D.CS.stage(x).stage_id for x in ("B", "C")]
    _capture_root(monkeypatch, driver)
    driver.stage_de()
    driver.parent = _FakeStep(2)
    assert driver.completed == [D.CS.stage(x).stage_id
                                for x in ("B", "C", "D", "E")]
    _capture_root(monkeypatch, driver)

    monkeypatch.setattr(
        D, "materialize_fixed_path_suffix",
        lambda spec, **kw: ([_FakeStep(2), _FakeStep(3)], {"premise": "faked"}))
    with pytest.raises(D.C1DriverError, match="executed steps"):
        driver.stage_f()


# --- the deadline the fixed path never had ----------------------------------
#
# `OperatorContext.deadline` has existed for a long time and
# `depth.causal_kl_greedy_v1` checks it once per candidate — it was added because
# that operator ran 10.78 h against a 3.0 h budget. The SEARCH passes one. The
# fixed path did not, so on the C1 path the field was always None and the check
# was a no-op.

def test_the_operator_deadline_is_the_sessions_existing_budget(harness):
    """No invented timeout: the same two numbers every other decision uses."""
    driver = D.C1Driver(_args())
    d = driver.operator_deadline("stage D/E replay")

    assert d.remaining_usd() == pytest.approx(
        driver.a.soft_stop_usd - driver.usd(), abs=1e-6)
    assert not d.expired()
    d.check("before the first candidate")          # must not raise
    assert d.checks == 1
    assert d.as_dict()["soft_stop_usd"] == round(driver.a.soft_stop_usd, 4)

    # DERIVED, not snapshotted: move the budget and the deadline moves with it.
    # A timeout of its own would not.
    before = d.remaining_usd()
    driver.a.soft_stop_usd += 5.0
    assert d.remaining_usd() == pytest.approx(before + 5.0, abs=1e-6)


def test_a_fake_clock_past_the_soft_stop_stops_the_operator(harness, monkeypatch):
    """Fake clock, so the assertion is about the rule and not about timing."""
    driver = D.C1Driver(_args())
    d = driver.operator_deadline("stage D/E replay")
    monkeypatch.setattr(driver, "usd", lambda: driver.a.soft_stop_usd + 0.01)

    assert d.expired()
    with pytest.raises(D.C1DriverError, match="soft-stop budget is spent"):
        d.check("candidate 7 of 32")
    assert d.fired_at == "candidate 7 of 32"
    assert d.as_dict()["expired"] is True


def test_an_expired_deadline_is_C1_FAILED_and_never_a_replay_mismatch(
        harness, monkeypatch):
    """The classification, end to end through the REAL `run()`.

    A budget stop compares no digest, so calling it a replay mismatch would put
    a false claim about the frozen path's reproducibility into the record — the
    same error the launcher's canned failure note used to make in prose.
    """
    _fake_hardware(monkeypatch, harness)

    def expired_stage_de(self):
        D.mark("STAGE_START:D")
        self.usd = lambda: self.a.soft_stop_usd + 1.0      # the fake clock
        self.operator_deadline("stage D/E replay").check("candidate 7 of 32")
    monkeypatch.setattr(D.C1Driver, "stage_de", expired_stage_de)

    driver = D.C1Driver(_args())
    code = driver.run()

    assert code == 40
    status = D.STATUS.read_text()
    assert "MARKER:C1_FAILED" in status
    assert "C1_REPLAY_MISMATCH" not in status
    assert not (D.AUDIT / "c1_replay_record.json").exists()
    assert not (D.AUDIT / "c1_treatment_record.json").exists()
    assert driver.ev["training_started"] is False
    assert driver.ev["outcome"] == "C1_FAILED"
    stage = driver.ev["stages"]["D"]
    assert stage["passed"] is False
    assert "soft-stop budget is spent" in stage["reason"]
    # The message DENIES a mismatch rather than merely omitting the word, which
    # is what a reader of the failure tail actually needs to see.
    assert "not a replay mismatch" in stage["reason"]
    assert "mismatch_record" not in stage


def test_both_materializers_are_given_the_deadline(harness, monkeypatch,
                                                   treatment_registered):
    driver = D.C1Driver(_args())
    driver.teacher_path = str(harness.tmp / "teacher")
    #: stage_de now completes BOTH D and E through the real stage-order
    #: contract, so the two stages before them must have happened. Declared,
    #: not faked away — an ordering regression must still be visible here.
    driver.completed = [D.CS.stage(x).stage_id for x in ("B", "C")]

    seen = _capture_root(monkeypatch, driver)
    driver.stage_de()
    assert isinstance(seen["deadline"], D.C1OperatorDeadline)

    driver.parent = _FakeStep(2)
    assert driver.completed == [D.CS.stage(x).stage_id
                                for x in ("B", "C", "D", "E")]
    seen = _capture_root(monkeypatch, driver)
    driver.stage_f()
    assert isinstance(seen["deadline"], D.C1OperatorDeadline)


def test_no_c1_root_is_loaded_outside_the_adapter():
    """The adapter owns the model lifecycle: path, dtype AND placement.

    A raw `from_pretrained` is exactly the shape that lost the transfer, and it
    loses it silently — the model is real, the config is right, and only the
    weights are in the wrong place.
    """
    code = _executable_text(DRIVER_SRC)
    assert "AutoModelForCausalLM" not in code
    assert "from_pretrained" not in code


def test_the_driver_uses_the_real_fixed_path_executor_and_its_device_gate():
    """The refusal only protects the driver if the driver calls that function."""
    from aadistill.autoinit import fixed_path

    assert D.materialize_fixed_path is fixed_path.materialize_fixed_path
    assert "require_root_on_declared_device" in \
        (REPO / "src/aadistill/autoinit/fixed_path.py").read_text()


def test_the_trainer_headroom_is_derived_from_the_committed_measurement():
    """39.79 + 1.35 + 0.51 GiB. Not a written constant."""
    assert D._trainer_bytes() == int((39.79 + 1.35 + 0.51) * 2**30)
    src = (REPO / "scripts/pod/autoinit_c1_driver.py").read_text()
    assert "39.79" not in src.split("def _trainer_bytes")[1].split("return")[1]


# --- the harness ------------------------------------------------------------

class Recorder:
    """The order in which hardware-bound things were asked to happen."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, what: str) -> None:
        self.calls.append(what)

    def index(self, prefix: str) -> list[int]:
        return [i for i, c in enumerate(self.calls) if c.startswith(prefix)]


#: The `behavior_v0` record shape `uncapped_eval` writes. Restated from the
#: consumer's own reader (`usable_rollout._from_behavior_v0`) rather than
#: invented: a fake whose shape matches what the CONSUMER expects instead of what
#: the PRODUCER emits is how a defective line once got certified by an
#: end-to-end harness.
GENERATION_RECORD = {
    "raw": "<think>\nreasoning.\n</think>\nThe answer is 42.",
    "think_preopened": True,
    "natural_termination": True,
    "degeneration_triggered": False,
    "context_limit_reached": False,
    "generated_tokens": 12,
    "stop_reason": "eos",
}


def _write_battery_generations(gen_dir: Path, *, drift: bool = False) -> None:
    gen_dir.mkdir(parents=True, exist_ok=True)
    for name in C1_BATTERY_SETS:
        ids = [json.loads(x)["id"] for x in (BATTERY / f"{name}.jsonl").open()
               if x.strip()]
        with (gen_dir / f"{name}.generations.jsonl").open("w") as f:
            for i in ids:
                f.write(json.dumps({**GENERATION_RECORD, "id": i}) + "\n")
        summary = {**_summary_template(), "label": gen_dir.name,
                   "prompts": str(BATTERY / f"{name}.jsonl"), "n_samples": len(ids)}
        if drift:
            #: One MATERIAL generation field moved. The engine still produced a
            #: complete set of rollouts; they were simply not produced under the
            #: protocol the session attested.
            summary["sampling"] = {**summary.get("sampling", {}), "temperature": 0.7}
        (gen_dir / f"{name}.json").write_text(json.dumps(summary))


#: The per-set summary `uncapped_eval` writes, taken from a REAL retained one so
#: `observe_generation_protocol` — which fails closed on any missing declared
#: field — runs for real in the harness. Writing a plausible subset instead made
#: the production path fail, which is the harness working.
_SUMMARY_CACHE: dict = {}


def _summary_template() -> dict:
    if _SUMMARY_CACHE:
        return _SUMMARY_CACHE
    import verify_c1_scoring_equivalence as EQ
    for d in EQ.find_generations():
        p = d / "gsm8k.json"
        if p.is_file():
            _SUMMARY_CACHE.update(json.loads(p.read_text()))
            return _SUMMARY_CACHE
    pytest.skip("no retained generation summary to model the fake on")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Redirect every C1 root into tmp, and fake only the hardware."""
    rec = Recorder()
    audit, train, evald, work = (tmp_path / n for n in
                                 ("audit", "train", "eval", "work"))
    for attr, val in (("AUDIT", audit), ("TRAIN", train), ("EVAL", evald),
                      ("WORK", work), ("STATUS", tmp_path / "c1.status")):
        monkeypatch.setattr(D, attr, val)
    for d in (audit, audit / "probes", audit / "configs", train, evald, work):
        d.mkdir(parents=True, exist_ok=True)

    # A structurally valid C1 authorization, at a scratch path outside the repo.
    from aadistill.autoinit.c1_authorization import (
        C1Authorization, c1_harness_digest, c1_hard_ceiling_usd, load_pricing,
    )
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "c1l_h", REPO / "scripts/pod/autoinit_c1_launch.py")
    launcher = importlib.util.module_from_spec(spec)
    sys.modules["c1l_h"] = launcher
    spec.loader.exec_module(launcher)
    auth_path = tmp_path / "auth.json"
    auth = C1Authorization(
        authorization_id="TEST-NOT-A-GRANT", granted_utc="2026-09-02T00:00:00Z",
        granted_by="regression harness", plan_id="autoinit.v1.phase_c1",
        plan_hash=launcher._plan_hash(),
        science_plan_hash=(
            "fb2eeea531f9f0d11f84b77cd47dff30697122de90a072a7a80c3a7535e89280"),
        expected_usd=float(load_pricing(REPO)["totals"]["expected_usd"]),
        hard_cap_usd=c1_hard_ceiling_usd(REPO),
        per_launch_hard_usd=c1_hard_ceiling_usd(REPO),
        authorized_stages=(0, 1, 2, 3, 4, 5),
        stage_conditions={}, scope_note="test",
        authorized_session_commit="0" * 40,
        harness_source_digest=c1_harness_digest(REPO)["digest"],
        provenance_commit="0" * 40)
    auth_path.write_text(json.dumps(auth.as_dict(), indent=1))
    monkeypatch.setattr(D.C1Driver, "AUTHORIZATION_PATH", str(auth_path))
    monkeypatch.setattr(D, "REPO", REPO)
    return types.SimpleNamespace(rec=rec, tmp=tmp_path, audit=audit,
                                 train=train, eval=evald)


def _args(**over):
    ap = D.build_parser()
    a = ap.parse_args(["--soft-stop-usd", "14.7841", "--authorized-usd", "15.1475"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _fake_hardware(monkeypatch, h, *, mismatch_at=None, train_fails=None,
                   handoff_fails=False, protocol_drift_at=None):
    """Replace ONLY the hardware-bound operations. The contracts stay real."""
    rec = h.rec

    def stage_b(self):
        rec("teacher_verify")
        self.teacher_path = str(h.tmp / "teacher")
        self.complete("B", repo_id="fake", revision=D.CS.TEACHER_REVISION)
    monkeypatch.setattr(D.C1Driver, "stage_b", stage_b)

    class Step:
        def __init__(self, i, digest, expected, matches):
            self.index, self.digest_expected, self.digest_matches = i, expected, matches
            self.identity = types.SimpleNamespace(artifact_digest=digest)
            self.checkpoint_path = str(h.tmp / f"ckpt{i}")
            self.selection = {"step": i}
            self.impl_id = f"op{i}"
        def as_dict(self):
            return {"index": self.index, "artifact_digest":
                    self.identity.artifact_digest,
                    "digest_matches": self.digest_matches}

    def stage_de(self):
        D.mark("STAGE_START:D")
        rec("replay")
        runtime = {"gpu": "fake"}
        seen = []
        for i, (digest, expected) in enumerate([
                (None, None), (None, None),
                (D.CS.EXPECTED_PARENT_DIGEST, D.CS.EXPECTED_PARENT_DIGEST),
                (D.CS.EXPECTED_INCUMBENT_DIGEST, D.CS.EXPECTED_INCUMBENT_DIGEST)]):
            matches = None if expected is None else True
            if mismatch_at == i:
                digest, matches = "0" * 64, False
            step = Step(i, digest, expected, matches)
            seen.append(step)
            if matches is False:
                exc = D.FixedPathDigestMismatch(i, f"step{i}", expected, digest,
                                                {"steps": []})
                self.replay_mismatch(exc, runtime, seen)
                raise D.C1ReplayMismatch(str(exc))
            if expected is not None and matches:
                if i == 2:
                    self.complete("D", artifact_digest=digest)
                    D.mark("STAGE_START:E")
                else:
                    self.complete("E", artifact_digest=digest)
        (D.AUDIT / "c1_replay_record.json").write_text(json.dumps(
            {"schema": "aadistill.autoinit.fixed_path_replay/v1",
             "all_pinned_digests_matched": True, "runtime": runtime}, indent=1))
        self.parent, self.incumbent_step = seen[2], seen[3]
    monkeypatch.setattr(D.C1Driver, "stage_de", stage_de)

    def stage_f(self):
        D.mark("STAGE_START:F")
        rec("materialize_treatment")
        (D.AUDIT / "c1_arm_identities.json").write_text(json.dumps(
            {"schema": "aadistill.autoinit.c1_arm_identities/v1",
             "parent": self.parent.as_dict(), "shared_parent": True}, indent=1))
        self.arm_init = {"incumbent": (str(h.tmp / "inc"),
                                       D.CS.EXPECTED_INCUMBENT_DIGEST),
                         "treatment": (str(h.tmp / "trt"), "t" * 64)}
        self.complete("F", **{a: d for a, (_, d) in self.arm_init.items()})
    monkeypatch.setattr(D.C1Driver, "stage_f", stage_f)

    def release_device(self):
        rec("cuda_handoff")
        if handoff_fails:
            raise D.DeviceHandoffError("the card is still held")
        return {"handoff": {"verdict": "released"}, "need_bytes": D._trainer_bytes()}
    monkeypatch.setattr(D.C1Driver, "release_device", release_device)

    #: The ONLY hardware step of stage G. The loop, the override check, the
    #: journal, the completion count and the six-probe guard all stay REAL, so a
    #: mutation to any of them is visible here.
    def train_one(self, name, config):
        rec(f"train:{name}")
        trained = len([c for c in rec.calls if c.startswith("train:")])
        if train_fails is not None and trained == train_fails:
            raise D.C1DriverError(f"{name}: training failed rc=1")
        out_dir = D.TRAIN / name
        (out_dir / "checkpoints" / "t0" / "model").mkdir(parents=True, exist_ok=True)
        (out_dir / "checkpoints" / "latest.txt").write_text("t0")
        (out_dir / "checkpoints" / "t0" / "model" / "config.json").write_text("{}")
        (out_dir / "train_log.jsonl").write_text('{"step":1}\n')
        (out_dir / "run_manifest.json").write_text("{}")
        (out_dir / "run_completion.json").write_text(
            json.dumps({"final_step": 1023, "config_sha256": "x"}))
        return out_dir
    monkeypatch.setattr(D.C1Driver, "train_one", train_one)

    def attest(self):
        """Only the engine probe is faked. The protocol object is REAL.

        `admit_generation` compares against `self.evaluation_protocol`, so a
        placeholder here would make the comparison vacuous — the harness would
        certify a gate that never ran. The attested protocol is therefore built
        from a clean synthetic generation set, exactly as the driver builds the
        observed one, so the drift case genuinely differs.
        """
        rec("engine_probe")
        clean = D.EVAL / "_attest_reference"
        _write_battery_generations(clean)
        summaries = [json.loads(q.read_text()) for q in sorted(clean.glob("*.json"))
                     if not q.name.endswith(".generations.jsonl")]
        gen = D.observe_generation_protocol(summaries).protocol
        manifest = json.loads((D.BATTERY / "manifest.json").read_text())
        contract = D.c1_scoring_contract(D.REPO)
        self.evaluation_protocol = D.RecoveryEvaluationProtocol(
            generation=gen, scoring_contract=contract["contract"],
            scoring_digest=contract["digest"],
            battery_artifact=manifest["artifact"],
            battery_manifest_sha256=manifest["manifest_sha256"],
            battery_content_sha256=manifest["content_sha256"])
        doc = {"generation_protocol_fingerprint": gen.fingerprint,
               "evaluation_protocol_hash":
                   self.evaluation_protocol.evaluation_protocol_hash,
               "battery": {"content_sha256": D.C1_BATTERY_CONTENT_SHA256}}
        (D.AUDIT / "c1_attested_evaluation_protocol.json").write_text(
            json.dumps(doc, indent=2))
        return doc
    monkeypatch.setattr(D.C1Driver, "attest", attest)

    #: The ONLY hardware step of stage H. `require_all_trained`, the packaging,
    #: the REAL C1 scorer subprocess and the ordering all stay production code.
    def generate_one(self, name, package, gen_dir, sets):
        rec(f"evaluate:{name}")
        n = len([c for c in rec.calls if c.startswith("evaluate:")])
        _write_battery_generations(Path(gen_dir), drift=(n == protocol_drift_at))
    monkeypatch.setattr(D.C1Driver, "generate_one", generate_one)

    def build_package(model_dir, *, tokenizer_source, dest,
                      expected_sidecar_sha256):
        Path(dest).mkdir(parents=True, exist_ok=True)
        return {"tokenizer_source_rule": "the evaluated checkpoint"}
    monkeypatch.setattr(D, "build_evaluation_package", build_package)

    #: The fakes record ABSOLUTE paths, and `Path(repo) / "/abs"` is "/abs", so
    #: stage I's `REPO / path` resolves correctly without repointing REPO — which
    #: would break the REAL stage C, whose scoring-contract digest reads the
    #: repository.
    return rec


def _run(monkeypatch, h, **kw) -> tuple[int, D.C1Driver]:
    rec = _fake_hardware(monkeypatch, h, **kw)
    driver = D.C1Driver(_args())
    code = driver.run()
    return code, driver


# --- the happy path ---------------------------------------------------------

def test_the_full_session_runs_b_through_i_in_order(harness, monkeypatch):
    code, driver = _run(monkeypatch, harness)
    assert code == 0, driver.ev["stages"]
    assert driver.completed == [
        "teacher_fetch_verify", "register_operator", "replay_parent",
        "replay_incumbent", "materialize_arms", "recovery_probes", "evaluate",
        "decide"]
    assert driver.ev["outcome"] == "ALL_DONE"
    assert driver.ev["probes_trained"] == 6
    assert driver.ev["probes_evaluated"] == 6
    assert driver.ev["decision_ran"] is True
    assert driver.ev["formal_recovery_evidence"] == "OUT OF SCOPE"
    assert driver.ev["followon_started"] is False


def test_all_six_trainings_precede_the_first_evaluation(harness, monkeypatch):
    """The freshness of the confirmation battery depends on this ordering."""
    _run(monkeypatch, harness)
    calls = harness.rec.calls
    last_train = max(i for i, c in enumerate(calls) if c.startswith("train:"))
    first_eval = min(i for i, c in enumerate(calls) if c.startswith("evaluate:"))
    assert last_train < first_eval, calls
    assert len([c for c in calls if c.startswith("train:")]) == 6


def test_the_cuda_handoff_gate_precedes_every_training(harness, monkeypatch):
    _run(monkeypatch, harness)
    calls = harness.rec.calls
    assert calls.index("cuda_handoff") < min(
        i for i, c in enumerate(calls) if c.startswith("train:"))


def test_the_session_writes_every_report_the_launcher_fetches(harness, monkeypatch):
    _run(monkeypatch, harness)
    for name in ("c1_evidence.json", "c1_replay_record.json",
                 "c1_arm_identities.json", "c1_probe_results.json",
                 "c1_decision.json", "c1_attested_evaluation_protocol.json"):
        assert (harness.audit / name).is_file(), name


def test_the_decision_consumes_all_six_probes_and_the_real_rule(harness,
                                                                monkeypatch):
    _run(monkeypatch, harness)
    results = json.loads((harness.audit / "c1_probe_results.json").read_text())
    assert results["n_probes"] == 6
    assert len(results["probes"]) == 6
    assert sorted(results["seeds"]) == sorted(SEEDS)
    decision = json.loads((harness.audit / "c1_decision.json").read_text())
    assert decision["verdict"] in ("GO", "NO-GO", "INCONCLUSIVE")
    assert len(decision["per_seed_delta"]) == 3
    assert len(decision["mcnemar"]["per_seed"]) == 3
    assert decision["probe_results_sha256"] == results["results_sha256"]


def test_only_the_c1_battery_is_opened_in_stage_h(harness, monkeypatch):
    _run(monkeypatch, harness)
    for name in sorted(harness.audit.glob("*_c1_confirmation.json")):
        result = json.loads(name.read_text())
        assert result["battery"]["content_sha256"] == D.C1_BATTERY_CONTENT_SHA256
        assert result["scoring_contract"]["contract"] == "c1_confirmation_scoring@v1"
        assert result["n"] == 950 and result["n_scorable"] == 850


# --- the stop conditions ----------------------------------------------------

def test_a_replay_mismatch_trains_nothing_and_writes_its_evidence_first(
        harness, monkeypatch):
    code, driver = _run(monkeypatch, harness, mismatch_at=2)
    assert code == 30
    assert driver.ev["training_started"] is False
    assert not [c for c in harness.rec.calls if c.startswith("train:")]
    record = json.loads((harness.audit / "c1_replay_record.json").read_text())
    assert record["schema"] == "aadistill.autoinit.c1_replay_mismatch/v1"
    assert record["training_started"] is False
    status = (harness.tmp / "c1.status").read_text()
    assert status.index("C1_REPLAY_MISMATCH") > status.index("STAGE_FAILED:D")
    assert "STAGE_PASSED:D" not in status


def test_a_failed_handoff_stops_before_any_training(harness, monkeypatch):
    code, driver = _run(monkeypatch, harness, handoff_fails=True)
    assert code == 40
    assert not [c for c in harness.rec.calls if c.startswith("train:")]
    assert driver.ev["probes_trained"] == 0


def test_a_failed_fourth_training_produces_no_evaluation_and_no_decision(
        harness, monkeypatch):
    code, driver = _run(monkeypatch, harness, train_fails=4)
    assert code == 40
    assert not [c for c in harness.rec.calls if c.startswith("evaluate:")]
    assert driver.ev["probes_evaluated"] == 0
    assert driver.ev["decision_ran"] is False
    assert not (harness.audit / "c1_decision.json").exists()
    assert "C1_INCOMPLETE" in (harness.tmp / "c1.status").read_text()


# --- mutations --------------------------------------------------------------

@pytest.mark.parametrize("n_trained", [0, 1, 5])
def test_mutation_evaluating_before_six_trainings_is_refused(harness, monkeypatch,
                                                             n_trained):
    """Stage H's own guard, exercised directly — including at FIVE.

    The first version of this test only ever supplied one completion, so a guard
    weakened from `!= 6` to `< 5` still passed it. Five is the case that matters:
    it is what a session that lost exactly one probe looks like.
    """
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    keys = [(a, s) for a in ("incumbent", "treatment") for s in SEEDS]
    driver.training = {k: {} for k in keys[:n_trained]}
    with pytest.raises(D.C1DriverError, match="six training completions"):
        D.C1Driver.stage_h(driver)


def test_mutation_deciding_on_five_results_is_refused(harness, monkeypatch):
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.scored = {("incumbent", s): {} for s in SEEDS}
    with pytest.raises(D.C1DriverError, match="not 6"):
        driver.stage_i()


def test_mutation_an_out_of_order_stage_is_refused(harness, monkeypatch):
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.complete("B")
    with pytest.raises(Exception, match="out of order"):
        driver.complete("F")


def test_mutation_a_descriptor_without_its_identity_is_refused(harness,
                                                              monkeypatch):
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.arm_init = {"incumbent": (str(harness.tmp), "i" * 64)}
    with pytest.raises(KeyError):
        driver.descriptors()


def test_mutation_a_probe_config_that_moves_a_frozen_field_is_refused(
        harness, monkeypatch):
    _fake_hardware(monkeypatch, harness)
    driver = D.C1Driver(_args())
    driver.arm_init = {a: (str(harness.tmp / a), a * 32) for a in ("incumbent",
                                                                  "treatment")}
    d = driver.descriptors()[0]
    monkeypatch.setattr(D, "C1_PROBE_OVERRIDES", frozenset({"run_name"}))
    with pytest.raises(D.C1DriverError, match="outside the allowed override set"):
        driver.probe_config(d)


# --- the paired aggregation -------------------------------------------------

def _rows(n_scorable=850, ids=None, dup=False):
    rows = []
    for name, (n, domain, scorable) in C1_BATTERY_SETS.items():
        for i in range(n):
            rows.append({"id": f"{name}-{i}", "set": name, "domain": domain,
                         "scorable": scorable, "usable": True,
                         "correct": scorable and i % 7 == 0})
    if dup:
        rows.append(dict(rows[0]))
    return rows


def test_the_aggregation_refuses_a_duplicate_prompt_id():
    ps = {(a, s): _rows(dup=(a == "incumbent" and s == SEEDS[0]))
          for a in ("incumbent", "treatment") for s in SEEDS}
    with pytest.raises(C1ResultsError, match="duplicate prompt id"):
        decision_inputs(ps, seeds=SEEDS)


def test_the_aggregation_refuses_a_missing_probe():
    ps = {(a, s): _rows() for a in ("incumbent", "treatment") for s in SEEDS}
    ps.pop(("treatment", SEEDS[-1]))
    with pytest.raises(C1ResultsError, match="2 arms x 3 seeds"):
        decision_inputs(ps, seeds=SEEDS)


def test_the_aggregation_refuses_a_short_battery():
    ps = {(a, s): _rows() for a in ("incumbent", "treatment") for s in SEEDS}
    ps[("treatment", SEEDS[0])] = ps[("treatment", SEEDS[0])][:-1]
    with pytest.raises(C1ResultsError, match="949 prompts, not 950"):
        decision_inputs(ps, seeds=SEEDS)


def test_the_aggregation_uses_the_frozen_denominators_and_strata():
    ps = {(a, s): _rows() for a in ("incumbent", "treatment") for s in SEEDS}
    inputs = decision_inputs(ps, seeds=SEEDS)
    assert inputs.audit["n_prompts"] == 950
    assert inputs.audit["n_scorable"] == 850
    assert sum(inputs.audit["strata_sizes"].values()) == 850
    assert "code" not in inputs.audit["strata_sizes"]
    assert len(inputs.correct["incumbent"][SEEDS[0]]) == 850


# --- the generation-protocol admission gate ---------------------------------

def test_a_drifted_generation_protocol_is_refused_before_scoring(harness,
                                                                 monkeypatch):
    """A complete set of rollouts produced under the wrong protocol is not a result.

    The attestation is a statement about the runtime, made once from an engine
    probe. It is not evidence about any particular probe's rollouts. Before this
    gate, stage H recorded the observed fingerprint AFTER scoring and never
    compared it to anything — so a drifted probe would have been scored, and its
    result would have carried the EXPECTED identity.
    """
    code, driver = _run(monkeypatch, harness, protocol_drift_at=1)
    assert code == 40
    calls = harness.rec.calls
    assert len([c for c in calls if c.startswith("evaluate:")]) == 1, calls
    assert driver.ev["probes_evaluated"] == 0
    assert driver.ev["decision_ran"] is False
    assert not (harness.audit / "c1_decision.json").exists()
    assert not (harness.audit / "c1_probe_results.json").exists()
    assert not list(harness.audit.glob("*_c1_confirmation.json"))
    assert "C1_INCOMPLETE" in (harness.tmp / "c1.status").read_text()
    #: The generations and the refusal both survive; the artifact spec collects
    #: them, so a refusal is a diagnosis rather than a loss.
    admission = json.loads(
        next(harness.audit.glob("*_generation_admission.json")).read_text())
    assert admission["comparable"] is False
    assert admission["reason"]
    assert list(harness.eval.glob("*/*.generations.jsonl"))


def test_every_admitted_probe_carries_its_observed_protocol(harness, monkeypatch):
    _run(monkeypatch, harness)
    results = json.loads((harness.audit / "c1_probe_results.json").read_text())
    attested = json.loads(
        (harness.audit / "c1_attested_evaluation_protocol.json").read_text())
    hashes = {p["observed_evaluation_protocol_hash"] for p in results["probes"]}
    prints = {p["observed_generation_fingerprint"] for p in results["probes"]}
    assert len(hashes) == 1 and next(iter(hashes))
    assert len(prints) == 1 and next(iter(prints))
    assert results["observed_evaluation_protocol_hash"] == next(iter(hashes))
    assert (results["attested_evaluation_protocol_hash"]
            == attested["evaluation_protocol_hash"] == next(iter(hashes)))
    assert "admission_rule" in results
    for probe in results["probes"]:
        rec = json.loads((harness.audit
                          / f"{probe['probe_id']}_generation_admission.json").read_text())
        assert rec["comparable"] is True
        assert rec["evaluation_protocol_hash"] == probe[
            "observed_evaluation_protocol_hash"]


def test_the_scorer_is_given_the_observed_fingerprint_not_the_attested_one():
    """Provenance direction. They are equal once admitted; which one is recorded
    is the difference between evidence and expectation."""
    src = DRIVER_SRC.read_text()
    call = src.split('"--generation-fingerprint"', 1)[1][:200]
    assert 'observed["generation_fingerprint"]' in call
    assert "attested[" not in call


def test_admission_runs_before_the_scorer_in_source_order():
    src = DRIVER_SRC.read_text()
    body = src.split("def stage_h", 1)[1]
    assert body.index("self.admit_generation(") < body.index("_scoring")


def test_probe_results_refuse_a_probe_with_no_admitted_protocol():
    """Mutation target: dropping the admission would leave these fields empty."""
    from aadistill.autoinit.c1_probe_results import (
        C1ProbeRecord, C1ResultsError, build_probe_results, decision_inputs,
    )
    rows = _rows()
    ps = {(a, s): rows for a in ("incumbent", "treatment") for s in SEEDS}
    inputs = decision_inputs(ps, seeds=SEEDS)
    def rec(arm, seed, **over):
        return C1ProbeRecord(
            probe_id=f"{arm}.{seed}", arm=arm, seed=seed,
            initialization_artifact_digest="d" * 64, trained_run={},
            result_path="r", result_sha256="h", per_sample_path="p",
            per_sample_sha256="h", generations={},
            counts={"n": 950, "usable": 1, "correct": 1, "n_scorable": 850,
                    "usable_scorable": 1},
            rates={}, per_capability={}, scoring_contract={"digest": "c"},
            battery={"content_sha256": "b"},
            **{"observed_generation_fingerprint": "f" * 64,
               "observed_evaluation_protocol_hash": "e" * 64, **over})
    good = [rec(a, s) for a in ("incumbent", "treatment") for s in SEEDS]
    build_probe_results(good, plan_hash="p", seeds=SEEDS, inputs=inputs,
                        attested_evaluation_protocol_hash="e" * 64)
    blank = [rec(a, s, observed_evaluation_protocol_hash="")
             if (a, s) == ("treatment", SEEDS[0]) else rec(a, s)
             for a in ("incumbent", "treatment") for s in SEEDS]
    with pytest.raises(C1ResultsError, match="observed evaluation-protocol"):
        build_probe_results(blank, plan_hash="p", seeds=SEEDS, inputs=inputs)
    with pytest.raises(C1ResultsError, match="not the attested"):
        build_probe_results(good, plan_hash="p", seeds=SEEDS, inputs=inputs,
                            attested_evaluation_protocol_hash="z" * 64)


# --- D versus E: which gate actually failed ---------------------------------
#
# `run()` walks `(("DE", self.stage_de), ...)` and reports `self.fail(letter[0])`,
# so EVERY ordinary exception inside that method is written as `STAGE_FAILED:D`.
# But `stage_de` owns two observable gates: the moment the parent digest matches
# it calls `complete("D")` and emits `STAGE_START:E`. So a failure during the
# INCUMBENT step was recorded against D — overwriting a Stage-D entry that had
# already passed, while `stages_completed` still carried `replay_parent`. The
# session's own evidence then contradicted itself about which gate held.
#
# These drive the REAL `stage_de` with a faked materializer, so the on_step
# callback, the completion order and the record write are all production code.

#: Captured before anything patches it, so the cases below can restore the REAL
#: `stage_de` after `_fake_hardware` has replaced it for every other stage.
_REAL_STAGE_DE = D.C1Driver.stage_de


class _ReplayStep:
    """A `StepResult` as far as `stage_de` and `write_replay_record` read one."""

    def __init__(self, index, expected=None, actual=None, matches=None):
        self.index = index
        self.impl_id = f"op{index}"
        self.profile_id = "calib.none@v1"
        self.kind = ("DEPTH", "FFN", "RESIDUAL_WIDTH", "ATTENTION")[index]
        self.result_spec_hash = "s" * 64
        self.identity = types.SimpleNamespace(artifact_digest=actual or "d" * 64)
        self.checkpoint_path = f"/tmp/ckpt{index}"
        self.seconds = 0.0
        self.selection = {"step": index}
        self.trace = {}
        self.digest_expected = expected
        self.digest_matches = matches

    def as_dict(self):
        return {"index": self.index, "impl_id": self.impl_id,
                "artifact_digest": self.identity.artifact_digest,
                "digest_expected": self.digest_expected,
                "digest_matches": self.digest_matches,
                "checkpoint_path": self.checkpoint_path}


def _replay_plan(*, stop_after=None, mismatch_at=None):
    """The four steps the incumbent replay emits, with the two frozen pins."""
    pins = {2: D.CS.EXPECTED_PARENT_DIGEST, 3: D.CS.EXPECTED_INCUMBENT_DIGEST}
    out = []
    for i in range(4):
        exp = pins.get(i)
        if exp is None:
            out.append(_ReplayStep(i))
        elif mismatch_at == i:
            out.append(_ReplayStep(i, expected=exp, actual="0" * 64, matches=False))
        else:
            out.append(_ReplayStep(i, expected=exp, actual=exp, matches=True))
        if stop_after is not None and i == stop_after:
            break
    return out


def _drive_replay(monkeypatch, harness, *, stop_after=None, mismatch_at=None,
                  boom=None, record_write=None, through_de_only=False):
    """Install a materializer that feeds the REAL `stage_de`.

    `through_de_only` runs B, C and DE instead of the whole session. The
    successful case needs that: a full `run()` reaches stage H's `attest()`,
    which models its fake on a RETAINED generation summary that exists only on
    the dev box — so the case would SKIP under the pod simulation, and the one
    test proving stage E waits for its record would not run on the machine that
    matters. The failing cases stop inside DE anyway and are unaffected.
    """
    _fake_hardware(monkeypatch, harness)
    monkeypatch.setattr(D.C1Driver, "stage_de", _REAL_STAGE_DE)

    steps = _replay_plan(stop_after=stop_after, mismatch_at=mismatch_at)

    def fake_materialize(spec, *, adapter, root_loader, workdir, repo_root,
                         on_step=None, **kw):
        for s in steps:
            if on_step is not None:
                on_step(s)
            if s.digest_matches is False:
                raise D.FixedPathDigestMismatch(
                    s.index, f"step{s.index}", s.digest_expected,
                    s.identity.artifact_digest, {"steps": []})
        if boom is not None:
            raise boom
        return steps

    monkeypatch.setattr(D, "materialize_fixed_path", fake_materialize)
    from aadistill.autoinit.adapters.qwen3 import QWEN3_ADAPTER
    monkeypatch.setattr(QWEN3_ADAPTER, "load",
                        lambda *a, **k: types.SimpleNamespace())
    if record_write is not None:
        monkeypatch.setattr(D, "write_replay_record", record_write)

    def stage_b(self):
        self.teacher_path = str(harness.tmp / "teacher")
        self.complete("B", repo_id="fake", revision=D.CS.TEACHER_REVISION)
    monkeypatch.setattr(D.C1Driver, "stage_b", stage_b)

    driver = D.C1Driver(_args())
    if through_de_only:
        driver.stage_b()
        driver.stage_c()
        driver.stage_de()
        code = 0
    else:
        code = driver.run()
    return code, driver, (harness.tmp / "c1.status").read_text()


def test_A_an_ordinary_failure_before_the_parent_match_is_stage_D(
        harness, monkeypatch):
    code, driver, status = _drive_replay(
        monkeypatch, harness, stop_after=1,
        boom=RuntimeError("CUDA OOM while loading shard 2"))

    assert code == 40
    assert "MARKER:STAGE_FAILED:D" in status
    assert "MARKER:STAGE_PASSED:D" not in status
    assert "MARKER:STAGE_START:E" not in status
    assert "E" not in driver.ev["stages"]
    assert driver.ev["stages"]["D"]["passed"] is False
    assert "MARKER:C1_FAILED" in status
    assert "C1_REPLAY_MISMATCH" not in status
    assert not (D.AUDIT / "c1_replay_record.json").exists()
    assert "replay_parent" not in driver.completed


def test_B_an_ordinary_failure_after_the_parent_match_is_stage_E(
        harness, monkeypatch):
    """The defect: this used to be written as STAGE_FAILED:D, on top of a
    Stage-D entry that had already passed."""
    code, driver, status = _drive_replay(
        monkeypatch, harness, stop_after=2,
        boom=RuntimeError("CUDA OOM during the incumbent ATTENTION"))

    assert code == 40
    assert status.index("MARKER:STAGE_PASSED:D") < status.index("MARKER:STAGE_START:E")
    assert "MARKER:STAGE_FAILED:E" in status
    assert "MARKER:STAGE_FAILED:D" not in status
    assert "MARKER:STAGE_PASSED:E" not in status

    assert driver.ev["stages"]["D"]["passed"] is True
    assert driver.ev["stages"]["E"]["passed"] is False
    assert "replay_parent" in driver.completed
    assert "replay_incumbent" not in driver.completed
    assert driver.ev["stages_completed"] == driver.completed

    assert "MARKER:C1_FAILED" in status
    assert "C1_REPLAY_MISMATCH" not in status
    assert "mismatch" not in driver.ev["stages"]["E"]["reason"].lower()


def test_C_a_replay_record_that_cannot_be_written_fails_stage_E_only(
        harness, monkeypatch):
    def explode(*a, **k):
        raise OSError("No space left on device")

    code, driver, status = _drive_replay(monkeypatch, harness,
                                         record_write=explode)

    assert code == 40
    assert driver.ev["stages"]["D"]["passed"] is True
    assert driver.ev["stages"]["E"]["passed"] is False
    assert "MARKER:STAGE_PASSED:E" not in status
    assert "MARKER:C1_FAILED" in status
    assert "C1_REPLAY_MISMATCH" not in status
    assert "replay_incumbent" not in driver.completed


def test_C2_a_replay_record_that_reads_back_wrong_fails_stage_E_only(
        harness, monkeypatch):
    """Written, readable, and WRONG. The readback must be a check, not a ritual."""
    def bad_record(spec, results, path, **k):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(
            {"schema": "aadistill.autoinit.fixed_path_replay/v1",
             "path_hash": "0" * 64,            # not this path
             "all_pinned_digests_matched": True, "n_pinned": 2, "steps": []}))
        return Path(path)

    code, driver, status = _drive_replay(monkeypatch, harness,
                                         record_write=bad_record)
    assert code == 40
    assert driver.ev["stages"]["D"]["passed"] is True
    assert driver.ev["stages"]["E"]["passed"] is False
    assert "MARKER:STAGE_PASSED:E" not in status
    assert "C1_REPLAY_MISMATCH" not in status


def test_D_a_complete_replay_marks_E_only_after_the_record_is_readable(
        harness, monkeypatch):
    order: list = []
    real = D.write_replay_record

    def watched(*a, **k):
        order.append("record_written")
        return real(*a, **k)
    monkeypatch.setattr(D, "write_replay_record", watched)

    code, driver, status = _drive_replay(monkeypatch, harness,
                                         through_de_only=True)

    assert status.count("MARKER:STAGE_PASSED:D") == 1
    assert status.count("MARKER:STAGE_PASSED:E") == 1
    assert "STAGE_FAILED" not in status
    assert "C1_REPLAY_MISMATCH" not in status
    assert order == ["record_written"]

    doc = json.loads((D.AUDIT / "c1_replay_record.json").read_text())
    assert doc["schema"] == "aadistill.autoinit.fixed_path_replay/v1"
    assert doc["all_pinned_digests_matched"] is True
    assert doc["n_pinned"] == 2
    assert driver.completed.index("replay_parent") < \
        driver.completed.index("replay_incumbent")
    assert driver.ev["stages"]["E"]["replay_record"].endswith(
        "c1_replay_record.json")
    assert driver.ev["stages"]["E"]["all_pinned_digests_matched"] is True


def test_E_a_step2_mismatch_is_still_stage_D_and_a_replay_mismatch(
        harness, monkeypatch):
    code, driver, status = _drive_replay(monkeypatch, harness, mismatch_at=2)
    assert code == 30
    assert "MARKER:STAGE_FAILED:D" in status
    assert "MARKER:STAGE_PASSED:D" not in status
    assert status.index("MARKER:STAGE_FAILED:D") < status.index(
        "MARKER:C1_REPLAY_MISMATCH")
    rec = json.loads((D.AUDIT / "c1_replay_record.json").read_text())
    assert rec["schema"] == "aadistill.autoinit.c1_replay_mismatch/v1"
    assert rec["stage"] == "D"


def test_E2_a_step3_mismatch_is_stage_E_and_a_replay_mismatch(
        harness, monkeypatch):
    code, driver, status = _drive_replay(monkeypatch, harness, mismatch_at=3)
    assert code == 30
    assert status.index("MARKER:STAGE_PASSED:D") < status.index(
        "MARKER:STAGE_FAILED:E")
    assert "MARKER:STAGE_FAILED:D" not in status
    assert status.index("MARKER:STAGE_FAILED:E") < status.index(
        "MARKER:C1_REPLAY_MISMATCH")
    rec = json.loads((D.AUDIT / "c1_replay_record.json").read_text())
    assert rec["schema"] == "aadistill.autoinit.c1_replay_mismatch/v1"
    assert rec["stage"] == "E"
    assert driver.ev["stages"]["D"]["passed"] is True


# --- a DEVICE failure in stage F: what attempt 9 actually was ----------------
#
# Attempt 9 passed both replay gates and then died inside the treatment operator
# with `Expected all tensors to be on the same device`. Nothing compared a digest
# after that, so classifying it as a replay mismatch would have put a false claim
# about the frozen path's reproducibility into the record — and the record it had
# already written says the opposite: both digests MATCHED.
#
# The launcher's neutral failure prose was repaired for this, but prose is not a
# gate. These tests hold the driver to the classification and to the artifacts,
# for the exact exception the L40S raised.

ATTEMPT_9_ERROR = ("Expected all tensors to be on the same device, but found at "
                   "least two devices, cuda:0 and cpu!")


def _drive_stage_f_device_failure(monkeypatch, harness):
    """Replay passes, then the treatment operator raises on device placement."""
    _fake_hardware(monkeypatch, harness)

    def exploding_stage_f(self):
        D.mark("STAGE_START:F")
        raise RuntimeError(ATTEMPT_9_ERROR)
    monkeypatch.setattr(D.C1Driver, "stage_f", exploding_stage_f)

    driver = D.C1Driver(_args())
    return driver.run(), driver, D.STATUS.read_text()


def test_a_stage_f_device_failure_is_C1_FAILED_and_never_a_replay_mismatch(
        harness, monkeypatch):
    code, driver, status = _drive_stage_f_device_failure(monkeypatch, harness)

    assert code == 40
    assert "MARKER:C1_FAILED" in status
    assert "C1_REPLAY_MISMATCH" not in status
    assert "MARKER:STAGE_FAILED:F" in status
    assert driver.ev["outcome"] == "C1_FAILED"
    assert driver.ev["training_started"] is False

    stage = driver.ev["stages"]["F"]
    assert stage["passed"] is False
    assert "RuntimeError" in stage["reason"] and "same device" in stage["reason"]
    assert "mismatch_record" not in stage, (
        "a device failure compared no digest; it cannot cite mismatch evidence")


def test_a_stage_f_device_failure_leaves_the_passing_replay_record_intact(
        harness, monkeypatch):
    """The stage-D/E product must survive stage F, and must still say PASSED.

    This is the half of attempt 9 that IS a scientific observation: both frozen
    digests reproduced under a real runtime. A stage-F failure must not overwrite
    it, downgrade it, or convert it into a mismatch record.
    """
    _drive_stage_f_device_failure(monkeypatch, harness)

    rec = json.loads((D.AUDIT / "c1_replay_record.json").read_text())
    assert rec["schema"] == "aadistill.autoinit.fixed_path_replay/v1"
    assert rec["all_pinned_digests_matched"] is True
    assert rec["schema"] != "aadistill.autoinit.c1_replay_mismatch/v1"


def test_a_stage_f_device_failure_writes_no_treatment_artifacts(
        harness, monkeypatch):
    """Nothing may suggest a treatment arm exists when the operator never returned."""
    _, driver, _ = _drive_stage_f_device_failure(monkeypatch, harness)

    assert not (D.AUDIT / "c1_treatment_record.json").exists()
    assert not (D.AUDIT / "c1_arm_identities.json").exists()
    assert not (D.AUDIT / "c1_decision.json").exists()
    assert driver.ev["stages"]["F"]["passed"] is False
    assert "G" not in driver.ev["stages"], "stage G must not have been entered"


def test_the_real_stage_f_writes_no_record_when_the_operator_raises(
        harness, monkeypatch, treatment_registered):
    """The REAL `stage_f`, with the materializer raising the attempt-9 error.

    `_fake_hardware` replaces stage F wholesale, so the test above proves the
    run-loop classification but not that the real stage writes its record only
    after materialization succeeds. This drives the production `stage_f`.
    """
    driver = D.C1Driver(_args())
    driver.teacher_path = str(harness.tmp / "teacher")
    driver.completed = [D.CS.stage(x).stage_id for x in ("B", "C")]
    _capture_root(monkeypatch, driver)
    driver.stage_de()
    driver.parent = _FakeStep(2)
    _capture_root(monkeypatch, driver)

    def exploding(*a, **kw):
        raise RuntimeError(ATTEMPT_9_ERROR)
    monkeypatch.setattr(D, "materialize_fixed_path_suffix", exploding)

    with pytest.raises(RuntimeError, match="same device"):
        driver.stage_f()

    assert not (D.AUDIT / "c1_treatment_record.json").exists()
    assert not (D.AUDIT / "c1_arm_identities.json").exists()
    assert not driver.arm_init, (
        f"stage F registered arms after the operator raised: {driver.arm_init}")
