"""The recovery continuation must be an executable session, not a set of helpers.

`bd6ca1e` shipped the strict importer, the device handoff and
`continuation_budget()` — and the production path still priced with `budget()`,
launched `--stage all` against the full driver, and ran `run_phase_a_search()`
unconditionally. Authorizing it would have rerun the 203-minute search this work
exists to avoid. The helpers existing is not the same as the path using them,
which is what these tests are for.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in ("src", "scripts/pod", "scripts/autoinit"):
    sys.path.insert(0, str(REPO / p))

LAUNCH = REPO / "scripts/pod/autoinit_recovery_continuation_launch.py"
DRIVER = REPO / "scripts/pod/autoinit_recovery_continuation_driver.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def launcher():
    return load(LAUNCH, "rc_launch")


@pytest.fixture(scope="module")
def spec(launcher):
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/x", "--session-commit", "0" * 40, "--bundle", "b"])
    return launcher.spec(args)


# --- 1. the search must be structurally unreachable -------------------------

def test_the_continuation_driver_never_imports_the_search_module():
    """Structural, not conventional. `phase_a_search` defines
    `run_phase_a_search`; importing it puts the search one attribute away."""
    tree = ast.parse(DRIVER.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    offenders = {m for m in imported if "phase_a_search" in m}
    assert not offenders, f"the continuation driver imports {offenders}"


def test_the_continuation_driver_never_calls_a_search():
    src = DRIVER.read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    body = code.split('"""', 2)[-1]          # strip the module docstring
    assert "run_phase_a_search" not in body, (
        "a search call appeared in the continuation driver")


def test_the_continuation_stage1_imports_and_does_not_delegate():
    """It overrides `stage1` and never calls the parent's, which is the one that
    searches."""
    tree = ast.parse(DRIVER.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    stage1 = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "stage1")
    body = ast.unparse(stage1)
    assert "import_stage1_result" in body
    assert "super().stage1" not in body and "PhaseADriver.stage1" not in body


def test_the_continuation_has_no_stage_flag_that_searches():
    mod = load(DRIVER, "rc_driver_parser")
    action = next(a for a in mod.build_parser()._actions
                  if a.dest == "stage")
    assert set(action.choices) == {"all"}
    assert "search" not in " ".join(action.choices)


# --- 2. the derived budget is the one that prices it ------------------------

def test_the_session_is_priced_by_the_continuation_budget(spec, launcher):
    plan = spec.budget.plan(price_per_hour=0.99, authorized_usd=16.7456)
    assert plan.expected_minutes == pytest.approx(904.44, abs=0.01)
    assert plan.expected_usd == pytest.approx(14.9233, abs=1e-4)
    assert plan.hard_terminate_usd == pytest.approx(16.7456, abs=1e-4)


def test_the_session_carries_no_stage_1_phase_or_reserve(spec):
    plan = spec.budget.plan(price_per_hour=0.99, authorized_usd=16.7456)
    assert not [p for p in plan.breakdown if p.name == "stage1_beam_search"]
    assert plan.soft_stop_reserves == ()


def test_it_is_not_priced_at_the_full_search_ceiling(spec):
    """$23.0484 funds a search this session does not run."""
    plan = spec.budget.plan(price_per_hour=0.99, authorized_usd=16.7456)
    assert plan.hard_terminate_usd < 23.0484 - 6.0


# --- 3. the five leaves are declared session inputs -------------------------

def leaf_inputs(spec, launcher):
    return [r for r in spec.setup.relay_inputs
            if r.dest and r.dest.startswith(launcher.STAGED_INTO)]


def test_the_five_leaves_are_declared_in_the_selected_order(spec, launcher):
    """They travel by RELAY now, not by scp — attempt 2 died pushing the first
    one — but the order is still the ranking and still comes from the record."""
    if not launcher.transport_is_verified():
        pytest.skip("no verified transport; covered by the refusal test below")
    staged = leaf_inputs(spec, launcher)
    seen: list[str] = []
    for r in staged:
        sid = r.dest.rsplit("/", 1)[-1]
        if sid not in seen:
            seen.append(sid)
    recorded = [l["state_id"] for l in launcher.selected_leaf_identities()]
    assert seen == recorded, "order is the ranking"
    assert len(staged) == 15, "five leaves x three files"


def test_the_leaf_identities_come_from_the_committed_record(launcher):
    """Not restated in the launcher: a second copy of five ids is a second
    thing to keep in step."""
    dur = json.loads(
        (launcher.STAGE1_EVIDENCE / "selected_leaf_durability.json").read_text())
    ids = [l["state_id"] for l in launcher.selected_leaf_identities()]
    assert ids == [r["state_id"] for r in dur["leaves"]]
    src = LAUNCH.read_text()
    for sid in ids:
        assert sid not in src, f"{sid[:12]} is hard-coded in the launcher"


def test_the_leaves_are_staged_through_the_declared_relay_contract(spec, launcher):
    """Through `SESSION_RELAY_INPUTS`, from the transport repo, with a digest on
    every file — so a session that failed to declare them gets none rather than
    finding them on some undeclared path.

    They are no longer `LOCAL_ASSETS`: pushing 1.110 GiB by scp needs 1.99 MB/s
    to fit the launcher's 600 s per-asset timeout, and the dev box is 0.72 MB/s.
    """
    assert "SESSION_RELAY_INPUTS" in spec.setup.required_env
    if not launcher.transport_is_verified():
        pytest.skip("no verified transport; covered by the refusal test below")
    staged = leaf_inputs(spec, launcher)
    assert staged, "the leaves are not declared as relay inputs"
    for r in staged:
        assert r.repo == launcher.TRANSPORT_REPO
        assert r.sha256, f"{r.path} is staged with no digest"
        assert r.path.startswith("phase_a_attempt12/"), (
            "the remote path must identify the attempt")
        assert r.dest.rsplit("/", 1)[-1] in r.path, (
            "the remote path must identify the state id")


def test_no_large_checkpoint_travels_by_scp_any_more(spec, launcher):
    """The regression that cost attempt 2: a multi-GiB LOCAL_ASSET is a push
    across the dev-box uplink under a 600 s timeout."""
    for a in spec.setup.local_assets:
        src = REPO / a.repo_path if not a.repo_path.startswith("/") else Path(a.repo_path)
        if not src.exists():
            continue
        size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) \
            if src.is_dir() else src.stat().st_size
        assert size < 200 * 2**20, (
            f"{a.repo_path} is {size / 2**20:.0f} MiB on the scp path; at the "
            "measured 0.72 MB/s that cannot fit the 600 s per-asset timeout")


def test_the_frozen_phase_a_assets_are_still_declared(spec):
    """The continuation still needs what the shared setup verifies."""
    installed = {f"{a.install_to}/{a.dest_name}" for a in spec.setup.local_assets}
    assert "artifacts/stage1/state_eval_v1" in installed
    assert "artifacts/stage3/recovery_search_v2" in installed


def test_a_missing_preserved_leaf_refuses_before_a_pod_exists(launcher, tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(launcher, "CKPT_STORE", tmp_path / "empty")

    class Ctx:
        evidence: dict = {}
    ok, why = launcher.selected_leaves_present_gate(Ctx)
    assert not ok and "absent" in why


def test_the_leaf_gate_reflects_whether_a_verified_transport_exists(launcher):
    """Both branches are asserted, because both are real states of this repo.

    With a verified transport the gate passes and names five leaves. Without
    one — which is where the account-wide Hugging Face private-storage limit
    left it on 2026-08-22 — it must **refuse**, at `$0`, naming the missing
    transport. Silence in that state would let a session launch with nothing to
    stage.
    """
    class Ctx:
        evidence: dict = {}
    if not Path(launcher.CKPT_STORE).is_dir():
        pytest.skip("the canonical leaves are not on this host")
    ok, why = launcher.selected_leaves_present_gate(Ctx)
    if launcher.transport_is_verified():
        assert ok, why
        assert Ctx.evidence["precheck"]["selected_leaves"]["n"] == 5
        assert Ctx.evidence["precheck"]["selected_leaves"]["transport_files"] == 15
    else:
        assert not ok, "the gate passed with no verified transport"
        assert "no verified transport" in why
        assert Ctx.evidence["precheck"]["selected_leaves"][
            "transport_verified"] is False


def test_an_unverified_transport_declares_no_leaf_inputs(launcher):
    """The other half: a session with no verified transport must not silently
    declare relay inputs that would 404 on the pod."""
    if launcher.transport_is_verified():
        pytest.skip("the transport is verified; the empty branch is not live")
    assert launcher.selected_leaf_inputs() == ()


# --- 4. the strict importer is the one used ---------------------------------

def test_the_driver_uses_the_shared_importer_not_its_own(launcher):
    """No second reconstruction implementation."""
    src = DRIVER.read_text()
    assert "from aadistill.autoinit.stage1_import import" in src
    tree = ast.parse(src)
    defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not {d for d in defs if "reconstruct" in d or "rebuild" in d}, (
        "the driver defines its own reconstruction")


# --- 5 and 6. control measurement, then the handoff, then stage 2 -----------

def test_the_control_is_measured_then_admitted_then_handed_off():
    """Order is the contract: import, measure, admit, release, headroom."""
    tree = ast.parse(DRIVER.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    body = ast.unparse(next(n for n in cls.body
                            if isinstance(n, ast.FunctionDef) and n.name == "stage1"))
    order = [body.index(tok) for tok in (
        "import_stage1_result", "measure_control", "attach_evaluation",
        "admit_leaves", "release_to_subprocess", "require_headroom")]
    assert order == sorted(order), "the stage-1 contract is out of order"


def test_the_control_measurement_is_not_invented():
    """It is measured on the frozen suite through the real evaluator, never
    backfilled from the record."""
    src = DRIVER.read_text()
    assert "StateEvaluator" in src and "prime_reference" in src
    assert "STATE_EVAL" in src, "the control must be measured on the frozen suite"
    for forbidden in ("copy_evaluation", "infer_", "backfill", "fake_eval"):
        assert forbidden not in src


def test_the_handoff_releases_the_teacher_and_evaluator():
    src = DRIVER.read_text()
    assert "release_to_subprocess(drop=[teacher, evaluator])" in src
    assert "del teacher, evaluator" in src


def test_the_preserved_leaf_gate_is_wired_into_the_spec(spec, launcher):
    """Testing the gate function proves nothing about whether it runs.

    Dropping it from `precheck` passed every other test in this file — the same
    helper-versus-wiring gap that let `bd6ca1e` ship unusable primitives.
    """
    assert launcher.selected_leaves_present_gate in spec.precheck


# --- the actual entrypoint, executed ----------------------------------------

def test_the_real_stage1_entrypoint_imports_measures_admits_and_hands_off(
        tmp_path, monkeypatch):
    """`stage1()` itself, on the real preserved bytes.

    Only the control measurement is substituted, because it needs a GPU: the
    import, the admission gate, the handoff and the ordering are the real ones.
    """
    if not Path("/home/ecs-user/aad-artifacts/autoinit/phase_a").is_dir():
        pytest.skip("the preserved leaves are not on this host")

    from aadistill.autoinit.metrics import StateEvaluator  # noqa: F401
    from write_preregistration import build_frozen_plan

    mod = load(DRIVER, "rc_driver_exec")
    # The leaves are staged into the repo on a pod; here they already exist in
    # the canonical store, so point the driver at it.
    monkeypatch.setattr(mod, "STAGED_LEAVES",
                        Path("/home/ecs-user/aad-artifacts/autoinit/phase_a"))
    monkeypatch.setattr(mod, "AUDIT", tmp_path / "audit")
    (tmp_path / "audit").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "mark", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "say", lambda *_a, **_k: None)

    class Args:
        stage = "all"; image_digest = "sha256:test"; rate = 0.99
        spent_usd = 0.0; soft_stop_usd = 16.42; authorized_usd = 16.7456
        probe_train_minutes = 61.55; probe_battery_minutes = 9.82
        search_minutes = 0.0; search_deadline_minutes = 0.0

    driver = object.__new__(mod.RecoveryContinuationDriver)
    driver.ev = {}
    driver.plan = build_frozen_plan(REPO)
    driver.leaves, driver.control_state, driver.search_result = [], None, None
    driver.enter = lambda _s: None
    recorded = {}
    driver.record = lambda stage, ok, *a, **kw: (
        recorded.update({"stage": stage, "ok": ok, "args": a, "kw": kw}) or ok)

    # The one substitution: a real StateEvaluation bound to the control's own
    # artifact digest, which is exactly what a GPU measurement would produce.
    from aadistill.autoinit.state import StateEvaluation

    def toy_measure(control, adapter):
        ev = StateEvaluation(
            artifact_digest=control.artifact_digest,
            suite_id="state_eval@v1", suite_hash="0" * 64,
            reference="teacher", values={"state.teacher_kl.equal_domain_mean": 1.0,
                                         "state.teacher_kl.worst_domain": 1.0,
                                         "state.critical_token_kl": 1.0},
            positions=1, detail={}, measured_utc=None, runtime={})
        return ev, object(), object()

    monkeypatch.setattr(mod.RecoveryContinuationDriver, "measure_control",
                        lambda self, c, a: toy_measure(c, a))

    assert driver.stage1() is True, recorded

    # the five, in the ranking's order
    dur = json.loads(
        (mod.EVIDENCE / "selected_leaf_durability.json").read_text())
    assert [s.state_id for s in driver.leaves] == [r["state_id"] for r in dur["leaves"]]
    assert all(s.validity.value == "measured" for s in driver.leaves)
    # the control was measured and admitted
    assert driver.control_state.evaluation is not None
    driver.control_state.require_recovery_admissible()
    # and the boundary ran before stage 1 reported success
    assert "device_handoff" in driver.ev.get("runtime", {})
    assert (tmp_path / "audit" / "stage1_import.json").is_file()
    assert (tmp_path / "audit" / "control_measurement.json").is_file()
    assert (tmp_path / "audit" / "device_handoff.json").is_file()
    assert recorded["kw"]["imported"] is True


def test_the_entrypoint_refuses_a_substituted_leaf(tmp_path, monkeypatch):
    """The same entrypoint, with one leaf's bytes swapped — it must not start."""
    store = Path("/home/ecs-user/aad-artifacts/autoinit/phase_a")
    if not store.is_dir():
        pytest.skip("the preserved leaves are not on this host")

    mod = load(DRIVER, "rc_driver_bad")
    dur = json.loads((mod.EVIDENCE / "selected_leaf_durability.json").read_text())
    ids = [r["state_id"] for r in dur["leaves"]]
    fake = tmp_path / "staged"; fake.mkdir()
    (fake / ids[0]).symlink_to(store / ids[1])          # symlink: costs nothing
    for sid in ids[1:]:
        (fake / sid).symlink_to(store / sid)

    monkeypatch.setattr(mod, "STAGED_LEAVES", fake)
    monkeypatch.setattr(mod, "AUDIT", tmp_path / "audit")
    (tmp_path / "audit").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "mark", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "say", lambda *_a, **_k: None)

    driver = object.__new__(mod.RecoveryContinuationDriver)
    driver.ev = {}
    driver.enter = lambda _s: None
    said = {}
    driver.record = lambda stage, ok, *a, **kw: (
        said.update({"ok": ok, "why": a[0] if a else ""}) or ok)

    assert driver.stage1() is False
    assert "did not verify" in said["why"]
