"""The setup contract C2 declares is the setup the pod will actually run.

This is the regression that would have caught attempt 2, and the gap it closes
is specific. The 23-test C2 preflight models the pod's **CPU test gate**: it
proves the staged inputs resolve and the seams import. Nothing proved the
**setup contract** was satisfiable — that the sections the shared shell runs
before that gate are the sections C2 declared, and that the questions they ask
are questions C2 can answer.

So attempt 2 passed eight `$0` gates, created a pod, reached `TRAIN_ENV`, and
was refused at `SETUP_RC=91` for a recovery corpus it neither stages nor needs
and a scoring contract from a source set it does not execute. `$0.0552`, no
driver stage, nothing measured.

Two rules, asked of the real declaration and the real shell:

* an optional setup section runs **only** when the session declared its marker,
  through one rule in `aadistill.runtime.setup_steps` that both sides read;
* a session declaring `ASSETS_READY` must name its own frozen-asset
  expectation, because the verifier's fallback is another experiment's
  historical constants.

No shell emulator. The step-selection rule is executed for real — it is a
module, precisely so it can be — and the shell is read to confirm that it is
what the shell asks and that no gated section escapes its guard.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _root in ("src", "scripts", "scripts/pod", "tests/pod"):
    if str(REPO / _root) not in sys.path:
        sys.path.insert(0, str(REPO / _root))

from session_specs import load_session_launcher, session_args  # noqa: E402

SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"
#: The sections that are a session's choice. `ENV_READY`, `REPO_READY` and
#: `TRAIN_ENV` are the substrate every session's script produces and are not
#: gated — `SetupManifest.SUBSTRATE_MARKERS` refuses a declaration that omits
#: them, so a session cannot quietly disown them either.
GATED = ("ASSETS_READY", "VLLM_READY", "TEACHER_READY", "ROPE_OK", "TESTS_OK",
         "AUTHORIZATION_OK")


@pytest.fixture(scope="module")
def c2():
    launcher = load_session_launcher("autoinit_phase_c2_launch")
    args = session_args(launcher)
    spec = launcher.spec(args).validate()
    return launcher, args, spec, spec.setup_environment(
        session_commit="0" * 40, bundle="aad_autoinit_00000000.bundle")


# --- what C2 declares -------------------------------------------------------

def test_c2_declares_assets_ready_and_names_its_own_expectation(c2):
    launcher, _args, _spec, env = c2
    markers = env["SESSION_SETUP_MARKERS"].split()
    assert "ASSETS_READY" in markers
    assert env["SESSION_FROZEN_EXPECT"] == launcher.FROZEN_EXPECT
    assert env["SESSION_FROZEN_EXPECT"] == (
        "configs/experiments/phase_c2/frozen_assets.json")
    assert (REPO / env["SESSION_FROZEN_EXPECT"]).is_file()
    #: And the setup script is told, so it cannot fall back.
    assert "SESSION_FROZEN_EXPECT" in _spec.setup.required_env
    assert "SESSION_SETUP_MARKERS" in _spec.setup.required_env


def test_c2_does_not_declare_vllm_and_every_other_session_does(c2):
    """The one asymmetry that makes gating worth anything — and the audit that
    gating no other session weakens, done by reading their declarations."""
    _launcher, _args, _spec, env = c2
    assert "VLLM_READY" not in env["SESSION_SETUP_MARKERS"].split(), (
        "C2 Search-1 never calls vLLM: it measures candidates with the training "
        "stack and generates nothing")

    from session_specs import SESSION_LAUNCHERS

    declaring, silent = [], []
    for name, extra in SESSION_LAUNCHERS:
        other = load_session_launcher(name)
        markers = tuple(other.spec(session_args(other, extra)).setup.setup_markers)
        (declaring if markers else silent).append((name, markers))

    #: THE AUDIT. Every session that declares markers declares ALL the gated
    #: ones, so making the declaration an execution contract weakens none of
    #: them: each still runs exactly the sections it ran before.
    for name, markers in declaring:
        missing = [m for m in GATED if m not in markers]
        assert not missing, (
            f"{name} declares markers but not {missing}, which are now gated — "
            "it would silently lose those steps")

    #: And a session that declares NOTHING is refused, loudly, rather than
    #: silently skipping every optional step. Three launchers are in that
    #: position, and every one of them is a CLOSED experiment whose
    #: authorization is consumed:
    #:
    #:   autoinit_phase_a_launch        Phase A: COMPLETE 2026-08-24, $12.8587
    #:   autoinit_continuation_launch   the behavioural continuation, closed
    #:   autoinit_device_canary_launch  the paid canary path: TERMINATED
    #:                                  2026-08-18, evidence kept
    #:
    #: None of them is made WEAKER — none loses a check it used to run. Each is
    #: made inoperable until it declares, by `SESSION_SETUP_MARKERS:?` failing
    #: with a message that names exactly what is missing. Inoperable-and-loud is
    #: the safe direction; silently-weaker is the defect being repaired.
    #:
    #: Pinned as a set so a NEW session cannot quietly join it. A session added
    #: tomorrow that declares nothing fails here, on the dev box, instead of on
    #: a pod at its first `step_declared`.
    assert sorted(n for n, _ in silent) == [
        "autoinit_continuation_launch",
        "autoinit_device_canary_launch",
        "autoinit_phase_a_launch",
    ], ("the set of launchers declaring no setup markers has changed: "
        f"{sorted(n for n, _ in silent)}. Every one must be a closed "
        "experiment, because the shell refuses an empty declaration — declare "
        "the markers or confirm the session is closed")
    body = SETUP.read_text()
    assert 'SESSION_SETUP_MARKERS:?' in body, (
        "the shell no longer refuses an empty declaration, so a session that "
        "declares nothing would skip every optional step")


def test_the_expectation_verifies_the_staged_state_eval_and_nothing_else(c2):
    """What C2 actually consumes: the suite every candidate is ranked on."""
    _launcher, _args, _spec, env = c2
    doc = json.loads((REPO / env["SESSION_FROZEN_EXPECT"]).read_text())

    assert set(doc["assets"]) == {"state_eval_v1"}
    for absent in ("recovery_search_v2", "recovery_search_v1",
                   "recovery_search"):
        assert absent not in doc["assets"], (
            f"{absent} is the asset attempt 2 was refused for")
    assert "scoring_contract" not in doc, (
        "C2 trains nothing and scores no battery; a declared contract would "
        "make the verifier digest a source set this session does not execute, "
        "which is the other half of the attempt-2 refusal")

    #: DERIVED, not transcribed: the same three identities the project pins in
    #: one canonical place. A drifted copy would be a second source of truth.
    import importlib.util

    loader = importlib.util.spec_from_file_location(
        "_vfa", REPO / "scripts/autoinit/verify_frozen_assets.py")
    vfa = importlib.util.module_from_spec(loader)
    sys.modules["_vfa"] = vfa
    loader.loader.exec_module(vfa)
    assert doc["assets"]["state_eval_v1"] == vfa.FROZEN["state_eval_v1"], (
        "the C2 expectation has drifted from the canonical state_eval_v1 pin")


def test_the_expectation_verifies_against_this_tree_for_real(c2):
    """The check whose absence cost the attempt, run the way the pod runs it.

    `state_eval_v1` is staged as a whole tree, so the bytes verified here are
    the bytes the pod receives.
    """
    _launcher, _args, _spec, env = c2
    with_out = REPO / "artifacts/audit/_c2_setup_contract_check.json"
    try:
        done = subprocess.run(
            [sys.executable, "scripts/autoinit/verify_frozen_assets.py",
             "--expect", env["SESSION_FROZEN_EXPECT"],
             "--out", str(with_out.relative_to(REPO))],
            cwd=REPO, capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTHONPATH": "src:scripts"})
        assert done.returncode == 0, done.stdout + done.stderr
        report = json.loads(with_out.read_text())
    finally:
        with_out.unlink(missing_ok=True)
    assert report["passed"] is True, report["problems"]
    assert report["scoring_contract"] == {
        "checked": False,
        "why": ("the expectation document declares no scoring_contract, so "
                "this session consumes none. Absence is a declaration, not "
                "a fallback to the historical contract."),
    }
    checks = report["assets"]["state_eval_v1"]["checks"]
    assert {"content_sha256", "manifest_sha256", "items_sha256"} <= set(checks)
    assert all(c["match"] for c in checks.values())


# --- what the shell does with it --------------------------------------------

def test_the_step_rule_answers_declared_not_declared_and_unusable():
    """One rule, executed. The shell asks this module, so a test can too."""
    from aadistill.runtime import setup_steps as steps

    c2_markers = ("ENV_READY REPO_READY ASSETS_STAGED TRAIN_ENV ASSETS_READY "
                  "TEACHER_READY ROPE_OK TESTS_OK AUTHORIZATION_OK SETUP_DONE")
    assert steps.requires(c2_markers, "ASSETS_READY") is True
    assert steps.requires(c2_markers, "VLLM_READY") is False
    #: Case and separator tolerant on the question, not on the declaration.
    assert steps.requires(c2_markers, "vllm_ready") is False
    assert steps.requires("A,B  C", "b") is True
    assert steps.parse("A B A") == ("A", "B")

    env = {steps.DECLARATION_ENV: c2_markers}
    assert steps.main(["ASSETS_READY"], env) == steps.DECLARED
    assert steps.main(["VLLM_READY"], env) == steps.NOT_DECLARED
    #: FAIL CLOSED. An absent or empty declaration is not permission to skip
    #: every optional step.
    assert steps.main(["VLLM_READY"], {}) == steps.UNUSABLE
    assert steps.main(["VLLM_READY"], {steps.DECLARATION_ENV: "  "}) == steps.UNUSABLE
    assert steps.main([], env) == steps.UNUSABLE

    #: AND THE CODES CANNOT COLLIDE WITH AN INTERPRETER FAILURE. `1` and `2` are
    #: reserved: a python that cannot run the module exits 1 with a traceback,
    #: runpy and argparse exit 2. `NOT_DECLARED` was 1, and it was measured
    #: reading a SyntaxError as "skip this step" — which on a pod would have
    #: silently skipped the frozen-asset gate, the test gate and the
    #: authorization check.
    assert steps.DECLARED == 0
    assert steps.NOT_DECLARED not in (1, 2)
    assert steps.UNUSABLE not in (1, 2)
    assert len({steps.DECLARED, steps.NOT_DECLARED, steps.UNUSABLE}) == 3


def test_an_interpreter_that_cannot_run_the_rule_aborts_rather_than_skipping():
    """The failure this design exists to make impossible, executed for real.

    A broken module exits 1. The shell's wrapper maps 0 and 3; everything else
    aborts the setup. If 1 meant "not declared", a module that would not parse
    on the pod's interpreter would skip every optional section and the pod would
    look like it had passed them.
    """
    import tempfile

    body = SETUP.read_text()
    guard = body[body.index("step_declared() {"):body.index("\n}", body.index(
        "step_declared() {"))]
    assert "3) say" in guard, "the wrapper no longer maps 3 to 'skip'"
    assert re.search(r"\*\) say .*UNUSABLE", guard), (
        "the wrapper has no catch-all abort, so an interpreter failure would "
        "fall through")
    assert re.search(r"^\s*1\)", guard, re.M) is None, (
        "the wrapper maps 1, which is what a crashing interpreter returns")

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "aadistill/runtime"
        pkg.mkdir(parents=True)
        (pkg.parent / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")
        (pkg / "setup_steps.py").write_text("this is not python(\n")
        done = subprocess.run(
            [sys.executable, "-m", "aadistill.runtime.setup_steps", "TESTS_OK"],
            cwd=tmp, capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONPATH": tmp,
                 "SESSION_SETUP_MARKERS": "ENV_READY REPO_READY TRAIN_ENV TESTS_OK"})
    from aadistill.runtime import setup_steps as steps

    assert done.returncode not in (steps.DECLARED, steps.NOT_DECLARED), (
        "a module that cannot even be parsed returned a code the shell reads "
        "as an answer")


def test_the_rule_parses_on_an_old_interpreter():
    """The shell asks it with whatever `python3` the pod image ships, before any
    project venv exists. A file that will not parse there is indistinguishable
    from an undeclared step, so the syntax stays conservative — this was
    measured failing on 3.6 with `from __future__ import annotations`."""
    import ast

    src = (REPO / "src/aadistill/runtime/setup_steps.py").read_text()
    tree = ast.parse(src)
    #: From the TREE, not the text: the module's docstring NAMES the syntax it
    #: avoids, in order to say why. A substring check reads that prose as the
    #: thing it warns about — the same mistake as grepping for a name and
    #: matching the comment explaining why the name is there.
    futures = [n for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    assert not futures, (
        "a __future__ import makes this unparseable on older interpreters")
    for node in ast.walk(tree):
        #: `X | Y` in an annotation needs 3.10 without the future import.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            raise AssertionError("PEP 604 union syntax in setup_steps")
    #: And it imports only the standard library, available everywhere.
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.Import)}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom)}
    assert imported <= {"os", "sys"}, imported


def test_the_step_rule_runs_as_the_shell_invokes_it():
    """The exact command line in the script, executed as a subprocess."""
    body = SETUP.read_text()
    assert "python3 -m aadistill.runtime.setup_steps" in body, (
        "the shell no longer asks the shared rule; a second rule in shell would "
        "be the one nobody can test")
    from aadistill.runtime import setup_steps as steps

    for marker, code in (("ASSETS_READY", steps.DECLARED),
                         ("VLLM_READY", steps.NOT_DECLARED)):
        done = subprocess.run(
            [sys.executable, "-m", "aadistill.runtime.setup_steps", marker],
            cwd=REPO, capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONPATH": "src",
                 "SESSION_SETUP_MARKERS": "ENV_READY REPO_READY TRAIN_ENV "
                                          "ASSETS_READY TESTS_OK"})
        assert done.returncode == code, (marker, done.returncode, done.stderr)


def test_every_gated_section_is_guarded_and_the_guard_is_balanced():
    """No gated section escapes, and the shell still parses.

    Read from the script rather than emulated: a `bash` interpreter written in
    a test is a second implementation of the thing under test.
    """
    body = SETUP.read_text()
    for marker in GATED:
        assert f"if step_declared {marker}; then" in body, (
            f"the {marker} section is unconditional again")
        assert re.search(rf"^mark {marker}", body, re.M) or \
            re.search(rf'^mark "{marker}', body, re.M), marker
    #: Balanced: one `fi` per guard, and bash agrees the file parses.
    assert body.count("if step_declared ") == len(GATED)
    syntax = subprocess.run(["bash", "-n", str(SETUP)], capture_output=True,
                            text=True, timeout=120)
    assert syntax.returncode == 0, syntax.stderr

    #: And the vLLM section really is inside the guard — the marker is not
    #: enough, because a guard around the marker alone would still install it.
    start = body.index("if step_declared VLLM_READY; then")
    end = body.index("mark VLLM_READY")
    section = body[start:end]
    for spent in ("requirements-vllm.txt", "/opt/vllm"):
        assert spent in section, (
            f"{spent} is outside the VLLM_READY guard, so C2 would still pay "
            "for the environment it does not use")


def test_a_session_declaring_assets_ready_cannot_inherit_the_fallback():
    """The refusal that replaces the fallback, read from the script."""
    body = SETUP.read_text()
    guard = body.index("if step_declared ASSETS_READY; then")
    verifier = body.index("verify_frozen_assets.py")
    required = body.index('SESSION_FROZEN_EXPECT:?')
    assert guard < required < verifier, (
        "the expectation must be REQUIRED after the marker guard and before "
        "the verifier runs, or a session can still reach the historical "
        "constants")


def test_the_substrate_markers_cannot_be_disowned():
    """Gating is not a licence to declare nothing."""
    from aadistill.infrastructure.session import SessionSpecError, SetupManifest

    ok = SetupManifest(setup_markers=("ENV_READY", "REPO_READY", "TRAIN_ENV",
                                      "TESTS_OK"))
    assert ok.setup_markers_env() == "ENV_READY REPO_READY TRAIN_ENV TESTS_OK"
    #: A session may decline to declare markers at all — every launcher before
    #: 2026-09-16 did, and the shell refuses an empty declaration rather than
    #: skipping everything.
    assert SetupManifest().setup_markers_env() == ""
    with pytest.raises(SessionSpecError, match="omits"):
        SetupManifest(setup_markers=("REPO_READY", "TESTS_OK")).setup_markers_env()


# --- the two things this must not have changed ------------------------------

def test_the_c2_test_selection_and_readiness_source_are_unchanged(c2):
    launcher, _args, spec, env = c2
    assert launcher.POD_TEST_SELECTION == "tests/c2_preflight"
    assert "--ignore=tests/c2_preflight" not in env["SESSION_TEST_IGNORES"]
    #: The paid pod still runs exactly the preflight directory.
    from aadistill.runtime.staging_contract import ignores_for_selection

    assert tuple(spec.setup.test_ignores) == ignores_for_selection(
        "tests/c2_preflight", REPO)
    #: And readiness is still compared against the RUN-OWNED record.
    from experiments.phase_c2 import pod_environment as PE

    assert PE.record_path_for("attempt3", "1").endswith(
        "runs/attempt3/governance/readiness.json")
    assert PE.c2_record_contract("attempt3", "1").record_path == (
        PE.record_path_for("attempt3", "1"))
