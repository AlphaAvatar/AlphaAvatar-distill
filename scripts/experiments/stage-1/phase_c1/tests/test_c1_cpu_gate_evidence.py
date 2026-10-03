"""phase_c1's launcher: the assertions that need it by name.

Split from `tests/integration/test_cpu_test_outcome_evidence.py` by the 2026-10-03 convergence round. The generic behaviour
stays there; these drive phase_c1's own launcher, which makes them phase_c1's.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest  # noqa: F401

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "tests"))

from support.session_specs import load_session_launcher, session_args  # noqa: E402,F401

#: The shared setup script, whose pod invocation these read.
SETUP = REPO / "scripts/pod/autoinit_preflight_setup.sh"


def test_the_summary_survives_a_setup_abort():
    """A setup abort never reaches artifact collection, and the launcher's own
    window is `tail -40`. The file must be pulled while the pod still exists."""
    sys.path.insert(0, str(REPO / "tests/pod"))
    from support.session_specs import load_session_launcher, session_args
    mod = load_session_launcher("autoinit_c1_launch")
    spec = mod.spec(session_args(mod))
    assert "/workspace/pytest_outcomes.json" in spec.setup_failure_files

    runner = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    assert "_collect_setup_failure_evidence(target, draw)" in runner
    assert 'return "setup_failed"' in runner
    # Collected BEFORE the return that leads to teardown.
    assert runner.index("_collect_setup_failure_evidence(target, draw)") < \
        runner.index('return "setup_failed"')


def test_the_pod_invocation_resolves_this_session_without_being_told():
    """The wiring, end to end, on the shipped setup script and the real spec.

    Three things have to line up and none of them is asserted by the pieces
    above: the setup gate passes `--repo` so a repo-relative authorization path
    resolves; it does NOT override `--session-authorization`, so the environment
    default applies; and the launcher puts `SESSION_AUTH_PATH` into that
    environment pointing inside the run's own governance directory.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts/pod"))
    from support.session_specs import load_session_launcher, session_args

    text = SETUP.read_text()
    call = text[text.index("summarize_pytest_outcomes.py"):][:600]
    assert '--repo "$REPO"' in call, call
    assert "--session-authorization" not in call, (
        "the gate overrides the session default; if that is deliberate it must "
        "name the run's own record, not the repository-root pointer")

    launcher = load_session_launcher("autoinit_c1_launch")
    spec = launcher.spec(session_args(launcher))
    auth = spec.authorization_path
    assert auth.endswith("/governance/authorization.json"), auth
    assert not Path(auth).is_absolute(), (
        f"{auth} is absolute; --repo would not compose with it")


def test_the_raw_cpu_test_artifacts_are_retrieved_before_teardown():
    """A parser bug must not again be the only surviving evidence."""
    sys.path.insert(0, str(REPO / "tests/pod"))
    from support.session_specs import load_session_launcher, session_args
    mod = load_session_launcher("autoinit_c1_launch")
    files = mod.spec(session_args(mod)).setup_failure_files
    for raw in ("/workspace/pytest_outcomes.json", "/workspace/pytest_junit.xml",
                "/workspace/pytest.log"):
        assert raw in files, f"{raw} would not survive a setup abort"
    assert not any("token" in f.lower() or "credential" in f.lower() for f in files)
