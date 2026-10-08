"""phase_a's launcher: the assertions that need it by name.

Split from `tests/integration/test_multi_repo_relay.py` by the 2026-10-03 convergence round. The generic behaviour
stays there; these drive phase_a's own launcher, which makes them phase_a's.
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
from shared.deployment import MAIN_RELAY  # noqa: E402


def test_the_ten_main_relay_science_inputs_are_unchanged(monkeypatch):
    """The whole point of a default: nothing that already worked moved."""
    sys.path.insert(0, str(REPO / "scripts/pod"))
    from support.session_specs import load_session_launcher, session_args

    mod = load_session_launcher("autoinit_phase_a_launch")
    spec = mod.spec(session_args(mod))
    inputs = spec.setup.relay_inputs
    assert len(inputs) == 10, f"expected the 10 science inputs, got {len(inputs)}"
    assert {r.repo for r in inputs} == {MAIN_RELAY}
    env = json.loads(spec.setup.relay_env())
    assert all(i["repo"] == MAIN_RELAY for i in env)
