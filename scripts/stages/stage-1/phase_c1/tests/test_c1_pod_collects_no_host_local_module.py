"""phase_c1's launcher: the assertions that need it by name.

Split from `tests/integration/test_pod_script_paths.py` by the 2026-10-03 convergence round. The generic behaviour
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


def test_c1_runs_no_host_local_module_on_a_paid_pod():
    """Its premise is a retained byte store that is deliberately host-local, and
    running it on a pod cost three of attempt 4's six failures.

    It was named in a four-module exclusion list, then in a derived complement.
    Both pinned the mechanism. The property is that the module is NOT COLLECTED,
    which is stronger, covers every other host-local module at the same time,
    and survives the next change of declaration shape.
    """
    mod = load_session_launcher("autoinit_c1_launch")
    spec = mod.spec(session_args(mod))
    collected = gate_selection(spec)
    for host_local in ("test_phase_b_reuse_hostlocal.py", "test_stage1_import.py",
                       "test_recovery_continuation_session.py"):
        hits = [c for c in collected if c.endswith(host_local)]
        assert not hits, f"a C1 pod would collect {hits}"
    #: And the selection is C1's own suite, wherever that suite lives.
    assert mod.POD_TEST_SELECTION.endswith("phase_c1/tests")
    assert collected, "a gate that collects nothing is not a gate"
