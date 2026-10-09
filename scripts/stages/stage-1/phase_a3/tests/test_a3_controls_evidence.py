"""A3's controls gate, against the REAL durable evidence. Dev box only.

This lived in `tests/a3_preflight`, which became A3's pod test selection --
and its premise is a dev-box fact, so it cost a pod to discover. attempt75's
control evidence lives at an ABSOLUTE path outside the repository
(`~/aad-artifacts/...`, named by the committed control record), so the pod
simulator cannot hide it and the pod never receives it: the test RAN in the
launch-bound sweep and SKIPPED on the pod, the skip sets differed by exactly
one nodeid, and the shared setup refused at `SETUP_RC=1` with
`50 passed, 5 skipped, 0 failed`.

The refusal was right. A pod that did not run the suite the sweep certified
must not then train three probes under an environment whose difference from
the rehearsal is unnamed. What was wrong was the test's location: a check
whose subject is the dev box belongs with the other dev-box launcher checks,
not in the directory that models a pod.

The gate itself is unchanged and still runs on the pod as part of the
launcher's own pre-provider sequence -- this file is about the TEST, not about
the gate's coverage.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "pod"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location(
        "a3lau_controls", REPO / "scripts/stages/stage-1/phase_a3/autoinit_a3_launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["a3lau_controls"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_controls_evidence_gate_passes_on_the_real_evidence(launcher):
    """A3's whole saving is that it does not retrain the controls, so a
    session that trains three probes and then finds nothing to compare against
    has produced half an experiment at full price."""
    args = launcher.build_parser().parse_args(
        ["--scr", "/tmp/a3-controls", "--run-id", "a3_attempt1",
         "--session-commit", "0" * 40, "--bundle", "aad_test.bundle",
         "--max-price", "1.09"])
    args.disk_gb = 60
    ctx = types.SimpleNamespace(
        args=args, evidence={}, say=lambda *_: None,
        auth=types.SimpleNamespace(hard_cap_usd=8.2525,
                                   harness_source_digest="x" * 64,
                                   harness_source_files=()))
    ok, why = launcher.controls_evidence_gate(ctx)
    if not ok and "absent" in why:
        pytest.skip("attempt75's durable control evidence is not mounted here")
    assert ok, why
    ev = ctx.evidence["controls_evidence"]
    assert len(ev["seeds"]) == 3
    assert ev["files_verified"] == 6, ev
    assert ev["problems"] == []
