"""phase_c1's launcher: the assertions that need it by name.

Split from `tests/integration/test_session_architecture.py` by the 2026-10-03 convergence round. The generic behaviour
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


def test_c1_stages_the_canonical_rope_config_specifically():
    """C1's own binding, by hash: the object verified against the relay."""
    mod = load_session_launcher("autoinit_c1_launch")
    spec = mod.spec(session_args(mod, ()))
    staged = {r.path: r for r in spec.setup.staged_relay_inputs()}
    want = "stage1/qwen3_0p6b_init_v0/checkpoint/config.json"
    assert want in staged, sorted(staged)
    assert staged[want].sha256 == (
        "a7131bb092b38a078edc213961f0eb57eaead24f1396e25741f4887b1a694054")
    assert staged[want].dest == "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
    #: and NOT the weights, which the RoPE gate never reads
    assert not [p for p in staged if p.endswith("model.safetensors")]
    assert not [p for p in staged if p.endswith("generation_config.json")]
