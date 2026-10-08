"""The C3 pilot must not build a step the causal-KL operator will refuse.

Split from `tests/initialization/test_attention_causal_kl.py` by the 2026-10-03
convergence round. The operator's own refusal of a non-integer batch size — and
the executable check that a future edit cannot reintroduce the coercion — stay in
core, because they are properties of a reusable operator. This asks whether C3's
pilot constructor agrees with it, which is a question about C3's caller.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def test_the_pilot_step_constructor_refuses_the_same_values():
    from experiments.phase_c3.pilot import causal_step

    for bad in ("4", 4.0, True, None):
        with pytest.raises((TypeError, ValueError)):
            causal_step(bad)
