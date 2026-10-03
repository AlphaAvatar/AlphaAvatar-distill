"""The mixtures C1's own fixed path names.

Split from `tests/initialization/test_calibration_item_preparation.py` by the
2026-10-03 convergence round. That module covers the real frozen mixtures and the
item conversion, which are reusable; this asks whether the ones it covers are the
ones C1 prepared -- otherwise that module could pass while C1's fixed path
resolves something nobody prepared.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.calibration import DOMAIN_BALANCED_V1, REASONING_HEAVY_V2  # noqa: E402

REAL_IDS = [p.qualified_id for p in (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)]


def test_the_c1_prefix_profiles_are_the_ones_this_covers():
    """The mixtures above are the mixtures C1's own path names.

    Otherwise this module could pass while the fixed path resolves something
    nobody prepared.
    """
    from experiments.phase_c1.session import INCUMBENT_ATTENTION, PREFIX_STEPS, TREATMENT_ATTENTION

    named = {p for _, p in (*PREFIX_STEPS, INCUMBENT_ATTENTION,
                            TREATMENT_ATTENTION)}
    assert named - {"calib.none@v1"} <= set(REAL_IDS)
