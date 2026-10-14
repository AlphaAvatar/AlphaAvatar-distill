"""D2 version 3's records at $0 that the stage-W tests do not own: the
real-logits engineering evidence and the programme's hold on the proposed
grant. The proposal, plan and stage are pinned in
`test_d2v3_gpu_validation.py`; the numerics in the core
(`tests/initialization/test_union_kl_objective.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]


def test_the_programme_holds_the_proposed_grant_until_it_closes():
    from stages.d_series import d2_budget as BUD
    from stages.phase_d2 import d2v3_gpu_validation_plan as P
    s = BUD.programme_scenarios(
        {"project_remaining_usd": 119.4879, "project_cap_usd": 590.0,
         "formal_remaining_usd": 127.9624,
         "engineering_remaining_usd": 3.6835, "per_session_cap_usd": 30.0})
    held = s["engineering"]["campaigns"]["d2v3_gpu_validation"]
    plan = P.build()
    assert held["closed"] is False
    assert held["grant_usd"] == plan["totals"]["proposed_grant_usd"]
    assert held["outstanding_usd"] == pytest.approx(held["grant_usd"])
    assert s["engineering_fits_allowance"] is True


def test_the_real_logits_record_shows_no_bound_violation():
    """The $0 CPU check on real logits, when its record is committed."""
    path = REPO / ("logs/stages/stage-1/families/d_series/analyses/"
                   "d2v3_real_logits_check.json")
    if not path.is_file():
        pytest.skip("the real-logits record is not committed")
    rec = json.loads(path.read_text())
    assert rec["status"].startswith("ENGINEERING EVIDENCE")
    assert rec["bounds"] == {"lower_violations_beyond_float32_slack": 0,
                             "upper_violations_beyond_float32_slack": 0}
    assert rec["top_k"] == 200
