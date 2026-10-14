"""D2 version 3's own records at $0: the GPU validation proposal and its
place in the programme.

The version-3 objective, support and preregistration are pinned in
`test_d2_search_chain.py` and the D-series objective tests; the numerics in
the core (`tests/initialization/test_union_kl_objective.py`). This file pins
what only the proposal says: that it is derived, minimal, reuses Stage V and
is held -- not spent -- by the programme.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[5]

from stages.phase_d2 import d2v3_gpu_validation_plan as P  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return json.loads((REPO / P.PLAN).read_text())


class TestTheProposal:

    def test_the_committed_plan_is_what_the_builder_derives(self):
        assert P.main([]) == 0, "run d2v3_gpu_validation_plan.py --write"

    def test_it_validates_version_3_and_authorizes_nothing(self, plan):
        from stages.d_series import objectives as O
        from stages.phase_d2 import d2_session as S
        assert plan["objective_hash"] == O.D2_OBJECTIVE.objective_hash
        assert plan["support"] == O.D2_SUPPORT.as_dict()
        assert plan["preregistration_hash"] == S.preregistration_hash(REPO)
        assert plan["status"].startswith("PROPOSAL")
        assert plan["authorizes"] == "nothing"
        assert not (REPO / P.VALIDATION_DIR / "authorization.json").exists()

    def test_it_is_minimal_and_reuses_stage_v(self, plan):
        ids = [s["id"] for s in plan["steps"]]
        assert not any("b_rebuild" in i for i in ids)
        assert "rebuild B" in plan["does_not"]
        assert plan["stage_v_reuse"]["run"] == "d2val_20261014_103301"
        assert any("union KL" in n
                   for n in plan["stage_v_reuse"]["not_reused"])
        gated = {k for k, v in plan["acceptance"].items()
                 if isinstance(v, dict) and v.get("gate")}
        assert gated == {"arithmetic", "integrity", "row_equivalence"}

    def test_its_bounds_come_from_stage_vs_measurements(self, plan):
        from stages.d_series import d2_budget as BUD
        v = BUD.adopted_validation()["record"]
        w4 = next(s for s in plan["steps"]
                  if s["id"] == "W4_depth_union_all_v1")
        assert w4["bound_minutes"] > \
            v["V4_depth_root_vs_parent"]["wall_seconds"] / 60
        totals = plan["totals"]
        assert totals["proposed_grant_usd"] == pytest.approx(
            totals["proposed_session_cap_usd"] + totals["teardown_reserve_usd"])
        assert totals["bound"]["all_in_usd"] <= totals[
            "proposed_session_cap_usd"]

    def test_the_programme_holds_its_grant_until_it_closes(self):
        from stages.d_series import d2_budget as BUD
        s = BUD.programme_scenarios(
            {"project_remaining_usd": 119.4879, "project_cap_usd": 590.0,
             "formal_remaining_usd": 127.9624,
             "engineering_remaining_usd": 3.6835,
             "per_session_cap_usd": 30.0})
        held = s["engineering"]["campaigns"]["d2v3_gpu_validation"]
        assert held["closed"] is False
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
