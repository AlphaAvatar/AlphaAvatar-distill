"""The experiment's configs and plans agree with each other and with their owners.

Nothing here re-derives science; it refuses the drift a hand edit would cause:
a stratum without a catalogued source, shares that do not sum, a plan naming a
different initialization than the selection, or a proposal restating money.
"""
from __future__ import annotations

import math

from stages.reasoning_data import assets as A

SPEC = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
SOURCES = {s["source_id"]: s for s in A.read_json(A.CONFIG_DIR / "sources.json")["sources"]}
PLANS = A.LOG_DIR / "plans"


def test_shares_sum_to_one_and_every_stratum_has_a_usable_source():
    assert math.isclose(sum(SPEC["shares"].values()), 1.0, abs_tol=1e-9)
    assert set(SPEC["shares"]) == set(SPEC["strata"]) == set(SPEC["generation"])
    for stratum, cfg in SPEC["strata"].items():
        for src in cfg["sources"]:
            assert src in SOURCES, (stratum, src)
            assert not SOURCES[src]["status"].startswith("excluded"), (stratum, src)


def test_rungs_increase_and_every_role_is_declared():
    assert SPEC["rung_targets"] == sorted(set(SPEC["rung_targets"]))
    roles = set(A.read_json(A.CONFIG_DIR / "sources.json")["roles"])
    assert {s["role"] for s in SOURCES.values()} <= roles


def test_the_design_uses_the_selected_initialization():
    sel = A.read_json(PLANS / "initialization_selection.json")["proposal"]
    design = A.read_json(PLANS / "comparison_design.json")
    assert design["fixed"]["initialization_checkpoint_sha256"] == sel["checkpoint_sha256"]


def test_the_proposal_cites_the_forecast_and_every_total_is_a_sum():
    prop = A.read_json(PLANS / "authorization_proposal.json")
    fc = A.read_json(PLANS / "cost_forecast.json")
    for row in prop["phases"][1:]:
        key = row["cost"].split("proposal_phases.")[1]
        assert key in fc["proposal_phases"]
        assert "expected_usd" not in row and "hard_cap_usd" not in row
    phases = fc["proposal_phases"].values()
    assert math.isclose(fc["proposal_totals"]["all_phases_expected_usd"],
                        sum(p["expected_usd"] for p in phases), abs_tol=0.011)


def test_nothing_here_authorizes_anything():
    for p in list(PLANS.glob("*.json")) + list(A.CONFIG_DIR.glob("*.json")):
        assert A.read_json(p)["authorizes"] == "nothing", p
