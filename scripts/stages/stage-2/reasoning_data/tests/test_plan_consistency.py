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
        assert "expected_usd" not in row and "hard_cap_usd" not in row
        ref = row["cost"].split(":: ")[1]
        if ref.startswith(("proposal_phases.", "p1_application.")):
            block, key = ref.split(".", 1)
            assert key in fc[block]
        else:
            assert ref in fc
    p1 = fc["p1_application"]
    rows = [v for k, v in p1.items() if k not in ("totals", "_what")]
    for k in ("expected_usd", "hard_cap_usd"):
        assert math.isclose(p1["totals"][k], sum(r[k] for r in rows), abs_tol=0.011)
    phases = fc["proposal_phases"].values()
    assert math.isclose(fc["proposal_totals"]["h1_expected_usd"],
                        sum(p["expected_usd"] for p in phases), abs_tol=0.011)
    assert math.isclose(fc["proposal_totals"]["h1_hard_cap_usd"],
                        sum(p["hard_cap_usd"] for p in phases), abs_tol=0.011)


def test_the_fixed_initialization_is_materialized_and_verified():
    design = A.read_json(PLANS / "comparison_design.json")
    mat = A.read_json(A.LOG_DIR / "assets/d4_top1_materialization.json")
    assert mat["verified"] is True
    ident = mat["materialization_identity"]
    assert ident["checkpoint_sha256"] == design["fixed"]["initialization_checkpoint_sha256"]
    assert ident["artifact_digest"] == design["fixed"]["initialization_artifact_digest"]
    assert mat["read_back"]["identity"]["matched"] and mat["read_back"]["identity"]["weights_digest_matched"]


def test_the_budget_position_bounds_are_derived_from_its_own_categories():
    pos = A.read_json(A.LOG_DIR / "budget/h1_budget_position.json")
    cat, b = pos["categories"], pos["bounds"]
    assert math.isclose(b["headroom_if_records_are_complete_usd"],
                        b["project_cap_usd"] - cat["confirmed_project_spend_records_usd"], abs_tol=1e-4)
    assert math.isclose(b["headroom_if_all_provider_billing_is_project_usd"],
                        b["project_cap_usd"] - cat["provider_billed_all_time_usd"], abs_tol=1e-4)
    pods = pos["pods_no_run_record_created"]
    parts = ("named_by_untracked_project_launch_logs", "named_by_project_records_in_git_history", "unnamed")
    assert sum(pods[p]["count"] for p in parts) == pods["count"]
    assert math.isclose(sum(pods[p]["usd"] for p in parts), pods["usd"], abs_tol=0.001)


def test_nothing_here_authorizes_anything():
    for p in [*PLANS.glob("*.json"), *A.CONFIG_DIR.glob("*.json"),
              *(A.LOG_DIR / "assets").glob("*.json"), *(A.LOG_DIR / "budget").glob("*.json")]:
        assert A.read_json(p)["authorizes"] == "nothing", p


def test_the_source_allocation_splits_each_stratum_exactly_and_names_usable_sources():
    for stratum, alloc in SPEC["source_allocation"].items():
        assert math.isclose(sum(alloc.values()), SPEC["shares"][stratum], abs_tol=1e-9), stratum
        for src in alloc:
            assert src in SOURCES and not SOURCES[src]["status"].startswith("excluded"), (stratum, src)


def test_every_candidate_external_module_lands_in_a_declared_stratum():
    mods = A.read_json(A.CONFIG_DIR / "external_modules.json")["modules"]
    for m in mods:
        if str(m.get("decision", "")).startswith("candidate"):
            assert m["stratum"] in SPEC["shares"], m["module"]
            assert m["source"] in SPEC["source_allocation"][m["stratum"]], (m["source"], m["stratum"])


def test_the_committed_budget_position_attributes_none_of_the_pods_it_lists():
    """Committing a reconciliation must not change attribution: every pod the
    H1 budget position lists as no-run-record (OPEN included) is still named by
    no tracked file other than billing evidence and reports derived from it."""
    from maintenance.consolidation import provider_billing as PB
    pos = A.read_json(A.LOG_DIR / "budget/h1_budget_position.json")["pods_no_run_record_created"]
    listed = set(pos["unnamed"]["pods"])
    for k in ("named_by_untracked_project_launch_logs", "named_by_project_records_in_git_history"):
        listed |= set(pos[k]["pods"])
    assert len(listed) == pos["count"] and pos["unnamed"]["status"] == "OPEN"
    assert PB.mentioned(A.REPO, sorted(listed)) == set()


def test_the_p1_application_names_the_generation_plan_on_disk_and_the_forecast_prices_it():
    prop = A.read_json(PLANS / "authorization_proposal.json")
    sha = A.sha256_file(PLANS / "generation_plan.json")
    assert prop["generation_list"]["generation_plan_sha256"] == sha
    fc = A.read_json(PLANS / "cost_forecast.json")
    assert fc["generation"]["generation_plan_sha256"] == sha
    gp = A.read_json(PLANS / "generation_plan.json")
    assert fc["generation"]["generated_tokens"] == gp["totals"]["expected_generated_tokens_upper"]
    assert gp["corpus_id"] == SPEC["chosen_variant"]["id"] and not gp["shortfalls"]
