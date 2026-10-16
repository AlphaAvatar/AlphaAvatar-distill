"""The screening-completion contract: a reservation is released by evidence only.

Every case starts from a COPY of the real D2 screening attempt that completed
(which satisfies the contract) and breaks one record, so each clause is shown
to bite on the field it claims to read.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[6]
RUNS = REPO / "logs/stages/stage-1/phase_d2/runs"
RUN = "d2_screening_20261015_203346"
BOOKED = 12.5212

from stages.d_series import screening_completion as SC  # noqa: E402


def _spec(booked=(BOOKED, "money.all_in_usd"), attempt_cost=None):
    from stages.phase_d2 import d2_screening as S
    from stages.phase_d2 import d2_screening_authorization as A

    contract = S.session_contract(REPO, verify_bytes=False)
    return SC.ScreeningCompletionSpec(
        experiment_id="phase_d2",
        closeout_schema="aadistill.phase_d2.screening_closeout/v1",
        contract_hash=S.contract_hash(contract),
        probe_ids=tuple(p["probe_id"] for p in contract["probes"]),
        load_authorization=A.D2ScreeningAuthorization.load,
        ranking_filename="d2_screening_ranking.json", disk_gb=120,
        booked_cost=lambda e, r: booked if r == RUN else None,
        attempt_cost=attempt_cost or A._subrun_cost)


@pytest.fixture
def run(tmp_path):
    """Copies of every screening attempt's records; the one that completed is
    returned (no out-of-tree bytes are needed)."""
    for d in sorted(RUNS.glob("d2_screening_*")):
        shutil.copytree(d, tmp_path / "runs" / d.name)
    return tmp_path / "runs" / RUN


def _edit(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc, indent=1) + "\n")


def _set(*keys, value):
    def fn(doc):
        node = doc
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return fn


def test_the_real_attempt_satisfies_every_clause(run):
    got = SC.assess(run, _spec())
    assert got["complete"] is True, got["problems"]
    assert got["booked_all_in_usd"] == BOOKED


def test_the_d2_budget_releases_the_real_rung_and_keeps_d3s_blocks():
    from stages.d_series import d2_budget as BUD

    done = BUD.completed_d2_screening()
    assert done["run_id"] == RUN and done["booked_all_in_usd"] == BOOKED
    budget = BUD.build()
    limits = {"project_remaining_usd": 100.0, "project_cap_usd": 595.0,
              "formal_remaining_usd": 100.0, "engineering_remaining_usd": 3.0,
              "per_session_cap_usd": 30.0}
    view = BUD.programme_scenarios(limits, budget)
    inf = view["search_pricing"][view["search_pricing_in_force"]]
    assert inf["d2_block"]["sessions"] == ["d2_confirmation"]
    #: D3 still prices its own screening AND confirmation.
    steps = budget["steps"]
    d3_behaviour = sum(steps[n]["hard"]["all_in_usd"] + BUD.TEARDOWN_RESERVE_USD
                       for n in ("d2_screening", "d2_confirmation"))
    assert inf["d3_block"]["hard_with_reserves"] > d3_behaviour


CLOSEOUT = "closeout/outcome.json"
RUNTIME = "runtime/session.json"
RANKING = "evidence/d2_screening_ranking.json"

MUTATIONS = {
    "authorized_identity": [
        (CLOSEOUT, _set("schema", value="aadistill.phase_d2.search_closeout/v1")),
        (CLOSEOUT, _set("run_id", value="d2_screening_other")),
        (CLOSEOUT, _set("authorization_sha256", value="0" * 64)),
        (CLOSEOUT, _set("contract_hash", value="0" * 64)),
        ("governance/authorization.json", _set("hard_cap_usd", value=99.0)),
    ],
    "terminal_complete": [
        (RUNTIME, _set("terminal", value="RUN_FAILED")),
        (RUNTIME, _set("passed", value=False)),
        (CLOSEOUT, _set("driver_status", value="FAILED")),
    ],
    "every_unit_verified": [
        (RANKING, _set("contract_hash", value="0" * 64)),
        (RANKING, lambda d: d["units_verified"].pop(
            sorted(d["units_verified"])[0])),
        (RANKING, lambda d: next(iter(d["units_verified"].values()))[
            "checks"].update(rescore_reproduces_the_result=False)),
        (RANKING, lambda d: next(iter(d["units_verified"].values())).update(
            aligned_with_B=False)),
        (RANKING, _set("selection", value={})),
    ],
    "products_secured": [
        (RUNTIME, _set("required_products_secured", "ok", value=False)),
        (RUNTIME, _set("teardown_gate", "emergency", value=True)),
        (RUNTIME, _set("artifact_loss_risk", value={"condition": "x"})),
    ],
    "provider_teardown": [
        (CLOSEOUT, _set("provider_teardown_verified", value=False)),
        (RUNTIME, _set("provider_confirms_gone", value=False)),
        (RUNTIME, _set("final_pod_state", "billing", value=True)),
    ],
    "cost_closed": [
        (CLOSEOUT, _set("money", "all_in_usd", value=0.0)),
        (CLOSEOUT, _set("money", "all_in_usd", value=12.0)),
        (CLOSEOUT, _set("money", "gpu_usd", value=1.0)),
        (CLOSEOUT, _set("money", "provider_balance_cross_check",
                        "measured_delta_usd", value=13.0)),
    ],
    "no_unsettled_resources": [
        (RUNTIME, _set("draws", value=[{"pod_id": "p", "release": {
            "confirmed_not_billing": False}}])),
        (RUNTIME, _set("blocked_teardown_ownership", value={"pod": "p"})),
    ],
}
CASES = [(clause, i) for clause, ms in MUTATIONS.items()
         for i in range(len(ms))]


@pytest.mark.parametrize("clause,i", CASES)
def test_each_clause_refuses_a_mutation_of_its_premise(run, clause, i):
    rel, fn = MUTATIONS[clause][i]
    _edit(run / rel, fn)
    got = SC.assess(run, _spec())
    assert got["complete"] is False
    assert clause in got["failed_clauses"], got["problems"]


def test_every_clause_is_mutated():
    assert set(MUTATIONS) == set(SC.CLAUSES)


def test_a_booking_under_another_shape_or_missing_keeps_the_reservation(run):
    assert "cost_closed" in SC.assess(
        run, _spec(booked=(BOOKED, "cost.actual_usd")))["failed_clauses"]
    assert "cost_closed" in SC.assess(run, _spec(booked=None))[
        "failed_clauses"]


def test_an_attempt_of_unknown_cost_keeps_the_reservation(run):
    (run.parent / "d2_screening_20261015_183412" / "closeout"
     / "outcome.json").unlink()
    got = SC.assess(run, _spec())
    assert "no_unsettled_resources" in got["failed_clauses"]


def test_a_contradiction_keeps_the_reservation(run):
    other = run.parent / "d2_screening_20261015_193502"
    _edit(other / CLOSEOUT, _set("classification", value="COMPLETE (claimed)"))
    status = SC.completion(run.parent, _spec(), pattern="d2_screening_*")
    assert status["complete"] is None and status["contradictions"]


def test_missing_records_never_raise(run):
    for rel in (CLOSEOUT, RUNTIME, RANKING):
        (run / rel).unlink()
    got = SC.assess(run, _spec())
    assert got["complete"] is False and len(got["failed_clauses"]) >= 5
