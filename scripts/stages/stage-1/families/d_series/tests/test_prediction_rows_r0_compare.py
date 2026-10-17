"""R0's equivalence comparison and launch wiring, at $0.

The comparison reads its tolerances from the registered plan, so these tests
build records that sit just inside and just outside them. The wiring tests
cover the three defects found while pricing R0: the launcher resolved the
repository two levels short, it funded itself from a closed campaign, and the
execution mode never reached the pod.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[6]
PHASE = REPO / "scripts/stages/stage-1/phase_d1"


def _module():
    spec = importlib.util.spec_from_file_location(
        "r0_compare", PHASE / "prediction_rows_r0_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _module()


@pytest.fixture(scope="module")
def plan(mod):
    return json.loads((REPO / mod.PLAN).read_text())


@pytest.fixture(scope="module")
def history(plan):
    return json.loads((REPO / plan["historical_oracle"]["record"]).read_text())


def _oracle(history, mod):
    """The historical record, shaped as R0's all_v1 run would be."""
    rec = copy.deepcopy(history)
    rec["prediction_rows"] = "all_v1"
    rec["status"] = "COMPLETE"
    for prof, row in rec["P_production_timing"]["per_profile"].items():
        row["round_tables"] = [[[[k], 0.5 + 0.01 * k] for k in range(4)]
                               for _ in row["removed"]]
    ranked = mod.ranked_objectives()
    rec["D_state_eval"] = {"arms": {arm: {
        "measurement_protocol_id": f"{arm}-all",
        "values": {**{k: 0.1 for k in ranked}, "state.nll.code": 1.0},
        "worst_domain": "code", "seconds": 10.0, "peak_memory_bytes": 2**30}
        for arm in ("full_vocab_v1", "reference_topk_tail_v1")}}
    return rec


def _treatment(oracle):
    rec = copy.deepcopy(oracle)
    rec["prediction_rows"] = "scored_v1"
    for key in ("bound_full_vocab", "bound_reference_topk_tail"):
        rec[key]["measurement_protocol_id"] += "-scored"
    for arm in rec["D_state_eval"]["arms"].values():
        arm["measurement_protocol_id"] += "-scored"
    return rec


def test_identical_measurements_pass(mod, plan, history):
    o = _oracle(history, mod)
    got = mod.compare(o, _treatment(o), history, plan)
    assert got["verdict"] == "PASS", got["failures"]


def test_a_different_removal_order_fails(mod, plan, history):
    o = _oracle(history, mod)
    t = _treatment(o)
    prof = next(iter(t["P_production_timing"]["per_profile"]))
    removed = t["P_production_timing"]["per_profile"][prof]["removed"]
    removed[0], removed[1] = removed[1], removed[0]
    got = mod.compare(o, t, history, plan)
    assert got["verdict"] == "FAIL"
    assert any("removal order" in f for f in got["failures"])


@pytest.mark.parametrize("scale,verdict", [(1 + 0.5e-5, "PASS"),
                                           (1 + 2e-5, "FAIL")])
def test_candidate_scores_are_held_to_the_registered_relative_bound(
        mod, plan, history, scale, verdict):
    o = _oracle(history, mod)
    t = _treatment(o)
    prof = next(iter(t["P_production_timing"]["per_profile"]))
    table = t["P_production_timing"]["per_profile"][prof]["round_tables"]
    table[0][0][1] *= scale
    assert mod.compare(o, t, history, plan)["verdict"] == verdict


@pytest.mark.parametrize("delta,verdict", [(0.9e-5, "PASS"), (1.1e-5, "FAIL")])
def test_ranked_state_metrics_are_held_to_the_absolute_budget(
        mod, plan, history, delta, verdict):
    o = _oracle(history, mod)
    t = _treatment(o)
    key = mod.ranked_objectives()[0]
    t["D_state_eval"]["arms"]["reference_topk_tail_v1"]["values"][key] += delta
    assert mod.compare(o, t, history, plan)["verdict"] == verdict


def test_an_unranked_value_is_reported_not_gated(mod, plan, history):
    o = _oracle(history, mod)
    t = _treatment(o)
    t["D_state_eval"]["arms"]["full_vocab_v1"]["values"]["state.nll.code"] += 1e-3
    got = mod.compare(o, t, history, plan)
    assert got["verdict"] == "PASS"
    assert got["state_eval"]["full_vocab_v1"][
        "max_absolute_drift_all_values"] == pytest.approx(1e-3)


def test_scored_rows_must_carry_their_own_protocol_identity(mod, plan, history):
    o = _oracle(history, mod)
    t = _treatment(o)
    t["bound_full_vocab"]["measurement_protocol_id"] = \
        o["bound_full_vocab"]["measurement_protocol_id"]
    got = mod.compare(o, t, history, plan)
    assert got["verdict"] == "FAIL"


def test_the_oracle_must_be_the_historical_path(mod, plan, history):
    o = _oracle(history, mod)
    t = _treatment(o)
    o2 = copy.deepcopy(o)
    o2["bound_reference_topk_tail"]["measurement_protocol_id"] = "elsewhere"
    assert mod.compare(o2, t, history, plan)["verdict"] == "FAIL"


# --- launch wiring ----------------------------------------------------------

LAUNCH = PHASE / "topk_adoption_launch.sh"
REMOTE = PHASE / "topk_adoption_remote.sh"


def test_the_launcher_resolves_the_repository_root():
    """Two levels up from phase_d1/ is scripts/stages, not the repository.

    The line is evaluated with `$0` set to the launcher's own path, which is
    what bash gives it when the launcher runs.
    """
    line = next(l for l in LAUNCH.read_text().splitlines()
                if l.startswith("REPO_DIR="))
    out = subprocess.run(["bash", "-c", line + '\necho "$REPO_DIR"',
                          str(LAUNCH)], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert Path(out) == REPO


def test_the_launcher_names_no_campaign_of_its_own():
    text = LAUNCH.read_text()
    assert 'CAMPAIGN="${ADOPT_CAMPAIGN:?' in text
    assert "topk-adoption/v1/campaign.json" not in text


def test_every_mode_reaches_the_driver_on_the_pod():
    launch, remote = LAUNCH.read_text(), REMOTE.read_text()
    assert "ADOPT_PREDICTION_ROWS='${MODES}'" in launch
    assert '--prediction-rows "$MODE"' in remote
    assert 'adoption_${MODE}' in remote


def test_the_r0_campaign_cannot_fund_a_session_before_a_grant():
    spec = importlib.util.spec_from_file_location(
        "budget", REPO / "scripts/shared/pod/engineering_campaign_budget.py")
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    campaign = REPO / ("logs/stages/stage-1/families/d_series/validations/"
                       "prediction-rows-r0/v1/campaign.json")
    doc = json.loads(campaign.read_text())
    auth = REPO / doc["authorization"]
    if doc["status"].startswith("AWAITING_GRANT"):
        #: Ungranted means NO authorization record, and therefore no money.
        assert not auth.exists()
        with pytest.raises(budget.BudgetError):
            budget.derive(campaign, session_cap_usd=3.6, rate_usd_per_hour=1.09)
