"""Phase B's repricing, and the reuse verification it is conditional on.

The point of the cross-phase procedure is that Phase A's probes are *evidence*,
not something to buy again. But "evidence" has to be earned: the first version of
this pricing treated a probe as reusable because its **file existed**, which is
the absence of the check rather than the check. Reuse now comes from the strict
reconstruction record, and pricing fails closed without it.

This module runs **everywhere**, including inside the pod's setup gate. The
strict reconstruction itself — which reads the dev-box retained checkpoint store
— lives in `test_phase_b_reuse_hostlocal.py`, because a pod has no such store and
was never meant to. Nothing here may reach for those bytes; if a test needs
`verify()`, it belongs in the host-local module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

import price_phase_b  # noqa: E402
from price_phase_b import (  # noqa: E402
    CONTROL, PHASE_A_FINALISTS, PHASE_B_SEARCHED_LEAVES, SURVIVORS_AT_SB,
    observed_probes, price,
)
from verify_historical_probe_reuse import ADMITTED, CHECKPOINTS  # noqa: E402


# --- the historical evidence ------------------------------------------------


def test_the_historical_inventory_is_read_off_disk_not_assumed():
    seen = observed_probes()
    assert sum(len(v) for v in seen.values()) == 11, seen
    for finalist in PHASE_A_FINALISTS:
        assert seen[finalist] == {"sa", "sb", "sc"}, finalist
    # The fact that drives the whole sc worst case, and the easiest to get wrong.
    assert seen[CONTROL] == {"sa", "sb"}, "the control has no sc on record"


def test_all_five_leaves_are_declared_so_unlooked_is_not_reported_as_unverified():
    """'We did not look' and 'it does not check out' must not print the same."""
    assert set(CHECKPOINTS) >= set(ADMITTED)
    assert len(CHECKPOINTS) == 6, "five searched leaves plus the control"
    unadmitted = set(CHECKPOINTS) - set(ADMITTED)
    assert unadmitted == {"158b96cf651f", "281a02c3ac18", "4e429f7ed722"}


# --- the pricing fails closed -----------------------------------------------


def test_pricing_refuses_when_the_reuse_record_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(price_phase_b, "REUSE_RECORD", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="missing"):
        price()


def test_pricing_refuses_an_unverified_reuse_record(monkeypatch, tmp_path):
    bad = tmp_path / "reuse.json"
    bad.write_text(json.dumps({"reuse_verified": False, "failures": ["x"]}))
    monkeypatch.setattr(price_phase_b, "REUSE_RECORD", bad)
    with pytest.raises(SystemExit, match="reuse_verified=false"):
        price()


def test_pricing_refuses_a_reuse_record_describing_different_probe_bytes(
        monkeypatch, tmp_path):
    """The staleness guard: verified once, then the evidence changed."""
    stale = tmp_path / "reuse.json"
    stale.write_text(json.dumps({
        "reuse_verified": True, "probes_dir_digest": "0" * 64,
        "admitted_reusable_probes": [], "open_precondition": {}}))
    monkeypatch.setattr(price_phase_b, "REUSE_RECORD", stale)
    with pytest.raises(SystemExit, match="re-run the verifier"):
        price()


# --- the arithmetic ---------------------------------------------------------


def test_reuse_removes_the_verified_priors_from_the_sa_bill():
    p = price()["probes"]
    assert p["sa_missing"] == PHASE_B_SEARCHED_LEAVES
    assert price()["procedure"]["candidates_at_sa"]["total"] == \
        PHASE_B_SEARCHED_LEAVES + len(PHASE_A_FINALISTS) + 1


def test_each_scenario_describes_ONE_world():
    """The trap: pricing sb as if the survivors were new while pricing sc as if
    they were the priors. Each end must be internally consistent."""
    p = price()["probes"]
    assert p["sb_missing_high"] == SURVIVORS_AT_SB
    assert p["sc_missing_high"] == SURVIVORS_AT_SB + 1
    assert p["sc_missing_high"] > p["sb_missing_high"], (
        "the control has no verified sc but does have sb, so sc must cost one "
        "more than sb in the same world")
    assert p["sb_missing_low"] == 0 and p["sc_missing_low"] == 0
    assert (p["total_low"], p["total_high"], p["total_no_reuse"]) == (5, 10, 14)


def test_the_no_reuse_path_is_a_REJECTED_counterfactual_not_the_ceiling():
    """Comparability failure terminates at stage 0; it does not buy a bigger run.

    The frozen feasibility floor and equivalence interval were materialized under
    Phase A's runtime. If comparability fails they describe nothing this session
    could produce, so re-running eight candidates is a differently-thresholded
    experiment, not a fallback — and funding it would be funding that experiment.
    """
    r = price()
    p, t = r["probes"], r["total"]
    n = PHASE_B_SEARCHED_LEAVES + len(PHASE_A_FINALISTS) + 1
    assert p["no_reuse_sa"] == n
    assert p["total_no_reuse"] > p["total_high"] > p["total_low"]
    # It is priced, and it is NOT the ceiling.
    assert t["authorization_ceiling_usd"] == t["hard_with_reuse_usd"]
    assert t["rejected_counterfactual_no_reuse_usd"] > t["authorization_ceiling_usd"]
    assert "TERMINATE" in r["comparability_gate"]["on_fail"]
    assert "rerunning" in r["comparability_gate"]["not_a_fallback"]
    assert "NOT an executable path" in r["scenarios"]["rejected_counterfactual_no_reuse"]


def test_the_floor_is_not_called_an_expectation():
    """No expected-value assumption over survivor identity or tie-break
    probability is defined anywhere, so nothing here may be called 'expected'."""
    r = price()
    assert set(r["total"]) == {"low_usd", "hard_with_reuse_usd",
                               "authorization_ceiling_usd",
                               "rejected_counterfactual_no_reuse_usd", "note"}
    assert "expected_usd" not in r["total"]
    assert "FLOOR, not an expectation" in r["total"]["note"]
    assert "not an expectation" in r["scenarios"]["low"]


def test_the_totals_are_ordered_and_composed_of_what_they_claim():
    r = price()
    lo = r["total"]["low_usd"]
    hi = r["total"]["hard_with_reuse_usd"]
    nr = r["total"]["rejected_counterfactual_no_reuse_usd"]
    assert 0 < lo < hi < nr
    p, s = r["probes"], r["search"]
    assert abs(lo - (s["usd_low"] + p["total_low"] * p["per_probe_usd"]
                     + r["setup_reserve_usd"])) < 0.01
    # The CEILING composes from the search's conservative bound, not its averaged
    # projection. Attempt 3 spent 9.08 h in a search whose `usd_high` claimed
    # 7.51 h and did not finish, so an average is not something to authorize.
    assert abs(hi - (s["usd_hard"] + p["total_high"] * p["per_probe_usd"]
                     + r["setup_reserve_usd"])) < 0.01
    assert abs(nr - (s["usd_hard"] + p["total_no_reuse"] * p["per_probe_usd"]
                     + r["setup_reserve_usd"])) < 0.01
    assert s["usd_hard"] > s["usd_high"] > s["usd_low"]


def test_it_prices_a_two_profile_search_not_a_one_profile_one():
    s = price()["search"]
    assert s["leaves_max"] >= 20
    assert s["peak_storage_gib_working"] > 200
    assert s["provision_container_disk_gib"] >= 300


def test_the_record_does_not_claim_to_be_an_authorization():
    r = price()
    assert "NOT an authorization" in r["_contract"]
    assert "grant" in r["_contract"]
    assert r["reuse"]["verified"] is True
    assert r["reuse"]["source"].endswith("autoinit_historical_probe_reuse.json")


# --- the three holes a first mutation pass found ----------------------------


def test_a_probe_that_EXISTS_but_is_not_verified_is_still_billed(monkeypatch, tmp_path):
    """M1: reuse must come from the verification verdict, not from the file.

    Today every admitted probe both exists and verifies, so the two sources agree
    and a test comparing them proves nothing. This forces them apart: a record
    that withholds one admitted probe must raise the bill by exactly one probe.
    """
    from verify_historical_probe_reuse import probes_dir_digest

    full = price()["probes"]["total_low"]
    real = json.loads((REPO / "logs/autoinit_historical_probe_reuse.json").read_text())
    withheld = [p for p in real["admitted_reusable_probes"]
                if p != f"{PHASE_A_FINALISTS[0]}/sa"]
    doctored = tmp_path / "reuse.json"
    doctored.write_text(json.dumps({
        "reuse_verified": True,
        "probes_dir_digest": probes_dir_digest(),
        "admitted_reusable_probes": withheld,
        "verifiable_but_not_admitted": [],
        "open_precondition": real["open_precondition"],
    }))
    monkeypatch.setattr(price_phase_b, "REUSE_RECORD", doctored)
    assert price()["probes"]["total_low"] == full + 1, (
        "withholding a verified sa probe did not raise the bill; the pricing is "
        "reading file existence rather than the verification verdict")


# M5 (bytes that disagree are not reusable) and M6 (a changed scoring contract
# invalidates reuse) both drive `verify()` against the retained store and now
# live in `test_phase_b_reuse_hostlocal.py`.
