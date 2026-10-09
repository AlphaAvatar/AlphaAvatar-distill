"""The owner index cannot contradict an experiment's own readiness.

The 2026-10-09 review found `logs/stages/index.json` describing D1 as
"blocked on evidence and funding" and citing the exhausted ORIGINAL C1 prompt
pool as the first of three blockers — a month after the realized D-series
family became the battery source and closed that blocker, and in direct
contradiction of D1's `current.json`.

The division these tests hold: the index owns STABLE ownership and
navigation; readiness is DYNAMIC and has exactly one owner, the experiment's
`current.json`. A reader starting at the index must be routed there, and must
not be able to read the exhausted pool as a live constraint.

Each test starts where a reviewer starts — the index — and follows only what
the index itself provides.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))

INDEX = REPO / "logs/stages/index.json"


@pytest.fixture(scope="module")
def d1_row() -> dict:
    rows = json.loads(INDEX.read_text())["experiments"]
    row = next((r for r in rows if r["experiment_id"] == "phase_d1"), None)
    assert row is not None, "the owner index does not carry phase_d1"
    return row


def test_the_index_routes_readiness_rather_than_restating_it(d1_row):
    """Following the index's own `live_state` leg must land on the readiness
    owner, and the index's status must not claim a readiness of its own."""
    live = d1_row["live_state"]
    assert live == "logs/stages/stage-1/phase_d1/current.json"
    assert (REPO / live).is_file()

    status = d1_row["status"].lower()
    for claim in ("blocked on", "blocker", "authorized", "funded",
                  "exhausted", "paused"):
        assert claim not in status, (
            f"the index's status claims readiness ({claim!r}: "
            f"{d1_row['status']!r}); readiness belongs to {live}")


def test_from_the_index_alone_the_exhausted_pool_is_not_a_live_blocker(d1_row):
    """The walk a reviewer actually does: index -> evidence -> the record.

    Every evidence entry the index offers must either declare itself
    historical or agree that the blocker is closed. Nothing reachable in one
    step may present the exhausted C1 pool as an active D1 battery blocker.
    """
    capacity_seen = False
    for ev in d1_row["evidence"]:
        rel = ev["path"]
        if "d1_evidence_capacity" not in rel:
            continue
        capacity_seen = True
        doc = json.loads((REPO / rel).read_text())
        assert doc["record_role"] == "historical_analysis"
        assert doc["live_state"] is False
        assert doc["superseded_for_readiness_by"] == (
            "logs/stages/stage-1/families/d_series/current.json")
        # and the index's own assertion about it pins that, not a count
        assert ev.get("field") == "live_state" and ev.get("equals") is False, (
            "the index asserts something other than this record's historical "
            "status; a `batteries_remaining == 0` assertion reads as a live "
            "constraint, which is the defect this test exists for")
        assert "blocker" not in ev["says"].lower() or "historical" in ev["says"].lower()
    assert capacity_seen, (
        "the capacity analysis is no longer cited by the index; if it was "
        "removed rather than marked historical, this test must be replaced "
        "by one that checks whatever took its place")


def test_the_index_also_offers_the_actual_readiness_authority(d1_row):
    """Marking the old record historical is only half the repair: the index
    must name the realized family that superseded it."""
    paths = [ev["path"] for ev in d1_row["evidence"]]
    manifest = ("logs/stages/stage-1/families/d_series/analyses/"
                "autoinit_d_series_family_manifest.json")
    assert manifest in paths, paths
    fam = json.loads((REPO / manifest).read_text())
    assert fam["family_id"] == "d_series_behavioural_v1"
    assert len(fam["roles"]) == 6 and len(fam["output_files"]) == 42


def test_the_index_and_current_json_agree_about_readiness(d1_row):
    cur = json.loads((REPO / d1_row["live_state"]).read_text())
    r = cur["readiness"]
    assert r["d1_evidence_blocker"] == "CLOSED"
    assert r["d1_screening_battery"] == "AVAILABLE"
    assert r["original_c1_pool_capacity"] == "EXHAUSTED, HISTORICAL ONLY"
    # the index's stable status and the live readiness must not contradict:
    # the index says DESIGNED, the live record says why it is not running.
    assert d1_row["status"].startswith("DESIGNED")
    assert r["d1_screening"].startswith("PAUSED")


def test_the_four_gates_stay_separate_in_the_live_record(d1_row):
    """Formal funding, one-use execution authorization, open scientific
    blockers and the maintainer pause are four questions. The index's old
    status fused two of them ("blocked on evidence and funding") and a reader
    could not tell which was true."""
    gates = json.loads((REPO / d1_row["live_state"]).read_text())["gates"]
    assert set(gates) >= {"formal_funding", "one_use_execution_authorization",
                          "open_scientific_blockers", "maintainer_pause"}
    assert gates["open_scientific_blockers"] == []
    assert re.search(r"review", gates["maintainer_pause"], re.I), (
        "the pause gate must say what the pause is waiting for")
