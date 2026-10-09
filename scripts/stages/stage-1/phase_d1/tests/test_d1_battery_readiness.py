"""D1's battery readiness is answerable from the canonical entry point.

The failure this regression pins: a reviewer landing on a stale historical
analysis (`d1_evidence_capacity.json`, written when the C1 pool was the only
battery source) and concluding D1 is blocked, while the realized D-series
family — the actual readiness authority — sits verified one directory over.
The capacity analysis now declares itself historical and names its
superseder; the experiment's `current.json` derives the readiness answers
from their owners. These tests hold all of that in place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CURRENT = REPO / "logs/stages/stage-1/phase_d1/current.json"
CAPACITY = REPO / "logs/stages/stage-1/phase_d1/analyses/d1_evidence_capacity.json"
FAMILY_CURRENT = REPO / "logs/stages/stage-1/families/d_series/current.json"


def _readiness() -> dict:
    return json.loads(CURRENT.read_text())["readiness"]


def test_the_seven_answers_a_reviewer_needs():
    r = _readiness()
    assert r["d_series_family"] == "d_series_behavioural_v1"
    assert r["family_status"] == "BUILT / VERIFIED"
    assert r["d1_screening_battery"] == "AVAILABLE"
    assert r["d1_confirmation_battery"] == "AVAILABLE"
    assert r["original_c1_pool_capacity"] == "EXHAUSTED, HISTORICAL ONLY"
    assert r["d1_evidence_blocker"] == "CLOSED"
    assert r["d1_screening"].startswith("PAUSED"), r["d1_screening"]


def test_the_answers_are_derived_not_retyped():
    """The committed record agrees with a live derivation, and every answer
    names its owner."""
    from stages.phase_d1.current import build

    live = build(REPO)["readiness"]
    assert live == _readiness()
    owners = live["_owners"]
    for key in ("family_status", "battery_roles", "pool_capacity",
                "blockers", "pause"):
        assert owners.get(key), f"readiness answer without an owner: {key}"


def test_the_capacity_analysis_cannot_reopen_readiness():
    """`d1_evidence_capacity.json` is a historical analysis of the EXHAUSTED
    C1 pool. If these fields ever regress, a reader that finds it first will
    treat a solved problem as a live blocker again."""
    cap = json.loads(CAPACITY.read_text())
    assert cap["record_role"] == "historical_analysis"
    assert cap["live_state"] is False
    assert cap["superseded_for_readiness_by"] == (
        "logs/stages/stage-1/families/d_series/current.json")
    # and the superseder actually answers.
    fam = json.loads(FAMILY_CURRENT.read_text())
    assert fam["status"] == "BUILT / VERIFIED"
    assert {"d1_screening", "d1_confirmation"} <= set(fam["roles"])


def test_the_four_gates_are_distinct():
    """Funding, one-use execution authorization, scientific blockers and the
    maintainer pause are four different questions; the record keeps them
    apart so no reader has to infer one from another."""
    gates = json.loads(CURRENT.read_text())["gates"]
    assert set(gates) >= {"formal_funding", "one_use_execution_authorization",
                          "open_scientific_blockers", "maintainer_pause"}
    assert gates["open_scientific_blockers"] == []
    assert gates["maintainer_pause"], "the pause gate must state its decision"
