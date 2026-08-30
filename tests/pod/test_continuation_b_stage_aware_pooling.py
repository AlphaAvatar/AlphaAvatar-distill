"""`sc` must not leak backward into the rung-2 decision.

Attempt 4 ran to `ALL_DONE` and reported `resolved / winner=fe9683e6a9c7`. The
decision was wrong, and the probe it bought was not.

**What happened.** The continuation IMPORTS completed evidence before stage 3,
and that evidence includes `85bde4ded2c3/sc` — a rung-3 record sitting in the
probe store while the rung-2 decision is being formed. The inherited
`pooled_over_rungs` pools *every completed rung*, which in Phase A was correct
because rung 3 could only exist after rung 2 had decided. Here it produced an
asymmetric comparison:

    fe9683e6a9c7   sa+sb        n=380   11/340 = 0.032353
    85bde4ded2c3   sa+sb+sc     n=570   10/510 = 0.019608     <- three rungs
    control        sa+sb        n=380    3/340 = 0.008824

The `0.012745` margin that "resolved" the session is not a quantity the frozen
rule is defined over. On `sa+sb` alone the same evidence gives `0.032353` against
`0.026471` — a margin of `0.005882`, **inside** the `0.011695` equivalence
interval, which is `tie_pending`.

These tests are built from the REAL retained attempt-4 journals, so the numbers
below are the measurement, not a fixture invented to match a conclusion.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

#: The attempt-4 evidence, exactly as retained.
PROBES = REPO / "logs/autoinit_continuation_b_attempt4/probes"
RESULT = REPO / "logs/autoinit_continuation_b_attempt4/phase_a_result.json"

FE = "fe9683e6a9c783bbc6fe276a78c851c6"
BD = "85bde4ded2c31953f802e39cf2252c87"
CTL = "control-qwen3_0p6b_init_v0"

#: The frozen equivalence interval, read from the artifact that owns it rather
#: than retyped. `2 * binomial_se` under the frozen
#: `seed_aware_max_binomial_seedrange` rule, whose dominant term is the binomial
#: one; the value is pinned below against the committed result so a typo here
#: cannot quietly widen or narrow the decision.
FROZEN_PLAN = REPO / "logs/autoinit_phase_a_recovery_plan_frozen.json"


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "continuation_b_driver_pool",
        REPO / "scripts/pod/autoinit_continuation_b_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continuation_b_driver_pool"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drv():
    return load_driver()


@pytest.fixture
def store(tmp_path, drv, monkeypatch):
    """A probe store containing the real attempt-4 journals — INCLUDING
    `85bde4ded2c3/sc`, which is the whole point."""
    audit = tmp_path / "audit"
    (audit / "probes").mkdir(parents=True)
    for p in PROBES.glob("*.json"):
        (audit / "probes" / p.name).write_text(p.read_text())
    monkeypatch.setattr(drv, "AUDIT", audit)
    names = {p.name for p in (audit / "probes").glob("*.json")}
    assert "autoinit.v1.phase_a.rung3.85bde4ded2c3.sc.json" in names, (
        "the fixture does not contain 85bde/sc; it cannot reproduce the defect")
    return audit


def make_driver(drv, rung2=None):
    d = drv.ContinuationDriver.__new__(drv.ContinuationDriver)
    d.rung1 = {"advancing": [CTL, FE, BD]}
    d.rung2 = rung2
    return d


def rows_by_state(rows):
    return {r["state_id"]: r for r in rows}


# --- 1 & 2: the rung-2 decision ignores a physically present sc --------------

def test_rung2_pooling_ignores_a_present_historical_sc(drv, store):
    d = make_driver(drv, rung2=None)
    rows = rows_by_state(drv.ContinuationDriver.pooled_over_rungs(d))

    assert set(rows) == {FE, BD, CTL}
    # Symmetric: every finalist contributes exactly its sa and sb.
    for sid in (FE, BD, CTL):
        assert rows[sid]["n"] == 380, (
            f"{sid[:12]} pooled n={rows[sid]['n']}, not the 380 of sa+sb — an "
            "asymmetric comparison is exactly the attempt-4 defect")
        assert rows[sid]["n_scorable"] == 340
        assert sorted(rows[sid]["seeds"]) == [20260726, 20260801]
        assert not any("rung3" in p for p in rows[sid]["probe_ids"])


# --- 3: the corrected numbers, and the decision they produce -----------------

def test_rung2_produces_the_corrected_values_and_tie_pending(drv, store):
    d = make_driver(drv, rung2=None)
    rows = rows_by_state(drv.ContinuationDriver.pooled_over_rungs(d))

    assert rows[FE]["correct"] == 11 and rows[FE]["n_scorable"] == 340
    assert rows[BD]["correct"] == 9 and rows[BD]["n_scorable"] == 340
    assert rows[CTL]["correct"] == 3 and rows[CTL]["n_scorable"] == 340

    assert rows[FE]["correct_overall"] == pytest.approx(0.032353, abs=5e-7)
    assert rows[BD]["correct_overall"] == pytest.approx(0.026471, abs=5e-7)
    assert rows[CTL]["correct_overall"] == pytest.approx(0.008824, abs=5e-7)

    margin = rows[FE]["correct_overall"] - rows[BD]["correct_overall"]
    assert margin == pytest.approx(0.005882, abs=5e-7)

    # The interval, from the artifact that owns it — and cross-checked against
    # the one the committed result recorded, so neither source can drift alone.
    interval = json.loads(RESULT.read_text())["equivalence_interval"]
    rule = json.loads(FROZEN_PLAN.read_text())
    rule = rule.get("plan", rule)["equivalence_rule"]
    assert interval == pytest.approx(2 * rule["binomial_se"], rel=1e-12), (
        "the recorded interval is not 2x the frozen binomial SE; the two "
        "sources of the equivalence interval have drifted")

    assert margin < interval, (
        f"margin {margin:.6f} is not inside the equivalence interval "
        f"{interval:.6f}; the corrected decision would not be tie_pending")
    # And the withdrawn margin WAS outside it, which is why attempt 4 resolved.
    assert 0.012745 > interval


def test_the_committed_attempt4_result_is_the_asymmetric_one(drv):
    """The defect, pinned against the artifact it produced.

    This is what the withdrawn result actually says. It is kept as a test so the
    record cannot quietly be re-read as a same-rung comparison.
    """
    committed = json.loads(RESULT.read_text())["final_selection"]["ranked"]
    by = {r["state_id"]: r for r in committed}
    assert by[BD]["n"] == 570 and len(by[BD]["probe_ids"]) == 3
    assert by[FE]["n"] == 380 and by[CTL]["n"] == 380
    assert committed[0]["state_id"] == FE
    assert json.loads(RESULT.read_text())["decision_status"] == "resolved"


# --- 4: the final stage admits sc only for tie candidates -------------------

def test_final_pooling_admits_sc_only_for_tie_candidates(drv, store):
    d = make_driver(drv, rung2={"tie_break_candidates": [FE, BD]})
    rows = rows_by_state(drv.ContinuationDriver.pooled_over_rungs(d))

    # 85bde has an sc and is a tie candidate, so it counts now.
    assert rows[BD]["n"] == 570 and sorted(rows[BD]["seeds"]) == [
        20260726, 20260801, 20260813]
    # fe9683 is a candidate but has no sc yet — the probe still to be bought.
    assert rows[FE]["n"] == 380
    # The control is NOT a tie candidate; it contributes no sc even if one existed.
    assert rows[CTL]["n"] == 380


def test_a_non_candidate_with_an_sc_is_still_excluded(drv, store):
    """The backward-leak guard, stated as its own property."""
    d = make_driver(drv, rung2={"tie_break_candidates": [FE]})
    rows = rows_by_state(drv.ContinuationDriver.pooled_over_rungs(d))
    assert rows[BD]["n"] == 380, (
        "85bde is not a tie candidate here, yet its sc was pooled")


def test_a_resolved_rung2_admits_no_sc_at_all(drv, store):
    d = make_driver(drv, rung2={"decision_status": "resolved"})
    rows = rows_by_state(drv.ContinuationDriver.pooled_over_rungs(d))
    for sid in (FE, BD, CTL):
        assert rows[sid]["n"] == 380


# --- 5: the missing observation is exactly fe9683/sc ------------------------

def test_exactly_one_sc_is_missing_and_it_is_fe9683(drv, store):
    have_sc = {json.loads(p.read_text())["state_id"]
               for p in (store / "probes").glob("*.json")
               if json.loads(p.read_text())["rung"] == 3}
    tied = {FE, BD}
    assert BD in have_sc, "85bde/sc must be reusable, not re-bought"
    assert sorted(tied - have_sc) == [FE], (
        "the missing tie-break observation is not exactly fe9683/sc")


def test_the_attempt4_sb_probe_is_strictly_reusable():
    """It must never be repurchased: its session's decision was withdrawn, the
    measurement was not."""
    rec = json.loads((REPO / "logs/autoinit_attempt4_probe_reuse.json").read_text())
    assert rec["reuse_verified"] is True
    assert rec["reusable_probes"] == ["fe9683e6a9c7/sb"]
    probe = rec["probes"][0]
    assert probe["seed"] == 20260801 and probe["rung"] == 2
    assert probe["correct"] == 3 and probe["n_scorable"] == 170
    assert all(probe["checks"].values()), probe["failed"]


# --- the survivor: the attempt-4 probe must actually be IMPORTED -------------
#
# The reuse record above proves the probe is citable. It does not prove the
# driver cites it. A mutation removing `ATTEMPT4_PROBES` from the import sources
# passed every other test in this file — and would silently re-buy ~72 minutes
# of L40S for a measurement already on disk.

def test_the_driver_actually_imports_the_attempt4_sb_probe(drv, tmp_path,
                                                           monkeypatch):
    """Drive the real import and require `fe9683e6a9c7/sb` to arrive."""
    audit = tmp_path / "audit"
    (audit / "probes").mkdir(parents=True)
    monkeypatch.setattr(drv, "AUDIT", audit)

    d = drv.ContinuationDriver.__new__(drv.ContinuationDriver)
    d.imported_probe_ids = set()
    report = drv.ContinuationDriver.import_completed_probes(d)

    sb = "autoinit.v1.phase_a.rung2.fe9683e6a9c7.sb"
    assert sb in report["imported"], (
        f"{sb} was not imported; the next session would re-buy the one probe "
        f"attempt 4 paid for. Imported: {report['imported']}")
    assert sb in d.imported_probe_ids
    assert (audit / "probes" / f"{sb}.json").is_file()

    # And the source it came from is declared, so removing it cannot be silent.
    assert drv.ATTEMPT4_PROBES.is_dir()
    assert drv.ATTEMPT4_REUSE.is_file()


def test_every_verified_reuse_record_is_wired_into_the_import(drv):
    """All three records, and the sources that satisfy them, checked together.

    Stated as one property rather than three so a fourth record cannot be added
    to `logs/` and left unimported.
    """
    import inspect

    src = inspect.getsource(drv.ContinuationDriver.import_completed_probes)
    for source in ("HISTORICAL_PROBES", "ATTEMPT5_PROBES", "ATTEMPT4_PROBES"):
        assert source in src, f"{source} is not among the import sources"
    for record in (drv.HISTORICAL_REUSE, drv.ATTEMPT5_REUSE, drv.ATTEMPT4_REUSE):
        assert record.is_file(), record
        assert json.loads(record.read_text())["reuse_verified"] is True, record
