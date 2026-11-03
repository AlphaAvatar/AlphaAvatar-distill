"""`mixture.validate_proportions`: a pure quota check on counts made before packing -- generic keys, the caller's
tolerances, no packing anywhere."""
from __future__ import annotations

from aadistill.data.mixture import validate_proportions

QUOTAS = {"a": {"target": 100, "below": 5, "above": 5}, "b": {"target": 100, "below": 0, "above": None},
          "all": {"of": ["a", "b"], "target": 200, "below": 0, "above": 2}}
SHARES = [{"name": "a_of_all", "part": ["a"], "whole": ["a", "b"], "target": 0.5, "max_deviation_points": 1.0}]


def test_within_every_bound_passes_with_its_rows():
    rep = validate_proportions({"a": 101, "b": 100}, QUOTAS, SHARES)
    assert rep["passed"] and not rep["failures"]
    assert rep["quotas"]["a"] == {"target": 100.0, "amount": 101.0, "deviation": 1.0, "deviation_fraction": 0.01,
                                  "bounds": [95.0, 105.0], "status": "OK"}
    assert rep["quotas"]["all"]["amount"] == 201.0 and rep["shares"]["a_of_all"]["ok"]


def test_a_shortfall_an_overshoot_and_a_share_each_fail_on_their_own_row():
    rep = validate_proportions({"a": 94, "b": 100}, QUOTAS, SHARES)
    assert rep["quotas"]["a"]["status"] == "SHORTFALL" and rep["quotas"]["all"]["status"] == "SHORTFALL"
    assert not rep["passed"]
    rep = validate_proportions({"a": 100, "b": 103}, QUOTAS, SHARES)
    assert rep["quotas"]["b"]["status"] == "OK" and rep["quotas"]["all"]["status"] == "OVER"   #: b's side is open
    rep = validate_proportions({"a": 105, "b": 95}, {"a": QUOTAS["a"], "b": {"target": 100, "below": 5, "above": 5}},
                               SHARES)
    assert rep["quotas"]["a"]["status"] == rep["quotas"]["b"]["status"] == "OK"
    assert not rep["shares"]["a_of_all"]["ok"] and abs(rep["shares"]["a_of_all"]["deviation_points"] - 2.5) < 1e-9
    assert not rep["passed"]


def test_a_surplus_elsewhere_never_covers_a_short_key():
    """The composite holds, the leaf does not: the leaf's row still fails."""
    rep = validate_proportions({"a": 80, "b": 120}, {**QUOTAS, "all": {"of": ["a", "b"], "target": 200, "below": 0,
                                                                        "above": 0}})
    assert rep["quotas"]["all"]["status"] == "OK" and rep["quotas"]["a"]["status"] == "SHORTFALL"
    assert not rep["passed"]


def test_an_unregistered_key_fails_and_nothing_has_a_default_tolerance():
    rep = validate_proportions({"a": 100, "b": 100, "c": 1}, QUOTAS)
    assert rep["unregistered"] == ["c"] and not rep["passed"]
    rep = validate_proportions({"a": 100.5}, {"a": {"target": 100, "below": 0, "above": 0}})
    assert rep["quotas"]["a"]["status"] == "OVER"                           #: exact means exact


def test_it_packs_nothing():
    """Its body imports nothing and names no packer: it can run where no packing code is loaded."""
    import inspect

    from aadistill.data import mixture
    body = inspect.getsource(mixture.validate_proportions).split('"""')[2]
    assert "pack" not in body and "import" not in body
