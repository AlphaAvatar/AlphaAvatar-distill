"""`aadistill.evaluation.isolation`: overlaps are counted per key and role pair,
and a required-disjoint key that the data violates refuses."""
from __future__ import annotations

import pytest

from aadistill.evaluation.isolation import (IsolationViolation, key_overlaps,
                                            require_disjoint)

KEYS = {"family": lambda r: r["f"], "content": lambda r: r["c"],
        "optional": lambda r: r.get("o")}


def test_overlaps_are_counted_per_key_and_none_is_ignored():
    rep = key_overlaps({"cal": [{"f": "x", "c": "1"}, {"f": "y", "c": "2"}],
                        "eval": [{"f": "x", "c": "3", "o": "z"}]}, KEYS)
    (pair,) = rep["pairs"]
    assert pair["overlap"] == {"family": 1, "content": 0, "optional": 0}
    assert pair["examples"]["family"] == ["x"]
    assert rep["roles"]["cal"]["optional"] == 0


def test_required_disjointness_refuses_a_violation_and_passes_otherwise():
    rep = key_overlaps({"a": [{"f": "x", "c": "1"}], "b": [{"f": "x", "c": "2"}]}, KEYS)
    assert require_disjoint(rep, [("a", "b", "content")])[0]["overlap"] == 0
    with pytest.raises(IsolationViolation):
        require_disjoint(rep, [("b", "a", "family")])
    with pytest.raises(IsolationViolation):
        require_disjoint(rep, [("a", "c", "family")])   # an unknown pair is not a pass
