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


def test_ngram_containment_flags_a_reworded_copy_and_not_a_stranger():
    from aadistill.evaluation.isolation import ngram_containment, ngram_index

    held = ["Tom has 3 apples and buys 5 more; how many apples does Tom have now?"]
    index = ngram_index(held, 5)
    rep = ngram_containment(
        [("copy", "TOM has 3 apples and buys 5 more;  how many apples does Tom have now?"),
         ("reflow", "Q: Tom has 3 apples and buys 5 more; how many apples does Tom have"),
         ("other", "A train leaves the station at noon travelling sixty miles per hour.")],
        index, 5, threshold=0.5)
    flagged = {r["id"] for r in rep["flagged_items"]}
    assert flagged == {"copy", "reflow"} and rep["checked"] == 3
    assert rep["n"] == 5 and rep["threshold"] == 0.5


def test_a_text_shorter_than_n_collides_only_with_its_own_normal_form():
    from aadistill.evaluation.isolation import ngram_containment, ngram_index

    index = ngram_index(["Is 7 prime?"], 13)
    rep = ngram_containment([("same", "is 7   PRIME?"), ("near", "Is 9 prime?")], index, 13,
                            threshold=1.0)
    assert [r["id"] for r in rep["flagged_items"]] == ["same"]
