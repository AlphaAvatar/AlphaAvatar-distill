"""The repository's permanent evaluation sets are intact and their scoring is unchanged.

`data/eval-set/` is read by formal measurement in every stage (AGENTS.md P10.3).
This is the check that makes an unauthorized change to it LOUD: any edit to an
item, a gold answer, a manifest or a scorer's verdict on the registered
conformance vectors fails here, and so in CI. A legitimate change appends a new
identity to the registry citing a maintainer authorization -- which this test
also checks exists as a record.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from aadistill.evaluation import eval_registry as ER
from shared import eval_sets as ES

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "logs/budget/decisions.md"

pytestmark = pytest.mark.skipif(not ES.registry_path().is_file(),
                                reason="no evaluation registry in this checkout")


def _sets():
    return [e["eval_set_id"] for e in ES.registry()["eval_sets"]]


def test_default_formal_set_is_registered():
    assert ES.default_formal() in _sets()


@pytest.mark.parametrize("set_id", _sets() if ES.registry_path().is_file() else [])
def test_registered_set_matches_its_authorized_identity(set_id):
    assert ER.verify(ES.registry_path(), set_id) == []


@pytest.mark.parametrize("set_id", _sets() if ES.registry_path().is_file() else [])
def test_every_identity_cites_an_existing_decision_record(set_id):
    headings = set(re.findall(r"^## (\d{4}-\d{2}-\d{2}) — (.+)$", DECISIONS.read_text(), re.M))
    for h in ER.entry(ES.registry(), set_id)["identity_history"]:
        m = re.match(r"logs/budget/decisions\.md (\d{4}-\d{2}-\d{2}) \((.+)\)$", h["authorization"])
        assert m, f"authorization must cite a decisions.md heading: {h['authorization']!r}"
        assert (m.group(1), m.group(2)) in headings, f"no such decision record: {m.groups()}"


@pytest.mark.parametrize("set_id", _sets() if ES.registry_path().is_file() else [])
def test_scoring_semantics_reproduce_the_conformance_vectors(set_id):
    pytest.importorskip("sympy")   # the math scorer degrades to string comparison without it
    from shared.evaluation.eval_set_scoring import conformance
    result = conformance(set_id)
    assert result, "no conformance vectors registered"
    bad = {c: r for c, r in result.items() if r["differences"]}
    assert not bad, bad


def test_held_out_units_are_reserved_by_the_corpus_builder():
    held = ES.held_out_paths()
    assert held and all((REPO / p).is_file() for p in held)
    assert not any("legacy_150" in p for p in held)     # in-sample units are training data
