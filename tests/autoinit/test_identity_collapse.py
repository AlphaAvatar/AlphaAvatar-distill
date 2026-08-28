"""Identity collapse decides on bytes, or it refuses.

Phase-B attempt 5 completed its P=2 search and died in Stage 2 on
``duplicate seeds``. Two searched leaves were byte-identical to two imported
candidates, so a universe frozen as "8 distinct" was really 6, and two ids each
carried a cited historical `sa` and a fresh one at the same seed.

The dangerous version of this fix is a dedupe that reaches for anything
convenient — a name, an id prefix, a score. These tests pin the two properties
that make it safe: it collapses only when the **materialized identity** agrees,
and it **refuses** when the same id claims different bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.autoinit.identity_collapse import (  # noqa: E402
    IdentityCollapseError, ROLE_PRECEDENCE, collapse, observations_per_seed,
    universe_identity,
)

A, B = "a" * 32, "b" * 32


def entry(state_id, digest, role, **extra):
    return {"state_id": state_id, "artifact_digest": digest, "role": role, **extra}


def test_the_attempt_5_universe_collapses_from_eight_to_six():
    entries = [entry(f"leaf{n}", f"d{n}", "searched") for n in range(3)]
    entries += [entry(A, "dA", "searched"), entry(B, "dB", "searched")]
    entries += [entry(A, "dA", "imported_finalist"), entry(B, "dB", "imported_finalist")]
    entries += [entry("control", "dC", "control")]
    universe = collapse(entries)
    assert len(entries) == 8 and len(universe) == 6
    collapsed = [c for c in universe if c.is_collapsed]
    assert {c.state_id for c in collapsed} == {A, B}
    for c in collapsed:
        assert c.primary_role == "searched"
        assert c.roles == ("searched", "imported_finalist")


def test_same_id_different_bytes_is_a_REFUSAL_not_a_merge():
    with pytest.raises(IdentityCollapseError, match="broken identity"):
        collapse([entry(A, "dA", "searched"), entry(A, "SOMETHING_ELSE", "imported_finalist")])


def test_collapse_never_consults_a_score_or_a_name():
    """The property that lets an amendment be written after the collision."""
    import inspect

    from aadistill.autoinit import identity_collapse

    source = inspect.getsource(identity_collapse.collapse)
    for forbidden in ("correct_overall", "score", "rank", "path_label", "name"):
        assert forbidden not in source, forbidden


def test_the_searched_role_takes_precedence():
    assert ROLE_PRECEDENCE[0] == "searched"
    universe = collapse([entry(A, "dA", "imported_finalist"), entry(A, "dA", "searched")])
    assert universe[0].primary_role == "searched"


def test_a_missing_identity_field_is_refused_rather_than_guessed():
    for bad in ({"state_id": A, "role": "searched"},
                {"artifact_digest": "dA", "role": "searched"},
                {"state_id": A, "artifact_digest": "dA"}):
        with pytest.raises(IdentityCollapseError, match="missing"):
            collapse([bad])


def test_one_observation_per_initialization_and_seed():
    """The condition that killed Stage 2, as a rule rather than a crash."""
    universe = collapse([entry(A, "dA", "searched"), entry(A, "dA", "imported_finalist")])
    cited = {"state_id": A, "seed": 20260726, "student_artifact_digest": "dA"}
    out = observations_per_seed(universe, [cited, dict(cited)])
    assert len(out[A]) == 1, "the same seed was counted twice"


def test_two_observations_of_one_seed_that_DISAGREE_are_refused():
    universe = collapse([entry(A, "dA", "searched")])
    with pytest.raises(IdentityCollapseError, match="conflict"):
        observations_per_seed(universe, [
            {"state_id": A, "seed": 1, "student_artifact_digest": "dA"},
            {"state_id": A, "seed": 1, "student_artifact_digest": "dOTHER"}])


def test_a_record_outside_the_universe_is_refused():
    universe = collapse([entry(A, "dA", "searched")])
    with pytest.raises(IdentityCollapseError, match="outside"):
        observations_per_seed(universe, [{"state_id": B, "seed": 1}])


def test_the_universe_identity_is_order_independent():
    one = collapse([entry(A, "dA", "searched"), entry(B, "dB", "control")])
    two = collapse([entry(B, "dB", "control"), entry(A, "dA", "searched")])
    assert universe_identity(one) == universe_identity(two)
    three = collapse([entry(A, "dA", "searched"), entry(B, "dOTHER", "control")])
    assert universe_identity(one) != universe_identity(three)


def test_the_real_attempt_5_amendment_matches_this_implementation():
    """The recorded universe must be what the code produces today."""
    import json

    amendment = json.loads(
        (Path(__file__).resolve().parents[2]
         / "logs/autoinit_phase_b_identity_collapse_amendment.json").read_text())
    entries = [entry(c["state_id"], c["artifact_digest"], role)
               for c in amendment["collapsed_universe"]["candidates"]
               for role in c["roles"]]
    assert universe_identity(collapse(entries)) == \
        amendment["collapsed_universe"]["universe_identity"]
    assert amendment["collapsed_universe"]["distinct_candidates"] == 6
