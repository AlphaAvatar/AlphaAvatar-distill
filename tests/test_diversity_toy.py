"""Tests for the adaptive-n divergence measures (src/aadistill/diversity.py).

These are ordinary property tests — the measures are cheap arithmetic, so the
value here is pinning down the edge cases that would silently distort a corpus:
a single candidate, unparseable answers, and identical candidates.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aadistill.diversity import (
    answer_agreement,
    choose_n,
    lexical_diversity,
    mean_token_entropy,
)


# ---------- entropy ----------

def test_entropy_zero_for_a_certain_distribution():
    # log p = 0 for the taken token, -inf elsewhere -> entropy 0.
    certain = [[0.0, -60.0, -60.0]] * 4
    assert mean_token_entropy(certain) == pytest.approx(0.0, abs=1e-20)


def test_entropy_matches_uniform_analytically():
    k = 4
    uniform = [[math.log(1 / k)] * k] * 7
    assert mean_token_entropy(uniform) == pytest.approx(math.log(k), rel=1e-9)


def test_entropy_of_empty_is_zero():
    assert mean_token_entropy([]) == 0.0


# ---------- lexical diversity ----------

def test_identical_candidates_have_zero_diversity():
    c = [1, 2, 3, 4, 5, 6]
    assert lexical_diversity([c, c, c]) == pytest.approx(0.0)


def test_disjoint_candidates_have_full_diversity():
    a = [1, 2, 3, 4, 5]
    b = [90, 91, 92, 93, 94]
    assert lexical_diversity([a, b]) == pytest.approx(1.0)


def test_partial_overlap_is_between():
    a = [1, 2, 3, 4, 5, 6]
    b = [1, 2, 3, 40, 50, 60]
    d = lexical_diversity([a, b])
    assert 0.0 < d < 1.0


def test_single_candidate_is_treated_as_deterministic():
    """No pairs -> unmeasurable. Must score 0 (keep n low), never 1."""
    assert lexical_diversity([[1, 2, 3]]) == 0.0
    assert lexical_diversity([]) == 0.0


# ---------- answer agreement ----------

def test_unanimous_answers_score_one():
    assert answer_agreement(["42", "42", " 42 ", "42"]) == pytest.approx(1.0)


def test_case_and_whitespace_do_not_split_agreement():
    assert answer_agreement(["Paris", "paris ", " PARIS"]) == pytest.approx(1.0)


def test_full_disagreement_scores_low():
    assert answer_agreement(["1", "2", "3", "4"]) == pytest.approx(0.25)


def test_unextractable_answers_count_against_determinism():
    """Two parsed agreeing + two unparseable is NOT unanimous."""
    assert answer_agreement(["7", "7", None, ""]) == pytest.approx(0.5)
    # All unparseable cannot look deterministic.
    assert answer_agreement([None, None, None]) == 0.0


# ---------- the rule ----------

def test_deterministic_prompt_collapses_to_n_min():
    assert choose_n(lexical=0.05, agreement=1.0, n_min=1, n_max=4) == 1


def test_divergent_on_both_axes_takes_n_max():
    assert choose_n(lexical=0.9, agreement=0.3, n_min=1, n_max=4) == 4


def test_stable_answer_but_varied_path_lands_in_between():
    """Same conclusion, different reasoning: worth more than one sample,
    because the warm-up trains on the path as well as the answer."""
    n = choose_n(lexical=0.8, agreement=1.0, n_min=1, n_max=4)
    assert 1 < n < 4


def test_varied_answer_but_repeated_wording_lands_in_between():
    n = choose_n(lexical=0.1, agreement=0.4, n_min=1, n_max=4)
    assert 1 < n < 4


def test_result_always_within_bounds_and_bounds_validated():
    for lex in (0.0, 0.3, 0.5, 0.7, 1.0):
        for agr in (0.0, 0.5, 0.75, 1.0):
            n = choose_n(lexical=lex, agreement=agr, n_min=2, n_max=3)
            assert 2 <= n <= 3
    assert choose_n(lexical=0.9, agreement=0.1, n_min=2, n_max=2) == 2
    with pytest.raises(ValueError):
        choose_n(lexical=0.5, agreement=0.5, n_min=4, n_max=1)
