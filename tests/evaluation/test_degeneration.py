"""The degeneration detector decides when an unrestricted generation is stopped.

It is the only thing standing between a repetition loop and 262,144 tokens of
paid decoding, and a false positive would truncate legitimate long reasoning —
the exact failure AGENTS.md P18 exists to prevent. So both directions matter.
"""
from __future__ import annotations

import random
from aadistill.evaluation.degeneration import check, distinct_ratio, find_cycle


def rnd(n, seed=0, vocab=5000):
    r = random.Random(seed)
    return [r.randint(0, vocab) for _ in range(n)]


def test_fires_on_exact_repetition_cycles():
    for period in (1, 2, 10, 51):
        toks = list(range(period)) * (900 // max(period, 1) + 5)
        ev = check(toks)
        assert ev and ev["kind"] == "cycle", period
        assert ev["period"] == period
        assert ev["repeats"] >= 4


def test_reports_where_the_cycle_started():
    prefix = rnd(700, seed=1)
    toks = prefix + [7, 8, 9] * 60
    ev = check(toks)
    assert ev["kind"] == "cycle" and ev["period"] == 3
    assert ev["start_index"] >= len(prefix) - 3


def test_silent_on_long_progressing_output():
    """A model legitimately reasoning for thousands of tokens must not be cut."""
    for seed in range(5):
        assert check(rnd(3000, seed=seed)) is None


def test_silent_below_the_minimum_length():
    """Nothing is judged before 600 tokens — under the teacher's own median
    natural completion of 727 — so short answers are never classified."""
    assert check([1, 2] * 100) is None
    assert check(rnd(599, seed=3) ) is None


def test_low_novelty_catches_non_periodic_degeneration():
    r = random.Random(0)
    toks = rnd(700, seed=4) + [r.choice([11, 12, 13]) for _ in range(900)]
    ev = check(toks)
    assert ev and ev["kind"] in ("cycle", "low_novelty")


def test_distinct_ratio_and_find_cycle_directly():
    assert distinct_ratio([1] * 800) < 0.01
    assert distinct_ratio(rnd(800, seed=5)) > 0.5
    assert find_cycle([1, 2, 3] * 10) == {"period": 3, "repeats": 10, "start_index": 0}
    assert find_cycle(rnd(1000, seed=6)) is None
