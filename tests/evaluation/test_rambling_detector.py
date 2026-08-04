"""The rambling signal: output that never repeats exactly and never terminates.

The first two signals (exact cycles, low token novelty) were tuned on
checkpoints that loop. A checkpoint that keeps minting new tokens while saying
nothing new slips past both, and on 2026-08-02 that cost an hour of L40S time
for a single evaluation wave. These tests pin the third signal and, just as
importantly, pin that it does NOT fire on long-but-progressing reasoning.
"""

from __future__ import annotations

import sys
from pathlib import Path


from aadistill.evaluation import degeneration  # noqa: E402


def progressing(n: int) -> list[int]:
    """A long generation that keeps introducing genuinely new content."""
    return list(range(n))


def circling(n: int, vocab: int = 400) -> list[int]:
    """Varied tokens, but the same phrasing re-treaded in a shuffled order.

    Deterministic and non-cyclic: the token sequence never repeats a fixed
    period, yet every 8-gram has been emitted before.
    """
    base = list(range(vocab))
    out: list[int] = []
    i = 0
    while len(out) < n:
        # rotate by a stride coprime with vocab: new token order, old n-grams
        out.extend(base[i:] + base[:i])
        i = (i + 1) % vocab
    return out[:n]


def test_progressing_generation_is_not_flagged():
    assert degeneration.check(progressing(6000)) is None


def test_rambling_is_flagged():
    ev = degeneration.check(circling(6000))
    assert ev is not None and ev["kind"] == "rambling", ev
    assert ev["novel_ngram_ratio"] < ev["threshold"]


def test_rambling_not_judged_before_its_floor():
    """Below `ramble_min_tokens` a long answer is not yet evidence of anything."""
    assert degeneration.check(circling(1500)) is None


def test_exact_cycle_still_wins_and_is_reported_as_cycle():
    toks = list(range(100)) + [7, 8, 9] * 400
    ev = degeneration.check(toks)
    assert ev is not None and ev["kind"] == "cycle", ev


def test_novel_ngram_ratio_bounds():
    assert degeneration.novel_ngram_ratio(progressing(4000)) == 1.0
    assert degeneration.novel_ngram_ratio(circling(6000)) < 0.15
    # too short to judge -> treated as fully novel rather than as evidence
    assert degeneration.novel_ngram_ratio(list(range(50))) == 1.0


def test_thresholds_are_fixed_not_per_checkpoint():
    """Same input, same verdict, regardless of call site — arms must be comparable."""
    toks = circling(6000)
    first = degeneration.check(toks)
    second = degeneration.check(list(toks))
    assert first == second
