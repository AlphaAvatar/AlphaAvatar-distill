"""Detect that a generation has stopped producing new information.

A model stuck in a repetition loop will never emit EOS, so running it to the
full context buys no information beyond "it degenerated" — but the stop must be
justified by the *content*, never by a token budget (AGENTS.md P18). This module
supplies that justification and the evidence for it, so a degeneration stop is a
recorded measurement (`degeneration_detected`) and is never conflated with
natural termination or with a context-limit hit.

Two independent signals, both computed on the token ids actually emitted:

* **cycle** — the tail is a block of period `p` repeated `min_repeats` times.
  Catches the classic "same sentence forever" and "A B A B" loops.
* **low entropy of novelty** — over the trailing window, the ratio of distinct
  tokens is below `min_distinct_ratio`, i.e. the model is shuffling a tiny
  vocabulary and producing nothing new.

Both are deliberately conservative: they fire on structural repetition, not on
long-but-progressing reasoning, which is exactly the behaviour this project
wants to preserve.
"""
from __future__ import annotations


def find_cycle(tokens: list[int], max_period: int = 200,
               min_repeats: int = 4) -> dict | None:
    """Smallest period `p` whose block repeats `min_repeats` times at the tail."""
    n = len(tokens)
    for p in range(1, min(max_period, n // min_repeats) + 1):
        block = tokens[-p:]
        ok = True
        for r in range(2, min_repeats + 1):
            if tokens[-p * r:-p * (r - 1)] != block:
                ok = False
                break
        if ok:
            # Walk back to where the cycle actually started, for the record.
            reps = min_repeats
            while (reps + 1) * p <= n and tokens[-p * (reps + 1):-p * reps] == block:
                reps += 1
            return {"period": p, "repeats": reps, "start_index": n - p * reps}
    return None


def distinct_ratio(tokens: list[int], window: int = 800) -> float:
    tail = tokens[-window:]
    return (len(set(tail)) / len(tail)) if tail else 1.0


def check(tokens: list[int], *, min_tokens: int = 600, max_period: int = 200,
          min_repeats: int = 4, window: int = 800,
          min_distinct_ratio: float = 0.06) -> dict | None:
    """Return evidence if `tokens` has degenerated, else None.

    `min_tokens` keeps the detector off short, healthy generations: the teacher's
    own median natural completion is 727 tokens, so nothing is judged before it
    has had at least that much room to be legitimately long.
    """
    if len(tokens) < min_tokens:
        return None
    cycle = find_cycle(tokens, max_period=max_period, min_repeats=min_repeats)
    if cycle:
        return {"kind": "cycle", **cycle,
                "covered_tokens": cycle["period"] * cycle["repeats"]}
    ratio = distinct_ratio(tokens, window)
    if ratio < min_distinct_ratio:
        return {"kind": "low_novelty", "distinct_ratio": round(ratio, 4),
                "window": window}
    return None
