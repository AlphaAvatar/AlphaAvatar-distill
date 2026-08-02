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


def novel_ngram_ratio(tokens: list[int], n: int = 8, window: int = 512) -> float:
    """Fraction of trailing-window n-grams never seen earlier in this generation.

    The third failure mode, and the one the first two miss: output that keeps
    emitting *new tokens* and never terminates, while recycling the same phrasing
    in a different order. There is no exact token cycle to find and the distinct
    ratio stays high, because the vocabulary really is varied — but nothing new
    is being said.

    Measured on n-grams rather than tokens, and against everything generated so
    far rather than a fixed window, because that is what separates "circling"
    from "long but progressing": genuine reasoning keeps minting n-grams it has
    not used before, whereas rambling re-treads them.
    """
    if len(tokens) < n + window:
        return 1.0
    tail = tokens[-window:]
    head = tokens[:-window]
    seen = {tuple(head[i:i + n]) for i in range(len(head) - n + 1)}
    grams = [tuple(tail[i:i + n]) for i in range(len(tail) - n + 1)]
    if not grams:
        return 1.0
    return sum(1 for g in grams if g not in seen) / len(grams)


def check(tokens: list[int], *, min_tokens: int = 600, max_period: int = 200,
          min_repeats: int = 4, window: int = 800,
          min_distinct_ratio: float = 0.06,
          ramble_ngram: int = 8, ramble_window: int = 512,
          ramble_min_tokens: int = 2048,
          min_novel_ngram_ratio: float = 0.15) -> dict | None:
    """Return evidence if `tokens` has degenerated, else None.

    `min_tokens` keeps the detector off short, healthy generations: the teacher's
    own median natural completion is 727 tokens, so nothing is judged before it
    has had at least that much room to be legitimately long.

    The thresholds are **fixed** and applied identically to every checkpoint —
    a detector tuned per arm would make the arms incomparable.
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
    # Non-repeating, non-terminating rambling. Held to a later floor than the
    # other two signals (`ramble_min_tokens`), because a long, genuinely
    # progressing answer must not be cut off for being long — by 2,048 tokens a
    # generation is already well past the teacher's p99 of 3,854/median 727 in
    # spirit, and still saying nothing new is the evidence, not the length.
    if len(tokens) >= ramble_min_tokens:
        novel = novel_ngram_ratio(tokens, n=ramble_ngram, window=ramble_window)
        if novel < min_novel_ngram_ratio:
            return {"kind": "rambling", "novel_ngram_ratio": round(novel, 4),
                    "ngram": ramble_ngram, "window": ramble_window,
                    "threshold": min_novel_ngram_ratio}
    return None
