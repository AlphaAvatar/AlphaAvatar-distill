"""Paired comparison statistics for arms scored on one fixed prompt set.

Every Stage 2/3 arm is evaluated on the *same* 150 examples, so the arms are
paired and the unpaired spread across prompts is the wrong noise model — it
ignores that a prompt hard for one arm is usually hard for the other. Pairing
removes that shared difficulty and asks only about the per-prompt *difference*.

Two things this module deliberately does not do:

* **It does not turn n=2 seeds into a seed-level inference.** A bootstrap here
  resamples *prompts*, so its interval describes sampling error from the
  150-example battery at a fixed pair of checkpoints. Seed-to-seed variation is
  a separate and larger source of uncertainty, reported alongside as the
  measured spread, never folded in.
* **It does not produce a p-value.** The project's rule is that an effect must
  clear a pre-registered noise floor; an interval that excludes zero on prompt
  resampling alone is not evidence that the recipe moved.

`mcnemar_counts` is the honest summary for binary paired outcomes: the only
prompts carrying information about a difference are the ones where the two arms
disagree.
"""

from __future__ import annotations

import random


def _aligned(a: dict, b: dict) -> tuple[list, list, list]:
    """Values for the ids present in both, in a fixed sorted order."""
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no shared prompt ids; the arms are not paired")
    return ids, [bool(a[i]) for i in ids], [bool(b[i]) for i in ids]


def mcnemar_counts(a: dict, b: dict) -> dict:
    """Discordant-pair census for two binary outcomes over matched prompt ids.

    `b_gained` are prompts b wins and a loses; `b_lost` the reverse. Concordant
    pairs carry no information about the difference and are reported only so the
    reader can see how few prompts the comparison actually rests on.
    """
    ids, va, vb = _aligned(a, b)
    gained = sum(y and not x for x, y in zip(va, vb))
    lost = sum(x and not y for x, y in zip(va, vb))
    return {
        "n_paired": len(ids),
        "b_gained": gained,
        "b_lost": lost,
        "net": gained - lost,
        "discordant": gained + lost,
        "both_true": sum(x and y for x, y in zip(va, vb)),
        "both_false": sum((not x) and (not y) for x, y in zip(va, vb)),
        "rate_a": round(sum(va) / len(ids), 4),
        "rate_b": round(sum(vb) / len(ids), 4),
        "delta": round((sum(vb) - sum(va)) / len(ids), 4),
    }


def paired_bootstrap_ci(a: dict, b: dict, *, iterations: int = 10000,
                        seed: int = 20260806, alpha: float = 0.05) -> dict:
    """Percentile CI for rate(b) − rate(a), resampling *prompts* with replacement.

    Deterministic given (a, b, iterations, seed): the same inputs always produce
    the same interval, so a reported CI can be recomputed exactly (P4/P5).
    """
    ids, va, vb = _aligned(a, b)
    n = len(ids)
    diffs = [y - x for x, y in zip(va, vb)]          # -1, 0 or +1 per prompt
    point = sum(diffs) / n
    rng = random.Random(seed)
    stats = []
    for _ in range(iterations):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        stats.append(total / n)
    stats.sort()
    lo = stats[int(alpha / 2 * iterations)]
    hi = stats[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return {
        "delta": round(point, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "level": 1 - alpha,
        "iterations": iterations,
        "seed": seed,
        "n_paired": n,
        "resamples": "prompts, at fixed checkpoints; NOT seeds",
    }


def joint_rate(correct: dict, terminated: dict) -> dict:
    """`correct_and_naturally_terminated`, plus the two marginals it comes from.

    A right answer that never stops is not a usable answer, and a clean stop with
    a wrong answer is not a useful one. This conjunction is the metric that
    refuses to trade one for the other; the marginals are reported beside it so
    the conjunction never hides which half moved.
    """
    ids = sorted(set(correct) & set(terminated))
    if not ids:
        raise ValueError("no shared prompt ids")
    both = {i: bool(correct[i]) and bool(terminated[i]) for i in ids}
    n = len(ids)
    return {
        "correct_and_naturally_terminated": round(sum(both.values()) / n, 4),
        "correct": round(sum(bool(correct[i]) for i in ids) / n, 4),
        "natural_termination": round(sum(bool(terminated[i]) for i in ids) / n, 4),
        "correct_but_unterminated": sum(
            bool(correct[i]) and not bool(terminated[i]) for i in ids),
        "n": n,
        "per_sample": both,
    }
