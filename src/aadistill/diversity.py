"""Per-prompt divergence measures for adaptive top-n candidate selection.

The rule these serve (decision record 2026-07-28): keep many candidates where
the teacher's distribution is divergent and few where it is deterministic.
Keeping k near-identical candidates is not coverage — it is a k-fold
upweighting of that prompt carrying no extra distributional information.

Three measures, because they answer different questions and the choice between
them is deliberately left to pilot data rather than assumed:

* `mean_token_entropy` — the teacher's own per-token predictive entropy over the
  candidate. This is the most direct reading of "how divergent is the teacher
  here", since it measures the distribution itself rather than inferring it from
  samples, and it is free if generation logs scores. Its weakness is that it is
  a property of *one* candidate, so it says how uncertain the teacher was while
  writing, not how far apart the candidates ended up.
* `answer_agreement` — do the candidates reach the same conclusion? Needs no
  gold key, so it stays inside the Stage 3 / Stage 4-5 split: this is agreement
  among candidates, not correctness against a reference.
* `lexical_diversity` — mean pairwise distinctness of the full outputs. Captures
  reasoning-path variety that answer agreement throws away, which matters
  because the warm-up trains on the whole output, not just the answer.

Measured on raw traces almost every prompt looks divergent (wording and
exploration order vary even at a fixed conclusion); measured on answers alone,
path variety is discarded. Hence: compute all, decide later.
"""

from __future__ import annotations

import math
from collections import Counter


def mean_token_entropy(logprobs: list[list[float]]) -> float:
    """Mean per-position predictive entropy, in nats, from per-step logprobs.

    `logprobs[i]` is the full (or top-k) log-probability vector at position i.
    With a top-k slice the result is a lower bound on the true entropy, which is
    fine for *ranking* prompts by divergence as long as k is fixed across the
    corpus — the comparison is between prompts, not against an absolute scale.
    """
    if not logprobs:
        return 0.0
    total = 0.0
    for step in logprobs:
        total += -sum(math.exp(lp) * lp for lp in step)
    return total / len(logprobs)


def _ngrams(tokens: list[int], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _pair_distinctness(a: list[int], b: list[int], n: int = 3) -> float:
    """1 - n-gram overlap between two token sequences, in [0, 1].

    Symmetric, and defined on *tokens* rather than characters so it measures the
    sequence the model actually produced (see generate.py on token-in/token-out).
    """
    ga, gb = _ngrams(a, n), _ngrams(b, n)
    if not ga and not gb:
        return 0.0
    if not ga or not gb:
        return 1.0
    overlap = sum((ga & gb).values())
    total = max(sum(ga.values()), sum(gb.values()))
    return 1.0 - overlap / total


def lexical_diversity(candidates: list[list[int]], n: int = 3) -> float:
    """Mean pairwise n-gram distinctness across candidates, in [0, 1].

    0.0 means every candidate is a near-copy (deterministic prompt); values near
    1.0 mean they share almost no phrasing. A single candidate has no pairs and
    scores 0.0 — unmeasurable, treated as deterministic, which is the
    conservative choice because it keeps `n` low rather than inventing coverage.
    """
    if len(candidates) < 2:
        return 0.0
    scores = [
        _pair_distinctness(candidates[i], candidates[j], n)
        for i in range(len(candidates))
        for j in range(i + 1, len(candidates))
    ]
    return sum(scores) / len(scores)


def answer_agreement(answers: list[str | None]) -> float:
    """Fraction of candidates sharing the most common answer, in [0, 1].

    1.0 = unanimous (deterministic conclusion), low = the teacher disagrees with
    itself. Candidates whose answer could not be extracted are counted in the
    denominator but can never form the majority: a prompt where the teacher
    often fails to produce a parseable answer is *not* a prompt we should treat
    as confidently deterministic.
    """
    if not answers:
        return 0.0
    normalized = [a.strip().lower() if isinstance(a, str) and a.strip() else None
                  for a in answers]
    extracted = [a for a in normalized if a is not None]
    if not extracted:
        return 0.0
    top = Counter(extracted).most_common(1)[0][1]
    return top / len(normalized)


def choose_n(
    lexical: float,
    agreement: float,
    *,
    n_min: int = 1,
    n_max: int = 4,
    lexical_threshold: float = 0.5,
    agreement_threshold: float = 0.75,
) -> int:
    """Candidates worth keeping for one prompt, from its divergence measures.

    Deterministic on both axes (candidates say the same thing in the same words)
    collapses to `n_min`. Divergent on both goes to `n_max`. Disagreeing on one
    axis only lands in between — most often a prompt whose conclusion is stable
    but whose reasoning path varies, which is worth more than one sample because
    the warm-up trains on the path as well as the answer.

    Thresholds are defaults to be *fitted from pilot data*, not tuned constants:
    a slice whose prompts all collapse to one value means the threshold is
    wrong, not that the slice is uniform (decision record 2026-07-28).
    """
    if n_max < n_min:
        raise ValueError("n_max must be >= n_min")
    divergent_text = lexical >= lexical_threshold
    divergent_answer = agreement < agreement_threshold
    if divergent_text and divergent_answer:
        return n_max
    if divergent_text or divergent_answer:
        return max(n_min, min(n_max, (n_min + n_max) // 2))
    return n_min
