"""Prefix/continuation splitting, shared by Experiment 5's C and R arms.

E5 asks whether training on **student-visited** prefix states beats training on
**teacher-native** prefix states at matched budget. For that contrast to be
about the state distribution and nothing else, both arms must have the same
*shape*: condition on a prefix, supervise a continuation.

So both arms are built here, through one truncation policy:

* **C** — the prefix is the teacher's own trajectory. No generation is needed:
  the tokens are already in the corpus, and truncation only moves the loss mask,
  demoting the first `k` supervised tokens to context. Byte-for-byte the same
  token stream the arm would otherwise have trained on, supervised later.
* **R** — the prefix is the student's own rollout, and the continuation is the
  teacher's recovery from that state. The tokens differ; the mask rule does not.

**CE applies to the continuation only, in both arms.** `kd_scope` stays `all` in
both, which for R means KD is applied over the student-generated prefix states as
well. That is deliberate and is why R is described as *student-prefix on-policy
KD plus teacher recovery continuation* — not as continuation-only recovery. A
result from R may not be attributed to the recovery continuation alone.

Every guard below exists because its absence produces a plausible-looking corpus
that trains the wrong thing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# A prefix must leave the model something to recover *from* and something to
# recover *to*; both ends are registered rather than tuned.
MIN_PREFIX_TOKENS = 1
MIN_CONTINUATION_TOKENS = 8


class TruncationError(ValueError):
    """A sample that cannot be split soundly. Always carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}{': ' + detail if detail else ''}")


def supervised_span(mask: list[bool]) -> tuple[int, int]:
    """`[start, end)` of the single supervised run, or raise.

    A split assumes one contiguous target span. Turn-expanded sessions supervise
    only the final assistant turn, so this holds — but a corpus change could
    break it silently, and a mask with two runs would make "the first k
    supervised tokens" ambiguous.
    """
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        raise TruncationError("no_supervised_tokens")
    start, end = idx[0], idx[-1] + 1
    if len(idx) != end - start:
        raise TruncationError("supervised_span_not_contiguous",
                              f"{len(idx)} tokens across [{start},{end})")
    return start, end


def truncation_points(n_supervised: int, *, seed_material: str, count: int = 2,
                      min_prefix: int = MIN_PREFIX_TOKENS,
                      min_continuation: int = MIN_CONTINUATION_TOKENS) -> list[int]:
    """`count` distinct truncation points, deterministic in `seed_material`.

    `k` is the number of leading supervised tokens demoted to prefix, so the
    continuation is `n_supervised - k` tokens. Points are drawn without
    replacement from the legal range — the registration requires the two
    truncations of one rollout to differ, and drawing with replacement would
    silently produce duplicates on short spans.

    Determinism comes from hashing `seed_material` (the sample's identity), not
    from a global RNG, so a rebuild reproduces the corpus exactly and adding a
    sample cannot shift another sample's cut.
    """
    lo, hi = min_prefix, n_supervised - min_continuation
    if hi < lo:
        raise TruncationError(
            "too_short_to_split",
            f"{n_supervised} supervised tokens cannot yield a {min_prefix}-token "
            f"prefix and a {min_continuation}-token continuation")
    legal = hi - lo + 1
    if legal < count:
        raise TruncationError("too_few_distinct_truncations",
                              f"{legal} legal points < {count} requested")
    digest = hashlib.sha256(seed_material.encode()).digest()
    picks: list[int] = []
    pool = legal
    # Fisher-Yates-style selection without replacement, driven by the digest.
    remaining = list(range(lo, hi + 1))
    for i in range(count):
        word = int.from_bytes(digest[i * 4:(i + 1) * 4] or b"\0\0\0\0", "big")
        j = word % pool
        picks.append(remaining.pop(j))
        pool -= 1
    return sorted(picks)


def truncation_fractions(*, seed_material: str, count: int = 2) -> list[float]:
    """`count` distinct cut *fractions* in (0, 1), deterministic in `seed_material`.

    C and R cannot match on absolute prefix length: a student rollout at this
    stage often runs for thousands of tokens while the teacher target averages
    ~641, so demanding equal token counts would force C to cut at a completely
    different relative depth than R. Matching the *fraction* of the trajectory
    consumed keeps "how far into its own reasoning the model is" comparable,
    which is the quantity the state-distribution contrast is about.

    Paired C and R examples pass the same `seed_material`, so they cut at the
    same relative depth by construction. The residual difference in absolute
    prefix tokens is then measured and reported rather than assumed away.
    """
    digest = hashlib.sha256(("frac:" + seed_material).encode()).digest()
    out: list[float] = []
    for i in range(count):
        word = int.from_bytes(digest[i * 4:(i + 1) * 4], "big")
        # (0,1) exclusive: a fraction of 0 or 1 would mean an empty prefix or an
        # empty continuation, both of which split_at refuses anyway.
        out.append((word % 9_999 + 1) / 10_000)
    if len(set(out)) != count:
        raise TruncationError("duplicate_fractions", str(out))
    return sorted(out)


def k_from_fraction(n_supervised: int, fraction: float, *,
                    min_prefix: int = MIN_PREFIX_TOKENS,
                    min_continuation: int = MIN_CONTINUATION_TOKENS) -> int:
    """Clamp a cut fraction to the legal integer range for this span."""
    lo, hi = min_prefix, n_supervised - min_continuation
    if hi < lo:
        raise TruncationError(
            "too_short_to_split",
            f"{n_supervised} supervised tokens cannot yield a {min_prefix}-token "
            f"prefix and a {min_continuation}-token continuation")
    return max(lo, min(hi, round(fraction * n_supervised)))


@dataclass
class Split:
    """One prefix/continuation training example."""

    ids: list[int]
    mask: list[bool]              # True only on the supervised continuation
    k: int                        # supervised tokens demoted to prefix
    n_prefix_tokens: int          # context length before the continuation starts
    n_continuation_tokens: int
    span_start: int
    span_end: int

    @property
    def n_tokens(self) -> int:
        return len(self.ids)


def split_at(ids: list[int], mask: list[bool], k: int, *,
             stop_ids: frozenset[int] | set[int] = frozenset(),
             min_continuation: int = MIN_CONTINUATION_TOKENS) -> Split:
    """Demote the first `k` supervised tokens to context; supervise the rest.

    Guards, each mapping to a registered requirement:

    * `k >= MIN_PREFIX_TOKENS` — never an empty assistant prefix;
    * `n_supervised - k >= min_continuation` — never a truncation at or past the
      end of the answer, so there is always real continuation to supervise;
    * the token immediately before the continuation is never a stop token — that
      would mean cutting *after* a terminal answer, leaving the model asked to
      continue past its own `<|im_end|>`.

    Token ids are never modified. Only the mask moves, which is what makes C's
    prefix provably teacher-native rather than a re-render.
    """
    start, end = supervised_span(mask)
    n_supervised = end - start
    if k < MIN_PREFIX_TOKENS:
        raise TruncationError("empty_assistant_prefix", f"k={k}")
    if n_supervised - k < min_continuation:
        raise TruncationError(
            "continuation_too_short",
            f"{n_supervised - k} < {min_continuation} (k={k}, span={n_supervised})")
    boundary = start + k
    if ids[boundary - 1] in stop_ids:
        raise TruncationError("truncation_after_terminal_token",
                              f"token {ids[boundary - 1]} at prefix end")
    new_mask = [False] * len(mask)
    for i in range(boundary, end):
        new_mask[i] = True
    return Split(ids=list(ids), mask=new_mask, k=k,
                 n_prefix_tokens=boundary,
                 n_continuation_tokens=end - boundary,
                 span_start=start, span_end=end)


def build_splits(ids: list[int], mask: list[bool], *, seed_material: str,
                 count: int = 2, stop_ids: frozenset[int] | set[int] = frozenset(),
                 max_total_tokens: int | None = None,
                 by_fraction: bool = True) -> list[Split]:
    """All `count` splits for one rollout, or raise with a single reason.

    `max_total_tokens` rejects a pathological sample deterministically *before*
    it reaches the packer, rather than letting the packer truncate it — packing
    must never cut a prefix or a supervised continuation, because a cut prefix
    changes the state being trained on and a cut continuation silently shortens
    the supervision.
    """
    if max_total_tokens is not None and len(ids) > max_total_tokens:
        raise TruncationError("exceeds_context_budget",
                              f"{len(ids)} > {max_total_tokens}")
    start, end = supervised_span(mask)
    n_supervised = end - start
    if by_fraction:
        fracs = truncation_fractions(seed_material=seed_material, count=count)
        points = [k_from_fraction(n_supervised, f) for f in fracs]
        # Clamping can collapse two nearby fractions onto one integer on a short
        # span. The registration forbids identical truncations, so fall back to
        # the integer policy rather than emitting a silent duplicate.
        if len(set(points)) != count:
            points = truncation_points(n_supervised, seed_material=seed_material,
                                       count=count)
    else:
        points = truncation_points(n_supervised, seed_material=seed_material,
                                   count=count)
    if len(set(points)) != len(points):
        raise TruncationError("duplicate_truncations", str(points))
    return [split_at(ids, mask, k, stop_ids=stop_ids) for k in points]


def prefix_length_profile(splits: list[Split]) -> dict:
    """Distribution summary used to match C's prefix lengths to R's."""
    if not splits:
        return {"n": 0}
    pre = sorted(s.n_prefix_tokens for s in splits)
    con = sorted(s.n_continuation_tokens for s in splits)

    def q(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    return {
        "n": len(splits),
        "prefix_tokens": {"min": pre[0], "p25": q(pre, .25), "p50": q(pre, .5),
                          "p75": q(pre, .75), "max": pre[-1],
                          "mean": round(sum(pre) / len(pre), 1)},
        "continuation_tokens": {"min": con[0], "p25": q(con, .25), "p50": q(con, .5),
                                "p75": q(con, .75), "max": con[-1],
                                "mean": round(sum(con) / len(con), 1),
                                "total": sum(con)},
    }
