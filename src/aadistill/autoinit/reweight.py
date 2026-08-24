"""Deterministic with-replacement reweighting of a frozen calibration pool.

`calib.reasoning_heavy@v1` declared itself a reweighted draw from the
domain-balanced pool at the pool's own budget. That is unsatisfiable twice over:
four of the five domain targets are non-integers, and the budget *is* the whole
pool, so a draw without replacement is the identity. v2 replaces it with an
explicit, mechanical procedure whose every approximation is recorded.

The rule, in five steps:

**R1 — domain apportionment.** Largest-remainder (Hamilton) over the declared
domain weights; remainder ties by ascending domain id. Sums to the budget exactly.

**R2 — unreachable domain quota.** A quota no whole-item multiset can hit is moved
to the nearest reachable value, ties toward the LOWER, and the difference is
transferred to the reachable domain with the most negative apportionment
remainder (ties by ascending id). The budget total is preserved exactly.

**R3 — sub-type apportionment.** Within a domain, split its quota across sub-types
by largest remainder over *the pool's own* sub-type position shares, so only the
domain mix changes and within-domain composition is held fixed.

**R4 — unreachable sub-type quota, repaired INSIDE its domain.** The difference
goes to another sub-type of the same domain, so no domain weight moves and the
deviation stays local.

**R5 — realization.** Among all exact whole-item multisets for a sub-type quota,
take the one maximizing distinct items used, then minimizing multiset size, then
**broken by a seed-derived ordering** — not lexicographically. The seed is part of
the profile hash and must therefore change the result; a lexicographic final
tie-break would make it decorative.

`max distinct` and `min size` are optima over the whole solution set and so are
invariant to item order. The order only decides *which* optimal realization is
returned, which is exactly where the seed belongs: permuting the items by
``sha256(seed:item_id)`` and running the same dynamic program yields a different
but equally optimal multiset for a different seed, and the identical one for the
same seed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any


class ReweightError(RuntimeError):
    """A reweighting cannot be carried out as specified."""


def largest_remainder(total: int, shares: Mapping[str, Fraction]) -> tuple[dict, dict]:
    """Integer apportionment summing to ``total`` exactly.

    Returns (quota, error), where error[k] is quota[k] minus the exact share --
    the quantity R2/R4 use to decide who absorbs a repair.
    """
    if total < 0:
        raise ReweightError(f"total must be non-negative, got {total}")
    if not shares:
        raise ReweightError("no shares to apportion over")
    if sum(shares.values()) != 1:
        raise ReweightError(f"shares sum to {sum(shares.values())}, not 1")
    exact = {k: Fraction(total) * s for k, s in shares.items()}
    quota = {k: exact[k].numerator // exact[k].denominator for k in exact}
    remainder = {k: exact[k] - quota[k] for k in exact}
    deficit = total - sum(quota.values())
    for k in sorted(exact, key=lambda k: (-remainder[k], k))[:deficit]:
        quota[k] += 1
    return quota, {k: quota[k] - exact[k] for k in exact}


def reachable_upto(sizes: Sequence[int], hi: int) -> bytearray:
    """Which sums in [0, hi] a whole-item multiset can hit, with replacement."""
    distinct = sorted({int(s) for s in sizes})
    if not distinct or distinct[0] <= 0:
        raise ReweightError(f"item sizes must be positive, got {sorted(sizes)}")
    reach = bytearray(hi + 1)
    reach[0] = 1
    for value in range(1, hi + 1):
        for size in distinct:
            if size <= value and reach[value - size]:
                reach[value] = 1
                break
    return reach


def max_distinct_upto(sizes: Sequence[int], hi: int) -> list[int]:
    """For every sum in [0, hi], the most distinct items an exact multiset can use.

    ``-1`` marks an unreachable sum. One pass, so a repair can consult the whole
    frontier instead of probing it value by value.
    """
    dp = [-1] * (hi + 1)
    dp[0] = 0
    for size in sorted({int(s) for s in sizes}):
        nxt = dp[:]
        for total in range(hi + 1):
            if dp[total] < 0:
                continue
            reached = total + size
            while reached <= hi:
                if nxt[reached] < dp[total] + 1:
                    nxt[reached] = dp[total] + 1
                reached += size
        dp = nxt
    return dp


#: How an unreachable quota is repaired. The two levels differ deliberately.
#:
#: ``NEAREST`` — used for DOMAIN quotas. A domain's deviation distorts the very
#: weights the profile exists to declare, so it is minimized: nearest reachable,
#: ties toward the lower value.
#:
#: ``MAX_SUPPORT`` — used for SUB-TYPE quotas. Here the deviation is absorbed by a
#: sibling sub-type and the domain weight does not move at all, so it can be spent
#: on something that matters more: among reachable values at or below the quota,
#: the one using the most distinct sessions, ties by smallest deviation. This is
#: what makes `multihop_qa` 7,074 (4 of 5 sessions) rather than 7,340 (four copies
#: of one session) — an item-level concentration confound bought off for 452
#: positions inside one domain.
NEAREST = "nearest_reachable_ties_lower"
MAX_SUPPORT = "max_distinct_support_at_or_below_ties_smallest_deviation"


def repair_quotas(quota: Mapping[str, int], error: Mapping[str, Fraction],
                  sizes: Mapping[str, Sequence[int]], *, headroom: int = 3000,
                  strategy: str = NEAREST) -> tuple[dict, list]:
    """R2/R4: move an unreachable quota to a reachable value.

    Applies ONLY to a quota no whole-item multiset can hit. A reachable quota is
    never "improved" — doing so would trade real deviation for support everywhere
    rather than only where the arithmetic forces a choice, and would move quotas
    that are currently exact.

    The total is conserved: whatever an unreachable quota gives up is taken by a
    reachable one, chosen by most-negative apportionment remainder so the repair
    lands where rounding was already least generous.
    """
    if strategy not in (NEAREST, MAX_SUPPORT):
        raise ReweightError(f"unknown repair strategy {strategy!r}")
    hi = max(quota.values()) + headroom
    reach = {k: reachable_upto(sizes[k], hi) for k in quota}
    adjusted, repairs, debt = dict(quota), [], 0
    for key in sorted(quota):
        target = quota[key]
        if reach[key][target]:
            continue
        lower = max(v for v in range(target + 1) if reach[key][v])
        upper = min((v for v in range(target, hi + 1) if reach[key][v]), default=None)
        if upper is None:
            raise ReweightError(f"{key}: no reachable value at or above {target}")
        if strategy == NEAREST:
            chosen = lower if (target - lower) <= (upper - target) else upper
            support = None
        else:
            frontier = max_distinct_upto(sizes[key], target)
            chosen = max(range(target + 1),
                         key=lambda v: (frontier[v], v))
            support = frontier[chosen]
        adjusted[key] = chosen
        debt += target - chosen
        repairs.append({"key": key, "quota": target, "chosen": chosen,
                        "strategy": strategy, "distinct_items_available": support,
                        "nearest_lower": lower, "nearest_upper": upper,
                        "deviation": chosen - target})

    repaired = {r["key"] for r in repairs}
    step = 1 if debt > 0 else -1
    for _ in range(abs(debt)):
        candidates = sorted(
            (k for k in adjusted
             if k not in repaired and 0 <= adjusted[k] + step <= hi
             and reach[k][adjusted[k] + step]),
            key=lambda k: (error[k] * step, k))
        if not candidates:
            raise ReweightError(f"no reachable key can absorb a {step:+d} transfer")
        adjusted[candidates[0]] += step
    if sum(adjusted.values()) != sum(quota.values()):
        raise ReweightError("repair did not conserve the total")
    return adjusted, repairs


def seed_order(item_ids: Sequence[str], seed: int) -> list[int]:
    """Indices of ``item_ids`` in a seed-derived order.

    ``sha256(f"{seed}:{item_id}")``, ascending, with the id as the final
    discriminator so two items can never compare equal. Seed-dependent by
    construction: this is what stops R5's tie-break from being lexicographic.
    """
    def key(i: int) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{item_ids[i]}".encode()).hexdigest()
        return (digest, str(item_ids[i]))
    return sorted(range(len(item_ids)), key=key)


def realize(item_ids: Sequence[str], sizes: Sequence[int], target: int, seed: int
            ) -> list[int]:
    """R5. Multiplicity per item for an exact multiset summing to ``target``.

    Maximizes distinct items, then minimizes multiset size, then takes the
    seed-derived order. Raises if the target is unreachable -- a realization that
    quietly missed its quota is the failure this whole module exists to prevent.
    """
    if len(item_ids) != len(sizes):
        raise ReweightError("item_ids and sizes disagree in length")
    if target < 0:
        raise ReweightError(f"target must be non-negative, got {target}")
    order = seed_order(item_ids, seed)

    # dp[r] = (distinct, -draws, counts) for the best multiset summing to r using
    # the items considered so far. Lexicographic max on the first two fields is
    # exactly "most distinct, then fewest draws".
    dp: dict[int, tuple[int, int, tuple[int, ...]]] = {0: (0, 0, (0,) * len(sizes))}
    for idx in order:
        size = int(sizes[idx])
        nxt = dict(dp)
        for total, (distinct, neg_draws, counts) in dp.items():
            k, reached = 1, total + size
            while reached <= target:
                cand = (distinct + 1, neg_draws - k,
                        counts[:idx] + (k,) + counts[idx + 1:])
                if reached not in nxt or cand[:2] > nxt[reached][:2]:
                    nxt[reached] = cand
                k, reached = k + 1, reached + size
        dp = nxt
    if target not in dp:
        raise ReweightError(
            f"{target} is not reachable from item sizes {sorted(set(sizes))}")
    counts = list(dp[target][2])
    if sum(c * int(s) for c, s in zip(counts, sizes)) != target:
        raise ReweightError("the realization does not sum to its target")
    return counts


def summarize(counts: Sequence[int], sizes: Sequence[int]) -> dict[str, Any]:
    return {
        "draws": int(sum(counts)),
        "distinct_items": int(sum(1 for c in counts if c > 0)),
        "positions": int(sum(c * int(s) for c, s in zip(counts, sizes))),
    }
