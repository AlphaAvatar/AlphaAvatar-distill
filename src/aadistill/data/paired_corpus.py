"""Paired construction for Experiment 5's C and R arms.

The experiment compares two prefix *state distributions* at matched budget. That
only holds if the two arms train on the **same prompts, the same source seeds and
the same truncation instances**. Two things would silently break it:

* **Asymmetric losses.** R's examples can fail generation or a quality gate; C's
  effectively cannot. Left alone, C would train on prompts R never saw, and any
  difference could be composition rather than state distribution. So the corpora
  are built from the **paired acceptance intersection**: an R rejection removes
  the C example with the same `(source_session_id, source_seed, truncation_index)`.
* **Convenience down-selection.** Cutting to a block budget by taking the first N
  examples inherits whatever order the builder happened to emit — usually corpus
  order, which is sorted by task. That would give the two arms different task
  and length mixes at different budgets. Selection here is **deterministic and
  stratified**, preserving task, source seed, truncation index and prefix-length
  bucket.

Neither arm's samples are ever truncated to fit. Whole paired samples are dropped
in stratum order instead, which is why exact supervised-token equality between
the arms is not achievable and a tolerance is pre-registered instead.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

# Prefix-length strata, in tokens. Coarse on purpose: fine buckets would make
# many strata of size 1, and a stratified draw over singletons is just an
# arbitrary order with extra steps.
PREFIX_BUCKETS = (0, 128, 256, 512, 1024, 2048, 4096)


def pair_key(example: dict) -> tuple:
    """The identity an arm-C example shares with its arm-R counterpart."""
    return (example["source_session_id"],
            example.get("source_seed"),
            int(example["truncation_index"]))


def prefix_bucket(n_prefix_tokens: int) -> str:
    """Label for the prefix-length stratum this example falls in."""
    lo = 0
    for edge in PREFIX_BUCKETS:
        if n_prefix_tokens >= edge:
            lo = edge
        else:
            return f"{lo}-{edge}"
    return f"{PREFIX_BUCKETS[-1]}+"


def stratum(example: dict) -> tuple:
    """The cell a paired example belongs to for proportional selection."""
    return (example["data_type"],
            example.get("source_seed"),
            int(example["truncation_index"]),
            prefix_bucket(int(example["n_prefix_tokens"])))


def intersect(c_examples: list[dict], r_examples: list[dict]) -> tuple[list, list, dict]:
    """Keep only the pair keys present and accepted in **both** arms.

    Returns `(c_kept, r_kept, census)` with both lists in one canonical order, so
    downstream selection and packing see the arms in the same sequence.
    """
    c_by = {pair_key(e): e for e in c_examples}
    r_by = {pair_key(e): e for e in r_examples}
    if len(c_by) != len(c_examples):
        raise ValueError("duplicate pair keys in arm C")
    if len(r_by) != len(r_examples):
        raise ValueError("duplicate pair keys in arm R")
    common = sorted(set(c_by) & set(r_by), key=lambda k: (str(k[0]), str(k[1]), k[2]))
    census = {
        "c_candidates": len(c_examples),
        "r_candidates": len(r_examples),
        "paired_common": len(common),
        "c_dropped_for_pairing": len(c_by) - len(common),
        "r_dropped_for_pairing": len(r_by) - len(common),
        "c_only_examples": sorted(str(k) for k in set(c_by) - set(r_by))[:20],
        "r_only_examples": sorted(str(k) for k in set(r_by) - set(c_by))[:20],
    }
    return [c_by[k] for k in common], [r_by[k] for k in common], census


def _cells(examples: list[dict], salt: str) -> dict[tuple, list[int]]:
    """Example indices grouped by stratum, each group in a stable hashed order."""
    cells: dict[tuple, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        cells[stratum(e)].append(i)
    for key in cells:
        cells[key].sort(
            key=lambda i: hashlib.sha256(
                (salt + "|" + "|".join(map(str, pair_key(examples[i])))).encode()
            ).hexdigest())
    return cells


def stratified_take(examples: list[dict], n: int, *, salt: str = "e5") -> list[int]:
    """`n` indices whose stratum shares track the full set (largest remainder).

    Round-robin was the obvious approach and is wrong: at every depth it takes
    one example from each stratum that still has any, so small strata are drained
    proportionally faster than large ones and a half-size cut over-represents
    them. Measured drift on a synthetic 240-example corpus was 6.7 pp.

    Largest-remainder allocation instead gives each stratum
    `floor(n * size/total)` up front and distributes what is left by descending
    fractional part, which is the same rule the token ladder uses to hit a
    declared mixture. Ties break on the stratum key so the result is
    deterministic.
    """
    n = max(0, min(n, len(examples)))
    cells = _cells(examples, salt)
    total = len(examples) or 1
    quota: dict[tuple, int] = {}
    remainders = []
    for key, idxs in cells.items():
        exact = n * len(idxs) / total
        quota[key] = min(len(idxs), int(exact))
        remainders.append((exact - int(exact), key))
    short = n - sum(quota.values())
    for _, key in sorted(remainders, key=lambda t: (-t[0], str(t[1]))):
        if short <= 0:
            break
        if quota[key] < len(cells[key]):
            quota[key] += 1
            short -= 1
    out: list[int] = []
    for key in sorted(cells, key=str):
        out += cells[key][:quota[key]]
    return sorted(out)


def stratified_order(examples: list[dict], *, salt: str = "e5") -> list[int]:
    """A deterministic stratum-balanced ordering of every index."""
    cells = _cells(examples, salt)
    order_keys = sorted(cells, key=lambda k: (-len(cells[k]), str(k)))
    out: list[int] = []
    depth = 0
    while len(out) < len(examples):
        added = False
        for key in order_keys:
            if depth < len(cells[key]):
                out.append(cells[key][depth])
                added = True
        if not added:
            break
        depth += 1
    return out


@dataclass
class SelectionReport:
    kept: int
    dropped: int
    strata_before: dict
    strata_after: dict
    max_share_drift: float


def select_paired(c_kept: list[dict], r_kept: list[dict], n: int,
                  *, salt: str = "e5") -> tuple[list[dict], list[dict], SelectionReport]:
    """Take `n` paired examples in stratified order, identically from both arms.

    Strata are computed from **C** so that one ordering drives both arms; using
    each arm's own prefix lengths would select different pairs and destroy the
    pairing the intersection just established.
    """
    if len(c_kept) != len(r_kept):
        raise ValueError("arms differ in length; intersect() first")
    n = min(n, len(c_kept))
    order = stratified_take(c_kept, n, salt=salt)
    before = Counter(stratum(e) for e in c_kept)
    after = Counter(stratum(c_kept[i]) for i in order)
    total_b, total_a = sum(before.values()) or 1, sum(after.values()) or 1
    drift = max((abs(after[k] / total_a - before[k] / total_b) for k in before),
                default=0.0)
    report = SelectionReport(
        kept=len(order), dropped=len(c_kept) - len(order),
        strata_before={str(k): v for k, v in sorted(before.items())},
        strata_after={str(k): v for k, v in sorted(after.items())},
        max_share_drift=round(drift, 5))
    return [c_kept[i] for i in order], [r_kept[i] for i in order], report


def length_profile(examples: list[dict]) -> dict:
    """Prefix / continuation / total length distributions and packing inputs."""
    if not examples:
        return {"n": 0}

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    pre = [int(e["n_prefix_tokens"]) for e in examples]
    con = [int(e["n_continuation_tokens"]) for e in examples]
    tot = [int(e["n_total_tokens"]) for e in examples]
    frac = [float(e["truncation_fraction"]) for e in examples
            if e.get("truncation_fraction") is not None]
    out = {
        "n": len(examples),
        "supervised_continuation_tokens": sum(con),
        "total_nonpadding_tokens": sum(tot),
        "prefix_tokens": {"min": min(pre), "p25": q(pre, .25), "p50": q(pre, .5),
                          "p75": q(pre, .75), "max": max(pre),
                          "mean": round(sum(pre) / len(pre), 1)},
        "continuation_tokens": {"min": min(con), "p25": q(con, .25), "p50": q(con, .5),
                                "p75": q(con, .75), "max": max(con),
                                "mean": round(sum(con) / len(con), 1)},
        "total_tokens": {"p50": q(tot, .5), "max": max(tot),
                         "mean": round(sum(tot) / len(tot), 1)},
        "by_task_continuation_tokens": dict(
            sorted(Counter({e["data_type"]: 0 for e in examples}).items())),
        "prefix_buckets": dict(sorted(Counter(
            prefix_bucket(int(e["n_prefix_tokens"])) for e in examples).items())),
    }
    by_task: Counter = Counter()
    for e in examples:
        by_task[e["data_type"]] += int(e["n_continuation_tokens"])
    out["by_task_continuation_tokens"] = dict(by_task.most_common())
    if frac:
        out["truncation_fraction"] = {
            "min": round(min(frac), 4), "p50": round(q(frac, .5), 4),
            "max": round(max(frac), 4),
            "mean": round(sum(frac) / len(frac), 4)}
    return out


def comparability_report(c: list[dict], r: list[dict], *,
                         supervised_tolerance: float) -> dict:
    """Everything the registration requires reported for both arms, plus the gate.

    `supervised_tolerance` is a **pre-registered** relative bound on the
    difference in supervised continuation tokens between the arms. Whole samples
    are never truncated to close that gap, so some residual is expected; the
    bound is fixed before any training or evaluation is seen.
    """
    cp, rp = length_profile(c), length_profile(r)
    cs = cp.get("supervised_continuation_tokens", 0)
    rs = rp.get("supervised_continuation_tokens", 0)
    denom = max(1, (cs + rs) / 2)
    rel = abs(cs - rs) / denom
    return {
        "paired_examples": len(c),
        "arm_c": cp, "arm_r": rp,
        "supervised_token_delta": cs - rs,
        "supervised_token_relative_delta": round(rel, 5),
        "supervised_tolerance": supervised_tolerance,
        "within_tolerance": bool(rel <= supervised_tolerance),
        "note": ("prefix lengths are matched on truncation FRACTION by "
                 "construction; absolute prefix tokens differ because student "
                 "and teacher trajectories differ in length, and that difference "
                 "is part of what E5 tests"),
    }


# The registered nested-rung increment: unique supervised tokens added when the
# 0.86M rung is extended to 1.60M. It is the *scale* E5 is trying to reproduce,
# and it is NOT interchangeable with a block or step count.
NESTED_RUNG_INCREMENT = 735_603


def suffix_overlap(examples: list[dict]) -> dict:
    """Repeated supervision caused by two truncations of one trajectory.

    Two cuts `k1 < k2` of the same span yield continuations `[k1,end)` and
    `[k2,end)`: the second is entirely inside the first. So the *union* is the
    longer continuation and the shorter one is counted twice in any naive sum.
    Reporting the sum alone overstates coverage by exactly that overlap, which is
    why the candidate count and the unique count are both required.
    """
    by_source: dict[str, list[int]] = defaultdict(list)
    for e in examples:
        by_source[e["source_session_id"]].append(int(e["n_continuation_tokens"]))
    candidate = sum(sum(v) for v in by_source.values())
    unique = sum(max(v) for v in by_source.values())
    return {
        "candidate_continuation_tokens": candidate,
        "unique_supervised_tokens": unique,
        "repeated_presentation_tokens": candidate - unique,
        "repeated_share": round((candidate - unique) / candidate, 4) if candidate else 0.0,
        "source_trajectories": len(by_source),
    }


def select_paired_to_token_target(c_kept: list[dict], r_kept: list[dict],
                                  target: int = NESTED_RUNG_INCREMENT,
                                  *, salt: str = "e5") -> tuple[list, list, dict]:
    """Choose the paired subset whose CE-token totals sit closest to `target`.

    The tolerance is a hard ceiling, not an aim: this searches the paired count
    `n` to minimise the *worse* of the two arms' relative deviations from the
    target, using the composition-preserving stratified selector at every
    candidate `n`. Whole samples only — nothing is truncated to close a gap.

    Both arms are scored because a pair contributes a different number of tokens
    to each: C's continuation and R's continuation come from different
    trajectories even at the same relative cut depth.
    """
    if len(c_kept) != len(r_kept):
        raise ValueError("arms differ in length; intersect() first")

    def totals(n: int) -> tuple[int, int, list[int]]:
        idx = stratified_take(c_kept, n, salt=salt)
        return (sum(int(c_kept[i]["n_continuation_tokens"]) for i in idx),
                sum(int(r_kept[i]["n_continuation_tokens"]) for i in idx), idx)

    def cost(n: int) -> float:
        ct, rt, _ = totals(n)
        return max(abs(ct - target), abs(rt - target)) / max(1, target)

    lo, hi = 1, len(c_kept)
    best_n, best_cost = hi, cost(hi)
    # Coarse sweep then local refinement: the totals are monotone in n up to
    # stratum granularity, but not perfectly, so a pure binary search can stop
    # one stratum early.
    step = max(1, len(c_kept) // 64)
    for n in range(lo, hi + 1, step):
        c = cost(n)
        if c < best_cost:
            best_n, best_cost = n, c
    for n in range(max(lo, best_n - step), min(hi, best_n + step) + 1):
        c = cost(n)
        if c < best_cost:
            best_n, best_cost = n, c

    ct, rt, idx = totals(best_n)
    c_sel = [c_kept[i] for i in idx]
    r_sel = [r_kept[i] for i in idx]
    return c_sel, r_sel, {
        "target_supervised_tokens": target,
        "selected_pairs": best_n,
        "available_pairs": len(c_kept),
        "arm_c_supervised": ct,
        "arm_r_supervised": rt,
        "arm_c_vs_target": round(ct / target, 4),
        "arm_r_vs_target": round(rt / target, 4),
        "worst_relative_deviation_from_target": round(best_cost, 5),
    }


def packing_report(examples: list[dict], n_blocks: int, block_len: int) -> dict:
    """Utilisation of a fixed block budget by a corpus that is never truncated."""
    nonpad = sum(int(e["n_total_tokens"]) for e in examples)
    ce = sum(int(e["n_continuation_tokens"]) for e in examples)
    capacity = n_blocks * block_len
    return {
        "blocks": n_blocks, "block_len": block_len, "capacity_tokens": capacity,
        "total_nonpadding_tokens": nonpad,
        "padding_tokens": max(0, capacity - nonpad),
        "packing_efficiency": round(nonpad / capacity, 4) if capacity else 0.0,
        # kd_scope="all" is literally every real token, so the KD mask is the
        # non-padding count and the CE mask is the continuation count.
        "ce_mask_tokens": ce,
        "kd_mask_tokens": nonpad,
        "ce_share_of_kd": round(ce / nonpad, 4) if nonpad else 0.0,
        "fits": nonpad <= capacity,
    }
