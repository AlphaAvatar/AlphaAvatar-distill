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


def bundle_key(example: dict) -> tuple:
    """The atomic unit of selection: one source trajectory under one seed.

    The registered design is *two truncations per prompt*. Letting a session
    contribute one surviving cut would quietly change that design for part of
    the corpus, so bundles survive or fail together and are selected together.
    """
    return (example["source_session_id"], example.get("source_seed"))


def bundle_stratum(bundle: list[dict]) -> tuple:
    """Stratum for a whole bundle. `truncation_index` drops out — every bundle
    holds both indices by construction — and the prefix bucket is taken from the
    shallower cut, which is the one that determines how much context the bundle
    carries at its lightest."""
    first = bundle[0]
    return (first["data_type"], first.get("source_seed"),
            prefix_bucket(min(int(e["n_prefix_tokens"]) for e in bundle)))


def group_bundles(examples: list[dict], *, expected: int = 2) -> dict[tuple, list[dict]]:
    """Group examples into bundles, keeping only complete ones.

    An incomplete bundle is not an error here — it is the normal consequence of
    one truncation failing an R gate — but it must never reach selection, so it
    is dropped whole.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in examples:
        groups[bundle_key(e)].append(e)
    out = {}
    for key, members in groups.items():
        members.sort(key=lambda e: int(e["truncation_index"]))
        if len(members) == expected and len({
                int(e["truncation_index"]) for e in members}) == expected:
            out[key] = members
    return out


def intersect(c_examples: list[dict], r_examples: list[dict], *,
              truncations: int = 2) -> tuple[list, list, dict]:
    """Keep only the **complete bundles** accepted in both arms.

    Bundle-atomic by requirement: if either truncation of a `(session, seed)`
    fails generation or any R quality gate, both truncations are removed from
    both arms. A single-cut survivor would silently violate the registered
    two-truncations-per-prompt design for part of the corpus.

    Returns `(c_kept, r_kept, census)` in one canonical order so downstream
    selection and packing see the arms in the same sequence.
    """
    c_by = {pair_key(e): e for e in c_examples}
    r_by = {pair_key(e): e for e in r_examples}
    if len(c_by) != len(c_examples):
        raise ValueError("duplicate pair keys in arm C")
    if len(r_by) != len(r_examples):
        raise ValueError("duplicate pair keys in arm R")

    c_bundles = group_bundles(c_examples, expected=truncations)
    r_bundles = group_bundles(r_examples, expected=truncations)
    common = sorted(set(c_bundles) & set(r_bundles), key=lambda k: (str(k[0]), str(k[1])))

    c_kept, r_kept = [], []
    for key in common:
        c_kept += c_bundles[key]
        r_kept += r_bundles[key]

    c_all = {bundle_key(e) for e in c_examples}
    r_all = {bundle_key(e) for e in r_examples}
    census = {
        "truncations_per_bundle": truncations,
        "c_candidate_examples": len(c_examples),
        "r_candidate_examples": len(r_examples),
        "c_candidate_bundles": len(c_all),
        "r_candidate_bundles": len(r_all),
        "c_complete_bundles": len(c_bundles),
        "r_complete_bundles": len(r_bundles),
        "r_incomplete_bundles_dropped": len(r_all) - len(r_bundles),
        "paired_bundles": len(common),
        "paired_examples": len(c_kept),
        "c_bundles_dropped_for_pairing": len(c_bundles) - len(common),
        "r_bundles_dropped_for_pairing": len(r_bundles) - len(common),
        "c_only_bundles": sorted(str(k) for k in set(c_bundles) - set(common))[:20],
        "r_only_bundles": sorted(str(k) for k in set(r_bundles) - set(common))[:20],
    }
    return c_kept, r_kept, census


def _bundle_cells(bundles: list[list[dict]], salt: str) -> dict[tuple, list[int]]:
    """Bundle indices grouped by bundle stratum, each group in hashed order."""
    cells: dict[tuple, list[int]] = defaultdict(list)
    for i, b in enumerate(bundles):
        cells[bundle_stratum(b)].append(i)
    for key in cells:
        cells[key].sort(
            key=lambda i: hashlib.sha256(
                (salt + "|" + "|".join(map(str, bundle_key(bundles[i][0])))).encode()
            ).hexdigest())
    return cells


def take_bundles(bundles: list[list[dict]], n: int, *, salt: str = "e5") -> list[int]:
    """`n` bundle indices, largest-remainder over bundle strata."""
    n = max(0, min(n, len(bundles)))
    cells = _bundle_cells(bundles, salt)
    total = len(bundles) or 1
    quota, remainders = {}, []
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


def as_bundles(examples: list[dict], *, truncations: int = 2) -> list[list[dict]]:
    """Canonical bundle list for an already-intersected arm."""
    groups = group_bundles(examples, expected=truncations)
    return [groups[k] for k in sorted(groups, key=lambda k: (str(k[0]), str(k[1])))]


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
    # Bundle-atomic: `n` counts EXAMPLES for the caller's convenience but is
    # honoured in whole bundles, so a session never contributes one cut.
    c_bundles, r_bundles = as_bundles(c_kept), as_bundles(r_kept)
    per = len(c_bundles[0]) if c_bundles else 1
    n_bundles = min(len(c_bundles), max(0, n) // per)
    picked = take_bundles(c_bundles, n_bundles, salt=salt)
    before = Counter(bundle_stratum(b) for b in c_bundles)
    after = Counter(bundle_stratum(c_bundles[i]) for i in picked)
    total_b, total_a = sum(before.values()) or 1, sum(after.values()) or 1
    drift = max((abs(after[k] / total_a - before[k] / total_b) for k in before),
                default=0.0)
    c_sel = [e for i in picked for e in c_bundles[i]]
    r_sel = [e for i in picked for e in r_bundles[i]]
    report = SelectionReport(
        kept=len(c_sel), dropped=len(c_kept) - len(c_sel),
        strata_before={str(k): v for k, v in sorted(before.items())},
        strata_after={str(k): v for k, v in sorted(after.items())},
        max_share_drift=round(drift, 5))
    return c_sel, r_sel, report


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

    c_bundles = as_bundles(c_kept)
    r_bundles = as_bundles(r_kept)
    if len(c_bundles) != len(r_bundles):
        raise ValueError("bundle counts differ; intersect() first")

    def totals(n: int) -> tuple[int, int, list[int]]:
        idx = take_bundles(c_bundles, n, salt=salt)
        ct = sum(int(e["n_continuation_tokens"]) for i in idx for e in c_bundles[i])
        rt = sum(int(e["n_continuation_tokens"]) for i in idx for e in r_bundles[i])
        return ct, rt, idx

    def cost(n: int) -> float:
        ct, rt, _ = totals(n)
        return max(abs(ct - target), abs(rt - target)) / max(1, target)

    lo, hi = 1, len(c_bundles)
    best_n, best_cost = hi, cost(hi)
    # Coarse sweep then local refinement: the totals are monotone in n up to
    # stratum granularity, but not perfectly, so a pure binary search can stop
    # one stratum early.
    step = max(1, len(c_bundles) // 64)
    for n in range(lo, hi + 1, step):
        c = cost(n)
        if c < best_cost:
            best_n, best_cost = n, c
    for n in range(max(lo, best_n - step), min(hi, best_n + step) + 1):
        c = cost(n)
        if c < best_cost:
            best_n, best_cost = n, c

    ct, rt, idx = totals(best_n)
    c_sel = [e for i in idx for e in c_bundles[i]]
    r_sel = [e for i in idx for e in r_bundles[i]]
    return c_sel, r_sel, {
        "target_supervised_tokens": target,
        "selection_unit": "two-truncation session bundle",
        "selected_bundles": best_n,
        "available_bundles": len(c_bundles),
        "selected_pairs": len(c_sel),
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
