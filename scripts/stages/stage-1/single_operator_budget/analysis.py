"""SOB1's registered decision rule (`design.json :: analysis`), as pure functions over per-item correctness; the
statistics are TPI1's (`teacher_parent_initialization.analysis`: group-paired differences, the sign-flip test, the
group bootstrap), only the multiplicity is new: three pairwise comparisons per mode.

* Family 1 (primary): the three thinking-mode pairs, Holm at family-wise alpha 0.05 with **m = 3 always** (the
  registered pairs, not the pairs that happen to have records): a pair that cannot be tested still counts in m.
* Family 2 (key secondary): the non-thinking pairs, Holm at 0.05 with m = 3, tested only if family 1 rejects all three
  (serial gatekeeping, so the six hypotheses keep a family-wise error of 0.05); otherwise descriptive.
* **A mode is decided only when the primary units are complete**: every registered arm's records cover every
  registered item under every registered seed. Anything less -- an arm, a seed or an item missing -- leaves EVERY pair
  of that mode ``descriptive_only`` (a pair whose own two arms are complete still gets its delta and interval; no
  test, never partially pooled). Verdicts are ``<X>_BETTER``, ``<Y>_BETTER`` or ``NOT_DISTINGUISHED`` (not
  equivalence). No ranking or pooled score is defined beyond the pairwise verdicts, and nothing here drops an arm.
* Per arm, descriptive only (`arm_report`): the initialization damage (untrained vs teacher), the recovery (recovered
  vs untrained) and the remaining gap (recovered vs teacher), all at 8,192 tokens, with group-bootstrap intervals.

The reader of the generations is not here yet: it is TPI1's (`rescore`, `correctness`), applied per arm, when there
are records to read.
"""
from __future__ import annotations

from stages.teacher_parent_initialization import analysis as TA

ALPHA = 0.05
TEST_SEED, BOOT_SEED = 20261119, 20261120


def holm(pvalues: dict[str, float], alpha: float = ALPHA, m: int | None = None) -> dict[str, dict]:
    """Holm's step-down procedure over ``m`` hypotheses (default: the p-values given): ``{key: {"p", "p_adjusted",
    "rejected"}}``; ties keep their key order. With ``m`` larger than the p-values given, the absent hypotheses count
    as never rejected (p = 1): the adjustment is never weakened by a comparison that could not be made."""
    m = len(pvalues) if m is None else int(m)
    if m < len(pvalues):
        raise ValueError(f"m = {m} for {len(pvalues)} p-values")
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    running, out, stop = 0.0, {}, False
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        stop = stop or pvalues[k] > alpha / (m - i)
        out[k] = {"p": pvalues[k], "p_adjusted": running, "rejected": not stop, "m": m}
    return out


def pair_comparison(y: dict, pair: tuple[str, str], mode: str, groups: dict, seeds: list, *, test: bool) -> dict:
    """One pair (X, Y) in one mode, ``y[arm][mode][seed][item]`` in {0, 1}: TPI1's statistic, interval and test with
    this design's registered generators."""
    x_arm, y_arm = pair
    ya, yb = y[x_arm][mode], y[y_arm][mode]
    items = [i for v in groups.values() for i in v]
    _keys, d, n = TA.pooled_group_differences(ya, yb, groups, seeds)
    acc = {a: sum(int(v[s][i]) for s in seeds for i in items) / (len(seeds) * len(items))
           for a, v in ((x_arm, ya), (y_arm, yb))}
    out = {"n_items": len(items), "n_seeds": len(seeds), "n_paired_records": int(n.sum()), "n_groups": len(groups),
           "delta": float(d.sum() / n.sum()), "accuracy": acc,
           "effect_interval": TA.LA.group_bootstrap(d, n, draws=TA.BOOT_DRAWS, seed=BOOT_SEED)}
    if test:
        out["test"] = TA.LA.sign_flip_test(d, draws=TA.TEST_DRAWS, seed=TEST_SEED)
        out["p_two_sided"] = out["test"]["p_two_sided"]
    return out


def verdict(pair: tuple[str, str], delta: float, rejected: bool) -> str:
    if not rejected or delta == 0:
        return "NOT_DISTINGUISHED"
    return f"{pair[0] if delta > 0 else pair[1]}_BETTER"


def completeness(y: dict, arms: list[str], mode: str, seeds: list, items: list[str]) -> dict:
    """Which registered arms lack a seed's records or an item in ``mode`` (empty ``missing``: complete)."""
    missing = {}
    for a in arms:
        rec = (y.get(a) or {}).get(mode)
        gaps = [s for s in seeds if rec is None or rec.get(s) is None or any(i not in rec[s] for i in items)]
        if gaps:
            missing[a] = gaps
    return {"complete": not missing, "missing": missing}


def decide(y: dict, pairs: list[tuple[str, str]], groups: dict, seeds: list, *, primary: str = "H-T",
           secondary: str = "H-N") -> dict:
    """The registered decision over ``pairs`` in both modes (module docstring)."""
    items = [i for v in groups.values() for i in v]
    arms = sorted({a for p in pairs for a in p})
    m = len(pairs)
    out: dict = {"_rule": {"alpha": ALPHA, "family_1": primary, "family_2": secondary, "m": m,
                           "multiplicity": "Holm within each mode over the registered pairs (m fixed); the second "
                                           "family only if the first rejects all",
                           "formal_requires": "every registered arm x every registered seed covers every item",
                           "seeds": [int(s) for s in seeds], "arms": arms, "registered_items": len(items),
                           "groups": len(groups)}}

    def family(mode: str, gate_open: bool) -> dict:
        comp = completeness(y, arms, mode, seeds, items)
        formal = gate_open and comp["complete"]
        res, pv = {"_completeness": comp}, {}
        for pair in pairs:
            key = f"{pair[0]} - {pair[1]}"
            if not all(TA.covers((y.get(a) or {}).get(mode), seeds, items) for a in pair):
                res[key] = {"status": "descriptive_only",
                            "why": "an arm's records do not cover every item in every seed: not computable"}
                continue
            res[key] = {"status": "formal" if formal else "descriptive_only",
                        **pair_comparison(y, pair, mode, groups, seeds, test=formal)}
            if formal:
                pv[key] = res[key]["p_two_sided"]
            elif not comp["complete"]:
                res[key]["why"] = (f"the mode's primary units are incomplete ({comp['missing']}): descriptive, "
                                   "no test")
        if formal:
            for key, h in holm(pv, m=m).items():
                pair = tuple(key.split(" - "))
                res[key].update(p_holm=h["p_adjusted"], holm_m=h["m"], rejected=h["rejected"],
                                verdict=verdict(pair, res[key]["delta"], h["rejected"]))
        return res

    out[primary] = family(primary, True)
    tested = [v for k, v in out[primary].items() if not k.startswith("_")]
    all_rejected = len(tested) == m and all(v.get("rejected") for v in tested)
    out[secondary] = family(secondary, all_rejected)
    if not all_rejected:
        for k, v in out[secondary].items():
            if not k.startswith("_"):
                v.setdefault("why", "not tested: the primary family did not reject every pair (serial gatekeeping)")
    return out


def _paired(a: dict, b: dict, groups: dict, seeds: list) -> dict:
    """``a`` against ``b`` (``[seed][item]``), pooled over ``seeds``: delta and the group-bootstrap interval."""
    items = [i for v in groups.values() for i in v]
    _k, d, n = TA.pooled_group_differences(a, b, groups, seeds)
    return {"delta": float(d.sum() / n.sum()), "n_paired_records": int(n.sum()),
            "accuracy": [sum(int(x[s][i]) for s in seeds for i in items) / (len(seeds) * len(items)) for x in (a, b)],
            "effect_interval": TA.LA.group_bootstrap(d, n, draws=TA.BOOT_DRAWS, seed=BOOT_SEED)}


def arm_report(pre: dict | None, post: dict | None, teacher: dict | None, groups: dict, seeds: list) -> dict:
    """One arm in one mode, descriptive only. ``pre`` / ``teacher``: ``{item: y}`` (one generation each); ``post``:
    ``{seed: {item: y}}``. Initialization damage = untrained - teacher; recovery = recovered - untrained (per seed and
    pooled); remaining gap = recovered - teacher. A part whose records are incomplete is reported ``not_available``."""
    items = [i for v in groups.values() for i in v]
    full = lambda r: r is not None and all(i in r for i in items)                      # noqa: E731
    one = [0]
    out = {"role": "descriptive only: initial KL, CE and generation scores eliminate nothing"}
    out["initialization_damage"] = (_paired({0: pre}, {0: teacher}, groups, one) if full(pre) and full(teacher)
                                    else "not_available")
    post_ok = post is not None and all(full(post.get(s)) for s in seeds)
    if post_ok and full(pre):
        out["recovery"] = {"pooled": _paired(post, {s: pre for s in seeds}, groups, seeds),
                           "per_seed": {str(s): _paired({s: post[s]}, {s: pre}, groups, [s]) for s in seeds}}
    else:
        out["recovery"] = "not_available"
    out["remaining_gap"] = (_paired(post, {s: teacher for s in seeds}, groups, seeds) if post_ok and full(teacher)
                            else "not_available")
    return out
