"""SOB1's registered decision rule (`design.json :: analysis`), as pure functions over per-item correctness; the
statistics are TPI1's (`teacher_parent_initialization.analysis`: group-paired differences, the sign-flip test, the
group bootstrap), only the multiplicity is new: three pairwise comparisons per mode.

* Family 1 (primary): the thinking-mode pairs, Holm at family-wise alpha 0.05.
* Family 2 (key secondary): the non-thinking pairs, Holm at 0.05, tested only if family 1 rejects all three (serial
  gatekeeping, so the six hypotheses keep a family-wise error of 0.05); otherwise descriptive.
* A pair is decided only when both arms cover every registered item in both seeds; otherwise ``descriptive_only``.
  Verdicts are ``<X>_BETTER``, ``<Y>_BETTER`` or ``NOT_DISTINGUISHED`` (not equivalence). No ranking or pooled score
  is defined beyond the pairwise verdicts.

The reader of the generations is not here yet: it is TPI1's (`rescore`, `correctness`), applied per arm, when there
are records to read.
"""
from __future__ import annotations

from stages.teacher_parent_initialization import analysis as TA

ALPHA = 0.05
TEST_SEED, BOOT_SEED = 20261119, 20261120


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm's step-down procedure: ``{key: {"p", "p_adjusted", "rejected"}}``; ties keep their key order."""
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m, running, out, stop = len(order), 0.0, {}, False
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        stop = stop or pvalues[k] > alpha / (m - i)
        out[k] = {"p": pvalues[k], "p_adjusted": running, "rejected": not stop}
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


def decide(y: dict, pairs: list[tuple[str, str]], groups: dict, seeds: list, *, primary: str = "H-T",
           secondary: str = "H-N") -> dict:
    """The registered decision over ``pairs`` in both modes."""
    items = [i for v in groups.values() for i in v]
    out: dict = {"_rule": {"alpha": ALPHA, "family_1": primary, "family_2": secondary,
                           "multiplicity": "Holm within each mode; the second family only if the first rejects all",
                           "seeds": [int(s) for s in seeds], "registered_items": len(items), "groups": len(groups)}}

    def family(mode: str, test: bool) -> dict:
        res, pv = {}, {}
        for pair in pairs:
            key = f"{pair[0]} - {pair[1]}"
            if not all(TA.covers(y.get(a, {}).get(mode), seeds, items) for a in pair):
                res[key] = {"status": "descriptive_only", "why": "an arm's records do not cover every item in both seeds"}
                continue
            res[key] = {"status": "formal" if test else "descriptive_only",
                        **pair_comparison(y, pair, mode, groups, seeds, test=test)}
            if test:
                pv[key] = res[key]["p_two_sided"]
        if test and pv:
            for key, h in holm(pv).items():
                pair = tuple(key.split(" - "))
                res[key].update(p_holm=h["p_adjusted"], rejected=h["rejected"],
                                verdict=verdict(pair, res[key]["delta"], h["rejected"]))
        return res

    out[primary] = family(primary, True)
    all_rejected = all(v.get("rejected") for v in out[primary].values()) and len(out[primary]) == len(pairs)
    out[secondary] = family(secondary, all_rejected)
    if not all_rejected:
        for v in out[secondary].values():
            v.setdefault("why", "not tested: the primary family did not reject every pair (serial gatekeeping)")
    return out
