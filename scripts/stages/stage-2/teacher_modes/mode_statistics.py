"""Research A's decision rule, executable before any result exists. $0.

The preregistration (``d4_scale_preregistration.json :: decision_rule``)
binds this file's sha256 and those of the code it calls: H1's pairing
(`h1_statistics._paired`, imported unchanged -- that file stays byte-identical,
H1-PA's frozen preregistration binds it too) and the bootstrap
(`aadistill.evaluation.paired_stats.cluster_bootstrap`); `decide` refuses when
any differs from what was registered.

Input rows, one per (arm, seed, item) on the 570 formal STATIC items of
fluid_reasoning_720 under nucleus_v1: ``{"arm", "seed", "item_id", "cluster"
(the rule family), "axis", "correct", "decoding_sha256"}``.

Comparisons, each Y - X with X the non-thinking arm, paired by item and by
training-seed number, each at its own 97.5% percentile interval (Bonferroni
over the two decided comparisons): S = H-T-S - H-N-S; C = H-T-S - H-N-C, or,
under the registered fallback, H-T-C - H-N-S. Per item, d_i = the mean over
seeds of (Y correct - X correct); the statistic is the item-weighted mean; a
rule-family cluster bootstrap stratified by axis, 10,000 resamples, seed
20261029. Labels, in order: NOT_RESOLVED (fewer than 10 discordant item-seed
pairs, fewer than 5 informative clusters, a degenerate bootstrap, or both arms
below 2%: FLOOR); GO_THINKING (wholly above +5 pp and every seed above);
GO_NON_THINKING (wholly below -5 pp and every seed below); NO_GO (wholly inside
(-5, +5) pp, every seed inside, neither arm below 2%); INCONCLUSIVE. Within a
comparison the static axes are a Holm family at 0.05 on each axis's one-sided
bootstrap share at or beyond +/-8 pp. S and C are read separately, never
merged.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from aadistill.evaluation.paired_stats import cluster_bootstrap
from stages.reasoning_data import h1_statistics as HS

RULE_ID = "stage2.teacher_modes.statistics/v1"
REPO = Path(__file__).resolve().parents[4]
SESOI, SESOI_AXIS, LEVEL, ITERATIONS, BOOTSTRAP_SEED = 0.05, 0.08, 0.975, 10_000, 20261029
FLOOR_RATE, MIN_DISCORDANT, MIN_INFORMATIVE_CLUSTERS, ALPHA_AXES = 0.02, 10, 5, 0.05
EXPLORATORY_AXES = frozenset({"interactive_adaptation"})
#: name -> (X, Y): the contrast is Y - X
COMPARISONS = {"S": ("H-N-S", "H-T-S"), "C": ("H-N-C", "H-T-S")}
FALLBACK_C = ("H-N-S", "H-T-C")
BOUND = ("scripts/stages/stage-2/teacher_modes/mode_statistics.py", "scripts/stages/stage-2/reasoning_data/h1_statistics.py",
         "src/aadistill/evaluation/paired_stats.py")


def bound_identity(repo_root: Path = REPO) -> dict:
    return {p: hashlib.sha256((Path(repo_root) / p).read_bytes()).hexdigest() for p in BOUND}


def check_bound(prereg: Mapping, repo_root: Path = REPO) -> None:
    want = prereg["decision_rule"]["bound_sha256"]
    got = bound_identity(repo_root)
    if got != want:
        raise ValueError(f"the decision rule's bytes are not the registered ones: {got} vs {want}")


def _label(items: dict, seeds: list, sesoi: float, *, stratify: bool) -> dict:
    ids = sorted(items)
    n, k = len(ids), len(seeds)
    d = [sum(b - a for a, b in zip(items[i]["A"], items[i]["B"])) / k for i in ids]
    boot = cluster_bootstrap(d, [items[i]["cluster"] for i in ids], [items[i]["axis"] for i in ids] if stratify else None,
                             iterations=ITERATIONS, seed=BOOTSTRAP_SEED, level=LEVEL, thresholds=(sesoi, -sesoi))
    rate = {arm: sum(sum(items[i][arm]) for i in ids) / (n * k) for arm in ("A", "B")}
    per_seed = {str(s): round(sum(items[i]["B"][j] - items[i]["A"][j] for i in ids) / n, 6) for j, s in enumerate(seeds)}
    discordant = sum(items[i]["A"][j] != items[i]["B"][j] for i in ids for j in range(k))
    informative = len({items[i]["cluster"] for i in ids if any(items[i]["A"][j] != items[i]["B"][j] for j in range(k))})
    floor = {arm: rate[arm] < FLOOR_RATE for arm in rate}
    resolved = discordant >= MIN_DISCORDANT and informative >= MIN_INFORMATIVE_CLUSTERS and not boot["degenerate"]
    lo, hi, pts = boot["ci_low"], boot["ci_high"], list(per_seed.values())
    if not resolved or all(floor.values()):
        label = "NOT_RESOLVED"
    elif lo > sesoi and all(p > sesoi for p in pts):
        label = "GO_THINKING"
    elif hi < -sesoi and all(p < -sesoi for p in pts):
        label = "GO_NON_THINKING"
    elif -sesoi < lo and hi < sesoi and all(-sesoi < p < sesoi for p in pts) and not any(floor.values()):
        label = "NO_GO"
    else:
        label = "INCONCLUSIVE"
    flags = ["FLOOR"] if all(floor.values()) else [f"FLOOR_{'X' if a == 'A' else 'Y'}" for a, f in sorted(floor.items()) if f]
    return {"label": label, "flags": flags, "sesoi": sesoi, "delta": boot["point"], "ci": [lo, hi], "level": LEVEL,
            "per_seed_delta": per_seed, "rates": {"X": round(rate["A"], 6), "Y": round(rate["B"], 6)},
            "resolution": {"resolved": resolved, "discordant_item_seed_pairs": discordant,
                           "informative_clusters": informative, "degenerate_bootstrap": boot["degenerate"]},
            "bootstrap": boot}


def compare(rows: list[Mapping], x: str, y: str) -> dict:
    sub = [{**r, "arm": {x: "A", y: "B"}[r["arm"]]} for r in rows if r["arm"] in (x, y)]
    bad = sorted({r["axis"] for r in sub} & EXPLORATORY_AXES)
    if bad:
        raise ValueError(f"{bad} is exploratory: reported beside the decision, never inside it")
    items, seeds = HS._paired(sub)
    primary = _label(items, seeds, SESOI, stratify=True)
    axes: dict = defaultdict(dict)
    for i, it in items.items():
        axes[it["axis"]][i] = it
    by_axis = {ax: _label(s, seeds, SESOI_AXIS, stratify=False) for ax, s in sorted(axes.items())}
    for sign, key in ((+1, "gain"), (-1, "loss")):
        t = str(sign * SESOI_AXIS)
        p = {ax: r["bootstrap"]["tail_shares"][t]["at_or_below" if sign > 0 else "at_or_above"]
             for ax, r in by_axis.items()}
        alive, m = True, len(p)
        for rank, ax in enumerate(sorted(p, key=lambda a: (p[a], a))):
            alive = alive and p[ax] <= ALPHA_AXES / (m - rank)
            want = "GO_THINKING" if sign > 0 else "GO_NON_THINKING"
            by_axis[ax][f"holm_{key}_survives"] = bool(alive and by_axis[ax]["label"] == want)
    return {"contrast": f"{y} - {x}", "seeds": seeds, "n_items": len(items), "primary": primary, "by_axis": by_axis}


def decide(rows: Iterable[Mapping], *, prereg: Mapping, c_fallback: bool, repo_root: Path = REPO) -> dict:
    check_bound(prereg, repo_root)
    want = prereg["evaluation"]["decoding"]["sha256"]
    rows = list(rows)
    bad = sorted({str(r.get("decoding_sha256"))[:12] for r in rows if r.get("decoding_sha256") != want})
    if bad:
        raise ValueError(f"rows under decoding {bad}, not the registered {want[:12]}")
    comps = {"S": COMPARISONS["S"], "C": FALLBACK_C if c_fallback else COMPARISONS["C"]}
    return {"rule": RULE_ID, "comparisons": {k: compare(rows, x, y) for k, (x, y) in comps.items()},
            "c_fallback": c_fallback,
            "constants": {"sesoi": SESOI, "sesoi_axis": SESOI_AXIS, "level": LEVEL, "iterations": ITERATIONS,
                          "bootstrap_seed": BOOTSTRAP_SEED, "floor_rate": FLOOR_RATE, "min_discordant": MIN_DISCORDANT,
                          "min_informative_clusters": MIN_INFORMATIVE_CLUSTERS, "alpha_axes": ALPHA_AXES},
            "reading": "S and C are read separately and never merged: a mode that wins S and not C wins by compute; "
                       "one that wins C and not S wins by data efficiency"}
