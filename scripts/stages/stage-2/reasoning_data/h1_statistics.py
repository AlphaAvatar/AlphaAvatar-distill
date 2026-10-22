"""H1's decision rule, executable before any result exists. $0.

The protocol (`plans/comparison_design.json :: statistics`) cites this module;
every constant below is declared at preregistration and none is fitted to a
result.

Input: one row per (arm, seed, item) on the formal fluid static items --
``{"arm": "A" | "B", "seed": int, "item_id": str, "cluster": str, "axis": str,
"correct": bool}``. ``cluster`` is the item's rule family, with items of one
equivalence group in one cluster. Every item must be scored for both arms at
every seed (paired); anything else is refused.

* **Aggregation.** Per item, d_i = mean over seeds of (B correct - A correct):
  an item is ONE measurement unit and two seeds are two observations of it. The
  statistic is the item-weighted mean of d_i.
* **Interval.** A two-sided 95% percentile interval from a rule-family cluster
  bootstrap stratified by axis (`aadistill.evaluation.paired_stats
  .cluster_bootstrap`), 10,000 resamples, bootstrap seed ``BOOTSTRAP_SEED``. It
  covers evaluation sampling only, never seed variation.
* **Seeds.** Each seed's own point estimate (mean over items of B - A at that
  seed) is reported, and every directional label requires the seeds to agree.
* **Resolution.** An interval carries an equivalence claim or the exclusion of
  a meaningful gain only when the evaluation discriminated between the arms at
  all: at least ``MIN_DISCORDANT`` discordant (item, seed) pairs, in at least
  ``MIN_INFORMATIVE_CLUSTERS`` clusters, and a non-degenerate bootstrap. When
  successes are absent or sparse, an interval such as [0, 0] says that neither
  arm solved the items, not that the arms are equivalent.
* **Floor.** An arm whose pooled correct rate is below ``FLOOR_RATE`` is at
  the floor.

Labels, in this order:

1. ``NOT_RESOLVED`` -- resolution fails, or both arms are at the floor (then
   flagged ``FLOOR``). No equivalence, no exclusion of a gain, no GO.
2. ``GO`` -- the interval is wholly above +SESOI and every seed's point
   estimate is above +SESOI.
3. ``REVERSED`` -- wholly below -SESOI, and every seed below -SESOI.
4. ``NO_GO`` -- wholly inside (-SESOI, +SESOI), every seed inside it, and
   NEITHER arm at the floor: equivalence between an arm that cannot do the
   items and one that barely can is not evidence that the data scheme does
   not matter.
5. ``INCONCLUSIVE`` -- anything else.

``FLOOR_A`` / ``FLOOR_B`` flags go beside any label when one arm alone is at
the floor; a GO off the floor is still a GO. Axis contrasts run the same rule with ``SESOI_AXIS`` inside each
axis (clusters unstratified), and a Holm step-down over the axes at
``ALPHA_AXES`` on the one-sided bootstrap share at or below +SESOI_AXIS marks
which axes' gains survive multiplicity.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from aadistill.evaluation.paired_stats import cluster_bootstrap

RULE_ID = "stage2.reasoning_data.h1_statistics/v1"
SESOI = 0.05
SESOI_AXIS = 0.08
LEVEL = 0.95
ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261022
FLOOR_RATE = 0.02
MIN_DISCORDANT = 10
MIN_INFORMATIVE_CLUSTERS = 5
ALPHA_AXES = 0.05


def _paired(rows: Iterable[Mapping]) -> tuple[dict, list]:
    cell: dict = {}
    meta: dict = {}
    seeds: set = set()
    for r in rows:
        key = (r["item_id"], int(r["seed"]), r["arm"])
        if key in cell:
            raise ValueError(f"duplicate row {key}")
        if r["arm"] not in ("A", "B"):
            raise ValueError(f"unknown arm {r['arm']!r}")
        cell[key] = bool(r["correct"])
        m = meta.setdefault(r["item_id"], (r["cluster"], r["axis"]))
        if m != (r["cluster"], r["axis"]):
            raise ValueError(f"item {r['item_id']} has two clusters or axes")
        seeds.add(int(r["seed"]))
    seeds = sorted(seeds)
    for item in meta:
        for s in seeds:
            for arm in ("A", "B"):
                if (item, s, arm) not in cell:
                    raise ValueError(f"unpaired: {item} has no arm {arm} at seed {s}")
    items = {i: {"cluster": c, "axis": a,
                 "A": [cell[(i, s, "A")] for s in seeds], "B": [cell[(i, s, "B")] for s in seeds]}
             for i, (c, a) in sorted(meta.items())}
    return items, seeds


def _label(items: dict, seeds: list, sesoi: float, *, stratify: bool) -> dict:
    ids = sorted(items)
    n, k = len(ids), len(seeds)
    d = [sum(b - a for a, b in zip(items[i]["A"], items[i]["B"])) / k for i in ids]
    boot = cluster_bootstrap(d, [items[i]["cluster"] for i in ids],
                             [items[i]["axis"] for i in ids] if stratify else None,
                             iterations=ITERATIONS, seed=BOOTSTRAP_SEED, level=LEVEL,
                             thresholds=(sesoi,))
    rate = {arm: sum(sum(items[i][arm]) for i in ids) / (n * k) for arm in ("A", "B")}
    per_seed = {str(s): round(sum(items[i]["B"][j] - items[i]["A"][j] for i in ids) / n, 6)
                for j, s in enumerate(seeds)}
    discordant = sum(items[i]["A"][j] != items[i]["B"][j] for i in ids for j in range(k))
    informative = len({items[i]["cluster"] for i in ids
                       if any(items[i]["A"][j] != items[i]["B"][j] for j in range(k))})
    floor = {arm: rate[arm] < FLOOR_RATE for arm in rate}
    resolved = (discordant >= MIN_DISCORDANT and informative >= MIN_INFORMATIVE_CLUSTERS
                and not boot["degenerate"])
    lo, hi, pts = boot["ci_low"], boot["ci_high"], list(per_seed.values())
    if not resolved or all(floor.values()):
        label = "NOT_RESOLVED"
    elif lo > sesoi and all(p > sesoi for p in pts):
        label = "GO"
    elif hi < -sesoi and all(p < -sesoi for p in pts):
        label = "REVERSED"
    elif (-sesoi < lo and hi < sesoi and all(-sesoi < p < sesoi for p in pts)
          and not any(floor.values())):
        label = "NO_GO"
    else:
        label = "INCONCLUSIVE"
    flags = (["FLOOR"] if all(floor.values()) else
             [f"FLOOR_{arm}" for arm, f in sorted(floor.items()) if f])
    return {"label": label, "flags": flags, "sesoi": sesoi,
            "delta": boot["point"], "ci": [lo, hi], "level": LEVEL,
            "per_seed_delta": per_seed, "rates": {k2: round(v, 6) for k2, v in rate.items()},
            "resolution": {"resolved": resolved, "discordant_item_seed_pairs": discordant,
                           "informative_clusters": informative, "degenerate_bootstrap": boot["degenerate"],
                           "min_discordant": MIN_DISCORDANT, "min_informative_clusters": MIN_INFORMATIVE_CLUSTERS},
            "bootstrap": boot,
            "reading": ("no equivalence and no exclusion of a gain can be read from this comparison"
                        if label == "NOT_RESOLVED" else None)}


def decide(rows: Iterable[Mapping]) -> dict:
    """The primary label, then the axis contrasts with Holm over axes."""
    items, seeds = _paired(rows)
    primary = _label(items, seeds, SESOI, stratify=True)
    axes: dict = defaultdict(dict)
    for i, it in items.items():
        axes[it["axis"]][i] = it
    by_axis = {ax: _label(sub, seeds, SESOI_AXIS, stratify=False) for ax, sub in sorted(axes.items())}
    # Holm step-down on the one-sided share of resamples at or below +SESOI_AXIS
    p = {ax: r["bootstrap"]["tail_shares"][str(SESOI_AXIS)]["at_or_below"] for ax, r in by_axis.items()}
    order = sorted(p, key=lambda ax: (p[ax], ax))
    m, alive = len(order), True
    for rank, ax in enumerate(order):
        alive = alive and p[ax] <= ALPHA_AXES / (m - rank)
        by_axis[ax]["holm_gain_survives"] = bool(alive and by_axis[ax]["label"] == "GO")
        by_axis[ax]["one_sided_share_at_or_below_sesoi"] = p[ax]
    return {"rule": RULE_ID, "seeds": seeds, "n_items": len(items), "primary": primary,
            "by_axis": by_axis,
            "constants": {"sesoi": SESOI, "sesoi_axis": SESOI_AXIS, "level": LEVEL,
                          "iterations": ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
                          "floor_rate": FLOOR_RATE, "min_discordant": MIN_DISCORDANT,
                          "min_informative_clusters": MIN_INFORMATIVE_CLUSTERS,
                          "alpha_axes": ALPHA_AXES}}
