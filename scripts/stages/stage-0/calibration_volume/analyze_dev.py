#!/usr/bin/env python3
"""Dev-battery analysis: solvability, floor, seed noise and resolution, at $0.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/analyze_dev.py \
        --rows-dir <dir with <label>.rows.jsonl> --teacher teacher_dev \
        --pair ctl_r0860k_sa_dev ctl_r0860k_sb_dev --reference e1_r2960k_sa_pca_dev \
        --out logs/stages/stage-0/calibration_volume/design/dev_analysis.json

What it answers, each separately (no weighted scalar combines axes):

1. **Solvability** -- the teacher's verified rate per family and difficulty. A
   family the teacher cannot solve measures the task, not the student.
2. **Floor** -- the recovered students' rate per axis and difficulty, with a
   family-cluster bootstrap interval.
3. **Seed noise** -- the paired difference between two recovery seeds of one
   initialization: the discordance rate is the per-item noise a comparison of
   two arms inherits even when the arms are identical.
4. **Resolution** -- the minimal detectable effect, per axis, of an arm-vs-arm
   comparison on a final battery of a given size, from the measured seed
   discordance and the measured family design effect. It is computed, not
   assumed, and it is what a SESOI must be compared against.

Development numbers only: dev_v1 tunes difficulty and sizes the final battery;
it is never the endpoint.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

AXES = ("inductive_rule_discovery", "compositional_generalization",
        "task_contained_deduction", "abstract_analogical", "interactive_adaptation")


def load(rows_dir: Path, label: str) -> dict[str, dict]:
    return {r["task_id"]: r for r in
            (json.loads(l) for l in (rows_dir / f"{label}.rows.jsonl").read_text().splitlines())}


def rate(rows) -> dict:
    n = len(rows)
    c = sum(bool(r["correct"]) for r in rows)
    return {"n": n, "correct": c, "rate": round(c / n, 4) if n else None}


def cluster_ci(rows, key=lambda r: f"{r['generator_id']}/{r['family_id']}",
               iters=2000, seed=0):
    """Percentile interval of the rate, resampling FAMILIES (the isolation unit)."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(bool(r["correct"]))
    names = sorted(groups)
    if not names:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(iters):
        pick = [groups[rng.choice(names)] for _ in names]
        n = sum(len(g) for g in pick)
        stats.append(sum(sum(g) for g in pick) / n if n else 0.0)
    stats.sort()
    return [round(stats[int(0.025 * iters)], 4), round(stats[int(0.975 * iters) - 1], 4)]


def design_effect(rows_a, rows_b, key):
    """Variance inflation of the paired difference from clustering by family."""
    d = collections.defaultdict(list)
    for tid, ra in rows_a.items():
        rb = rows_b.get(tid)
        if rb is None:
            continue
        d[key(ra)].append(int(bool(rb["correct"])) - int(bool(ra["correct"])))
    vals = [x for g in d.values() for x in g]
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    if var == 0:
        return 1.0
    # between-cluster variance of cluster sums, relative to independent items
    k = len(d)
    sums = [sum(g) - mean * len(g) for g in d.values()]
    cl_var = sum(s * s for s in sums) * k / (k - 1) / n
    return round(max(1.0, cl_var / var), 3)


def mde(discordance: float, n_items: int, seeds_per_arm: int, deff: float,
        alpha_z=1.96, power_z=0.8416) -> float:
    """Minimal detectable paired difference in rate for one arm-vs-arm contrast.

    Two arms, each scored with `seeds_per_arm` recovery seeds on the same items;
    the arm difference is the item-mean of the seed-mean paired difference. Its
    per-item variance is bounded by the measured discordance rate (the variance
    of a difference of two binaries under the null), divided by the seed count.
    """
    var = discordance / seeds_per_arm * deff / n_items
    return round((alpha_z + power_z) * math.sqrt(var), 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--pair", nargs=2, required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--final-sizes", nargs="*", type=int, default=[400, 800, 1200, 2000])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows_dir = Path(args.rows_dir)
    teacher = load(rows_dir, args.teacher)
    a, b = (load(rows_dir, l) for l in args.pair)
    ref = load(rows_dir, args.reference) if args.reference else None
    fam = lambda r: f"{r['generator_id']}/{r['family_id']}"  # noqa: E731

    out = {"schema": "aadistill.stage0.dev_analysis/v1", "labels": {
        "teacher": args.teacher, "pair": args.pair, "reference": args.reference}}

    # 1. solvability
    solv = {}
    for key in sorted({fam(r) for r in teacher.values()}):
        sub = [r for r in teacher.values() if fam(r) == key]
        solv[key] = {"overall": rate(sub), **{d: rate([r for r in sub if r["difficulty"] == d])
                                              for d in ("L1", "L2", "L3")}}
    out["teacher_by_family"] = solv
    out["teacher_by_axis_difficulty"] = {
        f"{ax}|{d}": rate([r for r in teacher.values() if r["axis"] == ax and r["difficulty"] == d])
        for ax in AXES for d in ("L1", "L2", "L3")}

    # 2. floor
    labels = {args.pair[0]: a, args.pair[1]: b, **({args.reference: ref} if ref else {}),
              args.teacher: teacher}
    floor = {}
    for label, rows in labels.items():
        floor[label] = {}
        for ax in AXES:
            sub = [r for r in rows.values() if r["axis"] == ax]
            floor[label][ax] = {"all": {**rate(sub), "family_ci95": cluster_ci(sub)},
                                **{d: rate([r for r in sub if r["difficulty"] == d])
                                   for d in ("L1", "L2", "L3")}}
            stat = [r for r in sub if not r["interactive"]]
            if stat:
                floor[label][ax]["usable_rate"] = round(
                    sum(bool(r.get("usable")) for r in stat) / len(stat), 4)
                floor[label][ax]["answer_correct_rate"] = round(
                    sum(bool(r.get("answer_correct")) for r in stat) / len(stat), 4)
        floor[label]["overall"] = rate(list(rows.values()))
    out["rates"] = floor

    # 3. seed noise, and 4. resolution
    res = {}
    for ax in (*AXES, "all"):
        ids = [t for t in a if (ax == "all" or a[t]["axis"] == ax) and t in b]
        disc = sum(bool(a[t]["correct"]) != bool(b[t]["correct"]) for t in ids)
        q = disc / len(ids) if ids else None
        sub_a = {t: a[t] for t in ids}
        deff = design_effect(sub_a, b, fam) if ids else None
        entry = {"n": len(ids), "discordant": disc, "discordance_rate": round(q, 4) if q is not None else None,
                 "delta_b_minus_a": round((sum(bool(b[t]["correct"]) for t in ids)
                                           - sum(bool(a[t]["correct"]) for t in ids)) / len(ids), 4)
                 if ids else None,
                 "family_design_effect": deff}
        if q:
            share = len(ids) / len(a)
            entry["mde_by_final_size"] = {
                str(n): {f"seeds{s}": mde(q, max(1, round(n * (share if ax != 'all' else 1))),
                                          s, deff or 1.0) for s in (1, 2, 3)}
                for n in args.final_sizes}
        res[ax] = entry
    out["seed_noise_and_resolution"] = res
    if ref:
        out["reference_gap"] = {
            ax: round((rate([r for r in ref.values() if ax == "all" or r["axis"] == ax])["rate"] or 0)
                      - (rate([r for r in a.values() if ax == "all" or r["axis"] == ax])["rate"] or 0), 4)
            for ax in (*AXES, "all")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"overall": {k: v["overall"] for k, v in floor.items()},
                      "resolution_all": res["all"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
