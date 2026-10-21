#!/usr/bin/env python3
"""Can fluid_reasoning_720 resolve models in the student's capacity range? At $0.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/analyze_resolution.py \
        --rows <label>=<score.rows.jsonl> ... \
        --out logs/stages/stage-0/calibration_volume/design/resolution_check.json

Reads the scorer's per-task rows for every model measured on the set (the
teacher, the recovered students at four recovery doses, and the released
same-geometry reference) and reports, per model and per axis, each separately:

* correct counts and rates, static and interactive apart, with a rule-family
  cluster bootstrap interval;
* for every pair of models, the paired discordance (tasks one gets right and the
  other wrong) and the paired difference with its family-cluster interval --
  the quantity a treatment comparison would read;
* the ordering of the recovered students by recovery dose.

The verdict is descriptive and computed: whether the set separates the
same-geometry reference from the floor, and whether the recovered students
move off the floor with recovery dose. It never claims anything about a
D1-initialized student, which none of these is.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
from pathlib import Path

AXES = ("inductive_rule_discovery", "compositional_generalization",
        "task_contained_deduction", "abstract_analogical", "interactive_adaptation")
STATIC = AXES[:4]
B = 10000
SEED = 20261019


def load(path: Path) -> dict[str, dict]:
    return {r["task_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def families(rows: dict[str, dict]) -> dict[str, list[str]]:
    fam = collections.defaultdict(list)
    for tid, r in rows.items():
        fam[f"{r['generator_id']}/{r['family_id']}"].append(tid)
    return dict(fam)


def boot(values: dict[str, float], fams: dict[str, list[str]], rng: random.Random) -> list[float]:
    keys = [f for f in fams if any(t in values for t in fams[f])]
    if not keys:
        return [0.0, 0.0]
    stats = []
    for _ in range(B):
        pick = [rng.choice(keys) for _ in keys]
        num = den = 0.0
        for f in pick:
            for t in fams[f]:
                if t in values:
                    num += values[t]
                    den += 1
        stats.append(num / den if den else 0.0)
    stats.sort()
    return [round(stats[int(0.025 * B)], 4), round(stats[int(0.975 * B) - 1], 4)]


def subset(rows: dict[str, dict], axes) -> list[str]:
    return [t for t, r in rows.items() if r["axis"] in axes]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True, help="label=path")
    ap.add_argument("--doses", nargs="*", default=[], help="label=recovery tokens (students)")
    ap.add_argument("--reference", default=None, help="the same-geometry released model's label")
    ap.add_argument("--floor-static-rate", type=float, default=0.02)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rng = random.Random(SEED)
    data = {k: load(Path(v)) for k, v in (x.split("=", 1) for x in args.rows)}
    ids = set.intersection(*(set(d) for d in data.values()))
    any_rows = next(iter(data.values()))
    fams = families({t: any_rows[t] for t in ids})

    per_model = {}
    for label, rows in data.items():
        out = {}
        for name, axes in (("static", STATIC), ("interactive", AXES[4:]),
                           *((a, (a,)) for a in AXES)):
            tids = [t for t in subset(rows, axes) if t in ids]
            vals = {t: float(bool(rows[t]["correct"])) for t in tids}
            c = int(sum(vals.values()))
            out[name] = {"n": len(tids), "correct": c,
                         "rate": round(c / len(tids), 4) if tids else None,
                         "ci95_family_bootstrap": boot(vals, fams, rng)}
        out["all"] = {"n": len(ids), "correct": sum(bool(rows[t]["correct"]) for t in ids)}
        per_model[label] = out

    pairs = {}
    for a, b in itertools.combinations(sorted(data), 2):
        tids = [t for t in subset(data[a], STATIC) if t in ids]
        diff = {t: float(bool(data[b][t]["correct"])) - float(bool(data[a][t]["correct"]))
                for t in tids}
        disc = sum(1 for v in diff.values() if v != 0)
        pairs[f"{b} - {a}"] = {"static_n": len(tids), "discordant": disc,
                               "discordance_rate": round(disc / len(tids), 4),
                               "delta_static": round(sum(diff.values()) / len(tids), 4),
                               "ci95_family_bootstrap": boot(diff, fams, rng)}

    doses = dict(x.split("=", 1) for x in args.doses)
    ladder = sorted(((float(v.rstrip("M")), k) for k, v in doses.items() if k in per_model))
    ladder_rates = [(f"{d}M", k, per_model[k]["static"]["rate"]) for d, k in ladder]
    floor = args.floor_static_rate
    ref = per_model.get(args.reference) if args.reference else None
    verdict = {
        "floor_static_rate": floor,
        "reference_off_floor": (ref["static"]["ci95_family_bootstrap"][0] > floor) if ref else None,
        "students_off_floor": {k: per_model[k]["static"]["ci95_family_bootstrap"][0] > floor
                               for _, k in ladder},
        "student_static_rate_by_recovery_dose": ladder_rates,
        "reading": None,
    }
    if ref is not None:
        off = verdict["students_off_floor"]
        verdict["reading"] = (
            ("the set separates the same-geometry reference from the floor "
             f"(static {ref['static']['rate']}, lower bound {ref['static']['ci95_family_bootstrap'][0]})"
             if verdict["reference_off_floor"] else
             "even the same-geometry reference is not separable from the floor: the set's "
             "difficulty is above student scale")
            + "; recovered students off the floor: "
            + (", ".join(k for k, v in off.items() if v) or "none")
            + ". Every student here is PCA-initialized; a D1-initialized student is unmeasured.")
    doc = {"schema": "aadistill.stage0.resolution_check/v1",
           "set": "eval-set-1/fluid_reasoning_720/dev_v1", "tasks": len(ids),
           "bootstrap": {"resamples": B, "unit": "rule family", "seed": SEED},
           "per_model": per_model, "pairs_static": pairs, "verdict": verdict,
           "development_numbers_only": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: (v["static"]["correct"], v["static"]["rate"], v["interactive"]["correct"])
                      for k, v in per_model.items()}, indent=1))
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
