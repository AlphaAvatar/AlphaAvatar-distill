#!/usr/bin/env python3
"""The layer-merge round's comparison of M-avg and M-act with D, by the registered rule ($0, dev box).

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \\
        scripts/stages/stage-1/reasoning_inheritance/layer_merge_analysis.py \\
        --d-store artifacts/stages/stage-1/reasoning_inheritance/runs/ri_diagnosis_20261113_193508/scr/store \\
        [--d-store <earlier RI-S1 store> ...] --store <this round's store> [--store <a later attempt's> ...] \\
        [--out logs/stages/stage-1/reasoning_inheritance/analyses/layer_merge.json]

The rule is `design.json :: analysis` (configs/stages/stage-1/reasoning_inheritance/layer_merge/), fixed before any
result. Primary: thinking mode, the 570 static items of the fluid formal core, M - D for each candidate, paired by item
(D's sealed RI-S1 records, rescored with the current scorer like every row here), over the 558 registered independence
groups:

* the statistic: delta = (1/570) sum_items (y_M - y_D); d_g, the summed difference of group g;
* the test (`sign_flip_test`): two-sided, paired sign-flip randomization at the group level, B = 100,000;
* Holm over the TWO registered comparisons (`holm`, m = 2 always); BETTER / WORSE / NOT DISTINGUISHED at an adjusted
  0.05;
* beside it, labeled as an unadjusted effect interval and never the decision: the percentile 95% interval over
  10,000 group-bootstrap resamples (`group_bootstrap`), and the rule-family interval RI-S1 reported.

The items are the REGISTERED ones (`registered_primary_items`: the unit's formal core minus its interactive episodes,
from the eval set and its masks, pinned by count and sha256 in the design), never whatever a store happens to hold.
A primary unit is complete when its records cover every one of them; D's must. Formal verdicts exist only when both
candidates' primary units are complete: otherwise each measured candidate is reported descriptively (its delta and
the unadjusted interval, no test, no verdict), the other `not_measured`, and the family is never shrunk to the
candidates that happen to be measured. Nothing is pooled. Everything else -- non-thinking mode, interactive episodes,
each axis, the supporting units, M-act - M-avg -- is reported, never decisive. eval-set-1 is development-exposed:
nothing here is an untouched test.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = REPO / "configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json"
OUT = REPO / "logs/stages/stage-1/reasoning_inheritance/analyses/layer_merge.json"
CANDIDATES = ("M-avg", "M-act")
MODES = ("H-T", "H-N")
TEST_SEED, BOOT_SEED = 20261114, 20261115
TEST_DRAWS, BOOT_DRAWS = 100_000, 10_000
ALPHA = 0.05


# ---- the registered statistics (pure, tested) ----------------------------------------------------------------------

def group_differences(y_m: dict, y_d: dict, groups: dict[str, list[str]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(group keys in sorted order, d_g, n_g) over the items both hold; every item of a group must be present."""
    keys = sorted(groups)
    d, n = [], []
    for k in keys:
        its = groups[k]
        missing = [i for i in its if i not in y_m or i not in y_d]
        if missing:
            raise ValueError(f"group {k}: items {missing[:3]} have no paired record")
        d.append(sum(int(y_m[i]) - int(y_d[i]) for i in its))
        n.append(len(its))
    return keys, np.asarray(d, dtype=np.int64), np.asarray(n, dtype=np.int64)


def sign_flip_test(d: np.ndarray, *, draws: int = TEST_DRAWS, seed: int = TEST_SEED, chunk: int = 10_000) -> dict:
    """Two-sided paired sign-flip test: p = (1 + #{b : |sum_g e_bg d_g| >= |sum_g d_g|}) / (1 + B), e_bg in {-1, +1}
    drawn for EVERY group, in the given group order, from numpy.random.default_rng(seed)."""
    rng = np.random.default_rng(seed)
    t = abs(int(d.sum()))
    hits, done = 0, 0
    while done < draws:
        m = min(chunk, draws - done)
        e = rng.integers(0, 2, size=(m, d.shape[0]), dtype=np.int8).astype(np.int64) * 2 - 1
        hits += int((np.abs(e @ d) >= t).sum())
        done += m
    return {"statistic_abs_sum": t, "draws": draws, "seed": seed, "exceed_or_equal": hits,
            "p_two_sided": (1 + hits) / (1 + draws), "nonzero_groups": int((d != 0).sum())}


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values: sorted ascending, p~_(i) = min(1, max_{j <= i} (m - j + 1) p_(j))."""
    order = sorted(p, key=lambda k: (p[k], k))
    m, running, out = len(order), 0.0, {}
    for j, k in enumerate(order):
        running = max(running, min(1.0, (m - j) * p[k]))
        out[k] = running
    return out


def group_bootstrap(d: np.ndarray, n: np.ndarray, *, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> dict:
    """Percentile 95% interval of the item-weighted mean difference over group resamples (with replacement,
    numpy.random.default_rng(seed); numpy's linear-interpolation percentile). Unadjusted for multiplicity."""
    rng = np.random.default_rng(seed)
    g = d.shape[0]
    idx = rng.integers(0, g, size=(draws, g))
    est = d[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {"estimate": float(d.sum() / n.sum()), "ci95_percentile": [float(lo), float(hi)], "draws": draws,
            "seed": seed, "groups": int(g), "label": "effect interval, unadjusted for multiplicity; not the decision"}


def verdict(delta: float, p_adjusted: float, alpha: float = ALPHA) -> str:
    if p_adjusted <= alpha:
        return "BETTER" if delta > 0 else "WORSE" if delta < 0 else "NOT_DISTINGUISHED"
    return "NOT_DISTINGUISHED"


def decide(y: dict[str, dict | None], y_d: dict, groups: dict[str, list[str]],
           family: tuple[str, ...] = CANDIDATES) -> dict:
    """The registered decision. ``family``: the registered comparisons (each candidate - D); Holm runs over all of
    them, so its m never shrinks. ``groups`` hold exactly the registered items. A candidate's primary unit is
    complete when ``y[name]`` covers every one; D's must. Only when every comparison of the family is complete are
    there tests and verdicts; otherwise every measured candidate is described (delta, accuracy, the unadjusted
    effect interval: no test, no verdict) and the rest are ``not_measured``."""
    items = [i for v in groups.values() for i in v]
    if any(i not in y_d for i in items):
        raise ValueError("D's records do not cover the registered primary items")
    complete = {name: y.get(name) is not None and all(i in y[name] for i in items) for name in family}
    formal = all(complete.values())
    out: dict = {"_family": {"comparisons": [f"{c} - D" for c in family], "holm_m": len(family),
                             "complete": complete, "formal": formal,
                             "status": "formal" if formal else "descriptive_only"}}
    p = {}
    for name in family:
        ym = y.get(name)
        if not complete[name]:
            out[name] = {"verdict": "not_measured", "why": "the primary unit is not complete for this candidate",
                         "items_present": 0 if ym is None else sum(i in ym for i in items), "items_registered": len(items)}
            continue
        keys, d, n = group_differences(ym, y_d, groups)
        out[name] = {"n_items": int(n.sum()), "n_groups": len(keys), "delta": float(d.sum() / n.sum()),
                     "accuracy": {"candidate": sum(int(ym[i]) for i in items) / len(items),
                                  "D": sum(int(y_d[i]) for i in items) / len(items)},
                     "effect_interval": group_bootstrap(d, n)}
        if formal:
            out[name]["test"] = sign_flip_test(d)
            p[name] = out[name]["test"]["p_two_sided"]
        else:
            out[name]["verdict"] = "descriptive_only"
            out[name]["why"] = ("the other registered comparison's primary unit is not complete: no test and no "
                                "verdict (the Holm family is never shrunk to the measured candidates)")
    if formal:
        adj = holm(p)
        for name, pa in adj.items():
            out[name]["p_holm"] = pa
            out[name]["verdict"] = verdict(out[name]["delta"], pa)
    return out


# ---- reading the records ---------------------------------------------------------------------------------------------

def registered_primary_items() -> list[str]:
    """The primary comparison's items, from the registered set alone: the fluid unit's formal core (the set's validity
    and core masks, `eval_set_scoring.unit_masks`) minus its interactive episodes; refused unless they are the count
    and sha256 the design registered."""
    import shared.evaluation.eval_set_scoring as S
    from shared import eval_sets as ES
    want = json.loads(DESIGN.read_text())["analysis"]["primary_items"]
    comp = ES.manifest("eval-set-1")["components"]["fluid_reasoning_720"]
    mask = S.unit_masks("eval-set-1", "fluid_reasoning_720", comp)["dev_v1"]
    tasks = {t["task_id"]: t for t in ES.items("eval-set-1", "fluid_reasoning_720", "dev_v1")}
    ids = sorted(i for i in set(tasks) - set(mask["exclude"]) if not tasks[i]["interactive"])
    got = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if len(ids) != int(want["n"]) or got != want["sha256"]:
        raise SystemExit(f"the registered primary items are {len(ids)} / {got[:12]}, the design pins "
                         f"{want['n']} / {want['sha256'][:12]}")
    return ids


def independence_groups(items: list[str]) -> dict[str, list[str]]:
    from shared import eval_sets as ES
    fluid_path = ES.unit_path("eval-set-1", "fluid_reasoning_720", "dev_v1")
    p = fluid_path.with_name("independence_v1.json")
    want = json.loads(DESIGN.read_text())["analysis"]["units_of_independence"]
    if hashlib.sha256(p.read_bytes()).hexdigest() not in want:
        raise SystemExit(f"{p} is not the registered independence file")
    doc = json.loads(p.read_text())
    in_group = {i: k for k, grp in enumerate(doc["groups"]) for i in grp["ids"]}
    out = collections.defaultdict(list)
    for i in items:
        out[f"g{in_group[i]}" if i in in_group else i].append(i)
    return dict(out)


def gens(name: str, mode: str, stores: list[Path]) -> Path | None:
    for st in reversed(stores):
        d = st / "out" / "eval" / f"{name}_{mode}" / "generations"
        if d.is_dir():
            return d
    return None


def static_core(gd: Path, label: str, tmp: Path) -> tuple[dict, dict] | None:
    """(static formal-core correctness by item, interactive formal-core correctness by item), rescored."""
    from stages.reasoning_inheritance import analysis as RA
    fr = RA.fluid_rows(gd, label, tmp)
    if fr is None:
        return None
    _rep, rows = RA.unit_rows(gd, label, tmp)
    formal = rows.get(RA.FLUID, {})
    core = [fr[i] for i in fr if (formal.get(i) or {}).get("formal")]
    return ({r["task_id"]: int(bool(r["correct"])) for r in core if not r["interactive"]},
            {r["task_id"]: int(bool(r["correct"])) for r in core if r["interactive"]})


def unit_headlines(gd: Path, label: str) -> dict:
    """Every registered unit's headline as the registered scorer reports it (`eval_set_scoring.report`), with the
    termination rates it carries: descriptive, reported beside the decision and never part of it."""
    import shared.evaluation.eval_set_scoring as S
    rep = S.report(gd, label, {}, "eval-set-1")
    out = {}
    for unit, v in rep["units"].items():
        m = v.get("metrics") or {}
        head = v.get("headline")
        val = v.get("rate") if head == "correct" and "rate" in v else (m.get(head) if head else None)
        out[unit] = {"status": v.get("status"), "n": v.get("n"), "in_formal_core": v.get("in_formal_core"),
                     "headline": head, "value": val,
                     "natural_termination_rate": m.get("natural_termination_rate"),
                     "degeneration_rate": m.get("degeneration_rate")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d-store", action="append", default=[], help="RI-S1 stores holding D's sealed records")
    ap.add_argument("--store", action="append", default=[], help="this round's stores (later ones win per unit)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    from stages.reasoning_inheritance import analysis as RA
    from shared import eval_sets as ES
    fluid_items = {t["task_id"]: t for t in ES.items("eval-set-1", "fluid_reasoning_720", "dev_v1")}
    fam = collections.defaultdict(list)
    for i, t in fluid_items.items():
        fam[t["family_id"]].append(i)
    d_stores, stores = [Path(s) for s in a.d_store], [Path(s) for s in a.store]
    res: dict = {"schema": "aadistill.stage1.reasoning_inheritance.layer_merge.analysis/v1",
                 "eval_set": "eval-set-1 (development-exposed; not an untouched test)",
                 "rule": "configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json :: analysis",
                 "inputs": {}, "primary": {}, "secondary": {}}
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        rows: dict = {}
        for mode in MODES:
            for name in ("D", *CANDIDATES):
                gd = gens(name, mode, d_stores if name == "D" else stores)
                if gd is None:
                    rows[(name, mode)] = None
                    continue
                res["inputs"][f"{name}_{mode}"] = {"generations": str(gd), "sha256": RA.dir_digest(gd)}
                rows[(name, mode)] = static_core(gd, f"{name}_{mode}", tmp)
        d_ht = rows.get(("D", "H-T"))
        if d_ht is None:
            raise SystemExit("D's thinking-mode fluid records are not in --d-store")
        items = registered_primary_items()
        lost = [i for i in items if i not in d_ht[0]]
        if lost:
            raise SystemExit(f"D's sealed records lack {len(lost)} registered primary items ({lost[:3]})")
        groups = independence_groups(items)
        res["primary"] = {"mode": "H-T", "unit": "fluid static formal core", "n_items": len(items),
                          "items_sha256": json.loads(DESIGN.read_text())["analysis"]["primary_items"]["sha256"],
                          "n_groups": len(groups),
                          "comparisons": decide({c: (rows[(c, "H-T")] or (None,))[0] for c in CANDIDATES}, d_ht[0],
                                                groups)}
        for mode in MODES:
            dm = rows.get(("D", mode))
            if dm is None:
                continue
            for c in CANDIDATES:
                cm = rows.get((c, mode))
                if cm is None:
                    res["secondary"].setdefault(mode, {})[c] = {"status": "not_run"}
                    continue
                entry = {}
                for part, k in (("static", 0), ("interactive", 1)):
                    common = sorted(set(cm[k]) & set(dm[k]))
                    if not common:
                        continue
                    acc = {"M": {i: cm[k][i] for i in common}, "D": {i: dm[k][i] for i in common}}
                    cl = {f: [i for i in v if i in set(common)] for f, v in fam.items()}
                    cl = {f: v for f, v in cl.items() if v}
                    entry[part] = {"n": len(common),
                                   "rule_family_interval": RA.bootstrap(
                                       cl, acc, lambda s: sum(acc["M"][i] - acc["D"][i] for i in s) / len(s),
                                       n=10_000, seed=20261113)}
                res["secondary"].setdefault(mode, {})[c] = entry
        #: every registered unit, per model and mode, by the registered scorer: reported, never decisive
        res["supporting"] = {"label": "every registered unit's headline by eval_set_scoring.report; descriptive, "
                                      "never decisive (design.json :: analysis.supplementary)", "by_mode": {}}
        for mode in MODES:
            for name in ("D", *CANDIDATES):
                gd = gens(name, mode, d_stores if name == "D" else stores)
                if gd is not None:
                    res["supporting"]["by_mode"].setdefault(mode, {})[name] = unit_headlines(gd, f"{name}_{mode}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"status": res["primary"]["comparisons"]["_family"]["status"],
                      **{c: res["primary"]["comparisons"][c].get("verdict") for c in CANDIDATES}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
