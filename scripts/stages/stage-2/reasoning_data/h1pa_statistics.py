"""H1-PA's decision rule, executable before any result exists. $0.

The preregistration (`plans/h1pa_preregistration.json`) binds this file's
sha256, the sha256 of the rule it applies (`h1_statistics.py`, H1's frozen
rule, imported unchanged rather than re-typed) and of the bootstrap it calls
(`aadistill.evaluation.paired_stats`); `decide` refuses to run when any of
them differs from what was preregistered, so the rule cannot change after a
result is seen.

Input: one row per (arm, seed, item) on the formal fluid STATIC items --
``{"arm": "PCA-Original" | "D4-Original" | "D4-Augmented", "seed": int,
"item_id": str, "cluster": str, "axis": str, "correct": bool,
"decoding_sha256": str}``. ``cluster`` is the item's rule family. Every row
must have been generated under the preregistered decoding protocol
(``nucleus_v1``): a greedy row, or a row with no decoding identity, refuses --
greedy and nucleus results are never mixed, E1's and H1's greedy results
included.

Comparisons (Y - X, paired by item and by training-seed number):

* **A** = D4-Original - PCA-Original: the initialization effect. DECIDED.
* **B** = D4-Augmented - D4-Original: adding the fluid increment together
  with the optimizer steps and FLOPs it brings. DECIDED; the study's
  question. Never read as the data content's isolated causal effect.
* **C** = D4-Augmented - PCA-Original: end to end. DESCRIPTIVE: its
  estimate, interval, per-seed estimates and rates are reported, and no label
  is assigned or may be cited.

A and B each answer their own question; no claim combines them, so each
keeps its own 95% interval. Within a decided comparison the four static-axis
contrasts are a Holm family (`h1_statistics.decide`).

For each decided comparison `h1_statistics.decide` applies H1's frozen rule
with X in the rule's arm "A" and Y in its arm "B": the item-weighted mean of
per-item seed-mean differences; a 95% percentile interval from a rule-family
cluster bootstrap stratified by axis (10,000 resamples); SESOI 0.05 overall
and 0.08 per axis; the 2% floor; resolution (>= 10 discordant item-seed pairs
in >= 5 clusters, non-degenerate bootstrap); labels NOT_RESOLVED, GO,
REVERSED, NO_GO, INCONCLUSIVE in that order, each directional label requiring
every seed's own estimate on its side; FLOOR flags. `LABELS` says what each
label licenses here.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path

from stages.reasoning_data import h1_statistics as HS

RULE_ID = "stage2.reasoning_data.h1pa_statistics/v1"
REPO = Path(__file__).resolve().parents[4]
ARMS = ("PCA-Original", "D4-Original", "D4-Augmented")
#: name -> (X, Y, role): the contrast is Y - X
COMPARISONS = {"A": ("PCA-Original", "D4-Original", "decided"),
               "B": ("D4-Original", "D4-Augmented", "decided"),
               "C": ("PCA-Original", "D4-Augmented", "descriptive")}
PRIMARY = "B"
#: the files whose bytes ARE the rule; the preregistration binds each sha256
BOUND = ("scripts/stages/stage-2/reasoning_data/h1pa_statistics.py",
         "scripts/stages/stage-2/reasoning_data/h1_statistics.py",
         "src/aadistill/evaluation/paired_stats.py")
LABELS = {
    "GO": "the interval lies wholly above +SESOI and every seed's estimate is above +SESOI: for B, adding the fluid "
          "increment with its compute raises static fluid correctness by a meaningful amount; for A, D4 Top-1 "
          "initializes better than PCA under E1's recipe",
    "NO_GO": "the interval and every seed's estimate lie inside (-SESOI, +SESOI) and neither arm is at the floor: a "
             "meaningful effect is excluded at this scale",
    "REVERSED": "the interval lies wholly below -SESOI and every seed's estimate is below -SESOI: a meaningful harm",
    "INCONCLUSIVE": "the evaluation resolved but the interval or a seed crosses a SESOI boundary: no decision either "
                    "way, and no claim of an effect or of its absence",
    "NOT_RESOLVED": "fewer than 10 discordant item-seed pairs or 5 informative clusters, a degenerate bootstrap, or "
                    "both arms below 2% (flag FLOOR): the evaluation did not discriminate between the arms; no "
                    "equivalence, no exclusion of a gain, no GO",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_identity(repo_root: Path = REPO) -> dict:
    return {f: sha256(Path(repo_root) / f) for f in BOUND}


def check_bound(prereg: Mapping, repo_root: Path = REPO) -> None:
    got, want = bound_identity(repo_root), prereg["decision_rule"]["bound_sha256"]
    bad = sorted(f for f in set(got) | set(want) if got.get(f) != want.get(f))
    if bad:
        raise ValueError(f"the decision rule differs from the preregistered one in {bad}; it is never changed "
                         "after the preregistration")


def _rows_for(rows: list[Mapping], x: str, y: str) -> list[dict]:
    return [{**r, "arm": {x: "A", y: "B"}[r["arm"]]} for r in rows if r["arm"] in (x, y)]


def decide(rows: Iterable[Mapping], *, prereg: Mapping, repo_root: Path = REPO) -> dict:
    """Every comparison under the preregistered rule; C carries no label."""
    check_bound(prereg, repo_root)
    want = prereg["evaluation"]["decoding"]["sha256"]
    rows = list(rows)
    bad = sorted({str(r.get("decoding_sha256"))[:12] for r in rows if r.get("decoding_sha256") != want})
    if bad:
        raise ValueError(f"rows generated under decoding {bad}, not the preregistered {want[:12]}: greedy and "
                         "nucleus results are never mixed")
    unknown = sorted({r["arm"] for r in rows} - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms {unknown}")
    out = {}
    for name, (x, y, role) in COMPARISONS.items():
        res = HS.decide(_rows_for(rows, x, y))
        res["contrast"] = f"{y} - {x}"
        res["role"] = role
        if role == "descriptive":
            p = res["primary"]
            res = {"contrast": res["contrast"], "role": role, "seeds": res["seeds"], "n_items": res["n_items"],
                   "delta": p["delta"], "ci": p["ci"], "level": p["level"], "per_seed_delta": p["per_seed_delta"],
                   "rates": p["rates"], "flags": p["flags"],
                   "by_axis": {ax: {k: v[k] for k in ("delta", "ci", "per_seed_delta", "rates")}
                               for ax, v in res["by_axis"].items()},
                   "label": None, "reading": "descriptive only: initialization and data change together"}
        out[name] = res
    return {"rule": RULE_ID, "applies": HS.RULE_ID, "primary_comparison": PRIMARY, "comparisons": out,
            "primary_label": out[PRIMARY]["primary"]["label"], "labels": LABELS}
