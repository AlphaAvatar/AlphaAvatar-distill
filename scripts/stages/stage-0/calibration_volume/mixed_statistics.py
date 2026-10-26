"""The mixed-calibration study's decision rule, executable before any result exists. $0.

The preregistration (`plans/mixed_preregistration.json`) binds this file's
sha256, the sha256 of the rule it applies (H1's frozen
`stages.reasoning_data.h1_statistics`, imported unchanged rather than
re-typed) and of the bootstrap it calls (`aadistill.evaluation.paired_stats`);
`decide` refuses to run when any of them differs, so the rule cannot change
after a result is seen.

Input: one row per (arm, seed, item) on the formal fluid STATIC items --
``{"arm": "PCA-Original" | "D4-Augmented" | "Mix-Augmented", "seed": int,
"item_id": str, "cluster": str, "axis": str, "correct": bool,
"decoding_sha256": str}``; ``cluster`` is the item's rule family. Every row
must carry the preregistered decoding identity (``nucleus_v1``).
D4-Augmented's and PCA-Original's rows are H1-PA's, reused as they were
scored; only Mix-Augmented's are new.

Comparisons (Y - X, paired by item and by training-seed number):

* **M** = Mix-Augmented - D4-Augmented. DECIDED; the study's question. The
  two arms differ only in the initialization the recovery starts from:
  pack, recipe, schedule, seeds, hardware class and evaluation are
  D4-Augmented's. The initialization itself changes calibration composition,
  volume, interactive coverage and the per-operator profile choice together,
  so M is never read as any one of those factors' isolated effect.
* **E** = Mix-Augmented - PCA-Original. DESCRIPTIVE: estimate, interval,
  per-seed estimates and rates; no label is assigned or may be cited.

Within M the four static-axis contrasts are a Holm family
(`h1_statistics.decide`). Interactive episodes are exploratory and refused
here, as in H1 and H1-PA.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path

from stages.reasoning_data import h1_statistics as HS

RULE_ID = "stage0.calibration_volume.mixed_statistics/v1"
REPO = Path(__file__).resolve().parents[4]
ARMS = ("PCA-Original", "D4-Augmented", "Mix-Augmented")
#: name -> (X, Y, role): the contrast is Y - X
COMPARISONS = {"M": ("D4-Augmented", "Mix-Augmented", "decided"),
               "E": ("PCA-Original", "Mix-Augmented", "descriptive")}
PRIMARY = "M"
#: the files whose bytes ARE the rule; the preregistration binds each sha256
BOUND = ("scripts/stages/stage-0/calibration_volume/mixed_statistics.py",
         "scripts/stages/stage-2/reasoning_data/h1_statistics.py",
         "src/aadistill/evaluation/paired_stats.py")
LABELS = {
    "GO": "the interval lies wholly above +SESOI and every seed's estimate is above +SESOI: the mixed-calibration "
          "Top-1 recovers meaningfully better static fluid correctness than D4 Top-1 under D4-Augmented's recovery",
    "NO_GO": "the interval and every seed's estimate lie inside (-SESOI, +SESOI) and neither arm is at the floor: a "
             "meaningful difference between the two initializations is excluded at this scale",
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
    """M under the preregistered rule; E carries no label."""
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
                   "label": None, "reading": "descriptive only: initialization and recovery data change together"}
        out[name] = res
    return {"rule": RULE_ID, "applies": HS.RULE_ID, "primary_comparison": PRIMARY, "comparisons": out,
            "primary_label": out[PRIMARY]["primary"]["label"], "labels": LABELS}
