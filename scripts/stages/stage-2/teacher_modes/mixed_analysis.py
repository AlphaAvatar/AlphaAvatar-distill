#!/usr/bin/env python3
"""A-MIX-6cat-sr's result on eval-set-1, each checkpoint in both of its teacher's modes, every axis apart. $0, dev box.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.mixed_analysis \\
        --units <dir holding <unit>/{train,evaluation_thinking,evaluation_non_thinking}> \\
        --single-units <dir holding A1-6cat-sr-s* and A2-6cat-sr-s*/evaluation> [--out <json>]

Maintainer 2026-11-09 (UTC), section IV. Per unit and mode, as `stage3_sr_analysis` reports a unit: the 570 formal
static fluid items pooled and by axis (the four static axes, abstract/analogical as its own sub-category), correctness
among usable rollouts beside the usable rate, the 144 interactive episodes (exploratory), rollout integrity, and every
supporting capability as the formal report measured it. Per mode, the two seeds' item-paired mean.

Comparisons, all DESCRIPTIVE, each item-paired over both seeds with the registered statistics
(`mode_statistics.compare`: rule-family cluster bootstrap stratified by axis, SESOI 5 pp, per-axis Holm at 8 pp) but
never a registered decision: Mixed-thinking - A1-6cat-sr, Mixed-non-thinking - A2-6cat-sr, and the mixed checkpoint's
thinking - non-thinking (identical weights). A mixed - single difference confounds mode mixing with about twice the
unique data and a different compute and exposure, which are reported beside it from the frozen records. The registered
A1 - A2 (INCONCLUSIVE) and A1 - C1 (NO_GO) are not recomputed and do not change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data.h1_analysis import STATIC_AXES, capabilities, fluid_meta  # noqa: E402
from stages.teacher_modes import a1_analysis as AA  # noqa: E402
from stages.teacher_modes import mixed_packs as MP  # noqa: E402
from stages.teacher_modes import mixed_training_plan as MTP  # noqa: E402
from stages.teacher_modes import mode_statistics as MS  # noqa: E402
from stages.teacher_modes import stage3_sr_analysis as SA  # noqa: E402

OUT = REPO / "logs/stages/stage-2/teacher_modes/analyses/mixed_6cat_sr_result.json"
SINGLE = ("A1-6cat-sr", "A2-6cat-sr")
#: (X, Y, why): the contrast is Y - X; every one descriptive
COMPARISONS = {"mixed_thinking_minus_A1": ("A1-6cat-sr", f"{MP.ARM}/thinking",
                                           "the same thinking documents plus A2's non-thinking ones; ~2x unique data"),
               "mixed_non_thinking_minus_A2": ("A2-6cat-sr", f"{MP.ARM}/non_thinking",
                                               "the same non-thinking documents plus A1's thinking ones"),
               "mixed_thinking_minus_mixed_non_thinking": (f"{MP.ARM}/non_thinking", f"{MP.ARM}/thinking",
                                                           "one checkpoint, identical weights, two template states")}


def label(arm: str, mode: str | None) -> str:
    return arm if mode is None else f"{arm}/{mode}"


def unit_report(ev: dict, items: dict) -> tuple[dict, dict]:
    f = AA.fluid(ev, items)
    f["_axis"] = {i: items[i]["axis"] for i in f["item_correct"]}
    caps = capabilities(ev["formal_report"])
    return f, {"fluid": {k: v for k, v in f.items() if k not in ("item_correct", "_axis")},
               "worst_static_axis": SA.worst(f["by_axis"]), "usable": SA.usable_correct(ev, items),
               "capabilities": caps, "worst_unit_per_capability": {c: SA.worst(v["units"]) for c, v in caps.items()}}


def collect(units_dir: Path, single_dir: Path, items: dict) -> tuple[dict, list[dict], dict]:
    """``(per unit, decision rows, {label: [fluid]})`` for the mixed units' modes and the single-mode units."""
    plan = json.loads(MTP.BOUND.read_text())
    per, rows, by = {}, [], defaultdict(list)
    for u in plan["units"]:
        d = Path(units_dir) / u["unit"]
        out = {"seed": u["seed"], "modes": {}, "training": AA.training(d) if d.is_dir() else {"status": "not_run"}}
        for e in u["evaluations"]:
            ed = d / e["dir"]
            if not (ed / "formal_report.json").is_file():
                out["modes"][e["mode"]] = {"status": "not_run"}
                continue
            f, rep = unit_report(AA.evidence(ed), items)
            lab = label(u["arm"], e["mode"])
            rows += SA.decision_rows(u["unit"], lab, u["seed"], f, items)
            by[lab].append(f)
            out["modes"][e["mode"]] = {"template": e["template"], **rep}
        per[u["unit"]] = out
    for arm in SINGLE:
        for seed in MTP.SEEDS:
            d = Path(single_dir) / f"{arm}-s{seed}"
            if not (d / "evaluation" / "formal_report.json").is_file():
                continue
            f, _ = unit_report(AA.evidence(d), items)
            rows += SA.decision_rows(f"{arm}-s{seed}", arm, seed, f, items)
            by[arm].append(f)
    return per, rows, by


def data_and_compute() -> dict:
    mix = json.loads(MP.FREEZE.read_text())["arms"][MP.ARM]
    src = json.loads(MP.SOURCE_FREEZE.read_text())["a1_a2_compute"]
    return {"A-MIX-6cat-sr": {"before_packing": mix["before_packing"]["training"], **mix["compute"],
                              "exposure_by_mode": mix["packing"]["by_mode"]},
            "A1-6cat-sr_and_A2-6cat-sr": src,
            "_": "training FLOPs follow packed sequences x epochs (one student geometry, 8,192 causal positions)"}


def analyse(units_dir: Path, single_dir: Path) -> dict:
    prereg = json.loads(SA.PREREG.read_text())
    MS.check_bound(prereg)
    items, _ = fluid_meta()
    per, rows, by = collect(units_dir, single_dir, items)
    modes = {lab: {"seeds": len(fs), "static_all": AA.paired_mean(fs, None) if len(fs) == 2 else None,
                   "by_axis": {ax: AA.paired_mean(fs, ax) for ax in STATIC_AXES} if len(fs) == 2 else None}
             for lab, fs in by.items()}
    comps = {}
    for key, (x, y, why) in COMPARISONS.items():
        if len(by.get(x, [])) == 2 and len(by.get(y, [])) == 2:
            c = MS.compare(rows, x, y)
            comps[key] = {"role": "descriptive", "why": why, **c, "reading": SA.READING[c["primary"]["label"]]}
        else:
            comps[key] = {"role": "descriptive", "why": why, "status": "not_run"}
    return {"schema": "aadistill.stage2.teacher_modes.mixed_result/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "plan_sha256": hashlib.sha256(MTP.BOUND.read_bytes()).hexdigest(),
            "set": "eval-set-1 v2 (frozen), scoring protocol v2, validity mask v1: 570 static + 144 interactive formal",
            "decoding": {"protocol_id": "nucleus_v1", "sha256": AA.DECODING_SHA},
            "statistics": {"id": MS.RULE_ID, "bound_sha256": MS.bound_identity(), "used_as": "descriptive only"},
            "units": per, "by_mode": modes, "comparisons": comps, "data_and_compute": data_and_compute(),
            "registered_verdicts_unchanged": {"A1_minus_A2": "INCONCLUSIVE", "A1_minus_C1": "NO_GO",
                                              "owner": "logs/stages/stage-2/teacher_modes/analyses/stage3_sr_result.json"},
            "limits": ["every mixed comparison is descriptive: about twice the unique supervised tokens, A1 + A2's "
                       "training FLOPs and 4.5x / 1.3x A1's / A2's optimizer steps; never a pure mode-mixing effect",
                       "two training seeds; the bootstrap is over items and rule families, not over seeds",
                       "interactive episodes are exploratory and outside every comparison"],
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", required=True)
    ap.add_argument("--single-units", required=True)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    doc = analyse(Path(a.units), Path(a.single_units))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({k: (v["primary"]["label"], v["primary"]["delta"], v["primary"]["ci"]) if "primary" in v
                      else v.get("status") for k, v in doc["comparisons"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
