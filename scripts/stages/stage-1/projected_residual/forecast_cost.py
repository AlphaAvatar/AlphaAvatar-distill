#!/usr/bin/env python3
"""Price Research B from measured D4-Augmented sessions and the architecture's FLOPs ($0).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-1/projected_residual/forecast_cost.py \\
        --write logs/stages/stage-1/projected_residual/plans/cost_forecast.json

Measured inputs, each read from its owner: D4-Augmented's and Mix-Augmented's
per-unit training and evaluation seconds (mixed_result.json), MC2's elapsed
minutes and price (its closeout). Scaled by the architecture report's ratios:
training by the FLOPs ratio with the teacher's forward (expected) and
student-only (conservative); evaluation by the folded model's inference FLOPs
ratio (expected) and its bf16 weight-bytes ratio (conservative, the
bandwidth-bound decode). A forecast, never a measurement: the engineering
validation measures step time and evaluation throughput before the formal
session's cap is issued.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MIXED = "logs/stages/stage-0/calibration_volume/analyses/mixed_result.json"
MC2 = "logs/stages/stage-0/calibration_volume/runs/mixr_20261026_140539/closeout/outcome.json"
REPORT = "logs/stages/stage-1/projected_residual/analyses/architecture_report.json"
DISK_USD_PER_GB_MONTH, DISK_GB = 0.1, 150


def load(rel):
    return json.loads((REPO / rel).read_text())


def ceil_to(x, q=0.5):
    return math.ceil(x / q) * q


def build() -> dict:
    units = load(MIXED)["units"]
    d4 = [u["seconds"] for k, u in units.items() if k.startswith("D4-Augmented")]
    mix = [u["seconds"] for k, u in units.items() if k.startswith("Mix-Augmented")]
    train = sum(s["train"] for s in d4) / len(d4)
    evaluate = sum(s["evaluate"] for s in d4) / len(d4)
    mc2 = load(MC2)["money"]
    overhead_min = mc2["elapsed_minutes"] - sum(s["train"] + s["evaluate"] for s in mix) / 60
    rate = mc2["price_per_hour"] + DISK_USD_PER_GB_MONTH * DISK_GB / 720
    rep = load(REPORT)
    r = rep["pw_900m_vs_d4"]
    m = rep["models"]
    eval_expected = m["pw_900m_folded"]["forward_gflops_per_token"]["4096"] / m["d4_top1_596m"]["forward_gflops_per_token"]["4096"]
    eval_conservative = m["pw_900m_folded"]["weights_gb_bf16"] / m["d4_top1_596m"]["weights_gb_bf16"]
    scen = {}
    for name, tr, ev, mat_min, oh in (("expected", r["training_with_teacher"], eval_expected, 15, overhead_min),
                                      ("conservative", r["training_student"], eval_conservative, 30, 20)):
        per_seed = train * tr + evaluate * ev
        seconds = 2 * per_seed + 60 * (mat_min + oh)
        hours = seconds / 3600
        scen[name] = {"train_seconds_per_seed": round(train * tr), "evaluate_seconds_per_seed": round(evaluate * ev),
                      "materialization_minutes": mat_min, "overhead_minutes": round(oh, 1),
                      "hours": round(hours, 2), "usd": round(hours * rate, 2)}
    hard = ceil_to(scen["conservative"]["usd"] * 1.12)
    eng = {"hours_expected": 1.0, "hours_conservative": 2.0}
    eng["usd_expected"], eng["usd_conservative"] = round(eng["hours_expected"] * rate, 2), round(eng["hours_conservative"] * rate, 2)
    return {
        "schema": "aadistill.stage1.projected_residual.cost_forecast/v1",
        "inputs": {"d4_augmented_mean_train_seconds": round(train, 1), "d4_augmented_mean_evaluate_seconds": round(evaluate, 1),
                   "mc2_overhead_minutes": round(overhead_min, 1), "l40s_usd_per_hour_all_in": round(rate, 4),
                   "_sources": [MIXED, MC2, REPORT],
                   "ratios": {"train_expected (FLOPs with teacher)": r["training_with_teacher"],
                              "train_conservative (student-only FLOPs)": r["training_student"],
                              "evaluate_expected (folded inference FLOPs at 4,096)": round(eval_expected, 4),
                              "evaluate_conservative (bf16 weight bytes)": round(eval_conservative, 4)}},
        "B_E_engineering_validation": {**eng, "hard_cap_usd": 3.0, "gpu": "NVIDIA L40S",
                                       "what": "design.json's initialization on the real teacher, the fold check in bf16, a "
                                               "20-step trainer smoke at the recipe's shape (step time, peak memory), "
                                               "save/load/resume on CUDA, the folded model under vLLM 0.27.1"},
        "B1_formal_session": {**scen, "hard_cap_usd": hard, "gpu": "NVIDIA L40S, EUR-IS-2, am5m373kuq",
                              "what": "the formal initialization, two recoveries and every item of eval-set-1 per seed, "
                                      "one billing pod"},
        "storage": {"checkpoints": "two factored fp32 checkpoints (~3.6 GB each) on am5m373kuq, which then holds "
                                   "~22 GB of 30; no new volume and no new storage cost",
                    "folded_checkpoints": "built at evaluation time from the factored ones, not retained"},
        "total_hard_cap_usd": round(3.0 + hard, 2),
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    doc = build()
    text = json.dumps(doc, indent=1) + "\n"
    if a.write:
        Path(a.write).write_text(text)
        print(a.write)
    print(json.dumps({k: doc[k] for k in ("B_E_engineering_validation", "B1_formal_session", "total_hard_cap_usd")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
