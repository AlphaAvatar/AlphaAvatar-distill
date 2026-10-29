#!/usr/bin/env python3
"""Research A's D4-scale phase, priced from the project's own measurements, alone. $0.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/phase_forecast.py --write

Every rate is a measurement a record owns (``MEASURED`` names it); every scale
is the design's; the scenarios differ only in the declared ``SCENARIOS``
multipliers (the hybrid's tokens per task in each mode against Thinking-2507's,
the share of tasks admitted in both modes, the step time and evaluation time
against their measured references). A session's hard cap is the larger of its
expected cost x 1.5 and its conservative cost. Nothing here measures the new
paths: every formal session is re-priced from the sessions before it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "logs/budget/applications/research_a_2026-10-29/forecast.json"

MEASURED = {
    "a100_usd_per_hour": (1.79, "logs/stages/stage-2/teacher_modes/runs/ae_20261028_093428/runtime/session.json :: cost.price_per_hour (A-E, A100 SXM 80GB)"),
    "l40s_usd_per_hour": (1.09, "logs/stages/stage-2/reasoning_data/runs/h1pa2_20261025_055042 (H1-PA, L40S)"),
    "t0_generated_tokens_2507": (12_920_708, "logs/stages/stage-2/teacher_modes/plans/cost_forecast.json :: measured.generated_tokens_2507 (H1's P1 + P1t + PA1 records, both candidates)"),
    "tokens_per_session_minute_a100": (80_198, "the same :: measured.tokens_per_session_minute (H1's three generation sessions, A100)"),
    "t0_ht_s_sequences_at_reference": (1235, "the same :: scenarios.expected.sequences.H-T-S (T0, admitted_both 0.70, H-T tokens 1.0x)"),
    "isolated_seconds_per_step_a100": (5.19, "logs/stages/stage-2/teacher_modes/runs/ae_20261028_093428/evidence/trainer_smoke.json (document_isolated, on-the-fly teacher, A100)"),
    "student_eval_seconds_l40s": (3917.1, "logs/stages/stage-2/reasoning_data/runs/h1pa2_20261025_055042/closeout/outcome.json :: D4-Augmented-s20260801 evaluate (all of eval-set-1, nucleus_v1)"),
    "operator_seconds_d4_path_a100": (2446.4, "the dose_d4 search telemetry: D4 Top-1's ATTENTION 98.9 + DEPTH 2,258.9 + WIDTH 43.9 + FFN 44.7 s (A100)"),
    "save_point_bytes": (5_908_258_491, "configs/stages/stage-2/reasoning_data/h1pa.json :: storage.bytes.save_point (this geometry)"),
    "checkpoint_bytes": (2_384_234_968, "the same :: storage.bytes.model_safetensors"),
}
DESIGN = {"t0_tasks": 1761, "v_tasks": 240, "wave": 400, "e_waves": 9, "target_sequences": 2468, "calibration_prompts": 160,
          "seeds": 2, "epochs": 3, "batch": 2, "x_tasks": 14000}
SCENARIOS = {
    "expected": {"ht_tokens": 1.0, "hn_tokens": 0.25, "admitted_both": 0.70, "step": 1.0, "student_eval": 0.75,
                 "teacher_eval_ht": 3.5, "teacher_eval_hn": 1.75, "init_minutes": 60, "hn_admitted": 0.75},
    "conservative": {"ht_tokens": 1.15, "hn_tokens": 0.25, "admitted_both": 0.55, "step": 1.25, "student_eval": 1.0,
                     "teacher_eval_ht": 5.0, "teacher_eval_hn": 3.5, "init_minutes": 90, "hn_admitted": 0.65},
}
SETUP_H, CLOSE_H, UNIT_FIXED_H = 20 / 60, 0.25, 10 / 60
HARD_CAP_FACTOR = 1.5
DISK_USD_PER_HOUR = 150 * 0.10 / (24 * 30.4375)        #: 150 GB container disk at $0.10/GB-month
VOLUME_GB, VOLUME_USD_GB_MONTH, VOLUME_MONTHS = 60, 0.07, 2


def m(k: str) -> float:
    return MEASURED[k][0]


def teacher_session(s: dict) -> dict:
    per_task = m("t0_generated_tokens_2507") / DESIGN["t0_tasks"]
    both = per_task * (s["ht_tokens"] + s["hn_tokens"])
    s_t0 = m("t0_ht_s_sequences_at_reference") * s["ht_tokens"] * s["admitted_both"] / 0.70
    e_needed = math.ceil(DESIGN["t0_tasks"] * max(0.0, DESIGN["target_sequences"] / s_t0 - 1))
    e_generated = min(DESIGN["e_waves"] * DESIGN["wave"], (math.ceil(e_needed / DESIGN["wave"]) + 1) * DESIGN["wave"])
    #: X: H-N documents of X fill H-N-C to H-T-S's count; H-N-S already brings hn/ht of it (by tokens)
    hts_generated = DESIGN["target_sequences"] * 8192 * 0.94 * 0.9
    x_tokens = max(0.0, (0.7 - s["hn_tokens"] / s["ht_tokens"])) * hts_generated / s["hn_admitted"] * 1.15
    tokens = {"calibration": DESIGN["calibration_prompts"] * both * 0.6, "t0": DESIGN["t0_tasks"] * both,
              "v": DESIGN["v_tasks"] * both, "e": e_generated * both, "x": x_tokens}
    gen_h = sum(tokens.values()) / m("tokens_per_session_minute_a100") / 60
    stu = m("student_eval_seconds_l40s") / 3600
    teval_h = stu * (s["teacher_eval_ht"] + s["teacher_eval_hn"])
    hours = {"setup": SETUP_H, "generation": gen_h, "initialization": s["init_minutes"] / 60,
             "teacher_evaluation": teval_h, "closeout": CLOSE_H}
    h = sum(hours.values())
    return {"tokens_m": {k: round(v / 1e6, 2) for k, v in tokens.items()}, "tokens_total_m": round(sum(tokens.values()) / 1e6, 2),
            "e_tasks_needed": e_needed, "e_tasks_generated": e_generated, "t0_ht_s_sequences": round(s_t0),
            "hours": {k: round(v, 2) for k, v in hours.items()}, "hours_total": round(h, 2), "gpu": "1 x A100 SXM 80GB",
            "usd": round(h * (m("a100_usd_per_hour") + DISK_USD_PER_HOUR), 2)}


def recovery(s: dict, teacher: dict) -> dict:
    hts = DESIGN["target_sequences"] + 0.5 * DESIGN["wave"] * (1.0 * teacher["t0_ht_s_sequences"] / DESIGN["t0_tasks"])
    hns = hts * s["hn_tokens"] / s["ht_tokens"] * 1.4      #: H-N documents carry relatively more prompt per target
    steps = {"H-T-S": math.ceil(3 * hts / 2), "H-N-S": math.ceil(3 * hns / 2), "H-N-C": math.ceil(3 * hts / 2)}
    step_s = m("isolated_seconds_per_step_a100") * s["step"]
    train_h = DESIGN["seeds"] * sum(n * step_s / 3600 + UNIT_FIXED_H for n in steps.values())
    eval_h = 6 * m("student_eval_seconds_l40s") / 3600 * s["student_eval"]
    h = SETUP_H + train_h + eval_h + CLOSE_H
    peak = (2 * 2 * m("save_point_bytes") + 6 * m("checkpoint_bytes") + 3e9 + 2e9) / 1e9
    return {"optimizer_steps": steps, "seconds_per_step": round(step_s, 3), "hours": {"train": round(train_h, 2),
            "evaluate": round(eval_h, 2)}, "hours_total": round(h, 2), "gpu": "A100 SXM 80GB (1 or 2 on one pod)",
            "usd": round(h * (m("a100_usd_per_hour") + DISK_USD_PER_HOUR), 2),
            "volume_peak_gb_two_gpus": round(peak, 1)}


def forecast() -> dict:
    out = {"schema": "aadistill.budget.research_a_forecast/v1", "written_utc": "2026-10-29T00:00:00Z",
           "measured": {k: {"value": v, "source": src} for k, (v, src) in MEASURED.items()}, "design": DESIGN,
           "scenarios": {}}
    for key, s in SCENARIOS.items():
        t = teacher_session(s)
        out["scenarios"][key] = {"multipliers": s, "teacher_session": t, "recovery": recovery(s, t)}
    e, c = out["scenarios"]["expected"], out["scenarios"]["conservative"]
    cap = lambda k: round(max(e[k]["usd"] * HARD_CAP_FACTOR, c[k]["usd"]), 2)  # noqa: E731
    storage = round(VOLUME_GB * VOLUME_USD_GB_MONTH * VOLUME_MONTHS, 2)
    out["hard_caps_usd"] = {"teacher_session": cap("teacher_session"), "recovery": cap("recovery"), "storage": storage,
                            "engineering_reserve": 10.0,
                            "_storage": f"one new {VOLUME_GB} GB network volume in an A100 datacenter, {VOLUME_MONTHS} months at "
                                        f"${VOLUME_USD_GB_MONTH}/GB-month",
                            "_reserve": "pre-science engineering subruns across the phase (P12.1), cumulative"}
    hc = out["hard_caps_usd"]
    hc["phase_total"] = round(hc["teacher_session"] + hc["recovery"] + storage + hc["engineering_reserve"], 2)
    out["expected_usd"] = round(e["teacher_session"]["usd"] + e["recovery"]["usd"] + storage, 2)
    out["authorizes"] = "nothing"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    doc = forecast()
    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"hard_caps_usd": doc["hard_caps_usd"], "expected_usd": doc["expected_usd"],
                      **{k: {x: v[x]["usd"] for x in ("teacher_session", "recovery")} for k, v in doc["scenarios"].items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
