#!/usr/bin/env python3
"""Price Research A's phases from measured sessions and declared planning assumptions ($0).

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/forecast_cost.py \\
        --write logs/stages/stage-2/teacher_modes/plans/cost_forecast.json

MEASURED (read from their owners): P1/P1t/PA1's generated tokens and session
minutes on the A100 (the generation records and closeouts), D4-Augmented's
seconds per optimizer step and per evaluation on the L40S (mixed_result.json),
the isolated packing efficiency on H1-PA's documents (token_identity_audit.json),
and both GPUs' prices. ASSUMED (declared, and replaced by A1's pilot slice before
the rest is generated): the hybrid's tokens per task in each mode relative to
Thinking-2507, the share of T0 admitted in both modes, the H-N extension's size,
and the 4B teacher's evaluation time relative to a 0.6B student's.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
GEN = {"h1gen_20261022_190653": "logs/stages/stage-2/reasoning_data/runs/h1gen_20261022_190653/closeout/outcome.json",
       "h1p1t_20261023_094007": "logs/stages/stage-2/reasoning_data/runs/h1p1t_20261023_094007/closeout/outcome.json",
       "h1pat_20261024_115517": "logs/stages/stage-2/reasoning_data/runs/h1pat_20261024_115517/closeout/outcome.json"}
RECORDS = "artifacts/stages/stage-2/reasoning_data/generation"
MIXED = "logs/stages/stage-0/calibration_volume/analyses/mixed_result.json"
AUDIT = "logs/stages/stage-2/teacher_modes/audit/token_identity_audit.json"
T0_TASKS, SEQ, D4A_STEPS = 1761, 8192, 3702
A100, L40S = 1.79 + 0.1 * 150 / 720, 1.09 + 0.1 * 150 / 720

ASSUMPTIONS = {
 "expected": {"ht_tokens_vs_2507": 1.0, "hn_tokens_vs_2507": 0.25, "admitted_both": 0.70, "x_tokens_m": 3.0,
              "teacher_eval_vs_student": 3.5, "hn_eval_vs_ht": 0.5, "calibration_tokens_m": 0.5},
 "conservative": {"ht_tokens_vs_2507": 1.15, "hn_tokens_vs_2507": 0.5, "admitted_both": 0.80, "x_tokens_m": 6.0,
                  "teacher_eval_vs_student": 5.0, "hn_eval_vs_ht": 0.7, "calibration_tokens_m": 0.8},
}


def load(rel):
    return json.loads((REPO / rel).read_text())


def measured() -> dict:
    tokens = 0
    for run in GEN:
        for f in (REPO / RECORDS / run).glob("pass*_generation_input_*.jsonl"):
            for line in f.read_text().split("\n"):
                if line.strip():
                    tokens += sum(t["generated_tokens"] for t in json.loads(line)["turns"])
    minutes = sum(load(p)["money"]["elapsed_minutes"] for p in GEN.values())
    units = load(MIXED)["units"]
    d4 = [u["seconds"] for k, u in units.items() if k.startswith("D4-Augmented")]
    audit = load(AUDIT)["as_generated_documents"]["isolated_pack"]
    return {"generated_tokens_2507": tokens, "generation_session_minutes": round(minutes, 2),
            "tokens_per_session_minute": round(tokens / minutes),
            "seconds_per_step": round(sum(s["train"] for s in d4) / len(d4) / D4A_STEPS, 4),
            "evaluate_seconds_student": round(sum(s["evaluate"] for s in d4) / len(d4), 1),
            "isolated_packing_fill": round(1 - audit["padding_fraction"], 4),
            "a100_usd_per_hour": round(A100, 4), "l40s_usd_per_hour": round(L40S, 4)}


def scenario(m: dict, a: dict) -> dict:
    ht = m["generated_tokens_2507"] * a["ht_tokens_vs_2507"]
    hn = m["generated_tokens_2507"] * a["hn_tokens_vs_2507"]
    gen_tokens = ht + hn + (a["x_tokens_m"] + a["calibration_tokens_m"]) * 1e6
    gen_hours = gen_tokens / m["tokens_per_session_minute"] / 60 + 0.5     # + setup and six engine starts
    # training data: admitted documents of I ~ admitted share of each mode's tokens (prompts ~5%)
    ht_seq = math.ceil(ht * a["admitted_both"] * 1.05 / (SEQ * m["isolated_packing_fill"]))
    hn_seq = math.ceil(hn * a["admitted_both"] * 1.05 / (SEQ * m["isolated_packing_fill"]))
    steps = {"H-T-S": math.ceil(3 * ht_seq / 2), "H-N-S": math.ceil(3 * hn_seq / 2), "H-N-C": math.ceil(3 * ht_seq / 2)}
    train_h = 2 * sum(steps.values()) * m["seconds_per_step"] / 3600
    ev = m["evaluate_seconds_student"]
    eval_h = 2 * (ev + 2 * ev * a["hn_eval_vs_ht"]) / 3600
    teacher_h = (ev * a["teacher_eval_vs_student"] * (1 + a["hn_eval_vs_ht"])) / 3600
    a3_hours = train_h + eval_h + teacher_h + 0.75
    return {"assumptions": a, "generation_tokens_m": round(gen_tokens / 1e6, 2), "sequences": {"H-T-S": ht_seq, "H-N-S": hn_seq,
            "H-N-C": ht_seq}, "optimizer_steps": steps,
            "A1_generation": {"hours": round(gen_hours, 2), "usd": round(gen_hours * A100, 2)},
            "A2_initialization": {"hours": 1.0 if a is ASSUMPTIONS["expected"] else 1.5},
            "A3_recovery_and_evaluation": {"hours": round(a3_hours, 2), "train_hours": round(train_h, 2),
                                           "student_eval_hours": round(eval_h, 2), "teacher_eval_hours": round(teacher_h, 2),
                                           "usd": round(a3_hours * L40S, 2)}}


def build() -> dict:
    m = measured()
    sc = {k: scenario(m, a) for k, a in ASSUMPTIONS.items()}
    for k in sc:
        sc[k]["A2_initialization"]["usd"] = round(sc[k]["A2_initialization"]["hours"] * A100, 2)
    caps = {"A_E_engineering_validation": 3.5,
            "A1_generation": math.ceil(sc["conservative"]["A1_generation"]["usd"] * 1.15 * 2) / 2,
            "A2_initialization": 3.5,
            "A3_recovery_and_evaluation": math.ceil(sc["conservative"]["A3_recovery_and_evaluation"]["usd"] * 1.1 * 2) / 2,
            "storage": 3.5, "engineering_reserve": 3.0}
    return {
        "schema": "aadistill.stage2.teacher_modes.cost_forecast/v1",
        "measured": m, "sources": list(GEN.values()) + [MIXED, AUDIT],
        "scenarios": sc,
        "A_E_engineering_validation": {"hours_expected": 1.0, "hours_conservative": 1.75, "gpu": "NVIDIA A100-SXM4-80GB",
                                       "what": "the hybrid teacher under vLLM 0.27.1 in both modes on a few T0 tasks (prompt identity, "
                                               "think states, stop ids), the real-teacher teacher-forced isolation check in bf16 "
                                               "and fp32, a 20-step isolated-pack trainer smoke with kd_scope assistant and the "
                                               "on-the-fly hybrid teacher (step time, peak memory)"},
        "storage": "a new 50 GB network volume beside the A3 pod for six checkpoints (~2.4 GB each); am5m373kuq "
                   "(14.7 of 30 GB, kept for the H1-PA/mixed review) cannot hold them; reserved at 50 GB x $0.07 x 1 month",
        "hard_caps_usd": caps,
        "total_hard_cap_usd": round(sum(caps.values()), 2),
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
    print(json.dumps({"measured": doc["measured"], "expected": doc["scenarios"]["expected"],
                      "conservative": {k: v for k, v in doc["scenarios"]["conservative"].items() if k != "assumptions"},
                      "caps": doc["hard_caps_usd"], "total": doc["total_hard_cap_usd"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
