"""The revised Research A/B programme's cost forecast, from the project's own measurements. $0.

    PYTHONPATH=src:scripts python scripts/maintenance/consolidation/research_ab_forecast.py --write

Every rate is a measurement a record owns (``MEASURED`` names the record); every
scale is a design's (``configs/...``); the two scenarios differ only in the
declared multipliers (``SCENARIOS``). A session's hard cap is its expected cost
x the project's hard-cap factor 1.5, never below its conservative cost. Nothing
here is a measurement of the new code paths: the engineering validations exist
to measure those, and every formal session is re-priced from the measured step
time of the sessions before it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "logs/budget/applications/research_ab_2026-10-28/forecast.json"

MEASURED = {
    "l40s_usd_per_hour": (1.11, "logs/stages/stage-2/teacher_modes/plans/cost_forecast.json :: measured.l40s_usd_per_hour"),
    "a100_usd_per_hour": (1.81, "logs/stages/stage-2/teacher_modes/plans/cost_forecast.json :: measured.a100_usd_per_hour"),
    "d4a_seconds_per_step_l40s": (13385.3 / 3702, "logs/stages/stage-2/reasoning_data/runs/h1pa2_20261025_055042/closeout/"
                                                 "outcome.json :: units.D4-Augmented-s20260801.seconds.train / 3,702 steps"),
    "d4a_eval_seconds_l40s": (3917.1, "the same outcome :: units.D4-Augmented-s20260801.seconds.evaluate (all of eval-set-1)"),
    "isolated_seconds_per_step_a100": (5.19, "logs/stages/stage-2/teacher_modes/runs/ae_20261028_093428/evidence/"
                                             "trainer_smoke.json (E1 student, document_isolated, on-the-fly teacher)"),
    "generation_tokens_per_session_minute_a100": (80198, "logs/stages/stage-2/teacher_modes/plans/cost_forecast.json :: "
                                                         "measured.tokens_per_session_minute"),
    "t0_generation_tokens_m": (19.65 - 0.5, "the same forecast :: scenarios.expected.generation_tokens_m less calibration"),
    "t0_generation_tokens_m_conservative": (28.12 - 0.8, "the same :: scenarios.conservative, less calibration"),
    "t0_ht_s_sequences": (1235, "the same :: scenarios.expected.sequences.H-T-S"),
    "t0_ht_s_sequences_conservative": (1623, "the same :: scenarios.conservative.sequences.H-T-S"),
    "t0_hn_s_sequences": (309, "the same :: scenarios.expected.sequences.H-N-S"),
    "t0_hn_s_sequences_conservative": (706, "the same :: scenarios.conservative.sequences.H-N-S"),
}
#: design facts
D4A_SEQUENCES, D4A_STEPS = 2468, 3702
TRAIN_FLOPS_RATIO_PROJECTED = 21.646 / 18.364   # aadistill.models.cost_model, with the teacher's forward, ctx 4,096
INFER_FLOPS_RATIO_PROJECTED = 3.185 / 2.131
UNIT_FIXED_H = 10 / 60                          # h1pa.json :: pricing.training_fixed_minutes
SESSION_OVERHEAD_H = 0.25                       # setup (measured 3.7-7.3 min) + closeout and teardown
HARD_CAP_FACTOR = 1.5
SCENARIOS = {
    "expected": {"step": 1.0, "eval": 1.0, "projected_eval": INFER_FLOPS_RATIO_PROJECTED, "gpu_train_a": "l40s",
                 "teacher_eval_vs_student": 3.5, "hn_eval_vs_ht": 0.5},
    "conservative": {"step": 1.25, "eval": 1.25, "projected_eval": 2.0, "gpu_train_a": "a100",
                     "teacher_eval_vs_student": 5.0, "hn_eval_vs_ht": 0.7},
}


def m(key: str) -> float:
    return MEASURED[key][0]


def research_b(s: dict) -> dict:
    l40s = m("l40s_usd_per_hour")
    d4a_step, d4a_eval = m("d4a_seconds_per_step_l40s") * s["step"], m("d4a_eval_seconds_l40s") * s["eval"]
    control_h = 2 * (D4A_STEPS * d4a_step / 3600 + UNIT_FIXED_H + d4a_eval / 3600) + 2 * SESSION_OVERHEAD_H
    proj_step = m("d4a_seconds_per_step_l40s") * TRAIN_FLOPS_RATIO_PROJECTED * s["step"]
    proj_eval = m("d4a_eval_seconds_l40s") * s["projected_eval"]
    proj_h = 2 * (D4A_STEPS * proj_step / 3600 + UNIT_FIXED_H + proj_eval / 3600) + 2 * SESSION_OVERHEAD_H
    init_h = (55 if s is SCENARIOS["expected"] else 80) / 60
    return {"engineering_validation": {"usd": 3.0, "_": "its proposal's hard cap (plans/independent_gated_engineering_proposal.json)"},
            "initialization": {"hours": round(init_h, 2), "gpu": "L40S", "usd": round(init_h * l40s, 2)},
            "control_D4-Augmented-topk200": {"hours": round(control_h, 2), "gpu": "L40S", "usd": round(control_h * l40s, 2),
                                             "seconds_per_step": round(d4a_step, 3)},
            "projected_1123M": {"hours": round(proj_h, 2), "gpu": "L40S", "usd": round(proj_h * l40s, 2),
                                "seconds_per_step": round(proj_step, 3)}}


def research_a(s: dict, key: str) -> dict:
    a100, l40s = m("a100_usd_per_hour"), m("l40s_usd_per_hour")
    cons = key == "conservative"
    scale = D4A_SEQUENCES / m("t0_ht_s_sequences_conservative" if cons else "t0_ht_s_sequences")
    gen_tokens = m("t0_generation_tokens_m_conservative" if cons else "t0_generation_tokens_m") * scale + (0.8 if cons else 0.5)
    gen_h = gen_tokens * 1e6 / m("generation_tokens_per_session_minute_a100") / 60 + 4 * SESSION_OVERHEAD_H
    hn_s_seq = m("t0_hn_s_sequences_conservative" if cons else "t0_hn_s_sequences") * scale
    steps = {"H-T-S": D4A_STEPS, "H-N-S": math.ceil(3 * hn_s_seq / 2), "H-N-C": D4A_STEPS}
    step_s = m("isolated_seconds_per_step_a100") * s["step"]
    train_h = 2 * sum(n * step_s / 3600 + UNIT_FIXED_H for n in steps.values()) + 3 * SESSION_OVERHEAD_H
    stu = m("d4a_eval_seconds_l40s") * s["eval"] / 3600
    eval_h = 2 * stu + 4 * stu * s["hn_eval_vs_ht"] + stu * s["teacher_eval_vs_student"] * (1 + s["hn_eval_vs_ht"]) \
        + 2 * SESSION_OVERHEAD_H
    train_rate = l40s if s["gpu_train_a"] == "l40s" else a100
    init_h = 1.5 if cons else 1.0
    return {"generation": {"tokens_m": round(gen_tokens, 2), "scale_vs_t0": round(scale, 3), "hours": round(gen_h, 2),
                           "gpu": "A100 80GB", "usd": round(gen_h * a100, 2)},
            "initialization": {"hours": init_h, "gpu": "A100 80GB", "usd": round(init_h * a100, 2)},
            "recovery": {"optimizer_steps": steps, "seconds_per_step": round(step_s, 3), "hours": round(train_h, 2),
                         "gpu": s["gpu_train_a"].upper(), "usd": round(train_h * train_rate, 2)},
            "evaluation": {"hours": round(eval_h, 2), "gpu": "L40S", "usd": round(eval_h * l40s, 2),
                           "what": "six students in their own modes and the teacher in both modes, every item of eval-set-1"}}


def forecast() -> dict:
    out = {"schema": "aadistill.budget.research_ab_forecast/v1", "written_utc": "2026-10-28T18:30:00Z",
           "measured": {k: {"value": round(v, 4), "source": src} for k, (v, src) in MEASURED.items()},
           "design_facts": {"d4a_sequences": D4A_SEQUENCES, "d4a_steps": D4A_STEPS,
                            "projected_training_flops_ratio": round(TRAIN_FLOPS_RATIO_PROJECTED, 4),
                            "projected_inference_flops_ratio": round(INFER_FLOPS_RATIO_PROJECTED, 4)},
           "scenarios": {}}
    for key, s in SCENARIOS.items():
        b, a = research_b(s), research_a(s, key)
        out["scenarios"][key] = {"multipliers": s, "research_b": b, "research_a": a,
                                 "research_b_usd": round(sum(v["usd"] for v in b.values()), 2),
                                 "research_a_usd": round(sum(v["usd"] for v in a.values()), 2)}
    e, c = out["scenarios"]["expected"], out["scenarios"]["conservative"]

    def cap(study, item):
        ev, cv = e[study][item]["usd"], c[study][item]["usd"]
        return round(max(ev * HARD_CAP_FACTOR, cv), 2) if item != "engineering_validation" else ev

    out["hard_caps_usd"] = {
        "research_b": {k: cap("research_b", k) for k in e["research_b"]},
        "research_a": {k: cap("research_a", k) for k in e["research_a"]},
        "storage": 14.0, "engineering_reserve": 10.0,
        "_storage": "one 100 GB network volume for the phase's save points, checkpoints and corpora (or am5m373kuq "
                    "resized to 100 GB), at $0.07/GB-month for two months",
        "_reserve": "pre-science engineering subruns across the whole phase (P12.1), cumulative"}
    hc = out["hard_caps_usd"]
    hc["research_b_total"] = round(sum(hc["research_b"].values()), 2)
    hc["research_a_total"] = round(sum(hc["research_a"].values()), 2)
    hc["phase_total"] = round(hc["research_b_total"] + hc["research_a_total"] + hc["storage"] + hc["engineering_reserve"], 2)
    out["expected_usd"] = round(e["research_a_usd"] + e["research_b_usd"] + 14.0, 2)
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
                      "expected": {k: doc["scenarios"]["expected"][k] for k in ("research_a_usd", "research_b_usd")},
                      "conservative": {k: doc["scenarios"]["conservative"][k] for k in ("research_a_usd", "research_b_usd")}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
