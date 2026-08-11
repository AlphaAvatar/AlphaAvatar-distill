#!/usr/bin/env python3
"""Price E8b's pair-matched, session-split design through the four-threshold planner.

    PYTHONPATH=src python scripts/training/plan_e8b_budget.py

Hardware is **pair-matched inside each causal comparison** rather than uniform
across the 2x2, because the two primary effects are `DC - DP` (depth-only) and
`FC - FP` (fully compressed) and each is measured within one class:

    DP, DC   A100 SXM 80 GB   — the depth-only arms need 72.9 GB with margin
    FP, FC   L40S 48 GB       — FP is retained from L40S, so FC matches it

Hardware class is therefore **nested** with compression regime. The interaction
`(FC - FP) - (DC - DP)` inherits that nesting and cannot by itself exclude a
hardware x depth-map interaction; a conditional bridge, priced below but not run
prospectively, resolves it only if a material sign reversal actually occurs.

Four sessions, none longer than what this project has already run (E7 was 635 min):

    S1  L40S    step-0: build DP/DC, measure all FOUR initializations on ONE
                device through one canonical reload path, probe DP/DC behaviour
    S2  A100    DP-sa + DC-sa      seed-paired, so a lost session costs a seed,
    S3  A100    DP-sb + DC-sb      not a cell
    S4  L40S    FC-sa + FC-sb

A session split cannot alter token exposure, the schedule, seeds or evaluator
semantics: every arm reads the same config, and `save_every` 880 of 1,761 steps
makes exact mid-arm resume possible.

Hardware is selected on **cost per completed step**, not hourly price. Among
High-stock >=70 GB classes at the 2026-08-11 quotes, A100 SXM is cheapest per step;
the spread to H100 SXM is only 3.5%, so the choice is sensitive to the assumed
relative efficiency and the first-arm gate is what settles it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.budget import (  # noqa: E402
    BudgetError, Phase, StepTime, plan_session,
)

A100, A100_PRICE = "NVIDIA A100-SXM4-80GB", 1.59
L40S, L40S_PRICE = "NVIDIA L40S", 0.99
ACTUAL_BASELINE_USD = 163.8833
E8A_SPENT = 3.7253

STEPS = 1761
TOKENS_PER_STEP = 2 * 8192
N_DEPTH, N_TARGET, N_TEACHER = 3_215_021_568, 596_049_920, 4_022_468_096
L40S_MEASURED_STEP = 4.15
A100_EFFICIENCY_RATIO = 1.6
SAFETY = 1.15

SETUP_MIN = 45.0
BUILD_DEPTH_MIN = 6.0
RESERVE_MIN = 30.0
CONTINGENCY = 0.10
TRANSFER_MB_PER_MIN = 720.0


def derived_step(n_student: int, price: float, ratio: float) -> tuple[float, dict]:
    flops = (8 * n_student + 2 * N_TEACHER) * TOKENS_PER_STEP
    ref = (8 * N_TARGET + 2 * N_TEACHER) * TOKENS_PER_STEP
    l40s = ref / L40S_MEASURED_STEP / 1e12
    tflops = l40s * ratio
    raw = flops / (tflops * 1e12)
    s = raw * SAFETY
    return s, {"step_flops": flops, "l40s_effective_tflops": round(l40s, 1),
               "assumed_tflops": round(tflops, 1), "raw_seconds": round(raw, 2),
               "priced_seconds": round(s, 2), "safety_factor": SAFETY,
               "usd_per_step": round(price / 3600 * s, 6)}


def s1(authorized: float) -> dict:
    """Step-0 on one device: build DP/DC, measure all four inits, probe DP/DC."""
    p = plan_session(
        price_per_hour=L40S_PRICE, authorized_usd=authorized, arms=0, steps_per_arm=0,
        step_time=StepTime(L40S_MEASURED_STEP, "unused; S1 does not train"),
        setup_minutes=SETUP_MIN,
        other_phases=(
            Phase("build_DP_DC_on_pod_and_assert_hashes", 2 * BUILD_DEPTH_MIN),
            Phase("init_nll_DP", 14.0), Phase("init_nll_DC", 14.0),
            Phase("init_nll_FP_remeasured", 7.0), Phase("init_nll_FC_remeasured", 7.0),
            Phase("step0_probe_DP_eval_behavior_v0", 20.0),
            Phase("step0_probe_DC_eval_behavior_v0", 20.0),
            Phase("artifact_manifest_and_verify", 8.0),
            Phase("artifact_synchronization", 5.0)),
        contingency_fraction=CONTINGENCY, artifact_recovery_reserve_minutes=RESERVE_MIN)
    d = p.as_dict()
    d.update(label="S1 step-0, one evaluator, all four inits", gpu=L40S,
             price_per_hour=L40S_PRICE)
    return d


def s_depth(authorized: float, seed: str) -> dict:
    """One seed's matched DP+DC pair on the 80 GB card."""
    s, detail = derived_step(N_DEPTH, A100_PRICE, A100_EFFICIENCY_RATIO)
    train = 2 * STEPS * s / 60.0
    p = plan_session(
        price_per_hour=A100_PRICE, authorized_usd=authorized, arms=0, steps_per_arm=0,
        step_time=StepTime(s, "derived from FLOPs and E6b's measured 4.15 s/step; "
                              "the first-arm gate converts it to a measurement"),
        setup_minutes=SETUP_MIN,
        other_phases=(
            Phase("build_DP_DC_on_pod_and_assert_hashes", 2 * BUILD_DEPTH_MIN),
            Phase("throughput_vram_usd_per_step_gate_20_steps", 3.0),
            Phase("pretraining_gate", 3.0),
            Phase(f"train_DP_{seed}_and_DC_{seed}", round(train, 1)),
            Phase("evaluate_2_arms_frozen_battery", 60.0),
            Phase("general_text_diagnostics", 6.0),
            Phase("artifact_manifest_and_verify", 8.0),
            Phase("artifact_synchronization_12.9GB",
                  round(2 * 6430 / TRANSFER_MB_PER_MIN, 1))),
        contingency_fraction=CONTINGENCY, artifact_recovery_reserve_minutes=RESERVE_MIN)
    d = p.as_dict()
    d.update(label=f"S{2 if seed == 'sa' else 3} depth-only DP+DC, seed {seed}",
             gpu=A100, price_per_hour=A100_PRICE, step_seconds=round(s, 2),
             step_derivation=detail)
    return d


def s4(authorized: float) -> dict:
    """FC's two seeds on the L40S, matching FP's retained hardware."""
    s, detail = derived_step(N_TARGET, L40S_PRICE, 1.0 / 1.6)  # i.e. the L40S itself
    s = L40S_MEASURED_STEP  # measured, not derived: this is exactly E6b's workload
    train = 2 * STEPS * s / 60.0
    p = plan_session(
        price_per_hour=L40S_PRICE, authorized_usd=authorized, arms=2,
        steps_per_arm=STEPS,
        step_time=StepTime(s, "E6b measured 4.15 s/step for this exact model, rung "
                              "and card"),
        setup_minutes=SETUP_MIN, eval_minutes_per_arm=8.25,
        transfer_minutes=round(2 * 1190 / TRANSFER_MB_PER_MIN, 1),
        other_phases=(Phase("pretraining_gate", 3.0),
                      Phase("general_text_diagnostics", 6.0),
                      Phase("artifact_manifest_and_verify", 8.0)),
        contingency_fraction=CONTINGENCY, artifact_recovery_reserve_minutes=RESERVE_MIN)
    d = p.as_dict()
    d.update(label="S4 fully compressed FC, seeds sa+sb", gpu=L40S,
             price_per_hour=L40S_PRICE, step_seconds=s,
             train_minutes=round(train, 1))
    return d


def bridge(authorized: float) -> dict:
    """CONDITIONAL, not run prospectively: FP and FC reruns on the 80 GB card."""
    s, _ = derived_step(N_TARGET, A100_PRICE, A100_EFFICIENCY_RATIO)
    train = 4 * STEPS * s / 60.0
    p = plan_session(
        price_per_hour=A100_PRICE, authorized_usd=authorized, arms=0, steps_per_arm=0,
        step_time=StepTime(s, "derived; the bridge would carry its own gate"),
        # 2.98 s/step is below the 4.15 s floor, which is the only *measured* Stage 3
        # step time and was measured on an L40S. The floor exists so nobody prices a
        # run on an optimistic number; here the reduction is a card change, not
        # optimism, and the bridge would carry its own first-arm gate exactly as the
        # depth-only sessions do.
        below_floor_reason=("the floor is an L40S measurement; the bridge runs the "
                            "same 596M workload on an A100 SXM, and its own "
                            "first-arm throughput gate would confirm the rate "
                            "before the remaining arms are paid for"),
        setup_minutes=SETUP_MIN,
        other_phases=(Phase("train_FP_and_FC_both_seeds_on_A100", round(train, 1)),
                      Phase("evaluate_4_arms", 33.0),
                      Phase("init_nll_recheck", 12.0),
                      Phase("artifact_manifest_and_verify", 8.0),
                      Phase("artifact_synchronization", 6.6)),
        contingency_fraction=CONTINGENCY, artifact_recovery_reserve_minutes=RESERVE_MIN)
    d = p.as_dict()
    d.update(label="CONDITIONAL hardware bridge: FP+FC on A100", gpu=A100,
             price_per_hour=A100_PRICE, step_seconds=round(s, 2))
    return d


def smallest(fn, *args) -> dict:
    for cents in range(100, 12000, 25):
        try:
            return fn(cents / 100.0, *args)
        except BudgetError:
            continue
    raise SystemExit(f"no cap under $120 contains {fn.__name__}")


def show(d: dict) -> None:
    print(f"\n=== {d['label']} — {d['gpu']} @ ${d['price_per_hour']}/h ===")
    for ph in d["breakdown"]:
        if ph["minutes"]:
            print(f"  {ph['name']:42s} {ph['minutes']:7.1f} min  "
                  f"${ph['minutes'] / 60 * d['price_per_hour']:6.2f}")
    print(f"  {'-' * 42} {'-' * 7}")
    for k, n in (("expected", "expected completion"), ("soft_stop", "soft stop"),
                 ("artifact_recovery_reserve", "recovery reserve"),
                 ("hard_terminate", "absolute termination")):
        print(f"  {n:42s} {d[k + '_minutes']:7.1f} min  ${d[k + '_usd']:6.2f}")
    if "step_seconds" in d:
        print(f"  step time {d['step_seconds']} s"
              + (f", ${d['step_derivation']['usd_per_step']:.6f}/step"
                 if "step_derivation" in d else ""))


def main() -> int:
    sessions = [smallest(s1), smallest(s_depth, "sa"), smallest(s_depth, "sb"),
                smallest(s4)]
    for d in sessions:
        show(d)
    br = smallest(bridge)
    show(br)

    exp = sum(d["expected_usd"] for d in sessions)
    hard = sum(d["hard_terminate_usd"] for d in sessions)
    wall = sum(d["expected_minutes"] for d in sessions)
    print("\n=== E8b total, pair-matched, four sessions ===")
    for d in sessions:
        print(f"  {d['label'][:48]:48s} exp ${d['expected_usd']:6.2f}  "
              f"hard ${d['hard_terminate_usd']:6.2f}  "
              f"{d['expected_minutes'] / 60:5.1f} h")
    print(f"  {'TOTAL':48s} exp ${exp:6.2f}  hard ${hard:6.2f}  {wall / 60:5.1f} h")
    print(f"\n  longest single session "
          f"{max(d['expected_minutes'] for d in sessions) / 60:.1f} h "
          f"(E7 ran 10.6 h)")
    print(f"\n  cumulative spend to date        ${ACTUAL_BASELINE_USD:.4f}")
    print(f"  of which E8a                    ${E8A_SPENT:.4f}")
    print(f"  ADDITIONAL AUTHORIZATION        ${hard:.2f}")
    print(f"  PROPOSED NEW CUMULATIVE CAP     ${ACTUAL_BASELINE_USD + hard:.2f}")
    print(f"\n  CONDITIONAL bridge, only on a material reversal: "
          f"hard ${br['hard_terminate_usd']:.2f} "
          f"(cap would become ${ACTUAL_BASELINE_USD + hard + br['hard_terminate_usd']:.2f})")

    out = {"design": "pair-matched hardware, nested with compression regime",
           "actual_cumulative_baseline_usd": ACTUAL_BASELINE_USD,
           "e8a_spend_already_incurred_usd": E8A_SPENT,
           "sessions": sessions, "conditional_bridge": br,
           "e8b_expected_usd": round(exp, 2),
           "e8b_hard_backstop_usd": round(hard, 2),
           "proposed_new_cumulative_cap_usd": round(ACTUAL_BASELINE_USD + hard, 2),
           "expected_wall_hours": round(wall / 60, 1)}
    dest = REPO_ROOT / "logs/e8b_budget_plans.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
