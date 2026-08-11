#!/usr/bin/env python3
"""Price E8b's factorial through the four-threshold planner.

    PYTHONPATH=src python scripts/training/plan_e8b_budget.py

E8b trains six or eight arms across two model sizes, and the depth-only cells are
5.39x the target student's parameters. Two consequences drive every number here:

**The depth-only arms do not fit a 48 GB card.** `size_e8b_memory.py` puts their
expected peak at 63.4 GB and 72.9 GB with the 15% margin every prior session has
needed, against 23.1/26.6 GB for the target-size cells. So the whole session runs on
one 80 GB card, which also satisfies the requirement that every step-0 measurement
share one evaluator and one environment.

**No 3.2B step time has ever been measured in this project.** The 4.15 s/step figure
is the 596M student on an L40S. E8b's depth-only step time is therefore *derived*
from FLOPs and the L40S's measured efficiency, and the derivation is printed so the
assumption is auditable rather than buried:

    step FLOPs = 8 * N_student * tokens        (fwd + bwd + checkpoint recompute)
               + 2 * N_teacher * tokens        (the frozen teacher's KD forward)

At 596M on L40S that gives 2.10e14 FLOPs against a measured 4.15 s/step, i.e. ~50.6
effective TFLOPS. An A100 SXM is assumed to reach 1.6x that — below its 1.72x peak
ratio, because the 152k-vocabulary softmax and the optimizer step are
bandwidth-bound and the A100's bandwidth advantage is larger than its FLOPs
advantage. A **throughput gate** on the first arm turns this estimate into a
measurement before the remaining arms are paid for.
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

# Live RunPod secure pricing, queried 2026-08-11. A100 SXM was the cheapest
# >=70 GB class with High stock; L40S at $0.99 cannot hold the depth-only arms.
GPU = "NVIDIA A100-SXM4-80GB"
PRICE = 1.59
VRAM_GB = 80
ACTUAL_BASELINE_USD = 163.8833        # 160.158 after E7 + 3.7253 on E8a
E8_ALREADY_SPENT = 3.7253             # inside the $13.24 E8 backstop

STEPS_PER_ARM = 1761                  # the canonical 1.60M rung, unchanged
TOKENS_PER_STEP = 2 * 8192

N_DEPTH_ONLY = 3_215_021_568
N_TARGET = 596_049_920
N_TEACHER = 4_022_468_096

L40S_MEASURED_STEP = 4.15             # 596M, block_len 8192, E6b measured
A100_EFFICIENCY_RATIO = 1.6           # below the 1.72x peak FLOPs ratio, on purpose
SAFETY = 1.15                         # the step-time margin the gate then removes

SETUP_MIN = 45.0
INIT_NLL_DEPTH_MIN = 10.0             # 3.2B student + 4B teacher over 3 series
INIT_NLL_TARGET_MIN = 6.0
STEP0_PROBE_MIN = 20.0                # eval_behavior_v0, 76 prompts, per model
EVAL_DEPTH_MIN = 30.0                 # frozen battery, 3.2B, per arm
EVAL_TARGET_MIN = 8.25                # E6b measured, per arm
GENERAL_TEXT_MIN = 3.0
GATE_MIN = 3.0
MANIFEST_MIN = 8.0
RESERVE_MIN = 30.0
CONTINGENCY = 0.10
# 6.43 GB per depth-only checkpoint, 1.19 GB per target checkpoint, at the ~12 MB/s
# E6b measured moving 2 x 5.6 GB in ~15 min.
TRANSFER_MB_PER_MIN = 720.0


def step_seconds(n_student: int) -> tuple[float, dict]:
    flops = (8 * n_student + 2 * N_TEACHER) * TOKENS_PER_STEP
    ref = (8 * N_TARGET + 2 * N_TEACHER) * TOKENS_PER_STEP
    l40s_tflops = ref / L40S_MEASURED_STEP / 1e12
    a100_tflops = l40s_tflops * A100_EFFICIENCY_RATIO
    raw = flops / (a100_tflops * 1e12)
    return raw * SAFETY, {
        "step_flops": flops, "reference_flops": ref,
        "l40s_effective_tflops": round(l40s_tflops, 1),
        "a100_assumed_tflops": round(a100_tflops, 1),
        "raw_seconds": round(raw, 2), "with_safety": round(raw * SAFETY, 2),
        "safety_factor": SAFETY,
    }


def plan(n_arms_depth: int, n_arms_target: int, authorized: float, label: str) -> dict:
    depth_s, depth_detail = step_seconds(N_DEPTH_ONLY)
    target_s, target_detail = step_seconds(N_TARGET)
    depth_train = n_arms_depth * STEPS_PER_ARM * depth_s / 60.0
    target_train = n_arms_target * STEPS_PER_ARM * target_s / 60.0
    n_inits = 4                        # DP, DC, FP, FC — all measured on one device
    ckpt_mb = n_arms_depth * 6430 + n_arms_target * 1190
    phases = (
        Phase("init_nll_DP_DC", 2 * INIT_NLL_DEPTH_MIN),
        Phase("init_nll_FP_FC", 2 * INIT_NLL_TARGET_MIN),
        Phase("step0_autonomous_probe_DP_DC", 2 * STEP0_PROBE_MIN),
        Phase("pretraining_gates", GATE_MIN * 2),
        Phase("train_depth_only", round(depth_train, 1)),
        Phase("train_target_size", round(target_train, 1)),
        Phase("evaluate_depth_only", n_arms_depth * EVAL_DEPTH_MIN),
        Phase("evaluate_target_size", n_arms_target * EVAL_TARGET_MIN),
        Phase("general_text_diagnostics",
              (n_arms_depth + n_arms_target) * GENERAL_TEXT_MIN),
        Phase("artifact_manifest_and_verify", MANIFEST_MIN),
        Phase("artifact_synchronization", round(ckpt_mb / TRANSFER_MB_PER_MIN, 1)),
    )
    p = plan_session(
        price_per_hour=PRICE, authorized_usd=authorized,
        arms=0, steps_per_arm=0,
        step_time=StepTime(depth_s, "derived from FLOPs and E6b's measured 4.15 "
                                    "s/step; a throughput gate on the first arm "
                                    "converts it to a measurement"),
        setup_minutes=SETUP_MIN, other_phases=phases,
        contingency_fraction=CONTINGENCY,
        artifact_recovery_reserve_minutes=RESERVE_MIN)
    d = p.as_dict()
    d.update(label=label, gpu=GPU, price_per_hour=PRICE,
             n_arms_depth=n_arms_depth, n_arms_target=n_arms_target,
             depth_step_seconds=round(depth_s, 2),
             target_step_seconds=round(target_s, 2),
             depth_step_derivation=depth_detail,
             target_step_derivation=target_detail,
             checkpoint_transfer_mb=ckpt_mb, initializations_measured=n_inits)
    return d


def smallest(n_depth: int, n_target: int, label: str) -> dict:
    for cents in range(1000, 12000, 25):
        try:
            return plan(n_depth, n_target, cents / 100.0, label)
        except BudgetError:
            continue
    raise SystemExit(f"no cap under $120 contains {label}")


def show(d: dict) -> None:
    print(f"\n=== {d['label']} — {d['gpu']} @ ${d['price_per_hour']}/h ===")
    for p in d["breakdown"]:
        if p["minutes"]:
            print(f"  {p['name']:34s} {p['minutes']:8.1f} min  "
                  f"${p['minutes'] / 60 * d['price_per_hour']:6.2f}")
    print(f"  {'-' * 34} {'-' * 8}")
    for key, name in (("expected", "expected completion"),
                      ("soft_stop", "soft stop"),
                      ("artifact_recovery_reserve", "artifact-recovery reserve"),
                      ("hard_terminate", "absolute termination")):
        print(f"  {name:34s} {d[key + '_minutes']:8.1f} min  ${d[key + '_usd']:6.2f}")
    print(f"  step time: depth-only {d['depth_step_seconds']} s, "
          f"target {d['target_step_seconds']} s")


def main() -> int:
    full = smallest(4, 4, "E8b FULL — DP/DC/FC/FP all trained, hardware-matched")
    retained = smallest(4, 2, "E8b RETAINED-FP — FP reused from L40S")
    show(retained)
    show(full)

    dd = retained["depth_step_derivation"]
    print("\n=== step-time derivation (no 3.2B step has ever been measured) ===")
    print(f"  step FLOPs, depth-only        {dd['step_flops']:.3e}")
    print(f"  step FLOPs, 596M reference    {dd['reference_flops']:.3e}")
    print(f"  L40S effective TFLOPS         {dd['l40s_effective_tflops']} "
          f"(from 4.15 s/step measured)")
    print(f"  A100 assumed TFLOPS           {dd['a100_assumed_tflops']} "
          f"({A100_EFFICIENCY_RATIO}x, under the 1.72x peak ratio)")
    print(f"  raw / with {SAFETY}x safety      {dd['raw_seconds']} s / "
          f"{dd['with_safety']} s")

    print("\n=== authorization ===")
    for d in (retained, full):
        inc = d["hard_terminate_usd"]
        print(f"  {d['label'][:44]:44s} hard ${inc:6.2f}  "
              f"-> cap ${ACTUAL_BASELINE_USD + inc:8.2f}")
    print(f"\n  cumulative spend to date        ${ACTUAL_BASELINE_USD:.4f}")
    print(f"  of which E8/E8a                 ${E8_ALREADY_SPENT:.4f}")
    print(f"  E8's remaining unspent backstop  ${13.24 - E8_ALREADY_SPENT:.4f} "
          f"(NOT carried over; E8b is a new design)")

    out = {"gpu": GPU, "price_per_hour": PRICE, "vram_gb": VRAM_GB,
           "priced_utc_note": "RunPod secure pricing queried 2026-08-11; "
                              "A100 SXM was the cheapest >=70 GB class at High stock",
           "actual_cumulative_baseline_usd": ACTUAL_BASELINE_USD,
           "e8a_spend_already_incurred_usd": E8_ALREADY_SPENT,
           "retained_fp": retained, "full_factorial": full}
    dest = REPO_ROOT / "logs/e8b_budget_plans.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
