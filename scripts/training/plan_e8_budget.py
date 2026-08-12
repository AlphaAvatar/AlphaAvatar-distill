#!/usr/bin/env python3
"""Price E8 through the four-threshold planner, phase by phase.

Every figure in `logs/archive/e8_preregistration.md` comes from here, so the proposal and
the arithmetic that would enforce it are the same code. Touches no GPU, creates
no pod.

    PYTHONPATH=src python scripts/training/plan_e8_budget.py

Two paid pods, split where the artifact flow forces a split
----------------------------------------------------------
Constructing the treatment initialization needs the 1.95 GB Stage 0 activation
statistics; training needs only the 1.19 GB initialized checkpoint. The dev-box
uplink is 0.72 MB/s and the relay is near its private-storage limit, so the
cheaper artifact is the one that crosses: **the dev box builds the initialization
(12 s on CPU, free) and ships the checkpoint**, rather than shipping the
statistics to a pod. That places a natural boundary between the search and the
training, which is also where the step-0 comparison belongs.

    pod A   contribution search over the frozen calibration set
    dev box build the treatment init; prove the baseline init reproduces; upload
    pod B   measure both inits' NLL, gate, train sa + sb, evaluate, sync

Search cost, derived rather than guessed
----------------------------------------
260 subset evaluations over 67 calibration items. With the intact reference cached
once, the search performs `n_items * (1 + 260)` forward passes — 17,487 — of which
260/261 are over 28 of the teacher's 36 blocks. Priced from the teacher's
parameter count and the L40S's sustained bf16 throughput, then given a wide
margin, because this project has no *measured* number for this workload yet and
`StepTime` refuses an estimate that does not say so.
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

PRICE = 0.99                       # L40S secure, the rate every Stage 3 arm paid
ACTUAL_BASELINE_USD = 160.158      # actual cumulative spend after E7
AUTHORIZED_CAP_USD = 162.49        # current cumulative authorization

# --- pod B: the training that carries the experiment --------------------------
STEPS_PER_ARM = 2916               # the 2.96M rung's schedule, copied from E1/P1
E8_STEP = StepTime(4.15, "E6b measured 4.15 s/step at block_len 8192; E8 trains "
                         "the identical recipe with no extra stream")
SETUP_MIN = 45.0                   # not the warm-image 5-8.5 min; it has been 150+
EVAL_MIN_PER_ARM = 8.25            # E6b frozen battery, measured
INIT_NLL_MIN_PER_CKPT = 6.0        # holdout_v1 + 512x1024 dense + val slice, with
                                   # teacher KL, on one GPU
TRANSFER_MIN_PER_ARM = 10.0        # E6b moved 2 x 5.6 GB in ~15 min
RESERVE_MIN = 30.0
CONTINGENCY = 0.10

# --- pod A: the search --------------------------------------------------------
CALIBRATION_ITEMS = 67
CALIBRATION_POSITIONS = 59_763
SUBSET_EVALUATIONS = 260           # 36+35+...+29, asserted by the search code
TEACHER_LAYERS, STUDENT_LAYERS = 36, 28
TEACHER_PARAMS = 4.02e9
L40S_SUSTAINED_TFLOPS = 45.0       # deliberately below peak for a small-batch
                                   # prefill workload with a 152k-vocab softmax
SEARCH_OVERHEAD = 1.6              # python, dtype casts, per-pass launch, KL
                                   # reduction over 152k logits


def search_minutes() -> tuple[float, dict]:
    """Forward-pass arithmetic, not a guess, with the margin named separately."""
    layer_params = (TEACHER_PARAMS - 151_936 * 2560) / TEACHER_LAYERS
    head_params = 151_936 * 2560
    # One reference pass per item, then one ablated pass per item per candidate.
    ref_flops = CALIBRATION_ITEMS * CALIBRATION_POSITIONS / CALIBRATION_ITEMS
    tokens_per_item = CALIBRATION_POSITIONS / CALIBRATION_ITEMS
    full_pass = 2 * (TEACHER_LAYERS * layer_params + head_params) * tokens_per_item
    abl_pass = 2 * (STUDENT_LAYERS * layer_params + head_params) * tokens_per_item
    total_flops = CALIBRATION_ITEMS * (full_pass + SUBSET_EVALUATIONS * abl_pass)
    seconds = total_flops / (L40S_SUSTAINED_TFLOPS * 1e12) * SEARCH_OVERHEAD
    return seconds / 60.0, {
        "forward_passes": CALIBRATION_ITEMS * (1 + SUBSET_EVALUATIONS),
        "tokens_per_item": round(tokens_per_item, 1),
        "petaflops": round(total_flops / 1e15, 2),
        "sustained_tflops_assumed": L40S_SUSTAINED_TFLOPS,
        "overhead_factor": SEARCH_OVERHEAD,
        "unused": ref_flops,
    }


def pod_a_plan(authorized: float) -> dict:
    minutes, detail = search_minutes()
    plan = plan_session(
        price_per_hour=PRICE, authorized_usd=authorized,
        arms=0, steps_per_arm=0,
        step_time=StepTime(4.15, "unused; pod A does not train"),
        setup_minutes=SETUP_MIN,
        other_phases=(
            Phase("contribution_search_260_evaluations", round(minutes, 1)),
            Phase("self_consistency_and_positional_baseline", 4.0),
            Phase("artifact_manifest_and_verify", 8.0),
            Phase("artifact_synchronization", 5.0),
        ),
        contingency_fraction=CONTINGENCY,
        artifact_recovery_reserve_minutes=RESERVE_MIN,
    )
    d = plan.as_dict()
    d["label"] = "pod A — contribution search"
    d["search_cost_model"] = detail
    return d


def pod_b_plan(authorized: float, n_arms: int = 2) -> dict:
    plan = plan_session(
        price_per_hour=PRICE, authorized_usd=authorized,
        arms=n_arms, steps_per_arm=STEPS_PER_ARM, step_time=E8_STEP,
        setup_minutes=SETUP_MIN,
        eval_minutes_per_arm=EVAL_MIN_PER_ARM,
        transfer_minutes=n_arms * TRANSFER_MIN_PER_ARM,
        other_phases=(
            # Mandatory and priced: both initializations are measured here, on one
            # device, so step 0 is comparable. The baseline is remeasured rather
            # than inherited — that is the rule, not an extra.
            Phase("init_nll_treatment", INIT_NLL_MIN_PER_CKPT),
            Phase("init_nll_baseline_remeasured", INIT_NLL_MIN_PER_CKPT),
            Phase("pretraining_gate_validate_e8_arms", 3.0),
            Phase("artifact_manifest_and_verify", 8.0),
        ),
        contingency_fraction=CONTINGENCY,
        artifact_recovery_reserve_minutes=RESERVE_MIN,
    )
    d = plan.as_dict()
    d["label"] = f"pod B — init NLL + {n_arms} x 2.96M recovery + evaluation"
    d["n_trained_arms"] = n_arms
    d["train_minutes"] = round(n_arms * STEPS_PER_ARM * E8_STEP.seconds / 60.0, 1)
    return d


def smallest_cap(fn, *args) -> dict:
    """Let the planner decide: a cap that cannot hold its own reserve raises."""
    for cents in range(50, 6000, 25):
        try:
            return fn(cents / 100.0, *args)
        except BudgetError:
            continue
    raise SystemExit(f"no cap under $60 contains {fn.__name__}")


def show(d: dict) -> None:
    print(f"\n=== {d['label']} ===")
    for p in d["breakdown"]:
        if p["minutes"]:
            print(f"  {p['name']:44s} {p['minutes']:7.1f} min  "
                  f"${p['minutes'] / 60 * PRICE:5.2f}")
    print(f"  {'-' * 44} {'-' * 7}")
    for key, name in (("expected", "expected completion"),
                      ("soft_stop", "soft stop (start nothing new)"),
                      ("artifact_recovery_reserve", "artifact-recovery reserve"),
                      ("hard_terminate", "absolute termination")):
        print(f"  {name:44s} {d[key + '_minutes']:7.1f} min  "
              f"${d[key + '_usd']:5.2f}")


def main() -> int:
    a = smallest_cap(pod_a_plan)
    b = smallest_cap(pod_b_plan)
    show(a)
    show(b)

    total_expected = a["expected_usd"] + b["expected_usd"]
    total_hard = a["hard_terminate_usd"] + b["hard_terminate_usd"]
    available = max(0.0, AUTHORIZED_CAP_USD - ACTUAL_BASELINE_USD)
    shortfall = max(0.0, total_hard - available)

    print("\n=== E8 total, and what it needs ===")
    print(f"  free dev-box work (Stage 0 regeneration, init construction,")
    print(f"  calibration set, preregistration, tests)              $ 0.00")
    print(f"  pod A hard terminate                                  ${a['hard_terminate_usd']:5.2f}")
    print(f"  pod B hard terminate                                  ${b['hard_terminate_usd']:5.2f}")
    print(f"  E8 expected completion                                ${total_expected:5.2f}")
    print(f"  E8 hard backstop                                      ${total_hard:5.2f}")
    print(f"\n  actual cumulative spend                              ${ACTUAL_BASELINE_USD:7.3f}")
    print(f"  current cumulative authorization                      ${AUTHORIZED_CAP_USD:7.2f}")
    print(f"  available under it                                    ${available:7.2f}")
    print(f"  ADDITIONAL AUTHORIZATION REQUIRED                     ${shortfall:7.2f}")
    print(f"  proposed new cumulative cap                           "
          f"${ACTUAL_BASELINE_USD + total_hard:7.2f}")
    print("\n  Not reduced to one seed to fit: the behaviour-metric seed noise "
          "floor is 0.1290,\n  wider than any effect E8 could claim, so a "
          "single-seed arm would be unreadable.")

    out = {
        "price_per_hour": PRICE,
        "actual_cumulative_baseline_usd": ACTUAL_BASELINE_USD,
        "current_authorized_cap_usd": AUTHORIZED_CAP_USD,
        "available_under_cap_usd": round(available, 2),
        "pod_a": a, "pod_b": b,
        "e8_expected_usd": round(total_expected, 2),
        "e8_hard_backstop_usd": round(total_hard, 2),
        "additional_authorization_required_usd": round(shortfall, 2),
        "proposed_new_cumulative_cap_usd": round(
            ACTUAL_BASELINE_USD + total_hard, 2),
        "single_seed_not_offered_because":
            "the behaviour-metric seed noise floor is 0.1290, wider than any "
            "effect E8 could claim; a one-seed arm cannot be read",
        "calibration": {"items": CALIBRATION_ITEMS,
                        "prediction_positions": CALIBRATION_POSITIONS},
        "subset_evaluations": SUBSET_EVALUATIONS,
    }
    dest = REPO_ROOT / "logs/e8_budget_plans.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
