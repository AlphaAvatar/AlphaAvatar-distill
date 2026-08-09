#!/usr/bin/env python3
"""Price E7 and the control-plane canary through the four-threshold planner.

Every number in the preregistration comes from here, so the proposal and the
thing that would actually enforce it are the same arithmetic. Run it to
regenerate the tables; it touches no GPU and creates no pod.

    PYTHONPATH=src python scripts/training/plan_e7_budget.py

Phase costs are named separately (P6) and derive from **measured E6b wall
clock**, not from printed step time:

* printed per-step timing on E6b was 4.1485 / 4.1099 s;
* wall clock per step, driver command to `TRAIN_DONE`, was 4.211 / 4.215 s;
* the difference is the periodic evaluations and checkpoint writes, which are
  therefore priced as their own phases rather than hidden inside a step rate.

Setup is budgeted at 45 min rather than the 5–8.5 min a warm image has taken,
because the same script, image and GPU has also taken 150+ min on a cold host.
Budgeting the warm number is how a session arrives at its deadline with the work
unfinished.
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

PRICE = 0.99                      # L40S secure, the rate every Stage 3 arm paid
ACTUAL_BASELINE_USD = 149.59      # actual cumulative spend, not the old cap

STEPS_PER_ARM = 1761              # the 1.60M rung's schedule, unchanged
# 4.15 s/step measured on E6b, plus 10% for the extra stream: +1,024 forward
# tokens per model per step (+6.25% tokens) and +1,023 KD positions through the
# 151k-vocabulary softmax (+22.6% of the KD reduction). Rounded up, and above
# the enforced floor so the planner accepts it as a measured-derived estimate.
E7_STEP = StepTime(4.60, "E6b measured 4.15 s/step + 10% for the extra stream")

SETUP_MIN = 45.0                  # see the docstring; not the warm-image number
EVAL_MIN_PER_ARM = 10.5           # E6b frozen battery 8.25 + general-text ~2
TRANSFER_MIN_PER_ARM = 10.0       # E6b moved 2 x 5.6 GB in ~15 min
RESERVE_MIN = 30.0
CONTINGENCY = 0.10


def arm_plan(n_arms: int, authorized: float, *, label: str) -> dict:
    train_min = n_arms * STEPS_PER_ARM * E7_STEP.seconds / 60.0
    plan = plan_session(
        price_per_hour=PRICE,
        authorized_usd=authorized,
        arms=n_arms,
        steps_per_arm=STEPS_PER_ARM,
        step_time=E7_STEP,
        setup_minutes=SETUP_MIN,
        eval_minutes_per_arm=EVAL_MIN_PER_ARM,
        transfer_minutes=n_arms * TRANSFER_MIN_PER_ARM,
        other_phases=(
            # Arm A is retained: its checkpoints are on the relay and its frozen
            # battery artifacts already exist. Only the general-text diagnostics
            # are new, and they are cheap.
            Phase("arm_A_general_text_diagnostics", 4.0),
            Phase("artifact_manifest_and_verify", 8.0),
        ),
        contingency_fraction=CONTINGENCY,
        artifact_recovery_reserve_minutes=RESERVE_MIN,
    )
    d = plan.as_dict()
    d["label"] = label
    d["n_trained_arms"] = n_arms
    d["train_minutes"] = round(train_min, 1)
    return d


def canary_plan(authorized: float) -> dict:
    """A disposable pod running a harmless short process, not training."""
    plan = plan_session(
        price_per_hour=PRICE,
        authorized_usd=authorized,
        arms=0, steps_per_arm=0,
        step_time=StepTime(4.60, "unused; the canary does not train"),
        setup_minutes=12.0,           # minimal image, no venv build, no weights
        other_phases=(
            Phase("detached_start_and_descriptor", 2.0),
            Phase("log_relay_cycles", 6.0),
            Phase("watchdog_threshold_and_termination", 8.0),
            Phase("artifact_manifest_and_hash_verify", 4.0),
        ),
        contingency_fraction=0.25,     # short sessions are proportionally noisier
        artifact_recovery_reserve_minutes=10.0,
    )
    d = plan.as_dict()
    d["label"] = "live control-plane canary"
    return d


def show(d: dict) -> None:
    print(f"\n=== {d['label']} ===")
    for p in d["breakdown"]:
        if p["minutes"]:
            print(f"  {p['name']:36s} {p['minutes']:7.1f} min  "
                  f"${p['minutes'] / 60 * PRICE:5.2f}")
    print(f"  {'-' * 36} {'-' * 7}")
    for key, name in (("expected", "expected completion"),
                      ("soft_stop", "soft stop (start nothing new)"),
                      ("artifact_recovery_reserve", "artifact-recovery reserve"),
                      ("hard_terminate", "absolute termination")):
        print(f"  {name:36s} {d[key + '_minutes']:7.1f} min  "
              f"${d[key + '_usd']:5.2f}")


def main() -> int:
    out = {"price_per_hour": PRICE,
           "actual_cumulative_baseline_usd": ACTUAL_BASELINE_USD,
           "step_time": {"seconds": E7_STEP.seconds, "source": E7_STEP.source},
           "steps_per_arm": STEPS_PER_ARM}

    canary = canary_plan(1.00)
    out["canary"] = canary
    show(canary)

    # Requested caps are searched upward from the expected cost in 25c steps and
    # the *planner* decides: a cap that cannot contain its own reserve raises.
    for label, n in (("E7 full (B x2 + C x2)", 4), ("E7 reduced (B x2 only)", 2)):
        cap = None
        for cents in range(100, 4000, 25):
            try:
                cap = arm_plan(n, cents / 100.0, label=label)
                break
            except BudgetError:
                continue
        if cap is None:
            raise SystemExit(f"no cap under $40 contains {label}")
        out["full" if n == 4 else "reduced"] = cap
        show(cap)

    print("\n=== proposed cumulative caps, from the ACTUAL baseline ===")
    base = ACTUAL_BASELINE_USD
    c = out["canary"]["hard_terminate_usd"]
    for key, label in (("full", "canary + E7 full"),
                       ("reduced", "canary + E7 reduced")):
        e = out[key]["hard_terminate_usd"]
        print(f"  {label:26s} ${base:.2f} + ${c:.2f} + ${e:.2f} = "
              f"${base + c + e:.2f}")
        out[key + "_proposed_cumulative_cap"] = round(base + c + e, 2)
    print("\n  the historical $149.03 authorization is NOT counted as remaining "
          "balance; it is exceeded and closed.")

    dest = REPO_ROOT / "logs/e7_budget_plans.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
