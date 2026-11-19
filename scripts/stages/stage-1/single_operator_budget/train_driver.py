#!/usr/bin/env python3
"""On-pod driver of SOB1's phase 2 (session kind ``sob1_train``): one arm's two seeds trained with A-MIX's recipe from
the arm's initialization and evaluated at 8,192 tokens -- TPI1's `train_driver` run with this study's authorization,
plan (``--plan plans/training_plan_<arm>.json``), relay prefix root and evaluation package (`launch.package`).

    cd /workspace/aad && HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume \\
        /opt/train/bin/python scripts/stages/stage-1/single_operator_budget/train_driver.py --authorization <a.json> \\
        --out <evidence> --status <file> --plan logs/stages/stage-1/single_operator_budget/plans/training_plan_<arm>.json \\
        --prefix-root durable/stage1_single_operator_budget --unit-deadline-hours <h> [--session-deadline-utc <t>] \\
        --vllm-python /opt/vllm/bin/python [--resume <secured evidence>]

Everything the state machine does is TPI1's (the save-point rotation and its handoff, the listing read with retries
and never taken as empty, the final's takeover, the per-sample sealed evaluation, the memory probe of the first
optimizer steps); the plan's ``durability`` block sets its rates and retries.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))


def main(argv=None) -> int:
    from stages.single_operator_budget import launch as L
    from stages.teacher_parent_initialization import train_driver as TD
    return TD.main(argv, study_launch=L)


if __name__ == "__main__":
    raise SystemExit(main())
