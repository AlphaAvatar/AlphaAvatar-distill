#!/usr/bin/env python
"""Emit the Experiment 2 run configs, one phase at a time.

    uv run python scripts/data/build_experiment2_configs.py --phase d1

Experiment 2 is three sequential single-variable diagnostics at the Experiment 1
2.96M rung. Each phase changes exactly one thing against the arm it is compared
with, and every arm restarts from the Stage 1 PCA init at the two Experiment 1
seeds — never from a trained checkpoint.

| phase | arm | the one change |
|---|---|---|
| 1 data  | `e2_d1` | cleaned targets (`clean-v1`); loss, LR, steps unchanged |
| 2 loss  | `e2_l1` | `ce_weight` 0.25 -> 0.0; data, LR, steps unchanged |
| 3 lr    | `e2_r1` / `e2_r2` | whole LR schedule scaled by 1/2 and 1/4 |

The step budget is **pinned to Experiment 1's 2,916**, not recomputed from the
new pack's block count, because optimizer-step parity is what makes the arms
comparable (2026-08-03 decision). The D1 pack is cut at 1,944 blocks precisely so
that `ceil(1944 * 3 / 2)` reproduces it.

Phase 2 and 3 read `--data-dir` / `--rung` from whatever phase 1 selected, so the
same script emits them once that decision exists.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = REPO_ROOT / "configs/stage3/recovery.json"

# Experiment 1's 2.96M PCA control, which every phase-1 arm must match.
D0_STEPS = 2916
D0_BLOCKS = 1944
EPOCHS = 3
SEEDS = [("a", 20260726), ("b", 20260801)]
PCA_INIT = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"

PHASES = {
    "d1": {
        "prefix": "e2_d1",
        "what": "cleaned teacher targets (aadistill.data.cleaning, clean-v1)",
        "lr_scale": 1.0,
        "ce_weight": None,   # inherit
    },
    "l1": {
        "prefix": "e2_l1",
        "what": "KL-only: the CE term removed, KD preserved exactly",
        "lr_scale": 1.0,
        "ce_weight": 0.0,
    },
    "r1": {
        "prefix": "e2_r1",
        "what": "the complete LR schedule scaled to eta/2",
        "lr_scale": 0.5,
        "ce_weight": None,
    },
    "r2": {
        "prefix": "e2_r2",
        "what": "the complete LR schedule scaled to eta/4",
        "lr_scale": 0.25,
        "ce_weight": None,
    },
}


def build(phase: str, out_dir: Path, data_dir: str, rung: int, val_blocks: int,
          evals: int) -> list[dict]:
    spec = PHASES[phase]
    canonical = json.loads(CANONICAL.read_text())
    bps = canonical["batch"]["blocks_per_step"]
    derived = math.ceil(D0_BLOCKS * EPOCHS / bps)
    if derived != D0_STEPS:
        raise SystemExit(
            f"step parity broken: {D0_BLOCKS} blocks at {EPOCHS} epochs and "
            f"{bps} blocks/step gives {derived}, not Experiment 1's {D0_STEPS}")

    out_dir.mkdir(parents=True, exist_ok=True)
    arms = []
    for seed_tag, seed in SEEDS:
        name = f"{spec['prefix']}_s{seed_tag}_pca"
        cfg = dict(canonical)
        cfg["run_name"] = name
        cfg["_purpose"] = (
            f"Experiment 2 phase {phase}: {spec['what']}. Rung {rung:,} at "
            f"{D0_BLOCKS} blocks / {D0_STEPS} steps, matching the Experiment 1 "
            f"2.96M PCA control exactly. Seed {seed}, Stage 1 PCA init. "
            "Differs from configs/stage3/recovery.json only in the fields this "
            "phase is testing plus data source, rung, seed and schedule length."
        )
        cfg["student_path"] = PCA_INIT
        cfg["data_dir"] = data_dir
        cfg["packing"] = "ladder"
        cfg["rung"] = rung
        cfg["val_blocks"] = val_blocks
        cfg["seed"] = seed

        loss = dict(canonical["loss"])
        if spec["ce_weight"] is not None:
            loss["ce_weight"] = spec["ce_weight"]
        cfg["loss"] = loss

        optim = dict(canonical["optim"])
        optim["lr"] = canonical["optim"]["lr"] * spec["lr_scale"]
        cfg["optim"] = optim

        # The whole schedule scales, not just the peak: `min_lr_frac` is a
        # fraction of the peak, so the floor follows automatically, and warmup
        # stays a fixed proportion of the run rather than a fixed step count.
        cfg["schedule"] = {
            "total_steps": D0_STEPS,
            "warmup_steps": max(10, round(0.05 * D0_STEPS)),
            "min_lr_frac": canonical["schedule"]["min_lr_frac"],
        }
        # Experiment 1 kept only the final checkpoint, so no arm in it has a
        # held-out-NLL trajectory. Every Experiment 2 arm retains one checkpoint
        # per eval point, which is what phase 3 needs to locate the step where
        # deterioration starts.
        eval_every = max(25, D0_STEPS // evals)
        cfg["checkpoint"] = {"save_every": eval_every, "keep_last": evals + 1}
        cfg["intervals"] = {"log_every": 10, "eval_every": eval_every,
                            "eval_blocks": val_blocks}
        cfg["out_dir"] = f"artifacts/stage3/{name}"
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        arms.append({"name": name, "config": str(path.relative_to(REPO_ROOT)),
                     "step_tag": f"step_{D0_STEPS:06d}", "rung": rung,
                     "blocks": D0_BLOCKS, "steps": D0_STEPS, "seed": seed,
                     "lr": optim["lr"], "ce_weight": loss["ce_weight"],
                     "kd_weight": loss["kd_weight"]})
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--out", default="configs/stage3/e2")
    ap.add_argument("--data-dir",
                    default="artifacts/stage3/ladder_uniform_clean_anchored",
                    help="the packed ladder this phase's arms read")
    ap.add_argument("--rung", type=int, default=2_968_828,
                    help="the block-matched cleaned rung (1,944 blocks)")
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--evals", type=int, default=8,
                    help="eval + checkpoint points per run")
    ap.add_argument("--seconds-per-step", type=float, default=3.53,
                    help="measured Experiment 1 rate at this rung")
    ap.add_argument("--hourly-usd", type=float, default=0.99)
    args = ap.parse_args()

    arms = build(args.phase, REPO_ROOT / args.out, args.data_dir, args.rung,
                 args.val_blocks, args.evals)
    hours = sum(a["steps"] for a in arms) * args.seconds_per_step / 3600
    print(f"wrote {len(arms)} configs to {args.out}")
    for a in arms:
        print(f"  {a['name']:22s} rung {a['rung']:>9,}  {a['blocks']:>5} blocks  "
              f"{a['steps']:>5} steps  lr {a['lr']:.2e}  "
              f"ce {a['ce_weight']} kd {a['kd_weight']}")
    print(f"\ntraining only: {hours:.2f} h ~ ${hours * args.hourly_usd:.2f} "
          f"at ${args.hourly_usd}/h (excludes setup, evaluation and teardown)")


if __name__ == "__main__":
    main()
