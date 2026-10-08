"""Emit the 24 Experiment-1 run configs from the canonical recovery recipe.

    uv run python scripts/data/build_experiment1_configs.py --out configs/stage3/e1

Experiment 1 asks one question — does behavioural recovery scale with
teacher-generated supervised tokens — so every arm is the canonical
`configs/stage3/recovery.json` with exactly four fields changed: the packed
ladder it reads, the rung, the seed, and the start checkpoint. Nothing about the
objective, optimizer, freeze set, packing or precision moves, because anything
that moves is a second variable.

Matrix: 6 rungs x 2 seeds x 2 initializations. Steps are fixed *passes* (3
epochs) rather than a fixed optimizer budget, so each rung is trained to
comparable exposure and the curve measures data quantity — a fixed step budget
would conflate "more data" with "fewer passes", which is what made the 137-step
runs uninterpretable (2026-07-30).

Arm order is deliberate: the whole rung series for the primary cell first, then
the second seed, then the random-init axis. If the budget stops the session
early, what exists is a complete curve rather than four partial ones.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = REPO_ROOT / "configs/stage3/recovery.json"

# (target supervised tokens, blocks) — measured from the uniform pack, not
# nominal. Blocks are what set the step count.
RUNGS = [
    (250_000, 216),
    (460_000, 380),
    (860_000, 682),
    (1_600_000, 1174),
    (2_960_000, 1944),
    (5_500_000, 2941),
]
RUNG_TAGS = {250_000: "0250k", 460_000: "0460k", 860_000: "0860k",
             1_600_000: "1600k", 2_960_000: "2960k", 5_500_000: "5500k"}

# The pinned comparability seed first (every logged Stage 3 run used it).
SEEDS = [("a", 20260726), ("b", 20260801)]
INITS = [
    ("pca", "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"),
    ("rand", "artifacts/stage1/qwen3_0p6b_init_v0/random_baseline"),
]
EPOCHS = 3


def arm_order():
    """Primary cell's full rung series first, so an early stop still yields a curve."""
    for init_tag, init_path in INITS:
        for seed_tag, seed in SEEDS:
            for rung, blocks in RUNGS:
                yield init_tag, init_path, seed_tag, seed, rung, blocks


def build(out_dir: Path, packed_dir: str, val_blocks: int) -> list[dict]:
    canonical = json.loads(CANONICAL.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = []
    for init_tag, init_path, seed_tag, seed, rung, blocks in arm_order():
        bps = canonical["batch"]["blocks_per_step"]
        total_steps = math.ceil(blocks * EPOCHS / bps)
        name = f"e1_r{RUNG_TAGS[rung]}_s{seed_tag}_{init_tag}"
        cfg = dict(canonical)
        cfg["run_name"] = name
        cfg["_purpose"] = (
            f"Experiment 1 (data scaling): rung {rung:,} supervised tokens, "
            f"{blocks} blocks, {EPOCHS} epochs, seed {seed}, {init_tag} init. "
            "Differs from configs/stage3/recovery.json only in data source, "
            "rung, seed, student_path and the derived schedule."
        )
        cfg["student_path"] = init_path
        cfg["data_dir"] = packed_dir
        cfg["packing"] = "ladder"
        cfg["rung"] = rung
        cfg["val_blocks"] = val_blocks
        cfg["seed"] = seed
        cfg["schedule"] = {
            "total_steps": total_steps,
            "warmup_steps": max(10, round(0.05 * total_steps)),
            "min_lr_frac": canonical["schedule"]["min_lr_frac"],
        }
        # One mid-run checkpoint is enough to resume a lost pod without paying
        # for saves that are never read.
        cfg["checkpoint"] = {"save_every": max(200, total_steps // 2),
                             "keep_last": 1}
        cfg["intervals"] = {"log_every": 10,
                            "eval_every": max(25, total_steps // 8),
                            "eval_blocks": val_blocks}
        cfg["out_dir"] = f"artifacts/stage3/{name}"
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        arms.append({"name": name, "config": str(path.relative_to(REPO_ROOT)),
                     "step_tag": f"step_{total_steps:06d}", "rung": rung,
                     "blocks": blocks, "steps": total_steps, "seed": seed,
                     "init": init_tag})
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/stage3/e1")
    ap.add_argument("--packed-dir", default="artifacts/stage3/ladder_uniform",
                    help="the uniform token-ladder pack the arms read")
    ap.add_argument("--val-blocks", type=int, default=16)
    ap.add_argument("--seconds-per-step", type=float, default=4.3,
                    help="measured L40S rate, for the cost projection")
    ap.add_argument("--hourly-usd", type=float, default=0.99)
    args = ap.parse_args()

    arms = build(REPO_ROOT / args.out, args.packed_dir, args.val_blocks)
    total_steps = sum(a["steps"] for a in arms)
    hours = total_steps * args.seconds_per_step / 3600
    print(f"{len(arms)} arms, {total_steps:,} steps")
    print(f"projected training: {hours:.1f} h = ${hours * args.hourly_usd:.2f} "
          f"at {args.seconds_per_step}s/step, ${args.hourly_usd}/h")
    print("ARMS=(")
    for a in arms:
        print(f'  "{a["name"]}|{a["config"]}|{a["step_tag"]}"')
    print(")")


if __name__ == "__main__":
    main()
