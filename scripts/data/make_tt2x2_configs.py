"""Emit the four configs of the Stage 3 teacher-target 2x2.

Pre-registration: logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md

Generated rather than hand-written on purpose. The experiment's validity rests on
the four runs being identical in everything except (a) which arm's data dir they
read and (b) the seed. Four hand-edited JSON files are exactly how that
invariant drifts — one stale `total_steps` and the comparison silently becomes a
compute ablation. Here the shared body is built once and the only per-file
differences are the two intended ones, which is also checked before writing.

Sizing follows the maintainer's decision of 2026-07-30: **hold total training
tokens equal** (steps x blocks_per_step x block_len), and let the number of
passes over the prompt set differ between arms because teacher targets are much
longer. The step count is therefore derived from the *treatment* arm — the arm
whose epochs we actually want to control — and then applied unchanged to both.

Usage:
    uv run python scripts/data/make_tt2x2_configs.py \
        --pilot-manifest data/stage3_pilot/manifest.json --epochs 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Identical to every logged Stage 3 run: 2 x 8192 = 16,384 tokens per optimizer
# step, the same tokens/step as 16 x 1024 and 8 x 2048. Holding this fixed is
# what makes "total training tokens" comparable with the runs before it.
BLOCKS_PER_STEP = 2
MICRO_BLOCKS = 1  # a block cannot be split; at 8192 this is the only option

TRAINABLE = [
    "\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\\.",
    "\\.mlp\\.(gate_proj|up_proj|down_proj)\\.",
    "input_layernorm",
    "post_attention_layernorm",
    "model\\.norm\\.",
]

ARMS = {
    "ctrl": ("control", "public v1 targets"),
    "treat": ("treatment", "teacher-native targets (trace + answer)"),
}
SEEDS = {"a": 20260726, "b": 20260728}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-manifest", default="data/stage3_pilot/manifest.json")
    ap.add_argument("--pilot-dir", default="data/stage3_pilot")
    ap.add_argument("--start-checkpoint",
                    default="artifacts/stage3/s2v1_from_init/checkpoints/step_002700/model")
    ap.add_argument("--block-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=3.0,
                    help="passes over the TREATMENT arm; sets the shared budget")
    ap.add_argument("--total-steps", type=int, default=None,
                    help="override the derived step count")
    # A short warm-up from an already-recovered checkpoint, not a 2700-step run
    # from a random init: the baseline's 2e-4 would move this checkpoint a long
    # way in ~150 steps. Both arms share whatever is set here, so the comparison
    # holds either way; R4 aborts an arm whose val CE rises above step 0.
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="set from the block_len 8192 memory smoke test")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-blocks", type=int, default=16)
    ap.add_argument("--out-dir", default="configs/stage3")
    args = ap.parse_args()

    manifest = json.loads((REPO_ROOT / args.pilot_manifest).read_text())
    key = f"packed_best_fit_{args.block_len}"
    try:
        treat_blocks = manifest["arms"]["treatment"]["tokens"][key]["train"]["blocks"]
        ctrl_blocks = manifest["arms"]["control"]["tokens"][key]["train"]["blocks"]
    except KeyError:
        raise SystemExit(
            f"{args.pilot_manifest} has no packing stats at block_len "
            f"{args.block_len}; rebuild the arms with --block-len {args.block_len}"
        )

    if args.total_steps:
        total_steps = args.total_steps
    else:
        total_steps = max(1, math.ceil(args.epochs * treat_blocks / BLOCKS_PER_STEP))
    warmup = max(1, round(total_steps * args.warmup_frac))
    eval_every = args.eval_every or max(1, total_steps // 6)

    tokens_total = total_steps * BLOCKS_PER_STEP * args.block_len
    # Reported, not corrected: the arms cannot have equal total tokens AND equal
    # passes when one arm's samples are several times longer (preflight 6).
    ctrl_epochs = total_steps * BLOCKS_PER_STEP / ctrl_blocks
    treat_epochs = total_steps * BLOCKS_PER_STEP / treat_blocks

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for arm_tag, (arm_dir, description) in ARMS.items():
        for seed_tag, seed in SEEDS.items():
            name = f"tt2x2_{arm_tag}_{seed_tag}"
            cfg = {
                "stage": "stage3_recovery",
                "run_name": name,
                "_purpose": f"teacher-target 2x2 — {description}, seed {seed}",
                "student_path": args.start_checkpoint,
                "teacher": {
                    "model_id": "Qwen/Qwen3-4B-Thinking-2507",
                    "revision": "768f209d9ea81521153ed38c47d515654e938aea",
                    "dtype": "bfloat16",
                },
                "data_dir": f"{args.pilot_dir}/{arm_dir}",
                "groups": None,
                "packing": "best_fit",
                "block_len": args.block_len,
                "dtype": "float32",
                "autocast_bf16": True,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "device": "auto",
                "seed": seed,
                "trainable_patterns": TRAINABLE,
                "loss": {"ce_weight": 0.25, "kd_weight": 1.0,
                         "kd_temperature": 1.0, "kd_scope": "all"},
                "optim": {"lr": args.lr, "weight_decay": 0.01,
                          "betas": [0.9, 0.95], "eps": 1e-08, "grad_clip": 1.0},
                "schedule": {"total_steps": total_steps, "warmup_steps": warmup,
                             "min_lr_frac": 0.1},
                "batch": {"blocks_per_step": BLOCKS_PER_STEP,
                          "micro_blocks": MICRO_BLOCKS},
                "checkpoint": {"save_every": max(1, total_steps // 3),
                               "keep_last": 2},
                "intervals": {"log_every": 5, "eval_every": eval_every,
                              "eval_blocks": args.eval_blocks},
                "out_dir": f"artifacts/stage3/{name}",
            }
            path = out_dir / f"{name}.json"
            path.write_text(json.dumps(cfg, indent=2) + "\n")
            written[name] = cfg

    # The invariant this script exists to protect: every config identical except
    # data_dir, seed, run_name, out_dir and the human-readable purpose note.
    allowed = {"data_dir", "seed", "run_name", "out_dir", "_purpose"}
    base_name, base = next(iter(written.items()))
    for name, cfg in written.items():
        differing = {k for k in base if json.dumps(base[k], sort_keys=True)
                     != json.dumps(cfg[k], sort_keys=True)}
        unexpected = differing - allowed
        if unexpected:
            raise SystemExit(
                f"{name} differs from {base_name} in {sorted(unexpected)}, which "
                "would confound the comparison")
    print(f"wrote {len(written)} configs to {args.out_dir}: "
          f"{', '.join(sorted(written))}")
    print(f"  identical except {sorted(allowed)}")
    print(f"\n  train blocks   : treatment {treat_blocks}, control {ctrl_blocks}")
    print(f"  total steps    : {total_steps} (warmup {warmup}, eval every {eval_every})")
    print(f"  tokens/step    : {BLOCKS_PER_STEP * args.block_len}")
    print(f"  total tokens   : {tokens_total:,} — IDENTICAL across arms")
    print(f"  passes         : treatment {treat_epochs:.2f}, control "
          f"{ctrl_epochs:.2f} — UNEQUAL by construction, reported not corrected")
    print(f"  lr {args.lr}, gradient_checkpointing "
          f"{bool(args.gradient_checkpointing)}")

    step_tag = f"step_{total_steps:06d}"
    print(f"\n  STEP_TAG for run_env.sh: {step_tag}")
    print(f"  ABORT_CHECK_STEP for run_env.sh: {eval_every}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
