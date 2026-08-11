#!/usr/bin/env python3
"""Size E8b's depth-only arms before renting anything, and pick the GPU from it.

The depth-only checkpoints (DP/DC) are **not** target-size students: 3.215B
parameters against the 596M student, 5.39x. The canonical Stage 3 recipe trains in
float32 master weights with bf16 autocast and gradient checkpointing, and holds the
4B teacher resident for KD. Whether that fits a 48 GB card is an arithmetic
question, and this answers it rather than assuming.

    PYTHONPATH=src python scripts/training/size_e8b_memory.py

Every term is named so a wrong assumption is visible rather than buried:

* **params** — float32 master weights for the whole model (`dtype: float32` in the
  canonical config, with `autocast_bf16: true` for the math).
* **grads** — float32, allocated only for the *trainable* set: the canonical
  `trainable_patterns` exclude embeddings and the tied lm_head.
* **Adam states** — two float32 moments over the trainable set.
* **teacher** — bf16, resident, no grads.
* **logits** — student and teacher, `blocks_per_step x block_len x vocab`, plus a
  float32 reduction working set for the KD/CE terms.
* **activations** — with gradient checkpointing, the stored set is one tensor per
  layer boundary; recompute peak is bounded by a single layer's internals.

Attention is SDPA/flash, so the O(L^2) score matrix is not materialized; that is
recorded as an assumption because an eager fallback would break it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

GB = 1e9

# From the canonical E1/P1 KD-heavy config, unchanged by E8b.
BLOCK_LEN = 8192
BLOCKS_PER_STEP = 2
MICRO_BLOCKS = 1
VOCAB = 151_936

ARMS = {
    # name: (total params, trainable params, hidden, layers, intermediate)
    "DP/DC depth-only": (3_215_021_568, 2_826_065_408, 2560, 28, 9728),
    "FP/FC target": (596_049_920, 440_467_456, 1024, 28, 3072),
}
TEACHER_PARAMS = 4_022_468_096


def size(total: int, trainable: int, hidden: int, layers: int, inter: int) -> dict:
    params_fp32 = total * 4
    grads_fp32 = trainable * 4
    adam_fp32 = trainable * 4 * 2
    teacher_bf16 = TEACHER_PARAMS * 2

    tok = MICRO_BLOCKS * BLOCK_LEN
    # Student and teacher logits for the microbatch, bf16 under autocast, plus a
    # float32 chunked reduction working set for CE/KD.
    logits = 2 * tok * VOCAB * 2
    kd_reduction = 3 * 512 * VOCAB * 4          # chunked, as the trainer reduces

    # Gradient checkpointing: one saved tensor per layer boundary, bf16.
    ckpt_boundaries = (layers + 1) * tok * hidden * 2
    # Recompute peak inside one layer: the SwiGLU intermediate dominates.
    recompute = 3 * tok * inter * 2 + 6 * tok * hidden * 2
    # Teacher forward, no grads, one layer's working set at a time.
    teacher_fwd = 3 * tok * 9728 * 2 + 6 * tok * 2560 * 2

    activations = ckpt_boundaries + recompute + teacher_fwd
    student_state = params_fp32 + grads_fp32 + adam_fp32
    peak = student_state + teacher_bf16 + logits + kd_reduction + activations
    return {
        "params_fp32_gb": params_fp32 / GB,
        "grads_fp32_gb": grads_fp32 / GB,
        "adam_states_fp32_gb": adam_fp32 / GB,
        "student_state_gb": student_state / GB,
        "teacher_bf16_gb": teacher_bf16 / GB,
        "logits_gb": logits / GB,
        "kd_reduction_gb": kd_reduction / GB,
        "activations_gb": activations / GB,
        "expected_peak_gb": peak / GB,
        # Allocator fragmentation and cuBLAS/cuDNN workspaces are real and not
        # captured above; 15% is the margin every prior session has needed.
        "peak_with_15pct_margin_gb": peak * 1.15 / GB,
        "checkpoint_bf16_gb": total * 2 / GB,
    }


def main() -> int:
    out = {"block_len": BLOCK_LEN, "blocks_per_step": BLOCKS_PER_STEP,
           "micro_blocks": MICRO_BLOCKS, "vocab": VOCAB,
           "precision": "float32 master weights, bf16 autocast, "
                        "gradient checkpointing on",
           "assumptions": [
               "SDPA/flash attention: the O(L^2) score matrix is never "
               "materialized. An eager fallback would add "
               f"{MICRO_BLOCKS * 32 * BLOCK_LEN**2 * 2 / GB:.0f} GB and break this.",
               "embeddings and the tied lm_head are frozen, so they carry no "
               "gradient or optimizer state",
               "one microbatch of one block at a time, as the canonical config sets",
           ],
           "arms": {}}
    for name, spec in ARMS.items():
        out["arms"][name] = size(*spec)

    for name, m in out["arms"].items():
        print(f"\n=== {name} ===")
        for k in ("params_fp32_gb", "grads_fp32_gb", "adam_states_fp32_gb",
                  "student_state_gb", "teacher_bf16_gb", "logits_gb",
                  "kd_reduction_gb", "activations_gb"):
            print(f"  {k:26s} {m[k]:8.2f} GB")
        print(f"  {'-'*26} {'-'*8}")
        print(f"  {'expected peak':26s} {m['expected_peak_gb']:8.2f} GB")
        print(f"  {'+15% margin':26s} {m['peak_with_15pct_margin_gb']:8.2f} GB")
        print(f"  {'checkpoint on disk (bf16)':26s} {m['checkpoint_bf16_gb']:8.2f} GB")

    print("\n=== does it fit? ===")
    for card, vram in (("L40S", 48), ("A100 80GB", 80), ("H100 80GB", 80),
                       ("H200", 141)):
        row = []
        for name, m in out["arms"].items():
            need = m["peak_with_15pct_margin_gb"]
            row.append(f"{name.split()[0]}: {'YES' if need <= vram else 'NO'}")
        print(f"  {card:12s} {vram:4d} GB  ->  " + " | ".join(row))
    dest = REPO_ROOT / "logs/e8b_memory_sizing.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
