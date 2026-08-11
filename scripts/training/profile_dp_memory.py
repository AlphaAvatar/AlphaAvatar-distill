#!/usr/bin/env python3
"""Locate E8b-S2's OOM by measurement, not by arithmetic.

    PYTHONPATH=src python scripts/training/profile_dp_memory.py \
        --config configs/stage3/e8b/e8b_dp_r1600k_sa.json \
        --steps 2 --out artifacts/audit/e8b_dp_memory_profile.json

The gate died with `Tried to allocate 298.00 MiB` at
`kd_forward_kl: torch.log_softmax(tp[i:i+chunk].float() / temperature)`, with
72.44 GiB allocated and 6.16 GiB reserved-but-unallocated on an 80 GB A100. Arithmetic
says the fixed cost is 54.82 GB (fp32 master weights 12.86 + fp32 grads 11.30 + AdamW
m,v 22.61 + teacher bf16 8.04) leaving ~17.6 GB of transients, and that the transients
are dominated by `[tokens, vocab]` materializations at block_len 8192 and vocab 151,936
— not by attention, which is already flash SDPA and O(T) in memory.

This script checks that claim on the real workload. It reports
`max_memory_allocated` and `max_memory_reserved` around each phase — teacher forward,
student forward, CE, KD, backward, optimizer step — plus the largest live CUDA tensors
by size, so the peak is attributed to an operation rather than guessed.

It runs the real trainer's own functions on the real config. It does not train: it
takes `--steps` steps and exits, and it never writes a checkpoint.

CPU-safe: without CUDA it reports the analytic ledger only and says so, which is how
the report's arithmetic is checked before any GPU time is bought.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

VOCAB = 151_936


def gib(n: float) -> float:
    return round(n / 1024 ** 3, 3)


class Probe:
    """Memory around each phase, keyed by name, in order."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.rows: list[dict] = []

    def mark(self, phase: str) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.rows.append({
            "phase": phase,
            "allocated_gib": gib(torch.cuda.memory_allocated()),
            "reserved_gib": gib(torch.cuda.memory_reserved()),
            "max_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "max_reserved_gib": gib(torch.cuda.max_memory_reserved()),
        })

    def largest_tensors(self, k: int = 12) -> list[dict]:
        """The biggest live CUDA tensors, so a peak has a name."""
        if not self.enabled:
            return []
        seen, out = set(), []
        for obj in gc.get_objects():
            try:
                if not (torch.is_tensor(obj) and obj.is_cuda):
                    continue
                key = (obj.data_ptr(), tuple(obj.shape))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"shape": list(obj.shape), "dtype": str(obj.dtype),
                            "mib": round(obj.numel() * obj.element_size() / 1e6, 1),
                            "requires_grad": bool(obj.requires_grad)})
            except Exception:
                continue
        return sorted(out, key=lambda r: -r["mib"])[:k]


def analytic_ledger(block_len: int, micro: int) -> dict:
    t = block_len * micro
    n = t - 1
    chunk = 512
    return {
        "note": "arithmetic, for cross-checking the measurement",
        "block_len": block_len, "micro_blocks": micro, "vocab": VOCAB,
        "student_logits_bf16_mb": round(t * VOCAB * 2 / 1e6, 1),
        "teacher_logits_bf16_mb": round(t * VOCAB * 2 / 1e6, 1),
        "masked_ce_sel_copy_bf16_mb": round(n * VOCAB * 2 / 1e6, 1),
        "masked_ce_sel_float_fp32_mb": round(n * VOCAB * 4 / 1e6, 1),
        "kd_sp_copy_bf16_mb": round(n * VOCAB * 2 / 1e6, 1),
        "kd_tp_copy_bf16_mb": round(n * VOCAB * 2 / 1e6, 1),
        "kd_chunk_buffer_fp32_mb": round(chunk * VOCAB * 4 / 1e6, 1),
        "kd_chunk_concurrent_peak_mb": round(3 * chunk * VOCAB * 4 / 1e6, 1),
        "fixed_gb": {
            "student_weights_fp32": 12.86, "grads_fp32_trainable": 11.30,
            "adamw_m_v_fp32": 22.61, "teacher_bf16": 8.04, "total": 54.82},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((REPO_ROOT / args.config).read_text()
                     if not Path(args.config).is_absolute()
                     else Path(args.config).read_text())
    cuda = torch.cuda.is_available()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cuda_available": cuda,
        "config": args.config,
        "analytic": analytic_ledger(cfg["block_len"],
                                   cfg["batch"]["micro_blocks"]),
    }

    if not cuda:
        report["status"] = ("CPU-only: analytic ledger reported, nothing measured. "
                            "Run this on the A100 to attribute the peak.")
        out = (REPO_ROOT / args.out if not Path(args.out).is_absolute()
               else Path(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["analytic"], indent=2))
        print(f"\n{report['status']}\n-> {out}")
        return 0

    # --- GPU path: the real trainer, the real config ------------------------
    import os

    from transformers import AutoModelForCausalLM

    from aadistill.data.ladder import ladder_blocks
    from aadistill.models.teacher import DTYPES, load_teacher
    from aadistill.training.train import Trainer

    probe = Probe(True)
    report["allocator_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)")
    report["device"] = torch.cuda.get_device_name(0)
    report["capacity_gib"] = gib(torch.cuda.get_device_properties(0).total_memory)
    probe.mark("start")

    pack = REPO_ROOT / cfg["data_dir"]
    train, val, _ = ladder_blocks(pack, cfg["rung"], n_val=cfg["val_blocks"])
    t = cfg["teacher"]
    teacher, _, _ = load_teacher(t["model_id"], t["revision"], dtype=t["dtype"],
                                 device="cuda")
    probe.mark("teacher_loaded")
    student = AutoModelForCausalLM.from_pretrained(
        REPO_ROOT / cfg["student_path"], dtype=DTYPES[cfg["dtype"]])
    probe.mark("student_loaded")
    report["resolved_attn_implementation"] = {
        "student": getattr(student.config, "_attn_implementation", "?"),
        "teacher": getattr(teacher.config, "_attn_implementation", "?")}
    report["module_classes"] = {
        "attention": type(student.model.layers[0].self_attn).__name__,
        "norm": type(student.model.layers[0].input_layernorm).__name__,
        "mlp": type(student.model.layers[0].mlp).__name__,
        "rope": type(student.model.rotary_emb).__name__}

    # Positional order matches scripts/training/train_stage3.py: the class is
    # `Trainer` and `train_blocks` is positional, so a keyword guess would fail
    # here for the first time on a billing pod.
    trainer = Trainer(cfg, student, train, val, teacher=teacher, device="cuda")
    probe.mark("trainer_built")
    report["optimizer"] = {
        "class": type(trainer.opt).__name__,
        "foreach": trainer.opt.param_groups[0].get("foreach"),
        "fused": trainer.opt.param_groups[0].get("fused"),
        "n_trainable_tensors": len(trainer.params),
        "n_trainable_params": int(sum(p.numel() for p in trainer.params))}
    report["teacher_frozen"] = {
        "training_mode": bool(trainer.teacher.training),
        "any_param_requires_grad": any(p.requires_grad
                                       for p in trainer.teacher.parameters()),
        "use_cache": bool(trainer.teacher.config.use_cache)}
    report["student_setup"] = {
        "use_cache": bool(trainer.student.config.use_cache),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing")),
        "autocast_bf16": bool(cfg.get("autocast_bf16")),
        "weight_dtype": str(next(trainer.student.parameters()).dtype)}

    torch.cuda.reset_peak_memory_stats()
    probe.mark("before_step")
    for i in range(args.steps):
        m = trainer.step_once()
        probe.mark(f"after_step_{i + 1}")
        report.setdefault("step_metrics", []).append(
            {k: v for k, v in m.items()
             if isinstance(v, (int, float, str))} if isinstance(m, dict) else str(m))

    report["peak"] = {"max_allocated_gib": gib(torch.cuda.max_memory_allocated()),
                      "max_reserved_gib": gib(torch.cuda.max_memory_reserved())}
    report["largest_live_tensors"] = probe.largest_tensors()
    report["phases"] = probe.rows
    try:
        report["memory_summary"] = torch.cuda.memory_summary(abbreviated=True)
    except Exception as exc:                                 # never fail the probe
        report["memory_summary_error"] = f"{type(exc).__name__}: {exc}"

    out = (REPO_ROOT / args.out if not Path(args.out).is_absolute()
           else Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"peak allocated {report['peak']['max_allocated_gib']} GiB, "
          f"reserved {report['peak']['max_reserved_gib']} GiB, "
          f"capacity {report['capacity_gib']} GiB")
    for r in probe.rows:
        print(f"  {r['phase']:22s} alloc {r['allocated_gib']:7.3f}  "
              f"reserved {r['reserved_gib']:7.3f}  peak {r['max_allocated_gib']:7.3f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
