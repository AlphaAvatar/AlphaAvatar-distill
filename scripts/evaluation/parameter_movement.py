#!/usr/bin/env python
"""How far each parameter group actually moved from the Stage 1 fork point.

    PYTHONPATH=src python scripts/evaluation/parameter_movement.py \
        --init artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
        --checkpoint <run>/checkpoints/step_001023/model \
        --label A1-sa --out artifacts/audit/e3_movement/A1-sa.json

Reports `‖ΔW‖_F` and the relative `‖ΔW‖_F / ‖W_init‖_F` per parameter, rolled
up per group and per decoder layer. This is what separates "the arm trained a
smaller parameter set" from "the arm actually moved less", which are different
claims and only one of them is interesting.

For a LoRA arm the checkpoint's `model/` holds *merged* weights, so the
attention-projection delta measured here **is** the adapter's contribution.
When `lora_state.safetensors` sits next to it, the delta is also recomputed
independently from `scaling · B @ A` and the two are compared — a merge that
silently dropped or double-counted the adapter would show up as a mismatch
rather than as a plausible-looking number.

Tensors are streamed one at a time (`safe_open`), so this runs on a CPU dev box
without holding two 596M-parameter models in memory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def group_of(name: str) -> str:
    if ".mlp." in name:
        return "ffn"
    if any(f".self_attn.{p}." in name for p in ATTN_PROJ):
        return "attn_proj"
    if ".self_attn.q_norm" in name or ".self_attn.k_norm" in name:
        return "attn_norm"
    if "layernorm" in name:
        return "decoder_norm"
    if name.startswith("model.norm."):
        return "final_norm"
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding"
    return "other"


def shard_files(d: Path) -> list[Path]:
    index = d / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [d / n for n in names]
    single = d / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"no safetensors under {d}")
    return [single]


def open_all(d: Path) -> dict[str, tuple[Path, object]]:
    handles = {}
    for path in shard_files(d):
        f = safe_open(str(path), framework="pt")
        for key in f.keys():
            handles[key] = f
    return handles


def lora_deltas(path: Path) -> dict[str, torch.Tensor]:
    """`scaling · B @ A` per adapted module, recomputed from the saved tensors."""
    meta = json.loads((path.parent / "checkpoint_meta.json").read_text())
    scaling = float(meta["lora_config"]["scaling"])
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        names = sorted(k.split("::", 1)[1] for k in keys if k.startswith("lora_A::"))
        return {
            n: scaling * (f.get_tensor(f"lora_B::{n}").float()
                          @ f.get_tensor(f"lora_A::{n}").float())
            for n in names
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    init = open_all(args.init)
    final = open_all(args.checkpoint)
    missing = sorted(set(init) - set(final)), sorted(set(final) - set(init))
    if any(missing):
        raise SystemExit(f"key sets differ: missing={missing[0][:4]} "
                         f"unexpected={missing[1][:4]}")

    per_param, groups, layers = {}, defaultdict(lambda: [0.0, 0.0, 0]), defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0]))
    for name in sorted(init):
        w0 = init[name].get_tensor(name).float()
        w1 = final[name].get_tensor(name).float()
        if w0.shape != w1.shape:
            raise SystemExit(f"{name}: shape {tuple(w0.shape)} != {tuple(w1.shape)}")
        d = float((w1 - w0).norm())
        n0 = float(w0.norm())
        g = group_of(name)
        per_param[name] = {
            "group": g,
            "numel": w0.numel(),
            "delta_fro": d,
            "init_fro": n0,
            "relative": (d / n0) if n0 > 0 else None,
            "moved": d > 0.0,
        }
        # Group totals accumulate squared norms: the Frobenius norm of the
        # concatenation, not a mean of per-tensor ratios, which would weight a
        # 1024-element norm the same as a 3M-element projection.
        acc = groups[g]
        acc[0] += d * d
        acc[1] += n0 * n0
        acc[2] += w0.numel()
        m = LAYER_RE.match(name)
        if m:
            la = layers[int(m.group(1))][g]
            la[0] += d * d
            la[1] += n0 * n0

    by_group = {
        g: {"delta_fro": v[0] ** 0.5, "init_fro": v[1] ** 0.5,
            "relative": (v[0] ** 0.5 / v[1] ** 0.5) if v[1] > 0 else None,
            "numel": v[2]}
        for g, v in sorted(groups.items())
    }
    by_layer = {
        str(i): {g: {"delta_fro": v[0] ** 0.5,
                     "relative": (v[0] ** 0.5 / v[1] ** 0.5) if v[1] > 0 else None}
                 for g, v in sorted(gs.items())}
        for i, gs in sorted(layers.items())
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "label": args.label,
        "init": str(args.init),
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_sha256": (
            sha256_file(args.checkpoint / "model.safetensors")
            if (args.checkpoint / "model.safetensors").is_file() else None),
        "note": ("relative = ||ΔW||_F / ||W_init||_F, aggregated by summing "
                 "squared norms within a group (not by averaging per-tensor "
                 "ratios). For a LoRA arm, attn_proj movement IS the merged "
                 "adapter delta."),
        "by_group": by_group,
        "by_layer": by_layer,
        "per_parameter": per_param,
    }

    lora_path = args.checkpoint.parent / "lora_state.safetensors"
    if lora_path.is_file():
        deltas = lora_deltas(lora_path)
        checks, per_module = [], {}
        for module, delta in deltas.items():
            key = f"{module}.weight"
            recomputed = float(delta.norm())
            observed = per_param[key]["delta_fro"]
            per_module[module] = {
                "lora_delta_fro": recomputed,
                "merged_minus_init_fro": observed,
                "relative": per_param[key]["relative"],
            }
            tol = max(1e-4, 1e-3 * max(recomputed, observed))
            checks.append(abs(recomputed - observed) <= tol)
        proj = defaultdict(lambda: [0.0, 0])
        for module, v in per_module.items():
            kind = module.rsplit(".", 1)[-1]
            proj[kind][0] += v["lora_delta_fro"] ** 2
            proj[kind][1] += 1
        report["lora"] = {
            "state_file": str(lora_path),
            "n_modules": len(per_module),
            "merged_delta_matches_recomputed_delta": all(checks),
            "by_projection": {k: {"delta_fro": v[0] ** 0.5, "n_modules": v[1]}
                              for k, v in sorted(proj.items())},
            "per_module": per_module,
        }
        if not all(checks):
            raise SystemExit("merged attention delta does not match scaling·B@A")

    report["code_state"] = code_state(REPO_ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    print(f"{args.label}: ||ΔW||_F / ||W_init||_F by group")
    for g, v in by_group.items():
        rel = "n/a" if v["relative"] is None else f"{v['relative']:.6f}"
        print(f"  {g:14s} numel {v['numel']:>12,}  delta {v['delta_fro']:>12.4f}  "
              f"relative {rel}")
    if "lora" in report:
        print("  LoRA delta by projection:")
        for k, v in report["lora"]["by_projection"].items():
            print(f"    {k:8s} {v['delta_fro']:.4f} over {v['n_modules']} modules")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
