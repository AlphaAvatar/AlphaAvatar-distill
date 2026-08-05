#!/usr/bin/env python
"""Experiment 3 pre-launch gate: prove the arms on the REAL student, on CPU.

    PYTHONPATH=src python scripts/training/validate_e3_arms.py \
        --out artifacts/audit/e3_prelaunch_validation.json

Everything here is cheap and hardware-free (P8), and every check is one that
would otherwise only fail after GPU money had been spent:

* the Stage 1 fork point still hashes to the pinned value;
* each arm's freeze policy, applied to the real 596M-parameter geometry,
  trains exactly the parameters it claims to and nothing else;
* A1 and A2 share one full-rank trainable set, so the LoRA adapter is the only
  difference between them;
* A2's adapter is a true no-op at initialization — on the real weights, in
  BF16, the logits before and after `apply_lora` are identical;
* merging reproduces the live LoRA model within BF16 tolerance and yields a
  plain checkpoint with no adapter keys.

No optimizer step is taken and nothing is written into any run directory.
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

from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402
from aadistill.training.lora import (  # noqa: E402
    LoRAConfig, apply_lora, lora_report, merged_state_dict,
)
from aadistill.training.train import select_trainable, validate_train_config  # noqa: E402

INIT = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
INIT_SHA = "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"

ARMS = {
    "A0_control_p2_sa": "configs/stage3/p2/p2_ceheavy_sa.json",
    "A0_control_p2_sb": "configs/stage3/p2/p2_ceheavy_sb.json",
    "A1_frozen_attn_sa": "configs/stage3/e3/e3_a1_frozen_attn_sa.json",
    "A1_frozen_attn_sb": "configs/stage3/e3/e3_a1_frozen_attn_sb.json",
    "A2_lora_attn_sa": "configs/stage3/e3/e3_a2_lora_attn_sa.json",
    "A2_lora_attn_sb": "configs/stage3/e3/e3_a2_lora_attn_sb.json",
}

ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


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


def load_student(dtype=torch.float32):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(INIT, dtype=dtype)


def by_group(model, lora_names: set[str]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        if name in lora_names:
            continue
        g = counts.setdefault(group_of(name), {"trainable": 0, "frozen": 0})
        g["trainable" if param.requires_grad else "frozen"] += param.numel()
    return dict(sorted(counts.items()))


def check_arm(alias: str, cfg_path: str) -> dict:
    cfg = json.loads((REPO_ROOT / cfg_path).read_text())
    validate_train_config(cfg)
    model = load_student()
    lora_modules, lora_meta = {}, None
    baseline_logits = None
    ids = torch.arange(1, 33).unsqueeze(0)

    if cfg.get("lora"):
        model.eval()
        with torch.no_grad():
            baseline_logits = model(ids).logits.to(torch.bfloat16).clone()
        lora_cfg = LoRAConfig.from_dict(cfg["lora"])
        lora_modules = apply_lora(model, lora_cfg)
        lora_meta = lora_report(lora_modules, lora_cfg)

    report = select_trainable(model, cfg["trainable_patterns"], lora_modules)
    lora_names = set(report["lora_trainable_names"])
    out = {
        "config": cfg_path,
        "config_sha256": sha256_json(cfg),
        "seed": cfg["seed"],
        "trainable_params": report["trainable_params"],
        "full_rank_trainable_params": report["full_rank_trainable_params"],
        "lora_trainable_params": report["lora_trainable_params"],
        "total_params": report["total_params"],
        "trainable_fraction": round(
            report["trainable_params"] / report["total_params"], 6),
        "by_group": by_group(model, lora_names),
    }
    if lora_meta:
        out["lora"] = {k: v for k, v in lora_meta.items() if k != "lora_modules"}
        out["lora"]["n_target_modules"] = lora_meta["n_lora_modules"]

        # The adapter must be a no-op at initialization, on the real weights.
        model.eval()
        with torch.no_grad():
            after = model(ids).logits.to(torch.bfloat16)
        out["lora"]["init_logits_identical_to_base"] = bool(
            torch.equal(baseline_logits, after))
        out["lora"]["init_max_abs_logit_delta"] = float(
            (after.float() - baseline_logits.float()).abs().max())

        # Merging must reproduce the live model and leave no adapter behind.
        merged = merged_state_dict(model, lora_modules)
        out["lora"]["merged_has_lora_keys"] = any("lora" in k for k in merged)
        out["lora"]["merged_key_count"] = len(merged)
        out["lora"]["merge_is_identity_at_init"] = all(
            torch.equal(merged[f"{n}.weight"], m.weight)
            for n, m in lora_modules.items())
    del model
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts/audit/e3_prelaunch_validation.json")
    args = ap.parse_args()

    init_sha = sha256_file(INIT / "model.safetensors")
    if init_sha != INIT_SHA:
        raise SystemExit(f"Stage 1 init hash mismatch: {init_sha}")

    arms = {}
    for alias, path in ARMS.items():
        print(f"--- {alias}", flush=True)
        arms[alias] = check_arm(alias, path)
        a = arms[alias]
        print(f"    trainable {a['trainable_params']:,} "
              f"(full-rank {a['full_rank_trainable_params']:,} + "
              f"lora {a['lora_trainable_params']:,}) of {a['total_params']:,}",
              flush=True)

    failures = []

    def require(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    for alias, a in arms.items():
        groups = a["by_group"]
        require(groups["embedding"]["trainable"] == 0,
                f"{alias}: embeddings must stay frozen")
        require(groups["ffn"]["frozen"] == 0, f"{alias}: all FFN must train")
        for g in ("decoder_norm", "final_norm", "attn_norm"):
            require(groups[g]["frozen"] == 0, f"{alias}: all {g} must train")
        if alias.startswith("A0"):
            require(groups["attn_proj"]["frozen"] == 0,
                    f"{alias}: the control trains attention projections")
        else:
            require(groups["attn_proj"]["trainable"] == 0,
                    f"{alias}: attention projection weights must be frozen")
        if alias.startswith("A2"):
            lora = a["lora"]
            require(lora["n_target_modules"] == 4 * 28,
                    f"{alias}: expected 112 adapted modules, got "
                    f"{lora['n_target_modules']}")
            require(lora["init_logits_identical_to_base"],
                    f"{alias}: adapter is not a no-op at initialization")
            require(not lora["merged_has_lora_keys"],
                    f"{alias}: merged state dict still carries adapter keys")
            require(lora["merge_is_identity_at_init"],
                    f"{alias}: merge changed weights at initialization")
        else:
            require(a["lora_trainable_params"] == 0,
                    f"{alias}: non-LoRA arm reports LoRA parameters")

    # A1 and A2 must share one full-rank trainable set; only the adapter differs.
    for seed in ("sa", "sb"):
        a1, a2 = arms[f"A1_frozen_attn_{seed}"], arms[f"A2_lora_attn_{seed}"]
        require(a1["full_rank_trainable_params"] == a2["full_rank_trainable_params"],
                f"{seed}: A1/A2 full-rank trainable sets differ")
        require(a1["by_group"] == a2["by_group"],
                f"{seed}: A1/A2 per-group freeze layout differs")
        require(a2["lora_trainable_params"] > 0, f"{seed}: A2 has no adapter")
        a0 = arms[f"A0_control_p2_{seed}"]
        require(a1["trainable_params"] < a0["trainable_params"],
                f"{seed}: A1 must train fewer parameters than the control")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": "Experiment 3 pre-launch validation on the real student geometry",
        "device": "cpu",
        "optimizer_step_called": False,
        "stage1_init": {"path": str(INIT.relative_to(REPO_ROOT)),
                        "model_safetensors_sha256": init_sha},
        "arms": arms,
        "failures": failures,
        "passed": not failures,
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("PASS: all arms match their intended policy")


if __name__ == "__main__":
    main()
