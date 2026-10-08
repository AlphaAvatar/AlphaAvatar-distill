#!/usr/bin/env python
"""Prove the CE/KD normalization is unaffected by prefix length, before E5 spends.

    PYTHONPATH=src python scripts/training/diagnose_e5_normalization.py \
        --out artifacts/audit/e5_normalization.json

E5's two arms carry different amounts of *context*: R's student prefixes are
expected to be longer than C's teacher prefixes at the same relative cut depth.
With `kd_scope=all`, KD is computed over every non-padding position, so the two
arms will have different KD-mask token counts even when their CE continuation
counts are matched.

The question this settles is whether that changes the **effective objective**. It
would if KD were summed without normalization, or normalized by block length, or
normalized by the CE denominator — in any of those cases a longer prefix would
silently reweight KD against CE and the comparison would be confounded.

Nothing here reimplements the loss. It drives `Trainer.step_once` on real packed
blocks and reads the values the trainer itself logs and returns, because a
diagnostic that computes its own normalizer proves nothing about the trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from aadistill.data.ladder import ladder_blocks  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.training.train import Trainer, prediction_mask  # noqa: E402

PACK = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
INIT = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"


def tiny_cfg(tmp: Path, *, ce: float, kd: float) -> dict:
    """The production loss/optimizer shape at a size that runs on CPU."""
    return {
        "stage": "stage3_recovery", "run_name": "e5_norm_probe",
        "student_path": "unused", "teacher": None, "data_dir": "unused",
        "groups": None, "block_len": 256, "dtype": "float32", "device": "cpu",
        "seed": 11, "trainable_patterns": "all",
        "loss": {"ce_weight": ce, "kd_weight": kd,
                 "kd_temperature": 1.0, "kd_scope": "all"},
        "optim": {"lr": 0.0, "weight_decay": 0.0, "betas": [0.9, 0.95],
                  "eps": 1e-8, "grad_clip": 1.0},
        "schedule": {"total_steps": 1, "warmup_steps": 0, "min_lr_frac": 1.0},
        "batch": {"blocks_per_step": 2, "micro_blocks": 1},
        "checkpoint": {"save_every": 0, "keep_last": 1},
        "intervals": {"log_every": 1, "eval_every": 0, "eval_blocks": 0},
        "out_dir": str(tmp),
    }


def make_arm(ids: torch.Tensor, ce_mask: torch.Tensor, content: torch.Tensor,
             prefix_tokens: int):
    """Move `prefix_tokens` of supervision into context, keeping tokens identical.

    This is exactly what the E5 split does: the CE mask shrinks from the left
    while the token stream and the content mask are untouched. A longer prefix
    therefore means fewer CE tokens over the *same* number of KD tokens.
    """
    new_mask = ce_mask.clone()
    for b in range(new_mask.shape[0]):
        sup = new_mask[b].nonzero().flatten()
        if sup.numel() == 0:
            continue
        cut = sup[:min(prefix_tokens, max(0, sup.numel() - 8))]
        new_mask[b, cut] = False
    return ids, new_mask, content


def probe(name: str, blocks, cfg, tmp: Path) -> dict:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    small = Qwen3Config(vocab_size=2048, hidden_size=64, num_hidden_layers=2,
                        intermediate_size=128, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16,
                        tie_word_embeddings=True, max_position_embeddings=512)
    student = Qwen3ForCausalLM(small).float()
    teacher = Qwen3ForCausalLM(small).float()
    trainer = Trainer(cfg, student, blocks, blocks, teacher=teacher, device="cpu")

    ids, ce_mask, content = blocks
    ce_den = int(ce_mask[:, 1:].sum())
    kd_den = int(prediction_mask(ce_mask, "all", content).sum())
    nonpad = int(content.sum())

    m = trainer.step_once()
    ce_num = m["ce"] * ce_den if m["ce"] is not None else 0.0
    kd_num = m["kd"] * kd_den if m["kd"] is not None else 0.0
    w = cfg["loss"]
    return {
        "arm": name,
        "ce_mask_tokens": ce_den,
        "kd_mask_tokens": kd_den,
        "nonpadding_tokens": nonpad,
        "kd_per_ce_token": round(kd_den / max(1, ce_den), 4),
        "ce_numerator": round(ce_num, 4),
        "ce_denominator": m["ce_targets"],
        "ce_mean": m["ce"],
        "kd_numerator": round(kd_num, 4),
        "kd_denominator": m["kd_positions"],
        "kd_mean": m["kd"],
        "weighted_ce_contribution": round(w["ce_weight"] * m["ce"], 6),
        "weighted_kd_contribution": round(w["kd_weight"] * m["kd"], 6),
        "reported_total_loss": m["loss"],
        # The trainer's own denominators must equal the masks, or it is
        # normalizing by something else.
        "ce_denominator_equals_ce_mask": m["ce_targets"] == ce_den,
        "kd_denominator_equals_kd_mask": m["kd_positions"] == kd_den,
        "kd_denominator_is_not_ce_denominator": m["kd_positions"] != m["ce_targets"],
        "kd_denominator_is_not_block_length": (
            m["kd_positions"] != ids.shape[0] * ids.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts/audit/e5_normalization.json")
    ap.add_argument("--short-prefix", type=int, default=8)
    ap.add_argument("--long-prefix", type=int, default=64)
    args = ap.parse_args()

    import tempfile

    (train, _, _) = ladder_blocks(PACK, 860000, n_val=16)
    ids = train[0][:2, :256].clone()
    ce_mask = train[1][:2, :256].clone().bool()
    content = train[2][:2, :256].clone().bool()
    if not ce_mask.any():
        ce_mask[:, 100:200] = True
    ids = torch.remainder(ids, 2048)

    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, prefix in (("C_like_short_prefix", args.short_prefix),
                             ("R_like_long_prefix", args.long_prefix)):
            blocks = make_arm(ids, ce_mask, content, prefix)
            results.append(probe(name, blocks, tiny_cfg(tmp, ce=1.0, kd=0.25), tmp))

    c, r = results
    failures = []

    def require(cond, msg):
        if not cond:
            failures.append(msg)

    for res in results:
        require(res["ce_denominator_equals_ce_mask"],
                f"{res['arm']}: CE is not normalized over CE-mask tokens")
        require(res["kd_denominator_equals_kd_mask"],
                f"{res['arm']}: KD is not normalized over KD-mask tokens")
        require(res["kd_denominator_is_not_ce_denominator"],
                f"{res['arm']}: KD appears to use the CE denominator")
        require(res["kd_denominator_is_not_block_length"],
                f"{res['arm']}: KD appears to be normalized by block length")
        require(res["kd_mask_tokens"] == res["nonpadding_tokens"] - ids.shape[0],
                f"{res['arm']}: KD mask is not exactly the non-padding "
                "prediction positions")

    # The load-bearing claim: a longer prefix changes the CE denominator but not
    # the KD denominator, and both terms stay per-token means, so the configured
    # coefficients remain the effective coefficients in both arms.
    require(r["ce_mask_tokens"] < c["ce_mask_tokens"],
            "the long-prefix arm did not actually have fewer CE tokens")
    require(r["kd_mask_tokens"] == c["kd_mask_tokens"],
            "KD mask changed with prefix length; the arms are not comparable")
    require(abs(r["kd_mean"] - c["kd_mean"]) < 1e-6,
            "KD mean moved with prefix length despite an identical KD mask")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": ("prove prefix length does not alter the effective CE/KD "
                    "weighting through the real training code"),
        "loss_config": {"ce_weight": 1.0, "kd_weight": 0.25,
                        "kd_temperature": 1.0, "kd_scope": "all"},
        "arms": results,
        "conclusion": {
            "ce_normalized_over": "CE-mask continuation tokens only",
            "kd_normalized_over": "KD-mask (non-padding) prediction positions only",
            "padding_excluded_from_both": True,
            "kd_summed_without_normalization": False,
            "kd_divided_by_block_length": False,
            "kd_divided_by_ce_denominator": False,
            "effective_coefficients_unchanged_by_prefix_length": not failures,
            "note": ("both terms are per-token MEANS, so ce_weight and kd_weight "
                     "are the effective coefficients regardless of how many "
                     "tokens each mask covers. A longer prefix shifts KD's "
                     "*composition* toward prefix states — which is the E5 "
                     "treatment — without reweighting KD against CE."),
        },
        "failures": failures,
        "passed": not failures,
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    hdr = f"{'arm':22s}{'CE tok':>9s}{'KD tok':>9s}{'CE mean':>10s}{'KD mean':>10s}{'w*CE':>10s}{'w*KD':>10s}"
    print(hdr)
    for res in results:
        print(f"{res['arm']:22s}{res['ce_mask_tokens']:9d}{res['kd_mask_tokens']:9d}"
              f"{res['ce_mean']:10.4f}{res['kd_mean']:10.4f}"
              f"{res['weighted_ce_contribution']:10.4f}{res['weighted_kd_contribution']:10.4f}")
    print(f"\nKD tokens per CE token: C {c['kd_per_ce_token']}  R {r['kd_per_ce_token']}")
    print(f"wrote {args.out}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("PASS: prefix length does not alter the effective CE/KD weighting")


if __name__ == "__main__":
    main()
