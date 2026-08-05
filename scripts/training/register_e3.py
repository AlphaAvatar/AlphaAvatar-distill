#!/usr/bin/env python
"""Write the immutable Experiment 3 registration, BEFORE anything trains.

    PYTHONPATH=src python scripts/training/register_e3.py --out logs/e3_registration.json

Pins, by hash, everything the experiment's conclusions will depend on: the six
arm configs, the Stage 1 fork point, the packed rung that trains, the corpus the
evaluation set is drawn from, the held-out corpus, and the 150-example inclusion
mask the control was measured on. Then records the decision rules and the cost
ceiling *in advance*, so the eventual verdict is judged against a criterion that
existed before the numbers did (AGENTS.md 4.5: any Stage 2/3 threshold must be
registered prospectively).

Registration is not a claim that the experiment will work. It is the thing that
makes a null result interpretable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.ladder import ladder_blocks, load_ladder_meta  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

PACK = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
INIT = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
CONFIGS = {
    "A0-P1-sa": "configs/stage3/e1/e1_r0860k_sa_pca.json",
    "A0-P1-sb": "configs/stage3/e1/e1_r0860k_sb_pca.json",
    "A1-frozen-attn-sa": "configs/stage3/e3/e3_a1_frozen_attn_sa.json",
    "A1-frozen-attn-sb": "configs/stage3/e3/e3_a1_frozen_attn_sb.json",
    "A2-lora-attn-sa": "configs/stage3/e3/e3_a2_lora_attn_sa.json",
    "A2-lora-attn-sb": "configs/stage3/e3/e3_a2_lora_attn_sb.json",
}

DECISION_RULES = [
    "R1 — if A1 improves rollout stability without materially reducing "
    "correctness, full-rank attention updates are likely causing harmful drift.",
    "R2 — if A2 outperforms both A1 and A0 across BOTH seeds, constrained "
    "attention adaptation is the preferred policy.",
    "R3 — if A2 improves only teacher-forced CE/top-1 and not autonomous "
    "rollout, do not claim it solved the main problem.",
    "R4 — if A1 and A2 improve FineWeb NLL but not rollout, stop freeze-policy "
    "exploration and recommend student-prefix / on-policy recovery.",
    "R5 — no arm is promoted on one seed alone.",
    "R6 — no arm is promoted merely for terminating earlier if correctness "
    "conditional on a usable rollout falls.",
]

NOISE_FLOORS = {
    "usable_rollout_rate": 0.0800,
    "free_rollout_correctness": 0.0600,
    "teacher_forced_reasoning_top1": 0.0025,
    "teacher_native_holdout_ce": 0.0063,
    "source": "P1 (= P0-real) two-seed spread on this same 150-example set",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs/e3_registration.json")
    args = ap.parse_args()

    meta = load_ladder_meta(PACK)
    _, _, stats = ladder_blocks(PACK, 860000, n_val=16)

    a0_report = json.loads(
        (REPO_ROOT / "artifacts/audit/three_mode/P0-real-sa/report.json").read_text())

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E3 — does restricting attention updates improve autonomous "
                      "generation stability at the 0.86M supervised-token rung?",
        "status": "REGISTERED BEFORE TRAINING",
        "hypothesis": (
            "Full-rank attention updates at this rung may be a source of drift "
            "that degrades autonomous rollout. Freezing them (A1) or confining "
            "them to a rank-8 subspace (A2) should improve usable-rollout rate "
            "without materially reducing correctness. The null — that attention "
            "capacity is needed and restricting it costs correctness, or that "
            "nothing moves beyond seed noise — is an equally reportable outcome."),
        "arms": {
            "A0": "existing 0.86M P1 control; FFN + all norms + attention "
                  "projections, all full-rank. NOT retrained; recorded results "
                  "are reused.",
            "A1": "FFN + all norms full-rank; all four attention projections "
                  "frozen. Everything else identical to A0.",
            "A2": "A1 + LoRA rank 8 / alpha 16 / dropout 0 / bias none on "
                  "self_attn q,k,v,o with base weights frozen; B=0 so the "
                  "initial merged model is exactly the Stage 1 model. LoRA "
                  "tensors share A0's single optimizer group, learning rate, "
                  "schedule and weight-decay semantics — no separate LoRA "
                  "learning rate, no separate parameter group, no rank or "
                  "module sweep.",
        },
        "held_fixed": [
            "Stage 1 PCA initialization (the fork point for every arm)",
            "the nested uniform 0.86M rung and its exact block order",
            "seeds 20260726 (sa) and 20260801 (sb)",
            "0.25·CE + 1.0·KD, temperature 1.0, kd_scope 'all'",
            "AdamW lr 5e-5, wd 0.01, betas (0.9, 0.95), eps 1e-8, clip 1.0",
            "1,023 steps, 51 warmup, cosine to 0.1, 2 blocks/step, block_len 8192",
            "the 150-example evaluation set, its inclusion mask, greedy decoding "
            "and unrestricted generation (P18)",
        ],
        "configs": {
            alias: {"path": path,
                    "config_sha256": sha256_json(json.loads(
                        (REPO_ROOT / path).read_text())),
                    "file_sha256": sha256_file(REPO_ROOT / path)}
            for alias, path in CONFIGS.items()
        },
        "initialization": {
            "path": "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint",
            "model_safetensors_sha256": sha256_file(INIT / "model.safetensors"),
            "config_sha256": sha256_file(INIT / "config.json"),
            "tokenizer_sha256": sha256_file(INIT / "tokenizer.json"),
        },
        "data": {
            "pack": str(PACK.relative_to(REPO_ROOT)),
            "blocks_npz_sha256": sha256_file(PACK / "blocks.npz"),
            "ladder_json_sha256": sha256_file(PACK / "ladder.json"),
            "audit_jsonl_sha256": sha256_file(PACK / "audit.jsonl"),
            "block_len": int(meta["block_len"]),
            "rung": 860000,
            "rung_blocks": stats["train_blocks"],
            "rung_supervised_tokens": stats["train_supervised_tokens"],
            "rung_token_mix": stats["train_token_mix"],
            "val_blocks": stats["val_blocks"],
            "val_block_indices": stats["val_block_indices"],
            "corpus_sessions_sha256": sha256_file(
                REPO_ROOT / "artifacts/stage3/corpus_v2/sessions.jsonl"),
            "holdout_v1_sha256": sha256_file(
                REPO_ROOT / "data/warmup/holdout_v1.jsonl"),
        },
        "evaluation": {
            "harness": "scripts/evaluation/run_three_mode_diagnostic.py",
            "modes": ["free", "oracle", "forced"],
            "n": 150,
            "inclusion_mask_sha256": a0_report["inclusion"]["mask_sha256"],
            "sampling_seed": 20260804,
            "context": 8192,
            "decoding": {"greedy": True, "temperature": 0.0},
            "generation_allowance": "context - prompt, no artificial cap (P18)",
            "degeneration_stop": False,
            "primary_metric": "usable_rollout, reported with all five components",
            "secondary_metrics": ["correct_overall", "correct_given_usable",
                                  "per-task correctness"],
            "diagnostics": ["teacher-forced reasoning top-1",
                            "teacher-native held-out CE",
                            "FineWeb held-out NLL (BF16 and INT8 fake-quant)"],
            "int8": "per-channel symmetric weight fake-quant, scopes 'all' and "
                    "'decoder'; NLL only. INT8 rollout behaviour is NOT measured "
                    "and no claim is made about it.",
        },
        "decision_rules": DECISION_RULES,
        "noise_floors": NOISE_FLOORS,
        "budget": {
            "gpu": "1x NVIDIA L40S secure",
            "authorized_rate_usd_per_hour": 0.99,
            "expected_minutes": 402,
            "expected_cost_usd": 6.63,
            "hard_backstop_minutes": 450,
            "hard_ceiling_usd": 7.43,
            "ledger_remaining_usd": 8.70,
            "note": "Costed from the measured P2-ceheavy session (174.6 min for "
                    "two arms of this exact rung), not by scaling token counts.",
        },
        "what_this_cannot_settle": [
            "n=2 seeds per arm: a spread is one draw per condition, so no "
            "variance claim is made.",
            "LoRA rank 8 is a single point; no rank, module or hyperparameter "
            "sweep is run, so 'LoRA does not help' would mean 'r8 on q/k/v/o "
            "under P1's optimizer settings does not help'.",
            "usable_rollout is blind to correctness by construction, and its "
            "five components are not independent (protocol_valid subsumes two).",
            "A0 is not re-trained, so its free/oracle/forced numbers come from "
            "an earlier session; only its FineWeb NLL is re-measured on the "
            "same machine as A1/A2.",
        ],
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {args.out}")
    for alias, c in payload["configs"].items():
        print(f"  {alias:20s} config_sha256 {c['config_sha256'][:16]}")
    d = payload["data"]
    print(f"  rung {d['rung']}: {d['rung_blocks']} blocks, "
          f"{d['rung_supervised_tokens']:,} supervised tokens")
    print(f"  inclusion mask {payload['evaluation']['inclusion_mask_sha256'][:16]}")


if __name__ == "__main__":
    main()
