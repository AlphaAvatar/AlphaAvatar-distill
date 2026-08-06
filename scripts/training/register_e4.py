#!/usr/bin/env python
"""Write the immutable Experiment 4 registration, BEFORE anything trains.

    PYTHONPATH=src python scripts/training/register_e3.py --out logs/e4_registration.json

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
    "P2-0.86M-sa": "configs/stage3/p2/p2_ceheavy_sa.json",
    "P2-0.86M-sb": "configs/stage3/p2/p2_ceheavy_sb.json",
    "P1-1.60M-sa": "configs/stage3/e1/e1_r1600k_sa_pca.json",
    "P1-1.60M-sb": "configs/stage3/e1/e1_r1600k_sb_pca.json",
    "E4-P2-1.60M-sa": "configs/stage3/e4/e4_p2_r1600k_sa.json",
    "E4-P2-1.60M-sb": "configs/stage3/e4/e4_p2_r1600k_sb.json",
}
REFERENCE_WEIGHTS = {
    "P2-0.86M-sa": ("local", "/home/ecs-user/aad-artifacts/p2_ceheavy/p2_ceheavy_sa",
                    "4aface45a12cd02e"),
    "P2-0.86M-sb": ("local", "/home/ecs-user/aad-artifacts/p2_ceheavy/p2_ceheavy_sb",
                    "9828b1780a5eb4e2"),
    "P1-1.60M-sa": ("relay", "e1_scaling_20260801/e1_r1600k_sa_pca/step_001761/model",
                    "6f77676ab8fde397"),
    "P1-1.60M-sb": ("relay", "e1_scaling_20260801/e1_r1600k_sb_pca/step_001761/model",
                    "e432d57e598d57e1"),
}

DECISION_RULES = [
    "R1 — if P2-1.60M improves mean correct_overall AND usable_rollout over "
    "P2-0.86M with no serious regression on either seed, P2 was data-limited "
    "at 0.86M.",
    "R2 — if teacher-native CE improves but autonomous rollout does not, "
    "increasing teacher-prefix data alone does not resolve the rollout gap.",
    "R3 — if P2-1.60M underperforms P1-1.60M on the primary behavioural "
    "metrics, do NOT promote P2 on the strength of better CE or NLL.",
    "R4 — if P2-1.60M beats both P2-0.86M and P1-1.60M, adopt it as the "
    "checkpoint anchor for the later matched-budget continuation vs "
    "student-prefix-recovery experiment.",
    "R5 — no result is promoted on one seed alone.",
    "R6 — account for the established seed spread and paired per-prompt "
    "outcomes; a mean delta inside the noise floor is not an effect.",
]
DECISION_PRIORITY = [
    "correct_overall",
    "correct_and_naturally_terminated",
    "usable_rollout_rate",
    "repetition and context-limit reduction",
    "correct_given_usable",
]

# Two families have been measured at this rung on this same set and their seed
# spreads disagree. The LARGER is used throughout: with n=2 a spread is a single
# draw, and P2's unusually tight one is recorded as suggestive rather than
# established (§18.7). Taking the smaller would make it too easy to call an
# effect on the very baseline this experiment is compared against.
P1_SEED_SPREAD = {"usable_rollout_rate": 0.0800, "free_rollout_correctness": 0.0600,
                  "teacher_forced_reasoning_top1": 0.0025,
                  "teacher_native_holdout_ce": 0.0063}
P2_SEED_SPREAD = {"usable_rollout_rate": 0.0267, "free_rollout_correctness": 0.0200,
                  "teacher_forced_reasoning_top1": 0.0112,
                  "teacher_native_holdout_ce": 0.0117}
NOISE_FLOORS = {
    **{k: max(P1_SEED_SPREAD[k], P2_SEED_SPREAD[k]) for k in P1_SEED_SPREAD},
    "rule": "max(P1 spread, P2 spread) — the conservative choice",
    "P1_seed_spread": P1_SEED_SPREAD,
    "P2_seed_spread": P2_SEED_SPREAD,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs/e4_registration.json")
    args = ap.parse_args()

    meta = load_ladder_meta(PACK)
    _, _, stats = ladder_blocks(PACK, 1600000, n_val=16)
    _, _, ref_stats = ladder_blocks(PACK, 860000, n_val=16)

    a0_report = json.loads(
        (REPO_ROOT / "artifacts/audit/three_mode/P2-ceheavy-sa/report.json").read_text())
    cost = json.loads((REPO_ROOT / "artifacts/audit/e4_cost_projection.json").read_text())

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E4 — does the P2-CE-heavy recipe improve autonomous "
                      "generation when scaled from 0.86M to the nested 1.60M "
                      "supervised-token rung?",
        "status": "REGISTERED BEFORE TRAINING",
        "hypothesis": (
            "P2-CE-heavy may be data-limited at 0.86M: 1.86x the unique "
            "supervised tokens, same objective and same optimizer semantics, "
            "should improve autonomous rollout and correctness. The null — that "
            "more teacher-prefix data moves teacher-forced CE but not autonomous "
            "behaviour — is an equally reportable outcome and is what E1's "
            "behaviour wave weakly suggests. This experiment establishes the "
            "larger-scale P2 anchor BEFORE any student-prefix recovery work; it "
            "is not itself a recovery experiment."),
        "arms": {
            "reference_P2_0.86M": "existing P2-ceheavy at 0.86M, both seeds. "
                  "NOT retrained; its recorded three-mode results are reused.",
            "reference_P1_1.60M": "existing E1 arms at 1.60M under the P1 "
                  "objective (ce 0.25 / kd 1.0), both seeds. NOT retrained, but "
                  "RE-EVALUATED: their recorded numbers came from the 76-prompt "
                  "behaviour wave with the degeneration stop ACTIVE, which is a "
                  "different measurement from the 150-example unrestricted "
                  "harness and must not be mixed with it.",
            "treatment_P2_1.60M": "new: P2-CE-heavy at the nested 1.60M rung, "
                  "two seeds, forked from the Stage 1 PCA init. Differs from "
                  "P2-0.86M only in the rung and the schedule/cadence E1 "
                  "derives from it (1,174 blocks, 1,600,353 supervised tokens, "
                  "1,761 steps = 3 passes, warmup 88 = 5%).",

        },
        "held_fixed": [
            "Stage 1 PCA initialization (the fork point for every arm)",
            "the nested uniform ladder pack; the 1.60M rung is a strict "
            "superset of the 0.86M rung on actual token ids and CE masks",
            "seeds 20260726 (sa) and 20260801 (sb)",
            "1.0·CE + 0.25·KD, temperature 1.0, kd_scope 'all' (the P2 objective)",
            "AdamW lr 5e-5, wd 0.01, betas (0.9, 0.95), eps 1e-8, clip 1.0",
            "2 blocks/step, block_len 8192, cosine to 0.1, 3 passes over the "
            "rung, warmup 5% of total steps",
            "the 16 pack-tail validation blocks, identical for both rungs",
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
            "rung": 1600000,
            "reference_rung": 860000,
            "reference_rung_blocks": ref_stats["train_blocks"],
            "reference_rung_supervised_tokens": ref_stats["train_supervised_tokens"],
            "strict_superset_of_reference_rung": True,
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
                    "'decoder'; NLL only, dev-box CPU, all six arms on one "
                    "device. INT8 rollout behaviour is NOT measured and no "
                    "claim is made about it.",
        },
        "decision_rules": DECISION_RULES,
        "decision_priority_order": DECISION_PRIORITY,
        "reference_weights": {k: {"where": v[0], "path": v[1], "sha256_prefix": v[2]}
                              for k, v in REFERENCE_WEIGHTS.items()},
        "noise_floors": NOISE_FLOORS,
        "budget": {**cost, "detail": {
            "gpu": "1x NVIDIA L40S secure",
            "authorized_rate_usd_per_hour": 0.99,
            "expected_minutes": 373,
            "expected_cost_usd": 6.16,
            "hard_backstop_minutes": 450,
            "hard_ceiling_usd": 7.43,
            "ledger_remaining_usd": 8.70,
            "note": "Costed from the measured P2-ceheavy session (174.6 min for "
                    "two arms of this exact rung), not by scaling token counts. "
                    "Held-out NLL was moved off the pod to the dev-box CPU "
                    "(25 s/model, reproduces the GPU value to 0.02%), removing "
                    "~24 min of paid time.",
        }},
        "what_this_cannot_settle": [
            "n=2 seeds per arm: a spread is one draw per condition, so no "
            "variance claim is made from it.",
            "The 150 evaluation prompts are drawn from the 0.86M rung and are "
            "therefore TRAINING prompts for every arm compared. Both rungs "
            "contain all 150, so no arm gains or loses exposure — but these are "
            "recall-style measurements, not held-out generalization.",
            "P2-0.86M's rollout numbers come from the 2026-08-05 session on "
            "different hardware; only the new arms and the P1-1.60M re-runs are "
            "measured in this session.",
            "Two rungs is not a scaling curve. A null at 1.60M does not "
            "establish that 2.96M or 5.50M would also be null.",
            "usable_rollout is blind to correctness by construction and its "
            "five components are not independent.",
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
