#!/usr/bin/env python
"""Write the immutable Experiment 6b registration, BEFORE anything trains.

    PYTHONPATH=src python scripts/training/register_e6b.py --authorized-usd X

E6b fills the missing cell of an objective × data-scale matrix:

                      1.60M        2.96M        5.50M
    E1/P1 KD-heavy    evaluated    evaluated    evaluated
    P2    CE-heavy    evaluated    **E6b**      not requested

E6 established that the *E1 lineage* improves 1.60M -> 2.96M and then plateaus.
It established nothing about P2, and the two objectives are tied at 1.60M under
the registered floors, so whether the plateau is a property of the data or of
that particular objective is currently unknown. E6b is the experiment that can
tell those apart, because it is the only cell that makes the interaction
computable.

Registered here, before any number exists: the arms and their config hashes, the
control-reuse rule, the metric hierarchy and floors carried unchanged, the
difference-in-differences definition, the winner-selection rules, the failure
conditions, and the cost backstop.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

PACK = REPO_ROOT / "artifacts/stage3/ladder_uniform_probe"
SESSIONS = REPO_ROOT / "artifacts/stage3/corpus_v2/sessions.jsonl"
INIT = REPO_ROOT / "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
SEEDS = {"sa": 20260726, "sb": 20260801}

# The four cells. `train` marks what E6b pays for; everything else is re-scored
# from retained generations, which E6 proved is a rescore and not a second
# measurement (token-for-token reproducibility across sessions).
ARMS = {
    "P2-2.96M-sa": {"config": "configs/stage3/e6b/e6b_p2_r2960k_sa.json",
                    "run": "e6b_p2_r2960k_sa", "train": True,
                    "objective": "CE-heavy", "rung": 2960000,
                    "retained_three_mode": None},
    "P2-2.96M-sb": {"config": "configs/stage3/e6b/e6b_p2_r2960k_sb.json",
                    "run": "e6b_p2_r2960k_sb", "train": True,
                    "objective": "CE-heavy", "rung": 2960000,
                    "retained_three_mode": None},
    "P2-1.60M-sa": {"config": "configs/stage3/e4/e4_p2_r1600k_sa.json",
                    "run": "e4_p2_r1600k_sa", "train": False,
                    "objective": "CE-heavy", "rung": 1600000,
                    "retained_three_mode": "E4-P2-1600k-sa"},
    "P2-1.60M-sb": {"config": "configs/stage3/e4/e4_p2_r1600k_sb.json",
                    "run": "e4_p2_r1600k_sb", "train": False,
                    "objective": "CE-heavy", "rung": 1600000,
                    "retained_three_mode": "E4-P2-1600k-sb"},
    "E1-2.96M-sa": {"config": "configs/stage3/e1/e1_r2960k_sa_pca.json",
                    "run": "e1_r2960k_sa_pca", "train": False,
                    "objective": "KD-heavy", "rung": 2960000,
                    "retained_three_mode": "E1-2.96M-sa"},
    "E1-2.96M-sb": {"config": "configs/stage3/e1/e1_r2960k_sb_pca.json",
                    "run": "e1_r2960k_sb_pca", "train": False,
                    "objective": "KD-heavy", "rung": 2960000,
                    "retained_three_mode": "E1-2.96M-sb"},
    "E1-1.60M-sa": {"config": "configs/stage3/e1/e1_r1600k_sa_pca.json",
                    "run": "e1_r1600k_sa_pca", "train": False,
                    "objective": "KD-heavy", "rung": 1600000,
                    "retained_three_mode": "E1-1.60M-sa"},
    "E1-1.60M-sb": {"config": "configs/stage3/e1/e1_r1600k_sb_pca.json",
                    "run": "e1_r1600k_sb_pca", "train": False,
                    "objective": "KD-heavy", "rung": 1600000,
                    "retained_three_mode": "E1-1.60M-sb"},
}

FLOORS = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600}

INTERACTION_METRICS = [
    "usable_rollout_rate", "correct_overall", "correct_given_usable",
    "natural_termination_rate", "context_limit_rate", "severe_repetition_rate",
    "empty_output_rate", "answer_parse_failure_rate_numeric",
]

QUESTIONS = [
    "Q1 — does P2 CE-heavy improve when scaled 1.60M -> 2.96M?",
    "Q2 — at the 2.96M rung, is P2 better than E1/P1?",
    "Q3 — does the objective interact with data scale?",
    "Q4 — does P2 preserve or improve correctness while gaining the "
    "rollout-stability improvement E1 showed?",
    "Q5 — should P2-2.96M replace E1-2.96M as the default behavioural anchor "
    "and control recipe?",
]

DECISION_RULES = [
    "R1 — the primary axis is usable_rollout_rate; correct_overall is secondary; "
    "correct_given_usable and the failure taxonomy are diagnostic. Teacher-forced "
    "CE/NLL/top-1 and training loss may NEVER select a winner.",
    "R2 — a difference below its registered floor is a TIE, stated as such. It "
    "may support a compute-efficiency preference, labelled as below-threshold "
    "evidence, but never as a demonstrated behavioural win.",
    "R3 — nothing is claimed from one seed. A pooled difference whose seeds "
    "disagree in direction is reported as disagreement, never hidden in a mean.",
    "R4 — best objective at 2.96M is decided on the primary axis first. If the "
    "objectives tie there, correctness may break the tie; if it also ties, the "
    "objectives are behaviourally indistinguishable at this scale and the "
    "already-canonical arm remains the anchor.",
    "R5 — the interaction is a difference-in-differences: "
    "(P2_2.96 - P2_1.60) - (E1_2.96 - E1_1.60). A nonzero point estimate is NOT "
    "evidence of interaction. It is claimed only if it exceeds the metric's "
    "floor AND is consistent in direction across seeds.",
    "R6 — P2-2.96M replaces E1-2.96M as the anchor only if it wins the primary "
    "axis above the floor and seed-consistently, or ties the primary axis and "
    "wins correctness above its floor and seed-consistently.",
    "R7 — P2-5.50M is NOT authorized and is not initiated by any outcome here. "
    "E6b states whether a preregistered justification exists and stops.",
]

FAILURE_CONDITIONS = [
    "F1 — a trained arm's final config_sha256 does not match this registration.",
    "F2 — an arm did not start from the Stage 1 PCA init 86fbba78…, or resumed "
    "from any trained checkpoint.",
    "F3 — realized optimizer steps != 2916, or realized supervised CE tokens "
    "!= 2,960,507 per exposure, or exposures != 3.",
    "F4 — the evaluation inclusion mask is not d6e24e0b…, or an arm is scored on "
    "a different prompt set.",
    "F5 — loss becomes non-finite, or a checkpoint fails to load.",
    "A poor MODEL result is not a failure condition. Only F1-F5 are, and only "
    "they justify a retry. Retries must be deterministic, documented, and "
    "limited to verified infrastructure faults.",
]

CLAIM_BOUNDARIES = [
    "The 150 evaluation prompts are TRAINING prompts for every arm. The ladder "
    "is nested and every arm trains 3 exposures, so each evaluated prompt is "
    "seen exactly 3 times by every arm at every rung — exposure is identical and "
    "the comparison is fair, but it measures recall-style autonomous behaviour, "
    "not held-out generalization.",
    "Rung and optimizer steps scale together by construction (3 exposures at "
    "every rung), so 'more distinct data' is not separated from 'more steps'.",
    "n=2 seeds per cell. Every spread is a single draw, and the "
    "difference-in-differences compounds four such draws — it is the noisiest "
    "quantity in the experiment and is reported with that caveat attached.",
    "E6b does not evaluate P2-5.50M, so the P2 curve has two points and cannot "
    "distinguish 'plateau' from 'still climbing' at the top rung.",
]


def rung_facts() -> dict:
    """Verified from the loader and the masks, never from the config's claims."""
    import torch
    from aadistill.data.ladder import ladder_blocks
    from aadistill.training.train import stream_block_indices

    out = {}
    prev = None
    for rung in (1600000, 2960000):
        (ids, ce, cm), (v_ids, v_ce, _), stats = ladder_blocks(PACK, rung, 16)
        cfg_steps = 1761 if rung == 1600000 else 2916
        visits = cfg_steps * 2
        idx = stream_block_indices(ids.shape[0], SEEDS["sa"], 0, visits)
        counts = {}
        for i in idx:
            counts[i] = counts.get(i, 0) + 1
        row = {
            "train_blocks": int(ids.shape[0]),
            "supervised_ce_tokens_unique": int(ce.sum()),
            "supervised_ce_tokens_audit": int(stats["train_supervised_tokens"]),
            "packed_tokens": int(ids.numel()),
            "real_tokens": int(cm.sum()),
            "padding_tokens": int(ids.numel() - cm.sum()),
            "packing_efficiency": round(float(cm.sum()) / int(ids.numel()), 4),
            "optimizer_steps": cfg_steps,
            "blocks_per_step": 2,
            "block_visits": visits,
            "exposures": round(visits / int(ids.shape[0]), 4),
            "exposures_uniform_per_block": len(set(counts.values())) == 1,
            "blocks_touched": len(counts),
            "cumulative_ce_tokens": int(ce.sum()) * (visits // int(ids.shape[0])),
            "token_mix": stats["train_token_mix"],
            "val_blocks": 16,
            "val_supervised_tokens": int(v_ce.sum()),
            "val_disjoint_from_all_rungs": bool(
                stats["val_disjoint_from_all_rungs"]),
        }
        if prev is not None:
            row["is_strict_token_prefix_of_this_rung"] = {
                "input_ids": bool(torch.equal(ids[:prev[0].shape[0]], prev[0])),
                "ce_mask": bool(torch.equal(ce[:prev[1].shape[0]], prev[1])),
                "content_mask": bool(torch.equal(cm[:prev[2].shape[0]], prev[2])),
                "prior_rung": 1600000,
                "block_increment": int(ids.shape[0]) - int(prev[0].shape[0]),
            }
        prev = (ids, ce, cm)
        out[f"rung_{rung}"] = row
    return out


def cost_model(rate: float, per_step_s: float, p90_step_s: float) -> dict:
    """Priced from two measured step-time series, never from token counts."""
    steps = 2916
    train_min = steps * per_step_s / 60
    train_min_p90 = steps * p90_step_s / 60
    stage = {"boot": 4.0, "setup": 12.0, "train_sa": train_min,
             "train_sb": train_min, "tokenizer_and_movement": 2.0,
             "evaluate_2_arms": 18.0, "fetch_and_hash": 10.0, "teardown": 2.0}
    expected = sum(stage.values())
    pessimistic = (expected - 2 * train_min) + 2 * train_min_p90 + 12.0
    return {
        "gpu": "NVIDIA L40S", "rate_usd_per_hour": rate,
        "why_this_gpu": (
            "E1-2.96M and P2-1.60M were both TRAINED on an L40S and E6's "
            "evaluations ran on one. Training elsewhere would add a device "
            "difference to the one comparison E6b exists to make."),
        "arms_trained": 2, "optimizer_steps_per_arm": steps,
        "measured_step_seconds": {
            "e4_p2_1600k_sa": 3.625, "e4_p2_1600k_sb": 3.598,
            "e1_2960k_sa_console_mean": 3.464, "e1_2960k_sa_console_p90": 3.820,
            "used_for_expected": per_step_s, "used_for_pessimistic": p90_step_s},
        "no_devbox_upload_required": (
            "Unlike E6, every input (Stage 1 init, probe pack, corpus) is on the "
            "relay, so the ~0.5 MB/s dev-box uplink is not on the critical path. "
            "Only the ~4.8 GB of produced weights move, and in the fast "
            "direction."),
        "stage_minutes": {k: round(v, 1) for k, v in stage.items()},
        "expected_minutes": round(expected, 1),
        "expected_usd": round(expected / 60 * rate, 2),
        "pessimistic_minutes": round(pessimistic, 1),
        "pessimistic_usd": round(pessimistic / 60 * rate, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs/e6b_registration.json")
    ap.add_argument("--authorized-usd", type=float, required=True)
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--per-step-seconds", type=float, default=3.625)
    ap.add_argument("--p90-step-seconds", type=float, default=3.820)
    args = ap.parse_args()

    arms = {}
    for alias, a in ARMS.items():
        cfg = json.loads((REPO_ROOT / a["config"]).read_text())
        arms[alias] = {
            **a, "seed": cfg["seed"], "config_sha256": sha256_json(cfg),
            "loss": cfg["loss"], "student_path": cfg["student_path"],
            "total_steps": cfg["schedule"]["total_steps"],
            "warmup_steps": cfg["schedule"]["warmup_steps"],
        }

    cost = cost_model(args.rate, args.per_step_seconds, args.p90_step_seconds)
    reg = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E6b — P2 CE-heavy scale completion at the 2.96M rung "
                      "(objective x data-scale interaction)",
        "kind": "training + evaluation",
        "questions": QUESTIONS,
        "matrix": {
            "rows": ["E1/P1 KD-heavy (ce 0.25 / kd 1.0)",
                     "P2 CE-heavy (ce 1.0 / kd 0.25)"],
            "columns": ["1.60M", "2.96M"],
            "cell_filled_by_e6b": "P2 CE-heavy x 2.96M",
        },
        "arms": arms,
        "single_variable_proof": json.loads(
            (REPO_ROOT / "configs/stage3/e6b/provenance.json").read_text()),
        "rung_facts_verified_from_loader": rung_facts(),
        "frozen_assets": {
            "pack_blocks_npz_sha256": sha256_file(PACK / "blocks.npz"),
            "pack_audit_sha256": sha256_file(PACK / "audit.jsonl"),
            "sessions_jsonl_sha256": sha256_file(SESSIONS),
            "stage1_init_sha256": sha256_file(INIT / "model.safetensors"),
            "chat_template_sha256": sha256_file(INIT / "chat_template.jinja"),
            "tokenizer_json_sha256": sha256_file(INIT / "tokenizer.json"),
            "inclusion_mask_sha256": EXPECTED_MASK,
            "eval_rung": 860000,
        },
        "evaluation_protocol": "identical to E6 — scripts/evaluation/"
                               "run_three_mode_diagnostic.py, 150 prompts, mask "
                               "d6e24e0b…, greedy, context 8192, unrestricted "
                               "(P18), no degeneration stop, no repair, no "
                               "truncation, no synthetic terminals",
        "control_reuse_rule": (
            "The six control arms are re-scored from retained generations, not "
            "regenerated. E6 established that this harness reproduces token for "
            "token across sessions on the same GPU model, image and evaluator "
            "commit (150/150 on both 1.60M seeds), so reuse is a rescore rather "
            "than a second measurement. Regenerating them for cosmetic "
            "consistency would spend money to reproduce bytes already in hand."),
        "metric_hierarchy": {
            "primary": "usable_rollout_rate",
            "secondary": "correct_overall",
            "diagnostic": ["correct_given_usable", "failure taxonomy",
                           "teacher-forced CE / NLL / top-1", "training loss"],
        },
        "floors": FLOORS,
        "floors_provenance": "carried unchanged from E3/E4/E5/E6; measured seed "
                             "spreads, not preferences, and not renegotiable "
                             "after seeing E6b's numbers",
        "comparisons": {
            "A_same_scale_objective": "P2-2.96M vs E1-2.96M (primary)",
            "B_p2_scale": "P2-2.96M vs P2-1.60M",
            "C_e1_scale_reference": "E1-2.96M vs E1-1.60M (from E6 artifacts)",
            "D_interaction": "(P2_2.96 - P2_1.60) - (E1_2.96 - E1_1.60)",
        },
        "interaction_metrics": INTERACTION_METRICS,
        "decision_rules": DECISION_RULES,
        "failure_conditions": FAILURE_CONDITIONS,
        "claim_boundaries": CLAIM_BOUNDARIES,
        "cost": cost,
        "authorization": {
            "remaining_authorized_usd": args.authorized_usd,
            "required_backstop_usd": cost["pessimistic_usd"],
            "fits": cost["pessimistic_usd"] <= args.authorized_usd,
            "additional_required_usd": max(
                0.0, round(cost["pessimistic_usd"] - args.authorized_usd, 2)),
            "excludes": ["P2-5.50M", "any FineWeb work",
                         "any initialization change", "any loss sweep"],
        },
        "code_state": code_state(str(REPO_ROOT)),
    }
    reg["registration_sha256"] = sha256_json(reg)
    args.out.write_text(json.dumps(reg, indent=2) + "\n")

    r29 = reg["rung_facts_verified_from_loader"]["rung_2960000"]
    print(f"wrote {args.out}")
    print(f"  rung 2.96M      {r29['train_blocks']} blocks, "
          f"{r29['supervised_ce_tokens_unique']:,} unique CE tokens")
    print(f"  exposures       {r29['exposures']} (uniform per block: "
          f"{r29['exposures_uniform_per_block']})")
    print(f"  cumulative CE   {r29['cumulative_ce_tokens']:,}")
    print(f"  nested prefix   {r29['is_strict_token_prefix_of_this_rung']}")
    print(f"  expected        {cost['expected_minutes']:.0f} min = "
          f"${cost['expected_usd']:.2f}")
    print(f"  backstop        {cost['pessimistic_minutes']:.0f} min = "
          f"${cost['pessimistic_usd']:.2f}")
    print(f"  authorized      ${args.authorized_usd:.2f} -> "
          f"fits={reg['authorization']['fits']}")
    if not reg["authorization"]["fits"]:
        print(f"  SHORTFALL       ${reg['authorization']['additional_required_usd']:.2f} "
              "additional authorization required")


if __name__ == "__main__":
    main()
