#!/usr/bin/env python
"""Write the immutable Experiment 6 registration, BEFORE any GPU is created.

    PYTHONPATH=src python scripts/evaluation/register_e6.py --out logs/e6_registration.json

E6 is **evaluation-only**. Nothing trains, nothing is merged, quantized or
overwritten. It places the existing high-rung Experiment 1 PCA checkpoints onto
the current frozen 150-prompt unrestricted protocol so the original scale curve
can be read on the same instrument as the current P2-1.60M anchor.

The scientific question it answers is whether the E1 PCA lineage improved,
plateaued or regressed as the rung grew 1.60M -> 2.96M -> 5.50M. That question
has only ever been answered on the retired 76-prompt behaviour wave with the
degeneration stop active, which is a different prompt population, a different
harness and a stop policy that changes the termination and context-limit
components outright (`EXPERIMENTS.md` §19.11).

Two things are registered here that the eventual verdict must be judged against:

* **the reuse rule** — which arms are regenerated and which are re-scored from
  retained raw generations, decided before any number is seen;
* **the interpretation floors** — carried unchanged from the existing registry
  (usable 0.0800, correct_overall 0.0600). No new composite score is defined,
  and no threshold may be invented after the data lands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
RELAY = "AlphaAvatar/aadistill-artifacts"
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"

# The commit whose evaluator produced every retained artifact E6 reuses. The
# reuse rule below is only sound because the harness and every scorer it calls
# are byte-identical between this commit and HEAD; `verify_evaluator_unchanged`
# proves it rather than asserting it.
E4_COMMIT = "1dafec29b1637d3e1412be7fcf453640c4cd97d9"
EVALUATOR_PATHS = (
    "scripts/evaluation/run_three_mode_diagnostic.py",
    "scripts/evaluation/diagnose_training_recall.py",
    "src/aadistill/evaluation",
    "src/aadistill/data/verify.py",
)

# alias -> identity. `source` says where the weights come from; `generate` says
# whether this arm's raw generations are produced in the E6 session or reused
# from a retained artifact. Both are fixed here, before anything runs.
ARMS = {
    "E1-1.60M-sa": {
        "run": "e1_r1600k_sa_pca", "seed": 20260726, "rung": 1600000,
        "step": "step_001761", "config": "configs/stage3/e1/e1_r1600k_sa_pca.json",
        "weights_sha256": "6f77676ab8fde397ef7af75fda3e62171b5c84f315c439a1abb49917e46f6697",
        "source": ("relay", "e1_scaling_20260801/e1_r1600k_sa_pca/step_001761/model"),
        "generate": True, "retained_three_mode": "P1-1600k-sa",
        "lineage": "E1 PCA scale curve",
    },
    "E1-1.60M-sb": {
        "run": "e1_r1600k_sb_pca", "seed": 20260801, "rung": 1600000,
        "step": "step_001761", "config": "configs/stage3/e1/e1_r1600k_sb_pca.json",
        "weights_sha256": "e432d57e598d57e1633392e92955c8185faab57909f75f44bc1c349db6ccf39e",
        "source": ("relay", "e1_scaling_20260801/e1_r1600k_sb_pca/step_001761/model"),
        "generate": True, "retained_three_mode": "P1-1600k-sb",
        "lineage": "E1 PCA scale curve",
    },
    "E1-2.96M-sa": {
        "run": "e1_r2960k_sa_pca", "seed": 20260726, "rung": 2960000,
        "step": "step_002916", "config": "configs/stage3/e1/e1_r2960k_sa_pca.json",
        "weights_sha256": "3f08482c2c8e7372fc87fd2496f50c1c618f61feaec021d73c0cc646413b80c3",
        "source": ("relay", "e1_scaling_20260801/e1_r2960k_sa_pca/step_002916/model"),
        "generate": True, "retained_three_mode": None,
        "lineage": "E1 PCA scale curve",
    },
    "E1-2.96M-sb": {
        "run": "e1_r2960k_sb_pca", "seed": 20260801, "rung": 2960000,
        "step": "step_002916", "config": "configs/stage3/e1/e1_r2960k_sb_pca.json",
        "weights_sha256": "b658fe392ab0db492c0df73c7008fc79ed89c0f526ad10edb10404c3bdb6f8c5",
        "source": ("devbox", "artifacts/stage3/rescued/e1_r2960k_sb_pca"),
        "generate": True, "retained_three_mode": None,
        "lineage": "E1 PCA scale curve",
    },
    "E1-5.50M-sa": {
        "run": "e1_r5500k_sa_pca", "seed": 20260726, "rung": 5500000,
        "step": "step_004412", "config": "configs/stage3/e1/e1_r5500k_sa_pca.json",
        "weights_sha256": "3069b329df3edfbd5edc0356516cd06ee7f02fe59663c19df7b30ef6acd8e397",
        "source": ("relay", "e1_scaling_20260801/e1_r5500k_sa_pca/step_004412/model"),
        "generate": True, "retained_three_mode": None,
        "lineage": "E1 PCA scale curve",
    },
    "E1-5.50M-sb": {
        "run": "e1_r5500k_sb_pca", "seed": 20260801, "rung": 5500000,
        "step": "step_004412", "config": "configs/stage3/e1/e1_r5500k_sb_pca.json",
        "weights_sha256": "bcb916cb3e544505770cddf021c680b0af6ded3ec7b5cfafe37eea5bb1541742",
        "source": ("devbox", "artifacts/stage3/rescued/e1_r5500k_sb_pca"),
        "generate": True, "retained_three_mode": None,
        "lineage": "E1 PCA scale curve",
    },
    "P2-1.60M-sa": {
        "run": "e4_p2_r1600k_sa", "seed": 20260726, "rung": 1600000,
        "step": "step_001761", "config": "configs/stage3/e4/e4_p2_r1600k_sa.json",
        "weights_sha256": "7ee1d9355b97563f095c15850dff51b7693d65e29d544a10c1575b63fdc78dce",
        "source": ("devbox", "/home/ecs-user/aad-artifacts/e4/e4_p2_r1600k_sa/model"),
        "generate": False, "retained_three_mode": "E4-P2-1600k-sa",
        "lineage": "external anchor (P2 CE-heavy objective)",
    },
    "P2-1.60M-sb": {
        "run": "e4_p2_r1600k_sb", "seed": 20260801, "rung": 1600000,
        "step": "step_001761", "config": "configs/stage3/e4/e4_p2_r1600k_sb.json",
        "weights_sha256": "98e8c9811414e982150bac934ae08cd17bb0772b797eaedae8efb2157721708c",
        "source": ("devbox", "/home/ecs-user/aad-artifacts/e4/e4_p2_r1600k_sb/model"),
        "generate": False, "retained_three_mode": "E4-P2-1600k-sb",
        "lineage": "external anchor (P2 CE-heavy objective)",
    },
}

SCALE_CURVE = ["E1-1.60M", "E1-2.96M", "E1-5.50M"]
ANCHOR = "P2-1.60M"

REUSE_RULE = {
    "regenerated_in_the_e6_session": sorted(a for a, v in ARMS.items() if v["generate"]),
    "reused_from_retained_generations": sorted(
        a for a, v in ARMS.items() if not v["generate"]),
    "why": (
        "The whole E1 PCA scale curve is regenerated inside one session so the "
        "primary comparison -- 1.60M vs 2.96M vs 5.50M -- is within-session on "
        "one GPU, satisfying the standing rule that checkpoints in a comparison "
        "are scored together (decisions.md 2026-07-27). The P2-1.60M anchor is "
        "NOT regenerated: its raw generations are retained complete, and its "
        "harness, scorers and engine version are byte-identical to the ones E6 "
        "runs, so re-scoring them is a rescore and not a new measurement."),
    "cross_session_replicate": (
        "E1-1.60M was already measured in the E4 session, in the same session as "
        "the P2-1.60M anchor. E6 therefore obtains a direct measurement of the "
        "session-to-session difference on identical weights and identical frozen "
        "assets -- an instrument check that costs nothing extra."),
    "primary_source_for_each_comparison": {
        "E1 scale curve (2.96M vs 1.60M, 5.50M vs 1.60M, 5.50M vs 2.96M)":
            "the E6-session generations for all three rungs",
        "each E1 rung vs the P2-1.60M anchor":
            "E4-session generations for BOTH P2-1.60M and E1-1.60M (same session, "
            "so the 1.60M contrast is clean), and E6-session generations for the "
            "2.96M/5.50M rungs against the E4-session anchor (cross-session; the "
            "measured E1-1.60M session difference bounds that contamination)",
    },
    "registered_before_data": True,
}

INTERPRETATION = {
    "hierarchy": (
        "Primary = autonomous rollout behaviour (usable_rollout and every "
        "component rate). Secondary = correctness, per-task correctness, "
        "correct_given_usable. Diagnostic only = teacher-native CE, FineWeb NLL, "
        "teacher-forced top-1, training loss. Do not invert; do not combine onto "
        "one scale; do not select a winner on a diagnostic."),
    "floors": {
        "usable_rollout_rate": 0.0800,
        "correct_overall": 0.0600,
    },
    "floors_provenance": (
        "Carried unchanged from the E3/E4/E5 registry. 0.0800 is P0-real's own "
        "measured seed spread on this battery; 0.0600 is the measured seed "
        "spread of free-rollout correctness on the same 150 examples. Neither is "
        "recomputed or renegotiated for E6."),
    "no_new_composite": (
        "No new composite score may be defined after the data is seen. "
        "usable_rollout is reported with all five component rates, and the "
        "components are known not to be independent (protocol_valid implies "
        "non_empty and natural_termination by construction), so the conjunction "
        "must never be presented as five agreeing checks."),
    "seed_rule": (
        "Nothing is claimed on one seed. A mean movement whose direction "
        "disagrees across seeds is reported as disagreement, never hidden "
        "inside the mean."),
    "paired_statistics_scope": (
        "The paired bootstrap resamples PROMPTS at fixed checkpoints. An "
        "interval excluding zero is not by itself evidence the rung moved -- "
        "the seed floor governs that, and both are reported side by side."),
}

QUESTIONS = [
    "Q1 -- Under the current 150-prompt unrestricted protocol, does the E1 PCA "
    "lineage improve, plateau or regress from 1.60M to 2.96M to 5.50M?",
    "Q2 -- Which rung is the best checkpoint within the E1 PCA scale curve?",
    "Q3 -- Which is the best checkpoint among all evaluated existing lineages, "
    "including the P2-1.60M anchor?",
    "Q4 -- Does higher exposure change stability, correctness, or only the "
    "teacher-forced/CE diagnostics?",
    "Q5 -- Does the standing conclusion that 1.60M is the practical high point "
    "survive normalization onto this harness?",
]

CLAIM_BOUNDARIES = [
    "The 150 evaluation prompts are TRAINING prompts for every arm. The ladder "
    "is nested, so the 0.86M subset the battery is drawn from is contained in "
    "the 1.60M, 2.96M and 5.50M rungs alike, and every arm trains 3 epochs -- so "
    "each evaluated prompt is seen exactly 3 times by every arm. Exposure to the "
    "evaluated prompts is therefore IDENTICAL across rungs and the comparison is "
    "fair, but it measures recall-style autonomous behaviour, not held-out "
    "generalization.",
    "Rung and optimizer steps scale together by construction (3 epochs at every "
    "rung), so E6 cannot separate 'more distinct data' from 'more optimizer "
    "steps'. It measures the effect of the RUNG as Experiment 1 defined it.",
    "P2-1.60M differs from E1-1.60M in the objective weights (CE 1.0 / KD 0.25 "
    "versus CE 0.25 / KD 1.0), not only in lineage. It is an external anchor, "
    "not a point on the E1 curve.",
    "n=2 seeds per rung. Every spread is a single draw.",
    "The 2.96M and 5.50M arms are measured in the E6 session while the P2-1.60M "
    "anchor is measured in the E4 session. The E1-1.60M arm, measured in both, "
    "bounds that cross-session difference; it does not eliminate it.",
    "E6 does not evaluate the random-init arms, any official pretrained "
    "reference, or any new initialization. It adds no arms beyond the eight "
    "registered here.",
]


def verify_evaluator_unchanged() -> dict:
    """Prove the evaluator is byte-identical to the one that made reused artifacts."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--stat", E4_COMMIT, "HEAD", "--",
             *EVALUATOR_PATHS],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"verified": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"verified": out == "", "diff_stat": out, "against_commit": E4_COMMIT,
            "paths": list(EVALUATOR_PATHS)}


def frozen_assets() -> dict:
    """Hash every asset the protocol is frozen against, and rebuild the mask."""
    from diagnose_training_recall import rung_session_ids, stratified_sample

    want = set(rung_session_ids(PACK, 860000))
    rung = [json.loads(l) for l in SESSIONS.open()
            if l.strip() and json.loads(l)["id"] in want]
    incl = [s for s in rung if s.get("correct") is True]
    picked = stratified_sample(incl, 150, 20260804)
    ids = sorted(s["id"] for s in picked)
    mask = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    by_task: dict[str, int] = {}
    for s in picked:
        by_task[s["data_type"]] = by_task.get(s["data_type"], 0) + 1
    return {
        "pack_blocks_npz_sha256": sha256_file(PACK / "blocks.npz"),
        "pack_ladder_json_sha256": sha256_file(PACK / "ladder.json"),
        "sessions_jsonl_sha256": sha256_file(SESSIONS),
        "stage1_init_sha256": sha256_file(INIT / "model.safetensors"),
        "chat_template_sha256": sha256_file(INIT / "chat_template.jinja"),
        "tokenizer_json_sha256": sha256_file(INIT / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(INIT / "tokenizer_config.json"),
        "generation_config_sha256": sha256_file(INIT / "generation_config.json"),
        "eval_rung": 860000,
        "rung_sessions": len(rung),
        "verified_correct": len(incl),
        "excluded_unverified": len(rung) - len(incl),
        "sampled": len(picked),
        "sample_seed": 20260804,
        "inclusion_mask_sha256": mask,
        "inclusion_mask_matches_binding": mask == EXPECTED_MASK,
        "inclusion_mask_by_task": dict(sorted(by_task.items())),
        "inclusion_mask_is_model_independent": (
            "The mask is a function of (pack, rung, sessions, n, seed) only. No "
            "model output enters it, and no prompt is ever excluded because of "
            "what a model produced."),
        "system_prompt": (
            "Carried per session inside sessions.jsonl and rendered by the "
            "official chat template; pinned by sessions_jsonl_sha256. It is a "
            "fixed project requirement, never an experimental variable."),
    }


def protocol() -> dict:
    return {
        "harness": "scripts/evaluation/run_three_mode_diagnostic.py",
        "modes": ["free", "oracle", "forced"],
        "primary_mode": "free",
        "n": 150,
        "context": 8192,
        "context_source": "trained_block_len",
        "generation_allowance": "context - len(prompt_ids), per sample (P18)",
        "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "greedy": True},
        "stop_token_ids": ["<|im_end|>", "<|endoftext|>"],
        "degeneration_stop": False,
        "degeneration_recorded_not_enforced": True,
        "output_repair": False,
        "post_generation_truncation": False,
        "synthetic_terminal_tokens": False,
        "engine": "vLLM 0.26.0, bfloat16, gpu_memory_utilization 0.9",
        "forced_mode_dtype": "float32 (HF, not vLLM)",
        "answer_extraction": "src/aadistill/evaluation/strict_answer.py",
        "correctness_scorer": "strict_answer.extract_final_answer + normalize_number "
                              "/ capability.normalize_answer, per task type",
        "usable_rollout_classifier": "src/aadistill/evaluation/usable_rollout.py",
        "rescore_policy": (
            "Every arm is re-scored from its raw generations with the current "
            "scorer. No historical correct/usable/termination field is copied "
            "into an E6 result."),
        "retry_policy": (
            "Only a proven infrastructure failure is retried, and only by "
            "re-running the identical deterministic command for that arm. A "
            "model failure -- empty output, non-termination, degeneration, "
            "context-limit -- is a result and is never retried."),
    }


def cost_model(gpu: str, rate: float) -> dict:
    """Priced from measured E4 timings, not from scaled token counts."""
    eval_minutes = 9.0          # E4 measured 8.67 / 7.42 / 6.53 / 8.31 min per arm
    arms = sum(1 for v in ARMS.values() if v["generate"])
    stage = {"boot": 4.0, "setup": 9.0, "checkpoints": 12.0,
             "evaluation": eval_minutes * arms, "retrieve_verify": 6.0,
             "teardown": 2.0}
    expected = sum(stage.values())
    pessimistic = expected * 1.35 + 12.0     # one cold-host redraw absorbed
    return {
        "gpu": gpu, "rate_usd_per_hour": rate,
        "arms_generated": arms,
        "prompts_per_arm": 150,
        "modes_per_arm": ["free", "oracle", "forced"],
        "generated_candidates": arms * 150 * 2,   # free + oracle are generative
        "max_possible_generated_tokens": arms * 150 * 2 * 8192,
        "stage_minutes": stage,
        "expected_minutes": round(expected, 1),
        "expected_usd": round(expected / 60 * rate, 2),
        "pessimistic_minutes": round(pessimistic, 1),
        "pessimistic_usd": round(pessimistic / 60 * rate, 2),
        "basis": (
            "Per-arm evaluation time is the measured E4 three-mode wall clock "
            "(8.67, 7.42, 6.53, 8.31 minutes for four arms on an L40S), rounded "
            "up to 9.0. Setup and boot are the warm-host E4 measurements; the "
            "pessimistic column carries one cold-host redraw, which is the "
            "dominant measured risk (setup has varied 5 -> 8.5 -> 150+ minutes "
            "on identical scripts and images)."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs/e6_registration.json")
    ap.add_argument("--gpu", default="L40S")
    ap.add_argument("--rate", type=float, default=0.99)
    ap.add_argument("--authorized-usd", type=float, required=True,
                    help="remaining authorized project budget, in USD")
    args = ap.parse_args()

    cost = cost_model(args.gpu, args.rate)
    assets = frozen_assets()
    evaluator = verify_evaluator_unchanged()

    reg = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E6 -- high-rung checkpoint normalization onto the frozen "
                      "150-prompt unrestricted protocol",
        "kind": "evaluation-only",
        "trains_anything": False,
        "modifies_any_checkpoint": False,
        "questions": QUESTIONS,
        "arms": ARMS,
        "scale_curve": SCALE_CURVE,
        "anchor": ANCHOR,
        "reuse_rule": REUSE_RULE,
        "protocol": protocol(),
        "frozen_assets": assets,
        "evaluator_unchanged_since_reused_artifacts": evaluator,
        "interpretation": INTERPRETATION,
        "claim_boundaries": CLAIM_BOUNDARIES,
        "cost": cost,
        "authorization": {
            "remaining_authorized_usd": args.authorized_usd,
            "hard_backstop_usd": min(args.authorized_usd, cost["pessimistic_usd"]),
            "covers": ["one evaluation pod", "six regenerated arms",
                       "artifact retrieval, verification and teardown"],
            "excludes": ["any training", "any new initialization",
                         "any FineWeb experiment", "any additional arm",
                         "any checkpoint upload or overwrite"],
            "fits": cost["pessimistic_usd"] <= args.authorized_usd,
        },
        "code_state": code_state(str(REPO_ROOT)),
    }
    reg["registration_sha256"] = sha256_json(reg)
    args.out.write_text(json.dumps(reg, indent=2) + "\n")

    print(f"wrote {args.out}")
    print(f"  mask                 {assets['inclusion_mask_sha256'][:16]}… "
          f"matches={assets['inclusion_mask_matches_binding']}")
    print(f"  evaluator unchanged  {evaluator['verified']}")
    print(f"  arms generated       {cost['arms_generated']} of {len(ARMS)}")
    print(f"  expected             {cost['expected_minutes']:.0f} min = "
          f"${cost['expected_usd']:.2f}")
    print(f"  pessimistic          {cost['pessimistic_minutes']:.0f} min = "
          f"${cost['pessimistic_usd']:.2f}")
    print(f"  authorized           ${args.authorized_usd:.2f} -> "
          f"fits={reg['authorization']['fits']}")
    if not assets["inclusion_mask_matches_binding"]:
        raise SystemExit("inclusion mask does not match the binding value; stop")
    if not evaluator["verified"]:
        raise SystemExit("evaluator changed since the reused artifacts; stop")
    if not reg["authorization"]["fits"]:
        raise SystemExit("pessimistic cost exceeds the authorization; stop and report")


if __name__ == "__main__":
    main()
