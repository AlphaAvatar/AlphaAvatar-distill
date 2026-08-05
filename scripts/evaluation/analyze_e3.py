#!/usr/bin/env python
"""Experiment 3 comparison: A0 vs A1 vs A2, paired at the prompt level.

    PYTHONPATH=src python scripts/evaluation/analyze_e3.py \
        --out artifacts/audit/e3_comparison.json

    A0  0.86M P1 control — FFN + norms + attention projections, all full-rank
    A1  attention projections FROZEN
    A2  attention projections adapted by LoRA r8, base frozen

Nothing is generated here: retained artifacts are re-read, correctness is
re-scored with the corrected scorer (never taken from the stored `correct`
field), and the decision rules registered before the run are applied
mechanically to whatever the numbers turn out to be.

The evaluation hierarchy is the one Stage 2/3 declares and is not inverted:
`usable_rollout` and its five components are primary, correctness is secondary,
and teacher-forced top-1, teacher-native CE and FineWeb NLL are diagnostics that
never rank an arm on their own.
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
from reevaluate_stage23 import three_mode_arm  # noqa: E402

AUDIT = REPO_ROOT / "artifacts/audit"

# alias -> (three-mode directory, training-log path, movement report)
ARMS = {
    "A0-P1-sa": ("P0-real-sa",
                 "artifacts/stage3/rescued/_relay/e1_r0860k_sa_pca/train_log.jsonl",
                 "A0-P1-sa"),
    "A0-P1-sb": ("P0-real-sb",
                 "artifacts/stage3/rescued/_relay/e1_r0860k_sb_pca/train_log.jsonl",
                 "A0-P1-sb"),
    "A1-frozen-attn-sa": ("A1-frozen-attn-sa",
                          "artifacts/stage3/e3_a1_frozen_attn_sa/train_log.jsonl",
                          "A1-frozen-attn-sa"),
    "A1-frozen-attn-sb": ("A1-frozen-attn-sb",
                          "artifacts/stage3/e3_a1_frozen_attn_sb/train_log.jsonl",
                          "A1-frozen-attn-sb"),
    "A2-lora-attn-sa": ("A2-lora-attn-sa",
                        "artifacts/stage3/e3_a2_lora_attn_sa/train_log.jsonl",
                        "A2-lora-attn-sa"),
    "A2-lora-attn-sb": ("A2-lora-attn-sb",
                        "artifacts/stage3/e3_a2_lora_attn_sb/train_log.jsonl",
                        "A2-lora-attn-sb"),
}
FAMILIES = {"A0": ["A0-P1-sa", "A0-P1-sb"],
            "A1": ["A1-frozen-attn-sa", "A1-frozen-attn-sb"],
            "A2": ["A2-lora-attn-sa", "A2-lora-attn-sb"]}
SEEDS = ("sa", "sb")

# Measured noise floors on this 150-example set, from the P1 arms. Any claimed
# effect is read against these, not against zero.
P1_SEED_SPREAD = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600,
                  "teacher_forced_reasoning_top1": 0.0025}


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open() if line.strip()]


def training_metrics(path: Path) -> dict:
    """Teacher-native held-out CE (the 16 pack-tail blocks) and run identity."""
    if not path.is_file():
        return {"not_evaluable": f"missing {path.name}"}
    rows = jsonl(path)
    start = next((r for r in rows if r["event"] == "run_start"), {})
    evals = [r for r in rows if r["event"] == "eval_result"
             and r.get("val_set") == "val"]
    final = evals[-1] if evals else {}
    steps = [r for r in rows if r["event"] == "train_step"]
    return {
        "config_sha256": start.get("config_sha256"),
        "total_steps": start.get("total_steps"),
        "train_blocks": start.get("train_blocks"),
        "trainable_params": start.get("trainable_params"),
        "full_rank_trainable_params": start.get("full_rank_trainable_params",
                                                start.get("trainable_params")),
        "lora_trainable_params": start.get("lora_trainable_params", 0),
        "total_params": start.get("total_params"),
        "teacher_native_holdout_ce": final.get("val_ce"),
        "teacher_native_holdout_ppl": final.get("val_ppl"),
        "final_val_kd": final.get("val_kd"),
        "final_train_loss": steps[-1].get("loss") if steps else None,
        "n_eval_points": len(evals),
    }


def nll_table(tag: str) -> dict:
    """FineWeb held-out NLL keyed by the model directory that produced it."""
    path = AUDIT / f"e3_holdout_nll_{tag}.json"
    if not path.is_file():
        return {}
    report = json.loads(path.read_text())
    out = {}
    for r in report["results"]:
        out[r["model"]] = {"mean_nll_nats": r["mean_nll_nats"],
                           "perplexity": r["perplexity"],
                           "eval_tokens": r["eval_tokens"]}
    return out


def match_nll(table: dict, alias: str) -> dict | None:
    """Find this arm's row by the run-directory stem inside the model path."""
    stems = {"A0-P1-sa": "e1_r0860k_sa_pca", "A0-P1-sb": "e1_r0860k_sb_pca",
             "A1-frozen-attn-sa": "e3_a1_frozen_attn_sa",
             "A1-frozen-attn-sb": "e3_a1_frozen_attn_sb",
             "A2-lora-attn-sa": "e3_a2_lora_attn_sa",
             "A2-lora-attn-sb": "e3_a2_lora_attn_sb"}
    stem = stems[alias]
    hits = [v for k, v in table.items() if stem in k]
    return hits[0] if len(hits) == 1 else None


def movement(label: str) -> dict:
    path = AUDIT / "e3_movement" / f"{label}.json"
    if not path.is_file():
        return {"not_evaluable": f"missing {path.name}"}
    rep = json.loads(path.read_text())
    out = {"by_group_relative": {g: v["relative"]
                                 for g, v in rep["by_group"].items()},
           "by_group_delta_fro": {g: v["delta_fro"]
                                  for g, v in rep["by_group"].items()}}
    if "lora" in rep:
        out["lora_delta_by_projection"] = {
            k: v["delta_fro"] for k, v in rep["lora"]["by_projection"].items()}
        out["merged_delta_matches_recomputed_delta"] = \
            rep["lora"]["merged_delta_matches_recomputed_delta"]
    # Per-layer relative movement for the two groups that can move in every arm.
    out["per_layer_relative"] = {
        g: [rep["by_layer"][str(i)].get(g, {}).get("relative")
            for i in range(len(rep["by_layer"]))]
        for g in ("ffn", "attn_proj")
    }
    return out


def mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def spread(values):
    vals = [v for v in values if v is not None]
    return round(max(vals) - min(vals), 4) if len(vals) > 1 else None


def family_summary(arms: dict, members: list[str]) -> dict:
    def pick(alias, *path, default=None):
        node = arms.get(alias, {})
        for key in path:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    fields = {
        "usable_rollout_rate": ("free", "primary", "usable_rollout_rate"),
        "non_empty": ("free", "primary", "non_empty"),
        "natural_termination": ("free", "primary", "natural_termination"),
        "no_severe_repetition": ("free", "primary", "no_severe_repetition"),
        "no_context_limit": ("free", "primary", "no_context_limit"),
        "protocol_valid": ("free", "primary", "protocol_valid"),
        "correct_overall": ("free", "secondary", "correct_overall"),
        "correct_given_usable": ("free", "secondary", "correct_given_usable"),
        "teacher_forced_reasoning_top1":
            ("free", "diagnostics", "teacher_forced_reasoning_top1"),
        "teacher_native_holdout_ce": ("training", "teacher_native_holdout_ce"),
        "fineweb_nll_bf16": ("nll", "bf16", "mean_nll_nats"),
        "fineweb_nll_int8_all": ("nll", "int8_all", "mean_nll_nats"),
        "fineweb_nll_int8_decoder": ("nll", "int8_decoder", "mean_nll_nats"),
    }
    out = {}
    for name, path in fields.items():
        per_seed = {a.rsplit("-", 1)[-1]: pick(a, *path) for a in members}
        out[name] = {"per_seed": per_seed,
                     "mean": mean(per_seed.values()),
                     "spread": spread(per_seed.values())}
    return out


def paired(arms: dict, a: str, b: str) -> dict:
    """Prompt-level wins/losses on the identical fixed example set."""
    ua = arms.get(a, {}).get("free", {}).get("per_sample_usable")
    ub = arms.get(b, {}).get("free", {}).get("per_sample_usable")
    ca = arms.get(a, {}).get("free", {}).get("per_sample_correct")
    cb = arms.get(b, {}).get("free", {}).get("per_sample_correct")
    if not (ua and ub):
        return {"not_evaluable": "missing per-sample records"}
    ids = sorted(set(ua) & set(ub))
    return {
        "n_paired": len(ids),
        "usable_gained": sum(bool(ub[i]) and not bool(ua[i]) for i in ids),
        "usable_lost": sum(bool(ua[i]) and not bool(ub[i]) for i in ids),
        "usable_net": sum(bool(ub[i]) for i in ids) - sum(bool(ua[i]) for i in ids),
        "correct_gained": sum(bool(cb[i]) and not bool(ca[i]) for i in ids),
        "correct_lost": sum(bool(ca[i]) and not bool(cb[i]) for i in ids),
        "correct_net": sum(bool(cb[i]) for i in ids) - sum(bool(ca[i]) for i in ids),
    }


def both_seeds_beat(fam: dict, ref: dict, field: str) -> bool | None:
    """True only when the arm wins on BOTH seeds — no single-seed promotion."""
    a, b = fam[field]["per_seed"], ref[field]["per_seed"]
    if any(a.get(s) is None or b.get(s) is None for s in SEEDS):
        return None
    return all(a[s] > b[s] for s in SEEDS)


def decide(fams: dict) -> dict:
    """The rules registered before the run, applied mechanically."""
    a0, a1, a2 = fams["A0"], fams["A1"], fams["A2"]

    def d(fam, ref, field):
        x, y = fam[field]["mean"], ref[field]["mean"]
        return None if x is None or y is None else round(x - y, 4)

    findings = {
        "a1_vs_a0": {
            "usable_rollout_delta": d(a1, a0, "usable_rollout_rate"),
            "usable_exceeds_p1_seed_spread": (
                None if d(a1, a0, "usable_rollout_rate") is None else
                abs(d(a1, a0, "usable_rollout_rate"))
                > P1_SEED_SPREAD["usable_rollout_rate"]),
            "usable_wins_on_both_seeds": both_seeds_beat(a1, a0,
                                                         "usable_rollout_rate"),
            "correct_overall_delta": d(a1, a0, "correct_overall"),
            "correct_given_usable_delta": d(a1, a0, "correct_given_usable"),
            "fineweb_nll_delta": d(a1, a0, "fineweb_nll_bf16"),
            "teacher_native_ce_delta": d(a1, a0, "teacher_native_holdout_ce"),
        },
        "a2_vs_a0": {
            "usable_rollout_delta": d(a2, a0, "usable_rollout_rate"),
            "usable_wins_on_both_seeds": both_seeds_beat(a2, a0,
                                                         "usable_rollout_rate"),
            "correct_overall_delta": d(a2, a0, "correct_overall"),
            "correct_given_usable_delta": d(a2, a0, "correct_given_usable"),
            "teacher_forced_top1_delta": d(a2, a0,
                                           "teacher_forced_reasoning_top1"),
            "teacher_native_ce_delta": d(a2, a0, "teacher_native_holdout_ce"),
            "fineweb_nll_delta": d(a2, a0, "fineweb_nll_bf16"),
        },
        "a2_vs_a1": {
            "usable_rollout_delta": d(a2, a1, "usable_rollout_rate"),
            "usable_wins_on_both_seeds": both_seeds_beat(a2, a1,
                                                         "usable_rollout_rate"),
            "correct_overall_delta": d(a2, a1, "correct_overall"),
            "correct_given_usable_delta": d(a2, a1, "correct_given_usable"),
        },
    }

    rules = []
    a1v, a2v = findings["a1_vs_a0"], findings["a2_vs_a0"]

    # R1 — A1 improves rollout stability without materially reducing correctness.
    if a1v["usable_rollout_delta"] is not None:
        fired = (a1v["usable_rollout_delta"] > 0
                 and bool(a1v["usable_exceeds_p1_seed_spread"])
                 and bool(a1v["usable_wins_on_both_seeds"])
                 and (a1v["correct_given_usable_delta"] or 0) >= 0)
        rules.append({
            "rule": "R1 full-rank attention updates cause harmful drift",
            "fired": fired,
            "basis": ("A1 usable-rollout gain clears the 0.0800 P1 seed spread on "
                      "both seeds with no loss in correctness | usable rollout"),
        })

    # R2 — A2 beats both A1 and A0 on both seeds.
    if a2v["usable_rollout_delta"] is not None:
        fired = (bool(a2v["usable_wins_on_both_seeds"])
                 and bool(both_seeds_beat(a2, a1, "usable_rollout_rate")))
        rules.append({
            "rule": "R2 constrained attention adaptation is the preferred policy",
            "fired": fired,
            "basis": "A2 > A0 and A2 > A1 on usable rollout, on both seeds",
        })

    # R3 — A2 moves only the teacher-forced diagnostics.
    if a2v["teacher_forced_top1_delta"] is not None:
        fired = (a2v["teacher_forced_top1_delta"] > 0
                 and (a2v["usable_rollout_delta"] or 0) <= 0)
        rules.append({
            "rule": "R3 do NOT claim the main problem is solved",
            "fired": fired,
            "basis": ("teacher-forced top-1 / CE improved while autonomous "
                      "rollout did not"),
        })

    # R4 — general language modelling improves but rollout does not.
    nll_gain = [v["fineweb_nll_delta"] for v in (a1v, a2v)
                if v.get("fineweb_nll_delta") is not None]
    roll = [v["usable_rollout_delta"] for v in (a1v, a2v)
            if v.get("usable_rollout_delta") is not None]
    if len(nll_gain) == 2 and len(roll) == 2:
        fired = all(x < 0 for x in nll_gain) and all(x <= 0 for x in roll)
        rules.append({
            "rule": ("R4 stop freeze-policy exploration; recommend "
                     "student-prefix / on-policy recovery"),
            "fired": fired,
            "basis": "both arms lower FineWeb NLL while neither improves rollout",
        })

    # R5/R6 — the two promotion guards, always reported.
    guards = []
    for name, fam in (("A1", a1), ("A2", a2)):
        wins = both_seeds_beat(fam, a0, "usable_rollout_rate")
        cgu = d(fam, a0, "correct_given_usable")
        guards.append({
            "arm": name,
            "single_seed_only": (wins is False
                                 and (d(fam, a0, "usable_rollout_rate") or 0) > 0),
            "terminates_earlier_but_less_correct_when_usable": (
                (d(fam, a0, "natural_termination") or 0) > 0 and (cgu or 0) < 0),
            "promotable": bool(wins) and (cgu or 0) >= 0,
        })

    return {"findings": findings, "rules": rules, "promotion_guards": guards,
            "noise_floors_used": P1_SEED_SPREAD}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=AUDIT / "e3_comparison.json")
    args = ap.parse_args()

    nll = {tag: nll_table(tag) for tag in ("bf16", "int8_all", "int8_decoder")}
    arms: dict[str, dict] = {}
    for alias, (tm_dir, log, mv) in ARMS.items():
        arms[alias] = {
            "free": three_mode_arm(tm_dir),
            "training": training_metrics(REPO_ROOT / log),
            "movement": movement(mv),
            "nll": {tag: match_nll(table, alias) for tag, table in nll.items()},
        }

    fams = {name: family_summary(arms, members)
            for name, members in FAMILIES.items()}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "experiment": "E3 — restricting attention updates at the 0.86M rung",
        "arms": {"A0": "0.86M P1 control: FFN + norms + attention projections",
                 "A1": "FFN + all norms; attention projections frozen",
                 "A2": "A1 + LoRA r8 on q/k/v/o, base projections frozen"},
        "evaluation": {
            "set": "150 fixed examples, inclusion mask d6e24e0b…",
            "decoding": "greedy, temperature 0, unrestricted generation (P18)",
            "harness": "run_three_mode_diagnostic.py — free, oracle, forced",
            "correctness": "RE-SCORED with the corrected scorer, never the "
                           "stored `correct` field",
            "hierarchy": "primary usable_rollout; secondary correctness; "
                         "diagnostic teacher-forced top-1, CE, FineWeb NLL",
        },
        "per_arm": arms,
        "per_family": fams,
        "paired_prompt_level": {
            f"{b} vs {a}": paired(arms, a, b)
            for a, b in [
                ("A0-P1-sa", "A1-frozen-attn-sa"),
                ("A0-P1-sb", "A1-frozen-attn-sb"),
                ("A0-P1-sa", "A2-lora-attn-sa"),
                ("A0-P1-sb", "A2-lora-attn-sb"),
                ("A1-frozen-attn-sa", "A2-lora-attn-sa"),
                ("A1-frozen-attn-sb", "A2-lora-attn-sb"),
            ]},
        "decision": decide(fams),
        "code_state": code_state(REPO_ROOT),
    }
    for arm in report["per_arm"].values():
        arm["free"].pop("per_sample_usable", None)
        arm["free"].pop("per_sample_correct", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    print(f"{'metric':38s} {'A0':>10s} {'A1':>10s} {'A2':>10s}")
    for metric in ("usable_rollout_rate", "natural_termination",
                   "no_context_limit", "no_severe_repetition", "non_empty",
                   "protocol_valid", "correct_overall", "correct_given_usable",
                   "teacher_forced_reasoning_top1", "teacher_native_holdout_ce",
                   "fineweb_nll_bf16", "fineweb_nll_int8_all"):
        row = [fams[f][metric]["mean"] for f in ("A0", "A1", "A2")]
        cells = "".join(f"{'  n/a' if v is None else f'{v:10.4f}'}" for v in row)
        print(f"{metric:38s}{cells}")
    print("\nRules that fired:")
    for r in report["decision"]["rules"]:
        print(f"  [{'X' if r['fired'] else ' '}] {r['rule']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
