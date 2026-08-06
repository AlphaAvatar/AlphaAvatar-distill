#!/usr/bin/env python
"""Experiment 4 comparison: does P2-CE-heavy improve when scaled 0.86M → 1.60M?

    PYTHONPATH=src python scripts/evaluation/analyze_e4.py \
        --out artifacts/audit/e4_comparison.json

Two questions, kept separate because they have different controls:

* **scale inside P2** — P2-0.86M vs P2-1.60M (the main question);
* **objective at matched scale** — P1-1.60M vs P2-1.60M.

Nothing is generated: retained artifacts are re-read and correctness is
re-scored with the corrected scorer, never taken from the stored `correct`
field. The decision priority is the registered one — `correct_overall`, then
`correct_and_naturally_terminated`, then `usable_rollout_rate`, then
repetition/context-limit, then `correct_given_usable`. Teacher-forced CE/top-1
and FineWeb NLL are explanatory diagnostics and never select a winner.

Paired statistics resample **prompts at fixed checkpoints**, so an interval that
excludes zero is not by itself evidence the recipe moved: the registered seed
noise floor governs that, and both are reported side by side.
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

from aadistill.evaluation.paired_stats import (  # noqa: E402
    joint_rate, mcnemar_counts, paired_bootstrap_ci,
)
from aadistill.infrastructure.env import code_state  # noqa: E402
from reevaluate_stage23 import three_mode_arm  # noqa: E402

AUDIT = REPO_ROOT / "artifacts/audit"

# alias -> (three-mode dir, training log or None, movement label or None)
ARMS = {
    "P2-0.86M-sa": ("P2-ceheavy-sa", "artifacts/stage3/p2_ceheavy_sa/train_log.jsonl",
                    "A0-P2-sa"),
    "P2-0.86M-sb": ("P2-ceheavy-sb", "artifacts/stage3/p2_ceheavy_sb/train_log.jsonl",
                    "A0-P2-sb"),
    "P1-1.60M-sa": ("P1-1600k-sa",
                    "artifacts/stage3/rescued/_relay/e1_r1600k_sa_pca/train_log.jsonl",
                    None),
    "P1-1.60M-sb": ("P1-1600k-sb",
                    "artifacts/stage3/rescued/_relay/e1_r1600k_sb_pca/train_log.jsonl",
                    None),
    "P2-1.60M-sa": ("E4-P2-1600k-sa", "artifacts/stage3/e4_p2_r1600k_sa/train_log.jsonl",
                    "E4-P2-1600k-sa"),
    "P2-1.60M-sb": ("E4-P2-1600k-sb", "artifacts/stage3/e4_p2_r1600k_sb/train_log.jsonl",
                    "E4-P2-1600k-sb"),
}
FAMILIES = {"P2-0.86M": ["P2-0.86M-sa", "P2-0.86M-sb"],
            "P1-1.60M": ["P1-1.60M-sa", "P1-1.60M-sb"],
            "P2-1.60M": ["P2-1.60M-sa", "P2-1.60M-sb"]}
SEEDS = ("sa", "sb")

# The larger of the two measured families' spreads, as registered for E3.
NOISE_FLOOR = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600,
               "teacher_forced_reasoning_top1": 0.0112,
               "teacher_native_holdout_ce": 0.0117}

PRIORITY = ["correct_overall", "correct_and_naturally_terminated",
            "usable_rollout_rate", "no_severe_repetition", "no_context_limit",
            "correct_given_usable"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open() if line.strip()]


def training_metrics(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"not_evaluable": f"missing {path.name if path else 'train log'}"}
    rows = jsonl(path)
    start = next((r for r in rows if r["event"] == "run_start"), {})
    evals = [r for r in rows if r["event"] == "eval_result"
             and r.get("val_set") == "val"]
    steps = [r for r in rows if r["event"] == "train_step"]
    return {
        "config_sha256": start.get("config_sha256"),
        "total_steps": start.get("total_steps"),
        "train_blocks": start.get("train_blocks"),
        "trainable_params": start.get("trainable_params"),
        "teacher_native_holdout_ce": evals[-1]["val_ce"] if evals else None,
        "final_train_loss": steps[-1]["loss"] if steps else None,
        "seconds": next((r["seconds"] for r in rows if r["event"] == "run_end"), None),
    }


def nll_row(alias: str, tag: str) -> dict | None:
    path = AUDIT / f"e4_holdout_nll_{tag}.json"
    if not path.is_file():
        return None
    stems = {"P2-0.86M-sa": "p2_ceheavy_sa", "P2-0.86M-sb": "p2_ceheavy_sb",
             "P1-1.60M-sa": "e1_r1600k_sa_pca", "P1-1.60M-sb": "e1_r1600k_sb_pca",
             "P2-1.60M-sa": "e4_p2_r1600k_sa", "P2-1.60M-sb": "e4_p2_r1600k_sb"}
    hits = [r for r in json.loads(path.read_text())["results"]
            if stems[alias] in r["model"]]
    if len(hits) != 1:
        return None
    r = hits[0]
    return {"mean_nll_nats": r["mean_nll_nats"], "perplexity": r["perplexity"],
            "eval_tokens": r["eval_tokens"]}


def movement(label: str | None) -> dict:
    if label is None:
        return {"not_measured": "reference arm; movement not recomputed"}
    for name in (f"e4_movement/{label}.json", f"e3_movement/{label}.json"):
        path = AUDIT / name
        if path.is_file():
            rep = json.loads(path.read_text())
            return {g: v["relative"] for g, v in rep["by_group"].items()}
    return {"not_evaluable": f"missing movement report for {label}"}


def per_sample(arm: dict) -> tuple[dict, dict, dict]:
    """(correct, naturally_terminated, usable) keyed by prompt id."""
    return (arm.get("per_sample_correct", {}),
            arm.get("per_sample_terminated", {}),
            arm.get("per_sample_usable", {}))


def mean(vals):
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def spread(vals):
    v = [x for x in vals if x is not None]
    return round(max(v) - min(v), 4) if len(v) > 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=AUDIT / "e4_comparison.json")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    arms: dict[str, dict] = {}
    for alias, (tm_dir, log, mv) in ARMS.items():
        a = three_mode_arm(tm_dir)
        if "not_evaluable" not in a:
            # natural termination per prompt, for the joint metric
            recs = jsonl(AUDIT / "three_mode" / tm_dir / "free.generations.jsonl")
            term = {r["id"]: bool(r["natural_termination"]) for r in recs}
            a["per_sample_terminated"] = term
            a["joint"] = joint_rate(a["per_sample_correct"], term)
        arms[alias] = {
            "free": a,
            "training": training_metrics(REPO_ROOT / log if log else None),
            "movement": movement(mv),
            "nll": {t: nll_row(alias, t)
                    for t in ("bf16", "int8_all", "int8_decoder")},
        }

    def pick(alias, *path):
        node = arms.get(alias, {})
        for k in path:
            if not isinstance(node, dict):
                return None
            node = node.get(k)
        return node

    fields = {
        "correct_overall": ("free", "secondary", "correct_overall"),
        "correct_and_naturally_terminated":
            ("free", "joint", "correct_and_naturally_terminated"),
        "usable_rollout_rate": ("free", "primary", "usable_rollout_rate"),
        "no_severe_repetition": ("free", "primary", "no_severe_repetition"),
        "no_context_limit": ("free", "primary", "no_context_limit"),
        "natural_termination": ("free", "primary", "natural_termination"),
        "non_empty": ("free", "primary", "non_empty"),
        "protocol_valid": ("free", "primary", "protocol_valid"),
        "correct_given_usable": ("free", "secondary", "correct_given_usable"),
        "teacher_forced_reasoning_top1":
            ("free", "diagnostics", "teacher_forced_reasoning_top1"),
        "teacher_native_holdout_ce": ("training", "teacher_native_holdout_ce"),
        "fineweb_nll_bf16": ("nll", "bf16", "mean_nll_nats"),
        "fineweb_nll_int8_all": ("nll", "int8_all", "mean_nll_nats"),
    }
    families = {}
    for fam, members in FAMILIES.items():
        families[fam] = {}
        for name, path in fields.items():
            ps = {m.rsplit("-", 1)[-1]: pick(m, *path) for m in members}
            families[fam][name] = {"per_seed": ps, "mean": mean(ps.values()),
                                   "spread": spread(ps.values())}

    # ---- paired, per seed, on the identical fixed battery ---------------
    comparisons = {}
    for label, (ref_fam, arm_fam) in (
            ("scale: P2-1.60M vs P2-0.86M", ("P2-0.86M", "P2-1.60M")),
            ("objective: P2-1.60M vs P1-1.60M", ("P1-1.60M", "P2-1.60M"))):
        per_seed = {}
        for s in SEEDS:
            ref, arm = f"{ref_fam}-{s}", f"{arm_fam}-{s}"
            ca, ta, ua = per_sample(arms[ref]["free"])
            cb, tb, ub = per_sample(arms[arm]["free"])
            if not (ca and cb):
                per_seed[s] = {"not_evaluable": "missing per-sample records"}
                continue
            ja = arms[ref]["free"]["joint"]["per_sample"]
            jb = arms[arm]["free"]["joint"]["per_sample"]
            per_seed[s] = {
                "correct_overall": {
                    **mcnemar_counts(ca, cb),
                    **paired_bootstrap_ci(ca, cb, iterations=args.bootstrap)},
                "correct_and_naturally_terminated": {
                    **mcnemar_counts(ja, jb),
                    **paired_bootstrap_ci(ja, jb, iterations=args.bootstrap)},
                "usable_rollout_rate": {
                    **mcnemar_counts(ua, ub),
                    **paired_bootstrap_ci(ua, ub, iterations=args.bootstrap)},
            }
        deltas = {m: round((families[arm_fam][m]["mean"] or 0)
                           - (families[ref_fam][m]["mean"] or 0), 4)
                  for m in PRIORITY if families[arm_fam][m]["mean"] is not None
                  and families[ref_fam][m]["mean"] is not None}
        wins_both = {}
        for m in PRIORITY:
            a_, b_ = families[ref_fam][m]["per_seed"], families[arm_fam][m]["per_seed"]
            wins_both[m] = (None if any(a_.get(s) is None or b_.get(s) is None
                                        for s in SEEDS)
                            else all(b_[s] > a_[s] for s in SEEDS))
        comparisons[label] = {"mean_deltas": deltas,
                              "wins_on_both_seeds": wins_both,
                              "per_seed_paired": per_seed}

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "experiment": "E4 — P2-CE-heavy scaled from the 0.86M to the 1.60M rung",
        "decision_priority": PRIORITY,
        "noise_floors": NOISE_FLOOR,
        "evaluation": {
            "set": "150 fixed examples, inclusion mask d6e24e0b…, rung pinned to "
                   "860000 for every arm so the battery cannot resample",
            "decoding": "greedy, temperature 0, unrestricted generation (P18)",
            "correctness": "RE-SCORED, never the stored `correct` field",
            "paired_stats": "bootstrap resamples PROMPTS at fixed checkpoints; "
                            "it does not model seed variation",
        },
        "per_arm": arms,
        "per_family": families,
        "comparisons": comparisons,
        "code_state": code_state(REPO_ROOT),
    }
    for a in report["per_arm"].values():
        for k in ("per_sample_usable", "per_sample_correct", "per_sample_terminated"):
            a["free"].pop(k, None)
        if isinstance(a["free"].get("joint"), dict):
            a["free"]["joint"].pop("per_sample", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    order = ["P2-0.86M", "P1-1.60M", "P2-1.60M"]
    print(f"{'metric':36s}" + "".join(f"{f:>12s}" for f in order))
    for m in PRIORITY + ["teacher_native_holdout_ce", "fineweb_nll_bf16"]:
        cells = "".join(
            "         n/a" if families[f][m]["mean"] is None
            else f"{families[f][m]['mean']:12.4f}" for f in order)
        print(f"{m:36s}{cells}")
    print()
    for label, c in comparisons.items():
        print(f"{label}:")
        for m, d in c["mean_deltas"].items():
            both = c["wins_on_both_seeds"].get(m)
            floor = NOISE_FLOOR.get(m)
            note = ""
            if floor is not None:
                note = f" (floor {floor}, {'CLEARS' if abs(d) > floor else 'inside'})"
            print(f"   {m:34s} {d:+.4f}  both-seeds={both}{note}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
