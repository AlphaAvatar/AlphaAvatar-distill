#!/usr/bin/env python
"""Experiment 6b analysis: objective x data-scale interaction, on one battery.

    PYTHONPATH=src python scripts/evaluation/analyze_e6b.py --bootstrap 10000

Four cells, one frozen 150-prompt battery, every arm re-scored from its retained
raw generations with the current scorer:

                      1.60M            2.96M
    E1/P1 KD-heavy    E1-1.60M         E1-2.96M
    P2    CE-heavy    P2-1.60M         **P2-2.96M**

Three comparisons fall out of that, plus the one the matrix exists for:

    A  P2-2.96M vs E1-2.96M     does the objective matter at this scale?
    B  P2-2.96M vs P2-1.60M     does P2 gain from the rung?
    C  E1-2.96M vs E1-1.60M     the E6 reference, re-derived here
    D  (B) - (C)                does the objective INTERACT with scale?

D is the point of the experiment and the noisiest thing in it: a
difference-in-differences over four two-seed cells compounds four single draws.
It is claimed only when it clears the metric's registered floor **and** agrees in
direction across seeds — a nonzero point estimate is not evidence.

Nothing is generated here. Given the artifacts, the report is reproducible
without a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from aadistill.evaluation.paired_stats import (  # noqa: E402
    mcnemar_counts, paired_bootstrap_ci,
)
from aadistill.infrastructure.env import code_state  # noqa: E402
from analyze_e6 import (  # noqa: E402  — one scorer, shared by both experiments
    arm_alias, load_sessions, rescore_arm, token_stream_sha256,
)

AUDIT = REPO_ROOT / "artifacts/audit"
THREE_MODE = AUDIT / "three_mode"
REGISTRATION = REPO_ROOT / "logs/e6b_registration.json"

FLOORS = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600}
SEEDS = ("sa", "sb")

# family -> (three_mode directory stem per seed, unique CE tokens)
FAMILIES = {
    "E1-1.60M": ({"sa": "E1-1.60M-sa", "sb": "E1-1.60M-sb"}, 1600353),
    "E1-2.96M": ({"sa": "E1-2.96M-sa", "sb": "E1-2.96M-sb"}, 2960507),
    "P2-1.60M": ({"sa": "E4-P2-1600k-sa", "sb": "E4-P2-1600k-sb"}, 1600353),
    "P2-2.96M": ({"sa": "P2-2.96M-sa", "sb": "P2-2.96M-sb"}, 2960507),
}
EPOCHS = 3

# Every metric the interaction is computed on. Rates where LOWER is better are
# flagged so the report never calls a reduction in failures a regression.
INTERACTION_METRICS = {
    "usable_rollout_rate": "higher",
    "correct_overall": "higher",
    "correct_given_usable": "higher",
    "natural_termination_rate": "higher",
    "context_limit_rate": "lower",
    "severe_repetition_rate": "lower",
    "empty_output_rate": "lower",
    "answer_parse_failure_rate_numeric": "lower",
}

QUALITATIVE_BUCKETS = (
    "newly_usable", "newly_unusable", "newly_correct", "newly_incorrect",
    "termination_improved", "termination_regressed", "both_usable_disagree",
)


def family_metric(arms: dict, family: str, metric: str, seed: str):
    a = arms.get(f"{family}-{seed}")
    return None if a is None else a.get(metric)


def family_mean(arms: dict, family: str, metric: str):
    vals = [family_metric(arms, family, metric, s) for s in SEEDS]
    if any(v is None for v in vals):
        return None
    return round(statistics.fmean(vals), 4)


def compare(a_alias: str, b_alias: str, arms: dict, iterations: int) -> dict | None:
    """b against a, paired on the shared 150-prompt mask."""
    a, b = arms.get(a_alias), arms.get(b_alias)
    if a is None or b is None:
        return None
    out = {"a": a_alias, "b": b_alias}
    for axis in ("usable", "correct", "natural_termination", "context_limit"):
        pa, pb = a["per_sample"][axis], b["per_sample"][axis]
        counts = mcnemar_counts(pa, pb)
        row = {"rate_a": counts["rate_a"], "rate_b": counts["rate_b"],
               "delta": counts["delta"], "win": counts["b_gained"],
               "loss": counts["b_lost"],
               "tie": counts["both_true"] + counts["both_false"],
               "n_paired": counts["n_paired"]}
        if axis in ("usable", "correct"):
            ci = paired_bootstrap_ci(pa, pb, iterations=iterations)
            row["bootstrap_ci"] = [ci["ci_low"], ci["ci_high"]]
            row["ci_excludes_zero"] = ci["ci_excludes_zero"]
        out[axis] = row
    return out


def _verdict(pooled: float, directions: set, floor: float, better: str) -> str:
    if abs(pooled) < floor:
        return "inside the floor — a tie"
    if len(directions) != 1:
        return "above the floor but seeds disagree — not claimable"
    improved = (pooled > 0) if better == "higher" else (pooled < 0)
    return f"above the floor and seed-consistent ({'better' if improved else 'worse'})"


def family_compare(fa: str, fb: str, arms: dict, iterations: int) -> dict:
    per_seed = {s: compare(f"{fa}-{s}", f"{fb}-{s}", arms, iterations) for s in SEEDS}
    per_seed = {s: v for s, v in per_seed.items() if v is not None}
    out = {"a": fa, "b": fb, "per_seed": per_seed}
    if len(per_seed) != len(SEEDS):
        out["incomplete"] = (f"needs both seeds of {fa} and {fb}; have "
                             f"{sorted(per_seed) or 'neither'}. No claim from one seed.")
        return out
    for axis, floor_key, better in (("usable", "usable_rollout_rate", "higher"),
                                    ("correct", "correct_overall", "higher")):
        deltas = [per_seed[s][axis]["delta"] for s in SEEDS]
        pooled = round(statistics.fmean(deltas), 4)
        directions = {"up" if d > 0 else "down" if d < 0 else "flat" for d in deltas}
        out[axis] = {
            "delta_per_seed": {s: per_seed[s][axis]["delta"] for s in SEEDS},
            "pooled_delta": pooled,
            "seed_consistent": len(directions) == 1 and directions != {"flat"},
            "floor": FLOORS[floor_key],
            "exceeds_floor": abs(pooled) >= FLOORS[floor_key],
            "both_cis_exclude_zero": all(
                per_seed[s][axis]["ci_excludes_zero"] for s in SEEDS),
            "verdict": _verdict(pooled, directions, FLOORS[floor_key], better),
        }
    return out


def interaction(arms: dict) -> dict:
    """(P2_2.96 - P2_1.60) - (E1_2.96 - E1_1.60), per metric and per seed.

    Seed pairing is valid here because `sa` and `sb` name the same two training
    seeds (20260726 / 20260801) in all four cells, so a per-seed interaction
    compares like with like. It is still four single draws stacked, which is why
    the direction agreement across seeds matters more than the point estimate.
    """
    out = {}
    for metric, better in INTERACTION_METRICS.items():
        cells = {f: {s: family_metric(arms, f, metric, s) for s in SEEDS}
                 for f in FAMILIES}
        if any(v is None for f in cells.values() for v in f.values()):
            out[metric] = {"not_evaluable": "a cell is missing"}
            continue
        per_seed = {}
        for s in SEEDS:
            p2 = cells["P2-2.96M"][s] - cells["P2-1.60M"][s]
            e1 = cells["E1-2.96M"][s] - cells["E1-1.60M"][s]
            per_seed[s] = {"p2_scale_delta": round(p2, 4),
                           "e1_scale_delta": round(e1, 4),
                           "interaction": round(p2 - e1, 4)}
        p2_pool = round(statistics.fmean(
            [cells["P2-2.96M"][s] for s in SEEDS])
            - statistics.fmean([cells["P2-1.60M"][s] for s in SEEDS]), 4)
        e1_pool = round(statistics.fmean(
            [cells["E1-2.96M"][s] for s in SEEDS])
            - statistics.fmean([cells["E1-1.60M"][s] for s in SEEDS]), 4)
        inter = round(p2_pool - e1_pool, 4)
        dirs = {"up" if per_seed[s]["interaction"] > 0 else
                "down" if per_seed[s]["interaction"] < 0 else "flat" for s in SEEDS}
        floor = FLOORS.get(metric)
        out[metric] = {
            "better_is": better,
            "cells": {f: {s: round(cells[f][s], 4) for s in SEEDS} for f in FAMILIES},
            "p2_scale_delta_pooled": p2_pool,
            "e1_scale_delta_pooled": e1_pool,
            "interaction_pooled": inter,
            "interaction_per_seed": {s: per_seed[s]["interaction"] for s in SEEDS},
            "direction_consistent": len(dirs) == 1 and dirs != {"flat"},
            "floor": floor,
            "claimable": bool(floor is not None and abs(inter) >= floor
                              and len(dirs) == 1 and dirs != {"flat"}),
            "verdict": (
                "no registered floor for this metric — reported, never claimed"
                if floor is None else
                _verdict(inter, dirs, floor, "higher")),
        }
    return out


def qualitative(arms: dict, sessions: dict, a_alias: str, b_alias: str,
                k: int) -> dict:
    """A deterministic sample of what changed, for explanation only."""
    a, b = arms.get(a_alias), arms.get(b_alias)
    if a is None or b is None:
        return {}
    buckets: dict[str, list] = {n: [] for n in QUALITATIVE_BUCKETS}
    for pid in sorted(set(a["per_sample"]["usable"]) & set(b["per_sample"]["usable"])):
        ua, ub = a["per_sample"]["usable"][pid], b["per_sample"]["usable"][pid]
        ca, cb = a["per_sample"]["correct"][pid], b["per_sample"]["correct"][pid]
        ta = a["per_sample"]["natural_termination"][pid]
        tb = b["per_sample"]["natural_termination"][pid]
        row = {"id": pid, "data_type": sessions[pid]["data_type"],
               "tokens_a": a["per_sample"]["generated_tokens"][pid],
               "tokens_b": b["per_sample"]["generated_tokens"][pid]}
        if ub and not ua:
            buckets["newly_usable"].append(row)
        if ua and not ub:
            buckets["newly_unusable"].append(row)
        if cb and not ca:
            buckets["newly_correct"].append(row)
        if ca and not cb:
            buckets["newly_incorrect"].append(row)
        if tb and not ta:
            buckets["termination_improved"].append(row)
        if ta and not tb:
            buckets["termination_regressed"].append(row)
        if ua and ub and ca != cb:
            buckets["both_usable_disagree"].append({**row, "a_correct": ca,
                                                    "b_correct": cb})
    return {n: {"n": len(rows), "sample": rows[:k]} for n, rows in buckets.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs/e6b_results.json")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "logs/e6b_report.md")
    ap.add_argument("--per-prompt", type=Path,
                    default=AUDIT / "e6b_per_prompt.jsonl")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    sessions = load_sessions()
    arms, provenance, missing = {}, {}, []
    for family, (dirs, _) in FAMILIES.items():
        for seed, stem in dirs.items():
            alias = f"{family}-{seed}"
            d = THREE_MODE / stem
            scored = rescore_arm(d, sessions)
            if scored is None:
                missing.append(alias)
                continue
            arms[alias] = scored
            provenance[alias] = {
                "directory": str(d.relative_to(REPO_ROOT)),
                "family": family, "seed": seed,
                "token_stream_sha256": token_stream_sha256(d),
            }

    comparisons = {
        "A P2-2.96M vs E1-2.96M": family_compare("E1-2.96M", "P2-2.96M", arms, args.bootstrap),
        "B P2-2.96M vs P2-1.60M": family_compare("P2-1.60M", "P2-2.96M", arms, args.bootstrap),
        "C E1-2.96M vs E1-1.60M": family_compare("E1-1.60M", "E1-2.96M", arms, args.bootstrap),
        "P2-1.60M vs E1-1.60M": family_compare("E1-1.60M", "P2-1.60M", arms, args.bootstrap),
    }
    families = {}
    for f, (_, unique) in FAMILIES.items():
        present = [f"{f}-{s}" for s in SEEDS if f"{f}-{s}" in arms]
        if len(present) != len(SEEDS):
            continue
        families[f] = {
            "unique_ce_tokens": unique,
            "cumulative_ce_tokens": unique * EPOCHS,
            "seeds": {a: {k: arms[a][k] for k in
                          ("usable_rollout_rate", "correct_overall",
                           "correct_given_usable", "protocol_valid_rate",
                           "natural_termination_rate", "context_limit_rate",
                           "severe_repetition_rate", "empty_output_rate",
                           "answer_parse_failure_rate_numeric",
                           "generated_tokens", "counts")} for a in present},
            "mean": {k: family_mean(arms, f, k) for k in
                     ("usable_rollout_rate", "correct_overall",
                      "correct_given_usable", "natural_termination_rate",
                      "context_limit_rate", "severe_repetition_rate",
                      "empty_output_rate")},
            "correct_given_usable_pooled": round(
                sum(arms[a]["counts"]["correct_and_usable"] for a in present)
                / max(1, sum(arms[a]["counts"]["usable"] for a in present)), 4),
        }

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E6b — P2 CE-heavy at 2.96M; objective x data-scale interaction",
        "floors": FLOORS,
        "arms": {a: {k: v for k, v in m.items() if k != "per_sample"}
                 for a, m in arms.items()},
        "provenance": provenance,
        "missing_arms": missing,
        "families": families,
        "comparisons": comparisons,
        "interaction": interaction(arms),
        "qualitative_examples": {
            f"P2-2.96M vs E1-2.96M [{s}]":
                qualitative(arms, sessions, f"E1-2.96M-{s}", f"P2-2.96M-{s}",
                            args.examples) for s in SEEDS},
        "code_state": code_state(str(REPO_ROOT)),
    }
    if REGISTRATION.is_file():
        reg = json.loads(REGISTRATION.read_text())
        result["registration_sha256"] = reg["registration_sha256"]
        result["inclusion_mask_sha256"] = reg["frozen_assets"]["inclusion_mask_sha256"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")

    args.per_prompt.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.per_prompt.open("w") as fh:
        for alias, m in sorted(arms.items()):
            ps = m["per_sample"]
            for pid in sorted(ps["usable"]):
                fh.write(json.dumps({
                    "arm": alias, "id": pid,
                    "data_type": sessions[pid]["data_type"],
                    "usable_rollout": ps["usable"][pid],
                    "correct": ps["correct"][pid],
                    "natural_termination": ps["natural_termination"][pid],
                    "context_limit": ps["context_limit"][pid],
                    "generated_tokens": ps["generated_tokens"][pid],
                }) + "\n")
                n += 1
    print(f"wrote {args.per_prompt} ({n} scored records)")
    if missing:
        print(f"  MISSING ARMS: {missing}")

    args.report.write_text(render(result))
    print(f"wrote {args.report}")


def render(r: dict) -> str:
    L = ["# Experiment 6b — objective × data-scale interaction", "",
         f"Generated {r['created_utc']} from retained generations only. Inclusion",
         f"mask `{r.get('inclusion_mask_sha256', '?')[:16]}…`, 150 prompts, greedy,",
         "unrestricted generation (P18), every arm re-scored with the current scorer.",
         "", "## Headline", "",
         "| model | unique CE | cumulative CE | seed | usable | correct | correct\\|usable |",
         "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for fam in ("E1-1.60M", "E1-2.96M", "P2-1.60M", "P2-2.96M"):
        f = r["families"].get(fam)
        if f is None:
            L.append(f"| {fam} | — | — | — | not evaluated | | |")
            continue
        for alias, m in f["seeds"].items():
            cgu = m["correct_given_usable"]
            L.append(f"| {fam} | {f['unique_ce_tokens']:,} | "
                     f"{f['cumulative_ce_tokens']:,} | {alias.rsplit('-', 1)[-1]} | "
                     f"{m['usable_rollout_rate']:.4f} | {m['correct_overall']:.4f} | "
                     f"{'—' if cgu is None else f'{cgu:.4f}'} |")
        L.append(f"| **{fam}** | | | **mean** | "
                 f"**{f['mean']['usable_rollout_rate']:.4f}** | "
                 f"**{f['mean']['correct_overall']:.4f}** | "
                 f"{f['correct_given_usable_pooled']:.4f} |")

    L += ["", "## Paired comparisons on the shared mask", ""]
    for name, c in r["comparisons"].items():
        L.append(f"### {name}")
        if "usable" not in c:
            L += ["", f"Not evaluable: {c.get('incomplete', 'missing arms')}.", ""]
            continue
        for axis in ("usable", "correct"):
            a = c[axis]
            L.append(f"* **{axis}** pooled Δ {a['pooled_delta']:+.4f} "
                     f"(floor {a['floor']:.4f}) — {a['verdict']}; seeds "
                     f"{a['delta_per_seed']}, seed-consistent {a['seed_consistent']}")
            for s, v in c["per_seed"].items():
                x = v[axis]
                L.append(f"  * {s}: {x['rate_a']:.4f} → {x['rate_b']:.4f}, "
                         f"win/tie/loss {x['win']}/{x['tie']}/{x['loss']}, "
                         f"95% CI [{x['bootstrap_ci'][0]:+.4f}, "
                         f"{x['bootstrap_ci'][1]:+.4f}]"
                         f"{' excludes 0' if x['ci_excludes_zero'] else ''}")
        L.append("")

    L += ["## Objective × scale interaction", "",
          "`(P2_2.96 − P2_1.60) − (E1_2.96 − E1_1.60)`. Four two-seed cells, so",
          "this compounds four single draws — the direction agreement across seeds",
          "carries more weight than the point estimate, and a nonzero value is not",
          "by itself evidence of interaction.", "",
          "| metric | better | P2 scale Δ | E1 scale Δ | interaction | per seed | consistent | claimable |",
          "| --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for metric, v in r["interaction"].items():
        if "not_evaluable" in v:
            L.append(f"| {metric} | — | — | — | — | — | — | not evaluable |")
            continue
        L.append(f"| {metric} | {v['better_is']} | {v['p2_scale_delta_pooled']:+.4f} | "
                 f"{v['e1_scale_delta_pooled']:+.4f} | "
                 f"**{v['interaction_pooled']:+.4f}** | "
                 f"{v['interaction_per_seed']} | {v['direction_consistent']} | "
                 f"{'**yes**' if v['claimable'] else 'no'} |")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
