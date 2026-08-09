#!/usr/bin/env python
"""Experiment 7 analysis: does restored general language modelling reach behaviour?

    PYTHONPATH=src python scripts/evaluation/analyze_e7.py --bootstrap 10000

Three arms, one frozen 150-prompt battery, every arm re-scored from its retained
raw generations with the current scorer. Arm A is the **retained** E1/P1 KD-heavy
1.60M baseline — the same generations E6 produced, not a new run.

    A  E1-1.60M            rollout stream only
    B  A + FineWeb-Edu raw-text teacher KD
    C  A + an exactly matched in-domain KD-only stream

Three comparisons, and the third is the one the control exists for:

    B vs A    FineWeb plus extra KD: the total effect
    C vs A    matched extra KD alone: how much is just more KD signal
    B vs C    what FineWeb's CONTENT adds beyond that

The interpretation is fixed in advance (`logs/e7_preregistration.md` 7.4) and
separates three things that are easy to blur: general-language restoration,
autonomous stability, and autonomous reasoning correctness. **If B improves the
general-text diagnostics but does not beat C on autonomous correctness, FineWeb
did not solve the reasoning bottleneck** — and this script must not be read as
saying otherwise.

The general-text diagnostics are reported alongside, clearly marked. They may
not promote a checkpoint (decision record 2026-08-09).

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
REGISTRATION = REPO_ROOT / "logs/e7_preregistration.md"

FLOORS = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600}
SEEDS = ("sa", "sb")

# family -> (three_mode directory stem per seed, unique CE tokens)
FAMILIES = {
    "A-Baseline": ({"sa": "E1-1.60M-sa", "sb": "E1-1.60M-sb"}, 1600353),
    "B-FineWeb": ({"sa": "E7-B-FineWeb-sa", "sb": "E7-B-FineWeb-sb"}, 1600353),
    "C-Control": ({"sa": "E7-C-Control-sa", "sb": "E7-C-Control-sb"}, 1600353),
}
# Every arm trains the identical 1.60M rollout stream; the extra KD-only stream
# is the only difference, and B and C consume the SAME number of extra positions.
EXTRA_KD_POSITIONS = {"A-Baseline": 0, "B-FineWeb": 1801503, "C-Control": 1801503}
GENERAL_TEXT = AUDIT / "e7_general_text"
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


def general_text(reports: Path = GENERAL_TEXT) -> dict:
    """The general-language diagnostics, carried into the record but fenced off.

    They answer question 1 of 3 and may not promote anything. Reported here so
    the three questions can be read side by side without any of them being
    substituted for another.
    """
    out = {"status": "DIAGNOSTIC ONLY — never promotes a checkpoint "
                     "(decision record 2026-08-09)", "arms": {}}
    if not reports.is_dir():
        return out
    for p in sorted(reports.glob("*.json")):
        if p.name.endswith(".holdout_v1.json"):
            continue
        d = json.loads(p.read_text())
        out["arms"][p.stem] = {
            "stream": d.get("stream"),
            "stream_manifest_sha256": d.get("stream_manifest_sha256"),
            **{k: d["metrics"][k] for k in
               ("nll", "ppl", "kl", "top1", "mean_rank", "mean_target_prob",
                "mean_entropy", "positions") if k in d.get("metrics", {})},
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
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs/e7_results.json")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "logs/e7_report.md")
    ap.add_argument("--per-prompt", type=Path,
                    default=AUDIT / "e7_per_prompt.jsonl")
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
        "B vs A — FineWeb + extra KD, total effect":
            family_compare("A-Baseline", "B-FineWeb", arms, args.bootstrap),
        "C vs A — matched extra KD alone":
            family_compare("A-Baseline", "C-Control", arms, args.bootstrap),
        "B vs C — FineWeb content, beyond extra KD":
            family_compare("C-Control", "B-FineWeb", arms, args.bootstrap),
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
        "experiment": "E7 — FineWeb teacher-KD mixture at the fixed 1.60M rung",
        "general_text_diagnostics": general_text(),
        "extra_kd_positions": EXTRA_KD_POSITIONS,
        "floors": FLOORS,
        "arms": {a: {k: v for k, v in m.items() if k != "per_sample"}
                 for a, m in arms.items()},
        "provenance": provenance,
        "missing_arms": missing,
        "families": families,
        "comparisons": comparisons,
        "qualitative_examples": {
            f"P2-2.96M vs E1-2.96M [{s}]":
                qualitative(arms, sessions, f"A-Baseline-{s}", f"B-FineWeb-{s}",
                            args.examples) for s in SEEDS},
        "qualitative_b_vs_c": {s: qualitative(
            arms, sessions, f"C-Control-{s}", f"B-FineWeb-{s}", args.examples)
            for s in SEEDS},
        "code_state": code_state(str(REPO_ROOT)),
    }
    # The preregistration is prose, not JSON; pin it by content hash. The
    # inclusion mask comes from the arms' own reports, where the driver asserted
    # it against the binding value after every evaluation.
    if REGISTRATION.is_file():
        result["preregistration"] = {
            "path": str(REGISTRATION.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(REGISTRATION.read_bytes()).hexdigest()}
    masks = set()
    for prov in provenance.values():
        rep = REPO_ROOT / prov["directory"] / "report.json"
        if rep.is_file():
            masks.add(json.loads(rep.read_text())["inclusion"]["mask_sha256"])
    if len(masks) == 1:
        result["inclusion_mask_sha256"] = masks.pop()
    elif masks:
        result["inclusion_mask_mismatch"] = sorted(masks)

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
    L = ["# Experiment 7 — FineWeb teacher-KD mixture at the fixed 1.60M rung", "",
         f"Generated {r['created_utc']} from retained generations only. Inclusion",
         f"mask `{r.get('inclusion_mask_sha256', '?')[:16]}…`, 150 prompts, greedy,",
         "unrestricted generation (P18), every arm re-scored with the current",
         "scorer. Arm A is the **retained** E1/P1 KD-heavy 1.60M baseline — the",
         "same generations E6 produced, not a new run.", "",
         "All three arms train the identical 1.60M rollout stream. B and C differ",
         "from A by an added KD-only stream and from each other **only** in that",
         "stream's content: both consume exactly "
         f"{r['extra_kd_positions']['B-FineWeb']:,} extra KD positions.", "",
         "## 1. General-language restoration — DIAGNOSTIC ONLY", "",
         "These may not promote a checkpoint (decision record 2026-08-09). They",
         "answer whether general language modelling came back, and nothing else.", "",
         "| arm | FineWeb NLL | teacher KL | top-1 | mean rank |",
         "| --- | ---: | ---: | ---: | ---: |"]
    gt = (r.get("general_text_diagnostics") or {}).get("arms", {})
    for name in sorted(gt):
        m = gt[name]
        L.append(f"| {name} | {m.get('nll', float('nan')):.4f} | "
                 f"{m.get('kl', float('nan')):.4f} | {m.get('top1', 0):.4f} | "
                 f"{m.get('mean_rank', 0):.1f} |")

    L += ["", "## 2. Autonomous behaviour — THE PROMOTION CRITERION", "",
          "| arm | seed | usable | correct | correct\\|usable | nat. term | ctx limit | repetition | empty |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fam in ("A-Baseline", "B-FineWeb", "C-Control"):
        f = r["families"].get(fam)
        if f is None:
            L.append(f"| {fam} | — | not evaluated | | | | | | |")
            continue
        for alias, m in f["seeds"].items():
            cgu = m["correct_given_usable"]
            L.append(f"| {fam} | {alias.rsplit('-', 1)[-1]} | "
                     f"{m['usable_rollout_rate']:.4f} | {m['correct_overall']:.4f} | "
                     f"{'—' if cgu is None else f'{cgu:.4f}'} | "
                     f"{m['natural_termination_rate']:.4f} | "
                     f"{m['context_limit_rate']:.4f} | "
                     f"{m['severe_repetition_rate']:.4f} | "
                     f"{m['empty_output_rate']:.4f} |")
        L.append(f"| **{fam}** | **mean** | "
                 f"**{f['mean']['usable_rollout_rate']:.4f}** | "
                 f"**{f['mean']['correct_overall']:.4f}** | "
                 f"{f['correct_given_usable_pooled']:.4f} | "
                 f"{f['mean']['natural_termination_rate']:.4f} | "
                 f"{f['mean']['context_limit_rate']:.4f} | "
                 f"{f['mean']['severe_repetition_rate']:.4f} | "
                 f"{f['mean']['empty_output_rate']:.4f} |")

    L += ["", "`usable_rollout` is reported with every component rate, never as a",
          "weighted average. It is blind to correctness by construction, and its",
          "components are not independent — `protocol_valid` subsumes two of them.",
          "", "## 3. Paired comparisons on the shared mask", ""]
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

    L += ["## 4. The three questions, kept separate", "",
          "The preregistration fixed this reading before the run "
          "(`e7_preregistration.md` §7.4):", "",
          "1. **general-language restoration** — section 1, diagnostics only;",
          "2. **autonomous stability** — `usable_rollout` and its components;",
          "3. **autonomous reasoning correctness** — `correct_overall`, "
          "`correct_given_usable`, GSM8K.", "",
          "**If B improves the general-text diagnostics but does not beat C on",
          "autonomous correctness, FineWeb did not solve the reasoning",
          "bottleneck.** A restored NLL is not a restored capability, and this",
          "report must not be summarised as though it were.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
