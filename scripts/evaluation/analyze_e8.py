#!/usr/bin/env python
"""Experiment 8 analysis: does a contribution-guided depth map reach behaviour?

    PYTHONPATH=src python scripts/evaluation/analyze_e8.py --bootstrap 10000

Two families, one frozen 150-prompt battery, both re-scored from retained raw
generations with the current scorer:

    control     E1/P1 KD-heavy 2.96M from the positional depth init — RETAINED,
                not retrained, the same generations E6/E6b produced
    treatment   the identical recipe from the contribution-guided depth init

One comparison, `treatment − control`, on the floors E6/E6b/E7 registered:
`usable_rollout_rate` 0.0800 and `correct_overall` 0.0600, each requiring the
same sign on both seeds.

**The initialization diagnostics are reported and may decide nothing.** They are
here so the reading of E8 can distinguish four preregistered outcomes (§12 of
`e8_preregistration.md`) — in particular "the map reconstructs the teacher better
and it does not survive recovery" from "initialization NLL is worse and behaviour
improves anyway", which is the result that would matter most.

Nothing is generated here. Given the artifacts, the report is reproducible
without a GPU.
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
    mcnemar_counts, paired_bootstrap_ci,
)
from aadistill.infrastructure.env import code_state  # noqa: E402
from analyze_e6 import (  # noqa: E402  — one scorer, shared by every experiment
    arm_alias, load_sessions, rescore_arm, token_stream_sha256,
)

AUDIT = REPO_ROOT / "artifacts/audit"
THREE_MODE = AUDIT / "three_mode"
REGISTRATION = REPO_ROOT / "logs/archive/e8_preregistration.md"
GENERAL_TEXT = AUDIT / "e8_general_text"
STEP0 = AUDIT / "e8_step0_comparison.json"
FROZEN_MAP = REPO_ROOT / "artifacts/stage1/e8_depth_search/e8_frozen_depth_map.json"

FLOORS = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600,
          "correct_given_usable": 0.0600}
SEEDS = ("sa", "sb")
UNIQUE_CE = 2_960_507
CUMULATIVE_CE = 8_881_521

# family -> three_mode directory stem per seed
FAMILIES = {
    "Control-Positional": {"sa": "E1-2.96M-sa", "sb": "E1-2.96M-sb"},
    "Treatment-Contribution": {"sa": "E8-T-Contrib-sa", "sb": "E8-T-Contrib-sb"},
}
COMPONENTS = ("natural_termination_rate", "context_limit_rate",
              "severe_repetition_rate", "empty_output_rate",
              "answer_parse_failure_rate_numeric")


def family_metric(arms: dict, family: str, metric: str, seed: str):
    a = arms.get(f"{family}-{seed}")
    return None if a is None else a.get(metric)


def family_mean(arms: dict, family: str, metric: str):
    vals = [family_metric(arms, family, metric, s) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _verdict(pooled: float, directions: set, floor: float, better: str) -> str:
    if abs(pooled) < floor:
        return "inside the floor — a tie"
    if len(directions) != 1:
        return "above the floor but seeds disagree — not claimable"
    improved = (pooled > 0) if better == "higher" else (pooled < 0)
    return f"above the floor and seed-consistent ({'better' if improved else 'worse'})"


def compare(a_alias: str, b_alias: str, arms: dict, iterations: int) -> dict | None:
    """b against a, paired per prompt on the shared 150-prompt mask."""
    a, b = arms.get(a_alias), arms.get(b_alias)
    if a is None or b is None:
        return None
    out = {"a": a_alias, "b": b_alias}
    for axis in ("usable", "correct", "natural_termination", "context_limit"):
        pa, pb = a["per_sample"][axis], b["per_sample"][axis]
        counts = mcnemar_counts(pa, pb)
        row = {"a_rate": counts["rate_a"], "b_rate": counts["rate_b"],
               "delta": counts["delta"], "win": counts["b_gained"],
               "loss": counts["b_lost"],
               "tie": counts["both_true"] + counts["both_false"],
               "n_paired": counts["n_paired"]}
        if axis in ("usable", "correct"):
            ci = paired_bootstrap_ci(pa, pb, iterations=iterations)
            row["ci95"] = [ci["ci_low"], ci["ci_high"]]
            row["excludes_zero"] = ci["ci_excludes_zero"]
        out[axis] = row
    return out


def family_compare(fa: str, fb: str, arms: dict, iterations: int) -> dict:
    """The registered comparison: pooled delta, per-seed sign, floor verdict."""
    per_seed = {s: compare(f"{fa}-{s}", f"{fb}-{s}", arms, iterations) for s in SEEDS}
    out = {"a_family": fa, "b_family": fb,
           "per_seed": {s: v for s, v in per_seed.items() if v}}
    for axis, floor_key, better in (("usable", "usable_rollout_rate", "higher"),
                                    ("correct", "correct_overall", "higher")):
        deltas = {s: v[axis]["delta"] for s, v in per_seed.items() if v}
        if not deltas:
            out[axis] = {"not_evaluable": "no seed pair available"}
            continue
        pooled = sum(deltas.values()) / len(deltas)
        directions = {d > 0 for d in deltas.values() if d != 0}
        out[axis] = {
            "pooled_delta": round(pooled, 4), "per_seed": deltas,
            "floor": FLOORS[floor_key],
            "exceeds_floor": abs(pooled) >= FLOORS[floor_key],
            "seed_consistent": len(directions) <= 1,
            "verdict": _verdict(pooled, directions, FLOORS[floor_key], better),
        }
    return out


def step0_diagnostics() -> dict:
    """Initialization-time numbers. Diagnostic only, and labelled as such."""
    out = {"metric_status": "DIAGNOSTIC ONLY — may neither promote nor cancel",
           "present": STEP0.is_file()}
    if STEP0.is_file():
        out.update(json.loads(STEP0.read_text()))
    if FROZEN_MAP.is_file():
        f = json.loads(FROZEN_MAP.read_text())
        out["frozen_map"] = {
            "kept_teacher_layers": f["kept_teacher_layers"],
            "removed_teacher_layers": f["removed_teacher_layers"],
            "removal_order": f["removal_order"],
            "positional_removed": f["positional_removed"],
            "primary_kl": f["primary_kl"],
            "positional_baseline_primary_kl": f["positional_baseline_primary_kl"],
            "lower_kl_than_positional": f["lower_kl_than_positional"],
            "per_domain_kl": f["per_domain_kl"],
            "depth_map_sha256": f["depth_map_sha256"],
            "search_report_sha256": f["search_report_sha256"],
        }
    return out


def general_text(reports: Path = GENERAL_TEXT) -> dict:
    out = {}
    if not reports.is_dir():
        return out
    for p in sorted(reports.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out[p.stem] = {k: v for k, v in (r.get("metrics") or {}).items()
                       if k in ("nll", "kl", "top1", "mean_rank", "ppl")}
    return out


def outcome_reading(behaviour: dict, step0: dict, missing: list) -> dict:
    """Map the result onto the four preregistered outcomes (§12), mechanically.

    A missing arm is **not** a tie. Reading "behaviour did not move" off an
    absent cell would turn an incomplete run into a scientific conclusion, which
    is the one reading this function must never produce.
    """
    if missing:
        return {"not_evaluable": missing,
                "reading": "NOT EVALUABLE — an arm is missing, which is not a tie",
                "axes_that_moved": [], "behaviour_improved": None,
                "behaviour_regressed": None, "stability_only": None,
                "calibration_kl_lower_than_positional":
                    (step0.get("frozen_map") or {}).get(
                        "lower_kl_than_positional"),
                "caveat": "no comparison was computed"}
    usable = behaviour.get("usable", {})
    correct = behaviour.get("correct", {})
    moved = []
    for axis, entry in (("usable", usable), ("correct", correct)):
        if entry.get("exceeds_floor") and entry.get("seed_consistent"):
            moved.append((axis, entry["pooled_delta"] > 0))
    init_better = None
    fm = step0.get("frozen_map") or {}
    if "lower_kl_than_positional" in fm:
        init_better = bool(fm["lower_kl_than_positional"])
    behaviour_improved = any(sign for _, sign in moved)
    behaviour_regressed = any(not sign for _, sign in moved)
    if behaviour_improved and init_better:
        reading = ("position-based depth compression was discarding important "
                   "teacher computation")
    elif behaviour_improved and init_better is False:
        reading = ("initialization calibration KL is worse and behaviour improved "
                   "anyway — contribution-aware structure preserves something the "
                   "calibration objective does not rank first")
    elif behaviour_regressed:
        reading = "reject the contribution-guided map"
    else:
        reading = ("the new map changed the initialization and did not survive "
                   "recovery — initialization structure is not the bottleneck at "
                   "this rung")
    stability_only = (usable.get("exceeds_floor") and usable.get("seed_consistent")
                      and not (correct.get("exceeds_floor")
                               and correct.get("seed_consistent")))
    return {
        "axes_that_moved": [a for a, _ in moved],
        "behaviour_improved": behaviour_improved,
        "behaviour_regressed": behaviour_regressed,
        "calibration_kl_lower_than_positional": init_better,
        "stability_only": bool(stability_only),
        "reading": reading,
        "caveat": ("`usable_rollout` is blind to correctness by construction and "
                   "its components are not independent; a stability-only move is "
                   "not progress on reasoning"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "logs/e8_results.json")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "logs/e8_report.md")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    sessions = load_sessions()
    arms, missing = {}, []
    for family, stems in FAMILIES.items():
        for seed, stem in stems.items():
            d = THREE_MODE / stem
            if not (d / "report.json").is_file():
                missing.append(f"{family}-{seed} ({stem})")
                continue
            arms[f"{family}-{seed}"] = rescore_arm(d, sessions)

    masks = set()
    for family, stems in FAMILIES.items():
        for stem in stems.values():
            rep = THREE_MODE / stem / "report.json"
            if rep.is_file():
                masks.add(json.loads(rep.read_text())["inclusion"]["mask_sha256"])
    behaviour = family_compare("Control-Positional", "Treatment-Contribution",
                               arms, args.bootstrap)
    step0 = step0_diagnostics()
    result = {
        "experiment": "E8 — contribution-guided depth initialization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "registration": str(REGISTRATION.relative_to(REPO_ROOT)),
        "floors": FLOORS,
        "unique_ce_tokens": UNIQUE_CE,
        "cumulative_ce_exposure": CUMULATIVE_CE,
        "families": {f: {"stems": s} for f, s in FAMILIES.items()},
        "missing_arms": missing,
        "inclusion_masks": sorted(masks),
        "inclusion_mask_shared": len(masks) == 1,
        "arms": {a: {k: v for k, v in m.items() if k != "per_sample"}
                 for a, m in arms.items()},
        "family_means": {
            f: {m: family_mean(arms, f, m) for m in
                ("usable_rollout_rate", "correct_overall", "correct_given_usable",
                 *COMPONENTS)}
            for f in FAMILIES},
        "behaviour_comparison": behaviour,
        "step0_diagnostics": step0,
        "general_text_diagnostics": general_text(),
        "outcome": outcome_reading(behaviour, step0, missing),
        "code_state": code_state(str(REPO_ROOT)),
    }
    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.report.write_text(render(result))
    print(f"-> {rel(args.out)}")
    print(f"-> {rel(args.report)}")
    if missing:
        print(f"  MISSING ARMS: {missing}")
    print(f"\nusable  {behaviour['usable']}")
    print(f"correct {behaviour['correct']}")
    print(f"reading: {result['outcome']['reading']}")


def render(r: dict) -> str:
    L = [f"# Experiment 8 — contribution-guided depth initialization", ""]
    L.append(f"Generated {r['created_utc']} from retained generations only. "
             f"150 prompts, greedy, unrestricted generation (P18), every arm "
             f"re-scored with the current scorer. The control is the **retained** "
             f"E1/P1 KD-heavy 2.96M baseline — the same generations E6/E6b "
             f"produced, not a new run.")
    L.append("")
    L.append(f"Both families train the identical 2.96M rollout stream: "
             f"{r['unique_ce_tokens']:,} unique CE tokens, "
             f"{r['cumulative_ce_exposure']:,} cumulative. The only intended "
             f"causal variable is the Stage 1 depth map.")
    L.append("")

    fm = (r["step0_diagnostics"].get("frozen_map") or {})
    L.append("## 1. The depth map — the causal variable")
    L.append("")
    if fm:
        L.append(f"* contribution-guided removes `{fm['removed_teacher_layers']}`")
        L.append(f"* positional removes `{fm['positional_removed']}`")
        L.append(f"* removal order `{fm['removal_order']}`")
        L.append(f"* calibration KL {fm['primary_kl']:.6f} vs positional "
                 f"{fm['positional_baseline_primary_kl']:.6f} "
                 f"({'lower' if fm['lower_kl_than_positional'] else 'higher'})")
        L.append(f"* per-domain KL: " + ", ".join(
            f"{k} {v:.6f}" for k, v in sorted(fm["per_domain_kl"].items())))
        L.append(f"* map sha256 `{fm['depth_map_sha256'][:16]}…`")
    else:
        L.append("*The frozen map artifact is not present.*")
    L.append("")

    L.append("## 2. Step-0 initialization diagnostics — DIAGNOSTIC ONLY")
    L.append("")
    L.append("These may neither promote nor cancel an arm. They exist so that what "
             "the initialization changed is known before recovery obscures it.")
    L.append("")
    summaries = (r["step0_diagnostics"].get("summaries") or {})
    if summaries:
        keys = sorted({k for v in summaries.values() for k in v if "." in k})
        L.append("| checkpoint | " + " | ".join(keys) + " |")
        L.append("| --- | " + " | ".join("---:" for _ in keys) + " |")
        for label, row in sorted(summaries.items()):
            L.append(f"| {label} | " + " | ".join(
                f"{row.get(k):.4f}" if isinstance(row.get(k), float)
                else str(row.get(k, "—")) for k in keys) + " |")
    else:
        L.append("*No step-0 comparison artifact found.*")
    L.append("")

    L.append("## 3. Autonomous behaviour — THE PROMOTION CRITERION")
    L.append("")
    cols = ("usable_rollout_rate", "correct_overall", "correct_given_usable",
            *COMPONENTS)
    L.append("| arm | seed | " + " | ".join(c.replace("_rate", "") for c in cols) + " |")
    L.append("| --- | --- | " + " | ".join("---:" for _ in cols) + " |")
    for family in FAMILIES:
        for seed in SEEDS:
            a = r["arms"].get(f"{family}-{seed}")
            if not a:
                continue
            L.append(f"| {family} | {seed} | " + " | ".join(
                (f"{a[c]:.4f}" if isinstance(a.get(c), (int, float)) else "—")
                for c in cols) + " |")
        mean = r["family_means"][family]
        L.append(f"| **{family}** | **mean** | " + " | ".join(
            f"**{mean[c]:.4f}**" if mean.get(c) is not None else "—"
            for c in cols) + " |")
    L.append("")
    L.append("`usable_rollout` is reported with every component rate, never as a "
             "weighted average. It is blind to correctness by construction, and "
             "its components are not independent — `protocol_valid` subsumes two "
             "of them.")
    L.append("")

    L.append("## 4. The registered comparison")
    L.append("")
    b = r["behaviour_comparison"]
    for axis in ("usable", "correct"):
        a = b.get(axis) or {}
        if "not_evaluable" in a:
            L.append(f"* **{axis}** — not evaluable: {a['not_evaluable']}")
            continue
        L.append(f"* **{axis}** pooled Δ {a['pooled_delta']:+.4f} "
                 f"(floor {a['floor']:.4f}) — {a['verdict']}; seeds "
                 f"{a['per_seed']}, seed-consistent {a['seed_consistent']}")
        for seed, v in (b.get("per_seed") or {}).items():
            e = v[axis]
            L.append(f"  * {seed}: {e['a_rate']:.4f} → {e['b_rate']:.4f}, "
                     f"win/tie/loss {e['win']}/{e['tie']}/{e['loss']}, "
                     f"95% CI [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
                     + (" excludes 0" if e["excludes_zero"] else ""))
    L.append("")

    gt = r.get("general_text_diagnostics") or {}
    if gt:
        L.append("## 5. General-text diagnostics on the trained arms — DIAGNOSTIC ONLY")
        L.append("")
        L.append("| arm | NLL | teacher KL | top-1 | mean rank |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for name, m in sorted(gt.items()):
            L.append(f"| {name} | {m.get('nll', float('nan')):.4f} | "
                     f"{m.get('kl', float('nan')):.4f} | "
                     f"{m.get('top1', float('nan')):.4f} | "
                     f"{m.get('mean_rank', float('nan')):.1f} |")
        L.append("")

    o = r["outcome"]
    L.append("## 6. The preregistered reading")
    L.append("")
    L.append(f"**{o['reading']}.**")
    L.append("")
    if o.get("not_evaluable"):
        L.append(f"Missing arms: `{o['not_evaluable']}`. An absent arm is not a "
                 f"tie, and no outcome is claimed.")
        L.append("")
        return "\n".join(L)
    L.append(f"* axes that moved above their floor with both seeds agreeing: "
             f"`{o['axes_that_moved'] or 'none'}`")
    L.append(f"* calibration KL lower than the positional map: "
             f"`{o['calibration_kl_lower_than_positional']}`")
    L.append(f"* stability-only move: `{o['stability_only']}`")
    L.append("")
    L.append(o["caveat"] + ".")
    L.append("")
    if r["missing_arms"]:
        L.append(f"**Missing arms:** {r['missing_arms']}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
