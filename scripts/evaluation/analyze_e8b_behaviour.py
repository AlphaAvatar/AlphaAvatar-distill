#!/usr/bin/env python
"""E8b level 3: does the depth map only fail once composed with compression?

    PYTHONPATH=src python scripts/evaluation/analyze_e8b_behaviour.py \
        --bootstrap 10000 --out logs/e8b_results.json

Four cells on one frozen 150-prompt battery, every arm re-scored from its retained
raw generations with the current scorer — no stored `usable`, `correct` or
termination field is carried into an E8b number. `rescore_arm` is imported from
`analyze_e6`, so E6, E7 and E8b share one scorer.

    DP  depth-only 3.22B, positional map        A100 SXM 80 GB
    DC  depth-only 3.22B, contribution map      A100 SXM 80 GB
    FP  fully compressed 596M, positional       L40S 48 GB   (RETAINED from E6)
    FC  fully compressed 596M, contribution     L40S 48 GB

Two contrasts and one interaction:

    DC vs DP    the depth map at full width, within one hardware class
    FC vs FP    the depth map fully compressed, within one hardware class
    (FC-FP) - (DC-DP)    whether the map's effect depends on the compression

The interaction sign convention is the **registered** one
(`logs/e8b_preregistration.md` §8): `(FC - FP) - (DC - DP)`. Negative means the map
does worse once compression is applied than it does at full width. Reversing this is
an easy and invisible way to invert a conclusion, so it is asserted in the tests.

**Hardware is nested inside compression regime.** Each contrast is within one
hardware class, so each is clean. The interaction is not hardware-controlled: it
cannot by itself distinguish "the map interacts with compression" from "the map
interacts with the card". The registered conditional bridge exists for that, and
fires only on a material reversal.

The interaction is a difference of paired differences, and it is legitimate to
bootstrap here only because **all four cells answer the same 150 prompts** — the
frozen inclusion mask is identical, so prompts can be resampled jointly. That is
done explicitly rather than by combining two independent intervals.

Step-0 NLL, teacher-forced top-1 and the general-text diagnostics may not promote
anything (decision 2026-08-05 / 2026-08-09). `scripts/training/analyze_e8b.py`
carries those levels; this script is the endpoint.
"""

from __future__ import annotations

import argparse
import json
import random
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
from analyze_e6 import (  # noqa: E402  — one scorer, shared by E6, E7 and E8b
    load_sessions, rescore_arm,
)

AUDIT = REPO_ROOT / "artifacts/audit"
THREE_MODE = AUDIT / "three_mode"
EXPECTED_MASK = "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba"
FLOORS = {"usable_rollout_rate": 0.0800, "correct_overall": 0.0600}
SEEDS = ("sa", "sb")

# cell -> (per-seed three_mode directory stem, regime, depth map, hardware)
CELLS = {
    "DP": ({"sa": "E8b-DP-sa", "sb": "E8b-DP-sb"},
           "depth_only", "positional", "A100_SXM_80GB"),
    "DC": ({"sa": "E8b-DC-sa", "sb": "E8b-DC-sb"},
           "depth_only", "contribution", "A100_SXM_80GB"),
    # FP is retained: the same generations E6 produced, not a new run.
    "FP": ({"sa": "E1-1.60M-sa", "sb": "E1-1.60M-sb"},
           "fully_compressed", "positional", "L40S_48GB"),
    "FC": ({"sa": "E8b-FC-sa", "sb": "E8b-FC-sb"},
           "fully_compressed", "contribution", "L40S_48GB"),
}
CONTRASTS = (("DP", "DC", "depth_only"), ("FP", "FC", "fully_compressed"))
AXES = (("usable", "usable_rollout_rate", "higher"),
        ("correct", "correct_overall", "higher"))
# Reported alongside; `usable_rollout` is blind to correctness by construction, and
# its components are not independent — protocol_valid subsumes two of them.
COMPONENT_METRICS = (
    ("usable_rollout_rate", "higher"), ("protocol_valid_rate", "higher"),
    ("natural_termination_rate", "higher"), ("context_limit_rate", "lower"),
    ("severe_repetition_rate", "lower"), ("empty_output_rate", "lower"),
    ("correct_overall", "higher"), ("correct_given_usable", "higher"),
)


def load_cells(sessions: dict) -> tuple[dict, list[str]]:
    arms, missing = {}, []
    for cell, (stems, *_rest) in CELLS.items():
        for seed, stem in stems.items():
            d = THREE_MODE / stem
            m = rescore_arm(d, sessions) if d.is_dir() else None
            if m is None:
                missing.append(f"{cell}-{seed} ({stem})")
                continue
            report = json.loads((d / "report.json").read_text())
            mask = report["inclusion"]["mask_sha256"]
            if mask != EXPECTED_MASK:
                raise SystemExit(f"{stem}: inclusion mask {mask} is not the "
                                 f"binding {EXPECTED_MASK}")
            arms[f"{cell}-{seed}"] = m
    return arms, missing


def _verdict(pooled: float, directions: set, floor: float, better: str) -> str:
    if abs(pooled) < floor:
        return "inside the floor — a tie"
    if len(directions) != 1:
        return "above the floor but seeds disagree — not claimable"
    improved = (pooled > 0) if better == "higher" else (pooled < 0)
    return f"above the floor and seed-consistent ({'better' if improved else 'worse'})"


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


def contrast(a_cell: str, b_cell: str, arms: dict, iterations: int) -> dict:
    """Both seeds independently, the two-seed direction, and the pooled delta."""
    per_seed = {s: compare(f"{a_cell}-{s}", f"{b_cell}-{s}", arms, iterations)
                for s in SEEDS}
    per_seed = {s: v for s, v in per_seed.items() if v is not None}
    out = {"a": a_cell, "b": b_cell,
           "regime": CELLS[a_cell][1], "hardware": CELLS[a_cell][3],
           "per_seed": per_seed}
    if len(per_seed) != len(SEEDS):
        # Name what is actually absent. `compare` needs both sides, so an empty
        # per_seed does NOT mean both cells are missing — saying "neither" when
        # one side is fully present sends the reader looking in the wrong place.
        absent = [f"{c}-{s}" for c in (a_cell, b_cell) for s in SEEDS
                  if f"{c}-{s}" not in arms]
        out["incomplete"] = (f"needs both seeds of {a_cell} and {b_cell}; missing "
                             f"{absent}. No claim from one seed — behaviour varies "
                             "0.1290 on seed alone.")
        return out
    for axis, floor_key, better in AXES:
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


def interaction(arms: dict, axis: str, iterations: int, seed: int = 20260811) -> dict:
    """`(FC-FP) - (DC-DP)` as registered, bootstrapped over the shared prompts.

    Resampling is joint: one draw of prompt ids is applied to all four arms, which
    is only valid because the frozen inclusion mask is identical across cells. A
    difference of two independently-computed intervals would not be a CI on this
    quantity.
    """
    need = [f"{c}-{s}" for c in CELLS for s in SEEDS]
    if any(a not in arms for a in need):
        return {"incomplete": f"needs all 8 arms; missing "
                              f"{[a for a in need if a not in arms]}"}
    out = {"axis": axis, "per_seed": {}}
    for s in SEEDS:
        cells = {c: arms[f"{c}-{s}"]["per_sample"][axis] for c in CELLS}
        ids = sorted(set.intersection(*(set(v) for v in cells.values())))
        if not ids:
            return {"incomplete": "no shared prompt ids"}

        def dd(sample: list[str]) -> float:
            def rate(c):
                return statistics.fmean(float(bool(cells[c][i])) for i in sample)
            # Registered convention (preregistration §8): compressed effect minus
            # full-width effect. Negative = the map degrades under compression.
            return (rate("FC") - rate("FP")) - (rate("DC") - rate("DP"))

        point = dd(ids)
        rng = random.Random(seed)
        draws = sorted(dd([ids[rng.randrange(len(ids))] for _ in ids])
                       for _ in range(iterations))
        lo = draws[int(0.025 * iterations)]
        hi = draws[min(int(0.975 * iterations), iterations - 1)]
        out["per_seed"][s] = {
            "depth_only_delta": round(
                statistics.fmean(float(bool(cells["DC"][i])) for i in ids)
                - statistics.fmean(float(bool(cells["DP"][i])) for i in ids), 4),
            "fully_compressed_delta": round(
                statistics.fmean(float(bool(cells["FC"][i])) for i in ids)
                - statistics.fmean(float(bool(cells["FP"][i])) for i in ids), 4),
            "interaction": round(point, 4),
            "bootstrap_ci": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "n_prompts": len(ids),
        }
    vals = [out["per_seed"][s]["interaction"] for s in SEEDS]
    out["pooled_interaction"] = round(statistics.fmean(vals), 4)
    out["seed_consistent"] = len({v > 0 for v in vals}) == 1
    out["both_cis_exclude_zero"] = all(
        out["per_seed"][s]["ci_excludes_zero"] for s in SEEDS)
    out["definition"] = "(FC - FP) - (DC - DP), the registered convention"
    out["hardware_caveat"] = (
        "Hardware is NESTED in regime (DP/DC A100, FP/FC L40S). A nonzero "
        "interaction does not separate depth-map x compression from "
        "depth-map x hardware; that is what the conditional bridge is for.")
    return out


def bridge_trigger(results: dict) -> dict:
    """The registered trigger: `DC > DP` while `FC < FP` on recovered behaviour.

    Registered prospectively and evaluated only after the fact. It is defined on
    **recovered behaviour**, not on step-0 NLL — the step-0 reversal does not fire
    it, however striking that reversal is.
    """
    d = results["contrasts"].get("depth_only", {}).get("usable", {})
    f = results["contrasts"].get("fully_compressed", {}).get("usable", {})
    if not d or not f:
        return {"evaluable": False,
                "reason": "needs both contrasts complete on usable_rollout"}
    dd, ff = d["pooled_delta"], f["pooled_delta"]
    material = (abs(dd) >= FLOORS["usable_rollout_rate"]
                and abs(ff) >= FLOORS["usable_rollout_rate"])
    return {
        "evaluable": True,
        "depth_only_pooled_delta": dd, "fully_compressed_pooled_delta": ff,
        "sign_reversal": (dd > 0) != (ff > 0),
        "both_exceed_floor": material,
        "seed_consistent_both": bool(d["seed_consistent"] and f["seed_consistent"]),
        "fires": bool((dd > 0) and (ff < 0) and material
                      and d["seed_consistent"] and f["seed_consistent"]),
        "note": "Fires only on a material, seed-consistent DC>DP with FC<FP. "
                "Not fired by the step-0 NLL reversal.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="logs/e8b_results.json")
    args = ap.parse_args()

    sessions = load_sessions()
    arms, missing = load_cells(sessions)

    by_cell_seed = {a: {k: v for k, v in m.items() if k != "per_sample"}
                    for a, m in arms.items()}
    by_cell = {}
    for cell in CELLS:
        present = [arms[f"{cell}-{s}"] for s in SEEDS if f"{cell}-{s}" in arms]
        if len(present) != len(SEEDS):
            continue
        by_cell[cell] = {
            m: round(statistics.fmean(p[m] for p in present), 4)
            for m, _better in COMPONENT_METRICS if all(m in p for p in present)}
        by_cell[cell].update(
            {"regime": CELLS[cell][1], "depth_map": CELLS[cell][2],
             "hardware": CELLS[cell][3], "n_seeds": len(present)})

    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "experiment": "E8b level 3 — depth map x compression on the frozen battery",
        "inclusion_mask_sha256": EXPECTED_MASK,
        "floors": FLOORS,
        "hardware_nesting": "DP/DC on A100 SXM 80GB, FP/FC on L40S 48GB. Each "
                            "contrast is within one hardware class; the "
                            "interaction is not hardware-controlled.",
        "retained_arms": {"FP": "E1/P1 KD-heavy 1.60M sa+sb, retained from E6 — "
                                "the same generations, not a new run"},
        "missing_arms": missing,
        "by_cell": by_cell,
        "by_cell_seed": by_cell_seed,
        "contrasts": {label: contrast(a, b, arms, args.bootstrap)
                      for a, b, label in CONTRASTS},
    }
    results["interaction"] = {
        axis: interaction(arms, axis, args.bootstrap) for axis in ("usable", "correct")}
    results["conditional_bridge"] = bridge_trigger(results)

    out = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    if missing:
        print(f"MISSING ARMS ({len(missing)}): {missing}")
    for label, c in results["contrasts"].items():
        print(f"\n=== {c['b']} vs {c['a']} ({label}, {c['hardware']}) ===")
        if "incomplete" in c:
            print("  " + c["incomplete"])
            continue
        for axis, _f, _b in AXES:
            r = c[axis]
            print(f"  {axis:8s} pooled {r['pooled_delta']:+.4f}  "
                  f"per-seed {r['delta_per_seed']}  {r['verdict']}")
    for axis, i in results["interaction"].items():
        print(f"\n=== interaction on {axis} ===")
        print("  " + i["incomplete"] if "incomplete" in i else
              f"  pooled {i['pooled_interaction']:+.4f}  "
              f"seed-consistent {i['seed_consistent']}  "
              f"CIs exclude zero {i['both_cis_exclude_zero']}")
    b = results["conditional_bridge"]
    print(f"\nconditional bridge fires: {b.get('fires', 'not evaluable')}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
