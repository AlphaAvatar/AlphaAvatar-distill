#!/usr/bin/env python3
"""E8b's three evidence levels, kept separate because they disagree.

    PYTHONPATH=src python scripts/training/analyze_e8b.py --level step0
    PYTHONPATH=src python scripts/training/analyze_e8b.py --level behaviour \
        --results logs/e8b_results.json

E8b asks one question — does contribution-guided depth selection beat positional
selection — and answers it at three levels that must never be pooled:

1. **E8a full-width KL.** Frozen teacher vs teacher-with-8-blocks-bypassed, on the
   frozen calibration mixture. No student, no training, no width or FFN compression.
2. **Actual step-0 NLL.** The four initializations reloaded through one canonical
   `from_pretrained` path on one device, measured on three held-out series.
3. **Matched 1.60M recovery.** The registered endpoint: `usable_rollout` and
   correctness on the frozen 150-prompt battery after identical training.

Level 2 already contradicts the naive reading of level 1: the contribution map is
better at full width and worse fully compressed, so a single number for "is the map
better" does not exist. Only level 3 can decide the recipe, and its hardware is
NESTED inside the compression regime (DP/DC on A100, FP/FC on L40S), so a
depth-map effect is within-hardware while the interaction is not.

The `DC-DP` and `FC-FP` contrasts are each computed inside one regime and one
hardware class. The interaction is reported in the **registered** direction,
`(FC-FP) - (DC-DP)` (`logs/e8b_preregistration.md` §8) — negative means the map does
worse once compression is applied than at full width. It carries the nesting and
cannot on its own exclude a hardware x depth-map interaction.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CELLS = ("DP", "DC", "FP", "FC")
REGIME = {"DP": "depth_only", "DC": "depth_only",
          "FP": "fully_compressed", "FC": "fully_compressed"}
DEPTH_MAP = {"DP": "positional", "DC": "contribution",
             "FP": "positional", "FC": "contribution"}
HARDWARE = {"DP": "A100_SXM_80GB", "DC": "A100_SXM_80GB",
            "FP": "L40S_48GB", "FC": "L40S_48GB"}

# (series, key, lower_is_better)
STEP0_METRICS = (
    ("holdout_v1", "nll", True),
    ("fineweb_val_e7", "nll", True),
    ("fineweb_val_e7", "top1", False),
    ("fineweb_val_e7", "mean_rank", True),
    ("fineweb_val_e7", "kl", True),
    ("teacher_native_val", "nll", True),
    ("teacher_native_val", "top1", False),
    ("teacher_native_val", "mean_rank", True),
    ("teacher_native_val", "kl", True),
)
# Behaviour is primary; correctness is a separate secondary axis (Stage 2/3
# hierarchy). `usable_rollout` is blind to correctness by construction.
BEHAVIOUR_METRICS = (
    ("usable_rollout_rate", False),
    ("non_empty", False),
    ("natural_termination", False),
    ("no_severe_repetition", False),
    ("no_context_limit", False),
    ("protocol_valid", False),
    ("correct_overall", False),
    ("correct_given_usable", False),
)

E8A_KL = {"positional": 1.932531, "contribution": 0.620586}


def load_step0(records: Path) -> dict:
    out = {}
    for cell in CELLS:
        p = records / f"{cell}_init_nll.json"
        if not p.is_file():
            raise SystemExit(f"missing step-0 record: {p}")
        out[cell] = json.loads(p.read_text())
    return out


def one_evaluator(recs: dict) -> dict:
    """The step-0 table is only a table if one evaluator produced every row."""
    def env(c):
        r = recs[c]
        return {"device": r["device"], "dtype": r["dtype"],
                "environment": r["environment"],
                "sources": {k: {kk: vv for kk, vv in v.items() if "sha" in kk}
                            for k, v in r["sources"].items()}}
    envs = {c: json.dumps(env(c), sort_keys=True) for c in CELLS}
    hashes = {c: recs[c]["checkpoint"]["model_sha256"] for c in CELLS}
    per_series = {
        c: all(m.get("measured_checkpoint_sha256") == hashes[c]
               for m in recs[c]["measurements"].values()) for c in CELLS}
    return {
        "identical_evaluator": len(set(envs.values())) == 1,
        "evaluator": json.loads(envs["DP"]),
        "distinct_checkpoints": len(set(hashes.values())) == 4,
        "checkpoint_sha256": hashes,
        "every_series_bound_to_its_own_checkpoint": all(per_series.values()),
        "rope_base": {c: recs[c]["checkpoint"]["resolved_rope_base"] for c in CELLS},
        "geometry_matched": {
            "DP_DC": recs["DP"]["checkpoint"]["config_sha256"]
            == recs["DC"]["checkpoint"]["config_sha256"],
            "FP_FC": recs["FP"]["checkpoint"]["config_sha256"]
            == recs["FC"]["checkpoint"]["config_sha256"]},
    }


def contrasts(value, metrics) -> dict:
    """DC-DP and FC-FP, each inside one regime, plus the nested interaction."""
    rows = []
    for name, *rest in metrics:
        if len(rest) == 2:
            series, lower = name, rest[1]
            key, label = rest[0], f"{name}.{rest[0]}"
        else:
            lower, label = rest[0], name
            series = key = None
        try:
            dp, dc, fp, fc = (value(c, series, key) if series
                              else value(c, None, name) for c in CELLS)
        except (KeyError, TypeError):
            continue
        if any(v is None for v in (dp, dc, fp, fc)):
            continue
        depth, full = dc - dp, fc - fp
        better = (lambda d: d < 0) if lower else (lambda d: d > 0)
        rows.append({
            "metric": label, "lower_is_better": lower,
            "DP": dp, "DC": dc, "FP": fp, "FC": fc,
            "DC_minus_DP": depth, "FC_minus_FP": full,
            # Registered direction: compressed effect minus full-width effect.
            "interaction_nested": full - depth,
            "contribution_better_depth_only": better(depth),
            "contribution_better_fully_compressed": better(full),
            "sign_reversal": better(depth) != better(full),
        })
    return {"rows": rows,
            "reversal_count": sum(r["sign_reversal"] for r in rows),
            "n_metrics": len(rows),
            "unanimous_reversal": bool(rows) and all(r["sign_reversal"]
                                                    for r in rows)}


def render(rows) -> str:
    w = max((len(r["metric"]) for r in rows), default=8)
    head = (f"{'metric':{w}s} {'DP':>11s} {'DC':>11s} {'DC-DP':>10s}  "
            f"{'FP':>11s} {'FC':>11s} {'FC-FP':>10s}  rev")
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(f"{r['metric']:{w}s} {r['DP']:11.4f} {r['DC']:11.4f} "
                   f"{r['DC_minus_DP']:+10.4f}  {r['FP']:11.4f} {r['FC']:11.4f} "
                   f"{r['FC_minus_FP']:+10.4f}  {'YES' if r['sign_reversal'] else '.'}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=("step0", "behaviour", "both"),
                    default="step0")
    ap.add_argument("--records", default="logs/e8b_step0_records")
    ap.add_argument("--results", default="logs/e8b_results.json",
                    help="per-seed recovered behaviour, written by the sessions")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else REPO_ROOT / p

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "design": {"regime": REGIME, "depth_map": DEPTH_MAP,
                   "hardware": HARDWARE,
                   "hardware_nesting": "HARDWARE IS NESTED INSIDE REGIME. Each "
                   "depth-map contrast is within one hardware class; the "
                   "interaction is not hardware-controlled."},
        "level_1_e8a_full_width_kl": {
            "teacher_vs_bypassed_kl": E8A_KL,
            "ratio_positional_over_contribution": E8A_KL["positional"]
            / E8A_KL["contribution"],
            "scope": "frozen teacher, 8 blocks bypassed, no student and no width "
                     "or FFN compression — it bounds nothing about a compressed "
                     "student",
        },
    }

    if args.level in ("step0", "both"):
        recs = load_step0(resolve(args.records))
        report["level_2_validity"] = one_evaluator(recs)

        def value(cell, series, key):
            return recs[cell]["measurements"][series][key]
        report["level_2_step0"] = contrasts(value, STEP0_METRICS)
        print("=== level 2: actual step-0 NLL, one evaluator, one reload path ===")
        print(render(report["level_2_step0"]["rows"]))
        v = report["level_2_validity"]
        print(f"\none evaluator: {v['identical_evaluator']}  "
              f"4 distinct checkpoints: {v['distinct_checkpoints']}  "
              f"each series bound to its checkpoint: "
              f"{v['every_series_bound_to_its_own_checkpoint']}")
        s = report["level_2_step0"]
        print(f"sign reversal on {s['reversal_count']}/{s['n_metrics']} metrics "
              f"(unanimous: {s['unanimous_reversal']})")

    if args.level in ("behaviour", "both"):
        p = resolve(args.results)
        if not p.is_file():
            print(f"\n(level 3 not available yet: {p} absent)")
        else:
            res = json.loads(p.read_text())

            def value(cell, _series, key):
                return res["by_cell"][cell][key]
            report["level_3_behaviour"] = contrasts(
                value, [(k, low) for k, low in BEHAVIOUR_METRICS])
            report["level_3_per_seed"] = res.get("by_cell_seed")
            print("\n=== level 3: recovered 1.60M behaviour, frozen battery ===")
            print(render(report["level_3_behaviour"]["rows"]))

    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
