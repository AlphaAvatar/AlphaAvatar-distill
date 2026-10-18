#!/usr/bin/env python3
"""Root-level structural selection agreement (the pilot's H1/H3 evidence), at $0.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/analyze_structure.py \
        --structure <pilot structure.json> --projections-dir <dir> \
        --profiles-doc configs/stages/stage-0/calibration_volume/profiles.json \
        --reference-structure <D1 structure.json> --out <json>

Implements `logs/stages/stage-0/calibration_volume/plans/pilot_analysis_plan.json`
(frozen before any result) and nothing else:

* agreement per operator -- FFN `kept_neurons` and ATTENTION `kept_heads` as
  per-layer Jaccard averaged over layers; DEPTH as the Jaccard of the removed
  sets and the fraction of greedy rounds with the same choice; RESIDUAL_WIDTH as
  the subspace overlap ||Q1^T Q2||_F^2 / k of the orthonormalized projections --
  each beside its chance level;
* groupings -- between resamples at each dose; each resample's 1x and 2x
  against its own (nested) 4x; and a DESCRIPTIVE contrast with the historical
  root decisions of the D1 profiles;
* the plan's noise-calibrated reading rule.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

CHANCE_N = {"FFN": 9728, "ATTENTION": 32, "DEPTH": 36}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def chance_jaccard(k: int, n: int) -> float:
    e = k * k / n
    return e / (2 * k - e)


def depth_removed(decision: dict) -> tuple[list[int], list[int]]:
    """(removed set, greedy order). `chosen` is the ORIGINAL layer index: D2's
    rounds reproduce its recorded removal order exactly."""
    order = [int(r["chosen"]) for r in decision.get("search_rounds") or []]
    return sorted(order), order


def per_layer(a, b) -> float:
    return sum(jaccard(x, y) for x, y in zip(a, b)) / max(1, min(len(a), len(b)))


def load_roots(path: Path) -> dict[str, dict[str, dict]]:
    """kind -> profile -> root decision record."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in json.loads(path.read_text()):
        if r.get("depth") != 1:
            continue
        out[(r.get("applied_kinds") or ["?"])[0]][(r.get("profiles") or ["?"])[0]] = r
    return out


def agreement(kind: str, da: dict, db: dict, proj: dict, pa: str, pb: str) -> dict:
    if kind == "FFN":
        ka, kb = da["kept_neurons"], db["kept_neurons"]
        return {"value": round(per_layer(ka, kb), 4),
                "chance": round(chance_jaccard(len(ka[0]), CHANCE_N["FFN"]), 4)}
    if kind == "ATTENTION":
        ka, kb = da["kept_heads"], db["kept_heads"]
        return {"value": round(per_layer(ka, kb), 4),
                "chance": round(chance_jaccard(len(ka[0]), CHANCE_N["ATTENTION"]), 4)}
    if kind == "DEPTH":
        ra, oa = depth_removed(da)
        rb, ob = depth_removed(db)
        return {"value": round(jaccard(ra, rb), 4),
                "same_greedy_choice": round(sum(x == y for x, y in zip(oa, ob)) / max(1, len(oa)), 4),
                "chance": round(chance_jaccard(len(ra), CHANCE_N["DEPTH"]), 4),
                "removed_a": ra, "removed_b": rb}
    if kind == "RESIDUAL_WIDTH":
        if pa in proj and pb in proj:
            qa, qb = proj[pa], proj[pb]
            k = qa.shape[1]
            return {"value": round(float((qa.T @ qb).pow(2).sum() / k), 4),
                    "chance": round(k / qa.shape[0], 4)}
        return {"value": None, "note": "projection not recovered"}
    return {"value": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--projections-dir", default=None)
    ap.add_argument("--profiles-doc", required=True)
    ap.add_argument("--reference-structure", default=None)
    ap.add_argument("--reference-profiles", nargs="*",
                    default=["calib.domain_balanced@v1", "calib.reasoning_heavy@v2"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    doc = json.loads(Path(args.profiles_doc).read_text())
    meta = {f"{p['profile_id']}@v{p['version']}": p["metadata"] for p in doc["profiles"]}
    resample_of = {}
    for t, tv in doc["treatments"].items():
        if t.startswith("d") and "_r" in t:
            for pid in tv["profiles"]:
                resample_of[pid] = int(t.split("_r")[1])
    roots = load_roots(Path(args.structure))
    refs = load_roots(Path(args.reference_structure)) if args.reference_structure else {}

    proj = {}
    if args.projections_dir:
        import torch
        from safetensors.torch import load_file
        for kind_rows in (roots.get("RESIDUAL_WIDTH", {}),):
            for pid, r in kind_rows.items():
                f = Path(args.projections_dir) / f"width_projection_{r['state_id']}.safetensors"
                if f.is_file():
                    q, _ = torch.linalg.qr(load_file(str(f))["P"].double())
                    proj[pid] = q

    out = {"schema": "aadistill.stage0.structure_agreement/v2",
           "plan": "logs/stages/stage-0/calibration_volume/plans/pilot_analysis_plan.json",
           "structure": args.structure, "covariates": {
               pid: {k: meta[pid].get(k) for k in ("dose", "uesp", "tokens", "n_rule_families",
                                                     "n_units")} | {"resample": resample_of.get(pid)}
               for pid in sorted(meta) if pid in resample_of},
           "by_operator": {}}
    for kind, profs in sorted(roots.items()):
        pilot = [p for p in profs if p in resample_of]
        dec = {p: profs[p].get("decision") or {} for p in pilot}
        by_dose = defaultdict(list)
        for p in pilot:
            by_dose[meta[p]["dose"]].append(p)
        between = {}
        for d, ps in sorted(by_dose.items()):
            pairs = [{"a": a, "b": b, **agreement(kind, dec[a], dec[b], proj, a, b)}
                     for a, b in itertools.combinations(sorted(ps), 2)]
            vals = [x["value"] for x in pairs if x["value"] is not None]
            between[str(d)] = {"pairs": pairs, "n_pairs": len(vals),
                               "mean": round(sum(vals) / len(vals), 4) if vals else None,
                               "min": min(vals) if vals else None,
                               "max": max(vals) if vals else None,
                               "noise": round(max(vals) - min(vals), 4) if vals else None}
        nested = []
        top = max(by_dose) if by_dose else None
        for p in pilot:
            if meta[p]["dose"] == top:
                continue
            twin = [q for q in by_dose.get(top, []) if resample_of[q] == resample_of[p]]
            if twin:
                nested.append({"profile": p, "dose": meta[p]["dose"], "resample": resample_of[p],
                               "vs": twin[0], **agreement(kind, dec[p], dec[twin[0]], proj, p, twin[0])})
        reference = []
        for rp in args.reference_profiles:
            rdec = (refs.get(kind, {}).get(rp) or {}).get("decision")
            if not rdec:
                continue
            for p in pilot:
                reference.append({"reference": rp, "profile": p, "dose": meta[p]["dose"],
                                  **agreement(kind, dec[p], rdec, proj, p, rp)})
        lo, hi = between.get("1", {}), between.get(str(top), {})
        label = None
        if kind != "DEPTH" and lo.get("n_pairs") and hi.get("n_pairs"):
            noise = max(lo["noise"], hi["noise"])
            effect = hi["mean"] - lo["mean"]
            mid = between.get("2", {}).get("mean")
            between_ok = mid is not None and min(lo["mean"], hi["mean"]) <= mid <= max(lo["mean"], hi["mean"])
            label = ("stabilizes with volume" if effect > noise and between_ok else
                     "destabilizes" if effect < -noise else
                     "no detectable volume effect" if abs(effect) <= noise else
                     "effect exceeds noise but non-monotone")
            reading = {"effect_4x_minus_1x": round(effect, 4), "noise": round(noise, 4)}
        else:
            reading = None
        # the plan's saturation reading is DESCRIBED, never labeled. A nested
        # profile shares its units with its own 4x, so its agreement with that
        # 4x is an upper bound on independent agreement, not a like-for-like one.
        nested_by_dose = defaultdict(list)
        for x in nested:
            if x["value"] is not None:
                nested_by_dose[str(x["dose"])].append(x["value"])
        saturation = {"nested_mean_by_dose": {d: round(sum(v) / len(v), 4)
                                              for d, v in sorted(nested_by_dose.items())},
                      "between_resamples_mean_at_largest": hi.get("mean"),
                      "caveat": "nested profiles share units with their own largest dose"}
        out["by_operator"][kind] = {"between_resamples": between, "nested_to_largest": nested,
                                    "reference_contrast_descriptive": reference,
                                    "saturation_description": saturation,
                                    "reading": reading, "label": label}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {"between": {d: (v["mean"], v["noise"]) for d, v in o["between_resamples"].items()},
                          "label": o["label"]} for k, o in out["by_operator"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
