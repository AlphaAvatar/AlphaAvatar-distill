#!/usr/bin/env python3
"""Structural selection agreement between calibration profiles (H1), at $0.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/analyze_structure.py \
        --structure <structure.json> --projections-dir <dir> --groups <plan.json> --out <json>

Reads the ROOT-level decisions of a search (each operator applied to the
teacher under each profile, `structure.json` from `search_driver.py`) and asks,
operator by operator, how much two profiles' decisions agree:

* FFN `kept_neurons` and ATTENTION `kept_heads`: per-layer Jaccard, averaged
  over layers;
* DEPTH: Jaccard of the removed-layer sets, and the fraction of greedy rounds
  that chose the same layer;
* RESIDUAL_WIDTH: subspace overlap `||P1^T P2||_F^2 / k` of the recovered
  projections (1 = the same k-dimensional subspace).

Each agreement is reported beside its CHANCE level (a uniformly random
selection of the same size), so "0.8" can be read as near-identical or as
near-random. Pairs are grouped by the plan's (dose, resample) labels: agreement
BETWEEN resamples at one dose is the selection noise H1 is about; agreement
with the reference profiles is descriptive context, never a mixture verdict.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def chance_jaccard(k: int, n: int) -> float:
    """Expected Jaccard of two independent uniform k-subsets of n."""
    e_inter = k * k / n
    return e_inter / (2 * k - e_inter)


def depth_removed(decision: dict) -> tuple[list[int], list[int]]:
    """(removed set, greedy removal order). `chosen` is the ORIGINAL layer index:
    D2's rounds reproduce its recorded removal order exactly."""
    order = [int(r["chosen"]) for r in decision.get("search_rounds") or []]
    return sorted(order), order


def per_layer_jaccard(a: list[list[int]], b: list[list[int]]) -> float:
    return sum(jaccard(x, y) for x, y in zip(a, b)) / max(1, min(len(a), len(b)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--projections-dir", default=None)
    ap.add_argument("--groups", required=True,
                    help="JSON {profile_qualified_id: {dose, resample, role}}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = json.loads(Path(args.structure).read_text())
    groups = json.loads(Path(args.groups).read_text())
    roots = [r for r in rows if r.get("depth") == 1]
    by_kind: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in roots:
        kind = (r.get("applied_kinds") or ["?"])[0]
        prof = (r.get("profiles") or ["?"])[0]
        by_kind[kind][prof] = r

    proj = {}
    if args.projections_dir:
        import torch
        from safetensors.torch import load_file
        for r in roots:
            if r.get("applied_kinds") == ["RESIDUAL_WIDTH"]:
                f = Path(args.projections_dir) / f"width_projection_{r['state_id']}.safetensors"
                if f.is_file():
                    p = load_file(str(f))["P"].double()
                    q, _ = torch.linalg.qr(p)
                    proj[(r.get("profiles") or ["?"])[0]] = q

    def agree(kind, pa, pb) -> dict:
        da = (by_kind[kind][pa].get("decision") or {})
        db = (by_kind[kind][pb].get("decision") or {})
        if kind == "FFN":
            ka, kb = da["kept_neurons"], db["kept_neurons"]
            k = len(ka[0])
            return {"value": round(per_layer_jaccard(ka, kb), 4), "chance": round(
                chance_jaccard(k, int(groups.get("_teacher_intermediate", 9728))), 4)}
        if kind == "ATTENTION":
            ka, kb = da["kept_heads"], db["kept_heads"]
            k = len(ka[0])
            return {"value": round(per_layer_jaccard(ka, kb), 4), "chance": round(
                chance_jaccard(k, int(groups.get("_teacher_heads", 32))), 4)}
        if kind == "DEPTH":
            ra, oa = depth_removed(da)
            rb, ob = depth_removed(db)
            same = sum(x == y for x, y in zip(oa, ob)) / max(1, len(oa))
            return {"value": round(jaccard(ra, rb), 4), "same_greedy_choice": round(same, 4),
                    "chance": round(chance_jaccard(len(ra), int(groups.get("_teacher_layers", 36))), 4),
                    "removed_a": ra, "removed_b": rb}
        if kind == "RESIDUAL_WIDTH":
            if pa in proj and pb in proj:
                k = proj[pa].shape[1]
                ov = float((proj[pa].T @ proj[pb]).pow(2).sum() / k)
                n = proj[pa].shape[0]
                return {"value": round(ov, 4), "chance": round(k / n, 4)}
            return {"value": None, "note": "projection not recovered"}
        return {"value": None}

    out = {"schema": "aadistill.stage0.structure_agreement/v1", "structure": args.structure,
           "by_operator": {}}
    labelled = {p: g for p, g in groups.items() if not p.startswith("_")}
    for kind, profs in sorted(by_kind.items()):
        present = [p for p in profs if p in labelled]
        pairs = []
        for pa, pb in itertools.combinations(sorted(present), 2):
            ga, gb = labelled[pa], labelled[pb]
            pairs.append({"a": pa, "b": pb, "a_group": ga, "b_group": gb, **agree(kind, pa, pb)})
        within = defaultdict(list)
        for p in pairs:
            if (p["a_group"].get("role") == p["b_group"].get("role") == "treatment"
                    and p["a_group"]["dose"] == p["b_group"]["dose"] and p["value"] is not None):
                within[str(p["a_group"]["dose"])].append(p["value"])
        out["by_operator"][kind] = {
            "pairs": pairs,
            "between_resamples_by_dose": {d: {"n_pairs": len(v), "mean": round(sum(v) / len(v), 4),
                                              "min": min(v), "max": max(v)}
                                          for d, v in sorted(within.items(), key=lambda x: float(x[0]))},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["between_resamples_by_dose"] for k, v in out["by_operator"].items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
