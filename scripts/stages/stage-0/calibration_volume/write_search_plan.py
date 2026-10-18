#!/usr/bin/env python3
"""Write a search-session plan: treatments, frozen profile hashes, priced bounds.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/write_search_plan.py \
        --name pilot_search --treatments pilot_root --max-depth 1 --purpose "..."

The plan is what a search authorization binds (`governance.plan_hash`). It
freezes, per treatment, every profile's `profile_hash`, which the driver
re-checks on the pod, and it PRICES each treatment from measured rates rather
than a guess:

* DEPTH is GPU-bound and linear in calibration tokens: the bound is the
  measured MAX minutes per invocation at its measured token count, scaled by
  each profile's tokens (`cost_source`, the D2 scored_v1 search telemetry);
* every other operator at its measured max, plus per-expansion overhead;
* a full search adds the measured non-root expansions of the baseline's beam.

A bound is a ceiling, not a forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

PLANS = REPO / "configs/stages/stage-0/calibration_volume"
PROFILES_DOC = PLANS / "profiles.json"
COST_SOURCE = "logs/stages/stage-1/phase_d2/runs/d2_search_20261015_051722/evidence/search_analysis.json"
COST_TOKENS = 59830          # the D2 profiles' token count the timing was measured at
OVERHEAD_MIN_PER_EXPANSION = 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--treatments", nargs="+", required=True)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--finalists-k", type=int, default=0)
    ap.add_argument("--depth-profiles", nargs="*", default=None,
                    help="restrict DEPTH (the token-linear, expensive operator) to these "
                         "profiles; every other operator expands every profile")
    args = ap.parse_args(argv)

    doc = json.loads(PROFILES_DOC.read_text())
    by_id = {f"{p['profile_id']}@v{p['version']}": p for p in doc["profiles"]}
    timing = json.loads((REPO / COST_SOURCE).read_text())["operator_timing"]["per_impl"]
    depth_max = timing["depth.causal_kl_greedy_v1"]["operator_minutes_max"]
    other_max = max(v["operator_minutes_max"] for k, v in timing.items() if not k.startswith("depth"))
    treatments, plan_treatments = {}, []
    for t in args.treatments:
        ids = doc["treatments"][t]["profiles"]
        prof = [by_id[i] for i in ids]
        from aadistill.initialization.calibration.profiles import get_profile, load_profiles
        load_profiles(doc)
        hashes = {i: get_profile(i).profile_hash for i in ids}
        depth_ids = set(args.depth_profiles) if args.depth_profiles is not None else set(ids)
        if not depth_ids <= set(ids):
            raise SystemExit(f"--depth-profiles names profiles outside {t}: {depth_ids - set(ids)}")
        root = sum((depth_max * p["metadata"]["tokens"] / COST_TOKENS
                    if f"{p['profile_id']}@v{p['version']}" in depth_ids else 0.0)
                   # FFN/WIDTH/ATTENTION are activation passes over the same
                   # calibration tokens, so they scale too (never below one call)
                   + 3 * other_max * max(1.0, p["metadata"]["tokens"] / COST_TOKENS)
                   + 4 * OVERHEAD_MIN_PER_EXPANSION for p in prof)
        if args.max_depth == 1:
            bound = root
        else:
            # the baseline's measured full-search trajectory beyond the root
            nonroot = sum(v["operator_minutes_total"] for v in timing.values()) * \
                max(p["metadata"]["tokens"] for p in prof) / COST_TOKENS
            bound = root + nonroot
        treatments[t] = {"profile_hashes": hashes, "profiles": ids}
        if args.depth_profiles is not None:
            treatments[t]["impl_profiles"] = {"depth.causal_kl_greedy_v1": sorted(depth_ids)}
        plan_treatments.append({"treatment": t, "max_depth": args.max_depth,
                                "bound_minutes": round(bound + 10.0, 1),
                                "profile_tokens": {i: by_id[i]["metadata"]["tokens"] for i in ids}})
    items_dir = str(Path(by_id[doc["treatments"][args.treatments[0]]["profiles"][0]]["items_path"]).parent)
    plan = {
        "schema": "aadistill.stage0.calibration_volume.session_plan/v1",
        "session_kind": "search", "name": args.name, "purpose": args.purpose,
        "teacher_revision": "768f209d9ea81521153ed38c47d515654e938aea",
        "profiles_doc": str(PROFILES_DOC.relative_to(REPO)),
        "profiles_doc_sha256": hashlib.sha256(PROFILES_DOC.read_bytes()).hexdigest(),
        "profile_items_dir": items_dir,
        "treatments_doc": f"configs/stages/stage-0/calibration_volume/{args.name}_session.json",
        "treatments": plan_treatments,
        "search": {"finalists_k": args.finalists_k},
        "cost_source": {"path": COST_SOURCE, "depth_max_minutes": depth_max,
                        "at_tokens": COST_TOKENS, "other_operator_max_minutes": other_max,
                        "overhead_minutes_per_expansion": OVERHEAD_MIN_PER_EXPANSION},
        "hardware": {"gpu": "NVIDIA L40S", "why": "the hardware every D-series search ran on; the operator costs above were measured on it"},
        "contingency_fraction": 0.2, "reserve_minutes": 30,
        "authorizes": "nothing"}
    # the driver reads treatments[t]["profile_hashes"] from the treatments doc
    plan["treatments_by_id"] = treatments
    out = PLANS / f"{args.name}_session.json"
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({t["treatment"]: t["bound_minutes"] for t in plan_treatments}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
