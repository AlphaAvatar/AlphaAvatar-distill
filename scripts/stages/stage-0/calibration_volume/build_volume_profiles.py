#!/usr/bin/env python3
"""Sample the volume treatments' calibration profiles from the base pool.

    PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-0/calibration_volume/build_volume_profiles.py \
        --design configs/stages/stage-0/calibration_volume/volume_design.json

STRATIFIED NESTED SAMPLING WITH INDEPENDENT RESAMPLING:

* a CELL is one (subtype, family) of the design's FIXED family set -- the same
  cells at every dose, so a higher dose adds independent units inside the same
  rule families rather than new families (the volume factor is not confounded
  with family count);
* for each sampling seed, every cell's units are ordered by a seed-keyed shuffle
  of that cell alone; a dose takes units from the front of that order until
  the cell's quota of scored positions is met (at least one unit), so within a
  seed the higher dose CONTAINS the lower one;
* a different seed reorders every cell independently: a genuinely different
  draw, not a renumbered copy of the same one;
* a cell's quota is dose x reference UESP x the domain weight, split equally
  over the domain's subtypes and each subtype's cells -- the same proportions
  at every dose;
* the episode (`unit_id`) is the unit: an interactive episode's turn items are
  taken or left together.

The two profiles of a treatment (the search's free per-operator choice, as in
D1) share each cell's order, so where both draw a cell the smaller quota is a
prefix of the larger.

Writes each profile's items under `artifacts/` and the profile document
(`profiles.json`) with every hash the core's `resolve()` re-checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.initialization.calibration.profiles import (  # noqa: E402
    load_profiles, mixture_content_sha256)

ITEM_FIELDS = ("item_id", "domain", "subtype", "source", "session_id", "n_tokens",
               "n_prediction_positions", "templated", "ids", "tags", "unit_id",
               "generator_id", "family_id", "difficulty", "turn")


def load_units(pool_dir: Path) -> dict[str, dict]:
    units: dict[str, dict] = {}
    for line in (pool_dir / "items.jsonl").open():
        it = json.loads(line)
        u = units.setdefault(it["unit_id"], {"unit_id": it["unit_id"], "items": [],
                                              "domain": it["domain"], "subtype": it["subtype"],
                                              "family": it.get("family_id") or it["subtype"],
                                              "scored": 0})
        u["items"].append(it)
        u["scored"] += it["scored_positions"]
    for u in units.values():
        u["items"].sort(key=lambda i: i.get("turn") or 0)
    return units


def cells(design: dict, units: dict[str, dict]) -> dict[tuple, list[str]]:
    """{domain: {subtype: [families]}}; a family list of ["*"] POOLS the subtype's
    families into one cell, so families are drawn within it per seed (their
    count is then a covariate of the dose, reported per profile)."""
    want = design["cells"]
    out: dict[tuple, list[str]] = defaultdict(list)
    for uid, u in sorted(units.items()):
        fams = want.get(u["domain"], {}).get(u["subtype"])
        if fams is None:
            continue
        if fams == ["*"]:
            out[(u["domain"], u["subtype"], "*")].append(uid)
        elif u["family"] in fams:
            out[(u["domain"], u["subtype"], u["family"])].append(uid)
    missing = [(d, s, f) for d, subs in want.items() for s, fams in subs.items()
               for f in fams if not out.get((d, s, f))]
    if missing:
        raise SystemExit(f"design cells with no eligible unit in the pool: {missing}")
    return out


def order(cell: tuple, uids: list[str], seed: int) -> list[str]:
    rng = random.Random(f"{seed}|{'/'.join(cell)}")
    o = sorted(uids)
    rng.shuffle(o)
    return o


def sample(design, units, cellmap, weights, dose, seed) -> tuple[list[dict], dict]:
    ref = float(design["reference_uesp"])
    by_domain = defaultdict(lambda: defaultdict(list))
    for cell in cellmap:
        by_domain[cell[0]][cell[1]].append(cell)
    items, census = [], {}
    for domain, w in weights.items():
        if w <= 0:
            continue
        subs = by_domain[domain]
        for subtype, cs in subs.items():
            for cell in cs:
                quota = dose * ref * w / len(subs) / len(cs)
                got, taken = 0, []
                for uid in order(cell, cellmap[cell], seed):
                    if got >= quota and taken:
                        break
                    taken.append(uid)
                    got += units[uid]["scored"]
                if got < quota:
                    census.setdefault("_short_cells", []).append("/".join(cell))
                census["/".join(cell)] = {"quota": round(quota, 1), "scored": got,
                                          "units": len(taken), "pool_units": len(cellmap[cell])}
                for uid in taken:
                    items.extend(units[uid]["items"])
    return items, census


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    args = ap.parse_args(argv)
    design = json.loads((REPO / args.design).read_text())
    pool_dir = REPO / design["pool"]["dir"]
    pool_manifest = json.loads((pool_dir / "manifest.json").read_text())
    if pool_manifest["items_sha256"] != design["pool"]["items_sha256"]:
        raise SystemExit("the pool on disk is not the pool the design names")
    units = load_units(pool_dir)
    cellmap = cells(design, units)
    out_root = REPO / design["output"]["items_dir"]
    profiles, treatments, report = [], {}, {}
    for t_id, t in design["treatments"].items():
        ids = []
        for p_name, p in design["profiles"].items():
            if t.get("profiles") and p_name not in t["profiles"]:
                continue
            items, census = sample(design, units, cellmap, p["domain_weights"],
                                   t["dose"], t["sampling_seed"])
            pid = f"calib.s0v_{p_name}_{t_id}"
            rel = f"{design['output']['items_dir']}/{pid}.items.jsonl"
            body = "".join(json.dumps({k: it[k] for k in ITEM_FIELDS if k in it},
                                      sort_keys=True) + "\n" for it in items)
            (out_root).mkdir(parents=True, exist_ok=True)
            (REPO / rel).write_text(body)
            weights = {d: w for d, w in p["domain_weights"].items() if w > 0}
            uesp = sum(len(it["tags"]["assistant"]) for it in items)
            profiles.append({
                "profile_id": pid, "version": 1,
                "description": (f"Stage-0 volume treatment {t_id}, profile {p_name}: "
                                f"dose {t['dose']} x {design['reference_uesp']} UESP, "
                                f"sampling seed {t['sampling_seed']}"),
                "sources": [{"dataset_id": f"aadistill/stage0_base_pool_v{pool_manifest['version']}",
                             "revision": pool_manifest["items_sha256"][:16], "domain": d,
                             "n_items": sum(1 for it in items if it["domain"] == d),
                             "content_sha256": None, "license": "project-generated + corpus_v2",
                             "note": "stratified nested sample; see the design"}
                            for d in weights],
                "domain_weights": weights,
                "token_budget": int(round(t["dose"] * design["reference_uesp"])),
                "sample_rule": design["sample_rule"],
                "seed": int(t["sampling_seed"]),
                "role": "operator_calibration", "materialized": True,
                "items_path": rel, "rendered_manifest_sha256": None,
                "content_sha256": mixture_content_sha256([json.loads(l) for l in body.splitlines()]),
                "items_file_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "leakage_exclusions": design["leakage_exclusions"],
                "leakage_proof_path": design["pool"]["dir"] + "/manifest.json",
                "metadata": {"treatment": t_id, "dose": t["dose"], "uesp": uesp,
                             "n_items": len(items),
                             "n_units": len({it["unit_id"] for it in items}),
                             "tokens": sum(it["n_tokens"] for it in items)},
                "is_null": False})
            ids.append(f"{pid}@v1")
            fams = {(it.get("generator_id"), it.get("family_id")) for it in items
                    if it.get("family_id")}
            profiles[-1]["metadata"]["n_rule_families"] = len(fams)
            report[f"{pid}"] = {"uesp": uesp, "items": len(items),
                                "units": len({it["unit_id"] for it in items}),
                                "rule_families": len(fams),
                                "tokens": profiles[-1]["metadata"]["tokens"],
                                "short_cells": census.get("_short_cells", [])}
        treatments[t_id] = {"profiles": ids, "dose": t["dose"],
                            "sampling_seed": t["sampling_seed"]}
    #: A SEARCH treatment may run several design treatments' profiles together
    #: (the root-level pilot expands all of them under one teacher).
    for s_id, members in (design.get("search_treatments") or {}).items():
        treatments[s_id] = {"profiles": [i for m in members for i in treatments[m]["profiles"]],
                            "members": list(members)}
    doc = {"schema": "aadistill.calibration_profiles/v1",
           "_contract": "Stage-0 calibration-volume profiles, built by build_volume_profiles.py from the design named below; data, registered by search_session.register_stage0_profiles. AUTHORIZES NOTHING.",
           "design": args.design,
           "design_sha256": hashlib.sha256((REPO / args.design).read_bytes()).hexdigest(),
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "profiles": profiles, "treatments": treatments}
    load_profiles(doc, register=False)   # refuse a document the core would refuse
    out = REPO / design["output"]["profiles_doc"]
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
