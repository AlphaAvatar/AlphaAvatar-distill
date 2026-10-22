#!/usr/bin/env python3
"""A corpus variant's source allocation, made executable. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.source_plan \
        [--variant <corpus id>] [--check <pool.jsonl> ...]

`corpus_spec.json :: variants` names each corpus B can be cut from, and each
variant's per-stratum source shares. For every (stratum, source) cell this
writes the share of unique supervised positions, the U target at 2.96M and
with the reserve, the estimated independent units (per-unit supervision from
`plans/h1_accounting.json :: quotas_arm_B`), the source's kind and whether it
is approved for H1. With ``--check``, it counts what the given task pools hold
per cell and reports every shortfall: the check between the plan and the
artifacts that are supposed to fill it.

A plan is GENERATABLE only when every source it names is ``approved_for_h1``
(`sources.json`); otherwise it is a review plan, labelled so, and
`generation_adapter` refuses it. Dropping a source means choosing another
variant, which carries another corpus id -- never a quiet substitution inside
one. Writes `logs/stages/stage-2/reasoning_data/plans/source_plan.json`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A

OUT = A.LOG_DIR / "plans/source_plan.json"
TARGET = 2_960_000


def allocation(spec: dict, variant: str) -> dict[str, dict[str, float]]:
    if variant not in spec["variants"]:
        raise SystemExit(f"unknown corpus variant {variant!r}; declared: {sorted(spec['variants'])}")
    a = spec["variants"][variant]["source_allocation"]
    a = spec[a] if isinstance(a, str) else a
    for stratum, cells in a.items():
        if abs(sum(cells.values()) - spec["shares"][stratum]) > 1e-9:
            raise SystemExit(f"{variant}: {stratum} sources sum to {sum(cells.values())}, "
                             f"not its share {spec['shares'][stratum]}")
    return a


def stratum_of(item: dict, spec: dict) -> str | None:
    if item.get("stratum"):
        return item["stratum"]
    return next((s for s, c in spec["strata"].items() if c.get("axis") == item.get("axis")), None)


def plan(variant: str, pools: list[Path] = ()) -> dict:
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    sources = {s["source_id"]: s for s in A.read_json(A.CONFIG_DIR / "sources.json")["sources"]}
    quotas = A.read_json(A.LOG_DIR / "plans/h1_accounting.json")["quotas_arm_B"]
    top = TARGET + spec["reserve_tokens"]
    have = Counter()
    pool_ids = []
    for p in pools:
        rp = Path(p).resolve()
        pool_ids.append({"path": str(rp.relative_to(A.REPO)) if rp.is_relative_to(A.REPO) else f"(outside the repository) {rp.name}",
                         "sha256": A.sha256_file(rp)})
        for it in A.jsonl(Path(p)):
            have[(stratum_of(it, spec), it.get("source"))] += 1
    cells, unknown = [], []
    for stratum, srcs in allocation(spec, variant).items():
        per_unit = quotas[stratum]["supervised_per_unit_estimate"]
        for source, share in sorted(srcs.items()):
            if source not in sources:
                unknown.append(source)
                continue
            u = TARGET * share
            row = {"stratum": stratum, "source": source, "share": share, "kind": sources[source]["kind"],
                   "approved_for_h1": bool(sources[source].get("approved_for_h1")),
                   "U_target_2p96M": round(u), "U_target_with_reserve": round(top * share),
                   "independent_units_estimate": round(top * share / per_unit) if share else 0,
                   "supervised_per_unit_estimate": per_unit}
            if pools:
                row["tasks_in_pools"] = have[(stratum, source)]
                row["shortfall"] = share > 0 and have[(stratum, source)] < row["independent_units_estimate"]
            cells.append(row)
    if unknown:
        raise SystemExit(f"{variant}: sources not in sources.json: {sorted(set(unknown))}")
    unapproved = sorted({c["source"] for c in cells if c["share"] > 0 and not c["approved_for_h1"]})
    return {"schema": "aadistill.stage2.reasoning_data.source_plan/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "corpus_id": variant, "generatable": not unapproved,
            "status": ("GENERATABLE: every source is approved" if not unapproved else
                       f"REVIEW ONLY: not approved for H1: {unapproved}"),
            "unapproved_sources": unapproved, "cells": cells,
            "pools_checked": pool_ids,
            "shortfalls": [f"{c['stratum']}/{c['source']}: {c['tasks_in_pools']} tasks for ~{c['independent_units_estimate']} units"
                           for c in cells if c.get("shortfall")],
            "_units": "independent units needed before acceptance losses; a task builder draws more (its oversupply over the measured accept rate)",
            "authorizes": "nothing"}


def project_demand_shares(spec: dict) -> dict[str, float]:
    """The largest share any variant gives the project generators, per primary
    stratum: one project pool then serves every variant."""
    out: dict[str, float] = {}
    for v in spec["variants"]:
        for stratum, srcs in allocation(spec, v).items():
            if stratum.startswith("primary."):
                out[stratum] = max(out.get(stratum, 0.0), srcs.get("s0_generators_v2_calsplit", 0.0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--check", nargs="*", default=[])
    args = ap.parse_args(argv)
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    variants = [args.variant] if args.variant else sorted(spec["variants"])
    doc = {"schema": "aadistill.stage2.reasoning_data.source_plans/v1",
           "plans": {v: plan(v, [Path(p) for p in args.check]) for v in variants},
           "authorizes": "nothing"}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), {v: p["status"] for v, p in doc["plans"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
