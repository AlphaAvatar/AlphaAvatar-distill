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

The variant arm B is cut from is the one `corpus_spec.json :: chosen_variant`
names (the maintainer's choice); a plan says whether it is that one
(``chosen``). A plan is ADMITTED only when every source it names is ``approved_for_h1``
(`sources.json`: a recorded maintainer decision); otherwise it is a review
plan, labelled so, and `generation_adapter` refuses it. Admission is design:
which TASKS are generated, at what scale, is `generation_plan.py`, which also
drops every task without a usable verifier and reports each cell it then
cannot fill. Dropping a source means choosing another variant, which carries
another corpus id -- never a quiet substitution inside one. Writes
`logs/stages/stage-2/reasoning_data/plans/source_plan.json`.
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


def repo_path(path: Path) -> str:
    rp = Path(path).resolve()
    return str(rp.relative_to(A.REPO)) if rp.is_relative_to(A.REPO) else f"(outside the repository) {rp.name}"


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


def candidate_spec(path: Path) -> dict:
    """A separately identified corpus candidate, as the spec it is cut under.

    The candidate file names its base (`derived_from`: a corpus id and the
    sha256 of the spec file holding it) and declares its own `shares`,
    `source_allocation` and `salt`; everything else (size measure, block length,
    rungs, reserve, acceptance contract) is the base's, unchanged. The base file
    is never edited: one whose bytes differ from the recorded hash refuses. The
    candidate is its own and only variant, and it is never the chosen one
    (`chosen_variant` stays the base's)."""
    c = A.read_json(Path(path))
    base_file = A.REPO / c["derived_from"]["spec"]
    if A.sha256_file(base_file) != c["derived_from"]["spec_sha256"]:
        raise SystemExit(f"{c['corpus_id']}: its base {c['derived_from']['spec']} is not the bytes it was derived from")
    base = A.read_json(base_file)
    if base["corpus_id"] != c["derived_from"]["corpus_id"]:
        raise SystemExit(f"{c['corpus_id']}: base is {base['corpus_id']}, not {c['derived_from']['corpus_id']}")
    if c["corpus_id"] == base["corpus_id"] or c["corpus_id"] in base["variants"]:
        raise SystemExit(f"{c['corpus_id']}: a candidate needs an id of its own")
    if abs(sum(c["shares"].values()) - 1) > 1e-9:
        raise SystemExit(f"{c['corpus_id']}: stratum shares sum to {sum(c['shares'].values())}")
    spec = {**base, "corpus_id": c["corpus_id"], "salt": c["salt"], "shares": c["shares"],
            "variants": {c["corpus_id"]: {"source_allocation": c["source_allocation"], "what": c["role"]}},
            "candidate": {"path": repo_path(Path(path)), "sha256": A.sha256_file(Path(path)), "role": c["role"],
                          "base_sha256": c["derived_from"]["spec_sha256"]}}
    allocation(spec, c["corpus_id"])
    return spec


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
        pool_ids.append({"path": repo_path(Path(p)), "sha256": A.sha256_file(Path(p))})
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
            "corpus_id": variant, "admitted": not unapproved,
            "chosen": variant == (spec.get("chosen_variant") or {}).get("id"),
            "status": ("ADMITTED: every source is approved for H1 (design; not a spending authorization)"
                       if not unapproved else f"REVIEW ONLY: not approved for H1: {unapproved}"),
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
