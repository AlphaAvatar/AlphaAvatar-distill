#!/usr/bin/env python3
"""Apply `configs/stages/stage-2/reasoning_data/isolation_rules.json` to a pool.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.isolation_check \
        --pool <items.jsonl> --out <report.json>

A pool item is a dict with any of: `messages` / `prompt_text` (what the model is
shown), `content_sha256`, `generator_id`, `family_id`, `task_type`, `source`,
`source_split`, `seed`. Every rule reads only the fields it needs; a rule whose
fields an item lacks is reported as not applicable to that item, never passed.

The evaluation set is read through `shared.eval_sets`, which refuses a set whose
bytes are not its authorized identity, so this never measures against a
modified copy -- and it writes nothing into `data/eval-set/`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping

from stages.reasoning_data import assets as A

from aadistill.evaluation import isolation as ISO
from shared import eval_sets as ES
from stages.calibration_volume.tasks import registry as R

RULES_PATH = A.CONFIG_DIR / "isolation_rules.json"


def rules() -> dict:
    return A.read_json(RULES_PATH)


def _rule(doc: dict, rule_id: str) -> dict:
    return next(r for r in doc["rules"] if r["id"] == rule_id)


def fluid_eval_items() -> list[dict]:
    es = ES.default_formal()
    return ES.items(es, "fluid_reasoning_720", "dev_v1")


def held_out_prompt_texts() -> dict[str, list[str]]:
    """Every held-out unit's static prompt texts, by unit."""
    held = ES.component_items(roles=ES.HELD_OUT_ROLES)
    return {u: [ES.prompt_text(r) for r in rows
                if not r.get("interactive") and ES.prompt_text(r).strip()]
            for u, rows in held.items()}


def check_pool(items: Iterable[Mapping], *, sources: Mapping | None = None,
               refuse: bool = False, historical: bool = False) -> dict:
    """Every rule's verdict for one pool. `refuse=True` raises on a violation.

    `historical=True` audits a pool built before these rules existed: R5's
    training seed range is then reported, not counted as a violation, because
    such a pool is never rebuilt to fit a later rule. A NEW pool is never
    historical.
    """
    doc = rules()
    items = list(items)
    report: dict = {"schema": "aadistill.stage2.reasoning_data.isolation_report/v1",
                    "rules": RULES_PATH.relative_to(A.REPO).as_posix(),
                    "eval_set": ES.default_formal(), "n_items": len(items), "results": {}}
    res = report["results"]
    violations: list[str] = []

    # R1: prompt / content keys against every held-out unit.
    r1 = ES.overlap_report(items)
    held_units = {u for u, rows in ES.component_items(roles=ES.HELD_OUT_ROLES).items()}
    shared = {f"{p['b']}:{k}": n for p in r1["pairs"] if p["a"] == "candidate"
              for k, n in p["overlap"].items() if n}
    shared_held = {k: v for k, v in shared.items() if k.split(":")[0] in held_units}
    shared_insample = {k: v for k, v in shared.items() if k.split(":")[0] not in held_units}
    res["R1_held_out_keys"] = {"held_out_overlap": shared_held,
                               "pass": not shared_held}
    res["R8_in_sample_overlap"] = shared_insample
    if shared_held:
        violations.append("R1_held_out_keys")

    # R2: n-gram near-duplicates of held-out prompts, one index per unit.
    r2cfg = _rule(doc, "R2_near_duplicate")
    n, thr = r2cfg["ngram_n"], r2cfg["containment_threshold"]
    texts = [(str(it.get("unit_id") or it.get("id") or i), ES.prompt_text(it))
             for i, it in enumerate(items) if not it.get("interactive")]
    by_unit, flagged, any_shared = {}, {}, set()
    for unit, prompts in sorted(held_out_prompt_texts().items()):
        index = ISO.ngram_index(prompts, n,
                                max_document_frequency=r2cfg.get("max_document_frequency"))
        rep = ISO.ngram_containment(texts, index, n, threshold=thr)
        by_unit[unit] = {k: rep[k] for k in ("with_any_shared_ngram", "flagged")}
        for row in rep["flagged_items"]:
            prev = flagged.get(row["id"])
            if prev is None or row["containment"] > prev["containment"]:
                flagged[row["id"]] = {**row, "unit": unit}
    r2 = {"n": n, "threshold": thr,
          "max_document_frequency": r2cfg.get("max_document_frequency"),
          "checked": len(texts), "by_unit": by_unit, "flagged": len(flagged),
          "flagged_by_unit": dict(Counter(v["unit"] for v in flagged.values())),
          "flagged_ids": sorted(flagged),
          "flagged_items": sorted(flagged.values(), key=lambda r: (-r["containment"], r["id"]))[:50],
          "pass": not flagged}
    res["R2_near_duplicate"] = r2
    if not r2["pass"]:
        violations.append("R2_near_duplicate")

    # R3: rule families against the fluid evaluation items.
    fluid = fluid_eval_items()
    fam_keys = {"generator_family": lambda r: (f"{r['generator_id']}/{r['family_id']}"
                                               if r.get("family_id") else None),
                "family": lambda r: r.get("family_id")}
    rep = ISO.key_overlaps({"pool": items, "eval": fluid}, fam_keys)
    pair = rep["pairs"][0]
    applicable = sum(1 for it in items if it.get("family_id"))
    res["R3_rule_family"] = {"applicable_items": applicable, "overlap": pair["overlap"],
                             "examples": pair["examples"],
                             "pass": not any(pair["overlap"].values())}
    if not res["R3_rule_family"]["pass"]:
        violations.append("R3_rule_family")

    # R4: evaluation-only generators and their task types.
    r4cfg = _rule(doc, "R4_eval_only_generators")
    eval_only_bare = {g.split("@")[0] for g in R.EVAL_ONLY_GENERATORS}
    task_types = set(r4cfg["eval_only_task_types"])
    by_gen = Counter(it.get("generator_id") for it in items if it.get("generator_id"))
    bad_gen = {g: c for g, c in by_gen.items() if g in R.EVAL_ONLY_GENERATORS
               or g.split("@")[0] in eval_only_bare}
    by_type = Counter(it.get("task_type") for it in items if it.get("task_type"))
    bad_type = {t: c for t, c in by_type.items() if t in task_types}
    res["R4_eval_only_generators"] = {"generators": dict(by_gen), "eval_only_hits": bad_gen,
                                      "task_type_hits": bad_type,
                                      "items_without_task_type": sum(
                                          1 for it in items if it.get("generator_id")
                                          and it["generator_id"] not in R.BY_ID
                                          and it["generator_id"] not in R.BY_ID_V2
                                          and not it.get("task_type")),
                                      "pass": not bad_gen and not bad_type}
    if not res["R4_eval_only_generators"]["pass"]:
        violations.append("R4_eval_only_generators")

    # R5: shared generators -> calibration-split families and the training seed range.
    r5cfg = _rule(doc, "R5_shared_generator_exposure")
    lo, hi = r5cfg["training_seed_range"]
    shared_gens = {g.generator_id for g in R.GENERATORS_V2} | {g.generator_id for g in R.GENERATORS}
    exposure, r5_bad = Counter(), []
    for it in items:
        g = it.get("generator_id")
        if g not in shared_gens:
            continue
        exposure["shared"] += 1
        gen = R.BY_ID_V2.get(g) or R.BY_ID.get(g)
        cal = set(R.family_split(gen)["calibration"])
        seed = it.get("seed")
        if it.get("family_id") not in cal:
            r5_bad.append({"id": it.get("unit_id") or it.get("task_id"), "why": "not a calibration-split family"})
        elif seed is not None and not (lo <= int(seed) < hi):
            r5_bad.append({"id": it.get("unit_id") or it.get("task_id"),
                           "why": f"seed {seed} outside the training range"})
    res["R5_shared_generator_exposure"] = {
        "items_from_shared_generators": exposure["shared"],
        "training_seed_range": [lo, hi], "violations": r5_bad[:50],
        "n_violations": len(r5_bad),
        "historical_pool": historical,
        "pass": not r5_bad}
    if r5_bad and not historical:
        violations.append("R5_shared_generator_exposure")

    # R6: source splits, when the pool names its sources.
    if sources is not None:
        bad6 = Counter()
        cat = {s["source_id"]: s for s in sources.get("sources", [])}
        for it in items:
            src = it.get("source")
            if src is None or src not in cat:
                continue
            spec = cat[src]
            split = it.get("source_split")
            if split in spec.get("forbidden_splits", []) or (
                    spec.get("trainable_splits") and split not in spec["trainable_splits"]):
                bad6[f"{src}:{split}"] += 1
        res["R6_source_split"] = {"violations": dict(bad6), "pass": not bad6}
        if bad6:
            violations.append("R6_source_split")

    report["violations"] = violations
    report["pass"] = not violations
    if refuse and violations:
        raise ISO.IsolationViolation(f"isolation rules violated: {violations}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refuse", action="store_true")
    args = ap.parse_args(argv)
    sources = A.read_json(A.CONFIG_DIR / "sources.json")
    rep = check_pool(A.jsonl(A.Path(args.pool)), sources=sources, refuse=args.refuse)
    A.Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    print("PASS" if rep["pass"] else f"VIOLATIONS {rep['violations']}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
