#!/usr/bin/env python3
"""Apply `configs/stages/stage-2/reasoning_data/isolation_rules.json` to a pool.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.isolation_check \
        --pool <items.jsonl> --out <report.json>

A pool item is a dict with: `messages` / `prompt_text` (what the model is
shown), `source`, and the fields its source's KIND requires (sources.json ::
kinds) -- `generator_id`, `family_id`, `seed` for a project generator;
`module`, `family_id`, `task_type`, `seed` for an external generator;
`source_split` for a dataset -- plus, optionally, `content_sha256` and `gold`.
R0 runs first and refuses an item with an unregistered or untrainable source
or a missing required field: a rule that cannot run on an item does not pass
it, so missing information is a violation, never "no conflict detected".

What each rule establishes is stated in the report (`establishes`) and is no
more than its own check: no shared key (R1), no 13-gram near-copy (R2), no
held-out family or generator or task type (R3-R5), no forbidden split (R6), no
ARC lineage or ARC grid (R7), no training rule that computes an evaluation
rule (R9). Mechanism isolation for families without tabulated rules and for
external modules is the declared review in `external_modules.json`, and the
report says so.

The evaluation set is read through `shared.eval_sets`, which refuses a set whose
bytes are not its authorized identity, so this never measures against a
modified copy -- and it writes nothing into `data/eval-set/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import lru_cache

from stages.reasoning_data import assets as A

from aadistill.evaluation import isolation as ISO
from shared import eval_sets as ES
from stages.calibration_volume.tasks import registry as R
from stages.reasoning_data import mechanism_check as MC

RULES_PATH = A.CONFIG_DIR / "isolation_rules.json"
ARC_INDEX = A.REPO / "data/stages/stage-2/reasoning_data/isolation/arc_grid_index.json"
REQUIRED = {"project_generator": ("generator_id", "family_id", "seed"),
            "external_generator": ("module", "family_id", "task_type", "seed"),
            "external_dataset": ("source_split",),
            "reused_targets": ("source_split",)}
ESTABLISHES = {
    "R0_provenance": "every item names a registered, trainable source and carries the fields its kind needs",
    "R1_held_out_keys": "no item shares a prompt or content key with a held-out unit",
    "R2_near_duplicate": "no item is a 13-gram near-copy of a held-out prompt",
    "R3_rule_family": "no item comes from a rule family a fluid evaluation item uses",
    "R4_eval_only_generators": "no item comes from an evaluation-only generator or its task type",
    "R5_shared_generator_exposure": "shared-generator items are calibration-split families in the training seed range",
    "R6_source_split": "every dataset item comes from a trainable split",
    "R7_arc_agi": "no ARC-lineage source or module, and no item carrying ARC-AGI-1 grids",
    "R9_mechanism_equivalence": "no item comes from a project family whose rule computes an evaluation family's rule",
}
NOT_ESTABLISHED = ("semantic mechanism isolation beyond R9's tabulated rules: families without tabulated "
                   "rules and external modules are covered by the declared review "
                   "(external_modules.json :: mechanism), not by these checks")
_SLASH_GRID = re.compile(r"(?<![\d/])(\d+(?:\s*/\s*\d+)+)(?![\d/])")
_JSON_GRID = re.compile(r"\[\s*\[[\d,\s\[\]]*\]\s*\]")


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


def text_grids(text: str) -> list[str]:
    """Every digit grid in a text, canonical (rows of digits joined by '/'):
    consecutive lines of single digits (space-separated or not), ' / '-separated
    rows, and JSON nested lists. Grids of fewer than two rows or columns, or
    with ragged rows, are not grids."""
    out, run = [], []
    for line in (text or "").splitlines() + [""]:
        toks = line.split()
        if toks and (all(len(t) == 1 and t.isdigit() for t in toks)
                     or (len(toks) == 1 and toks[0].isdigit() and len(toks[0]) >= 2)):
            run.append("".join(toks))
            continue
        if len(run) >= 2 and len({len(r) for r in run}) == 1 and len(run[0]) >= 2:
            out.append("/".join(run))
        run = []
    for m in _SLASH_GRID.finditer(text or ""):
        rows = [re.sub(r"\s", "", r) for r in m.group(1).split("/")]
        if len(rows) >= 2 and len({len(r) for r in rows}) == 1 and len(rows[0]) >= 2:
            out.append("/".join(rows))
    for m in _JSON_GRID.finditer(text or ""):
        try:
            g = json.loads(m.group(0))
        except ValueError:
            continue
        if (isinstance(g, list) and len(g) >= 2 and all(isinstance(r, list) and r for r in g)
                and len({len(r) for r in g}) == 1 and len(g[0]) >= 2
                and all(isinstance(c, int) and 0 <= c <= 9 for r in g for c in r)):
            out.append("/".join("".join(str(c) for c in r) for r in g))
    return out


@lru_cache(maxsize=1)
def arc_index() -> dict:
    doc = A.read_json(ARC_INDEX)
    return {"index": doc["index"], "source": doc["source"],
            "sha256": A.sha256_file(ARC_INDEX)}


def _informative(g: str) -> bool:
    cells = g.replace("/", "")
    return len(cells) >= 9 and sum(c != "0" for c in cells) >= 4


def arc_matches(texts: list[str]) -> dict:
    """Grids of these texts that equal an ARC-AGI-1 grid. An item is refused
    when one matching grid has at least 25 cells and more than one value, when
    two INFORMATIVE grids (>= 9 cells, >= 4 non-zero) match one ARC task, or
    when three grids of any size do. A sparse 3x3 grid or two of them matching
    one task is counted as a coincidence: the project's own grid generator
    produces such grids, and one historical calibration item shares two of
    them with ARC training task 794b24be."""
    idx = arc_index()["index"]
    hits = {}
    for t in texts:
        for g in text_grids(t):
            tasks = idx.get(hashlib.sha256(g.encode()).hexdigest()[:16])
            if tasks:
                hits[g] = tasks
    by_task: dict[str, set] = {}
    for g, tasks in hits.items():
        for t in tasks:
            by_task.setdefault(t, set()).add(g)
    big = [g for g in hits if len(g.replace("/", "")) >= 25 and len(set(g.replace("/", ""))) > 1]
    multi = sorted(t for t, gs in by_task.items()
                   if len(gs) >= 3 or sum(_informative(g) for g in gs) >= 2)
    return {"matched_grids": len(hits), "refuse": bool(big or multi),
            "large_grid_matches": len(big), "tasks_matched_by_several_grids": multi[:5]}


def check_pool(items: Iterable[Mapping], *, sources: Mapping | None = None,
               refuse: bool = False, historical: bool = False) -> dict:
    """Every rule's verdict for one pool. `refuse=True` raises on a violation.

    `historical=True` audits a pool built before these rules existed (arm A's
    corpus_v2, the Stage-0 calibration units): R0's provenance fields and R5's
    training seed range are then reported, not counted as violations, because
    such a pool is never rebuilt to fit a later rule -- and its report says so.
    A NEW pool is never historical.
    """
    doc = rules()
    items = list(items)
    sources = sources if sources is not None else A.read_json(A.CONFIG_DIR / "sources.json")
    report: dict = {"schema": "aadistill.stage2.reasoning_data.isolation_report/v2",
                    "rules": RULES_PATH.relative_to(A.REPO).as_posix(),
                    "eval_set": ES.default_formal(), "n_items": len(items), "results": {},
                    "establishes": ESTABLISHES, "not_established": NOT_ESTABLISHED}
    res = report["results"]
    violations: list[str] = []
    cat = {s["source_id"]: s for s in sources.get("sources", [])}
    whitelist = {(m["source"], m["module"]) for m in
                 A.read_json(A.CONFIG_DIR / "external_modules.json")["modules"]
                 if str(m.get("decision", "")).startswith("candidate")}

    # R0: provenance complete enough for every later rule to run.
    bad0: Counter = Counter()
    for it in items:
        src = it.get("source")
        spec = cat.get(src)
        if spec is None:
            bad0[f"unregistered source {src!r}"] += 1
            continue
        kind = spec.get("kind")
        if kind not in REQUIRED:
            bad0[f"{src}: not trainable"] += 1
            continue
        for f in REQUIRED[kind]:
            if it.get(f) in (None, ""):
                bad0[f"{src}: missing {f}"] += 1
        if kind == "external_generator" and it.get("module") and (src, it["module"]) not in whitelist:
            bad0[f"{src}: module {it['module']!r} is not a candidate module"] += 1
    res["R0_provenance"] = {"violations": dict(sorted(bad0.items())), "historical_pool": historical,
                            "pass": not bad0}
    if bad0 and not historical:
        violations.append("R0_provenance")

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
                                               if r.get("family_id") and r.get("generator_id") else None),
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

    # R6: source splits (an unregistered source is R0's).
    bad6 = Counter()
    for it in items:
        spec = cat.get(it.get("source"))
        if spec is None or spec.get("kind") not in ("external_dataset", "reused_targets", "project_generator"):
            continue
        split = it.get("source_split")
        if split in spec.get("forbidden_splits", []) or (
                spec.get("trainable_splits") and split not in spec["trainable_splits"]):
            bad6[f"{it['source']}:{split}"] += 1
    res["R6_source_split"] = {"violations": dict(bad6), "pass": not bad6}
    if bad6:
        violations.append("R6_source_split")

    # R7: ARC-AGI -- lineage, then content against ARC-AGI-1's grids.
    lineage = Counter(it.get("source") for it in items
                      if cat.get(it.get("source"), {}).get("arc_lineage", "none") != "none")
    arc_bad, matched = [], 0
    for it in items:
        texts = [ES.prompt_text(it)] + [str(it.get("gold") or it.get("answer") or "")]
        m = arc_matches(texts)
        matched += m["matched_grids"] > 0
        if m["refuse"]:
            arc_bad.append({"id": it.get("unit_id") or it.get("task_id"), **m})
    res["R7_arc_agi"] = {"arc_lineage_items": dict(lineage), "content_refused": arc_bad[:50],
                         "n_content_refused": len(arc_bad), "items_with_any_grid_match": matched,
                         "index": {"path": ARC_INDEX.relative_to(A.REPO).as_posix(),
                                   "sha256": arc_index()["sha256"], "source": arc_index()["source"]},
                         "pass": not lineage and not arc_bad}
    if not res["R7_arc_agi"]["pass"]:
        violations.append("R7_arc_agi")

    # R9: a project family whose rule computes an evaluation family's rule.
    mech = MC.review()
    equiv = set(mech["equivalent_training_families"])
    hit9 = Counter(f"{it.get('generator_id')}/{it.get('family_id')}" for it in items
                   if f"{it.get('generator_id')}/{it.get('family_id')}" in equiv)
    res["R9_mechanism_equivalence"] = {
        "refused_families": sorted(equiv), "items": dict(hit9),
        "pairs": [p for p in mech["pairs"] if p["verdict"] == "equivalent"],
        "overlapping_reported": [p for p in mech["pairs"] if p["verdict"] == "overlapping"],
        "families_without_tabulated_rules": mech["training_families_unchecked"],
        "pass": not hit9}
    if hit9:
        violations.append("R9_mechanism_equivalence")

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
