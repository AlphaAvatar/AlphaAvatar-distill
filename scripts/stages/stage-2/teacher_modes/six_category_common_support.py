#!/usr/bin/env python3
"""A NEW comparable control for the non-thinking schema_arguments shortfall: the six-category common-support design
(`six_category_common_support@v1`). $0, dev box; it designs, prices and recomputes, and starts nothing.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.six_category_common_support \\
        --write logs/stages/stage-2/teacher_modes/audit/six_category_common_support.json \\
        [--price-pool data/stages/stage-2/teacher_modes/tool_pool_v1]

Why (maintainer 2026-11-04, review of a143b7ed, section 2): the registered A2-6cat cannot be built. Its non-thinking
schema_arguments training cell projects 19,968 [11,803, 33,175] supervised tokens against the registered 74,013, and
no compliant independent supply exists. An `approved_exceptions` entry that let A2-6cat stand in for the
equal-proportion arm is refused. So this is a separate study identity with its own research question, controls,
budget, proportions, data reuse and interpretation limits. It never replaces the registered six-category identity,
and its results are never reported as the registered equal-proportion, equal-token A1/A2 comparison.

The design (`design`):
* **ONE training composition for every arm.** The tool type's schema_arguments quota is fixed IN ADVANCE at the
  non-thinking cell's projected lower bound, rounded down to a hundred tokens (`QUOTA_RULE`). The other six tool
  categories share the freed tokens in their registered proportions. Every other type, the fluid category, each arm's
  total supervised tokens, the teacher lineages, the protocols, the verifier, packing and the loss stay as
  registered. If the realized non-thinking supply falls below the quota, the fallback is the realized supply for
  every arm (recorded).
* **Arms:** A1-6cat-cs, A2-6cat-cs and C1-6cat-cs, two seeds each: the registered matrix's count. The registered
  A1-6cat and C1-6cat stay buildable as registered and are not trained in this design (`alternatives` prices the
  variant that keeps them).

Everything is counted before packing (P19). `recompute` derives every number from the committed records, so the
analysis is reproducible from them.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SIX = REPO / "configs/stages/stage-2/teacher_modes/six_category_design.json"
PLAN = REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_plan.json"
BUDGET = REPO / "logs/stages/stage-2/teacher_modes/plans/packed_training_budget.json"
REJECTED = REPO / "logs/stages/stage-2/teacher_modes/audit/schema_arguments_shortfall_v2.json"
OUT = REPO / "logs/stages/stage-2/teacher_modes/audit/six_category_common_support.json"
SCHEMA = "aadistill.stage2.teacher_modes.six_category_common_support/v1"
STUDY_ID = "six_category_common_support@v1"
CAPPED = "schema_arguments"
QUOTA_RULE = ("the H-N training schema_arguments cell's projected LOWER bound (Wilson 90% x measured units per admitted "
              "episode over its whole supply, tool_generation_plan.json), rounded DOWN to a hundred supervised tokens, "
              "fixed before any further generation; the same share of the tool type in every arm")
MODE_ARM = {"H-T": "A1-6cat-cs", "H-N": "A2-6cat-cs", "T2507": "C1-6cat-cs"}


def _sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def registered() -> dict:
    six = json.loads(SIX.read_text())
    shares = {k: float(v["share"]) for k, v in six["tool_calling"]["categories"].items()}
    base = six["base_category"]
    fluid = 2962514          #: six_category_design :: base_category.A1_6cat_and_A2_6cat.fluid ("2,962,514 target")
    return {"shares": shares, "base_total": float(base["total_supervised_tokens"]), "per_type": float(base["per_type"]),
            "fluid": fluid, "c1_tool_target": float(base["C1_6cat"]["tool_calling_target"])}


def cs_shares(shares: dict, quota_share: float) -> dict:
    rest = 1.0 - shares[CAPPED]
    return {c: (quota_share if c == CAPPED else s * (1.0 - quota_share) / rest) for c, s in shares.items()}


def training_cells(plan: dict) -> dict:
    return {(li, m, c["category"]): c for li, v in plan["lineages"].items() for m, mm in v["modes"].items()
            for c in mm["cells"] if c["split"] == "training"}


def recompute(plan: dict | None = None) -> dict:
    plan = plan or json.loads(PLAN.read_text())
    reg = registered()
    shares, T = reg["shares"], reg["per_type"]
    cells = training_cells(plan)
    hn = cells[("hybrid", "H-N", CAPPED)]
    low, exp, high = hn["projection"]["whole_supply_units"]
    quota = math.floor(low / 100.0) * 100.0
    q_share = quota / T
    cs = cs_shares(shares, q_share)
    arm_tool = {"H-T": T, "H-N": T, "T2507": reg["c1_tool_target"]}
    per_arm = {}
    for mode, tool in arm_tool.items():
        rows = {}
        for c in sorted(shares):
            a, b = tool * shares[c], tool * cs[c]
            rows[c] = {"registered": round(a, 1), "common_support": round(b, 1), "change": round(b - a, 1),
                       "change_pct": round(100 * (b - a) / a, 2)}
        per_arm[MODE_ARM[mode]] = {"tool_type_tokens": round(tool, 2), "categories": rows,
                                   "moved_tokens": round(tool * (shares[CAPPED] - cs[CAPPED]), 1)}
    whole_a = reg["base_total"] + reg["fluid"]
    moved = T * (shares[CAPPED] - q_share)
    feas = {}
    for (li, m, cat), c in sorted(cells.items()):
        want = arm_tool[m] * cs[cat]
        lo, ex, hi = c["projection"]["whole_supply_units"]
        feas[f"{li}|{m}|{cat}"] = {"cs_target": round(want, 1), "registered_target": round(c["target"], 1),
                                   "projection": [lo, ex, hi], "supply_tasks": c.get("supply"),
                                   "sealed_units": (c.get("sealed") or {}).get("units"),
                                   "status": ("REACHABLE" if lo >= want else "AT_RISK" if ex >= want else "SHORTFALL")}
    rejected = json.loads(REJECTED.read_text())["recommended_design"]["mixture_impact_A2_6cat"]
    return {
        "quota": {"rule": QUOTA_RULE, "hn_projection": [low, exp, high], "quota_tokens": quota,
                  "share_of_tool_type": round(q_share, 6), "registered_share": shares[CAPPED],
                  "fallback": "realized H-N supply below the quota: the realized supply, rounded down to a hundred, "
                              "for every arm (recorded; never filled by duplication, relabelling, another split or "
                              "a loss weight)"},
        "shares": {"registered": shares, "common_support": {k: round(v, 6) for k, v in cs.items()}},
        "per_arm": per_arm,
        "what_moves": {"tokens_per_A_arm": round(moved, 1),
                       "share_of_tool_type_pct": round(100 * moved / T, 2),
                       "share_of_the_six_supporting_types_pct": round(100 * moved / reg["base_total"], 3),
                       "share_of_the_whole_A_arm_pct": round(100 * moved / whole_a, 3),
                       "whole_A_arm_supervised_tokens": whole_a,
                       "unchanged": "each arm's total supervised tokens, the tool type's 1/6 of the six supporting "
                                    "types, the five other types, the fluid category, fluid : supporting"},
        "feasibility": feas,
        "rejected_design_arithmetic": {
            "record": REJECTED.relative_to(REPO).as_posix(),
            "its_shares": {k: rejected["registered"][k] for k in ("tool_type_share_of_six_types_pct",)},
            "correction": "the rejected v2 record divided by the six supporting types' total PLUS the tool type, "
                          "counting the tool type twice: the registered tool share of the six types is 1/6 "
                          f"({round(100 * T / reg['base_total'], 2)}%), not "
                          f"{rejected['registered']['tool_type_share_of_six_types_pct']}%; its whole-set change "
                          f"({rejected['at_expected']['whole_set_change_pct']}%) is "
                          f"{round(100 * rejected['at_expected']['tool_type_change'] / reg['base_total'], 2)}% of the "
                          f"six types and {round(100 * rejected['at_expected']['tool_type_change'] / whole_a, 2)}% of "
                          "the whole A arm. The record stays as written (refused, not amended)"}}


def priced(pool_dir: Path, quota_share: float) -> dict:
    """The hybrid and Thinking-2507 sessions priced at the common-support TRAINING targets with the plan builder itself
    (calibration, state evaluation and validation targets unchanged), against the registered plan's prices."""
    from stages.teacher_modes import tool_generation_plan as TGP
    reg = registered()
    cs = cs_shares(reg["shares"], quota_share)
    orig = TGP.cell_targets

    def cell_targets(bp, shares, lineage, mode, splits):
        out = orig(bp, shares, lineage, mode, splits)
        total = {r["split"]: float(bp["targets"][r["split"]]["per_mode"][mode]) for r in out
                 if r["split"] == "training"}
        for r in out:
            if r["split"] == "training":
                r["target"] = total["training"] * cs[r["category"]]
        return out
    TGP.cell_targets = cell_targets
    try:
        plan = TGP.build(Path(pool_dir))
    finally:
        TGP.cell_targets = orig
    now = json.loads(PLAN.read_text())["price"]
    cells = training_cells(plan)
    return {"sessions": {li: {k: plan["price"]["sessions"][li][k] for k in ("expected", "upper", "runner_cap")}
                         for li in plan["price"]["sessions"]},
            "registered_plan_sessions": {li: {k: now["sessions"][li][k] for k in ("expected", "upper", "runner_cap")}
                                         for li in now["sessions"]},
            "training_cells": {f"{li}|{m}|{cat}": {k: c.get(k) for k in ("target", "bound", "status")}
                               for (li, m, cat), c in sorted(cells.items())},
            "_": "the plan builder's own measured price (priors, import, throughput) at the common-support training "
                 "targets; a dry run written nowhere in the repository"}


def topup(rec: dict, realized: dict) -> dict:
    """``realized``: {category: admitted supervised tokens} of the Thinking-2507 session's training cells."""
    need = {}
    for key, f in rec["feasibility"].items():
        li, mode, cat = key.split("|")
        if li == "thinking_2507":
            need[cat] = round(max(0.0, f["cs_target"] - float(realized.get(cat, 0.0))), 1)
    return {"status": "MEASURED", "realized": realized, "missing_supervised_tokens": need,
            "total": round(sum(need.values()), 1)}


def design(rec: dict, price: dict | None, realized_t2507: dict | None = None) -> dict:
    budget = json.loads(BUDGET.read_text())
    arm = {a: budget["arms"][a]["usd_both_seeds"] for a in ("A1-6cat", "A2-6cat", "C1-6cat")}
    t2507_topup = topup(rec, realized_t2507) if realized_t2507 else {
        "status": "UNMEASURED until the running Thinking-2507 session closes",
        "rule": "per training category, the common-support target less the units the session admitted (the session "
                "stops each cell at its REGISTERED target); the schema cell needs no top-up; priced from that "
                "session's measured units and tokens per task, generated later from each cell's frozen order"}
    return {
        "schema": SCHEMA, "study_id": STUDY_ID,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "DESIGN at $0 for the maintainer's review: nothing generated, packed or trained for it",
        "decided_by": "maintainer 2026-11-04 (review of a143b7ed), section 2: H-N schema_arguments training SHORTFALL "
                      "NO-GO; no approved_exceptions; design a new, scientifically comparable control with its research "
                      "question, controls, token budget, category proportions, data reuse and interpretation limits; a "
                      "registered matched-task or common-support design allowed, never presented as the original "
                      "equal-proportion, equal-token A1/A2 experiment, nor replacing the original study identity",
        "inputs": {p.relative_to(REPO).as_posix(): _sha(p) for p in (SIX, PLAN, BUDGET, REJECTED)},
        "registered_identity": {
            "id": "six_category_design.json (A1-6cat / A2-6cat / C1-6cat)",
            "status": "unchanged and kept: A2-6cat is NOT realizable as registered (schema_arguments 74,013 "
                      "registered vs 19,968 [11,803, 33,175] projected); A1-6cat and C1-6cat stay buildable as "
                      "registered; no result of this design is a result of that identity"},
        "research_question": "With ONE hybrid teacher (Qwen/Qwen3-4B@1cfa9a72), does distilling from its thinking mode "
                             "or its non-thinking mode give the better student, at the same unique supervised tokens "
                             "and the same category composition -- the composition both modes can supply in full "
                             "(common support)? Secondary, with the same composition: hybrid thinking vs Thinking-2507 "
                             "(A1-6cat-cs vs C1-6cat-cs), each lineage end to end",
        "arms": {"A1-6cat-cs": "hybrid thinking mode", "A2-6cat-cs": "hybrid non-thinking mode",
                 "C1-6cat-cs": "Thinking-2507", "seeds": [20260726, 20260801]},
        "changed_scientific_variable": {
            "only": "the tool type's internal category composition, identically in every arm: schema_arguments from "
                    f"15% to {round(100 * rec['quota']['share_of_tool_type'], 2)}% of the tool type, the six other "
                    f"tool categories x{round((1 - rec['quota']['share_of_tool_type']) / 0.85, 4)} in their registered "
                    "proportions",
            "size": rec["what_moves"]},
        "controls": ["each arm's total unique supervised tokens (A arms 5,923,021; C1 its registered total)",
                     "the six supporting types at 1/6 each and the fluid category, cells and shares unchanged",
                     "one teacher lineage per arm end to end (P17.1); each teacher's official template and preset",
                     "the pool, splits, frozen orders, tool_verification@v2 and every admission rule unchanged",
                     "selection: the registered largest-remainder interleave on supervised tokens, whole examples, "
                     "frozen order, never by observed quality",
                     "Stage 3: random shuffle, sequential packing with truncation, causal attention, counted before "
                     "packing; the registered recipe, loss and schedule rule (training_plan.schedule)"],
        "not_controlled_and_reported": ["compute: equal supervised tokens are not equal compute -- A2's non-thinking "
                                        "turns carry far more context per supervised token (registered budget: "
                                        "5,161 vs 1,436 expected packed sequences per seed); report non-padding "
                                        "tokens, packed sequences, optimizer steps, exposure and training FLOPs beside "
                                        "every result",
                                        "the schema_arguments tasks differ between arms: each arm's own admitted "
                                        "episodes, by frozen order, as in the registered design"],
        "token_budget": rec["per_arm"], "proportions": rec["shares"], "quota": rec["quota"],
        "feasibility": rec["feasibility"],
        "data_reuse": ["every sealed episode is reused: the pilot's 397 tasks x 2 modes (the explicit import), the "
                       "running Thinking-2507 session's shards, the hybrid session's when it runs -- nothing "
                       "regenerated, relabelled, duplicated or borrowed across splits",
                       "the hybrid plan is rebuilt at these training targets before its session (one session, no top-up); "
                       "its non-thinking schema cell still generates its whole remaining supply (the realized number "
                       "is reported), and its thinking schema cell needs fewer tasks (the pilot's import already "
                       f"seals {rec['feasibility']['hybrid|H-T|schema_arguments'].get('sealed_units')} supervised "
                       f"tokens of the {rec['quota']['quota_tokens']:.0f} quota)",
                       "Thinking-2507: the running session's episodes are kept; the six other categories' extra quota "
                       "is a later top-up from each cell's frozen order",
                       "admitted episodes beyond a quota stay in the records, unused, never selected by outcome"],
        "price": price, "t2507_topup": t2507_topup,
        "training_budget_usd_both_seeds": {"registered_estimates": arm,
                                           "this_design": "the same three arms: A2-6cat-cs ~ A2-6cat, A1-6cat-cs ~ "
                                                          "A1-6cat, C1-6cat-cs ~ C1-6cat (the composition moves "
                                                          f"{rec['what_moves']['share_of_the_whole_A_arm_pct']}% of an "
                                                          "A arm's tokens between tool categories)"},
        "alternatives": {
            "keep_the_registered_A1_and_C1_too": {
                "what": "train A1-6cat and C1-6cat as registered AND A1-6cat-cs and A2-6cat-cs (4 arms x 2 seeds)",
                "extra_usd_both_seeds": arm["A1-6cat"],
                "buys": "the registered A1-vs-C1 contrast unchanged, plus A1-6cat vs A1-6cat-cs as a composition "
                        "sensitivity check whose difference (well under 1% of the tokens) two seeds cannot resolve",
                "verdict": "not recommended: about $11-18 more for a contrast the seeds cannot measure"},
            "realized_quota": {
                "what": "the quota set to the REALIZED non-thinking schema supply after the hybrid session",
                "verdict": "not recommended: it keeps about 8k more schema tokens, but the composition would then depend "
                           "on an observed generation outcome; the fixed quota is registered before it"},
            "matched_task_schema_cell": {
                "what": "both arms' schema cell restricted to the tasks admitted in BOTH modes",
                "verdict": "not recommended: a same-tasks cell has unequal tokens (~19x per task), so the arms could "
                           "not stay token-matched; a same-tasks comparison is a different comparison and is never "
                           "pooled with a token-matched one"},
            "drop_schema_arguments": {"what": "quota 0 in every arm",
                                      "verdict": "not recommended: discards verified data both modes supply"}},
        "interpretation_limits": [
            "a result of six_category_common_support@v1, never of the registered six-category identity, never pooled "
            "with it, and never described as the registered equal-proportion, equal-token A1/A2 comparison",
            "argument construction (schema_arguments) carries about 2.4% of the tool type instead of 15% in every arm: "
            "the contrast says less about schema-heavy tool use, and nothing about the registered composition",
            "the composition is supply-conditioned: it was chosen because the non-thinking teacher's verified output "
            "on this pool is short -- a property of the data, fixed before generation, not of any training or "
            "evaluation outcome",
            "tool-calling and every other capability reported separately (P10.2, P10.3); eval-set-1 unchanged",
            "two seeds per arm: differences below the seed-to-seed variance are not findings"],
        "authorizes": "nothing: no generation, pack or training; the hybrid's grant stays deferred"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=str(OUT))
    ap.add_argument("--price-pool", default=None)
    ap.add_argument("--keep-price", action="store_true", help="reuse the price already in --write")
    ap.add_argument("--t2507-summary", default=None, help="the Thinking-2507 session's evidence/generation_summary.json")
    a = ap.parse_args(argv)
    rec = recompute()
    if a.keep_price:
        price = json.loads(Path(a.write).read_text()).get("price")
    else:
        price = priced(Path(a.price_pool), rec["quota"]["share_of_tool_type"]) if a.price_pool else None
    realized = None
    if a.t2507_summary:
        summ = json.loads(Path(a.t2507_summary).read_text())
        realized = {c["category"]: c["supervised_tokens"] for c in summ["cells"] if c["split"] == "training"}
    doc = design(rec, price, realized)
    if a.t2507_summary:
        src = Path(a.t2507_summary).resolve()
        doc["t2507_topup"]["source"] = {"path": src.relative_to(REPO).as_posix(), "sha256": _sha(src),
                                        "run_id": summ["run_id"]}
    Path(a.write).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"quota": doc["quota"], "moves": rec["what_moves"],
                      "price": (price or {}).get("sessions") and {li: {k: v["usd"] for k, v in s.items() if k != "runner_cap"}
                                                                   for li, s in price["sessions"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
