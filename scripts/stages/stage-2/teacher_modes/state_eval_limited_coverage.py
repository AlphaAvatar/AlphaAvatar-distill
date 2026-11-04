#!/usr/bin/env python3
"""The non-thinking state evaluation's schema_arguments cell: a limited-coverage plan (maintainer 2026-11-04, review of
a143b7ed, section 3). $0, dev box; it measures from the committed records and starts nothing.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.state_eval_limited_coverage \\
        --write logs/stages/stage-2/teacher_modes/audit/state_eval_hn_schema_limited_coverage.json

The cell's 1,063 scored positions (of 1,800) are accepted as a real measurement: the split's 29 schema_arguments tasks
were all attempted (the pilot, imported), and 11 episodes were admitted. This plan proposes a SEPARATE, versioned
suite identity, `state_eval_hybrid@v2-lc1`. It keeps the same KL, scored-position and domain-aggregation definitions
and the same frozen shared inputs, and it never marks the cell as meeting its quota. It reports what that coverage can
and cannot support:

* the cell's tasks, episodes and positions, their sources and verification;
* its weight in each ranking objective, against the registered 1,800;
* the episode-level uncertainty: positions inside an episode are not independent, and four episodes carry most of
  them;
* a sensitivity rule for every ranking decision, follow-up validation, and how the identity differs from the
  1,800-target one.

The registration of `@v2-lc1`, and any change of evaluation identity or protocol, is bound separately by the
maintainer. Nothing here approves a SHORTFALL.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
IMPORT = REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_import.json"
PLAN = REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_plan.json"
TYPES = REPO / "configs/stages/stage-2/teacher_modes/full_search_types_v2.json"
OUT = REPO / "logs/stages/stage-2/teacher_modes/audit/state_eval_hn_schema_limited_coverage.json"
SCHEMA = "aadistill.stage2.teacher_modes.state_eval_limited_coverage/v1"
SUITE = "state_eval_hybrid@v2-lc1"
CAT, SPLIT = "schema_arguments", "state_eval"
#: the ranking policy's practical-equivalence epsilon per objective (aadistill.initialization.planning.ranking.PARETO_V1)
EPSILON = 1e-4


def _sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kish(weights) -> float:
    w = [float(x) for x in weights]
    return round(sum(w) ** 2 / sum(x * x for x in w), 2) if w else 0.0


def episodes(imp: dict, mode: str) -> list[dict]:
    out = []
    for tid, v in sorted(imp["modes"][mode]["tasks"].items()):
        if v["split"] == SPLIT and v["category"] == CAT:
            out.append({"task_id": tid, "shape": v["shape"], "outcome": v["final"]["outcome"],
                        "candidate": v["final"]["candidate"], "positions": float(v["final"]["units"])})
    return out


def cell(imp: dict, pool_tasks: dict | None, mode: str) -> dict:
    rows = episodes(imp, mode)
    adm = [r for r in rows if r["outcome"] == "admitted"]
    pos = sorted((r["positions"] for r in adm), reverse=True)
    total = sum(pos)
    src = Counter()
    levels = Counter()
    for r in rows:
        t = (pool_tasks or {}).get(r["task_id"])
        if t:
            src[t["source"]["dataset"]] += 1
            levels["+".join((t["source"].get("verification") or {}).get("levels") or [])] += 1
    return {"tasks_in_split": len(rows), "attempted": len(rows),
            "admitted_episodes": len(adm), "refused": len(rows) - len(adm),
            "admitted_by_shape": dict(Counter(r["shape"] for r in adm)),
            "admitted_on_candidate": dict(Counter(str(r["candidate"]) for r in adm)),
            "scored_positions": total, "positions_per_episode": pos,
            "largest_episode_share_pct": round(100 * pos[0] / total, 1) if pos else None,
            "four_largest_share_pct": round(100 * sum(pos[:4]) / total, 1) if pos else None,
            "kish_effective_episodes": kish(pos),
            "sources": dict(src) or "UNMEASURED (the pool is not on this machine)",
            "verification_levels": dict(levels) or "UNMEASURED (the pool is not on this machine)"}


def weights(target: float, realized: float, quotas: dict, n_supporting_types: int, n_domains: int) -> dict:
    """The cell's weight in the equal-domain mean: the supporting domain's mean over its (sub-type, mode) entries, the
    tool sub-type's KL pooled over its positions (the other categories at their quotas)."""
    def at(x):
        sub = x + sum(v for k, v in quotas.items() if k != CAT)
        within = x / sub
        entries = 2 * n_supporting_types                    #: every supporting sub-type in both modes
        return {"within_tool_subtype": round(within, 4),
                "in_supporting_domain_mean": round(within / entries, 5),
                "in_equal_domain_mean": round(within / entries / n_domains, 6)}
    reg, lim = at(target), at(realized)
    return {"registered": reg, "limited": lim,
            "kl_difference_that_moves_the_objective_by_epsilon_nats": {
                "equal_domain_mean": round(EPSILON / lim["in_equal_domain_mean"], 4),
                "worst_domain (when supporting is the worst domain)": round(EPSILON / lim["in_supporting_domain_mean"], 4)},
            "_": "pooled over positions inside the tool sub-type (DistortionSums merge), unweighted over sub-types and "
                 "domains (aadistill.initialization.statistics.contribution.domain_balanced_score); the other tool "
                 "categories at their quotas (whole-episode overshoot moves these by a little)"}


def build(pool_dir: Path | None = None) -> dict:
    imp = json.loads(IMPORT.read_text())
    plan = json.loads(PLAN.read_text())
    types = json.loads(TYPES.read_text())
    pool_tasks = None
    if pool_dir and (Path(pool_dir) / "state_eval.tasks.jsonl").is_file():
        pool_tasks = {}
        for line in (Path(pool_dir) / "state_eval.tasks.jsonl").read_text().splitlines():
            t = json.loads(line)
            if t["category"] == CAT:
                pool_tasks[t["task_id"]] = t
    cells = {c["category"]: c for c in plan["lineages"]["hybrid"]["modes"]["H-N"]["cells"] if c["split"] == SPLIT}
    hn, ht = cell(imp, pool_tasks, "H-N"), cell(imp, pool_tasks, "H-T")
    both = {r["task_id"] for r in episodes(imp, "H-N") if r["outcome"] == "admitted"} & \
           {r["task_id"] for r in episodes(imp, "H-T") if r["outcome"] == "admitted"}
    n_sup = sum(1 for t in types["types"] if t["domain"] == "supporting")
    n_dom = len({t["domain"] for t in types["types"]})
    target = float(cells[CAT]["target"])
    quotas = {k: float(c["target"]) for k, c in cells.items()}
    at_risk = {k: {x: c.get(x) for x in ("target", "supply", "bound", "status")} |
               {"projection": c["projection"]["whole_supply_units"]}
               for k, c in cells.items() if c.get("status") == "AT_RISK"}
    return {
        "schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PLAN at $0 for the maintainer's review: no suite built, frozen or scored",
        "decided_by": "maintainer 2026-11-04 (review of a143b7ed), section 3: 1,063 / 1,800 accepted as a real "
                      "measurement (29 tasks exhausted, 11 episodes admitted); prepare an independent, versioned "
                      "evaluation plan with the same KL, scored-position and domain-aggregation definitions and frozen "
                      "shared inputs; never marked as meeting the quota; any identity or protocol change bound "
                      "separately",
        "inputs": {p.relative_to(REPO).as_posix(): _sha(p) for p in (IMPORT, PLAN, TYPES)},
        "cell": {"suite": "state_eval_hybrid (tool_calling@v1, non-thinking mode)", "category": CAT,
                 "target_scored_positions": target, "status": cells[CAT].get("status"), "measured": hn},
        "thinking_mode_same_cell": ht | {"admitted_in_both_modes": len(both)},
        "weights": weights(target, hn["scored_positions"], quotas, n_sup, n_dom),
        "uncertainty": {
            "unit": "the episode: 1,063 positions are NOT 1,063 independent observations; the cell's KL is a ratio "
                    "(sum of position KL / positions) over 11 clusters, four of which carry "
                    f"{hn['four_largest_share_pct']}% of the positions",
            "kish_effective_episodes": hn["kish_effective_episodes"],
            "method": "a delete-one-episode jackknife of the cell's position-weighted KL (and of its contribution to "
                      "each objective), t with n - 1 = 10 degrees of freedom; the Kish effective count is reported "
                      "beside it; no normal approximation over positions",
            "registered_comparison": "at the registered 1,800 positions with episodes like these (about 97 positions "
                                     "each) the cell would hold about 19 episodes; its standard error is expected to be "
                                     "about sqrt(19/11) = 1.3x larger here, more where the largest episodes dominate"},
        "decision_rule": {
            "what": "a REPORTING rule beside every ranking that reads this suite; the ranking itself is unchanged",
            "report": ["each objective as defined (equal-domain mean, worst domain, critical-token KL)",
                       "each objective with this cell left out (the tool sub-type pooled over the other six categories' "
                       "positions; nothing else reweighted)",
                       "for the Top-1 against each state it is compared with (the Pareto front and the runner-up), the "
                       "cell's contribution to their difference with its jackknife 90% interval"],
            "flag": "SENSITIVE_TO_THE_LIMITED_CELL when leaving the cell out changes the Top-1 or the Pareto front, or "
                    "when the Top-1's margin on an objective is within that interval of the cell's contribution; a "
                    "flagged decision is reported as such, never re-decided by dropping the cell"},
        "follow_up_validation": [
            "re-score the retained Pareto front on a later suite version with full coverage when a compliant, "
            "admitted source supplies it (the search keeps the front's states; re-scoring is a forward pass, not a "
            "new search)",
            "the trained students' formal measurement reads eval-set-1 (P10.3), where tool calling is reported as its "
            "own unit; it never selects a state",
            "closed-loop tool success belongs to a registered tool evaluation (eval-set-2), not to teacher-forced KL"],
        "identity": {
            "proposed": SUITE,
            "differs_from_the_registered": "state_eval_hybrid@v2 requires each category's share of 12,000 positions "
                                           "per mode and >= 5 tasks; on this pool it is not realizable and stays "
                                           "PENDING, never re-labelled. @v2-lc1 has its own items file, manifest, "
                                           "content sha256 and a coverage declaration naming this cell, its target, its "
                                           "realized positions, tasks and episodes and the status LIMITED_COVERAGE "
                                           "(never REACHED); every other cell meets the v2 rule or the suite does not "
                                           "freeze",
            "unchanged": ["v1's 1,135 items byte for byte", "the type table (full_search_types_v2.json)",
                          "teacher-forced KL against the root teacher, scored positions, the domain aggregation",
                          "the critical groups (tool_call_open / tool_call_close added in both modes) and their 50:50",
                          "the forward context (Stage-1 packing) and the measurement protocol"],
            "not_used": "bounded_plan.approved_shortfalls (no SHORTFALL is approved by this plan)",
            "binding": "a separate maintainer decision registers @v2-lc1; only then does freeze_state_eval gain a "
                       "declared limited-coverage cell (a small code path), and every search that reads it names it"},
        "at_risk_cells": at_risk,
        "at_risk_rule": "monitored in the hybrid session: such a cell runs to its bound; a cell that ends short is "
                        "reported with its realized positions and tasks (SHORTFALL_EXHAUSTED) and handled by the "
                        "existing mechanism -- nothing is pre-approved",
        "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=str(OUT))
    ap.add_argument("--pool", default=str(REPO / "data/stages/stage-2/teacher_modes/tool_pool_v1"))
    a = ap.parse_args(argv)
    doc = build(Path(a.pool))
    Path(a.write).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: doc[k] for k in ("cell", "weights", "uncertainty")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
