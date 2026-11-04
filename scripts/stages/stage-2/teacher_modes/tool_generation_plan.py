#!/usr/bin/env python3
"""The bounded teacher-native tool generation plan: targets, frozen orders, rounds, bounds and price. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_generation_plan \\
        --pool data/stages/stage-2/teacher_modes/tool_pool_v1 --write logs/stages/stage-2/teacher_modes/plans/tool_generation_plan.json

Generation is bounded by what the arms consume, never by the pool's size. The terms are
tool_generation_protocol.json :: bounded_plan:

* cells -- per lineage, mode and split, one target in the unit its consumer counts BEFORE packing: training in unique
  supervised tokens, split over the seven categories by six_category_design.json's shares (unchanged); calibration
  and state evaluation in unique scored positions, by the same shares (state evaluation also >= 5 tasks per
  category the split holds); validation as registered (the hybrid's 32 tasks, Thinking-2507's 12,624 supervised
  tokens). A category that cannot reach its target is a SHORTFALL, never filled from another;
* order -- each cell walks its split's frozen order (pool.json :: orders, verified by `tool_pool.load_pool`) filtered
  to its category, whole tasks only; every teacher walks the same order and stops at its own target;
* rounds -- round 1 takes, per cell, the prefix whose EXPECTED yield reaches the target; each later round takes
  ``ceil(remaining / observed yield per attempted task x 1.1)`` more; at most ``max_rounds``. Each round is one
  engine invocation per mode for its first candidates and one for the second candidates of its refused tasks;
* bound -- a cell never attempts more than the prefix whose PESSIMISTIC yield (the lower tokens per turn and pass
  rate) reaches its target, nor more than its supply: that prefix is what the price's upper bound buys;
* yield -- a task's expected turns are its recorded steps plus one answer per user turn (a gold task: its gold
  steps); expected units per task = turns x tokens per turn (the mode's prior) x the probability that one of two
  candidates is admitted, ``1 - (1 - p)^2``;
* price -- generated tokens / the supplement's measured throughput (scaled by a tool-workload efficiency range),
  plus the first prompts' prefill, a model load per invocation, setup and teardown, at the L40S rate. A pilot's
  measured pass rates, turn lengths and throughput replace the priors before the cap is final.

`round_tasks` is the rule the session (`tool_session.py`) runs; nothing here generates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/tool_generation_protocol.json"
SIX = REPO / "configs/stages/stage-2/teacher_modes/six_category_design.json"
SCHEMA = "aadistill.stage2.teacher_modes.tool_generation_plan/v1"
PLAN = REPO / "logs/stages/stage-2/teacher_modes/plans/tool_generation_plan.json"


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_turns(task: Mapping) -> int:
    """Turns an episode of ``task`` is expected to take: its gold steps, else its recorded steps plus one answer per
    user turn."""
    if task.get("gold_steps"):
        return len(task["gold_steps"])
    rec = (task.get("source") or {}).get("recorded") or {}
    return int(rec.get("steps", 0)) + 1 + len(task.get("later_user_turns") or [])


def admit_probability(p: float, candidates: int) -> float:
    return 1.0 - (1.0 - p) ** candidates


def per_task_units(task: Mapping, unit: str, tpt: float, p: float, candidates: int) -> float:
    """Expected units one attempted task yields."""
    a = admit_probability(p, candidates)
    if unit == "tasks":
        return a
    return a * expected_turns(task) * tpt


def prefix_reaching(tasks: Sequence[Mapping], target: float, unit: str, tpt: float, p: float, candidates: int) -> int:
    """The shortest prefix of ``tasks`` whose expected units reach ``target`` (all of them when none does)."""
    got = 0.0
    for i, t in enumerate(tasks):
        if got >= target:
            return i
        got += per_task_units(t, unit, tpt, p, candidates)
    return len(tasks)


def cell_targets(bp: Mapping, shares: Mapping[str, float], lineage: str, mode: str, splits: Sequence[str]) -> list[dict]:
    """The lineage-mode's cells: ``{split, category, unit, target, min_tasks}``."""
    out = []
    tg = bp["targets"]
    for split in splits:
        spec = tg[split]
        if split == "validation":
            v = spec[mode]
            out.append({"split": split, "category": None, "unit": v["unit"], "target": float(v["value"]),
                        "min_tasks": 0})
            continue
        total = float(spec["per_mode"][mode])
        for cat, share in sorted(shares.items()):
            out.append({"split": split, "category": cat, "unit": spec["unit"], "target": total * share,
                        "min_tasks": int(spec.get("min_tasks_per_category", 0))})
    return out


def cell_order(order: Sequence[str], by_id: Mapping[str, Mapping], category: str | None) -> list[Mapping]:
    return [by_id[t] for t in order if category is None or by_id[t]["category"] == category]


class PlanIdentityError(RuntimeError):
    """A plan that does not belong to the pool, protocol or design it is read with, or whose cells drifted."""


def lineage_cells(proto: Mapping, shares: Mapping[str, float], lineage: str, man: Mapping, by_id: Mapping) -> dict:
    """One lineage's ``{mode: {"cells": [...]}}``: targets, supply, expected tasks and bound per cell."""
    bp, li = proto["bounded_plan"], proto["lineages"][lineage]
    cand = int(proto["execution"]["candidates"])
    modes = {}
    for mode in sorted(li["modes"]):
        tpt = bp["priors"]["supervised_tokens_per_turn"][mode]
        pr = bp["priors"]["episode_pass_rate"]
        cells = []
        for c in cell_targets(bp, shares, lineage, mode, li["splits"]):
            tasks = cell_order(man["orders"].get(c["split"], []), by_id, c["category"])
            tasks_at = {lvl: prefix_reaching(tasks, c["target"], c["unit"], tpt[i], pr[i], cand)
                        for lvl, i in (("lower", 2), ("expected", 1), ("upper", 0))}
            bound = max(tasks_at["upper"], min(len(tasks), c["min_tasks"]))
            reach = sum(per_task_units(t, c["unit"], tpt[0], pr[0], cand) for t in tasks)
            cells.append({**c, "supply": len(tasks), "tasks": tasks_at, "bound": bound,
                          "shortfall_at_pessimistic_yield": reach < c["target"],
                          "expected_turns_in_bound": sum(expected_turns(t) for t in tasks[:bound])})
        modes[mode] = {"cells": cells}
    return modes


def design_shares() -> dict[str, float]:
    return {k: float(v["share"]) for k, v in json.loads(SIX.read_text())["tool_calling"]["categories"].items()}


def build(pool_dir: Path, *, protocol: Mapping | None = None) -> dict:
    from stages.teacher_modes import tool_pool as TP
    proto = protocol or json.loads(PROTOCOL.read_text())
    bp = proto["bounded_plan"]
    shares = design_shares()
    man, splits = TP.load_pool(Path(pool_dir))
    by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
    lineages = {li: {"modes": lineage_cells(proto, shares, li, man, by_id), "splits": proto["lineages"][li]["splits"]}
                for li in sorted(proto["lineages"])}
    plan = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pool": {"dir": str(Path(pool_dir)), "content_sha256": man["content_sha256"],
                     "pool_json_sha256": sha_file(Path(pool_dir) / "pool.json")},
            "protocol_sha256": sha_file(PROTOCOL), "six_category_design_sha256": sha_file(SIX),
            "shares": shares, "rules": bp["rounds"],
            "approved_shortfalls": list(bp.get("approved_shortfalls") or []),
            "lineages": lineages, "authorizes": "nothing"}
    plan["price"] = price(plan, by_id, man, proto)
    return plan


def verified(protocol: Mapping | None = None, pool_dir: Path | None = None, lineage: str | None = None
             ) -> tuple[dict, dict, dict]:
    """``(plan, pool manifest, tasks by id)`` after checking every identity the plan binds, for the session and for
    every dev-box consumer alike: `tool_pool.load_pool` re-verifies the split files, their task counts, sources,
    frozen orders and the pool's content sha256; the plan must name that content, the exact bytes of pool.json, the
    protocol file and the design (the shares) it is read with; and its cells must be the cells this pool, protocol
    and design produce now -- an edited target, bound, order or binding refuses (`PlanIdentityError`). ``lineage``
    restricts the reading and the cell check to that lineage's splits."""
    from stages.teacher_modes import tool_pool as TP
    proto = protocol or json.loads(PROTOCOL.read_text())
    pool_dir = Path(pool_dir) if pool_dir else REPO / proto["pool"]["dir"]
    splits_wanted = proto["lineages"][lineage]["splits"] if lineage else None
    man, splits = TP.load_pool(pool_dir, splits_wanted)
    plan = json.loads((REPO / proto["pool"]["plan"]).read_text())
    bad = []
    if plan.get("schema") != SCHEMA:
        bad.append("not a tool generation plan")
    pool = plan.get("pool") or {}
    if pool.get("content_sha256") != man["content_sha256"]:
        bad.append("built from another pool content")
    if pool.get("pool_json_sha256") != sha_file(pool_dir / "pool.json"):
        bad.append("built from another pool.json (its orders, files or sources differ)")
    if plan.get("protocol_sha256") != sha_file(PROTOCOL):
        bad.append("built under another protocol")
    if plan.get("six_category_design_sha256") != sha_file(SIX) or plan.get("shares") != design_shares():
        bad.append("built under other category shares")
    if plan.get("approved_shortfalls") != list(proto["bounded_plan"].get("approved_shortfalls") or []):
        bad.append("its approved shortfalls are not the protocol's")
    if not bad:
        by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
        for li in ([lineage] if lineage else sorted(proto["lineages"])):
            now = lineage_cells(proto, plan["shares"], li, man, by_id)
            if json.dumps(now, sort_keys=True) != json.dumps((plan.get("lineages") or {}).get(li, {}).get("modes"),
                                                             sort_keys=True):
                bad.append(f"its {li} cells are not the cells this pool, protocol and design produce")
    if bad:
        raise PlanIdentityError("the generation plan refuses: " + "; ".join(bad))
    return plan, man, by_id


#: per price level: (the task count used, the prior index for tokens per turn, for the pass rate, for the throughput
#: ranges). Lower and expected are consistent worlds; upper bounds: the cells' bound prefixes, long turns, the low
#: pass rate (more second candidates), the slow end of every throughput range and every round used.
LEVELS = {"lower": ("lower", 2, 2, 2), "expected": ("expected", 1, 1, 1), "upper": ("bound", 2, 0, 0)}


def price(plan: Mapping, by_id: Mapping, man: Mapping, proto: Mapping, *, rounds: int | None = None) -> dict:
    """lower / expected / upper per lineage session (USD), from the plan's task counts and the protocol's terms. A
    cell may name its tasks (``task_ids``, a pilot's sample) instead of a prefix count; ``rounds`` fixes the engine
    invocations' rounds for every level (a pilot runs one)."""
    bp = proto["bounded_plan"]
    tp = bp["throughput"]
    cand = int(proto["execution"]["candidates"])
    rate = float(bp["rate_usd_per_hour"])
    out = {}
    for lineage, li in plan["lineages"].items():
        row = {}
        for lvl, (count, i_tpt, i_p, i_tp) in LEVELS.items():
            hours = (tp["setup_minutes"] + tp["teardown_minutes"]) / 60
            gen_tokens = prefill = 0.0
            loads = 0
            for mode, m in li["modes"].items():
                tpt = bp["priors"]["supervised_tokens_per_turn"][mode][i_tpt]
                p = bp["priors"]["episode_pass_rate"][i_p]
                retry = 1.0 + (1.0 - p) * (cand - 1)
                mode_tokens = mode_prefill = 0.0
                for c in m["cells"]:
                    n = c["bound"] if count == "bound" else c["tasks"][count]
                    tasks = ([by_id[t] for t in c["task_ids"]] if "task_ids" in c else
                             cell_order(man["orders"].get(c["split"], []), by_id, c["category"])[:n])
                    mode_tokens += sum(expected_turns(t) for t in tasks) * tpt * retry
                    mode_prefill += sum((t.get("first_prompt_tokens") or tp["prompt_tokens_default"])
                                        * expected_turns(t) for t in tasks) * retry
                decode_tps = tp["decode_tokens_per_s"]["thinking" if mode in tp["thinking_modes"] else "non_thinking"]
                hours += mode_tokens / (decode_tps * tp["tool_workload_efficiency"][i_tp]) / 3600
                hours += mode_prefill / tp["prefill_tokens_per_s"][i_tp] / 3600
                r = rounds or (bp["rounds"]["max_rounds"] if count == "bound" else bp["rounds"]["expected_rounds"][count])
                loads += r * cand
                gen_tokens += mode_tokens
                prefill += mode_prefill
            hours += loads * tp["model_load_minutes"][i_tp] / 60
            row[lvl] = {"usd": round(hours * rate, 2), "hours": round(hours, 2), "generated_tokens": int(gen_tokens),
                        "prefill_tokens": int(prefill), "engine_invocations": loads}
        out[lineage] = row
    tot = {lvl: round(sum(out[li][lvl]["usd"] for li in out), 2) for lvl in LEVELS}
    return {"sessions": out, "total_usd": tot, "proposed_hard_cap_usd": round(tot["upper"] * 1.1, 2),
            "rate_usd_per_hour": rate, "levels": {k: list(v) for k, v in LEVELS.items()},
            "basis": "plan task counts (expected / bound) x expected turns x the mode's tokens per turn x (1 + the "
                     "second candidates' share); decode at the supplement's measured throughput x a tool-workload "
                     "efficiency; prefill of each turn's prompt (first prompt x turns; prefix caching would lower "
                     "it); a model load per engine invocation (rounds x candidates per mode); setup and teardown"}


#: The L40S pilot PROPOSED for the maintainer's next review (logs/budget/applications/research_a_tool_pilot_2026-11-04):
#: the hybrid lineage, both modes, one round (first candidates, then second candidates of refused tasks). Per mode:
#: the first ``training_per_category`` tasks of every training cell's frozen order (full episodes, the order's first
#: tier); the first ``single_step_training`` single-step tasks of the schema_arguments training cell; and EVERY
#: calibration and state-evaluation schema_arguments task of both shapes -- their supply is below every target, so
#: generating all of them under the unchanged verifier is what settles whether that category's remaining SHORTFALL is
#: real. Proposal parameters, not a plan term: nothing reads them to generate.
PILOT = {"lineage": "hybrid", "training_per_category": 24, "single_step_training": 100,
         "whole_cells": (("calibration", "schema_arguments"), ("state_eval", "schema_arguments")), "rounds": 1}


def pilot_sample(plan: Mapping, by_id: Mapping, man: Mapping, spec: Mapping = PILOT) -> dict:
    """``{mode: [cell with task_ids]}``: the pilot's tasks, each cell's in its frozen order."""
    from stages.teacher_modes import tool_episodes as TE
    out = {}
    for mode, m in plan["lineages"][spec["lineage"]]["modes"].items():
        cells = []
        for c in m["cells"]:
            order = cell_order(man["orders"].get(c["split"], []), by_id, c["category"])
            if (c["split"], c["category"]) in {tuple(x) for x in spec["whole_cells"]}:
                ids = [t["task_id"] for t in order]
            elif c["split"] == "training":
                full = [t["task_id"] for t in order if TE.episode_shape(t) == TE.FULL_EPISODE]
                one = [t["task_id"] for t in order if TE.episode_shape(t) == TE.SINGLE_STEP]
                ids = full[:spec["training_per_category"]] + one[:spec["single_step_training"]]
            else:
                continue
            if ids:
                cells.append({**c, "task_ids": ids, "bound": len(ids), "tasks": dict.fromkeys(
                    ("lower", "expected", "upper"), len(ids)),
                              "shapes": {s: sum(TE.episode_shape(by_id[t]) == s for t in ids) for s in TE.EPISODE_SHAPES}})
        out[mode] = cells
    return out


def round_tasks(cell: Mapping, order_tasks: Sequence[Mapping], state: Mapping, round_no: int, *,
                bp: Mapping, mode: str, candidates: int) -> list[str]:
    """The task ids a cell attempts in round ``round_no`` (1-based), given its ``state`` after earlier rounds:
    ``{"attempted": int, "units": float, "admitted": int}``. Empty when the cell is done, bounded or out of rounds."""
    if round_no > int(bp["rounds"]["max_rounds"]):
        return []
    attempted, units = int(state.get("attempted", 0)), float(state.get("units", 0.0))
    done = units >= cell["target"] and int(state.get("admitted", 0)) >= cell.get("min_tasks", 0)
    room = min(int(cell["bound"]), len(order_tasks)) - attempted
    if done or room <= 0:
        return []
    if round_no == 1:
        n = int(cell["tasks"]["expected"])
    else:
        rest = order_tasks[attempted:]
        if units > 0 and attempted:
            y = units / attempted
            n = math.ceil(max(cell["target"] - units, 0.0) / y * float(bp["rounds"]["top_up_margin"]))
        else:                                   #: nothing admitted yet: the pessimistic prefix of what is left
            pr, tpt = bp["priors"]["episode_pass_rate"][0], bp["priors"]["supervised_tokens_per_turn"][mode][0]
            n = prefix_reaching(rest, cell["target"] - units, cell["unit"], tpt, pr, candidates)
        if int(state.get("admitted", 0)) < cell.get("min_tasks", 0):
            n = max(n, cell["min_tasks"] - int(state.get("admitted", 0)))
    n = max(1, min(n, room))
    return [t["task_id"] for t in order_tasks[attempted:attempted + n]]


def select(order: Sequence[str], units: Mapping[str, float], target: float, min_tasks: int = 0
           ) -> tuple[list[str], float]:
    """A consumer's selection: admitted tasks (``units``) in the frozen ``order``, whole, until ``target`` units and
    ``min_tasks`` tasks -- the same rule the session stopped generation by."""
    out, got = [], 0.0
    for tid in order:
        if got >= target and len(out) >= min_tasks:
            break
        if tid in units:
            out.append(tid)
            got += units[tid]
    return out, got


def cells_for(plan: Mapping, lineage: str, mode: str, split: str) -> list[dict]:
    return [c for c in plan["lineages"][lineage]["modes"][mode]["cells"] if c["split"] == split]


def for_consumers(protocol: Mapping | None = None, pool_dir: Path | None = None) -> tuple[dict, dict]:
    """``(plan, pool orders)`` for a dev-box consumer (admission's packs, packed inputs), through `verified`: the
    same identity checks the session runs before it generates."""
    plan, man, _ = verified(protocol, pool_dir)
    return plan, man["orders"]


def approved(plan: Mapping, split: str, mode: str, category: str | None) -> dict | None:
    """The maintainer's recorded approval of a cell's SHORTFALL (``bounded_plan.approved_shortfalls``), or None."""
    return next((a for a in plan.get("approved_shortfalls") or []
                 if (a.get("split"), a.get("mode"), a.get("category")) == (split, mode, category)), None)


def pilot_application(plan: Mapping, by_id: Mapping, man: Mapping, proto: Mapping, spec: Mapping = PILOT) -> dict:
    """The L40S pilot's application for the maintainer's next review: the sample (`pilot_sample`), its price by the
    plan's own terms (`price`, one round), its hard cap by the plan's rule, and what it measures. AUTHORIZES NOTHING."""
    sample = pilot_sample(plan, by_id, man, spec)
    pplan = {"lineages": {spec["lineage"]: {"modes": {m: {"cells": cs} for m, cs in sample.items()}}}}
    pr = price(pplan, by_id, man, proto, rounds=spec["rounds"])
    row = pr["sessions"][spec["lineage"]]
    cells = {m: {f"{c['split']}/{c['category']}": {"tasks": len(c["task_ids"]), "shapes": c["shapes"],
                                                    "plan_target": c["target"], "unit": c["unit"],
                                                    "cell_supply": c["supply"]} for c in cs}
             for m, cs in sample.items()}
    return {"schema": "aadistill.budget.application/tool_pilot_v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "PROPOSED for the maintainer's next review; AUTHORIZES NOTHING: no GPU, no pod, no spend",
            "decided_by": "maintainer 2026-11-04 (review of 647eea0e), item 7: prepare the L40S pilot's application "
                          "only; GPU and spend are not authorized",
            "plan": {"pool": plan["pool"], "protocol_sha256": plan["protocol_sha256"],
                     "six_category_design_sha256": plan["six_category_design_sha256"]},
            "sample_rule": {k: (list(map(list, v)) if k == "whole_cells" else v) for k, v in spec.items()},
            "sample": cells, "tasks": sum(len(c["task_ids"]) for cs in sample.values() for c in cs),
            "task_ids": {m: {f"{c['split']}/{c['category']}": c["task_ids"] for c in cs} for m, cs in sample.items()},
            "price": {**row, "basis": pr["basis"] + "; ONE round (first candidates, then the refused tasks' second)",
                      "rate_usd_per_hour": pr["rate_usd_per_hour"]},
            "hard_cap_usd": round(row["upper"]["usd"] * 1.1, 2),
            "hard_cap_rule": "the plan's rule: 1.1 x the upper level (bound task counts, long turns, low pass rate, "
                             "slow ends of every unmeasured range), which already includes setup and teardown",
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--write", default=None)
    ap.add_argument("--pilot-application", default=None,
                    help="write the proposed L40S pilot's quantitative application here (authorizes nothing)")
    a = ap.parse_args(argv)
    plan = build(Path(a.pool))
    if a.pilot_application:
        from stages.teacher_modes import tool_pool as TP
        man, splits = TP.load_pool(Path(a.pool))
        by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
        out = Path(a.pilot_application)
        old = json.loads(out.read_text()) if out.is_file() else {}
        doc = pilot_application(plan, by_id, man, json.loads(PROTOCOL.read_text()))
        doc.update({k: old[k] for k in ("request",) if k in old})      #: the written request, kept on regeneration
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    summary = {li: {m: {f"{c['split']}/{c['category']}": (c["supply"], c["tasks"]["expected"], c["bound"])
                        for c in mm["cells"]} for m, mm in v["modes"].items()} for li, v in plan["lineages"].items()}
    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps({"price": plan["price"]["total_usd"], "cap": plan["price"]["proposed_hard_cap_usd"],
                      "cells": summary}, indent=1)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
