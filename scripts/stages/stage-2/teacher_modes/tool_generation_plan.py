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


def build(pool_dir: Path, *, protocol: Mapping | None = None) -> dict:
    from stages.teacher_modes import tool_pool as TP
    proto = protocol or json.loads(PROTOCOL.read_text())
    bp = proto["bounded_plan"]
    shares = {k: float(v["share"]) for k, v in json.loads(SIX.read_text())["tool_calling"]["categories"].items()}
    man, splits = TP.load_pool(Path(pool_dir))
    by_id = {t["task_id"]: t for rows in splits.values() for t in rows}
    cand = int(proto["execution"]["candidates"])
    lineages = {}
    for lineage, li in sorted(proto["lineages"].items()):
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
        lineages[lineage] = {"modes": modes, "splits": li["splits"]}
    plan = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pool": {"dir": str(Path(pool_dir)), "content_sha256": man["content_sha256"],
                     "pool_json_sha256": sha_file(Path(pool_dir) / "pool.json")},
            "protocol_sha256": sha_file(PROTOCOL), "six_category_design_sha256": sha_file(SIX),
            "shares": shares, "rules": bp["rounds"], "lineages": lineages, "authorizes": "nothing"}
    plan["price"] = price(plan, by_id, man, proto)
    return plan


#: per price level: (the task count used, the prior index for tokens per turn, for the pass rate, for the throughput
#: ranges). Lower and expected are consistent worlds; upper bounds: the cells' bound prefixes, long turns, the low
#: pass rate (more second candidates), the slow end of every throughput range and every round used.
LEVELS = {"lower": ("lower", 2, 2, 2), "expected": ("expected", 1, 1, 1), "upper": ("bound", 2, 0, 0)}


def price(plan: Mapping, by_id: Mapping, man: Mapping, proto: Mapping) -> dict:
    """lower / expected / upper per lineage session (USD), from the plan's task counts and the protocol's terms."""
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
                    tasks = cell_order(man["orders"].get(c["split"], []), by_id, c["category"])[:n]
                    mode_tokens += sum(expected_turns(t) for t in tasks) * tpt * retry
                    mode_prefill += sum((t.get("first_prompt_tokens") or tp["prompt_tokens_default"])
                                        * expected_turns(t) for t in tasks) * retry
                decode_tps = tp["decode_tokens_per_s"]["thinking" if mode in tp["thinking_modes"] else "non_thinking"]
                hours += mode_tokens / (decode_tps * tp["tool_workload_efficiency"][i_tp]) / 3600
                hours += mode_prefill / tp["prefill_tokens_per_s"][i_tp] / 3600
                rounds = bp["rounds"]["max_rounds"] if count == "bound" else bp["rounds"]["expected_rounds"][count]
                loads += rounds * cand
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
    """``(plan, pool orders)`` for a dev-box consumer: the committed plan, checked to name the pool's content."""
    proto = protocol or json.loads(PROTOCOL.read_text())
    pool_dir = Path(pool_dir) if pool_dir else REPO / proto["pool"]["dir"]
    man = json.loads((pool_dir / "pool.json").read_text())
    plan = json.loads((REPO / proto["pool"]["plan"]).read_text())
    if plan["pool"]["content_sha256"] != man.get("content_sha256"):
        raise ValueError("the generation plan was built from another pool than the one the consumer reads")
    return plan, man["orders"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    plan = build(Path(a.pool))
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
