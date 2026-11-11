#!/usr/bin/env python3
"""The hybrid full search's data, type by type and mode by mode, against A1/A2's training data. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.full_search_audit

Maintainer review 2026-11-01, section three: calibration and state evaluation share one (domain, subtype, mode) set,
are the hybrid's own native trajectories in both modes, give every type and mode real scored positions and the two
modes half of the ranking each, and the final audit lists, per type and per mode, the independent tasks, the scored
positions, the source distribution and every shortfall -- no total hiding a thin type. It also maps every training
source to its ability axis and states how the search data's sources differ from the training data's.

Reads (nothing is generated or changed): the type table (full_search_types.json), the profiles and their census
(hybrid_d4_profiles.json, the items files under artifacts/), the calibration code prompts, the state evaluation's
prompt queues and their planning yields (build_state_eval_hybrid.yields), and the admission record. Writes
logs/stages/stage-2/teacher_modes/audit/full_search_data_audit.json.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

TYPES = REPO / "configs/stages/stage-2/teacher_modes/full_search_types.json"
PROFILES = REPO / "configs/stages/stage-2/teacher_modes/hybrid_d4_profiles.json"
SE_TASKS = REPO / "data/stages/stage-2/teacher_modes/tasks/state_eval_hybrid_v1.tasks.jsonl"
CAL_CODE = REPO / "data/stages/stage-2/teacher_modes/tasks/calibration_code_v1.jsonl"
ADMISSION = REPO / "logs/stages/stage-2/teacher_modes/data/token_matched_admission.json"
OUT = REPO / "logs/stages/stage-2/teacher_modes/audit/full_search_data_audit.json"
SCHEMA = "aadistill.stage2.teacher_modes.full_search_data_audit/v1"
MODES = ("H-T", "H-N")
#: below this many independent tasks a (type, mode) is reported THIN: positions alone would hide it
THIN_TASKS = 5
#: a state-evaluation queue whose prompts cover less than this multiple of its expected need is AT RISK
QUEUE_RISK = 1.15


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _source_of(row: dict) -> str:
    return row.get("generator_id") and f"{row['generator_id']}" or row.get("source") or "?"


def _family(row: dict) -> str:
    """A generator item's rule family; a dataset item's dataset (its family id names the row)."""
    fam = row.get("family_id") or row.get("source_item", "").split("#")[0]
    if row.get("generator_id"):
        return fam
    return fam.split("/", 1)[-1].rsplit("-", 1)[0] if "/" in fam else fam or "?"


def calibration(types: list[dict]) -> dict:
    prof = json.loads(PROFILES.read_text())
    out: dict = {}
    for p in prof["profiles"]:
        pid = p["profile_id"]
        items = _jsonl(REPO / p["items_path"])
        census = prof["census"][pid]
        by = defaultdict(list)
        for it in items:
            by[it["subtype"]].append(it)
        for t in types:
            for m in MODES:
                key = f"{t['domain']}/{t['subtype']}|{m}"
                c = census.get(key) or {}
                rows = by.get(f"{t['subtype']}|{m}", [])
                fam = Counter(_family(r) for r in rows)
                out.setdefault(f"{t['subtype']}|{m}", {})[pid] = {
                    "quota": c.get("quota"), "scored_positions": c.get("scored"),
                    "deficit": max(0.0, (c.get("quota") or 0) - (c.get("scored") or 0)),
                    "items": c.get("items"), "independent_tasks": len({r["unit_id"].split("|")[0] for r in rows}),
                    "not_naturally_terminated": c.get("not_naturally_terminated"),
                    "short": bool(c.get("short")), "refill_units": c.get("refill_units"),
                    "sources": dict(fam.most_common())}
    # the union of tasks over both profiles, per (type, mode)
    units = defaultdict(set)
    for p in prof["profiles"]:
        for it in _jsonl(REPO / p["items_path"]):
            units[it["subtype"]].add(it["unit_id"].split("|")[0])
    for key in out:
        out[key]["union_independent_tasks"] = len(units.get(key, ()))
    return out


def state_evaluation(types: list[dict]) -> dict:
    from stages.teacher_modes import build_state_eval_hybrid as BSE
    rows = _jsonl(SE_TASKS)
    queue = defaultdict(list)
    for r in rows:
        queue[r["subtype"]].append(r)
    y = BSE.yields()
    out = {}
    for t in types:
        sub = t["subtype"]
        short_sub = sub.split("@")[0] if sub.split("@")[0] in queue and sub not in queue else sub
        q = queue.get(sub) or queue.get(short_sub) or []
        src = Counter(r.get("family_id") or r.get("source") for r in q)
        for m in MODES:
            nat, per = y.get((sub, m)) or y.get((short_sub, m)) or (None, None)
            need = (BSE.SE_POSITIONS / (nat * per)) if nat and per else None
            out[f"{sub}|{m}"] = {
                "quota_scored_positions": BSE.SE_POSITIONS, "queue_prompts": len(q),
                "planning_yield": {"natural_termination_share": nat, "scored_positions_per_terminated_task": per},
                "expected_tasks_needed": None if need is None else math.ceil(need),
                "queue_over_expected_need": None if need is None else round(len(q) / need, 2),
                "at_risk": None if need is None else len(q) / need < QUEUE_RISK,
                "sources": dict(src.most_common()), "generated": False}
    return out


def training_sources() -> dict:
    """A1/A2's supervised tokens by ability axis and by source (the admission record)."""
    adm = json.loads(ADMISSION.read_text())
    out = {}
    for arm in ("A1", "A2"):
        cells = adm["arms"][arm]["by_cell"]
        tot = sum(cells.values())
        axis, src = Counter(), Counter()
        for c, n in cells.items():
            stratum, source = c.split("/", 1)
            axis[stratum] += n
            src[f"{stratum}/{source}"] += n
        out[arm] = {"by_axis_share": {k: round(v / tot, 4) for k, v in sorted(axis.items())},
                    "by_cell_share": {k: round(v / tot, 4) for k, v in sorted(src.items())}}
    return out


def axis_map(types: list[dict]) -> dict:
    """Every training cell -> its ability axis and whether a search type stands for it."""
    adm = json.loads(ADMISSION.read_text())
    tdoc = json.loads(TYPES.read_text())
    typed = defaultdict(list)
    for t in types:
        for c in t.get("training_cells") or []:
            for cell in re.findall(r"(?:primary|supporting)\.[a-z0-9_]+/[A-Za-z0-9_]+", c):
                typed[cell].append(t["subtype"])
    out = {}
    for c in sorted(adm["quotas"]):
        stratum, source = c.split("/", 1)
        axis = stratum.split(".", 1)[1] if "." in stratum else stratum
        role = "primary (fluid)" if stratum.startswith("primary.") else "supporting (base-corpus counterpart)"
        hit = list(typed.get(c, []))
        why = None
        if not hit:
            why = (" ".join(tdoc["interactive_adaptation"]["why"][:1]) if axis == "interactive_adaptation" else
                   next((v for k, v in tdoc["training_cells_not_typed_here"].items() if source in k), None))
        out[c] = {"ability_axis": axis, "role": role, "search_types": sorted(set(hit)), "typed_in_search": bool(hit),
                  "quota_supervised_tokens": adm["quotas"][c], "why_not_typed": why}
    return out


def distribution_differences(amap: dict) -> dict:
    """How the search data's sources differ from the training data's, per category, by the training quotas."""
    fluid = {c: v for c, v in amap.items() if v["role"].startswith("primary")}
    base = {c: v for c, v in amap.items() if not v["role"].startswith("primary")}
    out = {}
    for name, cells in (("fluid", fluid), ("base_counterpart", base)):
        tot = sum(v["quota_supervised_tokens"] for v in cells.values())
        untyped = {c: round(v["quota_supervised_tokens"] / tot, 4) for c, v in cells.items() if not v["typed_in_search"]}
        out[name] = {"training_share_typed_in_search": round(1 - sum(untyped.values()), 4),
                     "training_share_not_typed": untyped}
    out["notes"] = [
        "the fluid types are the same @v1 project generators as the training cells, on calibration-split families and "
        "the calibration seed range: the generators and rule families' kinds are shared, the instances, families "
        "and seeds are not (isolation by design)",
        "reasoning_core and reasoning_gym (compositional and deduction axes) and interactive adaptation are training "
        "sources with no search type: their axes are represented in the search by the project generators only",
        "MBPP is exhausted by training; the search's code type is reasoning_core's code modules (the code cell's "
        "main training source)",
        "the openmath type is the same type as training's competition math from another source (corpus_v2's "
        "OpenMathInstruct-2 units of base_pool_v1, against MATH train); gsm8k, rag_evidence (SQuAD v2 answerable) and "
        "multihop_qa (HotpotQA) are the training sources' own datasets, rows past every training and calibration row",
        "weights differ by design: training follows D4's increment shares by supervised tokens; calibration D4's "
        "domain weights (base 0.18 x 4 + 0.28, rh 0.22 x 4 + 0.12) split equally over a domain's types and halved "
        "per mode; state evaluation a frozen 12,000 scored positions per (type, mode), reduced equal sub-type then "
        "equal domain"]
    return out


def main(argv=None) -> int:
    types = json.loads(TYPES.read_text())["types"]
    cal = calibration(types)
    se = state_evaluation(types)
    code_prompts = len(_jsonl(CAL_CODE))
    amap = axis_map(types)
    keys = [f"{t['subtype']}|{m}" for t in types for m in MODES]
    table = []
    for k in keys:
        c, s = cal.get(k, {}), se.get(k, {})
        prof = {pid: v for pid, v in c.items() if isinstance(v, dict)}
        thin = [pid for pid, v in prof.items() if (v.get("independent_tasks") or 0) < THIN_TASKS]
        table.append({"type_mode": k, "calibration": c, "state_evaluation": s,
                      "flags": {"calibration_short": sorted(pid for pid, v in prof.items() if v.get("short")),
                                "calibration_thin": thin, "state_evaluation_not_generated": True,
                                "state_evaluation_queue_at_risk": bool(s.get("at_risk"))}})
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "inputs": {p.name: str(p.relative_to(REPO)) for p in (TYPES, PROFILES, SE_TASKS, CAL_CODE, ADMISSION)},
           "rules": {"thin": f"fewer than {THIN_TASKS} independent tasks in a profile's (type, mode)",
                     "queue_at_risk": f"state-evaluation prompts below {QUEUE_RISK} x the expected tasks needed for "
                                      "12,000 scored positions at the planning yield"},
           "summary": {"type_modes": len(keys),
                       "calibration_short": [r["type_mode"] for r in table if r["flags"]["calibration_short"]],
                       "calibration_thin": {r["type_mode"]: r["flags"]["calibration_thin"] for r in table
                                            if r["flags"]["calibration_thin"]},
                       "state_evaluation_queue_at_risk": [r["type_mode"] for r in table
                                                          if r["flags"]["state_evaluation_queue_at_risk"]],
                       "calibration_code_prompts": code_prompts},
           "table": table, "training_axis_map": amap, "distribution_differences": distribution_differences(amap),
           "training_source_shares": training_sources(),
           "authorizes": "nothing"}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
