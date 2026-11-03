#!/usr/bin/env python3
"""The packed search's inputs: the type table v2, the hybrid profiles v2, state_eval_hybrid@v2 and its freeze. $0.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.packed_inputs types
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.packed_inputs build \\
        --tool-items-ht <H-T tool_admission>/tool_items.jsonl --tool-items-hn <H-N tool_admission>/tool_items.jsonl

The design is `packed_round_design.json` (`calibration_v2`, `state_eval_hybrid_v2`):

* TYPES v2 (`full_search_types_v2.json`): v1's thirteen types plus ``(supporting, tool_calling@v1)``; each mode's
  critical classes add ``tool_call_open`` / ``tool_call_close``. Written from v1's table by `types_v2`, so the
  derivation is the code, not a transcription.
* PROFILES v2: each v1 profile's rule unchanged -- D4's domain weights, a domain's share split equally over its
  sub-types and halved per mode, of the profile's token budget -- with the supporting domain now split over SIX
  sub-types. Every non-supporting cell is v1's, item for item; each existing supporting cell is the PREFIX of v1's
  cell (v1's item order) that first reaches the new quota, whole items; the tool cell takes the hybrid's admitted
  calibration-split tool items (`tool_admission.py`) selected per category by the frozen generation plan
  (`tool_episodes_by_plan`: whole episodes in the pool's frozen calibration order, each category to its share of
  the cell) and then, in that frozen order, whole episodes until the profile's quota (the reasoning-heavy profile's
  is a prefix of the base's). A short cell leaves the profile document's status SHORT and nothing downstream may
  read it.
* STATE_EVAL v2: v1's 1,135 items byte for byte, plus per mode the state_eval split's tool episodes selected by the
  plan the same way (each category to its share of ``SE_POSITIONS`` and at least ``SE_MIN_TASKS`` tasks where the
  split holds them); reached when the total is at least ``SE_POSITIONS``, with every category short of
  ``SE_MIN_TASKS`` listed as THIN. Its manifest declares v2's critical groups and type table. `freeze` writes its record with the measurement protocol id of the packed search's
  evaluator (forward baseline v2 in the design's forward context).

Every item keeps its source fields; nothing generates, and nothing is cut.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

CONF = REPO / "configs/stages/stage-2/teacher_modes"
TYPES_V1, TYPES_V2 = CONF / "full_search_types.json", CONF / "full_search_types_v2.json"
PROFILES_V1, PROFILES_V2 = CONF / "hybrid_d4_profiles.json", CONF / "hybrid_profiles_v2.json"
SUITE_V1 = REPO / "artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v1"
SUITE_V2 = REPO / "artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v2"
ITEMS_V2 = REPO / "artifacts/stages/stage-2/teacher_modes/calibration_v3"
FREEZE_V2 = REPO / "logs/stages/stage-2/teacher_modes/audit/state_eval_hybrid_v2_freeze.json"
DESIGN = CONF / "packed_round_design.json"
TEACHER = CONF / "teacher_identity.json"
MODES = ("H-T", "H-N")
TOOL_SUBTYPE = "tool_calling@v1"
SUPPORTING = "supporting"
SE_POSITIONS, SE_MIN_TASKS = 12_000, 5
#: a quota is a product of float weights (0.28 x 180,000 / 12 = 4200.000000000001): whole positions reaching it
#: within this are reaching it
QUOTA_EPS = 1e-6
CRITICAL_GROUPS_V2 = {"H-T": ["think_close", "final_answer", "eos", "tool_call_open", "tool_call_close"],
                      "H-N": ["final_answer", "eos", "tool_call_open", "tool_call_close"]}


class InputsError(RuntimeError):
    pass


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def scored(item: dict) -> int:
    return len(item["tags"].get("assistant") or [])


def types_v2(v1: dict | None = None) -> dict:
    """v1's type table plus the tool-calling type, its critical classes, as a new version."""
    d = copy.deepcopy(v1 or json.loads(TYPES_V1.read_text()))
    if any(t["subtype"] == TOOL_SUBTYPE for t in d["types"]):
        raise InputsError("the v1 table already holds the tool-calling type")
    d["types"].append({
        "domain": SUPPORTING, "ability_axis": "supporting (P10.2): tool calling, a required agent capability",
        "subtype": TOOL_SUBTYPE,
        "source_type": "tool_pool@v1 (task material from candidate sources under the maintainer's review: "
                       "Toucan-1.5M and the NeMo Gym Pivot pair), the hybrid's own verified episodes; calibration "
                       "and state_eval splits, atom-disjoint from training",
        "training_cells": ["tool_calling: supporting.tool_calling/tool_pool@v1 (training split)"]})
    d["status"] = "v2 (packed round, 2026-11-03): v1's thirteen types plus tool_calling@v1"
    d["critical_tokens"] = {**d["critical_tokens"],
                            "H-T": d["critical_tokens"]["H-T"] + ["tool_call_open (the position predicting the "
                                                                  "generated <tool_call>)",
                                                                  "tool_call_close (the position predicting the "
                                                                  "generated </tool_call>)"],
                            "H-N": d["critical_tokens"]["H-N"] + ["tool_call_open", "tool_call_close"],
                            "groups_v2": CRITICAL_GROUPS_V2}
    d["supersedes"] = {"path": str(TYPES_V1.relative_to(REPO)), "sha256": sha(TYPES_V1)}
    return d


def cell_key(item: dict) -> tuple[str, str]:
    return item["domain"], item["subtype"]


def quotas(weights: dict, token_budget: int, subtypes_by_domain: dict) -> dict:
    """(domain, 'subtype|mode') -> scored positions: a domain's share split equally over its sub-types, halved."""
    out = {}
    for dom, w in weights.items():
        subs = subtypes_by_domain[dom]
        for s in subs:
            for m in MODES:
                out[(dom, f"{s}|{m}")] = w * token_budget / len(subs) / len(MODES)
    return out


def take_whole(items: list[dict], quota: float, key=None) -> tuple[list[dict], int]:
    """Items in order until the scored positions reach ``quota``, grouped by ``key`` (whole episodes)."""
    groups, order = defaultdict(list), []
    for it in items:
        k = key(it) if key else it["item_id"]
        if k not in groups:
            order.append(k)
        groups[k].append(it)
    out, got = [], 0
    for k in order:
        if got >= quota - QUOTA_EPS:
            break
        out += groups[k]
        got += sum(scored(x) for x in groups[k])
    return out, got


def tool_episodes_by_plan(items: list[dict], split: str, mode: str, *, plan: dict | None = None,
                          orders: dict | None = None) -> tuple[list[dict], dict]:
    """``(items, census)``: the mode's admitted ``split`` tool items, whole episodes per category in the pool's
    frozen order up to the plan's cell targets (`tool_generation_plan.select`), returned in the frozen order."""
    from stages.teacher_modes import tool_generation_plan as TGP
    if plan is None:
        plan, orders = TGP.for_consumers()
    pool = [x for x in items if x.get("split") == split]
    by_task, cat_of, units = defaultdict(list), {}, defaultdict(float)
    for x in pool:
        by_task[x["task_id"]].append(x)
        cat_of[x["task_id"]] = x["category"]
        units[x["task_id"]] += scored(x)
    order = list(orders.get(split, []))
    stray = sorted(set(by_task) - set(order))
    if stray:
        raise InputsError(f"{len(stray)} {split} tool episode(s) outside the pool's frozen order, e.g. {stray[:3]}")
    chosen, census = set(), {}
    for c in TGP.cells_for(plan, "hybrid", mode, split):
        ids, got = TGP.select([t for t in order if cat_of.get(t) == c["category"]], units, c["target"],
                              c.get("min_tasks", 0))
        chosen |= set(ids)
        supply = sum(1 for t in by_task if cat_of[t] == c["category"])
        census[c["category"]] = {"target": c["target"], "scored": got, "tasks": len(ids), "admitted_supply": supply,
                                 "short": got < c["target"],
                                 "thin": len(ids) < c.get("min_tasks", 0)}
    return [x for t in order if t in chosen for x in by_task[t]], census


def profiles_v2(tool_items: dict[str, list[dict]], *, out_dir: Path = ITEMS_V2) -> dict:
    """The v2 profile document, and every profile's items written under ``out_dir``."""
    from aadistill.initialization.calibration.profiles import mixture_content_sha256
    v1 = json.loads(PROFILES_V1.read_text())
    tv2 = types_v2()
    subs = defaultdict(list)
    for t in tv2["types"]:
        subs[t["domain"]].append(t["subtype"])
    doc = copy.deepcopy(v1)
    doc["profiles"], doc["census"] = [], {}
    short_all = []
    for p in v1["profiles"]:
        items = jsonl(REPO / p["items_path"])
        if sha(REPO / p["items_path"]) != p["items_file_sha256"]:
            raise InputsError(f"{p['items_path']} is not the registered v1 items file")
        q = quotas(p["domain_weights"], int(p["token_budget"]), subs)
        by_cell = defaultdict(list)
        for it in items:
            by_cell[cell_key(it)].append(it)
        chosen, census = [], {}
        for (dom, sm), quota in sorted(q.items()):
            sub, mode = sm.rsplit("|", 1)
            if dom == SUPPORTING and sub == TOOL_SUBTYPE:
                pool, _ = tool_episodes_by_plan(tool_items.get(mode, []), "calibration", mode)
                got_items, got = take_whole(pool, quota, key=lambda x: x["task_id"])
            elif dom == SUPPORTING:
                got_items, got = take_whole(by_cell[(dom, sm)], quota)
            else:
                got_items = by_cell[(dom, sm)]
                got = sum(scored(x) for x in got_items)
            chosen += got_items
            census[f"{dom}/{sm}"] = {"quota": quota, "scored": got, "items": len(got_items),
                                     "short": got < quota - QUOTA_EPS, "rule": ("v1's cell" if dom != SUPPORTING else
                                                                    "tool items, whole episodes" if sub == TOOL_SUBTYPE
                                                                    else "prefix of v1's cell")}
        short = sorted(k for k, v in census.items() if v["short"])
        short_all += [f"{p['profile_id']}:{k}" for k in short]
        path = Path(out_dir) / f"{p['profile_id']}.v2.items.jsonl"
        write_jsonl(path, chosen)
        e = copy.deepcopy(p)
        e.update(version=2, materialized=True, items_path=str(path.relative_to(REPO)) if path.is_relative_to(REPO)
                 else str(path), items_file_sha256=sha(path), content_sha256=mixture_content_sha256(chosen),
                 description=p["description"] + "; v2: the supporting domain over six sub-types, tool calling "
                                                "added (packed_inputs.profiles_v2)")
        e["metadata"] = {"uesp": sum(scored(x) for x in chosen),
                         "scored_positions_by_mode": {m: sum(scored(x) for x in chosen if x["mode"] == m) for m in MODES},
                         "n_items": len(chosen), "tokens": sum(len(x["ids"]) for x in chosen), "short_cells": short}
        e["sources"] = list(e["sources"]) + [{"dataset_id": "aadistill/hybrid_tool_calling_calibration",
                                              "revision": "1cfa9a7208912126", "domain": SUPPORTING,
                                              "n_items": sum(1 for x in chosen if x["subtype"].startswith(TOOL_SUBTYPE)),
                                              "content_sha256": None,
                                              "license": "the hybrid teacher's own episodes over tool_pool@v1 tasks",
                                              "note": "tool_pool@v1 calibration split"}]
        doc["profiles"].append(e)
        doc["census"][p["profile_id"]] = census
    doc["status"] = ("COMPLETE: every cell reaches its v2 quota" if not short_all
                     else f"SHORT: {short_all}")
    doc["supersedes"] = {"path": str(PROFILES_V1.relative_to(REPO)), "sha256": sha(PROFILES_V1)}
    doc["types"] = {"path": str(TYPES_V2.relative_to(REPO))}
    return doc


def suite_v2(tool_items: dict[str, list[dict]], *, out_dir: Path = SUITE_V2) -> dict:
    """state_eval_hybrid@v2's items and manifest under ``out_dir``; returns the manifest."""
    from aadistill.initialization.calibration.profiles import mixture_content_sha256
    v1m = json.loads((SUITE_V1 / "manifest.json").read_text())
    items = jsonl(SUITE_V1 / "items.jsonl")
    counts = dict(v1m["counts"])
    reached = {}
    thin = {}
    for m in MODES:
        got_items, census = tool_episodes_by_plan(tool_items.get(m, []), "state_eval", m)
        got = sum(scored(x) for x in got_items)
        tasks = len({x["task_id"] for x in got_items})
        thin[m] = sorted(k for k, v in census.items() if v["thin"] and v["admitted_supply"])
        reached[m] = got >= SE_POSITIONS and tasks >= SE_MIN_TASKS
        items += got_items
        counts[f"{TOOL_SUBTYPE}|{m}"] = {"tasks": tasks, "scored_positions": got, "by_category": census}
    man = copy.deepcopy(v1m)
    man["version"] = 2
    man["domains"] = {d: list(s) for d, s in v1m["domains"].items()}
    man["domains"][SUPPORTING] = man["domains"][SUPPORTING] + [f"{TOOL_SUBTYPE}|{m}" for m in MODES]
    man["critical_tags"] = sorted({c for g in CRITICAL_GROUPS_V2.values() for c in g})
    man["critical_groups"] = dict(CRITICAL_GROUPS_V2)
    man["types_sha256"] = None              #: set when the v2 table is committed (`build`)
    man["counts"] = counts
    man["tool_calling"] = {"reached": reached, "thin_categories": thin,
                           "rule": f"the state_eval split's episodes selected per category by the frozen generation "
                                   f"plan (whole episodes, frozen order, each category to its share of "
                                   f"{SE_POSITIONS} scored positions and >= {SE_MIN_TASKS} tasks); reached at >= "
                                   f"{SE_POSITIONS} in total"}
    man["content_sha256"] = mixture_content_sha256(items)
    man["n_items"] = len(items)
    man["supersedes"] = {"suite": f"{v1m['suite_id']}@v{v1m['version']}", "content_sha256": v1m["content_sha256"]}
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "items.jsonl", items)
    man["items_sha256"] = sha(out_dir / "items.jsonl")
    return man


def freeze_record(suite_dir: Path = SUITE_V2, types_path: Path = TYPES_V2) -> dict:
    """The v2 suite's identity under the packed search's evaluator (CPU suffices: every term is a declaration)."""
    from aadistill.initialization.planning.metrics import StateEvaluator
    from aadistill.initialization.specs.metrics import ReferenceStrategy
    from shared import forward_baseline as FB
    from stages.teacher_modes import hybrid_state_eval as HSE
    from stages.teacher_modes import packed_full_search as PFS
    ident = json.loads(TEACHER.read_text())
    suite, items, man = HSE.load(suite_dir, ident, types_path, critical_groups=CRITICAL_GROUPS_V2)
    b = FB.selected(REPO)
    terms = FB.session_terms(b)
    fc = PFS.forward_context(b)
    ev = StateEvaluator(suite, items, device="cpu", vocab_size=int(ident["architecture"]["vocab_size"]),
                        suite_content_sha256=man["content_sha256"], position_policy=terms["position_policy"],
                        distribution_support=terms["distribution_support"], execution=terms["execution"],
                        numerics=terms["numerics"], reference_strategy=ReferenceStrategy.RECOMPUTE,
                        forward_context=fc)
    return {"schema": "aadistill.stage2.teacher_modes.state_eval_hybrid_freeze/v2",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "suite_identity": {"suite_id": f"{man['suite_id']}@v{man['version']}",
                               "content_sha256": man["content_sha256"], "items_sha256": man["items_sha256"],
                               "suite_hash": suite.suite_hash, "measurement_protocol_id": ev.measurement_protocol_id,
                               "reduction": ev.reduction.as_dict(), "baseline": b.qualified_id,
                               "forward_context": fc.as_dict()},
            "tool_calling": man.get("tool_calling"), "authorizes": "nothing"}


def build(tool_items: dict[str, list[dict]], *, profiles_out: Path = PROFILES_V2, suite_out: Path = SUITE_V2,
          items_out: Path = ITEMS_V2, types_out: Path = TYPES_V2, freeze_out: Path = FREEZE_V2) -> dict:
    Path(types_out).write_text(json.dumps(types_v2(), indent=1) + "\n")
    doc = profiles_v2(tool_items, out_dir=items_out)
    doc["types"] = {"path": str(Path(types_out)), "sha256": sha(types_out)}
    Path(profiles_out).write_text(json.dumps(doc, indent=1) + "\n")
    man = suite_v2(tool_items, out_dir=suite_out)
    man["types_sha256"] = sha(types_out)
    (Path(suite_out) / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    out = {"profiles_status": doc["status"], "suite_tool_calling": man["tool_calling"]}
    if doc["status"].startswith("COMPLETE") and all(man["tool_calling"]["reached"].values()):
        rec = freeze_record(suite_out, types_out)
        Path(freeze_out).parent.mkdir(parents=True, exist_ok=True)
        Path(freeze_out).write_text(json.dumps(rec, indent=1) + "\n")
        out["measurement_protocol_id"] = rec["suite_identity"]["measurement_protocol_id"]
    else:
        out["frozen"] = False
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("types")
    b = sub.add_parser("build")
    b.add_argument("--tool-items-ht", required=True)
    b.add_argument("--tool-items-hn", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "types":
        TYPES_V2.write_text(json.dumps(types_v2(), indent=1) + "\n")
        print(TYPES_V2)
        return 0
    res = build({"H-T": jsonl(Path(a.tool_items_ht)), "H-N": jsonl(Path(a.tool_items_hn))})
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
