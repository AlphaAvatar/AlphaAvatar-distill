#!/usr/bin/env python3
"""The hybrid teacher's D4-scale calibration: audit the sealed v1 calibration trajectories, then the units still needed. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_calibration_supplement build \\
        --v1-records <v1 store>/generation --v1-identity <v1 study identity>
    # after the supplement session: both hybrid D4 profiles from the sealed records
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_calibration_supplement profiles \\
        --v1-records <v1 store>/generation --v1-identity <id> --records <supplement store>/generation --identity <id>

D4's profiles (``calib.s0v_base_d4@v1``, ``calib.s0v_rh_d4@v1``; configs/stages/stage-0/calibration_volume/
formal_design.json and profiles_formal.json) are a stratified nested sample of ``base_pool_v1``: per cell (one
subtype, families pooled), the cell's units in a seed-keyed order (seed 1308798386, `build_volume_profiles.order`),
taken from the front until the cell's quota of scored positions -- 4 x 45,000 x the domain weight, split over the
domain's subtypes -- is met. The two profiles share each cell's order.

The hybrid's profiles keep all of that -- cells, order, quotas, the two weightings -- and replace the positions:
each mode's own scored positions (the generated tokens of its naturally terminated trajectory on the unit's
prompt), and each mode fills HALF of every cell's quota, so the two modes contribute equal scored positions to
every cell. Research A v1 already generated both modes on every unit of D4's 4x prefix (the sealed ``cal_``
groups); `audit` measures them against the quotas and `build` lists only the units beyond that prefix, in the
cell's D4 order, that a (mode, cell) still needs, with margin ``MARGIN`` on the measured positions per unit.

ISOLATION. A new calibration unit's prompt shares no key with an eval-set-1 held-out unit and no 13-gram copy
(R1, R2), its rule family is not a fluid evaluation family or one R9 refuses (R3, R9), and it is held out of
every training order: order_v2 is built with these prompt keys excluded, exactly as with the v1 calibration
prompts. Reference logits and activations are never stored here: the search computes them from the hybrid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.teacher_modes import hybrid_calibration as CAL  # noqa: E402

DESIGN = REPO / "configs/stages/stage-0/calibration_volume/formal_design.json"
POOL = REPO / "artifacts/stages/stage-0/calibration_volume/base_pool_v1"
V1_INPUTS = REPO / "logs/stages/stage-2/teacher_modes/plans/calibration_inputs/calibration_input_g0.jsonl"
OUT = REPO / "data/stages/stage-2/teacher_modes/tasks/calibration_v2.jsonl"
AUDIT = REPO / "logs/stages/stage-2/teacher_modes/audit/calibration_coverage.json"
PROFILES = {"base": "calib.s0v_base_d4@v1", "rh": "calib.s0v_rh_d4@v1"}
DOSE, SEED = 4, 1308798386
MODES = ("H-T", "H-N")
MARGIN = 1.3
SCHEMA = "aadistill.stage2.teacher_modes.calibration_supplement/v1"


def design() -> dict:
    return json.loads(DESIGN.read_text())


TYPES = REPO / "configs/stages/stage-2/teacher_modes/full_search_types.json"
CODE_TASKS = REPO / "data/stages/stage-2/teacher_modes/tasks/calibration_code_v1.jsonl"


def cell_quotas(d: dict, types: list[dict] | None = None) -> dict[str, dict[tuple, float]]:
    """profile -> cell -> its D4 quota of scored positions (both modes together). With the full-search type table,
    a domain's share is split over the table's sub-types for it (supporting: D4's four and code)."""
    subs_of = defaultdict(list)
    for t in types or []:
        subs_of[t["domain"]].append(t["subtype"])
    out = {}
    for name, prof in d["profiles"].items():
        q = {}
        for domain, w in prof["domain_weights"].items():
            subs = subs_of.get(domain) or list(d["cells"][domain])
            for sub in subs:
                q[(domain, sub, "*")] = DOSE * float(d["reference_uesp"]) * w / len(subs)
        out[name] = q
    return out


def r9_refused(generator_id, family_id) -> bool:
    from stages.teacher_modes.build_state_eval_hybrid import r9_refused as by_name
    return by_name(generator_id, family_id)


def pool_order(d: dict) -> tuple[dict, dict[tuple, list[str]]]:
    import build_volume_profiles as BVP
    units = BVP.load_units(POOL)
    cm = BVP.cells(d, units)
    return units, {c: BVP.order(c, uids, SEED) for c, uids in cm.items()}


def v1_positions(store: Path, identity: str) -> dict[tuple[str, str], int | None]:
    """(unit id, mode) -> the hybrid's scored positions on the unit's prompt (None: not naturally terminated)."""
    from stages.reasoning_data import generation_driver as GD
    meta = {r["task_id"]: r["calibration"] for r in A.jsonl(V1_INPUTS)}
    unit_of = {it["item_id"]: it["unit_id"] for prof in CAL.PROFILES for it in CAL.items(prof)}
    out = {}
    for mode in MODES:
        for tid, rec in GD.sealed(store, f"cal_{mode}_pass0_cal_g0", identity).items():
            t = rec["turns"][-1]
            uid = unit_of[meta[tid]["item_id"]]
            n = int(t["generated_tokens"]) if t.get("stop_reason") == "eos" else None
            if out.get((uid, mode)) is None:
                out[(uid, mode)] = n
    return out


def plan(store: Path, identity: str) -> dict:
    d = design()
    quotas = cell_quotas(d)
    units, order = pool_order(d)
    have = v1_positions(store, identity)
    per_unit = defaultdict(list)
    for (uid, mode), n in have.items():
        if n is not None and uid in units:
            u = units[uid]
            per_unit[(mode, u["domain"], u["subtype"])].append(n)
    audit, need = {}, defaultdict(set)
    for prof, q in quotas.items():
        for cell, quota in sorted(q.items()):
            for mode in MODES:
                half = quota / 2
                got, used, new = 0, 0, []
                mean = per_unit[(mode, cell[0], cell[1])]
                est = sum(mean) / len(mean) if mean else 0.0
                for uid in order[cell]:
                    if got >= half * (MARGIN if new else 1.0):
                        break
                    if (uid, mode) in have:
                        got += have[(uid, mode)] or 0
                        used += 1
                    else:
                        new.append(uid)
                        got += est
                audit[f"{prof}|{mode}|{'/'.join(cell[:2])}"] = {
                    "quota_per_mode": round(half, 1),
                    "v1_positions": sum(have[(u, mode)] or 0 for u in order[cell] if (u, mode) in have),
                    "v1_units": sum(1 for u in order[cell] if (u, mode) in have),
                    "positions_per_unit_v1": round(est, 1), "new_units": len(new), "pool_units": len(order[cell]),
                    "short_after_pool": got < half}
                for uid in new:
                    need[uid].add(mode)
    return {"quotas": {p: {"/".join(c[:2]): v for c, v in q.items()} for p, q in quotas.items()}, "audit": audit,
            "need": {u: sorted(m) for u, m in sorted(need.items())}, "units": units}


def tasks(p: dict) -> list[dict]:
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoTokenizer
    m, rev = CAL.SOURCE_TOKENIZER
    tok = AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE) / f"models--{m.replace('/', '--')}" / "snapshots" / rev)
    out = []
    for uid, modes in p["need"].items():
        u = p["units"][uid]
        it = u["items"][0]
        msgs = CAL.prompt_messages(it, tok)
        out.append({"task_id": "calib2_" + hashlib.sha256(f"base_pool_v1|{uid}".encode()).hexdigest()[:20],
                    "messages": [x for x in msgs if x["role"] != "system"],
                    "system": msgs[0]["content"] if msgs[0]["role"] == "system" else "You are a helpful Assistant.",
                    "generator_id": it.get("generator_id") or it["source"], "family_id": it.get("family_id") or it["source"],
                    "axis": it["domain"], "difficulty": str(it.get("difficulty") or ""), "interactive": False,
                    "modes": modes, "source": it["source"],
                    "calibration": {"pool": "base_pool_v1", "unit_id": uid, "item_id": it["item_id"],
                                    "domain": it["domain"], "subtype": it["subtype"]}})
    return out


def isolation(rows: list[dict]) -> dict:
    """R1 (held-out prompt keys), R2 (13-gram copies) on the prompts; R3/R9 on the rule families."""
    from shared import eval_sets as ES
    from stages.reasoning_data import isolation_check as IC
    held = {ES.prompt_key(ES.prompt_text(r)) for rs in ES.component_items(roles=ES.HELD_OUT_ROLES).values()
            for r in rs if ES.prompt_text(r).strip()}
    r1 = [r["task_id"] for r in rows if ES.prompt_key(ES.prompt_text(r)) in held]
    items = [{**r, "unit_id": r["task_id"], "source": "stage0_calibration"} for r in rows]
    rep = IC.check_pool(items)
    r2 = rep["results"]["R2_near_duplicate"]["flagged_ids"]
    r3 = rep["results"].get("R3_rule_family", {})
    r9 = rep["results"]["R9_mechanism_equivalence"]["refused_families"]
    fam9 = [r["task_id"] for r in rows if f"{r['generator_id']}/{r['family_id']}" in set(r9)]
    return {"R1_held_out_prompt_keys": r1, "R2_flagged": r2, "R3": {k: v for k, v in r3.items() if k != "examples"},
            "R9_refused_family_units": fam9, "pass": not (r1 or r2 or fam9)}


def keys(rows: list[dict]) -> set[str]:
    from shared import eval_sets as ES
    return {ES.prompt_key(ES.prompt_text(r)) for r in rows}


PROFILE_DIR = REPO / "artifacts/stages/stage-2/teacher_modes/calibration_v2"
PROFILE_DOC = REPO / "configs/stages/stage-2/teacher_modes/hybrid_d4_profiles.json"
HYBRID_IDS = {"base": "calib.hybrid_d4_base", "rh": "calib.hybrid_d4_rh"}


def hybrid_records(v1_store: Path, v1_identity: str, store: Path | None, identity: str | None) -> dict:
    """(unit id, mode) -> the hybrid's sealed calibration record: v1's cal_ groups, then the supplement's s00 groups."""
    from stages.reasoning_data import generation_driver as GD
    meta = {r["task_id"]: r["calibration"] for r in A.jsonl(V1_INPUTS)}
    unit_of = {it["item_id"]: it["unit_id"] for prof in CAL.PROFILES for it in CAL.items(prof)}
    out = {}
    for mode in MODES:
        for tid, rec in GD.sealed(v1_store, f"cal_{mode}_pass0_cal_g0", v1_identity).items():
            out.setdefault((unit_of[meta[tid]["item_id"]], mode), rec)
    if store is not None:
        cal = {r["task_id"]: r["calibration"]["unit_id"] for r in A.jsonl(OUT)}
        for mode in MODES:
            label = f"s00{'t' if mode == 'H-T' else 'n'}_{mode}_pass0_s00{'t' if mode == 'H-T' else 'n'}_g0"
            if (Path(store) / f"{label}.complete").is_file():
                for tid, rec in GD.sealed(Path(store), label, identity).items():
                    out.setdefault((cal[tid], mode), rec)
    return out


def item_of(rec: dict, unit: dict, mode: str, stop_ids: set[int]) -> dict | None:
    """hybrid_calibration's item format: the trajectory exactly as generated, None unless naturally terminated."""
    turn = rec["turns"][0]
    p, o = turn.get("prompt_token_ids") or [], turn.get("output_token_ids") or []
    if not p or not o or int(o[-1]) not in stop_ids:
        return None
    ids = list(p) + list(o)
    it = unit["items"][0]
    return {"item_id": f"{unit['unit_id']}|{mode}", "unit_id": f"{unit['unit_id']}|{mode}", "ids": ids,
            "n_tokens": len(ids), "n_prediction_positions": len(ids) - 1,
            "tags": {"assistant": list(range(len(p) - 1, len(ids) - 1))}, "domain": it["domain"],
            "subtype": f"{it['subtype']}|{mode}", "mode": mode, "source": "hybrid_teacher_native",
            "source_item": it["item_id"], "generator_id": it.get("generator_id"), "family_id": it.get("family_id"),
            "turn": 0}


def held_out_of_profiles(units: dict) -> dict[str, str]:
    """unit id -> why it may not enter a hybrid profile: R9 by family name, or its prompt is a training prompt of
    either lineage (A1/A2's sets or C1's D4-Augmented sessions)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoTokenizer

    from shared import eval_sets as ES
    from stages.teacher_modes.build_state_eval_hybrid import training_sets
    train = {ES.prompt_key(ES.prompt_text(t)) for rows in training_sets().values() for t in rows}
    m, rev = CAL.SOURCE_TOKENIZER
    tok = AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE) / f"models--{m.replace('/', '--')}" / "snapshots" / rev)
    out = {}
    for uid, u in units.items():
        it = u["items"][0]
        if r9_refused(it.get("generator_id"), it.get("family_id")):
            out[uid] = "R9 (family name)"
        elif "ids" in it:
            msgs = CAL.prompt_messages(it, tok)
            if ES.prompt_key(ES.prompt_text({"messages": msgs})) in train:
                out[uid] = "a training prompt"
    return out


def profiles(v1_store: Path, v1_identity: str, store: Path | None, identity: str | None) -> dict:
    """Both hybrid D4 profiles: per cell and mode, D4's order, half the cell quota of the mode's own positions; units
    held out (`held_out_of_profiles`) are passed over and the order continues. A (profile, mode, cell) still short
    lists the next units of its order a calibration top-up would generate (``refill``)."""
    from aadistill.initialization.calibration.profiles import mixture_content_sha256
    d = design()
    types = json.loads(TYPES.read_text())["types"]
    quotas = cell_quotas(d, types)
    units, order = pool_order(d)
    if CODE_TASKS.is_file():                #: the code cell: its own prompt list, in file order (no D4 pool unit)
        code = list(A.jsonl(CODE_TASKS))
        order[("supporting", "code", "*")] = [t["task_id"] for t in code]
        for t in code:
            units[t["task_id"]] = {"unit_id": t["task_id"], "items": [{"item_id": t["task_id"], "domain": "supporting",
                                   "subtype": "code", "generator_id": t.get("module"), "family_id": t.get("family_id")}]}
    recs = hybrid_records(v1_store, v1_identity, store, identity)
    held = held_out_of_profiles(units)
    per_unit = defaultdict(list)
    for (uid, mode), rec in recs.items():
        t = rec["turns"][0]
        if t.get("stop_reason") == "eos" and uid in units:
            u = units[uid]["items"][0]
            per_unit[(mode, u["subtype"])].append(int(t["generated_tokens"]))
    refill = defaultdict(set)
    proto = json.loads((REPO / "configs/stages/stage-2/teacher_modes/generation_protocol.json").read_text())
    stop = {int(x) for x in proto["teacher"]["stop_ids"]}
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    doc, report = [], {}
    for prof, q in quotas.items():
        items, census = [], {}
        for cell, quota in sorted(q.items()):
            for mode in MODES:
                got, taken, skipped = 0, 0, 0
                refused, est, new = 0, 0.0, []
                ppu = per_unit[(mode, cell[1])]
                mean = sum(ppu) / len(ppu) if ppu else 0.0
                for uid in order.get(cell, []):
                    if got + est >= quota / 2 and (taken or new):
                        break
                    if uid in held:
                        refused += 1        #: R9 by family name, or a training prompt of either lineage
                        continue
                    rec = recs.get((uid, mode))
                    if rec is None:
                        new.append(uid)     #: not generated in this mode: a top-up's next unit
                        est += mean * MARGIN if mean else quota
                        continue
                    it = item_of(rec, units[uid], mode, stop)
                    if it is None:
                        skipped += 1
                        continue
                    items.append(it)
                    got += len(it["tags"]["assistant"])
                    taken += 1
                census[f"{'/'.join(cell[:2])}|{mode}"] = {"quota": round(quota / 2, 1), "scored": got, "items": taken,
                                                          "not_naturally_terminated": skipped, "held_out_units": refused,
                                                          "short": got < quota / 2, "refill_units": len(new)}
                for uid in new:
                    refill[uid].add(mode)
        pid = HYBRID_IDS[prof]
        path = PROFILE_DIR / f"{pid}.items.jsonl"
        body = "".join(json.dumps(it, sort_keys=True) + "\n" for it in items)
        path.write_text(body)
        pos = {m: sum(len(i["tags"]["assistant"]) for i in items if i["mode"] == m) for m in MODES}
        doc.append({"profile_id": pid, "version": 1,
                    "description": f"the hybrid's D4 {prof} profile: D4's cells, order and quotas, half of every cell from each mode",
                    "domain_weights": d["profiles"][prof]["domain_weights"], "token_budget": int(DOSE * d["reference_uesp"]),
                    "seed": SEED, "role": "operator_calibration", "materialized": True,
                    "items_path": str(path.relative_to(REPO)), "items_file_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "content_sha256": mixture_content_sha256(items) if items else None,
                    "metadata": {"uesp": sum(pos.values()), "scored_positions_by_mode": pos, "n_items": len(items),
                                 "tokens": sum(i["n_tokens"] for i in items),
                                 "short_cells": sorted(k for k, v in census.items() if v["short"])}})
        report[pid] = census
    short = sorted(f"{p['profile_id']}:{c}" for p in doc for c in p["metadata"]["short_cells"])
    rec = {"schema": "aadistill.calibration_profiles/v1",
           "status": ("COMPLETE: every cell holds half its D4 quota in each mode" if not short else
                      f"INCOMPLETE: {len(short)} (profile, cell, mode) short until the top-up's units are generated"),
           "_contract": ("Research A v2's hybrid D4-scale calibration profiles (build_calibration_supplement.py profiles); data "
                         "for the PROPOSED full search -- nothing searches with them before the maintainer's approval"),
           "teacher": {"model": proto["teacher"]["model"], "revision": proto["teacher"]["revision"]},
           "profiles": doc, "census": report,
           "held_out": dict(Counter(held.values())),
           "refill": {"units": len(refill), "by_cell_and_modes": dict(Counter(
               f"{units[u]['items'][0]['domain']}/{units[u]['items'][0]['subtype']}|{'+'.join(sorted(m))}" for u, m in refill.items())),
               "unit_ids": {u: sorted(m) for u, m in sorted(refill.items())}},
           "authorizes": "nothing"}
    PROFILE_DOC.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--v1-records", required=True)
    b.add_argument("--v1-identity", required=True)
    pr = sub.add_parser("profiles")
    pr.add_argument("--v1-records", required=True)
    pr.add_argument("--v1-identity", required=True)
    pr.add_argument("--records", default=None)
    pr.add_argument("--identity", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "profiles":
        rec = profiles(Path(a.v1_records), a.v1_identity, Path(a.records) if a.records else None, a.identity)
        print(json.dumps({p["profile_id"]: p["metadata"] for p in rec["profiles"]}, indent=1))
        return 0
    p = plan(Path(a.v1_records), a.v1_identity)
    rows = tasks(p)
    iso = isolation(rows)
    bad = set(iso["R1_held_out_prompt_keys"]) | set(iso["R2_flagged"]) | set(iso["R9_refused_family_units"])
    kept = [r for r in rows if r["task_id"] not in bad]
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(REPO)} exists: a frozen task list is never rebuilt")
    OUT.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in kept))
    by = Counter((r["calibration"]["domain"], r["calibration"]["subtype"], "|".join(r["modes"])) for r in kept)
    rec = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "d4": {"design": {"path": str(DESIGN.relative_to(REPO)), "sha256": A.sha256_file(DESIGN)},
                  "pool_items_sha256": A.sha256_file(POOL / "items.jsonl"), "dose": DOSE, "sampling_seed": SEED,
                  "profiles": PROFILES},
           "rule": ("per profile, mode and cell: D4's cell order, the mode's own naturally terminated scored positions, "
                    "half of D4's cell quota per mode; v1's sealed calibration covers D4's 4x prefix"),
           "v1_records": {"dir": a.v1_records, "identity": a.v1_identity, "inputs_sha256": A.sha256_file(V1_INPUTS)},
           "quotas": p["quotas"], "audit": p["audit"],
           "new_units": len(kept), "by_cell_and_modes": {"/".join(k[:2]) + f"|{k[2]}": v for k, v in sorted(by.items())},
           "removed_by_isolation": sorted(bad), "isolation": iso,
           "output": {"path": str(OUT.relative_to(REPO)), "sha256": A.sha256_file(OUT)},
           "held_out_of_training": "order_v2 excludes these prompt keys (build_supplement_tasks.isolate)",
           "authorizes": "nothing"}
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"new_units": len(kept), "removed": len(bad), "by": rec["by_cell_and_modes"],
                      "short_after_pool": [k for k, v in p["audit"].items() if v["short_after_pool"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
