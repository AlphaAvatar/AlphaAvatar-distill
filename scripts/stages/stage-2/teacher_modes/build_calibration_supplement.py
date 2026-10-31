#!/usr/bin/env python3
"""The hybrid teacher's D4-scale calibration: audit the sealed v1 calibration trajectories, then the units still needed. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_calibration_supplement build \\
        --v1-records <v1 store>/generation --v1-identity <v1 study identity>

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


def cell_quotas(d: dict) -> dict[str, dict[tuple, float]]:
    """profile -> cell -> its D4 quota of scored positions (both modes together)."""
    out = {}
    for name, prof in d["profiles"].items():
        q = {}
        for domain, w in prof["domain_weights"].items():
            subs = d["cells"][domain]
            for sub in subs:
                q[(domain, sub, "*")] = DOSE * float(d["reference_uesp"]) * w / len(subs)
        out[name] = q
    return out


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--v1-records", required=True)
    b.add_argument("--v1-identity", required=True)
    a = ap.parse_args(argv)
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
