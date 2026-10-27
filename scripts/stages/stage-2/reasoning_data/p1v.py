#!/usr/bin/env python3
"""P1v: one P1 run's secured records -> verified, selected, packed arm B. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.p1v \
        --run-id <run> --store <scr>/store --pack-out artifacts/stages/stage-2/reasoning_data/packs/B

1. INPUTS. Every secured shard is re-hashed against its marker, every record
   must carry this run's launch identity, and each pass group's ``.complete``
   must list exactly the shards present.
2. VERIFY AND SELECT. `generation_adapter.adapt` runs over the plan's pools
   and every record: explicit unique candidate identity, record consistency
   with the tokenizer, every check apart, code in the probed sandbox only, the
   lowest accepted candidate per unit. Per (stratum, source) cell it reports
   tasks, candidates, accepted units and rejection reasons.
3. ISOLATION RECHECK. The selected units' tasks, carrying their TEACHER TEXT,
   go through `isolation_check.check_pool` (R0-R9; R1/R2 now on prompts and
   outputs).
4. SUPPLY AGAINST THE CELLS. Units are rendered and admitted whole
   (`build_master_ladder.render_all`); each cell's supervised tokens are set
   beside its share of 2.96M plus the reserve. A cell that falls short is a
   GAP, reported, and the pack is NOT built: no paid top-up, no lower gate, no
   repetition, no other source, no other mix (maintainer 2026-10-22).
5. PACK AND M1. With every cell covered, `build_arm_b` writes B's real pack.
   Run `h1_accounting --pack-b` and `forecast_cost` on it next.

Writes ``logs/stages/stage-2/reasoning_data/audit/p1v_verification.json`` and
the selected sessions (``<pack-out>/../B_sessions.jsonl``, gitignored with the
pack).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_arm_b as AB
from stages.reasoning_data import build_master_ladder as B
from stages.reasoning_data import generation_adapter as G
from stages.reasoning_data import governance as GOV
from stages.reasoning_data import isolation_check as IC
from stages.reasoning_data import source_plan as SP

OUT = A.LOG_DIR / "audit/p1v_verification.json"
TARGET = 2_960_000


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inputs(store: Path, lid: str) -> tuple[list[dict], dict]:
    shards = sorted(store.glob("pass*.part*.jsonl"))
    bad = [s.name for s in shards if not s.with_suffix(".done").is_file()
           or _sha(s) != s.with_suffix(".done").read_text().strip()]
    if bad:
        raise SystemExit(f"shards that do not hash to their markers: {bad[:5]}")
    recs = [r for s in shards for r in A.jsonl(s)]
    foreign = [r["task_id"] for r in recs if r["generation_pass"]["launch_identity_sha256"] != lid]
    if foreign:
        raise SystemExit(f"{len(foreign)} records were generated under another launch identity")
    groups = {}
    for c in sorted(store.glob("*.complete")):
        doc = json.loads(c.read_text())
        listed = {s["name"]: s["sha256"] for s in doc["shards"]}
        have = {s.name: _sha(s) for s in shards if s.name.startswith(doc["label"] + ".part")}
        groups[doc["label"]] = {"tasks": doc["tasks"], "complete_and_matching": listed == have}
    return recs, {"shards": len(shards), "records": len(recs), "groups": groups,
                  "by_pass": dict(Counter(r["candidate_index"] for r in recs)),
                  "shard_sha256": {s.name: _sha(s) for s in shards}}


def cell_of(task: dict, spec: dict) -> str:
    return f"{task.get('stratum') or SP.stratum_of(task, spec)}/{task.get('source')}"


def verify_run(run_id: str, store: Path) -> dict:
    """Steps 1-2 for one run (P1 or P1t): its secured inputs, verified and
    selected, against the plan its authorization's phase names -- whose launch
    identity on disk must still be the authorized one. Every later reader of a
    run's accepted units (this file, `corpus_feasibility`) takes them from here,
    never from a stored selection."""
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    auth = GOV.load(A.REPO / GOV.authorization_path(run_id))
    plan_path = GOV.plan_of(auth)
    if GOV.launch_hash(A.REPO, plan_path) != auth.plan_hash:
        raise SystemExit(f"{run_id}: {plan_path} on disk is not the plan its authorization binds")
    plan = A.read_json(A.REPO / plan_path)
    recs, ins = inputs(Path(store), auth.plan_hash)
    tasks = {t["task_id"]: t for p in plan["pools"] for t in A.jsonl(A.REPO / p["path"])}
    tok = B.load_tokenizer()
    got = G.adapt(tasks, recs, lambda t: SP.stratum_of(t, spec), tokenizer=tok)
    return {"spec": spec, "plan": plan, "plan_path": plan_path, "auth": auth, "ins": ins, "tasks": tasks,
            "tok": tok, "got": got, "records": recs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--pack-out", required=True)
    args = ap.parse_args(argv)
    v = verify_run(args.run_id, Path(args.store))
    spec, plan, auth, ins, tasks, tok, got = (v[k] for k in ("spec", "plan", "auth", "ins", "tasks", "tok", "got"))

    # per cell: what was generated, what was accepted, why the rest was not
    per: dict[str, dict] = defaultdict(lambda: {"tasks": set(), "candidates": 0, "accepted_units": set(),
                                                "reasons": Counter()})
    for c in got["candidates"]:
        row = per[cell_of(tasks[c.unit_id], spec)]
        row["tasks"].add(c.unit_id)
        row["candidates"] += 1
        if c.accepted_for_training:
            row["accepted_units"].add(c.unit_id)
        row["reasons"].update(c.reasons)

    # whole units rendered and admitted; supervised tokens per cell
    reused = AB._reused([A.REPO / p["path"] for p in plan.get("reuse", [])])
    sessions = got["sessions"] + reused
    shares = AB.cell_shares(spec, plan["corpus_id"])
    placed = AB.place(sessions, shares)
    rendered, excluded, _ = B.render_all(tok, placed, block_len=int(spec["block_len"]))
    think = tok.convert_tokens_to_ids("</think>")
    supply = Counter()
    units_by_cell = Counter()
    for u in B.ladder_units(rendered, think):
        supply[u.stratum] += u.size
        units_by_cell[u.stratum] += 1
    top = TARGET + int(spec["reserve_tokens"])
    cells = {}
    for cell, share in sorted(shares.items()):
        need = top * share
        g = per.get(cell)
        cells[cell] = {"share": share, "supervised_tokens_needed": round(need),
                       "supervised_tokens_admitted": supply[cell], "units_admitted": units_by_cell[cell],
                       "covered": supply[cell] >= need,
                       **({"tasks_generated": len(g["tasks"]), "candidates": g["candidates"],
                           "units_accepted": len(g["accepted_units"]),
                           "unit_accept_rate": round(len(g["accepted_units"]) / len(g["tasks"]), 4),
                           "rejection_reasons": dict(g["reasons"].most_common(8))} if g else {"reused": True})}
    gaps = {c: round(v["supervised_tokens_needed"] - v["supervised_tokens_admitted"])
            for c, v in cells.items() if not v["covered"]}
    capacity = min(supply[c] / s for c, s in shares.items())

    # isolation recheck on the selected units WITH their teacher text
    sel = {s["unit_id"]: s for s in got["sessions"]}
    items = [{**tasks[uid], "unit_id": uid, "messages": s["messages"]} for uid, s in sorted(sel.items())]
    iso = IC.check_pool(items)

    doc = {"schema": "aadistill.stage2.reasoning_data.p1v_verification/v1",
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "run_id": args.run_id, "authorization_id": auth.authorization_id,
           "launch_identity_sha256": auth.plan_hash, "inputs": ins,
           "contract": got["summary"], "units_selected": len(got["selected"]),
           "reused_units": len({s["unit_id"] for s in reused}),
           "admission": B.exclusion_summary(excluded),
           "cells": cells, "gaps_supervised_tokens": gaps,
           "composition_capacity_supervised_tokens": round(capacity),
           "isolation_recheck": {"pass": iso["pass"], "violations": iso["violations"][:20],
                                 "n_items": iso["n_items"],
                                 "R2_flagged": iso["results"]["R2_near_duplicate"].get("flagged"),
                                 "establishes": iso.get("establishes")},
           "authorizes": "nothing"}
    out = Path(args.pack_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "B_sessions.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in got["sessions"]))
    if gaps or not iso["pass"]:
        doc["pack"] = {"built": False, "why": ("cells short of their share: " + json.dumps(gaps) if gaps else
                                               "the isolation recheck failed"),
                       "rule": ("no paid top-up, no lower correctness gate, no repetition, no other source, "
                                "no other mix: the gap is reported and P1v stops here")}
    else:
        built, ladder_spec = AB.build(sessions, spec, shares, spec["rung_targets"], tok)
        meta = B.write_pack(out, built, ladder_spec, {
            "corpus_id": plan["corpus_id"], "run_id": args.run_id, "launch_identity_sha256": auth.plan_hash,
            "sessions_sha256": _sha(out.parent / "B_sessions.jsonl")})
        rung = next(r for r in meta["rungs"] if r["target_supervised_tokens"] == TARGET)
        doc["pack"] = {"built": True, "dir": str(out), "outputs": meta["outputs"],
                       "ladder_sha256": _sha(out / "ladder.json"), "blocks_sha256": _sha(out / "blocks.npz"),
                       "rung_2p96M": {k: rung.get(k) for k in ("n_units", "n_sessions", "actual_supervised_tokens",
                                                               "n_blocks", "real_tokens", "padding_tokens")},
                       "rungs": [{k: r.get(k) for k in ("target_supervised_tokens", "reachable", "n_blocks")}
                                 for r in meta["rungs"]]}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO), doc["pack"].get("built"), gaps or doc["pack"].get("rung_2p96M"))
    return 0 if doc["pack"]["built"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
