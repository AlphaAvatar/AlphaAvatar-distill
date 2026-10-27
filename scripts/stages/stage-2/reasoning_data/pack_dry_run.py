#!/usr/bin/env python3
"""$0 dry run of the master-ladder builder on EXISTING verified material.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.pack_dry_run \
        [--scratch <dir>]

Two inputs, both already on disk, neither adopted as training data here:

* **old arm** -- every corpus_v2 unit except those with an isolation-flagged
  session (`audit/asset_audit.json :: isolation.corpus_v2_sessions` R2), under the
  historical uniform six-way token mixture, cut at 2.96M unique supervised
  tokens and packed by the shared builder (never cut);
* **fluid traces** -- the 519 verified Stage-0 teacher-native units, rendered
  from their raw teacher records (`teacher_cal.jsonl`) through the chat
  template, to measure what teacher-native fluid traces cost to pack and train
  on, and to prove the reconstruction is token-exact against `base_pool_v1`.

Writes `logs/stages/stage-2/reasoning_data/audit/packing_dry_run.json`. With
`--scratch`, the old-arm pack is also written there and loaded back through the
trainer's own reader (`aadistill.data.ladder.ladder_blocks`). The pack bytes are
never written into the repository.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import build_master_ladder as B

OUT = A.LOG_DIR / "audit/packing_dry_run.json"
AUDIT = A.LOG_DIR / "audit/asset_audit.json"
EPOCHS, BLOCKS_PER_STEP = 3, 2
OLD_ARM_SPEC = {"block_len": 8192, "salt": "stage2.reasoning_data.old_arm/v1",
                "shares": {t: 1 / 6 for t in ("code", "gsm8k", "multihop_qa", "openmath",
                                              "rag_evidence", "tool_calling")},
                "rung_targets": [2_960_000], "reserve_tokens": 150_000}


def old_arm_sessions() -> list[dict]:
    """corpus_v2 as it was, less every UNIT one of whose sessions isolation
    flagged: a unit is excluded whole, never trained on in part."""
    flagged = set(A.read_json(AUDIT)["isolation"]["corpus_v2_sessions"]["results"]
                  ["R2_near_duplicate"]["flagged_ids"])
    every = list(A.corpus_v2_sessions())
    size = Counter(s["source_id"] for s in every)
    gone = {s["source_id"] for s in every if s["id"] in flagged}
    out = []
    for s in every:
        if s["source_id"] in gone:
            continue
        out.append({"id": s["id"], "unit_id": s["source_id"], "unit_sessions": size[s["source_id"]],
                    "stratum": s["data_type"],
                    "family": s["source"], "attributes": {"source": s["source"]},
                    "messages": s["messages"], "tools": s.get("tools"),
                    "data_type": s["data_type"], "candidate_index": s.get("candidate_index"),
                    "candidate_sha256": s.get("candidate_sha256")})
    return out


def old_arm_isolation_exclusion() -> dict:
    flagged = set(A.read_json(AUDIT)["isolation"]["corpus_v2_sessions"]["results"]
                  ["R2_near_duplicate"]["flagged_ids"])
    every = list(A.corpus_v2_sessions())
    gone = {s["source_id"] for s in every if s["id"] in flagged}
    return {"sessions_flagged": len(flagged), "units_excluded": len(gone),
            "sessions_excluded": sum(1 for s in every if s["source_id"] in gone)}


def fluid_sessions() -> tuple[list[dict], dict[str, int]]:
    """Turn-expanded sessions of the 519 units, plus base_pool's own token counts."""
    pool = [it for it in A.base_pool_items() if it.get("pool_class") == "primary"]
    expected = {it["item_id"]: int(it["n_tokens"]) for it in pool}
    units = {it["unit_id"]: it for it in pool}
    recs = {r["task_id"]: r for r in A.teacher_cal_records()}
    out = []
    for uid, it in sorted(units.items()):
        # The record's own conversation, as base_pool_v1 rendered it: turn k is
        # the history up to and including the k-th assistant message.
        history = recs[uid]["messages"]
        at = [i for i, m in enumerate(history) if m["role"] == "assistant"]
        for k, idx in enumerate(at):
            out.append({"id": f"{uid}#t{k}", "unit_id": uid, "unit_sessions": len(at),
                        "stratum": it["axis"],
                        "family": f"{it['generator_id']}/{it['family_id']}",
                        "attributes": {"generator": it["generator_id"],
                                       "difficulty": it["difficulty"]},
                        "messages": history[:idx + 1], "data_type": it["axis"]})
    return out, expected


def summarize(built: dict, block_len: int) -> dict:
    rows = []
    for r in built["ladder"]["rungs"]:
        if not r["reachable"]:
            rows.append({"target": r["target"], "reachable": False})
            continue
        p = r["packed"]
        steps = math.ceil(p["blocks"] * EPOCHS / BLOCKS_PER_STEP)
        acc = r["accounting"]["counts"]
        rows.append({
            "target": r["target"], "units": r["n_units"], "sessions": p["sessions"],
            "unique_supervised_tokens": p["supervised_tokens"],
            "reasoning_tokens": acc.get("reasoning_tokens"), "answer_tokens": acc.get("answer_tokens"),
            "rendered_tokens": acc.get("rendered_tokens"),
            "blocks": p["blocks"], "packing_efficiency": p["packing_efficiency"],
            "padding_tokens": p["padding_tokens"], "multi_session_blocks": p["multi_session_blocks"],
            "steps_at_3_epochs": steps, "block_passes": steps * BLOCKS_PER_STEP,
            "forward_positions": steps * BLOCKS_PER_STEP * block_len,
            "supervised_per_forward_position": round(
                p["supervised_tokens"] * EPOCHS / (steps * BLOCKS_PER_STEP * block_len), 4),
            "realized_shares": r["accounting"]["realized_shares"],
            "max_share_deviation": r["accounting"].get("max_share_deviation"),
            "distinct_families": r["accounting"]["distinct_families"],
        })
    return {"rungs": rows, "admission": "whole units only (build_master_ladder.render_all)",
            "excluded": B.exclusion_summary(built["excluded"]),
            "excluded_examples": built["excluded"][:5], "rendered_sessions": built["n_rendered"],
            "units": built["n_units"], "system_prompt_groups": built["system_prompt_groups"],
            "validation_tail": built["validation_tail"],
            "composition_capacity": built["ladder"]["composition_capacity"]}


def write_arm_a(out: Path) -> dict:
    """Arm A's formal pack, written ONCE (np.savez stamps a time, so only a
    written file has a sha256): the same build as the dry run, refused unless its
    2.96M rung equals the measured composition the dry run recorded."""
    if (out / "ladder.json").exists():
        raise SystemExit(f"{out} already holds a pack; a recorded pack is never rewritten")
    old = B.build(old_arm_sessions(), OLD_ARM_SPEC, B.load_tokenizer())
    rec = A.read_json(OUT)["old_arm"]["rungs"][0]
    meta = B.write_pack(out, old, OLD_ARM_SPEC, {"arm": "A", "formal_pack": True,
                                                 "built_by": "pack_dry_run.write_arm_a"})
    rung = next(r for r in meta["rungs"] if r["target_supervised_tokens"] == 2_960_000)
    got = {"n_units": rung["n_units"], "n_sessions": rung["n_sessions"],
           "actual_supervised_tokens": rung["actual_supervised_tokens"], "n_blocks": rung["n_blocks"]}
    want = {"n_units": rec["units"], "n_sessions": rec["sessions"],
            "actual_supervised_tokens": rec["unique_supervised_tokens"], "n_blocks": rec["blocks"]}
    if got != want:
        raise SystemExit(f"arm A rebuilt as {got}, not the measured {want}")
    return {"composition": got, "blocks_sha256": A.sha256_file(out / "blocks.npz"),
            "ladder_sha256": A.sha256_file(out / "ladder.json")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--write-arm-a", default=None, metavar="DIR",
                    help="write arm A's formal pack to DIR (once) and print its identity")
    args = ap.parse_args(argv)
    if args.write_arm_a:
        print(json.dumps(write_arm_a(Path(args.write_arm_a))))
        return 0
    tok = B.load_tokenizer()

    old = B.build(old_arm_sessions(), OLD_ARM_SPEC, tok)
    hist = A.ladder_meta()
    hist_rung = next(r for r in hist["rungs"] if r["target_supervised_tokens"] == 2_960_000)
    hist_audit = A.ladder_audit()[:hist_rung["n_blocks"]]
    hist_sessions = {s["session_id"] for b in hist_audit for s in b["sessions"]}
    cut = old["ladder"]["rungs"][0]["n_units"]
    sess_of_unit: dict[str, list[str]] = {}
    for s in old_arm_sessions():
        sess_of_unit.setdefault(s["unit_id"], []).append(s["id"])
    new_sessions = {sid for u in old["ladder"]["order"][:cut] for sid in sess_of_unit[u]}
    old_summary = summarize(old, OLD_ARM_SPEC["block_len"])
    old_summary["isolation_exclusion"] = old_arm_isolation_exclusion()
    old_summary["historical_2p96m_rung"] = {
        "blocks": hist_rung["n_blocks"], "sessions": hist_rung["n_sessions"],
        "unique_supervised_tokens_retained": hist_rung["actual_supervised_tokens"],
        "terminal_truncations": hist_rung["terminal_truncations"],
        "block_passes": math.ceil(hist_rung["n_blocks"] * EPOCHS / BLOCKS_PER_STEP) * BLOCKS_PER_STEP,
        "session_overlap_with_rebuilt_old_arm": len(hist_sessions & new_sessions),
        "jaccard": round(len(hist_sessions & new_sessions) / len(hist_sessions | new_sessions), 4)}

    fl_sessions, expected = fluid_sessions()
    fluid_spec = {"block_len": 8192, "salt": "stage2.reasoning_data.fluid_dry_run/v1",
                  "shares": {"task_contained_deduction": 0.35, "inductive_rule_discovery": 0.2,
                             "compositional_generalization": 0.17, "abstract_analogical": 0.08,
                             "interactive_adaptation": 0.2},
                  "rung_targets": [1_000_000, 1_800_000], "reserve_tokens": 150_000,
                  "allow_composition_drift": True}
    fl = B.build(fl_sessions, fluid_spec, tok)
    # Token-exact reconstruction against base_pool_v1's own renders.
    tokenized = {}
    from aadistill.data.sessions import render_session
    mismatches = []
    for s in fl_sessions:
        try:
            r = render_session(tok, s, block_len=1 << 20)
        except ValueError:
            continue
        tokenized[s["id"]] = r.n_rendered_tokens
        if s["id"] in expected and expected[s["id"]] != r.n_rendered_tokens:
            mismatches.append({"id": s["id"], "base_pool": expected[s["id"]], "rebuilt": r.n_rendered_tokens})
    fluid_summary = summarize(fl, 8192)
    fluid_summary["spec"] = fluid_spec
    fluid_summary["reconstruction"] = {
        "items_compared": sum(1 for k in expected if k in tokenized),
        "token_count_mismatches": len(mismatches), "examples": mismatches[:5]}

    doc = {"schema": "aadistill.stage2.reasoning_data.packing_dry_run/v1",
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "tokenizer": str(A.TOKENIZER_DIR.relative_to(A.REPO)),
           "rule": "3 epochs, 2 blocks per step, 8,192 positions forwarded per block (the frozen recipe)",
           "old_arm": {"spec": OLD_ARM_SPEC, **old_summary},
           "fluid_traces": fluid_summary,
           "not_adopted": "neither input is adopted as training data by this dry run",
           "authorizes": "nothing"}

    if args.scratch:
        out = Path(args.scratch) / "old_arm_pack"
        meta = B.write_pack(out, old, OLD_ARM_SPEC, {"dry_run": True})
        from aadistill.data.ladder import ladder_blocks
        train, val, stats = ladder_blocks(out, 2_960_000, n_val=16)
        doc["old_arm"]["trainer_reader_roundtrip"] = {
            "train_blocks": int(train[0].shape[0]), "val_blocks": int(val[0].shape[0]),
            "train_supervised_tokens": stats["train_supervised_tokens"],
            "val_disjoint_from_all_rungs": stats["val_disjoint_from_all_rungs"],
            "blocks_sha256": meta["outputs"]["blocks.npz"]}
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(OUT.relative_to(A.REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
