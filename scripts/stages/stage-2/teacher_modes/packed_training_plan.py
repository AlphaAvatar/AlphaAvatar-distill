#!/usr/bin/env python3
"""A1-6cat, A2-6cat and C1-6cat under the sequential causal packing: sequences, steps, schedule and cost. $0, estimates.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_training_plan \\
        --write logs/stages/stage-2/teacher_modes/plans/packed_training_budget.json

Each arm's packed-sequence count is the six-category design's isolated-pack estimate (lower / expected / upper,
`six_category_design.json :: packing_steps_budget`) times the factor the same packing measured on that arm's own
existing data (`audit/packing_audit.json`: A1's documents 1,102 -> 1,288 sequences, A2's 4,577 -> 4,935). C1-6cat is
re-derived: its kept documents measured at 1,042 sequences, plus the tool part at the historical tool turns'
non-padding tokens per supervised token, packed densely (no longer one sequence per conversation) with the repeated
share A1's documents showed.

Steps T = ceil(3 x sequences / 2) (3 epochs, batch size 2, micro-batch size 1); warmup max(10, round(0.05 T)), evaluation
every max(25, T // 8), saves every max(200, T // 2). Seconds per step are the measured figures the design used, in a
range 0.85x-1.3x: the packed sequences are read with full causal attention over 8,192 tokens instead of per-document
attention, which no run has timed yet. Cost at $1.11/h (L40S + container disk). Nothing here is a measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))
AUDIT = REPO / "logs/stages/stage-2/teacher_modes/audit/packing_audit.json"
SCHEMA = "aadistill.stage2.teacher_modes.packed_training_budget/v1"
RATE = 1.11
SECONDS_PER_STEP = {"A1-6cat": 6.19, "A2-6cat": 5.66, "C1-6cat": 6.19}
SPS_RANGE = (0.85, 1.0, 1.3)
#: six_category_design.json :: packing_steps_budget, per seed: isolated-pack sequence estimates
ISOLATED_SEQUENCES = {"A1-6cat": (1106, 1228, 1350), "A2-6cat": (4210, 4786, 6277)}
#: in-training evaluation and saves; the formal evaluation; fixed time (seconds), lower / expected / upper
EVAL_SAVES, FORMAL_EVAL, FIXED = (100, 320, 640), (2653, 3558, 7834), (300, 600, 1200)
C1_TOOL_SUPERVISED, C1_TOOL_NONPAD_PER_SUP = 492587, (1.6, 2.03, 3.0)
SEEDS = (20260726, 20260801)


def schedule(seqs: int) -> dict:
    T = math.ceil(3 * seqs / 2)
    return {"packed_sequences": seqs, "optimizer_steps": T, "warmup_steps": max(10, round(0.05 * T)),
            "eval_every": max(25, T // 8), "save_every": max(200, T // 2)}


PACKS = "artifacts/stages/stage-2/teacher_modes/packs_6cat"
TO_BUILD = "TO_BE_SET_WHEN_THE_PACK_IS_BUILT"


def packed_units() -> list[dict]:
    """The six units' recovery configs under the sequential causal packing, with their lineage blocks.

    `training_plan.units`' construction (E1's recipe, the registered objective, the lineage block) with the
    packed round's differences: the pack is `pack_sequential`'s (no `batch.context_strategy` override, so the
    trainer reads it with standard causal attention), each arm's data and pack are the packed round's, A1-6cat
    and A2-6cat start from the packed search's Top-1 (unset until it exists) and C1-6cat from Thinking-2507's D4
    Top-1. The steps, the pack's files and A1/A2's initialization are placeholders until their owners exist;
    `training_plan.admit` refuses every placeholder.
    """
    import copy
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    from stages.teacher_modes import training_plan as TP
    e1 = json.loads(TP.E1.read_text())
    h = json.loads(TP.H1PA.read_text())
    ts, tf = TP.teachers(), TP.tokenizer_files()
    c1 = json.loads(TP.C1_AUDIT.read_text())
    arms = {
        "A1-6cat": {"teacher": ts["hybrid"], "mode": "thinking", "init_sha256": TP.UNSET, "student_path": TP.UNSET,
                    "provenance": {"record": TP.UNSET, "record_sha256": TP.UNSET},
                    "tokenizer_source": TP.TOKENIZERS["hybrid"], "tokenizer_files": tf["hybrid"]},
        "A2-6cat": {"teacher": ts["hybrid"], "mode": "non_thinking", "init_sha256": TP.UNSET, "student_path": TP.UNSET,
                    "provenance": {"record": TP.UNSET, "record_sha256": TP.UNSET},
                    "tokenizer_source": TP.TOKENIZERS["hybrid"], "tokenizer_files": tf["hybrid"]},
        "C1-6cat": {"teacher": ts["thinking_2507"], "mode": "thinking",
                    "init_sha256": c1["initialization"]["model_safetensors_sha256"],
                    "student_path": h["initializations"]["d4_top1"]["path"],
                    "provenance": {"record": str(TP.D4_SEARCH.relative_to(REPO)),
                                   "record_sha256": TP.A.sha256_file(TP.D4_SEARCH)},
                    "tokenizer_source": TP.TOKENIZERS["thinking_2507"], "tokenizer_files": tf["thinking_2507"]}}
    out = []
    for arm, a in arms.items():
        files = {n: TO_BUILD for n in ("blocks.npz", "audit.jsonl", "ladder.json")}
        for seed in SEEDS:
            cfg = copy.deepcopy(e1)
            cfg.pop("_purpose", None)
            cfg.update(run_name=f"research_a_packed_{arm}_s{seed}", seed=seed, data_dir=f"{PACKS}/{arm}",
                       packing="ladder", student_path=a["student_path"], tokenizer_source=a["tokenizer_source"],
                       tokenizer_sha256=a["teacher"]["tokenizer_sha256"],
                       teacher={"model_id": a["teacher"]["model_id"], "revision": a["teacher"]["revision"],
                                "dtype": "bfloat16"}, loss=dict(TP.OBJECTIVE), rung=TO_BUILD, val_blocks=16,
                       out_dir=f"/runpod-volume/teacher_modes/packed_units/{arm}-s{seed}/train")
            cfg["schedule"] = {"total_steps": TO_BUILD, "warmup_steps": TO_BUILD, "min_lr_frac": 0.1,
                               "_rule": "training_plan.schedule(packed sequences of the built pack)"}
            cfg["checkpoint"] = {**cfg["checkpoint"], "bind_identity": True}
            cfg["batch"] = {k: v for k, v in cfg["batch"].items() if k != "context_strategy"}
            cfg["lineage"] = {"teacher": a["teacher"], "mode": a["mode"],
                              "tokenizer": {"files_sha256": a["tokenizer_files"]},
                              "initialization": {"root_teacher": {k: a["teacher"][k] for k in ("model_id", "revision")},
                                                 "model_safetensors_sha256": a["init_sha256"],
                                                 "provenance": dict(a["provenance"])},
                              "data": [{"files_sha256": files, "mode": a["mode"],
                                        "teacher": {k: a["teacher"][k] for k in ("model_id", "revision")}}]}
            out.append({"unit": f"{arm}-s{seed}", "arm": arm, "seed": seed, "config": cfg})
    for u in out:
        if u["config"]["batch"].get("context_strategy") == "document_isolated":
            raise SystemExit(f"{u['unit']}: a new experiment may not train document-isolated (AGENTS.md P19)")
    return out


BOUND = REPO / "logs/stages/stage-2/teacher_modes/plans/packed_training_plan.json"


def bind(units: list[dict], packs_dir: Path, top1_products: Path | None = None) -> dict:
    """The units bound to their BUILT packs (files, rung, steps and schedule from the pack) and, for A1-6cat and
    A2-6cat, to the packed search's committed Top-1 (`packed_full_search` products.json). A built pack must pass
    `packed_packs.verify_pack` (an admitted, unchanged data manifest, its documents, the design's packing spec, and
    its actual packed tokens and supervision against the packing's recorded content identity and the manifest's
    documents), or the binding refuses; the lineage then binds the files, that content identity and the manifest. An arm whose pack is not built stays unbound and `training_plan.admit` refuses it."""
    import copy
    from aadistill.infrastructure.manifest import sha256_file
    from stages.teacher_modes import packed_packs as PP
    from stages.teacher_modes import training_plan as TP
    top1 = json.loads(Path(top1_products).read_text()) if top1_products else None
    out = []
    for u in units:
        cfg = copy.deepcopy(u["config"])
        arm = u["arm"]
        d = Path(packs_dir) / arm
        if (d / "ladder.json").is_file():
            man = PP.verify_pack(d)
            lad = json.loads((d / "ladder.json").read_text())
            files = {n: sha256_file(d / n) for n in ("blocks.npz", "audit.jsonl", "ladder.json")}
            sch = TP.schedule(int(lad["n_blocks"]))
            cfg.update(data_dir=str(d.relative_to(REPO)) if d.is_relative_to(REPO) else str(d),
                       rung=int(lad["rungs"][0]["target_supervised_tokens"]),
                       val_blocks=int(lad.get("n_val_blocks") or 16))
            cfg["schedule"] = {k: sch[k] for k in ("total_steps", "warmup_steps", "min_lr_frac")}
            cfg["intervals"] = {**cfg["intervals"], "eval_every": sch["eval_every"], "eval_blocks": cfg["val_blocks"]}
            cfg["checkpoint"] = {**cfg["checkpoint"], "save_every": sch["save_every"], "bind_identity": True}
            cfg["lineage"]["data"][0].update(files_sha256=files,
                                             packed_content_sha256=lad["data_admission"]["packed_content_sha256"],
                                             data_manifest_content_sha256=man["content_sha256"])
        if arm in ("A1-6cat", "A2-6cat") and top1 is not None:
            t = top1["top1"]
            cfg["student_path"] = t["path"]
            cfg["lineage"]["initialization"]["model_safetensors_sha256"] = t["files_sha256"]["model.safetensors"]
            cfg["lineage"]["initialization"]["provenance"] = {"record": str(top1_products),
                                                              "record_sha256": sha256_file(top1_products)}
        out.append({**u, "config": cfg})
    return {"schema": "aadistill.stage2.teacher_modes.packed_training_plan/v1", "seed_order": list(SEEDS),
            "units": out, "lineage_validation": TP.validate(out), "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    ap.add_argument("--bind-packs", default=None, help="bind the units to built packs under this directory")
    ap.add_argument("--top1-products", default=None, help="the packed search's products.json")
    a = ap.parse_args(argv)
    if a.bind_packs:
        doc = bind(packed_units(), Path(a.bind_packs), Path(a.top1_products) if a.top1_products else None)
        BOUND.write_text(json.dumps(doc, indent=1) + "\n")
        print(BOUND)
        return 0
    audit = json.loads(AUDIT.read_text())["stage3"]
    factor = {"A1-6cat": audit["A1"]["B_training_coverage"]["packed_sequences"]
              / audit["A1"]["isolated_pack_for_comparison"]["packed_sequences"],
              "A2-6cat": audit["A2"]["B_training_coverage"]["packed_sequences"]
              / audit["A2"]["isolated_pack_for_comparison"]["packed_sequences"]}
    rep_share = audit["A1"]["B_training_coverage"]["repeated_tokens"] / audit["A1"]["A_selected_data"]["unique_supervised_tokens"]
    kept = audit["C1_kept_part"]["packed_sequences_without_tool_part"]
    seqs = {arm: [math.ceil(n * factor[arm]) for n in ISOLATED_SEQUENCES[arm]] for arm in factor}
    seqs["C1-6cat"] = [kept + math.ceil(C1_TOOL_SUPERVISED * r * (1 + rep_share) / 8192)
                       for r in C1_TOOL_NONPAD_PER_SUP]
    arms = {}
    for arm, ns in seqs.items():
        rows = []
        for i, case in enumerate(("lower", "expected", "upper")):
            sch = schedule(ns[i])
            sec = sch["optimizer_steps"] * SECONDS_PER_STEP[arm] * SPS_RANGE[i] + EVAL_SAVES[i] + FORMAL_EVAL[i] + FIXED[i]
            rows.append({"case": case, **sch, "seconds_per_step": round(SECONDS_PER_STEP[arm] * SPS_RANGE[i], 3),
                         "hours_per_seed": round(sec / 3600, 2), "usd_per_seed": round(sec / 3600 * RATE, 2)})
        arms[arm] = {"per_seed": rows, "seeds": list(SEEDS),
                     "usd_both_seeds": {r["case"]: round(2 * r["usd_per_seed"], 2) for r in rows}}
    total = {c: round(sum(v["usd_both_seeds"][c] for v in arms.values()), 2) for c in ("lower", "expected", "upper")}
    from stages.teacher_modes import training_plan as TP
    us = packed_units()
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "units": [{"unit": u["unit"], "config": u["config"]} for u in us],
           "lineage_validation": TP.validate(us),
           "is_an_estimate": True, "measured_packing_factors": {k: round(v, 4) for k, v in factor.items()},
           "c1_kept_sequences_measured": kept, "repeated_share_used_for_c1_tool_part": round(rep_share, 4),
           "arms": arms, "six_trainings_usd": total,
           "not_included": "pod setup/teardown per session (0.6-1.8 h), storage, the teacher-native tool generation",
           "authorizes": "nothing"}
    text = json.dumps(doc, indent=1) + "\n"
    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(text)
    print(json.dumps({k: {r["case"]: (r["optimizer_steps"], r["usd_per_seed"]) for r in v["per_seed"]}
                      for k, v in arms.items()} | {"six": total}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
