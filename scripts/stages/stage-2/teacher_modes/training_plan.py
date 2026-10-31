#!/usr/bin/env python3
"""The final 3 x 2 training matrix as a PROPOSAL: every unit's recovery config with its lineage block, validated. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.training_plan \\
        --write logs/stages/stage-2/teacher_modes/plans/training_plan_proposal.json

Maintainer 2026-10-31 (Teacher-Student Lineage Invariant and Final 3 x 2 Training Matrix): A1 (the hybrid D4
full-search Top-1; the hybrid in thinking mode as KD teacher and data), A2 (the same Top-1; non-thinking), C1
(Thinking-2507's D4 Top-1; Thinking-2507; D4-Augmented's own pack), seeds 20260726 and 20260801. Six units, no
other. Every unit gets E1's / D4-Augmented's recipe with the registered objective -- CE 0.25 + KD 1.0, T 1, KD on
the assistant target positions only, Top-200 + tail -- and a ``lineage`` block (aadistill.training.lineage) that
config validation (here), launch admission (`admit`) and the trainer at runtime check, failing closed.

A1/A2's Top-1 does not exist yet: their configs carry ``student_path`` and the initialization digest as
``TO_BE_SET_BY_THE_FULL_SEARCH``, and `admit` refuses them until the search's committed record binds it. Nothing
here trains, launches or authorizes anything.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.training import lineage as LIN  # noqa: E402
from stages.reasoning_data import assets as A  # noqa: E402

E1 = REPO / "configs/stages/stage-3/e1/e1_r2960k_sa_pca.json"
H1PA = REPO / "configs/stages/stage-2/reasoning_data/h1pa.json"
ADMISSION = REPO / "logs/stages/stage-2/teacher_modes/data/token_matched_admission.json"
C1_AUDIT = REPO / "logs/stages/stage-2/teacher_modes/audit/c1_identity_audit.json"
HYBRID = REPO / "configs/stages/stage-2/teacher_modes/teacher_identity.json"
T2507_BINDING = REPO / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
SEEDS = (20260726, 20260801)
UNSET = "TO_BE_SET_BY_THE_FULL_SEARCH"
SCHEMA = "aadistill.stage2.teacher_modes.training_plan_proposal/v1"
#: measured on the L40S (logs/stages/stage-2/teacher_modes/assets/l40s_check.json and v1's recovery: H-T-C 6.7 s per
#: optimizer step of two 8,192-token sequences), the student evaluation's 3,917.1 s per unit, 10 min fixed per unit
SECONDS_PER_STEP, EVAL_SECONDS, FIXED_SECONDS, USD_PER_HOUR = 6.7, 3917.1, 600.0, 1.09
OBJECTIVE = {"ce_weight": 0.25, "kd_weight": 1.0, "kd_temperature": 1.0, "kd_scope": "assistant",
             "kd_support": {"support_id": "reference_topk_tail_v1", "top_k": 200}}


def schedule(blocks: int, batch: int = 2, epochs: int = 3) -> dict:
    """E1's builder rule (h1pa.json :: recipe.derive): T = ceil(3 x sequences / 2); warmup max(10, round(0.05 T));
    eval every max(25, T // 8); save every max(200, T // 2)."""
    t = math.ceil(epochs * blocks / batch)
    return {"total_steps": t, "warmup_steps": max(10, round(0.05 * t)), "min_lr_frac": 0.1,
            "eval_every": max(25, t // 8), "save_every": max(200, t // 2)}


def teachers() -> dict:
    h = json.loads(HYBRID.read_text())
    b = json.loads(T2507_BINDING.read_text())
    hist = h["historical_teacher"]
    return {
        "hybrid": {"model_id": h["model"], "revision": h["revision"],
                   "weights_sha256": {s["file"]: s["lfs_sha256"] for s in h["weight_shards"]},
                   "tokenizer_sha256": h["vocabulary_sha256"], "chat_template_sha256": h["chat_template_sha256"],
                   "rope_theta": 1000000.0, "max_position_embeddings": 40960, "vocab_size": 151936},
        "thinking_2507": {"model_id": hist["model"], "revision": hist["revision"],
                          "weights_sha256": dict(b["expected_shard_sha256"]),
                          "tokenizer_sha256": hist.get("vocabulary_sha256") or h["vocabulary_sha256"],
                          "chat_template_sha256": hist["chat_template_sha256"],
                          "rope_theta": 5000000.0, "max_position_embeddings": 262144, "vocab_size": 151936}}


def units(adm: dict | None, c1: dict) -> list[dict]:
    e1 = json.loads(E1.read_text())
    h = json.loads(H1PA.read_text())
    ts = teachers()
    out = []
    arms = {}
    for arm, mode, tmode in (("A1", "H-T", "thinking"), ("A2", "H-N", "non_thinking")):
        a = (adm or {}).get("arms", {}).get(arm) or {}
        pack = a.get("pack") or {}
        arms[arm] = {"teacher": ts["hybrid"], "mode": tmode, "data_dir": f"artifacts/stages/stage-2/teacher_modes/packs_tm/{arm}",
                     "packing": "ladder", "context_strategy": "document_isolated",
                     "files_sha256": pack.get("files_sha256") or {}, "blocks": a.get("sequences"),
                     "rung": a.get("supervised_tokens"), "init_sha256": UNSET, "student_path": UNSET,
                     "tokenizer_source": "the hybrid Top-1's own tokenizer files (the teacher snapshot's, by sha256)",
                     "val_blocks": int((pack.get("n_val_blocks") or 16)), "accounting": a}
    ds = h["datasets"]["e1_2p96M_plus_fluid"]
    arms["C1"] = {"teacher": ts["thinking_2507"], "mode": "thinking", "data_dir": ds["pack"], "packing": "ladder",
                  "context_strategy": "cross_context (D4-Augmented's own pack, kept and disclosed)",
                  "files_sha256": {"blocks.npz": ds["blocks_sha256"], "audit.jsonl": ds["audit_sha256"],
                                   "ladder.json": ds["ladder_sha256"]},
                  "blocks": ds["rung"]["n_blocks"], "rung": ds["rung"]["actual_supervised_tokens"],
                  "init_sha256": c1["initialization"]["model_safetensors_sha256"],
                  "student_path": h["initializations"]["d4_top1"]["path"],
                  "tokenizer_source": h["initializations"]["pca"]["path"], "val_blocks": 16, "accounting": {}}
    for arm, a in arms.items():
        for seed in SEEDS:
            cfg = copy.deepcopy(e1)
            cfg.update(run_name=f"research_a_v2_{arm}_s{seed}", seed=seed, data_dir=a["data_dir"],
                       packing=a["packing"], student_path=a["student_path"], tokenizer_source=a["tokenizer_source"],
                       teacher={"model_id": a["teacher"]["model_id"], "revision": a["teacher"]["revision"],
                                "dtype": "bfloat16"}, loss=dict(OBJECTIVE), rung=a["rung"], val_blocks=a["val_blocks"],
                       out_dir=f"/runpod-volume/teacher_modes/v2_units/{arm}-s{seed}/train")
            cfg.pop("_purpose", None)
            if a["blocks"]:
                sch = schedule(int(a["blocks"]))
                cfg["schedule"] = {k: sch[k] for k in ("total_steps", "warmup_steps", "min_lr_frac")}
                cfg["intervals"] = {**cfg["intervals"], "eval_every": sch["eval_every"], "eval_blocks": a["val_blocks"]}
                cfg["checkpoint"] = {**cfg["checkpoint"], "save_every": sch["save_every"], "bind_identity": True}
            if a["context_strategy"] == "document_isolated":   #: the token-document packs declare it (write_pack)
                cfg["batch"] = {**cfg["batch"], "context_strategy": "document_isolated"}
            cfg["lineage"] = {"teacher": a["teacher"], "mode": a["mode"],
                              "initialization": {"root_teacher": {k: a["teacher"][k] for k in ("model_id", "revision")},
                                                 "model_safetensors_sha256": a["init_sha256"]},
                              "data": [{"files_sha256": a["files_sha256"], "mode": a["mode"],
                                        "teacher": {k: a["teacher"][k] for k in ("model_id", "revision")}}]}
            out.append({"unit": f"{arm}-s{seed}", "arm": arm, "seed": seed, "config": cfg, "arm_facts": a})
    return out


def validate(us: list[dict]) -> dict:
    """Config validation: each unit's declared lineage against its own config (KD teacher, declared identities)."""
    res = {}
    for u in us:
        cfg = u["config"]
        bad = LIN.problems(cfg["lineage"], kd_teacher=cfg["teacher"])
        pending = [b for b in bad if UNSET in json.dumps(cfg["lineage"]["initialization"]) and "initialization" in b]
        res[u["unit"]] = {"problems": [b for b in bad if b not in pending],
                          "pending_full_search": bool(UNSET in json.dumps(cfg["lineage"]))}
    return res


def admit(us: list[dict], root: Path = REPO) -> None:
    """Launch admission (for the future training launcher): every unit's lineage against the COMMITTED bytes -- the
    initialization's model.safetensors, the pack's files, the student config -- fail closed, all units reported."""
    rows = []
    for u in us:
        cfg = u["config"]
        sp, dd = root / cfg["student_path"], root / cfg["data_dir"]
        obs = {"kd_teacher": cfg["teacher"],
               "student_weights_sha256": A.sha256_file(sp / "model.safetensors") if (sp / "model.safetensors").is_file() else "missing",
               "pack_files_sha256": {n: A.sha256_file(dd / n) for n in ("blocks.npz", "ladder.json", "audit.jsonl")
                                     if (dd / n).is_file()},
               "student_config": json.loads((sp / "config.json").read_text()) if (sp / "config.json").is_file() else None}
        rows.append((u["unit"], cfg["lineage"], obs))
    LIN.require_all(rows)


def accounting(u: dict) -> dict:
    a = u["arm_facts"]
    blocks = a["blocks"]
    if not blocks:
        return {"pending": "the arm's pack does not exist yet"}
    t = math.ceil(3 * int(blocks) / 2)
    hours = (t * SECONDS_PER_STEP + EVAL_SECONDS + FIXED_SECONDS) / 3600
    acc = dict(a.get("accounting") or {})
    return {"packed_sequences": blocks, "optimizer_steps": t, "unique_supervised_tokens": a["rung"],
            "supervised_token_exposure": 3 * int(a["rung"]), "hours": round(hours, 2), "usd": round(hours * USD_PER_HOUR, 2),
            **{k: acc[k] for k in ("non_padding_tokens", "padding_share", "training_flops", "documents", "tasks",
                                   "by_category", "category_share_points") if k in acc}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", required=True)
    a = ap.parse_args(argv)
    adm = json.loads(ADMISSION.read_text()) if ADMISSION.is_file() else None
    c1 = json.loads(C1_AUDIT.read_text())
    us = units(adm, c1)
    val = validate(us)
    acc = {u["unit"]: accounting(u) for u in us}
    total = sum(v.get("usd", 0) for v in acc.values())
    rec = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "status": "PROPOSED for the maintainer's review; nothing trains before an explicit authorization",
           "matrix": {"A1": "hybrid Top-1 + hybrid thinking", "A2": "hybrid Top-1 + hybrid non-thinking",
                      "C1": "Thinking-2507 D4 Top-1 + Thinking-2507 + D4-Augmented"}, "seeds": list(SEEDS),
           "objective": OBJECTIVE, "baseline_recipe": {"path": str(E1.relative_to(REPO)), "sha256": A.sha256_file(E1)},
           "units": [{"unit": u["unit"], "config": u["config"], "lineage_validation": val[u["unit"]],
                      "accounting": acc[u["unit"]]} for u in us],
           "cost": {"usd_expected_training": round(total, 2), "basis": {"seconds_per_step": SECONDS_PER_STEP,
                    "evaluation_seconds_per_unit": EVAL_SECONDS, "fixed_seconds_per_unit": FIXED_SECONDS,
                    "usd_per_hour": USD_PER_HOUR}},
           "authorizes": "nothing"}
    Path(a.write).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({"validation": val, "accounting": acc, "usd": round(total, 2)}, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
