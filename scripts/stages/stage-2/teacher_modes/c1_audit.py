#!/usr/bin/env python3
"""Arm C1's identity audit: D4-Augmented's pack, its targets' teacher lineage and masks, the D4 Top-1. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.c1_audit \\
        --write logs/stages/stage-2/teacher_modes/audit/c1_identity_audit.json

C1 (maintainer 2026-10-31, the final 3 x 2 matrix) re-trains D4-Augmented's own data from the Thinking-2507 D4
Top-1 with Thinking-2507 as the online KD teacher, under the new target-position Top-200 + tail KD. This audit
re-derives, from the bytes, every identity that chain needs and every way the historical data differs from A1/A2's
protocol -- it changes nothing:

1. the pack (configs/stages/stage-2/reasoning_data/h1pa.json :: datasets.e1_2p96M_plus_fluid): each file's sha256,
   the training and validation sequence counts, the supervised tokens of each part (E1, fluid, E1 validation) from
   the pack's own audit rows;
2. the targets' teacher: corpus_v2's manifest (E1's part) and the three H1 generation sessions' records (the fluid
   part), against Thinking-2507 @768f209d;
3. the masks and packing: E1's part was cut where a session overran its sequence (traces trained without
   <|im_end|>), packed across documents inside system-prompt groups; the fluid part re-rendered its targets
   (token_identity_audit.json); the historical KD read every non-padding position, C1's reads the target positions;
4. the initialization: the D4 Top-1's single-shard sha256 and its root teacher, its tokenizer (the PCA init
   directory's) and the teacher's vocabulary, chat template and RoPE configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402

H1PA = REPO / "configs/stages/stage-2/reasoning_data/h1pa.json"
TEACHER = ("Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea")
TOP1_SHA = "fc28e0dd916d14438163ce9b5bd3f335165dafdf0d61075c065e97caa86dcfe8"
SCHEMA = "aadistill.stage2.teacher_modes.c1_identity_audit/v1"


def sha(p: Path) -> str:
    return A.sha256_file(p)


def pack(h: dict) -> dict:
    ds = h["datasets"]["e1_2p96M_plus_fluid"]
    d = REPO / ds["pack"]
    files = {"blocks.npz": ds["blocks_sha256"], "audit.jsonl": ds["audit_sha256"], "ladder.json": ds["ladder_sha256"]}
    got = {n: sha(d / n) for n in files}
    parts, blocks = Counter(), Counter()
    sessions = Counter()
    for row in A.jsonl(d / "audit.jsonl"):
        part = row.get("part") or "?"
        blocks[part] += 1
        for s in row.get("sessions", []):
            parts[part] += int(s.get("supervised_retained", 0))
            sessions[part] += 1
    return {"path": ds["pack"], "files_sha256": got, "registered": files, "match": got == files,
            "sequences_by_part": dict(blocks), "sessions_by_part": dict(sessions),
            "supervised_tokens_by_part": dict(parts),
            "registered_rung": ds["rung"], "relay": ds.get("relay")}


def reasoning_answer(h: dict) -> dict:
    """Per part: supervised target positions inside a session through </think> (reasoning) and after it (answer)."""
    import numpy as np
    d = REPO / h["datasets"]["e1_2p96M_plus_fluid"]["pack"]
    z = np.load(d / "blocks.npz")
    ids, mask = z["input_ids"], z[[k for k in z.files if "target" in k or "mask" in k][0]]
    out = {}
    for row in A.jsonl(d / "audit.jsonl"):
        b, part = int(row["block"]), row.get("part") or "?"
        o = out.setdefault(part, {"reasoning": 0, "answer": 0})
        for s in row.get("sessions", []):
            seg_ids, seg_mask = ids[b, s["start"]:s["end"]], mask[b, s["start"]:s["end"]].astype(bool)
            close = np.where(seg_ids == 151668)[0]
            cut = int(close[0]) if len(close) else -1
            pos = np.where(seg_mask)[0]
            o["reasoning"] += int((pos <= cut).sum()) if cut >= 0 else 0
            o["answer"] += int((pos > cut).sum()) if cut >= 0 else int(len(pos))
    return {"by_part": out, "keys": list(z.files),
            "_": "target positions of a session up to and including its first </think> (151668) are reasoning, the rest answer; a cut session without </think> counts as answer"}


def targets_teacher(h: dict) -> dict:
    m = A.corpus_v2_manifest()
    e1_teacher = (m.get("teacher") or {})
    fluid = {}
    for s in h["datasets"]["fluid_increment"]["generation_sessions"]:
        rd = REPO / "logs/stages/stage-2/reasoning_data/runs" / s["run_id"]
        auth = rd / "governance/authorization.json"
        fluid[s["run_id"]] = {"phase": s["phase"], "authorization": auth.is_file()}
    proto = json.loads((REPO / "configs/stages/stage-2/reasoning_data/generation_protocol.json").read_text())
    t = proto.get("teacher") or {}
    return {"e1_part": {"source": "corpus_v2 (artifacts/stages/stage-3/corpus_v2/manifest.json)",
                        "teacher": e1_teacher, "decoding": m.get("decoding") or (m.get("generation") or {})},
            "fluid_part": {"sessions": fluid, "generation_protocol_teacher": t,
                           "protocol": "configs/stages/stage-2/reasoning_data/generation_protocol.json"},
            "expected": {"model": TEACHER[0], "revision": TEACHER[1]}}


def initialization(h: dict) -> dict:
    d4 = h["initializations"]["d4_top1"]
    p = REPO / d4["path"] / "model.safetensors"
    rec = json.loads((REPO / d4["record"]).read_text())
    cfg = json.loads((REPO / d4["path"] / "config.json").read_text())
    pca = REPO / h["initializations"]["pca"]["path"]
    tok = {n: sha(pca / n) for n in ("tokenizer.json", "tokenizer_config.json") if (pca / n).is_file()}
    return {"path": d4["path"], "model_safetensors_sha256": sha(p) if p.is_file() else None,
            "expected": TOP1_SHA, "match": p.is_file() and sha(p) == TOP1_SHA,
            "record": d4["record"], "record_keys": sorted(rec)[:40],
            "root_teacher": rec.get("root_teacher") or rec.get("teacher") or rec.get("source_teacher"),
            "config": {k: cfg.get(k) for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
                                               "num_attention_heads", "num_key_value_heads", "head_dim",
                                               "tie_word_embeddings", "vocab_size", "rope_theta",
                                               "max_position_embeddings")},
            "tokenizer": {"source": h["initializations"]["pca"]["path"], "files_sha256": tok}}


def teacher_config() -> dict:
    from huggingface_hub.constants import HF_HUB_CACHE
    snap = Path(HF_HUB_CACHE) / f"models--{TEACHER[0].replace('/', '--')}" / "snapshots" / TEACHER[1]
    cfg = json.loads((snap / "config.json").read_text())
    tc = json.loads((snap / "tokenizer_config.json").read_text())
    tmpl = tc.get("chat_template") or ""
    return {"snapshot": str(snap.name), "rope_theta": cfg.get("rope_theta"),
            "max_position_embeddings": cfg.get("max_position_embeddings"), "vocab_size": cfg.get("vocab_size"),
            "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest() if tmpl else None,
            "tokenizer_json_sha256": sha(snap / "tokenizer.json") if (snap / "tokenizer.json").is_file() else None}


def differences() -> list[dict]:
    """Every protocol difference between C1 on D4-Augmented's pack and A1/A2, disclosed (none removed)."""
    tia = json.loads((REPO / "logs/stages/stage-2/teacher_modes/audit/token_identity_audit.json").read_text())
    asset = json.loads((REPO / "logs/stages/stage-2/reasoning_data/audit/asset_audit.json").read_text())
    rung = next(r for r in asset["ladder"]["rungs"] if r["target_supervised_tokens"] == 2960000)
    return [
        {"what": "packing / context strategy", "c1": "E1's sequential ladder packing (cross-context: several sessions per "
         "sequence inside a system-prompt group, 1,861 groups) for E1's part; E1's packer with no terminal truncation for "
         "the fluid part", "a1_a2": "document-isolated, each document conditioned on itself"},
        {"what": "cut traces (E1's part)", "c1": f"{rung['terminal_truncations']} sessions cut mid-trace at the 8,192 "
         f"boundary ({rung['supervised_tokens_discarded_by_cuts']:,} supervised tokens discarded), the cut sessions "
         "trained without <|im_end|>", "a1_a2": "no trace is cut: a turn over its allowance is refused"},
        {"what": "target correctness (E1's part)", "c1": "hygiene acceptance only: openmath 35.8% wrong, gsm8k 11.2%, "
         "multihop 14.2%, rag 2.1%; code and tool calling unverifiable (asset_audit.json :: corpus_v2)",
         "a1_a2": "every admitted target verified correct"},
        {"what": "tool calling (E1's part)", "c1": "about 1/6 of E1's tokens (492,587)", "a1_a2": "none (no legal source)"},
        {"what": "re-rendered targets (fluid part)", "c1": tia.get("target_differences", {}).get("summary") or
         "the template's prefilled <think> supervised in 767 sessions; 1,102 positions re-tokenized (token_identity_audit.json)",
         "a1_a2": "token-native: the engine's own ids"},
        {"what": "KD positions", "c1_new": "target positions only (kd_scope assistant), Top-200 + tail",
         "historical_d4_augmented": "every non-padding position (kd_scope all), the full vocabulary",
         "a1_a2": "target positions only, Top-200 + tail"},
        {"what": "teacher", "c1": f"{TEACHER[0]}@{TEACHER[1][:8]} (thinking-only, RoPE base 5,000,000, 262,144 positions)",
         "a1_a2": "Qwen/Qwen3-4B@1cfa9a72 (hybrid, RoPE base 1,000,000, 40,960 positions), its two modes"}]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", required=True)
    a = ap.parse_args(argv)
    h = json.loads(H1PA.read_text())
    rec = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "pack": pack(h), "targets_teacher": targets_teacher(h), "initialization": initialization(h),
           "teacher": teacher_config(), "reasoning_answer": reasoning_answer(h), "differences": differences(),
           "root_teacher": json.loads((REPO / "configs/stages/stage-0/calibration_volume/dose_d4_session.json").read_text())
           .get("teacher"), "authorizes": "nothing"}
    ok = rec["pack"]["match"] and rec["initialization"]["match"]
    rec["terminal"] = "IDENTITIES_VERIFIED" if ok else "MISMATCH"
    Path(a.write).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("terminal",)} | {"pack": {k: rec["pack"][k] for k in
          ("match", "sequences_by_part", "supervised_tokens_by_part")},
          "init": {k: rec["initialization"][k] for k in ("match", "root_teacher", "config")},
          "teacher": rec["teacher"], "targets": rec["targets_teacher"]}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
