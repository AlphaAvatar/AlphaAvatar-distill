#!/usr/bin/env python3
"""What sequential causal packing does to Research A's existing data. $0, dev box, nothing trained or forwarded.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packing_audit \\
        --write logs/stages/stage-2/teacher_modes/audit/packing_audit.json

The maintainer's 2026-11-03 decision (random shuffle -> sequential packing -> truncation -> standard causal attention,
`aadistill.data.sequence_packing`) applied, at the registered sequence length and shuffle seeds of
`packed_round_design.json`, to data that already exists:

* STAGE 1. The hybrid search's two calibration profiles and `state_eval_hybrid@v1`: packed sequences, repeated and
  scored positions, the largest scored-row count of one sequence (what sizes DEPTH's float32 KL), each set's packing
  identity, and the attention cost against reading every item alone (sum of T^2 over the forwards).
* STAGE 3. A1's and A2's five-type training documents, taken token for token out of their document-isolated packs
  (each document is one contiguous prompt + generation run there) and re-packed sequentially: the three layers the
  maintainer asked to keep apart -- (A) the selected data before packing, (B) what the packed sequences actually
  train on (fragments, repeated supervised tokens, truncation, padding), (C) the loss weighting -- and the steps and
  FLOPs at batch size 2 over 3 epochs, beside the isolated packs' own figures.

(C) replays the trainer's own sequence stream (`train.stream_block_indices`, per seed) and its normalization: each
optimizer step divides its CE sum and its KD sum by that step's supervised-position counts, so a token's loss weight is
(w_CE + w_KD) / N_step. The category shares of that weight are set beside the category shares of the supervised-token
exposure; a gap between the two is a step-level weighting effect, reported and not corrected (the training objective
is frozen).

This reads no student result and no evaluation item. It authorizes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DESIGN = REPO / "configs/stages/stage-2/teacher_modes/packed_round_design.json"
PROFILES_DOC = REPO / "configs/stages/stage-2/teacher_modes/hybrid_d4_profiles.json"
SUITE_ITEMS = REPO / "artifacts/stages/stage-2/teacher_modes/state_eval_hybrid_v1/items.jsonl"
PACKS = REPO / "artifacts/stages/stage-2/teacher_modes/packs_tm"
SCHEMA = "aadistill.stage2.teacher_modes.packing_audit/v1"
EPOCHS, BATCH, SEEDS, CE_W, KD_W = 3, 2, (20260726, 20260801), 0.25, 1.0


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def design() -> dict:
    return json.loads(DESIGN.read_text())


# --- stage 1 -------------------------------------------------------------------

def _items(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _scored_prediction_mask(item: dict) -> list[bool]:
    """supervised_target_v1 on a materialized item: the prediction positions whose target is in its `assistant`
    tag (token indices or a prediction mask, as the item stores it)."""
    import torch
    from aadistill.initialization.scoring.positions import PREDICTION_AXIS, SUPERVISED_TARGET_V1
    it = {**item, "input_ids": torch.tensor(item["ids"])[None]}
    return (SUPERVISED_TARGET_V1.weights(it, axis=PREDICTION_AXIS).weights > 0).tolist()


def alternatives(lengths_in_order: list[int], capacity: int, fragment_scored: int, placed_scored: int) -> dict:
    """Decision support only, NOT the registered protocol: what the two obvious alternatives would give.

    * truncate=False -- the reference algorithm's own other branch: no fragments, the room left is padding (and an
      example longer than the capacity is skipped);
    * fragments as context only -- the registered packing, with fragment positions excluded from the loss/score.
    """
    from aadistill.data.sequence_packing import packing_conversation
    seqs = packing_conversation(list(lengths_in_order), capacity, truncate=False)
    placed = sum(lengths_in_order[i] for s in seqs for i in s)
    return {"truncate_false": {"sequences": len(seqs),
                               "padding_share": round(1 - placed / (capacity * len(seqs)), 6) if seqs else 0.0,
                               "skipped_longer_than_capacity": sum(n > capacity for n in lengths_in_order)},
            "fragments_context_only": {"scored_or_supervised_positions": placed_scored - fragment_scored,
                                       "positions_computed_but_unscored": fragment_scored}}


def stage1_set(name: str, items: list[dict], sequence_length: int, seed: int) -> dict:
    import torch
    from aadistill.initialization.calibration.packing import ForwardContext
    fc = ForwardContext(sequence_length=sequence_length, shuffle_seed=seed)
    prepared = [{"item_id": it["item_id"], "input_ids": torch.tensor(it["ids"])[None]} for it in items]
    plan = fc.plan(prepared)
    report = fc.report(prepared)
    scored = [_scored_prediction_mask(it) for it in items]
    per_seq, repeated_scored, total_scored = [], 0, 0
    for seq in plan:
        n = 0
        for p in seq:
            k = sum(scored[p.index][:p.tokens - 1])
            n += k
            total_scored += k
            repeated_scored += 0 if p.final else k
        per_seq.append(n)
    lengths = [len(it["ids"]) for it in items]
    rows = [sum(p.tokens for p in seq) for seq in plan]
    unique_scored = sum(map(sum, scored))
    by_mode = Counter()
    for it, s in zip(items, scored):
        by_mode[it.get("mode")] += sum(s)
    return {"set": name, "items": len(items), **{k: report[k] for k in (
        "sequences", "unique_tokens", "placed_tokens", "repeated_tokens", "truncated_tokens", "padding_tokens",
        "fragments", "examples_truncated", "examples_per_sequence_mean")},
            "unique_scored_positions": unique_scored, "placed_scored_positions": total_scored,
            "repeated_scored_positions": repeated_scored,
            "repeated_scored_share": round(repeated_scored / total_scored, 6) if total_scored else 0.0,
            "unique_scored_positions_by_mode": dict(by_mode),
            "max_scored_rows_in_one_sequence": max(per_seq), "max_item_tokens": max(lengths),
            "forwards": {"packed_batch_1": len(plan), "one_item_per_forward": len(items)},
            "attention_cost_ratio_sum_T2": round(sum(r * r for r in rows) / sum(n * n for n in lengths), 4),
            "token_positions_ratio": round(sum(rows) / sum(lengths), 6),
            "packing_identity": fc.packing_identity(prepared), "forward_context": fc.as_dict(),
            "alternatives_not_adopted": alternatives(
                [lengths[i] for i in _shuffled(items, seed)], sequence_length, repeated_scored, total_scored)}


def _shuffled(items: list[dict], seed: int) -> list[int]:
    from aadistill.data.sequence_packing import shuffled_order
    return shuffled_order([it["item_id"] for it in items], seed)


def stage1(d: dict) -> dict:
    fc = d["stage1_forward_context"]
    L, seed = int(fc["sequence_length"]), int(fc["shuffle_seed"])
    prof = json.loads(PROFILES_DOC.read_text())
    out = {}
    for p in prof["profiles"]:
        path = REPO / p["items_path"]
        if sha_file(path) != p["items_file_sha256"]:
            raise SystemExit(f"{path} is not the registered profile's items file")
        out[f"{p['profile_id']}@v{p['version']}"] = stage1_set(p["profile_id"], _items(path), L, seed)
    out["state_eval_hybrid@v1"] = stage1_set("state_eval_hybrid@v1", _items(SUITE_ITEMS), L, seed)
    return out


# --- stage 3 -------------------------------------------------------------------

def documents_from_isolated_pack(pack_dir: Path, split: str = "train"):
    """Every document of a `pack_isolated` pack, token for token: (TokenDocument, data_type)."""
    import numpy as np
    from aadistill.data.token_documents import TokenDocument
    arr = np.load(pack_dir / "blocks.npz")
    ids, ce = arr["input_ids"], arr["ce_mask"]
    out = []
    for line in (pack_dir / "audit.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] != split:
            continue
        b = row["block"]
        for s in row["sessions"]:
            tok = [int(x) for x in ids[b, s["start"]:s["start"] + s["length"]]]
            m = [bool(x) for x in ce[b, s["start"]:s["start"] + s["length"]]]
            k = m.index(True)
            if not all(m[k:]) or any(m[:k]):
                raise SystemExit(f"{s['session_id']}: its targets are not one contiguous generation")
            if sum(m) != s["supervised_retained"]:
                raise SystemExit(f"{s['session_id']}: {sum(m)} targets, the audit says {s['supervised_retained']}")
            out.append((TokenDocument(s["session_id"], tuple(tok[:k]), tuple(tok[k:])), s["data_type"]))
    return out


def stream_weights(seq_sup: list[Counter], seed: int, steps: int) -> dict:
    """Replay the trainer's sequence stream and normalization for one seed."""
    from aadistill.training.train import stream_block_indices
    n = len(seq_sup)
    w_by, x_by, per_step = Counter(), Counter(), []
    for step in range(steps):
        idx = stream_block_indices(n, seed, step * BATCH, BATCH)
        c = Counter()
        for i in idx:
            c.update(seq_sup[i])
        total = sum(c.values())
        per_step.append(total)
        if not total:
            continue
        for cat, v in c.items():
            x_by[cat] += v
            w_by[cat] += (CE_W + KD_W) * v / total
    tw, tx = sum(w_by.values()), sum(x_by.values())
    mean = sum(per_step) / len(per_step)
    sd = (sum((v - mean) ** 2 for v in per_step) / len(per_step)) ** 0.5
    return {"seed": seed, "steps": steps,
            "supervised_positions_per_step": {"mean": round(mean, 1), "cv": round(sd / mean, 4),
                                              "min": min(per_step), "max": max(per_step)},
            "category": {cat: {"exposure_share": round(x_by[cat] / tx, 5),
                               "loss_weight_share": round(w_by[cat] / tw, 5),
                               "ratio": round((w_by[cat] / tw) / (x_by[cat] / tx), 4)}
                         for cat in sorted(x_by)}}


def schedule(T: int) -> dict:
    return {"optimizer_steps": T, "warmup_steps": max(10, round(0.05 * T)), "eval_every": max(25, T // 8),
            "save_every": max(200, T // 2)}


def stage3_arm(arm: str, d: dict, per_position_flops: float | None) -> dict:
    from aadistill.data.token_documents import pack_sequential
    pack_dir = PACKS / arm
    seed = int(d["stage3_packing"]["shuffle_seed"])
    L = int(d["stage3_packing"]["sequence_length"])
    pairs = documents_from_isolated_pack(pack_dir)
    docs = [x for x, _ in pairs]
    dtype = {x.doc_id: t for x, t in pairs}
    old = json.loads((pack_dir / "ladder.json").read_text())
    packed = pack_sequential(docs, L, pad_id=0, order_seed=seed)
    st = packed.stats()
    # (A) before packing
    cat_sup = Counter()
    cat_tok = Counter()
    reasoning = 0
    from stages.teacher_modes.token_admission import THINK_CLOSE
    for x in docs:
        cat_sup[dtype[x.doc_id]] += len(x.output_ids)
        cat_tok[dtype[x.doc_id]] += len(x)
        if THINK_CLOSE in x.output_ids:
            reasoning += list(x.output_ids).index(THINK_CLOSE) + 1
    A = {"examples_or_turns": len(docs), "unique_supervised_tokens": sum(cat_sup.values()),
         "prompt_context_tokens": sum(len(x.prompt_ids) for x in docs), "reasoning_tokens": reasoning,
         "answer_tokens": sum(cat_sup.values()) - reasoning, "tool_calling_tokens": 0,
         "unique_supervised_tokens_by_category": dict(sorted(cat_sup.items())),
         "_note": "examples_or_turns counts documents (one assistant turn each); tool calling is absent from the five-type data"}
    # (B) after packing
    seq_sup_new = [Counter() for _ in packed.members]
    rep_by = Counter()
    for b, ms in enumerate(packed.members):
        for m in ms:
            seq_sup_new[b][dtype[m["doc_id"]]] += m["supervised"]
            if not m["final"]:
                rep_by[dtype[m["doc_id"]]] += m["supervised"]
    N = st["sequences"]
    T = math.ceil(EPOCHS * N / BATCH)
    B = {"packed_sequences": N, "non_padding_tokens": st["non_padding_tokens"],
         "padding_tokens": N * L - st["non_padding_tokens"], "truncated_tokens": st["truncated_tokens"],
         "truncated_supervised_tokens": st["truncated_supervised_tokens"],
         "repeated_tokens": st["repeated_tokens"], "repeated_supervised_tokens": st["repeated_supervised_tokens"],
         "repeated_supervised_tokens_by_category": dict(sorted(rep_by.items())),
         "fragments": st["fragments"], "ce_target_positions_per_epoch": st["supervised_tokens"],
         "kd_scored_positions_per_epoch": st["supervised_tokens"],
         "supervised_token_exposure_3_epochs": EPOCHS * st["supervised_tokens"],
         "_kd_equals_ce": "kd_scope assistant: KD is scored at exactly the CE target positions"}
    positions = T * BATCH * L
    C = {"ce_weight": CE_W, "kd_weight": KD_W, "kd_temperature": 1.0,
         "normalization": "per optimizer step: CE sum / that step's CE target count, KD sum / that step's KD count",
         "by_seed": [stream_weights(seq_sup_new, s, T) for s in SEEDS]}
    # the isolated pack, for comparison
    import json as _j
    seq_sup_old = []
    for line in (pack_dir / "audit.jsonl").read_text().splitlines():
        row = _j.loads(line)
        if row["split"] == "train":
            c = Counter()
            for s in row["sessions"]:
                c[s["data_type"]] += s["supervised_retained"]
            seq_sup_old.append(c)
    T_old = math.ceil(EPOCHS * old["n_blocks"] / BATCH)
    out = {"arm": arm, "mode": old.get("mode"), "source_pack": {
        "dir": str(pack_dir.relative_to(REPO)), "blocks_npz_sha256": sha_file(pack_dir / "blocks.npz"),
        "audit_sha256": sha_file(pack_dir / "audit.jsonl"), "packing": old["packing"]},
        "A_selected_data": A, "B_training_coverage": B, "C_loss_weighting": C,
        "schedule_sequential": schedule(T),
        "isolated_pack_for_comparison": {"packed_sequences": old["n_blocks"],
                                         "non_padding_tokens": old["train"]["non_padding_tokens"],
                                         "padding_fraction": old["train"]["padding_fraction"],
                                         "optimizer_steps": T_old,
                                         "loss_weighting": [stream_weights(seq_sup_old, s, T_old) for s in SEEDS]},
        "packing_spec": {k: packed.packing[k] for k in ("packing", "materialization", "shuffle", "attention",
                                                         "sequence_length", "shuffle_seed")},
        "alternatives_not_adopted": alternatives(
            [len(docs[i]) for i in __import__("aadistill.data.sequence_packing", fromlist=["x"]).shuffled_order(
                [x.doc_id for x in docs], seed)], L, st["repeated_supervised_tokens"], st["supervised_tokens"])}
    if per_position_flops:
        out["training_flops"] = {"positions_computed": positions,
                                 "flops": float(f"{positions * per_position_flops:.4g}"),
                                 "isolated_pack_flops": float(f"{T_old * BATCH * L * per_position_flops:.4g}"),
                                 "per_position": per_position_flops,
                                 "_method": "token_admission.per_position_flops (student 4 x forward + teacher forward per "
                                            "computed position, the A1 accounting); the attention term grows with the "
                                            "context each position sees, which packing lengthens, and is not in this "
                                            "per-position figure"}
    return out


D4A = REPO / "artifacts/stages/stage-2/reasoning_data/packs/D4A_merged_whole"
D4A_TRAIN_SEQUENCES = 2468


def documents_from_d4a(pack_dir: Path = D4A, exclude_type: str = "tool_calling"):
    """C1's kept training data as standalone documents: each session's block system prompt + its own tokens.

    D4A_merged_whole was packed by E1's packer, sequential within system-prompt group: a sequence opens with its
    group's system block (`n_system_tokens`) and its sessions follow back to back, so `system + block[start:end]`
    is the session exactly as rendered (`original_rendered_tokens` = system + body; a session cut by E1's terminal
    truncation stays cut). Its supervised tokens are the pack's own `ce_mask`, so a multi-turn session keeps every
    supervised span (`MaskedDocument`)."""
    import numpy as np
    from aadistill.data.token_documents import MaskedDocument
    arr = np.load(pack_dir / "blocks.npz")
    ids, ce = arr["input_ids"], arr["ce_mask"]
    out = []
    for line in (pack_dir / "audit.jsonl").read_text().splitlines():
        row = json.loads(line)
        b = row["block"]
        if b >= D4A_TRAIN_SEQUENCES:
            continue
        ns = int(row["n_system_tokens"])
        system = [int(x) for x in ids[b, :ns]]
        if any(bool(x) for x in ce[b, :ns]):
            raise SystemExit(f"block {b}: its system block carries supervised tokens")
        for sess in row["sessions"]:
            if sess["data_type"] == exclude_type:
                continue
            body = [int(x) for x in ids[b, sess["start"]:sess["end"]]]
            m = [bool(x) for x in ce[b, sess["start"]:sess["end"]]]
            if sum(m) != sess["supervised_retained"]:
                raise SystemExit(f"{sess['session_id']}: {sum(m)} targets, the audit says {sess['supervised_retained']}")
            if not sess["truncated"] and len(system) + len(body) != sess.get("original_rendered_tokens",
                                                                         len(system) + len(body)):
                raise SystemExit(f"{sess['session_id']}: system + body is not the rendered session")
            label = sess["data_type"] if row["part"] == "e1" else sess["data_type"].split("/")[0]
            out.append((MaskedDocument(f"{b}:{sess['session_id']}", tuple(system + body),
                                       tuple([False] * ns + m)), label, bool(sess["truncated"])))
    return out


def stage3_c1(d: dict) -> dict:
    """C1-6cat's kept (tool-free) part under the new packing: its sequences, repeated supervised tokens, steps."""
    from aadistill.data.token_documents import pack_sequential
    seed, L = int(d["stage3_packing"]["shuffle_seed"]), int(d["stage3_packing"]["sequence_length"])
    rows = documents_from_d4a()
    docs = [x for x, *_ in rows]
    label = {x.doc_id: t for x, t, _ in rows}
    packed = pack_sequential(docs, L, pad_id=0, order_seed=seed)
    st = packed.stats()
    by = Counter()
    for x, t, _ in rows:
        by[t] += sum(x.target_mask)
    rep = Counter()
    for ms in packed.members:
        for m in ms:
            if not m["final"]:
                rep[label[m["doc_id"]]] += m["supervised"]
    return {"source_pack": {"dir": str(D4A.relative_to(REPO)), "blocks_npz_sha256": sha_file(D4A / "blocks.npz"),
                            "audit_sha256": sha_file(D4A / "audit.jsonl")},
            "kept_documents": len(docs), "cut_traces_kept_as_cut": sum(1 for *_, tr in rows if tr),
            "unique_supervised_tokens": st["unique_supervised_tokens"],
            "unique_supervised_tokens_by_type": dict(sorted(by.items())),
            "packed_sequences_without_tool_part": st["sequences"], "non_padding_tokens": st["non_padding_tokens"],
            "repeated_supervised_tokens": st["repeated_supervised_tokens"],
            "repeated_supervised_tokens_by_type": dict(sorted(rep.items())),
            "historical_kept_sequences": 989,
            "_note": "the tool part (target 492,587 supervised tokens, Thinking-2507-native, from the canonical pool) is "
                     "packed together with these documents once it exists; the shuffle covers every document, so no "
                     "kept sequence is carried over byte for byte -- the tokens of every kept session are carried over unchanged"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None)
    ap.add_argument("--skip-stage3", action="store_true")
    a = ap.parse_args(argv)
    d = design()
    try:
        from stages.teacher_modes.token_admission import per_position_flops
        ppf = per_position_flops()["training_total"]
    except Exception:  # noqa: BLE001 -- the FLOPs figure needs the teacher config in the HF cache
        ppf = None
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "design_sha256": sha_file(DESIGN), "stage1": stage1(d),
           "stage3": {} if a.skip_stage3 else {**{arm: stage3_arm(arm, d, ppf) for arm in ("A1", "A2")},
                                                "C1_kept_part": stage3_c1(d)},
           "authorizes": "nothing"}
    text = json.dumps(doc, indent=1) + "\n"
    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(text)
    print(json.dumps({k: {kk: {x: vv.get(x) for x in ("sequences", "repeated_scored_share",
                                                        "max_scored_rows_in_one_sequence",
                                                        "attention_cost_ratio_sum_T2")}
                           for kk, vv in v.items()} if k == "stage1" else None
                      for k, v in doc.items() if k == "stage1"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
