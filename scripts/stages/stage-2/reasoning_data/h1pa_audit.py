#!/usr/bin/env python3
"""H1-PA (the PCA-aligned redesign of H1): its $0 data and identity audit.

    # 1. D4-Reasoning's sessions: @v3's verified teacher data exactly as its
    #    pack read them, written as input for E1's own packer
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_audit sessions \
        --store h1gen_20261022_190653=<P1 scr>/store --store h1p1t_20261023_094007=<P1t scr>/store \
        --out artifacts/stages/stage-2/reasoning_data/packs/D4R_seq_sessions.jsonl
    # 2. E1's packer, unchanged (the command `sessions` prints)
    # 3. the audit
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_audit audit \
        --reasoning-pack artifacts/stages/stage-2/reasoning_data/packs/D4R_seq \
        --reasoning-sessions artifacts/stages/stage-2/reasoning_data/packs/D4R_seq_sessions.jsonl \
        --e1-regen <dir E1's packer rebuilt E1's pack into> --relay-sa <dir holding E1 sa's relay run manifest>

The three arms (maintainer, 2026-10-24): PCA-Original is E1's 2.96M PCA arm
as it ran, re-evaluated and never retrained; D4-Original is D4 Top-1 trained
on E1's own ``blocks.npz`` under E1's configuration; D4-Reasoning is D4 Top-1
trained on ``stage2.reasoning_master@v3`` packed by E1's sequential packer
(``scripts/shared/data/build_token_ladder.py``), never best-fit.

What the packer receives is @v3's sessions with two fields mapped, nothing
else: ``data_type`` is the session's mixture cell (``stratum/source``, the key
@v3 declares its shares in, as ``data_type`` is the key E1's mixture is
declared in), and ``source_id`` is the unit (the conversation an episode's
turns were expanded from, as corpus_v2's ``source_id`` is). The packer itself
decides order, sequences, truncation and the validation tail.

Writes ``logs/stages/stage-2/reasoning_data/audit/h1pa_audit.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A

E1_PACK = A.REPO / "artifacts/shared/instruments/ladder_uniform_probe"
E1_SESSIONS = A.REPO / "artifacts/stages/stage-3/corpus_v2/sessions.jsonl"
E1_CONFIGS = {"20260726": A.REPO / "configs/stages/stage-3/e1/e1_r2960k_sa_pca.json",
              "20260801": A.REPO / "configs/stages/stage-3/e1/e1_r2960k_sb_pca.json"}
PCA_INIT = A.REPO / "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint"
D4_INIT_RECORD = A.LOG_DIR / "assets/d4_top1_materialization.json"
REASONING_PACK_RECORD = A.REPO / "artifacts/stages/stage-2/reasoning_data/packs/B/ladder.json"
OUT = A.LOG_DIR / "audit/h1pa_audit.json"
RUNG = 2_960_000
EPOCHS = 3
BATCH_SIZE = 2


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def builder_rows(placed: list[dict]) -> list[dict]:
    """@v3's placed sessions as E1's packer reads them: the mixture cell as
    `data_type`, the unit as `source_id`; every other field as it is."""
    return [{**s, "data_type": s["stratum"], "source_id": s["unit_id"]} for s in placed]


def reasoning_sessions(stores: dict[str, Path]) -> tuple[list[dict], dict]:
    """The sessions @v3's pack was built from, re-derived from the secured
    stores by the feasibility tool's own functions, and their recorded hash
    checked: anything else refuses."""
    from stages.reasoning_data import build_arm_b as AB
    from stages.reasoning_data import build_master_ladder as B
    from stages.reasoning_data import corpus_feasibility as CF
    from stages.reasoning_data import source_plan as SP
    rec = A.read_json(REASONING_PACK_RECORD)
    cand_path = A.REPO / "configs/stages/stage-2/reasoning_data/corpus_v3.json"
    if A.sha256_file(cand_path) != rec["provenance"]["candidate_sha256"]:
        raise SystemExit(f"{cand_path} is not the corpus file the @v3 pack names")
    spec, cand = SP.candidate_spec(cand_path), A.read_json(cand_path)
    runs = CF.verify_sessions(CF.declared_sessions(cand), stores)
    reuse = [A.REPO / r["path"] for r in cand["data"]["reuse"]]
    for r, p in zip(cand["data"]["reuse"], reuse):
        if A.sha256_file(p) != r["sha256"]:
            raise SystemExit(f"{r['path']} is not the reuse selection the corpus names")
    generated, _ = CF.merge(runs, tag="generation_sessions" in cand["data"])
    generated, aside = CF.set_aside(generated, spec)
    sessions = generated + AB._reused(reuse)
    sha = CF._sha("".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions))
    if sha != rec["provenance"]["sessions_sha256"]:
        raise SystemExit(f"the re-derived sessions hash {sha}, not the {rec['provenance']['sessions_sha256']} "
                         "the @v3 pack was built from")
    shares = AB.cell_shares(spec, spec["corpus_id"])
    placed = AB.place(sessions, shares)
    _, excluded, _ = B.render_all(runs[0]["tok"], placed, block_len=int(spec["block_len"]))
    if excluded:
        raise SystemExit(f"{len(excluded)} units no longer admit; the pool is not the one @v3 packed")
    return placed, {"sessions_sha256": sha, "n_sessions": len(placed),
                    "n_units": len({s["unit_id"] for s in placed}), "set_aside_units": len(aside),
                    "declared_shares": shares, "corpus_id": spec["corpus_id"]}


def cmd_sessions(args) -> int:
    stores = {k: Path(v) for k, v in (kv.split("=", 1) for kv in args.store)}
    placed, meta = reasoning_sessions(stores)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in builder_rows(placed):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta["builder_input_sha256"] = A.sha256_file(out)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    mixture = ",".join(f"{k}={v}" for k, v in meta["declared_shares"].items())
    print(json.dumps({k: v for k, v in meta.items() if k != "declared_shares"}, indent=1))
    print("\nE1's packer, unchanged:\n.venv/bin/python scripts/shared/data/build_token_ladder.py "
          f"--sessions {out} --model {PCA_INIT.relative_to(A.REPO)} --block-len 8192 "
          f"--mixture {mixture} --out <pack dir>")
    return 0


# -- identity -----------------------------------------------------------------
def check(name: str, got, want) -> dict:
    return {"check": name, "got": got, "want": want, "ok": got == want}


def pca_original_identity(study: dict, relay_sa: Path | None, e1_regen: Path | None) -> list[dict]:
    """E1's 2.96M PCA arm as it ran: configs, pack, sessions, init, the two
    checkpoints and, when given, the packer's byte-for-byte rebuild."""
    ds, out = study["datasets"]["e1_2p96M"], []
    pack = A.REPO / ds["pack"]
    for f, key in (("blocks.npz", "blocks_sha256"), ("ladder.json", "ladder_sha256"), ("audit.jsonl", "audit_sha256")):
        out.append(check(f"E1 pack {f}", A.sha256_file(pack / f), ds[key]))
    out.append(check("E1 sessions", A.sha256_file(A.REPO / ds["sessions"]), ds["sessions_sha256"]))
    out.append(check("E1 pack built from those sessions", A.read_json(pack / "ladder.json")["sessions_sha256"],
                     ds["sessions_sha256"]))
    for seed, c in study["recipe"]["e1_configs"].items():
        out.append(check(f"E1 config seed {seed}", canonical_sha(A.read_json(A.REPO / c["path"])), c["canonical_sha256"]))
    init = study["initializations"]["pca"]
    out.append(check("PCA init weights", A.sha256_file(A.REPO / init["path"] / "model.safetensors"),
                     init["model_safetensors_sha256"]))
    ck = study["arms"]["PCA-Original"]["checkpoints"]
    sb = A.REPO / ck["20260801"]["local"]
    out.append(check("PCA-Original seed 20260801 weights (dev box)", A.sha256_file(sb / "model.safetensors"),
                     ck["20260801"]["model_safetensors_sha256"]))
    pod = dict(reversed(line.split()) for line in (sb / "pod_hashes.txt").read_text().splitlines() if line.strip())
    out.append(check("PCA-Original seed 20260801 files against the pod-side hashes",
                     {f: A.sha256_file(sb / f) for f in sorted(pod)}, dict(sorted(pod.items()))))
    if relay_sa:
        man = A.read_json(next(relay_sa.rglob("run_manifest.json")))
        hashes = next(relay_sa.rglob("*_artifact_hashes_*.txt")).read_text()
        out.append(check("E1 seed 20260726 manifest: config", man["config_sha256"],
                         study["recipe"]["e1_configs"]["20260726"]["canonical_sha256"]))
        out.append(check("E1 seed 20260726 manifest: pack", man["ladder"]["blocks_sha256"], ds["blocks_sha256"]))
        out.append(check("PCA-Original seed 20260726 weights (pod-side hash on the relay)",
                         ck["20260726"]["model_safetensors_sha256"] in hashes, True))
    if e1_regen:
        import numpy as np
        a, b = np.load(e1_regen / "blocks.npz"), np.load(pack / "blocks.npz")
        out.append(check("E1's packer rebuilds E1's pack: blocks.npz bytes", A.sha256_file(e1_regen / "blocks.npz"),
                         ds["blocks_sha256"]))
        out.append(check("E1's packer rebuilds E1's pack: audit.jsonl", A.sha256_file(e1_regen / "audit.jsonl"),
                         ds["audit_sha256"]))
        out.append(check("E1's packer rebuilds E1's pack: arrays",
                         all(np.array_equal(a[k], b[k]) for k in ("input_ids", "ce_mask", "content_mask")), True))
    return out


def d4_identity(study: dict) -> list[dict]:
    """D4 Top-1 is the PCA student's architecture: the same tensors, names and
    shapes, so the only factor Comparison A varies is the weights."""
    from safetensors import safe_open
    d4, pca = study["initializations"]["d4_top1"], study["initializations"]["pca"]
    out = [check("D4 Top-1 weights", A.sha256_file(A.REPO / d4["path"] / "model.safetensors"),
                 d4["model_safetensors_sha256"]),
           check("D4 Top-1 record", A.read_json(A.REPO / d4["record"])["materialization_identity"]["checkpoint_sha256"],
                 d4["model_safetensors_sha256"])]
    shapes = {}
    for name, p in (("pca", pca["path"]), ("d4", d4["path"])):
        with safe_open(str(A.REPO / p / "model.safetensors"), "pt") as f:
            shapes[name] = {k: list(f.get_slice(k).get_shape()) for k in f.keys()}
    out.append(check("D4 Top-1 tensors equal the PCA student's (names and shapes)", shapes["d4"], shapes["pca"]))
    ca, cb = (A.read_json(A.REPO / p / "config.json") for p in (pca["path"], d4["path"]))
    differ = {k: {"pca": ca.get(k), "d4": cb.get(k)} for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)}
    out.append({"check": "config keys that differ", "got": differ, "ok": set(differ) <= {"max_window_layers", "use_cache"},
                "why_harmless": "max_window_layers is read only with use_sliding_window (false in both); use_cache "
                                "is an inference cache switch"})
    return out


def training_order_replay(study: dict) -> list[dict]:
    """Today's trainer against E1's own logs: the sequences each logged step
    drew (through their supervised-token and KD-position counts) and its
    learning rate. E1 sb's log is gone; sb_rand ran the same pack and seed."""
    from aadistill.data.ladder import ladder_blocks
    from aadistill.training.train import lr_factor, prediction_mask, stream_block_indices
    ds = study["datasets"]["e1_2p96M"]
    (ids, mask, content), _, stats = ladder_blocks(A.REPO / ds["pack"], ds["rung"], n_val=16)
    cfg = A.read_json(A.REPO / study["recipe"]["e1_configs"]["20260726"]["path"])
    sch, bps = cfg["schedule"], cfg["batch"]["blocks_per_step"]
    logs = {"20260726": ("e1_r2960k_sa_pca", A.REPO / "artifacts/stages/stage-3/rescued/_relay/e1_r2960k_sa_pca/train_log.jsonl"),
            "20260801": ("e1_r2960k_sb_rand (same pack and seed as sb_pca)",
                         A.REPO / "artifacts/stages/stage-3/rescued/_logs_rand/e1_r2960k_sb_rand/train_log.jsonl")}
    out = []
    for seed, (run, path) in logs.items():
        steps = [r for r in A.jsonl(path) if r.get("event") == "train_step"]
        order = lr = 0
        for r in steps:
            s = r["step"] - 1
            idx = stream_block_indices(ids.shape[0], int(seed), s * bps, bps)
            order += (int(mask[idx][:, 1:].sum()) != r["ce_targets"]
                      or int(prediction_mask(mask[idx], "all", content[idx]).sum()) != r["kd_positions"])
            lr += not math.isclose(cfg["optim"]["lr"] * lr_factor(s, sch["total_steps"], sch["warmup_steps"],
                                                                  sch["min_lr_frac"]), r["lr"], rel_tol=1e-12)
        out.append({"check": f"seed {seed}: {run}", "logged_steps": len(steps), "order_or_mask_mismatches": order,
                    "lr_mismatches": lr, "ok": bool(steps) and order == 0 and lr == 0})
    out.append({"check": "validation sequences", "got": stats["val_block_indices"], "ok": True})
    return out


# -- data ---------------------------------------------------------------------
def pctl(xs: list, q: float):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def truncation_detail(trunc: list[dict], sessions: dict[str, dict], tok) -> dict:
    """Each truncated session re-rendered: how many of the tokens the cut
    discarded were reasoning (before `</think>`) and how many answer, and
    whether the retained target still holds its answer."""
    from aadistill.data.sessions import render_session
    close = tok.convert_tokens_to_ids("</think>")
    disc = Counter()
    kinds = Counter()
    for m in trunc:
        r = render_session(tok, sessions[m["session_id"]], block_len=8192)
        kept = m["end"] - m["start"]
        sup = [i for i, f in enumerate(r.body_mask) if f]
        closes = [i for i in sup if r.body_ids[i] == close]
        c = closes[-1] if closes else None
        for i in sup:
            if i >= kept:
                disc["reasoning" if c is not None and i <= c else "answer" if c is not None else "unsplit"] += 1
        kinds["no_think_close" if c is None else "answer_lost" if kept <= c else
              "answer_cut" if kept <= sup[-1] else "intact"] += 1
    return {"discarded_supervised": dict(disc), "retained_target": dict(kinds),
            "_retained_target": "answer_lost: the cut falls inside the reasoning, so the retained target holds no "
                                "answer; answer_cut: the cut falls inside the answer"}


def pack_metrics(pack: Path, rung: int, sessions: dict[str, dict], tok, *, unit_of, declared: dict | None) -> dict:
    from aadistill.data.ladder import load_ladder_meta, select_val_blocks
    from aadistill.data.sessions import rung_n_blocks
    meta = load_ladder_meta(pack)
    audit = list(A.jsonl(pack / "audit.jsonl"))
    n = rung_n_blocks(meta, rung)
    train = audit[:n]
    largest = max(int(r["n_blocks"]) for r in meta["rungs"] if r.get("reachable"))
    val = [audit[i] for i in select_val_blocks(audit, largest, 16)]
    members = [m for r in train for m in r["sessions"]]
    trunc = [m for m in members if m["truncated"]]
    sup = sum(r["supervised_tokens"] for r in train)
    real = sum(r["unpadded_length"] for r in train)
    per_seq = [r["supervised_tokens"] for r in train]
    fill = [r["unpadded_length"] / r["final_length"] for r in train]
    mix = Counter()
    for m in members:
        mix[m["data_type"]] += m["supervised_retained"]
    out = {"pack": str(pack.relative_to(A.REPO)), "packing": meta["packing"], "ordering": meta["ordering"],
           "sequence_length": meta["block_len"], "packed_sequences": n, "source_sessions": len(members),
           "units": len({unit_of(sessions[m["session_id"]]) for m in members}),
           "supervised_tokens": sup, "non_padding_tokens": real, "padding_tokens": n * meta["block_len"] - real,
           "padding_ratio": round(1 - real / (n * meta["block_len"]), 4),
           "supervised_tokens_per_sequence": {"p10": pctl(per_seq, 0.1), "p50": pctl(per_seq, 0.5),
                                              "p90": pctl(per_seq, 0.9), "mean": round(sup / n, 1)},
           "sequences_under_15pct_full": sum(f < 0.15 for f in fill),
           "truncated_sessions": len(trunc),
           "supervised_discarded_by_truncation": sum(m["supervised_discarded"] for m in members),
           "truncation": truncation_detail(trunc, sessions, tok),
           "token_mix": {k: round(v / sup, 4) for k, v in sorted(mix.items())}}
    if declared:
        out["mix_minus_declared_pp"] = {k: round(100 * (out["token_mix"].get(k, 0) - v), 2)
                                        for k, v in sorted(declared.items())}
    train_units = {unit_of(sessions[m["session_id"]]) for m in members}
    later = {unit_of(sessions[m["session_id"]]) for r in audit[n:] for m in r["sessions"]}
    out["units_split_by_the_cut"] = len(train_units & later)
    val_members = [m for r in val for m in r["sessions"]]
    out["validation"] = {
        "sequences": len(val), "supervised_tokens": sum(r["supervised_tokens"] for r in val),
        "from_sequence": largest, "_from": "16 sequences past the largest reachable rung (ladder.select_val_blocks)",
        "units_also_in_training": len({unit_of(sessions[m["session_id"]]) for m in val_members} & train_units),
        "prompts_also_in_training": len({_prompt(sessions[m["session_id"]]) for m in val_members}
                                        & {_prompt(sessions[m["session_id"]]) for m in members}),
        "token_mix": _mix(val_members)}
    return out


def _prompt(s: dict) -> str:
    return hashlib.sha256(next(m["content"] for m in s["messages"] if m["role"] == "user").encode()).hexdigest()


def _mix(members: list[dict]) -> dict:
    c = Counter()
    for m in members:
        c[m["data_type"]] += m["supervised_retained"]
    t = sum(c.values()) or 1
    return {k: round(v / t, 4) for k, v in sorted(c.items())}


def by_axis(mix: dict) -> dict:
    out = Counter()
    for k, v in mix.items():
        out[k.split("/")[0].replace("supporting.", "supporting:").split(":")[0]] += v
    return {k: round(v, 4) for k, v in sorted(out.items())}


def family_coverage(pack: Path, rung: int, sessions: dict[str, dict]) -> dict:
    from aadistill.data.sessions import rung_n_blocks
    n = rung_n_blocks(A.read_json(pack / "ladder.json"), rung)
    seen = {sessions[m["session_id"]].get("family") for r in list(A.jsonl(pack / "audit.jsonl"))[:n]
            for m in r["sessions"]} - {None}
    pool = {s.get("family") for s in sessions.values()} - {None}
    return {"families_in_pool": len(pool), "families_trained": len(seen), "missing": sorted(pool - seen)}


def e1_isolation(pack: Path, rung: int) -> dict:
    """E1's pack against eval-set-1, from the isolation audit H1 already ran
    over every corpus_v2 session: which flagged sessions E1 trained on."""
    from aadistill.data.sessions import rung_n_blocks
    iso = A.read_json(A.LOG_DIR / "audit/asset_audit.json")["isolation"]["corpus_v2_sessions"]["results"]
    n = rung_n_blocks(A.read_json(pack / "ladder.json"), rung)
    trained = {m["session_id"] for r in list(A.jsonl(pack / "audit.jsonl"))[:n] for m in r["sessions"]}
    r2 = iso["R2_near_duplicate"]
    hit = [f for f in r2["flagged_items"] if f["id"] in trained]
    return {"R1_held_out_keys": iso["R1_held_out_keys"],
            "R2_near_duplicate_trained": {"n": len(hit), "of_flagged": r2["flagged"],
                                          "by_unit": dict(Counter(f["unit"] for f in hit)),
                                          "fluid_reasoning_720": r2["by_unit"]["fluid_reasoning_720/dev_v1"]},
            "R8_in_sample": iso["R8_in_sample_overlap"],
            "reading": ("the fluid units share no prompt key and no 13-gram near-copy with E1's training data, so "
                        "the primary fluid conclusions are isolated; the near-copies sit in the behaviour unit and "
                        "one multihop item, which are reported with that caveat; stage3_legacy_150 is made of E1's "
                        "own training prompts and is in-sample regression only")}


def no_truncation_variant(rows: list[dict], shares: dict, tok, *, block_len: int = 8192) -> dict:
    """A MEASURED ALTERNATIVE, not a pack: E1's packer step for step (the
    builder's render, interleave, pack, block order) with the one switch the
    packer itself provides, `allow_terminal_truncation=False`, which closes a
    sequence instead of cutting its last session. Reported so the maintainer
    can weigh it; nothing trains on it."""
    from aadistill.data.mixture import interleave, order_blocks
    from aadistill.data.sessions import pack_sessions, render_session, render_system_block
    rendered = [render_session(tok, s, block_len=block_len) for s in rows]
    by_type = defaultdict(list)
    for r in rendered:
        by_type[r.data_type].append(r)
    order = interleave(dict(by_type), shares)
    sys_ids = {}
    for r in rendered:
        if r.system_key not in sys_ids:
            sys_ids[r.system_key] = tok(render_system_block(tok, r.system_text, r.tools),
                                        add_special_tokens=False).input_ids
    blocks = order_blocks(pack_sessions(order, sys_ids, block_len=block_len, pad_id=tok.pad_token_id
                                        if tok.pad_token_id is not None else tok.eos_token_id,
                                        allow_terminal_truncation=False), shares)
    run, n = 0, None
    for i, b in enumerate(blocks):
        run += b.n_supervised
        if run >= RUNG:
            n = i + 1
            break
    used = blocks[:n]
    mix = Counter()
    for b in used:
        for m in b.audit["sessions"]:
            mix[m["data_type"]] += m["supervised_retained"]
    sup = sum(mix.values())
    real = sum(b.audit["unpadded_length"] for b in used)
    token_mix = {k: round(v / sup, 4) for k, v in sorted(mix.items())}
    return {"packed_sequences": n, "supervised_tokens": sup, "non_padding_tokens": real,
            "padding_ratio": round(1 - real / (n * block_len), 4), "truncated_sessions": 0,
            "optimizer_steps_at_3_epochs": math.ceil(n * EPOCHS / BATCH_SIZE),
            "axis_mix": by_axis(token_mix),
            "mix_minus_declared_pp_max": max(abs(round(100 * (token_mix.get(k, 0) - v), 2)) for k, v in shares.items()),
            "sequences_in_pool": len(blocks)}


# -- compute ------------------------------------------------------------------
def fwd_flops_per_position(cfg: dict, n_params: int) -> dict:
    """Kaplan et al. 2020 (Table 1): forward FLOPs per position = 2N + 2 n_layer
    n_ctx d_attn, N the non-embedding parameters; the unembedding matmul
    (2 d_model n_vocab), which that convention leaves out, is added because a
    151,936-token vocabulary makes it material at this scale."""
    d, v, layers = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    d_attn = cfg["num_attention_heads"] * cfg.get("head_dim", d // cfg["num_attention_heads"])
    non_emb = n_params - v * d
    ctx = 8192
    return {"non_embedding_params": non_emb, "matmul": 2 * non_emb, "attention": 2 * layers * ctx * d_attn,
            "unembedding": 2 * d * v, "total": 2 * non_emb + 2 * layers * ctx * d_attn + 2 * d * v}


def schedules(arms: dict, study: dict) -> dict:
    """What each arm's training costs under E1's batch size and epochs, and the
    schedules that would match it on another quantity instead."""
    cfg = A.read_json(A.REPO / study["recipe"]["e1_configs"]["20260726"]["path"])
    student = A.read_json(A.REPO / study["initializations"]["pca"]["path"] / "config.json")
    teacher = A.read_json(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
                          / cfg["teacher"]["revision"] / "config.json")
    s_f = fwd_flops_per_position(student, 596_049_920)
    t_f = fwd_flops_per_position(teacher, 4_022_468_096)
    # student forward + backward (3x forward) + the activation-recomputation forward; teacher forward (KD)
    per_pos = 4 * s_f["total"] + t_f["total"]
    bps, seq = cfg["batch"]["blocks_per_step"], cfg["block_len"]
    padding_computed = not cfg["batch"].get("truncate_padding", False)

    def plan(a: dict, steps: int) -> dict:
        passes = steps * bps
        positions = passes * seq if padding_computed else round(passes / a["packed_sequences"] * a["non_padding_tokens"])
        return {"optimizer_steps": steps, "epochs": round(passes / a["packed_sequences"], 3),
                "supervised_token_exposure": round(passes / a["packed_sequences"] * a["supervised_tokens"]),
                "non_padding_token_exposure": round(passes / a["packed_sequences"] * a["non_padding_tokens"]),
                "positions_computed": positions, "training_flops": float(f"{positions * per_pos:.4g}")}

    ref = arms["e1_2p96M"]
    e1_steps = math.ceil(ref["packed_sequences"] * EPOCHS / bps)
    r = arms["reasoning_v3_seq"]
    r_steps = math.ceil(r["packed_sequences"] * EPOCHS / bps)
    return {"recipe": {"batch_size": bps, "micro_batch_size": cfg["batch"]["micro_blocks"],
                       "gradient_accumulation_steps": bps // cfg["batch"]["micro_blocks"], "sequence_length": seq,
                       "epochs": EPOCHS, "padding_positions_computed": padding_computed,
                       "_padding": "batch.truncate_padding is absent from E1's config, so every one of a sequence's "
                                   "8,192 positions is forwarded by student and teacher, padding included"},
            "flops_per_position": {"student_forward": s_f, "teacher_forward": t_f, "training_total": per_pos,
                                   "_rule": "student 4 x forward (forward, backward = 2 x forward, recomputed "
                                            "forward for gradient checkpointing) + teacher 1 x forward"},
            "PCA-Original and D4-Original (E1's pack)": plan(ref, e1_steps),
            "D4-Reasoning, epoch-matched (3 epochs)": plan(r, r_steps),
            "D4-Reasoning, step- and FLOPs-matched (2,916 steps)": plan(r, e1_steps),
            "conflict": (f"at batch size {bps} the two packs give {e1_steps} and {r_steps} optimizer steps for 3 "
                         "epochs. No schedule matches epochs, optimizer steps, supervised-token exposure and FLOPs "
                         "at once: the packs hold the same supervised tokens in "
                         f"{ref['packed_sequences']} and {r['packed_sequences']} sequences "
                         f"({ref['supervised_tokens_per_sequence']['mean']} and "
                         f"{r['supervised_tokens_per_sequence']['mean']} supervised tokens per sequence)")}


def budget(sched: dict, *, price_usd_per_hour: float, volume_gb: int) -> dict:
    """L40S cost of each schedule option, from measured rates only: E1 sa's
    own wall time per optimizer step on L40S (its run_end, evaluations
    included), the L40S evaluation time per checkpoint and session overhead
    the H1 forecast measured, and the evaluation contingency it declares (a
    student off the floor writes longer replies than the floor students
    measured). Every arm computes all 8,192 positions of every sequence, so a
    step costs the same in every arm."""
    fc = A.read_json(A.LOG_DIR / "plans/cost_forecast.json")["measured"]
    k = A.read_json(A.LOG_DIR / "plans/cost_forecast.json")["contingency_factors_not_measurements"]
    run_end = [r for r in A.jsonl(A.REPO / "artifacts/stages/stage-3/rescued/_relay/e1_r2960k_sa_pca/train_log.jsonl")
               if r.get("event") == "run_end"][0]
    s_step = run_end["seconds"] / run_end["steps"]
    ev = (fc["eval_seconds_per_unit_L40S"]["max"], fc["eval_seconds_per_unit_L40S"]["max"] * k["eval_floor_to_reasoning"])
    over = fc["session_overhead_seconds_L40S"]["median"]
    e1 = sched["PCA-Original and D4-Original (E1's pack)"]["optimizer_steps"]
    ep = sched["D4-Reasoning, epoch-matched (3 epochs)"]["optimizer_steps"]
    options = {"A: D4-Reasoning epoch-matched": {"train_steps": [e1, e1, ep, ep], "evaluations": 6},
               "B: D4-Reasoning step- and FLOPs-matched": {"train_steps": [e1] * 4, "evaluations": 6},
               "C: both D4-Reasoning schedules": {"train_steps": [e1] * 4 + [ep, ep], "evaluations": 8}}
    storage = round(volume_gb * 0.07 * 2, 2)
    out = {"inputs": {"seconds_per_optimizer_step_L40S": round(s_step, 3), "eval_seconds_per_checkpoint": list(ev),
                      "session_overhead_seconds": over, "price_usd_per_hour": price_usd_per_hour,
                      "volume": f"{volume_gb} GB x $0.07/GB-month x 2 months = ${storage}",
                      "gpu_validation_hours": 0.5,
                      "_validation": "one short L40S session before formal training: a D4-Original seed's first "
                                     "logged steps against E1's logged counts, resume from a save point, and the "
                                     "decoding protocol on the real engine"}}
    for name, o in options.items():
        sessions = len(o["train_steps"]) + 1          # one per training run, one for PCA-Original's evaluation
        hours = [(sum(o["train_steps"]) * s_step + o["evaluations"] * e + sessions * over) / 3600 + 0.5 for e in ev]
        out[name] = {"gpu_hours": [round(h, 1) for h in hours],
                     "usd": [round(h * price_usd_per_hour + storage, 2) for h in hours], "sessions": sessions + 1}
    return out


def cmd_audit(args) -> int:
    from transformers import AutoTokenizer
    study = A.read_json(A.REPO / "configs/stages/stage-2/reasoning_data/h1pa.json")
    tok = AutoTokenizer.from_pretrained(str(PCA_INIT))
    e1_sessions = {s["id"]: s for s in A.jsonl(E1_SESSIONS)}
    r_sessions = {s["id"]: s for s in A.jsonl(Path(args.reasoning_sessions))}
    rds = study["datasets"]["reasoning_v3_seq"]
    pack_r = A.REPO / rds["pack"]
    ident = (pca_original_identity(study, Path(args.relay_sa) if args.relay_sa else None,
                                   Path(args.e1_regen) if args.e1_regen else None)
             + d4_identity(study)
             + [check("D4-Reasoning builder input", A.sha256_file(Path(args.reasoning_sessions)),
                      rds["builder_input_sha256"])]
             + [check(f"D4-Reasoning pack {f}", A.sha256_file(pack_r / f), rds[k])
                for f, k in (("blocks.npz", "blocks_sha256"), ("ladder.json", "ladder_sha256"),
                             ("audit.jsonl", "audit_sha256"))]
             + [check("D4-Reasoning pack built from that input", A.read_json(pack_r / "ladder.json")["sessions_sha256"],
                      rds["builder_input_sha256"])]
             + training_order_replay(study))
    declared_r = A.read_json(pack_r / "ladder.json")["declared_mixture"]
    arms = {"e1_2p96M": pack_metrics(A.REPO / study["datasets"]["e1_2p96M"]["pack"], RUNG, e1_sessions, tok,
                                     unit_of=lambda s: s.get("source_id") or s["id"],
                                     declared=A.read_json(E1_PACK / "ladder.json")["declared_mixture"]),
            "reasoning_v3_seq": pack_metrics(pack_r, RUNG, r_sessions, tok, unit_of=lambda s: s["unit_id"],
                                             declared=declared_r)}
    arms["reasoning_v3_seq"]["axis_mix"] = by_axis(arms["reasoning_v3_seq"]["token_mix"])
    arms["reasoning_v3_seq"]["axis_declared"] = by_axis(declared_r)
    arms["reasoning_v3_seq"]["families"] = family_coverage(pack_r, RUNG, r_sessions)
    arms["reasoning_v3_seq"]["isolation"] = {
        "pool": "R0-R9 passed on every generated unit and the reuse selection's own record "
                "(audit/corpus_v3_feasibility.json :: isolation); packing changes no content"}
    arms["e1_2p96M"]["isolation"] = e1_isolation(A.REPO / study["datasets"]["e1_2p96M"]["pack"], RUNG)
    arms["reasoning_v3_seq"]["alternative_without_terminal_truncation"] = no_truncation_variant(
        list(r_sessions.values()), declared_r, tok)
    doc = {"schema": "aadistill.stage2.reasoning_data.h1pa_audit/v1",
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "study": study["study_id"], "study_config_sha256": A.sha256_file(
               A.REPO / "configs/stages/stage-2/reasoning_data/h1pa.json"),
           "identity": ident, "identity_ok": all(c["ok"] for c in ident),
           "datasets": arms, "schedule": schedules(arms, study), "authorizes": "nothing"}
    doc["budget_L40S"] = budget(doc["schedule"], price_usd_per_hour=args.price, volume_gb=args.volume_gb)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(OUT.relative_to(A.REPO), "identity_ok", doc["identity_ok"])
    for c in ident:
        if not c["ok"]:
            print("  FAILED:", c["check"])
    print(json.dumps(doc["schedule"], indent=1)[:3000])
    return 0 if doc["identity_ok"] else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sessions")
    s.add_argument("--store", action="append", required=True, metavar="RUN_ID=DIR")
    s.add_argument("--out", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--reasoning-sessions", required=True)
    a.add_argument("--relay-sa", default=None, help="a directory holding E1 sa's relay run manifest and pod hashes")
    a.add_argument("--e1-regen", default=None, help="the directory E1's packer rebuilt E1's pack into")
    a.add_argument("--price", type=float, required=True, help="the L40S secure price per hour, read live")
    a.add_argument("--volume-gb", type=int, default=50)
    args = ap.parse_args(argv)
    return {"sessions": cmd_sessions, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
