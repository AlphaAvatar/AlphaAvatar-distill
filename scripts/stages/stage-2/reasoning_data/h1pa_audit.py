#!/usr/bin/env python3
"""H1-PA (the PCA-aligned redesign of H1): its $0 data, capacity and identity audit.

    # 1. the pure fluid-reasoning pool: @v3's PRIMARY verified sessions exactly as
    #    the @v3 pack read them, written as input for E1's own packer
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_audit sessions \
        --store h1gen_20261022_190653=<P1 scr>/store --store h1p1t_20261023_094007=<P1t scr>/store \
        --out artifacts/stages/stage-2/reasoning_data/packs/D4A_fluid_sessions.jsonl
    # 2. E1's packer with its no-truncation switch (the command `sessions` prints)
    # 3. E1's pack + the fluid pack -> D4-Augmented's training set
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_audit merge \
        --fluid-pack artifacts/stages/stage-2/reasoning_data/packs/D4A_fluid \
        --out artifacts/stages/stage-2/reasoning_data/packs/D4A_merged
    # 4. the audit
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.h1pa_audit audit --store ... --store ... \
        --fluid-sessions <1's output> --fluid-pack <2's output> --merged <3's output> \
        --relay-sa <dir with E1 sa's relay run manifest and pod hashes> --e1-regen <E1's pack rebuilt> --price <$/h>

The three arms (maintainer, 2026-10-24, second decision): PCA-Original is E1's
2.96M PCA arm as it ran, re-evaluated and never retrained; D4-Original is D4
Top-1 trained on E1's own ``blocks.npz`` under E1's configuration;
D4-Augmented is D4 Top-1 trained on E1's ``blocks.npz`` UNCHANGED plus 2.96M
supervised tokens of pure fluid-reasoning data (the five P10.2 primary axes),
packed separately by E1's sequential packer without terminal truncation and
appended. The fluid pool takes no @v3 quota: its mixture is the pool's own
supervised-token composition, declared before packing.

What the packer receives is @v3's primary sessions with two fields mapped,
nothing else: ``data_type`` is the session's cell (``stratum/source``) and
``source_id`` is the unit (the conversation an episode's turns were expanded
from, as corpus_v2's ``source_id`` is).

Writes ``logs/stages/stage-2/reasoning_data/audit/h1pa_augmented_audit.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A

E1_PACK = A.REPO / "artifacts/shared/instruments/ladder_uniform_probe"
E1_SESSIONS = A.REPO / "artifacts/stages/stage-3/corpus_v2/sessions.jsonl"
PCA_INIT = A.REPO / "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint"
REASONING_PACK_RECORD = A.REPO / "artifacts/stages/stage-2/reasoning_data/packs/B/ladder.json"
STUDY = A.REPO / "configs/stages/stage-2/reasoning_data/h1pa.json"
OUT = A.LOG_DIR / "audit/h1pa_augmented_audit.json"
RUNG = 2_960_000          #: E1's rung, and the fluid increment's target, in supervised tokens
EPOCHS = 3
PRIMARY = "primary."


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def builder_rows(placed: list[dict]) -> list[dict]:
    """Sessions as E1's packer reads them: the cell as `data_type`, the unit as
    `source_id`; every other field as it is."""
    return [{**s, "data_type": s["stratum"], "source_id": s["unit_id"]} for s in placed]


def derive(stores: dict[str, Path]) -> dict:
    """The sessions the @v3 pack was built from, re-derived from the secured
    stores by the feasibility tool's own functions, their recorded hash checked,
    rendered and admitted as @v3 admitted them; anything else refuses."""
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
    placed = AB.place(sessions, AB.cell_shares(spec, spec["corpus_id"]))
    rendered, excluded, _ = B.render_all(runs[0]["tok"], placed, block_len=int(spec["block_len"]))
    if excluded:
        raise SystemExit(f"{len(excluded)} units no longer admit; the pool is not the one @v3 packed")
    return {"runs": runs, "spec": spec, "placed": placed, "aside": aside, "rendered": rendered,
            "tok": runs[0]["tok"], "sessions_sha256": sha, "stores": stores}


def fluid_of(placed: list[dict]) -> list[dict]:
    return [s for s in placed if s["stratum"].startswith(PRIMARY)]


def natural_shares(rows: list[dict], rendered: dict) -> dict:
    """The pool's own supervised-token composition by cell: the declared
    mixture of the fluid increment, fixed before packing and never tuned."""
    c = Counter()
    for s in rows:
        c[s["stratum"]] += rendered[s["id"]].n_supervised
    t = sum(c.values())
    return {k: round(v / t, 6) for k, v in sorted(c.items())}


def cmd_sessions(args) -> int:
    stores = {k: Path(v) for k, v in (kv.split("=", 1) for kv in args.store)}
    d = derive(stores)
    rows = fluid_of(d["placed"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in builder_rows(rows):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shares = natural_shares(rows, d["rendered"])
    meta = {"source_sessions_sha256": d["sessions_sha256"], "n_sessions": len(rows),
            "n_units": len({s["unit_id"] for s in rows}),
            "supervised_tokens": sum(d["rendered"][s["id"]].n_supervised for s in rows),
            "declared_mixture": shares, "builder_input_sha256": A.sha256_file(out)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "declared_mixture"}, indent=1))
    print("\nE1's packer, its no-truncation switch on:\n.venv/bin/python scripts/shared/data/build_token_ladder.py "
          f"--sessions {out} --model {PCA_INIT.relative_to(A.REPO)} --block-len 8192 --no-terminal-truncation "
          f"--ladder {RUNG} --mixture {','.join(f'{k}={v}' for k, v in shares.items())} --out <pack dir>")
    return 0


# -- the merged training set --------------------------------------------------
def e1_split():
    """E1's training prefix and validation sequences, as E1's trainer reads them."""
    from aadistill.data.ladder import load_ladder_meta, select_val_blocks
    from aadistill.data.sessions import rung_n_blocks
    meta = load_ladder_meta(E1_PACK)
    audit = list(A.jsonl(E1_PACK / "audit.jsonl"))
    n = rung_n_blocks(meta, RUNG)
    largest = max(int(r["n_blocks"]) for r in meta["rungs"] if r.get("reachable"))
    return n, select_val_blocks(audit, largest, 16), audit


def fluid_cut(fluid_pack: Path) -> tuple[int, bool]:
    """The fluid pack's training prefix: the first sequences that reach the
    2.96M target (the packer's own rung), or every sequence of a pool that
    falls short -- which makes the merged set PROVISIONAL."""
    lad = A.read_json(fluid_pack / "ladder.json")
    r = next(x for x in lad["rungs"] if x["target_supervised_tokens"] == RUNG)
    return (int(r["n_blocks"]), True) if r.get("reachable") else (int(lad["n_blocks"]), False)


def cmd_merge(args) -> int:
    """E1's training sequences unchanged, then the fluid increment, then E1's
    16 validation sequences: E1's trainer reads the result as it reads E1's
    pack (`ladder_blocks`), trains on E1's 1,944 sequences plus the
    increment, and validates on E1's own validation sequences."""
    import numpy as np
    from aadistill.data.ladder import ladder_blocks
    fluid = Path(args.fluid_pack).resolve()
    n_e1, val_idx, e1_audit = e1_split()
    n_f, complete = fluid_cut(fluid)
    e1, fl = np.load(E1_PACK / "blocks.npz"), np.load(fluid / "blocks.npz")
    f_audit = list(A.jsonl(fluid / "audit.jsonl"))
    take = {"e1": list(range(n_e1)), "fluid": list(range(n_f)), "e1_validation": list(val_idx)}
    arrays = {k: np.concatenate([e1[k][:n_e1], fl[k][:n_f], e1[k][val_idx]]) for k in ("input_ids", "ce_mask",
                                                                                      "content_mask")}
    rows, cum = [], 0
    for part, src, idx in (("e1", e1_audit, take["e1"]), ("fluid", f_audit, take["fluid"]),
                           ("e1_validation", e1_audit, take["e1_validation"])):
        for i in idx:
            r = {k: v for k, v in src[i].items() if k not in ("block", "cumulative_supervised")}
            if part != "e1_validation":
                cum += r["supervised_tokens"]
            rows.append({"block": len(rows), "cumulative_supervised": cum, "part": part, "source_index": i, **r})
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "blocks.npz", **arrays)
    with (out / "audit.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    e1_lad, f_lad = A.read_json(E1_PACK / "ladder.json"), A.read_json(fluid / "ladder.json")
    train_sup = cum
    rung = {"target_supervised_tokens": train_sup, "reachable": True, "n_blocks": n_e1 + n_f,
            "actual_supervised_tokens": train_sup,
            "e1_supervised_tokens": sum(e1_audit[i]["supervised_tokens"] for i in take["e1"]),
            "fluid_supervised_tokens": sum(f_audit[i]["supervised_tokens"] for i in take["fluid"])}
    (out / "ladder.json").write_text(json.dumps({
        "schema": "aadistill.stage2.reasoning_data.h1pa_merged_pack/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "block_len": e1_lad["block_len"], "pad_id": e1_lad["pad_id"], "n_blocks": len(rows),
        "status": "COMPLETE: the fluid increment reaches 2.96M" if complete else
                  "PROVISIONAL: the fluid pool falls short of 2.96M; not a training set",
        "parts": {"e1": {"pack": str(E1_PACK.relative_to(A.REPO)), "blocks_sha256": A.sha256_file(E1_PACK / "blocks.npz"),
                         "sequences": [0, n_e1], "packing": e1_lad["packing"]},
                  "fluid": {"pack": str(fluid.relative_to(A.REPO)), "blocks_sha256": A.sha256_file(fluid / "blocks.npz"),
                            "sequences": [0, n_f], "packing": f_lad["packing"]},
                  "e1_validation": {"sequences": list(val_idx)}},
        "packing": "two packs appended, never re-packed: E1's sequential packing with terminal truncation for "
                   "E1's part, the same packer without truncation for the fluid part",
        "declared_mixture": {"e1": e1_lad["declared_mixture"], "fluid": f_lad["declared_mixture"]},
        "rungs": [rung],
        "outputs": {"blocks": A.sha256_file(out / "blocks.npz"), "audit": A.sha256_file(out / "audit.jsonl")},
    }, indent=1) + "\n")
    # E1's trainer reads it: E1's sequences byte for byte, then the increment; E1's validation
    (ids, mask, content), (vids, vmask, vcontent), stats = ladder_blocks(out, train_sup, n_val=16)
    same = (np.array_equal(ids[:n_e1].numpy(), e1["input_ids"][:n_e1]) and
            np.array_equal(mask[:n_e1].numpy(), e1["ce_mask"][:n_e1]) and
            np.array_equal(content[:n_e1].numpy(), e1["content_mask"][:n_e1]) and
            np.array_equal(vids.numpy(), e1["input_ids"][val_idx]) and
            np.array_equal(ids[n_e1:].numpy(), fl["input_ids"][:n_f]))
    print(json.dumps({"out": str(out), "status": "COMPLETE" if complete else "PROVISIONAL",
                      "train_sequences": n_e1 + n_f, "train_supervised": train_sup,
                      "e1_part_and_validation_byte_identical": same}, indent=1))
    return 0 if same else 3


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
    out.append({"check": "D4 Top-1 tensors equal the PCA student's (names and shapes)", "tensors": len(shapes["d4"]),
                "shape_map_sha256": canonical_sha(shapes["d4"]), "ok": shapes["d4"] == shapes["pca"]})
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


def merged_identity(merged: Path) -> list[dict]:
    """The merged set holds E1's training and validation sequences byte for
    byte, and the fluid pack's sequences as packed."""
    import numpy as np
    lad = A.read_json(merged / "ladder.json")
    n_e1, val_idx, _ = e1_split()
    m, e1 = np.load(merged / "blocks.npz"), np.load(E1_PACK / "blocks.npz")
    fluid = A.REPO / lad["parts"]["fluid"]["pack"]
    fl = np.load(fluid / "blocks.npz")
    n_f = lad["parts"]["fluid"]["sequences"][1]
    keys = ("input_ids", "ce_mask", "content_mask")
    return [check("merged: E1's 1,944 training sequences byte for byte",
                  all(np.array_equal(m[k][:n_e1], e1[k][:n_e1]) for k in keys), True),
            check("merged: E1's 16 validation sequences byte for byte",
                  all(np.array_equal(m[k][n_e1 + n_f:], e1[k][val_idx]) for k in keys), True),
            check("merged: the fluid sequences as packed",
                  all(np.array_equal(m[k][n_e1:n_e1 + n_f], fl[k][:n_f]) for k in keys), True),
            check("merged: its E1 part names E1's pack", lad["parts"]["e1"]["blocks_sha256"],
                  A.sha256_file(E1_PACK / "blocks.npz"))]


# -- capacity -----------------------------------------------------------------
def capacity(d: dict) -> dict:
    """Every primary cell's qualified supply: its verified units and supervised
    tokens, its measured acceptance over P1 and P1t, the tasks its task table
    still holds that neither session generated, and what was excluded and why."""
    from stages.reasoning_data import corpus_feasibility as CF
    from stages.reasoning_data import p1v
    spec, runs, rendered = d["spec"], d["runs"], d["rendered"]
    rows = fluid_of(d["placed"])
    units = defaultdict(lambda: defaultdict(int))
    for s in rows:
        units[s["stratum"]][s["unit_id"]] += rendered[s["id"]].n_supervised
    tasks = runs[0]["tasks"]
    if any(r["tasks"].keys() != tasks.keys() for r in runs):
        raise SystemExit("the sessions' task tables differ; the unused pool is not one set")
    used = {c.unit_id for r in runs for c in r["got"]["candidates"]}
    cells = {}
    for cell in sorted({p1v.cell_of(t, spec) for t in tasks.values()} | set(units)):
        if not cell.startswith(PRIMARY):
            continue
        in_cell = {tid for tid, t in tasks.items() if p1v.cell_of(t, spec) == cell}
        gens = [CF.generation_measured(r["records"], tasks, r["got"], cell, spec) for r in runs
                if any(c.unit_id in in_cell and c.candidate_index == 0 for c in r["got"]["candidates"])]
        gen_tasks = sum(g["tasks_generated"] for g in gens)
        acc = sum(sum(g["accepted"].values()) for g in gens)
        tok = sum(sum(g["generated_tokens"].values()) for g in gens)
        sizes = sorted(units.get(cell, {}).values())
        pool = [t for t in tasks.values() if p1v.cell_of(t, spec) == cell]
        unused = [t for t in pool if t["task_id"] not in used]
        cells[cell] = {"units": len(sizes), "supervised_tokens": sum(sizes),
                       "supervised_per_unit": {"mean": round(sum(sizes) / len(sizes), 1) if sizes else None,
                                               "median": statistics.median(sizes) if sizes else None},
                       "tasks_generated": gen_tasks, "units_accepted": acc,
                       "unit_accept_rate": round(acc / gen_tasks, 4) if gen_tasks else None,
                       "generated_tokens_per_task": round(tok / gen_tasks) if gen_tasks else None,
                       "task_table": len(pool), "unused_tasks": len(unused),
                       "in_this_pool": bool(sizes)}
    aside = [{"unit_id": a["unit_id"], "cell": a["cell"]} for a in d["aside"]]
    return {"cells": cells,
            "supervised_tokens": sum(c["supervised_tokens"] for c in cells.values() if c["in_this_pool"]),
            "units": sum(c["units"] for c in cells.values() if c["in_this_pool"]),
            "excluded": {
                "set_aside_by_v3": {"units": aside,
                                    "why": "the 2026-10-23 source decision: analogy rests on the project generator "
                                           "alone, Reasoning Core's analogy module leaves the stratum; it stays out "
                                           "unless the maintainer re-admits that module"},
                "supporting_cells": "never counted toward the fluid 2.96M (maintainer, 2026-10-24)",
                "base_pool_v1": "not legal training data: R5 (its seeds are calibration-split, outside the training "
                                "range) and R0 (historical provenance) in audit/asset_audit.json, and it is D4's own "
                                "initialization calibration pool",
                "rejected_candidates": "unverified or incorrect teacher outputs are never trained on"}}


def e1_overlap(rows: list[dict]) -> dict:
    """The fluid pool against every corpus_v2 session E1's pack could hold."""
    e1 = {_prompt(s) for s in A.jsonl(E1_SESSIONS)}
    hit = [s["id"] for s in rows if _prompt(s) in e1]
    return {"fluid_sessions": len(rows), "sharing_a_user_prompt_with_corpus_v2": len(hit), "examples": hit[:5]}


def topup(cap: dict, d: dict, *, gap: int, price_a100: float) -> dict:
    """The missing supervised tokens as unused tasks to generate, sized and
    priced from what P1 and P1t measured.

    Allocation, fixed before any count is seen: the gap is split over the
    cells in proportion to each cell's share of the existing primary pool (the
    pool's own composition), each cell's tasks sized for its measured unit
    acceptance and mean supervised tokens per unit with P1's planning rule
    (`tasks_for`: 1.5x oversupply or a 90% binomial chance, whichever is
    larger), and selected round-robin over its unused tasks' groups (P1's
    rule; no group chosen by acceptance). Cost: P1's measured session and
    decode speed, two passes, P1's hard-cap rule."""
    from stages.reasoning_data import corpus_feasibility as CF
    from stages.reasoning_data import generation_plan as GP
    from stages.reasoning_data import isolation_check as IC
    from stages.reasoning_data import p1v
    spec, runs = d["spec"], d["runs"]
    tasks = runs[0]["tasks"]
    used = {c.unit_id for r in runs for c in r["got"]["candidates"]}
    pool_cells = {k: v for k, v in cap["cells"].items() if v["in_this_pool"]}
    total = sum(v["supervised_tokens"] for v in pool_cells.values())
    rows, selected_all = {}, []
    for cell, v in pool_cells.items():
        want = gap * v["supervised_tokens"] / total
        units = math.ceil(want / v["supervised_per_unit"]["mean"])
        need = CF.tasks_for(units, v["unit_accept_rate"])
        unused = [t for t in tasks.values() if p1v.cell_of(t, spec) == cell and t["task_id"] not in used]
        sel = GP.round_robin(unused, need["tasks"])
        selected_all += sel
        rows[cell] = {"supervised_tokens_wanted": round(want), "units_wanted": units, "tasks": need,
                      "unused_tasks": len(unused), "enough_unused_tasks": len(unused) >= need["tasks"],
                      "expected_supervised_tokens": round(need["tasks"] * v["unit_accept_rate"]
                                                          * v["supervised_per_unit"]["mean"]),
                      "predicted_generated_tokens": need["tasks"] * v["generated_tokens_per_task"],
                      "selected_task_ids_sha256": canonical_sha(sorted(t["task_id"] for t in sel))}
    iso = IC.check_pool([{**t, "unit_id": t["task_id"], "messages": t.get("messages") or []} for t in selected_all])
    n_tasks = sum(r["tasks"]["tasks"] for r in rows.values())
    return {"gap_supervised_tokens": gap, "cells": rows, "tasks_total": n_tasks,
            "selected_task_ids": sorted(t["task_id"] for t in selected_all),
            "all_cells_have_enough_unused_tasks": all(r["enough_unused_tasks"] for r in rows.values()),
            "expected_supervised_tokens": sum(r["expected_supervised_tokens"] for r in rows.values()),
            "predicted_generated_tokens": sum(r["predicted_generated_tokens"] for r in rows.values()),
            "isolation_of_the_selected_tasks": {"pass": iso["pass"], "violations": iso["violations"][:10],
                                                "_": "prompt-level rules on the tasks before generation; R0-R9 run "
                                                     "again on the accepted units with their teacher text"},
            "generation": "P1's frozen protocol (the same teacher, sampling, allowance and verifiers; two candidates, "
                          "the second only when the first is rejected), then P1v verification at $0",
            "cost": generation_forecast(rows, d, price=price_a100)}


# -- the generation session's cost --------------------------------------------
def ps_seconds(tokens: list[int], rate: float, cap: float) -> float:
    """Processor sharing: while n requests are active each decodes at
    min(cap, rate / n) tokens/s -- an engine whose aggregate throughput
    saturates at `rate`, a single sequence at `cap`. An episode is one request
    of its turns' total tokens (its turns are generated one after another)."""
    rem, t, p = sorted(tokens), 0.0, 0.0
    for i, x in enumerate(rem):
        t += (x - p) / min(cap, rate / (len(rem) - i))
        p = x
    return t


def generation_calibration(runs: list[dict]) -> dict:
    """Every rate the session forecast uses, measured on P1's and P1t's four
    passes: the saturated aggregate throughput (a pass's tokens over its
    slowest request's latency, under P1's full load; the second pass as the
    conservative value), the single-sequence speed cap (the fastest per-pass
    median per-sequence speed), and the model's largest under-prediction of a
    measured pass, applied as a correction."""
    passes = []
    for r in runs:
        for ci in (0, 1):
            recs = [x for x in r["records"] if x["candidate_index"] == ci]
            toks = [sum(t["generated_tokens"] for t in x["turns"]) for x in recs]
            speeds = [t["generated_tokens"] / (t["seconds"] - t["ttft_seconds"]) for x in recs for t in x["turns"]
                      if t["generated_tokens"] > 1000 and t["seconds"] > t["ttft_seconds"]]
            passes.append({"run": r["identity"]["run_id"], "pass": ci, "requests": len(recs), "tokens": sum(toks),
                           "slowest_request_seconds": round(max(sum(t["seconds"] for t in x["turns"]) for x in recs)),
                           "median_sequence_speed": round(statistics.median(speeds), 1), "_tokens": toks})
    p1 = [p for p in passes if p["run"] == runs[0]["identity"]["run_id"]]
    rate = p1[0]["tokens"] / p1[0]["slowest_request_seconds"]
    rate_low = p1[1]["tokens"] / p1[1]["slowest_request_seconds"]
    cap = max(p["median_sequence_speed"] for p in passes)
    for p in passes:
        p["model_seconds"] = round(ps_seconds(p.pop("_tokens"), rate, cap))
    corr = max(p["slowest_request_seconds"] / p["model_seconds"] for p in passes)
    return {"aggregate_tokens_per_second": round(rate), "aggregate_tokens_per_second_conservative": round(rate_low),
            "sequence_speed_cap": cap, "correction": round(corr, 3), "passes": passes}


def generation_forecast(rows: dict, d: dict, *, price: float) -> dict:
    """The top-up session's minutes and dollars, from its own tasks.

    Each pass is simulated as processor sharing over its requests' generated
    tokens (`ps_seconds`), then multiplied by the calibration's correction:
    * EXPECTED: each cell's tasks take its measured per-request token
      distribution (evenly spaced quantiles of P1 and P1t's first and second
      candidates), and the second pass holds the share its first candidates
      measured as rejected;
    * CONSERVATIVE: every task at its cell's 90th-percentile first-candidate
      tokens and one at the cell's longest measured request, every task needing
      a second candidate likewise, and the conservative aggregate throughput.
    Session phases outside the passes (pod to driver, the contract check, both
    engine starts, ALL_DONE to the pod deleted) are P1t's measured ones; P1's
    hard-cap rule is 1.5 x (expected + a closeout reserve of setup + teardown +
    15 min)."""
    from stages.reasoning_data import corpus_feasibility as CF
    from stages.reasoning_data import p1v
    runs, spec = d["runs"], d["spec"]
    cal = generation_calibration(runs)
    tasks = runs[0]["tasks"]
    per = defaultdict(lambda: {0: [], 1: []})
    for r in runs:
        for x in r["records"]:
            per[p1v.cell_of(tasks[x["task_id"]], spec)][x["candidate_index"]].append(
                sum(t["generated_tokens"] for t in x["turns"]))
    acc0 = {}
    for cell in rows:
        in_cell = {tid for tid, t in tasks.items() if p1v.cell_of(t, spec) == cell}
        firsts = [c for r in runs for c in r["got"]["candidates"] if c.unit_id in in_cell and c.candidate_index == 0]
        acc0[cell] = sum(bool(c.accepted_for_training) for c in firsts) / len(firsts)

    def spread(xs: list[int], k: int) -> list[int]:
        xs = sorted(xs)
        return [xs[min(len(xs) - 1, int((i + 0.5) / k * len(xs)))] for i in range(k)]

    exp0, exp1, con0, con1 = [], [], [], []
    for cell, row in rows.items():
        k = row["tasks"]["tasks"]
        first, second = per[cell][0], per[cell][1] or per[cell][0]
        exp0 += spread(first, k)
        exp1 += spread(second, round(k * (1 - acc0[cell])))
        con0 += [pctl(first, 0.9)] * (k - 1) + [max(first)]
        con1 += [pctl(second, 0.9)] * (k - 1) + [max(second)]
    p1t = next(r for r in runs if r["identity"]["phase"] == "P1t")["identity"]["run_id"]
    sess = CF.session_measured(p1t, d["stores"][p1t])
    tl = A.read_json(A.LOG_DIR / f"runs/{p1t}/runtime/session.json")["timeline"]
    contract = next(e["elapsed_min"] for e in tl if "MARKER:CONTRACT_OK" in e["msg"]) - sess["pod_to_driver_minutes"]
    fixed = (sess["pod_to_driver_minutes"] + contract + sum(sess["engine_start_seconds_by_pass"]) / 60
             + sess["all_done_to_pod_deleted_minutes"])
    rate = price * sess["all_in_over_gpu"] / 60

    def minutes(p0, p1, agg):
        return fixed + cal["correction"] * (ps_seconds(p0, agg, cal["sequence_speed_cap"])
                                            + ps_seconds(p1, agg, cal["sequence_speed_cap"])) / 60

    m_exp = minutes(exp0, exp1, cal["aggregate_tokens_per_second"])
    m_con = minutes(con0, con1, cal["aggregate_tokens_per_second_conservative"])
    reserve = (sess["pod_to_driver_minutes"] + sess["all_done_to_pod_deleted_minutes"] + 15) * rate
    hard = 1.5 * (m_exp * rate + reserve)
    return {"basis": "A100-SXM4-80GB, the hardware P1 and P1t measured (an L40S session would first need its own "
                     "decode measurement)", "price_per_hour": price, "calibration": cal,
            "generated_tokens": {"expected": sum(exp0) + sum(exp1), "conservative": sum(con0) + sum(con1)},
            "requests": {"expected": [len(exp0), len(exp1)], "conservative": [len(con0), len(con1)]},
            "fixed_minutes": round(fixed, 1),
            "session_minutes": {"expected": round(m_exp, 1), "conservative": round(m_con, 1)},
            "expected_usd": round(m_exp * rate, 2), "conservative_usd": round(m_con * rate, 2),
            "hard_cap_usd": round(hard, 2), "watchdog_hard_minutes": round(hard / rate, 1),
            "conservative_fits_the_cap": m_con * rate <= hard}


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
    disc, kinds = Counter(), Counter()
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
    return {"discarded_supervised": dict(disc), "retained_target": dict(kinds)}


def seq_metrics(rows: list[dict], sessions: dict[str, dict], tok, *, unit_of, block_len: int = 8192) -> dict:
    members = [m for r in rows for m in r["sessions"]]
    trunc = [m for m in members if m["truncated"]]
    n, sup = len(rows), sum(r["supervised_tokens"] for r in rows)
    real = sum(r["unpadded_length"] for r in rows)
    per_seq = [r["supervised_tokens"] for r in rows]
    mix = Counter()
    for m in members:
        mix[m["data_type"]] += m["supervised_retained"]
    return {"packed_sequences": n, "source_sessions": len(members),
            "units": len({unit_of(sessions[m["session_id"]]) for m in members}),
            "supervised_tokens": sup, "non_padding_tokens": real, "padding_tokens": n * block_len - real,
            "padding_ratio": round(1 - real / (n * block_len), 4),
            "supervised_tokens_per_sequence": {"p10": pctl(per_seq, 0.1), "p50": pctl(per_seq, 0.5),
                                               "p90": pctl(per_seq, 0.9), "mean": round(sup / n, 1)},
            "truncated_sessions": len(trunc),
            "supervised_discarded_by_truncation": sum(m["supervised_discarded"] for m in members),
            "truncation": truncation_detail(trunc, sessions, tok),
            "token_mix": {k: round(v / sup, 4) for k, v in sorted(mix.items())}}


def packing_stats(rows: list[dict], sessions: dict[str, dict], *, unit_of, axis_of, source_of,
                  block_len: int = 8192) -> dict:
    """What the sequential packer actually put together: system-prompt groups,
    sequences holding several axes, cells or sources, samples per sequence,
    which axes share sequences, whole-answer retention, a unit never twice in
    one sequence, and every member traceable to its task, source, axis and unit."""
    groups, multi = Counter(), Counter()
    pairs, per_seq, units_per = Counter(), [], []
    whole = total = twice = traced = 0
    for r in rows:
        ms = r["sessions"]
        groups[r["system_sha256"]] += 1
        ss = [sessions[m["session_id"]] for m in ms]
        axes = sorted({axis_of(x) for x in ss})
        multi["axes"] += len(axes) > 1
        multi["cells"] += len({m["data_type"] for m in ms}) > 1
        multi["sources"] += len({source_of(x) for x in ss}) > 1
        for i, a in enumerate(axes):
            for b in axes[i + 1:]:
                pairs[f"{a} + {b}"] += 1
        per_seq.append(len(ms))
        u = [unit_of(x) for x in ss]
        units_per.append(len(set(u)))
        twice += len(u) != len(set(u))
        total += len(ms)
        whole += sum(not m["truncated"] and not m["supervised_discarded"] for m in ms)
        traced += sum(bool(unit_of(x) and axis_of(x) and source_of(x)) for x in ss)
    real = sum(r["unpadded_length"] for r in rows)
    return {"packed_sequences": len(rows), "system_prompt_groups": len(groups),
            "sequences_with_two_or_more": {"axes": multi["axes"], "cells": multi["cells"], "sources": multi["sources"]},
            "training_samples_per_sequence": round(sum(per_seq) / len(rows), 2),
            "units_per_sequence": round(sum(units_per) / len(rows), 2),
            "samples_per_sequence_distribution": dict(sorted(Counter(per_seq).items())),
            "axes_sharing_a_sequence": dict(pairs.most_common()),
            "sequences_holding_one_unit_twice": twice,
            "padding_ratio": round(1 - real / (len(rows) * block_len), 4),
            "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
            "samples_packed_whole": f"{whole}/{total}",
            "members_traceable_to_task_source_axis_unit": f"{traced}/{total}"}


def cmd_pregen(args) -> int:
    """The pre-generation review (maintainer, 2026-10-24): how the packer
    really groups, and the top-up session priced from its own tasks; writes the
    review and the generation application."""
    stores = {k: Path(v) for k, v in (kv.split("=", 1) for kv in args.store)}
    d = derive(stores)
    f_sessions = {s["id"]: s for s in A.jsonl(Path(args.fluid_sessions))}
    fluid = Path(args.fluid_pack).resolve()
    f_rows = list(A.jsonl(fluid / "audit.jsonl"))
    n_e1, _, e1_audit = e1_split()
    e1_sessions = {s["id"]: s for s in A.jsonl(E1_SESSIONS)}
    fl = packing_stats(f_rows, f_sessions, unit_of=lambda x: x["unit_id"], axis_of=lambda x: x["stratum"].split("/")[0],
                       source_of=lambda x: (x.get("attributes") or {}).get("source"))
    e1 = packing_stats(e1_audit[:n_e1], e1_sessions, unit_of=lambda x: x.get("source_id") or x["id"],
                       axis_of=lambda x: x["data_type"], source_of=lambda x: x.get("source"))
    default_key = Counter(r["system_sha256"] for r in f_rows).most_common(1)[0][0]
    e1_default = [r for r in e1_audit[:n_e1] if r["system_sha256"] == default_key]
    e1_def = packing_stats(e1_default, e1_sessions, unit_of=lambda x: x.get("source_id") or x["id"],
                           axis_of=lambda x: x["data_type"], source_of=lambda x: x.get("source"))
    pad = sum(r["padding_length"] for r in f_rows)
    joint = {"what": "re-packing E1's default-system-prompt sessions together with the fluid increment",
             "the_only_shareable_group": "E1's default system prompt is the fluid data's ('You are a helpful "
                                         "Assistant.'): E1's tool-calling sequences carry their own tool schemas "
                                         "and can share with nothing",
             "e1_default_sequences": e1_def["packed_sequences"], "e1_default_padding_ratio": e1_def["padding_ratio"],
             "fluid_padding_positions": pad,
             "upper_bound_sequences_saved": pad // 8192,
             "upper_bound_optimizer_steps_saved": math.ceil(pad // 8192 * EPOCHS / 2),
             "_bound": "every fluid padding position filled by E1 tokens; E1's default sequences are already this full, "
                       "so the real saving is smaller",
             "scientific_cost": "D4-Augmented's E1 part would stop being E1's pack: E1 sessions re-placed, their "
                                "truncation re-decided (or removed), so D4-Augmented no longer equals D4-Original's "
                                "training data plus an increment and Comparison B loses its clean attribution; "
                                "PCA-Original and D4-Original keep E1's pack either way",
             "recommendation": "keep the two-part layout; the bound is too small to pay that cost"}
    cap = capacity(d)
    gap = max(0, RUNG - cap["supervised_tokens"])
    scen = {"115 tasks: the gap": topup(cap, d, gap=gap, price_a100=args.a100_price),
            f"{0} tasks: the gap and a {args.reserve:,} reserve": topup(cap, d, gap=gap + args.reserve,
                                                                        price_a100=args.a100_price)}
    scen = {k.replace("0 tasks", f"{v['tasks_total']} tasks") if k.startswith("0 ") else k: v for k, v in scen.items()}
    review = {"schema": "aadistill.stage2.reasoning_data.h1pa_pregen_review/v1",
              "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "code": {"grouping": "aadistill.data.sessions.pack_sessions groups by system_group_key (the rendered "
                                   "system text and tool schemas) and by nothing else; pack_group fills each "
                                   "sequence in the given order",
                       "category_and_source": "read only to ORDER sessions (mixture.interleave) and sequences "
                                              "(mixture.order_blocks) and to record them in the audit; neither is a "
                                              "packing boundary",
                       "same_conversation": "a unit's turns never share a sequence (source_id deferral): one turn "
                                            "would read another's answer",
                       "no_category_boundary_found": True},
              "fluid_increment": fl, "e1_pack_2p96M": e1, "e1_default_prompt_sequences": e1_def,
              "joint_repacking": joint,
              "train_validation": "every arm validates on E1's 16 sequences; the fluid tokens past the 2.96M cut are "
                                  "never trained, and any later fluid validation drawn from them excludes every unit "
                                  "with a turn inside the cut",
              "authorizes": "nothing"}
    out = A.LOG_DIR / "audit/h1pa_pregen_review.json"
    out.write_text(json.dumps(review, indent=1) + "\n")
    rows = {}
    for name, t in scen.items():
        c = t["cost"]
        rows[name] = {"tasks": t["tasks_total"], "tasks_by_cell": {k: v["tasks"]["tasks"] for k, v in t["cells"].items()},
                      "expected_supervised_tokens": t["expected_supervised_tokens"],
                      "selected_task_ids_sha256": canonical_sha(t["selected_task_ids"]),
                      "isolation_pass": t["isolation_of_the_selected_tasks"]["pass"],
                      "generated_tokens": c["generated_tokens"], "session_minutes": c["session_minutes"],
                      "expected_usd": c["expected_usd"], "conservative_usd": c["conservative_usd"],
                      "hard_cap_usd": c["hard_cap_usd"], "watchdog_hard_minutes": c["watchdog_hard_minutes"],
                      "conservative_fits_the_cap": c["conservative_fits_the_cap"]}
    first = next(iter(scen.values()))
    app = {"schema": "aadistill.stage2.reasoning_data.topup_application/v2",
           "written_utc": review["written_utc"],
           "status": "APPLICATION for the maintainer's decision; NOT authorized, nothing launched",
           "what": "ONE A100-80GB session generating teacher candidates for unused primary fluid-reasoning tasks under "
                   "P1's frozen generation protocol, then P1v verification and the no-truncation fluid pack at $0",
           "requested": next(iter(rows)),
           "scenarios": rows,
           "calibration": first["cost"]["calibration"],
           "unchanged": ["the teacher, sampling, allowance, seeds rule and verifiers of P1 "
                         "(configs/stages/stage-2/reasoning_data/generation_protocol.json)",
                         "one unit per task, no repeated prompt or target, R0-R9 isolation on the accepted units",
                         "P1 and P1t's data: nothing regenerated"],
           "launch": "NOT built: on a grant, the task list, its generation plan, launch identity and harness digest are "
                     "built at $0 and the grant names them; the existing launcher, shard securing, teardown and "
                     "closeout apply unchanged",
           "authorizes": "nothing"}
    (A.LOG_DIR / "plans/h1pa_topup_application.json").write_text(json.dumps(app, indent=1) + "\n")
    print(json.dumps({"fluid": {k: fl[k] for k in ("system_prompt_groups", "sequences_with_two_or_more",
                                                   "training_samples_per_sequence", "samples_packed_whole")},
                      "scenarios": rows}, indent=1)[:4000])
    return 0


def _prompt(s: dict) -> str:
    return hashlib.sha256(next(m["content"] for m in s["messages"] if m["role"] == "user").encode()).hexdigest()


def by_axis(mix: dict) -> dict:
    out = Counter()
    for k, v in mix.items():
        out[k.split("/")[0]] += v
    return {k: round(v, 4) for k, v in sorted(out.items())}


def e1_isolation(pack: Path) -> dict:
    """E1's pack against eval-set-1, from the isolation audit H1 already ran
    over every corpus_v2 session: which flagged sessions E1 trained on."""
    from aadistill.data.sessions import rung_n_blocks
    iso = A.read_json(A.LOG_DIR / "audit/asset_audit.json")["isolation"]["corpus_v2_sessions"]["results"]
    n = rung_n_blocks(A.read_json(pack / "ladder.json"), RUNG)
    trained = {m["session_id"] for r in list(A.jsonl(pack / "audit.jsonl"))[:n] for m in r["sessions"]}
    r2 = iso["R2_near_duplicate"]
    hit = [f for f in r2["flagged_items"] if f["id"] in trained]
    return {"R1_held_out_keys": iso["R1_held_out_keys"],
            "R2_near_duplicate_trained": {"n": len(hit), "of_flagged": r2["flagged"],
                                          "by_unit": dict(Counter(f["unit"] for f in hit)),
                                          "fluid_reasoning_720": r2["by_unit"]["fluid_reasoning_720/dev_v1"]},
            "R8_in_sample": iso["R8_in_sample_overlap"]}


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


def schedules(e1: dict, fluid_measured: dict, fluid_forecast: dict, study: dict) -> dict:
    """D4-Original and D4-Augmented under E1's batch size and 3 epochs. The
    trainer forwards all 8,192 positions of a sequence (E1's config leaves
    `batch.truncate_padding` off), so FLOPs follow the sequence count."""
    cfg = A.read_json(A.REPO / study["recipe"]["e1_configs"]["20260726"]["path"])
    student = A.read_json(A.REPO / study["initializations"]["pca"]["path"] / "config.json")
    teacher = A.read_json(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
                          / cfg["teacher"]["revision"] / "config.json")
    s_f, t_f = fwd_flops_per_position(student, 596_049_920), fwd_flops_per_position(teacher, 4_022_468_096)
    per_pos = 4 * s_f["total"] + t_f["total"]
    bps, seq = cfg["batch"]["blocks_per_step"], cfg["block_len"]

    def plan(n: int, sup: int, real: int) -> dict:
        steps = math.ceil(n * EPOCHS / bps)
        passes = steps * bps
        return {"packed_sequences": n, "optimizer_steps": steps, "epochs": round(passes / n, 3),
                "warmup_steps": max(10, round(0.05 * steps)), "unique_supervised_tokens": sup,
                "supervised_token_exposure": round(passes / n * sup),
                "non_padding_token_exposure": round(passes / n * real),
                "positions_computed": passes * seq, "training_flops": float(f"{passes * seq * per_pos:.4g}")}

    aug_m = plan(e1["packed_sequences"] + fluid_measured["packed_sequences"],
                 e1["supervised_tokens"] + fluid_measured["supervised_tokens"],
                 e1["non_padding_tokens"] + fluid_measured["non_padding_tokens"])
    aug_f = plan(e1["packed_sequences"] + fluid_forecast["packed_sequences"],
                 e1["supervised_tokens"] + fluid_forecast["supervised_tokens"],
                 e1["non_padding_tokens"] + fluid_forecast["non_padding_tokens"])
    return {"recipe": {"batch_size": bps, "micro_batch_size": cfg["batch"]["micro_blocks"],
                       "gradient_accumulation_steps": bps // cfg["batch"]["micro_blocks"], "sequence_length": seq,
                       "epochs": EPOCHS, "padding_positions_computed": not cfg["batch"].get("truncate_padding", False),
                       "lr_rule": "E1's builder rule: warmup max(10, round(0.05 T)), cosine to 0.1 of peak; "
                                  "eval every max(25, T // 8), save every max(200, T // 2)"},
            "flops_per_position": {"student_forward": s_f, "teacher_forward": t_f, "training_total": per_pos,
                                   "_rule": "student 4 x forward (forward, backward = 2 x forward, recomputed "
                                            "forward for gradient checkpointing) + teacher 1 x forward"},
            "PCA-Original and D4-Original": plan(e1["packed_sequences"], e1["supervised_tokens"],
                                                 e1["non_padding_tokens"]),
            "D4-Augmented, measured on today's fluid pool (short of 2.96M)": aug_m,
            "D4-Augmented, forecast with the fluid increment at 2.96M": aug_f,
            "reading": "D4-Augmented's gain, if any, combines the added data and the added optimizer steps and "
                       "FLOPs it brings; it is not the data content's isolated causal effect (maintainer, 2026-10-24)"}


def budget(sched: dict, gen_cost: dict, *, price: float, volume_gb: int) -> dict:
    """L40S cost from measured rates only: E1 sa's own wall time per optimizer
    step on L40S (its run_end, evaluations included), the L40S evaluation time
    per checkpoint and session overhead the H1 forecast measured, and the
    evaluation contingency it declares; the top-up generation as priced above."""
    fc = A.read_json(A.LOG_DIR / "plans/cost_forecast.json")
    run_end = [r for r in A.jsonl(A.REPO / "artifacts/stages/stage-3/rescued/_relay/e1_r2960k_sa_pca/train_log.jsonl")
               if r.get("event") == "run_end"][0]
    s_step = run_end["seconds"] / run_end["steps"]
    m = fc["measured"]
    ev = (m["eval_seconds_per_unit_L40S"]["max"],
          m["eval_seconds_per_unit_L40S"]["max"] * fc["contingency_factors_not_measurements"]["eval_floor_to_reasoning"])
    over = m["session_overhead_seconds_L40S"]["median"]
    steps = ([sched["PCA-Original and D4-Original"]["optimizer_steps"]] * 2
             + [sched["D4-Augmented, forecast with the fluid increment at 2.96M"]["optimizer_steps"]] * 2)
    sessions = 4 + 1 + 1          # one per training run, PCA-Original's evaluation, the GPU validation
    storage = round(volume_gb * 0.07 * 2, 2)
    hours = [(sum(steps) * s_step + 6 * e + sessions * over) / 3600 + 0.5 for e in ev]
    return {"inputs": {"seconds_per_optimizer_step_L40S": round(s_step, 3), "training_steps": steps,
                       "evaluations": 6, "eval_seconds_per_checkpoint": list(ev), "session_overhead_seconds": over,
                       "price_usd_per_hour": price, "gpu_validation_hours": 0.5,
                       "volume": f"{volume_gb} GB x $0.07/GB-month x 2 months = ${storage}"},
            "training_and_evaluation": {"gpu_hours": [round(h, 1) for h in hours],
                                        "usd": [round(h * price, 2) for h in hours]},
            "top_up_generation": {"expected_usd": gen_cost["expected_usd"], "hard_cap_usd": gen_cost["hard_cap_usd"]},
            "storage_usd": storage,
            "total_usd": [round(h * price + storage + gen_cost["expected_usd"], 2) for h in hours]}


def cmd_audit(args) -> int:
    from transformers import AutoTokenizer
    study = A.read_json(STUDY)
    tok = AutoTokenizer.from_pretrained(str(PCA_INIT))
    stores = {k: Path(v) for k, v in (kv.split("=", 1) for kv in args.store)}
    d = derive(stores)
    fluid_rows_ = fluid_of(d["placed"])
    meta = A.read_json(Path(args.fluid_sessions).with_suffix(".meta.json"))
    fluid_pack, merged = Path(args.fluid_pack).resolve(), Path(args.merged).resolve()
    ident = (pca_original_identity(study, Path(args.relay_sa) if args.relay_sa else None,
                                   Path(args.e1_regen) if args.e1_regen else None)
             + d4_identity(study) + training_order_replay(study)
             + [check("fluid builder input", A.sha256_file(Path(args.fluid_sessions)), meta["builder_input_sha256"]),
                check("fluid pack built from that input", A.read_json(fluid_pack / "ladder.json")["sessions_sha256"],
                      meta["builder_input_sha256"]),
                check("fluid sessions are the primary cells of the @v3 pool", meta["source_sessions_sha256"],
                      d["sessions_sha256"])]
             + merged_identity(merged))
    cap = capacity(d)
    gap = max(0, RUNG - cap["supervised_tokens"])
    gen = topup(cap, d, gap=gap, price_a100=args.a100_price)
    gen_reserve = topup(cap, d, gap=gap + args.reserve, price_a100=args.a100_price)
    e1_sessions = {s["id"]: s for s in A.jsonl(E1_SESSIONS)}
    n_e1, _, e1_audit = e1_split()
    e1 = seq_metrics(e1_audit[:n_e1], e1_sessions, tok, unit_of=lambda s: s.get("source_id") or s["id"])
    f_sessions = {s["id"]: s for s in A.jsonl(Path(args.fluid_sessions))}
    f_audit = list(A.jsonl(fluid_pack / "audit.jsonl"))
    n_f, complete = fluid_cut(fluid_pack)
    fl = seq_metrics(f_audit[:n_f], f_sessions, tok, unit_of=lambda s: s["unit_id"])
    fl["axis_mix"] = by_axis(fl["token_mix"])
    fl["declared_mixture"] = A.read_json(fluid_pack / "ladder.json")["declared_mixture"]
    fl["mix_minus_declared_pp_max"] = max(abs(round(100 * (fl["token_mix"].get(k, 0) - v), 2))
                                          for k, v in fl["declared_mixture"].items())
    fl["families"] = len({f_sessions[m["session_id"]].get("family") for r in f_audit[:n_f] for m in r["sessions"]}
                         - {None})
    fl["reaches_2p96M"] = complete
    forecast_n = math.ceil(fl["packed_sequences"] * RUNG / fl["supervised_tokens"])
    forecast = {"packed_sequences": forecast_n, "supervised_tokens": RUNG,
                "non_padding_tokens": round(fl["non_padding_tokens"] * RUNG / fl["supervised_tokens"]),
                "_": "today's measured packing density scaled to 2.96M (the top-up's sessions sized as the pool's); "
                     "a FORECAST, replaced by the built pack"}
    doc = {"schema": "aadistill.stage2.reasoning_data.h1pa_augmented_audit/v1",
           "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "study": study["study_id"], "study_config_sha256": A.sha256_file(STUDY),
           "identity": ident, "identity_ok": all(c["ok"] for c in ident),
           "fluid_capacity": cap, "fluid_overlap_with_e1": e1_overlap(fluid_rows_),
           "gap_supervised_tokens": gap,
           "top_up": gen, "top_up_with_a_validation_reserve": {"reserve_supervised_tokens": args.reserve, **{
               k: gen_reserve[k] for k in ("gap_supervised_tokens", "tasks_total", "expected_supervised_tokens",
                                           "all_cells_have_enough_unused_tasks", "cost")}},
           "datasets": {"e1_2p96M": {**e1, "isolation": e1_isolation(E1_PACK)},
                        "fluid_increment": fl, "fluid_increment_forecast_at_2p96M": forecast,
                        "merged": {"pack": str(merged.relative_to(A.REPO)),
                                   "status": A.read_json(merged / "ladder.json")["status"]}},
           "authorizes": "nothing"}
    doc["schedule"] = schedules(e1, fl, forecast, study)
    doc["budget_L40S"] = budget(doc["schedule"], gen["cost"], price=args.price, volume_gb=args.volume_gb)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(OUT.relative_to(A.REPO), "identity_ok", doc["identity_ok"], "gap", gap)
    for c in ident:
        if not c["ok"]:
            print("  FAILED:", c["check"])
    return 0 if doc["identity_ok"] else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sessions")
    s.add_argument("--store", action="append", required=True, metavar="RUN_ID=DIR")
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--fluid-pack", required=True)
    m.add_argument("--out", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--store", action="append", required=True, metavar="RUN_ID=DIR")
    a.add_argument("--fluid-sessions", required=True)
    a.add_argument("--fluid-pack", required=True)
    a.add_argument("--merged", required=True)
    a.add_argument("--relay-sa", default=None, help="a directory holding E1 sa's relay run manifest and pod hashes")
    a.add_argument("--e1-regen", default=None, help="the directory E1's packer rebuilt E1's pack into")
    a.add_argument("--price", type=float, required=True, help="the L40S secure price per hour, read live")
    a.add_argument("--a100-price", type=float, required=True,
                   help="the A100-SXM4-80GB secure price per hour, read live (the top-up's measured hardware)")
    a.add_argument("--reserve", type=int, default=150_000,
                   help="an independent fluid validation reserve, priced as an option")
    a.add_argument("--volume-gb", type=int, default=50)
    g = sub.add_parser("pregen")
    g.add_argument("--store", action="append", required=True, metavar="RUN_ID=DIR")
    g.add_argument("--fluid-sessions", required=True)
    g.add_argument("--fluid-pack", required=True)
    g.add_argument("--a100-price", type=float, required=True)
    g.add_argument("--reserve", type=int, default=150_000)
    args = ap.parse_args(argv)
    return {"sessions": cmd_sessions, "merge": cmd_merge, "audit": cmd_audit, "pregen": cmd_pregen}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
