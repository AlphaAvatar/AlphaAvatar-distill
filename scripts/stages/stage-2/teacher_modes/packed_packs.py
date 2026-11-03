#!/usr/bin/env python3
"""A1-6cat, A2-6cat and C1-6cat training packs under the sequential causal packing. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_packs \\
        --arm A1-6cat --tool-documents <tool_admission out>/tool_documents.jsonl \\
        [--v1-records ... --v1-identity ... --records ... --identity ...] --out artifacts/stages/stage-2/teacher_modes/packs_6cat

Each arm's training set is its selected documents, token for token, plus its own teacher's admitted tool-calling
documents (`tool_admission.py`), packed by `token_documents.pack_sequential` at the design's sequence length and
seed (`packed_round_design.json :: stage3_packing`), validation likewise -- EVERY validation document, no sequence
cap (a cap after the shuffle kept a random part of each category) -- written in the trainer's ladder contract
(`write_pack`, no position ids: the trainer reads it with standard causal attention).

Data volume and proportions are counted BEFORE packing (maintainer 2026-11-03): unique supervised tokens per
document, once. The tool-calling documents enter as whole episodes in the pool's frozen training order up to the
arm's registered target (`tool_documents`, the shortest prefix that reaches it), and `shares` checks the six
base-corpus types at ~1/6 each of the supporting total and fluid : supporting at ~50:50 within the design's
tolerance (``stage3_packing.share_check``), refusing a pack outside it. Packing's repeated, truncated and padding
tokens and the supervised-token exposure are recorded beside them as packing / compute measures only, never as the
data's size or its proportions, and no loss weight corrects a quota. The record also lists the training data types
the validation set does not cover (`validation_coverage`).

* A1-6cat / A2-6cat: the mode's documents of the tasks `six_category_selection.json` selected. They are read from
  A1's / A2's packs (`packing_audit.documents_from_isolated_pack`, token for token); a selected task whose
  documents those packs do not hold is read from the verified admission records (`token_admission.load`), and
  without them the build refuses, naming the task. Validation: the pack's own validation documents plus the tool
  validation split's documents.
* C1-6cat: D4A_merged_whole's tool-free sessions (`packing_audit.documents_from_d4a`, system prompt + session
  tokens, the 352 historical cut traces kept as cut) plus Thinking-2507's tool documents; validation: the D4A
  validation tail's tool-free sessions plus the tool validation documents.

The tool documents must be the arm's own teacher's and mode's (checked by their document ids' mode suffix): no arm
reads another teacher's assistant tokens. Writes one directory per arm and returns its ladder record (file sha256,
the A/B layers of the packing audit). Nothing is trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import packing_audit as PA  # noqa: E402

DESIGN = REPO / "configs/stages/stage-2/teacher_modes/packed_round_design.json"
SELECTION = REPO / "logs/stages/stage-2/teacher_modes/data/six_category_selection.json"
OUT = REPO / "artifacts/stages/stage-2/teacher_modes/packs_6cat"
ARMS = {"A1-6cat": {"mode": "H-T", "source_pack": "A1", "teacher": "hybrid"},
        "A2-6cat": {"mode": "H-N", "source_pack": "A2", "teacher": "hybrid"},
        "C1-6cat": {"mode": "T2507", "source_pack": None, "teacher": "thinking_2507"}}
TOOL_TYPE = "supporting.tool_calling"
PAD_ID = 151643
SIX = REPO / "configs/stages/stage-2/teacher_modes/six_category_design.json"
TOOL_PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/tool_generation_protocol.json"
#: a document's data type -> the base-corpus type it counts toward (six_category_design.json :: base_category; C1's
#: D4A types by their sources). A1's / A2's packs carry RAG and multi-hop reading as one data type,
#: ``supporting.reading_verified``: its split between the two comes from the selection record and must sum to it.
SIX_TYPES = {"supporting.math_new": "competition_math", "openmath": "competition_math",
             "supporting.math_gsm8k": "gsm8k", "gsm8k": "gsm8k",
             "supporting.code_algorithmic": "code", "code": "code",
             "rag_evidence": "rag_reading", "multihop_qa": "multihop_reading",
             "supporting.reading_verified": None, TOOL_TYPE: "tool_calling"}
SELECTION_ARM = {"A1-6cat": "A1", "A2-6cat": "A2"}


class PackBuildError(RuntimeError):
    pass


def tool_target(arm: str) -> int:
    """The arm's registered tool-calling supervised-token target (six_category_design.json)."""
    return int(round(json.loads(SIX.read_text())["tool_calling"]["expected_per_arm"][arm]["target"]))


def tool_documents(path: Path | None, mode: str, *, target: int | None = None, order: list[str] | None = None):
    """``(train, validation, quota record)``: one lineage's admitted tool episodes as TokenDocuments. With a target,
    the training episodes are taken WHOLE in the pool's frozen training ``order`` until their unique supervised
    tokens reach it (the shortest prefix that does); without one, all of them."""
    from aadistill.data.token_documents import TokenDocument
    if path is None:
        return [], [], None
    by_task, val = {}, []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if f"|{mode}|" not in r["doc_id"]:
            raise PackBuildError(f"{r['doc_id']}: not a {mode} document; an arm reads only its own teacher's tokens")
        d = TokenDocument(r["doc_id"], tuple(r["prompt_ids"]), tuple(r["output_ids"]), {"category": r["category"]})
        if r["split"] == "training":
            by_task.setdefault(r["task_id"], []).append(d)
        elif r["split"] == "validation":
            val.append(d)
    if target is None:
        return [d for ds in by_task.values() for d in ds], val, None
    stray = sorted(set(by_task) - set(order or ()))
    if stray:
        raise PackBuildError(f"{len(stray)} tool training episode(s) not in the pool's training order, e.g. {stray[:3]}")
    train, got, used = [], 0, 0
    for tid in order:
        if got >= target:
            break
        if tid in by_task:
            train += by_task[tid]
            got += sum(len(d.output_ids) for d in by_task[tid])
            used += 1
    if got < target:
        raise PackBuildError(f"{mode}: the admitted tool episodes hold {got} supervised tokens, short of the "
                             f"target {target}; the type is never filled from another")
    return train, val, {"target": target, "selected_supervised_tokens": got, "episodes": used,
                        "admitted_episodes": len(by_task), "rule": "whole episodes in the pool's frozen training "
                        "order, the shortest prefix reaching the target"}


def shares(arm: str, train, dtype: dict, check: dict) -> dict:
    """The six base-corpus types and fluid : supporting on unique supervised tokens BEFORE packing; raises
    PackBuildError outside ``check`` (``per_type_relative``: each type's distance from 1/6 of the supporting total;
    ``fluid_fraction``: [lo, hi] of fluid / (fluid + supporting))."""
    by_dtype = Counter()
    for x in train:
        by_dtype[dtype[x.doc_id]] += int(sum(x.target_mask))
    unknown = sorted(k for k in by_dtype if not k.startswith("primary.") and k not in SIX_TYPES)
    if unknown:
        raise PackBuildError(f"{arm}: data types outside the six base-corpus types: {unknown}")
    fluid = sum(v for k, v in by_dtype.items() if k.startswith("primary."))
    six = Counter()
    for k, v in by_dtype.items():
        if not k.startswith("primary.") and SIX_TYPES[k]:
            six[SIX_TYPES[k]] += v
    if by_dtype.get("supporting.reading_verified"):
        sel = json.loads(SELECTION.read_text())["per_type_supervised_tokens"][SELECTION_ARM[arm]]
        rag, multi = int(sel["rag_reading"]), int(sel["multihop_reading"])
        if rag + multi != by_dtype["supporting.reading_verified"]:
            raise PackBuildError(f"{arm}: reading_verified holds {by_dtype['supporting.reading_verified']} supervised "
                                 f"tokens, not the selection's RAG {rag} + multi-hop {multi}")
        six.update(rag_reading=rag, multihop_reading=multi)
    supporting = sum(six.values())
    per_type = supporting / 6
    bad = []
    if sorted(six) != sorted(set(v for v in SIX_TYPES.values() if v) | {"rag_reading", "multihop_reading"}):
        bad.append(f"types present {sorted(six)}, not all six")
    dev = {k: round(v / per_type - 1, 5) for k, v in sorted(six.items())}
    bad += [f"{k} is {d:+.2%} from 1/6" for k, d in dev.items() if abs(d) > check["per_type_relative"]]
    frac = fluid / (fluid + supporting) if fluid + supporting else 0.0
    lo, hi = check["fluid_fraction"]
    if not lo <= frac <= hi:
        bad.append(f"fluid is {frac:.4f} of the total, outside [{lo}, {hi}]")
    if bad:
        raise PackBuildError(f"{arm}: proportions outside the registered check: " + "; ".join(bad))
    return {"counted": "unique supervised tokens before packing", "fluid": fluid, "supporting": supporting,
            "fluid_fraction": round(frac, 6), "six_types": dict(sorted(six.items())),
            "six_type_deviation_from_one_sixth": dev, "by_data_type": dict(sorted(by_dtype.items())),
            "check": check}


def validation_coverage(train, val, dtype: dict) -> dict:
    t = sorted({dtype[x.doc_id] for x in train})
    v = sorted({dtype[x.doc_id] for x in val})
    return {"training_types": t, "validation_types": v, "training_types_without_validation":
            sorted(set(t) - set(v))}


def a_arm_documents(arm: str, *, records: dict | None = None):
    """``(train docs, val docs, data_type, missing tasks)`` for A1-6cat or A2-6cat, before tool calling."""
    spec = ARMS[arm]
    mode = spec["mode"]
    sel = json.loads(SELECTION.read_text())
    selected = {t for cat in sel["selection"][mode].values() for t in cat}
    pack = PA.PACKS / spec["source_pack"]
    train_pairs = PA.documents_from_isolated_pack(pack, "train")
    val_pairs = PA.documents_from_isolated_pack(pack, "val")
    task_of = lambda d: d.doc_id.split("|", 1)[0]  # noqa: E731
    have = {task_of(d) for d, _ in train_pairs}
    train = [(d, t) for d, t in train_pairs if task_of(d) in selected]
    missing = sorted(selected - have)
    if missing:
        if records is None:
            raise PackBuildError(f"{arm}: {len(missing)} selected task(s) have no document in {pack} (e.g. "
                                 f"{missing[:3]}); pass the admission records (--v1-records/--records)")
        from stages.teacher_modes import token_admission as TA
        got = TA.load(Path(records["v1_records"]), records["v1_identity"], Path(records["records"]),
                      records["identity"])
        for tid in missing:
            v = got["verdicts"].get((tid, mode))
            if not v or not v.get("admitted"):
                raise PackBuildError(f"{arm}: selected task {tid} is not admitted in the records")
            stratum = TA.cell_of(got["tasks"][tid]).split("/", 1)[0]
            train += [(d, stratum) for d in v["docs"]]
    dtype = {d.doc_id: t for d, t in train + val_pairs}
    return [d for d, _ in train], [d for d, _ in val_pairs], dtype, missing


def c1_documents():
    """``(train docs, val docs, data_type)`` for C1-6cat, before tool calling."""
    rows = PA.documents_from_d4a()
    val = []
    for line in (PA.D4A / "audit.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["block"] >= PA.D4A_TRAIN_SEQUENCES:
            val.append(row)
    import numpy as np
    from aadistill.data.token_documents import MaskedDocument
    arr = np.load(PA.D4A / "blocks.npz")
    ids, ce = arr["input_ids"], arr["ce_mask"]
    val_docs, dtype = [], {d.doc_id: t for d, t, _ in rows}
    for row in val:
        b, ns = row["block"], int(row["n_system_tokens"])
        system = [int(x) for x in ids[b, :ns]]
        for s in row["sessions"]:
            if s["data_type"] == "tool_calling":
                continue
            body = [int(x) for x in ids[b, s["start"]:s["end"]]]
            m = [bool(x) for x in ce[b, s["start"]:s["end"]]]
            d = MaskedDocument(f"{b}:{s['session_id']}", tuple(system + body), tuple([False] * ns + m))
            val_docs.append(d)
            dtype[d.doc_id] = s["data_type"]
    return [d for d, _, _ in rows], val_docs, dtype


def build(arm: str, *, tool_docs: Path | None, out: Path = OUT, records: dict | None = None,
          require_tools: bool = True, pool_dir: Path | None = None) -> dict:
    from aadistill.data.token_documents import pack_sequential, write_pack
    spec = ARMS[arm]
    d = json.loads(DESIGN.read_text())["stage3_packing"]
    seq, seed = int(d["sequence_length"]), int(d["shuffle_seed"])
    if spec["source_pack"]:
        train, val, dtype, missing = a_arm_documents(arm, records=records)
    else:
        train, val, dtype = c1_documents()
        missing = []
    order = None
    if tool_docs is not None:
        pool_dir = Path(pool_dir) if pool_dir else REPO / json.loads(TOOL_PROTOCOL.read_text())["pool"]["dir"]
        order = json.loads((pool_dir / "pool.json").read_text())["training_order"]
    t_train, t_val, quota = tool_documents(tool_docs, spec["mode"], target=tool_target(arm) if tool_docs else None,
                                           order=order)
    if require_tools and not t_train:
        raise PackBuildError(f"{arm}: no tool-calling training documents; the sixth type is required")
    for x in t_train + t_val:
        dtype[x.doc_id] = TOOL_TYPE
    share = shares(arm, train + t_train, dtype, d["share_check"]) if require_tools else None
    tr = pack_sequential(train + t_train, seq, PAD_ID, order_seed=seed)
    v = pack_sequential(val + t_val, seq, PAD_ID, order_seed=seed)
    by_type = Counter()
    for x in train + t_train:
        by_type[dtype[x.doc_id]] += int(sum(x.target_mask))
    rec = write_pack(tr, v, Path(out) / arm, data_type=dtype,
                     extra={"arm": arm, "mode": spec["mode"], "teacher": spec["teacher"],
                            "context_strategy": "cross_context", "missing_tasks_read_from_records": missing,
                            "unique_supervised_tokens_by_type": dict(sorted(by_type.items())),
                            "proportions_before_packing": share,
                            "validation_coverage": validation_coverage(train + t_train, val + t_val, dtype),
                            "tool_documents": {"path": str(tool_docs) if tool_docs else None, "quota": quota,
                                               "train": len(t_train), "validation": len(t_val)}})
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--tool-documents", default=None)
    ap.add_argument("--tool-pool", default=None, help="the pool directory (default: tool_generation_protocol.json)")
    ap.add_argument("--out", default=str(OUT))
    for k in ("v1-records", "v1-identity", "records", "identity"):
        ap.add_argument(f"--{k}", default=None)
    a = ap.parse_args(argv)
    records = ({"v1_records": a.v1_records, "v1_identity": a.v1_identity, "records": a.records,
                "identity": a.identity} if a.v1_records else None)
    rec = build(a.arm, tool_docs=Path(a.tool_documents) if a.tool_documents else None, out=Path(a.out),
                records=records, pool_dir=Path(a.tool_pool) if a.tool_pool else None)
    print(json.dumps({k: rec[k] for k in ("packing", "n_blocks", "n_val_blocks", "files_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
