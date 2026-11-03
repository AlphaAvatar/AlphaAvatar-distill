#!/usr/bin/env python3
"""A1-6cat, A2-6cat and C1-6cat training packs under the sequential causal packing. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_packs \\
        --arm A1-6cat --tool-documents <tool_admission out>/tool_documents.jsonl \\
        [--v1-records ... --v1-identity ... --records ... --identity ...] --out artifacts/stages/stage-2/teacher_modes/packs_6cat

Each arm's training set is its selected documents, token for token, plus its own teacher's admitted tool-calling
documents (`tool_admission.py`), packed by `token_documents.pack_sequential` at the design's sequence length and
seed (`packed_round_design.json :: stage3_packing`), validation likewise into at most 16 sequences, written in the
trainer's ladder contract (`write_pack`, no position ids: the trainer reads it with standard causal attention).

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
VAL_SEQUENCES = 16
PAD_ID = 151643


class PackBuildError(RuntimeError):
    pass


def tool_documents(path: Path | None, mode: str):
    """``(train, validation)`` TokenDocuments of one lineage's admitted tool episodes."""
    from aadistill.data.token_documents import TokenDocument
    if path is None:
        return [], []
    train, val = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if f"|{mode}|" not in r["doc_id"]:
            raise PackBuildError(f"{r['doc_id']}: not a {mode} document; an arm reads only its own teacher's tokens")
        d = TokenDocument(r["doc_id"], tuple(r["prompt_ids"]), tuple(r["output_ids"]), {"category": r["category"]})
        (train if r["split"] == "training" else val if r["split"] == "validation" else []).append(d)
    return train, val


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
          require_tools: bool = True) -> dict:
    from aadistill.data.token_documents import PackedDocuments, pack_sequential, write_pack
    spec = ARMS[arm]
    d = json.loads(DESIGN.read_text())["stage3_packing"]
    seq, seed = int(d["sequence_length"]), int(d["shuffle_seed"])
    if spec["source_pack"]:
        train, val, dtype, missing = a_arm_documents(arm, records=records)
    else:
        train, val, dtype = c1_documents()
        missing = []
    t_train, t_val = tool_documents(tool_docs, spec["mode"])
    if require_tools and not t_train:
        raise PackBuildError(f"{arm}: no tool-calling training documents; the sixth type is required")
    for x in t_train + t_val:
        dtype[x.doc_id] = TOOL_TYPE
    tr = pack_sequential(train + t_train, seq, PAD_ID, order_seed=seed)
    v = pack_sequential(val + t_val, seq, PAD_ID, order_seed=seed)
    n = min(VAL_SEQUENCES, v.input_ids.shape[0])
    #: the first `n` validation sequences; their spec, not the full plan's counts, travels with them
    v = PackedDocuments(v.input_ids[:n], v.target_mask[:n], v.content_mask[:n], v.position_ids[:n],
                        v.members[:n], [], v.strategy, 0,
                        {k: v.packing[k] for k in ("packing", "truncate", "materialization", "shuffle",
                                                   "attention", "sequence_length", "shuffle_seed")})
    by_type = Counter()
    for x in train + t_train:
        by_type[dtype[x.doc_id]] += int(sum(x.target_mask))
    rec = write_pack(tr, v, Path(out) / arm, data_type=dtype,
                     extra={"arm": arm, "mode": spec["mode"], "teacher": spec["teacher"],
                            "context_strategy": "cross_context", "missing_tasks_read_from_records": missing,
                            "unique_supervised_tokens_by_type": dict(sorted(by_type.items())),
                            "tool_documents": {"path": str(tool_docs) if tool_docs else None,
                                               "train": len(t_train), "validation": len(t_val)}})
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--tool-documents", default=None)
    ap.add_argument("--out", default=str(OUT))
    for k in ("v1-records", "v1-identity", "records", "identity"):
        ap.add_argument(f"--{k}", default=None)
    a = ap.parse_args(argv)
    records = ({"v1_records": a.v1_records, "v1_identity": a.v1_identity, "records": a.records,
                "identity": a.identity} if a.v1_records else None)
    rec = build(a.arm, tool_docs=Path(a.tool_documents) if a.tool_documents else None, out=Path(a.out),
                records=records)
    print(json.dumps({k: rec[k] for k in ("packing", "n_blocks", "n_val_blocks", "files_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
