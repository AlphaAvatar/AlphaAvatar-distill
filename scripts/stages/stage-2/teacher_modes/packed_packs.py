#!/usr/bin/env python3
"""A1-6cat, A2-6cat and C1-6cat training data: selection, proportion validation, admission and packing. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_packs \\
        --arm A1-6cat --tool-documents <tool_admission out>/tool_documents.jsonl \\
        [--v1-records ... --v1-identity ... --records ... --identity ...] [--validate-only] \\
        --out artifacts/stages/stage-2/teacher_modes/packs_6cat

Four separate steps (maintainer 2026-11-04); `build` is only the workflow that runs them in order and stops at the
first refusal:

1. **Data selection** (`select`): the arm's documents token for token, before packing -- its selected tasks plus its
   own teacher's admitted tool-calling episodes (`tool_admission.py`), whole, per category in the pool's frozen
   order up to each cell's target in the frozen generation plan (`tool_documents`, `tool_generation_plan.select`:
   the rule the session stopped generating by). A short cell is recorded as SHORTFALL, never filled from another.
   Nothing is packed and nothing is judged.
2. **Proportion validation** (`proportions`): unique supervised tokens before packing per six base-corpus type and
   fluid, against the arm's registered quotas and tolerances (`registered_quotas`: token_matched_design.json ::
   scale and its whole-task selection rule, as six_category_design.json :: base_category keeps it), through the
   generic `aadistill.data.mixture.validate_proportions`. It reads no packed sequence and no loss weight. The
   report carries the content identity of the selection it validated (`selection_identity`) and its own seal.
3. **Admission / freeze** (`admit`): admits only the selection its proportion report validated (the identity is
   recomputed and compared, never the proportions: `report_problems`), whose proportion check passed and whose tool cells all
   reached their targets, a failure otherwise covered only by the maintainer's recorded approval
   (packed_round_design.json :: data_admission.approved_exceptions; the plan's approved_shortfalls for a tool
   cell). It writes the frozen data manifest -- every document's id, split, data type, category, whole unit,
   supervised tokens and content sha256, the counts and proportions, the sources, the lineage -- or refuses
   (`AdmissionError`) and writes nothing. The documents' token content is frozen beside it (`write_documents`).
4. **Packing** (`pack`): random shuffle -> sequential packing -> truncation (`token_documents.pack_sequential`) at
   the design's sequence length and seed (`stage3_packing`), every validation document included (no sequence cap),
   written in the trainer's ladder contract (`write_pack`, no position ids: standard causal attention). It checks
   that its input is exactly the admitted manifest's documents and that the manifest is intact; it computes no
   proportion. The ladder names the manifest, which keeps the data's size and proportions; packing's repeated,
   truncated and padding tokens and the supervised-token exposure are packing / compute measures beside them.

`verify_pack` is the formal consumer's check (`packed_training_plan.bind`): an admitted, unchanged manifest, the
manifest's documents and the design's packing spec, and the actual packed tokens and supervision mask -- the content
identity the packing recorded (`packed_content_sha256`), and the sequences re-derived exactly from the documents the
admission froze beside the manifest (`data_documents.npz`, by sha256), overlong documents truncated as the formal
packing truncates them -- or the pack is refused.

* A1-6cat / A2-6cat: the mode's documents of the tasks `six_category_selection.json` selected. They are read from
  A1's / A2's packs (`packing_audit.documents_from_isolated_pack`, token for token); a selected task whose
  documents those packs do not hold is read from the verified admission records (`token_admission.load`), and
  without them the selection refuses, naming the task. Validation: the pack's own validation documents plus the
  tool validation split's documents.
* C1-6cat: D4A_merged_whole's tool-free sessions (`packing_audit.documents_from_d4a`, system prompt + session
  tokens, the 352 historical cut traces kept as cut) plus Thinking-2507's tool documents; validation: the D4A
  validation tail's tool-free sessions plus the tool validation documents.

The tool documents must be the arm's own teacher's and mode's (checked by their document ids' mode suffix): no arm
reads another teacher's assistant tokens. Nothing is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import packing_audit as PA  # noqa: E402

DESIGN = REPO / "configs/stages/stage-2/teacher_modes/packed_round_design.json"
TOKEN_MATCHED = REPO / "configs/stages/stage-2/teacher_modes/token_matched_design.json"
SIX = REPO / "configs/stages/stage-2/teacher_modes/six_category_design.json"
SELECTION = REPO / "logs/stages/stage-2/teacher_modes/data/six_category_selection.json"
OUT = REPO / "artifacts/stages/stage-2/teacher_modes/packs_6cat"
#: ``base``: how the arm's five existing base-corpus types and its fluid part were obtained -- re-selected under the
#: whole-task rule (A1/A2) or kept byte for byte from D4A_merged_whole (C1); `registered_quotas` reads them so.
#: ``composition``: the tool type's internal composition the arm's identity registers -- six_category_design.json's
#: (A1/A2/C1-6cat) or six_category_common_support@v1's (the -cs arms, maintainer 2026-11-05); the generation plan's
#: training cells must carry the same one, or the selection refuses (`select`)
ARMS = {"A1-6cat": {"mode": "H-T", "source_pack": "A1", "teacher": "hybrid", "base": "selected", "composition": "registered"},
        "A2-6cat": {"mode": "H-N", "source_pack": "A2", "teacher": "hybrid", "base": "selected", "composition": "registered"},
        "C1-6cat": {"mode": "T2507", "source_pack": None, "teacher": "thinking_2507", "base": "kept",
                    "composition": "registered"},
        "A1-6cat-cs": {"mode": "H-T", "source_pack": "A1", "teacher": "hybrid", "base": "selected",
                       "composition": "common_support"},
        "A2-6cat-cs": {"mode": "H-N", "source_pack": "A2", "teacher": "hybrid", "base": "selected",
                       "composition": "common_support"},
        "C1-6cat-cs": {"mode": "T2507", "source_pack": None, "teacher": "thinking_2507", "base": "kept",
                       "composition": "common_support"}}
COMMON_SUPPORT = REPO / "configs/stages/stage-2/teacher_modes/six_category_common_support.json"
TOOL_TYPE = "supporting.tool_calling"
PAD_ID = 151643
THINK_CLOSE = 151668
#: a document's data type -> the base-corpus type it counts toward (six_category_design.json :: base_category; C1's
#: D4A types by their sources). A1's / A2's packs carry RAG and multi-hop reading as one data type,
#: ``supporting.reading_verified``: its split between the two comes from the selection record and must sum to it.
SIX_TYPES = {"supporting.math_new": "competition_math", "openmath": "competition_math",
             "supporting.math_gsm8k": "gsm8k", "gsm8k": "gsm8k",
             "supporting.code_algorithmic": "code", "code": "code",
             "rag_evidence": "rag_reading", "multihop_qa": "multihop_reading",
             "supporting.reading_verified": None, TOOL_TYPE: "tool_calling"}
SIX_NAMES = ("competition_math", "gsm8k", "rag_reading", "multihop_reading", "code", "tool_calling")
SELECTION_ARM = {"A1-6cat": "A1", "A2-6cat": "A2", "A1-6cat-cs": "A1", "A2-6cat-cs": "A2"}
MANIFEST = "data_manifest.json"
MANIFEST_SCHEMA = "aadistill.stage2.teacher_modes.data_manifest/v2"
#: the frozen documents themselves, beside the manifest that holds their sha256 (`write_documents`)
DOCUMENTS = "data_documents.npz"
COLUMNS = ["doc_id", "data_type", "category", "unit", "supervised_tokens", "sha256"]


class PackBuildError(RuntimeError):
    pass


class AdmissionError(PackBuildError):
    """A selection the data admission refuses; ``verdict`` names every unapproved failure."""

    def __init__(self, message: str, verdict: dict):
        super().__init__(message)
        self.verdict = verdict


def sha_file(path: Path) -> str:
    from aadistill.infrastructure.manifest import sha256_file
    return sha256_file(Path(path))


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def doc_sha256(d) -> str:
    """A document's token content (ids and supervision), for TokenDocument and MaskedDocument alike."""
    return canonical_sha([list(d.ids), [int(x) for x in d.target_mask]])


def tool_documents(path: Path | None, mode: str, *, cells: dict | None = None, orders: dict | None = None):
    """``(train, validation, quota record)``: one lineage's admitted tool episodes as TokenDocuments. With the plan's
    ``cells`` (``{"training": [...], "validation": [...]}``) and the pool's frozen ``orders``, whole episodes are
    selected per cell by `tool_generation_plan.select`; without them, every admitted episode."""
    from aadistill.data.token_documents import TokenDocument
    from stages.teacher_modes import tool_generation_plan as TGP
    if path is None:
        return [], [], None
    docs = {"training": {}, "validation": {}}
    cat_of, units = {}, {"training": {}, "validation": {}}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if f"|{mode}|" not in r["doc_id"]:
            raise PackBuildError(f"{r['doc_id']}: not a {mode} document; an arm reads only its own teacher's tokens")
        if r["split"] not in docs:
            continue
        d = TokenDocument(r["doc_id"], tuple(r["prompt_ids"]), tuple(r["output_ids"]),
                          {"category": r["category"], "task_id": r["task_id"]})
        docs[r["split"]].setdefault(r["task_id"], []).append(d)
        cat_of[r["task_id"]] = r["category"]
        units[r["split"]][r["task_id"]] = units[r["split"]].get(r["task_id"], 0) + len(d.output_ids)
    if cells is None:
        return ([d for ds in docs["training"].values() for d in ds],
                [d for ds in docs["validation"].values() for d in ds], None)
    out, quota = {"training": [], "validation": []}, {}
    for split in ("training", "validation"):
        stray = sorted(set(docs[split]) - set(orders.get(split, [])))
        if stray:
            raise PackBuildError(f"{len(stray)} {split} tool episode(s) not in the pool's frozen order, e.g. {stray[:3]}")
        for c in cells[split]:
            order = [t for t in orders[split] if c["category"] is None or cat_of.get(t) == c["category"]]
            u = units[split] if c["unit"] == "supervised_tokens" else {t: 1.0 for t in units[split]}
            ids, got = TGP.select(order, u, c["target"], c.get("min_tasks", 0))
            out[split] += [d for t in ids for d in docs[split][t]]
            quota[f"{split}/{c['category']}"] = {"split": split, "category": c["category"], "unit": c["unit"],
                                                 "target": c["target"], "selected": got, "episodes": len(ids),
                                                 "status": "REACHED" if got >= c["target"] else "SHORTFALL"}
    return out["training"], out["validation"], {"cells": quota, "rule": "whole episodes per cell in the pool's "
                                                "frozen order, until the plan's target (tool_generation_plan.select)"}


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
    if missing and records is None:            #: each task replayed alone through the frozen records (`recover`)
        rows, docs, dt = recover(mode, missing)
        bad = [r["task_id"] for r in rows if not (r["admitted"] and not r["isolation_flagged"])]
        if bad:
            raise PackBuildError(f"{arm}: selected task(s) {bad} are not admitted in the frozen records")
        train += [(d, dt[d.doc_id]) for d in docs]
    elif missing:
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


# --- 1. data selection ---------------------------------------------------------------------------------------------

def _files(d: Path) -> dict:
    """A source pack's identity: the files its reader reads (a reader that ran found them)."""
    return {"dir": os.path.relpath(d, REPO), "files_sha256": {n: sha_file(d / n) for n in ("blocks.npz", "audit.jsonl")
                                                             if (d / n).is_file()}}


def select(arm: str, *, tool_docs: Path | None, records: dict | None = None, pool_dir: Path | None = None) -> dict:
    """The arm's documents before packing, with each one's data type and whole unit (an example or an episode with
    all its turns), the tool cells' quota record and the sources read. Packs nothing and judges nothing: a short
    tool cell is SHORTFALL in the quota record, for the admission to refuse."""
    from stages.teacher_modes import tool_generation_plan as TGP
    spec = ARMS[arm]
    plan = cells = orders = None
    if tool_docs is not None:                       #: the plan's identity first: a drifted plan refuses before any read
        plan, orders = TGP.for_consumers(pool_dir=pool_dir)
        has = "common_support" if plan.get("common_support") else "registered"
        if has != spec["composition"]:
            raise PackBuildError(f"{arm} registers the {spec['composition']} tool composition; the generation plan's "
                                 f"training cells carry the {has} one")
        cells = {s: TGP.cells_for(plan, spec["teacher"], spec["mode"], s) for s in ("training", "validation")}
    if spec["source_pack"]:
        train, val, dtype, missing = a_arm_documents(arm, records=records)
        unit_of = {d.doc_id: d.doc_id.split("|", 1)[0] for d in train + val}
        pack = PA.PACKS / spec["source_pack"]
        sources = {"selection_record_sha256": sha_file(SELECTION), "source_pack": _files(pack),
                   "admission_records": ({k: records[k] for k in ("v1_identity", "identity")} if missing else None)}
    else:
        train, val, dtype = c1_documents()
        missing = []
        unit_of = {d.doc_id: d.doc_id for d in train + val}
        sources = {"d4a": _files(PA.D4A)}
    t_train, t_val, quota = tool_documents(tool_docs, spec["mode"], cells=cells, orders=orders)
    for x in t_train + t_val:
        dtype[x.doc_id] = TOOL_TYPE
        unit_of[x.doc_id] = x.meta["task_id"]
    sources["tool_documents"] = ({"path": str(tool_docs), "sha256": sha_file(tool_docs),
                                  "pool": plan.get("pool") if plan else None,
                                  "plan_protocol_sha256": plan.get("protocol_sha256") if plan else None}
                                 if tool_docs is not None else None)
    return {"arm": arm, "mode": spec["mode"], "teacher": spec["teacher"], "train": train + t_train,
            "validation": val + t_val, "data_type": dtype, "unit_of": unit_of,
            "missing_tasks_read_from_records": missing, "sources": sources,
            "tool": {"quota": quota, "train": len(t_train), "validation": len(t_val),
                     "approved_shortfalls": list((plan or {}).get("approved_shortfalls") or [])}}


# --- 2. proportion validation --------------------------------------------------------------------------------------

def type_counts(arm: str, train, dtype: dict, unit_of: dict) -> tuple[dict, dict, dict]:
    """``(counts, by data type, largest whole unit)`` on unique supervised tokens BEFORE packing: per six
    base-corpus type and ``fluid`` (every ``primary.*`` type), and the largest whole unit each holds."""
    by_dtype, per_unit, unit_type = Counter(), Counter(), {}
    for x in train:
        t, n = dtype[x.doc_id], int(sum(x.target_mask))
        by_dtype[t] += n
        per_unit[unit_of[x.doc_id]] += n
        unit_type[unit_of[x.doc_id]] = t
    unknown = sorted(k for k in by_dtype if not k.startswith("primary.") and k not in SIX_TYPES)
    if unknown:
        raise PackBuildError(f"{arm}: data types outside the six base-corpus types: {unknown}")
    leaf = lambda t: "fluid" if t.startswith("primary.") else SIX_TYPES[t]  # noqa: E731
    counts = Counter()
    for t, v in by_dtype.items():
        if leaf(t):
            counts[leaf(t)] += v
    if by_dtype.get("supporting.reading_verified"):
        sel = json.loads(SELECTION.read_text())["per_type_supervised_tokens"][SELECTION_ARM[arm]]
        rag, multi = int(sel["rag_reading"]), int(sel["multihop_reading"])
        if rag + multi != by_dtype["supporting.reading_verified"]:
            raise PackBuildError(f"{arm}: reading_verified holds {by_dtype['supporting.reading_verified']} supervised "
                                 f"tokens, not the selection's RAG {rag} + multi-hop {multi}")
        counts.update(rag_reading=rag, multihop_reading=multi)
    largest = Counter()
    for u, n in per_unit.items():
        for k in ((leaf(unit_type[u]),) if leaf(unit_type[u]) else ("rag_reading", "multihop_reading")):
            largest[k] = max(largest[k], n)
    return dict(counts), dict(sorted(by_dtype.items())), dict(largest)


def registered_quotas(arm: str, largest: dict, tool_cells: int) -> tuple[dict, list, dict]:
    """``(quotas, shares, rule)`` for `aadistill.data.mixture.validate_proportions`, every number from its owner:
    token_matched_design.json :: scale (the per-mode total, the fluid and base-category targets, its ``tolerance``)
    and the selection rule six_category_design.json :: base_category keeps unchanged -- whole tasks; each category
    at least its target and the total at most ``total_over_target_max_fraction`` over it; fluid's share of the
    total within ``category_share_deviation_max_points`` of its target share; each cell within one task of its
    share, so a type within one largest whole unit per selection cell (the tool type's cells: the generation plan's
    training categories). C1-6cat keeps its five types and its fluid part byte for byte (exact) and adds the tool
    type at its registered target. Nothing here is a new tolerance."""
    scale = json.loads(TOKEN_MATCHED.read_text())["scale"]
    base = json.loads(SIX.read_text())["base_category"]
    tol, cats, total = scale["tolerance"], scale["categories"], float(scale["per_mode_total"])
    cells = {t: len(v["cells"]) for t, v in base["A1_6cat_and_A2_6cat"].items() if isinstance(v, dict)}
    cells["tool_calling"] = tool_cells
    width = lambda t: cells[t] * largest.get(t, 0)  # noqa: E731
    quotas = {}
    if ARMS[arm]["base"] == "kept":
        c1 = base["C1_6cat"]
        for k, v in c1["kept_byte_for_byte"].items():
            if k != "_":
                name = k.split(" ")[0]
                quotas[SIX_TYPES.get(name) or name] = {"target": float(v), "below": 0, "above": 0}
        quotas["fluid"] = {"target": float(cats["fluid"]), "below": 0, "above": 0}
        tool_target = float(c1["tool_calling_target"])
    else:
        for t in SIX_NAMES[:-1]:
            quotas[t] = {"target": float(base["per_type"]), "below": width(t), "above": width(t)}
        quotas["fluid"] = {"target": float(cats["fluid"]), "below": 0, "above": None}
        tool_target = float(base["per_type"])
    quotas["tool_calling"] = {"target": tool_target, "below": width("tool_calling"), "above": width("tool_calling")}
    quotas["supporting"] = {"of": list(SIX_NAMES), "target": float(cats["base_counterpart"]), "below": 0,
                            "above": None}
    quotas["total"] = {"of": [*SIX_NAMES, "fluid"], "target": total, "below": 0,
                       "above": float(tol["total_over_target_max_fraction"]) * total}
    shares = [{"name": "fluid_of_total", "part": ["fluid"], "whole": [*SIX_NAMES, "fluid"],
               "target": float(cats["fluid"]) / total,
               "max_deviation_points": float(tol["category_share_deviation_max_points"])}]
    rule = {"owners": {p.name: sha_file(p) for p in _designs(arm)},
            "tolerance": tol, "cells_per_type": cells, "base": ARMS[arm]["base"],
            "rule": "whole tasks; fluid and supporting each at least their targets, the per-mode total at most "
                    f"{tol['total_over_target_max_fraction']:.1%} over; fluid's share within "
                    f"{tol['category_share_deviation_max_points']} points; each type within one largest whole unit "
                    "per selection cell of its target" + ("; C1's five kept types and fluid exact"
                                                          if ARMS[arm]["base"] == "kept" else "")}
    return quotas, shares, rule


def selection_identity(arm: str, sel: dict) -> str:
    """The content identity a proportion report is bound to (`proportions`, checked by `admit`): per split every
    document's id, data type, category, whole unit, supervised tokens and token content (ids and supervision mask,
    `doc_sha256`); the tool cells with their targets and the approvals the plan carries; the sources read (the
    selection record, the source packs, the tool documents, the pool and the plan); and the frozen designs the
    quotas come from (token_matched_design.json, six_category_design.json). A hash, not a validation: computing it
    judges nothing."""
    return canonical_sha({"arm": arm, "mode": sel["mode"], "teacher": sel["teacher"],
                          "documents": {"training": _entries(sel["train"], sel),
                                        "validation": _entries(sel["validation"], sel)},
                          "tool": sel["tool"], "sources": sel["sources"],
                          "missing_tasks_read_from_records": sel["missing_tasks_read_from_records"],
                          "designs": {p.name: sha_file(p) for p in _designs(arm)}})


def _designs(arm: str) -> tuple[Path, ...]:
    return (TOKEN_MATCHED, SIX) + ((COMMON_SUPPORT,) if ARMS[arm]["composition"] == "common_support" else ())


def proportions(arm: str, sel: dict) -> dict:
    """Data proportion validation of a selection, before and without packing: its counts against the arm's
    registered quotas (`registered_quotas`) by `aadistill.data.mixture.validate_proportions`. Returns the report
    (every row's target, amount, deviation, bounds and status, the share rows, ``passed``), bound to the selection
    it validated (``selection_sha256``, `selection_identity`) and sealed (``report_sha256``); deciding admission
    is `admit`'s."""
    from aadistill.data.mixture import validate_proportions
    counts, by_dtype, largest = type_counts(arm, sel["train"], sel["data_type"], sel["unit_of"])
    quota = (sel["tool"]["quota"] or {}).get("cells") or {}
    quotas, shares, rule = registered_quotas(arm, largest, sum(c["split"] == "training" for c in quota.values()))
    rep = {"counted": "unique supervised tokens before packing", **validate_proportions(counts, quotas, shares),
           "counts": counts, "by_data_type": by_dtype, "largest_whole_unit": largest, "registered": rule,
           "arm": arm, "selection_sha256": selection_identity(arm, sel)}
    rep["report_sha256"] = canonical_sha(rep)
    return rep


def report_problems(arm: str, sel: dict, report: dict) -> list[str]:
    """Why ``report`` is not the proportion report of this arm's ``sel``, as it was written (empty when it is): an
    edited report, another arm's, or another selection's -- including this selection after a document's tokens,
    supervision mask, data type, category, split or unit, a tool cell, a source or a quota design changed."""
    body = {k: v for k, v in report.items() if k != "report_sha256"}
    if report.get("report_sha256") != canonical_sha(body):
        return ["the proportion report was edited after its validation (report_sha256)"]
    if report.get("arm") != arm:
        return [f"the proportion report is {report.get('arm')!r}'s, not {arm!r}'s"]
    if report.get("selection_sha256") != selection_identity(arm, sel):
        return ["the proportion report validated another selection: this one's documents (tokens, supervision "
                "masks, data types, categories, splits or units), tool cells, sources or quota designs differ "
                "(selection_sha256)"]
    return []


# --- 3. admission / freeze -----------------------------------------------------------------------------------------

def _entries(docs, sel: dict) -> list[list]:
    return [[d.doc_id, sel["data_type"][d.doc_id], (d.meta or {}).get("category"), sel["unit_of"][d.doc_id],
             int(sum(d.target_mask)), doc_sha256(d)] for d in docs]


def _composition(docs, sel: dict) -> dict:
    """Examples, documents and supervised tokens; supervised tokens split into reasoning (each supervised span up to
    and including its first ``</think>``), answer (the rest) and tool calling (the tool type's documents)."""
    out = {"examples": len({sel["unit_of"][d.doc_id] for d in docs}), "documents": len(docs),
           "unique_supervised_tokens": 0, "reasoning_tokens": 0, "answer_tokens": 0, "tool_calling_tokens": 0}
    for d in docs:
        if hasattr(d, "output_ids"):                        #: a TokenDocument: one trailing supervised span
            spans = [d.output_ids]
        else:
            ids, mask, spans, i = d.ids, d.target_mask, [], 0
            while i < len(mask):
                j = i
                while j < len(mask) and mask[j] == mask[i]:
                    j += 1
                if mask[i]:
                    spans.append(ids[i:j])
                i = j
        n = sum(len(s) for s in spans)
        out["unique_supervised_tokens"] += n
        if sel["data_type"][d.doc_id] == TOOL_TYPE:
            out["tool_calling_tokens"] += n
        for span in spans:
            r = span.index(THINK_CLOSE) + 1 if THINK_CLOSE in span else 0
            out["reasoning_tokens"] += r
            out["answer_tokens"] += len(span) - r
    return out


def admission_verdict(arm: str, sel: dict, report: dict) -> dict:
    """Every failure of the proportion report and every SHORTFALL tool cell, each either approved by the maintainer
    (``data_admission.approved_exceptions`` / the plan's ``approved_shortfalls``) or unapproved."""
    from stages.teacher_modes import tool_generation_plan as TGP
    exc = {e["quota"]: e for e in json.loads(DESIGN.read_text())["data_admission"]["approved_exceptions"]
           if e.get("arm") == arm}
    unapproved, used = [], []
    for k, row in report["quotas"].items():
        if row["status"] != "OK":
            (used.append(exc[k]) if k in exc else
             unapproved.append(f"{k}: {row['status']} ({row['amount']:.0f} against {row['target']:.0f}, "
                               f"bounds {row['bounds']})"))
    for name, row in report["shares"].items():
        if not row["ok"]:
            (used.append(exc[name]) if name in exc else
             unapproved.append(f"{name}: {row['deviation_points']:+.3f} points (max {row['max_deviation_points']})"))
    unapproved += [f"{k}: no registered quota" for k in report["unregistered"]]
    quota = sel["tool"]["quota"]
    if quota is None:
        unapproved.append("the tool-calling documents were not selected by the frozen generation plan")
    for key, c in ((quota or {}).get("cells") or {}).items():
        if c["status"] != "REACHED":
            a = TGP.approved({"approved_shortfalls": sel["tool"]["approved_shortfalls"]}, c["split"], sel["mode"],
                             c["category"])
            (used.append({**a, "cell": key}) if a else
             unapproved.append(f"tool cell {key}: SHORTFALL ({c['selected']:.0f} of {c['target']:.0f} {c['unit']})"))
    return {"admitted": not unapproved, "unapproved": unapproved, "approved_exceptions": used}


def admit(arm: str, sel: dict, report: dict, out: Path = OUT) -> dict:
    """Admit and freeze a selection, or refuse it (`AdmissionError`, nothing written). ``report`` must be this
    selection's own proportion report, unedited (`report_problems`: the selection's content identity is recomputed
    and compared; the proportions are not validated again here). The frozen manifest
    (``<out>/<arm>/data_manifest.json``) holds every document (``COLUMNS``), the counts before packing, the
    proportion report, the tool cells, the approvals used, the sources and the lineage, sealed by its
    ``content_sha256``. It authorizes nothing."""
    from stages.teacher_modes import tool_generation_plan as TGP
    bound = report_problems(arm, sel, report)
    if bound:
        raise AdmissionError(f"{arm}: data admission refused: " + "; ".join(bound),
                             {"admitted": False, "unapproved": bound, "approved_exceptions": []})
    verdict = admission_verdict(arm, sel, report)
    if not verdict["admitted"]:
        raise AdmissionError(f"{arm}: data admission refused: " + "; ".join(verdict["unapproved"]), verdict)
    li = json.loads(TGP.PROTOCOL.read_text())["lineages"][sel["teacher"]]
    docs = {"training": _entries(sel["train"], sel), "validation": _entries(sel["validation"], sel)}
    p = Path(out) / arm / MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    frozen = write_documents(sel, p.parent / DOCUMENTS)
    man = {"schema": MANIFEST_SCHEMA, "arm": arm, "mode": sel["mode"],
           "teacher": {"lineage": sel["teacher"], "model": li["model"], "revision": li["revision"]},
           "admitted": True, "columns": COLUMNS, "documents": docs, "documents_sha256": canonical_sha(docs),
           "documents_file": frozen, "selection_sha256": report["selection_sha256"],
           "before_packing": {"training": _composition(sel["train"], sel),
                              "validation": _composition(sel["validation"], sel)},
           "proportions": report, "tool_calling": sel["tool"], "approved_exceptions": verdict["approved_exceptions"],
           "validation_coverage": validation_coverage(sel["train"], sel["validation"], sel["data_type"]),
           "missing_tasks_read_from_records": sel["missing_tasks_read_from_records"], "sources": sel["sources"],
           "authorizes": "nothing"}
    man["content_sha256"] = canonical_sha(man)
    p.write_text(json.dumps(man, indent=1) + "\n")
    return {"path": str(p), "sha256": sha_file(p), "manifest": man}


def write_documents(sel: dict, path: Path) -> dict:
    """The admitted documents' token content, frozen beside the manifest: per split the concatenated token ids,
    supervision masks and offsets, in the manifest's row order. Its sha256 enters the manifest, so the training
    binding can re-derive the packed sequences from exactly the admitted documents (`verify_pack`) -- a document
    the packing truncates at the sequence length included."""
    import numpy as np
    arrays = {}
    for split, docs in (("training", sel["train"]), ("validation", sel["validation"])):
        n = [len(d.ids) for d in docs]
        arrays[f"{split}_ids"] = np.fromiter((t for d in docs for t in d.ids), dtype=np.int32, count=sum(n))
        arrays[f"{split}_mask"] = np.fromiter((bool(x) for d in docs for x in d.target_mask), dtype=bool, count=sum(n))
        arrays[f"{split}_offsets"] = np.concatenate([[0], np.cumsum(n, dtype=np.int64)]).astype(np.int64)
    np.savez(path, **arrays)
    return {"name": path.name, "sha256": sha_file(path), "arrays": sorted(arrays),
            "order": "per split, the manifest's document rows in order"}


def read_documents(manifest_path: Path, man: dict) -> dict:
    """``{split: [MaskedDocument]}``: the frozen documents of an admitted manifest, the file checked against the
    manifest's sha256 and every document against its row's content sha256 (`doc_sha256`), or PackBuildError."""
    import numpy as np
    from aadistill.data.token_documents import DocumentError, MaskedDocument
    f = man.get("documents_file") or {}
    path = Path(manifest_path).parent / str(f.get("name"))
    if not path.is_file() or sha_file(path) != f.get("sha256"):
        raise PackBuildError(f"{path}: the frozen documents are missing or not the admitted ones (sha256)")
    out = {}
    with np.load(path) as z:
        for split, rows in man["documents"].items():
            ids, mask, off = z[f"{split}_ids"], z[f"{split}_mask"], z[f"{split}_offsets"]
            if len(off) != len(rows) + 1:
                raise PackBuildError(f"{path}: {len(off) - 1} {split} documents, the manifest admits {len(rows)}")
            docs = []
            for k, r in enumerate(rows):
                a, b = int(off[k]), int(off[k + 1])
                try:
                    d = MaskedDocument(r[0], tuple(int(x) for x in ids[a:b]), tuple(bool(x) for x in mask[a:b]))
                except DocumentError as exc:
                    raise PackBuildError(f"{path}: {r[0]} is not a document ({exc})") from exc
                if doc_sha256(d) != r[-1]:
                    raise PackBuildError(f"{path}: {r[0]}'s tokens or supervision are not the admitted ones")
                docs.append(d)
            out[split] = docs
    return out


def read_manifest(path: Path) -> dict:
    """An admitted, intact manifest, or PackBuildError."""
    man = json.loads(Path(path).read_text())
    body = {k: v for k, v in man.items() if k != "content_sha256"}
    if man.get("schema") != MANIFEST_SCHEMA or man.get("admitted") is not True:
        raise PackBuildError(f"{path}: not an admitted data manifest")
    if canonical_sha(body) != man.get("content_sha256") or canonical_sha(man["documents"]) != man["documents_sha256"]:
        raise PackBuildError(f"{path}: the manifest was edited after its admission (content_sha256)")
    return man


# --- 4. packing ----------------------------------------------------------------------------------------------------

PACKED_CONTENT = "aadistill.stage2.teacher_modes.packed_content/v1"


def packed_content_sha256(input_ids, ce_mask, content_mask, train_sequences: int) -> str:
    """The packed training set's content identity: the token ids, the supervision mask and the content mask of every
    sequence, training sequences first, each array named, with its normalized dtype and shape, and the number of
    training sequences -- so no array, split boundary or layout can stand in for another."""
    import numpy as np
    h = hashlib.sha256(PACKED_CONTENT.encode())
    for name, a, dt in (("input_ids", input_ids, "<i8"), ("ce_mask", ce_mask, "u1"), ("content_mask", content_mask, "u1")):
        a = np.ascontiguousarray(np.asarray(a).astype(dt))
        h.update(f"|{name}|{dt}|{list(a.shape)}|".encode())
        h.update(a.tobytes())
    h.update(f"|train_sequences|{int(train_sequences)}".encode())
    return h.hexdigest()

def pack(train, validation, manifest_path: Path, out_dir: Path) -> dict:
    """Pack FROZEN data: random shuffle -> sequential packing -> truncation at the design's sequence length and seed
    (``stage3_packing``), in the trainer's ladder contract. First checks that ``train`` / ``validation`` are exactly
    the admitted manifest's documents (ids and token content) and that the manifest is intact; reads no quota and
    computes no proportion. The ladder names the manifest."""
    from aadistill.data.token_documents import pack_sequential, write_pack
    man = read_manifest(manifest_path)
    for split, docs in (("training", train), ("validation", validation)):
        want = sorted((r[0], r[-1]) for r in man["documents"][split])
        got = sorted((d.doc_id, doc_sha256(d)) for d in docs)
        if want != got:
            raise PackBuildError(f"the {split} documents are not the frozen manifest's ({len(got)} given, "
                                 f"{len(want)} admitted, {len(set(got) ^ set(want))} differing)")
    d = json.loads(DESIGN.read_text())["stage3_packing"]
    seq, seed = int(d["sequence_length"]), int(d["shuffle_seed"])
    import torch
    tr = pack_sequential(list(train), seq, PAD_ID, order_seed=seed)
    v = pack_sequential(list(validation), seq, PAD_ID, order_seed=seed)
    cat = lambda name: torch.cat([getattr(tr, name), getattr(v, name)]).numpy()  # noqa: E731
    content = packed_content_sha256(cat("input_ids"), cat("target_mask"), cat("content_mask"), len(tr.members))
    dtype = {r[0]: r[1] for rows in man["documents"].values() for r in rows}
    return write_pack(tr, v, Path(out_dir), data_type=dtype, extra={
        "arm": man["arm"], "mode": man["mode"], "teacher": man["teacher"], "context_strategy": "cross_context",
        "data_admission": {"manifest": os.path.relpath(manifest_path, out_dir), "manifest_sha256": sha_file(manifest_path),
                           "content_sha256": man["content_sha256"], "packed_content_sha256": content,
                           "packed_content": f"{PACKED_CONTENT}: `packed_content_sha256` of the packed sequences as "
                                             "the packing produced them, before they were written"}})


def verify_pack(pack_dir: Path) -> dict:
    """A formal consumer's check of a built pack: it names an admitted manifest whose bytes are unchanged, holds
    exactly that manifest's documents (by id, per split; its rung = the manifest's unique supervised tokens) and was
    packed with the design's packing spec; and its ACTUAL sequences are the ones its packing produced and the
    packing of the manifest's documents: (1) `packed_content_sha256` of ``blocks.npz``'s token ids, supervision mask
    and content mask equals the identity the packing recorded before writing them; (2) the frozen documents
    (`read_documents`: the file by the manifest's sha256, every document by its row's content sha256), packed again
    by the formal algorithm at the design's spec -- truncation and prefix restarts included -- reproduce every
    sequence (tokens, supervision, content) and every placement exactly. Returns the manifest, or PackBuildError."""
    import numpy as np
    from aadistill.data.sequence_packing import SequencePackingSpec
    from aadistill.data.token_documents import pack_sequential
    pack_dir = Path(pack_dir)
    lad = json.loads((pack_dir / "ladder.json").read_text())
    adm = lad.get("data_admission")
    if not adm:
        raise PackBuildError(f"{pack_dir}: the pack names no admitted data manifest")
    mp = pack_dir / adm["manifest"]
    if not mp.is_file() or sha_file(mp) != adm["manifest_sha256"]:
        raise PackBuildError(f"{pack_dir}: its data manifest is missing or not the one it was packed from")
    man = read_manifest(mp)
    d = json.loads(DESIGN.read_text())["stage3_packing"]
    want = SequencePackingSpec(sequence_length=int(d["sequence_length"]), shuffle_seed=int(d["shuffle_seed"])).as_dict()
    for key in ("packing_spec", "val_packing_spec"):
        if {k: (lad.get(key) or {}).get(k) for k in want} != want:
            raise PackBuildError(f"{pack_dir}: {key} is not the design's stage3_packing ({want})")
    ids = {"train": set(), "val": set()}
    for line in (pack_dir / "audit.jsonl").read_text().splitlines():
        row = json.loads(line)
        ids[row["split"]].update(s["session_id"] for s in row["sessions"])
    for split, key in (("training", "train"), ("validation", "val")):
        if ids[key] != {r[0] for r in man["documents"][split]}:
            raise PackBuildError(f"{pack_dir}: its {split} sequences do not hold exactly the manifest's documents")
    if lad["rungs"][0]["target_supervised_tokens"] != man["before_packing"]["training"]["unique_supervised_tokens"]:
        raise PackBuildError(f"{pack_dir}: its rung is not the manifest's unique supervised tokens")
    with np.load(pack_dir / "blocks.npz") as arr:
        tok, ce, content = arr["input_ids"], arr["ce_mask"], arr["content_mask"]
    n_train = int(lad["n_blocks"])
    if packed_content_sha256(tok, ce, content, n_train) != adm.get("packed_content_sha256"):
        raise PackBuildError(f"{pack_dir}: its packed token ids, supervision mask or content mask are not the ones its "
                             "packing produced (packed_content_sha256)")
    rows = [json.loads(x) for x in (pack_dir / "audit.jsonl").read_text().splitlines() if x.strip()]
    frozen = read_documents(mp, man)
    for split, key, lo, hi in (("training", "train", 0, n_train), ("validation", "val", n_train, len(tok))):
        again = pack_sequential(frozen[split], int(d["sequence_length"]), PAD_ID, order_seed=int(d["shuffle_seed"]))
        placed = [[(m["session_id"], m["start"], m["length"], m["final"]) for m in r["sessions"]]
                  for r in rows if r["split"] == key]
        if not (np.array_equal(again.input_ids.numpy(), tok[lo:hi]) and np.array_equal(again.target_mask.numpy(), ce[lo:hi])
                and np.array_equal(again.content_mask.numpy(), content[lo:hi])
                and placed == [[(m["doc_id"], m["start"], m["length"], m["final"]) for m in ms] for ms in again.members]):
            raise PackBuildError(f"{pack_dir}: its {split} sequences are not the packing of the manifest's documents "
                                 "at the design's spec")
    return man


# --- the workflow --------------------------------------------------------------------------------------------------

def build(arm: str, *, tool_docs: Path | None, out: Path = OUT, records: dict | None = None,
          pool_dir: Path | None = None) -> dict:
    """Selection -> proportion validation -> admission / freeze -> packing. A refused admission raises
    (`AdmissionError`) before anything is packed. Returns the pack's ladder record and its manifest."""
    sel = select(arm, tool_docs=tool_docs, records=records, pool_dir=pool_dir)
    report = proportions(arm, sel)
    adm = admit(arm, sel, report, out)
    rec = pack(sel["train"], sel["validation"], Path(adm["path"]), Path(out) / arm)
    return {**rec, "data_manifest": {"path": adm["path"], "sha256": adm["sha256"]}, "manifest": adm["manifest"]}


def proportion_audit() -> dict:
    """The registered tolerance on every arm's real five base-corpus types and fluid part (the tool type is not
    generated yet), beside the withdrawn ``stage3_packing.share_check``. A1/A2: counts from the selection record
    (its five types and fluid as selected), largest whole units from the source packs (a selected task missing from
    a pack is named); C1: D4A_merged_whole's kept sessions."""
    from aadistill.data.mixture import validate_proportions
    sel = json.loads(SELECTION.read_text())
    arms = {}
    for arm, spec in ARMS.items():
        if spec["source_pack"]:
            selected = {t for c in sel["selection"][spec["mode"]].values() for t in c}
            pairs = [(d, t) for d, t in PA.documents_from_isolated_pack(PA.PACKS / spec["source_pack"], "train")
                     if d.doc_id.split("|", 1)[0] in selected]
            docs, dtype = [d for d, _ in pairs], {d.doc_id: t for d, t in pairs}
            _, _, largest = type_counts(arm, docs, dtype, {d.doc_id: d.doc_id.split("|", 1)[0] for d in docs})
            counts = {**{k: int(v) for k, v in sel["per_type_supervised_tokens"][SELECTION_ARM[arm]].items()
                         if k != "tool_calling"}, "fluid": sel["arms"][arm]["by_category"]["fluid"]}
            note = {"tasks_missing_from_the_pack": sorted(selected - {d.doc_id.split("|", 1)[0] for d in docs})}
        else:
            docs, _, dtype = c1_documents()
            counts, _, largest = type_counts(arm, docs, dtype, {d.doc_id: d.doc_id for d in docs})
            note = {}
        quotas, _, rule = registered_quotas(arm, largest, 7)
        five = {k: quotas[k] for k in (*SIX_NAMES[:-1], "fluid")}
        rep = validate_proportions(counts, five)
        withdrawn = round(0.02 * sum(counts[k] for k in SIX_NAMES[:-1]) / 5, 1)    #: 2% of 1/6 of the total
        arms[arm] = {**note, "registered": {k: {**r, "width": five[k]["above"]} for k, r in rep["quotas"].items()},
                     "passed": rep["passed"], "largest_whole_unit": largest, "withdrawn_width_per_type": withdrawn,
                     "types_the_withdrawn_check_held_wider": sorted(
                         k for k in SIX_NAMES[:-1] if five[k]["above"] is not None and withdrawn > five[k]["above"])}
    return {"schema": "aadistill.stage2.teacher_modes.proportion_tolerance_audit/v1",
            "command": "PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_packs "
                       "--proportion-audit logs/stages/stage-2/teacher_modes/audit/proportion_tolerance_audit.json",
            "question": "maintainer 2026-11-04: is stage3_packing.share_check (per_type_relative 0.02, fluid_fraction "
                        "[0.49, 0.51]) the tolerance of the frozen data-selection protocol?",
            "registered_rule": rule["rule"], "owners": rule["owners"], "tolerance": rule["tolerance"],
            "withdrawn_check": {"per_type_relative": 0.02, "fluid_fraction": [0.49, 0.51],
                                "introduced": "77be6e13 (2026-11-03), by the agent, without a maintainer decision",
                                "status": "withdrawn 2026-11-04; the admission reads the registered rule"},
            "verdict": "NOT the registered tolerance, and looser where it matters: (1) no floor -- the registered rule "
                       "holds fluid and supporting each at least their targets, the withdrawn check measured each type "
                       "against 1/6 of the ACTUAL supporting total, so a type ~2% short (or all six short together) "
                       "passed; (2) fluid's share +-1 point instead of +-0.5; (3) no bound on the total's overshoot "
                       "(registered: 0.5%); (4) a fixed ~2% width per type instead of one whole unit per selection "
                       "cell, wider for most types (types_the_withdrawn_check_held_wider)",
            "arms": arms, "tool_type": "not generated: its six-type row and its seven cells are checked at admission",
            "authorizes": "nothing"}


def recover(mode: str, missing: list[str]) -> tuple[list[dict], list, dict]:
    """``(rows, documents, data_type)``: each ``missing`` task recovered from the FROZEN sealed generation records
    (six_category_selection.json :: records) by `token_admission.load`'s own functions for that (task, mode) only --
    every label's shards checked against its .complete and the store identity, the admission, the code-stability
    recheck and the isolation recheck. Documents only for a task admitted and not flagged."""
    from stages.reasoning_data import generation_driver as GD
    from stages.teacher_modes import admission as ADM
    from stages.teacher_modes import build_supplement_tasks as BST
    from stages.teacher_modes import code_stability as CS
    from stages.teacher_modes import token_admission as TA
    rec = json.loads(SELECTION.read_text())
    proto = ADM.protocol()
    tasks = ADM.tasks_and_stages(proto)[0]
    tasks.update({t["task_id"]: t for t in BST.read_jsonl(TA.ORDER_V2)})
    found, stores = {}, {}
    for which in ("v1", "supplement"):                       #: load(): the supplement's records replace v1's
        store, ident = REPO / rec["records"][which]["dir"], rec["records"][which]["identity"]
        labels = 0
        for done in sorted(store.glob("*.complete")):
            label = done.name[:-len(".complete")]
            if label.startswith(ADM.CALIBRATION_GROUPS) or label.split("_")[1] != mode:
                continue
            listed = {x["name"]: x["sha256"] for x in json.loads(done.read_text())["shards"]}
            if {q.name: GD.sha(q) for q in store.glob(f"{label}.part*.jsonl")} != listed:
                raise PackBuildError(f"{label}: the shards present are not the ones its .complete lists")
            labels += 1
            sealed = GD.sealed(store, label, ident)
            for tid in missing:
                if tid in sealed:
                    found.setdefault(which, {}).setdefault(tid, []).append(sealed[tid])
        stores[which] = {**rec["records"][which], "labels_verified": labels}
    rows, docs, dtype = [], [], {}
    for tid in missing:
        recs = sorted((found.get("supplement", {}).get(tid) or found.get("v1", {}).get(tid) or []),
                      key=lambda r: r["candidate_index"])
        key = (tid, mode)
        verdicts = {key: ADM.admit(tasks[tid], mode, recs, proto)}
        code = CS.recheck(verdicts, tasks, {key: recs}, proto, ADM.admit)
        flagged, iso = ADM.isolation_recheck({k: v for k, v in verdicts.items() if v["admitted"]}, tasks,
                                             sources=BST.catalogue())
        v, stratum = verdicts[key], TA.cell_of(tasks[tid]).split("/", 1)[0]
        docs += v["docs"] if v["admitted"] and key not in flagged else []
        dtype.update({d.doc_id: stratum for d in v["docs"]})
        rows.append({"task_id": tid, "cell": TA.cell_of(tasks[tid]), "records_from": [w for w in ("v1", "supplement")
                                                                                      if tid in found.get(w, {})],
                     "candidates": [r["candidate_index"] for r in recs], "admitted": v["admitted"],
                     "isolation_flagged": key in flagged, "isolation_recheck_pass": iso.get("pass"),
                     "code_stability_changed": code.get("changed"),
                     "documents": [{"doc_id": d.doc_id, "mode_suffix": d.doc_id.split("|")[1],
                                    "prompt_tokens": len(d.prompt_ids), "supervised_tokens": len(d.output_ids),
                                    "ends_on_a_stop_id": d.output_ids[-1] in (151645, 151643),
                                    "content_sha256": doc_sha256(d)} for d in v["docs"]],
                     "stores": stores})
    return rows, docs, dtype


def recovery_audit(arm: str) -> dict:
    """The arm's selected tasks that its source pack lacks, each recovered from the FROZEN sealed generation records
    (six_category_selection.json :: records) by `token_admission.load`'s own functions for that (task, mode) only --
    every label's shards checked against its .complete and the store identity, the admission, the code-stability
    recheck and the isolation recheck -- with its documents and the arm's counts recomputed with them against the
    selection record. The full path (`select` with the records) re-derives every verdict, including the sandboxed
    stability recheck of every program and MBPP task, which runs far longer on the dev box; the functions are the same."""
    spec, rec = ARMS[arm], json.loads(SELECTION.read_text())
    mode = spec["mode"]
    selected = {t for c in rec["selection"][mode].values() for t in c}
    pairs = [(d, t) for d, t in PA.documents_from_isolated_pack(PA.PACKS / spec["source_pack"], "train")
             if d.doc_id.split("|", 1)[0] in selected]
    missing = sorted(selected - {d.doc_id.split("|", 1)[0] for d, _ in pairs})
    rows, docs, dt = recover(mode, missing)
    stores = rows[0]["stores"] if rows else {}
    out = [{k: v for k, v in r.items() if k != "stores"} for r in rows]
    dtype = {**{d.doc_id: t for d, t in pairs}, **dt}
    train = [d for d, _ in pairs] + docs
    counts, _, _ = type_counts(arm, train, dtype, {d.doc_id: d.doc_id.split("|", 1)[0] for d in train})
    want = {k: int(x) for k, x in rec["per_type_supervised_tokens"][SELECTION_ARM[arm]].items() if k != "tool_calling"}
    want["fluid"] = rec["arms"][arm.removesuffix("-cs")]["by_category"]["fluid"]   #: one base selection per mode
    return {"schema": "aadistill.stage2.teacher_modes.recovery_audit/v1",
            "command": "PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_packs --arm "
                       f"{arm} --recovery-audit <out>",
            "question": "maintainer 2026-11-04 (review of 647eea0e), item 6: the selected task(s) the old source pack "
                        "lacks, recovered through the frozen admission records, tokens and source identity verified",
            "arm": arm, "mode": mode, "source_pack": _files(PA.PACKS / spec["source_pack"]),
            "selection_record_sha256": sha_file(SELECTION), "records": stores, "missing_from_the_pack": out,
            "counts_with_them": counts, "selection_record_counts": want,
            "equal_to_the_selection_record": all(counts.get(k) == want[k] for k in want), "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--tool-documents", default=None)
    ap.add_argument("--tool-pool", default=None, help="the pool directory (default: tool_generation_protocol.json)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--validate-only", action="store_true",
                    help="select and validate the proportions; admit and pack nothing")
    ap.add_argument("--proportion-audit", default=None, help="write the registered-tolerance audit here and stop")
    ap.add_argument("--recovery-audit", default=None,
                    help="with --arm A1-6cat|A2-6cat: write the recovery audit of the selected tasks its pack lacks")
    for k in ("v1-records", "v1-identity", "records", "identity"):
        ap.add_argument(f"--{k}", default=None)
    a = ap.parse_args(argv)
    if a.proportion_audit:
        doc = proportion_audit()
        Path(a.proportion_audit).write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({k: v["passed"] for k, v in doc["arms"].items()}))
        return 0
    if not a.arm:
        ap.error("--arm is required")
    if a.recovery_audit:
        doc = recovery_audit(a.arm)
        Path(a.recovery_audit).write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({k: doc[k] for k in ("missing_from_the_pack", "equal_to_the_selection_record")}, indent=1))
        return 0
    records = ({"v1_records": a.v1_records, "v1_identity": a.v1_identity, "records": a.records,
                "identity": a.identity} if a.v1_records else None)
    tool_docs = Path(a.tool_documents) if a.tool_documents else None
    pool_dir = Path(a.tool_pool) if a.tool_pool else None
    if a.validate_only:
        sel = select(a.arm, tool_docs=tool_docs, records=records, pool_dir=pool_dir)
        rep = proportions(a.arm, sel)
        print(json.dumps({"proportions": rep, "admission": admission_verdict(a.arm, sel, rep)}, indent=1))
        return 0
    rec = build(a.arm, tool_docs=tool_docs, out=Path(a.out), records=records, pool_dir=pool_dir)
    print(json.dumps({k: rec[k] for k in ("packing", "n_blocks", "n_val_blocks", "files_sha256", "data_manifest")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
