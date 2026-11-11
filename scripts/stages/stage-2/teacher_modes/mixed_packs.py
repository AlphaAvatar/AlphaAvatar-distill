#!/usr/bin/env python3
"""A-MIX-6cat-sr's training data: A1-6cat-sr's thinking and A2-6cat-sr's non-thinking documents, admitted as they
were frozen, in ONE mixed manifest and ONE sequential pack. $0, dev box.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.mixed_packs --build
    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.mixed_packs --verify

Maintainer 2026-11-09 (UTC), "Research A 新增 Hybrid Mixed Stage 3 与 Checkpoint 精简", sections I-III. Nothing is
generated, selected, copied to fill a quota or re-tokenized:

1. **Sources** (`sources`): each source arm's admitted manifest and frozen documents (`data_documents.npz`), the
   manifest's bytes, content identity and documents file checked against the -sr freeze record
   (`six_category_schema_restored_freeze.json`), every document against its row's content sha256
   (`packed_packs.read_documents`). The packed sequences of A1/A2 are never read: two packs are not concatenated.
2. **Think state** (`think_states`): every supervised span checked on its tokens -- a thinking document's span opens
   with ``<think>`` right after ``<|im_start|>assistant\\n`` (the template's self-opening state), a non-thinking
   document's span follows ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` (preclosed). Teacher and student
   read the same packed token ids (`aadistill.training.train.Trainer`), so the mode is carried by the tokens alone.
3. **Isolation** (`isolation`): across BOTH modes, no task (unit), task content (the catalogue's content sha256) or
   first user message (its token ids) of a training document is a validation one, and the union stays disjoint from
   eval-set-1's held-out units (`shared.eval_sets.require_disjoint`). The same training task in both modes is two
   legitimate trajectories. Rule families are reported: a validation family that only the OTHER mode trains on would
   be the one exposure mixing adds.
4. **Manifest** (`manifest`): every document with its source arm, mode, teacher revision, source document id, data
   type, category, whole unit, supervised tokens and content sha256 (`COLUMNS`); the counts before packing per mode
   and in total, the category proportions per mode (the source admissions' reports, unchanged), the sources and the
   isolation result. Document ids are the sources' own, which carry their mode (``<task>|H-T|t0``, ``<task>|H-N|t0``);
   only if two collided would every id be prefixed with its mode (`mixed_ids`), never the tokens.
5. **Packing** (`pack`): random shuffle (keyed by document id under the design's seed, so the two modes are permuted
   together at the document level) -> ONE sequential packing -> truncation, at the design's ``stage3_packing``
   (8,192, prefix_restart_v1, cross-context, continuous positions), validation the same; checked by the formal
   consumer's `packed_packs.verify_pack`, which re-derives every sequence from the frozen documents.

`freeze` writes the arm's freeze record: the identities `packed_training_plan.bind_pack` binds, the counts before
packing, the packing / exposure / compute measures beside them (never a data size), and the comparison with A1/A2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.teacher_modes import packed_packs as PP  # noqa: E402

ARM = "A-MIX-6cat-sr"
#: source arm -> its mode: the lineage's name, the manifest's code, the template state and kwargs it was generated in
SOURCES = {"A1-6cat-sr": {"mode": "thinking", "code": "H-T", "think_state": "self_opening",
                          "template": {"enable_thinking": True}},
           "A2-6cat-sr": {"mode": "non_thinking", "code": "H-N", "think_state": "preclosed",
                          "template": {"enable_thinking": False}}}
SOURCE_FREEZE = REPO / "logs/stages/stage-2/teacher_modes/audit/six_category_schema_restored_freeze.json"
FREEZE = REPO / "logs/stages/stage-2/teacher_modes/audit/mixed_6cat_sr_freeze.json"
ISOLATION = REPO / "logs/stages/stage-2/teacher_modes/audit/mixed_6cat_sr_isolation.json"
OUT = PP.OUT
SCHEMA = PP.MIXED_MANIFEST_SCHEMA
COLUMNS = ["doc_id", "source_arm", "mode", "teacher_revision", "source_doc_id", "data_type", "category", "unit",
           "supervised_tokens", "sha256"]
#: the hybrid template's tokens around an assistant turn (Qwen3: <|im_start|> assistant \n <think> \n\n </think> \n\n)
IM_START, IM_END, ASSISTANT, NL, NLNL, THINK, THINK_CLOSE = 151644, 151645, 77091, 198, 271, 151667, 151668
OPENS = {"self_opening": (IM_START, ASSISTANT, NL), "preclosed": (IM_START, ASSISTANT, NL, THINK, NLNL, THINK_CLOSE, NLNL)}
SPLITS = ("training", "validation")
CATALOGUE_V2 = REPO / "artifacts/stages/stage-2/teacher_modes/tasks/order_v2.jsonl.gz"


class MixedPackError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return PP.sha_file(path)


# --- 1. sources ------------------------------------------------------------------------------------------------------

def sources(packs: Path = OUT, freeze: Path = SOURCE_FREEZE) -> dict:
    """``{arm: {"manifest", "path", "docs": {split: [MaskedDocument]}, "identity"}}``: each source arm's admitted manifest
    and frozen documents, bound to the -sr freeze record (manifest bytes, content identity, documents file), every
    document re-hashed against its row. Refuses (`MixedPackError`) on any difference."""
    fr = json.loads(Path(freeze).read_text())["arms"]
    out = {}
    for arm, spec in SOURCES.items():
        mp = Path(packs) / arm / PP.MANIFEST
        ident = fr[arm]["identity"]
        if fr[arm]["status"] != "FROZEN" or _sha(mp) != ident["data_manifest_sha256"]:
            raise MixedPackError(f"{arm}: {mp} is not the frozen manifest")
        man = PP.read_manifest(mp)
        if man["content_sha256"] != ident["content_sha256"] or man["documents_file"]["sha256"] != \
                ident["documents_file"]["sha256"] or man["mode"] != spec["code"]:
            raise MixedPackError(f"{arm}: the manifest's content, documents file or mode is not the frozen one")
        out[arm] = {"manifest": man, "path": mp, "docs": PP.read_documents(mp, man), "identity": ident}
    teachers = {json.dumps(s["manifest"]["teacher"], sort_keys=True) for s in out.values()}
    if len(teachers) != 1:
        raise MixedPackError(f"the sources come from more than one teacher: {sorted(teachers)}: one lineage per student")
    return out


# --- 2. think state ------------------------------------------------------------------------------------------------

def spans(mask) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def span_state(ids, start: int) -> str | None:
    """The template state a supervised span starting at ``start`` was generated in, read from its tokens."""
    pre = tuple(ids[max(0, start - len(OPENS["preclosed"])):start])
    if pre == OPENS["preclosed"]:
        return "preclosed"
    if tuple(ids[max(0, start - 3):start]) == OPENS["self_opening"] and ids[start] == THINK:
        return "self_opening"
    return None


def think_states(src: dict) -> dict:
    """Every supervised span's think state per source arm and split, against the state its mode declares."""
    out = {}
    for arm, s in src.items():
        want = SOURCES[arm]["think_state"]
        for split, docs in s["docs"].items():
            c = Counter(span_state(d.ids, a) for d in docs for a, _ in spans(d.target_mask))
            out[f"{arm}/{split}"] = {"declared": want, "spans": sum(c.values()), "by_state": dict(c),
                                     "all_declared": set(c) == {want}}
    return out


# --- 3. isolation --------------------------------------------------------------------------------------------------

def _catalogue() -> dict:
    """task id -> its catalogue row (order_v2, then the v1 tasks the admission reads)."""
    import gzip
    from stages.teacher_modes import admission as ADM
    rows = {}
    with gzip.open(CATALOGUE_V2, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["task_id"]] = r
    for tid, r in ADM.tasks_and_stages(ADM.protocol())[0].items():
        rows.setdefault(tid, r)
    return rows


def first_user(ids) -> str | None:
    """sha256 of the first non-system, non-assistant message's tokens (what a task asks), or None."""
    for i in range(len(ids) - 2):
        if ids[i] == IM_START and ids[i + 1] not in (8948, ASSISTANT):     #: 8948 "system"
            j = i + 2
            while j < len(ids) and ids[j] != IM_END:
                j += 1
            return hashlib.sha256(json.dumps(list(ids[i + 1:j])).encode()).hexdigest()
    return None


def isolation(src: dict, *, catalogue: dict | None = None, eval_set: bool = True) -> dict:
    """Training vs validation across BOTH modes: tasks, task content and first user messages must not cross (refusal
    in `build`); rule families reported per mode and mixed; the union against eval-set-1's held-out units."""
    cat = catalogue if catalogue is not None else _catalogue()
    rows = {s: [] for s in SPLITS}                          #: (arm, row, document)
    for arm, x in src.items():
        for split in SPLITS:
            for r, d in zip(x["manifest"]["documents"][split], x["docs"][split]):
                rows[split].append((arm, r, d))
    unit = {s: {r[3] for _, r, _ in rows[s]} for s in SPLITS}
    content = {s: {cat[u]["content_sha256"] for u in unit[s] if u in cat and cat[u].get("content_sha256")}
               for s in SPLITS}
    user = {s: {first_user(d.ids) for _, _, d in rows[s]} - {None} for s in SPLITS}
    fam = lambda u, dt: (cat[u].get("family_id") if u in cat else None) or f"tool:{dt}"  # noqa: E731
    fams = {(arm, s): {fam(r[3], r[1]) for a, r, _ in rows[s] if a == arm} for arm in src for s in SPLITS}
    val_fams = {arm: fams[(arm, "validation")] for arm in src}
    train_all = set().union(*(fams[(a, "training")] for a in src))
    family = {}
    for arm in src:
        own = fams[(arm, "training")]
        other = set().union(*(fams[(a, "training")] for a in src if a != arm))
        family[arm] = {"validation_families": len(val_fams[arm]),
                       "trained_in_own_mode": len(val_fams[arm] & own),
                       "trained_only_in_the_other_mode": sorted(val_fams[arm] & (other - own)),
                       "not_trained": len(val_fams[arm] - train_all)}
    crossing = {"units": sorted(unit["training"] & unit["validation"]),
                "task_content": sorted(content["training"] & content["validation"]),
                "first_user_message": sorted(user["training"] & user["validation"])}
    shared = {s: sorted({r[3] for a, r, _ in rows[s] if a == "A1-6cat-sr"}
                        & {r[3] for a, r, _ in rows[s] if a == "A2-6cat-sr"}) for s in SPLITS}
    out = {"crossing": {k: {"n": len(v), "examples": v[:5]} for k, v in crossing.items()},
           "passed": not any(crossing.values()),
           "units": {s: len(unit[s]) for s in SPLITS},
           "units_in_both_modes": {s: len(v) for s, v in shared.items()},
           "units_without_catalogue_row": {s: len([u for u in unit[s] if u not in cat]) for s in SPLITS},
           "families": family,
           "rule": "no training task, task content (catalogue content_sha256) or first user message (token ids) of "
                   "either mode is a validation one of either mode; a task trained in both modes is two legitimate "
                   "trajectories; a tool episode's family is its category"}
    if eval_set:
        from shared import eval_sets as ES
        cand = [cat[u] for u in sorted(unit["training"] | unit["validation"]) if u in cat]
        rep = ES.require_disjoint(cand)
        out["eval_set_1_held_out"] = {"candidate_tasks": len(cand), "checked": rep.get("checked"),
                                      "overlaps": {k: v for k, v in rep.items() if k != "checked"} or None,
                                      "tool_episodes": "isolated by the tool sources' formal admission, which both "
                                                       "modes share (same pool, same admitted tasks)"}
    return out


# --- 4. manifest ---------------------------------------------------------------------------------------------------

def mixed_ids(src: dict) -> dict[tuple[str, str], str]:
    """(source arm, source doc id) -> the mixed id: the source id when every id is unique across the sources, else
    every id prefixed with its mode code. Tokens are never touched."""
    keys = [(arm, r[0]) for arm, x in src.items() for s in SPLITS for r in x["manifest"]["documents"][s]]
    ids = [k[1] for k in keys]
    if len(set(ids)) == len(ids):
        return {k: k[1] for k in keys}
    return {k: f"{SOURCES[k[0]]['code']}::{k[1]}" for k in keys}


def _composition(docs, rows) -> dict:
    out = {"examples": len({r[3] for r in rows}), "documents": len(docs), "unique_supervised_tokens": 0,
           "reasoning_tokens": 0, "answer_tokens": 0, "tool_calling_tokens": 0}
    for d, r in zip(docs, rows):
        for a, b in spans(d.target_mask):
            span = d.ids[a:b]
            k = span.index(THINK_CLOSE) + 1 if THINK_CLOSE in span else 0
            out["unique_supervised_tokens"] += b - a
            out["reasoning_tokens"] += k
            out["answer_tokens"] += b - a - k
            if r[1] == PP.TOOL_TYPE:
                out["tool_calling_tokens"] += b - a
    return out


def manifest(src: dict, iso: dict, states: dict) -> tuple[dict, dict]:
    """``(manifest without its seal, {split: [MaskedDocument]})`` in row order: per split, the sources in `SOURCES`
    order, each in its own manifest's order (the packing's order is the shuffle's, not this one)."""
    from aadistill.data.token_documents import MaskedDocument
    ids = mixed_ids(src)
    docs, rows = {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    for arm, x in src.items():
        rev = x["manifest"]["teacher"]["revision"]
        for s in SPLITS:
            for r, d in zip(x["manifest"]["documents"][s], x["docs"][s]):
                new = ids[(arm, r[0])]
                rows[s].append([new, arm, SOURCES[arm]["mode"], rev, r[0], r[1], r[2], r[3], r[4], r[5]])
                docs[s].append(d if new == d.doc_id else MaskedDocument(new, d.ids, d.target_mask))
    before = {}
    for s in SPLITS:
        per = {SOURCES[arm]["mode"]: _composition(x["docs"][s], x["manifest"]["documents"][s]) for arm, x in src.items()}
        for arm, x in src.items():                         #: the mixed counts are the sources' own, exactly
            if per[SOURCES[arm]["mode"]] != x["manifest"]["before_packing"][s]:
                raise MixedPackError(f"{arm} {s}: the counts before packing are not its admitted manifest's")
        tot = {k: sum(p[k] for p in per.values()) for k in next(iter(per.values()))}
        tot["examples"] = len({r[7] for r in rows[s]})
        before[s] = {**tot, "by_mode": per}
    total = before["training"]["unique_supervised_tokens"]
    proportions = {
        "counted": "unique supervised tokens before packing",
        "mode_shares": {m: round(p["unique_supervised_tokens"] / total, 6) for m, p in before["training"]["by_mode"].items()},
        "by_mode": {SOURCES[arm]["mode"]: {"report_sha256": x["manifest"]["proportions"]["report_sha256"],
                                           "passed": x["manifest"]["proportions"]["passed"],
                                           "six_types_and_fluid": x["manifest"]["proportions"]["counts"],
                                           "by_data_type": x["manifest"]["proportions"]["by_data_type"]}
                    for arm, x in src.items()},
        "rule": "each mode keeps its own admitted selection and proportion report unchanged; the mixed set adds, "
                "removes, copies and re-weights nothing"}
    by_type = Counter()
    for arm, x in src.items():
        by_type.update(x["manifest"]["proportions"]["by_data_type"])
    proportions["by_data_type_total"] = dict(sorted(by_type.items()))
    man = {"schema": SCHEMA, "arm": ARM, "modes": [SOURCES[a]["mode"] for a in src],
           "teacher": next(iter(src.values()))["manifest"]["teacher"], "admitted": True, "columns": COLUMNS,
           "documents": rows, "documents_sha256": PP.canonical_sha(rows),
           "sources": {arm: {"mode": SOURCES[arm]["mode"], "mode_code": SOURCES[arm]["code"],
                             "think_state": SOURCES[arm]["think_state"], "template": SOURCES[arm]["template"],
                             "manifest": os.path.relpath(x["path"], REPO), "manifest_sha256": _sha(x["path"]),
                             "content_sha256": x["manifest"]["content_sha256"],
                             "documents_file_sha256": x["manifest"]["documents_file"]["sha256"]}
                       for arm, x in src.items()},
           "source_freeze_record": {"path": os.path.relpath(SOURCE_FREEZE, REPO), "sha256": _sha(SOURCE_FREEZE)},
           "document_ids": ("the sources' own ids, unique across the sources (each carries its mode code)"
                            if all(k[1] == v for k, v in ids.items()) else "each prefixed with its mode code"),
           "before_packing": before, "proportions": proportions, "think_states": states,
           "isolation": {"passed": iso["passed"], "record": os.path.relpath(ISOLATION, REPO),
                         "crossing": {k: v["n"] for k, v in iso["crossing"].items()}},
           "authorizes": "nothing"}
    return man, docs


# --- 5. packing and the workflow -----------------------------------------------------------------------------------

def pack(docs: dict, manifest_path: Path, out_dir: Path) -> dict:
    """The mixed documents in one sequential stream at the design's ``stage3_packing``, in the trainer's ladder
    contract, named to the manifest (as `packed_packs.pack`, whose packing it is; the ladder records both modes)."""
    import torch
    from aadistill.data.token_documents import pack_sequential, write_pack
    man = PP.read_manifest(manifest_path)
    d = json.loads(PP.DESIGN.read_text())["stage3_packing"]
    seq, seed = int(d["sequence_length"]), int(d["shuffle_seed"])
    tr = pack_sequential(docs["training"], seq, PP.PAD_ID, order_seed=seed)
    v = pack_sequential(docs["validation"], seq, PP.PAD_ID, order_seed=seed)
    cat = lambda name: torch.cat([getattr(tr, name), getattr(v, name)]).numpy()  # noqa: E731
    content = PP.packed_content_sha256(cat("input_ids"), cat("target_mask"), cat("content_mask"), len(tr.members))
    dtype = {r[0]: r[5] for rows in man["documents"].values() for r in rows}
    return write_pack(tr, v, Path(out_dir), data_type=dtype, extra={
        "arm": ARM, "mode": "mixed", "modes": man["modes"], "teacher": man["teacher"], "context_strategy": "cross_context",
        "data_admission": {"manifest": os.path.relpath(manifest_path, out_dir), "manifest_sha256": _sha(manifest_path),
                           "content_sha256": man["content_sha256"], "packed_content_sha256": content,
                           "packed_content": f"{PP.PACKED_CONTENT}: `packed_content_sha256` of the packed sequences "
                                             "as the packing produced them, before they were written"}})


def mode_of_rows(man: dict) -> dict[str, str]:
    return {r[0]: r[2] for rows in man["documents"].values() for r in rows}


def mode_exposure(pack_dir: Path, man: dict) -> dict:
    """Per mode, what the sequences train on (from the pack's own audit rows): placements, fragments, supervised
    tokens placed and repeated, and the sequences holding each mode -- packing / exposure measures, not data sizes."""
    mode = mode_of_rows(man)
    out = {m: Counter() for m in man["modes"]}
    seqs = {m: set() for m in man["modes"]}
    mixed_seqs = 0
    for line in (Path(pack_dir) / "audit.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] != "train":
            continue
        here = set()
        for s in row["sessions"]:
            m = mode[s["session_id"]]
            here.add(m)
            out[m]["placements"] += 1
            out[m]["fragments"] += 0 if s["final"] else 1
            out[m]["supervised_tokens_placed"] += int(s["supervised_retained"])
            seqs[m].add(row["block"])
        mixed_seqs += len(here) > 1
    res = {m: {**dict(c), "sequences_holding_it": len(seqs[m])} for m, c in out.items()}
    for m in res:
        uniq = man["before_packing"]["training"]["by_mode"][m]["unique_supervised_tokens"]
        res[m]["repeated_supervised_tokens"] = res[m]["supervised_tokens_placed"] - uniq
        res[m]["supervised_token_exposure_3_epochs"] = 3 * res[m]["supervised_tokens_placed"]
    return {"by_mode": res, "sequences_with_both_modes": mixed_seqs}


def build(out: Path = OUT, packs: Path = OUT) -> dict:
    """Sources -> think states -> isolation (refuses a crossing) -> manifest and frozen documents -> one packing."""
    src = sources(packs)
    states = think_states(src)
    bad = [k for k, v in states.items() if not v["all_declared"]]
    if bad:
        raise MixedPackError(f"supervised spans not in their mode's think state: {bad}")
    iso = isolation(src)
    ISOLATION.write_text(json.dumps({"schema": "aadistill.stage2.teacher_modes.mixed_isolation_audit/v1", "arm": ARM,
                                     "written_utc": _now(), **iso, "authorizes": "nothing"}, indent=1) + "\n")
    if not iso["passed"]:
        raise MixedPackError(f"training and validation cross between the modes: {iso['crossing']}")
    man, docs = manifest(src, iso, states)
    d = Path(out) / ARM
    d.mkdir(parents=True, exist_ok=True)
    man["documents_file"] = PP.write_documents({"train": docs["training"], "validation": docs["validation"]},
                                               d / PP.DOCUMENTS)
    man["content_sha256"] = PP.canonical_sha(man)
    mp = d / PP.MANIFEST
    mp.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    rec = pack(docs, mp, d)
    return {"manifest": str(mp), "manifest_sha256": _sha(mp), "pack": rec}


def verify(pack_dir: Path = OUT / ARM) -> dict:
    """The formal consumer's check (`packed_packs.verify_pack`: the admitted manifest, its documents, the design's
    spec and every sequence re-derived from the frozen documents), then the manifest's sources against the -sr
    freeze record as it is now. Returns the manifest."""
    man = PP.verify_pack(Path(pack_dir))
    if man.get("schema") != SCHEMA or man["arm"] != ARM:
        raise MixedPackError(f"{pack_dir}: not {ARM}'s mixed manifest")
    fr = json.loads(SOURCE_FREEZE.read_text())["arms"]
    for arm, s in man["sources"].items():
        if s["manifest_sha256"] != fr[arm]["identity"]["data_manifest_sha256"] or \
                s["content_sha256"] != fr[arm]["identity"]["content_sha256"]:
            raise MixedPackError(f"{pack_dir}: source {arm} is not the frozen one")
    return man


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def freeze(pack_dir: Path = OUT / ARM) -> dict:
    """The arm's freeze record (`FREEZE`): what `packed_training_plan.bind_pack` binds, and the comparison with A1/A2
    in the field's terms. Written after `verify`."""
    import math
    man = verify(pack_dir)
    lad = json.loads((Path(pack_dir) / "ladder.json").read_text())
    src_fr = json.loads(SOURCE_FREEZE.read_text())
    tr = lad["train"]
    T = math.ceil(3 * int(lad["n_blocks"]) / 2)
    a = src_fr["a1_a2_compute"]
    compute = {"packed_sequences": int(lad["n_blocks"]), "validation_sequences": int(lad["n_val_blocks"]),
               "optimizer_steps": T, "non_padding_tokens": int(tr["non_padding_tokens"]),
               "supervised_tokens_in_sequences": int(tr["supervised_tokens"]),
               "supervised_token_exposure": 3 * int(tr["supervised_tokens"]),
               "training_flops_ratio": {
                   "over_A1-6cat-sr": round(int(lad["n_blocks"]) / a["packed_sequences"]["A1-6cat-sr"], 4),
                   "over_A2-6cat-sr": round(int(lad["n_blocks"]) / a["packed_sequences"]["A2-6cat-sr"], 4),
                   "over_A1_plus_A2": round(int(lad["n_blocks"]) / (a["packed_sequences"]["A1-6cat-sr"]
                                                                   + a["packed_sequences"]["A2-6cat-sr"]), 4),
                   "rule": "every packed sequence is 8,192 positions with standard causal attention, so training "
                           "FLOPs scale with packed sequences x epochs (the same student, teacher and objective)"}}
    doc = {"schema": "aadistill.stage2.teacher_modes.mixed_freeze/v1", "study_id": "six_category_schema_restored@v1",
           "written_utc": _now(),
           "decided_by": "maintainer 2026-11-09 (UTC), 'Research A 新增 Hybrid Mixed Stage 3 与 Checkpoint 精简', "
                         "sections I-II",
           "source_freeze_record": man["source_freeze_record"],
           "isolation_record": {"path": os.path.relpath(ISOLATION, REPO), "sha256": _sha(ISOLATION)},
           "arms": {ARM: {
               "arm": ARM, "modes": man["modes"], "teacher": man["teacher"], "status": "FROZEN",
               "identity": {"data_manifest_sha256": _sha(Path(pack_dir) / PP.MANIFEST),
                            "content_sha256": man["content_sha256"], "documents_sha256": man["documents_sha256"],
                            "documents_file": man["documents_file"],
                            "packed_content_sha256": lad["data_admission"]["packed_content_sha256"],
                            "pack_files_sha256": {n: _sha(Path(pack_dir) / n)
                                                  for n in ("blocks.npz", "ladder.json", "audit.jsonl")}},
               "sources": man["sources"], "before_packing": man["before_packing"],
               "proportions": man["proportions"], "think_states": man["think_states"],
               "packing": {"spec": lad["packing_spec"], "train": tr, "validation": lad["val"],
                           "by_mode": mode_exposure(pack_dir, man)},
               "compute": compute}},
           "comparison": {
               "with_A1": "the same thinking documents, plus A2's non-thinking ones: unique supervised tokens ~2.0x, "
                          "packed sequences and training FLOPs as `compute.training_flops_ratio`",
               "with_A2": "the same non-thinking documents, plus A1's thinking ones",
               "reading": "a mixed-minus-single difference confounds mode mixing with about twice the unique data and "
                          "a different compute and exposure; it is descriptive, never a pure mixing effect"},
           "authorizes": "nothing: training needs its own decision"}
    FREEZE.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="admit, write the manifest and documents, pack")
    ap.add_argument("--verify", action="store_true", help="the formal consumer's check of the built pack")
    ap.add_argument("--freeze", action="store_true", help="verify, then write the freeze record")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.build:
        rec = build(Path(a.out))
        print(json.dumps({"manifest_sha256": rec["manifest_sha256"],
                          **{k: rec["pack"][k] for k in ("n_blocks", "n_val_blocks", "files_sha256")}}, indent=1))
    if a.verify or a.freeze:
        man = verify(Path(a.out) / ARM)
        print(json.dumps({"verified": True, "content_sha256": man["content_sha256"]}))
    if a.freeze:
        doc = freeze(Path(a.out) / ARM)
        print(json.dumps(doc["arms"][ARM]["compute"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
