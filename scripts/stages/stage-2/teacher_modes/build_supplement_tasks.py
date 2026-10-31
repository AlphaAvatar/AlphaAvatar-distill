#!/usr/bin/env python3
"""Research A v2's supplementary tasks: one queue per cell still short of its token quota. $0, CPU, before generation.

    # 1. the need per (mode, cell) from the sealed v1 records' audit, the planning yields, the export whitelists
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_supplement_tasks plan \\
        --rows artifacts/stages/stage-2/teacher_modes/audit/supply_rows.jsonl --out-dir <scratch>
    # 2. in the isolated third-party env, one export per whitelist (seed base SEED_BASE):
    #    PYTHONHASHSEED=0 <rgenv>/bin/python -I scripts/stages/stage-2/reasoning_data/external_export.py \\
    #        --whitelist <scratch>/whitelist_n<N>.json --n <N> --seed-base 2026103100 --out <export.jsonl>
    # 3. the order, its isolation report and its manifest (refuses unless every rule passes)
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_supplement_tasks build --exports <export.jsonl> ...

THE QUOTAS are token_matched_design.json's: per mode, each cell's share of its category's tokens. THE NEED of a
(mode, cell) is its quota minus the supervised tokens the frozen v1 verdict admits for it in the sealed records
(T0, E and X; V is validation), and a cell is built here only when some mode is short. Every task is NEW and
built from its own source's construction, never another's:

* the project generators on a fresh block of R5's training range (``lo + PROJECT_BLOCK + group x 10,000 + i``;
  order_v1 used ``lo + 500,000 + ...``, the L40S check ``lo + 900,000 + ...``), per T0 group (generator x
  difficulty), the calibration-split families minus the mechanism-excluded ones;
* the third-party generators (reasoning_core, reasoning_gym) through H1's isolated export at a new seed base
  (``SEED_BASE``; order_v1 used 2026102900 and 2026102901), per T0 module and level;
* MATH train continued in its pool builder's hash order past every pooled and ordered task;
* GSM8K train, SQuAD v2 train (answerable) and HotpotQA distractor train in their own ROW order, rendered exactly
  as corpus_v2 rendered them (`shared.data.build_stage2_v0`: the raw question, RAG_PROMPT, HOTPOT_PROMPT), so E1's
  own prompts -- the first rows -- come first; the rendering is checked byte for byte against corpus_v2's sessions.

A family with no admitted task in either mode over at least ``SKIP_MIN`` generated v1 tasks is not built. The
first ``V2_TASKS`` tasks of each new dataset cell are V2, the validation reserve. Nothing is chosen by looking at a
teacher generation of these tasks: none exists.

ISOLATION. Every new task passes `isolation_check.check_pool(refuse=True)` with this study's source catalogue
(R0-R9 against eval-set-1 and its held-out rule families; R1/R2/ARC-grid items, held-out prompt keys and the
calibration prompt keys -- the v1 profiles' and calibration_v2's -- are removed first and counted), and shares no task id or content with T0, the
pools, order_v1, the L40S check's diagnostic tasks or another new task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.reasoning_data import build_supporting_tasks as BS  # noqa: E402
from stages.reasoning_data import build_training_tasks as BT  # noqa: E402
from stages.reasoning_data import external_tasks as ET  # noqa: E402
from stages.reasoning_data import generation_plan as GP  # noqa: E402
from stages.reasoning_data import isolation_check as IC  # noqa: E402
from stages.teacher_modes import build_tasks as B1  # noqa: E402

DESIGN = REPO / "configs/stages/stage-2/teacher_modes/token_matched_design.json"
CATALOGUE = REPO / "configs/stages/stage-2/teacher_modes/token_matched_sources.json"
PLAN = REPO / "logs/stages/stage-2/teacher_modes/data/supplement_plan.json"
#: the hybrid's D4-scale calibration supplement (build_calibration_supplement.py): held out of training, like v1's
CALIBRATION_V2 = REPO / "data/stages/stage-2/teacher_modes/tasks/calibration_v2.jsonl"
OUT_DIR = REPO / "data/stages/stage-2/teacher_modes/tasks"
NAME = "order_v2"
#: the order itself is heavy (about 232 MB of rendered prompts): a deterministic gzip outside git, bound by sha256
ORDER = REPO / "artifacts/stages/stage-2/teacher_modes/tasks/order_v2.jsonl.gz"
SCHEMA = "aadistill.stage2.teacher_modes.supplement_order/v1"
PLAN_SCHEMA = "aadistill.stage2.teacher_modes.supplement_plan/v1"
MODES = ("H-T", "H-N")
PROJECT, EXTERNAL = B1.PROJECT, ("reasoning_gym", "reasoning_core")
PROJECT_BLOCK, SPAN = 600_000, 10_000
SEED_BASE = 2026103100
SKIP_MIN = 25
V2_TASKS = 32
#: tasks built per cell: the largest mode's planned tasks times this (the pod generates only what each round needs)
BUILD_FACTOR = 1.6
#: dataset candidates passed to isolation per cell, over its build count (isolation removes about 1%)
ISOLATION_HEADROOM = 1.1
NEW_DATASETS = ("gsm8k_train", "squad_v2_train_answerable", "hotpot_qa_train")
#: planning yields for the three new dataset cells, before any record of them exists: (accepted share, supervised
#: tokens per accepted document, generated tokens per task). Lengths: the hybrid's own eval-set-1 gsm8k, rag and
#: multihop generations in each mode (means; teacher_eval evidence, lengths only); accepted shares are planning
#: assumptions. Round 1 replaces every one with the session's measured figure.
PLANNING = {("H-T", "gsm8k_train"): (0.85, 1239, 1400), ("H-N", "gsm8k_train"): (0.80, 238, 250),
            ("H-T", "squad_v2_train_answerable"): (0.90, 283, 320), ("H-N", "squad_v2_train_answerable"): (0.85, 26, 28),
            ("H-T", "hotpot_qa_train"): (0.75, 506, 600), ("H-N", "hotpot_qa_train"): (0.70, 73, 80)}
STRATUM = {"gsm8k_train": "supporting.math_gsm8k", "squad_v2_train_answerable": "supporting.reading_verified",
           "hotpot_qa_train": "supporting.reading_verified"}


def design() -> dict:
    return json.loads(DESIGN.read_text())


def catalogue() -> dict:
    """sources.json with this study's extension entries added or replacing the base entry of their source id."""
    ext = json.loads(CATALOGUE.read_text())
    base_path = REPO / ext["extends"]["path"]
    if A.sha256_file(base_path) != ext["extends"]["sha256"]:
        raise SystemExit(f"{ext['extends']['path']} is not the catalogue the extension was written against")
    base = json.loads(base_path.read_text())
    by = {s["source_id"]: s for s in base["sources"]}
    by.update({s["source_id"]: s for s in ext["sources"]})
    return {**base, "sources": list(by.values())}


def quotas(d: dict | None = None, mbpp_cap: int | None = None) -> dict[tuple[str, str], float]:
    """(stratum, source) -> supervised-token quota per mode; the MBPP cell's cap is the smaller mode's supply."""
    d = d or design()
    out = {tuple(k.split("/", 1)): v * d["scale"]["categories"]["fluid"]
           for k, v in d["categories"]["fluid"]["cells"].items()}
    base = d["scale"]["categories"]["base_counterpart"]
    for name, t in d["categories"]["base_counterpart"]["types"].items():
        if not t.get("cells"):
            continue
        q = t["share"] * base
        if name == "code":
            if mbpp_cap is None:
                raise ValueError("the code type's split needs the MBPP cap")
            out[("supporting.code_algorithmic", "mbpp_train")] = float(mbpp_cap)
            out[("supporting.code_algorithmic", "reasoning_core")] = q - mbpp_cap
        else:
            out[tuple(t["cells"][0])] = q
    return out


def category_of(cell: tuple[str, str]) -> str:
    return "fluid" if cell[0].startswith("primary.") else "base_counterpart"


# ---- the plan: need per (mode, cell), yields, skips --------------------------------------------------------

def plan(rows_path: Path) -> dict:
    rows = [r for r in A.jsonl(rows_path) if r["slice"] != "V"]
    by = defaultdict(lambda: {"tasks": 0, "admitted": 0, "tok": 0, "gen": 0})
    fam = defaultdict(lambda: {"tasks": 0, "admitted": 0, "tok": 0})
    for r in rows:
        k = (r["mode"], r["stratum"], r["source"])
        for a in (by[k],):
            a["tasks"] += 1
            a["admitted"] += r["admitted"]
            a["tok"] += r["supervised_tokens"]
            a["gen"] += r["generated_tokens"]
        f = fam[(r["mode"], r["stratum"], r["source"], r["family"])]
        f["tasks"] += 1
        f["admitted"] += r["admitted"]
        f["tok"] += r["supervised_tokens"]
    mbpp_cap = min(by[(m, "supporting.code_algorithmic", "mbpp_train")]["tok"] for m in MODES)
    q = quotas(mbpp_cap=mbpp_cap)
    skip = {m: sorted(f"{s}/{src}/{f}" for (mm, s, src, f), a in fam.items()
                      if mm == m and a["tasks"] >= SKIP_MIN and a["admitted"] == 0) for m in MODES}
    never = sorted(set(skip["H-T"]) & set(skip["H-N"]))
    cells, total_gen = {}, Counter()
    for cell, quota in sorted(q.items()):
        rec = {"category": category_of(cell), "quota": round(quota, 1), "modes": {}}
        for m in MODES:
            a = by[(m, *cell)]
            have = a["tok"]
            need = max(0.0, quota - have)
            skipped = [f for f in skip[m] if f.startswith(f"{cell[0]}/{cell[1]}/")]
            kept = [(k, v) for k, v in fam.items() if k[:3] == (m, *cell) and f"{cell[0]}/{cell[1]}/{k[3]}" not in skipped]
            kt, ka, kk = (sum(v["tasks"] for _, v in kept), sum(v["admitted"] for _, v in kept),
                          sum(v["tok"] for _, v in kept))
            if (m, cell[1]) in PLANNING:
                p, tpd, gpt = PLANNING[(m, cell[1])]
                tok_per_task, gen_per_task, basis = p * tpd, gpt, "planning figure (PLANNING)"
            else:
                tok_per_task = kk / kt if kt else 0.0
                gen_per_task = a["gen"] / a["tasks"] if a["tasks"] else 0.0
                basis = "v1 records of this (mode, cell), skipped families out"
            tasks = math.ceil(need / tok_per_task) if need and tok_per_task else 0
            rec["modes"][m] = {"have": have, "need": round(need, 1), "admitted_tasks": a["admitted"],
                               "generated_tasks": a["tasks"], "tokens_per_generated_task": round(tok_per_task, 2),
                               "generated_tokens_per_task": round(gen_per_task, 1), "yield_basis": basis,
                               "planned_tasks": tasks, "skipped_families": skipped,
                               "unreachable": bool(need) and not tok_per_task}
            total_gen[m] += tasks * gen_per_task
        cells["/".join(cell)] = rec
    return {"schema": PLAN_SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "design": {"path": str(DESIGN.relative_to(REPO)), "sha256": A.sha256_file(DESIGN)},
            "supply_rows": {"path": str(rows_path), "sha256": A.sha256_file(rows_path)},
            "mbpp_cap": mbpp_cap, "skip": skip, "never_built": never, "cells": cells,
            "planned_generated_tokens": {m: round(v) for m, v in total_gen.items()},
            "_": ("need = quota - the supervised tokens the frozen v1 verdict admits for the (mode, cell) in T0, E and X; "
                  "planned_tasks = need / admitted tokens per generated task; the session re-sizes every round from its "
                  "own records"), "authorizes": "nothing"}


def build_counts(p: dict) -> dict[str, int]:
    """Tasks to build per cell: the largest mode's planned tasks x BUILD_FACTOR, over the families it generates."""
    out = {}
    for cell, rec in p["cells"].items():
        n = max(v["planned_tasks"] for v in rec["modes"].values())
        if n:
            out[cell] = math.ceil(n * BUILD_FACTOR) + (V2_TASKS if cell.split("/", 1)[1] in NEW_DATASETS else 0)
    return out


def t0_rows() -> list[dict]:
    by = B1.pools()
    return [by[t] for t in B1.t0_ids()]


def whitelists(p: dict, t0: list[dict]) -> dict[int, list[dict]]:
    """Per export size n: the (source, module, levels) entries of the external cells to build, never-built
    families out; n is about the cell's build count over its (module, level) groups."""
    counts, by_n = build_counts(p), defaultdict(list)
    never = set(p["never_built"])
    for cell, n_cell in counts.items():
        stratum, source = cell.split("/", 1)
        if source not in EXTERNAL:
            continue
        mods = defaultdict(set)
        for t in t0:
            if (t.get("stratum") or B1.cell_of(t)[0]) == stratum and t["source"] == source \
                    and f"{stratum}/{source}/{source}/{t['module']}" not in never:
                mods[t["module"]].add(int(t["level"]))
        groups = sum(len(v) for v in mods.values())
        per = 1 << max(0, math.ceil(math.log2(math.ceil(n_cell / groups * B1.MARGIN))))
        for m, levels in sorted(mods.items()):
            by_n[per].append({"source": source, "module": m, "levels": sorted(levels)})
    return dict(by_n)


# ---- new instances -------------------------------------------------------------------------------------------

def _parquet_rows(repo: str, pattern: str) -> tuple[list[dict], list[dict]]:
    snap = BS._snapshot(repo)
    files = sorted(snap.glob(pattern))
    rows = [r for f in files for r in BS._parquet(f)]
    return rows, [{"file": f.relative_to(snap).as_posix(), "sha256": A.sha256_file(f), "snapshot": snap.name}
                  for f in files]


def _dataset_task(source: str, idx: int, prompt: str, answer: str, **fields) -> dict:
    t = BS._task(source, f"row{idx:06d}", prompt, source_split="train", stratum=STRATUM[source], family_id=source,
                 difficulty="", answer=answer, axis=STRATUM[source], row_index=idx, **fields)
    return t


def gsm8k_tasks() -> tuple[list[dict], dict]:
    rows, files = _parquet_rows("openai/gsm8k", "main/train-*.parquet")
    out = [_dataset_task("gsm8k_train", i, r["question"].strip(), r["answer"].strip(), verifier="final_number",
                         answer_type="final_number", e1_id=f"gsm8k-{i:06d}") for i, r in enumerate(rows)]
    return out, {"dataset": "openai/gsm8k main, train", "files": files, "render": "build_stage2_v0.build_gsm8k"}


def squad_answerable_tasks() -> tuple[list[dict], dict]:
    from shared.data.build_stage2_v0 import RAG_PROMPT
    rows, files = _parquet_rows("rajpurkar/squad_v2", "squad_v2/train-*.parquet")
    out = []
    for i, r in enumerate(rows):
        if not r["answers"]["text"]:
            continue
        prompt = RAG_PROMPT.format(context=r["context"].strip(), question=r["question"].strip())
        out.append(_dataset_task("squad_v2_train_answerable", i, prompt, r["answers"]["text"][0].strip(),
                                 verifier="span", answer_type="span", group="rag_evidence", corpus_source="squad_v2",
                                 squad_id=r["id"], e1_id=f"squad_v2-{i:06d}",
                                 context_sha256=hashlib.sha256(r["context"].encode()).hexdigest()))
    return out, {"dataset": "rajpurkar/squad_v2, train, answerable questions", "files": files,
                 "render": "build_stage2_v0.build_squad (RAG_PROMPT)"}


def hotpot_tasks() -> tuple[list[dict], dict]:
    from shared.data.build_stage2_v0 import HOTPOT_PROMPT
    rows, files = _parquet_rows("hotpotqa/hotpot_qa", "distractor/train-*.parquet")
    out = []
    for i, r in enumerate(rows):
        paragraphs = "\n\n".join(f"[{k + 1}] {title}\n{' '.join(s.strip() for s in sents)}" for k, (title, sents)
                                 in enumerate(zip(r["context"]["title"], r["context"]["sentences"])))
        prompt = HOTPOT_PROMPT.format(paragraphs=paragraphs, question=r["question"].strip())
        out.append(_dataset_task("hotpot_qa_train", i, prompt, r["answer"].strip(), verifier="span",
                                 answer_type="span", group="multihop_qa", corpus_source="hotpot_qa",
                                 hotpot_id=r["id"], e1_id=f"hotpot-{i:06d}"))
    return out, {"dataset": "hotpotqa/hotpot_qa distractor, train", "files": files,
                 "render": "build_stage2_v0.build_hotpot (HOTPOT_PROMPT)"}


DATASET_BUILDERS = {"gsm8k_train": gsm8k_tasks, "squad_v2_train_answerable": squad_answerable_tasks,
                    "hotpot_qa_train": hotpot_tasks}


def e1_rendering_check(tasks: dict[str, list[dict]]) -> dict:
    """Every corpus_v2 session of the three types must be rendered byte for byte by its row's task."""
    want = {"gsm8k": "gsm8k_train", "rag_evidence": "squad_v2_train_answerable", "multihop_qa": "hotpot_qa_train"}
    by = {src: {t["e1_id"]: t for t in ts} for src, ts in tasks.items()}
    got = defaultdict(Counter)
    for s in A.corpus_v2_sessions():
        src = want.get(s["data_type"])
        if src is None:
            continue
        t = by[src].get(s["source_id"])
        user = next(m["content"] for m in s["messages"] if m["role"] == "user")
        got[src]["sessions"] += 1
        got[src]["identical" if t is not None and t["messages"][0]["content"] == user else "different"] += 1
        if t is not None and src != "gsm8k_train" and t["answer"].strip() != str(s.get("gold", "")).strip():
            got[src]["gold_differs"] += 1
    bad = {k: dict(v) for k, v in got.items() if v["different"] or v["gold_differs"]}
    if bad:
        raise SystemExit(f"the rendering does not reproduce corpus_v2's own prompts: {bad}")
    return {k: dict(v) for k, v in got.items()}


def taken() -> tuple[set[str], set[str]]:
    """Task ids and content keys no new task may repeat: T0, every pool, order_v1, the diagnostic tasks, H1's
    and Stage-0's instances."""
    by = B1.pools()
    rows = list(by.values()) + list(A.jsonl(OUT_DIR / "order_v1.jsonl")) + list(A.jsonl(OUT_DIR / "diagnostic_v1.jsonl"))
    return {t["task_id"] for t in rows}, BT.taken_instances() | {t["content_sha256"] for t in rows}


def project_tasks(cells: dict[str, int], t0: list[dict], content: set[str]) -> tuple[list[dict], dict]:
    need = {tuple(c.split("/", 1)): {"new_needed": math.ceil(n / B1.MARGIN)} for c, n in cells.items()
            if c.split("/", 1)[1] == PROJECT}
    if not need:
        return [], {}
    lo, hi = next(r for r in IC.rules()["rules"] if r["id"] == "R5_shared_generator_exposure")["training_seed_range"]
    groups = sorted({g for c in need for g in B1.groups_of(t0, c)})
    if PROJECT_BLOCK + len(groups) * SPAN > B1.DIAG_BLOCK:
        raise SystemExit("the project groups' fresh block would reach the L40S check's diagnostic block")
    return B1.project_extension(t0, need, content, block=PROJECT_BLOCK, span=SPAN)


def isolate(new: list[dict], cat: dict) -> tuple[list[dict], set, dict, dict]:
    """build_tasks.isolate with this study's catalogue."""
    from shared import eval_sets as ES
    cal_keys = B1.calibration_keys() | {ES.prompt_key(ES.prompt_text(r)) for r in A.jsonl(CALIBRATION_V2)}
    first = IC.check_pool(new, sources=cat)
    flagged = set(first["results"]["R2_near_duplicate"]["flagged_ids"])
    held = {ES.prompt_key(ES.prompt_text(r)) for rows in ES.component_items(roles=ES.HELD_OUT_ROLES).values()
            for r in rows if ES.prompt_text(r).strip()}
    flagged |= {t["unit_id"] for t in new if ES.prompt_key(ES.prompt_text(t)) in held}
    flagged |= {t["unit_id"] for t in new if ES.prompt_key(ES.prompt_text(t)) in cal_keys}
    flagged |= {t["unit_id"] for t in new
                if IC.arc_matches([ES.prompt_text(t), str(t.get("gold") or t.get("answer") or "")])["refuse"]}
    new = [t for t in new if t["unit_id"] not in flagged]
    return new, flagged, IC.check_pool(new, refuse=True, sources=cat), first


def build(exports: list[Path]) -> dict:
    p = json.loads(PLAN.read_text())
    counts = build_counts(p)
    never = set(p["never_built"])
    t0 = t0_rows()
    ids, content = taken()
    new, report_parts = [], {}
    proj, short = project_tasks(counts, t0, content)
    new += proj
    report_parts["project_group_shortfalls"] = short
    if any(c.split("/", 1)[1] in EXTERNAL for c in counts):
        ext = ET.build(exports)
        report_parts["external_drops"] = ext["drops"]
        new += [t for t in ext["tasks"] if f"{t['stratum']}/{t['source']}/{t['source']}/{t['module']}" not in never]
    ds_meta, ds_tasks = {}, {}
    for src in NEW_DATASETS:
        ds_tasks[src], ds_meta[src] = DATASET_BUILDERS[src]()
    report_parts["e1_rendering"] = e1_rendering_check(ds_tasks)
    for src in NEW_DATASETS:
        cell = f"{STRATUM[src]}/{src}"
        if cell in counts:
            new += ds_tasks[src][:math.ceil(counts[cell] * ISOLATION_HEADROOM)]  #: the cut is per cell below
    if "supporting.math_new/math_train" in counts:
        cands, _ = BS.math_candidates()
        cands.sort(key=lambda t: BS._order(t["task_id"]))
        gone, _ = BS.math500_screen(cands)
        cands = [t for t in cands if t["task_id"] not in gone and t["task_id"] not in ids
                 and t["content_sha256"] not in content]
        new += cands[:math.ceil(counts["supporting.math_new/math_train"] * ISOLATION_HEADROOM)]
    seen, fresh = set(), []
    for t in new:                       #: project_extension already refused `content`; every other source here
        if t["task_id"] in ids or t["content_sha256"] in content or t["content_sha256"] in seen:
            continue
        seen.add(t["content_sha256"])
        fresh.append(t)
    cat = catalogue()
    kept, flagged, report, first = isolate(fresh, cat)
    sp = B1.spec()
    queues = defaultdict(list)
    for t in kept:
        c = "/".join(B1.cell_of(t, sp))         #: project instances carry their axis, not a stratum field
        if c in counts:
            queues[c].append(t)
    rows, events = [], []
    for c in sorted(counts):
        q = queues[c]
        if c.split("/", 1)[1] in (PROJECT, *EXTERNAL):
            q = GP.round_robin(q, len(q))
        if len(q) < counts[c]:
            events.append({"cell": c, "event": f"built {len(q)} of {counts[c]}"})
        for i, t in enumerate(q[:counts[c]]):
            s = "V2" if c.split("/", 1)[1] in NEW_DATASETS and i < V2_TASKS else "S"
            rows.append({**{k: v for k, v in t.items() if k != "_pool"}, "stratum": c.split("/", 1)[0], "cell": c,
                         "cell_position": i, "slice": s})
    for k, r in enumerate(rows):
        r["position"] = k
    return {"rows": rows, "counts": counts, "events": events, "isolation": report, "removed": sorted(flagged),
            "first_pass_violations": first["violations"], "datasets": ds_meta, **report_parts, "plan": p}


def read_jsonl(path: Path):
    """A JSONL file, gzip when its name ends in .gz."""
    import gzip
    with (gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write(got: dict, exports: list[Path]) -> dict:
    import gzip
    out = ORDER
    if out.exists():
        raise SystemExit(f"{out.relative_to(REPO)} exists: a frozen order is never rebuilt -- bump NAME")
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in got["rows"]).encode()
    with open(out, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, compresslevel=9, mtime=0) as gz:
        gz.write(raw)
    (OUT_DIR / f"{NAME}.isolation.json").write_text(json.dumps(got["isolation"], indent=1, default=str) + "\n")
    by_cell = Counter(f"{r['cell']}|{r['slice']}" for r in got["rows"])
    fams = defaultdict(Counter)
    for r in got["rows"]:
        fams[r["cell"]][GP.group_of(r) if r.get("generator_id") else (r.get("module") or r["source"])] += 1
    manifest = {
        "schema": SCHEMA, "name": NAME, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "python -m stages.teacher_modes.build_supplement_tasks build --exports " + " ".join(e.name for e in exports),
        "design": {"path": str(DESIGN.relative_to(REPO)), "sha256": A.sha256_file(DESIGN)},
        "plan": {"path": str(PLAN.relative_to(REPO)), "sha256": A.sha256_file(PLAN)},
        "catalogue": {"path": str(CATALOGUE.relative_to(REPO)), "sha256": A.sha256_file(CATALOGUE)},
        "n_tasks": len(got["rows"]), "built_per_cell": got["counts"],
        "by_cell_and_slice": dict(sorted(by_cell.items())), "groups_per_cell": {c: dict(v) for c, v in sorted(fams.items())},
        "events": got["events"], "never_built_families": got["plan"]["never_built"],
        "removed_before_isolation": len(got["removed"]), "first_pass_violations": got["first_pass_violations"],
        "project_group_shortfalls": got.get("project_group_shortfalls", {}),
        "external_exports": [{"name": e.name, "sha256": A.sha256_file(e), "rows": sum(1 for _ in A.jsonl(e))} for e in exports],
        "external_drops": got.get("external_drops", {}), "datasets": got["datasets"], "e1_rendering": got["e1_rendering"],
        "slices": {"S": "the supplement: a (mode, cell)'s generation takes its next tasks in cell_position order",
                   "V2": f"the first {V2_TASKS} tasks of each new dataset cell: validation documents only, both modes"},
        "seed_rules": {"project": f"R5 lo + {PROJECT_BLOCK:,} + group index x {SPAN:,} + i",
                       "external": f"external_export.entry_seed(seed base {SEED_BASE}, source, module, level)",
                       "math_train": "build_supporting_tasks._order (its SALT), past every pooled and ordered task",
                       "gsm8k, squad v2, hotpotqa": "the dataset's own row order (E1's prompts are its first rows)"},
        "jsonl_sha256": hashlib.sha256(raw).hexdigest(), "path": str(out.relative_to(REPO)),
        "sha256": A.sha256_file(out), "bytes": out.stat().st_size,
        "storage": ("outside git (232 MB uncompressed): the gzip (mtime 0, level 9) under artifacts/, its sha256 here "
                    "and in supplement_protocol.json; staged to the pod as a local asset and copied to the network "
                    "volume (teacher_modes/inputs/) by the session"),
        "isolation_report": f"{NAME}.isolation.json",
        "license": ("project-generated tasks; third-party generators' licenses as external_modules.json records; MATH (MIT), "
                    "GSM8K (MIT), SQuAD v2 and HotpotQA (CC-BY-SA-4.0) train splits only; no ARC-AGI material"),
        "status": "FROZEN at $0 before any of its tasks is generated (token_matched_design.json :: supplement)",
        "authorizes": "nothing"}
    (OUT_DIR / f"{NAME}.manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--rows", required=True)
    p.add_argument("--out-dir", required=True)
    b = sub.add_parser("build")
    b.add_argument("--exports", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        rec = plan(Path(a.rows))
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps(rec, indent=1) + "\n")
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        wl = whitelists(rec, t0_rows())
        for n, entries in wl.items():
            (out / f"whitelist_n{n}.json").write_text(json.dumps(entries, indent=1) + "\n")
        print(json.dumps({"build": build_counts(rec), "whitelists": {n: e for n, e in wl.items()},
                          "planned_generated_tokens": rec["planned_generated_tokens"], "never": rec["never_built"]},
                         indent=1))
        return 0
    m = write(build([Path(e) for e in a.exports]), [Path(e) for e in a.exports])
    print(json.dumps({k: m[k] for k in ("n_tasks", "built_per_cell", "events", "removed_before_isolation", "sha256")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
