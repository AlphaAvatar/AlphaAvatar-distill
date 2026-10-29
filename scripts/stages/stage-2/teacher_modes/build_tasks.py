#!/usr/bin/env python3
"""Research A's tasks beyond T0, in one deterministic order: V, then E, then X. $0, CPU, before any generation.

    # 1. how many new instances each source must supply; the external modules' export whitelists
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_tasks plan --out-dir <scratch>
    # 2. in each isolated env (external_export.py, as H1 built external_tasks_v1), one export per whitelist:
    #    PYTHONHASHSEED=0 <venv>/bin/python -I scripts/stages/stage-2/reasoning_data/external_export.py \\
    #        --whitelist <scratch>/whitelist_n<N>.json --n <N> --seed-base 2026102900 --out <export.jsonl>
    # 3. the order, its isolation report and its manifest (refuses unless every rule passes)
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_tasks build --exports <export.jsonl> ...
    # the L40S check's DIAGNOSTIC tasks (never training, calibration or evaluation data)
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_tasks diagnostic

THE CELLS are T0's (stratum, source) pairs, P1's cells, and each cell's share is
its share of T0's 1,761 tasks (d4_scale_design.json :: tasks). A cell's supply,
in this order:

1. the unused tasks of T0's own pools (train_tasks_v1 and v2 deduplicated by
   content, external_tasks_v2, math_train_v1, mbpp_train_v1,
   squad_v2_train_unanswerable_v1), round-robin over the cell's groups
   (`generation_plan.round_robin`, P1's rule);
2. then NEW tasks from the same source's own construction, never another
   source's: the project's generators on a fresh training seed block
   (``lo + 500,000 + cell index x 10,000 + i`` inside R5's range, which v1 and
   v2 never reached); the third-party generators through H1's isolated export
   at a new seed base (``SEED_BASE``) and `external_tasks.build`; MATH, MBPP and
   SQuAD v2 continued in their pool builder's own hash order
   (`build_supporting_tasks`), past every task their pools already hold.

THE ORDER interleaves the cells by largest remainder, so every prefix -- each
wave -- holds T0's cell proportions. A cell whose supply ends hands its share
to the other cells of its STRATUM in proportion to their shares (MBPP is the
one bounded source expected to: its 474 eligible tasks are fewer than its
share of the order), and the order records each substitution where it
happens; a stratum's own share never moves while any of its sources has
supply. V is the first ``V_TASKS`` (the per-mode validation documents), E the
next ``E_TASKS`` (2 x T0, the D4-scale extension, waves of ``WAVE``), X the
rest (H-N only).

ISOLATION. Every new task passes `isolation_check.check_pool(refuse=True)`
(R0-R9 against eval-set-1 and its held-out rule families; items R1, R2 or R7's
ARC-grid rule flag are removed first and counted), shares no content key with T0, the Stage-0 task
sets or a fluid evaluation item, and no prompt key with the calibration
profiles' prompts. Nothing is chosen by looking at a teacher
generation: none exists.

DIAGNOSTIC TASKS (``diagnostic``) are the L40S engineering check's inputs:
``DIAG_TASKS`` tasks in T0's proportions over the cells whose source can
construct new instances without an external export -- the project generators,
on their own seed block (``lo + DIAG_BLOCK + group index x 1,000 + i``, beyond
every seed the order used), and MATH and SQuAD v2 continued past every pooled
and ordered task -- through the same isolation (R0-R9, the held-out prompts,
the calibration prompts) and disjoint from T0, the order and the pools by task
id and content. Their generations measure memory and throughput only and are
never training, calibration or evaluation data.
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
from stages.reasoning_data import source_plan as SP  # noqa: E402

DESIGN = REPO / "configs/stages/stage-2/teacher_modes/d4_scale_design.json"
OUT_DIR = REPO / "data/stages/stage-2/teacher_modes/tasks"
NAME = "order_v1"
SCHEMA = "aadistill.stage2.teacher_modes.task_order/v1"
from stages.teacher_modes.teacher_session import POOLS  # noqa: E402 -- one owner: the pod reads the same pools
POOL_DIR = REPO / "data/stages/stage-2/reasoning_data/tasks"
CALIBRATION_ITEMS = ("artifacts/stages/stage-0/calibration_volume/profiles_formal_v1/calib.s0v_base_d4.items.jsonl",
                     "artifacts/stages/stage-0/calibration_volume/profiles_formal_v1/calib.s0v_rh_d4.items.jsonl")
V_TASKS, E_TASKS, X_TASKS, WAVE = 240, 3522, 14000, 400
#: the third-party exports' seed bases (external_tasks_v1/v2 ran at 2026102300): the extension's, and the
#: code stratum's supplement (its substitute modules supply MBPP's share once MBPP's 474 tasks run out)
SEED_BASE, SUPPLEMENT_SEED_BASE = 2026102900, 2026102901
#: the project generators' fresh block inside R5's training range
PROJECT_BLOCK = 500_000
#: new instances requested beyond a cell's need: drops (duplicates, size caps, isolation) are refilled from it
MARGIN = 1.3
#: the L40S check's diagnostic tasks: their own name, count and project seed block (beyond the order's
#: lo + 500,000 + 30 groups x 10,000)
DIAG_NAME, DIAG_TASKS, DIAG_BLOCK, DIAG_SPAN = "diagnostic_v1", 64, 900_000, 1_000
DIAG_SCHEMA = "aadistill.stage2.teacher_modes.diagnostic_tasks/v1"
PROJECT, EXTERNAL = "s0_generators_v2_calsplit", ("reasoning_gym", "reasoning_core", "prontoqa", "mathematics_dataset")
DATASETS = {"math_train": BS.math_candidates, "mbpp_train": BS.mbpp_candidates,
            "squad_v2_train_unanswerable": BS.squad_candidates}


def spec() -> dict:
    return A.read_json(A.CONFIG_DIR / "corpus_spec.json")


def cell_of(task: dict, sp: dict | None = None) -> tuple[str, str]:
    return (task.get("stratum") or SP.stratum_of(task, sp or spec()), task["source"])


def t0_ids() -> list[str]:
    ids = []
    for lst in t0_lists():
        ids += [json.loads(l)["task_id"] for l in (REPO / lst["path"]).read_text().splitlines() if l.strip()]
    return ids


def t0_lists() -> list[dict]:
    """T0's three frozen H1 lists, as hybrid_v1's design pins them (the D4-scale design carries T0 over)."""
    hybrid = json.loads((REPO / "configs/stages/stage-2/teacher_modes/design.json").read_text())
    lists = hybrid["tasks"]["T0"]["lists"]
    for lst in lists:
        if A.sha256_file(REPO / lst["path"]) != lst["sha256"]:
            raise SystemExit(f"{lst['path']}: not the T0 list the design pins")
    return lists


def pools() -> dict[str, dict]:
    """task id -> pool row, every pool T0 was drawn from (first listing wins: v2 before v1)."""
    by: dict[str, dict] = {}
    for name in POOLS:
        for t in A.jsonl(POOL_DIR / f"{name}.jsonl"):
            by.setdefault(t["task_id"], {**t, "_pool": name})
    return by


def t0_cells(by: dict) -> tuple[Counter, list[dict]]:
    sp = spec()
    rows = [by[t] for t in t0_ids()]
    return Counter(cell_of(t, sp) for t in rows), rows


def unused(by: dict, t0: list[dict]) -> dict[tuple, list[dict]]:
    """Each T0 cell's unused, verifiable pool tasks, distinct by content, round-robin over its groups."""
    sp, used = spec(), {t["task_id"] for t in t0} | {t["content_sha256"] for t in t0}
    cells: dict[tuple, list[dict]] = defaultdict(list)
    seen = set()
    for t in by.values():
        if t["task_id"] in used or t["content_sha256"] in used or t["content_sha256"] in seen:
            continue
        if GP.verifier_of(t)[0] is None:
            continue
        seen.add(t["content_sha256"])
        cells[cell_of(t, sp)].append(t)
    return {c: GP.round_robin(v, len(v)) for c, v in cells.items()}


def needs(t0c: Counter, supply: dict) -> dict[tuple, dict]:
    total, n = sum(t0c.values()), V_TASKS + E_TASKS + X_TASKS
    out = {}
    for c, k in sorted(t0c.items()):
        want = math.ceil(k / total * n)
        have = len(supply.get(c, []))
        out[c] = {"t0": k, "share": k / total, "order_need": want, "unused_supply": have,
                  "new_needed": max(0, want - have)}
    return out


def groups_of(t0: list[dict], cell: tuple) -> list[str]:
    sp = spec()
    return sorted({GP.group_of(t) for t in t0 if cell_of(t, sp) == cell})


def whitelists(t0: list[dict], need: dict) -> dict[int, list[dict]]:
    """Per export size n: the (source, module, levels) entries whose T0 groups need about n new rows each."""
    by_n: dict[int, list[dict]] = defaultdict(list)
    cands = ET.candidates()
    for c, row in need.items():
        if c[1] not in EXTERNAL or not row["new_needed"]:
            continue
        mods: dict[str, set] = defaultdict(set)
        for t in t0:
            if cell_of(t) == c:
                mods[t["module"]].add(int(t["level"]))
        groups = sum(len(v) for v in mods.values())
        per = math.ceil(row["new_needed"] / groups * MARGIN)
        n = 1 << max(0, math.ceil(math.log2(per)))          # a few export sizes, not one per module
        for m, levels in sorted(mods.items()):
            if (c[1], m) not in cands:
                raise SystemExit(f"{c[1]}/{m}: in T0 but no longer a candidate module")
            by_n[n].append({"source": c[1], "module": m, "levels": sorted(levels)})
    return dict(by_n)


# ---- new instances, each from its own source's construction ------------------------------------------

def project_extension(t0: list[dict], need: dict, taken: set[str], *, block: int = PROJECT_BLOCK,
                      span: int = 10_000) -> tuple[list[dict], dict]:
    """New project-generator instances, per T0 group (generator x difficulty), on the fresh seed block.

    Group j of the sorted project groups draws seeds ``lo + block + j x span + i`` (the order's:
    ``lo + PROJECT_BLOCK + j x 10,000 + i``). A group whose small instance space yields no further distinct
    instance inside its block stops there (the round-robin's other groups of the cell continue); the
    shortfall is returned and recorded."""
    from stages.calibration_volume.tasks import registry as R
    from stages.calibration_volume.tasks.common import Unsatisfiable, build_instance
    lo, hi = next(r for r in IC.rules()["rules"] if r["id"] == "R5_shared_generator_exposure")["training_seed_range"]
    gens = {g.generator_id: g for g in R.GENERATORS_V2}
    cells = [(c, row) for c, row in sorted(need.items()) if c[1] == PROJECT and row["new_needed"]]
    all_groups = sorted({g for c, _ in cells for g in groups_of(t0, c)})
    if lo + block + len(all_groups) * span > hi:
        raise SystemExit("the project groups do not fit the fresh block inside R5's training range")
    out, seen, short = [], set(taken), {}
    for c, row in cells:
        grp = groups_of(t0, c)
        per = math.ceil(row["new_needed"] / len(grp) * MARGIN)
        for g in grp:
            gid, difficulty = g.split("/", 1)
            gen = gens[gid]
            fams = [f for f in R.family_split(gen)["calibration"] if f"{gid}/{f}" not in BT.MECHANISM_EXCLUDED]
            j = all_groups.index(g)
            made = 0
            for i in range(span):
                if made >= per:
                    break
                seed = lo + block + j * span + i
                try:
                    t = build_instance(gen, gen.family(fams[i % len(fams)]), difficulty, seed)
                except (Unsatisfiable, IndexError, ValueError):
                    continue
                if t["content_sha256"] in seen:
                    continue
                seen.add(t["content_sha256"])
                t.update(role=BT.ROLE, unit_id=t["task_id"], source=PROJECT, source_split="calibration_families")
                out.append(t)
                made += 1
            if made < per:
                short[g] = {"wanted": per, "made": made}
    return out, short


def dataset_extension(need: dict, by: dict, taken: set[str]) -> list[dict]:
    """MATH, MBPP and SQuAD v2 continued in their builder's hash order, past every pooled task."""
    out = []
    for c, row in sorted(need.items()):
        if c[1] not in DATASETS or not row["new_needed"]:
            continue
        cands, _ = DATASETS[c[1]]()
        cands.sort(key=lambda t: BS._order(t["task_id"]))
        if c[1] == "math_train":
            gone, _ = BS.math500_screen(cands)
            cands = [t for t in cands if t["task_id"] not in gone]
        if c[1] == "squad_v2_train_unanswerable":
            pooled_ctx = {t.get("context_sha256") for t in by.values() if t["source"] == c[1]}
            seen, one = set(pooled_ctx), []
            for t in cands:
                if t["context_sha256"] not in seen:
                    seen.add(t["context_sha256"])
                    one.append(t)
            cands = one
        fresh = [t for t in cands if t["task_id"] not in by and t["content_sha256"] not in taken]
        out += fresh[:math.ceil(row["new_needed"] * MARGIN)]
    return out


def calibration_keys() -> set[str]:
    """Prompt keys of the calibration profiles' prompts (recovered from their ids, `calibration.prompt_messages`)."""
    from transformers import AutoTokenizer

    from shared import eval_sets as ES
    from stages.teacher_modes import hybrid_calibration as CAL
    from huggingface_hub.constants import HF_HUB_CACHE
    m, rev = CAL.SOURCE_TOKENIZER
    tok = AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE) / f"models--{m.replace('/', '--')}" / "snapshots" / rev)
    return {ES.prompt_key(ES.prompt_text({"messages": CAL.prompt_messages(it, tok)}))
            for prof in CAL.PROFILES for it in CAL.items(prof)}


# ---- the order ---------------------------------------------------------------------------------------------

def interleave(t0c: Counter, queues: dict[tuple, list[dict]], n: int) -> tuple[list[dict], list[dict]]:
    """Largest remainder over the cells; an exhausted cell's share goes to its stratum's other cells."""
    total = sum(t0c.values())
    share = {c: k / total for c, k in t0c.items()}
    pos = {c: 0 for c in share}
    taken = Counter()
    order, events = [], []
    live = {c for c in share if queues.get(c)}
    for i in range(n):
        if not live:
            events.append({"position": i, "event": "every cell exhausted"})
            break
        deficit = {d: share[d] * (i + 1) - taken[d] for d in live}
        for c in share:                                    # a dead cell's growing deficit goes to its stratum
            if c not in live:
                peers = [d for d in live if d[0] == c[0]]
                for d in peers:
                    deficit[d] += (share[c] * (i + 1) - taken[c]) * share[d] / sum(share[e] for e in peers)
        c = max(sorted(live), key=lambda d: deficit[d])
        order.append({**queues[c][pos[c]], "_cell": c})
        pos[c] += 1
        taken[c] += 1
        if pos[c] >= len(queues[c]):
            live.discard(c)
            events.append({"position": i + 1, "event": f"{c[0]}/{c[1]} exhausted; its share goes to "
                                                          f"{sorted(d[1] for d in live if d[0] == c[0]) or 'nothing'}"})
            if not any(d[0] == c[0] for d in live):
                events.append({"position": i + 1, "event": f"stratum {c[0]} exhausted"})
    return order, events


def build(exports: list[Path]) -> dict:
    by = pools()
    t0c, t0 = t0_cells(by)
    supply = unused(by, t0)
    need = needs(t0c, supply)
    taken = BT.taken_instances() | {t["content_sha256"] for t in by.values()}
    new, project_short = project_extension(t0, need, taken)
    taken |= {t["content_sha256"] for t in new}
    new += dataset_extension(need, by, taken)
    ext = ET.build(exports) if exports else {"tasks": [], "drops": {}}
    ext_tasks = [t for t in ext["tasks"] if t["task_id"] not in by and t["content_sha256"] not in taken]
    new += ext_tasks
    new, flagged, report, first = isolate(new)
    sp = spec()
    for t in new:
        supply.setdefault(cell_of(t, sp), [])
    fresh: dict[tuple, list[dict]] = defaultdict(list)
    for t in new:
        fresh[cell_of(t, sp)].append(t)
    queues = {c: supply.get(c, []) + GP.round_robin(fresh.get(c, []), len(fresh.get(c, []))) for c in t0c}
    order, events = interleave(t0c, queues, V_TASKS + E_TASKS + X_TASKS)
    return {"order": order, "events": events, "need": need, "isolation": report, "t0_cells": t0c,
            "project_group_shortfalls": project_short,
            "removed": sorted(flagged), "external_drops": ext["drops"], "new": len(new),
            "first_pass_violations": first["violations"]}


def isolate(new: list[dict]) -> tuple[list[dict], set, dict, dict]:
    """R1/R2-flagged, held-out-prompt, calibration-key and ARC-grid items out first, then every rule must pass."""
    cal_keys = calibration_keys()
    from shared import eval_sets as ES
    first = IC.check_pool(new)
    flagged = set(first["results"]["R2_near_duplicate"]["flagged_ids"])
    held = {ES.prompt_key(ES.prompt_text(r)) for rows in ES.component_items(roles=ES.HELD_OUT_ROLES).values()
            for r in rows if ES.prompt_text(r).strip()}
    flagged |= {t["unit_id"] for t in new if ES.prompt_key(ES.prompt_text(t)) in held}
    flagged |= {t["unit_id"] for t in new if ES.prompt_key(ES.prompt_text(t)) in cal_keys}
    #: R7's content rule per item (the report lists only the first 50): an instance carrying ARC-AGI-1 grids
    flagged |= {t["unit_id"] for t in new
                if IC.arc_matches([ES.prompt_text(t), str(t.get("gold") or t.get("answer") or "")])["refuse"]}
    new = [t for t in new if t["unit_id"] not in flagged]
    return new, flagged, IC.check_pool(new, refuse=True), first


def diagnostic(n: int = DIAG_TASKS) -> dict:
    """The L40S check's tasks (module docstring): T0's proportions over the self-constructing cells."""
    by = pools()
    t0c, t0 = t0_cells(by)
    order = list(A.jsonl(OUT_DIR / f"{NAME}.jsonl"))
    known = {**by, **{t["task_id"]: t for t in order}}
    cells = Counter({c: k for c, k in t0c.items() if c[1] == PROJECT or c[1] in ("math_train",
                                                                                   "squad_v2_train_unanswerable")})
    total = sum(cells.values())
    need = {c: {"new_needed": math.ceil(k / total * n)} for c, k in sorted(cells.items())}
    taken = BT.taken_instances() | {t["content_sha256"] for t in known.values()}
    new, short = project_extension(t0, need, taken, block=DIAG_BLOCK, span=DIAG_SPAN)
    taken |= {t["content_sha256"] for t in new}
    new += dataset_extension(need, known, taken)
    new, flagged, report, _ = isolate(new)
    sp = spec()
    fresh: dict[tuple, list[dict]] = defaultdict(list)
    for t in new:
        fresh[cell_of(t, sp)].append(t)
    queues = {c: GP.round_robin(fresh.get(c, []), len(fresh.get(c, []))) for c in cells}
    rows, events = interleave(cells, queues, n)
    clash = [t["task_id"] for t in rows if t["task_id"] in known or t["content_sha256"] in
             {k["content_sha256"] for k in known.values()}]
    if clash or len(rows) != n:
        raise SystemExit(f"diagnostic tasks: {len(rows)} of {n}, {len(clash)} shared with T0, the order or the pools")
    return {"rows": rows, "events": events, "cells": cells, "isolation": report, "removed": sorted(flagged),
            "project_group_shortfalls": short}


def write_diagnostic(got: dict) -> dict:
    out = OUT_DIR / f"{DIAG_NAME}.jsonl"
    if out.exists():
        raise SystemExit(f"{out.relative_to(REPO)} exists: diagnostic tasks are never rebuilt -- bump DIAG_NAME")
    rows = []
    for i, t in enumerate(got["rows"]):
        c = t.pop("_cell")
        t.pop("_pool", None)
        rows.append({**t, "stratum": c[0], "position": i, "slice": "diagnostic"})
    out.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "schema": DIAG_SCHEMA, "name": DIAG_NAME,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "python -m stages.teacher_modes.build_tasks diagnostic",
        "use": ("the L40S engineering check's inputs (configs/stages/stage-2/teacher_modes/l40s_check.json): their "
                "generations measure memory and throughput only. NEVER training, calibration or evaluation data; "
                "admission.py, build_tasks' order and the teacher session never read this file"),
        "n_tasks": len(rows), "cells": {f"{c[0]}/{c[1]}": k for c, k in sorted(got["cells"].items())},
        "by_cell": dict(sorted(Counter(f"{r['stratum']}/{r['source']}" for r in rows).items())),
        "not_covered": ("the external generators' cells (their construction needs an isolated export) and MBPP's "
                        "(exhausted by the order): their share goes to their stratum's other cells, and the code "
                        "stratum has none here"),
        "events": got["events"], "project_group_shortfalls": got["project_group_shortfalls"],
        "removed_before_isolation": len(got["removed"]),
        "seed_rules": {"project": f"R5 lo + {DIAG_BLOCK:,} + group index x {DIAG_SPAN:,} + i (the order's block is "
                                  f"lo + {PROJECT_BLOCK:,} + group index x 10,000 + i)",
                       "datasets": "build_supporting_tasks._order (its SALT), past every pooled and ordered task"},
        "disjoint_from": "T0, order_v1 and every pool, by task id and content; eval-set-1's held-out prompts and the "
                         "calibration prompts by prompt key (isolate)",
        "isolation": {"pass": got["isolation"].get("pass"), "violations": got["isolation"].get("violations")},
        "sha256": A.sha256_file(out),
        "license": "project-generated tasks; MATH (MIT) and SQuAD v2 (CC-BY-SA-4.0) train splits only",
        "authorizes": "nothing",
    }
    (OUT_DIR / f"{DIAG_NAME}.manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


def slice_of(i: int) -> tuple[str, int | None]:
    if i < V_TASKS:
        return "V", None
    if i < V_TASKS + E_TASKS:
        return "E", (i - V_TASKS) // WAVE + 1
    return "X", (i - V_TASKS - E_TASKS) // WAVE + 1


def write(got: dict, exports: list[Path]) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{NAME}.jsonl"
    if out.exists():
        raise SystemExit(f"{out.relative_to(REPO)} exists: a frozen order is never rebuilt -- bump NAME")
    rows = []
    for i, t in enumerate(got["order"]):
        s, w = slice_of(i)
        c = t.pop("_cell")
        t.pop("_pool", None)
        rows.append({**t, "stratum": c[0], "position": i, "slice": s, "wave": w})
    out.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    (OUT_DIR / f"{NAME}.isolation.json").write_text(json.dumps(got["isolation"], indent=1, default=str) + "\n")
    by_slice = defaultdict(Counter)
    for r in rows:
        by_slice[r["slice"]][f"{r['stratum']}/{r['source']}"] += 1
    t0n = sum(got["t0_cells"].values())
    shift = {}
    for s, cnt in by_slice.items():
        n = sum(cnt.values())
        strata = Counter()
        for k, v in cnt.items():
            strata[k.split("/")[0]] += v
        t0s = Counter()
        for (st, _), v in got["t0_cells"].items():
            t0s[st] += v
        shift[s] = {st: round(100 * (strata[st] / n - t0s[st] / t0n), 3) for st in sorted(t0s)}
    manifest = {
        "schema": SCHEMA, "name": NAME,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "python -m stages.teacher_modes.build_tasks build --exports " + " ".join(
            e.name for e in exports),
        "design": {"path": str(DESIGN.relative_to(REPO)), "section": "tasks"},
        "t0_lists": [{"path": l["path"], "sha256": l["sha256"]} for l in t0_lists()],
        "t0_cells": {f"{c[0]}/{c[1]}": k for c, k in sorted(got["t0_cells"].items())},
        "slices": {"V": {"positions": [0, V_TASKS], "use": "per-mode validation documents (diagnostic only)"},
                   "E": {"positions": [V_TASKS, V_TASKS + E_TASKS], "waves_of": WAVE,
                         "use": "the D4-scale extension, both modes, generated wave by wave after T0"},
                   "X": {"positions": [V_TASKS + E_TASKS, len(rows)], "waves_of": WAVE,
                         "use": "the compute-matching extension, H-N only"}},
        "n_tasks": len(rows), "by_slice": {s: dict(sorted(c.items())) for s, c in by_slice.items()},
        "stratum_share_shift_points_vs_t0": shift,
        "substitutions": got["events"],
        "need": {f"{c[0]}/{c[1]}": v for c, v in got["need"].items()},
        "new_tasks": got["new"], "removed_before_isolation": len(got["removed"]),
        "external_exports": [{"name": e.name, "sha256": A.sha256_file(e), "rows": sum(1 for _ in A.jsonl(e))}
                             for e in exports],
        "external_drops": got["external_drops"],
        "project_group_shortfalls": got["project_group_shortfalls"],
        "seed_rules": {"project": f"R5 lo + {PROJECT_BLOCK:,} + cell index x 10,000 + i",
                       "external": (f"external_export.entry_seed(seed base {SEED_BASE}, source, module, level); the "
                                    f"code stratum's supplement at {SUPPLEMENT_SEED_BASE}"),
                       "datasets": "build_supporting_tasks._order (its SALT), past every pooled task"},
        "sha256": A.sha256_file(out),
        "isolation_report": f"{NAME}.isolation.json",
        "license": "project-generated tasks; third-party generators' licenses as external_modules.json records; "
                   "MATH (MIT), MBPP (CC-BY-4.0), SQuAD v2 (CC-BY-SA-4.0) train splits only; no ARC-AGI material",
        "status": "FROZEN at $0 before any of its tasks is generated (d4_scale_design.json :: tasks.E)",
        "authorizes": "nothing",
    }
    (OUT_DIR / f"{NAME}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out-dir", required=True)
    b = sub.add_parser("build")
    b.add_argument("--exports", nargs="*", default=[])
    sub.add_parser("diagnostic")
    a = ap.parse_args(argv)
    if a.cmd == "diagnostic":
        m = write_diagnostic(diagnostic())
        print(json.dumps({k: m[k] for k in ("n_tasks", "by_cell", "sha256")}, indent=1))
        return 0
    if a.cmd == "plan":
        by = pools()
        t0c, t0 = t0_cells(by)
        need = needs(t0c, unused(by, t0))
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        wl = whitelists(t0, need)
        for n, entries in wl.items():
            (out / f"whitelist_n{n}.json").write_text(json.dumps(entries, indent=1) + "\n")
        print(json.dumps({"need": {f"{c[0]}/{c[1]}": v for c, v in need.items()},
                          "whitelists": {n: len(e) for n, e in wl.items()}}, indent=1))
        return 0
    m = write(build([Path(e) for e in a.exports]), [Path(e) for e in a.exports])
    print(json.dumps({k: m[k] for k in ("n_tasks", "new_tasks", "removed_before_isolation", "sha256",
                                        "stratum_share_shift_points_vs_t0")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
