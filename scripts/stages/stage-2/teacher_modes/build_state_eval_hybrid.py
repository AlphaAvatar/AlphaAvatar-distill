#!/usr/bin/env python3
"""The hybrid full search's state-evaluation prompts and the calibration's code cell, with the isolation proof. $0.

    # 1. the code type's two exports, in the isolated third-party env (one seed base each)
    PYTHONHASHSEED=0 <rgenv>/bin/python -I scripts/stages/stage-2/reasoning_data/external_export.py \\
        --whitelist <scratch>/whitelist_code.json --n 8 --seed-base 2026310000 --out <cal_code.jsonl>
    PYTHONHASHSEED=0 <rgenv>/bin/python -I scripts/stages/stage-2/reasoning_data/external_export.py \\
        --whitelist <scratch>/whitelist_code.json --n 16 --seed-base 2026320000 --out <se_code.jsonl>
    # 2. the prompt lists and the isolation proof
    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.build_state_eval_hybrid build \\
        --calibration-code-export <cal_code.jsonl> --state-eval-code-export <se_code.jsonl>

Both sets read ONE type table (configs/stages/stage-2/teacher_modes/full_search_types.json): 13 types over five
domains, each in both modes. Nothing here generates: the trajectories are the hybrid's own, generated later in
each mode under its official template and sampling, only on the maintainer's approval.

STATE EVALUATION (`state_eval_hybrid@v1`): its own frozen quota -- ``SE_POSITIONS`` scored positions per (type,
mode) -- whole units, a mode taking its type's queue in order until its positions reach the quota. The queue per
type is sized from the hybrid's measured positions per unit (the calibration records) with margin ``SE_MARGIN``.
Sources, every one new to the project's calibration and training:
* the eight fluid types: new instances of the same @v1 generators, calibration-split families (R9's refused out),
  difficulties L1/L2/L3 in turn, seeds ``SE_SEED_BLOCK + group x 10,000 + i`` inside the calibration range
  [2,000,000, 3,000,000) (Stage-0's calibration_v2 used 2,000,000-2,200,024);
* gsm8k, rag_evidence, multihop_qa: their dataset's train rows past every row the training order or a calibration
  set holds, rendered as corpus_v2 rendered them; one question per SQuAD paragraph;
* openmath: base_pool_v1's openmath units that no calibration profile and no training set holds;
* code: reasoning_core's code modules at their own export seed base. These modules draw instance k of an entry
  from seed (entry seed + k), so two bases closer than an export's length repeat each other's instances (the
  2026102900-2026103120 bases did): the code bases here are 2026310000 and 2026320000, far from every earlier one,
  and every set is deduplicated by content against every earlier set anyway.

CALIBRATION'S CODE CELL: the type table adds code to the supporting domain, which D4's cells lacked: its prompts
come from another export seed base, sized for both profiles' halves of the cell quota.

THE ISOLATION PROOF: calibration vs state evaluation, each vs A1/A2's training (T0, the pools, order_v1, order_v2)
and C1's (D4-Augmented's E1 sessions and fluid units), each vs eval-set-1 (R1-R4, R7, R9 through
`isolation_check.check_pool`), by task id, content, prompt key and 13-gram containment (R2's protocol: n 13,
threshold 0.5, template n-grams in more than 5 prompts dropped), with each set's generator, family, seed range and
source lineage tabulated.
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
for _p in ("src", "scripts", "scripts/stages/stage-0/calibration_volume"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.reasoning_data import build_supporting_tasks as BS  # noqa: E402
from stages.reasoning_data import build_training_tasks as BT  # noqa: E402
from stages.reasoning_data import external_tasks as ET  # noqa: E402
from stages.teacher_modes import build_calibration_supplement as BCS  # noqa: E402
from stages.teacher_modes import build_supplement_tasks as BST  # noqa: E402
from stages.teacher_modes import build_tasks as B1  # noqa: E402
from stages.teacher_modes import hybrid_calibration as CAL  # noqa: E402

TYPES = REPO / "configs/stages/stage-2/teacher_modes/full_search_types.json"
SE_OUT = REPO / "data/stages/stage-2/teacher_modes/tasks/state_eval_hybrid_v1.tasks.jsonl"
CAL_CODE_OUT = REPO / "data/stages/stage-2/teacher_modes/tasks/calibration_code_v1.jsonl"
PROOF = REPO / "logs/stages/stage-2/teacher_modes/audit/full_search_isolation.json"
MODES = ("H-T", "H-N")
SE_POSITIONS = 12_000
SE_MARGIN, CAL_MARGIN = 1.4, 1.3
#: dataset candidates drawn per planned unit before the 13-gram screens
CANDIDATES = 1.35
SE_SEED_BLOCK, SPAN = 2_500_000, 10_000
CODE_MODULES = ("controlled_code_execution", "dynamic_programming")
#: the code type's planning yield before any calibration record of it exists: (natural-termination share, scored
#: positions per naturally terminated unit), from the sealed training records of reasoning_core's code modules
CODE_YIELD = {"H-T": (0.518, 2179), "H-N": (0.937, 819)}
NGRAM, THRESHOLD, MAX_DF = 13, 0.5, 5


def types() -> list[dict]:
    return json.loads(TYPES.read_text())["types"]


def yields() -> dict[tuple[str, str], tuple[float, float]]:
    """(subtype, mode) -> (natural-termination share, positions per naturally terminated unit) from the hybrid's
    calibration records (calibration_coverage.json :: audit), the code type's from training records."""
    audit = json.loads(BCS.AUDIT.read_text())["audit"]
    out = {}
    for k, v in audit.items():
        prof, mode, cell = k.split("|")
        sub = cell.split("/", 1)[1]
        if prof == "base" and v["positions_per_unit_v1"]:
            out[(sub, mode)] = (1.0, float(v["positions_per_unit_v1"]))
    for m, y in CODE_YIELD.items():
        out[("code", m)] = y
    return out


def units_needed(positions: float, y: tuple[float, float], margin: float) -> int:
    return math.ceil(positions / max(y[0] * y[1], 1.0) * margin)


def prompt_key(t: dict) -> str:
    from shared import eval_sets as ES
    return ES.prompt_key(ES.prompt_text(t))


def text(t: dict) -> str:
    from shared import eval_sets as ES
    return ES.prompt_text(t)


# ---- what no new unit may repeat --------------------------------------------------------------------------------

def training_sets() -> dict[str, list[dict]]:
    by = B1.pools()
    sets = {"a1a2_pools_t0": list(by.values()),
            "a1a2_order_v1": list(A.jsonl(B1.OUT_DIR / "order_v1.jsonl")),
            "a1a2_order_v2": list(BST.read_jsonl(BST.ORDER)),
            "diagnostic_v1": list(A.jsonl(B1.OUT_DIR / "diagnostic_v1.jsonl"))}
    h = json.loads((REPO / "configs/stages/stage-2/reasoning_data/h1pa.json").read_text())
    pack = REPO / h["datasets"]["e1_2p96M_plus_fluid"]["pack"]
    want = {s["session_id"] for row in A.jsonl(pack / "audit.jsonl") for s in row.get("sessions", [])}
    cv2 = [{"messages": [m for m in s["messages"] if m["role"] in ("system", "user")], "task_id": s["id"]}
           for s in A.corpus_v2_sessions() if s["id"] in want]
    fluid = [{"messages": [m for m in s["messages"] if m["role"] in ("system", "user")][:2], "task_id": s["id"]}
             for s in A.jsonl(REPO / "artifacts/stages/stage-2/reasoning_data/packs/D4A_fluid_whole_sessions.jsonl")
             if s["id"] in want]
    sets["c1_d4_augmented"] = cv2 + fluid
    return sets


def calibration_sets() -> dict[str, list[dict]]:
    v1 = list(A.jsonl(BCS.V1_INPUTS))
    v2 = list(A.jsonl(BCS.OUT))
    return {"hybrid_calibration_v1": v1, "calibration_v2": v2}


# ---- new units ----------------------------------------------------------------------------------------------------

def fluid_units(n_by_subtype: dict[str, int], taken: set[str]) -> tuple[list[dict], dict]:
    from stages.calibration_volume.tasks import registry as R
    from stages.calibration_volume.tasks.common import Unsatisfiable, build_instance
    from stages.reasoning_data import isolation_check as IC
    gens = {g.generator_id: g for g in R.GENERATORS}
    out, short = [], {}
    for j, (sub, n) in enumerate(sorted(n_by_subtype.items())):
        gen = gens[sub]
        fams = [f for f in R.family_split(gen)["calibration"]
                if f"{sub}/{f}" not in BT.MECHANISM_EXCLUDED and not r9_refused(sub, f)]
        made = 0
        for i in range(SPAN):
            if made >= n:
                break
            seed = SE_SEED_BLOCK + j * SPAN + i
            try:
                t = build_instance(gen, gen.family(fams[i % len(fams)]), ("L1", "L2", "L3")[i % 3], seed)
            except (Unsatisfiable, IndexError, ValueError):
                continue
            if t["content_sha256"] in taken:
                continue
            taken.add(t["content_sha256"])
            t.update(role="state_evaluation", unit_id=t["task_id"], source="stage0_generators_calibration_families",
                     subtype=sub, domain=t.get("axis"))
            out.append(t)
            made += 1
        if made < n:
            short[sub] = {"wanted": n, "made": made}
    return out, short


def dataset_units(name: str, n: int, start: int, block_keys: set[str], one_per_context: bool = False) -> list[dict]:
    rows = {"gsm8k": BST.gsm8k_tasks, "rag_evidence": BST.squad_answerable_tasks, "multihop_qa": BST.hotpot_tasks}[name]()[0]
    out, ctx = [], set()
    for t in rows:
        if t["row_index"] < start or prompt_key(t) in block_keys:
            continue
        if one_per_context:
            c = t.get("context_sha256")
            if c in ctx:
                continue
            ctx.add(c)
        out.append({**t, "role": "state_evaluation", "subtype": name, "domain": "supporting"})
        if len(out) >= n:
            break
    return out


def openmath_units(n: int, block_keys: set[str]) -> list[dict]:
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoTokenizer
    m, rev = CAL.SOURCE_TOKENIZER
    tok = AutoTokenizer.from_pretrained(Path(HF_HUB_CACHE) / f"models--{m.replace('/', '--')}" / "snapshots" / rev)
    d = BCS.design()
    units, order = BCS.pool_order(d)
    cal_units = {json.loads(l)["calibration"]["unit_id"] for l in BCS.OUT.read_text().splitlines()}
    cal_units |= {it["unit_id"] for prof in CAL.PROFILES for it in CAL.items(prof)}
    out = []
    for uid in order[("supporting", "openmath", "*")]:
        if uid in cal_units:
            continue
        it = units[uid]["items"][0]
        msgs = CAL.prompt_messages(it, tok)
        t = {"task_id": "se_" + hashlib.sha256(f"base_pool_v1|{uid}".encode()).hexdigest()[:20],
             "messages": [x for x in msgs if x["role"] != "system"], "source": "corpus_v2_openmath",
             "role": "state_evaluation", "subtype": "openmath", "domain": "supporting", "pool_unit_id": uid}
        if prompt_key(t) in block_keys:
            continue
        out.append(t)
        if len(out) >= n:
            break
    return out


def code_units(export: Path, n: int, taken: set[str], role: str) -> list[dict]:
    got = ET.build([export])["tasks"]
    out = []
    for t in got:
        if t["module"] not in CODE_MODULES or t["content_sha256"] in taken:
            continue
        taken.add(t["content_sha256"])
        out.append({**t, "role": role, "subtype": "code", "domain": "supporting"})
        if len(out) >= n:
            break
    return out


# ---- the isolation proof ------------------------------------------------------------------------------------------

def r9_refused(generator_id: str | None, family_id: str | None) -> bool:
    """R9 by the family's NAME: its table registers the v2 roster's ids (string_induction@v2/str.caesar ...), and the
    v1 generator of the same name carries the same rule under the same family id."""
    from stages.reasoning_data import mechanism_check as MC
    global _R9
    if "_R9" not in globals():
        _R9 = {(e.split("/", 1)[0].split("@")[0], e.split("/", 1)[1]) for e in MC.review()["equivalent_training_families"]}
    return bool(generator_id and family_id) and (str(generator_id).split("@")[0], family_id) in _R9


SOURCE_KEY = {"gsm8k_train": "gsm8k", "squad_v2_train_answerable": "rag_evidence", "hotpot_qa_train": "multihop_qa",
              "corpus_v2_openmath": "openmath", "gsm8k": "gsm8k", "squad_v2": "rag_evidence", "hotpot": "multihop_qa",
              "hotpot_qa": "multihop_qa", "openmath": "openmath", "openmath_instruct_2": "openmath", "mbpp": "code"}


def source_key(t: dict) -> str:
    """One key per type for the 13-gram screens: the generator's name, the dataset, the code modules."""
    cal = t.get("calibration") or {}
    if cal.get("subtype"):
        return str(cal["subtype"]).split("@")[0]
    if t.get("generator_id"):
        return str(t["generator_id"]).split("@")[0].split(":")[0]
    if t.get("module") in CODE_MODULES or t.get("subtype") == "code":
        return "code"
    if t.get("subtype"):
        return str(t["subtype"]).split("@")[0]
    if t.get("source") in SOURCE_KEY:
        return SOURCE_KEY[t["source"]]
    prefix = str(t.get("task_id", "")).split("-")[0]
    return SOURCE_KEY.get(prefix, str(t.get("source")))


def overlap(a: list[dict], b: list[dict]) -> dict:
    ka, kb = {prompt_key(t) for t in a}, {prompt_key(t) for t in b}
    ia, ib = {t.get("task_id") for t in a}, {t.get("task_id") for t in b}
    ca = {t.get("content_sha256") for t in a if t.get("content_sha256")}
    cb = {t.get("content_sha256") for t in b if t.get("content_sha256")}
    return {"prompt_keys": len(ka & kb), "task_ids": len(ia & ib - {None}), "content": len(ca & cb)}


def near(a: list[dict], b: list[dict]) -> dict:
    from aadistill.evaluation import isolation as ISO
    index = ISO.ngram_index([text(t) for t in b], NGRAM, max_document_frequency=MAX_DF)
    rep = ISO.ngram_containment([(t["task_id"], text(t)) for t in a], index, NGRAM, threshold=THRESHOLD)
    return {k: rep[k] for k in ("n", "threshold", "checked", "with_any_shared_ngram", "flagged")} | {
        "flagged_ids": [r["id"] for r in rep["flagged_items"]]}


EVAL_RULES = ("R1_held_out_keys", "R2_near_duplicate", "R3_rule_family", "R4_eval_only_generators", "R7_arc_agi")


def eval_rules(rep: dict, rows: list[dict]) -> dict:
    """The rules that isolate a set from eval-set-1, each with its own verdict (R0, R5, R6 are training rules), and R9
    by family name."""
    out = {k: {"pass": bool((rep["results"].get(k) or {}).get("pass", True))} for k in EVAL_RULES if k in rep["results"]}
    hits = Counter(f"{t.get('generator_id')}/{t.get('family_id')}" for t in rows if r9_refused(t.get("generator_id"), t.get("family_id")))
    out["R9_mechanism_equivalence_by_family_name"] = {"pass": not hits, "items": dict(hits)}
    out["R1_held_out_keys"]["overlap"] = (rep["results"].get("R1_held_out_keys") or {}).get("held_out_overlap")
    return out


def lineage(rows: list[dict]) -> dict:
    seeds = [int(t["seed"]) for t in rows if t.get("seed") is not None and str(t.get("seed")).isdigit()]
    return {"by_subtype": dict(Counter(t.get("subtype") or t.get("calibration", {}).get("subtype") for t in rows)),
            "generators": sorted({t.get("generator_id") or t.get("module") or t.get("source") for t in rows} - {None}),
            "families": len({(t.get("generator_id"), t.get("family_id")) for t in rows if t.get("family_id")}),
            "seed_range": [min(seeds), max(seeds)] if seeds else None,
            "sources": dict(Counter(t.get("source") for t in rows))}


def by_source(rows: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for t in rows:
        g[source_key(t)].append(t)
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--calibration-code-export", required=True)
    b.add_argument("--state-eval-code-export", required=True)
    a = ap.parse_args(argv)
    for out in (SE_OUT, CAL_CODE_OUT):
        if out.exists():
            raise SystemExit(f"{out.relative_to(REPO)} exists: a frozen prompt list is never rebuilt")
    y = yields()
    ts = types()
    train, cal = training_sets(), calibration_sets()
    block = {prompt_key(t) for rows in (*train.values(), *cal.values()) for t in rows}
    taken = {t.get("content_sha256") for rows in (*train.values(), *cal.values()) for t in rows} - {None}
    taken |= BT.taken_instances()
    # calibration's code cell: both profiles' halves of the cell quota (D4's supporting weight over five types)
    d = BCS.design()
    half = {p: 4 * float(d["reference_uesp"]) * d["profiles"][p]["domain_weights"]["supporting"] / 5 / 2 for p in ("base", "rh")}
    n_cal_code = max(units_needed(max(half.values()), y[("code", m)], CAL_MARGIN) for m in MODES)
    cal_code = code_units(Path(a.calibration_code_export), n_cal_code, taken, "calibration")
    block |= {prompt_key(t) for t in cal_code}
    # the state evaluation's queues
    need = {t["subtype"]: max(units_needed(SE_POSITIONS, y[(t["subtype"], m)], SE_MARGIN) for m in MODES) for t in ts}
    fluid, short = fluid_units({s: n for s, n in need.items() if "@" in s}, taken)
    rows = list(fluid)
    #: past the last row order_v2 holds (every earlier row is a training task or was refused from one)
    last = {}
    for t in train["a1a2_order_v2"]:
        if "row_index" in t:
            last[t["source"]] = max(last.get(t["source"], 0), int(t["row_index"]))
    starts = {"gsm8k": last["gsm8k_train"] + 1, "rag_evidence": last["squad_v2_train_answerable"] + 1,
              "multihop_qa": last["hotpot_qa_train"] + 1}
    for name in starts:                     #: candidates over the need: the 13-gram screens drop shared paragraphs
        rows += dataset_units(name, math.ceil(need[name] * CANDIDATES), starts[name], block,
                              one_per_context=name == "rag_evidence")
    rows += openmath_units(need["openmath"], block)
    rows += code_units(Path(a.state_eval_code_export), need["code"], taken, "state_evaluation")
    # the proof
    from stages.reasoning_data import isolation_check as IC
    cat = BST.catalogue()
    se_items = [{**t, "unit_id": t["task_id"]} for t in rows]
    rep = IC.check_pool(se_items, sources=cat)
    flagged = set(rep["results"]["R2_near_duplicate"]["flagged_ids"])
    flagged |= {t["task_id"] for t in rows if r9_refused(t.get("generator_id"), t.get("family_id"))}
    cal_all = cal["hybrid_calibration_v1"] + cal["calibration_v2"] + cal_code
    near_cal, near_train = {}, {}
    se_src, cal_src = by_source(rows), by_source(cal_all)
    train_all = [t for r in train.values() for t in r]
    tr_src = by_source(train_all)
    for s, group in sorted(se_src.items()):
        near_cal[s] = near(group, cal_src.get(s, [])) if cal_src.get(s) else {"checked": 0, "flagged": 0, "flagged_ids": []}
        near_train[s] = near(group, tr_src.get(s, [])) if tr_src.get(s) else {"checked": 0, "flagged": 0, "flagged_ids": []}
        flagged |= set(near_cal[s]["flagged_ids"]) | set(near_train[s]["flagged_ids"])
    kept, n_kept = [], Counter()
    for t in rows:                          #: each type's queue cut at its planned size, in its source order
        if t["task_id"] in flagged or n_kept[t["subtype"]] >= need[t["subtype"]]:
            continue
        n_kept[t["subtype"]] += 1
        kept.append(t)
    se_out = []
    for k, t in enumerate(sorted(kept, key=lambda t: (t["subtype"], rows.index(t)))):
        se_out.append({**t, "position": k})
    pos = Counter()
    for t in se_out:
        t["queue_position"] = pos[t["subtype"]]
        pos[t["subtype"]] += 1
    SE_OUT.write_text("".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in se_out))
    CAL_CODE_OUT.write_text("".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in cal_code))
    final = IC.check_pool([{**t, "unit_id": t["task_id"]} for t in se_out], sources=cat)
    cal_rep = IC.check_pool([{**t, "unit_id": t["task_id"]} for t in cal_code], sources=cat)
    proof = {"schema": "aadistill.stage2.teacher_modes.full_search_isolation/v1",
             "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "types": {"path": str(TYPES.relative_to(REPO)), "sha256": A.sha256_file(TYPES)},
             "state_evaluation": {"path": str(SE_OUT.relative_to(REPO)), "sha256": A.sha256_file(SE_OUT), "units": len(se_out),
                                  "queue_by_type": dict(pos), "positions_per_type_and_mode": SE_POSITIONS,
                                  "planned_units": need, "fluid_shortfalls": short, "lineage": lineage(se_out)},
             "calibration_code": {"path": str(CAL_CODE_OUT.relative_to(REPO)), "sha256": A.sha256_file(CAL_CODE_OUT),
                                  "units": len(cal_code), "lineage": lineage(cal_code)},
             "calibration_sets": {k: {"units": len(v), "lineage": lineage(v)} for k, v in cal.items()},
             "removed": {"R2_vs_eval_set_1": len(rep["results"]["R2_near_duplicate"]["flagged_ids"]),
                         "R9_refused_family": sum(r9_refused(t.get("generator_id"), t.get("family_id")) for t in rows),
                         "near_copy_of_calibration": sum(v["flagged"] for v in near_cal.values()),
                         "near_copy_of_training": sum(v["flagged"] for v in near_train.values())},
             "pairs": {"state_eval_vs_calibration": {"exact": overlap(se_out, cal_all), "ngram_by_type": near_cal},
                       "state_eval_vs_training": {"exact": {k: overlap(se_out, v) for k, v in train.items()},
                                                  "ngram_by_type": near_train},
                       "calibration_vs_training": {"exact": {f"{ck}|{tk}": overlap(cv, tv) for ck, cv in
                                                             {**cal, "calibration_code": cal_code}.items()
                                                             for tk, tv in train.items()}},
                       "state_eval_vs_eval_set_1": eval_rules(final, se_out),
                       "calibration_vs_eval_set_1": {
                           "calibration_code": eval_rules(cal_rep, cal_code),
                           "hybrid_calibration_v1_and_calibration_v2": eval_rules(
                               IC.check_pool([{**t, "unit_id": t["task_id"], "generator_id": t.get("generator_id"),
                                               "family_id": t.get("family_id")} for t in cal_all], sources=cat), cal_all)}},
             "_": ("calibration and state evaluation share types, generators and rule families BY DESIGN (one type table); "
                   "they share no task id, content, prompt key or 13-gram near-copy. R5 and R6 are training rules: these "
                   "sets sit in the calibration seed range [2,000,000, 3,000,000), disjoint from training's [3,000,000, "
                   "4,000,000)"),
             "authorizes": "nothing"}
    PROOF.write_text(json.dumps(proof, indent=1, default=str) + "\n")
    print(json.dumps({"se": proof["state_evaluation"]["queue_by_type"], "cal_code": len(cal_code), "removed": proof["removed"],
                      "se_vs_cal": proof["pairs"]["state_eval_vs_calibration"]["exact"],
                      "eval_set_1": {k: v["pass"] for k, v in proof["pairs"]["state_eval_vs_eval_set_1"].items()}},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
