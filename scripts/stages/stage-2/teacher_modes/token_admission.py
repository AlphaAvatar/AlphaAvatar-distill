#!/usr/bin/env python3
"""Research A v2's admission: sealed v1 and supplement records -> each mode's token-matched selection, the same-tasks
set, every pack. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.token_admission \\
        --v1-records <v1 scr>/store/generation --v1-identity <v1 study identity> \\
        --records <supplement scr>/store/generation --identity <this study's identity> \\
        --out artifacts/stages/stage-2/teacher_modes/packs_tm

The registered rules (token_matched_design.json :: scale, categories, selection, validation, packing):

1. VERDICTS. Every (task, mode) of both record sets through admission.py's frozen `admit` (verified correct,
   naturally terminated, protocol-valid, token-native documents under the mode's generation prompt; MBPP in the
   sandbox), then its isolation recheck (R1, R2, R9, then every rule) with this study's source catalogue.
2. SUPPLY. Per mode and cell, the admitted tasks outside V and V2, reuse first: T0 in its lists' order, then
   order_v1's positions (E, X), then order_v2's cell positions.
3. SELECTION, per mode and category: the next task comes from the cell furthest below its share of the
   category's running total (largest remainder on supervised tokens), and inside the cell from the family
   furthest below its target share of the cell's running total -- the mean of the two modes' admitted-supply
   family shares in that cell; a family that runs dry hands its share to the cell's others. The category stops
   at the first task that reaches its target. A cell that runs dry first is a SHORTFALL: reported, the
   selection refused, nothing substituted.
4. SAME TASKS: the tasks both selections hold, each mode with its own documents.
5. PACKS: each arm's documents `pack_isolated` at 8,192 (doc ids sorted, the protocol's order seed), with its
   mode's validation tail (V and V2 documents, at most 16 sequences), `write_pack`.

Nothing reads a student result, an eval-set-1 item, or a length or score beyond the frozen verdict.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.teacher_modes import admission as ADM  # noqa: E402
from stages.teacher_modes import build_supplement_tasks as BST  # noqa: E402

RECORD = REPO / "logs/stages/stage-2/teacher_modes/data/token_matched_admission.json"
ORDER_V2 = BST.ORDER
MODES = ADM.MODES
SCHEMA = "aadistill.stage2.teacher_modes.token_matched_admission/v1"
#: the student's parameter count (the 596M geometry) and the teacher's, for the FLOPs accounting only
STUDENT_PARAMS, TEACHER_PARAMS = 596_049_920, 4_022_468_096
EPOCHS, BATCH = 3, 2


def family_of(t: dict) -> str:
    return f"{t['generator_id']}/{t.get('family_id')}" if t.get("generator_id") else (t.get("family_id") or t["source"])


def load(v1_store: Path, v1_identity: str, store: Path | None, identity: str | None) -> dict:
    proto = ADM.protocol()
    tasks, t0, order1, stage = ADM.tasks_and_stages(proto)
    order2 = list(BST.read_jsonl(ORDER_V2))
    tasks.update({t["task_id"]: t for t in order2})
    recs = dict(ADM.records(Path(v1_store), v1_identity))
    if store is not None and Path(store).is_dir():
        recs.update(ADM.records(Path(store), identity))
    verdicts = {k: ADM.admit(tasks[k[0]], k[1], v, proto) for k, v in recs.items()}
    flagged, iso = ADM.isolation_recheck({k: v for k, v in verdicts.items() if v["admitted"]}, tasks,
                                         sources=BST.catalogue())
    for k in flagged:
        verdicts[k] = {**verdicts[k], "admitted": False, "docs": [], "isolation_flagged": True}
    rank = {t: (0, i) for i, t in enumerate(t0)}
    for t in order1:
        rank.setdefault(t["task_id"], ({"V": 9, "E": 1, "X": 2}[t["slice"]], int(t["position"])))
    for t in order2:
        rank[t["task_id"]] = (9 if t["slice"] == "V2" else 3, int(t["cell_position"]))
    val_ids = [t["task_id"] for t in order1 if t["slice"] == "V"] + [t["task_id"] for t in order2 if t["slice"] == "V2"]
    return {"proto": proto, "tasks": tasks, "verdicts": verdicts, "flagged": sorted(flagged), "isolation": iso,
            "rank": rank, "val_ids": val_ids, "recs": recs}


def cell_of(t: dict) -> str:
    return f"{t.get('stratum') or ADM.SP.stratum_of(t, A.read_json(A.CONFIG_DIR / 'corpus_spec.json'))}/{t['source']}"


def supply(got: dict, cells: set[str]) -> dict[str, dict[str, dict[str, list[tuple[str, int]]]]]:
    """mode -> cell -> family -> [(task, supervised tokens)] in reuse-first order."""
    out: dict = {m: defaultdict(lambda: defaultdict(list)) for m in MODES}
    for (tid, mode), v in sorted(got["verdicts"].items(), key=lambda kv: (got["rank"].get(kv[0][0], (8, 0)), kv[0])):
        r = got["rank"].get(tid, (8, 0))
        if not v["admitted"] or r[0] == 9:
            continue
        t = got["tasks"][tid]
        c = cell_of(t)
        if c in cells:
            out[mode][c][family_of(t)].append((tid, sum(len(d.output_ids) for d in v["docs"])))
    return out


def family_targets(sup: dict, cells: set[str]) -> dict[str, dict[str, float]]:
    """Per cell: the mean of the two modes' admitted-supply family shares (by supervised tokens)."""
    out = {}
    for c in cells:
        shares = []
        for m in MODES:
            tot = sum(n for f in sup[m][c].values() for _, n in f)
            shares.append({f: sum(n for _, n in v) / tot for f, v in sup[m][c].items()} if tot else {})
        fams = sorted(set().union(*shares))
        out[c] = {f: sum(s.get(f, 0.0) for s in shares) / len(MODES) for f in fams}
    return out


def select(sup_m: dict, quotas: dict[str, float], targets: dict, category_total: float) -> dict:
    share = {c: q / category_total for c, q in quotas.items()}
    pos = {c: Counter() for c in quotas}
    taken = {c: Counter() for c in quotas}
    tok = Counter()
    chosen, total, shortfall = [], 0, []
    live = {c for c in quotas if any(sup_m[c].values())}
    shortfall += sorted(set(quotas) - live)
    while total < category_total and not shortfall:
        c = max(sorted(live), key=lambda d: share[d] * total - tok[d])
        fams = [f for f in sup_m[c] if pos[c][f] < len(sup_m[c][f])]
        if not fams:
            shortfall.append(c)
            break
        tgt = targets[c]
        f = max(sorted(fams), key=lambda g: tgt.get(g, 0.0) * tok[c] - taken[c][g]
                if sum(tgt.get(h, 0.0) for h in fams) else 0.0)
        tid, n = sup_m[c][f][pos[c][f]]
        pos[c][f] += 1
        taken[c][f] += n
        tok[c] += n
        total += n
        chosen.append((tid, c, f, n))
    return {"tasks": chosen, "tokens": total, "by_cell": dict(tok), "by_family": {c: dict(v) for c, v in taken.items()},
            "shortfall": shortfall}


def tvd(a: dict[str, float], b: dict[str, float]) -> float:
    ta, tb = sum(a.values()) or 1, sum(b.values()) or 1
    return round(0.5 * sum(abs(a.get(k, 0) / ta - b.get(k, 0) / tb) for k in set(a) | set(b)), 4)


def decide(got: dict, design: dict | None = None) -> dict:
    d = design or BST.design()
    plan = json.loads(BST.PLAN.read_text())
    q = BST.quotas(d, mbpp_cap=plan["mbpp_cap"])
    cells = {"/".join(c) for c in q}
    sup = supply(got, cells)
    targets = family_targets(sup, cells)
    sel = {}
    for m in MODES:
        sel[m] = {}
        for cat, total in d["scale"]["categories"].items():
            cq = {"/".join(c): v for c, v in q.items() if BST.category_of(c) == cat}
            sel[m][cat] = select({c: sup[m][c] for c in cq}, cq, targets, float(total))
    ids = {m: [t for cat in sel[m].values() for t, *_ in cat["tasks"]] for m in MODES}
    same = sorted(set(ids["H-T"]) & set(ids["H-N"]))
    val = {m: sorted({dd.doc_id for t in got["val_ids"] for dd in (got["verdicts"].get((t, m)) or {}).get("docs", [])})
           for m in MODES}
    arms = {"H-T-TM": ("H-T", ids["H-T"]), "H-N-TM": ("H-N", ids["H-N"]), "H-T-ST": ("H-T", same), "H-N-ST": ("H-N", same)}
    return {"quotas": {"/".join(c): v for c, v in q.items()}, "targets": targets, "selection": sel, "arms": arms,
            "same_tasks": same, "val": val, "supply": sup}


def accounting(got: dict, dec: dict, packs: dict | None) -> dict:
    out = {}
    for name, (mode, ids) in dec["arms"].items():
        docs = [dd for t in ids for dd in got["verdicts"][(t, mode)]["docs"]]
        cat, cell, fam, axis, src = Counter(), Counter(), Counter(), Counter(), Counter()
        reuse = Counter()
        for t in ids:
            n = sum(len(dd.output_ids) for dd in got["verdicts"][(t, mode)]["docs"])
            task = got["tasks"][t]
            c = cell_of(task)
            cat[BST.category_of(tuple(c.split("/", 1)))] += n
            cell[c] += n
            fam[f"{c}/{family_of(task)}"] += n
            axis[c.split("/", 1)[0]] += n
            src[task["source"]] += n
            reuse["supplement" if got["rank"][t][0] == 3 else "reused_v1"] += n
        sup = sum(len(dd.output_ids) for dd in docs)
        ctx = sum(len(dd) for dd in docs)
        row = {"mode": mode, "tasks": len(ids), "documents": len(docs), "supervised_tokens": sup,
               "non_padding_tokens": ctx, "by_category": dict(cat), "by_axis": dict(axis), "by_cell": dict(cell),
               "by_source": dict(src), "by_family": dict(fam), "reused_vs_new_tokens": dict(reuse),
               "category_share_points": {k: round(100 * v / sup, 3) for k, v in cat.items()} if sup else {}}
        if packs and name in packs:
            p = packs[name]
            blocks = int(p["n_blocks"])
            steps = math.ceil(EPOCHS * blocks / BATCH)
            positions = steps * BATCH * 8192            #: the trainer forwards every position of a sequence
            per = per_position_flops()
            row.update(sequences=blocks, padding_share=round(1 - ctx / (blocks * 8192), 4),
                       optimizer_steps_at_3_epochs_batch_2=steps, supervised_token_exposure=EPOCHS * sup,
                       non_padding_token_exposure=EPOCHS * ctx, positions_computed=positions,
                       training_flops=float(f"{positions * per['training_total']:.4g}"), pack=p)
        out[name] = row
    return out


def per_position_flops() -> dict:
    """H1-PA's accounting (h1pa_audit.fwd_flops_per_position): the student 4 x forward (forward, backward, the
    recomputed forward of gradient checkpointing) plus the teacher's forward, per computed position."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from stages.reasoning_data.h1pa_audit import fwd_flops_per_position
    proto = ADM.protocol()
    student = json.loads((REPO / "artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint/config.json").read_text())
    teacher = json.loads((Path(HF_HUB_CACHE) / "models--Qwen--Qwen3-4B" / "snapshots" / proto["teacher"]["revision"]
                          / "config.json").read_text())
    s, t = fwd_flops_per_position(student, STUDENT_PARAMS), fwd_flops_per_position(teacher, TEACHER_PARAMS)
    return {"student_forward": s["total"], "teacher_forward": t["total"], "training_total": 4 * s["total"] + t["total"]}


def build_packs(got: dict, dec: dict, out: Path) -> dict:
    from aadistill.data.token_documents import PackedDocuments, pack_isolated, write_pack
    proto = got["proto"]
    seq, seed, pad = (int(proto["stopping"]["sequence_length"]), int(proto["stopping"]["pack_order_seed"]),
                      int(proto["teacher"]["pad_id"]))
    dtype = {}
    for (tid, mode), v in got["verdicts"].items():
        for dd in v.get("docs", []):
            dtype[dd.doc_id] = cell_of(got["tasks"][tid]).split("/", 1)[0]
    bydoc = {dd.doc_id: dd for v in got["verdicts"].values() for dd in v.get("docs", [])}
    val = {}
    for m, ids in dec["val"].items():
        w = pack_isolated([bydoc[k] for k in ids], seq, pad, order_seed=seed)
        n = min(ADM.VAL_SEQUENCES, w.input_ids.shape[0])
        val[m] = PackedDocuments(w.input_ids[:n], w.target_mask[:n], w.content_mask[:n], w.position_ids[:n],
                                 w.members[:n], [], w.strategy)
    res = {}
    for name, (mode, ids) in dec["arms"].items():
        docs = sorted((dd for t in ids for dd in got["verdicts"][(t, mode)]["docs"]), key=lambda dd: dd.doc_id)
        tr = pack_isolated(docs, seq, pad, order_seed=seed)
        res[name] = write_pack(tr, val[mode], Path(out) / name, data_type=dtype,
                               extra={"arm": name, "mode": mode, "context_strategy": "document_isolated"})
    return res


def refusals(got: dict, cells: set[str]) -> dict:
    out = defaultdict(Counter)
    for (tid, mode), v in got["verdicts"].items():
        c = cell_of(got["tasks"][tid])
        if v["admitted"]:
            continue
        why = "isolation recheck" if v.get("isolation_flagged") else (
            v["candidates"][-1]["reasons"][0].split(":")[0] if v["candidates"] and v["candidates"][-1]["reasons"] else "none")
        out[f"{mode}|{'design cell' if c in cells else 'outside the design'}"][why] += 1
    return {k: dict(v.most_common()) for k, v in sorted(out.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v1-records", required=True)
    ap.add_argument("--v1-identity", required=True)
    ap.add_argument("--records", default=None)
    ap.add_argument("--identity", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=str(RECORD))
    ap.add_argument("--no-packs", action="store_true", help="the selection and accounting only (a supply check)")
    a = ap.parse_args(argv)
    got = load(Path(a.v1_records), a.v1_identity, Path(a.records) if a.records else None, a.identity)
    dec = decide(got)
    short = {m: {cat: s["shortfall"] for cat, s in dec["selection"][m].items() if s["shortfall"]} for m in MODES}
    short = {m: v for m, v in short.items() if v}
    packs = None if (a.no_packs or short) else build_packs(got, dec, Path(a.out))
    acc = accounting(got, dec, packs)
    cells = set(dec["quotas"])
    fam_tvd = {}
    for c in sorted(cells):
        fa = {k: v for k, v in acc["H-T-TM"]["by_family"].items() if k.startswith(c + "/")}
        fb = {k: v for k, v in acc["H-N-TM"]["by_family"].items() if k.startswith(c + "/")}
        fam_tvd[c] = tvd(fa, fb)
    targets = BST.design()["scale"]
    checks = {}
    for m, arm in (("H-T", "H-T-TM"), ("H-N", "H-N-TM")):
        tot = acc[arm]["supervised_tokens"]
        checks[arm] = {"total": tot, "over_target_fraction": round(tot / targets["per_mode_total"] - 1, 6),
                       "category_deviation_points": {k: round(100 * (acc[arm]["by_category"].get(k, 0) / tot - v / targets["per_mode_total"]), 4)
                                                     for k, v in targets["categories"].items()} if tot else {}}
    tol = targets["tolerance"]
    ok = not short and all(0 <= c["over_target_fraction"] <= tol["total_over_target_max_fraction"] and
                           all(abs(x) <= tol["category_share_deviation_max_points"] for x in c["category_deviation_points"].values())
                           for c in checks.values())
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "terminal": "ADMITTED" if ok and packs else ("SHORTFALL" if short else "CHECK_ONLY" if a.no_packs else "OUT_OF_TOLERANCE"),
           "design_sha256": A.sha256_file(BST.DESIGN), "plan_sha256": A.sha256_file(BST.PLAN),
           "order_v2_sha256": A.sha256_file(ORDER_V2),
           "records": {"v1": {"dir": a.v1_records, "identity": a.v1_identity},
                       "supplement": {"dir": a.records, "identity": a.identity,
                                      "complete_groups": sorted(p.name for p in Path(a.records).glob("*.complete"))
                                      if a.records and Path(a.records).is_dir() else []}},
           "quotas": dec["quotas"], "family_targets": dec["targets"], "shortfall": short, "tolerance_checks": checks,
           "family_tvd_between_modes": fam_tvd, "arms": acc,
           "same_tasks": {"tasks": len(dec["same_tasks"])},
           "validation": {m: {"documents": len(v)} for m, v in dec["val"].items()},
           "selection": {m: {cat: [t for t, *_ in s["tasks"]] for cat, s in dec["selection"][m].items()} for m in MODES},
           "isolation_recheck": {**got["isolation"], "flagged": got["flagged"]},
           "refusals": refusals(got, cells), "authorizes": "nothing"}
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"terminal": doc["terminal"], "shortfall": short, "checks": checks,
                      "arms": {k: {x: v.get(x) for x in ("tasks", "documents", "supervised_tokens", "sequences")}
                               for k, v in acc.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
