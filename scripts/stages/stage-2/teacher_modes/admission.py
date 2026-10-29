#!/usr/bin/env python3
"""Research A's admission: secured teacher records -> I, the exact stopping rule, X, every arm's pack. $0, dev box.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.admission \\
        --records <scr>/store/generation --identity <study identity> --out artifacts/stages/stage-2/teacher_modes/packs

The registered rules (d4_scale_design.json :: admission, tasks; the
preregistration's data_budget_rules), every verifier the frozen H1 contract
has -- MBPP's sandbox included, which is why this runs here and not on the pod:

1. INPUTS. Every sealed shard re-hashed against its marker, every record of
   the study identity (`generation_driver.sealed`), one record per (task,
   mode, candidate).
2. ADMISSION per (task, mode): the lowest candidate that
   `generation_adapter.candidate` accepts (verified correct, naturally
   terminated, protocol-valid under the turn's think state), whose every turn
   is a token-native document (`document_from_turn`: the engine's prompt and
   output ids, ending on a stop id) under its mode's generation prompt
   (`prompt_mode_problems`), and whose teacher text passes the isolation
   recheck (`isolation_check.check_pool` on prompts and outputs). Units whole.
3. I = the T0 and E tasks admitted in both modes. STOPPING WAVE: the first wave
   W00, W01, ... whose I packs (H-T documents, `pack_isolated` at 8,192, doc ids
   sorted, the protocol's order seed) to at least the target; the waves after it
   were generated and are never trained on. H-T-S and H-N-S are the two modes'
   documents of that I.
4. X (H-N-C). X's H-N-admitted tasks, taken per stratum in X's order and
   interleaved by largest remainder against I's stratum shares, until H-N-C's
   pack (I's H-N documents and those tasks') reaches H-T-S's sequence count.
   C is FEASIBLE when that count is reached and every stratum's share of H-N-C's
   tasks is within 2 points of its share of I; otherwise the registered fallback
   H-T-C (a seeded whole-task prefix of I's H-T documents packed to H-N-S's
   count) against H-N-S. Decided here, before any training.
5. VALIDATION (diagnostic only): each mode's admitted V documents, packed the
   same way, at most ``VAL_SEQUENCES``; every arm validates in its own mode.
6. LENGTH LIMIT. A turn that reached its allowance is training-length-limited
   (right-censored at 8,184 minus the prompt, never a context-limit hit: the
   teacher supports 40,960) and is never admitted, cut or scored. Per cell, mode
   and wave: the first candidates' limited share, their prompt lengths, the
   admissions only a second candidate gave. A primary cell whose H-T first
   candidates are limited more than 20% carries the DESCRIPTIVE warning
   ``LENGTH_CENSORED``: reported in full, never a reason to drop or re-run.

``build_packs`` writes each arm's pack (`token_documents.write_pack`) from the
records and this record's document lists; the recovery pod calls it on the
volume's copy of the same shards and must reproduce every file's sha256.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.reasoning_data import assets as A  # noqa: E402
from stages.reasoning_data import generation_driver as GD  # noqa: E402
from stages.reasoning_data import source_plan as SP  # noqa: E402

PROTOCOL = REPO / "configs/stages/stage-2/teacher_modes/generation_protocol.json"
RECORD = REPO / "logs/stages/stage-2/teacher_modes/data/admission.json"
MODES = ("H-T", "H-N")
VAL_SEQUENCES = 16
MIXTURE_POINTS = 2.0
CENSOR_WARNING = 0.20
FALLBACK_SEED = 20261029
SCHEMA = "aadistill.stage2.teacher_modes.admission/v1"


def protocol(repo: Path = REPO) -> dict:
    return json.loads((Path(repo) / PROTOCOL.relative_to(REPO)).read_text())


def tasks_and_stages(proto: dict, repo: Path = REPO) -> tuple[dict, list[str], list[dict], dict]:
    """Every task by id, T0's ids, the order's rows, and the generation stage of every id."""
    from stages.teacher_modes.teacher_session import POOLS
    by = {}
    for name in POOLS:
        for t in A.jsonl(Path(repo) / "data/stages/stage-2/reasoning_data/tasks" / f"{name}.jsonl"):
            by.setdefault(t["task_id"], t)
    t0 = [json.loads(l)["task_id"] for lst in proto["inputs"]["t0"]
          for l in (Path(repo) / lst["path"]).read_text().splitlines() if l.strip()]
    order = list(A.jsonl(Path(repo) / proto["inputs"]["order"]["path"]))
    stage = {t: "w00" for t in t0}
    for t in order:
        stage[t["task_id"]] = "w00" if t["slice"] == "V" else f"{'w' if t['slice'] == 'E' else 'x'}{t['wave']:02d}"
    return {**{t: by[t] for t in t0}, **{t["task_id"]: t for t in order}}, t0, order, stage


def records(store: Path, identity: str) -> dict[tuple[str, str], list[dict]]:
    """(task, mode) -> its sealed records in candidate order, from every complete group."""
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for done in sorted(store.glob("*.complete")):
        label = done.name[:-len(".complete")]
        if label.startswith("cal_"):
            continue
        mode = label.split("_")[1]
        listed = {s["name"]: s["sha256"] for s in json.loads(done.read_text())["shards"]}
        have = {p.name: GD.sha(p) for p in store.glob(f"{label}.part*.jsonl")}
        if have != listed:
            raise GD.ShardError(f"{label}: the shards present are not the ones its .complete lists")
        for tid, rec in GD.sealed(store, label, identity).items():
            out[(tid, mode)].append(rec)
    for k, recs in out.items():
        idx = [r["candidate_index"] for r in recs]
        if len(set(idx)) != len(idx):
            raise GD.ShardError(f"{k}: candidate {idx} recorded twice")
        recs.sort(key=lambda r: r["candidate_index"])
    return out


def admit(task: dict, mode: str, recs: list[dict], proto: dict) -> dict:
    """The (task, mode) verdict: the admitted candidate and its documents, every candidate's reasons."""
    from aadistill.data.token_documents import DocumentError, document_from_turn, prompt_mode_problems
    from stages.reasoning_data.generation_adapter import candidate
    stop = proto["teacher"]["stop_ids"]
    suffix = proto["modes"][mode]["generation_suffix_ids"]
    seen = []
    for rec in recs:
        c = candidate(task, rec, rec["candidate_index"])
        why = list(c.reasons)
        docs = []
        try:
            docs = [document_from_turn(t, doc_id=f"{task['task_id']}|{mode}|t{k}", stop_ids=stop)
                    for k, t in enumerate(rec["turns"])]
        except DocumentError as exc:
            why.append(f"document: {str(exc)[:120]}")
        if docs and any(prompt_mode_problems(d, suffix) for d in docs):
            why.append("a prompt does not end in its mode's generation prompt")
        first = rec["turns"][0] if rec["turns"] else {}
        seen.append({"candidate": rec["candidate_index"], "accepted": bool(c.accepted_for_training and docs
                                                                            and not why[len(c.reasons):]),
                     "correct": c.answer.state == "correct" if hasattr(c, "answer") else None,
                     "length_limited": any(t.get("stop_reason") == "length_limited" for t in rec["turns"]),
                     "prompt_tokens": first.get("prompt_tokens"), "reasons": why,
                     "generated_tokens": sum(int(t["generated_tokens"]) for t in rec["turns"])})
        if seen[-1]["accepted"]:
            return {"admitted": True, "candidate": rec["candidate_index"], "docs": docs, "record": rec,
                    "candidates": seen}
    return {"admitted": False, "docs": [], "candidates": seen}


def isolation_recheck(admitted: dict, tasks: dict) -> tuple[set[tuple[str, str]], dict]:
    """(task, mode) pairs whose admitted record R1 or R2 flags against eval-set-1, P1v's recheck: the
    task with the record's own messages (an episode's observations are the teacher-dependent text)."""
    from shared import eval_sets as ES
    from stages.reasoning_data import isolation_check as IC
    items = [{**tasks[tid], "task_id": f"{tid}|{mode}", "unit_id": f"{tid}|{mode}", "messages": v["record"]["messages"]}
             for (tid, mode), v in sorted(admitted.items())]
    if not items:
        return set(), {"n_items": 0}
    rep = IC.check_pool(items)
    flagged = set(rep["results"]["R2_near_duplicate"]["flagged_ids"])
    if rep["results"]["R1_held_out_keys"]["held_out_overlap"]:
        held = {ES.prompt_key(ES.prompt_text(r)) for rows in ES.component_items(roles=ES.HELD_OUT_ROLES).values()
                for r in rows if ES.prompt_text(r).strip()}
        flagged |= {i["unit_id"] for i in items if not i.get("interactive") and ES.prompt_key(ES.prompt_text(i)) in held}
    return {tuple(f.rsplit("|", 1)) for f in flagged}, {"n_items": rep["n_items"], "pass": rep["pass"],
                                                        "violations": rep["violations"][:20]}


def count(docs: dict[str, int], proto: dict) -> int:
    from aadistill.data.token_documents import isolated_sequence_count
    st = proto["stopping"]
    return isolated_sequence_count(dict(sorted(docs.items())), int(st["sequence_length"]),
                                   order_seed=int(st["pack_order_seed"])) if docs else 0


def docs_of(ids, mode: str, verdicts: dict) -> dict[str, int]:
    out = {}
    for tid in ids:
        v = verdicts.get((tid, mode))
        if v and v["admitted"]:
            out.update({d.doc_id: len(d) for d in v["docs"]})
    return out


def stratum_shares(ids, tasks: dict) -> dict[str, float]:
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    c = Counter(tasks[t].get("stratum") or SP.stratum_of(tasks[t], spec) for t in ids)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in sorted(c.items())}


def decide(store: Path, identity: str, repo: Path = REPO) -> dict:
    proto = protocol(repo)
    tasks, t0, order, stage = tasks_and_stages(proto, repo)
    recs = records(store, identity)
    verdicts = {k: admit(tasks[k[0]], k[1], v, proto) for k, v in recs.items()}
    flagged, iso = isolation_recheck({k: v for k, v in verdicts.items() if v["admitted"]}, tasks)
    for k in flagged:
        verdicts[k] = {**verdicts[k], "admitted": False, "docs": [], "isolation_flagged": True}
    target = int(proto["stopping"]["target_sequences"])
    e_waves = sorted({s for s in stage.values() if s.startswith("w")})
    generated = sorted({stage[t] for (t, _) in recs if t in stage})
    v_ids = [t["task_id"] for t in order if t["slice"] == "V"]
    pool = [t for t in t0] + [t["task_id"] for t in order if t["slice"] == "E"]
    both = lambda ids: [t for t in ids if all(verdicts.get((t, m), {}).get("admitted") for m in MODES)]  # noqa: E731
    waves, stop = [], None
    for w in e_waves:
        if w not in generated:
            break
        ids = both([t for t in pool if stage[t] <= w])
        n = count(docs_of(ids, "H-T", verdicts), proto)
        waves.append({"wave": w, "I": len(ids), "H-T-S_sequences": n})
        if n >= target:
            stop = w
            break
    if stop is None:
        stop = waves[-1]["wave"] if waves else None
        e_state = ("E_EXHAUSTED: the reached count is the scale" if len(waves) == len(e_waves) else
                   "UNDER_GENERATED: a resumed teacher session generates the next wave")
    else:
        e_state = "REACHED"
    I = both([t for t in pool if stop is not None and stage[t] <= stop])
    s_ht = count(docs_of(I, "H-T", verdicts), proto)
    s_hn = count(docs_of(I, "H-N", verdicts), proto)
    # X: per stratum in X order, interleaved against I's stratum shares
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    share = stratum_shares(I, tasks)
    queues = defaultdict(list)
    for t in order:
        if t["slice"] == "X" and verdicts.get((t["task_id"], "H-N"), {}).get("admitted"):
            queues[t.get("stratum") or SP.stratum_of(t, spec)].append(t["task_id"])
    taken, x_ids, pos = Counter(), [], Counter()
    base = docs_of(I, "H-N", verdicts)
    s_hnc = s_hn
    while s_hnc < s_ht:
        live = [s for s in share if pos[s] < len(queues[s])]
        if not live:
            break
        n = len(x_ids) + 1
        s = max(sorted(live), key=lambda k: share[k] * (len(I) + n) - (share[k] * len(I) + taken[k]))
        x_ids.append(queues[s][pos[s]])
        pos[s] += 1
        taken[s] += 1
        if len(x_ids) % 25 == 0 or not any(pos[k] < len(queues[k]) for k in share):
            s_hnc = count({**base, **docs_of(x_ids, "H-N", verdicts)}, proto)
    s_hnc = count({**base, **docs_of(x_ids, "H-N", verdicts)}, proto)
    while x_ids and count({**base, **docs_of(x_ids[:-1], "H-N", verdicts)}, proto) >= s_ht:
        x_ids.pop()                                 # the smallest whole-task prefix that reaches the count
    s_hnc = count({**base, **docs_of(x_ids, "H-N", verdicts)}, proto)
    hnc_share = stratum_shares(I + x_ids, tasks)
    shift = {k: round(100 * (hnc_share.get(k, 0.0) - share.get(k, 0.0)), 3) for k in sorted(set(share) | set(hnc_share))}
    x_generated = [t for t in order if t["slice"] == "X" and stage[t["task_id"]] in generated]
    c_feasible = s_hnc >= s_ht and all(abs(v) <= MIXTURE_POINTS for v in shift.values())
    arms = {"H-T-S": {"ids": I, "mode": "H-T"}, "H-N-S": {"ids": I, "mode": "H-N"}}
    if c_feasible:
        arms["H-N-C"] = {"ids": I + x_ids, "mode": "H-N"}
        c_state = "FEASIBLE"
    else:
        rng = random.Random(FALLBACK_SEED)
        order_i = sorted(I)
        rng.shuffle(order_i)
        pre = []
        for t in order_i:
            if count(docs_of(pre, "H-T", verdicts), proto) >= s_hn:
                break
            pre.append(t)
        arms["H-T-C"] = {"ids": pre, "mode": "H-T"}
        c_state = ("FALLBACK: H-T-C (a seeded whole-task prefix of I's H-T documents at H-N-S's count) against "
                   "H-N-S" + ("" if s_hnc >= s_ht else
                              f"; X's admitted tasks reach {s_hnc} of {s_ht} sequences"
                              + (" (UNDER_GENERATED: X waves remain)" if len(x_generated) < sum(
                                  t['slice'] == 'X' for t in order) else " (X exhausted)")))
    for name, a in arms.items():
        docs = docs_of(a["ids"], a["mode"], verdicts)
        a.update(sequences=count(docs, proto), documents=len(docs), supervised_tokens=sum(
            len(d.output_ids) for t in a["ids"] for d in (verdicts.get((t, a["mode"])) or {}).get("docs", [])),
                 optimizer_steps=math.ceil(3 * count(docs, proto) / 2), doc_ids=sorted(docs))
    val = {m: sorted(docs_of([t for t in v_ids if verdicts.get((t, m), {}).get("admitted")], m, verdicts))
           for m in MODES}
    return {"proto": proto, "tasks": tasks, "verdicts": verdicts, "stage": stage, "waves": waves, "stop": stop,
            "e_state": e_state, "I": I, "arms": arms, "c_state": c_state, "x_ids": x_ids,
            "x_mixture_shift_points": shift, "val": val, "flagged": sorted(flagged), "isolation": iso,
            "t0": t0, "order": order}


def censoring(got: dict) -> dict:
    """Per cell, mode and wave: first candidates' length-limited share, prompt lengths, second-candidate admissions."""
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    rows = defaultdict(lambda: {"tasks": 0, "first_length_limited": 0, "first_correct": 0, "admitted": 0,
                                "admitted_by_second": 0, "prompt_tokens_limited": []})
    for (tid, mode), v in got["verdicts"].items():
        t = got["tasks"][tid]
        cell = f"{t.get('stratum') or SP.stratum_of(t, spec)}/{t.get('source')}"
        r = rows[(cell, mode, got["stage"].get(tid))]
        first = v["candidates"][0] if v["candidates"] else {}
        r["tasks"] += 1
        r["first_length_limited"] += bool(first.get("length_limited"))
        r["first_correct"] += bool(first.get("correct"))
        r["admitted"] += bool(v["admitted"])
        r["admitted_by_second"] += bool(v["admitted"] and v.get("candidate", 0) > 0)
        if first.get("length_limited"):
            r["prompt_tokens_limited"].append(first.get("prompt_tokens"))
    out, by_cell = [], defaultdict(lambda: [0, 0])
    for (cell, mode, wave), r in sorted(rows.items(), key=lambda kv: tuple(map(str, kv[0]))):
        out.append({"cell": cell, "mode": mode, "wave": wave, **r,
                    "first_length_limited_share": round(r["first_length_limited"] / r["tasks"], 4)})
        if mode == "H-T":
            by_cell[cell][0] += r["first_length_limited"]
            by_cell[cell][1] += r["tasks"]
    warn = sorted(c for c, (k, n) in by_cell.items() if c.startswith("primary.") and n and k / n > CENSOR_WARNING)
    return {"rows": out, "LENGTH_CENSORED": warn,
            "_": ("training-length-limited (right-censored at 8,184 minus the prompt; the teacher supports 40,960); a "
                  "DESCRIPTIVE warning above 20% of a primary cell's H-T first candidates: reported in full, never a "
                  "reason to drop a cell, change admission or re-run")}


def selection(got: dict) -> dict:
    """What the packs are made of, serializable: each admitted (task, mode)'s candidate, each arm's and each
    mode's validation documents. Everything else about a pack follows from the records and the protocol."""
    return {"admitted": {f"{t}|{m}": v["candidate"] for (t, m), v in sorted(got["verdicts"].items()) if v["admitted"]},
            "arms": {k: {"mode": a["mode"], "doc_ids": a["doc_ids"]} for k, a in got["arms"].items()},
            "val": {m: list(v) for m, v in got["val"].items()}}


def build_packs(sel: dict, store: Path, identity: str, out: Path, repo: Path = REPO) -> dict:
    """Every arm's pack and its mode's validation tail, from the sealed records and `sel` alone (the
    dev box writes them; the recovery pod rebuilds them from the volume's copy of the same shards)."""
    from aadistill.data.token_documents import document_from_turn, pack_isolated, write_pack
    proto = protocol(repo)
    tasks, *_ = tasks_and_stages(proto, repo)
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    seq, seed, pad = (int(proto["stopping"]["sequence_length"]), int(proto["stopping"]["pack_order_seed"]),
                      int(proto["teacher"]["pad_id"]))
    recs = records(Path(store), identity)
    docs, dtype = {}, {}
    for key, cand in sel["admitted"].items():
        tid, mode = key.rsplit("|", 1)
        rec = next(r for r in recs[(tid, mode)] if r["candidate_index"] == cand)
        stratum = tasks[tid].get("stratum") or SP.stratum_of(tasks[tid], spec)
        for k, t in enumerate(rec["turns"]):
            d = document_from_turn(t, doc_id=f"{tid}|{mode}|t{k}", stop_ids=proto["teacher"]["stop_ids"])
            docs[d.doc_id], dtype[d.doc_id] = d, stratum
    val = {}
    for m, ids in sel["val"].items():
        whole = pack_isolated([docs[k] for k in sorted(ids)], seq, pad, order_seed=seed)
        keep = sorted(x["doc_id"] for ms in whole.members[:VAL_SEQUENCES] for x in ms)
        val[m] = pack_isolated([docs[k] for k in keep], seq, pad, order_seed=seed)
    res = {}
    for name, a in sel["arms"].items():
        tr = pack_isolated([docs[k] for k in sorted(a["doc_ids"])], seq, pad, order_seed=seed)
        res[name] = write_pack(tr, val[a["mode"]], Path(out) / name, data_type=dtype,
                               extra={"arm": name, "mode": a["mode"], "context_strategy": "document_isolated"})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=str(RECORD))
    a = ap.parse_args(argv)
    got = decide(Path(a.records), a.identity)
    sel = selection(got)
    packs = build_packs(sel, Path(a.records), a.identity, Path(a.out))
    doc = {"schema": SCHEMA, "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "study_identity": a.identity, "protocol_sha256": A.sha256_file(PROTOCOL),
           "records": {"dir": a.records, "complete_groups": sorted(p.name for p in Path(a.records).glob("*.complete"))},
           "stopping": {"waves": got["waves"], "stopping_wave": got["stop"], "state": got["e_state"],
                        "target_sequences": got["proto"]["stopping"]["target_sequences"]},
           "I": {"tasks": len(got["I"]), "ids": got["I"]},
           "C": {"state": got["c_state"], "x_tasks": len(got["x_ids"]), "x_ids": got["x_ids"],
                 "stratum_share_shift_points": got["x_mixture_shift_points"]},
           "arms": {k: {**{x: v[x] for x in ("mode", "sequences", "documents", "supervised_tokens", "optimizer_steps")},
                        "pack": packs[k]} for k, v in got["arms"].items()},
           "validation": {m: {"documents": len(v)} for m, v in got["val"].items()},
           "selection": sel,
           "isolation_recheck": {**got["isolation"], "flagged": got["flagged"]},
           "length_limit": censoring(got),
           "authorizes": "nothing"}
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"stopping": doc["stopping"], "I": doc["I"]["tasks"], "C": doc["C"]["state"],
                      "arms": {k: {x: v[x] for x in ("sequences", "optimizer_steps")} for k, v in doc["arms"].items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
