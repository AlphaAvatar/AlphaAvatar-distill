#!/usr/bin/env python3
"""Stage 3 (-sr): why the students' rollouts fail, from what is already saved. $0, dev box, read-only.

    PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python -m stages.teacher_modes.stage3_failure_analysis \\
        --units <dir holding <unit>/evaluation/generations> --teacher-ht <teacher H-T dir> --teacher-hn <teacher H-N dir> \\
        [--packs artifacts/stages/stage-2/teacher_modes/packs_6cat] [--out <json>]

Nothing is generated, re-scored formally or changed: the formal numbers stay `stage3_sr_result.json`'s. Every count
here is DESCRIPTIVE and reads the saved token ids, the frozen packs and the registered detector and scorer:

* rollouts (the 570 formal static items and the 144 interactive episodes per unit): the stop reason and the
  detector's kind; whether the stop fell inside the reasoning (no ``</think>`` yet) or in the answer; the tail's
  repetition period; whether an answer was already there when the rollout stopped and whether it matched the gold
  (scored with the registered static scorer, ignoring termination: what ``correct = usable AND answer_correct`` leaves
  out); for episodes, the turn at which the first non-EOS stop came;
* the formal rate split into the usable rate and correctness among usable rollouts, and the thinking - non-thinking
  difference split the same way (a midpoint decomposition: d(u*c) = du*mean(c) + dc*mean(u));
* the training data each arm learned from (the frozen packs): whether every assistant segment's ``<|im_end|>`` is a
  supervised target (CE and, with ``kd_scope: assistant``, KD read the same mask), the think tokens a segment opens
  with and holds, the teacher targets' own lengths and what the registered degeneration detector says of them, and
  what packing did to the end of sequence: fragments, repeated prefixes, supervised end-of-turn tokens per epoch.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts", "scripts/stages/stage-1"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.evaluation import degeneration as DG  # noqa: E402
from stages.calibration_volume import score_battery as SB  # noqa: E402
from stages.reasoning_data.h1_analysis import fluid_meta  # noqa: E402

OUT = REPO / "logs/stages/stage-2/teacher_modes/analyses/stage3_failure_analysis.json"
PACKS = REPO / "artifacts/stages/stage-2/teacher_modes/packs_6cat"
ARMS = {"A1-6cat-sr": "thinking (hybrid)", "A2-6cat-sr": "non-thinking (hybrid)", "C1-6cat-sr": "thinking (Thinking-2507)"}
SEEDS = (20260726, 20260801)
#: Qwen3's chat-template tokens, the same ids in both lineages' tokenizers (checked against the pinned files below)
IM_START, IM_END, THINK_OPEN, THINK_CLOSE = 151644, 151645, 151667, 151668
FLUID = "fluid_reasoning_720__dev_v1"
PROTOCOL = "v2"
BOXED = re.compile(r"\\boxed\{")


def pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 1) if b else None


def dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "mean": round(st.mean(s), 1), "median": s[len(s) // 2], "p90": s[int(0.9 * (len(s) - 1))],
            "max": s[-1]}


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().split("\n") if x.strip()]


def check_tokens() -> dict:
    from transformers import AutoTokenizer
    from stages.teacher_modes import training_plan as TP
    out = {}
    for lineage, rel in TP.TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(str(REPO / rel))
        got = [tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")]
        if got != [IM_START, IM_END, THINK_OPEN, THINK_CLOSE]:
            raise SystemExit(f"{lineage}: template token ids {got} are not Qwen3's")
        out[lineage] = got
    return out


def last_boxed(text: str) -> str | None:
    """The last ``\\boxed{...}`` in ``text``, braces balanced; None when there is none."""
    starts = [m.end() for m in BOXED.finditer(text)]
    for s in reversed(starts):
        depth, i = 1, s
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            return text[s:i - 1]
    return None


def answer_matches(task: dict, text: str) -> bool | None:
    """The registered static scorer's answer check on ``text`` alone (termination ignored); None: no boxed answer."""
    b = last_boxed(text)
    if b is None:
        return None
    got = SB.CANON_BY_PROTOCOL[PROTOCOL][task["answer_type"]](b)
    return got is not None and got == SB.gold_key_for(task, PROTOCOL)


def split_turn(t: dict) -> tuple[str, str, bool]:
    """(reasoning text, answer text, think closed) of a turn as generated."""
    raw = (t.get("raw") or "").replace("<|im_end|>", "")
    if t.get("think_state") == "preclosed":
        return "", raw, True
    if "</think>" in raw:
        r, a = raw.split("</think>", 1)
        return r, a, True
    return raw, "", False


def tail_cycle(ids: list[int]) -> dict | None:
    return DG.find_cycle(ids, max_period=200, min_repeats=4)


def static_unit(gens: list[dict], rows: dict, items: dict) -> dict:
    recs = [g for g in gens if rows.get(g["task_id"], {}).get("formal") and not items[g["task_id"]]["interactive"]]
    n = len(recs)
    stops, kinds, where, tokens_at_stop, periods = Counter(), Counter(), Counter(), defaultdict(list), Counter()
    answer_before = Counter()
    reasoning_tok = defaultdict(list)
    examples = defaultdict(list)
    for g in recs:
        t = g["turns"][-1]
        task = items[g["task_id"]]
        stop = t["stop_reason"]
        stops[stop] += 1
        reasoning, answer, closed = split_turn(t)
        ids = t.get("output_token_ids") or []
        if closed and t.get("think_state") != "preclosed" and THINK_CLOSE in ids:
            reasoning_tok[stop].append(ids.index(THINK_CLOSE))
        if stop == "eos":
            continue
        kinds[f"{stop}:{t.get('degeneration_kind') or '-'}"] += 1
        seg = "answer" if closed else "reasoning"
        where[f"{stop} in {seg}"] += 1
        tokens_at_stop[stop].append(t["generated_tokens"])
        c = tail_cycle(ids)
        periods["no exact cycle" if c is None else "period 1-4" if c["period"] <= 4 else
                "period 5-30" if c["period"] <= 30 else "period 31-200"] += 1
        ok = answer_matches(task, answer) if closed else answer_matches(task, reasoning)
        answer_before[f"{seg}: " + ("no boxed answer yet" if ok is None else "boxed, matches gold" if ok
                                    else "boxed, wrong")] += 1
        key = f"{stop}:{t.get('degeneration_kind') or '-'}:{seg}"
        if len(examples[key]) < 2:
            examples[key].append({"task_id": g["task_id"], "axis": task["axis"], "family_id": task["family_id"],
                                  "generated_tokens": t["generated_tokens"],
                                  "cycle": c and {k: c[k] for k in ("period", "repeats")},
                                  "tail": (answer if closed else reasoning)[-420:]})
    usable = [g for g in recs if rows[g["task_id"]].get("reason") != "unusable"]
    correct = sum(bool(rows[g["task_id"]]["correct"]) for g in recs)
    #: termination ignored: the registered answer check on whatever answer each rollout had produced
    any_answer_ok = sum(bool(answer_matches(items[g["task_id"]], split_turn(g["turns"][-1])[1]))
                        for g in recs if split_turn(g["turns"][-1])[2])
    wrong = Counter(rows[g["task_id"]].get("reason") for g in usable)
    return {"n": n, "stops": dict(stops), "stops_pct": {k: pct(v, n) for k, v in stops.items()},
            "non_eos_by_kind": dict(kinds), "non_eos_where": dict(where),
            "non_eos_tail_cycle": dict(periods), "non_eos_answer_present": dict(answer_before),
            "generated_tokens_at_stop": {k: dist(v) for k, v in tokens_at_stop.items()},
            "reasoning_tokens_when_closed": {k: dist(v) for k, v in reasoning_tok.items()},
            "formal": {"correct": correct, "rate": round(correct / n, 4), "usable": len(usable),
                       "usable_rate": round(len(usable) / n, 4),
                       "correct_among_usable": round(correct / len(usable), 4) if usable else None},
            "usable_outcomes": dict(wrong),
            "descriptive_answer_correct_ignoring_termination": {
                "correct": any_answer_ok, "rate": round(any_answer_ok / n, 4),
                "_": "an answer after </think> (or the whole reply when the think block is pre-closed) that the "
                     "registered scorer accepts, whether or not the rollout terminated; never a formal number"},
            "examples": dict(examples)}


def episode_unit(gens: list[dict], rows: dict, items: dict) -> dict:
    recs = [g for g in gens if rows.get(g["task_id"], {}).get("formal") and items[g["task_id"]]["interactive"]]
    ends, first_bad, where, turns = Counter(), Counter(), Counter(), []
    per_family = defaultdict(lambda: [0, 0])
    examples = []
    for g in recs:
        ts = g["turns"]
        turns.append(len(ts))
        ends[g.get("end")] += 1
        fam = items[g["task_id"]]["family_id"].split(".")[0]
        per_family[fam][0] += 1
        per_family[fam][1] += bool(rows[g["task_id"]].get("correct"))
        bad = next((i for i, t in enumerate(ts) if t["stop_reason"] != "eos"), None)
        first_bad["none" if bad is None else "turn 0" if bad == 0 else "turn 1" if bad == 1 else "turn 2+"] += 1
        if bad is not None:
            t = ts[bad]
            r, a, closed = split_turn(t)
            where[f"{t['stop_reason']}:{t.get('degeneration_kind') or '-'} in {'answer' if closed else 'reasoning'}"] += 1
            if len(examples) < 3 and bad == 0:
                examples.append({"task_id": g["task_id"], "family_id": items[g["task_id"]]["family_id"],
                                 "generated_tokens": t["generated_tokens"], "tail": (a if closed else r)[-420:]})
    return {"n": len(recs), "ends": dict(ends), "first_non_eos_turn": dict(first_bad), "first_non_eos_where": dict(where),
            "turns_per_episode": dist(turns), "correct_by_family_group": {k: v for k, v in per_family.items()},
            "examples": examples}


def decomposition(per_unit: dict) -> dict:
    """Formal rate = usable rate x correctness among usable; the A2 - A1 difference split at the midpoint."""
    out = {}
    for seed in SEEDS:
        a1, a2 = (per_unit[f"{arm}-s{seed}"]["static"]["formal"] for arm in ("A1-6cat-sr", "A2-6cat-sr"))
        du, dc = a2["usable_rate"] - a1["usable_rate"], a2["correct_among_usable"] - a1["correct_among_usable"]
        mu, mc = (a1["usable_rate"] + a2["usable_rate"]) / 2, (a1["correct_among_usable"] + a2["correct_among_usable"]) / 2
        out[str(seed)] = {"A2_minus_A1_pp": round(100 * (a2["rate"] - a1["rate"]), 2),
                          "from_usable_rate_pp": round(100 * du * mc, 2),
                          "from_correct_among_usable_pp": round(100 * dc * mu, 2)}
    out["_"] = ("descriptive, per seed, unpaired: d(u*c) = du*mean(c) + dc*mean(u) is exact; the registered A1 - A2 "
                "comparison (stage3_sr_result.json) is the only inferential statement")
    return out


def teacher_overlap(student: list[dict], teacher: list[dict], rows_s: dict, rows_t: dict, items: dict) -> dict:
    ids = [g["task_id"] for g in student if rows_s.get(g["task_id"], {}).get("formal")
           and not items[g["task_id"]]["interactive"]]
    c = Counter((bool(rows_t[i]["correct"]), bool(rows_s[i]["correct"]), rows_s[i].get("reason")) for i in ids)
    by_axis = defaultdict(lambda: [0, 0])
    for i in ids:
        if rows_t[i]["correct"]:
            by_axis[items[i]["axis"]][0] += 1
            by_axis[items[i]["axis"]][1] += bool(rows_s[i]["correct"])
    return {"teacher_correct": sum(v for (t, _, _), v in c.items() if t),
            "teacher_correct_student_correct": sum(v for (t, s, _), v in c.items() if t and s),
            "teacher_correct_student_wrong_by_reason": {r: v for (t, s, r), v in c.items() if t and not s},
            "teacher_wrong_student_correct": sum(v for (t, s, _), v in c.items() if not t and s),
            "retained_on_teacher_correct_by_axis": {a: {"teacher_correct": v[0], "student_correct": v[1],
                                                        "retained_pct": pct(v[1], v[0])} for a, v in by_axis.items()}}


LENGTH_BUCKETS = ((0, 1024), (1024, 2048), (2048, 4096), (4096, 10 ** 9))


def bucket(n: int) -> str:
    lo, hi = next(b for b in LENGTH_BUCKETS if b[0] <= n < b[1])
    return f"{lo}-{hi}" if hi < 10 ** 9 else f">={lo}"


def by_teacher_length(student: list[dict], teacher: list[dict], rows_s: dict, items: dict) -> dict:
    """The student's stops and correctness on each static formal item, bucketed by how many reasoning tokens the
    hybrid teacher's thinking mode spent on the SAME item (a difficulty proxy; the teacher ran at 40,960)."""
    tlen = {}
    for g in teacher:
        ids = g["turns"][-1].get("output_token_ids") or []
        tlen[g["task_id"]] = ids.index(THINK_CLOSE) if THINK_CLOSE in ids else len(ids)
    out = defaultdict(lambda: Counter())
    for g in student:
        i = g["task_id"]
        if not rows_s.get(i, {}).get("formal") or items[i]["interactive"] or i not in tlen:
            continue
        c = out[bucket(tlen[i])]
        c["n"] += 1
        c[g["turns"][-1]["stop_reason"]] += 1
        c["correct"] += bool(rows_s[i]["correct"])
    return {b: {**v, "non_eos_pct": pct(v["n"] - v["eos"], v["n"]), "correct_pct": pct(v["correct"], v["n"])}
            for b, v in sorted(out.items(), key=lambda kv: int(kv[0].split("-")[0].lstrip(">=")))}


def length_coverage(arm_dir: Path) -> dict:
    """Teacher targets' reasoning tokens (to ``</think>``) by data group, and the share that reach each length the
    students' failures sit at: what the training set offered at the lengths eval-set-1 asks for."""
    z = np.load(arm_dir / "data_documents.npz")
    ids, mask, offs = z["training_ids"], z["training_mask"], z["training_offsets"]
    man = json.loads((arm_dir / "data_manifest.json").read_text())
    rows = [dict(zip(man["columns"], r)) for r in man["documents"]["training"]]
    groups = defaultdict(list)
    for d, r in enumerate(rows):
        a, b = int(offs[d]), int(offs[d + 1])
        seg = ids[a:b][mask[a:b]].tolist()
        dt = r["data_type"]
        g = "primary (fluid)" if dt.startswith("primary.") else "tool calling" if "tool_calling" in dt else "supporting"
        groups[g].append(seg.index(THINK_CLOSE) if THINK_CLOSE in seg else None)
    out = {}
    for g, v in groups.items():
        r = [x for x in v if x is not None]
        out[g] = {"targets": len(v), "with_reasoning": len(r), **{k: x for k, x in dist(r).items() if k != "n"},
                  **{f"reach_{t}": pct(sum(x >= t for x in r), len(r)) for t in (2048, 2816, 4096)}}
    return out


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2], d[1::2]))


def pack_unit(arm_dir: Path, epochs: int = 3) -> dict:
    docs = np.load(arm_dir / "data_documents.npz")
    ids_all, mask_all, offs = docs["training_ids"], docs["training_mask"], docs["training_offsets"]
    seg_end, seg_open, seg_close, prompt_tail, lens, reason_lens = Counter(), Counter(), Counter(), Counter(), [], []
    flagged, kinds, system = 0, Counter(), Counter()
    n_docs = len(offs) - 1
    n_segs = 0
    for d in range(n_docs):
        a, b = int(offs[d]), int(offs[d + 1])
        ids, mask = ids_all[a:b], mask_all[a:b]
        system["system message" if len(ids) > 2 and ids[0] == IM_START and ids[1] == 8948 else "none"] += 1
        for s, e in segments(mask):
            n_segs += 1
            seg = ids[s:e].tolist()
            lens.append(e - s)
            seg_end["ends with <|im_end|>" if seg[-1] == IM_END else f"ends with {seg[-1]}"] += 1
            seg_open["opens with <think>" if seg[0] == THINK_OPEN else "opens with text"] += 1
            seg_close["holds </think>" if THINK_CLOSE in seg else "no </think>"] += 1
            pre = ids[max(0, s - 6):s].tolist()
            prompt_tail["prompt ends with an empty closed think block" if THINK_CLOSE in pre
                        else "prompt ends with <think>" if THINK_OPEN in pre else "prompt ends at the assistant tag"] += 1
            if THINK_CLOSE in seg:
                reason_lens.append(seg.index(THINK_CLOSE))
            body = seg[:-1] if seg[-1] == IM_END else seg
            hit = DG.check(body)
            if hit:
                flagged += 1
                kinds[hit.get("kind")] += 1
    z = np.load(arm_dir / "blocks.npz")
    meta = json.loads((arm_dir / "ladder.json").read_text())
    nb = int(meta["n_blocks"])
    inp, ce, content = z["input_ids"][:nb], z["ce_mask"][:nb], z["content_mask"][:nb]
    end_tok = (inp == IM_END) & content
    sup_end = int((end_tok & ce).sum())
    #: an <|im_end|> that closes an assistant turn is the token right after a supervised run; is it supervised itself?
    closing = unsup_closing = frag_tokens = frag_runs = 0
    for i in range(nb):
        runs = segments(ce[i])
        for s, e in runs:
            if inp[i, e - 1] == IM_END:
                closing += 1
            elif e < inp.shape[1] and inp[i, e] == IM_END and content[i, e]:
                unsup_closing += 1
        if runs:
            s, e = runs[-1]
            if inp[i, e - 1] != IM_END:
                frag_runs += 1
                frag_tokens += e - s
    sup = int(ce.sum())
    return {"documents": n_docs, "assistant_segments": n_segs,
            "system_message": dict(system), "segment_end": dict(seg_end), "segment_open": dict(seg_open),
            "segment_think_close": dict(seg_close), "prompt_before_segment": dict(prompt_tail),
            "supervised_tokens_per_segment": dist(lens), "reasoning_tokens_per_segment": dist(reason_lens),
            "teacher_targets_flagged_by_the_detector": {"n": flagged, "pct": pct(flagged, n_segs),
                                                        "kinds": dict(kinds),
                                                        "_": "the registered detector on each complete target"},
            "packed": {"sequences": nb, "supervised_tokens": sup, "supervised_im_end": sup_end,
                       "im_end_in_sequences": int(end_tok.sum()),
                       "assistant_runs_ending_in_supervised_im_end": closing,
                       "assistant_runs_followed_by_unsupervised_im_end": unsup_closing,
                       "fragments_at_sequence_end": frag_runs, "fragment_supervised_tokens": frag_tokens,
                       "fragment_share_of_supervised_pct": pct(frag_tokens, sup),
                       "supervised_im_end_per_1k_supervised": round(1000 * sup_end / sup, 3),
                       "supervised_im_end_over_training": sup_end * epochs, "epochs": epochs}}


def training_evidence(unit_dir: Path) -> dict:
    rows = jsonl(unit_dir / "train" / "train_log.jsonl")
    val = [r for r in rows if r.get("event") in ("eval_result", "val") or "val_ce" in r]
    st_ = [r for r in rows if r.get("event") == "train_step"]
    pick = lambda r: {k: r.get(k) for k in ("step", "val_ce", "val_kd", "ce", "kd", "loss") if r.get(k) is not None}  # noqa: E731
    return {"train_steps_logged": len(st_), "first_step": pick(st_[0]) if st_ else None,
            "last_step": pick(st_[-1]) if st_ else None,
            "validation": [pick(r) for r in val][:1] + [pick(r) for r in val][-1:]}


def analyse(units: Path, teacher_ht: Path | None, teacher_hn: Path | None, packs: Path) -> dict:
    items, _ = fluid_meta()
    tokens = check_tokens()
    load = lambda d: (jsonl(d / "evaluation/generations" / f"{FLUID}.jsonl"),  # noqa: E731
                      {r["id"]: r for r in jsonl(d / "evaluation/rows" / f"{FLUID}.rows.jsonl")})
    per = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}-s{seed}"
            gens, rows = load(units / name)
            per[name] = {"static": static_unit(gens, rows, items), "episodes": episode_unit(gens, rows, items),
                         "training": training_evidence(units / name)}
    teachers = {}
    for key, d in (("teacher_H-T", teacher_ht), ("teacher_H-N", teacher_hn)):
        if d is not None:
            gens, rows = load(d)
            teachers[key] = {"static": static_unit(gens, rows, items), "episodes": episode_unit(gens, rows, items),
                             "_gens": gens, "_rows": rows}
    overlap, by_len = {}, {}
    for arm, tkey in (("A1-6cat-sr", "teacher_H-T"), ("A2-6cat-sr", "teacher_H-N")):
        if tkey in teachers:
            for seed in SEEDS:
                gens, rows = load(units / f"{arm}-s{seed}")
                overlap[f"{arm}-s{seed}"] = teacher_overlap(gens, teachers[tkey]["_gens"], rows, teachers[tkey]["_rows"],
                                                            items)
    if "teacher_H-T" in teachers:
        for arm in ARMS:
            for seed in SEEDS:
                gens, rows = load(units / f"{arm}-s{seed}")
                by_len[f"{arm}-s{seed}"] = by_teacher_length(gens, teachers["teacher_H-T"]["_gens"], rows, items)
    for t in teachers.values():
        t.pop("_gens"), t.pop("_rows")
    return {"schema": "aadistill.stage2.teacher_modes.stage3_failure_analysis/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": {"units": "each unit's saved generations and rows (the evidence archives)",
                       "teachers": "the hybrid teacher's eval-set-1 evaluation in both modes (Research A v1), "
                                   "context 40,960; the students ran at their trained context 8,192",
                       "packs": str(packs.relative_to(REPO)) if packs.is_relative_to(REPO) else str(packs),
                       "template_token_ids": tokens},
            "units": per, "teachers": teachers, "teacher_correct_items": overlap,
            "by_teacher_reasoning_length": by_len,
            "decomposition_A2_minus_A1": decomposition(per),
            "packs": {arm: {**pack_unit(packs / arm), "reasoning_length_coverage": length_coverage(packs / arm)}
                      for arm in ARMS},
            "formal_scoring": "correct = usable_rollout AND answer_correct over all 570 static formal items: a "
                              "degeneration stop or a context-limit hit is unusable and scores 0; context-limit "
                              "hits are recorded as right-censored for the length statistics, not excluded from "
                              "the rate",
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", required=True)
    ap.add_argument("--teacher-ht", default=None)
    ap.add_argument("--teacher-hn", default=None)
    ap.add_argument("--packs", default=str(PACKS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    doc = analyse(Path(a.units), Path(a.teacher_ht) if a.teacher_ht else None,
                  Path(a.teacher_hn) if a.teacher_hn else None, Path(a.packs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(doc["decomposition_A2_minus_A1"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
