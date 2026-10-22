#!/usr/bin/env python3
"""From generation records to training sessions: verify, assess, select, emit. $0.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.generation_adapter \
        --variant <corpus id> --tasks <pool.jsonl> [--tasks ...] --records <generation.jsonl> \
        --out <sessions.jsonl> [--review]
    # $0 validation on the Stage-0 teacher records (real teacher outputs):
    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.generation_adapter --validate-stage0

THE ENGINE is not here. The paid step runs the driver that produced Stage-0's
teacher records, `scripts/stages/stage-0/calibration_volume/run_battery.py
--mode generate`: the teacher's recommended sampling, one seeded request per
turn, and real action -> observation loops for episodes (the environment reads
each reply's action and its observation becomes the next user turn). It writes
generation records and executes no model-written code, so the generation pod's
credentials, network and volume never meet teacher programs.

THIS MODULE then, on the dev box:

1. refuses to process records for a corpus variant whose source plan is not
   GENERATABLE (`source_plan`: an unapproved source), unless ``--review``,
   which labels its output review-only;
2. verifies each candidate, keeping every check in its own record
   (`aadistill.data.target_quality` v2):
   * project static tasks: the battery's v2 scorer (boxed answer vs gold) for
     the answer; text consistency as an observation; steps ``not_assessed``
     (no step checker exists for these families);
   * episodes: environment REPLAY for the answer (goal reached) and for the
     ``action_legality`` observation; the natural-language reasoning is not
     verified by a legal action sequence, so steps stay ``not_assessed``;
   * external static tasks: the boxed final answer against the generator's
     gold with the module's declared comparator (`external_modules.json ::
     answer_types`); a module whose answers need a checker the project does not
     have yet (``needs_checker``: several valid answers, a plan, a derivation)
     is ``unverifiable`` -- never accepted -- rather than guessed at; a library's
     own scorer is never used;
3. accepts under the group's policy (episodes: an illegal action rejects) and
   selects the lowest accepted candidate index per unit, never by length;
4. writes sessions in `build_master_ladder`'s input contract: a static task is
   one session; an accepted episode is its turn-expanded sessions, each
   declaring ``unit_sessions``, so the packer admits the episode whole or not at
   all.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import process_check as PC
from stages.reasoning_data import source_plan as SP

from aadistill.data.target_quality import AcceptancePolicy, Check, assess, select_target, summarize
from aadistill.data.verify import boxed_answer

ENDS = {"natural": ("eos",), "length": ("length_limited", "length", "context_limit"),
        "degeneration": ("degeneration",)}
STATIC = AcceptancePolicy()
EPISODE = AcceptancePolicy(reject_observations={"action_legality": ("illegal_action",)})
OUT = A.LOG_DIR / "audit/generation_adapter_validation.json"


def _text(s) -> str:
    return " ".join(str(s).lower().split()).strip(" .")


def _numbers(s) -> list[float]:
    out = []
    for a, b in re.findall(r"(-?\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?", str(s)):
        out.append(float(a) / float(b) if b else float(a))
    return out


def _tokens(s) -> list[str]:
    return re.findall(r"[^\s,;\[\]\\]+", str(s))


#: answer_type -> (boxed answer, gold) -> equal. A comparator, never a score.
COMPARE = {
    "text": lambda got, gold: _text(got) == _text(gold),
    "number": lambda got, gold: (len(_numbers(got)) == len(_numbers(gold)) == 1
                                 and abs(_numbers(got)[0] - _numbers(gold)[0]) < 1e-9),
    "number_set": lambda got, gold: (len(_numbers(got)) == len(_numbers(gold)) > 0 and all(
        abs(a - b) < 1e-6 for a, b in zip(sorted(_numbers(got)), sorted(_numbers(gold))))),
    "matrix": lambda got, gold: _tokens(got) == _tokens(gold) and bool(_tokens(gold)),
    "bool": lambda got, gold: _text(got) in ("true", "false") and _text(got) == _text(gold),
}


def external_answer(task: dict, final: str) -> Check:
    kind = task.get("answer_type")
    scope = f"the boxed final answer against the generator's gold ({kind} comparator)"
    if kind not in COMPARE:
        return Check("unverifiable", f"no comparator for answer type {kind!r}", scope)
    got = boxed_answer(final or "")
    if got is None:
        return Check("incorrect", "no boxed answer", scope)
    return Check("correct", "", scope) if COMPARE[kind](got, task["answer"]) else \
        Check("incorrect", "wrong answer", scope)


def _stop(rec: dict) -> str:
    return next((t["stop_reason"] for t in rec["turns"] if t["stop_reason"] != "eos"), "eos")


def _protocol(rec: dict) -> str | None:
    return None if rec["turns"] and all(A.split_raw(t.get("raw", "")) for t in rec["turns"]) else "think_not_closed"


def candidate(task: dict, rec: dict, index: int = 0):
    """One generation record -> its CandidateRecord, every check apart."""
    from stages.calibration_volume import score_battery as SB
    obs: dict[str, Check] = {}
    if task.get("interactive"):
        s = SB.score_episode(task, rec)
        inv = int((s.get("env") or {}).get("invalid_actions", 0))
        obs["action_legality"] = Check("all_legal" if inv == 0 else "illegal_action",
                                       f"invalid_actions={inv}", "environment replay of every action")
        ans = Check("correct" if s["correct"] else "incorrect", "" if s["correct"] else "goal not reached",
                    "the environment's goal reached on replay")
        policy = EPISODE
    elif task.get("generator_id"):
        s = SB.score_static(task, rec, "v2")
        ok = bool(s.get("answer_correct"))
        ans = Check("correct" if ok else "incorrect", "" if ok else str(s.get("reason", "")),
                    "the boxed final answer against the generator's gold (battery protocol v2)")
        parts = A.split_raw(rec["turns"][-1].get("raw", "")) if rec["turns"] else None
        if parts:
            obs["text_consistency"] = PC.text_consistency(
                parts[0], s.get("extracted"), task["answer_type"],
                prompt=" ".join(m.get("content") or "" for m in task["messages"]))
        policy = STATIC
    elif task.get("module"):
        parts = A.split_raw(rec["turns"][-1].get("raw", "")) if rec["turns"] else None
        ans = external_answer(task, parts[1] if parts else "")
        policy = STATIC
    else:
        ans = Check("unverifiable", f"no verifier for source {task.get('source')!r}", "none")
        policy = STATIC
    return assess(task["task_id"], index, stop_reason=_stop(rec), protocol_reason=_protocol(rec),
                  answer=ans, steps=PC.NO_STEP_CHECKER, observations=obs, policy=policy,
                  end_reasons=ENDS, generated_tokens=sum(int(t["generated_tokens"]) for t in rec["turns"]))


def sessions_for(task: dict, rec: dict, stratum: str) -> list[dict]:
    """The accepted record as packer sessions: turn k is the history through the
    k-th assistant message, and every turn declares the unit's size."""
    history = rec["messages"]
    at = [i for i, m in enumerate(history) if m["role"] == "assistant"]
    fam = f"{task['generator_id']}/{task['family_id']}" if task.get("generator_id") else task.get("family_id")
    return [{"id": f"{task['task_id']}#t{k}", "unit_id": task["task_id"], "unit_sessions": len(at),
             "stratum": stratum, "family": fam,
             "attributes": {"source": task.get("source", ""), "difficulty": task.get("difficulty", "")},
             "messages": history[:idx + 1], "data_type": stratum} for k, idx in enumerate(at)]


def adapt(tasks: dict[str, dict], records: list[dict], stratum_of) -> dict:
    by_unit: dict[str, list] = {}
    for i, rec in enumerate(records):
        task = tasks.get(rec["task_id"])
        if task is None:
            raise SystemExit(f"record for unknown task {rec['task_id']}")
        idx = int(rec.get("candidate_index", 0))
        by_unit.setdefault(rec["task_id"], []).append((candidate(task, rec, idx), rec))
    out, cands = [], []
    for uid, rows in sorted(by_unit.items()):
        cands += [c for c, _ in rows]
        best = select_target([c for c, _ in rows])
        if best is None:
            continue
        rec = next(r for c, r in rows if c.candidate_index == best.candidate_index)
        out += sessions_for(tasks[uid], rec, stratum_of(tasks[uid]))
    return {"sessions": out, "candidates": cands,
            "summary": summarize(cands, group_of=lambda c: tasks[c.unit_id].get("axis") or "external")}


def validate_stage0() -> dict:
    """Run the whole post-generation path on the 672 Stage-0 teacher records and
    pack the result with the shared builder; compare with base_pool_v1."""
    from stages.reasoning_data import build_master_ladder as B
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    tasks = A.calibration_tasks()
    recs = A.teacher_cal_records()
    got = adapt(tasks, recs, lambda t: SP.stratum_of(t, spec))
    rendered, excluded, _ = B.render_all(B.load_tokenizer(), got["sessions"], block_len=spec["block_len"])
    admitted = {r.source_id for r in rendered.values()}
    base = {it["unit_id"] for it in A.base_pool_items() if it.get("pool_class") == "primary"}
    accepted = {c.unit_id for c in got["candidates"] if c.accepted_for_training}
    return {"schema": "aadistill.stage2.reasoning_data.generation_adapter_validation/v1",
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input": {"records": "Stage-0 teacher_cal.jsonl (sha256-verified by assets.teacher_cal_records)",
                      "n_records": len(recs)},
            "contract": got["summary"],
            "accepted_units": len(accepted),
            "sessions_written": len(got["sessions"]),
            "packer_admission": {"admitted_units": len(admitted), **B.exclusion_summary(excluded)},
            "against_base_pool_v1": {"base_pool_units": len(base), "both": len(admitted & base),
                                     "admitted_not_in_base_pool": sorted(admitted - base)[:20],
                                     "in_base_pool_not_admitted": sorted(base - admitted)[:20]},
            "reading": ("the adapter + whole-unit packer reproduce base_pool_v1's unit set when 'both' equals both "
                        "counts; any difference is listed by unit"),
            "authorizes": "nothing"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate-stage0", action="store_true")
    ap.add_argument("--variant")
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--records")
    ap.add_argument("--out")
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args(argv)
    if args.validate_stage0:
        doc = validate_stage0()
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
        print(OUT.relative_to(A.REPO), doc["packer_admission"], doc["against_base_pool_v1"]["both"])
        return 0
    if not (args.variant and args.tasks and args.records and args.out):
        ap.error("--variant, --tasks, --records and --out are required")
    plan = SP.plan(args.variant)
    if not plan["generatable"] and not args.review:
        raise SystemExit(f"{args.variant}: {plan['status']}; pass --review for a review-only output")
    spec = A.read_json(A.CONFIG_DIR / "corpus_spec.json")
    tasks = {t["task_id"]: t for p in args.tasks for t in A.jsonl(Path(p))}
    got = adapt(tasks, list(A.jsonl(Path(args.records))), lambda t: SP.stratum_of(t, spec))
    out = Path(args.out)
    out.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in got["sessions"]))
    meta = {"corpus_id": args.variant, "review_only": not plan["generatable"],
            "records_sha256": A.sha256_file(Path(args.records)), "sessions_sha256": A.sha256_file(out),
            "tasks_sha256": {p: A.sha256_file(Path(p)) for p in args.tasks},
            "contract": got["summary"], "by_stratum": dict(Counter(s["stratum"] for s in got["sessions"]))}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(out, meta["review_only"], meta["by_stratum"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
