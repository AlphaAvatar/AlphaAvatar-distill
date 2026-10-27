#!/usr/bin/env python3
"""Pools for the SUPPORTING strata of stage2.reasoning_master@v1. $0, CPU.

    PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.build_supporting_tasks \
        --source {reuse,math_train,mbpp_train,squad_v2_train_unanswerable,all} [--version 1]

Two kinds of pool, one per source of `corpus_spec.json :: source_allocation`:

* REUSE (corpus_v2_gsm8k_verified, corpus_v2_rag_verified,
  corpus_v2_multihop_verified): no generation. corpus_v2's SELECTED sessions
  (sha-verified file) whose target passes the H1 contract again here: the
  session matches its candidate byte for byte (the candidate's sha256 is
  corpus_v2's ``candidate_sha256``; the session's answer and reasoning are the
  candidate's), the candidate terminated naturally and is protocol-valid, and
  its final answer passes `aadistill.data.verify`'s key (`supporting_verify`:
  final number for gsm8k, span for reading). A reading answer that LEADS with an
  abstention is refused even when the span check passed: corpus_v2's span check
  accepts "I cannot answer from the context. The context states ..." when the
  quoted context contains the gold, and that is not a correct answer to an
  answerable question. The committed file is a SELECTION (ids, each session's
  canonical sha256, verdict), never teacher text; `reused_sessions` re-reads
  corpus_v2 and refuses any session whose bytes changed. These sessions are
  corpus_v2's own selections -- its lowest-index hygiene-accepted candidate
  (`build_recovery_corpus.py`; checked on all 11,174 selected rows) -- kept only
  when that target passes B's contract; a unit whose selected target fails is
  dropped, never re-selected, so a reused session is byte-for-byte the one
  corpus_v2 trained on.
* TASKS (math_train, mbpp_train, squad_v2_train_unanswerable): prompts for the
  paid generation step, each carrying the gold its `supporting_verify` check
  reads and every field isolation rule R0 requires. Demand-sized per (stratum,
  source) cell: `source_plan.plan` units / a recorded unit-accept-rate basis x
  ``--oversupply``, drawn in a fixed hash order (`SALT`) -- nothing is chosen by
  looking at results.

Every pool passes `isolation_check.check_pool(..., refuse=True)`; items R1 or
R2 flag are removed first and listed in the manifest. MATH is also screened
against all of MATH-500 (exact key and R2's 13-gram containment), not only the
100 eval-set-1 holds. Writes `data/stages/stage-2/reasoning_data/tasks/`.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stages.reasoning_data import assets as A
from stages.reasoning_data import isolation_check as IC
from stages.reasoning_data import source_plan as SP
from stages.reasoning_data import supporting_verify as SV

from aadistill.data import cleaning
from aadistill.data.target_quality import AcceptancePolicy, assess
from aadistill.evaluation import isolation as ISO
from shared import eval_sets as ES
from stages.calibration_volume.tasks.common import ANSWER_INSTRUCTION

OUT_DIR = A.REPO / "data/stages/stage-2/reasoning_data/tasks"
VARIANT = "stage2.reasoning_master@v1"
SALT = "stage2.supporting/v1"
SCHEMA = "aadistill.stage2.reasoning_data.supporting_pool/v1"
HF = Path.home() / ".cache/huggingface/hub"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MBPP_SPLITS = ("train", "validation", "prompt")
CODE_INSTRUCTION = "Write the complete Python solution in a single ```python code block in your final answer."
#: corpus_v2 type -> (B stratum, B source, verifier); the reuse sources.
REUSE = {"gsm8k": ("supporting.math_reused_gsm8k", "corpus_v2_gsm8k_verified", "final_number"),
         "rag_evidence": ("supporting.reading_verified", "corpus_v2_rag_verified", "span"),
         "multihop_qa": ("supporting.reading_verified", "corpus_v2_multihop_verified", "span")}
#: unit-accept-rate basis per task source: the measured n = 1 candidate accept
#: rate of the nearest corpus_v2 type under the contract (asset audit), or the
#: corpus_spec planning assumption. The second candidate is not counted.
BASIS = {"math_train": ("corpus_v2", "openmath"),
         "squad_v2_train_unanswerable": ("corpus_v2", "rag_evidence"),
         "mbpp_train": ("corpus_spec", "supporting.code_algorithmic")}
POLICY = AcceptancePolicy()


def _order(key: str) -> str:
    return hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest()


def canonical_sha(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _snapshot(repo: str, revision: str | None = None) -> Path:
    snaps = sorted(glob.glob(str(HF / f"datasets--{repo.replace('/', '--')}/snapshots/*")))
    if revision:
        snaps = [s for s in snaps if Path(s).name == revision]
    if not snaps:
        raise SystemExit(f"{repo}{'@' + revision if revision else ''} is not in the local HF cache")
    return Path(snaps[-1])


def _parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def cell_units(source: str) -> dict[str, int]:
    """Independent units the variant's plan needs from `source`, per stratum."""
    return {c["stratum"]: c["independent_units_estimate"] for c in SP.plan(VARIANT)["cells"]
            if c["source"] == source and c["share"] > 0}


def accept_basis(source: str) -> dict:
    where, key = BASIS[source]
    if where == "corpus_v2":
        per = A.read_json(A.LOG_DIR / "audit/asset_audit.json")["corpus_v2"]["under_target_quality_contract"]
        rate = per["per_type"][key]["candidate_accept_rate"]
        return {"rate": rate, "basis": f"corpus_v2 {key}: measured n = 1 candidate accept rate under the "
                                       f"contract (audit/asset_audit.json), a proxy for this source"}
    rate = float(A.read_json(A.CONFIG_DIR / "corpus_spec.json")["generation"][key]["yield_from"].split(":")[1])
    return {"rate": rate, "basis": f"corpus_spec.json :: generation.{key} planning assumption "
                                   "(no measured rate exists)"}


def demand(source: str, oversupply: float) -> dict:
    units = cell_units(source)
    if len(units) != 1:
        raise SystemExit(f"{source}: expected one stratum in {VARIANT}, got {units}")
    (stratum, need), = units.items()
    basis = accept_basis(source)
    return {"stratum": stratum, "units_needed": need, "unit_accept_rate": basis["rate"],
            "accept_rate_basis": basis["basis"], "oversupply": oversupply,
            "tasks": math.ceil(need / basis["rate"] * oversupply)}


def _task(source: str, key: str, prompt: str, **fields) -> dict:
    tid = hashlib.sha256(f"{source}|{key}".encode()).hexdigest()[:20]
    return {"task_id": tid, "unit_id": tid, "source": source, "interactive": False,
            "messages": [{"role": "user", "content": prompt}],
            "content_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **fields}


# ---------------------------------------------------------------- tasks

def math_candidates() -> tuple[list[dict], dict]:
    snap = _snapshot("EleutherAI/hendrycks_math", MATH_REVISION)
    files = sorted(snap.glob("*/train-00000-of-00001.parquet"))
    if len(files) != 7:
        raise SystemExit(f"MATH train: {len(files)} of 7 subject files in {snap}")
    out, dropped = [], Counter()
    for f in files:
        subject = f.parent.name
        for i, r in enumerate(_parquet(f)):
            gold = SV.boxed_answer(r["solution"])
            if gold is None:
                dropped["no boxed answer in the reference solution"] += 1
                continue
            prompt = r["problem"].rstrip() + "\n\n" + ANSWER_INSTRUCTION
            out.append(_task("math_train", f"{subject}|{i}|{r['problem']}", prompt,
                             source_split="train", stratum="supporting.math_new", family_id=subject,
                             difficulty=r["level"], verifier="boxed_math", answer=gold,
                             answer_type="math_string", row=i))
    files_rec = [{"file": f.relative_to(snap).as_posix(), "bytes": f.stat().st_size,
                  "sha256": A.sha256_file(f)} for f in files]
    return out, {"dataset": f"EleutherAI/hendrycks_math @ {MATH_REVISION}", "files": files_rec,
                 "dropped": dict(dropped)}


def math500_screen(tasks: list[dict]) -> tuple[set[str], dict]:
    """MATH-500 in full (eval-set-1 holds 100 of it): exact key and R2 containment."""
    rows = [json.loads(l) for l in (_snapshot("HuggingFaceH4/MATH-500") / "test.jsonl").read_text().splitlines()
            if l.strip()]
    r2 = next(r for r in IC.rules()["rules"] if r["id"] == "R2_near_duplicate")
    keys = {ES.prompt_key(r["problem"]) for r in rows}
    index = ISO.ngram_index([r["problem"] for r in rows], r2["ngram_n"],
                            max_document_frequency=r2.get("max_document_frequency"))
    by_key = {t["task_id"] for t in tasks
              if ES.prompt_key(t["messages"][0]["content"].rsplit("\n\n", 1)[0]) in keys}
    rep = ISO.ngram_containment([(t["task_id"], t["messages"][0]["content"]) for t in tasks], index,
                                r2["ngram_n"], threshold=r2["containment_threshold"])
    flagged = {row["id"] for row in rep["flagged_items"]}
    return by_key | flagged, {"math500_items": len(rows), "exact_key_matches": len(by_key),
                              "ngram_flagged": len(flagged), "with_any_shared_ngram": rep["with_any_shared_ngram"]}


def mbpp_candidates() -> tuple[list[dict], dict]:
    snap = _snapshot("google-research-datasets/mbpp")
    check = A.read_json(A.LOG_DIR / "audit/code_verifier_check.json")
    failing = {(r["split"], r["task_id"]) for r in check["reference_failures"]}
    out, dropped, files = [], Counter(), []
    for split in MBPP_SPLITS:
        path = snap / "full" / f"{split}-00000-of-00001.parquet"
        sha = A.sha256_file(path)
        if sha != check["per_split"][split]["file_sha256"]:
            raise SystemExit(f"MBPP {split}: file sha {sha[:12]} is not the calibrated one")
        files.append({"file": path.relative_to(snap).as_posix(), "sha256": sha})
        for r in _parquet(path):
            if 11 <= r["task_id"] <= 510:
                raise SystemExit(f"MBPP test task {r['task_id']} in the {split} split")
            if (split, r["task_id"]) in failing:
                dropped["the reference solution fails its own asserts (code_verifier_check)"] += 1
                continue
            if r.get("challenge_test_list"):
                raise SystemExit("MBPP challenge tests present: decide whether they are stated or held back")
            tests = list(r["test_list"])
            prompt = (f"{r['text'].strip()}\n\nYour code should pass these tests:\n\n" + "\n".join(tests)
                      + "\n\n" + CODE_INSTRUCTION)
            out.append(_task("mbpp_train", f"{split}|{r['task_id']}", prompt, source_split=split,
                             stratum="supporting.code_algorithmic", family_id="mbpp", difficulty="",
                             verifier="code_asserts", test_list=tests,
                             test_setup_code=r.get("test_setup_code") or "", mbpp_task_id=r["task_id"],
                             answer_type="code",
                             gold_in_prompt="the asserts ARE the stated specification (MBPP's own format); "
                                            "the reference solution is never shown"))
    return out, {"dataset": f"google-research-datasets/mbpp @ {snap.name} (full; test split never read)",
                 "files": files, "dropped": dict(dropped),
                 "verification_calibration": "audit/code_verifier_check.json"}


def squad_candidates() -> tuple[list[dict], dict]:
    from shared.data.build_stage2_v0 import RAG_PROMPT
    snap = _snapshot("rajpurkar/squad_v2")
    path = snap / "squad_v2" / "train-00000-of-00001.parquet"
    out, dropped = [], Counter()
    for r in _parquet(path):
        if r["answers"]["text"]:
            continue
        prompt = RAG_PROMPT.format(context=r["context"], question=r["question"])
        out.append(_task("squad_v2_train_unanswerable", r["id"], prompt, source_split="train",
                         stratum="supporting.answerability", family_id="squad_v2_train_unanswerable",
                         difficulty="", verifier="abstention", answer="", answer_type="abstention",
                         squad_id=r["id"], context_sha256=hashlib.sha256(r["context"].encode()).hexdigest()))
    return out, {"dataset": f"rajpurkar/squad_v2 @ {snap.name} train, unanswerable questions",
                 "files": [{"file": path.relative_to(snap).as_posix(), "sha256": A.sha256_file(path)}],
                 "prompt_template": "scripts/shared/data/build_stage2_v0.py :: RAG_PROMPT (corpus_v2 rag_evidence's)",
                 "draw_rule": "one question per context paragraph", "dropped": dict(dropped)}


def abstention_validation() -> dict:
    """The abstention check on the teacher's corpus_v2 replies to ANSWERABLE
    questions (rag_evidence, multihop_qa; no eval-set-1 item is read)."""
    n, hit = Counter(), Counter()
    for ex in A.corpus_v2_candidates():
        if ex["data_type"] not in ("rag_evidence", "multihop_qa"):
            continue
        for c in ex["candidates"]:
            k = f"{ex['data_type']}:{'span_correct' if str(c.get('correct')) == 'True' else 'span_incorrect'}"
            n[k] += 1
            hit[k] += SV.is_abstention(c["answer"])[0]
    return {"material": "corpus_v2 teacher candidates for answerable questions (4 per prompt)",
            "candidates": dict(sorted(n.items())), "detected_as_abstention": dict(sorted(hit.items())),
            "reading": ("detections among span-correct replies are the teacher DECLINING an answerable question "
                        "while quoting context that contains the gold (corpus_v2's span check accepts them); a "
                        "reading on 2026-10-22 of 25 random such detections and of every one not led by "
                        "'I cannot' / 'The context does not' found no reply that answers the question. Before the "
                        "lead-sentence rule, an answer followed by 'the context does not specify ...' was detected "
                        "(614 rag detections then, against 483 now)")}


CANDIDATES = {"math_train": math_candidates, "mbpp_train": mbpp_candidates,
              "squad_v2_train_unanswerable": squad_candidates}


def _isolate(items: list[dict], key=lambda it: it["unit_id"]) -> tuple[list[dict], dict, dict]:
    """Drop what R1/R2 flag (by unit), then the rest must pass every rule."""
    first = IC.check_pool(items)
    r1 = first["results"]["R1_held_out_keys"]["held_out_overlap"]
    flagged = set(first["results"]["R2_near_duplicate"]["flagged_ids"])
    if r1:
        held = {ES.prompt_key(ES.prompt_text(r)) for rows in ES.component_items(roles=ES.HELD_OUT_ROLES).values()
                for r in rows if ES.prompt_text(r).strip()}
        flagged |= {key(it) for it in items if ES.prompt_key(ES.prompt_text(it)) in held}
    kept = [it for it in items if key(it) not in flagged]
    report = IC.check_pool(kept, refuse=True)
    removed = {"R1_held_out_overlap": r1, "R2_flagged": first["results"]["R2_near_duplicate"]["flagged"],
               "R2_flagged_by_unit": first["results"]["R2_near_duplicate"]["flagged_by_unit"],
               "removed_units": sorted(flagged), "first_pass_violations": first["violations"]}
    return kept, report, removed


def build_tasks(source: str, oversupply: float) -> dict:
    cands, provenance = CANDIDATES[source]()
    want = demand(source, oversupply)
    cands.sort(key=lambda t: _order(t["task_id"]))
    screened = {}
    if source == "math_train":
        gone, screened = math500_screen(cands)
        cands = [t for t in cands if t["task_id"] not in gone]
    if source == "squad_v2_train_unanswerable":
        seen, one = set(), []
        for t in cands:
            if t["context_sha256"] not in seen:
                seen.add(t["context_sha256"])
                one.append(t)
        cands = one
    # draw in hash order, then screen; refill from the order so the pool keeps its size
    pool, removed_all, i = [], {"R2_flagged": 0, "removed_units": []}, 0
    while len(pool) < want["tasks"] and i < len(cands):
        batch = cands[i:i + want["tasks"] - len(pool)]
        i += len(batch)
        kept, _, removed = _isolate(pool + batch)
        removed_all["R2_flagged"] += removed["R2_flagged"]
        removed_all["removed_units"] += removed["removed_units"]
        pool = kept
    kept, report, _ = _isolate(pool)
    if len(kept) < want["tasks"]:
        raise SystemExit(f"{source}: {len(kept)} eligible tasks for {want['tasks']} (no substitute source is drawn)")
    return {"tasks": kept, "report": report, "demand": want, "provenance": provenance,
            "eligible_candidates": len(cands), "removed_by_isolation": removed_all, "screened": screened}


# ---------------------------------------------------------------- reuse

def _cv2_stop(c: dict) -> str:
    from stages.reasoning_data.audit_assets import _cv2_stop as stop
    return stop(c)


def reuse_selection() -> dict:
    from stages.reasoning_data.audit_assets import CV2_ENDS
    cands = {ex["id"]: ex for ex in A.corpus_v2_candidates() if ex["data_type"] in REUSE}
    rows, why, items, scopes = [], Counter(), [], {}
    sessions = [s for s in A.corpus_v2_sessions() if s["data_type"] in REUSE]
    size = Counter(s["source_id"] for s in sessions)
    for s in sessions:
        stratum, source, verifier = REUSE[s["data_type"]]
        ex = cands.get(s["id"])
        c = next((c for c in (ex or {}).get("candidates", []) if int(c["index"]) == int(s["candidate_index"])), None)
        asst = s["messages"][-1]
        if c is None or hashlib.sha256(c["raw"].encode()).hexdigest() != s["candidate_sha256"] \
                or asst.get("content") != c["answer"] or asst.get("reasoning_content") != c["think"]:
            why[f"{source}: session does not match its candidate"] += 1
            continue
        task = {"answer": s["gold"], "group": s["group"], "corpus_source": s["source"]}
        check = SV.VERIFIERS[verifier](task, c["think"], c["answer"])
        if check.state == "correct" and verifier == "span" and SV.is_abstention(c["answer"])[0]:
            check = type(check)("incorrect", "leads with an abstention; the span matched only quoted context",
                                check.scope)
        proto = cleaning.check_serialization(c) if c.get("raw") is not None else "no_raw"
        rec = assess(s["source_id"], int(c["index"]), stop_reason=_cv2_stop(c), protocol_reason=proto,
                     answer=check, policy=POLICY, end_reasons=CV2_ENDS,
                     generated_tokens=int(c.get("new_tokens") or 0))
        if not rec.accepted_for_training:
            why[f"{source}: {rec.reasons[0] if rec.reasons else 'rejected'}"] += 1
            continue
        if size[s["source_id"]] != 1:
            raise SystemExit(f"{s['source_id']}: a {size[s['source_id']]}-session reuse unit; whole-unit reuse "
                             "of multi-session units is not implemented")
        scopes[source] = check.scope
        rows.append({"session_id": s["id"], "unit_id": s["source_id"], "unit_sessions": size[s["source_id"]],
                     "session_sha256": canonical_sha(s), "stratum": stratum, "source": source,
                     "candidate_index": int(c["index"]), "supervised_tokens": int(s["n_supervised_tokens"]),
                     "verdict": check.state})
        items.append({"unit_id": s["source_id"], "source": source, "source_split": "train",
                      "messages": s["messages"]})
    kept_items, report, removed = _isolate(items)
    keep = {it["unit_id"] for it in kept_items}
    rows = [r for r in rows if r["unit_id"] in keep]
    return {"rows": rows, "report": report, "removed_by_isolation": removed, "rejected": dict(sorted(why.items())),
            "verdict_scope": dict(sorted(scopes.items())),
            "corpus_v2": {"manifest_sessions_sha256": A.corpus_v2_manifest()["outputs"]["sessions"],
                          "manifest_candidates_sha256": A.corpus_v2_manifest()["outputs"]["candidates"]}}


def reused_sessions(selection_path: Path) -> list[dict]:
    """The selected corpus_v2 sessions in build_master_ladder's input contract,
    each re-verified against its recorded canonical sha256; a unit is returned
    whole or the call refuses."""
    sel = list(A.jsonl(Path(selection_path)))
    want = {r["session_id"]: r for r in sel}
    out = []
    for s in A.corpus_v2_sessions():
        r = want.get(s["id"])
        if r is None:
            continue
        if canonical_sha(s) != r["session_sha256"]:
            raise ValueError(f"{s['id']}: corpus_v2 session bytes differ from the selection's sha256")
        out.append({"id": s["id"], "unit_id": r["unit_id"], "unit_sessions": r["unit_sessions"],
                    "stratum": r["stratum"], "family": r["source"],
                    "attributes": {"source": r["source"], "reused_from": "corpus_v2"},
                    "messages": s["messages"], "tools": s.get("tools"), "data_type": r["stratum"],
                    "candidate_index": s["candidate_index"], "candidate_sha256": s["candidate_sha256"],
                    "target": {"reused_from": "corpus_v2", "session_sha256": r["session_sha256"],
                               "candidate_index": s["candidate_index"],
                               "candidate_sha256": s["candidate_sha256"]}})
    got = Counter(o["unit_id"] for o in out)
    missing = sorted(set(want) - {o["id"] for o in out})
    short = sorted(u for u, n in got.items() if n != next(r["unit_sessions"] for r in sel if r["unit_id"] == u))
    if missing or short:
        raise ValueError(f"selection not reproducible: {len(missing)} sessions missing, units incomplete: {short[:5]}")
    return out


# ---------------------------------------------------------------- write

def _write(name: str, rows: list[dict], manifest: dict, report: dict) -> None:
    out = OUT_DIR / f"{name}.jsonl"
    if out.exists():
        raise SystemExit(f"{out.relative_to(A.REPO)} exists; a built pool is never rebuilt -- bump --version")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows))
    (OUT_DIR / f"{name}.isolation.json").write_text(json.dumps(report, indent=1) + "\n")
    manifest = {**manifest, "sha256": A.sha256_file(out), "isolation_report": f"{name}.isolation.json",
                "isolation_pass": report["pass"], "authorizes": "nothing"}
    (OUT_DIR / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(out.relative_to(A.REPO), len(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=("reuse", *CANDIDATES, "all"))
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--oversupply", type=float, default=1.5)
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    common = {"schema": SCHEMA, "corpus_variant": VARIANT, "created_utc": now, "version": args.version,
              "status": "PROPOSED pool: sources admitted for H1 design (maintainer, 2026-10-22); no teacher target "
                        "is generated for it until a generation grant exists"}
    if args.source in ("reuse", "all"):
        got = reuse_selection()
        rows = got["rows"]
        by = defaultdict(lambda: {"units": 0, "supervised_tokens": 0})
        for r in rows:
            by[r["source"]]["units"] += 1
            by[r["source"]]["supervised_tokens"] += r["supervised_tokens"]
        need = {c["source"]: {"units_needed": c["independent_units_estimate"],
                              "U_target_with_reserve": c["U_target_with_reserve"]}
                for c in SP.plan(VARIANT)["cells"] if c["source"] in {v[1] for v in REUSE.values()}}
        _write(f"supporting_reuse_v{args.version}.selection", rows, {
            **common, "kind": "reuse selection (no generation)",
            "command": f"python -m stages.reasoning_data.build_supporting_tasks --source reuse --version {args.version}",
            "by_source": dict(sorted(by.items())), "plan_needs": need,
            "rejected": got["rejected"], "removed_by_isolation": got["removed_by_isolation"],
            "verdict_scope": got["verdict_scope"], "family_key": "the source id",
            "corpus_v2": got["corpus_v2"],
            "selection_rule": "corpus_v2's selected session (its lowest-index hygiene-accepted candidate; not "
                              "reselected) whose target passes the contract again: matches its "
                              "candidate, natural termination, protocol-valid, answer key passes (no length rule); "
                              "a reading answer leading with an abstention is refused",
            "reader": "build_supporting_tasks.reused_sessions(<this selection>)",
            "teacher_text": "not in this file: artifacts/stages/stage-3/corpus_v2/sessions.jsonl (sha-verified)"},
            got["report"])
    for source in CANDIDATES:
        if args.source not in (source, "all"):
            continue
        got = build_tasks(source, args.oversupply)
        tasks = got["tasks"]
        _write(f"{source}_v{args.version}", tasks, {
            **common, "kind": "generation task pool", "source": source,
            "command": (f"python -m stages.reasoning_data.build_supporting_tasks --source {source} "
                        f"--version {args.version} --oversupply {args.oversupply}"),
            "demand": got["demand"], "n_tasks": len(tasks), "eligible_candidates": got["eligible_candidates"],
            "draw_order": f"sha256('{SALT}|<task_id>') ascending", "provenance": got["provenance"],
            "math500_screen": got["screened"] or None, "removed_by_isolation": got["removed_by_isolation"],
            "by_family": dict(sorted(Counter(t["family_id"] for t in tasks).items())),
            "by_difficulty": dict(sorted(Counter(t["difficulty"] for t in tasks).items())),
            "verifier": f"supporting_verify.VERIFIERS[{tasks[0]['verifier']!r}]",
            **({"verifier_validation": abstention_validation()} if source == "squad_v2_train_unanswerable"
               else {})}, got["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
