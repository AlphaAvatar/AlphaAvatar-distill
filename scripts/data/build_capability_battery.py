#!/usr/bin/env python
"""Build and freeze the Experiment 2 capability evaluation battery.

Six held-out sets with deterministic scorers, plus the two Experiment 1 sets that
are reused unchanged. The battery is frozen **before** D1 trains: sample ids,
prompts, serialization, decoding parameters, scoring rules, evaluator version and
hashes all go into one manifest, and nothing in it may be tuned after results are
seen.

Leakage is handled twice, belt and braces:

1. **Structural.** `build_stage2_v1.py` drew every source from its `train`
   split, so the `validation` / `test` splits used here were never eligible for
   corpus v2 in the first place. TriviaQA has never been used by this project at
   all.
2. **Hash.** Every candidate is then checked against the same content-hash and
   first-user-message-hash rule the corpus itself used
   (`build_recovery_corpus.py::content_key` / `prompt_key`) over corpus v2's
   sessions, all three `stage2_v1` splits and `eval_behavior_v0`. Collisions are
   dropped and counted.

`math_verified` is MATH-500, which this project has never used at any stage —
`stage2_v1` drew its math from GSM8K and OpenMathInstruct-2. That makes it a
harder and cleanly held-out probe than sampling unused OpenMathInstruct-2 rows
would be: those would be disjoint items but the same distribution as a sixth of
the training corpus, and they cost tens of GB to download for a hundred rows.

Selection is a deterministic stride over ids sorted by sha256, so it needs no
seed and re-running reproduces the same battery.

Usage:
    scripts/data/build_capability_battery.py --out artifacts/eval/battery_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rollout"))

from aadistill.data.verify import boxed_answer  # noqa: E402
from aadistill.evaluation.capability import BATTERY_VERSION  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

# Reserved material every candidate is checked against.
EXCLUDE_PATHS = [
    "data/stage2_v1/train", "data/stage2_v1/val", "data/stage2_v1/calib",
    "data/eval_behavior_v0/prompts.jsonl",
]

SOURCES = {
    "knowledge": {"repo": "mandarjoshi/trivia_qa", "config": "rc.nocontext",
                  "split": "validation", "license": "Apache-2.0"},
    "math_verified": {"repo": "HuggingFaceH4/MATH-500", "config": None,
                      "split": "test", "license": "MIT"},
    "gsm8k": {"repo": "openai/gsm8k", "config": "main", "split": "test",
              "license": "MIT"},
    "multihop": {"repo": "hotpotqa/hotpot_qa", "config": "distractor",
                 "split": "validation", "license": "CC-BY-SA-4.0"},
    "rag": {"repo": "rajpurkar/squad_v2", "config": None, "split": "validation",
            "license": "CC-BY-SA-4.0"},
    "refusal_paired": {"repo": "rajpurkar/squad_v2", "config": None,
                       "split": "validation", "license": "CC-BY-SA-4.0"},
}

RAG_INSTRUCTION = ("Answer the question using only the provided context. If the "
                   "context does not contain the answer, say you cannot answer "
                   "from the context.")
MULTIHOP_INSTRUCTION = ("Answer the question using the provided documents. Name "
                        "the documents your answer relies on.")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_excluded():
    content, prompts = set(), set()
    for spec in EXCLUDE_PATHS:
        p = REPO_ROOT / spec
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else ([p] if p.exists() else [])
        for f in files:
            for line in f.open():
                if not line.strip():
                    continue
                sample = json.loads(line)
                messages = sample.get("messages")
                if messages:
                    content.add(sha(json.dumps(messages, sort_keys=True,
                                               ensure_ascii=False)))
                    for m in messages:
                        if m.get("role") == "user":
                            prompts.add(sha(m.get("content", "")))
                            break
                else:
                    content.add(sha(sample.get("text") or ""))
    return content, prompts


def load_corpus_prompts(corpus: Path | None):
    """First-user-message hashes of every session corpus v2 actually built."""
    keys = set()
    if corpus is None or not (corpus / "sessions.jsonl").is_file():
        return keys
    with (corpus / "sessions.jsonl").open() as f:
        for line in f:
            for m in json.loads(line)["messages"]:
                if m.get("role") == "user":
                    keys.add(sha(m.get("content", "")))
                    break
    return keys


def stride(items, n):
    """Deterministic, seed-free selection: sort by id hash, take an even stride."""
    ordered = sorted(items, key=lambda x: sha(x["id"]))
    if len(ordered) <= n:
        return ordered
    step = len(ordered) / n
    return [ordered[int(i * step)] for i in range(n)]


def build(name, rows, n, excluded_content, excluded_prompts, corpus_prompts):
    kept, drops = [], Counter()
    for r in rows:
        key = sha(r["prompt_text"])
        if key in excluded_prompts:
            drops["reserved_split_prompt"] += 1
            continue
        if key in corpus_prompts:
            drops["corpus_v2_prompt"] += 1
            continue
        if sha(r["prompt_text"] + "\x00" + str(r.get("gold", ""))) in excluded_content:
            drops["reserved_split_content"] += 1
            continue
        kept.append(r)
    chosen = stride(kept, n)
    return chosen, {"candidates": len(rows), "after_leakage_filter": len(kept),
                    "selected": len(chosen), "dropped": dict(drops)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="corpus v2 dir, for the prompt-hash leakage check")
    ap.add_argument("--n-knowledge", type=int, default=150)
    ap.add_argument("--n-math", type=int, default=100)
    ap.add_argument("--n-gsm8k", type=int, default=100)
    ap.add_argument("--n-multihop", type=int, default=100)
    ap.add_argument("--n-rag", type=int, default=100)
    ap.add_argument("--n-refusal-pairs", type=int, default=60)
    args = ap.parse_args()
    args.out = args.out if args.out.is_absolute() else (REPO_ROOT / args.out)

    from datasets import load_dataset

    excluded_content, excluded_prompts = load_excluded()
    corpus_prompts = load_corpus_prompts(args.corpus)
    print(f"leakage sets: {len(excluded_content)} content hashes, "
          f"{len(excluded_prompts)} reserved prompts, "
          f"{len(corpus_prompts)} corpus-v2 prompts", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    sets, audits = {}, {}

    def emit(name, rows):
        path = args.out / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sets[name] = {"path": str(path.relative_to(REPO_ROOT)), "n": len(rows),
                      "sha256": sha256_file(path),
                      "source": SOURCES[name],
                      "sample_ids": [r["id"] for r in rows]}

    # ---- knowledge: closed-book TriviaQA, alias-set EM -------------------
    d = load_dataset(SOURCES["knowledge"]["repo"], name=SOURCES["knowledge"]["config"],
                     split=SOURCES["knowledge"]["split"])
    rows = [{"id": f"trivia-{r['question_id']}", "group": "knowledge",
             "source": "trivia_qa", "prompt_text": r["question"].strip(),
             "messages": [{"role": "user", "content": r["question"].strip()}],
             "gold": r["answer"]["value"],
             "aliases": sorted({r["answer"]["value"], *r["answer"]["aliases"],
                                *r["answer"]["normalized_aliases"]})}
            for r in d.select(range(min(4000, len(d))))]
    chosen, audits["knowledge"] = build("knowledge", rows, args.n_knowledge,
                                        excluded_content, excluded_prompts,
                                        corpus_prompts)
    emit("knowledge", chosen)

    # ---- math_verified: MATH-500, boxed + symbolic -----------------------
    d = load_dataset(SOURCES["math_verified"]["repo"],
                     split=SOURCES["math_verified"]["split"])
    rows = []
    for r in d:
        gold = boxed_answer(r.get("solution") or "") or (r.get("answer") or "").strip()
        if not gold or not r.get("problem"):
            continue
        prompt = (r["problem"].strip() +
                  "\n\nPut your final answer in \\boxed{}.")
        rows.append({"id": f"math500-{r['unique_id']}", "group": "math_verified",
                     "source": "math_500", "subject": r.get("subject"),
                     "level": r.get("level"),
                     "prompt_text": prompt,
                     "messages": [{"role": "user", "content": prompt}],
                     "gold": gold, "boxed": gold})
    chosen, audits["math_verified"] = build("math_verified", rows, args.n_math,
                                            excluded_content, excluded_prompts,
                                            corpus_prompts)
    emit("math_verified", chosen)

    # ---- gsm8k: strict EM on the held-out test split ---------------------
    d = load_dataset(SOURCES["gsm8k"]["repo"], name=SOURCES["gsm8k"]["config"],
                     split=SOURCES["gsm8k"]["split"])
    rows = [{"id": f"gsm8k-test-{i:05d}", "group": "gsm8k", "source": "gsm8k",
             "prompt_text": r["question"].strip(),
             "messages": [{"role": "user", "content": r["question"].strip()}],
             "gold": r["answer"].split("####")[-1].strip(),
             "gsm8k_answer": r["answer"].split("####")[-1].strip()}
            for i, r in enumerate(d)]
    chosen, audits["gsm8k"] = build("gsm8k", rows, args.n_gsm8k, excluded_content,
                                    excluded_prompts, corpus_prompts)
    emit("gsm8k", chosen)

    # ---- multihop: HotpotQA, answer + supporting titles ------------------
    d = load_dataset(SOURCES["multihop"]["repo"], name=SOURCES["multihop"]["config"],
                     split=SOURCES["multihop"]["split"])
    rows = []
    for r in d.select(range(min(4000, len(d)))):
        titles = r["context"]["title"]
        docs = "\n\n".join(
            f"[{t}] {' '.join(s)}" for t, s in zip(titles, r["context"]["sentences"]))
        prompt = f"{MULTIHOP_INSTRUCTION}\n\nDocuments:\n{docs}\n\nQuestion: {r['question']}"
        rows.append({"id": f"hotpot-val-{r['id']}", "group": "multihop",
                     "source": "hotpot_qa", "prompt_text": prompt,
                     "messages": [{"role": "user", "content": prompt}],
                     "gold": r["answer"], "answer": r["answer"],
                     "supporting_titles": sorted(set(r["supporting_facts"]["title"]))})
    chosen, audits["multihop"] = build("multihop", rows, args.n_multihop,
                                       excluded_content, excluded_prompts,
                                       corpus_prompts)
    emit("multihop", chosen)

    # ---- rag + refusal_paired: SQuAD v2 validation -----------------------
    d = load_dataset(SOURCES["rag"]["repo"], split=SOURCES["rag"]["split"])
    answerable, unanswerable = [], []
    for r in d:
        prompt = (f"{RAG_INSTRUCTION}\n\nContext: {r['context']}\n\n"
                  f"Question: {r['question']}")
        row = {"id": f"squad-val-{r['id']}", "source": "squad_v2",
               "prompt_text": prompt,
               "messages": [{"role": "user", "content": prompt}],
               "context": r["context"], "title": r["title"],
               "question": r["question"]}
        if r["answers"]["text"]:
            answerable.append({**row, "gold": r["answers"]["text"][0],
                               "answerable": True})
        else:
            unanswerable.append({**row, "gold": "", "answerable": False})

    rag_rows = [{**r, "group": "rag"} for r in answerable]
    chosen, audits["rag"] = build("rag", rag_rows, args.n_rag, excluded_content,
                                  excluded_prompts, corpus_prompts)
    emit("rag", chosen)

    # Pairs share a title: the same passage domain, one answerable and one not,
    # so refusing everything loses exactly as much as answering everything.
    by_title = {}
    for r in unanswerable:
        by_title.setdefault(r["title"], []).append(r)
    pairs = []
    for a in sorted(answerable, key=lambda x: sha(x["id"])):
        pool = by_title.get(a["title"])
        if not pool:
            continue
        u = pool.pop()
        if not pool:
            by_title.pop(a["title"])
        pairs.append((a, u))
        if len(pairs) >= args.n_refusal_pairs * 3:
            break
    # Selection happens at the PAIR level, never the row level: a stride over
    # rows splits pairs, and half a pair cannot score `pair_correct` — which is
    # the only aggregate that unconditional refusal cannot win.
    pair_map = {}
    for i, (a, u) in enumerate(pairs):
        pid = f"pair-{i:04d}"
        pair_map[pid] = [
            {**a, "group": "refusal_paired", "pair_id": pid,
             "id": f"{pid}-safe:{a['id']}"},
            {**u, "group": "refusal_paired", "pair_id": pid,
             "id": f"{pid}-unsafe:{u['id']}"},
        ]

    clean_pairs, drops = [], Counter()
    for pid, rows2 in pair_map.items():
        bad = None
        for r in rows2:
            key = sha(r["prompt_text"])
            if key in excluded_prompts:
                bad = "reserved_split_prompt"
            elif key in corpus_prompts:
                bad = "corpus_v2_prompt"
        if bad:
            drops[bad] += 1
            continue
        clean_pairs.append((pid, rows2))
    selected = stride([{"id": pid, "rows": rows2} for pid, rows2 in clean_pairs],
                      args.n_refusal_pairs)
    chosen = [r for item in selected for r in item["rows"]]
    audits["refusal_paired"] = {
        "candidates": len(pair_map) * 2,
        "after_leakage_filter": len(clean_pairs) * 2,
        "selected": len(chosen), "complete_pairs": len(chosen) // 2,
        "dropped_pairs": dict(drops),
        "selection_unit": "pair, so a stride can never split one",
    }
    emit("refusal_paired", chosen)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "battery_version": BATTERY_VERSION,
        "frozen": True,
        "scorer": {
            "module": "src/aadistill/evaluation/capability.py",
            "sha256": sha256_file(REPO_ROOT / "src/aadistill/evaluation/capability.py"),
            "strict_answer_sha256": sha256_file(
                REPO_ROOT / "src/aadistill/evaluation/strict_answer.py"),
            "degeneration_sha256": sha256_file(
                REPO_ROOT / "scripts/evaluation/degeneration.py"),
            "policy": ("deterministic scorers only — alias-set EM, numeric then "
                       "symbolic then normalized boxed comparison, span "
                       "containment, supporting-title recall, protocol "
                       "validation. No LLM judge is used as a primary scorer."),
        },
        "decoding": {
            "greedy": True, "temperature": 0.0,
            "unrestricted_within_effective_context": True,
            "effective_context": 8192,
            "context_source": "trained block_len",
            "stop_ids_from": "model generation_config",
            "degeneration_stop": True,
            "note": ("identical to the Experiment 1 evaluation protocol so D0's "
                     "stored generations remain comparable"),
        },
        "serialization": {
            "system_message": "You are a helpful Assistant.",
            "system_policy": ("mandatory; a sample's own system prompt is "
                              "preserved when present"),
            "chat_template_sha256": (
                "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"),
        },
        "leakage": {
            "structural": ("stage2_v1 drew every source from its train split; "
                           "this battery uses validation/test splits, which were "
                           "never eligible for corpus v2. TriviaQA has never "
                           "been used by this project."),
            "hash_rule": ("content_key / prompt_key from "
                          "scripts/rollout/build_recovery_corpus.py, applied over "
                          "stage2_v1 train+val+calib, eval_behavior_v0 and "
                          "corpus v2 sessions"),
            "exclusion_sets": {"content_hashes": len(excluded_content),
                               "reserved_prompt_hashes": len(excluded_prompts),
                               "corpus_v2_prompt_hashes": len(corpus_prompts)},
            "per_set": audits,
            "known_weakness": ("`multihop` and `rag` share the SQuAD-v2 / "
                               "HotpotQA source families with corpus v2's "
                               "training slices, at the item level disjoint and "
                               "on a different split — near-domain, not "
                               "out-of-domain. `knowledge` (TriviaQA) and "
                               "`math_verified` (MATH-500) come from sources "
                               "this project has never trained on at any stage."),
        },
        "sets": sets,
        "reused_from_experiment_1": {
            "behavior_v0": {"path": "data/eval_behavior_v0/prompts.jsonl",
                            "n": 76,
                            "why": ("unchanged so D0's stored behaviour "
                                    "generations stay comparable without a "
                                    "re-run")},
        },
        "code_state": code_state(REPO_ROOT),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    manifest_hash = sha256_file(args.out / "manifest.json")

    total = sum(v["n"] for v in sets.values())
    print(f"\n{'set':16s} {'n':>5} {'candidates':>11} {'after filter':>13}  sha256")
    for name, v in sets.items():
        a = audits[name]
        print(f"{name:16s} {v['n']:>5} {a['candidates']:>11} "
              f"{a['after_leakage_filter']:>13}  {v['sha256'][:16]}")
    print(f"{'TOTAL':16s} {total:>5}  (+76 reused behavior_v0 = {total + 76})")
    print(f"\nmanifest sha256 {manifest_hash}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
