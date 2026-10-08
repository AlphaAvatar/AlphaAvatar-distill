#!/usr/bin/env python
"""Build a cleaned recovery corpus by re-selecting from retained candidates.

Reads corpus v2's `candidates.jsonl` (all `n=4` sampled completions per prompt,
with token ids and verdicts) plus its `sessions.jsonl` (which supplies the
prompt context, tool schema and the target the corpus actually selected), and
writes a `sessions_clean.jsonl` in the identical schema — so the existing
`build_token_ladder.py` consumes it with no change.

**Nothing is generated.** Every emitted target is a completion the teacher
already produced under the recorded preset, so this script is CPU-only and adds
no teacher-inference cost. Because KD teacher distributions are computed online
during training (`aadistill.training.train`, no cached logits), a changed target
also needs no logit recomputation: the teacher simply runs on the new packed
blocks.

Rules and their order live in `aadistill.data.cleaning`; this driver supplies
the tokenizer-dependent parts (rendering, supervised-token length) and the
audit.

Usage:
    scripts/data/build_cleaned_corpus.py \
        --corpus <dir with candidates.jsonl + sessions.jsonl + manifest.json> \
        --tokenizer <teacher path or repo@revision> \
        --out artifacts/stage3/corpus_v2_clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation import degeneration  # noqa: E402

from aadistill.data.cleaning import (  # noqa: E402
    RULES_VERSION,
    SELECTION_RULES,
    is_verifiable,
    select_clean,
)
from aadistill.data.sessions import render_session  # noqa: E402
from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402
from aadistill.models.teacher import tokenizer_hash  # noqa: E402

BLOCK_LEN = 8192


def load_tokenizer(spec: str):
    from transformers import AutoTokenizer

    if "@" in spec and not Path(spec).exists():
        repo, revision = spec.split("@", 1)
        return AutoTokenizer.from_pretrained(repo, revision=revision)
    return AutoTokenizer.from_pretrained(spec)


def session_for(example: dict, context: dict, candidate: dict) -> dict:
    """The session dict a chosen candidate produces, in sessions.jsonl schema."""
    return {
        "id": example["id"],
        "source_id": example["source_id"],
        "data_type": example["data_type"],
        "group": example["group"],
        "source": example["source"],
        "turn_index": example["turn_index"],
        "predecessor_role": context["predecessor_role"],
        "n_context_assistant_turns": example["n_context_assistant_turns"],
        "messages": context["context"] + [{
            "role": "assistant",
            "reasoning_content": candidate["think"],
            "content": candidate["answer"],
        }],
        "tools": context["tools"],
        "candidate_index": candidate["index"],
        "candidate_seed": candidate["seed"],
        "candidate_sha256": hashlib.sha256(candidate["raw"].encode()).hexdigest(),
        "correct": candidate["correct"],
        "correctness_verdict": candidate["correctness_verdict"],
        "gold": example["gold"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--block-len", type=int, default=BLOCK_LEN)
    ap.add_argument("--selection", default="median", choices=list(SELECTION_RULES),
                    help="which survivor replaces a failing original: 'median' "
                         "(the survivor closest to the median supervised-token "
                         "length) or 'shortest'. Length is consulted only among "
                         "candidates that already passed every gate.")
    ap.add_argument("--limit", type=int, default=None,
                    help="screen only the first N examples (smoke tests)")
    args = ap.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)

    # sessions.jsonl supplies the prompt context; an example the corpus rejected
    # outright cannot survive cleaning either, because every cleaning gate is a
    # superset of the hygiene gate that rejected it.
    context_of: dict[str, dict] = {}
    original_index: dict[str, int] = {}
    original_supervised: dict[str, int] = {}
    with (args.corpus / "sessions.jsonl").open() as f:
        for line in f:
            s = json.loads(line)
            context_of[s["id"]] = {
                "context": s["messages"][:-1],
                "tools": s.get("tools"),
                "predecessor_role": s.get("predecessor_role"),
            }
            original_index[s["id"]] = s["candidate_index"]
            original_supervised[s["id"]] = s["n_supervised_tokens"]

    stats = defaultdict(lambda: {
        "examples": 0, "kept": 0, "retained_original": 0, "replaced": 0,
        "no_valid_candidate": 0, "unverifiable": 0,
        "reasons": Counter(), "first_stage": Counter(),
        "sup_before": [], "sup_after": [],
        "answer_words_before": [], "answer_words_after": [],
    })
    per_example = []
    n_seen = 0
    render_cache: dict[tuple[str, int], int | None] = {}

    out_sessions = (args.out / "sessions_clean.jsonl").open("w")
    with (args.corpus / "candidates.jsonl").open() as f:
        for line in f:
            example = json.loads(line)
            eid = example["id"]
            context = context_of.get(eid)
            if context is None:
                continue  # corpus rejected every candidate; cleaning cannot revive it
            n_seen += 1
            if args.limit and n_seen > args.limit:
                break
            # candidates.jsonl carries no tool schema — it lives on the session.
            # Without this the tool gate sees "no schema declared" and rejects
            # every legitimate call as unlicensed.
            example["tools"] = context["tools"]
            dtype = example["data_type"]
            st = stats[dtype]
            st["examples"] += 1
            if not is_verifiable(example):
                st["unverifiable"] += 1

            def length_of(candidate, _eid=eid, _ctx=context, _ex=example):
                """Supervised-token length of the session this candidate makes."""
                key = (_eid, candidate["index"])
                if key not in render_cache:
                    try:
                        rendered = render_session(
                            tokenizer, session_for(_ex, _ctx, candidate),
                            block_len=args.block_len)
                        render_cache[key] = rendered.n_supervised
                    except ValueError:
                        render_cache[key] = None  # render_overflow / drift
                return render_cache[key] if render_cache[key] is not None else 1 << 30

            verdict = select_clean(example, degeneration, length_of,
                                   original_index[eid], rule=args.selection)

            for index, reason in verdict["reasons"].items():
                st["reasons"][reason] += 1
                st["first_stage"][reason.split(":")[0]] += 1

            chosen = verdict["chosen"]
            # A survivor that cannot be rendered is not usable; fall through to
            # the next survivor rather than emitting a session the packer rejects.
            while chosen is not None and length_of(chosen) >= (1 << 30):
                st["reasons"]["render_overflow"] += 1
                example = {**example, "candidates": [
                    c for c in example["candidates"] if c["index"] != chosen["index"]]}
                verdict = select_clean(example, degeneration, length_of,
                                       original_index[eid], rule=args.selection)
                chosen = verdict["chosen"]

            if chosen is None:
                st["no_valid_candidate"] += 1
                per_example.append({"id": eid, "data_type": dtype, "kept": False,
                                    "reasons": verdict["reasons"]})
                continue

            session = session_for(example, context, chosen)
            rendered = render_session(tokenizer, session, block_len=args.block_len)
            session["n_rendered_tokens"] = rendered.n_rendered_tokens
            session["n_supervised_tokens"] = rendered.n_supervised
            session["n_system_tokens"] = rendered.n_system_tokens
            session["system_key"] = rendered.system_key
            out_sessions.write(json.dumps(session) + "\n")

            st["kept"] += 1
            st["retained_original"] += int(verdict["retained_original"])
            st["replaced"] += int(not verdict["retained_original"])
            st["sup_before"].append(original_supervised[eid])
            st["sup_after"].append(rendered.n_supervised)
            original = next(c for c in example["candidates"]
                            if c["index"] == original_index[eid])
            st["answer_words_before"].append(len(original["answer"].split()))
            st["answer_words_after"].append(len(chosen["answer"].split()))
            per_example.append({
                "id": eid, "data_type": dtype, "kept": True,
                "original_index": original_index[eid],
                "chosen_index": chosen["index"],
                "retained_original": verdict["retained_original"],
                "n_survivors": verdict["n_survivors"],
                "survivor_lengths": verdict["survivor_lengths"],
                "selection_rule": args.selection,
                "sup_before": original_supervised[eid],
                "sup_after": rendered.n_supervised,
                "reasons": verdict["reasons"],
            })
    out_sessions.close()

    def dist(values):
        if not values:
            return None
        s = sorted(values)
        return {"n": len(s), "mean": round(statistics.mean(s), 1),
                "p50": s[len(s) // 2], "p90": s[int(len(s) * 0.9)],
                "p99": s[int(len(s) * 0.99)], "max": s[-1], "sum": sum(s)}

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rules_version": RULES_VERSION,
        "selection_rule": {
            "name": args.selection,
            "applies": "only among candidates that passed every cleaning gate",
            "length_unit": ("assistant supervised tokens after exact chat "
                            "serialization (render_session), not characters and "
                            "not raw pre-template tokens"),
            "retain_original": ("the corpus's own candidate is kept whenever it "
                                "passes every gate; length is consulted only "
                                "when it fails"),
            "tie_break": "original candidate index",
        },
        "command": " ".join(sys.argv),
        "source_corpus": {
            "dir": str(args.corpus),
            "candidates_sha256": sha256_file(args.corpus / "candidates.jsonl"),
            "sessions_sha256": sha256_file(args.corpus / "sessions.jsonl"),
        },
        "tokenizer": {
            "spec": args.tokenizer,
            "vocab_sha256": hashlib.sha256(
                json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
            "vocab_hash": tokenizer_hash(tokenizer),
            "chat_template_sha256": hashlib.sha256(
                tokenizer.get_chat_template().encode()).hexdigest(),
        },
        "degeneration_detector": {
            "module": "scripts/evaluation/degeneration.py",
            "sha256": sha256_file(REPO_ROOT / "scripts/evaluation/degeneration.py"),
            "note": ("stricter than the corpus build: the `rambling` "
                     "novel-n-gram signal did not exist on 2026-08-01"),
        },
        "block_len": args.block_len,
        "per_type": {},
        "code_state": code_state(REPO_ROOT),
        "hardware": hardware_report(),
        "elapsed_s": round(time.time() - started, 1),
    }
    for dtype, st in sorted(stats.items()):
        audit["per_type"][dtype] = {
            "examples": st["examples"],
            "kept": st["kept"],
            "keep_rate": round(st["kept"] / st["examples"], 4) if st["examples"] else 0.0,
            "retained_original": st["retained_original"],
            "replaced": st["replaced"],
            "replacement_rate": (round(st["replaced"] / st["kept"], 4)
                                 if st["kept"] else 0.0),
            "no_valid_candidate": st["no_valid_candidate"],
            "unverifiable_slice": st["unverifiable"],
            "candidate_reasons": dict(st["reasons"].most_common()),
            "candidate_first_failed_stage": dict(st["first_stage"].most_common()),
            "supervised_before": dist(st["sup_before"]),
            "supervised_after": dist(st["sup_after"]),
            "answer_words_before": dist(st["answer_words_before"]),
            "answer_words_after": dist(st["answer_words_after"]),
        }
    (args.out / "cleaning_audit.json").write_text(json.dumps(audit, indent=1))
    with (args.out / "cleaning_per_example.jsonl").open("w") as f:
        for row in per_example:
            f.write(json.dumps(row) + "\n")

    total_ex = sum(s["examples"] for s in stats.values())
    total_kept = sum(s["kept"] for s in stats.values())
    print(f"screened {total_ex} examples -> kept {total_kept} "
          f"({total_kept / total_ex:.1%}) in {audit['elapsed_s']}s")
    for dtype, row in sorted(audit["per_type"].items()):
        print(f"  {dtype:14s} kept {row['kept']:5d}/{row['examples']:5d} "
              f"({row['keep_rate']:.3f})  replaced {row['replaced']:5d} "
              f"({row['replacement_rate']:.3f})  "
              f"sup {row['supervised_before']['sum'] if row['supervised_before'] else 0:>9,}"
              f" -> {row['supervised_after']['sum'] if row['supervised_after'] else 0:>9,}")


if __name__ == "__main__":
    main()
