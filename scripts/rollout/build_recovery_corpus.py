"""Build the maximal reusable teacher corpus for the recovery-data scaling study.

    uv run python scripts/rollout/build_recovery_corpus.py \
        --engine vllm --limits 'gsm8k=1700,...' --out artifacts/stage3/corpus_v2/bulk

Distinct from `generate_teacher_answers.py`, which stays as the record of the
pinned `teacher_corpus_750` artifact it produced. That script selects a
*verified-correct* target and falls back to the public one; this corpus is
hygiene-selected, has no fallback, retains all four candidates, is bounded by an
8,192-token end-to-end session limit, and turn-expands multi-turn sources.

What this enforces, and why each one is load-bearing
----------------------------------------------------
* **8,192 tokens end to end.** The completion budget is derived *per example* as
  `8192 - rendered_prompt - allowance`, never a flat cap: a flat cap lets
  long-context examples overrun the limit and be rejected after they were paid
  for. A candidate that reaches its budget, fails to terminate, or does not fit
  once re-rendered is **rejected**, never trimmed.

* **Turn expansion.** A multi-turn source `(s, u1, a1ᵒ, u2, a2ᵒ, u3, a3ᵒ)` becomes
  three independent generation examples — `(s, u1, a1ᵗ)`,
  `(s, u1, a1ᵒ, u2, a2ᵗ)`, `(s, u1, a1ᵒ, u2, a2ᵒ, u3, a3ᵗ)` — one per eligible
  turn. In each, only the newly generated `aᵗ` is supervised; the system prompt,
  user messages, template tokens and every preceding *original* `aᵒ` are context
  only, masked from loss and from supervised-token accounting by
  `final_assistant_loss_mask`. This is what makes multi-turn agentic data usable
  without mixing public and teacher targets in one block (P17).

  It also lines up with the chat template: it renders `<think>` only for the
  assistant turn after the last user message, which under turn expansion is
  exactly the teacher-generated turn. The preceding `aᵒ` are public and carry no
  reasoning, so nothing is lost.

* **The officially recommended sampling preset**, `temperature 0.6 / top_p 0.95 /
  top_k 20 / min_p 0`, replacing the untruncated preset the 2026-07-29 record
  chose for rejection sampling. That choice existed to feed a *correctness*
  verifier with diverse candidates; this corpus is hygiene-selected, so the
  vendor preset applies.

* **`n=4` independently seeded candidates.** Seeds are spaced by a large stride
  per candidate index, because a serving engine seeds per *request*: replicating
  a prompt inside one request decodes it identically and makes n>1 a silent
  no-op (measured 2026-07-30 — 92.7% byte-identical pairs).

* **Every candidate is kept**, accepted or not, with its rejection reason, seed,
  termination status, token ids and correctness verdict — the retained material
  for the later candidate-diversity/difficulty work.

* **Acceptance is hygiene, not correctness.** Correctness verdicts are computed
  and stored, never acted on (decision 2026-07-28: Stage 3 trains the teacher's
  unfiltered distribution; selection is Stage 4/5).

  One deliberate departure from `verify.hygiene_reason`: its `too_long` rule
  rejects any answer over `MAX_ANSWER_WORDS = 600`. That is a generic word-count
  limit, which AGENTS.md P3/P10 forbid as a framework-level gate, and applying it
  here would bias the corpus against exactly the long derivations this study
  exists to supervise. Structural hygiene is applied; the length rule is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation import degeneration  # noqa: E402

from aadistill.data.dataset import load_jsonl  # noqa: E402
from aadistill.data.sessions import (  # noqa: E402
    SYSTEM_DEFAULT,
    RENDER_ALLOWANCE,
    completion_budget,
    generation_prompt,
    render_session,
    split_system,
)
from aadistill.data.verify import STRAY_MARKERS, verify  # noqa: E402
from aadistill.evaluation.behavior import THINK_CLOSE, split_generation  # noqa: E402
from aadistill.infrastructure.env import code_state, hardware_report  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

# Data type -> selection rule.
#
# Scope follows the README's declared objectives — reasoning and self-correction,
# RAG, tool use, code and math — weighted by measured difficulty rather than
# equally (maintainer, 2026-07-31).
#
# `tool_calling` enters the mixture for the first time: it is the single largest
# behaviour-axis gap to the teacher (+0.667) and is named in the README's agentic
# objective. Turn expansion makes its 4,375 multi-turn conversations usable, which
# is the "long-running agentic workload" the README targets.
#
# `long_context` is **excluded**: that group is `format: "text"` (raw fineweb-edu
# documents, no messages), so there is no question for the teacher to answer.
# Including it would require synthesizing prompts over documents — a new
# data-construction experiment, not a mixture choice.
#
# `refusal_uncertainty`, `instruction` and `short_realtime` remain out of scope
# (decision 2026-07-30): refusal is an alignment-tax slice nothing has justified,
# and the other two are general conversation rather than the declared capability
# target. Multi-turn coverage comes from `tool_calling`, which is both multi-turn
# and on-target.
TYPES = {
    "rag_evidence": {"group": "rag_evidence", "sources": ("squad_v2",)},
    "multihop_qa": {"group": "multihop_qa", "sources": ("hotpot_qa",)},
    "gsm8k": {"group": "code_math", "sources": ("gsm8k",)},
    "openmath": {"group": "code_math", "sources": ("openmath_instruct_2",)},
    "code": {"group": "code_math", "sources": ("magicoder_oss", "mbpp")},
    "tool_calling": {"group": "tool_calling",
                     "sources": ("glaive_fc_v2", "xlam_fc_60k")},
}

# Reserved splits that must never contribute a training prompt. The Stage 2 build
# already deduplicated globally and excluded the holdout, but a direct check
# found 27 train rows duplicating `val` and 4 duplicating `calib`, so the
# exclusion is recomputed here rather than trusted.
DEFAULT_EXCLUDE = (
    "data/stage2_v1/val",
    "data/stage2_v1/calib",
    "data/stage2/val",
    "data/warmup/holdout_v1.jsonl",
    "data/eval_behavior_v0/prompts.jsonl",
)

# Larger than any plausible batch count, so candidate i of batch b never reuses
# candidate j of batch b'.
SEED_STRIDE = 1_000_003

IM_END = "<|im_end|>"


def content_key(messages) -> str:
    """Content hash used for dedup and leakage exclusion; ids are excluded."""
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prompt_key(messages) -> str | None:
    """Hash of the first user message — catches reuse of a prompt with a new target."""
    for message in messages:
        if message.get("role") == "user":
            return hashlib.sha256(message.get("content", "").encode()).hexdigest()
    return None


def load_excluded(paths) -> tuple[set[str], set[str]]:
    """Content and first-user-message hashes of every reserved example."""
    content, prompts = set(), set()
    for spec in paths:
        p = REPO_ROOT / spec
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else ([p] if p.exists() else [])
        for f in files:
            for line in open(f):
                if not line.strip():
                    continue
                sample = json.loads(line)
                messages = sample.get("messages")
                if messages:
                    content.add(content_key(messages))
                    key = prompt_key(messages)
                    if key:
                        prompts.add(key)
                else:
                    content.add(hashlib.sha256(
                        (sample.get("text") or "").encode()).hexdigest())
    return content, prompts


def expand_turns(sample: dict, data_type: str) -> list[dict]:
    """One independent generation example per eligible assistant turn.

    An assistant turn is eligible when it is preceded by a `user` or `tool`
    message — both are points where the model must produce the next assistant
    message, and a turn after a tool response is where tool-output use is
    actually exercised.
    """
    messages = sample.get("messages") or []
    system_text, _ = split_system(messages)
    has_system = bool(messages) and messages[0].get("role") == "system"
    examples = []
    for i, message in enumerate(messages):
        if message.get("role") != "assistant" or i == 0:
            continue
        predecessor = messages[i - 1].get("role")
        if predecessor not in ("user", "tool"):
            continue
        context = list(messages[:i])
        if not has_system:
            context = [{"role": "system", "content": system_text}] + context
        examples.append({
            "example_id": f"{sample['id']}#t{i}",
            "source_id": sample["id"],
            "data_type": data_type,
            "group": sample["group"],
            "source": sample["source"],
            "context": context,
            "gold": message.get("content", ""),
            "turn_index": i,
            "predecessor_role": predecessor,
            "n_context_assistant_turns": sum(
                1 for m in messages[:i] if m.get("role") == "assistant"),
            "tools": sample.get("tools"),
        })
    return examples


def load_examples(data_dir: Path, name: str, limit: int | None, select: str,
                  excluded_content: set[str], excluded_prompts: set[str],
                  seen: set[str]) -> tuple[list[dict], dict]:
    """Expand, filter and take `limit` generation examples for one data type.

    Filtering happens **before** the take, so a requested count is a count of
    usable examples rather than of raw rows minus whatever the filters removed.
    Leakage is checked at the *source conversation* level — if a conversation
    appears in any reserved split, none of its turns may be used — and `seen`
    carries prefix hashes across types so the same conversation prefix cannot
    enter twice through two sources.
    """
    spec = TYPES[name]
    path = data_dir / "train" / f"{spec['group']}.jsonl"
    raw = [s for s in load_jsonl(path) if s["source"] in spec["sources"]]

    stats = {"raw_conversations": len(raw), "dropped_no_messages": 0,
             "dropped_reserved_conversation": 0, "expanded": 0,
             "dropped_duplicate": 0, "turns_per_conversation": Counter()}
    eligible: list[dict] = []
    for sample in raw:
        messages = sample.get("messages")
        if not messages:
            stats["dropped_no_messages"] += 1
            continue
        pkey = prompt_key(messages)
        if content_key(messages) in excluded_content or (
                pkey and pkey in excluded_prompts):
            stats["dropped_reserved_conversation"] += 1
            continue
        examples = expand_turns(sample, name)
        stats["expanded"] += len(examples)
        stats["turns_per_conversation"][len(examples)] += 1
        for example in examples:
            # Key on the conversation prefix *including* the target turn: two
            # examples with the same context but different targets are different
            # training signals and both may stand.
            key = content_key(example["context"] +
                              [{"role": "assistant", "content": example["gold"]}])
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(key)
            eligible.append(example)

    stats["eligible"] = len(eligible)
    stats["turns_per_conversation"] = dict(sorted(stats["turns_per_conversation"].items()))
    rows = eligible
    if limit and limit < len(rows):
        rows = (rows[:limit] if select == "prefix"
                else rows[:: len(rows) // limit][:limit])
    if not rows:
        raise ValueError(f"no eligible examples for type {name} in {path}")
    stats["taken"] = len(rows)
    stats["multi_turn_taken"] = sum(1 for r in rows if r["n_context_assistant_turns"])
    return rows, stats


def structural_hygiene(answer: str, raw: str) -> str | None:
    """Structural rejections only — no generic length rule (see module docstring)."""
    if not answer.strip():
        return "empty_answer"
    if IM_END not in raw:
        return "not_terminated"
    if any(marker in answer for marker in STRAY_MARKERS):
        return "stray_marker"
    if THINK_CLOSE not in raw:
        return "think_not_closed"
    return None


def stop_ids(generation_config, tokenizer) -> set[int]:
    """Every id that ends a turn, taken from the model's own generation config."""
    configured = generation_config.eos_token_id
    ids = set(configured if isinstance(configured, (list, tuple)) else [configured])
    ids.add(tokenizer.eos_token_id)
    return {i for i in ids if i is not None}


def build_engine(args, tokenizer):
    from aadistill.rollout.engines import HFEngine, SGLangEngine, VLLMEngine
    from aadistill.rollout.engines import SGLangServerEngine, VLLMServerEngine

    spec = args.model
    path, revision = (spec.split("@", 1) if "@" in spec else (spec, None))
    if args.engine == "vllm":
        # `max_model_len` is the session limit: it bounds the KV cache per
        # sequence at 8,192 x 144 KiB = 1.18 GB, which is what lets a 48 GB card
        # hold enough concurrent sequences to stay saturated.
        return VLLMEngine(path, dtype=args.dtype, revision=revision,
                          max_model_len=args.block_len)
    if args.engine == "sglang":
        return SGLangEngine(path, dtype=args.dtype, revision=revision)
    if args.engine in ("vllm_server", "sglang_server"):
        if not args.server_url:
            raise SystemExit(f"--engine {args.engine} requires --server-url")
        if args.engine == "vllm_server":
            return VLLMServerEngine(args.server_url, model=path)
        return SGLangServerEngine(args.server_url)

    import torch
    from aadistill.models.teacher import load_causal_lm

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model, _ = load_causal_lm(args.model, dtype, device)
    return HFEngine(model, tokenizer.pad_token_id, batch_size=args.batch_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@"
                                       "768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--data-dir", default="data/stage2_v1")
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--limit-per-type", type=int, default=None)
    ap.add_argument("--limits", default=None,
                    help="per-type example counts, e.g. 'gsm8k=1700,rag_evidence=4100'")
    ap.add_argument("--mixture-note", default=None,
                    help="free-text rationale recorded in the manifest")
    ap.add_argument("--exclude-from", default=",".join(DEFAULT_EXCLUDE))
    ap.add_argument("--select", default="stride", choices=["prefix", "stride"])
    ap.add_argument("--n", type=int, default=4, help="independent candidates per example")
    ap.add_argument("--block-len", type=int, default=8192,
                    help="end-to-end session limit, prompt + template + completion")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--engine", default="vllm",
                    choices=["vllm", "sglang", "vllm_server", "sglang_server", "hf"])
    ap.add_argument("--server-url", default=None)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock budget; stops cleanly at a batch boundary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, GenerationConfig
    from transformers import __version__ as transformers_version

    names = [s.strip() for s in args.types.split(",") if s.strip()]
    unknown = set(names) - set(TYPES)
    if unknown:
        raise SystemExit(f"unknown type(s): {sorted(unknown)}; known: {sorted(TYPES)}")

    limits = {}
    if args.limits:
        for item in args.limits.split(","):
            key, _, value = item.partition("=")
            key = key.strip()
            if key not in TYPES:
                raise SystemExit(f"--limits names unknown type {key!r}")
            limits[key] = int(value)

    spec = args.model
    path, revision = (spec.split("@", 1) if "@" in spec else (spec, None))
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    generation_config = GenerationConfig.from_pretrained(path, revision=revision)
    stops = stop_ids(generation_config, tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    think_close_id = tokenizer.convert_tokens_to_ids(THINK_CLOSE)

    exclude_paths = [p.strip() for p in args.exclude_from.split(",") if p.strip()]
    excluded_content, excluded_prompts = load_excluded(exclude_paths)
    print(f"reserved: {len(excluded_content)} content hashes, "
          f"{len(excluded_prompts)} prompt hashes", flush=True)

    data_dir = REPO_ROOT / args.data_dir
    seen: set[str] = set()
    work, selection_stats = {}, {}
    for name in names:
        rows, stats = load_examples(
            data_dir, name, limits.get(name, args.limit_per_type), args.select,
            excluded_content, excluded_prompts, seen)
        work[name] = rows
        selection_stats[name] = stats
        print(f"  {name:<14} conv {stats['raw_conversations']:>6}  "
              f"expanded {stats['expanded']:>6}  eligible {stats['eligible']:>6}  "
              f"taken {stats['taken']:>6}  (multi-turn ctx {stats['multi_turn_taken']}, "
              f"reserved-conv {stats['dropped_reserved_conversation']}, "
              f"dup {stats['dropped_duplicate']})", flush=True)

    total = sum(len(v) for v in work.values())
    print(f"{total} generation examples across {len(work)} types, n={args.n}, "
          f"session limit {args.block_len}, engine {args.engine}", flush=True)

    engine = build_engine(args, tokenizer)
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    stats = {name: {"examples": 0, "accepted": 0, "reasons": Counter(),
                    "supervised": [], "rendered": [], "new_tokens": [],
                    "correct": 0, "verdicts": Counter()}
             for name in names}
    skipped_no_budget: list[str] = []
    done = 0
    budget_stopped = False

    with open(out_dir / "candidates.jsonl", "w") as f_cand, \
            open(out_dir / "sessions.jsonl", "w") as f_sess:
        batches = [(name, rows[i: i + args.batch_size])
                   for name, rows in work.items()
                   for i in range(0, len(rows), args.batch_size)]

        for batch_index, (name, batch) in enumerate(batches):
            if args.max_hours and (time.time() - started) / 3600 >= args.max_hours:
                budget_stopped = True
                print(f"\n!! wall-clock budget {args.max_hours}h reached after "
                      f"{done}/{total} examples — stopping cleanly", flush=True)
                break

            prompts, budgets, kept = [], [], []
            for example in batch:
                budget = completion_budget(
                    tokenizer, example["context"], example.get("tools"),
                    block_len=args.block_len, allowance=RENDER_ALLOWANCE)
                if budget <= 0:
                    skipped_no_budget.append(example["example_id"])
                    stats[name]["reasons"]["prompt_exceeds_session_limit"] += 1
                    continue
                rendered = generation_prompt(
                    tokenizer, example["context"], example.get("tools"))
                prompts.append(tokenizer(rendered, add_special_tokens=False).input_ids)
                budgets.append(budget)
                kept.append((example, rendered))
            if not kept:
                continue

            per_prompt: list[list[dict]] = [[] for _ in kept]
            for candidate_index in range(args.n):
                seed = args.seed + batch_index + candidate_index * SEED_STRIDE
                completions = engine.generate(
                    list(prompts), max_new_tokens=list(budgets), stop_ids=stops,
                    greedy=False, temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, min_p=args.min_p, seed=seed)
                for i, completion in enumerate(completions):
                    per_prompt[i].append({**completion, "seed": seed,
                                          "index": candidate_index})

            for (example, rendered_prompt), prompt_ids, budget, raws in zip(
                    kept, prompts, budgets, per_prompt):
                candidates = []
                for item in raws:
                    ids = item["tokens"]
                    raw = tokenizer.decode(ids, skip_special_tokens=False)
                    parts = split_generation(raw)
                    answer, think = parts["answer"], parts["think"]

                    length_limited = bool(
                        item["hit_cap"] or item.get("over_budget")
                        or not item["finished"])
                    reason = "length_limited" if length_limited else \
                        structural_hygiene(answer, raw)
                    degen = degeneration.check(ids) if reason is None else None
                    if reason is None and degen:
                        reason = "degenerate"

                    shim = {"group": example["group"], "source": example["source"],
                            "messages": [{"role": "assistant",
                                          "content": example["gold"]}]}
                    correct, verdict = verify(shim, answer, raw)
                    think_tokens = (ids.index(think_close_id)
                                    if think_close_id in ids else item["n_new"])
                    candidates.append({
                        "index": item["index"], "seed": item["seed"],
                        "accepted": reason is None, "reason": reason or "ok",
                        "answer": answer, "think": think, "raw": raw,
                        "tokens": ids, "new_tokens": item["n_new"],
                        "budget": budget, "hit_cap": item["hit_cap"],
                        "over_budget": bool(item.get("over_budget")),
                        "finished": item["finished"],
                        "length_limited": length_limited,
                        "think_tokens": think_tokens, "degeneration": degen,
                        "correct": correct, "correctness_verdict": verdict,
                    })

                slice_stats = stats[name]
                slice_stats["examples"] += 1
                for candidate in candidates:
                    slice_stats["reasons"][candidate["reason"]] += 1
                    slice_stats["verdicts"][candidate["correctness_verdict"]] += 1
                    slice_stats["new_tokens"].append(candidate["new_tokens"])
                    slice_stats["correct"] += int(candidate["correct"])

                # Deterministic selection: the lowest-index accepted candidate.
                chosen = next((c for c in candidates if c["accepted"]), None)
                session = None
                if chosen is not None:
                    session = {
                        "id": example["example_id"],
                        "source_id": example["source_id"],
                        "data_type": name,
                        "group": example["group"],
                        "source": example["source"],
                        "turn_index": example["turn_index"],
                        "predecessor_role": example["predecessor_role"],
                        "n_context_assistant_turns": example["n_context_assistant_turns"],
                        "messages": example["context"] + [{
                            "role": "assistant",
                            "reasoning_content": chosen["think"],
                            "content": chosen["answer"],
                        }],
                        "tools": example.get("tools"),
                        "candidate_index": chosen["index"],
                        "candidate_seed": chosen["seed"],
                        "candidate_sha256": hashlib.sha256(
                            chosen["raw"].encode()).hexdigest(),
                        "correct": chosen["correct"],
                        "correctness_verdict": chosen["correctness_verdict"],
                        "gold": example["gold"],
                    }
                    try:
                        r = render_session(tokenizer, session,
                                           block_len=args.block_len)
                    except ValueError as e:
                        slice_stats["reasons"]["render_overflow"] += 1
                        session = None
                        for c in candidates:
                            if c["index"] == chosen["index"]:
                                c["accepted"] = False
                                c["reason"] = f"render_overflow: {e}"
                    else:
                        session["n_rendered_tokens"] = r.n_rendered_tokens
                        session["n_supervised_tokens"] = r.n_supervised
                        session["n_system_tokens"] = r.n_system_tokens
                        session["system_key"] = r.system_key
                        slice_stats["accepted"] += 1
                        slice_stats["supervised"].append(r.n_supervised)
                        slice_stats["rendered"].append(r.n_rendered_tokens)

                f_cand.write(json.dumps({
                    "id": example["example_id"],
                    "source_id": example["source_id"],
                    "data_type": name, "group": example["group"],
                    "source": example["source"],
                    "turn_index": example["turn_index"],
                    "n_context_assistant_turns": example["n_context_assistant_turns"],
                    "system": example["context"][0]["content"],
                    "rendered_prompt": rendered_prompt,
                    "prompt_tokens": len(prompt_ids),
                    "completion_budget": budget,
                    "gold": example["gold"],
                    "candidates": candidates,
                    "selected_index": chosen["index"] if session else None,
                }, ensure_ascii=False) + "\n")
                if session:
                    f_sess.write(json.dumps(session, ensure_ascii=False) + "\n")
                done += 1

            elapsed = time.time() - started
            print(f"  {done}/{total} examples  {elapsed:.0f}s "
                  f"({elapsed / max(done, 1):.2f}s/ex)", flush=True)

    def pct(values):
        if not values:
            return None
        return {"n": len(values), "mean": round(statistics.mean(values), 1),
                "sd": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
                "p50": round(statistics.median(values), 1),
                "min": min(values), "max": max(values)}

    summary = {}
    for name, s in stats.items():
        if not s["examples"]:
            summary[name] = {"examples": 0, "not_reached": True}
            continue
        n_cand = sum(s["reasons"].values())
        summary[name] = {
            "examples": s["examples"],
            "accepted": s["accepted"],
            "example_accept_rate": round(s["accepted"] / s["examples"], 4),
            "candidate_accept_rate": round(s["reasons"]["ok"] / n_cand, 4) if n_cand else 0.0,
            "reject_reasons": dict(s["reasons"].most_common()),
            "supervised_tokens": pct(s["supervised"]),
            "rendered_tokens": pct(s["rendered"]),
            "new_tokens": pct(s["new_tokens"]),
            "correctness_rate": round(s["correct"] / n_cand, 4) if n_cand else 0.0,
            "correctness_verdicts": dict(s["verdicts"].most_common()),
        }

    template = tokenizer.get_chat_template()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "teacher": args.model,
        "revision": revision,
        "tokenizer": {
            "sha256": hashlib.sha256(
                json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
            "transformers_version": transformers_version,
        },
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "generation_config": {
            "eos_token_id": generation_config.eos_token_id,
            "stop_ids": sorted(stops), "source": "generation_config.json",
        },
        "thinking_mode": True,
        "session_limit_tokens": args.block_len,
        "render_allowance": RENDER_ALLOWANCE,
        "turn_expansion": {
            "enabled": True,
            "rule": ("one independent example per assistant turn preceded by a "
                     "user or tool message; context = the conversation prefix"),
            "supervision": ("only the newly generated teacher turn; system, user, "
                            "template and all preceding original assistant turns "
                            "are context and are excluded from loss and from "
                            "supervised-token accounting"),
            "mask": "aadistill.data.dataset.final_assistant_loss_mask",
        },
        "system_prompt_policy": {
            "default": SYSTEM_DEFAULT, "source_prompt_preserved": True,
            "mandatory": True,
        },
        "decoding": {
            "n": args.n, "all_candidates_sampled": True,
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "min_p": args.min_p,
            "preset": "Qwen3-4B-Thinking-2507 official recommendation",
            "seed_base": args.seed,
            "seed_rule": "seed + batch_index + candidate_index * 1000003",
            "seed_stride": SEED_STRIDE,
            "max_new_tokens": "per example: block_len - rendered_prompt - allowance",
            "dtype": args.dtype, "engine": args.engine,
            "engine_version": getattr(engine, "version", None),
        },
        "acceptance": {
            "basis": "hygiene",
            "rules": ["length_limited", "empty_answer", "not_terminated",
                      "stray_marker", "think_not_closed", "degenerate",
                      "render_overflow"],
            "correctness_computed_but_not_enforced": True,
            "deviation": (
                "verify.hygiene_reason's too_long rule (MAX_ANSWER_WORDS=600) is "
                "NOT applied: a generic word-count gate is forbidden by AGENTS.md "
                "P3/P10 and would bias the corpus against long derivations"),
            "selection": "lowest-index accepted candidate, one per example",
        },
        "data_dir": args.data_dir,
        "prompt_selection": {
            "mode": args.select, "limit_per_type": args.limit_per_type,
            "limits": limits, "types": names,
            "type_definitions": {n: TYPES[n] for n in names},
            "per_type_filtering": selection_stats,
            "excluded_sources": exclude_paths,
            "excluded_content_hashes": len(excluded_content),
            "excluded_prompt_hashes": len(excluded_prompts),
            "dedup": ("content sha256 over the conversation prefix including the "
                      "target turn (ids excluded), global across types"),
            "leakage_rule": ("a source conversation is dropped whole if its content "
                             "hash or its first-user-message hash appears in any "
                             "reserved val / calib / holdout / behaviour-eval split"),
        },
        "mixture_rationale": args.mixture_note,
        "complete": not budget_stopped,
        "examples_requested": total,
        "examples_generated": done,
        "skipped_no_budget": skipped_no_budget,
        "types": summary,
        "seconds": round(time.time() - started, 1),
        "outputs": {
            "candidates": sha256_file(out_dir / "candidates.jsonl"),
            "sessions": sha256_file(out_dir / "sessions.jsonl"),
        },
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nwrote {out_dir} ({manifest['seconds']}s)")
    for name, s in summary.items():
        if s.get("not_reached"):
            print(f"  {name:16s} not reached")
            continue
        sup = s["supervised_tokens"]
        print(f"  {name:16s} ex {s['examples']:6d}  accept {s['example_accept_rate']:.3f}  "
              f"sup/accepted {sup['mean'] if sup else 0:.0f}  "
              f"top rejects {list(s['reject_reasons'].items())[:3]}")


if __name__ == "__main__":
    main()
