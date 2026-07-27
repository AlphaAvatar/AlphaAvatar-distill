"""Build the `eval_behavior_v0` prompt set from held-out val data.

    uv run python scripts/build_eval_behavior_v0.py \
        --data-dir data/stage2_v1 --out-dir data/eval_behavior_v0

Selection is deterministic: candidates are filtered, sorted by id, and sampled
with a pinned seed, so the same command reproduces the same prompt set. The
prompts come from the **val** splits — never trained on.

One prompt per sample: the conversation prefix up to the first assistant turn.
That turn becomes the gold reference. `long_context` is excluded (format=="text"
continuation data, no conversation to prompt from); `tool_calling` is restricted
to samples whose first assistant turn is a tool call, so the tool-call scorers
apply to every prompt in that group.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.behavior import BEHAVIOR_GROUPS, final_number, is_refusal
from aadistill.data import load_split
from aadistill.env import code_state
from aadistill.manifest import sha256_file

SEED = 20260727
PER_GROUP = 12
# Matches `block_len` in the Stage 3 configs: training packs samples into
# 1024-token blocks, so a longer prompt would be evaluated in a context regime
# the student never saw contiguously. Keeping the cap here means every prompt is
# in-distribution for context length; the cost is that `multihop_qa` (hotpot,
# 10 paragraphs per question, p50 1515 tokens) can only contribute 4 prompts.
MAX_PROMPT_TOKENS = 1024
# A group may fall short of PER_GROUP after that filter, but not below this —
# fewer than 4 prompts is not worth reporting as a group-level number.
MIN_PER_GROUP = 4
# Recall floor for the refusal detector on gold refusals. The whole
# refusal_uncertainty val split is squad_v2 unanswerable questions, so every
# gold answer is a refusal; a detector that misses them cannot measure the
# student either. Checked at build time so it fails here, not in analysis.
MIN_REFUSAL_RECALL = 0.95


def first_assistant_index(messages: list[dict]) -> int | None:
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            return i
    return None


def build_entry(sample: dict, tokenizer) -> dict | None:
    """Turn a val sample into a prompt-set entry, or None if unsuitable."""
    messages = sample["messages"]
    idx = first_assistant_index(messages)
    if idx is None or idx == 0:
        return None
    prefix, gold = messages[:idx], messages[idx]
    gold_calls = gold.get("tool_calls")

    if sample["group"] == "tool_calling" and not gold_calls:
        return None
    if sample["group"] != "tool_calling" and not gold.get("content", "").strip():
        return None

    rendered = tokenizer.apply_chat_template(
        prefix,
        tools=sample.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
    )
    n_tokens = len(tokenizer(rendered, add_special_tokens=False).input_ids)
    if n_tokens > MAX_PROMPT_TOKENS:
        return None

    entry = {
        "id": sample["id"],
        "group": sample["group"],
        "source": sample["source"],
        "messages": prefix,
        "prompt_tokens": n_tokens,
        # Concatenated non-assistant text: what an "echoing" answer would copy.
        "prompt_text": "\n".join(m.get("content", "") for m in prefix),
        "gold_answer": gold.get("content", ""),
    }
    if sample.get("tools"):
        entry["tools"] = sample["tools"]
    if gold_calls:
        entry["gold_tool_calls"] = gold_calls
    if sample["source"] == "gsm8k":
        entry["gsm8k_answer"] = final_number(gold.get("content", ""))
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/stage2_v1")
    ap.add_argument("--out-dir", default="data/eval_behavior_v0")
    ap.add_argument("--tokenizer", default="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint")
    ap.add_argument("--per-group", type=int, default=PER_GROUP)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok_path = REPO_ROOT / args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_path) if tok_path.exists() else args.tokenizer
    )

    val = load_split(REPO_ROOT / args.data_dir, "val")

    # Calibrate the refusal detector on the gold refusals before using it.
    golds = [
        m["content"]
        for s in val["refusal_uncertainty"]
        for m in s["messages"]
        if m["role"] == "assistant"
    ]
    recall = sum(is_refusal(g) for g in golds) / len(golds)
    if recall < MIN_REFUSAL_RECALL:
        raise SystemExit(
            f"refusal detector recall on gold refusals is {recall:.3f} "
            f"(< {MIN_REFUSAL_RECALL}); fix aadistill.behavior._REFUSAL first"
        )

    entries, per_group_stats = [], {}
    for group in BEHAVIOR_GROUPS:
        samples = sorted(val[group], key=lambda s: s["id"])
        candidates = [e for e in (build_entry(s, tokenizer) for s in samples) if e]
        if len(candidates) < MIN_PER_GROUP:
            raise SystemExit(
                f"group {group}: only {len(candidates)} usable candidates "
                f"(floor is {MIN_PER_GROUP})"
            )
        take = min(args.per_group, len(candidates))
        rng = random.Random(f"{args.seed}:{group}")
        picked = sorted(rng.sample(candidates, take), key=lambda e: e["id"])
        entries.extend(picked)
        per_group_stats[group] = {
            "candidates": len(candidates),
            "selected": len(picked),
            "short_of_target": take < args.per_group,
            "sources": sorted({e["source"] for e in picked}),
            "max_prompt_tokens": max(e["prompt_tokens"] for e in picked),
        }

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "prompts.jsonl"
    with open(prompts_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    manifest = {
        "name": "eval_behavior_v0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": (
            "Mechanically scored behavior eval for Stage 3 recovery gates. "
            "Complements holdout_v1 (language modeling) with chat-format "
            "validity, question-echo, degeneracy, evidence grounding, refusal, "
            "tool-call validity and gsm8k exact match."
        ),
        "source_split": {
            "data_dir": args.data_dir,
            "split": "val",
            "note": "held out from all Stage 3 training; never trained on",
            "group_sha256": {
                g: sha256_file(REPO_ROOT / args.data_dir / "val" / f"{g}.jsonl")
                for g in BEHAVIOR_GROUPS
            },
        },
        "excluded_groups": {
            "long_context": "format=='text' continuation data; no conversation prefix to prompt from"
        },
        "selection": {
            "seed": args.seed,
            "per_group": args.per_group,
            "rule": "first assistant turn is the gold; prefix is the prompt; "
                    "tool_calling restricted to tool-call golds; "
                    f"prompt <= {MAX_PROMPT_TOKENS} tokens; "
                    "random.Random(f'{seed}:{group}').sample over id-sorted candidates",
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_prompt_tokens_rationale": (
                "equals block_len in the Stage 3 configs, so every prompt is "
                "in-distribution for context length"
            ),
            "group_imbalance": (
                "multihop_qa contributes fewer than per_group: only 4 of its 25 "
                "val samples fit under the token cap (hotpot p50 is 1515 tokens). "
                "Read its group row as indicative only; overall rows are "
                "prompt-weighted, so it carries proportionally little weight."
            ),
            "per_group_stats": per_group_stats,
        },
        "tokenizer": {
            "path": args.tokenizer,
            "sha256": sha256_file(tok_path / "tokenizer.json") if tok_path.exists() else None,
        },
        "refusal_detector_recall_on_gold": round(recall, 4),
        "prompts": {
            "path": str(prompts_path.relative_to(REPO_ROOT)),
            "count": len(entries),
            "sha256": sha256_file(prompts_path),
            "total_prompt_tokens": sum(e["prompt_tokens"] for e in entries),
        },
        "code_state": code_state(str(REPO_ROOT)),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {prompts_path} ({len(entries)} prompts)")
    for g, s in per_group_stats.items():
        print(f"  {g:22s} {s['selected']:3d} of {s['candidates']:4d} candidates "
              f"(max {s['max_prompt_tokens']} tok) {','.join(s['sources'])}")
    print(f"  refusal detector recall on gold: {recall:.3f}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
