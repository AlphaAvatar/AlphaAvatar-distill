"""Audit the chat rendering the evaluation actually sends to the model.

Written after the harness was found injecting a second `<|im_start|>system`
turn for the 6 behaviour prompts that carry their own. That defect was invisible
in the metrics — it only showed up by rendering a prompt and looking at it — so
the rendering is now checked mechanically for every prompt in every suite.

Asserts, per prompt:
  * the chat template is the one the corpus was generated with (hash-pinned)
  * exactly one system turn
  * the system content is either the sample's own or the project default
  * the prompt ends at the assistant generation prompt with `<think>` open
  * tool schemas render into the system block when the sample carries tools
  * no assistant turn leaks into the prompt

    uv run python scripts/evaluation/audit_prompt_rendering.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM = "You are a helpful Assistant."
# The chat template recorded in the corpus manifest; generation, training and
# evaluation must all use this exact template or the comparison is unsound.
CORPUS_TEMPLATE_SHA = "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"


def render(tok, sample: dict, system: str = DEFAULT_SYSTEM) -> tuple[str, str]:
    """Exactly what `uncapped_eval.py` does — kept in step with it deliberately."""
    turns = [m for m in sample["messages"] if m["role"] != "assistant"]
    has_system = any(m.get("role") == "system" for m in turns)
    if system and not has_system:
        turns = [{"role": "system", "content": system}] + turns
    text = tok.apply_chat_template(turns, tools=sample.get("tools"),
                                   tokenize=False, add_generation_prompt=True)
    return text, ("sample" if has_system else "default")


def audit(tok, path: Path, label: str) -> list[str]:
    problems = []
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n_tools = n_own_system = 0
    for s in rows:
        text, src = render(tok, s)
        sid = s["id"]
        n_sys = text.count("<|im_start|>system")
        if n_sys != 1:
            problems.append(f"{label}/{sid}: {n_sys} system turns")
        if src == "sample":
            n_own_system += 1
            own = next(m["content"] for m in s["messages"] if m["role"] == "system")
            if own[:40] not in text:
                problems.append(f"{label}/{sid}: sample system prompt not rendered")
        elif DEFAULT_SYSTEM not in text:
            problems.append(f"{label}/{sid}: default system prompt missing")
        if not text.endswith("<|im_start|>assistant\n<think>\n"):
            problems.append(f"{label}/{sid}: does not end at the assistant "
                            f"generation prompt; tail={text[-40:]!r}")
        if "<|im_start|>assistant" in text[:-len("<|im_start|>assistant\n<think>\n")]:
            problems.append(f"{label}/{sid}: an assistant turn leaked into the prompt")
        if s.get("tools"):
            n_tools += 1
            name = s["tools"][0].get("function", {}).get("name") or \
                s["tools"][0].get("name", "")
            if name and name not in text:
                problems.append(f"{label}/{sid}: tool schema '{name}' not rendered")
    print(f"{label}: {len(rows)} prompts | own system {n_own_system} | "
          f"with tools {n_tools} | problems {len(problems)}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint")
    ap.add_argument("--prompts", nargs="*", default=[
        "data/eval_behavior_v0/prompts.jsonl",
        "artifacts/eval/e1/gsm8k_reasoning_100.jsonl",
    ])
    args = ap.parse_args()

    tpl = Path(args.model) / "chat_template.jinja"
    got = hashlib.sha256(tpl.read_bytes()).hexdigest()
    print(f"chat_template sha256 {got}")
    if got != CORPUS_TEMPLATE_SHA:
        print(f"FAIL: template differs from the corpus manifest "
              f"({CORPUS_TEMPLATE_SHA})")
        raise SystemExit(1)
    print("template matches the corpus manifest (generation == evaluation)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    problems = []
    for p in args.prompts:
        path = Path(p)
        if not path.is_file():
            print(f"SKIP (missing): {p}")
            continue
        problems += audit(tok, path, path.stem)

    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for x in problems[:20]:
            print("  " + x)
        raise SystemExit(1)
    print("\nall prompts render correctly")


if __name__ == "__main__":
    main()
