"""The evaluation must send the model the context it was trained on.

A second `<|im_start|>system` turn slipped into 6 of 76 behaviour prompts and
was invisible in every metric — it only surfaced by rendering a prompt and
reading it. These tests make that class of defect fail loudly instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "evaluation"))

from audit_prompt_rendering import DEFAULT_SYSTEM, render  # noqa: E402


class FakeTok:
    """Minimal stand-in for the Qwen chat template's observable behaviour."""

    def apply_chat_template(self, turns, tools=None, tokenize=False,
                            add_generation_prompt=False):
        out = []
        for m in turns:
            content = m["content"]
            if m["role"] == "system" and tools:
                content += "\n\n# Tools\n<tools>\n" + str(tools) + "\n</tools>"
            out.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n<think>\n")
        return "".join(out)


def test_default_system_injected_when_sample_has_none():
    s = {"id": "x", "messages": [{"role": "user", "content": "hi"}]}
    text, src = render(FakeTok(), s)
    assert src == "default"
    assert text.count("<|im_start|>system") == 1
    assert DEFAULT_SYSTEM in text


def test_sample_system_preserved_and_not_duplicated():
    s = {"id": "x", "messages": [
        {"role": "system", "content": "You are a specialist."},
        {"role": "user", "content": "hi"}]}
    text, src = render(FakeTok(), s)
    assert src == "sample"
    assert text.count("<|im_start|>system") == 1
    assert "You are a specialist." in text
    assert DEFAULT_SYSTEM not in text        # the default must not be added on top


def test_prompt_ends_at_the_assistant_generation_prompt():
    s = {"id": "x", "messages": [{"role": "user", "content": "hi"}]}
    text, _ = render(FakeTok(), s)
    assert text.endswith("<|im_start|>assistant\n<think>\n")


def test_assistant_turns_are_stripped_from_the_prompt():
    s = {"id": "x", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "LEAKED"},
        {"role": "user", "content": "again"}]}
    text, _ = render(FakeTok(), s)
    assert "LEAKED" not in text
    assert text.count("<|im_start|>assistant") == 1   # only the generation prompt


def test_tool_schema_renders_into_the_system_block():
    s = {"id": "x", "messages": [{"role": "user", "content": "hi"}],
         "tools": [{"function": {"name": "search_recipes"}}]}
    text, _ = render(FakeTok(), s)
    head = text.split("<|im_start|>user")[0]
    assert "search_recipes" in head


@pytest.mark.parametrize("suite", ["data/eval_behavior_v0/prompts.jsonl"])
def test_real_prompt_set_renders_one_system_turn_each(suite):
    """Runs against the committed behaviour set, not a fixture."""
    import json
    path = Path(__file__).resolve().parents[2] / suite
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    tok = FakeTok()
    for s in rows:
        text, _ = render(tok, s)
        assert text.count("<|im_start|>system") == 1, s["id"]
