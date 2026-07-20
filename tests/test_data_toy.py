"""Tests for the Stage 2 data loader (src/aadistill/data.py).

Packing and schema validation run with no downloads. Loss-mask tests need the
pinned teacher tokenizer's chat template; they load it from the local HF cache
and are skipped when it is absent (tests never touch the network).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aadistill.data import encode_sample, pack_blocks, validate_sample

TEACHER = "Qwen/Qwen3-4B-Thinking-2507"
REVISION = "768f209d9ea81521153ed38c47d515654e938aea"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            TEACHER, revision=REVISION, local_files_only=True)
    except Exception:
        pytest.skip("teacher tokenizer not in local HF cache")


def chat_sample(messages, group="instruction", tools=None, sid="t-000001"):
    sample = {"id": sid, "group": group, "source": "test", "format": "chat",
              "messages": messages}
    if tools is not None:
        sample["tools"] = tools
    return sample


SINGLE_TURN = chat_sample([
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 is 4."},
])

MULTI_TURN = chat_sample([
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 is 4."},
    {"role": "user", "content": "And times 3?"},
    {"role": "assistant", "content": "12."},
])

TOOL_SAMPLE = chat_sample(
    [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": '{"temp_c": 21}'},
        {"role": "assistant", "content": "21 degrees and clear."},
    ],
    group="tool_calling",
    tools=[{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}],
)

TEXT_SAMPLE = {"id": "t-text-01", "group": "long_context", "source": "test",
               "format": "text", "text": "A plain document about tides."}


# ---------- schema validation (no tokenizer needed) ----------

def test_validate_accepts_good_samples():
    for s in (SINGLE_TURN, MULTI_TURN, TOOL_SAMPLE, TEXT_SAMPLE):
        validate_sample(s)


@pytest.mark.parametrize("mutate,reason", [
    (lambda s: s.update(group="nonsense"), "unknown group"),
    (lambda s: s.update(format="text"), "text without text field"),
    (lambda s: s.pop("source"), "missing source"),
    (lambda s: s["messages"][0].update(role="oracle"), "invalid role"),
    (lambda s: s["messages"][1].update(content=""), "empty assistant"),
    (lambda s: s["messages"][0].update(
        tool_calls=[{"type": "function",
                     "function": {"name": "f", "arguments": {}}}]),
     "tool_calls on user"),
    (lambda s: s["messages"][0].update(content="hi <|im_start|> there"),
     "forbidden marker"),
])
def test_validate_rejects_bad_samples(mutate, reason):
    import copy

    s = copy.deepcopy(SINGLE_TURN)
    mutate(s)
    with pytest.raises(ValueError):
        validate_sample(s)


def test_validate_rejects_tool_response_without_call():
    s = chat_sample([
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "hello"},
    ])
    with pytest.raises(ValueError):
        validate_sample(s)


# ---------- packing (no tokenizer needed) ----------

def test_pack_blocks_shapes_and_content():
    encoded = [([1, 2, 3], [0, 1, 1]), ([4, 5, 6, 7, 8], [1, 0, 1, 0, 1])]
    input_ids, loss_mask, dropped = pack_blocks(encoded, block_len=4)
    assert input_ids.shape == (2, 4) and loss_mask.shape == (2, 4)
    assert input_ids.dtype == torch.long and loss_mask.dtype == torch.bool
    assert input_ids.flatten().tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert loss_mask.flatten().tolist() == [False, True, True, True,
                                            False, True, False, True]
    assert dropped == 0


def test_pack_blocks_drops_tail():
    input_ids, loss_mask, dropped = pack_blocks([([1] * 10, [1] * 10)], block_len=4)
    assert input_ids.shape == (2, 4) and dropped == 2


def test_pack_blocks_empty():
    input_ids, loss_mask, dropped = pack_blocks([], block_len=4)
    assert input_ids.shape == (0, 4) and dropped == 0


def test_pack_blocks_rejects_length_mismatch():
    with pytest.raises(ValueError):
        pack_blocks([([1, 2], [1])], block_len=2)


# ---------- loss masks with the real chat template ----------

def trainable_runs(mask):
    """Contiguous [start, end) runs where mask == 1."""
    runs, start = [], None
    for i, m in enumerate(mask + [0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    return runs


def test_single_turn_mask(tokenizer):
    ids, mask = encode_sample(tokenizer, SINGLE_TURN)
    assert len(ids) == len(mask)
    runs = trainable_runs(mask)
    assert len(runs) == 1
    span = tokenizer.decode(ids[runs[0][0]:runs[0][1]])
    # Final assistant turn carries the template-injected empty think block and
    # is trained through its closing <|im_end|>.
    assert span.startswith("<think>")
    assert span.endswith("2+2 is 4.<|im_end|>")
    masked = tokenizer.decode([t for t, m in zip(ids, mask) if not m])
    assert "What is 2+2?" in masked and "2+2 is 4." not in masked


def test_multi_turn_mask(tokenizer):
    ids, mask = encode_sample(tokenizer, MULTI_TURN)
    runs = trainable_runs(mask)
    assert len(runs) == 2
    first = tokenizer.decode(ids[runs[0][0]:runs[0][1]])
    second = tokenizer.decode(ids[runs[1][0]:runs[1][1]])
    assert first.endswith("2+2 is 4.<|im_end|>") and "<think>" not in first
    assert second.endswith("12.<|im_end|>") and second.startswith("<think>")
    masked = tokenizer.decode([t for t, m in zip(ids, mask) if not m])
    assert "And times 3?" in masked


def test_tool_call_mask(tokenizer):
    ids, mask = encode_sample(tokenizer, TOOL_SAMPLE)
    runs = trainable_runs(mask)
    assert len(runs) == 2
    call_span = tokenizer.decode(ids[runs[0][0]:runs[0][1]])
    assert "<tool_call>" in call_span and '"get_weather"' in call_span
    masked = tokenizer.decode([t for t, m in zip(ids, mask) if not m])
    # Tool definitions (system) and the tool response are context, not targets.
    assert "<tools>" in masked and "<tool_response>" in masked


def test_encode_deterministic(tokenizer):
    assert encode_sample(tokenizer, MULTI_TURN) == encode_sample(tokenizer, MULTI_TURN)
    assert encode_sample(tokenizer, TOOL_SAMPLE) == encode_sample(tokenizer, TOOL_SAMPLE)


def test_text_format_all_trainable(tokenizer):
    ids, mask = encode_sample(tokenizer, TEXT_SAMPLE)
    assert all(mask)
    assert ids[-1] == tokenizer.eos_token_id


def test_truncation(tokenizer):
    ids, mask = encode_sample(tokenizer, MULTI_TURN, max_seq_len=8)
    assert len(ids) == 8 and len(mask) == 8
