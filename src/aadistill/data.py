"""Stage 2 offline data: schema validation, chat rendering with assistant-token
loss masks, and fixed-length block packing.

This is the loader the Stage 3 trainer consumes. `encode_sample` turns one
jsonl sample into parallel (input_ids, loss_mask) lists; `pack_blocks`
concatenates many encoded samples into fixed-length training blocks. All
functions are deterministic in input order — shuffling is the trainer's job.

Loss-mask method: the Qwen3-Thinking-2507 chat template is *not* prefix-stable
(it injects an empty ``<think>\\n\\n</think>\\n\\n`` block into the final
assistant turn only), so per-turn prefix diffing cannot locate assistant
spans. Instead the full conversation is rendered once, assistant segments are
found in the rendered string (``<|im_start|>assistant\\n ... <|im_end|>``),
and character spans are mapped to token indices with the fast tokenizer's
offset mapping. Builder-side hygiene guarantees content never contains the
template's control markers (see FORBIDDEN_MARKERS), so the scan cannot be
spoofed by data content. The injected empty think block is trained on purpose:
the realtime student should learn to close its think block immediately.

Trainable tokens per assistant turn: content through the closing
``<|im_end|>``. Role headers, system/user/tool turns, and separators are
masked. For ``format=="text"`` samples every token is trainable and the
tokenizer's EOS token is appended as a document separator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

GROUPS = (
    "instruction",
    "rag_evidence",
    "multihop_qa",
    "tool_calling",
    "refusal_uncertainty",
    "code_math",
    "short_realtime",
    "long_context",
)

SPLITS = ("train", "val", "calib")

# Strings that must never appear inside message/text content: they collide
# with the chat template's control tokens or the assistant-span scan.
FORBIDDEN_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)

_ROLES = ("system", "user", "assistant", "tool")
_ASSISTANT_HEADER = "<|im_start|>assistant\n"
_ASSISTANT_SEG = re.compile(
    re.escape(_ASSISTANT_HEADER) + ".*?" + re.escape("<|im_end|>"), re.DOTALL
)


def content_fields(sample: dict):
    """Yield every free-text field of a sample (for hygiene checks)."""
    if sample.get("format") == "text":
        yield sample.get("text", "")
        return
    for msg in sample.get("messages", []):
        yield msg.get("content", "")


def validate_sample(sample: dict) -> None:
    """Raise ValueError (mentioning the sample id) if the schema is violated."""
    sid = sample.get("id", "<missing id>")

    def fail(reason: str):
        raise ValueError(f"sample {sid}: {reason}")

    for key in ("id", "group", "source", "format"):
        if not isinstance(sample.get(key), str) or not sample[key]:
            fail(f"missing or non-string field {key!r}")
    if sample["group"] not in GROUPS:
        fail(f"unknown group {sample['group']!r}")
    if sample["format"] not in ("chat", "text"):
        fail(f"unknown format {sample['format']!r}")

    if sample["format"] == "text":
        if not isinstance(sample.get("text"), str) or not sample["text"].strip():
            fail("format 'text' requires a non-empty 'text' field")
    else:
        messages = sample.get("messages")
        if not isinstance(messages, list) or not messages:
            fail("format 'chat' requires a non-empty 'messages' list")
        saw_assistant = False
        saw_tool_call = False
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role not in _ROLES:
                fail(f"message {i} has invalid role {role!r}")
            if i == 0 and role not in ("system", "user"):
                fail("conversation must start with a system or user message")
            content = msg.get("content")
            if not isinstance(content, str):
                fail(f"message {i} content must be a string")
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None:
                if role != "assistant":
                    fail(f"message {i}: tool_calls only allowed on assistant")
                if not isinstance(tool_calls, list) or not tool_calls:
                    fail(f"message {i}: tool_calls must be a non-empty list")
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    if tc.get("type") != "function" or not isinstance(
                        fn.get("name"), str
                    ) or not isinstance(fn.get("arguments"), dict):
                        fail(f"message {i}: malformed tool_call {tc!r}")
                saw_tool_call = True
            if role == "assistant":
                if not content.strip() and not tool_calls:
                    fail(f"message {i}: assistant needs content or tool_calls")
                saw_assistant = True
            if role == "tool" and not saw_tool_call:
                fail(f"message {i}: tool response without a prior tool_call")
        if not saw_assistant:
            fail("conversation has no assistant message")
        tools = sample.get("tools")
        if tools is not None:
            if not isinstance(tools, list) or not tools:
                fail("tools must be a non-empty list when present")
            for tool in tools:
                fn = tool.get("function", {}) if isinstance(tool, dict) else {}
                if tool.get("type") != "function" or not isinstance(
                    fn.get("name"), str
                ):
                    fail(f"malformed tool definition {tool!r}")

    for text in content_fields(sample):
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                fail(f"content contains forbidden marker {marker!r}")


def load_jsonl(path: str | Path, validate: bool = True) -> list[dict]:
    samples = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid json: {e}") from e
            if validate:
                validate_sample(sample)
            samples.append(sample)
    return samples


def load_split(data_dir: str | Path, split: str) -> dict[str, list[dict]]:
    """Load ``<data_dir>/<split>/<group>.jsonl`` for every group file present."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    split_dir = Path(data_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"missing split directory {split_dir}")
    groups: dict[str, list[dict]] = {}
    for path in sorted(split_dir.glob("*.jsonl")):
        group = path.stem
        if group not in GROUPS:
            raise ValueError(f"{path}: filename is not a known group")
        samples = load_jsonl(path)
        for s in samples:
            if s["group"] != group:
                raise ValueError(f"{path}: sample {s['id']} has group {s['group']}")
        groups[group] = samples
    if not groups:
        raise FileNotFoundError(f"no group jsonl files in {split_dir}")
    return groups


def render_chat(tokenizer, sample: dict) -> str:
    """Render a chat sample to the training string via the chat template."""
    return tokenizer.apply_chat_template(
        sample["messages"],
        tools=sample.get("tools"),
        tokenize=False,
        add_generation_prompt=False,
    )


def encode_sample(
    tokenizer, sample: dict, max_seq_len: int | None = None
) -> tuple[list[int], list[int]]:
    """Encode one sample into (input_ids, loss_mask) of equal length.

    loss_mask[i] == 1 means token i is a supervised target when used as a
    label (the trainer applies the usual next-token shift). Truncation at
    max_seq_len may cut mid-turn; the mask is truncated consistently.
    """
    if not tokenizer.is_fast:
        raise ValueError("encode_sample requires a fast tokenizer (offset mapping)")

    if sample["format"] == "text":
        ids = tokenizer(sample["text"], add_special_tokens=False).input_ids
        ids.append(tokenizer.eos_token_id)
        mask = [1] * len(ids)
    else:
        text = render_chat(tokenizer, sample)
        spans = [
            (m.start() + len(_ASSISTANT_HEADER), m.end())
            for m in _ASSISTANT_SEG.finditer(text)
        ]
        if not spans:
            raise ValueError(f"sample {sample['id']}: no assistant segment in render")
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc.input_ids
        mask = [0] * len(ids)
        span_iter = iter(spans)
        span = next(span_iter)
        for i, (a, b) in enumerate(enc.offset_mapping):
            while span is not None and a >= span[1]:
                span = next(span_iter, None)
            if span is None:
                break
            if a < span[1] and b > span[0]:
                mask[i] = 1

    if max_seq_len is not None:
        ids, mask = ids[:max_seq_len], mask[:max_seq_len]
    return ids, mask


def pack_blocks(
    encoded, block_len: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Concatenate (ids, mask) pairs and cut into fixed-length blocks.

    Returns (input_ids [n, block_len] int64, loss_mask [n, block_len] bool,
    dropped_tail_tokens). Samples are packed back-to-back without padding;
    a sample may straddle a block boundary.
    """
    ids_buf: list[int] = []
    mask_buf: list[int] = []
    for ids, mask in encoded:
        if len(ids) != len(mask):
            raise ValueError("ids and mask length mismatch")
        ids_buf.extend(ids)
        mask_buf.extend(mask)
    n_blocks = len(ids_buf) // block_len
    kept = n_blocks * block_len
    input_ids = torch.tensor(ids_buf[:kept], dtype=torch.long).view(n_blocks, block_len)
    loss_mask = torch.tensor(mask_buf[:kept], dtype=torch.bool).view(n_blocks, block_len)
    return input_ids, loss_mask, len(ids_buf) - kept
