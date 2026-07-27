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

import bisect
import json
import random
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
    **a sample may straddle a block boundary**, which is why `best_fit_blocks`
    exists — see its docstring. Kept for the four logged runs, whose data path
    this is.
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


def best_fit_blocks(
    encoded, block_len: int, pad_id: int = 0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Pack samples into blocks so that **no sample is split across a boundary**.

    Concatenate-then-cut (`pack_blocks`) tears samples at every block edge. For
    short targets that is a rounding error; for reasoning traces it is a
    correctness problem — the second half of a trace trains as a sequence whose
    premises are not in context, teaching the student to continue reasoning it
    cannot see. Ding et al., *Fewer Truncations Improve Language Modeling*
    (ICML 2024, arXiv:2404.10830) measure exactly this and fix it with
    length-aware bin packing: same token efficiency, no unnecessary truncation,
    and materially better reading comprehension and context following.

    Placement is best-fit — each sample goes into the fullest block that still
    has room — but the samples are visited in **seeded random order, not
    longest-first**. Best-fit-*decreasing* packs marginally tighter and is what
    the paper describes, but sorting by length is itself a distribution change:
    it groups long samples with long ones and short with short, so a block's
    composition correlates with length in a way no deployment context does. The
    student is trained for the mixed contexts it will actually see (decision
    record 2026-07-28). The seed keeps this reproducible, which the project's
    "block order is a pure function of (seed, epoch)" rule requires.

    A sample longer than `block_len` cannot be kept whole by any packing — it is
    truncated (and counted, because a rising count means `block_len` is too
    small for the corpus).

    Residual capacity is padded; `loss_mask` is False there, so padding never
    contributes to the loss. Returns (input_ids, loss_mask, stats) where stats
    carries the packing efficiency and truncation count that make the trade
    visible in the run manifest.

    Samples sharing a block **do** attend to each other, and that is deliberate:
    Krell et al., *Efficient Sequence Packing without Cross-contamination*
    (arXiv:2107.02027) block-diagonal that away, but a deployed assistant reads
    a context window holding several unrelated things and has to attend across
    it correctly. Training it to ignore irrelevant preceding content is the job,
    not an artifact to mask out (decision record 2026-07-28).
    """
    items = []
    for ids, mask in encoded:
        if len(ids) != len(mask):
            raise ValueError("ids and mask length mismatch")
        items.append((list(ids), list(mask)))

    truncated = 0
    prepared = []
    for ids, mask in items:
        if len(ids) > block_len:
            ids, mask = ids[:block_len], mask[:block_len]
            truncated += 1
        prepared.append((ids, mask))
    random.Random(seed).shuffle(prepared)

    # Best fit = the tightest block the sample still fits in. Kept in a
    # capacity-sorted list so placement is a binary search rather than a scan
    # over every open block: the scan is O(samples x blocks), which measured
    # 44 s on an option-B-shaped corpus and grows quadratically with it.
    blocks: list[list[tuple[list[int], list[int]]]] = []
    used: list[int] = []
    capacities: list[tuple[int, int]] = []  # (remaining, block index), sorted
    for ids, mask in prepared:
        size = len(ids)
        slot = bisect.bisect_left(capacities, (size, -1))
        if slot == len(capacities):
            index = len(blocks)
            blocks.append([(ids, mask)])
            used.append(size)
            remaining = block_len - size
        else:
            remaining, index = capacities.pop(slot)
            blocks[index].append((ids, mask))
            used[index] += size
            remaining -= size
        if remaining > 0:
            bisect.insort(capacities, (remaining, index))

    ids_rows, mask_rows = [], []
    for block in blocks:
        row_ids: list[int] = []
        row_mask: list[int] = []
        for ids, mask in block:
            row_ids.extend(ids)
            row_mask.extend(mask)
        pad = block_len - len(row_ids)
        ids_rows.append(row_ids + [pad_id] * pad)
        mask_rows.append(row_mask + [0] * pad)

    stats = {
        "blocks": len(blocks),
        "samples": len(prepared),
        "truncated_samples": truncated,
        "real_tokens": sum(used),
        "padding_tokens": len(blocks) * block_len - sum(used),
        "efficiency": round(sum(used) / (len(blocks) * block_len), 4) if blocks else 0.0,
    }
    return (torch.tensor(ids_rows, dtype=torch.long),
            torch.tensor(mask_rows, dtype=torch.bool),
            stats)
