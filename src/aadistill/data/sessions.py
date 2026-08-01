"""Session rendering and system-prompt-aware packing for the 8,192-token corpus.

Why this exists rather than `best_fit_blocks`
---------------------------------------------
Three independent requirements make the existing packer unusable for this corpus.

**The official chat template deletes reasoning from non-final assistant turns.**
`Qwen3-4B-Thinking-2507`'s template renders `<think>…</think>` only for assistant
messages that follow the *last* user message (`chat_template.jinja` lines 43-51:
the `loop.index0 > ns.last_query_index` branch). Build one logical message list
`[system, u1, a1, u2, a2, …]`, apply the template once as the naive reading of
the packing spec suggests, and every trace except the last is silently dropped
from the render. Under option B (decision 2026-07-28) the trace *is* the training
target, so a block holding six sessions would lose five of its six traces and
nothing would raise. Measured directly, not inferred from reading the jinja.

Sessions are therefore rendered **independently** — each through the official
template, where its own trace survives — and the resulting token sequences are
concatenated with the shared system block emitted once. The concatenation is
exact at the token level, which is asserted rather than assumed:

    ids(system) + ids(body₁) + … + ids(body_k) == ids(system + body₁ + … + body_k)

**The system prompt is a hard packing boundary.** Only sessions whose system
content is byte-identical may share a block; the system block is emitted exactly
once, first; no system message may appear mid-block.

**Placement is sequential, not best-fit.** Best-fit reshuffles when samples are
added, which would break the nested token ladder — rung k's blocks must be a
strict prefix of rung k+1's. Sequential fill over a fixed session order gives
prefix-nested blocks by construction, and because the terminal session may be
cut at the boundary, packing efficiency stays at ~100% anyway. Nothing is traded
away for the nesting property.

Padding needs no attention mask, and passing one would be actively harmful
--------------------------------------------------------------------------
Attention is causal and padding is right-aligned, so no real token can attend to
a pad: a real position's hidden state is identical whether the pad run is present
or not. That is also what makes terminal truncation provably safe (§5) — removing
a suffix cannot change any preceding token, mask, logit or loss position.

Passing an explicit attention mask would additionally create fully-masked rows at
pad positions, whose softmax is NaN, and `NaN * 0` is NaN — poisoning a loss that
correctly excludes those positions. The masks below therefore govern loss, KD and
accounting only, exactly as `best_fit_blocks` already does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import final_assistant_loss_mask

# The project's mandatory default. Applied only when a source session carries no
# system message of its own; an existing one is preserved byte-for-byte and is
# never overwritten or prepended to (protocol requirement, 2026-07-31).
SYSTEM_DEFAULT = "You are a helpful Assistant."

# Slack between the generation budget and the 8,192-token limit. The stored
# target is re-rendered through the chat template, which strips and re-inserts
# newlines around the think block (`.strip('\n')` / `.lstrip('\n')` in the
# template), so the re-rendered length can differ from prompt+generation by a
# few tokens in either direction. The allowance absorbs that; `render_overflow`
# below is the backstop that catches any case it does not.
RENDER_ALLOWANCE = 8


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Return `(system_text, body_messages)` for one source session.

    A source-provided system prompt is preserved exactly. Its absence — which is
    the case for every prompt in this corpus's four in-scope types — yields
    `SYSTEM_DEFAULT`.
    """
    if messages and messages[0].get("role") == "system":
        return messages[0].get("content", ""), list(messages[1:])
    return SYSTEM_DEFAULT, list(messages)


def system_group_key(system_text: str, tools: list | None) -> str:
    """Stable hash of everything that lands in the rendered system block.

    `tools` is part of the key because the template renders tool signatures
    *inside* the system block: two sessions with identical system text but
    different tools produce different system blocks and must not share one.
    """
    payload = json.dumps(
        {"system": system_text, "tools": tools}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def render_system_block(tokenizer, system_text: str, tools: list | None = None) -> str:
    """The rendered system block alone, as the template emits it."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )


def generation_prompt(tokenizer, messages: list[dict], tools: list | None = None) -> str:
    """The prompt the teacher answers, with the system message always present."""
    system_text, body = split_system(messages)
    # Drop a trailing assistant message if the caller passed a full session.
    if body and body[-1].get("role") == "assistant":
        body = body[:-1]
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}] + body,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def completion_budget(
    tokenizer,
    messages: list[dict],
    tools: list | None = None,
    *,
    block_len: int = 8192,
    allowance: int = RENDER_ALLOWANCE,
) -> int:
    """Tokens the teacher may generate so the whole session fits `block_len`.

    Derived per prompt from the fully rendered prompt, never set to a flat
    `block_len`: the rendered prompt already consumes part of the budget, and a
    flat cap would let long-prompt sessions overflow the limit and be rejected
    after they were paid for.

    Returns a value that may be <= 0, which means the prompt alone leaves no room
    for an answer; the caller must skip such a prompt rather than generate.
    """
    prompt = generation_prompt(tokenizer, messages, tools)
    n_prompt = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return block_len - n_prompt - allowance


@dataclass
class RenderedSession:
    """One complete source session, rendered and tokenized, ready to pack.

    `body_ids` excludes the system block, which is emitted once per packed block.
    `body_mask` marks assistant-target tokens under the project's established
    convention (the span from after `<|im_start|>assistant\\n` through the
    closing `<|im_end|>`), restricted to the **final** assistant turn. Under turn
    expansion a session's earlier assistant turns are the original public
    responses kept as context, so supervising them would mix public and teacher
    targets in one block (P17). For a single-turn session the two rules coincide.
    """

    session_id: str
    data_type: str
    system_text: str
    system_key: str
    body_ids: list[int]
    body_mask: list[bool]
    n_system_tokens: int
    tools: list | None = None
    # Conversation this example was turn-expanded from. Two examples sharing it
    # are prefixes of one another, so they must never share a packed block.
    source_id: str | None = None
    candidate_index: int | None = None
    candidate_sha256: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_body_tokens(self) -> int:
        return len(self.body_ids)

    @property
    def n_supervised(self) -> int:
        """Supervised tokens *before* any packing-time terminal truncation."""
        return int(sum(self.body_mask))

    @property
    def n_rendered_tokens(self) -> int:
        """Full rendered length including the system block — the §1 quantity."""
        return self.n_system_tokens + len(self.body_ids)


def render_session(
    tokenizer,
    session: dict,
    *,
    block_len: int = 8192,
) -> RenderedSession:
    """Render and tokenize one session, verifying every property packing relies on.

    Raises `ValueError` if the session does not fit `block_len` — that is the
    `render_overflow` backstop behind `completion_budget`'s allowance, and it
    fires *before* a bad sample can reach the packer.
    """
    messages = session["messages"]
    tools = session.get("tools")
    system_text, body_messages = split_system(messages)

    full_messages = [{"role": "system", "content": system_text}] + body_messages
    full = tokenizer.apply_chat_template(
        full_messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    system_block = render_system_block(tokenizer, system_text, tools)
    if not full.startswith(system_block):
        raise ValueError(
            f"session {session.get('id')!r}: rendered system block is not a prefix "
            "of the rendered session — the chat template changed under us"
        )
    body = full[len(system_block):]

    system_ids = tokenizer(system_block, add_special_tokens=False).input_ids
    body_ids, body_mask = final_assistant_loss_mask(tokenizer, body)

    # The property the whole packing design rests on: rendering per session and
    # concatenating token sequences must equal rendering the concatenation. If a
    # tokenizer ever merged across the boundary this would silently shift every
    # mask in the block, so it is checked per session rather than trusted.
    whole = tokenizer(full, add_special_tokens=False).input_ids
    if system_ids + body_ids != whole:
        raise ValueError(
            f"session {session.get('id')!r}: token concatenation drift — "
            f"{len(system_ids)}+{len(body_ids)} pieces != {len(whole)} whole"
        )
    if not any(body_mask):
        raise ValueError(
            f"session {session.get('id')!r}: no assistant-supervised token in render"
        )

    n_rendered = len(system_ids) + len(body_ids)
    if n_rendered > block_len:
        raise ValueError(
            f"session {session.get('id')!r}: render_overflow — {n_rendered} tokens "
            f"exceeds the {block_len}-token session limit"
        )

    return RenderedSession(
        session_id=str(session.get("id")),
        data_type=str(session.get("data_type") or session.get("slice") or session.get("group")),
        system_text=system_text,
        system_key=system_group_key(system_text, tools),
        body_ids=body_ids,
        body_mask=body_mask,
        n_system_tokens=len(system_ids),
        tools=tools,
        source_id=str(session.get("source_id") or session.get("id")),
        candidate_index=session.get("candidate_index"),
        candidate_sha256=session.get("candidate_sha256"),
        meta={"n_rendered_tokens": n_rendered},
    )


@dataclass
class PackedBlock:
    """One fixed-length training block plus everything needed to audit it (§8)."""

    input_ids: list[int]
    ce_mask: list[bool]
    content_mask: list[bool]
    audit: dict

    @property
    def n_supervised(self) -> int:
        """Authoritative post-packing supervised-token count (§7)."""
        return int(sum(self.ce_mask))


def _cut_boundary_kind(mask: list[bool], offset: int) -> str:
    """Whether the truncation offset lands inside an assistant-supervised span."""
    if offset <= 0 or offset > len(mask):
        return "out_of_range"
    return "assistant" if mask[offset - 1] else "non_assistant"


def pack_group(
    sessions: list[RenderedSession],
    system_ids: list[int],
    *,
    block_len: int,
    pad_id: int,
) -> list[PackedBlock]:
    """Pack one exact-system-prompt group sequentially into `block_len` blocks.

    Sessions are consumed in the order given — that order is the caller's
    responsibility and is what makes the ladder nested and type-balanced.

    Only the final session appended to a block may be truncated, and only at the
    block boundary. Its discarded suffix is dropped, never carried into a later
    block: re-packing the tail would train a continuation whose premises are
    absent, which is the defect best-fit packing was adopted to remove.

    **Two examples turn-expanded from the same conversation may never share a
    block.** They are prefixes of one another: example `#t1` is supervised on
    `a1ᵗ`, while `#t3` carries the original `a1ᵒ` in its context. Packed together
    in a causal block, whichever lands second can read the other's answer, so the
    supervision would be duplicated and partly leaked. A colliding session is
    skipped and stays queued for a later block rather than being dropped.

    Skipping preserves the prefix nesting the ladder needs: block k's contents
    depend only on the ordered session list up to the point where it closed, so
    appending more sessions never changes an earlier block.
    """
    if not sessions:
        return []
    blocks: list[PackedBlock] = []
    pending = list(range(len(sessions)))

    while pending:
        ids = list(system_ids)
        ce = [False] * len(system_ids)
        members: list[dict] = []
        truncation: dict | None = None
        sources: set[str] = set()
        consumed: list[int] = []
        deferred = 0

        for position in pending:
            session = sessions[position]
            room = block_len - len(ids)
            if room <= 0:
                break
            if session.source_id is not None and session.source_id in sources:
                # Same original conversation already in this block — defer it.
                deferred += 1
                continue

            if session.n_body_tokens <= room:
                start = len(ids)
                ids.extend(session.body_ids)
                ce.extend(session.body_mask)
                members.append({
                    "session_id": session.session_id,
                    "data_type": session.data_type,
                    "candidate_index": session.candidate_index,
                    "candidate_sha256": session.candidate_sha256,
                    "start": start,
                    "end": len(ids),
                    "original_rendered_tokens": session.n_rendered_tokens,
                    "original_body_tokens": session.n_body_tokens,
                    "truncated": False,
                    "supervised_retained": session.n_supervised,
                    "supervised_discarded": 0,
                })
                consumed.append(position)
                if session.source_id is not None:
                    sources.add(session.source_id)
                if len(ids) >= block_len:
                    break
                continue

            # The session does not fit whole: it can only be the terminal one.
            retained = int(sum(session.body_mask[:room]))
            if retained == 0:
                # Appending would add prompt tokens and no supervision. §5
                # forbids filling space that way; pad instead and let this
                # session open the next block intact.
                break

            start = len(ids)
            ids.extend(session.body_ids[:room])
            ce.extend(session.body_mask[:room])
            discarded = session.n_supervised - retained
            truncation = {
                "session_id": session.session_id,
                "original_body_tokens": session.n_body_tokens,
                "packed_start_offset": start,
                "truncation_offset": len(ids),
                "kept_body_tokens": room,
                "supervised_retained": retained,
                "supervised_discarded": discarded,
                "cut_boundary_kind": _cut_boundary_kind(session.body_mask, room),
                "cut_token_id": session.body_ids[room - 1],
            }
            members.append({
                "session_id": session.session_id,
                "data_type": session.data_type,
                "candidate_index": session.candidate_index,
                "candidate_sha256": session.candidate_sha256,
                "start": start,
                "end": len(ids),
                "original_rendered_tokens": session.n_rendered_tokens,
                "original_body_tokens": session.n_body_tokens,
                "truncated": True,
                "supervised_retained": retained,
                "supervised_discarded": discarded,
            })
            consumed.append(position)
            if session.source_id is not None:
                sources.add(session.source_id)
            break

        if not members:
            raise RuntimeError(
                "packing made no progress — a session larger than the block was "
                "not rejected at render time"
            )
        taken = set(consumed)
        pending = [i for i in pending if i not in taken]

        unpadded = len(ids)
        content = [True] * unpadded
        pad_n = block_len - unpadded
        ids.extend([pad_id] * pad_n)
        ce.extend([False] * pad_n)
        content.extend([False] * pad_n)

        supervision_spans = []
        run_start = None
        for position, flag in enumerate(ce):
            if flag and run_start is None:
                run_start = position
            elif not flag and run_start is not None:
                supervision_spans.append([run_start, position])
                run_start = None
        if run_start is not None:
            supervision_spans.append([run_start, len(ce)])

        blocks.append(PackedBlock(
            input_ids=ids,
            ce_mask=ce,
            content_mask=content,
            audit={
                "system_sha256": hashlib.sha256(
                    json.dumps(system_ids).encode()).hexdigest(),
                "n_system_tokens": len(system_ids),
                "sessions": members,
                "session_ids": [m["session_id"] for m in members],
                "supervision_spans": supervision_spans,
                "unpadded_length": unpadded,
                "padding_length": pad_n,
                "final_length": len(ids),
                "deferred_same_source": deferred,
                "terminal_truncated": truncation is not None,
                "terminal_truncation": truncation,
                "supervised_tokens": int(sum(ce)),
            },
        ))

    return blocks


def load_packed_blocks(packed_dir, n_blocks: int | None = None):
    """Load pre-packed blocks as the `(ids, loss_mask, content_mask)` the Trainer takes.

    A ladder rung is *the first `n_blocks` blocks* of the single pack, so a rung
    is selected here by slicing rather than by re-packing. That is what makes the
    twelve runs share one packing layout: nothing about a rung or a training seed
    can change a block's contents (§10).

    Imports numpy/torch locally so the packing logic above stays importable — and
    unit-testable — without them.
    """
    import numpy as np
    import torch

    data = np.load(Path(packed_dir) / "blocks.npz")
    ids = torch.from_numpy(data["input_ids"].astype(np.int64))
    ce = torch.from_numpy(data["ce_mask"])
    content = torch.from_numpy(data["content_mask"])
    if n_blocks is not None:
        if n_blocks > ids.shape[0]:
            raise ValueError(
                f"requested {n_blocks} blocks but the pack holds {ids.shape[0]}")
        ids, ce, content = ids[:n_blocks], ce[:n_blocks], content[:n_blocks]
    return ids, ce, content


def rung_n_blocks(ladder: dict, target_supervised_tokens: int) -> int:
    """Block count for one ladder rung, from a `ladder.json` payload."""
    for rung in ladder["rungs"]:
        if rung["target_supervised_tokens"] == target_supervised_tokens:
            if not rung.get("reachable", True):
                raise ValueError(
                    f"rung {target_supervised_tokens} is unreachable: the corpus "
                    f"holds {rung['actual_supervised_tokens']} supervised tokens")
            return rung["n_blocks"]
    raise KeyError(
        f"no rung at {target_supervised_tokens}; have "
        f"{[r['target_supervised_tokens'] for r in ladder['rungs']]}")


def pack_sessions(
    sessions: list[RenderedSession],
    system_ids_by_key: dict[str, list[int]],
    *,
    block_len: int = 8192,
    pad_id: int = 151643,
) -> list[PackedBlock]:
    """Pack an ordered session list, never mixing system-prompt groups (§4).

    Groups are packed independently and emitted in sorted key order. This corpus
    has exactly one group (no source sample in the four in-scope types carries a
    system message, so all take `SYSTEM_DEFAULT`), which makes the emitted order
    identity. With several groups, prefix type-balance would only hold *within* a
    group, so the ladder builder checks the group count rather than assuming it.
    """
    by_key: dict[str, list[RenderedSession]] = {}
    for session in sessions:
        by_key.setdefault(session.system_key, []).append(session)

    blocks: list[PackedBlock] = []
    for key in sorted(by_key):
        if key not in system_ids_by_key:
            raise KeyError(f"no rendered system block for group {key}")
        blocks.extend(pack_group(
            by_key[key], system_ids_by_key[key],
            block_len=block_len, pad_id=pad_id,
        ))
    return blocks
