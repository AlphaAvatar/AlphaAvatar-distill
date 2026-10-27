"""Teacher-native training documents taken token-in/token-out, and isolated packing.

A **document** is one generation exactly as the teacher produced it: the
``prompt_ids`` the engine was given (system, user, the assistant header and
whatever the chat template prefilled after it) and the ``output_ids`` the
teacher sampled. Only ``output_ids`` are targets. Nothing is re-rendered:
rendering a finished conversation again is not token-identical to "prompt +
generated tokens" (the template strips newlines around a reasoning span and in
front of an answer, and appends a ``\\n`` the model never emitted), so a
re-rendered target would supervise tokens the teacher did not produce and
condition on a context it did not see.

**Isolated packing.** ``pack_isolated`` places whole documents into fixed-length
sequences, never cutting one (P17: a trace that does not fit is not trained,
it is not trimmed), and gives every document position ids restarting at 0. Run
with those position ids and no attention mask, transformers builds a
block-diagonal causal mask, so each document attends only to itself -- the
context the teacher had when it generated it -- and is positioned as it was.
The trailing pad run gets its own restarting positions, so it is a separate
segment that no real token can see (padding stays a contiguous suffix, which
`training.train.nonpad_extent` checks).

Everything model- or mode-specific -- which template, which kwargs, which stop
ids -- arrives as arguments (AGENTS.md P3).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

__all__ = [
    "DocumentError",
    "TokenDocument",
    "document_from_turn",
    "generation_suffix",
    "prompt_mode_problems",
    "PackedDocuments",
    "pack_isolated",
    "isolation_report",
]


class DocumentError(ValueError):
    """A record that cannot become a token-in/token-out training document."""


@dataclass(frozen=True)
class TokenDocument:
    """One teacher generation: context ``prompt_ids``, targets ``output_ids``."""

    doc_id: str
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    meta: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not self.prompt_ids:
            raise DocumentError(f"{self.doc_id}: empty prompt; a document's first "
                                "token must be context, never a target")
        if not self.output_ids:
            raise DocumentError(f"{self.doc_id}: empty generation")

    def __len__(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.output_ids

    @property
    def target_mask(self) -> tuple[bool, ...]:
        return (False,) * len(self.prompt_ids) + (True,) * len(self.output_ids)

    def content_sha256(self) -> str:
        return hashlib.sha256(json.dumps([self.prompt_ids, self.output_ids]).encode()).hexdigest()


def document_from_turn(turn: Mapping, *, doc_id: str, stop_ids: Iterable[int],
                       meta: Mapping | None = None) -> TokenDocument:
    """A document from a stored generation turn, or a refusal.

    The turn must carry the engine's own ``prompt_token_ids`` and
    ``output_token_ids`` and must have ended on one of ``stop_ids`` -- the stop
    token is the teacher's own and is a target; a turn cut at a limit or by a
    degeneration stop has no natural end to teach and is refused.
    """
    p, o = turn.get("prompt_token_ids"), turn.get("output_token_ids")
    if not p or not o:
        raise DocumentError(f"{doc_id}: the turn carries no prompt_token_ids/output_token_ids "
                            "(a measurement record, or one written before token capture)")
    stops = set(int(x) for x in stop_ids)
    if int(o[-1]) not in stops:
        raise DocumentError(f"{doc_id}: the generation did not end on a stop id "
                            f"({turn.get('stop_reason')!r}); only naturally terminated "
                            "generations become documents")
    return TokenDocument(doc_id, tuple(int(x) for x in p), tuple(int(x) for x in o), dict(meta or {}))


def generation_suffix(tokenizer, template_kwargs: Mapping | None = None) -> tuple[int, ...]:
    """The token ids the chat template appends after the last user turn.

    Derived by rendering one probe conversation with and without the generation
    prompt, so it is whatever *this* template and *these* kwargs produce: for a
    hybrid template, ``<|im_start|>assistant\\n`` in thinking mode and the same
    plus an empty closed think block in non-thinking mode.
    """
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    kw = dict(template_kwargs or {})
    with_gp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    without = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw)
    if not with_gp.startswith(without):
        raise DocumentError("the generation prompt is not a suffix of the rendered conversation")
    return tuple(tokenizer(with_gp[len(without):], add_special_tokens=False)["input_ids"])


def prompt_mode_problems(doc: TokenDocument, suffix: Sequence[int]) -> list[str]:
    """Why ``doc``'s prompt was not produced under the generation prompt ``suffix``."""
    n = len(suffix)
    if n == 0:
        return ["empty generation suffix"]
    if tuple(doc.prompt_ids[-n:]) != tuple(suffix):
        return [f"{doc.doc_id}: the prompt does not end in the declared mode's generation prompt"]
    return []


@dataclass
class PackedDocuments:
    """Fixed-length isolated sequences and where every document went."""

    input_ids: torch.Tensor          # [N, L] long
    target_mask: torch.Tensor        # [N, L] bool: the documents' generated tokens
    content_mask: torch.Tensor       # [N, L] bool: real (non-pad) tokens
    position_ids: torch.Tensor       # [N, L] long: restart at every document and at the pad run
    members: list[list[dict]]        # per sequence: {"doc_id", "start", "length"}
    refused: list[dict]              # documents that fit no sequence

    def blocks(self):
        """The trainer's 4-tuple (ids, loss mask, content mask, position ids)."""
        return self.input_ids, self.target_mask, self.content_mask, self.position_ids

    def stats(self) -> dict:
        n, length = self.input_ids.shape
        real = int(self.content_mask.sum())
        return {"sequences": n, "sequence_length": length, "documents": sum(map(len, self.members)),
                "refused": len(self.refused), "non_padding_tokens": real,
                "supervised_tokens": int(self.target_mask.sum()),
                "context_tokens": real - int(self.target_mask.sum()),
                "padding_fraction": round(1 - real / (n * length), 6) if n else None}


def pack_isolated(docs: Sequence[TokenDocument], seq_len: int, pad_id: int, *,
                  order_seed: int | None = None) -> PackedDocuments:
    """Whole documents, best fit, into ``seq_len`` sequences with restarting positions.

    Documents are placed in the given order (or a ``random.Random(order_seed)``
    shuffle of it), each into the open sequence with the least room that still
    holds it, a new one when none does. A document longer than ``seq_len`` is
    refused and reported, never cut. The result is a pure function of the
    documents, ``seq_len`` and ``order_seed``.
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    order = list(range(len(docs)))
    if order_seed is not None:
        random.Random(order_seed).shuffle(order)
    bins: list[list[int]] = []
    room: list[int] = []
    refused = []
    seen = set()
    for i in order:
        d = docs[i]
        if d.doc_id in seen:
            raise DocumentError(f"duplicate document id {d.doc_id}")
        seen.add(d.doc_id)
        if len(d) > seq_len:
            refused.append({"doc_id": d.doc_id, "length": len(d), "reason": "longer than seq_len"})
            continue
        best = None
        for b, r in enumerate(room):
            if r >= len(d) and (best is None or r < room[best]):
                best = b
        if best is None:
            bins.append([])
            room.append(seq_len)
            best = len(bins) - 1
        bins[best].append(i)
        room[best] -= len(d)
    n = len(bins)
    ids = torch.full((n, seq_len), int(pad_id), dtype=torch.long)
    target = torch.zeros((n, seq_len), dtype=torch.bool)
    content = torch.zeros((n, seq_len), dtype=torch.bool)
    pos = torch.zeros((n, seq_len), dtype=torch.long)
    members = []
    for b, idxs in enumerate(bins):
        at, mem = 0, []
        for i in idxs:
            d = docs[i]
            k = len(d)
            ids[b, at:at + k] = torch.tensor(d.ids)
            target[b, at:at + k] = torch.tensor(d.target_mask)
            content[b, at:at + k] = True
            pos[b, at:at + k] = torch.arange(k)
            mem.append({"doc_id": d.doc_id, "start": at, "length": k})
            at += k
        pos[b, at:] = torch.arange(seq_len - at)
        members.append(mem)
    return PackedDocuments(ids, target, content, pos, members, refused)


@torch.no_grad()
def isolation_report(model, packed: PackedDocuments, docs: Mapping[str, TokenDocument], *,
                     sequences: Sequence[int] = (0,), dtype=None, hidden: bool = False) -> dict:
    """Packed forward against each document run alone, at the documents' target positions.

    The teacher-forced conditional-context check: under isolation every logit a
    target is trained against must be the logit of the document on its own.
    Returns the largest absolute difference per sequence and overall.
    ``hidden=True`` compares the final hidden states instead (the logits are a
    fixed linear map of them), which is what a large vocabulary on a host needs.
    """
    out = {"sequences": {}, "max_abs_diff": 0.0, "compared": "final_hidden" if hidden else "logits"}
    dev = next(model.parameters()).device

    def run(ids, **kw):
        if hidden:
            return model.model(input_ids=ids, use_cache=False, **kw).last_hidden_state[0]
        return model(ids, use_cache=False, **kw).logits[0]

    for b in sequences:
        ids = packed.input_ids[b:b + 1].to(dev)
        logits = run(ids, position_ids=packed.position_ids[b:b + 1].to(dev))
        worst = 0.0
        for m in packed.members[b]:
            d = docs[m["doc_id"]]
            alone = run(torch.tensor([d.ids], device=dev))
            s, k = m["start"], m["length"]
            sel = torch.tensor(d.target_mask, device=dev)[1:]          # positions predicting a target
            a = logits[s:s + k - 1][sel]
            r = alone[: k - 1][sel]
            if dtype is not None:
                a, r = a.to(dtype), r.to(dtype)
            worst = max(worst, float((a - r).abs().max()))
        out["sequences"][int(b)] = worst
        out["max_abs_diff"] = max(out["max_abs_diff"], worst)
    return out
