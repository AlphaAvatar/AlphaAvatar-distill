"""Teacher-native training documents taken token-in/token-out, and their packing.

A **document** is one generation exactly as the teacher produced it: the
``prompt_ids`` the engine was given (system, user, the assistant header and
whatever the chat template prefilled after it) and the ``output_ids`` the
teacher sampled. Only ``output_ids`` are targets. Nothing is re-rendered:
rendering a finished conversation again is not token-identical to "prompt +
generated tokens" (the template strips newlines around a reasoning span and in
front of an answer, and appends a ``\\n`` the model never emitted), so a
re-rendered target would supervise tokens the teacher did not produce.

**Two context strategies, both supported** (``CONTEXT_STRATEGIES``). They choose
the training distribution; neither is a correction of the other.

* ``cross_context`` -- several documents share one causal context per packed
  sequence, positions continuous, so a later document is conditioned on the
  earlier ones. Teacher and student run on the same composed sequence, so KD
  compares them on one context; what changes is that the context is not the one
  the teacher generated under. ``pack_cross_context`` records the composition
  (order, boundaries, and how much foreign context precedes each document) so it
  is part of the run's identity. A packed composition of independent documents
  is not a real conversation and must not be reported as one.
* ``document_isolated`` -- ``pack_isolated`` gives every document position ids
  restarting at 0. Run with those and no attention mask, transformers builds a
  block-diagonal causal mask, so each document is conditioned only on itself,
  the context its generation had. The trailing pad run gets its own restarting
  positions, a separate segment no real token can see.

Neither cuts a document (P17: a trace that does not fit is not trained, it is
not trimmed). Everything model- or mode-specific -- which template, which
kwargs, which stop ids, which shared system block -- arrives as arguments
(AGENTS.md P3).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = [
    "CONTEXT_STRATEGIES",
    "DocumentError",
    "TokenDocument",
    "document_from_turn",
    "generation_suffix",
    "prompt_mode_problems",
    "message_terminator",
    "PackedDocuments",
    "pack_isolated",
    "isolated_sequence_count",
    "pack_cross_context",
    "pack_documents",
    "write_pack",
    "context_exposure",
    "isolation_report",
]

#: The trainer's ``batch.context_strategy`` values; the first is the default.
CONTEXT_STRATEGIES = ("cross_context", "document_isolated")


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


def message_terminator(tokenizer, end_of_turn_id: int, template_kwargs: Mapping | None = None) -> tuple[int, ...]:
    """What the chat template writes after a message's end-of-turn token.

    A generation ends ON ``end_of_turn_id``; the template follows every rendered
    message with more (``\\n`` for the Qwen templates). Composing documents into
    one context puts this between them, as context, so the composition reads the
    way the template writes consecutive turns.
    """
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "u"}], tokenize=False,
                                        add_generation_prompt=False, **dict(template_kwargs or {}))
    ids = tokenizer(ids, add_special_tokens=False)["input_ids"]
    if int(end_of_turn_id) not in ids:
        raise DocumentError("the end-of-turn id does not occur in a rendered message")
    last = len(ids) - 1 - ids[::-1].index(int(end_of_turn_id))
    return tuple(ids[last + 1:])


@dataclass
class PackedDocuments:
    """Fixed-length packed sequences and where every document went."""

    input_ids: torch.Tensor          # [N, L] long
    target_mask: torch.Tensor        # [N, L] bool: the documents' generated tokens
    content_mask: torch.Tensor       # [N, L] bool: real (non-pad) tokens
    position_ids: torch.Tensor       # [N, L] long: restarting per document (isolated) or continuous
    members: list[list[dict]]        # per sequence: {"doc_id", "start", "length", "foreign_context", "supervised"}
    refused: list[dict]              # documents that fit no sequence
    strategy: str = "document_isolated"
    shared_prefix_length: int = 0    # cross_context: the shared block emitted once per sequence

    def blocks(self):
        """The trainer's tuple: (ids, loss mask, content mask, position ids) under
        isolation, (ids, loss mask, content mask) under cross_context, whose
        positions are the default continuous ones."""
        if self.strategy == "document_isolated":
            return self.input_ids, self.target_mask, self.content_mask, self.position_ids
        return self.input_ids, self.target_mask, self.content_mask

    def stats(self) -> dict:
        n, length = self.input_ids.shape
        real = int(self.content_mask.sum())
        return {"strategy": self.strategy, "sequences": n, "sequence_length": length,
                "documents": sum(map(len, self.members)),
                "refused": len(self.refused), "non_padding_tokens": real,
                "supervised_tokens": int(self.target_mask.sum()),
                "context_tokens": real - int(self.target_mask.sum()),
                "padding_fraction": round(1 - real / (n * length), 6) if n else None}

    def composition(self) -> list[list[str]]:
        """Each sequence's documents in order: the composition a run's identity records."""
        return [[m["doc_id"] for m in ms] for ms in self.members]


def _best_fit(costs_new: Sequence[int], costs_join: Sequence[int], seq_len: int, order: Sequence[int]):
    """Best-fit placement: item ``i`` costs ``costs_new[i]`` opening a sequence and
    ``costs_join[i]`` joining one. Returns the bins and the refused items."""
    bins: list[list[int]] = []
    room: list[int] = []
    refused = []
    for i in order:
        if costs_new[i] > seq_len:
            refused.append(i)
            continue
        best = None
        for b, r in enumerate(room):
            if r >= costs_join[i] and (best is None or r < room[best]):
                best = b
        if best is None:
            bins.append([])
            room.append(seq_len)
            best = len(bins) - 1
            room[best] -= costs_new[i]
        else:
            room[best] -= costs_join[i]
        bins[best].append(i)
    return bins, refused


def _order(docs: Sequence[TokenDocument], order_seed: int | None) -> list[int]:
    seen = set()
    for d in docs:
        if d.doc_id in seen:
            raise DocumentError(f"duplicate document id {d.doc_id}")
        seen.add(d.doc_id)
    order = list(range(len(docs)))
    if order_seed is not None:
        random.Random(order_seed).shuffle(order)
    return order


def isolated_sequence_count(lengths: Mapping[str, int], seq_len: int, *, order_seed: int | None = None) -> int:
    """The number of sequences `pack_isolated` makes of documents with these lengths, given in
    `pack_isolated`'s input order (a mapping keeps it), without building any tensor."""
    order = list(range(len(lengths)))
    if order_seed is not None:
        random.Random(order_seed).shuffle(order)
    sizes = list(lengths.values())
    bins, _ = _best_fit(sizes, sizes, seq_len, order)
    return len(bins)


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
    order = _order(docs, order_seed)
    lengths = [len(d) for d in docs]
    bins, refused = _best_fit(lengths, lengths, seq_len, order)
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
            mem.append({"doc_id": d.doc_id, "start": at, "length": k, "foreign_context": 0,
                        "supervised": len(d.output_ids)})
            at += k
        pos[b, at:] = torch.arange(seq_len - at)
        members.append(mem)
    return PackedDocuments(ids, target, content, pos, members,
                           [{"doc_id": docs[i].doc_id, "length": len(docs[i]), "reason": "longer than seq_len"}
                            for i in refused], "document_isolated", 0)


def pack_cross_context(docs: Sequence[TokenDocument], seq_len: int, pad_id: int, *,
                       shared_prefix: Sequence[int] = (), separator_ids: Sequence[int] = (),
                       order_seed: int | None = None) -> PackedDocuments:
    """Whole documents, best fit, composed into one causal context per sequence.

    A sequence is ``shared_prefix`` once, then each document's body (its prompt
    after the shared prefix, then its generation), consecutive bodies joined by
    ``separator_ids`` (context, e.g. the template's ``message_terminator``).
    Every document's prompt must begin with ``shared_prefix``: a system block is
    a hard boundary, never repeated or mixed mid-sequence. Placement is the same
    seeded best fit as ``pack_isolated``, so the two strategies differ in the
    conditioning context and not in which documents meet. Positions are
    continuous; each member records how many tokens of other documents precede
    it (``foreign_context``).
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    prefix, sep = tuple(int(x) for x in shared_prefix), tuple(int(x) for x in separator_ids)
    for d in docs:
        if tuple(d.prompt_ids[:len(prefix)]) != prefix:
            raise DocumentError(f"{d.doc_id}: the prompt does not begin with the shared prefix")
        if len(d.prompt_ids) == len(prefix) and prefix:
            raise DocumentError(f"{d.doc_id}: nothing of the prompt follows the shared prefix")
    order = _order(docs, order_seed)
    body = [len(d) - len(prefix) for d in docs]
    bins, refused = _best_fit([len(prefix) + b for b in body], [len(sep) + b for b in body], seq_len, order)
    n = len(bins)
    ids = torch.full((n, seq_len), int(pad_id), dtype=torch.long)
    target = torch.zeros((n, seq_len), dtype=torch.bool)
    content = torch.zeros((n, seq_len), dtype=torch.bool)
    members = []
    for b, idxs in enumerate(bins):
        if prefix:
            ids[b, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)
        at, mem = len(prefix), []
        for j, i in enumerate(idxs):
            d = docs[i]
            if j and sep:
                ids[b, at:at + len(sep)] = torch.tensor(sep)
                at += len(sep)
            seg = d.ids[len(prefix):]
            ids[b, at:at + len(seg)] = torch.tensor(seg)
            target[b, at:at + len(seg)] = torch.tensor(d.target_mask[len(prefix):])
            mem.append({"doc_id": d.doc_id, "start": at, "length": len(seg),
                        "foreign_context": at - len(prefix), "supervised": len(d.output_ids)})
            at += len(seg)
        content[b, :at] = True
        members.append(mem)
    pos = torch.arange(seq_len).expand(n, seq_len).clone()
    return PackedDocuments(ids, target, content, pos, members,
                           [{"doc_id": docs[i].doc_id, "length": len(prefix) + body[i],
                             "reason": "longer than seq_len"} for i in refused],
                           "cross_context", len(prefix))


def pack_documents(docs: Sequence[TokenDocument], seq_len: int, pad_id: int, *, strategy: str,
                   order_seed: int | None = None, shared_prefix: Sequence[int] = (),
                   separator_ids: Sequence[int] = ()) -> PackedDocuments:
    """The packer for a declared ``strategy`` (one of ``CONTEXT_STRATEGIES``)."""
    if strategy == "document_isolated":
        return pack_isolated(docs, seq_len, pad_id, order_seed=order_seed)
    if strategy == "cross_context":
        return pack_cross_context(docs, seq_len, pad_id, shared_prefix=shared_prefix,
                                  separator_ids=separator_ids, order_seed=order_seed)
    raise ValueError(f"unknown context strategy {strategy!r}; one of {CONTEXT_STRATEGIES}")


def write_pack(train: PackedDocuments, val: PackedDocuments, out_dir, *, data_type: Mapping[str, str],
               extra: Mapping | None = None) -> dict:
    """``blocks.npz``, ``ladder.json`` and ``audit.jsonl`` in the trainer's ladder contract.

    One rung covering ``train``'s sequences; ``val``'s sequences follow them as
    the tail the loader validates on (`aadistill.data.ladder.ladder_blocks`).
    Isolated packs also store ``position_ids``. ``data_type`` maps every doc id
    to the category its validation mixture and audit count it under. Returns
    the ladder record, with the sha256 of each written file.
    """
    import numpy as np

    from aadistill.infrastructure.manifest import sha256_file

    out = Path(out_dir)
    if train.strategy != val.strategy or train.input_ids.shape[1] != val.input_ids.shape[1]:
        raise DocumentError("train and validation sequences must share one strategy and sequence length")
    out.mkdir(parents=True, exist_ok=True)
    cat = lambda name: torch.cat([getattr(train, name), getattr(val, name)]).numpy()  # noqa: E731
    arrays = {"input_ids": cat("input_ids").astype(np.int32), "ce_mask": cat("target_mask"),
              "content_mask": cat("content_mask")}
    if train.strategy == "document_isolated":
        arrays["position_ids"] = cat("position_ids").astype(np.int32)
    np.savez(out / "blocks.npz", **arrays)
    with (out / "audit.jsonl").open("w") as f:
        for b, members in enumerate(train.members + val.members):
            f.write(json.dumps({"block": b, "split": "train" if b < len(train.members) else "val",
                                "sessions": [{"session_id": m["doc_id"], "data_type": data_type[m["doc_id"]],
                                              "supervised_retained": m["supervised"], "start": m["start"],
                                              "length": m["length"], "foreign_context": m["foreign_context"]}
                                             for m in members]}) + "\n")
    st = train.stats()
    ladder = {"packing": f"token_documents.{train.strategy}", "block_len": int(train.input_ids.shape[1]),
              "n_blocks": st["sequences"], "n_val_blocks": len(val.members),
              "rungs": [{"target_supervised_tokens": st["supervised_tokens"], "n_blocks": st["sequences"],
                         "reachable": True, "actual_supervised_tokens": st["supervised_tokens"]}],
              "train": st, "val": val.stats(), **dict(extra or {})}
    (out / "ladder.json").write_text(json.dumps(ladder, indent=1) + "\n")
    return {**ladder, "files_sha256": {n: sha256_file(out / n) for n in ("blocks.npz", "ladder.json", "audit.jsonl")}}


def context_exposure(packed: PackedDocuments) -> dict:
    """How much foreign context the supervised tokens are conditioned on.

    Per supervised token, the number of tokens of OTHER documents earlier in its
    sequence (zero under isolation by construction). Two training sets with
    different document lengths packed the same way differ here, which is a
    difference in their training distributions to report beside any comparison.
    """
    per_doc = [(m["foreign_context"], m["supervised"]) for ms in packed.members for m in ms]
    total = sum(s for _, s in per_doc)
    docs_per_seq = [len(ms) for ms in packed.members]
    if not total:
        return {"strategy": packed.strategy, "supervised_tokens": 0}
    ordered = sorted(per_doc)

    def quantile(q):
        acc, cut = 0, q * total
        for f, s in ordered:
            acc += s
            if acc >= cut:
                return f
        return ordered[-1][0]

    return {"strategy": packed.strategy, "supervised_tokens": total,
            "documents_per_sequence_mean": round(sum(docs_per_seq) / len(docs_per_seq), 4) if docs_per_seq else None,
            "documents_per_sequence_max": max(docs_per_seq, default=0),
            "share_of_supervised_tokens_with_foreign_context": round(
                sum(s for f, s in per_doc if f > 0) / total, 6),
            "foreign_context_tokens_per_supervised_token": {
                "mean": round(sum(f * s for f, s in per_doc) / total, 2),
                "p50": quantile(0.5), "p90": quantile(0.9), "max": ordered[-1][0]}}


@torch.no_grad()
def isolation_report(model, packed: PackedDocuments, docs: Mapping[str, TokenDocument], *,
                     sequences: Sequence[int] = (0,), dtype=None, hidden: bool = False) -> dict:
    """Packed forward against each document run alone, at the documents' target positions.

    The teacher-forced conditional-context check for ``document_isolated``
    packing: every logit a target is trained against must be the logit of the
    document on its own.
    Returns the largest absolute difference per sequence and overall.
    ``hidden=True`` compares the final hidden states instead (the logits are a
    fixed linear map of them), which is what a large vocabulary on a host needs.
    """
    if packed.strategy != "document_isolated":
        raise ValueError("isolation_report checks document_isolated packing only")
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
