"""The ROOT reference, computed once per calibration mixture and reused.

A scoring objective whose KL reference is the lineage ROOT (the frozen
original model, not the parent being compressed) and whose weights are that
root's own confidence needs, per calibration item and prediction position:

* the root's reference sketch on the declared support (Top-K indices, their
  log probabilities and the K+1 tail) -- the KL's reference;
* the root's FULL-vocabulary Shannon entropy, and the confidence
  ``1 - H / log |V|`` -- the weight;
* the root's log probability of the item's own next token and its top-1
  token -- diagnostics, never a decision input.

None of it depends on a candidate, so it is computed ONCE from the root model
and every consumer -- every DEPTH candidate of every expansion, every weighted
activation-statistics pass -- reads the stored values. Nothing ``[T, V]`` is
kept: the store is ``O(T*K)`` for the sketch and ``O(T)`` for the rest.

**Same forwards as the operator would run.** The root is forwarded over the
mixture's canonical batches under the run's execution protocol (micro-batch,
packing, padding), exactly as DEPTH forwards a parent, and always over ALL
prediction rows. So at the root of a search, where the parent IS the root, the
stored sketch is the sketch DEPTH would compute itself (a CPU test pins this),
and the reference is the same tensor whichever ``prediction_rows`` mode reads
it -- only the candidate side differs between modes.

**Identity is a declaration, not a tensor hash.** What a store means is the
root it came from, the mixture, the support and the definitions; the bytes of
the tensors depend on kernels and devices and are recorded only as evidence
(:meth:`RootReferenceStore.content_digest`). A replay on other hardware must
reproduce the identity, not the bits.

Nothing here names an experiment, a model family, a vocabulary or a size.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from aadistill.infrastructure.manifest import sha256_json
from aadistill.initialization.scoring.objective import (
    REFERENCE_CONFIDENCE_MAPPING, REFERENCE_ENTROPY_DEFINITION,
    confidence_from_vocabulary_entropy, full_vocabulary_entropy,
)
from aadistill.initialization.scoring.support import (
    DistributionSupport, ReferenceDistributionSketch, sketch_reference,
)

SCHEMA = "aadistill.root_reference_store/v1"


class ReferenceStoreError(RuntimeError):
    """A root reference that cannot serve the items or identity asked of it."""


def item_content_sha256(item: Mapping[str, Any]) -> str:
    """The item's token content, which is what the stored values describe."""
    ids = item["input_ids"].reshape(-1).to(torch.int64).cpu().numpy()
    return hashlib.sha256(ids.tobytes()).hexdigest()


@dataclass
class _Entry:
    support_indices: torch.Tensor      # [T, K] long
    support_log_probs: torch.Tensor    # [T, K] float32
    tail_log_prob: torch.Tensor        # [T] float32
    entropy: torch.Tensor              # [T] float32, full vocabulary
    confidence: torch.Tensor           # [T] float32
    target_log_prob: torch.Tensor      # [T] float32
    top1_token: torch.Tensor           # [T] long
    content_sha256: str


@dataclass
class RootReferenceStore:
    """Per item: the root's sketch, entropy, confidence and diagnostics."""

    identity: dict[str, Any]
    vocab_size: int
    top_k: int
    has_tail: bool
    entries: dict[str, _Entry] = field(default_factory=dict)
    hits: int = 0

    @property
    def identity_hash(self) -> str:
        return sha256_json(self.identity)

    # --- binding -----------------------------------------------------------

    def require_items(self, items: Sequence[Mapping[str, Any]]) -> None:
        """Every item present, with the content the store was built from.

        Checked once per consumer invocation rather than per lookup: a store
        built from another mixture would otherwise weight one item's positions
        by another's root confidence and still look like a measurement.
        """
        for item in items:
            entry = self.entries.get(str(item["item_id"]))
            if entry is None:
                raise ReferenceStoreError(
                    f"item {item['item_id']!r} is not in root reference "
                    f"{self.identity_hash[:12]}; it was built for another "
                    "mixture")
            if entry.content_sha256 != item_content_sha256(item):
                raise ReferenceStoreError(
                    f"item {item['item_id']!r} has content the root reference "
                    f"{self.identity_hash[:12]} was not built from")

    def require_identity(self, **expected: Any) -> None:
        """The store's declared identity agrees with what a consumer expects."""
        bad = {k: (self.identity.get(k), v) for k, v in expected.items()
               if v is not None and self.identity.get(k) != v}
        if bad:
            raise ReferenceStoreError(
                f"root reference {self.identity_hash[:12]} does not match its "
                f"consumer: {bad}")

    def _entry(self, item: Mapping[str, Any]) -> _Entry:
        entry = self.entries.get(str(item["item_id"]))
        if entry is None:
            raise ReferenceStoreError(
                f"item {item['item_id']!r} is not in this root reference")
        return entry

    # --- per-item, per-block and per-row views (DEPTH's three paths) -------

    def sketch_item(self, item: Mapping[str, Any]) -> ReferenceDistributionSketch:
        e = self._entry(item)
        self.hits += 1
        return ReferenceDistributionSketch(
            support_indices=e.support_indices,
            support_log_probs=e.support_log_probs,
            tail_log_prob=e.tail_log_prob, top1_token=e.top1_token,
            target_log_prob=e.target_log_prob, top_k=self.top_k,
            vocab_size=self.vocab_size)

    def sketch_block(self, batch: Any):
        """``([B, T, K], [B, T, K], [B, T])`` padded with zeros, never read."""
        entries = [self._entry(item) for item in batch.items]
        self.hits += len(entries)
        width = int(batch.input_ids.shape[1]) - 1
        dev = entries[0].support_indices.device
        k = int(entries[0].support_indices.shape[-1])
        idx = torch.zeros(len(entries), width, k, dtype=torch.long, device=dev)
        lp = torch.zeros(len(entries), width, k, dtype=torch.float32, device=dev)
        tail = torch.zeros(len(entries), width, dtype=torch.float32, device=dev)
        for i, e in enumerate(entries):
            n = int(e.support_indices.shape[0])
            idx[i, :n] = e.support_indices
            lp[i, :n] = e.support_log_probs
            tail[i, :n] = e.tail_log_prob
        return idx, lp, tail

    def per_position_block(self, batch: Any, name: str) -> torch.Tensor:
        """``[B, T]`` of a per-position quantity (``entropy``, ``confidence``,
        ``target_log_prob``), zero on padding."""
        entries = [self._entry(item) for item in batch.items]
        width = int(batch.input_ids.shape[1]) - 1
        out = torch.zeros(len(entries), width, dtype=torch.float32,
                          device=entries[0].entropy.device)
        for i, e in enumerate(entries):
            v = getattr(e, name)
            out[i, :int(v.shape[0])] = v.float()
        return out

    def sketch_rows(self, batch: Any, rows: torch.Tensor):
        """``([N, K], [N, K], [N])`` at the selected rows, row-major.

        The order boolean indexing of a ``[B, T]`` grid produces -- the order
        the candidate's row-selected logits come in.
        """
        parts = [[], [], []]
        for i, item in enumerate(batch.items):
            e = self._entry(item)
            n = int(e.support_indices.shape[0])
            sel = rows[i, :n].to(e.support_indices.device)
            if bool(rows[i, n:].any()):
                raise ReferenceStoreError(
                    f"rows select padding of item {item['item_id']!r}")
            parts[0].append(e.support_indices[sel])
            parts[1].append(e.support_log_probs[sel])
            parts[2].append(e.tail_log_prob[sel])
        self.hits += len(batch.items)
        return tuple(torch.cat(p) for p in parts)

    def per_position_rows(self, batch: Any, rows: torch.Tensor,
                          name: str) -> torch.Tensor:
        out = []
        for i, item in enumerate(batch.items):
            v = getattr(self._entry(item), name)
            out.append(v[rows[i, :int(v.shape[0])].to(v.device)].float())
        return torch.cat(out)

    def per_position(self, item: Mapping[str, Any], name: str) -> torch.Tensor:
        return getattr(self._entry(item), name).float()

    # --- token-axis weights for activation statistics ----------------------

    def token_confidence(self, item: Mapping[str, Any]) -> torch.Tensor:
        """``[L]`` float64: token ``j`` carries the confidence of prediction
        ``j`` (the logits its state produces); the last token feeds no
        prediction inside the item and carries zero."""
        conf = self._entry(item).confidence.detach().to("cpu", torch.float64)
        return torch.cat([conf, torch.zeros(1, dtype=torch.float64)])

    # --- evidence -----------------------------------------------------------

    def content_digest(self) -> str:
        """sha256 over every stored tensor, in item order. EVIDENCE only."""
        h = hashlib.sha256()
        for key in sorted(self.entries):
            e = self.entries[key]
            for t in (e.support_indices, e.support_log_probs, e.tail_log_prob,
                      e.entropy, e.target_log_prob, e.top1_token):
                h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def decision(self) -> dict[str, Any]:
        positions = sum(int(e.entropy.shape[0]) for e in self.entries.values())
        return {"mode": "root_reference_store", "cached": True,
                "identity_hash": self.identity_hash,
                "distribution_support": self.identity.get("support"),
                "top_k": self.top_k, "has_tail": self.has_tail,
                "items_total": len(self.entries), "positions": positions,
                "sketch_bytes": positions * (self.top_k * 12 + 4 * 5)}

    def counters(self) -> dict[str, int]:
        return {"reference_hits": self.hits, "reference_fills": 0,
                "reference_recomputes": 0}

    def confidence_report(self, items: Sequence[Mapping[str, Any]] | None = None,
                          scored: Mapping[str, torch.Tensor] | None = None,
                          ) -> dict[str, Any]:
        """Distribution of the confidence over (optionally scored) positions.

        ``scored`` maps item id to a ``[T]`` bool of the positions a policy
        scores; absent, every real prediction position counts. Reports the
        weight sum, mean weight and the effective sample size
        ``(sum w)^2 / sum w^2`` -- how many equally weighted positions the
        weighting is worth.
        """
        keys = ([str(i["item_id"]) for i in items] if items is not None
                else sorted(self.entries))
        values = []
        for key in keys:
            c = self.entries[key].confidence.detach().cpu().double()
            if scored is not None and key in scored:
                c = c[scored[key].cpu().bool()[:int(c.shape[0])]]
            values.append(c)
        c = torch.cat(values) if values else torch.empty(0, dtype=torch.float64)
        n = int(c.numel())
        if n == 0:
            return {"positions": 0}
        qs = torch.tensor([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                          dtype=torch.float64)
        quant = torch.quantile(c, qs).tolist()
        hist = torch.histc(c, bins=20, min=0.0, max=1.0).tolist()
        return {"positions": n, "mean": float(c.mean()), "sd": float(c.std()),
                "min": float(c.min()), "max": float(c.max()),
                "quantiles": {f"q{int(q * 100):02d}": v
                              for q, v in zip(qs.tolist(), quant)},
                "histogram_20_bins_0_1": [int(x) for x in hist],
                "weight_sum": float(c.sum()), "mean_weight": float(c.mean()),
                "effective_sample_size": float(c.sum() ** 2 / (c * c).sum())
                if float((c * c).sum()) > 0 else 0.0,
                "effective_fraction": (float(c.sum() ** 2 / (c * c).sum()) / n
                                       if float((c * c).sum()) > 0 else 0.0)}


def root_reference_identity(*, root_identity: Mapping[str, Any],
                            profile_hash: str,
                            support: DistributionSupport) -> dict[str, Any]:
    """What a store means: the root, the mixture, the support, the definitions."""
    return {"schema": SCHEMA, "root": dict(root_identity),
            "profile_hash": str(profile_hash), "support": support.as_dict(),
            "entropy_definition": REFERENCE_ENTROPY_DEFINITION,
            "confidence_mapping": REFERENCE_CONFIDENCE_MAPPING,
            "rows": "all_real_prediction_positions/v1"}


@torch.no_grad()
def build_root_reference_store(model: Any, items: Sequence[Mapping[str, Any]], *,
                               support: DistributionSupport,
                               identity: Mapping[str, Any],
                               micro_batch_size: int, packing: str,
                               device: Any, chunk: int = 512
                               ) -> RootReferenceStore:
    """Forward the ROOT over the mixture once and keep what every consumer needs.

    ``micro_batch_size <= 1`` forwards one unpadded item at a time, as DEPTH's
    per-item path does; otherwise the canonical packed batches, as its batched
    path does. The full-vocabulary logits of a block live only for the
    duration of that block.
    """
    from aadistill.initialization.calibration.batching import resolve_pad_id
    from aadistill.initialization.calibration.packing import packed_batches

    if support.is_full_vocab:
        raise ReferenceStoreError(
            "a root reference store holds Top-K sketches; the full-vocabulary "
            "support keeps O(T*V) logits and is not what this store is for")
    items = list(items)
    if not items:
        raise ReferenceStoreError("a root reference over no items")
    keys = [str(i["item_id"]) for i in items]
    store: RootReferenceStore | None = None

    def add(item: Mapping[str, Any], logits: torch.Tensor) -> None:
        nonlocal store
        vocab = int(logits.shape[-1])
        if store is None:
            k = max(1, min(int(support.top_k), vocab))
            store = RootReferenceStore(identity=dict(identity), vocab_size=vocab,
                                       top_k=k, has_tail=k < vocab)
        targets = item["input_ids"].reshape(-1)[1:].to(logits.device)
        sk = sketch_reference(logits, targets, top_k=store.top_k, chunk=chunk)
        entropy = full_vocabulary_entropy(logits, chunk=chunk)
        key = str(item["item_id"])
        content = item_content_sha256(item)
        if key in store.entries:
            if store.entries[key].content_sha256 != content:
                raise ReferenceStoreError(
                    f"item id {key!r} names two different contents in one "
                    "mixture; a store keyed by it would conflate them")
            return
        store.entries[key] = _Entry(
            support_indices=sk.support_indices,
            support_log_probs=sk.support_log_probs,
            tail_log_prob=sk.tail_log_prob, entropy=entropy,
            confidence=confidence_from_vocabulary_entropy(entropy, vocab),
            target_log_prob=sk.target_log_prob, top1_token=sk.top1_token,
            content_sha256=content)

    if int(micro_batch_size) <= 1:
        for item in items:
            ids = item["input_ids"].to(device)
            add(item, model(ids).logits[0, :-1])
    else:
        for packed in packed_batches(items, int(micro_batch_size),
                                     packing=packing,
                                     pad_id=resolve_pad_id(model),
                                     device=device):
            batch = packed.batch
            block = model(batch.input_ids.to(device),
                          attention_mask=batch.attention_mask.to(device)
                          ).logits[:, :-1]
            for row, (item, length) in enumerate(zip(batch.items,
                                                     batch.lengths)):
                add(item, block[row, :int(length) - 1])
            del block
    assert store is not None
    missing = sorted(set(keys) - set(store.entries))
    if missing:
        raise ReferenceStoreError(f"items not stored: {missing[:5]}")
    if not math.isfinite(sum(float(e.confidence.sum())
                             for e in store.entries.values())):
        raise ReferenceStoreError("a non-finite root confidence")
    return store


__all__ = ["ReferenceStoreError", "RootReferenceStore", "SCHEMA",
           "build_root_reference_store", "item_content_sha256",
           "root_reference_identity"]
