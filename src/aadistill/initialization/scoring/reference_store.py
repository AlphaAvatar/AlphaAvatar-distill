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

**The union-support store** (an objective of the union-KL form, on
``reference_student_union_topk_tail_v1``) keeps two more per-position values:
the root's final hidden state ``[T, H]`` (in the model's dtype, exactly what
its output head reads) and its float32 full-vocabulary log-normalizer ``[T]``.
With the root's output head (an adapter capability,
:data:`~aadistill.initialization.specs.arch.Capability.OUTPUT_HEAD_LINEAR`)
they recover the root's log probability of ANY entry at any stored position --
which the union KL needs for the candidate-only entries -- without a teacher
forward per candidate and without ``O(T*V)`` logits. Its sketch, entropy and
confidence are all formed from the float32 head logits
(:data:`~aadistill.initialization.scoring.output_head.LOGIT_DEFINITION`), its
Top-K by :func:`~aadistill.initialization.scoring.support.deterministic_topk`,
and its confidence is the ``K+1``-category one the union form declares. The
identity gains a ``build`` block stating what the builder actually did (head,
adapter capability, dtype, device type, micro-batch, packing): bf16 forwards
are batch-shape dependent, so the execution is part of what these values are.

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
    REFERENCE_TOPK_CATEGORIES, REFERENCE_TOPK_CONFIDENCE_MAPPING,
    REFERENCE_TOPK_ENTROPY_DEFINITION, confidence_from_vocabulary_entropy,
    full_vocabulary_entropy, reference_topk_confidence,
)
from aadistill.initialization.scoring.support import (
    DistributionSupport, ReferenceDistributionSketch, TeacherRetrieval,
    sketch_reference, sketch_reference_deterministic,
)

SCHEMA = "aadistill.root_reference_store/v1"
#: The union-support store: same per-item views plus retrieval state.
UNION_SCHEMA = "aadistill.root_reference_store/v2"
RETRIEVAL_DEFINITION = "final_hidden_state_and_full_vocab_log_normalizer/v1"


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
    #: Union-support stores only: the root's final hidden state [T, H] in the
    #: model's dtype, and its float32 full-vocabulary log-normalizer [T].
    hidden: torch.Tensor | None = None
    log_z: torch.Tensor | None = None


@dataclass
class RootReferenceStore:
    """Per item: the root's sketch, entropy, confidence and diagnostics."""

    identity: dict[str, Any]
    vocab_size: int
    top_k: int
    has_tail: bool
    entries: dict[str, _Entry] = field(default_factory=dict)
    hits: int = 0
    #: The root's output head (a `LinearOutputHead`), for union-support
    #: stores; `None` otherwise. Retrieval refuses without it.
    head: Any = None
    head_checksum: float | None = None

    @property
    def identity_hash(self) -> str:
        return sha256_json(self.identity)

    @property
    def retrieves(self) -> bool:
        """Whether this store can recover the root's probability of any entry."""
        return self.head is not None

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
        self.verify_head()

    def require_identity(self, **expected: Any) -> None:
        """The store's declared identity agrees with what a consumer expects."""
        bad = {k: (self.identity.get(k), v) for k, v in expected.items()
               if v is not None and self.identity.get(k) != v}
        if bad:
            raise ReferenceStoreError(
                f"root reference {self.identity_hash[:12]} does not match its "
                f"consumer: {bad}")

    def require_execution(self, *, micro_batch_size: int, packing: str,
                          device_type: str, model_dtype: str) -> None:
        """The store was built under the numerical protocol its consumer runs.

        Only a store whose identity declares an ``execution`` block can answer
        (:func:`root_reference_identity` with ``execution=``); one that declares
        none is refused rather than assumed to agree. bf16 forwards are
        batch-shape dependent, so the micro-batch, the packing, the device
        type and the dtype are part of what the stored values are.
        """
        declared = self.identity.get("execution")
        if not isinstance(declared, Mapping):
            raise ReferenceStoreError(
                f"root reference {self.identity_hash[:12]} declares no "
                "execution protocol, so it cannot be shown to hold the values "
                "its consumer would compute")
        want = {"micro_batch_size": int(micro_batch_size),
                "calibration_batch_packing": str(packing),
                "device_type": str(device_type),
                "model_dtype": str(model_dtype)}
        bad = {k: (declared.get(k), v) for k, v in want.items()
               if declared.get(k) != v}
        if bad:
            raise ReferenceStoreError(
                f"root reference {self.identity_hash[:12]} was built under "
                f"another execution protocol than its consumer runs: {bad}")

    def _entry(self, item: Mapping[str, Any]) -> _Entry:
        entry = self.entries.get(str(item["item_id"]))
        if entry is None:
            raise ReferenceStoreError(
                f"item {item['item_id']!r} is not in this root reference")
        return entry

    # --- per-item, per-block and per-row views (DEPTH's three paths) -------

    def sketch_item(self, item: Mapping[str, Any],
                    rows: torch.Tensor | None = None
                    ) -> ReferenceDistributionSketch:
        """One item's sketch; at the rows a ``[T]`` bool mask keeps, when given."""
        e = self._entry(item)
        self.hits += 1
        pick = ((lambda t: t) if rows is None else
                (lambda t: t[rows.to(t.device).bool()]))
        return ReferenceDistributionSketch(
            support_indices=pick(e.support_indices),
            support_log_probs=pick(e.support_log_probs),
            tail_log_prob=pick(e.tail_log_prob),
            top1_token=pick(e.top1_token),
            target_log_prob=pick(e.target_log_prob), top_k=self.top_k,
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

    # --- retrieval views (union support) -----------------------------------
    #
    # Each is aligned to the sketch view of the same name: the same rows in the
    # same order, so position n of the sketch, of the retrieval and of the
    # candidate is the same prediction of the same item.

    def _require_retrieval(self) -> None:
        if self.head is None:
            raise ReferenceStoreError(
                f"root reference {self.identity_hash[:12]} holds no retrieval "
                "state; it was built for a support that needs none")

    def verify_head(self) -> None:
        """The head still holds the weights the store was built from.

        Checked once per consumer invocation (`require_items`), not per view:
        the probe synchronizes with the device."""
        if (self.head is not None and self.head_checksum is not None
                and self.head.checksum() != self.head_checksum):
            raise ReferenceStoreError(
                f"root reference {self.identity_hash[:12]}: the output head "
                "no longer holds the weights the store was built from")

    def retrieval_item(self, item: Mapping[str, Any]) -> TeacherRetrieval:
        self._require_retrieval()
        e = self._entry(item)
        return TeacherRetrieval(e.hidden, e.log_z, self.head)

    def retrieval_block(self, batch: Any) -> TeacherRetrieval:
        """``[B * W]`` rows, row-major, zero on padding (never read)."""
        self._require_retrieval()
        entries = [self._entry(item) for item in batch.items]
        width = int(batch.input_ids.shape[1]) - 1
        h0 = entries[0].hidden
        hidden = torch.zeros(len(entries), width, int(h0.shape[-1]),
                             dtype=h0.dtype, device=h0.device)
        log_z = torch.zeros(len(entries), width, dtype=torch.float32,
                            device=h0.device)
        for i, e in enumerate(entries):
            n = int(e.hidden.shape[0])
            hidden[i, :n] = e.hidden
            log_z[i, :n] = e.log_z
        return TeacherRetrieval(hidden.reshape(-1, int(h0.shape[-1])),
                                log_z.reshape(-1), self.head)

    def retrieval_rows(self, batch: Any, rows: torch.Tensor
                       ) -> TeacherRetrieval:
        """The selected rows, row-major -- `sketch_rows`' order."""
        self._require_retrieval()
        hid, lz = [], []
        for i, item in enumerate(batch.items):
            e = self._entry(item)
            n = int(e.hidden.shape[0])
            sel = rows[i, :n].to(e.hidden.device)
            hid.append(e.hidden[sel])
            lz.append(e.log_z[sel])
        return TeacherRetrieval(torch.cat(hid), torch.cat(lz), self.head)

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
            for t in (e.hidden, e.log_z):
                if t is not None:
                    h.update(t.detach().cpu().contiguous().view(torch.uint8)
                             .numpy().tobytes())
        return h.hexdigest()

    def decision(self) -> dict[str, Any]:
        positions = sum(int(e.entropy.shape[0]) for e in self.entries.values())
        out = {"mode": "root_reference_store", "cached": True,
               "identity_hash": self.identity_hash,
               #: The declaration itself, so a consumer's record states which
               #: root, mixture, support and execution its KL was measured
               #: against without a second lookup.
               "identity": dict(self.identity),
               "distribution_support": self.identity.get("support"),
               "top_k": self.top_k, "has_tail": self.has_tail,
               "items_total": len(self.entries), "positions": positions,
               "sketch_bytes": positions * (self.top_k * 12 + 4 * 5)}
        if self.retrieves:
            out["retrieval_bytes"] = sum(
                e.hidden.numel() * e.hidden.element_size()
                + e.log_z.numel() * e.log_z.element_size()
                for e in self.entries.values())
            out["output_head"] = self.head.identity()
        return out

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


def store_execution(*, micro_batch_size: int, packing: str, device: Any,
                    model: Any) -> dict[str, Any]:
    """The numerical protocol a store is built under, as identity.

    Passed to :func:`root_reference_identity` by a caller that binds it; the
    consumer checks it with :meth:`RootReferenceStore.require_execution`.
    """
    dtype = next(model.parameters()).dtype
    return {"micro_batch_size": int(micro_batch_size),
            "calibration_batch_packing": str(packing),
            "device_type": torch.device(device).type,
            "model_dtype": str(dtype).replace("torch.", "")}


def root_reference_identity(*, root_identity: Mapping[str, Any],
                            profile_hash: str,
                            support: DistributionSupport,
                            objective: Any = None,
                            execution: Mapping[str, Any] | None = None,
                            ) -> dict[str, Any]:
    """What a store means: the root, the mixture, the support, the definitions.

    Without an objective, or under the full-vocabulary reference-confidence
    form, the historical identity, byte for byte. Under the union-KL form the
    union store's: its own schema, the ``K+1``-category confidence and the
    retrieval definition. The builder appends what it actually executed.

    ``execution`` (:func:`store_execution`) binds the numerical protocol into
    a non-union identity; absent, the identity is the historical one. A
    consumer that needs the binding asks for it with
    :meth:`RootReferenceStore.require_execution`.
    """
    if objective is not None and getattr(objective, "uses_union_support",
                                         False):
        if not getattr(support, "is_union", False):
            raise ReferenceStoreError(
                f"{objective.qualified_id} needs the union support; "
                f"got {support}")
        return {"schema": UNION_SCHEMA, "root": dict(root_identity),
                "profile_hash": str(profile_hash),
                "support": support.as_dict(),
                "entropy_definition": REFERENCE_TOPK_ENTROPY_DEFINITION,
                "confidence_categories": REFERENCE_TOPK_CATEGORIES,
                "confidence_mapping": REFERENCE_TOPK_CONFIDENCE_MAPPING,
                "retrieval": RETRIEVAL_DEFINITION,
                "rows": "all_real_prediction_positions/v1"}
    return {"schema": SCHEMA, "root": dict(root_identity),
            "profile_hash": str(profile_hash), "support": support.as_dict(),
            "entropy_definition": REFERENCE_ENTROPY_DEFINITION,
            "confidence_mapping": REFERENCE_CONFIDENCE_MAPPING,
            "rows": "all_real_prediction_positions/v1",
            **({} if execution is None else {"execution": dict(execution)})}


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
    if support.is_union:
        return _build_union_store(
            model, items, support=support, identity=identity,
            micro_batch_size=micro_batch_size, packing=packing,
            device=device, chunk=chunk)
    if identity.get("schema") == UNION_SCHEMA:
        raise ReferenceStoreError(
            "a union-store identity declared for a non-union support; the "
            "store would not hold what its identity says")
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


def _root_forwards(model: Any, items: Sequence[Mapping[str, Any]], *,
                   micro_batch_size: int, packing: str, device: Any, run):
    """Yield ``(item, prediction-row tensor)`` for every item, running
    ``run(ids, attention_mask)`` -> ``[B, T, *]`` under the execution protocol:
    one unpadded item per forward at ``micro_batch_size <= 1``, else the
    canonical packed batches. The ``[B, T, *]`` output dies with its batch."""
    from aadistill.initialization.calibration.batching import resolve_pad_id
    from aadistill.initialization.calibration.packing import packed_batches

    if int(micro_batch_size) <= 1:
        for item in items:
            ids = item["input_ids"].to(device)
            yield item, run(ids, None)[0, :-1]
        return
    for packed in packed_batches(items, int(micro_batch_size),
                                 packing=packing, pad_id=resolve_pad_id(model),
                                 device=device):
        batch = packed.batch
        block = run(batch.input_ids.to(device),
                    batch.attention_mask.to(device))
        for row, (item, length) in enumerate(zip(batch.items, batch.lengths)):
            yield item, block[row, :int(length) - 1]
        del block


@torch.no_grad()
def _build_union_store(model: Any, items: Sequence[Mapping[str, Any]], *,
                       support: DistributionSupport,
                       identity: Mapping[str, Any], micro_batch_size: int,
                       packing: str, device: Any, chunk: int
                       ) -> RootReferenceStore:
    """The union-support store: one ROOT forward per batch, to final hidden
    states; float32 head logits per block of rows; deterministic sketch,
    ``K+1`` confidence, and the hidden state and log-normalizer kept."""
    from aadistill.initialization.specs.arch import (
        Capability, adapter_for_config,
    )

    if identity.get("schema") != UNION_SCHEMA:
        raise ReferenceStoreError(
            "the union support's store needs the union identity "
            "(root_reference_identity with the union-KL objective); a "
            "historical identity would describe a different store")
    if identity.get("support") != support.as_dict():
        raise ReferenceStoreError(
            f"the declared identity names support {identity.get('support')} "
            f"and the store is built for {support.as_dict()}")
    adapter = adapter_for_config(model.config)
    adapter.requires(Capability.OUTPUT_HEAD, Capability.OUTPUT_HEAD_LINEAR)
    head = adapter.linear_output_head(model)
    vocab = head.vocab_size
    k = max(1, min(int(support.top_k), vocab))
    dtype = next(model.parameters()).dtype
    build = {"output_head": head.identity(),
             "adapter": {"family": adapter.family,
                         "adapter_version": adapter.adapter_version,
                         "capabilities": [Capability.OUTPUT_HEAD,
                                          Capability.OUTPUT_HEAD_LINEAR]},
             "model_dtype": str(dtype).replace("torch.", ""),
             "device_type": torch.device(device).type,
             "micro_batch_size": int(micro_batch_size),
             "calibration_batch_packing": str(packing),
             "effective_top_k": k, "vocab_size": vocab}
    store = RootReferenceStore(identity={**dict(identity), "build": build},
                               vocab_size=vocab, top_k=k, has_tail=k < vocab,
                               head=head, head_checksum=head.checksum())
    keys = [str(i["item_id"]) for i in items]
    step = max(1, int(chunk))
    for item, hidden in _root_forwards(
            model, items, micro_batch_size=micro_batch_size, packing=packing,
            device=device,
            run=lambda ids, attn: adapter.final_hidden_states(model, ids, attn)):
        key = str(item["item_id"])
        content = item_content_sha256(item)
        if key in store.entries:
            if store.entries[key].content_sha256 != content:
                raise ReferenceStoreError(
                    f"item id {key!r} names two different contents in one "
                    "mixture; a store keyed by it would conflate them")
            continue
        targets = item["input_ids"].reshape(-1)[1:].to(hidden.device)
        parts = []
        for start in range(0, int(hidden.shape[0]), step):
            stop = min(start + step, int(hidden.shape[0]))
            logits = head.logits(hidden[start:stop])
            parts.append(sketch_reference_deterministic(
                logits, targets[start:stop], top_k=k, chunk=step))
            del logits
        sk = [p[0] for p in parts]
        support_indices = torch.cat([x.support_indices for x in sk])
        support_log_probs = torch.cat([x.support_log_probs for x in sk])
        tail = torch.cat([x.tail_log_prob for x in sk])
        entropy, confidence = reference_topk_confidence(
            support_log_probs, tail, top_k=k, vocab_size=vocab)
        store.entries[key] = _Entry(
            support_indices=support_indices,
            support_log_probs=support_log_probs, tail_log_prob=tail,
            entropy=entropy, confidence=confidence,
            target_log_prob=torch.cat([x.target_log_prob for x in sk]),
            top1_token=torch.cat([x.top1_token for x in sk]),
            content_sha256=content,
            #: A COPY of the row slice: a view would keep the whole padded
            #: [B, T, H] batch alive for the life of the store.
            hidden=hidden.detach().clone(),
            log_z=torch.cat([p[1] for p in parts]))
    missing = sorted(set(keys) - set(store.entries))
    if missing:
        raise ReferenceStoreError(f"items not stored: {missing[:5]}")
    if not math.isfinite(sum(float(e.confidence.sum())
                             for e in store.entries.values())):
        raise ReferenceStoreError("a non-finite root confidence")
    return store


__all__ = ["RETRIEVAL_DEFINITION", "ReferenceStoreError", "RootReferenceStore",
           "SCHEMA", "UNION_SCHEMA", "build_root_reference_store",
           "item_content_sha256", "root_reference_identity", "store_execution"]
