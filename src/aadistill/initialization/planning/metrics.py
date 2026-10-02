"""The evaluator that measures a state, and turns distortion into metrics.

The measurement CONTRACTS -- what a metric name means, what a suite is, what a
result looks like -- live in `aadistill.initialization.specs.metrics`, one layer
down, because the operators and the state spec need them and must not depend on
this module to get them.

**The beam reads what this produces, so the scoring positions have to reach
here.** A candidate chosen by target-aware operators and then pruned by a
full-sequence beam metric would be an experiment with two different answers to
one question. `StateEvaluator` therefore takes the same
`ScoringPositionPolicy` object the operators were handed, applies it to the
suite's own items, and stamps its id and hash into the evaluation's `detail` so a
ranked result states which positions it ranked on.

The default policy is the incumbent one and is numerically inert: `distortion`
receives `weights=None` and performs the operations the state-eval drift
certification measured to 9.032e-06 of its 1e-5 budget. A multiply by 1.0 is not
something to spend the remainder on.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from aadistill.initialization.specs.metrics import (
    _jsonable,
    DEFAULT_REFERENCE_CACHE_BUDGET_BYTES,
    MeasurementError,
    ReferenceStrategy,
    StateEvalSuite,
    StateEvaluation,
    SuiteItem,
    reference_cache_bytes,
)
from aadistill.initialization.calibration.batching import resolve_pad_id
from aadistill.initialization.calibration.packing import (
    ORIGINAL_ORDER_V1,
    item_lengths,
    packed_batches,
    padding_profile,
)
from aadistill.initialization.execution import ExecutionConfig
from aadistill.initialization.scoring.positions import (
    ALL_POSITIONS_V1,
    PREDICTION_AXIS,
    ScoringPositionPolicy,
)
from aadistill.initialization.statistics.contribution import (
    DistortionSums,
    distortion,
    domain_balanced_score,
)

#: The evaluator's default, and deliberately NOT
#: `aadistill.initialization.execution.DEFAULT_EXECUTION`. That value is 4, which
#: is the right default for an operator's statistics pass and would silently
#: change what every existing caller of this evaluator does. One item per forward
#: in the suite's own order is the path every committed state measurement was
#: produced by; a caller that wants the batched one asks for it.
REFERENCE_EXECUTION = ExecutionConfig(micro_batch_size=1,
                                      calibration_batch_packing=ORIGINAL_ORDER_V1)

#: Ceiling on the two logit blocks a batched evaluation materializes at once.
#: Checked BEFORE the first forward, because the alternative is discovering the
#: limit as an OOM on a paid pod — the same argument `CACHE_IN_MEMORY`'s budget
#: already makes, and the same failure that killed the causal-depth rehearsal.
#:
#: 12 GiB: the frozen 80-item suite at batch 3 and `length_sorted_v1` needs
#: 3.40 GiB for both models' bf16 logits over its widest group (2002 tokens at a
#: 151,936 vocabulary), so this leaves room for a wider suite or a larger batch
#: without leaving room for a mistake. It bounds the LOGIT BLOCKS only; the
#: reduction's own float32 chunks are bounded by `chunk` and are ~0.87 GiB per
#: item at that width, which is why they are not in this number.
DEFAULT_STATE_EVAL_BATCH_BUDGET_BYTES = 12 * 2**30


class StateEvaluator:
    """Scores a candidate checkpoint against the **original teacher**.

    Every checkpoint is measured on its own weights and the result is stamped
    with that artifact's digest. The teacher is the global reference for every
    state, at every depth, which is what makes states from different paths
    comparable at all.
    """

    def __init__(
        self,
        suite: StateEvalSuite,
        items: Sequence[SuiteItem],
        *,
        device: str = "cpu",
        chunk: int = 512,
        reference_strategy: ReferenceStrategy = ReferenceStrategy.RECOMPUTE,
        cache_budget_bytes: int = DEFAULT_REFERENCE_CACHE_BUDGET_BYTES,
        vocab_size: int | None = None,
        position_policy: ScoringPositionPolicy = ALL_POSITIONS_V1,
        execution: ExecutionConfig = REFERENCE_EXECUTION,
        batch_budget_bytes: int = DEFAULT_STATE_EVAL_BATCH_BUDGET_BYTES,
    ) -> None:
        if not items:
            raise MeasurementError(f"{suite.qualified_id}: no items to score")
        declared = {(d, s) for d, subs in suite.subtypes.items() for s in subs}
        seen = {(i.domain, i.subtype) for i in items}
        unknown = sorted(seen - declared)
        if unknown:
            raise MeasurementError(
                f"{suite.qualified_id}: items carry undeclared (domain, subtype) {unknown}")
        missing = sorted(declared - seen)
        if missing:
            raise MeasurementError(
                f"{suite.qualified_id}: declared sub-types with no items {missing}; "
                "a silently absent sub-type reweights its domain")
        self.suite = suite
        self.items = list(items)
        self.device = device
        self.chunk = chunk
        self.reference_strategy = reference_strategy
        self.position_policy = position_policy
        #: Evaluated ONCE, at construction, against the suite's own items — the
        #: suite is fixed for the life of the evaluator and a search calls
        #: `evaluate` once per candidate, so re-deriving the masks per candidate
        #: would repeat the same tag normalization hundreds of times.
        #:
        #: `None` under the incumbent policy, which is what `distortion` reads as
        #: "take the certified arithmetic". `require_nonempty` has already run, so
        #: a suite containing an item this policy cannot score is refused here
        #: rather than producing a zero denominator mid-search.
        self._weights: dict[str, torch.Tensor] | None = None
        if position_policy.policy_hash != ALL_POSITIONS_V1.policy_hash:
            self._weights = {}
            for item in self.items:
                w = position_policy.weights(item, axis=PREDICTION_AXIS)
                w.require_nonempty()
                self._weights[item.item_id] = w.weights
        self.execution = execution
        self.batch_budget_bytes = int(batch_budget_bytes)
        #: THE REFERENCE PATH, named once. One item per forward in the suite's own
        #: order with no reassembly — the loop every committed state measurement
        #: ran, and the one the drift certification measured. Anything else
        #: batches the two FORWARDS and still reduces per item through the same
        #: `distortion` call, so the certified arithmetic is not on the diff
        #: either way.
        self._reference_path = (int(execution.micro_batch_size) <= 1
                                and execution.calibration_batch_packing
                                == ORIGINAL_ORDER_V1)
        #: Built at the first `evaluate`, because the pad id comes from a model.
        #: Built ONCE and reused for every candidate: the grouping depends only on
        #: the suite's lengths, and a grouping that could differ between
        #: candidates would make comparing them depend on something other than
        #: the candidate. The same rule `depth.causal_kl_greedy_v1` states.
        self._groups: list[Any] | None = None
        self._teacher = None
        self._ref_logits: dict[str, torch.Tensor] = {}
        self._ref_ready = False

        if reference_strategy is ReferenceStrategy.CACHE_IN_MEMORY:
            if vocab_size is None:
                raise MeasurementError(
                    "CACHE_IN_MEMORY needs vocab_size to check its budget before "
                    "allocating; a budget checked after the fact is an OOM")
            needed = reference_cache_bytes(items, vocab_size)
            if needed > cache_budget_bytes:
                raise MeasurementError(
                    f"caching the reference logits for this suite would take "
                    f"{needed / 2**30:.1f} GiB, over the {cache_budget_bytes / 2**30:.1f} "
                    "GiB budget. Use ReferenceStrategy.RECOMPUTE: one teacher forward "
                    "per candidate is seconds, and it does not scale with vocabulary.")
            #: THE BUDGET NOW MEANS DEVICE MEMORY when `device` is not the host,
            #: because `prime_reference` keeps the cached references where the
            #: reduction runs. Stated rather than left to be discovered: the
            #: same number used to bound host RAM, and a caller that had sized
            #: it against 64 GiB of system memory would be sizing it against a
            #: GPU. The frozen suite needs 33.8 GiB and this budget is 2 GiB, so
            #: the real search is refused here and runs RECOMPUTE either way --
            #: which is why no committed run's behaviour changes.
            self._cache_on_device = device != "cpu"
            self._cache_bytes = needed

    @torch.no_grad()
    def prime_reference(self, teacher) -> None:
        """Bind the original teacher, and cache its logits only if asked to."""
        self._teacher = teacher
        if self.reference_strategy is ReferenceStrategy.CACHE_IN_MEMORY:
            for item in self.items:
                ids = item.input_ids.to(self.device)
                #: On the COMPUTE DEVICE, not the host. See `_reference_for`.
                self._ref_logits[item.item_id] = (
                    teacher(ids).logits[0, :-1].float())
        self._ref_ready = True

    @torch.no_grad()
    def _reference_for(self, item: SuiteItem) -> torch.Tensor:
        """The reference logits, left ON THE COMPUTE DEVICE.

        These used to be `.float().cpu()`. Both tensors are `[T_pred, V]` with
        V ~152k, so at the frozen suite that is ~0.5 GiB crossing the bus per
        model per item -- and then the whole float32 reduction ran on the host,
        where a 152k-wide log-softmax is orders of magnitude slower than on the
        accelerator that had just produced the logits.
        `state_evaluation_seconds` is 67-80% of every non-DEPTH expansion in
        both committed telemetry files, so this is where that time went.

        `distortion` is device-agnostic and accumulates on the device when the
        logits are not on the host, returning only reduced scalars -- which is
        what the DEPTH operator has always relied on.
        """
        if self.reference_strategy is ReferenceStrategy.CACHE_IN_MEMORY:
            return self._ref_logits[item.item_id]
        ids = item.input_ids.to(self.device)
        return self._teacher(ids).logits[0, :-1].float()

    # --- how the two forwards are issued ------------------------------------

    def batch_plan(self, vocab_size: int, bytes_per_logit: int = 2) -> dict[str, Any]:
        """What the batched path will materialize, derived at `$0` from lengths.

        Separate from the evaluation and callable before any model exists, so a
        preflight can price and bound the protocol from the frozen suite instead
        of discovering it as an allocation failure. ``bytes_per_logit`` is the
        candidate's dtype width — 2 for bf16/fp16, 4 for fp32.

        The figure that matters is ``peak_logit_bytes``: the two blocks alive at
        once, over the batch's WIDEST group rather than its average, because the
        allocation that fails is the largest one. Averages are not bounds.

        **What it bounds, exactly.** The two logit blocks. It does NOT include
        the reduction's own transients, which are bounded by ``chunk`` instead —
        and that separation is only true because the rows are handed to
        ``distortion`` in the model's own dtype rather than eagerly upcast; an
        eager ``.float()`` would add two full ``[L-1, V]`` float32 copies per
        item, ~2.4 GiB at the frozen suite's widest, to a figure that does not
        count them. See ``_logit_pairs``.
        """
        lengths = [int(i.input_ids.shape[1]) for i in self.items]
        profile = padding_profile(
            lengths, max(int(self.execution.micro_batch_size), 1),
            packing=self.execution.calibration_batch_packing)
        width = int(profile["max_group_width"])
        rows = max(int(self.execution.micro_batch_size), 1)
        peak = 2 * rows * max(width - 1, 1) * int(vocab_size) * int(bytes_per_logit)
        return {**profile, "reference_path": self._reference_path,
                "peak_logit_bytes": peak,
                "budget_bytes": self.batch_budget_bytes,
                "within_budget": peak <= self.batch_budget_bytes,
                "bytes_per_logit": int(bytes_per_logit),
                "vocab_size": int(vocab_size)}

    def _ensure_groups(self, model) -> list[Any]:
        if self._groups is not None:
            return self._groups
        vocab = int(getattr(model.config, "vocab_size", 0) or 0)
        itemsize = next(model.parameters()).dtype.itemsize
        plan = self.batch_plan(vocab, itemsize)
        if not plan["within_budget"]:
            raise MeasurementError(
                f"a batched state evaluation at batch "
                f"{self.execution.micro_batch_size} / "
                f"{self.execution.calibration_batch_packing} would hold "
                f"{plan['peak_logit_bytes']:,} bytes of logits for both models "
                f"over its widest group ({plan['max_group_width']} tokens at a "
                f"{vocab} vocabulary), over the "
                f"{self.batch_budget_bytes:,}-byte budget. Lower the batch size "
                "or raise the budget deliberately; discovering this as an OOM "
                "mid-search is what the budget exists to prevent.")
        #: Items in the SUITE's order, as mappings `packed_batches` understands.
        #: `SuiteItem` is a dataclass, not a mapping, so the shim is here rather
        #: than in the batcher: the batcher's contract is tokens, and teaching it
        #: about suite items would couple it to the evaluator.
        shim = [{"input_ids": i.input_ids, "item_id": i.item_id} for i in self.items]
        self._groups = list(packed_batches(
            shim, max(int(self.execution.micro_batch_size), 1),
            packing=self.execution.calibration_batch_packing,
            pad_id=resolve_pad_id(model), device=self.device))
        return self._groups

    @torch.no_grad()
    def _logit_pairs(self, model):
        """``(item, reference, candidate)`` per item, however the forwards ran.

        **The reduction is identical on both paths and that is the point.** The
        reference path issues one forward per item; the batched path issues one
        forward per GROUP and slices each row back to its own ``[L-1, V]`` block.
        Either way what reaches :func:`distortion` is one item's own logits at
        one item's own shape, through the same call with the same chunk
        boundaries — so batching moves the forward and the drift certification's
        subject is not on the diff.

        What batching does move is the forward's own batch shape, and this
        project has measured bf16 GEMMs reducing shape-dependently. That is a
        numerical execution property, recorded in the fingerprint, and not a
        change of estimand.
        """
        if self._reference_path:
            for item in self.items:
                ids = item.input_ids.to(self.device)
                # One item's reference and candidate logits exist at a time. At
                # the intended suite that is ~0.5 GiB each rather than 33.8 GiB
                # held for the whole run.
                yield item, self._reference_for(item), \
                    model(ids).logits[0, :-1].float()
            return

        by_id = {i.item_id: i for i in self.items}
        for packed in self._ensure_groups(model):
            batch = packed.batch
            ids = batch.input_ids.to(self.device)
            cand_block = model(ids, attention_mask=batch.attention_mask.to(
                self.device)).logits[:, :-1]
            cand_rows = batch.split_predictions(cand_block)
            ref_rows = self._reference_rows(batch, ids)
            for row, index in enumerate(packed.original_indices):
                item = by_id[batch.items[row]["item_id"]]
                if item is not self.items[index]:
                    raise MeasurementError(
                        f"row {row} of a packed group claims item "
                        f"{item.item_id!r} at suite index {index}, which holds "
                        f"{self.items[index].item_id!r}; a permutation this "
                        "evaluator cannot trust would score one item's logits "
                        "under another's tags")
                #: HANDED OVER IN THE MODEL'S DTYPE, not eagerly upcast, and
                #: that is a memory decision rather than a numerical one.
                #: `distortion` already does `[a:b].float()` on each chunk, and
                #: bfloat16 -> float32 is exact — same exponent width, more
                #: mantissa — so the reduction receives identical float32 values
                #: either way. What differs is the peak: a full `[L-1, V]`
                #: float32 copy of each row is ~1.2 GiB at the frozen suite's
                #: widest item and a 151,936 vocabulary, times two models, ON
                #: TOP of the batch's own blocks. Upcasting per chunk instead
                #: keeps the transient proportional to `chunk`.
                #:
                #: The reference path above keeps its eager `.float()`. It is
                #: the line the drift certification measured, it holds one item
                #: at a time so the saving does not arise, and touching it to
                #: tidy a batched path's memory would spend a budget for
                #: nothing.
                yield item, ref_rows[row], cand_rows[row]
            del cand_block, cand_rows, ref_rows

    @torch.no_grad()
    def _reference_rows(self, batch, ids) -> list[torch.Tensor]:
        """The intact teacher's per-row logits for one group.

        Under ``CACHE_IN_MEMORY`` the per-item references already exist, so no
        forward is issued and the cached tensors are returned in row order —
        reassembling them into a padded block only to slice it again would cost a
        copy of the whole block for nothing.
        """
        if self.reference_strategy is ReferenceStrategy.CACHE_IN_MEMORY:
            return [self._ref_logits[item["item_id"]] for item in batch.items]
        block = self._teacher(ids, attention_mask=batch.attention_mask.to(
            self.device)).logits[:, :-1]
        return batch.split_predictions(block)

    @torch.no_grad()
    def evaluate(self, model, artifact_digest: str, *, reference: str = "root_teacher",
                 runtime: Mapping[str, Any] | None = None) -> StateEvaluation:
        if not self._ref_ready or self._teacher is None:
            raise MeasurementError(
                "the reference teacher was never bound; a candidate cannot be scored "
                "against a teacher that was not run")
        if not artifact_digest:
            raise MeasurementError("refusing to measure without the artifact digest")

        per_subtype: dict[str, DistortionSums] = {}
        totals = DistortionSums()
        for item, ref, cand in self._logit_pairs(model):
            if cand.shape != ref.shape:
                raise MeasurementError(
                    f"item {item.item_id}: candidate logits {tuple(cand.shape)} do not "
                    f"match the reference {tuple(ref.shape)}; the two models are not "
                    "logit-comparable, so no KL against the original teacher exists")
            #: On the device the logits are on: they meet those logits inside
            #: `distortion`'s `gather`, and a host target would drag the whole
            #: reduction back to the host -- which is what the DEPTH operator
            #: documents having already been bitten by.
            targets = item.input_ids[0, 1:].to(ref.device)
            #: Tag masks on the logits' device too. They index `per_pos`
            #: inside `distortion`, and a host boolean mask indexing a CUDA
            #: tensor RAISES -- so this line is not an optimization, it is what
            #: makes the device-resident path run at all. `SuiteItem.tags` are
            #: built on the host by the battery renderer, which is correct;
            #: moving them is the consumer's job and it is cheap (one bool per
            #: position, against ~152k floats per position of logits).
            tags = {name: mask.to(ref.device) for name, mask in item.tags.items()}
            #: On the logits' device for the same reason the tags are: the
            #: weights multiply a per-position vector inside `distortion`, and a
            #: host float vector meeting a CUDA tensor raises.
            weights = (None if self._weights is None
                       else self._weights[item.item_id].to(ref.device))
            sums = distortion(ref, cand, targets, tags=tags, weights=weights,
                              chunk=self.chunk)
            per_subtype.setdefault(item.subtype, DistortionSums()).merge(sums)
            totals.merge(sums)
            del ref, cand

        subtype_kl = {k: v.as_dict()["kl"] for k, v in per_subtype.items()}
        domain_map = {d: list(self.suite.subtypes[d]) for d in self.suite.domains}
        primary, per_domain = domain_balanced_score(subtype_kl, domain_map)

        agg = totals.as_dict()
        values: dict[str, float] = {
            "state.teacher_kl.equal_domain_mean": float(primary),
            "state.teacher_kl.worst_domain": float(max(per_domain.values())),
            "state.teacher_kl.token_mean": float(agg["kl"]),
            "state.reverse_kl.token_mean": float(agg["reverse_kl"]),
            # Pooled over every declared domain — reasoning, code and tool text
            # included. Named for what it is; it is NOT "general NLL", and the
            # earlier key that claimed to be was this quantity.
            "state.nll.pooled_all_domains": float(agg["abl_ce"]),
            "state.nll.teacher_reference_pooled": float(agg["ref_ce"]),
            "state.nll_delta_vs_teacher_pooled": float(agg["ce_delta"]),
            "state.top1_agreement": float(agg["top1_agreement"]),
        }
        for domain, score in per_domain.items():
            values[f"state.teacher_kl.{domain}"] = float(score)

        # Per-domain candidate NLL, and `state.nll.general` from the general
        # domain alone. Omitted entirely when the suite declares no general
        # domain, rather than falling back to the pooled number.
        per_domain_ce = _per_domain_ce(per_subtype, domain_map)
        for domain, ce in per_domain_ce.items():
            values[f"state.nll.{domain}"] = float(ce)
        general = self.suite.general_domain
        if general and general in per_domain_ce:
            values["state.nll.general"] = float(per_domain_ce[general])

        tagged = agg["tagged"]
        for tag, entry in tagged.items():
            if entry["kl"] is not None:
                values[f"state.critical_token_kl.{tag}"] = float(entry["kl"])
        present = [tagged[t]["kl"] for t in self.suite.critical_tags
                   if t in tagged and tagged[t]["kl"] is not None]
        if present:
            # Unweighted mean over declared critical-token classes: a token-count
            # mean would be dominated by whichever class is common, and the rare
            # ones (think_close, eos) are the ones that decide termination.
            values["state.critical_token_kl"] = float(sum(present) / len(present))

        return StateEvaluation(
            artifact_digest=artifact_digest,
            suite_id=self.suite.qualified_id,
            suite_hash=self.suite.suite_hash,
            reference=reference,
            values=values,
            positions=int(agg["positions"]),
            detail={"per_subtype_kl": subtype_kl, "per_domain_kl": per_domain,
                    "per_domain_nll": per_domain_ce, "tagged": tagged,
                    "reference_strategy": self.reference_strategy.value,
                    #: WHICH POSITIONS these numbers are means over. Recorded in
                    #: the evaluation rather than left to the run config, because
                    #: this object is what the beam ranks on and what a journal
                    #: restores — a ranked value whose position set is only
                    #: inferable from elsewhere is a value a reader cannot check.
                    #: `scored_weight` is the denominator; `positions` above is
                    #: the count, and under a restriction they differ.
                    "position_policy": self.position_policy.qualified_id,
                    "position_policy_hash": self.position_policy.policy_hash,
                    "scored_weight": float(agg["weight"]),
                    #: HOW the forwards were issued. Execution evidence, not
                    #: identity — it changes no estimand — but a state metric
                    #: whose forwards were batched should say so, because the
                    #: batch shape is a numerical condition and the fingerprint
                    #: that binds it lives on the state, not here.
                    "execution": self.execution.as_trace(),
                    "reference_path": self._reference_path},
            runtime=dict(runtime or {}),
        )


def _per_domain_ce(per_subtype: Mapping[str, DistortionSums],
                   domains: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Equal-sub-type mean candidate CE per domain.

    Same two-level unweighted aggregation as the KL, for the same reason: a
    token-weighted domain mean is a mean over whichever sub-type tokenizes
    longest.
    """
    out: dict[str, float] = {}
    for domain, subtypes in domains.items():
        values = [per_subtype[s].as_dict()["abl_ce"] for s in subtypes
                  if s in per_subtype]
        if len(values) == len(list(subtypes)) and values:
            out[domain] = sum(values) / len(values)
    return out


