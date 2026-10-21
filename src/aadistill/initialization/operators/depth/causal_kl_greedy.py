"""``depth.causal_kl_greedy_v1`` — iterative greedy block removal by forward KL.

Wraps :mod:`aadistill.initialization.statistics.contribution`: block bypass,
forward KL against the *unbypassed parent*, domain-balanced aggregation, greedy
removal with a stated tie-break. It is re-run against whatever checkpoint it is
handed, which is what makes it conditional rather than a precomputed teacher
decision.

It has been measured to disagree almost completely with ``depth.positional_v0``
— for 36 -> 28 they share one removed layer out of eight — so both stay
registered and neither is a default.

**This is the only DEPTH implementation that runs model forwards**, and it runs
a great many: one per candidate subset per calibration item. Those forwards are
micro-batched (see :mod:`aadistill.initialization.calibration.batching`); the
per-item KL, the subtype mean and the domain balance above it are untouched by
that, because batching moves the forward and not the reduction.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from aadistill.initialization.scoring.rows import (
    scatter_rows, scored_prediction_logits, scored_rows, split_rows,
)
from aadistill.initialization.scoring.kl_reference import (
    KL_REFERENCE_KEY, ROOT_REFERENCE_V1, KLReferenceError,
    validate_kl_reference,
)
from aadistill.initialization.scoring.objective import (
    ObjectiveError, add_joint_histograms, confidence_histogram,
    item_component_means, position_terms, require_objective_support,
)
from aadistill.initialization.scoring.support import (
    SUPPORT_REFERENCE_TOPK_TAIL_V1, ReferenceDistributionSketch,
    masked_item_means, sketch_forward_kl, sketch_forward_kl_mean,
    sketch_forward_kl_mean_batch, sketch_forward_kl_terms, sketch_reference,
    union_forward_kl_terms,
)
from aadistill.initialization.statistics.contribution import (
    bypassed_blocks,
    captured_block_inputs,
    domain_balanced_score,
    has_suffix,
    resumed_after_block,
    forward_kl_mean,
    forward_kl_mean_batch,
    expected_evaluations,
    greedy_removal,
)
from aadistill.initialization.device import model_device
from aadistill.initialization.specs.arch import (
    ArchitectureAdapter,
    ArchSpec,
    Capability,
)
from aadistill.initialization.specs.metrics import OperatorLocalMetrics
from aadistill.initialization.calibration.batching import (
    build_batch,
    resolve_pad_id,
)
from aadistill.initialization.calibration.packing import packed_batches
from aadistill.initialization.calibration.profiles import CalibrationNeed
from aadistill.initialization.scoring.batches import active_positions
from aadistill.initialization.operators.base import (
    OperatorContext,
    OperatorError,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
)
from aadistill.initialization.operators.depth._common import (
    DEPTH_FIELD,
    _build_child_with_layers,
)


class DepthCausalKLGreedyV1(OperatorImplementation):
    impl_id = "depth.causal_kl_greedy_v1"
    kind = "DEPTH"
    version = 1
    description = (
        "Iterative greedy block removal minimizing forward KL(parent || "
        "parent-with-S-bypassed) over real prediction positions, aggregated as the "
        "unweighted mean over domains of the unweighted mean over each domain's "
        "sub-types. Ties break to the lower layer index. E8a's algorithm, re-run "
        "against the checkpoint it is handed rather than always against the teacher.")
    required_capabilities = frozenset({Capability.BLOCK_LIST, Capability.RESIDUAL_STREAM,
                                       Capability.LOGIT_COMPARABLE})
    modifies = frozenset({DEPTH_FIELD})
    preserves = frozenset({"hidden_size", "intermediate_size", "num_attention_heads",
                           "num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.FORWARD_LOGITS
    objective = "forward KL(parent || parent-with-S-bypassed), domain-balanced"
    #: Under a run's `ScoringObjective` the per-position quantity this averages
    #: is that objective's integrand, formed from the SAME candidate forward:
    #: the candidate here is the parent with S bypassed, so its logits are the
    #: student's and its own Top-K is read from them. See `scoring.objective`.
    consumes_scoring_objective = True
    #: The KL's reference may be DECLARED (`scoring.kl_reference`): the intact
    #: parent by default, or the stored lineage root under
    #: ``kl_reference: root_reference/v1``. The same greedy removal, the same
    #: reducer and the same candidates either way; only the distribution each
    #: candidate is compared with changes.
    accepts_kl_reference = True
    deterministic = True
    requires_seed = False
    produces = ("depth_map", "search_rounds")
    target_validation = "result num_hidden_layers equals the target exactly"

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        n_remove = spec[DEPTH_FIELD] - target[DEPTH_FIELD]
        if n_remove <= 0:
            raise OperatorError(
                f"{self.impl_id}: nothing to remove ({spec[DEPTH_FIELD]} -> "
                f"{target[DEPTH_FIELD]})")
        evals = expected_evaluations(spec[DEPTH_FIELD], n_remove)
        n_items = int((config or {}).get("n_calibration_items", 0))
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{DEPTH_FIELD: target[DEPTH_FIELD]}),
            # +1 for the intact-parent reference pass over the mixture.
            forward_passes=(evals + 1) * max(n_items, 1),
            stats_passes=0,
            notes=f"{evals} subset evaluations over {n_items} calibration items")

    @torch.no_grad()
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        adapter = ctx.adapter
        model = ctx.model
        items = list(ctx.calibration_items)
        if getattr(model.config, "use_cache", False):
            model.config.use_cache = False

        domains = _domain_map(items)
        # From the weights, not from `ctx.device`: category 1 of the device
        # contract. The two agree on every path the search takes, so this
        # changes no behaviour — it removes the last operator that read the
        # intent instead of the fact.
        compute = model_device(model)
        # NO TARGETS. They existed for `distortion`'s two cross-entropy
        # gathers, and this operator's objective is forward KL: `forward_kl_mean`
        # needs the two logit tensors and nothing else. The comment they carried
        # -- "a host target would drag the whole reduction back to the host" --
        # was correct and is now enforced one level down, where `distortion` and
        # `forward_kl_mean` both keep their accumulators wherever the logits are.
        #: WHICH REFERENCE STATE, decided by the run's declared support. The
        #: full-vocabulary path is unchanged and is what produced every frozen
        #: DEPTH decision; the sketch path holds O(T*K) instead of O(T*V).
        support = ctx.distribution_support
        #: PRE-VOCABULARY ROW SELECTION, opt-in through the execution config.
        #: Only on the reference-support path: the full-vocabulary path is the
        #: oracle every Top-K comparison is measured against, and it stays on the
        #: historical all-rows arithmetic so it remains one.
        select_rows = ctx.execution.selects_prediction_rows
        if select_rows:
            if support.is_full_vocab:
                raise OperatorError(
                    f"{self.impl_id}: prediction_rows="
                    f"{ctx.execution.prediction_rows!r} is implemented for the "
                    "reference-support reduction only. full_vocab_v1 is the "
                    "oracle the coarsened support is validated against, and it "
                    "stays on the historical all-rows path")
            adapter.requires(Capability.OUTPUT_HEAD)
        #: A STUDENT-CONFIDENCE OBJECTIVE, when the run declares one. Defined on
        #: the reference-support KL only: full_vocab_v1 is the oracle path and
        #: stays the historical KL-only objective.
        objective = ctx.scoring_objective
        if objective is not None and support.is_full_vocab:
            raise OperatorError(
                f"{self.impl_id}: scoring objective {objective.qualified_id} "
                "is defined on the reference Top-K + tail KL; full_vocab_v1 is "
                "the KL-only oracle path")
        #: THE UNION SUPPORT AND THE UNION-KL FORM COME TOGETHER, at one K:
        #: the reference's probability of a candidate-only entry is retrieved
        #: from the ROOT store, which only that form requests.
        try:
            require_objective_support(objective, support)
        except ObjectiveError as exc:
            raise OperatorError(f"{self.impl_id}: {exc}") from exc
        union = support.is_union
        #: A DECLARED KL REFERENCE (`scoring.kl_reference`). Absent, the
        #: historical parent. Validated where the config was built and again
        #: in `execute`; checked here because this is where it is read.
        try:
            kl_reference = validate_kl_reference(
                self, ctx.config, objective_consumed=objective is not None)
        except KLReferenceError as exc:
            raise OperatorError(f"{self.impl_id}: {exc}") from exc
        declared_root = kl_reference == ROOT_REFERENCE_V1
        if declared_root and support.is_full_vocab:
            raise OperatorError(
                f"{self.impl_id}: {KL_REFERENCE_KEY}={kl_reference!r} reads "
                "a stored root reference of Top-K sketches; full_vocab_v1 is "
                "the KL-only oracle path")
        #: Per candidate subset: every component's domain-balanced value, so a
        #: low objective can always be read against the raw KL it came from.
        objective_candidates: dict[str, dict[str, Any]] = {}
        #: Per candidate subset under a DECLARED root reference: the decision
        #: KL (against the root) and the same candidate's KL against the
        #: intact parent. The parent value is a diagnostic and is never read
        #: by the selection.
        reference_candidates: dict[str, dict[str, float]] = {}
        #: THE KL'S REFERENCE. Historically the intact PARENT: this operator
        #: measured how far bypassing S moves the model it is handed. An
        #: objective declaring the ROOT reference measures instead how far the
        #: candidate is from the frozen original model -- read from the root
        #: reference store built once for this mixture, never re-forwarded --
        #: and weights by that root's own confidence. The parent still
        #: produces every candidate; it is just not what they are compared to.
        root_store = None
        parent_sketches = None
        #: Under a DECLARED root reference the parent-KL diagnostic covers
        #: EVERY round's candidates, not round 0's only: the intact parent is
        #: one cached reference, so the diagnostic costs reductions, not
        #: forwards, and it is what lets the record set the parent-reference
        #: ranking beside the root one at every decision.
        parent_kl_every_round = False
        if declared_root:
            root_store = ctx.root_reference
            if root_store is None:
                raise OperatorError(
                    f"{self.impl_id}: its config declares "
                    f"{KL_REFERENCE_KEY}={kl_reference!r} and no root "
                    "reference store was handed to this invocation")
            root_store.require_items(items)
            root_store.require_identity(
                support=support.as_dict(),
                profile_hash=(None if ctx.profile is None
                              else ctx.profile.profile_hash))
            dtype = next(model.parameters()).dtype
            root_store.require_execution(
                micro_batch_size=ctx.execution.micro_batch_size,
                packing=ctx.execution.calibration_batch_packing,
                device_type=torch.device(compute).type,
                model_dtype=str(dtype).replace("torch.", ""))
            sketches = _RootSketches(root_store)
            reference = None
            parent_sketches = _ReferenceSketches(model, items, compute,
                                                 top_k=support.top_k)
            parent_kl_every_round = True
        elif objective is not None and objective.requires_root_reference:
            root_store = ctx.root_reference
            if root_store is None:
                raise OperatorError(
                    f"{self.impl_id}: {objective.qualified_id} measures KL "
                    "against the ROOT reference and no root reference store "
                    "was handed to this invocation")
            root_store.require_items(items)
            root_store.require_identity(
                support=support.as_dict(),
                profile_hash=(None if ctx.profile is None
                              else ctx.profile.profile_hash))
            if union and not root_store.retrieves:
                raise OperatorError(
                    f"{self.impl_id}: {objective.qualified_id} needs the "
                    "root's probability of candidate-only entries and root "
                    f"reference {root_store.identity_hash[:12]} holds no "
                    "retrieval state")
            sketches = _RootSketches(root_store)
            reference = None
            #: The DIAGNOSTIC parent reference: round 0's candidates are also
            #: reduced against the intact parent, so the record can set the
            #: historical parent-KL ranking beside the root one. One parent
            #: forward over the mixture, cached; not a decision input.
            parent_sketches = _ReferenceSketches(model, items, compute,
                                                 top_k=support.top_k)
        elif support.is_full_vocab:
            reference = _ReferenceLogits(model, items, compute)
            sketches = None
        else:
            sketches = _ReferenceSketches(model, items, compute,
                                          top_k=support.top_k)
            reference = None

        #: The micro-batch grouping, built ONCE and reused for every candidate
        #: subset. Two reasons it is not rebuilt per candidate: the padded id
        #: tensors are identical for all 260 of them, and a grouping that could
        #: differ between candidates would make the comparison between them
        #: depend on something other than the skip set.
        #:
        #: `batch_size == 1` keeps the original per-item calls — `reference.get`
        #: and `_forward_logits`, not their batched twins — so the reference
        #: path is the code that produced every frozen DEPTH decision, not a
        #: batched path that happens to agree with it.
        batch_size = ctx.execution.micro_batch_size
        packing = ctx.execution.calibration_batch_packing
        #: `(ItemBatch, original_indices)` pairs. The indices are what map a ROW
        #: back to its item in the mixture, and under a reordering packing policy
        #: they are not the identity — the per-item scoring weights below are
        #: looked up through them, so a row can never be scored on a neighbour's
        #: supervised positions.
        groups = [(p.batch, p.original_indices) for p in packed_batches(
            items, max(int(batch_size), 1), packing=packing,
            pad_id=(resolve_pad_id(model) if batch_size > 1 else 0),
            device=compute)]
        #: WHICH PREDICTION POSITIONS the causal KL is a mean over. The objective
        #: is `KL(parent || parent-with-S-bypassed)` averaged over positions, and
        #: this decides which: every real prediction under the incumbent policy,
        #: or only the ones the item's own objective contract supervises. It comes
        #: from the hashed config and therefore forks the state, unlike the batch
        #: size and packing above it.
        active = active_positions(items, ctx.position_policy)
        if select_rows:
            #: The reference state now holds only the scored rows; size it so.
            sketches.restrict_to_scored(active)
            if parent_sketches is not None:
                parent_sketches.restrict_to_scored(active)
        components_of = (() if objective is None else objective.components())

        # Operational timings, kept OUT of every returned metric and hash.
        #
        # The phase split is honest only across a synchronization boundary. On
        # CUDA the two forwards are asynchronous and `distortion(...).as_dict()`
        # is what forces the sync, so timing them naively would bill both
        # forwards to the reduction. Inserting syncs to fix that perturbs the
        # hot path this pass exists to make faster, so it is OPT-IN: without it
        # the totals and the counts are exact and the split is simply not
        # reported. A diagnostic run sets AADISTILL_DEPTH_SYNC_TELEMETRY=1.
        sync_split = os.environ.get("AADISTILL_DEPTH_SYNC_TELEMETRY") == "1"
        cuda_sync = (torch.cuda.synchronize
                     if sync_split and torch.device(compute).type == "cuda"
                     else None)
        #: `ablated_forwards` counts FORWARD PASSES, which at `batch_size > 1`
        #: is fewer than the number of items — that reduction is the entire
        #: point of batching, so the field keeps its literal meaning and
        #: `ablated_items` is added beside it rather than redefining it. A
        #: reader comparing this run against a pre-batching telemetry file
        #: should compare `ablated_items`, which is what `ablated_forwards`
        #: counted when every forward held one item.
        timing = {"reference_seconds": 0.0, "ablated_seconds": 0.0,
                  "distortion_seconds": 0.0, "item_seconds": 0.0,
                  "candidate_subsets": 0, "ablated_forwards": 0,
                  "ablated_items": 0, "micro_batch_size": batch_size,
                  "calibration_batch_packing": packing,
                  "distortion_calls": 0, "split_is_attributed": bool(cuda_sync)
                  or torch.device(compute).type != "cuda"}
        if select_rows:
            #: How many prediction slots reached the output head, against how
            #: many the forwards held. Telemetry only, like everything here.
            timing.update({"prediction_rows": ctx.execution.prediction_rows,
                           "scored_rows": 0, "forward_rows": 0})
        if objective is not None:
            timing["objective_seconds"] = 0.0
        if parent_sketches is not None:
            timing["parent_kl_diagnostic"] = (
                "every round's candidates" if parent_kl_every_round
                else "round 0 candidates only")
            #: The diagnostic's own reductions, so its cost can be measured
            #: apart from the decision's. Unsynchronized like every split here
            #: unless AADISTILL_DEPTH_SYNC_TELEMETRY=1.
            timing["parent_kl_seconds"] = 0.0

        def reference_kl(idx, lp, tail, abls, retrieval):
            """``(kl, log_z, union evidence | None)`` against the reference.

            Under the union support: the KL on the root's Top-K ∪ the
            candidate's own Top-K + one tail, the root's probabilities of
            candidate-only entries retrieved from the store (`retrieval()`).
            Otherwise the reference-support KL, unchanged.
            """
            if union:
                terms = union_forward_kl_terms(
                    idx, lp, tail, abls, retrieval(), top_k=support.top_k,
                    has_tail=sketches.has_tail)
                return terms.kl, terms.log_z, terms.diagnostics()
            kl, log_z = sketch_forward_kl_terms(idx, lp, tail, abls,
                                                has_tail=sketches.has_tail)
            return kl, log_z, None

        def objective_values(kl, log_z, cand_logits, tail_log, mask, weights,
                             group, components, histogram, joint,
                             rows=None, ref=None, union_terms=None
                             ) -> torch.Tensor:
            """Per-item objective means from one group's per-position KL.

            `cand_logits` are the candidate's -- the student whose confidence
            the student forms read. The reference-confidence form reads `ref`
            instead: the root's ``(entropy, confidence)`` aligned to `kl`, and
            nothing of the candidate but its KL. Every component is averaged
            over the same scored positions with the same fixed denominator,
            and kept per sub-type. On the row path the inputs are the selected
            ROWS and only the per-row scalars are scattered back; no
            ``[B, T, V]`` block is built.
            """
            t0 = time.perf_counter()
            terms = position_terms(
                objective, kl, cand_logits, log_z,
                tail_log if sketches.has_tail else None,
                reference_entropy=None if ref is None else ref[0],
                reference_confidence=None if ref is None else ref[1],
                union=union_terms)
            if rows is not None:
                terms = {k: scatter_rows(v.float(), rows)
                         for k, v in terms.items()}
            means, _ = item_component_means(terms, mask, weights,
                                            names=components_of)
            scored = mask.bool() if weights is None else (
                mask.bool() & (weights.to(mask.device) > 0))
            hist = confidence_histogram(terms[objective.confidence_key()],
                                        scored)
            for i, h in enumerate(hist):
                histogram[i] += h
            joint[0] = add_joint_histograms(joint[0], terms, scored)
            host = {k: v.tolist() for k, v in means.items()}
            for row, item in enumerate(group.items):
                bucket = components.setdefault(item["subtype"], {})
                for name in components_of:
                    bucket.setdefault(name, []).append(host[name][row])
            timing["objective_seconds"] += time.perf_counter() - t0
            return means["objective"]

        def root_ref(group, rows=None, item=None):
            """The root's (entropy, confidence) aligned to one path's KL."""
            if root_store is None:
                return None
            if item is not None:
                return (root_store.per_position(item, "entropy")[None],
                        root_store.per_position(item, "confidence")[None])
            if rows is not None:
                return (root_store.per_position_rows(group, rows, "entropy"),
                        root_store.per_position_rows(group, rows, "confidence"))
            return (root_store.per_position_block(group, "entropy"),
                    root_store.per_position_block(group, "confidence"))

        def score(skip: frozenset[int]) -> float:
            per_subtype: dict[str, list[float]] = {}
            #: Round 0 only, and only under the root reference: the same
            #: candidate logits reduced against the intact parent as well.
            parent_kl: dict[str, list[float]] | None = (
                {} if parent_sketches is not None
                and (parent_kl_every_round or progress["rounds"] == 0)
                else None)
            components: dict[str, dict[str, list[float]]] = {}
            histogram = [0] * 10
            joint: list[Any] = [None]
            timing["candidate_subsets"] += 1
            item_started = time.perf_counter()
            for group, indices in groups:
                if select_rows:
                    #: Same groups, same weights, same per-item means; only the
                    #: rows that reach the output head differ. See
                    #: `aadistill.initialization.scoring.rows`.
                    t0 = time.perf_counter()
                    batch_weights = (None if active is None else
                                     active.prediction_weights_for(group, indices))
                    mask = group.prediction_mask().to(compute)
                    rows = scored_rows(mask, batch_weights)
                    #: The reference first and OUTSIDE the bypass: it is the
                    #: intact parent, at exactly the rows the candidate scores.
                    idx, lp, tail = sketches.sketch_rows(group, rows, adapter)
                    if cuda_sync:
                        cuda_sync()
                    t1 = time.perf_counter()
                    ids = group.input_ids.to(compute)
                    attn = group.attention_mask.to(compute)
                    if skip:
                        with bypassed_blocks(model, skip):
                            abls = scored_prediction_logits(adapter, model, ids,
                                                            attn, rows)
                    else:
                        abls = scored_prediction_logits(adapter, model, ids,
                                                        attn, rows)
                    if cuda_sync:
                        cuda_sync()
                    t2 = time.perf_counter()
                    timing["reference_seconds"] += t1 - t0
                    timing["ablated_seconds"] += t2 - t1
                    timing["ablated_forwards"] += 1
                    timing["ablated_items"] += len(group.items)
                    if objective is None:
                        kl = sketch_forward_kl(idx, lp, tail, abls,
                                               has_tail=sketches.has_tail)
                        values = masked_item_means(scatter_rows(kl, rows), mask,
                                                   weights=batch_weights)
                    else:
                        kl, log_z, uterms = reference_kl(
                            idx, lp, tail, abls,
                            lambda: root_store.retrieval_rows(group, rows))
                        values = objective_values(
                            kl, log_z, abls, tail, mask, batch_weights, group,
                            components, histogram, joint, rows=rows,
                            ref=root_ref(group, rows=rows),
                            union_terms=uterms)
                    if parent_kl is not None:
                        tp = time.perf_counter()
                        pi, pl, pt = parent_sketches.sketch_rows(
                            group, rows, adapter)
                        pkl = sketch_forward_kl(pi, pl, pt, abls,
                                                has_tail=parent_sketches.has_tail)
                        for item, v in zip(group.items, masked_item_means(
                                scatter_rows(pkl, rows), mask,
                                weights=batch_weights).tolist()):
                            parent_kl.setdefault(item["subtype"], []).append(v)
                        timing["parent_kl_seconds"] += time.perf_counter() - tp
                    for item, value in zip(group.items, values.tolist()):
                        per_subtype.setdefault(item["subtype"], []).append(value)
                        timing["distortion_calls"] += 1
                    if ctx.score_observer is not None:
                        #: Execution only, as on the all-rows path. `refs` and
                        #: `abls` are ROW tensors here (`rows` says which), so an
                        #: observer that reshapes them as `[B, T, ...]` blocks
                        #: must not be attached to a row-selected invocation.
                        ctx.score_observer(
                            skip=frozenset(skip), group=group, indices=indices,
                            refs=(idx, lp, tail), abls=abls, mask=mask,
                            weights=batch_weights, values=values,
                            support=support, rows=rows)
                    timing["scored_rows"] += int(rows.sum())
                    timing["forward_rows"] += int(mask.numel())
                    timing["distortion_seconds"] += time.perf_counter() - t2
                    del abls, kl
                    continue
                # Order matters: the reference is the UNBYPASSED parent, so when
                # it is being recomputed it must not be taken inside the bypass.
                t0 = time.perf_counter()
                if sketches is not None:
                    refs = (sketches.sketch_block(group) if batch_size > 1
                            else sketches.sketch_item(group.items[0]))
                    #: The parent's reference too, OUTSIDE the bypass, when the
                    #: round-0 diagnostic is on.
                    prefs = (None if parent_kl is None else
                             (parent_sketches.sketch_block(group)
                              if batch_size > 1 else
                              parent_sketches.sketch_item(group.items[0])))
                elif batch_size > 1:
                    refs = reference.reference_block(group)
                else:
                    refs = reference.get(group.items[0])
                if cuda_sync:
                    cuda_sync()
                t1 = time.perf_counter()
                if batch_size > 1:
                    abls = _forward_logit_block(model, group, compute, skip)
                else:
                    abls = _forward_logits(model, group.items[0], compute, skip)
                if cuda_sync:
                    cuda_sync()
                t2 = time.perf_counter()
                timing["reference_seconds"] += t1 - t0
                timing["ablated_seconds"] += t2 - t1
                timing["ablated_forwards"] += 1
                timing["ablated_items"] += len(group.items)
                #: FORWARD KL ONLY, and ONE VALUE PER ORIGINAL ITEM.
                #:
                #: At batch_size > 1 the `[B, T_pred, V]` block goes straight
                #: into the masked reducer: no split into B tensors, no Python
                #: loop, no B host synchronisations. `forward_kl_mean_batch`
                #: returns `[B]` per-item means over each item's OWN valid
                #: positions -- never a token-weighted pool, which would hand a
                #: 1000-position item ten times the influence of a 100-position
                #: one and change the objective.
                #:
                #: Batching moved the FORWARD and the reduction's schedule. It
                #: did not move the estimand or the weighting: an item that
                #: shared a forward with three others still contributes exactly
                #: one KL to exactly one subtype, and the subtype/domain
                #: aggregation below is untouched.
                if batch_size > 1:
                    #: SAME estimand, same per-item weighting, same aggregation
                    #: whichever support ran. Only the partition the KL is reduced
                    #: over differs, and that difference is carried in the state
                    #: identity rather than in how the result is recorded -- which
                    #: is why the recording below is SHARED. Branching it per
                    #: support once left the sketch path computing its values and
                    #: never recording them, and the failure surfaced as a missing
                    #: subtype in the domain aggregation.
                    batch_weights = (None if active is None else
                                     active.prediction_weights_for(group, indices))
                    mask = group.prediction_mask().to(abls.device)
                    if sketches is not None and objective is not None:
                        idx, lp, tail = refs
                        kl, log_z, uterms = reference_kl(
                            idx, lp, tail, abls,
                            lambda: root_store.retrieval_block(group))
                        values = objective_values(
                            kl, log_z, abls, tail, mask, batch_weights, group,
                            components, histogram, joint, ref=root_ref(group),
                            union_terms=uterms)
                    elif sketches is not None:
                        idx, lp, tail = refs
                        values = sketch_forward_kl_mean_batch(
                            idx, lp, tail, abls, mask,
                            has_tail=sketches.has_tail, weights=batch_weights)
                    else:
                        values = forward_kl_mean_batch(
                            refs, abls, mask, weights=batch_weights, chunk=512)
                    #: THE PARENT-KL DIAGNOSTIC, from the SAME candidate logits,
                    #: after the decision value exists and never feeding it.
                    if parent_kl is not None:
                        tp = time.perf_counter()
                        pidx, plp, ptail = prefs
                        for item, v in zip(group.items,
                                           sketch_forward_kl_mean_batch(
                                               pidx, plp, ptail, abls, mask,
                                               has_tail=parent_sketches.has_tail,
                                               weights=batch_weights
                                               ).tolist()):
                            parent_kl.setdefault(item["subtype"], []).append(v)
                        timing["parent_kl_seconds"] += time.perf_counter() - tp
                    #: ONE host transfer for the whole batch.
                    for item, value in zip(group.items, values.tolist()):
                        per_subtype.setdefault(item["subtype"], []).append(value)
                        timing["distortion_calls"] += 1
                    if ctx.score_observer is not None:
                        #: The SAME `refs` and `abls` this round already paid for.
                        #: Execution only -- the return value is ignored and this
                        #: cannot reach a score, a config or an identity.
                        ctx.score_observer(
                            skip=frozenset(skip), group=group, indices=indices,
                            refs=refs, abls=abls, mask=mask,
                            weights=batch_weights, values=values,
                            support=support)
                elif sketches is not None and objective is not None:
                    dev = abls.device
                    kl, log_z, uterms = reference_kl(
                        refs.support_indices.to(dev),
                        refs.support_log_probs.to(dev),
                        refs.tail_log_prob.to(dev), abls,
                        lambda: root_store.retrieval_item(group.items[0]))
                    item_w = (None if active is None else
                              active.prediction_weights_for_item(indices[0]))
                    value = objective_values(
                        kl[None], log_z[None], abls[None],
                        refs.tail_log_prob.to(dev)[None],
                        torch.ones(1, int(kl.shape[0]), dtype=torch.bool,
                                   device=dev),
                        None if item_w is None else item_w.to(dev)[None],
                        group, components, histogram, joint,
                        ref=root_ref(group, item=group.items[0]),
                        union_terms=(None if uterms is None else
                                     {k: v[None] for k, v in uterms.items()}))
                    if parent_kl is not None:
                        tp = time.perf_counter()
                        parent_kl.setdefault(group.items[0]["subtype"],
                                             []).append(sketch_forward_kl_mean(
                            prefs, abls, weights=item_w))
                        timing["parent_kl_seconds"] += time.perf_counter() - tp
                    per_subtype.setdefault(group.items[0]["subtype"], []).append(
                        float(value[0]))
                    timing["distortion_calls"] += 1
                elif sketches is not None:
                    item_w = (None if active is None else
                              active.prediction_weights_for_item(indices[0]))
                    per_subtype.setdefault(group.items[0]["subtype"], []).append(
                        sketch_forward_kl_mean(refs, abls, weights=item_w))
                    timing["distortion_calls"] += 1
                    if parent_kl is not None:
                        tp = time.perf_counter()
                        parent_kl.setdefault(group.items[0]["subtype"],
                                             []).append(sketch_forward_kl_mean(
                            prefs, abls, weights=item_w))
                        timing["parent_kl_seconds"] += time.perf_counter() - tp
                else:
                    per_subtype.setdefault(group.items[0]["subtype"], []).append(
                        forward_kl_mean(
                            refs, abls,
                            weights=(None if active is None else
                                     active.prediction_weights_for_item(
                                         indices[0])),
                            chunk=512))
                    timing["distortion_calls"] += 1
                timing["distortion_seconds"] += time.perf_counter() - t2
                del abls
            timing["item_seconds"] += time.perf_counter() - item_started
            means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
            primary, _ = domain_balanced_score(means, domains)
            if declared_root:
                #: The decision KL and, beside it, the parent-reference KL of
                #: the same candidate under the same aggregation.
                diagnostic, _ = domain_balanced_score(
                    {sub: sum(v) / len(v) for sub, v in parent_kl.items()},
                    domains)
                reference_candidates[_subset_key(skip)] = {
                    "round": progress["rounds"],
                    "root_reference_kl": float(primary),
                    "parent_reference_kl": float(diagnostic)}
            if objective is not None:
                record = {}
                for name in components_of:
                    by_subtype = {sub: sum(c[name]) / len(c[name])
                                  for sub, c in components.items()}
                    record[name], _ = domain_balanced_score(by_subtype, domains)
                record["confidence_histogram"] = histogram
                record["joint_histograms"] = joint[0]
                if parent_kl is not None:
                    #: Same aggregation as the KL the decision reads.
                    record["parent_kl"], _ = domain_balanced_score(
                        {sub: sum(v) / len(v) for sub, v in parent_kl.items()},
                        domains)
                objective_candidates[_subset_key(skip)] = record
            return primary

        #: PREFIX REUSE (`ExecutionConfig.depth_prefix_reuse`), on the forward
        #: baseline's path only: scored rows, the KL-only objective, the parent
        #: reference, no observer. Elsewhere `score` runs unchanged.
        n_layers = ctx.parent_spec[DEPTH_FIELD]
        prefix_reuse = bool(ctx.execution.depth_prefix_reuse and select_rows
                            and objective is None and parent_sketches is None
                            and ctx.score_observer is None)

        def score_round(skipped: frozenset[int], candidates: list[int]) -> list[float]:
            """Every candidate of one round, sharing the current model's prefix.

            The arithmetic per (candidate, micro-batch) is `score`'s scored-rows
            branch line for line, and each candidate's values are appended in
            the same micro-batch and item order, so its subtype means and its
            domain balance are formed exactly as `score(skipped | {c})` forms
            them. Only the forward differs: block `c`'s input is taken from one
            capture forward of the current model instead of being recomputed.
            """
            per_candidate: dict[int, dict[str, list[float]]] = {c: {} for c in candidates}
            timing["candidate_subsets"] += len(candidates)
            item_started = time.perf_counter()
            for group, indices in groups:
                t0 = time.perf_counter()
                batch_weights = (None if active is None else
                                 active.prediction_weights_for(group, indices))
                mask = group.prediction_mask().to(compute)
                rows = scored_rows(mask, batch_weights)
                idx, lp, tail = sketches.sketch_rows(group, rows, adapter)
                ids = group.input_ids.to(compute)
                attn = group.attention_mask.to(compute)
                with captured_block_inputs(model, skipped) as inputs:
                    adapter.final_hidden_states(model, ids, attn)
                if cuda_sync:
                    cuda_sync()
                t1 = time.perf_counter()
                timing["reference_seconds"] += t1 - t0
                timing["prefix_capture_forwards"] += 1
                for c in candidates:
                    ta = time.perf_counter()
                    if has_suffix(n_layers, skipped, c):
                        with resumed_after_block(model, skipped, c, inputs[c]):
                            abls = scored_prediction_logits(adapter, model, ids,
                                                            attn, rows)
                    else:
                        with bypassed_blocks(model, skipped | {c}):
                            abls = scored_prediction_logits(adapter, model, ids,
                                                            attn, rows)
                    if cuda_sync:
                        cuda_sync()
                    t2 = time.perf_counter()
                    timing["ablated_seconds"] += t2 - ta
                    timing["ablated_forwards"] += 1
                    timing["ablated_items"] += len(group.items)
                    kl = sketch_forward_kl(idx, lp, tail, abls,
                                           has_tail=sketches.has_tail)
                    values = masked_item_means(scatter_rows(kl, rows), mask,
                                               weights=batch_weights)
                    for item, value in zip(group.items, values.tolist()):
                        per_candidate[c].setdefault(item["subtype"], []).append(value)
                        timing["distortion_calls"] += 1
                    timing["scored_rows"] += int(rows.sum())
                    timing["forward_rows"] += int(mask.numel())
                    timing["distortion_seconds"] += time.perf_counter() - t2
                    del abls, kl
                del inputs
                if ctx.deadline is not None:
                    ctx.deadline.check(
                        f"depth.causal_kl_greedy_v1 round {len(skipped)} "
                        f"(prefix reuse, {len(candidates)} candidates)")
            timing["item_seconds"] += time.perf_counter() - item_started
            scores = []
            for c in candidates:
                means = {k: sum(v) / len(v) for k, v in per_candidate[c].items()}
                primary, _ = domain_balanced_score(means, domains)
                scores.append(primary)
            return scores

        if prefix_reuse:
            timing["depth_prefix_reuse"] = True
            timing["prefix_capture_forwards"] = 0

        n_remove = ctx.parent_spec[DEPTH_FIELD] - ctx.target_spec[DEPTH_FIELD]

        # Bounded progress, and the deadline checked where the cost is.
        #
        # A round is 29-36 model evaluations and a record is written only when
        # one commits, so attempt 10 was silent for 10 h 47 m and nobody could
        # tell a working search from a stalled one. This prints at most one line
        # per candidate — 260 lines for the whole expansion — and it is the same
        # callback that enforces the wall clock, because the two questions
        # ("where is it?" and "has it run too long?") are asked at exactly the
        # same instant.
        started = time.monotonic()
        progress = {"evaluations": 0, "rounds": 0}

        def on_candidate(p: dict) -> None:
            progress["evaluations"] = p["evaluations"]
            mins = (time.monotonic() - started) / 60.0
            rate = p["evaluations"] / mins if mins > 0 else 0.0
            print(f"depth.causal_kl_greedy_v1: round {p['round']} "
                  f"candidate {p['index']}/{p['of']} (layer {p['candidate']}) "
                  f"score {p['score']:.6f} · {p['evaluations']} evals · "
                  f"{mins:.1f} min · {rate:.2f} eval/min", flush=True)
            if ctx.deadline is not None:
                ctx.deadline.check(
                    f"depth.causal_kl_greedy_v1 round {p['round']} "
                    f"candidate {p['index']}/{p['of']} "
                    f"({p['evaluations']} evaluations done)")

        def on_round(r: dict) -> None:
            progress["rounds"] = r["round"] + 1
            ranked = sorted(r["table"], key=lambda x: (x["score"], x["candidate"]))
            margin = (ranked[1]["score"] - ranked[0]["score"]
                      if len(ranked) > 1 else float("nan"))
            print(f"depth.causal_kl_greedy_v1: ROUND {r['round']} chose layer "
                  f"{r['chosen']} score {r['chosen_score']:.6f}; runner-up "
                  f"{ranked[1]['candidate'] if len(ranked) > 1 else None} "
                  f"margin {margin:.3e}", flush=True)

        result = greedy_removal(score, ctx.parent_spec[DEPTH_FIELD], n_remove,
                                on_round=on_round, on_candidate=on_candidate,
                                score_round=score_round if prefix_reuse else None)
        kept = result["kept"]
        child = _build_child_with_layers(ctx, kept)

        final_score = result["rounds"][-1]["chosen_score"] if result["rounds"] else 0.0
        values = {"op.depth.causal_kl.final": float(final_score),
                  "op.depth.causal_kl.evaluations": float(result["evaluations"])}
        if declared_root and result["rounds"]:
            chosen = reference_candidates[_subset_key(result["removed"])]
            values["op.depth.parent_kl_diagnostic.final"] = float(
                chosen["parent_reference_kl"])
        if objective is not None and result["rounds"]:
            #: Under an objective the chosen score IS the objective, so the KL
            #: key carries the chosen subset's raw KL and the objective has its
            #: own key -- the KL never silently means something else.
            final = objective_candidates[_subset_key(result["removed"])]
            values["op.depth.causal_kl.final"] = float(final["raw_kl"])
            values["op.depth.objective.final"] = float(final["objective"])
        return OperatorOutcome(
            model=child,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference=("root_reference" if root_store is not None
                           else "parent_state"),
                values=values,
                detail={"removal_order": result["removal_order"],
                        **({} if objective is None else
                           {"scoring_objective": objective.qualified_id}),
                        **({KL_REFERENCE_KEY: kl_reference}
                           if declared_root else {})}),
            trace={"micro_batch_size": batch_size,
                   #: Execution evidence, recorded even at the default so a run's
                   #: trace states which grouping its forwards used.
                   "calibration_batch_packing": packing,
                   #: Recorded only when it is not the historical path, so a
                   #: trace of an all-rows run is unchanged field for field.
                   **({"prediction_rows": ctx.execution.prediction_rows}
                      if select_rows else {}),
                   #: Execution evidence too, and only when it ran.
                   **({"depth_prefix_reuse": True} if prefix_reuse else {}),
                   #: SCIENCE evidence: what the policy named in the hashed config
                   #: actually restricted.
                   "scoring_positions": (None if active is None
                                         else active.report()),
                   **({} if objective is None else
                      {"scoring_objective": objective.as_dict()}),
                   #: Recorded only when declared, so a parent-reference
                   #: trace is unchanged field for field.
                   **({KL_REFERENCE_KEY: kl_reference} if declared_root
                      else {}),
                   "kept_layers": kept, "removed_layers": result["removed"],
                   "removal_order": result["removal_order"],
                   "source": "causal_kl_greedy"},
            # The memory decision is an artifact, NOT part of the trace. The
            # trace describes the deterministic result — `test_operator_is_
            # deterministic` compares two invocations of it field for field —
            # and which path ran depends on how much memory the host happened to
            # have free. The numbers are identical either way, so the decision
            # belongs with the byproducts.
            artifacts={"search_rounds": result["rounds"],
                       **({} if objective is None else
                          {"objective_candidates": objective_candidates}),
                       **({"reference_candidates": reference_candidates}
                          if declared_root else {}),
                       "reference_cache": (reference or sketches).decision(),
                       **({} if parent_sketches is None else
                          {"parent_reference_cache": parent_sketches.decision(),
                           "parent_kl_diagnostic": (
                               "every round's candidates"
                               if parent_kl_every_round
                               else "round 0 candidates only")}),
                       # Operational only. `artifacts` is already excluded from
                       # the trace and from state identity — see the note above
                       # — which is exactly why the timings belong here and in
                       # no other field the operator returns.
                       "timing": {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in timing.items()},
                       "reference_counters": (reference or sketches).counters()},
        )



def _subset_key(skip) -> str:
    """A candidate subset as a record key: its layers, ascending."""
    return ",".join(str(i) for i in sorted(int(x) for x in skip))


def _domain_map(items) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in items:
        subs = out.setdefault(item["domain"], [])
        if item["subtype"] not in subs:
            subs.append(item["subtype"])
    return {k: sorted(v) for k, v in sorted(out.items())}


@torch.no_grad()
def _forward_logits_batch(model, batch, device: str, skip=frozenset()):
    """One micro-batch's prediction-position logits, one tensor per ORIGINAL item.

    The batched twin of `_forward_logits`, and it returns a *list* rather than a
    padded block on purpose: every consumer of this operator measures per item,
    so the padding is undone here, once, and nothing downstream has to know a
    batch existed. Element ``i`` is exactly what ``_forward_logits`` would have
    returned for that item — ``[L_i - 1, V]``, same positions, same dtype, same
    device — so the per-item `forward_kl_mean` reduction that follows is the
    same call over the same shape with the same chunk boundaries.

    The `attention_mask` is passed, and it is honest to say that under **right**
    padding it is defensive rather than load-bearing: causality already stops a
    real token at position ``i < L`` from reaching a pad at ``>= L``, and
    dropping the mask here was measured to change the real-position logits by
    exactly ``0.0``. What is load-bearing is the right padding itself — it keeps
    every real token at the position index it occupies alone, so the RoPE phase
    is unchanged. The mask is kept because it costs nothing, because it is
    required the instant anything pads on the other side, and because a reader
    should not have to re-derive the causality argument to trust this line.
    """
    ids = batch.input_ids.to(device)
    mask = batch.attention_mask.to(device)
    if not skip:
        return batch.split_predictions(model(ids, attention_mask=mask).logits[:, :-1])
    with bypassed_blocks(model, skip):
        return batch.split_predictions(model(ids, attention_mask=mask).logits[:, :-1])


@torch.no_grad()
def _forward_logit_block(model, batch, device: str, skip=frozenset()):
    """The same forward, left as ONE ``[B, T_pred, V]`` block.

    The hot path does not split. `forward_kl_mean_batch` reduces the block
    against its prediction mask and returns one KL per row, so splitting into B
    tensors and calling the scalar reducer B times would add a Python loop, B
    host synchronisations and B small kernel launches for no scientific
    difference. `_forward_logits_batch` above still returns the split form,
    because the reference CACHE stores per-item rows.
    """
    ids = batch.input_ids.to(device)
    mask = batch.attention_mask.to(device)
    if not skip:
        return model(ids, attention_mask=mask).logits[:, :-1]
    with bypassed_blocks(model, skip):
        return model(ids, attention_mask=mask).logits[:, :-1]


@torch.no_grad()
def _forward_logits(model, item, device: str, skip=frozenset()):
    """One item's prediction-position logits, optionally with blocks bypassed.

    Left in the model's own dtype **and on the model's device**. ``distortion``
    upcasts to float32 in chunks internally, so widening here would only double
    the bytes held per item without changing a single reduced value.

    **This returned ``.cpu()`` once, and it cost a full paid search.** The
    implementation this operator ports keeps prepared inputs, reference logits,
    ablated logits and the
    ``distortion`` reduction on the selected accelerator. The port introduced the
    transfer, and with it a full 151,936-vocabulary softmax/KL on the host, 260
    evaluations x 67 items per expansion: ~86 TiB of CPU traffic and ~8.6 TiB
    copied off the device. A paid run spent over ten hours inside a single
    expansion at 0-1 % GPU utilisation, and was stopped without finishing it.

    Nothing about the reduction changed to fix this. The tensors simply stay
    where the ported implementation left them.
    """
    ids = item["input_ids"].to(device)
    if not skip:
        return model(ids).logits[0, :-1]
    with bypassed_blocks(model, skip):
        return model(ids).logits[0, :-1]


class _RootSketches:
    """The ROOT reference store behind the parent cache's interface.

    Every view -- per item, per padded block, per selected row -- reads the
    values the store computed once from the root; nothing is forwarded here.
    """

    def __init__(self, store) -> None:
        self.store = store
        self.has_tail = store.has_tail

    def restrict_to_scored(self, active) -> None:
        """Nothing to restrict: the store holds every row and the selection
        happens per batch, from the same rows the candidate scores."""

    def sketch_block(self, batch):
        return self.store.sketch_block(batch)

    def sketch_item(self, item):
        return self.store.sketch_item(item)

    def sketch_rows(self, batch, rows, adapter):
        return self.store.sketch_rows(batch, rows)

    def decision(self) -> dict[str, Any]:
        return self.store.decision()

    def counters(self) -> dict[str, int]:
        return self.store.counters()


class _ReferenceSketches:
    """The intact parent's reference as ``O(T*K)`` sketches, not ``O(T*V)`` logits.

    This is the whole point of ``reference_topk_tail_v1`` for DEPTH. The
    full-logit sibling below sizes its cache at ``positions * V * itemsize`` --
    **33.8 GiB** for the frozen ``calib.domain_balanced@v1`` mixture at a 151,936
    vocabulary -- and when that does not fit it falls back to recomputing the
    reference for every one of 260 candidates, roughly doubling the forwards for
    the whole expansion. A sketch holds ``positions * K`` log probabilities and
    indices plus three per-position vectors, so at a few hundred K it is megabytes
    and the fallback never triggers.

    **The forward's composition still does not depend on what is cached.** The
    sketches are built from the SAME canonical padded batches the candidate
    forwards use, so the reference is computed under the same kernel shapes as
    the thing it is compared against. Building them from a sub-batch of the
    missing rows would put the boundary between cached and recomputed inside a
    scientific measurement, which is the mistake the full-logit sibling's
    docstring records having already been made once.
    """

    def __init__(self, model, items, device: str, *, top_k: int) -> None:
        self.model = model
        self.device = device
        self.top_k = int(top_k)
        self.vocab = int(model.config.vocab_size)
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        items = list(items)
        self._items = items
        self.scored_rows_only = False
        positions = sum(int(i["input_ids"].shape[1]) - 1 for i in items)
        self.positions = positions
        k = max(1, min(self.top_k, self.vocab))
        self.effective_top_k = k
        #: float32 log probabilities + int64 indices + three float32 per-position
        #: vectors. Stated rather than guessed: the full-logit cache's budget
        #: being wrong once cost a 33.8 GiB overrun and an OOM kill.
        self.estimate_bytes = positions * (k * 4 + k * 8) + positions * 4 * 3
        self.full_vocab_estimate_bytes = (
            positions * self.vocab
            * next(model.parameters()).dtype.itemsize)
        self.memory_before = memory_snapshot(device)
        self.available_bytes, self.headroom_source = _available_memory_bytes(device)
        #: AFTER the probe, which returns the allocator's hoard first: the pair
        #: of snapshots is the evidence that the release happened and what it
        #: freed. Diagnostic only, like `memory_before`.
        self.memory_after_probe = memory_snapshot(device)
        budget = (None if self.available_bytes is None
                  else _ReferenceLogits.BUDGET_FRACTION * self.available_bytes)
        self.enabled = budget is None or self.estimate_bytes <= budget
        self.mode = "cached" if self.enabled else "recomputed"
        self.hits = 0
        self.recomputes = 0
        self.fills = 0
        self.has_tail = k < self.vocab

    def sketch_block(self, batch):
        """``([B, T, K] indices, [B, T, K] log probs, [B, T] tail)`` for a batch."""
        rows = list(batch.items)
        cached = [self._cache.get(item["item_id"]) for item in rows]
        if all(c is not None for c in cached):
            self.hits += len(rows)
            return self._assemble(batch, cached)

        block = _forward_logit_block(self.model, batch, self.device)
        width = int(batch.input_ids.shape[1]) - 1
        for index, item in enumerate(rows):
            if cached[index] is not None:
                self.hits += 1
                continue
            length = int(batch.lengths[index])
            #: The item's VALID PREDICTION SLICE only, reduced immediately. The
            #: `[B, T, V]` block dies at the end of this call -- that is the
            #: memory this protocol exists to stop holding.
            sk = sketch_reference(
                block[index, :length - 1],
                torch.zeros(length - 1, dtype=torch.long, device=block.device),
                top_k=self.top_k, chunk=512)
            cached[index] = (sk.support_indices, sk.support_log_probs,
                             sk.tail_log_prob)
            if self.enabled:
                self._cache[item["item_id"]] = cached[index]
                self.fills += 1
            else:
                self.recomputes += 1
        del block
        return self._assemble(batch, cached)

    def restrict_to_scored(self, active) -> None:
        """Hold sketches for the SCORED rows only, and re-size the budget so.

        Under pre-vocabulary row selection a position the policy gives weight
        zero is never projected, so its sketch would never be read. The cache
        decision is re-taken against the smaller footprint; the values do not
        depend on it either way.
        """
        scored = 0
        for index, item in enumerate(self._items):
            weights = (None if active is None
                       else active.prediction_weights_for_item(index))
            scored += (int(item["input_ids"].shape[1]) - 1 if weights is None
                       else int((weights > 0).sum()))
        k = self.effective_top_k
        self.scored_rows_only = True
        self.positions = scored
        self.estimate_bytes = scored * (k * 4 + k * 8) + scored * 4 * 3
        budget = (None if self.available_bytes is None
                  else _ReferenceLogits.BUDGET_FRACTION * self.available_bytes)
        self.enabled = budget is None or self.estimate_bytes <= budget
        self.mode = "cached" if self.enabled else "recomputed"

    def sketch_rows(self, batch, rows, adapter):
        """``([N, K] indices, [N, K] log probs, [N] tail)`` for the scored rows.

        Row-major over the batch -- the order `scored_prediction_logits` returns
        the candidate's rows in -- so position ``n`` of the reference and of the
        candidate is the same prediction of the same item. Filled from the SAME
        canonical batch the candidate forwards use, for the reason the class
        docstring gives.
        """
        items = list(batch.items)
        cached = [self._cache.get(item["item_id"]) for item in items]
        if not all(c is not None for c in cached):
            logits = scored_prediction_logits(
                adapter, self.model, batch.input_ids.to(self.device),
                batch.attention_mask.to(self.device), rows)
            n = int(logits.shape[0])
            sk = sketch_reference(
                logits, torch.zeros(n, dtype=torch.long, device=logits.device),
                top_k=self.top_k, chunk=512)
            del logits
            parts = list(zip(split_rows(sk.support_indices, rows),
                             split_rows(sk.support_log_probs, rows),
                             split_rows(sk.tail_log_prob, rows)))
            for index, item in enumerate(items):
                if cached[index] is not None:
                    self.hits += 1
                    continue
                cached[index] = tuple(x.clone() for x in parts[index])
                if self.enabled:
                    self._cache[item["item_id"]] = cached[index]
                    self.fills += 1
                else:
                    self.recomputes += 1
        else:
            self.hits += len(items)
        return (torch.cat([c[0] for c in cached]),
                torch.cat([c[1] for c in cached]),
                torch.cat([c[2] for c in cached]))

    def _assemble(self, batch, rows):
        """Cached per-item sketches back into the canonical padded shapes.

        Padding is left at zero and never read: the caller masks it out, exactly
        as it does for a padded logit block.
        """
        width = int(batch.input_ids.shape[1]) - 1
        k = rows[0][0].shape[-1]
        dev = rows[0][0].device
        idx = torch.zeros(len(rows), width, k, dtype=torch.long, device=dev)
        lp = torch.zeros(len(rows), width, k, dtype=torch.float32, device=dev)
        tail = torch.zeros(len(rows), width, dtype=torch.float32, device=dev)
        for i, (a, b, c) in enumerate(rows):
            idx[i, :a.shape[0]] = a
            lp[i, :b.shape[0]] = b
            tail[i, :c.shape[0]] = c
        return idx, lp, tail

    def sketch_item(self, item) -> ReferenceDistributionSketch:
        """One item's sketch, for the ``batch_size == 1`` path."""
        item_id = item["item_id"]
        hit = self._cache.get(item_id)
        if hit is None:
            logits = _forward_logits(self.model, item, self.device)
            n = int(logits.shape[0])
            sk = sketch_reference(
                logits, torch.zeros(n, dtype=torch.long, device=logits.device),
                top_k=self.top_k, chunk=512)
            hit = (sk.support_indices, sk.support_log_probs, sk.tail_log_prob)
            if self.enabled:
                self._cache[item_id] = hit
                self.fills += 1
            else:
                self.recomputes += 1
            del logits
        else:
            self.hits += 1
        return ReferenceDistributionSketch(
            support_indices=hit[0], support_log_probs=hit[1],
            tail_log_prob=hit[2],
            #: Neither is read by the forward-KL reducer; the sketch type carries
            #: them for the six-quantity path and they are not recomputed here.
            top1_token=torch.zeros(hit[0].shape[0], dtype=torch.long,
                                   device=hit[0].device),
            target_log_prob=torch.zeros(hit[0].shape[0], dtype=torch.float32,
                                        device=hit[0].device),
            top_k=self.effective_top_k, vocab_size=self.vocab)

    def counters(self) -> dict[str, int]:
        """Telemetry only, and the same three names the full-logit cache uses so
        a reader comparing two expansions does not have to know which ran."""
        return {"reference_hits": self.hits, "reference_fills": self.fills,
                "reference_recomputes": self.recomputes}

    def decision(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cached": self.enabled,
            "distribution_support": SUPPORT_REFERENCE_TOPK_TAIL_V1,
            "top_k": self.effective_top_k,
            **({"scored_rows_only": True} if self.scored_rows_only else {}),
            "has_tail": self.has_tail,
            "items_total": len(self._cache) or None,
            "positions": self.positions,
            "sketch_bytes": self.estimate_bytes,
            "full_vocab_bytes_avoided": self.full_vocab_estimate_bytes,
            "bytes_ratio_vs_full_vocab": round(
                self.estimate_bytes / max(self.full_vocab_estimate_bytes, 1), 8),
            "available_bytes": self.available_bytes,
            "headroom_source": self.headroom_source,
            "memory_before": self.memory_before,
            "memory_after_probe": self.memory_after_probe,
        }


class _ReferenceLogits:
    """The unbypassed parent's logits, cached only when the memory fits.

    The reference is identical for every candidate in the whole greedy search,
    so caching it turns ``2 * evals * n_items`` forward passes into
    ``(evals + 1) * n_items`` — the cost structure ``plan()`` reports.

    It had been cached **unconditionally**, in float32, in host RAM. For the
    frozen ``calib.domain_balanced@v1`` mixture that is 59,763 prediction
    positions x 151,936 vocabulary x 4 B = **33.8 GiB**, held for the duration of
    the search, on every invocation of this operator. Nothing had ever executed
    it against the real mixture: the one paid run that would have reached it
    died earlier, and every zero-cost run used a toy mixture. Its first real
    execution — this rehearsal — was killed by the OOM killer.

    The script whose algorithm this operator re-runs already had the answer,
    and it was dropped in the port: size
    the cache, and if it does not fit, recompute. Recomputing is numerically
    identical — same deterministic no-grad forward, same tensor, and
    ``distortion`` upcasts either way — so the fallback is automatic and loud
    rather than a flag a caller has to remember.
    """

    #: E8a's fraction. The rest holds the ablated logits, the float32 reduction
    #: chunks, and the activations of the next forward pass. **Unchanged**: the
    #: partial mode below spends less than this allowance, never more.
    BUDGET_FRACTION = 0.66

    def __init__(self, model, items, device: str) -> None:
        self.model = model
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}
        items = list(items)
        itemsize = next(model.parameters()).dtype.itemsize
        vocab = int(model.config.vocab_size)
        #: Per item, in the order the mixture draws them. The order is load-bearing
        #: for the admission decision below and must not be sorted or reordered.
        self._item_bytes = {i["item_id"]: (int(i["input_ids"].shape[1]) - 1) * vocab * itemsize
                            for i in items}
        positions = sum(int(i["input_ids"].shape[1]) - 1 for i in items)
        self.estimate_bytes = positions * vocab * itemsize
        #: BEFORE the availability probe, so the snapshot describes the state
        #: the probe is about to read rather than the state after it.
        self.memory_before = memory_snapshot(device)
        self.available_bytes, self.headroom_source = _available_memory_bytes(device)

        budget = (None if self.available_bytes is None
                  else self.BUDGET_FRACTION * self.available_bytes)
        self.enabled = budget is None or self.estimate_bytes <= budget

        # PARTIAL CACHING. This was once all-or-nothing, and a paid run landed
        # exactly in the gap: a 16.9 GiB reference against a
        # 66%-of-20.3 GiB = 13.4 GiB allowance. 79% of it fit and none of it was
        # kept, so every one of 260 candidates recomputed the ENTIRE reference —
        # ~2x the forward passes for the whole expansion, twelve times over.
        #
        # Admission is by the mixture's own item order and nothing else: no
        # sorting by size, no packing heuristic. A size-greedy admission would
        # make the resident set depend on how much memory the host happened to
        # have free, so two runs on different hosts would cache different items.
        # They would still compute identical numbers — recompute and hit return
        # the same tensor — but the *telemetry* would stop being comparable, and
        # a rule that reads "the first k items that fit" is one a reader can
        # verify against the frozen mixture.
        self.admitted: set[str] = set()
        self.admitted_bytes = 0
        if self.enabled:
            self.admitted = set(self._item_bytes)
            self.admitted_bytes = self.estimate_bytes
        elif budget is not None:
            for item in items:                     # mixture order, deliberately
                size = self._item_bytes[item["item_id"]]
                if self.admitted_bytes + size > budget:
                    break
                self.admitted.add(item["item_id"])
                self.admitted_bytes += size

        self.mode = ("cached" if self.enabled
                     else "partial" if self.admitted else "recomputed")
        if not self.enabled:
            # Loud, like E8a's. A run that quietly doubled its forward passes is
            # a run whose wall-clock estimate no longer holds.
            share = (100.0 * len(self.admitted) / max(len(self._item_bytes), 1))
            print(f"depth.causal_kl_greedy_v1: reference cache "
                  f"{self.estimate_bytes / 2**30:.1f} GiB does not fit in "
                  f"{self.BUDGET_FRACTION:.0%} of {self.available_bytes / 2**30:.1f} "
                  f"GiB ({self.headroom_source}) -> {self.mode}: caching "
                  f"{len(self.admitted)}/{len(self._item_bytes)} items "
                  f"({share:.0f}%, {self.admitted_bytes / 2**30:.1f} GiB), "
                  "recomputing the rest. Identical numbers either way.",
                  flush=True)

        #: Telemetry only. Counted, never returned into a metric or a hash.
        self.hits = 0
        self.recomputes = 0
        self.fills = 0

    def get(self, item) -> torch.Tensor:
        item_id = item["item_id"]
        hit = self._cache.get(item_id)
        if hit is not None:
            self.hits += 1
            return hit
        computed = _forward_logits(self.model, item, self.device)
        if item_id in self.admitted:
            self._cache[item_id] = computed
            self.fills += 1
        else:
            self.recomputes += 1
        return computed

    def get_batch(self, batch) -> list[torch.Tensor]:
        """References for a whole micro-batch, in row order, as per-item rows.

        Kept for callers that want the split form. The KL hot path uses
        :meth:`reference_block`, which never splits.
        """
        block = self.reference_block(batch)
        return batch.split_predictions(block)

    def reference_block(self, batch) -> torch.Tensor:
        """The intact parent's ``[B, T_pred, V]`` logits for THIS canonical batch.

        **The forward's composition does not depend on what is cached.** Three
        states, and only the first two differ in how much work they do:

        * *fully cached* — no forward at all. That is what the cache is for, and
          it is the one case where skipping is free of consequence: the rows are
          reassembled into the canonical padded shape, and the padding they are
          written around is excluded by the caller's prediction mask, so it can
          hold anything.
        * *fully uncached* — one intact forward over the whole batch.
        * *partially cached* — one intact forward over the **whole batch**
          anyway, not over a sub-batch of the missing rows.

        That last case used to build a smaller `ItemBatch` of just the missing
        rows. It produced the same per-item logits to within float tolerance,
        but through a different batch width and therefore a different kernel
        shape — which made the numeric path a function of how much memory the
        host happened to have free, and put the boundary between "cached" and
        "recomputed" *inside* a scientific measurement. A runtime cache hit must
        never move batch membership, batch shape or position alignment. The
        price is one redundant forward on the single boundary batch, which is
        the right trade for a result that does not depend on runtime state.
        """
        rows = list(batch.items)
        cached = [self._cache.get(item["item_id"]) for item in rows]
        if all(c is not None for c in cached):
            self.hits += len(rows)
            return self._assemble(batch, cached)

        block = _forward_logit_block(self.model, batch, self.device)
        for index, item in enumerate(rows):
            item_id = item["item_id"]
            if cached[index] is not None:
                self.hits += 1
                continue
            if item_id in self.admitted:
                #: CLONE the item's VALID PREDICTION SLICE only. A row of this
                #: block is a view into `[B, T_max, V]`, so caching it unchanged
                #: would pin the whole padded block — every other item's logits
                #: and all the padding — for the lifetime of the search, while
                #: `_item_bytes` went on sizing this cache at
                #: `(T_i - 1) * V * itemsize`. That is the budget whose being
                #: wrong once already cost a 33.8 GiB overrun.
                length = batch.lengths[index]
                self._cache[item_id] = block[index, :length - 1].clone()
                self.fills += 1
            else:
                self.recomputes += 1
        return block

    def _assemble(self, batch, rows: list[torch.Tensor]) -> torch.Tensor:
        """Cached per-item rows back into the canonical padded block.

        Built at the batch's own width so the shape a fully cached batch hands
        the reducer is the shape a forward would have produced. Padding is left
        at zero and is never read: the caller masks it out.
        """
        width = batch.input_ids.shape[1] - 1
        block = torch.zeros(len(rows), width, rows[0].shape[-1],
                            dtype=rows[0].dtype, device=rows[0].device)
        for index, row in enumerate(rows):
            block[index, :row.shape[0]] = row
        return block

    def decision(self) -> dict[str, Any]:
        n_items = len(self._item_bytes)
        return {
            "mode": self.mode,
            # Kept as the original boolean so existing readers of this artifact
            # keep their meaning: True iff the WHOLE reference is resident.
            "cached": self.enabled,
            "items_total": n_items,
            "items_cached": len(self.admitted),
            "items_recomputed_per_candidate": n_items - len(self.admitted),
            "estimate_bytes": int(self.estimate_bytes),
            "estimate_gib": round(self.estimate_bytes / 2**30, 3),
            "admitted_bytes": int(self.admitted_bytes),
            "admitted_gib": round(self.admitted_bytes / 2**30, 3),
            "available_bytes": self.available_bytes,
            "budget_fraction": self.BUDGET_FRACTION,
            #: DIAGNOSTIC. Why the availability above was what it was -- the
            #: distinction between a card held by live tensors and one held by
            #: an allocator hoarding freed blocks. See `memory_snapshot`.
            "memory_at_admission": self.memory_before,
            "headroom_source": self.headroom_source,
            "fallback": None if self.enabled else (
                f"{self.mode}: {len(self.admitted)}/{n_items} items resident, the "
                "remainder recomputed per candidate; identical numbers"),
        }

    def counters(self) -> dict[str, int]:
        """Telemetry only — hits, fills and recomputes over the whole expansion."""
        return {"reference_hits": self.hits, "reference_fills": self.fills,
                "reference_recomputes": self.recomputes}


def memory_snapshot(device: Any) -> dict[str, Any]:
    """What is actually occupying the device, at one instant.

    DIAGNOSTIC ONLY: never returned into a metric, a score or a hash.

    It exists because the reference cache's admission decision varied wildly
    across real runs -- 67/67 items cached in one expansion, 11/67 in six
    consecutive ones, with `mem_get_info` reporting 26.3 GiB free in the first
    case and 2.6 GiB in the last, on the same 48 GB card. That cost 14,560
    reference recomputes in a single expansion, and pooled over the nineteen
    committed DEPTH expansions the recomputes are 36.1% of every forward pass
    the operator performed.

    Whether that is recoverable depends on a distinction `mem_get_info` alone
    cannot make:

    * `memory_allocated` is what the allocator has handed to live tensors;
    * `memory_reserved` is what it has taken from the driver, INCLUDING freed
      blocks it is holding for reuse;
    * `mem_get_info` free is what the driver has left, which counts reserved-
      but-unused blocks as unavailable.

    So if `reserved - allocated` is large, the cache is being sized against
    memory that PyTorch is merely hoarding and `empty_cache()` would return --
    a real, cheap fix. If `allocated` is itself large, live tensors hold the
    card and releasing the allocator's cache would achieve nothing. The two
    have opposite remedies, which is why this measures before anything decides.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda":
        return {"device": str(dev), "kind": dev.type,
                "_why_empty": "not an accelerator; there is no device allocator "
                              "to distinguish from the driver's view"}
    try:
        free, total = torch.cuda.mem_get_info(dev)
        allocated = int(torch.cuda.memory_allocated(dev))
        reserved = int(torch.cuda.memory_reserved(dev))
    except Exception as exc:                                       # noqa: BLE001
        return {"device": str(dev), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "device": str(dev),
        "kind": "cuda",
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
        "allocator_allocated_bytes": allocated,
        "allocator_reserved_bytes": reserved,
        #: THE NUMBER THE DIAGNOSIS TURNS ON. Reserved-but-unallocated is what
        #: `empty_cache()` could return to the driver, and it is invisible to
        #: `mem_get_info`.
        "allocator_reserved_not_allocated_bytes": max(reserved - allocated, 0),
        "driver_free_gib": round(free / 2**30, 3),
        "allocator_allocated_gib": round(allocated / 2**30, 3),
        "allocator_reserved_gib": round(reserved / 2**30, 3),
        "reclaimable_by_empty_cache_gib": round(
            max(reserved - allocated, 0) / 2**30, 3),
        #: What the driver cannot account for through this process's allocator:
        #: another context, another process, or a non-PyTorch allocation.
        "unaccounted_gib": round(
            max(total - free - reserved, 0) / 2**30, 3),
    }


def _available_memory_bytes(device: Any) -> tuple[int | None, str]:
    """Free memory **where the cache will actually live**.

    Since 2026-08-19 the cache is device-resident, as E8a's always was, so on an
    accelerator this asks the accelerator. The previous version measured the
    host — correctly for the code as it then stood, because ``_forward_logits``
    returned ``.cpu()``. Its own docstring said so:

        "the ported implementation kept its cache on the accelerator and
        therefore checked
        ``torch.cuda.mem_get_info``; copying that probe here would measure free
        VRAM against an allocation that never touches it."

    That reasoning was sound and the premise was the defect. The probe follows
    the cache rather than the other way round.

    The fallback when it does not fit is **recompute**, never a silent move to
    the host: a host-resident reference would drag the whole reduction back with
    it, which is the failure that cost a full paid search.

    **The allocator's hoard is returned first** (2026-10-13). `mem_get_info`
    counts blocks PyTorch has reserved but holds unused as taken, and inside a
    search the earlier operators and state evaluations leave tens of GiB of
    them. D1's formal search shows it: 15 of its 16 DEPTH expansions fell back
    to recompute with 0.12 GiB "free" and 36.0 GiB reclaimable by
    `empty_cache()` -- the remedy `memory_snapshot` names -- for a 0.14 GiB
    sketch, and the recomputed invocations took 1,336-2,486 s against the
    cached one's 1,179 s. `empty_cache()` changes no tensor and no kernel, and
    cached and recomputed references are the same numbers, so only the
    admission decision and the wall clock move.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":                       # pragma: no cover - needs a GPU
        try:
            torch.cuda.empty_cache()
            free, _total = torch.cuda.mem_get_info(dev)
            return int(free), f"cuda.mem_get_info:{dev} after empty_cache"
        except Exception:                                          # noqa: BLE001
            return None, "cuda.mem_get_info unavailable"
    return _host_available_memory_bytes()


def _host_available_memory_bytes() -> tuple[int | None, str]:
    """Free host memory, for a CPU-resident cache.

    The **cgroup** grant comes first: inside a container ``/proc/meminfo``
    reports the *host's* memory, and this project has already shipped one bug
    from trusting a host-wide number over the cgroup limit (``nproc`` versus the
    CPU quota). The limit that binds is the smaller of the two.
    """
    candidates: list[tuple[int, str]] = []
    try:                                            # cgroup v2
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            candidates.append((max(int(limit) - used, 0), "cgroup.v2"))
    except (OSError, ValueError):
        pass
    try:                                            # cgroup v1
        limit = int(Path(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip())
        used = int(Path(
            "/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip())
        # An "unlimited" v1 cgroup reports a sentinel near 2**63.
        if limit < 2**62:
            candidates.append((max(limit - used, 0), "cgroup.v1"))
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append((int(line.split()[1]) * 1024, "proc.meminfo"))
                break
    except (OSError, ValueError):
        pass
    if not candidates:
        return None, "unknown"
    return min(candidates)


#: The instance, NOT a registration. Registering at import made the registry's
#: contents depend on who had imported what first, which is the same coupling
#: the adapter bootstrap removed. `aadistill.initialization.operators.register`
#: is the one place the shipped operators are registered.


#: The instance, NOT a registration.
DEPTH_CAUSAL_KL_GREEDY_V1 = DepthCausalKLGreedyV1()
