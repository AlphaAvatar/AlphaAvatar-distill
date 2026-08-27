"""DEPTH implementations: which blocks survive.

Two algorithms, deliberately kept as separate immutable ids because E8a showed
they disagree almost completely — for 36 -> 28 they share exactly one removed
layer out of eight, and the causal search preserved the full-width teacher
distribution 3.11x better while initializing 2.8 nats worse once composed with
width/FFN/attention compression. Which is right is the open question the search
exists to answer, so both stay registered and neither is a default.

``depth.positional_v0`` wraps ``init.sandwich.depth_span_map``; the map it
produces at 36 -> 28 is the one behind the canonical Stage-1 checkpoint
``86fbba78...``.

``depth.causal_kl_greedy_v1`` wraps ``init.contribution`` — block bypass, forward
KL against the *unbypassed parent*, domain-balanced aggregation, iterative greedy
removal with a stated tie-break. It is re-run against whatever checkpoint it is
handed, which is what makes it conditional rather than a precomputed teacher
decision.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ...init.contribution import (
    bypassed_blocks,
    distortion,
    domain_balanced_score,
    expected_evaluations,
    greedy_removal,
)
from ..device import model_device
from ...init.sandwich import depth_span_map
from ..arch import ArchitectureAdapter, ArchSpec, Capability
from ..metrics import OperatorLocalMetrics
from ._common import ChildBuilder, copy_embeddings_and_final_norm, copy_module_except
from .base import (
    CalibrationNeed,
    OperatorContext,
    OperatorError,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
    register_implementation,
)

DEPTH_FIELD = "num_hidden_layers"


def _build_child_with_layers(ctx: OperatorContext, kept: list[int]) -> Any:
    """A child holding exactly the parent blocks ``kept``, in order, verbatim."""
    adapter = ctx.adapter
    new_spec = ctx.parent_spec.replace(**{DEPTH_FIELD: len(kept)})
    builder = ChildBuilder(adapter, ctx.model, new_spec, seed=ctx.seed)
    parent_blocks = adapter.blocks(ctx.model)
    child_blocks = adapter.blocks(builder.model)
    for dst, src_idx in zip(child_blocks, kept):
        copy_module_except(builder, parent_blocks[src_idx], dst)
    copy_embeddings_and_final_norm(builder, adapter, ctx.model)
    return builder.finish()


class DepthPositionalV0(OperatorImplementation):
    impl_id = "depth.positional_v0"
    kind = "DEPTH"
    version = 0
    description = (
        "Positional pairwise merge in a middle band: ~1/5 of the surviving 1:1 "
        "layers stay before the band, the rest after, so both the earliest and "
        "the latest blocks map 1:1. The incumbent map behind qwen3_0p6b_init_v0.")
    required_capabilities = frozenset({Capability.BLOCK_LIST})
    modifies = frozenset({DEPTH_FIELD})
    preserves = frozenset({"hidden_size", "intermediate_size", "num_attention_heads",
                           "num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.NONE
    objective = "none (fixed positional heuristic; no measurement is taken)"
    deterministic = True
    requires_seed = False
    produces = ("depth_map",)
    target_validation = "result num_hidden_layers equals the target exactly"

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        depth_span_map(spec[DEPTH_FIELD], target[DEPTH_FIELD])  # raises if infeasible
        return OperatorPlan(
            impl_id=self.impl_id,
            result_spec=spec.replace(**{DEPTH_FIELD: target[DEPTH_FIELD]}),
            forward_passes=0, stats_passes=0,
            notes="no calibration; the map is a function of the two layer counts")

    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        spans = depth_span_map(ctx.parent_spec[DEPTH_FIELD],
                               ctx.target_spec[DEPTH_FIELD])
        kept = [s["representative"] for s in spans]
        model = _build_child_with_layers(ctx, kept)
        removed = sorted(set(range(ctx.parent_spec[DEPTH_FIELD])) - set(kept))
        return OperatorOutcome(
            model=model,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference="none",
                values={"op.depth.positional.n_removed": float(len(removed))},
                detail={"note": "a positional heuristic takes no measurement, so it "
                                "reports no comparable objective value"}),
            trace={"kept_layers": kept, "removed_layers": removed,
                   "spans": spans, "source": "positional_pairwise_merge"},
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
        # On the compute device, like E8a's `prepare()`: "Token tensors, targets
        # and boolean tag masks, once, on the device." These meet the logits
        # inside `distortion`'s `gather`, so a host target would drag the whole
        # reduction back to the host — which is exactly what happened.
        targets = [item["input_ids"][0, 1:].to(compute) for item in items]
        reference = _ReferenceLogits(model, items, compute)

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
        timing = {"reference_seconds": 0.0, "ablated_seconds": 0.0,
                  "distortion_seconds": 0.0, "item_seconds": 0.0,
                  "candidate_subsets": 0, "ablated_forwards": 0,
                  "distortion_calls": 0, "split_is_attributed": bool(cuda_sync)
                  or torch.device(compute).type != "cuda"}

        def score(skip: frozenset[int]) -> float:
            per_subtype: dict[str, list[float]] = {}
            timing["candidate_subsets"] += 1
            item_started = time.perf_counter()
            for item, tgt in zip(items, targets):
                # Order matters: the reference is the UNBYPASSED parent, so when
                # it is being recomputed it must not be taken inside the bypass.
                t0 = time.perf_counter()
                ref = reference.get(item)
                if cuda_sync:
                    cuda_sync()
                t1 = time.perf_counter()
                abl = _forward_logits(model, item, compute, skip)
                if cuda_sync:
                    cuda_sync()
                t2 = time.perf_counter()
                sums = distortion(ref, abl, tgt, chunk=512).as_dict()
                t3 = time.perf_counter()
                timing["reference_seconds"] += t1 - t0
                timing["ablated_seconds"] += t2 - t1
                timing["distortion_seconds"] += t3 - t2
                timing["ablated_forwards"] += 1
                timing["distortion_calls"] += 1
                per_subtype.setdefault(item["subtype"], []).append(sums["kl"])
                del abl
            timing["item_seconds"] += time.perf_counter() - item_started
            means = {k: sum(v) / len(v) for k, v in per_subtype.items()}
            primary, _ = domain_balanced_score(means, domains)
            return primary

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
                                on_round=on_round, on_candidate=on_candidate)
        kept = result["kept"]
        child = _build_child_with_layers(ctx, kept)

        final_score = result["rounds"][-1]["chosen_score"] if result["rounds"] else 0.0
        return OperatorOutcome(
            model=child,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id,
                objective=self.objective,
                reference="parent_state",
                values={
                    "op.depth.causal_kl.final": float(final_score),
                    "op.depth.causal_kl.evaluations": float(result["evaluations"]),
                },
                detail={"removal_order": result["removal_order"]}),
            trace={"kept_layers": kept, "removed_layers": result["removed"],
                   "removal_order": result["removal_order"],
                   "source": "causal_kl_greedy"},
            # The memory decision is an artifact, NOT part of the trace. The
            # trace describes the deterministic result — `test_operator_is_
            # deterministic` compares two invocations of it field for field —
            # and which path ran depends on how much memory the host happened to
            # have free. The numbers are identical either way, so the decision
            # belongs with the byproducts.
            artifacts={"search_rounds": result["rounds"],
                       "reference_cache": reference.decision(),
                       # Operational only. `artifacts` is already excluded from
                       # the trace and from state identity — see the note above
                       # — which is exactly why the timings belong here and in
                       # no other field the operator returns.
                       "timing": {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in timing.items()},
                       "reference_counters": reference.counters()},
        )


def _domain_map(items) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in items:
        subs = out.setdefault(item["domain"], [])
        if item["subtype"] not in subs:
            subs.append(item["subtype"])
    return {k: sorted(v) for k, v in sorted(out.items())}


@torch.no_grad()
def _forward_logits(model, item, device: str, skip=frozenset()):
    """One item's prediction-position logits, optionally with blocks bypassed.

    Left in the model's own dtype **and on the model's device**. ``distortion``
    upcasts to float32 in chunks internally, so widening here would only double
    the bytes held per item without changing a single reduced value.

    **This returned ``.cpu()`` until 2026-08-19, and that cost $11.43.** E8a —
    ``scripts/training/search_depth_map.py``, the implementation this operator
    ports — keeps prepared inputs, reference logits, ablated logits and the
    ``distortion`` reduction on the selected accelerator. The port introduced the
    transfer, and with it a full 151,936-vocabulary softmax/KL on the host, 260
    evaluations x 67 items per expansion: ~86 TiB of CPU traffic and ~8.6 TiB
    copied off the device. Attempt 10 ran 10 h 47 m inside one expansion, GPU at
    0-1 %, and was stopped without finishing it.

    Nothing about the reduction changed to fix this. The tensors simply stay
    where E8a left them.
    """
    ids = item["input_ids"].to(device)
    if not skip:
        return model(ids).logits[0, :-1]
    with bypassed_blocks(model, skip):
        return model(ids).logits[0, :-1]


class _ReferenceLogits:
    """The unbypassed parent's logits, cached only when the memory fits.

    The reference is identical for every candidate in the whole greedy search,
    so caching it turns ``2 * evals * n_items`` forward passes into
    ``(evals + 1) * n_items`` — the cost structure ``plan()`` reports.

    It had been cached **unconditionally**, in float32, in host RAM. For the
    frozen ``calib.domain_balanced@v1`` mixture that is 59,763 prediction
    positions x 151,936 vocabulary x 4 B = **33.8 GiB**, held for the duration of
    the search, on every invocation of this operator. Nothing had ever executed
    it against the real mixture: Phase-A attempt 5 died earlier, at the
    calibration file, and every zero-cost run used a toy mixture. Its first real
    execution — this rehearsal — was killed by the OOM killer.

    ``scripts/training/search_depth_map.py``, the E8a script whose algorithm this
    operator re-runs, already had the answer and it was dropped in the port: size
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
        self.available_bytes, self.headroom_source = _available_memory_bytes(device)

        budget = (None if self.available_bytes is None
                  else self.BUDGET_FRACTION * self.available_bytes)
        self.enabled = budget is None or self.estimate_bytes <= budget

        # PARTIAL CACHING. Until 2026-08-27 this was all-or-nothing, and Phase-B
        # attempt 3 landed exactly in the gap: a 16.9 GiB reference against a
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
            "headroom_source": self.headroom_source,
            "fallback": None if self.enabled else (
                f"{self.mode}: {len(self.admitted)}/{n_items} items resident, the "
                "remainder recomputed per candidate; identical numbers"),
        }

    def counters(self) -> dict[str, int]:
        """Telemetry only — hits, fills and recomputes over the whole expansion."""
        return {"reference_hits": self.hits, "reference_fills": self.fills,
                "reference_recomputes": self.recomputes}


def _available_memory_bytes(device: Any) -> tuple[int | None, str]:
    """Free memory **where the cache will actually live**.

    Since 2026-08-19 the cache is device-resident, as E8a's always was, so on an
    accelerator this asks the accelerator. The previous version measured the
    host — correctly for the code as it then stood, because ``_forward_logits``
    returned ``.cpu()``. Its own docstring said so:

        "E8a kept its cache on the accelerator and therefore checked
        ``torch.cuda.mem_get_info``; copying that probe here would measure free
        VRAM against an allocation that never touches it."

    That reasoning was sound and the premise was the defect. The probe follows
    the cache rather than the other way round.

    The fallback when it does not fit is **recompute**, never a silent move to
    the host: a host-resident reference would drag the whole reduction back with
    it, which is the $11.43 failure.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":                       # pragma: no cover - needs a GPU
        try:
            free, _total = torch.cuda.mem_get_info(dev)
            return int(free), f"cuda.mem_get_info:{dev}"
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


DEPTH_POSITIONAL_V0 = register_implementation(DepthPositionalV0())
DEPTH_CAUSAL_KL_GREEDY_V1 = register_implementation(DepthCausalKLGreedyV1())
