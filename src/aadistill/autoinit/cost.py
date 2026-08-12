"""Cost and branching model for an AutoInitializer search.

Zero-cost output whose only job is to make the next decision — beam width, Top-N,
survivor count — an arithmetic one rather than a guess. Everything is derived
from an ``ArchSpec`` through the adapter, so the same model prices a 4B -> 596M
pilot and a 30B -> 4.xB study with no edit.

Anchors, and how good each one is:

* **88.4 effective TFLOP/s on an L40S**, *measured*. E8a's depth search ran 260
  subset evaluations over a 67-item / 59,763-position mixture in 1,300 s at full
  4B width. The FLOP accounting below, applied to that exact workload, gives
  1.150e17 FLOPs, hence 88.4 TFLOP/s and ~24% MFU. That is a real end-to-end
  number including the Python loop, not a spec sheet.
* **Other accelerators are scaled by peak bf16 and marked ESTIMATED.** Scaling
  MFU across architectures is an assumption; it is labelled as one.
* **Activation-statistics collection is a range, not a point.** The one
  measurement available is 4,972 s for 949,859 tokens (5.24 ms/token) on this
  CPU-only dev box, which includes the model forward. On a GPU the forward
  collapses but the float64 ``X^T X`` accumulation stays on the CPU, and the
  split has never been measured. So the model reports a lower bound (GPU forward
  only) and an upper bound (the full CPU rate) and says so. Measuring the split
  is a cheap pre-pilot task and is listed as one.

Nothing here is allowed to quietly shrink a plan to fit a budget. ``price_search``
reports what the search costs; comparing that against an authorization is the
caller's job, and reporting a shortfall specifically is the maintainer's rule.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .arch import ArchitectureAdapter, ArchSpec
from .operators.base import OperatorImplementation

BYTES_PER_PARAM_BF16 = 2


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    price_per_hour_usd: float
    effective_tflops: float
    vram_gb: float
    measured: bool
    source: str

    def seconds_for(self, flops: float) -> float:
        return flops / (self.effective_tflops * 1e12)

    def usd_for(self, seconds: float) -> float:
        return seconds / 3600.0 * self.price_per_hour_usd

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "price_per_hour_usd": self.price_per_hour_usd,
                "effective_tflops": self.effective_tflops, "vram_gb": self.vram_gb,
                "measured": self.measured, "source": self.source}


L40S_MEASURED = HardwareProfile(
    name="L40S", price_per_hour_usd=0.99, effective_tflops=88.83, vram_gb=48.0,
    measured=True,
    source=("E8a: 260 subset evaluations over a 67-item / 59,763-position mixture in "
            "1,300 s at 4.02B full width. This module's own accounting puts that "
            "workload at 1.1548e17 FLOPs, hence 88.83 TFLOP/s and ~24.5% of the "
            "362 TFLOPS bf16 dense peak. Asserted round-trip in tests/autoinit."),
)

A100_80GB_ESTIMATED = HardwareProfile(
    name="A100-SXM-80GB", price_per_hour_usd=1.59, effective_tflops=76.56, vram_gb=80.0,
    measured=False,
    source=("ESTIMATED: the L40S's measured 88.83 scaled by the bf16 dense peak ratio "
            "312/362. MFU is assumed to transfer across architectures; it has not "
            "been measured on this workload."),
)

DEVBOX_CPU_MEASURED = HardwareProfile(
    name="dev-box CPU (16 vCPU)", price_per_hour_usd=0.0, effective_tflops=1.59,
    vram_gb=0.0, measured=True,
    source=("Stage 0 regeneration: 949,859 tokens of 4.02B forward + float64 X^T X "
            "accumulation in 4,972 s"),
)

#: End-to-end statistics-collection rate on the CPU-only dev box, model forward
#: included. Upper bound for a GPU run, because the float64 accumulation does not
#: move to the accelerator in the current implementation.
CPU_STATS_SECONDS_PER_TOKEN = 4972.0 / 949_859.0


# --- FLOP accounting --------------------------------------------------------


def forward_flops_per_token(spec: ArchSpec, seq_len: int, *,
                            layers: float | None = None) -> float:
    """Forward FLOPs for one token, at ``seq_len`` context.

    Attention score/value products are included with a 0.5 causal factor; at the
    sequence lengths a calibration mixture uses they are ~9% of the total, which
    is small but not nothing. ``layers`` may be fractional so a greedy depth
    search — whose candidates have progressively fewer blocks — can be priced
    with its true average rather than the full stack.
    """
    d = spec["hidden_size"]
    n_q, n_kv = spec["num_attention_heads"], spec["num_key_value_heads"]
    head_dim, inter = spec["head_dim"], spec["intermediate_size"]
    vocab = spec["vocab_size"]
    n_layers = spec["num_hidden_layers"] if layers is None else layers

    proj = 2.0 * (n_q * head_dim * d + 2 * n_kv * head_dim * d + d * n_q * head_dim)
    attn = 0.5 * 2.0 * (2.0 * seq_len * n_q * head_dim)
    mlp = 2.0 * 3.0 * inter * d
    return n_layers * (proj + attn + mlp) + 2.0 * vocab * d


def checkpoint_bytes(spec: ArchSpec, adapter: ArchitectureAdapter,
                     bytes_per_param: int = BYTES_PER_PARAM_BF16) -> int:
    return adapter.param_count(spec) * bytes_per_param


def activation_stats_bytes(spec: ArchSpec) -> int:
    """float64 residual second moments dominate: (L+1) x d x d x 8 bytes.

    Reported because it scales as ``d^2`` and is a real 30B -> 4.xB risk: at the
    4B teacher it is 1.94 GB, and the same formula at a 6144-wide, 64-layer
    teacher is ~19.6 GB of float64 state per statistics pass.
    """
    d = spec["hidden_size"]
    n_points = spec["num_hidden_layers"] + 1
    moments = n_points * d * d * 8
    ffn = 2 * spec["num_hidden_layers"] * spec["intermediate_size"] * 8
    return moments + ffn + spec["vocab_size"] * 8


# --- per-operator cost ------------------------------------------------------


@dataclass(frozen=True)
class OperatorCost:
    impl_id: str
    kind: str
    parent_spec_hash: str
    flops: float
    gpu_seconds: float
    stats_seconds_low: float
    stats_seconds_high: float
    child_bytes: int
    peak_resident_bytes: int
    notes: str = ""

    def seconds_low(self) -> float:
        return self.gpu_seconds + self.stats_seconds_low

    def seconds_high(self) -> float:
        return self.gpu_seconds + self.stats_seconds_high

    def as_dict(self) -> dict[str, Any]:
        return {
            "impl_id": self.impl_id, "kind": self.kind,
            "parent_spec_hash": self.parent_spec_hash,
            "flops": self.flops, "gpu_seconds": self.gpu_seconds,
            "stats_seconds_low": self.stats_seconds_low,
            "stats_seconds_high": self.stats_seconds_high,
            "seconds_low": self.seconds_low(), "seconds_high": self.seconds_high(),
            "child_gib": self.child_bytes / 2**30,
            "peak_resident_gib": self.peak_resident_bytes / 2**30,
            "notes": self.notes,
        }


def greedy_depth_flops(spec: ArchSpec, n_remove: int, tokens: int,
                       seq_len: int) -> tuple[float, float]:
    """FLOPs and the average surviving-layer count for a full greedy removal.

    Round ``r`` evaluates ``L - r`` candidates, each with ``r + 1`` blocks
    bypassed, so the work is not ``260 x full model`` — it is the exact sum below,
    plus one intact reference pass.
    """
    n_layers = spec["num_hidden_layers"]
    total, evaluations = 0.0, 0
    weighted_layers = 0.0
    for r in range(n_remove):
        candidates = n_layers - r
        surviving = n_layers - (r + 1)
        total += candidates * tokens * forward_flops_per_token(spec, seq_len,
                                                               layers=surviving)
        weighted_layers += candidates * surviving
        evaluations += candidates
    total += tokens * forward_flops_per_token(spec, seq_len)  # intact reference
    return total, (weighted_layers / evaluations if evaluations else 0.0)


def operator_cost(impl: OperatorImplementation, parent_spec: ArchSpec,
                  target_spec: ArchSpec, adapter: ArchitectureAdapter, *,
                  calibration_tokens: int, seq_len: int,
                  hardware: HardwareProfile) -> OperatorCost:
    plan = impl.plan(parent_spec, target_spec, adapter,
                     {"n_calibration_items": 1})
    child_spec = plan.result_spec
    flops = 0.0
    stats_low = stats_high = 0.0
    notes = plan.notes

    if impl.impl_id == "depth.causal_kl_greedy_v1":
        n_remove = parent_spec["num_hidden_layers"] - target_spec["num_hidden_layers"]
        flops, avg_layers = greedy_depth_flops(parent_spec, n_remove,
                                               calibration_tokens, seq_len)
        notes = f"{plan.notes}; mean {avg_layers:.2f} surviving blocks per evaluation"
    elif plan.stats_passes:
        # One forward with hooks, then a float64 eigendecomposition or top-k.
        flops = calibration_tokens * forward_flops_per_token(parent_spec, seq_len)
        stats_low = hardware.seconds_for(flops)
        stats_high = calibration_tokens * CPU_STATS_SECONDS_PER_TOKEN
        flops = 0.0  # accounted inside the stats bounds instead of twice
        notes = (f"{plan.notes}; statistics pass priced as a range: GPU-forward-only "
                 f"lower bound vs the measured CPU end-to-end rate")

    child_bytes = checkpoint_bytes(child_spec, adapter)
    parent_bytes = checkpoint_bytes(parent_spec, adapter)
    peak = parent_bytes + child_bytes
    if plan.stats_passes:
        peak += activation_stats_bytes(parent_spec)

    return OperatorCost(
        impl_id=impl.impl_id, kind=impl.kind,
        parent_spec_hash=parent_spec.spec_hash, flops=flops,
        gpu_seconds=hardware.seconds_for(flops),
        stats_seconds_low=stats_low, stats_seconds_high=stats_high,
        child_bytes=child_bytes, peak_resident_bytes=peak, notes=notes)


def evaluation_cost(spec: ArchSpec, *, suite_tokens: int, seq_len: int,
                    hardware: HardwareProfile) -> float:
    """One state evaluation: a single forward of the candidate over the suite."""
    return hardware.seconds_for(suite_tokens * forward_flops_per_token(spec, seq_len))


# --- path enumeration and branching ----------------------------------------


@dataclass(frozen=True)
class PathNode:
    level: int
    impl_id: str
    kind: str
    parent_spec: ArchSpec
    child_spec: ArchSpec


def enumerate_paths(root_spec: ArchSpec, target_spec: ArchSpec,
                    adapter: ArchitectureAdapter,
                    impls: Sequence[OperatorImplementation]) -> list[list[PathNode]]:
    """Every complete decomposed path: kind orderings x implementation choices.

    Calibration profiles are *not* multiplied in here — they do not change the
    geometry, so they multiply the node count without changing the per-node cost.
    ``branching_estimate`` applies them.
    """
    differing = sorted(root_spec.diff(target_spec))
    by_field: dict[str, list[OperatorImplementation]] = {}
    for impl in impls:
        for field_name in impl.modifies:
            if field_name in differing and len(impl.modifies) == 1:
                by_field.setdefault(field_name, []).append(impl)
    if sorted(by_field) != differing:
        missing = sorted(set(differing) - set(by_field))
        raise ValueError(f"no single-field implementation covers {missing}")

    paths: list[list[PathNode]] = []
    for order in itertools.permutations(differing):
        choices = [sorted(by_field[f], key=lambda i: i.impl_id) for f in order]
        for combo in itertools.product(*choices):
            spec = root_spec
            nodes = []
            for level, impl in enumerate(combo):
                child = impl.plan(spec, target_spec, adapter,
                                  {"n_calibration_items": 1}).result_spec
                nodes.append(PathNode(level, impl.impl_id, impl.kind, spec, child))
                spec = child
            if not spec.matches(target_spec):  # pragma: no cover - defensive
                raise ValueError("enumerated path does not reach the target")
            paths.append(nodes)
    return paths


def branching_estimate(root_spec: ArchSpec, target_spec: ArchSpec,
                       adapter: ArchitectureAdapter,
                       impls: Sequence[OperatorImplementation],
                       *, n_profiles: int, beam_width: int,
                       include_composite: Sequence[OperatorImplementation] = ()) -> dict[str, Any]:
    """How many states a beam of this width actually materializes.

    Reported as a range rather than a point because the per-level branching
    depends on *which* kinds the surviving beam members happen to have applied —
    a parent that has already used DEPTH has fewer remaining options than one
    that has not, and which parents survive is a measurement outcome.
    """
    paths = enumerate_paths(root_spec, target_spec, adapter, impls)
    differing = sorted(root_spec.diff(target_spec))
    n_kinds = len(differing)

    by_field: dict[str, int] = {}
    for impl in impls:
        if len(impl.modifies) == 1:
            f = next(iter(impl.modifies))
            if f in differing:
                by_field[f] = by_field.get(f, 0) + 1

    per_level: list[dict[str, Any]] = []
    total_min = total_max = 0
    for level in range(n_kinds):
        parents = 1 if level == 0 else beam_width
        remaining = n_kinds - level
        options = sorted(by_field.values())
        # Best case for branching width: the beam kept the parents that still
        # have the most-implemented kinds available. Worst case: the opposite.
        max_pairs = sum(options[-remaining:])
        min_pairs = sum(options[:remaining])
        lo, hi = parents * min_pairs * n_profiles, parents * max_pairs * n_profiles
        per_level.append({"level": level, "parents": parents,
                          "children_min": lo, "children_max": hi})
        total_min += lo
        total_max += hi

    composite_children = len(include_composite) * n_profiles
    return {
        "differing_fields": differing,
        "n_kinds": n_kinds,
        "implementations_per_field": dict(sorted(by_field.items())),
        "n_calibration_profiles": n_profiles,
        "beam_width": beam_width,
        "complete_paths_unbeamed": len(paths) * (n_profiles ** n_kinds),
        "complete_paths_geometry_only": len(paths),
        "kind_orderings": len(list(itertools.permutations(differing))),
        "per_level": per_level,
        "states_materialized_min": total_min + composite_children,
        "states_materialized_max": total_max + composite_children,
        "composite_leaves": composite_children,
        "leaves_min": per_level[-1]["children_min"] + composite_children,
        "leaves_max": per_level[-1]["children_max"] + composite_children,
    }


# --- whole-search pricing ---------------------------------------------------


@dataclass
class SearchCostEstimate:
    hardware: HardwareProfile
    branching: Mapping[str, Any]
    seconds_low: float
    seconds_high: float
    usd_low: float
    usd_high: float
    peak_storage_bytes_retained: int
    peak_storage_bytes_working: int
    total_bytes_written: int
    peak_resident_bytes: int
    per_operator: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hardware": self.hardware.as_dict(),
            "branching": dict(self.branching),
            "seconds_low": self.seconds_low, "seconds_high": self.seconds_high,
            "hours_low": self.seconds_low / 3600, "hours_high": self.seconds_high / 3600,
            "usd_low": round(self.usd_low, 4), "usd_high": round(self.usd_high, 4),
            "peak_storage_gib_retained": self.peak_storage_bytes_retained / 2**30,
            "peak_storage_gib_working": self.peak_storage_bytes_working / 2**30,
            "total_gib_written": self.total_bytes_written / 2**30,
            "peak_resident_gib": self.peak_resident_bytes / 2**30,
            "per_operator": self.per_operator,
            "notes": list(self.notes),
        }


def price_search(root_spec: ArchSpec, target_spec: ArchSpec,
                 adapter: ArchitectureAdapter,
                 impls: Sequence[OperatorImplementation], *,
                 calibration_tokens: int, suite_tokens: int, seq_len: int,
                 n_profiles: int, beam_width: int, hardware: HardwareProfile,
                 composite: Sequence[OperatorImplementation] = ()) -> SearchCostEstimate:
    """Price the search by averaging real per-node costs over the enumerated paths.

    Per-node cost genuinely depends on position: a causal depth search at the
    head of a path runs against the full-width teacher, and the same search at
    the tail runs against an already-compressed state for a small fraction of the
    FLOPs. Averaging over the enumerated geometry captures that instead of
    pricing every node at the worst case.
    """
    paths = enumerate_paths(root_spec, target_spec, adapter, impls)
    branching = branching_estimate(root_spec, target_spec, adapter, impls,
                                   n_profiles=n_profiles, beam_width=beam_width,
                                   include_composite=composite)
    by_impl = {i.impl_id: i for i in impls}

    level_costs: dict[int, list[OperatorCost]] = {}
    child_sizes: dict[int, list[int]] = {}
    for path in paths:
        for node in path:
            cost = operator_cost(by_impl[node.impl_id], node.parent_spec, target_spec,
                                 adapter, calibration_tokens=calibration_tokens,
                                 seq_len=seq_len, hardware=hardware)
            eval_seconds = evaluation_cost(node.child_spec, suite_tokens=suite_tokens,
                                           seq_len=seq_len, hardware=hardware)
            level_costs.setdefault(node.level, []).append(cost)
            child_sizes.setdefault(node.level, []).append(cost.child_bytes)
            level_costs[node.level][-1] = OperatorCost(
                **{**cost.__dict__, "gpu_seconds": cost.gpu_seconds + eval_seconds})

    seconds_low = seconds_high = 0.0
    per_operator: list[dict[str, Any]] = []
    for entry in branching["per_level"]:
        level = entry["level"]
        costs = level_costs.get(level, [])
        if not costs:
            continue
        mean_low = sum(c.seconds_low() for c in costs) / len(costs)
        mean_high = sum(c.seconds_high() for c in costs) / len(costs)
        seconds_low += entry["children_min"] * mean_low
        seconds_high += entry["children_max"] * mean_high
        per_operator.append({
            "level": level,
            "children_min": entry["children_min"], "children_max": entry["children_max"],
            "mean_node_seconds_low": mean_low, "mean_node_seconds_high": mean_high,
            "max_node_seconds_high": max(c.seconds_high() for c in costs),
            "implementations": sorted({c.impl_id for c in costs}),
        })

    # The teacher's reference logits over the suite are computed once for the
    # whole search, not per candidate.
    teacher_ref = evaluation_cost(root_spec, suite_tokens=suite_tokens,
                                  seq_len=seq_len, hardware=hardware)
    seconds_low += teacher_ref
    seconds_high += teacher_ref

    # Storage has three distinct numbers and conflating them understates the
    # disk a run actually needs.
    #
    #   peak_working  the worst moment: a level's whole child set is on disk at
    #                 once, because ranking cannot start until every child has
    #                 been measured, and the beam parents are still there because
    #                 they were the things expanded.
    #   total_written I/O volume over the run, which is what a network volume
    #                 charges for and what a slow disk turns into wall clock.
    #   retained_end  what survives once pruned weights are released.
    peak_working = 0
    total_written = 0.0
    for entry in branching["per_level"]:
        level = entry["level"]
        if level not in child_sizes:
            continue
        sizes = child_sizes[level]
        mean_child = sum(sizes) / len(sizes)
        parents_resident = beam_width * max(
            c.peak_resident_bytes - c.child_bytes for c in level_costs[level])
        peak_working = max(peak_working,
                           int(parents_resident + entry["children_max"] * mean_child))
        total_written += entry["children_max"] * mean_child

    retained = beam_width * max(max(v) for v in child_sizes.values())
    peak_resident = max(max(c.peak_resident_bytes for c in costs)
                        for costs in level_costs.values())

    notes = [
        f"hardware anchor: {hardware.source}",
        "statistics-pass cost is a range; the GPU/CPU split of the float64 "
        "accumulation has never been measured separately",
        "storage: 'working' is the concurrent peak (a level's whole child set is "
        "on disk before ranking can start), 'written' is total I/O volume, "
        "'retained' is what survives once pruned weights are released",
    ]
    if not hardware.measured:
        notes.append(f"{hardware.name} throughput is ESTIMATED, not measured")

    return SearchCostEstimate(
        hardware=hardware, branching=branching,
        seconds_low=seconds_low, seconds_high=seconds_high,
        usd_low=hardware.usd_for(seconds_low), usd_high=hardware.usd_for(seconds_high),
        peak_storage_bytes_retained=int(retained),
        peak_storage_bytes_working=int(peak_working),
        total_bytes_written=int(total_written),
        peak_resident_bytes=int(peak_resident),
        per_operator=per_operator, notes=notes)
