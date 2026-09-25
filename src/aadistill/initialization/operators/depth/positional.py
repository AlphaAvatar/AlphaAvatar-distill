"""``depth.positional_v0`` — the incumbent positional depth map.

Positional pairwise merge in a middle band. Takes no measurement and reads no
calibration, so it issues no model forward and micro-batching does not apply to
it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aadistill.initialization.transforms.sandwich import depth_span_map
from aadistill.initialization.specs.arch import (
    ArchitectureAdapter,
    ArchSpec,
    Capability,
)
from aadistill.initialization.specs.metrics import OperatorLocalMetrics
from aadistill.initialization.calibration.profiles import CalibrationNeed
from aadistill.initialization.operators.base import (
    OperatorContext,
    OperatorImplementation,
    OperatorOutcome,
    OperatorPlan,
)
from aadistill.initialization.operators.depth._common import (
    DEPTH_FIELD,
    _build_child_with_layers,
)


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




#: The instance, NOT a registration.
DEPTH_POSITIONAL_V0 = DepthPositionalV0()
