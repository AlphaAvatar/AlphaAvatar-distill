"""The incumbent Stage-1 recipe, exposed as a single composite operator.

``init_student`` does depth, width, FFN and attention **simultaneously**, entirely
in float64 from the teacher's weights, casting once at assignment. A four-step
decomposition cannot reproduce it bitwise even in principle: each intermediate
checkpoint is materialized in the working dtype, so rounding enters three times
that the monolithic path never incurs, and — more importantly — every operator
after the first would measure a checkpoint rather than the teacher. Those are
different algorithms, not different spellings of one.

So the incumbent stays whole, under its own id, and enters the search as a
**leaf-producing composite**: one operator, teacher to target, the exact code path
that produced ``86fbba78...``. Two things follow.

*The search can be compared against its own baseline.* A beam that cannot
re-derive the incumbent has not beaten it; a beam that ranks a decomposed path
above it has, on that ranking's terms.

*The kind set is demonstrably open.* ``COMPOSITE_STAGE1`` is registered here, in
an operator module, not in ``base.py`` alongside the four structural kinds — the
same path a future ``MOE_EXPERT_SET`` takes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...init.sandwich import init_student
from ..arch import ArchitectureAdapter, ArchSpec, Capability
from ..metrics import OperatorLocalMetrics
from ._common import collect_activation_stats, model_dtype
from .base import (
    CalibrationNeed,
    OperatorContext,
    OperatorError,
    OperatorImplementation,
    OperatorKindSpec,
    OperatorOutcome,
    OperatorPlan,
    register_implementation,
    register_kind,
)

COMPOSITE_STAGE1 = register_kind(OperatorKindSpec(
    "COMPOSITE_STAGE1", "all four structural dimensions at once",
    "A single transformation from teacher to target that decides depth, residual "
    "width, FFN width and head structure jointly in one numerical pass. Exists so "
    "the incumbent recipe is a comparable member of the search space rather than "
    "an untested baseline outside it."))

ALL_STRUCTURAL = frozenset({"hidden_size", "num_hidden_layers", "intermediate_size",
                            "num_attention_heads"})


class CompositeStage1SandwichV0(OperatorImplementation):
    impl_id = "composite.stage1_sandwich_v0"
    kind = "COMPOSITE_STAGE1"
    version = 0
    description = (
        "aadistill.init.sandwich.init_student verbatim: global activation-PCA stream "
        "projection, norm folding with sqrt(d_t/d_s) compensation, per-group query "
        "head selection, per-layer activation-importance FFN top-k, and either the "
        "positional depth map or an explicit kept-layer list. The recipe that "
        "produced qwen3_0p6b_init_v0 (86fbba78...) and e8_contribution_init_v1 "
        "(7a0694a5...).")
    required_capabilities = frozenset({Capability.RESIDUAL_STREAM,
                                       Capability.PRENORM_BLOCKS,
                                       Capability.DENSE_FFN,
                                       Capability.ATTENTION_GQA,
                                       Capability.TIED_EMBEDDINGS,
                                       Capability.ACTIVATION_STATS})
    modifies = ALL_STRUCTURAL
    preserves = frozenset({"num_key_value_heads", "head_dim", "vocab_size",
                           "tie_word_embeddings"})
    calibration = CalibrationNeed.ACTIVATION_STATS
    objective = "captured energy fraction (projection diagnostic)"
    deterministic = True
    requires_seed = True
    produces = ("init_diagnostics", "depth_map")
    target_validation = "result spec must equal the target exactly in one step"

    def applicable(self, spec: ArchSpec, target: ArchSpec,
                   adapter: ArchitectureAdapter) -> tuple[bool, str]:
        if not self.supported_by(adapter):
            return False, f"adapter lacks {sorted(self.required_capabilities - adapter.capabilities)}"
        changed = spec.diff(target)
        if not changed:
            return False, "already at target"
        illegal = sorted(changed - self.modifies)
        if illegal:
            return False, f"cannot reach the target: it also differs in {illegal}"
        # All-or-nothing. This is the incumbent *whole* recipe, teacher to target
        # in one numerical pass; a partially compressed parent is not a state it
        # was ever defined on, and running it there would silently mean something
        # else while carrying the id of the frozen historical algorithm.
        if changed != self.modifies:
            return False, (f"applies only from an uncompressed root: {sorted(self.modifies - changed)} "
                           "already match the target")
        return True, "ok"

    def plan(self, spec: ArchSpec, target: ArchSpec, adapter: ArchitectureAdapter,
             config: Mapping[str, Any] | None = None) -> OperatorPlan:
        n_items = int((config or {}).get("n_calibration_items", 0))
        return OperatorPlan(
            impl_id=self.impl_id, result_spec=target,
            forward_passes=0, stats_passes=max(n_items, 1),
            notes="one statistics pass over the mixture, then a single float64 pass")

    @torch.no_grad()
    def apply(self, ctx: OperatorContext) -> OperatorOutcome:
        adapter = ctx.adapter
        parent = ctx.model
        cfg = dict(ctx.config or {})
        kept_layers = cfg.get("kept_layers")

        state = cfg.get("activation_state")
        if state is None:
            state = ctx.cached_stats(lambda: collect_activation_stats(
                adapter, parent, (i["input_ids"] for i in ctx.calibration_items),
                ctx.device))

        dtype = model_dtype(adapter, parent)
        student_config = adapter.build_config(parent.config, ctx.target_spec)
        child = adapter.build_model(student_config, dtype, ctx.seed)
        diag = init_student(parent, child, state, kept_layers=kept_layers)

        if cfg.get("verify_full_assignment"):
            # Any parameter init_student failed to write would still hold its
            # random draw, so a second build under a different seed would differ.
            other = adapter.build_model(student_config, dtype, ctx.seed + 1)
            init_student(parent, other, state, kept_layers=kept_layers)
            for (name, a), (_, b) in zip(child.named_parameters(),
                                         other.named_parameters()):
                if not torch.equal(a, b):
                    raise OperatorError(
                        f"{self.impl_id}: {name} depends on the build seed, so the "
                        "composite did not assign it")

        child.eval()
        proj = diag.get("projection", {})
        values = {}
        if "energy_captured_frac" in proj:
            values["op.composite.energy_captured_frac"] = float(proj["energy_captured_frac"])
        values["op.composite.scale_compensation"] = float(diag["scale_compensation"])
        return OperatorOutcome(
            model=child,
            local_metrics=OperatorLocalMetrics(
                impl_id=self.impl_id, objective=self.objective,
                reference="parent_state", values=values,
                detail={"depth_map_source": diag["depth_map_source"]}),
            trace={"kept_layers": diag["kept_teacher_layers"],
                   "removed_layers": diag["removed_teacher_layers"],
                   "source": diag["depth_map_source"]},
            artifacts={"init_diagnostics": {
                "projection": {k: v for k, v in proj.items() if k != "points"},
                "final_norm_weight_range": diag["final_norm_weight_range"],
                "depth_map": diag["depth_map"],
            }},
        )


COMPOSITE_STAGE1_SANDWICH_V0 = register_implementation(CompositeStage1SandwichV0())
