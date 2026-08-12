"""A deliberately alien architecture family, defined entirely outside the core.

Nothing in ``src/aadistill/autoinit`` knows this exists. It is not a Qwen3 model,
it is not even a transformers model, its structural fields have names the Qwen3
adapter has never heard of, and its operator kind (``MOE_EXPERT_SET``) is not one
of the four the v1 library ships. If a beam search runs over it end to end
without a core edit, the family-agnosticism claim is demonstrated rather than
asserted.

Also registers ``attention.toy_mla_v1`` — a second implementation under the
*existing* ``ATTENTION`` kind with different required capabilities — to show the
other extension axis: a new attention family joins an existing kind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from aadistill.autoinit.arch import ArchitectureAdapter, ArchSpec, Capability
from aadistill.autoinit.metrics import OperatorLocalMetrics
from aadistill.autoinit.operators.base import (
    CalibrationNeed,
    OperatorImplementation,
    OperatorKindSpec,
    OperatorOutcome,
    OperatorPlan,
    register_implementation,
    register_kind,
)

TOY_FAMILY = "toymoe"
TOY_FIELDS = ("d_model", "n_experts", "expert_width", "vocab_size")

MOE_EXPERT_SET = register_kind(OperatorKindSpec(
    "MOE_EXPERT_SET", "expert count", "how many experts survive a reduction"))
MOE_EXPERT_WIDTH = register_kind(OperatorKindSpec(
    "MOE_EXPERT_WIDTH", "per-expert width", "the hidden width inside each expert"))


@dataclass
class ToyConfig:
    d_model: int
    n_experts: int
    expert_width: int
    vocab_size: int
    model_type: str = TOY_FAMILY

    def to_dict(self) -> dict[str, Any]:
        return {"d_model": self.d_model, "n_experts": self.n_experts,
                "expert_width": self.expert_width, "vocab_size": self.vocab_size,
                "model_type": self.model_type}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToyConfig":
        return cls(**{k: v for k, v in d.items() if k != "model_type"})


class ToyOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class ToyModel(torch.nn.Module):
    """embed -> mean over experts -> unembed. Enough to have real logits."""

    def __init__(self, config: ToyConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.experts_in = torch.nn.Parameter(
            torch.randn(config.n_experts, config.d_model, config.expert_width) * 0.05)
        self.experts_out = torch.nn.Parameter(
            torch.randn(config.n_experts, config.expert_width, config.d_model) * 0.05)
        self.unembed = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> ToyOutput:
        x = self.embed(input_ids)
        h = torch.einsum("btd,edw->btew", x, self.experts_in).relu()
        y = torch.einsum("btew,ewd->btd", h, self.experts_out) / self.config.n_experts
        return ToyOutput(self.unembed(x + y))


class ToyAdapter(ArchitectureAdapter):
    family = TOY_FAMILY
    adapter_version = "toymoe.test_v1"
    capabilities = frozenset({Capability.MOE_FFN, Capability.MOE_ROUTER,
                              Capability.LOGIT_COMPARABLE})
    structural_fields = TOY_FIELDS

    def spec_from_config(self, config: Any) -> ArchSpec:
        return ArchSpec.of(self.family, {k: getattr(config, k) for k in TOY_FIELDS})

    def build_config(self, base_config: Any, spec: ArchSpec) -> Any:
        config = ToyConfig.from_dict(base_config.to_dict())
        for key, value in spec.fields:
            setattr(config, key, value)
        return config

    def param_count(self, spec: ArchSpec) -> int:
        d, e, w, v = (spec["d_model"], spec["n_experts"], spec["expert_width"],
                      spec["vocab_size"])
        return v * d + e * d * w + e * w * d + v * d

    def validate_target(self, spec: ArchSpec) -> None:
        if spec["n_experts"] < 1:
            raise ValueError("need at least one expert")

    def build_model(self, config: Any, dtype: Any, seed: int) -> Any:
        torch.manual_seed(seed)
        return ToyModel(config).to(dtype or torch.float32).eval()

    def save(self, model: Any, path: str, *,
             max_shard_size: str | int | None = None) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()},
                  str(p / "model.safetensors"))
        (p / "config.json").write_text(json.dumps(model.config.to_dict(), sort_keys=True))

    def load(self, path: str, dtype: Any = None, device: str = "cpu") -> Any:
        p = Path(path)
        config = ToyConfig.from_dict(json.loads((p / "config.json").read_text()))
        model = ToyModel(config)
        model.load_state_dict(load_file(str(p / "model.safetensors")))
        return model.to(device).eval()

    def weight_files(self, path: str) -> list[str]:
        return sorted(f.name for f in Path(path).glob("*.safetensors"))


class ToyExpertSetReduction(OperatorImplementation):
    """A structural transformation that has no counterpart in the v1 library."""

    impl_id = "moe.expert_set_topk_v1"
    kind = "MOE_EXPERT_SET"
    version = 1
    description = "keep the experts with the largest ||in||*||out|| product"
    required_capabilities = frozenset({Capability.MOE_FFN})
    modifies = frozenset({"n_experts"})
    preserves = frozenset({"d_model", "expert_width", "vocab_size"})
    calibration = CalibrationNeed.NONE
    objective = "retained share of expert magnitude"

    def plan(self, spec, target, adapter, config=None):
        return OperatorPlan(self.impl_id, spec.replace(n_experts=target["n_experts"]),
                            0, 0, "weight-only expert scoring")

    @torch.no_grad()
    def apply(self, ctx):
        keep = ctx.target_spec["n_experts"]
        parent = ctx.model
        scores = (parent.experts_in.flatten(1).norm(dim=1)
                  * parent.experts_out.flatten(1).norm(dim=1))
        kept = scores.topk(keep).indices.sort().values
        spec = ctx.parent_spec.replace(n_experts=keep)
        child = ctx.adapter.build_model(
            ctx.adapter.build_config(parent.config, spec), torch.float32, ctx.seed)
        child.embed.weight.copy_(parent.embed.weight)
        child.unembed.weight.copy_(parent.unembed.weight)
        child.experts_in.copy_(parent.experts_in[kept])
        child.experts_out.copy_(parent.experts_out[kept])
        return OperatorOutcome(
            model=child.eval(),
            local_metrics=OperatorLocalMetrics(
                self.impl_id, self.objective, "parent_state",
                {"op.moe.retained_expert_share": float(scores[kept].sum() / scores.sum())}),
            trace={"kept_experts": kept.tolist()})


class ToyExpertWidthReduction(OperatorImplementation):
    impl_id = "moe.expert_width_topk_v1"
    kind = "MOE_EXPERT_WIDTH"
    version = 1
    description = "narrow each expert by per-unit magnitude"
    required_capabilities = frozenset({Capability.MOE_FFN})
    modifies = frozenset({"expert_width"})
    preserves = frozenset({"d_model", "n_experts", "vocab_size"})
    calibration = CalibrationNeed.NONE
    objective = "retained share of unit magnitude"

    def plan(self, spec, target, adapter, config=None):
        return OperatorPlan(self.impl_id,
                            spec.replace(expert_width=target["expert_width"]), 0, 0)

    @torch.no_grad()
    def apply(self, ctx):
        keep = ctx.target_spec["expert_width"]
        parent = ctx.model
        scores = parent.experts_in.norm(dim=1).sum(0) * parent.experts_out.norm(dim=2).sum(0)
        kept = scores.topk(keep).indices.sort().values
        spec = ctx.parent_spec.replace(expert_width=keep)
        child = ctx.adapter.build_model(
            ctx.adapter.build_config(parent.config, spec), torch.float32, ctx.seed)
        child.embed.weight.copy_(parent.embed.weight)
        child.unembed.weight.copy_(parent.unembed.weight)
        child.experts_in.copy_(parent.experts_in[:, :, kept])
        child.experts_out.copy_(parent.experts_out[:, kept, :])
        return OperatorOutcome(
            model=child.eval(),
            local_metrics=OperatorLocalMetrics(
                self.impl_id, self.objective, "parent_state",
                {"op.moe.retained_width_share": float(scores[kept].sum() / scores.sum())}),
            trace={"kept_units": kept.tolist()})


class ToyMLAAttention(OperatorImplementation):
    """A second ATTENTION implementation for a family that is not GQA.

    Never executed here; it exists to prove that a new attention family joins an
    existing kind through its declared capabilities, and that the dispatcher does
    not offer it to an adapter that cannot support it.
    """

    impl_id = "attention.toy_mla_v1"
    kind = "ATTENTION"
    version = 1
    description = "latent-attention head reduction for an MLA family"
    required_capabilities = frozenset({Capability.ATTENTION_MLA})
    modifies = frozenset({"num_attention_heads"})
    preserves = frozenset({"hidden_size"})
    calibration = CalibrationNeed.NONE
    objective = "latent reconstruction error"

    def plan(self, spec, target, adapter, config=None):  # pragma: no cover
        return OperatorPlan(self.impl_id, spec, 0, 0)

    def apply(self, ctx):  # pragma: no cover
        raise NotImplementedError("registration-only fixture")


def register_all():
    register_implementation(ToyExpertSetReduction())
    register_implementation(ToyExpertWidthReduction())
    register_implementation(ToyMLAAttention())
