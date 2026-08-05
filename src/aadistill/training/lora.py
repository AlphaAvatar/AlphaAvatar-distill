"""Minimal native LoRA for constrained attention adaptation (Experiment 3, A2).

Why this exists rather than a dependency: the trainer needs three properties
that a runtime-only adapter library does not give for free — exact resume,
LoRA identity recorded in checkpoint metadata, and **no adapter required at
evaluation time**. Formal measurement must run the same inference architecture
for every arm (A0/A1/A2), so a LoRA run's saved checkpoint is a plain
Hugging Face checkpoint with the delta already folded into q/k/v/o.

Design, and the two decisions that make it work:

1. **`LoRALinear` subclasses `nn.Linear` and *shares* the base weight tensor.**
   The replacement therefore keeps the exact `state_dict` key of the module it
   replaced (`….q_proj.weight`), costs no extra memory for the base matrix, and
   still satisfies `isinstance(m, nn.Linear)` for any code that scans linears.
   `nn.Linear.__init__` is deliberately bypassed: it would allocate a second
   weight matrix only to have it thrown away.

2. **The saved `model/` directory is always the *merged* model.** The live
   training graph keeps base and delta separate; saving folds
   `scaling · B @ A` into a copy. Exact resume is preserved by writing the
   frozen base attention weights and the raw LoRA tensors next to it
   (`lora_state.safetensors`), so nothing about the training state is inferred
   from the merged weights by subtraction — which would not be exact in
   floating point.

Initialization is `B = 0`, so the initial merged model is *exactly* the model
before LoRA was applied and the initial output delta is exactly zero. `A` is
drawn from `U(-1/sqrt(fan_in), 1/sqrt(fan_in))` — `nn.Linear`'s own default —
using an explicit `torch.Generator`, so the draw is a pure function of
(seed, module order, shapes) and does not depend on, or perturb, global RNG.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

# Suffix-anchored so a pattern can never match a parent module by accident.
DEFAULT_TARGET_PATTERNS = (r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",)


@dataclass(frozen=True)
class LoRAConfig:
    """The A2 adapter contract. Serialized verbatim into checkpoint metadata."""

    rank: int
    alpha: float
    dropout: float = 0.0
    bias: str = "none"
    seed: int = 0
    target_patterns: tuple[str, ...] = field(default=DEFAULT_TARGET_PATTERNS)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"lora.rank must be >= 1, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"lora.alpha must be > 0, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"lora.dropout must be in [0, 1), got {self.dropout}")
        if self.bias != "none":
            raise ValueError(
                f"only lora.bias='none' is implemented, got {self.bias!r}")
        if not self.target_patterns:
            raise ValueError("lora.target_patterns must not be empty")

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        known = {"rank", "alpha", "dropout", "bias", "seed", "target_patterns"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown lora config fields: {sorted(unknown)}")
        if "rank" not in d or "alpha" not in d:
            raise ValueError("lora config needs 'rank' and 'alpha'")
        return cls(
            rank=int(d["rank"]),
            alpha=float(d["alpha"]),
            dropout=float(d.get("dropout", 0.0)),
            bias=str(d.get("bias", "none")),
            seed=int(d.get("seed", 0)),
            target_patterns=tuple(d.get("target_patterns", DEFAULT_TARGET_PATTERNS)),
        )

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "seed": self.seed,
            "target_patterns": list(self.target_patterns),
            "scaling": self.scaling,
        }

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


class LoRALinear(nn.Linear):
    """`y = W x + b + scaling · B A x`, with `W` frozen and shared with the base."""

    def __init__(self, base: nn.Linear, cfg: LoRAConfig,
                 generator: torch.Generator) -> None:
        # NOT nn.Linear.__init__: that allocates a weight we would discard.
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight                     # shared, not copied
        self.register_parameter("bias", base.bias)    # accepts None
        self.rank = cfg.rank
        self.alpha = cfg.alpha
        self.scaling = cfg.scaling
        self.lora_dropout = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity())
        w = base.weight
        a = torch.empty(cfg.rank, self.in_features, dtype=w.dtype, device=w.device)
        bound = 1.0 / math.sqrt(self.in_features)
        # Draw on CPU: a CUDA generator would make the values device-dependent,
        # and the initialization must be reproducible on any hardware (P8.1).
        a.copy_(torch.empty(cfg.rank, self.in_features, dtype=torch.float32)
                .uniform_(-bound, bound, generator=generator).to(w.dtype))
        self.lora_A = nn.Parameter(a)
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, cfg.rank, dtype=w.dtype, device=w.device))
        self.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + delta * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """`W + scaling · B A`, as a new tensor. The live weight is untouched."""
        delta = (self.lora_B.detach().float() @ self.lora_A.detach().float())
        return (self.weight.detach().float() + self.scaling * delta).to(
            self.weight.dtype)

    @torch.no_grad()
    def delta_norm(self) -> float:
        """Frobenius norm of the merged delta, for the parameter-movement report."""
        delta = self.lora_B.detach().float() @ self.lora_A.detach().float()
        return float((self.scaling * delta).norm())

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, lora_rank={self.rank}, "
                f"lora_alpha={self.alpha}")


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> dict[str, LoRALinear]:
    """Wrap every matching `nn.Linear` in place; returns {qualified name: module}.

    Modules are wrapped in sorted-name order with a single seeded generator, so
    the whole adapter set is a pure function of (cfg.seed, rank, architecture).
    """
    regexes = [re.compile(p) for p in cfg.target_patterns]
    targets = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and any(r.search(name) for r in regexes)
    ]
    if not targets:
        raise ValueError(
            f"lora.target_patterns {list(cfg.target_patterns)} matched no nn.Linear")
    generator = torch.Generator().manual_seed(cfg.seed)
    wrapped: dict[str, LoRALinear] = {}
    modules = dict(model.named_modules())
    for name in sorted(targets):
        parent_name, _, attr = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        base = getattr(parent, attr)
        if isinstance(base, LoRALinear):
            raise ValueError(f"{name} already has LoRA applied")
        layer = LoRALinear(base, cfg, generator)
        setattr(parent, attr, layer)
        wrapped[name] = layer
    return wrapped


def merged_state_dict(model: nn.Module,
                      modules: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    """The model's state dict with deltas folded in and every LoRA key removed.

    The result is indistinguishable from a checkpoint of a model that never had
    LoRA: same keys, same shapes, standard Hugging Face naming.
    """
    sd = model.state_dict()
    out = {}
    for key, tensor in sd.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        out[key] = tensor
    for name, module in modules.items():
        key = f"{name}.weight"
        if key not in out:
            raise KeyError(f"{key} missing from state_dict; cannot merge {name}")
        out[key] = module.merged_weight()
    return out


def lora_and_base_tensors(modules: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    """Everything needed to reconstruct the *unmerged* training state exactly.

    The frozen base weight is stored explicitly rather than recovered from the
    merged checkpoint by subtracting the delta: `(w + d) - d` is not exactly `w`
    in floating point, and an inexact resume is not a resume.
    """
    out = {}
    for name, module in sorted(modules.items()):
        out[f"lora_A::{name}"] = module.lora_A.detach().cpu().contiguous()
        out[f"lora_B::{name}"] = module.lora_B.detach().cpu().contiguous()
        out[f"base::{name}.weight"] = module.weight.detach().cpu().contiguous()
    return out


@torch.no_grad()
def load_lora_and_base_(modules: dict[str, LoRALinear],
                        tensors: dict[str, torch.Tensor]) -> None:
    """Restore base and LoRA tensors in place, preserving parameter identity."""
    expected = set(lora_and_base_tensors(modules))
    got = set(tensors)
    if expected != got:
        missing, extra = sorted(expected - got), sorted(got - expected)
        raise ValueError(
            f"lora state mismatch; missing={missing[:4]} unexpected={extra[:4]}")
    for name, module in modules.items():
        module.lora_A.copy_(tensors[f"lora_A::{name}"].to(module.lora_A.device))
        module.lora_B.copy_(tensors[f"lora_B::{name}"].to(module.lora_B.device))
        module.weight.copy_(
            tensors[f"base::{name}.weight"].to(module.weight.device))


def lora_report(modules: dict[str, LoRALinear], cfg: LoRAConfig) -> dict:
    """Counts and identity for the run manifest and the freeze report."""
    n = sum(m.lora_A.numel() + m.lora_B.numel() for m in modules.values())
    return {
        "lora_config": cfg.to_dict(),
        "lora_modules": sorted(modules),
        "n_lora_modules": len(modules),
        "lora_trainable_params": n,
    }
