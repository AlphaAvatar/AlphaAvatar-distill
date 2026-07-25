"""INT8 weight fake-quantization for deployment-matching evaluation (P9).

Deployment target (2026-07-13 precision policy): INT8 weights, activation
quantization deferred to Stage 6. This module simulates that target inside
the normal eval graph: selected weight matrices are replaced in-place by
``dequant(quant(W))`` with per-output-channel symmetric scales, so an
ordinary forward pass measures the quality an INT8-weight deployment would
see. Real INT8 kernels accumulate in int32/fp32; fake-quant runs the matmul
in the model dtype instead, which differs only by matmul rounding, not by
the quantization grid.

Quantization math, computed in fp32 per output channel (row) ``i``:

    scale_i = max(|W[i, :]|) / 127          (scale_i = 1.0 for all-zero rows)
    W'[i, :] = clamp(round(W[i, :] / scale_i), -127, 127) * scale_i

then cast back to the parameter's original dtype.

Scopes:

- ``"decoder"``: every ``nn.Linear`` under ``model.layers.`` (attention
  q/k/v/o and FFN gate/up/down) — the matmuls every INT8 runtime quantizes.
- ``"all"``: every ``nn.Linear`` in the model, i.e. decoder + lm_head. With
  tied embeddings (our student) the shared matrix is quantized once, so
  embedding lookups see the same INT8 grid as the head projection — matching
  runtimes that store one quantized copy of the tied matrix.

Shared/tied parameters are quantized and counted exactly once.
"""

from __future__ import annotations

import torch
from torch import nn

SCOPES = ("all", "decoder")
QMAX = 127


def int8_fake_quant_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Per-row symmetric INT8 fake-quant; returns (new tensor, error stats)."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    w = weight.detach().float()
    scale = w.abs().amax(dim=1, keepdim=True) / QMAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(w / scale), -QMAX, QMAX)
    dq = (q * scale).to(weight.dtype)
    err = (dq.float() - w).norm() / w.norm().clamp_min(1e-12)
    return dq, {"rel_fro_err": err.item(), "max_scale": scale.max().item()}


def int8_fake_quantize_(model: nn.Module, scope: str = "all") -> dict:
    """Fake-quantize matching Linear weights in-place; returns a summary."""
    if scope not in SCOPES:
        raise ValueError(f"unknown fake-quant scope {scope!r}, expected one of {SCOPES}")
    seen: set[int] = set()
    n_params = 0
    per_module: dict[str, float] = {}
    max_scale = 0.0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if scope == "decoder" and ".layers." not in name:
            continue
        if id(module.weight) in seen:
            continue
        seen.add(id(module.weight))
        dq, stats = int8_fake_quant_tensor(module.weight)
        with torch.no_grad():
            module.weight.copy_(dq)
        n_params += module.weight.numel()
        per_module[name] = round(stats["rel_fro_err"], 6)
        max_scale = max(max_scale, stats["max_scale"])
    if not per_module:
        raise RuntimeError(f"fake-quant scope {scope!r} matched no Linear modules")
    errs = list(per_module.values())
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    return {
        "scheme": "int8_weight_perchannel_symmetric",
        "scope": scope,
        "n_linear_quantized": len(per_module),
        "n_params_quantized": n_params,
        "tied_embeddings": bool(tied),
        "mean_rel_fro_err": round(sum(errs) / len(errs), 6),
        "max_rel_fro_err": round(max(errs), 6),
        "max_scale": max_scale,
    }
