"""Exact parameter counts and analytic per-token FLOPs for dense Qwen3-family decoders.

Covers the standard block and the projected-residual block
(`aadistill.initialization.adapters.projected_residual`): a geometry that names an
``internal_size`` has a projection per layer (even at ``internal_size ==
hidden_size``); one without it is the standard block.
Every number is arithmetic over the geometry; nothing here is a measurement,
and nothing names a particular model (AGENTS.md P3).

Conventions, stated once:

* a matrix multiply ``[n, k] @ [k, m]`` costs ``2 n k m`` FLOPs (multiply and
  add both counted), the convention of Kaplan et al. 2020 and Chinchilla;
* attention scores and the value mix cost ``2 * 2 * n_q * head_dim * c`` per
  token at context ``c`` (``QK^T`` and ``AV``); a causal sequence of length
  ``L`` has mean context ``(L + 1) / 2``;
* norms, RoPE, softmax and the SwiGLU product are elementwise and excluded
  (well under 1% at these sizes);
* the embedding lookup is a gather and costs nothing; the tied output head is a
  full ``hidden x vocab`` matmul;
* training FLOPs: a linear layer's backward is ``2x`` its forward when its weight
  trains (input gradient + weight gradient) and ``1x`` when it is frozen but a
  trainable layer sits below it (input gradient only); gradient checkpointing
  recomputes every block's forward once more.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "parameter_counts",
    "forward_flops_per_token",
    "training_flops_per_token",
    "kv_cache_bytes_per_token",
]

STRUCTURAL = ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size",
              "num_attention_heads", "num_key_value_heads", "head_dim", "tie_word_embeddings")


def _geometry(geom: Mapping[str, Any]) -> dict:
    missing = [k for k in STRUCTURAL if k not in geom]
    if missing:
        raise ValueError(f"geometry is missing {missing}")
    g = {k: geom[k] for k in STRUCTURAL}
    g["projected"] = geom.get("internal_size") is not None
    g["internal_size"] = int(geom["internal_size"]) if g["projected"] else int(geom["hidden_size"])
    if g["projected"] and g["internal_size"] > g["hidden_size"]:
        raise ValueError("internal_size cannot exceed hidden_size")
    return g


def parameter_counts(geom: Mapping[str, Any]) -> dict:
    """Exact parameter counts by component, tied weights counted once."""
    g = _geometry(geom)
    d, r, hd = g["hidden_size"], g["internal_size"], g["head_dim"]
    q, kv = g["num_attention_heads"] * hd, g["num_key_value_heads"] * hd
    projection = d * r if g["projected"] else 0
    attention = r * q + 2 * r * kv + q * r + 2 * hd
    ffn = 3 * r * g["intermediate_size"]
    norms = 2 * d
    per_layer = projection + attention + ffn + norms
    embedding = g["vocab_size"] * d
    head = 0 if g["tie_word_embeddings"] else g["vocab_size"] * d
    blocks = per_layer * g["num_hidden_layers"]
    total = embedding + head + blocks + d
    return {
        "embedding": embedding, "output_head_untied": head, "final_norm": d,
        "per_layer": {"projection": projection, "attention": attention, "ffn": ffn,
                      "norms": norms, "total": per_layer},
        "blocks": blocks, "total": total,
        "non_embedding": total - embedding - head,
    }


def forward_flops_per_token(geom: Mapping[str, Any], context: float) -> dict:
    """Forward FLOPs for one token attending to ``context`` positions."""
    g = _geometry(geom)
    d, r, hd = g["hidden_size"], g["internal_size"], g["head_dim"]
    q, kv = g["num_attention_heads"] * hd, g["num_key_value_heads"] * hd
    n = g["num_hidden_layers"]
    projection = (2 * 2 * 2 * d * r) if g["projected"] else 0     # down + up, attn and ffn
    linear_attn = 2 * (r * q + 2 * r * kv + q * r)
    scores = 2 * 2 * q * float(context)
    ffn = 2 * 3 * r * g["intermediate_size"]
    per_layer = projection + linear_attn + scores + ffn
    head = 2 * d * g["vocab_size"]
    return {
        "per_layer": {"projection": projection, "attention_linear": linear_attn,
                      "attention_scores": scores, "ffn": ffn, "total": per_layer},
        "blocks": per_layer * n, "output_head": head, "total": per_layer * n + head,
    }


def training_flops_per_token(geom: Mapping[str, Any], context: float, *,
                             train_blocks: bool = True, train_embedding: bool = True,
                             gradient_checkpointing: bool = True) -> dict:
    """Forward + backward (+ recompute) FLOPs per supervised position.

    The tied embedding and output head share one weight, so ``train_embedding``
    decides whether the head's weight gradient is computed.
    """
    fwd = forward_flops_per_token(geom, context)
    blocks, head = fwd["blocks"], fwd["output_head"]
    backward_blocks = (2 if train_blocks else 1) * blocks
    backward_head = (2 if train_embedding else 1) * head
    recompute = blocks if gradient_checkpointing else 0.0
    total = blocks + head + backward_blocks + backward_head + recompute
    return {"forward": blocks + head, "backward": backward_blocks + backward_head,
            "recompute": recompute, "total": total}


def kv_cache_bytes_per_token(geom: Mapping[str, Any], bytes_per_value: int = 2) -> int:
    """K and V for every layer, at the attention width (unchanged by a projection)."""
    g = _geometry(geom)
    return 2 * g["num_hidden_layers"] * g["num_key_value_heads"] * g["head_dim"] * bytes_per_value
