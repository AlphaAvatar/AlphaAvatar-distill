"""Per-query-head second moments of the attention output, streamed.

`attention.activation_importance_v1` ranks a query head by the energy it
actually writes into the residual stream:

    z_h(t) = W_o,h @ a_h(t)          score_h = mean_t ||z_h(t)||^2

where `a_h(t)` is head h's own slice of the concatenated attention output — the
tensor that `o_proj` consumes — and `W_o,h` is o_proj's column block for that
head.

Retaining `a_h(t)` for every calibration token is unnecessary and would be far
larger than the model. Expanding the score gives an exact sufficient statistic:

    mean_t ||W_o,h a_h||^2 = mean_t a_h^T (W_o,h^T W_o,h) a_h
                           = <W_o,h^T W_o,h , mean_t a_h a_h^T>_F

so accumulating the per-head second moment `M_h = sum_t a_h a_h^T` (head_dim x
head_dim) and a token count is **exact**, not an approximation. The operator
contracts `M_h` against the weights at selection time.

Size: `n_layers * n_heads * head_dim^2` float64. At the Phase-C1 parent
(28 layers, 32 heads, head_dim 128) that is 117 MiB, against the 1.85 GiB the
residual second moments already cost — so this adds a small fraction of an
existing budget rather than a new one.

**Accumulate on the model's device.** The existing residual/FFN collector carries
a comment earned the hard way: accumulating anywhere else is a cross-device add,
and that is what killed Phase-A attempt 7. The same rule applies here, and
`state()` moves the result to the host once, at the end.

Hooking `o_proj`'s *input* is deliberate: it already holds the concatenated
per-head outputs, so nothing about the attention kernel, the GQA grouping or the
RoPE basis has to change to observe it.
"""

from __future__ import annotations

import torch


def _decoder_layers(model):
    """The decoder blocks, without assuming a wrapper class."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("model has no .model.layers; unsupported architecture")
    return layers


class AttentionHeadStatsCollector:
    """Streaming per-head second moments of each block's attention output.

    Mirrors the shape of `init.collect.ActivationStatsCollector`: construct,
    `process(input_ids)` repeatedly, `close()`, then `state()`.
    """

    def __init__(self, model, *, num_heads: int, head_dim: int):
        self.model = model
        self.device = next(model.parameters()).device
        layers = _decoder_layers(model)
        self.num_layers = len(layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")

        # (layers, heads, head_dim, head_dim), float64, on the model's device.
        self.head_sqsum = torch.zeros(
            self.num_layers, self.num_heads, self.head_dim, self.head_dim,
            dtype=torch.float64, device=self.device)
        self.token_count = 0

        self._hooks = []
        for idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            o_proj = getattr(attn, "o_proj", None) if attn is not None else None
            if o_proj is None:
                raise ValueError(
                    f"Layer {idx} has no self_attn.o_proj; unsupported architecture")
            self._hooks.append(
                o_proj.register_forward_pre_hook(self._make_hook(idx)))

    def _make_hook(self, idx: int):
        def hook(_module, args):
            x = args[0]
            if x.device != self.head_sqsum.device:      # never add across devices
                x = x.to(self.head_sqsum.device)
            # (..., n_heads*head_dim) -> (tokens, n_heads, head_dim)
            flat = x.reshape(-1, x.shape[-1])
            expected = self.num_heads * self.head_dim
            if flat.shape[-1] != expected:
                raise ValueError(
                    f"layer {idx}: o_proj input width {flat.shape[-1]} != "
                    f"num_heads*head_dim ({expected}); the head layout this "
                    "collector assumes does not hold for this model")
            a = flat.to(torch.float64).reshape(-1, self.num_heads, self.head_dim)
            # per head: sum_t a_h a_h^T  ->  (heads, head_dim, head_dim)
            self.head_sqsum[idx] += torch.einsum("thi,thj->hij", a, a)
            if idx == 0:
                self.token_count += a.shape[0]
        return hook

    @torch.no_grad()
    def process(self, input_ids: torch.Tensor) -> None:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        self.model(input_ids.to(self.device))

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def state(self) -> dict[str, torch.Tensor]:
        """Host-resident sufficient statistics. Moved once, at the end."""
        if self.token_count == 0:
            raise ValueError(
                "no tokens were processed; refusing to return an all-zero "
                "attention statistic that would rank every head identically")
        return {
            "attn_head_sqsum": self.head_sqsum.to("cpu"),
            "attn_token_count": torch.tensor(self.token_count, dtype=torch.int64),
        }


def head_write_energy(state: dict[str, torch.Tensor], layer: int,
                      o_proj_weight: torch.Tensor, num_heads: int,
                      head_dim: int) -> torch.Tensor:
    """`score_h = mean_t ||W_o,h a_h(t)||^2` for every head of one block.

    Computed exactly from the second moment:
    `<W_o,h^T W_o,h, M_h>_F / n_tokens`.
    """
    m = state["attn_head_sqsum"][layer].to(torch.float64)     # (heads, d, d)
    n = int(state["attn_token_count"])
    w = o_proj_weight.to(torch.float64)                        # (hidden, heads*d)
    if w.shape[-1] != num_heads * head_dim:
        raise ValueError(
            f"o_proj input width {w.shape[-1]} != num_heads*head_dim "
            f"({num_heads * head_dim})")
    scores = torch.empty(num_heads, dtype=torch.float64)
    for h in range(num_heads):
        wh = w[:, h * head_dim:(h + 1) * head_dim]             # (hidden, d)
        gram = wh.T @ wh                                       # (d, d)
        scores[h] = (gram * m[h]).sum() / n
    return scores
