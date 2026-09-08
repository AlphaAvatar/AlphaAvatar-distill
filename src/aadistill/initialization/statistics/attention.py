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

Hooking the attention-output projection's *input* is deliberate: it already holds
the concatenated per-head outputs, so nothing about the attention kernel, the GQA
grouping or the RoPE basis has to change to observe it.

**Which modules those are is not this file's business.** The collector is given
an ordered sequence of projection modules and knows only: the model to run, where
to hook, `num_heads`, and `head_dim`. It does not walk `.model.layers` and does
not read `.self_attn.o_proj`. Family module-tree knowledge belongs to
`ArchitectureAdapter` (`blocks()`, `stream_out_projections()`), and a statistics
collector that duplicates it is a second place to update when a family is added
and a silent wrong answer when only one of them is.
"""

from __future__ import annotations

import torch


class AttentionHeadStatsCollector:
    """Streaming per-head second moments of each block's attention output.

    Mirrors the shape of `init.collect.ActivationStatsCollector`: construct,
    `process(input_ids)` repeatedly, `close()`, then `state()`.

    `out_projections` is the ordered sequence of attention-output projection
    modules — one per block, in block order — resolved by the caller's
    `ArchitectureAdapter`. Passing them in rather than discovering them is the
    whole point: this class then contains no family knowledge at all, and a new
    architecture is supported by writing an adapter rather than by editing here.
    """

    def __init__(self, model, out_projections, *, num_heads: int, head_dim: int):
        modules = list(out_projections)
        if not modules:
            raise ValueError(
                "no attention-output projections were supplied; the caller "
                "resolves them from its adapter (`stream_out_projections`) and "
                "this collector cannot discover them")
        for i, m in enumerate(modules):
            if not hasattr(m, "register_forward_pre_hook"):
                raise ValueError(
                    f"out_projections[{i}] is {type(m).__name__}, which cannot be "
                    "hooked; expected a module with register_forward_pre_hook")

        self.model = model
        self.device = next(model.parameters()).device
        self.num_layers = len(modules)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")

        # (layers, heads, head_dim, head_dim), float64, on the model's device.
        self.head_sqsum = torch.zeros(
            self.num_layers, self.num_heads, self.head_dim, self.head_dim,
            dtype=torch.float64, device=self.device)
        self.token_count = 0

        self._hooks = [m.register_forward_pre_hook(self._make_hook(i))
                       for i, m in enumerate(modules)]

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

    def release(self) -> None:
        """Drop the device-resident accumulator once its snapshot has been taken.

        `state()` copies `head_sqsum` to the host; until this is called the
        original stays on the model device, so the working copy the caller then
        builds is a THIRD copy of a `(layers, heads, d, d)` float64 tensor.
        Freeing here keeps at most two alive at once. `state()` afterwards
        raises rather than returning a stale or absent accumulator.
        """
        self.head_sqsum = None

    def state(self) -> dict[str, torch.Tensor]:
        """Host-resident sufficient statistics. Moved once, at the end."""
        if self.head_sqsum is None:
            raise ValueError(
                "the accumulator was released; state() must be called before "
                "release(), and exactly once")
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
    # FAIL CLOSED, and do not repair it here. C1 attempt 9 died on this exact
    # product with the statistics on the host and `o_proj.weight` on cuda:0.
    # Transferring silently would make this function guess which device the
    # caller meant, and hide a caller that forgot to build a working copy; the
    # co-location is the CALLER's contract (`attention_activation.apply` moves
    # the snapshot with `stats_to`), so a mismatch is reported, not absorbed.
    if m.device != w.device:
        raise ValueError(
            f"attention statistics are on {m.device} and o_proj.weight is on "
            f"{w.device}. `head_write_energy` does not transfer: the caller "
            "owns the compute-device working copy — see "
            "`aadistill.initialization.device.stats_to`.")
    # Placed from the tensor it meets. A bare `torch.empty(num_heads,
    # dtype=torch.float64)` defaults to CPU, so on a GPU the very first
    # `scores[h] = ...` assignment is a second cross-device use — latent behind
    # the first one, and invisible on a single-device box.
    #
    # DO NOT "restore" this to a host tensor by citing `autoinit/device.py`'s
    # list of intentional host-only per-head score vectors. Those two —
    # `operators/attention.py` and `init.sandwich.select_q_heads` — are
    # `torch.tensor([float(...), ...])` over comprehensions that have ALREADY
    # reduced each element to a Python float, so their buffers never receive a
    # device tensor. This one is filled with `(gram * m[h]).sum() / n`, a 0-dim
    # DEVICE tensor, which makes it device-coupled under the same contract.
    # Two per-head score vectors, opposite categories: read the store, not the
    # variable name.
    scores = torch.empty(num_heads, dtype=torch.float64, device=w.device)
    for h in range(num_heads):
        wh = w[:, h * head_dim:(h + 1) * head_dim]             # (hidden, d)
        gram = wh.T @ wh                                       # (d, d)
        scores[h] = (gram * m[h]).sum() / n
    # ONE transfer of a `num_heads`-element vector, after the whole vector is
    # built. Every caller ranks, sums and tie-breaks these on the host, and that
    # deterministic selection path is unchanged by this repair.
    return scores.to("cpu")
