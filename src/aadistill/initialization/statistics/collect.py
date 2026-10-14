"""Stage 0 activation statistics collection.

Instead of caching raw activations, we accumulate streaming sufficient
statistics, which keeps the cache small and within a fixed budget:

- residual stream, per collection point (embedding output, each decoder layer
  input boundary, final norm output): token count, sum vector, and uncentered
  second moment ``X^T X``. These are exactly the sufficient statistics for the
  per-layer (grouped) activation PCA used by Stage 1 initialization.
- FFN intermediate, per layer: per-neuron ``sum |a|`` and ``sum a^2`` for
  activation-importance top-k neuron selection.
- token frequency counts for frequency-weighted embedding PCA.

Accumulation is float64 because residual streams contain large-magnitude
outlier dimensions; float32 accumulation would lose precision in the
``E[xx^T] - mu mu^T`` centering step downstream.

Sequences may be processed one at a time (``process``) or in padded micro-batches
(``process_batch``). Both reduce over **real token positions only**: a padded
position never reaches a sum, a second moment or a token count. ``process`` is
the reference path and is implemented as a one-row batch, so there is one
accumulation rule rather than two that could drift apart; when nothing is padded
the mask is skipped entirely and the arithmetic is the arithmetic this collector
has always performed.

Both entry points also take an optional ``active_mask``: the token positions a
:class:`~aadistill.initialization.scoring.positions.ScoringPositionPolicy` says
this statistic may read. It is combined with the padding mask by ``and``, and the
combined mask drives the same single ``_keep_valid`` used for padding — so
"restrict the statistic to supervised positions" and "keep padding out of the
statistic" are one mechanism rather than two, and the all-active case remains
byte-for-byte the collector's historical arithmetic.

**Continuous token weights** arrive as ``token_weights`` instead of
``active_mask`` -- a ``[B, T]`` float64 ``w_t >= 0`` (e.g. the root reference's
confidence times the policy's token weight). Every accumulator becomes
weighted: ``sum_t w_t x_t``, ``sum_t w_t x_t x_t^T``, ``sum_t w_t |a_t|`` and
``sum_t w_t a_t^2``, and the divisor becomes ``sum_t w_t``, carried as the
float64 ``residual_weight`` that :func:`statistics_denominator` prefers over
the integer ``residual_count``. A position with ``w_t = 0`` is dropped like
padding, so ``residual_count`` and ``token_counts`` stay integer counts of the
positions that carried any weight. ``residual_weight`` exists in the state
ONLY when the pass was weighted, so an unweighted state is exactly the state
this collector has always produced, and one collector refuses to mix the two.
"""

from __future__ import annotations

import torch

#: Reconciling a scoring policy's position mask with the `[B, T_max]` shape of
#: the tokens being processed is BATCH knowledge, so it lives beside `ItemBatch`
#: and both collectors import the one implementation.
from aadistill.initialization.calibration.batching import (
    active_rows,
    active_weights,
)


class ActivationStatsCollector:
    """Streaming residual/FFN sufficient statistics.

    **Family knowledge belongs to the adapter, not here.** This used to walk
    `model.model.layers` and reach into `layer.mlp.down_proj`, which is the one
    module tree Qwen3 happens to have — so a family with a different FFN
    attribute name, or an MoE block with several down projections, could not use
    this collector without editing it. It now takes the ordered FFN-output
    projections from its caller, exactly as
    `AttentionHeadStatsCollector` takes the attention-output ones, and
    `ArchitectureAdapter.stats_collector` resolves them by ROLE
    (`stream_out_projections(block)["ffn_out"]`). Adding an architecture is then
    writing an adapter rather than editing statistics code.

    `ffn_out_projections` is optional only so that existing callers constructing
    `ActivationStatsCollector(model)` keep working; when it is omitted the
    adapter for the model's family is resolved and asked, which is the same
    answer by the same route.
    """

    def __init__(self, model, ffn_out_projections=None):
        self.model = model
        # Read from the weights, never from a config field or a caller's intent.
        self.device = next(model.parameters()).device
        modules = (list(ffn_out_projections) if ffn_out_projections is not None
                   else _ffn_out_projections_via_adapter(model))
        if not modules:
            raise ValueError(
                "no FFN-output projections were supplied and the adapter "
                "resolved none; this collector does not discover them")
        self.num_layers = len(modules)
        self.hidden_size = model.config.hidden_size
        self.intermediate_size = model.config.intermediate_size
        self.vocab_size = model.config.vocab_size

        # transformers hidden_states tuple: embedding output (= layer 0 input),
        # inputs of layers 1..N-1, then the final-norm output — N+1 points.
        n_points = self.num_layers + 1
        d, i = self.hidden_size, self.intermediate_size
        # ACCUMULATE ON THE MODEL'S DEVICE. The hooks receive activations from
        # the model, so an accumulator anywhere else is a cross-device add: that
        # is what killed a paid session in `ffn_abs_sum[idx] += ...`, and it
        # had never fired because this collector's only previous execution was
        # the Stage-0 regeneration on the CPU-only dev box, where the model and
        # the accumulators coincide.
        #
        # Host accumulation would be the other way to make the devices agree,
        # and it is the wrong one: `x.T @ x` is (H, H) float64 per collection
        # point, so a 4B parent would push ~1.85 GiB across PCIe per calibration
        # item. Accumulating here and transferring once in `state()` moves 1.81
        # GiB in total instead. See `autoinit.device` for the contract.
        dev = self.device
        self.res_count = 0
        self.res_sum = torch.zeros(n_points, d, dtype=torch.float64, device=dev)
        self.res_sqsum = torch.zeros(n_points, d, d, dtype=torch.float64, device=dev)
        self.ffn_abs_sum = torch.zeros(self.num_layers, i, dtype=torch.float64,
                                       device=dev)
        self.ffn_sq_sum = torch.zeros(self.num_layers, i, dtype=torch.float64,
                                      device=dev)
        self.token_counts = torch.zeros(self.vocab_size, dtype=torch.int64,
                                        device=dev)

        #: Set for the duration of one forward when that forward is padded, and
        #: `None` otherwise. Initialized here so a hook that fires outside
        #: `_accumulate` — a caller running the model directly while the hooks
        #: are attached — reads "nothing is padded" rather than an AttributeError.
        self._valid_mask = None
        #: The kept rows' token weights for the forward in flight, aligned with
        #: `_keep_valid`'s output; `None` on an unweighted pass.
        self._row_weights = None
        #: `None` until the first pass, then whether passes are weighted. A
        #: collector refuses to mix the two: one denominator cannot be both a
        #: count and a weight sum.
        self._weighted: bool | None = None
        self.res_weight = torch.zeros((), dtype=torch.float64, device=dev)

        self._hooks = []
        for idx, projection in enumerate(modules):
            if not hasattr(projection, "register_forward_pre_hook"):
                raise ValueError(
                    f"ffn_out_projections[{idx}] is {type(projection).__name__}, "
                    "which cannot be hooked; expected a module with "
                    "register_forward_pre_hook")
            self._hooks.append(
                projection.register_forward_pre_hook(self._make_ffn_hook(idx))
            )

    def _make_ffn_hook(self, idx: int):
        def hook(_module, args):
            a = args[0].detach().reshape(-1, self.intermediate_size)
            a = self._keep_valid(a).to(torch.float64)
            w = self._row_weights
            if w is None:
                self.ffn_abs_sum[idx] += a.abs().sum(0)
                self.ffn_sq_sum[idx] += (a * a).sum(0)
            else:
                self.ffn_abs_sum[idx] += w.to(a.device) @ a.abs()
                self.ffn_sq_sum[idx] += w.to(a.device) @ (a * a)
        return hook

    def _keep_valid(self, flat: torch.Tensor) -> torch.Tensor:
        """Drop padded rows from a ``[B*T, ...]`` tensor.

        Returns the tensor UNTOUCHED when nothing is padded — which is every
        ``process`` call, and every batch whose items happen to share a length.
        That is not only a saving: it means the unbatched reference path
        performs exactly the operations it performed before this collector
        learned to batch, so its accumulators are bit-identical rather than
        merely equivalent.
        """
        mask = self._valid_mask
        if mask is None:
            return flat
        if mask.shape[0] != flat.shape[0]:
            raise ValueError(
                f"activation has {flat.shape[0]} rows but the batch mask "
                f"covers {mask.shape[0]}; the hooked module did not receive "
                "the batch this collector is processing")
        return flat[mask]

    @torch.no_grad()
    def process(self, input_ids: torch.Tensor, *,
                active_mask: torch.Tensor | None = None,
                token_weights: torch.Tensor | None = None) -> int:
        """Accumulate statistics from one unpadded sequence of shape (1, T).

        The reference path. Kept as its own entry point — every existing caller
        uses it — and implemented through the batched one so the two cannot
        diverge. ``active_mask`` is an optional ``[1, T]`` or ``[T]`` bool over
        the token positions the scoring policy admits; ``token_weights`` the
        continuous alternative (see the module docstring).
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected shape (1, T), got {tuple(input_ids.shape)}")
        _one_restriction(active_mask, token_weights)
        return self._accumulate(input_ids.to(self.model.device),
                                attention_mask=None,
                                active_mask=active_rows(active_mask, input_ids.shape,
                                                     self.model.device),
                                token_weights=active_weights(
                                    token_weights, input_ids.shape,
                                    self.model.device))

    @torch.no_grad()
    def process_batch(self, batch, *,
                      active_mask: torch.Tensor | None = None,
                      token_weights: torch.Tensor | None = None) -> int:
        """Accumulate statistics from a padded :class:`ItemBatch`.

        The sufficient statistics are identical in definition to the per-item
        path: `sum_t x`, `sum_t x x^T`, `sum_t |a|`, `sum_t a^2` and the token
        histogram, all over the batch's **real** tokens. Only the order in which
        the accelerator adds them moves.

        ``active_mask`` is an optional ``[B, T_max]`` bool over the token
        positions the scoring policy admits, in the batch's ROW ORDER — which
        under a reordering packing policy is not the mixture's order, so the
        caller builds it from ``PackedBatch.original_indices``.
        ``token_weights`` is the continuous ``[B, T_max]`` alternative, in the
        same row order.
        """
        _one_restriction(active_mask, token_weights)
        return self._accumulate(batch.input_ids.to(self.model.device),
                                attention_mask=batch.attention_mask.to(
                                    self.model.device),
                                active_mask=active_rows(active_mask,
                                                     batch.input_ids.shape,
                                                     self.model.device),
                                token_weights=active_weights(
                                    token_weights, batch.input_ids.shape,
                                    self.model.device))

    def _accumulate(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor | None,
                    active_mask: torch.Tensor | None = None,
                    token_weights: torch.Tensor | None = None) -> int:
        weighted = token_weights is not None
        if self._weighted is None:
            self._weighted = weighted
        elif self._weighted != weighted:
            raise ValueError(
                "this collector has accumulated "
                f"{'weighted' if self._weighted else 'unweighted'} passes and "
                f"was handed {'a weighted' if weighted else 'an unweighted'} one; "
                "one denominator cannot be both a count and a weight sum")
        #: ONE mask, from two sources. Padding says which positions exist; the
        #: scoring policy says which existing positions this statistic may read.
        #: Combining them here means every accumulator below — the residual sums,
        #: the FFN hooks, the token histogram and the token count — is restricted
        #: by both without any of them knowing there are two reasons.
        valid = None
        if attention_mask is not None and bool((attention_mask == 0).any()):
            valid = attention_mask.reshape(-1).bool()
        if active_mask is not None and not bool(active_mask.all()):
            flat = active_mask.reshape(-1).bool()
            valid = flat if valid is None else (valid & flat)
        #: A zero weight is dropped like padding, so the integer count and the
        #: token histogram count exactly the positions that carried weight.
        if weighted:
            positive = token_weights.reshape(-1) > 0
            if not bool(positive.all()):
                valid = positive if valid is None else (valid & positive)
        #: Read by the FFN hooks during the forward below, and cleared after it.
        #: `None` means "every position counts", which is what keeps the
        #: reference path free of an indexing op it never had.
        self._valid_mask = valid
        if weighted:
            flat_w = token_weights.reshape(-1)
            self._row_weights = flat_w if valid is None else flat_w[valid]
        try:
            out = self.model(
                input_ids,
                output_hidden_states=True,
                **({"attention_mask": attention_mask}
                   if attention_mask is not None else {}))
            hs = out.hidden_states
            assert len(hs) == self.res_sum.shape[0], (
                f"Expected {self.res_sum.shape[0]} hidden state points, got {len(hs)}"
            )
            w = self._row_weights
            for point, h in enumerate(hs):
                x = self._keep_valid(h.reshape(-1, self.hidden_size)).to(torch.float64)
                if w is None:
                    self.res_sum[point] += x.sum(0)
                    self.res_sqsum[point] += x.T @ x
                else:
                    wx = x * w.to(x.device)[:, None]
                    self.res_sum[point] += wx.sum(0)
                    self.res_sqsum[point] += wx.T @ x
            if w is not None:
                self.res_weight += w.sum().to(self.res_weight.device)
            flat_ids = input_ids.reshape(-1)
            if self._valid_mask is not None:
                flat_ids = flat_ids[self._valid_mask]
            n_tokens = int(flat_ids.shape[0])
            self.res_count += n_tokens
            # On the accumulator's device: `bincount` on the host would produce a
            # host tensor and the `+=` would be the same cross-device add again.
            self.token_counts += torch.bincount(
                flat_ids.to(self.token_counts.device), minlength=self.vocab_size
            )
            return n_tokens
        finally:
            self._valid_mask = None
            self._row_weights = None

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def state(self) -> dict[str, torch.Tensor]:
        """The statistics, ON THE HOST. This is the transfer boundary.

        The persistent cache holds one of these and it is 1.81 GiB at a 4B
        parent, so it does not live in VRAM between operator invocations. A
        consumer that needs it for compute moves a working copy back with
        `aadistill.initialization.device.stats_to`, for the duration of one call.
        """
        state = {
            "residual_sum": self._to_host(self.res_sum),
            "residual_sqsum": self._to_host(self.res_sqsum),
            "residual_count": torch.tensor([self.res_count], dtype=torch.int64),
            "ffn_abs_sum": self._to_host(self.ffn_abs_sum),
            "ffn_sq_sum": self._to_host(self.ffn_sq_sum),
            "token_counts": self._to_host(self.token_counts),
        }
        #: ONLY WHEN WEIGHTED, so an unweighted state is key-for-key the state
        #: every committed artifact was built from.
        if self._weighted:
            state[RESIDUAL_WEIGHT] = self._to_host(self.res_weight).reshape(1)
        return state

    @staticmethod
    def _to_host(t: torch.Tensor) -> torch.Tensor:
        """The transfer, as a named seam.

        `.cpu()` inline would be untestable on a CPU-only box: it is a no-op
        there and even returns `self`, so a regression that deleted it would
        still pass. Routing every accumulator through one method lets a test
        assert the boundary was crossed rather than assert `device == cpu`,
        which is true either way.
        """
        return t.to("cpu")

    def save(self, path: str) -> dict:
        from safetensors.torch import save_file

        state = self.state()
        save_file(state, path)
        return {
            "tokens_processed": self.res_count,
            "tensors": {k: [list(v.shape), str(v.dtype)] for k, v in state.items()},
        }


def _one_restriction(active_mask, token_weights) -> None:
    if active_mask is not None and token_weights is not None:
        raise ValueError(
            "a pass is restricted by a binary `active_mask` OR weighted by "
            "`token_weights`, not both; fold the mask into the weights")


#: The weighted denominator's state key. Present only in a weighted state.
RESIDUAL_WEIGHT = "residual_weight"


def statistics_denominator(state: dict[str, torch.Tensor],
                           count_key: str = "residual_count",
                           weight_key: str = RESIDUAL_WEIGHT) -> float | int:
    """What a mean over this state divides by: ``sum_t w_t`` when the pass was
    weighted, else the integer position count -- the one place the choice is
    made, so the three divisor call sites cannot disagree."""
    #: No tensor factory here: the state's own tensors are read in place, so
    #: nothing is allocated off the statistics' device.
    if weight_key in state:
        return float(state[weight_key].reshape(-1)[0])
    return int(state[count_key].reshape(-1)[0])


def _ffn_out_projections_via_adapter(model):
    """The FFN-output projection of every block, in block order, BY ROLE.

    Imported inside the function because `specs.arch` imports nothing from this
    module and the reverse edge would be a cycle at import time.
    """
    from aadistill.initialization.specs.arch import adapter_for_config

    adapter = adapter_for_config(model.config)
    return [adapter.stream_out_projections(block)["ffn_out"]
            for block in adapter.blocks(model)]


def residual_covariance(state: dict[str, torch.Tensor], point: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and centered covariance at one residual collection point.

    This is the Stage 1 projection entry point; it exists here so the Stage 0
    validation gate can prove the cache is consumable by a projection dry run.
    """
    count = int(state["residual_count"][0])
    if count < 2:
        raise ValueError(f"Need at least 2 tokens, have {count}")
    n = statistics_denominator(state)
    mean = state["residual_sum"][point] / n
    cov = state["residual_sqsum"][point] / n - torch.outer(mean, mean)
    return mean, cov
