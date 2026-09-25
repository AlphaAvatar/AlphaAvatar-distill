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
"""

from __future__ import annotations

import torch


def _decoder_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError(f"Cannot locate decoder layers on {type(model).__name__}")
    return layers


class ActivationStatsCollector:
    def __init__(self, model):
        self.model = model
        # Read from the weights, never from a config field or a caller's intent.
        self.device = next(model.parameters()).device
        layers = _decoder_layers(model)
        self.num_layers = len(layers)
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

        self._hooks = []
        for idx, layer in enumerate(layers):
            down_proj = getattr(getattr(layer, "mlp", None), "down_proj", None)
            if down_proj is None:
                raise ValueError(f"Layer {idx} has no mlp.down_proj; unsupported architecture")
            self._hooks.append(
                down_proj.register_forward_pre_hook(self._make_ffn_hook(idx))
            )

    def _make_ffn_hook(self, idx: int):
        def hook(_module, args):
            a = args[0].detach().reshape(-1, self.intermediate_size)
            a = self._keep_valid(a).to(torch.float64)
            self.ffn_abs_sum[idx] += a.abs().sum(0)
            self.ffn_sq_sum[idx] += (a * a).sum(0)
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
    def process(self, input_ids: torch.Tensor) -> int:
        """Accumulate statistics from one unpadded sequence of shape (1, T).

        The reference path. Kept as its own entry point — every existing caller
        uses it — and implemented through the batched one so the two cannot
        diverge.
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected shape (1, T), got {tuple(input_ids.shape)}")
        return self._accumulate(input_ids.to(self.model.device),
                                attention_mask=None)

    @torch.no_grad()
    def process_batch(self, batch) -> int:
        """Accumulate statistics from a padded :class:`ItemBatch`.

        The sufficient statistics are identical in definition to the per-item
        path: `sum_t x`, `sum_t x x^T`, `sum_t |a|`, `sum_t a^2` and the token
        histogram, all over the batch's **real** tokens. Only the order in which
        the accelerator adds them moves.
        """
        return self._accumulate(batch.input_ids.to(self.model.device),
                                attention_mask=batch.attention_mask.to(
                                    self.model.device))

    def _accumulate(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor | None) -> int:
        padded = (attention_mask is not None
                  and bool((attention_mask == 0).any()))
        #: Read by the FFN hooks during the forward below, and cleared after it.
        #: `None` means "nothing is padded", which is what keeps the reference
        #: path free of an indexing op it never had.
        self._valid_mask = (attention_mask.reshape(-1).bool() if padded else None)
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
            for point, h in enumerate(hs):
                x = self._keep_valid(h.reshape(-1, self.hidden_size)).to(torch.float64)
                self.res_sum[point] += x.sum(0)
                self.res_sqsum[point] += x.T @ x
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
        return {
            "residual_sum": self._to_host(self.res_sum),
            "residual_sqsum": self._to_host(self.res_sqsum),
            "residual_count": torch.tensor([self.res_count], dtype=torch.int64),
            "ffn_abs_sum": self._to_host(self.ffn_abs_sum),
            "ffn_sq_sum": self._to_host(self.ffn_sq_sum),
            "token_counts": self._to_host(self.token_counts),
        }

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


def residual_covariance(state: dict[str, torch.Tensor], point: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and centered covariance at one residual collection point.

    This is the Stage 1 projection entry point; it exists here so the Stage 0
    validation gate can prove the cache is consumable by a projection dry run.
    """
    n = int(state["residual_count"][0])
    if n < 2:
        raise ValueError(f"Need at least 2 tokens, have {n}")
    mean = state["residual_sum"][point] / n
    cov = state["residual_sqsum"][point] / n - torch.outer(mean, mean)
    return mean, cov
