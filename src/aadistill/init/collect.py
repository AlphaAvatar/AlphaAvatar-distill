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

Sequences are processed one at a time (batch size 1) so no padding-mask
handling can silently corrupt the statistics. Throughput on this stage is
dominated by the teacher forward pass, not by batching.
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
        layers = _decoder_layers(model)
        self.num_layers = len(layers)
        self.hidden_size = model.config.hidden_size
        self.intermediate_size = model.config.intermediate_size
        self.vocab_size = model.config.vocab_size

        # transformers hidden_states tuple: embedding output (= layer 0 input),
        # inputs of layers 1..N-1, then the final-norm output — N+1 points.
        n_points = self.num_layers + 1
        d, i = self.hidden_size, self.intermediate_size
        self.res_count = 0
        self.res_sum = torch.zeros(n_points, d, dtype=torch.float64)
        self.res_sqsum = torch.zeros(n_points, d, d, dtype=torch.float64)
        self.ffn_abs_sum = torch.zeros(self.num_layers, i, dtype=torch.float64)
        self.ffn_sq_sum = torch.zeros(self.num_layers, i, dtype=torch.float64)
        self.token_counts = torch.zeros(self.vocab_size, dtype=torch.int64)

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
            a = args[0].detach().reshape(-1, self.intermediate_size).to(torch.float64)
            self.ffn_abs_sum[idx] += a.abs().sum(0)
            self.ffn_sq_sum[idx] += (a * a).sum(0)
        return hook

    @torch.no_grad()
    def process(self, input_ids: torch.Tensor) -> int:
        """Accumulate statistics from one unpadded sequence of shape (1, T)."""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected shape (1, T), got {tuple(input_ids.shape)}")
        out = self.model(input_ids.to(self.model.device), output_hidden_states=True)
        hs = out.hidden_states
        assert len(hs) == self.res_sum.shape[0], (
            f"Expected {self.res_sum.shape[0]} hidden state points, got {len(hs)}"
        )
        n_tokens = input_ids.shape[1]
        for point, h in enumerate(hs):
            x = h[0].to(torch.float64)
            self.res_sum[point] += x.sum(0)
            self.res_sqsum[point] += x.T @ x
        self.res_count += n_tokens
        self.token_counts += torch.bincount(
            input_ids[0].cpu(), minlength=self.vocab_size
        )
        return n_tokens

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def state(self) -> dict[str, torch.Tensor]:
        return {
            "residual_sum": self.res_sum,
            "residual_sqsum": self.res_sqsum,
            "residual_count": torch.tensor([self.res_count], dtype=torch.int64),
            "ffn_abs_sum": self.ffn_abs_sum,
            "ffn_sq_sum": self.ffn_sq_sum,
            "token_counts": self.token_counts,
        }

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
