"""A model's output head as a value: logits of every entry, or of chosen ones.

The union-support KL (``reference_student_union_topk_tail_v1``) needs the
REFERENCE's probability of entries the reference sketch does not hold -- the
candidate's own Top-K entries outside the reference's. Re-forwarding the
reference per candidate would cost a transformer forward per candidate; caching
its ``[T, V]`` logits would cost ``O(T*V)``. Neither is needed: the reference's
logits are a ROW-WISE function of its final hidden state, so storing the
``[T, H]`` hidden state and the ``[T]`` log-normalizer lets any entry's log
probability be recovered on demand, exactly.

That factorization is a family fact, so it sits behind an adapter capability
(:data:`~aadistill.initialization.specs.arch.Capability.OUTPUT_HEAD_LINEAR`):
a family declaring it hands back a :class:`LinearOutputHead` describing the
affine map, any logit scale and any final soft-cap. Nothing here knows a
family, a vocabulary size or a hidden size; they are read from the tensors.

**The logit definition is float32, and it is stated.** ``logits =
post(h · Wᵀ + b)`` computed in float32 from the stored hidden state and the
head's own parameters (an exact widening of bf16/fp16), accumulated in
float32. The model's own forward rounds the same product to its dtype; this
definition does not, so a sketch built here and the model's own bf16 logits
differ by that rounding and nothing else. Both the full-vocabulary pass
(:meth:`LinearOutputHead.logits`) and the selected-entry pass
(:meth:`LinearOutputHead.token_logits`) use this definition, so a retrieved
entry and a sketched one are values of ONE distribution (equal up to float32
summation order, which the tests bound).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

SCHEMA = "aadistill.output_head/v1"
LOGIT_DEFINITION = "float32_affine_head_of_model_dtype_final_hidden/v1"

#: Vocabulary rows per float32 widening in :meth:`LinearOutputHead.logits`, and
#: the ``[rows, distinct entries]`` product per block in
#: :meth:`LinearOutputHead.token_logits`: both bound a transient, neither is
#: semantics.
_VOCAB_BLOCK = 32768
_GATHER_BUDGET_ELEMENTS = 1 << 25


class OutputHeadError(ValueError):
    """An output head that cannot produce the logits asked of it."""


@dataclass(frozen=True)
class LinearOutputHead:
    """``logits = softcap(scale * (h @ W.T + b))``, row-wise.

    ``weight`` ``[V, H]`` and ``bias`` ``[V]`` are held as the model holds them
    (no copy is made here; the reference model is frozen). ``tied`` records
    whether the family shares ``weight`` with its input embedding -- evidence
    of what the head is, not a different computation.
    """

    weight: torch.Tensor
    bias: torch.Tensor | None = None
    logit_scale: float = 1.0
    final_softcap: float | None = None
    tied: bool = False

    def __post_init__(self) -> None:
        if self.weight.ndim != 2:
            raise OutputHeadError(
                f"an output head weight is [V, H], got {tuple(self.weight.shape)}")
        if self.bias is not None and tuple(self.bias.shape) != (self.vocab_size,):
            raise OutputHeadError("the head bias does not match its vocabulary")
        if not math.isfinite(float(self.logit_scale)) or self.logit_scale == 0:
            raise OutputHeadError(f"logit_scale {self.logit_scale!r}")
        if self.final_softcap is not None and not float(self.final_softcap) > 0:
            raise OutputHeadError(f"final_softcap {self.final_softcap!r}")

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.weight.shape[1])

    def identity(self) -> dict[str, Any]:
        """What the head IS, as a declaration. The teacher's checkpoint
        identity binds its bytes; this binds the semantics a store relies on."""
        return {"schema": SCHEMA, "logit_definition": LOGIT_DEFINITION,
                "vocab_size": self.vocab_size, "hidden_size": self.hidden_size,
                "weight_dtype": str(self.weight.dtype).replace("torch.", ""),
                "bias": self.bias is not None,
                "logit_scale": float(self.logit_scale),
                "final_softcap": (None if self.final_softcap is None
                                  else float(self.final_softcap)),
                "tied_embeddings": bool(self.tied)}

    @torch.no_grad()
    def checksum(self) -> float:
        """A cheap content probe of the weight (float64 sum of a strided row
        sample). EVIDENCE that a retrieval reads the head a store was built
        with -- not an identity, which is the checkpoint's."""
        stride = max(1, self.vocab_size // 1024)
        return float(self.weight[::stride].double().sum())

    def _post(self, z: torch.Tensor) -> torch.Tensor:
        if self.logit_scale != 1.0:
            z = z * float(self.logit_scale)
        if self.final_softcap is not None:
            cap = float(self.final_softcap)
            z = torch.tanh(z / cap) * cap
        return z

    def _require_hidden(self, hidden: torch.Tensor) -> None:
        if int(hidden.shape[-1]) != self.hidden_size:
            raise OutputHeadError(
                f"hidden states of width {int(hidden.shape[-1])} for a head of "
                f"width {self.hidden_size}: they belong to different models")

    @torch.no_grad()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[N, V]`` float32 logits of every entry at ``hidden`` ``[N, H]``."""
        self._require_hidden(hidden)
        h = hidden.to(self.weight.device).float()
        out = torch.empty(int(h.shape[0]), self.vocab_size,
                          dtype=torch.float32, device=h.device)
        for v0 in range(0, self.vocab_size, _VOCAB_BLOCK):
            v1 = min(v0 + _VOCAB_BLOCK, self.vocab_size)
            block = h @ self.weight[v0:v1].float().T
            if self.bias is not None:
                block = block + self.bias[v0:v1].float()
            out[:, v0:v1] = block
        return self._post(out)

    @torch.no_grad()
    def token_logits(self, hidden: torch.Tensor,
                     token_ids: torch.Tensor) -> torch.Tensor:
        """``[N, M]`` float32 logits of entries ``token_ids`` ``[N, M]`` at
        ``hidden`` ``[N, H]`` -- row ``n``'s own entries only.

        Exact under the same definition as :meth:`logits`. Per block of rows,
        the block's DISTINCT entries are projected once -- one float32 GEMM of
        the widened hidden states against the widened head rows of just those
        entries -- and each row reads its own columns. Neighbouring positions
        share most of their candidate entries, so this reads far fewer head
        rows than gathering ``[N, M, H]``; the block is sized so the
        ``[rows, distinct]`` product stays bounded.
        """
        self._require_hidden(hidden)
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != int(hidden.shape[0]):
            raise OutputHeadError(
                f"token ids {tuple(token_ids.shape)} do not index the rows of "
                f"hidden {tuple(hidden.shape)}")
        n, m = int(token_ids.shape[0]), int(token_ids.shape[1])
        out = torch.empty(n, m, dtype=torch.float32, device=hidden.device)
        if n == 0 or m == 0:
            return out
        ids = token_ids.to(self.weight.device).long()
        if bool(((ids < 0) | (ids >= self.vocab_size)).any()):
            raise OutputHeadError("a token id outside the head's vocabulary")
        #: rows * distinct <= rows * min(rows * M, V) <= budget.
        rows = max(1, int(math.isqrt(max(1, _GATHER_BUDGET_ELEMENTS // max(1, m)))))
        for r0 in range(0, n, rows):
            r1 = min(r0 + rows, n)
            distinct, inverse = torch.unique(ids[r0:r1], return_inverse=True)
            z = (hidden[r0:r1].to(self.weight.device).float()
                 @ self.weight[distinct].float().T)
            if self.bias is not None:
                z = z + self.bias[distinct].float()
            out[r0:r1] = self._post(z.gather(1, inverse)).to(out.device)
        return out

__all__ = ["LOGIT_DEFINITION", "LinearOutputHead", "OutputHeadError", "SCHEMA"]
