"""How far one causal LM's next-token distributions are from another's, on given sequences.

The measurement an initialization owes when it is an approximation of the
model it was built from: a student whose sub-layers read and write through
low-rank projections computes a different function from its full-width parent,
and how different is a number to record, not an assumption to state.

At every position a caller marks (a ``[B, T-1]`` boolean mask over prediction
positions, the trainer's layout), both models' FULL-vocabulary distributions are
compared in float32 from their own output heads under one definition:

* ``kl``: forward ``KL(reference || candidate)`` in nats (mean, p50, p90, max);
* ``top1_agreement``: the share of positions where the argmaxes agree;
* ``logit_abs_diff``: mean and max of ``|z_ref - z_cand|`` after removing each
  row's mean (logits are defined up to a per-row constant);
* ``reference_ce`` / ``candidate_ce``: mean CE at the next tokens.

Both backbones run the whole sequence; only the marked rows reach a head
(:func:`aadistill.training.topk_kd.head_rows`), in blocks of ``chunk`` rows.
Nothing here knows a family, a width or a vocabulary.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from aadistill.training.topk_kd import head_rows, output_head

SCHEMA = "aadistill.function_error/v1"

__all__ = ["SCHEMA", "function_error"]


@torch.no_grad()
def function_error(reference: Any, candidate: Any, batches: Iterable[tuple], *,
                   chunk: int = 256, autocast_dtype: torch.dtype | None = None) -> dict:
    """``batches`` yields ``(input_ids [B, T], rows [B, T-1] bool)`` or ``(..., forward_kwargs)``."""
    ref_head, cand_head = output_head(reference), output_head(candidate)
    kls, agree, mean_abs, max_abs, ref_ce, cand_ce = [], [], [], 0.0, [], []
    for batch in batches:
        ids, rows = batch[0], batch[1].bool()
        fwd = batch[2] if len(batch) > 2 else {}
        dev = ids.device
        ctx = (torch.autocast(device_type=dev.type, dtype=autocast_dtype) if autocast_dtype
               else torch.autocast(device_type=dev.type, enabled=False))
        with ctx:
            hr = head_rows(reference, ids, rows, **fwd)
            hc = head_rows(candidate, ids, rows, **fwd)
        targets = ids[:, 1:][rows]
        for start in range(0, int(hr.shape[0]), max(1, int(chunk))):
            s = slice(start, start + max(1, int(chunk)))
            with ctx:
                zr, zc = ref_head(hr[s]).float(), cand_head(hc[s]).float()
            lr, lc = torch.log_softmax(zr, -1), torch.log_softmax(zc, -1)
            kls.append((lr.exp() * (lr - lc)).sum(-1))
            agree.append(zr.argmax(-1) == zc.argmax(-1))
            d = (zr - zr.mean(-1, keepdim=True)) - (zc - zc.mean(-1, keepdim=True))
            mean_abs.append(d.abs().mean(-1))
            max_abs = max(max_abs, float(d.abs().max()))
            t = targets[s][:, None]
            ref_ce.append(-lr.gather(-1, t).squeeze(-1))
            cand_ce.append(-lc.gather(-1, t).squeeze(-1))
    if not kls:
        raise ValueError("function_error over zero positions")
    kl = torch.cat(kls).double()
    q = torch.quantile(kl, torch.tensor([0.5, 0.9], dtype=kl.dtype, device=kl.device))
    return {"schema": SCHEMA, "positions": int(kl.numel()),
            "kl": {"mean": float(kl.mean()), "p50": float(q[0]), "p90": float(q[1]),
                   "max": float(kl.max())},
            "top1_agreement": float(torch.cat(agree).double().mean()),
            "logit_abs_diff": {"mean": float(torch.cat(mean_abs).double().mean()), "max": max_abs},
            "reference_ce": float(torch.cat(ref_ce).double().mean()),
            "candidate_ce": float(torch.cat(cand_ce).double().mean()),
            "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype).replace("torch.", "")}
