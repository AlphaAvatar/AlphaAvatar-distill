"""General-text diagnostics on a dense held-out stream.

E7's first question is whether general language modelling can be restored. These
are the numbers that answer it — and every one of them is a **diagnostic**. They
may describe what the treatment did to the student's distribution over ordinary
prose; they may not promote a checkpoint. That rule is not a style preference:
E6b showed two objectives improving validation CE by the same amount while only
one moved autonomous behaviour, and the FineWeb NLL of the E1 lineage actually
*reverses* against behaviour past the 2.96M rung.

Four quantities, deliberately separate:

``nll``
    Mean next-token cross-entropy against the real text. The headline
    general-LM number, comparable to `holdout_nll` in `e1_consolidated.json`.
``kl``
    Forward KL from the teacher's distribution to the student's, at the same
    positions. This is what the extra stream actually optimizes, so reporting it
    beside `nll` distinguishes "matched the teacher better" from "modelled the
    text better" — which can come apart when the teacher is itself wrong.
``top1`` / ``mean_rank``
    Where the true token lands in the student's ordering. Rank moves long before
    top-1 does and is the more sensitive early signal.
``mean_target_prob`` / ``mean_entropy``
    Confidence. A model that restores NLL by becoming uniformly less confident
    is not the same result as one that restores it by being right, and the two
    are indistinguishable from NLL alone.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F


@torch.no_grad()
def general_text_metrics(
    student,
    ids: torch.Tensor,
    *,
    teacher=None,
    device: str = "cpu",
    micro_blocks: int = 1,
    max_blocks: int | None = None,
    chunk: int = 512,
    autocast: bool = False,
) -> dict:
    """Diagnostics over a dense block tensor `[N, L]` of real tokens.

    The stream is dense by construction (`aadistill.data.extra_stream`), so
    every position is a real next-token prediction and there is no mask to
    apply — which is also why the position count is exactly `N * (L - 1)` and
    can be checked against the manifest rather than trusted.
    """
    n = ids.shape[0] if max_blocks is None else min(int(max_blocks), ids.shape[0])
    if n == 0:
        raise ValueError("no blocks to evaluate")
    was_training = student.training
    student.eval()
    if teacher is not None:
        teacher.eval()

    tot = {"ce": 0.0, "kl": 0.0, "top1": 0, "rank": 0, "prob": 0.0,
           "entropy": 0.0, "n": 0}
    for i in range(0, n, micro_blocks):
        batch = ids[i : i + micro_blocks].to(device)
        ctx = (torch.autocast(
            device_type="cuda" if str(device).startswith("cuda") else "cpu",
            dtype=torch.bfloat16) if autocast else nullcontext())
        with ctx:
            s_logits = student(batch).logits
            t_logits = teacher(batch).logits if teacher is not None else None
        targets = batch[:, 1:].reshape(-1)
        s_flat = s_logits[:, :-1].reshape(-1, s_logits.shape[-1])
        t_flat = (None if t_logits is None
                  else t_logits[:, :-1].reshape(-1, t_logits.shape[-1]))
        # Chunked so a 150k-vocabulary float32 softmax over thousands of
        # positions cannot spike memory on the eval path.
        for a in range(0, s_flat.shape[0], chunk):
            s = s_flat[a : a + chunk].float()
            tg = targets[a : a + chunk]
            logp = F.log_softmax(s, dim=-1)
            tok_lp = logp.gather(1, tg[:, None]).squeeze(1)
            tot["ce"] += float(-tok_lp.sum())
            tot["prob"] += float(tok_lp.exp().sum())
            tot["entropy"] += float(-(logp.exp() * logp).sum(-1).sum())
            tot["top1"] += int((s.argmax(-1) == tg).sum())
            # Rank is 1-based: how many logits strictly beat the target, plus one.
            tot["rank"] += int((s > s.gather(1, tg[:, None])).sum(-1).sum()) + \
                int(tg.shape[0])
            if t_flat is not None:
                t = t_flat[a : a + chunk].float()
                t_logp = F.log_softmax(t, dim=-1)
                tot["kl"] += float((t_logp.exp() * (t_logp - logp)).sum(-1).sum())
            tot["n"] += int(tg.shape[0])

    if was_training:
        student.train()
    m = tot["n"]
    out = {
        "blocks": n,
        "positions": m,
        "nll": round(tot["ce"] / m, 6),
        "ppl": round(math.exp(min(tot["ce"] / m, 30.0)), 4),
        "top1": round(tot["top1"] / m, 6),
        "mean_rank": round(tot["rank"] / m, 4),
        "mean_target_prob": round(tot["prob"] / m, 6),
        "mean_entropy": round(tot["entropy"] / m, 6),
    }
    if teacher is not None:
        out["kl"] = round(tot["kl"] / m, 6)
    return out
