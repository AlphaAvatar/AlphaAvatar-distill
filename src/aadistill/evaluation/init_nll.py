"""Teacher-native held-out metrics over masked positions.

`general_text.general_text_metrics` scores a **dense** stream where every
position is a real prediction. The pack's validation slice is not dense: it is
8,192-token blocks of rendered sessions with padding, and the quantity every
Stage 3 arm has ever reported is CE over the **assistant-target** positions only.
Scoring all positions there would report a different number under the same name.

So this is the masked counterpart, and it reports the same four things the
general-text diagnostic does — NLL, teacher KL, top-1, mean rank — restricted to
the mask the trainer itself uses.

One naming caution, because this project has a binding scope rule about it: the
`top1` here is top-1 over *assistant-target* positions of the validation slice.
It is **not** the rollout harness's teacher-forced *reasoning* top-1, which is
measured on a different set through a different path. The two must not be
compared or substituted.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F


@torch.no_grad()
def masked_teacher_native_metrics(
    student,
    ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    teacher=None,
    device: str = "cpu",
    micro_blocks: int = 1,
    max_blocks: int | None = None,
    chunk: int = 512,
    autocast: bool = False,
) -> dict:
    """NLL / KL / top-1 / rank over the positions `loss_mask` selects.

    `ids` and `loss_mask` are `[N, L]` as `aadistill.data.ladder.ladder_blocks`
    returns them. A position `i` is scored when `loss_mask[:, i + 1]` is set,
    matching the trainer's shift convention exactly — the mask marks *target*
    tokens, and the prediction for a target sits one position earlier.
    """
    if ids.shape != loss_mask.shape:
        raise ValueError(f"ids {tuple(ids.shape)} != mask {tuple(loss_mask.shape)}")
    n = ids.shape[0] if max_blocks is None else min(int(max_blocks), ids.shape[0])
    if n == 0:
        raise ValueError("no blocks to evaluate")
    was_training = student.training
    student.eval()
    if teacher is not None:
        teacher.eval()

    tot = {"ce": 0.0, "kl": 0.0, "top1": 0, "rank": 0, "n": 0}
    for i in range(0, n, micro_blocks):
        batch = ids[i : i + micro_blocks].to(device)
        mask = loss_mask[i : i + micro_blocks].to(device).bool()
        keep = mask[:, 1:].reshape(-1)
        if not bool(keep.any()):
            continue
        ctx = (torch.autocast(
            device_type="cuda" if str(device).startswith("cuda") else "cpu",
            dtype=torch.bfloat16) if autocast else nullcontext())
        with ctx:
            s_logits = student(batch).logits
            t_logits = teacher(batch).logits if teacher is not None else None
        targets = batch[:, 1:].reshape(-1)[keep]
        s_flat = s_logits[:, :-1].reshape(-1, s_logits.shape[-1])[keep]
        t_flat = (None if t_logits is None
                  else t_logits[:, :-1].reshape(-1, t_logits.shape[-1])[keep])
        del s_logits, t_logits
        for a in range(0, s_flat.shape[0], chunk):
            s = s_flat[a : a + chunk].float()
            tg = targets[a : a + chunk]
            logp = F.log_softmax(s, dim=-1)
            tot["ce"] += float(-logp.gather(1, tg[:, None]).sum())
            tot["top1"] += int((s.argmax(-1) == tg).sum())
            # 1-based rank: how many logits strictly beat the target, plus one.
            tot["rank"] += int((s > s.gather(1, tg[:, None])).sum(-1).sum()) + \
                int(tg.shape[0])
            if t_flat is not None:
                t_logp = F.log_softmax(t_flat[a : a + chunk].float(), dim=-1)
                tot["kl"] += float((t_logp.exp() * (t_logp - logp)).sum(-1).sum())
            tot["n"] += int(tg.shape[0])
        del s_flat, t_flat

    if was_training:
        student.train()
    m = tot["n"]
    if m == 0:
        raise ValueError("the mask selected no prediction position")
    out = {
        "blocks": n,
        "positions": m,
        "nll": round(tot["ce"] / m, 6),
        "ppl": round(math.exp(min(tot["ce"] / m, 30.0)), 4),
        "top1": round(tot["top1"] / m, 6),
        "mean_rank": round(tot["rank"] / m, 4),
        "position_scope": "assistant-target positions of the validation slice; "
                          "NOT the rollout harness's reasoning top-1",
    }
    if teacher is not None:
        out["kl"] = round(tot["kl"] / m, 6)
    return out
