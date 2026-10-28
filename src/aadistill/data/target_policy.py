"""Which tokens of a packed sequence are training TARGETS, beyond what the pack's loss mask says.

A pack's loss mask marks the tokens a builder supervised. Some builders
supervised tokens the teacher never generated: a chat template that PREFILLS
the start of the assistant turn (for example an opening reasoning delimiter)
puts those tokens in the rendered turn, and a mask over "the assistant turn"
then supervises them although they are part of the prompt the deployment
template supplies. :func:`drop_template_prefill` removes exactly those
positions, from the recorded token ids alone: in every supervised run that
begins immediately after the generation prompt and whose first tokens are the
prefill, the prefill positions stop being targets. Nothing is re-rendered or
re-tokenized, the sequences are byte-identical, and the ids are the caller's
(a recipe's template facts), never this module's.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["drop_template_prefill"]


def drop_template_prefill(input_ids: torch.Tensor, loss_mask: torch.Tensor, *,
                          generation_prompt: Sequence[int], prefill: Sequence[int]
                          ) -> tuple[torch.Tensor, dict]:
    """``(new_mask, stats)``: ``loss_mask`` with every template-prefilled target unset.

    A supervised run (a maximal run of True along a row) is MATCHED when the
    tokens just before it are ``generation_prompt`` and its first tokens are
    ``prefill``; its first ``len(prefill)`` positions are then unset. ``stats``
    counts the runs, the matched runs, the runs after the generation prompt that
    do NOT start with the prefill (a template the policy does not describe; the
    caller decides whether that refuses) and the positions removed.
    """
    gp, pf = [int(t) for t in generation_prompt], [int(t) for t in prefill]
    if not gp or not pf:
        raise ValueError("generation_prompt and prefill must be non-empty")
    ids, mask = input_ids.cpu(), loss_mask.cpu().bool()
    out = mask.clone()
    runs = matched = unmatched = removed = 0
    for b in range(int(ids.shape[0])):
        m = mask[b].to(torch.int8)
        edges = torch.nonzero(torch.diff(torch.cat([m.new_zeros(1), m, m.new_zeros(1)])), as_tuple=False).flatten()
        row = ids[b].tolist()
        for s, e in zip(edges[0::2].tolist(), edges[1::2].tolist()):
            runs += 1
            if s < len(gp) or row[s - len(gp):s] != gp:
                continue
            if e - s >= len(pf) and row[s:s + len(pf)] == pf:
                out[b, s:s + len(pf)] = False
                matched += 1
                removed += len(pf)
            else:
                unmatched += 1
    return out.to(loss_mask.device), {"runs": runs, "matched_runs": matched,
                                      "unmatched_runs_after_generation_prompt": unmatched,
                                      "removed_positions": removed,
                                      "generation_prompt": gp, "prefill": pf}
