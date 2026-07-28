"""Batched token-in / token-out generation inside the training stack.

Why this exists rather than a serving engine
--------------------------------------------
Stage 4/5 trains on data the model itself produced. If the rollouts are made by
an engine whose numerics differ from the trainer's, the updates are only
nominally on-policy: the rollout policy and the training policy are different
distributions, which is the documented "training-inference mismatch" behind a
lot of RL instability (README References; decision record 2026-07-28). The
cheapest way not to have that problem is not to have two stacks — generate with
the same modeling code, kernels and dtype that the trainer uses.

That is what this module is: `model.generate` with proper batching, no separate
runtime. It is deliberately small. The throughput problem it solves is not
"transformers is slow", it is that the existing eval path decodes at
`batch_size` 1; batching is most of the available speedup at this project's
model sizes.

Token-in / token-out
--------------------
Callers pass prompt **token ids** and get completion **token ids** back. No
string round-trip anywhere in the path. This matters for building training
corpora: `decode` then `encode` is not the identity for most tokenizers, so a
corpus stored as text can train the model on a *different token sequence* than
the one it actually generated. Storing the ids removes that class of drift; the
text is derived for readability only.

What is verified, and what is not
---------------------------------
`tests/test_generate_toy.py` asserts that a prompt decodes to the same tokens
whether it is generated alone or inside a batch, and at different batch
positions. That property is **not** free: left-padding plus reduction-order
differences in batched attention can change logits in the last bits and flip an
argmax. The test is there to catch that, and `assert_batch_invariant` exposes
the same check at runtime for a real model, because passing on a toy model does
not prove it holds for a 4B one in bf16.
"""

from __future__ import annotations

import torch


def _left_pad(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad to a common length. Returns (input_ids, attention_mask).

    Left, not right: decoding continues from the final position, so every row's
    real last token must sit at the same index for the generated step to be
    aligned across the batch.
    """
    width = max(len(s) for s in sequences)
    ids, mask = [], []
    for s in sequences:
        pad = width - len(s)
        ids.append([pad_id] * pad + list(s))
        mask.append([0] * pad + [1] * len(s))
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long))


@torch.no_grad()
def generate_ids(
    model,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int | None = None,
    batch_size: int = 8,
    greedy: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int | None = None,
    device: str | torch.device | None = None,
) -> list[dict]:
    """Generate completions for `prompts` (lists of token ids).

    Returns one dict per prompt, in input order:
        {"tokens": [int], "n_new": int, "hit_cap": bool, "finished": bool}

    `tokens` holds the completion only — the prompt is not echoed back, so the
    caller never has to slice it off and cannot get that slice wrong.

    Sampling is seeded per batch from `seed` so a run is reproducible; with
    `greedy=True` the sampling parameters are ignored entirely (and no
    generator is consumed), which is the mode every evaluation uses.
    """
    if pad_token_id is None:
        pad_token_id = eos_token_id
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    out: list[dict | None] = [None] * len(prompts)
    # Group by length so a batch is not dominated by its longest member; the
    # original index rides along so results can be restored to input order.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        ids, mask = _left_pad([prompts[i] for i in idxs], pad_token_id)
        ids, mask = ids.to(device), mask.to(device)
        prompt_width = ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if greedy:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

        if greedy or seed is None:
            seq = model.generate(ids, attention_mask=mask, **gen_kwargs)
        else:
            # `generate` samples from the global RNG — it takes no generator
            # argument — so seeding means setting it. Save and restore around
            # the call: this may run inside a training loop, and silently
            # advancing (or resetting) the global stream would make the *training*
            # run irreproducible, which is a far worse bug than a slow decode.
            cpu_state = torch.get_rng_state()
            cuda_state = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            )
            try:
                # Vary per batch so different batches are not identical draws,
                # while staying a pure function of (seed, batch index).
                torch.manual_seed(seed + start)
                seq = model.generate(ids, attention_mask=mask, **gen_kwargs)
            finally:
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

        for row, i in enumerate(idxs):
            new = seq[row, prompt_width:].tolist()
            # generate() pads finished rows out to the longest row in the
            # batch; cut at the first eos so a completion's length does not
            # depend on who it shared a batch with.
            finished = False
            for pos, tok in enumerate(new):
                if tok == eos_token_id:
                    new = new[:pos + 1]
                    finished = True
                    break
            out[i] = {
                "tokens": new,
                "n_new": len(new),
                "hit_cap": (not finished) and len(new) >= max_new_tokens,
                "finished": finished,
            }

    if was_training:
        model.train()
    return [o for o in out if o is not None]


def assert_batch_invariant(
    model,
    prompts: list[list[int]],
    *,
    eos_token_id: int,
    max_new_tokens: int = 32,
    pad_token_id: int | None = None,
    device: str | torch.device | None = None,
) -> dict:
    """Check that batching does not change the tokens a prompt generates.

    Generates every prompt alone and again as one batch, greedily, and reports
    where they diverge. Returns a dict with `identical` (bool), `n` and
    `first_divergence` (per-prompt index of the first differing token, or None).

    Run this on the real model before trusting a generated corpus: the property
    holds by construction for the *modeling code* but not for the *arithmetic*,
    and bf16 on a 4B model is exactly where it is most likely to break.
    """
    alone = generate_ids(
        model, prompts, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
        pad_token_id=pad_token_id, batch_size=1, greedy=True, device=device,
    )
    batched = generate_ids(
        model, prompts, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
        pad_token_id=pad_token_id, batch_size=len(prompts), greedy=True, device=device,
    )
    divergence: list[int | None] = []
    for a, b in zip(alone, batched):
        ta, tb = a["tokens"], b["tokens"]
        first = None
        for i in range(min(len(ta), len(tb))):
            if ta[i] != tb[i]:
                first = i
                break
        if first is None and len(ta) != len(tb):
            first = min(len(ta), len(tb))
        divergence.append(first)
    return {
        "identical": all(d is None for d in divergence),
        "n": len(prompts),
        "n_diverged": sum(d is not None for d in divergence),
        "first_divergence": divergence,
    }
