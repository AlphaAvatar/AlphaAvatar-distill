"""Rollout snapshots and off-policy correction diagnostics.

Why this exists
---------------
Stage 4/5 generates rollouts on an efficient engine that is **not** the trainer
(AGENTS.md §4.6, decision 2026-07-30). That engine is a different implementation,
so its policy differs measurably from the trainer's. The project's earlier
instinct — require the engine to reproduce the trainer's tokens — was retired,
because token equality is not a prerequisite for on-policy training and is not a
property the trainer even has against itself (decoding is not batch-invariant
within one stack: 7/8 in-stack, 4/8 vLLM, measured 2026-07-30).

The mismatch is handled instead by measuring and correcting it. That needs three
things, and this module is all three:

1. **A snapshot**: the exact rollout token ids, the rollout policy's log-prob for
   each one, and enough identity to say which policy produced them. Hashed,
   because the snapshot *is* the ground truth a correction is computed against —
   not a re-derived policy, which would not reproduce (P4/P5).
2. **A trainer-side scorer**: recompute the trainer policy's log-prob of those
   exact tokens, teacher-forced.
3. **Diagnostics**: the importance-ratio distribution, its tail, and the
   off-policy fraction, which are what a pre-registered stability bound is
   stated in.

What this module does not do
----------------------------
It does not train. Applying the correction inside a loss belongs to the Stage 5
trainer, and is deliberately separate: these functions are measurable on their
own, on CPU, against a toy model, before any of it costs money (P8).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

SCHEMA = "aadistill.rollout.v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def score_tokens(
    model,
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    device: str | torch.device | None = None,
) -> list[float]:
    """Trainer-policy log-probabilities of `completion_ids`, teacher-forced.

    Returns one value per completion token: `log p(token_t | prompt, tokens<t)`
    under `model`. This is the numerator of the importance ratio; the rollout
    engine supplied the denominator at sampling time.

    The alignment is the whole point and is easy to get wrong. Feeding
    `prompt + completion`, the logits at position `i` predict token `i+1`, so the
    distribution for the first completion token sits at the **last prompt
    position**. Off by one here would score every token against its neighbour's
    distribution and produce ratios that look plausible and are wrong.
    """
    if not completion_ids:
        return []
    device = device or next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=device)
    logits = model(ids).logits.float()

    start = len(prompt_ids) - 1               # predicts completion token 0
    window = logits[0, start:start + len(completion_ids)]
    logprobs = torch.log_softmax(window, dim=-1)
    targets = torch.tensor(completion_ids, device=device)
    return logprobs.gather(1, targets.unsqueeze(1)).squeeze(1).tolist()


def importance_stats(
    rollout_logprobs: list[float | None],
    trainer_logprobs: list[float],
    *,
    band: float = 2.0,
) -> dict:
    """Importance-ratio diagnostics for one rollout sequence.

    `ratio_t = exp(logp_trainer_t - logp_rollout_t)`. A ratio of 1 means the two
    policies agree on that token; the distribution's *tail* is what destabilises
    training, so the summary reports quantiles rather than a mean.

    Positions where the rollout engine reported no log-probability are **masked,
    not imputed** — a fabricated denominator would silently bias every statistic
    here. `n_masked` is reported so a caller can see how much was dropped.

    `band` is the two-sided factor defining "off-policy": a token counts as
    off-policy when `ratio > band` or `ratio < 1/band`. `off_policy_rate` against
    a pre-registered band is the quantity a stability bound is stated in.

    `kl` is the standard k3 estimator of KL(trainer || rollout),
    `mean(r - 1 - log r)`, which is non-negative and lower-variance than the
    naive `mean(-log r)`.
    """
    if len(rollout_logprobs) != len(trainer_logprobs):
        raise ValueError(
            f"length mismatch: {len(rollout_logprobs)} rollout vs "
            f"{len(trainer_logprobs)} trainer log-probs")

    pairs = [(r, t) for r, t in zip(rollout_logprobs, trainer_logprobs) if r is not None]
    n_masked = len(rollout_logprobs) - len(pairs)
    if not pairs:
        return {"n": 0, "n_masked": n_masked, "ratio_median": None,
                "ratio_p95": None, "ratio_p99": None, "ratio_max": None,
                "off_policy_rate": None, "kl": None, "band": band}

    # Clamp in log space before exponentiating, so a pathological pair cannot
    # overflow to inf and poison every summary statistic; 80 in log space is
    # already astronomically large. The clamped log ratio is then the *only*
    # source for both the ratio and the KL term — deriving them independently
    # would leave the KL using an unclamped value the quantiles never saw.
    log_ratios = [min(max(t - r, -80.0), 80.0) for r, t in pairs]
    ratio_pairs = sorted((math.exp(x), x) for x in log_ratios)
    ratios = [r for r, _ in ratio_pairs]

    def q(p: float) -> float:
        return ratios[min(int(p * len(ratios)), len(ratios) - 1)]

    off = sum(1 for x in ratios if x > band or x < 1.0 / band)
    kl = sum(r - 1.0 - lr for r, lr in ratio_pairs) / len(ratio_pairs)
    return {
        "n": len(pairs),
        "n_masked": n_masked,
        "ratio_median": round(q(0.5), 6),
        "ratio_p95": round(q(0.95), 6),
        "ratio_p99": round(q(0.99), 6),
        "ratio_max": round(ratios[-1], 6),
        "off_policy_rate": round(off / len(ratios), 6),
        "kl": round(kl, 6),
        "band": band,
    }


def aggregate_stats(per_sequence: list[dict], *, band: float = 2.0) -> dict:
    """Pool per-sequence diagnostics into the numbers a bound is stated against.

    Pools by **token**, not by sequence: a stability bound of "under 5% of tokens
    clipped" is a statement about tokens, and averaging per-sequence rates would
    weight a 10-token sequence like a 4,000-token one.
    """
    usable = [s for s in per_sequence if s.get("n")]
    if not usable:
        return {"sequences": len(per_sequence), "tokens": 0}
    tokens = sum(s["n"] for s in usable)
    return {
        "sequences": len(per_sequence),
        "sequences_with_tokens": len(usable),
        "tokens": tokens,
        "tokens_masked": sum(s["n_masked"] for s in per_sequence),
        "off_policy_rate": round(
            sum(s["off_policy_rate"] * s["n"] for s in usable) / tokens, 6),
        "kl_mean": round(sum(s["kl"] * s["n"] for s in usable) / tokens, 6),
        "ratio_median_of_medians": round(
            sorted(s["ratio_median"] for s in usable)[len(usable) // 2], 6),
        "ratio_max": round(max(s["ratio_max"] for s in usable), 6),
        "band": band,
    }


def write_snapshot(
    out_dir: str | Path,
    records: list[dict],
    *,
    policy: dict,
    engine: dict,
    sampling: dict,
) -> dict:
    """Write a hashed rollout snapshot and return its manifest.

    `records` are dicts with at least `prompt_id`, `prompt_tokens`, `tokens` and
    `logprobs`. `policy` identifies the checkpoint that generated them (id, step,
    hash); `engine` identifies the backend and its version.

    Both identities are mandatory rather than nice-to-have: a rollout is only
    interpretable against the policy that produced it, and an engine change moves
    the sampling distribution, so a snapshot missing either cannot support a
    correction term later (AGENTS.md §4.6).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in ("checkpoint", "step"):
        if field not in policy:
            raise ValueError(f"policy identity is missing {field!r}")
    if "name" not in engine:
        raise ValueError("engine identity is missing 'name'")

    path = out_dir / "rollouts.jsonl"
    with open(path, "w") as f:
        for record in records:
            missing = {"prompt_id", "prompt_tokens", "tokens"} - set(record)
            if missing:
                raise ValueError(f"rollout record is missing {sorted(missing)}")
            if record.get("logprobs") is not None and \
                    len(record["logprobs"]) != len(record["tokens"]):
                raise ValueError(
                    f"record {record['prompt_id']!r}: "
                    f"{len(record['logprobs'])} logprobs for "
                    f"{len(record['tokens'])} tokens")
            f.write(json.dumps(record) + "\n")

    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "engine": engine,
        "sampling": sampling,
        "n_records": len(records),
        "n_tokens": sum(len(r["tokens"]) for r in records),
        "rollouts_sha256": _sha256_file(path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_snapshot(out_dir: str | Path) -> tuple[list[dict], dict]:
    """Read a snapshot back, verifying its hash.

    The hash check is the point of the format. If the rollout file changed after
    the manifest was written, every correction computed from it is against
    different data than the record claims, so this fails loudly rather than
    proceeding.
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    path = out_dir / "rollouts.jsonl"
    actual = _sha256_file(path)
    if actual != manifest["rollouts_sha256"]:
        raise RuntimeError(
            f"rollout snapshot hash mismatch in {out_dir}: manifest says "
            f"{manifest['rollouts_sha256'][:12]}…, file is {actual[:12]}…")
    with open(path) as f:
        return [json.loads(line) for line in f], manifest
