"""Train directly on a pre-packed token ladder.

The scaling study's data-size variable is the supervised-token count that
survives packing, and its rungs are **nested prefixes of one pack**
(`scripts/data/build_token_ladder.py`). Re-packing at training time would
destroy both properties: block contents would depend on the trainer's own
packing seed, rung k would stop being a prefix of rung k+1, and each rung's
supervised-token count would become an independent draw rather than the
measured quantity the curve is plotted against.

So the trainer reads the pack. A rung is `blocks[:n]` — nothing is re-encoded,
re-ordered or re-packed, and the tokens that train are byte-identical to the
tokens the gate validated.

Validation blocks come from the pack's **tail**, the blocks beyond the largest
rung, so no rung ever trains on them and every rung is validated on exactly the
same data. The tail's natural composition is skewed (the types with the biggest
pools are what is left over), so the val slice is selected by the same
largest-remainder rule the ladder uses, giving a near-uniform mixture. The
realized mixture is reported rather than assumed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_ladder_meta(packed_dir: str | Path) -> dict:
    """Read `ladder.json` from a pack directory."""
    path = Path(packed_dir) / "ladder.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — is this a token-ladder pack?")
    return json.loads(path.read_text())


def rung_n_blocks(meta: dict, rung: int) -> int:
    """Blocks in `rung`, which must be a declared and reachable rung."""
    for entry in meta["rungs"]:
        if int(entry["target_supervised_tokens"]) == int(rung):
            if not entry.get("reachable", False):
                raise ValueError(
                    f"rung {rung} is declared unreachable in this pack "
                    f"({entry.get('actual_supervised_tokens')} supervised tokens "
                    "available)")
            return int(entry["n_blocks"])
    declared = [e["target_supervised_tokens"] for e in meta["rungs"]]
    raise ValueError(f"rung {rung} is not in this pack; declared rungs {declared}")


def _block_type_tokens(audit_row: dict) -> Counter:
    """Supervised tokens per data type inside one packed block."""
    counts: Counter = Counter()
    for session in audit_row["sessions"]:
        counts[session["data_type"]] += int(session["supervised_retained"])
    return counts


def select_val_blocks(audit: list[dict], start: int, count: int) -> list[int]:
    """Pick `count` block indices from `audit[start:]`, mixture as even as possible.

    Largest-remainder on supervised tokens, the same rule the ladder orders by:
    repeatedly take the block whose dominant type is furthest behind an equal
    share. Deterministic, seed-free, and a pure function of (audit, start,
    count) — so every rung and both training seeds validate on one identical
    block set.
    """
    pool = list(range(start, len(audit)))
    if count > len(pool):
        raise ValueError(
            f"pack has {len(pool)} blocks past the largest rung, need {count} for "
            "validation; cut a larger pack or ask for fewer val blocks")
    types = sorted({s["data_type"] for i in pool for s in audit[i]["sessions"]})
    if not types:
        raise ValueError("no typed sessions in the validation tail")
    share = 1.0 / len(types)
    emitted: Counter = Counter()
    total = 0
    chosen: list[int] = []
    remaining = set(pool)
    for _ in range(count):
        best, best_deficit = None, None
        for idx in sorted(remaining):
            counts = _block_type_tokens(audit[idx])
            if not counts:
                continue
            dominant = max(sorted(counts), key=lambda t: counts[t])
            deficit = share * total - emitted[dominant]
            if best is None or deficit > best_deficit:
                best, best_deficit, best_counts = idx, deficit, counts
        if best is None:
            raise ValueError("validation tail ran out of typed blocks")
        chosen.append(best)
        remaining.discard(best)
        for t, v in best_counts.items():
            emitted[t] += v
            total += v
    return sorted(chosen)


def ladder_blocks(packed_dir: str | Path, rung: int, n_val: int = 16):
    """Return (train, val, stats) for one rung of a packed ladder.

    `train` and `val` are each `(input_ids, loss_mask, content_mask)` in the
    layout `Trainer` expects. Train is the rung prefix; val is a fixed slice of
    the pack tail, disjoint from every rung by construction.
    """
    packed_dir = Path(packed_dir)
    meta = load_ladder_meta(packed_dir)
    n_blocks = rung_n_blocks(meta, rung)

    audit = [json.loads(line) for line in open(packed_dir / "audit.jsonl")
             if line.strip()]
    arrays = np.load(packed_dir / "blocks.npz")
    input_ids = torch.from_numpy(arrays["input_ids"].astype(np.int64))
    loss_mask = torch.from_numpy(arrays["ce_mask"])
    content_mask = torch.from_numpy(arrays["content_mask"])
    if input_ids.shape[0] != len(audit):
        raise ValueError(
            f"pack has {input_ids.shape[0]} blocks but {len(audit)} audit rows")

    max_rung_blocks = max(int(e["n_blocks"]) for e in meta["rungs"]
                          if e.get("reachable", False))
    val_idx = select_val_blocks(audit, max_rung_blocks, n_val)
    val_tensor = torch.tensor(val_idx, dtype=torch.long)

    train = (input_ids[:n_blocks], loss_mask[:n_blocks], content_mask[:n_blocks])
    val = (input_ids[val_tensor], loss_mask[val_tensor], content_mask[val_tensor])

    train_tokens: Counter = Counter()
    for row in audit[:n_blocks]:
        train_tokens += _block_type_tokens(row)
    val_tokens: Counter = Counter()
    for idx in val_idx:
        val_tokens += _block_type_tokens(audit[idx])
    train_total = sum(train_tokens.values()) or 1
    val_total = sum(val_tokens.values()) or 1

    stats = {
        "source": "token_ladder",
        "packed_dir": str(packed_dir),
        "rung_target_supervised_tokens": int(rung),
        "train_blocks": int(n_blocks),
        "train_supervised_tokens": int(train_total),
        "train_token_mix": {t: round(v / train_total, 4)
                            for t, v in sorted(train_tokens.items())},
        "val_block_indices": val_idx,
        "val_blocks": len(val_idx),
        "val_supervised_tokens": int(val_total),
        "val_token_mix": {t: round(v / val_total, 4)
                          for t, v in sorted(val_tokens.items())},
        "val_disjoint_from_all_rungs": min(val_idx) >= max_rung_blocks,
        "declared_mixture": meta.get("declared_mixture"),
        "block_len": int(meta["block_len"]),
        "pack_blocks": int(input_ids.shape[0]),
    }
    if not stats["val_disjoint_from_all_rungs"]:
        raise ValueError("validation blocks overlap a training rung")
    return train, val, stats
