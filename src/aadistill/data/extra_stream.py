"""A second, KD-only token stream that runs alongside the rollout stream.

E7 asks whether adding general-text teacher KD restores general language
modelling without disturbing the rollout trajectory that produced the current
behavioural anchor. "Without disturbing" is the hard part, and it is a data
question before it is a loss question: if the extra text were merged into the
rollout pack, every block boundary, every permutation and every rollout example's
position against the learning-rate schedule would move, and the comparison to
`e1_r1600k_*_pca` would be gone.

So the extra text is a **separate stream with its own cursor**, consumed inside
the same optimizer steps. The rollout stream sees the same blocks, in the same
order, at the same step indices, under the same schedule. Nothing about it is a
function of the extra stream's existence.

The format is deliberately not the ladder's. A ladder block is a padded packing
of chat sessions and is 72% padding at the 1.60M rung (packing efficiency
0.2767); that is fine for supervised sessions, which cannot cross a system-prompt
boundary, and useless here. An extra block is **dense**: documents concatenated
with an explicit separator, no padding at all, incomplete tails dropped. Two
consequences that the experiment depends on:

* KD positions per block are exactly ``block_len - 1``, so a stream's token
  budget is an exact integer known before training rather than a property of how
  well something packed;
* two streams with the same ``(n_blocks, block_len)`` have **identical** KD
  position counts and identical teacher/student forward workloads, whatever they
  contain. That is what makes the FineWeb arm and its matched control differ in
  content and in nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class DocRow:
    """One source document's placement in the packed stream."""

    ord: int
    source_index: int
    source_id: str
    sha256: str
    n_tokens: int
    first_block: int
    first_offset: int

    def as_dict(self) -> dict:
        return asdict(self)


def content_sha256(text: str) -> str:
    """Hash of document *content*, excluding any id or metadata.

    Deduplication and disjointness proofs run on this, not on dataset indices.
    An index range says which rows were read; only the content hash says whether
    the same text arrived twice through two different ranges.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_dense(
    docs: list[tuple[str, str, int, list[int]]],
    block_len: int,
    separator_id: int,
    *,
    max_blocks: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[DocRow]]:
    """Concatenate tokenized documents into full, unpadded blocks.

    `docs` rows are `(source_id, content_sha256, source_index, token_ids)`.
    Each document is followed by `separator_id`, so a boundary is an explicit
    token rather than an implicit change of subject — the prediction at that
    position is a real next-token prediction and the teacher's distribution
    there is meaningful.

    The trailing partial block is **dropped**. Keeping it would mean either
    padding (which this format exists to avoid) or a stream whose last block has
    fewer KD positions than the rest, which would break exact budget matching
    between two streams.
    """
    if block_len < 2:
        raise ValueError("block_len must be at least 2")
    buf: list[int] = []
    rows: list[DocRow] = []
    blocks: list[list[int]] = []
    for i, (source_id, digest, source_index, tokens) in enumerate(docs):
        if not tokens:
            continue
        rows.append(DocRow(
            ord=i, source_index=source_index, source_id=source_id,
            sha256=digest, n_tokens=len(tokens),
            first_block=len(buf) // block_len + len(blocks),
            first_offset=len(buf) % block_len,
        ))
        buf.extend(tokens)
        buf.append(separator_id)
        while len(buf) >= block_len:
            blocks.append(buf[:block_len])
            buf = buf[block_len:]
            if max_blocks is not None and len(blocks) >= max_blocks:
                ids = np.asarray(blocks, dtype=np.int32)
                return ids, np.ones_like(ids, dtype=bool), rows
    if not blocks:
        raise ValueError(
            f"no complete blocks of {block_len} tokens; the source yielded "
            f"{len(buf)} tokens")
    ids = np.asarray(blocks, dtype=np.int32)
    return ids, np.ones_like(ids, dtype=bool), rows


def kd_positions(n_blocks: int, block_len: int) -> int:
    """Exact KD positions for a dense stream. No packing efficiency term."""
    return n_blocks * (block_len - 1)


def write_stream(out_dir: str | Path, ids: np.ndarray, content: np.ndarray,
                 docs: list[DocRow], manifest: dict) -> dict:
    """Write `blocks.npz` + `docs.jsonl` + `manifest.json`, hashes included."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "blocks.npz", input_ids=ids, content_mask=content)
    with open(out / "docs.jsonl", "w") as f:
        for row in docs:
            f.write(json.dumps(row.as_dict()) + "\n")

    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    full = {
        **manifest,
        "n_blocks": int(ids.shape[0]),
        "block_len": int(ids.shape[1]),
        "kd_positions": kd_positions(int(ids.shape[0]), int(ids.shape[1])),
        "total_tokens": int(ids.size),
        "padding_tokens": int((~content).sum()),
        "n_documents": len(docs),
        "outputs": {"blocks": _sha(out / "blocks.npz"),
                    "docs": _sha(out / "docs.jsonl")},
    }
    (out / "manifest.json").write_text(json.dumps(full, indent=2) + "\n")
    return full


def load_extra_stream(data_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Read a packed extra stream. Fails loudly rather than degrading."""
    d = Path(data_dir)
    manifest_path = d / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found — an extra stream must carry its "
            "manifest, because its identity is the experiment's treatment")
    meta = json.loads(manifest_path.read_text())
    z = np.load(d / "blocks.npz")
    ids = torch.from_numpy(z["input_ids"].astype(np.int64))
    content = torch.from_numpy(z["content_mask"])
    if ids.shape[0] != meta["n_blocks"] or ids.shape[1] != meta["block_len"]:
        raise ValueError(
            f"{d} holds {tuple(ids.shape)} blocks but its manifest declares "
            f"({meta['n_blocks']}, {meta['block_len']})")
    if int((~content).sum()) != 0:
        raise ValueError(
            f"{d} contains padding; an extra stream is dense by construction "
            "and a padded one would silently change the KD budget")
    return ids, content, meta


def stream_budget(n_blocks: int, block_len: int, *, total_steps: int,
                  blocks_per_step: int, every_n_steps: int = 1) -> dict:
    """The exact, auditable extra-KD budget a config will consume.

    Computed from the config alone, before anything trains, so a preregistration
    can state the number and a run can be checked against it.
    """
    if every_n_steps < 1:
        raise ValueError("every_n_steps must be >= 1")
    active_steps = (total_steps + every_n_steps - 1) // every_n_steps
    visits = active_steps * blocks_per_step
    return {
        "active_steps": active_steps,
        "block_visits": visits,
        "exposures": round(visits / n_blocks, 6) if n_blocks else 0.0,
        "kd_positions": visits * (block_len - 1),
        "forward_tokens": visits * block_len,
        "stream_blocks": n_blocks,
        "stream_block_len": block_len,
        "stream_unique_kd_positions": kd_positions(n_blocks, block_len),
    }
