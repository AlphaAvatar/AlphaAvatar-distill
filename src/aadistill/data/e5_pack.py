"""Pack E5's already-tokenized prefix/continuation examples into the trainer's format.

E5 examples are not sessions to be rendered: they are token streams that already
exist — for arm C a teacher trajectory with the mask moved, for arm R a student
prefix followed by a teacher recovery. Re-applying the chat template to them
would delete earlier reasoning traces (the finding behind the corpus builder) and
would not reproduce arm R's tokens at all.

So this adds an **already-tokenized path into the production packer** rather than
a second packing algorithm. Examples are adapted to `RenderedSession` and handed
to `pack_sessions`, which keeps every production semantic: the system prompt is a
hard group boundary and is emitted once per block, and two examples sharing a
source trajectory are never co-packed — for E5 that matters as much as it did for
turn expansion, because the two truncations of one trajectory are prefixes of one
another and co-packing would leak one into the other's context.

The one behavioural difference is registered: `allow_terminal_truncation=False`.
Production packing may cut the last session at a block boundary; E5 may not,
because a cut prefix changes the state being trained on and a cut continuation
silently shortens supervision.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .sessions import PackedBlock, RenderedSession, pack_sessions

REQUIRED_FIELDS = ("ids", "mask", "n_system_tokens", "system_key",
                   "source_session_id", "truncation_index", "data_type")


def example_to_rendered(example: dict) -> RenderedSession:
    """Adapt one E5 example record to the packer's input type.

    `body_ids` excludes the system block, which `pack_sessions` emits once per
    block — so the example's leading `n_system_tokens` are stripped here and must
    not be double-counted.

    `source_id` is the source *trajectory*, which is what makes the packer defer
    a bundle sibling to another block instead of co-packing it.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in example]
    if missing:
        raise ValueError(f"example {example.get('id')!r} missing {missing}")
    ids, mask = list(example["ids"]), [bool(m) for m in example["mask"]]
    if len(ids) != len(mask):
        raise ValueError(f"example {example['id']!r}: ids/mask length mismatch")
    n_sys = int(example["n_system_tokens"])
    if any(mask[:n_sys]):
        raise ValueError(f"example {example['id']!r}: system block is supervised")
    body_ids, body_mask = ids[n_sys:], mask[n_sys:]
    if not any(body_mask):
        raise ValueError(f"example {example['id']!r}: no supervised token in body")
    return RenderedSession(
        session_id=str(example["id"]),
        data_type=str(example["data_type"]),
        system_text="",                       # supplied by system_ids_by_key
        system_key=str(example["system_key"]),
        body_ids=body_ids, body_mask=body_mask, n_system_tokens=n_sys,
        source_id=str(example["source_session_id"]),
        candidate_index=int(example["truncation_index"]),
        candidate_sha256=None,
        meta={"n_rendered_tokens": n_sys + len(body_ids)},
    )


def pack_e5(examples: list[dict], system_ids_by_key: dict[str, list[int]], *,
            block_len: int = 8192, pad_id: int = 151643,
            target_blocks: int | None = None) -> list[PackedBlock]:
    """Pack E5 examples with truncation forbidden.

    `target_blocks` raises the block count to a common value shared by both arms.
    C and R pack to different minima — R's prefixes are longer — and the
    experiment needs one block count so that three passes give one optimizer-step
    budget. The easier-to-pack arm is therefore spread across more blocks,
    carrying additional ordinary padding.

    Spreading is done by **splitting** blocks that hold more than one example,
    never by duplicating an example, truncating one, or emitting a padding-only
    block: each half keeps the system prefix and a non-empty share of the
    examples, so every block still trains something.
    """
    rendered = [example_to_rendered(e) for e in examples]
    by_id = {r.session_id: r for r in rendered}
    blocks = pack_sessions(rendered, system_ids_by_key, block_len=block_len,
                           pad_id=pad_id, allow_terminal_truncation=False)
    if target_blocks is None:
        return blocks
    if target_blocks < len(blocks):
        raise ValueError(
            f"cannot pack into {target_blocks} blocks: {len(blocks)} is the "
            "minimum without truncating a sample")
    if target_blocks == len(blocks):
        return blocks

    # Groups of example ids, one per block. Splitting a group in two adds
    # exactly one block, so the target is reached exactly rather than overshot.
    groups: list[list[str]] = [[m["session_id"] for m in b.audit["sessions"]]
                               for b in blocks]
    keys: list[str] = [by_id[g[0]].system_key for g in groups]
    while len(groups) < target_blocks:
        # Split the largest splittable group; ties break on position so the
        # result is deterministic.
        idx = max((i for i, g in enumerate(groups) if len(g) > 1),
                  key=lambda i: (len(groups[i]), -i), default=None)
        if idx is None:
            raise ValueError(
                f"cannot reach {target_blocks} blocks: {len(groups)} is the "
                "maximum without duplicating examples (every block already "
                "holds a single example)")
        g = groups[idx]
        half = len(g) // 2
        groups[idx:idx + 1] = [g[:half], g[half:]]
        keys[idx:idx + 1] = [keys[idx], keys[idx]]

    out: list[PackedBlock] = []
    for g, key in zip(groups, keys):
        packed = pack_sessions([by_id[i] for i in g], {key: system_ids_by_key[key]},
                               block_len=block_len, pad_id=pad_id,
                               allow_terminal_truncation=False)
        if len(packed) != 1:
            raise RuntimeError(
                f"splitting produced {len(packed)} blocks for a group of "
                f"{len(g)} examples; expected exactly 1")
        out += packed
    if len(out) != target_blocks:
        raise RuntimeError(f"repack produced {len(out)} blocks, wanted {target_blocks}")
    return out


def write_pack(blocks: list[PackedBlock], out_dir: Path, *, arm: str, seed: str,
               block_len: int, pad_id: int, target_ce_tokens: int,
               extra: dict | None = None) -> dict:
    """Emit `blocks.npz`, `ladder.json` and `audit.jsonl` in the loader's contract.

    `ladder.json` declares a single rung covering every block, because an E5 pack
    is one fixed budget rather than a nested ladder. The loader only needs the
    rung it is asked for, so a one-rung ladder is a complete artifact and not a
    stub.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(blocks)
    ids = np.zeros((n, block_len), dtype=np.int32)
    ce = np.zeros((n, block_len), dtype=bool)
    content = np.zeros((n, block_len), dtype=bool)
    for i, b in enumerate(blocks):
        ids[i, :len(b.input_ids)] = np.asarray(b.input_ids, dtype=np.int32)
        ce[i, :len(b.ce_mask)] = np.asarray(b.ce_mask, dtype=bool)
        content[i, :len(b.content_mask)] = np.asarray(b.content_mask, dtype=bool)
    np.savez(out_dir / "blocks.npz", input_ids=ids, ce_mask=ce, content_mask=content)

    with (out_dir / "audit.jsonl").open("w") as f:
        for i, b in enumerate(blocks):
            row = dict(b.audit)
            row["block"] = i
            # The bundle each member belongs to, so atomicity stays verifiable
            # from the packed artifact alone.
            for m in row.get("sessions", []):
                m["bundle_id"] = m["session_id"].rsplit("#", 1)[0]
            f.write(json.dumps(row) + "\n")

    supervised = int(sum(b.n_supervised for b in blocks))
    real = int(content.sum())
    types: Counter = Counter()
    sessions_seen, bundles_seen = set(), set()
    truncations = 0
    for b in blocks:
        for m in b.audit.get("sessions", []):
            types[m["data_type"]] += int(m["supervised_retained"])
            sessions_seen.add(m["session_id"])
            bundles_seen.add(m["session_id"].rsplit("#", 1)[0])
            truncations += int(bool(m.get("truncated")))
    ladder = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "packing": "e5_prefix_continuation",
        "arm": arm, "seed": seed,
        "block_len": block_len, "pad_id": pad_id,
        "n_blocks": n, "n_sessions": len(sessions_seen),
        "n_bundles": len(bundles_seen),
        "terminal_truncations": truncations,
        "allow_terminal_truncation": False,
        "corpus_supervised_tokens": supervised,
        "corpus_type_mix": {k: round(v / max(1, supervised), 4)
                            for k, v in sorted(types.items())},
        "packing_efficiency": round(real / (n * block_len), 4) if n else 0.0,
        "real_tokens": real,
        "padding_tokens": n * block_len - real,
        "ordering": "paired stratified token-target selection",
        "block_ordering": "sequential within system-prompt group",
        "rungs": [{
            "target_supervised_tokens": target_ce_tokens,
            "reachable": True,
            "n_blocks": n,
            "actual_supervised_tokens": supervised,
            "n_sessions": len(sessions_seen),
            "real_tokens": real,
            "padding_tokens": n * block_len - real,
            "terminal_truncations": truncations,
            "token_mix": {k: round(v / max(1, supervised), 4)
                          for k, v in sorted(types.items())},
            "token_counts": dict(sorted(types.items())),
        }],
        **(extra or {}),
    }
    (out_dir / "ladder.json").write_text(json.dumps(ladder, indent=1))
    return ladder


def verify_pack(out_dir: Path, examples: list[dict], *, expected_blocks: int,
                target_ce_tokens: int, tolerance: float,
                steps: int, blocks_per_step: int) -> dict:
    """Prove the registered conditions against the ARTIFACTS, not estimates."""
    arrays = np.load(out_dir / "blocks.npz")
    ids, ce, content = arrays["input_ids"], arrays["ce_mask"], arrays["content_mask"]
    audit = [json.loads(l) for l in (out_dir / "audit.jsonl").open() if l.strip()]
    ladder = json.loads((out_dir / "ladder.json").read_text())
    n = int(ids.shape[0])

    per_example = {str(e["id"]): int(sum(bool(m) for m in e["mask"])) for e in examples}
    packed_supervised: Counter = Counter()
    bundles: dict[str, set] = defaultdict(set)
    for row in audit:
        for m in row["sessions"]:
            packed_supervised[m["session_id"]] += int(m["supervised_retained"])
            bundles[m["session_id"].rsplit("#", 1)[0]].add(m["session_id"])

    ce_tokens = int(ce.sum())
    checks = {
        "blocks_exact": n == expected_blocks,
        "n_blocks": n,
        "every_block_has_real_tokens": bool((content.sum(axis=1) > 0).all()),
        "every_block_has_supervision": bool((ce.sum(axis=1) > 0).all()),
        "no_terminal_truncation": ladder["terminal_truncations"] == 0
        and not any(m.get("truncated") for r in audit for m in r["sessions"]),
        "no_duplicate_examples": len(packed_supervised) == len(set(packed_supervised)),
        "all_bundles_complete": all(len(v) == 2 for v in bundles.values()),
        "n_bundles": len(bundles),
        "supervision_preserved": all(
            packed_supervised[k] == per_example.get(k, -1) for k in packed_supervised),
        "ce_mask_tokens": ce_tokens,
        "kd_mask_tokens": int(content.sum()),
        "nonpadding_tokens": int(content.sum()),
        "padding_tokens": int(n * ids.shape[1] - content.sum()),
        "packing_efficiency": round(float(content.sum()) / (n * ids.shape[1]), 4),
        "ce_target_relative_error": round(
            abs(ce_tokens - target_ce_tokens) / target_ce_tokens, 5),
        "passes": round(steps * blocks_per_step / n, 4) if n else 0.0,
        "three_passes_equal_registered_steps": n and steps * blocks_per_step == 3 * n,
    }
    checks["ce_target_within_tolerance"] = (
        checks["ce_target_relative_error"] <= tolerance)
    checks["packed_examples"] = len(packed_supervised)
    checks["all_examples_packed"] = len(packed_supervised) == len(examples)
    failures = [k for k in ("blocks_exact", "every_block_has_real_tokens",
                            "every_block_has_supervision", "no_terminal_truncation",
                            "no_duplicate_examples", "all_bundles_complete",
                            "supervision_preserved", "ce_target_within_tolerance",
                            "three_passes_equal_registered_steps",
                            "all_examples_packed")
                if not checks[k]]
    checks["failures"] = failures
    checks["passed"] = not failures
    return checks
