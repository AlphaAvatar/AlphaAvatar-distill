#!/usr/bin/env python3
"""Build the matched extra-KD control stream from unused in-domain rollout text.

E7's treatment adds general-text teacher KD. Two things could explain any effect
it has: **what FineWeb contains**, or simply **more KD positions, more gradient
signal and more compute**. Arm C separates them by supplying the identical extra
budget from the same teacher-rollout distribution the student already trains on.

Matched exactly, by construction rather than by tuning:

| quantity | how it is matched |
| --- | --- |
| extra KD positions | both streams are dense, so it is `n_blocks * (block_len - 1)` — identical integers |
| teacher/student forward tokens | same `n_blocks`, same `block_len` |
| CE positions | zero in both; the stream carries no loss mask at all |
| microbatch schedule | same `blocks_per_step`, `micro_blocks`, `every_n_steps` |
| optimizer steps | unchanged from the baseline in every arm |
| document-boundary policy | same `<|endoftext|>` separator, same dropped tail |

The source is the corpus the rollout rung is cut from, taken from the blocks
**after** the trained rung and **before** the validation tail — so it is
in-domain and unseen, and it touches neither the 1.60M supervised data nor
anything any rung validates on.

One interpretive caveat, recorded here because it bounds the claim: this control
is in-domain, and E6 showed that more in-domain data improves rollout stability.
Arm C is therefore a *strong* control — if it moves behaviour as much as arm B
does, the honest reading is "extra KD positions did it", not "FineWeb did
nothing special". That is the intended attribution, and it is why C is not a
neutral filler stream.

    python3 scripts/data/build_control_kd.py \\
        --pack artifacts/stage3/ladder_uniform --rung 1600000 \\
        --match artifacts/stage3/e7_fineweb_kd \\
        --out artifacts/stage3/e7_control_kd
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import (  # noqa: E402
    content_sha256, pack_dense, write_stream,
)
from aadistill.data.ladder import load_ladder_meta, rung_n_blocks  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402


def largest_rung_blocks(meta: dict) -> int:
    """Where the validation tail begins: past every declared reachable rung."""
    return max(int(r["n_blocks"]) for r in meta["rungs"] if r.get("reachable"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="token-ladder pack directory")
    ap.add_argument("--rung", type=int, required=True,
                    help="the trained rung, whose blocks are excluded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", default="",
                    help="an existing stream whose (n_blocks, block_len) this "
                         "must equal; the whole point of the arm")
    ap.add_argument("--n-blocks", type=int, default=0)
    ap.add_argument("--block-len", type=int, default=0)
    ap.add_argument("--separator-id", type=int, default=151643,
                    help="<|endoftext|> for the Qwen3 tokenizer")
    args = ap.parse_args()

    if args.match:
        target = json.loads((Path(args.match) / "manifest.json").read_text())
        n_blocks, block_len = int(target["n_blocks"]), int(target["block_len"])
        match_id = {"stream": args.match,
                    "blocks_sha256": target["outputs"]["blocks"],
                    "kd_positions": target["kd_positions"]}
    else:
        n_blocks, block_len = args.n_blocks, args.block_len
        match_id = None
    if n_blocks < 1 or block_len < 2:
        raise SystemExit("give --match, or both --n-blocks and --block-len")

    pack = Path(args.pack)
    if not pack.is_absolute():
        pack = REPO_ROOT / pack
    meta = load_ladder_meta(pack)
    trained = rung_n_blocks(meta, args.rung)
    val_start = largest_rung_blocks(meta)
    if trained >= val_start:
        raise SystemExit(
            f"rung {args.rung} uses {trained} blocks and the validation tail "
            f"starts at {val_start}; no unused in-domain region exists")

    z = np.load(pack / "blocks.npz")
    ids_all, content_all = z["input_ids"], z["content_mask"]

    # Real tokens only. A ladder block is 72% padding at this rung, so taking it
    # verbatim would either need padding (which breaks the dense contract) or
    # ~3.6x the forward workload for the same KD positions. Extracting the
    # content and re-packing densely is what makes the two arms comparable.
    segments: list[tuple[str, str, int, list[int]]] = []
    tokens_needed = n_blocks * block_len + block_len
    tokens = 0
    for b in range(trained, val_start):
        real = ids_all[b][content_all[b]].astype(np.int64).tolist()
        if not real:
            continue
        digest = content_sha256(json.dumps(real))
        segments.append((f"pack_block_{b}", digest, b, real))
        tokens += len(real) + 1
        if tokens >= tokens_needed:
            break
    if tokens < n_blocks * block_len:
        raise SystemExit(
            f"unused in-domain region [{trained}, {val_start}) yields "
            f"{tokens:,} tokens; {n_blocks * block_len:,} are needed to match")

    ids, content, rows = pack_dense(segments, block_len, args.separator_id,
                                    max_blocks=n_blocks)
    if ids.shape[0] != n_blocks:
        raise SystemExit(f"packed {ids.shape[0]} blocks, needed {n_blocks}")

    manifest = {
        "stream": Path(args.out).name,
        "kind": "in_domain_kd_control",
        "purpose": "train",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "source": {
            "pack": str(args.pack),
            "pack_blocks_sha256": sha256_file(pack / "blocks.npz"),
            "sessions_sha256": meta.get("sessions_sha256"),
            "trained_rung": args.rung,
            "excluded_trained_blocks": [0, trained],
            "excluded_validation_tail": [val_start, int(ids_all.shape[0])],
            "block_range_used": [trained, trained + len(segments)],
            "note": "content tokens of in-domain rollout blocks that the "
                    "trained rung does not use and no rung validates on",
        },
        "matched_to": match_id,
        "boundary_policy": {
            "separator_token": "<|endoftext|>",
            "separator_id": int(args.separator_id),
            "note": "identical policy to the FineWeb stream so the arms differ "
                    "in content and in nothing else",
        },
        "chat_template_applied": "inherited from the rollout corpus "
                                 "(these are already-rendered session tokens)",
        "assistant_ce_positions": 0,
        "code_state": code_state(str(REPO_ROOT)),
    }
    out = Path(args.out)
    full = write_stream(out if out.is_absolute() else REPO_ROOT / out,
                        ids, content, rows, manifest)

    if match_id and full["kd_positions"] != match_id["kd_positions"]:
        raise SystemExit(
            f"KD positions {full['kd_positions']} != target "
            f"{match_id['kd_positions']}; the arms would not be budget-matched")
    print(json.dumps({k: full[k] for k in
                      ("n_blocks", "block_len", "kd_positions", "total_tokens",
                       "padding_tokens", "n_documents", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
