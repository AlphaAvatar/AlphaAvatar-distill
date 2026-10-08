#!/usr/bin/env python3
"""Build a dense FineWeb-Edu KD stream (E7 treatment, or its validation set).

Raw general text, exactly as the source publishes it. No user/assistant
fabrication, no chat template, no assistant CE — the student is asked to match
the teacher's next-token distribution on ordinary prose, which is the thing the
rollout recipe appears to destroy: held-out FineWeb NLL falls to 6.16 by the
0.46M rung and then climbs back to ~9.7 by 1.60M (`e1_consolidated.json`).

Everything that identifies the slice is pinned and recorded: dataset, config,
split, revision, the exact stream index range consumed, the filter, the
tokenizer hash, the document-boundary policy, and a sha256 per document. The
index range says which rows were read; the per-document hashes are what
`check_stream_disjointness.py` actually proves disjointness with, because two
different ranges can still deliver the same text.

    python3 scripts/data/build_fineweb_kd.py \\
        --out artifacts/stage3/e7_fineweb_kd --n-blocks 1761 --block-len 1024 \\
        --start-index 20000

Reserved ranges, and why the default start is 20,000:

| consumer | range | note |
| --- | --- | --- |
| `warmup_v1` Stage 0 statistics | from index 0, 848 docs kept under a char budget | the exact last index read was not recorded, so the whole prefix is treated as consumed |
| `holdout_v1` Stage 1 gate | `skip_docs=5000`, 40 docs kept | the historical FineWeb holdout, preserved for continuity |

Neither reaches 20,000, and the gap is deliberately far larger than either
window so that a re-run of an older builder cannot collide with E7.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import (  # noqa: E402
    content_sha256, pack_dense, write_stream,
)
from aadistill.infrastructure.env import code_state  # noqa: E402

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
SPLIT = "train"
# Pinned in `data/warmup/{warmup_v1,holdout_v1}.manifest.json`; asserted below
# rather than trusted, so a silent upstream re-tag cannot change what E7 trains
# on while the manifest still claims this revision.
EXPECTED_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
LICENSE = "ODC-By 1.0"

# Historical consumption, treated as fully reserved.
RESERVED_PREFIX_END = 20000

DOC_CHAR_MIN = 500      # same floor as holdout_v1
DOC_CHAR_CAP = None     # deliberately uncapped: dense packing needs whole prose


def tokenizer_identity(tokenizer) -> str:
    from aadistill.models.teacher import tokenizer_hash
    return tokenizer_hash(tokenizer)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-blocks", type=int, required=True)
    ap.add_argument("--block-len", type=int, default=1024)
    ap.add_argument("--start-index", type=int, default=RESERVED_PREFIX_END)
    ap.add_argument("--tokenizer",
                    default="artifacts/stage1/qwen3_0p6b_init_v0/checkpoint")
    ap.add_argument("--purpose", default="train",
                    choices=("train", "validation"))
    ap.add_argument("--kind", default="general_text_kd")
    ap.add_argument("--exclude-hashes", action="append", default=[],
                    help="docs.jsonl or *.jsonl whose content hashes must not "
                         "appear in this stream")
    ap.add_argument("--max-docs", type=int, default=200000,
                    help="hard stop so a bad filter cannot stream forever")
    args = ap.parse_args()

    if args.start_index < RESERVED_PREFIX_END:
        raise SystemExit(
            f"--start-index {args.start_index} is inside the reserved prefix "
            f"[0, {RESERVED_PREFIX_END}) consumed by warmup_v1 and holdout_v1. "
            "Overlapping them would put Stage 0 statistics or the historical "
            "holdout into E7 training data.")

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    revision = HfApi().dataset_info(DATASET).sha
    if revision != EXPECTED_REVISION:
        raise SystemExit(
            f"{DATASET} now resolves to {revision}, not the pinned "
            f"{EXPECTED_REVISION}. Every earlier FineWeb artifact in this "
            "project was built at the pinned revision; refusing to mix.")

    tok = AutoTokenizer.from_pretrained(REPO_ROOT / args.tokenizer)
    sep_id = tok.convert_tokens_to_ids("<|endoftext|>")
    if sep_id is None or sep_id < 0:
        raise SystemExit("tokenizer has no <|endoftext|> for the document "
                         "separator; boundaries must be explicit")

    excluded: set[str] = set()
    for path in args.exclude_hashes:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            digest = row.get("sha256") or content_sha256(row.get("text", ""))
            excluded.add(digest)
    print(f"excluding {len(excluded)} document hash(es) from "
          f"{len(args.exclude_hashes)} file(s)")

    # Enough tokens to fill n_blocks, plus headroom so the last block is complete.
    need_tokens = args.n_blocks * args.block_len + args.block_len
    ds = load_dataset(DATASET, CONFIG, split=SPLIT, revision=revision,
                      streaming=True)

    docs: list[tuple[str, str, int, list[int]]] = []
    seen: set[str] = set()
    tokens_so_far = 0
    skipped_short = skipped_dup = skipped_excluded = 0
    last_index = args.start_index - 1
    for idx, row in enumerate(ds):
        if idx < args.start_index:
            continue
        if idx - args.start_index >= args.max_docs:
            break
        last_index = idx
        text = row.get("text") or ""
        if len(text) < DOC_CHAR_MIN:
            skipped_short += 1
            continue
        digest = content_sha256(text)
        if digest in excluded:
            skipped_excluded += 1
            continue
        if digest in seen:
            skipped_dup += 1
            continue
        seen.add(digest)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        docs.append((str(row.get("id", f"idx{idx}")), digest, idx, ids))
        tokens_so_far += len(ids) + 1
        if tokens_so_far >= need_tokens:
            break
        if len(docs) % 500 == 0:
            print(f"  {len(docs)} docs, {tokens_so_far:,}/{need_tokens:,} tokens",
                  flush=True)

    if tokens_so_far < args.n_blocks * args.block_len:
        raise SystemExit(
            f"only {tokens_so_far:,} tokens available for "
            f"{args.n_blocks * args.block_len:,} needed; raise --max-docs")

    ids, content, rows = pack_dense(docs, args.block_len, sep_id,
                                    max_blocks=args.n_blocks)
    if ids.shape[0] != args.n_blocks:
        raise SystemExit(f"packed {ids.shape[0]} blocks, asked for {args.n_blocks}")

    manifest = {
        "stream": Path(args.out).name,
        "kind": args.kind,
        "purpose": args.purpose,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "source": {
            "dataset": DATASET, "config": CONFIG, "split": SPLIT,
            "revision": revision, "license": LICENSE,
            "start_index": args.start_index, "last_index_read": last_index,
            "index_range": [args.start_index, last_index + 1],
        },
        "filter": {"doc_char_min": DOC_CHAR_MIN, "doc_char_cap": DOC_CHAR_CAP,
                   "skipped_short": skipped_short,
                   "skipped_duplicate_content": skipped_dup,
                   "skipped_excluded": skipped_excluded},
        "dedup": "exact content sha256 within the stream and against "
                 "--exclude-hashes",
        "boundary_policy": {
            "separator_token": "<|endoftext|>", "separator_id": int(sep_id),
            "note": "documents are concatenated with an explicit separator; the "
                    "trailing partial block is dropped so the stream is fully "
                    "dense and its KD budget is exactly n_blocks*(block_len-1)",
        },
        "chat_template_applied": False,
        "assistant_ce_positions": 0,
        "tokenizer": {"path": args.tokenizer, "sha256": tokenizer_identity(tok)},
        "code_state": code_state(str(REPO_ROOT)),
    }
    full = write_stream(REPO_ROOT / args.out if not Path(args.out).is_absolute()
                        else args.out, ids, content, rows, manifest)
    print(json.dumps({k: full[k] for k in
                      ("n_blocks", "block_len", "kd_positions", "total_tokens",
                       "padding_tokens", "n_documents", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
