#!/usr/bin/env python3
"""Fetch raw FineWeb-Edu documents from a pinned revision and index range.

Split out of `build_e8_calibration.py` on purpose. The download needs
`datasets`; the calibration build needs the *pinned* `transformers` whose chat
template renders the teacher-native sessions. Those live in different
environments on this dev box, and separating them also separates a network
operation from a deterministic build: the documents are written once, hashed, and
the build then has no network dependency at all.

    python3 scripts/data/fetch_fineweb_docs.py \\
        --start-index 40000 --max-docs 40 \\
        --out artifacts/stage1/e8_calibration_v1/general_docs.jsonl

Reserved consumption already on record — warmup_v1 (from index 0), holdout_v1
(skip 5,000), `e7_fineweb_val` [20000, 20454), `e7_fineweb_kd` [30000, 31902) —
so `--start-index` refuses anything below 40,000.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.extra_stream import content_sha256  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
LICENSE = "ODC-By 1.0"
RESERVED_END = 40000
DOC_CHAR_MIN = 500


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-index", type=int, default=RESERVED_END)
    ap.add_argument("--max-docs", type=int, default=40)
    ap.add_argument("--char-min", type=int, default=DOC_CHAR_MIN)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.start_index < RESERVED_END:
        raise SystemExit(
            f"--start-index {args.start_index} is inside the range already "
            f"consumed by warmup_v1 / holdout_v1 / the E7 streams "
            f"(reserved through {RESERVED_END})")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET).sha
    if revision != REVISION:
        raise SystemExit(
            f"{DATASET} is now revision {revision}, not the pinned {REVISION}; "
            "every earlier FineWeb artifact in this project reads the pinned one")

    ds = load_dataset(DATASET, CONFIG, split="train", revision=revision,
                      streaming=True)
    docs, last_index = [], args.start_index - 1
    for idx, row in enumerate(ds):
        if idx < args.start_index:
            continue
        if len(docs) >= args.max_docs:
            break
        last_index = idx
        text = row["text"].strip()
        if len(text) < args.char_min:
            continue
        docs.append({"id": f"fineweb-{idx}", "index": idx, "text": text,
                     "sha256": content_sha256(text)})
        if len(docs) % 10 == 0:
            print(f"  {len(docs)} docs, index {idx}", flush=True)
    if len(docs) < args.max_docs:
        raise SystemExit(f"stream ended with {len(docs)} of {args.max_docs} docs")

    out = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "dataset": DATASET, "config": CONFIG, "revision": revision,
        "license": LICENSE, "char_min": args.char_min,
        "index_range": [args.start_index, last_index + 1],
        "docs": len(docs), "chat_template_applied": False,
        "output": {"path": str(out), "sha256": sha256_file(out)},
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
