"""Build the small held-out eval set for the Stage 1 initial evaluation.

Usage:
    uv run python scripts/build_holdout_v1.py

Streams fineweb-edu documents starting well past the range consumed by
warmup_v1 (which took the first ~860 stream positions), so held-out text was
never seen by the Stage 0 statistics. Same revision pinning and manifest
scheme as the warm-up builder.

Output:
    data/warmup/holdout_v1.jsonl           gitignored
    data/warmup/holdout_v1.manifest.json   committed
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.manifest import sha256_file

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
SKIP_DOCS = 5000   # far beyond warmup_v1's consumption window
NUM_DOCS = 40
DOC_CHAR_CAP = 4500
DOC_CHAR_MIN = 500


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, CONFIG, split="train", revision=revision, streaming=True)

    out_path = REPO_ROOT / "data/warmup/holdout_v1.jsonl"
    samples = []
    for idx, row in enumerate(ds):
        if idx < SKIP_DOCS:
            continue
        text = row["text"].strip()
        if len(text) < DOC_CHAR_MIN:
            continue
        samples.append({"id": f"holdout-{idx:05d}", "category": "general_text_web",
                        "format": "text", "text": text[:DOC_CHAR_CAP]})
        if len(samples) >= NUM_DOCS:
            break

    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "holdout_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": "Held-out perplexity eval for Stage 1 gate (never in warmup_v1)",
        "source": {"dataset": DATASET, "config": CONFIG, "split": "train",
                   "revision": revision, "license": "ODC-By 1.0"},
        "skip_docs": SKIP_DOCS,
        "num_docs": len(samples),
        "doc_char_cap": DOC_CHAR_CAP,
        "doc_char_min": DOC_CHAR_MIN,
        "output": {"path": str(out_path.relative_to(REPO_ROOT)),
                   "sha256": sha256_file(out_path),
                   "bytes": out_path.stat().st_size,
                   "tracked_in_git": False},
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_path} ({len(samples)} docs) and {manifest_path}")


if __name__ == "__main__":
    main()
