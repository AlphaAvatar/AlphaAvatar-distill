"""Stage 2 gate check: loader dry run over the built offline mixture.

Usage:
    uv run python scripts/dry_run_stage2.py \
        [--data-dir data/stage2] [--block-len 1024] \
        [--tokenizer Qwen/Qwen3-4B-Thinking-2507@<revision>] \
        [--out artifacts/stage2/dry_run_v0_report.json]

Validates every sample against the loader schema, encodes every split with
the teacher tokenizer (chat template rendering + assistant-token loss masks),
packs each train group into fixed-length blocks, and re-encodes a probe
subset to confirm determinism. Fails loudly on any schema, rendering,
masking, or determinism error — a clean exit is the "small-batch dry run
passes" evidence for the Stage 2 validation gate (AGENTS.md 4.4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data import SPLITS, encode_sample, load_split, pack_blocks
from aadistill.env import code_state, hardware_report
from aadistill.manifest import sha256_file
from aadistill.teacher import tokenizer_hash

DEFAULT_TOKENIZER = (
    "Qwen/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea"
)
DETERMINISM_PROBES = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/stage2")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--block-len", type=int, default=1024)
    parser.add_argument("--out", default="artifacts/stage2/dry_run_v0_report.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model_id, _, revision = args.tokenizer.partition("@")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision or None)

    data_dir = REPO_ROOT / args.data_dir
    report_splits: dict[str, dict] = {}
    checks: dict[str, bool] = {}
    started = time.time()

    for split in SPLITS:
        groups = load_split(data_dir, split)  # validates schema of every sample
        report_splits[split] = {}
        for group, samples in groups.items():
            t0 = time.time()
            encoded = [encode_sample(tokenizer, s) for s in samples]
            tokens = sum(len(ids) for ids, _ in encoded)
            trainable = sum(sum(m) for _, m in encoded)
            record = {
                "samples": len(samples),
                "tokens": tokens,
                "trainable_tokens": trainable,
                "trainable_frac": round(trainable / tokens, 4),
            }
            if split == "train":
                _, _, dropped = pack_blocks(encoded, args.block_len)
                record["blocks"] = tokens // args.block_len
                record["packed_dropped_tail_tokens"] = dropped
                probes = samples[:DETERMINISM_PROBES]
                again = [encode_sample(tokenizer, s) for s in probes]
                if again != encoded[: len(probes)]:
                    raise RuntimeError(f"non-deterministic encoding in {group}")
            report_splits[split][group] = record
            print(f"[{split}/{group}] {record} ({time.time() - t0:.1f}s)", flush=True)

    # Group-specific sanity checks on the train split.
    train_groups = load_split(data_dir, "train")
    tool_encoded = [encode_sample(tokenizer, s) for s in train_groups["tool_calling"][:50]]
    tool_spans = [
        tokenizer.decode([t for t, m in zip(ids, mask) if m])
        for ids, mask in tool_encoded
    ]
    checks["tool_call_format_rendered"] = any("<tool_call>" in s for s in tool_spans)
    checks["every_group_has_masked_context"] = all(
        rec["trainable_frac"] < 1.0 or group == "long_context"
        for group, rec in report_splits["train"].items()
    )
    checks["deterministic_encoding"] = True  # raised above otherwise
    if not all(checks.values()):
        raise RuntimeError(f"dry-run checks failed: {checks}")

    manifest_path = data_dir / "stage2_offline_v0.manifest.json"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "data_dir": args.data_dir,
        "mixture_manifest_sha256": sha256_file(manifest_path),
        "tokenizer": {"model_id": model_id, "revision": revision or None,
                      "sha256": tokenizer_hash(tokenizer)},
        "block_len": args.block_len,
        "splits": report_splits,
        "checks": checks,
        "seconds": round(time.time() - started, 1),
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nAll checks passed: {checks}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
