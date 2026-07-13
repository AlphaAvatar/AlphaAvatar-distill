"""Build warm-up v1 for Stage 0 activation-statistics collection.

Usage:
    uv run python scripts/build_warmup_v1.py

Mixture (user-approved 2026-07-13, see logs/decisions.md): permissively
licensed public sources plus the 47 handcrafted v0 samples. Char budgets
approximate a ~1M-token total at ~4 chars/token; exact token counts are
logged later by the Stage 0 collection manifest.

Output:
    data/warmup/warmup_v1.jsonl           gitignored (third-party text)
    data/warmup/warmup_v1.manifest.json   committed reproducibility record

Determinism: each source is pinned to an exact dataset revision and read in
its native order; selection is "first N documents passing filters", so the
same script + manifest revisions reproduce the same file byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.manifest import sha256_file

# Per-sample cap: Stage 0 truncates at max_seq_len 1024 tokens (~4.2K chars),
# so longer text would be downloaded but never forwarded.
DOC_CHAR_CAP = 4500
DOC_CHAR_MIN = 200

SOURCES = [
    {
        "name": "fineweb_edu",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "license": "ODC-By 1.0",
        "category": "general_text_web",
        "char_budget": 2_400_000,
        "streaming": True,
    },
    {
        "name": "dolly",
        "dataset": "databricks/databricks-dolly-15k",
        "config": None,
        "split": "train",
        "license": "CC-BY-SA 3.0",
        "category": "instruction_chat",
        "char_budget": 800_000,
        "streaming": False,
    },
    {
        "name": "gsm8k",
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "license": "MIT",
        "category": "math_reasoning",
        "char_budget": 480_000,
        "streaming": False,
    },
    {
        "name": "mbpp",
        "dataset": "google-research-datasets/mbpp",
        "config": "full",
        "split": "train",
        "license": "CC-BY-4.0",
        "category": "code",
        "char_budget": 160_000,
        "streaming": False,
    },
]


def to_sample(source_name: str, idx: int, row: dict) -> dict | None:
    """Convert one source row into the warm-up jsonl schema (v0-compatible)."""
    sid = f"{source_name}-{idx:05d}"
    if source_name == "fineweb_edu":
        text = row["text"].strip()
        if len(text) < DOC_CHAR_MIN:
            return None
        return {"id": sid, "category": "general_text_web", "format": "text",
                "text": text[:DOC_CHAR_CAP]}
    if source_name == "dolly":
        user = row["instruction"].strip()
        if row.get("context", "").strip():
            user += "\n\n" + row["context"].strip()
        response = row["response"].strip()
        if not user or not response:
            return None
        return {"id": sid, "category": "instruction_chat", "format": "chat",
                "messages": [{"role": "user", "content": user[:DOC_CHAR_CAP]},
                             {"role": "assistant", "content": response[:DOC_CHAR_CAP]}]}
    if source_name == "gsm8k":
        return {"id": sid, "category": "math_reasoning", "format": "chat",
                "messages": [{"role": "user", "content": row["question"].strip()},
                             {"role": "assistant", "content": row["answer"].strip()}]}
    if source_name == "mbpp":
        tests = "\n".join(row.get("test_list", []))
        user = row["text"].strip()
        if tests:
            user += "\n\nYour code should pass these tests:\n" + tests
        return {"id": sid, "category": "code", "format": "chat",
                "messages": [{"role": "user", "content": user},
                             {"role": "assistant", "content": row["code"].strip()}]}
    raise ValueError(source_name)


def sample_chars(sample: dict) -> int:
    if sample["format"] == "text":
        return len(sample["text"])
    return sum(len(m["content"]) for m in sample["messages"])


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    out_path = REPO_ROOT / "data/warmup/warmup_v1.jsonl"
    v0_path = REPO_ROOT / "data/warmup/warmup_v0.jsonl"

    samples: list[dict] = []
    seen_hashes: set[str] = set()
    source_records = []

    for src in SOURCES:
        revision = api.dataset_info(src["dataset"]).sha
        print(f"[{src['name']}] {src['dataset']} @ {revision[:12]} "
              f"(budget {src['char_budget']:,} chars)", flush=True)
        ds = load_dataset(
            src["dataset"], src["config"], split=src["split"],
            revision=revision, streaming=src["streaming"],
        )
        chars = 0
        count = 0
        skipped_dup = 0
        for idx, row in enumerate(ds):
            if chars >= src["char_budget"]:
                break
            sample = to_sample(src["name"], idx, row)
            if sample is None:
                continue
            digest = hashlib.sha256(
                json.dumps(sample, sort_keys=True, ensure_ascii=False)
                .replace(sample["id"], "").encode()
            ).hexdigest()
            if digest in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(digest)
            samples.append(sample)
            chars += sample_chars(sample)
            count += 1
        source_records.append({
            **{k: src[k] for k in ("name", "dataset", "config", "split", "license")},
            "revision": revision,
            "samples": count,
            "chars": chars,
            "skipped_duplicates": skipped_dup,
        })
        print(f"[{src['name']}] took {count} samples, {chars:,} chars", flush=True)

    v0_samples = [json.loads(line) for line in v0_path.read_text().splitlines() if line.strip()]
    for s in v0_samples:
        s["id"] = f"v0-{s['id']}"
        samples.append(s)
    source_records.append({
        "name": "warmup_v0_handcrafted",
        "dataset": "data/warmup/warmup_v0.jsonl",
        "config": None, "split": None,
        "license": "project-authored (license-clean, see v0 record)",
        "revision": sha256_file(v0_path),
        "samples": len(v0_samples),
        "chars": sum(sample_chars(s) for s in v0_samples),
        "skipped_duplicates": 0,
    })

    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "warmup_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": "Stage 0 initialization warm-up statistics (not training data)",
        "doc_char_cap": DOC_CHAR_CAP,
        "doc_char_min": DOC_CHAR_MIN,
        "dedup": "exact content sha256 (id excluded)",
        "sources": source_records,
        "total_samples": len(samples),
        "total_chars": sum(sample_chars(s) for s in samples),
        "output": {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(out_path),
            "bytes": out_path.stat().st_size,
            "tracked_in_git": False,
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_path} ({manifest['total_samples']} samples, "
          f"{manifest['total_chars']:,} chars) and {manifest_path}")


if __name__ == "__main__":
    main()
