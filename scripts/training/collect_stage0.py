"""Stage 0: collect teacher activation statistics for student initialization.

Usage:
    uv run python scripts/collect_stage0.py --config configs/stage0_qwen3_4b_thinking.json [--limit N]

Reads a JSON config, forwards the warm-up dataset through the teacher one
sequence at a time, and writes:

    <output_dir>/activation_stats.safetensors   streaming sufficient statistics
    <output_dir>/manifest.json                  full reproducibility record

The run is CPU/GPU-portable: device is auto-detected unless pinned in config.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.collect import ActivationStatsCollector, residual_covariance
from aadistill.env import code_state, hardware_report, set_determinism
from aadistill.manifest import sha256_file, sha256_json, write_manifest
from aadistill.teacher import load_teacher


def load_warmup_dataset(path: Path) -> list[dict]:
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError(f"Empty warm-up dataset: {path}")
    for s in samples:
        if s["format"] not in ("text", "chat"):
            raise ValueError(f"Sample {s['id']}: unknown format {s['format']!r}")
    return samples


def encode_sample(sample: dict, tokenizer, max_seq_len: int) -> torch.Tensor:
    if sample["format"] == "chat":
        out = tokenizer.apply_chat_template(
            sample["messages"], add_generation_prompt=False, return_tensors="pt"
        )
        # transformers 5.x may return a BatchEncoding rather than a bare tensor.
        ids = out["input_ids"] if not isinstance(out, torch.Tensor) else out
    else:
        ids = tokenizer(sample["text"], return_tensors="pt").input_ids
    return ids[:, :max_seq_len]


def determinism_check(model, input_ids: torch.Tensor) -> dict:
    """Forward the same sequence twice and compare logits (AGENTS.md P5)."""
    with torch.no_grad():
        a = model(input_ids.to(model.device)).logits
        b = model(input_ids.to(model.device)).logits
    max_abs_diff = (a - b).abs().max().item()
    return {"max_abs_logit_diff": max_abs_diff, "bitwise_identical": max_abs_diff == 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N samples (dry run)")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    config_hash = sha256_json(config)
    set_determinism(config["seed"])
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = REPO_ROOT / config["dataset"]
    samples = load_warmup_dataset(dataset_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    print(f"Loading teacher {config['teacher_model_id']} on {device} ...", flush=True)
    model, tokenizer, identity = load_teacher(
        config["teacher_model_id"],
        revision=config.get("teacher_revision"),
        dtype=config["dtype"],
        device=device,
    )

    first_ids = encode_sample(samples[0], tokenizer, config["max_seq_len"])
    det = determinism_check(model, first_ids)
    print(f"Determinism check: {det}", flush=True)

    collector = ActivationStatsCollector(model)
    started = time.time()
    per_sample = []
    for i, sample in enumerate(samples):
        ids = encode_sample(sample, tokenizer, config["max_seq_len"])
        n_tokens = collector.process(ids)
        per_sample.append({"id": sample["id"], "category": sample["category"],
                           "tokens": n_tokens})
        print(f"[{i + 1}/{len(samples)}] {sample['id']} "
              f"({sample['category']}, {n_tokens} tokens)", flush=True)
    collector.close()
    wall_clock_s = time.time() - started

    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "activation_stats.safetensors"
    cache_meta = collector.save(str(stats_path))
    cache_bytes = stats_path.stat().st_size
    budget_bytes = int(config["cache_budget_gb"] * 1e9)
    if cache_bytes > budget_bytes:
        raise RuntimeError(
            f"Cache {cache_bytes} bytes exceeds logged budget {budget_bytes} bytes"
        )

    # Projection dry run: prove the cache is consumable by Stage 1 (gate item).
    from safetensors.torch import load_file
    state = load_file(str(stats_path))
    mean, cov = residual_covariance(state, point=identity["num_hidden_layers"] // 2)
    eigvals = torch.linalg.eigvalsh(cov)
    projection_dry_run = {
        "point": identity["num_hidden_layers"] // 2,
        "mean_norm": mean.norm().item(),
        "top_eigenvalue": eigvals[-1].item(),
        "min_eigenvalue": eigvals[0].item(),
        "rank_proxy_evals_above_1e-6_of_top": int(
            (eigvals > 1e-6 * eigvals[-1]).sum()
        ),
    }
    print(f"Projection dry run (mid layer): {projection_dry_run}", flush=True)

    manifest = {
        "stage": "stage0_init_warmup_collection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config": config,
        "config_sha256": config_hash,
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
        "teacher": identity,
        "dataset": {
            "path": str(dataset_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(dataset_path),
            "num_samples": len(samples),
            "samples": per_sample,
        },
        "determinism_check": det,
        "wall_clock_seconds": round(wall_clock_s, 1),
        "cache": {
            "path": str(stats_path.relative_to(REPO_ROOT)),
            "bytes": cache_bytes,
            "sha256": sha256_file(stats_path),
            "budget_gb": config["cache_budget_gb"],
            **cache_meta,
        },
        "projection_dry_run": projection_dry_run,
    }
    write_manifest(output_dir / "manifest.json", manifest)
    print(f"Wrote {output_dir / 'manifest.json'}")
    print(f"Cache: {cache_bytes / 1e9:.2f} GB, "
          f"{cache_meta['tokens_processed']} tokens, {wall_clock_s:.0f}s")


if __name__ == "__main__":
    main()
