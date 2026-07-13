"""Token-level NLL / perplexity evaluation on a warm-up-format jsonl.

Usage:
    uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
        --model artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
        [--model Qwen/Qwen3-4B-Thinking-2507@<revision> ...] \
        [--max-seq-len 1024] [--out <report.json>]

Each --model is a local checkpoint dir or an HF id (optionally @revision).
Reports mean next-token NLL (nats/token) and perplexity per model, teacher
included, so the Stage 1 gate has an initial evaluation with baselines.
Deterministic: fixed data order, no sampling, batch size 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.env import code_state, hardware_report
from aadistill.manifest import sha256_file


def load_model(spec: str, dtype: torch.dtype, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (REPO_ROOT / spec).exists():
        path, revision = str(REPO_ROOT / spec), None
    elif "@" in spec:
        path, revision = spec.split("@", 1)
    else:
        path, revision = spec, None
    model = AutoModelForCausalLM.from_pretrained(
        path, revision=revision, dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    return model, tokenizer


@torch.no_grad()
def mean_nll(model, tokenizer, samples: list[dict], max_seq_len: int, device: str):
    total_nll, total_tokens = 0.0, 0
    for s in samples:
        if s["format"] != "text":
            raise ValueError(f"eval_ppl only supports text samples, got {s['id']}")
        ids = tokenizer(s["text"], return_tensors="pt").input_ids[:, :max_seq_len]
        if ids.shape[1] < 2:
            continue
        logits = model(ids.to(device)).logits.float()
        nll = F.cross_entropy(
            logits[0, :-1], ids[0, 1:].to(device), reduction="sum")
        total_nll += nll.item()
        total_tokens += ids.shape[1] - 1
    return total_nll / total_tokens, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    data_path = REPO_ROOT / args.data
    samples = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]

    results = []
    for spec in args.model:
        print(f"Evaluating {spec} ...", flush=True)
        started = time.time()
        model, tokenizer = load_model(spec, dtype, device)
        nll, n_tokens = mean_nll(model, tokenizer, samples, args.max_seq_len, device)
        results.append({
            "model": spec,
            "mean_nll_nats": round(nll, 4),
            "perplexity": round(float(torch.tensor(nll).exp()), 2),
            "eval_tokens": n_tokens,
            "seconds": round(time.time() - started, 1),
        })
        print(f"  nll={nll:.4f} ppl={results[-1]['perplexity']} "
              f"({n_tokens} tokens, {results[-1]['seconds']}s)", flush=True)
        del model

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "data": {"path": args.data, "sha256": sha256_file(data_path),
                 "num_samples": len(samples)},
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "device": device,
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
        "results": results,
    }
    print(json.dumps(results, indent=2))
    if args.out:
        out_path = REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
