#!/usr/bin/env python3
"""Boot vLLM once and read what the engine ACTUALLY uses. Runs in /opt/vllm.

Stage 0 needs the observed half of `RecoveryGenerationProtocolFingerprint`, and
`max_num_seqs` / `max_num_batched_tokens` are never set by `uncapped_eval.py` —
they are whatever the installed vLLM defaults to, and that default is what caps
the batch. Recording them is the difference between "the evaluator batches" and
"the evaluator batched 37 sequences because the scheduler said so".

Booting here rather than at first use also converts "the engine works on this
image" into a Stage-0 blocking gate, ahead of $2.80 of permanent controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/evaluation"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--trained-context", type=int, default=8192)
    ap.add_argument("--image-digest", default=None,
                    help="the pod image digest; defaults to AADISTILL_IMAGE_DIGEST")
    args = ap.parse_args()

    import os

    import vllm
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    from aadistill.autoinit.generation import generation_runtime_fingerprint

    from uncapped_eval import engine_config, resolve_context, resolve_stop_ids

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    ctx_info = resolve_context(cfg, None, args.trained_context)
    ctx = ctx_info["resolved_context"]
    stop_ids = resolve_stop_ids(args.model, tok, cfg, None)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=ctx,
              gpu_memory_utilization=args.gpu_mem_util)
    engine = engine_config(llm, ctx, args.gpu_mem_util)

    tok_files = sorted(Path(args.model).glob("tokenizer*.json"))
    tok_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in tok_files)).hexdigest() if tok_files else None
    # The runtime the ROLLOUTS execute under — this interpreter, not the
    # trainer's. Stage 0 used to fill the generation protocol's torch and
    # transformers versions from the training venv, which describes a stack that
    # never generates a token and could never match what an evaluation wave
    # observes.
    runtime = generation_runtime_fingerprint(
        args.image_digest or os.environ.get("AADISTILL_IMAGE_DIGEST") or None)
    out = {
        "runtime": runtime.as_dict(),
        "runtime_digest": runtime.digest,
        "torch_version": runtime.torch_version,
        "transformers_version": runtime.transformers_version,
        "dtype": engine.get("dtype"),
        "vllm_version": getattr(vllm, "__version__", None) or engine.get("vllm_version"),
        "max_num_seqs": engine.get("max_num_seqs"),
        "max_num_batched_tokens": engine.get("max_num_batched_tokens"),
        "enforce_eager": engine.get("enforce_eager"),
        "gpu_memory_utilization": args.gpu_mem_util,
        "resolved_context": ctx,
        "context_source": ctx_info["context_source"],
        "context_resolution": ctx_info,
        "stop_token_ids": stop_ids,
        "tokenizer_sha256": tok_hash,
        "tokenizer_files": [p.name for p in tok_files],
        "chat_template_sha256": hashlib.sha256(
            (tok.chat_template or "").encode()).hexdigest(),
        "engine": engine,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("vllm_version", "max_num_seqs", "resolved_context",
                       "stop_token_ids")}, indent=2))


if __name__ == "__main__":
    main()
