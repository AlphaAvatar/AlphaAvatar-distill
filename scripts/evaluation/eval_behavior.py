"""Score a checkpoint on `eval_behavior_v0` — deterministic greedy generation
plus the mechanical scorers in `aadistill.behavior`.

    uv run python scripts/evaluation/eval_behavior.py \
        --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model \
        --out artifacts/stage3/s1_ffn_norm_v0/eval_behavior_v0.json

Writes a scorecard JSON (aggregates + per-sample scores + full reproducibility
metadata) and, next to it, the raw generations as jsonl. Greedy decoding, batch
size 1, fixed max_new_tokens: no sampling, so re-running on the same checkpoint
and hardware reproduces the file. `--fake-quant int8` scores the INT8
deployment target (P9) with the same path `eval_ppl.py` uses.

Run this at every recovery gate next to holdout_v1 and the INT8 evals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation.behavior import aggregate, score_sample, split_generation
from aadistill.models.teacher import load_causal_lm
from aadistill.infrastructure.env import code_state, hardware_report
from aadistill.infrastructure.manifest import sha256_file


@torch.no_grad()
def generate(model, tokenizer, entry: dict, max_new_tokens: int, device: str):
    """Greedy-decode one prompt. Returns (raw_text, n_new_tokens, hit_cap)."""
    text = tokenizer.apply_chat_template(
        entry["messages"],
        tools=entry.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    out = model.generate(
        ids.to(device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    n_new = out.shape[1] - ids.shape[1]
    raw = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    return raw, n_new, n_new >= max_new_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="data/eval_behavior_v0/prompts.jsonl")
    ap.add_argument("--out", required=True)
    # 512 keeps the cap off the critical path: at 200, every single
    # non-termination in the s2_blocks_v1 baseline was a cap hit, which made
    # `terminated` a verbosity measure rather than a format one.
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fake-quant", choices=["int8"], default=None)
    ap.add_argument("--fake-quant-scope", choices=["all", "decoder"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N prompts")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    prompts_path = REPO_ROOT / args.prompts
    # Iterate the file handle, not `splitlines()`: the latter also splits on
    # \v, \f and \u2028, which appear in LaTeX-heavy prompts.
    with open(prompts_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        entries = entries[: args.limit]

    model, tokenizer = load_causal_lm(args.model, dtype, device)
    quant_summary = None
    if args.fake_quant == "int8":
        from aadistill.models.quant import int8_fake_quantize_

        quant_summary = int8_fake_quantize_(model, scope=args.fake_quant_scope)
        print(f"fake-quant int8 scope={quant_summary['scope']}: "
              f"{quant_summary['n_linear_quantized']} linears", flush=True)

    started = time.time()
    generations, scored = [], []
    for i, entry in enumerate(entries, 1):
        raw, n_new, hit_cap = generate(
            model, tokenizer, entry, args.max_new_tokens, device)
        parts = split_generation(raw)
        generations.append({
            "id": entry["id"], "group": entry["group"], "source": entry["source"],
            "raw": raw, "answer": parts["answer"], "think": parts["think"],
            "new_tokens": n_new, "truncated_at_cap": hit_cap,
            "gold_answer": entry.get("gold_answer", ""),
        })
        scored.append(score_sample(entry, raw, hit_cap=hit_cap))
        if i % 10 == 0 or i == len(entries):
            elapsed = time.time() - started
            print(f"  {i}/{len(entries)} prompts  {elapsed:.0f}s "
                  f"({elapsed / i:.1f}s/prompt)", flush=True)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path = out_path.with_suffix(".generations.jsonl")
    with open(gen_path, "w") as f:
        for g in generations:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    agg = aggregate(scored)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "eval": "eval_behavior_v0",
        "model": args.model,
        "prompts": {
            "path": args.prompts,
            "sha256": sha256_file(prompts_path),
            "count": len(entries),
        },
        "decoding": {
            "greedy": True, "max_new_tokens": args.max_new_tokens,
            "batch_size": 1, "dtype": args.dtype, "device": device,
        },
        "fake_quant": quant_summary,
        "seconds": round(time.time() - started, 1),
        "aggregate": agg,
        "per_sample": scored,
        "generations_file": str(
            gen_path.relative_to(REPO_ROOT)
            if gen_path.is_relative_to(REPO_ROOT)
            else gen_path
        ),
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    o = agg["overall"]
    print(f"\n{args.model}  ({report['seconds']}s)")
    print(f"  format_ok        {o['format_ok']:.3f}   terminated {o['terminated']:.3f}   "
          f"truncated_at_cap {o['truncated_at_cap']:.3f}   "
          f"think_closed {o['think_closed']:.3f}   think_immediate {o['think_immediate']:.3f}")
    print(f"  empty_answer     {o['empty_answer']:.3f}   answer_is_echo {o['answer_is_echo']:.3f}   "
          f"echo_4gram {o['echo_4gram']:.3f}   rep_3gram {o['rep_3gram']:.3f}   "
          f"answer_words {o['answer_words']:.1f}")
    for g, s in agg["by_group"].items():
        extra = " ".join(
            f"{k}={s[k]:.3f}" for k in
            ("evidence_hit", "evidence_hit_credited", "refusal", "refusal_credited",
             "tool_call_parsed", "tool_name_valid", "tool_args_schema_ok",
             "tool_call_exact_match", "answer_em", "answer_em_credited")
            if k in s
        )
        print(f"  {g:22s} format_ok={s['format_ok']:.3f} echo={s['echo_4gram']:.3f} {extra}")
    print(f"Wrote {out_path}\nWrote {gen_path}")


if __name__ == "__main__":
    main()
