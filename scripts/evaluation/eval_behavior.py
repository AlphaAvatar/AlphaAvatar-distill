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


def supported_context(model, tokenizer) -> int:
    """The model's actual supported context, resolved and never guessed (P18).

    Taken from the model config's `max_position_embeddings`, adjusted by any RoPE
    scaling factor, and floored by the tokenizer's own limit when it declares a
    smaller one. This is deliberately NOT lowered for memory: P18 requires that a
    memory constraint be met by reducing batch size or changing hardware, not by
    silently shrinking the measurement window.
    """
    cfg = model.config
    ctx = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    scaling = getattr(cfg, "rope_scaling", None) or {}
    factor = scaling.get("factor") if isinstance(scaling, dict) else None
    if factor:
        ctx = int(ctx * float(factor))
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 10**7:
        ctx = min(ctx, tok_max) if ctx else tok_max
    if ctx <= 0:
        raise ValueError("could not resolve a supported context length")
    return ctx


@torch.no_grad()
def generate(model, tokenizer, entry: dict, max_new_tokens: int | None, device: str,
             context_len: int | None = None):
    """Greedy-decode one prompt.

    `max_new_tokens=None` means **unrestricted** (P18): the allowance is
    `context_len - len(rendered_prompt)`, so generation stops only on the
    model's own EOS / `<|im_end|>` or at the actual context limit. A stop at the
    context limit is a right-censored observation, not a failure.

    Returns (raw_text, n_new, hit_limit, allowance).
    """
    text = tokenizer.apply_chat_template(
        entry["messages"],
        tools=entry.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_len = int(ids.shape[1])
    if max_new_tokens is None:
        if context_len is None:
            raise ValueError("unrestricted generation needs a context length")
        allowance = context_len - prompt_len
        if allowance <= 0:
            raise ValueError(
                f"prompt of {prompt_len} tokens leaves no room in a "
                f"{context_len}-token context")
    else:
        allowance = max_new_tokens
    out = model.generate(
        ids.to(device),
        max_new_tokens=allowance,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    n_new = out.shape[1] - ids.shape[1]
    raw = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    return raw, n_new, n_new >= allowance, allowance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="data/eval_behavior_v0/prompts.jsonl")
    ap.add_argument("--out", required=True)
    # A fixed cap is NOT permitted for formal measurement (AGENTS.md P18): it
    # censors exactly the behaviour being measured. `--unrestricted` derives the
    # allowance per sample from the actual supported context. The numeric flag
    # is retained only for reproducing historical capped runs and for cheap
    # smoke tests, and it is recorded in the manifest as a censored measurement.
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--unrestricted", action="store_true",
                    help="P18: generate to natural EOS or the actual context "
                         "limit; no artificial token budget")
    ap.add_argument("--context-len", type=int, default=None,
                    help="override the resolved supported context (recorded)")
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
    # Resolve the measurement window once, and record where it came from.
    if args.context_len:
        ctx_len, ctx_source = args.context_len, "cli:--context-len"
    else:
        ctx_len = supported_context(model, tokenizer)
        ctx_source = "config:max_position_embeddings(+rope_scaling, tokenizer floor)"
    if args.unrestricted:
        print(f"P18 unrestricted: context {ctx_len}, allowance = context - prompt",
              flush=True)
    else:
        print(f"CENSORED measurement: max_new_tokens={args.max_new_tokens} "
              f"(context {ctx_len}). Not valid for natural-termination claims.",
              flush=True)

    quant_summary = None
    if args.fake_quant == "int8":
        from aadistill.models.quant import int8_fake_quantize_

        quant_summary = int8_fake_quantize_(model, scope=args.fake_quant_scope)
        print(f"fake-quant int8 scope={quant_summary['scope']}: "
              f"{quant_summary['n_linear_quantized']} linears", flush=True)

    started = time.time()
    generations, scored = [], []
    for i, entry in enumerate(entries, 1):
        raw, n_new, hit_limit, allowance = generate(
            model, tokenizer, entry,
            None if args.unrestricted else args.max_new_tokens,
            device, context_len=ctx_len)
        parts = split_generation(raw)
        # Under P18 a stop at the context limit is right-censored, not a cap
        # hit: `context_limit_reached` is the honest label and the two must not
        # be conflated when reading `terminated` / `empty_answer`.
        generations.append({
            "id": entry["id"], "group": entry["group"], "source": entry["source"],
            "raw": raw, "answer": parts["answer"], "think": parts["think"],
            "new_tokens": n_new,
            "truncated_at_cap": hit_limit and not args.unrestricted,
            "context_limit_reached": hit_limit and args.unrestricted,
            "right_censored": hit_limit,
            "generation_allowance": allowance,
            "stop_reason": ("context_limit" if (hit_limit and args.unrestricted)
                            else "cap" if hit_limit else "eos"),
            "gold_answer": entry.get("gold_answer", ""),
        })
        scored.append(score_sample(entry, raw, hit_cap=hit_limit))
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
            "greedy": True,
            # P18: `unrestricted` means the allowance came from the resolved
            # context minus each rendered prompt, so nothing but the model's own
            # EOS or the context limit stopped generation. A capped run is
            # recorded as a censored measurement so it can never be mistaken for
            # a natural-termination measurement later.
            "unrestricted": bool(args.unrestricted),
            "censored_measurement": not bool(args.unrestricted),
            "max_new_tokens": None if args.unrestricted else args.max_new_tokens,
            "context_len_resolved": ctx_len,
            "context_len_source": ctx_source,
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
