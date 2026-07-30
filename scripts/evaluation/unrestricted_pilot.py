"""P18-compliant unrestricted generation pilot.

Generates until the model emits its native EOS / `<|im_end|>` or reaches the
**actual** supported context. No artificial token budget, no wall-clock cutoff,
concurrency 1, complete raw output captured for every sample.

The per-sample allowance is `context_len - len(rendered_prompt)`, with
`context_len` resolved from the model config (and any RoPE scaling) rather than
chosen. A sample that stops at that boundary is `context_limit_reached` and is
**right-censored**, not a failure (AGENTS.md P18).

One checkpoint per invocation = one "paired wave" over the same prompt ids, so
the caller can enforce a budget between waves without ever interrupting an
active generation.

Usage (on a vLLM pod):
    python scripts/evaluation/unrestricted_pilot.py \
        --model <ckpt-dir> --label ttb_treat_a \
        --prompt-ids "a,b,c" --out artifacts/pilot/ttb_treat_a.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--targets", default="artifacts/stage2_v2/teacher_corpus_750/targets.jsonl")
    ap.add_argument("--prompt-ids", required=True, help="comma/space separated")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--revision", default="768f209d9ea81521153ed38c47d515654e938aea")
    ap.add_argument("--context-len", type=int, default=None,
                    help="override; default resolves from the model config")
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    # Degeneration is a SEMANTIC stop, not a token budget: a model in a
    # repetition loop never emits EOS, so the full context buys no information
    # beyond "it degenerated". Recorded as its own class with evidence, never
    # conflated with natural termination or a context-limit hit.
    ap.add_argument("--no-degeneration-stop", action="store_true")
    ap.add_argument("--check-every", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
    import degeneration  # noqa: E402
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    cfg = AutoConfig.from_pretrained(args.model)
    ctx = args.context_len or int(cfg.max_position_embeddings)
    scaling = getattr(cfg, "rope_scaling", None) or {}
    if isinstance(scaling, dict) and scaling.get("factor"):
        ctx = int(ctx * float(scaling["factor"]))
    # Native stopping semantics come from the MODEL's generation_config, which
    # may list several eos ids (the teacher lists <|im_end|> AND <|endoftext|>).
    # Taking only the tokenizer's single eos would silently narrow the stop set
    # and turn a natural termination into a non-termination.
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cfg_eos = getattr(cfg, "eos_token_id", None)
    cfg_eos = cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos]
    try:
        from transformers import GenerationConfig
        gen_eos = GenerationConfig.from_pretrained(args.model).eos_token_id
        gen_eos = gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos]
    except Exception:
        gen_eos = []
    stop_ids = sorted({i for i in (*cfg_eos, *gen_eos, im_end, tok.eos_token_id)
                       if isinstance(i, int)})
    print(f"[{args.label}] resolved context {ctx}; stop ids {stop_ids}", flush=True)

    ids = [x for x in args.prompt_ids.replace(",", " ").split() if x]
    rows = {}
    for line in (REPO_ROOT / args.targets).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["id"] in ids:
                rows[r["id"]] = r
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"prompt ids not found: {missing}")

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=ctx,
              gpu_memory_utilization=args.gpu_mem_util, enforce_eager=False)

    out = []
    wave_start = time.time()
    for n, sid in enumerate(ids, 1):
        r = rows[sid]
        user_turns = [m for m in r["messages"] if m["role"] != "assistant"]
        prompt = tok.apply_chat_template(
            user_turns, tools=r.get("tools"), tokenize=False,
            add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        allowance = ctx - len(p_ids)
        if allowance <= 0:
            raise SystemExit(f"{sid}: prompt {len(p_ids)} tok leaves no room in {ctx}")

        # Greedy, concurrency 1, allowance = the entire remaining context.
        params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                                max_tokens=allowance, stop_token_ids=stop_ids)
        t0 = time.time()
        gen_ids, degen, ttft = [], None, None
        if args.no_degeneration_stop:
            res = llm.generate([{"prompt_token_ids": p_ids}], params, use_tqdm=False)[0]
            gen_ids = list(res.outputs[0].token_ids)
            finish = res.outputs[0].finish_reason
        else:
            # Stream so the content can be judged while it is produced; abort the
            # request the moment it stops producing new information.
            eng = llm.llm_engine
            rid = f"{args.label}-{sid}"
            eng.add_request(rid, {"prompt_token_ids": p_ids}, params)
            finish, last_check = None, 0
            while eng.has_unfinished_requests():
                for step_out in eng.step():
                    if step_out.request_id != rid:
                        continue
                    gen_ids = list(step_out.outputs[0].token_ids)
                    if ttft is None and gen_ids:
                        ttft = time.time() - t0
                    if step_out.finished:
                        finish = step_out.outputs[0].finish_reason
                if finish is not None:
                    break
                if len(gen_ids) - last_check >= args.check_every:
                    last_check = len(gen_ids)
                    degen = degeneration.check(gen_ids)
                    if degen:
                        eng.abort_request(rid)
                        finish = "degeneration"
                        break
        dt = time.time() - t0
        n_new = len(gen_ids)
        hit_ctx = n_new >= allowance
        ended_stop = bool(gen_ids) and gen_ids[-1] in stop_ids
        raw = tok.decode(gen_ids, skip_special_tokens=False)
        stop_reason = ("eos" if ended_stop else
                       "degeneration" if degen else
                       "context_limit" if hit_ctx else finish or "unknown")
        rec = {
            "id": sid, "label": args.label, "group": r["group"], "source": r["source"],
            "prompt_tokens": len(p_ids), "context_len": ctx,
            "generation_allowance": allowance,
            "generated_tokens": n_new,
            "stop_reason": stop_reason,
            "natural_termination": stop_reason == "eos",
            "context_limit_reached": stop_reason == "context_limit",
            # Both non-EOS outcomes are right-censored: neither observed the
            # model's own stopping decision.
            "right_censored": stop_reason in ("context_limit", "degeneration"),
            "degeneration": degen,
            "vllm_finish_reason": finish,
            "ttft_seconds": round(ttft, 3) if ttft else None,
            "seconds": round(dt, 2),
            "decode_tok_per_s": round(n_new / dt, 1) if dt > 0 else None,
            "raw": raw,
            "token_ids": gen_ids,
        }
        out.append(rec)
        print(f"[{args.label}] {n}/{len(ids)} {sid}: {n_new} tok in {dt:.0f}s "
              f"({rec['decode_tok_per_s']} tok/s) stop={stop_reason}", flush=True)

    payload = {
        "label": args.label, "model": args.model,
        "context_len": ctx, "stop_token_ids": stop_ids,
        "decoding": {"greedy": True, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                     "concurrency": 1, "unrestricted": True,
                     "max_tokens": "context_len - prompt_len (per sample)"},
        "wave_seconds": round(time.time() - wave_start, 1),
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                            if torch.cuda.is_available() else None),
        "samples": out,
    }
    p = REPO_ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    nat = sum(s["natural_termination"] for s in out)
    deg = sum(1 for s in out if s["stop_reason"] == "degeneration")
    ctx_hit = sum(s["context_limit_reached"] for s in out)
    print(f"[{args.label}] WAVE DONE: {nat} natural / {deg} degenerated / "
          f"{ctx_hit} context-limited of {len(out)}, "
          f"{payload['wave_seconds']}s -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
