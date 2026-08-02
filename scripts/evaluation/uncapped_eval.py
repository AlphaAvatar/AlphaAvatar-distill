"""Uncapped (P18) behavioural evaluation on vLLM, batched, with a semantic stop.

Same measurement contract as `unrestricted_pilot.py` — allowance is
`context - prompt`, no artificial token budget, complete raw output kept for
every sample, and both non-EOS outcomes recorded as right-censored — but it
drives many requests through one engine pass instead of one at a time. The
pilot's concurrency of 1 exists to make paired waves comparable across
checkpoints; here the comparison is between checkpoints evaluated with
identical settings, and 25 checkpoints x ~176 prompts is not reachable serially.

Degeneration remains a *semantic* stop, not a token budget: a model in a
repetition loop never emits EOS, so the remaining context buys no information
beyond "it degenerated". Each request is checked independently every
`--check-every` new tokens and aborted on its own evidence, and the outcome is
recorded as its own class.

    python scripts/evaluation/uncapped_eval.py \
        --model <ckpt> --label <arm> \
        --prompts data/eval_behavior_v0/prompts.jsonl \
        --out artifacts/eval/<arm>_behavior_uncapped.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_context(cfg, override: int | None, trained_context: int | None) -> dict:
    """Resolve the effective context and return the derivation, not just a number.

    `config.max_position_embeddings` is 262,144 for this student because it
    inherits the Qwen3 geometry — but the student was trained exclusively on
    `block_len` 8,192 blocks and has never seen a position beyond that. Running
    it to 262k does not measure it more faithfully; it spends ~97% of the
    compute on a regime it was never trained for, and one wave cost an hour of
    L40S time when we tried (2026-08-02).

    So the *runtime effective* context is the trained context, and the
    derivation is recorded with every result. Output described this way must
    never be reported as a 262K-context evaluation.
    """
    arch_ctx = int(cfg.max_position_embeddings)
    scaling = getattr(cfg, "rope_scaling", None) or {}
    if isinstance(scaling, dict) and scaling.get("factor"):
        arch_ctx = int(arch_ctx * float(scaling["factor"]))
    if override:
        ctx, source = int(override), "cli:--context-len"
    elif trained_context:
        ctx, source = min(int(trained_context), arch_ctx), "trained_block_len"
    else:
        ctx, source = arch_ctx, "config:max_position_embeddings"
    return {"resolved_context": ctx, "context_source": source,
            "config_max_position_embeddings": int(cfg.max_position_embeddings),
            "rope_scaling_factor": (scaling or {}).get("factor"),
            "architectural_context": arch_ctx,
            "trained_context": trained_context,
            "note": ("effective context is the TRAINED context; this is not a "
                     "262K-context evaluation")
            if source == "trained_block_len" else None}


def resolve_stop_ids(model: str, tok, cfg) -> list[int]:
    """Native stopping semantics come from the model's own generation_config."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cfg_eos = getattr(cfg, "eos_token_id", None)
    cfg_eos = cfg_eos if isinstance(cfg_eos, (list, tuple)) else [cfg_eos]
    try:
        from transformers import GenerationConfig
        gen_eos = GenerationConfig.from_pretrained(model).eos_token_id
        gen_eos = gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos]
    except Exception:
        gen_eos = []
    return sorted({i for i in (*cfg_eos, *gen_eos, im_end, tok.eos_token_id)
                   if isinstance(i, int)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to the checkpoint's own tokenizer")
    ap.add_argument("--context-len", type=int, default=None,
                    help="explicit override; bypasses the trained-context rule")
    ap.add_argument("--trained-context", type=int, default=8192,
                    help="the block_len the student was trained on; the runtime "
                         "effective context is derived from this, not from the "
                         "architectural max_position_embeddings")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--check-every", type=int, default=256)
    ap.add_argument("--no-degeneration-stop", action="store_true")
    ap.add_argument("--system", default="You are a helpful Assistant.",
                    help="mandatory project system message (protocol, not a variable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
    import degeneration  # noqa: E402
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    from aadistill.evaluation.behavior import (  # noqa: E402
        aggregate, behavior_score, score_sample,
    )

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    ctx_info = resolve_context(cfg, args.context_len, args.trained_context)
    ctx = ctx_info["resolved_context"]
    stop_ids = resolve_stop_ids(args.model, tok, cfg)
    print(f"[{args.label}] effective context {ctx} "
          f"(source {ctx_info['context_source']}, architectural "
          f"{ctx_info['architectural_context']}); stop ids {stop_ids}", flush=True)

    samples = [json.loads(l) for l in Path(args.prompts).read_text().splitlines()
               if l.strip()]
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=ctx,
              gpu_memory_utilization=args.gpu_mem_util)
    eng = llm.llm_engine

    pending = {}
    for s in samples:
        turns = [m for m in s["messages"] if m["role"] != "assistant"]
        # A system message is mandatory (project protocol), but the sample's own
        # system prompt is PRESERVED when it has one — that is the rule the
        # corpus was generated under (`system_prompt_policy.source_prompt_
        # preserved`). Injecting unconditionally rendered two `<|im_start|>system`
        # turns for the 6 behaviour prompts that carry their own, which is a
        # context the model never saw in training.
        has_system = any(m.get("role") == "system" for m in turns)
        if args.system and not has_system:
            turns = [{"role": "system", "content": args.system}] + turns
        prompt = tok.apply_chat_template(turns, tools=s.get("tools"),
                                         tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        allowance = ctx - len(p_ids)
        if allowance <= 0:
            raise SystemExit(f"{s['id']}: prompt {len(p_ids)} leaves no room in {ctx}")
        rid = f"{args.label}::{s['id']}"
        params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                                max_tokens=allowance, stop_token_ids=stop_ids)
        eng.add_request(rid, {"prompt_token_ids": p_ids}, params)
        pending[rid] = {"sample": s, "p_ids": p_ids, "allowance": allowance,
                        "system_source": ("sample" if has_system else "default"),
                        "gen": [], "last_check": 0, "degen": None,
                        "finish": None, "t0": time.time(), "ttft": None}

    t_wave = time.time()
    done = {}
    while eng.has_unfinished_requests():
        for out in eng.step():
            st = pending.get(out.request_id)
            if st is None:
                continue
            st["gen"] = list(out.outputs[0].token_ids)
            if st["ttft"] is None and st["gen"]:
                st["ttft"] = time.time() - st["t0"]
            if out.finished:
                st["finish"] = out.outputs[0].finish_reason
                done[out.request_id] = st
                pending.pop(out.request_id, None)
                continue
            if (not args.no_degeneration_stop
                    and len(st["gen"]) - st["last_check"] >= args.check_every):
                st["last_check"] = len(st["gen"])
                d = degeneration.check(st["gen"])
                if d:
                    st["degen"] = d
                    eng.abort_request(out.request_id)
                    st["finish"] = "degeneration"
                    done[out.request_id] = st
                    pending.pop(out.request_id, None)
    done.update(pending)

    records, scored = [], []
    for rid, st in done.items():
        s = st["sample"]
        gen = st["gen"]
        dt = time.time() - st["t0"]
        hit_ctx = len(gen) >= st["allowance"]
        ended_stop = bool(gen) and gen[-1] in stop_ids
        raw = tok.decode(gen, skip_special_tokens=False)
        reason = ("eos" if ended_stop else
                  "degeneration" if st["degen"] else
                  "context_limit" if hit_ctx else st["finish"] or "unknown")
        rec = {
            "id": s["id"], "label": args.label, "group": s.get("group"),
            "source": s.get("source"), "prompt_tokens": len(st["p_ids"]),
            "system_source": st["system_source"],
            "context_len": ctx, "context_resolution": ctx_info,
            "generation_allowance": st["allowance"],
            "generated_tokens": len(gen), "stop_reason": reason,
            "degeneration_triggered": st["degen"] is not None,
            "degeneration_kind": (st["degen"] or {}).get("kind"),
            "natural_termination": reason == "eos",
            "context_limit_reached": reason == "context_limit",
            # Neither a context hit nor a degeneration abort observed the
            # model's own stopping decision, so both are right-censored.
            "right_censored": reason in ("context_limit", "degeneration"),
            "degeneration": st["degen"], "vllm_finish_reason": st["finish"],
            "ttft_seconds": round(st["ttft"], 3) if st["ttft"] else None,
            "seconds": round(dt, 2), "raw": raw,
        }
        records.append(rec)
        # `hit_cap` marks a censored generation for the scorer. Under P18 the
        # only cap is the real context, so it is set from that, never from a
        # chosen token budget.
        scored.append(score_sample(s, raw, hit_cap=rec["context_limit_reached"]))

    records.sort(key=lambda r: r["id"])
    n = len(records)
    summary = {
        "label": args.label, "model": args.model, "prompts": args.prompts,
        "n_samples": n,
        # "unrestricted within the model's effective context" — the allowance is
        # not a chosen token budget, but it is NOT the architectural 262K either.
        "unrestricted_within_effective_context": True,
        "censored_measurement": False,
        "context_len": ctx, "context_resolution": ctx_info,
        "stop_ids": stop_ids,
        "system_message": args.system,
        "degeneration_stop": not args.no_degeneration_stop,
        "wave_seconds": round(time.time() - t_wave, 1),
        "stop_reasons": {r: sum(1 for x in records if x["stop_reason"] == r)
                         for r in sorted({x["stop_reason"] for x in records})},
        "natural_termination_rate": round(
            sum(x["natural_termination"] for x in records) / n, 4),
        "degeneration_rate": round(
            sum(x["stop_reason"] == "degeneration" for x in records) / n, 4),
        "degeneration_kinds": {k: sum(1 for x in records
                                      if x.get("degeneration_kind") == k)
                               for k in ("cycle", "low_novelty", "rambling")},
        "context_limit_rate": round(
            sum(x["context_limit_reached"] for x in records) / n, 4),
        "right_censored_rate": round(
            sum(x["right_censored"] for x in records) / n, 4),
        "generated_tokens_p50": sorted(x["generated_tokens"] for x in records)[n // 2],
        "generated_tokens_mean": round(
            sum(x["generated_tokens"] for x in records) / n, 1),
        "behavior": behavior_score(scored),
        "aggregate": aggregate(scored),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    with open(out.with_suffix(".generations.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("aggregate",)}, indent=1)[:1200], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
