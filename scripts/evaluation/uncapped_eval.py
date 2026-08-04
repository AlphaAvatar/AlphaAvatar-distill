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


def _pct(values, q: float):
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None


def engine_config(llm, ctx: int, gpu_mem_util: float) -> dict:
    """The vLLM knobs that decide batching, read back from the live engine.

    Read rather than assumed: `max_num_seqs` and `max_num_batched_tokens` are
    never set by this script, so they are whatever the installed vLLM defaults
    to, and that default is what actually caps the batch. Recording them is the
    difference between "the evaluator batches" and "the evaluator batched 37
    sequences because the scheduler said so".
    """
    out = {"max_model_len": ctx, "gpu_memory_utilization": gpu_mem_util,
           "max_num_seqs": None, "max_num_batched_tokens": None,
           "vllm_version": None, "enforce_eager": None}
    try:
        import vllm
        out["vllm_version"] = getattr(vllm, "__version__", None)
    except Exception:
        pass
    for path in (("vllm_config", "scheduler_config"),
                 ("llm_engine", "vllm_config", "scheduler_config"),
                 ("llm_engine", "scheduler_config")):
        obj = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is None:
            continue
        for key in ("max_num_seqs", "max_num_batched_tokens"):
            if getattr(obj, key, None) is not None:
                out[key] = getattr(obj, key)
        break
    for path in (("vllm_config", "model_config"),
                 ("llm_engine", "model_config")):
        obj = llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            out["enforce_eager"] = getattr(obj, "enforce_eager", None)
            break
    return out


def gpu_sample() -> dict:
    """One nvidia-smi sample. Cheap, and it is the only utilization evidence."""
    import subprocess
    try:
        q = ("utilization.gpu,utilization.memory,memory.used,memory.total,"
             "power.draw,clocks.sm")
        raw = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        vals = [v.strip() for v in raw.split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:  # never let instrumentation break a paid run
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", required=True, nargs="+",
                    help="one or more prompt files. Several files are run through "
                         "ONE engine instance: loading the engine per set cost "
                         "~1.75 min each in Experiment 1, which with the "
                         "seven-set capability battery would be ~12 min of pure "
                         "init per checkpoint.")
    ap.add_argument("--out-dir", default=None,
                    help="with several prompt files, write <stem>.json here "
                         "instead of a single --out")
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
    # Additive: the defaults below reproduce the frozen behaviour exactly. They
    # exist so a reference model can also be run under ITS OWN chat protocol,
    # which is a different question from how our student is evaluated.
    ap.add_argument("--protocol", choices=("project", "native"), default="project",
                    help="'native' skips the project system injection and applies "
                         "the model's own template defaults")
    ap.add_argument("--chat-template-kwargs", default="{}",
                    help='JSON passed to apply_chat_template, e.g. '
                         '\'{"enable_thinking": true}\'')
    ap.add_argument("--revision", default=None,
                    help="pin the model revision (reference models)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu-sample-every", type=int, default=200,
                    help="scheduler steps between nvidia-smi samples")
    ap.add_argument("--diagnostics", action="store_true",
                    help="record engine settings, concurrency samples, effective "
                         "batch size and GPU utilization alongside the results")
    args = ap.parse_args()
    if not args.out and not args.out_dir:
        raise SystemExit("one of --out or --out-dir is required")
    if len(args.prompts) > 1 and not args.out_dir:
        raise SystemExit("--out-dir is required with several prompt files")

    from aadistill.evaluation import degeneration  # noqa: E402
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

    template_kwargs = json.loads(args.chat_template_kwargs)
    t_init = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=ctx,
              gpu_memory_utilization=args.gpu_mem_util,
              **({"revision": args.revision} if args.revision else {}))
    eng = llm.llm_engine
    init_seconds = round(time.time() - t_init, 1)
    engine_settings = engine_config(llm, ctx, args.gpu_mem_util)
    print(f"[{args.label}] engine ready in {init_seconds}s: {engine_settings}",
          flush=True)

    results = []
    for prompts_path in args.prompts:
        samples = [json.loads(l)
                   for l in Path(prompts_path).read_text().splitlines() if l.strip()]
        pending = {}
        for s in samples:
            turns = [m for m in s["messages"] if m["role"] != "assistant"]
            # A system message is mandatory (project protocol), but the sample's
            # own system prompt is PRESERVED when it has one — the rule the
            # corpus was generated under. Injecting unconditionally rendered two
            # `<|im_start|>system` turns for the 6 behaviour prompts that carry
            # their own, a context the model never saw in training.
            has_system = any(m.get("role") == "system" for m in turns)
            inject = args.system and not has_system and args.protocol == "project"
            if inject:
                turns = [{"role": "system", "content": args.system}] + turns
            prompt = tok.apply_chat_template(turns, tools=s.get("tools"),
                                             tokenize=False,
                                             add_generation_prompt=True,
                                             **template_kwargs)
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            allowance = ctx - len(p_ids)
            if allowance <= 0:
                raise SystemExit(
                    f"{s['id']}: prompt {len(p_ids)} leaves no room in {ctx}")
            rid = f"{args.label}::{Path(prompts_path).stem}::{s['id']}"
            # `detokenize=False` skips vLLM's incremental detokenization, which
            # this evaluator never reads — the record is decoded once from the
            # final token ids. It changes no sampling semantics.
            params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                                    max_tokens=allowance, stop_token_ids=stop_ids,
                                    detokenize=False)
            eng.add_request(rid, {"prompt_token_ids": p_ids}, params)
            pending[rid] = {"sample": s, "p_ids": p_ids, "allowance": allowance,
                            "system_source": ("sample" if has_system
                                              else ("default" if inject
                                                    else "none")),
                            "gen": [], "n_gen": 0, "last_check": 0, "degen": None,
                            "finish": None, "t0": time.time(), "ttft": None}
        t_wave = time.time()
        done, conc, step_s, gpu_samples = {}, [], [], []
        steps = 0
        t_prev = time.time()
        while eng.has_unfinished_requests():
            outs = eng.step()
            now = time.time()
            step_s.append(now - t_prev)
            t_prev = now
            steps += 1
            conc.append(len(outs))
            # Sample utilization *during* the wave, not after it. One sample at
            # the end says nothing about starvation while decoding.
            if args.diagnostics and steps % args.gpu_sample_every == 0:
                gpu_samples.append({"step": steps, **gpu_sample()})
            for out in outs:
                st = pending.get(out.request_id)
                if st is None:
                    continue
                # Track the LENGTH per step, not a copy of the whole token list.
                # The old line rebuilt every running request's full token list on
                # every scheduler step — O(sum of L^2) list copies on the decode
                # critical path. The tokens are materialised only when something
                # actually reads them: the degeneration check, or completion.
                st["n_gen"] = len(out.outputs[0].token_ids)
                if st["ttft"] is None and st["n_gen"]:
                    st["ttft"] = time.time() - st["t0"]
                if out.finished:
                    st["gen"] = list(out.outputs[0].token_ids)
                    st["finish"] = out.outputs[0].finish_reason
                    done[out.request_id] = st
                    pending.pop(out.request_id, None)
                    continue
                if (not args.no_degeneration_stop
                        and st["n_gen"] - st["last_check"] >= args.check_every):
                    st["last_check"] = st["n_gen"]
                    st["gen"] = list(out.outputs[0].token_ids)
                    d = degeneration.check(st["gen"])
                    if d:
                        st["degen"] = d
                        eng.abort_request(out.request_id)
                        st["finish"] = "degeneration"
                        done[out.request_id] = st
                        pending.pop(out.request_id, None)
        done.update(pending)
        wave_seconds = round(time.time() - t_wave, 1)

        records, scored = [], []
        for rid, st in done.items():
            s_ = st["sample"]
            gen = st["gen"]
            dt = time.time() - st["t0"]
            hit_ctx = len(gen) >= st["allowance"]
            ended_stop = bool(gen) and gen[-1] in stop_ids
            raw = tok.decode(gen, skip_special_tokens=False)
            reason = ("eos" if ended_stop else
                      "degeneration" if st["degen"] else
                      "context_limit" if hit_ctx else st["finish"] or "unknown")
            rec = {
                "id": s_["id"], "label": args.label, "group": s_.get("group"),
                "source": s_.get("source"), "prompt_tokens": len(st["p_ids"]),
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
            scored.append(score_sample(s_, raw, hit_cap=rec["context_limit_reached"]))

        records.sort(key=lambda r: r["id"])
        n = len(records)
        in_tok = sum(r["prompt_tokens"] for r in records)
        out_tok = sum(r["generated_tokens"] for r in records)
        lens = sorted(r["generated_tokens"] for r in records)
        summary = {
            "label": args.label, "model": args.model, "prompts": prompts_path,
            "n_samples": n,
            # "unrestricted within the model's effective context" — the allowance
            # is not a chosen token budget, but it is NOT the architectural 262K.
            "unrestricted_within_effective_context": True,
            "censored_measurement": False,
            "context_len": ctx, "context_resolution": ctx_info,
            "stop_ids": stop_ids,
            "system_message": args.system if args.protocol == "project" else None,
            "protocol": args.protocol,
            "chat_template_kwargs": template_kwargs,
            "model_revision": args.revision,
            "degeneration_stop": not args.no_degeneration_stop,
            "wave_seconds": wave_seconds,
            "throughput": {
                "input_tokens": in_tok, "output_tokens": out_tok,
                "output_tokens_p50": lens[n // 2],
                "output_tokens_p95": lens[min(n - 1, int(n * 0.95))],
                "output_tokens_max": lens[-1],
                "generation_wall_seconds": wave_seconds,
                "output_tokens_per_second": round(out_tok / wave_seconds, 1)
                if wave_seconds else None,
                "prompts_per_second": round(n / wave_seconds, 4)
                if wave_seconds else None,
                "scheduler_steps": steps,
                "seconds_per_step": round(wave_seconds / steps, 4) if steps else None,
                # Median, not just the mean: a few long prefill steps skew the
                # mean, and the gate is written against the median.
                "step_seconds_p50": round(_pct(step_s, 0.50), 5) if step_s else None,
                "step_seconds_p95": round(_pct(step_s, 0.95), 5) if step_s else None,
                "step_ms_p50": round(_pct(step_s, 0.50) * 1000, 2) if step_s else None,
                "concurrency_start": conc[0] if conc else 0,
                "concurrency_max": max(conc) if conc else 0,
                "effective_batch_size_mean": round(sum(conc) / len(conc), 2)
                if conc else None,
                "submission": ("all requests added to the engine before the first "
                               "step; vLLM schedules them with continuous "
                               "batching — not a serial loop"),
            },
            "engine": {**engine_settings, "init_seconds": init_seconds,
                       "init_amortised_over_sets": len(args.prompts)},
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
            "generated_tokens_p50": lens[n // 2],
            "generated_tokens_mean": round(out_tok / n, 1),
            "behavior": behavior_score(scored),
            "aggregate": aggregate(scored),
        }
        if args.diagnostics:
            summary["gpu"] = {"final": gpu_sample(), "during_wave": gpu_samples}
            utils = [int(g["utilization.gpu"]) for g in gpu_samples
                     if str(g.get("utilization.gpu", "")).isdigit()]
            if utils:
                summary["gpu"]["utilization_p50"] = _pct(sorted(utils), 0.50)
                summary["gpu"]["utilization_min"] = min(utils)
                summary["gpu"]["samples"] = len(utils)
        out = (Path(args.out) if args.out
               else Path(args.out_dir) / f"{Path(prompts_path).stem}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        with open(out.with_suffix(".generations.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results.append(summary)
        t = summary["throughput"]
        print(f"[{args.label}] {Path(prompts_path).stem}: {n} prompts, "
              f"{out_tok:,} out-tok in {wave_seconds}s = "
              f"{t['output_tokens_per_second']} tok/s, "
              f"{t['prompts_per_second']} prompts/s, "
              f"eff.batch {t['effective_batch_size_mean']}, "
              f"{t['seconds_per_step']}s/step", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
