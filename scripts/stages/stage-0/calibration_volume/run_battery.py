#!/usr/bin/env python3
"""Run a procedural task set through one vLLM engine: static tasks and episodes.

    /opt/vllm/bin/python scripts/stages/stage-0/calibration_volume/run_battery.py \
        --model <dir or hub id> --label <name> --tasks <set.jsonl> \
        --mode eval --trained-context 8192 --out <records.jsonl>

Two modes, one engine loop:

* ``eval`` -- a MEASUREMENT (AGENTS.md P18). Greedy decoding, the frozen
  sampling of `shared.evaluation.uncapped_eval`, and a per-turn allowance of
  `context - prompt` where the context comes from that module's
  `resolve_context` (the trained-context rule for a student, the configured
  context for a model with no `--trained-context`). No token budget.
* ``generate`` -- DATA CONSTRUCTION for teacher-native calibration sessions. The
  teacher's officially recommended sampling preset, one seeded request per turn,
  and a recorded session limit (`--session-limit`) that is a data-construction
  bound, never a measurement budget; a session that reaches it is recorded as
  `length_limited`, never trimmed.

A static task is an episode of one turn. An interactive task is stepped by its
environment (`tasks.interactive`): every active episode's next turn is
generated in one engine pass, the environment reads the reply's final answer,
and its observation becomes the next user turn. Earlier assistant turns are
rendered with their final answer only -- the chat template drops earlier
reasoning -- so the scored turn is always the last one, exactly as in training.

The degeneration stop is the same semantic stop `uncapped_eval` uses; it is
recorded as its own outcome and is right-censored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from stages.calibration_volume.tasks.interactive import make_env, max_turns  # noqa: E402

#: The teacher's officially recommended preset (the corpus_v2 build's), used
#: only in `generate` mode.
GENERATE_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


def request_seed(base: int, task_id: str, turn: int) -> int:
    h = hashlib.sha256(f"{base}|{task_id}|{turn}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31)


def run_episodes(tasks, *, tok, eng, make_sampling, limit, ctx, stop_ids, mode,
                 seed_base, system, check_every, degeneration_check,
                 split_generation, template_opens_think, label, log=print):
    """Generate every task's turns through `eng`; returns (episodes, rounds, tokens).

    `eng` is a vLLM engine (add_request / step / has_unfinished_requests /
    abort_request) and `make_sampling(max_tokens, seed)` builds its sampling
    params; both are passed in so the episode logic can be exercised without a
    GPU.
    """
    episodes = {}
    for t in tasks:
        env = make_env(t["env"]) if t.get("interactive") else None
        episodes[t["task_id"]] = {
            "task": t, "env": env, "history": [{"role": "system", "content": system},
                                              *t["messages"]],
            "turns": [], "done": False, "end": None,
            "max_turns": max_turns(t["env"]) if env else 1}

    rnd, out_tokens = 0, 0
    t_gen = time.time()
    while True:
        active = [e for e in episodes.values() if not e["done"]]
        if not active:
            break
        rnd += 1
        pending = {}
        for e in active:
            tid = e["task"]["task_id"]
            prompt = tok.apply_chat_template(e["history"], tokenize=False,
                                             add_generation_prompt=True)
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            allowance = limit - len(p_ids)
            if allowance <= 0:
                e["done"], e["end"] = True, "context_exhausted"
                continue
            seed = request_seed(seed_base, tid, len(e["turns"])) if mode == "generate" else None
            sp = make_sampling(allowance, seed)
            rid = f"{tid}::{len(e['turns'])}"
            eng.add_request(rid, {"prompt_token_ids": p_ids}, sp)
            pending[rid] = {"ep": e, "p_ids": p_ids, "allowance": allowance,
                            "preopened": template_opens_think(prompt),
                            "gen": [], "n": 0, "last": 0, "degen": None,
                            "finish": None, "t0": time.time(), "ttft": None,
                            "seed": seed}
        done = {}
        while eng.has_unfinished_requests():
            for out in eng.step():
                st = pending.get(out.request_id)
                if st is None or out.request_id in done:
                    continue
                st["n"] = len(out.outputs[0].token_ids)
                if st["ttft"] is None and st["n"]:
                    st["ttft"] = time.time() - st["t0"]
                if out.finished:
                    st["gen"] = list(out.outputs[0].token_ids)
                    st["finish"] = out.outputs[0].finish_reason
                    done[out.request_id] = st
                    continue
                if st["n"] - st["last"] >= check_every:
                    st["last"] = st["n"]
                    st["gen"] = list(out.outputs[0].token_ids)
                    d = degeneration_check(st["gen"])
                    if d:
                        st["degen"], st["finish"] = d, "degeneration"
                        eng.abort_request(out.request_id)
                        done[out.request_id] = st
        for rid, st in {**pending, **done}.items():
            e, gen = st["ep"], st["gen"]
            out_tokens += len(gen)
            raw = tok.decode(gen, skip_special_tokens=False)
            hit = len(gen) >= st["allowance"]
            eos = bool(gen) and gen[-1] in stop_ids
            reason = ("eos" if eos else "degeneration" if st["degen"] else
                      ("context_limit" if limit == ctx else "length_limited")
                      if hit else st["finish"] or "unknown")
            parts = split_generation(raw, think_preopened=st["preopened"])
            turn = {
                "turn": len(e["turns"]), "prompt_tokens": len(st["p_ids"]),
                "generation_allowance": st["allowance"], "generated_tokens": len(gen),
                "stop_reason": reason, "natural_termination": reason == "eos",
                "context_limit_reached": reason == "context_limit",
                "length_limited": reason == "length_limited",
                "right_censored": reason in ("context_limit", "degeneration"),
                "degeneration_triggered": st["degen"] is not None,
                "degeneration_kind": (st["degen"] or {}).get("kind"),
                "think_preopened": st["preopened"], "seed": st["seed"],
                "ttft_seconds": round(st["ttft"], 3) if st["ttft"] else None,
                "seconds": round(time.time() - st["t0"], 2),
                "raw": raw, "answer": parts["answer"],
                "prompt_token_ids": st["p_ids"] if mode == "generate" else None,
                "output_token_ids": gen,
            }
            e["turns"].append(turn)
            e["history"].append({"role": "assistant", "content": parts["answer"],
                                 "reasoning_content": parts.get("think", "")})
            if reason != "eos":
                e["done"], e["end"] = True, reason
                continue
            if e["env"] is None:
                e["done"], e["end"] = True, "answered"
                continue
            obs = e["env"].step(parts["answer"])
            turn["observation"] = obs
            if obs is None or e["env"].done:
                e["done"], e["end"] = True, "episode_done"
            elif len(e["turns"]) >= e["max_turns"]:
                e["done"], e["end"] = True, "turn_limit"
            else:
                e["history"].append({"role": "user", "content": obs})
        log(f"[{label}] round {rnd}: {len(pending)} turns, "
              f"{sum(not e['done'] for e in episodes.values())} episodes active, "
              f"{out_tokens:,} tokens, {time.time() - t_gen:.0f}s", flush=True)

    return episodes, rnd, out_tokens


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--mode", required=True, choices=("eval", "generate"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trained-context", type=int, default=None,
                    help="a student's training block length; omit for a model "
                         "whose configured context is its supported context")
    ap.add_argument("--session-limit", type=int, default=None,
                    help="generate mode only: the data-construction bound")
    ap.add_argument("--seed-base", type=int, default=20261018)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--check-every", type=int, default=256)
    ap.add_argument("--system", default="You are a helpful Assistant.")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks (smoke)")
    args = ap.parse_args(argv)
    if args.mode == "eval" and args.session_limit:
        raise SystemExit("--session-limit is a data-construction bound; a measurement "
                         "has none (P18)")

    from aadistill.evaluation import degeneration
    from aadistill.evaluation.behavior import split_generation, template_opens_think
    from aadistill.infrastructure.env import library_versions
    from shared.evaluation.uncapped_eval import (SAMPLING, engine_config,
                                                 resolve_context, resolve_stop_ids,
                                                 tokenizer_files_sha256)
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    rev = {"revision": args.revision} if args.revision else {}
    tok = AutoTokenizer.from_pretrained(args.model, **rev)
    cfg = AutoConfig.from_pretrained(args.model, **rev)
    ctx_info = resolve_context(cfg, None, args.trained_context)
    ctx = ctx_info["resolved_context"]
    stop_ids = resolve_stop_ids(args.model, tok, cfg, args.revision)
    limit = min(ctx, args.session_limit) if args.session_limit else ctx
    sampling = dict(SAMPLING) if args.mode == "eval" else dict(GENERATE_SAMPLING)
    print(f"[{args.label}] mode={args.mode} context={ctx} "
          f"({ctx_info['context_source']}) limit={limit} stop={stop_ids}", flush=True)

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[:args.limit]

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=ctx,
              gpu_memory_utilization=args.gpu_mem_util, **rev)
    eng = llm.llm_engine
    init_s = round(time.time() - t0, 1)
    engine = engine_config(llm, ctx, args.gpu_mem_util)
    print(f"[{args.label}] engine ready in {init_s}s", flush=True)

    def make_sampling(max_tokens, seed):
        params = {k: v for k, v in sampling.items() if k != "detokenize"}
        if seed is not None:
            params["seed"] = seed
        return SamplingParams(max_tokens=max_tokens, stop_token_ids=stop_ids,
                              detokenize=False, **params)

    t_gen = time.time()
    episodes, rnd, out_tokens = run_episodes(
        tasks, tok=tok, eng=eng, make_sampling=make_sampling, limit=limit, ctx=ctx,
        stop_ids=stop_ids, mode=args.mode, seed_base=args.seed_base,
        system=args.system, check_every=args.check_every,
        degeneration_check=degeneration.check, split_generation=split_generation,
        template_opens_think=template_opens_think, label=args.label)
    gen_s = round(time.time() - t_gen, 1)
    identity = {
        "label": args.label, "model": args.model, "revision": args.revision,
        "mode": args.mode, "tasks_file": args.tasks,
        "tasks_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
        "context_resolution": ctx_info, "session_limit": args.session_limit,
        "effective_limit": limit, "stop_ids": stop_ids, "sampling": sampling,
        "seed_base": args.seed_base if args.mode == "generate" else None,
        "system_message": args.system,
        "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
        "tokenizer_sha256": tokenizer_files_sha256(args.model),
        "degeneration_check_every": args.check_every,
        "engine": {**engine, "init_seconds": init_s},
        "libraries": library_versions(),
        "image_digest": os.environ.get("AADISTILL_IMAGE_DIGEST"),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for tid, e in episodes.items():
            t = e["task"]
            rec = {"task_id": tid, "label": args.label, "mode": args.mode,
                   "generator_id": t["generator_id"], "family_id": t["family_id"],
                   "axis": t["axis"], "difficulty": t["difficulty"],
                   "interactive": t["interactive"], "end": e["end"],
                   "turns": e["turns"],
                   "env_result": e["env"].result() if e["env"] else None,
                   "messages": e["history"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {"identity": identity, "n_tasks": len(episodes), "rounds": rnd,
               "output_tokens": out_tokens, "generation_seconds": gen_s,
               "output_tokens_per_second": round(out_tokens / gen_s, 1) if gen_s else None,
               "ends": {k: sum(e["end"] == k for e in episodes.values())
                        for k in sorted({e["end"] for e in episodes.values()})}}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[{args.label}] done: {len(episodes)} tasks, {out_tokens:,} tokens in "
          f"{gen_s}s; ends {summary['ends']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
