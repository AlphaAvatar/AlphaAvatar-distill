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


def read_tasks(path) -> list[dict]:
    """The --tasks JSON Lines, split on "\n" only: str.splitlines() also breaks at U+0085, U+2028 and
    U+2029, which an ensure_ascii=False input carries raw (a GSM8K problem holding U+2028 failed s01n of
    a_supplement_20261031_092108 before the engine loaded)."""
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]


def run_episodes(tasks, *, tok, eng, make_sampling, limit, ctx, stop_ids, mode,
                 seed_base, system, check_every, degeneration_check,
                 split_generation, template_opens_think, label, log=print,
                 action_protocol="v1", on_done=None, seed_of=None, template_kwargs=None):
    """Generate every task's turns through `eng`; returns (episodes, turns, tokens).

    ASYNCHRONOUS: one engine pass for everything. A finished turn is processed
    at once -- its episode's environment is stepped and the next turn submitted
    -- so no episode waits for another's slowest turn. `make_sampling` must
    return DELTA output (each step carries only the new tokens), and each
    request accumulates its own tokens: a cumulative output copies every running
    request's whole token list on every scheduler step.

    `eng` is a vLLM engine (add_request / step / has_unfinished_requests /
    abort_request) and `make_sampling(max_tokens, seed)` builds its sampling
    params; both are passed in so the episode logic can be exercised without a
    GPU.

    `on_done(task_id, episode)` is called once per episode, the moment it ends
    (on every ending: answered, episode done, turn limit, a non-eos stop,
    context exhausted), so a caller can persist finished work before the whole
    batch ends.

    `seed_of(task_id, turn)` gives a measurement's per-turn seed under a
    decoding protocol; without it only `generate` mode seeds its requests.

    `template_kwargs` go to every `apply_chat_template` call (a hybrid
    template's `enable_thinking`). Each turn records the `think_state` its
    rendered prompt left the continuation in, and is split under it.
    """
    from aadistill.evaluation.behavior import template_think_state

    template_kwargs = dict(template_kwargs or {})
    episodes = {}
    for t in tasks:
        env = (make_env(t["env"], action_protocol=action_protocol)
               if t.get("interactive") else None)
        episodes[t["task_id"]] = {
            "task": t, "env": env, "history": [{"role": "system", "content": system},
                                              *t["messages"]],
            "turns": [], "done": False, "end": None,
            "max_turns": max_turns(t["env"]) if env else 1}

    pending: dict[str, dict] = {}
    n_turns, out_tokens = 0, 0
    t_gen = time.time()
    last_log = time.time()

    def close(e, end) -> None:
        e["done"], e["end"] = True, end
        if on_done is not None:
            on_done(e["task"]["task_id"], e)

    def submit(e) -> None:
        tid = e["task"]["task_id"]
        prompt = tok.apply_chat_template(e["history"], tokenize=False,
                                         add_generation_prompt=True, **template_kwargs)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        allowance = limit - len(p_ids)
        if allowance <= 0:
            close(e, "context_exhausted")
            return
        seed = (seed_of(tid, len(e["turns"])) if seed_of is not None else
                request_seed(seed_base, tid, len(e["turns"])) if mode == "generate" else None)
        rid = f"{tid}::{len(e['turns'])}"
        eng.add_request(rid, {"prompt_token_ids": p_ids}, make_sampling(allowance, seed))
        pending[rid] = {"ep": e, "p_ids": p_ids, "allowance": allowance,
                        "preopened": template_opens_think(prompt),
                        "think_state": template_think_state(prompt), "gen": [],
                        "last": 0, "degen": None, "finish": None, "t0": time.time(),
                        "ttft": None, "seed": seed}

    def finish(st) -> None:
        nonlocal n_turns, out_tokens
        e, gen = st["ep"], st["gen"]
        n_turns += 1
        out_tokens += len(gen)
        raw = tok.decode(gen, skip_special_tokens=False)
        hit = len(gen) >= st["allowance"]
        eos = bool(gen) and gen[-1] in stop_ids
        reason = ("eos" if eos else "degeneration" if st["degen"] else
                  ("context_limit" if limit == ctx else "length_limited")
                  if hit else st["finish"] or "unknown")
        parts = split_generation(raw, think_preopened=st["preopened"], think_state=st["think_state"])
        turn = {
            "turn": len(e["turns"]), "prompt_tokens": len(st["p_ids"]),
            "generation_allowance": st["allowance"], "generated_tokens": len(gen),
            "stop_reason": reason, "natural_termination": reason == "eos",
            "context_limit_reached": reason == "context_limit",
            "length_limited": reason == "length_limited",
            "right_censored": reason in ("context_limit", "degeneration"),
            "degeneration_triggered": st["degen"] is not None,
            "degeneration_kind": (st["degen"] or {}).get("kind"),
            "think_preopened": st["preopened"], "think_state": st["think_state"],
            "seed": st["seed"],
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
            close(e, reason)
            return
        if e["env"] is None:
            close(e, "answered")
            return
        obs = e["env"].step(parts["answer"])
        turn["observation"] = obs
        if obs is None or e["env"].done:
            close(e, "episode_done")
        elif len(e["turns"]) >= e["max_turns"]:
            close(e, "turn_limit")
        else:
            e["history"].append({"role": "user", "content": obs})
            submit(e)

    for e in episodes.values():
        submit(e)
    while eng.has_unfinished_requests():
        for out in eng.step():
            st = pending.get(out.request_id)
            if st is None:
                continue
            st["gen"].extend(out.outputs[0].token_ids)
            if st["ttft"] is None and st["gen"]:
                st["ttft"] = time.time() - st["t0"]
            if out.finished:
                st["finish"] = out.outputs[0].finish_reason
                pending.pop(out.request_id)
                finish(st)
                continue
            if len(st["gen"]) - st["last"] >= check_every:
                st["last"] = len(st["gen"])
                d = degeneration_check(st["gen"])
                if d:
                    st["degen"], st["finish"] = d, "degeneration"
                    eng.abort_request(out.request_id)
                    pending.pop(out.request_id)
                    finish(st)
        if time.time() - last_log > 60:
            last_log = time.time()
            log(f"[{label}] {n_turns} turns done, {len(pending)} running, "
                f"{sum(not e['done'] for e in episodes.values())} episodes active, "
                f"{out_tokens:,} tokens, {time.time() - t_gen:.0f}s", flush=True)
    for st in list(pending.values()):   # anything the engine dropped
        pending.pop(f"{st['ep']['task']['task_id']}::{len(st['ep']['turns'])}", None)
        finish(st)
    log(f"[{label}] {n_turns} turns, {out_tokens:,} tokens, {time.time() - t_gen:.0f}s",
        flush=True)
    return episodes, n_turns, out_tokens


def episode_record(tid: str, e: dict, *, label: str, mode: str, action_protocol: str) -> dict:
    """One finished episode as a record line (the final file and the stream
    write the same thing)."""
    t = e["task"]
    return {"task_id": tid, "label": label, "mode": mode,
            "generator_id": t["generator_id"], "family_id": t["family_id"],
            "axis": t["axis"], "difficulty": t["difficulty"],
            "interactive": t["interactive"], "end": e["end"],
            "turns": e["turns"],
            "env_result": e["env"].result() if e["env"] else None,
            **({"action_protocol": action_protocol} if e["env"] else {}),
            "messages": e["history"]}


def streamer(path: Path, *, label: str, mode: str, action_protocol: str):
    """`on_done` that appends each finished episode's record to `path` at once,
    flushed and fsynced: a caller reading complete lines sees finished work
    while the batch still runs."""
    f = open(path, "a")

    def write(tid: str, e: dict) -> None:
        line = json.dumps(episode_record(tid, e, label=label, mode=mode,
                                         action_protocol=action_protocol), ensure_ascii=False)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return write, f


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
    ap.add_argument("--diagnostic-context-cap", type=int, default=None,
                    help="cap the context BELOW the model's supported one, for a matched-"
                         "allowance diagnostic only; recorded as context_source="
                         "diagnostic_cap with the uncapped value, never as a P18 measurement")
    ap.add_argument("--session-limit", type=int, default=None,
                    help="generate mode only: the data-construction bound")
    ap.add_argument("--seed-base", type=int, default=20261018)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--check-every", type=int, default=256)
    ap.add_argument("--system", default="You are a helpful Assistant.")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks (smoke)")
    ap.add_argument("--stream-completed", default=None,
                    help="also append each episode's record here the moment it ends "
                         "(flushed, fsynced); the --out file is written at the end as before")
    ap.add_argument("--decoding", default=None,
                    help="eval mode: a decoding protocol file (configs/shared/evaluation/decoding/); "
                         "omitted, the frozen greedy SAMPLING is used unchanged")
    ap.add_argument("--seed-namespace", default=None,
                    help="with --decoding: the evaluated unit's key, e.g. fluid_reasoning_720__dev_v1")
    ap.add_argument("--action-protocol", default="v1", choices=("v1", "v2"),
                    help="how the environment reads action lines (tasks/interactive.py); "
                         "recorded per episode, and the scorer replays with it")
    ap.add_argument("--chat-template-kwargs", default="{}",
                    help='JSON passed to every apply_chat_template call, e.g. '
                         '\'{"enable_thinking": false}\' for a hybrid template; recorded')
    ap.add_argument("--generate-sampling", default=None,
                    help="generate mode: JSON replacing the default teacher sampling "
                         "(GENERATE_SAMPLING), e.g. a mode's own official preset; recorded")
    args = ap.parse_args(argv)
    template_kwargs = json.loads(args.chat_template_kwargs)
    if args.generate_sampling and args.mode != "generate":
        raise SystemExit("--generate-sampling is a data-generation setting; a measurement "
                         "uses --decoding")
    if args.mode == "eval" and args.session_limit:
        raise SystemExit("--session-limit is a data-construction bound; a measurement "
                         "has none (P18)")
    from shared.evaluation import decoding as DEC
    protocol = DEC.load(args.decoding) if args.decoding else None
    if protocol and (args.mode != "eval" or not args.seed_namespace
                     or int(protocol["samples_per_item"]) != 1):
        raise SystemExit("--decoding is a measurement protocol: eval mode, a --seed-namespace, "
                         "one sample per item")

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
    if args.diagnostic_context_cap and args.diagnostic_context_cap < ctx_info["resolved_context"]:
        ctx_info = {**ctx_info, "uncapped_context": ctx_info["resolved_context"],
                    "uncapped_context_source": ctx_info["context_source"],
                    "resolved_context": args.diagnostic_context_cap,
                    "context_source": "diagnostic_cap",
                    "not_a_p18_measurement": (
                        "generation is capped below the model's supported context to match "
                        "another model's measurement allowance; context-limit hits are "
                        "right-censored and the result is a diagnostic, not this model's "
                        "P18 score")}
    ctx = ctx_info["resolved_context"]
    stop_ids = resolve_stop_ids(args.model, tok, cfg, args.revision)
    limit = min(ctx, args.session_limit) if args.session_limit else ctx
    sampling = ((json.loads(args.generate_sampling) if args.generate_sampling
                 else dict(GENERATE_SAMPLING)) if args.mode != "eval" else
                dict(SAMPLING) if protocol is None else DEC.sampling(protocol))
    print(f"[{args.label}] mode={args.mode} context={ctx} "
          f"({ctx_info['context_source']}) limit={limit} stop={stop_ids}", flush=True)

    tasks = read_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]

    t0 = time.time()
    # The engine is sized to what can be generated: the resolved context for a
    # measurement, the data-construction session limit for generation.
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=limit,
              gpu_memory_utilization=args.gpu_mem_util, **rev)
    eng = llm.llm_engine
    init_s = round(time.time() - t0, 1)
    engine = engine_config(llm, limit, args.gpu_mem_util)
    print(f"[{args.label}] engine ready in {init_s}s", flush=True)

    from vllm.sampling_params import RequestOutputKind

    def make_sampling(max_tokens, seed):
        params = {k: v for k, v in sampling.items() if k != "detokenize"}
        if seed is not None:
            params["seed"] = seed
        return SamplingParams(max_tokens=max_tokens, stop_token_ids=stop_ids,
                              detokenize=False, output_kind=RequestOutputKind.DELTA,
                              **params)

    stream = (streamer(Path(args.stream_completed), label=args.label, mode=args.mode,
                       action_protocol=args.action_protocol) if args.stream_completed else None)
    t_gen = time.time()
    episodes, rnd, out_tokens = run_episodes(
        tasks, tok=tok, eng=eng, make_sampling=make_sampling, limit=limit, ctx=ctx,
        stop_ids=stop_ids, mode=args.mode, seed_base=args.seed_base,
        system=args.system, check_every=args.check_every,
        degeneration_check=degeneration.check, split_generation=split_generation,
        template_opens_think=template_opens_think, label=args.label,
        action_protocol=args.action_protocol, on_done=stream[0] if stream else None,
        seed_of=(lambda tid, turn: DEC.request_seed(protocol, args.seed_namespace, tid, turn))
        if protocol else None, template_kwargs=template_kwargs)
    if stream:
        stream[1].close()
    gen_s = round(time.time() - t_gen, 1)
    identity = {
        "label": args.label, "model": args.model, "revision": args.revision,
        "mode": args.mode, "tasks_file": args.tasks,
        "tasks_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
        "context_resolution": ctx_info, "session_limit": args.session_limit,
        "effective_limit": limit, "stop_ids": stop_ids, "sampling": sampling,
        "seed_base": args.seed_base if args.mode == "generate" else None,
        **({"decoding_protocol": DEC.identity(protocol), "seed_namespace": args.seed_namespace}
           if protocol else {}),
        "action_protocol": args.action_protocol,
        "system_message": args.system,
        "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
        "chat_template_kwargs": template_kwargs,
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
            rec = episode_record(tid, e, label=args.label, mode=args.mode,
                                 action_protocol=args.action_protocol)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {"identity": identity, "n_tasks": len(episodes), "turns": rnd,
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
