"""Generate verified-correct teacher targets for a mixture slice (Stage 2 v2).

    uv run python scripts/generate_teacher_answers.py \
        --slices rag_evidence,multihop_qa,refusal_uncertainty,gsm8k,openmath \
        --limit-per-slice 200 --n 4 --out artifacts/stage2_v2/pilot

For each prompt the teacher produces **n sampled candidates in its native
thinking mode**, every candidate is verified against the gold key with
`aadistill.verify`, and one accepted candidate becomes the new target. A prompt
with no accepted candidate **keeps its v1 public target** — no unverified
teacher text ever enters training (decision record 2026-07-28).

Sampling is **untruncated** (temperature 1.0, top_p 1.0, top_k off) and there is
**no greedy candidate** (maintainer, 2026-07-29), following mainstream on-policy
rollout practice rather than the vendor's single-answer serving preset. With a
verifier downstream this is rejection sampling, where candidate diversity is
what makes accept@n exceed accept@1. `accept_at_1` therefore now means "one
sample was accepted", not "greedy was accepted".

The teacher is never prefilled with a closed think block. Suppressing its
reasoning would be ~10× cheaper and would measure our prompt convention rather
than the teacher; the framework distills the teacher's actual capability
(decision record 2026-07-28). Budget accordingly: a candidate costs its
reasoning trace plus its answer.

Outputs, all under `--out`:

* `candidates.jsonl` — every candidate with its verdict and reason, so the
  selection can be re-derived without regenerating;
* `targets.jsonl` — one record per prompt: the selected target or the v1
  fallback, with provenance. A teacher target carries the **whole generation** —
  `reasoning_content` (the trace) plus `content` (the answer) — because the
  student is trained to inherit the reasoning, not just the conclusion
  (decision record 2026-07-28, option B);
* `manifest.json` — accept@1 / accept@n and reject-reason histograms per slice,
  thinking-length stats, decode config, hashes, code state, hardware.

Determinism: candidates are seeded per batch, but batched generation is not
bitwise reproducible, so **the corpus is the artifact** and its hash pins the
experiment (P5). That stance is now measured rather than assumed — even *greedy*
bf16 decoding is not batch-invariant on this project's own hardware
(`logs/experiments/2026-07-29_engine_adapter_and_bf16_invariance.md`), which is
part of why no candidate is generated greedily any more.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.behavior import THINK_CLOSE, split_generation
from aadistill.data import load_jsonl
from aadistill.engines import HFEngine, SGLangEngine, VLLMEngine
from aadistill.env import code_state, hardware_report
from aadistill.manifest import sha256_file
from aadistill.teacher import load_causal_lm
from aadistill.verify import VERIFIABLE, select, verify

# Slice name -> (group, source). The group is also the mixture file name.
SLICES = {
    "rag_evidence": ("rag_evidence", "squad_v2"),
    "multihop_qa": ("multihop_qa", "hotpot_qa"),
    "refusal_uncertainty": ("refusal_uncertainty", "squad_v2"),
    "gsm8k": ("code_math", "gsm8k"),
    "openmath": ("code_math", "openmath_instruct_2"),
}
assert set(SLICES.values()) == set(VERIFIABLE), "slice table drifted from the verifier"


def load_slice(data_dir: Path, name: str, limit: int | None) -> list[dict]:
    group, source = SLICES[name]
    path = data_dir / "train" / f"{group}.jsonl"
    # `load_jsonl` iterates the file handle: `read_text().splitlines()` also
    # splits on \v, \f and \u2028, which LaTeX-heavy openmath targets contain,
    # and would tear a sample in half.
    rows = [s for s in load_jsonl(path) if s["source"] == source]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no samples for slice {name} in {path}")
    return rows


def generation_prompt(tokenizer, sample: dict) -> str:
    """Render the prompt exactly as training renders everything before the answer.

    Asserted prefix-exact against the training-time rendering of the same
    sample: if the two ever diverge, the teacher would be answering in a
    different position from the one the student is trained on, and the corpus
    would be quietly wrong.
    """
    prompt = tokenizer.apply_chat_template(
        sample["messages"][:-1], tools=sample.get("tools"),
        tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        sample["messages"], tools=sample.get("tools"),
        tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError(
            f"generation prompt is not a prefix of the training rendering for "
            f"{sample['id']!r} — the chat template changed under us")
    return prompt


def stop_ids(model, tokenizer) -> set[int]:
    """Every id that ends a turn — Qwen3 stops on `<|im_end|>`, not on `<|endoftext|>`."""
    configured = model.generation_config.eos_token_id
    ids = set(configured if isinstance(configured, (list, tuple)) else [configured])
    ids.add(tokenizer.eos_token_id)
    return {i for i in ids if i is not None}


def build_engine(args, model, tokenizer):
    """Construct the decode backend named by `--engine`.

    The in-stack engine reuses the already-loaded model. The serving engines
    load their own copy, so the training-stack model is moved off the GPU first
    — otherwise it holds memory that vLLM/SGLang want for their KV cache and the
    build fails at allocation time rather than for any real reason.
    """
    if args.engine == "hf":
        return HFEngine(model, tokenizer.pad_token_id, batch_size=args.batch_size)

    if torch.cuda.is_available():
        model.to("cpu")
        torch.cuda.empty_cache()
    spec = args.model
    path, revision = (spec.split("@", 1) if "@" in spec else (spec, None))
    if args.engine == "vllm":
        return VLLMEngine(path, dtype=args.dtype, revision=revision)
    return SGLangEngine(path, dtype=args.dtype, revision=revision)


@torch.no_grad()
def generate_candidates(engine, tokenizer, prompt_ids: list[list[int]], *, n: int,
                        max_new_tokens: int, temperature: float, top_p: float,
                        top_k: int, seed: int, stops: set[int], think_close: int):
    """n **sampled** candidates for each prompt in the batch.

    No greedy candidate (maintainer, 2026-07-29). Every candidate is an equal
    draw from the teacher's distribution, following mainstream on-policy rollout
    practice: DAPO and GRPO-style systems sample untruncated because truncating
    the tail suppresses low-probability tokens at exactly the high-entropy
    positions where the interesting branches live, so the corpus would reflect a
    *truncated* teacher rather than the teacher. With `n` candidates and a
    verifier, this is rejection sampling, and diversity is what makes accept@n
    exceed accept@1.

    Two further reasons the greedy candidate is gone. It was mode-collapsed by
    construction — n candidates of which one is the argmax path is n-1 samples
    plus a duplicate-prone outlier. And the determinism that justified
    privileging it in `verify.select` does not exist: bf16 greedy decoding is
    not batch-invariant on this project's own measurement (experiment log
    2026-07-29), so "the deterministic candidate" was never deterministic across
    batch compositions.

    One engine call per batch, with each prompt replicated `n` times
    contiguously: batch-1 decoding of a thinking teacher is ~50 h for a 1k-prompt
    pilot, and replicating into a single call keeps a continuous-batching engine
    saturated, which is most of the throughput on the serving arms.

    Driven through `aadistill.engines` rather than `model.generate` so the corpus
    can be built by whichever engine the benchmark selected
    (`logs/proposals/2026-07-29_engine_benchmark.md`) without this script knowing
    which one it is. The adapter is token-in/token-out and does the stop-cutting
    itself, in one shared code path for every engine, so a corpus built on vLLM
    is trimmed identically to one built in-stack; the text here is derived for
    verification and readability only.

    Returns, per prompt, a list of (raw_text, n_new_tokens, hit_cap, think_tokens).
    """
    repeated = [ids for ids in prompt_ids for _ in range(n)]
    completions = engine.generate(
        repeated, max_new_tokens=max_new_tokens, stop_ids=stops, greedy=False,
        temperature=temperature, top_p=top_p, top_k=top_k, seed=seed)

    per_prompt: list[list] = [[] for _ in prompt_ids]
    for index, completion in enumerate(completions):
        ids = completion["tokens"]
        think_tokens = ids.index(think_close) if think_close in ids else completion["n_new"]
        per_prompt[index // n].append(
            (tokenizer.decode(ids, skip_special_tokens=False),
             completion["n_new"], completion["hit_cap"], think_tokens))
    return per_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@768f209d")
    ap.add_argument("--data-dir", default="data/stage2_v1")
    # Defaults to the dense baseline's in-scope slices, not to every slice the
    # verifier *can* check. `refusal_uncertainty` is evaluation-only for this
    # recipe (decision 2026-07-30): it is outside the declared capability target,
    # so generating it would spend student capacity and GPU budget on an
    # alignment tax nothing has justified. Pass it explicitly to override.
    IN_SCOPE = ("rag_evidence", "multihop_qa", "gsm8k", "openmath")
    ap.add_argument("--slices", default=",".join(IN_SCOPE))
    ap.add_argument("--limit-per-slice", type=int, default=None)
    ap.add_argument("--n", type=int, default=4, help="candidates per prompt")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts per generate call; raise until GPU memory is the limit")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="must fit the teacher's reasoning trace plus its answer")
    # Untruncated sampling, following mainstream on-policy rollout practice
    # (DAPO/GRPO) rather than Qwen3-Thinking's single-answer serving preset
    # (0.6 / 0.95 / top_k 20). The preset optimizes one good answer; this job
    # wants n *diverse* candidates whose distribution is the teacher's own, with
    # the verifier — not the sampler — doing the filtering. See the decision
    # record 2026-07-29. The pilot measures accept@1/accept@n, which is the
    # empirical check on whether this is too hot.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0,
                    help="0 disables top-k; threaded explicitly because engine "
                         "defaults disagree (HF 50, vLLM/SGLang off)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm", "sglang"],
                    help="decode backend; `hf` is in-stack and training-identical")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock budget; stops cleanly at the next batch "
                         "boundary and writes complete artifacts for the "
                         "prompts finished so far (P6)")
    ap.add_argument("--engine-from", default=None,
                    help="path to a bench_engines.py decision.json — takes the "
                         "winner from it, so a session can chain benchmark → "
                         "generation without an agent in the loop")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.engine_from:
        decision = json.loads(Path(args.engine_from).read_text())
        winner = decision.get("winner")
        if not winner:
            raise SystemExit(
                f"{args.engine_from} selected no engine "
                f"({decision.get('reason')}) — refusing to guess a backend for a "
                "corpus build")
        args.engine = winner
        print(f"engine {winner!r} from {args.engine_from} "
              f"(rule {decision.get('rule')})", flush=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    names = [s.strip() for s in args.slices.split(",") if s.strip()]
    unknown = set(names) - set(SLICES)
    if unknown:
        raise SystemExit(f"unknown slice(s): {sorted(unknown)}; known: {sorted(SLICES)}")

    data_dir = REPO_ROOT / args.data_dir
    work = {name: load_slice(data_dir, name, args.limit_per_slice) for name in names}
    total = sum(len(v) for v in work.values())
    print(f"{total} prompts across {len(work)} slices, n={args.n}, "
          f"cap={args.max_new_tokens}, device={device}", flush=True)

    model, tokenizer = load_causal_lm(args.model, dtype, device)
    # Batched generation pads; left padding keeps every prompt flush against the
    # first generated token, which is what makes prompt-length slicing valid.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stops = stop_ids(model, tokenizer)
    think_close = tokenizer.convert_tokens_to_ids(THINK_CLOSE)

    engine = build_engine(args, model, tokenizer)
    print(f"engine: {engine.name}", flush=True)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stats = {name: {"accept_at_1": 0, "accept_at_n": 0, "prompts": 0,
                    "reasons": Counter(), "think_tokens": [], "answer_words": []}
             for name in names}
    done = 0

    with open(out_dir / "candidates.jsonl", "w") as f_cand, \
            open(out_dir / "targets.jsonl", "w") as f_target:
        batches = [(name, samples[i: i + args.batch_size])
                   for name, samples in work.items()
                   for i in range(0, len(samples), args.batch_size)]
        budget_stopped = False
        for batch_index, (name, batch) in enumerate(batches):
            # Checked at the batch boundary so the artifacts on disk are always
            # a complete prefix of the corpus, never a half-written batch. An
            # unattended run on a paid pod needs a backstop that does not depend
            # on anyone watching it (P6/P12).
            if args.max_hours and (time.time() - started) / 3600 >= args.max_hours:
                budget_stopped = True
                print(f"\n!! wall-clock budget {args.max_hours}h reached after "
                      f"{done}/{total} prompts — stopping cleanly", flush=True)
                break
            prompt_ids = [
                tokenizer(generation_prompt(tokenizer, s),
                          add_special_tokens=False).input_ids
                for s in batch
            ]
            batch_raws = generate_candidates(
                engine, tokenizer, prompt_ids, n=args.n,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k, seed=args.seed + batch_index,
                stops=stops, think_close=think_close)

            for sample, raws in zip(batch, batch_raws):
                candidates = []
                for index, (raw, n_new, hit_cap, think_tokens) in enumerate(raws):
                    parts = split_generation(raw)
                    accepted, reason = verify(sample, parts["answer"], raw)
                    if hit_cap:
                        accepted, reason = False, "truncated_at_cap"
                    candidates.append({
                        "index": index, "answer": parts["answer"], "think": parts["think"],
                        "raw": raw, "new_tokens": n_new, "accepted": accepted,
                        "reason": reason, "think_tokens": think_tokens,
                    })

                chosen = select(candidates)
                slice_stats = stats[name]
                slice_stats["prompts"] += 1
                slice_stats["accept_at_1"] += int(candidates[0]["accepted"])
                slice_stats["accept_at_n"] += int(chosen is not None)
                for candidate in candidates:
                    slice_stats["reasons"][candidate["reason"]] += 1
                    slice_stats["think_tokens"].append(candidate["think_tokens"])
                if chosen:
                    slice_stats["answer_words"].append(len(chosen["answer"].split()))

                f_cand.write(json.dumps({
                    "id": sample["id"], "slice": name, "group": sample["group"],
                    "source": sample["source"],
                    "gold": sample["messages"][-1]["content"],
                    "candidates": [{k: v for k, v in c.items() if k != "raw"}
                                   for c in candidates],
                    "chosen_index": chosen["index"] if chosen else None,
                }, ensure_ascii=False) + "\n")

                target = dict(sample)
                if chosen:
                    # Option B (decision 2026-07-28): the trace is part of the
                    # target. `reasoning_content` is what the Qwen3-Thinking
                    # template renders inside <think>…</think> for the final
                    # assistant turn, and the loss mask spans the whole
                    # assistant block, so the trace is supervised as-is.
                    target["messages"] = (sample["messages"][:-1] + [{
                        "role": "assistant",
                        "reasoning_content": chosen["think"],
                        "content": chosen["answer"],
                    }])
                    target["target_source"] = "teacher_verified"
                    target["candidate_index"] = chosen["index"]
                    target["think_tokens"] = chosen["think_tokens"]
                else:
                    target["target_source"] = "v1_public"
                    target["candidate_index"] = None
                f_target.write(json.dumps(target, ensure_ascii=False) + "\n")
                done += 1

            elapsed = time.time() - started
            print(f"  {done}/{total} prompts  {elapsed:.0f}s "
                  f"({elapsed / done:.1f}s/prompt)", flush=True)

    summary = {}
    for name, s in stats.items():
        if not s["prompts"]:
            # Reachable when `--max-hours` stops the run before a slice starts.
            # Reporting 0.0 would read as "nothing was accepted" rather than
            # "nothing was attempted", which is a different and much worse claim.
            summary[name] = {"prompts": 0, "not_reached": True}
            continue
        summary[name] = {
            "prompts": s["prompts"],
            "accept_at_1": round(s["accept_at_1"] / s["prompts"], 4),
            "accept_at_n": round(s["accept_at_n"] / s["prompts"], 4),
            "reject_reasons": dict(s["reasons"].most_common()),
            "think_tokens_median": statistics.median(s["think_tokens"]) if s["think_tokens"] else 0,
            "think_tokens_p90": (statistics.quantiles(s["think_tokens"], n=10)[-1]
                                 if len(s["think_tokens"]) > 9 else None),
            "answer_words_median": (statistics.median(s["answer_words"])
                                    if s["answer_words"] else None),
        }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "teacher": args.model,
        "thinking_mode": True,  # never suppressed — decision record 2026-07-28
        "decoding": {
            "n": args.n, "all_candidates_sampled": True,
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
            "seed": args.seed, "dtype": args.dtype, "device": device,
            # The engine changes the numerics, so it is part of the corpus's
            # identity: two corpora built by different backends are not
            # interchangeable even at identical decode settings (P4/P9).
            "engine": engine.name,
            "engine_selected_by": args.engine_from,
        },
        "data_dir": args.data_dir,
        # A budget-stopped corpus is a valid artifact but not a complete one;
        # anything training on it must know that, so it is recorded next to the
        # hashes rather than inferred from a prompt count (P4/P11).
        "complete": not budget_stopped,
        "prompts_requested": total,
        "prompts_generated": done,
        "max_hours": args.max_hours,
        "slices": summary,
        "seconds": round(time.time() - started, 1),
        "outputs": {
            "candidates": sha256_file(out_dir / "candidates.jsonl"),
            "targets": sha256_file(out_dir / "targets.jsonl"),
        },
        "code_state": code_state(str(REPO_ROOT)),
        "hardware": hardware_report(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nwrote {out_dir} ({manifest['seconds']}s)")
    for name, s in summary.items():
        print(f"  {name:22s} accept@1 {s['accept_at_1']:.3f}  accept@n {s['accept_at_n']:.3f}  "
              f"think_median {s['think_tokens_median']:.0f} tok  "
              f"top rejects {list(s['reject_reasons'].items())[:3]}")


if __name__ == "__main__":
    main()
