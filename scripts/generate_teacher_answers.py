"""Generate verified-correct teacher targets for a mixture slice (Stage 2 v2).

    uv run python scripts/generate_teacher_answers.py \
        --slices rag_evidence,multihop_qa,refusal_uncertainty,gsm8k,openmath \
        --limit-per-slice 200 --n 4 --out artifacts/stage2_v2/pilot

For each prompt the teacher produces **n candidates in its native thinking
mode** (candidate 0 greedy, the rest sampled), every candidate is verified
against the gold key with `aadistill.verify`, and one accepted candidate becomes
the new target. A prompt with no accepted candidate **keeps its v1 public
target** — no unverified teacher text ever enters training (decision record
2026-07-28).

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

Determinism: greedy candidate 0 is reproducible on fixed hardware; sampled
candidates are seeded per (prompt, candidate) but batched generation is not
bitwise reproducible, so **the corpus is the artifact** and its hash pins the
experiment (P5).
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


def _cut_at_stop(row: list[int], stops: set[int], cap: int,
                 think_close: int | None) -> tuple[list[int], int, bool, int]:
    """Trim a generated row at its first stop token (kept).

    Returns (ids, n_new, hit_cap, think_tokens). The trace length comes from the
    position of the `</think>` token, which is a single id — re-tokenizing the
    trace text instead would re-encode ~1.5k tokens per candidate, hundreds of
    millions over a bulk run, to recover a number the ids already carry.
    """
    kept, n_new, hit_cap = row, len(row), len(row) >= cap
    for position, token in enumerate(row):
        if token in stops:
            kept, n_new, hit_cap = row[: position + 1], position + 1, False
            break
    think_tokens = kept.index(think_close) if think_close in kept else n_new
    return kept, n_new, hit_cap, think_tokens


@torch.no_grad()
def generate_candidates(model, tokenizer, prompts: list[str], *, n: int, max_new_tokens: int,
                        temperature: float, top_p: float, seed: int, device: str):
    """n candidates for each prompt in the batch — candidate 0 greedy, rest sampled.

    Two `generate` calls per batch rather than n per prompt: batch-1 decoding of
    a thinking teacher is ~50 h for a 1k-prompt pilot, which is the difference
    between an affordable session and an unaffordable one. Left padding, and
    each row is cut at its first stop token so a short answer is not scored with
    another sequence's padding attached.

    Returns, per prompt, a list of (raw_text, n_new_tokens, hit_cap, think_tokens).
    """
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
    prompt_len = encoded.input_ids.shape[1]
    stops = stop_ids(model, tokenizer)
    think_close = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    shared = dict(max_new_tokens=max_new_tokens, top_k=None,
                  pad_token_id=tokenizer.pad_token_id)

    def rows_of(output):
        for row in output[:, prompt_len:].tolist():
            ids, n_new, hit_cap, think_tokens = _cut_at_stop(
                row, stops, max_new_tokens, think_close)
            yield (tokenizer.decode(ids, skip_special_tokens=False),
                   n_new, hit_cap, think_tokens)

    per_prompt: list[list] = [[] for _ in prompts]
    greedy = model.generate(**encoded, do_sample=False, temperature=None, top_p=None, **shared)
    for index, item in enumerate(rows_of(greedy)):
        per_prompt[index].append(item)

    if n > 1:
        torch.manual_seed(seed)
        sampled = model.generate(**encoded, do_sample=True, temperature=temperature,
                                 top_p=top_p, num_return_sequences=n - 1, **shared)
        for index, item in enumerate(rows_of(sampled)):
            per_prompt[index // (n - 1)].append(item)  # generate keeps prompts contiguous
    return per_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@768f209d")
    ap.add_argument("--data-dir", default="data/stage2_v1")
    ap.add_argument("--slices", default=",".join(SLICES))
    ap.add_argument("--limit-per-slice", type=int, default=None)
    ap.add_argument("--n", type=int, default=4, help="candidates per prompt")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="prompts per generate call; raise until GPU memory is the limit")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="must fit the teacher's reasoning trace plus its answer")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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
    # first generated token, which is what makes `prompt_len` slicing valid.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        for batch_index, (name, batch) in enumerate(batches):
            prompts = [generation_prompt(tokenizer, s) for s in batch]
            batch_raws = generate_candidates(
                model, tokenizer, prompts, n=args.n,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_p=args.top_p, seed=args.seed + batch_index, device=device)

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
            "n": args.n, "candidate_0": "greedy", "temperature": args.temperature,
            "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
            "seed": args.seed, "dtype": args.dtype, "device": device,
        },
        "data_dir": args.data_dir,
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
