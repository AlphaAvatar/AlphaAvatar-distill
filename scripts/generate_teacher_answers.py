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
  fallback, with provenance;
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

from aadistill.behavior import split_generation
from aadistill.env import code_state, hardware_report
from aadistill.manifest import sha256_file
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
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        if sample["source"] != source:
            continue
        rows.append(sample)
        if limit and len(rows) >= limit:
            break
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


@torch.no_grad()
def generate_candidates(model, tokenizer, prompt: str, *, n: int, max_new_tokens: int,
                        temperature: float, top_p: float, seed: int, device: str):
    """Candidate 0 greedy, 1..n-1 sampled. Returns a list of (raw, n_new, hit_cap)."""
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = []
    for index in range(n):
        if index:
            torch.manual_seed(seed + index)
        result = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=bool(index),
            temperature=temperature if index else None,
            top_p=top_p if index else None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        new = result.shape[1] - ids.shape[1]
        text = tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=False)
        out.append((text, new, new >= max_new_tokens))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@768f209d")
    ap.add_argument("--data-dir", default="data/stage2_v1")
    ap.add_argument("--slices", default=",".join(SLICES))
    ap.add_argument("--limit-per-slice", type=int, default=None)
    ap.add_argument("--n", type=int, default=4, help="candidates per prompt")
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

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from eval_behavior import load_model  # same loader, same revision pinning

    model, tokenizer = load_model(args.model, dtype, device)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stats = {name: {"accept_at_1": 0, "accept_at_n": 0, "prompts": 0,
                    "reasons": Counter(), "think_tokens": [], "answer_words": []}
             for name in names}
    done = 0

    with open(out_dir / "candidates.jsonl", "w") as f_cand, \
            open(out_dir / "targets.jsonl", "w") as f_target:
        for name, samples in work.items():
            for sample in samples:
                prompt = generation_prompt(tokenizer, sample)
                raws = generate_candidates(
                    model, tokenizer, prompt, n=args.n,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, seed=args.seed + abs(hash(sample["id"])) % 10_000,
                    device=device)

                candidates = []
                for index, (raw, n_new, hit_cap) in enumerate(raws):
                    parts = split_generation(raw)
                    accepted, reason = verify(sample, parts["answer"], raw)
                    if hit_cap:
                        accepted, reason = False, "truncated_at_cap"
                    candidates.append({
                        "index": index, "answer": parts["answer"], "think": parts["think"],
                        "raw": raw, "new_tokens": n_new, "accepted": accepted,
                        "reason": reason,
                    })

                chosen = select(candidates)
                slice_stats = stats[name]
                slice_stats["prompts"] += 1
                slice_stats["accept_at_1"] += int(candidates[0]["accepted"])
                slice_stats["accept_at_n"] += int(chosen is not None)
                for candidate in candidates:
                    slice_stats["reasons"][candidate["reason"]] += 1
                    think_tokens = len(tokenizer(candidate["think"]).input_ids)
                    slice_stats["think_tokens"].append(think_tokens)
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
                    target["messages"] = (sample["messages"][:-1]
                                          + [{"role": "assistant", "content": chosen["answer"]}])
                    target["target_source"] = "teacher_verified"
                    target["candidate_index"] = chosen["index"]
                else:
                    target["target_source"] = "v1_public"
                    target["candidate_index"] = None
                f_target.write(json.dumps(target, ensure_ascii=False) + "\n")

                done += 1
                if done % 10 == 0 or done == total:
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
