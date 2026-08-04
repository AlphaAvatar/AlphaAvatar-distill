#!/usr/bin/env python
"""Can an overfitted checkpoint reproduce the targets it actually trained on?

    PYTHONPATH=src python scripts/evaluation/diagnose_training_recall.py \
        --model /workspace/ckpt/e1_ctl_r0250k_sa_pca_stepmatched/model \
        --pack artifacts/stage3/ladder_uniform_probe --rung 250000 \
        --sessions artifacts/stage3/corpus_v2/sessions.jsonl \
        --n 150 --out artifacts/audit/training_recall

The control arm `e1_ctl_r0250k_sa_pca_stepmatched` ran 4,412 steps x 2 blocks
over the 216-block 0.25M rung — roughly **41 passes** over the same data. If a
model that has seen a target forty times still cannot reproduce it under free
generation, the failure is not "not enough data": it is somewhere in the
template, the EOS supervision, the loss masking or the target construction, and
those must be audited before any on-policy machinery is switched on.

If instead it reproduces its training targets well while held-out correctness
stays at the floor, the problem is generalization, not exposure, and the
on-policy remedy would be aimed at the wrong thing.

Five measurements per sampled session
-------------------------------------
1. **free generation** from the prompt alone: correctness where a key exists,
   protocol validity, natural termination, degeneration;
2. **token-level prefix agreement** with the gold target and the index of the
   first divergence — how far it stays on the trained trajectory;
3. **gold-prefix next-token top-1 accuracy** under teacher forcing, which
   separates "knows the next token given the true prefix" from "can sustain its
   own prefix";
4. **forced-prefix release** at assistant-prefix lengths 0, 16, 64 and 256: the
   gold prefix is supplied, then generation continues unconstrained. Where the
   model recovers as the handed prefix grows, the failure is in getting started
   rather than in knowing the content;
5. raw outputs and per-sample fields are written for replay and error analysis.

Sampling is stratified by `data_type` so no single slice dominates, and the
sampled ids are recorded so the draw can be reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation import degeneration  # noqa: E402
from aadistill.evaluation.capability import normalize_answer  # noqa: E402
from aadistill.evaluation.strict_answer import (  # noqa: E402
    extract_final_answer, normalize_number, protocol_valid,
)
from aadistill.infrastructure.env import code_state  # noqa: E402

PREFIX_LENGTHS = (0, 16, 64, 256)


def rung_session_ids(pack: Path, rung: int) -> list[str]:
    meta = json.loads((pack / "ladder.json").read_text())
    entry = next(r for r in meta["rungs"]
                 if r["target_supervised_tokens"] == rung)
    n_blocks = entry["n_blocks"]
    ids: list[str] = []
    with (pack / "audit.jsonl").open() as f:
        for i, line in enumerate(f):
            if i >= n_blocks:
                break
            ids.extend(s["session_id"] for s in json.loads(line)["sessions"])
    return ids


def stratified_sample(sessions: list[dict], n: int, seed: int) -> list[dict]:
    by_type: dict[str, list] = defaultdict(list)
    for s in sessions:
        by_type[s["data_type"]].append(s)
    rng = random.Random(seed)
    for v in by_type.values():
        rng.shuffle(v)
    out, types = [], sorted(by_type)
    i = 0
    while len(out) < n and any(by_type[t] for t in types):
        t = types[i % len(types)]
        if by_type[t]:
            out.append(by_type[t].pop())
        i += 1
    return out


def correctness(sample: dict, answer_text: str) -> tuple[bool | None, str]:
    """Exact/final-answer correctness where the corpus defines a key."""
    gold = sample.get("gold")
    if gold is None or sample.get("correct") is not True:
        return None, "no_key"
    pred, _ = extract_final_answer(answer_text)
    if pred is None:
        return False, "no_final_answer"
    if sample["data_type"] in ("gsm8k", "openmath"):
        p, g = normalize_number(pred), normalize_number(str(gold))
        if p is not None and g is not None:
            return p == g, "numeric"
    return normalize_answer(str(gold)) in normalize_answer(pred), "containment"


def prefix_agreement(gen_ids: list[int], gold_ids: list[int]) -> dict:
    n = 0
    for a, b in zip(gen_ids, gold_ids):
        if a != b:
            break
        n += 1
    return {
        "prefix_match_tokens": n,
        "first_divergence_index": (None if n == len(gold_ids) and
                                   n == len(gen_ids) else n),
        "prefix_match_fraction": round(n / len(gold_ids), 4) if gold_ids else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--system", default="You are a helpful Assistant.")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    want = set(rung_session_ids(args.pack, args.rung))
    trained = [json.loads(l) for l in args.sessions.open()
               if l.strip() and json.loads(l)["id"] in want]
    print(f"rung {args.rung}: {len(want)} trained sessions, "
          f"{len(trained)} resolved from the corpus", flush=True)
    picked = stratified_sample(trained, args.n, args.seed)
    print(f"sampled {len(picked)}: {dict(Counter(s['data_type'] for s in picked))}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    prepared = []
    for s in picked:
        turns = [m for m in s["messages"] if m["role"] != "assistant"]
        gold_text = next(m["content"] for m in reversed(s["messages"])
                         if m["role"] == "assistant")
        prompt = tok.apply_chat_template(turns, tools=s.get("tools"),
                                         tokenize=False,
                                         add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        gold_ids = tok(gold_text, add_special_tokens=False).input_ids
        prepared.append({"sample": s, "p_ids": p_ids, "gold_ids": gold_ids,
                         "gold_text": gold_text})

    # --- generation: one engine, every prefix length ----------------------
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.context,
              gpu_memory_utilization=args.gpu_mem_util)
    stop_ids = [i for i in (151645, 151643) if i is not None]
    reqs, meta = [], {}
    for item in prepared:
        for k in PREFIX_LENGTHS:
            if k and k > len(item["gold_ids"]):
                continue
            ids = item["p_ids"] + item["gold_ids"][:k]
            allowance = args.context - len(ids)
            if allowance <= 0:
                continue
            rid = f"{item['sample']['id']}::k{k}"
            reqs.append((rid, ids, allowance))
            meta[rid] = (item, k)
    print(f"{len(reqs)} generation requests", flush=True)

    outputs = llm.generate(
        [{"prompt_token_ids": ids} for _, ids, _ in reqs],
        [SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                        max_tokens=allow, stop_token_ids=stop_ids,
                        detokenize=False) for _, _, allow in reqs])

    rows = []
    for (rid, ids, allow), out in zip(reqs, outputs):
        item, k = meta[rid]
        gen_ids = list(out.outputs[0].token_ids)
        # skip_special_tokens=False on purpose: `protocol_valid` looks for
        # <|im_end|> and the think delimiters IN THE TEXT, so stripping them
        # makes every generation read as `not_terminated` regardless of what the
        # model did. Decoding them out was a scoring bug, not a model result.
        text = tok.decode(gen_ids, skip_special_tokens=False)
        degen = degeneration.check(gen_ids)
        valid, reason = protocol_valid(text)
        ended = bool(gen_ids) and gen_ids[-1] in stop_ids
        ok, how = correctness(item["sample"], text)
        rows.append({
            "id": item["sample"]["id"], "data_type": item["sample"]["data_type"],
            "forced_prefix_tokens": k,
            "generated_tokens": len(gen_ids),
            "natural_termination": ended,
            "context_limit_reached": len(gen_ids) >= allow,
            "protocol_valid": valid, "protocol_reason": reason,
            "degenerate": degen is not None,
            "degeneration": degen,
            "correct": ok, "correctness_basis": how,
            "gold_tokens": len(item["gold_ids"]),
            **prefix_agreement(gen_ids, item["gold_ids"][k:]),
            "raw": text,
            # Persist the ids. The E2 phase-1 audit could only reconstruct them
            # by re-encoding text, which is not always the tokenization the model
            # emitted; saving them makes this run exactly replayable.
            "token_ids": gen_ids,
        })
    (args.out / "generations.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    del llm
    torch.cuda.empty_cache()

    # --- teacher-forced next-token top-1 over the gold target -------------
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).cuda().eval()
    top1 = []
    with torch.no_grad():
        for item in prepared:
            ids = item["p_ids"] + item["gold_ids"]
            if len(ids) > args.context:
                continue
            t = torch.tensor([ids], device="cuda")
            logits = model(t).logits[0]
            start = len(item["p_ids"]) - 1
            pred = logits[start:len(ids) - 1].argmax(-1)
            gold = t[0, start + 1:]
            hits = int((pred == gold).sum())
            top1.append({"id": item["sample"]["id"],
                         "data_type": item["sample"]["data_type"],
                         "gold_tokens": int(gold.numel()),
                         "top1_hits": hits,
                         "top1_accuracy": round(hits / max(int(gold.numel()), 1), 4)})
    (args.out / "gold_prefix_top1.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in top1))

    # --- summary ----------------------------------------------------------
    def agg(pred):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return None
        scored = [r for r in sel if r["correct"] is not None]
        return {
            "n": len(sel),
            "correct": (round(sum(bool(r["correct"]) for r in scored) / len(scored), 4)
                        if scored else None),
            "n_scored": len(scored),
            "protocol_valid": round(sum(r["protocol_valid"] for r in sel) / len(sel), 4),
            "natural_termination": round(
                sum(r["natural_termination"] for r in sel) / len(sel), 4),
            "degeneration": round(sum(r["degenerate"] for r in sel) / len(sel), 4),
            "median_prefix_match_tokens": sorted(
                r["prefix_match_tokens"] for r in sel)[len(sel) // 2],
            "mean_prefix_match_fraction": round(sum(
                r["prefix_match_fraction"] or 0 for r in sel) / len(sel), 4),
        }

    summary = {
        "by_forced_prefix": {str(k): agg(lambda r, k=k: r["forced_prefix_tokens"] == k)
                             for k in PREFIX_LENGTHS},
        "free_generation_by_type": {
            t: agg(lambda r, t=t: r["forced_prefix_tokens"] == 0
                   and r["data_type"] == t)
            for t in sorted({r["data_type"] for r in rows})},
        "gold_prefix_top1_mean": (
            round(sum(r["top1_accuracy"] for r in top1) / len(top1), 4)
            if top1 else None),
        "gold_prefix_top1_by_type": {
            t: round(sum(r["top1_accuracy"] for r in top1 if r["data_type"] == t)
                     / max(sum(r["data_type"] == t for r in top1), 1), 4)
            for t in sorted({r["data_type"] for r in top1})},
    }
    gen_sha = hashlib.sha256(
        (args.out / "generations.jsonl").read_bytes()).hexdigest()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "model": args.model,
        "rung": args.rung,
        "n_sampled": len(picked),
        "sampled_ids": [s["id"] for s in picked],
        "seed": args.seed,
        "prefix_lengths": list(PREFIX_LENGTHS),
        "decoding": {"greedy": True, "temperature": 0.0,
                     "effective_context": args.context},
        "generations_sha256": gen_sha,
        "summary": summary,
        "code_state": code_state(REPO_ROOT),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}/  generations sha256 {gen_sha[:16]}…")


if __name__ == "__main__":
    main()
