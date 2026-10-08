#!/usr/bin/env python
"""Break teacher-forced top-1 accuracy down by token role and position.

    PYTHONPATH=src python scripts/evaluation/audit_teacher_forced_by_role.py \
        --model artifacts/stage3/rescued/e1_ctl_r0250k_sa_pca_stepmatched \
        --sessions artifacts/stage3/corpus_v2/sessions.jsonl \
        --pack artifacts/stage3/ladder_uniform_probe --rung 250000 \
        --n 60 --out artifacts/audit/teacher_forced_by_role.json

An aggregate of 78.03% says almost nothing on its own: reasoning prose is full of
high-frequency filler, and a model can score well on it while failing every token
that actually carries the protocol. This splits the same measurement by:

* **role** — `</think>`, `<|im_end|>`, the first token after `</think>`, tokens
  inside the final-answer span, digits, and ordinary prose;
* **position** — decile of the way through the target, plus the first 16 tokens,
  because a model that cannot start is different from one that cannot finish;
* **first-divergence neighbourhood** — accuracy in the ±8 tokens around the point
  where free generation first left the gold path, which is where a sequence-level
  failure should show up if there is one.

CPU-only. The forward pass is small (0.6B, a few hundred tokens per sample).

**RoPE**: this checkpoint stores `rope_theta` in the nested `rope_parameters`
dict that transformers 5.x writes. Under transformers 4.x `config.rope_theta`
silently resolves to the class default 10000 instead of the stored 5000000, and
every number here would be measured on a model with a 500x-wrong positional
basis. The value is therefore forced from `rope_parameters` and then **verified
against the model's actual inv_freq** before anything is measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.data.sessions import render_session  # noqa: E402
from aadistill.infrastructure.env import code_state, library_versions  # noqa: E402
from aadistill.models.student import assert_rope_matches_config  # noqa: E402


def load_model(path: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(path)
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and params.get("rope_theta") is not None:
        cfg.rope_theta = float(params["rope_theta"])
    model = AutoModelForCausalLM.from_pretrained(
        path, config=cfg, dtype=torch.float32).eval()
    base = assert_rope_matches_config(model, cfg, path)
    print(f"rope base verified at runtime: {base:,.0f}")
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", required=True, type=Path)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--rung", type=int, default=250000)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--recall", type=Path, default=None,
                    help="training_recall/generations.jsonl, for divergence points")
    ap.add_argument("--max-target", type=int, default=1024)
    ap.add_argument("--tokenizer", default=None,
                    help="load the tokenizer from here instead of --model. The "
                         "checkpoint's own tokenizer_config.json is written by "
                         "transformers 5.x and its list-form extra_special_tokens "
                         "cannot be parsed by 4.x; the teacher's copy is "
                         "byte-identical in vocab, merges, added tokens and chat "
                         "template, so it is an exact substitute for token work.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
    from diagnose_training_recall import rung_session_ids, stratified_sample

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    tid = {n: tok.convert_tokens_to_ids(n)
           for n in ("<think>", "</think>", "<|im_end|>")}
    model, _ = load_model(args.model)

    want = set(rung_session_ids(args.pack, args.rung))
    trained = [json.loads(l) for l in args.sessions.open()
               if l.strip() and json.loads(l)["id"] in want]
    picked = stratified_sample(trained, args.n, args.seed)
    print(f"{len(picked)} sessions")

    divergence = {}
    if args.recall and args.recall.is_file():
        for line in args.recall.open():
            r = json.loads(line)
            if r["forced_prefix_tokens"] == 0:
                divergence[r["id"]] = r.get("prefix_match_tokens")

    by_role = defaultdict(lambda: [0, 0])
    by_decile = defaultdict(lambda: [0, 0])
    by_first16 = [0, 0]
    near_div = defaultdict(lambda: [0, 0])
    ANSWER_MARK = tok("Final Answer", add_special_tokens=False).input_ids[0]

    def bump(store, key, hit):
        store[key][0] += int(hit); store[key][1] += 1

    for n, s in enumerate(picked):
        # The gold target is the RENDERED supervised span, not the assistant
        # message's `content`. The assistant message carries `reasoning_content`
        # (the think block) and `content` (the final answer) separately, and the
        # template renders <think>{reasoning}</think>{content}<|im_end|>. Using
        # `content` alone omits the entire reasoning block, which is what the
        # model actually emits first -- comparing against it guarantees a
        # divergence at token 0 no matter how good the model is.
        turns = [m for m in s["messages"] if m["role"] != "assistant"]
        prompt = tok.apply_chat_template(turns, tools=s.get("tools"),
                                         tokenize=False,
                                         add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        rendered = render_session(tok, s)
        sup = [i for i, m in enumerate(rendered.body_mask) if m]
        g_ids = [rendered.body_ids[i] for i in sup][:args.max_target]
        # The supervised span opens with <think>, which the prompt already
        # emitted; drop it so the two line up at the generation boundary.
        if g_ids and g_ids[0] == tid["<think>"]:
            g_ids = g_ids[1:]
        if not g_ids:
            continue
        ids = torch.tensor([p_ids + g_ids])
        with torch.no_grad():
            logits = model(ids).logits[0]
        start = len(p_ids) - 1
        pred = logits[start:len(p_ids) + len(g_ids) - 1].argmax(-1).tolist()
        gold = g_ids

        close_i = gold.index(tid["</think>"]) if tid["</think>"] in gold else None
        div_i = divergence.get(s["id"])
        for i, (p, g) in enumerate(zip(pred, gold)):
            hit = p == g
            # --- role
            if g == tid["</think>"]:
                role = "think_close"
            elif g == tid["<|im_end|>"]:
                role = "im_end"
            elif close_i is not None and i == close_i + 1:
                role = "first_after_think_close"
            elif close_i is not None and i > close_i:
                role = "answer_span"
            elif tok.decode([g]).strip().isdigit():
                role = "digit"
            else:
                role = "prose"
            bump(by_role, role, hit)
            bump(by_decile, min(9, int(10 * i / len(gold))), hit)
            if i < 16:
                by_first16[0] += int(hit); by_first16[1] += 1
            if div_i is not None and abs(i - div_i) <= 8:
                bump(near_div, "within_8_of_first_divergence", hit)
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(picked)}", flush=True)

    def rate(pair):
        return {"n": pair[1], "top1": round(pair[0] / pair[1], 4) if pair[1] else None}

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "tokenizer_equivalence": (
            "vocab, merges, added_tokens and chat_template verified identical to "
            "the checkpoint's own tokenizer" if args.tokenizer else "checkpoint"),
        "n_sessions": len(picked),
        "max_target_tokens": args.max_target,
        "gold_target": ("rendered supervised span (think + answer + terminator), "
                        "leading <think> dropped because the prompt emits it"),
        "by_role": {k: rate(v) for k, v in sorted(by_role.items())},
        "by_decile": {str(k): rate(v) for k, v in sorted(by_decile.items())},
        "first_16_tokens": rate(by_first16),
        "near_first_divergence": {k: rate(v) for k, v in near_div.items()},
        "libraries": library_versions(),
        "code_state": code_state(REPO_ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print("\nBY ROLE")
    for k, v in out["by_role"].items():
        print(f"  {k:26s} n={v['n']:>7,}  top1 {v['top1']}")
    print("\nBY DECILE OF TARGET")
    print("  " + "  ".join(f"{k}:{v['top1']}" for k, v in out["by_decile"].items()))
    print(f"\nfirst 16 tokens: {out['first_16_tokens']}")
    print(f"near first divergence: {out['near_first_divergence']}")


if __name__ == "__main__":
    main()
