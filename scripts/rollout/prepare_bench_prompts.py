"""Pre-tokenize a slice-balanced prompt set for the pod-side engine benchmark.

Runs on the dev box, where the tokenizer, chat template and dataset live. The
engine images have none of those, so the benchmark ships token ids instead —
which also removes any chance of the two arms rendering prompts differently.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from transformers import AutoTokenizer, GenerationConfig
from generate_teacher_answers import SLICES, generation_prompt, load_slice

IN_SCOPE = ("rag_evidence", "multihop_qa", "gsm8k", "openmath")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
ap.add_argument("--revision", default="768f209d9ea81521153ed38c47d515654e938aea")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--out", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
gen = GenerationConfig.from_pretrained(args.model, revision=args.revision)
eos = gen.eos_token_id
stops = sorted({*(eos if isinstance(eos, (list, tuple)) else [eos]), tok.eos_token_id}
               - {None})

per = max(1, args.n // len(IN_SCOPE))
rows = []
for name in IN_SCOPE:
    samples = load_slice(REPO_ROOT / "data/stage2_v1", name, limit=per * 4)
    stride = max(1, len(samples) // per)
    for s in samples[::stride][:per]:
        ids = tok(generation_prompt(tok, s), add_special_tokens=False).input_ids
        rows.append({"id": s["id"], "slice": name, "ids": ids})
rows = rows[: args.n]

body = {"model": args.model, "revision": args.revision, "stop_ids": stops,
        "slices": IN_SCOPE, "prompts": rows,
        "mean_prompt_tokens": round(sum(len(r["ids"]) for r in rows) / len(rows), 1)}
body["sha256"] = hashlib.sha256(
    json.dumps(body["prompts"], sort_keys=True).encode()).hexdigest()
Path(args.out).write_text(json.dumps(body))
print(f"{len(rows)} prompts, stops={stops}, mean {body['mean_prompt_tokens']} tokens")
print(f"sha256 {body['sha256'][:16]}…  -> {args.out}")
