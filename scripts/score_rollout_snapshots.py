"""Score engine rollouts against the trainer policy and report importance stats.

Runs in the **project environment**, not an engine image: the numerator of an
importance ratio must come from the real training stack. The engine supplied the
denominator at sampling time and it is read from the recorded snapshot, never
recomputed — recomputation is not batch-invariant on this project's own
measurements, so a re-derived denominator would not be the policy that sampled.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from aadistill.rollout import aggregate_stats, importance_stats, score_tokens
from aadistill.teacher import load_causal_lm

ap = argparse.ArgumentParser()
ap.add_argument("--reports", nargs="+", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--limit", type=int, default=4, help="sequences per engine to score")
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--out", required=True)
args = ap.parse_args()

dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
model, _ = load_causal_lm(args.model, dtype, "cpu")
print(f"trainer policy loaded ({args.dtype}, cpu)", flush=True)

out = {}
for path in args.reports:
    report = json.loads(Path(path).read_text())
    engine = report["engine"] + ("_deterministic" if "det" in Path(path).stem else "")
    per_seq = []
    for record in report.get("rollouts", [])[: args.limit]:
        tokens = record["tokens"][: args.max_tokens]
        rollout_lp = (record.get("logprobs") or [])[: args.max_tokens]
        if not tokens or len(rollout_lp) != len(tokens):
            continue
        trainer_lp = score_tokens(model, record["prompt_tokens"], tokens)
        stats = importance_stats(rollout_lp, trainer_lp)
        stats["prompt_id"] = record["prompt_id"]
        per_seq.append(stats)
        print(f"  {engine} {record['prompt_id'][:28]:28s} n={stats['n']:>4} "
              f"median={stats['ratio_median']} p99={stats['ratio_p99']} "
              f"off={stats['off_policy_rate']} kl={stats['kl']}", flush=True)
    out[engine] = {"per_sequence": per_seq, "aggregate": aggregate_stats(per_seq)}
    print(f"{engine}: {json.dumps(out[engine]['aggregate'])}\n", flush=True)

Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
print(f"wrote {args.out}")
