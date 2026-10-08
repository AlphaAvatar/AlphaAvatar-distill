#!/usr/bin/env python
"""Re-score stored GSM8K generations under the strict final-answer rule.

Offline and CPU-only: it reads the `*_gsm8k.generations.jsonl` files an
evaluation run already wrote and re-derives exact match with
`aadistill.evaluation.strict_answer`, which prefers `\\boxed{…}`, otherwise
requires an explicit `Final Answer:` / `Answer:` marker, never reads a number
out of a tool call, and refuses to credit a protocol-invalid or degenerate
generation.

Nothing is regenerated, so a historical control can be re-scored with the
corrected evaluator without retraining or re-running it — which is what makes
the Experiment 1 arms reusable as controls.

Usage:
    scripts/evaluation/rescore_gsm8k.py --eval-dir artifacts/eval/e1 \
        --prompts artifacts/eval/e1/gsm8k_reasoning_100.jsonl \
        --out artifacts/eval/e1/gsm8k_strict_rescore.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation.strict_answer import score_numeric  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-sample", type=Path, default=None,
                    help="optional jsonl of every per-sample verdict")
    args = ap.parse_args()

    gold = {}
    with args.prompts.open() as f:
        for line in f:
            row = json.loads(line)
            gold[row["id"]] = row["gsm8k_answer"]

    arms = {}
    per_sample_rows = []
    for path in sorted(args.eval_dir.glob("*_gsm8k.generations.jsonl")):
        arm = path.name[: -len("_gsm8k.generations.jsonl")]
        verdicts, reasons, extractions = [], Counter(), Counter()
        lenient_hits = 0
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                verdict = score_numeric(record, gold[record["id"]])
                verdicts.append(verdict)
                reasons[verdict["reason"]] += 1
                extractions[verdict["extraction"]] += 1
                lenient_hits += int(verdict["answer_matches_ignoring_protocol"])
                if args.per_sample is not None:
                    per_sample_rows.append({"arm": arm, "id": record["id"], **verdict})
        n = len(verdicts)
        arms[arm] = {
            "n": n,
            "strict_em": round(sum(v["correct"] for v in verdicts) / n, 4),
            "answer_match_ignoring_protocol": round(lenient_hits / n, 4),
            "protocol_valid_rate": round(
                sum(v["protocol_valid"] for v in verdicts) / n, 4),
            "final_answer_present_rate": round(
                sum(v["extraction"] != "no_final_answer" for v in verdicts) / n, 4),
            "natural_termination_rate": round(
                sum(v["natural_termination"] for v in verdicts) / n, 4),
            "degeneration_rate": round(sum(v["degenerate"] for v in verdicts) / n, 4),
            "extraction_paths": dict(extractions.most_common()),
            "reasons": dict(reasons.most_common()),
        }

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "rule": ("boxed > explicit Final Answer:/Answer: marker; tool-call "
                 "payloads stripped; protocol-invalid or degenerate generations "
                 "are incorrect regardless of content"),
        "prompts": {"path": str(args.prompts), "sha256": sha256_file(args.prompts),
                    "n": len(gold)},
        "scorer_sha256": sha256_file(
            REPO_ROOT / "src/aadistill/evaluation/strict_answer.py"),
        "code_state": code_state(REPO_ROOT),
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    if args.per_sample is not None:
        args.per_sample.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample.open("w") as f:
            for row in per_sample_rows:
                f.write(json.dumps(row) + "\n")

    print(f"{'arm':38s} {'strict':>7} {'lenient':>8} {'proto':>7} {'final':>7} "
          f"{'natterm':>8} {'degen':>7}")
    for arm, row in sorted(arms.items()):
        print(f"{arm:38s} {row['strict_em']:7.3f} "
              f"{row['answer_match_ignoring_protocol']:8.3f} "
              f"{row['protocol_valid_rate']:7.3f} "
              f"{row['final_answer_present_rate']:7.3f} "
              f"{row['natural_termination_rate']:8.3f} "
              f"{row['degeneration_rate']:7.3f}")


if __name__ == "__main__":
    main()
