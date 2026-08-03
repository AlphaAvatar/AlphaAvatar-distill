"""Curate a readable sample of Experiment 1 generations for human review.

The sweep saves every generation, which is the right thing for reproducibility
and the wrong thing for reading: 25 checkpoints x 176 prompts is ~4,400 raw
outputs. This picks a stratified, deterministic sample so a person can judge
the model's actual behaviour rather than only its aggregate scores.

Stratified over the things that differ, not at random:
  * stop reason — natural EOS, and each degeneration kind (cycle, low_novelty,
    rambling), plus any context-limit hits
  * data rung and initialization, so the reader sees the full span
  * on GSM8K, correct and incorrect answers side by side

Writes two files:
  * a Markdown file to read
  * a JSONL file with the same cases, for programmatic analysis

    uv run python scripts/evaluation/build_test_cases.py \
        --eval-dir artifacts/eval/e1 --out logs/e1_test_cases.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MAX_CHARS = 2000   # per generation in the Markdown; the JSONL keeps it whole


def load_prompts(paths: list[Path]) -> dict:
    """id -> prompt sample, so a case shows what was actually asked.

    The generation records carry ids, not prompts; without this join the file
    is unreadable for review, which is the whole point of it.
    """
    out = {}
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def load(eval_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(eval_dir.glob("*.generations.jsonl")):
        suite = "gsm8k" if "_gsm8k" in p.name else "behavior"
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["suite"] = suite
            rows.append(r)
    return rows


def arm_sort_key(arm: str) -> tuple:
    """Order arms by rung, then seed, then init — the axes of the experiment."""
    rung = 0
    for tag, val in (("0250k", 1), ("0460k", 2), ("0860k", 3),
                     ("1600k", 4), ("2960k", 5), ("5500k", 6)):
        if tag in arm:
            rung = val
    return (0 if "ctl" not in arm else 1, rung, "sb" in arm, "rand" in arm)


def pick(rows: list[dict], per_bucket: int) -> list[dict]:
    """Deterministic stratified pick: (suite, stop class) x spread of arms."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        kind = r.get("degeneration_kind") or r["stop_reason"]
        # keep the key sortable: `answer_correct` is None for non-GSM8K rows,
        # and bool vs None cannot be ordered.
        correct = {True: "correct", False: "incorrect"}.get(
            r.get("answer_correct"), "n/a")
        buckets[(r["suite"], kind, correct)].append(r)
    chosen = []
    for key in sorted(buckets):
        rs = sorted(buckets[key], key=lambda r: (arm_sort_key(r["label"]), r["id"]))
        # spread across arms rather than taking the first N of one arm
        step = max(1, len(rs) // per_bucket)
        chosen.extend(rs[::step][:per_bucket])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="artifacts/eval/e1")
    ap.add_argument("--out", default="logs/e1_test_cases.md")
    ap.add_argument("--per-bucket", type=int, default=6)
    ap.add_argument("--prompts", nargs="*",
                    default=["data/eval_behavior_v0/prompts.jsonl",
                             "artifacts/eval/e1/gsm8k_reasoning_100.jsonl"],
                    help="prompt sets to join by id, so cases show the question")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    rows = load(eval_dir)
    prompts = load_prompts([Path(x) for x in args.prompts])
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from aadistill.evaluation.behavior import final_number, split_generation
    for r in rows:
        src = prompts.get(r["id"], {})
        r["prompt_text"] = src.get("prompt_text") or (
            src.get("messages", [{}])[0].get("content") if src.get("messages") else None)
        if r["suite"] == "gsm8k" and src.get("gsm8k_answer") is not None:
            answer = split_generation(r.get("raw", "")).get("answer", "")
            r["gold_answer_value"] = src["gsm8k_answer"]
            r["predicted_answer_value"] = final_number(answer)
            r["answer_correct"] = r["predicted_answer_value"] == src["gsm8k_answer"]
    if not rows:
        raise SystemExit(f"no *.generations.jsonl under {eval_dir}")
    cases = pick(rows, args.per_bucket)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(".jsonl")
    with open(jsonl, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.get("degeneration_kind") or r["stop_reason"]] += 1

    lines = [
        "# Experiment 1 — sample generations for review",
        "",
        f"{len(cases)} cases selected from {len(rows)} generations across "
        f"{len({r['label'] for r in rows})} checkpoints, stratified by stop "
        "reason and spread across rungs, seeds and initializations. Selection is "
        "deterministic (no sampling), so this file regenerates identically.",
        "",
        "**Protocol for every case:** uncapped within an effective context of "
        "**8,192**, derived from the trained `block_len` — *not* the "
        "architectural 262,144. Greedy decoding, mandatory system message, one "
        "fixed degeneration detector with identical thresholds for all "
        "checkpoints.",
        "",
        "Population by outcome: "
        + ", ".join(f"`{k}` {v}" for k, v in sorted(counts.items())),
        "",
        f"Machine-readable copy with untruncated text: `{jsonl.name}`",
        "",
        "---",
        "",
    ]
    for i, c in enumerate(cases, 1):
        raw = c.get("raw", "")
        trunc = len(raw) > MAX_CHARS
        body = raw[:MAX_CHARS] + ("\n…[truncated for reading; full text in the "
                                  "jsonl]" if trunc else "")
        lines += [
            f"## {i}. `{c['label']}` — {c['suite']} — {c['id']}",
            "",
            f"- stop reason: **{c['stop_reason']}**"
            + (f" (`{c['degeneration_kind']}`)" if c.get("degeneration_kind") else ""),
            f"- generated tokens: {c['generated_tokens']} of "
            f"{c['generation_allowance']} allowed; prompt {c['prompt_tokens']}",
            f"- right-censored: {c['right_censored']}",
        ] + ([
            f"- gsm8k: gold **{c.get('gold_answer_value')}**, predicted "
            f"**{c.get('predicted_answer_value')}** -> "
            f"{'CORRECT' if c.get('answer_correct') else 'incorrect'}",
        ] if c["suite"] == "gsm8k" and "gold_answer_value" in c else []) + [
            "",
            "**Prompt**", "",
            "```text",
            (c.get("prompt_text") or "")[:800] or "(see prompt set)",
            "```",
            "",
            "**Generation**", "",
            "```text", body, "```", "", "---", "",
        ]
    out.write_text("\n".join(lines))
    print(f"{len(cases)} cases -> {out} and {jsonl}")


if __name__ == "__main__":
    main()
