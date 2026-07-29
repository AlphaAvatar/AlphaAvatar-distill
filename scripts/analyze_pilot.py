"""Analyze a teacher-generation pilot to decide slice-level policy.

    uv run python scripts/analyze_pilot.py --pilot artifacts/stage2_v2/pilot

The 2026-07-29 pilot produced two slice failures that look alike in the summary
(low accept@n) but have opposite causes, and the summary cannot tell them apart:

* `openmath` rejected 28/40 as `truncated_at_cap` — a **budget** failure, where
  most candidates never reached an answer at all;
* `refusal_uncertainty` rejected 29/40 as `refusal_too_long` — a **protocol**
  failure, where the answer arrived and was the wrong shape.

Deciding what to do about either needs the distributions behind the counts, not
the counts: whether a higher cap would actually recover openmath, and whether
the teacher's refusals are 70 words or 700. This script reads the recorded
candidates and answers exactly those questions, so the decision is reproducible
from a hashed artifact rather than from an impression (P4).

It is read-only and rewrites nothing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.behavior import normalize_text
from aadistill.verify import REFUSAL_MAX_WORDS

# Thresholds probed for the refusal rule. The question is not "what is the
# teacher's median length" but "is there a threshold that recovers most
# candidates without accepting answers too long to be a realtime refusal".
REFUSAL_THRESHOLDS = (60, 80, 100, 150, 200, 300)


def load(pilot: Path) -> list[dict]:
    with open(pilot / "candidates.jsonl") as f:
        return [json.loads(line) for line in f]


def quantiles(values: list[int]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    return {
        "n": len(values),
        "min": values[0],
        "p25": values[len(values) // 4],
        "median": int(statistics.median(values)),
        "p75": values[3 * len(values) // 4],
        "max": values[-1],
    }


def answer_words(candidate: dict) -> int:
    return len(normalize_text(candidate["answer"]).split())


def report_slice(name: str, rows: list[dict]) -> None:
    candidates = [c for row in rows for c in row["candidates"]]
    reasons = Counter(c["reason"] for c in candidates)
    print(f"\n=== {name} — {len(rows)} prompts, {len(candidates)} candidates ===")
    print("  reasons:", dict(reasons.most_common()))
    print("  think_tokens:", quantiles([c["think_tokens"] for c in candidates]))


def refusal_analysis(rows: list[dict]) -> None:
    candidates = [c for row in rows for c in row["candidates"]]
    too_long = [c for c in candidates if c["reason"] == "refusal_too_long"]
    not_refusal = [c for c in candidates if c["reason"] == "not_a_refusal"]

    print("\n--- refusal_uncertainty: how long are the rejected refusals? ---")
    print("  answer words (refusal_too_long):", quantiles([answer_words(c) for c in too_long]))
    print(f"  current threshold: {REFUSAL_MAX_WORDS} words")

    # Threshold sensitivity: of candidates the detector already calls refusals,
    # how many would pass at each cut? A threshold only helps if the mass sits
    # just above the current line rather than far beyond it.
    detected = [c for c in candidates if c["reason"] in ("ok", "refusal_too_long")]
    print(f"  candidates detected as refusals: {len(detected)}/{len(candidates)}")
    for threshold in REFUSAL_THRESHOLDS:
        passing = sum(answer_words(c) <= threshold for c in detected)
        prompts_covered = sum(
            any(c["reason"] in ("ok", "refusal_too_long") and answer_words(c) <= threshold
                for c in row["candidates"])
            for row in rows
        )
        print(f"    <= {threshold:>3} words: {passing:>3}/{len(candidates)} candidates, "
              f"accept@n would be {prompts_covered}/{len(rows)} = "
              f"{prompts_covered / len(rows):.3f}")

    if not_refusal:
        print(f"\n  `not_a_refusal` ({len(not_refusal)}): the teacher answered instead of "
              "declining, or phrased the refusal outside the detector's vocabulary.")
        for c in not_refusal[:3]:
            snippet = " ".join(c["answer"].split())[:150]
            print(f"    - [{answer_words(c):>3}w] {snippet}…")


def openmath_analysis(rows: list[dict]) -> None:
    candidates = [c for row in rows for c in row["candidates"]]
    truncated = [c for c in candidates if c["reason"] == "truncated_at_cap"]
    finished = [c for c in candidates if c["reason"] != "truncated_at_cap"]

    print("\n--- openmath: is the cap the binding constraint? ---")
    print(f"  truncated at cap: {len(truncated)}/{len(candidates)}")
    print("  think_tokens among candidates that FINISHED:",
          quantiles([c["think_tokens"] for c in finished]))
    if finished:
        accepted = sum(c["accepted"] for c in finished)
        print(f"  accuracy among finished candidates: {accepted}/{len(finished)} = "
              f"{accepted / len(finished):.3f}")
        print("    ^ this is the number that separates 'needs a longer budget' from "
              "'gets it wrong'. High accuracy here means the cap is the problem.")
    # Per-prompt: did any candidate finish at all?
    none_finished = sum(
        all(c["reason"] == "truncated_at_cap" for c in row["candidates"]) for row in rows)
    print(f"  prompts where NO candidate finished: {none_finished}/{len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="artifacts/stage2_v2/pilot")
    args = ap.parse_args()

    rows = load(REPO_ROOT / args.pilot)
    by_slice: dict[str, list[dict]] = {}
    for row in rows:
        by_slice.setdefault(row["slice"], []).append(row)

    for name in sorted(by_slice):
        report_slice(name, by_slice[name])

    if "refusal_uncertainty" in by_slice:
        refusal_analysis(by_slice["refusal_uncertainty"])
    if "openmath" in by_slice:
        openmath_analysis(by_slice["openmath"])


if __name__ == "__main__":
    main()
