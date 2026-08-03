#!/usr/bin/env python
"""Score stored generations against the frozen capability battery. CPU only.

Generation and scoring are separate steps on purpose: generation is the paid
part, scoring is free and re-runnable, and keeping them apart is what let the
Experiment 1 GSM8K evaluator be corrected after the fact without re-running a
single checkpoint.

Every scorer is deterministic (`aadistill.evaluation.capability`), so this is a
pure function of (generations, battery, scorer version) — all three of which it
hashes into the output.

Aggregates that need care, and are therefore computed here rather than left to
whoever reads the file:

* `refusal_paired` reports **pair accuracy**, not per-row accuracy. A model that
  refuses everything scores 0.5 per row and 0.0 per pair, and only the second
  number is honest.
* `multihop` reports answer correctness and evidence recall side by side, never
  averaged.
* `rag` reports correctness, attribution, unsupported-claim rate and echo rate as
  four numbers.
* every set reports `answer_matches_ignoring_protocol` alongside `correct`, so
  the share of the score lost to protocol and degeneration is visible instead of
  inferred.

Usage:
    scripts/evaluation/score_battery.py --battery artifacts/eval/battery_v1 \
        --generations artifacts/eval/e2/<arm> --label <arm> \
        --out artifacts/eval/e2/<arm>_battery.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.evaluation.capability import BATTERY_VERSION, SCORERS  # noqa: E402
from aadistill.evaluation.strict_answer import score_numeric  # noqa: E402
from aadistill.infrastructure.env import code_state  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file  # noqa: E402


def mean(values):
    return round(statistics.mean(values), 4) if values else None


def aggregate(name: str, verdicts: list[dict], rows: list[dict]) -> dict:
    n = len(verdicts)
    out = {
        "n": n,
        "correct": mean([v["correct"] for v in verdicts]),
        "answer_matches_ignoring_protocol": mean(
            [v.get("answer_matches_ignoring_protocol", v["correct"]) for v in verdicts]),
        "protocol_valid_rate": mean([v["protocol_valid"] for v in verdicts]),
        "natural_termination_rate": mean([v["natural_termination"] for v in verdicts]),
        "degeneration_rate": mean([v["degenerate"] for v in verdicts]),
        "reasons": dict(Counter(v["reason"] for v in verdicts).most_common()),
    }
    if name == "multihop":
        recalls = [v["evidence_recall"] for v in verdicts
                   if v.get("evidence_recall") is not None]
        out["evidence_recall"] = mean(recalls)
        out["answer_f1"] = mean([v["f1"] for v in verdicts])
    if name == "rag":
        rates = [v["unsupported_claim_rate"] for v in verdicts
                 if v.get("unsupported_claim_rate") is not None]
        out["evidence_attributed_rate"] = mean(
            [v["evidence_attributed"] for v in verdicts])
        out["unsupported_claim_rate"] = mean(rates)
        out["echo_rate"] = mean([v["echoed"] for v in verdicts])
        out["answer_f1"] = mean([v["f1"] for v in verdicts])
    if name == "math_verified":
        out["verification_paths"] = dict(
            Counter(v["verification_path"] for v in verdicts).most_common())
    if name == "refusal_paired":
        pairs = defaultdict(dict)
        for v, r in zip(verdicts, rows):
            pairs[r["pair_id"]][bool(r["answerable"])] = v
        complete = [p for p in pairs.values() if len(p) == 2]
        out["pairs"] = len(complete)
        # The headline. Per-row accuracy is 0.5 for a model that refuses
        # everything; pair accuracy is 0.
        out["pair_correct"] = mean(
            [p[True]["correct"] and p[False]["correct"] for p in complete])
        out["answerable_correct"] = mean(
            [p[True]["correct"] for p in complete])
        out["unanswerable_refused"] = mean(
            [p[False]["correct"] for p in complete])
        out["refusal_rate_on_answerable"] = mean(
            [p[True]["refused"] for p in complete])
        out["incomplete_pairs"] = len(pairs) - len(complete)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True, type=Path)
    ap.add_argument("--generations", required=True, type=Path,
                    help="dir of <set>.generations.jsonl for one checkpoint")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-sample", type=Path, default=None)
    args = ap.parse_args()

    manifest = json.loads((args.battery / "manifest.json").read_text())
    if manifest["battery_version"] != BATTERY_VERSION:
        raise SystemExit(
            f"battery is {manifest['battery_version']} but the installed scorer "
            f"is {BATTERY_VERSION} — two runs scored under different rules are "
            "not comparable; check out the matching commit rather than mixing")

    results, per_sample = {}, []
    for name in list(SCORERS) + ["gsm8k"]:
        gen_path = args.generations / f"{name}.generations.jsonl"
        set_path = args.battery / f"{name}.jsonl"
        if not gen_path.is_file() or not set_path.is_file():
            continue
        rows = {json.loads(l)["id"]: json.loads(l) for l in set_path.open()}
        verdicts, ordered = [], []
        for line in gen_path.open():
            record = json.loads(line)
            sample = rows.get(record["id"])
            if sample is None:
                raise SystemExit(
                    f"{name}: generation {record['id']!r} is not in the frozen "
                    "set — the battery and the generations disagree")
            if name == "gsm8k":
                verdict = score_numeric(record, sample["gsm8k_answer"])
                verdict.setdefault("protocol_valid", verdict["protocol_valid"])
            else:
                verdict = SCORERS[name](record, sample)
            verdicts.append(verdict)
            ordered.append(sample)
            if args.per_sample is not None:
                per_sample.append({"label": args.label, "set": name,
                                   "id": record["id"], **verdict})
        missing = set(rows) - {json.loads(l)["id"] for l in gen_path.open()}
        results[name] = aggregate(name, verdicts, ordered)
        results[name]["missing_generations"] = len(missing)

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "label": args.label,
        "battery_version": BATTERY_VERSION,
        "battery_manifest_sha256": sha256_file(args.battery / "manifest.json"),
        "scorer_sha256": sha256_file(
            REPO_ROOT / "src/aadistill/evaluation/capability.py"),
        "code_state": code_state(REPO_ROOT),
        "results": results,
        "note": ("`correct` counts a protocol-invalid or degenerate generation as "
                 "incorrect; `answer_matches_ignoring_protocol` is the same "
                 "measure without that rule, so the gap between them is the "
                 "score lost to malformed output rather than to the task."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    if args.per_sample is not None:
        args.per_sample.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample.open("w") as f:
            for row in per_sample:
                f.write(json.dumps(row) + "\n")

    print(f"{args.label}")
    for name, r in results.items():
        extra = ""
        if name == "refusal_paired":
            extra = (f"  pair {r['pair_correct']}  safe {r['answerable_correct']}"
                     f"  unsafe {r['unanswerable_refused']}")
        if name == "multihop":
            extra = f"  evidence_recall {r['evidence_recall']}"
        if name == "rag":
            extra = (f"  attributed {r['evidence_attributed_rate']}"
                     f"  unsupported {r['unsupported_claim_rate']}")
        print(f"  {name:16s} n={r['n']:4d}  correct {r['correct']}"
              f"  (ignoring protocol {r['answer_matches_ignoring_protocol']})"
              f"  degen {r['degeneration_rate']}{extra}")


if __name__ == "__main__":
    main()
