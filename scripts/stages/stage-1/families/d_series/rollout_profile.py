#!/usr/bin/env python3
"""Text-level profile of behavioural rollouts, at $0, from secured generations.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/families/d_series/rollout_profile.py [--write]

What a unit's autonomous rollouts look like, beside whether they were right:
generated length (the engine's exact count), reasoning and answer length,
natural termination, degeneration, and correctness by reasoning length; and,
for two arms measured on the same prompts and seeds, which prompts each got
right and how their lengths differ when both did.

**What this can and cannot see.** The secured generations hold the decoded
text and the engine's token count -- no token ids, no log-probabilities, no
logits. So every quantity here is TEXT-LEVEL. Student confidence, student-
prefix KL and per-position anything are NOT recoverable for a unit whose
checkpoint is retired, and nothing here infers them. Reasoning and answer
lengths are re-tokenized from the text, which reproduces the engine's total
exactly for most samples and not all; the record reports how many, and the
engine's own count is used wherever a total is meant.

Generic over units: it reads a generations directory and a per-sample file.
This module's `main` profiles the ten imported B/q2 units of the shared
protocol as the descriptive baseline D2/D3 candidates are read against.
DESCRIPTIVE ONLY: no rule reads these numbers and no claim is made from them.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[5]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aadistill.evaluation.behavior import split_generation  # noqa: E402

SCHEMA = "aadistill.d_series.rollout_profile/v1"
RECORD = "logs/stages/stage-1/families/d_series/analyses/d1_rollout_profile.json"
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _q(values: Sequence[float]) -> dict[str, float] | None:
    v = sorted(values)
    if not v:
        return None
    out = {}
    for q in QUANTILES:
        out[f"p{int(q * 100):02d}"] = float(v[min(len(v) - 1,
                                                 int(q * len(v)))])
    out["mean"] = float(statistics.fmean(v))
    out["n"] = len(v)
    return out


def unit_rows(generations_dir: Path, per_sample: Iterable[Mapping[str, Any]],
              tokenizer) -> list[dict[str, Any]]:
    """One row per sample: the per-sample score joined to its generation."""
    by_id = {str(r["id"]): r for r in per_sample}
    rows = []
    for path in sorted(Path(generations_dir).glob("*.generations.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            score = by_id.get(str(g["id"]))
            if score is None:
                raise ValueError(f"{g['id']}: a generation with no per-sample "
                                 "row; the two files do not describe one unit")
            raw = g.get("raw") or ""
            parts = split_generation(raw, think_preopened=bool(
                g.get("think_preopened", True)))
            n_raw = len(tokenizer(raw, add_special_tokens=False)["input_ids"])
            rows.append({
                "id": str(g["id"]), "set": score.get("set"),
                "scorable": bool(score.get("scorable")),
                "correct": bool(score.get("correct")),
                "usable": bool(score.get("usable")),
                "natural_termination": bool(g.get("natural_termination")),
                "stop_reason": g.get("stop_reason"),
                "context_limit_reached": bool(g.get("context_limit_reached")),
                "generated_tokens": int(g["generated_tokens"]),
                "think_closed": bool(parts["think_closed"]),
                "reasoning_tokens": len(tokenizer(
                    parts["think"] or "", add_special_tokens=False)["input_ids"]),
                "answer_tokens": len(tokenizer(
                    parts["answer"] or "", add_special_tokens=False)["input_ids"]),
                "retokenization_exact": n_raw == int(g["generated_tokens"]),
            })
    if len(rows) != len(by_id):
        raise ValueError(f"{len(rows)} generations for {len(by_id)} per-sample "
                         "rows")
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    scorable = [r for r in rows if r["scorable"]]
    closed = [r for r in scorable if r["think_closed"]]
    by_quartile = None
    if len(closed) >= 8:
        lengths = sorted(r["reasoning_tokens"] for r in closed)
        edges = [lengths[int(q * len(lengths))] for q in (0.25, 0.5, 0.75)]
        bins: list[list[bool]] = [[], [], [], []]
        for r in closed:
            k = sum(r["reasoning_tokens"] > e for e in edges)
            bins[k].append(r["correct"])
        by_quartile = {"edges_reasoning_tokens": edges,
                       "correct_rate": [round(sum(b) / len(b), 4) if b else None
                                        for b in bins],
                       "n": [len(b) for b in bins]}
    return {
        "n": n, "n_scorable": len(scorable),
        "correct_rate_scorable": round(sum(r["correct"] for r in scorable)
                                       / len(scorable), 4) if scorable else None,
        "usable_rate": round(sum(r["usable"] for r in rows) / n, 4),
        "natural_termination_rate": round(
            sum(r["natural_termination"] for r in rows) / n, 4),
        "degeneration_rate": round(
            sum(r["stop_reason"] == "degeneration" for r in rows) / n, 4),
        "context_limit_rate": round(
            sum(r["context_limit_reached"] for r in rows) / n, 4),
        "think_closed_rate": round(sum(r["think_closed"] for r in rows) / n, 4),
        "generated_tokens": {
            "all": _q([r["generated_tokens"] for r in rows]),
            "correct": _q([r["generated_tokens"] for r in scorable
                           if r["correct"]]),
            "usable_incorrect": _q([r["generated_tokens"] for r in scorable
                                    if r["usable"] and not r["correct"]]),
            "unusable": _q([r["generated_tokens"] for r in rows
                            if not r["usable"]])},
        "reasoning_tokens_think_closed": {
            "all": _q([r["reasoning_tokens"] for r in rows if r["think_closed"]]),
            "correct": _q([r["reasoning_tokens"] for r in closed if r["correct"]]),
            "incorrect": _q([r["reasoning_tokens"] for r in closed
                             if not r["correct"]])},
        "answer_tokens_think_closed": _q([r["answer_tokens"] for r in rows
                                          if r["think_closed"]]),
        "correct_by_reasoning_length_quartile": by_quartile,
        "retokenization_exact_fraction": round(
            sum(r["retokenization_exact"] for r in rows) / n, 4),
    }


def paired(candidate: Mapping[int, Sequence[Mapping[str, Any]]],
           reference: Mapping[int, Sequence[Mapping[str, Any]]]
           ) -> dict[str, Any]:
    """Same prompt, same seed: who got it right, and the length difference."""
    if sorted(candidate) != sorted(reference):
        raise ValueError("the two arms do not share their seeds")
    both = cand_only = ref_only = neither = 0
    delta_all, delta_both_correct, delta_cand_only = [], [], []
    for seed in sorted(candidate):
        ref = {r["id"]: r for r in reference[seed]}
        for c in candidate[seed]:
            r = ref[c["id"]]
            d = c["generated_tokens"] - r["generated_tokens"]
            delta_all.append(d)
            if not (c["scorable"] and r["scorable"]):
                continue
            if c["correct"] and r["correct"]:
                both += 1
                delta_both_correct.append(d)
            elif c["correct"]:
                cand_only += 1
                delta_cand_only.append(d)
            elif r["correct"]:
                ref_only += 1
            else:
                neither += 1
    return {"pairs_scorable": both + cand_only + ref_only + neither,
            "both_correct": both, "candidate_only_correct": cand_only,
            "reference_only_correct": ref_only, "neither_correct": neither,
            "generated_token_delta": {
                "all": _q(delta_all), "both_correct": _q(delta_both_correct),
                "candidate_only_correct": _q(delta_cand_only)},
            "_reading": ("candidate minus reference, same prompt and seed; "
                         "descriptive, pooled over the seeds")}


def build(repo_root: Path = REPO_ROOT, evidence_root: Path | None = None,
          tokenizer=None) -> dict[str, Any]:
    from stages.d_series import shared_evaluation as S

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Thinking-2507",
            revision="768f209d9ea81521153ed38c47d515654e938aea")
    units = S.historical_units(repo_root, evidence_root)
    by: dict[tuple[str, str], dict[int, list[dict]]] = {}
    summaries = {}
    for unit in units:
        store = S.durable_store(unit.rung, repo_root, evidence_root)
        probe = store / "store" / "probes" / unit.probe_id
        per_sample = [json.loads(x) for x in
                      (probe / "per_sample.jsonl").read_text().splitlines()
                      if x.strip()]
        rows = unit_rows(probe / "generations", per_sample, tokenizer)
        by.setdefault((unit.rung, unit.arm), {})[unit.seed] = rows
        summaries[unit.probe_id] = summarize(rows)
    pooled = {f"{rung}:{arm}": summarize([r for rows in seeds.values()
                                          for r in rows])
              for (rung, arm), seeds in sorted(by.items())}
    pairs = {rung: paired(by[(rung, "q2")], by[(rung, "B")])
             for rung in S.RUNGS}
    return {
        "schema": SCHEMA,
        "_contract": (
            "DESCRIPTIVE text-level profile of the ten imported B/q2 units. "
            "No rule reads it and no claim is made from it. Student "
            "confidence, student-prefix KL and per-position quantities are "
            "NOT available for these units: their checkpoints are retired and "
            "their generations carry no ids or log-probabilities."),
        "tokenizer": "Qwen/Qwen3-4B-Thinking-2507@768f209d (re-tokenization "
                     "for reasoning/answer split only)",
        "per_unit": summaries,
        "pooled_over_seeds": pooled,
        "q2_vs_b_paired": pairs,
        "availability": {
            "engine_generated_tokens": "exact, per sample",
            "decoded_text_with_think_and_answer": "complete, per sample",
            "token_ids": "NOT stored",
            "per_token_logprobs_or_logits": "NOT stored",
            "student_confidence_on_own_rollouts": (
                "NOT obtainable: requires the trained checkpoint, retired"),
            "teacher_surprisal_on_these_rollouts": (
                "obtainable on a GPU from the text by a teacher forward, as a "
                "re-tokenized approximation of the generated sequence; not "
                "computed"),
        },
        "authorizes": "nothing",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--evidence-root", type=Path, default=None)
    args = ap.parse_args(argv)
    record = build(evidence_root=args.evidence_root)
    body = json.dumps(record, indent=1) + "\n"
    path = REPO_ROOT / RECORD
    if args.write:
        path.write_text(body)
        print(f"wrote {RECORD}")
        return 0
    same = path.is_file() and path.read_text() == body
    print(f"{RECORD}: {'current' if same else 'STALE (run with --write)'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
