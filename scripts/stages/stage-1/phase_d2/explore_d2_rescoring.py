#!/usr/bin/env python3
"""EXPLORATORY: the rollout rescoring of a D2 candidate against its correctness.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/explore_d2_rescoring.py \
        --run <run> --store <scr>/store --rung confirmation [--write]

Joins each scorable sample's per-sample row (correct, usable) with the
rescoring diagnostic of the same sample on the candidate's own rollout
(student confidence and entropy, teacher surprisal, KL(teacher || student),
over all positions and over the reasoning and answer spans, and the
high-confidence/high-KL position fraction at the reference calibration's
thresholds). Per seed and pooled: the mean of each quantity on correct and on
incorrect samples, their difference, and the point-biserial correlation.

DESCRIPTIVE AND EXPLORATORY. It reads secured evidence only, decides nothing,
and rests on few correct samples (~3% of 850 per seed); B and q2 have no
rescoring (retired checkpoints, no token ids), so nothing here compares arms.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
QUANTITIES = (("all", "kl_teacher_student"), ("all", "teacher_surprisal"),
              ("all", "student_confidence"), ("all", "student_entropy"),
              ("reasoning", "kl_teacher_student"),
              ("answer", "student_confidence"), ("answer", "kl_teacher_student"))


def _value(summary: dict[str, Any], span: str, key: str) -> float | None:
    v = (summary.get(span) or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) \
        else None


def joined(probe_dir: Path) -> list[dict[str, Any]]:
    rows = {r["id"]: r for r in (json.loads(x) for x in (
        probe_dir / "per_sample.jsonl").read_text().splitlines() if x.strip())}
    resc = json.loads((probe_dir / "rescoring.json").read_text())
    out = []
    for r in resc.get("rows", []):
        sid = (r.get("alignment") or {}).get("id")
        row = rows.get(sid)
        if row is None or not row.get("scorable"):
            continue
        s = r.get("summary") or {}
        out.append({"id": sid, "correct": bool(row["correct"]),
                    "usable": bool(row["usable"]),
                    "exact_ids": bool((r.get("alignment") or {}).get("exact")),
                    "hchk": s.get("high_confidence_high_kl_fraction"),
                    **{f"{span}.{key}": _value(s, span, key)
                       for span, key in QUANTITIES}})
    return out


def contrast(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(samples),
                           "n_correct": sum(s["correct"] for s in samples)}
    for name in [f"{a}.{b}" for a, b in QUANTITIES] + ["hchk"]:
        pairs = [(s[name], s["correct"]) for s in samples
                 if s.get(name) is not None]
        good = [v for v, c in pairs if c]
        bad = [v for v, c in pairs if not c]
        row: dict[str, Any] = {"n": len(pairs)}
        if good and bad:
            row.update({"mean_correct": round(statistics.fmean(good), 5),
                        "mean_incorrect": round(statistics.fmean(bad), 5),
                        "difference": round(statistics.fmean(good)
                                            - statistics.fmean(bad), 5)})
            xs = [v for v, _ in pairs]
            ys = [1.0 if c else 0.0 for _, c in pairs]
            if len(set(xs)) > 1:
                row["point_biserial_r"] = round(
                    statistics.correlation(xs, ys), 4)
        out[name] = row
    return out


def analyse(session: dict[str, Any], store: Path) -> dict[str, Any]:
    per_seed, pooled = {}, []
    for probe in session.get("probes", []):
        samples = joined(store / "probes" / probe["probe_id"])
        per_seed[str(probe["seed"])] = contrast(samples)
        pooled += samples
    return {"schema": "aadistill.phase_d2.rescoring_vs_correctness/v1",
            "run_id": session.get("run_id"), "per_seed": per_seed,
            "pooled": contrast(pooled),
            "exact_engine_ids_fraction": round(
                sum(s["exact_ids"] for s in pooled) / max(1, len(pooled)), 4),
            "_role": ("EXPLORATORY and DESCRIPTIVE: no decision reads it, it "
                      "compares no arms, and its correct-sample counts are "
                      "small")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--rung", default="confirmation",
                    choices=("screening", "confirmation"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    run_dir = REPO_ROOT / "logs/stages/stage-1/phase_d2/runs" / args.run
    session = json.loads((run_dir / "evidence" / f"d2_{args.rung}.json"
                          ).read_text())
    out = analyse(session, Path(args.store))
    if args.write:
        (run_dir / "evidence" / f"d2_{args.rung}_rescoring_vs_correctness.json"
         ).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["pooled"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
