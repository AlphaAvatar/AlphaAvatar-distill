#!/usr/bin/env python3
"""The D-series Top-K adoption validation. Engineering evidence only.

    python scripts/pod/topk_adoption_driver.py --out DIR [--deadline-s N]

**One set of model forwards, two reductions.** Every comparison below comes from
the SAME logits: paying for a second set of forwards to compare two reducers
would be buying nothing. What it answers, against the completed full-vocabulary
qualification as the baseline:

```text
A  the reference Top-200 probability mass -- distribution, not just the mean
B  full-vocabulary vs K+1 KL: absolute, relative, rank, worst case
C  DEPTH discrete decisions, both scores per candidate from shared forwards
D  state evaluation: equal-domain KL, worst domain, critical-token KL, ordering
E  performance: sketch bytes, recomputes, wall clock, peak memory
```

**A moved decision is not a failure.** This is an explicitly authorized new
scientific protocol (maintainer decision 2026-10-04), so a difference means the
D-series protocol is observably different from the historical full-vocabulary one
— which is the thing being characterized, not a defect.

**What C can and cannot say.** The greedy search is run ONCE and chooses by the
FULL-VOCABULARY score, so the trajectory is the historical one and both scores are
recorded for every candidate it evaluates. Once the two scores would choose
different layers, the Top-K protocol's own trajectory diverges and is no longer
observable from these forwards — so the comparison is exact up to and including
the first disagreeing round, and counterfactual after it. That limit is inherent
to sharing forwards and is reported rather than papered over.

Nothing instance-specific lives in core: `K` comes from the D-series policy
module, the vocabulary and geometry from the model, the paths from frozen records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[2]
STATE_EVAL = "artifacts/stage1/state_eval_v1"
TEACHER_BINDING = "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"


def _bootstrap(repo: Path) -> None:
    for extra in ("src", "scripts", "scripts/data", "scripts/pod"):
        path = str(repo / extra)
        if path not in sys.path:
            sys.path.insert(0, path)


class AdoptionError(RuntimeError):
    """A stage could not produce the evidence it exists to produce."""


def quantiles(values: list[float]) -> dict[str, float]:
    """mean/min/P1/P5/P50/P95/P99/max. A mean alone hides the tail.

    Item 10A asks for the distribution because the question is whether the Top-K
    support holds enough mass EVERYWHERE, and an average over 59,763 positions
    says nothing about the thousand worst.
    """
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        #: Nearest-rank, which for a descriptive report needs no interpolation
        #: policy to argue about.
        return ordered[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {"n": n, "mean": sum(ordered) / n, "min": ordered[0],
            "p1": q(0.01), "p5": q(0.05), "p50": q(0.50), "p95": q(0.95),
            "p99": q(0.99), "max": ordered[-1]}


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, by hand because adding scipy for one number is not worth it.

    Ties are averaged, so a scorer that produced many equal values cannot inflate
    the correlation by accident.
    """
    if len(a) != len(b) or len(a) < 3:
        return None

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return None if da == 0 or db == 0 else num / (da * db)


def compare_scalars(full: list[float], topk: list[float]) -> dict[str, Any]:
    """Item 10B's comparison, for any paired list of scores."""
    if not full or len(full) != len(topk):
        return {"n": 0, "_why": "no paired scores"}
    diffs = [t - f for f, t in zip(full, topk)]
    rel = [(t - f) / abs(f) for f, t in zip(full, topk) if f != 0]
    worst = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
    return {
        "n": len(full),
        "absolute_difference": quantiles([abs(d) for d in diffs]),
        "signed_difference": quantiles(diffs),
        "relative_difference": quantiles([abs(r) for r in rel]) if rel else {},
        "spearman": spearman(full, topk),
        "worst": {"index": worst, "full": full[worst], "topk": topk[worst],
                  "absolute": abs(diffs[worst]),
                  "relative": (abs(diffs[worst]) / abs(full[worst])
                               if full[worst] else None)},
        "_sign_of_the_difference": (
            "the K+1 KL is a coarsening of the partition and therefore a LOWER "
            "bound on the full-vocabulary KL, so every signed difference should "
            "be <= 0 up to float error. A positive one would be a defect, not a "
            "finding."),
        "n_positive_signed": sum(1 for d in diffs if d > 1e-6),
    }
