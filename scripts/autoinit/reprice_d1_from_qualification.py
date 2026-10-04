#!/usr/bin/env python3
"""Reprice D1's search from the GPU qualification's measurements.

    python scripts/autoinit/reprice_d1_from_qualification.py RECORD.json

D1's chain figures rest on a per-expansion cost table derived from two
historical searches (`phase_b_attempt5`, `phase_c2_attempt4`), per-cell maximum.
Both were measured with an UNBATCHED state evaluation. The design says in as
many words that the batched implementation's direction relative to that table is
UNKNOWN — A3 measured batching 8-10% SLOWER on the ATTENTION scorer and packing
faster on causal-KL, and the net across four operators had never been measured.

This derives that direction from the qualification, and it is careful about one
thing that makes the comparison easy to get wrong:

    a fixed-path STEP is not a search EXPANSION.

An expansion is `materialize + identify + canonical_reload + validation +
state_evaluation` (`EXPANSION_PHASES`). The fixed path materializes and
identifies; it does not reload canonically, validate, or state-evaluate per
step. So a step's seconds are a LOWER BOUND on the matching expansion cell, and
reporting the raw ratio of one to the other would overstate the speedup — the
batched path would look faster partly because it is doing less work.

So the comparison is built the other way round: the measured components are
added up, the unmeasured phases are named as a residual rather than assumed to
be zero, and the resulting ratio is reported as a RANGE whose optimistic end
assumes the residual scales with the measured part and whose conservative end
assumes it does not scale at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.phase_c2.full_search_space import (  # noqa: E402
    derive_cost_table,
)

#: Phases an expansion has and a fixed-path step does not. Named, not ignored:
#: treating them as zero is what would turn "we measured less work" into "the
#: implementation got faster".
UNMEASURED_PHASES = ("identify_seconds", "canonical_reload_seconds",
                     "validation_seconds")


def _steps(record: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    return list((record.get(arm) or {}).get("steps") or [])


def _state_eval_minutes(record: dict[str, Any]) -> dict[str, float]:
    """Measured state-evaluation minutes, by arm label."""
    out: dict[str, float] = {}
    for m in record.get("D_state_eval") or []:
        label, seconds = m.get("label"), m.get("seconds")
        if label and isinstance(seconds, (int, float)):
            out[label] = seconds / 60.0
    return out


def compare(record: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
    """Per-cell measured-vs-frozen comparison, for both arms."""
    minutes = table["minutes"]
    rows: list[dict[str, Any]] = []

    for arm, key in (("incumbent", "A_incumbent"),
                     ("target_aware", "B_target_aware")):
        steps = _steps(record, key)
        for i, step in enumerate(steps):
            impl = step.get("impl_id")
            cell = minutes.get(impl) or {}
            #: step 0 starts from the teacher, which is the ROOT; every later
            #: step starts from the previous step's child, which is DEEPER.
            scope = "root" if i == 0 else "deeper"
            frozen_max = cell.get(f"{scope}_max")
            frozen_mean = cell.get(f"{scope}_mean")
            measured = (step.get("seconds") or 0.0) / 60.0
            rows.append({
                "arm": arm, "index": i, "impl_id": impl, "scope": scope,
                "measured_materialize_minutes": round(measured, 4),
                "frozen_expansion_minutes_max": frozen_max,
                "frozen_expansion_minutes_mean": frozen_mean,
                "_measured_is_a_lower_bound_because": (
                    "the frozen cell is a whole expansion and this is a "
                    f"materialize+identify step: it omits {UNMEASURED_PHASES}"),
                "ratio_to_frozen_max": (round(measured / frozen_max, 4)
                                        if frozen_max else None),
                "ratio_to_frozen_mean": (round(measured / frozen_mean, 4)
                                         if frozen_mean else None),
            })
    return {"_rule": table["_rule"], "sources": table["sources"], "cells": rows}


def direction(rows: list[dict[str, Any]], state_eval: dict[str, float]
              ) -> dict[str, Any]:
    """The one question the design left open: faster, slower, or unknown.

    Conservative end: the measured materialization plus the measured state
    evaluation, against the frozen per-cell MAXIMUM, with the unmeasured phases
    assumed not to improve at all. Optimistic end: against the frozen MEAN.
    """
    out: dict[str, Any] = {"per_arm": {}}
    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        se = state_eval.get(arm)
        measured_total = sum(r["measured_materialize_minutes"] for r in arm_rows)
        frozen_max = sum(r["frozen_expansion_minutes_max"] or 0.0
                         for r in arm_rows)
        frozen_mean = sum(r["frozen_expansion_minutes_mean"] or 0.0
                          for r in arm_rows)
        #: One state evaluation per expansion, so four steps carry four of them.
        with_se = (measured_total + se * len(arm_rows)) if se is not None else None
        entry = {
            "measured_materialize_minutes_total": round(measured_total, 4),
            "measured_state_eval_minutes_each": (round(se, 4)
                                                 if se is not None else None),
            "measured_comparable_total_minutes": (round(with_se, 4)
                                                  if with_se else None),
            "frozen_total_minutes_max": round(frozen_max, 4),
            "frozen_total_minutes_mean": round(frozen_mean, 4),
        }
        if with_se:
            entry["ratio_vs_frozen_max"] = round(with_se / frozen_max, 4)
            entry["ratio_vs_frozen_mean"] = round(with_se / frozen_mean, 4)
            entry["verdict"] = (
                "FASTER than the frozen table at both ends"
                if with_se < frozen_mean else
                "faster than the per-cell maximum but not than the mean"
                if with_se < frozen_max else
                "SLOWER than the frozen table, which bounds nothing")
        else:
            entry["verdict"] = ("state evaluation was not measured for this "
                                "arm, so no comparable total exists and the "
                                "direction stays UNKNOWN")
        entry["_still_unmeasured"] = list(UNMEASURED_PHASES)
        out["per_arm"][arm] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    record = json.loads(Path(args.record).read_text())
    table = derive_cost_table(REPO)
    comparison = compare(record, table)
    verdicts = direction(comparison["cells"], _state_eval_minutes(record))

    doc = {
        "schema": "aadistill.d1.reprice_from_qualification/v1",
        "_what": __doc__.strip().splitlines()[0],
        "frozen_table": {"_rule": comparison["_rule"],
                         "sources": comparison["sources"]},
        "cells": comparison["cells"],
        "direction": verdicts,
        "_authorizes": ("nothing. A measured direction makes D1's chain figures "
                        "better founded; it does not fund them, and the funding "
                        "and per-session blockers are untouched by it."),
    }
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    for arm, entry in verdicts["per_arm"].items():
        print(f"{arm}: {entry['verdict']}")
        print(f"   measured {entry['measured_comparable_total_minutes']} min "
              f"vs frozen max {entry['frozen_total_minutes_max']} / "
              f"mean {entry['frozen_total_minutes_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
