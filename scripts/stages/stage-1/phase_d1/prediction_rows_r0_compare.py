#!/usr/bin/env python3
"""R0: does pre-vocabulary row selection change any D-series measurement?

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d1/prediction_rows_r0_compare.py \
        --oracle <out>/adoption_all_v1/adoption.json \
        --treatment <out>/adoption_scored_v1/adoption.json \
        --out <out>/r0_comparison.json

Off-pod, $0, recomputable from the two records one session produced. Every
tolerance is READ from the plan registered before the session
(``--plan``, default :data:`PLAN`); nothing here is a threshold of its own, so
a failing comparison cannot be rescued by editing this file without the edit
showing up as a different plan hash in the verdict.

Four questions, in order:

1. **identity** -- the oracle ran ``all_v1`` and binds the SAME protocol ids
   the historical all-rows run bound (the oracle is the historical path, not
   a new one); the treatment ran ``scored_v1`` and binds DIFFERENT ids for both
   supports (a non-default arithmetic path is visible in its identity);
2. **DEPTH decisions** -- per profile, the treatment's removal order equals the
   oracle's, and the oracle's equals the historical run's;
3. **DEPTH values** -- every candidate of every round, relative difference
   within the plan's bound;
4. **state evaluation** -- per support, the ranked objectives within the
   plan's absolute drift budget; every other value reported, not gated.

Wall time and peak memory are reported and never gated: R0 asks whether the
optimization is SAFE, and its speed is what the report is for.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[4]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

SCHEMA = "aadistill.d_series.prediction_rows_r0_comparison/v1"
PLAN = ("logs/stages/stage-1/families/d_series/validations/prediction-rows-r0/"
        "v1/plan.json")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _rel(a: float, b: float) -> float:
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


def ranked_objectives() -> tuple[str, ...]:
    """The metrics the D-series beam ranks on, from the policy that ranks."""
    from aadistill.initialization.planning.ranking import PARETO_V1

    return tuple(o.key for o in PARETO_V1.objectives)


def compare(oracle: Mapping[str, Any], treatment: Mapping[str, Any],
            history: Mapping[str, Any], plan: Mapping[str, Any]
            ) -> dict[str, Any]:
    tol = plan["tolerances"]
    failures: list[str] = []
    out: dict[str, Any] = {}

    # 1. identity
    ident = {
        "oracle_mode": oracle.get("prediction_rows"),
        "treatment_mode": treatment.get("prediction_rows"),
        "oracle_status": oracle.get("status"),
        "treatment_status": treatment.get("status"),
    }
    if ident["oracle_mode"] != "all_v1":
        failures.append(f"oracle ran {ident['oracle_mode']!r}, not all_v1")
    if ident["treatment_mode"] != "scored_v1":
        failures.append(f"treatment ran {ident['treatment_mode']!r}, not scored_v1")
    for side, rec in (("oracle", oracle), ("treatment", treatment)):
        if rec.get("status") != "COMPLETE":
            failures.append(f"{side} status is {rec.get('status')!r}")
    for support in ("full_vocab", "reference_topk_tail"):
        key = f"bound_{support}"
        o = (oracle.get(key) or {}).get("measurement_protocol_id")
        t = (treatment.get(key) or {}).get("measurement_protocol_id")
        h = (history.get(key) or {}).get("measurement_protocol_id")
        ident[support] = {"oracle": o, "treatment": t, "history": h,
                          "oracle_equals_history": o == h,
                          "treatment_differs": t is not None and t != o}
        if o != h:
            failures.append(
                f"{support}: the all_v1 oracle binds {o}, the historical run "
                f"bound {h}; the oracle is not the historical path")
        if t is None or t == o:
            failures.append(
                f"{support}: scored_v1 binds the oracle's protocol id; the "
                "row-selected path is not visible in its identity")
    out["identity"] = ident

    # 2 + 3. DEPTH
    depth: dict[str, Any] = {}
    po = (oracle.get("P_production_timing") or {}).get("per_profile") or {}
    pt = (treatment.get("P_production_timing") or {}).get("per_profile") or {}
    ph = (history.get("P_production_timing") or {}).get("per_profile") or {}
    profiles = sorted(set(po) | set(pt))
    if not profiles:
        failures.append("neither record carries a stage P result")
    for prof in profiles:
        o, t, h = po.get(prof), pt.get(prof), ph.get(prof)
        row: dict[str, Any] = {}
        if o is None or t is None:
            failures.append(f"{prof}: missing from "
                            f"{'oracle' if o is None else 'treatment'}")
            depth[prof] = {"complete": False}
            continue
        row["removed"] = {"oracle": o["removed"], "treatment": t["removed"],
                          "history": h["removed"] if h else None}
        row["order_equal"] = o["removed"] == t["removed"]
        row["oracle_equals_history"] = bool(h) and o["removed"] == h["removed"]
        if not row["order_equal"]:
            failures.append(f"{prof}: removal order {t['removed']} differs from "
                            f"the oracle's {o['removed']}")
        if not row["oracle_equals_history"]:
            failures.append(f"{prof}: the oracle's removal order does not "
                            "reproduce the historical run's")
        to, tt = o.get("round_tables"), t.get("round_tables")
        if not to or not tt:
            failures.append(f"{prof}: a record carries no round_tables, so the "
                            "candidates cannot be compared")
        else:
            worst_rel, worst_abs, n = 0.0, 0.0, 0
            mismatched = []
            for r, (ro, rt) in enumerate(zip(to, tt)):
                so = {json.dumps(c): s for c, s in ro}
                st = {json.dumps(c): s for c, s in rt}
                if set(so) != set(st):
                    mismatched.append(r)
                    continue
                for c, s in so.items():
                    n += 1
                    worst_rel = max(worst_rel, _rel(float(s), float(st[c])))
                    worst_abs = max(worst_abs, abs(float(s) - float(st[c])))
            if len(to) != len(tt) or mismatched:
                failures.append(f"{prof}: candidate sets differ at rounds "
                                f"{mismatched or 'count'}")
            row["candidates_compared"] = n
            row["max_relative_difference"] = worst_rel
            row["max_absolute_difference"] = worst_abs
            if worst_rel > float(tol["depth_candidate_relative"]):
                failures.append(
                    f"{prof}: a candidate differs by {worst_rel:.3e} relative, "
                    f"over the registered {tol['depth_candidate_relative']}")
        row["timing"] = {
            "oracle_seconds": o.get("operator_invocation_seconds"),
            "treatment_seconds": t.get("operator_invocation_seconds"),
            "ratio": (round(t["operator_invocation_seconds"]
                            / o["operator_invocation_seconds"], 4)
                      if o.get("operator_invocation_seconds") else None),
            "oracle_peak_gib": round(o.get("peak_memory_bytes", 0) / 2**30, 3),
            "treatment_peak_gib": round(t.get("peak_memory_bytes", 0) / 2**30, 3),
        }
        depth[prof] = row
    out["depth"] = depth

    # 4. state evaluation
    state: dict[str, Any] = {}
    ranked = ranked_objectives()
    do = ((oracle.get("D_state_eval") or {}).get("arms")) or {}
    dt = ((treatment.get("D_state_eval") or {}).get("arms")) or {}
    if "D" in (plan.get("stages") or ()) and not (do and dt):
        failures.append("a record carries no stage D state evaluation")
    for arm in sorted(set(do) & set(dt)):
        vo, vt = do[arm]["values"], dt[arm]["values"]
        drift = {k: abs(float(vt[k]) - float(vo[k])) for k in sorted(set(vo) & set(vt))
                 if isinstance(vo[k], (int, float)) and isinstance(vt[k], (int, float))}
        gated = {k: drift[k] for k in ranked if k in drift}
        missing = [k for k in ranked if k not in drift]
        if missing:
            failures.append(f"{arm}: ranked objectives {missing} not recorded")
        worst = max(gated.values()) if gated else None
        if worst is not None and worst > float(tol["state_eval_ranked_absolute"]):
            failures.append(
                f"{arm}: a ranked objective drifts {worst:.3e}, over the "
                f"registered {tol['state_eval_ranked_absolute']}")
        if do[arm].get("measurement_protocol_id") == dt[arm].get(
                "measurement_protocol_id"):
            failures.append(f"{arm}: both modes report one protocol id")
        state[arm] = {
            "ranked_absolute_drift": gated,
            "max_ranked_absolute_drift": worst,
            "max_absolute_drift_all_values": max(drift.values()) if drift else None,
            "worst_domain_agrees": do[arm].get("worst_domain") == dt[arm].get(
                "worst_domain"),
            "seconds": {"oracle": do[arm].get("seconds"),
                        "treatment": dt[arm].get("seconds")},
            "peak_gib": {"oracle": round(do[arm].get("peak_memory_bytes", 0) / 2**30, 3),
                         "treatment": round(dt[arm].get("peak_memory_bytes", 0) / 2**30, 3)},
        }
    out["state_eval"] = state

    out["verdict"] = "PASS" if not failures else "FAIL"
    out["failures"] = failures
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--treatment", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=REPO / PLAN)
    ap.add_argument("--history", type=Path, default=None,
                    help="the historical all-rows record (default: the plan's)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    plan = _load(args.plan)
    history_path = args.history or (REPO / plan["historical_oracle"]["record"])
    result = compare(_load(args.oracle), _load(args.treatment),
                     _load(history_path), plan)
    record = {
        "schema": SCHEMA,
        "plan": str(args.plan.relative_to(REPO) if args.plan.is_relative_to(REPO)
                    else args.plan),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "inputs_sha256": {
            "oracle": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
            "treatment": hashlib.sha256(args.treatment.read_bytes()).hexdigest(),
            "history": hashlib.sha256(Path(history_path).read_bytes()).hexdigest(),
        },
        **result,
        "authorizes": "nothing; a PASS makes scored_v1 ELIGIBLE for a later "
                      "preregistration to name",
    }
    body = json.dumps(record, indent=1, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(body)
    print(f"R0 {record['verdict']}" + "".join(f"\n  - {f}" for f in record["failures"]))
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
