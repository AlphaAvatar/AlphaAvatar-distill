#!/usr/bin/env python3
"""Recompute the rung-2 decision from retained evidence, `sa+sb` only. `$0`.

    PYTHONPATH=src python scripts/autoinit/recompute_continuation_rung2.py

Attempt 4 reported `resolved / winner=fe9683e6a9c7`. That decision is
**withdrawn**: the inherited `pooled_over_rungs` pooled every completed rung, and
the continuation imports `85bde4ded2c3/sc` before stage 3, so the comparison was
`sa+sb+sc` (n=570) for one candidate against `sa+sb` (n=380) for the other two.
The `0.012745` margin it produced is not a quantity the frozen rule is defined
over.

This recomputes the same decision from the same retained journals, admitting
**only** `sa` and `sb`, and applies the **real** frozen
`SuccessiveHalvingPlan.select_final_winner` — not a reimplementation of it. The
pooling and row-building come from the driver's own `selection_row`, so what runs
here is the code the pod runs.

It buys nothing and changes no frozen identity. Its output is the corrected
scientific state and the input the re-pricing needs: which `sc` probes, if any,
are still owed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/pod"))
sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))

PROBES = REPO_ROOT / "logs/autoinit_continuation_b_attempt4/probes"
WITHDRAWN = REPO_ROOT / "logs/autoinit_continuation_b_attempt4/phase_a_result.json"
AMENDMENT = REPO_ROOT / "logs/autoinit_phase_b_identity_collapse_amendment.json"


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "continuation_b_driver_recompute",
        REPO_ROOT / "scripts/pod/autoinit_continuation_b_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["continuation_b_driver_recompute"] = mod
    spec.loader.exec_module(mod)
    return mod


def recompute() -> dict:
    from write_preregistration import build_frozen_plan

    drv = load_driver()
    drv.AUDIT = PROBES.parent          # the driver reads <AUDIT>/probes
    plan = build_frozen_plan(REPO_ROOT)

    amendment = json.loads(AMENDMENT.read_text())
    advancing = amendment["rung1_selection"]["advancing"]

    d = drv.ContinuationDriver.__new__(drv.ContinuationDriver)
    d.rung1 = {"advancing": advancing}
    d.rung2 = None                     # -> the RUNG-2 decision: sa+sb only
    rows = drv.ContinuationDriver.pooled_over_rungs(d)
    decision = plan.select_final_winner(rows)

    have_sc = {json.loads(p.read_text())["state_id"]
               for p in sorted(PROBES.glob("*.json"))
               if json.loads(p.read_text())["rung"] == drv.ContinuationDriver.TIE_BREAK_RUNG}
    tied = list(decision.get("tie_break_candidates") or ())
    owed = sorted(s for s in tied if s not in have_sc)

    withdrawn = json.loads(WITHDRAWN.read_text())
    return {
        "schema": "aadistill.autoinit.continuation_b_corrected_rung2/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "_contract": (
            "The rung-2 decision recomputed from the retained Attempt-4 journals "
            "with sa+sb ONLY, using the driver's own pooling and the real frozen "
            "select_final_winner. Supersedes the decision in the Attempt-4 "
            "result, which pooled an imported sc into a rung-2 comparison. Buys "
            "nothing; moves no frozen identity. NOT an authorization."),
        "source_probes": "logs/autoinit_continuation_b_attempt4/probes",
        "admitted_rungs": list(drv.ContinuationDriver.RUNG2_ADMITTED),
        "science_plan_hash": plan.plan_hash,
        "equivalence_interval": withdrawn["equivalence_interval"],
        "pooled_rows": [
            {"state_id": r["state_id"], "is_control": r["is_control"],
             "seeds": r["seeds"], "probe_ids": r["probe_ids"],
             "n": r["n"], "n_scorable": r["n_scorable"], "correct": r["correct"],
             "usable": r["usable"],
             "usable_rollout_rate": r["usable_rollout_rate"],
             "correct_overall": r["correct_overall"],
             "correct_given_usable": r["correct_given_usable"]}
            for r in sorted(rows, key=lambda r: -r["correct_overall"])],
        "decision": decision,
        "decision_status": decision["decision_status"],
        "winner": decision["winner"],
        "tie_break_candidates": tied,
        "sc_already_held": sorted(have_sc),
        "sc_still_owed": owed,
        "withdrawn_decision": {
            "source": "logs/autoinit_continuation_b_attempt4/phase_a_result.json",
            "decision_status": withdrawn["decision_status"],
            "winner": withdrawn["winner"],
            "why_withdrawn": (
                "pooled every completed rung, so 85bde4ded2c3 was compared over "
                "sa+sb+sc (n=570) against sa+sb (n=380) for the others"),
            "asymmetric_rows": {
                r["state_id"]: {"n": r["n"], "probes": len(r["probe_ids"])}
                for r in withdrawn["final_selection"]["ranked"]}},
        "ranking_metric": (
            "correct_overall — the frozen equivalence metric. usable_rollout is a "
            "feasibility/behaviour axis reported alongside it and does NOT rank."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="logs/autoinit_continuation_b_corrected_rung2.json")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / args.out
    result = recompute()
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("pooled sa+sb only:")
    for r in result["pooled_rows"]:
        print(f"  {r['state_id'][:28]:30} {r['correct']:2d}/{r['n_scorable']} = "
              f"{r['correct_overall']:.6f}   usable {r['usable_rollout_rate']:.4f}")
    rows = result["pooled_rows"]
    if len(rows) > 1:
        print(f"\nmargin           {rows[0]['correct_overall'] - rows[1]['correct_overall']:.6f}")
    print(f"interval         {result['equivalence_interval']:.6f}")
    print(f"decision         {result['decision_status']}  winner={result['winner']}")
    print(f"tie candidates   {[s[:12] for s in result['tie_break_candidates']]}")
    print(f"sc already held  {[s[:12] for s in result['sc_already_held']]}")
    print(f"sc still owed    {[s[:12] for s in result['sc_still_owed']]}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
