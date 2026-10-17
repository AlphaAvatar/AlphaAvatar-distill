#!/usr/bin/env python3
"""D2's shared-confirmation VERDICT, computed OFF-POD at `$0`.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/verdict_d2_confirmation.py \
        --run d2_confirmation_<utc> --store <scr>/store [--write]

Reads the secured evidence of a COMPLETE confirmation session and applies the
frozen rule (`d2_confirmation.verdict`: C0's three-way decision over the
candidate's rows paired with B's imported confirmation rows, the protocol's
stratified prompt-cluster bootstrap at the registered seed, one-sided alpha
0.025, SESOI 0.01, >= 2 of 3 seed deltas positive, the usable-rollout veto):

1. every unit is VERIFIED first, by the same gate the screening ranking uses
   (`rank_d2_screening.verify_unit`, under the shared CONFIRMATION rung): the
   frozen unit's identity, its admission, rows and generations, a re-score by
   the pinned confirmation scorer reproducing every figure, and its training
   records; and its rows are aligned id-for-id with B's at the same seed;
2. the VERDICT: SHARED_GO, NO_GO or INCONCLUSIVE, with the interval;
3. DESCRIPTIVE ONLY, never read by the rule: per seed the candidate's and B's
   correctness and the paired per-sample counts, and q2 as the no-decision
   secondary comparison.

A unit that fails (1) refuses the verdict: deciding on a partial field
decides on who happened to pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stages.phase_d2 import d2_confirmation as CF  # noqa: E402
from stages.phase_d2 import rank_d2_screening as R  # noqa: E402


class VerdictRefused(RuntimeError):
    """The secured evidence does not support a verdict."""


def verdict_run(session: dict[str, Any], store: Path,
                repo_root: Path = REPO_ROOT,
                evidence_root: Path | None = None,
                rescore: Any = None) -> dict[str, Any]:
    from stages.d_series import shared_evaluation as SE

    if session.get("status") != "COMPLETE":
        raise VerdictRefused(f"the session status is {session.get('status')!r}"
                             "; only a complete rung is decided")
    contract = CF.session_contract(repo_root, verify_bytes=False)
    if session.get("contract_hash") != CF.contract_hash(contract):
        raise VerdictRefused("the session ran under a different contract")
    arm = contract["field"]["arms"][0]
    verified_arms = {a.get("state_id") for a in
                     (session.get("arm_verification") or {}).get("arms", [])}
    problems = []
    if verified_arms != {arm["state_id"]}:
        problems.append(f"the session's arm verification covers "
                        f"{sorted(verified_arms)}, not the candidate")
    reference = {r: SE.load_reference_rows(CF.SHARED_RUNG, r,
                                           repo_root=repo_root,
                                           evidence_root=evidence_root)
                 for r in ("B", "q2")}
    by_probe = {p["probe_id"]: p for p in session.get("probes", [])}
    units, rows_by_seed, per_seed = {}, {}, {}
    for probe in contract["probes"]:
        pid, seed = probe["probe_id"], int(probe["seed"])
        if not by_probe.get(pid, {}).get("scored"):
            problems.append(f"{pid}: not scored")
            continue
        unit = R.verify_unit(
            store / "probes" / pid, by_probe[pid],
            {"probe_id": pid, "arm": probe["arm"], "seed": seed,
             "state_id": arm["state_id"],
             "artifact_digest": arm["identities"]["artifact_digest"]},
            repo_root=repo_root, rescore=rescore,
            shared_rung=CF.SHARED_RUNG,
            config_path=(store / "extracted/audit/autoinit_d2_confirmation/"
                                 "configs" / f"{pid}.json"))
        unit["aligned_with_B"] = R.aligned(unit["rows"], reference["B"][seed])
        if not (unit["verified"] and unit["aligned_with_B"]):
            problems.append(f"{pid}: "
                            f"{[k for k, v in unit['checks'].items() if v is not True]}"
                            f", aligned {unit['aligned_with_B']}")
        unit["paired_vs_B"] = R.paired_counts(unit["rows"], reference["B"][seed])
        unit["paired_vs_q2_descriptive"] = R.paired_counts(
            unit["rows"], reference["q2"][seed])
        rows_by_seed[seed] = unit["rows"]
        b_unit = next(u for u in contract["comparator"]["B"]
                      if int(u["seed"]) == seed)
        per_seed[seed] = {
            "candidate_correct_overall": unit["correct_overall"],
            "candidate_usable_rollout_rate": unit["usable_rollout_rate"],
            "B_correct_overall": b_unit["correct_overall"],
            "B_usable_rollout_rate": b_unit["usable_rollout_rate"],
            "B_unit": b_unit["probe_id"],
            "paired_vs_B": unit["paired_vs_B"]}
        units[pid] = {k: v for k, v in unit.items() if k != "rows"}
    if problems:
        raise VerdictRefused("the field is not decidable: " + "; ".join(problems))
    decided = CF.verdict(rows_by_seed, repo_root=repo_root,
                         evidence_root=evidence_root)
    return {
        "schema": "aadistill.phase_d2.confirmation_verdict/v1",
        "run_id": session["run_id"],
        "contract_hash": session["contract_hash"],
        "candidate": contract["candidate"],
        "units_verified": units,
        "outcome": decided["outcome"],
        "verdict": decided,
        "per_seed_descriptive": per_seed,
        "comparator_units": [u["probe_id"] for u in contract["comparator"]["B"]],
        "_claim_boundary": (
            "prompt-distribution uncertainty CONDITIONAL ON the three "
            "preregistered recovery seeds; B's rows were already observed in "
            "D1 (adaptivity limits registered in the shared protocol). "
            "SHARED_GO nominates for a fresh independent confirmation and "
            "never promotes; NO_GO and INCONCLUSIVE are terminal for this "
            "candidate on these rows."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    run_dir = REPO_ROOT / "logs/stages/stage-1/phase_d2/runs" / args.run
    session = json.loads((run_dir / "evidence" / "d2_confirmation.json"
                          ).read_text())
    out = verdict_run(session, Path(args.store))
    if args.write:
        (run_dir / "evidence" / "d2_confirmation_verdict.json").write_text(
            json.dumps(out, indent=1) + "\n")
    print(json.dumps({"outcome": out["outcome"],
                      "decision": out["verdict"]["primary_vs_B"]["decision"]},
                     indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
