#!/usr/bin/env python3
"""Did each formal session pass the formal-allowance gate at issuance? $0.

    PYTHONPATH=src python scripts/maintenance/consolidation/audit_formal_allowance.py
    PYTHONPATH=src python scripts/maintenance/consolidation/audit_formal_allowance.py --write

The package's 2026-09-28 amendment makes

    remaining formal allowance >= derived session ceiling

one of the three conditions a formal authorization must satisfy. Nothing ever
checked it retrospectively, and `derive_budget` could not have: it attributed
the formal allowance to one hardcoded experiment, so formal C3 sessions reached
the project cumulative and spent nothing from the $55.00 allowance they are
funded by.

With the funding scope read from the package configuration instead, this walks
the funded sessions in chronological order and reconstructs, for each, what the
allowance had left immediately BEFORE it was issued. Ordering is the whole
point: "spend before attempt66" must not include a session that ran two days
after it.

It states no verdict about whether a session's evidence is admissible. That is
a maintainer decision. It states only what the gate would have answered.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "consolidate"))

from maintenance.consolidation import derive_budget as B  # noqa: E402

OUT = "logs/stages/stage-1/phase_c3/analyses/formal_allowance_audit.json"

#: Phases in the order they ran. Within a phase the attempt number is
#: chronological; across phases it is not, and attempt numbers restart.
PHASE_ORDER = ("phase_c1", "phase_c3")


def _phase(session: dict) -> str:
    for key in ("root", "source"):
        value = session.get(key) or ""
        for phase in PHASE_ORDER:
            if phase in value:
                return phase
    return PHASE_ORDER[0]


def audit(root: Path, derived_ceiling_usd: float) -> dict:
    pkg = B.load(B.PACKAGE, root)["execution_package"]
    allowance = float(pkg["formal_allowance_usd"])
    sessions = B.formal_sessions(root, B.funded_experiment_ids(pkg),
                                 pkg.get("package_id"))

    rows = []
    for s in sessions:
        if s["cost_usd"] is None:
            continue
        digits = re.sub(r"\D", "", s["run_id"])
        rows.append({"experiment": _phase(s), "run_id": s["run_id"],
                     "_n": int(digits) if digits else 0,
                     "cost_usd": round(float(s["cost_usd"]), 4)})
    rows.sort(key=lambda r: (PHASE_ORDER.index(r["experiment"]), r["_n"]))

    running = 0.0
    for row in rows:
        row["formal_spent_before_usd"] = round(running, 4)
        row["formal_remaining_before_usd"] = round(allowance - running, 4)
        row["gate_would_have_passed"] = (
            row["formal_remaining_before_usd"] >= derived_ceiling_usd)
        row["shortfall_usd"] = round(
            max(0.0, derived_ceiling_usd - row["formal_remaining_before_usd"]), 4)
        running += row["cost_usd"]
        row.pop("_n")

    total = round(running, 4)
    failed = [r for r in rows if not r["gate_would_have_passed"]]
    return {
        "schema": "aadistill.formal_allowance_audit/v1",
        "_what_this_is": (
            "What the formal-allowance gate would have answered for each "
            "funded session, reconstructed in chronological order from the "
            "package's declared funding scope. Derived on every run; no figure "
            "here is maintained by hand."),
        "_the_condition": "remaining formal allowance >= derived session ceiling",
        "_authority": ("configs/stages/stage-1/phase_c1/authorization.json :: "
                       "execution_package._amendment_2026_09_28"),
        "allowance_usd": allowance,
        "derived_session_ceiling_usd": derived_ceiling_usd,
        "_ceiling_note": (
            "the ceiling C3 derived from the live L40S securePrice at both "
            "issuances. C1's sessions were issued under earlier, smaller "
            "ceilings, so their rows answer the question 'would TODAY's C3 "
            "ceiling have fitted then' and are context rather than a finding."),
        "total_spent_usd": total,
        "remaining_usd": round(allowance - total, 4),
        "allowance_exceeded_by_usd": round(max(0.0, total - allowance), 4),
        "sessions": rows,
        "sessions_that_would_have_been_refused": [r["run_id"] for r in failed],
        "authorizes": "nothing",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--derived-ceiling-usd", type=float, default=22.1452,
                    help="the ceiling a C3 session derives from the live L40S "
                         "securePrice; both C3 formal issuances used this one")
    a = ap.parse_args()
    doc = audit(REPO_ROOT, a.derived_ceiling_usd)

    print(f"formal allowance ${doc['allowance_usd']:.2f}, "
          f"spent ${doc['total_spent_usd']:.4f}, "
          f"remaining ${doc['remaining_usd']:+.4f}")
    for r in doc["sessions"]:
        mark = "ok  " if r["gate_would_have_passed"] else "FAIL"
        print(f"  {mark} {r['experiment']:9s} {r['run_id']:11s} "
              f"cost ${r['cost_usd']:>8.4f}  "
              f"remaining-before ${r['formal_remaining_before_usd']:>9.4f}"
              + (f"  short ${r['shortfall_usd']:.4f}"
                 if not r["gate_would_have_passed"] else ""))
    if a.write:
        (REPO_ROOT / OUT).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
