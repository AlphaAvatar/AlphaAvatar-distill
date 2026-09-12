#!/usr/bin/env python3
"""The four balances, computed from the records that own them.

    PYTHONPATH=src python scripts/consolidate/derive_budget.py
    PYTHONPATH=src python scripts/consolidate/derive_budget.py --json

Four limits bind SEPARATELY and do not transfer into one another:

    formal       the formal-session allowance, spent only by formal launcher
                 sessions
    engineering  the GPU engineering allowance, spent only by engineering
                 validation
    package      formal + engineering, the approved package
    project      the cumulative project cap

Dividing the wrong one by the per-session ceiling is how the repository came to
state that three full-ceiling sessions were fundable when two were. `$51.0465`
of package balance buys three ceilings only if the engineering allowance can pay
for a formal probe, and it cannot: the formal allowance had `$45.0465` left,
which is `$0.3960` short of a third ceiling. The arithmetic was never wrong —
the wrong quantity was divided.

So this computes each balance from its own source and reports how many complete
sessions the FORMAL allowance funds. Nothing here decides policy, raises a cap
or authorizes a launch; it replaces a number that was being maintained by hand
in several documents at once.

Sources, all already owned elsewhere:

* the allowances and the ceiling — `configs/experiments/phase_c1/authorization.json`
  → `execution_package`, the maintainer decision;
* what each formal session actually cost — that session's own
  `closeout/outcome.json`, discovered through the run index so a session cannot
  be missed by being in the layout this script did not think of;
* engineering spend — the campaign record, which is cumulative by construction.

A session whose closeout records no cost is reported, not assumed to be free: a
missing cost is an unknown, and an unknown silently read as zero understates
what has been spent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PACKAGE = "configs/experiments/phase_c1/authorization.json"
RUN_INDEX = "logs/runs/index.json"
CAMPAIGN = "logs/validations/cuda-stage-f/v1/campaign.json"

#: The experiment whose sessions spend the formal allowance. An instance fact,
#: which is why it is here in the application layer and not in the core.
FORMAL_EXPERIMENT = "phase_c1"


def load(rel: str, root: Path) -> dict:
    return json.loads((root / rel).read_text())


def formal_sessions(root: Path) -> list[dict]:
    """Every formal session, with what it cost, discovered through the index.

    Through the index rather than by globbing a directory, because sessions
    exist in two layouts and a glob written for one silently omits the other —
    the same defect that made `by_experiment` count only legacy runs.
    """
    index = load(RUN_INDEX, root)
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in [*index.get("runs", []), *index.get("unrecorded", [])]:
        if entry.get("experiment_id") != FORMAL_EXPERIMENT:
            continue
        key = (entry["experiment_id"], entry["run_id"])
        if key in seen:
            continue
        seen.add(key)
        rec = {"run_id": entry["run_id"], "cost_usd": None,
               "source": None, "root": entry.get("root")}
        for cand in _outcome_paths(root, entry):
            if not cand.is_file():
                continue
            try:
                doc = json.loads(cand.read_text())
            except json.JSONDecodeError:
                continue
            #: Two shapes exist in the record set: `budget.this_attempt` and
            #: `cost.actual_usd`. Both are read rather than one being normalised
            #: away, because rewriting a closed session's evidence to suit a
            #: reader is not a repair.
            cost = (doc.get("budget") or {}).get("this_attempt")
            if cost is None:
                cost = (doc.get("cost") or {}).get("actual_usd")
            if cost is not None:
                rec["cost_usd"] = float(cost)
                rec["source"] = cand.relative_to(root).as_posix()
            break
        out.append(rec)
    return sorted(out, key=lambda r: r["run_id"])


def _outcome_paths(root: Path, entry: dict) -> list[Path]:
    """Where a session's closeout could be, across every entry shape.

    Three shapes exist and all three are read. A modern entry carries `root`. A
    legacy entry carries `components`, which is a MAPPING of role to path in the
    committed index — iterating it as a list of strings yields its keys, finds
    nothing, and reports every session as costless. That is how this first ran:
    it printed `$0.0000 spent` against a package that had booked `$0.3960`.
    """
    out: list[Path] = []
    if entry.get("root"):
        out.append(root / entry["root"] / "closeout/outcome.json")
    comps = entry.get("components")
    if isinstance(comps, dict):
        if comps.get("root"):
            out.append(root / comps["root"] / "closeout/outcome.json")
        for v in comps.values():
            if isinstance(v, str) and v.endswith("outcome.json"):
                out.append(root / v)
    elif isinstance(comps, list):
        for c in comps:
            if isinstance(c, str) and c.endswith("outcome.json"):
                out.append(root / c)
            elif isinstance(c, dict) and str(c.get("path", "")).endswith(
                    "outcome.json"):
                out.append(root / c["path"])
    return out


def derive(root: Path = REPO_ROOT) -> dict:
    pkg = load(PACKAGE, root)["execution_package"]
    formal_allow = float(pkg["formal_allowance_usd"])
    eng_allow = float(pkg["gpu_engineering_allowance_usd"])
    total = float(pkg["package_total_usd"])
    ceiling = float(pkg["per_attempt_hard_ceiling_usd"])
    cap = float(pkg["cumulative_cap_usd"]) if "cumulative_cap_usd" in pkg else None

    sessions = formal_sessions(root)
    #: Only sessions run UNDER this package spend its formal allowance. Earlier
    #: attempts were funded by earlier decisions and are in the project
    #: cumulative, not in this package's book.
    booked = float(pkg.get("booked_before_amendment_usd", 0.0))
    priced = [s for s in sessions if s["cost_usd"] is not None]
    unknown = [s["run_id"] for s in sessions if s["cost_usd"] is None]

    #: The package's own book is authoritative for what it has spent; the
    #: per-session costs are what that book is checked against.
    formal_spent = _package_booked(root, sessions)
    approved = str(pkg.get("approved_utc") or "")
    campaigns = [_attribute(c, pkg.get("package_id"), approved)
                 for c in engineering_campaigns(root)]
    eng_spent = round(sum(c["cost_usd"] for c in campaigns
                          if c["attribution"] == "this_package"), 4)
    #: Unknowns are NAMED, never folded into a total as zero. A balance computed
    #: over unestablished attribution is not a balance a launch may rest on.
    unattributed = [c for c in campaigns if c["attribution"] == "unknown"]

    #: A zero here is a claim, and it was wrong once: the package's own book is
    #: cross-checked against the sessions, and a disagreement is raised rather
    #: than reported. A deriver that silently under-reports spend is worse than
    #: no deriver.
    summed = round(sum(s["cost_usd"] for s in priced), 4)
    if priced and abs(summed - formal_spent) > 0.0001:
        raise SystemExit(
            f"the package book says {formal_spent:.4f} booked but the priced "
            f"sessions sum to {summed:.4f}: {[(s['run_id'], s['cost_usd']) for s in priced]}")
    if formal_spent == 0.0 and booked == 0.0 and not priced:
        raise SystemExit(
            "no session cost could be read at all, so spend would be reported "
            "as zero. Check the run index entry shapes before trusting this.")

    formal_left = round(formal_allow - formal_spent, 4)
    eng_left = round(eng_allow - eng_spent, 4)
    return {
        "schema": "aadistill.derived_budget/v1",
        "_what_this_is": (
            "the four balances, each computed from its own source. Derived, not "
            "maintained: a document that states one of these numbers should "
            "name this deriver rather than carry a hand-updated copy."),
        "package_id": pkg.get("package_id"),
        "per_session_ceiling_usd": ceiling,
        "formal": {"allowance_usd": formal_allow, "spent_usd": formal_spent,
                   "remaining_usd": formal_left},
        "engineering": {"allowance_usd": eng_allow, "spent_usd": eng_spent,
                        "remaining_usd": eng_left},
        "package": {"allowance_usd": total,
                    "spent_usd": round(formal_spent + eng_spent, 4),
                    "remaining_usd": round(total - formal_spent - eng_spent, 4)},
        "project": {"cap_usd": cap} if cap else {},
        "engineering_campaigns": campaigns,
        "pending_reconciliation": [c["campaign"] for c in unattributed],
        #: `None` when anything is unreconciled: an arithmetic summary computed
        #: over an unknown is not a number a launch may be planned against. It
        #: is also NOT a permission when it is a number -- see below.
        "full_ceiling_sessions_fundable": (
            None if unattributed else int(formal_left // ceiling)),
        "_fundable_is_arithmetic_not_permission": (
            "how many complete sessions the remaining FORMAL allowance would "
            "cover. It restores no cap and grants nothing: a session still needs "
            "a grant, a launch-bound readiness record, an authorization and a "
            "bundle. `null` means an engineering cost is unattributed and the "
            "balance is provisional until reconciled."),
        "_fundable_rule": (
            "floor(FORMAL remaining / per-session ceiling). The formal allowance "
            "is the divisor's numerator because the engineering allowance cannot "
            "pay for a formal probe -- the limits do not transfer. Dividing the "
            "PACKAGE balance instead is what produced a count one too high."),
        "sessions": sessions,
        "sessions_without_recorded_cost": unknown,
        "_unknown_rule": (
            "a session whose closeout records no cost is listed, not treated as "
            "free. Reading an unknown as zero understates spend."),
        "authorizes": "nothing",
    }


def _package_booked(root: Path, sessions: list[dict]) -> float:
    """What this package has booked, from the latest session that states it."""
    best = 0.0
    for s in sessions:
        if not s.get("source"):
            continue
        doc = json.loads((root / s["source"]).read_text())
        b = doc.get("budget") or {}
        c = doc.get("cost") or {}
        stated = b.get("package_booked_after")
        if stated is None:
            stated = c.get("package_booked_unchanged_usd")
        if stated is not None:
            best = max(best, float(stated))
    return round(best, 4)


def engineering_campaigns(root: Path) -> list[dict]:
    """Every engineering campaign, with what it cost and which package it is on.

    DISCOVERED, not named. This read one hardcoded path and returned `0.0` for
    anything whose status began with `CLOSED`, so a second campaign was invisible
    and a campaign that spent inside this package stopped counting the moment it
    was closed. Closure is a statement about whether more may be spent, not about
    whether anything was.

    Attribution is explicit where a record states it and DERIVED from dates where
    it does not: a campaign authorized before the package was approved was paid
    for by an earlier decision and is already inside
    `cumulative_spend_at_approval_usd`. Counting it again would charge one
    resource twice.

    A campaign whose cost or package cannot be established is returned with
    `attribution: "unknown"`. It is never silently read as zero.
    """
    out: list[dict] = []
    for p in sorted(root.glob("logs/validations/*/*/campaign.json")):
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            out.append({"campaign": p.relative_to(root).as_posix(),
                        "cost_usd": None, "attribution": "unknown",
                        "why": "the campaign record cannot be read"})
            continue
        cost = doc.get("booked_usd")
        if cost is None:
            cost = doc.get("campaign_cost_usd")
        if cost is None and isinstance(doc.get("subruns"), list):
            parts = [s.get("cost_usd", s.get("subrun_cost_usd"))
                     for s in doc["subruns"]]
            cost = sum(x for x in parts if x is not None) if all(
                x is not None for x in parts) else None
        rec = {"campaign": p.relative_to(root).as_posix(),
               "campaign_id": doc.get("campaign_id"),
               "cost_usd": None if cost is None else round(float(cost), 4),
               "package_id": doc.get("package_id"),
               "granted_utc": None, "attribution": None}
        auth_rel = doc.get("authorization")
        if isinstance(auth_rel, str):
            try:
                rec["granted_utc"] = json.loads(
                    (root / auth_rel).read_text()).get("granted_utc")
            except (json.JSONDecodeError, OSError):
                pass
        out.append(rec)
    return out


def _attribute(camp: dict, package_id: str, approved_utc: str) -> dict:
    """Decide whether a campaign's cost is charged to THIS package."""
    if camp["cost_usd"] is None:
        camp["attribution"] = "unknown"
        camp["why"] = "no cost could be read from the campaign record"
        return camp
    if camp.get("package_id"):
        camp["attribution"] = ("this_package" if camp["package_id"] == package_id
                               else "other_package")
        camp["why"] = f"the record states package_id {camp['package_id']!r}"
        return camp
    if not camp.get("granted_utc"):
        camp["attribution"] = "unknown"
        camp["why"] = ("the record states no package_id and its authorization "
                       "no date, so which decision paid for it is unestablished")
        return camp
    #: Date comparison on the ISO prefix: both are UTC and only the day matters.
    if camp["granted_utc"][:10] < approved_utc[:10]:
        camp["attribution"] = "before_this_package"
        camp["why"] = (f"authorized {camp['granted_utc'][:10]}, before this "
                       f"package was approved {approved_utc[:10]}; already "
                       "inside cumulative_spend_at_approval_usd")
    else:
        camp["attribution"] = "this_package"
        camp["why"] = (f"authorized {camp['granted_utc'][:10]}, on or after "
                       f"{approved_utc[:10]}")
    return camp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = derive()
    if a.json:
        print(json.dumps(d, indent=1))
        return 0
    c = d["per_session_ceiling_usd"]
    for name in ("formal", "engineering", "package"):
        b = d[name]
        print(f"{name:12s} {b['remaining_usd']:>10.4f} remaining "
              f"of {b['allowance_usd']:.4f}  (spent {b['spent_usd']:.4f})")
    print(f"\nper-session ceiling {c:.4f}")
    print(f"full-ceiling sessions fundable from the FORMAL allowance: "
          f"{d['full_ceiling_sessions_fundable']}")
    if d["sessions_without_recorded_cost"]:
        print(f"\nsessions with no recorded cost: "
              f"{d['sessions_without_recorded_cost']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
