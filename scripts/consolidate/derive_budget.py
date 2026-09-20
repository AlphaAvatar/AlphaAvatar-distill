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
RUN_INDEX = "logs/index.json"
CAMPAIGN = "logs/stages/stage-1/phase_c1/validations/cuda-stage-f/v1/campaign.json"

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


def project_sessions(root: Path) -> list[dict]:
    """EVERY recorded run with a stated cost, whatever experiment it belongs to.

    `formal_sessions` above is scoped to the experiment whose package is being
    booked, and that scope is correct for a package balance: an allowance is
    spent only by the sessions it funds. It is wrong for the PROJECT balance,
    and the omission was not theoretical — Phase-C2 attempt 2 spent `$0.0552`
    and the project cumulative could not see it, because this module's only
    session discovery filtered on `FORMAL_EXPERIMENT`.

    Generic, and deliberately not a second experiment list: a new experiment
    contributes to the project total the day it records a closeout cost, with no
    edit here. That is the property the fix has to have — a phase name in a
    conditional would have to be added again for Stage 2, Stage 3 and every
    model family after them.
    """
    index = load(RUN_INDEX, root)
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in [*index.get("runs", []), *index.get("unrecorded", [])]:
        key = (entry.get("experiment_id"), entry.get("run_id"))
        if None in key or key in seen:
            continue
        seen.add(key)
        rec = {"experiment_id": key[0], "run_id": key[1],
               "cost_usd": None, "source": None, "shape": None}
        for cand in _outcome_paths(root, entry):
            if not cand.is_file():
                continue
            doc = json.loads(cand.read_text())
            #: THE CLOSEOUT'S OWN AUTHORITATIVE FIGURE FIRST, then the older
            #: shapes, which remain supported because the closeouts that use
            #: them are frozen evidence and are not being rewritten.
            #:
            #: `money.all_in_usd` is the behavioural closeout's shape and it is
            #: the only one of the three that is ALL-IN: `cost.actual_usd`
            #: carries GPU alone, because the runner reads GPU alone from the
            #: provider. Container disk is billed separately at $0.10/GB/month
            #: and a GPU-only figure silently understates a 120 GB session.
            #: Ordered last all the same: a document that states both must be
            #: read the way its own experiment's ledger reads it, and only the
            #: behavioural closeout states `money` at all.
            for shape, cost in (
                    ("budget.this_attempt",
                     (doc.get("budget") or {}).get("this_attempt")),
                    ("cost.actual_usd",
                     (doc.get("cost") or {}).get("actual_usd")),
                    ("money.all_in_usd",
                     (doc.get("money") or {}).get("all_in_usd")),
                    #: AFFIRMATIVELY $0, which is not the same as unknown. A
                    #: closeout that states no provider resource was ever
                    #: created has said what it cost: nothing was billed
                    #: because nothing existed to bill. Without this, a
                    #: post-anchor attempt that was retired before reaching a
                    #: provider lands in `sessions_without_recorded_cost`
                    #: beside the pre-anchor runs, and a reader can no longer
                    #: tell "excluded because it predates the anchor" from
                    #: "post-anchor and nobody knows". It is the same rule the
                    #: launcher's `prior_attempt_actual` applies, and it is
                    #: guarded the same way: the document must SAY `false`.
                    #: Missing, null or true all fall through to UNKNOWN.
                    ("no_provider_resource",
                     0.0 if doc.get("provider_resource_created") is False
                     else None)):
                if cost is None:
                    continue
                rec["cost_usd"] = float(cost)
                rec["source"] = str(cand.relative_to(root))
                rec["shape"] = shape
                break
            if rec["cost_usd"] is not None:
                break
        out.append(rec)
    return out


def project_balance(root: Path, pkg: dict, cap: float | None) -> dict:
    """The PROJECT cumulative, across every experiment, from one stated anchor.

    `anchor + every recorded closeout cost`. The anchor is the maintainer's
    stated project total at the moment the current package opened, and every
    run that predates it records no cost — so the sum is exactly the spend
    since, with nothing double counted. That is an invariant rather than a
    coincidence, so it is stated here and the runs that carry no cost are NAMED:
    adding a cost to a pre-anchor closeout would break it, and this is where a
    reader would look.

    It is kept apart from the formal/engineering/package balances above on
    purpose. Those are one experiment's execution package; this is the whole
    project's cap. An unused package allowance does not become another
    experiment's headroom, and nothing here transfers it.
    """
    #: From the MAINTAINER's own package decision, which states the project
    #: total at the moment it opened. The renderer used to read this anchor off
    #: the state snapshot, which is derived navigation: a figure computed from a
    #: document that is itself rendered from the figure is self-consistent
    #: whatever it says.
    anchor = pkg.get("cumulative_spend_at_approval_usd")
    if anchor is None or cap is None:
        return {"unavailable": ("the package states no project anchor or cap, "
                                "so a project cumulative cannot be derived")}
    sessions = project_sessions(root)
    priced = [s for s in sessions if s["cost_usd"] is not None]
    unpriced = [f"{s['experiment_id']}/{s['run_id']}" for s in sessions
                if s["cost_usd"] is None]
    since = round(sum(s["cost_usd"] for s in priced), 4)
    by_experiment: dict[str, float] = {}
    for s in priced:
        by_experiment[s["experiment_id"]] = round(
            by_experiment.get(s["experiment_id"], 0.0) + s["cost_usd"], 4)
    cumulative = round(float(anchor) + since, 4)
    return {
        "cap_usd": float(cap),
        "anchor_usd": float(anchor),
        "_anchor_is": ("the project total when the current package opened, "
                       "stated by the maintainer and never back-computed"),
        "spent_since_anchor_usd": since,
        "cumulative_spend_usd": cumulative,
        "remaining_usd": round(float(cap) - cumulative, 4),
        "contributions_by_experiment": dict(sorted(by_experiment.items())),
        "priced_sessions": len(priced),
        "sessions_without_recorded_cost": unpriced,
        "_invariant": ("every run predating the anchor records no closeout "
                       "cost, so `anchor + all recorded costs` double counts "
                       "nothing. Adding a cost to a pre-anchor closeout would "
                       "break that, which is why the costless runs are named."),
        "_not_a_permission": ("remaining project headroom is not authorization "
                              "to spend it, and an unused package allowance "
                              "does not transfer between experiments"),
    }


def derive(root: Path = REPO_ROOT) -> dict:
    config = load(PACKAGE, root)
    pkg = config["execution_package"]
    #: The PROJECT cap lives with the accepted pricing, not in the package: the
    #: package is one experiment's allowance and the cap bounds everything ever
    #: spent. Reading it from the wrong block is what returned "no project cap".
    project_cap = (config.get("accepted_pricing") or {}).get("cumulative_cap_usd")
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
        #: The PROJECT balance, generic across experiments. It was `{cap_usd}`
        #: and nothing else, so the one number every document quotes as
        #: "cumulative project spend" was maintained by hand in several places
        #: and could not include a non-C1 session at all.
        "project": project_balance(root, pkg, project_cap),
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


def _historical(rel: str, root: Path) -> str:
    """Where a path recorded before the log reorganization lives now.

    Delegates to the migration's own table rather than hardcoding a prefix
    rewrite here: the mapping has one owner, and a second copy of it would be a
    second thing to update at the next move. Returns `rel` unchanged when the
    table has nothing for it, so a caller can simply try both.
    """
    try:
        sys.path.insert(0, str(root / "scripts"))
        from architecture.record_run_index import resolve_historical
    except ImportError:
        return rel
    try:
        return resolve_historical(rel, root)
    except Exception:                                          # noqa: BLE001
        return rel


def _campaign_records(root: Path) -> list[Path]:
    """Every engineering campaign ledger in the tree, wherever `logs/` puts it.

    This was `root.glob("logs/validations/*/*/campaign.json")`, a shape that
    stopped existing when the logs were reorganized under
    `logs/stages/stage-<n>/phase_<x>/validations/…`. The glob then matched
    NOTHING while two campaigns holding real spend sat in the tree, and the
    deriver reported `engineering … spent 0.0000` — the under-report its own
    docstring calls worse than no deriver. Found when the C2 full-search CUDA
    validation booked `$0.1453` and the project cumulative could not see a cent
    of it.

    So it searches by NAME rather than by a path shape. A campaign contributes
    the day it exists, at whatever depth the log layout happens to use, and the
    next reorganization does not silently zero the engineering book again.
    """
    found = {p.resolve(): p for p in root.rglob("campaign.json")
             if "validations" in p.parts and ".git" not in p.parts}
    return sorted(found.values())


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
    for p in _campaign_records(root):
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
            #: Through the HISTORICAL-PATH table, because a campaign record
            #: written before the logs were reorganized names its authorization
            #: at the old `logs/validations/…` path. Stage F's does, and with the
            #: discovery glob fixed the deriver could suddenly find that record,
            #: fail to resolve its date, call the attribution "unknown" and
            #: correctly refuse to state how many sessions remain fundable.
            #:
            #: The record is NOT repointed to fix that. Its bytes are bound by
            #: sha256 in the stage-F interpretation amendment, which is
            #: hash-anchored evidence about a closed validation -- editing it to
            #: satisfy an accounting deriver is backwards, and a test caught the
            #: attempt. `resolve_historical` is the table that already exists
            #: for exactly this, and it is what the amendment's own test uses.
            for cand in (root / auth_rel, root / _historical(auth_rel, root)):
                try:
                    rec["granted_utc"] = json.loads(
                        cand.read_text()).get("granted_utc")
                except (json.JSONDecodeError, OSError):
                    continue
                break
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
