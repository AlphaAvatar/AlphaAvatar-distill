"""THE D-SERIES IS CLOSED. Its closeout record, and the gate every issuer runs.

Maintainer final decision, 2026-10-17 UTC: close D1, D2 and D3 and adopt D1's
initialization methodology together with ``prediction_rows=scored_v1`` as the
official forward baseline (`configs/shared/baselines/`, resolved through
`shared.forward_baseline`).

    PYTHONPATH=src:scripts python scripts/stages/stage-1/families/d_series/programme_closeout.py          # print
    PYTHONPATH=src:scripts python scripts/stages/stage-1/families/d_series/programme_closeout.py --write  # record

What this module owns, and what it only reads:

* the CLOSEOUT RECORD (`CLOSEOUT`) -- assembled from records that already own
  each fact: every phase's terminal outcome is READ from that run's closeout,
  every dollar is computed by the budget deriver, and each closed authority is
  bound by the sha256 of its file as it stood at closure. Nothing scientific is
  restated, re-decided or rewritten: D1 stays NO-GO, D2 stays NO_GO, D3 stays
  STRUCTURAL_CHANGE with INCONCLUSIVE probes. The maintainer's selection is a
  new, additive decision recorded beside them;
* `refuse_closed_authority` -- the one check the D-series issuers run before
  issuing anything. A closed grant can no longer produce a one-use
  authorization, so no further D-series launch can be authorized under it. The
  grant files themselves are NOT edited: existing authorizations bind their
  hashes, and a closure that rewrote them would break the verification of every
  run it is closing.

The `phase_d1/d2/d3` entries in the C1 package's `funds_formal_sessions_of`
stay as they are: they attribute money already spent, and removing them would
orphan those sessions in the ledger. Closing the AUTHORITIES is what stops new
spending; the funding scope is history.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]

CLOSEOUT = "logs/stages/stage-1/families/d_series/closeout/programme_closeout.json"
SCHEMA = "aadistill.d_series.programme_closeout/v1"
DECIDED_UTC = "2026-10-17"
STATUS = "CLOSED -- MAINTAINER SELECTED D1 + scored_v1 AS FORWARD BASELINE"
DECISION_RECORD = ("logs/budget/decisions.md 2026-10-17 (D-series closed; D1 + "
                   "scored_v1 adopted as the forward baseline)")
BASELINE_SELECTION = "configs/shared/baselines/selection.json"

#: Each phase's TERMINAL runs, by rung. The outcome of each is read from its
#: own closeout by `_outcome`; this table only says where to look.
TERMINAL_RUNS: dict[str, dict[str, str]] = {
    "phase_d1": {
        "search": "logs/stages/stage-1/phase_d1/runs/d1_search_20261006_210210",
        "screening": "logs/stages/stage-1/phase_d1/runs/d1_behavioural_20261011_135910",
        "confirmation": "logs/stages/stage-1/phase_d1/runs/d1_behavioural_20261012_064341",
    },
    "phase_d2": {
        "search": "logs/stages/stage-1/phase_d2/runs/d2_search_20261015_051722",
        "screening": "logs/stages/stage-1/phase_d2/runs/d2_screening_20261015_203346",
        "confirmation": "logs/stages/stage-1/phase_d2/runs/d2_confirmation_20261016_105802",
    },
    "phase_d3": {
        "structural": "logs/stages/stage-1/phase_d3/runs/d3_structural_20261017_083554",
        "probes": "logs/stages/stage-1/phase_d3/runs/d3_probes_20261017_110022",
    },
}

#: Every D-series spending authority, and the issuer that reads it. Closed
#: together; `refuse_closed_authority` refuses each by path AND by content, so
#: a copy of a closed grant elsewhere is refused too.
AUTHORITIES: dict[str, dict[str, str]] = {
    "logs/budget/approvals/autoinit_d1_grant.json": {
        "phase": "phase_d1", "rung": "search",
        "issuer": "scripts/stages/stage-1/phase_d1/issue_d1_authorization.py"},
    "logs/budget/approvals/autoinit_d1_replay_authorization.json": {
        "phase": "phase_d1", "rung": "finalist rematerialization (engineering)",
        "issuer": "scripts/stages/stage-1/phase_d1/issue_d1_replay_authorization.py"},
    "logs/budget/approvals/autoinit_d1_behavioural_phase_grant.json": {
        "phase": "phase_d1", "rung": "behavioural (phase allowance)",
        "issuer": "scripts/stages/stage-1/phase_d1/issue_d1_behavioural_authorization.py"},
    "logs/budget/approvals/autoinit_d1_behavioural_screening_grant.json": {
        "phase": "phase_d1", "rung": "screening",
        "issuer": "scripts/stages/stage-1/phase_d1/issue_d1_behavioural_authorization.py"},
    "logs/budget/approvals/autoinit_d1_behavioural_screening_grant_amendment.json": {
        "phase": "phase_d1", "rung": "screening",
        "issuer": "scripts/stages/stage-1/phase_d1/issue_d1_behavioural_authorization.py"},
    "logs/budget/approvals/autoinit_d1_behavioural_confirmation_grant.json": {
        "phase": "phase_d1", "rung": "confirmation",
        "issuer": "scripts/stages/stage-1/phase_d1/issue_d1_behavioural_authorization.py"},
    "logs/budget/approvals/autoinit_d2_grant.json": {
        "phase": "phase_d2", "rung": "search",
        "issuer": "scripts/stages/stage-1/phase_d2/issue_d2_authorization.py"},
    "logs/budget/approvals/autoinit_d2_screening_grant.json": {
        "phase": "phase_d2", "rung": "screening",
        "issuer": "scripts/stages/stage-1/phase_d2/issue_d2_screening_authorization.py"},
    "logs/budget/approvals/autoinit_d2_confirmation_grant.json": {
        "phase": "phase_d2", "rung": "confirmation",
        "issuer": "scripts/stages/stage-1/phase_d2/issue_d2_screening_authorization.py"},
    "logs/budget/approvals/autoinit_d3_structural_grant.json": {
        "phase": "phase_d3", "rung": "structural",
        "issuer": "scripts/stages/stage-1/phase_d3/issue_d3_authorization.py"},
    "logs/budget/approvals/autoinit_d3_probes_grant.json": {
        "phase": "phase_d3", "rung": "probes",
        "issuer": "scripts/stages/stage-1/phase_d3/issue_d3_authorization.py"},
}

#: The two grants the maintainer closed with money still unspent. Their
#: envelope field is the grant's own; the spend is the deriver's.
UNSPENT_CLOSED = ("logs/budget/approvals/autoinit_d3_structural_grant.json",
                  "logs/budget/approvals/autoinit_d3_probes_grant.json")


class AuthorityClosed(SystemExit):
    """A closed D-series authority was offered to an issuer."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(rel: str, repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / rel).read_text())


def closed_authorities(repo_root: str | Path = REPO_ROOT) -> dict[str, str]:
    """``{path: sha256}`` of every authority the RECORDED closeout closed.

    Read from the committed record, not from `AUTHORITIES`, so what an issuer
    refuses is what was actually closed. No record means nothing is closed
    (a fixture repository); the real tree's record is pinned by a test.
    """
    path = Path(repo_root) / CLOSEOUT
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    return {a["path"]: a["sha256_at_closure"]
            for a in doc.get("closed_authorities", [])}


def refuse_closed_authority(grant_path: str | Path, *,
                            repo_root: str | Path = REPO_ROOT) -> None:
    """Raise `AuthorityClosed` if `grant_path` is a closed D-series authority.

    Matched by repository path and by content hash, so neither a relative path
    nor a copy of the bytes under another name gets past it.
    """
    closed = closed_authorities(repo_root)
    if not closed:
        return
    p = Path(grant_path)
    root = Path(repo_root).resolve()
    try:
        rel = str(p.resolve().relative_to(root))
    except ValueError:
        rel = str(p)
    digest = _sha256(p) if p.is_file() else None
    if rel in closed or (digest and digest in closed.values()):
        raise AuthorityClosed(
            f"refusing to issue: {rel} is a CLOSED D-series authority. The "
            f"D-series was closed by maintainer decision {DECIDED_UTC} "
            f"({CLOSEOUT}); no further D-series launch may be authorized under "
            "it. New work needs its own grant.")


def _deriver(repo_root: Path):
    spec = importlib.util.spec_from_file_location(
        "derive_budget_for_d_series_closeout",
        repo_root / "scripts/maintenance/consolidation/derive_budget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _outcome(run_rel: str, repo_root: Path) -> dict[str, Any]:
    """A terminal run's recorded outcome, READ from its closeout."""
    rel = f"{run_rel}/closeout/outcome.json"
    doc = _load(rel, repo_root)
    out: dict[str, Any] = {
        "run_id": Path(run_rel).name, "closeout": rel,
        "closeout_sha256": _sha256(repo_root / rel),
        "classification": doc.get("classification")}
    verdict = doc.get("verdict") or doc.get("decision")
    if isinstance(verdict, dict) and verdict.get("verdict"):
        out["verdict"] = verdict["verdict"]
        out["delta"] = verdict.get("delta")
    if doc.get("structural_outcome"):
        out["structural_outcome"] = doc["structural_outcome"]
    result = doc.get("exploratory_result")
    if isinstance(result, dict):
        out["labels"] = {"vs_q2": result.get("primary_label_vs_q2"),
                         "vs_B": result.get("anchor_label_vs_B")}
        out["deltas"] = {"vs_q2": (result.get("vs_q2") or {}).get("delta"),
                         "vs_B": (result.get("vs_B") or {}).get("delta")}
    return out


_RUN_SUFFIX = re.compile(r"_(\d{8}_\d{6}|\d{3})$")


def _money(repo_root: Path) -> dict[str, Any]:
    """Every D-series dollar, computed by the deriver from the run records."""
    derived = _deriver(repo_root).derive(repo_root)
    by_rung: dict[str, dict[str, Any]] = {}
    for s in derived["sessions"]:
        src = s.get("source") or s.get("root") or ""
        phase = next((p for p in TERMINAL_RUNS if f"/{p}/" in src), None)
        if phase is None:
            continue
        key = f"{phase}:{_RUN_SUFFIX.sub('', s['run_id'])}"
        row = by_rung.setdefault(key, {"usd": 0.0, "sessions": 0})
        row["usd"] = round(row["usd"] + float(s.get("cost_usd") or 0), 4)
        row["sessions"] += 1
    campaigns = [
        {"campaign": c["campaign"], "usd": c["cost_usd"]}
        for c in derived["engineering_campaigns"]
        if "/phase_d" in c["campaign"] or "/families/d_series/" in c["campaign"]]
    runs = round(sum(r["usd"] for r in by_rung.values()), 4)
    engineering = round(sum(c["usd"] for c in campaigns), 4)
    return {
        "run_closeouts_by_rung_usd": dict(sorted(by_rung.items())),
        "_d1_replay_books_zero": (
            "the two d1_replay runs book $0 in their own closeouts; their "
            "cost is the finalist-rematerialization campaign's, listed below"),
        "run_closeouts_usd": runs,
        "engineering_campaigns": campaigns,
        "engineering_campaigns_usd": engineering,
        "d_series_total_usd": round(runs + engineering, 4),
        "limits_after": {
            "project_remaining_usd": derived["project"]["remaining_usd"],
            "formal_remaining_usd": derived["formal"]["remaining_usd"],
            "package_remaining_usd": derived["package"]["remaining_usd"],
            "engineering_remaining_usd": derived["engineering"]["remaining_usd"],
            "cumulative_spend_usd": derived["project"]["cumulative_spend_usd"],
            "pending_reconciliation": derived["pending_reconciliation"]},
        "_derived_by": "scripts/maintenance/consolidation/derive_budget.py",
    }


def _authorities(repo_root: Path, money: dict[str, Any]) -> list[dict[str, Any]]:
    spent = money["run_closeouts_by_rung_usd"]
    out = []
    for rel, meta in AUTHORITIES.items():
        p = repo_root / rel
        row: dict[str, Any] = {"path": rel, "sha256_at_closure": _sha256(p),
                               "status": "CLOSED", **meta}
        if rel in UNSPENT_CLOSED:
            grant = json.loads(p.read_text())
            envelope = float(grant["cumulative_envelope_usd"])
            used = spent[f"{meta['phase']}:d3_{meta['rung']}"]["usd"]
            row.update({
                "cumulative_envelope_usd": envelope,
                "spent_usd": used,
                "unspent_usd": round(envelope - used, 4),
                "unspent_disposition": (
                    "CLOSED UNSPENT: not spent, not transferred, not "
                    "repurposed. It funds nothing from this decision on.")})
        out.append(row)
    return out


def build(repo_root: str | Path = REPO_ROOT, *,
          provider: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(repo_root)
    money = _money(root)
    phases = {phase: {rung: _outcome(run, root) for rung, run in runs.items()}
              for phase, runs in TERMINAL_RUNS.items()}
    selection = root / BASELINE_SELECTION
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "decided_utc": DECIDED_UTC,
        "decision_record": DECISION_RECORD,
        "_contract": (
            "The terminal record of the D-series programme. Outcomes are READ "
            "from each terminal run's closeout and bound by its sha256; money is "
            "computed by the budget deriver; each closed authority is bound by "
            "the sha256 of its file at closure. AUTHORIZES NOTHING."),
        "scientific_outcomes": {
            "phase_d1": "COMPLETE / historical NO-GO (q2 advanced from "
                        "screening and did not confirm against B)",
            "phase_d2": "COMPLETE / NO_GO (c4 advanced from screening and did "
                        "not confirm against B)",
            "phase_d3": "structural STRUCTURAL_CHANGE; behavioural comparison "
                        "INCONCLUSIVE vs q2 and vs B (exploratory)",
            "student_confidence_d3": "CANCELLED / NEVER EXECUTED",
            "_unchanged": (
                "every verdict above is the run's own and is not rewritten, "
                "re-analysed or upgraded by the selection below. D1's NO-GO "
                "stays NO-GO."),
        },
        "terminal_runs": phases,
        "maintainer_selection": {
            "decision": ("ADOPT D1's initialization methodology together with "
                         "prediction_rows=scored_v1 as the official forward "
                         "baseline. A selection, not a verdict."),
            "selection": BASELINE_SELECTION,
            "selection_sha256": _sha256(selection) if selection.is_file() else None,
            "resolved_by": "shared.forward_baseline.selected()",
            "historical_comparator": (
                "B (C1's promoted treatment) stays the archived comparator "
                "every historical D-series record names; "
                "stages.d_series.incumbent.standing_incumbent() is unchanged."),
        },
        "closed_authorities": _authorities(root, money),
        "funding_scope": (
            "phase_d1, phase_d2 and phase_d3 stay in the C1 package's "
            "funds_formal_sessions_of: they attribute money already spent. "
            "No cap or allowance changed."),
        "money": money,
        "provider": provider,
        "pending": [],
        "not_pending": [
            "no further D-series stage, probe, rerun, seed or confirmation",
            "no rematerialization: the forward profile has no checkpoint of its "
            "own yet, and none is owed for this closure",
            "no Stage-0 experiment: designed and authorized separately"],
        "authorizes": "nothing",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--provider-json", type=Path, default=None,
                    help="the provider inventory taken at closure, recorded as is")
    args = ap.parse_args(argv)
    provider = (json.loads(args.provider_json.read_text())
                if args.provider_json else None)
    if args.write and provider is None:
        existing = REPO_ROOT / CLOSEOUT
        provider = (json.loads(existing.read_text()).get("provider")
                    if existing.is_file() else None)
    doc = build(REPO_ROOT, provider=provider)
    text = json.dumps(doc, indent=1) + "\n"
    if args.write:
        out = REPO_ROOT / CLOSEOUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {CLOSEOUT}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    raise SystemExit(main())
