#!/usr/bin/env python3
"""Book one engineering session into its campaign: small evidence, cost, hash.

    .venv/bin/python scripts/shared/pod/book_engineering_subrun.py \
        --run-dir artifacts/.../topk_adoption/<run> \
        --validation-dir logs/stages/.../validations/<name>/v1 \
        --subrun-id <id> --outcome "<one line>" \
        [--evidence-session out/adoption_all_v1/adoption.json] [--close "<status>"]

What a launcher left in the run directory becomes, in the validation record:

* every small text file (JSON, JSONL, logs; nothing under ``work/``, nothing
  over 5 MB) copied to ``runs/<run>/`` with its sha256;
* one subrun row in ``campaign.json``: the ALL-IN cost from the launcher's
  ``cost.json`` (written by its teardown from the live rate and the wall
  clock, container disk included), the provider-confirmed pod id, whether the
  provider confirmed teardown, and -- when given -- the session's evidence
  file and its sha256, which `d2_budget` and `derive_budget.py` read and
  nothing else.

THE LEDGER ONLY GROWS. A subrun id names ONE billed run: rebooking it with
the same run, cost and evidence is idempotent (the row is unchanged); naming
a DIFFERENT run, cost or evidence under a booked id is refused -- a billed
run must never be replaced or drop out of the cumulative cost. A run without
``cost.json`` is refused -- an unpriced subrun is not a free one.

TEARDOWN IS EVIDENCE. The launcher's extensionless ``teardown_confirmed``
(written only when the provider stops listing the pod) is copied and hashed
like every other record. A run without it is still booked -- its cost is real
-- but carries a ``resource_risk`` and blocks ``--close``: a campaign whose
resources are not confirmed released is not closed safely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[3]
SMALL = (".json", ".jsonl", ".log", ".txt")
#: Launcher markers with no extension that ARE evidence.
MARKERS = ("teardown_confirmed", "pod_id", "pod_start_epoch",
           "live_price_usd_per_hour")
TEARDOWN = "teardown_confirmed"
TEARDOWN_CONTENT = "confirmed"
MAX_BYTES = 5_000_000


class BookingError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def book(*, run_dir: Path, validation_dir: Path, subrun_id: str, outcome: str,
         evidence_session: str | None = None, close: str | None = None,
         notes: str = "", repo: Path = REPO) -> dict[str, Any]:
    run_dir, validation_dir = Path(run_dir), Path(validation_dir)
    cost_path = run_dir / "cost.json"
    if not cost_path.is_file():
        raise BookingError(f"{cost_path} is missing: the launcher writes it in "
                           "its teardown, and an unpriced subrun is not free")
    cost = json.loads(cost_path.read_text())
    camp_path = validation_dir / "campaign.json"
    campaign = json.loads(camp_path.read_text())
    dst = validation_dir / "runs" / run_dir.name
    copied: dict[str, str] = {}
    for p in sorted(run_dir.rglob("*")):
        rel = p.relative_to(run_dir)
        small = p.suffix in SMALL or (len(rel.parts) == 1
                                      and p.name in MARKERS)
        if (not p.is_file() or not small or "work" in rel.parts
                or p.stat().st_size > MAX_BYTES):
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        copied[str(rel)] = _sha(out)
    marker = dst / TEARDOWN
    #: NO RESOURCE EXISTED when the provider refused the create: the launcher
    #: records no pod id anywhere, so nothing bills and nothing needs tearing
    #: down. Its cost.json still prices the second between quote and refusal
    #: with no pod; that is not a bill.
    no_resource = (cost.get("pod_id") in (None, "")
                   and not (run_dir / "pod_id").is_file())
    confirmed = (None if no_resource else
                 TEARDOWN in copied
                 and marker.read_text().strip() == TEARDOWN_CONTENT)
    row: dict[str, Any] = {
        "subrun_id": subrun_id, "run_id": run_dir.name,
        "cost_usd": 0.0 if no_resource else float(cost["all_in_usd"]),
        **({"resource": ("none created: the provider refused the create; "
                         f"cost.json's ${cost.get('all_in_usd')} is the "
                         "launcher's wall-clock formula over no pod")}
           if no_resource else {}),
        "cost_record": cost,
        "pod_id": cost.get("pod_id"),
        "teardown_confirmed": confirmed,
        "teardown_evidence": ({"path": str(marker.relative_to(repo)),
                               "sha256": copied[TEARDOWN]}
                              if confirmed else None),
        "outcome": outcome, "notes": notes,
        "evidence": str(dst.relative_to(repo)),
        "evidence_sha256": copied}
    if confirmed is False:
        row["resource_risk"] = (
            f"no provider teardown confirmation for pod {cost.get('pod_id')!r}: "
            "reconcile against the provider (it must not list the pod) before "
            "the campaign may close; the cost is booked regardless")
    if evidence_session:
        ev = dst / evidence_session
        if not ev.is_file():
            raise BookingError(f"{evidence_session} was not copied from the "
                               "run; nothing to bind")
        row["evidence_session"] = str(ev.relative_to(repo))
        row["evidence_session_sha256"] = _sha(ev)
    subruns = list(campaign.get("subruns") or [])
    prior = [r for r in subruns if r.get("subrun_id") == subrun_id]
    if prior:
        old = prior[0]
        same = (old.get("run_id") == row["run_id"]
                and float(old.get("cost_usd", -1)) == row["cost_usd"]
                and old.get("evidence_sha256") == row["evidence_sha256"]
                and old.get("evidence_session_sha256")
                == row.get("evidence_session_sha256"))
        if not same:
            raise BookingError(
                f"subrun {subrun_id!r} is already booked for run "
                f"{old.get('run_id')!r} at ${old.get('cost_usd')}; booking "
                f"run {row['run_id']!r} under it would replace a billed run. "
                "Give the new run its own subrun id")
        row = old                                  # idempotent: unchanged
    else:
        subruns.append(row)
    campaign["subruns"] = subruns
    campaign["booked_usd"] = round(sum(float(r["cost_usd"]) for r in subruns), 4)
    #: THE COST IS WRITTEN FIRST, whatever happens to the close below.
    camp_path.write_text(json.dumps(campaign, indent=1, ensure_ascii=False)
                         + "\n")
    if close:
        risky = [r["subrun_id"] for r in subruns
                 if r.get("teardown_confirmed") is False]
        if risky:
            raise BookingError(
                f"cannot close: subruns {risky} have no provider teardown "
                "confirmation. Their cost is booked; reconcile the resources "
                "first")
        campaign["status"] = close
        camp_path.write_text(json.dumps(campaign, indent=1, ensure_ascii=False)
                             + "\n")
    return row


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--subrun-id", required=True)
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--evidence-session", default=None,
                    help="path, relative to the run dir, of the session record")
    ap.add_argument("--close", default=None,
                    help="set the campaign status, e.g. 'COMPLETE and CLOSED -- ...'")
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    try:
        row = book(run_dir=Path(a.run_dir).resolve(),
                   validation_dir=(REPO / a.validation_dir).resolve(),
                   subrun_id=a.subrun_id, outcome=a.outcome,
                   evidence_session=a.evidence_session, close=a.close,
                   notes=a.notes)
    except BookingError as exc:
        print(f"NOT BOOKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in row.items()
                      if k not in ("evidence_sha256", "cost_record")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
