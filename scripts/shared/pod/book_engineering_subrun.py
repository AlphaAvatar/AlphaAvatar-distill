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

A subrun id is booked ONCE: rebooking replaces its row, never adds a second.
A run without ``cost.json`` is refused -- an unpriced subrun is not a free one.
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
        if (not p.is_file() or p.suffix not in SMALL or "work" in rel.parts
                or p.stat().st_size > MAX_BYTES):
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        copied[str(rel)] = _sha(out)
    row: dict[str, Any] = {
        "subrun_id": subrun_id, "run_id": run_dir.name,
        "cost_usd": float(cost["all_in_usd"]),
        "cost_record": cost,
        "pod_id": cost.get("pod_id"),
        "teardown_confirmed": (run_dir / "teardown_confirmed").is_file(),
        "outcome": outcome, "notes": notes,
        "evidence": str(dst.relative_to(repo)),
        "evidence_sha256": copied}
    if evidence_session:
        ev = dst / evidence_session
        if not ev.is_file():
            raise BookingError(f"{evidence_session} was not copied from the "
                               "run; nothing to bind")
        row["evidence_session"] = str(ev.relative_to(repo))
        row["evidence_session_sha256"] = _sha(ev)
    subruns = [r for r in campaign.get("subruns") or []
               if r.get("subrun_id") != subrun_id]
    subruns.append(row)
    campaign["subruns"] = subruns
    campaign["booked_usd"] = round(sum(float(r["cost_usd"]) for r in subruns), 4)
    if close:
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
