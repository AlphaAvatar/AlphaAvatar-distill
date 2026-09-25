#!/usr/bin/env python3
"""What the batch-invariance diagnostic may spend on one session.

Extracted from the launcher so it can be tested. The launcher used to carry the
arithmetic inline in bash, where the only way to find out what it computed was
to create a pod.

    python scripts/pod/batch_invariance_budget.py CAMPAIGN.json [--session-cap N]

prints one line of shell-readable fields:

    campaign_ceiling spent reserve remaining session_ceiling max_seconds ok

`ok` is `yes` or `no`. The caller refuses on `no`; it does not re-decide.

ONE OWNER PER NUMBER. The cumulative ceiling lives in the AUTHORIZATION, which
the campaign record points at by path; the campaign record holds the costs. The
first draft restated the ceiling in both files, where two equal numbers hide
which one a gate is actually reading.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

#: Below this, a session cannot reach a measurement: setup alone has taken 5,
#: 8.5 and 150+ minutes on the same script and image in this project.
MIN_USEFUL_USD = 0.60


class BudgetError(RuntimeError):
    """The budget cannot be established. Never read as 'plenty'."""


def derive(campaign_path: str | Path, *, session_cap_usd: float,
           rate_usd_per_hour: float) -> dict:
    """The session's ceiling, from the records that own each part of it."""
    campaign_path = Path(campaign_path)
    try:
        campaign = json.loads(campaign_path.read_text())
    except (OSError, ValueError) as exc:
        raise BudgetError(f"cannot read {campaign_path}: {exc}") from exc

    auth_rel = campaign.get("authorization")
    if not isinstance(auth_rel, str):
        raise BudgetError(
            f"{campaign_path}: `authorization` must be a PATH to the record "
            f"that owns the ceiling, got {type(auth_rel).__name__}")
    #: Relative to the repository root, which is this file's grandparent --
    #: the same convention derive_budget.py resolves campaign records by.
    root = Path(__file__).resolve().parents[2]
    try:
        auth = json.loads((root / auth_rel).read_text())
    except (OSError, ValueError) as exc:
        raise BudgetError(f"cannot read {auth_rel}: {exc}") from exc

    ceiling = auth.get("all_in_ceiling_usd")
    if not isinstance(ceiling, (int, float)):
        raise BudgetError(f"{auth_rel} states no numeric all_in_ceiling_usd")
    reserve = float(campaign.get("teardown_reserve_usd", 0.0))

    #: RECOMPUTED from components. A roll-up total has silently dropped $0.0073
    #: in this project, and every figure beside it was right.
    spent = 0.0
    for key in ("inherited_spend", "subruns"):
        for entry in campaign.get(key) or ():
            cost = entry.get("cost_usd")
            if cost is None:
                raise BudgetError(
                    f"{campaign_path}: {key} entry "
                    f"{entry.get('subrun_id', entry)!r} has no cost_usd; an "
                    "unpriced subrun is not a free one")
            spent += float(cost)

    remaining = float(ceiling) - spent
    #: FLOORED. A limit handed downward rounds DOWN or it is not a limit.
    affordable = math.floor((remaining - reserve) * 100) / 100
    session = min(affordable, float(session_cap_usd))
    return {
        "campaign_ceiling": round(float(ceiling), 4),
        "spent": round(spent, 4),
        "reserve": round(reserve, 2),
        "remaining": round(remaining, 4),
        "session_ceiling": round(session, 2),
        "max_seconds": max(0, int(session / float(rate_usd_per_hour) * 3600)),
        "ok": session >= MIN_USEFUL_USD,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaign")
    ap.add_argument("--session-cap", type=float, default=2.50,
                    help="leave a second attempt affordable; four of this "
                         "project's paid sessions died on infrastructure "
                         "before measuring anything")
    ap.add_argument("--rate", type=float, default=1.09,
                    help="securePrice USD/h, NOT communityPrice")
    args = ap.parse_args(argv)
    try:
        d = derive(args.campaign, session_cap_usd=args.session_cap,
                   rate_usd_per_hour=args.rate)
    except BudgetError as exc:
        print(f"BUDGET UNAVAILABLE: {exc}", file=sys.stderr)
        return 4
    print(f"{d['campaign_ceiling']:.4f} {d['spent']:.4f} {d['reserve']:.2f} "
          f"{d['remaining']:.4f} {d['session_ceiling']:.2f} {d['max_seconds']} "
          f"{'yes' if d['ok'] else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
