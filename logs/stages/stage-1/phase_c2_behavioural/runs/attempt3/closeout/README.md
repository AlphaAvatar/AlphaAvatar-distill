# closeout/

how it ended: the outcome classification, the measured cost and the provider teardown confirmation.

Roles this run declares here: `outcome`.

Described by `../manifest.json`, which is authoritative.

## Known anomaly in this attempt's governance record

`../governance/grant.json` records `granted_utc = 2026-09-21`. The session
actually ran on **2026-09-20 UTC** (pod created 16:38, deleted 18:56). The
grant was written by hand and the date came from the dev box's local timezone,
which is ahead of UTC, so a `+1` date landed in a field whose name says `utc`.

**The grant is consumed evidence and is not rewritten.** It is recorded here
instead, prospectively, so a later reader finds the discrepancy explained
rather than having to rediscover it.

**It distorts no money.** Run costs reach the project balance from closeouts
through `logs/index.json`; `derive_budget._attribute()` reads timestamps only
from engineering campaign ledgers, and this is not one. The figure this attempt
contributes — `$2.5425` — is derived from `outcome.json :: money.all_in_usd`
and carries no date.

**It cannot recur silently.** `issue_c2_behavioural_authorization.py` now
refuses a grant dated after the current UTC date, and the authorization's own
`granted_utc` has always come from `datetime.now(timezone.utc)` in code.
