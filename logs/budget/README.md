# logs/budget

Money and permission. Nothing here is a scientific result.

| what | where |
| --- | --- |
| every cost, per session, append-only | [`ledger.md`](ledger.md) |
| decision records | [`decisions.md`](decisions.md) |
| grants and issued authorizations | [`approvals/`](approvals/) |

**The live balances are derived, not stored here.** Run
[`derive_budget.py`](../../scripts/consolidate/derive_budget.py): it reads the
approved package from `configs/` and each session's own closeout, and reports
the formal, engineering, package and project balances separately. They bind
separately and do not transfer into one another — dividing the wrong one by the
per-session ceiling once produced a fundable-session count that was too high.

`approvals/` holds **consumed** artifacts. They are evidence: never edited,
never re-issued, never made valid again by changing their contents. A path
written inside one states where a file was when it was issued — see
[`../index.json`](../index.json)`.historical_paths` to follow it forward.
