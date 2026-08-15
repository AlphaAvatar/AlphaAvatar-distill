# Continuation attempt 6 — INCOMPLETE. $0.1324, 8.0 min, pod deleted and confirmed gone.

Launched under authorization `f21b4038…` (now **consumed**), checkout
`db8bf1f7…`, bundle `aad_autoinit_db8bf1f7.bundle` (`f69d4120…`), harness
`a1234f01…`, plan `79da6d7a…`, transport relay.

## Setup SUCCEEDED. The launcher misread it.

```
[21:01:06] verifying logs/autoinit_continuation_authorization.json binds to this session's plan
  autoinit.control_characterization.2026-08-14T2039Z: stages [0,1,2,3], hard $4.54, phase A False
MARKER:AUTHORIZATION_OK
NVIDIA L40S, 46068 MiB, 580.159.03
MARKER:SETUP_DONE
[21:01:07] setup complete
SETUP_RC=0
[21:01:14] ABORT after draw 1: setup_failed
```

**The design fix worked.** The session-scoped authorization gate passed on the
pod, naming this session's own artifact — the exact failure of attempt 5 is
closed.

What failed is one line older than either bug. The shared setup script wrote its
markers to a hardcoded `$WS/autoinit_preflight.status`, while the continuation
launcher probes `$WS/autoinit_continuation.status`. So `SETUP_DONE=0`, and
`setup_done in ("", "0")` returned `setup_failed` for a setup that had finished
cleanly. `setup_rc` was `0` in the session record while `setup_done` was empty —
the two disagreed, and only one was consulted.

Attempt 5 recorded exactly the same empty `setup_done`; its `setup_rc` was `1`,
so the real defect was masked by a genuine failure.

## Same class as attempt 5, one layer down

Both are a **shared** script hardcoding a preflight-specific name while the
continuation assumes its own. The authorization one was fixed by passing the
session's values in; this is now fixed the same way — `SESSION_STATUS`, forwarded
from the same `STATUS` expression the launcher probes with, so the two cannot
drift.

A sweep found no third instance: every other marker read in the launcher already
goes through that same `STATUS`, and the driver's own hardcoded path agrees. That
agreement is now pinned by a test rather than left to coincidence.

## Also found, unpaid

The first draft of the fix wrote
`STATUS=${SESSION_STATUS:?...this session's status file}`. An unquoted `${v:?word}`
expands its word, so the apostrophe opened a quote — and **`bash -n` still passed**,
because it paired with a later one. Caught by executing the line rather than
linting it. The message is now apostrophe-free and double-quoted.

## Not retried

Per the maintainer's instruction. `f21b4038…` is consumed.
