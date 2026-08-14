# Continuation attempt 5 — INCOMPLETE. $0.1369, 8.3 min, pod deleted and confirmed gone.

**Launched under the maintainer's GO**, at the frozen identity:
checkout `40c3d53a…`, bundle `aad_autoinit_40c3d53a.bundle` (`a18f22a5…`),
transport relay, authorization `e4854818…`, plan `79da6d7a…`, harness `a1d1f3fc…`.
That authorization is **consumed** — one invocation, success or failure.

## What passed

| | |
| --- | --- |
| ssh reachable | 1.8 min — a warm host, not a cold draw |
| bundle + relay staging | OK |
| **offline train env** | installed, no PyPI resolve on the critical path |
| **offline vLLM env, 196/196 hash-verified** | the wheel-byte gate passed on the pod |
| teacher + RoPE across both venvs | 5,000,000 stored = runtime, in transformers 5.13.1 **and** 5.15.0 |
| CPU suite on the pod | 1564 passed, 61 skipped, 163 s |

Every problem the last four attempts died of is fixed and was demonstrated on
real hardware. The offline dependency work did what it was meant to do.

## What failed

The **last line of setup**, after the tests:

```
a = SpendAuthorization.load('logs/autoinit_micro_preflight_authorization.json')
a.require_plan(PREFLIGHT_PLAN_V1.plan_hash)
AuthorizationError: bound to preflight plan afd08be7… but the plan about to run
hashes to 83218ddd…
```

**The guard is correct and the binding was stale.** `autoinit_preflight_setup.sh`
is shared, and its final gate hardcodes the *micro-preflight* authorization and
the *preflight* plan. `PREFLIGHT_PLAN_V1.plan_hash` moved from `afd08be7…` to
`83218ddd…` when `pooled_counts@v2` changed the plan's description string in
`recovery.py`; the historical micro-preflight artifact was never re-issued
against it. So a session that has nothing to do with that authorization died on
it.

The continuation's own binding was right and was verified at $0 before the pod
existed: `CONTINUATION_PLAN_V1.plan_hash` == `79da6d7a…` == the authorization's.

## Why rehearsal missed it, for the third time

`simulate_pod_env.sh` runs the **test suite** the pod runs. Nothing executes the
setup script's authorization block. This is the same failure mode as the two
defects found in the wheelhouse gate hours earlier — a line that is read but
never run — and it is the third time an unexecuted line in this one script has
cost money ($0.07, $1.3672, $0.1369).

## Do not retry without review

Per the maintainer's instruction. The fix is small and obvious, but the
authorization is consumed and the pattern — patch, relaunch, discover the next
unexecuted line — is what the last five attempts were.
