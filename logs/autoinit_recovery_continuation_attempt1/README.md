# Recovery continuation attempt 1 — no stage ran, $0.01

**Verdict: fail-closed on a launcher transport defect, 27 seconds after pod
creation. Not a gate failure and not a scientific result.** No stage executed,
no leaf was read on the pod, no science changed. The five preserved Attempt-12
Stage-1 leaves are untouched on the dev box.

| | |
| --- | --- |
| authorization | `autoinit.recovery_continuation.2026-08-21T1642Z`, sha256 `846bd2d2eb7d…` |
| grant | `logs/autoinit_recovery_continuation_grant.json` |
| base commit | `b1ebbb641ca16ed6e11fd0746bd7cf6162297e3f` |
| session commit | `8c7c42e1a6867042a433ddd7e97fbc2622906185` |
| harness digest | `f2ea4332272e153fc46e9f0264abd4dca0bd11d0e366f67d64361b7f5dbd0f37` over 22 files, search excluded |
| bundle | `aad_autoinit_8c7c42e1.bundle`, sha256 `34fc7689e3b058d3…` |
| priced | 904.44 min, expected $14.9233, soft $16.4156, hard $16.7456 |
| pod | `dckc72mtoe9ijw`, L40S $0.99/h, 0.7 min, **$0.01**, provider confirms gone |
| terminal | `INCOMPLETE`, `launcher_error` |

## The chain executed correctly, all the way to the pod

Every step of the eight-step chain passed, and each is recorded in
`session.json`:

* the grant is a one-use document with `grant_type: recovery_continuation`;
* the pre-authorization base `b1ebbb6` was clean;
* the authorization was issued by the continuation-specific issuer, deriving
  `$14.9233 / $16.7456` from `continuation_budget()`;
* the authorization-only commit `8c7c42e` differs from its base in **exactly one
  path**, the authorization artifact;
* the bundle round-tripped: identical sha256 from the relay, checkout at the
  session commit, and the continuation harness digest **recomputed from the
  relay checkout** matched the authorized value;
* all four pre-provider `$0` gates passed — session commit and lineage, frozen
  science plan, continuation harness (22 files, search excluded), and all five
  preserved leaves verifying locally.

## What failed

```
[16:46:33] created dckc72mtoe9ijw at $0.99/h
[16:46:33] watchdog detached — hard 1015 min = $16.75
[16:47:00] LAUNCHER ERROR: URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] …>
[16:47:00] deleting pod (launcher error)
[16:47:14] pod deleted — 0.7 min, $0.01; provider confirms gone: True
```

**Root cause, measured afterwards at $0** (`endpoint_probe_20x.txt`): the RunPod
GraphQL endpoint was failing **5 of 20 requests — 25%** — with `SSL:
UNEXPECTED_EOF_WHILE_READING`, `ECONNRESET` and `RemoteDisconnected`.

`session_runner.wait_endpoint()` polls that endpoint every 10 s for up to 15
minutes — **up to 90 requests** — by calling `provider._gql()` **directly**.
`_gql` raises on any transport error and `wait_endpoint` catches nothing, so one
blip is fatal. `provider.get()` is the wrapper that exists for exactly this and
carries the comment *"Never raises. A watchdog that dies on a transient 502 is
not a backstop."* The same reasoning applies to the launcher and was not applied.

At a 25% per-request failure rate, surviving even five polls is `0.75^5 ≈ 24%`.
**Relaunching unchanged would repeat, not gamble.**

## Blast radius, which is bounded

Three call sites use `_gql` uncaught, all short-lived and all before the driver:

| site | calls | when | cost of a blip |
| --- | ---: | --- | --- |
| `check_gpu_offered` | 1 | before any pod exists | $0 |
| `wait_endpoint` | ≤90 | pod created and **billing** | ~$0.01–0.25 ← this run |
| `read_image_digest` | 1 | after SSH is up | ~$0.05 |

The **15-hour main poll loop uses `provider.get()`** and is not exposed. A
transient error mid-run does not kill the session.

## Why this stopped here rather than retrying

The grant is explicit: *"Consumed by exactly one issuance."* Fixing
`session_runner.py` moves the continuation harness digest, which invalidates the
authorization by design and would require a **second** issuance from a spent
one-use grant. That is the maintainer's decision, not this agent's.

## Proposed fix, for review — not applied

Give the three pre-driver `_gql` sites the tolerance `get()` already documents:
retry a transport error a bounded number of times before failing. `wait_endpoint`
already loops on a deadline, so the smallest correct change is to catch
`URLError`/`OSError` inside its loop and continue polling until the existing
`startup_limit_min` deadline — no new deadline, no new constant, and the
15-minute bound still fails closed.

This costs one paid launch to validate and needs: a new one-use grant, a new
pre-authorization base, re-issuance, a new bundle, and the full `$0` chain again.

## What is unchanged and still good

* the five Attempt-12 Stage-1 leaves, verified locally, `digest=MATCHED` 5/5;
* the frozen science — nothing was reopened;
* the authorization chain itself, which behaved exactly as designed;
* `$213.4814` cumulative against the `$234.00` cap.
