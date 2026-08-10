# Live control-plane canary — PROPOSAL

> **STATUS: EXECUTED 2026-08-09, twice.** Frozen as the prospective record and
> not edited to match the outcome. Run 1 **FAILED 9/10**
> ([`e7_canary_report.md`](e7_canary_report.md), $0.045); run 2 **PASSED 12/12**
> unaided ([`e7_canary_rerun_report.md`](e7_canary_rerun_report.md), $0.033).
> Four defects were found across the two runs, all fixed with regression tests.
> Records: [`EXPERIMENTS.md`](EXPERIMENTS.md) §32–33.

**Status when written: nothing has been launched. This requires separate
explicit authorization.**

## Why

The post-E6b hardening is verified against a local simulator: a shell standing in
for the pod, an in-memory control plane standing in for RunPod. 1,029 tests pass
and the E6b failure sequence replays cleanly. But the **live provider control
plane remains unverified**, and one transport in it has *never* been exercised
from this repository:

| path | status |
| --- | --- |
| GraphQL `pod(input:{podId})` polling | **verified by use** — every launcher since E2 |
| `runpodctl remove pod` termination | **verified by use** — every session |
| GraphQL `podTerminate` mutation (fallback) | **NEVER RUN**; journalled as `verified_transport: false` |
| `--terminate-after` | never observed to fire; demoted to a redundant third layer |
| detached start against a real ssh channel | **never run** — E6b is the only live data point, and it blocked |
| log relay against a real pod | never run |
| watchdog reaching a threshold on a real billing pod | never run |

A 776-minute, $12.82 E7 session must not be the first live test of any of these.
The canary is ~32 minutes and $0.53 expected.

## What it runs

A **harmless short process, not model training**: a shell loop that appends a
structured jsonl event every few seconds, writes markers, and sleeps. No GPU
work, no weights, no dataset. The cheapest GPU or CPU-only pod that the account
can create is acceptable — the object under test is the control plane, not the
accelerator.

## Success criteria — all nine must pass

1. **Detached start returns promptly.** `scripts/pod/start_job.py` returns a
   descriptor in under 120 s while the remote process is still running. Recorded:
   wall time, `start_channel_closed`, `confirmed_by`.
2. **Durable job descriptor exists on the pod.** `<job_dir>/<id>.job.json` and
   the pidfile are readable over a *second, later* ssh connection, and the pid is
   alive.
3. **Structured logs reach the relay before teardown.** `LogRelay` syncs the
   remote jsonl incrementally; at least three cycles land, and the local copy
   parses as JSONL with a strictly increasing event counter.
4. **The provider-only watchdog observes a billing pod.** With SSH deliberately
   left idle, the watchdog's poll journal shows `pod_billing: true` and an
   increasing `elapsed_minutes` computed from its own clock.
5. **The watchdog reaches its configured threshold.** `hard_limit_reached`
   appears in the journal at the configured minute, not before.
6. **Termination is issued** and every attempt and response is journalled.
7. **Provider disappearance is verified by polling** — `terminate_verify` rows
   followed by `terminated`, i.e. the pod is *gone*, not merely requested to go.
8. **Artifact manifest records the surviving outputs.**
   `collect_artifacts.py manifest` produces a complete manifest; the archive is
   built from it and `verify-archive` passes.
9. **Local hash verification completes.** `verify-local` re-hashes the retrieved
   copies against the pod-side manifest with zero mismatches, and the teardown
   gate returns `allowed: true`.

## Forcing the unverified fallback

The primary termination path (`runpodctl remove pod`) has historical use; the
GraphQL `podTerminate` mutation does not. The canary must exercise the fallback
**and** the post-termination verification, safely.

**Proposed method — make the primary path fail without touching the pod.** Run
the watchdog with `--runpodctl /nonexistent/runpodctl-canary`, a path that does
not exist. `RunPodProvider._terminate_cli` then raises `OSError`, records a
failed `TerminationAttempt` with `verified_transport: true, ok: false`, and
`terminate()` falls through to `_terminate_gql`. Nothing about the pod, the
account or the API key is altered; only this process's view of the CLI changes.

Two properties make this safe rather than clever:

* it is **reversible and local** — no provider state is modified to induce the
  failure;
* the **verification poll is unchanged**, so if the fallback also fails, the
  watchdog escalates to `TERMINATION_FAILED` exactly as designed, and the run is
  torn down by hand with the journal as evidence.

**Two-phase design so a fallback failure cannot strand a pod.** Phase 1 runs the
watchdog with the broken CLI path and a short threshold. If `terminated` appears,
the GraphQL transport is verified and the pod is gone. If `TERMINATION_FAILED`
appears, phase 2 immediately re-runs the watchdog with the **real** `runpodctl`
and the same pod id — the verified path — and the canary reports "fallback does
not work" as its finding while still leaving nothing running.

`--terminate-after` is set as a redundant third layer and **is not counted as a
success criterion**; if it fires first, that is recorded as new information
(it would be the first observation of it working) but does not substitute for
criteria 6–7.

## Failure criteria — any one fails the canary

* the detached start does not return within 120 s, or returns without a
  confirmable descriptor;
* no structured events reach the relay before teardown;
* the watchdog does not observe the pod, or its elapsed clock disagrees with pod
  age by more than a minute;
* the threshold does not fire at the configured minute;
* termination is issued and the pod is still `RUNNING` after every configured
  round **and** phase 2 is required (fallback unusable — a real finding, and a
  failed canary);
* the artifact manifest is incomplete, the archive does not cover it, or any
  local hash mismatches;
* the pod outlives the absolute termination threshold by more than 10 minutes.

## Cost

Live price must be re-quoted at launch and refused if above the authorized rate
(the existing `securePrice` guard). Priced here at the L40S rate every Stage 3
session has paid, **$0.99/h** — a cheaper pod only improves it.

| threshold | minutes | $ |
| --- | ---: | ---: |
| setup (minimal image, no venv build, no weights) | 12.0 | 0.20 |
| detached start + descriptor | 2.0 | 0.03 |
| log-relay cycles | 6.0 | 0.10 |
| watchdog threshold + termination | 8.0 | 0.13 |
| artifact manifest + hash verify | 4.0 | 0.07 |
| **expected completion** | **32.0** | **$0.53** |
| soft stop (start nothing new) | 40.0 | $0.66 |
| artifact-recovery reserve | 10.0 | $0.17 |
| **absolute termination (hard backstop)** | **50.0** | **$0.82** |

**Live quoted pod price: not yet obtained.** Quoting it does not require creating
a pod — the launcher's existing `securePrice` GraphQL guard reads it — but the
figure changes hourly, so it is deliberately not frozen into this document. It
will be re-read and reported at launch, and the launch aborts if it exceeds the
authorized rate.

## What the canary does **not** establish

* nothing about training throughput, memory or numerics;
* nothing about `--terminate-after` unless it happens to fire;
* nothing about a *long* session — a 32-minute pod does not exercise the
  multi-hour behaviours that produced E6b's 434-minute block. It verifies the
  transports and the control loop, which is what E7 would otherwise be testing
  for the first time with $12.82 at stake.

## Authorization required

```
canary hard backstop:      $0.82
actual cumulative spend:  $149.59   (the planning baseline)
proposed cap for canary:  $150.41
```

Do not launch without separate explicit authorization.
