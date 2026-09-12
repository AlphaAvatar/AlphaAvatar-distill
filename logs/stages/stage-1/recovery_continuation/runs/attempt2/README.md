# Recovery continuation attempt 2 — no stage ran, $0.2389

**Verdict: fail-closed on local-asset staging, 10.5 minutes into the pod. The
five Stage-1 leaves have no viable transport to a pod today.** No stage executed,
no science changed. The leaves are untouched in the dev-box checkpoint store.

| | |
| --- | --- |
| authorization | `autoinit.recovery_continuation.2026-08-21T2004Z`, sha256 `803c84f833d6…` |
| grant | `logs/autoinit_recovery_continuation_attempt2_grant.json` (sha256 `a29dac6fd120…`) |
| base commit | `b2f0dc5c07a36b729b0902c04cb2fe5ed5f734dd` |
| session commit | `3e3f22955af4835e9be1474ddf0d245bbb4503ce` |
| harness digest | `e5a7183a0cf10d5af7c2d4422656a122e88c3ff391af24fcd9ac4ce159c98198`, 22 files, search excluded |
| bundle | `aad_autoinit_3e3f2295.bundle`, sha256 `96c77475e85f61cf…` |
| pod | `7hthdteyc25xgx`, L40S $0.99/h, **14.48 min, $0.2389**, provider confirms gone |
| terminal | `INCOMPLETE`, `launcher_error` |

## What worked — everything up to the bytes

The provider-resilience closure did its job: the readiness poll that killed
attempt 1 succeeded here, reaching **TCP 22 at 3.7 min**, and the image identity
was confirmed. The full chain passed: one-use grant, clean base, continuation
issuance, an authorization-only commit differing from its base in exactly one
path, a bundle round-trip verified by bytes, checkout and **harness digest
recomputed from the relay checkout**, and all four pre-provider `$0` gates.

## What failed

```
[20:10:17] draw 1: ssh reachable — $0.06
[20:20:49] LAUNCHER ERROR: TimeoutExpired: Command '['scp', …
           /home/ecs-user/aad-artifacts/autoinit/phase_a/cca699c93f34…,
           'root@64.247.206.216:/workspace/assets/cca699c93f34…']'
           timed out after 600 seconds
[20:21:00] pod deleted — 14.5 min, $0.24; provider confirms gone: True
```

`SessionRunner` stages each declared `LOCAL_ASSET` with
`subprocess.run(scp …, timeout=600)`. That timeout is hard-coded, and
`subprocess.run` **raises** `TimeoutExpired`, so it propagated to the top-level
handler and tore the pod down.

**It could not have succeeded.** One leaf is **1.110 GiB (1192 MB)**. Fitting
that into 600 s needs **1.99 MB/s sustained**:

| observed uplink | one leaf | five leaves |
| --- | ---: | ---: |
| 0.44 MB/s — this session's own bundle upload, minutes earlier | 45 min | 226 min |
| 0.72 MB/s — the recorded dev-box figure | 28 min | 138 min |
| **1.99 MB/s — required** | 10 min | 50 min |

The dev box has never been observed within **3–4.5×** of the rate this needs. The
first leaf timing out was the only possible outcome, on this attempt or any
other, and it would have repeated four more times.

## Why no $0 gate caught it

Each gate was true and none was sufficient:

* `selected_leaves_present_gate` asks whether the leaves **exist and verify
  locally**. They do. It does not ask whether they can be *delivered*.
* `test_each_real_session_manifest_stages_through_the_real_block` exercises
  `SESSION_RELAY_INPUTS` — assets the pod **pulls** from the relay. The leaves
  are `LOCAL_ASSETS`, which the launcher **pushes** by scp. Different transport,
  not covered.
* The pod simulator never scps anything.

Declared, verified, and undeliverable: the gap is between "the bytes are correct"
and "the bytes can arrive".

## The transport is closed on both sides

* **scp** — needs 1.99 MB/s against ≤0.72 MB/s observed. Infeasible.
* **relay** — `--stage-leaves-to-relay` is off by default and documented as such
  in the launcher: the relay reported `usedStorage` 91.54 GiB against an inferred
  93.13 GiB limit, **1.60 GiB of headroom** against **5.55 GiB** of leaves.

So the five leaves currently have **no route to a pod**. That is the finding, and
it is a design decision for the maintainer, not something to route around
mid-run.

## Not attempted, and why

No relaunch. The grant makes a failed staging gate a fail-closed stop, the
authorization is spent, and — decisively — the arithmetic above says a rerun
fails identically. Nothing was repaired on the live pod.

## State

* pod deleted, **provider confirms gone**; watchdog ended `pod_gone` after 16
  ticks; nothing billing;
* the five Attempt-12 leaves are **untouched** in
  `/home/ecs-user/aad-artifacts/autoinit/phase_a/`, still verifying locally;
* frozen science untouched; `$213.7203` cumulative against the `$234.00` cap.
