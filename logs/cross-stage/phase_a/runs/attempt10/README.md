# Phase-A attempt 10 — 2026-08-18/19, $11.43, INCOMPLETE: operator runtime-cost failure

**Not a scientific result. Not a Stage-1 selection result.** Stage 0 passed and
attested; Stage 1 entered its third operator expansion and was still inside it
11.5 hours later, with the paid L40S effectively idle. Stopped on maintainer
instruction through the normal collection path; pod deleted, provider confirmed.

| | |
| --- | --- |
| authorization | `autoinit.phase_a.2026-08-18T1746Z`, sha256 `399290cbaf235f503173d3c1359b1928d3edabfb55e036c7fcf12917fc7bb6fd` |
| grant | [`../autoinit_phase_a_attempt10_grant.json`](../autoinit_phase_a_attempt10_grant.json), sha256 `3ef080d91d58e54a27b346d174b6b6c63863fe5c63664fb4016aac0809be8f24` |
| authorized base | `f6ced658a42a0b4f81e5e6e706a6cd4935fadcf0` |
| session commit | `8183a23ef74fb9fba738ef62feef43dc72445b53` |
| harness digest | `24d89b9ffe8e41f2712d2f105ca2076fd0552ccea7a089aeb5f2a844c39e1f42` (unchanged; the Stage-1 device fix rides in the session commit, not the digest) |
| bundle | `transfer/aad_autoinit_8183a23e.bundle`, sha256 `7d42d783500f7fd3962e3dbdb2c297cc98e1a70e1aa6308073b8ae72fbd57aad` |
| pod | `nex2u8no79k6pl`, 1×L40S @ $0.99/h, image `…cu1300-torch291-ubuntu2404@580.159.03` |
| lifetime | 2026-08-18 17:48:22 → 2026-08-19 05:20:49 UTC, **692.5 min** |
| cost | **$11.43** |
| terminal | `DRIVER_EXITED:143` — SIGTERM, sent deliberately on maintainer instruction |

## Timeline

```
17:48:22  pod created, watchdog armed 1397 min = $23.05
17:57:32  SETUP_DONE           ($0.15)   TESTS_OK in 211 s
17:57:42  DRIVER_START → STAGE_START:0
17:59:50  STAGE_PASSED:0 — attested interval 0.011695, floor 0.3000, plan 02be33b9
17:59:50  STAGE_START:1
18:15:13  states.jsonl last written (2 states)      <-- last progress of any kind
   …      10 h 47 m with no state, no probe, GPU 0-1 %
05:18:07  driver stopped; launcher polls DRIVER_EXITED:143
05:20:02  manifest rc=0, 9 files across 5 classes, streams quiescent
05:20:38  teardown gate allowed=True
05:20:49  pod deleted — provider confirms gone: True
```

## The three device fixes held

The attempt-9 defect did not recur. `stream_projection` ran — the composite
expansion below it completed and wrote a state — so `avg`, the orthonormality
`eye` and `_head_rows` all placed correctly on a real CUDA device. The bounded
audit did what it was for.

## What was actually running

Two states were written, in the deterministic registry order:

| # | implementation | state |
| --- | --- | --- |
| 1 | `attention.weight_proxy_v0` | `ATTENTION`, h=2560 L=36 ffn=9728 |
| 2 | `composite.stage1_sandwich_v0` | `COMPOSITE_STAGE1`, h=1024 L=28 ffn=3072 |
| 3 | **`depth.causal_kl_greedy_v1`** | never completed — **this is what consumed 10 h 47 m** |

`depth.causal_kl_greedy_v1` is third in `registered_implementations()`, and the
two completed states are exactly the first two. The identification is by
deterministic order, not by inference from the symptom.

### Why it is slow, in its own numbers

`apply()` calls `greedy_removal(score, 36, 8)`. `expected_evaluations(36, 8)` is
**260** — its docstring says so — and `score()` loops over all **67** calibration
items:

```
260 evaluations x 67 items = 17,420 forward + distortion pairs
```

Each pair does a GPU forward, **copies the logits to the host**
(`_forward_logits` returns `.cpu()`; `targets` are `.cpu()` too), and then runs
`distortion()` — a full-vocabulary softmax/KL — **on the CPU**:

```
vocab 151,936 · 59,763 calibration positions
one logits tensor, all positions        33.8 GiB float32
CPU traffic per evaluation              ~0.33 TiB   (~10 elementwise passes)
CPU traffic, all 260 evaluations        ~86 TiB
device -> host copies over the expansion ~8.6 TiB
```

The `.cpu()` is deliberate and documented — `distortion`'s docstring says a
full-sequence float32 softmax of both models at once is "a needless memory spike
on the one device that also has to hold the teacher". **The transfer was
reasoned about; the CPU cost of the reduction was never priced.**

### The multiplier that makes it worse

This run's own setup log: **`128 vCPUs visible, cgroup budget 13; cpu set 0-12`**.
`autoinit_preflight_setup.sh` computes that budget correctly and applies
`OMP_NUM_THREADS` + `taskset -c 0-12` — **only to the test suite** (lines 463-4).
The driver is started by `start_job.py` with no such constraint, so torch sized
its pools from the 128 it could see: **192 threads on 13 granted CPUs**, measured
at ~65 min accumulated CPU each. On a bandwidth-bound reduction with BLAS
barriers, ~15x oversubscription costs far more than it buys.

Bandwidth alone predicts ~3 h for 86 TiB at 10 GB/s. The observed >10.78 h
without completion is consistent with that estimate degraded by the
oversubscription, but the two were not separated experimentally and this
directory does not claim they were.

### Measured versus priced

| | |
| --- | --- |
| priced | the **whole** beam search at `--search-minutes 180.0` = 3.0 h |
| measured | the causal-depth expansion **alone**, 647 min = 10.78 h, **not finished** |

At least **3.6x the entire search budget for one expansion**, and that is a lower
bound — it had not completed.

### Nothing enforced the budget

`--search-minutes 180.0` is consumed only by `self.afford(self.a.search_minutes,
"beam search")` at `autoinit_phase_a_driver.py:433` — an **affordability check
before the search starts**. `search.py` records `elapsed` and `wall_seconds` but
never compares them to a deadline, and `_expand_one` has no clock at all. A single
expensive expansion therefore runs unbounded. The only backstop was the watchdog
at the full $23.05 ceiling.

### How far through the 260 evaluations it got is UNKNOWN

`greedy_removal` accumulates rounds in memory and the state is journalled only
when the expansion **completes**. There is no per-round artifact, so the run
emitted no external progress signal between 18:15 and the stop. It could have
been 5 % through or 95 %; this record does not guess.

## Collected

The normal collector ran against `spec_failed`: manifest `rc=0`, 9 files across 5
classes, `final_streams_quiescent: True`, teardown gate `allowed=True`.
[`search_states.jsonl`](search_states.jsonl) preserves both states' specs and
hashes; [`driver_run.log`](driver_run.log), [`driver.status`](driver.status),
[`phase_a_evidence.json`](phase_a_evidence.json),
[`attested_evaluation_protocol.json`](attested_evaluation_protocol.json),
[`launcher.out`](launcher.out), [`watchdog.jsonl`](watchdog.jsonl),
[`poll.log`](poll.log).

**The two states' weights were NOT fetched** — they are not in the `spec_failed`
artifact set — and are gone with the pod. Their identities survive. They must not
be resumed against a repaired numerical path without demonstrated compatibility;
the loss of the weights makes that impossible to do accidentally.

## Disposition

Recorded as **incomplete / operator runtime-cost failure**. No scientific claim,
no Stage-1 selection result, no probe. The pod was not repaired; the stop used
the supported path (the driver stopping makes the launcher's poll break on
`DRIVER_*` and run its normal `collect_and_teardown`).

Cumulative spend $194.5830 → **$206.0130** of the $219.00 cap, leaving
**$12.9870** — not enough for another full Phase-A attempt at the $23.0484
ceiling. A retry requires a separate budget decision after the runtime fix is
reviewed.
