# Recovery continuation attempt 5 — both memory repairs VERIFIED, first probe trained, $1.3511

**Verdict: the memory-contract repairs are confirmed on hardware and the first
recovery probe trained to completion. Stage 2 then failed reading that probe's
output, on a checkpoint-path contract the driver had wrong.** Fail-closed, torn
down with provider confirmation, no relaunch.

| | |
| --- | --- |
| authorization | `autoinit.recovery_continuation.2026-08-22T1925Z`, sha256 `7f575d5f…` |
| grant | `logs/autoinit_recovery_continuation_attempt5_grant.json` |
| base commit | `4794193` |
| session commit | `63625e159b8992b88c0ddb700f6b55d20051a0dc` |
| harness digest | `95cf336d…`, 22 files, search excluded |
| bundle | `aad_autoinit_63625e15.bundle`, sha256 `0e4e989b…` |
| pod | `9jxov5bjtiy2xu`, L40S $0.99/h, **81.88 min, $1.3511**, provider confirms gone |
| terminal | `DRIVER_EXITED:22`, `PHASE_A_FAILED` at stage 2 |

## The two memory repairs are verified on hardware

`device_handoff.json`, against attempt 4's identical `before`:

| | attempt 4 | attempt 5 |
| --- | ---: | ---: |
| allocated **before** | 8,110,229,504 | 8,110,229,504 |
| allocated **after** | 8,110,229,504 | **8,519,680** |
| `freed_allocated_bytes` | **0** | **8,101,709,824** |
| free after | 36.32 GiB | **43.87 GiB** |
| `live_retention` | **true** | **false** |

The caller-owned release freed **7.54 GiB** that attempt 4 could not free at all.
`require_released` passed because there was nothing retained, and `require_headroom`
passed on **43.87 GiB against the 43.65 GiB requirement** — clearing by 0.22 GiB,
which is the figure the repair predicted before the run, to two decimal places.

**The 41.65 GiB requirement was therefore not merely conservative — it was very
nearly exact.** The old 22 GiB constant would have passed here too, but only by
accident; the probe that followed is the thing it had to admit, and it fitted.

## The first recovery probe trained

```
2026-08-22T20:03:25Z MARKER:STAGE_START:2
2026-08-22T21:05:06Z MARKER:PROBE_TRAINED:autoinit.v1.phase_a.rung1.cca699c93f34.sa
```

**61.7 minutes**, against the 61.55-minute figure the budget is priced from — the
first recovery probe this continuation has ever completed, and a direct
confirmation of the pricing basis. Its tail shows the model shards written.

## What failed

```
File "scripts/pod/autoinit_phase_a_driver.py", line 737, in run_probe
    latest = (trained / "latest.txt").read_text().strip()
FileNotFoundError: [Errno 2] No such file or directory:
  '/workspace/aad/artifacts/stage3/phase_a/autoinit.v1.phase_a.rung1.cca699c93f34.sa/latest.txt'
```

A **writer/consumer contract gap**, with the writer right and one of its two
consumers wrong:

| role | path |
| --- | --- |
| **writer** — `src/aadistill/training/train.py:1208` | `out_dir / "checkpoints" / "latest.txt"`, checkpoints under `out_dir/checkpoints/<tag>/model` |
| **consumer A** — `scripts/training/train_stage3.py:74-86` (resume) | `ckpt_root = out_dir / "checkpoints"`; `ckpt_root / "latest.txt"`; `ckpt_root / tag / "model"` — **correct** |
| **consumer B** — `scripts/pod/autoinit_phase_a_driver.py:736-738` | `trained / "latest.txt"`; `trained / latest / "model"` — **missing the `checkpoints/` component in both** |

The resume path has had it right all along. The driver's post-training read is
the only place the component is dropped, and it is reached exactly once per
probe — **after** ~62 minutes of GPU training.

### Why no `$0` gate caught it

Every gate was true and none was sufficient. Reaching this line requires a *real*
completed probe: the driver spawns `train_stage3.py` as a subprocess and reads
its output only after `PROBE_TRAINED`. No `$0` path trains a probe, and the pod
simulator and rehearsal stub the training subprocess rather than producing a
checkpoint tree, so the read has never been executed against real training
output. This is the same shape as
[`autoinit_recovery_continuation_attempt3/`](../autoinit_recovery_continuation_attempt3/)
— an unexecuted line on a path only a paid pod reaches — but at the far end of
the run rather than in setup.

## Not attempted, and why

No repair on the live pod and no relaunch. The grant makes any failure a
fail-closed stop for review. The candidate repair touches
`autoinit_phase_a_driver.py`, which **is** in the 22-file harness, so it moves
the continuation harness digest as well as the session commit.

**The probe's own artifacts are lost with the pod.** `checkpoints_fetched` is
empty and `required_products_secured` reports "stage 1 did not stage any selected
leaves" — the fetch spec collects finalists, and a rung-1 probe that was never
scored is not one. 62 minutes of training was paid for and not retained; whether
a future attempt should journal or fetch a trained-but-unscored probe is a
question for the maintainer, not something to change unasked.

## State

* pod deleted, **provider confirms gone**; watchdog ended `pod_gone` after 82
  ticks; poller stopped; provider returns zero pods; nothing billing;
* evidence collected through the supported path before teardown, 15 entries,
  none missing;
* the five Attempt-12 leaves are **untouched**; the permanent controls were not
  retrained; frozen science untouched;
* `$215.6837` cumulative against the `$234.00` cap.
