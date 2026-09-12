# Recovery continuation attempt 4 — Stages 0 and 1 PASSED, Stage 2 OOM, $0.4112

**Verdict: the furthest any continuation has reached. Stage 0 and Stage 1 both
passed on hardware — the five leaves were imported and re-identified from the
staged bytes and the canonical control was measured — and Stage 2 died on the
first rung-1 probe with a CUDA OOM.** Fail-closed, torn down with provider
confirmation, no relaunch.

| | |
| --- | --- |
| authorization | `autoinit.recovery_continuation.2026-08-22T1454Z`, sha256 `874d54f6…` |
| grant | `logs/autoinit_recovery_continuation_attempt4_grant.json` |
| base commit | `38db4f2` |
| session commit | `ef4353c96fb5fb0693549a6484651850f2579e94` |
| harness digest | `162c09ed…`, 22 files, search excluded — unchanged from attempt 3 |
| bundle | `aad_autoinit_ef4353c9.bundle`, sha256 `d6ad9696…` |
| pod | `k1mgu38q0y6sei`, L40S $0.99/h, **24.92 min, $0.4112**, provider confirms gone |
| terminal | `DRIVER_EXITED:22`, `PHASE_A_FAILED` at stage 2 |

## What passed — and it is real scientific progress

```
15:28:30 MARKER:DRIVER_START
15:28:30 MARKER:STAGE_START:0
15:31:19 attested: interval 0.011695, floor 0.3000, plan 02be33b9a7a8…
15:31:19 MARKER:STAGE_PASSED:0
15:31:19 MARKER:STAGE_START:1
15:31:27 imported 5 stage-1 leaves, config 567d32789ba6…, re-identified from bytes
15:31:27 MARKER:STAGE1_IMPORTED:5
15:35:22 control measured on state_eval@v1 (74022 positions)
15:35:22 MARKER:STAGE_PASSED:1
```

* **Stage 0** attested the frozen protocol on hardware: equivalence interval
  `0.011695`, feasibility floor `0.3000`, science plan `02be33b9…`.
* **Stage 1 imported the five Attempt-12 leaves in the frozen selected order**
  — `cca699c9, 85bde4de, 158b96cf, 4e429f7e, 281a02c3` — each **re-identified
  from the bytes that arrived on the pod** and required to match the Stage-1
  artifact and shard digests. `config_hash 567d32789ba6…`,
  `target_spec_hash 09147a5c…`. **No search was run or reachable.**
* **The canonical control was measured once** on the frozen suite
  `state_eval@v1` (`suite_hash 6421fa4c…`, 74022 positions), producing
  `artifact_digest dc9500d3…`.
* Admission accepted the five leaves plus the measured control.

This is the first end-to-end demonstration that transport → staging → strict
byte re-identification → control measurement works on a paid pod.

## What failed

```
15:35:22 device handoff: 7.55 GiB is still ALLOCATED after the release, so
         something holds live tensors — this is a genuine retention, not
         allocator caching
15:35:22 MARKER:STAGE_PASSED:1
15:35:22 MARKER:STAGE_START:2
15:36:02 MARKER:STAGE_FAILED:2
15:36:02 STAGE 2 FAILED: autoinit.v1.phase_a.rung1.cca699c93f34.sa:
         training failed rc=1
  File "src/aadistill/training/train.py", line 404, in kd_forward_kl
    kl = (t.exp() * (t - s)).sum()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 298.00 MiB.
GPU 0 has a total capacity of 44.39 GiB of which 12.06 MiB is free.
Process 6324 has 8.06 GiB memory in use.
Including non-PyTorch memory, this process has 36.30 GiB memory in use.
```

### Two compounding defects, either of which alone was survivable

**1. The release freed nothing.** `release_to_subprocess(drop=[teacher, evaluator])`
reports `freed_allocated_bytes: 0`. Allocated was 8,110,229,504 B before **and
after**; only 4.64 GB of *reserved* cache was returned. The handoff diagnosed
this correctly and said so — `live_retention: true`, verdict *"a genuine
retention, not allocator caching"* — so something still holds references to those
tensors after `drop=` and `del`.

**2. The headroom gate's estimate of the trainer is ~14 GiB too low.**
`require_headroom` refuses unless `free_bytes >= RECOVERY_TRAINER_BYTES + margin`
= 22 + 2 = **24.00 GiB**. The handoff reported **36.32 GiB** free, so the gate
passed with 12.32 GiB of apparent slack. The probe then used **36.30 GiB** and
OOM'd asking for 298 MiB more — it actually needs **≳36.6 GiB**, not 22.

| | GiB |
| --- | ---: |
| free after handoff | 36.32 |
| gate demanded (22 + 2 margin) | 24.00 |
| trainer actually used before OOM | 36.30 |
| driver's retained allocation | 7.55 |
| free had the release worked | **43.87** |

Had the release worked, ~43.87 GiB would have been available and the probe would
have fit. Had the threshold matched the real trainer, the gate would have refused
at `$0.36` with a diagnosis instead of an OOM. Both are needed; neither is
sufficient.

### This is attempt 12's class, and the gate built to stop it passed

`require_headroom`'s own refusal message names attempt 12 — *"Attempt 12 spent
203.8 min on a successful search and then lost the probe here, because the parent
still held the card."* That gate exists precisely for this failure and did not
fire, because it was calibrated to a trainer footprint 14 GiB smaller than the
one that ran. Attempt 12 retained ~24.05 GiB from the search; attempt 4 retains
7.55 GiB from the control measurement. **The retention got smaller, the gate got
added, and the probe still did not fit** — because the trainer's real appetite
was never measured against it.

## Not attempted, and why

No repair on the live pod and no relaunch. The grant makes a failed
device-handoff or recovery gate a fail-closed stop. Both candidate repairs —
finding what retains the teacher/evaluator, and calibrating
`RECOVERY_TRAINER_BYTES` against a measured probe — touch
`autoinit_phase_a_driver.py` and/or `device_handoff.py`, which **are** in the
22-file harness, so unlike attempt 3's repair they would move the continuation
harness digest as well as the session commit.

## State

* pod deleted, **provider confirms gone**; watchdog ended `pod_gone` after 26
  ticks; independent poller stopped; provider returns zero pods; nothing billing;
* evidence collected through the supported path before teardown — the teardown
  gate ran as an **emergency** collection, recording that artifacts for
  `training_complete` and `evaluation_complete` are LOST because those stages
  never ran, which is correct rather than a defect;
* the five Attempt-12 leaves are **untouched**, canonical on the dev box and
  mirrored in the transport repo; the permanent controls were not retrained;
  frozen science untouched;
* `$214.3326` cumulative against the `$234.00` cap.
