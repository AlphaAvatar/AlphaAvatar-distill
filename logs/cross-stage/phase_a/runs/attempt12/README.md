# Phase-A attempt 12 — 2026-08-20/21, $3.7872, **Stage 1 PASSED and its result survived**

Stage 0 and Stage 1 passed. Stage 2 failed on a **CUDA OOM**, not on the
tokenizer — that defect is closed. **All five selected leaves reached the dev-box
store and were digest-verified before teardown**, which is the thing attempt 11
could not do. **Authorization CONSUMED.**

| | |
| --- | --- |
| authorization | `autoinit.phase_a.2026-08-20T1856Z`, sha256 `459bf9f1df61ef98e920529f48e6ad2ccae7c5babaefae7db861f421c3a3edab` |
| grant | [`../autoinit_phase_a_attempt12_grant.json`](../autoinit_phase_a_attempt12_grant.json), sha256 `69e8150ea9d4d50d…` |
| authorized base | `6075a6d97b0a0d516d5eddb23675341f8c4af3b8` |
| session commit | `0c7cb42a9a177241718fb6c43388b5aea1e7bf43` (authorization-only, one path) |
| harness digest | `b9ff56681e84267a2cc71e10e958beea48699259f39ae5e8dc02b56a35c50215` |
| bundle | `aad_autoinit_0c7cb42a.bundle`, sha256 `b52d271679bebd8f…` |
| pod | `o7hj2r3z2w8xat`, 1×L40S @ $0.99/h, first draw |
| lifetime | 19:19:48 → 23:09:20 UTC, **229.53 min** |
| cost | **$3.7872** of a $23.0484 ceiling |
| terminal | `PHASE_A_FAILED` at stage 2, no launcher error |

## Stage by stage

| stage | result | elapsed |
| --- | --- | ---: |
| setup | `SETUP_RC=0`, first draw | 6.4 min |
| **0 — attestation** | **PASSED** | 1.9 min |
| **1 — beam search** | **PASSED**, `SEARCH_DONE:5` | **203.8 min** |
| **2 — rung-1 probes** | **FAILED, CUDA OOM** | 0.15 min |
| 3, 4, 5 | not reached | — |

## The closure worked: five leaves off-pod, verified

```
[23:01:36]  stage-1 leaf cca699c93f34 -> …/phase_a/cca699c93f34…  rc=0 digest=MATCHED
[23:03:29]  stage-1 leaf 85bde4ded2c3 -> …                        rc=0 digest=MATCHED
[23:05:20]  stage-1 leaf 158b96cf651f -> …                        rc=0 digest=MATCHED
[23:07:14]  stage-1 leaf 4e429f7ed722 -> …                        rc=0 digest=MATCHED
[23:09:08]  stage-1 leaf 281a02c3ac18 -> …                        rc=0 digest=MATCHED
[23:09:08]  teardown gate: allowed=True failed=training_complete
[23:09:20]  pod deleted — 229.5 min, $3.79; provider confirms gone: True
```

`required_products_secured: {ok: true, why: "all 5 stage-1 selected leaves
verified off-pod"}`. The transfer ran **after a Stage-2 failure**, which is
exactly the case that returned early before — 5.55 GiB, ~2 minutes per leaf, by
the product `scp` path rather than the archive.

**Re-verified independently afterwards**, not merely trusted from the launcher's
log: each checkpoint re-identified from local bytes, `artifact_digest` and
`single_shard_sha256` both matching the Stage-1 record, 1.110 GiB each,
`tokenizer_sha256: None` — still weight-only, so the identity the search metrics
hang on is untouched. **5/5.**

Attempt 11 produced these same five leaves and destroyed all of them. The
difference is not luck: the durability report is fetched before products, the
transfer is not gated on Stage 2, and the teardown gate refuses while any owed
leaf is still only on the pod.

## The search is deterministic — two paid runs, byte-identical

| | attempt 11 | attempt 12 |
| --- | --- | --- |
| `config_hash` | `567d32789ba6dcef…` | **identical** |
| states / complete leaves / pruned | 43 / 7 / 18 | **43 / 7** |
| selected state ids | five | **identical, in order** |
| first depth invocation, ROUND 7 | layer 21, 0.625600, runner-up 17, margin 1.529e-03 | **identical** |
| search wall time | 180.3 min | 203.8 min |

Different pods, different hosts, three days apart. Every scientific output
matches; only wall time differs, which is host speed. This is direct evidence
that the frozen search is deterministic in the way the science plan claims —
something no `$0` test could establish, and it arrives as a free by-product of
having had to run twice.

The reference cache fell back again (~6.8 eval/min against the standalone
12.07), consistent with the recorded finding that the cached path is not
reachable inside the search at this teacher size.

## The new failure: two processes, one GPU

```
torch.OutOfMemoryError: Tried to allocate 3.58 GiB.
GPU 0 has a total capacity of 44.39 GiB of which 2.36 GiB is free.
Process 6820 has 24.05 GiB memory in use.
  … this process has 17.97 GiB memory in use.
  Of the allocated memory 17.46 GiB is allocated by PyTorch …
```

at `train.py:380`, `masked_ce` → `F.cross_entropy`.

The driver runs the beam search **in-process** — deliberately, because the rungs
need live `InitializationState` objects — so at the end of Stage 1 it still holds
the teacher and candidate state on the device. Stage 2 then spawns
`train_stage3.py` as a **subprocess**, which needs its own ~18 GiB while ~24 GiB
is still resident in the parent. 44.39 GiB does not fit both.

**This is structural, not a race.** It will recur on every attempt on this
hardware, at the same point, six seconds after Stage 1 succeeds. It is also the
same shape as the reference-cache finding: the search's residency is larger than
any standalone measurement of a stage suggests, and stage boundaries that look
sequential in the code are not sequential in device memory.

Note what it is **not**: not the tokenizer (checked before training, and training
reached a loss computation), not a leak, not the leaves.

**No fix is applied.** The run stopped for review, per the grant.

## What worked

The whole chain, and every gate that should have fired. Setup passed on the first
draw. Stage 0 attested. The new pre-provider capacity gate ran before the
provider was contacted and logged `20.3 GiB free for 11.6 GiB of leaves and
working room`. `manifest rc=0`, 14 files across 7 classes. The teardown gate
allowed on `training_complete` failing — after `required_products_secured`
passed — and the provider confirmed the pod gone at 229.5 min against a
1397-minute bound.

## Disposition

Recorded as **consumed; Stage-1 PASSED with its result preserved; Stage-2
fail-closed stop**. Grant and authorization spent and not reusable. Cumulative
spend $209.6842 → **$213.4714** of $234.00, leaving **$20.5286** — less than the
$23.0484 per-launch ceiling, so a further full attempt does not fit without
another cap decision.

**No Attempt 13 is prepared, granted, funded or implied.** The OOM is a diagnosis
for the maintainer, not a licence to retry.
