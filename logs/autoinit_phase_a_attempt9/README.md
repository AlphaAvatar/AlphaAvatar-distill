# Phase-A attempt 9 — 2026-08-18, $0.34, Stage 0 PASSED, Stage 1 failed on device placement

**Stage 0 passed and attested. Stage 1 failed. Nothing was trained; no
checkpoint, probe or search leaf was produced. The pod was deleted by the
launcher and the provider confirmed it gone.**

| | |
| --- | --- |
| authorization | `autoinit.phase_a.2026-08-18T1512Z`, sha256 `0ab3747c7d5ed45abf6d3de8eba26fa752ba923476cf16f484f65ccc0210ac7d` |
| grant | [`../autoinit_phase_a_attempt9_grant.json`](../autoinit_phase_a_attempt9_grant.json), sha256 `7b62b5c516be725781f8c13878f5749520882518f8fa73c728281cec7f4f7894` |
| authorized base | `ab4138a7c5f6da19775072577fbfe54fa4fb2c47` |
| session commit | `9f3ff7f53e1863e8e7601f071cb13fa71c57b26a` |
| harness digest | `24d89b9ffe8e41f2712d2f105ca2076fd0552ccea7a089aeb5f2a844c39e1f42` — **identical to attempt 8**; the $0 fix touched only `tests/docs/` and `logs/` |
| bundle | `transfer/aad_autoinit_9f3ff7f5.bundle`, sha256 `c8d2478d8d911230eac5997703fd462f7c4ee606784c2d16309fde2104d484db` |
| pod | `grgiu5atibea3x`, 1×L40S @ $0.99/h, image `…cu1300-torch291-ubuntu2404@580.159.03` |
| lifetime | 15:14:11 → 15:34:37 UTC, 20.4 min |
| cost | **$0.34** (pod lifetime × $0.99/h; the watchdog's last tick read $0.3498 at 21.2 min, which counts until it noticed the pod was already gone) |
| outcome | `PHASE_A_FAILED`, `failed_stages: [1]` |

## The attempt-8 fix held

Setup reached `SETUP_DONE` in 7.4 minutes and the **blocking test gate passed** —
the same `tests/docs` suite that failed attempt 8 on two assertions with invalid
environment semantics. The driver detached (pid 6773, confirmed by
`descriptor_probe`) and Stage 0 began. That question is closed.

## Stage 0 — PASSED, and attested

```
evaluation_protocol_hash         250f72efbd43b86a475e8dda293b45f07ee61a4d858e147f4a5bd7681c32c2e4
comparable_identity              70a26e0b43df20e469385115b49683b50936919e02986088b76d3c30ffd87103
science_plan_hash                02be33b9a7a8e26bc8bfb75795351e8cdc9ffd441b47066cc81887cfc511b55c
generation_source_digest         a1b51736d6a6b1276322a04d44855d7b574011e165e77d6d23c3e31b3036f743
runtime  torch 2.11.0+cu128 · transformers 5.13.1 · python 3.12.3 · cuda 12.8 · sdpa
```

All three frozen identities matched under `generation_runtime_comparability@v2`.
This host drew driver `580.159.03` — the same as Stage 3 and as attempt 5 — so
as on attempt 5, this run does not by itself discriminate v2 from v1.

## Stage 1 — FAILED, with the traceback recovered

```
RuntimeError: Expected all tensors to be on the same device,
              but found at least two devices, cuda:0 and cpu!

  scripts/pod/autoinit_phase_a_driver.py:443   stage1
  scripts/autoinit/phase_a_search.py:169       run_phase_a_search
  src/aadistill/autoinit/search.py:506         run          -> _expand_one
  src/aadistill/autoinit/operators/base.py:278 execute      -> apply
  src/aadistill/autoinit/operators/composite.py:135         init_student
  src/aadistill/init/sandwich.py:195           stream_projection
  src/aadistill/init/project.py:60             avg += w * (m / m.trace())
```

Full text in [`stage1_traceback.log`](stage1_traceback.log). **This is the
evidence attempt 7 lost with its pod**; the session-architecture work that added
full tracebacks for unexpected in-process driver exceptions is what made the
cause readable without paying again.

### The defect, exactly

`src/aadistill/init/project.py:57`:

```python
avg = torch.zeros(d_teacher, d_teacher, dtype=torch.float64)   # no device=
for p, w in zip(points, weights):
    m = uncentered_moment(state, p)                            # follows `state`
    avg += w * (m / m.trace())                                 # cpu += cuda:0
```

The accumulator is allocated with a dtype and **no device**, so it lands on CPU.
`uncentered_moment` returns a tensor on whichever device `state` holds — CPU on
the dev box, `cuda:0` on a pod. The `+=` then mixes them.

**A CPU rehearsal cannot see this.** On a CPU-only machine `state` is on CPU,
both operands agree, and the arithmetic is correct — which is why the full
suite, the pod simulator and every $0 gate pass over this line.

### It is the third of its kind in this stage

| attempt | cost | Stage-1 device defect |
| --- | ---: | --- |
| 6 | $0.3552 | `_validate` probe built on `config.device`, child on the host |
| 7 | $0.3955 | `ActivationStatsCollector` accumulators unplaced |
| 9 | $0.3400 | `stream_projection`'s `avg` accumulator unplaced |

`autoinit.stage1_device_contract@v1` was written after 6 and 7 and closed both.
It did not cover a freshly-allocated accumulator inside a projection helper two
call levels below the operator. The pattern is now three-for-three: **every one
is a tensor allocated without a device in a code path only a GPU executes**, and
each has been found by paying for it.

## Artifacts collected

Manifest `rc=0`, 10 files across 5 classes, `final_required: 10`,
`final_streams_quiescent: True`; teardown gate `allowed=True`. Retrieved here:
[`launcher.out`](launcher.out), [`session.json`](session.json),
[`phase_a_evidence.json`](phase_a_evidence.json),
[`stage1_traceback.log`](stage1_traceback.log),
[`attested_evaluation_protocol.json`](attested_evaluation_protocol.json),
[`engine_probe.json`](engine_probe.json), [`watchdog.jsonl`](watchdog.jsonl),
[`poll.log`](poll.log).

No checkpoint, probe result or search leaf exists to collect — Stage 1 died
inside the first operator expansion. The permanent controls are inputs to this
session and were untouched.

## Disposition

Failed closed as instructed: evidence preserved, pod terminated with provider
confirmation, **stopped for review**. No repair was attempted on the live pod,
no retry was launched, and **no attempt 10 is authorized**. The grant covered one
launch and is spent.
