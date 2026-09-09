# CUDA stage-F engineering validation — request v1

**Status: REQUESTED, NOT AUTHORIZED.** Nothing is created, nothing is running,
nothing is billing. This document stops before provider creation. It is not a
grant, and it does not itself permit a launch.

**Formal science: none.** No probe is trained, none evaluated, no endpoint
computed, no decision made. It does not touch the Attempt-9 checkpoint, the
formal C1 seeds, the confirmation battery, or Stage G/H/I. **No Attempt-10
grant, authorization, bundle or formal endpoint exists.**

## Why it is owed

`attention.activation_importance_v1` has **never executed on a GPU**. Attempt 9
died inside it:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

The repair is in the tree and is **logical / CPU-structural evidence only**. The
defect is a cross-device placement, which a single-device machine cannot observe
— so no amount of CPU testing can validate it, and no further CPU device
simulator would change that. This run answers one question: **does the repaired
treatment path execute correctly on a real CUDA device?**

## Resource

| | |
| --- | --- |
| resources | **1** |
| retry / replacement | **none** — a failure is diagnosed at `$0`, not repeated |
| minimum capability | **CUDA compute capability ≥ 8.0, BF16** |
| minimum free VRAM | **2 GiB** |
| formal science | **none** |

The fixture is **720,896 parameters — 1.44 MB in bf16**, built from config. The
attention statistics tensor is 0.131 MB. No teacher weights, no checkpoint, no
tokenizer, no second venv. The 2 GiB floor is CUDA context and the torch
runtime, not the model.

`cc ≥ 8.0` is what BF16 requires, and BF16 is what the config runs because it is
what C1 would run. A Turing card (T4, cc 7.5) cannot satisfy it.

## Cost, derived from the live secure price

Quoted from the provider at request time, `securePrice` — never
`communityPrice`, which has under-reported two runs:

| GPU | securePrice | VRAM | at `$0.25` soft cap | at `$0.40` absolute |
| --- | --- | --- | --- | --- |
| RTX 2000 Ada | `$0.24/h` | 16 GB | 62.5 min | 100.0 min |
| RTX A4000 | `$0.25/h` | 16 GB | 60.0 min | 96.0 min |
| RTX A4500 | `$0.25/h` | 20 GB | 60.0 min | 96.0 min |
| *(L40S, for contrast)* | *`$1.09/h`* | *48 GB* | *13.8 min* | *22.0 min* |

| | |
| --- | --- |
| engineering soft cap | **`$0.25`** |
| absolute provider cap | **`$0.40`**, inclusive of teardown |
| teardown reserve | **`$0.15`** held inside the `$0.40` |
| time allowance | **derived, not fixed** — `$0.25 ÷ securePrice`, which is ~60 min at `$0.25/h`. It is not a 20-minute rule: at a different price the same money buys different time, and the cap is the constraint |

**Budget arithmetic.** Spend is `$267.8598` of the unchanged `$283.7600` cap,
leaving `$15.9002`. Worst case here is `$268.2598`, leaving **`$15.5002`** —
which still exceeds the formal `$15.1475` Attempt-10 ceiling by `$0.3527`.

The previous `$1.00` request was correctly refused: fully consumed it would have
left `$14.9002`, less than the ceiling, so paying for engineering validation
would have priced the formal attempt out of existence.

An L40S is not requested. At `$1.09/h` it is 4.4× the price for a 1.44 MB model,
and the review directs against assuming it when a cheaper GPU meets the
requirement.

## Exact command

Branch `migration/initialization-milestone-a`, at the committed SHA recorded in
`bound_sha.txt` beside this file:

```bash
PYTHONPATH=src:scripts python scripts/validation/cuda_engineering_check.py \
    --config configs/validation/cuda_engineering.json \
    --run-id <run-id>
```

Exit codes: `0` pass, `1` a real failure, `3` **NOT RUN** (no CUDA) — and on `3`
**no report artifact is written at all**, so an absent file can never later be
mistaken for a pass.

## What it proves

The per-operator matrix (4 operators × 2 geometries), plus the end-to-end case
the matrix cannot reach, per suffix geometry:

* a real prefix materialized, the parent obtained and **verified**;
* `materialize_fixed_path_suffix` invoked on the full frozen path;
* `attention.activation_importance_v1` executed — the treatment operator, not
  `attention.weight_proxy_v0`;
* the original suffix index and checkpoint number retained;
* the prefix proven not to execute again, **from the filesystem**;
* the treatment execution record written through `RunLayout` and read back.

Six device facts, **observed rather than assumed**: the collector's host
snapshot; exactly one `stats_to` working copy on the compute device; statistics
and `o_proj.weight` co-located at every layer; the score vector allocated on the
operand device; the returned vector host-resident; and the suffix completing
with a record that validates. An unreached observation site reports
`observed: false` and `holds: false` — a check nobody ran is not a check that
passed.

## Evidence

Written through `RunLayout` under **one engineering run root**,
`artifacts/validation/cuda_engineering/<run-id>/`:

| role | path |
| --- | --- |
| report | `report.json` |
| environment | `environment.json` |
| suffix evidence | `suffix_evidence.json` |
| per-operator workdirs | `operators/<geometry>/<impl_id>/` |
| suffix workdirs | `suffix/suffix_<geometry>/` |

The outcome is copied back **into this validation directory**
(`logs/validations/cuda-stage-f/v1/`), not to a new flat file at `logs/`.

## Setup

A stock PyTorch CUDA image, the repository at the bound SHA, `transformers`
only. No teacher fetch, no HF cache warm-up, no vLLM, no second venv.

Setup time is the risk, not runtime: the same script, image and GPU class have
taken 5, 8.5 and 150+ minutes across this project's history, so the warm-image
number must not be budgeted as expected. The soft cap exists for that, and the
absolute cap bounds it.

## What is still owed

A maintainer decision. This document confers nothing.
