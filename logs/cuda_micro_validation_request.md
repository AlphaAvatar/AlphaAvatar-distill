# CUDA micro-validation — request, NOT an authorization

**Status: REQUESTED. Nothing is created, nothing is running, nothing is billing.**
This document stops before provider creation, as the merge review directs. It is
not a grant, not an authorization, and it does not itself permit a launch.

## What it is for

`attention.activation_importance_v1` has **never executed on a GPU**. Every `$0`
regression runs it on CPU, where the host statistics snapshot and the host
weights are trivially co-located; no C1 attempt before 9 reached stage F, and
attempt 9 died there:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

The repair is in the tree and is **logical / CPU-structural evidence only**. The
defect was a cross-device placement, which a single-device machine cannot
observe — so the repair cannot be validated on the dev box by construction, and
saying it "passes on CPU" would be the same substitution that let attempt 9
reach a paid pod in the first place.

This run answers exactly one question: **does the repaired treatment path
execute correctly on a real CUDA device?** It trains nothing, evaluates nothing,
compares nothing and decides nothing.

## Hardware requirement

| | |
| --- | --- |
| minimum capability | **CUDA compute capability ≥ 8.0** (Ampere or newer) |
| why that floor | the config runs `bfloat16`, to match the numerics C1 would run. bf16 needs cc ≥ 8.0; a Turing card (T4, cc 7.5) cannot satisfy it |
| VRAM required | **< 2 GiB**, essentially all of it CUDA context and the torch runtime |
| why so little | the fixture is **720,896 parameters — 1.44 MB in bf16**, built from config, not downloaded. The attention statistics tensor is 0.131 MB. Calibration is 4 synthetic items of 64 tokens |
| GPU count | **one** |
| **not** L40S | an L40S at `$1.09/h` is ~3 orders of magnitude over-specified for a 1.44 MB model. Any 8–16 GiB Ampere-or-newer card (RTX A4000, RTX 3090, A10) meets the requirement, and the review explicitly says not to assume L40S when a cheaper GPU will do |
| no teacher weights | this needs **no** checkpoint, no tokenizer, no `~8 GB` teacher fetch, and no second venv — unlike every C1 session setup to date |

## Exact command

Branch `migration/initialization-milestone-a`, at the committed SHA recorded in
the launch record (the entry point and its config are committed):

```bash
PYTHONPATH=src:scripts python scripts/validation/cuda_engineering_check.py \
    --config configs/validation/cuda_engineering.json \
    --run-id <run-id>
```

Exit codes: `0` pass, `1` a real failure, `3` **NOT RUN** (no CUDA) — and on
`3` **no report artifact is written at all**, so an absent file can never later
be mistaken for a pass.

## What it will prove, and what it cannot

It keeps the per-operator matrix (4 operators × 2 geometries) and adds the
end-to-end case that the matrix cannot reach, per suffix geometry:

* a real prefix materialized, the parent obtained and **verified**;
* `materialize_fixed_path_suffix` invoked on the full frozen path;
* `attention.activation_importance_v1` executed — the treatment operator, not
  `attention.weight_proxy_v0`;
* the original suffix index and checkpoint number retained;
* the prefix proven not to execute again, **from the filesystem**;
* the treatment execution record written through `RunLayout` and read back.

Six device facts, **observed rather than assumed**:

1. the collector's state is the intended **host** snapshot;
2. exactly **one** `stats_to` working copy reaches the model device;
3. attention statistics and `o_proj.weight` are **co-located** at every layer;
4. the score vector is **allocated on the operand device**;
5. the small returned score vector is **host-resident**;
6. the treatment suffix completes and its record validates.

An unreached observation site reports `observed: false` and `holds: false`. A
check nobody ran is not a check that passed.

**It proves nothing about efficacy.** No probe is trained, none evaluated, no
endpoint computed, no decision made. It does not touch the Attempt-9 checkpoint,
the formal C1 seeds, the confirmation battery, or Stage G/H/I.

## Budget

Spend is `$267.8598` of the unchanged `$283.7600` cap, leaving `$15.9002`.

| | |
| --- | --- |
| engineering soft stop | **20 minutes** of GPU time after the environment is ready. The check itself runs in seconds; anything longer means the environment is wrong, not the code |
| absolute cost ceiling | **`$1.00`**, inclusive of setup and teardown |
| teardown reserve | **`$0.25`** held inside that ceiling |
| worst case | `$268.8598` of `$283.7600` — leaving `$14.9002`, which still covers a full `$15.1475` C1 attempt **only if** the ceiling is not fully consumed. Sequence this accordingly, or accept that it does not |
| retry policy | **none.** One resource, no retry, no replacement pod. A failure is a finding to be diagnosed at `$0`, not repeated |

Setup time is the risk, not runtime: the same script, image and GPU class have
taken 5, 8.5 and 150+ minutes across this project's history, so the warm-image
number must not be budgeted as expected. The soft stop exists for that.

## Setup method

Minimal, and deliberately **not** the C1 session setup:

* a stock PyTorch CUDA image, so no torch build and no second venv;
* the repository at the committed SHA;
* `transformers` only — no teacher weights, no datasets, no HF cache warm-up,
  no vLLM;
* run the command above; collect the artifacts; delete the pod.

## Evidence paths

Written under the config's declared run root, `artifacts/validation/`, through
`RunLayout` with its declared roles:

| role | path |
| --- | --- |
| report | `<run>/report.json` |
| environment | `<run>/environment.json` |
| suffix evidence | `<run>/suffix_evidence.json` |
| per-operator workdirs | `<run>/operators/<geometry>/<impl_id>/` |
| suffix workdirs | `<run>/suffix/suffix_<geometry>/` |

To be copied back into `logs/` on completion, alongside the launcher log and the
teardown confirmation.

## What is still owed before this could run

A maintainer decision. This document is a request; it confers nothing, and the
merge review states that it does not itself issue a paid-resource
authorization.
