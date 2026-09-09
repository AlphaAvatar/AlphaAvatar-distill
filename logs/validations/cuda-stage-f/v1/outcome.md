# CUDA stage-F engineering validation — outcome

**EXECUTED 2026-09-09. INFRASTRUCTURE FAIL. No CUDA observation was made.**

| | |
| --- | --- |
| execution SHA | `355a1f7d7b0f09840dc8f4d84923e97a2039003f` |
| authorization | `authorization.json`, `a23416687d69209b…` |
| provider resource | `ij54bzvcyldm9j` — NVIDIA RTX 2000 Ada Generation, 16 GB |
| accepted / actual rate | `$0.24/h` quoted, `$0.24/h` confirmed |
| create attempts | **1** |
| watchdog launches | **1** |
| elapsed | 1.84 min |
| cost | **`$0.0073`** (estimate: elapsed × confirmed rate; the watchdog journal independently accrued `$0.0070`) |
| teardown | **provider-confirmed** — pod absent from inventory, journal records `pod_billing: false`, `TERMINATED` |
| verdict | **FAIL (setup)** |

## What happened

The spend contract held end to end. One bounded quote pass over nine candidates
chose the cheapest available; one create attempt was issued; the watchdog was
detached in the same second the pod id existed; the provisioned rate was
confirmed at the quote; work stopped at the first substantive exception; and
teardown was verified rather than assumed.

The **launcher's dependency step** failed, and both defects were mine:

1. `pip install ... 2>&1 | tail -5` makes the shell report **tail's** exit
   status, which is always `0`. The install had already been refused and the
   run recorded `pip_rc: 0`.
2. The image's interpreter is PEP 668 *externally managed*, so a plain
   `pip install` is refused by design. The refusal message names the flag it
   wants; the launcher did not pass it.

So the validation entry point started and exited `1` four seconds later on
`ModuleNotFoundError: No module named 'numpy'` — inside
`aadistill.data.extra_stream`, reached while registering operators.

## What this does NOT say

**Nothing about the stage-F repair.** No operator executed. No device placement
was observed. `attention.activation_importance_v1` has still never run on a GPU,
and the repair remains **logical / CPU-structural evidence only**.

It is not a C1 attempt, not a C1 result, and changes no formal C1 status:
replay MEASURED 2/2 PASS, formal treatment UNMEASURED, endpoint UNMEASURED,
Attempt 9 NO DECISION.

## What was done about it

Both defects are fixed in `scripts/validation/cuda_engineering_launch.py` and
pinned by regressions in `tests/validation/test_cuda_engineering_launch.py`,
including one that reads the dependency step's **code** (not its comments) and
fails if the pipe returns. An import probe now runs before the validation, so a
missing module costs seconds and is reported as a setup failure rather than a
validation failure.

**The fixed launcher has not been run on a GPU.** One resource was authorized
and it has been consumed. No replacement was created and none will be without a
new maintainer decision.

## Why a $0 rehearsal did not catch it

The payload probe extracted the shipped tar into a bare directory and ran the
check from it alone, which proved the *payload* was import-complete — on a
machine that already had numpy. What it could not test is the *pod's*
interpreter and its package policy. That gap is what the new import probe
closes, on the pod, before the money-spending step.
