# CUDA stage-F engineering validation — outcome

**CUDA ENGINEERING VALIDATION PASS**, 2026-09-10, at subrun 3 of 3.

| | |
| --- | --- |
| execution SHA (the PASS) | `7027a8f4c0a7684c892b483a2193025cc26c1b58` |
| authorization | `authorization.json` + `authorization_amendment_1.json` |
| device | NVIDIA RTX 2000 Ada Generation — cc **8.9**, bf16 supported, 15.48 GiB free |
| runtime | torch `2.9.1+cu130`, CUDA runtime 13.0, numpy 2.5.3, transformers 5.17.0, safetensors 0.8.0 |
| resources | **3 total, never more than 1 at a time**; 3 creates, one per subrun, no provider retries |
| campaign cost | **`$0.0400`** of the `$0.4000` ceiling and the `$0.2500` soft cap |
| teardown | **all three provider-confirmed non-billing**; final inventory zero pods |

## The subruns

| subrun | pod | cost | class | outcome |
| --- | --- | --- | --- | --- |
| `cuda_stage_f_20260910` | `ij54bzvcyldm9j` | $0.0073 | setup | FAIL |
| `cuda_stage_f_20260910_s2` | `zoz95844krv2ze` | $0.0145 | harness criterion | FAIL |
| `cuda_stage_f_20260910_s3` | `8tbsixglzz64ox` | $0.0182 | — | **PASS** |

**Subrun 1 — setup.** A shell pipeline reported `tail`'s exit status, hiding a
PEP 668 refusal; the check exited on `ModuleNotFoundError: numpy`. Repaired by
one interpreter for install, probe and validation (`--system-site-packages`
venv, so the image's cu130 torch is inherited rather than resolved afresh),
pip's own return code through a marker, and a readiness probe that requires
CUDA, the capability floor, the dtype and a real device matmul.

**Subrun 2 — a wrong acceptance criterion.** Setup passed and both suffix
geometries executed the treatment operator successfully, but the per-operator
matrix failed 8 of 8 with every case recording `applied: True` and no error. The
criterion demanded the operator's CHILD be on the requested device;
`initialization/device.py` documents that `ChildBuilder` deliberately does not
place it. On CPU the check passed trivially, so it had never fired. Corrected to
ask what the matrix is for: did the operator run against a parent **on** the
device, read from the parent's weights, and is the child host-resident as the
builder contract states. A host parent still fails.

Neither failure was a defect in the code under test.

## What subrun 3 observed

Matrix **8/8**. Suffix case **both geometries**, `device_proofs_are_meaningful:
true`.

Per geometry (`suffix_narrow`, `suffix_mid`): `attention.activation_importance_v1`
executed through the real `materialize_fixed_path_suffix` from a genuinely gated
parent; suffix index **3** retained and `03_attention` kept its number; **no
prefix checkpoint written**; the output stayed bound to the full frozen path;
the treatment record wrote and read back, is not a replay, and carries
`n_pinned: 0`.

Five placements **observed**, not assumed:

| proof | evidence |
| --- | --- |
| collector state is the host snapshot | `['cpu', 'cpu']` |
| exactly one `stats_to` working copy on the device | `n_calls: 1`, `cuda:0` |
| statistics and `o_proj.weight` co-located | `cuda:0` / `cuda:0` at every layer |
| score vector allocated on the operand device | `cuda:0` |
| returned score vector host-resident | `cpu` |

That is the Attempt-9 failure class — statistics on the host meeting weights on
`cuda:0` — exercised on real hardware and passing.

## What this does and does not mean

**The repaired and migrated treatment suffix has now successfully completed
real-CUDA engineering validation.** The operator itself had entered GPU
execution before, in formal Attempt 9, and failed there; that is why the repair
exists.

It is **engineering evidence only**: not a C1 treatment result, not an endpoint
measurement, not a decision. No formal seeds, no confirmation battery, no
Attempt-9 checkpoint, no recovery training. Formal C1 is unchanged — replay
MEASURED 2/2 PASS, formal treatment UNMEASURED, endpoint UNMEASURED, Attempt 9
NO DECISION.

A pod-side readiness probe is **billed setup after provider creation**, not a
zero-dollar step. It costs seconds rather than the whole attempt.
