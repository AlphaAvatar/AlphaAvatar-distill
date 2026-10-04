# D1 engineering GPU qualification

Engineering evidence only. **No D1 science runs here**: no recovery training, no
behavioural screening, no confirmation, no formal search, no promotion, no
GO/NO-GO. No D-series behavioural prompt is read.

Authorized by the maintainer decision of 2026-10-03, which raised the GPU
engineering allowance `10.0000 -> 20.0000` and the package total
`86.6523 -> 96.6523` and changed nothing else — not the formal allowance, not the
project cap, not the per-session envelope, and not
`funds_formal_sessions_of`, which still does not contain `phase_d1`.

## What it must answer

| | question | why a GPU |
| --- | --- | --- |
| A | does the incumbent fixed path reconstruct the frozen incumbent artifact identity on real CUDA? | the identity is produced by CUDA arithmetic |
| B | does the target-aware `supervised_target_v1` path execute correctly at `bsz=3` under CUDA/bf16, through the real evaluator? | bf16 kernels and the batching machinery |
| C | does bf16/CUDA move any discrete operator or selection decision? | a near-tie over float scores can flip |
| D | what is the real state-eval peak memory at the real vocabulary and batch plan? | memory is a property of the device |
| E | what does a representative expansion actually cost? | to reprice D1 from measurement, not from the unbatched table |

A CPU-vs-GPU selection difference is **evidence, not automatically a failure**.
What is unacceptable is an unbound numerical identity, unexplained
same-environment nondeterminism, a resume/materialization identity disagreement,
or quietly changing the formal semantics so CUDA matches CPU.

## Layout

```text
v1/authorization.json   the ceiling and what it does and does not cover
v1/campaign.json        cumulative cost across every resource and subrun
v1/runs/<subrun>/       per-attempt evidence; retries stay here
```

One canonical owner per fact: the authorization owns the ceiling, the campaign
owns the costs, each run owns its own measurements.
