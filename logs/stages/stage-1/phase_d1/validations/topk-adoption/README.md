# D-series Top-K adoption validation

Engineering evidence only. **No D-series science runs here**: no recovery
training, no behavioural screening, no confirmation, no formal search, no
promotion, no GO/NO-GO. No D-series behavioural prompt is read.

Authorized by the maintainer decision of **2026-10-04**, which replaced the
D-series full-vocabulary KL contract with `reference_topk_tail_v1` at **K = 200**
before any formal D1/D2/D3. The completed
[full-vocabulary GPU qualification](../gpu-qualification/)
is the **baseline** this is measured against, and its repricing is explicitly not
the final D1 price.

## What it must answer

| | question | why a GPU |
| --- | --- | --- |
| A | how much probability mass does the reference's own Top-200 hold, across the distribution and not just on average? | the real teacher on the real mixture |
| B | how far does the K+1 KL sit from the full-vocabulary KL — absolutely, relatively, in rank, and at worst? | both reductions from real logits |
| C | does the coarser partition move a DEPTH decision? | real bf16 CUDA scores over 36 candidates |
| D | does it move the state evaluation's equal-domain KL, worst domain, critical-token KL or candidate ordering? | the real suite at the real vocabulary |
| E | what does it save: sketch bytes, recomputes, wall clock, peak memory? | measurable only on the device |

**One set of forwards, two reductions.** Paying for a second set of 260 model
forwards to compare two reducers would buy nothing, so the DEPTH operator carries
an execution-only observer and the Top-K score is computed from the logits the
operator already produced.

**A moved decision is not a failure.** This is an authorized new protocol, so a
difference means the D-series protocol is observably different from the historical
full-vocabulary one — which is the thing being characterized.

**What C cannot say.** The greedy chooses by full vocabulary, so its removal order
is exact and the Top-K column is counterfactual from the first disagreeing round
onward. That is inherent to sharing forwards and the record states it.

## Layout

```text
v1/authorization.json   the ceiling and what it does and does not cover
v1/campaign.json        cumulative cost across every resource and subrun
v1/runs/<subrun>/       per-attempt evidence; retries stay here
v1/closeout.json        the verdict and the answers — written only when complete
```

Nothing here authorizes D1. `K = 200` is a maintainer-selected protocol
parameter: there is no sweep and it must never be tuned against D1/D2/D3 results.
