# Phase C3 — plans

Frozen, pre-result documents. Each was written before the result it governs
existed, and none of them authorizes spend.

| file | what it owns |
| --- | --- |
| `c3_preregistration.json` | **CANONICAL.** The three arms, the nine probes, the three fresh seeds and the bootstrap seed, the claim boundary, the contrast hierarchy, the decision rule, the guardrails and the interpretation matrix. Hash-bound by `preregistration_sha256` over its own content. |
| `c3_pricing_9probe.json` | **CANONICAL C3 pricing.** Nine probes, every component measured. Carries the fit verdict and the two shortfalls as they stood when C3 was priced. |
| `c3_live_pricing.json` | The live re-price taken immediately before attempt75's authorization, on the device it used. |
| `c3_pricing.json` | **SUPERSEDED.** The six-probe pricing, obsolete once the design became three arms. Kept because it is what the six-probe fit decision was made from, not because it is current. |
| `a_bsz3_adoption.json` | **CANONICAL A-bsz3 design.** The shortened engineering adoption study: one same-device structural/runtime comparison, then at most three newly trained treatment probes against attempt75's existing A controls, with a derived fail-fast stop. Derived by `scripts/autoinit/write_a_bsz3_adoption.py`; hash-bound by `design_sha256`. |
| `a_bsz3_pricing.json` | **CANONICAL A-bsz3 pricing.** Three session shapes priced from measured components, each judged against the book that actually binds it. Derived by `scripts/experiments/phase_c3/a_bsz3_pricing.py`. |
| `a_bsz3_noninferiority.json` | **WITHDRAWN**, 2026-10-01, before any A-bsz3 result existed. The 16-probe design. Kept for the two findings that survive it — why non-inferiority needs more data than superiority, and why a margin below the instrument's half-width is not a criterion. |

## Current state

**C3 itself is COMPLETE, and its verdict is `NO_GO`** on the primary
operator-isolation contrast. attempt75 trained, preserved and scored all nine
probes; its stage-I aggregation failed on the pod and the verdict was computed
off pod at `$0` from the immutable evidence. The decision artifact lives in
[`../analyses/attempt75_stage_i/`](../analyses/attempt75_stage_i/), and the
run's own evidence under [`../runs/attempt75/`](../runs/attempt75/).

*This section used to say `NOT LAUNCHED` and `there are none` of the runs. Both
were true when C3 was priced and neither has been true since 2026-09-28.*

**A-bsz3 is DESIGNED and NOT FUNDED.** Step 1 fits the GPU engineering
allowance; step 2 trains probes and consumes the frozen confirmation battery,
so its book is the formal allowance, which is overspent. The pricing record
carries both figures and the shortfalls.

Executed-run evidence belongs under `../runs/attemptN/`, never here.
