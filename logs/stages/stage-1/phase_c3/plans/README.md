# Phase C3 — plans

Frozen, pre-result documents for the formal C3 experiment. Nothing here is
written after a behavioural result exists.

| file | what it owns |
| --- | --- |
| `c3_preregistration.json` | **CANONICAL.** The three arms, the nine probes, the three fresh seeds and the bootstrap seed, the claim boundary, the contrast hierarchy, the decision rule, the guardrails and the interpretation matrix. Hash-bound by `preregistration_sha256` over its own content. |
| `c3_pricing_9probe.json` | **CANONICAL pricing.** Nine probes, every component measured. Carries the fit verdict and the two shortfalls. |
| `c3_pricing.json` | **SUPERSEDED.** The six-probe pricing, obsolete once the design became three arms. Kept because it is what the six-probe fit decision was made from, not because it is current. |

**Current state: NOT LAUNCHED.** The nine-probe hard ceiling does not fit the
project envelope; the shortfall is recorded in the pricing and in
`logs/state/current.md`. No provider resource was created for formal C3.

Historical vs current: everything in this directory is current. Executed-run
evidence belongs under `../runs/attemptN/`, and there are none.
