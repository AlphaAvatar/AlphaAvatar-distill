# Phase C3 — plans

Frozen, pre-result documents. Each was written before the result it governs
existed, and none of them authorizes spend.

| file | what it owns |
| --- | --- |
| `c3_preregistration.json` | **CANONICAL.** The three arms, the nine probes, the three fresh seeds and the bootstrap seed, the claim boundary, the contrast hierarchy, the decision rule, the guardrails and the interpretation matrix. Hash-bound by `preregistration_sha256` over its own content. |
| `c3_pricing_9probe.json` | **CANONICAL C3 pricing.** Nine probes, every component measured. Carries the fit verdict and the two shortfalls as they stood when C3 was priced. |
| `c3_live_pricing.json` | The live re-price taken immediately before attempt75's authorization, on the device it used. |
| `c3_pricing.json` | **SUPERSEDED.** The six-probe pricing, obsolete once the design became three arms. Kept because it is what the six-probe fit decision was made from, not because it is current. |
| `a3_design.json` | **CANONICAL A3 design.** ONE end-to-end experiment: initialization → recovery training → evaluation → aggregation → closeout. The structural/runtime diagnostics are collected inside it and gate nothing; a differing A-bsz3 artifact digest is a FINDING. Six integrity stops, seven explicit non-stops. Derived by `scripts/autoinit/write_a3_design.py`; hash-bound by `design_sha256`. |
| `a3_pricing.json` | **CANONICAL A3 pricing.** The complete chain at a live rate, with a derived container provision and every applicable limit checked — including the package total. Derived by `scripts/experiments/phase_c3/a3_pricing.py`. |
| `a_bsz3_adoption.json` | **SUPERSEDED**, 2026-10-01. The cost-ordered step-1/step-2 split. It required an intermediate approval and could end on a diagnostic. |
| `a_bsz3_pricing.json` | **SUPERSEDED**, 2026-10-01. Three session shapes; its producer was deleted. |
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

**A3 is DESIGNED and FUNDED.** One chain, `$5.8680` expected and `$8.2525`
hard at a live `$1.09/h` L40S, on a derived 60 GB disk. The 2026-10-01
amendment raised the formal allowance to `$65.6523` and the package total to
`$75.6523` — funding that ceiling plus one pre-science restart, `$9.7031` on
the formal book, and nothing more. All four limits pass, the package total
among them for the first time.

Executed-run evidence belongs under `../runs/attemptN/`, never here.
