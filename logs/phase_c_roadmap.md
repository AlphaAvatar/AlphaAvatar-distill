# Phase C — ATTENTION operator R&D · roadmap

**Status: C0 COMPLETE / APPROVED / FROZEN · C1 NOT STARTED · C2 NOT STARTED.
NOT PRICED · NOT AUTHORIZED · NO COMPUTE.**

The Phase-C0 protocol is frozen in
[`phase_c0_preregistration.json`](phase_c0_preregistration.json), with its sizing
evidence in [`phase_c0_sizing_evidence.json`](phase_c0_sizing_evidence.json).
Those two files, not this page, are the record of what C1 must do. This page is
the surrounding structure and rationale.

No Phase-C experiment has run. Nothing here is authorization for compute.

Formal Stage-2/Stage-3 recovery training remains **deferred** until the
operator-development programme is complete. The Phase-B winner is **not**
authorization for it.

---

## Why ATTENTION, and why first

> **Interpretation corrected 2026-09-01.** An earlier version of this section
> argued from the fact that competitive Phase-B paths "selected `calib.none@v1`"
> for ATTENTION. **That is not evidence and must not be repeated.**
> `attention.weight_proxy_v0` declares `CalibrationNeed.NONE`
> (`src/aadistill/autoinit/operators/attention.py`), and
> `BeamSearch._candidate_expansions` offers such an implementation **exactly
> once, against the `NO_CALIBRATION` sentinel**, however many profiles are active
> (`src/aadistill/autoinit/search.py`, `src/aadistill/autoinit/calibration.py`).
> The search never had a second option to reject. See
> [`decisions.md`](decisions.md) 2026-09-01.

The correct motivation:

> Phase A/B exercised only `attention.weight_proxy_v0`. That implementation
> declares `CalibrationNeed.NONE`, so its no-calibration assignment was
> mechanical rather than a choice between competing calibration profiles.
>
> No activation-based, forward-logit, or causal ATTENTION formulation competed
> against it.
>
> Therefore Phase A/B contain **no operator-level evidence** that such ATTENTION
> formulations are inferior. Phase C1 creates the missing fixed-path ATTENTION
> comparison.

The operator's own docstring records the gap it was built with: activation-based
head importance "would need attention hooks Stage 0 never cached", and a future
`attention.activation_importance_v1` or `attention.causal_kl_v1` was anticipated
as a separate registered id.

Neither phase can attribute anything to a single operator, because neither varied
one operator with the rest held fixed. That is the gap C1 closes.

---

## Phase C0 — protocol and power design · **COMPLETE / FROZEN 2026-09-01**

**Output:** [`phase_c0_preregistration.json`](phase_c0_preregistration.json)
(protocol, `aadistill.autoinit.phase_c0_protocol/v1`) and
[`phase_c0_sizing_evidence.json`](phase_c0_sizing_evidence.json) (the power
evidence behind the battery size). Both are binding on C1.

What C0 settled:

| question | answer |
| --- | --- |
| primary endpoint | `correct_overall` over 850 scorable prompts. `usable_rollout` was **not** promoted to a ranking metric |
| estimand | prompt-mean of the seed-mean paired difference, over 3 **fixed** fresh seeds |
| inference | stratified **prompt**-cluster bootstrap; the CI is conditional on those three seed pairs and is not a seed-population claim |
| SESOI | `+0.010` absolute — a **decision boundary**, not a power target |
| design alternative | `+0.015`, at which the design achieves `P(GO) = 0.8379` |
| battery | **950 prompts** (850 scorable + 100 code), historical mixture preserved exactly |
| seeds | exactly **3 fresh**, paired, fixed blocks; exact IDs deliberately **not** chosen here |
| rule | three-way GO / NO-GO / INCONCLUSIVE, no forced winner |

Two structural findings drove the design, both measured rather than assumed:

* **Correctness is strongly prompt-clustered.** Same-prompt cross-seed ICC is
  `0.25 ± 0.095`; `P(correct | correct on another seed) = 0.257` against a
  `0.022` marginal. The 510 historical prompt-seed rows were never 510
  independent observations.
* **The old design could not have answered this question.** At the Phase-A/B
  design the frozen `0.011695` interval sits at ~1.2 standard errors of the arm
  difference, and Phase B's margin of `0.011765` is ~1.23 — which is why that
  result is protocol-resolved but scientifically weak. C1 therefore does **not**
  reuse `SuccessiveHalvingPlan` or `EquivalenceRule`; both remain untouched and
  in force for Phase A/B.

`usable_rollout` stays secondary and gates via a veto only. It is blind to
correctness by construction — a terse contentless reply scores perfectly on it —
and promoting it would still require its own decision record.

---

## Phase C1 — fixed-path ATTENTION isolation · **NOT STARTED**

**The causal-ish test neither Phase A nor Phase B contains.** Protocol frozen in
[`phase_c0_preregistration.json`](phase_c0_preregistration.json); nothing is
implemented, priced or authorized.

**What C1 is, stated precisely.** C1 *does* execute compute: **2 arms × 3 fresh
recovery seeds = 6 `E1_KD_HEAVY_0860K` recovery probes**. Calling it "not a
recovery run" would be wrong. The correct boundary is:

> Phase C1 is a fixed-path ATTENTION isolation experiment using short 0.86M
> recovery probes. It is **not formal recovery evidence** and does not establish
> recovered-model capability.

Freeze `fe9683e6a9c783bbc6fe276a78c851c6` as the behavioural incumbent. Hold
**everything** fixed except the ATTENTION operator:

| held fixed | at the incumbent's value |
| --- | --- |
| DEPTH | `depth.causal_kl_greedy_v1` @ `calib.domain_balanced@v1` |
| FFN | `ffn.activation_importance_v0` @ `calib.domain_balanced@v1` |
| RESIDUAL_WIDTH | `width.global_pca_v0` @ `calib.reasoning_heavy@v2` |
| operator ordering | `DEPTH → FFN → RESIDUAL_WIDTH → ATTENTION` |
| geometry, teacher, tokenizer | frozen student spec, unchanged |
| battery, seeds, protocol | the frozen behavioural protocol |

**Vary only:** the ATTENTION operator.

> **Question C1.** Does a new ATTENTION operator improve the frozen `fe9683`
> initialization when ATTENTION is the only intended variable?

**Both arms are measured fresh, and the historical evidence is not an arm.**
C0 requires fresh seeds and a fresh battery, so `fe9683`'s retained `sa/sb/sc`
result **cannot** stand in for the incumbent arm: it was produced under the seeds
that selected it and on the development battery. The incumbent is re-run with the
current `attention.weight_proxy_v0` on the same three fresh seeds and the same
fresh battery as the replacement, so the comparison is like-for-like and paired.
`recovery_search_v2` and `sa/sb/sc` remain development/historical evidence only.

**What C1 can conclude:** that a specific ATTENTION operator does or does not
improve *this* fixed path, under a conditional-on-three-seed-pairs interval.
**What it cannot:** that the result generalizes to other paths, orders, student
sizes, or to a population of recovery seeds. Say so in the record.

---

## Phase C2 — ATTENTION-aware joint re-search

**Only if C1 identifies a worthwhile new ATTENTION formulation.** If C1 finds
nothing, C2 does not run.

Put the improved operator back into the joint initialization search and let the
beam reconsider what it was previously never able to exploit:

* operator ordering;
* ATTENTION placement in the sequence;
* ATTENTION calibration profile;
* interactions with DEPTH / FFN / WIDTH calibration.

> **Question C2.** Once ATTENTION itself is improved, does the globally preferred
> initialization composition change?

**Anchors C2 must preserve**, so the two effects stay separable:

1. original `fe9683` — the Phase-B winner;
2. the best C1 fixed-path ATTENTION replacement;
3. the C2 re-search candidate(s).

That three-way comparison is what distinguishes **(a)** improvement from the
ATTENTION operator itself from **(b)** additional benefit from re-optimizing
composition and order around it. Collapsing them into one number would repeat the
exact ambiguity Phase A and Phase B leave behind.

---

## After ATTENTION

Only once ATTENTION is understood, decide whether to run the same
isolation-then-re-search structure for:

1. **FFN** — note it is currently a *constant*: `ffn.activation_importance_v0` @
   `calib.domain_balanced@v1` appears in every non-composite Top-5 path, so
   nothing in Phase A/B says anything about it either way;
2. **RESIDUAL_WIDTH** — the reasoning-heavy width in the winner is a single
   confounded instance and the most interesting untested hypothesis.

Then, in order:

3. **joint confirmation search**;
4. **canonical Stage-1 NLL** — only after the final initialization is uniquely
   selected. Diagnostic, never a promotion criterion;
5. **formal Stage-2/Stage-3 recovery training** — last.

---

## Standing constraints

* **Do not rerun Phase B.** Its result is closed and its evidence retained.
* **Do not treat the Phase-B winner as a benchmark.** It clears the equivalence
  interval by `0.000070` and is not distinguishable from Phase A's leader.
* **Do not upgrade joint-search association into causal evidence.** That is the
  specific gap Phase C exists to close.
* Every paid session follows the existing contract: preregistration, one-use
  authorization bound to an executable digest, authorization-only launch commit,
  all pre-provider gates green, teardown confirmed by the provider.
