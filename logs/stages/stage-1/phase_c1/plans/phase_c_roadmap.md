# Phase C — ATTENTION operator R&D · roadmap


> **Restructured 2026-09-17** by maintainer decision, after C1 returned a
> verdict and C2's restricted Search-1 was accepted as validation evidence. The
> programme is now four phases, C1–C4, and the shape they share is written down
> once in [`../../../../../docs/OPERATOR_PROMOTION_CYCLE.md`](../../../../../docs/OPERATOR_PROMOTION_CYCLE.md)
> as a family-neutral pattern rather than restated per phase.
>
> This file owns the **plan**. For current status read
> [`current.md`](../../../../state/current.md); for what any phase actually
> measured, read that phase's own records.

**Status: C0 COMPLETE / FROZEN · C1 COMPLETE (verdict `GO`) · C2 SEARCH-1 DONE
AND FROZEN, FULL JOINT RE-SEARCH PLANNED AND PRICED BUT NOT FUNDED · C3 NOT
STARTED · C4 CONDITIONAL ON C3.**

The Phase-C0 protocol is frozen in
[`phase_c0_preregistration.json`](phase_c0_preregistration.json), with its sizing
evidence in [`phase_c0_sizing_evidence.json`](phase_c0_sizing_evidence.json).

Nothing on this page is authorization for compute. The C2 full joint re-search
is **blocked on budget**, not on design: the complete chain at the standing beam
width does not fit the remaining project headroom — see
[`phase_c2_full_search_pricing.json`](../../phase_c2/plans/phase_c2_full_search_pricing.json),
which owns every figure and states the minimum cap that would contain both
ceilings. Stating that minimum is not requesting it.

Formal Stage-2/Stage-3 recovery training remains **deferred** until the
operator-development programme is complete. No search-stage front and no Phase-B
winner is authorization for it.

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
> [`decisions.md`](../../../../budget/decisions.md) 2026-09-01.

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

## Phase C1 — fixed-path ATTENTION isolation · **COMPLETE, verdict `GO`**

`attention.weight_proxy_v0` → `attention.activation_importance_v1`.

**The causal-ish test neither Phase A nor Phase B contains.** Protocol frozen in
[`phase_c0_preregistration.json`](phase_c0_preregistration.json). Executed by
attempt 18: both replay gates passed, six probes trained, six evaluated on the
frozen battery, and the frozen Stage-I rule returned **`GO`** at `$10.2018`. A
complete valid verdict ends the round, and the promotion it produced is what the
later phases build on — the figures live in
[`../runs/attempt18/evidence/c1_decision.json`](../runs/attempt18/evidence/c1_decision.json),
not here.

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

## Phase C2 — restricted evidence, then the full joint re-search

`C2 = restricted Search-1 evidence + full joint re-search + behavioural
selection of the new incumbent.`

### C2a — Search-1, restricted · **DONE / FROZEN**

Search-1 held DEPTH, FFN and RESIDUAL_WIDTH at the Phase-B incumbent's mixtures
and varied only ATTENTION's, with order free. It ran to completion, committed
five candidates, and the B→C comparison was completed separately once the
baseline was rebuilt and measured.

**What it established, and the boundary.** Four of five committed candidates
landed in a better ε-Pareto front than the frozen C1 treatment baseline B, and
two dominated it on all three ranked objectives, at margins roughly 50× the
disclosure threshold. That is evidence **about the search procedure** — after
promoting the new ATTENTION operator, changing order and composition alone
produces a real structural signal on the cheap metric. It is **not** behavioural
evidence, it selected **no** incumbent, and its records are frozen: Search-1 is
not rerun, B is not remeasured, no frozen C candidate is remeasured, and no
selection or comparison record is rewritten.

**The local Search-2 refinement is WITHDRAWN.** Polishing locally around the
winners of a restricted search would inherit that restriction. The accepted
reading of Search-1 points the other way — the procedure works here, so widen it.

### C2b — full joint re-search · **PLANNED, PRICED, NOT FUNDED**

Protocol and pricing:
[`phase_c2_full_search_protocol.json`](../../phase_c2/plans/phase_c2_full_search_protocol.json) ·
[`phase_c2_full_search_pricing.json`](../../phase_c2/plans/phase_c2_full_search_pricing.json).
The space is **derived from the registry**, never written down:
[`full_search_space.py`](../../../../../scripts/experiments/phase_c2/full_search_space.py).

> **Question C2.** With the promoted ATTENTION operator in the accepted library,
> what is the globally preferred initialization composition when
> implementations, applicable calibration profiles and operator **order** all
> compete in one search?

**Nothing is held fixed, and that is the point.** The incumbent DEPTH / FFN /
WIDTH calibration assignments were chosen by a search in which ATTENTION could
not consume calibration at all — `attention.weight_proxy_v0` declares
`CalibrationNeed.NONE` and is therefore offered once, against the
no-calibration sentinel, however many mixtures are active. Those assignments
cannot be assumed optimal now that the operator beside them consumes
calibration. So implementations, profiles and order all compete jointly, and
calibration choices are allowed to affect pruning.

**Why the space grew.** The promoted operator consumes calibration where the one
it replaced did not, so ATTENTION branches over every active mixture instead of
once. Same kinds, same order freedom, one more branching factor. **Do not quote
a remembered size** — it is enumerated, and the enumeration distinguishes the
decomposed four-operator subspace from the total, because a composite operator
reaches the target in a single step and contributes leaves outside the
decomposition.

**One exclusion, recorded.** `attention.weight_proxy_v0` is out — not for cost,
but because C1 *is* the isolation experiment between it and the promoted
operator and it has a completed verdict. Promotion is what an isolation verdict
is for. Everything else applicable competes, including the cheap
`depth.positional_v0` and `composite.stage1_sandwich_v0`. If the promotion is
ever withdrawn, the exclusion is withdrawn with it.

**Beam width 6 is part of the design.** The ranking policy, its objectives and
its ε are frozen and unchanged, and the standing width is what `SCHEDULE_V1`
declares and what this protocol proposes. A narrower beam leaves the space
intact but carries fewer partial paths forward, so it explores less of it and
can return a different front: adopting one is a **changed experiment** with
reduced breadth, to be registered as the width before launch. **Narrowing the
beam merely to fit an existing cap is not permitted** — the pricing record
prices the alternatives so breadth can be traded deliberately, not so the
authorization number can be made smaller. The goal is never exhaustive
enumeration of every leaf — it is that every admissible alternative *competes
inside one search*.

### C2c — Top-K, then bounded behavioural selection · **DEFINED, PRICED, NOT FUNDED**

The search commits a **Top-5** candidate set by the frozen ε-Pareto ranking, and
that set is closed when it is committed. A candidate set that can grow once
behavioural results are visible is not a preregistered set.

Then, and only then, behaviour decides. Cheap-metric order is **not** behavioural
order: E7 measured a `-5.22` nat NLL swing that moved behaviour by `+0.0000`. So
each admitted candidate is trained under the frozen `0.86M` recovery recipe and
scored on the **frozen Phase-C battery** `c1_confirmation_v1` under **C1's**
statistical discipline.

> **This is not Phase-B-style selection, and an earlier draft of it wrongly was.**
> C0 retired the Phase-A/B behavioural design as scientifically weak — its
> equivalence interval was ~1.2 SE and Phase B's resolving margin ~1.23 SE — and
> that is exactly why C1 stopped using `SuccessiveHalvingPlan` and
> `EquivalenceRule`. C2c therefore uses the 950-prompt / 850-scorable Phase-C
> battery, `c1_confirmation_scoring@v1`, the prompt-cluster bootstrap with seeds
> as fixed blocks, **SESOI `+0.010`**, and **GO / NO-GO / INCONCLUSIVE with no
> forced winner**. Phase B's `sa/sb/sc` and its `0.011695…` interval are **not
> used**. The frozen Search-1 plan says the same thing independently: a later
> behavioural B→C test belongs on the Phase-C battery and semantics.

**Fresh paired C2 seeds, derived rather than chosen.** Four seeds are drawn by
C1's own rule under a `:phase-c2:` domain from C0's frozen base digest, skipping
both the Phase-A/B selection seeds *and* C1's three confirmation seeds — the
latter because the anchor this stage tests against was **promoted** under them,
which is the same winner's-curse channel one step along. The values are
materialized and hash-bound in the protocol before any candidate behavioural
result exists.

**The stage is bounded by a select-then-confirm split on disjoint seeds**, which
is what lets it be cheaper than five full three-seed comparisons without
becoming uninterpretable:

| rung | seeds | arms | probes | decides |
| --- | --- | --- | --- | --- |
| screening | 1 | Top-5 + B | 6 | which **one** candidate advances — ranking only, no veto, no promotion |
| confirmation | 3 | advanced + B + control | 9 | the C2 incumbent, under the frozen Phase-C rule |

**15 probes, exact.** There is no conditional rung: C0 fixed three confirmation
seeds and `fourth_seed: never`.

Exactly **one** candidate advances, so exactly one hypothesis is confirmed and
the one-sided 95% LCB needs no multiplicity correction. The claim boundary is
stated rather than glossed: the confirmed candidate was *selected on disjoint
screening data*, so its interval is a valid bound for **that** candidate against
B conditional on the three seeds — not a simultaneous statement about all five,
and the eliminated candidates receive no verdict. Advancing two would require
Holm and three more probes; advancing one is a deliberate trade of breadth for a
clean single-hypothesis confirmation.

Two anchors advance unconditionally: the **frozen C1 treatment baseline B**,
because the question is whether re-optimizing composition beats B
*behaviourally* and Search-1's `state_eval` evidence cannot substitute for that;
and the **canonical control**, which binds the catastrophic-capability veto's
control operand and supplies the absolute floor an anchor-relative veto cannot.
C0 requires that operand to be named explicitly — C1 bound it to the incumbent
for want of a control arm; C2 probes a control and binds it there.

B's frozen `state_eval` measurement is not touched; these are fresh recovery
probes of the same initialization under fresh seeds.

**Every probe is trained fresh.** The historical-probe-reuse ruling records
`reuse_verified: false`, so there is no admissible reuse to net off.

Only this stage may name a **C2 incumbent**, and **INCONCLUSIVE** with no
incumbent is a legitimate terminal result — not a reason for a fourth seed, a
second screening rung, or a re-run.

---

## Phase C3 — causal-KL ATTENTION isolation on the C2 incumbent · **NOT STARTED**

`attention.activation_importance_v1` vs the future `attention.causal_kl_v1`,
with the rest of the C2 incumbent held fixed as far as scientifically possible.

The same isolation shape as C1, one rung further along: the incumbent moves from
the Phase-B winner to the C2 incumbent, and the operator under test moves from
the activation-based formulation to a causal-KL one. C1's structure — paired
fresh seeds, a fixed path, everything but the one operator held — is what makes
the result attributable, and it is reused rather than redesigned.

**C3 cannot start before C2 names an incumbent.** Running it against a candidate
the behavioural stage never selected would isolate an operator on a path nobody
chose.

`attention.causal_kl_v1` does not exist yet. Building it is operator R&D and is
not a paid experiment.

---

## Phase C4 — full joint re-search with the promoted causal-KL ATTENTION · **CONDITIONAL**

**Only if C3 promotes.** If C3's verdict promotes causal-KL ATTENTION, then the
C2b/C2c pair repeats with the newly promoted operator in the accepted library:
another full joint re-search, another Top-K, another behavioural selection.

If C3 does not promote, C4 does not run and the C2 incumbent stands.

> **A >1-session search is not currently available.** Search state ids are
> content-derived, which gives a state an identity — not its bytes. The frozen
> Search-1 plan records that the multi-gigabyte search workdir *cannot be
> relayed for resume*, so a fresh provider resource must re-derive lost state.
> No durable cross-session mechanism was implemented or validated, and none was
> built: it is a possible future design option, and no plan here may assume it.

This is the same cycle as C1→C2, and that repetition is deliberate: it is a
**pattern**, documented family-neutrally in
[`../../../../../docs/OPERATOR_PROMOTION_CYCLE.md`](../../../../../docs/OPERATOR_PROMOTION_CYCLE.md),
so a future teacher/student family, geometry or compression ratio reuses the
machinery instead of a Qwen3-shaped copy of it.

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
