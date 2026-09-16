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

**The question is simple, and it is about the current incumbent.** C1 already
established B behaviourally: after the frozen `0.86M` recovery, pooled
`correct_overall` was `105/2550 = 4.12%` against the old incumbent's
`70/2550 = 2.75%`, a paired `+0.01372549` with a `GO` verdict. So C2c asks only:

> Does the selected full-search initialization **C**, after the same frozen
> `0.86M` recovery, outperform **B**?

The target is *improve on the current accepted incumbent*, not *beat the original
initialization again*. The project's original control is therefore **not a C2
arm**: if C beats it but loses to B, C must not promote, and if C beats B the old
control adds no promotion information. The guardrails use the incumbent-relative
semantics C1 already used, with B in the comparator position. This is also what
keeps the cycle scalable — C4 and beyond challenge whatever incumbent the
previous turn left, rather than repeatedly retraining the original.

> **Phase-C discipline, not Phase B's.** C0 retired the Phase-A/B behavioural
> design as scientifically weak — its equivalence interval was ~1.2 SE and Phase
> B's resolving margin ~1.23 SE — which is why C1 stopped using
> `SuccessiveHalvingPlan` and `EquivalenceRule`. C2c uses the frozen
> `c1_confirmation_v1` battery, `c1_confirmation_scoring@v1`, the prompt-cluster
> bootstrap with seeds as fixed blocks, **SESOI `+0.010`**, and **GO / NO-GO /
> INCONCLUSIVE with no forced winner**. Phase B's `sa/sb/sc` and its `0.011695…`
> interval are **not used**.

**Fresh paired C2 seeds, derived rather than chosen** — four by C1's own rule
under a `:phase-c2:` domain from C0's frozen base digest, skipping the Phase-A/B
selection seeds *and* C1's three, the latter because the anchor this stage tests
against was **promoted** under them. Materialized and hash-bound in the protocol
before any candidate behavioural result exists.

**Screening is disjoint in BOTH dimensions, and that is the point.** Disjoint
seeds alone are insufficient: C0's inferential unit is the **prompt**, and it
measured substantial same-prompt cross-seed dependence — ICC `0.25 ± 0.095`, and
`P(correct | correct on another seed) = 0.257` against a `0.022` marginal, an
`11.7×` lift. Selecting and confirming on the same prompts would let the
selection leak into the confirmation through that dependence. So screening gets
its own prompts as well as its own seed:

| rung | seeds | arms | battery | probes | decides |
| --- | --- | --- | --- | --- | --- |
| screening | 1 | Top-5 + B | `c2_screening_v1` | 6 | which **one** candidate advances |
| confirmation | 3 | C + B | `c1_confirmation_v1` | 6 | the C2 incumbent |

**12 probes, exact.** No conditional rung: C0 fixed three confirmation seeds and
`fourth_seed: never`.

[`c2_screening_v1`](../../phase_c2/plans/c2_screening_battery.json) is built and
frozen. It preserves C1's mixture exactly — `correct_overall` and its SESOI are
defined *on* the mixture, so a screening delta only informs a confirmation delta
if both measure the same distribution — and it is **measured** disjoint from
`c1_confirmation_v1` by stable id *and* normalized prompt content, as well as
from calibration, `state_eval`, the recovery-training corpus and the reserved
final-promotion battery. It produces no verdict and may promote nothing.

The screening ranking rule is frozen: **maximize the paired single-seed
Δ`correct_overall`(candidate − B)** on `c2_screening_v1`. B is the anchor, never
an advancing candidate, and `usable_rollout` is never positive ranking credit.
An exact tie is broken by the already-frozen full-search ordering and then the
deterministic state id — both fixed before any behavioural datum exists.

Exactly **one** candidate advances, so one hypothesis is confirmed and the
one-sided 95% LCB needs no multiplicity correction. The claim boundary is
recorded: the confirmed candidate was *selected on disjoint screening prompts and
a disjoint seed*, so its interval is a valid bound for **that** candidate against
B conditional on the three confirmation seeds — not a simultaneous statement
about all five, and the eliminated candidates receive no verdict.

**Every probe is trained fresh.** The historical-probe-reuse ruling records
`reuse_verified: false`.

Only this stage may name a **C2 incumbent**, and **INCONCLUSIVE** with no
incumbent is a legitimate terminal result.

### The interpretation boundary, stated

* Search-1 and the full joint re-search are **initialization-search stages
  only**. They perform **no** `0.86M` recovery training.
* Their KL / `state_eval` results are **hypothesis-generation and
  candidate-selection** evidence. They may narrow the field; they may never
  promote, rank behaviourally, or stand in for a recovery comparison.
* C1's `4.12%` is a **post-recovery behavioural measurement**, not raw
  initialization accuracy, and must never be quoted as the latter.
* C2 promotion depends **only** on the fresh recovery comparison of the selected
  C against B.

### Execution path · **search session implemented, behavioural session owed**

Two sessions, two authorizations, deliberately not combined:

1. **full joint search** —
   [`autoinit_phase_c2_full_search_driver.py`](../../../../../scripts/pod/autoinit_phase_c2_full_search_driver.py):
   `bind_identities` → `full_joint_search` → `commit_top_k`, and it **stops**.
   It trains nothing, measures no behaviour, and has no code path into a
   behavioural stage. Executed end to end at toy scale, which found and closed a
   real defect.
2. **behavioural selection** — screening → freeze the selected C → confirmation
   C vs B → derive GO / NO-GO / INCONCLUSIVE. **Not implemented**: its inputs do
   not exist until session 1 commits a candidate set.

The launcher and governance chain are **owed at authorization time** and are
deliberately not built for an unfunded experiment.

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
