# Phase C — ATTENTION operator R&D · roadmap

**Status: NOT STARTED · NOT DESIGNED · NOT PRICED · NOT AUTHORIZED.**

No Phase-C experiment exists. Nothing below is a plan of record for a paid
session — it is the agreed *structure*, recorded so the next session starts from
it rather than reinventing it. A Phase-C design must be written, reviewed and
authorized before any compute.

Formal Stage-2/Stage-3 recovery training remains **deferred** until the
operator-development programme is complete. The Phase-B winner is **not**
authorization for it.

---

## Why ATTENTION, and why first

From [`phase_a_vs_phase_b_comparison.md`](phase_a_vs_phase_b_comparison.md) §4:

`attention.weight_proxy_v0` appears in **every** non-composite Phase-B Top-5
path, and in **every one of them the calibration selected is `calib.none@v1`** —
including the winner, the runner-up, Phase A's leader, and `ab7632b00788`, which
placed ATTENTION *first* and still declined to calibrate it. The search was free
to use either profile and consistently did not.

That does **not** show attention manipulation is useless. It supports the
narrower, defensible claim:

> **The current ATTENTION operator and its search formulation did not provide a
> competitive positive transformation in Phase B.**

Two readings remain open and the programme is designed to separate them: the
*operator* may be weak, or the *search formulation around it* may be. Neither
phase can distinguish them, because neither varied one operator with the rest
held fixed.

---

## Phase C0 — protocol and power design

**Before any probe is bought.** This step exists because of a measured property
of the evidence, not as ceremony.

The effects that decided both phases are **1–6 correct answers out of 510**, and
the frozen equivalence interval is `0.011695` — six samples. A single-variable
ATTENTION experiment powered like Phase A/B will most likely return
`unresolved_equivalence` regardless of whether the new operator helps.

C0 must settle, and record:

1. **The minimum effect worth detecting.** What size of improvement would change
   what we build? State it in `correct_overall` and in correct-answer counts.
2. **Whether the current sample size can detect it.** At n=510 across three
   seeds, one answer is `0.001961`. Compute the detectable effect honestly; if it
   exceeds the minimum worth detecting, the design is not adequate as-is.
3. **The implications of the equivalence interval.** `0.011695` derives from the
   control's own binomial SE under the frozen
   `seed_aware_max_binomial_seedrange` rule. It is a property of the *control*,
   not of the candidates. Whether it is the right yardstick for an
   operator-isolation experiment is a live question — changing it requires a
   decision record, and it must be decided **before** results exist.
4. **Options if power is inadequate**, priced: more prompts per probe, more
   seeds, a larger probe rung, a different battery, or accepting that the
   experiment answers a coarser question. Each has a cost; none may be chosen
   after seeing an outcome.
5. **`usable_rollout` as a measurement question only.** It separates more cleanly
   than `correct_overall` in every comparison to date
   (`0.6842 / 0.6561 / 0.5456 / 0.4947` against correctness rates clustered near
   the floor). That is worth understanding. It must **not** become the ranking
   metric without a separate, explicit, reviewed decision record — it is blind to
   correctness by construction, and a terse contentless reply scores perfectly on
   it.

**Output:** a written protocol with a registered decision rule, before probes.

---

## Phase C1 — fixed-path ATTENTION isolation

**The causal-ish test neither Phase A nor Phase B contains.**

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

**Vary only:** the ATTENTION operator (and, if C0 decides it is in scope, its
calibration).

> **Question C1.** Does a new ATTENTION operator improve the frozen `fe9683`
> initialization when ATTENTION is the only intended variable?

**Anchors that must be measured or cited in the same comparison:**

* `fe9683` as-is — the incumbent, with its retained three-seed evidence;
* the incumbent with the *current* `attention.weight_proxy_v0` re-run if the
  protocol changes at all, so the comparison is like-for-like;
* each candidate ATTENTION formulation.

**What C1 can conclude:** that a specific ATTENTION operator does or does not
improve *this* fixed path. **What it cannot:** that the result generalizes to
other paths, orders, or student sizes. Say so in the record.

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
