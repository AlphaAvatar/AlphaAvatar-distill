# Phase A vs Phase B — final scientific comparison

**Status:** complete, from retained evidence only. No GPU work; nothing here was
measured for this document. Every number is traceable to a committed artifact,
cited inline.

**What this document is for.** It is the scientific handoff into Phase C. It
therefore keeps four kinds of evidence apart on purpose, because they answer
different questions and have very different strength:

| kind | what it can support |
| --- | --- |
| **search-side ranking** | which initializations a Pareto beam preferred on teacher-KL objectives. Says nothing about behaviour |
| **behavioural selection** | which initialization the frozen `sa/sb/sc` protocol selects, under a fixed 0.86M probe |
| **diagnostic** | teacher-forced top-1, NLL, per-capability rates. Reported, never ranks |
| **recovery** | what a recovered model can do. **This project has none.** No formal Stage-2/3 recovery has been run |

Nothing below is recovery evidence.

---

## 1. Phase A — final result

**Question.** Under a *fixed* calibration distribution, which initialization and
operator order gives the best behavioural starting point at the frozen student
size?

**Regime.** Search under `calib.domain_balanced@v1` alone. Selection by the
frozen successive-halving plan `02be33b9a7a8e26b…` over three seeds
`sa=20260726`, `sb=20260801`, `sc=20260813`, ranking on `correct_overall` with a
preregistered equivalence interval of **`0.011695296982299022`** and a
feasibility floor of `0.3`.

**Finalists.** `cca699c93f34`, `85bde4ded2c3`, and the canonical initialization
control `qwen3_0p6b_init_v0`.

**Evidence used.** Eleven probes across three rungs; the tie-break rung ran.

**Final metrics** ([`autoinit_recovery_continuation_attempt7/phase_a_result.json`](autoinit_recovery_continuation_attempt7/phase_a_result.json)):

| candidate | seeds | correct | `correct_overall` | `usable_rollout` | `correct_given_usable` |
| --- | --- | --- | --- | --- | --- |
| `cca699c93f34` | sa+sb+sc | 15/510 | **0.029412** | 0.6561 | 0.042254 |
| `85bde4ded2c3` | sa+sb+sc | 10/510 | 0.019608 | 0.5456 | 0.033670 |
| control | sa+sb | 3/340 | 0.008824 | 0.4947 | 0.016304 |

**Status: `unresolved_equivalence`, `winner: None`.** The leader's margin over
the runner-up is `0.009804`, **inside** the `0.011695` interval, so the two were
not separated even after the third seed.

### What Phase A established

* Both searched finalists are **separated from the canonical control** on
  `correct_overall` (`0.029412` and `0.019608` against `0.008824`), and on
  `usable_rollout` (`0.6561`, `0.5456` against `0.4947`). A searched
  initialization beat the canonical one under this protocol.
* The frozen three-seed procedure runs end to end and terminates honestly on a
  null.

### What Phase A did **not** establish

* **No winner.** `unresolved_equivalence` is the result, not a formality.
* **No single-operator causal claim.** `cca699c93f34` is a *joint* path —
  `FFN → DEPTH → RESIDUAL_WIDTH → ATTENTION`. That FFN ran first, or that DEPTH
  used `causal_kl_greedy_v1`, is **not** evidence that either choice caused the
  outcome. Nothing in Phase A varied one operator with the rest held fixed.
* **No capability claim.** 15 correct out of 510. Every arm including the control
  sits near the floor. These are selection probes at 0.86M tokens, not recovery.

---

## 2. Phase B — final result

**Question.** Does the preferred composition change when the calibration
distribution is allowed to vary? Specifically, does a joint search over **two**
calibration profiles select a different initialization?

**Regime.** Joint `P=2` beam search under `beam.pareto_multi_objective@v2`
(`30a5305bc06dd1e8…`) over `calib.domain_balanced@v1` (`11f36a88…` / content
`d65c1f40…`) and `calib.reasoning_heavy@v2` (`6c67b8df…`), evaluated on
`state_eval@v1` (`6421fa4cf12ee2a1…`). Same frozen behavioural plan, same seeds,
same interval, same ranking metric as Phase A.

### Stage 1 — authoritative Top-5

Selection `84fd6496851995ae…`
([`autoinit_phase_b_attempt5/stage1_selection.json`](autoinit_phase_b_attempt5/stage1_selection.json)):

| state | Pareto front | path |
| --- | --- | --- |
| `cca699c93f34` | **0** | `FFN(db) → DEPTH(db) → RESIDUAL_WIDTH(db) → ATTENTION(none)` |
| `85bde4ded2c3` | 1 | `DEPTH(none) → FFN(db) → RESIDUAL_WIDTH(db) → ATTENTION(none)` |
| `ab7632b00788` | 2 | `ATTENTION(none) → DEPTH(rh) → RESIDUAL_WIDTH(db) → FFN(db)` |
| `fe9683e6a9c7` | **3** | `DEPTH(db) → FFN(db) → RESIDUAL_WIDTH(rh) → ATTENTION(none)` |
| `bf5ae3b6ae00` | 4 | `COMPOSITE_STAGE1(rh)` |

*(db = `calib.domain_balanced@v1`, rh = `calib.reasoning_heavy@v2`, none =
`calib.none@v1`.)*

**The search-side ranking and the behavioural outcome disagree sharply.** On the
search's own objectives the eventual behavioural winner `fe9683e6a9c7` sat on
**front 3**, with teacher-KL `9.2001` equal-domain / `9.7656` worst-domain
against `cca699`'s `8.0046` / `8.2859` — nearly the worst of the Top-5 except the
composite. This is direct evidence that **search-side KL is a poor proxy for
behavioural selection here**, and it is why the frozen rule forbids breaking a
behavioural tie with Stage-1 ranking.

### Identity collapse

Two Top-5 leaves reproduced retained Phase-A finalists byte-for-byte
(`cca699c93f34`, `85bde4ded2c3`). The amendment `df413bd99119dab7…` collapses on
state id **and** re-derived artifact digest, giving a **six-candidate**
behavioural universe `e94f15d2e648d86a…` with no rank-6/7 backfill.

### Rung 1 — and the cut that matters

Ranked on seed `sa` alone, top-2 among searched leaves, control advances free:

| candidate | `correct_overall` (sa only) | |
| --- | --- | --- |
| `fe9683e6a9c7` | 0.047059 | **advances** |
| `85bde4ded2c3` | 0.041176 | **advances** |
| `bf5ae3b6ae00` | 0.035294 | cut |
| `cca699c93f34` | 0.035294 | cut |
| `ab7632b00788` | 0.017647 | cut |
| control | 0.011765 | advances unconditionally |

**Read this carefully.** Phase A's leader `cca699c93f34` was eliminated here on
**one seed's** evidence, `0.005882` behind the second qualifier — a gap *inside*
the `0.011695` equivalence interval — and it **tied `bf5ae3b6ae00` exactly**. The
protocol permits this: rung 1 is a halving rung, not an equivalence test. But it
means Phase B never ran a behavioural head-to-head between `fe9683e6a9c7` and
`cca699c93f34` at rungs 2–3.

### Corrected rung 2, and a withdrawn decision

Attempt 4 reported `resolved / winner=fe9683e6a9c7` and that decision was
**withdrawn**: the inherited pooling admitted the *imported* `85bde4ded2c3/sc`
into a rung-2 comparison, making it `sa+sb+sc` (n=570) against `sa+sb` (n=380)
for the others. Recomputed over `sa+sb` alone
([`autoinit_continuation_b_corrected_rung2.json`](autoinit_continuation_b_corrected_rung2.json)),
and independently reproduced on the pod in the final session:

| candidate | `sa+sb` | `correct_overall` |
| --- | --- | --- |
| `fe9683e6a9c7` | 11/340 | 0.032353 |
| `85bde4ded2c3` | 9/340 | 0.026471 |
| control | 3/340 | 0.008824 |

Margin `0.005882` < `0.011695` → **`tie_pending`**, candidates
`{fe9683e6a9c7, 85bde4ded2c3}`; the control is outside the interval and does not
advance.

### Tie-break and final result

`85bde4ded2c3/sc` was **reused** from retained Phase-A evidence.
`fe9683e6a9c7/sc` was the **single purchased observation** ($1.5433).

| final pooled | seeds | correct | `correct_overall` | `usable_rollout` | `correct_given_usable` |
| --- | --- | --- | --- | --- | --- |
| **`fe9683e6a9c7`** | sa+sb+sc | 16/510 | **0.031373** | 0.6842 | 0.042328 |
| `85bde4ded2c3` | sa+sb+sc | 10/510 | 0.019608 | 0.5456 | 0.033670 |
| control | sa+sb | 3/340 | 0.008824 | 0.4947 | 0.016304 |

**`resolved`, winner `fe9683e6a9c783bbc6fe276a78c851c6`**, `tie_break_ran: true`,
report `8c8842b84fe85cec…`.

> **The separation is razor-thin.** Margin `0.011765` against interval
> `0.011695` — it clears by **`0.000070`**. One correct answer is
> `1/510 = 0.001961`, so the excess is **~3.6% of a single correct sample**. At
> 15 correct instead of 16 the result would have been
> `unresolved_equivalence`.

### Evidence lineage

| source | probes | role |
| --- | --- | --- |
| historical (`a5115816861ad239…`) | 8: `85bde` sa/sb/sc, `cca699` sa/sb/sc, control sa/sb | cited |
| Attempt-5 (`2e5030fadc786faa…`) | 3: `ab7632`, `bf5ae3`, `fe9683` sa | cited |
| Attempt-4 (`4d2ef1c9f113dbda…`) | 1: `fe9683/sb` | cited |
| **final session** | 1: `fe9683/sc` | **purchased** |

All three reuse records are strictly reconstructed — completeness, frozen seed,
artifact digest re-derived from retained checkpoint bytes, frozen battery, live
scoring contract, attested protocol, poolable counts, no duplicate observation —
and all three are authorization-bound.

### What Phase B established / did not

**Established.** A joint two-profile search produces a Top-5 that differs from
Phase A's preference; the frozen behavioural protocol, applied to it, is
protocol-resolved in favour of `fe9683e6a9c7`; that candidate is clearly
separated from the canonical control on both axes; and search-side KL ranking
does not predict behavioural outcome.

**Not established.** That `fe9683e6a9c7` is *decisively* better than the
alternatives — the margin is 3.6% of one sample. That any component operator or
ordering is causal. Anything about recovered capability.

---

## 3. Direct comparison

**Did Phase B select a different initialization from Phase A?**
Phase A selected **nothing** — it ended `unresolved_equivalence`. So the honest
statement is: *Phase A's numerical leader was `cca699c93f34`; Phase B's
protocol-resolved winner is `fe9683e6a9c7`.* Phase B did not overturn a Phase-A
decision, because Phase A never made one.

**Did the P=2 / calibration change alter the landscape?**
Yes, at the search level: `fe9683e6a9c7` uses `RESIDUAL_WIDTH(reasoning_heavy)`
and could not exist in Phase A's single-profile regime. It entered the Top-5 on
front 3 and then led on behaviour. Whether the *reasoning-heavy width* is what
made the difference is **not** established — see §4.

**`correct_overall`, Phase-A leader vs Phase-B winner** — both have complete
`sa+sb+sc` under the same battery, seeds and comparable protocol, so they can be
put side by side *post hoc*:

| | correct | `correct_overall` | `usable_rollout` |
| --- | --- | --- | --- |
| `fe9683e6a9c7` (Phase-B winner) | 16/510 | 0.031373 | 0.6842 |
| `cca699c93f34` (Phase-A leader) | 15/510 | 0.029412 | 0.6561 |
| difference | **1 answer** | **0.001961** | 0.0281 |

**The two differ by exactly one correct answer out of 510.** `0.001961` is about
**one sixth** of the equivalence interval. Under the frozen rule these two are
**behaviourally equivalent** — they are not separated, and Phase B never tested
them head-to-head because `cca699c93f34` was cut at rung 1 on single-seed
evidence.

*(This is a post-hoc comparison of retained evidence. It was not preregistered
and decides nothing. It is reported because omitting it would leave a false
impression that Phase B demonstrated superiority over Phase A's leader.)*

**`usable_rollout`.** `fe9683e6a9c7` leads on every arm — `0.6842` vs `0.6561`
(Phase-A leader), `0.5456`, `0.4947`. This is **supporting behavioural evidence
only**. It does not rank and must not be used to retroactively strengthen the
correctness decision.

**So which statement is true?**

> **Phase B is protocol-resolved in favour of `fe9683e6a9c7`, and the separation
> is extremely weak.** It is *not* strong evidence that `fe9683e6a9c7` is
> intrinsically or decisively superior. Against Phase A's leader it is
> **"different, and not distinguishable"**; against the runner-up it is
> **"resolved by 3.6% of one sample"**; against the canonical control it is
> **"clearly better under this protocol"** — that last comparison is the only
> robust one.

**How much confidence in the winner?** Low as a capability claim, adequate as a
*protocol* outcome. It is the correct incumbent to build on, not a benchmark to
defend.

**Structural differences vs the main Phase-A candidates:**

| | DEPTH | FFN | RESIDUAL_WIDTH | ATTENTION | order |
| --- | --- | --- | --- | --- | --- |
| `cca699c93f34` | db, `causal_kl_greedy_v1` | db | db | none | FFN first |
| `85bde4ded2c3` | **none**, `positional_v0` | db | db | none | DEPTH first |
| **`fe9683e6a9c7`** | db, `causal_kl_greedy_v1` | db | **rh** | none | DEPTH first |

`fe9683e6a9c7` differs from `cca699c93f34` in **two** coupled ways at once —
operator order (DEPTH-first vs FFN-first) *and* width calibration (rh vs db). It
differs from `85bde4ded2c3` in the depth implementation and calibration
(`causal_kl_greedy_v1`/db vs `positional_v0`/none) *and* the width calibration.
**No single difference is isolated anywhere in the design**, so none of them can
be credited with the outcome.

---

## 4. Operator-level evidence carried forward

Classification is deliberately conservative. "Joint-path association" is the
weakest useful category and is where almost everything sits.

### DEPTH
**Repeated association in strong joint paths.** `depth.causal_kl_greedy_v1`
appears in the Phase-B winner and in the Phase-A leader; `depth.positional_v0`
with `calib.none@v1` appears in `85bde4ded2c3`, the consistently weaker of the
two long-standing finalists (`0.019608` in both phases). DEPTH-first ordering
appears in the winner and in `85bde4ded2c3` — i.e. in both the best and the
weakest — so **ordering is not supported**. Depth is also the dominant search
cost (~71% of the P=2 search).
*Not causal evidence. No single-variable DEPTH experiment exists.*

### FFN
**Ubiquitous, therefore uninformative.** `ffn.activation_importance_v0` under
`calib.domain_balanced@v1` appears in **all four** non-composite Top-5 paths,
including every finalist and every non-survivor. A factor present in every arm
cannot explain differences between arms.
*Insufficient evidence either way. It is a constant, not a variable.*

### RESIDUAL_WIDTH
**Weak association, single instance.** `width.global_pca_v0` is used everywhere;
only the calibration differs. `calib.reasoning_heavy@v2` for width appears in
exactly one Top-5 path — the winner. That is a sample of one, confounded with
operator order.
*Not causal evidence. The most interesting single hypothesis to test properly,
and the least supported today.*

### ATTENTION — the Phase-C motivation
**Negative / weak search signal, and it is consistent.** `attention.weight_proxy_v0`
appears in **every** non-composite Top-5 path, and in **every one of them the
calibration selected is `calib.none@v1`** — including the winner, the runner-up,
the Phase-A leader, and `ab7632b00788`, which placed ATTENTION *first* and still
took `calib.none@v1`.

The search was free to calibrate attention with either profile and, across every
competitive path it found, declined to.

This does **not** show that attention manipulation is useless. It supports the
narrower and defensible conclusion:

> **The current ATTENTION operator and its search formulation did not provide a
> competitive positive transformation in Phase B.**

That is the motivation for Phase C.

### COMPOSITE
**Negative search signal, one instance.** `composite.stage1_sandwich_v0` under
`calib.reasoning_heavy@v2` (`bf5ae3b6ae00`) landed on Pareto front 4 — well
behind the multi-operator paths — and was cut at rung 1 with `0.035294` on `sa`.
*One data point, consistent with the composite being worse than an explicit
operator sequence here. Not established.*

---

## 5. Recommended frozen incumbent for Phase C

**Recommendation: yes — `fe9683e6a9c783bbc6fe276a78c851c6`.**

Justified as:

* it **won the frozen Phase-B behavioural protocol**, applied exactly as
  preregistered;
* it has the **best supporting `usable_rollout`** of every candidate measured in
  either phase (`0.6842`);
* it is **clearly separated from the canonical control** on both axes — the one
  robust comparison available;
* it is the **correct incumbent for operator R&D**: a fixed, fully identified,
  retained checkpoint (`c313d1b4081b…`) with complete three-seed behavioural
  evidence to compare against.

Recorded against it, with equal weight:

* the correctness margin over the runner-up is **`0.000070`** — 3.6% of one
  sample;
* it is **not distinguishable** from Phase A's leader `cca699c93f34`
  (`0.001961` apart, ~1/6 of the interval), and the two were never tested
  head-to-head;
* it is **not a strong benchmark** — 16 correct of 510, near the floor;
* it is **not recovered** — no Stage-2/3 training has been run on it;
* **none of its component operators is individually validated** by this win.

The incumbent is a *starting point*, not a result to defend.

---

## 6. Phase C handoff — proposed starting point

**Phase C — ATTENTION operator R&D.** Not launched, not designed, not priced,
not authorized. The agreed structure — C0 power design, C1 fixed-path isolation,
C2 ATTENTION-aware re-search — is recorded in
[`phase_c_roadmap.md`](phase_c_roadmap.md).

Proposed direction:

1. **Freeze** the Phase-A/B evidence and the incumbent `fe9683e6a9c7`.
2. **Do not rerun Phase B.** Its result is closed and its evidence retained.
3. **Do not begin formal recovery training.** Deferred until the
   operator-development program is complete.
4. **Redesign the ATTENTION operator itself.** The evidence motivating this is
   §4: the current formulation was never competitively calibrated by a search
   that was free to calibrate it.
5. **Test the new operator against the frozen incumbent** with as clean a
   single-variable protocol as practical — one operator varied, the rest of the
   path held at the incumbent's. This is the design property both phases lack
   and the reason no operator-level causal claim exists today.
6. Only once ATTENTION is understood, decide on **FFN**, then
   **RESIDUAL_WIDTH**.
7. **Joint confirmation search** later.
8. **Canonical Stage-1 NLL** only once the final initialization is uniquely
   selected.
9. **Formal Stage-2/3 recovery** last.

### Two design cautions for Phase C

* **The interval is wide relative to the effects.** Differences that matter here
  are ~1–6 correct answers out of 510, and the frozen interval is `0.011695` — 6
  samples. A single-variable ATTENTION experiment powered like Phase A/B will
  most likely return `unresolved_equivalence`. Decide the power question
  *before* buying probes, not after.
* **`usable_rollout` separates more cleanly than `correct_overall`** in every
  comparison here (`0.6842 / 0.6561 / 0.5456 / 0.4947` against correctness rates
  clustered near the floor). That is worth investigating as a *measurement*
  question — but it must not be promoted to the ranking metric without an
  explicit, reviewed decision record, because it is blind to correctness by
  construction.

---

## Provenance

| fact | artifact |
| --- | --- |
| Phase-A final | [`autoinit_recovery_continuation_attempt7/phase_a_result.json`](autoinit_recovery_continuation_attempt7/phase_a_result.json) |
| Phase-B Stage-1 Top-5, paths, Pareto fronts | [`autoinit_phase_b_attempt5/stage1_selection.json`](autoinit_phase_b_attempt5/stage1_selection.json) |
| identity collapse, rung-1 ranking | [`autoinit_phase_b_identity_collapse_amendment.json`](autoinit_phase_b_identity_collapse_amendment.json) |
| corrected rung 2 | [`autoinit_continuation_b_corrected_rung2.json`](autoinit_continuation_b_corrected_rung2.json) |
| Phase-B final | [`autoinit_continuation_b_attempt5.json`](autoinit_continuation_b_attempt5.json) |
| reuse records | [`autoinit_historical_probe_reuse.json`](autoinit_historical_probe_reuse.json), [`autoinit_attempt5_probe_reuse.json`](autoinit_attempt5_probe_reuse.json), [`autoinit_attempt4_probe_reuse.json`](autoinit_attempt4_probe_reuse.json) |
| spend | [`BUDGET_LEDGER.md`](BUDGET_LEDGER.md) |
