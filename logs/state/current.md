# Current state

**Updated:** 2026-09-18. The human view. Every number here has an owner named
beside it, and this file restates none of them from memory — a second
hand-maintained copy of a cost or a status is how two documents come to
disagree.

Start at [`README.md`](../README.md) if you do not know which document you want.

## Right now

**Nothing is running. Nothing is billing. No pod exists.** The last three were
the performance round's, all provider-confirmed gone; attempt 8's pod
`9s9pw0c8873y5g` was deleted after 32.32 min behind its teardown gate, GraphQL
returns `pod(9s9pw0c8873y5g) = null`, and an account-wide list returns `[]`.
**Nothing is prepared for launch and nothing may start.**

**TWO DECISIONS ARE OWED, and the second one arrived today.** The launch review
is the first. The second is that adopting the measured `state_eval`
optimization moved an evaluator hash that C2's **frozen** baseline-completion
protocol binds by content, so that protocol's gate now refuses a future B
re-measurement — correctly. No completed result is affected and the full search
is not gated on it, but amending a frozen protocol, re-measuring B, and
reverting a measured optimization are all maintainer calls. The options are
laid out under *[The full suite is not green](#the-full-suite-is-not-green-20-failures-two-families-one-of-them-new)*
and nothing was done in any of those directions.

**B WAS MEASURED, THE B→C COMPARISON EXISTS, AND IT HAS BEEN REVIEWED AND
ACCEPTED** as valid **search-stage** evidence. Baseline-completion attempt 8
(`$0.5872`) rebuilt the frozen C1 treatment baseline to its expected digest
`53e30566…`, measured it once on the frozen `state_eval@v1` suite, and computed
the preregistered comparison:
[`c2_baseline_comparison.json`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/evidence/c2_baseline_comparison.json).
Verdict `CANDIDATE_IN_A_BETTER_FRONT_THAN_BASELINE` — front 0 holds four
candidates, B sits in front 1 with one, and B is dominated on all three ranked
objectives by two candidates while dominating none. All of that evidence is now
**frozen**.

**The next C2 step changed.** The maintainer decision of 2026-09-17 **withdrew**
the preregistered local Search-2 refinement. Search-1 is kept as a
**restricted-space validation experiment**: its value is that after promoting
`attention.activation_importance_v1`, changing order and composition *alone*
produced a real structural signal on the cheap metric. That is evidence about
the **search procedure**, so the informative next step is to widen the search
rather than polish locally inside a restriction.

```text
C2 Search-1 restricted search [DONE / FROZEN]
  → C2 full joint re-search
  → Top-K / Top-5 candidate selection
  → bounded 0.86M behavioural recovery selection
  → C2 incumbent
```

**The full joint space is derived, and `576` is not the space.** Enumerating the
live registry gives **578** reachable leaves — **576** four-operator leaves plus
**2** single-step `COMPOSITE_STAGE1` leaves that reach the target directly.
Phase B's comparable space was **290**, and the growth is exactly one extra
branching factor: the promoted ATTENTION operator consumes calibration where
`attention.weight_proxy_v0` declared `CalibrationNeed.NONE` and was therefore
offered once however many mixtures were active. Nothing is pinned — every
applicable implementation, every applicable profile and every order compete, so
calibration choices can affect pruning. Owner:
[`full_search_space.py`](../../scripts/experiments/phase_c2/full_search_space.py),
with a test that refuses those integers as literals.

**One exclusion, and it is scientific, not economic.**
`attention.weight_proxy_v0` is out because **C1 is** the isolation experiment
between it and the promoted operator, and it has a completed `GO` verdict.
Re-admitting the loser would cost ~50% more search (866 leaves) to re-decide a
closed question. The cheap alternatives `depth.positional_v0` and
`composite.stage1_sandwich_v0` are **in**.

**Cost is better measured than it was.** The table pools both committed searches
and takes the per-cell maximum. `attention.activation_importance_v1` is **no
longer an unmeasured input** — Search-1 priced it at `1.5×
width.global_pca_v0`, C2 attempt 4 then ran it 14 times *below* that proxy, so
the margin was conservative in the safe direction and is retired. One cell moved
the other way and it is the one the price turns on:
`depth.causal_kl_greedy_v1` deeper, `31.10 → 36.07` min.

**C2's BEHAVIOURAL QUESTION IS NOW STATED SIMPLY.** Two rounds of review
corrected it. C1 already established the behavioural incumbent **B**: after the
frozen `0.86M` recovery, pooled `correct_overall` was `105/2550 = 4.12%` against
the old incumbent's `70/2550 = 2.75%`, a paired `+0.01372549` with a `GO`
verdict. Neither Search-1 nor the full joint search performs any recovery
training, so the only behavioural question after the search is:

> Does the selected full-search initialization **C**, after the same frozen
> `0.86M` recovery, outperform **B**?

**The original control is no longer a C2 arm.** For this comparison it answers
nothing extra: a candidate that beats the old control but loses to B must not
promote, and one that beats B gains no promotion information from it. The
guardrails use the incumbent-relative semantics C1 already used, with B in the
comparator position. That also keeps the cycle scalable — C4 and beyond challenge
whatever incumbent the previous turn left, instead of repeatedly retraining the
project's original initialization.

**Screening is now disjoint in BOTH dimensions, and it had to be.** Disjoint
recovery seeds alone are insufficient: C0's inferential unit is the **prompt**,
and it measured substantial same-prompt cross-seed dependence — ICC `0.25 ±
0.095`, `P(correct | correct on another seed) = 0.257` against a `0.022`
marginal, an `11.7×` lift. Selecting and confirming on the same prompts would
leak the selection into the confirmation however fresh the seeds were. So
[`c2_screening_v1`](../stages/stage-1/phase_c2/plans/c2_screening_battery.json)
was built and frozen: 950 prompts / 850 scorable, C1's mixture preserved exactly,
content `0ad76fc7…`, and **measured** disjoint from
`c1_confirmation_v1` by stable id *and* normalized prompt content — zero shared
on both — as well as from calibration, `state_eval`, the recovery corpus and the
reserved final-promotion battery. It produces no verdict and may promote nothing.

> **It was rebuilt once, for a real bug.** `rank_take` accepted a `domain`
> argument and did not forward it to `rank_key`, so the first build silently used
> C1's rank domain: disjoint and deterministic, but drawn under an ordering its
> own manifest did not claim. The repaired sample differs in **every** stratum
> (`c04d9d64…` → `0ad76fc7…`). Three regressions now cover it,
> each confirmed to fail with the bug reinstated — including one that re-derives
> a stratum under the declared domain and requires the frozen sample to be that
> one and *not* the default domain's, which is the provenance claim itself.

| rung | seeds | arms | battery | probes | decides |
| --- | --- | --- | --- | --- | --- |
| screening | 1 | Top-5 + B | `c2_screening_v1` | 6 | which **one** candidate advances |
| confirmation | 3 | C + B | `c1_confirmation_v1` | 6 | the C2 incumbent |

**12 probes, exact** — down from 15, and no conditional rung. The screening rule
is frozen: maximize the paired single-seed Δ`correct_overall`(candidate − B), B
as anchor, `usable_rollout` never positive credit, ties broken by the frozen
full-search ordering then the deterministic state id.

**The interpretation boundary is recorded.** The search stages are
initialization-search only and train nothing; their KL/`state_eval` output is
hypothesis-generation and candidate-selection evidence that may never promote;
C1's `4.12%` is a *post-recovery* measurement, not raw initialization accuracy;
and C2 promotion depends only on the fresh recovery comparison of C against B.

**The search execution path exists and was run for real.** The
[full-search driver](../../scripts/pod/autoinit_phase_c2_full_search_driver.py)
does `bind_identities` → `full_joint_search` → `commit_top_k` and **stops**, with
no code path into a behavioural stage. It was executed end to end at toy scale —
real operators, real checkpoints, real reloads, real measurement — which found
and closed a real defect (a `relative_to` that raises when the workdir sits
outside the repository). The behavioural session and the launcher/governance
chain are **owed at authorization time** and deliberately unbuilt: two
authorizations, never one.

**THE BLOCKER IS BUDGET, NOT DESIGN.** A funding decision is required, at the
**standing** beam width 6.

| | expected | ceiling |
| --- | --- | --- |
| full search, **beam 6 — standing design** | `$11.3878` | `$26.2606` |
| behavioural selection (12 probes) | `$20.6926` | `$29.8788` |
| full search, container disk (400 GB) | | `$1.3385` |
| behavioural selection, disk upper bound | | `$1.5229` |
| **complete standing chain, TOTAL** | **`$32.0804`** | **`$59.0009`** |
| remaining headroom | | `$72.1205` |
| **headroom after the chain's total** | | **`$13.1196`** |
| minimum cumulative cap that contains both | | `$356.8804` |

The search rows fell by `$7.2748` on **2026-09-18**, from `1826.57` bounding
minutes to `1445.54`. That is a **measured** reduction, not a re-estimate: see
the performance round below. The disk term fell with it because the pod's
billed window is its wall clock.

The two ceiling rows above are **GPU runtime only**. The provider bills
Container Disk separately at `$0.10/GB/month`, and this session
provisions 400 GB of it, so a GPU-only ceiling did not cover the
session — and no figure in the record disagreed with any other, which is
why review found it rather than a gate. The GPU rate is re-quoted live;
the storage price is a dated stated basis and
[`provider_storage_pricing.json`](../../configs/infrastructure/provider_storage_pricing.json)
says so in a field a machine reads. The behavioural session's disk term
is **bounded, not derived**: its launcher and provision do not exist yet,
so it is bounded above by the search's own 400 GB and should fall when
that session is bound.

The chain **fits the accounting envelope and is still NOT AUTHORIZED**: the
maintainer raised the cumulative cap to `$370.0000` on 2026-09-17 as an
accounting envelope, explicitly not a spend authorization and not transferable
to C3 or C4. Fitting is not permission.

Stating that minimum is **not** requesting it, and these are **not yet**
funding-decision numbers: the behavioural protocol they price has just been
repaired and is awaiting review.

**Beam 2/3/4 are priced as scientific alternatives, not as cost options.**
Narrowing the beam merely to fit the existing cap is not permitted; the scope was
not shrunk to fit.

**A mean is not doing a bound's job.** The per-probe ceiling rests on the
observed **maxima** from C1 attempt 18's per-probe marker stream — training
spread `1.003×` and is effectively deterministic, scoring spread `1.241×` and is
not, so a named `generation_length_risk` reserve funds a doubling of its observed
maximum for unseen checkpoints.

**A >1-session search is not currently available.** Search state ids are
content-derived, which gives a state an identity — not its bytes. The frozen
Search-1 plan records that the multi-gigabyte search workdir *cannot be relayed
for resume*, so a fresh provider resource must re-derive lost state. No durable
cross-session mechanism was implemented or validated this round, and none was
built: it is a possible future design option and no plan here assumes it.

**Every probe is trained fresh.** The historical-probe-reuse ruling records
`reuse_verified: false` — all eleven examined probes fail
`scoring_contract_matches_live` — so there is no admissible reuse to net off.

**The roadmap is now C1–C4**, and the repeated shape is written down once as a
family-neutral pattern in
[`OPERATOR_PROMOTION_CYCLE.md`](../../docs/OPERATOR_PROMOTION_CYCLE.md):
operator R&D → isolation → promotion → full joint re-search → behavioural
selection → new incumbent. C3 is causal-KL ATTENTION isolation on the C2
incumbent and cannot start before C2 names one; C4 is conditional on C3
promoting.

## The performance round — `$0.2252`, and the price it moved

**Closed 2026-09-18.** Four candidates, two adopted, one refused on evidence
and one instrumented to a negative finding. **No scientific semantics changed:**
the 578/576 space, beam 6 / warmup 1, both calibration profiles, every
calibration item, the 260-evaluation causal-KL greedy rule, the frozen
`state_eval` suite, the Pareto and ranking policy, Top-5 semantics and the C2c
behavioural protocol are all as they were. Evidence:
[`validations/full-search-performance/v1/closeout.json`](../stages/stage-1/phase_c2/validations/full-search-performance/v1/closeout.json)
· [`analyses/full_search_performance_round.md`](../stages/stage-1/phase_c2/analyses/full_search_performance_round.md).

| candidate | outcome | measured |
| --- | --- | --- |
| **1.** keep the `state_eval` reduction on the card | **ADOPTED** | **`76.0×`** (`1.8699` → `0.0246` ms/position) |
| **2.** DEPTH forward-KL-only hot path | **ADOPTED** | `1.10×` on the scoring loop |
| **3.** reuse reference-side normalization | **NOT ADOPTED** | needs `33.83 GiB` against a `16.91 GiB` constraint |
| **4.** diagnose reference-cache variability | **instrumented only** | `0.013 GiB` reclaimable; cache admits `67/67` |

**Equivalence is of the decisions, not of the digits.** Worst relative drift is
`3.03e-05` — `257×` below the smallest decision threshold the search is known
to use (`0.007782`), and just *under* float32's own `sqrt(V)·ε` floor of
`4.65e-05` for a `151936`-class vocabulary. Item ordering identical, top-1
agreement exact, DEPTH's removal order `[17, 18]` in three independent
measurements. A tighter bound was not achievable by any implementation: the
round's second subrun died against a `1e-06` bound that sat *below* the
arithmetic's noise, and the repair was to derive the bound from `sqrt(V)·ε`
rather than guess it.

**Candidate 4 refuted its own hypothesis.** It was instrumented to test whether
allocator hoarding explains the historical `2.6 GiB`-free observations. On a
card holding only the teacher there is nothing for `empty_cache()` to return
and the whole cache is admitted, so the answer is **live tensors elsewhere in
the search** — and the measurement has to be taken mid-search, not on a clean
card. **Nothing was flushed and no saving is claimed.** The reference-cache
recompute waste — `182,780` recomputes against `323,180` ablated forwards,
`36.1%` of every forward pass — remains the largest known saving in the search
and is deliberately **not** priced in.

**The cost model was refreshed from the measurement, not from the speedup.**
Each cell is adjusted by the component saving it actually contains, capped at
the phase that saving belongs to — never a ratio applied to a whole cell —
and every input is named in
[`phase_c2_measured_optimization.json`](../stages/stage-1/phase_c2/plans/phase_c2_measured_optimization.json).
DEPTH `36.07` → `30.79` min, the others `3.93` → `1.65` and below. Beam 6 is
unchanged and was never a lever.

**One thing for a reviewer to notice:** the protocol document *embeds its own
cost model*, so re-pricing moved its hash from `26de0bb6` to `d7678d7d`. Exactly
**two** top-level keys differ — `cost_model` and the document's own
`protocol_sha256` — which is `29` changed leaves plus the self-hash. The
science subtree hashes **identically** before and after at
`4897d470eed2b5a3…` — the document with those two keys removed and the rest
canonicalized with sorted keys, so `git show HEAD:<protocol>` against the
working tree reproduces both figures. A document
declared frozen as science should probably not move when a price does; that is
a structural remark, not a change made here.

Three subruns, all on the formal target card, all torn down
provider-confirmed, `$0.2252` of a `$1.50` ceiling — the `$0.90` soft stop was
never reached. Each failure was in the instrumentation rather than in the
optimizations, and each is written down with its root cause.

## Getting here cost three aborted sessions and `$0.1033`

The maintainer decision of 2026-09-17 authorized two formal sessions to establish
B, measure it once on the frozen suite, and compute the preregistered B→C
comparison. **Both were consumed without reaching a rebuild.**

**The retry rule then changed shape.** The decision of 2026-09-16 continued the
work, raised **no** envelope, and prospectively replaced the two-session limit
with a **money** boundary: a fresh formal chain requires
`cumulative completion spend + $1.1950 <= $2.3900`. There is no fixed maximum
attempt number, and an incrementing attempt number is not a scope expansion — a
cheap pre-measurement abort consumes its actual cost and its one-use chain,
nothing more. The ceiling, the `$1.09/h` L40S boundary and every frozen
scientific identity stayed unchanged throughout; the project cap was
`$320.0000` for those attempts and is now `$370.0000`, owned by
`configs/experiments/phase_c1/authorization.json ::
accepted_pricing.cumulative_cap_usd`. Attempts 7
and 8 both ran under that rule, and it is what let the work finish without
another approval round.

| attempt | where it stopped | cost |
| --- | --- | --- |
| [5](../stages/stage-1/phase_c2_baseline_completion/runs/attempt5/closeout/outcome.json) | the launcher's **first statement**: `claim_output_root` took the stage id positionally where the signature takes `outputs` by keyword. No gate ran, no price was queried, **no provider resource existed**. The chain was consumed anyway — its one-use rule counts the invocation | `$0.0000` |
| [6](../stages/stage-1/phase_c2_baseline_completion/runs/attempt6/closeout/outcome.json) | **10/10 `$0` gates passed** and setup refused at **`ROPE_OK`**, which globs `artifacts/stage1/*/checkpoint/config.json`. This session stages no checkpoint — it rebuilds B on the pod — so the step had nothing to look at. `SETUP_RC=1`, no driver stage | `$0.0412` |
| [7](../stages/stage-1/phase_c2_baseline_completion/runs/attempt7/closeout/outcome.json) | **10/10 `$0` gates passed twice**, the pod came up and SSH answered — and the **launcher process was killed two minutes in**, by the agent's own blocking tool call. Setup never ran; `stages` is `{}`. The pod outlived its orchestrator and an explicit provider query removed it | `$0.0621` |
| [8](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/closeout/outcome.json) | **COMPLETE.** `SETUP_RC=0`, driver detached and confirmed by descriptor probe, both stages passed, B rebuilt to its expected digest, measured once, comparison computed. Pod deleted behind its teardown gate | `$0.5872` |

**Attempt 7 was not a repository failure, and recording it as one would hide the
real defect.** Every gate passed, the live price sat exactly on the `$1.09/h`
boundary, the bundle round-tripped to the authorized commit. The launcher was
started from a single tool call that also contained a foreground `sleep`, which
that harness blocks; the call ran to its two-minute timeout and was killed, and
the kill took the whole process tree — the `setsid`-detached launcher included.
**`setsid` defeats a process-*group* signal, not a supervisor that walks
descendants.** Worse, the watchdog was inside the tree it was meant to outlive:
its journal has exactly two polls, `13:10:28Z` and `13:11:29Z`, so the recorded
65.78-minute hard terminate could never fire, and this project has never once
seen the provider's own `--terminate-after` fire either. Attempt 8 starts the
launcher inside a **`tmux`** server — not a descendant of the starting call —
and that call returns immediately. No repository code changed, no gate was added,
and the closure and all six frozen identities are byte-identical.

Attempts 5 and 6 share a different class, and it is the one this repository keeps
paying for: **an inherited declaration rather than an inherited need.** C1
attempt 2 died on that same `ROPE_OK` line for `$0.1013`.

The repair does not drop the marker's job. `transformers` 4.x reads the flat
`rope_theta` where 5.x records the nested one and the two disagree by 500×, so a
loader taking the wrong field would give B a different positional basis and a
silently wrong `state_eval` — the one number the session exists to produce. The
guard **moved to where the artifact exists**: onto the rebuilt B, in the
interpreter that measures it, after materialization and before the measurement,
through the same two helpers the setup step uses. Its reading is carried into the
durable measurement block.

The six identities the decision froze by value are **unchanged**, derived through
the production assembler and now asserted by a regression — the completion
closure moved, as that decision said it would, so a moved closure is no longer
the only visible difference between a permitted repair and a change to closed
science:

| artifact | identity |
| --- | --- |
| frozen candidate side of B→C | [`c2_frozen_comparison_inputs.json`](../stages/stage-1/phase_c2/runs/attempt4/evidence/c2_frozen_comparison_inputs.json) · `55f6677067392fd0…` |
| completion protocol | [`phase_c2_baseline_completion_protocol.json`](../stages/stage-1/phase_c2/plans/phase_c2_baseline_completion_protocol.json) · `9f566eb6f8d71d57…` |
| completion pricing | [`phase_c2_baseline_completion_pricing.json`](../stages/stage-1/phase_c2/plans/phase_c2_baseline_completion_pricing.json) · `dcc64bcf9b3dc9fb…` |
| the reserve defect | [`c2_baseline_reserve_defect.json`](../stages/stage-1/phase_c2/analyses/c2_baseline_reserve_defect.json) · `f389350cca7782ca…` |
| completion executable closure | [`c2_baseline_completion_closure.json`](../stages/stage-1/phase_c2/analyses/c2_baseline_completion_closure.json) — **derived live and expected to move; the digest is pinned there, not restated here** |

The **driver and a thin formal launcher** were repaired against seven execution
defects that would each have surfaced only after B had been rebuilt and measured
— the suite root, an unprimed evaluator, a second teacher, a caller-supplied
search identity, the wrong ranking citation, a nested verdict key, and a record
mutated after its own hash — then against six authorized enforcement repairs,
then against the two failures above. An eighth, a missing `SESSION_KIND` branch
in the shared setup script, would have exited 98 on a billing pod. Every one is
held by a mutation-verified regression, and the chain machinery has now been
executed end to end at `$0`: entry path, all ten gates, closeout.

The authorization is a **distinct type** reporting `authorizes_c2_search1 =
False`, and the two loaders refuse each other's schemas, so a completion grant
cannot buy a beam. The derived closure contains no Search-1 module.

The completion is priced at **`$0.7212` expected and a `$1.1950` hard ceiling**
(39.70 / 65.78 min at `$1.09/h`) — derived from attempt 4's own telemetry, not
from the reserve that failed. **No budget increase is requested**, the rate
boundary is unchanged, and the old Search-1 pricing document is preserved
exactly as the authorization basis attempts 2–4 ran under.

**Attempt 7 runs under the money rule, and it stops at the measurement.** A
failure *before* the durable `baseline_measurement` exists is handled
autonomously — teardown with provider-confirmed zero billing, preserve,
reconcile, diagnose, minimal repair, minimal regression, fresh identity, fresh
chain — for as long as `spend + $1.1950 <= $2.3900` holds, and an identical
unchanged failure is never retried. The instant that measurement exists the
authority inverts: **no GPU remeasurement of B**, no second `state_eval`, and a
downstream comparison failure is repaired at `$0` from the durable measurement
plus the frozen five. `$2.3488` of the `$2.3900` envelope is unspent and that is
still not permission for anything outside this scope.

**One open question belongs to the maintainer.** Cross-session numerical
comparability is preserved structurally — every checkpoint is scored
independently against the original teacher under `RECOMPUTE`, with no
candidate normalized against another — but its *magnitude* is unmeasured: no
state was ever measured twice anywhere in this project, the per-measurement
`runtime` block is empty, and the image name does not pin the host driver
(attempt 3 saw `595.91.07`, attempt 4 `580.126.09`). The five candidates are
separated by 78–1273 epsilons, so the front structure is not balanced at that
scale; the protocol therefore pre-registers a disclosure rule, before B exists,
for any B→C margin at or below the tightest observed gap of `0.007782`.

Attempts 2 and 3 aborted **before** the beam for `$0.0552` and `$0.1674` and
measured nothing; both root causes are repaired and both repairs were confirmed
on real hardware by attempt 4 — setup and stage A passed, and the artifact
collection that raised `ArtifactError` in attempt 3 returned `manifest rc=0`
with all 7 files and `missing: []`.

**C1 is COMPLETE.** Attempt 18 executed the whole frozen protocol — both replay
gates, both arms, six probes trained, six evaluated on the frozen battery — and
the frozen Stage-I rule returned **`GO`** at `$10.2018`, under its own planning
floor. A complete valid verdict ends the round.

| | | owner |
| --- | --- | --- |
| phase | C1 — fixed-path ATTENTION isolation, **CLOSED by a `GO` verdict**. C2 Search-1 is **DONE and FROZEN** and its B→C comparison has been **reviewed and accepted** as search-stage evidence; the local Search-2 refinement is **WITHDRAWN**, and C2 now continues as a **full joint re-search → Top-5 → behavioural selection under Phase-C/C1 discipline**, designed and priced but **NOT FUNDED**. C3 (causal-KL isolation on the C2 incumbent) and C4 (conditional re-search) are not started | [`phase_c2/plans/phase_c2_full_search_protocol.json`](../stages/stage-1/phase_c2/plans/phase_c2_full_search_protocol.json) · [`phase_c1/plans/phase_c_roadmap.md`](../stages/stage-1/phase_c1/plans/phase_c_roadmap.md) |
| replay | **MEASURED — 2/2 PASS**, for the third time (attempts 9, 17, 18). Passing replay is not a result: 9 and 17 are **NO DECISION**, pre-treatment aborts that measured no endpoint. Attempt 18 is the only attempt that decided anything | [`attempt18/closeout/outcome.json`](../stages/stage-1/phase_c1/runs/attempt18/closeout/outcome.json) |
| treatment, endpoint | **MEASURED** — six probes trained and six evaluated on the frozen battery; the frozen Stage-I rule returned **`GO`**. Figures in the block below | [`attempt18/evidence/c1_decision.json`](../stages/stage-1/phase_c1/runs/attempt18/evidence/c1_decision.json) |
| launch chain | **every C2 chain is consumed and nothing is prepared** — Search-1 attempts 1–4 and completion attempts 5–8. No further C1 attempt is authorized or prepared either; a complete verdict ended that round. The next chain cannot be built until the full search is funded | [`phase_c2_baseline_completion/runs/attempt8/governance/`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/governance/) |
| last attempt | **baseline completion attempt 8 — COMPLETE, `$0.5872`.** Both stages passed, B was rebuilt to digest `53e30566…`, measured **once** on the frozen suite, and the B→C comparison was computed; the pod was deleted behind its teardown gate after 32.32 min. Attempts 5, 6 and 7 aborted before any measurement for `$0.1033` between them | [`attempt8/closeout/outcome.json`](../stages/stage-1/phase_c2_baseline_completion/runs/attempt8/closeout/outcome.json) |
| blocker | **THE FORMAL FULL-SEARCH CHAIN IS BUILT AND UNCONSUMED; THE LAUNCH REVIEW IS OWED — and so is a frozen-record decision the performance round created, see the suite section.** The cap rose to `$370.0000` on 2026-09-17, so the standing beam-6 chain's `$59.0009` of ceilings — GPU **and** separately billed disk — FITS the `$72.1205` remaining with `$13.1196` to spare, an accounting envelope, **not** a spend authorization and not transferable to C3/C4. That headroom roughly doubled on 2026-09-18 because the measured performance round took the search's bounding minutes from `1826.57` to `1445.54`; the beam width was not touched. C2c asks only whether the selected C beats the incumbent **B** after the frozen 0.86M recovery, on **12 probes**. The real-GPU engineering validation of the search driver is **PASSED and CLOSED** for `$0.1453`, and the chain it validated is now built — with three repairs an independent review found before any pod existed: the shared setup script had **no `c2_full_search` authorization branch**, so a formal pod would have completed paid setup and the whole test gate and then refused; the storage walk counted parents, children and leaves but **not the expanded ancestors the search never releases**, understating the peak; and the ceiling was **GPU runtime only** while the launcher provisioned hundreds of GB of separately billed disk. The chain is: launcher, grant contract, launch-bound readiness, one-use authorization type and issuer, derived executable closure over 90 files, bundle transport, artifact contract and live budget position, with eleven `$0` gates and 45 exercised checks. The live securePrice was re-quoted at `$1.09/h` — **unchanged** — so the beam-6 ceiling rests on re-derived minutes rather than on a coincidence: `$26.2606` GPU plus `$1.3385` disk, `$27.5992` for the session. **No grant is approved, so no authorization is issued and nothing is consumed.** What a review reads is [`phase_c2_full_search_grant_proposal.json`](../stages/stage-1/phase_c2/plans/phase_c2_full_search_grant_proposal.json), which approves nothing and which the issuer refuses. Narrower beams are scientific **alternatives**, never a way to fit the cap. Nothing may start — the search session, the behavioural session, the withdrawn Search-2, C3 and any remeasurement of B each need a decision, and the two C2 sessions need **separate** authorizations | [`phase_c2_full_search_pricing.json`](../stages/stage-1/phase_c2/plans/phase_c2_full_search_pricing.json) · [`budget/decisions.md`](../budget/decisions.md) |
| spend | owned by the budget block below | [`budget/ledger.md`](../budget/ledger.md) |

## Readiness

<!-- readiness:begin -->

| readiness | | owner |
| --- | --- | --- |
| latest POINTED-TO sweep — C1 attempt18 | **launch_bound — PASS**, swept at `e80eb60b`; **does not describe the current tree** | [`c1_pod_environment_verification.json`](../stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json) |
| every other experiment's readiness | **run-owned and not pointed at from here** — one record per attempt, under that attempt's `governance/readiness.json`, so a later sweep cannot overwrite what an earlier one launched under | [`stages/stage-1/`](../stages/stage-1/) |
| launch-bound for the next session | **not prepared** — a launch-bound sweep on the final clean pre-authorization tree is owed | this file's launch-chain section |
| last launch-bound failure | swept at `82745981` on 2026-09-12 — kept as history, not a current state | [`readiness_history.json`](../stages/stage-1/phase_c1/history/readiness_history.json) |

*Generated from the record by `scripts/consolidate/render_log_navigation.py`; do not edit by hand — it went stale within hours when it was prose.*

<!-- readiness:end -->

## The full suite is not green: 20 failures, two families, one of them new

Measured on the settled tree, 2026-09-18, over
`tests/{docs,architecture,autoinit,initialization,validation,init,runtime}` plus
every `tests/pod` file these touch: **20 failures**, `3381 passed` in the
widest run. Every one fails in the direction that **refuses** rather than
permits, and each is listed by nodeid below so a reader can check the count
rather than take it.

| family | tests | why it stays red |
| --- | --- | --- |
| **A committed digest no longer describes the live tree.** Phase B's amendments ledger accounts to `c20e3a80b6c0` while the tree digests to `c9121aadff77`; C1's preregistration, readiness record, skip-predicate audit and preflight selection are in the same position. ~37 commits since 2026-09-15 touched files in those declared sets, shared runtime mostly — and this round's three core-file changes are among them | 14 | The gates return `ok = False`, so a paid Phase-B or C1 launch is **refused** — correct. The remedy each message names is *"re-freeze it"*, which edits a frozen scientific record, and no C1 or Phase-B launch exists to justify a `launch_bound` sweep (AGENTS.md P8.3). Reconstructing ~20 ledger entries protects no current experiment |
| **NEW — the evaluator identity a frozen protocol binds has moved**, because candidate 1 was adopted | 6 | See below. It arrived **today** and is the second thing the launch review owes |

<details><summary>the 20, by nodeid</summary>

```text
# a committed digest no longer describes the live tree (14)
autoinit/test_phase_b_historical_amendments.py::test_the_ledger_verifies_against_the_live_tree
autoinit/test_phase_b_historical_amendments.py::test_an_incorrect_source_commit_is_refused
autoinit/test_phase_b_historical_amendments.py::test_a_missing_changed_file_is_refused
autoinit/test_phase_b_historical_amendments.py::test_a_false_numstat_is_refused
autoinit/test_phase_b_historical_amendments.py::test_a_false_after_file_hash_is_refused
autoinit/test_phase_b_historical_amendments.py::test_a_false_patch_hash_is_refused
autoinit/test_phase_b_plan.py::test_completed_phase_b_drift_is_historically_accounted_for
autoinit/test_skip_predicate_audit.py::test_the_committed_audit_record_matches_the_live_one
autoinit/test_staging_contract.py::test_c1_runs_only_its_own_preflight_on_a_paid_pod
autoinit/test_c1_readiness_gates.py::test_the_committed_record_still_binds_the_live_executable
autoinit/test_c1_readiness_gates.py::test_the_pod_selection_is_exactly_the_preflight_directory
pod/test_c1_session_contract.py::test_the_writer_refuses_to_rewrite_the_frozen_preregistration
pod/test_continuation_b_one_probe_contract.py::test_the_preregistration_binds_the_live_executable_digest
pod/test_phase_b_driver_and_launcher.py::test_the_preregistration_gate_refuses_a_tree_the_freeze_does_not_describe

# the evaluator identity a frozen protocol binds has moved (6)
pod/test_phase_c2_baseline_completion.py::test_the_protocol_binds_the_identities_it_claims_to
pod/test_phase_c2_baseline_completion.py::test_stage_b_orchestration_is_correct_end_to_end
pod/test_phase_c2_baseline_completion.py::test_exactly_one_state_eval_is_performed
pod/test_phase_c2_baseline_completion.py::test_a_close_baseline_is_flagged_without_changing_the_verdict
pod/test_phase_c2_baseline_completion.py::test_the_b_measurement_survives_a_post_measurement_failure
pod/test_phase_c2_baseline_completion.py::test_the_rope_reading_is_carried_into_the_durable_measurement
```

</details>

**Attribute against the commit the session started from, not `HEAD`.** Eleven
architecture-guard cases were red in *both* the working tree and a detached
worktree at `HEAD`, which read as pre-existing — they were mine, from a commit
two back in the same session that changed three core files without declaring
them. And a worktree has no untracked files, so four of the six new failures
looked pre-existing there for an unrelated reason (`artifacts/stage1/
state_eval_v1` is simply absent). Both traps were hit in one afternoon.

### The new family, and the decision it needs

Adopting the `state_eval` reduction changed
`src/aadistill/initialization/planning/metrics.py`, whose hash moved
`a6dd5d56…` → `d193cc90…`. That file is **one of four bound by content** in

```text
logs/stages/stage-1/phase_c2/plans/phase_c2_baseline_completion_protocol.json
  :: cross_session_comparability_contract.bound.evaluator_implementation_sha256
```

which is a **frozen record of completed work** — attempt 8's B measurement and
the B→C comparison. The other three are unchanged. `bind_identities` in the
baseline driver now refuses:

> *the evaluator implementation has moved since the candidate side was frozen …
> STOP: the frozen C measurements and a new B measurement would not be the same
> measurement series, and that is not something to compensate for.*

**That is the gate working.** What it is protecting, precisely:

* **No completed result is affected.** Both sides of every finished comparison
  — Search-1's C candidates and attempt 8's B — were measured by *one*
  implementation. `c2_baseline_comparison.json` stands as it is.
* **The full joint re-search is not gated on this** and does not bind the
  evaluator by hash: it rescores all 578 leaves with one implementation, so it
  is internally consistent.
* **What is refused is a *future* B re-measurement joining the old series** —
  which the maintainer has already barred without a new decision, so nothing
  currently planned is blocked.
* The measured disagreement between the two implementations is `3.03e-05`,
  **257× below** the smallest decision threshold the search is known to use.

Three ways forward, and **all three are the maintainer's call**, not an
autonomous repair (AGENTS.md P12.1 — changing a frozen scientific protocol is
an explicit stop condition):

1. **Amend the contract** to name both hashes with the measured equivalence as
   the stated justification. Cheapest, and it edits a frozen record.
2. **Re-measure B** with the new evaluator. Scientifically cleanest, costs a
   GPU session, and is currently barred.
3. **Revert candidate 1.** Forfeits the measured 76×, and with it the
   `$7.2748` the search ceiling fell by.

Nothing was done in any of those directions. The gate is left firing, the six
tests are left red, and the optimization is left adopted with its equivalence
measured — because loosening the guard to make the suite green is exactly the
move the guard exists to prevent.

A permanently red suite is a hazard — it is what let 14 of an earlier 28 sit
unnoticed at remote HEAD — so both families are named here, by nodeid, rather
than left for the next reader to re-derive.

## Budget — four limits that do not transfer

Derived by [`scripts/consolidate/derive_budget.py`](../../scripts/consolidate/derive_budget.py)
from the approved package and each session's own closeout. **Do not restate
these by hand; run the deriver.**

<!-- budget:begin -->

| limit | remaining |
| --- | --- |
| formal sessions | `$22.8249` of `$45.4425` |
| GPU engineering | `$6.0000` of `$6.0000` |
| package | `$28.8249` of `$51.4425` |
| project cap | `$297.8795` spent of `$370.0000`, leaving `$72.1205` |

**Full-ceiling sessions the FORMAL allowance funds: 1.** 2 ceilings cost `$30.2950` and the formal allowance has `$22.8249`. Dividing the PACKAGE balance instead gives 1, which is the error: the engineering allowance cannot pay for a formal probe.

*Generated by `scripts/consolidate/render_log_navigation.py` from `derive_budget.py`; do not edit by hand.*

<!-- budget:end -->

Remaining balance is not permission.

## The C1 result

The verdict and every figure behind it live in
[`runs/attempt18/evidence/c1_decision.json`](../stages/stage-1/phase_c1/runs/attempt18/evidence/c1_decision.json).
Cited, not restated from memory:

| | |
| --- | --- |
| verdict | **`GO`** |
| delta, `correct_overall` | `0.013725` |
| one-sided LCB | `0.005490` — above zero |
| two-sided CI | `[0.004314, 0.023137]` |
| SESOI | `0.010` — the point estimate clears it |
| seed robustness | 3 of 3 positive, 2 required |
| guardrails | passed, **no vetoes** |
| per-seed delta | `0.01529`, `0.00941`, `0.01647` |

**Read it with care, and read the record.** Absolute correctness is low on both
arms: of 850 scorable prompts the incumbent scores 18 / 26 / 26 and the treatment
31 / 34 / 40. A 1.37-point delta on a ~2.7% base is a large relative change over
a small absolute one. `usable_rollout` covers 555–605 prompts of 850 for the
incumbent and 582–605 for the treatment, so roughly a third of the battery
produces no usable rollout on either arm.

The decision record states its own claim boundary: *prompt-distribution
uncertainty conditional on the three preregistered fresh recovery-seed
checkpoint pairs — not a CI over hypothetical future recovery seeds.* This is
selection evidence about one operator under a fixed 0.86M-token recovery budget.
It is not a capability claim and not a statement about a trained model, and
**nothing has been added to the README Optim record**: an official record needs
the full §3.8 package and maintainer approval.

## Two engineering defects attempt 18 exposed

Neither affects the C1 result, and both are held separately from it.

**1. The relay stream copy of `c1_evidence.json` arrived corrupted — REPAIRED.**
The driver rewrites that document on every state change; the relay mirrors files
by byte offset because its other streams are append-only. It had synced 11,343
bytes of an early version, the driver replaced the file with a 23,425-byte one,
and `tail -c +11344` appended the new document's tail to the old document's
head. Exactly the right size, and not JSON.

`RelaySpec` now carries `whole_file`, and such a spec is written by temp file
plus atomic `os.replace` — the local copy becomes exactly the new bytes or is
left alone — and refuses rather than writing a document truncated at the chunk
cap. The evidence document is the one spec that declares it; the event streams
are still appended, because re-reading a growing train log every poll is what
the offset scheme exists to avoid.

Two regressions cover it, and both were confirmed by mutation: a long document
replaced by a shorter one leaves exactly the shorter one and parses; and a
`whole_file` spec ignores a *stored* offset rather than merely never writing
one — the first version of the fix passed every other test with that guard
removed, and an offsets file written by the attempt-18 relay carries `11343`
for exactly this path.

The closeout also noted that **the collector still prefers the stream copy**.
That is no longer a defect and needs no change: the preference was only
dangerous because the preferred copy could be corrupt. The runner performs a
final `sync_once` after the terminal marker, which for a whole-file spec is a
complete re-read, so the stream copy is now the finished document.

**2. The probe-durability mechanism preserved nothing — NOT repaired, and not
mine to repair.** Added before this run so completed probes survive a later
failure, it ran on all six and every upload was refused: *"Private repository
storage limit reached"*, 2.22 GiB per probe. The mechanism behaved correctly —
never raised, disturbed no stage, recorded each probe's identity, seed, config
hash, content hash and the exact reason — but the bytes are gone with the pod.
It cost this run nothing because the run succeeded; had stage H failed again,
six probes would have been lost a second time.

What it needs is a durable large-artifact backend with capacity. The private
quota is account-wide, and freeing it means permanently deleting historical LFS
objects or changing a paid plan — a maintainer decision either way. The
requirement is recorded for future long experiments in AGENTS.md P8.2.1.

**It does not block C2 Search-1.** Search-1 trains no probes and exports no
checkpoint: its candidates are measured on the pod and their identities come
home in the search journal, which is kilobytes. The capacity decision becomes a
precondition only for a later behavioural-confirmation experiment, which would
produce six 2.22 GiB probes and is separately authorized.

## The launch chain — nothing is prepared; the next one waits on funding

**No chain is owed and none is prepared.** C1's round ended with a verdict and
C2's baseline completion ended with a measurement. The seven steps below have
now been executed four times end to end (completion attempts 5–8), so the
ordering below is not theoretical — and the next chain cannot be built until the
full joint re-search is funded. The readiness
block above says the latest sweep does not describe the current tree; that is
correct and is not a debt — a `launch_bound` sweep describes the tree a launch
will use, so it is run once, when a launch is actually imminent (AGENTS.md P8.3).

One authorization funds one launcher session: up to three acquisition draws
inside it, never two billing resources, all sharing that session's single
ceiling. The ordering constraints, which cost real money to learn:

1. the run's **grant**, committed on a clean tree
2. a **`launch_bound` sweep** on that clean pre-authorization tree
3. commit **ONLY the readiness record**
4. the one-use **authorization**, issued against that clean commit. The issuer
   refuses a dirty tree by default since attempt 15 skipped step 3
5. commit **ONLY the authorization artifact**
6. the exact-session **bundle**, staged with `--run-id`
7. a live quote, every pre-provider gate, the single launch

Steps 3 and 5 are separate commits because `session_commit_and_lineage` permits
exactly one tracked path to differ between the authorized base and the session
commit. Attempt 15 combined them and was refused at `$0`.

Step 2 must follow step 1: `verify_record` permits exactly two tracked paths to
differ after a sweep — the readiness record and the issued authorization — so a
grant committed after a sweep invalidates it.

A chain is consumed by the launcher's invocation, whether or not a provider
resource followed, and is never reused. A launcher invocation that aborts before
formal training is an engineering subrun: it is closed, repaired and retried
under a fresh chain without a further approval (AGENTS.md P12.1). That exception
governs retries **before** measurement and never after a complete verdict.

Full terms — attempt counting, the six retry conditions, the stop list:
`execution_package` in
[`../configs/experiments/phase_c1/authorization.json`](../../configs/experiments/phase_c1/authorization.json).
Those terms are C1's. A C2 session would need its own grant and its own ceiling;
neither the project headroom above nor C1's unused formal allowance is
authorization for one.

## What ends a round

A complete `GO`, `NO-GO` **or** `INCONCLUSIVE` all end it. `INCONCLUSIVE` is a
result and is never re-run in pursuit of a `GO`.

Stop and report if: the first probe has started training, or whether it started
cannot be confirmed; a real replay mismatch at stage D or E; an input-identity
conflict; a limit reached; or a resource whose billing state is unknown.

## The log tree: stage-first, and every experiment attributed

`logs/` is **Stage → Experiment → Run**. Four stages have repository evidence
and therefore exist — stage-0 and stage-2 as pipeline activity with no
experiment-run logs, stage-1 and stage-3 with both.

Every historical experiment has a stage, rebuilt from repository facts and
checked rather than asserted: [`stages/index.json`](../stages/index.json) is the
stage index. It carries the evidence for each assignment and the rules that
decided it, and it also says what each stage is *for*, what it consumes and
produces, and where its canonical configs, data manifests and artifacts live —
so `logs/stages/` is the pipeline's stage-level entry point rather than a run
container. **20 experiments, 20 in one stage, 0 genuinely cross-stage, 0
unresolved.** There is no `cross-stage/` directory, because no experiment takes
several pipeline stages as its subject; a run whose stage nobody declared is
refused rather than shelved.

Five Stage-3 experiments ran and produced no log files of their own — `ttb`,
`p0_real`, `d0`, `p0`, `p2`. They are listed in
[`stages/stage-3/`](../stages/stage-3/) with their configs, artifacts and index
sections, and given no directory: an empty one would assert material that does
not exist. `e2` now has one, holding the single document it left behind.

`migrations/` and `archive/` are gone. A superseded document is deleted, since
git history holds it; a historical document that is still part of the record —
a preregistration, a consumed authorization, the pre-layout experiment
chronology — sits under the experiment or stage that owns it. The few old paths
current tooling must still resolve are a flat table in
[`index.json`](../index.json)`.historical_paths`.

One declared exception: `shared/analyses/autoinit_*` and four
`shared/validations/` directories are **Stage-1 material, not stage-neutral**.
They stay where they are because frozen pod drivers read those exact paths —
recorded in the stage index with its blocker rather than left looking ownerless.

This changed no experiment result, no authorization and no frozen evidence, and
the launch chain is unaffected by it.

## Engineering, not results

The initialization migration and the two CUDA validations are engineering
records. The stage-F device repair is **CONFIRMED ON REAL CUDA** at execution
SHA `7027a8f4`. Neither is a C1 result and neither authorizes anything:
[`maintenance/source-relocations/initialization-core/v1/`](../maintenance/source-relocations/initialization-core/v1/) ·
[`phase_c1/validations/cuda-stage-f/v1/`](../stages/stage-1/phase_c1/validations/cuda-stage-f/v1/)

The **C2 full-search driver** is likewise **CONFIRMED ON REAL CUDA**: one
L40S (cc 8.9, bf16, torch 2.9.1+cu130) drove all three driver stages to
`ALL_DONE` over the real 578-leaf joint space, and every one of the 35
materialized states reloaded on `cuda` in `bfloat16`; the real 1024x28
student (596,049,920 parameters) built, saved, reloaded canonically and kept
its rope base of `5000000.0` and its tied head, peaking at 1.118 GiB. Three
subruns, `$0.1453` of a `$0.9000` ceiling, every teardown provider-confirmed.
Two of the three failed first, each on a different producer of non-source
input, which is now derived from code rather than listed. It measures no
behaviour and authorizes nothing, least of all the formal search:
[`phase_c2/validations/full-search-cuda/v1/`](../stages/stage-1/phase_c2/validations/full-search-cuda/v1/)

The **performance round** is a third engineering campaign on the same card:
`$0.2252` of a `$1.50` ceiling, three subruns, all torn down
provider-confirmed. It changed no science, adopted two optimizations on
measured equivalence, refused one on memory evidence and refuted candidate 4's
hypothesis. It is what the current price rests on — see the section above:
[`phase_c2/validations/full-search-performance/v1/`](../stages/stage-1/phase_c2/validations/full-search-performance/v1/)

## History

This file holds the current state only. The narrative lives with what it is
about, and is unedited:

* [`phase_c1/history/operational_history.md`](../stages/stage-1/phase_c1/history/operational_history.md)
  — every C1 session, its cost, failure and repair
* [`stages/stage-3/history/EXPERIMENTS.md`](../stages/stage-3/history/EXPERIMENTS.md)
  — the pre-layout experiment chronology, and the only record several Stage-3
  experiments have
* [`experiment_index.md`](experiment_index.md) — what each experiment proved,
  what it does **not** support, and which conclusions still bind
* [`phase_index.md`](phase_index.md) — the same history organized by phase
* [`decisions.md`](../budget/decisions.md) — decision records
* [`budget/ledger.md`](../budget/ledger.md) — every cost, per session
