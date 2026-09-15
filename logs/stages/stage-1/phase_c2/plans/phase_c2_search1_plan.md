# Phase C2 — Search-1: ATTENTION-aware composition/order re-search

**Status: IMPLEMENTED AND PRICED · NOT AUTHORIZED · NO COMPUTE · NOTHING IS
PREPARED FOR LAUNCH.**

This page is a plan and a price. It authorizes nothing, and a review verdict is
not a spend authorization. The surrounding Phase-C structure is
[`../../phase_c1/plans/phase_c_roadmap.md`](../../phase_c1/plans/phase_c_roadmap.md),
which owns the C0/C1/C2 shape; this page owns C2 Search-1 only.

**C1 is closed and stays closed.** Its verdict was `GO` at `$10.2018` and a
complete valid verdict ends a round. Nothing here reruns it, adds a seed,
enlarges its battery or reopens the operator-selection experiment.

---

## The question, and the separation that makes it answerable

> With ATTENTION fixed to the C1-selected `attention.activation_importance_v1`,
> does re-optimizing the initialization path — the operator ORDER, ATTENTION's
> position in it, and ATTENTION's calibration profile — produce a better
> initialization than the frozen C1 treatment?

Three labels, used consistently below and kept apart on purpose:

| | what it is |
| --- | --- |
| **A** | the Phase-B winner `fe9683e6a9c783bbc6fe276a78c851c6`, artifact `c313d1b4081b…`, with `attention.weight_proxy_v0`. A **historical frozen anchor** |
| **B** | the C1 treatment: the same fixed path with `attention.activation_importance_v1` @ `calib.domain_balanced@v1`, spec `3a233a9017b3…`. The **baseline C2 is measured against** |
| **C** | a C2 Search-1 candidate: a different order and/or a different ATTENTION profile |

* **A → B is already frozen C1 evidence** for the ATTENTION operator change,
  under three preregistered paired recovery seeds and the frozen battery. It is
  not re-measured.
* **B → C is the new C2 question** about re-optimizing composition and order
  around that operator.
* **A → C would be a chained Phase-C contrast**, and must be reported as a chain
  — explicitly *not* a simultaneous three-arm estimate. `A` and `B` were
  measured against each other; `B` and `C` would be measured against each other;
  no design measures all three together, and presenting one number as if it did
  would reintroduce exactly the ambiguity Phase C exists to remove.

`attention.causal_kl_v1` is **not** a C2 prerequisite and is not implemented. It
is a separate future ATTENTION-R&D hypothesis.

---

## The space

Owned by [`scripts/experiments/phase_c2/search_space.py`](../../../../../scripts/experiments/phase_c2/search_space.py),
not by this page. Every identity in it is re-derived from committed evidence by
`tests/autoinit/test_phase_c2_search_space.py`; the values are repeated here to
be read, and that module is what a run would load.

```text
PYTHONPATH=src:scripts python -m experiments.phase_c2.search_space
```

**Four operator kinds, one implementation each, order free.**

| kind | implementation | calibration profile(s) |
| --- | --- | --- |
| DEPTH | `depth.causal_kl_greedy_v1` | `calib.domain_balanced@v1` — held |
| FFN | `ffn.activation_importance_v0` | `calib.domain_balanced@v1` — held |
| RESIDUAL_WIDTH | `width.global_pca_v0` | `calib.reasoning_heavy@v2` — held |
| ATTENTION | `attention.activation_importance_v1` | **both**, branched |

The three held mixtures are the Phase-B winner's own, read off its path label
rather than assumed. ATTENTION cannot inherit the incumbent's `calib.none@v1`:
that sentinel belongs to `attention.weight_proxy_v0`, which declares
`CalibrationNeed.NONE`; the replacement consumes activation statistics and
therefore has a real choice to make for the first time.

**Order is searched, and DEPTH is NOT pinned to position 1.** All 4! = 24
permutations are reachable, times two for ATTENTION's profile, so 48 complete
leaves exist in principle and the beam visits a subset. A DEPTH-first
enumeration would be a restricted screening experiment, not this question.

**The Phase-B implementation search is not reopened.**
`attention.weight_proxy_v0`, `depth.positional_v0` and
`composite.stage1_sandwich_v0` are absent from `allowed_impls`. That absence is
the only way to say it: `BeamSearch._allowed_impl_ids` falls back to the entire
registry when `allowed_impls` is `None`, which is how Phase A and Phase B came
to search everything registered.

**B is a leaf of this search, not an injected checkpoint.** Its order and its
ATTENTION profile are both inside the space, the search seed is the frozen
`20260815`, and a deterministic search re-derives a prior state byte for byte —
so the baseline is measured on the same cheap metric as every candidate, in the
same run, without importing anything.

### The baseline's availability is a real risk, and injection cannot fix it

**There is no B checkpoint to inject.** `configs/autoinit/c1_artifacts.json`
collects twenty-two artifact classes and not one of them is a weight file: C1
came home with `c1_arm_identities.json` — the incumbent's `c313d1b4081b…` and
the treatment's `53e30566c5f7…` — and no bytes. So `retained_candidates`, the
mechanism Phase B used to inject two finalists by digest, has nothing to stage.

B therefore reaches the comparison only if the beam reaches it. Phase B's
ranking did retain exactly this prefix — `DEPTH(db)→FFN(db)` was kept at level 1
and `fe9683` was a level-3 leaf — so the evidence is favourable, but a beam of 6
out of 18 is not a guarantee.

**Resolved 2026-09-16, and no longer discretionary.** The fallback is
implemented, preregistered here, and priced with its own named reserve. It is
`BaselineFallback` in
[`scripts/experiments/phase_c2/baseline.py`](../../../../../scripts/experiments/phase_c2/baseline.py),
and the rule has no judgement in it:

1. **If the search generates and evaluates B, that candidate IS the baseline and
   nothing is rebuilt.** Detection is by construction — the ordered
   `(kind, impl_id, profile_id)` of every complete leaf — over all complete
   leaves rather than the top-N, because a leaf that ranked poorly is still
   measured, still hash-bound and still the baseline.
2. **If it does not, B is rebuilt exactly once** through the frozen C1 treatment
   fixed path, imported from `experiments.phase_c1.session.build_arm_specs`
   rather than re-declared, and then passed through the `retained_candidates`
   seam so it takes the identical `identify_checkpoint` → canonical reload →
   `state_eval` path every searched candidate takes.

**Never both.** Their identities collide by construction — that is the same fact
that lets the search re-derive B at all — so injecting a rebuilt B beside a
searched one would put two states with one `artifact_digest` into the candidate
set. `BaselineFallback` is single-shot and refuses a second invocation.

**What proves a rebuilt B is B**, strongest appropriate identity first:

| gate | identity | when |
| --- | --- | --- |
| the construction | `FixedPathSpec.spec_hash == 3a233a9017b3…`, the preregistration's `treatment_spec_hash` | **`$0`**, in stage `bind_identities`, before a tensor moves |
| the shared prefix | `eea90c91…`, the pin the frozen prefix already carries and C1 replayed three times | inside `materialize_fixed_path` |
| the output | `artifact_digest == 53e30566…` — weights **and** config **and** arch signature **and** tokenizer **and** index | the final step's pin, then again at injection |

All three fail closed. On a mismatch the digest is **decomposed**, because "the
digest differs" is not a diagnosis: different `weights_digest` is a scientific
finding about determinism or an operator, a different `config_sha256` with
identical weights is a runtime fact — a `rope_theta` that moved between
transformers major versions changed a holdout NLL by 0.35 nats in this project
with nothing raising — and a reader must be able to tell which happened. C1's
recorded runtime travels with the explanation for exactly that reason.

The output digest is **observed, not preregistered**:
`treatment_output_digest_was_pre_pinned` is `false` in attempt 18's record,
because that run was this operator's first execution and there was no earlier
digest to pin it to. That makes it the strongest available content identity of B
and not a prior expectation, and the module says which it is.

**Cost**: **27.66 min**, priced as its own conditional
`baseline_rebuild_reserve` — **`$0.5026`**, which is **3.3%** of the
`$15.0446` ceiling, not the half a percent an earlier draft of this page
claimed. Derived from C1's own stage timestamps: stage C→D is the shared parent
from the teacher (24.185 min), stage E→F is the treatment's ATTENTION step and
its materialization (0.240 min), plus the maximum per-candidate
reload/identify/validate/`state_eval` overhead observed in Phase-B attempt 5
(3.24 min). Stage E is excluded: it replays the *incumbent's* ATTENTION, which
this baseline does not need.

**Schedule and metric are unchanged.** `SCHEDULE_V1` (one warmup level, beam
width 6), `PARETO_V1`, and the existing `state_eval` suite. No new search metric
is introduced for C2.

---

## Predicted size

From the configured space, using the real registry and the real
`expansion_profiles`:

| level | parents | states generated |
| --- | --- | --- |
| 0 | 1 (root/teacher) | **5** — DEPTH, FFN, WIDTH, ATTENTION×2 |
| 1 | 5 (warmup retains all) | **18** |
| 2 | 6 | 12–18 |
| 3 | 6 | 6–12 |

**Total expansions: 41 to 53**, each one a full
`apply → materialize → canonical reload → hash → validate → measure` cycle.

---

## Price

**Live quote, 2026-09-15: `NVIDIA L40S` `securePrice` `$1.09/h`, stock Medium.**
`securePrice`, not the `$0.79` `communityPrice` the same query returns — the
launcher's `check_gpu_offered` reads `securePrice` and aborts above
`--max-price`. A launch re-quotes; an hour-old price is not a price.

Per-expansion minutes come from Phase-B attempt 5's committed
`search_telemetry.jsonl`, end to end (operator + parent load + materialize +
identify + reload + validate + `state_eval`), split by root versus deeper
parent. The ceiling uses the **maximum** observed, not the mean.

**The bound is structural, not statistical.** `children_max × mean node cost` is
what authorized Phase-B attempt 3 at a 1.91–7.51 h projection for a run that
took 9.08 h and did not finish. Here the recursion enumerates every beam the
ranking policy could return and takes the exact extremum, so the maximum is
attained by some ranking and no ranking exceeds it.

**Five envelopes, separately named**, because each answers a different question
and one number for two of them is one number nobody can interpret. All of them
are derived by
[`scripts/autoinit/price_c2.py`](../../../../../scripts/autoinit/price_c2.py)
into the hash-verified
[`phase_c2_pricing.json`](phase_c2_pricing.json), which
`experiments.phase_c2.session.c2_budget_spec` is the only reader of.

| envelope | minutes | at `$1.09/h` | conditional |
| --- | --- | --- | --- |
| expected path — 95 min of session phases plus the DEPTH-early search (300.2) | 395.2 | **`$7.1787`** | no |
| + 10% contingency, on the expected path only | 39.5 | `$0.7178` | no |
| + `beam_composition_risk` — the exact worst case minus DEPTH-early | 335.8 | `$6.1004` | no |
| + `baseline_rebuild_reserve` — one B rebuild | 27.7 | `$0.5026` | **YES** |
| + `artifact_recovery_reserve` — teardown and collection | 30.0 | `$0.5450` | no |
| **hard planning ceiling** | **828.1** | **`$15.0446`** | |

The two reserves land **after** the contingency multiplier and **before** the
soft stop, which is what makes them protect the work: a reserve added as a phase
would be inflated by contingency, and one added after the soft stop would not
protect anything, because `afford()` refuses to start work that would cross the
soft stop. `baseline_rebuild_reserve` is deliberately **not** folded into
`artifact_recovery_reserve` — those buy different things, and paying for a
baseline out of the teardown reserve is how a session ends with a result it
cannot collect.

Session overhead is 95 min of named phases — setup and asset staging 45,
transfer 6, teacher fetch and verify 8, machine gates 22, selection commit and
artifact manifest 8, artifact synchronization 6. The setup figure is the one this
repository plans with, not C1 attempt 18's 6-minute observation: the same script
on the same image and card has also taken 8.5 minutes and over 150.

**The whole cost risk is beam composition, not node cost.** DEPTH costs 26–34
minutes wherever it runs; what the price turns on is how many beam members still
owe it. At Phase-B level 1 all six retained states already contained DEPTH and
the front put `FFN→DEPTH` in front 0, which is why DEPTH-early is the trajectory
with evidence behind it — but evidence about a ranking is not a bound on one.

So the lever, if a tighter ceiling is wanted, is beam width rather than a
deadline. A deadline that fires is fail-closed and produces nothing; Phase-B
attempt 3 lost 9.08 h that way, and the search workdir holds multi-gigabyte
intermediates that cannot be relayed for resume.

| beam width | expansions | DEPTH-early | worst case | expected | ceiling |
| --- | --- | --- | --- | --- | --- |
| 3 | 32–38 | 4.31 h | 7.33 h | `$6.4239` | `$11.4037` |
| 4 | 35–43 | 4.47 h | 8.42 h | `$6.5938` | `$12.6091` |
| 5 | 38–48 | 4.73 h | 9.51 h | `$6.8863` | `$13.8268` |
| **6** (`SCHEDULE_V1`) | **41–53** | **5.00 h** | **10.60 h** | **`$7.1787`** | **`$15.0446`** |

Every ceiling in that menu now includes the `baseline_rebuild_reserve`. Beam
width 6 is what `SCHEDULE_V1` declares and what this plan proposes: the menu is
here so a maintainer can trade breadth for cost deliberately, **not** so the
authorization number can be made smaller by narrowing the science.

### The one unmeasured input

`attention.activation_importance_v1` **has never run inside a search.** C1 drove
it through `fixed_path` only, where stage F took **14.4 s** end to end on an
already-narrowed 28-layer/1024-hidden parent. Its in-search cost is therefore
priced at **1.5 ×** the most expensive measured `ACTIVATION_STATS` operator
(`width.global_pca_v0`). That factor is a margin, not a measurement, and it is
generous in the affordable direction: per token the attention second moment
accumulates 36 × 32 × 128² = 18.9M MAC against the residual covariance's
36 × 2560² = 236M, roughly 12× less, and it reads no statistics cache by design.

It is the only unmeasured input to the price, it is named in
`bound().unmeasured`, and it is not blended into the measured table.

### Back-test

Phase-B attempt 5's **real beams** through the same branching model predict
every level's expansion count exactly — 10 / 46 / 19 / 7, total 82 — and its
minutes to within 0.2% (460.9 predicted against 461.7 observed). A model that
cannot reproduce the past cannot bound the future.

---

## Budget arithmetic

Recomputed from committed evidence, not restated from memory:
[`../../../../budget/ledger.md`](../../../../budget/ledger.md) and
`scripts/consolidate/derive_budget.py`.

```text
project   spent 290.5174  of 320.0000   leaving 29.4826
C1 package      22.6176   of  51.4425   leaving 28.8249
C1 formal       22.6176   of  45.4425   leaving 22.8249
```

A `$15.0446` C2 ceiling fits the project headroom — `290.5174 + 15.0446 =
305.5620`, leaving `$14.4380`.

**It fitting is not permission.** The C1 package is C1's: its formal allowance
funds C1 sessions, and C1 is finished. A C2 session needs its own maintainer
grant with its own ceiling. Neither the project headroom nor C1's unused formal
allowance is authorization for a different experiment, and "the arithmetic fits"
has never been a reason to launch.

---

## Search-2, predeclared and conditional

Predeclared now so that running it later is not a decision made after seeing the
data — and deliberately small, because the roadmap's "interactions with DEPTH /
FFN / WIDTH calibration" must not be answered by recreating the full P=2
factorial. That factorial is what Phase B ran; it cost 9.08 h without finishing.

Search-2 runs **only if** Search-1 produces one of:

1. a structurally different promising candidate — a different operator order in
   the top rank, not merely a different ATTENTION profile; or
2. a top ranking that is materially ambiguous or sensitive, by the same
   epsilon-Pareto semantics Search-1 ranks with.

If it runs, it takes the top **one, or at most two**, structural paths and
toggles the DEPTH, FFN and RESIDUAL_WIDTH profiles **one factor at a time**
against their incumbent values. That is 3 toggles per path: at most 6 additional
expansions of an already-materialized prefix.

Anything wider — a combinatorial calibration expansion, a second profile for
more than one held operator at a time — needs a concrete scientific reason from
Search-1's evidence *and* a separate maintainer approval.

---

## What Search-1 can and cannot conclude

**The `state_eval` ranking is a hypothesis generator, not a recovery result.** A
search winner is not a demonstrated initialization improvement, and must never
be reported as one.

* If **B remains preferred** and no candidate justifies behavioural testing, the
  result is a **constrained C2 search-stage null**: under this space, this
  schedule and this cheap metric, re-optimizing composition and order around the
  C1-selected ATTENTION operator did not beat the frozen C1 treatment. That is
  not a statement that ATTENTION research is closed, and not a statement about
  any path, order, student size or metric outside this space.
* If a candidate **C** is preferred strongly enough to justify confirmation:
  freeze its exact spec, record lineage, config hash, search metrics and the
  actual spend, report the search result — and **STOP**. Behavioural recovery is
  a separately authorized paid experiment. Default to **one** finalist; do not
  buy recovery for several close candidates.

### If confirmation is later authorized

The clean experiment is **B vs C only**: three preregistered paired recovery
seeds, the frozen 0.86M-token recovery budget, the frozen Phase-C/C1 behavioural
battery and evaluation semantics, and the same narrow interpretation discipline
C1 used. **A / `attention.weight_proxy_v0` is not rerun** — A → B is already
frozen C1 evidence. One confirmation round; no second round.

Cost class: roughly C1's measured `$10.2018` for six probes, to be repriced and
separately authorized.

**And it has a storage precondition.** C1 attempt 18 trained six probes at
**2.22 GiB each** and preserved **none** of them: every durability upload was
refused for private storage quota. The mechanism behaved correctly and recorded
each probe's identity, inputs and content hash together with the exact reason,
but the bytes went with the pod. Before a confirmation run whose completed
checkpoints are expected to survive a later stage's failure, a durable
large-artifact backend with room for ~13.3 GiB must exist. Freeing that quota
means permanently deleting historical LFS objects or changing a paid plan —
a maintainer decision either way, never an autonomous repair. Saving checkpoint
bytes and authorizing their reuse across attempts are separate questions.

This does **not** block Search-1, which produces no probe checkpoints.

---

## Implemented, and what is deliberately not

**Search-1 is executable.** As of 2026-09-16 it has a driver, a launcher, an
evidence contract, an authorization *type* and a derived budget:

| what | where |
| --- | --- |
| the session, declared | [`scripts/pod/autoinit_phase_c2_launch.py`](../../../../../scripts/pod/autoinit_phase_c2_launch.py) |
| the pod-side driver, two stages | [`scripts/pod/autoinit_phase_c2_driver.py`](../../../../../scripts/pod/autoinit_phase_c2_driver.py) |
| the space | [`scripts/experiments/phase_c2/search_space.py`](../../../../../scripts/experiments/phase_c2/search_space.py) |
| the baseline rule | [`scripts/experiments/phase_c2/baseline.py`](../../../../../scripts/experiments/phase_c2/baseline.py) |
| plan, authorization type, budget | [`scripts/experiments/phase_c2/session.py`](../../../../../scripts/experiments/phase_c2/session.py) |
| the evidence contract | `configs/autoinit/c2_artifacts.json` and `…_failed.json` |
| the price | [`phase_c2_pricing.json`](phase_c2_pricing.json), from `scripts/autoinit/price_c2.py` |

The driver runs the beam through the same `run_phase_a_search` Phase A and
Phase B ran, so it inherits the durability boundary that commits the ranking the
instant the expensive part succeeds. It is **standalone rather than a
`PhaseADriver` subclass**: that base class owns probe training, rungs, batteries
and elimination, and inheriting it would mean satisfying every contract that
machinery requires in order to use none of it.

**It still cannot run, and not for want of a flag.** The grant it names,
`logs/budget/approvals/autoinit_c2_authorization.json`, does not exist, and
`C2Authorization.load` refuses anything that is not a C2 grant under its own
schema. There is no readiness record, no bundle and no provider resource.

Three gates run before a pod is contacted, each refusing at `$0`: the volume
against the derived 87.4 GiB peak working set; the grant's ceiling against the
pricing record's own hash-verified figure; and the grant's plan hash against the
live configured space, so a grant issued before the space moved cannot authorize
a run after it.

**Deliberately not built:** the grant, the `launch_bound` readiness record, the
authorization, the bundle, and any provider resource. Those are the governance
chain and a maintainer decision, in that order, on the final clean
pre-authorization tree when a launch is genuinely imminent. Search-2 is not
implemented either: it is conditional on Search-1's evidence.
