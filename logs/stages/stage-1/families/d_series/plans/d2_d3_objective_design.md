# D2 / D3 scientific objectives: definition, student identity, implementation

**Status:** contract revision 7 (2026-10-14; revision 6 redesigned D2, revision 7 separated D2's and D3's sections): **D2 REDESIGNED** by maintainer
scientific decision -- see [§11](#11-d2-revision-6-the-frozen-teachers-confidence-weighting-a-root-reference-kl),
which is the current D2. The student-uncertainty D2 described in §0-§10
(`objective.d2_student_uncertainty_weighted_kl@v2`, `063bafd4…`) is
**SUPERSEDED -- NEVER EXECUTED**; those sections are kept as history and
remain the current description of **D3**, which is unchanged (`7b254538…`,
λ = 7.34, thresholds frozen from stage L, `refcal_20261013_170714`). GPU
evidence: [`../analyses/d2_gpu_evidence_review.md`](../analyses/d2_gpu_evidence_review.md).
Authorizes nothing.

**Revision 4, after the post-revision-3 review.** The objective identities,
the ranking and the λ rule are unchanged.

- The contract no longer says that state evaluation averages items equally.
  A generator test refuses that text.
- Exact engine token ids come from a read-only generator wrapper, and the
  shared protocol is unchanged (§8a).
- The D2 search is priced from D1's own search records (§7).
- The reference cache's hoarded-memory fallback is fixed in core, and its
  GPU validation is owed (§7).
- The reference calibration is wired as stage `L`.
- D2 has a draft preregistration, a session and a driver (§10).

**Revision 3, after an independent review.** The rest of this page reflects
it. The objectives were re-versioned to v2: D2 is `063bafd4…`, and D3 is
pending λ.

1. **State-evaluation aggregation.** It now uses the KL-only path's `T_x`
   weighting, with a built-in KL-only control (§1).
2. **λ and the diagnostic thresholds.** Both are measured **before D2's
   search**, on 147 unique items rather than 209 draws (§4).
3. **Scope.** The objective acts on DEPTH, state evaluation and the beam, not
   on the FFN/WIDTH/ATTENTION local rules. Confidence read on intermediate
   candidates is not the final student's (§3).
4. **High raw KL** is a registered diagnostic flag and never a selection
   rule. Top-K mass is flagged beside confidence (§8).
5. **Rescoring integration contract.** It covers exact ids, alignment
   evidence, failure isolation and cost measured before pricing (§8a).
6. **Budget.** There is a derived budget at K = 4, in
   [`d2_budget_v1.json`](d2_budget_v1.json) (§7).

This page follows the maintainer decision of 2026-10-13, which corrects the
D2/D3 objectives. It supersedes the D2/D3 sections of
`phase_c1/plans/phase_c_roadmap.md`; those are kept there verbatim as history,
with a pointer here.

| owner | what it holds |
| --- | --- |
| [`d2_d3_objective_contract_v1.json`](d2_d3_objective_contract_v1.json) | the registered contract: both objectives' identities, the ranking policy, the λ rule, and where each objective applies. Carries `contract_hash` |
| `src/aadistill/initialization/scoring/objective.py` | the generic forms, the student Top-K entropy and the fixed-denominator reducer |
| `scripts/stages/stage-1/families/d_series/objectives.py` | this campaign's instances (D2, D3(λ), `OBJECTIVE_PARETO_V1`) and the λ rule |
| `scripts/stages/stage-1/families/d_series/reference_calibration.py`, stage `L` of `phase_d1/topk_adoption_driver.py` | λ, the diagnostic thresholds and measured costs; one GPU session, after R0 and before D2's search |
| `scripts/stages/stage-1/families/d_series/rollout_profile.py`, `scripts/shared/evaluation/rollout_rescoring.py` | the behavioural diagnostics: text-level for every unit; per-position for new candidates |
| `tests/initialization/test_scoring_objective.py`, `families/d_series/tests/test_d2_d3_objectives.py`, `test_rollout_diagnostics.py`, `test_production_operator_timing.py`, `tests/evaluation/test_uncapped_eval_token_ids.py`, `phase_d2/tests/` | the tests |
| `scripts/stages/stage-1/families/d_series/d2_budget.py` → [`d2_budget_v1.json`](d2_budget_v1.json) | the derived budget through D2 |

## 0. Why the student's confidence enters: capacity-aware student reasoning

This section records the maintainer's supplement of 2026-10-13. It is
contract revision 2; the objective identities are unchanged.

- **Premise.** The teacher (Qwen3-4B-Thinking) and the ~0.6B student differ
  in capacity, expressiveness and learning ability. The student need not,
  and may not be able to, reproduce the teacher's reasoning distribution. On
  some tasks it may reach the same correct answer by a shorter, simpler or
  entirely different trajectory.
- **Consequence.** A large teacher token-level KL does not by itself mean
  worse student reasoning. KL-based compression selection may over-reward
  structures that imitate the teacher and under-rate structures that
  preserve the student's own effective way of reasoning.
- **D2 tests** whether relaxing strict imitation where the student is
  certain helps a small model keep computation suited to its capacity. It
  does **not** assume the student is right where it is confident.
- **D3 tests** whether the student's own certainty has structural-selection
  value independent of the KL.
- **The research question** is neither "maximize confidence" nor "minimize
  KL". It is: which structural initialization, under the capacity limit,
  best lets the student keep and recover its own effective reasoning?
  Student Top-K confidence is a **candidate proxy**, and only independent
  behavioural evidence after recovery training decides whether it helps.

**Four quantities, kept apart:**

| quantity | where it is measured | role in D2/D3 |
| --- | --- | --- |
| student Top-K entropy on **teacher-native** prefixes | search (calibration, state evaluation) | **is** the objective's input |
| student Top-K entropy on the student's **own rollout** prefixes | `rollout_rescoring.py`, after generation | diagnostic only, never selects |
| student–teacher distribution fit (Top-K + tail KL) | teacher-native in search; own-rollout in the diagnostic, beside teacher surprisal | objective component / diagnostic |
| autonomous rollout correctness | shared protocol, after recovery | **the endpoint**; the only thing a promotion question uses |

**Calibration stays the frozen teacher-native profiles**, so the objective is
the only variable. D2 and D3 introduce no student-prefix calibration.

## 1. Definitions

D1, D2 and D3 share everything except what a KL-scored decision minimizes:

- the target position policy, `positions.supervised_target_v1`;
- the two calibration profiles;
- the reference Top-200 + tail KL;
- geometry, operators, beam and recovery recipe;
- the behavioural endpoint and the shared evaluation protocol.

**Student Top-K conditional entropy and confidence (K = 200).** For the
candidate's own logits `z_t`, take its own `S_t = TopK(z_t, 200)` and
renormalize inside it:

```text
H_s(t) = − Σ_{i∈S_t} q̃_t,i log q̃_t,i     c_s(t) = 1 − H_s(t) / log 200   ∈ [0, 1]
```

This entropy is **not** any of:

- the reference's entropy;
- the full-vocabulary entropy;
- the student's entropy on the reference's support;
- a K+1-bucket entropy.

The mass outside `S_t` is recorded beside it as `student_topk_mass` and is
never folded in.

**The three objectives**, each per item over `T_x`, the item's scored
positions:

| | per-item objective | direction | hypothesis |
| --- | --- | --- | --- |
| D1 | `(1/T_x) Σ KL_t` | min | target-aware fidelity (unchanged, frozen) |
| D2 | `(1/T_x) Σ (1 − c_s(t)) KL_t` | min | a confident student needs less fidelity, even where KL is large |
| D3 | `(1/T_x) Σ [KL_t − λ c_s(t)]` | min (S = −L max) | self-confidence earns its own reward |

**The denominator.** It is `T_x`, the count of scored positions, never
`Σ w`. A weight-sum normalization would cancel a uniform rise in confidence,
which D2 exists to let through.

**Aggregation.** Per item first, then the mean of item means per sub-type,
then D1's sub-type → domain rule. There is no token pool across items.

**Infinite KL.** `KL_t = +∞` stays `+∞` under both forms, even at `c = 1`.
`0·∞` is undefined, and resolving it to 0 would make an impossible candidate
the best one. A finite KL at `c = 1` contributes 0 to D2, as intended.

**D2 and D3 are two identities.** Their `objective_hash` differs by form, and
D3's also carries λ. Every definitional choice is a named field of the
identity:

- the entropy definition and K;
- the confidence mapping;
- the denominator and the aggregation;
- the direction, and λ.

**Aggregation across items keeps D1's weighting in each consumer
(revision 3).**

- **Per item:** the formula keeps the fixed `1/T_x` and never divides by a
  sum of confidence weights.
- **DEPTH:** items are averaged equally within a sub-type, exactly as D1's
  DEPTH did.
- **State evaluation:**
  - within a sub-type, items are weighted by `T_x`, which is the same as
    pooling the integrand over the sub-type's scored positions and dividing
    by their count;
  - each critical class is pooled over items;
  - this is exactly D1's state-KL weighting.
- **Identity:** `aggregation: kl_path_weighting_per_consumer/v2`.

Revision 2 had averaged state-evaluation items equally. That changed the
length weighting as well as the integrand, so a ranking change could not be
attributed to confidence.

**The KL-only control is built in.** Every objective evaluation reports
`state.objective.raw_kl.*`: the raw KL aggregated exactly as the objective
is.

- Tested to 1e-6 relative on a suite of unequal item lengths, it equals the
  same evaluation's `state.teacher_kl.equal_domain_mean`, `.worst_domain`,
  each domain, and `state.critical_token_kl`.
- A companion test shows that item-equal weighting would *not* reproduce
  them on that suite, so the control is not vacuous.
- A mutation that reverts to item-equal weighting is caught.

So, on the same states, rankings by raw KL and by the objective differ only
through the integrand.

## 2. Computation order: one forward, with selection before the head

For every KL-scored decision:

1. The full sequence runs through the backbone, so every scored position keeps
   its causal context.
2. The scored rows are selected from **D1's target mask and padding only**.
   Confidence exists only after logits, so selecting by it would be circular.
   The selection is a superset because `w ∈ {0, 1}`.
3. Only those rows reach the output head (`scored_v1`, once R0 passes; until
   then `all_v1`, with the same arithmetic on more rows).
4. The reference Top-200 + tail KL is computed from those rows. Its
   full-vocabulary normalization semantics are unchanged.
5. The student Top-200 entropy comes from the **same candidate logits**, with
   no second student forward. The full-vocabulary log-normalizer the KL
   already computed is shared (`sketch_forward_kl_terms`, and the
   `position_sink` of `distortion_on_reference_support`). The only new
   vocabulary-wide step is the Top-K selection, done in the logits' own dtype
   without a float32 copy of the row.
6. `c`, then the D2 weight or the D3 reward, then the per-item fixed
   denominator, then sub-type and domain.

The tests show `scored_v1` and `all_v1` give identical objective scores and
decisions, for DEPTH and for state evaluation, at batch sizes 1 and 3.

## 3. Which model is the student, operator by operator

**Scope** (contract field `objective_scope`). The new objective acts directly
on exactly three places:

- DEPTH's operator-local candidate evaluation;
- global state evaluation;
- beam ranking and pruning.

The FFN, WIDTH and ATTENTION local rules keep D1's statistic. **Do not claim
that all four operators' local rules use student confidence.**

**The confidence the search reads belongs to intermediate candidates:**

- in DEPTH, the parent with layers bypassed;
- in state evaluation, partially compressed children.

All of this is before recovery training. It is not the final ~0.6B student's
confidence, and it is not evidence about that model's capacity adaptation;
only the behavioural evaluation after recovery speaks to that.

Each operator is answered against the user's eight questions.

### DEPTH (`depth.causal_kl_greedy_v1`): exact

1. **Student:** the parent with the candidate subset `S = removed ∪ {c}`
   bypassed, held in memory. **Reference:** the intact parent, as in D1.
2. **Same candidate for KL and entropy?** Yes: both come from one candidate
   forward (`abls`).
3. **Forward to reuse?** Yes. It already exists per candidate for the KL.
4. **New forward needed?** None.
5. **What is reused:** the scored rows, and the log-normalizer, returned by
   `sketch_forward_kl_terms` with the KL's operations unchanged.
6. **Caching:** the parent's reference sketch is cached across candidates, as
   before. The candidate's KL and entropy are **never** cached, because they
   are the candidate.
7. **How the objective is applied:** integrand → per-item `1/T_x` → sub-type →
   domain → greedy argmin. Every candidate's components are recorded in
   `artifacts.objective_candidates`:
   - objective, raw KL, entropy, confidence;
   - weight or reward;
   - student Top-K mass and reference tail mass;
   - a 10-bin confidence histogram.

   `op.depth.causal_kl.final` still means the raw KL of the chosen subset;
   `op.depth.objective.final` is the objective.
8. **Cost:** one Top-200 selection per scored row. See §7.

### FFN, WIDTH, ATTENTION (activation statistics): no operator-local student exists

1. **Student:** **none at selection time.** The operator reads activation
   statistics of the parent and *produces* the child. The candidate student
   does not exist until the selection is made.
2. **Same candidate?** Not applicable locally. At state evaluation the child
   is the student for both KL and entropy.
3. **Forward to reuse?** Not inside the operator. The state evaluator's
   forward of the child is the candidate forward.
4. **Should a candidate forward be added?** It would be cheap: one child
   forward at 67 items is about the cost of one DEPTH candidate, about 5 s.
   But each of these operators yields **exactly one** child, so there is
   nothing for an extra forward to choose between. Making several children
   would change the operator space, which the decision fixes.
5. **What is reused:** the state evaluator's KL and entropy of the child.
6. **Caching:** the statistics stay cached per (parent, profile), unchanged.
   Objective terms are per child and never cached across children.
7. **How the objective is applied:** the local statistic stays D1's
   target-aware binary statistic. The operator does not declare
   `consumes_scoring_objective`, so its config and state id **do not fork**:
   the same parent gives the same child as under D1. D2/D3 reach these
   children through state evaluation and beam ranking, which is a real
   selection among them.
8. **Cost:** 0 at the operator.

**The parent-confidence proxy is not adopted.** Weighting the activation
statistics by the parent's Top-K uncertainty would use the parent, not the
candidate, so it would be an approximation and not scientifically equivalent.
D3's additive reward also has no weight analogue: turning it into one is
exactly the D2/D3 confusion the decision forbids.

If a later decision wants the proxy for D2:

- **Activation weighting:** it needs true float weighting in all four
  collectors and their divisors. Today those collectors refuse continuous
  weights by name (`require_binary_token_weights`), so nothing is silently
  binarized:
  - `uncentered_moment`;
  - `ffn_neuron_importance`;
  - `residual_covariance`;
  - `head_write_energy`.
- **Validation:** the per-position agreement between parent and child
  confidence. The proxy's statistics pass and the child's state evaluation
  already produce both, so this comes at no extra cost.

### State evaluation: exact

- **Student:** the canonical reload of the child.
- **Reference:** the original teacher, whose sketches stay cached across
  candidates.
- **KL and entropy:** from the same candidate forward. The KL reaches the
  objective through the reducer's `position_sink`, and the sink cannot change
  any sum.
- **Metrics:**
  - ranked: `state.objective.equal_domain_mean`, `.worst_domain` and
    `.critical_token` (per item over class positions, then items, then the
    unweighted mean over classes);
  - per-domain values;
  - every component's equal-domain mean.

  `detail.objective` carries the definition, `T_x` statistics, the confidence
  histogram and quantiles, the weight or reward quantiles, and per-item
  records.
- **D1's metrics are byte-for-byte unchanged** in the same evaluation, which
  is tested.

### Beam ranking and pruning

`OBJECTIVE_PARETO_V1` is PARETO_V1's shape (equal-domain mean, worst domain,
critical token, ε 1e-4, `parent_path` diversity, `state_id` tie-break) with
the KL metrics replaced by the objective's. The search **refuses**:

- objective metrics without a declared objective;
- an objective ranked on the KL;
- a measurer, or any single measurement, under another objective.

**How the three levels relate.** The same integrand is used at every level.
The reference is the parent for DEPTH's local choice and the teacher for
states, exactly D1's split. The beam ranks the state-level objective.

## 4. D3's λ

**The rule is registered now** (`objectives.lambda_rule`, `rule_hash` in the
contract):

```text
λ = IQR(per-item KL) / IQR(per-item confidence)     rounded to 3 significant figures
```

- **Reference pair:** the teacher (reference) and B (student), a pair no
  D-series search produced.
- **Items:** the three sets hold **209 draws but 147 unique items** by
  content identity, and each unique item counts once (rule v2):
  - `reasoning_heavy@v2`'s 62 draws are 51 unique items, and every one of
    them is also in `domain_balanced@v1` (67 items, the same pool sampled
    with replacement);
  - `state_eval_v1` adds 80 disjoint items.

  The with-duplicates value is reported as sensitivity only.
- **Per-item quantities:** fixed-denominator means of the Top-200 + tail KL
  and of B's own Top-200 confidence.
- **Reported as sensitivity, never used to choose:** the SD ratio, the
  209-draw value, each set alone, and q2 as the student.
- **Refused:** fewer than 20 items, `IQR(c) = 0`, a non-finite KL, or a λ
  that is not positive and finite.

**The scale argument.** λ converts confidence into KL units. Equating the two
components' between-item spread on the reference pair means one
interquartile range of confidence trades for one interquartile range of KL,
so neither dominates the ranking by units alone. The IQR is used because
per-item KL is heavy-tailed; a test shows one extreme item does not move λ.

**When.** λ is measured **once, before D2's formal search**. This
corrects revision 2, whose execution plan said "before D3" while the rule
said "before D2". It is measured by `reference_calibration.py`, in the same
pass as:

- the **diagnostic thresholds**: p90 per-position KL, p90 confidence and p10
  Top-K mass of B against the teacher, all registered in
  `diagnostic_threshold_rule`;
- a **measured cost** of the rollout rescoring on the real teacher and a
  0.6B model.

It is committed as `analyses/d2_d3_reference_calibration.json`. Until that
record exists, `frozen_d3_objective` and `frozen_diagnostic_thresholds`
refuse. No D2 or D3 search or behavioural result enters it.

**It needs a GPU.** It needs teacher and B forwards, and B is rebuilt on the
pod (1279.8 s measured). This dev box has neither the teacher's weights nor
the memory.

**Cost.** Setup 1.6 / 15 min (expected / bound), other overhead 2.8 / 13.6,
B rebuild 21.3, forwards about 3–5, teardown about 1:

From [`d2_budget_v1.json`](d2_budget_v1.json):

| | time | all-in |
| --- | --- | --- |
| expected | 29.7 min | $0.54 |
| hard | 64.6 min | $1.18 |
| proposed grant (with $0.30 reserve) | — | $1.52 |

The session wiring, which runs `frozen_baseline_spec` and then this script, is
owed **before D2's search**. It needs its own grant.

## 5. Identity, and why D1 and R0 are unchanged

The objective follows exactly the route `distribution_support` takes, and is
**absent everywhere when undeclared**:

| owner | when the objective is set |
| --- | --- |
| `SearchConfig.as_dict` | adds `scoring_objective` |
| `hashed_operator_config` | adds it only for implementations declaring `consumes_scoring_objective` (DEPTH today) |
| `OperatorImplementation.execute` | refuses a declaration that disagrees with the handed object, and refuses a declaration on an operator that does not consume it |
| `ReductionSemantics.as_dict` | adds it, so `measurement_protocol_id` moves |
| `FixedPathSpec` | carries it, so a D2/D3 path replays under the objective it ran with |

`consumes_scoring_objective` is **not** in `signature()`, so no
implementation's signature hash moves. With no objective declared, every
dict, hash and arithmetic path is the historical one:

- the existing row-selection, support, operator-config and fixed-path tests
  (164) pass unchanged;
- a test asserts the D1 metrics are identical with and without an objective.

**R0 is unaffected.** It runs D1's path with no objective, its plan and
tolerances are unchanged, and no file R0 executes changed in arithmetic. The
`support.py` refactor returns the same KL from the same operations.

## 6. What changed in the core, and what was reused

**New:** `scoring/objective.py`, about 300 lines:

- the forms and the student Top-K terms;
- the fixed-denominator reducer;
- the binary-weight refusal;
- the confidence histogram.

**Minimal changes:**

- `support.py`:
  - `sketch_forward_kl_terms` returns the log-normalizer it already computed;
  - an optional `position_sink`.
- `protocol_identity.py`: optional `scoring_objective`.
- `operator_config.py`: `scoring_objective_config`.
- `operators/base.py`: the context field, the class flag and the two-sides
  check.
- `depth/causal_kl_greedy.py`: the objective path on all three DEPTH row paths
  (per item, batched, scored rows) and the records.
- `planning/metrics.py`: `_ObjectiveAccumulator`.
- `planning/search.py`: the config field, plumbing and the three checks.
- `planning/fixed_path.py`: carrying the objective.

**Reused unchanged:**

- the reference sketch;
- the Top-200 + tail KL arithmetic;
- the position policy and `scored_rows`;
- `domain_balanced_score`;
- the stats cache;
- packing and batching;
- the beam algorithm and `BeamRankingPolicy`.

Nothing in `src/aadistill` names D2, D3, Qwen, a dimension, a K or a λ. Every
instance value lives in `scripts/stages/stage-1/families/d_series/objectives.py`.

## 7. Costs

| component | new work | estimate |
| --- | --- | --- |
| candidate forwards | **none**: DEPTH and state evaluation reuse their existing candidate forward | 0 |
| student Top-200 selection | one `topk` per scored row, in the logits' dtype, chunked at 512 rows | DEPTH: the reduction was 14% of a candidate (a6, synced split: 0.76 s), against 4.59 s per candidate in production (a8). A selection pass is a fraction of the KL reduction's four vocabulary-wide passes, so **+3 to 7% of DEPTH wall time (estimate)**. State evaluation: **+5 to 10% (estimate)** |
| log-normalizer for Top-K mass | shared with the KL | 0 |
| activation statistics | unchanged | 0 |
| memory | a `512 × 151,936` bf16 selection block, about 0.16 GB transient | against peaks of 16.3 GB (P) and 20.5 GB (D): negligible |

DEPTH records `objective_seconds`, so the first D2 expansion measures the
estimate rather than relying on it.

**Search price basis.** D1's search measured **552 min, $10.03** under the
same Top-200 + tail support with `all_v1` (the previous design page said
"full vocabulary" and is now corrected). For D2 and D3:

- `scored_v1` is expected to subtract about 8% of DEPTH time, if R0 passes;
- the objective adds the estimates above.

**The executable budget** is derived by `d2_budget.py` into
[`d2_budget_v1.json`](d2_budget_v1.json), at K = 4.

**Records model (post-revision-3 review).** The D2 search is now priced
from D1's own search records, not D1's pre-search design:

- **Telemetry.** D1's telemetry (92 expansions) is committed as
  [`../analyses/d1_search_telemetry.jsonl`](../analyses/d1_search_telemetry.jsonl).
  It is refused unless its sha256 is the one D1's evidence manifest records.
- **Pricing machinery.** D1's own `search_cost` prices it, so there is no
  second formula, with two changes to the cost model:
  - **DEPTH** takes D1's measured end-to-end maxima: root 41.71 min,
    deeper 35.60 min.
  - **The other operators** keep the frozen cross-host cells. They are more
    conservative: D1's own were 3–4× smaller on a fast host.
- **The objective's overhead** applies only to the DEPTH scoring loop and
  the state evaluation, as added seconds per candidate and per evaluation:
  - **provisional:** +5% / +10% of D1's means;
  - **after stage L:** measured. `d2_budget.py --calibration-session <adoption.json>`
    substitutes the measurement.
- **Expectation:** D1's measured session plus the expected added minutes over
  D1's actual trajectory.

**Why DEPTH moved.** D1's design priced DEPTH from a8's single cached
invocation (1,194 s). D1's search then recomputed the reference cache in 15
of its 16 DEPTH expansions, taking 1,336–2,486 s each.

The cause is in the telemetry:

- 0.12 GiB driver-free;
- **36.0 GiB reserved but unused** by the allocator;
- a 0.14 GiB sketch.

`_available_memory_bytes` read `mem_get_info`, which counts the allocator's
hoard as taken. It now calls `empty_cache()` first. That is the remedy
`memory_snapshot` names, and it changes no tensor, kernel or identity. It is
declared in `test_cuda_surface_preserved.py` and covered by a CPU regression,
which a mutation check confirms.

**GPU validation is owed.** Stage L's DEPTH pair runs under a deliberately
hoarded pool and records `cache_fix_validated`. Until that flag is true, the
records price is in force.

| step | expected | hard ceiling | actual |
| --- | --- | --- | --- |
| R0 | $1.53 | $3.60 (grant $4.00) | **$1.5074**, PASS |
| reference calibration (stage L) | $0.87 | $1.42 (grant $1.76) | **$0.8069**, complete |
| D2 search, price in force `cache_fix_validated` | $10.80 | $22.38 | — |
| D2 search, records price (not in force) | $10.80 | $32.01, over the $30 cap | — |
| D2 screening (8 probes, 4 arms; no B/q2; measured rescoring inside) | $16.61 | $20.83 | — |
| D2 confirmation (3 probes, 1 arm; measured rescoring inside) | $6.97 | $8.55 | — |

- **The switch to `cache_fix_validated`** was made by `d2_budget.py` itself.
  It reads the stage-L session booked in the calibration campaign
  (`evidence_session_sha256` `6806eb84…`), verifies the hash and the
  identities, and takes `cache_fix_validated = true` from it. No field was
  set by hand.
- **The measured overheads:**
  - DEPTH: +0.184 s per candidate;
  - state evaluation: +0.466 s per evaluation;
  - rescoring: 1.83 min per probe.

  Each enters the expectation as measured and the planning bound with a 1.5×
  margin, because one timing is not a bound.

**What each figure is:**

| layer | D2 search | status |
| --- | --- | --- |
| measured | D1's search: 551.99 min, $10.03; stage L: +0.184 s per DEPTH candidate, +0.466 s per state evaluation; cache fix validated | completed runs |
| estimate | 565.5 min, $10.80 = D1 + the measured added minutes | from records and one measurement |
| planning bound | 1,172 min, $22.38: D1's worst-case trajectory, DEPTH at the cached cell, overheads ×1.5 | **not** a proven runtime bound |
| enforceable stop | the session cap in the D2 search authorization, at the live rate | the watchdog at hard-terminate; the soft stop and the cache-regression guard fail the search closed before it |

**Programme scenarios** (`d_series_programme_budget.json`):

- **Status:** a dated snapshot against the live limits:
  - project remaining $120.14 of $590;
  - formal remaining $127.96;
  - engineering remaining $4.34;
  - $30 per session.
- **Reserves:** each formal session's teardown reserve is counted once. R0's
  and the calibration's grants are released against their bills: both
  campaigns are closed, and their recorded costs are already inside the live
  limits.

| scenario (price in force) | expected | hard + reserves | left at hard |
| --- | --- | --- | --- |
| D2 | $34.38 | $52.66 | $67.48 |
| D2 + D3 | $68.76 | $105.32 | $14.82 |
| D2 + D3 + one fresh confirmation | $80.99 | $120.82 | **−$0.68** |

**Reservation rule (proposed):** not satisfied, short by $0.68. Hard ceilings
and unverified diagnostic estimates are reported separately; with the
rescoring measured, no unverified estimate remains.

The levers, all the maintainer's:

- D2 under `scored_v1`, now eligible: about 5% of DEPTH time, not priced in;
- re-deriving after the D2 search from actual spend (D1 spent $10.03 of
  $21.49);
- D3 at K = 2, saving $9.66;
- a cap decision.

**Behavioural field.** The shared protocol trains 2K + 3 probes per
experiment and never trains B or q2. **K = 4 is prepared**, as the independent
review recommends, for wider structural coverage. It is frozen only in D2's
preregistration. For reference:

| | K = 2 | K = 4 |
| --- | --- | --- |
| winner's-curse inflation | 0.43 SESOI | 0.78 SESOI |
| P(advance the better candidate), given it is in the field | 0.82 | 0.65 |
| GPU | $10.87 | $17.08 |

What remains unmeasured is how often the search's quality order contains the
behaviourally better candidate at all, and that favours a wider field. D1 is
one data point:

- its screening advanced its quality position 2, not position 1;
- position 1 placed second.

## 8. Diagnostics, and D2's degeneracy

Recorded on every DEPTH candidate and every state:

- raw KL, Top-200 entropy and confidence;
- D2's weight or D3's reward, and the objective;
- `T_x`;
- the confidence histogram (states also get quantiles);
- student Top-K mass and reference tail mass;
- D1's KL metrics, unchanged, beside them.

**D2's degeneracy.** A candidate can lower D2's objective by becoming more
confident at high-KL positions. D2's ranking never sees raw KL, so it is
always recorded next to the objective, and **a low D2 score is never read as
higher fidelity**.

**High raw KL is not a reason to drop a candidate.** A confident student
departing from the teacher is the hypothesis under test. Review is a
**registered diagnostic flag** (`diagnostic_threshold_rule`):

- it is computed offline from records the search already writes;
- it never enters a score, a ranking, pruning or retention;
- it is never a manual step in a running search.

**Thresholds** come from the B–teacher reference calibration and are frozen
before any D2 candidate exists:

| threshold | value |
| --- | --- |
| high KL | p90 of per-position KL |
| high confidence | p90 of B's confidence |
| low Top-K mass | p10 of B's Top-K mass |

**What is recorded.** Each DEPTH candidate and each state records two joint
histograms:

- confidence decile × a fixed log grid of KL;
- confidence decile × Top-K-mass decile. This exposes a high *conditional*
  confidence sitting on a Top-K that holds little of the student's real
  probability.

`objectives.flag_counts` applies a threshold at the first grid edge at or
beyond it, so a flag never counts a position the exact threshold would not.
The rollout rescoring applies the thresholds per position, exactly.

## 8a. Behavioural diagnostics and the claim boundary

Beyond the endpoint, every D2/D3 behavioural round analyses the following.
All of it is **exploratory**: none of it decides `SHARED_GO` or `NO_GO`.

| question | instrument | arms it applies to |
| --- | --- | --- |
| does the student still solve the task when far from the teacher's distribution? | rescoring: per-sample KL and teacher surprisal on its own rollout, × correctness | new candidates only |
| autonomous reasoning length, correctness, usable rollout | `rollout_profile.py` (text-level, $0) | every unit, B/q2 included |
| how does confidence relate to correctness? | rescoring: confidence over the reasoning and answer spans, × correctness | new candidates only |
| are high-confidence, high-KL positions valid alternative reasoning or wrong overconfidence? | rescoring: quadrant fraction × correctness. Thresholds are registered in each preregistration before any result; none are in code | new candidates only |
| did D2/D3 improve autonomous generation, not only the search objective? | the endpoint and the rollout profile against B/q2, read beside the search objective and raw KL | candidates vs B/q2 |

**The rescoring instrument** (`scripts/shared/evaluation/rollout_rescoring.py`):

- **Per position of the student's own greedy rollout:** student confidence,
  student token log-prob, teacher surprisal, and KL(teacher ‖ student) on
  the Top-200 + tail.
- **Where it runs:** it needs the trained checkpoint, so it runs **on the
  pod, after generation and before the checkpoint is released**.
- **What it does not touch:** it reads the generations and changes neither
  generation nor scoring, so the shared protocol's evaluation identity is
  untouched.
- **Cost:** estimated 3–5 GPU-minutes per probe, about 0.95M tokens per
  probe at D1's measured lengths.
- **Tests:** an independent computation with the causal shift; a mutation of
  the shift is caught.
- **Owed:** the D2/D3 behavioural driver must call it.

**Integration contract** (contract field `rollout_rescoring_integration`,
from the independent review):

- **Exact engine ids (revision 4).**
  - The generator (`uncapped_eval.py`) holds the ids in memory but records
    only decoded text. Editing it to record them would change a file in the
    generation source set, moving the generation fingerprint and the shared
    protocol's evaluation identity -- a **versioned decision**
    (`d_series_shared_eval` v2), not taken.
  - Instead, `scripts/shared/evaluation/uncapped_eval_token_ids.py` runs the
    **unchanged** generator with a read-only hook on its single post-wave
    `tok.decode(gen)` per record, after the engine loop has exited. It writes
    `<stratum>.token_ids.jsonl` (never `*.json`, which admission reads as a
    summary) and verifies every row against its record: id count, prompt
    count, decoded-text sha256, EOS vs `stop_reason`.
  - Identity: the hook installs only when the generator's source digest is
    the protocol's; the wrapper is **not** in the generation source set. On
    the real generator with a stub engine, plain and wrapped runs give
    identical records, summaries and reconstructed generation fingerprint
    (`tests/evaluation/test_uncapped_eval_token_ids.py`; two mutations of the
    hook are caught). Only the behavioural driver's own executable closure
    moves, because its argv names the wrapper.
  - Residual: vLLM's scheduling is timing-dependent in the plain run too. The
    wrapper adds no work inside the engine loop, but token equality with a
    counterfactual plain run cannot be proven on a non-batch-invariant engine.
  - A record without an exact sidecar row is rescored from re-tokenized text,
    labelled `ids_source=retokenized_text`. `alignment_evidence` refuses
    "engine" ids that do not reproduce the record's count and text.
- **Alignment evidence per sample:** prompt and completion id hashes, the
  tokenizer hash, the generation fingerprint, and the count and text match.
- **Untouched:** D1's evidence, and the shared protocol's scoring contract
  and endpoint.
- **Release order:** a checkpoint is released only after the diagnostic
  evidence is persisted, when a consumer exists.
- **Failure isolation:** `rescore_unit` records failures per sample and never
  touches a completed score.
- **Cost:** measured by the reference calibration before any ceiling
  includes it.

**What B and q2 can and cannot contribute.** Their ten units hold decoded
text and the engine's token counts. They hold **no** token ids,
log-probabilities or logits, and their checkpoints are retired. So:

- their confidence on their own rollouts is **not available** and is not
  estimated;
- comparisons with them are text-level and endpoint-level only;
- re-tokenizing their text reproduces the engine's count exactly for 82–85%
  of samples, so even a teacher-only pass over their text would be an
  approximation of the generated sequence.

**Baseline facts** (`analyses/d1_rollout_profile.json`, descriptive). These
set what the diagnostics can resolve:

- B and q2 answer 2–4% of scorable prompts. On confirmation, B is right on
  103 of 2550.
- Their correct sets barely overlap: both are right on 29 prompt-seeds, q2
  alone on 73, and B alone on 74.
- Median generation is 444–539 tokens and median reasoning about 300.
- Correctness is roughly flat across reasoning-length quartiles.
- So any confidence–correctness or quadrant analysis rests on about 100
  correct samples per arm and is low-powered.

**Claim boundary** (the contract's `claim_boundary`):

- **Goal:** initializations that help the student keep and recover its own
  effective reasoning under the capacity limit.
- **Not the goal:** maximizing confidence or minimizing KL.
- **May claim:**
  - a candidate's endpoint result on the shared protocol, within its limits;
  - labelled exploratory diagnostics of measured candidates.
- **May not claim:**
  - that a lower search objective means better reasoning or fidelity;
  - that confident, far-from-teacher positions are better or alternative
    reasoning, without rollout evidence linking them to correct completions;
  - anything per-position or about confidence for B or q2;
  - a mechanism from the diagnostics alone.

## 9. Evidence

**Revision 3 additions:**

- **Core:** 52 tests, up from 49. Added:
  - the toy suite now has unequal item lengths;
  - `T_x` weighting;
  - the KL-only control equals D1's metrics;
  - a non-vacuity test;
  - joint histograms count every scored position.

  A mutation reverting to item-equal weighting is caught.
- **Family:** 28 tests, up from 19. Added:
  - unique-item dedup;
  - with-duplicates sensitivity and thresholds in the record;
  - conservative offline flags;
  - scope and ordering;
  - the measured-rate projection;
  - the budget: derived, no B/q2 probes, the diagnostic excluded from
    ceilings.
- **Rescoring:** 12 tests, up from 8. Added:
  - engine ids must reproduce the record;
  - re-tokenized ids are labelled;
  - low Top-K mass is flagged;
  - one failing sample does not stop a unit.

The lists below are revision 2's.

**`tests/initialization/test_scoring_objective.py`: 49 tests.**

- The entropy against an independent sort-and-renormalize computation, plus:
  - its own Top-K only;
  - not the full-vocabulary entropy;
  - uniform → c = 0 and one-hot → c = 1, with unclamped bounds;
  - Top-K mass;
  - bf16 equals fp32 of the same values;
  - K > V and K < 2 refused.
- **Direction:**
  - D2's weight is 1 − c;
  - more confidence lowers both objectives;
  - the reward is not a weight;
  - infinite KL is never rescued;
  - λ must be positive and finite, and absent for D2.
- **Denominator:**
  - c = 0.5 halves the mean, where a weight-sum normalization would not;
  - a uniform rise in confidence is not cancelled;
  - unscored positions are never read;
  - continuous position weights are refused.
- **Identity:**
  - forms, λ and K are each in the hash;
  - no objective leaves the reduction identity unchanged;
  - only DEPTH hashes the objective.
- **DEPTH:**
  - every score equals an independent computation from the bypassed model's
    logits;
  - the confidence follows the candidate subset;
  - all components are recorded;
  - the KL key keeps its meaning;
  - scored rows equal all rows;
  - batched equals per-item;
  - the oracle refuses an objective;
  - an undeclared objective is a contract violation.
- **State evaluation:**
  - D1's metrics are identical;
  - the primary is per item → sub-type → domain;
  - D3's per-item `objective = raw_kl − λc`;
  - scored rows equal all rows;
  - the oracle refuses an objective.
- **Search:**
  - the hash moves only with an objective;
  - all three mismatches are refused;
  - a real toy search runs and ranks on the objective.

**`families/d_series/tests/test_rollout_diagnostics.py`: 8 tests.**

- The text profile:
  - its join;
  - the engine count kept, with a re-tokenization flag;
  - an unscored generation refused;
  - rates and paired deltas.
- The rescoring:
  - against an independent computation of the student's token log-prob,
    the teacher's surprisal and the student's own Top-K confidence, with the
    causal shift;
  - the same model gives zero KL;
  - spans and thresholds.

**`families/d_series/tests/test_d2_d3_objectives.py`: 19 tests.** These are
the 15 below plus 4 on contract revision 2:

- the four quantities are kept apart;
- the claim boundary;
- mechanism analyses are limited to measured candidates;
- the identities are unchanged.

The original 15:

- The two instances.
- D3 refused until λ is frozen, and refused under another rule.
- The ranking policy.
- The λ rule:
  - it is the IQR ratio;
  - it rounds to 3 significant figures;
  - a heavy tail does not move it;
  - its refusals;
  - the record carries the rule.
- `item_terms` on a tiny model pair, where the same model as student gives
  KL = 0.
- The committed contract is current.

**Mutation checks**, each caught:

- flipping D2's weight to `c`;
- normalizing by the mean weight.

## 10. Before D2 and D3 can be preregistered

| step | state | cost |
| --- | --- | --- |
| R0: `scored_v1` GPU equivalence (KL path) | **DONE, PASS** -- `r0_20261013_154325`, $1.5074 | — |
| reference calibration, stage `L` | **DONE** -- `refcal_20261013_170714`, $0.8069: λ 7.34, thresholds, overhead, cache fix validated, both equivalences PASS | — |
| D2 preregistration | draft `phase_d2/plans/d2_preregistration_v1.json` (`489f49ba…`), accepted in principle -- **SUPERSEDED 2026-10-14, never executed; the current draft is v2 (§11)** | — |
| D2 session, driver, launcher, grant type | **built, $0-tested**: `d2_session.py`, `autoinit_d2_driver.py`, `autoinit_d2_launch.py`, `d2_authorization.py`, pod branch `SESSION_KIND=d2` | — |
| D2 search price | `d2_budget.py`, switched by the booked, hash-verified stage-L evidence to `cache_fix_validated` | $10.80 expected / $22.38 bound (records price $32.01) |
| `phase_d2` in `funds_formal_sessions_of` | **owed: maintainer amendment**; the issuer refuses without it | — |
| exact engine ids for the rescoring | wrapper done, stub-validated; **real vLLM validation and cost owed** before a behavioural driver enables it | — |
| behavioural driver with the wrapper and the rescoring | owed engineering | — |
| D3 driver and preregistration (λ frozen in its identity) | after D2 | — |

**The D2 launch chain, and what each guard is for:**

- **Soft stop.** The driver's deadline is `plan.soft_stop_minutes` minus the
  session minutes already elapsed. D1's launcher read
  `plan.soft_stop_seconds`, a field `BudgetPlan` does not have, so D1's driver
  ran with no deadline and only the watchdog bounded it. D1 is closed, and its
  file is left as it is.
- **Reference-cache regression guard.** When the cached-cell price is in
  force, the first DEPTH expansion that does not cache its reference expires
  the deadline. The search then fails closed with every record written, and
  the regression is named in `d2_search.json`. A records-priced search is
  unguarded, because its price already assumes recomputes.
- **Failure path.** It keeps `d2_search.json` (budget, pricing, objective
  telemetry, deadline), the journal, the per-expansion telemetry with cache
  modes, and the state journal. None of them is required, so an early stop
  still collects what exists.
- **No readiness record.** P8.2.1 applies: the grant binds the harness by
  membership and digest, the session-commit gate binds the tree, and the
  bundle round trip binds what the pod fetches.

Behavioural screening and confirmation run on `d_series_shared_eval@v1`,
which this decision leaves unchanged.

## 11. D2, revision 6: the frozen teacher's confidence weighting a root-reference KL

**Maintainer scientific decision, 2026-10-14.** It supersedes the
student-uncertainty D2 of revisions 1-5, which was never executed: no D2
search, recovery or behavioural session ran under it. Stage L measured it as
an engineering arm only. That measurement now prices D3, which does the same
per-position work, and never the new D2.

| owner | what it holds |
| --- | --- |
| [`d2_d3_objective_contract_v1.json`](d2_d3_objective_contract_v1.json), revision 7 (`9ad4ec23…`) | `d2` and `d3` each own their objective, motivation, scope, where-it-applies, diagnostics and claim boundary (`layout`); shared keys apply to both; `d2_superseded` (the old D2, `SUPERSEDED -- NEVER EXECUTED`) |
| [`../../../phase_d2/plans/d2_preregistration_v2.json`](../../../phase_d2/plans/d2_preregistration_v2.json) | the frozen design, DRAFT; v1 (`489f49ba…`) is stamped SUPERSEDED with its science and hash intact |
| `src/aadistill/initialization/scoring/objective.py` | the generic form `reference_confidence_weighted_kl`, the full-vocabulary entropy and the confidence mapping |
| `src/aadistill/initialization/scoring/reference_store.py` | the ROOT reference store, built once per calibration mixture |
| `scripts/stages/stage-1/families/d_series/teacher_confidence.py` | the validity checks, descriptive |
| [`../validations/d2-gpu-validation/v1/plan.json`](../validations/d2-gpu-validation/v1/plan.json) | the GPU validation (stage V of `topk_adoption_driver.py`, `phase_d2/d2_gpu_validation.py`): registered plan and numeric acceptance protocol; a PROPOSAL, priced, not granted |

### 11.1 Definition

```text
L_D2(x) = (1/T_x) Σ_{t∈T_x} c_T(t) · KL(p_T ‖ q_C)_t        minimized
c_T(t)  = 1 − H_T(t) / log|V|
```

- `H_T` is the full-vocabulary Shannon entropy of the **frozen original
  teacher**. It is computed in float32 and in row chunks, and no `[T, V]`
  float32 copy outlives its chunk.
- The KL is the D-series Top-200 + tail KL, measured against the **root**
  (the teacher) for every consumer.
- `T_x` is the item's scored positions under the binary supervised-target
  policy. The denominator is fixed: no dynamic weight sum, threshold,
  exponent or hard mask.
- The weight is a property of the teacher alone, so **no candidate can move
  it**.
- Aggregation is D1's, per consumer:
  - DEPTH: per item, then sub-type, then domain.
  - State evaluation: `T_x`-weighted sub-types and pooled critical classes.
  - The beam ranks on `state.objective.*` (`OBJECTIVE_PARETO_V1`).
- The new identities are:
  - objective `objective.d2_teacher_confidence_weighted_kl@v1`, `079031e3…`;
  - the narrow-scope comparison `5c4c3aaf…`;
  - the ranking identity = policy hash + objective hash, in the
    preregistration.

  The SearchConfig, measurement protocol and operator configs all move with
  the objective.

### 11.2 Two declared changes from D1, never "one variable"

1. **DEPTH's reference changes from the parent to the root.** D1 measured
   `KL(parent ‖ parent with S bypassed)`; D2 measures `KL(root ‖ parent with
   S bypassed)`. At the search root the two coincide. A CPU test pins this:
   there the root KL of every round-0 candidate equals D1's DEPTH score.
2. **Teacher-confidence weighting.** It applies to DEPTH, state evaluation,
   the ranked metrics and, under the full scope, the activation statistics.

The two changes are only separated descriptively:

- every DEPTH round-0 candidate also records its parent-reference KL;
- every candidate and state records the unweighted root KL.

### 11.3 Scope: FULL, frozen

The FFN, residual-WIDTH and ATTENTION operators' activation statistics are
weighted by `c_T`:

- the residual sum and second moment;
- FFN `Σ|a|` and `Σa²`;
- the per-head attention-output second moment.

All of them use the denominator `Σ w`. Two details of the weighting:

- Token `j` carries the `c_T` of prediction `j`, which is the logits its
  state produces.
- The item's last token carries 0. The supervised-target token weights
  already are the prediction weights followed by that trailing zero.

**Review.** No technical blocker was found.

- **Implementation.** `ActivePositions` gained token multipliers, and the
  collectors gained a `token_weights` path on both the per-item and the
  packed path. They use the denominators `residual_weight` and
  `attn_token_weight`, which appear only in weighted states.
- **Cache isolation.** The statistics cache key carries the weighting
  store's identity, so a weighted pass and an unweighted one (the composite
  operator's) can never share an entry.
- **Identity.** `StatsSpec` is unchanged, so no historical hash moves.

These are risks, not blockers. The validity checks measure each of them
before the search:

- **Low dynamic range.** With `log|V| ≈ 11.9` nats, `c_T` may sit near 1.
- WIDTH's trace normalization cancels any uniform part of the weighting.
- Teacher-uncertain branch points lose weight.
- The effective sample size of the weighted statistics drops.

**The narrow scope is documented for comparison only.** Under it, the
statistics stay D1's and their state ids equal D1's. It is not wired.

### 11.4 Validity checks before the formal search (descriptive)

`teacher_confidence.validity_report` runs on each profile's root store in the
D2 GPU validation. It reports:

- the distribution, histogram, weight sum, mean and ESS;
- thinking vs answer (`reasoning` / `final_answer`) and the structural tags;
- domain and sub-type;
- the predicted token's type;
- the relation to the teacher's own target probability and top-1;
- with B's root KL: Spearman(`c_T`, KL), and each group's share of the
  weighted objective against its share of the unweighted KL.

**No result may change a definition, the scope or a rule.**

### 11.5 Cost (re-estimated; every new term PROVISIONAL)

| | expected | planning bound |
| --- | --- | --- |
| D2 search (`cache_fix_validated` DEPTH cell) | $11.25 | **$25.24** (cap $30) |
| for comparison: D3 search, measured student-entropy overhead | $10.80 | $22.38 |
| D2 GPU validation (proposal, §11.6) | $0.77 | $1.40 core cap, $1.54 with V5 (grant $1.70) |

The new D2's added work over D1, with the source of each estimate:

| added work | estimate |
| --- | --- |
| `c_T × KL` and stored lookups | +5% expected / +10% bound on DEPTH and state evaluation |
| the round-0 parent-KL diagnostic | 36 candidates × D1's per-candidate reduction (0.5× expected, 1.0× bound) |
| weighted statistics | +5% / +10% on the FFN, WIDTH and ATTENTION cells |
| two root stores | 2× / 5× candidate evaluations per profile |

None of these is measured. The GPU validation replaces them.

**Programme**, with live limits of $120.14 remaining and $30 per session, at
hard ceilings with reserves. The proposed validation grant is held as
engineering.

| scenario | hard ceiling | expected | outcome |
| --- | --- | --- | --- |
| D2 | $57.22 | $35.60 | fits; $62.92 left |
| D2 + D3 | $109.88 | $69.98 | fits; $10.26 left |
| D2 + D3 + one fresh confirmation | $125.38 | $82.21 | **short by $5.24** at hard; fits at expected |

The reservation rule is short by the same $5.24. **K stays 4 and the cap is
unchanged.** The levers are listed in the scenario record; none is applied.

### 11.6 The GPU validation: stage V, and its numeric acceptance protocol

The redesigned D2 has never run on a GPU. Its validation is **stage `V` of
`phase_d1/topk_adoption_driver.py`**, implemented in
`phase_d2/d2_gpu_validation.py`. It reuses the chain stage L and R0 ran on:

- the same launcher, with a live quote, a campaign-derived ceiling, a
  watchdog, evidence fetched before teardown, and teardown on every exit path;
- the same remote payload and `$0` preflight;
- the same `adoption.json`, flushed after every step.

**The plan is registered in
[`validations/d2-gpu-validation/v1/plan.json`](../validations/d2-gpu-validation/v1/plan.json).**
The proposed grant is `authorization.proposed.json`. **No `authorization.json`
exists**, so `engineering_campaign_budget.py` cannot read a ceiling and the
launcher creates nothing. A test pins this.

**Numeric acceptance falls into three classes, never mixed:**

| class | what | gate |
| --- | --- | --- |
| arithmetic | one set of inputs, two computations; details below | **gated**: entropy 1e-4 nats, confidence 1e-5, item mean 1e-5 relative, raw-KL 1e-6 relative |
| execution drift | the real BF16 model at bsz=1 vs bsz=3 (root store, state evaluation) | **never**: a different GEMM shape is a different computation, and D2 is frozen at bsz=3 |
| row equivalence | `all_v1` vs `scored_v1` at the formal bsz=3: one DEPTH round and the state evaluation | R0's 1e-5; decides `scored_v1` only |

The arithmetic class compares:

- float32 chunked entropy against float64, on the SAME teacher logits;
- chunk sizes 512 and 64;
- the confidence mapping;
- the `c × KL` item mean against float64, on the SAME (KL, c);
- the raw-KL control.

**Integrity is gated too:**

- the teacher binding and B's identities;
- every store's identity against its declaration;
- DEPTH reads the root;
- every decision score equals the recorded objective.

**Diagnostics are recorded as measured and never gate:**

- per profile:
  - the confidence distribution, with IQR, ESS and histogram;
  - the validity report, including B's KL;
- on V3's full-scope FFN child, a non-root parent: parent KL, unweighted root
  KL and weighted root KL of the same DEPTH candidates, with rank agreement,
  each metric's choice and top-3 overlap;
- the full- vs narrow-scope selection overlap:
  - FFN and ATTENTION: per-layer Jaccard;
  - WIDTH: subspace overlap.

A narrow dynamic range or an unchanged ranking is a finding. It is not a
reason to change anything.

**Cost, derived from stage L, R0 and D1's records:**

- expected 42.2 min, $0.77;
- core bound 76.4 min, giving the **session cap of $1.40**;
- with the extended V5, 84.1 min and $1.54;
- **proposed grant $1.70**.

This is $0.05 above the earlier $1.35 / $1.65 proposal, because V4 and V5 now
carry the round-0 parent-KL diagnostic's cost. The earlier plan omitted it.

The CPU toy runs stage V end to end through the same function.

