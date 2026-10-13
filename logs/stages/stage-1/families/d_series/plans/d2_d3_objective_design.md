# D2 / D3 scientific objectives, v1: definition, student identity, implementation

**Status:** contract and core implementation, 2026-10-13. Authorizes nothing.

This page follows the maintainer decision of 2026-10-13, which corrects the
D2/D3 objectives. It supersedes the D2/D3 sections of
`phase_c1/plans/phase_c_roadmap.md`; those are kept there verbatim as history,
with a pointer here.

| owner | what it holds |
| --- | --- |
| [`d2_d3_objective_contract_v1.json`](d2_d3_objective_contract_v1.json) | the registered contract: both objectives' identities, the ranking policy, the λ rule, and where each objective applies. Carries `contract_hash` |
| `src/aadistill/initialization/scoring/objective.py` | the generic forms, the student Top-K entropy and the fixed-denominator reducer |
| `scripts/stages/stage-1/families/d_series/objectives.py` | this campaign's instances (D2, D3(λ), `OBJECTIVE_PARETO_V1`) and the λ rule |
| `scripts/stages/stage-1/families/d_series/calibrate_d3_lambda.py` | the λ measurement; runs on a GPU, before D3's search |
| `scripts/stages/stage-1/families/d_series/rollout_profile.py`, `scripts/shared/evaluation/rollout_rescoring.py` | the behavioural diagnostics: text-level for every unit; per-position for new candidates |
| `tests/initialization/test_scoring_objective.py`, `families/d_series/tests/test_d2_d3_objectives.py`, `test_rollout_diagnostics.py` | 76 tests |

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

**One aggregation difference from D1, disclosed.** D1's *state evaluation*
pools tokens within a sub-type, while D2/D3 average per item first, as
decided. D2/D3 therefore also record D1's KL under the per-item rule
(`state.objective.raw_kl.*`). That lets a reader separate the effect of the
aggregation from the effect of confidence. DEPTH already averaged per item
under D1, so it has no such difference.

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
- **Items:** every item of `calib.domain_balanced@v1`,
  `calib.reasoning_heavy@v2` and `state_eval_v1`, each counted once (209
  items; 44,746 + 46,825 + 54,014 scored positions).
- **Per-item quantities:** fixed-denominator means of the Top-200 + tail KL
  and of B's own Top-200 confidence.
- **Reported as sensitivity, never used to choose:** the SD ratio, each set
  alone, and q2 as the student.
- **Refused:** fewer than 20 items, `IQR(c) = 0`, a non-finite KL, or a λ
  that is not positive and finite.

**The scale argument.** λ converts confidence into KL units. Equating the two
components' between-item spread on the reference pair means one
interquartile range of confidence trades for one interquartile range of KL,
so neither dominates the ranking by units alone. The IQR is used because
per-item KL is heavy-tailed; a test shows one extreme item does not move λ.

**When.** λ is measured **once, before D3's search**, by
`calibrate_d3_lambda.py`, and committed as
`analyses/d3_lambda_calibration.json`. Until that record exists,
`objectives.frozen_d3_objective` refuses and D3 has no objective. No D2 search
or behavioural result enters it.

**It needs a GPU.** It needs teacher and B forwards, and B is rebuilt on the
pod (1279.8 s measured). This dev box has neither the teacher's weights nor
the memory.

**Cost.** Setup 1.6 / 15 min (expected / bound), other overhead 2.8 / 13.6,
B rebuild 21.3, forwards about 3–5, teardown about 1:

| | time | all-in |
| --- | --- | --- |
| expected | about 30 min | about $0.55 |
| ceiling | about 57 min | about $1.05 |

The session wiring, which runs `frozen_baseline_spec` and then this script, is
owed before D3. It does not block D2.

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

Expect roughly D1's measured cost, about ±10%. The formal price comes from
the search cost model at each preregistration. D3 adds the λ session (§4).

**Behavioural cost** is unchanged from the shared protocol: 2K + 3 probes per
experiment.

**The choice of K is left to D2's preregistration, not taken by cost:**

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
higher fidelity**. A D2 preregistration should name a raw-KL diagnostic
threshold that triggers review.

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
| R0: `scored_v1` GPU equivalence | ready, needs its grant | $1.53 expected, $3.60 hard ceiling |
| D2 search driver: `SearchConfig(scoring_objective=D2_OBJECTIVE, policy=OBJECTIVE_PARETO_V1)`, evaluator with the same objective, `scored_v1` if R0 passes | owed engineering, $0 | — |
| D2 retention rule and K, raw-KL review threshold, preregistration, pricing | owed, maintainer decisions inside | — |
| D3 λ session wiring and run | owed before D3 | ~$0.55, ceiling ~$1.05 |
| D3 driver and preregistration (λ frozen in its identity) | after λ | — |

Behavioural screening and confirmation run on `d_series_shared_eval@v1`,
which this decision leaves unchanged.
