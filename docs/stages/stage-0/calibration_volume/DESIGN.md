# Stage-0 calibration volume: design

The first Stage-0 calibration-data study (maintainer decision 2026-10-18,
`logs/budget/decisions.md`). Results and the terminal verdict live in
`logs/stages/stage-0/calibration_volume/`; this page explains the instruments
so they can be reused without the conversation that built them.

## 1. The question and what is held fixed

Does more independent, capability-aligned calibration data:
- **H1:** stabilize the AutoInitializer's structural selection;
- **H2:** improve the recovered student on unseen rule families;
- **H3:** saturate?

The method is fixed: the forward baseline
`stage1_init.d1_target_aware.scored_v1@v1`, resolved through
`shared.forward_baseline`. That fixes:
- the four operators;
- `positions.supervised_target_v1`;
- `reference_topk_tail_v1` at K=200;
- KL references, beam width 6 with delayed pruning, Pareto v2 ranking;
- micro-batch 3, `length_sorted_v1`, `scored_v1`.

`search_session.build_session` takes every one of these from the resolver.
The experiment supplies only:
- the calibration profiles;
- the frozen geometry and state-evaluation suite (`state_eval@v1`);
- the seed and the run id.

Operator order and the profile per operator stay free search variables.

## 2. The capability-aligned battery (`scripts/.../tasks/`)

There are fifteen project-authored procedural generators over the five P10.2
axes, with no external dataset and no ARC-AGI material:

| axis | generators (calibration-eligible) | evaluation-only generator |
| --- | --- | --- |
| inductive rule discovery | `list_induction`, `string_induction` | `sequence_induction` |
| compositional generalization | `defined_ops`, `scan_language` | `counterfactual_arith` |
| task-contained deduction | `ordering`, `entity_tracking`, `knights_knaves` | `graph_paths` |
| abstract / analogical | `grid_transform` | `letter_analogy` |
| interactive adaptation | `probe_function`, `hidden_buttons` | `code_breaker` |

- **Uniqueness.** Every instance is seeded and deterministic. Static induction
  and analogy instances are kept only when every rule in the generator's
  library that fits the examples agrees on the query. Deduction instances are
  kept only when brute force finds exactly one consistent world.
- **Two isolation tiers, checked by key** (`aadistill.evaluation.isolation`):
  - **Generators.** The five evaluation-only generators never contribute
    calibration data.
  - **Families.** Inside every other generator, families are split by a salted
    hash rank (`registry.SPLIT_SALT`), about 45% calibration and 55%
    evaluation.
  - Dev and final evaluation share the evaluation families and use disjoint
    seed ranges.
  - Calibration may share no family, content hash or static prompt with any
    evaluation set; the build refuses otherwise.
- **Interactive tasks are episodes.**
  - `run_battery.py` steps a deterministic environment on the last action line
    of each reply's final answer.
  - `score_battery.py` replays the replies through a fresh environment rather
    than trusting pod bookkeeping.
- **Scoring.** Correct = usable rollout AND the canonicalized last `\boxed{}`
  equals the gold answer (the D-series rule).
- **Measurement protocol** (P18):
  - greedy decoding, with the allowance = context − prompt;
  - students at their trained 8,192-token context (the D-series rule);
  - the teacher at its configured 262,144-token context, on an 80 GB card.

## 3. Calibration data contract and the base pool

- **Teacher-native sessions** (`teacher_cal`, `calibration_v2`):
  - the official sampling preset;
  - a 16,384-token data-construction limit (never a measurement budget);
  - kept only when verified correct and naturally terminated.
- **Turn expansion for interactive episodes,** under the corpus_v2 contract
  (`final_assistant_loss_mask`):
  - turn k becomes one item whose context is the episode so far, with earlier
    reasoning dropped by the chat template and earlier answers and
    observations kept;
  - its supervised span is turn k;
  - every assistant turn is covered exactly once, and
    `positions.supervised_target_v1` is unchanged;
  - the episode is the sampling unit.
- **Supporting data.** Verified-correct corpus_v2 gsm8k, openmath,
  multihop_qa and rag_evidence, outside every session, source prompt and prompt
  text used by the recovery rung, `state_eval@v1` and the D-series batteries.
- **Labels.**
  - Items carry `domain` = axis (or `supporting`) and `subtype` = generator.
  - The DEPTH objective averages per subtype and then per domain, so these
    labels are part of the objective and are fixed once, in the pool.
- **Length bound.** `base_pool_v1` admits items up to 8,192 rendered tokens.
  - D1's [256, 2,048] window would keep 66 of the 572 verified teacher units
    and no analogy or interactive unit at all.
  - The bound is the measured answer to the audit's length-selection
    question.

## 4. Volume profiles (`build_volume_profiles.py`)

The sampling is stratified, nested and independently resampled:
- **Cells** are generators or supporting subtypes, with families pooled
  inside a cell. The distinct-family count per profile is recorded as a
  covariate of the dose.
- **Ordering.** Each cell's units are ordered by a seed-keyed shuffle.
- **Doses.** A dose takes units from the front of that order until its quota
  of scored positions is met (at least one unit per cell). Within a seed, the
  higher dose contains the lower.
- **Resamples.** A different seed is a genuinely different draw.
- **Quotas.** Dose × reference UESP × domain weight, split equally over the
  domain's cells. The proportions are the same at every dose.

The core's `resolve()` re-checks each profile's item-file hash and token
content hash on the pod.

## 5. The root-level pilot

`pilot_design.json` and `pilot_search_session.json`, with the analysis plan
frozen in `logs/.../plans/pilot_analysis_plan.json`.
- **What runs.** A search with `max_depth=1` expands only the root's
  children: every operator applied to the teacher under every profile.
- **Profiles.** Nine: 1×, 2× and 4× of 45,000 UESP, times three resamples.
- **DEPTH,** the token-linear expensive operator, runs on resamples 1–2 at 1×
  and 2× through `SearchConfig.impl_profiles`.
- **Decisions.** Read from the search's own state journal. WIDTH bases are
  recovered from the root child's embedding (E_s = E_t P).
- **Interactive excluded.** The interactive axis is not in the pilot: 13
  admissible episodes, each larger than a 1× cell quota.
- **Status.** Exploratory local-sensitivity evidence about H1/H3, never a
  substitute for the formal full search.
- **Hardware: an 80 GB card.**
  - With items up to 8,192 tokens, one `length_sorted_v1` micro-batch of 3
    carries up to 22,141 scored rows.
  - DEPTH's forward-KL sketch then holds a 12.5 GiB float32 logits tensor
    plus its temporaries.
  - The 44 GiB L40S OOMs (`runs/pilot_20261018_122808`); an A100-80GB does
    not.
  - The baseline's numerics declare cuda / bf16 / fp32 accumulation, not a
    card.
- **Results and the terminal verdict:**
  `logs/stages/stage-0/calibration_volume/closeout/REPORT.md`.

## 5a. The formal study, as amended by the maintainer (2026-10-18)

The three-resamples-per-dose design is withdrawn. The formal comparison is:

| dose | full search | recovery | evaluation |
| --- | --- | --- | --- |
| 1x (45,000 UESP) | one D1 full search | one, fixed recipe and seed | eval-set-1 |
| 2x | one | one | eval-set-1 |
| 4x | one | one | eval-set-1 |

- **Fixed across doses:**
  - the teacher and student geometry;
  - the forward baseline, resolved;
  - the recovery recipe `e1_p1_kd_heavy@0.86M` and its one seed;
  - the evaluation contract, `data/eval-set/eval-set-1` at its registered
    identity.
- **One frozen base mixture and one nested draw:**
  - the pilot's weights (0.18 per static primary axis, 0.28 supporting) and
    their frozen reasoning-heavy variant;
  - 1x ⊂ 2x ⊂ 4x;
  - the operator order and the profile per operator stay free.
- **Covariates per dose:** realized UESP, tokens, items/units,
  sources/subtypes, distinct rule families and generators. Family diversity
  rises with dose in the pilot draws, so every contrast is a
  volume-plus-diversity contrast, never a pure volume effect.
- **What it can claim:** comparisons between these three treatments only.
  - No selection variance or stability. That needs resamples; the pilot's
    root-level resampled measurement stands as the only one, and it is not
    repeated.
  - No statistical saturation.
  - The H2 intervals cover evaluation sampling only.
- **Resolution gate:**
  - **Before any search:** the reference check
    (`configs/.../resolution_session.json`) measures whether the 720 resolves
    models at student scale.
  - **After the 1x unit:** the recovered D1-initialized student must answer at
    least 2% of the static fluid items, or H2 is EVALUATION NOT RESOLVABLE.
    That verdict is measured on a D1 student, not assumed from the PCA
    students, and 2x/4x are not launched without a new decision.
- **The 4x search exceeds the $30 per-session cap. It is split, never
  shrunk:**
  - two sessions over ONE search identity;
  - a persistent network volume in one datacenter holds the journal and every
    materialized state;
  - part 2 runs the identical plan, config, seed, profiles and baseline
    resolution, and the engine's exact journal resume restores every finished
    state from disk by content id and hash;
  - a declined restore stops the run.

  The driver/launcher support for this is specified, not yet built
  (`plans/formal_design_proposal.json :: split_execution_of_4x`).
- **Launch order:**
  1. 1x search + recovery + eval-set-1, the gate;
  2. the 2x search;
  3. the 4x search, parts 1 and 2;
  4. 2x and 4x recovery + eval-set-1.

  All sessions share the volume, which also keeps every Top-1 and recovered
  student durable.
- **Price:** `plans/formal_design_proposal.json`; the verdict and the decision
  it needs are in `closeout/REPORT.md`.

## 6. Sessions and money

- **One governance type per session kind**
  (`governance.py :: CalVolAuthorization`).
  - It is bound to the session plan hash and the harness digest.
  - It is issued only inside the programme grant's phase envelope, which is
    read from this experiment's own closeouts.
- **Reused runner.** Sessions run through `SessionRunner`: watchdog, poll
  loop, teardown, provider confirmation.
- **Durability.** `launch.py` pulls every finished unit off-pod and
  hash-checks it as it completes.
- **Booking.** Spend is booked from each run's
  `closeout/outcome.json :: money.all_in_usd`, which `derive_budget.py`
  discovers through `logs/index.json`.

## 7. Where things live

| what | where |
| --- | --- |
| executables (audit, generators, builders, runner, scorer, governance, launcher, drivers, closeouts) | `scripts/stages/stage-0/calibration_volume/` |
| experiment tests | `pytest scripts/stages/stage-0/calibration_volume/tests` |
| calibration prompt sets and manifests | `data/stages/stage-0/calibration_volume/tasks/` |
| the dev battery (`dev_v1`), since 2026-10-18 a registered evaluation set | `data/eval-set/eval-set-1/fluid_reasoning_720/` (old path resolves through `logs/index.json :: historical_paths`) |
| audit inputs, session plans, artifact specs, pilot design, profile document | `configs/stages/stage-0/calibration_volume/` |
| audit, design analyses, frozen plans, runs, programme closeout | `logs/stages/stage-0/calibration_volume/` (README generated) |
| heavy bytes: raw generations, base pool, profile items, search records | `artifacts/stages/stage-0/calibration_volume/` (gitignored; each run's closeout names them by sha256) |
| money | `logs/budget/approvals/stage0_calibration_volume_programme_grant.json` |
