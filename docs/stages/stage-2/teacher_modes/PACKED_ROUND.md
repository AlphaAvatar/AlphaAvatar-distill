# Research A — packed round: Stage-1 forward and Stage-3 training packing, preparation for the new search and trainings

**Status: `PREPARATION_BLOCKED` (2026-11-04).** The maintainer's brief "Stage-1 Forward / Stage-3 Training Packing
重构及 Research A 前置准备" authorized code, data protocol, $0 data preparation, tests and cost estimates only.
- Code, identities, tests, the $0 data audits, the new full-search chain and the training plans are done.
- The one real blocker is the tool-calling data. Its sources are candidates under the maintainer's review (section 5),
  and its teacher-native generation is a paid step that needs its own authorization. The code for every step from
  source rows to trained arms is in place (section 5a).
- The new full search cannot start without tool items in its calibration and state evaluation.
- Nothing was searched, trained, generated or evaluated. No pod was created and `main` is unchanged.
- **The maintainer's independent review of `649d25dd` (2026-11-03)** kept the packing core and its consumers.
  - It accepted the fragments (`prefix_restart_v1`) as specified, so they are no longer a decision owed.
  - It ruled that every formal data volume and proportion is counted before packing (AGENTS.md P19).
  - It asked for the $0 repairs recorded in sections 5, 5a and 7: the tool pool's isolation and length bound,
    Toucan's categories and replay conflicts, verification by level, each lineage's generation kind, and the packs'
    validation and share check.
- **The maintainer's review of `77be6e13` (2026-11-03)** confirmed those repairs and asked for six more, done at $0
  (sections 5, 5a and 8):
  - the tool generation is BOUNDED by a frozen plan, so it no longer walks the whole pool;
  - the pool's split files are content-identified and re-verified at every read;
  - exact prompts are isolated across splits;
  - schema validation and the benchmark caches fail closed;
  - the Pivot pair is converted and decontaminated;
  - the `a_search` position reconciliation and an end-to-end consumer regression are added.
  Loss normalization stays as it is, and no further ruling is owed on it.
- **The maintainer's third review, of `acbbbbfb` (2026-11-03),** confirmed those repairs and asked for two more,
  done at $0 (sections 5 and 5a):
  - Calibration v2 and `state_eval_hybrid@v2` now check every tool category on its own. A category that falls short,
    or is THIN, is never covered by another category's surplus. Nothing freezes unless the maintainer has approved
    that category's shortfall.
  - Every consumer now reads the pool and the plan through the same identity check as the session.
  - Toucan SFT and Pivot Conversational may be prepared as priority candidate sources. Pivot Function-Calling is not
    adopted, and Nemotron-SFT-Agentic-v4 stays deferred. None of this authorizes paid generation.
- **The maintainer's review of `fc6fa1d6` (2026-11-04)** kept the per-category gates and the unified identity check,
  and ruled that data selection, proportion validation, admission / freeze and sequential packing are four separate
  functions. Done at $0 (sections 7 and 9):
  - `packed_packs.py` now runs `select`, then `proportions` (the generic `aadistill.data.mixture.validate_proportions`),
    then `admit` (a frozen `data_manifest.json`, or a refusal with nothing written), then `pack`. The packing judges
    no proportion; it checks that its input is the manifest's documents. The training binding reads only a pack
    `verify_pack` accepts.
  - The withdrawn `share_check` (each type within 2% of 1/6, fluid 49-51%) was not the registered tolerance; the
    admission reads the registered one (`audit/proportion_tolerance_audit.json`).
  - Option 1 (call-only gold episodes) and option 2 (WorkBench) for the non-thinking schema_arguments SHORTFALL are
    compared at $0 (`audit/schema_arguments_options.json`); neither is implemented.
- **The pilot's one authorized launch (2026-11-04): NOT RUN.** `a_tool_pilot_20261104_140656` passed every gate, but
  no L40S instance existed in EUR-IS-2 during its $0 create window (10 × 120 s). Nothing was created, $0.00 was
  spent, the artifact is consumed and every measurement is UNMEASURED. Another attempt needs a new maintainer
  authorization.
- **The maintainer's metric review and pilot authorization (2026-11-04, after `0b96ed38`).** The pilot (option B) is
  authorized: the pilot-only source admission is signed and the grant amended exactly as proposed. The L40S check's
  whole-phase gate refused it (forward baseline v2 changed `selection.json`), so the pilot is admitted by the
  maintainer's generation-scoped gate instead (`l40s_check.generation_scope_problems`, section 8). The tool-call
  verifier now reports only what it checked: `recorded_execution` is None without a call, and `calls_tools` is
  `target_tools_called`, which measures recall. The pilot report names admission rates. See
  [`docs/shared/METRIC_CONTRACT.md`](../../../shared/METRIC_CONTRACT.md). Admission verdicts are unchanged.
- **The maintainer's final authorization review of `56381f55` (2026-11-04): conditional GO for option B.** Two $0 items,
  done (section 8): issue requires the grant's explicit pilot-only scope (drafted as an amendment, the envelope
  reconciled read-only, nothing raised), and resumption is bound to the pilot plan, the round input, the protocol, the
  verification version and the generation implementation. The grant and the source admission are untouched.
- **The maintainer's sixth review, of `29b62c93` (2026-11-04),** accepted the truncation fix, `tool_verification@v2` and
  the pilot's structure, recommended option B ($2.50) and asked for the following, done at $0 (section 8): the pilot
  report reads partial, verified results with explicit states; the source admission's signature is checked strictly by
  one shared function; the volume is checked read-only now and measured on the pod. Nothing is authorized.
- **The maintainer's fifth review, of `2468c1aa` (2026-11-04),** accepted the identity hardening and the single-step
  protocol and asked for the following, done at $0 (sections 5, 7, 8):
  - `verify_pack` verifies truncated documents instead of refusing them (the frozen documents re-packed);
  - V1's tail rule on every turn that makes a call (`tool_verification@v2`), with a $0 audit;
  - the L40S pilot's execution preparation: its own kind and refusals, the frozen cells and task file, the session
    wiring, telemetry, the runner's hard-cap check, the pilot-only source admission (unsigned) and a re-verified
    application. Nothing is authorized.
- **The maintainer's fourth review, of `647eea0e` (2026-11-04),** accepted and froze the packing protocol, the
  pre-packing accounting and the four-step decoupling, and asked for the following, done at $0 (sections 5, 7, 8, 10):
  - the admission takes only the proportion report of the selection it admits (`selection_identity`);
  - the training binding verifies the actual packed tokens and supervision (`packed_content_sha256`, the documents
    against the manifest, the sequences re-derived);
  - option 1 implemented as `single_step_gold_tool_call@v1`, with the dry pool and plan rebuilt; no SHORTFALL approved;
  - the token identity audit's intermittent failure diagnosed and repaired; A2's missing task recovered from the
    frozen records; the environment-blocked tests recorded;
  - an L40S pilot application prepared, not authorized.

| record | owns |
| --- | --- |
| [`packed_round_design.json`](../../../../configs/stages/stage-2/teacher_modes/packed_round_design.json) | this round's registered terms: forward context and training packing (sequence length, seeds), calibration v2, `state_eval_hybrid@v2`, the new search, the three arms and their lineage, budget, decisions owed |
| [`initialization_forward_baseline_v2.json`](../../../../configs/shared/baselines/initialization_forward_baseline_v2.json) | the official Stage-1 method, version 2 (D1's method in the packed forward context), selected in `selection.json`; v1 superseded and still loadable |
| [`audit/packing_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/packing_audit.json) | the packing applied to existing data: Stage-1 sets; A1, A2 and C1's kept part (`packing_audit.py`) |
| [`plans/packed_search_budget.json`](../../../../logs/stages/stage-2/teacher_modes/plans/packed_search_budget.json) | the new search's cost estimate (`packed_search_forecast.py`) |
| [`plans/packed_training_budget.json`](../../../../logs/stages/stage-2/teacher_modes/plans/packed_training_budget.json) | steps, schedule and cost of A1-6cat, A2-6cat, C1-6cat (`packed_training_plan.py`) |
| [`validations/packed_round/dry_run_20261103.json`](../../../../logs/stages/stage-2/teacher_modes/validations/packed_round/dry_run_20261103.json) | the $0 dry run of the new search chain, and the historical search's identities reproduced |
| [`data/six_category_selection.json`](../../../../logs/stages/stage-2/teacher_modes/data/six_category_selection.json) | the five existing base types re-selected at 1/6 each (`six_category_selection.py`; see section 7) |
| [`audit/tool_pool_dry_run_v2.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_pool_dry_run_v2.json) | Toucan converted and assembled dry through `tool_pool.py`'s entry point: census, verification levels, atoms, every exclusion by reason, source, category and split, the decontamination references |
| [`audit/tool_pool_dry_run_v3.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_pool_dry_run_v3.json) | the candidate pool (Toucan + Pivot Conversational) dry: identity (every split file's sha256), census, verification levels, atoms, every exclusion, T1/T2/T3 findings, the seven categories' coverage |
| [`audit/pivot_conversion_v1.json`](../../../../logs/stages/stage-2/teacher_modes/audit/pivot_conversion_v1.json) | the Pivot pair rebuilt from the first user turn: what is kept and skipped, step gold, replay, feasibility, coverage |
| [`plans/tool_generation_plan_dry_run.json`](../../../../logs/stages/stage-2/teacher_modes/plans/tool_generation_plan_dry_run.json) | the bounded generation plan over the dry pool: every cell's target, supply, expected tasks and bound, and the price |
| [`audit/schema_arguments_shortfall.json`](../../../../logs/stages/stage-2/teacher_modes/audit/schema_arguments_shortfall.json) | the non-thinking schema_arguments SHORTFALL: reach, impact, the verifiable options and the minimal preparation |
| [`audit/schema_arguments_options.json`](../../../../logs/stages/stage-2/teacher_modes/audit/schema_arguments_options.json) | option 1 (call-only gold episodes) against option 2 (WorkBench): volume per split, reach, verification, protocol impact, cost (`schema_arguments_audit.py`) |
| [`audit/single_step_supply_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/single_step_supply_audit.json) | the single-step implementation's dry pool (schema v4) against v3: every earlier task's admission, split and order; `schema_arguments` supply per split and shape; the plan's `schema_arguments` cells; reach at the priors (`schema_arguments_audit.py --supply-pool`) |
| [`audit/a2_recovery_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/a2_recovery_audit.json) | A2-6cat's one selected task outside A2's pack, recovered from the frozen records, with the counts it completes (`packed_packs.py --recovery-audit`) |
| [`budget/applications/research_a_tool_pilot_2026-11-04/application.json`](../../../../logs/budget/applications/research_a_tool_pilot_2026-11-04/application.json) | the proposed L40S tool pilot: sample, measurements, price, hard cap, stop conditions; authorizes nothing (`tool_generation_plan.py --pilot-application`) |
| [`plans/tool_pilot_plan.json`](../../../../logs/stages/stage-2/teacher_modes/plans/tool_pilot_plan.json) | the pilot's frozen cells: exact task ids per mode and cell, every task's sha256, the task file's sha256, the pool, protocol, design and plan cells it binds (`tool_pilot.py build`) |
| [`audit/tool_pilot_source_admission.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_pilot_source_admission.json) | Toucan SFT and Pivot Conversational admitted for the pilot's measurement only; SIGNED 2026-11-04 by the maintainer's written decision (`tool_pilot.py admission`) |
| [`audit/l40s_generation_scope.json`](../../../../logs/stages/stage-2/teacher_modes/audit/l40s_generation_scope.json) | the L40S check's identity files at its session commit (reproducing `848f6508…`), the engine its generation ran and the measurement, read by the generation-scoped gate (`l40s_check.py generation-scope`) |
| [`validations/packed_round/pilot_storage_precheck_20261104.json`](../../../../logs/stages/stage-2/teacher_modes/validations/packed_round/pilot_storage_precheck_20261104.json) | the pilot volume read-only: the provider's report, the recorded contents, where the pilot writes, the pod-side rule (`tool_pilot.py storage-precheck`) |
| [`audit/v1_tail_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/v1_tail_audit.json) | V1's tail rule on the hybrid's recorded tool-call turns (`tool_admission.py --v1-tail-audit`) |
| [`validations/packed_round/environment_blocked_20261104.json`](../../../../logs/stages/stage-2/teacher_modes/validations/packed_round/environment_blocked_20261104.json) | the 25 simulator-restore tests the core suite could not run (11 GiB free, 20 GiB gate unchanged): ENVIRONMENT_BLOCKED, not verified |
| [`audit/proportion_tolerance_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/proportion_tolerance_audit.json) | the registered proportion tolerance on every arm's real data, beside the withdrawn share check (`packed_packs.py --proportion-audit`) |
| [`audit/experiment_suite_baseline_20261103.json`](../../../../logs/stages/stage-2/teacher_modes/audit/experiment_suite_baseline_20261103.json) | the four experiment-suite failures run at `b7ee0c93` and `c861d704`: pre-existing, with their causes |

## 1. The rule, by stage (AGENTS.md P19)

| stage | context |
| --- | --- |
| Stage 1 search forwards | random shuffle → sequential packing → truncation → standard causal attention |
| Stage 3 recovery / SFT / KD | the same |
| Stage 0 | not packed by rule; data that Stage 1 consumes is read under Stage 1's rule |
| Stage 2 | prepares teacher-native data and builds Stage 3's packed training set |
| Stage 4 / 5 / 6 | the real context, fixed before generation; standard causal attention; rollouts are batched, never packed together |

The packing is one generic implementation, [`aadistill.data.sequence_packing`](../../../../src/aadistill/data/sequence_packing.py).
It names no model, stage, sequence length or seed.

- **Order.** Examples are sorted by `sha256("sha256_keyed_v1||<seed>|<stable id>")`. The result is a pure function of
  the ids and the seed, independent of the input order. Removing an example keeps every other example's relative
  order.
- **Packing.** The maintainer's `packing_conversation` is kept verbatim, with `truncate=True`.
  `sequential_pack` reproduces its index lists and checks them against the reference on every call.
- **Token materialization** (`prefix_restart_v1`):
  - every placement is a prefix of its example: `tokens = min(length, room)`;
  - an example that does not fit the room left fills it with its first tokens (a fragment);
  - it is read again from its first token in the next sequence;
  - an example has at most one fragment, and the fragment is the last placement of its sequence;
  - tokens beyond the sequence length are never placed (truncated tokens).
  - A fragment's supervised and scored positions are trained and scored like any placement, and counted as
    repeated exposure.
- **Attention.** Each packed sequence is read with standard causal attention and position ids `0..T-1`. There is no
  block-diagonal mask and no per-example reset. Padding appears only at the end of the last sequence. The position
  after an example predicts the next example's first token, which is a prompt token, so it is never supervised.

`tests/data/test_sequence_packing.py` checks:
- index lists against the reference on 40 random length sets;
- worked examples, and the materialization rule;
- that the accounting closes (unique − truncated + repeated = placed);
- the shuffle;
- the Stage-3 pack's tokens, masks, positions and padding.

## 2. Stage 1: old and new forward protocol

| | v1 (every search before 2026-11-03) | v2 (packed forward context) |
| --- | --- | --- |
| composition | one example per row; right padding; `original_order_v1` or `length_sorted_v1` | examples packed into sequences of the experiment's length (8,192 here) by the rule above |
| what an example sees | itself | itself and every earlier example of its sequence |
| batch size | examples per forward (3 in v1's record; 1 in the hybrid search) | packed sequences per forward (1) |
| identity | execution knob (fingerprint only) | science: `forward_context` in every calibrated operator's hashed config, in the search's `config_hash`, in the measurement protocol id, in the statistics-cache key |
| unchanged | `supervised_target_v1`, Top-200 + tail, `scored_v1`, DEPTH vs the parent, state evaluation vs the root, beam, Pareto v2, the four operators | the same |

**Where it lives** (generic; every instance value is an argument):
- `ForwardContext` (`calibration/packing.py`) and packed `ItemBatch` segments (`calibration/batching.py`);
- segment masks in `ActivePositions` (`scoring/batches.py`);
- the statistics operators (FFN, ATTENTION, WIDTH, composite) through `collect_activation_stats` and attention's own
  loop;
- DEPTH's packed scorer (`_PackedDepthScorer`):
  - the parent's sketch per packed group;
  - per-example means pool every placement;
  - float32 KL in 2,048-row chunks;
  - prefix reuse supported;
- the state evaluator's packed path: teacher sketches cached per packed segment, so the teacher is forwarded once per
  group;
- `SearchConfig`, `BeamSearch` (it refuses a measurer reading another context) and `FixedPathSpec`.

**Refused under a forward context.** These paths would still read one example per row, so they are refused:
- full-vocabulary support, the union support and scoring objectives;
- a declared root KL reference;
- `attention.causal_kl_v1`;
- `CACHE_IN_MEMORY` references.

**Evidence** (`tests/initialization/test_forward_context.py`, on a tiny real Qwen3):
- segment contents match their examples;
- the first segment reads as its example alone, and a later segment does not;
- default positions equal explicit continuous ones;
- masks land at segment offsets, and the boundary position belongs to no example;
- DEPTH's packed scores match a from-scratch computation, and prefix reuse matches too;
- the packed state evaluation matches a from-scratch per-segment reduction, and the teacher against itself scores 0;
- the context moves the operator config hash, the protocol id and the search config hash;
- a real `BeamSearch` (DEPTH + FFN) runs and every state records the context.

**On the hybrid's existing sets** (`audit/packing_audit.json`, sequence length 8,192, seed 20261105):

| set | examples | packed sequences | tokens: unique → placed | scored positions: unique → placed (repeated) | largest scored rows in one sequence | Σ T² vs one example per forward |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `calib.hybrid_d4_base@v1` | 565 | 64 | 472,966 → 521,613 | 197,084 → 219,083 (21,999, 10.0%) | 7,437 | 5.92× |
| `calib.hybrid_d4_rh@v1` | 377 | 47 | 339,452 → 379,534 | 198,206 → 222,986 (24,780, 11.1%) | 7,549 | 5.57× |
| `state_eval_hybrid@v1` | 1,135 | 131 | 956,195 → 1,068,600 | 350,415 → 403,017 (52,602, 13.1%) | 7,736 | 5.69× |

No example exceeds 8,192 tokens, so nothing is truncated. **`truncate=False` would give exactly the same sequences**
(64 / 47 / 131), because a fragment occupies the slot that would otherwise be padding. The rule's only effect on this
data is the fragments: 10–13% of scored positions are scored twice, in another context.

## 3. Stage 3: old and new training protocol

| | Research A v2 (A1, A2) | packed round |
| --- | --- | --- |
| packer | `pack_isolated`: best fit, whole documents, positions restart per document, block-diagonal attention | `pack_sequential`: seeded shuffle, sequential, truncation, continuous positions, standard causal attention |
| trainer | `batch.context_strategy: document_isolated` | `cross_context` (the default); the pack carries no position ids, and a sequential pack cannot be declared isolated |
| epochs | per-epoch permutation of the packed sequences | unchanged |
| objective | CE 0.25, KD 1.0, T 1.0, assistant target positions, Top-200 + tail, 3 epochs, batch size 2, micro-batch size 1, seeds 20260726 / 20260801 | unchanged |

The trainer needed no change: it already ran both models on the same packed ids without an attention mask.
`tests/training/test_sequential_pack_training.py` checks three things:
- CE and KD land exactly on the pack's supervised positions;
- a step trains on the packed context;
- an isolated declaration is refused.

**The three layers, on A1's and A2's existing documents** (taken token for token out of their isolated packs and
repacked at 8,192, seed 20261106):

| | A1 (thinking mode) | A2 (non-thinking mode) |
| --- | ---: | ---: |
| **A. selected data** examples (one turn each) | 6,105 | 53,136 |
| unique supervised tokens | 5,930,273 | 5,923,646 |
| prompt / context tokens | 2,994,498 | 31,406,627 |
| reasoning / answer tokens | 4,983,377 / 946,896 | 0 / 5,923,646 |
| **B. training coverage** packed sequences (isolated) | 1,288 (1,102) | 4,935 (4,577) |
| non-padding / padding tokens | 10,545,409 / 5,887 | 40,426,422 / 1,098 |
| repeated tokens / repeated supervised tokens | 1,620,638 / 1,110,226 (18.7%) | 3,096,149 / 257,500 (4.3%) |
| truncated tokens | 0 | 0 |
| CE target = KD scored positions per epoch | 7,040,499 | 6,181,146 |
| supervised-token exposure, 3 epochs | 21,121,497 | 18,543,438 |
| optimizer steps (isolated) | 1,932 (1,653) | 7,403 (6,866) |
| training FLOPs, linear term (isolated) | 6.01e17 (5.14e17) | 2.30e18 (2.14e18) |
| **C. loss weighting** supervised positions per step: mean, CV | 10,933, 0.20 | 2,505, 0.46 |
| category weight / exposure ratio, min–max | 0.92–1.14 | 0.84–1.28 |

- **(C) is the trainer's own normalization.** Each step divides its CE and KD sums by that step's supervised
  positions, so a token weighs (0.25 + 1.0) / N_step.
- **The skew is not new.** Categories whose sequences carry fewer supervised tokens (reading: long evidence, short
  answers) carry more weight than their exposure share. The isolated packs show the same skew (0.92–1.12 and
  0.83–1.27).
- **Nothing was changed.** It is reported, not corrected; the objective is frozen.
- **Layer A is the data's size** (maintainer 2026-11-03). Examples, turns, unique supervised tokens and the category
  shares are counted before packing. A1 stays at 5,930,273 unique supervised tokens whatever its sequences repeat.
  Layer B is packing, compute and exposure; layer C is the objective's arithmetic. Neither redefines a size or a
  share, and no loss weight corrects a quota.

**C1's kept part.** `D4A_merged_whole`'s 3,681 tool-free sessions, each its block's system prompt plus its own tokens,
5,430,434 supervised tokens, 352 historical cut traces kept as cut. Repacked, they fill 1,042 sequences (989 before)
with 1,265,950 repeated supervised tokens (23.3%).

## 4. Stage 5: on-policy distillation is not packed

A rollout's context is fixed before generation. `prompt_tokens` is what the engine was given, rendered from the real
conversation by the policy's own template.
- Every on-policy quantity reads that rollout's own `prompt_tokens + tokens` causally. `rollout.snapshots.score_tokens`
  takes one rollout at a time.
- Rollouts may be batched with padding. They are never packed together or re-composed after generation.
- An off-policy SFT on rollouts would be a separate, separately authorized experiment.
- The contract is stated in `aadistill/rollout/snapshots.py` and AGENTS.md P19. No Stage-5 framework was built.

## 5. Tool calling

The pool and protocol are [`SIX_CATEGORY.md`](SIX_CATEGORY.md)'s: `tool_pool@v1`, seven categories, the
6,144-token prompt bound, whole-episode admission, and only native assistant ids supervised. The maintainer's review
found five defects, repaired at $0:

| defect (`649d25dd`) | repair |
| --- | --- |
| a task's family was its servers joined into one string (`a\|b`), so one MCP server could land in two splits | splits per **atom**: every declared MCP server, and every tool no declared server owns (bare name + schema). A task enters the split all its atoms share and is refused otherwise; `assemble` asserts that no atom is in two splits. Connected components are not an option: Toucan's multi-server tasks chain 99.8% of it into one |
| `main()` passed no length checker, so the 6,144-token bound was never applied | `length_checkers`: each lineage's and mode's own tokenizer, official template, tools and system text, exactly `run_episodes`' first turn |
| "parallel" was two calls in total; "missing-parameter recovery" was any response containing `error`, including `"error": null` | steps are read from the trajectory: parallel = a step with two or more calls. An error is one the response reports. `error_recovery` is split into `corrected_retry` (an error, then a successful call of the same tool with other arguments) and `tool_failure`. No Toucan task is "ask for the missing value" |
| a call recorded twice with different responses kept the first silently | listed under `backend.conflicts`; the pool refuses the task |
| `calls_tools` set `task_success`, and checks without a verifier passed | `verify_episode` reports `LEVELS`: protocol, schema validity, recorded execution, step gold, tool selection, task success. A level without a verifier is `None`, never passed. A target-tools check is tool selection, and every task registers its levels |
| (`77be6e13`) the session generated every task of its splits, pass 1 for every refusal: about 238,790 first candidates for ~493K supervised tokens per arm | the **bounded plan** (`tool_generation_plan.py`, section 5a): per mode, split and category a target in the consumer's unit, the frozen order, rounds that stop at the target, a hard bound |
| (`77be6e13`) `pool.json` registered no split file; the session read the files unchecked | `pool.json` registers every file's sha256, task count, sources and order; `load_pool` re-verifies all of it at every read and refuses on any difference; the tool kinds bind `pool.json` and the plan |
| (`77be6e13`) one prompt in several splits could pass the 13-gram screen (template n-grams are dropped) and T5 (keyed by family) | T4 also refuses an earlier split's exact prompt key and any earlier split's tool identity |
| (`77be6e13`) `schema_problems` checked a subset and ignored other keywords | the common assertion keywords are enforced (numbers, strings, arrays, `const`, `oneOf` / `allOf` / `not`, local `$ref`, `additionalProperties` as a schema); any other keyword fails closed, and the pool refuses such tasks up front |
| (`77be6e13`) a cached benchmark copy was reused without checking it | `tool_benchmarks.fetch` re-hashes every registered file on every use and refuses a missing or changed one |
| (`acbbbbfb`) a calibration tool cell and a state-evaluation mode were judged on their totals, so one category's whole-episode surplus could hide another's SHORTFALL or THIN | `packed_inputs.tool_cell` selects each category to its own share of the cell (the plan's shares, unchanged) and its task minimum. A cell or mode is met only when every category is. A category with no admitted episode is short. A shortfall counts as met only if the maintainer approved it (`bounded_plan.approved_shortfalls`, carried and checked by the plan); otherwise nothing is frozen |
| (`acbbbbfb`) `for_consumers` compared only the pool's content sha | `tool_generation_plan.verified` is the one check for the session and every consumer. It re-verifies the pool files, the orders and a content sha that now includes the orders, then requires the plan's bindings to pool.json's exact bytes, the protocol and the design, and unchanged cells |

**Candidates, none approved** (maintainer, 2026-11-03).
- Review: [`audit/tool_calling_source_review_v2.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_calling_source_review_v2.json),
  built on review v1 and not repeating it.
- What was read: each source's card, pinned revision, licence and real rows. Toucan's SFT config and ToolACE were read
  in full, the Pivot files' first 20,000 step rows, and HTTP range heads of the two Nemotron files.

| rank | candidate | revision / licence | tool results | gold / verifier | role |
| --- | --- | --- | --- | --- | --- |
| 1 | Agent-Ark/Toucan-1.5M (SFT config, 119,287 rows) | `0df3cf37` / Apache-2.0 | **real MCP executions** (June–Sept 2025), failures and recovery included | target tools and order (G2); no-call (B) | primary: selection, no-tool, parallel, multi-turn, errors, MCP |
| 2 | NeMo Gym Pivot pair (9,620 + ~97–170k step rows) | `0ac34ec8`, `bfc7a4dd` / CC-BY-4.0 | recorded, model-generated | **deterministic per-step `expected_action`** (G2); never task success | selection and arguments; trajectories rebuilt so the teacher runs from step one |
| 3 | nvidia/Nemotron-SFT-Agentic-v4 `tool_calling` (549,707 rows) | `a4593a26` (2026-10-15) / OpenMDW-1.1 | model-generated (Qwen3.5-397B-A17B, GLM-5.2) | none (agreement with the generator's call, G2-weak) | conditional supplement once three blockers are resolved |
| 4 | nvidia/Nemotron-Agentic-v1 `interactive_agent` | `650d5909` / CC-BY-4.0 | simulated | none | hold; its `tool_calling` subset excluded (Glaive tool sets, T3) |
| 5 | Team-ACE/ToolACE (11,300 rows, 93% single-turn) | `6bda777c` / Apache-2.0 tag | simulated | none | not recommended (undisclosed generators, 2024) |

**Is a new source clearly better?** No single candidate dominates.
- Nemotron-SFT-Agentic-v4 is the newest and largest, but it adds scale without a verifier or real results.
- Toucan's real executions and the Pivot sets' step gold are what the verification protocol needs.
- Recommendation: Toucan plus Pivot as the base, and v4 as a conditional supplement. Nothing frozen is replaced.

**v4's blockers:**
- OpenMDW-1.1's terms are not read in full;
- GLM-5.2's output terms are not reviewed;
- the origin of its tool sets is unstated, and its predecessor v2 drew on Glaive/xLAM (T3).

**Pivot** ([`audit/pivot_conversion_v1.json`](../../../../logs/stages/stage-2/teacher_modes/audit/pivot_conversion_v1.json),
`tool_sources_pivot.py`). A Pivot row is one step, and its context already holds the expert's earlier turns. So a
task can be rebuilt only from the first user turn, and the teacher then runs every step itself:
- **Step gold is real.** Each step's expected action is compared exactly after canonicalization; a message step must
  make no call.
- **Replay is partial.** A gold call is answered only if a later row recorded the expert's identical call.
- **Later user turns are not used.** They were written in reply to the expert's message.
- **Conversational:** 9,051 tasks — no_tool 7,265, tool_selection 1,112, schema_arguments 430, multi-step 244. For
  9,558 trajectories a gold call's output is missing. The data has no parallel calls and no recorded errors.
  Qwen3-235B's per-step pass rates, multiplied over a task's steps, have a median of 0.375; under exact comparison a
  4B teacher should expect less.
- **Function-Calling fails T3.** Of its 78 rebuildable tasks, 15 repeat an xLAM-60k or Glaive-v2 prompt exactly, so
  it is left out of the pool.

**Isolation and decontamination** (Toucan and Pivot Conversational in full,
[`audit/tool_pool_dry_run_v3.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_pool_dry_run_v3.json)):
- **T1 (eval-set-1).** 0 prompt keys and 0 near-duplicates. 1,878 tasks were refused for a held-out tool's bare name
  (`factorial`, `get_news`); none shares a name and a schema.
- **T2 (BFCL v3, When2Call, API-Bank, tau-bench, ToolSandbox).** 1 exact prompt refused; 0 near-duplicates; 0 tool
  identities.
- **T3 (xLAM-60k, Glaive-v2, newly pinned and verified).**
  - Refused: 7 prompt matches in Toucan, and 56 Pivot tasks with a tool identical in name and schema to a Glaive
    tool.
  - Tool signatures (name plus parameter names) are counted, not refused. Refusing them removed 20,536 tasks in a
    first run, all through 22 generic signatures served by real MCP servers: `get_definitions(word)` 6,907,
    `get_current_weather(location)` 3,737, `get_user_profile(...)` 4,507, and so on.
- **T4 (between splits).** 2,512 exact prompts and 7 near-duplicates of an earlier split. 6,825 tool identities
  already used by an earlier split: the same bare tool and schema served by several MCP servers. 5,707 tasks whose
  atoms disagree.
- **Schema.** 157 Pivot tasks use keywords the validator does not check (fail closed).

**Data prepared ($0, dry)** into scratch, not under `data/`. 83,764 of 128,338 tasks were admitted, and every split
file is registered with its sha256 (content `98fed03b…`; the sha now also covers the frozen orders):

| split | tasks | no_tool | tool_selection | schema_arguments | multi_turn | multiple_parallel | mcp_agent | error_recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 74,762 | 37,880 | 12,499 | 368 | 12,224 | 4,581 | 4,542 | 2,668 |
| validation | 1,517 | 867 | 230 | 5 | 181 | 51 | 133 | 50 |
| state_eval | 2,312 | 1,535 | 333 | 7 | 205 | 125 | 78 | 29 |
| calibration | 4,814 | 2,479 | 1,050 | 28 | 685 | 427 | 105 | 40 |
| reserve | 359 | 329 | 10 | 1 | 5 | 5 | 4 | 5 |

- **Atoms:** 498 in total (333 servers, 155 Pivot environments, 10 tool identities). Each evaluation split holds 17–20
  of them, and no atom is in two splits.
- **Verification levels:**
  - Toucan tasks verify protocol, schema validity, recorded execution and tool selection;
  - Pivot tasks verify step gold instead of tool selection;
  - nothing verifies task success.

**The seven categories** (the plan uses the design's proposed shares unchanged; a short category is reported as
SHORTFALL and never filled from another):

| category | source and verification | gap |
| --- | --- | --- |
| tool_selection | Toucan (tool selection); Pivot (step gold) | — |
| schema_arguments | Pivot Conversational only (step gold): 368 training, 28 calibration, 7 state-evaluation tasks | **SHORTFALL in non-thinking mode** (training 43,277 of 74,013 tokens expected; state evaluation 823 of 1,800 positions). It refuses the A2-6cat pack and blocks the suite's freeze. Options are below |
| no_tool | Toucan irrelevant subset, Pivot message steps (tool selection: no call) | — |
| multiple_parallel | Toucan only (tool selection; no gold parallel arguments) | no gold arguments |
| multi_turn | Toucan (2–4 user turns), Pivot (sequential steps, step gold) | — |
| error_recovery | Toucan's recorded errors (`corrected_retry` / `tool_failure`; tool selection) | **"ask for the missing value": no verified source in any mode** |
| mcp_agent | Toucan only (tool selection) | no task-success check |

**The non-thinking schema_arguments SHORTFALL**
([`audit/schema_arguments_shortfall.json`](../../../../logs/stages/stage-2/teacher_modes/audit/schema_arguments_shortfall.json)):
- **Impact.**
  - A2-6cat's tool type would come in about 6% short, and the data admission refuses that selection (the short cell
    and the supporting category below its target).
  - The non-thinking category of `state_eval_hybrid@v2` stays short, so the suite cannot freeze. The packed search
    needs the suite, so the search cannot start either.
  - Thinking mode's 7 state-evaluation tasks may also leave it THIN.
- **Option 1 against option 2, measured at $0** (the maintainer's review of `fc6fa1d6`;
  [`audit/schema_arguments_options.json`](../../../../logs/stages/stage-2/teacher_modes/audit/schema_arguments_options.json),
  `schema_arguments_audit.py`). Neither is implemented; reach is a prior until the pilot measures the pass rate.

  | | option 1: call-only gold episodes (Pivot Conversational) | option 2: WorkBench generated instances |
  | --- | --- | --- |
  | schema_arguments supply | 1,595 training, 72 calibration, 22 state-evaluation, 19 validation tasks, split by the frozen environment rule | 2 of 69 templates qualify (direct `analytics.create_plot` requests, one tool); every other template looks something up first and is multi_turn |
  | non-thinking training (74,013) | 85,015 at the expected pass rate, 42,079 at the pessimistic | possible only if its one environment hashes to training (87%), with one tool |
  | non-thinking state evaluation (1,800) | 1,399 at the expected pass rate (a call turn is ~50 tokens), 2,416 at the optimistic; 1,800 needs a per-attempt pass rate ≥ ~0.70 | none expected: one environment means one split, and state evaluation is 4% of the hash range |
  | verification | step gold + schema validity; nothing executed (recorded execution and task success stay unset) | final state, including side effects (G1): the strongest |
  | what a sample is | a tool-call decision (the right call and arguments at step one), never a completed task | a completed task |
  | protocol change | an episode that ends on a call; a new item version and pool identity | an executable backend; its sandbox pins numpy<2 (this venv: 2.5.1) |
  | cost | $0 code; ~$0.2–0.6 of generation inside the hybrid session | days of engineering; cheap generation |

  Option 1 is the stronger choice for this gap: it alone adds state-evaluation supply. Neither closes non-thinking
  state evaluation at the expected pass rate; with option 1 it would reach about 1,400 of 1,800 positions. The
  earlier estimate ("about the target") assumed 40–120 tokens per call turn and is corrected in the shortfall record.
- **V1's tail rule on every call turn: `tool_verification@v2`** (the maintainer's review of `2468c1aa`, item 3). The
  registered V1 says nothing but whitespace may follow the last `</tool_call>`; the shared verifier did not enforce it
  on full episodes. It now applies to every assistant turn that makes a call (`tool_episodes.text_after_last_call`);
  a final-answer turn without a call is not subject to it. The version is registered
  (`six_category_design.json :: protocol.verification.version`) and recorded with every admission. $0 audit
  ([`audit/v1_tail_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/v1_tail_audit.json)): the
  hybrid's 22 recorded teacher-native tool-call turns (11 per mode, its eval-set-1 teacher evaluation, read as a
  format statistic only) have no text after the call, so no measured change to the admission set or the cost;
  Thinking-2507 has no recorded raw tool output locally (D4A's tool turns are template re-renderings, which cannot
  show a tail), and the pilot measures the hybrid's rate. Nothing was generated or admitted under v1, and no
  historical data or measurement identity is re-verified.
- **Option 1 implemented at $0: `single_step_gold_tool_call@v1`** (the maintainer's review of `647eea0e`, item 4;
  [`six_category_design.json :: protocol.episode_shapes`](../../../../configs/stages/stage-2/teacher_modes/six_category_design.json),
  [`audit/single_step_supply_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/single_step_supply_audit.json)).
  - **What a sample is.** One assistant turn from the first real user input: the teacher's own tool call, ending the
    sample at its natural termination. No tool response or final answer exists or is required. It is a tool-call
    decision, never a completed task.
  - **Verification.** Protocol (natural termination, the mode's think rule, every block parsed, the turn ends on its
    call with nothing after it but the stop token, no response attached), schema validity, and the exact gold call
    (name + arguments after canonicalization). Recorded execution, tool selection and task success stay `None` --
    unverified, never passed (core `verify_episode(unexecuted_final_call=True)`). The gold only verifies; the
    target is the teacher's generated ids.
  - **Scope.** Pivot Conversational trajectories that the converter skipped only because their first gold call has
    no recorded output, with structured arguments: `schema_arguments` only, one turn, no backend, task id variant
    `turn1-step1`. 1,709 tasks converted; 7,034 unstructured first calls (tool_selection decisions) are counted and
    not converted. The existing 9,051 converted tasks are byte-identical, environments included. Refused for any
    other category (multi_turn, error_recovery, mcp_agent included) or shape rule.
  - **Pool (dry, schema v4).** Episode shapes are tiers of each split's frozen order: every full episode first, then
    the single steps; a T5 duplicate keeps the full episode. Every earlier task is admitted as before, in the same
    split and relative order. Single steps admitted: 1,560 training, 72 calibration, 22 state evaluation,
    19 validation, 1 reserve (35 refused by the screens).
  - **Plan.** Every thinking-mode and Thinking-2507 cell is unchanged: their full episodes fill them before the
    second tier. Non-thinking `schema_arguments`: training expects 891 tasks (bound 1,928, the whole supply), state
    evaluation all 29 (7 full + 22 single), calibration unchanged (6 expected, full episodes). Price: $2.27 / $3.72 /
    $11.01, cap $12.11 (was $3.62 / $10.31, $11.34).
  - **Counting and measurement.** Unchanged: one turn = one token-in/token-out document; unique supervised tokens
    before packing; the same packing and scored positions. An item carries `episode_shape`, which enters the
    profile's and the suite's content identity; the evaluator's protocol id is unchanged.
  - **Reach at the priors** (a prior, not a measurement). Non-thinking training: 85,126 of 74,013 supervised tokens at
    the expected pass rate (42,136 pessimistic; the target needs a per-attempt pass rate of about 0.22). Non-thinking
    state evaluation: 1,413 of 1,800 scored positions (696 pessimistic, 2,436 optimistic); 1,800 needs about 0.67. No SHORTFALL is approved, so the
    A2-6cat admission refuses and `state_eval_hybrid@v2` stays PENDING. The pilot application (section 8) generates
    every state-evaluation `schema_arguments` task to measure the real supply.
- **Rejected: relabelling Toucan.** Toucan has no gold arguments, so it would fill the category with weaker
  verification.
- **Approving the shortfall** is not granted (2026-11-04). It would also need acceptance of A2-6cat's proportion.

**Gaps that remain:**
- gold arguments for parallel calls;
- "ask for the missing value" with gold;
- a task-success verifier (Toucan verifies tool selection, Pivot step gold);
- a registered tool-calling evaluation (eval-set-2): teacher-forced KL on tool turns is not tool-call success.

## 5a. The code chain, end to end (ready, nothing run)

| step | code | paid? |
| --- | --- | --- |
| task records from a source | `tool_sources_toucan.py`, `tool_sources_pivot.py` | no |
| pinned, verified reference copies: T2 benchmarks, T3 xLAM / Glaive | `tool_benchmarks.py fetch` | no |
| the pool: atoms, splits, T1–T5, schema, replay conflicts, length bound, frozen orders (episode shapes as order tiers, schema v4), content identity | `tool_pool.py` | no |
| the bounded plan: cells, targets, frozen orders, rounds, bounds, price | `tool_generation_plan.py` | no |
| single-step gold tool calls (`single_step_gold_tool_call@v1`): Pivot Conversational trajectories whose first gold call has no recorded output, schema_arguments only; verified for protocol, schema and the exact gold call, never executed | `tool_sources_pivot.py`, `tool_episodes.py` (`EPISODE_SHAPES`), `aadistill.data.tool_calls.verify_episode(unexecuted_final_call=True)` | no |
| teacher-native episodes in rounds to each cell's target, one lineage per session under its own kind and the L40S teacher gate; the plan and the pool verified first | `tool_session.py` (hybrid `tool_generation`, Thinking-2507 `tool_generation_t2507`) over `run_battery --env-factory` (each task renders its own system text) | yes |
| verification by level; documents and items | `tool_admission.py` (from the sealed shards), `aadistill.data.tool_calls`, `tool_episodes.py` | no |
| types v2, profiles v2, `state_eval_hybrid@v2` (tool items selected by the plan, THIN categories listed), the packed freeze | `packed_inputs.py` | no |
| the packed hybrid full search | `packed_full_search.py` (kind `packed_search`) | yes |
| the A1-6cat / A2-6cat / C1-6cat data: selection (the plan's per-category tool selection), proportion validation, admission / freeze, then packing with every validation document | `packed_packs.py` (`select`, `proportions`, `admit`, `pack`; `verify_pack` for the binding) | no |
| the units bound to packs and Top-1 | `packed_training_plan.py --bind-packs` | no |
| training and formal evaluation per arm | `a1_units.py --arm` (kinds `train_6cat_a1`, `_a2`, `_c1`) | yes |

Every paid kind refuses at issue until the grant names its hard cap.

**The bounded plan** (`bounded_plan` in `tool_generation_protocol.json`;
[`plans/tool_generation_plan_dry_run.json`](../../../../logs/stages/stage-2/teacher_modes/plans/tool_generation_plan_dry_run.json)):
- **Cells.** Training targets are each arm's sixth type (493,418 supervised tokens for each hybrid mode, 492,587 for
  Thinking-2507), split over the seven categories by the design's shares. Calibration takes 4,200 and state
  evaluation 12,000 scored positions per mode, by the same shares; state evaluation also needs at least 5 tasks per
  category. Validation is as registered: 32 tasks per hybrid mode, 12,624 supervised tokens for Thinking-2507.
- **Rounds.**
  - Round 1 takes each cell's frozen-order prefix whose expected yield reaches its target.
  - Later rounds add `remaining / observed yield × 1.1`, up to 4 rounds.
  - Each round is one engine invocation per mode for first candidates, and one for the second candidates of its
    refused tasks.
  - A cell never passes the prefix whose pessimistic yield reaches its target, nor its supply. A cell still short is
    recorded as SHORTFALL.
- **Resume.** Each round's input is derived from the plan and from what earlier rounds sealed, so a replacement
  session recomputes the same rounds. `test_tool_generation_rounds.py` checks this on a scripted engine.
- **Size.** 4,698 tasks expected and 8,547 at the bound:
  - non-thinking: 3,089 training tasks (bound 6,053);
  - thinking: 683 (1,150);
  - Thinking-2507: 683 (951).
  - Before this fix the session would have generated about 238,790 first candidates.
- **Consumers select by the same rule.** Admission reads the verified pool in its frozen orders. The packs and the
  packed inputs take whole episodes per category, in frozen order, up to the plan's targets
  (`tool_generation_plan.select`). `test_tool_chain_consumers.py` runs pool → plan → admission → profiles, suite and
  freeze → the C1-6cat pack → plan binding on synthetic episodes.

## 6. The new hybrid full search

Driver: [`packed_full_search.py`](../../../../scripts/stages/stage-2/teacher_modes/packed_full_search.py), which reuses
`full_search.py`'s root verification, first-evaluation device check and Top-1 persistence.
- **Method:** the selected baseline, version 2.
- **Root:** `Qwen/Qwen3-4B@1cfa9a72`, every shard hashed.
- **Target:** C3's 596M geometry and search seed.
- **Forward context:** 8,192 tokens, seed 20261105.
- **Execution:** one packed sequence per forward, DEPTH prefix reuse, `prune_weights` false.
- **Inputs:** calibration v2 COMPLETE (2026-11-05). `state_eval_hybrid@v2` is not realizable (non-thinking
  schema_arguments 1,063 / 1,800, source exhausted) and stays PENDING; `state_eval_hybrid@v2-lc1` is FROZEN
  (measurement protocol `d09a4849`). Binding the search to `@v2-lc1` is an identity change the maintainer owes
  (section 9).
- **Run id:** `research_a_packed_hybrid_full_search`.

**Calibration v2 and `state_eval_hybrid@v2`** (the design's sections):
- v1's 13 types are reused at $0, plus `(supporting, tool_calling@v1)` in both modes.
- The supporting share is split over six sub-types: 4,200 (base) / 1,800 (reasoning-heavy) scored positions per mode
  each, and the five existing supporting cells shrink to prefixes of v1's.
- The suite adds at least 12,000 tool scored positions per mode, and `tool_call_open` / `tool_call_close` in each mode's
  critical group.
- Each gets its own items, manifest, hashes and protocol id. v1 stays frozen.

**Dry run ($0).** The chain built over v1's inputs (a structural check, never a search input):
- config hash `2102c005…`, protocol `85332c64…`, fingerprint `1dcaad69…`;
- every set's packing identity is recorded.

The historical v2 driver, pinned to baseline v1, still reproduces its run exactly: config hash `e8a43631…`, protocol
`1c830524…`.

**Checkpoint and resume.**
- The beam journal and per-state checkpoints live on the network volume.
- A replacement session of the same identities restores every journaled state.
- A deadline ends INCOMPLETE; a partial ranking is never a Top-1.
- The Top-1 is copied with its sha256 to `search_packed/top1`, and a second copy off the volume is owed.

**Cost** (one L40S, scaled from the v2 search's telemetry: 1,299 pod-minutes, $23.96; nothing packed has run on a
GPU):
- lower / expected / upper: $23.72 / $32.43 / $42.14;
- proposed hard cap: $46.35;
- backbone cost per set: 1.45–1.88× v2's (longer attention context, repeated tokens, tool items).
- The largest packed sequence holds about 7.7k scored rows. DEPTH's float32 KL is chunked, so an L40S should fit; an
  A100-80GB only after a measured OOM.
- The first state evaluation and the first DEPTH round are the measurement, and they replace this estimate before the
  cap is final.

## 7. Six base-corpus types and the three trainings

**A1-6cat and A2-6cat** (hybrid root, thinking / non-thinking mode):
- initialization: the packed search's Top-1, unknown until it completes, consumed by its state id and digests. It is
  never the v2 Top-1 `159e73be…`.
- base category: six types at 493,418 supervised tokens each. The five existing types are re-selected at $0 from A1/A2's
  verified records ([`six_category_selection.py`](../../../../scripts/stages/stage-2/teacher_modes/six_category_selection.py)).
- the re-selection is done: `FIVE_TYPES_SELECTED`, no shortfall;
  - A1-6cat: 492,038–495,452 supervised tokens per type; 5,163 tasks and 5,436,912 supervised tokens before tool
    calling;
  - A2-6cat: 493,276–493,701 per type; 42,606 tasks and 5,430,311.
- tool calling: pending.
- fluid: 2,962,514, unchanged.

**Pack repairs** (the 2026-11-03 review; `packed_packs.py`):
- **Tool quota.** The tool type enters as whole episodes, in the pool's frozen training order, up to the arm's target:
  493,418 for A1-6cat and A2-6cat, 492,587 for C1-6cat. It is the shortest prefix that reaches the target, so at most
  one episode over.
- **Share check — withdrawn 2026-11-04.** `shares` checked each of the six types at 1/6 ± 2% of the supporting total
  and fluid at 49–51% of the whole. That tolerance was the agent's (`77be6e13`), never the maintainer's, and it was
  looser than the registered one (below). The proportion check now belongs to the data admission, not the packing.
- **Validation keeps every document.** It used to be capped at 16 sequences after the shuffle:
  - A1-6cat: 77 validation documents fill 18 sequences, so the cap dropped two, among them 44% of its
    abstract-analogical and 36% of its GSM8K validation tokens;
  - A2-6cat: 201 documents fill 17 sequences, and the cap kept every token;
  - C1-6cat: 47 documents fill 12 sequences.
- **Coverage gaps, now listed in the record.** C1-6cat's validation has no fluid type. A1-6cat's has no
  `interactive_adaptation`, and it carries answerability, which the six-type set no longer trains.

**Four separate steps** (the maintainer's ruling of 2026-11-04; `packed_packs.py`). `build` only runs them in order
and stops at the first refusal:

| step | function | does | never does |
| --- | --- | --- | --- |
| data selection | `select` | the arm's documents token for token, whole examples and episodes, each with its data type, category, whole unit and source; the tool cells per the frozen plan, a short cell SHORTFALL | pack, judge, fill a cell from another |
| proportion validation | `proportions` -> core `aadistill.data.mixture.validate_proportions` | unique supervised tokens before packing per type and fluid, against `registered_quotas`: every row's target, amount, deviation, bounds and status | read a packed sequence, an exposure or a loss weight; choose a tolerance |
| admission / freeze | `admit` | admits only the selection its report validated (`report_problems` recomputes `selection_identity` and compares; it never validates again), a passing one whose tool cells all reached their targets, a failure covered only by the maintainer's recorded approval (`data_admission.approved_exceptions`, the plan's `approved_shortfalls`); writes `<pack>/data_manifest.json` (every document's id, split, data type, category, unit, supervised tokens and content sha256; counts, reasoning / answer / tool-calling tokens, proportions, `selection_sha256`, sources, lineage), sealed by its `content_sha256` | admit an unapproved SHORTFALL or another selection's report; write anything when it refuses |
| packing | `pack` | checks its input is exactly the manifest's documents and the manifest is intact, then random shuffle -> sequential packing -> truncation at `stage3_packing`; records `packed_content_sha256` of the sequences it produced before writing them; the ladder names the manifest | compute or correct a proportion, add or move data |

**The report is bound to its selection** (the maintainer's review of `647eea0e`, P1). `proportions` returns its
report with `selection_sha256` -- per split every document's id, data type, category, unit, supervised tokens and
token content (ids and supervision mask), the tool cells with their targets and approvals, the sources read, and
the quota designs (`token_matched_design.json`, `six_category_design.json`) -- and seals it (`report_sha256`).
`admit` refuses an edited report, another arm's, and another selection's: a passing report of selection A submitted
with a failing selection B is refused and nothing is frozen, and so is any change after the validation to a
document's tokens, its supervision mask, its data type or category, its split or unit, a tool cell's target, a
source or a quota design. The hash is recomputed; the proportions are not, and the packing still reads none.

**The training binding verifies the actual packed content** (P1). `packed_training_plan.bind` reads a built pack
only through `verify_pack`:
1. the manifest unchanged and admitted, the pack holding exactly its documents, the design's packing spec, the rung;
2. `packed_content_sha256` recomputed from `blocks.npz` -- token ids, supervision mask and content mask of every
   sequence, each array named with its normalized dtype and shape, and the training-sequence count -- equal to the
   one the packing recorded before writing;
3. the frozen documents themselves -- `data_documents.npz`, which the admission writes beside the manifest (every
   document's token ids, supervision mask and offsets, per split, in the manifest's row order) and whose sha256 the
   manifest carries -- checked file and document by document against the manifest, then packed again by the formal
   algorithm at the design's spec, reproducing every sequence, mask and placement exactly.

So one changed non-padding token, a changed mask bit or a replaced `blocks.npz` refuses at (2), and a forged pack
whose recorded identity was rewritten to match refuses at (3): the identity alone never makes a pack correct. The
bound lineage carries `packed_content_sha256` and the manifest's `content_sha256` beside the files' sha256. A pack
without a manifest is refused.

**Truncation is verified, not refused** (the maintainer's review of `2468c1aa`, item 2). The first version re-read
each document from its final placement in the pack, which cannot recover a document the formal packing truncated, so
it refused every overlong document -- incompatible with the frozen `truncate=True`. Re-deriving from the frozen
documents reproduces the truncation and the prefix restarts exactly, with the packing algorithm unchanged and no
document excluded. The manifest schema is now `data_manifest/v2`. Regressions: an overlong example packs truncated
(its tail counted as truncated supervised tokens, the fragments restarted) while the manifest's unique supervised
tokens still count it whole and equal the rung; the pack binds; a changed token in the truncated placement refuses
(and, with the identity re-sealed, the re-derivation refuses); changed frozen documents refuse, and so does a manifest
re-pointed at them.

**The registered tolerance** (`registered_quotas`, read from its owners: `token_matched_design.json :: scale` and the
whole-task selection rule `six_category_design.json :: base_category` keeps):
- fluid and supporting each at least their targets, and the per-mode total at most 0.5% over;
- fluid's share within 0.5 points of its target share;
- each type within one largest whole unit per selection cell of its target (code has two cells; the tool type one
  per category cell of the plan);
- C1-6cat's five kept types and its fluid part exact.
On real data every arm's five types and fluid pass (`audit/proportion_tolerance_audit.json`). Against it, the
withdrawn check had no floor (a type about 2% short, or all six short together, passed), allowed fluid ±1 point
instead of ±0.5, did not bound the total, and was wider than one unit for most types. The tool type has not been
generated, so its row and its seven cells are checked when the data are admitted.

**A2-6cat's one selected task outside A2's pack** (the maintainer's review of `647eea0e`, item 6;
[`audit/a2_recovery_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/a2_recovery_audit.json),
`packed_packs.py --arm A2-6cat --recovery-audit`). The six-type re-selection took one task that the token-matched A2
selection had not: `79c86b17e0b933ee3483`, an MBPP task (`supporting.code_algorithmic/mbpp_train`). `select` reads it
from the frozen admission records. Replayed for that key with `token_admission.load`'s own functions:
- the records: 84 non-thinking labels of the v1 store and 4 of the supplement store, every shard checked against its
  `.complete` and its store identity; the task's one record is in v1 (candidate 0);
- admitted; not flagged by the isolation recheck (which passes); unchanged by the code-stability recheck;
- one document, `79c86b17e0b933ee3483|H-N|t0`: 113 prompt tokens, 14 supervised tokens, ending on the stop id, no
  `</think>` generated, content sha256 `9e46361c…`;
- with it, A2-6cat's five types and fluid equal the selection record exactly (competition math 493,404, GSM8K 493,495,
  RAG 493,276, multi-hop 493,296, code 493,701, fluid 2,963,139). A1-6cat's pack holds every task it selected.

The full `select` path re-derives every verdict, including the sandboxed stability recheck of every program and MBPP
task; on the dev box that ran for over 40 minutes at ~11.5 GB and was stopped. The functions are the same.

**C1-6cat** (Thinking-2507):
- its own D4 Top-1 `fc28e0dd…` (a v1-context search; never called a packed-context initialization);
- its kept 3,681 documents plus Thinking-2507-native tool turns (target 492,587), the whole set repacked sequentially.

**Lineage.** Root, calibration, search, initialization, targets, online KD teacher, tokenizer, template and config are
one teacher per arm. The canonical task pool is shared; generated outputs and reference logits are not.

**The six unit configs** (`plans/packed_training_budget.json :: units`) are built by `training_plan.py`'s
construction:
- E1's recipe, the frozen objective, batch size 2, micro-batch size 1, `checkpoint.bind_identity`;
- a lineage block per unit, validated with no problems;
- no `batch.context_strategy` override, so the trainer reads the sequential pack with standard causal attention;
- the pack's files, the steps and A1/A2's initialization are placeholders that launch admission refuses until their
  owners exist.

**Estimates** (`plans/packed_training_budget.json`, both seeds per arm). Seconds per step are A1's / A2's measured
figures widened to 0.85–1.3×, since full causal attention over 8,192 tokens has not been timed.

| arm | steps per seed, lower / expected / upper | USD, both seeds |
| --- | --- | --- |
| A1-6cat | 1,940 / 2,154 / 2,367 | $8.18 / $10.98 / $17.72 |
| A2-6cat | 6,810 / 7,742 / 10,152 | $22.08 / $29.78 / $52.02 |
| C1-6cat | 1,748 / 1,797 / 1,908 | $7.56 / $9.62 / $15.44 |
| six trainings | | $37.82 / $50.38 / $85.18 |

**A Thinking-2507 packed search** is not necessary for C1-6cat as registered: its comparison is end-to-end, and
initialization is already unmatched. It is needed only if the maintainer wants the initialization method held fixed
across lineages. The cost would be about $35–50: D4's search cost $28.22, scaled by 1.45–1.88×, plus its own tool
items and a suite without Glaive.

## 8. Budget and provider state

**Position** (live 2026-11-04T08:06Z, read-only):
- $837.95 committed of the $868.00 project cap; $30.05 left (volume storage accrued since 2026-11-03's $836.73);
- 0 pods; volumes `4rhubprswo` (350 GB, US-TX-3, holds the v2 Top-1's only copy) and `am5m373kuq`.

**Phases, lower / expected / upper:**
- **Bounded tool generation: $2.27 / $3.72 / $11.01, cap $12.11** (with the single-step tier; was $3.62 / $10.31,
  cap $11.34).
  - hybrid session (both modes): $1.37 / $2.48 / $8.55;
  - Thinking-2507: $0.90 / $1.24 / $2.46.
  - The basis is the plan's task counts × expected turns × each mode's tokens per turn × second candidates.
  - Decode uses the supplement's measured throughput (1,170 / 755 tokens/s). The tool-workload efficiency (0.5–1.0),
    the prefill rate (5–20k tokens/s) and model loads (1.5–4 min) are unmeasured ranges.
  - Setup and teardown add 35 min. The upper bound adds the bound's task counts, long turns, the low pass rate, the
    slow ends and every round.
  - A pilot (about 1 h) replaces the priors before the cap is final. The earlier $4.4 / $6.6 priced only a fraction
    of what the unbounded session would have generated.
- packed search: $23.72 / $32.43 / $42.14, cap $46.35;
- six trainings: $37.82 / $50.38 / $85.18.

The Full Search Authorization (generation plus search) needs $36.15 expected and $53.15 at the upper bounds, with caps
of $58.46. That is more than the $30.05 left.

**The L40S pilot** (design accepted in principle in the review of `2468c1aa`; AUTHORIZED 2026-11-04 as option B: one
L40S, 55 driver minutes, new-cost hard cap $2.50, measurement only;
[`application.json`](../../../../logs/budget/applications/research_a_tool_pilot_2026-11-04/application.json), written
by `tool_pilot.py application --live-position`).
- **Frozen cells** ([`plans/tool_pilot_plan.json`](../../../../logs/stages/stage-2/teacher_modes/plans/tool_pilot_plan.json),
  `tool_pilot.py build`): the plan's pilot sample -- per mode, the first 24 full episodes of every training category,
  the first 100 single-step `schema_arguments` training tasks, every calibration (28 + 72) and state-evaluation
  (7 + 22) `schema_arguments` task -- 397 unique tasks, 794 mode-specific runs. Every task's sha256 is frozen, and
  the pod receives a 4.8 MB task file (by its sha256) instead of the 700 MB pool. The plan binds the pool content
  (`be4652d7…`), its converted sources, the protocol, the design (hence `tool_verification@v2`) and the plan's cells.
- **Kind `tool_pilot`** (`token_matched_phase.py`): claims teacher generation only. Issue refuses until the grant names
  its cap, the L40S check admits the pilot's generation, the pilot plan and task file load, and the pilot-only source
  admission is signed, pilot-scoped and names this plan.
- **The generation-scoped gate.** The pilot only generates, so the check's GENERATION measurement admits it, and
  only when the pilot runs the generation that was measured (`tool_pilot.l40s_generation_problems`). The check
  measured 40.23 of 44.99 GiB at peak. The gate requires:
  - the record valid and its generation measured with the 4 GiB margin;
  - the identity files at the check's session commit `f37578a7` reproducing `848f6508…`
    ([`audit/l40s_generation_scope.json`](../../../../logs/stages/stage-2/teacher_modes/audit/l40s_generation_scope.json));
  - every identity file changed since then (only `selection.json`) neither a pilot input nor named anywhere on the
    generation path;
  - the same engine: teacher, tokenizer, chat template, `LLM(...)` construction, 8,184 context, stop ids, vLLM pins,
    and per-mode switch and sampling.

  Two allowances are declared: gpu_memory_utilization 0.80 ≤ 0.87, and the `--engine-metrics` stats flag. The
  evaluation engine and initialization are never read for the pilot. Every other kind keeps the whole-identity gate,
  which still refuses. The launch adds `the_pilot_runs_on_an_l40s` (one L40S; an A100 is
  never a fallback) to the runner's prechecks (unused artifact, the grant's price, all-in within the cap, the study
  identity, the bound files, the live project position, the pilot's placement, the bound records, the bundle).
- **Session** (`tool_session.py --pilot`): verifies the plan, the task file and the signed admission again, then one
  round per mode: first candidates of every frozen task, second candidates of the refused tasks only. Sealed shards
  make a replacement session (a new run) regenerate nothing; a generation failure ends RUN_FAILED with every finished
  episode sealed.
- **Telemetry** (`telemetry.jsonl`, one line per engine invocation): wall time; nvidia-smi peak memory and
  utilization; the engine's model-load, KV-cache-size and concurrency lines and preemption lines; run_battery's
  summary (engine init seconds, output tokens/s) and, with its new `--engine-metrics`, the engine's preemption and
  KV-cache-usage metrics; per-turn prompt tokens, generated tokens, TTFT and seconds.
- **Pilot-only source admission**
  ([`audit/tool_pilot_source_admission.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_pilot_source_admission.json)):
  Toucan SFT (Apache-2.0; 135 tasks) and Pivot Conversational (CC-BY-4.0; 68 full + 194 single-step) for this
  pilot's measurement only, by exact task ids and sha256. SIGNED 2026-11-04 on the maintainer's written decision. It is not
  a training admission, builds no canonical pool and authorizes no spend.
- **Hard cap, by the runner's own arithmetic** (`tool_pilot.runner_cap`): the hard deadline is (setup 20 + transfer 5
  + the driver's minutes) x 1.25 + the 12.5-min teardown reserve + the 10-min recovery reserve; the all-in adds the
  watchdog's 1.25 min and 150 GB of container disk; the volume `am5m373kuq` (55 GB, EUR-IS-2) is retained, its
  storage already committed. The plan-priced $1.80 did not include the runner's contingency and reserves:

  | option | hard cap | driver minutes | all-in at $0.86 / $1.1067 / $1.19 per h |
  | --- | --- | --- | --- |
  | A | $1.80 | 27 (the plan's expected 26.8) | $1.30 / $1.67 / $1.79 |
  | B | $2.50 | 55 (the plan's upper 54.4) | $1.82 / $2.33 / $2.50 |

  Option A stops at its deadline with everything sealed if the workload is slower than expected; B covers the upper
  case in one session. The plan's own price is $0.97 / $1.14 / $1.64.
- **Option B recommended** by the maintainer's review of `29b62c93` ($2.50, 55 driver minutes): one session should
  finish the 794 runs, the 29 state-evaluation `schema_arguments` tasks above all. It is a ceiling, not a promise:
  the launch recomputes the all-in at the live quote and refuses above it.
- **Partial results** (`tool_pilot.report`, review of `29b62c93`): sealed shards are read whether or not their group
  completed, each verified by its sha256 marker and the launch identity; per task and candidate the outcome is
  admitted, rejected, attempted but unsealed, or unattempted (never a failure); candidate 1 is counted only where
  candidate 0 was sealed and rejected, a task once; first-candidate pass rates divide by sealed attempts, two-candidate
  rates by determined tasks. State: COMPLETE (every required attempt sealed, the session ALL_DONE), PARTIAL (some
  required attempt unsealed), INCOMPLETE (all sealed, the session's terminal evidence missing or not ALL_DONE), FAILED
  (nothing sealed). Without `tool_session.json` the run's authorization `plan_hash` gives the identity and the report
  records the missing evidence; copies of a resumed run merge, and a task sealed differently in two copies refuses.
  An unreadable measurement is UNMEASURED, never 0.
- **The admission's signature** (`tool_pilot.admission_problems`, the one check issue and the pod both apply): signed
  by the maintainer, approving exactly `tool_pilot measurement only`, a decision quote, an ISO date between the
  record's writing and today, and the plan sha256, task file sha256 and task count it signs equal to the record's;
  the record's exact scope and exclusions, the plan, the task file, the count and the sources (datasets, revisions,
  counts) equal to the files. `signed_by` alone is no signature. The agent never fills these fields.
- **Placement and storage since 2026-11-04T16Z** (maintainer, "GPU/RunPod 操作知识更新、Pilot 跨数据中心调度与恢复修正"):
  any Secure Cloud datacenter offering one L40S at or below the grant's $1.19/h, and NO network volume
  (`tool_pilot.PLACEMENT`). The pilot reads nothing a volume holds: its tasks are a local asset and the teacher comes
  from the hub at its pinned revision. `am5m373kuq` locked every draw to EUR-IS-2, which had no L40S on 2026-11-04,
  and EUR-IS-2 supports no new network volume. The session writes to the container disk under `/workspace/teacher_modes`
  (`token_matched_phase.durable`).
  - The poll pulls every sealed shard by its sha256 into `<scr>/store/tool_pilot_hybrid` (`secure_pilot`).
  - A replacement pod gets that copy through `--resume-from` (`tool_session.bring_sealed`, then `resumable` re-verifies
    each shard).
  - The launch refuses any volume (`the_pilot_placement_is_the_planned_one`); `4rhubprswo` and `am5m373kuq` are never
    attached.
  - The kind cap is cumulative: every attempt's hard cap plus the earlier tool_pilot runs' closeouts must fit $2.50
    (`kind_cap_problems`).
  - `tool_pilot placement` reads each datacenter's L40S stock (read-only) as a hint for `--data-center-ids`.
  - The runner computes the provider deadline at each create, settles an answer without a pod id from the account,
    and rotates datacenter groups (section 8 of [`aadistill-gpu-experiments`](../../../../.agents/skills/aadistill-gpu-experiments/SKILL.md)).
- **Pilot result** (`a_tool_pilot_20261104_164531`, COMPLETE; $0.4328 all-in ESTIMATED at closeout, $0.4286 POSTED by
  the provider (v1 = v2, read 2026-11-04T18:34Z: GPU $0.4216 + disk $0.0069, 23.21 billed minutes); no new network
  volume (the 150 GB container disk is inside the all-in); records in
  [`runs/a_tool_pilot_20261104_164531/evidence/`](../../../../logs/stages/stage-2/teacher_modes/runs/a_tool_pilot_20261104_164531/evidence/tool_pilot_report.json)).
  Teacher data admission under `tool_verification@v2`, never a benchmark score (METRIC_CONTRACT 2.2):

  | mode | first-candidate admission rate | admission yield, up to two candidates | supervised tokens | scored positions |
  | --- | --- | --- | --- | --- |
  | non-thinking | 99/397 = 0.249 | 105/397 = 0.265 | 14,037 | 1,835 |
  | thinking | 118/397 = 0.297 | 135/397 = 0.340 | 91,490 | 28,386 |

  - By source and shape: Toucan full episodes admit 0.43 / 0.46 (non-thinking / thinking). Pivot full episodes
    admit 0.16 / 0.19 and Pivot single-step gold calls 0.19 / 0.31.
  - Refusals (`evidence/refusal_diagnosis.json`, re-verified on the dev box): the verifier behaved as registered, and no
    harness defect was found. The refusals come from two causes:
    - recorded-response replay has no answer for an unrecorded call, so the episode ends on a call;
    - G2 gold disagreement, of which 20 / 23 single-step tasks differ only in a free-text argument's wording.
  - The formal plan projects SHORTFALL for non-thinking `schema_arguments` training (20,088 of 74,013 supervised
    tokens) and state evaluation (1,063 of 1,800 scored positions). Nothing is approved.
  - Telemetry: peak 37,965 MiB at a memory fraction of 0.80; KV cache ~197k tokens, 0.99 used in thinking mode with
    298 + 126 preemptions; decode 536-1,020 tokens/s; TTFT p50 45.6 s, p90 142 s.
- **The bounded tool generation, revised on the pilot (maintainer 2026-11-04, "the pilot accepted ... 正式 tool_generation 前的完整修订"; nothing generated, nothing launched)**:
  - **Priors measured** ([`audit/tool_generation_priors.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_generation_priors.json),
    `tool_generation_priors.py`): per lineage, mode, split, category and shape, the units per attempted task under the
    two-candidate rule ([Wilson 90% low, measured, high]), generated tokens of every candidate, the second-candidate
    rate, prompt tokens and turns; throughput and engine starts per mode. Thinking-2507: nothing measured.
  - **Reuse** ([`plans/tool_generation_import.json`](../../../../logs/stages/stage-2/teacher_modes/plans/tool_generation_import.json),
    `tool_generation_import.py`): all 794 pilot tasks (H-N 695 candidates, H-T 676), each candidate's shard, sha256,
    the pilot identity, its candidate index and verdict, imported after checking the same input (397 task sha256s), the
    lineage, the execution terms and seeds, `teacher_identity.json`, the vLLM pins and the generation code. An imported
    task is DONE (never re-sampled); the dev box's admission reads the pilot's shards (`--import-record`).
  - **Plan** ([`plans/tool_generation_plan_dry_run.json`](../../../../logs/stages/stage-2/teacher_modes/plans/tool_generation_plan_dry_run.json),
    over the candidate pool v4, content `be4652d7`): per cell, apart, the SEALED units, the WHOLE SUPPLY's projection,
    what the BOUND buys and the unmeasured share (proxy or prior), with a status. The corrected readings:

    | cell | sealed | whole supply (low / expected / high) | new tasks: round 1 / bound | status |
    | --- | --- | --- | --- | --- |
    | H-N calibration schema_arguments (630) | 772 (all 100 tasks) | measured | 0 / 0 (was 12) | REACHED |
    | H-N state_eval schema_arguments (1,800) | 1,063 (all 29 tasks) | measured | 0 / 0 | SHORTFALL_EXHAUSTED |
    | H-N training schema_arguments (74,013) | 1,292 | 11,803 / 19,968 / 33,175 (full 368 x 29.25 + single 1,560 x 5.9) | 1,804 / 1,804 | SHORTFALL_PROJECTED |
    | H-T state_eval schema_arguments (1,800) | 4,236 (all 29 tasks) | measured | 0 / 0 (was 6) | REACHED |
    | H-N training multi_turn (98,684) | 557 | 116,262 / 283,695 / 623,257 | 4,229 / 10,347 (was 561) | REACHABLE |

    Validation and the calibration / state-evaluation categories other than schema_arguments are UNMEASURED strata
    (priced and bounded by the training split's measurement of the same mode, category and shape, labelled
    ``proxy``); H-N state_eval multi_turn is AT_RISK (its low projection 1,944 < 2,400).
  - **Price** (measured tokens of every candidate, measured throughput 574 / 1,003 tokens per s, engine starts,
    session overhead; $1.09/h, the upper level at the grant's $1.19/h):

    | session | lower | expected | upper | the runner's hard cap (all-in) |
    | --- | --- | --- | --- | --- |
    | hybrid (both modes) | $1.53 | $2.54 | $6.49 (327 min) | $8.47 |
    | Thinking-2507 (UNMEASURED: protocol priors) | $0.39 | $0.66 | $2.00 | $2.77 |

  - **Launch path** (`token_matched_phase.CONTAINER_KINDS`): the formal kinds now run like the pilot -- no network
    volume, every sealed shard pulled and RECEIPTED on the dev box (`receipts.json`), `--resume-from` a receipted copy,
    the bound task file (hybrid 17,868 tasks / 206 MB; Thinking-2507 979) staged on the relay at $0
    (`stage_tasks`) and fetched by setup at hub speed, never the 700 MB pool. The generation-scoped L40S gate admits
    the hybrid's generation (the same engine as the check) and REFUSES Thinking-2507's (another teacher, template and
    revision: no L40S evidence for it). Issue also needs the canonical pool (`TGP.verified`): the sources'
    admission is the maintainer's.
  - **Reuse inventory** (nothing regenerated or re-sampled; heavy bytes outside git, identities in the records):

    | reused | where | identity |
    | --- | --- | --- |
    | the pilot's sealed episodes (794 tasks, 1,371 candidates, verdicts) | `artifacts/.../tool_pilot_sessions/a_tool_pilot_20261104_164531/` (`polled/`, the closeout archive) | `plans/tool_generation_import.json` (each shard's sha256; archive `2336efd3…`; pilot identity `4c208793…`) |
    | the measured priors and throughput | `audit/tool_generation_priors.json` | bound by the plan |
    | the candidate pool v4 (Toucan SFT `83b288fd…`, Pivot Conversational `4e60dac0…`, both tiers, decontaminated) | `artifacts/.../tool_pool_candidates/v4_be4652d7/` (verified by `tool_pool.load_pool`) | content `be4652d7…`, pool.json `b893e965…` |
    | the bound task files | `artifacts/.../tool_generation/<lineage>/tasks.jsonl` (relay at launch) | hybrid `9f7c9f2e…`, Thinking-2507 `c555322b…` |
    | the L40S check's generation measurement | `audit/l40s_generation_scope.json` | the generation-scoped gate (hybrid only) |
    | the teacher at its revision | the hub (`Qwen/Qwen3-4B@1cfa9a72`) | `teacher_identity.json` |

    The import applies only while the canonical pool's content is `be4652d7`; a rebuilt pool of other content
    imports nothing and is priced on the protocol's priors (`tool_generation_plan.build`).
  - **schema_arguments** ([`audit/schema_arguments_shortfall_v2.json`](../../../../logs/stages/stage-2/teacher_modes/audit/schema_arguments_shortfall_v2.json)):
    no adaptation defect, no compliant independent supply; the recommended design and its mixture, evaluation and cost
    impacts are there (A2-6cat's tool type 439,373 tokens, -10.95%, schema 4.5% of it; the state-evaluation category
    at 59% of its positions, SE x1.30; ~$0.11).
  - **Refusals** ([`runs/a_tool_pilot_20261104_164531/evidence/refusal_diagnosis.json`](../../../../logs/stages/stage-2/teacher_modes/runs/a_tool_pilot_20261104_164531/evidence/refusal_diagnosis.json),
    `tool_pilot_diagnosis.py`; v2 verdicts unchanged): causes apart and their overlap -- replay misses (and their
    consequence, the episode ending on the unanswered call), G2 disagreements (other tool, no call, value, free text),
    target tools not called (no call at all, or other tools), schema and the model's own protocol breaks -- with
    traceable cases and admission against the source's recorded Qwen3-235B pass rate. The 45 free-text cases were read
    one by one ([`audit/tool_pilot_free_text_review.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_pilot_free_text_review.json)):
    24 equivalent, 20 omit user-supplied facts, 1 contradicts the input; an identifier rule agreed on 39 of 45 and is
    not proposed; `tool_verification@v2` stays.
- **Storage until 2026-11-04T16Z** ([`pilot_storage_precheck_20261104.json`](../../../../logs/stages/stage-2/teacher_modes/validations/packed_round/pilot_storage_precheck_20261104.json),
  read-only, superseded by the placement above): the provider reported `am5m373kuq` at 55 GB in EUR-IS-2, `4rhubprswo` beside it, 0 pods, but no used
  bytes; the records put at least 14.3 GB there (H1-PA's and the mixed calibration's checkpoints) plus Research A v1's
  `teacher_modes` entries, not re-listed since 2026-10-30. So the pod measures it: before generating,
  `storage_problems` requires 2 GB free (the size less the bytes on the mount; the pilot writes well under 0.5 GB) and a
  pilot directory that is new or holds only the pilot's own files; the teacher snapshot goes to the container disk. The
  launch refused any volume but `am5m373kuq`.
- **The grant's pilot-only scope** (review of `56381f55`). A kind cap alone no longer issues: `pilot_grant_problems`
  requires a `tool_pilot` block in the study grant -- approving exactly `tool_pilot measurement only`, granted by the
  maintainer with a decision and a date, naming the frozen plan and task file by sha256, the source admission, the
  cap, the L40S, `am5m373kuq` and the ceiling it was written under (still the grant's). The exact amendment is drafted,
  not applied ([`grant_amendment_proposal.json`](../../../../logs/budget/applications/research_a_tool_pilot_2026-11-04/grant_amendment_proposal.json)):
  the block (decision and date left to the maintainer), `session_hard_caps_usd.tool_pilot = 2.50`, and appends to the
  scope, the amendments and `authorizes`; the ceiling ($79.5289), the per-session cap, the $868 project cap, the card
  list, `never`, the A1 kinds' caps and `phase_a1` unchanged.
- **Envelope, read-only** (2026-11-04T10:33Z): 11 sessions, no unknown cost; spent $49.613 (supplement $15.8689,
  inputs $0.5713, search $24.1526, train_a1 $9.0202) of the $79.5289 ceiling, $29.9159 left, $27.4159 after the pilot's
  cap. The A1 phase: $33.7441 sessions + $13.34 booked storage = $47.0841 of its $77.00. Research A cumulative: $62.17
  before the phase (cross-checked: $142.19 less the first ceiling $95.8883 plus the supplement = $62.1706) + $47.0841 =
  $109.2541 of $142.19, $111.7541 with the pilot's cap. The 350 GB volume's storage past its 2026-11-15 reservation is
  not booked here.
- **Resumption identity** (review of `56381f55`). Episodes are sealed and resumed under the pilot identity -- the study
  identity, the pilot plan's sha256, the protocol's sha256, `tool_verification@v2` and the sha256 of the generation
  implementation (run_battery, generation_driver, tool_session, tool_episodes, `aadistill.data.tool_calls`)
  (`tool_pilot.pilot_identity`) -- so `generation_driver.sealed` refuses another plan's, protocol's or implementation's
  shards; before resuming, `tool_session.resumable` requires every sealed episode's group input, protocol and runner
  sha256 to be this round's. The same sealing, no second system. The report derives the pilot identity from the study
  identity when `tool_session.json` is missing (`--study-identity`).
- **Owed from the maintainer:** sign the source admission (every signature field), write the pilot-only grant
  amendment, and the final authorization. The project position is re-read at issue and at launch.
- **Separate:** the packed 8,192-token forward / DEPTH / training GPU validation is its own engineering request on
  existing frozen inputs (v1 items, A1's documents); it is not combined with the pilot.
- **Formal sources, the canonical pool and Thinking-2507** (maintainer review of `a143b7ed`, 2026-11-04):
  - **Formal source admission, signed**
    ([`audit/tool_source_admission.json`](../../../../logs/stages/stage-2/teacher_modes/audit/tool_source_admission.json),
    `tool_source_admission.py`). Toucan-1.5M SFT `@0df3cf37` and Pivot Conversational `@bfc7a4dd` are admitted as
    formal task sources for pool content `be4652d7` only. Training targets and online KD come only from the student's
    own teacher; another model's output in a source record is task material or registered reference verification
    only. Pivot Function-Calling, Nemotron-SFT-Agentic-v4 and every other source are not admitted. The record
    authorizes no stage and no spend. The issue (against the pool) and the pod (against the plan and its bound task
    file) apply the same check.
  - **Canonical pool** `data/stages/stage-2/teacher_modes/tool_pool_v1` (gitignored, about 700 MB). It was rebuilt
    with the current `tool_pool.py` from the two verified converted sources. All five split files are byte-identical to
    the reviewed v4 copy and `content_sha256` is `be4652d7`. `pool.json` differs only in `written_utc` and in the
    descriptive `verification_levels` census (the v2 level names of `1c9c2226`), so its sha256 is now `4b75c1ed`.
  - **Canonical plan** (`plans/tool_generation_plan.json`): the dry run's cells, task files and prices exactly, on the
    canonical pool. Thinking-2507 has 979 tasks (`c555322b`), staged to the relay at $0.
  - **Thinking-2507's in-session L40S device admission**
    ([`t2507_device_admission.json`](../../../../configs/stages/stage-2/teacher_modes/t2507_device_admission.json),
    `tool_session.run_device_admission`). It replaces the generation-scoped gate, which measured only the hybrid's
    engine, for this kind. Before any formal invocation the session checks:
    - the teacher snapshot's files and three weight shards by sha256 against the hub's tree at `768f209d`;
    - then one engine invocation, pass 0, on the first task of each of the 8 cells;
    - the runner's recorded model, revision, chat template `3802169b`, template kwargs, sampling, stop ids, vLLM
      against `requirements-vllm.txt`, the 8,184 context (session limit, effective limit, `max_model_len`), memory
      fraction and dtype;
    - an engine that starts with a KV cache of at least 8,184 tokens, no CUDA out-of-memory, a completed generation,
      and at least 4 GiB free at the sampled peak (`l40s_check.json`).

    Its outputs are device evidence under `device_admission/`, never a formal shard. A failure ends the session
    `RUN_FAILED` before formal generation. An A100 is possible only after a measured out-of-memory. The registered
    `tokenizer_sha256` `3ec3c124` (corpus_v2) uses a convention these files do not reproduce, so each tokenizer file
    is checked instead. The hybrid keeps the generation-scoped gate.
  - **Grant:** `tool_generation_t2507` gets $2.78, the plan's $2.7742 runner all-in rounded up, cumulative over
    attempts, with its block (`t2507_grant_problems`). It covers the device admission and the formal generation only.
    The hybrid's `tool_generation` grant (at least $8.48) is deferred until the A2 design and the state-evaluation
    coverage plan are settled.
  - **Thinking-2507's formal tool generation: COMPLETE** (2026-11-04 UTC).
    - **Attempt 1** (`a_tool_generation_t2507_20261104_194048`) failed at pod setup, $0.1459: the binding step
      recomputes `kind_conditions` on the pod, and its tool-generation branch read the canonical pool, which the pod
      never holds. Repaired: the conditions read the plan and its bound task file, and the pool check moved to
      `issue_problems` at issue. The repair was then checked at $0 in a pool-less worktree of the session commit.
    - **Attempt 2** (`a_tool_generation_t2507_20261104_195411`, pod `0lbf43rtt6yn2a`, US-TX-4, $1.09/h): ALL_DONE in
      59.3 billed minutes, $1.097 all-in. The kind has spent $1.2429 of its cumulative $2.78. The pod is confirmed
      gone and the watchdog ended `pod_gone`.
    - **Device admission: PASSED**, every one of 16 checks.
      - Teacher files and shards were hashed in 7.0 s.
      - Engine ready in 157.9 s, with a KV cache of 195,009 tokens (concurrency 23.8 at 8,184).
      - Peak 37,611 of 46,068 MiB, so 8.26 GiB free against the 4 GiB wanted.
      - vLLM 0.27.1; template `3802169b`.
    - **Formal generation:** 2,277,197 output tokens over two rounds (four invocations), 900 tokens/s.
    - **Cells** (supervised tokens before packing, admitted / attempted tasks):
      - REACHED: multi_turn 113,880 (20/64), multiple_parallel 123,617 (63/116), no_tool 100,510 (151/168),
        schema_arguments 75,152 (39/136), tool_selection 106,617 (76/126), validation 16,345 (14/18).
      - SHORTFALL at the plan's bound, every bounded task attempted: error_recovery 45,631 of 49,259 (11/39) and
        mcp_agent 48,874 of 73,888 (16/86). The bounds came from protocol priors, which overestimated this teacher's
        yield. The supply is far from exhausted (2,668 and 4,542 tasks). Neither shortfall is approved.
    - **Dev-box admission** (`tool_admission.py` on the secured shards) reproduces every cell's episodes and tokens.
    - **Top-up at the measured yields** (expected values), from each cell's frozen order, new tasks only: about 49
      tasks, about $0.42, for the registered targets; about 74 tasks, about $0.46, for the common-support targets. It
      needs the plan rebuilt with these measured strata, so it is left to the maintainer's choice of composition.
    - Records: `runs/a_tool_generation_t2507_20261104_195411/evidence/generation_summary.json`; the device admission
      and session record inside the secured archive (`artifacts/.../tool_generation_sessions/`, sha256 `7cdb61f3`).
    - Provider posting (`logs/budget/provider/runpod_posting_t2507_2026-11-04T2100Z.json`): attempt 1 is posted at
      $0.1417. Attempt 2 was only partly posted at fetch time; the ledger keeps the estimates.

- **The review of `f06ccd4` (2026-11-05): cross-plan reuse, common support registered, the Thinking-2507 top-up
  and the hybrid's formal generation.**
  - **Cross-plan import** (`tool_generation_import.build_session`). A rebuilt plan's session cannot resume an earlier
    plan's shards: its round inputs differ, and every shard is bound to the session that sealed it. A COMPLETED
    formal store is imported instead: its shards, hashes and identity stay untouched; each label's `.complete` and
    round input are verified; today's verifier re-derives every candidate; the generation code, teacher, template,
    execution terms and inputs are compared at the session commit; and the imported cells must reconcile EXACTLY
    with the session's own record (including what that session's plan had itself imported). A refused task with
    both candidates spent is done. A rebuilt plan binds every import record (`plans/tool_generation_import*.json`),
    so its cells start from the imported units and walk the frozen order without the imported tasks. The dev box
    admits old and new shards side by side (`tool_admission.py --import-record ... --import-store ...`, repeatable),
    and refuses a task generated again although imported. Interrupted sessions resume under their own identity
    (`--resume-from`); a cross-plan import of one is refused.
  - **`six_category_common_support@v1` registered**
    ([`configs/.../six_category_common_support.json`](../../../../configs/stages/stage-2/teacher_modes/six_category_common_support.json)).
    The training split's tool composition: schema_arguments 11,800 of the A arms' 493,417.83 tool-type tokens (C1
    the same share of 492,587), the six other categories in their registered proportions. 11,800 is the low end of
    a projection, not a supply guarantee. The fallback was registered before generation and reads only verified
    teacher supply. The freeze comes before any search or training. Calibration, state evaluation and validation
    keep the registered shares. It is a supervised-token-budget and composition-matched comparison, not a
    compute-matched one; C1 against A1 is secondary, confounded by its initialization search protocol. The
    registered `six_category_design.json` identity is unchanged.
  - **The plan** (`plans/tool_generation_plan.json`, rebuilt):
    - the training cells at the common-support targets;
    - Thinking-2507's bounds from its own measured strata (`tool_generation_priors --add-import` / `--lineage-imports`);
    - the non-thinking training schema_arguments cell bound by its whole remaining supply (1,804 tasks);
    - the last round takes the whole bound;
    - a driver deadline (`--deadline-minutes`, the launch's driver minutes) under which a session stops at a ROUND
      boundary, so it is importable whole;
    - for a lineage whose remainder holds about one round (Thinking-2507), round 1 takes the bound, and the bound is
      at least the prefix reaching the remaining target with 95% probability at the stratum's lower Wilson
      admission yield (`reach_bound`).
  - **Device admission reuse.** Thinking-2507's admitted result
    (`runs/a_tool_generation_t2507_20261104_195411/evidence/device_admission.json`) is registered in
    `t2507_device_admission.json`. A later pod hashes the teacher snapshot first and applies the same verdict to
    its first formal invocation (`tool_session.device_confirmation`), without re-sampling the eight admission tasks.
  - **Thinking-2507 top-up: COMPLETE**, inside the cumulative $2.78. Before it, both earlier attempts' charges were
    checked against the provider: $0.1417 and $1.0943 posted, against $0.1459 and $1.097 booked, so the booked
    estimates stand.
    - **Attempt 3** (`a_tool_generation_t2507_20261105_064829`, US-NC-1, $0.3752): device confirmation 15/15 (KV cache
      195,009 tokens, 8.11 GiB free); one round at the bounds (15 + 93 new tasks). mcp_agent REACHED (101,551 /
      84,848). error_recovery 55,433 / 56,566 at its bound: 2 of 15 new tasks admitted, against a measured 28%.
    - **Attempt 4** (`a_tool_generation_t2507_20261105_072112`, EUR-IS-2, $0.3385): the plan imports attempts 2 and
      3; 18 new tasks (the 95% reach bound); 4 admitted; error_recovery REACHED (72,874).
    - **Every Thinking-2507 cell reaches its common-support target.** Dev-box admission over the three stores
      (`tool_admission/thinking_2507_cs_final`) reproduces every cell. Training supervised tokens before packing:
      error_recovery 72,874, mcp_agent 101,551, multi_turn 113,880, multiple_parallel 123,617, no_tool 100,510,
      schema_arguments 75,152, tool_selection 106,617; validation 16,345. The kind spent $1.9566 of $2.78.
  - **C1-6cat-cs frozen** (`packed_packs.py --arm C1-6cat-cs`; `audit/six_category_common_support_freeze.json`):
    - data manifest `e2b5a4e1`;
    - 3,940 examples, 4,211 turns, 5,931,091 unique supervised tokens;
    - tool calling 500,657 against 492,587 (whole-episode overshoot: schema_arguments 15,276 against 11,780, the other
      categories +0.2% to +1.5%); the five kept types and fluid exact;
    - 1,264 packed sequences, 1,896 optimizer steps (3 epochs), 10,346,968 non-padding tokens per epoch.
  - **`state_eval_hybrid@v2-lc1` registered conditionally**
    ([`configs/.../state_eval_hybrid_v2_lc1.json`](../../../../configs/stages/stage-2/teacher_modes/state_eval_hybrid_v2_lc1.json)).
    The same KL, scored positions, aggregation, critical tokens and frozen shared inputs; the cell's 1,063 of 1,800
    positions, 11 episodes and Kish 4.98 reported every time. The jackknife is a finite-sample sensitivity analysis.
    A Top-1 flagged SENSITIVE_TO_THE_LIMITED_CELL is not promoted without further independent validation; no
    candidate or result is deleted. `@v2` stays PENDING. The whole suite gets one identity, coverage and freeze check
    after the hybrid generation (`packed_inputs.py build`, which writes the lc1 suite and freezes it only when every
    other cell meets the v2 rule).

  - **Hybrid formal tool generation: COMPLETE** (cumulative $9.13; spent $3.5885).
    - **Attempt 1** (`a_tool_generation_20261105_074146`, US-NC-1, $1.09/h): ALL_DONE in 156 driver minutes, 8.05M output
      tokens. The non-thinking training schema_arguments cell attempted its whole remaining supply (1,804 tasks) and
      REACHED 17,290 of 11,800, so the registered fallback was not needed. The non-thinking state-evaluation
      multi_turn cell (AT_RISK) REACHED 33,512 of 2,400. Short at their bounds with supply left: non-thinking training
      mcp_agent, thinking training error_recovery, and calibration / state-evaluation cells whose bounds were
      training-split proxies. **Teardown:** the gate refused only because one poll's failed listing stayed in
      `listing_unknown`; all 193 shards were secured and the archive re-hashed equal. The products were re-verified
      on the dev box, and the pod was released by hand 2.1 minutes after the gate. The provider confirmed it gone and
      the watchdog ended `pod_gone` (`closeout/teardown.json`). Repair: `phase.listing_outcome` (the latest listing
      decides) and a one-retry `phase._listing`. Cost: pod life 169.85 minutes, $3.1438 booked (provider $3.1244).
    - **Attempt 2** (`a_tool_generation_20261105_105512`, $0.4447): the plan imports the pilot and attempt 1 and bounds
      each short cell at its 95% reach prefix at the measured lower yield. A split's own stratum decides from 5
      attempted tasks; a measured split that admitted none takes its whole remaining supply; a cell's minimum tasks
      count as admissions. One round per mode: every short cell REACHED.
    - **End state** (`plans/tool_generation_plan.json`, binding all six import records): 51 cells REACHED; one
      SHORTFALL_EXHAUSTED, the non-thinking state-evaluation schema_arguments (1,063 / 1,800, the registered
      `@v2-lc1` cell). Dev-box admission over the pilot and both sessions reproduces every cell in both modes.
    - **Admitted, before packing:**
      - thinking mode: training 713,551 supervised tokens (583 episodes), validation 47,557, calibration 43,480
        scored positions, state evaluation 90,348;
      - non-thinking mode: training 1,772,801 (3,281 episodes), validation 14,061, calibration 23,702, state evaluation
        49,368;
      - Thinking-2507: training 694,201 (402 episodes), validation 16,345.
  - **The common-support freeze** (`audit/six_category_common_support_freeze.json`; packs `artifacts/.../packs_6cat/<arm>`,
    each verified by `packed_packs.verify_pack`):

    | arm | manifest | examples / turns | unique supervised tokens | tool calling | packed sequences | optimizer steps | non-padding tokens / epoch |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | A1-6cat-cs (thinking) | `ee6937d2` | 5,594 / 6,230 | 5,939,049 | 502,137 | 1,437 | 2,156 | 11,766,017 |
    | A2-6cat-cs (non-thinking) | `89fa58a0` | 44,447 / 49,784 | 5,928,313 | 498,002 | 4,988 | 7,482 | 40,861,497 |
    | C1-6cat-cs (Thinking-2507) | `e2b5a4e1` | 3,940 / 4,211 | 5,931,091 | 500,657 | 1,264 | 1,896 | 10,346,968 |

    - Every arm passes the registered proportion check, and every tool category reaches its common-support target.
    - Whole-episode overshoot per tool category is +0.07% to +6.2%, except C1's schema_arguments: 15,276 against
      11,780, +29.7%.
    - A2 / A1: 3.47x the packed sequences, optimizer steps and training FLOPs for one shared student, and 0.87x the
      supervised-token exposure. Equal unique supervised tokens are not equal compute.
    - Admitted episodes beyond a target stay in the records, unused.
  - **State-evaluation suite check after generation** (`packed_inputs.py build`):
    - the calibration v2 profiles are COMPLETE (`configs/.../hybrid_profiles_v2.json`, items in
      `artifacts/.../calibration_v3`);
    - `state_eval_hybrid@v2`: thinking mode reached, non-thinking not (schema_arguments only), so NOT frozen and still
      PENDING;
    - `state_eval_hybrid@v2-lc1`: every other cell meets the v2 rule, and the registered cell is LIMITED_COVERAGE with
      exactly the registered 1,063 positions and 11 episodes. FROZEN: `audit/state_eval_hybrid_v2_lc1_freeze.json`,
      measurement protocol `d09a4849`.

## 9. Decisions owed to the maintainer

The maintainer has settled two former items: fragments are accepted, and loss normalization is kept. No Thinking-2507
packed search runs unless it is requested. The review of `f06ccd4` (2026-11-05) settled the A2 design
(`six_category_common_support@v1`, registered), the limited-coverage identity (`state_eval_hybrid@v2-lc1`, registered
conditionally), the Thinking-2507 top-up and the hybrid generation; all four are done (section 8).

0. **The registered composition restored: design R1** (section 11; application
   `logs/budget/applications/research_a_schema_restoration_2026-11-05/application.json`). One package: admit
   `schema_synth@v1` as a task source of the extension `tool_pool@v1+schema_synth@v1`, register
   `six_category_schema_restored@v1` (A1/A2/C1-6cat-sr), and two generation caps (hybrid $6.16, Thinking-2507 $2.31).
   R2 (the non-thinking cell only, cap $4.96) is priced and not recommended.
1. **Before the packed full search** (not authorized):
   - its inputs are now the calibration v2 profiles (COMPLETE) and `state_eval_hybrid@v2-lc1` (FROZEN with one
     LIMITED_COVERAGE cell); `packed_search`'s bound records still name `state_eval_hybrid_v2_freeze.json`, which
     does not exist because `@v2` is not realizable -- binding the search to `@v2-lc1` instead is the maintainer's
     decision (an identity change);
   - the project cap: the search's registered $41.41 needs a raised project cap (project remaining $27.18 on
     2026-11-05).
2. **Stage 3 training of the three -cs arms** (not authorized): the frozen packs exist; their training plan is not
   built (`packed_training_plan` binds the registered arms).
3. **Tool-calling sources.** Toucan SFT and Pivot Conversational are ADMITTED (2026-11-04). Pivot Function-Calling and
   Nemotron-SFT-Agentic-v4 are not; Glaive and xLAM are isolation references only.
4. **Isolation costs.** Each evaluation split holds only 17-20 atoms; eval-set-1's tool names are refused; a tool
   identity served by several servers is kept to one split (6,825 tasks); T3 signatures are counted, not refused.
5. **Validation coverage.** C1-6cat(-cs)'s validation has no fluid type, and A1-6cat(-cs)'s no `interactive_adaptation`.
6. **eval-set-2** for closed-loop tool success.
7. **Storage.** Volume `4rhubprswo`. The v2 Top-1's only copy stays protected until a verified copy exists off the
   volume.

## 10. The token identity audit's intermittent failure (diagnosed 2026-11-04)

`test_the_token_identity_audit_regenerates_identically` failed in two of three full experiment-suite runs at
`647eea0e` and passed alone. The differing field was `as_generated_documents.isolation_check`, a float64 measurement
on a random 2-layer probe; every token-identity field (contexts, targets, their differences) regenerated identically.

**Reproduction** (scratch probes, no test order involved). In a fresh process the audit's exact computation -- the
probe's forward over the first 8,192-token isolated sequence, then each document alone -- was run with a second,
identical packed forward after it:
- the FIRST packed forward in the process differed from the identical second one, by 1.0e-5 to 1.2e-4 over a
  contiguous block of positions (e.g. rows 1,020-1,550 of one document and all of the next); the same few values
  recurred across processes;
- the standalone forwards and every later packed forward in the same process never moved;
- on the default eight intra-op threads it happened in 14 of 90 processes over three series (1 of 20, 7 of 40, 6 of
  30, all beside one CPU-heavy job); on one thread in 0 of 30, run alternately with the last eight-thread series.

**Cause.** A cold-path, thread-partition-dependent result of the CPU kernels on the first long forward in a process
-- a test-environment effect, not token-identity drift and not an isolation failure (a real isolation failure moves
whole documents by orders of magnitude more, from their first position). It turned the audit's correct 0.0 into a
value above its own 1e-9 tolerance, so even the audit's `pass` was intermittent.

**Repair.** `token_identity_audit.isolation_check` runs the probe on one intra-op thread (restoring the caller's
count) and records `intra_op_threads: 1`; the regenerated audit differs from the committed one by that field only.
The assertion, the identity comparison and the 1e-9 tolerance are unchanged. Regression:
`test_the_isolation_check_runs_on_one_thread_and_restores_the_callers`.

## 11. The registered composition restored, and what packing really costs (2026-11-05)

The maintainer's brief "原始数据配比恢复、Packed Forward 效率审查与 Full Search 前置准备". Nothing was generated,
searched, trained or evaluated; no pod. The decision record is `logs/budget/decisions.md` 2026-11-05.

**schema_arguments supply, per teacher** (`audit/schema_arguments_supply_v3.json`, from the six import records and the
admitted documents):

| teacher mode | training tasks attempted / supply | admitted (full / single step) | supervised tokens / registered 74,013 | per admitted episode | state evaluation |
| --- | --- | --- | --- | --- | --- |
| hybrid thinking | 124 / 1,928 (1,804 untried) | 6 / 31 | 27,837 (reasoning 92%) | 1,010 / 702 | 4,236 / 1,800 positions |
| hybrid non-thinking | 1,928 / 1,928 (exhausted) | 55 / 248 | 17,290 (tool call 62%) | 147 / 37 | 1,063 / 1,800 (29 / 29 tried) |
| Thinking-2507 | 136 / 1,928 (full episodes only) | 39 / 0 | 75,152 of 73,888 (reasoning 92%) | 1,927 | none in its plan |

Admission per attempted task (two candidates): thinking 25% / 31%, non-thinking 15% / 16%, Thinking-2507 29%. Every
task is Pivot Conversational; the gold calls carry about two arguments, 21-29% of them nested. The thinking teachers'
token targets fill on ~40 episodes because their reasoning is long, which is not evidence of better tool calling.

**New source.** No public dataset closes the gap legally and robustly (27 candidates surveyed beyond reviews v1/v2;
Synth-APIGen is conditional and not proposed). `schema_synth@v1` (`scripts/.../schema_synth.py`) writes tasks whose
gold call a program computes from a sampled ground truth: relative dates and times from the stated current date, unit
and count conversions, totals and scaled quantities, rules stated in the tool description, enum meanings, corrections
and distractors, nested objects and ordered arrays (4.3 top-level arguments per call, 96% nested). Every assistant turn
is still the teacher's own. `audit/schema_synth_v1_audit.json`: all 3,750 tasks admit their gold episode in both modes
and refuse a changed leaf, a dropped required field, an unrequested optional field and a reordered array; the bytes
are deterministic; no tool name or signature is shared with eval-set-1, the benchmarks or the excluded lineage
sources. Whole domains are assigned to splits (22 training, 3 state evaluation, 2 calibration, 2 validation, 1
reserve).

**Pool and reuse.** `tool_pool.extend` writes tool_pool@v1 plus the addition (content `6139e647...`, base
`be4652d7...` byte for byte, every frozen order a prefix). `tool_generation_plan.import_records` reuses a record by
task identity, so the six sealed sessions stay DONE and no task is attempted twice. Strata are per source
(`tool_generation_priors.stratum_key`): a new source is priced from Pivot's measured strata only under the label
`proxy:other_source`. `source_groups` splits a category into per-source cells for generation and selection.

**Design R1** (`configs/.../six_category_schema_restored.json`, DESIGN): the registered shares; inside schema_arguments
every arm holds Pivot 17,200 (C1 17,171) and schema_synth the rest; state evaluation Pivot 1,000 + schema_synth 800 per
hybrid mode. Price (`plans/schema_restoration_plan_dry.json`): $0.66 / $1.12 / $6.29, caps hybrid $6.16 and
Thinking-2507 $2.31. R2 fills only the non-thinking cell ($0.37 / $0.64 / $3.72, cap $4.96) but makes 77% of A2's
schema tokens synthetic against 0% of A1's.

**What packing computes** (`audit/packed_forward_compute.json`). The v2 search read each item alone (micro-batch 1, no
padding). For the same frozen v2 items:

| set | forwards alone -> packed | computed positions | attention sum of T^2 | scored rows | FLOPs per pass (root) |
| --- | --- | --- | --- | --- | --- |
| base profile | 558 -> 79 | +13.2% (75,223 repeated) | x4.63 | +9.5% | 4.64 -> 6.42 PF |
| reasoning-heavy profile | 388 -> 59 | +14.0% (58,394) | x4.44 | +11.1% | 3.46 -> 4.80 PF |
| state evaluation | 1,345 -> 216 | +17.3% (260,683) | x4.05 | +12.2% | 12.31 -> 17.45 PF |

Attention rises from 8-9% to 25% of the root's backbone FLOPs (21-24% to 52% for the 596M target). A DEPTH expansion
does 1.27-1.68x the FLOPs (by parent geometry) in 7x fewer forwards. Re-priced from v2's measured seconds: $34.12
expected ($26.46-$39.09), against $25.58 for the same inputs read alone. Packing is the protocol (P19), not a speed-up:
it wins only if 8,192-token sequences run about a third more FLOPs per second, which no GPU has measured. The first
state evaluation and first DEPTH round of the next authorized search are that measurement; the evaluator now writes
`evaluation_cache.jsonl`. Safe, unimplemented optimizations: skipping the last DEPTH candidate's whole-model rerun
(about 1.1-1.4 h packed; needs the prefix-reuse CUDA bit-identity check) and `logits_to_keep` in the statistics passes
(a 2.49 GB transient per 8,192-token forward). Storage: the search needs about 350 GB; `4rhubprswo` is not to be
mounted and `am5m373kuq` holds 55 GB, so a new volume belongs to the search's budget.

## 12. R1 approved: the registered composition's generation (2026-11-05)

The maintainer's R1 approval. The decision record is `logs/budget/decisions.md` 2026-11-05 ("R1 approved").

**The source, final.** Training instances 160 -> 220 per domain (5,070 instances, every earlier one byte for byte).
An independent semantic review (`audit/schema_synth_v1_semantic_review.json`) re-derived every gold from the request
and the stated date alone (5,070 / 5,070 equal, no leakage) and found 226 requests with a second reasonable call; they
are left out by `schema_synth.ambiguity` (its set is exactly the reviewer's). The final audit
(`audit/schema_synth_v1_audit.json`, 4,844 tasks): 0 problems, every mutation refused, the renderer unchanged by the
reference in both modes. What the levels mean: schema validity is structural, step_gold an exact match with the
program's call, recorded execution a replay hit keyed by that call -- one piece of correctness evidence, G2, never G1.
The extension `e771393a...` (4,843 schema_synth tasks admitted) is signed in `audit/tool_source_admission_schema_synth.json`.

**The generation.** Protocol `tool_generation_protocol_sr.json` (the common-support engine terms unchanged), plan
`plans/schema_restoration_plan.json` (six import records reused by task identity; schema_synth cells priced at the
labelled `proxy:other_source` stratum; $0.66 / $0.79 / $5.08), kinds `tool_generation_sr` (cumulative $6.16) and
`tool_generation_sr_t2507` ($2.31): round 1 the expected prefix, later rounds from the session's own measurement.

## Reproduce ($0)

```
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packing_audit --write logs/stages/stage-2/teacher_modes/audit/packing_audit.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_search_forecast --write logs/stages/stage-2/teacher_modes/plans/packed_search_budget.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_training_plan --write logs/stages/stage-2/teacher_modes/plans/packed_training_budget.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_packs --proportion-audit logs/stages/stage-2/teacher_modes/audit/proportion_tolerance_audit.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_arguments_audit --pivot <Pivot Conversational train.jsonl> \
    --workbench <WorkBench@49c7dfd0 task files> --write logs/stages/stage-2/teacher_modes/audit/schema_arguments_options.json
PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume .venv/bin/python -m stages.teacher_modes.packed_full_search \
    --check-only --inputs v1_dry_run --device cuda --durable-root <scratch> --status <scratch>/status --deadline-minutes 1
# the single-step tier (review of 647eea0e)
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_sources_pivot --which conversational \
    --jsonl <Pivot Conversational train.jsonl> --out <scratch>/pivot_conversational_v2.jsonl
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pool --source toucan_sft=<scratch>/toucan_tasks_v2.jsonl \
    --source pivot_conversational=<scratch>/pivot_conversational_v2.jsonl --out <scratch>/tool_pool_dry_v4
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_generation_plan --pool <scratch>/tool_pool_dry_v4 \
    --write logs/stages/stage-2/teacher_modes/plans/tool_generation_plan_dry_run.json
# the pilot (review of 2468c1aa)
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot build --pool <scratch>/tool_pool_dry_v4
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot admission
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot storage-precheck \
    --write logs/stages/stage-2/teacher_modes/validations/packed_round/pilot_storage_precheck_20261104.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot grant-amendment
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_pilot application --live-position --tests <evidence>
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_admission --v1-tail-audit \
    logs/stages/stage-2/teacher_modes/audit/v1_tail_audit.json --generations hybrid_thinking=<H-T eval generations> \
    --generations hybrid_non_thinking=<H-N eval generations>
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_arguments_audit --supply-pool <scratch>/tool_pool_dry_v4 \
    --before-pool <scratch>/tool_pool_dry_v3 --write logs/stages/stage-2/teacher_modes/audit/single_step_supply_audit.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_packs --arm A2-6cat \
    --recovery-audit logs/stages/stage-2/teacher_modes/audit/a2_recovery_audit.json
PYTHONPATH=src:scripts .venv/bin/python scripts/stages/stage-2/teacher_modes/token_identity_audit.py \
    --write logs/stages/stage-2/teacher_modes/audit/token_identity_audit.json
# the registered composition restored (section 11)
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_supply_audit \
    --write logs/stages/stage-2/teacher_modes/audit/schema_arguments_supply_v3.json
HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_synth \
    --out artifacts/stages/stage-2/teacher_modes/schema_synth_v1 --extend data/stages/stage-2/teacher_modes/tool_pool_v1 \
    --pool-out artifacts/stages/stage-2/teacher_modes/tool_pool_v1_schema_synth
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_synth_audit \
    --tasks artifacts/stages/stage-2/teacher_modes/schema_synth_v1/tasks.jsonl \
    --write logs/stages/stage-2/teacher_modes/audit/schema_synth_v1_audit.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_generation_plan \
    --pool artifacts/stages/stage-2/teacher_modes/tool_pool_v1_schema_synth \
    --composition configs/stages/stage-2/teacher_modes/six_category_schema_restored.json --no-task-files \
    --write logs/stages/stage-2/teacher_modes/plans/schema_restoration_plan_dry.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_forward_compute \
    --write logs/stages/stage-2/teacher_modes/audit/packed_forward_compute.json
# R1 approved (section 12): the final source, its extension, admission and plan
HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_synth \
    --out artifacts/stages/stage-2/teacher_modes/schema_synth_v1 --extend data/stages/stage-2/teacher_modes/tool_pool_v1 \
    --pool-out data/stages/stage-2/teacher_modes/tool_pool_v1_schema_synth
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.schema_synth_audit \
    --tasks artifacts/stages/stage-2/teacher_modes/schema_synth_v1/tasks.jsonl \
    --write logs/stages/stage-2/teacher_modes/audit/schema_synth_v1_audit.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_source_admission \
    --pool data/stages/stage-2/teacher_modes/tool_pool_v1_schema_synth \
    --audit logs/stages/stage-2/teacher_modes/audit/schema_synth_v1_audit.json \
    --review logs/stages/stage-2/teacher_modes/audit/schema_synth_v1_semantic_review.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.tool_generation_plan \
    --pool data/stages/stage-2/teacher_modes/tool_pool_v1_schema_synth \
    --composition configs/stages/stage-2/teacher_modes/six_category_schema_restored.json \
    --protocol configs/stages/stage-2/teacher_modes/tool_generation_protocol_sr.json \
    --write logs/stages/stage-2/teacher_modes/plans/schema_restoration_plan.json
```

The plan and the application record the scratch pool as `<scratch>/tool_pool_dry_v4` (content `be4652d7…`).
