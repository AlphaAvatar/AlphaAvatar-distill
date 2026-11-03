# Research A — packed round: Stage-1 forward and Stage-3 training packing, preparation for the new search and trainings

**Status: `PREPARATION_BLOCKED` (2026-11-03).** The maintainer's brief "Stage-1 Forward / Stage-3 Training Packing
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
  - A2-6cat's tool type would come in about 6% short, and the share check refuses that pack.
  - The non-thinking category of `state_eval_hybrid@v2` stays short, so the suite cannot freeze. The packed search
    needs the suite, so the search cannot start either.
  - Thinking mode's 7 state-evaluation tasks may also leave it THIN.
- **Minimal verifiable fix: call-only gold episodes for this category.**
  - Source: Pivot Conversational trajectories whose first gold step is one structured call, but whose output was
    never recorded. Estimated supply: about 1,560 training, 72 calibration and 22 state-evaluation tasks.
  - Verification: step gold on the call alone, plus schema validity.
  - Cost: it needs one approval, of an episode that ends on a call.
- **Stronger option: WorkBench instances.** They are verified against the sandbox's final state (G1), at the cost of
  building an executable backend.
- **Rejected: relabelling Toucan.** Toucan has no gold arguments, so it would fill the category with weaker
  verification.
- **Approving the shortfall** would also need acceptance of A2-6cat's proportion.

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
| the pool: atoms, splits, T1–T5, schema, replay conflicts, length bound, frozen orders, content identity | `tool_pool.py` | no |
| the bounded plan: cells, targets, frozen orders, rounds, bounds, price | `tool_generation_plan.py` | no |
| teacher-native episodes in rounds to each cell's target, one lineage per session under its own kind and the L40S teacher gate; the plan and the pool verified first | `tool_session.py` (hybrid `tool_generation`, Thinking-2507 `tool_generation_t2507`) over `run_battery --env-factory` (each task renders its own system text) | yes |
| verification by level; documents and items | `tool_admission.py` (from the sealed shards), `aadistill.data.tool_calls`, `tool_episodes.py` | no |
| types v2, profiles v2, `state_eval_hybrid@v2` (tool items selected by the plan, THIN categories listed), the packed freeze | `packed_inputs.py` | no |
| the packed hybrid full search | `packed_full_search.py` (kind `packed_search`) | yes |
| the packs of A1-6cat / A2-6cat / C1-6cat: the plan's per-category tool selection, the share check before packing, every validation document | `packed_packs.py` | no |
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
- **Inputs:** calibration v2 and `state_eval_hybrid@v2`, both `PENDING_TOOL_ITEMS`.
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
- **Share check.** `shares` checks each of the six types at 1/6 ± 2% of the supporting total, and fluid at 49–51% of
  the whole. It counts unique supervised tokens before packing and refuses a pack outside the check. The five existing
  types sit within 0.7%.
- **Validation keeps every document.** It used to be capped at 16 sequences after the shuffle:
  - A1-6cat: 77 validation documents fill 18 sequences, so the cap dropped two, among them 44% of its
    abstract-analogical and 36% of its GSM8K validation tokens;
  - A2-6cat: 201 documents fill 17 sequences, and the cap kept every token;
  - C1-6cat: 47 documents fill 12 sequences.
- **Coverage gaps, now listed in the record.** C1-6cat's validation has no fluid type. A1-6cat's has no
  `interactive_adaptation`, and it carries answerability, which the six-type set no longer trains.

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

**Position** (live 2026-11-03T09:06Z):
- $836.73 committed of the $868.00 project cap; $31.27 left;
- 0 pods; volumes `4rhubprswo` (350 GB, US-TX-3, holds the v2 Top-1's only copy) and `am5m373kuq`.

**Phases, lower / expected / upper:**
- **Bounded tool generation: $2.27 / $3.62 / $10.31, cap $11.34.**
  - hybrid session (both modes): $1.37 / $2.38 / $7.85;
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

The Full Search Authorization (generation plus search) needs $36.05 expected and $52.45 at the upper bounds, with caps
of $57.69. That is more than the $31.27 left.

## 9. Decisions owed to the maintainer

The maintainer has settled two former items: fragments are accepted, and loss normalization is kept. No Thinking-2507
packed search runs unless it is requested.

1. **Tool-calling sources.**
   - Admit Toucan and Pivot Conversational, the priority candidates.
   - Pivot Function-Calling fails T3; Nemotron-SFT-Agentic-v4 stays deferred.
   - Also owed: the category shares (used unchanged by the plan), the prompt bound and the non-thinking state-evaluation
     floor.
2. **Expected SHORTFALLS.**
   - `schema_arguments` in non-thinking mode, in training and in state evaluation. It refuses the A2-6cat pack and
     blocks the suite's freeze. Choose one: call-only gold episodes (recommended, minimal), WorkBench, or an approved
     shortfall together with A2-6cat's proportion.
   - "Ask for the missing value" in every mode.
3. **Isolation costs.**
   - Each evaluation split holds only 17–20 atoms.
   - eval-set-1's tool names are refused.
   - A tool identity served by several servers is kept to one split (6,825 tasks).
   - T3 signatures are counted, not refused.
4. **Validation coverage.** C1-6cat's validation has no fluid type, and A1-6cat's no `interactive_adaptation`. Both
   gaps are recorded.
5. **eval-set-2** for closed-loop tool success.
6. **Budget.** A raised project cap for the Full Search Authorization.
7. **Storage.** Volume `4rhubprswo`. The v2 Top-1's only copy stays protected until a verified copy exists off the
   volume.

## Reproduce ($0)

```
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packing_audit --write logs/stages/stage-2/teacher_modes/audit/packing_audit.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_search_forecast --write logs/stages/stage-2/teacher_modes/plans/packed_search_budget.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_training_plan --write logs/stages/stage-2/teacher_modes/plans/packed_training_budget.json
PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume .venv/bin/python -m stages.teacher_modes.packed_full_search \
    --check-only --inputs v1_dry_run --device cuda --durable-root <scratch> --status <scratch>/status --deadline-minutes 1
```
