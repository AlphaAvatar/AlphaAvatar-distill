# Research A — packed round: Stage-1 forward and Stage-3 training packing, preparation for the new search and trainings

**Status: `PREPARATION_BLOCKED` (2026-11-03).** The maintainer's brief "Stage-1 Forward / Stage-3 Training Packing
重构及 Research A 前置准备" authorized code, data protocol, $0 data preparation, tests and cost estimates only.
- Code, identities, tests, the $0 data audits, the new full-search chain and the training plans are done.
- The one real blocker is the tool-calling data. Its sources are candidates under the maintainer's review (section 5),
  and its teacher-native generation is a paid step that needs its own authorization. The code for every step from
  source rows to trained arms is in place (section 5a).
- The new full search cannot start without tool items in its calibration and state evaluation.
- Nothing was searched, trained, generated, downloaded or evaluated. No pod was created and `main` is unchanged.

| record | owns |
| --- | --- |
| [`packed_round_design.json`](../../../../configs/stages/stage-2/teacher_modes/packed_round_design.json) | this round's registered terms: forward context and training packing (sequence length, seeds), calibration v2, `state_eval_hybrid@v2`, the new search, the three arms and their lineage, budget, decisions owed |
| [`initialization_forward_baseline_v2.json`](../../../../configs/shared/baselines/initialization_forward_baseline_v2.json) | the official Stage-1 method, version 2 (D1's method in the packed forward context), selected in `selection.json`; v1 superseded and still loadable |
| [`audit/packing_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/packing_audit.json) | the packing applied to existing data: Stage-1 sets; A1, A2 and C1's kept part (`packing_audit.py`) |
| [`plans/packed_search_budget.json`](../../../../logs/stages/stage-2/teacher_modes/plans/packed_search_budget.json) | the new search's cost estimate (`packed_search_forecast.py`) |
| [`plans/packed_training_budget.json`](../../../../logs/stages/stage-2/teacher_modes/plans/packed_training_budget.json) | steps, schedule and cost of A1-6cat, A2-6cat, C1-6cat (`packed_training_plan.py`) |
| [`validations/packed_round/dry_run_20261103.json`](../../../../logs/stages/stage-2/teacher_modes/validations/packed_round/dry_run_20261103.json) | the $0 dry run of the new search chain, and the historical search's identities reproduced |
| [`data/six_category_selection.json`](../../../../logs/stages/stage-2/teacher_modes/data/six_category_selection.json) | the five existing base types re-selected at 1/6 each (`six_category_selection.py`; see section 7) |

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

The pool, protocol and verifier are [`SIX_CATEGORY.md`](SIX_CATEGORY.md)'s, reused unchanged:
- `tool_pool@v1`, seven categories, splits by tool family;
- isolation T1–T5 and the 6,144-token prompt bound;
- the five-check verifier and whole-episode admission;
- only native assistant ids are supervised.

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

**Isolation screen.** No sampled row of any candidate shares an exact prompt key or a tool family with eval-set-1's
1,419 held-out prompts and 12 tool families. The 13-gram near-duplicate screen and benchmark decontamination (BFCL,
API-Bank, tau-bench, ToolSandbox, When2Call test) are still to run on the full sources.

**Data prepared ($0, dry).** Toucan's SFT config was converted (`tool_sources_toucan.py`) and assembled
(`tool_pool.assemble`) into scratch, not under `data/`:
- 91,044 tasks after T5 collapsed 28,243 duplicates, in 5,401 tool families;
- every category in every split: about 81,900 training, 3,200 state-evaluation, 2,500 calibration and 2,500
  validation tasks;
- every calling task has its replay table;
- the 6,144-token bound would exclude about 1%.

The Pivot pair and v4 are not converted yet.

**Gaps that remain:**
- gold arguments for parallel calls;
- "ask for the missing parameter" with gold;
- an MCP task-success check beyond recorded agreement;
- a registered tool-calling evaluation (eval-set-2): teacher-forced KL on tool turns is not tool-call success.

## 5a. The code chain, end to end (ready, nothing run)

| step | code | paid? |
| --- | --- | --- |
| task records from a source | `tool_sources_toucan.py` (others to write) | no |
| the pool: families, splits, T1/T2/T5, length bound, frozen order | `tool_pool.py` | no |
| teacher-native episodes, one lineage per session | `tool_session.py` (kinds `tool_generation`, `tool_generation_t2507`) over `run_battery --env-factory` | yes |
| the five checks; documents and items | `tool_admission.py` (from the sealed shards), `aadistill.data.tool_calls`, `tool_episodes.py` | no |
| types v2, profiles v2, `state_eval_hybrid@v2`, the packed freeze | `packed_inputs.py` | no |
| the packed hybrid full search | `packed_full_search.py` (kind `packed_search`) | yes |
| the packs of A1-6cat / A2-6cat / C1-6cat | `packed_packs.py` | no |
| the units bound to packs and Top-1 | `packed_training_plan.py --bind-packs` | no |
| training and formal evaluation per arm | `a1_units.py --arm` (kinds `train_6cat_a1`, `_a2`, `_c1`) | yes |

Every paid kind refuses at issue until the grant names its hard cap.

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

**Phases, expected (upper):**
- pool and generation: $4.4 ($6.6);
- packed search: $32.43 ($42.14);
- six trainings: $50.38 ($85.18).

The Full Search Authorization (pool generation plus search) needs about $37, with a cap around $53, which is more than
is left.

## 9. Decisions owed to the maintainer

1. **Fragments.** `truncate=True` replaces padding with a repeated prefix. The sequences are identical to
   `truncate=False` on this data. The fragments re-score 10–13% of Stage-1 scored positions and add 18.7% (A1), 4.3%
   (A2) and 23.3% (C1's kept part) of supervised tokens per epoch. Confirm, or choose fragments as context only, which
   is numerically `truncate=False` at every scored position.
2. **Tool calling.** Which candidates enter the pool (recommended: Toucan plus Pivot, with v4 conditional on its three
   blockers), the category shares, the prompt bound, the non-thinking state-evaluation floor, and an eval-set-2.
3. **Budget.** A raised project cap for the Full Search Authorization.
4. **Thinking-2507.** Whether a Thinking-2507 packed search is wanted.
5. **Storage.** Volume `4rhubprswo` and the v2 Top-1's single copy.
6. **Loss weighting.** Keep the step-level weighting, or register a fixed-denominator objective.

## Reproduce ($0)

```
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packing_audit --write logs/stages/stage-2/teacher_modes/audit/packing_audit.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_search_forecast --write logs/stages/stage-2/teacher_modes/plans/packed_search_budget.json
PYTHONPATH=src:scripts .venv/bin/python -m stages.teacher_modes.packed_training_plan --write logs/stages/stage-2/teacher_modes/plans/packed_training_budget.json
PYTHONPATH=src:scripts:scripts/stages/stage-1:scripts/stages/stage-0/calibration_volume .venv/bin/python -m stages.teacher_modes.packed_full_search \
    --check-only --inputs v1_dry_run --device cuda --durable-root <scratch> --status <scratch>/status --deadline-minutes 1
```
