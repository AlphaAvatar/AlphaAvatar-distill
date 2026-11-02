# Research A v2 — token-matched data, the hybrid D4 full search (proposed), the 3 × 2 training matrix (proposed)

**Status: the A1 phase is executing (maintainer 2026-11-01, "Execute A1 First"; section D). Before it, the $0 work
after the maintainer's independent review (2026-11-01).**
- Nothing searches, initializes, trains or evaluates a student, and no inputs session runs. Research B stays
  pending.
- The review accepted A in principle, accepted B on conditions and confirmed C's matrix. It asked for:
  - the MBPP false pass removed and every other code verdict audited (A);
  - the full-search data's per type and mode audit, its completion conditions and the mode-aware critical-token
    aggregation (B);
  - the lineage and tokenizer gaps closed and C1's tool-calling sources checked (C);
  - the budget re-estimated (Budget).

Decided by the maintainer on 2026-10-31, in four briefs and one answer in session (logs/budget/decisions.md,
2026-10-31):
1. Research A v1 stops. Its six units are historical evidence and are never read as v2 results.
2. The scale is unique supervised tokens: per mode 5,923,021 = 2,960,507 + 2,962,514.
3. The base-corpus category corresponds to E1's corpus_v2 type by type.
4. Teacher–student lineage is a hard invariant.
5. The final matrix is A1, A2 and C1, each on seeds 20260726 and 20260801. These six runs are the only ones.

| record | owns |
| --- | --- |
| [`token_matched_design.json`](../../../../configs/stages/stage-2/teacher_modes/token_matched_design.json) | scale, categories, cells, selection, supplement rule, lineage, matrix |
| [`plans/token_matched_preregistration.json`](../../../../logs/stages/stage-2/teacher_modes/plans/token_matched_preregistration.json) | arms, comparisons, the bound rule (v2: the 3 × 2 matrix) |
| [`audit/supply_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/supply_audit.json) | v1's sealed records, token by token |
| [`audit/calibration_coverage.json`](../../../../logs/stages/stage-2/teacher_modes/audit/calibration_coverage.json) | the hybrid calibration against D4's 4× profiles |
| [`audit/c1_identity_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/c1_identity_audit.json) | C1's pack, targets, initialization, protocol differences |
| [`data/token_matched_admission.json`](../../../../logs/stages/stage-2/teacher_modes/data/token_matched_admission.json) | A1's and A2's selections, packs, accounting |
| [`full_search_proposal.json`](../../../../configs/stages/stage-2/teacher_modes/full_search_proposal.json) | B |
| [`plans/training_plan_proposal.json`](../../../../logs/stages/stage-2/teacher_modes/plans/training_plan_proposal.json) | C: every unit's config, lineage block and accounting |
| [`audit/code_stability.json`](../../../../logs/stages/stage-2/teacher_modes/audit/code_stability.json) | every code verdict under hash seeds 0–7 |
| [`audit/full_search_data_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/full_search_data_audit.json) | B's data per type and mode; the training sources' axes |
| [`audit/c1_tool_calling_sources.json`](../../../../logs/stages/stage-2/teacher_modes/audit/c1_tool_calling_sources.json) | C1's tool-calling sources and their terms |
| [`audit/provider_resources_20261101.json`](../../../../logs/stages/stage-2/teacher_modes/audit/provider_resources_20261101.json) | L40S price and stock, volume price, balance |
| [`plans/budget_v2.json`](../../../../logs/stages/stage-2/teacher_modes/plans/budget_v2.json) | the budget: lower, expected, deadline |

## A. Data (admitted 2026-11-01; re-admitted the same day after the review removed one MBPP example)

Owner: [`data/token_matched_admission.json`](../../../../logs/stages/stage-2/teacher_modes/data/token_matched_admission.json),
terminal `ADMITTED` (design `2ba937fb…`, plan `a48ce65a…`, order_v2 `02c98b81…`). Built on the dev box at $0 from
v1's sealed records (identity `848f6508…`) and the supplement's (`0d39ad93…`), through v1's frozen verdict and the
code-stability recheck. The maintainer accepted the data phase in principle (2026-11-01).

**Both arms meet the target** (per mode 5,923,021; total over target ≤ 0.5%, category share within 0.5 points).

| | A1 (thinking) | A2 (non-thinking) |
| --- | ---: | ---: |
| examples | 5,957 | 49,830 |
| turns | 6,105 | 53,136 |
| packed sequences (8,192 tokens) | 1,102 | 4,577 |
| unique supervised tokens | 5,930,273 (+0.122%) | 5,923,646 (+0.011%) |
| base-corpus / fluid share | 49.923% / 50.077% (0.060 points) | 49.978% / 50.022% (0.005 points) |
| reasoning / answer tokens | 4,983,377 / 946,896 | 0 / 5,923,646 |
| non-padding tokens (prompt/context) | 8,924,771 (2,994,498) | 37,330,273 (31,406,627) |
| padding share | 1.14% | 0.44% |
| optimizer steps (3 epochs, batch size 2) | 1,653 | 6,866 |
| supervised-token / non-padding-token exposure | 17,790,819 / 26,774,313 | 17,770,938 / 111,990,819 |
| training FLOPs | 5.14e17 | 2.14e18 |
| supervised tokens reused from v1 / new | 3,293,941 / 2,636,332 | 2,774,685 / 3,148,961 |
| validation | 16 sequences, 77 turns (pool 217) | 16 sequences, 201 turns (pool 201) |

**Token-matched is not compute-matched.** Non-thinking turns pair long prompts with short answers (5.3 context
tokens per supervised token, against 0.5 in thinking mode), so at equal supervised tokens A2 carries 4.2× A1's
non-padding tokens and 4.15× its packed sequences, optimizer steps and training FLOPs.

**By cell** (supervised tokens). Quota = the category total × the cell's share: the base-corpus category's five types
one fifth each (code = MBPP capped at the smaller mode's supply + reasoning_core), the fluid cells D4's increment
shares. Whole examples; each category stops at the first example reaching its total; inside it, the cell furthest
behind its share goes next, and inside a cell, the family furthest behind the two modes' mean family shares. The
tolerance is on the categories, so a cell may differ by its last examples (A1's interactive cell +3.5%: thinking-mode
episodes average about 1,260 supervised tokens).

| cell | quota | A1 | A2 | family TVD between modes |
| --- | ---: | ---: | ---: | ---: |
| competition math (MATH train) | 592,101 | 591,618 | 592,022 | 0.019 |
| GSM8K train | 592,101 | 592,205 | 591,639 | 0.000 |
| SQuAD v2 answerable | 592,101 | 591,549 | 591,618 | 0.000 |
| HotpotQA | 592,101 | 591,757 | 591,606 | 0.000 |
| code: MBPP | 38,260 | 38,512 | 38,120 | 0.000 |
| code: reasoning_core | 553,841 | 554,937 | 555,502 | 0.147 |
| abstract_analogical / project generators | 365,838 | 366,150 | 365,686 | 0.153 |
| compositional_generalization / reasoning_core | 119,546 | 118,760 | 119,661 | 0.000 |
| compositional_generalization / reasoning_gym | 100,877 | 99,330 | 100,625 | 0.085 |
| compositional_generalization / project generators | 416,322 | 415,768 | 416,045 | 0.233 |
| inductive_rule_discovery / project generators | 729,632 | 725,310 | 729,532 | 0.015 |
| interactive_adaptation / project generators | 497,314 | 514,586 | 497,683 | 0.183 |
| task_contained_deduction / reasoning_core | 171,405 | 170,571 | 171,399 | 0.000 |
| task_contained_deduction / reasoning_gym | 109,160 | 108,857 | 110,414 | 0.136 |
| task_contained_deduction / project generators | 452,420 | 450,363 | 452,094 | 0.032 |

Family TVD is between the two modes' admitted family shares inside a cell; where a family's yield differs by mode
(non-thinking answers more of some generator families wrong), the shares differ and are reported, not equalized.

**Refusals and isolation.**
- Isolation recheck against eval-set-1 (R1, R2, R9): 193 (example, mode) pairs of R9's refused families removed
  (string_induction@v2 `str.caesar` 64, `str.duplicate_each` 53, `str.double_last` 76); the remaining 70,580 pass.
- Refused in design cells (every candidate failed): thinking: answer 1,412, ending 1,610 (training-length limits,
  repetition stops), isolation 102, protocol 1; non-thinking: answer 19,244, ending 2,675, isolation 91, action
  legality 24. Sources outside the design (not selected): thinking 887, non-thinking 3,597 examples.
- Calibration units (v1's and the supplement's round 0) are never training data and are held out of both packs.
- Code-stability recheck: 1 (non-thinking MBPP, below).

**The MBPP false pass removed** (maintainer review 2026-11-01).
- The candidate: an MBPP candidate in A2 (`d934be3a…`, 171 supervised tokens) unpacks `set(s)` into two names.
  - Its asserts pass at string-hash seeds 0, 1, 5 and 6 and fail at 2, 3, 4 and 7.
  - Pinning seed 0 (`code_exec` v4) made the verdict reproducible, not correct.
- The admission now runs a code-stability recheck (`code_stability.recheck`, before the isolation recheck), with no
  other verdict loosened:
  - an MBPP candidate must pass under hash seeds 0–7 and at seed 0 twice; otherwise it is refused and the task's
    next candidate is considered;
  - a reasoning_core program task's gold must be its program's value under every one of those seeds; otherwise the
    task is refused in both modes.
- Effect: `d934be3a…` has no second non-thinking candidate, so it leaves A2.
- The registered rule then applies (code = MBPP capped at the smaller mode's admitted supply, the rest of the code
  type to reasoning_core):
  - the cap falls from 38,431 to 38,260, and the reasoning_core code quota rises by 171 to 553,841;
  - A2's reasoning_core code cell already held 555,502 tokens, above that quota, so the selection gave the
    category's last tokens to the cells furthest below their shares: SQuAD v2 +72, HotpotQA +75 (5 examples);
  - A2's code type holds 593,622 supervised tokens against its 592,101 quota;
  - A1's selection and pack are unchanged (byte-identical files).
- The superseded packs are kept outside git (`packs_tm_superseded_20261101/`), not deleted.

**Every other code verdict is stable** (`audit/code_stability.json`, the probed sandbox, seeds 0–7 plus seed 0
again):
- 862 MBPP candidates of both record sets: 663 pass at every seed, 198 fail at every seed, and 1 is unstable
  (the one above).
- 1,029 reasoning_core `controlled_code_execution` tasks of the training supply: the gold is stable for every one.
- 31 program tasks of the full search's inputs (calibration's code cell, the state evaluation's code type): the gold
  is stable for every one.
- 1,028 `dynamic_programming` tasks state their numbers in the prompt and break ties lexicographically, so they hash
  no Python value and are not executed.
- Regression tests: `teacher_modes/tests/test_code_stability.py`, and `run_asserts_across_seeds` in core.

**Generation (the supplement).**
- `a_supplement_20261031_092108`: round 0 (calibration: 801 units non-thinking, 12 thinking) and thinking round 1
  (7,232 examples, 18,977,409 generated tokens) sealed. It failed at non-thinking round 1 before the engine loaded:
  the runner's `splitlines()` split a GSM8K problem at a raw U+2028 (fixed in 7d31676f). $5.7873.
- `a_supplement_20261031_144618`: replayed the sealed rounds without regenerating; non-thinking round 1 (65,813
  examples, 23,171,660 generated tokens, 8.8 h with second candidates) and round 2 (HotpotQA's last 3,332 examples,
  154,022 tokens). `ALL_DONE`, $10.0816; pod deleted, provider confirms only am5m373kuq.
- The non-thinking HotpotQA queue ended at 1.03× need on the pod (below its 1.05 margin); the admission reaches the
  cell's share of the category.

**Manifest and storage.**
- A1 pack `artifacts/stages/stage-2/teacher_modes/packs_tm/A1`: `blocks.npz` `32a87943…` (91.6 MB), `ladder.json`
  `dccfe323…`, `audit.jsonl` `7d71e9e1…`.
- A2 pack, rebuilt: `blocks.npz` `a583763d…` (376.3 MB), `ladder.json` `506e42e9…`, `audit.jsonl` `e75c1ddd…`
  (superseded: `cd9e1a23…`, `d4d53cc8…`, `d1f0f077…`).
- Both `token_documents.document_isolated` (each turn attends only to itself), sequence length 8,192, teacher
  `Qwen/Qwen3-4B@1cfa9a72` in its own mode. Packs and stores stay outside git (`artifacts/`, `.scratch/`); the
  supplement's sealed shards are also on am5m373kuq.

**Actual cost.** The supplement's two sessions: $15.8689 ($5.7873 + $10.0816), inside its original $23.00
(`a_supplement_20261031_091618` created no resource). Study A-TM: $15.8689 of $95.8883 spent, $80.0194 left.
Project: $786.3228 of $868 committed, $81.6772 left. The provider account's balance is $79.23 (2026-11-01 10:21 UTC),
(The account's $80 spend limit is per hour, not cumulative.)

## B. The hybrid D4 full search (proposed, not run)

Owners:
- [`full_search_proposal.json`](../../../../configs/stages/stage-2/teacher_modes/full_search_proposal.json);
- [`full_search_types.json`](../../../../configs/stages/stage-2/teacher_modes/full_search_types.json);
- [`audit/full_search_data_audit.json`](../../../../logs/stages/stage-2/teacher_modes/audit/full_search_data_audit.json)
  (per type and mode: tasks, scored positions, sources, shortfalls; the training sources' axes);
- [`audit/full_search_isolation.json`](../../../../logs/stages/stage-2/teacher_modes/audit/full_search_isolation.json);
- [`hybrid_d4_profiles.json`](../../../../configs/stages/stage-2/teacher_modes/hybrid_d4_profiles.json).

**The maintainer's review (2026-11-01): conditionally accepted.**
- Accepted: the 13 types over 5 domains, each in both modes, in calibration and state evaluation alike; interactive
  adaptation excluded from both; reasoning_core and reasoning_gym kept as training sources only; no sixth domain;
  D4's domain weights and search arms unchanged.
- Static teacher-forced KL is never read as interactive ability. Interactive ability is measured closed-loop on
  eval-set-1 after training.
- Owed before any paid step:
  - the ten short calibration cells closed, code included;
  - the state evaluation's 1,722 prompts generated natively in both modes, with a verified manifest;
  - mode-aware critical-token aggregation (done, below);
  - the micro-batch-1 fingerprint (recorded) and its path validated on the L40S;
  - a volume that holds the whole search.
- None of the inputs session, the search or the trainings is authorized.

**Root teacher.**
- `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`.
- Weight shards `328a91d3…`, `6cd087b3…`, `e4bf4369…`.
- Chat template `a55ee1b1…`, RoPE base 1,000,000, 40,960 positions.
- H-T means `enable_thinking=True` and H-N means `enable_thinking=False`, each with its official sampling.

**Method: D4's, resolved from the selected forward baseline.**
- Baseline: `stage1_init.d1_target_aware.scored_v1@v1`.
- Operators: ATTENTION `activation_importance_v1`, DEPTH `causal_kl_greedy_v1`, RESIDUAL_WIDTH `global_pca_v0`, FFN
  `activation_importance_v0`. Operator order and each operator's profile are searched.
- Beam: `beam.delayed_prune@v1`, width 6, warmup 1.
- Ranking: `pareto_multi_objective@v2`.
- Positions, support and rows: `positions.supervised_target_v1@v1`, `reference_topk_tail_v1` with K = 200, `scored_v1`.
- Execution: `length_sorted_v1`, DEPTH prefix reuse.
- Geometry: 596M (28 / 1024 / 16 Q / 8 KV / 3072, tied embeddings).
- Seed 20260815; Top-1 is the first of 12 leaves in the Pareto order; one search.

**One type table for both data sets** (`full_search_types.json`). There are 13 types over five domains, each in both
modes, and the item sub-type is `<type>|<mode>`:

| domain (ability axis) | types | A1/A2 training cells they stand for |
| --- | --- | --- |
| abstract_analogical | grid_transform@v1 | abstract_analogical / project generators |
| compositional_generalization | defined_ops@v1, scan_language@v1 | compositional / project generators |
| inductive_rule_discovery | list_induction@v1, string_induction@v1 | inductive / project generators |
| task_contained_deduction | entity_tracking@v1, knights_knaves@v1, ordering@v1 | deduction / project generators |
| supporting (E1's base-corpus counterpart) | openmath, gsm8k, rag_evidence, multihop_qa, code | competition math, GSM8K, SQuAD, HotpotQA, code |

**How the search data's sources differ from the training data's** (`full_search_data_audit.json`).
- Fluid category:
  - The types stand for 66.3% of A1/A2's fluid quota: the project generators on all four static axes.
  - Not typed, 33.7%: interactive adaptation (16.8%), and reasoning_core / reasoning_gym on the compositional (7.4%)
    and deduction (9.5%) axes.
  - Those axes are represented in the search by their project generators only.
- Base-corpus category:
  - Typed 98.7%. MBPP (1.3%) is exhausted by training, so the code type is reasoning_core's code modules.
  - openmath is the competition-math type from another source (OpenMathInstruct-2 units of base_pool_v1, against
    MATH train).
  - gsm8k, rag_evidence and multihop_qa use the training datasets themselves, on rows past every training and
    calibration row.
- The fluid types use the training's own @v1 generators. Their families are calibration-split and their seeds are
  in the calibration range, so the kinds of rule are shared and the instances, families and seeds are not.
- Weights differ by design:
  - training: D4's increment shares by supervised tokens;
  - calibration: D4's domain weights (Base 0.18 × 4 + 0.28, RH 0.22 × 4 + 0.12), split equally over a domain's
    types and halved per mode;
  - state evaluation: 12,000 scored positions per (type, mode), reduced equal sub-type then equal domain.

**Per type and mode** (calibration as built; state evaluation planned, not generated). "Tasks" are independent
tasks. Thin means fewer than 5 tasks in a profile's cell; short means below its quota.

| type, mode | Base: scored positions / quota, tasks | RH: scored positions / quota, tasks | tasks in either | calibration sources | state evaluation: queue prompts / expected tasks needed, sources |
| --- | ---: | ---: | ---: | --- | --- |
| grid_transform|H-T | 17,211 / 16,200, 6 | 21,405 / 19,800, 7 | 7 | 3 rule families | 21 / 4, 5 rule families |
| grid_transform|H-N | 16,322 / 16,200, 20 | 20,646 / 19,800, 25 | 25 | 5 rule families | 21 / 15, 5 rule families |
| defined_ops|H-T | 8,273 / 8,100, 7 | 10,512 / 9,900, 9 | 9 | 2 rule families | 53 / 11, 2 rule families |
| defined_ops|H-N | 8,144 / 8,100, 24 | 10,052 / 9,900, 30 | 30 | 2 rule families | 53 / 38, 2 rule families |
| scan_language|H-T | 10,930 / 8,100, 3 (thin) | 10,930 / 9,900, 3 (thin) | 3 | 2 rule families | 39 / 3, 3 rule families |
| scan_language|H-N | 8,409 / 8,100, 19 | 10,469 / 9,900, 22 | 22 | 3 rule families | 39 / 28, 3 rule families |
| list_induction|H-T | 9,249 / 8,100, 5 | 10,851 / 9,900, 6 | 6 | 5 rule families | 23 / 6, 11 rule families |
| list_induction|H-N | 8,306 / 8,100, 10 | 10,221 / 9,900, 12 | 12 | 8 rule families | 23 / 17, 11 rule families |
| string_induction|H-T | 4,294 / 8,100, 2 (thin) **short** | 4,294 / 9,900, 2 (thin) **short** | 2 | 1 rule families | 31 / 7, 4 rule families |
| string_induction|H-N | 4,578 / 8,100, 8 **short** | 4,578 / 9,900, 8 **short** | 8 | 4 rule families | 31 / 22, 4 rule families |
| entity_tracking|H-T | 6,214 / 5,400, 5 | 7,505 / 6,600, 6 | 6 | 2 rule families | 53 / 10, 2 rule families |
| entity_tracking|H-N | 5,473 / 5,400, 19 | 6,798 / 6,600, 23 | 23 | 2 rule families | 53 / 40, 2 rule families |
| knights_knaves|H-T | 6,257 / 5,400, 2 (thin) | 10,724 / 6,600, 3 (thin) | 3 | 1 rule families | 21 / 4, 1 rule families |
| knights_knaves|H-N | 6,130 / 5,400, 6 | 6,905 / 6,600, 7 | 7 | 1 rule families | 21 / 16, 1 rule families |
| ordering|H-T | 5,982 / 5,400, 4 (thin) | 7,809 / 6,600, 5 | 5 | 1 rule families | 70 / 9, 1 rule families |
| ordering|H-N | 5,459 / 5,400, 21 | 6,747 / 6,600, 25 | 25 | 1 rule families | 70 / 50, 1 rule families |
| openmath|H-T | 5,306 / 5,040, 4 (thin) | 2,560 / 2,160, 2 (thin) | 4 | openmath | 43 / 5, corpus_v2_openmath |
| openmath|H-N | 5,967 / 5,040, 11 | 2,194 / 2,160, 6 | 11 | openmath | 43 / 31, corpus_v2_openmath |
| gsm8k|H-T | 8,182 / 5,040, 6 | 2,385 / 2,160, 2 (thin) | 6 | gsm8k | 73 / 9, gsm8k_train |
| gsm8k|H-N | 5,150 / 5,040, 22 | 2,175 / 2,160, 9 | 22 | gsm8k | 73 / 52, gsm8k_train |
| rag_evidence|H-T | 5,633 / 5,040, 15 | 2,384 / 2,160, 12 | 15 | squad_v2 | 590 / 38, squad_v2_train_answerable |
| rag_evidence|H-N | 4,984 / 5,040, 187 **short** | 2,136 / 2,160, 79 **short** | 187 | squad_v2 | 590 / 422, squad_v2_train_answerable |
| multihop_qa|H-T | 5,117 / 5,040, 8 | 4,056 / 2,160, 5 | 8 | hotpot | 683 / 19, hotpot_qa_train |
| multihop_qa|H-N | 5,077 / 5,040, 129 | 2,163 / 2,160, 53 | 129 | hotpot | 683 / 488, hotpot_qa_train |
| code|H-T | 0 / 5,040, 0 (thin) **short** | 0 / 2,160, 0 (thin) **short** | 0 | none yet (9 reasoning_core code prompts) | 22 / 11, reasoning_core/controlled_code_execution |
| code|H-N | 0 / 5,040, 0 (thin) **short** | 0 / 2,160, 0 (thin) **short** | 0 | none yet (9 reasoning_core code prompts) | 22 / 16, reasoning_core/controlled_code_execution |

- **Ten short calibration cells**, the same five cells in each profile:
  - string_induction H-T and H-N: R9's refused families hold out most of D4's pool;
  - code H-T and H-N: no records yet;
  - rag_evidence H-N: 56 and 24 positions short.
- **Thin cells.** Long thinking traces fill a positions quota with few tasks: scan_language, string_induction,
  knights_knaves, ordering and openmath (H-T), and gsm8k H-T in RH. Under D4's design the quota is in positions;
  setting a minimum number of tasks per cell would change that design and is the maintainer's choice.
- **State evaluation.** Every queue holds at least 1.36× the tasks expected for 12,000 positions at the planning
  yield. The planning yield is measured for the eleven non-code types, and taken from training records for code.

**Completion conditions** (`full_search_proposal.json :: completion_conditions`). Nothing below is authorized.
- Calibration:
  - 0 short cells in both profiles, each filled with the mode's own naturally terminated trajectories under that
    mode's template and sampling;
  - the top-up is the 14 refill units in D4's cell order plus the code cell's 9 prompts; a cell still short continues
    its own queue, never another type's or mode's;
  - every task is disjoint from A1/A2/C1 training, the state evaluation and eval-set-1 (R1, R2, R3, R9), with the
    isolation proof re-run on the final items;
  - the code prompts' gold is stable under every hash seed;
  - every cell's task count is reported beside its positions;
  - the profiles are rebuilt and their hashes recorded.
- State evaluation:
  - all 26 (type, mode) cells reach 12,000 scored positions from whole, naturally terminated tasks in queue order;
  - a reserve queue is built at $0 before the session, so a short cell continues inside it; a cell still short is a
    SHORTFALL that stops before the search;
  - every (type, mode) has at least one scored position of each of its critical classes;
  - the manifest binds the root teacher, the tokenizer files and vocabulary, the template, both modes, the type
    table, every item's ids, masks and critical group, the critical groups, the isolation proof and the content
    sha256;
  - `hybrid_state_eval.load` accepts it;
  - the state evaluation is independent of calibration, all training and eval-set-1, re-checked on the final items.

**Scored and critical positions.**
- Scored positions are the positions that predict the hybrid's own generated tokens, `<|im_end|>` included. No
  prompt, template or preclosed think block is scored.
- Critical tokens follow each mode's native output: H-T has `think_close`, `final_answer`, `eos`; H-N has
  `final_answer`, `eos`, never a `</think>` it did not generate.
- **Mode-aware aggregation, implemented at $0** (`aadistill.initialization.planning.metrics`, `group_balanced/v1`):
  - The suite declares `critical_groups` {H-T: three classes, H-N: two} and each item its mode as `critical_group`.
  - `state.critical_token_kl` = 0.5 × H-T's mean over its classes + 0.5 × H-N's. A flat mean over the five
    (mode, class) pairs would have weighted them 60:40.
  - The objective's `critical_token` follows the same rule. A class with no scored position in its group is refused.
  - The aggregation is recorded in the reduction semantics, so the suite's measurement protocol id is new.
  - `state_eval@v1` keeps its values and ids.
  - `hybrid_state_eval.load` refuses a suite without one group per mode.
  - Tests: `tests/initialization/test_critical_token_groups.py`.

**Calibration** (`calib.hybrid_d4_base@v1`, `calib.hybrid_d4_rh@v1`).
- D4's weights: 0.18 / 0.28 and 0.22 / 0.12.
- D4's volume of 180,000 scored positions per profile, over the 13 types.
- Rule: per cell, D4's order, each mode filling half the cell's quota with its own naturally terminated
  trajectories.
- Reused: v1's sealed hybrid calibration (D4's 4× prefix) and the supplement's 801 units.
- Held out, the order continuing past them:
  - 33 pool units of R9-refused families (`string_induction@v1` `str.caesar`, `str.double_last`, `str.duplicate_each`;
    D4's own profiles held 4 of them each);
  - 5,034 pool units whose prompt is a training prompt of A1/A2 or of C1.
- Coverage now (scored positions): Base H-T 92,648 / H-N 83,999, RH H-T 95,415 / H-N 85,084.

**State evaluation `state_eval_hybrid@v1` (new; `state_eval@v1` is unchanged and unused).**
- 1,722 prompts, all new.
- Fluid prompts are new instances of the same @v1 generators: calibration-split families, seeds 2,500,000–2,570,030
  in the calibration range.
- Dataset prompts are train rows past everything order_v2 holds, rendered as corpus_v2 rendered them.
- openmath comes from base_pool_v1's unused units; code from reasoning_core at seed base 2026320000.
- Quota: frozen at 12,000 scored positions per (type, mode), 312,000 in all.
- Scoring: D4's three objectives under equal sub-type (mode included) then equal domain, so the modes are 50:50 and
  no profile gains from raw tokens. Per-mode KLs are recorded beside the ranking values.
- Loader: `hybrid_state_eval.load` refuses any other binding and then uses the shared loader; phase_d1's frozen
  loader is not edited.

**Isolation** (`full_search_isolation.json`).
- State evaluation against calibration and against all training of both lineages: no shared prompt key, example id
  (`task_id`) or content. 4 and 208 13-gram near-copies were removed.
- Calibration against A1/A2's training: 0.
- Calibration against C1's training: 232 shared corpus_v2 prompts, all held out of the profiles.
- State evaluation against eval-set-1: R1, R2, R3, R4, R7 and R9 by family name pass.
- Rule families and generators are shared between calibration and state evaluation by design; instances, seeds and
  rows are not.

**Execution.**
- One L40S at micro-batch 1 for the whole search. Micro-batch 3 measured out of memory in DEPTH on an L40S with this
  root; micro-batch 1 completed DEPTH (2,371 s, 26.88 GiB peak reserved). No A100.
- **Numerical execution fingerprint** at micro-batch 1: `24787ea9c00dbbba2006530f70397076998b95915199f03270e4130c6298c6ef`.
  - The same call at micro-batch 3 reproduces the forward baseline's `e627f461…`, so the two differ by the
    micro-batch alone.
  - Semantic state ids are shared with a micro-batch-3 run. Materialization ids and bytes are this search's own.
- Validation owed on the L40S, inside the search session and before level 1:
  - every operator on both profiles (level 0 expands all eight);
  - each child's state evaluation under the new protocol id;
  - the Pareto v2 ranking of level 0;
  - the epsilon repeatability (1e-4) re-measured at micro-batch 1 under the frozen review gate.
  The session stops on any failure, or when level 0 overruns its share of the deadline.
- **Storage: a network volume, no `prune_weights`.**
  - It holds every state, the journal and the training save points, and no candidate evidence is deleted before the
    evidence audit.
  - D4's tree stored 274 GB, and the six trainings add about 50 GB. 350 GB holds both; 300 GB does only if non-Top-1
    states are retired after the audit, which is a maintainer decision.
  - Price: $0.07 per GB-month ($0.70/day for 300 GB, $0.82/day for 350 GB).
  - Datacenter: EU-NL-1 or US-TX-3, which have storage and L40S quotes. L40S stock reads "Low" everywhere.
    `am5m373kuq` (55 GB, EUR-IS-2, no storage support now) stays the supplement's archive.
  - Evidence: `audit/provider_resources_20261101.json`.

## C. The 3 × 2 training matrix (proposed, not run)

| arm | initialization | online KD teacher | data |
| --- | --- | --- | --- |
| A1 | the hybrid D4 full-search Top-1 (to be searched) | `Qwen/Qwen3-4B@1cfa9a72`, thinking | A1's pack (hybrid thinking, about 5.923M) |
| A2 | the same Top-1 | `Qwen/Qwen3-4B@1cfa9a72`, non-thinking | A2's pack (hybrid non-thinking, about 5.923M) |
| C1 | Thinking-2507 D4 Top-1, model sha256 `fc28e0dd…` | `Qwen/Qwen3-4B-Thinking-2507@768f209d` | D4-Augmented's pack (5,923,021), hash-verified |

**Common recipe: E1's / D4-Augmented's, with the registered objective.**
- Batch size 2 sequences, micro-batch size 1, 2 gradient accumulation steps, sequence length 8,192, 3 epochs.
- AdamW: lr 5e-5, betas (0.9, 0.95), eps 1e-8, weight decay 0.01, gradient clipping 1.0.
- Schedule: warmup max(10, round(0.05 T)), cosine to 0.1 of peak, where T = ceil(3 × sequences / 2).
- Evaluation every max(25, T // 8), save every max(200, T // 2), keep the last 1.
- fp32 weights with bf16 autocast and gradient checkpointing. E1's trainable patterns: attention, MLP and norms; the
  embedding is frozen.
- Objective: CE 0.25 + KD 1.0, T 1, KD and CE only at the assistant target positions (`kd_scope` `assistant`),
  `reference_topk_tail_v1` with K = 200.
- 16 validation sequences. Checkpoints keep optimizer, step and RNG on the network volume, and a run resumes from its
  latest save point.

**Per arm.**

| | A1 | A2 | C1 |
| --- | --- | --- | --- |
| packing / context strategy | token-native turns, each turn attends only to itself (`document_isolated`) | the same | D4-Augmented's ladder: cross-context within system-prompt groups (kept, disclosed) |
| packed sequences, optimizer steps | 1,102 sequences, 1,653 steps (warmup 83, save every 826) | 4,577 sequences, 6,866 steps (warmup 343, save every 3,433) | 2,468 sequences, 3,702 steps (warmup 185, eval every 462, save every 1,851) |
| supervised tokens / exposure | 5,930,273 / 17,790,819 | 5,923,646 / 17,770,938 | 5,923,021 / 17,769,063 |
| non-padding tokens | 8,924,771 (1.1% padding) | 37,330,273 (0.4% padding) | 7,882,298 (61.0% padding) |
| reasoning / answer tokens | 4,983,377 / 946,896 | 0 / 5,923,646 (preclosed think) | E1 part 2,142,358 / 818,149; fluid part 2,677,056 / 285,458 |
| tokenizer (source, `tokenizer_sha256` `7781771a…`) | the hybrid's pinned files (`tokenizers/qwen3_4b_1cfa9a72`) | the same | the PCA init directory's (Thinking-2507's files) |
| evaluation | eval-set-1 v2 (frozen), nucleus_v1, thinking mode | the same, non-thinking mode | the same, Thinking-2507's template |

**Lineage gate** (`aadistill.training.lineage`; every unit's `lineage` block in `training_plan_proposal.json`). One
teacher identity per run: model id, revision, weight shards, vocabulary, chat template, RoPE base, position limit.
It is checked three times, and fails closed every time:
- at config validation (the KD teacher against the declared lineage);
- at launch admission (`training_plan.admit`);
- by the trainer before a step runs (`train_stage3.py`), and again on the model it loaded.

**Repaired at $0 after the maintainer's review (2026-11-01).** The review found four gaps; each is closed with tests
(`tests/training/test_lineage.py`, `teacher_modes/tests/test_training_plan.py`):
1. **`tokenizer_sha256` was missing** from every unit's config, and the trainer's tokenizer contract refuses that.
   Each config now carries the vocabulary hash `7781771a…`, the same for both teachers.
2. **A1/A2's `tokenizer_source` was a description**, not a directory.
   - It is now `artifacts/stages/stage-2/teacher_modes/tokenizers/qwen3_4b_1cfa9a72`: the hybrid's `tokenizer.json`,
     `tokenizer_config.json`, `vocab.json` and `merges.txt` at its pinned revision.
   - `training_plan.py --stage-tokenizers` stages them from the snapshot and refuses any file whose sha256 differs.
   - A searched Top-1 carries no tokenizer, and none is assumed.
   - C1 keeps the PCA init directory E1 and D4-Augmented trained with (Thinking-2507's files and template).
   - The lineage binds each tokenizer by its files. The two vocabularies hash alike, so only the files (the template)
     tell the tokenizers apart.
3. **A field the student config left out was skipped.**
   - A missing RoPE base, position limit or vocabulary size is now a refusal.
   - The gate reads the student's EFFECTIVE config: what the loader builds, and the loaded model's own config after
     loading. A config.json without its RoPE base loads at the library default (10,000 for Qwen3), not the root's.
   - C1's D4 Top-1 states `rope_parameters.rope_theta` 5,000,000, and its effective config and Thinking-2507's
     snapshot both give 5,000,000.
4. **A resume skipped the weight identity.**
   - A run's first start now writes a lineage stamp into every checkpoint's bound identity: the lineage's digest and
     the initialization bytes verified at that start.
   - A resume must carry a stamp for this lineage, the stamp's initialization must be the lineage's, and staged
     initialization bytes are re-hashed. A checkpoint without a stamp, or of another lineage, is refused.

Beyond those four, the initialization is now bound by its **search record**, not by the model id a config declares
for it:
- `initialization.provenance` names the committed `search_record.json` by sha256;
- the gate reads that record's root teacher id, its shard-set hash (sha256 over `shard:sha256` lines, which
  reproduces D4's recorded `6de20cc7…`) and the selected checkpoint's sha256, and compares them with the lineage.

Refused, with tests:
- a Thinking-2507-initialized student distilled by the hybrid, or the reverse;
- data of another teacher or mode, or another arm's pack;
- a missing field, the wrong teacher, the wrong RoPE base or the wrong tokenizer;
- a resumed checkpoint of another lineage.

**Evidence that the six units pass** (`training_plan_proposal.json :: units[].launch_admission`, run on the dev box).
- C1-s20260726 and C1-s20260801: no problems and nothing pending. The dev box checked:
  - the D4 Top-1's bytes (`fc28e0dd…`) and its effective config;
  - the D4 search record's provenance;
  - Thinking-2507's cached snapshot (shards, RoPE, template);
  - the pack's three files;
  - the tokenizer through the trainer's own contract.
- A1 and A2 (four units): no problems in the KD teacher, tokenizer files and vocabulary, or pack. Two items pending:
  - the initialization (the full search's Top-1, its record and digest);
  - the hybrid's weight shards, which are not cached on the dev box; the trainer checks them on the pod.
  `admit` refuses A1/A2 until the search's record exists.

Equal vocabularies (both `7781771a…`) and an equal geometry do not make two teachers one lineage: the RoPE base, the
template and the weights tell them apart.

**Disclosed: v1's units trained with Thinking-2507's tokenizer files.**
- Research A v1's recovery units (stopped, historical) named the PCA directory as `tokenizer_source` with the hybrid
  as teacher.
- The vocabularies are identical and the packs pre-tokenized, so the token ids trained on are the same.
- The files were not the hybrid's, though, and v1's results are not reinterpreted.

**C1's differences, disclosed.** C1 against A1/A2, and C1 against the historical D4-Augmented:
- **C1 against A1/A2.** These are separate routes: a different teacher and initialization, and cross-context ladder
  packing against per-turn isolation. C1's E1 part carries 352 traces cut at 8,192 (trained without `<|im_end|>`),
  hygiene-only targets (openmath 35.8% wrong; code and tool calling unverifiable) and 492,587 tool-calling tokens.
  The fluid part supervised the template's prefilled `<think>` and 1,102 re-tokenized positions.
- **C1 against the historical D4-Augmented.** C1 adds KD on the target positions only with Top-200 + tail, where
  D4-Augmented used every non-padding position and the full vocabulary. Teacher, initialization, data, packing and
  recipe are otherwise unchanged.

**What each comparison can say.**
- **A1 vs A2.** The same teacher, initialization, unique supervised tokens and category and cell composition; the
  compute is not equal.
- **C1 vs D4-Augmented.** The KD objective's effect, with every other difference listed.
- **A1 vs C1.** End to end, the hybrid route against the Thinking-2507 route. It is not a single-variable contrast.

Token-matched is neither step-matched nor FLOPs-matched.

**Cost** (`plans/budget_v2.json`, re-estimated 2026-11-01 from measured L40S steps).
- v1's hybrid units measured 6.41–6.71 s per optimizer step (median), worst 6.94 s; per-turn isolation is already
  included.
- C1 is cross-context and is priced on D4-Augmented's measured 3.6 s per step on the same pack, not 6.7 s.
- A2 carries about 2,600 KD positions per step, and the L40S check measured 5.66 s at that kind of load.
- Per seed (expected / deadline):
  - A1: 4.28 h, $4.76 / 6.35 h, $7.05;
  - A2: 12.14 h, $13.48 / 17.91 h, $19.89;
  - C1: 5.25 h, $5.83 / 8.54 h, $9.48.
- Six runs with their sessions: $49.44 expected, $74.78 at the per-unit deadlines (the bounded hard cap).
- The deadlines are the worst measured step × 1.15 (1.25 for C1's unmeasured KD support), with evaluation allowed
  to twice its measured time.

**C1's tool-calling sources** ([`audit/c1_tool_calling_sources.json`](../../../../logs/stages/stage-2/teacher_modes/audit/c1_tool_calling_sources.json)).
C1's pack (D4-Augmented's `D4A_merged_whole`, files matching their registered sha256) carries 492,587 tool-calling
supervised tokens in its E1 part: 8.3% of C1's 5,923,021, and 16.6% of the E1 part. Each one was generated by
Thinking-2507 (Apache-2.0). The upstream rows enter only as prompt context: queries, tool schemas, and the source's
earlier turns and function responses. Counts are exact, from the pack's `audit.jsonl`.

| source | examples / turns (training) | supervised tokens (training) | validation | license and terms | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| glaiveai/glaive-function-calling-v2 @e7f4b645 | 1,421 / 1,591 | 482,892 | 13 turns, 12,624 | Apache-2.0 tag only; the card says nothing about how the rows were generated; the repo marks it "provenance undocumented" | unproven |
| Salesforce/xlam-function-calling-60k @26d14ebf | 19 / 19 | 9,695 | 0 | CC-BY-4.0, gated, "for research purposes only"; rows 0–116 are DeepSeek-V2-Chat output, whose license treats a model trained on its synthetic data as a Derivative bound by its use restrictions | conditional |

- **Restricted sources, for the maintainer's ruling: both.** Neither is clearly covered for C1's training use.
  - glaive: is a permissive tag with undocumented generation provenance enough?
  - xLAM: research-only use, under an access agreement nobody has reviewed. DeepSeek's derivative restrictions
    would bind any later distribution of C1's checkpoints.
- C1 is not launched and its pack is not modified. Dropping these rows would make it a different pack from
  D4-Augmented's, so it could no longer be called the same data.
- Seen in passing: one glaive conversation (`glaive-000941`) has turns in both training and validation.

---

## D. The A1 phase (maintainer 2026-11-01, "Execute A1 First": executing)

A1 end to end: the hybrid D4 full search's inputs, the search, A1's two seeds trained from its Top-1 and formally
evaluated. The 3 × 2 matrix is unchanged; A2 and C1 are neither trained nor evaluated and consume no budget. Research
B stays pending.

**Money.**
- A1's new spend has a hard cap of $77.00. The phase caps are:
  - inputs $2.65;
  - search $41.41;
  - 350 GB of storage for 14 days, $13.34;
  - $7.05 per seed.
- The grant (`approvals/teacher_modes_token_matched_grant.json`, amended) enforces a cap per session kind:
  - `inputs` $2.65, `search` $41.41, `train_a1` $14.10;
  - every kind draws on one sessions envelope of $63.66;
  - the volume's $13.34 sits outside that envelope.
- Neither Research A's $142.19 nor the project's $868 is raised. No A100 is used.
- The provider's `spendLimit` ($80) is per hour, not cumulative (provider docs). The cumulative constraint at the
  provider is the balance.

**Resources.**
- One L40S at a time.
- The new volume `4rhubprswo` (350 GB, US-TX-3, created 2026-11-01): the only datacenter at the time with both network
  storage and L40S stock.
- `am5m373kuq` stays the historical archive.

**Phase A (inputs)** (`inputs_session.py`, kind `inputs`, `inputs_protocol.json`).
- Calibration top-up (`calibration_topup_v1.jsonl`): 22 string_induction units (refill plus reserve, in D4's order)
  and 14 code prompts, both modes.
  - rag_evidence H-N was never short: its two refill units were the ones isolation had removed. Passing over them,
    the existing records reach the quota.
  - Quotas are unchanged. Cells under 5 tasks are reported THIN.
- State evaluation, re-queued before any generation:
  - the maintainer's minimum of 5 independent tasks per (type, mode) beside the 12,000 scored positions;
  - queues at 2× the expected need: 2,459 prompts, of which 1,716 of the first build's 1,722 are unchanged in place;
  - code prompts re-exported with reasoning-core 0.5.0 at the same seed bases (supersets);
  - the isolation proof regenerated (R1–R4, R7 and R9 pass).
- On the pod, each type's queue is generated in order until both thresholds hold, both modes from the queue's start.
- Then, at $0:
  - the profiles are rebuilt (`build_calibration_supplement profiles --inputs-records`);
  - the suite is frozen (`freeze_state_eval.py`: tags, one critical group per mode, manifest, measurement protocol id);
  - the isolation and type audits are re-run.

**Phase A result (2026-11-01).**
- Session `a_inputs_20261101_125916`: `ALL_DONE` in 30.9 min, $0.5713 of the $2.65 cap. The pod was deleted and the
  provider confirms it gone. Two earlier launches were refused at $0 with nothing created.
- Calibration is COMPLETE (0 short cells):

  | profile | thinking positions | non-thinking positions | items | tokens |
  | --- | ---: | ---: | ---: | ---: |
  | Base | 103,961 | 93,123 | 565 | 472,966 |
  | RH | 105,057 | 93,149 | 377 | 339,452 |

- `state_eval_hybrid@v1` is FROZEN (`audit/state_eval_hybrid_freeze.json`):
  - 1,135 items; every one of the 26 (type, mode) cells holds at least 12,000 scored positions and 5 tasks;
  - thinking mode: 146 items, 192,246 positions; non-thinking mode: 989 items, 158,169 positions;
  - every critical class is present in each mode's group;
  - measurement protocol id `1c8305246073213e58d0c1098ba4e4e0`.
- The search driver's $0 check-only run with the real inputs passes:
  - config hash `1dcae927…`;
  - the protocol id is the frozen record's;
  - fingerprint `24787ea9…`.

**Phase B (the search)** (`full_search.py`, kind `search`).
- The method is resolved from the forward baseline, with these choices:
  - the hybrid root, its shards hashed on the pod;
  - micro-batch 1, prefix reuse and on-device validation;
  - `prune_weights` false;
  - seed 20260815 and C3's 596M geometry.
- Level 0's first evaluation is also the device check:
  - both modes' critical groups must be present and their 50:50 mean must hold;
  - the frozen epsilon rule: the child materialized, reloaded and measured 10 times, with the range below 1e-4.
- A deadline stops the search with its journal intact; a partial ranking is never a Top-1.
- On completion:
  - the commit (k = 1);
  - the Pareto order of every complete leaf;
  - the Top-1 copied to `search_a1/top1` on the volume, by sha256, and verified there by sha256 at closeout (its one
    consumer, the `train_a1` pod, reads it on the volume; the bytes are not carried to the dev box).
- Attempt 1, `a_search_20261101_142659`, failed before any measurement:
  - the pod came up after 11 stock retries;
  - the driver failed 2 s after `CONTRACT_OK`: it looked the teacher up offline as the whole repository, which
    huggingface_hub 1.x refuses (README.md, LICENSE and .gitattributes were not fetched by setup), the known A-E
    failure;
  - the $0 check-only run stops before this step, so it never reached the line;
  - the pod was deleted, the provider confirms it gone, $0.1973; no state was evaluated and the journal is empty;
  - fix: `full_search.load_root` resolves the root through `shared.pod.teacher_cache.local_snapshot` (setup's file
    patterns), with a regression test;
  - the search phase's cap is cumulative, so the next authorization's hard cap is $41.21 ($41.41 − $0.20).

**Phase B result (2026-11-02).** Run `a_search_20261101_153803` (records under its run directory).
- COMPLETE: the pod came up after 43 stock retries (3.5 h, $0); the driver ran 2026-11-01T19:16Z to
  2026-11-02T15:03Z (beam search 19.8 h).
- The first evaluation's check passed: both critical groups present and their mean exact; epsilon range 0.0 over 10
  repeats.
- 93 states, 12 complete leaves, 8 distinct weight sets. DEPTH under either calibration profile selects the same
  layers, so leaves that differ only in DEPTH's profile have identical weights and metrics.
- The complete leaves in Pareto order, one row per distinct weight set (KL to the teacher on `state_eval_hybrid@v1`;
  the critical-token KL at `</think>`, final-answer and EOS positions, per mode):

  | rank | path (profile per operator) | teacher KL, equal-domain mean | worst domain | critical, thinking | critical, non-thinking |
  | ---: | --- | ---: | ---: | ---: | ---: |
  | 1–2 | DEPTH → FFN(base) → ATTENTION(base) → RESIDUAL_WIDTH(rh) | 7.338 | 7.694 | 12.637 | 11.367 |
  | 3–4 | FFN(rh) → DEPTH → RESIDUAL_WIDTH(base) → ATTENTION(rh) | 8.965 | 10.617 | 11.713 | 10.685 |
  | 5–6 | FFN(rh) → DEPTH → RESIDUAL_WIDTH(base) → ATTENTION(base) | 9.074 | 10.876 | 11.388 | 10.515 |
  | 7–8 | DEPTH → FFN(base) → ATTENTION(base) → RESIDUAL_WIDTH(base) | 7.471 | 7.848 | 12.710 | 11.706 |
  | 9 | ATTENTION(base) → DEPTH(rh) → FFN(rh) → RESIDUAL_WIDTH(base) | 11.404 | 14.770 | 20.121 | 20.409 |
  | 10 | RESIDUAL_WIDTH(rh) → DEPTH(base) → ATTENTION(rh) → FFN(base) | 11.497 | 12.583 | 19.403 | 17.618 |
  | 11 | RESIDUAL_WIDTH(rh) → DEPTH(base) → ATTENTION(rh) → FFN(rh) | 11.632 | 12.764 | 19.354 | 17.589 |
  | 12 | ATTENTION(base) → DEPTH(rh) → FFN(rh) → RESIDUAL_WIDTH(rh) | 12.981 | 16.909 | 23.806 | 22.615 |

- **Top-1:** state `159e73be3e1157e8367a476eb72e6afc`, path DEPTH(base) → FFN(base) → ATTENTION(base) →
  RESIDUAL_WIDTH(rh):
  - 596,049,920 parameters, provenance `search`;
  - root `Qwen/Qwen3-4B@1cfa9a72…`, shard-set sha256 `a8b9dec3…`;
  - `model.safetensors` sha256 `3459b9c9…`, artifact digest `ab8ffb0d…`;
  - on volume `4rhubprswo` at `/runpod-volume/teacher_modes/search_a1/top1`.
  These are KL and ranking facts about initializations, not reasoning results (P10.2).
- Cost: $23.97 of $41.21, from the provider's hourly billing.
  - $1.28 of it is the 69 minutes the pod was retained after the search had finished. The closeout tried to pull the
    1.19 GB Top-1 to the dev box (5.7 MB in 1,200 s), and its gate refused teardown.
  - The Top-1 was then hashed on the volume, and the pod deleted (provider 404, the watchdog ended `pod_gone`).
- Two repairs, each with a regression test:
  - the search closeout verifies the Top-1 on the volume and pulls no weights;
  - a closeout whose session recorded no cost writes an unknown instead of $0, so the envelope refuses the next issue
    until the run is reconciled.
- `closeout/teardown.json` records the reconciliation. `outcome.json` keeps what the launcher wrote.

**Phases C and D (A1)** (`a1_units.py`, kind `train_a1`; `training_plan_a1.json` from the committed search record).
- Seed 20260726, then seed 20260801. Each:
  - the hybrid's tokenizer staged by sha256;
  - launch admission on the pod, which checks the lineage, Top-1 bytes and effective config, the search record, the
    tokenizer and the pack;
  - training with the frozen recipe, resuming from the latest save point with its lineage stamp;
  - eval-set-1 v2 with nucleus_v1 in thinking mode.
- A failure of the first seed stops the session. The second seed is never skipped because of the first one's result.
- `a1_analysis.py` reports:
  - the static fluid correctness by axis per seed (Wilson intervals) and the two seeds' paired bootstrap;
  - the interactive axis as exploratory;
  - rollout integrity and reasoning and answer lengths;
  - the gap to the teacher's thinking mode;
  - the training logs.

**Phase C attempts (2026-11-02).** Both ended before a training step. Each is closed under its own run directory.
- `a_train_a1_20261102_165858` (pod, 13.2 min, $0.2436): launch admission refused the pinned teacher.
  - Cause: the pod's huggingface_hub links `blobs/<LFS sha256>` on to a shared Xet store
    (`hub/blobs/<xx>/<Xet hash>`). `lineage.snapshot_facts` resolved the whole chain and read the Xet hash as
    the shard's sha256.
  - The Hub's paths-info confirms each refused digest is that shard's own Xet hash, so the teacher was the pinned
    one (`closeout/diagnosis.json`).
  - Repair: `snapshot_facts` reads the link's first hop (1b8c658b).
- `a_train_a1_20261102_183510` (no pod, $0): stopped in its create-retry wait.
  - Found at $0 in the meantime: `train_stage3`'s own lineage check looked the teacher up offline as the whole
    repository, which huggingface_hub 1.x refuses when setup fetched only `teacher_cache.TEACHER_FILES`.
  - Repair: it goes through `teacher_cache.local_snapshot`, as admission does (ddd88001).
- `a_train_a1_20261102_184122`: the current session, hard cap $13.85 (the kind's $14.10 less attempt 1).

## Budget (re-estimated 2026-11-01; nothing below is authorized)

Owner: [`plans/budget_v2.json`](../../../../logs/stages/stage-2/teacher_modes/plans/budget_v2.json). Provider facts:
[`audit/provider_resources_20261101.json`](../../../../logs/stages/stage-2/teacher_modes/audit/provider_resources_20261101.json).
- Rate: the L40S secure price, still $1.09/h, plus 150 GB of container disk: $1.1105/h.
- Each item has three values:
  - lower: the fastest measured rates;
  - expected: the measured medians and the historical placement;
  - deadline: the slowest measured rates with stated allowances. It is the proposed hard stop for each paid phase.

| item | lower | expected | deadline |
| --- | ---: | ---: | ---: |
| inputs session (calibration top-up, the state evaluation's generation: about 0.40 M generated tokens) | $1.13 | $1.24 | $2.65 |
| hybrid D4 full search, one L40S at micro-batch 1 (hours) | $22.59 (20.3 h) | $27.16 (24.5 h) | $41.41 (37.3 h) |
| network volume (300 GB 3 / 7 days; 350 GB 14 days at the bills' +17%) | $2.10 | $4.90 | $13.34 |
| six trainings with their sessions (hours) | $43.16 (38.8 h) | $49.44 (44.5 h) | $74.78 (67.4 h) |
| **total** | **$68.98** | **$82.74** | **$132.18** |
| **with a $10 engineering reserve** | $78.98 | **$92.74** | **$142.18** |

What changed against the proposal ($86.38 / $92.18):
- **State evaluation.** It forwards every prompt token: about 1.00 M forward tokens for 312,000 scored positions,
  since non-thinking HotpotQA items forward about 1,420 prompt tokens for about 39 scored positions. That takes
  106 s per state on the L40S at micro-batch 1 (66–162 s), and about 2.9 h over the search, not 0.9 h.
- **DEPTH.** It stays token-linear. Its placement is the mean of the three past searches (14.67 expansions) and,
  at the deadline, the historical maximum (16).
  - Every position the beam schedule allows (38) would give 72.8 h and $80.84. That is a mathematical bound, not a
    forecast.
- **Trainings.** Priced per arm from measured steps (C above): C1 at D4-Augmented's 3.6 s, A2 at 5.66 s. The
  proposal priced every step at 6.7 s.
- **Storage.** It needs 350 GB, or 300 GB with non-Top-1 states retired after the audit.

**Against what is left.** Remaining: Research A $80.02 (of $142.19), the project $81.68 (of $868), the provider
balance $79.23. (The provider's $80 spend limit is per hour, not a cumulative cap: corrected 2026-11-01.)

| need | Research A cap | project cap | provider |
| --- | ---: | ---: | --- |
| **minimum**: expected + $10 reserve, $92.74 | $154.91 (+$12.72) | $879.06 (+$11.06) | balance +$13.51 |
| **conservative**: every phase at its deadline + $10, $142.18 | $204.35 (+$62.16) | $928.50 (+$60.50) | balance +$62.95 |
| the maintainer's suggested figures | $165 (left $102.83) | $890 (left $103.68) | +$25–30 (about $104–109) |

- The suggested figures cover the minimum with about $10 to spare. They do not cover every phase running to its
  deadline.
- **Staged authorization keeps the bound without funding the worst case up front.** Each phase gets its own deadline
  and stop rule, and the next phase is authorized on the previous one's actual cost:
  - inputs session: deadline $2.65;
  - search and volume: deadline $41.41 + $13.34;
  - trainings: per-unit deadlines A1 6.35 h, A2 17.91 h, C1 8.54 h.
- **GPU resources:**
  - one NVIDIA L40S (secure, $1.09/h) in EU-NL-1 or US-TX-3, used sequentially: the inputs session (about 1.1 h),
    the search (about 24.5 h, deadline 37.3 h), and six trainings (about 44.5 h, deadline 67.4 h);
  - one network volume of 350 GB (or 300 GB) in the same datacenter for one to two weeks;
  - no A100.
- **Levers that change no protocol:**
  - `prune_weights` after the evidence audit (volume only);
  - running C1 first, since its cost is the best measured;
  - padding-free attention for the per-turn-isolated packs. The trainer now builds a dense block-diagonal mask,
    measured at 1.6–1.8× a cross-context step. Varlen attention over the packed turns would cut A1/A2's attention
    work, but it needs flash-attention in the training image and an equivalence check on the L40S;
  - an A100 for the search at micro-batch 3 (a hardware change, separately approved).
- No cap is raised, no A100 is used, and the epochs, supervised-token scale and distillation objective are
  unchanged.
