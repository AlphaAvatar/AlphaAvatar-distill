# Research A v2 — token-matched data, the hybrid D4 full search (proposed), the 3 × 2 training matrix (proposed)

**Status: data preparation only.** Nothing in this round searches, initializes, trains or evaluates a student,
and Research B stays pending. The maintainer reviews three things: A, the data audit; B, the full-search plan;
C, the training plan.

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

## A. Data (admitted 2026-11-01)

Owner: [`data/token_matched_admission.json`](../../../../logs/stages/stage-2/teacher_modes/data/token_matched_admission.json),
terminal `ADMITTED` (design `2ba937fb…`, plan `a48ce65a…`, order_v2 `02c98b81…`). Built on the dev box at $0 from
v1's sealed records (identity `848f6508…`) and the supplement's (`0d39ad93…`), through v1's frozen verdict.

**Both arms meet the target** (per mode 5,923,021; total over target ≤ 0.5%, category share within 0.5 points).

| | A1 (thinking) | A2 (non-thinking) |
| --- | ---: | ---: |
| examples | 5,957 | 49,826 |
| turns | 6,105 | 53,132 |
| packed sequences (8,192 tokens) | 1,102 | 4,576 |
| unique supervised tokens | 5,930,273 (+0.122%) | 5,923,670 (+0.011%) |
| base-corpus / fluid share | 49.923% / 50.077% (0.060 points) | 49.978% / 50.022% (0.005 points) |
| reasoning / answer tokens | 4,983,377 / 946,896 | 0 / 5,923,670 |
| non-padding tokens (prompt/context) | 8,924,771 (2,994,498) | 37,327,460 (31,403,790) |
| padding share | 1.14% | 0.42% |
| optimizer steps (3 epochs, batch size 2) | 1,653 | 6,864 |
| supervised-token / non-padding-token exposure | 17,790,819 / 26,774,313 | 17,771,010 / 111,982,380 |
| training FLOPs | 5.14e17 | 2.14e18 |
| supervised tokens reused from v1 / new | 3,293,941 / 2,636,332 | 2,774,856 / 3,148,814 |
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
| SQuAD v2 answerable | 592,101 | 591,549 | 591,546 | 0.000 |
| HotpotQA | 592,101 | 591,757 | 591,531 | 0.000 |
| code: MBPP | 38,431 | 38,512 | 38,291 | 0.000 |
| code: reasoning_core | 553,670 | 554,937 | 555,502 | 0.147 |
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

**One known false pass, disclosed.** An MBPP candidate in A2 (`d934be3a…`, 171 supervised tokens) unpacks
`set(s)` into two names: its asserts pass at string-hash seeds 0 and 1 and fail at 2 and 3. The code sandbox now
pins the seed (`aadistill.data.code_exec` v4, e5d34d3a), so it is admitted on every run; before, the verdict was
random per process (the supply audit passed it, the first admission failed it and came up 171 tokens short of the
MBPP cap). It is the only hash-dependent one of v1's 812 MBPP candidates. Removing it would move 171 tokens of the
MBPP cap to the reasoning_core code cell: the maintainer's call.

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
- A2 pack: `blocks.npz` `cd9e1a23…` (376.2 MB), `ladder.json` `d4d53cc8…`, `audit.jsonl` `d1f0f077…`.
- Both `token_documents.document_isolated` (each turn attends only to itself), sequence length 8,192, teacher
  `Qwen/Qwen3-4B@1cfa9a72` in its own mode. Packs and stores stay outside git (`artifacts/`, `.scratch/`); the
  supplement's sealed shards are also on am5m373kuq.

**Actual cost.** The supplement's two sessions: $15.8689 ($5.7873 + $10.0816), inside its original $23.00
(`a_supplement_20261031_091618` created no resource). Study A-TM: $15.8689 of $95.8883 spent, $80.0194 left.
Project: $786.3228 of $868 committed, $81.6772 left; the provider account's balance is $79.31.

## B. The hybrid D4 full search (proposed, not run)

Owners:
- [`full_search_proposal.json`](../../../../configs/stages/stage-2/teacher_modes/full_search_proposal.json);
- [`full_search_types.json`](../../../../configs/stages/stage-2/teacher_modes/full_search_types.json);
- [`audit/full_search_isolation.json`](../../../../logs/stages/stage-2/teacher_modes/audit/full_search_isolation.json);
- [`hybrid_d4_profiles.json`](../../../../configs/stages/stage-2/teacher_modes/hybrid_d4_profiles.json).

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

- **Interactive adaptation is excluded from both sets.** A token-level KL on a fixed transcript is not closed-loop
  interaction, and D4's profiles carry no interactive domain (their weights are kept). Interactive adaptation stays
  in A1/A2's training and is measured closed-loop on eval-set-1. Including it as a sixth domain in both sets is the
  maintainer's alternative.
- **Training sources not typed here**, each listed with its reason in the table: the reasoning_core and
  reasoning_gym fluid cells, and MBPP (exhausted by training).
- Scored positions are the positions that predict the hybrid's own generated tokens, `<|im_end|>` included; no
  prompt, template or preclosed think block is scored.
- Critical tokens follow each mode's native output: H-T has `think_close`, `final_answer`, `eos`; H-N has
  `final_answer`, `eos`, never a `</think>` it did not generate.

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
- Coverage now (scored positions): base H-T 92,648 / H-N 83,999, rh H-T 95,415 / H-N 85,084.
- Short cells: string_induction, code (not yet generated) and rag_evidence (H-N). The top-up is the code cell's 9
  prompts plus about 13 refill units.

**State evaluation `state_eval_hybrid@v1` (new; `state_eval@v1` is unchanged and unused).**
- 1,722 prompts, all new.
- Fluid prompts are new instances of the same @v1 generators: calibration-split families, seeds 2,500,000–2,570,030
  in the calibration range.
- Dataset prompts are train rows past everything order_v2 holds, rendered as corpus_v2 rendered them.
- openmath comes from base_pool_v1's unused units; code from reasoning_core at seed base 2026320000.
- Quota: frozen at 12,000 scored positions per (type, mode), 312,000 in all.
- Scoring: D4's three objectives under equal sub-type (mode included) then equal domain, so the modes are 50:50
  and no profile gains from raw tokens. Critical-token KL is 0.5 × the H-T classes' mean + 0.5 × the H-N classes'
  mean, which needs a small evaluator extension before the search. Per-mode KLs are recorded beside the ranking
  values.
- The manifest binds the root teacher (weights, revision), the tokenizer, the template, both modes, the type
  table, every item's prompt and output ids with their masks, the critical classes, the isolation proof, the content
  sha256, the suite hash and the measurement protocol id.
- Loader: `hybrid_state_eval.load` refuses any other binding and then uses the shared loader; phase_d1's frozen
  loader is not edited.

**Isolation** (`full_search_isolation.json`).
- State evaluation against calibration and against all training of both lineages: no shared prompt key, example id (`task_id`) or
  content. 4 and 208 13-gram near-copies were removed.
- Calibration against A1/A2's training: 0.
- Calibration against C1's training: 232 shared corpus_v2 prompts, all held out of the profiles.
- State evaluation against eval-set-1: R1, R2, R3, R4, R7 and R9 by family name pass.
- Rule families and generators are shared between calibration and state evaluation by design; instances, seeds and
  rows are not.

**Execution.**
- One L40S at micro-batch 1 for the whole search. Micro-batch 3 measured out of memory in DEPTH on an L40S with this
  root. An A100 at micro-batch 3 needs a separate approval.
- Journal and states go to a network volume after every expansion, and a resume restores only re-identified bytes.
- Storage needs about 300 GB, or `prune_weights` (a maintainer decision).

**Budget** (D4's measurements, DEPTH token-linear).
- The hybrid profiles hold 449,155 and 319,611 tokens, 1.97× and 1.35× D4's, because non-thinking supporting items
  pair long prompts with short answers.
- Expected 20.5 h on the L40S ($22.35) and conservative 25.6 h ($27.9), plus about $5 of storage.
- The inputs session (calibration top-up and the state evaluation's generation) is about $1.0–1.5.

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
| packed sequences, optimizer steps | 1,102 sequences, 1,653 steps (warmup 83, save every 826) | 4,576 sequences, 6,864 steps | 2,468 sequences, 3,702 steps (warmup 185, eval every 462, save every 1,851) |
| supervised tokens / exposure | 5,930,273 / 17,790,819 | 5,923,670 / 17,771,010 | 5,923,021 / 17,769,063 |
| non-padding tokens | 8,924,771 (1.1% padding) | 37,327,460 (0.4% padding) | 7,882,298 (61.0% padding) |
| reasoning / answer tokens | 4,983,377 / 946,896 | 0 / 5,923,670 (preclosed think) | E1 part 2,142,358 / 818,149; fluid part 2,677,056 / 285,458 |
| tokenizer | the hybrid's | the hybrid's | the PCA init directory's (Thinking-2507's) |
| evaluation | eval-set-1 v2 (frozen), nucleus_v1, thinking mode | the same, non-thinking mode | the same, Thinking-2507's template |

**Lineage gate** (`aadistill.training.lineage`; every unit's `lineage` block in `training_plan_proposal.json`). One
teacher identity per run: model id, revision, weight shards, vocabulary, chat template, RoPE base, position limit.
It is checked three times, and fails closed every time:
- at config validation (the KD teacher against the declared lineage);
- at launch admission (`training_plan.admit`: the initialization's bytes, the student config and the pack's bytes);
- by the trainer before a step runs (`train_stage3.py`: the loaded teacher and its snapshot shards, template and
  RoPE; the student config and initialization bytes; the pack; the tokenizer).

Two things are refused, with tests:
- a Thinking-2507-initialized student distilled by the hybrid, or the reverse;
- data of another teacher or mode.

Equal vocabularies (both `7781771a…`) and an equal geometry do not make two teachers one lineage: the RoPE base, the
template and the weights tell them apart.

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

**Cost (estimates, `training_plan_proposal.json`).**
- Basis: 6.7 s per optimizer step on the L40S (v1's measured H-T-C), the student evaluation's 3,917 s per unit,
  10 min fixed per unit, $1.09/h.
- A1: 1,653 steps, 4.33 h, $4.72 per seed. A2: 6,864 steps, 14.03 h, $15.29 per seed. C1: 3,702 steps, 8.14 h,
  $8.88 per seed.
- Six runs: $57.78.

---

## Budget (exact figures; nothing below is authorized)

| item | expected | conservative |
| --- | ---: | ---: |
| full-search inputs session (calibration top-up, the state evaluation's generation) | $1.25 | $1.50 |
| hybrid D4 full search, one L40S at micro-batch 1 (20.5 h / 25.6 h) | $22.35 | $27.90 |
| search storage (about 300 GB for a week, or `prune_weights`) | $5.00 | $5.00 |
| six trainings (A1, A2, C1 × 2 seeds) | $57.78 | $57.78 |
| **total** | **$86.38** | **$92.18** |
| left: Research A ($142.19 cumulative) | $80.02 | $80.02 |
| **gap against Research A** | **$6.36** | **$12.16** |
| left: project ($868 cumulative) | $81.68 | $81.68 |
| **gap against the project** | **$4.70** | **$10.50** |

- The provider account's balance ($79.31) is below either total.
- A2 is the largest item: its 37.3M non-padding tokens make each seed about 3.2× A1's cost.
- Neither cap is raised and no A100 is used without the maintainer. Options that change the total:
  - `prune_weights` (about $5 less storage);
  - micro-batch 3 on an A100 for the search (a hardware change);
  - a cap decision.
