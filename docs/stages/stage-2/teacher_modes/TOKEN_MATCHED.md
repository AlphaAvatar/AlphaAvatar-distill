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

## A. Data

_Filled from the admission record once the supplement session's records are secured._

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
- State evaluation against calibration and against all training of both lineages: no shared prompt key, task id or
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
| packing / context strategy | token documents, document-isolated | token documents, document-isolated | D4-Augmented's ladder: cross-context within system-prompt groups (kept, disclosed) |
| packed sequences, optimizer steps | from the admission | from the admission | 2,468 sequences, 3,702 steps (warmup 185, eval every 462, save every 1,851) |
| supervised tokens / exposure | about 5.923M / 3 × | about 5.923M / 3 × | 5,923,021 / 17,769,063 |
| non-padding tokens | from the admission | from the admission | 7,882,298 (61.0% padding) |
| reasoning / answer tokens | from the admission | answer only (preclosed think) | E1 part 2,142,358 / 818,149; fluid part 2,677,056 / 285,458 |
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
  packing against document isolation. C1's E1 part carries 352 traces cut at 8,192 (trained without `<|im_end|>`),
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

**Cost (estimates; the admission's numbers replace the A rows).**
- Basis: 6.7 s per optimizer step on the L40S (v1's measured H-T-C), the student evaluation's 3,917 s per unit,
  10 min fixed per unit, $1.09/h.
- C1: 3,702 steps, about 8.1 h and $8.9 per seed.
- A1 and A2: from their packs below.

---

_Sections A and the budget summary are completed after the supplement session._
