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

**Root teacher.**
- `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`.
- Weight shards' LFS sha256 `328a91d3…`, `6cd087b3…`, `e4bf4369…` (`teacher_identity.json`).
- Chat template `a55ee1b1…`, RoPE base 1,000,000, 40,960 positions.
- Its thinking and non-thinking modes are two official modes of one set of weights, so A1 and A2 share one
  initialization.

**Method: D4's own, resolved from the selected forward baseline, never copied.**

| term | value |
| --- | --- |
| baseline | `stage1_init.d1_target_aware.scored_v1@v1` |
| operators | ATTENTION `activation_importance_v1`, DEPTH `causal_kl_greedy_v1`, RESIDUAL_WIDTH `global_pca_v0`, FFN `activation_importance_v0` |
| free variables | operator order and each operator's calibration profile (hybrid Base or hybrid Reasoning-Heavy) |
| beam | `beam.delayed_prune@v1`, width 6, warmup 1 |
| ranking | `beam.pareto_multi_objective@v2`: equal-domain mean KL, worst-domain KL, critical-token KL, ε 1e-4 |
| positions / support / rows | `positions.supervised_target_v1@v1` / `reference_topk_tail_v1` (K = 200 + tail) / `scored_v1` |
| packing / DEPTH prefix reuse | `length_sorted_v1` / on, re-validated on the search card |
| KL references | DEPTH against the parent state; state evaluation and ranking against the root, the hybrid, recomputed per state |
| geometry | 596,049,920 parameters: 28 layers, hidden 1024, 16 Q / 8 KV heads, head dim 128, FFN 3072, tied embeddings |
| seed, commit | 20260815; Top-1 = the first of the 12 complete leaves in the Pareto quality order |

The tree is D4's: 8 + 48 + 24 + 12 = 92 expansions.

**Calibration: both modes, balanced on scored positions, v1's trajectories first.**
- Two profiles keep D4's design: `calib.hybrid_d4_base@v1` (0.18 per static primary axis, 0.28 supporting) and
  `calib.hybrid_d4_rh@v1` (0.22, 0.12).
- Each has D4's 4× volume, 180,000 scored positions, over D4's 12 cells (one per subtype) and in D4's cell order
  over `base_pool_v1` (seed 1308798386).
- **Each mode fills half of every cell's quota with its own naturally terminated trajectories.** The subtype carries
  the mode (`<subtype>|<mode>`), so the subtype-then-domain reduction weighs the modes equally inside every domain.
  The result is ONE initialization, not one per mode.
- Coverage audit: v1's sealed hybrid calibration is D4's 4× prefix. It covers the thinking mode, one cell 829
  positions short. The non-thinking mode has about 27,500 of 90,000 positions per profile, because its answers are
  short.
- 801 further units of D4's own pool were therefore generated (round 0 of the supplement session, one candidate, v1's
  rule), continuing D4's cell order: all 801 in the non-thinking mode, 12 of them also in the thinking mode. Every
  cell stayed inside its pool.
- The profiles are written from the sealed records on the dev box (`build_calibration_supplement.py profiles`).
- Isolation: the calibration prompts share no key with any training order, T0, the pools or eval-set-1's held-out
  prompts (R1, R2), and come from no family R9 refuses.

**State evaluation.** D4's method (`state_eval@v1`) with hybrid references. One choice for the maintainer:
- (a) The suite's items as they are, which are Thinking-2507 trajectories.
- (b, recommended) The same 80 prompts with the hybrid's own trajectories in both modes, balanced on prediction
  positions. This is 160 short generations at the start of the search session.

The suite loader binds Thinking-2507's suite today, so it needs a hybrid binding in either case.

**Execution.**
- **Card and micro-batch.** One L40S at micro-batch 1 for the whole search, declared before the run. Micro-batch 3
  ran out of memory in DEPTH on an L40S with this root on D4's items (`a_check_20261029_190241`, 10.99 GiB asked of
  44.39 GiB); micro-batch 1 completed with a 26.88 GiB peak. Micro-batch is a numerical-execution field: semantic
  state ids are unchanged, while materialization ids and the execution fingerprint differ.
- **Alternative.** The historical profile, an A100 at micro-batch 3, would need a separate approval.
- **Persistence and resume.** The journal and every state are on a network volume after each expansion. A resumed
  search replays the journal and restores only states whose bytes re-identify.
- **Storage (a decision for the maintainer).** D4 kept 274 GB of states. That needs a volume of about 300 GB in the
  card's datacenter, or `prune_weights` true (pruned states' bytes deleted, the journal kept).
- **Estimate.** From D4's measured 11.1 h of operator time on an A100 at micro-batch 3 (DEPTH 9.7 h) and the L40S
  check's DEPTH ratio of 1.05: about 13.5 h expected, 18 h conservative, $15.2–20.1 at $1.09/h, plus storage.

**Product and sharing.**
- The Top-1's checkpoint identity (artifact digest, weights digest, single-shard sha256, config sha256, 596,049,920
  parameters) and its materialization identity, with the root teacher recorded.
- A1's and A2's four units bind its single-shard sha256 in their lineage blocks.
- The full journal, every candidate's order and profiles, every expansion's KL and state-evaluation metrics, the
  Pareto decisions, memory and time telemetry, and a structural comparison with D4 Top-1 (ATTENTION(rh) →
  DEPTH(base) → RESIDUAL_WIDTH(base) → FFN(base)) that changes nothing.

**Differences from D4's search:**
- the root teacher;
- the calibration trajectories, which are the hybrid's own, half from each mode;
- the state-evaluation references;
- the L40S at micro-batch 1;
- everything else is unchanged.

**Differences from Research A v1's fixed path:**
- v1 ran D4 Top-1's path without searching;
- v1's profiles held about 55,000 positions balanced by whole items, against D4's 180,000 here with per-cell mode
  balance.

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
