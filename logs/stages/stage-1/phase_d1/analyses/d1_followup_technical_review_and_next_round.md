# D1 follow-up: where target-aware KL really saves compute, and the next round

**Date:** 2026-10-12 (UTC). **Status:** a technical review against the live code
and a next-round design. Nothing here re-runs D1, edits its record, or promotes
q2. D1's confirmation verdict stands: **NO-GO**.

**Maintainer decision recorded with this review** (logs/budget/decisions.md,
2026-10-12):
- D1's technical route is kept.
- **B** (`53e30566…`) stays the standing incumbent and the scientific anchor.
- **q2** (`3a4b4ede…`, state `afd50c9f…`) becomes the **working initialization
  baseline**: a starting point for method work, not an incumbent.

Owners of every number below:
- `d1_scored_position_profile.json`, written by
  `scripts/shared/data/scored_position_profile.py` and reproducible at $0.
- The closed Top-K adoption campaign, `validations/topk-adoption/v1/`, runs a4,
  a6 and a8.
- The GPU qualification, `validations/gpu-qualification/v1/`.

---

## 1. What D1 computes per position: verified in code

| path | where | positions that reach the LM head | positions that get the full-vocabulary work | where the target mask applies |
|---|---|---|---|---|
| DEPTH, reference | `operators/depth/causal_kl_greedy.py` `_ReferenceSketches.sketch_block` → `_forward_logit_block` | every slot of the `[B,T]` forward, padding included | every **valid** row: logsumexp, Top-K, complement logsumexp | nowhere; the sketch covers all valid rows |
| DEPTH, each of 260 candidates | `score()` → `_forward_logit_block` + `sketch_forward_kl_mean_batch` | every slot, padding included | **all B×T rows**: fp32 copy, logsumexp over V, gather, complement logsumexp over V | `masked_item_means`, after the per-row KL exists |
| state evaluation, teacher and candidate | `planning/metrics.py` `_logit_pairs` → `distortion` / `sketch_reference` + `distortion_on_reference_support` | every position of every item | every row: KL, reverse KL, two CEs, two argmaxes | multiplied into the per-chunk sums |
| FFN, WIDTH, ATTENTION statistics | `operators/_common.collect_activation_stats` | none (no vocabulary work) | none | `ActivePositions.token_mask_for` restricts what is **accumulated** |

The maintainer's code review is confirmed. The assistant-target mask
(`positions.supervised_target_v1`) is implemented in two places only: the
**aggregation** of DEPTH and state evaluation, and the **accumulation** of the
three activation statistics. No vocabulary-sized work is skipped for a
non-target row anywhere.

One refinement: the DEPTH reference sketch excludes padding but not prompt
rows. The candidate reducer excludes neither until the masked mean.

**What a mask after the logits saves:** nothing in the LM head, and nothing in
any V-wide reduction. It changes only which already-computed per-row values
are summed.

## 2. What Top-200 + tail actually bought: measured

| quantity | full vocabulary | reference Top-200 + tail | source |
|---|---|---|---|
| reference state for 59,763 positions | 18.16 GB (bf16 `[T,V]`) | **144.1 MB** (ratio 0.0079) | a8 `reference_cache` |
| DEPTH cell at the root (one expansion, end to end, max) | 34.35 min | **22.98 min** | closeout `D1_repricing` |
| DEPTH operator invocation (production code, treatment arm) | not measured on this code | 19.90 min max, 17.49 GB peak | a8 |
| per-candidate split, synchronized (domain_balanced) | not measured | forward 4.0–4.5 s (86%), reduction 0.76 s (14%) | a6 |
| decisions | — | 0 of 8 DEPTH rounds move; per-candidate Spearman 0.99998; zero lower-bound violations | a4 |

**What it saves:**
- **Storage:** the reference state shrinks by 126×. It now always fits beside
  the model, so the reference is cached, never recomputed per candidate.
- **The reference side's per-row softmax:** p·log p over V on every candidate
  evaluation.
- **Both effects are visible** in the measured cell, 34.35 → 22.98 min.

**What it does not save:** every candidate row still pays the full-vocabulary
normalizer, which the coarse partition needs to mean anything. Its
`logsumexp` over V, the gather onto the reference support and the complement's
own `logsumexp` over V are all O(V) per row, and the LM head GEMM is unchanged.
The per-row cost on the candidate side is still O(V). It is not O(K), and this
review does not claim that.

## 3. Pre-vocabulary position selection: implemented, opt-in, versioned

**Mechanism** (`src/aadistill/initialization/scoring/rows.py`):
1. The backbone runs over the whole sequence, so causal context is unchanged.
2. The rows a reducer weights are picked from the final hidden states: valid,
   and with weight > 0. Prediction row `j` reads token `j`'s state, the same
   shift as `logits[:, :-1]`.
3. Only those rows go through the output head and the V-wide reductions.
4. The per-row values are scattered back into the dense layout. The **same**
   reducer then forms the per-item means: `masked_item_means` for DEPTH, and
   `distortion` / `distortion_on_reference_support` for state evaluation.

**Contract additions, all generic:**
- `ArchitectureAdapter.final_hidden_states` and `.output_logits`, behind
  `Capability.OUTPUT_HEAD`. These are implemented for Qwen3 only, in its
  adapter. Families with softcapping or a logit scale put it in their own
  `output_logits`.
- `ExecutionConfig.prediction_rows` takes `all_v1` (the default and the
  historical path) or `scored_v1`. It is an **execution** knob: the estimand,
  the weights and the aggregation are identical.
- The arithmetic path differs: a smaller GEMM, and different row grouping in
  the float32 chunk sums. So `scored_v1` is a **fingerprint** field, through
  the new `OPTIONAL_FINGERPRINT_FIELDS`.

**Identity: nothing historical moves.**
- `all_v1` adds no trace key and no fingerprint key. The default fingerprint is
  byte-identical to the pre-field hash, and a test derives it independently.
- No operator signature changed. DEPTH does **not** declare the new capability,
  because declaring it would fork every DEPTH state id; it checks the
  capability at run time instead.
- A `scored_v1` state evaluation gets a different `measurement_protocol_id`.
  It is refused unless a numerical environment is declared, because an
  undeclared execution would otherwise share the all-rows identity.
- D1's frozen protocol module (`families/d_series/scoring_protocol.py`) is
  untouched. D1 remains reproducible exactly as recorded.

**Invariants preserved:**

| invariant | how it holds |
|---|---|
| causal shift | `hidden[:, :-1][rows]` |
| teacher and student at the same positions | one selection vector per item, used for both |
| tail numerics | unchanged: the same `sketch_forward_kl` and `complement_log_mass` |
| per-item and domain aggregation | unchanged: the same reducers on the same weights |
| counts | `positions`, and the per-tag position counts, restored from the full masks |

**Refusals:**
- `full_vocab_v1` with `scored_v1` is refused in DEPTH. The full-vocabulary
  path is the oracle the coarse support is validated against, so it stays on
  the all-rows arithmetic.
- An adapter without `output_head` is refused.

**CPU evidence** (`tests/initialization/test_prediction_row_selection.py`,
24 tests on genuine tiny Qwen3 models with right-padded, length-sorted batches):
- DEPTH: removal order identical, and every per-candidate score within 1e-6
  relative. Covered under both policies and at batch sizes 1 and 3.
- State evaluation: every value within 1e-6 relative, with identical positions,
  tag counts and denominators. Covered under both supports, both policies and
  both groupings.
- Deltas of two cross-entropies are bounded at the CE scale, 1e-6 absolute.
  Reordering the float32 sums moves them by about 6e-8.
- Two mutations are caught: a wrong causal shift (13 failures) and a dropped
  count restoration (4 failures).

**Expected gain on D1-shaped data.** This is derived; the GPU measurement in §6
owes the measured figure.

| mixture or suite | scored / prediction positions | scored / forward slots at micro-batch 3 |
|---|---|---|
| `calib.domain_balanced@v1` | 0.7487 | 0.7198 |
| `calib.reasoning_heavy@v2` | 0.7835 | 0.7624 |
| `state_eval@v1` | 0.7297 | 0.7146 |
| prompt-heavy domains (RAG/multihop, tool) | 0.43–0.46, 0.56–0.63 | — |

**DEPTH's parent at the root** is the FFN-compressed teacher: hidden 2560,
36 layers, intermediate 3072, vocabulary 151,936.
- Its output head is 2·2560·151,936 = 0.78 GFLOP per row.
- Its backbone is about 3.6–4.1 GFLOP per row.
- So the head is about 16–19% of the forward.

**What that implies for DEPTH time and memory:**
- Combined with the measured 14% reduction share, vocabulary-dependent work is
  about 28% of DEPTH time.
- Skipping the roughly 25–28% unscored rows should therefore save **about 8% of
  DEPTH wall time**: roughly 1.6 of 19.9 min per invocation.
- It should also cut about 28% of the vocabulary-sized transient memory.

**Where it matters more or less:**
- **Prompt-heavy data:** about 15% or more, with RAG near 43% scored.
- **Larger vocabularies, or a narrower parent:** both raise the head's share.
- **Long teacher-native reasoning traces:** smaller, because they push the
  scored fraction toward 1.

Its main near-term value is memory headroom, which allows a larger micro-batch
on the same card, and readiness for long-context or retrieval mixtures. It is
not a large speed-up on D1's mixtures, and this review does not claim one.

## 4. q2 as the working baseline, B as the anchor

| | q2: working initialization baseline | B: standing incumbent and scientific anchor |
|---|---|---|
| identity | state `afd50c9ff0e3d20542bb5473ea745b55`, artifact `3a4b4ede6888e2b0…` | artifact `53e30566c5f795f1…` (C1's promoted treatment) |
| bytes | durable at `aad-artifacts/phase_d1/d1_replay_002/products/afd50c9f…`, identity re-verified | not on the dev host |
| on a pod | digest-pinned replay from the committed search journal; 948 s measured in the confirmation | frozen fixed path, `phase_c2.baseline.frozen_baseline_spec`; 1,280 s measured |
| what it is evidence for | the D1 search's best-ranked finalist under the frozen quality rule; usable-rollout +9.5 points vs B on all five seeds across both rungs (exploratory) | the arm every formal comparison is made against |
| what it is not | a better initialization than B on the preregistered endpoint (NO-GO); an incumbent | — |

**Rules for the next round:**
- D2, D3 and Stage-0 data experiments start method work from q2's chain:
  - the operators `ffn.activation_importance_v0`, `depth.causal_kl_greedy_v1`,
    `width.global_pca_v0` and `attention.activation_importance_v1`;
  - D1's position policy and support;
  - D1's execution protocol.
- They reuse D1's search journal, operators and identities. They do not re-run
  the D1 search.
- **Every formal behavioural comparison includes B**, under the C0 rule:
  stratified prompt-cluster bootstrap, SESOI 0.01, seeds as fixed blocks.
- q2 may appear as a second comparator. A claim of "better than q2" is never a
  claim of "better than B".
- Promoting anything, q2 included, needs a new preregistered confirmation on
  fresh prompts.

## 5. Stage-0 teacher-native data: what exists and what is missing

**What exists already.** D1's calibration is mostly teacher-native thinking
data:
- `calib.domain_balanced@v1` has 51 of 67 items templated; `@v2` has 50 of 62.
  The templated items are whole `corpus_v2` sessions:
  - generated by `Qwen/Qwen3-4B-Thinking-2507@768f209d` in native thinking mode,
    chat template `3802169b…`;
  - vLLM 0.26.0, T=0.6, top-p 0.95, top-k 20, n=4;
  - an 8,192-token session cap, and acceptance on hygiene only;
  - `reasoning_content` and `content` both rendered.
- Tags are prediction-position indices, scoped to the final assistant turn
  (`build_e8_calibration.tag_positions`):
  - `assistant` = reasoning + final answer;
  - `reasoning` = strictly inside `<think>…</think>`;
  - `final_answer` = from after `</think>` through `<|im_end|>`;
  - `think_close`, `eos` and `tool_close` each get their own tag.
  - Prompt, system and user turns, earlier assistant turns and padding are
    excluded.
- `supervised_target_v1` therefore already scores thinking plus final answer.

**The two D1 profiles are one pool.**
- `calib.domain_balanced@v1` (built 2026-08-10) has 67 unique items:
  - 51 `corpus_v2` sessions, with prompts from MBPP, GSM8K,
    OpenMathInstruct-2, HotpotQA, SQuAD v2 and Glaive;
  - 16 untemplated FineWeb-Edu documents: the whole `general` domain, 8,287
    positions, 13.9%.
- `calib.reasoning_heavy@v2` (2026-08-24) is a with-replacement reweighting of
  that same frozen pool, seed 20260812. Its domain weights are math 0.35,
  rag 0.25, code 0.20, general 0.10 and tool 0.10.
  - It has 62 draws but only 51 unique items, and some sessions appear up to 3
    times.
  - Code is 8 draws of 4 sessions, and openmath is 6 draws of 3.
  - It contains no item `domain_balanced@v1` lacks, so it cannot add length
    coverage.
- The FineWeb block is the only part of either profile that is not teacher
  behaviour. A teacher-native profile must decide explicitly whether to keep
  it, and report it as its own domain if kept.

**What is missing is coverage.** These are measured, against the same corpus.

| slice | sessions | median tokens | p90 | admitted by the builder's 256–2,048 window |
|---|---|---|---|---|
| code | 1,123 | 4,431 | 7,285 | **14.4%** |
| openmath | 579 | 2,771 | 7,347 | **39.0%** |
| multihop_qa | 1,074 | 2,210 | 3,735 | 37.8% |
| gsm8k | 1,698 | 957 | 2,191 | 88.8% |
| rag_evidence | 4,100 | 563 | 1,044 | 96.3% |
| tool_calling | 2,600 | 640 | 1,100 | 97.1% |

**The resulting gaps:**
- **Window bias:** the reasoning-heavy slices are calibrated on their
  short tail.
- **Generation cap:** the 8,192-token cap had already removed the longest
  traces: openmath lost 1,541 of 3,600 candidates and code 702 of 4,800.
- **Selection order:** items are taken first-come in corpus order, with no
  length stratification.
- **Correctness:** unfiltered by design. openmath is 0.38 correct.

**Design.** Cheapest first, and nothing is regenerated per attempt.

1. **$0, from `corpus_v2`:** a versioned profile, for example
   `calib.teacher_native@v1`.
   - Length-stratified sampling matched to each slice's own distribution, up to
     the generation cap.
   - The same leakage exclusions: the training-rung sessions, reserved
     evaluation prompts by content hash, and both D1 battery roles.
   - The same tag builder, and a content hash.
   - It is profiled with `scored_position_profile.py` before any use. Only the
     profile id enters the hashed operator config, so the four operators and
     the state evaluator need no code change.
2. **Paid generation, only if step 1 shows long-trace coverage matters:**
   - a raised session cap, using the Stage-2 v2 generation tooling unchanged;
   - persisted to durable storage with teacher revision, template hash,
     sampling, seed and corpus hash, and reused by identity.
3. **Separate position policies, only as ablations:**
   `positions.reasoning_only_v1` and `positions.final_answer_only_v1` get new
   ids, never a redefinition of `supervised_target_v1`.

**Stage-2's filtered distribution is not assumed right for Stage-0.**
Calibration wants coverage of where the teacher's computation happens, so
coverage and length are reported per slice before a profile is adopted.

## 6. Next round: one factor at a time

**Factors:**
- **S** — structural strategy (the operator chain)
- **P** — position policy
- **K** — support approximation
- **E** — execution: batch, packing, `prediction_rows`
- **D** — calibration data
- **R** — recovery training and seeds

| round | varies | held fixed | preregistered decision | cost |
|---|---|---|---|---|
| R0 engineering | E: `all_v1` vs `scored_v1` | S, P, K, D | **equivalence gate:** identical DEPTH removal order on both D1 profiles; per-candidate relative difference ≤ 1e-5; state-eval ranked metrics within the certified 1e-5 drift budget. Wall time and peak memory are reported, not gated. | one L40S session, about 1 h, **about $1.5; needs an engineering grant** |
| R1 data, $0 | D: `teacher_native@v1` vs the D1 profiles | S, P, K, E | per-slice coverage and length match; scored fraction; a frozen profile hash. No model is run. | $0 |
| R2 initialization | D, with the R1 winner fixed beforehand | S, P, K, E; q2's chain | state metrics on a **held-out** teacher-native suite, built disjoint from both profiles and both batteries: equal-domain teacher KL, worst domain, critical-token KL. Decision rule fixed before the search. | one search session (D1's measured ceiling was $21.49) |
| R3 behaviour | the R2 candidate vs **B**, with q2 as a second arm | R: the frozen recipe and fresh seeds | C0 rule: correct_overall, SESOI 0.01, stratified prompt-cluster bootstrap. Usable rollout is a guardrail, or the primary endpoint only if declared before data. | a **new battery family** is required first |

**Why one factor at a time:**
- R0 settles that E moves no decision before any science uses it.
- R1 and R2 change D alone.
- Behaviour is measured only after an initialization change has shown itself
  in the cheap metrics.
- D1's usable-rollout gain stays **exploratory**. To test it formally, a
  preregistration must name usable rollout as the primary endpoint, or as
  co-primary with correctness non-inferiority, and use new prompts.

**Hard prerequisite for R3.** The D-series battery family's two roles,
`d1_screening` and `d1_confirmation`, were consumed by D1. The earlier prompt
pools are already exhausted. Any new formal behavioural comparison needs a
fresh, contamination-checked battery family first.

## 7. Owed, in order

1. **R0:** a GPU validation of `scored_v1`. It is engineering work, and needs a
   maintainer grant (about $1.5 ceiling; engineering book). The instrument is
   ready, and it is the existing production-operator timing path:
   - run `topk_adoption_driver.py --stages D,P --prediction-rows all_v1`, then
     the same with `--prediction-rows scored_v1`, on one pod, through the
     existing `topk_adoption_launch.sh`, under a new campaign authorization;
   - stage P now records every candidate's score (`round_tables`);
   - stage C is refused under `scored_v1`;
   - the two records are compared offline.
2. **R1:** build and profile `calib.teacher_native@v1` at $0.
3. **A held-out teacher-native state-eval suite, and a fresh battery family:**
   both are needed before R2 and R3 can be judged.
4. **Housekeeping (not blocking):** the D1 search's out-of-band product copy,
   4.6 GB, holds byte-identical duplicates. It is registry disposition
   `review`. Its `states.jsonl` is the replay journal q2's rematerialization
   reads, so only the duplicate checkpoints are retirement candidates, and only
   through the registry.
