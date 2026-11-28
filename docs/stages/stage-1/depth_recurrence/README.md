# DMR1 variant 2 (revised 2026-11-28): recurrent execution of the kept blocks from both initializations

Stage 1 owns this variant: it is structural-initialization and reasoning-inheritance research, from a
teacher-derived student before any recovery; the training that tests it is Stage 3 recovery (`stage3_recovery`, its
registered recipe and packing). It is DMR1's second variant (review report:
[`../reasoning_inheritance/DMR1_REVIEW.md`](../reasoning_inheritance/DMR1_REVIEW.md), sections 0R and 0S; variant 1: [`../depth_merge/README.md`](../depth_merge/README.md)). Variant 3
(per-iteration hidden-state distillation) is NOT STARTED; its term is kept in section 4 as designed.

**Status: REVISED at $0, READY_FOR_REVIEW (2026-11-28 UTC). Proposed, not granted; no paid run is authorized.**

| file | what it is |
| --- | --- |
| [`configs/stages/stage-1/depth_recurrence/design.json`](../../../../configs/stages/stage-1/depth_recurrence/design.json) | the recurrent-execution identity, the schedule, each arm's fixed base, the engine validation, the state probe, R1-R4, the compute accounting; the 2026-11-26 selection design under `superseded_2026_11_28` |
| [`configs/stages/stage-1/depth_recurrence/recurrent_config.json`](../../../../configs/stages/stage-1/depth_recurrence/recurrent_config.json) | the recurrent model's config (sha256 `3b8c0a8a…`), derived from `base_config.json` (sha256 `191c51bb…`, the base's file) |
| [`logs/stages/stage-1/depth_recurrence/plans/proposal_dmr1_loop_train.json`](../../../../logs/stages/stage-1/depth_recurrence/plans/proposal_dmr1_loop_train.json) | the two sessions' time, money and memory (proposed) |
| [`logs/stages/stage-1/depth_recurrence/plans/storage_plan.json`](../../../../logs/stages/stage-1/depth_recurrence/plans/storage_plan.json) | the relay's live room against the two sessions, under each save-point rotation |
| [`scripts/stages/stage-1/depth_recurrence/`](../../../../scripts/stages/stage-1/depth_recurrence/) | `schedule.py` (schedule, config, each arm's recurrent init record), `engine_check.py`, `launch.py`, `tests/` |
| `src/aadistill/initialization/adapters/recurrent_depth.py`, `recurrent_depth_vllm.py`; `src/aadistill/training/hidden_state.py` | the core: the family (HF and vLLM), the trainer's hidden-state term and its eval-only state probe |

## 1. Question and conditions

Does the alpha merge's single-execution deficit (V1 seed 20260726 below SOB1's same-seed depth_only: thinking −3.86
pp) persist when the same 16 blocks run recurrently on the 36-logical-layer schedule? One training seed (20260726),
four conditions:

| | single execution (16 calls) | recurrent (36 calls) |
| --- | --- | --- |
| D: the kept teacher blocks, unmerged | SOB1 depth_only (sealed, reused) | V2-D `depth_loop_kept` |
| M: V1's alpha-merged blocks | V1 depth_merge (sealed, reused) | V2-M `depth_loop_merged` |

The recurrent conditions differ only in the training-pre initialization, and each starts from it, never from a
recovered final. R1 = V2-D − D, R2 = V2-M − M, R3 = V2-M − V2-D and R4 = (V2-M − V2-D) − (M − D) are descriptive
single-seed estimates with item-paired group-bootstrap intervals and no test. This round cannot by itself separate a
damaged representation from a recipe that suits it less; no mechanism is concluded from it.

## 2. The schedule (from the real mapping)

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| teacher layers (span) | 0 | 1-3 | 4 | 5 | 6 | 7-9 | 10-11 | 12-17 | 18 | 19-21 | 22 | 23 | 24-28 | 29-30 | 31-33 | 34-35 |
| executions | 1 | 3 | 1 | 1 | 1 | 3 | 2 | 6 | 1 | 3 | 1 | 1 | 5 | 2 | 3 | 2 |

16 parameter blocks, 36 logical layers; logical layer p stands where teacher layer p stood. `layer_to_block =
[0, 1,1,1, 2, 3, 4, 5,5,5, 6,6, 7×6, 8, 9,9,9, 10, 11, 12×5, 13,13, 14,14,14, 15,15]`. The core takes the schedule
as data (`schedule_from_spans`); nothing in `src/aadistill` names 16, 36 or a span.

**What an iteration is.** A whole Qwen3 decoder block -- input RMSNorm, attention with q/k RMSNorm and RoPE, residual
add, post-attention RMSNorm, gated FFN, residual add -- with the block's shared parameters. Each iteration's input is
the previous one's output; the same token keeps its position id at every iteration; each logical layer has its own KV
cache entry. Gradients flow through every iteration and sum into the shared parameters.

## 3. The initializations, and how they are bound

Each arm's base is fixed by the design (`arms.<arm>.base`), with no selection:

| arm | base | single-execution control |
| --- | --- | --- |
| `depth_loop_kept` | KEPT: SOB1's depth init (`413686f2…`) | SOB1's `depth_only` arm, its sealed seed-20260726 sets |
| `depth_loop_merged` | MERGED: variant 1's COMPLETE build record (`f5713179…`, bound by its training plan) | variant 1's `depth_merge` seed-20260726 sets |

The recurrent init is the base's weights, byte for byte, with `recurrent_config.json` (both bases share the base
config file). Its identity is weights AND config: `execution_digest(weights, config)` (KEPT `c09176af…`, MERGED
`ccdc8f5e…`). Commands:
- `schedule.py init-record --arm <arm> --write` writes `inits/<KEPT|MERGED>/init_record.json`;
- `plan.py --arm <arm> --init-record <it> --write` writes the arm's plan.

`init_record_problems` refuses anything else: another base than the arm's, a record of another arm, another config,
another schedule, or a digest that is not the execution digest. The 2026-11-26 rule (one base selected from
comparison 1's two seeds) is kept in the design under `superseded_2026_11_28`, and its code in git history.

## 4. Variant 3's hidden-state term (NOT STARTED; kept as designed 2026-11-26)

    L = 0.25 L_CE + 1.0 L_KD + lambda_hidden L_hidden,    lambda_hidden = 1.0

| | |
| --- | --- |
| pairs | every logical layer p with teacher layer p: 36 pairs |
| what is compared | the raw output of the whole block call (residual stream after the block), student and teacher alike, before any final norm; the last logical layer against teacher layer 35's raw output. The final RMSNorm's output reaches the loss only through the head (CE / KD) |
| iterations | every iteration of a looped block, the first included |
| single-execution blocks | included (0, 4, 5, 6, 18, 22, 23): the one-iteration case of the same rule |
| rows | the KD rows (assistant prediction positions): row t is the position whose state predicts token t + 1, the same rows, with no extra offset, whose final states reach the head for KD |
| error | per (pair, row) `||s - h||^2 / ||h||^2`: a per-position **relative** squared error (each term divided by that row's target energy), not ordinary MSE (whose scale would follow the residual norm at each depth and position); dimensionless, the width cancels, not bounded above |
| normalization | the step's sum over pairs and rows / (36 x the step's KD rows) |
| teacher side | the same packed ids, no position ids (cross_context), no gradient; layer outputs by forward hooks in the forward the KD sketch comes from, each layer once; targets only, never the student's input |
| student side | handed over by the model's own logical-layer loop, outside the checkpointed block calls (a recomputation never registers it twice); no detach between iterations |

lambda_hidden = 1.0 is a **proposal**, not tuned and not validated (no search, no eval-set-1 reading); 0.25 is an
equally unvalidated alternative, not a "KD-dominant" setting. Measured at $0 on the dev box's CPU
(`logs/stages/stage-1/depth_recurrence/plans/step0_diagnostic_20261126.json`) is only the scalar size at step 0: on
the KEPT recurrent init the hidden term is 14.56 against KD 12.30. That does not make their gradient contributions
similar (not measured); the global gradient-norm clip rescales the summed gradient and so neither guarantees that KD
dominates nor changes the terms' shares; and it bounds the gradient's norm, not AdamW's per-coordinate update. The
same diagnostic shows the untrained init itself starts degenerate (CE 12.46 against the single-execution base's
3.46 on the same two sequences): evidence about this initialization, not a verdict on recovery training, and no
reason to change the registered schedule, scaling or initialization. Review report, section 5.

## 5. Training, evaluation, compute

SOB1's depth recipe for both arms, compared field by field (`plan.py`): only the names, run name, paths,
`lineage.initialization` and the eval-only `state_probe` change (`stage` stays `stage3_recovery`), and the two arms'
plans differ from each other only in the initialization.
- **Same as the single-execution controls:** the data, proportions, order, 9,701 steps and pre-packing token budget.
  Recurrence deepens the computation per token and adds no training token, and the loop arms are not shortened to
  match FLOPs.
- **Not the same compute:** the comparison is parameter-matched (16 blocks), not FLOPs- or latency-matched. Per
  training token the teacher's 36 layer forwards plus about 4 forward-equivalents per logical layer (100 → 180,
  1.8x); decode work and KV cache 2.25x. Measured GPU time, memory and evaluation time are reported.

Evaluation: SOB1's protocol verbatim, through vLLM 0.27.1 with the recurrent out-of-tree model (one KV cache per
logical layer). **Not measured on a GPU.** The first approved session (V2-D) runs `engine_check.py` before any
training:
- HF cached vs no-cache, and vLLM's greedy generation re-scored by HF;
- each against the same weights as a stock 16-layer Qwen3;
- a shared-cache fault that must fail;
- a FAIL trains nothing.
