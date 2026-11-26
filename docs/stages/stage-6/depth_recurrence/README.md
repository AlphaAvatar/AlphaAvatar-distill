# DMR1 variants 2 and 3: recurrent execution of the kept blocks, and per-iteration hidden-state distillation

Stage 6 owns these variants (AGENTS.md 4.8: shared-parameter repeated execution). They are DMR1's second and third
variants (review report: [`../../stage-1/reasoning_inheritance/DMR1_REVIEW.md`](../../stage-1/reasoning_inheritance/DMR1_REVIEW.md);
variant 1: [`../../stage-1/depth_merge/README.md`](../../stage-1/depth_merge/README.md)).

**Status: PREPARED and REPAIRED at $0, READY_FOR_REVIEW (2026-11-26 UTC; review report section 0). No paid run of either variant is authorized this
round; their sessions are estimated, not proposed.**

| file | what it is |
| --- | --- |
| [`configs/stages/stage-6/depth_recurrence/design.json`](../../../../configs/stages/stage-6/depth_recurrence/design.json) | the Stage-6 identity, the schedule, the initialization selection, the hidden-state term, the round's registered analysis, the compute accounting |
| [`configs/stages/stage-6/depth_recurrence/recurrent_config.json`](../../../../configs/stages/stage-6/depth_recurrence/recurrent_config.json) | the recurrent model's config (sha256 `3b8c0a8a…`), derived from `base_config.json` (sha256 `191c51bb…`, the base's file) |
| [`logs/stages/stage-6/depth_recurrence/plans/estimates.json`](../../../../logs/stages/stage-6/depth_recurrence/plans/estimates.json) | time, money and memory on the 36-layer execution graph |
| [`scripts/stages/stage-6/depth_recurrence/`](../../../../scripts/stages/stage-6/depth_recurrence/) | `schedule.py` (schedule, config, selection, recurrent init record), `launch.py`, `tests/` |
| `src/aadistill/initialization/adapters/recurrent_depth.py`, `recurrent_depth_vllm.py`; `src/aadistill/training/hidden_state.py` | the core: the family (HF and vLLM) and the trainer's hidden-state term |

## 1. Questions

* **DMR1-R (`depth_loop`)**: executing each of SOB1's 16 kept blocks once per teacher layer of its span (36 logical
  layers, shared parameters, each iteration reading the previous one's output), from the same training-pre
  initialization as its single-execution control, does more static fluid reasoning survive the same recovery?
* **DMR1-RH (`depth_loop_hidden`)**: with variant 2's initialization, sharing, schedule, data and optimizer steps, does
  adding a per-iteration hidden-state loss change what survives? It trains from the same training-pre initialization,
  independently: never from variant 2's trained weights.

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

## 3. The initialization, and how it is bound

1. comparison 1 (variant 1 − SOB1 depth, after recovery) completes first;
2. the selection rule picks KEPT (SOB1's depth init) or MERGED (variant 1's init):
   **MERGED iff comparison 1's thinking-mode primary units are complete AND its pooled delta > 0 AND each seed's
   delta > 0; otherwise KEPT.** A tie or seeds that disagree select KEPT (the simpler initialization, whose
   single-execution control is already measured). The p-value plays no part: this is an engineering choice, not a
   statistical claim. An incomplete or invalid comparison 1 selects nothing and the maintainer decides: `complete`
   must be the boolean True, the record must be comparison 1's (its pair, protocol, the 570 items, 558 groups and the
   two registered seeds), and the pooled delta and each registered seed's delta must be finite numbers
   (`schedule.decide`, `comparison_1_problems`);
3. the recurrent init is the selected base's weights, byte for byte, with `recurrent_config.json` (both bases share
   the base config file). Its identity is weights AND config: `execution_digest(weights, config)`.

| path | base | control of comparison 2 |
| --- | --- | --- |
| KEPT | SOB1's depth init (`413686f2…`) | SOB1's `depth_only` arm, its sealed sets |
| MERGED | variant 1's COMPLETE build record (bound by its training plan) | variant 1's `depth_merge` arm |

The selection is written once and never rewritten (`select --write` refuses another content); it names comparison 1's
record by path and sha256 and stands only while that record re-derives the same choice and deltas
(`schedule.selection_problems`). Comparisons 2 and 3 read the initialization a trained unit's committed plan bound
(`analysis.bound_init`: plan → recurrent init record → selection), never the current selection file.

**What C2 then measures.** Both arms start from the initialization comparison 1 chose on the same development-exposed
eval-set-1 items C2 is scored on, so C2 is the recurrence's effect at that selected initialization, conditional on the
selection -- not its effect at an initialization chosen independently of this set. Each comparison is its own family
at alpha 0.05; no error rate is controlled or claimed over the round's three comparisons together (the Holm line over
the three is a sensitivity report only).

`schedule.py select --write` → `plans/init_selection.json`; `schedule.py init-record --write` →
`inits/<path>/init_record.json`; `plan.py --arm depth_loop|depth_loop_hidden --init-record <it> --write`.
`init_record_problems` refuses anything else: another base, a selection that names another path, another config,
another schedule, a digest that is not the execution digest. Until comparison 1 exists the reference stays unbound,
and the launchers refuse (no plan, no proposal, no grant).

## 4. Variant 3's hidden-state term

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
(`logs/stages/stage-6/depth_recurrence/plans/step0_diagnostic_20261126.json`) is only the scalar size at step 0: on
the KEPT recurrent init the hidden term is 14.56 against KD 12.30. That does not make their gradient contributions
similar (not measured); the global gradient-norm clip rescales the summed gradient and so neither guarantees that KD
dominates nor changes the terms' shares; and it bounds the gradient's norm, not AdamW's per-coordinate update. The
same diagnostic shows the untrained init itself starts degenerate (CE 12.46 against the single-execution base's
3.46 on the same two sequences): evidence about this initialization, not a verdict on recovery training, and no
reason to change the registered schedule, scaling or initialization. Review report, section 5.

## 5. Training, evaluation, compute

SOB1's depth recipe for both arms, compared field by field (`plan.py`): only the names, run name, paths,
`lineage.initialization`, `stage` (`stage6_recurrent_depth`) and, for variant 3, `loss.hidden_state` change. The same
data, proportions, order, 9,701 steps and pre-packing token budget as the single-execution control: recurrence
deepens the computation per token and adds no training token; the loop arms are not shortened to match FLOPs.
Parameter-matched (16 blocks), not FLOPs- or latency-matched: per training token the teacher's 36 layer forwards plus
about 4 forward-equivalents per logical layer (100 → 180, 1.8x); decode work and KV cache 2.25x. Measured time and
memory are reported apart.

Evaluation: SOB1's protocol verbatim, through vLLM 0.27.1 with the recurrent out-of-tree model (one KV cache per
logical layer). **Not measured on a GPU**; its agreement with the HF reference is the first thing a GPU must show
(review report section 8).
