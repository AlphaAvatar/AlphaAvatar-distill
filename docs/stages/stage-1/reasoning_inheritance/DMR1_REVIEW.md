# DMR1: depth recovery at SOB1's depth-only structure -- review report (READY_FOR_REVIEW)

**Terminal state of this round: READY_FOR_REVIEW** (2026-11-26 UTC). The designs, the code of all three variants,
the $0 verification and this report are committed and pushed on `research/reasoning-inheritance` (not merged into
`main`). **This is not a scientific COMPLETE:** no initialization was built, nothing trained or generated, no pod or
volume created, no grant written, no artifact issued. Variant 1 starts only after this review and the maintainer's
explicit authorization; variants 2 and 3 have no paid authorization this round.

Contents: 1 scope and entry points · 2 baseline, teacher, mapping, schedule · 3 variant 1 · 4 variant 2's initialization
and binding · 5 variant 3's hidden-state term · 6 comparisons and statistics · 7 the recurrent mechanism · 8 tests,
fixes, what is not measured · 9 time, memory, storage, money · 10 $0 state and authorization boundaries.

## 1. Scope, files and real entry points

The round **DMR1** (reasoning-inheritance research) has three variants:

| variant | arm | stage | design | question |
| --- | --- | --- | --- | --- |
| 1, DMR1-M | `depth_merge` | Stage 1 init, `stage3_recovery` training | [`configs/stages/stage-1/depth_merge/design.json`](../../../../configs/stages/stage-1/depth_merge/design.json) | alpha-merged init vs direct deletion, same recovery |
| 2, DMR1-R | `depth_loop` | Stage 6 (`stage6_recurrent_depth`) | [`configs/stages/stage-6/depth_recurrence/design.json`](../../../../configs/stages/stage-6/depth_recurrence/design.json) | 16 shared blocks on 36 logical layers vs single execution, same init |
| 3, DMR1-RH | `depth_loop_hidden` | Stage 6 | the same | variant 2 + per-iteration hidden-state loss |

Design documents: [`docs/stages/stage-1/depth_merge/README.md`](../depth_merge/README.md),
[`docs/stages/stage-6/depth_recurrence/README.md`](../../stage-6/depth_recurrence/README.md).

**Changed, by area** (commit message lists every file):

* core (`src/aadistill`, declared in `tests/architecture/test_cuda_surface_preserved.py` as round `4a76fde2`):
  `initialization/adapters/recurrent_depth.py` (new: the family, HF), `initialization/adapters/recurrent_depth_vllm.py`
  (new: the family in vLLM), `training/hidden_state.py` (new), `training/train.py` (`loss.hidden_state`),
  `initialization/adapters/families.py` (one registry line). No Qwen3-4B, 16, 36, span, path or experiment name in
  any of them.
* the round's chain: `scripts/stages/stage-1/depth_merge/` (`common`, `merge`, `build_driver`, `plan`,
  `train_driver`, `session`, `launch`, `analysis`, `proposal`, tests) and `scripts/stages/stage-6/depth_recurrence/`
  (`schedule`, `launch`, tests);
* reused chain, generalized minimally and backwards-compatibly: SOB1's `common.bind` (point SOB1's phase-2 chain at
  another design with SOB1's schema), `train_driver.run_unit(stage_init=)`, `launch.relay_state / reclaim_superseded
  (plan=, record=)`, `retain` for any training kind; the pod setup's two session-kind branches; the vLLM plugin's second
  entry point; the checkpoint registry's declared consumers; the stage index (Stage 6 and the two experiments).

**Real entry points** (`PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python <file>`):

| | entry point |
| --- | --- |
| variant 1 build (pod) | `scripts/stages/stage-1/depth_merge/build_driver.py` (session kind `dmr1_merge_build`) |
| every variant's training + evaluation (pod) | `scripts/stages/stage-1/depth_merge/train_driver.py` (`dmr1_merge_train`, `dmr1_loop_train`, `dmr1_loop_hidden_train`) |
| launchers (dev box) | `scripts/stages/stage-1/depth_merge/launch.py` (variant 1), `scripts/stages/stage-6/depth_recurrence/launch.py` (variants 2, 3) |
| binding (dev box, $0) | `scripts/stages/stage-1/depth_merge/plan.py`; `scripts/stages/stage-6/depth_recurrence/schedule.py {schedule,config,select,init-record}` |
| analysis (dev box, $0) | `scripts/stages/stage-1/depth_merge/analysis.py comparison --id C1 / C2 / C3`, `sensitivity` |
| trainer | `scripts/shared/training/train_stage3.py` (unchanged; it loads the recurrent family through the Auto classes) |

## 2. Fixed baseline, teacher, mapping and schedule

Checked against SOB1's sealed records (sha256 pinned in the variant-1 design; `test_depth_merge.py` re-hashes them):

* **Teacher**: the original hybrid `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` (root sha256
  `a8b9dec3…`), NOT Thinking-2507. It is the calibration source, the initialization's parent, the online KD teacher and
  the hidden-state target; tokenizer, chat template, training pack and config all from it (P17.1).
* **Baseline**: SOB1's depth-only **training-pre** initialization (36 → 16 layers, depth only): `init_record.json`
  `413686f2…`, weights `76c8ca19…`, config file `191c51bb…`, 2,003,851,776 parameters, mapping `36888440…`, width
  2,560, FFN 9,728, 32 / 8 heads, head_dim 128, tied embedding. Its depth selection is reused; no search runs again.
* **Kept blocks** (SOB1's DEPTH, removal order `[2, 32, 3, 26, 30, 8, 20, 28, 27, 9, 16, 35, 15, 21, 14, 25, 11, 17,
  33, 13]`): `[0, 1, 4, 5, 6, 7, 10, 12, 18, 19, 22, 23, 24, 29, 31, 34]`.
* **Control**: SOB1's two recovered depth seeds (`sob1_train_depth_only_20261122_223829`, finals `5635c33e…` /
  `e4159f1f…`) and their sealed sets, reused, never retrained or regenerated; their recovery plan
  `training_plan_depth_only.json` (`4eeab53f…`) is what every DMR1 unit is derived from.
* **Spans and schedule** (from the mapping; `merge.py spans`, `schedule.py schedule`):

  | block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | span (teacher layers) | 0 | 1-3 | 4 | 5 | 6 | 7-9 | 10-11 | 12-17 | 18 | 19-21 | 22 | 23 | 24-28 | 29-30 | 31-33 | 34-35 |
  | executions in variants 2, 3 | 1 | 3 | 1 | 1 | 1 | 3 | 2 | 6 | 1 | 3 | 1 | 1 | 5 | 2 | 3 | 2 |

  16 parameter blocks, 36 logical layers (`layer_to_block` in `recurrent_config.json`, sha256 `3b8c0a8a…`); logical
  layer p stands where teacher layer p stood.

## 3. Variant 1: the alpha merge

**Definition** (the brief's): per slot, `theta~_j = sum_{i in S_j} alpha_{j,i} A_{j<-i}(theta_i)`, `alpha_{j,i} =
I_{j,i} / sum_k I_{j,k}`, S_j = kept block j and the removed blocks after it. It is RI-LM1's M-act, unchanged, at this
structure's 9 merged spans (`[1-3]`, `[7-9]`, `[10-11]`, `[12-17]`, `[19-21]`, `[24-28]`, `[29-30]`, `[31-33]`,
`[34-35]`); the 7 single spans stay the base's bit for bit.

| slot | importance I | alignment A |
| --- | --- | --- |
| FFN neuron (gate row, up row, down column) | `E|a| x ||down column||_2` | exact linear assignment on the gated-product Pearson correlation |
| query head (q rows, o_proj columns) | write energy `mean_t ||W_o,h a_h||^2` | within the matched KV group, on the heads' write cosine |
| KV group (k / v rows) | its 4 heads' summed energy | assignment on the groups' summed-write cosine |
| q_norm, k_norm, folded gains | the block's share of the span's head energy | -- |

RMSNorm: input / post-attention gains folded into q/k/v and gate/up input columns, merged gains 1. All-zero slot:
uniform `1/|S_j|`, counted. Each merged block executes once (16 logical layers); alpha only builds the initial weights.
Not registered: beta addition, learnable coefficients, fitting. Convex weights guarantee neither the output scale nor
the span's cumulative update; both are diagnostics.

**Implementation**: `merge.py` calls RI-LM1's `collect_statistics`, `alignments` (one pair at a time),
`statistics_tensors`, `merged_tensors('M-act')`, `build_candidate`, `equivalence_checks` and `diagnostics` with THIS
design's spans, kept blocks and the base's 178 tensor digests (`depth_init_tensor_digests.json`, from the dev box's
copy). The build records RI-LM1's code identities (ast digests).

**Reused vs recomputed.**

| reused (by identity) | recomputed at this structure |
| --- | --- |
| the M-act code; the calibration `calib.hybrid_d4_base@v2` at `ffn.activation_importance_v0`'s positions under the packed search's session (inputs v3); SOB1's selection and base (rebuilt from the teacher by `keep_blocks`, every tensor checked against its digests); SOB1's recipe, plan and control sets | the statistics of the 29 span blocks (20 pairs), every permutation, every merged tensor |

RI-LM1's statistics and deltas are not inputs (other kept set; alpha normalizes over S_j). Its pairs (10,11),
(24,25), (24,26), (29,30), (31,32) recur here and may be compared as a reproduction diagnostic only
(`merge.statistics_reproduction`).

**Gates verified at $0** (`logs/stages/stage-1/depth_merge/plans/preparation_checks_20261126.json`): the session
derives the journal's operator config hashes; the calibration gives exactly **558 examples, 79 packed sequences, 231,242
statistics positions** (the build's gate, counted on a meta-device teacher without a forward).

**Durability and takeover.** Each pair's permutations are written the moment they exist with the sha256 of their
correlation; a later attempt recomputes the one-pass statistics (about 5 min) and reuses a pair only under the same
correlation. The init and statistics go to the relay, re-downloaded and re-hashed; `init_record.json` is COMPLETE
only when every copy verified; verified copies are never sent again.

## 4. Variant 2: initialization selection and binding

**Order**: comparison 1 first; then the selection; then the recurrent init from the selected TRAINING-PRE init.

**Rule** (registered; `schedule.decide`): **MERGED iff comparison 1's thinking-mode primary units are complete AND
the pooled delta (merge − deletion) > 0 AND each seed's delta > 0; otherwise KEPT.**

* Tie (pooled 0) or seeds that disagree in sign → KEPT: no merge step, and its single-execution control (SOB1's depth
  arm) is already measured.
* Comparison 1 incomplete → no selection; the binding refuses and the maintainer decides.
* The p-value plays no part. This is an engineering choice of a starting point; comparison 1's formal verdict is a
  separate statistical statement (section 6), and a MERGED selection claims nothing about it.

| path | base | comparison 2's control |
| --- | --- | --- |
| KEPT | SOB1's depth init (`413686f2…`) | SOB1's `depth_only` (sealed sets reused) |
| MERGED | variant 1's COMPLETE build record, as its committed training plan binds it | variant 1's `depth_merge` |

**Binding mechanism**: `schedule.py select --write` (from `results/comparison_1.json`) → `plans/init_selection.json`;
`schedule.py init-record --write` → `inits/<path>/init_record.json`: the base's weights and relay copy byte for byte,
`recurrent_config.json`, the schedule, the selection, identity `execution_digest(weights sha256, config sha256)` (not
the base's digest: the same weights on another schedule are another model). `init_record_problems` (asked by
`plan.py` before any recurrent unit is derived) refuses any base but the path's, a selection naming another path,
another config, another schedule, a digest that is not the execution digest, an unverified relay copy. Today the
reference is unbound: no comparison 1 exists, so no selection, plan, proposal or grant exists for variants 2 and 3,
and their launcher refuses.

## 5. Variant 3: the hidden-state term (proposal for review)

    L = 0.25 L_CE + 1.0 L_KD + lambda_hidden L_hidden,   lambda_hidden = 1.0   (design.json :: hidden_state)

| field | registered value |
| --- | --- |
| boundary | the raw output of the whole block call (after both residual adds), student logical layer p against teacher layer p's decoder-block output; both before any final norm. The last logical layer is compared with teacher layer 35's raw output; the final RMSNorm output reaches the loss only through the head |
| iterations | every iteration of a looped block, the first included (block j's k-th call against teacher layer j + k − 1) |
| single-execution blocks | included (0, 4, 5, 6, 18, 22, 23): a block run once is the one-iteration case; excluding them would treat calls identical in structure to every loop's first iteration differently |
| pairs | `[[p, p] for p in 0..35]`: 36 |
| rows / mask | the KD rows: assistant prediction positions (`kd_scope` 'assistant'); row t is the position whose state predicts token t + 1, i.e. the same rows, with no further offset, whose final states reach the head for KD. Prompt and padding positions carry no hidden term |
| error | per (pair, row) `||s − h||^2 / ||h||^2` over the full 2,560-d state (target energy; dimensionless, the width cancels, not bounded above); accumulated in float32 from the fp32 student state and the bf16 target |
| normalization | step sum / (36 pairs × the step's KD rows): a mean over layers, iterations and rows, the same row count that normalizes KD |
| inputs | the same packed ids, no position ids (cross_context), standard causal attention, both models; the teacher under no_grad, its states targets only; the student always runs on its own states, no detach between iterations |
| lambda_hidden | 1.0 (below) |

**Why the per-row relative error rather than raw MSE.** Raw MSE would weight each layer and row by the residual
stream's norm, which grows with depth and is dominated at a few positions by very large activations; the per-row
denominator makes every (layer, row) count once. It is dimensionless but **not bounded above**: a student state larger
than the teacher's gives values above 1.

**Measured at $0, step 0** (`logs/stages/stage-6/depth_recurrence/plans/step0_diagnostic_20261126.json`; the real
trainer's validation path on the dev box's CPU in bf16, the teacher's shards re-hashed to its identity; the first two
validation sequences of the pack truncated to 2,048 tokens, 2,108 KD rows; diagnostic only, eval-set-1 not read):

| initialization | logical layers | val CE | val KD | L_hidden (mean over 36 pairs) |
| --- | ---: | ---: | ---: | ---: |
| SOB1's depth init, single execution | 16 | 3.46 | 3.32 | -- |
| the same weights on the 36-layer schedule (KEPT path, untrained) | 36 | **12.46** | **12.30** | 14.56 |

Per logical layer the relative error is 0 at layers 0-1 (the teacher's own blocks on its own input), below 0.5 up to
layer 8, 1.1-1.6 at 9-13, and then explodes inside block 7's six iterations (teacher layers 12-17): 3.3, 10.1, 35.1,
83.2 at layers 14-17, the student's state about 9x the teacher's norm, decaying through the later blocks (74.5 at 18,
16.0 at 24, 2.8 at 34, 7.4 at 35). On the same weights, the recurrent logits equal the unrolled tied-weight Qwen3's
exactly.

* **What this says for variant 2**: executing the kept blocks on the schedule without training starts far from the
  teacher -- CE 12.46 against the single-execution base's 3.46 on the same rows, above the uniform distribution's
  ln(151,936) = 11.93. Recovery has to undo that before it gains anything from the added depth; SOB1's width arm
  recovered from 9.78 to 0.50, so a high start is not by itself fatal, but this is a materially worse starting point
  than variant 2's control, and the comparison measures it together with recurrence. The registered design is
  unchanged; this is flagged for the review (question R1 below).
* **lambda_hidden = 1.0** (proposed, not tuned): at step 0 the hidden term is about KD's size (14.56 against 12.30,
  1.2x), its mass in logical layers 14-28; the gradient clip at 1.0 bounds each update; KD fell from 3.54 to 0.41 over
  SOB1's depth recovery, and the per-layer log shows whether the hidden term falls with it. The alternative a reviewer
  may prefer is a KD-led weight such as 0.25; it is not registered.

**Questions for the review.** R1: should variant 2's registered initialization stay the plain schedule of the
selected weights (the brief's definition), given its step-0 distance? R2: lambda_hidden 1.0 or a KD-led 0.25? R3: the
relay room (section 9).

**Implementation** (`aadistill.training.hidden_state`, `train.py`): the teacher's layer outputs at the KD rows are
taken by forward hooks in the same no-grad forward the KD sketch comes from (each hooked layer must run once); the
student's are handed over by `RecurrentDepthModel.forward(hidden_supervision=)` from its own logical-layer loop,
outside the checkpointed block calls, so recomputation never registers the loss twice. A custom autograd function keeps
a reference to the student state (which the next checkpoint holds anyway), the bf16 targets and their inverse
energies, and builds the `[T, 2560]` gradient only in the backward. Without `loss.hidden_state` the path is the
unchanged one (tested). Logged per step: `hidden`, `hidden_weight`, `hidden_by_layer`; per validation: `val_hidden`,
`val_hidden_by_layer`; the run manifest's `hidden_state_objective`.

## 6. The three comparisons and the statistics plan (registered before any result)

| id | A − B | question | control's records |
| --- | --- | --- | --- |
| C1 | `depth_merge` − `depth_only` | alpha-merged init vs deletion, same structure and recovery | SOB1's sealed sets, reused |
| C2 | `depth_loop` − the selected path's single-execution arm | recurrence vs single execution, same init | KEPT: SOB1's; MERGED: variant 1's |
| C3 | `depth_loop_hidden` − `depth_loop` | adding the hidden-state term to the same recurrent run | variant 2's |

* **Primary** (each comparison): thinking mode, eval-set-1's static fluid formal core, 570 items in 558 independence
  groups, both seeds pooled (1,140 paired records), each record paired with the other arm's record of the SAME
  training seed and item (the seed fixes the data order in both arms).
* **Statistic, test, interval**: SOB1's / TPI1's: delta = mean paired difference; two-sided group-level sign-flip test
  (B = 100,000, rng 20261119); 95% group-bootstrap interval (10,000, rng 20261120), reported beside the test.
* **Family**: each comparison is its own family at alpha 0.05: thinking at 0.05; **non-thinking (key secondary) at 0.05
  only if thinking rejects** (fixed sequence); otherwise non-thinking is descriptive. Verdicts A_BETTER / B_BETTER /
  NOT_DISTINGUISHED (not equivalence).
* **Across comparisons, no adjustment** of the per-comparison verdicts, and SOB1's m = 3 is not copied: SOB1 compared
  three arms pairwise (one family of exchangeable contrasts); here three different interventions are each tested
  against their own control, decided when their own records complete, and no claim pools or chains them. As a
  sensitivity line only, once all three exist: Holm (m = 3) over the three thinking-mode p-values
  (`analysis.py sensitivity`).
* **Completeness**: formal in a mode only when both arms × both seeds cover all 570 items AND both sessions are
  complete on their files (`completion.session`); otherwise deltas and intervals only, never partially pooled.
* **Seeds vs items**: the test and the interval resample items and groups with each arm's two checkpoints fixed --
  item sampling, not training randomness. Per-seed deltas with intervals and sign agreement are reported beside; two
  seeds are limited evidence about training randomness; 1,140 records are 570 items, not 1,140 independent ones.
* **Secondary, descriptive**: per reasoning axis; interactive episodes; every supporting unit's registered headline;
  terminations (natural, degeneration stop, context-limit hit right-censored) and usable rollout; mean generated
  tokens; validation CE / KD; variant 3's per-layer hidden error; compute (FLOPs per token, s/step, peak memory,
  evaluation time). CE, KL and hidden errors are diagnostics: they never eliminate a candidate and are never reported
  as reasoning gains. eval-set-1 is development-exposed.

## 7. The recurrent mechanism

* **Independent blocks vs logical layers** (`RecurrentDepthConfig`): `num_hidden_layers` = logical layers (36),
  `num_parameter_blocks` = 16, `layer_to_block`. The model holds 16 `Qwen3DecoderLayer` modules; repetition is in
  the call schedule, never in module aliasing, so each parameter is registered once (one AdamW state) and autograd sums
  every call's gradient into it.
* **Save / load**: the checkpoint stores the 16 blocks under `model.layers.0..15` -- the names of SOB1's 16-layer
  checkpoint -- and the schedule in `config.json`; a reload rebuilds the same model. The recurrent init's weight file
  is the base's, byte for byte; its identity includes the config.
* **KV cache**: one entry per logical layer. HF: `DynamicCache(config)` sized by the 36 logical layers; logical layer p
  receives `LogicalLayerCache(cache, p)`, whose `update` writes entry p whatever `layer_idx` the shared attention
  passes (the module is never re-pointed). vLLM: the 16 blocks own no `Attention`; the model owns one `Attention` per
  logical layer (prefix `model.attn_layers.<p>`, one integer, so `bind_kv_cache` orders 36 caches). A token's
  position id is the same at every logical layer.
* **Execution context** is passed with the call, so gradient checkpointing's recomputation sees the same context.
* **Training, saving, resume and generation run the same schedule**: the trainer loads the family through the Auto
  classes (`register_custom_model_families`); save points and finals carry the recurrent config; the evaluation
  runners load it through the vLLM plugin, which the training driver puts on their `PYTHONPATH`.
* **Identity**: model and recovery identity cover the weights, the init method, the schedule (config), the hidden-state
  mapping and loss (in the training config's hash and the run manifest), the teacher, the data, the packing and the
  protocol (`plan.py` compares every other field with SOB1's depth unit).

## 8. Verification, repairs, and what is not measured

**Evidence, by kind** (one convergence validation after the implementation, then one full core suite; the two small
early checks were API questions about transformers 5.13's Qwen3 internals):

```text
convergence, core (tests/docs, tests/architecture, tests/models, tests/training,
  tests/initialization/test_layer_merge.py)               : 678 passed / 0 failed, 2.5 min
convergence, experiment suites (depth_merge 11, depth_recurrence 12,
  single_operator_budget and reasoning_inheritance 69)     : 92 passed / 0 failed, 5.6 min
full core suite (pytest, testpaths = tests)                 : 4,187 passed / 0 failed / 50 skipped, 14.4 min
DMR1 suites again after the last fixes (items 4, 6 below)   : 23 passed / 0 failed, 1.1 min
GPU validation                                              : none (no GPU; nothing paid)
historical suites                                           : SOB1's and RI's ran, because this round changed or
                                                              reuses their code; no other historical suite
```

**What the tests establish** (all on CPU, tiny real Qwen3 geometries unless noted):

| requirement | test |
| --- | --- |
| alpha merge = sum_i alpha_i A_i(theta_i) per slot, alpha normalized over the span, gains folded | `depth_merge/tests :: test_each_merged_slot_is_the_importance_normalized_sum_of_the_aligned_blocks` (a 4-block span, recomputed from the definitions) |
| all-zero slot uniform and counted | `test_an_all_zero_slot_takes_uniform_weights_and_is_counted` |
| single spans and every other tensor the base's; a wrong base refused | the same, and `test_the_base_check_refuses_a_rebuilt_base_that_is_not_the_bases` |
| alignments durable per pair, reused only under the same correlation | `test_alignments_are_reused_only_under_the_same_correlation` |
| build products durable only when verified, never sent twice; build closeout | `test_the_init_record_is_complete_only_when_every_relay_copy_verifies_and_nothing_is_sent_twice` |
| recurrent forward = unrolled Qwen3 whose layers ARE the blocks, bit for bit | `tests/models/test_recurrent_depth.py :: test_the_forward_is_the_unrolled_tied_qwen3_bit_for_bit`; on the REAL weights too (CPU, `step0_diagnostic`) |
| parameters registered once; each shared gradient = the sum over its calls | `test_parameters_are_registered_once_and_each_gradient_sums_its_calls` (against independent copies) |
| cached generation = no-cache forward; a cache entry per logical layer; positions fixed across iterations; the shared module never mutated | `test_cached_decoding_equals_the_no_cache_forward_and_each_logical_layer_has_its_own_entry`, `test_positions_do_not_advance_between_logical_layers`, `test_the_logical_layer_cache_never_mutates_the_shared_module` |
| gradient checkpointing on / off | `test_gradient_checkpointing_changes_neither_the_loss_nor_the_gradients` (model) and `test_gradient_checkpointing_gives_the_same_hidden_loss_and_gradients` (trainer) |
| save / load keeps sharing and schedule; a Qwen3 checkpoint becomes the recurrent model; the identity schedule is Qwen3 | `test_save_and_load_keep_the_schedule_the_sharing_and_the_logits`, `test_a_qwen3_checkpoint_becomes_the_recurrent_model_over_the_same_weights` |
| hidden-state error and its backward; rows = KD rows; teacher states once, no graph; each logical layer handed over once under checkpointing; teacher gets no gradient, every block does; absent key = unchanged path; refusals | `tests/training/test_hidden_state.py` (10) |
| resume of a recurrent + hidden run continues exactly; save points reload on the schedule | `depth_recurrence/tests :: test_a_resumed_recurrent_hidden_run_continues_exactly_and_saves_reload_on_the_schedule` (the real `Trainer`) |
| selection rule; KEPT and MERGED the only legal bases; refusals | `test_the_selection_rule_prefers_kept_unless_both_seeds_and_the_pool_favour_merged`, `test_the_kept_paths_...`, `test_the_merged_paths_...`, `test_a_selection_naming_the_other_paths_base_is_refused` |
| plans = SOB1's depth unit outside the declared fields; variant 3 differs from 2 only by the hidden term | `test_the_merge_arms_plan_...`, `test_the_loop_arms_plans_differ_from_sob1s_depth_unit_only_where_declared` |
| the three arms through the REAL driver entry (`train_driver.main`) to a complete session, the closeout's completion COMPLETE, a takeover that trains and generates nothing; the recurrent config in place when training starts; the plugin only for the loop arms | `test_each_arm_runs_through_the_real_driver_to_a_complete_session_and_a_takeover_does_nothing` x 3 (SOB1's stand-ins for the trainer, the engine, the package and the relay transport; the state machine, completion, binding and takeover are real) |
| the comparison's family rule | `test_each_comparison_is_its_own_fixed_sequence_family_and_formal_only_when_complete` |
| core declared, nothing else moved | `tests/architecture/test_cuda_surface_preserved.py` (81) |

**Fixed during the round**:

1. the hidden error was accumulated in float32 even for a float64 state (a 4e-8 relative difference in the float64
   test): it now accumulates in float64 for float64 states (training is fp32 student / bf16 targets, unchanged);
2. the recurrent init record bound its config by a root-relative path, wrong under any other root: now repo-relative;
3. my first lambda rationale claimed the relative error is bounded at 1 -- wrong, the step-0 measurement shows 14.6
   (section 5); the design and docs are corrected;
4. the training driver put the vLLM plugin on every arm's runner path: a stock-Qwen3 arm (variant 1) would have
   imported the recurrent module into vLLM's processes for nothing; it is now only for the loop arms (tested);
5. the renderer escaped one em dash in `logs/state/current.json` (a known quirk); restored;
6. a launch spec for a kind with neither grant nor proposal failed with a raw `FileNotFoundError`; it is now an
   `AuthorizationError` naming the missing proposal (tested).

**Not measured (CUDA or real scale), each owed before or at the first paid run:**

* the recurrent HF model on a GPU (bf16 autocast, fp32 weights) and its L40S memory and step time;
* the vLLM recurrent model: never imported in a vLLM environment (vLLM 0.27.1 is not on the dev box; its source was
  read from the wheel); unverified: the plugin's import under the vLLM env's transformers 5.15, torch.compile and CUDA
  graphs over 36 per-logical-layer `Attention` modules, the KV-cache sizing, agreement with the HF reference;
* the hidden-state term's memory and time at 8,192 tokens;
* variant 1's build on the 4B teacher: the 15.1 GB of Pearson moments and the 20 assignments' time on the pod's CPU.

## 9. Time, memory, storage and money

Bases: SOB1's measured depth session (4.83 s/step; 805-806 min of training per seed; 34.5-35.5 / 6.9-7.2 min of
evaluation per seed and mode; $31.6179 all-in); RI-LM1's measured statistics pass (117 s for 14 blocks) and
assignments (178 s each on the pod's CPU); SOB1's bounds (7.241 s/step, 3.0 h evaluation per unit); one L40S at $1.09/h
+ 150 GB container disk ($1.1105/h all-in), price cap $1.11/h. The loop arms are scaled on the work a step does
(teacher 36 + student ~4 per logical layer: 100 → 180, 1.8x), evaluation by the decode work (2.25x), the hidden
term +5%. **Estimates, not measurements**: the first four optimizer steps of each first unit record the real time and
memory.

| session | per seed training h, expected (bound) | eval h per unit | driver h, expected (bound) | expected $ | hard cap $ |
| --- | ---: | ---: | ---: | ---: | ---: |
| variant 1 build (`dmr1_merge_build`) | -- | -- | 1.59 (3.73) | 2.50 | 5.43 |
| variant 1 training (`dmr1_merge_train`) | 13.0 (19.5) | 1.22 (3.0) | 28.8 (53.5) | 32.68 | 64.46 |
| variant 2 (`dmr1_loop_train`), estimate | 23.4 (35.1) | 2.75 (6.75) | 52.6 (92.2) | 59.20 | 110.43 |
| variant 3 (`dmr1_loop_hidden_train`), estimate | 24.6 (36.9) | 2.75 (6.75) | 55.0 (95.7) | 61.79 | 114.60 |

Build detail (expected minutes): session and contract 3, teacher and base check 3, statistics 4, **20 assignments 59**,
merge (float64, CPU) 8, checks 1, save / reload 2, state evaluation 10, diagnostics 3, relay 2. Failure takeover: a
lost build re-does the statistics (~5 min) and only the unpersisted pairs; a lost training session resumes from the
latest relay save point (at most 1,104 steps lost: 1.5 h for variant 1, about 2.7 h for the loops) plus setup and a
save-point restore (20.9 GB). The caps cover their bounds once; a replacement attempt's cap is what its kind's
envelope leaves.

**Memory (one L40S, 44.39 GiB)**: variant 1 is SOB1's depth structure and recipe (measured 39.38-39.39 GiB peak
allocated, 40.90-40.98 reserved). Variant 2 adds 20 checkpointed block inputs ([8,192, 2,560] fp32): 1.56 GiB →
~40.95 GiB allocated, ~42.5 reserved. Variant 3 adds the bf16 targets at the KD rows (36 layers: 0.31 GiB at the
median 1,790 rows, 1.41 GiB at a full sequence) and one 0.08 GiB gradient buffer → 41.3-42.4 allocated, up to ~44.0
reserved: within ~1 GiB of the device, the recipe's OOM steps first (expandable segments, `kd_chunk`, AdamW without
multi-tensor temporaries), an 80 GB card only after a measured OOM. Generation: the loop's KV cache is the teacher's
size per token (36 layers, 147,456 B in bf16), 2.25x the depth arm's. Not measured.

**Storage.**

| | size | where |
| --- | ---: | --- |
| init (bf16) | 4.01 GB | relay, the variant's prefix (variant 1's build; the loops reuse their base's relay copy, no new init bytes) |
| statistics (variant 1) | ~0.5 MB | relay and evidence |
| save point (fp32 + AdamW + state) | 20.93 GB | relay, rotated: at most two at once |
| final (fp32) | 8.02 GB per seed | relay (`design.json :: storage.phase_2_finals.placement`) |
| evidence | ~0.35 GB per training session | dev box (`<scr>/store/out`) and the relay archive |
| a training session's relay peak | **46.46 GiB** (two save points + one final) | -- |

**The relay is short.** Its live headroom read **35.24-35.28 GiB** (2026-11-26 11:32Z, read-only bisection). Every
training session (each of the three variants) needs 46.46 GiB at its peak: **11.2 GiB short (12.2 with the 1 GiB
margin)**, and each finished variant's two finals then hold 14.93 GiB more. The build session needs 3.8 GiB and fits.
The dev box has 18-23 GiB free (shared with another project; a 4.4 GiB swing was measured). With no disk growth and no
storage purchase (the brief), the training session's launch gate refuses until the maintainer frees room. Options, the
maintainer's to choose:

1. retire the 7.75 GiB of session code bundles on the relay (reproducible from git) plus about 4.5 GiB more (e.g. the
   1.11 GiB random baseline, and a further decision) -- each a maintainer decision (SOB1 README section 8);
2. move SOB1's two FFN + attention finals (15.03 GiB) from the relay to the dev box -- leaves the dev box ~5-8 GiB free,
   too thin beside the 4.4 GiB swing;
3. approve a one-save-point rotation (remove the previous save point before the next upload): peak 28.95 GB, fits,
   but a failure in that window falls back to the previous final or step 0 -- a change of the durability rule.

**Money** (live 2026-11-26 11:48Z, read-only):

* account balance **$123.7283**; 0 pods, 0 volumes, $0.00/h;
* project cap $1,161.06, committed $1,086.2713 (provider-settled), **$74.7887** left under it;
* SOB1's closed grants' remainders ($73.8079 and $1.00) are void and fund nothing here;
* **variant 1's proposal** (two kinds, each its own grant): build $2.50 expected / **$5.43** cap, training $32.68
  expected / **$64.46** cap, $1.00 teardown reserve each: **$71.89** with reserves, inside the $74.79 the cap leaves and
  the balance. `launch.py budget` reads both fundable from their proposals at that position;
* variants 2 and 3 (not proposed): about $121 expected, $225.03 of caps + $2 reserves -- they need a project-cap
  decision, a grant per kind, the relay room above, and the engine validation below first.

**Proposed GPU validation before variant 2** (not authorized): one L40S, about 1 h, about $1.2 (cap $2.5): load the
KEPT recurrent init in vLLM 0.27.1 with the plugin; compare its prompt log-probabilities and greedy continuations
with the HF reference on a few hundred eval prompts (the engine-admission rule: |vLLM − HF bf16| within 2x the HF
bf16-vs-fp32 difference); run the hidden-state trainer's first four steps on 8,192-token sequences to measure memory.

## 10. $0 state and authorization boundaries

* Nothing billed this round: no pod, no volume, no grant, no artifact, no upload, no relay deletion. Read-only
  queries: the provider position, the relay's headroom, the vLLM 0.27.1 wheel (for its source) and the public teacher
  weights (for the step-0 diagnostic; deleted afterwards).
* Retention: SOB1's depth init and its two finals are registered as DMR1's declared consumers
  (`build_checkpoint_registry.py`); the three SOB1 initializations stay; nothing retired.
* Owed to the maintainer: this review; variant 1's two grants and its relay room (section 9); for variants 2 and 3, a
  project-cap decision, their proposals and grants after the selection, the engine validation.
* After review and the maintainer's authorization, variant 1 starts with `launch.py grant --kind dmr1_merge_build
  ...` (its design document, section 5).
