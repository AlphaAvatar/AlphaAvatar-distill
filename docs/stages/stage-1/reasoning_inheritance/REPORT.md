# Stage 1 reasoning inheritance: structural ablation of the packed Top-1 (RI-S1)

**Status: RUNNING** (2026-11-14 UTC). Attempt `ri_diagnosis_20261113_193508` has measured every thinking-mode unit
of D, W and DW; the non-thinking-mode units are being generated. This header and sections 7, 8 and 10 are replaced
when the attempt closes.

Branch `research/reasoning-inheritance`, base `main` `50ed1a4f1f0d4dfdaa1cf30436c7e1ce5bc4ebc2` (fetched 2026-11-13,
unchanged since the brief). Experiment: `configs/stages/stage-1/reasoning_inheritance/design.json`; code and commands:
`scripts/stages/stage-1/reasoning_inheritance/README.md`; evidence: `logs/stages/stage-1/reasoning_inheritance/`.

## 1. Question and stage boundary

Stage 1 only: how much of the teacher's fluid reasoning the current initialization loses to depth pruning alone (D),
to all non-depth compression alone (W: FFN, residual width and query heads) and to both (DW); whether the two
interact; how the loss shows (wrong answers, degeneration, protocol failure, context exhaustion); and whether the
evidence favours adjacent-layer merging next. No recovery training, no new search. Stage 6 was redefined as recurrent
depth (AGENTS.md 4.8).

## 2. Anchor and teacher lineage

* Packed full search `a_packed_search_20261105_195011`, root `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`
  (shard set `a8b9dec3...`), Top-1 state `1f5e4b420125a345f1d190f0b8f87cad`, single shard
  `b74dede833fba34478a3319fb0b9c98b415e2f5b307983e2441599101cf83b6d`, path
  FFN(base@v2) -> DEPTH(base@v2) -> RESIDUAL_WIDTH(rh@v2) -> ATTENTION(rh@v2).
* Journal `artifacts/stages/stage-2/teacher_modes/search_sessions/a_packed_search_20261105_195011/states.jsonl`,
  sha256 `617d1512...` (the selection record's). `selection.py` reads the Top-1's ancestor chain into
  `frozen_selection.json` and `--check` reproduces it byte for byte.
* One lineage throughout: the teacher's tokenizer files, chat template (`a55ee1b1...`), generation config and stop ids
  `[151643, 151645]` package every group; calibration, state-evaluation reference and evaluation are the hybrid's.
  Nothing of Thinking-2507 is read. H-T (`enable_thinking=True`) and H-N are separate evaluations of the same
  checkpoint.

## 3. The frozen mapping, and the historical Top-1 reproduced exactly

| object | rebuilt from | artifact digest | journal | result |
| --- | --- | --- | --- | --- |
| FFN child, 36 blocks | teacher + the FFN step's `kept_neurons` (36 blocks) | `e2c3cabb...` | `e2c3cabb...` | equal (weights and config) |
| FFN -> DEPTH parent, 28 blocks | FFN child + kept blocks K | `b25f7fc0...` | `b25f7fc0...` | equal |
| width child | `width.global_pca_v0` replayed on the parent (config hash `958e14e6...` pinned) | `bb9464b5...` | `bb9464b5...` | equal |
| DW | recorded query heads on the width child | `536742c5...` (shard `b74dede8...`) | Top-1 | **equal** |

* K (root numbering): 0 1 2 4 5 6 7 8 9 10 12 13 14 15 16 19 20 21 22 23 24 27 28 29 31 33 34 35; removed
  3 11 17 18 25 26 30 32 (removal order 26 30 25 17 18 11 32 3).
* `P` (2560 x 1024), its scale `sqrt(2560/1024)` and the final-norm vector were recomputed once from the width
  operator's own statistics (calibration `calib.hybrid_d4_rh@v2`, the search's packed forward context, positions
  `supervised_target_v1`); every recorded diagnostic is equal (captured energy 0.9420766896432972, top eigenvalue
  0.24721078558529144, minimum kept eigenvalue 7.0743e-05), and the width child rebuilt from `P` equals the operator's
  bit for bit. **The historical Top-1 exact reproduction is established.**
* The eight removed blocks' query heads were chosen by `attention.activation_importance_v1` on the 36-block parent with
  the frozen FFN selection and `P`, frozen before any evaluation. On that parent the operator would also have changed
  24 of the 28 kept blocks' choices (387 of 448 head slots shared); the kept blocks keep the journal's selection.
* Mapping content digest `913433664fce0fed77fcf276fe2b2693bd43a8082abd25f48cc19c88f4f40034` (tensors
  `59d504cf...`), saved with every tensor's sha256.

## 4. The four groups

| group | layers | hidden | FFN | Q/KV heads | parameters | artifact digest | shard sha256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| T | 36 | 2560 | 9728 | 32/8 | 4,022,468,096 | the root teacher | -- |
| D | 28 | 2560 | 9728 | 32/8 | 3,215,021,568 | `08aa8097...` | `9de449db...` |
| W | 36 | 1024 | 3072 | 16/8 | 721,897,472 | `78d263b2...` | `64d367fe...` |
| DW | 28 | 1024 | 3072 | 16/8 | 596,049,920 | `536742c5...` | `b74dede8...` |

Checks (all passed): D's blocks are T's blocks K bit for bit and its embedding and final norm are T's; W's 28 kept
blocks and its embedding and final norm are DW's bit for bit; parameter counts equal the design's. None is trained.
Durable copy: private relay `AlphaAvatar/aadistill-artifacts`, `durable/stage1_reasoning_inheritance/913433664fce0fed/`
(mapping and the three checkpoints, 8.46 GiB), downloaded again and re-hashed.

## 5. Data and execution volumes

Counted before packing (the formal measure):

| set | role | examples | tokens |
| --- | --- | ---: | ---: |
| `calib.hybrid_d4_base@v2` | FFN and DEPTH selections (from the journal; not recomputed) | 558 | 569,270 |
| `calib.hybrid_d4_rh@v2` | width statistics (recomputed once) and the attention selections | 388 | 418,072 |
| `state_eval_hybrid@v3` | state KL / NLL against the root (`c39d0ec9...`, protocol `b760ab7e...`) | 1,366 | 1,521,086 |

What the packed forwards executed (packing exposure, never a data size): the rh profile in 59 packed sequences of
8,192 (476,466 non-padding tokens, of which 58,394 repeated fragment prefixes; 6,862 padding), read three times (width
statistics, attention on the 28-block parent, attention on the 36-block parent); 213,361 supervised-target positions
admitted. The state suite in 217 packed sequences (1,773,112 non-padding tokens, 252,026 repeated), once per group,
the teacher reference recomputed. Random order `sha256_keyed_v1` seed 20261105, sequential packing with truncation
(`prefix_restart_v1`), causal attention with continuous positions, one packed sequence per forward.

## 6. State evaluation (against the root teacher, `state_eval_hybrid@v3`)

| group | KL, equal-domain mean | KL, worst domain | critical-token KL | NLL, pooled | top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| T | 0 | 0 | 0 | 0.2039 | 1 |
| D | 0.6473 | 0.9495 | 1.3455 | 0.9408 | 0.800 |
| W | 6.4528 | 7.8589 | 12.0631 | 7.1848 | 0.151 |
| DW | 6.7768 | 7.9122 | 12.5671 | 7.3153 | 0.109 |

DW's value equals the search's journaled measurement. Per axis (KL): D 0.42-0.70 on the four fluid axes, W 4.77-6.89,
DW 5.37-7.15. In distribution terms the non-depth compression accounts for almost all of the damage; removing the 8
blocks adds 0.32 nats on top of W and costs 0.65 from T, and the two are sub-additive in KL (0.65 + 6.45 > 6.78).
**This is not a reasoning result** (P10.2): KL and NLL say how far the next-token distribution moved, not whether a
problem gets solved.

## 7. Eval-set-1

* **T** (both modes): reused from `a_teacher_20261030_012648` after every check passed (archives `7e323314...`,
  `6f6c93ef...`; every file against its descriptor; every unit's model revision, template switch, system message,
  tokenizer and template hashes, stop ids, 40,960-token context, `nucleus_v1` and seed namespace, action protocol and
  degeneration check), rescored with the current scorer (`analyses/teacher_reuse.json`). H-T: static fluid 511/570
  (89.6%), interactive 62/144 episodes, fluid formal core 573/714, GSM8K 88/100. H-N: fluid formal core 442/714.
* **D, W, DW: no unit is complete.** Generation was measured in two sessions:

| group (H-T, fluid unit) | session | tasks started | sealed | how the sealed ended | still generating when stopped |
| --- | --- | ---: | ---: | --- | ---: |
| D | attempt 2, concurrency 256 | 720 | 160 | 147 degeneration stops (cycle 55, low novelty 53, rambling 39), 13 answers at EOS; max 4,608 tokens | 560, none finished in the last 78 min |
| W | attempt 3, concurrency 12, 50 min | 36 | 24 | 24 degeneration stops (cycle 9, low novelty 15), median 768 tokens, one at 12,800 | 12 |
| DW | attempt 3, concurrency 12, 50 min | 35 | 24 | 24 degeneration stops (cycle 16, low novelty 8), median 768 tokens, one at 8,448 | 11 |
| D | attempt 3, concurrency 12, 50 min | see `analyses/cost_profile.json` | | | |

  These are **descriptive** (a time box favours what ends early); they are not scores, and no partial unit is
  reported as one. They do show the mechanism: without recovery, all three groups' thinking-mode generation mostly
  collapses into repetition within a few hundred tokens, and about a third of the started tasks run on without
  terminating or tripping the degeneration checks.

### Prediction recorded before any non-thinking-mode result was read (2026-11-14, ~01:30 UTC)

In thinking mode D keeps 32 of T's 511 correct static items (6.3%, 95% cluster interval 3.1-9.6). It keeps 18 of 88
(20.5%) where T's own generation was under 1,000 tokens and none of 71 where it was 4,000 or more, and its
degeneration stops fall at 1,280 / 1,792 / 2,304 tokens (quartiles). Descriptive and post hoc; T's length goes with
the item's difficulty.

If D's loss is mainly that long autoregressive derivations break down, then in non-thinking mode, where T's static
generations are short (median 424, p90 1,024 tokens), D keeps a clearly larger share of T's correct static items:
**the lower end of D's non-thinking-mode retention interval lies above 6.3%**. W and DW stay at the floor in both modes
(state-evaluation top-1 agreement 15% and 11%). Written while D's non-thinking-mode fluid unit was generating and
before any of its output was read; it is a prediction, not a registered gate, and decides nothing.

## 8. Engineering finding: degeneration stops never left the engine

The four paid attempts before the repair were dominated by one defect in the shared runners. vLLM V1's
`LLMEngine.abort_request(request_ids: list[str])` iterates its argument; `run_battery.py`, `uncapped_eval.py` and
`unrestricted_pilot.py` passed a bare request id, so the engine looked up each **character** of it, found none and
aborted nothing. Every request the degeneration check stopped was recorded correctly at the stop, and then went on
generating, unseen, to the 40,960-token context.

* **No result changes.** A record is taken at the stop and the stopped request's later output is ignored, so no
  historical generation or score -- the teacher's included -- is affected. Batch composition was (those zombies shared
  the batches), which is ordinary batch-level nondeterminism, not a protocol difference.
* **The cost and the scheduling did change.** On untrained models most requests stop within a few thousand tokens,
  so the engine filled with stopped requests running to the context: they held every concurrency slot and the KV
  cache, real requests waited for hours, and long-running real requests were preempted near their end. The engine's
  request count exceeded the runner's pending count by exactly the zombies (7 at concurrency 7, throughout attempt 5).
* What looked like "a third of the tasks never end" in the time-boxed measurement (`analyses/cost_profile.json`) was
  this; its projections ($17-23 per unit) are void, kept only as the record of what was observed.
* Repair: `abort_request([rid])` at the three call sites; every fake engine now has V1's signature and refuses a
  string (two had accepted one, encoding the bug); regression
  `scripts/stages/stage-0/calibration_volume/tests/test_episode_loop.py::test_a_degeneration_stop_aborts_the_request_in_the_engine`
  fails on the old call. The concurrency is now floor(KV cache / context) per group (D 7, W 6, DW 8 on an L40S), at
  which an admitted request cannot be preempted.
* After the repair D's remaining 465 thinking-mode fluid tasks finished in 26 minutes (about 20 per minute, KV cache
  about 2% used, the engine's and the runner's counts equal).

## 10. Cost and resources

| run | what it did | all-in |
| --- | --- | ---: |
| `ri_diagnosis_20261113_103414` | setup refused the artifact (not in the session commit); repaired by a $0 precheck | $0.1153 |
| `ri_diagnosis_20261113_105643` | mapping, groups, relay copy, state evaluation; D's fluid unit stalled and was stopped | $2.2939 |
| `ri_diagnosis_20261113_130551` | time-boxed measurement of W, DW, D (thinking mode, fluid) | see closeout |

## 11. Next round (evidence, derivation, hypothesis kept apart)

**Evidence from this round.**

* Distribution (state evaluation): removing the 8 blocks alone moves the root distribution moderately (KL 0.65, top-1
  agreement 80%); the non-depth compression alone destroys it (KL 6.45, top-1 15%); removing the blocks on top of it
  adds 0.32 nats. Keeping the 8 blocks at the compressed width (W, 722M parameters) instead of removing them (DW, 596M)
  buys 0.32 nats for 126M parameters.
* Generation (descriptive, thinking mode, fluid unit): every compressed group, D included, collapses into repetition on
  most started tasks within a few hundred to a few thousand tokens, and about a third of the tasks run on without
  ending. Depth pruning alone, untrained, already breaks autonomous long-form reasoning on most items -- the same
  direction as Gromov et al.'s GSM8K result for Llama-2-70B without healing (LITERATURE.md), in a different model,
  selection and regime.
* Formal reasoning accuracy: D's thinking-mode fluid unit is the one being completed (section 7); W, DW and every other
  unit are not measured.

**Derivation.**

* With W and DW at the distribution floor (top-1 agreement 11-15%), their formal accuracies can be expected near zero
  in both modes. The contrasts that involve them (T-W, W-DW, D-DW) and the interaction D+W-T-DW are then
  floor-limited: the interaction would reduce to roughly A_D - A_T and say nothing about how the two compressions
  combine. On untrained Stage-1 models a generation-based evaluation can resolve the depth axis (T vs D) only.
* Any adjacent-layer merge acts on the depth axis, which in this mapping carries the smaller share of the distribution
  damage (0.65 of 6.78 nats at the DW end). It cannot repair the non-depth damage that dominates DW, whatever the merge.

**The three merge variants, and what each would have to overcome (to test; none is claimed to work).**

1. *Direct parameter addition or averaging* of two adjacent blocks. Two residual blocks compute
   `h + f_l(h) + f_{l+1}(h + f_l(h))`; one block with summed weights computes neither that nor the parallel
   approximation `h + f_l(h) + f_{l+1}(h)`: RMSNorm weights differ per block and must be folded into the projections
   first; summed query and key projections add cross terms `Q_l K_{l+1}^T` to the attention logits; SwiGLU's gate is
   nonlinear in its weights; neurons and heads of different blocks are not aligned, so a sum mixes unrelated features;
   and averaging only fixes the scale (a sum roughly doubles pre-activations), not the function. Cheapest to test (no
   calibration); expected to work only where the two blocks are already nearly the identity.
2. *Activation-weighted merging*: the merged block's projections fitted to reproduce the pair's output on calibration
   activations (least squares per projection, or the pair's causal KL against the parent, which the DEPTH operator
   already measures in the packed context). It handles scale and part of the alignment; the nonlinearity limits it. It
   is measurable with no training, by the state evaluation, against outright removal on the same K.
3. *Static learnable fusion weights* (one coefficient per matrix or per channel, trained with KD in Stage 3): few new
   parameters, but the result is a recovery-training result, not an initialization one, so it belongs to Stage 3.

**Recommendation.**

* Do **not** make adjacent-layer merging the next priority on this evidence. In this mapping the larger lever is the
  non-depth compression. The next Stage-1 step is to decompose W's damage by factor with the same frozen mapping --
  FFN only, residual width only, query heads only, and their pairs -- each a materialization and a one-minute state
  evaluation (well under a dollar for all), then re-allocate a fixed parameter budget toward the factor that costs the
  least (for example more residual width, fewer FFN neurons or blocks), compared at matched parameters.
* For untrained Stage-1 candidates, read reasoning retention first from a graded measure without a floor -- the state
  suite's per-axis KL and NLL, or teacher-forced measures on teacher-native trajectories of a non-formal set -- and
  keep generation-based eval-set-1 for checkpoints that generate (after recovery), subject to section 9. eval-set-1
  is never used to choose a candidate.
* If depth is revisited, compare activation-weighted merging with outright removal on the same K under the state
  evaluation first; a learnable fusion follows only in Stage 3.
* All of this is conditional on the current fixed scheme (one mapping, one budget): it does not show that depth
  compression is better or worse than width compression at an arbitrary parameter budget.
