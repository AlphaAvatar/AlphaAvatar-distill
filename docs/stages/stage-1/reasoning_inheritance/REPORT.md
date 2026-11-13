# Stage 1 reasoning inheritance: structural ablation of the packed Top-1 (RI-S1)

**Status: BLOCKED at the budget boundary** (2026-11-13). The frozen mapping and the four groups are built, verified and
durable; the state evaluation is measured; the teacher's eval-set-1 evaluation is reused and rescored. The formal
eval-set-1 evaluation of D, W and DW cannot be completed inside the study's ceiling: measured on the pod, one group's
thinking-mode fluid unit alone projects to $17-23 on an L40S, and the full protocol to several hundred dollars. What
is needed from the maintainer is in section 9.

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

## 8. Why the protocol does not fit the ceiling

P18 lets every sample generate to the 40,960-token context. An untrained group's long runs neither terminate nor trip
the degeneration checks, so each one costs a full context. An L40S holds 6-11 such sequences in its KV cache and
decodes them at 140-185 tokens/s in total once the cache is full. Measured projections for one thinking-mode fluid
unit (`cost_profile.py`, each long run taken to the context): **W 20.7 GPU-h ($22.6), DW 15.3 GPU-h ($16.6)**; D's
attempt-2 run made no completion in 78 minutes with 560 requests resident (preemption and recompute under a cache of
7.5 full-context sequences), which the concurrency limit of attempt 3 removes. With six group-mode pairs, each with a
fluid unit and about a thousand capability items, the full protocol is in the hundreds of dollars on this card. The
study's ceiling is $35.00, of which `$X` is spent (section 10).

## 9. Decisions owed to the maintainer

1. **Budget**: raise RI-S1's ceiling (and the project cap, $1,000, which has $33.99 left) to fund the full protocol on
   L40S -- an order of magnitude above today's remainder; and top up the provider balance, which also funds the three
   volumes ($1.75/day).
2. **Card**: the workload is KV-cache and bandwidth bound; a larger, faster card (80 GB HBM) holds about 2.5x the
   sequences at about 4x the bandwidth, roughly a third cheaper per generated token. The grant names the L40S only.
3. **Scope or protocol for untrained Stage-1 models** (frozen science, the maintainer's alone): e.g. the thinking mode
   first; or a registered *diagnostic* evaluation (a context cap below 40,960 recorded as `diagnostic_cap`, never a P18
   score) with T scored post hoc at the same allowance from its saved generations. Nothing of this was done.

Resume point for any of them: the mapping and groups on the relay, the sealed episodes under
`artifacts/stages/stage-1/reasoning_inheritance/runs/*/scr/store`, and `launch.py stage-resume` (README).

## 10. Cost and resources

| run | what it did | all-in |
| --- | --- | ---: |
| `ri_diagnosis_20261113_103414` | setup refused the artifact (not in the session commit); repaired by a $0 precheck | $0.1153 |
| `ri_diagnosis_20261113_105643` | mapping, groups, relay copy, state evaluation; D's fluid unit stalled and was stopped | $2.2939 |
| `ri_diagnosis_20261113_130551` | time-boxed measurement of W, DW, D (thinking mode, fluid) | see closeout |

## 11. Next round (evidence, derivation, hypothesis kept apart)

See section 11 of the final revision of this report.
