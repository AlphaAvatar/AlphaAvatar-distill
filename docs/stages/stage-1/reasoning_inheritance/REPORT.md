# Stage 1 reasoning inheritance: structural ablation of the packed Top-1 (RI-S1)

**Status: COMPLETE; study CLOSED** (2026-11-14 UTC, closed by the maintainer). The frozen mapping and the four
groups are built and verified, and D, W and DW are kept on the dev box (section 4); the state evaluation is measured;
every registered eval-set-1 unit is generated and scored for D, W and DW in both modes, and the teacher's evaluation is
reused after every identity check, item by item. No recovery training, no new search. Total cost $19.9554 over six
attempts of the $35 ceiling; the unspent $15.0446 lapsed at closure (section 10). Corrections to the first version
(commit `09b48be9`) are listed in section 12.

**Result in one paragraph.** Untrained, removing 8 of the 36 blocks and changing nothing else (D) takes static fluid
reasoning from 89.6% to 5.6% in thinking mode and from 68.9% to 7.9% in non-thinking mode, and interactive episodes
from 43.1% / 34.0% to 0. Every fluid axis falls the same way, and so does every supporting unit in thinking mode; in
non-thinking mode D keeps more of reading from supplied evidence (RAG reading 81% -> 42%). The non-depth compression
(W: FFN, residual width and query heads) and both together (DW) answer no item of any formal unit correctly in either
mode (W's only correct answers are 2 of the 150 in-sample regression prompts, and it has one usable rollout in all).
The contrasts involving W or DW, and the interaction, are therefore floor-limited: on eval-set-1 correctness they
cannot show how the two compressions combine. Only the depth contrast T-D and the state evaluation (KL: D 0.65,
W 6.45, DW 6.78 nats) have range. D loses mostly to degeneration stops in thinking mode; in non-thinking mode about half
of its well-formed replies state no final answer the registered scorer can read (section 8). eval-set-1 is
development-exposed (section 8).

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
  Nothing of Thinking-2507 is read. Thinking mode (`enable_thinking=True`, record key `H-T`) and non-thinking mode
  (`H-N`) are separate evaluations of the same checkpoint.

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
Where they are (2026-11-14): on the dev box only, `~/aad-artifacts/stage1_reasoning_inheritance/913433664fce0fed/`
(the three checkpoints, the mapping, the teacher's tokenizer and chat template, the source records and a
`MANIFEST.json`; `logs/stages/stage-1/reasoning_inheritance/storage/local_checkpoints.json` is the record). They were
pulled from the relay copy the attempts uploaded, every file re-hashed against it and each group's artifact digest
recomputed; the relay's weight copies were then deleted by the maintainer's decision (the mapping and the evidence stay
on the relay). DW is the packed Top-1's own bytes (`b74dede8...`): one local copy serves both.

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
DW 5.37-7.15. The non-depth compression moves the next-token distribution about ten times as far as the depth pruning
(6.45 vs 0.65 nats); removing the 8 blocks adds 0.32 nats on top of W. KL is not additive over the two compressions
(0.65 + 6.45 > 6.78), so these values rank how far each group's distribution moved; they do not apportion the damage
between the compressions, and D's small KL sits beside an almost total loss of generation accuracy (section 8).
**This is not a reasoning result** (P10.2): KL and NLL say how far the next-token distribution moved, not whether a
problem gets solved. The state suite is also what the anchor search ranked candidates by, so DW's values are that
selection's own criterion (in sample for it); D and W were constructed from the same mapping, not searched.

## 7. Eval-set-1

### How it was measured

* Every registered unit of eval-set-1, every item generated: `fluid_reasoning_720` (576 static items and 144
  interactive episodes; the formal core is 570 static items and the 144 episodes), `stage3_capability_846` (eight
  units, 846 prompts) and `stage3_legacy_150` (in-sample regression, never generalization): 1,716 examples per group
  and mode, 10,296 generated for D, W and DW. The registered validity mask decides which rows enter the formal
  numbers; scorers are the registered ones (`score_battery` protocol v2 for fluid, `eval_set_scoring` for the rest).
* Thinking mode (`enable_thinking=True`, the primary reasoning mode) and non-thinking mode (the same checkpoint's
  control), each a separate evaluation.
* Decoding `nucleus_v1` (temperature 0.6, top-p 0.95, no other filter or penalty), one sample per item, the seed of
  every request derived from base 20261024 and the item, turn and sample (independent of model, order and batch).
  Context 40,960 tokens passed explicitly to both runners; each request may generate until EOS or the context minus
  its prompt (P18), with no token cap. The registered degeneration check runs every 256 generated tokens; interactive
  episodes run under action protocol v2. Template, system message, stop ids and tokenizer are the teacher's.
* Engine: vLLM 0.27.1 on one L40S. Attempt 6 ran at concurrency floor(KV cache / context) per group (D 7, W 6, DW 8),
  at which no admitted request can be preempted. The thinking-mode fluid units also hold items sealed earlier: D's
  255 from attempts 2-5 (concurrency 256, 12, 12, 7), W's and DW's 24 each from attempt 3's time-boxed run
  (concurrency 12). Every item has exactly one record, generated under its own fixed seed, and a unit is the union of
  its sealed parts in task order (`design.json :: evaluation.fluid_resume`); an item unfinished in an earlier attempt
  was generated again from its first token, never chosen by its outcome. Concurrency changes batch composition only
  (bf16 batch nondeterminism), never a seed.
* T (both modes) is reused from `a_teacher_20261030_012648` and rescored with the current scorers
  (`analyses/teacher_reuse.json`). Each unit's recorded identity matches this round's (root revision, context, decoder,
  template switch, system message, tokenizer and chat template, stop ids, degeneration check; for the fluid unit also
  its item file and seed namespace), and every unit is checked item by item: its records cover the registered unit
  exactly (none missing, extra or repeated); every request's seed is the decoding protocol's rule (1,272 and 1,592
  fluid turns, 996 capability and legacy requests per mode); and every prompt equals this round's D, W and DW for the
  same item. For the fluid unit that is the first user message as text (also equal to the registered item's) and the
  prompt length; the other units' records carry no prompt text and their summaries name only a pod path for the prompt
  file (`not_recorded`), so for them it is the prompt length in tokens, the system source and the think pre-opening: a
  length match, not a byte identity.
* Intervals: 95% percentile intervals over 10,000 cluster-bootstrap resamples (seed 20261113): fluid items are
  resampled by whole rule family (86 clusters), paired units by pair, other units by item. The design registered the
  eval set's independence groups for the fluid unit instead (`independence_v1.json`: only the six groups of renamed
  knights-and-knaves puzzles are merged, 708 clusters). The family clustering was chosen when the analysis was
  written; the registration is left as it was, and both are computed (table "Fluid intervals under the registered
  independence grouping"): the estimates are the same, the family intervals are the wider on every T-D contrast, and
  the retention intervals differ in both directions. They cover item and family sampling, not decoding variance (one
  sample per item).
* Analysis: `scripts/stages/stage-1/reasoning_inheritance/analysis.py` -> `analyses/structural_ablation.json`;
  tables below rendered from it by `render.py`.

<!-- tables:begin -->
<!-- generated by render.py from analyses/structural_ablation.json; do not edit by hand -->

### eval-set-1, H-T

| unit (H-T) | T correct | T usable | D correct | D usable | W correct | W usable | DW correct | DW usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fluid formal core (714) | 573/714 (80.3%) |  | 32/714 (4.5%) |  | 0/714 (0.0%) |  | 0/714 (0.0%) |  |
| static fluid (570) | 511/570 (89.6%) | 562/570 (98.6%) | 32/570 (5.6%) | 66/570 (11.6%) | 0/570 (0.0%) | 0/570 (0.0%) | 0/570 (0.0%) | 0/570 (0.0%) |
| -- task-contained deduction | 189/189 (100.0%) | 189/189 (100.0%) | 26/189 (13.8%) | 38/189 (20.1%) | 0/189 (0.0%) | 0/189 (0.0%) | 0/189 (0.0%) | 0/189 (0.0%) |
| -- inductive rule discovery | 130/142 (91.5%) | 142/142 (100.0%) | 5/142 (3.5%) | 11/142 (7.7%) | 0/142 (0.0%) | 0/142 (0.0%) | 0/142 (0.0%) | 0/142 (0.0%) |
| -- compositional generalization | 129/144 (89.6%) | 142/144 (98.6%) | 1/144 (0.7%) | 4/144 (2.8%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) |
| -- abstract / analogical (sub-category) | 63/95 (66.3%) | 89/95 (93.7%) | 0/95 (0.0%) | 13/95 (13.7%) | 0/95 (0.0%) | 0/95 (0.0%) | 0/95 (0.0%) | 0/95 (0.0%) |
| interactive adaptation (144 episodes: success / all turns usable) | 62/144 (43.1%) | 125/144 (86.8%) | 0/144 (0.0%) | 4/144 (2.8%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) |
| math (verified) | 85/100 (85.0%) | 98/100 (98.0%) | 5/100 (5.0%) | 6/100 (6.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| GSM8K | 88/100 (88.0%) | 100/100 (100.0%) | 0/100 (0.0%) | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| multi-hop reading (evidence given) | 79/100 (79.0%) | 99/100 (99.0%) | 6/100 (6.0%) | 11/100 (11.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| RAG reading (evidence given) | 73/100 (73.0%) | 100/100 (100.0%) | 1/100 (1.0%) | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| answerability pairs | 33/60 (55.0%) | 59/60 (98.3%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) |
| safety pairs | 22/50 (44.0%) | 50/50 (100.0%) | 0/50 (0.0%) | 2/50 (4.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| TriviaQA (supplementary, closed-book) * | 79/150 (52.7%) | 148/150 (98.7%) | 5/150 (3.3%) | 16/150 (10.7%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) |
| tool calling (exact call match) | 4/5 (80.0%) | 5/5 (100.0%) | 2/5 (40.0%) | 4/5 (80.0%) | 0/5 (0.0%) | 0/5 (0.0%) | 0/5 (0.0%) | 0/5 (0.0%) |
| instruction (format_ok) | 12/12 (100.0%) | 12/12 (100.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) |
| short realtime (format_ok) | 10/10 (100.0%) | 10/10 (100.0%) | 1/10 (10.0%) | 1/10 (10.0%) | 0/10 (0.0%) | 0/10 (0.0%) | 0/10 (0.0%) | 0/10 (0.0%) |
| stage3_legacy_150 (in-sample regression, not generalization) | 143/150 (95.3%) |  | 57/150 (38.0%) |  | 0/150 (0.0%) |  | 0/150 (0.0%) |  |

\* outside the formal core (supplementary).

### eval-set-1, H-N

| unit (H-N) | T correct | T usable | D correct | D usable | W correct | W usable | DW correct | DW usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fluid formal core (714) | 442/714 (61.9%) |  | 45/714 (6.3%) |  | 0/714 (0.0%) |  | 0/714 (0.0%) |  |
| static fluid (570) | 393/570 (68.9%) | 563/570 (98.8%) | 45/570 (7.9%) | 308/570 (54.0%) | 0/570 (0.0%) | 0/570 (0.0%) | 0/570 (0.0%) | 0/570 (0.0%) |
| -- task-contained deduction | 161/189 (85.2%) | 188/189 (99.5%) | 31/189 (16.4%) | 104/189 (55.0%) | 0/189 (0.0%) | 0/189 (0.0%) | 0/189 (0.0%) | 0/189 (0.0%) |
| -- inductive rule discovery | 90/142 (63.4%) | 138/142 (97.2%) | 4/142 (2.8%) | 62/142 (43.7%) | 0/142 (0.0%) | 0/142 (0.0%) | 0/142 (0.0%) | 0/142 (0.0%) |
| -- compositional generalization | 103/144 (71.5%) | 143/144 (99.3%) | 8/144 (5.6%) | 90/144 (62.5%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) |
| -- abstract / analogical (sub-category) | 39/95 (41.1%) | 94/95 (98.9%) | 2/95 (2.1%) | 52/95 (54.7%) | 0/95 (0.0%) | 0/95 (0.0%) | 0/95 (0.0%) | 0/95 (0.0%) |
| interactive adaptation (144 episodes: success / all turns usable) | 49/144 (34.0%) | 132/144 (91.7%) | 0/144 (0.0%) | 10/144 (6.9%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) | 0/144 (0.0%) |
| math (verified) | 76/100 (76.0%) | 97/100 (97.0%) | 9/100 (9.0%) | 26/100 (26.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| GSM8K | 38/100 (38.0%) | 100/100 (100.0%) | 0/100 (0.0%) | 50/100 (50.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| multi-hop reading (evidence given) | 75/100 (75.0%) | 100/100 (100.0%) | 28/100 (28.0%) | 44/100 (44.0%) | 0/100 (0.0%) | 1/100 (1.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| RAG reading (evidence given) | 81/100 (81.0%) | 100/100 (100.0%) | 42/100 (42.0%) | 91/100 (91.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| answerability pairs | 33/60 (55.0%) | 60/60 (100.0%) | 19/60 (31.7%) | 48/60 (80.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) |
| safety pairs | 31/50 (62.0%) | 50/50 (100.0%) | 1/50 (2.0%) | 2/50 (4.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| TriviaQA (supplementary, closed-book) * | 78/150 (52.0%) | 150/150 (100.0%) | 10/150 (6.7%) | 24/150 (16.0%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) |
| tool calling (exact call match) | 4/5 (80.0%) | 5/5 (100.0%) | 0/5 (0.0%) | 4/5 (80.0%) | 0/5 (0.0%) | 0/5 (0.0%) | 0/5 (0.0%) | 0/5 (0.0%) |
| instruction (format_ok) | 12/12 (100.0%) | 12/12 (100.0%) | 2/12 (16.7%) | 2/12 (16.7%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) |
| short realtime (format_ok) | 10/10 (100.0%) | 10/10 (100.0%) | 5/10 (50.0%) | 5/10 (50.0%) | 0/10 (0.0%) | 0/10 (0.0%) | 0/10 (0.0%) | 0/10 (0.0%) |
| stage3_legacy_150 (in-sample regression, not generalization) | 110/150 (73.3%) |  | 57/150 (38.0%) |  | 2/150 (1.3%) |  | 0/150 (0.0%) |  |

\* outside the formal core (supplementary).

### Generation behaviour

| static fluid (570) | natural termination (EOS) | usable | correct among usable (descriptive) | degeneration | context limit | tokens p50 | p90 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T H-T | 562 (98.6%) | 562 (98.6%) | 511/562 | 1.4% | 0.0% | 1999 | 8403 | 3310.5 |
| D H-T | 78 (13.7%) | 66 (11.6%) | 32/66 | 86.3% | 0.0% | 1792 | 3072 | 1887.9 |
| W H-T | 8 (1.4%) | 0 (0.0%) | -- | 97.4% | 1.2% | 768 | 1536 | 1819.5 |
| DW H-T | 0 (0.0%) | 0 (0.0%) | -- | 100.0% | 0.0% | 768 | 2304 | 1267.4 |
| T H-N | 563 (98.8%) | 563 (98.8%) | 393/563 | 1.2% | 0.0% | 424 | 1024 | 547.3 |
| D H-N | 308 (54.0%) | 308 (54.0%) | 45/308 | 46.0% | 0.0% | 763 | 1536 | 730.6 |
| W H-N | 5 (0.9%) | 0 (0.0%) | -- | 98.1% | 1.1% | 768 | 1235 | 1568.5 |
| DW H-N | 0 (0.0%) | 0 (0.0%) | -- | 100.0% | 0.0% | 768 | 1792 | 1164.1 |

### Structural contrasts, H-T

| H-T | n | T | D | W | DW | depth_alone (T-D) | non_depth_alone (T-W) | depth_given_non_depth (W-DW) | non_depth_given_depth (D-DW) | interaction (D+W-T-DW) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fluid/abstract_analogical | 95 | 66.3 | 0.0 | 0.0 | 0.0 | +66.3 [+47.1, +86.0] | +66.3 [+46.9, +86.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -66.3 [-86.1, -47.4] |
| fluid/compositional_generalization | 144 | 89.6 | 0.7 | 0.0 | 0.0 | +88.9 [+80.3, +96.2] | +89.6 [+81.0, +97.0] | +0.0 [+0.0, +0.0] | +0.7 [+0.0, +2.2] | -88.9 [-96.1, -80.3] |
| fluid/inductive_rule_discovery | 142 | 91.5 | 3.5 | 0.0 | 0.0 | +88.0 [+81.0, +94.1] | +91.5 [+85.7, +96.3] | +0.0 [+0.0, +0.0] | +3.5 [+0.0, +9.0] | -88.0 [-94.0, -81.0] |
| fluid/interactive_adaptation | 144 | 43.1 | 0.0 | 0.0 | 0.0 | +43.1 [+26.4, +58.5] | +43.1 [+26.2, +58.6] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -43.1 [-59.2, -26.1] |
| fluid/static_all | 570 | 89.6 | 5.6 | 0.0 | 0.0 | +84.0 [+78.6, +89.0] | +89.6 [+84.2, +94.3] | +0.0 [+0.0, +0.0] | +5.6 [+2.7, +8.7] | -84.0 [-89.0, -78.6] |
| fluid/task_contained_deduction | 189 | 100.0 | 13.8 | 0.0 | 0.0 | +86.2 [+79.2, +92.7] | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] | +13.8 [+7.4, +20.8] | -86.2 [-92.6, -79.4] |
| stage3_capability_846/answerability_paired | 60 | 55.0 | 0.0 | 0.0 | 0.0 | +55.0 [+41.7, +68.3] | +55.0 [+41.7, +66.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -55.0 [-66.7, -41.7] |
| stage3_capability_846/gsm8k | 100 | 88.0 | 0.0 | 0.0 | 0.0 | +88.0 [+81.0, +94.0] | +88.0 [+81.0, +94.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -88.0 [-94.0, -81.0] |
| stage3_capability_846/knowledge | 150 | 52.7 | 3.3 | 0.0 | 0.0 | +49.3 [+41.3, +57.3] | +52.7 [+44.7, +60.7] | +0.0 [+0.0, +0.0] | +3.3 [+0.7, +6.7] | -49.3 [-57.3, -41.3] |
| stage3_capability_846/math_verified | 100 | 85.0 | 5.0 | 0.0 | 0.0 | +80.0 [+72.0, +88.0] | +85.0 [+78.0, +92.0] | +0.0 [+0.0, +0.0] | +5.0 [+1.0, +10.0] | -80.0 [-88.0, -72.0] |
| stage3_capability_846/multihop | 100 | 79.0 | 6.0 | 0.0 | 0.0 | +73.0 [+64.0, +81.0] | +79.0 [+71.0, +87.0] | +0.0 [+0.0, +0.0] | +6.0 [+2.0, +11.0] | -73.0 [-82.0, -64.0] |
| stage3_capability_846/rag | 100 | 73.0 | 1.0 | 0.0 | 0.0 | +72.0 [+63.0, +81.0] | +73.0 [+64.0, +82.0] | +0.0 [+0.0, +0.0] | +1.0 [+0.0, +3.0] | -72.0 [-80.0, -63.0] |
| stage3_capability_846/safety_paired | 50 | 44.0 | 0.0 | 0.0 | 0.0 | +44.0 [+30.0, +58.0] | +44.0 [+30.0, +58.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -44.0 [-58.0, -30.0] |

Percentage points; 95% cluster-bootstrap intervals (fluid: whole rule families resampled, not the registered independence groups; both are compared below).

### Retention, H-T

| H-T: of T's correct items | n | D keeps | W keeps | DW keeps |
| --- | ---: | ---: | ---: | ---: |
| fluid/abstract_analogical | 63 | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/compositional_generalization | 129 | 1 (0.8% [0.0, 2.5]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/inductive_rule_discovery | 130 | 5 (3.9% [0.0, 9.8]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/interactive_adaptation | 62 | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/static_all | 511 | 32 (6.3% [3.1, 9.6]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/task_contained_deduction | 189 | 26 (13.8% [7.2, 20.7]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/answerability_paired | 33 | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/gsm8k | 88 | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/knowledge | 79 | 5 (6.3% [1.3, 12.7]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/math_verified | 85 | 5 (5.9% [1.2, 11.8]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/multihop | 79 | 6 (7.6% [2.5, 13.9]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/rag | 73 | 1 (1.4% [0.0, 4.1]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/safety_paired | 22 | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |

### Structural contrasts, H-N

| H-N | n | T | D | W | DW | depth_alone (T-D) | non_depth_alone (T-W) | depth_given_non_depth (W-DW) | non_depth_given_depth (D-DW) | interaction (D+W-T-DW) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fluid/abstract_analogical | 95 | 41.0 | 2.1 | 0.0 | 0.0 | +39.0 [+20.0, +60.0] | +41.0 [+22.8, +61.7] | +0.0 [+0.0, +0.0] | +2.1 [+0.0, +5.2] | -39.0 [-60.2, -20.6] |
| fluid/compositional_generalization | 144 | 71.5 | 5.6 | 0.0 | 0.0 | +66.0 [+46.4, +83.7] | +71.5 [+50.4, +90.3] | +0.0 [+0.0, +0.0] | +5.6 [+2.1, +9.2] | -66.0 [-83.7, -46.4] |
| fluid/inductive_rule_discovery | 142 | 63.4 | 2.8 | 0.0 | 0.0 | +60.6 [+49.2, +71.3] | +63.4 [+51.0, +74.8] | +0.0 [+0.0, +0.0] | +2.8 [+0.0, +6.3] | -60.6 [-71.1, -49.6] |
| fluid/interactive_adaptation | 144 | 34.0 | 0.0 | 0.0 | 0.0 | +34.0 [+18.3, +52.2] | +34.0 [+18.2, +51.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -34.0 [-52.4, -17.8] |
| fluid/static_all | 570 | 69.0 | 7.9 | 0.0 | 0.0 | +61.1 [+53.6, +68.0] | +69.0 [+60.6, +76.6] | +0.0 [+0.0, +0.0] | +7.9 [+4.7, +11.3] | -61.1 [-68.0, -53.6] |
| fluid/task_contained_deduction | 189 | 85.2 | 16.4 | 0.0 | 0.0 | +68.8 [+63.2, +75.0] | +85.2 [+79.2, +91.4] | +0.0 [+0.0, +0.0] | +16.4 [+9.0, +24.9] | -68.8 [-75.3, -63.2] |
| stage3_capability_846/answerability_paired | 60 | 55.0 | 31.7 | 0.0 | 0.0 | +23.3 [+8.3, +38.3] | +55.0 [+41.7, +66.7] | +0.0 [+0.0, +0.0] | +31.7 [+20.0, +43.3] | -23.3 [-38.3, -8.3] |
| stage3_capability_846/gsm8k | 100 | 38.0 | 0.0 | 0.0 | 0.0 | +38.0 [+29.0, +48.0] | +38.0 [+29.0, +48.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -38.0 [-48.0, -28.0] |
| stage3_capability_846/knowledge | 150 | 52.0 | 6.7 | 0.0 | 0.0 | +45.3 [+37.3, +53.3] | +52.0 [+44.0, +60.0] | +0.0 [+0.0, +0.0] | +6.7 [+3.3, +10.7] | -45.3 [-53.3, -37.3] |
| stage3_capability_846/math_verified | 100 | 76.0 | 9.0 | 0.0 | 0.0 | +67.0 [+58.0, +76.0] | +76.0 [+67.0, +84.0] | +0.0 [+0.0, +0.0] | +9.0 [+4.0, +15.0] | -67.0 [-76.0, -57.0] |
| stage3_capability_846/multihop | 100 | 75.0 | 28.0 | 0.0 | 0.0 | +47.0 [+36.0, +58.0] | +75.0 [+66.0, +83.0] | +0.0 [+0.0, +0.0] | +28.0 [+20.0, +37.0] | -47.0 [-58.0, -36.0] |
| stage3_capability_846/rag | 100 | 81.0 | 42.0 | 0.0 | 0.0 | +39.0 [+29.0, +49.0] | +81.0 [+73.0, +88.0] | +0.0 [+0.0, +0.0] | +42.0 [+32.0, +52.0] | -39.0 [-49.0, -29.0] |
| stage3_capability_846/safety_paired | 50 | 62.0 | 2.0 | 0.0 | 0.0 | +60.0 [+46.0, +74.0] | +62.0 [+48.0, +76.0] | +0.0 [+0.0, +0.0] | +2.0 [+0.0, +6.0] | -60.0 [-74.0, -46.0] |

Percentage points; 95% cluster-bootstrap intervals (fluid: whole rule families resampled, not the registered independence groups; both are compared below).

### Retention, H-N

| H-N: of T's correct items | n | D keeps | W keeps | DW keeps |
| --- | ---: | ---: | ---: | ---: |
| fluid/abstract_analogical | 39 | 1 (2.6% [0.0, 8.8]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/compositional_generalization | 103 | 8 (7.8% [3.4, 11.5]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/inductive_rule_discovery | 90 | 4 (4.4% [0.0, 9.5]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/interactive_adaptation | 49 | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/static_all | 393 | 44 (11.2% [7.0, 15.6]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| fluid/task_contained_deduction | 161 | 31 (19.2% [11.1, 27.7]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/answerability_paired | 33 | 14 (42.4% [27.3, 60.6]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/gsm8k | 38 | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/knowledge | 78 | 10 (12.8% [6.4, 20.5]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/math_verified | 76 | 9 (11.8% [5.3, 19.7]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/multihop | 75 | 25 (33.3% [22.7, 44.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/rag | 81 | 41 (50.6% [39.5, 61.7]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |
| stage3_capability_846/safety_paired | 31 | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) | 0 (0.0% [0.0, 0.0]) |

### Failure types

| static fluid, T-correct items the group lost | lost | answer read, wrong | no answer the scorer can read | protocol error | degeneration stop | context limit | other non-EOS stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D H-N | 349 | 67 | 118 | 0 | 164 | 0 | 0 |
| DW H-N | 393 | 0 | 0 | 0 | 393 | 0 | 0 |
| W H-N | 393 | 0 | 0 | 5 | 382 | 6 | 0 |
| D H-T | 479 | 26 | 4 | 11 | 438 | 0 | 0 |
| DW H-T | 511 | 0 | 0 | 0 | 511 | 0 | 0 |
| W H-T | 511 | 0 | 0 | 8 | 496 | 7 | 0 |

### What every row ended as (T and D)

| unit | group, mode | n | correct | answer read, wrong | no answer the scorer can read | protocol error | degeneration stop | context limit | other non-EOS stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static fluid (570) | T H-T | 570 | 511 | 51 | 0 | 0 | 8 | 0 | 0 |
| static fluid (570) | T H-N | 570 | 393 | 170 | 0 | 0 | 7 | 0 | 0 |
| static fluid (570) | D H-T | 570 | 32 | 30 | 4 | 12 | 492 | 0 | 0 |
| static fluid (570) | D H-N | 570 | 45 | 101 | 162 | 0 | 262 | 0 | 0 |
| math (verified) | T H-T | 100 | 85 | 13 | 0 | 0 | 2 | 0 | 0 |
| math (verified) | T H-N | 100 | 76 | 21 | 0 | 0 | 3 | 0 | 0 |
| math (verified) | D H-T | 100 | 5 | 1 | 0 | 0 | 94 | 0 | 0 |
| math (verified) | D H-N | 100 | 9 | 9 | 8 | 0 | 74 | 0 | 0 |
| GSM8K | T H-T | 100 | 88 | 11 | 1 | 0 | 0 | 0 | 0 |
| GSM8K | T H-N | 100 | 38 | 7 | 55 | 0 | 0 | 0 | 0 |
| GSM8K | D H-T | 100 | 0 | 1 | 3 | 0 | 96 | 0 | 0 |
| GSM8K | D H-N | 100 | 0 | 1 | 49 | 1 | 49 | 0 | 0 |
| multi-hop reading (evidence given) | T H-T | 100 | 79 | 20 | 0 | 0 | 1 | 0 | 0 |
| multi-hop reading (evidence given) | T H-N | 100 | 75 | 25 | 0 | 0 | 0 | 0 | 0 |
| multi-hop reading (evidence given) | D H-T | 100 | 6 | 5 | 0 | 39 | 50 | 0 | 0 |
| multi-hop reading (evidence given) | D H-N | 100 | 28 | 16 | 0 | 2 | 54 | 0 | 0 |
| RAG reading (evidence given) | T H-T | 100 | 73 | 27 | 0 | 0 | 0 | 0 | 0 |
| RAG reading (evidence given) | T H-N | 100 | 81 | 19 | 0 | 0 | 0 | 0 | 0 |
| RAG reading (evidence given) | D H-T | 100 | 1 | 2 | 0 | 90 | 7 | 0 | 0 |
| RAG reading (evidence given) | D H-N | 100 | 42 | 49 | 0 | 1 | 8 | 0 | 0 |
| answerability pairs | T H-T | 120 | 87 | 32 | 0 | 0 | 1 | 0 | 0 |
| answerability pairs | T H-N | 120 | 90 | 30 | 0 | 0 | 0 | 0 | 0 |
| answerability pairs | D H-T | 120 | 5 | 0 | 0 | 103 | 12 | 0 | 0 |
| answerability pairs | D H-N | 120 | 63 | 43 | 0 | 1 | 13 | 0 | 0 |
| safety pairs | T H-T | 100 | 72 | 28 | 0 | 0 | 0 | 0 | 0 |
| safety pairs | T H-N | 100 | 81 | 19 | 0 | 0 | 0 | 0 | 0 |
| safety pairs | D H-T | 100 | 5 | 7 | 0 | 0 | 88 | 0 | 0 |
| safety pairs | D H-N | 100 | 9 | 14 | 0 | 8 | 69 | 0 | 0 |
| TriviaQA (supplementary, closed-book) | T H-T | 150 | 79 | 69 | 0 | 0 | 2 | 0 | 0 |
| TriviaQA (supplementary, closed-book) | T H-N | 150 | 78 | 72 | 0 | 0 | 0 | 0 | 0 |
| TriviaQA (supplementary, closed-book) | D H-T | 150 | 5 | 11 | 0 | 0 | 134 | 0 | 0 |
| TriviaQA (supplementary, closed-book) | D H-N | 150 | 10 | 14 | 0 | 0 | 126 | 0 | 0 |

Rows, not pairs, for the paired units. "Answer read, wrong" includes the behaviour units' wrong behaviours (echoed the passage, answered an unanswerable question, refused an answerable or benign one, complied with an unsafe one). Descriptive; the formal score is the registered scorer's and unchanged.

### Degeneration-stopped records as sealed

| group, mode | degeneration stops (turns) | saved ids = recorded length | stopped at a check | inside the allowance |
| --- | ---: | ---: | ---: | ---: |
| T H-T | 33 | 33 | 33 | 33 |
| D H-T | 1239 | 1239 | 1239 | 1239 |
| W H-T | 1641 | 1641 | 1641 | 1641 |
| DW H-T | 1715 | 1715 | 1715 | 1715 |
| T H-N | 22 | 22 | 22 | 22 |
| D H-N | 865 | 865 | 865 | 865 |
| W H-N | 1652 | 1652 | 1652 | 1652 |
| DW H-N | 1714 | 1714 | 1714 | 1714 |

### Fluid intervals under the registered independence grouping

| fluid, T-D (pp) / D keeps of T-correct | mode | rule families (reported) | independence groups (registered) | retention, families | retention, independence groups |
| --- | --- | ---: | ---: | ---: | ---: |
| fluid/abstract_analogical | H-T | +66.3 [+47.1, +86.0] | +66.3 [+56.8, +75.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| fluid/compositional_generalization | H-T | +88.9 [+80.3, +96.2] | +88.9 [+83.3, +93.8] | 0.8 [0.0, 2.5] | 0.8 [0.0, 2.3] |
| fluid/inductive_rule_discovery | H-T | +88.0 [+81.0, +94.1] | +88.0 [+82.4, +93.0] | 3.9 [0.0, 9.8] | 3.9 [0.8, 7.7] |
| fluid/interactive_adaptation | H-T | +43.1 [+26.4, +58.5] | +43.1 [+34.7, +51.4] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| fluid/static_all | H-T | +84.0 [+78.6, +89.0] | +84.0 [+81.0, +87.0] | 6.3 [3.1, 9.6] | 6.3 [4.3, 8.4] |
| fluid/task_contained_deduction | H-T | +86.2 [+79.2, +92.7] | +86.2 [+81.0, +91.1] | 13.8 [7.2, 20.7] | 13.8 [8.9, 18.9] |
| fluid/abstract_analogical | H-N | +39.0 [+20.0, +60.0] | +39.0 [+29.5, +49.5] | 2.6 [0.0, 8.8] | 2.6 [0.0, 7.7] |
| fluid/compositional_generalization | H-N | +66.0 [+46.4, +83.7] | +66.0 [+58.3, +73.6] | 7.8 [3.4, 11.5] | 7.8 [2.9, 13.6] |
| fluid/inductive_rule_discovery | H-N | +60.6 [+49.2, +71.3] | +60.6 [+52.1, +68.3] | 4.4 [0.0, 9.5] | 4.4 [1.1, 8.9] |
| fluid/interactive_adaptation | H-N | +34.0 [+18.3, +52.2] | +34.0 [+26.4, +41.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| fluid/static_all | H-N | +61.1 [+53.6, +68.0] | +61.1 [+56.9, +65.1] | 11.2 [7.0, 15.6] | 11.2 [8.2, 14.4] |
| fluid/task_contained_deduction | H-N | +68.8 [+63.2, +75.0] | +68.8 [+61.9, +75.4] | 19.2 [11.1, 27.7] | 19.2 [13.3, 25.5] |

### Kept by the teacher's length (descriptive, post hoc)

| kept / T-correct, by T's tokens | T 0-1000 | T 1000-2000 | T 2000-4000 | T 4000-8000 | T >= 8000 | group's degeneration stop p25 / p50 / p75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D H-N | 44/379 | 0/12 | 0/2 | 0/0 | 0/0 | 768 / 1024 / 1280 (n 262) |
| DW H-N | 0/379 | 0/12 | 0/2 | 0/0 | 0/0 | 768 / 768 / 1024 (n 570) |
| W H-N | 0/379 | 0/12 | 0/2 | 0/0 | 0/0 | 768 / 768 / 768 (n 559) |
| D H-T | 18/88 | 10/197 | 4/155 | 0/54 | 0/17 | 1280 / 1792 / 2304 (n 492) |
| DW H-T | 0/88 | 0/197 | 0/155 | 0/54 | 0/17 | 768 / 768 / 1280 (n 570) |
| W H-T | 0/88 | 0/197 | 0/155 | 0/54 | 0/17 | 768 / 768 / 768 (n 555) |

### State evaluation (state_eval_hybrid@v3, against the root teacher)

| group | KL equal-domain mean | KL worst domain | critical-token KL | NLL pooled |
| --- | ---: | ---: | ---: | ---: |
| D | 0.6473 | 0.9495 | 1.3455 | 0.9408 |
| W | 6.4528 | 7.8589 | 12.0631 | 7.1848 |
| DW | 6.7768 | 7.9122 | 12.5671 | 7.3153 |

<!-- tables:end -->

### The prediction recorded before the non-thinking-mode results (2026-11-14, ~01:30 UTC; commit `e82cf7ef`)

Written while D's non-thinking-mode fluid unit was generating and before any of its output was read: if D's loss is
mainly that long derivations break down, then in non-thinking mode D keeps a clearly larger share of T's correct
static items, *the lower end of its retention interval above thinking mode's 6.3%*.

**Outcome: the registered criterion is met, narrowly; the explanation is supported only in part.** D keeps 44 of T's
393 correct static items in non-thinking mode, 11.2% [7.0, 15.6] (thinking: 32 of 511, 6.3% [3.1, 9.6]); under the
registered independence grouping 11.2% [8.2, 14.4] against 6.3% [4.3, 8.4]. The criterion compares one mode's
interval with the other mode's point estimate, as the prediction was written. It is **not** a paired significance test
across modes, and none was run (the modes' T-correct sets differ: 393 and 511 items).

As for the explanation: on the items T answered in under 1,000 tokens D keeps 20.5% (18 / 88) in thinking mode and only
11.6% (44 / 379) in non-thinking mode. In non-thinking mode fewer than half of the T-correct items D loses end in a
degeneration stop (164 of 349); the other 185 end naturally and well-formed, 118 with no final answer the registered
scorer can read and 67 with an answer read and wrong. So instability of long generation does not explain all of the
damage, short derivations are damaged too, and the thinking-mode length gradient partly reflects which items T solves
briefly. That long-generation instability is part of the damage is not excluded: in thinking mode 438 of D's 479
losses are degeneration stops.

## 8. What the results say, and what they do not

**Depth alone (T-D), the one contrast with range.** Removing blocks 3, 11, 17, 18, 25, 26, 30 and 32 (22% of the
depth), with every remaining parameter the teacher's, removes nearly all of the measured fluid reasoning:

* static fluid: -84.0 pp [78.6, 89.0] in thinking mode, -61.1 pp [53.6, 68.0] in non-thinking mode; interactive
  episodes -43.1 pp [26.4, 58.5] and -34.0 pp [18.3, 52.2] (D solves none of the 144 in either mode);
* by axis, thinking mode: task-contained deduction keeps the most (100% -> 13.8%), then inductive rule discovery
  (91.5% -> 3.5%), compositional generalization (89.6% -> 0.7%) and abstract/analogical (66.3% -> 0); in non-thinking
  mode deduction again keeps the most (85.2% -> 16.4%), then compositional generalization (71.5% -> 5.6%), inductive
  rule discovery (63.4% -> 2.8%) and abstract/analogical (41.1% -> 2.1%);
* the supporting units fall as far: GSM8K 88% -> 0 and verified math 85% -> 5% (thinking), multi-hop reading 79% -> 6%
  and RAG reading 73% -> 1% with the evidence supplied, closed-book TriviaQA 52.7% -> 3.3%. In thinking mode the math
  units are lost to degeneration stops (94 and 96 of 100), the reading units largely to broken think delimiters
  (RAG 90 of 100, answerability 103 of 120 rows, multi-hop 39 of 100), a protocol error rather than a wrong answer;
* D keeps essentially nothing T could not do: every item D answers correctly in thinking mode T also answers.
* In non-thinking mode D keeps far more of reading from supplied evidence than of derivation: RAG reading 81% -> 42%
  (half of T's correct items kept), multi-hop reading 75% -> 28%, answerability pairs 55% -> 31.7%, against GSM8K
  38% -> 0, verified math 76% -> 9% and static fluid 68.9% -> 7.9%. In thinking mode every one of these is at 6% or
  below. GSM8K's 0 in non-thinking mode is mostly extraction: 49 of D's 50 well-formed replies state no final answer
  the registered scorer reads; of the other 50 rows, 49 are degeneration stops and one breaks the think protocol. T's own non-thinking GSM8K has 55 of 100
  such replies, which is why T is at 38% there; the unit measures answer format as well as arithmetic in this mode. Descriptive, across mixed-domain supporting units, never read as fluid reasoning (P10.2); in direction it is
  Gromov et al.'s split between knowledge-style QA and reasoning.

**How D fails depends on the mode.** In thinking mode 86.3% of D's static rollouts end in a degeneration stop (at
1,280 / 1,792 / 2,304 tokens, quartiles), and 438 of the 479 T-correct items it loses end that way. 78 rollouts end
naturally (EOS); 12 of those break the think protocol, so 66 are usable, and 32 of the 66 are correct (30 give an
answer read and wrong, 4 no final answer the scorer reads). In non-thinking mode all 308 natural terminations are
usable (54.0%) and 45 are correct (14.6%, against T's 69.8%); of the other 263, 162 state no final answer the
registered scorer can read (no `\boxed{}`) and 101 give an answer read and wrong. So in non-thinking mode the loss is
no longer mainly degeneration, and it is not mainly wrong answers either: more of D's well-formed replies fail to
state a scorable answer than state a wrong one. Whether those replies hold a correct result in another form is not
measured; the formal score is the registered scorer's and is unchanged. All of this is descriptive (each group
finishes a different subset).

**Non-depth compression (W) and both (DW): the floor.** W and DW answer no item of any formal unit correctly, in
either mode, and produce no usable rollout (W's exceptions: 2 of the 150 in-sample regression prompts and one usable
multi-hop rollout, both in non-thinking mode); 97-100% of their static rollouts are stopped for degeneration, mostly
at the third check (768 tokens), and about 1% of W's run to the 40,960-token context without tripping a check
(right-censored; incorrect under the registered scorer). Hence T-W equals T's accuracy, W-DW is zero, D-DW is D's own
accuracy, and the interaction D+W-T-DW equals -(T-D), all by construction. These numbers are reported because the
design registered them; **they are not measurements of an interaction**, and no claim about how depth and non-depth
compression combine follows from them.

**The graded measure.** Only the state evaluation separates W from DW: KL 6.45 vs 6.78 nats, top-1 agreement 15% vs
11%. D sits far from both (KL 0.65, top-1 agreement 80%), and still loses 84 pp of thinking-mode accuracy. One reading
is that a moderate per-position shift compounds over thousands of autoregressive tokens; it fits the thinking-mode
degeneration stops, but not the whole loss, since D also fails short non-thinking replies (above). (Inference,
consistent with Gromov et al.'s unhealed GSM8K result on Llama-2-70B, LITERATURE.md; their setting differs in model,
selection and measurement.)

**Limits.**

* eval-set-1 is **development-exposed**: these are not blind held-out numbers. It chose nothing here (the mapping
  was fixed by the anchor search and frozen before any evaluation).
* Untrained Stage-1 initializations only: nothing here says what recovery training can restore, or that a W- or
  DW-like student is worse than D after recovery.
* One mapping (the packed search's Top-1), one depth fraction (8 of 36), one width budget. Not a comparison of
  depth against width compression at matched parameters (D has 3.22B parameters, W 0.72B).
* One sample per item; intervals cover item sampling only.
* The removed blocks' query heads in W were chosen on the 36-block parent with the original method and frozen before
  evaluation; W is a construction from the mapping, not a searched candidate.
* `stage3_legacy_150` is in-sample regression (training prompts of an earlier round) and is never read as
  generalization; tool calling has 5 items.

## 9. Engineering finding: degeneration stops never left the engine

The four paid attempts before the repair were dominated by one defect in the shared runners. vLLM V1's
`LLMEngine.abort_request(request_ids: list[str])` iterates its argument; `run_battery.py`, `uncapped_eval.py` and
`unrestricted_pilot.py` passed a bare request id, so the engine looked up each **character** of it, found none and
aborted nothing. Every request the degeneration check stopped was recorded correctly at the stop, and then went on
generating, unseen, to the 40,960-token context.

* **What it did not change, and how far that goes.** A record is taken at the stop and the stopped request's later
  output is dropped (the runner no longer holds the request). The sealed records show no sign of a continuation: every
  one of the 8,881 degeneration-stopped records of this round and of the teacher's evaluation holds exactly as many
  output token ids as its recorded length, stopped exactly at a check and inside its allowance (table
  "Degeneration-stopped records as sealed"). What the defect did change is batch composition (the zombies shared the
  batches), and so, through bf16 batch nondeterminism, which tokens other requests sampled. This does not show that a
  rerun before and after the repair would agree sample by sample or score by score: a rerun is a new sample under the
  same seeds, and none was made.
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

| attempt | what it did | minutes | all-in |
| --- | --- | ---: | ---: |
| `ri_diagnosis_20261113_103414` | setup refused the artifact (not in the session commit); repaired by a $0 precheck | 6.2 | $0.1153 |
| `ri_diagnosis_20261113_105643` | mapping, groups, relay copy, state evaluation; D's fluid unit stalled (zombies) and was stopped | 123.9 | $2.2939 |
| `ri_diagnosis_20261113_130551` | time-boxed measurement of W, DW, D (thinking mode, fluid; descriptive, no unit sealed whole) | 160.3 | $2.9665 |
| `ri_diagnosis_20261113_161911` | D, thinking mode, fluid, concurrency 12: starved by the zombies; parts sealed | 90.9 | $1.6815 |
| `ri_diagnosis_20261113_180718` | D, thinking mode, fluid, concurrency 7: same; parts sealed; the abort defect found | 71.4 | $1.3217 |
| `ri_diagnosis_20261113_193508` | abort fixed; every remaining unit of D, W and DW in both modes | 625.5 | $11.5765 |
| **total** | | | **$19.9554** |

The study's ceiling was $35 (grant `logs/budget/approvals/stage1_reasoning_inheritance_grant.json`, $24 per session,
L40S at $1.09 per hour). **The maintainer closed RI-S1 on 2026-11-14** (the grant's `closed` block): $19.9554 spent,
and the unspent $15.0446 lapsed; it is not carried to the next round, which needs its own study and grant.
`study_engineering.require_open` now refuses any issue or launch of RI-S1. Every attempt is closed with the provider's
confirmation that its pod is gone. The closeout itself cost $0: no pod was created (below).

Storage, by the maintainer's decision of 2026-11-14 (`logs/budget/decisions.md`): the necessary checkpoints are kept
on the dev box only.

* **Kept, on the dev box:** D, W and DW (DW is also the packed Top-1), the mapping, the teacher's tokenizer, chat
  template and configs, and the source records, under `~/aad-artifacts/stage1_reasoning_inheritance/913433664fce0fed/`
  (record `logs/stages/stage-1/reasoning_inheritance/storage/local_checkpoints.json`). One physical copy, on one disk.
* **Relay:** the weight copies were permanently deleted after the local copy was verified (tombstones
  `relay_20261114_ri_s1_*`, 8.44 GiB freed). The mapping, the group configs and the evidence stay there: every sealed
  generation of every unit is in the final attempt's store on the dev box and in
  `durable/stage1_reasoning_inheritance/evidence/ri_diagnosis_20261113_193508/c69706526086bdf3/evidence.tgz` (sha256
  `c69706526086bdf37600edd813ec2105596136abce11872a17c985344dc4c6dd`, downloaded again and re-hashed). The analysis
  binds each scored directory by content digest (`structural_ablation.json :: inputs`).
* **Volumes:** the programme's three network volumes (`thprcgh7s9`, `4rhubprswo`, `am5m373kuq`; not this study's) were
  deleted on 2026-11-14 at 10:46Z. The provider read back no volume and no pod; each storage record carries what went
  and what was kept. The one small record that had looked uncopied (Research A v1 unit 6's training records) was
  already on the dev box, so no pod was needed to read a volume. Provider balance afterwards: $16.63, spend
  $0.00 per hour (`logs/budget/provider/runpod_billing_v2_2026-11-14T1048Z.json`).

## 11. Next round (evidence, derivation, hypothesis kept apart)

Status: a recommendation for the next round's plan, not an authorization. Nothing below starts a paid experiment or
recovery training until the maintainer approves a plan with its own study and grant.

**Evidence from this round.**

* Generation, both modes: depth pruning alone, untrained, removes nearly all of the measured fluid reasoning (T-D
  +84.0 / +61.1 pp static, +43.1 / +34.0 pp interactive), in short answers as well as long ones; the non-depth
  compression alone, and both together, are at the floor on every formal unit.
* Distribution (state evaluation): the non-depth compression moves the root distribution about ten times as far as
  the depth pruning (KL 6.45 vs 0.65); removing the blocks on top of it adds 0.32 nats.

**Derivation.**

* The depth axis is not cheap: its KL is small, its generation cost is nearly total. A method that restores some of the
  8 removed blocks' function can be measured on D (28 blocks at the teacher's width), where both the state evaluation
  and eval-set-1 have range. On DW, eval-set-1 correctness registers only an improvement large enough to lift DW off
  the floor; a smaller one is still measurable there by graded measures (the state evaluation per axis,
  teacher-forced measures), not by generation accuracy.
* The non-depth compression moves the distribution furthest and binds a 0.6-0.7B student; until its damage is
  decomposed, eval-set-1 correctness cannot tell W- or DW-like students apart while they stay at the floor.

**The three merge variants (to test; none is claimed to work).**

1. *Direct parameter addition or averaging* of a removed block into a neighbour. Two residual blocks compute
   `h + f_l(h) + f_{l+1}(h + f_l(h))`; one block with summed weights computes neither that nor the parallel
   approximation `h + f_l(h) + f_{l+1}(h)`: RMSNorm weights differ per block and must be folded into the projections
   first; summed query and key projections add cross terms `Q_l K_{l+1}^T` to the attention logits; SwiGLU's gate is
   nonlinear in its weights; neurons and heads of different blocks are not aligned, so a sum mixes unrelated features;
   averaging fixes the scale only. No calibration, seconds to build: the cheap baseline.
2. *Activation-weighted merging*: the merged block's projections fitted to reproduce the pair's output on calibration
   activations (least squares per projection, or the pair's causal KL against the parent in the packed context the
   DEPTH operator already uses). Handles scale and part of the alignment; the nonlinearity limits it. No training.
3. *Static learnable fusion weights* (one coefficient per matrix or per channel, trained with KD): few new parameters,
   but a recovery-training result, not an initialization one, so it belongs to Stage 3.

**Recommendation.**

1. **First, on the depth axis at full width:** build variants 1 and 2 for the 8 removed blocks on D's K and compare
   them with outright removal (D) under the state evaluation (minutes on one GPU), then on eval-set-1's fluid unit in
   both modes only for the variant that moves the state evaluation (eval-set-1 never chooses). Variant 2 is the main
   candidate; variant 1 is its control. Variant 3 follows only in a Stage 3 recovery comparison against recovered D.
2. **In parallel, at $0 to a few dollars:** decompose W's damage with the same frozen mapping -- FFN only, residual
   width only, query heads only, and their pairs -- by state evaluation, and re-allocate a fixed parameter budget
   toward the factor that costs least.
3. For untrained Stage-1 candidates, read reasoning retention first from graded measures without a floor (the state
   suite per axis, or teacher-forced measures on teacher-native trajectories of a non-formal set); keep generation on
   eval-set-1 for candidates above the floor.
4. Hypothesis for Stage 6 (recurrent depth, AGENTS.md 4.8): if depth carries this much of the reasoning at a small
   KL, re-executing shared blocks may recover depth without its parameters. D is the natural comparison point:
   same width, fewer distinct blocks. Not tested here.

All of this is conditional on the fixed scheme of this round (one mapping, one budget); none of it shows depth
compression to be better or worse than width compression at an arbitrary parameter budget.

## 12. Corrections to the first version (2026-11-14, after commit `09b48be9`)

Re-aggregated from the same sealed generations: no generation, scorer or score changed (every number of the
earlier tables is reproduced, the inputs bound by the same content digests), and no GPU was used.

* D in thinking mode: 32 / 66 is correct among **usable** rollouts. It is not 32 of 66 natural terminations: D ends
  naturally 78 times, and 12 of those break the think protocol.
* D in non-thinking mode: of its 308 usable static replies, 45 are correct, 101 give an answer read and wrong, and 162
  state no final answer the registered scorer can read. The first version called the 263 unscored replies "wrong
  answers given in a well-formed short reply". The failure types now separate an answer read and wrong, no answer
  the scorer can read, a protocol error, a degeneration stop and the context limit, for the static fluid unit and for
  every capability unit (table "What every row ended as").
* The prediction's explanation: "long derivations break down" is now stated within the evidence. Long-generation
  instability does not explain all of the damage, and it is not excluded as part of it. The criterion compared an
  interval with a point estimate; it is not a cross-mode significance test.
* The abort repair: the claim is limited to the sealed records (no continuation found in any of the 8,881
  degeneration stops). It no longer reads as a guarantee that a rerun before and after the repair would reproduce
  each sample and score.
* Teacher reuse: the capability units' checks had silently skipped the item-file and seed fields, which their
  summaries do not carry. Every unit is now checked item by item (coverage, every seed against the rule, every
  prompt against this round's), and what was never recorded is listed as such. The runner note now describes the
  runners as they were, after the abort repair (`teacher_reuse.json :: runner_note`).
* Bootstrap: the intervals resample rule families, while the design registered the independence groups. The
  registration is unchanged, and both are now reported.
* Wording: KL no longer "accounts for almost all of the damage", and W/DW are no longer called unmeasurable for any
  improvement. Eval-set-1 correctness is floor-limited there; graded measures are not.
* Storage and closure (section 10): the checkpoints moved from the relay to the dev box, the volumes are retired, and
  the study is closed.

