# Stage 1 reasoning inheritance: structural ablation of the packed Top-1 (RI-S1)

**Status: COMPLETE; study CLOSED** (2026-11-14 UTC, closed by the maintainer). The frozen mapping and the four
groups are built and verified, and D, W and DW are kept on the dev box (section 4); the state evaluation is measured;
every registered eval-set-1 unit is generated and scored for D, W and DW in both modes, and the teacher's evaluation is
reused after every identity check, item by item. No recovery training, no new search. Total cost $19.9554 over six
attempts of the $35 ceiling; the unspent $15.0446 lapsed at closure (section 10). Corrections after commits
`09b48be9`, `b191d555` and `bf577e3d` are listed in section 12; section 11 is the next round's plan,
prepared to the launch point and not authorized.

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
  fluid turns, 996 capability and legacy requests per mode); and every item's prompt fields equal this round's D, W
  and DW for the same item. A field absent or invalid on either side counts as not recorded, never as equal (none
  was). The prompt's fields and its content are kept apart:
  * fluid unit: the records carry the prompt's text, and the first user message equals this round's and the
    registered item's -- **content verified**;
  * every other unit: the records carry only the prompt's length, system source and think pre-opening (no text, no
    token ids) and the summaries only a pod path for the prompt file (`not_recorded`). Content identity is **derived,
    not compared**: the teacher's session commit (`f93d83a7`) and this round's (`8f74bc94`) have the same registered
    item files, staging code and rendering statements (`teacher_reuse.json :: construction_path`), and rendering
    every registered item on the dev box with the teacher's verified tokenizer reproduces every recorded prompt
    length, system source and think pre-opening, the teacher's and D's, W's and DW's. Equal lengths alone would not
    establish it.
* Intervals: 95% percentile intervals over 10,000 cluster-bootstrap resamples (seed 20261113): fluid items are
  resampled by whole rule family (86 clusters), paired units by pair, other units by item. The design registered the
  eval set's independence groups for the fluid unit instead (`independence_v1.json`: only the six groups of renamed
  knights-and-knaves puzzles are merged, 708 clusters). The family clustering was chosen when the analysis was
  written; the registration is left as it was, and both are computed (table "Fluid intervals under the registered
  independence grouping"): the estimates are the same; the family intervals are the wider on 11 of the 12 fluid
  T-D contrasts, but not on non-thinking-mode task-contained deduction (independence groups [+61.9, +75.4], families
  [+63.2, +75.0]); the retention intervals differ in both directions. The family clustering is a post-hoc analysis
  choice; the independence grouping is the registered one. They cover item and family sampling, not decoding variance (one
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
  below. D's GSM8K 0 in non-thinking mode, over all 100 items: 49 state no final answer the registered scorer can
  read, 49 are degeneration stops, 1 gives an answer read and wrong, 1 breaks the think protocol. Only within the 50
  well-formed replies is it 49 of 50 without a readable answer, so over the unit the failure is half degeneration and
  half no readable answer, not mainly extraction. A reply without a readable answer is shown neither wrong nor right;
  the registered scorer counts it as not correct. T's own non-thinking GSM8K has 55 of 100 replies without a readable
  answer, which is why T is at 38% there; in this mode the unit measures answer format as well as arithmetic.
  Descriptive, across mixed-domain supporting units, never read as fluid reasoning (P10.2); in direction it is
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
* **Other weights on the relay (reviews of 2026-11-14):** Research A's last Stage-3 relay copy
  (A-MIX-6cat-sr-s20260801, 2.22 GiB) had no consumer and was retired under AGENTS.md 2.5 (tombstone
  `relay_20261114_stage3_sr_A-MIX-6cat-sr-s20260801`, relay commit `92f9f4fc`; its config files stay). The remaining
  weight files of earlier rounds were then audited object by object (`scripts/maintenance/consolidation/
  audit_relay_weights.py` -> `logs/maintenance/inventories/relay_weight_audit_20261114.json`: registry protection,
  every declaring launcher and its study's state, every code reader classified, shared LFS objects). 24 of them
  (38.86 GiB) were retired, each tombstoned first, deleted and read back (`relay_20261114_*`): six E1 scaling students,
  twelve Stage-0 calibration-volume finalists, the D3 structural product and five early Stage-3 recoveries -- all
  terminal, their evaluation evidence beside them or in the logs, no open or proposed consumer. Six are kept, each for
  a named use: the 0.6B initialization and its random-initialization control (protected anchors), the two permanent
  controls (protected), D4-Augmented's D4 Top-1 (the declared input of the independent-gated study, whose proposal is
  pending, and of Research B's selection), and RI-S1's frozen mapping. Relay LFS: 64.16 -> 25.30 GiB.
* **Volumes:** the programme's three network volumes (`thprcgh7s9`, `4rhubprswo`, `am5m373kuq`; not this study's) were
  deleted on 2026-11-14 at 10:46Z. The provider read back no volume and no pod; each storage record carries what went
  and what was kept. The one small record that had looked uncopied (Research A v1 unit 6's training records) was
  already on the dev box, so no pod was needed to read a volume. Provider balance afterwards: $16.63, spend
  $0.00 per hour (`logs/budget/provider/runpod_billing_v2_2026-11-14T1048Z.json`).

## 11. Next round: layer merging at D's fixed structure (RI-LM1: prepared to the launch point, not authorized)

Status: prepared at $0 -- design, code, tests, launch commands and a budget draft -- for the maintainer to decide.
Nothing below starts a paid experiment, recovery training or a merge into `main`: the study `RI-LM1` holds a proposal
(`logs/stages/stage-1/reasoning_inheritance/plans/layer_merge_proposal.json`, `status: proposed`) and no grant. The
experiment is `configs/stages/stage-1/reasoning_inheritance/layer_merge/design.json`. Revised after the reviews of
2026-11-14 (section 12).

**The question.** At D's structure -- 28 blocks, hidden 2560, FFN 9728, 32 query / 8 KV heads, 3,215,021,568
parameters, the teacher's embedding, final norm and tied LM head -- does writing each span's one block from all of the
span's teacher blocks do better on eval-set-1 than keeping the span's first block alone (D)? Only six blocks' values
change; no new depth or width allocation is needed first, and D is the direct-deletion control.

**What this round's evidence does and does not support.**

* Generation: removing 8 of 36 blocks, nothing else changed, takes static fluid from 89.6% to 5.6% (thinking) and
  from 68.9% to 7.9% (non-thinking), interactive episodes to 0. D sits above 0 and far below T in both modes, so a
  merge that helps or hurts can be measured on eval-set-1.
* State evaluation and teacher-forced NLL/KL measure how far the next-token distribution moved on reference
  trajectories (distribution and trajectory fidelity). They are diagnostics, not a measured reasoning retention: D's
  KL of 0.65 sits beside an 84 pp generation loss, so a lower mean KL does not by itself mean better reasoning, and an
  unchanged one does not rule a merge out.
* D, W and DW have different parameter counts (3.215B, 0.722B, 0.596B). They describe the damage of this one fixed
  compression scheme. They do not show that depth pruning keeps reasoning better or worse than width pruning, or that
  the non-depth compression is the dominant bottleneck of every 0.6-0.7B structure (a separate study, below).

**Spans (all candidates).** Each removed block joins the nearest kept block before it, so the merged block stands
where its span starts and receives the span's input: the earlier-layer convention of LaCo's merge (Yang et al. 2024,
Eq. 1) and the teacher-span-to-student-layer mapping of AGENTS.md 4.3. The table is derived from the frozen
selection's kept blocks (`layer_merge.merge_spans`) and checked against the design.

| kept block (root numbering) | absorbs | span |
| ---: | --- | --- |
| 2 | 3 | [2, 3] |
| 10 | 11 | [10, 11] |
| 16 | 17, 18 (consecutive) | [16, 17, 18] |
| 24 | 25, 26 (consecutive) | [24, 25, 26] |
| 29 | 30 | [29, 30] |
| 31 | 32 | [31, 32] |

A kept block that receives two removed blocks (16, 24) combines its whole span in one step, never pairwise in
sequence. The other 22 blocks, the embedding, the final norm and the LM head are D's, checked bit for bit.

**Candidates (fixed with the plan; neither the list nor the decision rule changes after results are seen).**

1. **D, direct deletion.** The existing checkpoint (`9de449db...`, dev box) and its sealed eval-set-1 records: not
   rebuilt, not regenerated, never moved to the pod (the pod compares against D's per-tensor digests).
2. **M-avg, direct parameter averaging (the control).** For every tensor of a span's blocks (`input_layernorm`,
   `q/k/v/o_proj`, `q_norm`, `k_norm`, `post_attention_layernorm`, `gate/up/down_proj`; Qwen3 has no biases):
   `theta' = (1/|span|) * sum_{l in span} theta_l`. No folding, no alignment, no rescaling. It is the unaligned
   ("vanilla") average of Verma et al. (2025) and the equal-weight case of MKA's
   `theta~ = alpha * theta_l + (1 - alpha) * theta_m` (Liu et al. 2024).
   *Parameter addition is not registered.* LaCo's `theta*_l = theta_l + sum_k (theta_{l+k} - theta_l)` has no
   scaling: on the four two-block spans it reduces to `theta_{l+1}`, i.e. keeping the absorbed block instead of the
   kept one (a different deletion, not a merge), and summing every tensor doubles the RMSNorm gains, with no scaling
   rule in a primary source.
3. **M-act, aligned activation-weighted merging -- a heuristic initialization to be tested.** Per span, on the
   teacher's parameters:
   * *Fold the gains (exact in real arithmetic).* Each block's `input_layernorm` gain into its q/k/v input columns
     and its `post_attention_layernorm` gain into its gate/up input columns (Qwen3's RMSNorm multiplies by the gain,
     so `W (g * x_hat) = (W diag g) x_hat`); the merged block's two gains are 1. `q_norm`/`k_norm` act after the
     projection and are not folded.
   * *Align each absorbed block onto the span's first block (an exact re-parameterization).* FFN: neuron i is
     (gate row i, up row i, down column i), its feature the gated product `a_i`, the down projection's input; the
     permutation maximizing the summed Pearson cross-correlation between the two blocks' features, by linear
     assignment. Attention (GQA, 8 KV groups of 4 query heads): KV groups matched by linear assignment on the cosine of
     the groups' summed residual writes `sum_h W_o,h a_h` (2,560-d, the coordinates every block shares), then the
     query heads within matched groups the same way; q/k/v rows and `o_proj` columns move together, head dimensions
     never (RoPE pairs them).
   * *Weight by activation importance.* After alignment each FFN slot combines its span's neurons with
     `w_{l,i} = s_{l,i} / sum_l s_{l,i}`, `s = E|a| * ||down column||_2` (the FFN operator's importance); each
     query-head slot with the head's write energy `mean_t ||W_o,h a_h(t)||^2` (the attention operator's statistic);
     each KV group's k/v rows with its four heads' summed energy; `q_norm`/`k_norm` with each block's share of total
     head energy.
   * *What the weights do not do.* They are convex per slot. That guarantees neither the merged block's output scale
     nor that it reproduces the teacher span's cumulative update: a two-block span applies two residual updates in
     sequence, the merged block one. The first version of this plan said the weights keep one block's output scale;
     that was an assertion, not a property, and is withdrawn. The size of the merged update against the span's is
     measured (diagnostics, below), not assumed.
   * Not part of M-act: a least-squares refit of the output projections to the span's update, and learnable fusion
     coefficients (KD training: Stage 3). No fitting, training or further candidate is added for this round.

**How M-act relates to the literature (corrected).** Verma, Murray and Duh (arXiv:2501.06126v2) align FF sublayers by
the linear assignment on Pearson cross-correlation, average the aligned sublayers (`1/k`), then -- section 3.3 --
*replace each of them with the average and tie the weights*: the model keeps executing k FF sublayers, one at each
original position between that position's own attention sublayer, with one parameter set in memory. That reduces
parameters, not executed depth or FLOPs. This round removes the absorbed blocks from execution (28 blocks run, D's
depth), as LaCo and MKA do; it borrows the alignment, not the execution scheme. Their Appendix A already treats SwiGLU
(used for OLMo-7B, section 4.5): the feature is the gated product's output and one permutation moves the up and gate
rows and the down columns together -- exactly M-act's FFN alignment, not an extension made here. What is new here is
merging whole blocks (attention included; the paper merges FF sublayers only and reports that attention outputs lack
the FF outputs' similarity), the gain folding, and the importance weights; those are untested. Parameter sharing with
repeated execution is Stage 6's subject and is recorded there as related work (`LITERATURE.md`).

**Statistics (exactly).** One teacher forward over `calib.hybrid_d4_base@v2` (558 examples, 569,270 tokens before
packing), in the anchor search's own session (its operator config hashes are checked against the journal first), at
the FFN operator's own positions: `positions.supervised_target_v1@v1` (211,203 positions admitted before packing), under
Stage 1's packing (seeded sha256-keyed shuffle 20261105, sequential packing with truncation into 8,192-token sequences,
`prefix_restart_v1`, causal attention, continuous positions). That is 79 packed sequences, 644,493 non-padding tokens
and 231,242 statistics positions (a fragment's repeated prefix counted like any placement, P19); the driver refuses
other counts. Each block's statistics are taken on its own input in the teacher's forward, for the 14 blocks of the six
spans. Accumulation is streaming, one packed sequence at a time: cross products in float32 of the bf16 activations
(exact products; head writes in chunks of 1,024 positions), every sum across sequences in float64 (sum x, sum x^2,
sum y, sum y^2 and `sum x y^T` per pair; the write Gram per pair and per block): 8 x 9,728^2 float64 = 6.1 GB of device
memory beside the 8 GB teacher. Fixed rules:

* *Zero variance:* a feature whose variance on the statistics positions is at or below 1e-12 of its mean square (or
  whose mean square is 0) has correlation 0 with every feature; a zero-energy head set has cosine 0 (never NaN).
* *Zero denominator:* a slot whose importance or energy is 0 in every block of its span takes the uniform weights
  `1/|span|`; the count of such slots is recorded.
* *Assignment and ties:* an exact maximum-weight perfect matching (`layer_merge.linear_assignment`: the shortest
  augmenting path with dual variables of Crouse 2016, the algorithm `scipy.optimize.linear_sum_assignment`
  implements; scipy is not in the pinned environment, so it is implemented in float64 torch, checked against brute
  force, ~48 s for a hard random 9,728 x 9,728 matrix on the dev box CPU); among columns of equal reduced cost an
  unassigned column first, then the lowest index.
* The statistics the merge consumes (per block E|a| and head energies, per pair the three permutations) are saved as
  soon as they exist (`records/merge_statistics.safetensors` and `.json`); a later attempt reuses them.

**Numerical checks (the build stops on any failure; a failure is a defect of the mechanism).** Folding alone and each
absorbed block's permutation alone are checked against the unmodified block in float32 copies of the bf16 teacher
blocks, on every position of the first packed sequence: `||y' - y||_F / ||y - x||_F` (relative to the block's own
update) at most 1e-4, where float32 reordering and rounding give about 1e-6. Every candidate tensor outside the six
span blocks must equal D's per-tensor digest (`layer_merge/d_tensor_digests.json`, 310 tensors from D's checkpoint on
the dev box), every span block must differ from D's, the configuration must be D's and the parameter count
3,215,021,568. D itself is first rebuilt from the teacher with D's own builder and compared tensor by tensor (it is the
base; this is a tensor identity, not a file-hash rebuild of D). Model dimensions, GQA grouping and spans are read from
the configs and the selection; `src/aadistill` holds no instance value.

**Diagnostics (explain, never select).** On the statistics positions, for each span: the teacher span's attention
update (sum over its blocks), FFN update (sum) and whole residual update (span output minus input), against D's,
M-avg's and M-act's one block run on the same teacher input -- RMS norms, norm ratio, cosine and normalized error
`E|c - r|^2 / E|r|^2` per component (`records/merge_diagnostics.json`). Like state evaluation, they gate no generation
and decide nothing.

**Measurements.**

* eval-set-1, every registered unit, both modes, RI-S1's protocol unchanged (the driver refuses another: sha256
  `5e93555e...` of RI-S1's `design.json :: evaluation`): the runners as at session commit `8f74bc94` (after the abort
  repair), seeds from item, turn and sample (each item pairs with D's sealed record), context 40,960 with no token cap,
  the degeneration check every 256 tokens, concurrency 7 (D's: same structure, same KV cache). Order: both
  candidates' thinking-mode fluid unit first, then thinking-mode capability, then non-thinking, so a budget that ends
  early leaves the primary comparison measured. Both candidates are generated whatever the state evaluation or the
  diagnostics show. eval-set-1 stays development-exposed and is not presented as a new blind test.
* Reported apart, never pooled: correctness per unit and axis by the registered scorers; usable rollout and its
  components; natural termination, degeneration stops, context-limit hits; failure types; teacher-behaviour fidelity.
* State evaluation (`state_eval_hybrid@v3`, against the root) for each candidate, as a diagnostic.

**Decision rule (registered; `layer_merge_analysis.py` executes it).**

* *Primary comparison:* thinking mode, the static fluid formal core (570 items), M-avg - D and M-act - D, each item
  paired with D's sealed record of the same item (all rows rescored with the current scorer).
* *Units of independence:* the 570 items fall into 558 registered independence groups (`independence_v1.json`: six
  groups hold 18 items, every other item is its own group). 708 is the whole fluid unit's count (720 items) and is
  not the static comparison's.
* *Statistic:* `delta = (1/570) sum_items (y_M - y_D)`, y the registered scorer's correctness; `d_g` the summed
  difference of group g.
* *Test:* two-sided paired sign-flip randomization at the group level. Under H0 the two models' labels are
  exchangeable within every group, so each `d_g`'s sign is flipped independently with probability 1/2;
  `T = |sum_g d_g|`; `p = (1 + #{b : |sum_g e_bg d_g| >= T}) / (1 + B)`, B = 100,000 draws of `e_bg` in {-1, +1} for
  every group in sorted group order from `numpy.random.default_rng(20261114)`.
* *Direction:* two-sided; the sign of `delta` says better or worse.
* *Multiplicity:* Holm over the two candidates at alpha 0.05: p sorted ascending, adjusted
  `p_(1) = min(1, 2 p_(1))`, `p_(2) = min(1, max(adjusted p_(1), p_(2)))`.
* *Decision:* BETTER than deletion if the Holm-adjusted p <= 0.05 and `delta` > 0; WORSE if <= 0.05 and `delta` < 0;
  NOT DISTINGUISHED otherwise. A candidate whose primary unit is not complete has no verdict (`not_measured`); partial
  units are never pooled.
* *Effect interval, beside the test and labeled as such:* the percentile 95% interval of `delta` over 10,000
  group-bootstrap resamples (the 558 groups with replacement, `numpy.random.default_rng(20261115)`, numpy's linear
  percentile); unadjusted for multiplicity and not the decision. The rule-family interval RI-S1 reported is computed
  beside it as a supplement.
* *Secondary, reported and not decisive:* the non-thinking mode, interactive episodes, each fluid axis, the supporting
  units, M-act - M-avg (which differs in folding, alignment and weighting at once).
* At D's 5.6%, about 3 pp is roughly the smallest paired difference this design separates from 0 (a rough figure
  from paired discordance, not a power analysis).

**Reuse, storage and failure (arranged before launch).**

* *Reused, never redone:* D's checkpoint and sealed generations; T's reused evaluation; the frozen selection and
  mapping; the calibration and state-suite items (dev-box assets); eval-set-1; RI-S1's runners and its evaluation,
  sealing and resume paths. No corpus is regenerated, no baseline retrained, no valid measurement repeated.
* *Candidates (2 x 6.43 GB):* to the private relay from the pod as soon as built, under
  `durable/stage1_reasoning_inheritance/layer_merge/<statistics sha256[:16]>/groups/{M-avg,M-act}/`, downloaded again
  into a fresh cache and re-hashed. Before the launch the relay is asked whether it would accept both and the
  statistics at their true sizes (`the_relay_has_room`, the LFS batch endpoint, nothing uploaded): relay LFS is
  25.30 GiB after this round's retirements. They are not copied to the dev box by default (15 GB free on a shared
  disk); the relay copy is their durable copy until a later consumer asks for them.
* *Statistics and evidence:* every finished file, the statistics included, is pulled to the dev box on each poll
  (`<scr>/store`; about 0.4 GB for a full evaluation of two models, from RI-S1's 0.37 GB for three); the statistics also
  go to the relay with the candidates; the final store is archived to the relay at closeout (`archive-evidence`).
* *A failed attempt* takes over the previous attempt's records -- statistics and permutations, builds, checks,
  diagnostics, state evaluation -- and every sealed evaluation part (`driver.take_over`), restores the candidates from
  the relay by sha256 (`driver.relay_restore`), and generates only the items with no sealed record, each under its
  fixed seed. Nothing is rebuilt, re-collected or regenerated that a record or a sealed part holds.

**Budget (a draft for the maintainer; `layer_merge_proposal.json`).**

* *Scenario estimate, from RI-S1's measurements, not a measurement of these candidates:* setup 25 min; build about
  45 min (teacher download and hashing, the statistics forward over 79 packed sequences, eight 9,728 x 9,728
  assignments, the checks, both builds and saves, the relay copy); diagnostics and state evaluation about 20 min;
  generation 1.8 h per candidate if it behaves like D, 3.6 h if it behaves like T. Total 4.8-8.4 h on one L40S at
  $1.09/h: about $5-9 with container disk.
* *Caps proposed:* sessions may draw $13.00 cumulatively over every attempt (the ceiling; each closeout's all-in, an
  unknown cost refuses); $1.00 teardown reserve held outside the ceiling and never issued (every artifact's hard cap
  plus the reserve must fit what the project cap and the provider balance leave, at issue and again at launch);
  study cap $14.00; per session $11.00 all-in to the hard deadline plus the watchdog's teardown time, the deadline
  derived from the accepted price; one L40S at no more than $1.09/h, one pod.
* *Own envelope:* RI-S1 is closed and its unspent $15.0446 lapsed; it funds nothing here. The new study's envelope
  counts only its own runs. Live position at 2026-11-14T20:57Z: $963.24 committed of the $1,000 project cap
  ($36.76 left), provider balance $16.63, no pod, no volume -- one session plus the reserve fits the balance; a second
  full session would need a top-up, which is the maintainer's.
* *Terminal state:* BUDGET_EXHAUSTED when what the ceiling leaves is below the smallest useful attempt ($2.00: setup,
  the restore and at least one fluid unit of one candidate) or the next cap plus the reserve does not fit: the issuer
  refuses; every finished unit is kept and every other reported `not_run`; the primary comparison is evaluated only
  if both candidates' primary unit is complete (otherwise no verdict); the closeout records the spend and the remainder
  lapses at closure.

**Launch, once granted** (`scripts/stages/stage-1/reasoning_inheritance/README.md` has the full commands): write the
grant from the proposal (`layer_merge_launch.py grant --decision '<quoted>' --granted-by maintainer`); commit; `issue`
the one-use artifact (`--expected-usd 7.5 --hard-cap-usd 11`); commit it (that commit is the session commit); stage
its bundle; `launch --minutes 480 --max-price 1.09 --disk-gb 150 --driver-extra '--max-num-seqs 7'`; `closeout`;
`layer_merge_analysis.py`. A $0 dry run of `issue` and `launch` against the proposal exercises every gate before any
of it (section 12).

**A separate study, not a prerequisite: depth against width at a common budget.** Which axis -- depth, FFN width,
residual width or query heads -- loses the least reasoning per parameter needs structures at a common parameter budget,
with parameters, training and inference compute, and deployment latency reported apart. D, W and DW cannot answer it.
Decomposing W (FFN only, residual width only, query heads only) can still locate where this scheme's damage comes
from. Nothing here materializes a model or starts a search for that study.

**Elsewhere.** Learnable fusion coefficients trained with KD: Stage 3. Shared-parameter recurrent execution and its
training: Stage 6 (AGENTS.md 4.8), with Verma et al.'s align-average-tie scheme recorded as related work
(`LITERATURE.md`) and D a natural comparison point (same width, fewer distinct blocks).

References (read for this plan; `LITERATURE.md`): LaCo, Yang et al. 2024, arXiv 2402.11187 (Eq. 1, merge into the
earlier layer); Verma, Murray, Duh 2025, arXiv 2501.06126v2 (sections 3.2-3.3 and Appendix A: alignment by Pearson
cross-correlation and linear assignment, the SwiGLU feature, uniform averaging, weight tying at every original
position); MKA, Liu et al. 2024, arXiv 2406.16330 (weighted pair merge, alpha from a similarity). The
assignment algorithm is the one `scipy.optimize.linear_sum_assignment` implements and cites (Crouse 2016,
IEEE TAES 52(4)); that paper itself was not read, and the implementation is checked against brute force.

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

### Second review (2026-11-14, after commit `b191d555`)

Again from the same sealed generations, with no generation, scorer or score changed and no GPU used.

* Teacher reuse, non-fluid units: the prompt comparison compared `(prompt_tokens, system_source, think_preopened)`
  tuples, so two records that both lacked the fields compared equal and passed. Every field is now validated on both
  sides (`False` is a value; an absent or invalid field is not recorded and never equal), the prompt's fields and its
  content are reported apart, and the content identity of the non-fluid prompts is stated as derived (same item
  files, staging and rendering statements at both session commits; the registered items rendered on the dev box
  reproduce every recorded field), never as compared. Regression: `scripts/stages/stage-1/reasoning_inheritance/tests/`.
  On the sealed records no field was missing and every check passes; T stays reusable, its rescored content unchanged.
* D's non-thinking GSM8K: over all 100 items, 49 without a readable answer, 49 degeneration stops, 1 answer read and
  wrong, 1 protocol error. The first correction's "mostly extraction" described only the 50 well-formed replies.
* Bootstrap: the family intervals are the wider on 11 of the 12 fluid T-D contrasts, not on every one
  (non-thinking-mode deduction is the exception).
* Section 11 is rewritten: no state-metric gate before generation (the state evaluation is a diagnostic, never called
  reasoning retention); the next round is a layer-merging comparison at D's fixed structure with two candidates fixed
  in advance; the depth-against-width comparison at a common budget is a separate study, not a prerequisite.
* Tombstones: `retire_artifact.py` set `deterministic_reconstruction_possible` from the presence of a recipe, so the
  eight retrained Stage-3 students read as deterministically rebuildable. The tool now records the recipe's kind,
  whether it gives back the same bytes, and any same-hash copy apart; this round's entries are corrected in place
  (the eight retrains and D's and W's unverified rebuilds false, DW's rebuild verified true), the original values and
  the weights' hashes kept.

### Third review (2026-11-14 UTC, after commit `bf577e3d`)

No generation, scorer, score or legitimate checkpoint changed; no GPU used.

* **M-act's interpretation.** The plan said the merge weights, summing to 1, keep one block's output scale. Convex
  weights per slot guarantee neither the output scale nor the teacher span's cumulative update (a span applies two or
  three updates in sequence, the merged block one); the assertion is withdrawn. M-act stays a heuristic initialization
  to be tested, with update norms and normalized reconstruction errors of the attention, FFN and whole-span updates
  measured on the calibration positions as diagnostics only (section 11). No fit, training or candidate is added.
* **Literature.** Verma et al. v2 reread in full: section 3.3 ties the averaged parameters and keeps executing them at
  every original FF position (parameters shared, depth kept), where this round removes the absorbed blocks from
  execution; Appendix A already treats SwiGLU with the gated product as the feature and one permutation for up, gate
  and down, and the main text includes OLMo-7B. `LITERATURE.md` no longer calls the SwiGLU treatment an extension made
  here; the parameter-sharing scheme is recorded as Stage 6 related work.
* **Decision rule.** Written as an executable definition (section 11; `layer_merge_analysis.py`): a two-sided paired
  sign-flip test over the 558 independence groups the 570 static items fall into (708 is the whole fluid unit's),
  its p-value, Holm over the two candidates, and the group-bootstrap percentile interval reported apart as an
  unadjusted effect interval, never "a 95% percentile CI after Holm".
* **Numerical boundaries.** Fixed rules for zero-variance correlations, all-zero importance denominators and
  assignment ties; the statistics positions stated exactly (79 packed sequences, 231,242 positions) with float64
  streaming sums; folding alone and each permutation alone checked for forward equivalence in float32 (tolerance 1e-4
  of the block's update), the 22 unchanged blocks bit for bit against D's per-tensor digests; dimensions, GQA grouping
  and spans read from the configs and the selection (`src/aadistill/initialization/planning/layer_merge.py` holds no
  instance value).
* **Retirement tool.** `retire_artifact.py`'s same-hash-copy check skipped a path that did not exist, compared
  nothing for a directory, and did not exclude a copy inside the deletion scope. Every declared copy is now verified
  (`verify_copies`): it exists, it matches member by member, it lies outside the deletion scope (the target, a path
  inside it, a directory containing it, a hard link, or -- for a relay target -- the same repository's LFS object), and
  a remote copy is checked by the object the hub serves. A verified rebuild without a recipe is refused. The tool can
  now retire one relay object (tombstone first, exact `file_oid`, read back).
* **Relay weights.** The 29 remaining weight files were audited object by object: 24 retired, 6 kept with their uses
  (section 10).
* **D and W.** Their byte-identical rebuild is *not yet verified* -- each was built once and no second build was
  compared -- which is not the same as impossible; their tombstones say so, and nothing is rebuilt to settle it.
* **Next round.** Prepared to the launch point (section 11): design, code, tests, launch commands, storage, and a
  budget draft with its own ceiling, teardown reserve and terminal state; not authorized.
* **Record dates.** This round's first records were stamped 2026-11-15 from the dev box's +08:00 clock; they are
  corrected to UTC (2026-11-14). The 24 tombstones keep, in `id_in_relay_commit_message`, the id their relay commit
  messages carry.
