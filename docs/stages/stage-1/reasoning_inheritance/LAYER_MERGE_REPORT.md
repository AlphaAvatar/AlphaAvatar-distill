# Stage 1 layer merging at D's fixed structure (RI-LM1)

**Status: COMPLETE; study CLOSED** (2026-11-15 UTC, at its terminal condition: every registered unit measured).
Two attempts, $4.0989 of the $13.00 ceiling; the unspent $8.9011 and the $1.00 teardown reserve lapse. The plan and
its decision rule were fixed before any result (`REPORT.md` section 11; `configs/stages/stage-1/reasoning_inheritance/
layer_merge/design.json`). eval-set-1 is development-exposed: nothing here is an untouched test.

**Result in one paragraph.** At D's structure (the teacher's 28 kept blocks of 36, untrained), writing each of the six
spans' one block from all of the span's blocks does worse than keeping the span's first block as it is. Direct
averaging (M-avg) takes static fluid reasoning in thinking mode from D's 5.61% to 0.00%; the aligned,
activation-weighted merge (M-act) to 0.35%. Under the registered rule both are **WORSE than deletion** (Holm-adjusted
p = 2e-5 each, over the two registered comparisons; the effect intervals exclude 0). Non-thinking mode, the interactive
episodes and every supporting unit point the same way (descriptive). The diagnostics say why: in every span, D's kept
block alone reproduces the teacher span's residual update more closely than either merged block, and averaging also
shrinks the update to 6-37% of the span's norm. This is a statement about these two training-free merges at this
structure, not about merging with learned coefficients (Stage 3) or about recurrent depth (Stage 6).

## 1. What was compared

* **D** -- direct deletion (RI-S1's group D): blocks 3, 11, 17, 18, 25, 26, 30, 32 removed, nothing else changed. Its
  sealed eval-set-1 records were reused (never regenerated); its checkpoint is the base of both candidates.
* **M-avg** -- for each span (kept block plus the removed blocks after it: [2,3], [10,11], [16,17,18], [24,25,26],
  [29,30], [31,32]) every tensor averaged, `1/|span|`, norm gains included; no folding, no alignment.
* **M-act** -- RMSNorm gains folded into the reading linears; each absorbed block aligned onto the span's first block
  (FFN neurons by an exact linear assignment on the Pearson correlation of the gated-product features; KV groups,
  then query heads within a group, by the cosine of their residual writes); per slot convex weights from activation
  importance (FFN: E|a| times the down column's norm; heads: write energy). A heuristic initialization, tested here.

The 22 other blocks, the embedding, the final norm and the tied head are D's, checked bit for bit against D's 310
tensor digests; both candidates have D's configuration file (sha256 `4583de76...`) and 3,215,021,568 parameters.

## 2. Primary comparison (registered, formal)

Thinking mode, the 570 registered static items of the fluid formal core (sha256 `40293ab1...`), paired with D's
sealed records item by item, 558 independence groups; two-sided group sign-flip test (B = 100,000), Holm over the two
registered comparisons, the group-bootstrap interval reported apart (unadjusted, not the decision).

| comparison | candidate | D | delta | p (two-sided) | Holm p | 95% group-bootstrap interval (unadjusted) | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M-avg - D | 0/570 (0.00%) | 32/570 (5.61%) | -5.61 pp | 1.0e-5 | 2.0e-5 | [-7.58, -3.82] pp | **WORSE** |
| M-act - D | 2/570 (0.35%) | 32/570 (5.61%) | -5.26 pp | 1.0e-5 | 2.0e-5 | [-7.23, -3.48] pp | **WORSE** |

Both primary units were complete (every registered item has a record), so the rule gives formal verdicts. The p-value
is the test's floor at B = 100,000: no sign flip of the 32 nonzero groups reached the observed sum. The rule-family
intervals, a supplement: M-avg [-8.64, -2.80] pp, M-act [-8.07, -2.63] pp (76 families).

## 3. Everything else (reported, never decisive)

**Fluid reasoning, formal core** (static 570 / interactive 144):

| mode | model | static | interactive | how the 720 episodes ended |
| --- | --- | --- | --- | --- |
| thinking | D | 32 (5.61%) | 0 | 79 answered, 632 degeneration stops, 7 episodes done, 2 turn limits |
| thinking | M-avg | 0 (0.00%) | 0 | 9 answered, 708 degeneration stops, 3 episodes done |
| thinking | M-act | 2 (0.35%) | 0 | 25 answered, 695 degeneration stops |
| non-thinking | D | 45 (7.89%) | 0 | 311 answered, 399 degeneration stops, 6 episodes done, 4 turn limits |
| non-thinking | M-avg | 0 (0.00%) | 0 | 14 answered, 701 degeneration stops, 5 episodes done |
| non-thinking | M-act | 9 (1.58%) | 1 | 99 answered, 620 degeneration stops, 1 episode done |

**Supporting units** (the registered scorer's headline, D / M-avg / M-act; natural-termination rate beside it):

| unit | thinking | natural termination | non-thinking | natural termination |
| --- | --- | --- | --- | --- |
| math_verified (correct) | 5.0% / 0 / 0 | 0.06 / 0.01 / 0 | 9.0% / 0 / 0 | 0.26 / 0.01 / 0.08 |
| gsm8k (correct) | 0 / 0 / 0 | 0.04 / 0.03 / 0 | 0 / 0 / 0 | 0.51 / 0.01 / 0.18 |
| multihop (correct, evidence supplied) | 6.0% / 0 / 4.0% | 0.50 / 0.07 / 0.09 | 28.0% / 1.0% / 2.0% | 0.46 / 0.01 / 0.13 |
| rag (correct, evidence supplied) | 1.0% / 0 / 2.0% | 0.93 / 0.09 / 0.06 | 42.0% / 5.0% / 0 | 0.92 / 0.15 / 0.03 |
| answerability_paired (pair correct) | 0 / 0 / 0 | 0.90 / 0.10 / 0.09 | 31.7% / 0 / 0 | 0.89 / 0.12 / 0.06 |
| safety_paired (pair correct) | 0 / 0 / 0 | 0.12 / 0.01 / 0.04 | 2.0% / 0 / 0 | 0.31 / 0.01 / 0.02 |
| knowledge (TriviaQA; outside the formal core) | 3.3% / 0 / 0 | 0.11 / 0 / 0.02 | 6.7% / 0 / 0.7% | 0.16 / 0 / 0.02 |
| stage3_legacy_150 (in-sample regression, never generalization) | 38.0% / 18.7% / 23.3% | | 38.0% / 17.3% / 25.3% | |

`eval_behavior_v0` has no headline by design. The merges terminate naturally far less often than D everywhere; M-act
stays closer to D than M-avg in most cells and is above D in one (rag in thinking mode, 2 of 100 against 1 of 100).

**State evaluation** (`state_eval_hybrid@v3` against the root teacher; diagnostic only):

| model | equal-domain mean KL | worst domain | top-1 agreement | critical KL at `</think>` | at EOS |
| --- | --- | --- | --- | --- | --- |
| D (RI-S1) | 0.647 | 0.950 | 0.800 | 0.43 | 2.27 |
| M-act | 0.733 | 1.025 | 0.744 | 2.28 | 4.25 |
| M-avg | 1.067 | 1.437 | 0.689 | 2.01 | 7.34 |

**Residual-update diagnostics** (the 231,242 statistics positions; each model's one block on the teacher span's input
against the teacher span's update; normalized error `E|c - r|^2 / E|r|^2`, norm ratio, for the whole block update):

| span | D: error / ratio | M-avg: error / ratio | M-act: error / ratio |
| --- | --- | --- | --- |
| [2, 3] | 0.53 / 0.53 | 0.86 / 0.18 | 0.57 / 0.48 |
| [10, 11] | 0.55 / 0.79 | 0.81 / 0.37 | 0.70 / 0.47 |
| [16, 17, 18] | 0.06 / 0.97 | 0.97 / 0.06 | 0.75 / 0.34 |
| [24, 25, 26] | 0.50 / 0.54 | 0.92 / 0.13 | 0.76 / 0.23 |
| [29, 30] | 0.47 / 0.55 | 0.78 / 0.28 | 0.50 / 0.46 |
| [31, 32] | 0.48 / 0.57 | 0.79 / 0.29 | 0.50 / 0.56 |

**Alignment** (records/merge_statistics.json): the matched FFN neurons' mean Pearson correlation is 0.14-0.30 against
about 0 for the identity pairing (almost every neuron moves); KV-group write cosine 0.04-0.11 matched, -0.04-0.04
identity; no slot needed the uniform fallback. The equivalence checks (folding alone, each permutation alone) differ
from the unmodified block by at most 2.5e-6 of its update. Alignment finds real correspondences; they are weak, and
averaging weakly corresponding units cancels much of each block's update.

## 4. What this says and what it does not

* At this structure, untrained, neither merge recovers what deletion loses; both lose more. The ordering
  D > M-act > M-avg holds on the primary unit, in the state evaluation and in the update diagnostics.
* It does not test merging with learned or fitted coefficients (Stage 3), recurrent execution of shared blocks
  (Stage 6), other spans, or a depth/width trade at a common budget (a separate study).
* The primary comparison resolves about 3 pp at D's 5.6% (the design's rough figure); the observed differences are
  larger than that, and the interval excludes 0 for both candidates.

## 5. Runs, cost and engineering

| attempt | run | what happened | minutes | all-in |
| --- | --- | --- | --- | --- |
| 1 | `ri_merge_20261115_103303` | M-avg's build stopped after D was rebuilt and matched: the configuration check compared two different hashes (below) | 11.6 | $0.2147 |
| 2 | `ri_merge_20261115_104816` | both candidates built, durable and retained; diagnostics, state evaluation; all 8 registered units; ALL_DONE | 209.9 | $3.8842 |

Total $4.0989 of the $13.00 ceiling (one L40S at $1.09/h; the expected $9.27 assumed D-like generation, and the merges'
early degeneration stops made generation shorter). Provider balance $16.6313 before the first create, $12.5693 after
the last teardown (-$4.0620; the booked all-in is $0.037 higher). Both pods deleted with the provider's confirmation;
no pod, no volume afterwards.

* **Attempt 1's defect, a wrong assumption in the harness.** `identify_checkpoint`'s `config_sha256` hashes the parsed
  config as canonical JSON (D: `715daf5f...`); the design's D `config_sha256` is the sha256 of D's config FILE
  (`4583de76...`). Comparing the two read every candidate as "not D's configuration". The check now compares file
  with file (`config_against_d`, regression `test_the_configuration_check_compares_file_hashes`); the design and the
  digests, bound by the grant's study identity, were not changed.
* **Measured times on the pod:** the statistics forward 117 s; the eight 9,728 x 9,728 assignments 1,420 s on the
  pod's CPU (about three times the plan's estimate); M-avg's build 148 s, M-act's 1,696 s with the statistics; each
  relay copy about 25 s; diagnostics 77 s; generation 17-24 min per unit.
* Each candidate was built, checked (D plus its delta gives its artifact digest, verified from the delta file),
  durable on the relay and registered before the next; the capability units were sealed as they were written.

## 6. Storage at the end

* **Kept on the dev box:** D (`~/aad-artifacts/stage1_reasoning_inheritance/913433664fce0fed/groups/D`, protected, the
  base) and each candidate's span-block delta (`~/aad-artifacts/stage1_layer_merge/{M-avg,M-act}/<identity>/
  delta.safetensors`, 1.13 GiB each), verified by `retain` (each delta re-hashed; every span tensor equal to the pod's
  digest; D's file is D), recorded in `logs/stages/stage-1/reasoning_inheritance/storage/layer_merge_local.json`. A
  candidate is D plus its delta: `layer_merge_driver.restore` rebuilds it to its artifact digest
  (M-avg `33a5d192...`, M-act `bf90ab92...`).
* **Relay:** the two weight deltas retired after the terminal state and the local verification, as the maintainer
  directed (tombstones `relay_20261115_rilm1_M-avg_delta`, `relay_20261115_rilm1_M-act_delta`; the LFS objects
  permanently deleted and read back absent). Kept: the statistics
  (`durable/stage1_reasoning_inheritance/layer_merge/statistics/6a6a5676cd3ad50f/merge_statistics.safetensors`) and
  the evidence archive (`.../layer_merge/evidence/ri_merge_20261115_104816/d8b200f2dd71eb56/evidence.tgz`,
  downloaded again and re-hashed).
* **Records:** `logs/stages/stage-1/reasoning_inheritance/runs/ri_merge_*` (governance, runtime, closeout), the
  analysis `logs/stages/stage-1/reasoning_inheritance/analyses/layer_merge.json`, the grant with its `closed` block
  (`logs/budget/approvals/stage1_layer_merge_grant.json`).

Reproduce the decision ($0): `scripts/stages/stage-1/reasoning_inheritance/layer_merge_analysis.py --d-store
artifacts/stages/stage-1/reasoning_inheritance/runs/ri_diagnosis_20261113_193508/scr/store --store
artifacts/stages/stage-1/reasoning_inheritance/runs/ri_merge_20261115_104816/scr/store` (the stores are the dev box's;
the evidence archive above holds the same records and generations).
