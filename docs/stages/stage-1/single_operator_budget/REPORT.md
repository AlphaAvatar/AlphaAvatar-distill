# Single-operator compression at the same parameter budget (SOB1): phase-2 report

Stage 1 owns this round; the training execution stage is `stage3_recovery`. The design and how it was run are in
[`README.md`](README.md); the evidence is in
[`logs/stages/stage-1/single_operator_budget/`](../../../../logs/stages/stage-1/single_operator_budget/).
Recorded 2026-11-25 (UTC); revised 2026-11-26 (UTC) at $0: section 5's reading of the intervals and seeds and its
scope, section 7's settled provider billing, section 8's retention check. No verdict, rule or raw result changed.
Terminal state: **COMPLETE**: six training units (three arms x two seeds), six final
weights and twelve post-recovery sets (thinking and non-thinking) verified; the registered comparison is formal in
both modes (`results/study_completion.json`, `analysis.phase2_readout`). Branch `research/reasoning-inheritance`; not
merged into `main` (the maintainer decides after reviewing this report).

## 1. Result

From the same teacher (`Qwen/Qwen3-4B@1cfa9a72`, the hybrid), each arm compresses one operator to about 2.0B
parameters (N ≈ 2; largest pairwise difference 0.66%), and each gets the same Stage-3 recovery. Every number below is on
eval-set-1's static fluid-reasoning formal core: 570 items in 558 independence groups, both training seeds pooled,
1,140 paired records per arm.

**Thinking mode (primary).** The registered test rejects all three equalities under Holm (m = 3). FFN + attention
scores highest, then residual width, then depth.

| pair (A − B) | A | B | A − B | 95% interval | p (two-sided) | Holm p | registered verdict |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| FFN + attention − residual width | 32.72% (373) | 29.56% (337) | **+3.16 pp** | [+0.35, +6.03] | 0.0313 | 0.0313 | FFN + attention higher |
| FFN + attention − depth | 32.72% (373) | 16.75% (191) | **+15.96 pp** | [+13.00, +19.01] | < 0.0001 | < 0.0001 | FFN + attention higher |
| residual width − depth | 29.56% (337) | 16.75% (191) | **+12.81 pp** | [+9.93, +15.68] | < 0.0001 | < 0.0001 | residual width higher |

**Non-thinking mode (key secondary).** This second Holm family (m = 3) is tested because the first rejected all three
pairs. Both arms that keep the depth score higher than depth only. FFN + attention and residual width are not
distinguished.

| pair (A − B) | A | B | A − B | 95% interval | p (two-sided) | Holm p | registered verdict |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| FFN + attention − residual width | 29.82% (340) | 31.75% (362) | −1.93 pp | [−4.44, +0.54] | 0.160 | 0.160 | NOT_DISTINGUISHED |
| FFN + attention − depth | 29.82% (340) | 17.54% (200) | **+12.28 pp** | [+9.41, +15.25] | < 0.0001 | < 0.0001 | FFN + attention higher |
| residual width − depth | 31.75% (362) | 17.54% (200) | **+14.21 pp** | [+11.40, +17.14] | < 0.0001 | < 0.0001 | residual width higher |

**How the numbers were computed**
* The test is the registered group-level paired sign-flip test: 100,000 draws, seed 20261119.
  * In every pair with depth only (two per mode), no draw was as extreme as the observed statistic, so p is at its
    floor, 1/(draws + 1).
  * For the thinking-mode FFN + attention − residual width pair, 178 of 558 groups differ, |Σ d| = 36, and 3,128 draws
    were at least as extreme.
* The interval is the group bootstrap (10,000 resamples, seed 20261120). It is unadjusted for multiplicity and is
  shown beside the test, not as the decision.
* The test and the interval both work over items and groups with each arm's two trained checkpoints fixed. They do
  not measure training randomness (section 5).

**Seeds and multiplicity**
* Both seeds agree in sign for every pair in both modes (section 4). Two seeds are limited evidence about training
  randomness (section 5).
* The FFN + attention − residual width thinking-mode difference is the smallest effect that the test rejects:
  * per seed it is +4.21 pp [+0.53, +8.07] and +2.11 pp [−1.58, +5.89];
  * it passes Holm at its last step, where the threshold is α = 0.05, not α / 3.

**Exposure.** eval-set-1 is development-exposed, so nothing here is a never-seen blind result.

**The registered hypothesis.** "At N ≤ 2, compress depth first" (README section 1) is not supported at this point:
* at about 2.0B parameters, with the registered initialization and this recovery recipe, on the development-exposed
  eval-set-1, compressing only depth keeps the least static fluid accuracy in both modes;
* the published contrast (Minitron, width beat depth) had another teacher, about 600x the recovery tokens and
  benchmark scores. It predicted nothing here; this round's direction agrees with it.

## 2. What was compared

| arm | layers | width | FFN | Q / KV heads | parameters | trainable (embedding frozen and tied) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher | 36 | 2560 | 9728 | 32 / 8 | 4,022,468,096 | — |
| depth only | **16** | 2560 | 9728 | 32 / 8 | 2,003,851,776 | 1.615B |
| FFN + attention only | 36 | 2560 | **3840** | **16** / 8 | 2,017,066,496 | 1.628B |
| residual width only | 36 | **1280** | 9728 | 32 / 8 | 2,011,238,656 | 1.817B |

**Initialization.** Each arm was built in phase 1 from the unmodified teacher with A's operators, calibration profiles
and selection rules. Phase 1 is `sob1_phase1_20261119_194259`, COMPLETE, $4.4348.

**Recovery recipe.** Research A's A-MIX-6cat-sr recipe, unchanged across arms (README section 5):
* data: the same pack and proportions, counted before packing; random order, sequential packing with truncation,
  8,192 tokens, standard causal attention;
* batch: micro-batch 1 × 2 gradient accumulation steps, 3 epochs, 9,701 optimizer steps per seed;
* objective: the original teacher as the online KD teacher (one lineage), CE 0.25 + Top-200 + tail KD 1.0 on the
  assistant positions;
* optimizer and precision: AdamW, lr 5e-5 with cosine decay, fp32 parameters with bf16 autocast;
* seeds: 20260726, then 20260801.

**What is and is not matched.** The comparison is parameter-matched. Training FLOPs per token and inference latency
are not matched; the per-step times in section 3 show the difference.

## 3. Training

All six units ran on one L40S (secure cloud, $1.09/h, no network volume), each to 9,701 of 9,701 steps
(`run_completion.json :: completed_all_steps`). None of the six raised an out-of-memory error, and no memory
measure or recipe field was changed.

| arm | seed | training (TRAIN_START → TRAINED) | median s / step (expected) | step-0 val CE / KD | final val CE / KD |
| --- | --- | ---: | ---: | --- | --- |
| residual width | 20260726 | 1,066.6 min | 6.45 (6.58) | 9.7795 / 9.6442 | 0.5034 / 0.3644 |
| residual width | 20260801 | 1,066.5 min | 6.45 | 9.7795 / 9.6442 | 0.5014 / 0.3627 |
| depth | 20260726 | 805.2 min | 4.83 (5.43) | 3.6888 / 3.5393 | 0.5471 / 0.4104 |
| depth | 20260801 | 806.1 min | 4.83 | 3.6888 / 3.5393 | 0.5484 / 0.4116 |
| FFN + attention | 20260726 | 918.7 min | 5.52 (5.55) | 9.6627 / 9.5022 | 0.4896 / 0.3501 |
| FFN + attention | 20260801 | 918.1 min | 5.51 | 9.6627 / 9.5022 | 0.4876 / 0.3484 |

**Validation.** Validation runs every 1,212 steps on 64 packed sequences.
* From step 1,212 on, every validation ranks FFN + attention lowest, then residual width, then depth, in both seeds.
* At step 0, depth starts far lower (CE 3.69 against 9.66–9.78). Before recovery it was also the arm closest to the
  teacher by state-evaluation KL (3.47 against 7.70 and 9.35) and top-1 agreement (README).
* Neither proximity predicted the recovered arm's accuracy. These readings are diagnostics, not capability results.

**L40S memory, measured.** The first four optimizer steps of every unit were recorded and counted as training (GiB;
the device reports 44.39 GiB).

| arm | seed | peak allocated | peak reserved | least free |
| --- | --- | ---: | ---: | ---: |
| residual width | 20260726 | 42.37 | 43.40 | 0.48 |
| residual width | 20260801 | 42.37 | 43.70 | 0.18 |
| depth | 20260726 | 39.39 | 40.90 | 2.97 |
| depth | 20260801 | 39.38 | 40.98 | 2.90 |
| FFN + attention | 20260726 | 40.49 | 42.89 | 0.99 |
| FFN + attention | 20260801 | 40.49 | 42.51 | 1.37 |

The residual-width arm has the most trainable parameters (its embedding is half as wide). It was the tightest, as
priced, and still fit.

## 4. Eval-set-1

**Protocol.** Each arm × seed was evaluated on every item of every unit, in both modes, under the one registered
protocol: `nucleus_v1`, 8,192 tokens of context, action protocol v2, the hybrid's tokenizer files, and the system
message "You are a helpful Assistant.".

**Completeness.** Twelve sets are complete on their files (`completion.study`: coverage, records, protocol, report,
model identity). Each mode's sets share one protocol digest with phase 1's teacher and untrained sets.

### Per seed (static formal core)

| mode | arm | 20260726 | 20260801 | pooled |
| --- | --- | ---: | ---: | ---: |
| thinking | FFN + attention | 34.39% | 31.05% | 32.72% |
| thinking | residual width | 30.18% | 28.95% | 29.56% |
| thinking | depth | 16.14% | 17.37% | 16.75% |
| non-thinking | FFN + attention | 30.35% | 29.30% | 29.82% |
| non-thinking | residual width | 31.75% | 31.75% | 31.75% |
| non-thinking | depth | 17.54% | 17.54% | 17.54% |

**Repeated percentages.** The equal non-thinking percentages across seeds for residual width and for depth are
coincidences of the count: the two seeds' generations differ, and they disagree on 136 and 70 items respectively.

| mode | pair | seed 20260726 | seed 20260801 |
| --- | --- | --- | --- |
| thinking | FFN + attention − residual width | +4.21 [+0.53, +8.07] | +2.11 [−1.58, +5.89] |
| thinking | FFN + attention − depth | +18.25 [+14.39, +22.14] | +13.68 [+10.00, +17.48] |
| thinking | residual width − depth | +14.04 [+10.25, +17.86] | +11.58 [+8.01, +15.16] |
| non-thinking | FFN + attention − residual width | −1.40 [−5.24, +2.30] | −2.46 [−6.25, +1.37] |
| non-thinking | FFN + attention − depth | +12.81 [+9.12, +16.55] | +11.75 [+8.04, +15.41] |
| non-thinking | residual width − depth | +14.21 [+10.44, +18.04] | +14.21 [+10.43, +17.96] |

### Before recovery, after recovery, and the teacher (static core; descriptive)

**Inputs.** The untrained arms and the teacher are phase 1's sealed sets, at the same 8,192-token protocol.

**What each column means.**
* Untrained: every arm scores 0/570 in both modes, so its initialization damage is the teacher's whole score.
* Recovery: the recovered arm against its own untrained init, pooled over seeds; it equals the post-recovery score.
* Gap: the recovered arm minus the teacher.

| mode | arm | untrained | recovered (pooled) | recovery, pp | gap to the teacher, pp | teacher |
| --- | --- | ---: | ---: | --- | --- | ---: |
| thinking | FFN + attention | 0.00% | 32.72% | +32.72 [+29.31, +36.19] | −53.07 [−57.03, −49.04] | 85.79% |
| thinking | residual width | 0.00% | 29.56% | +29.56 [+26.11, +33.07] | −56.23 [−60.24, −52.20] | 85.79% |
| thinking | depth | 0.00% | 16.75% | +16.75 [+14.02, +19.64] | −69.04 [−72.74, −65.21] | 85.79% |
| non-thinking | FFN + attention | 0.00% | 29.82% | +29.82 [+26.53, +33.13] | −40.70 [−44.80, −36.60] | 70.53% |
| non-thinking | residual width | 0.00% | 31.75% | +31.75 [+28.45, +35.10] | −38.77 [−42.89, −34.62] | 70.53% |
| non-thinking | depth | 0.00% | 17.54% | +17.54 [+14.80, +20.42] | −52.98 [−57.20, −48.75] | 70.53% |

### By reasoning axis (static core, both seeds pooled; descriptive)

Records are one per item and seed. The teacher has one generation per item.

| mode | axis | items | records | FFN + attention | residual width | depth | teacher |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| thinking | task-contained deduction | 189 | 378 | 39.15% (148) | 38.89% (147) | 27.78% (105) | 99.47% (188) |
| thinking | inductive rule discovery | 142 | 284 | 25.00% (71) | 26.06% (74) | 10.92% (31) | 88.73% (126) |
| thinking | compositional generalization | 144 | 288 | 48.96% (141) | 35.42% (102) | 18.40% (53) | 82.64% (119) |
| thinking | abstract / analogical (sub-category) | 95 | 190 | 6.84% (13) | 7.37% (14) | 1.05% (2) | 58.95% (56) |
| non-thinking | task-contained deduction | 189 | 378 | 39.42% (149) | 42.33% (160) | 30.16% (114) | 88.36% (167) |
| non-thinking | inductive rule discovery | 142 | 284 | 20.07% (57) | 21.83% (62) | 10.21% (29) | 66.20% (94) |
| non-thinking | compositional generalization | 144 | 288 | 43.06% (124) | 42.36% (122) | 16.67% (48) | 71.53% (103) |
| non-thinking | abstract / analogical (sub-category) | 95 | 190 | 5.26% (10) | 9.47% (18) | 4.74% (9) | 40.00% (38) |

* The four rows of a mode are disjoint and sum to the core.
* In thinking mode, FFN + attention's lead over residual width (36 records) is more than accounted for by
  compositional generalization: 141 against 102 records, while the other three axes are within three records of each
  other.
* The counts per axis are small; no axis is tested or read on its own.

### Interactive adaptation and planning (144 episodes per seed, formal core; descriptive)

| mode | FFN + attention | residual width | depth | teacher (144) |
| --- | ---: | ---: | ---: | ---: |
| thinking | 17 / 288 (5.90%) | 16 / 288 (5.56%) | 7 / 288 (2.43%) | 36 (25.00%) |
| non-thinking | 31 / 288 (10.76%) | 22 / 288 (7.64%) | 27 / 288 (9.38%) | 42 (29.17%) |

### Terminations apart from correctness (static core; 1,140 records per arm, 570 for the teacher)

| mode | | FFN + attention | residual width | depth | teacher |
| --- | --- | ---: | ---: | ---: | ---: |
| thinking | natural termination (EOS) | 824 | 758 | 837 | 510 |
| thinking | repetition / degeneration stop | 166 | 197 | 224 | 6 |
| thinking | context-limit hit (8,192; right-censored length) | 150 | 185 | 79 | 54 |
| thinking | usable rollout | 72.3% | 66.5% | 73.4% | 89.5% |
| thinking | mean generated tokens, seed 1 / seed 2 | 4,144 / 4,210 | 4,158 / 4,441 | 3,673 / 4,103 | 2,854 |
| non-thinking | natural termination (EOS) | 1,040 | 1,061 | 1,037 | 566 |
| non-thinking | repetition / degeneration stop | 100 | 79 | 103 | 4 |
| non-thinking | context-limit hit | 0 | 0 | 0 | 0 |
| non-thinking | usable rollout | 91.2% | 93.1% | 91.0% | 99.3% |
| non-thinking | mean generated tokens, seed 1 / seed 2 | 750 / 765 | 709 / 764 | 676 / 747 | 539 |

**Usable rollout ranks differently from correctness.** In thinking mode depth only has the most natural terminations
and the fewest context-limit hits, yet the lowest correctness. Usable rollout is a guard rail, never evidence of
correct reasoning.

**Scoring of censored records.** A context-limit hit stays in every denominator and is scored by the unit's scorer
like any other record.

### Supporting units (registered headline, per seed s1 / s2, %; descriptive)

| mode | unit | n | FFN + attention | residual width | depth | teacher |
| --- | --- | ---: | --- | --- | --- | ---: |
| thinking | fluid_reasoning_720 (whole unit, its own headline) | 714 | 28.57 / 26.05 | 25.21 / 24.23 | 13.31 / 14.43 | 73.53 |
| thinking | gsm8k (math, a learned procedure) | 100 | 34 / 30 | 36 / 36 | 21 / 17 | 89 |
| thinking | math_verified | 100 | 3 / 11 | 3 / 4 | 6 / 6 | 76 |
| thinking | multihop (reading from supplied evidence) | 100 | 64 / 60 | 50 / 57 | 49 / 53 | 80 |
| thinking | rag (reading from supplied evidence) | 100 | 62 / 70 | 64 / 63 | 58 / 58 | 70 |
| thinking | knowledge (closed-book; outside the formal core) | 150 | 2.67 / 2.00 | 2.00 / 3.33 | 2.00 / 3.33 | 55.33 |
| thinking | answerability_paired (pair correct) | 120 | 10.00 / 13.33 | 8.33 / 10.00 | 11.67 / 8.33 | 53.33 |
| thinking | safety_paired (pair correct) | 100 | 0 / 0 | 0 / 0 | 4 / 0 | 42 |
| thinking | stage3_legacy_150 (in-sample regression; never generalization) | 150 | 64.00 / 71.33 | 62.67 / 64.67 | 54.00 / 60.00 | 94.67 |
| non-thinking | fluid_reasoning_720 | 714 | 26.61 / 25.35 | 26.89 / 26.89 | 15.41 / 16.39 | 62.18 |
| non-thinking | gsm8k | 100 | 23 / 22 | 19 / 10 | 17 / 19 | 40 |
| non-thinking | math_verified | 100 | 14 / 13 | 8 / 5 | 8 / 7 | 77 |
| non-thinking | multihop | 100 | 67 / 65 | 64 / 62 | 63 / 60 | 75 |
| non-thinking | rag | 100 | 69 / 72 | 77 / 76 | 77 / 67 | 81 |
| non-thinking | knowledge | 150 | 4.00 / 3.33 | 3.33 / 3.33 | 4.67 / 5.33 | 53.33 |
| non-thinking | answerability_paired | 120 | 10.00 / 18.33 | 11.67 / 18.33 | 8.33 / 10.00 | 55.00 |
| non-thinking | safety_paired | 100 | 2 / 0 | 2 / 2 | 2 / 2 | 62 |
| non-thinking | stage3_legacy_150 | 150 | 55.33 / 58.67 | 58.00 / 52.67 | 54.00 / 56.00 | 73.33 |

`eval_behavior_v0` (65) has no registered correctness headline. Every arm stays far below the teacher on every
supporting unit except reading from supplied evidence (multihop, rag), where the gap is smallest. No supporting unit is
pooled with the fluid core.

## 5. What the results say, and what they do not

**Scope.** Every statement in this section is about this round's point and nothing wider:
* about 2.0B parameters (N ≈ 2) from this 4B teacher;
* each arm initialized by the registered method (A's operators, calibration profiles and selection rules on the
  unmodified teacher, section 2);
* one recovery recipe (A-MIX-6cat-sr, 9,701 optimizer steps per seed);
* eval-set-1, which is development-exposed.

The preregistered rule, the formal verdicts of section 1 and the raw results stand as recorded.

* **They say:** within that scope:
  * compressing only depth (36 → 16 layers) keeps significantly less static fluid accuracy than compressing only the
    FFN and attention or only the residual width, in both modes, by 12–16 pp;
  * in thinking mode the FFN + attention arm is also above the residual-width arm, by +3.16 pp [+0.35, +6.03]
    (p 0.031, rejected at Holm's last step);
  * in non-thinking mode those two are not distinguished.
* **They do not say:**
  * that FFN + attention and residual width differ in non-thinking mode, or are equivalent there (NOT_DISTINGUISHED
    is not equivalence);
  * that the ordering holds at another compression ratio, another initialization, another recipe, a longer recovery
    or a blind set;
  * how stable the ranking is across training runs beyond these two seeds;
  * which part of the FFN + attention arm (the FFN's 60.5% or the attention's 40% parameter cut) carries its result;
  * anything about latency: the arms were not FLOPs- or latency-matched.

  Absolute accuracy is low for every arm (17–33% against the teacher's 71–86%).
* **What the intervals and the test measure.**
  * The group bootstrap resamples eval-set-1's independence groups, and the group sign-flip test flips the signs of
    their paired differences, while each arm's two trained checkpoints stay fixed. They measure the uncertainty from which items and groups the set holds, given
    the checkpoints these two training seeds produced. They do not measure how much a result moves from one training
    run to the next.
  * The two kinds of evidence answer different questions:
    * more independent items that were not used in development would test whether the ordering generalizes to
      other items;
    * more training seeds would estimate training randomness and how stable the ranking is across runs.
  * Both seeds agree in sign on every pair in both modes. That supports the direction. Two seeds are still limited
    evidence about training randomness. For the smallest effect, the thinking-mode FFN + attention − residual width
    difference, the two seeds give:
    * +4.21 pp [+0.53, +8.07] for seed 20260726;
    * +2.11 pp [−1.58, +5.89] for seed 20260801.
* **Observed in this round (descriptive, not a selection rule):**
  * Initial teacher-forced proximity (step-0 CE, state KL, top-1 agreement) favoured the depth-only arm and did not
    predict its recovered accuracy: ranking the arms by it before recovery would have kept the arm that ended last.
  * How a later round chooses its compression threshold and its path is left to that round's design discussion with
    the maintainer; this report recommends neither.

## 6. Execution

| session | pod | created → deleted (UTC) | outcome | all-in |
| --- | --- | --- | --- | ---: |
| `sob1_train_residual_width_20261121_074426` | `ombh51bu2anrmn` (datacenter not recorded) | 11-21 07:47 → 11-22 22:12 | COMPLETE (ALL_DONE), 2,305.8 min | $42.6771 |
| `sob1_train_depth_only_20261122_223829` | `6qvdu7iw1kbl14` (US-TX-4) | 11-22 22:39 → 11-24 03:08 | COMPLETE (ALL_DONE), 1,708.3 min | $31.6179 |
| `sob1_train_ffn_attention_20261124_033254` | `t58wmac140suz7` (US-TX-4) | 11-24 03:44 → 11-25 13:45 | COMPLETE (ALL_DONE), 2,040.5 min | $37.7671 |

**Attempts and teardown.** Each arm ran once, with no failed or repeated attempt.
* Each session launched from its own one-use authorization and session commit (`614e7e60` for the last).
* Each was closed by the launcher: manifest, teardown gate, pod delete. In every case the provider confirmed the pod
  gone and the account held 0 pods and 0 volumes.
* No watcher of any run is left.

**Evaluation time** (thinking / non-thinking, per seed):
* residual width 68.8 / 8.7 and 72.3 / 8.8 min;
* depth 34.5 / 7.2 and 35.5 / 6.9 min;
* FFN + attention 72.5 / 8.7 and 69.0 / 8.4 min.

**Save points.** Every 1,104 steps (8 per seed, 48 in all) each save point went to the relay and was verified by a
fresh download and re-hash. Each was removed once its successor or the final had handed off; none remains.

**Observations, none of which changed a result:**
* *Capacity.* The FFN + attention create met two capacity refusals (03:34Z, 03:39Z) and succeeded on the third
  attempt; acquisition is $0.
* *Transient poll misses.* The launcher's poll saw "driver unanswered for 0 s" three times. Twice it failed to copy a
  non-required log or status file to the dev box (three times in all). Each recovered on the next poll.
* *Relay slowdown.* The verify download of FFN + attention seed 20260801's final took 1,950 s, against 17 s for
  seed 1 (the private relay's download path, as in TPI1). It cost about $0.6 of idle GPU and verified.
* *Dev-box disk.*
  * During arm 3, another project's Claude job wrote about 13 GB under `~/.claude/jobs/206b5fe3/tmp`, and `/` fell
    from 18 to 11 GiB free. This arm's own closeout bound was 3.51 GiB.
  * On the maintainer's permission (2026-11-24, 「可以清理 ~/.claude/jobs/」), only that job's `tmp` was deleted. Before
    deleting, I checked that the job had finished its turn, held no open files there, and had copied its candidate
    out. Its two session-state files were kept. `/` went to 25 GiB free.

## 7. Cost

| arm | expected | hard cap | all-in | GPU | container disk |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual width | 42.59 | 63.43 | 42.6771 | 41.8879 | 0.7891 |
| depth | 35.70 | 61.09 | 31.6179 | 31.0332 | 0.5846 |
| FFN + attention | 36.41 | 61.35 | 37.7671 | 37.0687 | 0.6983 |
| **phase 2** | **114.70** | **185.87** (+ $1.00 reserve) | **112.0621** | 109.9898 | 2.0720 |

**Basis.** Each all-in is GPU at the accepted $1.09/h plus 150 GB of container disk at $0.1/GB-month over the
elapsed minutes, from the run's `closeout/outcome.json`.

**Remainder and caps.**
* $73.8079 of the ceiling and the $1.00 reserve were not used. They lapse with the closed kind and fund nothing later.
* The project cap ($1,161.06) was not raised.
* No arm's per-session hard cap was approached; the largest was residual width, at 67% of its cap.

**Reconciliation with the provider** (read-only; REST v2 and the account):
* **Balance:** $235.8108 before phase 2 (2026-11-21 07:40Z) and $123.7283 after it (2026-11-25 13:54Z), a drop of
  $112.0825. The records book $112.0621, a difference of +$0.0204 (0.02%).
* **Billed per pod** (`/v2/billing/pods`, daily buckets). Fetched first at 2026-11-25 14:03Z, while the last pod's
  2026-11-25 bucket was still posting, and again at 18:46Z, when it had posted (SETTLED):

  | arm | billed at 14:03Z | billed at 18:46Z | settled − record |
  | --- | ---: | ---: | ---: |
  | residual width | $42.6850 | $42.6850 | +$0.0079 |
  | depth | $31.6237 | $31.6237 | +$0.0058 |
  | FFN + attention | $36.9745 (−$0.7926: still posting, 18 min after the delete) | $37.7738 | +$0.0067 |
  | **phase 2** | $111.2832 | **$112.0825** | **+$0.0204** |

  * The settled billed total equals the balance drop, $112.0825, to the cent.
  * The +$0.0204 against the closeouts' all-in is under 0.03% per pod; its cause is not attributed further.
  * Nothing remains to post: the account has billed $0.00/h since the last delete, and the balance read $123.7283
    at 13:54Z and again at 18:46Z.
  * Snapshots: `logs/budget/provider/runpod_billing_v2_2026-11-25T1403Z.json` (posting) and
    `runpod_billing_v2_2026-11-25T1846Z.json` (settled). The grant's `phase_2_closed.provider_settlement` records the
    settled figures; its `provider_reconciliation` keeps the 13:54Z reading.
* **Project position** (`shared.project_position` on the full-history snapshot):
  * at 13:54Z, with the last bucket partial: $1,085.38 committed;
  * settled (18:46Z snapshot): **$1,086.2713** committed by the provider, against $1,086.2508 by the records
    ($974.1887 before phase 2 + $112.0621);
  * **$74.7887** remains under the $1,161.06 cap;
  * account balance $123.7283; 0 pods, 0 volumes.

  The phase-2 grant is closed: its unused $73.8079 and the $1.00 reserve are void. The $74.7887 is headroom under
  the project cap, not an authorization; a later round needs its own approval.

## 8. Checkpoints and storage

| final (fp32) | sha256 of `model.safetensors` | bytes | kept |
| --- | --- | ---: | --- |
| residual width, 20260726 | `00b430fdcef27257e8b4b08f8c98794eb5f8dd8c60e51de1b84df12ba4e82263` | 8,045,000,112 | dev box `~/aad-artifacts/stage1_single_operator_budget/residual_width-s20260726/final/00b430fdcef27257/` |
| residual width, 20260801 | `b701868f05c6d6e4d7ed8b2b193551060c5f0d8c7cadb3bb30f97561740cccc0` | 8,045,000,112 | dev box `~/aad-artifacts/stage1_single_operator_budget/residual_width-s20260801/final/b701868f05c6d6e4/` |
| depth, 20260726 | `5635c33e6b414bd0bd361c450f132c41050e69c68cd948a734c2bfd27a86f80b` | 8,015,427,400 | dev box `~/aad-artifacts/stage1_single_operator_budget/depth_only-s20260726/final/5635c33e6b414bd0/` |
| depth, 20260801 | `e4159f1fc5b1f884b5bd43017651a7f70f4ef54d5fa58ff14165962b4e341f98` | 8,015,427,400 | dev box `~/aad-artifacts/stage1_single_operator_budget/depth_only-s20260801/final/e4159f1fc5b1f884/` |
| FFN + attention, 20260726 | `589acf286481f495a587ac76ad8956af66a623d3134e029fb55bed12a8c8c16b` | 8,068,311,496 | dev box since 2026-11-26 `~/aad-artifacts/stage1_single_operator_budget/ffn_attention-s20260726/final/589acf286481f495/` (was the relay's `durable/stage1_single_operator_budget/ffn_attention/ffn_attention-s20260726/final/589acf286481f495/`) |
| FFN + attention, 20260801 | `64292e508e2d6d2e67b503a5267b088ae13be45aad1bb3fadbe15c8dcd14530e` | 8,068,311,496 | dev box since 2026-11-26 `~/aad-artifacts/stage1_single_operator_budget/ffn_attention-s20260801/final/64292e508e2d6d2e/` (was the relay's `.../ffn_attention-s20260801/final/64292e508e2d6d2e/`) |

**Dev-box finals** (residual width and depth): each was verified file by file on the dev box (`launch.py retain`;
`storage/local_retention_phase2.json`). After their arm's evaluation evidence was archived, their relay copies were
retired under the maintainer's 2026-11-20 rotation decision. The tombstones are `relay_sob1_train_*` in
`logs/maintenance/inventories/checkpoint_tombstones.json`.

**Relay finals** (FFN + attention): kept on the relay by design (`design.json :: storage.phase_2_finals.placement`).
Each object was listed and its LFS sha256 checked against the record (`storage/local_retention_phase2.json`).
**Moved 2026-11-26** (the maintainer's storage option a, for DMR1 variant 1's relay room): both were fetched to the dev
box, re-hashed file by file and read back. After they were registered, their relay copies were retired behind the
tombstones `relay_sob1_train_ffn_attention_20261124_033254_*_final` (`relocate_relay_finals.py`; decision log
2026-11-26). The weights, their hashes and every measurement are unchanged.

**Evidence archives.** Each session's evidence (generations, sealed parts, training records; no weights) is one
deterministic archive. Each is on the dev box (`<scr>/archive/`) and on the relay
(`durable/stage1_single_operator_budget/phase2_evidence/<run>/`), as recorded in `storage/relay_rotation_phase2.json`.

| arm | archive sha256 | files |
| --- | --- | ---: |
| residual width | `e145e450…` | 236 |
| depth | `9c863f89…` | 201 |
| FFN + attention | `b2dea49f…` | 238 |

**Evidence in git.** Each run's lightweight evidence is committed under `runs/<run>/evidence/`: train logs and
configs, `final.json`, `savepoints.json`, formal reports, completion and package records.

**The three phase-1 initializations** (dev box and relay) are kept until the next round's plan is fixed (maintainer,
2026-11-26). No run consumes them now, since phase 2 is complete and its kind closed. Their retirement under AGENTS.md
2.5 is then decided by that plan's explicit consumers. The six finals stay at the placement above.

| initialization | sha256 of `init/model.safetensors` | bytes | kept |
| --- | --- | ---: | --- |
| residual width | `599f86fe6c77339aaf09062ee29264c2769b90f0029beed33c774ed84bfd7ada` | 4,022,523,088 | dev box and relay `durable/stage1_single_operator_budget/residual_width/init/c853640b1ce2392c/` (with its mapping, `aeaaca73…`) |
| FFN + attention | `1b8eaa4c85d9a0309e390d2b3169c1af00d286ad64ad4394f77089a27bd27d14` | 4,034,178,856 | dev box and relay `durable/stage1_single_operator_budget/ffn_attention/init/4429c94994a5bd1f/` |
| depth | `76c8ca19f0d9a7df3669b80b9c80788d206a73d4ef8ffbc3150099a726336222` | 4,007,724,000 | dev box and relay `durable/stage1_single_operator_budget/depth_only/init/36888440ef83bfd6/` |

**Retention check, 2026-11-26 (UTC), read-only.**
* Dev box: the four finals, the three initializations and the residual-width mapping were re-hashed. Every sha256 and
  size equals the registry (`logs/maintenance/inventories/checkpoint_registry.json`) and the retention records
  (`storage/local_retention.json`, `storage/local_retention_phase2.json`).
* Relay, listed with LFS metadata:
  * present, each LFS sha256 equal to its record: the two FFN + attention finals, the three initializations and the
    mapping, and the four sessions' evidence archives;
  * absent: no residual-width or depth final (their tombstones) and no save point.
* Dependencies. The registry's `referenced_by` used to match a unit's bare name, so each `init` directory was listed
  as referenced by README.md and the shared baselines. It now matches the nearest identifying name, the content id
  (`build_checkpoint_registry.py :: identifying_name`). No config or script outside SOB1 names the initializations or
  the finals. SOB1's own driver and launcher read them through `design.json :: relay_prefix_root`, and both of its
  kinds are closed, so `require_kind_open` refuses any further session. Their remaining consumers are this study's
  records.

## 9. Reproducing

```
PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python \
  scripts/stages/stage-1/single_operator_budget/launch.py study-completion
```

**What it rebuilds.** This command rewrites `results/study_completion.json`, decision included, from the three
sessions' secured evidence (`<scr>/store/out` on the dev box; the archives above hold the same bytes) and phase 1's
sealed sets.

**Identities to check against.**
* Each set's scored-directory digest is in `readout.scored_digests`.
* The executables are each session's commit with its bundle: `d72a9b46` (residual width), `1b368f1e` (depth) and
  `614e7e60` (FFN + attention).
* The commands are in [`README.md`](README.md) section 9a.
