# Single-operator compression at the same parameter budget (SOB1)

Stage 1 owns this round; its training is `stage3_recovery`. Status: **phase 1 COMPLETE and closed in code
(2026-11-20 UTC); phase 2 prepared at $0. The second brief of 2026-11-20 fixed its four takeover and launch defects
and inventoried the project's storage. The third brief cleaned this project's dev-box files (10.73 GiB freed), set a
placement for the finals without a larger disk, and fixed the free relay's two interruptions. Waiting for the
maintainer**: the dev box is still 17.82 GiB short over the study, and the project cap and phase 2's grant are owed
(section 10). The maintainer fixed the protocol on 2026-11-19 (UTC), corrected the preparation three times the
same day and granted phase 1 (`logs/budget/decisions.md`, the 2026-11-19 entries). Phase 1 ran once,
`sob1_phase1_20261119_194259`: one L40S, 239.6 min, $4.4348 all-in of the $22.92 cap, the pod deleted and confirmed
gone; `study_completion.status` COMPLETE (`runs/sob1_phase1_20261119_194259/closeout/phase1_completion.json`), the
three initializations and mappings re-hashed on the dev box (`storage/local_retention.json`) and kept on the relay
for phase 2 (`decisions.md :: 2026-11-20 (UTC) — SOB1 phase 1 COMPLETE`). Branch `research/reasoning-inheritance`,
not merged to `main`.

**Phase 1's comparison** (`runs/sob1_phase1_20261119_194259/evidence/comparison/comparison.md`, descriptive: no arm is
ranked or eliminated by it). Static fluid formal core, 570 items (558 groups), 95% group-bootstrap intervals:

| mode | teacher | each untrained arm | teacher − arm, pp | degeneration stops on the core: width / FFN + attention / depth (teacher) |
| --- | ---: | ---: | --- | --- |
| thinking | 85.79% | 0.00% | 85.79 [82.87, 88.58] | 570 / 570 / 551 (6) |
| non-thinking | 70.53% | 0.00% | 70.53 [66.73, 74.30] | 570 / 568 / 549 (4) |

Untrained, every arm produces degenerate text (repeated digits, spaces, symbol runs); recovery (phase 2) is what the
study compares.

**Before recovery: each arm against the teacher on the build session's state evaluation suite** (not eval-set-1;
`results/pre_recovery_diagnostics.json`, `analysis.py diagnostics`, read from the committed init records). Diagnostic
only: like the table above, it ranks, keeps and eliminates nothing, and all three arms go to the two-seed comparison.

| arm | teacher KL, equal-domain mean | worst domain | token mean | critical-token KL | top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth only | **3.47** | 3.94 | 3.67 | 10.14 | **34.08%** |
| residual width only | 7.70 | 11.64 | 8.91 | 30.54 | 27.05% |
| FFN + attention only | 9.35 | 10.66 | 9.98 | 14.73 | 6.37% |

The depth-only arm is closest to the teacher by KL and by top-1 agreement, and still scores 0/570 on the static core in
both modes, like the other two.

**Phase 2, prepared at $0 (2026-11-20, the maintainer's brief of that day).** Each arm's two-seed plan is derived from
its committed init record and checked (`plans/training_plan_<arm>.json`, `plan.py --check`: A-MIX's and TPI1's A unit
outside the declared fields: the original Qwen3-4B as the online KD teacher, the same pack and proportions counted
before packing, random order, sequential packing with truncation, 8,192 tokens, causal attention, micro-batch 1 x 2
gradient accumulation steps, 9,701 steps, `stage3_recovery`). The residual-width arm trains first; its first four
formal optimizer steps record the memory and continue as training. Phase 2 has its own proposal
(`plans/proposal_phase2.json`): the Stage-3 recovery training and its evaluation are one budget, $185.87 ceiling
(per-arm caps $63.43 / $61.09 / $61.35) + $1.00 teardown reserve, needing a project cap of **$1,161.06** at the
2026-11-20 position (committed $974.1887, read again at 15:44Z). Phase 1's $18.4852 remainder is void. The relay
fits phase 2 on the free plan with the rotation, and FFN + attention's finals stay there. After this project's
cleanup the dev box has 16.12 GiB free, which is not enough for even the first arm's 18.50 GiB (section 8).

| file | what it is |
| --- | --- |
| [`configs/.../single_operator_budget/design.json`](../../../../configs/stages/stage-1/single_operator_budget/design.json) | the design: arms, operators, calibration, training (recipe, order, memory probe, durability), the one evaluation protocol, the registered analysis, the cost model's constants, phases and storage |
| [`configs/.../single_operator_budget/selection.json`](../../../../configs/stages/stage-1/single_operator_budget/selection.json) | what is reused from the packed search's journal and TPI1, and every gate: the eight committed DEPTH rounds verbatim with their resume checks (`selection.py --write / --check`) |
| [`plans/structures.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/structures.json) | the three structures from the real implementation (`structures.py`) |
| [`plans/proposal.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/proposal.json) | phase 1's proposal (ceiling, per-session hard cap, teardown reserve) and phase 2's separate draft; frozen as granted (`proposal.py --check`) |
| [`plans/proposal_phase2.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/proposal_phase2.json) | phase 2's own proposal: ceiling, per-arm caps, reserve, the project cap it needs, its storage (`proposal.py --phase2`) |
| [`plans/resource_snapshot_phase2.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/resource_snapshot_phase2.json), [`plans/disk_peak_phase2.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/disk_peak_phase2.json) | the position, relay headroom and dev-box free space read on 2026-11-20; phase 2's dev-box peak at its real landing paths (`disk.py --kind sob1_train --total`) |
| `plans/training_plan_<arm>.json` | each arm's two units, derived from its committed init record (`plan.py`) |
| [`results/pre_recovery_diagnostics.json`](../../../../logs/stages/stage-1/single_operator_budget/results/pre_recovery_diagnostics.json) | each untrained arm's state-evaluation KL and top-1 agreement against the teacher |
| [`storage/phase1_evidence_archive.json`](../../../../logs/stages/stage-1/single_operator_budget/storage/phase1_evidence_archive.json) | phase 1's raw generations and records as one hashed, rescorable archive (dev box and relay) |
| [`plans/resource_snapshot.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/resource_snapshot.json) | the project position, the relay's headroom and the dev box's free space as read on 2026-11-19 |
| [`plans/disk_peak.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/disk_peak.json) | where phase 1's bytes land on the dev box and the peak when they coexist (`disk.py`) |
| [`plans/dry_run_20261119.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/dry_run_20261119.json), [`plans/dry_run_20261119b.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/dry_run_20261119b.json), [`plans/dry_run_20261119c.json`](../../../../logs/stages/stage-1/single_operator_budget/plans/dry_run_20261119c.json) | the $0 dry runs of phase 1: everything up to the create, nothing created |
| `scripts/stages/stage-1/single_operator_budget/` | `phase1_driver.py` (the three builds and the phase-1 evaluations), `completion.py` (an evaluation set is complete only on its files, phase 1's and phase 2's), `train_driver.py` (phase 2's state machine on TPI1's durability helpers), `plan.py` (one arm's training plan), `launch.py` (sessions, budget check, closeout, retention), `analysis.py` (the decision rule and phase 1's comparison table), `disk.py` (the dev box's landing paths and peak), `common.py`, `tests/` |

## 1. Question and hypothesis

From the same ~4B teacher (`Qwen/Qwen3-4B@1cfa9a72`, the hybrid), compressed to about the same real parameter count,
how much fluid reasoning does compressing **only depth**, **only the FFN and attention**, or **only the residual
width** lose? After the same Stage-3 recovery, which of the three keeps more?

* N = P_teacher / P_student ≈ 2 is the **first compression point examined**. P counts every parameter of the real
  Qwen3 implementation, with the tied embedding counted once. One point speaks only for its neighbourhood.
* "At N ≤ 2, compress depth first" is a hypothesis to test. The priority among the operators is decided only after
  recovery; nothing in this round ranks them before that.
* Published evidence against it: Minitron (Sreenivas et al., arXiv:2408.11796v4, Table 1) pruned Llama 3.1 8B to
  4.5B and distilled on 94B tokens; its width-pruned model beat the depth-pruned one on every base benchmark
  reported (GSM8K 41.2 vs 16.8). Another teacher, about 600 times this round's recovery tokens and benchmark scores
  rather than fluid reasoning, so it frames the test and predicts nothing here.

## 2. The baseline: A's method, not A's weights

TPI1's A (teacher-parent initialization) is the research's default baseline; Research A's packed Top-1 is a
historical reference (maintainer, 2026-11-18; [research index](../reasoning_inheritance/README.md)).

* **Reused:** A's initialization method. Every selection is made on the unmodified teacher with the operator,
  calibration profile and selection rule A used; a selection already recorded on the teacher is reused when its
  target is this round's; every rebuild must equal its recorded digest.
* **Not reused:** A's 596M weights (a baseline only on their own structure). They stand beside the arms as context.
* **One lineage** (AGENTS.md P17.1): the teacher, the calibration profiles, the training pack (generated by that
  teacher), the online KD teacher, the tokenizer, the chat template and the configuration.

## 3. The three structures (fixed)

Counts are `transformers.Qwen3ForCausalLM` on the meta device from the teacher's pinned `config.json`; the same method
reproduces the teacher's 4,022,468,096 and A's 596,049,920 exactly.

| arm | layers | width | FFN | Q / KV heads | head_dim | parameters | N | vs 2.011B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher | 36 | 2560 | 9728 | 32 / 8 | 128 | 4,022,468,096 | 1 | |
| **depth only** | **16** | 2560 | 9728 | 32 / 8 | 128 | 2,003,851,776 | 2.007 | −0.37% |
| **FFN + attention only** | 36 | 2560 | **3840** | **16** / 8 | 128 | 2,017,066,496 | 1.994 | +0.29% |
| **residual width only** | 36 | **1280** | 9728 | 32 / 8 | 128 | 2,011,238,656 | 2.000 | 0.00% |

The largest pairwise difference is 13,214,720 parameters (0.66%). The split of the FFN + attention arm, 16 / 8 heads
with FFN 3,840, is the maintainer's (2026-11-19): attention removes 40% of its parameters and 18.8% of the arm's
reduction, the FFN 60.5% and 81.2%. Per-module changes: `structures.json :: arms.<arm>.changed_from_teacher`.

## 4. Building each arm: A's operators on the original teacher

One session (`sob1_phase1`, `phase1_driver.py`) builds the arms in the order residual width, FFN + attention, depth
(the cheap ones are durable before the long search). Each arm runs in its own GPU process on the unmodified teacher,
under the packed search's own session (inputs v3), every operator config hash pinned to the journal's.

| arm | what is reused | what is new | gate before anything is built on it |
| --- | --- | --- | --- |
| residual width | **TPI1's sealed teacher residual statistics** (`teacher_residual_statistics.safetensors`, sha256 `e4145e38…`, 37 points × 2,560, 236,939 statistics positions), staged from the relay; no second teacher statistics pass | P at 1,280 from them over all 37 residual points, weights 9 / 1 / 8 | the file's sha256, every tensor's layout and its statistics positions equal `selection.json :: residual_width.statistics` (derived from TPI1's committed init record, whose RESIDUAL_WIDTH digest gate and diagnostics gate passed); P at 1,024 recomputed from it reproduces the recorded 37-point diagnostics |
| FFN + attention | the journal's 16 / 8 `kept_heads` on the teacher (A's ATTENTION) | the FFN operator at 3,840, sharing one statistics pass with a run at 3,072 | the 3,072 run builds the journal's FFN state `3f743039…` bit for bit; the head child equals `87ee6205…` |
| depth | **the journal's 36 → 28 search, rounds 1-8** (state `fede45db…`, 260 candidate subsets) | rounds 9-20: 28 + 27 + … + 17 = **270 candidate subsets**, each against the unbypassed teacher; **each round persisted the moment it commits** | the operator re-checks the eight rounds, re-scores round 8's chosen subset (must equal 0.5305385485267972 exactly) and continues; rounds 1-8 must stay `[2, 32, 3, 26, 30, 8, 20, 28]` |

**Why the depth search can resume** (`selection.json :: depth_only.resume_checks`, all true, no field missing):

| condition | the journal's step |
| --- | --- |
| the reference is the original teacher | `parent_state`, a first-level state of the search root `aea5f999…` |
| the same teacher | `Qwen/Qwen3-4B@1cfa9a72`, root sha256 `a8b9dec3…` |
| the same operator and profile | `depth.causal_kl_greedy_v1` (its signature hash unchanged), `calib.hybrid_d4_base@v2` |
| the same items, scoring positions and packing | config hash `58c86cde…` (TPI1's pin); 558 items; random shuffle, sequential packing with truncation, standard causal attention, 8,192-token sequences (79 packed sequences) |
| the same execution | micro-batch 1, `original_order_v1`, scored rows, prefix reuse |
| greedy accumulation and the tie rule | each round removes only the earlier choices, scores every surviving layer, picks the lowest score, ties to the lower index |
| the removal order | `[2, 32, 3, 26, 30, 8, 20, 28]`, TPI1's committed one |

Every round's full table (each candidate's score) is kept, so the operator can check them. Had a field been missing,
the record names it and the first round that must be scored again (`minimal_recompute`). The resume is a core
capability (`OperatorContext.resume_state`, execution only, never hashed: a resumed search computes the state a whole
one does; `tests/initialization/test_depth_resume.py`).

**Rounds 9-20 are durable one by one.** The operator hands each committed round record (its choice and every
candidate's score) to `OperatorContext.on_progress` (core, execution only, refused by an operator that reports
nothing); the driver writes it atomically to `depth_only/records/depth_rounds.json`, which the next poll pulls to the
dev box. A later attempt resumes from the journal's eight rounds plus every persisted round (written for the same
step, contiguous after round 8), re-scoring round 8's chosen subset **and** the latest persisted round's exactly
(`resume_state.anchors`). An interruption loses at most the round in progress (17-28 candidates, at most ~13 min).

Each arm then: one frozen mapping keyed by root indices, saved and loaded back, materialized onto the teacher;
geometry and parameter count against `structures.json`; a finite forward after reload; state evaluation against the
teacher (diagnostic only); the checkpoint and mapping to the relay, downloaded again and re-hashed; the arm's
`init_record.json`.

**One manifest per arm (fixed in the second correction, 2026-11-19).** The mapping's files come from `frozen_mapping.save`'s own record
(`common.mapping_manifest`): `mapping.json` always, and `tensors.safetensors` only when the record holds a width
projection, which must be exactly when the design declares one (the residual-width arm). The checkpoint's are
`model.safetensors` and `config.json`. Before, the build hashed both mapping files for every arm, so the FFN +
attention and depth arms (no width projection, no tensor file) would have failed right after saving their mappings.
The build record's manifest is what the upload sends and verifies (each relay group only when its re-downloaded
hashes are exactly the manifest), what the takeover and the restore read (`durable_init`, `model_for`) and what local
retention re-hashes on the dev box (`common.products`, `common.product_problems`). An empty manifest or an empty
relay list proves nothing, and the relay's `put` refuses an empty upload.

**Calibration and training are different data sets, each counted before packing.**

| set | before packing | after packing (compute records only) |
| --- | --- | --- |
| calibration `calib.hybrid_d4_base@v2` (FFN, DEPTH) | 558 examples, 28 cells, 211,203 unique scored positions | 79 packed sequences |
| calibration `calib.hybrid_d4_rh@v2` (ATTENTION, WIDTH) | 388 examples, 28 cells, 213,361 unique scored positions | 58 packed sequences, 236,939 non-padding tokens |
| training (the A-MIX-6cat-sr pack) | 45,363 examples, 56,408 turns, 11,862,564 unique supervised tokens (thinking 5,935,809, non-thinking 5,926,755; reasoning 4,945,744, answer 6,916,820, tool calling 995,341) | 6,467 packed sequences of 8,192 tokens |

## 5. Recovery: Research A's recipe, unchanged

`plan.py --arm <arm>` derives each arm's two units from the A-MIX-6cat-sr plan (sha256 `6632bdbc…`) after phase 1
commits the arm's init record. Only the unit and arm names, run name, student path, output directory and
`lineage.initialization` change. Every other field is compared with A-MIX's unit, and with TPI1's A unit of the same
seed (A's recipe), and must be equal:

* the sealed pack and its data proportions; random order (a per-epoch permutation from the seed); sequential packing
  with truncation allowed; 8,192 tokens; standard causal attention;
* batch size 2 as micro-batch 1 × 2 gradient accumulation steps; 3 epochs, 9,701 optimizer steps;
* the teacher in bf16 as the online KD teacher (one lineage); CE 0.25 + Top-200 + tail KD 1.0 at the assistant
  positions; AdamW lr 5e-5, weight decay 0.01, betas (0.9, 0.95), 485 warmup steps, cosine to 0.1, clip 1.0;
* fp32 parameters with bf16 autocast and gradient checkpointing; the tied embedding frozen; save points every 1,104
  steps; seeds 20260726 then 20260801.

The comparison is parameter-matched; FLOPs and latency are not matched and are reported apart. Supervised-token
exposure is counted from each trainer's log.

| arm | training FLOPs per token vs A's 596M | expected s / step | expected h / seed (bound) |
| --- | ---: | ---: | ---: |
| residual width only | 1.93x | 6.58 | 17.7 (20.4) |
| depth only | 1.59x | 5.43 | 14.6 (19.5) |
| FFN + attention only | 1.63x | 5.55 | 14.9 (19.6) |

**Order and memory on the L40S.** Phase 2 trains one arm per session on one L40S, the **residual-width arm first**:
its small embedding leaves 1.82B trainable parameters, so it is the tightest (38.7 GiB expected peak allocated,
44.6 GiB conservative reserved, of 44.99 GiB).

* The trainer logs, for each unit's first four optimizer steps, the peak allocated, peak reserved and the device's
  free memory after each phase: forward with the KD / CE loss, backward, gradient clip, optimizer step (AdamW
  allocates its state at the first one). An out-of-memory error is logged with the phase it was raised in
  (`aadistill.training.train.MEMORY_PROBE_ENV`, execution only). The driver keeps them in `units.json :: memory`.
  If the steps fit, training simply continues.
* On an out-of-memory error, the phase decides the remedy. In the KD phase: `expandable_segments`, then a smaller
  `loss.kd_chunk`. In the optimizer step: `expandable_segments`, then AdamW without its multi-tensor temporaries.
  Each is recorded as a deviation. The sequence length, causal attention and effective batch size are never changed.
* An 80 GB card is proposed only for an affected arm, after a measured out-of-memory survives those steps.

## 6. Evaluation and the registered analysis

**One protocol for every model** (`design.json :: evaluation.protocol`): eval-set-1, every item of every unit, both
modes, `nucleus_v1`, action protocol v2, the system message "You are a helpful Assistant.", the degeneration check
every 256 tokens. Every model generates until EOS or 8,192 tokens minus its rendered prompt. Every model is packaged
by one rule: the weights linked, plus the teacher's tokenizer files and generation config, hashed against the
teacher identity. eval-set-1 is development-exposed.

| model | when | records |
| --- | --- | --- |
| each untrained arm | phase 1 | new, 8,192 |
| the teacher | phase 1 | **new, 8,192: no identity-matched record exists** |
| each arm × seed after recovery | phase 2 | new, 8,192 |

**The teacher reference.** Every generation summary under `logs/` and `artifacts/` naming the hybrid teacher was
checked on 2026-11-19. Its eval-set-1 records are `a_teacher_20261030_012648`'s, plus RI-S1's copy of them, and they
are at 40,960 tokens. Its other generations are training-data sessions, not eval-set-1. So the teacher is evaluated at
8,192 in phase 1. Capped at 8,192, its 40,960 records would decode 5.54M tokens in thinking mode (200 of 1,716
samples' turns reach the limit) and 0.62M in non-thinking mode. The 40,960 records stay as historical reference and
are never pooled with these.

**Prompts and packing.** Evaluation builds each item's prompt on its own: the runners render every item through the
chat template and batch requests, never packing them. Random-order sequential packing is the calibration's and the
training's (AGENTS.md P19), never the evaluation's.

**A set is complete only on its files** (`completion.py`). Phase 1 has eight sets: the three untrained arms and the
teacher, each in both modes. Each is checked against the registered manifest:

| condition | what is checked |
| --- | --- |
| item coverage, repeats | every registered unit of eval-set-1 has its generations file with every registered item exactly once (the set's own item files, 1,716 items) |
| generation records | each turn carries its raw output and the termination fields the scorers read; record by record, its seed is the seed rule's for its unit, item and turn, prompt plus allowance is 8,192, the rendered prompt has the mode's think state (`self_opening` / `preclosed`), a natural stop ends with the stop id, the fluid system turn is the registered message |
| protocol summary | every unit's summary and runner summaries; every key field recorded and equal to the registered value (decoding protocol and its seed rule, 8,192 context, system message, chat template and tokenizer, template kwargs, degeneration interval, action protocol, task file). The set's protocol digest is recorded; the four sets of a mode must share it |
| formal report | rescored from these generations; a `{}` or stale report is a missing part |
| model identity (provenance) | every sealed part names the generation that sealed it (`part_NNNN.<generation>.jsonl`), an id naming the attempt, the target model, the think mode and the round (`<attempt>-<target>-<mode>-g<n>`), which must be this set's; that generation's package record (`packages.json`, written before anything is generated) names the arm's init-record weights or the teacher identity's three shards, this target and mode, and the registered protocol (`protocol_sha256`); each of the part's records carries the runner label of this target (`label_of`: `<arm>_untrained` or `teacher`) and is a registered item under the per-record protocol (a teacher EOS id ends a natural stop). Checked whether or not a generations file exists, so an attempt stopped with only `parts/` is checked the same way. Every generations record must be one a proven part holds |

A set an earlier attempt completed is taken as it is. Otherwise the smallest repair, before anything is merged:

* a `{}` or stale report is rescored, and nothing is generated;
* every part whose generation's weights, mode or protocol, or whose own records, cannot be shown is isolated under
  `superseded/<generation>/` with its reason (`isolated_parts.json`), never pooled; the proven parts are kept and
  reused;
* a generations file with missing, extra or repeated items, or a record no proven part holds, is superseded, so the
  runners compose the unit from the proven parts and generate only the items none holds, under the same per-item
  seeds;
* a unit whose runner protocol cannot be shown (no runner summary, or a field that differs) is generated again.

The same item sealed twice in proven parts is refused. Each generation round has its own id
(`<attempt>-<target>-<mode>-g<n>`), and
the runners tag every part they seal with it (`reasoning_inheritance.driver.Sealer`, which also numbers a new part
past the highest, not by count, so a gap left by an isolated part never overwrites a kept one). Phase 1 ends
`ALL_DONE` only when the three inits are durable, the eight sets are complete and each mode's sets share one protocol
(`completion.phase`). The counter-example regression: a generation of other weights left partial answers in `parts/`
with no generations file, an earlier generation of the right weights sealed a few items, one part carries no
generation; the new attempt isolates the wrong and the untagged parts, reuses the proven ones and generates only the
rest, through the runners' real merge with a synthetic engine. A second regression runs on real runner records (TPI1's
sealed thinking-mode parts at 8,192, relabeled): the residual-width arm's parts filed in the depth-only arm's set,
a depth-only-named part holding residual-width records and a depth-only part of the other mode are each isolated by
what they say about themselves (the id's target and mode, each record's label); the depth-only parts are reused and
only the items none holds are generated. Checked on TPI1's four real A
sets at 8,192, coverage, records, protocol and the report all pass, and the two seeds share one protocol digest per
mode; model identity is missing there because TPI1 did not record the weights' hashes.

**Phase 1's comparison** (`analysis.py phase1-table`, written to `comparison/` on the pod and re-derivable on the dev
box at $0) reads the final sealed records once phase 1 is complete. Per mode (thinking and non-thinking apart) and per
arm, it gives the teacher's and the untrained arm's accuracy and the item-paired **teacher − arm** difference in
percentage points with its 95% interval:

* first the static fluid formal core (570 items, 95% bootstrap over the 558 registered units of independence);
* then the interactive episodes and every supporting unit with per-item correctness, each apart. A paired set's rows
  are per-item correctness; its registered headline, pair accuracy, stands beside the table.

It is descriptive: no arm is ranked, eliminated or kept by it, and all three go to recovery. Its code path was run on
real records (TPI1's A sets laid out as eight sets, identity synthesized): complete 8 of 8 in 27 s, the table in 35 s.

**Reported per arm**, both modes, descriptive (`analysis.arm_report`):

* initialization damage: the untrained arm against the teacher, plus its state-evaluation KL / NLL and step-0 CE / KD;
* recovery: the recovered arm against its own initialization, per seed and pooled;
* the remaining gap: the recovered arm against the teacher.

**Registered decision** (`analysis.decide`):

* **Primary:** thinking mode, static fluid formal core (570 items, 558 independence groups), after recovery: the
  three pairwise differences, pooled over both seeds and paired item by item; TPI1's group sign-flip test and group
  bootstrap interval.
* **Holm with m = 3 always.** A pair that cannot be tested still counts in m. The earlier code used m = the number of
  p-values, so two testable pairs at p 0.02 / 0.024 would both have been rejected (0.04 / 0.04); with m = 3 neither
  is (0.06 / 0.06).
* **Formal only when the primary units are complete:** all three arms × both registered seeds × every item. With
  anything missing, every pair of the mode is descriptive (a pair whose two arms are complete keeps its delta and
  interval, no test).
* **Non-thinking** (key secondary) is a second Holm family, m = 3, tested only if the first rejects all three pairs.
* **Nothing eliminates an arm:** no initial KL, CE or generation score stops or drops a unit.

## 7. Cost

L40S at $1.09/h plus 150 GB container disk: $1.1105/h all-in. A session's hard cap is priced at the highest price
the proposal admits ($1.11/h) over setup, transfer, the driver's bound with 5% contingency, and the closeout, recovery
and watchdog reserves. Every figure is an estimate; the basis is `design.json :: cost_model`.

**Phase 1 (proposed: one `sob1_phase1` session).**

| component | hours expected (bound) | basis |
| --- | ---: | --- |
| builds | 2.31 (5.26) | DEPTH rounds 9-20 1.45 h (2.36 h) at the journal's 27.2 s per candidate; the FFN statistics pass and attention gate 4.5 min; **no width statistics pass**: TPI1's sealed statistics, 1.81 GiB, 0.007 h at the median rate (0.27 h at the 2.6 MB/s floor × 1.25); three arms' build and state evaluation 42 min; three inits to the relay 0.03 h (1.65 h at the floor); completion checks and the comparison 3 min (6) |
| untrained arms at 8,192, three | 3.66 (9.0) | RI-S1's untrained D / W / DW capped at 8,192: 3.5-4.5M output tokens each over both modes |
| teacher at 8,192 | 2.81 (4.2) | 6.16M capped output tokens at its measured 608.8 tokens/s |
| **session** | **9.20 GPU h, $10.21 expected** | |

* **Hard cap $22.92** (was $22.55). Reusing the statistics saves the 172 s replay (3 min expected). The bound grows by
  the statistics' floor-rate download (16 min) and the completion checks (6 min), priced into the driver so the
  session's hard deadline covers them.
* **Teardown reserve $1.00**, outside the ceiling.
* **Total $23.92.** The live position on 2026-11-19 06:49Z: $30.2442 left under the $1,000 project cap; balance
  $40.24; no pod, no volume. `launch.py budget --kind sob1_phase1`: fundable.
* One session's ceiling; a replacement attempt takes over every arm whose init verified, every persisted DEPTH round
  and every sealed sample, within what the ceiling leaves.

**Phase 2 (its own proposal since 2026-11-20: `plans/proposal_phase2.json`, from `proposal.json :: phase_2_draft`'s
sessions unchanged).** One session per arm, both seeds, in the order residual width, depth, FFN + attention; the
Stage-3 recovery training and the post-recovery evaluation are this one budget.

| session | training h per seed, expected (bound) | driver h, expected (bound) | USD expected | hard cap |
| --- | ---: | ---: | ---: | ---: |
| residual width only | 17.7 (20.4) | 37.9 (52.6) | 42.59 | 63.43 |
| depth only | 14.6 (19.5) | 31.7 (50.6) | 35.70 | 61.09 |
| FFN + attention only | 14.9 (19.6) | 32.4 (50.8) | 36.41 | 61.35 |
| **both seeds** | | | **114.70** | **185.87 (+ $1 reserve)** |
| one seed (20260726) | | | 58.06 | |

The bound includes each unit's post-recovery evaluation (1.22 h, bound 3.0 h). It also includes the floor-rate copies
of the init, the finals and the last save point. Phase 2's envelope counts only `sob1_train` runs (its own grant):
phase 1's remainder never funds it. At the 2026-11-20 08:30Z position (committed $974.1887 of the $1,000 cap; balance
$35.8108) it needs a project cap of **$1,161.06** (committed + $185.87 + $1.00, rounded up to the cent); nothing
here raises the cap or assumes it raised, and a provider top-up is not a grant.

## 8. Storage and takeover

**Dev box (2026-11-19).** The maintainer authorized retiring eligible historical checkpoints. The consumer audit
covered the registry, readers in code and configs, and the declared consumers and retention purposes. Seven
tombstones were then written and the weights deleted:

* RI-S1's untrained D, W and DW;
* RI-LM1's two merge deltas (each has two hard links).

Their configs, hashes, logs, generations and measurements are kept
(`logs/maintenance/storage/devbox_cleanup_20261119.json`). Free space went from 3.23 GiB to **13.93 GiB (10.70 GiB
freed)**. Kept, with their reasons in that record:

* TPI1's A initialization and finals (the baseline's only copies);
* the teacher statistics;
* D1's reference initialization and the 0.6B canonical initialization (registry-protected);
* other projects' files.

**Second audit (2026-11-19, the corrected brief).** No further checkpoint met AGENTS.md 2.5's conditions, so
nothing more was retired (0 B freed). Every weight left in this project's stores has a recorded reason:

* A's initialization and both finals: the baseline's only copies;
* the teacher statistics: a declared input of this phase (the residual-width build reads their relay copy;
  `checkpoint_registry.json` now names SOB1 as their consumer);
* the Stage-0 activation cache, D1's reference initialization and the 0.6B canonical initialization: registry-protected;
* `~/aad-artifacts/e5` (0.90 GiB, "unreviewed") holds no weight file, only side archives and logs. It is evidence, not
  a checkpoint, so 2.5 does not cover it.

**The dev box's peak at the real landing paths** (`disk.py`, `plans/disk_peak.json`; every rate measured on TPI1's
records: 32.65 B per generated token at most, archives 8.41× smaller, records and logs 0.28%):

| item | path | bound | expected |
| --- | --- | ---: | ---: |
| evidence pulled on every poll (generations and sealed parts) | `<scr>/store/out` | 0.94 GiB | 0.54 GiB |
| one pull's archive in flight (deleted once unpacked; not at the peak) | `<scr>/store/pull_<n>.tgz` | 0.03 GiB | |
| teardown collector archive + its extracted copy | `<scr>/store/` | 0.005 GiB | |
| resume archive (failure path) | `<scr>/resume_<run>.tgz` | 0.11 GiB | 0 |
| three bf16 initializations | `~/aad-artifacts/stage1_single_operator_budget/<arm>/init/<digest16>/` | 11.24 GiB | 11.24 GiB |
| three mappings | `.../<arm>/mapping/<digest16>/` | 0.06 GiB | 0.06 GiB |
| **peak (end of retention)** | one filesystem (`/`) | **12.35 GiB** | **11.84 GiB** |

The evidence bound takes every model at the teacher's capped volume (6.16M tokens) × 1.25. A file `relay.get` is
fetching sits in the store's own cache and is moved, not copied, so it is never held twice. The launch checks the
free bytes of the filesystem the session's `--scr` and the local store are on (`the_dev_box_has_room`), at launch:
the dev box is shared with another project's session, and its free space moved from 14.72 to 19.09 GiB within the
hour this round ran (a 6.74 GiB margin at the dry run). For phase 2, see the next paragraph.

**Phase 2's storage, after the project-wide inventory (2026-11-20, the second brief; `logs/maintenance/storage/
inventory_20261120.json`, `plans/disk_peak_phase2.json`, `plans/resource_snapshot_phase2.json`).**

*The inventory* started from the files and the LFS objects themselves. On the dev box it covered every file of
16 MiB or more under the project's roots, de-duplicated by inode; on the hub, every LFS object of the account's two
private repos against their trees. Both were cross-checked with the registry (regenerated: SOB1's inits are now
protected rows), the tombstones, the retention records and every launcher.
* **Dev box: nothing retired, 0 B freed.** Its ten weight files of 100 MiB or more (22.4 GiB) are all kept:
  * SOB1's three inits: phase 2 trains from them;
  * A's initialization and both finals, and the teacher statistics: the baseline;
  * the Stage-0 activation cache: data;
  * the 0.6B canonical and D1 reference initializations: registry-protected.

  The duplicates found are hard links: the hub cache blob of `candidates.jsonl`, and Research A's scratch evidence.
  Removing either path frees 0 B. The two rows the registry proposes for retirement were refused by
  `retire_eligible_checkpoints.py` (their weights are already gone, and code still names them).
* **Hub: the teacher statistics' redundant relay copy retired, 1.81 GiB freed** (tombstone
  `relay_20261120_tpi1_teacher_residual_statistics`). The dev box's copy re-hashed `e4145e38…`; phase 1's archive is
  complete; no registered plan reads the remote object. Every one of its 13 code readers is classified in
  `relay_weight_dispositions_20261120.json`: they belong to TPI1 (closed) or to SOB1 phase 1 (closed). The path and
  its LFS object were confirmed gone. The relay had 48.74 GiB free before and **50.55 GiB** after (09:39Z).
* **Kept on the hub, the maintainer's decision owed** (each is the only copy, protected, declared by a launcher, or
  not a checkpoint):

  | object | size | why it stays |
  | --- | ---: | --- |
  | `transfer/*.bundle` (396 session code bundles) | 7.75 GiB | reproducible from git; no rule covers retiring them |
  | the two environment wheelhouses | 7.41 GiB | not checkpoints |
  | Phase A attempt 12's five selected leaves (`aadistill-transport`) | 5.55 GiB | only copies; `autoinit_recovery_continuation_launch` declares the path |
  | two permanent controls | 4.44 GiB | only copies; a launcher and a battery config declare them |
  | the 0.6B family's random baseline | 1.11 GiB | only copy; no registry row, no launcher |
  | a Stage-0 finalist | 1.11 GiB | declared by the proposed `independent-gated-E` study |

**This project's dev-box cleanup and the arrangement without a larger disk (2026-11-20, the third brief;
`logs/maintenance/storage/devbox_cleanup_20261120.json`).** The maintainer keeps the dev box at 200 GiB and the hub on
its free plan.

*The whole partition first.* `/` (`/dev/nvme0n1p3`) is the only data filesystem on the machine: there is no other
persistent mount (`/boot/efi` is the 200 MB EFI partition), and 8.20 GiB of `/` is reserved for root. One `du` pass
over `/`, files counted once by inode, found this project holding about 52 GiB and other projects more than 80 GiB.

*This project's cleanup: 5.39 -> 16.12 GiB free, **10.73 GiB freed**, measured on `/` before and after.* Hard links
count once.

| what | freed | rule |
| --- | ---: | --- |
| the 19 earlier Claude sessions' scratch | 0.31 GiB | 47 files (0.5 MB) without a copy elsewhere are kept as one read-back archive under `~/aad-artifacts/maintenance/` |
| this session's own dry-run bundles and outputs | 0.69 GiB | nothing cites them |
| closed experiments' extracted copies with their archive beside them (Research A's tool sessions, Stage 3's two runs, D1-D3, A3, C1, C2, Phase A) | 4.54 GiB | a file goes only when the archive holds the same path with the same sha256; a `RESTORE_FROM_ARCHIVE.txt` gives the command |
| byte-identical duplicates in archival zones made hard links of one copy (388 groups, 779 paths) | about 5.5 GiB | every path stays readable; the git-tracked files, `data/` and this session are excluded |

The 1.51 GiB of "old Claude scratch" in the 2026-11-20 inventory included the current session's own scratch; the
earlier sessions held 0.31 GiB by now. Nothing that is a weight, the A baseline, SOB1's inits and mappings, the teacher
statistics or a registered control was touched, and there is no resume save point on the dev box. Not cleaned: the
public tool-calling datasets (3.03 GiB, re-downloadable, but an open candidate list), the rebuilt caches, the closed
runs' remaining evidence. The hub freed nothing more this round: the relay still has **50.55 GiB** free (15:44Z).

*Other projects' files: read only, the maintainer's decision.* Nothing was changed.

| path | project | reclaimable | how |
| --- | --- | ---: | --- |
| `/var/log/alphaavatar/demo-agent.err.log` | AlphaAvatar's running demo service; no write since 2026-09-21 | 14.88 GiB | root truncation (the service holds it open: deleting the path frees 0 B until a restart) |
| `/var/log/alphaavatar/demo-agent.log` | the same service's live log | up to 19.16 GiB | rotation or truncation (root) |
| `~/alphaavatar-speaker-eval/sessions/v2/` | speaker evaluation; no process, last write 2026-11-13 | 13.24 GiB | if the `.npz` features are no longer needed |
| `~/alphaavatar-speaker-eval/data*/` archives | the same | up to 4.69 GiB | archives beside extracted copies, once checked |
| `/tmp/claude-1000/-home-ecs-user-AlphaAvatar/` | AlphaAvatar's Claude sessions | about 0.6 GiB | the finished sessions only; one is still active |
| `~/.vscode-server/cli/servers/` | the IDE | about 3.8 GiB | the six older server builds |
| `~/.cache/uv/` | every project's uv environments | at most 2.76 GiB | `uv cache prune` |
| the root reserve of `/` | the system | about 6.2 GiB | `tune2fs -m 1` (root) |

*The arrangement* (`design.json :: storage.phase_2_finals.placement`). During an arm's run the free relay has room
for that arm's rotation only (1.00 to 4.09 GiB beside it), so every earlier arm's finals must be on the dev box. The
last arm's finals need no room for a later rotation, so they stay on the relay. Each of the six is retrievable by its
sha256:
* **residual width and depth: on the dev box** (`~/aad-artifacts/stage1_single_operator_budget/<arm>-s<seed>/final/
  <sha16>/`, four fp32 finals, 29.91 GiB). Each is re-hashed there, then its relay copy is tombstoned and removed.
* **FFN + attention: on the relay** (`durable/stage1_single_operator_budget/ffn_attention/<unit>/final/<sha16>/`, two
  finals, 15.03 GiB, by LFS sha256). That leaves the relay 35.5 GiB free afterwards.

| launch, in order | the dev box must have free at that launch (`disk.py --total`: the session's bound peak + earlier arms' expected bytes) | free now 16.12 GiB | relay must accept / has |
| --- | ---: | ---: | --- |
| residual width (first) | 18.50 GiB (two finals land at its closeout) | **2.38 GiB short** | 49.55 / 50.55 GiB |
| depth only | 33.68 GiB (width's finals stay) | 17.56 GiB short | 46.46 / 50.55 GiB |
| FFN + attention | 33.94 GiB (its finals stay on the relay) | **17.82 GiB short** | 46.80 / 50.55 GiB |

The big step is the depth arm's closeout, when its two finals land beside the width arm's two: phase 2 then holds
30.45 GiB expected, 36.64 GiB bound. Its peak over the study, at FFN + attention's closeout, is 30.71 GiB expected and
39.81 GiB bound (`plans/disk_peak_phase2.json`). **Even the first arm does not fit after the cleanup.** Over the whole study the gap is
**17.82 GiB**, or **22.19 GiB** keeping the measured 4.37 GiB swing of the other projects free. Placing all six finals
on the dev box would need 33.94 + 15.03 GiB at the third launch: a gap of 32.8 GiB. No launch threshold was lowered.

*The relay, corrected.* A takeover's need is asked of the relay's live headroom, which already excludes what is held.
So the margin is never "50.55 minus the need" once something is held. The earlier table's 22.03 and 8.49 GiB margins
were wrong: with one save point, or with seed 1's final held, the width arm's true margin is **about 1.00 GiB**, as on
a first run.

| the residual-width arm | held on the relay | its live headroom | the remaining rotation needs | margin |
| --- | --- | ---: | ---: | ---: |
| first run | nothing | 50.55 GiB | 49.55 GiB | 1.00 GiB |
| takeover | seed 1's one valid save point (21.03 GiB) | 29.52 GiB | 28.52 GiB | 1.00 GiB |
| takeover | seed 1's final (7.49 GiB) | 43.06 GiB | 42.06 GiB | 1.00 GiB |
| takeover, two save points left by a failed removal | 2 x 21.03 GiB | 8.50 GiB before the dev box's reclaim, 29.52 after | 28.52 GiB | 1.00 GiB |
| depth only / FFN + attention, after the previous arm's rotation | nothing | 50.55 GiB | 46.46 / 46.80 GiB | 4.09 / 3.75 GiB |

**HF PRO is not needed.** No peak exceeds the free plan: the largest is the width arm's, 49.55 GiB against 50.55 GiB.

**Fixed: streams under `evaluation/live/`.** The runners write their live streams beside the units they serve,
`<unit>/../../live`, which is `evaluation/live/` in this layout. The poll's listing, `take_over` and the resume archive
excluded only a top-level `live/`, so they would have pulled, archived and carried growing streams. All three now
exclude a `live/` directory at any depth (`train_driver.is_live`, `FIND`).

**Fixed: the remote listing and the final hashes.** An unreadable listing (a timeout, a transport error, a non-zero
exit, an answer without its completion marker) is retried three times. If it still cannot be read, `pending` is
unknown, never 0, so the products cannot read as secured. Before, a failed `find` read as an empty listing and
"nothing pending". At teardown, every file of the pod's final listing is checked by sha256 against the dev box's copy;
a missing or different file is pulled again and checked once more (`final_hash_check`).

**Fixed (third brief, 2026-11-19): files the pod moved after the dev box pulled them.** Once the final listing is
read, every file this session pulled that the listing no longer holds at its path (the pod moves files only into
`superseded/`: an isolated part, a generations file composed again) leaves the dev box's active tree too: it goes to
the path where the pod keeps the same bytes (or, already pulled there, the active duplicate is removed), otherwise
under `superseded/_withdrawn_on_pod/` (`withdraw_moved`). So it is in neither the dev box's completion check nor a
resume archive nor the next takeover. The poll's `pending` now counts the listing's files not yet held; it used to
be the listing's size minus the index's, which a pod-side move drove below 0, so the products could never read as
secured. Regression: pull, isolate on the pod, then take over.

**Fixed: local retention.** `retain` is COMPLETE only when every arm's initialization and mapping is on the dev box's
disk as its init record's manifest, re-hashed there, not taken from the fetch's answer. A missing init record, a
relay list that is empty or not the manifest, a failed fetch, a filesystem without room or different bytes leave it
INCOMPLETE, naming what is missing (exit 12). The verified products stay recorded, so the next retain fetches only
the rest.

**Fixed (second correction, 2026-11-19): phase 1's completion on both closeout paths.** The shared closeout runs a study's own check
after the outcome's money, teardown and reconciliation record is written (`Launch.completion_check`), so the formal
`launch`, which closes out after teardown, and a manual `closeout` run the same `phase1_completion`. Before, only the
manual path re-checked phase 1. Phase 1 is recorded COMPLETE (`closeout/phase1_completion.json`) only when the eight
sets are complete on the secured copy AND `retain` has every arm's initialization and mapping on the dev box as its
manifest; otherwise INCOMPLETE, every unfinished item named (`sets`, `inits`, `modes`, `local_retention`). A failed or
interrupted session is torn down and reconciled as before; its durable products and sealed parts stay for the next
attempt's takeover.

**Fixed (2026-11-20): phase 2's takeover and completion.** Reproduced on `84c48bea`: SOB1's phase-2 driver was
TPI1's, whose state function (`a1_units.state`) called a unit `done` when every evaluation directory held a
`formal_report.json`. A `{}` report in both modes, with no final weights anywhere, skipped the unit. The phase-2
driver is now this study's own (`train_driver.py`, TPI1's durability helpers unchanged):
* a unit is done only when its seed's final weights are durable and published (`completion.final_problems`) AND both
  sets are complete against that final's `model.safetensors` sha256 (`completion.check`);
* every set is bound to the arm, the training seed, the mode, the registered protocol and those weights: each sealed
  part's generation id names the target `<arm>_s<seed>` and mode, and its package record names the arm, seed, unit,
  protocol and weights;
* a `{}` or stale report is rescored from the sealed generations, with no model loaded;
* another seed's parts, or parts under other weights, are isolated, and only the missing items are generated;
* a final that exists anywhere is restored, never retrained; training happens only when no final exists.

`completion.session` / `study` and `launch.phase2_completion` / `study_completion` put this on both closeout paths
(section 6). The regressions go through `train_driver.main` with a `{}` report and no final, a `{}` report beside a
durable final, and another seed's shards, and through a three-arm closeout to the study's completion:
`tests/test_single_operator_budget.py`.

**Fixed (2026-11-20, the second brief): four takeover and launch defects**, each with a counter-example through the
real entry point (`tests/test_single_operator_budget.py`):
1. *The authorized arm against the plan actually run.* The launcher's gates compared the artifact with records of its
   own arm, while the pod command ran the `--arm` plan. Now:
   * `launch.py launch` refuses at $0, before the bundle, the provider or the relay is read, when the artifact's arm,
     plan, plan sha256 or init record sha256 is not what `--arm` would run (`arm_binding_problems`);
   * the runner's gate `the_launch_runs_the_authorized_arm_and_plan` does the same;
   * the pod's driver checks the plan it loaded against the artifact before fetching anything.

   Counter-example: a residual-width artifact launched with the depth plan is refused all three ways.
2. *A final traceable to its seed.* A takeover checks the final found (`completion.final_identity_problems`): the arm,
   the seed's relay path `<root>/<arm>/<unit>/final/<sha16>`, and the training completion record archived with the
   weights. That record must match the evidence's `train/` copy by sha256, carry this unit's config hash, and record
   all 9,701 steps. A final that does not check out stops the unit: it is never adopted and never retrained.
   Counter-example: the two seeds' `final.json` swapped, each directory keeping its own legitimate record, is refused.
   Legitimate finals are still taken over; `{}` reports are only rescored; missing items are only filled in.
3. *The relay preflight by what a resume adds* (`rotation_need` / `relay_state`, the table above): 49.55 / 28.52 /
   42.06 GiB for a first run, one valid 21.03 GiB save point held, and seed 1's final durable.
4. *A failed removal never read as retired.* `relay.remove` answers `{"verified": false}` on a timeout, and the
   rotation took any non-empty `removed` as done. Now a record is retired only when `verified_absent` is true; old
   records keep that meaning. A failed removal stays PENDING (still counted) and is retried before the next large
   upload, before a final's upload, and at a takeover, only for a save point a later published one supersedes.
   `relay.remove` also keeps an LFS object that a path which stays still references, and reports what it freed. A
   save point found only in the relay listing now enters `savepoints.json`, so it is retired, not leaked.
   Counter-example: the first removal fails, the pod is lost, the next attempt retires it before uploading anything,
   and the latest save point is removed only after the final hands off.

**Fixed (2026-11-20, the third brief): the free relay's two interruptions**, each with an offline counter-example
(`tests/test_single_operator_budget.py`):
1. *Two save points left on the relay.* `step_002208` handed off, but `step_001104`'s removal failed and the pod was
   lost. The relay then held both (8.50 GiB of live headroom left), the preflight asked 21.03 GiB, and the launch was
   refused before any pod could retry the removal. Now the launch's relay gate first runs `launch.reclaim_superseded`
   on the dev box (a real or a dry run alike; it creates nothing):
   * the save point a takeover restores is checked: the latest listed and present, with every file, its LFS sha256 the
     record's, a planned step, and the same `resume_identity.json` sha256 as each one it supersedes;
   * each older valid save point is then removed (`relay.remove`, permanent, its pointer with it) and confirmed absent
     (`storage/relay_reclaim_phase2.json`);
   * the need is asked of the live headroom: 28.52 of 29.52 GiB.

   The latest valid save point always stays. An older one of another resume identity, or one beside a later save point
   whose bytes differ, is never removed. Beside the unit's own final, every save point is superseded, but only once
   the final's archived completion record proves it is the unit's (`completion.final_identity_problems`).
   `rotation_need` no longer credits a save point held beside a final before it is gone. `relay.remove` now always
   deletes the record's pointer.
2. *A final's retirement interrupted.* `rotate_relay` writes the tombstone, then removes the relay copy. When the
   removal failed, or took effect without its answer being recorded, the next closeout stopped on "a tombstone already
   exists". Now a pending tombstone with the same id, relay path, weights sha256 and members is continued:
   * the dev box's copy is re-hashed file by file first; a copy that does not re-hash keeps the relay copy;
   * the removal is retried (an absent copy confirms absent);
   * the tombstone and `relay_rotation_phase2.json` are completed, with whether the copy was still there.

   A tombstone of that id for other bytes stops that final. The arm's evidence archive is on the relay before any
   final's copy goes. The counter-examples cover both cases: the tombstone is written and the removal fails; the copy
   is removed and the answer is lost.

**Takeover.** TPI1's durability helpers, with this round's two repairs:

* *An unreadable relay listing* is read up to five times. If it still cannot be read, the evidence's own records are
  used when there are any; with none, the unit refuses to start. Before, a listing timeout with no local records
  started the unit from step 0.
* *Weights verified but their record unpublished:* the next watcher tick, and the final's bounded retries, publish
  only the record (`hand_off_with_retries`); the weights are never uploaded again. Before, a save point whose handoff
  failed was never handed off: its predecessor stayed, and a lost pod fell back to it.
* *Slow transfers:*
  * every copy's timeout comes from its bytes at the floor rates (uploads 120 MB/s, verifying downloads 2.6 MB/s,
    × 1.25);
  * a training session stops early enough to secure one save point and the final before its hard deadline (a 2B save
    point's floor-rate verify is up to 2.4 h);
  * an old save point is retired only after the new one hands off;
  * regressions: `test_takeover.py`.
* *Evaluation:* each finished sample is sealed as it is written. A later attempt takes over every complete set as it
  is, and for the rest generates only the items, per unit and seed, without a sealed record, each under the seed its
  item derives (the completion check above decides which).

## 9. Launch path (phase 1)

```text
launch.py budget --kind sob1_phase1                                      # $0: fundable now?
launch.py grant --decision '<the maintainer's words>' --granted-by maintainer
launch.py issue --kind sob1_phase1 --run-id <id> --expected-usd 10.21 --hard-cap-usd 22.92
git commit <the authorization> (the session commit) ; stage the bundle at that commit
launch.py launch --kind sob1_phase1 --run-id <id> --session-commit <c> --bundle <b> --scr <s> --max-price 1.11 --disk-gb 150
                                                                         # its closeout runs phase1_completion (sets + retain)
launch.py closeout --kind sob1_phase1 --run-id <id> --scr <s>            # the same closeout by hand, e.g. after a stalled fetch
launch.py retain --kind sob1_phase1 --run-id <id> --from <s>             # every arm's init and mapping; exit 12 until COMPLETE
analysis.py phase1-table --evidence <s>/store/out --write <dir>          # the comparison, re-derived at $0
plan.py --arm <arm> --init-record <its committed init_record.json> --write    # per arm, before phase 2
```

Phase 1 is closed in code (`kinds.sob1_phase1.closed`): `issue` and `launch` now refuse it, and `closeout`, `retain`
and `analysis.py` still run.

## 9a. Launch path (phase 2, after the maintainer's grant)

```text
launch.py budget --kind sob1_train --arm residual_width                  # $0: fundable now? (phase 2's own grant)
launch.py grant --kind sob1_train --decision '<the maintainer's words>' --granted-by maintainer
launch.py issue --kind sob1_train --arm residual_width --run-id <id> --expected-usd 42.59 --hard-cap-usd 63.43
git commit <the authorization> (the session commit) ; stage the bundle at that commit
launch.py launch --kind sob1_train --arm residual_width --run-id <id> --session-commit <c> --bundle <b> --scr <s> \
    --max-price 1.11 --disk-gb 150                                       # its closeout runs phase2_completion
launch.py closeout --kind sob1_train --arm residual_width --run-id <id> --scr <s>   # the same closeout by hand
launch.py retain --kind sob1_train --arm residual_width --run-id <id> --from <s>    # both finals at their placement
launch.py study-completion                                               # $0: six finals, twelve sets, the comparison
```

**Phase 2's second $0 dry run (2026-11-20 on `c8e28ba1`, clean tree, after the four fixes and the inventory;
`plans/dry_run_20261120b.json`).** Nothing was created and nothing uploaded.
* **Passed as measured:**
  * phase 1 refused as closed (`issue`, `launch`; `budget` $0);
  * a residual-width artifact launched with `--arm depth_only` refused at $0 by the real launcher ("No provider
    resource was created"), before the provider or the relay;
  * the bound records and each init on the relay with its recorded hashes, for all three arms;
  * the relay room for the remaining rotation on the live relay state (nothing held yet), each ACCEPTED (200):
    residual width 49.55 GiB, depth 46.46 GiB, FFN + attention 46.80 GiB;
  * the three plans re-derive; phase 1 still COMPLETE on its secured copy.
* **Refused as measured, for the owed decisions only:**
  * `issue`: no phase-2 grant; above an arm's cap; funding (cap $1,000, $25.8113 left, balance $35.81);
  * the teardown reserve, the same;
  * **the dev box's room for every arm** (5.34 GiB free for 18.44-18.54 GiB).
* **Would pass, assumed and not measured:**
  * with the project cap at $1,161.06 and the $151.06 top-up ($186.8708), every arm's funding and reserve checks are
    empty;
  * with the system disk grown by 64 GiB, 69.34 GiB would be free, against an 18.50 GiB first session and 54.84 GiB
    after three.

**Phase 2's $0 dry run (2026-11-20 on `5e47c52d`, clean tree; `plans/dry_run_20261120.json`).** Nothing was created and
nothing uploaded.
* Phase 1: `budget` CLOSED with $0 remaining; `issue --dry-run` and `launch --dry-run` refused at $0 ("the sob1_phase1
  kind is CLOSED").
* Phase 2's money: `budget` reads `plans/proposal_phase2.json` (per-arm caps $63.43 / $61.09 / $61.35). `issue` was
  refused at $0 three times:
  * a real artifact: no phase-2 grant;
  * $63.44 for the width arm: above its cap;
  * the dry-run artifact at $63.43: the cap plus the $1.00 reserve exceed the $25.8113 the $1,000 project cap leaves
    and the $35.81 balance.

  Re-read with the project cap at $1,161.06 and the balance after the $151.06 top-up ($186.8708), every arm's funding
  and reserve checks are empty, and the whole ceiling plus reserve fits.
* This study's gates, against live state:
  * the bound records: pass for all three arms;
  * each initialization on the relay with its recorded hashes: pass for all three;
  * relay room: pass for depth (46.46 GiB) and FFN + attention (46.80 GiB); **REFUSED for residual width** (49.55 GiB,
    403 "Private storage limit reached");
  * **dev-box room: REFUSED for all three** (6.90 GiB free for about 18.2 GiB);
  * the teardown reserve: refused until the cap and the top-up.
* The launch spec: driver `train_driver.py`, closeout check `phase2_completion` (arm bound), the A-MIX pack's three
  relay inputs by sha256, driver bound 3,155.2 min. The pod command passes the plan, the prefix root, the unit deadline
  20.4 h and the run id as the attempt.
* The three plans re-derive. Phase 1's completion re-checked COMPLETE on its secured copy under the generalized code.

**The third $0 dry run (2026-11-19 19:11Z on `8fa9f84c`, after the second correction; first run at 19:10Z on
`ee2ffc6b` with the same results; `plans/dry_run_20261119c.json`).** Nothing was created.

* The driver's `--check-only`: CONTRACT_OK; the four operator config hashes are the journal's (FFN, DEPTH, ATTENTION,
  RESIDUAL_WIDTH) and both selection records reproduce.
* `budget`: phase 1 fundable ($22.92 + $1 inside the $30.2442 left; balance $40.24); phase 2 refused (no cap).
* `issue --dry-run` at $10.21 / $22.92: an artifact outside the run directory; a cap above the kind's, a `sob1_train`
  artifact and a real artifact without a grant were each refused.
* `launch --dry-run`: every runner gate passed (the plan at $1.11/h, all-in $22.9165 inside $22.92, hard stop $22.48;
  the live L40S quote $1.09/h, stock low; account funds $40.24 against $22.48 and no unaccounted resources; study
  identity `0e79cde9836a`; bound files; the project cap). It stopped where a dry-run artifact must: the authorization
  is not in the session commit.
* This study's five gates passed: the reserve; the relay would accept 11.42 GiB; the bound records; not a training
  session; the dev box holds the bound peak (18.77 GiB free for 12.35 GiB, margin 6.41 GiB).
* The phase-1 launch spec carries `phase1_completion` as its closeout check; its one relay input is the sealed
  statistics at sha256 `e4145e38…`.

**The second $0 dry run (2026-11-19 10:23Z on `8012d22e`, run again at 10:35Z on `06d9fd39` after the digest fix,
with the same result; `plans/dry_run_20261119b.json`).** Nothing was created.

* The driver's `--check-only` passed: the session derives the journal's four operator config hashes, and both
  selection records reproduce.
* `budget`: phase 1 fundable ($22.92 + $1 inside the $30.2442 left; balance $40.24); phase 2 refused (no cap).
* `issue --dry-run` with $10.21 / $22.92: an artifact outside the run directory. A cap above the kind's, a
  `sob1_train` artifact and a real artifact without a grant were each refused.
* `launch --dry-run`: every runner gate passed (the plan at $1.11/h, all-in $22.9165 inside $22.92; the live L40S
  quote $1.09/h, stock low; account funds and resources; study identity, bound files, the project cap). It stopped
  where a dry-run artifact must: the authorization is not in the session commit.
* This study's five gates, called directly, passed: the reserve; the relay would accept 11.42 GiB; the bound records;
  not a training session; **the dev box holds the bound peak** (19.09 GiB free for 12.35 GiB).
* The sealed statistics are the session's one relay input, at sha256 `e4145e38…`.
* Not exercised: staging the bundle and the create. They follow a grant.

The first dry run (`plans/dry_run_20261119.json`, commit `f56b72c0`) found the proposal's first cap $0.02 under the
runner's all-in; that was fixed then (`test_the_hard_cap_is_the_runners_all_in`).

## 10. Decisions for the maintainer

1. ~~Grant phase 1~~: granted 2026-11-19 and COMPLETE 2026-11-20 ($4.4348 of $22.92). Closed in code the same day
   (the grant's `kinds.sob1_phase1.closed`): `issue` and `launch` refuse it and its $18.4852 remainder is void.
2. Phase 2 (prepared at $0 on 2026-11-20; the maintainer approves the three together):
   * **the project cap: $1,161.06** (committed $974.1887, read again at 15:44Z, + the $185.87 ceiling + the $1.00 reserve). The
     maintainer tops the provider balance up by $151.06 ($35.8108 -> $186.87);
   * **phase 2's grant** from `plans/proposal_phase2.json` (`launch.py grant --kind sob1_train --decision ...`);
   * ~~grow the dev box's disk~~: not approved (2026-11-20, the third brief). The arrangement without it (section 8)
     keeps residual width's and depth's finals on the dev box and FFN + attention's on the relay. After this project's
     cleanup (16.12 GiB free) the dev box is **2.38 GiB short for the first arm and 17.82 GiB short over the study**
     (22.19 GiB with the measured 4.37 GiB swing kept free).
   * **other projects' files** (read only, section 8's table) close it. The smallest set is the stale
     `/var/log/alphaavatar/demo-agent.err.log` (14.88 GiB, root truncation) plus about 3 GiB more from the same table,
     or 7.3 GiB more to keep the swing free.
   * The relay needs no purchase: the largest need is 49.55 GiB against 50.55 GiB free.
3. Kept on the hub, for the maintainer: 7.75 GiB of session code bundles, 7.41 GiB of environment wheels, Phase A's
   5.55 GiB of selected leaves, 4.44 GiB of permanent controls, the 1.11 GiB random baseline and the 1.11 GiB
   Stage-0 finalist (section 8; `inventory_20261120.json`).
