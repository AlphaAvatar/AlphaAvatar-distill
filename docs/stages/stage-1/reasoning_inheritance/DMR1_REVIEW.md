# DMR1: depth recovery at SOB1's depth-only structure -- review report

**State: READY_FOR_REVIEW (2026-11-28 UTC), the variant-2 revision after the review of `1e42b144` (section 0U).**
- **V1:** stopped by the maintainer during seed 2, and closed out. Seed 1 is complete and reported descriptively
  (section 00).
- **V2:** revised at $0 into two initializations x seed 20260726, with code, plans, budget and storage prepared
  (section 0R).
- **Not authorized:** no V2 paid run until the budget and the storage decision are approved. V3 is not started.
- **Resources:** nothing is running and nothing is billing: no pod, no volume, no watcher.
- **Stage:** DMR1's three variants are Stage 1 research (structural initialization and reasoning inheritance),
  and their recovery training is Stage 3 (`stage3_recovery`). Variants 2 and 3 moved from `stage-6` on 2026-11-28
  (section 0S). Stage 6 (AGENTS.md 4.8, clarified) starts from the model Stage 5 completes and has no experiment.
- **Branch:** committed and pushed on `research/reasoning-inheritance`, not merged into `main`.
- **Earlier sections:** sections 0 onward are the READY_FOR_REVIEW report of 2026-11-26, kept as written. Their
  "Stage 6" for variants 2 and 3, `stage6_recurrent_depth` and `stage-6/...` paths read as section 0S states;
  where the V2 design differs, section 0R holds.

## 0U. The review of `1e42b144` (2026-11-28): the fixes, the counterexamples, storage and money

The design and the stage were approved and are unchanged:
- two conditions from their own untrained initializations, one seed;
- everything but the initialization the same;
- no hidden-state term;
- the D / M controls reused;
- R1-R4 descriptive;
- Stage 1 research with Stage 3 recovery; Stage 6 not started.

This round fixed what had to be fixed before a launch, at $0. No grant, cap raise, top-up or paid run.

### 0U.1 Save-point rotation: a durable copy at every moment

- **Removed.** The single-copy rotation is gone entirely: TPI1 `savepoint_copies` / `remove_before_upload` and its
  `SavepointMirror` branch, SOB1 `secure_final`'s "remove the last save point, then upload the final" branch (which
  went on uploading even when the removal was refused), and the single branch of `rotation_need`.
  - Those files are byte-identical to before the V2 revision, and the design states the registered rotation
    (`training.durability.rotation_rule`).
  - From the first durable save point on, a valid, discoverable copy exists at every moment: each new save point is
    uploaded, re-downloaded and re-hashed, and handed off while the previous one is still on the relay, which only
    then is removed.
  - A failed removal stays PENDING and is never counted as freed. It is retried before the next large upload and at
    a takeover; `rotation_need` credits a superseded save point only when it goes.
- **Proven through the real takeover entry** (`test_an_interrupted_rotation_is_taken_over_from_a_durable_save_point_never_from_step_0`):
  - **Setup.** Attempt 1's mirror publishes step 1104, then is cut while publishing 2208. The dev box holds only
    what its last poll pulled. Attempt 2 runs `depth_merge.train_driver.main --resume <that evidence>`.
  - **Cuts and outcomes:**

    | cut | the trainer resumed from |
    | --- | --- |
    | the new save point uploading | 1104 (the only one listed and verified) |
    | being verified (the put answered unverified) | 1104 |
    | its handoff unfinished (pointer written, listing never confirmed) | the latest listed and verified |
    | the old one's removal refused | 2208 (handed off; 1104's removal still pending) |

  - **Never step 0.** Each cut leaves at least one durable copy, and the trainer was started with `resume=True`
    from the restored tag every time. The test would fail on the refused single-copy design (its upload cut leaves
    no copy).
  - **Arithmetic.** `test_the_rotation_never_leaves_the_relay_without_a_durable_save_point` checks the launch
    gate's arithmetic: peak = two save points, and no single-copy branch exists. TPI1's and SOB1's existing
    takeover tests (five interruption points, a failed removal retried at the takeover) pass unchanged.
- **The window, corrected.** A copy is the relay `put`: upload, a verifying re-download and a re-hash.

  | | one save point's copy (20.9 GB) |
  | --- | --- |
  | measured (V1's 13 save points) | 151-298 s (upload 104-121 s, verify 38-185 s) |
  | expected (the registered ~120 MB/s each way) | about 5.8 min |
  | bound (the registered floors: 120 MB/s up, 2.6 MB/s down, x 1.25) | **2.86 h** |
  | handoff (pointer, listing) | up to 4 x 120 s more |

  "About 2-3 minutes" was the upload alone. Under the registered rotation the previous save point covers all of
  it.

### 0U.2 Storage under the registered rotation (live 2026-11-28 11:57Z; `plans/storage_plan.json`)

Each V2 session needs **39.99 GiB** on the relay: its two overlapping 19.50 GiB save points + the 1 GiB margin. Its
final (7.46) uploads beside the last save point, below that peak. V2-M starts with V2-D's final and closeout
archive already held (7.59 GiB).

| what is freed first | relay headroom | V2-D | V2-M (after V2-D) |
| --- | --- | --- | --- |
| nothing (the only authorized state) | 27.22 | **short 12.77** | short 20.36 |
| SOB1's residual_width + ffn_attention init relay duplicates (7.50) | 34.72 | short 5.27 | short 12.86 |
| V1's seed-2 save point (19.50) | 46.72 | fits (6.73 margin) | short 0.87 |
| both | 54.22 | fits | **fits, 6.64 margin** |

The relay inventory (60.36 GiB, read live; `storage_plan.json :: inventory`):
- **KEEP, 49.93 GiB:**
  - V1 seed 2's save point (19.50, kept by you);
  - V1 seed 1's final (7.46, the M control's only weights);
  - the two V2 bases (V1's init 3.73 and SOB1's depth init 3.73);
  - the wheelhouses (7.42, every pod installs from them);
  - the permanent controls (4.44);
  - the canonical 0.6B init and random baseline (2.22);
  - the Stage-0 finalist (1.11, named by Stage-2 configs);
  - V2's training pack (0.31);
  - V1's statistics.
- **PROPOSE, 7.53 GiB:** the two SOB1 init duplicates. Their dev-box copies are re-hashed equal to the records and no
  plan reads the relay copies.
- **UNCLASSIFIED, 2.90 GiB:** small packs, corpora and old bundles. Their consumers are not audited, and even all of
  them would close no gap.

**The gap remains.** Even V2-D cannot launch under the recovery guarantee with what is authorized: its launch gate
(`the_relay_has_room`) refuses at 27.22 GiB, as it should. I removed nothing: the duplicates alone unblock nothing,
and V1's save point is yours to keep. Options for your decision:
1. **Free V1 seed 2's save point and the two SOB1 duplicates (27.0 GiB):** both sessions fit, with 6.64 GiB of
   margin at V2-M's peak. The cost: V1 seed 2 could then only be retrained from its initialization (its 5,520 durable
   steps lost).
2. **More relay storage** (an HF storage upgrade): at least 12.8 GiB for V2-D alone, and at least 20.4 GiB for both
   sessions (12.9 GiB if the duplicates go). It is a purchase; I bought nothing.
3. **A second durable backend for the overlapping save point** (for example a RunPod network volume of about 45 GB):
   it is NOT implemented, it pins the session's datacenter and it is a paid resource; listed only.

### 0U.3 The engine validation: what a PASS is bound to, and V2-M's dependency

- **The binding.** The record (`engine_check.json`, schema v2) carries a `binding`:
  - the recurrent config (the schedule) and the execution;
  - the control config;
  - the inputs and the training pack's identity;
  - the gate;
  - the implementation: docstring-free ast digests of `recurrent_depth.py`, `recurrent_depth_vllm.py`, the
    `projected_residual_vllm.py` it imports, `vllm_plugin_path.py`, and the plugin's entry points; plus
    `code_identity` of the check's own probe functions;
  - the environment: the train and vLLM environments' versions and the GPU.

  It binds neither a whole document nor a git commit: a docstring edit changes nothing, a statement edit does
  (tested). The validated weights are recorded and NOT bound: D's and M's weights are the experiment's variable, and
  the check validates the execution path for this schedule. So V2-D's PASS covers V2-M, which has the same schedule,
  control, inputs, gate and implementation (tested), and a plan with another schedule is refused.
- **Reuse re-derives** (`record_problems`):
  - the binding is compared against the plan it is asked for, and on a pod against the live environment;
  - completeness is checked;
  - the verdict is recomputed from the saved metrics; a stored `PASS` string is never trusted.
- **Complete or FAIL.** Before any threshold, the result must match the protocol:
  - exactly n_prompts x new_tokens scored positions per check, with finite metrics;
  - n_prompts vLLM outputs of new_tokens tokens each, ended by length, with one finite log-probability per token;
  - prompts of the registered length;
  - one cache entry per logical layer for both models.

  An empty result scores None, never 0, so two zeros can no longer pass a relative gate.
- **V2-M's dependency, both places:**
  - **plans:** V2-D's plan says `mode: run`; V2-M's says `mode: reuse` from `depth_loop_kept`, naming the canonical
    record `logs/stages/stage-1/depth_recurrence/engine_check.json`. `engine_check.py adopt` writes it only for a
    record without problems;
  - **dev box:** V2-M's bound conditions pin that record's sha256 (setup re-derives them on the pod), and issuing
    refuses while it does not exist; the launch gate `the_engine_validation_applies` re-checks it against V2-M's plan;
  - **pod:** the driver re-checks it against V2-M's plan and the pod's environment before any training, and trains
    nothing otherwise. The check is never re-run for V2-M.
- **The thresholds** are unchanged (the registered ones). The real GPU validation stays in the first approved
  session, before training.
- **Counterexamples** (all pass):
  - a PASS bound to another recurrent config, execution, control, inputs, pack, gate, implementation, check code
    or environment is refused (8 + 1 cases);
  - zero scored vLLM positions, both paths zero, a short or missing output, a non-length stop, mismatched
    log-probabilities, a non-finite log-probability, a non-finite metric, or a missing check are FAIL, and a forged
    PASS over them is refused (9 cases);
  - V2-M with no record, a record bound to another schedule, or an incomplete record: RUN_FAILED with nothing
    trained and no TRAIN_START; and on the dev box, `kind_conditions` refuses while no record exists;
  - a valid record made on V2-D's init stands for V2-M;
  - the shared-cache fault on a tiny real recurrent model still fails the gate.

### 0U.4 The D / M controls are the registered weights

- **The check.** `analysis.recurrence` refuses unless the control it reads is the registered one
  (`registered_controls`, `control_problems`):
  - D: its completion record hashes to the design's, its final weights are `5635c33e…` (SOB1 depth_only seed
    20260726), and both modes' sets are bound to them;
  - M: the same with `ed0ee3e1…` (V1 seed 20260726).
- **Explicit paths too.** The same check applies when `d_out` / `m_out` is passed: a copy anywhere passes, and a
  complete, self-consistent record bound to other weights does not.
- **Tested:** the real evidence passes; a replacement bound to SOB1's other seed's weights, or a half-bound one, is
  refused on both paths. Nothing was retrained or regenerated.

### 0U.5 V1 seed 2, stated precisely

Seed 2 stopped at step **6,590**; its durable, resumable save point is step **5,520**. A resume, if ever authorized,
continues from 5,520 and recomputes the **1,070** steps that were never persisted; it is not a lossless continuation
from 6,590. It stays kept and is not resumed. The pre-fixed rule's CONTINUE and your later stop stay two separate
records. Corrected in sections 00 and 0R.1, the state file, the V1 README, the stage index and the storage plan.

### 0U.6 Money (one ledger reading, 2026-11-28 11:58:20Z; `plans/proposal_dmr1_loop_train.json :: fits_position`)

The sessions' terms do not change: the registered rotation is what the pricing always assumed. The last save
point's copy is bounded at 2.856 h and each final at 1.094 h. V2-M reuses the check, so it has no engine hours.

| | |
| --- | --- |
| V2-D: expected / hard cap | $30.42 / $57.22 (26.7 / 47.4 driver h) |
| V2-M: expected / hard cap | $29.97 / $56.03 (26.3 / 46.4 driver h) |
| both caps + two $1.00 reserves | **$115.25** |
| project cap $1,161.06, billed $1,114.6477, left | **$46.41** → raise **$68.84** (or $11.81 for V2-D alone) |
| RunPod balance | **$95.35** → covers V2-D's $58.22; for both at their caps, top up **$19.90** |

Expected spend is $60.39. Nothing is approved by this table, and V1's closed grants fund nothing here.

### 0U.7 Verification this round ($0)

```text
convergence (one run): depth_recurrence, depth_merge, TPI1, SOB1, tests/docs, tests/architecture
                       : 607 passed / 1 failed / 3 skipped (9.0 min). The failure was the stage index, stale after the
                         seed-2 wording fix in the attribution row; regenerated, tests/docs rerun: 177 passed / 0 failed
live, read-only        : the V2 budget refuses for the project cap; V2-M's bound conditions refuse while no adopted
                         engine record exists; the storage plan and the inventory read live (relay 27.22 GiB free)
GPU validation         : none; still owed in the first approved session, before training
```

### 0U.8 What awaits your decision

1. **Storage** (0U.2): option 1, 2 or 3. Until then no V2 session can launch under the recovery guarantee.
2. **Money:**
   - the proposal ($115.25 at the caps with reserves, $60.39 expected);
   - a project-cap raise of $68.84;
   - a $19.90 top-up for the worst case of both sessions;
   - a grant for `dmr1_loop_train`.
3. **The memory risk on the L40S** (1.85 GiB estimated margin, unmeasured) and, only after a measured OOM, an
   80 GB card.
4. **Unchanged and still open:** whether V1 seed 2's save point stays (it is part of storage option 1).

## 0S. The stage correction (the maintainer's message of 2026-11-28, folded into this round)

### 0S.1 Scope

- **AGENTS.md 4.8.** It keeps its generic name, "Stage 6 — Recurrent depth: shared-parameter repeated execution";
  the root README is unchanged, at the maintainer's request. A 2026-11-28 clarification adds three things:
  - **Starting point:** the model Stage 5 completes (after Stage 3 recovery and Stage 5 RL). Its structure,
    construction, training and evaluation are designed when Stage 6 begins.
  - **Boundary:** shared-parameter repeated execution used as a structural initialization of a teacher-derived
    student before recovery is Stage 1 research, and its recovery is Stage 3, under Stage 3's recipe and packing.
  - **Status:** NOT STARTED.

  Also: the generic mechanisms stay in core; the maintainer's HRM-Text references are queued, not adopted as a
  design; P19's Stage-6 row notes that such Stage-1 studies train under Stage 3's rule; and the history note keeps
  the 2026-11-13 to 2026-11-28 records' wording.
- **The moves** (`git mv`, content unchanged at the move; 21 files with their blob hashes in
  `logs/maintenance/source-relocations/dmr1-stage-correction/v1/source-relocation.json`):

  | from | to |
  | --- | --- |
  | `configs/stages/stage-6/depth_recurrence` | `configs/stages/stage-1/depth_recurrence` |
  | `scripts/stages/stage-6/depth_recurrence` | `scripts/stages/stage-1/depth_recurrence` |
  | `docs/stages/stage-6/depth_recurrence` | `docs/stages/stage-1/depth_recurrence` |
  | `logs/stages/stage-6/depth_recurrence` | `logs/stages/stage-1/depth_recurrence` |

  The four pairs are in `logs/index.json :: historical_paths.map` (`run_layout.resolve_historical`). No stage-6
  directory remains, because a stage exists only with evidence (P2, `test_every_stage_present_has_repository_evidence`);
  Stage 6's definition lives in AGENTS.md 4.8 and the README.
- **The design** (`configs/stages/stage-1/depth_recurrence/design.json`):
  - schema `aadistill.stage1.depth_recurrence.design/v1`, `stage: 1`, execution stage `stage3_recovery`;
  - `config.stage` removed from the changed fields, so V2's units keep SOB1's `stage3_recovery`, and a unit that
    changed it would be refused as outside the declared fields;
  - `stage_6_identity` renamed `recurrent_execution_identity`, its data and packing being Stage 3's registered recipe
    (random order, sequential packing with truncation, 8,192 tokens, causal attention, the registered proportions
    and pre-packing counts);
  - relay root `durable/stage1_depth_recurrence` (nothing was ever written under `durable/stage6_depth_recurrence`),
    local store `~/aad-artifacts/stage1_depth_recurrence` (never created);
  - `stage_correction_2026_11_28` records all of this; `superseded_2026_11_28` is kept as it was written.
- **The chain:**
  - `depth_merge/common.py`: `RECURRENCE_DESIGN`, `RECURRENCE_LOGS`, `RECURRENT_CONFIG`, `recurrence_design()`;
  - analysis, plan, proposal, storage and session read them; the session commit's script list names the Stage-1
    paths;
  - the launcher's study `plan_id` is `stage1.depth_recurrence`, so its authorizations' wire schema is
    `aadistill.stage1.depth_recurrence.authorization/v1`; its logs, identity files and artifact spec are the Stage-1
    paths;
  - the init records' and the engine check's schemas are `aadistill.stage1.*`;
  - pod setup's `dmr1_depth_recurrence` branch imports `stages.depth_recurrence` through the `stages` package path,
    so it is unchanged apart from its comment.
- **The indexes:** the stage attribution moves the experiment to Stage 1 and drops Stage 6's definition; the run
  index, the stage index and the navigation are regenerated; the checkpoint registry's `referenced_by` names the
  moved design.
- **Regenerated, not yet consumed by any run:** the two init records, the two plans, the proposal and the storage
  plan. **Not rewritten:**
  - the 2026-11-26 step-0 diagnostic (moved byte for byte; its schema string keeps `stage6`);
  - V1's run records (authorizations and session records naming the old script paths as bound identity);
  - V1's `design.json` (bound by V1's plan, grants and session commits);
  - the decision records.

  Their old paths resolve through the map.

### 0S.2 What did not change

- **The science:** the two V2 conditions, seed 20260726, the 36-call schedule, the losses (CE / Top-200 + tail KD;
  no hidden-state term), the recipe, the comparisons R1-R4, the engine validation, the budget terms ($60.39
  expected, $113.25 in caps) and the storage need (28.09 GiB).
- **The artifacts:**
  - execution identities: KEPT `c09176af…`, MERGED `ccdc8f5e…`, as before the move;
  - the weights (`76c8ca19…`, `99f63821…`), the bases' init records (`413686f2…`, `f5713179…`) and their relay
    copies are untouched;
  - nothing was rebuilt, re-counted, retrained or regenerated because of the move;
  - the identity checks are not relaxed: `init_record_problems` still binds base, weights, relay copy, config,
    schedule and digest.
- **Takeover and reuse:**
  - V1's attempts (`--resume-from` their secured evidence) and every relay save point and final keep their
    prefixes and records;
  - the V2 chain's takeover is the tested one (`test_each_arm_runs_through_the_real_driver_..._a_takeover_does_nothing`,
    for depth_merge and both loop arms).

### 0S.3 Verified after the correction ($0)

```text
experiment suites at the new location (depth_merge, depth_recurrence) + tests/docs
                                         : 217 passed / 0 failed (after re-deriving the plans and the proposal
                                           against the final design, and repointing two link targets in this
                                           report's 2026-11-26 sections; their prose unchanged)
tests/architecture + test_hidden_state   : passed in the same convergence run (the core declaration covers this
                                           round's core changes; nothing in src/aadistill moved)
SOB1 / TPI1 suites                       : 112 passed / 3 skipped before the move; their code was not touched after
live, read-only                          : `launch.py --kind dmr1_loop_train --arm <arm> budget` refuses for both arms
                                           ONLY for the project cap; both committed init records pass
                                           `init_record_problems` (KEPT c09176af..., MERGED ccdc8f5e...); the plans,
                                           the proposal and V1's plan are their derivations; V1's training kind
                                           still reads $26.7722 spent / $37.6878 left
old paths                                :
  - `resolve_historical("logs/stages/stage-6/depth_recurrence/plans/step0_diagnostic_20261126.json")` gives the
    Stage-1 file, byte-identical to HEAD's blob;
  - V1's bound script paths resolve to `scripts/stages/stage-1/depth_recurrence/...`
stage index                              : stages present = stage-0, stage-1, stage-2, stage-3 (no stage-6)
resources                                : no pod, no volume, $0/h (runpodctl and the live position)
GPU validation                           : none; the engine validation is still owed in the first approved session
```

### 0S.4 What still needs your decision (the real open questions)

1. **Money.**
   - The proposal itself: $60.39 expected, $113.25 in caps plus $2.00 in reserves, one session per arm.
   - A **project-cap raise**: the cap leaves $46.41, so both sessions need about **$68.8** more, or about **$11.8**
     to run V2-D's session alone first.
   - A grant for `dmr1_loop_train`.
2. **Storage.** Superseded: see 0U.2 (the single copy was refused; a decision is needed before either session).
3. **The engine gate's thresholds.** For both checks: mean |dlogp| <= 3 x control + 0.02 nats and top-1 agreement
   >= control - 0.02, with the shared-cache fault required to fail. They are fixed now, before any reading; change
   them now or never.
4. **Memory on the L40S.** The estimated margin is 1.85 GiB, not measured. If a measured OOM survives the
   protocol-neutral steps, the 80 GB card is a separate decision.
5. **V1's seed-2 save point** (19.50 GiB on the relay). Seed 2 stopped at step 6,590; the durable, resumable point
   is step 5,520, so a resume would recompute 1,070 steps. It is kept as you asked; whether
   it stays after this round is yours to decide.

## 0R. Variant 2 revised: two initializations x one seed (the maintainer's brief of 2026-11-28)

### 0R.1 V1's stop: closeout and resources (verified, not re-run)

- **Records** (section 00, decisions.md 2026-11-28):
  - seed 20260726 is complete: final `ed0ee3e1...` with both modes' sets;
  - seed 20260801 was stopped at step 6,590 by the maintainer's decision;
  - the pre-fixed rule's CONTINUE (`runs/dmr1_merge_train_20261127_201532/evidence/seed1_stop_check.json`) and the
    maintainer's later stop are kept as two records; the threshold is unchanged;
  - V1 is NOT the registered two-seed completion.
- **Kept:** seed 1's final (relay), evaluation sets, training log and identity records (the dev box's secured
  evidence and the repo's closeout). Also kept: seed 2's verified save point at step 5,520 on the relay -- model,
  AdamW state, CPU and CUDA RNG states, `consumed_blocks` (the data position) and the resume identity. Seed 2
  stopped at step 6,590, but the last DURABLE save point is step 5,520. A later resume, if authorized, continues from
  5,520 and recomputes the 1,070 steps that were never persisted -- not a lossless continuation from 6,590, and
  never a retraining from step 0.
- **Resources** (2026-11-28 10:18-10:25Z):
  - the pod was deleted and the provider confirms it gone;
  - `runpodctl pod list` and `network-volume list` both read `[]`;
  - the watchers ended (`bh4vyz4so` the stop-rule watch, `bopinhajf` the session watch), and two stale log tails
    left from the build sessions were stopped.
- **Money:**
  - training kind: $26.7722 of $64.46 (two attempts);
  - build kind: $1.6091 of $5.43;
  - neither grant carries over to V2.

The conclusion is the brief's: the current alpha merge, on the completed seed 1 under the current recovery recipe, is
below the depth baseline, so no second seed was funded and the programme moves to the mechanism question below. This
says nothing about layer merging in general or about whether the representation is recoverable.

### 0R.2 The two conditions and their identities

| | single execution (16 block calls) | recurrent (36 block calls) |
| --- | --- | --- |
| **D**: the kept teacher blocks, unmerged | SOB1 depth_only s20260726 (reused, sealed) | **V2-D** = `depth_loop_kept-s20260726` (new) |
| **M**: V1's alpha-merged blocks | V1 depth_merge s20260726 (reused, sealed) | **V2-M** = `depth_loop_merged-s20260726` (new) |

D is the kept layers' original teacher weights, not a uniform average. M is V1's built and verified alpha-merged
init. Both V2 units start from their training-pre initialization, never from SOB1's or V1's recovered finals. No
statistics, alignment, mapping or initialization is rebuilt.

| | V2-D (`depth_loop_kept`) | V2-M (`depth_loop_merged`) |
| --- | --- | --- |
| fixed by | `design.json :: arms.depth_loop_kept.base = KEPT` | `arms.depth_loop_merged.base = MERGED` |
| base init record | SOB1 `sob1_phase1_20261119_194259/evidence/depth_only/init_record.json` (`413686f2...`) | V1 `dmr1_merge_build_20261127_083526/evidence/depth_merge/init_record.json` (`f5713179...`), via V1's training plan |
| weights (model.safetensors) | `76c8ca19...` | `99f63821...` |
| relay copy (verified) | `durable/stage1_single_operator_budget/depth_only/init/36888440ef83bfd6` | `durable/stage1_depth_merge/depth_merge/init/17fc769bd874e401` |
| recurrent config | `recurrent_config.json` `3b8c0a8a...` (the same file) | the same |
| recurrent init record | `logs/stages/stage-1/depth_recurrence/inits/KEPT/init_record.json` (`50fdbb03...`) | `.../inits/MERGED/init_record.json` (`b43b4927...`) |
| execution identity (`execution_digest`: weights + config) | `c09176af...` | `ccdc8f5e...` |
| training plan | `plans/training_plan_depth_loop_kept.json` | `plans/training_plan_depth_loop_merged.json` |
| engine validation | in this session, before training | none (it runs once, in V2-D's session) |

**Only the initialization differs.** Both plans are derived from SOB1's depth unit of seed 20260726 with the same
declared fields replaced (`plan.py`). Their configs differ in the initialization (weights sha, init record and its
sha) and in the names that follow from the arm (run name, student path, output directory) -- tested
(`test_the_loop_arms_plans_differ_..._only_in_the_init`).

Everything else is identical:
- **Schedule:** 16 blocks on 36 logical-layer calls, derived from the real mapping. Each call reads the previous
  call's output, with no detach, and every call's gradient sums into the shared parameters.
- **Generation:** one KV cache per logical layer.
- **Data:** the same teacher lineage, data, proportions, seeded order and packing (8,192 tokens, sequential, with
  truncation, causal attention).
- **Optimization:** micro-batch 1 x 2 accumulation steps, 9,701 optimizer steps, the same AdamW and learning rate,
  CE 0.25 + Top-200 + tail KD 1.0, fp32 weights with bf16 autocast, and the same trainable set.
- **Evaluation:** the same tokenizer, chat template, evaluation protocol and per-item seeds.

Nothing is changed for one condition only: no hidden-state loss, iteration count, residual scaling or learning rate.
Recurrence keeps the training data and the optimizer steps, but NOT the compute: about 1.8x the training work per
token and 2.25x the decode work. The report gives measured GPU time, memory and evaluation time, never
"same compute".

### 0R.3 What changed from the 2026-11-26 design, and where

| | 2026-11-26 (kept in `design.json :: superseded_2026_11_28`) | 2026-11-28 | code |
| --- | --- | --- | --- |
| initialization | ONE base, selected by C1 two-seed rule (MERGED iff pooled and both seeds > 0) | BOTH bases, each a fixed condition | `schedule.py` (`base_of`, `init_record(arm)`, `init_record_problems`); `select` / `decide` removed from the chain (git history) |
| arms | `depth_loop`, `depth_loop_hidden` | `depth_loop_kept`, `depth_loop_merged`; `depth_loop_hidden` NOT STARTED (no base, plan or kind) | `depth_merge/common.py :: ARMS`; `depth_recurrence/launch.py` (kind `dmr1_loop_train`, `--arm`) |
| seeds | 20260726, 20260801 | 20260726 per arm; no second seed starts | `design.json :: arms.<arm>.seeds`; `plan.py :: seeds_of` |
| comparisons | C2 (loop vs the selected single arm), C3 (V3 vs V2), tested, pooled over two seeds | R1 = V2-D - D, R2 = V2-M - M, R3 = V2-M - V2-D, R4 = I = (V2-M - V2-D) - (M - D); descriptive, single seed, 95% group-bootstrap intervals | `analysis.py :: recurrence`, `recurrence_modes`, `interaction_groups` |
| engine validation | "owed before or at the first paid run" | an explicit first stage of V2-D's session, in its budget, gated | `depth_recurrence/engine_check.py`; `depth_merge/train_driver.py :: engine_validation` |
| diagnostics | val CE / KD | + the eval-only state probe per logical layer | core: `hidden_state.StateProbe`, `train.py :: state_probe` |
| save points on the relay | two copies (registered rotation) | two copies, unchanged (a single copy was proposed 2026-11-28 and REFUSED in review; removed again, 0U.1) | -- |
| money | an estimate, never proposed (`estimates.json`, removed) | a proposal per arm (`plans/proposal_dmr1_loop_train.json`) | `proposal.py :: proposal_v2` |

C1 is not completed: no C1 state is written, nothing is derived from the old selector, and V1's seed 2 is not resumed
to satisfy it.

The core stays generic. `src/aadistill` gains the eval-only probe (pairs, sequences and stride are the caller's) and
nothing names an arm, a model size or a schedule. The arms, the 36-layer schedule, the seeds and the probe's
settings live in the design and the experiment scripts.

### 0R.4 The question, the statistics and how a result is read

- **Question:** does the alpha merge's single-execution deficit persist when the same block is executed
  recurrently?
- **What is reported:**
  - R1-R4 with the four conditions' accuracies, in both modes;
  - primary: thinking mode, the static fluid formal core (570 items, 558 independence groups), seed 20260726;
  - the item contrast is per item; R4 is formed per item from its four records (V2-M, V2-D, M, D), so each group
    resample keeps their pairing (10,000 resamples, `default_rng(20261120)`). Its arithmetic is tested: R4 = R3 -
    (M - D) item for item, and identical conditions give exactly [0, 0].
- **Reported beside it, as registered:** non-thinking, the interactive episodes and every supporting unit.
- **Not a test:** this is a single-seed exploration with no test and no verdict. The intervals cover items with each
  checkpoint fixed, not training randomness. eval-set-1 is development-exposed.
- **Reading:**
  - for reference, M - D is -3.86 pp in thinking mode (seed 20260726);
  - if the two recurrent conditions come out close, the deficit shrinks under recurrence. That does not mean the
    merge is better; the absolute accuracies are read beside it;
  - the round cannot separate "the weighting damaged the representation" from "the recipe suits it less", because
    recurrence also changes the graph, the state distribution and gradient flow. No mechanism is concluded, and V3 is
    discussed after these results;
  - nothing is changed after a result: not the metric, the rules or the threshold.
- **Diagnostics** (the same for both inits, never a loss, never a reason to drop or re-tune a run):
  - the state probe at every validation (step 0, every 1,212 steps, the final step), per logical layer against its
    teacher layer, at fixed rows (the first 2 validation sequences, every 8th KD row): relative squared error, both
    norms, and cosine;
  - validation CE / KD curves;
  - measured seconds per step, first-four-steps memory, evaluation time and termination types;
  - the KEPT recurrent init's 2026-11-26 CPU step-0 reading (val CE 12.46, against the single-execution base's
    3.46) is recorded as it is. It is not a verdict on recurrent training, and both inits get their own step-0
    reading from the run.

### 0R.5 Verification done, and what only a GPU can answer

Run at $0 this round (development checks only where a result decided the next step, then one convergence
validation; no full core suite was run: the changed core is covered by the convergence set, and P8.3 does not owe
one):

```text
core convergence (tests/training, tests/architecture, tests/models,
  tests/infrastructure)                    : all pass after the core changes were declared (the declaration
                                             also covers two earlier undeclared ones: session_runner's
                                             poll_lifetime, e5482309, and search.py's applicability check,
                                             506e83e3)
experiment suites (depth_merge, depth_recurrence, SOB1, TPI1)
                                           : 112 passed / 0 failed / 3 skipped (5.6 min)
budget chain, live and read-only           : `launch.py --kind dmr1_loop_train --arm <arm> budget` refuses ONLY for
                                             the project cap (0R.7)
GPU validation                             : none (no GPU)
```

What the tests establish:
- the state probe through the real trainer: its arithmetic against the plain formulas, and identical losses,
  gradients, weights and RNG state with and without it;
- (superseded: the single-copy rotation's tests were removed with it; the interrupted-rotation takeovers are
  in 0U.1)
- the real driver end to end for depth_merge and both loop arms: the engine check runs before any training, a
  failed check trains nothing, and a takeover repeats nothing;
- the HF cache check on a tiny real recurrent model: the true cache passes against the control, and the
  wrong-iteration-keys fault fails;
- R1-R4's arithmetic.

NOT VALIDATED until the first approved session's engine-validation stage (`engine_validation`, before any training):
- vLLM 0.27.1 importing the recurrent plugin under its own environment;
- torch.compile and CUDA graphs over 36 per-logical-layer `Attention` modules, and the KV-cache sizing;
- HF / vLLM agreement on generation and on independent KV caches -- the check compares each against the same weights
  as a stock 16-layer Qwen3 in the same session, and requires a deliberate shared-cache fault to fail;
- the recurrent model's L40S memory and step time (the first four steps' memory probe);
- the vLLM half of `engine_check.py`, which has never executed.

Gate (fixed now): for both checks, recurrent mean |dlogp| <= 3 x control + 0.02 nats and top-1 agreement >= control -
0.02, the fault failing, and one cache entry per logical layer. A FAIL ends the session with nothing trained.

### 0R.6 Storage (live 2026-11-28 10:25Z) -- SUPERSEDED by 0U.2

**Superseded.** The review refused the single-copy rotation below: it left no durable copy for up to 2.86 h per
save point (not "about 2-3 min": that was the upload alone). Section 0U.2 recomputes storage under the registered
rotation. The text below is kept as it was proposed.


The relay holds 60.36 GiB and accepts 27.22 GiB more. That includes V1's seed-1 final (7.46), V1's seed-2 save point
(19.50), V1's init (3.73: V2-M's base) and SOB1's depth init (3.73: V2-D's base), all kept. The dev box has 6.85 GiB
free (no expansion), and a session's evidence is bounded at 1.5 GiB.

| rotation | relay need for both sessions (peak + 1 GiB) | as is (27.22) | with the two duplicates removed (34.72) |
| --- | --- | --- | --- |
| registered: two save points | 47.58 GiB | does not fit | does not fit |
| **single copy** (proposed) | **28.09 GiB** | V2-D fits; V2-M short by 0.87 GiB | **fits, 6.64 GiB margin** |

The peak is during V2-M's first save point, with V2-D's final on the relay.

The proposal, for your decision:
1. **Single save-point copy** (`training.durability.savepoint_copies = 1`; already in the plans, implemented and
   tested):
   - the previous save point is removed before each upload, and the last one before the final's;
   - cost: during each upload (about 2-3 min, eight times per unit) the unit holds no durable save point; a pod lost
     exactly then restarts that unit's training from step 0 (up to about 24 h).
2. **Remove the relay copies of SOB1's residual_width and ffn_attention inits** (7.50 GiB):
   - their dev-box copies are re-hashed equal to the records, so no artifact is deleted;
   - this is needed only before the V2-M session.

Not proposed: deleting V1's seed-2 save point (you asked to keep it), buying storage, or expanding the dev box.

### 0R.7 Money, and the launch conditions (`logs/stages/stage-1/depth_recurrence/plans/proposal_dmr1_loop_train.json`)

| session | expected | hard cap | basis |
| --- | --- | --- | --- |
| V2-D: engine validation + 9,701 steps + evaluation | $30.42 (26.7 driver h) | $57.22 (47.4 driver h bound) | 8.69 s / step expected (SOB1's 4.83 x 1.8), 13.03 bound; evaluation 2.75 h / 6.75 h (x 2.25); engine check 0.4 / 1.0 h |
| V2-M: 9,701 steps + evaluation | $29.97 | $56.03 | the same, without the engine check |
| **both** | **$60.39** | **$113.25** (+ $2.00 reserves = **$115.25**) | |

- **The project cap does not cover it.** It leaves **$46.41** as the provider bills it (live at commit time, $1,161.06
  cap; $46.99 at 10:25Z, the difference being V1 attempt 2's last billing rows posting after its teardown, with no
  resource existing and the balance moving $0.02). Even the first session's $57.22 cap + $1 reserve does not fit,
  and the budget command refuses for that reason alone.
  - Both sessions at their caps need a cap raise of about **$68.8**.
  - A raise of about **$11.8** covers V2-D's session alone.
- **The balance does cover the first session.** $95.37 covers V2-D's $58.22; the second session is checked again at
  its issue.
- **The step time is not measured.** The 1.8x work model has not been measured on a GPU. The first four steps
  measure it, and the bound is 1.5x the expected time.
- **The memory risk is real.** The estimated reserved peak is 42.54 GiB on a 44.39 GiB L40S (1.85 GiB margin). An
  OOM is met first by protocol-neutral steps (expandable segments, `kd_chunk`); an 80 GB card would need a measured
  OOM and a decision of its own.

**Launch conditions**, after which the two sessions run without stopping in between:
- the maintainer approves this design, the budget, and the project-cap raise;
- a grant for `dmr1_loop_train` is written;
- a storage decision (0U.2), needed before V2-D as well: nothing authorized makes room for either session;
- each session's issue and launch gates pass live (session commit, room, balance, envelope).

Both units then train and are evaluated, and the round stops for review: no seed 2 is started, and V2-M is not
cancelled whatever V2-D shows.

## 00. Variant 1 executed (2026-11-27/28 UTC)

### 00.1 What ran

| step | run | result | all-in |
| --- | --- | --- | --- |
| build, attempt 1 | `dmr1_merge_build_20261127_081622` | statistics pass out of memory; repaired span-major (bitwise-equal statistics) | $0.2048 |
| build, attempt 2 | `dmr1_merge_build_20261127_083526` | COMPLETE: init `17fc769b...`, equivalence worst 2.5e-6 | $1.4043 |
| training, attempt 1 | `dmr1_merge_train_20261127_095843` | the shared launcher's 600 min poll default tore it down at 605.6 min; seed 1 at step 7,310; repaired in core (`session_runner.poll_lifetime`) | $11.2083 |
| training, attempt 2 | `dmr1_merge_train_20261127_201532` | seed 1 resumed from save point 6,624, trained to 9,701 and evaluated in both modes; seed 2 stopped by the maintainer at step 6,590 | $15.5639 |

Seed 1's final is `ed0ee3e1...` (model.safetensors), durable on the relay. Its steps 6,625-7,310 ran twice (attempt
1, lost; attempt 2, kept). The train log holds both copies, and a diagnostic takes the later row per step. The
resumed losses matched attempt 1's to the fourth decimal (step 6,630: 0.2795 vs 0.2799; same learning rate).

### 00.2 Seed 1 against depth-only (static fluid formal core, 570 items, 558 groups; descriptive, one seed)

The registered per-seed statistic (`single_operator_budget.analysis._paired`): mean paired difference by item and its
95% group-bootstrap interval. It is the same seed (20260726) and the same protocol as SOB1's depth-only control.
Record: `logs/stages/stage-1/depth_merge/runs/dmr1_merge_train_20261127_201532/evidence/seed1_stop_check.json`.

| mode | V1 | depth-only | difference | 95% interval |
| --- | --- | --- | --- | --- |
| thinking (primary) | 12.28% | 16.14% | −3.86 pp | [−6.79, −0.87] |
| non-thinking | 12.11% | 17.54% | −5.44 pp | [−8.39, −2.46] |

- The stop rule (STOP iff thinking ≤ −5.0 pp and the upper bound < 0, fixed before any seed-1 result) read
  CONTINUE. The maintainer then stopped the run on reading these numbers.
- Both intervals lie below 0, so on this seed V1 is below depth-only in both modes. The interval works over items
  with each checkpoint fixed. It does not measure training randomness, so this is not a verdict, and C1's registered
  two-seed test did not run.
- Terminations on the core, out of 570 turns per arm:
  - thinking: V1 had 448 natural terminations, 110 degeneration stops and 12 context-limit hits; depth-only had
    433, 106 and 31;
  - non-thinking: V1 519 / 51 / 0, depth-only 529 / 41 / 0.
  - Context-limit hits are right-censored and are scored by the registered scorer like every other turn.

### 00.3 The other units, seed 1 (descriptive; `evidence/seed1_supporting_units.json`)

| unit (n) | thinking V1 | thinking depth-only | non-thinking V1 | non-thinking depth-only |
| --- | --- | --- | --- | --- |
| interactive episodes (144) | 2.78% | 2.08% | 8.33% | 6.94% |
| rag (100) | 51.0% | 58.0% | 67.0% | 77.0% |
| multihop (100) | 35.0% | 49.0% | 55.0% | 63.0% |
| gsm8k (100) | 20.0% | 21.0% | 11.0% | 17.0% |
| math_verified (100) | 0.0% | 6.0% | 3.0% | 8.0% |
| answerability_paired (120) | 32.5% | 39.17% | 33.33% | 45.0% |
| safety_paired (100) | 28.0% | 28.0% | 40.0% | 47.0% |
| knowledge (150) | 2.67% | 2.0% | 2.67% | 4.67% |
| stage3_legacy_150 (150, in-sample: training prompts) | 52.67% | 54.0% | 50.67% | 54.0% |

V1 is at or below depth-only on every supporting unit except the interactive episodes. Those are 1-2 episodes out
of 144, inside noise. Each unit is reported on its own and no pooled score is defined.

### 00.4 Training diagnostic (not a score)

Validation CE gap, V1 minus depth-only, same seed and step:

| step | 0 | 1,212 | 4,848 | 6,060 | 9,701 (final) |
| --- | --- | --- | --- | --- | --- |
| seed 1 | +3.61 | +0.158 | +0.099 | +0.091 | +0.099 (0.646 vs 0.547) |
| seed 2 | +3.61 | +0.178 | +0.089 | +0.090 | — (stopped at 6,590) |

The alpha-merged initialization starts far worse (pre-recovery teacher KL 7.20 vs the deletion init's 3.47). After
about 4,000 steps the gap stops closing and holds near +0.09-0.10 to the end. Seed 2 follows the same path.

### 00.5 Conclusion, as far as it reaches

On the completed seed 1, under the current recovery recipe, the current alpha merge is below the depth baseline.
That is why no more was spent on a second seed, and the programme moves to the next mechanism question (V2 with
both initializations, section 0 of the next revision). This says nothing about layer merging in general, and
nothing about whether the representation is recoverable.

How the stop happened, kept as two separate records:
- the pre-fixed rule's reading, CONTINUE (`evidence/seed1_stop_check.json`, committed `93b4b179` before any seed-2
  result);
- the maintainer's later decision to stop (decisions.md 2026-11-28).

The stop did not meet the rule's threshold, and the threshold is not changed. V1 is not the registered two-seed
scientific completion: seed 1 is complete, and seed 2 was stopped early by the maintainer.

### 00.6 What is kept, money, and what is open

- Kept (nothing deleted):
  - on the relay: seed 1's final (7.46 GiB) and seed 2's verified save point at step 5,520 (19.50 GiB: model +
    trainer state, resumable under the registered contract). Seed 2 stopped at step 6,590; a resume continues from
    5,520 and recomputes the 1,070 unpersisted steps;
  - on the dev box: the secured evidence of both attempts (`~/aad-artifacts/stage1_depth_merge/<run>/store/out`);
  - in the repo: closeouts, `storage/finals_depth_merge.json` (seed 1 verified, seed 2 missing).
- Money:
  - training kind: $26.7722 of $64.46;
  - build kind: $1.6091 of $5.43;
  - DMR1 total: $28.3813;
  - project: $1,113.5131 billed of the $1,161.06 cap ($47.5469 left).
- For the maintainer's review:
  - resume seed 2 from its durable save point at step 5,520 (4,181 steps to 9,701, of which 1,070 repeat the
    unpersisted steps 5,521-6,590; ~5.7 h plus evaluation, inside the training kind's remainder);
  - or close V1 on one seed;
  - and the relay bytes' fate.

Contents: **0 this round** (fixes, the moved finals, storage, the search configuration, what is not GPU-validated,
conditions, money and state) · **0A the second round's report** (kept; section 0 holds where they differ) · 1 scope
and entry points · 2 baseline, teacher, mapping, schedule · 3 variant 1 · 4 variant 2's initialization and binding ·
5 variant 3's hidden-state term · 6 comparisons and statistics · 7 the recurrent mechanism · 8 tests, fixes, what is
not measured · 9 time, memory, storage, money · 10 $0 state and authorization boundaries.

## 0. Third repair round (2026-11-26 UTC; the maintainer's review of `d92f48f8`)

The review accepted most of the second round and found four concrete defects -- variant 1's identity takeover, its
remote handoff (and the window before the dev box receives the latest build record), the takeover's room arithmetic,
and the default search entry's PCA configuration -- and one sentence that overstated a measurement. It authorized
storage option a. This round repairs those, moves the two finals, and re-plans storage from the live state. Still $0:
no grant, no build, no training, no generation, no paid resource, no storage bought, no disk grown.

### 0.1 What was fixed, where, and the evidence

| # | defect | fix (code) | evidence |
| --- | --- | --- | --- |
| 1 | **The checkpoint identity was checked after the record's identity was overwritten** (`run` set `rec["identity"] = ident`, then compared the checkpoint against it: the current identity against itself) | `build_driver.run`: each product carries ITS OWN identity (`stamp`: the digest and the parts it is a function of). It is written only when the product is made (`statistics_step` / `weights_step` return their records; `run` stamps them) or verified for reuse, never copied onto a product made under another identity. The statistics and the checkpoint are each judged against the stamp they carry (`_checkpoint_problems`). Same statistics, different checkpoint identity (the base, the method, its seed or the candidate code): the statistics are reused (fetched from the relay only if the pod's file is gone and the weights must be built). The old weights, their candidate, equivalence check, diagnostics and relay record are set aside under `superseded` with their own identity. Only the weights, their upload and the diagnostics are made again. A prose change (the design file's bytes) moves no identity | `depth_merge/tests :: test_a_checkpoint_identity_change_reuses_the_statistics_and_rebuilds_only_the_weights[candidate_code / base / method]`, through `build_driver.main --resume`, the GPU steps replaced at their boundary only: no statistics forward, one weights build, the relay moves `[put init]` (the `base` case, the pod's statistics file gone: `[get statistics, put init]`); the init record under the new identity; the old checkpoint kept with its old identity in `superseded` and relay.json. **Mutation:** with the checkpoint identity check removed, all three cases fail |
| 2 | **A handoff that answered `verified: false` was recorded as `RELAY_OK`**, and the next attempt took the copy over without completing it | `secure` records two states. The files verified are written to relay.json at once (`RELAY_VERIFIED`). Then the handoff (`handoff.verified`: the record beside the files, the listing naming it): one that does not verify is recorded and left pending (`RELAY_HANDOFF_PENDING`). `hand_off` retries only the pointer and the listing: before the init record, and in the next attempt, which takes the verified copy over without sending it again. `product_problems` requires each group handed off, so neither the init record nor the session's completion reads it durable while one is pending | `test_a_handoff_that_does_not_verify_keeps_the_files_and_the_next_attempt_completes_only_it`. Four `verified: false` answers: both groups put once, no init record, completion INCOMPLETE, and a record built from that state names "not handed off". The next attempt (`main --resume`, the GPU and every product step replaced by failures): no put, no get, no GPU, exactly two handoff calls, init record durable. **Mutation:** reading any `handoff` key as handed off fails it |
| 3 | **The sync window:** products uploaded, verified and published, the pod lost before the dev box received the latest `build.json` (the second round's test kept a complete `build.json` and lacked only relay.json) | Each group's relay record, and so the record put beside its files, carries the product's own record and identity (`product_view`). `recover` restores a product the build record lacks from relay.json or the relay's listing. It does so only under this identity, only when the description names exactly the record's files, and, for the weights, only when they were made from the build record's statistics. The diagnostics, which come after the weights, are made again from the restored weights; no build is repeated | `test_products_published_before_their_latest_build_record_came_back_are_restored_from_the_relay[no build record / the statistics only]`. A pod run publishes both groups and is lost in the diagnostics. The next attempt starts from the stale evidence (none at all, or the record from before the weights): it fetches only the weights (for the diagnostics), builds and uploads nothing, and writes the pod's own checkpoint record |
| 4 | **A build takeover asked the relay for the whole chain again** (51.38 GiB even when the build's 3.79 GiB were already there) | `session.build_need` = `storage.remaining(build_driver.held(...))`. `held` performs `run`'s own lookups without side effects, on the build.json and relay.json the attempt resumes from (read from the `--resume-from` archive: `session.resume_records`) and the relay's listing. It holds a product only under this identity with its files present (`session.relay_present`: every path, the LFS sha256), and never the weights without the statistics. Each held product is credited at the chain's own count (init 4,007,703,552 B; statistics 64 MiB), because the live headroom already excludes its bytes; the margin is unchanged. A first run holds nothing and asks for the whole chain; an unreadable listing refuses the launch (unknown, not empty) | `test_a_build_takeover_asks_the_relay_only_for_what_the_chain_still_adds` (the real `held`, `remaining` and sizes; a simulated relay): first run 51.38 GiB; statistics held 51.32; both held 47.58. The brief's case is checked: headroom exactly 51.38 at the first run, 47.59 after the build's products went in, accepted. A pending handoff with its files present counts; another identity, or files gone, does not; an unreadable listing raises |
| 5 | **The default search entry carried no PCA configuration** (v2 selected and declaring none; `session_terms` and the packed search's `SearchConfig` passing none; `width.global_pca_v0` refusing only after its statistics pass) | **Forward baseline v3**, selected: v2's method with `method.operator_configs` giving `width.global_pca_v0` the historical recipe `endpoints_9_1_8_v0`, inline with its source (the values the core used to hold). The core parses and validates it (`ForwardBaseline.operator_configs`: only the record's own operators, no executor-owned or KL key, absent for v1 / v2). `shared.forward_baseline.session_terms` returns it as `operator_configs`, which a search passes as `SearchConfig.operator_configs`. **Before compute:** `OperatorImplementation.required_config` (width and the composite: `stream_weighting`; not part of `signature()`, so no frozen hash moves) is refused when undeclared: by a search when its root is known and before its first forward, for an implementation that can act on the target; by a fixed path before its root is loaded; and by the operator itself before its statistics pass, where a malformed declaration is refused too. `packed_full_search` pins v2 (`BASELINE_VERSION`, its journal's) and passes its baseline's `operator_configs`; `packed_inputs.freeze_record` reads the same pinned version; `full_search` (pinned to v1) passes its baseline's too. `dry_run_search` and the device canary declare the registered recipe by name | `tests/initialization/test_required_operator_config.py` (7): an undeclared or malformed weighting refused before the statistics pass, for both operators; a search refused before any forward, passing when declared; a fixed path refused before the root loads, and a declared step applied with the declaration in its hashed config. `tests/initialization/test_forward_baseline.py :: TestARecordDeclaresConfigsForItsOwnOperators` (7). `scripts/shared/tests/test_forward_baseline.py :: TestVersion3DeclaresTheWeightingAndNothingElseMoves` (3): v3 equals v2 but for the declaration; the packed session stays v2's (no `operator_configs`, its config hash unchanged), and the same builder at version 3 carries v3's. Variant 1's contract still derives TPI1's journal hashes (the variant-1 suite's real contract) |
| 6 | **"The student's state about 9x the teacher's norm"** read from a relative squared error of 83.2 | removed from section 5 and from the step-0 diagnostic record's `reading`, which now carries a `corrections` entry; the layers and every measured value stay | prose only, no rerun |
| 7 | **SOB1's readers after the move** (its design keeps the historical "relay" placement) | `single_operator_budget.launch.retain` checks a relocated final where the retention record says it is now (every file re-hashed on the disk; the relay never asked); `rotate_relay` keeps the ROTATED status of a relocated arm | `single_operator_budget/tests :: test_a_final_relocated_to_the_dev_box_is_checked_there_and_never_looked_for_on_the_relay` |

Accepted by the review and not redesigned: the alignment cache's inputs, the span diagnostic's devices, the
hidden-target lifetime, V2 / V3's selection and C2 / C3 binding, the hardware profiles out of the core, the README,
the environment and test-economy text. Nothing scientific was re-measured.

### 0.2 The two finals moved to the dev box (storage option a, executed)

`scripts/maintenance/consolidation/relocate_relay_finals.py` (a dry run, then `--fetch`, then `--retire`), live:

| final | weights sha256 | bytes (4 files) | now | relay copy |
| --- | --- | ---: | --- | --- |
| `ffn_attention-s20260726` | `589acf286481f495a587ac76ad8956af66a623d3134e029fb55bed12a8c8c16b` | 8,068,316,426 | `~/aad-artifacts/stage1_single_operator_budget/ffn_attention-s20260726/final/589acf286481f495/` | retired 18:17Z, tombstone `relay_sob1_train_ffn_attention_20261124_033254_ffn_attention-s20260726_final`, 8,068,311,496 B freed, 0 paths left |
| `ffn_attention-s20260801` | `64292e508e2d6d2e67b503a5267b088ae13be45aad1bb3fadbe15c8dcd14530e` | 8,068,316,425 | `.../ffn_attention-s20260801/final/64292e508e2d6d2e/` | retired 18:18Z, tombstone `relay_sob1_train_ffn_attention_20261124_033254_ffn_attention-s20260801_final`, 8,068,311,496 B freed, 0 paths left |

The steps, in order:

* **Before any byte moved:**
  - each manifest is the session's sealed `final.json` and equals the retention record's;
  - the relay held all 4 paths of each copy, with the weights' LFS sha256 the manifest's;
  - the dev box had 22.30 GiB free for 15.03 GiB to fetch, against a reserve of 3.5 GiB: DMR1's largest session
    evidence (1.5 GiB) plus 2 GiB for the shared disk.
* **Fetched:** in 585 s and 610 s, one copy on the disk at a time (`relay.get` moves its download).
* **Verified on the disk:**
  - every file re-hashed to the manifest;
  - the safetensors header read back: 398 fp32 tensors, 2,017,066,496 parameters, which is SOB1's
    `structures.json :: ffn_attention.num_parameters` (4P = the file's 8,068,265,984 tensor bytes);
  - one tensor loaded and finite; the config parsed (qwen3, 36 layers, 2,560 / 3,840, 16 heads).
* **Registered before the deletion:**
  - `storage/local_retention_phase2.json`: `kept_on: dev_box`, with `relocated` (from where, the decision, the relay
    check before);
  - the checkpoint registry, rebuilt with `--hash --relay`: both rows protected, weights sha256 hashed on the disk.
* **Retired:**
  - re-hashed once more;
  - the tombstone written FIRST (`REDUNDANT_REMOTE_COPY`, `byte_identical_duplicate`);
  - `relay.remove` (permanent, the pointer with it) and the relay read back absent;
  - the rotation record's `ffn_attention` row: ROTATED; the registry rebuilt again.
* **Live afterwards:**
  - the relay holds 29.45 GiB (was 44.47) and accepts **58.13 GiB** (was 43.09);
  - the dev box has **7.00 GiB** free (was 22.30).
* **Untouched:** every other weight, the evidence and the frozen records.
  - SOB1's design keeps its historical placement text; its README and report note the move.
  - The finals are kept; only where they are kept changed, so none needs retraining to be recovered.

### 0.3 Variant 1's storage now (live, 2026-11-26 18:19Z; `plans/storage_plan.json`)

| state of a build session | asks the relay for (chain peak − held + 1 GiB margin) | the relay's headroom then | spare |
| --- | ---: | ---: | ---: |
| first run (nothing held) | **51.38 GiB** | 58.13 GiB | **6.75 GiB** |
| takeover, the statistics on the relay | 51.32 | 58.06 (their 64 MiB in) | 6.74 |
| takeover, the statistics and the init on the relay (handed off, or verified with the handoff pending) | 47.58 | 54.33 (their 3.79 GiB in) | 6.75 |

The chain is 0A.4's, step for step: the build's products and archive, then the training rotation. Its peak, 50.38 GiB,
comes at seed 20260801's save-point rotation, while seed 20260726's final is held. The training session's own gate asks
for its remaining rotation (`rotation_need` from `relay_state`), as before.

**The dev box: 7.00 GiB free**, and at most one session's evidence (≤ 1.5 GiB) at a time: 5.50 GiB margin. Variant 1's
finals stay on the relay. The filesystem is shared with another project whose use moves without this one (it took
7 GiB in 4 h on 2026-11-24); each launch re-reads the free bytes (`the_dev_box_has_room`), so a squeeze stops a launch,
never a running session. One consequence: a real pod-environment sweep (`record_pod_environment`, 24 GiB floor)
cannot run on the dev box now. Variant 1's launch chain runs none; the core suite's simulator tests need their own
floor set (section 8).

### 0.4 The default search configuration: compatibility

* **New work** reads v3 (`shared.forward_baseline.selected()`). A search built from `session_terms` hands the width
  operator its stream weighting through `SearchConfig.operator_configs`, into its hashed config: v3 width invocations
  have new config hashes, as a new version should. Values: 9 / 1 / 8, `endpoints_9_1_8_v0`. Not re-validated for the
  hybrid teacher (the record says so).
* **The packed full search pins v2**, and so does every session built on it: TPI1's, SOB1's, DMR1's statistics session.
  Its journal and their identities are v2's: no `operator_configs`, the same session config hash, TPI1's journal
  hashes derived exactly. A width or composite expansion under v2 is refused before its first forward.
* **v1 and v2 records, journals and results are unchanged**:
  - the v2 file is not edited and stays sha256-bound in `selection.json :: superseded`;
  - every historical entry still derives its recorded config hashes; one that would APPLY width or the composite stops
    until it declares a weighting, and no closed search is re-run.
* **Boundary for variant 1:** its build applies neither operator. Its contract derives the journal's hashes and its
  statistics session never runs a search. Its path and its identity are unchanged by this fix.

### 0.5 GPU validation not run, and whose it is

* **Variant 1:** none separately (P8.2: CPU cannot answer it, and the build session is the smallest GPU run that can).
  The build's three GPU steps (the statistics forward, the weights build and reload, the diagnostics on the device) were
  exercised here only at their boundary: the tests replace those three functions. Identity, stamping, takeover, handoff,
  recovery and completion run as implemented. The build session is their GPU run.
* **Variants 2 and 3:** the vLLM 0.27.1 engine validation of the recurrent model against the HF reference, the
  hidden-loss trainer's real memory, and the two terms' gradient norms at step 0 (about 1 h, about $1.2, cap $2.5).
  These are conditions of variants 2 / 3 only, not of variant 1, and are unauthorized.

### 0.6 Conditions per variant

* **Variant 1:**
  - this review and the maintainer's grant for each of its two kinds;
  - the relay room is no longer a condition: it fits by 6.75 GiB (0.3).
* **Variants 2 and 3** (unchanged from 0A.3):
  - comparison 1 complete and valid; the selection written once and its init record committed;
  - the GPU validation above;
  - a project-cap decision; a proposal and a grant per kind;
  - their own relay room: after variant 1 the relay holds 18.97 GiB more, and their rotation peaks at 46.46 GiB
    (0.3 leaves 58.13 − 18.97 = 39.16 GiB against that peak + margin, so they need their own room decision);
  - R1 / R2 answered.

### 0.7 Money, commit, resources

* **Money** (live, read-only, 2026-11-26):
  - balance **$123.73**; project cap $1,161.06, committed $1,086.27, **$74.79** left;
  - variant 1's proposals unchanged in amount: build $2.50 / **$5.43** cap, training $32.68 / **$64.46** cap, $1.00
    teardown reserve each: **$71.89**. Both read fundable (`launch.py budget`), neither is granted;
  - variants 2 / 3: estimates only (caps $110.43 / $114.60);
  - the storage move cost $0 and bought nothing.
* **Commit:** this round's work is the commit that carries this section on `research/reasoning-inheritance`, pushed to
  `origin`; the second round's was `d92f48f8`. Not merged into `main`.
* **Resources:** 0 pods, 0 volumes, $0.00/h. No session, grant or issued artifact exists for any DMR1 kind. The fetch
  and registry jobs ended, and no watcher is left.
* **Generated files:**
  - in git: the v3 record, the tombstones, the updated storage records, the storage plan, the relocation script, the
    regenerated registry and navigation;
  - outside git: the two finals, 15.03 GiB, in the dev box's out-of-tree store.

## 0A. Second repair round (2026-11-26, second brief) -- kept as written; section 0 holds where they differ

The first round's preparation was reviewed and this round repairs what the review found, audits the repository
against the project's principles and settles variant 1's storage arrangement. Still $0: no grant, no build, no
training, no generation, no paid resource. Section 0 is the round's report; sections 1-10 are the design, updated
where this round changed them.

### 0A.1 What was fixed, where, and the evidence

| # | problem | fix (code) | evidence |
| --- | --- | --- | --- |
| 1 | **V1 build takeover.** Weights uploaded and verified, init record not yet written, pod lost: the next attempt found no `work/products` and called `build()` again | `build_driver.run`: the build is a sequence of products (statistics, weights, diagnostics, init record), each recorded in `build.json` and copied to the relay **the moment it exists** (statistics right after they are saved, weights right after the reload check, then diagnostics), each relay group's record put beside its files (`relay.POINTER`) so a later attempt finds it even when relay.json was lost with the pod. A takeover makes only what is missing: init record durable → nothing runs; weights durable → only the record (no teacher, no GPU); diagnostics missing → the weights restored from the verified relay copy (`relay.get`, re-hashed) for them; statistics surviving → weights rebuilt from the statistics file (no statistics forward); persisted pairs reused under the same inputs | `depth_merge/tests :: test_weights_uploaded_and_verified_but_no_init_record_is_taken_over_without_building` (the real interruption through the real entry `build_driver.main --resume`: the teacher, the statistics step, the weights step and the diagnostics are all replaced by a failure, and none is called; the weights are never sent again; a second takeover only adopts the record); `test_a_verified_copy_the_dev_box_never_pulled_is_found_by_the_relay_listing`; `test_missing_diagnostics_restore_the_weights_from_their_verified_copy_and_rebuild_nothing`; `test_a_fresh_build_puts_each_group_once_and_a_refused_copy_writes_no_record` |
| 2 | **Takeover not bound to the experiment.** `product_problems` / `durable_init` checked a record's internal consistency only | `build_driver.identity`: the statistics' identity = teacher (root id, weights sha256, config), mapping (kept blocks, spans, pairs), calibration (profile hashes, positions operator and policy hash, packed forward context, session config hash, measurement protocol, execution packing and micro-batch, the journal's operator config hashes) and the statistics code (ast digests); the checkpoint's adds the base (init record, weights, config, per-tensor digests) and the merge (method, seed, candidate code). Never the design file's bytes, so a changed description recomputes nothing. `product_problems(doc, ident)` names the part that differs; a build record under another identity is set aside (`build.superseded_<utc>.json`), never reused. The dev-box closeout (`session.completion`) and the training plan's derivation (`plan.init_problems`) check under THIS tree's identity (`current_identity`, $0, ~3 s on the CPU) | `test_products_made_under_another_identity_are_set_aside_and_never_adopted` (another calibration profile: the record is set aside, the teacher is opened to rebuild, `product_problems` names `calibration`); the closeout reads INCOMPLETE under another identity (in the fresh-build test) |
| 3 | **Alignment cache keyed on the FFN correlation alone**, which also decided the KV-group and query-head permutations | `merge.pair_inputs`: a pair is reused only when ALL its inputs are this pass's -- the FFN Pearson correlation, the attention matching's inputs (the pair's head-write Gram: cross products and both sides' energies; both blocks' own head Grams, which give the group energies), the head geometry and the alignment code's ast digest; the reuse report names what differed | `test_an_unchanged_ffn_correlation_does_not_carry_a_stale_attention_matching` (the counter-example: the absorbed block's heads in reverse order, its FFN moments untouched -- the FFN hash still matches, the KV / head permutations change, the pair is recomputed and equals a fresh computation; the other pairs are reused); `test_alignments_are_reused_only_under_the_same_inputs` |
| 4 | **Span diagnostic devices.** The base's kept blocks stayed on the host while the diagnostic's inputs came from the GPU teacher | `build_driver.span_candidates`: only the 9 kept blocks the diagnostic compares are moved to the teacher's device (about 1.6 GB; the rest of the base stays on the host), dropped with the base after the diagnostic; a failure is recorded as before, never a gate | `test_the_span_diagnostic_moves_only_the_kept_blocks_to_the_teachers_device` (to `meta`: the 9 move, every other block and the embedding stay). The real GPU path runs in the formal build session itself; no separate run |
| 5 | **V2 / V3 selection source.** `decide` accepted a truthy `complete`, an empty or foreign per-seed dict, non-finite deltas; C2 read the CURRENT selection file | `schedule.comparison_1_problems` + `decide`: REFUSED unless the record is comparison 1's (schema, id, pair, `analysis.protocol_identity`: set manifest, decoding, action protocol, modes, the 570 items' sha256, the 558 groups' file, the two registered seeds; both sessions complete), `complete is True`, a finite pooled delta and one finite delta per registered seed (no bool). `schedule.selection_problems`: a selection stands only while the comparison-1 file it names re-hashes and re-derives the same choice and deltas; `select --write` never rewrites a selection. Comparison records now carry `protocol`. `analysis.bound_init`: C2's control and C3's pair come from the init record each trained unit's committed plan bound (plan → recurrent init record → selection), C3 refusing two different inits | `depth_recurrence/tests :: test_an_invalid_or_incomplete_comparison_1_selects_nothing` (13 cases: `complete="yes"`, `{}`, a foreign seed, a missing seed, NaN, inf, a bool delta, another pair, another id, another protocol, other seeds, 569 items, an incomplete session); `test_the_selection_stands_only_while_it_rederives_from_the_comparison_1_record_it_read`; `test_c2_and_c3_read_the_initialization_the_trained_units_bound_not_a_later_selection` |
| 6 | **Hidden-state targets outlived their micro-batch.** After `self._micro_hidden = None` the local `sup` still held the previous micro-batch's teacher targets through the next forward, and the last one through the optimizer step | `HiddenSupervision.finish` drops the object's own references to the targets, inverse energies and rows (the autograd graph keeps what the backward needs and frees it in the backward); `_step_body` and `_eval_blocks` drop `sup` (and the step's loss tensors) at the end of each micro-batch | `tests/training/test_hidden_state.py :: test_a_micro_batchs_targets_are_gone_before_the_next_forward_and_the_optimizer_step`, `test_validation_releases_each_micro_batchs_targets_before_the_next` (weak references to every target, checked at every micro-batch's forward and at the optimizer step; **both fail on the previous code**, checked by stashing the fix). Only a run with `loss.hidden_state` (variant 3) builds a supervision: SOB1 and variant 1 were never affected |
| 7 | **Wording that overstated** the hidden loss and the statistics | stage-6 `design.json :: hidden_state.error` (a per-position RELATIVE squared error, not MSE), `hidden_state.lambda_rationale` (1.0 a proposal; 0.25 not a validated "KD-dominant" setting; a similar scalar size is not a similar gradient share; the global clip neither guarantees KD dominates nor bounds AdamW's per-coordinate update), `step0_diagnostic.scope` (evidence about this initialization, not about recovery), `analysis.family` (each comparison's own 0.05; no round-wide rate), `analysis.selection_conditioning` (C2 measured at the init C1 chose on the same items); the stage-6 README sections 3-4; sections 5-6 below; every comparison record carries `error_rate_scope`, and C2's `selection_conditioning` | no rerun of any diagnostic (the brief) |
| 8 | **Core hardcode: the PCA end-point weights 9 / 1 / 8** (one 2026-07-14 Qwen3-4B ablation) inside `init_student` and `width.global_pca_v0` | `transforms.project.stream_weighting` validates a CALLER's declaration (explicit points and weights, or first / interior / last over every point; finite, > 0; a source) and the core computes with it; `init_student(stream_weighting=)` is required (no default); `composite.stage1_sandwich_v0` (both `init_student` calls, the `verify_full_assignment` one included) and `width.global_pca_v0` read the run's declared `stream_weighting` from their hashed config and refuse without it; `SearchConfig.operator_configs` carries a run's per-operator declarations (omitted when unset, so every committed search keeps its hash). The historical recipe is data, `configs/shared/initialization/stream_weighting.json :: endpoints_9_1_8_v0`, with its source (`git show 633dc6b4c:logs/experiments/stage1/2026-07-14_stage1_qwen3_0p6b_init_v0.md`); `scripts/shared/stream_weighting.py` reads it, and `init_stage1.py` names it explicitly for the configs written before the key existed | `tests/initialization/test_stage1.py :: test_the_projection_weighting_is_the_callers_declaration_never_a_core_default` (no default; four malformed declarations refused; the historical recipe gives the recorded weights); the e8 frozen-record suite (the composite equals `init_student` bit for bit under the declared recipe); every core test that runs either operator now declares it; Phase A's search runner (`phase_a_search.run_phase_a_search`, the historical AutoInit search) names the historical recipe explicitly and its suite's searches pass. **Identity**: a run that declares the weighting hashes it into the operator config, so it gets new state ids (registered as any declared config); an undeclared historical replay still derives the config hash its record carries (DMR1's contract does), but applying width or the composite now needs the declaration -- the historical tree applies it as it was. No recorded result or hash was edited |
| 9 | **Core hardcode: hardware instances in `runtime/cost.py`** (L40S / A100 / dev-box profiles, prices, throughputs, `CPU_STATS_SECONDS_PER_TOKEN = 4972 / 949,859`) | moved to `configs/shared/infrastructure/hardware_profiles.json` with their sources (historical values unchanged), built by `scripts/shared/hardware_profiles.py` (the historical planners import the same names from there); `HardwareProfile.stats_seconds_per_token_upper` / `stats_rate_source`: a statistics pass is priced only with the rate its own profile declares (`CostModelError` otherwise), never with one measured for another teacher, device or implementation. The core keeps the type and the formulas | `tests/runtime/test_forward_pass_pricing.py :: test_a_profile_without_a_statistics_rate_refuses_to_price_a_statistics_pass`, `test_the_core_cost_model_holds_no_hardware_instance`; the historical planners' suites reproduce their prices (`scripts/shared/tests/test_cost_and_recovery.py`, `phase_b/tests/test_cost_model_replay.py`, `phase_c3/tests/test_pilot_pricing.py`) |
| 10 | **V1's storage proved only for the build** (the report also omitted the new init's 3.73 GiB) | `depth_merge/storage.py`: the relay's net change over the whole chain in the drivers' order (build products, build closeout archive, the training rotation from an empty state, closeout archive) against the live headroom; `session.relay_room` for the build kind asks the relay for the WHOLE chain plus a 1 GiB margin, so no init is built for a training session that would not fit; `logs/stages/stage-1/depth_merge/plans/storage_plan.json` | `storage.py --write` against the live relay (0A.4) |
| 11 | **Code bundles kept for no consumer** | `scripts/maintenance/consolidation/retire_relay_bundles.py`: a session bundle is retired only when its commit is exactly one commit on a pushed ref (rebuildable), no pod exists, and it is not under a protected prefix; tombstone first, exact-object permanent deletion, listing read back | 379 bundles retired, 8,418,278,418 B (7.84 GiB) freed, 0 left (`logs/maintenance/inventories/relay_bundle_tombstones_20261126.json`, written before the deletion); 20 bundles (0.07 GiB) whose commit is not on a pushed ref are KEPT; the headroom read back 43.09 GiB (it was 35.27) |
| 12 | README / AGENTS | README's goal now says the other capabilities are still measured separately and names the two environments (data generation and training); its student sizes include the ~2.0B rounds. AGENTS P8.3: the full-suite counts are upper bounds, never a quota | -- |

### 0A.2 Principle audit

Checked = run (a test or a live read), static (read in the code or records), or not implemented.

| principle | where it lives | check | result, gap or fix |
| --- | --- | --- | --- |
| 1 GPU validation where CPU cannot answer | P8.2 / P8.3; this report 0.3 | static | no CPU stand-in was built for a CUDA question this round. V1's real GPU paths (statistics, 20 assignments, the span diagnostic on the device, the reload) run in the formal build session itself. V2 / V3's engine and memory validation (~1 h, ~$1.2, cap $2.5: the recurrent model in vLLM 0.27.1 vs the HF reference, prefill / decode, the per-logical-layer KV cache, the first training steps' memory with and without the hidden term, the two terms' gradient norms at step 0) is a condition of V2 / V3 only, unauthorized |
| 2, 4 scalability; no instance hardcode in the core | `src/aadistill`; adapters; `configs/` | run + static | the two confirmed hardcodes fixed (0A.1 #8, #9). A grep of `src/aadistill` for teacher names, dimensions, seeds, experiment paths, device names and provider ids finds only docstrings / comments, the RunPod provider adapter (an infrastructure adapter, like a model-family adapter) and `cpu_test_env.host_local_store`'s default store name (`aad-artifacts`, a local-store name; noted, not changed). Stage 0-6, other families, geometries and compression ratios enter through configs and adapters (the recurrent family is an adapter; schedule, pairs and weights are config). **Gap left open, deliberately:** the other application search builders (`packed_full_search`, `full_search`, the D1 / D2 sessions, the calibration-volume session, `dry_run_search`, the device canary) and the forward baseline v2 record declare no stream weighting; their searches still derive the hashes their records carry, but one that APPLIES width or the composite now stops loudly until it declares one. The next forward run declares it in its own protocol (or a baseline v3); editing the registered v2 record or the closed builders here would move recorded identities for no consumer. **Closed in the third round** (0.1 #5, 0.4): a new v3 record declares it, the default entry passes it, the packed search pins v2, and an undeclared weighting is refused before compute |
| 3 logs by stage / experiment / run | `logs/stages/stage-1/depth_merge/`, `logs/stages/stage-6/depth_recurrence/`, generated READMEs and `logs/stages/index.json` | run | regenerated after this round's new records (the storage plan, the bundle tombstones); the historical layout is untouched |
| 5 test economy | P8.3 | static | mid-round checks only for specific questions (the lifetime regression against the old code; the counter-example; the operators' callers). One convergence validation and one full core suite after the implementation (0A.6). The P8.3 bound is now stated as an upper bound, not a duty |
| 6, 9 autonomous completion | P8.2.1, P12.1 | run | every repair above was made and verified in this round; the one thing outside the authority is the relay room (0A.4), listed as a decision with its objects and sizes |
| 7 historical weight retirement | AGENTS 2.5; `retire_eligible_checkpoints.py` | run | the standing-rule dry run finds **0** eligible rows (two refused: their bytes are already gone, or code still names them). Every remaining relay checkpoint is protected by an explicit maintainer clause (SOB1's finals at their approved placement; SOB1's three inits "kept until the next round's plan is fixed"; the permanent controls; the canonical 0.6B init) or is DMR1's own base. The 379 session code bundles (7.84 GiB) had no consumer and were retired (0A.1 #11) |
| 8 review priorities | P8.3 | static | this report leads with validity (the selection's source, the conditioning of C2, the identity binding, the lifetime bug), then reuse and identity, then storage and money |
| 10, 11 keep new products; reuse across attempts | P8.2.1, P8.4 | run | the build persists each product at once and takes over by identity (0A.1 #1-#3); training reuses SOB1's chain (finals restored, never retrained; sealed generation parts reused; save points resume optimizer, schedule, RNG and data position: the end-to-end driver tests of the first round) |
| 12 capability target | P10.2, P10.3; `data/eval-set/` | static | fluid reasoning is the primary target; the primary comparison reads eval-set-1's static fluid formal core; every other unit is reported on its own. eval-set-1 is development-exposed (its own manifest); `stage3_legacy_150` is in-sample training prompts and is never reported as generalization; nothing here is a never-seen result |
| 13 terminology | AGENTS 2.10 | static | packed sequence, sequence length, batch size (2 sequences per optimizer step), micro-batch size (1), gradient accumulation (2 steps), epoch; a **logical layer** is one execution of a Transformer block at one teacher-layer position (defined once, stage-6 README section 2); a **recurrent iteration** is one more call of the same block; a Transformer block keeps its meaning |
| 14 one teacher lineage | P17.1; `design.json :: teacher, lineage`; the plans' `lineage` | static | hybrid `Qwen/Qwen3-4B@1cfa9a72` for calibration, statistics, initialization, SOB1's training targets, the online KD teacher, tokenizer, chat template and config; the build checks the loaded teacher's sha256; a save point is restored only under the same trainer config sha256 (which binds teacher and data) |
| 15 packing and token counts | P19; the designs' `statistics`, `training.data` | static | V1's statistics forward: the packed search's context (seeded sha256 shuffle, sequential packing, truncation, continuous positions, causal, 8,192). Training: SOB1's pack (the same rule; 6,467 packed sequences x 8,192). Volumes counted before packing (558 calibration examples; 45,363 training examples, 56,408 turns, 11.86 M unique supervised tokens); 79 packed sequences / 231,242 positions and the per-step KD rows are packing and exposure measures only. V2 / V3 (Stage 6) register Stage 3's packing as their own identity. **Stage 5: no run exists -- not executed, not validated** |
| environments | `requirements-cu128.txt` → `/opt/train` (relay wheelhouse `wheelhouse_cu128_cp312`), `requirements-vllm.txt` → `/opt/vllm` (`wheelhouse_vllm_cp312`); `autoinit_preflight_setup.sh` | static | data generation and evaluation generation run in the vLLM environment, training in the training environment (`train_driver --vllm-python /opt/vllm/bin/python` hands evaluation to the other one); the online KD teacher's forward is part of the training computation, in the trainer's process. Both wheelhouses are the pods' environment and are kept |
| "never seen" | `data/eval-set/eval-set-1/manifest.json`; `shared.eval_sets` | static | unseen ITEMS: training, calibration and search data are checked disjoint from held-out units (`require_disjoint`, `held_out_paths`); unseen RULE FAMILIES: fluid_reasoning_720's generator families are evaluation-split and calibration builds are checked disjoint by family, content and prompt; unseen INTERACTIVE ENVIRONMENTS: the `interactive_adaptation` episodes (formal core; descriptive here); DEVELOPMENT EXPOSURE: eval-set-1's items and per-item results were read during development, so a claim of untouched-test generalization needs eval-set-2 (not built; out of scope) |

### 0A.3 Which conditions belong to which variant

**Variant 1 (before its build session is launched):** this review; the maintainer's grant for each of its two kinds;
**the relay room for the whole chain** (0A.4) -- the build's launch gate refuses until it exists. No separate GPU
validation: the build session's own run is the GPU check of its paths. **Not** conditions of variant 1: the vLLM
recurrent engine, the hidden-loss memory, the lambda question, anything of comparison 1's outcome.

**Variants 2 and 3:** comparison 1 complete and valid (the selection refuses otherwise); the selection written once and
its recurrent init record committed; the GPU engine validation above; a project-cap decision (their caps exceed what
the cap leaves after variant 1); a proposal and a grant per kind; their own relay room (their rotation is the same
46.46 GiB peak; after variant 1 its init, two finals and archives hold 18.97 GiB more); R1 / R2 answered (section 5).

### 0A.4 Storage, on the real objects and in the real order

Live (read-only, 2026-11-26 15:15Z): the relay holds **44.47 GiB** (the sum of its LFS objects, the billing basis) and
accepts **43.09 GiB** more (LFS batch-endpoint bisection, nothing stored). The sizes are SOB1's measured depth arm's:
init 4,007,703,552 B (bf16), save point 20,934,572,032 B (fp32 weights + AdamW + state), final 8,015,407,104 B (fp32);
statistics bounded at 64 MiB (about 4 MB), each closeout archive at 128 MiB (SOB1's were about 40 MB).
`logs/stages/stage-1/depth_merge/plans/storage_plan.json` holds every step.

| step (drivers' order; the previous save point is removed only after the next one verified and handed off) | relay, net vs now | dev box |
| --- | ---: | ---: |
| build session: the init and the statistics put (each verified, its pointer beside it) | +3.79 GiB | evidence ≤ 0.5 GiB |
| build closeout: its evidence archive; the init stays at its relay prefix | +3.92 | |
| training, seed 20260726: each save point uploaded while the previous one is still there (2 at once) | **+42.91** | evidence ≤ 1.5 GiB (one session at a time) |
| seed 20260726's final uploaded, then its last save point removed | +30.88 → +11.38 | |
| training, seed 20260801: seed 1's final held while two save points rotate | **+50.38 (the peak)** | |
| seed 20260801's final, then its last save point removed | +38.35 → +18.85 | |
| training closeout: its archive; both finals and the init stay on the relay (`storage.phase_2_finals.placement`) | +18.97 | |

**Need: 50.38 + 1.00 margin = 51.38 GiB against 43.09: short by 8.28 GiB.** The build session's launch gate asks for
exactly this (0A.1 #10), so it refuses now; the training session's gate asks for its remaining rotation. The dev box
(23 GiB free, shared with another project) is never the constraint for variant 1 itself: at most one session's
evidence (≤ 1.5 GiB) at a time; the finals stay on the relay.

**What this round could free, and did:** the standing-rule dry run finds no eligible checkpoint; the 379 session code
bundles (7.84 GiB) had no consumer and are rebuildable from pushed commits -- retired. **What it could not**, because a
maintainer clause or a consumer holds each one:

| object | size | held by |
| --- | ---: | --- |
| SOB1 `ffn_attention` finals, seeds 20260726 / 20260801 (relay only) | 2 x 7.51 GiB | "the endpoints the report measured, kept at their approved placement" (SOB1; placement = relay) |
| SOB1 `residual_width` and `ffn_attention` inits (relay copies; dev-box copies verified) | 3.75 + 3.76 GiB | "kept (here and on the relay) until the next round's plan is fixed, then decided by its consumers" (maintainer, 2026-11-26) |
| SOB1 `depth_only` init | 3.73 GiB | DMR1's own base (variant 2's KEPT path fetches it) |
| permanent controls (2), canonical 0.6B init and its random baseline | 4.44 + 2.22 GiB | protected prefixes |
| stage-0 calibration-volume finalist | 1.11 GiB | named by Stage-2 configs |
| the two wheelhouses | 7.42 GiB | the pods' training and vLLM environments (every session's setup) |

**The decision the maintainer owed before variant 1 launches** (any one that makes 8.28 GiB) -- **taken: option 1, executed in the third round (0.2, 0.3)**:

1. *(recommended)* move SOB1's two `ffn_attention` finals to the dev box -- downloaded, re-hashed, recorded as the width
   and depth finals were, then their relay objects permanently deleted with tombstones (a change of their approved
   placement): relay +15.03 GiB, **6.75 GiB to spare** at variant 1's peak; the dev box drops to about 8 GiB free
   (≥ 6.4 GiB during a variant-1 session);
2. buy at least 9 GiB of private storage;
3. *(not enough alone)* once DMR1's plan is approved and so "fixed", retire the relay copies of SOB1's width and
   `ffn_attention` inits (their dev-box copies stay): +7.50 GiB leaves 0.22 GiB of the 1 GiB margin -- too thin by
   itself; with option 2 or 1 it is ample.

Retiring the `ffn_attention` finals outright (their only copy; a $38 retraining to recover) is possible but not
recommended. After variant 1 the relay holds 18.97 GiB more (its init and two finals), so variants 2 / 3 need their own
room decision later (their rotation peaks at the same 46.46 GiB); that is theirs, not variant 1's.

### 0A.5 Budget and authorization (live, read-only, 2026-11-26 15:07Z)

* Account balance **$123.73**; 0 pods, 0 volumes, $0.00/h. Project cap $1,161.06, committed $1,086.27
  (provider-settled), **$74.79** left under it. SOB1's closed grants' remainders are void and fund nothing.
* **Variant 1's proposals, unchanged in amount** (regenerated only because the design file changed): build $2.50
  expected / **$5.43** cap; training $32.68 expected / **$64.46** cap; $1.00 teardown reserve each: **$71.89**, inside
  the $74.79 and the balance. Both kinds read fundable (`launch.py budget`); both still refuse to launch on the relay
  room. Nothing is granted or issued.
* Variants 2 and 3: estimates only (~$59 / $62 expected, caps $110.43 / $114.60), not proposed; no paid authorization.
  The proposed engine validation (~$1.2, cap $2.5) is not authorized.
* The balance, the remaining project cap and an experiment's authorization are three different things: the first two
  are room, only a grant authorizes.

### 0A.6 Commit, push, resources

* This round's work is the commit that carries this report on `research/reasoning-inheritance` (`git log -1 --
  docs/stages/stage-1/reasoning_inheritance/DMR1_REVIEW.md`), pushed to `origin`; the first round's was `49ce2014`.
  Not merged into `main`.
* Running resources: **none** -- 0 pods, 0 volumes, $0.00/h (live read 2026-11-26 15:07Z); no session, grant or issued
  artifact exists for any DMR1 kind; every background watcher of this round is closed.
* Generated files: the new records are small JSON in the tree (`storage_plan.json`, the bundle tombstones, the two
  recipe / profile configs); no weight, cache or archive was added to git; the dev box's scratch holds only the
  rendered report page.

## 1. Scope, files and real entry points

The round **DMR1** (reasoning-inheritance research) has three variants:

| variant | arm | stage | design | question |
| --- | --- | --- | --- | --- |
| 1, DMR1-M | `depth_merge` | Stage 1 init, `stage3_recovery` training | [`configs/stages/stage-1/depth_merge/design.json`](../../../../configs/stages/stage-1/depth_merge/design.json) | alpha-merged init vs direct deletion, same recovery |
| 2, DMR1-R | `depth_loop` | Stage 6 (`stage6_recurrent_depth`) | [`configs/stages/stage-6/depth_recurrence/design.json`](../../../../configs/stages/stage-1/depth_recurrence/design.json) | 16 shared blocks on 36 logical layers vs single execution, same init |
| 3, DMR1-RH | `depth_loop_hidden` | Stage 6 | the same | variant 2 + per-iteration hidden-state loss |

Design documents: [`docs/stages/stage-1/depth_merge/README.md`](../depth_merge/README.md),
[`docs/stages/stage-6/depth_recurrence/README.md`](../depth_recurrence/README.md).

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
83.2 at layers 14-17, decaying through the later blocks (74.5 at 18,
16.0 at 24, 2.8 at 34, 7.4 at 35). On the same weights, the recurrent logits equal the unrolled tied-weight Qwen3's
exactly.

* **What this says for variant 2**: executing the kept blocks on the schedule without training starts far from the
  teacher -- CE 12.46 against the single-execution base's 3.46 on the same rows, above the uniform distribution's
  ln(151,936) = 11.93. Recovery has to undo that before it gains anything from the added depth; SOB1's width arm
  recovered from 9.78 to 0.50, so a high start is not by itself fatal, but this is a materially worse starting point
  than variant 2's control, and the comparison measures it together with recurrence. The registered design is
  unchanged; this is flagged for the review (question R1 below).
* **lambda_hidden = 1.0 is a proposal**, not tuned and not validated; 0.25 is an equally unvalidated alternative, not a
  "KD-dominant" setting. What step 0 measured is only the scalar size: the hidden term 14.56 against KD 12.30, its mass
  in logical layers 14-28. That does not make their gradient contributions similar (not measured: the hidden term
  reaches every block through every logical layer's output, KD only through the head); the global gradient-norm clip at
  1.0 rescales the summed gradient as a whole, so it neither guarantees that KD dominates nor changes the two terms'
  shares; and it bounds the gradient's norm, not the parameter update -- AdamW normalizes each coordinate by its running
  second moment, so the update size follows the learning rate and those moments. KD fell from 3.54 to 0.41 over SOB1's
  depth recovery; the per-layer log shows whether the hidden term falls with it. No lambda search is part of this round;
  the two terms' gradient norms at step 0 are a cheap addition to variants 2 / 3's GPU validation.
* **The scope of the step-0 result**: two validation sequences show that THIS initialization starts degenerate on the
  36-layer schedule. It is not evidence that recovery fails, and it is no reason to change the number of iterations, add
  a residual scale or change the initialization rule; the registered design runs as registered.

**Questions for the review.** R1: should variant 2's registered initialization stay the plain schedule of the
selected weights (the brief's definition), given its step-0 distance? R2: lambda_hidden 1.0 or 0.25 (neither
validated)? (R3, the relay room, is settled: option a executed, section 0.2; variant 1's chain fits, 0.3.)

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
* **The error rate's scope**: each comparison's 0.05 is its own family's; no error rate is controlled or claimed over
  the round's three comparisons together (every record says so, `error_rate_scope`).
* **C2 is conditional on the selection**: both of its arms start from the initialization C1 chose on the SAME
  development-exposed items C2 is scored on, and the MERGED path's control is the arm that looked better on them. C2
  estimates the recurrence's effect at the selected initialization, not at one chosen independently of this set (the
  record carries `selection_conditioning`); C2 and C3 read the initialization their trained units bound
  (`analysis.bound_init`), never a later selection file.
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

**The third round's evidence** (one convergence validation after the repairs, then one full core suite; mid-round
checks only for specific questions):

```text
development checks (specific questions)  : two mutations -- the checkpoint identity check removed, and any handoff
                                           key read as handed off -- each fail their new tests (3 + 1), restored from
                                           a saved copy; the live build-session room gate (read-only): 51.38 GiB
                                           ACCEPTED, nothing held
convergence, core (changed code)          : tests/initialization + tests/validation 1,472 passed / 0 failed (2.4 min);
                                           tests/architecture 83 passed once this round's core changes were declared
convergence, experiment suites            : depth_merge 24, depth_recurrence, shared (incl. forward baseline v3),
                                           SOB1 (incl. the relocation regression), TPI1, teacher_modes:
                                           961 passed / 7 failed / 7 skipped (9.2 min). The 7 are PRE-EXISTING
                                           closed-experiment state checks in teacher_modes: 3 fail at d92f48f8 too
                                           (a worktree run); 4 tool_pilot tests look the frozen protocol up with
                                           `git log -- <file>`, which on this squash-cut branch finds one commit
                                           (a worktree without the gitignored pilot artifacts skips them)
full core suite (pytest, testpaths=tests)  : 4,183 passed / 25 failed / 50 skipped, 15.7 min. The 25 are
                                           tests/integration/test_simulator_restore.py: the pod simulator refuses
                                           below its default 20 GiB free, and the dev box has 7 GiB since the
                                           finals moved. Rerun of that file alone with the suite's own floor
                                           (AAD_TEST_PODSIM_MIN_FREE_GIB=2; the tests move a few-kB tree):
                                           36 passed / 0 failed. No code was changed for it
dev-box limit this leaves                 : a REAL pod-environment sweep (record_pod_environment.min_free_gib:
                                           24 GiB) cannot run on the dev box now. Variant 1's launch chain does not
                                           run one (SOB1's three sessions launched without one); a round that needs
                                           one needs room first
GPU validation                             : none (no GPU; nothing paid); what is owed and by whom: section 0.5
```

**The second round's evidence** (the repair round; one convergence validation after the repairs, then one full core suite;
mid-round checks only for specific questions -- the lifetime regressions run against the previous code, the
alignment counter-example, which callers the operators' declaration reaches):

```text
development checks (specific questions)  : the two lifetime regressions FAIL on the previous code (the fix stashed)
                                           and pass with it; the alignment counter-example; tests/initialization run
                                           to find every caller the operators' declaration had to reach
convergence, experiment suites            : depth_merge 17, depth_recurrence (27 + the 13 refusal cases), SOB1, RI,
  (incl. the historical planners' pricing   continuation_b, phase_b cost replay, shared cost and recovery, e8 frozen
   suites this round's imports touched)     records, phase_c3 pricing: 298 passed / 1 failed / 42 skipped, 6.7 min;
                                           the failure is PRE-EXISTING (continuation_b's frozen set names
                                           scripts/experiments/__init__.py, absent at HEAD too)
Phase A (historical; its search runner     : 156 passed / 8 failed, 15.2 min; the 8 are test_phase_a_attempt12_import,
  now declares the historical recipe)       PRE-EXISTING (its checkpoint was retired 2026-11-14)
Phase B / A3 frozen-digest suites          : 40 passed / 91 skipped
full core suite (pytest, testpaths=tests)  : 4,181 passed / 12 failed / 50 skipped, 14.3 min; the 12 were
                                           tests/validation/test_cuda_suffix_case.py (the shared CUDA engineering
                                           check applies width without a declaration): its case config now names the
                                           recipe; tests/validation rerun 83 passed / 0 failed, and the minimal
                                           neighbourhood (test_calibration_item_preparation 29, depth_merge 17) passed
GPU validation                             : none (no GPU; nothing paid)
historical suites not run                  : the other closed experiments' (D1, D2, C2, C3, calibration volume,
                                           teacher_modes); their search builders declare no weighting (0A.2, 2/4;
                                           the default entry declares it since the third round: 0.1 #5, 0.4)
```

**The first round's evidence** (kept for the record; one convergence validation after the implementation, then one
full core suite; the two small early checks were API questions about transformers 5.13's Qwen3 internals):

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

**Fixed during the first round** (the second round's repairs: section 0A.1; the third's: 0.1):

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
| a training session's relay peak | **46.46 GiB** (two save points + one final); variant 1's chain adds its init and archives: **50.38 GiB** (0A.4, 0.3) | -- |

**The relay.** Superseded by sections 0A.4 and 0.3. The chain counts the new init (3.73 GiB), follows the drivers'
real order, and asks the relay for the whole chain: 50.38 GiB at variant 1's peak, plus a 1 GiB margin. After the third
round moved SOB1's two FFN + attention finals to the dev box (0.2), the live headroom is 58.13 GiB: **6.75 GiB to
spare**. A takeover asks only for what the chain still adds (0.1 #4). The one-save-point rotation the first version
offered as an option is withdrawn: it would delete the only valid save point before the next one is uploaded.

**Money** (live 2026-11-26 15:07Z, read-only; unchanged since 11:48Z):

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

* Nothing billed in any round: no pod, no volume, no grant, no artifact, no build, no training, no generation.
  Read-only queries: the provider position, the relay's headroom and inventory, the vLLM 0.27.1 wheel (its source) and
  the public teacher weights (the step-0 diagnostic; deleted afterwards).
* **One relay deletion, this round, under the brief's rule for code packages**: 379 session code bundles (7.84 GiB),
  each rebuildable from a commit on a pushed ref, none read by anything (0 pods); tombstone
  `logs/maintenance/inventories/relay_bundle_tombstones_20261126.json` written before the deletion; 20 bundles whose
  commit is not on a pushed ref kept. No checkpoint, log or evidence was touched: the standing-rule dry run found none
  eligible.
* **Two relay deletions in the third round, under the maintainer's explicit authorization (storage option a):** the
  relay copies of SOB1's two FFN + attention finals (2 x 8,068,311,496 B). Each was deleted only after its dev-box copy
  had been fetched, re-hashed file by file, read back and registered, and after its tombstone was written; both were
  confirmed absent (0.2). No other weight, log or evidence was touched.
* Retention: SOB1's depth init and its two finals are DMR1's declared consumers (`build_checkpoint_registry.py`). The
  three SOB1 initializations and every SOB1 final stay where their maintainer clauses keep them: all six finals are now
  on the dev box.
* Owed to the maintainer: this review and variant 1's two grants (the relay room is settled: 0.2, 0.3). For variants 2 and 3: a
  project-cap decision, their proposals and grants after the selection, the GPU engine validation, their own relay room.
* After the review, the room decision and the maintainer's authorization, variant 1 starts with `launch.py grant
  --kind dmr1_merge_build ...` (its design document, section 5); the build's launch gate re-reads the relay for the
  whole chain before anything is created.
