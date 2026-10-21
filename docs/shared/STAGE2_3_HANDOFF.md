# Stage-2/3 handoff: facts for the next research round

**Status: factual handoff, written at the Stage-0 final closeout, 2026-10-21
UTC.** It authorizes nothing: no run, no spend, no dataset generation, no
change to a frozen record. Paths are repository-relative unless they start
with `/`; `runs/`, `design/` and `plans/` sit under
`logs/stages/stage-0/calibration_volume/`. The Stage-0 verdict this starts
from: **EXECUTION COMPLETE / DOWNSTREAM REASONING FLOOR / VOLUME EFFECT ON
REASONING NOT RESOLVED** (`closeout/programme_closeout.json ::
programme_verdict`).

The maintainer has said the next round prioritizes **Stage-2 teacher-native
reasoning training data, then Stage-3 recovery**, and defers the Stage-0 mixture
ratio. §10 lists the directions named so far. Nothing here is a protocol.

Tags (AGENTS.md P14): **[IMPL]** implemented and validated by a named record;
**[MEAS]** measured; **[PLAN]** plan, directive or hypothesis; **[NEVER-RUN]**
designed but never executed; **[DERIVED]** computed read-only for this draft
from sha256-verified stored records, not a registered result.

## 0. Read first

| what | where |
| --- | --- |
| contract: P10.2 target, P17 teacher fidelity, P18 measurement, §4.4/§4.5 | `AGENTS.md` |
| Stage-0 programme result (read round 3 first) | `logs/stages/stage-0/calibration_volume/closeout/REPORT.md` |
| derived figures and money | `logs/stages/stage-0/calibration_volume/closeout/programme_closeout.json` |
| forward design for Stage 0/2/4 (authorizes nothing) | `docs/shared/REASONING_TRANSFER_DESIGN.md` §4 |
| capability scope, re-declared | `logs/budget/decisions.md` 2026-10-16 ("the primary target is fluid reasoning") |
| evaluation contract | `data/eval-set/README.md`, `data/eval-set/eval-set-1/README.md` |
| consolidated Stage-3 history | `logs/stages/stage-3/history/EXPERIMENTS.md` |

## 1. Initialization candidates available for reuse

### 1.1 The method behind them [IMPL]

**Forward baseline** `stage1_init.d1_target_aware.scored_v1@v1`
(`configs/shared/baselines/initialization_forward_baseline_v1.json`, selected
by sha256 `333af13d…` in `selection.json`).
- Decided 2026-10-17 as "a selection … NOT a retrospective claim that D1
  passed"; D1's confirmation stays NO-GO.
- Terms (`method`, `execution`): FFN `activation_importance_v0`, DEPTH
  `causal_kl_greedy_v1`, WIDTH `global_pca_v0`, ATTENTION
  `activation_importance_v1`; `positions.supervised_target_v1`; Top-200 plus
  tail; beam 6 `delayed_prune` with `pareto_multi_objective@v2`; micro-batch 3,
  `length_sorted_v1`, `scored_v1`.

**The Stage-0 searches ran under this exact profile.** Both
`runs/dose_d2_20261020_002929/evidence/d2/search/search_record.json` and
`runs/dose_d4_20261020_212310/evidence/d4/search/search_record.json` record
fingerprint `e627f461…` and `prediction_rows: scored_v1`.

**Geometry** is the D-series target with `state_eval@v1`
(`plans/volume_preregistration.json :: fixed`). Every candidate has
596,049,920 parameters in one 1,192,135,096-byte `model.safetensors`.

### 1.2 The Stage-0 Top-4 per dose: 12 distinct initializations [IMPL]

**Rule.** Each dose's complete leaves in its search's quality order, with
byte-identical leaves counted once (`plans/top4_extension.json ::
candidates.rule`).

**Identity owners:** d1/d2 `plans/top4_extension.json :: candidates.by_dose`;
d4 `runs/dose_d4_20261020_212310/evidence/d4/finalists/finalists_manifest.json`;
per-file relay lists in `configs/stages/stage-0/calibration_volume/probes_<dose>_session.json`.
**Bytes:** HF relay `AlphaAvatar/aadistill-artifacts`, path
`stage0_calvol/finalists/formal_v1/<dose>/<state_id>/`.

| dose | rank | state_id | operator path (profile per step) | checkpoint sha256 |
| --- | --- | --- | --- | --- |
| d1 | 1 | `72b76a1c…` | DEPTH(base)→FFN(rh)→ATTN(rh)→WIDTH(rh) | `49bb3c56…` |
| d1 | 2 | `60e229ac…` | DEPTH(base)→FFN(rh)→ATTN(rh)→WIDTH(base) | `a092bc43…` |
| d1 | 3 | `4df438a8…` | DEPTH(base)→FFN(rh)→WIDTH(base)→ATTN(base) | `3d4cb00e…` |
| d1 | 4 | `91090067…` | ATTN(base)→DEPTH(base)→FFN(base)→WIDTH(base) | `11195871…` |
| d2 | 1 | `9af1989d…` | DEPTH(rh)→FFN(base)→ATTN(base)→WIDTH(base) | `e9eba225…` |
| d2 | 2 | `e2d10ba8…` | ATTN(base)→DEPTH(base)→WIDTH(base)→FFN(rh) | `34712d06…` |
| d2 | 3 | `3b2037bc…` | DEPTH(base)→FFN(base)→ATTN(base)→WIDTH(rh) | `551e6685…` |
| d2 | 4 | `b6a41ee8…` | ATTN(base)→DEPTH(base)→FFN(rh)→WIDTH(base) | `4da365c3…` |
| d4 | 1 | `1301deb7…` | ATTN(rh)→DEPTH(base)→WIDTH(base)→FFN(base) | `fc28e0dd…` |
| d4 | 2 | `167a43f4…` | ATTN(rh)→DEPTH(base)→WIDTH(base)→FFN(rh) | `a509ac31…` |
| d4 | 3 | `0363ad22…` | ATTN(rh)→DEPTH(base)→FFN(rh)→WIDTH(base) | `5d54e370…` |
| d4 | 4 | `912205b5…` | ATTN(rh)→DEPTH(base)→FFN(rh)→WIDTH(rh) | `ac7f3885…` |

- **Profiles.** "base" and "rh" are `calib.s0v_{base,rh}_<dose>@v1`, drawn from
  `base_pool_v1` (§2). Hashes and covariates:
  `plans/volume_preregistration.json :: treatments_by_id`.
- **Step-0 state of each Top-1 against the teacher** (diagnostic only,
  `design/formal_analysis.json :: separation.<dose>.step0_state_evaluation`):

  | | d1 | d2 | d4 |
  | --- | --- | --- | --- |
  | teacher-KL, equal-domain mean | 8.03 | 7.98 | 6.76 |
  | pooled NLL (teacher's own: 0.51) | 8.35 | 8.29 | 7.12 |

- **Search rank versus recovered usability** is described from n = 4 per dose
  and never tested (`design/extension_analysis.json ::
  quality_order_beside_outcome`).

### 1.3 Earlier reference initializations and anchors

Source: `logs/maintenance/inventories/checkpoint_registry.json` (created
2026-10-17) unless noted. Each local path was confirmed to exist while this was
drafted.

| what | weights sha256 | bytes live at | relay |
| --- | --- | --- | --- |
| Stage-1 PCA/sandwich init `qwen3_0p6b_init_v0`, the fork point of every E-series recovery | `86fbba78…` | `artifacts/stages/stage-1/qwen3_0p6b_init_v0/checkpoint` | verified |
| D1-q2, the baseline's **historical** reference init, made under `all_v1` | shard `ca2d7368…` (artifact `3a4b4ede…`) | `/home/ecs-user/aad-artifacts/phase_d1/d1_replay_002/products/afd50c9f…` | no |
| `e8_contribution_init_v1` | `7a0694a5…` | `artifacts/stages/stage-1/e8_contribution_init_v1/checkpoint` | no |
| C1 incumbent arm `fe9683e6…` (closed) | `d025f38d…` | `/home/ecs-user/aad-artifacts/autoinit/phase_a/fe9683e6…` | no |
| permanent 0.86M controls `ctl_r0860k_sa`/`sb`. These are recovered students, PCA-initialized | `573847a7…` / `4c6adcf8…` | `/home/ecs-user/aad-artifacts/autoinit/preflight_ctl_r0860k_s{a,b}/step_001023/model` | verified |
| E1 2.96M `e1_r2960k_sb_pca` (recovered) | `b658fe39…` | `artifacts/stages/stage-3/rescued/e1_r2960k_sb_pca` | no |
| E1 2.96M `e1_r2960k_sa_pca` (recovered) | not in the registry | relay `e1_scaling_20260801/e1_r2960k_sa_pca/step_002916/model` (`runs/battery_20261018_092345/runtime/session.json`) | yes |

### 1.4 Deleted versus surviving

- **Deleted 2026-10-21**, weights only, under the AGENTS.md 2.5 standing
  retirement; provider-confirmed (`runs/netvol_*/closeout/tombstone.json`).
  `ugllmlv19t` (300 GB) held d4's search states, the 3 A100 Top-1 students and
  d4's optimizer state. `b9jwu0g1eg` and `wn3avnine2` held 14 and 7 L40S
  students.
- **All 24 recovered Stage-0 students' weights are gone.** No student
  `model.safetensors` remains under `artifacts/stages/stage-0/calibration_volume/`
  or `/home/ecs-user/aad-artifacts/stage0_calvol/` (checked while drafting).
  - Their sha256s survive (`extension_analysis.json ::
    per_unit.*.recovery.model_sha256`; `recovery_products.json`).
  - A rebuild is "not bit-for-bit": about 1 h of L40S recovery per student
    (tombstone `reconstruction`).
- **The 12 initializations exist only on the relay.** There is no local
  `artifacts/…/candidates/`. All 12 were re-verified there on 2026-10-21 (LFS
  sha256 and size against the finalists manifests) and are registered
  PROTECTED (`reproducibility_required`) in
  `logs/maintenance/inventories/checkpoint_registry.json ::
  scientific_artifacts.stage0_calvol_<dose>_top<k>_*`. Search states can be
  replayed from each dose's journal (`plans/top4_rebuild_d1.json` mechanism),
  for "~$12-24" per dose (tombstone).

## 2. Existing Stage-2/3 data assets and their coverage

Each data overlap is from `data/eval-set/eval-set-1/audit/isolation.json`.

| asset (manifest) | content, size | teacher, decoding | verification | P10.2 coverage | bytes |
| --- | --- | --- | --- | --- | --- |
| `stage2_v1` (`data/stages/stage-2/stage2_v1/stage2_offline_v1.manifest.json`) | public data, **no teacher targets**: 64,484 / 1,916 / 200 rows; "22.13M train tokens" (`EXPERIMENTS.md` l.56) | — | — | by decisions.md 2026-10-16: `code_math` and `multihop_qa` mixed; rag, tool, instruction and long_context supporting; realtime and refusal evaluation-only; **no primary group** | local, gitignored; train shares 4 prompts with `eval_behavior_v0`, and val is its source |
| teacher corpus v1 / `stage3_pilot` (superseded) | 540 of 752 accepted | T = 1.0, "not the official preset" | not found | none | relay `stage3_teacher_corpus_20260730/` |
| **`corpus_v2`** (`artifacts/stages/stage-3/corpus_v2/manifest.json`) | 11,174 sessions in 6 families; 66.08M generated tokens; $25.56 | Qwen3-4B-Thinking-2507 @`768f209d`, vLLM 0.26.0, 0.6/0.95/20/0, n = 4, thinking mode | **hygiene only** (`correctness_computed_but_not_enforced`); openmath correct 0.380 | **0 primary-axis items, 0 rule families** (`audit/data_audit.json`; REPORT §2) | local and relay; legacy-150 overlap by construction; held-out 0 |
| `ladder_uniform_probe` (`artifacts/shared/instruments/ladder_uniform_probe/ladder.json`) | 6 nested rungs from `corpus_v2`, 0.25M→5.50M supervised tokens, 3,715 blocks × 8,192 | as `corpus_v2` | as `corpus_v2` | as `corpus_v2` | local; blocks `6f324cb0…` |
| E5, E7 streams; `corpus_v2_clean` | `artifact_manifests.md` l.19-52, l.336-383, l.872-899 | — | not found / length rule | none | relay / local |
| `calibration_v1` / `v2` (`data/stages/stage-0/calibration_volume/tasks/`) | procedural **prompts**, calibration-split families: 3,072 / 672 | — | generator uniqueness (DESIGN §2) | all five axes | git; overlap 0 |
| **`teacher_cal`** (`runs/teachercal_20261018_112755`) | 672 tasks → 1,011 turns, 4,528,160 output tokens; ends: answered 574, degeneration 17, episode_done 71, length_limited 10 | vLLM 0.27.1, 0.6/0.95/20/0, 16,384 data-construction limit | gold-checked downstream | all five axes | local + relay (`relay_copies.json`) |
| **`base_pool_v1`** (`artifacts/stages/stage-0/calibration_volume/base_pool_v1/manifest.json`) | 6,593 items, ≤ 8,192 tokens. **Fluid: 519 units (506 static + 13 episodes), 2,160,380 scored positions.** Supporting: 6,020 `corpus_v2` units, 5,142,118 positions | as above | verified correct and naturally terminated | fluid + supporting | **local only**; overlap 0 |
| `stage3_legacy_150` | 150 prompts from the 0.86M rung, no targets | — | `derivation.json` | — | git, and **is** in-sample |

**What this means for Stage 2 (facts only):**
- **No teacher-native fluid-reasoning training set exists.** The only verified
  teacher-native fluid units are `base_pool_v1`'s 519, used only as search
  calibration data; no record shows them used for recovery training.
  "Verified teacher-native traces on calibration-split families" as a recovery
  recipe is a proposal only [PLAN] (REPORT §10, step 1).
- **Every Stage-0 recovery trained only on `corpus_v2`** (the 0.86M rung, §5).
- **Measured teacher generation rate** [MEAS]: 4,528,160 output tokens in
  2,603.7 s (1,739 tok/s) on one A100-80GB with vLLM 0.27.1, for $1.6169
  all-in (`runs/teachercal_20261018_112755/closeout/outcome.json`;
  `teacher_cal.summary.json`). The round-synchronous runner was "too slow" at
  the teacher's ~4.5k tokens per trace (decisions.md 2026-10-18).
- **Length** (`docs/stages/stage-0/calibration_volume/DESIGN.md` §3):
  - `base_pool_v1` dropped 53 units over 8,192 tokens;
  - D1's `[256, 2048]` window would have kept 66 of 572 verified units, with no
    analogy or interactive unit.

## 3. Historical conclusions (mixed-domain or in-sample; AGENTS.md P10.2 "History is not reinterpreted")

- **`corpus_v2`** [MEAS]. Its gate "PASSED — every check for all six types"
  (`EXPERIMENTS.md` l.280-283). "Acceptance is hygiene, not correctness …
  roughly a third of accepted openmath targets teach a wrong final answer"
  (l.294-299). E2: "do not adopt the cleaned corpus" (decisions.md 2026-08-04).
  The Stage-0 audit found it "covers none of the primary axes" (REPORT §2).
- **"Nested token ladder"** names three different things:
  1. the warm-up sub-stage ladder, retired as "unnecessary at this data scale"
     (decisions.md 2026-07-27);
  2. the six-rung **nested `corpus_v2` ladder** [IMPL]: "nesting is exact and
     monotonic" (`EXPERIMENTS.md` l.302-340), with "3.35×" padding compute
     accepted (decisions.md 2026-08-01). At 0.86M it holds 1,472,149 real
     against 4,114,795 padding tokens (`ladder.json :: rungs[2]`);
  3. a **teacher-native ~60k→240k→960k→3.84M-position ladder**: "DIRECTIVE
     RECEIVED 2026-10-01. NOT STARTED" [NEVER-RUN]
     (`logs/stages/stage-1/phase_c1/plans/phase_c_roadmap.md` l.683-701).
- **E1 data-scaling matrix** [MEAS], complete 2026-08-02 (`EXPERIMENTS.md` §11;
  `logs/state/experiment_index.md` l.24-41).
  - Design: 24 arms (6 rungs × PCA/random × 2 seeds) plus a step-matched
    control; ce 0.25 / kd 1.0.
  - Verdict: "a **data** effect on teacher-native CE … **not** … a general
    capability effect — behaviour does not follow it, and reasoning is absent
    from every arm" (l.647-649).
  - Accepted: "PCA/structural initialization beats random". Not supported:
    "that scale improves reasoning *correctness*".
  - GSM8K EM maximum 0.050; `behavior_score_v0` noise floor 0.1290.
- **E6/E6b/E7** [MEAS] (`logs/state/supported_models.md`). "The E1/P1 lineage
  plateaus after 2.96M", measured on the in-sample 150. "The plateau belongs to
  the E1 objective, not to the data" (decisions.md 2026-08-09). E7's
  general-text KD restores language modelling by −5.22 nats and "moves
  autonomous behaviour by nothing".
- **The only fluid measurement of E1 students** [MEAS] (REPORT round 2;
  `design/resolution_check.json`). PCA students score 0/720 at 0.86M (sa and
  sb), 7 at 1.60M, 10 at 2.96M and 2 at 5.50M. "More recovery data of this kind
  does not help."

## 4. Verified mechanisms

Tags: **[GPU]** worked in a paid run; **[UNIT]** unit tests only; **[NOT FOUND]**
no record.

| mechanism | code | validated by | recorded limits |
| --- | --- | --- | --- |
| teacher-native generation | `scripts/shared/rollout/build_recovery_corpus.py` (vLLM; 0.6/0.95/20/0; n = 4; budget = 8,192 − prompt − 8 via `src/aadistill/data/sessions.py :: completion_budget`; `<think>` never prefilled; target stored as `reasoning_content` + `content`); Stage-0 `scripts/stages/stage-0/calibration_volume/run_battery.py --mode generate` (asynchronous runner; the limit is recorded as `length_limited`, never trimmed); engines in `src/aadistill/rollout/engines.py` | [GPU] `corpus_v2`: L40S, 16.5 h (`EXPERIMENTS.md` §10); [GPU] `teacher_cal` (§2) | no standing rollout service; `corpus_v2` manifest has no git commit and is not token-reproducible (bf16 batch variance; `artifact_manifests.md` l.140-148); the 8,192 budget rejected 1,541/3,600 openmath and 702/4,800 code candidates; `teacher_cal` records `tokenizer_sha256` and `image_digest` as null |
| target verification | `src/aadistill/data/verify.py` (span / final_number / boxed / refusal); `src/aadistill/data/cleaning.py`; Stage-0 gold in `score_battery.py` | [GPU] `corpus_v2` gate (hygiene); `base_pool_v1` keeps verified units only; [UNIT] `tests/data/test_verify.py` | code and tool calling have no key (`unverifiable_slice`) |
| packing for training | `src/aadistill/data/sessions.py :: pack_sessions`: sequential within a system-prompt group; only the last session is cut | [GPU] efficiency 0.47; the median block is "14.1% full (E1)" (`EXPERIMENTS.md` l.747) | **no length-aware packer for training**; `batch.truncate_padding` (2.69×, `EXPERIMENTS.md` l.1327) is off in the frozen recipe |
| packing for calibration | `src/aadistill/initialization/calibration/packing.py :: LENGTH_SORTED_V1`, part of the forward baseline | [GPU] Stage-0 searches | its C3 pilot verdict was `PACKED_BATCH_NOT_WORTH_ADOPTION`, 1.1884× against a 1.25 gate (`logs/stages/stage-1/phase_c3/pilots/packing-optimization/v2/result.json`) |
| recovery training | `scripts/shared/training/train_stage3.py` → `src/aadistill/training/train.py :: Trainer`; `masked_ce` + full-vocab `kd_forward_kl` with the teacher **online**; `kd_scope` all / assistant / all_no_think; AdamW; Stage-0 `dose_driver.py :: train_recovery` and `probe_driver.py`, which check `C1_PROBE_OVERRIDES` | [GPU] 24 Stage-0 recoveries, E1's 24 arms; [UNIT] `test_resume_is_bitwise_exact` | **no top-k KD** (decisions.md l.4390); the CE mask (`src/aadistill/data/dataset.py :: final_assistant_loss_mask`) weights thinking and answer alike; [NOT FOUND] a GPU resume in the middle of training |
| evaluation | `scripts/shared/evaluation/eval_set_scoring.py`; `scripts/shared/eval_sets.py`; `src/aadistill/evaluation/{eval_registry,usable_rollout}.py`; `score_battery.py` (replays episodes); `scripts/shared/evaluation/uncapped_eval.py` (P18 allowance = context − prompt; `right_censored`) | [GPU] every Stage-0 unit, e.g. `runs/dose_d1_20261019_133708/evidence/d1/eval/report.json`; [UNIT] conformance (`tests/data/test_eval_set_registry.py`) | students run at their trained 8,192 context ("not a 262K-context evaluation"); a degeneration abort, checked every 256 tokens, is an extra stop condition recorded as right-censored (`uncapped_eval.py` l.412-422); [NOT FOUND] conformance run inside a paid session |
| durability at completion | Stage-0 `launch.py` pulls each finished unit and hash-checks it; `volume_sync.py` / `require_pushed`; `src/aadistill/runtime/hub_capacity.py :: would_accept` | [GPU] the closeouts' per-file sha256s; restore verified against `recovery_products.json` (`runs/dose_d1_20261019_133708`) | `src/aadistill/runtime/durable_store.py` has "NO PRODUCTION CALLER"; the volumes are deleted, so **no large-artifact backend is provisioned**; the relay's current LFS room is not found in records, and earlier records say its quota filled and "ordinary deletion does not reclaim it" (`artifact_manifests.md` l.77-82, l.503-506) |

## 5. The fixed recovery recipe used by Stage 0

Recipe `e1_p1_kd_heavy@0.86M` (`configs/stages/stage-3/recipes/recovery.json`);
config `configs/stages/stage-3/e1/e1_r0860k_sa_pca.json`, sha256 `19e1e3f6…`
(`plans/volume_preregistration.json :: recovery`). Stage 0 changed only the run
identity, `out_dir`, pack path, `student_path` and seed
(`plans/top4_extension.json :: units.recovery`).

| field | value |
| --- | --- |
| data | 0.86M rung of `ladder_uniform_probe`: 682 blocks × 8,192; 864,750 supervised tokens; 1,502 `corpus_v2` sessions; 109 terminal truncations (`ladder.json :: rungs[2]`) |
| mixture | uniform six-way **token** mix (gsm8k, openmath, code, tool, rag, multihop); session mix 38% tool, 24% rag; "Data mixing is a later experiment" (`mixture_rationale`) |
| loss | CE 0.25 + KD 1.0, T = 1, `kd_scope: all`; teacher bf16, online |
| trainable | attention q/k/v/o and q/k norms, MLP gate/up/down, layer norms, final norm |
| optimizer | AdamW, lr 5e-5, wd 0.01, betas (0.9, 0.95), clip 1.0; 1,023 steps (warm-up 51, floor 0.1); 2 blocks per step, i.e. 3 epochs; fp32 weights with bf16 autocast |
| seeds | 20260726 (preregistered; the A100 Top-1s) and 20260727 (extension) |
| context | trained and evaluated at 8,192 |
| measured time | 3,019–3,149 s of training on an A100 (`extension_analysis.json :: per_unit.*/A100.recovery.seconds`); 6,264–7,900 s per L40S unit including eval (`runs/probes_d1_20261020_093955/evidence/probe_session.json`) |

**Recorded limitations:**
- **H2 is unresolvable under this recipe** (decisions.md 2026-10-18). For the
  PCA controls, "both controls score 0/720 and never disagree, so there is no
  minimal detectable effect". Changing the recipe "is a change … AGENTS.md
  §4.5 reserves for the maintainer" (REPORT §10).
- **D1-initialized students sit on the same floor**: all 24 units score at most
  3/570 and 0/144 (REPORT round 3). Final validation CE on pack-tail blocks is
  1.30–1.46 for every unit (`extension_analysis.json ::
  per_unit.*.recovery.final_val`). Training fits a corpus that has no fluid
  content.
- **Usability follows the seed** (+3.9 to +24.9 points); correctness does not
  (REPORT round 3).
- **Floor diagnostic** (`design/floor_diagnostic.json`, the PCA controls,
  `_not_an_endpoint`). Controls are often usable but rarely box an answer:
  `ctl_r0860k_sa` deduction has 101/192 usable, 1 boxed, 0 strict-correct.
  Episodes end at `turn_limit` or by `degeneration`, with invalid-action rates
  of 0.96 / 0.91.

## 6. Evaluation isolation that must be protected

| item | fact | owner |
| --- | --- | --- |
| registry | `default_formal_measurement` = `eval-set-1`, **frozen**, identity 2 `62a1a5d3…` (2026-10-19). A change needs an appended identity citing a maintainer authorization | `data/eval-set/registry.json`; `data/eval-set/README.md` rule 2; P10.3 |
| components | `fluid_reasoning_720` (576 static + 144 episodes): **development-exposed**, "not a never-seen blind set". `stage3_capability_846`: held out. `stage3_legacy_150`: **in-sample** (0.86M-rung training prompts), never reported beside held-out numbers | `data/eval-set/eval-set-1/README.md` |
| formal core | fluid 714 = 570 + 144 (`validity_v1.json`; 6 defective items excluded, none changed). TriviaQA `knowledge` is **withheld** from git and supplementary. `eval_behavior_v0`: 65 of 76, without 7 xLAM prompts and 4 Stage-2-train overlaps. Every item is still generated | same |
| scoring | protocol v2 (answer units) and action protocol v2 (episodes), pinned by `conformance/*.v2.jsonl`. `safety_paired`, `eval_behavior_v0` and historical results stay on v1 | same |
| isolation API | `require_disjoint` refuses data sharing a `prompt`/`content` key with a held-out unit; builders reserve `held_out_paths()`; every read passes `eval_registry.require_intact` | `scripts/shared/eval_sets.py` |
| generator split | 5 of 15 generators are evaluation-only (`sequence_induction`, `counterfactual_arith`, `graph_paths`, `letter_analogy`, `code_breaker`). The other 10 split families by salted hash, ≈45% calibration / 55% evaluation. New sets use generator **v2**; v1 rebuilds the frozen sets under `PYTHONHASHSEED=14` | `docs/stages/stage-0/calibration_volume/DESIGN.md` §2 |
| known overlaps | `corpus_v2` ↔ legacy 150; `stage2_v1` train ↔ `eval_behavior_v0`, 4 prompts; `stage2_v1` val is its source (64); every Stage-0 pool, profile and calibration set: 0 | `audit/isolation.json` |
| open decisions | xLAM terms; a TriviaQA redistribution basis; Code-Owners protection, "not binding" against the agents' own admin token | `registry.json :: open_decisions`; `data/eval-set/README.md` |

## 7. How the current students fail (24 D1-initialized units on eval-set-1)

**7.1 Recorded rates** (`design/extension_analysis.json :: per_unit`; formal
rows only).

| quantity, over 24 units | range |
| --- | --- |
| fluid static correct (/570); `answer_correct_ignoring_usable` | 0–3; 0–3, so usability hides no correct answers |
| fluid static usable | 40.0–71.4% |
| episode success (/144); all turns usable | 0 in every unit; 0–20.1% |
| `no_context_limit` (static) | 1.0 in every unit |
| math_verified / gsm8k / multihop / rag correct (/100) | 0–1 / 0–2 / 1–8 / 3–15 |
| safety / answerability pairs (/50, /60); knowledge (/150, supplementary) | 0–3 / 0–3; 0–2 |
| legacy 150 (in-sample) | 14–28 |

**7.2 Recorded failure reasons, capability units**
(`design/formal_analysis.json :: H2.per_dose.<dose>.units.*.metrics.reasons`).
Triples are d1 / d2 / d4 Top-1 at seed 20260726.

| unit | reasons |
| --- | --- |
| math_verified | `not_terminated` 45 / 40 / 34; `no_boxed` 43 / 44 / 44; `answer_mismatch` 6 / 14 / 15 |
| gsm8k | `answer_mismatch` 43 / 40 / 50; `no_final_answer` 27 / 23 / 21; `not_terminated` 27 / 34 / 28 |
| multihop | `answer_mismatch` 79 / 81 / 72; evidence recall 0.185 / 0.16 / 0.11 |
| rag | `echoed` 54 / 33 / 41 and `answer_mismatch` 27 / 46 / 43: the model copies the passage instead of answering |
| answerability | `answered_unanswerable` 52 / 51 / 56 of 60; refusal on answerable items ≈ 0: the model never abstains |
| safety_paired | `unexpected_tool_call` 23 / 44 / 66, i.e. a tool call where none was offered; `complied_with_unsafe` 7 / 7 / 4 |
| knowledge | `unexpected_tool_call` 48 / 56 / 79; `not_terminated` 59 / 42 / 27 |

**Label caveat.** `eval_behavior_v0` reports `truncated_at_cap` 0.18–0.25, but
`eval_set_scoring.py` (l.136-137) sets it to `stop_reason != "eos"`, so it
counts degeneration stops, not context hits.

**7.3 Fluid failures [DERIVED].**

*Scored rows.* All 24 units' `fluid_reasoning_720__dev_v1.rows.jsonl` match their
`extension_analysis.json :: inputs` sha256. The reasons follow
`scripts/stages/stage-0/calibration_volume/score_battery.py :: score_static`.
Over 13,680 formal static rows:

| reason | rows | share |
| --- | --- | --- |
| `no_boxed`: usable, but no `\boxed{}` final answer | 6,838 | 50.0% |
| `unusable` | 5,933 | 43.4% |
| `wrong_answer` | 892 | 6.5% |
| `ok` | 17 | 0.1% |

*Generation records.* From
`/home/ecs-user/aad-artifacts/stage0_calvol/{probes_*,dose_*}/store/…/generations/fluid_reasoning_720__dev_v1.jsonl`,
576 static items × 24 units = 13,824 rows. `stop_reason`: eos 9,018,
degeneration 4,806 (34.8%; `cycle` 4,135, `low_novelty` 436, `rambling` 235).
`</think>` is absent from 3,312 rows (24.0%). Eos-terminated replies have a
median of 312 generated tokens (p90 623). The teacher on the same items
(`teacher_dev.jsonl`, sha256 `98f08abc…` per `relay_copies.json`) has a median
of 3,264 (p90 7,965, maximum 22,411).

*Episodes.* 3,456 episodes end by degeneration 2,500, at the turn limit 936,
and `episode_done` 20, with 0 successes and 4.71 turns on average.

*Engine.* vLLM 0.27.1, greedy, `max_model_len` 8,192, degeneration check every
256 tokens (`artifacts/stages/stage-0/calibration_volume/dose_d2_20261020_002929/d2/eval/generations/fluid_reasoning_720__dev_v1.summary.json`).

**7.4 Reference point.** Qwen3-0.6B, at the same geometry and the same 8,192
allowance, scores 257/720 (REPORT round 2). By axis: deduction 114, composition
85, induction 56, analogy 2, interactive 0. The set resolves student-scale
models on three static axes but not on analogy or interactive. This is a
diagnostic, "not Qwen3-0.6B's P18 score".

## 8. Reusable artifacts and their durability

Per `artifacts/README.md`: `artifacts/` and `/home/ecs-user/aad-artifacts/` are
durable; the relay is "transport/second copies"; `aad-scratch` is disposable.

| artifact | where | bytes |
| --- | --- | --- |
| the 12 Top-4 initializations | relay `stage0_calvol/finalists/formal_v1/…` | relay only (§1.4) |
| the 24 recovered Stage-0 students | deleted | sha256 only |
| per-unit generations, scored rows, `report.json`, `train_log.jsonl` (24 units) | `/home/ecs-user/aad-artifacts/stage0_calvol/{dose_*,probes_*}/store/` (1.4 GB); A100 units also in `artifacts/stages/stage-0/calibration_volume/dose_*/` | durable, local |
| per-unit reports, configs, `recovery_products.json` | `runs/probes_*/evidence/<unit>/`, `runs/dose_*/evidence/<dose>/` | git |
| search journals, telemetry, `structure.json`, width projections | `/home/ecs-user/aad-artifacts/stage0_calvol/dose_*/store/<dose>/search/` | local, dev box only (no off-host copy) |
| the A100 Top-1 units laid out for `analyze_extension.py` | `/home/ecs-user/aad-artifacts/stage0_calvol/formal_v1_assembled/` (symlinks into the stores) | local |
| everything on the 300 GB volume except weights | `artifacts/stages/stage-0/calibration_volume/volaudit_20261021_142638/` | local |
| `teacher_cal`, `teacher_dev`, `ctl_*_dev`, `e1_r2960k_sa_pca_dev` records | `artifacts/…/teachercal_*/`, `…/battery_20261018_092345/` | local + relay (`relay_copies.json`) |
| `base_pool_v1`, `profiles_formal_v1`, `profiles_pilot_v1` | `artifacts/stages/stage-0/calibration_volume/` | local only |
| `corpus_v2`, `ladder_uniform_probe` | `artifacts/stages/stage-3/corpus_v2/`, `artifacts/shared/instruments/` | local; `corpus_v2` also on the relay |
| eval-set-1 | `data/eval-set/eval-set-1/` | git, except TriviaQA |

## 9. Budget-safety mechanisms in place, and recorded risks

**Money.** Project: $625.5798 spent of $643.6571, **$18.0773 remaining**,
`remaining_is_not_permission: true` (`logs/state/current.json :: budget`). The
Stage-0 programme spent $125.4326 of $143.5099 (`programme_closeout.json ::
money`). No session or authorization is open (`current.md`, "Right now").

| mechanism | code | enforces | exercised |
| --- | --- | --- | --- |
| one-use authorization | `src/aadistill/governance/authorization.py`; Stage-0 `governance.py :: issue` | one grant per run, bound to the plan hash and harness digest | [GPU] `runs/dose_d1_20261019_054503` refused on the pod ($0.2541); `dose_d4_20261020_045426` refused at $0 |
| programme envelope | `scripts/stages/stage-0/calibration_volume/governance.py` | reads closeouts directly; refuses an unknown cost; books a concurrent run at its hard cap | [GPU] `runs/dose_d4_20261020_212310/governance/authorization.json :: stage_conditions.envelope`; no refusal recorded |
| account balance | `session_runner.py :: check_account_funds`; Stage-0 `launch.py` requires $30 | the provider account must cover the session | [GPU] `runs/dose_d1_20261019_133708/runtime/session.json :: account_balance` |
| watchdog | `src/aadistill/infrastructure/watchdog.py`; `scripts/shared/pod/watchdog.py` | hard time limit, then termination verified | [GPU] fired: `logs/stages/stage-3/e7/history/e7_canary/watchdog_test.jsonl` |
| same-failure gate | `src/aadistill/infrastructure/failure_signature.py`; `session_prechecks.py :: same_failure_gate` | refuses an unchanged deterministic paid retry | [GPU] passed in A3 (`logs/stages/stage-1/phase_a3/runs/a3_attempt35/runtime/session.json`); **not wired into Stage-0 `launch.py`** |
| budget thresholds | `src/aadistill/infrastructure/budget.py` | expected / soft stop / reserve / hard terminate | — |

**Fixed at this closeout** (decisions.md and ledger.md 2026-10-21):
- **Run-index gap.** 28 closed Stage-0 runs (`$107.9227`) were missing from
  `logs/index.json`, the only place `derive_budget.project_sessions` looked, so
  the project total read ~$108 low while the programme's own gate (which reads
  closeouts) was right. `derive_budget.py :: _sessions_the_index_misses` now
  counts closeouts and open sessions found on disk, once, and names them; an
  open session with no hard cap leaves the project remainder null. Tests:
  `tests/docs/test_log_organisation.py :: TestTheProjectTotalDoesNotDependOnTheIndex`.
- **Unrecorded provider resources.** `provider.py :: reconcile_account` and
  `SessionRunner.check_account_resources` (on by default, before any create;
  also the shared staging and canary pods) refuse while the account holds a pod
  or network volume the session's records do not account for, or cannot be
  listed. A session that runs beside another or mounts a volume declares those
  ids in `BudgetSpec.accounted_provider_resources`. Tests:
  `tests/infrastructure/test_account_balance_gate.py`.
- **Provider billing beside the records.**
  `scripts/maintenance/consolidation/provider_billing.py` fetches the provider's
  billing (read-only) and attributes each billed pod to the run that created it;
  snapshot `logs/budget/provider/runpod_billing_2026-10-21.json`.

**Open risks:**
- **Provider billing versus records** (`closeout/budget_reconciliation.json`):
  the provider billed $658.15 against the records' $625.5798, a $32.5702
  difference that falls entirely before 2026-10-18 ($37.93 before the
  2026-09-11 anchor). 53 billed pods are named by no record. Attribution is
  UNVERIFIED and the decision is the maintainer's; if it is all project spend,
  the $643.6571 cap is exceeded by $14.49, and the $18.0773 "remaining" is not
  real headroom.
- **No large-artifact backend** is provisioned (§4). AGENTS.md P8.2.1 requires
  confirming one before a long run produces large artifacts.
- **Dev-box disk:** 13 GB free of 197 GB (`df` while drafting; not a record).

## 10. Directions the maintainer has named (DIRECTIONS ONLY)

**Source.** The maintainer's final-closeout instruction of 2026-10-21, recorded
in `logs/budget/decisions.md` 2026-10-21 (final closeout). These are **not** a
frozen protocol, and they carry **no budget and no grant**.

1. **Verified teacher-native reasoning training data** for Stage 2, then Stage-3
   recovery. The Stage-0 mixture ratio is deferred.
2. **Independent rule families and verifiable reasoning tasks.**
3. **A fixed initialization.**
4. **A controlled old-versus-new data-quality comparison**: the current
   `corpus_v2` recovery data against new verified reasoning data.
5. **Later**, a **nested training-data ladder**.
6. **eval-set-1 fixed** as the measurement.

Related text already in the repository, all unauthorized design:
- `AGENTS.md` §4.4 "Reasoning-transfer priority".
- `docs/shared/REASONING_TRANSFER_DESIGN.md` §4. Six factors (quantity,
  effective reasoning tokens, difficulty, rule-family diversity, verified versus
  unverified trajectories, mixture), with effective reasoning tokens held fixed
  across mixture comparisons and trajectory length controlled.
- REPORT §10 step 1: validate a new recipe with one probe of "about $2". The
  control must score above the floor, and two seeds must disagree enough for a
  minimal detectable effect at or below the 0.05 SESOI.
- The 2026-10-01 ladder directive (§3), never started.

## 11. Record inconsistencies found at the closeout

Items 2-4 and 6 were repaired at the closeout; 1 and 5 stay open because each
would change a frozen identity.

1. **Forward profile "unmaterialized".** Three records say no checkpoint
   exists under the forward profile:
   - `initialization_forward_baseline_v1.json`: `forward_materializations: []`,
     "NO checkpoint has been materialized";
   - `logs/state/current.md`, D-series block: "NOT MATERIALIZED";
   - `REASONING_TRANSFER_DESIGN.md` §6.

   The Stage-0 searches ran under `e627f461…` / `scored_v1` (§1.1). Updating
   the record means re-selecting its sha256 in `selection.json`. **OPEN**: the
   forward baseline is not reselected (maintainer, 2026-10-21); the Stage-0
   candidates are the forward profile's first materializations in fact.
2. **Stale experiment status.** "TERMINAL 2026-10-18, round 2 (formal study NOT
   RUN)" appears in `logs/stages/index.json` l.1043 and
   `logs/stages/stage-0/README.md` l.58, and in the headers of
   `docs/shared/STAGE0_HANDOFF.md` and `REASONING_TRANSFER_DESIGN.md` §3.
   REPORT round 3 and `current.md` say COMPLETE (2026-10-21). **REPAIRED**:
   `stage_attribution.py` (which renders both) and the two documents' headers
   now say CLOSED 2026-10-21.
3. **Relay-only initializations.** `artifacts/README.md` says a relay copy "is
   never the durability claim". Yet the 12 Top-4 initializations exist only on
   the relay, and the deletion of `ugllmlv19t` relied on that (its tombstone
   `why`). **REPAIRED in the records**: registered PROTECTED in
   `checkpoint_registry.json` and listed in `artifact_manifests.md`. The single
   copy remains a risk only a second store resolves (a maintainer decision).
4. **Evidence cited from scratch paths.**
   - `extension_analysis.json :: inputs` cites the A100 units' evidence under
     `/tmp/claude-1000/…/scratchpad/formal_all/…`. Durable byte-identical copies
     exist: 63/63 JSONL match by sha256 under
     `/home/ecs-user/aad-artifacts/stage0_calvol/dose_*/store/`. **REPAIRED**:
     the analysis was re-run from `formal_v1_assembled/`; only those 69 input
     paths changed.
   - `floor_diagnostic.json :: runs.*.records` point into the disposable
     `/home/ecs-user/aad-scratch/…`. Copies exist in
     `artifacts/…/battery_20261018_092345/` and on the relay.
5. **Mislabelled field.** `truncated_at_cap` is documented as a
   `max_new_tokens` cut (`behavior.py :: score_sample`) but computed as
   `stop_reason != "eos"` (§7.2).
6. **Moving tree.** `programme_closeout.json` recorded `dirty_at_write: true`
   while the closeout was in progress. **REPAIRED**: rewritten at the final
   closeout commit.
