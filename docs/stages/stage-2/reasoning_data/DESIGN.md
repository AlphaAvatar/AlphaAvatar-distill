# Stage-2 verified reasoning data → Stage-3 controlled recovery: the design

**Status: $0 preparation, 2026-10-21 (UTC). PROPOSED, not preregistered, not
funded.** It authorizes nothing.

> **Revised 2026-10-22.** After an independent review, the task pool is
> `train_tasks_v2` (32 families), the target contract is `target_quality` v2
> (answer, text consistency, steps and action legality kept apart), isolation
> is R0–R9, units are admitted whole, and arm A is 3,150 units / 1,459 blocks.
> Those figures, and every other changed one, are in [`H1.md`](H1.md) §10; the
> v1 figures below are the state of 2026-10-21.
>
> **H1 is the current round, and [`H1.md`](H1.md) is its preparation result.**
> The maintainer fixed D4 Top-1 as the initialization, put the budget and
> durability conditions first, set arm B to reasoning 0.70 / supporting 0.30,
> and named the sources to audit first.
> H1.md and the records it names supersede §6 (the comparison, now
> `plans/comparison_design.json` v2), §8 (the initialization, now fixed), §9
> (cost, now H1 only) and §11 (the application). The audit, quality contract
> and isolation sections below stand as background.

Each figure names the record that owns it; where this page and a record
disagree, the record is right.

| record | what it owns |
| --- | --- |
| `logs/stages/stage-2/reasoning_data/audit/asset_audit.json` | the asset audit (`audit_assets.py`) |
| `logs/stages/stage-2/reasoning_data/audit/packing_dry_run.json` | the builder dry run on existing material (`pack_dry_run.py`) |
| `configs/stages/stage-2/reasoning_data/sources.json` | source catalogue: roles, licenses, splits, verifiers |
| `configs/stages/stage-2/reasoning_data/isolation_rules.json` | rules R1–R8 (`isolation_check.py`) |
| `configs/stages/stage-2/reasoning_data/corpus_spec.json` | the proposed master corpus and its acceptance contract |
| `configs/stages/stage-2/reasoning_data/generation_protocol.json` | the teacher-native generation protocol (draft) |
| `data/stages/stage-2/reasoning_data/tasks/train_tasks_v1.*` | the verifiable primary-axis task pool (`build_training_tasks.py`) |
| `logs/stages/stage-2/reasoning_data/plans/*.json` | initialization selection, comparison design, ladder plan, cost forecast, durability plan, authorization proposal |

Everything under `scripts/stages/stage-2/reasoning_data/` is regenerable at $0:
`PYTHONPATH=src:scripts .venv/bin/python -m stages.reasoning_data.<script>`.

## 1. Starting point

- **Repository.** Stage 0 was squash-merged to `main` on 2026-10-21 as `2a8bae78`.
  The source branch `stage0/calibration-volume-mixture` is kept (tip `aee9c4ff`,
  merge base `32246d12`). This round's branch `stage2/reasoning-data` was cut
  from `2a8bae78`, whose tree equals the Stage-0 branch tip's.
- **Stage-0 verdict.** EXECUTION COMPLETE / DOWNSTREAM REASONING FLOOR / VOLUME
  EFFECT ON REASONING NOT RESOLVED (`programme_closeout.json`). Stage-0 volume
  is CLOSED, the mixture ratio is DEFERRED and the D-series is CLOSED. The
  forward baseline is D1 + `scored_v1`, not reselected.
- **Evaluation.** eval-set-1 is frozen at identity 2 (`62a1a5d3`). Its formal
  fluid set is 714 items (570 static + 144 episodes). It is read, never
  rebuilt.

## 2. What the existing assets are (`asset_audit.json`)

**`corpus_v2` covers no primary axis, and most of its targets cannot be shown
correct.**

- Acceptance was hygiene only; correctness was computed but not enforced.
- Selected targets by type:
  - **Wrong answer:** openmath 207/579 (35.8%), gsm8k 190/1,698 (11.2%),
    multihop 152/1,074 (14.2%), rag 88/4,100 (2.1%).
  - **No answer key at all:** code (1,123) and tool_calling (2,600).
- Of its 13.04M selected supervised tokens, **6.06M** carry a verified-correct
  answer.
- Under the new contract (verified correct, naturally terminated, protocol
  valid), openmath keeps 413 of 900 tasks. Its 1,541 length-limited candidates
  hit the 8,192 budget.
- **The openmath key is not a gold.** Its augmented problems' answers are
  majority votes of Llama-3.1-405B (`sources.json`). The historical "0.38
  correctness" therefore partly measures label noise.

**The historical ladder is mostly padding, and it cuts traces.**

- At the 2.96M rung: 1,944 blocks, 70.3% padding, a median block fill of 8.1%,
  and 1,405 single-session blocks.
- 352 sessions were cut mid-trace, discarding 629,315 supervised tokens. The
  cut sessions are still trained, without `<|im_end|>`.
- The cause is tool data: 2,178 tool-dominated blocks at a mean fill of 0.105,
  because 1,861 system-prompt groups (tool schemas) each force their own
  blocks.
- Every rung uses 3 epochs, so the 2.96M rung is 5,832 block passes, every one
  forwarding 8,192 positions through student and teacher.

**The 519 verified Stage-0 fluid units are reusable in principle, but not
adopted here.**

- Composition: 506 static + 13 episodes, 35 rule families, 10 generators.
- **Reconstruction is token-exact.** All 519 units rebuild from their teacher
  records' own conversations, and all 573 items match `base_pool_v1`'s renders
  (`packing_dry_run.json :: fluid_traces.reconstruction`).
- **They pass R1–R4.**
- **What reuse would need.** They were the calibration data the 12
  initialization candidates were selected on, and their seeds sit in the
  calibration range, not the training range (R5, reported for a historical
  pool). Reusing them needs a new corpus identity and a maintainer decision.

**Teacher yield per axis** (`teacher_cal`, n = 1, verified, ≤ 8,192):

| axis | yield (verified supervised tokens per generated token) |
| --- | --- |
| deduction | 0.996 |
| induction | 0.782 |
| composition | 0.464 |
| analogy | 0.438 |
| interactive | 0.127 |

Interactive episodes cost about 8× what deduction costs per trained token.

**Recovery history** (`resolution_check.json`). PCA students on the corpus_v2
rungs score 0, 0, 7, 10 and 2 of 720. E1/E6 found a data effect on
teacher-native CE, not on behaviour or reasoning.

**Isolation of the old data.**
- R1: corpus_v2 shares no prompt or content key with a held-out unit.
- R2: **28 sessions are near-copies of held-out prompts.** 27 copy
  `eval_behavior_v0` prompts, which came from the same `stage2_v1` source
  pool, and 1 copies a `multihop` prompt.
- The old arm below drops those 28.

## 3. Data sources and roles (`sources.json`)

Roles follow P3/P10.1. Licenses were checked against primary sources on
2026-10-21 (a desk check, not legal advice).

| role | sources (status) |
| --- | --- |
| **primary reasoning transfer** | `s0_generators_v2_calsplit` (recommended); `stage2_generators_v1` (planned: new generators disjoint from eval-set-1); `reasoning_gym` (Apache-2.0; candidate, needs a dependency decision and exclusions: `arc_agi` loads ARC-AGI-1 *evaluation*, `gsm_symbolic` derives from GSM8K-test, `acre` is GPL-3.0); RE-ARC (deferred, R7) |
| **supporting capability** | GSM8K train and MATH train (MIT, human gold; recommended); corpus_v2's span-verified rag and multihop targets (CC-BY-SA-4.0; reused with no new generation); MBPP and CodeContests train (planned: need a sandboxed test verifier); NuminaMath-1.5 (candidate, restricted to human-authored subsets) |
| **evaluation only** | tool calling (xLAM is "research purposes only" and DeepSeek-V2-generated; glaive's provenance is undocumented); refusal/answerability (outside the primary scope; the old corpus has none either); every benchmark test split |
| **out of scope / excluded** | non-commercial or unlicensed sets: Enigmata-Data, K&K HF data, CLUTRR, LogiQA 2.0, KodCode, ZebraLogic, ProofWriter (unverified), FOLIO (license conflict) |

**Family breadth is the main data risk.**
- The only primary source with project verifiers today has 35 calibration
  families. At 5.5M tokens that is about 40 accepted units per rule.
- `stage2_generators_v1` (new families, disjoint generators) or a reviewed
  reasoning-gym subset is the way to add rules without exposing eval formats.
- Both are maintainer decisions.

## 4. The target quality contract (`aadistill.data.target_quality`)

Every candidate keeps every category, accepted or not:

- `naturally_terminated`, `length_limited` and `degenerated`, mapped from the
  engine's own stop reasons (an unmapped reason is an error);
- `protocol_valid`;
- `answer`: `correct` / `incorrect` / `unverifiable`, from a task-specific
  verifier;
- `trace`: `valid` / `invalid` / `not_assessed`;
- `accepted_for_training`, set by the group's `AcceptancePolicy`.

Rules:
- **Primary and supporting groups require `correct`.** `unverifiable` passes
  only through an explicit switch that no group sets.
- **A correct answer is not a verified trace.** Episodes check that every
  action was legal; static traces are `not_assessed` and reported as such.
- **Selection** is the lowest accepted candidate index, never by length.
- **The data-construction limit** is `block_len − prompt − 8`. A trace that
  cannot fit a training block is `length_limited` and excluded, never trimmed
  (P17).
- **This limit selects against the longest traces.** The excluded share is
  reported per stratum and difficulty.

Applied to the history:
- `teacher_cal` accepts 571 of 672. The 4 episodes with an illegal action are
  rejected; base_pool's verified filter had accepted one of them (`d0fe3662…`,
  success with one illegal action).
- corpus_v2 per type is in `asset_audit.json :: corpus_v2.under_target_quality_contract`.

## 5. Isolation (`isolation_rules.json`)

| rule | requirement | on violation |
| --- | --- | --- |
| R1 | no prompt/content key shared with a held-out unit (`shared.eval_sets.require_disjoint`) | refuse |
| R2 | 13-gram containment < 0.5 against each held-out unit's prompts; an n-gram found in more than 5 of a unit's prompts counts as template and is dropped (`ngram_containment`) | refuse |
| R3 | no rule family of any fluid evaluation item | refuse |
| R4 | no evaluation-only generator, and no family of another generator whose task type is one of theirs | refuse |
| R5 | a shared generator only with calibration-split families, v2 roster, seeds in [3,000,000, 4,000,000) | refuse (historical pools reported) |
| R6 | only splits the catalogue declares trainable | refuse |
| R7 | no ARC-AGI evaluation material; ARC training derivatives only by separate decision | refuse |
| R8 | in-sample, Stage-0 calibration and historical-corpus overlap | report |

**Exposure is reported, not hidden.** Training on calibration-split families of
the 10 shared generators turns their held-out families into an unseen-RULE,
seen-FORMAT measurement. The evaluation therefore always splits the fluid set
into two tiers:

| tier | static | episodes | what it measures |
| --- | --- | --- | --- |
| the 5 evaluation-only generators | 190 | 48 | unseen format |
| the 10 shared generators' held-out families | 380 | 96 | unseen rule, seen format |

**The task pool.** `train_tasks_v1` has 2,517 tasks: 2× the demand derived per
axis from the spec and the measured yields, across 35 families. It passes R1–R6
and repeats no Stage-0 or eval instance. 27 draws were skipped and counted: 25
repeat a Stage-0 or eval instance (a small instance space repeats across
seeds), and 2 raise in a frozen generator. No generator was edited.

## 6. Comparison 1: 2.96M old vs 2.96M new (`comparison_design.json`)

**A treatment comparison of two complete data schemes**, not a causal estimate
of verification, mixture or packing alone.

- **OLD.** corpus_v2 minus the 28 R2 sessions, under its historical uniform
  six-way token mix, cut at 2.96M unique supervised tokens by the shared
  builder. That is 3,151 units and 1,460 blocks at packing efficiency 0.355.
- **NEW.** `D_2.96M` of `stage2.reasoning_master@v1`.

**Fixed for both arms:**
- the initialization (§8);
- teacher, tokenizer and chat template;
- geometry;
- recipe `e1_p1_kd_heavy` (CE 0.25 + KD 1.0, `kd_scope: all`, AdamW 5e-5,
  warm-up 5% + cosine);
- 3 epochs;
- seeds 20260726 and 20260727;
- the eval-set-1 path.

**Compute contract M1 (recommended).** Without it, the frozen config would give
OLD 4,380 block passes against about 1,250 for NEW: 3.5× the compute and 3.5×
the optimizer steps, almost all of it padding. M1 holds:
- equal supervised tokens, epochs and optimizer steps (per-arm blocks per step
  is derived);
- `truncate_padding` on in both arms. That is mathematically identical
  (padding is outside every loss mask) but not bitwise identical, and it is
  declared as an execution deviation;
- every executed-position and FLOPs quantity reported. OLD still executes about
  1.3× NEW's positions, because its prompts are context.

M0 (the frozen config byte for byte) is the alternative: a NEW win under M0 is
conservative, while a null result is uninterpretable.

**Sequential gate G1.** NEW at seed 20260726 runs first. Below 2% static it
stops, because the next study is the recovery objective, not the comparison.

**Endpoints and statistics.**
- **Primary:** formal fluid static correctness, item-paired, pooled over seeds.
- **Secondary:** axes, episodes and the two generator tiers.
- **Supporting capability units** each separately, plus guard rails
  (answerability, safety, behaviour, usable rollout and its components).
- **Diagnostics only:** teacher-forced metrics, CE/KD, reasoning length.
- **Statistics:** SESOI 0.05 pooled / 0.08 per axis, with a rule-family cluster
  bootstrap (the Stage-0 scale).
- **Labels:** GO requires the pooled interval wholly above +SESOI *and* each
  seed's point estimate above it. The other labels are NO-GO, REVERSED and
  INCONCLUSIVE, with a FLOOR flag.

Lower KL, lower loss or more usable rollouts are never reported as reasoning
transfer.

## 7. Study 2: the nested ladder (`ladder_plan.json`)

- **Rungs.** 0.86M ⊂ 1.60M ⊂ 2.96M ⊂ 5.50M, all prefixes of one master order
  (`aadistill.data.nested_ladder`).
  - Strata are interleaved by largest remainder, families are round-robin
    inside a stratum, and the validation tail is reserved outside every rung.
  - Each increment is packed on its own, so rung k is `blocks[:n_k]` for the
    unchanged trainer reader.
- **Held constant.** The mixture is checked per rung (`max_share_deviation`).
- **Measured per rung.** Supervised, reasoning and answer tokens, forward
  tokens, padding, exposure, FLOPs, steps and epochs.
- **Training.** Every rung trains independently from the same initialization.
  Epochs are fixed, so compute grows with the rung, as in E1. A fixed-compute
  ladder is a separate protocol.
- **Reuse.** The 2.96M rung's units are the comparison's NEW arm.
- **When.** The ladder runs only after a GO.

The dry run shows the builder on real material:
- teacher-native fluid traces pack at 0.97 efficiency, with 0.91 supervised
  tokens per forward position (the historical ladder: 0.25 or less);
- the trainer's own `ladder_blocks` reads the built pack (round trip in
  `packing_dry_run.json`).

## 8. The fixed initialization (`initialization_selection.json`)

**Proposed: Stage-0 d4 rank 1** — state `1301deb7…`, checkpoint `fc28e0dd…`,
relay `stage0_calvol/finalists/formal_v1/d4/1301deb7…/model.safetensors`,
PROTECTED.

- **The rule.** The candidate that Pareto-dominates all 12 on the searches'
  three minimized step-0 objectives (equal-domain teacher KL 6.76, worst-domain
  7.38, critical-token KL 11.27) under the one shared `state_eval@v1` suite.
- **It is the only point on the front.** It also has the lowest pooled step-0
  NLL (7.11; teacher 0.51).
- **No recovered outcome is read.** Over 24 students, recovered usability
  follows the seed. d4 r1's own two units (1/570 and 0/570 static) are shown
  beside the selection, never used by it.
- **Materialization identity.** All 12 candidates were materialized under the
  forward profile's fingerprint `e627f461…` with `scored_v1`. The baseline
  record's empty `forward_materializations` is the open inconsistency #1 of the
  handoff, left for the maintainer.
- **What stays unclaimed.** That lower step-0 KL recovers better reasoning is
  not established; this programme tests it.

## 9. Cost (`cost_forecast.json`, measured rates only)

**Measured rates:**
- teacher generation 1,739 tok/s (A100-80GB, vLLM 0.27.1);
- recovery 1.48–1.54 s (A100) and 1.78 s (L40S) per 8,192-position block pass;
- eval 3,343 s median per L40S unit for students at the floor, budgeted at 2×
  for students that reason at length;
- L40S $1.09/h and A100 $1.79/h, with disk +2.5%.

| phase | expected | hard cap |
| --- | --- | --- |
| P1 generation, 15.8M generated tokens (one A100 session) | $7.22 | $10.83 |
| P2a gate G1: NEW seed A | $3.08 | $4.61 |
| P2b: OLD × 2 + NEW seed B | $9.38 | $14.06 |
| P3 ladder: 3 rungs × 2 seeds | $17.13 | $25.69 |
| storage: 50 GB volume for about two months | $7.00 | $7.00 |
| **through P2** | **$19.68** | **$29.50** |
| **all phases** | **$43.81** | **$62.19** |

CPU work (every builder, audit, isolation check and pack) runs on the dev box
at $0.

## 10. Durability (`durability_plan.json`)

**Measured at $0 (2026-10-21T17:52Z, nothing uploaded).** The HF relay accepts
at most one ~2.0 GiB object (2,147,587,286 bytes). It holds one bf16 student
and refuses a single 2.38 GB fp32 student directory. The programme would
produce 10 students, 22.2 GiB. The dev box has about 12 GiB free.

**Recommendation:**
- a 50 GB network volume for the programme's lifetime;
- each student persisted at completion, before evaluation;
- the selected initialization copied once to the dev box, so it stops being a
  single relay copy;
- the capacity probe run at the true per-object size and total as a launch
  precondition.

The backend is the maintainer's choice.

## 11. Budget and authorization (`authorization_proposal.json`)

**Money.** Records: $625.5798 spent of the $643.6571 cap, so $18.0773 remains
on paper.
- The provider billed $32.5702 more than the records hold. Its attribution is
  **unverified and open for the maintainer**. If it is project spend, the cap
  is already exceeded by $14.49.
- The $101.23 provider balance is not a grant.

**Needed before any paid action:**
1. the gap attribution;
2. a new cap or envelope;
3. the durability backend;
4. the initialization;
5. the corpus scheme (shares, sources, training on shared-generator
   calibration families, the omitted tool/refusal data);
6. the compute contract;
7. the comparison design.

**Remaining $0 engineering** before the first paid run:
- a generation adapter over `train_tasks_v1` and the math sources;
- math task builders with R2 decontamination;
- a recovery driver that reads the master pack, applies M1 and persists each
  student;
- the launch-bound gates.

## 12. Core additions (generic; P3)

- `aadistill.data.nested_ladder`: stratified, family-round-robin nested
  ordering, rung cuts, composition capacity, a reserve and per-rung accounting.
- `aadistill.data.sessions.pack_best_fit`: best-fit-decreasing packing that
  never cuts a session.
- `aadistill.data.target_quality`: the candidate contract above.
- `aadistill.evaluation.isolation`: `ngrams`, `ngram_index` (with a
  template-frequency filter) and `ngram_containment`.

None names a dataset, model, stage, rung or eval set; the experiment layer
supplies all of them.
