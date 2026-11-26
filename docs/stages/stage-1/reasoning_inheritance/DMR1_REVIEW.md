# DMR1: depth recovery at SOB1's depth-only structure -- review report (READY_FOR_REVIEW)

**Terminal state of this round: READY_FOR_REVIEW** (2026-11-26 UTC, after the repair round of the maintainer's second
brief). The designs, the code of all three variants, this round's repairs, the principle audit, the $0 verification and
this report are committed and pushed on `research/reasoning-inheritance` (not merged into `main`). **This is not a
scientific COMPLETE:** no initialization was built, nothing trained or generated, no pod or volume created, no grant
written, no artifact issued. Variant 1 starts only after this review, the maintainer's explicit authorization and the
relay-room decision of section 0.4; variants 2 and 3 have no paid authorization.

Contents: **0 this round's repairs, principle audit, conditions per variant, storage, money, state** · 1 scope and entry
points · 2 baseline, teacher, mapping, schedule · 3 variant 1 · 4 variant 2's initialization and binding · 5 variant
3's hidden-state term · 6 comparisons and statistics · 7 the recurrent mechanism · 8 tests, fixes, what is not measured
· 9 time, memory, storage, money · 10 $0 state and authorization boundaries.

## 0. Repair round (2026-11-26, second brief)

The first round's preparation was reviewed and this round repairs what the review found, audits the repository
against the project's principles and settles variant 1's storage arrangement. Still $0: no grant, no build, no
training, no generation, no paid resource. Section 0 is the round's report; sections 1-10 are the design, updated
where this round changed them.

### 0.1 What was fixed, where, and the evidence

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
| 10 | **V1's storage proved only for the build** (the report also omitted the new init's 3.73 GiB) | `depth_merge/storage.py`: the relay's net change over the whole chain in the drivers' order (build products, build closeout archive, the training rotation from an empty state, closeout archive) against the live headroom; `session.relay_room` for the build kind asks the relay for the WHOLE chain plus a 1 GiB margin, so no init is built for a training session that would not fit; `logs/stages/stage-1/depth_merge/plans/storage_plan.json` | `storage.py --write` against the live relay (0.4) |
| 11 | **Code bundles kept for no consumer** | `scripts/maintenance/consolidation/retire_relay_bundles.py`: a session bundle is retired only when its commit is exactly one commit on a pushed ref (rebuildable), no pod exists, and it is not under a protected prefix; tombstone first, exact-object permanent deletion, listing read back | 379 bundles retired, 8,418,278,418 B (7.84 GiB) freed, 0 left (`logs/maintenance/inventories/relay_bundle_tombstones_20261126.json`, written before the deletion); 20 bundles (0.07 GiB) whose commit is not on a pushed ref are KEPT; the headroom read back 43.09 GiB (it was 35.27) |
| 12 | README / AGENTS | README's goal now says the other capabilities are still measured separately and names the two environments (data generation and training); its student sizes include the ~2.0B rounds. AGENTS P8.3: the full-suite counts are upper bounds, never a quota | -- |

### 0.2 Principle audit

Checked = run (a test or a live read), static (read in the code or records), or not implemented.

| principle | where it lives | check | result, gap or fix |
| --- | --- | --- | --- |
| 1 GPU validation where CPU cannot answer | P8.2 / P8.3; this report 0.3 | static | no CPU stand-in was built for a CUDA question this round. V1's real GPU paths (statistics, 20 assignments, the span diagnostic on the device, the reload) run in the formal build session itself. V2 / V3's engine and memory validation (~1 h, ~$1.2, cap $2.5: the recurrent model in vLLM 0.27.1 vs the HF reference, prefill / decode, the per-logical-layer KV cache, the first training steps' memory with and without the hidden term, the two terms' gradient norms at step 0) is a condition of V2 / V3 only, unauthorized |
| 2, 4 scalability; no instance hardcode in the core | `src/aadistill`; adapters; `configs/` | run + static | the two confirmed hardcodes fixed (0.1 #8, #9). A grep of `src/aadistill` for teacher names, dimensions, seeds, experiment paths, device names and provider ids finds only docstrings / comments, the RunPod provider adapter (an infrastructure adapter, like a model-family adapter) and `cpu_test_env.host_local_store`'s default store name (`aad-artifacts`, a local-store name; noted, not changed). Stage 0-6, other families, geometries and compression ratios enter through configs and adapters (the recurrent family is an adapter; schedule, pairs and weights are config). **Gap left open, deliberately:** the other application search builders (`packed_full_search`, `full_search`, the D1 / D2 sessions, the calibration-volume session, `dry_run_search`, the device canary) and the forward baseline v2 record declare no stream weighting; their searches still derive the hashes their records carry, but one that APPLIES width or the composite now stops loudly until it declares one. The next forward run declares it in its own protocol (or a baseline v3); editing the registered v2 record or the closed builders here would move recorded identities for no consumer |
| 3 logs by stage / experiment / run | `logs/stages/stage-1/depth_merge/`, `logs/stages/stage-6/depth_recurrence/`, generated READMEs and `logs/stages/index.json` | run | regenerated after this round's new records (the storage plan, the bundle tombstones); the historical layout is untouched |
| 5 test economy | P8.3 | static | mid-round checks only for specific questions (the lifetime regression against the old code; the counter-example; the operators' callers). One convergence validation and one full core suite after the implementation (0.6). The P8.3 bound is now stated as an upper bound, not a duty |
| 6, 9 autonomous completion | P8.2.1, P12.1 | run | every repair above was made and verified in this round; the one thing outside the authority is the relay room (0.4), listed as a decision with its objects and sizes |
| 7 historical weight retirement | AGENTS 2.5; `retire_eligible_checkpoints.py` | run | the standing-rule dry run finds **0** eligible rows (two refused: their bytes are already gone, or code still names them). Every remaining relay checkpoint is protected by an explicit maintainer clause (SOB1's finals at their approved placement; SOB1's three inits "kept until the next round's plan is fixed"; the permanent controls; the canonical 0.6B init) or is DMR1's own base. The 379 session code bundles (7.84 GiB) had no consumer and were retired (0.1 #11) |
| 8 review priorities | P8.3 | static | this report leads with validity (the selection's source, the conditioning of C2, the identity binding, the lifetime bug), then reuse and identity, then storage and money |
| 10, 11 keep new products; reuse across attempts | P8.2.1, P8.4 | run | the build persists each product at once and takes over by identity (0.1 #1-#3); training reuses SOB1's chain (finals restored, never retrained; sealed generation parts reused; save points resume optimizer, schedule, RNG and data position: the end-to-end driver tests of the first round) |
| 12 capability target | P10.2, P10.3; `data/eval-set/` | static | fluid reasoning is the primary target; the primary comparison reads eval-set-1's static fluid formal core; every other unit is reported on its own. eval-set-1 is development-exposed (its own manifest); `stage3_legacy_150` is in-sample training prompts and is never reported as generalization; nothing here is a never-seen result |
| 13 terminology | AGENTS 2.10 | static | packed sequence, sequence length, batch size (2 sequences per optimizer step), micro-batch size (1), gradient accumulation (2 steps), epoch; a **logical layer** is one execution of a Transformer block at one teacher-layer position (defined once, stage-6 README section 2); a **recurrent iteration** is one more call of the same block; a Transformer block keeps its meaning |
| 14 one teacher lineage | P17.1; `design.json :: teacher, lineage`; the plans' `lineage` | static | hybrid `Qwen/Qwen3-4B@1cfa9a72` for calibration, statistics, initialization, SOB1's training targets, the online KD teacher, tokenizer, chat template and config; the build checks the loaded teacher's sha256; a save point is restored only under the same trainer config sha256 (which binds teacher and data) |
| 15 packing and token counts | P19; the designs' `statistics`, `training.data` | static | V1's statistics forward: the packed search's context (seeded sha256 shuffle, sequential packing, truncation, continuous positions, causal, 8,192). Training: SOB1's pack (the same rule; 6,467 packed sequences x 8,192). Volumes counted before packing (558 calibration examples; 45,363 training examples, 56,408 turns, 11.86 M unique supervised tokens); 79 packed sequences / 231,242 positions and the per-step KD rows are packing and exposure measures only. V2 / V3 (Stage 6) register Stage 3's packing as their own identity. **Stage 5: no run exists -- not executed, not validated** |
| environments | `requirements-cu128.txt` → `/opt/train` (relay wheelhouse `wheelhouse_cu128_cp312`), `requirements-vllm.txt` → `/opt/vllm` (`wheelhouse_vllm_cp312`); `autoinit_preflight_setup.sh` | static | data generation and evaluation generation run in the vLLM environment, training in the training environment (`train_driver --vllm-python /opt/vllm/bin/python` hands evaluation to the other one); the online KD teacher's forward is part of the training computation, in the trainer's process. Both wheelhouses are the pods' environment and are kept |
| "never seen" | `data/eval-set/eval-set-1/manifest.json`; `shared.eval_sets` | static | unseen ITEMS: training, calibration and search data are checked disjoint from held-out units (`require_disjoint`, `held_out_paths`); unseen RULE FAMILIES: fluid_reasoning_720's generator families are evaluation-split and calibration builds are checked disjoint by family, content and prompt; unseen INTERACTIVE ENVIRONMENTS: the `interactive_adaptation` episodes (formal core; descriptive here); DEVELOPMENT EXPOSURE: eval-set-1's items and per-item results were read during development, so a claim of untouched-test generalization needs eval-set-2 (not built; out of scope) |

### 0.3 Which conditions belong to which variant

**Variant 1 (before its build session is launched):** this review; the maintainer's grant for each of its two kinds;
**the relay room for the whole chain** (0.4) -- the build's launch gate refuses until it exists. No separate GPU
validation: the build session's own run is the GPU check of its paths. **Not** conditions of variant 1: the vLLM
recurrent engine, the hidden-loss memory, the lambda question, anything of comparison 1's outcome.

**Variants 2 and 3:** comparison 1 complete and valid (the selection refuses otherwise); the selection written once and
its recurrent init record committed; the GPU engine validation above; a project-cap decision (their caps exceed what
the cap leaves after variant 1); a proposal and a grant per kind; their own relay room (their rotation is the same
46.46 GiB peak; after variant 1 its init, two finals and archives hold 18.97 GiB more); R1 / R2 answered (section 5).

### 0.4 Storage, on the real objects and in the real order

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
exactly this (0.1 #10), so it refuses now; the training session's gate asks for its remaining rotation. The dev box
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

**The decision the maintainer owes before variant 1 launches** (any one that makes 8.28 GiB):

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

### 0.5 Budget and authorization (live, read-only, 2026-11-26 15:07Z)

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

### 0.6 Commit, push, resources

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
| 2, DMR1-R | `depth_loop` | Stage 6 (`stage6_recurrent_depth`) | [`configs/stages/stage-6/depth_recurrence/design.json`](../../../../configs/stages/stage-6/depth_recurrence/design.json) | 16 shared blocks on 36 logical layers vs single execution, same init |
| 3, DMR1-RH | `depth_loop_hidden` | Stage 6 | the same | variant 2 + per-iteration hidden-state loss |

Design documents: [`docs/stages/stage-1/depth_merge/README.md`](../depth_merge/README.md),
[`docs/stages/stage-6/depth_recurrence/README.md`](../../stage-6/depth_recurrence/README.md).

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
83.2 at layers 14-17, the student's state about 9x the teacher's norm, decaying through the later blocks (74.5 at 18,
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
validated)? R3: the relay-room decision (section 0.4).

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

**This round's evidence** (the repair round; one convergence validation after the repairs, then one full core suite;
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
                                           teacher_modes); their search builders declare no weighting (0.2, 2/4)
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

**Fixed during the first round** (this round's repairs: section 0.1):

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
| a training session's relay peak | **46.46 GiB** (two save points + one final); variant 1's chain adds its init and archives: **50.38 GiB** (0.4) | -- |

**The relay.** Superseded by section 0.4, which counts the new init (3.73 GiB) the first version of this section left
out, follows the drivers' real order, and asks the relay for the whole chain: 50.38 GiB at variant 1's peak (+1 GiB
margin) against 43.09 GiB of live headroom after this round retired the unconsumed code bundles -- **8.28 GiB short, a
maintainer decision** (0.4 lists the objects and the options). The one-save-point rotation the first version offered
as an option is withdrawn: it would delete the only valid save point before the next one is uploaded.

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

* Nothing billed in either round: no pod, no volume, no grant, no artifact, no build, no training, no generation.
  Read-only queries: the provider position, the relay's headroom and inventory, the vLLM 0.27.1 wheel (its source) and
  the public teacher weights (the step-0 diagnostic; deleted afterwards).
* **One relay deletion, this round, under the brief's rule for code packages**: 379 session code bundles (7.84 GiB),
  each rebuildable from a commit on a pushed ref, none read by anything (0 pods); tombstone
  `logs/maintenance/inventories/relay_bundle_tombstones_20261126.json` written before the deletion; 20 bundles whose
  commit is not on a pushed ref kept. No checkpoint, log or evidence was touched: the standing-rule dry run found none
  eligible.
* Retention: SOB1's depth init and its two finals are DMR1's declared consumers (`build_checkpoint_registry.py`); the
  three SOB1 initializations and every SOB1 final stay where their maintainer clauses keep them.
* Owed to the maintainer: this review; **the relay-room decision (0.4)**; variant 1's two grants. For variants 2 and 3: a
  project-cap decision, their proposals and grants after the selection, the GPU engine validation, their own relay room.
* After the review, the room decision and the maintainer's authorization, variant 1 starts with `launch.py grant
  --kind dmr1_merge_build ...` (its design document, section 5); the build's launch gate re-reads the relay for the
  whole chain before anything is created.
