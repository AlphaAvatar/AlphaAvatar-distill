# DMR1 variant 1: importance-merged initialization at SOB1's depth-only structure (DMR1-M)

Stage 1 owns this variant; its training is `stage3_recovery`. It is the first of the three DMR1 variants (the round's
review report: [`../reasoning_inheritance/DMR1_REVIEW.md`](../reasoning_inheritance/DMR1_REVIEW.md); variants 2 and
3: [`../depth_recurrence/README.md`](../depth_recurrence/README.md)).

**v1-new (section 7): PREPARED at $0 on 2026-11-28, READY_FOR_REVIEW; nothing granted, issued, built or run.**

**Status: EXECUTED 2026-11-27/28 and STOPPED by the maintainer during seed 2 (2026-11-28 UTC): the build COMPLETE, seed 20260726 trained (resumed once from save point 6,624) and evaluated, reported descriptively; seed 20260801 stopped at step 6,590 (durable to step 5,520); its resume was abandoned by the maintainer and that save point retired (2026-11-28, storage option (a)). Not the registered two-seed completion (review report section 00).** Branch
`research/reasoning-inheritance`, not merged into `main`.

| file | what it is |
| --- | --- |
| [`configs/stages/stage-1/depth_merge/design.json`](../../../../configs/stages/stage-1/depth_merge/design.json) | the design: base, spans, the alpha merge, statistics, reuse, recipe, evaluation, storage, cost basis |
| [`configs/stages/stage-1/depth_merge/depth_init_tensor_digests.json`](../../../../configs/stages/stage-1/depth_merge/depth_init_tensor_digests.json) | the base's 178 tensor digests (`merge.py digests`, from the dev box's copy) |
| [`logs/stages/stage-1/depth_merge/plans/`](../../../../logs/stages/stage-1/depth_merge/plans/) | the two sessions' proposals (`proposal.py`) |
| [`scripts/stages/stage-1/depth_merge/`](../../../../scripts/stages/stage-1/depth_merge/) | `merge.py` (the instance), `split.py` (v1-new's instance), `build_driver.py`, `train_driver.py`, `plan.py`, `session.py` + `launch.py`, `analysis.py`, `proposal.py`, `storage.py`, `tests/` |
| [`configs/stages/stage-1/depth_merge/split.json`](../../../../configs/stages/stage-1/depth_merge/split.json), [`split_design.json`](../../../../configs/stages/stage-1/depth_merge/split_design.json), [`split_config.json`](../../../../configs/stages/stage-1/depth_merge/split_config.json) | v1-new (section 7): the maintainer's facts, the design (V1's outside its declared keys, `split.py design`), the 18-layer config (`split.py config`) |

## 1. Question

At SOB1's depth-only structure, does writing each kept block from its whole span of teacher blocks by the alpha merge,
instead of keeping it as it is, leave more static fluid reasoning after the **same** Stage-3 recovery?

The direct comparison is **comparison 1** of the round: `depth_merge` (alpha-merged init + SOB1's depth recipe)
minus `depth_only` (SOB1's direct-deletion init + the same recipe), thinking mode primary. SOB1's two recovered depth
seeds and their sealed eval-set-1 sets are reused as the control; nothing of them is retrained or regenerated.

RI-LM1's negative result (both training-free merges worse than deletion, untrained, at RI-S1's 28-block structure) is
history: another structure, no recovery. It does not stand in for this comparison.

## 2. The base

* Teacher: the original hybrid `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` (not Thinking-2507), the
  root of every product here (one lineage, AGENTS.md P17.1).
* Base: SOB1's depth-only **training-pre** initialization, `init_record.json` sha256 `413686f2…`, weights
  `76c8ca19…`, config file `191c51bb…`, 2,003,851,776 parameters; mapping `36888440…`.
* Kept blocks (unchanged; no depth search is run again): `[0, 1, 4, 5, 6, 7, 10, 12, 18, 19, 22, 23, 24, 29, 31,
  34]`. Width 2,560, FFN 9,728, 32 / 8 heads, head_dim 128, embedding and tied head: the base's.

## 3. The alpha merge

For each kept block j with span S_j (j and the removed blocks after it, `layer_merge.spans_from_kept_blocks`):

    theta~_j = sum_{i in S_j} alpha_{j,i} A_{j<-i}(theta_i),    alpha_{j,i} = I_{j,i} / sum_{k in S_j} I_{j,k}

per parameter slot, exactly RI-LM1's M-act at this structure's spans:

| | |
| --- | --- |
| merged spans (9) | `[1,2,3]`, `[7,8,9]`, `[10,11]`, `[12..17]`, `[19,20,21]`, `[24..28]`, `[29,30]`, `[31,32,33]`, `[34,35]` |
| single spans (7, unchanged bit for bit) | blocks 0, 4, 5, 6, 18, 22, 23 |
| FFN slot (gate row, up row, down column) | `I = E|a| x ||down column||_2` |
| query-head slot (q rows, o_proj columns) | `I = mean_t ||W_o,h a_h(t)||^2` (write energy) |
| KV-group slot (k / v rows) | the summed energy of its query heads |
| q_norm / k_norm, folded gains | each block's share of the span's head energy |
| alignment A | FFN neurons by exact assignment on the gated-product Pearson correlation; KV groups on the groups' write cosine; query heads within a matched group on their write cosine |
| RMSNorm | input / post-attention gains folded into their reading linears, the merged gains 1 |
| all-zero slot | uniform `1 / |S_j|`, counted |

* Each merged block is executed **once**: 16 blocks, 16 logical layers, the base's structure and config file.
* alpha builds the initial weights only; recovery then trains every trainable parameter as in SOB1's recipe.
* Not registered: the beta addition of original weights, learnable merge coefficients, any fitting or extra training.
* Convex per-slot weights guarantee neither the merged block's output scale nor that it reproduces the teacher span's
  cumulative update; both are measured as diagnostics, never assumed.

**Statistics.** One teacher forward over `calib.hybrid_d4_base@v2` at `ffn.activation_importance_v0`'s positions under
the packed search's session (inputs v3): 558 examples, 79 packed sequences, 231,242 statistics positions (the counts
RI-LM1 measured under the same session; the build stops on any other). 29 blocks (9 kept + 20 absorbed), 20 pairs:
15.1 GB of float64 Pearson moments beside the 8 GB teacher.

**Reuse.**

* Reused: the method's code by identity (RI-LM1's `statistics_code()` / `candidate_code()` ast digests), the
  calibration, positions and packing, SOB1's selection and base (rebuilt from the teacher and checked against its
  178 tensor digests), SOB1's recipe, plan and control sets.
* Recomputed: every statistic, permutation and merged tensor at this structure.
* Not reused: RI-LM1's statistics and deltas (other spans; alpha normalizes over S_j). Five of its pairs ((10,11),
  (24,25), (24,26), (29,30), (31,32)) are pairs here too and can be compared as a reproduction diagnostic only.

## 4. Recovery and evaluation

* **Recipe**: SOB1's depth recipe, compared field by field by `plan.py` against SOB1's committed depth plan
  (`training_plan_depth_only.json`, sha256 `4eeab53f…`): only the unit and arm names, run name, student path, output
  directory and `lineage.initialization` change. A-MIX-6cat-sr pack and proportions (45,363 examples, 11,862,564 unique
  supervised tokens before packing), random order, sequential packing with truncation, 8,192 tokens, standard causal
  attention, continuous position ids; micro-batch 1 x 2 accumulation steps; 9,701 optimizer steps; the original teacher
  online in bf16, CE 0.25 + Top-200 + tail KD 1.0 at the assistant positions; AdamW lr 5e-5, cosine to 0.1, 485
  warmup; fp32 weights, bf16 autocast, gradient checkpointing; tied embedding frozen; seeds 20260726, 20260801.
* **Evaluation**: SOB1's one protocol, verbatim (eval-set-1, every item, both modes, `nucleus_v1`, 8,192 tokens,
  action protocol v2, the teacher's tokenizer sidecars). eval-set-1 is development-exposed. The untrained merged init
  is not evaluated on eval-set-1 (SOB1: every untrained arm 0/570); its state evaluation is the diagnostic.
* **Analysis**: comparison C1 (`scripts/stages/stage-1/depth_merge/analysis.py :: C1`; registered 2026-11-26 in
  the recurrence design, kept there under `superseded_2026_11_28`; review report section 6).

## 5. Execution chain

| step | where | what |
| --- | --- | --- |
| build | `dmr1_merge_build`, `build_driver.py` | contract and the products' scientific identity; base check; statistics (each of the 20 alignments persisted with the identity of every input it read) -> relay at once; checks, candidate, save / reload -> relay at once; diagnostics (each candidate block on the teacher's device); `init_record.json` COMPLETE only when both copies verify AND are handed off, each product under this identity |
| retain / bind | dev box, $0 | commit the init record; `plan.py --arm depth_merge --init-record <it> --write` |
| train + evaluate | `dmr1_merge_train`, `train_driver.py` | SOB1's state machine bound to this design: both seeds, finals durable, four sets complete against them |
| closeout | dev box | `session.completion`: the init record (build) / finals + four sets + placement (training) |
| analysis | dev box, $0 | `analysis.py comparison --id C1 --a-evidence <secured out> --write` |

A replacement attempt takes over what exists, product by product. Each product carries its own **scientific
identity** (`build_driver.stamp`), written when it is made or verified for reuse and never copied onto another product.
The identity (`build_driver.identity`) covers the teacher, base, mapping and spans, the calibration with its packing and
positions, and the statistics and candidate code by ast digest -- never the design file's bytes, so a changed
description recomputes nothing. What a takeover reuses:

* **the init record:** a durable one is never rebuilt;
* **the weights:** weights whose relay copy verified are never rebuilt or re-sent (relay.json, the relay's own listing,
  or the record beside the files when the pod was lost before the dev box received the latest build record). They are
  restored from that copy, re-hashed, only when the diagnostics still need them;
* **the handoff:** a copy whose handoff did not verify keeps its files; only the handoff is retried, and no init record
  is written until it is done;
* **the statistics:** if they survive under their identity they are never recollected. When only the checkpoint's
  identity changed (the base, the method, its seed or the candidate code), just the weights are rebuilt from them; the
  old weights are set aside under their own identity;
* **the alignment pairs:** a persisted pair is reused only when the FFN correlation, the attention Grams, the head
  geometry and the alignment code are all the same.

A build record whose statistics were made under another identity is set aside, never reused. A final that exists
anywhere is restored, never retrained; save points resume the optimizer, schedule, RNG and data position; sealed
evaluation parts are reused and only missing items are generated.

```text
launch.py budget --kind dmr1_merge_build
launch.py grant  --kind dmr1_merge_build --decision '<the maintainer's words>' --granted-by maintainer
launch.py issue  --kind dmr1_merge_build --run-id <id> --expected-usd 2.50 --hard-cap-usd 5.43
git commit <the authorization> ; stage the bundle at that commit
launch.py launch --kind dmr1_merge_build --run-id <id> --session-commit <c> --bundle <b> --scr <s> --max-price 1.11 --disk-gb 150
# after the build is COMPLETE and its init record committed:
plan.py --arm depth_merge --init-record <the committed init_record.json> --write
launch.py grant  --kind dmr1_merge_train --decision '...' --granted-by maintainer
launch.py issue  --kind dmr1_merge_train --run-id <id> --expected-usd 32.68 --hard-cap-usd 64.46
launch.py launch --kind dmr1_merge_train --run-id <id> --session-commit <c> --bundle <b> --scr <s> --max-price 1.11 --disk-gb 150
```

(All commands run as `PYTHONPATH=src:scripts:scripts/stages/stage-1 .venv/bin/python scripts/stages/stage-1/depth_merge/<file>`.)

## 6. Storage and money

See the review report, sections 0.3, 9 and 10, and `logs/stages/stage-1/depth_merge/plans/storage_plan.json`
(`storage.py --write`).

* **A first build session's launch gate** asks the relay for the WHOLE chain, 50.38 GiB plus a 1 GiB margin, not for
  the build's products alone. The chain is the init and statistics, the closeout archives and the training session's
  rotation peak (seed 1's final held while seed 2 rotates two save points).
* **A takeover** asks only for what the chain still adds: less each product it holds on the relay under this identity
  (`session.build_need`).
* **Live (2026-11-26 18:19Z):** after SOB1's two FFN + attention finals moved to the dev box (maintainer, storage
  option a), the relay accepts 58.13 GiB, which is 6.75 GiB to spare. The dev box has 7.00 GiB free for at most
  1.5 GiB of one session's evidence.
* **Money:** the two proposals total $71.89 with reserves, inside the $74.79 the project cap leaves.

## 7. v1-new: long removed runs merged into a new layer (arm `depth_merge_split`)

Registered on 2026-11-28 beside V1, which keeps its design, results and identity. **PREPARED at $0,
READY_FOR_REVIEW: nothing granted, issued, built or run.** The review report's section 0W holds the order, the storage
and the money.

**The rule** (`layer_merge.groups_from_kept_blocks(36, kept, separate_run_min=3)`; the threshold counts removed layers
only). It starts from SOB1's committed depth selection, which is not searched again. For each kept layer and the run
of removed layers after it:

* **no removed layer:** the kept layer, as it is;
* **one or two removed layers:** merged into the kept layer, exactly as V1 merged them;
* **three or more removed layers:** the kept layer keeps its original weights, and the run alone becomes a NEW layer
  right after it.

At this selection two runs split off: 1-based, layer 13 is kept with a new layer from 14-18, and layer 25 is kept with
a new layer from 26-29. That gives 18 layers and 2,205,713,408 parameters, counted by instantiating
`split_config.json` on the meta device. V1 and D have 16 layers and 2,003,851,776 parameters.

| groups (1-based teacher layers) | kind | how the layer starts |
| --- | --- | --- |
| 1, 5, 6, 7, 13, 19, 23, 24, 25 | kept (9) | the base's layer (SOB1's depth init, rebuilt from the teacher, checked against its digests) |
| 2-4, 8-10, 11-12, 20-22, 30-31, 32-34, 35-36 | absorbed (7) | V1's built layer, byte for byte, from V1's init (`f57131792b50...`) |
| 14-18, 26-29 | separate (2) | M-act over the run's blocks only: alpha = I / sum I within the run, the run's first block the anchor |

A separate layer is never V1's span block with the kept layer put back in front of it. Every layer runs once: there
is no recurrence and no hidden-state loss. "The preceding layer unchanged" applies at initialization only. Recovery
is SOB1's depth recipe (Stage 3) with SOB1's trainable patterns: every block's attention, MLP and layer norms and the
final norm train, the embedding / tied head stays frozen, and no further freezing is added. One seed (20260726), one
final, both evaluation modes.

**Reused by identity, never recomputed:**

* SOB1's selection and the base digests;
* V1's calibration (the same profile, positions and session);
* V1's init, for the seven absorbed layers: fetched from its relay copy and re-hashed against V1's plan;
* SOB1's depth recovery recipe and data, and eval-set-1's protocol;
* the controls D (SOB1 `depth_only` s20260726) and M (V1 `depth_merge` s20260726), with their registered finals
  and sets.

**New:**

* the statistics of the separate runs' 9 blocks and their 7 alignments (13->14..17, 25->26..28; V1's pairs are
  anchored at 12 and 24, so they do not apply);
* the 18-layer init (4.11 GiB bf16);
* the training's save points (21.75 GiB each, rotated two at a time);
* the final (8.22 GiB fp32), the evaluation sets and the closeout archives.

**Chain** (the same state machine as V1's, bound to the arm):

* the build is `build_driver.py --arm depth_merge_split`, kind `dmr1_split_build`. Its identity covers the groups,
  the derived config, `split.py`'s candidate code and the V1 init it reuses. Takeover and completion are V1's, per
  arm;
* then `plan.py --arm depth_merge_split --init-record <it> --write`: one unit, seed 20260726, SOB1's depth unit
  outside the declared fields;
* then training, kind `dmr1_split_train`;
* then `analysis.py split --evidence <secured out>`: X1 = v1-new - D and X2 = v1-new - M, descriptive, with the
  percentile 95% group-bootstrap interval and no test.

**One seed per condition** (AGENTS.md 4.3): the full recovery and evaluation, then a report and a wait for review.
There is no automatic second seed and no promotion threshold.

**Memory:** about 42.4 GiB allocated at peak (V1's measured 39.24 GiB, plus about 3.0 GiB of fp32 weights, gradients
and AdamW state for two more trainable blocks, plus 0.16 GiB of activations). That is within about 1.5 GiB of the
L40S's usable memory: **HIGH out-of-memory risk**. The first steps' memory probe measures it. An out-of-memory error
is answered first by protocol-neutral steps; an 80 GB card needs a measured OOM and a decision of its own.

