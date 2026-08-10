# Experiment 8 — preregistration

**Status: DRAFT. Not authorized. No GPU has been used for any of it.** The
calibration set, the search implementation, the configs, the gates and the tests
exist and are verified on CPU; the Stage 0 statistics are regenerating on the dev
box at $0. Authorization is a separate decision and requires a cumulative-cap
increment (§11).

**Question.** Does the current position-based depth compression discard teacher
blocks that are disproportionately important to the teacher's predictive
function, and does a contribution-guided layer map produce a better student under
the same downstream training recipe?

---

## 1. What changes, and what must not

The canonical Stage 1 depth map is positional. For 36 → 28 it keeps

```
0 1 2 3 4 6 8 10 12 14 16 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
```

and drops `{5, 7, 9, 11, 13, 15, 17, 19}` — every odd layer of a middle band. That
band's *position* was chosen by a single-axis ablation (2026-07-14: merging the
early band collapsed the model to holdout NLL 10.48, the middle band kept it at
3.88), but which individual blocks die inside it was never measured at all. They
are the odd ones because the merge steps by two.

E8 replaces that rule with a causal one and changes **nothing else**. The
projection, hidden width, student depth, attention-head selection, KV-head
handling, FFN neuron selection, RMSNorm treatment and folding, embedding
handling, vocabulary, architecture, teacher checkpoint and tokenizer are all
reached by identical code on identical inputs.

That claim is mechanical, not editorial. `init_student` gained one optional
argument, `kept_layers`, and
`tests/init/test_contribution.py::test_the_explicit_positional_map_reproduces_the_init_bitwise`
asserts that feeding it the positional map's own representatives reproduces the
positional initialization **with zero differing state-dict entries**.

---

## 2. Frozen base lineage

```
teacher            Qwen/Qwen3-4B-Thinking-2507 @ 768f209d, 36 layers
control init       artifacts/stage1/qwen3_0p6b_init_v0/checkpoint
                   model.safetensors sha256 86fbba78e8a2a324…
                   596,049,920 parameters, resolved RoPE base 5,000,000
control arms       e1_r2960k_{sa,sb}_pca  — E1/P1 KD-heavy 2.96M
                   usable 0.8400, correct 0.2067  (EXPERIMENTS.md §29)
recovery recipe    ce_weight 0.25, kd_weight 1.0, kd_temperature 1.0,
                   kd_scope "all", lr 5e-5, warmup 146, 2,916 steps,
                   2 blocks/step, block_len 8192, seeds 20260726 / 20260801
```

**Verified from the loader, not assumed** (`ladder_blocks`, rung 2,960,000 of
`ladder_uniform_probe`, `ladder.json` byte-identical to
`scripts/pod/hashes_ladder.txt`):

| quantity | value |
| --- | ---: |
| blocks | 1,944 |
| unique supervised CE targets | **2,960,507** ✓ |
| optimizer steps × blocks/step | 2,916 × 2 = 5,832 = exactly **3.0** exposures ✓ |
| **cumulative CE exposure** | **8,881,521** ✓ |
| KD positions per exposure (`scope: all`, content-masked) | 4,728,804 |
| content (non-pad) tokens | 4,730,748 |
| packing efficiency | 0.2971 |
| validation slice | 16 tail blocks, 81,195 CE targets |

`scripts/training/validate_e8_arms.py` re-derives every one of these from the
pack before training and exits 6 on any disagreement.

---

## 3. Prerequisite discovered while preparing this experiment

**The Stage 0 activation-statistics cache is gone.**
`artifacts/stage0/qwen3_4b_thinking_v1/activation_stats.safetensors` (1.95 GB,
sha256 `aaeb2e4c1ec67e6f…`) is absent from the dev box and was never on the relay
— the relay's 780 files contain no `stage0/` path. Stage 1 cannot construct *any*
initialization without it, so E8 was blocked on an artifact whose loss nobody had
noticed. It is recoverable only because the collection is deterministic and its
hash was recorded in the init manifest (P4 working as intended).

Regeneration ran on the dev box at **$0** (`collect_stage0.py`, teacher already
cached at the pinned revision, 4,972 s of CPU, 949,859 tokens — exactly the
historical count). Two hashes then decide how E8 proceeds, and both were
registered here as a branch **before** the answer was known:

| outcome | reading | action |
| --- | --- | --- |
| rebuilt statistics hash `aaeb2e4c…` **and** a rebuilt positional init hashes `86fbba78…` | the projection, head rule, FFN rule and norm solve are bit-identical to the control's | proceed; E8 is a single-variable experiment |
| statistics differ but a rebuilt positional init still hashes `86fbba78…` | float64 accumulation drifted, the bf16 cast absorbed it, the *initialization* is identical | proceed, and record the statistics drift |
| a rebuilt positional init does **not** hash `86fbba78…` | the treatment init cannot be proven to share the control's projection | **stop and report.** Either the control is retrained from a rebuilt positional init (2 more arms, +$6.7) or E8 is blocked. Maintainer decision, not this document's |

> **RESOLVED 2026-08-10, the first row. The branch is recorded above as it was
> written, before the answer was known.**
>
> ```
> regenerated activation_stats.safetensors  aaeb2e4c1ec67e6f6dd21ca40eceb0c193a9da5b010e8d12fcd5d24376cc47c1  == recorded
> rebuilt positional init model.safetensors 86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54  == pinned control init
> ```
>
> Both bit-exact, from the logged config alone, four weeks later. Every
> projection diagnostic reproduces to the last digit — `energy_captured_frac`
> 0.9323228843289764, `top_eigenvalue` 0.5261361586510566, `min_kept_eigenvalue`
> 6.677785428271654e-05, `final_norm_weight_range`
> [-0.03870667333325841, 7.125069193436976] — and the kept-Q-head sets and depth
> map are identical objects.
>
> **E8 is therefore a single-variable experiment**: the treatment initialization
> will differ from the control's only in the depth map. The rebuild also confirms
> the `kept_layers` addition leaves the default path untouched, since it produced
> the pinned bytes with the new code in place.
>
> Reproduce: `PYTHONPATH=src python scripts/training/init_stage1.py --config
> artifacts/audit/e8_repro_stage1.json` (the canonical Stage 1 config with only
> `output_dir` and `save_random_baseline` changed).

The environment matters and is now pinned: **transformers 5.13.1 / torch 2.13.0**
(the repo `.venv`), which is what produced the Stage 1 artifacts. See §10.

---

## 4. The selector, frozen before any map is seen

### 4.1 Primary objective

For a candidate skipped set `S`, bypass those teacher blocks through the residual
path — literally removing them from the module list, which is what the depth map
does at initialization — and measure

```
forward KL( teacher || teacher-with-S-bypassed )
```

over **all prediction positions** of every calibration item, in float32.

Forward KL, so positions the intact teacher is confident about dominate. All
positions rather than assistant-only, because that is the scope the training
objective already uses (`kd_scope: "all"`, content-masked) — the assistant-only
variant is recorded as a diagnostic and may not select anything.

Not hidden-state magnitude, not activation norm, not position. A block can carry
a large residual delta while contributing almost nothing to the next-token
distribution, and the reverse.

### 4.2 Aggregation

Unweighted mean over **5 domains** of the unweighted mean over each domain's
**sub-types** of that sub-type's token-mean KL:

| domain | sub-types |
| --- | --- |
| `general` | `general` |
| `math` | `gsm8k`, `openmath` |
| `rag_multihop` | `rag_evidence`, `multihop_qa` |
| `code` | `code` |
| `tool` | `tool_calling` |

Two unweighted levels, so neither a long-tokenizing sub-type nor a domain that
happens to own more sub-types can dominate. A missing sub-type raises rather than
silently reweighting its domain.

### 4.3 Iterative greedy, not one-shot Top-N

```
S = {}
for round in 0..7:
    score S ∪ {c} for every remaining c
    commit argmin; ties break on the lower layer index
```

36 + 35 + … + 29 = **260** subset evaluations, asserted by
`expected_evaluations(36, 8) == 260`. Redundancy is conditional: two blocks can
each be individually removable because the other compensates, and removing both
can be fatal. `test_greedy_avoids_the_pair_that_one_shot_top_n_would_take`
constructs exactly that case and shows one-shot ranking taking the fatal pair.

**No positional constraint.** `protect` defaults to empty. Forbidding the search
from removing layer 0 or layer 35 would re-import the assumption the experiment
exists to test; if those layers matter, the objective will say so.

The **full per-round table** is saved — all 260 candidates with primary score,
per-domain scores, per-sub-type scores, CE delta and every diagnostic — ordered by
layer index rather than by score, because a table sorted by the outcome invites
reading a ranking the greedy rule never used.

### 4.4 Diagnostics that may not select

Recorded per candidate per round, aggregated the same domain-balanced way so they
are comparable to the primary, and **preregistered as unable to change the map**:
target-token CE delta, `assistant`, `reasoning` (inside `<think>…</think>`),
`final_answer` (after `</think>`), `think_close`, `eos` (the assistant turn's
`<|im_end|>`), `tool_close` (`</tool_call>`). E6/E6b showed termination and
reasoning moving independently, which is why they are measured — and why reading
one and then adjusting the map would be choosing the map on the outcome.

`tool_close` covers only 8 positions in 6 items. Stated here as a limitation
rather than discovered later: it is too thin to support any claim on its own.

### 4.5 The instrument is validated before it is trusted

The search refuses to run unless `KL(intact ‖ intact)` — the reference against a
fresh pass of the same model — is **≤ 1e-6** on every item. A forward pass is
deterministic for a fixed input and shape, so this must be zero; if it is not, a
candidate ranking would be measuring kernel noise. Reported as
`self_consistency` in the artifact.

The **positional map is scored by the same frozen objective**, one extra
evaluation, so "is the contribution map actually lower-KL than the heuristic it
replaces" is answered rather than assumed.

---

## 5. The calibration set — built, frozen, hashed, proven

`artifacts/stage1/e8_calibration_v1`

| field | value |
| --- | ---: |
| items | 67 |
| prediction positions | **59,763** |
| content sha256 | `d65c1f40e4837ea1bd5bcc33c68041a13b797c68f5be3c0686e0142ed761028f` |
| manifest sha256 | `ecb72aa3b88818e93fb058d5d012e66274db9bc7b90234219501f0df86cef460` |
| items.jsonl sha256 | `c7202338109e459b17b70456461e8f304fadea7929ea547accee21adbbe7fd0b` |
| tokenizer | teacher @ 768f209d, vocab sha256 `7781771acc3798ee…` |

> **Corrected before launch (2026-08-10):** the `items.jsonl` row above first
> carried `94d747c8…`, a hash transcribed from an intermediate build's console
> output before the validation-slice exclusion (§5.1, finding 2) was applied. The
> frozen artifact's hash is `c7202338…`, which is what its own manifest records
> and what the pod verifies. A mis-transcribed hash identifies the wrong artifact,
> so it is corrected rather than preserved; no design decision changed, and the
> pod check now derives the file hash from the manifest instead of re-typing it.

Per sub-type positions: `general` 8,287 · `rag_evidence` 8,619 · `multihop_qa`
8,749 · `gsm8k` 8,472 · `openmath` 8,309 · `code` 8,622 · `tool_calling` 8,705.
Tagged positions: `assistant` 36,459 · `reasoning` 27,797 · `final_answer` 8,560
· `think_close` 51 · `eos` 51 · `tool_close` 8.

**General text** is raw FineWeb-Edu prose, pinned revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, stream index range **[40000, 40040)**
— far past everything already consumed (`warmup_v1` from 0, `holdout_v1` skip
5000, `e7_fineweb_val` [20000, 20454), `e7_fineweb_kd` [30000, 31902)). No chat
template.

**The other six sub-types** are teacher-native renders through the official
template — system block emitted once, `<think>…</think>` preserved, final answer,
`<|im_end|>` — drawn only from the corpus tail **beyond the 5.50M rung**, whole
sessions only.

### 5.1 Leakage control, and the two collisions it caught

The frozen 150-prompt behaviour battery is stratified-sampled from the **0.86M
rung's** verified-correct sessions, and the 0.86M rung is a prefix of the 2.96M
rung the arms train on. So "leakage-safe" here means safe against the *promotion
decision*, not merely against itself.

Excluded: every session and `source_id` the pack consumes through the 5.50M rung
(2,941 blocks, 7,350 sessions, 7,030 source ids), plus the pack's canonical
16-block validation slice, plus 8,069 prompt-content hashes.

Two collisions were found and fixed, both invisible to a session-id filter:

1. **`glaive-000749#t3`** — a *different* source item whose prompt text is
   byte-identical to a consumed session's. Tool-calling prompts are formulaic.
   Fixed by excluding on prompt-content hash.
2. **`openmath-000712#t1`** — a session inside the pack's own validation slice.
   That slice lives in the tail the calibration set is drawn from, so the rung
   filter structurally cannot see it. It would have calibrated the depth map on
   the teacher-native held-out CE that §7 then reports.

Both proofs are independent of the builder and fail closed:

* `scripts/data/check_e8_calibration_leakage.py` — six checks (`source_id`,
  `session_id`, `candidate_sha256`, `prompt_content`,
  `duplicate_token_sequences`, `ladder_validation_slice`) → **clean**.
* `scripts/data/check_stream_disjointness.py` — content-hash and index-range
  separation against `holdout_v1`, `warmup_v1`, `eval_behavior_v0/prompts.jsonl`
  and both E7 FineWeb streams → **disjoint**.

### 5.2 Declared deviations

* Sessions are filtered to 256–2,048 rendered tokens, so one long `code` session
  cannot own a sub-type. This inherits the corpus's existing length skew
  (`STATE.md` §8) rather than correcting it.
* Sub-type budgets land at 8,287–8,749 positions rather than exactly 8,192,
  because whole sessions are never cut. Aggregation is a per-sub-type *mean*, so
  exact equality is not required.

---

## 6. Then: the treatment initialization

`configs/stage1/qwen3_0p6b_from_4b_thinking_contribution.json` differs from the
canonical Stage 1 config in exactly `{depth_map_path, output_dir,
save_random_baseline, _purpose}`, asserted by
`tests/training/test_e8_arms.py::test_the_stage1_treatment_config_changes_only_the_depth_map`.
`save_random_baseline` is off because a random baseline does not depend on the
depth map, so re-saving 1.2 GB of identical weights would be waste.

The immutable initialization manifest carries: teacher identity and revision, the
selected 28 and removed 8 teacher layers, the depth-map artifact and its sha256,
the full 260-candidate search report hash, the calibration content hash, the
projection diagnostics, the per-layer kept Q heads, the FFN kept fraction, the
config hash, the recipe hash, the resolved RoPE base, the environment, and the
resulting `model.safetensors` sha256.

---

## 7. Then, and only then: the mandatory initialization NLL

**An initialization checkpoint is not complete until its own NLL measurement
artifact exists.** Nothing may be inherited, copied, interpolated or assumed from
a previous initialization, however closely related the recipe is.

Mechanical, in `src/aadistill/init/nll_gate.py` (11 tests):

* the record names the checkpoint's `model.safetensors` sha256, and the gate
  **recomputes** that hash from the checkpoint about to be trained — so a record
  copied from a sibling initialization simply does not match;
* every individual measurement *also* carries the checkpoint hash it was taken
  on, so one series cannot be spliced in behind a correct envelope;
* a record advertising `inherited` / `copied_from` / `source_record` /
  `interpolated` is rejected outright;
* a missing required series is a failure, not a shorter report.

Three required series per checkpoint, never averaged or substituted:

| key | what |
| --- | --- |
| `holdout_v1` | the historical 40-document FineWeb series, `eval_ppl.mean_nll`, `max_seq_len` 1024 |
| `fineweb_val_e7` | E7's dense 512×1024 disjoint stream — 20× the tokens — plus teacher→student KL, top-1, mean rank, entropy |
| `teacher_native_val` | the pack's 16-block validation slice over assistant-target positions, plus teacher KL, top-1, mean rank |

**The gate never looks at whether an NLL is good.** A worse or better
initialization NLL may not cancel or promote E8; a test asserts a record with
nll 99.0 passes. The scientific endpoint is autonomous behaviour after matched
recovery. The measurement exists so that what the initialization changed is known
*before* recovery training obscures it.

**The baseline initialization is remeasured** on the same device by the same
evaluator, so step 0 is comparable. Its historical value is reported separately
and never substituted (§10).

### 7.1 Catastrophic-validity aborts, registered now

Only these may stop E8 after the initialization is built. Everything else,
including a worse initialization NLL, proceeds to training:

* non-finite logits from the initialized student, or the forward smoke test
  failing;
* the checkpoint failing its save/reload round-trip;
* the resolved RoPE base not equal to the teacher's;
* parameter count or student config hash differing from the control's;
* the depth map not being 28 strictly increasing teacher layers;
* `self_consistency` in the search exceeding 1e-6;
* the calibration leakage report not clean.

---

## 8. Step-0 comparison — diagnostic only

Reported per initialization, side by side, before any training:

checkpoint hash · depth map · `holdout_v1` NLL · `fineweb_val_e7` NLL / KL /
top-1 / mean rank · `teacher_native_val` NLL / KL / top-1 / mean rank ·
calibration primary KL under the frozen objective · parameter count ·
architecture (config) hash.

Its purpose is to understand whether the new map begins closer to or farther from
the teacher. It decides nothing.

---

## 9. The formal comparison

| arm | init | recipe | trained? | seeds |
| --- | --- | --- | --- | --- |
| **control** | canonical positional PCA init | E1/P1 KD-heavy 2.96M | **no — retained** | `sa` 20260726, `sb` 20260801 |
| **treatment** | contribution-guided init | the identical recipe | yes | both |

The control is not retrained: its checkpoints and frozen raw evaluation artifacts
exist from E6/E6b, re-scored with the current scorer, on the same inclusion mask
`d6e24e0b09da1bcc…`.

Config identity is asserted mechanically. The realized diff between each
treatment config and its control is exactly
`{student_path, run_name, out_dir, _purpose}` — `student_path` is the variable,
the other three cannot affect a gradient. Verified by
`scripts/training/build_e8_configs.py` at generation time,
`scripts/training/validate_e8_arms.py` before training, and
`tests/training/test_e8_arms.py` in CI:

| run | config sha256 | seed |
| --- | --- | --- |
| `e8_contrib_r2960k_sa` | `8ed52bdb0c9db986…` | 20260726 |
| `e8_contrib_r2960k_sb` | `29f044fe0bac2817…` | 20260801 |

### 9.1 Evaluation

The frozen autonomous protocol of E6/E6b/E7, unchanged: 150 prompts, greedy,
unrestricted generation (P18, allowance = trained `block_len` 8192 − prompt),
inclusion mask asserted equal to `d6e24e0b09da1bcc…`, complete raw generations
saved, re-scored with the current scorer.

Reported per seed and pooled: `usable_rollout_rate` **with every component rate**
— never as a weighted average, carrying the standing caveats that it is blind to
correctness by construction and that `protocol_valid` subsumes two of its
components — `correct_overall`, `correct_given_usable`,
`natural_termination_rate`, `context_limit_rate`, `severe_repetition_rate`,
`empty_output_rate`, `answer_parse_failure_rate_numeric`, per-subset metrics
including GSM8K, and exact paired prompt win/tie/loss.

### 9.2 Registered thresholds

The floors E6/E6b/E7 registered on this battery, reused unchanged:

| axis | floor | claim rule |
| --- | ---: | --- |
| `usable_rollout_rate` | **0.0800** | above the floor **and** same sign on both seeds |
| `correct_overall` | **0.0600** | above the floor **and** same sign on both seeds |
| `correct_given_usable` | 0.0600 | above the floor **and** same sign on both seeds |
| every initialization diagnostic | — | descriptive only; may never promote or cancel |

Primary comparison: **treatment − control** on `usable_rollout_rate` and
`correct_overall`.

No promotion on NLL, CE, KL, top-1, calibration KL or any initialization metric.

---

## 10. Environment, and a live defect it exposed

**Pinned: transformers 5.13.1, torch 2.13.0** (the repo `.venv`). This is the
environment that produced the Stage 1 artifacts, and it is not
interchangeable with the other venv on this dev box.

The Stage 1 checkpoint's `config.json` stores `rope_theta: 5000000` inside the
transformers-5 `rope_parameters` dict. A transformers **4.x** reader silently
falls back to the class default **10,000** — a 500× error in the positional basis
— and nothing raises. Measured on this exact checkpoint:

| environment | `holdout_v1` NLL | resolved RoPE base |
| --- | ---: | ---: |
| transformers 5.13.1 | **11.748248** — reproduces the Stage 1 gate's 11.7482 exactly, 21,080 positions | 5,000,000 |
| transformers 4.57.1 | 11.395313 | 10,000 |

The teacher is unaffected (2.6265 vs the recorded 2.6264) because its config
predates the format change, which is why the skew is easy to miss.

`assert_rope_matches_config` already existed and the pod setup already asserted
`ROPE_OK` in both of its venvs, so **no trained arm is affected**. What was
missing was the same assertion on the measurement path, which is why a wrong
number was produced at all. It is now called in `measure_init_nll.py` and in
`init_stage1.py`, both of which refuse to proceed and record the resolved base.

Consequence for E8: the pod-measured baseline is the comparison; the historical
11.7482 is reported as a separately-labelled third number, never substituted.

---

## 11. Cost, and what it needs

From `scripts/training/plan_e8_budget.py` (regenerates `logs/e8_budget_plans.json`).
L40S at $0.99/h. Setup budgeted at 45 min, not the 5–8.5 min a warm image has
taken, because the same script/image/GPU has also taken 150+ min.

**Two pods, split where the artifact flow forces it.** Building the treatment
init needs the 1.95 GB Stage 0 statistics; training needs only the 1.19 GB
initialized checkpoint. The dev-box uplink is 0.72 MB/s and the relay holds
84.69 GB against a private-storage limit it has already hit once, so the smaller
artifact is the one that crosses: the dev box builds the initialization (12 s of
CPU, free) and ships the checkpoint.

### Pod A — contribution search

| phase | minutes | $ |
| --- | ---: | ---: |
| setup | 45.0 | 0.74 |
| contribution search, 260 evaluations | 59.5 | 0.98 |
| self-consistency + positional-map baseline | 4.0 | 0.07 |
| artifact manifest + verification | 8.0 | 0.13 |
| artifact synchronization | 5.0 | 0.08 |
| **expected completion** | **121.5** | **$2.00** |
| soft stop | 133.7 | $2.21 |
| artifact-recovery reserve | 30.0 | $0.49 |
| **absolute termination** | **163.7** | **$2.70** |

Search cost is arithmetic, not a guess: 67 × (1 + 260) = **17,487 forward
passes**, 260/261 of them over 28 of 36 blocks, ≈ 160 PFLOP at an assumed 45
sustained bf16 TFLOPS with a 1.6× overhead factor for launch, casts and the
152k-vocabulary KL reduction. This project has no *measured* number for this
workload, which is why the margin is named separately.

### Pod B — initialization NLL, 2 × 2.96M recovery, evaluation

| phase | minutes | $ |
| --- | ---: | ---: |
| setup | 45.0 | 0.74 |
| training (2 arms × 2,916 steps @ 4.15 s) | 403.4 | 6.66 |
| initialization NLL — treatment | 6.0 | 0.10 |
| initialization NLL — baseline, remeasured | 6.0 | 0.10 |
| pre-training gate | 3.0 | 0.05 |
| evaluation (2 arms, frozen battery) | 16.5 | 0.27 |
| artifact manifest + verification | 8.0 | 0.13 |
| artifact synchronization | 20.0 | 0.33 |
| **expected completion** | **507.9** | **$8.38** |
| soft stop | 558.7 | $9.22 |
| artifact-recovery reserve | 30.0 | $0.49 |
| **absolute termination** | **588.7** | **$9.71** |

### The authorization

```
actual cumulative spend                 $160.158
current cumulative authorization        $162.49
available under it                        $2.33

E8 expected completion                   $10.38
E8 hard backstop                         $12.41

ADDITIONAL AUTHORIZATION REQUIRED        $10.08
proposed new cumulative cap             $172.57
```

**Not reduced to one seed to fit.** The behaviour-metric seed noise floor on this
battery is 0.1290 — wider than any effect E8 could claim — so a one-seed arm
cannot be read. Shrinking the experiment to fit a shortfall is exactly what
`budget.plan_session` refuses to do.

---

## 12. Outcome interpretation, fixed in advance

| initialization diagnostics | autonomous behaviour | reading |
| --- | --- | --- |
| improve | improves (above floor, both seeds) | **position-based depth compression was discarding important teacher computation.** The first structural intervention to move behaviour |
| improve | unchanged | the new map reconstructs the teacher better locally, and that **does not survive recovery**. Initialization quality is not the bottleneck at this rung |
| worse | improves | **important**: initialization NLL is again not a sufficient proxy, and contribution-aware structure preserves reasoning-relevant computation that general LM NLL does not see |
| worse | regresses | **reject the contribution-guided map** |
| improve or flat | `usable` improves without `correct` | another stability-only effect, the project's recurring pattern. Report as such; do not present it as progress on reasoning |

If behaviour moves, the report must also localise it: general LM,
termination/stability, autonomous correctness, or specific reasoning subsets —
using the per-component and per-subset rates, not the headline.

**A predicted null is a real outcome.** Eleven interventions have moved behaviour
or nothing and none has moved reasoning correctness; E8 is designed so "the map
changed, behaviour did not" is a clean answer about structure rather than a failed
run.

---

## 13. What is explicitly not in E8

No P2-5.50M. No 2.96M + FineWeb. No FineWeb-ratio sweep. No on-policy or GKD. No
second contribution-map variant. No E9 combination.

**And no uncontrolled layer-map sweep.** The selection algorithm, the calibration
objective and the calibration set are frozen above, before any map is seen. Only
the one map the frozen search returns receives the two-seed 2.96M recovery. A
zero-cost diagnostic revealing a bug or a degeneracy may fix the method before
execution; nothing may choose a map after seeing behaviour.

---

## 14. Reproduction of everything above

```bash
# free, dev box, transformers 5.13.1
PYTHONPATH=src python scripts/training/collect_stage0.py \
    --config configs/stage0/qwen3_4b_thinking_v1.json
PYTHONPATH=src python scripts/data/fetch_fineweb_docs.py \
    --start-index 40000 --max-docs 40 \
    --out artifacts/stage1/e8_calibration_v1/general_docs.jsonl
PYTHONPATH=src python scripts/data/build_e8_calibration.py \
    --out artifacts/stage1/e8_calibration_v1 \
    --general-docs artifacts/stage1/e8_calibration_v1/general_docs.jsonl \
    --reserved data/eval_behavior_v0/prompts.jsonl \
    --reserved 'artifacts/eval/battery_v2/*.jsonl'
PYTHONPATH=src python scripts/data/check_e8_calibration_leakage.py \
    --calibration artifacts/stage1/e8_calibration_v1 \
    --reserved data/eval_behavior_v0/prompts.jsonl \
    --reserved 'artifacts/eval/battery_v2/*.jsonl' \
    --out artifacts/stage1/e8_calibration_v1/leakage.json
PYTHONPATH=src python scripts/data/check_stream_disjointness.py \
    --stream artifacts/stage1/e8_calibration_v1 \
    --stream artifacts/stage3/e7_fineweb_kd --stream artifacts/stage3/e7_fineweb_val \
    --reserved data/warmup/holdout_v1.jsonl --reserved data/warmup/warmup_v1.jsonl \
    --reserved data/eval_behavior_v0/prompts.jsonl \
    --out artifacts/stage1/e8_calibration_v1/general_disjointness.json
PYTHONPATH=src python scripts/training/build_e8_configs.py
PYTHONPATH=src python scripts/training/validate_e8_arms.py \
    --out artifacts/audit/e8_preflight_configs.json
PYTHONPATH=src python scripts/training/plan_e8_budget.py
PYTHONPATH=src pytest tests/ -q          # 1,138 pass, 3 skipped

# pod A, after authorization
PYTHONPATH=src python scripts/training/search_depth_map.py \
    --calibration artifacts/stage1/e8_calibration_v1 \
    --student-layers 28 --out artifacts/stage1/e8_depth_search

# dev box, free, after pod A
PYTHONPATH=src python scripts/training/init_stage1.py \
    --config configs/stage1/qwen3_0p6b_from_4b_thinking_contribution.json

# pod B, before training — the gate, not a formality
PYTHONPATH=src python scripts/evaluation/measure_init_nll.py \
    --checkpoint artifacts/stage1/e8_contribution_init_v1/checkpoint \
    --label e8-contribution-init --teacher Qwen/Qwen3-4B-Thinking-2507 \
    --out artifacts/stage1/e8_contribution_init_v1/init_nll.json
PYTHONPATH=src python scripts/evaluation/measure_init_nll.py \
    --checkpoint artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
    --label baseline-positional-init --teacher Qwen/Qwen3-4B-Thinking-2507 \
    --out artifacts/stage1/qwen3_0p6b_init_v0/init_nll.json
PYTHONPATH=src python scripts/training/validate_e8_arms.py --require-init \
    --out artifacts/audit/e8_preflight.json
```

## 15. Operational contract

[`scripts/pod/AGENTS.md`](../scripts/pod/AGENTS.md) is binding for both pods:
detached start via `start_job.py`; `watchdog.py` beside the launcher from pod
creation; `LogRelay` mirroring event streams continuously;
`collect_artifacts.py` gating teardown with `final_required` semantics; hash
verification before normal teardown; recurring status polling from launch.
`--terminate-after` is a redundant third layer and has never been observed to
fire.

Every initialization checkpoint and every trained checkpoint is hashed and tied
to its metrics, and **an initialization checkpoint is not complete until its own
NLL artifact exists** (§7).
