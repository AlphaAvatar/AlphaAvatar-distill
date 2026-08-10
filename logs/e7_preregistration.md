# Experiment 7 — preregistration

> **STATUS: EXECUTED 2026-08-09. This document is frozen as the prospective
> record and is deliberately NOT edited to match the outcome.** It was written
> before any E7 GPU time was spent; every threshold, floor, budget and outcome
> interpretation below was fixed in advance. Figures that were forward-looking
> when written (test counts, open blockers, the authorization state) are left as
> they stood.
>
> The result is in [`EXPERIMENTS.md`](EXPERIMENTS.md) §34 and
> [`e7_report.md`](e7_report.md): **preregistered outcome 2** — general language
> modelling restored (−5.22 nats), autonomous behaviour unmoved (usable rollout
> +0.0000, every comparison inside its floor). The maintainer authorized the
> full B+C design at a $12.82 backstop and a $162.49 cumulative cap; the run
> cost $10.49.

**Status when written: DRAFT. Not authorized. Nothing has been launched, trained
or evaluated.** This document, the built data streams, the configs and the tests
exist; no GPU has been used for any of it. Authorization is a separate decision
and requires a cumulative-cap increment (§10).

**Question.** Does adding general-text teacher KD, while preserving the existing
teacher-rollout training trajectory exactly, restore general language modelling —
and if it does, does any of that restoration transfer to autonomous rollout
correctness?

---

## 1. Why this experiment, and what it is not

The rollout recipe appears to destroy general language modelling and then stop
improving. Held-out FineWeb NLL across the E1/P1 KD-heavy lineage
(`artifacts/stage3/e1_consolidated.json`, `holdout_v1`, `max-seq-len 1024`):

| rung | 0.25M | 0.46M | 0.86M | **1.60M** | 2.96M | 5.50M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sa` | 6.7169 | **6.1616** | 8.8758 | **9.7145** | 10.4031 | 10.7875 |
| `sb` | 7.8699 | 6.2948 | 9.3649 | **9.4845** | 9.7864 | 9.4548 |

The Stage 1 init starts at 11.75. Training improves general text to ~6.2 by
0.46M and then **gives it back**, ending ~9.6 at the 1.60M rung this experiment
trains at. Over the same range autonomous correctness never leaves 0.11–0.21.

E7 asks whether the two facts are connected. It is **not** a scaling experiment,
not an objective-reweighting experiment (that route is closed — see
`decisions.md`, KD dose-response), and not a proposal to change the rollout
recipe. It adds a strictly additional signal and holds everything else fixed.

**A predicted null is a real outcome.** The project's standing finding is that
nothing yet moves correctness. E7 is designed so that "general LM restored,
behaviour unchanged" is a clean, publishable answer rather than a failed run.

---

## 2. Frozen base lineage (fixed by the maintainer, not by this document)

```
initialization:   canonical Stage 1 PCA init
                  artifacts/stage1/qwen3_0p6b_init_v0/checkpoint
                  model.safetensors sha256 86fbba78e8a2a324…

rollout rung:     1,600,353 unique supervised CE tokens
                  1,174 blocks x 8,192, canonical three-exposure schedule
                  cumulative rollout CE exposure 4,801,059  (verified below)

rollout objective: E1/P1 KD-heavy — ce_weight 0.25, kd_weight 1.0,
                   kd_temperature 1.0, kd_scope "all"
```

**Verified from the loader, not assumed.** Reading `blocks[:1174]` of the
canonical pack (`blocks.npz` sha256 `6f324cb0f37bc0f0…`, byte-identical to
`scripts/pod/hashes_ladder.txt`):

| quantity | value |
| --- | ---: |
| blocks | 1,174 |
| CE targets per exposure | 1,600,353 |
| **cumulative CE exposure (x3)** | **4,801,059** ✓ |
| KD positions per exposure (`scope: all`, content-masked) | 2,660,125 |
| cumulative rollout KD positions | 7,980,375 |
| content (non-pad) tokens | 2,661,299 |
| packing efficiency | 0.2767 |
| optimizer steps x blocks/step | 1,761 x 2 = 3,522 = exactly 3.0 exposures |

**Not used, by instruction:** P2 CE-heavy as a base; any already-trained 1.60M or
2.96M checkpoint as a start point. Every new arm forks from the Stage 1 init.
`validate_e7_arms.py` asserts both.

**Retained external behavioural reference: E1/P1 KD-heavy 2.96M.** It is *not*
the E7 training scale and does not replace the 1.60M condition anywhere in this
design.

---

## 3. Arms

| arm | what it is | trained? | seeds |
| --- | --- | --- | --- |
| **A** | E1/P1 KD-heavy 1.60M — the retained baseline | **no** | `sa` 20260726, `sb` 20260801 |
| **B** | A's rollout stream + **FineWeb-Edu raw-text teacher KD** | yes | both |
| **C** | A's rollout stream + **matched extra KD from unused in-domain rollout text** | yes | both |

**Arm A costs nothing to re-train.** Its checkpoints are on the relay
(`e1_scaling_20260801/e1_r1600k_{sa,sb}_pca/step_001761`, sha256
`6f77676ab8fde397…` / `e432d57e598d57e1…`) and its frozen battery artifacts
exist from E6. Only the general-text diagnostics on the new validation stream are
new, and they are ~2 min per checkpoint.

**Config identity, asserted mechanically.** Each of B and C is
`configs/stage3/e1/e1_r1600k_{seed}_pca.json` with **one key added**. The
verified diff is exactly `{extra_stream, run_name, out_dir, _purpose}`:

| run | config sha256 | seed |
| --- | --- | --- |
| `e7_fineweb_r1600k_sa` | `ec36810366cc7ae6…` | 20260726 |
| `e7_fineweb_r1600k_sb` | `94d14b4d195e8eea…` | 20260801 |
| `e7_control_r1600k_sa` | `f723e3729bc16d09…` | 20260726 |
| `e7_control_r1600k_sb` | `0f6fc09b05dcd9cb…` | 20260801 |

B and C differ only in `extra_stream.data_dir` and `extra_stream.kind`.

### Why C is required, and what it can and cannot show

Two things could explain any effect B has: **what FineWeb contains**, or simply
**more KD positions, more gradient signal and more compute**. C supplies the
identical extra budget from the same teacher-rollout distribution.

**One interpretive caveat, stated before the run.** C is *in-domain*, and E6
showed that more in-domain data improves rollout stability. C is therefore a
**strong** control: if it moves behaviour as much as B, the honest reading is
"extra KD positions did it", not "FineWeb did nothing special". That is the
intended attribution and the reason C is not a neutral filler stream.

**Explicitly rejected as the control: E1/P1-2.96M.** It changes unique rollout
data, CE exposure, blocks and the whole training trajectory. It is a different
experiment, not a compute-matched control.

---

## 4. The matched extra-KD budget

Both extra streams are **dense**: documents concatenated with an explicit
`<|endoftext|>` separator, no padding, incomplete tail dropped. So KD positions
are exactly `n_blocks x (block_len - 1)` — an integer known before training, not
a property of how something packed.

### Candidate budgets (all: 1 block/step, every step, 1,761 blocks, 1 exposure)

| `block_len` | extra KD positions | vs rollout KD | FineWeb share of all KD | extra fwd tokens/model | vs rollout fwd |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 899,871 | 11.3% | 10.1% | 901,632 | 3.1% |
| **1024** | **1,801,503** | **22.6%** | **18.4%** | **1,803,264** | **6.25%** |
| 2048 | 3,604,767 | 45.2% | 31.1% | 3,606,528 | 12.5% |
| 4096 | 7,211,295 | 90.4% | 47.5% | 7,213,056 | 25.0% |

### Preregistered setting

```
block_len          1024
blocks_per_step    1
every_n_steps      1          (present on EVERY optimizer step)
n_blocks           1761       (exactly one exposure; no general text repeats)
extra KD positions 1,801,503  exact, auditable, checked against the run
lambda_extra       0.25
```

**Why 1024.** It matches the historical FineWeb NLL protocol
(`eval_ppl.py --max-seq-len 1024`), so the training context and the diagnostic
context agree; 22.6% is a standard replay ratio that leaves the rollout stream
81.6% of all KD positions; and +6.25% forward tokens is a cost the budget
absorbs.

**Why every step, not a subset.** A cadence of *k* makes one step in *k*
structurally different from the others, and that periodicity interacts with the
LR schedule. Every step keeps the per-step gradient composition uniform. The
budget is set by sequence length instead, which is exact either way.
(`every_n_steps` is implemented and tested; it is simply not used here.)

**Why λ = 0.25.** The rollout objective is `0.25·CE + 1.0·KD`; λ = 0.25 gives
general text the same weight the recipe already gives its secondary term,
against rollout KD's 1.0. The scale argument matters more than the symmetry:
rollout KD **falls** through training (E6b: val_kd 10.60 → 1.04) while FineWeb KD
stays high, so a λ near 1.0 would make general text the dominant late-training
gradient and turn E7 into a different experiment. λ = 0.25 keeps it a declared
minority term throughout.

**No sweep.** One setting is preregistered. A **non-training** preflight
diagnostic (`gradient_share`, `scripts/training/e7_preflight.py`) measures
‖∇(λ·KD_extra)‖ / ‖∇(rollout loss)‖ at the Stage 1 init over 4 steps, takes no
optimizer step, and leaves the run bit-identical.

* **Registered acceptance band: `ratio_mean` ∈ [0.05, 1.00].**
* Outside the band → **stop and report**. Do not auto-tune, do not re-run with a
  different λ without a separate decision. Selecting λ from this measurement
  after seeing it would be a sweep wearing a disguise.

---

## 5. Data

### 5.1 FineWeb (arm B, and the validation stream)

Pinned identity, asserted at build time and refused if it moves:

| field | value |
| --- | --- |
| dataset | `HuggingFaceFW/fineweb-edu` |
| config / split | `sample-10BT` / `train` |
| revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| license | ODC-By 1.0 |
| filter | `doc_char_min` 500, **no cap** |
| boundary policy | `<|endoftext|>` (151643) between documents; partial tail dropped |
| chat template applied | **false** |
| assistant CE positions | **0** |
| tokenizer | Stage 1 checkpoint tokenizer, hashed into the manifest |

| stream | blocks x len | index range | docs | KD positions | blocks sha256 |
| --- | --- | --- | ---: | ---: | --- |
| `e7_fineweb_kd` (train) | 1761 x 1024 | [30000, 31902) | 1,884 | 1,801,503 | `b70beffac337ee37…` |
| `e7_fineweb_val` (validation) | 512 x 1024 | [20000, 20454) | 450 | 523,776 | `e4002bbbbadf1a91…` |

**Reserved ranges and why the builder refuses `--start-index < 20000`:**

| consumer | range | note |
| --- | --- | --- |
| `warmup_v1` Stage 0 statistics | from index 0, 848 docs under a char budget | last index read was never recorded, so the whole prefix is treated as consumed |
| `holdout_v1` Stage 1 gate | `skip_docs=5000`, 40 docs | the historical holdout, **preserved unchanged** for continuity |

### 5.2 The matched control (arm C)

`e7_control_kd`: 1761 x 1024, 1,801,503 KD positions, 0 padding, blocks sha256
`4e54f8e18baf01dc…`. Built from the **content tokens** of canonical-pack blocks
`[1174, 1853)` — after the trained rung, before the validation tail at 2941 — and
re-packed densely under the identical boundary policy.

Re-packing rather than reusing padded blocks is what makes the match exact: a
ladder block is 72.3% padding at this rung, so taking blocks verbatim would need
~3.6x the forward workload for the same KD positions.

| matched quantity | B | C |
| --- | ---: | ---: |
| blocks x block_len | 1761 x 1024 | 1761 x 1024 |
| extra KD positions | 1,801,503 | 1,801,503 |
| extra forward tokens / model | 1,803,264 | 1,803,264 |
| padding tokens | 0 | 0 |
| CE positions | 0 | 0 |
| microbatch schedule | 1 block/step, every step | 1 block/step, every step |
| optimizer steps | 1,761 | 1,761 |

**Compute matching is exact.** Nothing is approximated and there is nothing to
report as a mismatch.

### 5.3 Disjointness — proven, not assumed

`artifacts/stage3/e7_disjointness.json`, regenerable, **fails closed**. Both
index-range separation *and* content-hash separation (sha256 over document text
alone, excluding ids), across: the three E7 streams, `holdout_v1`, `warmup_v1`,
`eval_behavior_v0/prompts.jsonl`, and all seven `capability-v2` battery files.

**Result: disjoint.** Zero content-hash overlaps involving any E7 stream, zero
index-range overlaps.

> **Incidental finding, recorded because it was found here and is not E7's.**
> The check reports one overlap **between two reserved artifacts**:
> `capability-v2`'s `rag.jsonl` and `answerability_paired.jsonl` share the SQuAD
> item `squad-val-57299021af94a219006aa50c` with byte-identical prompt text; the
> ids differ only by a `pair-0118-safe:` prefix, which is presumably why the
> battery's own leakage check (scoped to train/eval split leakage) did not see
> it. Exactly 1 of 846 prompts; zero within-file duplicates. It does not touch
> E7, and it is not fatal — but `rag` and `answerability_paired` are not fully
> independent subsets and per-subset comparisons should say so.

### 5.4 Is the historical holdout big enough? No.

`holdout_v1` is **40 documents, ~25k tokens**. Against E1's between-seed
`holdout_nll` spreads of 0.23 (1.60M), 0.62 (2.96M) and 1.34 nats (5.50M), a
40-document set adds avoidable evaluation variance to an already seed-noisy
quantity.

**Proposal, implemented:** `e7_fineweb_val`, 512 blocks x 1024 = 524,288 tokens —
**20x** the historical set, disjoint from everything. **`holdout_v1` is preserved
and still measured**, so the historical series stays continuous. The two are
*not* interchangeable (different documents, dense packing vs per-document
truncation) and must be reported as separate columns, never merged.

---

## 6. Trainer mechanics, and the guarantee they buy

The extra text is a **second stream with its own cursor**, consumed inside the
same optimizer steps — never merged into the rollout pack, which would move every
block boundary and every example's position against the LR schedule.

```
rollout loss:   0.25 x rollout CE   +  1.0 x rollout KD
extra loss:     lambda_extra x extra KD
update:         rollout loss + extra loss        (one backward pass group,
                                                  one clip, one optimizer step)
```

**Independent token counts, independent normalizers.** Each term is a mean over
its own positions. There is no pooled mean whose effective weight would depend on
padding, sequence length or packing efficiency — which matters concretely,
because the rollout pack is 72% padding at this rung.

Asserted by `tests/training/test_dual_stream.py` (19 tests), with the extra
stream present and absent:

| property | test |
| --- | --- |
| rollout block order unchanged | `test_rollout_block_order_is_unchanged_by_the_extra_stream` |
| rollout LR positions unchanged | `test_rollout_lr_positions_are_unchanged` |
| optimizer-step count unchanged | `test_the_optimizer_step_count_is_unchanged` |
| rollout CE/KD terms and normalizers unchanged | `test_the_rollout_loss_terms_are_identical_at_step_zero` |
| extra stream contributes zero CE | `test_the_extra_stream_contributes_zero_ce_positions` |
| padding excluded / refused | `test_a_padded_extra_stream_is_refused` |
| B and C budgets matched (shipped configs) | `test_the_real_e7_arm_configs_are_budget_matched` |
| update schedules matched | `test_the_cadence_is_a_pure_function_of_the_step` |
| normalizers independent | `test_the_two_streams_have_independent_normalizers` |
| both cursors reproduce on resume | `test_resume_reproduces_both_stream_cursors` |
| planned budget == consumed budget | `test_the_planned_budget_equals_what_the_run_consumes` |

Both cursors are pure functions of the step counter, so resume needs no
dataloader state for either stream.

---

## 7. Metrics — two questions, never mixed

### 7.1 General-language preservation — **DIAGNOSTICS ONLY**

On `e7_fineweb_val` (primary) and `holdout_v1` (continuity), per arm per seed:
`nll`, teacher→student `kl`, `top1`, `mean_rank`, `mean_target_prob`,
`mean_entropy` (`scripts/evaluation/eval_general_text.py`).

`kl` beside `nll` distinguishes "matched the teacher better" from "modelled the
text better"; `mean_entropy` catches a model that restores NLL by becoming
uniformly less confident. **None of these may promote a checkpoint**
(`decisions.md`, 2026-08-09).

### 7.2 Autonomous behavioural endpoints — **the decision**

The frozen current rollout battery, 150 prompts, greedy, unrestricted generation
(P18), re-scored with the current scorer: `usable_rollout_rate`,
`correct_overall`, `correct_given_usable`, `natural_termination_rate`,
`context_limit_rate`, `severe_repetition_rate`, `empty_output_rate`,
`answer_parse_failure_rate_numeric`, and per-subset metrics **including GSM8K**.

`usable_rollout_rate` is reported with **every component rate**, never as a
weighted average, carrying the standing caveats: it is blind to correctness by
construction, and its components are not independent.

### 7.3 Registered thresholds (prospective — registered before the run it judges)

Reusing the floors E6/E6b registered on this battery:

| axis | floor | claim rule |
| --- | ---: | --- |
| `usable_rollout_rate` | 0.0800 | above the floor **and** same sign on both seeds |
| `correct_overall` | 0.0600 | above the floor **and** same sign on both seeds |
| `correct_given_usable` | 0.0600 | above the floor **and** same sign on both seeds |
| general-text `nll` | — | **descriptive only**; "restoration" is reported as a magnitude, never as a pass |

Primary comparisons, fixed now:

1. **B − A** on `usable_rollout_rate` and `correct_overall` — does FineWeb KD
   change behaviour at all?
2. **B − C** on the same axes — is any change attributable to FineWeb's
   *content* rather than to extra KD positions and compute?
3. **B − A** on general-text `nll` and `kl` — is general LM restored, and by how
   much against A's ~9.6 and the 0.46M-rung minimum of ~6.2?

### 7.4 Outcome interpretation, fixed in advance

| general-text diagnostics | autonomous correctness | reading |
| --- | --- | --- |
| improve | improves (above floor, both seeds) | **FineWeb transfers into useful behaviour** — the first correctness movement in the project |
| improve | unchanged | **FineWeb preserves language modelling but does not solve reasoning** — the predicted null; closes the "lost general LM causes the correctness ceiling" hypothesis |
| improve or flat | `usable` improves without `correct` | **another stability-only effect**, the project's recurring pattern; report as such and do not present it as progress on reasoning |
| improve | regresses | **objective/distribution conflict** — general text is competing with the rollout objective for capacity; report the alignment-tax-style tradeoff explicitly (P10.1) |

If **C** matches B on every axis, the attribution is "extra KD positions and
compute", not FineWeb content — and B alone would have been uninterpretable.

---

## 8. What blocks a launch

| blocker | state |
| --- | --- |
| authorization + cumulative-cap increment | **OPEN — required** (§10) |
| live provider control-plane never verified | **OPEN** — see `e7_canary_proposal.md`; a multi-hour E7 must not be its first real test |
| streams built and disjointness proven | done |
| configs built and diff-verified | done |
| dual-stream trainer + tests | done, 1,029 tests pass |
| λ preflight diagnostic | implemented and tested; **not yet run** (needs the real teacher/student, i.e. a pod) |
| arm A general-text baseline on the new val stream | **not yet measured** (priced into the session) |

---

## 9. Cost — measured, phase-by-phase

From `scripts/training/plan_e7_budget.py` (regenerates `logs/e7_budget_plans.json`).
Priced at **4.60 s/step** = E6b's measured 4.15 s/step + 10% for the extra
stream, on L40S at $0.99/h. Setup is budgeted at **45 min, not** the 5–8.5 min a
warm image has taken, because the same script/image/GPU has also taken 150+ min.

### Full design — B x2 + C x2

| phase | minutes | $ |
| --- | ---: | ---: |
| setup | 45.0 | 0.74 |
| training (4 arms x 1,761 steps) | 540.0 | 8.91 |
| validation + checkpointing | *(in the step rate)* | — |
| evaluation (4 arms, battery + general text) | 42.0 | 0.69 |
| arm A general-text diagnostics | 4.0 | 0.07 |
| artifact manifest + verification | 8.0 | 0.13 |
| artifact synchronization | 40.0 | 0.66 |
| **expected completion** | **679.0** | **$11.20** |
| soft stop (start nothing new) | 746.9 | $12.32 |
| artifact-recovery reserve | 30.0 | $0.49 |
| **absolute termination** | **776.9** | **$12.82** |

### Reduced design — B x2 only · **ATTRIBUTION-INCOMPLETE**

| | minutes | $ |
| --- | ---: | ---: |
| expected completion | 368.0 | $6.07 |
| soft stop | 404.8 | $6.68 |
| artifact-recovery reserve | 30.0 | $0.49 |
| **absolute termination** | **434.8** | **$7.17** |

**Label, not a footnote.** Without C, a positive B result cannot distinguish
FineWeb's content from extra KD positions, gradient signal and compute — and
given E6, "more KD positions helped" is a live alternative, not a remote one. The
reduced design is offered because it was asked for. **It must not be selected
without a separate user decision.**

---

## 10. Proposed cumulative caps

```
previous authorized cumulative cap:  $149.03   (EXCEEDED and closed)
actual cumulative spend:             $149.59   <-- the planning baseline
recorded E6b overrun:                  $0.56
currently available authorization:     $0.00
```

New paid execution requires an explicit increment **above $149.59**. The historical
$149.03 is not remaining balance.

| option | canary | E7 | proposed new cumulative cap |
| --- | ---: | ---: | ---: |
| canary + **E7 full** | $0.82 | $12.82 | **$163.23** |
| canary + E7 reduced | $0.82 | $7.17 | $157.59 |

The planner refuses any requested cap that cannot contain its own artifact and
teardown reserve; every figure above is a `hard_terminate` threshold with the
30-minute recovery reserve already inside it.

---

## 11. Reproduction of everything above

```bash
PYTHONPATH=src python scripts/data/build_fineweb_kd.py \
    --out artifacts/stage3/e7_fineweb_val --n-blocks 512 --block-len 1024 \
    --start-index 20000 --purpose validation
PYTHONPATH=src python scripts/data/build_fineweb_kd.py \
    --out artifacts/stage3/e7_fineweb_kd --n-blocks 1761 --block-len 1024 \
    --start-index 30000 --purpose train \
    --exclude-hashes artifacts/stage3/e7_fineweb_val/docs.jsonl
PYTHONPATH=src python scripts/data/build_control_kd.py \
    --pack artifacts/stage3/ladder_uniform_probe --rung 1600000 \
    --match artifacts/stage3/e7_fineweb_kd --out artifacts/stage3/e7_control_kd
PYTHONPATH=src python scripts/data/check_stream_disjointness.py \
    --stream artifacts/stage3/e7_fineweb_kd --stream artifacts/stage3/e7_fineweb_val \
    --stream artifacts/stage3/e7_control_kd \
    --reserved data/warmup/holdout_v1.jsonl --reserved data/warmup/warmup_v1.jsonl \
    --reserved data/eval_behavior_v0/prompts.jsonl \
    --reserved artifacts/eval/battery_v2/*.jsonl \
    --out artifacts/stage3/e7_disjointness.json
PYTHONPATH=src python scripts/training/build_e7_configs.py
PYTHONPATH=src python scripts/training/validate_e7_arms.py --require-streams \
    --out artifacts/audit/e7_preflight.json
PYTHONPATH=src python scripts/training/plan_e7_budget.py
PYTHONPATH=src pytest tests/ -q
```

`artifacts/stage3/ladder_uniform_probe` is a byte-identical local copy of the
canonical `artifacts/stage3/ladder_uniform` pack that pods stage from the relay —
all three files match `scripts/pod/hashes_ladder.txt`.

**On a pod, before training:**

```bash
PYTHONPATH=src python scripts/training/e7_preflight.py \
    --config configs/stage3/e7/e7_fineweb_r1600k_sa.json \
    --out artifacts/audit/e7_gradient_share_sa.json
```
