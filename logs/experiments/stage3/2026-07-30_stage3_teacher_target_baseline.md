# 2026-07-30 — Corrected teacher-target baseline: **cold-start / non-convergence diagnostic**

> **RECLASSIFIED 2026-07-30 by maintainer direction. Read this first.**
>
> This run fixed the initialization confound of the earlier attempt — every arm
> forked from the Stage 1 structural init. **It does not establish that
> teacher-native supervision is unsuitable, and its R2 trigger must not be used
> to justify a public / no-think warm-up, final-only targets, shortened
> reasoning, or any other target-behaviour change** (AGENTS.md P17).
>
> Its interpretation is **both convergence-limited and measurement-limited**:
>
> * The teacher-treatment arm improved holdout NLL **11.7565 → 6.2255**, a
>   *larger* absolute improvement than the public arm's 11.7565 → 7.7260.
> * The run used **137 steps ≈ 5%** of the 2,700-step reference run.
> * The corpus is **487 training prompts, ~0.71M real tokens** (treatment).
> * It therefore cannot establish convergence or the final achievable behaviour
>   of either recipe.
> * **99.3% of teacher-treatment generations were forcibly stopped at 512
>   tokens**, so their natural termination behaviour was never measured. Those
>   are **right-censored** observations, not failures (AGENTS.md P18).
> * `empty_answer` means the parser found no final answer after a closing
>   reasoning delimiter. It does **not** prove the model emitted no useful
>   reasoning.
> * The public targets teach the model to close an **empty `<think>` block
>   immediately**. That makes short-window format compliance easier while moving
>   the behavioural protocol away from the thinking-only teacher.
> * Lower teacher-forced NLL alongside unstable free generation may indicate
>   insufficient optimization, exposure bias, target-rendering problems or
>   on-policy mismatch — not an intrinsically unsuitable target.
>
> The recorded metrics and the historical R2 trigger are preserved unchanged
> below. **An unrestricted re-evaluation (no artificial generation cap) is
> required before any route-level conclusion.**


- **Pre-registration:** [`proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md`](../../proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md)
- **Generated mechanically** by `scripts/pod/report_tt2x2.py` from the runs' own artifacts. Numbers are measured; the rules below are applied mechanically; the stage verdict is left to review.

## 1. Corpus

- accepted prompts: **540** (dropped 212 public fallbacks)
- corpus sha256: `18028f0cb8009132…`
- capability scope: {"allowed_groups": ["rag_evidence", "multihop_qa", "code_math"], "dropped_out_of_scope": {}}
- packing: {"mode": "best_fit", "block_len": 8192, "seed": 20260726}

- **control**: supervised 27526 tokens (fraction 0.0929), lossless=True
- **treatment**: supervised 519478 tokens (fraction 0.6591), lossless=True

## 2. Arm means (two seeds, mean ±half-spread)

| readout | control (2 seeds) | treatment (2 seeds) |
|---|---|---|
| holdout NLL (bf16) | 7.7260 ±1.1033 (n=2) | 6.2255 ±0.4844 (n=2) |
| holdout NLL (INT8) | 7.6878 ±0.9717 (n=2) | 6.2447 ±0.4824 (n=2) |
| p(`</think>`) | 0.7925 ±0.0515 (n=2) | 0.0000 ±0.0000 (n=2) |
| p(`<|im_end|>`) | 0.3678 ±0.0379 (n=2) | 0.0052 ±0.0011 (n=2) |
| `behavior_score_v0` | 0.2175 ±0.0025 (n=2) | 0.0008 ±0.0004 (n=2) |
| `format_ok` | 0.6250 ±0.0461 (n=2) | 0.0000 ±0.0000 (n=2) |
| `think_closed` | 0.9276 ±0.0461 (n=2) | 0.0066 ±0.0066 (n=2) |
| `terminated` | 0.6579 ±0.0395 (n=2) | 0.0066 ±0.0066 (n=2) |
| `empty_answer` | 0.0724 ±0.0461 (n=2) | 0.9803 ±0.0066 (n=2) |

## 3. Per run

| run | last step | holdout NLL | p(</think>) | p(<\|im_end\|>) | behavior | format_ok | terminated |
|---|---|---|---|---|---|---|---|
| `ttb_ctrl_a` | 135 | 6.6227 | 0.8440 | 0.4057 | 0.2200 | 0.6711 | 0.6974 |
| `ttb_ctrl_b` | 135 | 8.8293 | 0.7410 | 0.3299 | 0.2150 | 0.5789 | 0.6184 |
| `ttb_treat_a` | 135 | 5.7411 | 0.0000 | 0.0062 | 0.0004 | 0.0000 | 0.0000 |
| `ttb_treat_b` | 135 | 6.7098 | 0.0000 | 0.0041 | 0.0011 | 0.0000 | 0.0132 |

## 4. Pre-registered decision rules

Rules registered in `logs/proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md` §7, applied mechanically here. Effect sizes are read against the **seed spread**, not a p-value.

- **R1 — the treatment wins**: p(`</think>`) 0.0000 vs 0.7925 (control spread 0.1030) → inside spread; p(`<|im_end|>`) 0.0052 vs 0.3678 (control spread 0.0758) → inside spread; holdout NLL 6.2255 vs 7.7260 = -19.42% (band ±1%) → OUTSIDE.
  → **R1 does not fire.**
- **R2 — rejection**: holdout NLL -19.42% (reject if > +1%); `terminated` 0.0066 vs 0.6579 (control spread 0.0789) → REGRESSED.
  → **R2 FIRES: reject the treatment.** `terminated` regressed beyond the control's seed spread, which is the metric the exit gate is blocked on.
- R3 does not fire: the arms are separated by more than the seed spread on at least one probe, so the comparison is conclusive — in whichever direction R1/R2 report.
- **R4 — abort** is applied per-arm during the run by the orchestrator (val CE at the first eval above step 0, or non-finite loss). Arms with training logs: ttb_ctrl_a, ttb_ctrl_b, ttb_treat_a, ttb_treat_b.

**Context, not a decision input:** `behavior_score_v0` 0.0008 ±0.0004 vs 0.2175 ±0.0025, Δ=-0.2167 against the measured seed-only noise floor of 0.1290. This delta exceeds the noise floor, but the composite is still not a pre-registered readout for this experiment.

## 5. Declared asymmetry (P6)

Total training tokens are **identical** across arms (steps x blocks_per_step x block_len). Passes over the prompt set and supervised token counts are **not** equal, because teacher targets are several times longer than public ones on the same prompts. That is inherent to the comparison and is reported rather than engineered away (maintainer decision 2026-07-30). See the corpus section for the measured supervised-token counts per arm.


---

# 6. Reading this run (post-hoc analysis)

This is the **corrected baseline**: every arm forked from the Stage 1
structural-initialization checkpoint (`model.safetensors` sha256 `86fbba78…`),
before any Stage 3 training, so neither target set had a path-dependent
advantage. It supersedes the
[post-s2v1 continuation diagnostic](2026-07-30_stage3_post_s2v1_continuation_diagnostic.md),
whose arms both forked from a 2,700-step public-target checkpoint.

Read strictly as **an early fixed-compute comparison of two complete target
recipes from the common Stage 1 initialization** (maintainer, 2026-07-30). It is
not a convergence result and not a per-supervised-token comparison.

## 6.1 The shared step-0 model

Scored on the same GPU, in the same session, before any arm trained:

| readout | Stage 1 init (step 0) |
|---|---:|
| holdout NLL bf16 | 11.7565 |
| holdout NLL INT8 | 11.9286 |
| p(`</think>`) / p(`<|im_end|>`) | 0.0000 / 0.0000 |
| `format_ok` / `terminated` / `think_closed` | 0.000 / 0.000 / 0.000 |
| `empty_answer` / `truncated_at_cap` | **1.000** / **1.000** |

The bf16 figure reproduces the 2026-07-14 CPU measurement (11.748) to 0.07%,
confirming both the checkpoint identity and the eval path. The init generates
nothing usable: every prompt runs to the cap and yields an empty answer.

## 6.2 Result

| | step-0 | control (public) | treatment (teacher-native) | ext. ref `s2v1@2700` |
|---|---:|---:|---:|---:|
| holdout NLL | 11.7565 | 7.7260 ±1.10 | **6.2255** ±0.48 | 3.8285 |
| `format_ok` | 0.000 | **0.6250** ±0.046 | 0.0000 ±0.000 | 0.2237 |
| `terminated` | 0.000 | **0.6579** ±0.039 | 0.0066 ±0.007 | 0.3684 |
| `think_closed` | 0.000 | **0.9276** ±0.046 | 0.0066 ±0.007 | 0.6053 |
| `empty_answer` | 1.000 | **0.0724** ±0.046 | 0.9803 ±0.007 | 0.1711 |
| `truncated_at_cap` | 1.000 | **0.3421** ±0.039 | 0.9934 ±0.007 | 0.6316 |

*(± is half the two-seed spread.)*

**At 137 steps from the common init the two recipes diverge completely, and in
opposite directions on the two axes that matter.**

* **The public-target arm produces a usable model.** From an init that emitted
  nothing, it reaches `format_ok` 0.625 and `terminated` 0.658 — better on both
  than the external reference `s2v1@2700`, which took 2,700 steps on a 22M-token
  mixture (0.224 / 0.368).
* **The teacher-native arm produces better held-out language modelling and no
  usable generation.** Holdout NLL 6.23 vs 7.73, but **99.3% of generations hit
  the 512-token cap** and 98% yield no answer. It has learned to open a
  reasoning trace and has not learned to end one.

## 6.3 What is and is not resolvable here

* **The protocol difference is real and large**, not an instrument artifact of
  the *fork*: both arms started from an identical step-0 model that scored 0 on
  every protocol axis.
* **But the 512-token scorecard is fully saturated for the treatment arm**
  (`truncated_at_cap` 0.9934). Every protocol number for that arm is pinned at
  the truncation floor, so it measures "did not finish in 512 tokens" and
  nothing finer. The diagnostic hit the same wall at 84.2%; here it is total.
* **Holdout NLL is not resolvable at two seeds.** The control's own seed spread
  is **2.21 nats** (6.6227 vs 8.8293) — the arms differ by 1.50, which is
  *inside* that spread. From a cold init at this budget, seed noise on NLL
  dwarfs the effect; the diagnostic's arms agreed to 0.06% only because they
  started from a converged checkpoint. **Any NLL claim here needs more seeds.**
* **`p(</think>)` behaves exactly as §6.1 of the diagnostic predicted.** Both
  arms start at 0.0; the control rises to 0.79 because public targets carry an
  empty think block, and the treatment stays at 0.0000 because teacher-native
  targets never skip reasoning. It is measuring target style, not competence.

## 6.4 Verdict — **SUPERSEDED by the reclassification banner at the top**

The wording below was written before the maintainer reclassified this run. It is
kept verbatim because the measurements behind it stand; its *route-level*
reading does not. R2's trigger is a historical fact about a **capped,
non-converged** measurement, not a rejection of teacher-native supervision.

* **Pre-registered rule R2 fires at this fixed compute.** Unlike the earlier
  diagnostic, that trigger is not confounded by the start point — but it is
  confounded by a 99.3% censoring rate and by 5% of the reference step budget.
* **The honest statement of it:** at 137 steps from the Stage 1 init, on a
  487-prompt corpus, the public-target recipe yields a protocol-competent model
  and the teacher-native recipe does not yield a model that can finish an
  answer — while being the better language model on held-out text.
* **What it does not establish:** that teacher-native targets are worse. The
  treatment arm is 3.0 corpus passes into learning a 1,149-token-median target
  format; "has not learned to terminate yet" and "cannot learn to terminate"
  are not distinguishable at this budget, and the readout that would tell them
  apart is saturated.
* **Nothing here supports a per-supervised-token claim**: the treatment arm
  received 18.9× the supervised tokens, and still trails on every generation
  metric, which is itself informative about how much this student can absorb at
  this budget.

## 6.5 Next actions (superseded — see the audit directive)

Items 1-3 below are retained; the standing plan is now the maintainer's audit
and unrestricted re-evaluation directive of 2026-07-30, which supersedes any
"pick a bigger cap" framing: **no artificial generation cap at all** (P18).

1. **No cap may be used for formal measurement.** Generation runs to natural
   EOS / `<|im_end|>` or to the actual supported context, per sample.
2. **Holdout NLL needs >=4 seeds at this budget**, or it cannot separate arms
   from a cold init (2.21-nat control spread).
3. Protocol metrics reward terseness, and this design cannot separate
   "teacher-native" from "more supervised tokens".
