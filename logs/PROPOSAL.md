# Active proposal — recovery-data scaling study

**Status:** proposed 2026-07-31, **nothing run, no spend committed.**
Supersedes every earlier proposal (git history, commit `866dac2`).

Written strictly to the maintainer's instructions of 2026-07-31. No additional
experiments, comparisons or side investigations are included.

## 0. Cancelled

**The system-vs-no-system experiment is cancelled.** The system message is a
fixed project requirement, not an experimental variable. No budget will be spent
proving it.

Also dropped: the user-only convergence probe, corpus expansion as a separate
gate, exposure-bias and representation-KD pilots, metric redesign, LR sweeps.

## 1. Objective

Determine **how much teacher-native recovery data this student needs**, and
summarise the convergence pattern as a function of corpus size — working toward
a scaling-law-style relationship that estimates required recovery-data size from
teacher architecture/scale and student size.

## 2. Fixed protocol (not under test)

* **System message mandatory**, exact default:

  ```
  You are a helpful Assistant.
  ```

  Applied to teacher generation, student training, evaluation and inference.
  Teacher targets are generated **under this condition**; a system message is
  never prepended to an existing target.
* Thinking protocol preserved; `<think>` opened by the template.
* Stop ids from the model's `generation_config` (teacher `[151645, 151643]`).
* Evaluation generation is **uncapped** (P18): allowance = 262,144 − prompt,
  stopping on native EOS, the context limit, or detected degeneration.
* Existing 540 no-system targets are retained as **labelled auxiliary data
  only**; they do not enter the primary distribution.

## 3. Experiment 1 — corpus scaling (the one to run now)

**Step A — generate the largest recommended corpus in one pass.**

Recommended size: **8,000 prompts**, slice-balanced over the in-scope four
(`rag_evidence`, `multihop_qa`, `gsm8k`, `openmath`), system-conditioned, n=2
with the corrected per-candidate seeds, **no 4096 cap** (degeneration-stop
only, so openmath stops losing 69.7% of its instances).

Why 8,000: at the measured 71.8% accept rate this yields ≈5,700 accepted targets
≈ **9–12M real tokens**, the same order as the Stage 2 mixture that moved the
student from 11.75 to 3.83. Below ~2,000 prompts the largest arm cannot exceed
~2 corpus passes, which is where the current degeneration sits.

**Step B — train on progressively increasing TOKEN-COUNT subsets** (maintainer
correction 2026-07-31: the data-size variable is **token count**, not sample
count).

**The measured variable is supervised tokens** — the tokens that actually carry
loss, and therefore the recovery signal whose required quantity we are trying to
predict. Rendered and packed real tokens are reported alongside at every point so
the relationship can be re-expressed in whichever unit turns out to generalise.

Measured on the existing corpus, per accepted sample: **962 supervised tokens**,
1,460 rendered, 1,468 packed real (supervised fraction 0.659, packing efficiency
0.959 at `best_fit`@8192).

**Ladder — geometric, six points up to the corpus maximum:**

| supervised tokens | ≈ samples | blocks @8192 | real tokens |
|---:|---:|---:|---:|
| 0.25M | 259 | 47 | 0.38M |
| 0.50M | 519 | 94 | 0.76M |
| 1.00M | 1,039 | 187 | 1.53M |
| 2.00M | 2,079 | 373 | 3.05M |
| 4.00M | 4,158 | 746 | 6.10M |
| **5.50M** (max) | 5,717 | 1,025 | 8.39M |

Geometric rather than linear spacing: a scaling relationship is fitted in log
space, so evenly spaced *ratios* place points where the curve actually bends.
Linear spacing would cluster four of six points in the flat top decade.

**Subsets are nested and deterministic** — the 0.25M subset is a prefix of the
0.50M subset and so on — so the ladder is a clean scaling series rather than six
unrelated draws, and each subset stays slice-balanced so composition does not
drift with size (otherwise size and mixture are confounded).

**Readouts per token point**, uncapped and system-conditioned on a fixed held-out
prompt set:

* `natural_termination_rate` — target ≥0.8 (the teacher's own rate)
* `degeneration_rate` — target ≤0.05
* generated-length p50 against the teacher's 727
* holdout NLL (guard rail only)

Each reported against **supervised tokens**, with rendered and real token counts
recorded for re-expression.

**Output:** the convergence curve of each readout versus supervised-token count,
the token count at which it saturates, and the fitted form — the first point in
a relationship intended to predict required recovery tokens from teacher
architecture/scale and student size.

### 3.1 One decision this correction forces

Token-count subsets change what "one run" costs, because a bigger subset needs
more optimizer steps to be trained comparably. Two defensible designs:

* **(i) Fixed passes (3 epochs) per point** — each point trained to comparable
  exposure, so the curve measures *data quantity*. Steps scale with tokens: 71 →
  1,538. This is the standard shape for a data-scaling law and is what I
  recommend.
* **(ii) Fixed optimizer budget per point** — equal compute everywhere, so the
  curve measures *data diversity* at fixed compute. Cheaper and flat-cost, but it
  conflates "more data" with "fewer passes", which is what made the 137-step runs
  uninterpretable.

Costs below assume **(i)**.

## 4. Experiment 2 — difficulty curriculum (later, not now)

After Experiment 1 reports. Order training samples by difficulty estimated from
**top-n candidate diversity** — the disagreement among n sampled teacher
candidates for the same prompt. This requires the corrected per-candidate seeds
to be genuinely producing independent draws, which Experiment 1's generation
pass will establish as a by-product (it is not a separate experiment).

Note: the existing corpus cannot support this — it is effectively n=1, so its
measured diversity is zero everywhere.

## 5. Cost

Priced from measured rates: 5.8 s/prompt at n=2 generation; ~4.3 s/step training
at `block_len` 8192 with gradient checkpointing; ~10 min of gate evals per point;
$0.99/h L40S.

| item | detail | cost |
|---|---|---:|
| Step A — generate 8,000 prompts, n=2, uncapped | ~13 h | **$13–17** |
| Step B — 6 token points × 2 seeds, 3 passes each | 3,610 steps × 2 seeds ≈ 8.6 h + evals | **$10–13** |
| uncapped evaluation probes | ~$0.03 per checkpoint | ~$0.5 |
| **total** | | **$24–31** |

**This exceeds the remaining $7.07.** Options:

| option | corpus | token ladder | cost |
|---|---|---|---:|
| **A — full study** | 8,000 prompts (5.5M supervised) | 0.25M…5.5M, 6 points | **$24–31** |
| **B — half** | 4,000 prompts (2.8M supervised) | 0.25M…2.75M, 5 points | **$12–16** |
| **C — within current budget** | 2,000 prompts (1.4M supervised) | 0.25M / 0.5M / 1.0M, 3 points | **$6.5–7.0** |

Option C fits $7.07 and gives three log-spaced points — enough to see a trend and
its direction, not enough to fit a relationship with confidence. Option A is what
the scaling objective requires.

## 6. Stopping rules

* Stop a size point when val CE at the first eval exceeds its step-0 value, or on
  non-finite loss.
* Stop adding size points when `degeneration_rate` ≤0.05 and
  `natural_termination_rate` ≥0.8 are reached — that is the answer to "how much
  data is needed".
* Budget is checked before each size point, never mid-run.

## 7. What will not happen

No experiments beyond those above. No metric redesign, no LR/step sweeps, no
architecture changes, no engine comparisons, no system-prompt ablation, and no
side investigations. Anything not listed here requires explicit instruction.
