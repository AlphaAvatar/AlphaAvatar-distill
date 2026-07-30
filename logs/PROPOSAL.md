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

**Step B — train at 1,000 / 2,000 / 3,000 / … / 8,000 accepted samples**, each
from the pinned Stage 1 init, identical recipe, one variable: corpus size.

**Readouts per size point** (uncapped, system-conditioned, on a fixed held-out
prompt set):

* `natural_termination_rate` — target ≥0.8 (the teacher's own rate)
* `degeneration_rate` — target ≤0.05
* generated-length p50 vs the teacher's 727
* holdout NLL (guard rail only — it improved monotonically while generation
  degenerated, so it never decides adoption)

**Output:** the convergence curve of each readout against corpus size, and the
fitted relationship plus the point at which the curve saturates.

**Seeds:** 2 per size point for behaviour (noise floor 0.1290). Holdout NLL is
not seed-resolvable from cold start at 2 seeds (2.21-nat spread) and is reported
with that caveat rather than seeded to 4, which would double the cost.

## 4. Experiment 2 — difficulty curriculum (later, not now)

After Experiment 1 reports. Order training samples by difficulty estimated from
**top-n candidate diversity** — the disagreement among n sampled teacher
candidates for the same prompt. This requires the corrected per-candidate seeds
to be genuinely producing independent draws, which Experiment 1's generation
pass will establish as a by-product (it is not a separate experiment).

Note: the existing corpus cannot support this — it is effectively n=1, so its
measured diversity is zero everywhere.

## 5. Cost

| item | estimate |
|---|---:|
| Step A — generate 8,000 prompts, n=2, uncapped | **$13–17** |
| Step B — 8 size points × 2 seeds × ~137-step runs + evals | **$12–18** |
| uncapped evaluation probes | ~$0.5 |
| **total** | **$26–36** |

Priced from measured rates: 5.8 s/prompt at n=2, ~21 min per training arm
including gate evals, $0.99/h L40S. Uncapped generation is the widest
uncertainty because 19.9% of rollouts previously hit 4096.

**This exceeds the remaining $7.07.** Options, for the maintainer to choose:

| option | corpus | size points | cost |
|---|---|---|---|
| **A — full study** | 8,000 | 1k…8k (8 points) | $26–36 |
| **B — half study** | 4,000 | 1k…4k (4 points) | $13–18 |
| **C — within current budget** | 2,000 | 1k, 2k (2 points) | **$6.5–7.0** |

Option C fits $7.07 but yields only two points — enough to show direction, not
to fit a scaling relationship. Option A is what the objective actually requires.

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
