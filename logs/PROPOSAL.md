# Active proposal — recovery-data scaling study

**Status 2026-08-01:** Step A (corpus + ladder) is **done, gate-passed and
paid** — $26.59 of the $50 generation cap. Step B (the training matrix) is
**specified, costed and not started**; it needs a maintainer go-ahead against
the **$60** training budget. Supersedes the 2026-07-31 version of this file,
whose sizing assumptions were replaced by measurements (git history).

**Experiment order (maintainer, 2026-08-01).** Three questions, run in this
order, one at a time:

| # | question | status |
|---|---|---|
| **1** | **Does the student's behavioural recovery scale with teacher-generated supervised tokens?** | ready to run — §4 |
| 2 | Does the *mixture* of that data matter (capability-gap weighting vs uniform)? | later — §8 |
| 3 | Does *ordering* it by difficulty (curriculum) help? | later — §9 |

Experiment 1 therefore trains on a **uniform six-way type mixture**, held
constant across every rung. Size is the only variable; the capability-gap
weighting built into the original ladder cut is deferred to Experiment 2.

Written to the maintainer's instructions of 2026-07-31 and the four 2026-08-01
decisions. No additional experiments, comparisons or side investigations.

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

The maintainer added a second axis on 2026-07-31: the same relationship measured
against **Stage 1 initialization quality**, since a different init may need a
different amount of data.

Everything else about the data — which types, in what proportion, in what order —
is held fixed and neutral, and is the subject of Experiments 2 and 3.

## 2. Fixed protocol (not under test)

* **System message mandatory**, exact default:

  ```
  You are a helpful Assistant.
  ```

  Applied to teacher generation, student training, evaluation and inference.
  Teacher targets were generated **under this condition**; a system message is
  never prepended to an existing target.
* Thinking protocol preserved; `<think>` opened by the template.
* Stop ids from the model's `generation_config` (teacher `[151645, 151643]`).
* Evaluation generation is **uncapped** (P18): allowance = 262,144 − prompt,
  stopping on native EOS, the context limit, or detected degeneration.
* The 540 no-system corpus-v1 targets are retained as **labelled auxiliary data
  only**; they do not enter the primary distribution.

## 3. Step A — corpus and ladder · **DONE 2026-08-01**

Built with the official preset `0.6 / 0.95 / top_k 20 / min_p 0`, `n=4` with
per-candidate seeds, 8,192-token end-to-end session limit, turn expansion for
multi-turn sources, six types.

**Result:** 11,574 examples → 11,174 accepted (96.5%), 66.08M generated tokens,
16.5 h on one L40S, $25.56, gate PASSED. Full result, per-type table, correctness
verdicts and the reproducibility gap: [`EXPERIMENTS.md`](EXPERIMENTS.md) §10.
Design rationale: the four 2026-08-01 [decision records](decisions.md).

**The ladder cut is a free parameter, and Experiment 1 uses the uniform one.**
The corpus is packed once and cut into nested rungs; the type mixture is chosen
at cut time, not at generation time, so re-cutting costs nothing.

| cut | corpus supervised | blocks/epoch over 6 rungs | ceiling | used by |
|---|---:|---:|---:|---|
| **uniform 16.67% × 6** | 10,753,933 | **7,337** | **6.08M** | **Experiment 1** |
| capability-gap weighted | 10,805,451 | 6,907 | 10.81M | Experiment 2 |

**Three things Step A settled:**

1. The rungs are **measured, not estimated**: 0.25M / 0.46M / 0.86M / 1.60M /
   2.96M / 5.50M supervised tokens, all reachable under either cut.
2. Nesting is exact and monotonic, and each rung realizes its declared mixture
   within **0.3 pp** (uniform, worst case at the smallest rung; 0.03 pp at the
   top).
3. Packing efficiency at the top rung is **0.34**, because tool schemas render
   into the system block and the system prompt is a hard packing boundary. The
   maintainer kept the rule and raised the training budget to $60.

**One constraint the uniform cut imposes:** the corpus supports at most
**6.08M** uniform supervised tokens, bound by `multihop_qa` (1,012,726
post-packing). Saturation rungs meaningfully above 5.50M are therefore not
available to Experiment 1 — they need either more `multihop_qa`/`tool_calling`
generation or a non-uniform mixture, which is Experiment 2.

## 4. Step B — the training matrix (the thing awaiting approval)

**The measured variable is supervised tokens** — the tokens that actually carry
loss, and therefore the recovery signal whose required quantity we are trying to
predict. Rendered, real and padding tokens are recorded at every rung so the
relationship can be re-expressed in whichever unit turns out to generalise.

**Matrix: 6 rungs × 2 seeds × 2 inits = 24 runs.** Two seeds because behaviour
claims need ≥2 (noise floor 0.1290); two inits because initialization quality is
a declared axis (`checkpoint` sha `86fbba78…` vs `random_baseline` sha
`0e2e2b28…`).

**Design (i), fixed passes:** 3 epochs per rung, so each point is trained to
comparable exposure and the curve measures *data quantity*. The alternative —
fixed optimizer budget per point — conflates "more data" with "fewer passes",
which is what made the 137-step runs uninterpretable.

Rungs as cut for Experiment 1 (uniform mixture):

| rung | actual supervised | blocks | sessions | steps/run @ 3 epochs, 2 blocks/step |
|---:|---:|---:|---:|---:|
| 0.25M | 252,985 | 216 | 479 | 324 |
| 0.46M | 460,088 | 380 | 848 | 570 |
| 0.86M | 864,750 | 682 | 1,502 | 1,023 |
| 1.60M | 1,600,353 | 1,174 | 2,649 | 1,761 |
| 2.96M | 2,960,507 | 1,944 | 4,524 | 2,916 |
| 5.50M | 5,501,372 | 2,941 | 7,350 | 4,412 |

**Readouts per rung**, uncapped and system-conditioned on a fixed held-out
prompt set:

* `natural_termination_rate` — target ≥0.8 (the teacher's own rate)
* `degeneration_rate` — target ≤0.05
* generated-length p50 against the teacher's 727
* holdout NLL (guard rail only)

**Output:** the convergence curve of each readout versus supervised-token count,
the token count at which it saturates, and the fitted form — the first point in
a relationship intended to predict required recovery tokens from teacher
architecture/scale and student size.

## 5. Cost, and the cheaper cuts

Priced from measured rates: ~4.3 s/step at `block_len` 8192 with gradient
checkpointing, ~10 min of gate evals per checkpoint, $0.99/h L40S.

| item | detail | cost |
|---|---|---:|
| full matrix | 7,337 blocks × 3 epochs × 4 arms ÷ 2 = 44,022 steps × 4.3 s ≈ 52.6 h | **~$52** |
| gate + uncapped evals, 24 checkpoints | ~10 min each | **~$5** |
| **total** | | **~$57** |

That is the whole raised budget with almost no margin — the uniform cut costs
**+6.2%** over the weighted one (7,337 blocks/epoch vs 6,907), because it raises
the share of the badly-packing `tool_calling` type from 15% to 16.7%.
Projections for cutting it, computed from the same measured block counts:

| cut | runs | training | what is lost |
|---|---:|---:|---|
| **A — full matrix** | 24 | ~$52 | — |
| **B — drop the init axis** | 12 | ~$26 | the 2026-07-31 second axis |
| **C — drop the top rung (5.50M)** | 24 | ~$31 | the saturation end of the curve, where the answer probably is |
| **D — one seed** | 12 | ~$26 | any behaviour claim (noise floor 0.1290 > plausible effects) |

Recommendation: **A**, or **C** if the budget must hold — the init axis is a
declared objective and one seed cannot support a behaviour claim, whereas the
top rung can be added later from the same pack.

## 6. Prerequisite before any paid step

**Persist corpus v2 to the HF relay.** It exists only in a `/tmp` session
scratchpad (`STATE.md` §2). It is $25.56 and 16.5 h of teacher generation with
no second copy.

## 7. Stopping rules

* Stop a rung when val CE at the first eval exceeds its step-0 value, or on
  non-finite loss.
* Stop adding rungs when `degeneration_rate` ≤0.05 and
  `natural_termination_rate` ≥0.8 are reached — that is the answer to "how much
  data is needed".
* Budget is checked before each rung, never mid-run.

## 8. Experiment 2 — data mixing (later, not now)

After Experiment 1 reports. Does the *composition* of the recovery data matter
at a fixed token count? The comparison already exists in cut form: the
capability-gap weighting (`gsm8k` 22 / `rag_evidence` 20 / `openmath` 17 /
`code` 16 / `tool_calling` 15 / `multihop_qa` 10) against the uniform cut, at
one or two rungs on the Experiment-1 curve, same seeds and inits.

It is cheap for a second reason: both ladders come from the same corpus, so the
only cost is training. It also owns the question of **saturation rungs above
5.50M**, which uniform cannot reach (§3).

## 9. Experiment 3 — difficulty curriculum (last)

After Experiment 2. Order training samples by difficulty estimated from
**top-n candidate diversity** — the disagreement among the n=4 sampled teacher
candidates for the same prompt. Corpus v2 has per-candidate seeds and stores all
four candidates, so the diversity signal exists in the data already; corpus v1
could not support this (effectively n=1).

Ordering interacts with packing: the ladder is currently a seed-free stratified
interleave, and any curriculum order changes which sessions share a block. That
has to be handled before the experiment, not during it.

## 10. What will not happen

No experiments beyond those above. No metric redesign, no LR/step sweeps, no
architecture changes, no engine comparisons, no system-prompt ablation, and no
side investigations. Anything not listed here requires explicit instruction.
