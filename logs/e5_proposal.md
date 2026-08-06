# Experiment 5 — proposal: matched-budget teacher-prefix continuation (C) vs student-prefix recovery (R)

> **STATUS: PROPOSED, NOT REGISTERED, NOT LAUNCHED.** No GPU allocated, no paid
> teacher data generated. Everything below is offline analysis of retained
> artifacts plus a design and budget for approval.

## 1. What E4 established, and why it points here

E4 (§21) showed that scaling teacher-prefix data 0.86M → 1.60M **substantially
improves rollout stability** (+0.2000 usable, both seeds) and **does not
materially improve correctness** (0.1900 → 0.2000, inside the floor, one seed up
one down, `correct_given_usable` falling).

The per-prompt transition table makes the mechanism concrete
(`artifacts/audit/e5_transition_table.json`, pooled over both seeds, n = 300
prompt-arms, CPU, $0):

| P2-0.86M ↓ / P2-1.60M → | usable+correct | usable+wrong | unusable |
| --- | ---: | ---: | ---: |
| **usable+correct** | 32 | 16 | 4 |
| **usable+wrong** | 15 | 69 | 24 |
| **unusable** | 12 | **76** | 52 |

* **88 prompts became usable — and 76 of them (86.4%) are still wrong.** Scaling
  bought well-formed trajectories, overwhelmingly with wrong answers.
* **Correctness is churn, not stasis.** 15 prompts went wrong→correct while 16
  went correct→wrong. The flat aggregate hides a near-perfect swap.
* **28 prompts regressed out of usable** (4 + 24), so the +0.20 is a net of 88
  gained against 28 lost.
* **The dominant end state is `usable+wrong`: 161/300 = 53.7%.** The second is
  `unusable`: 80/300 = 26.7%.

What scaling repaired, by the failure the prompt originally showed:

| repaired | count | regressed into | count |
| --- | ---: | --- | ---: |
| `non_empty` (produced nothing) | 50 | `non_empty` | 22 |
| `protocol_valid` | 20 | `natural_termination` | 3 |
| `natural_termination` | 18 | `protocol_valid` | 3 |

**More teacher-prefix data mostly taught the model to emit an answer at all.** It
did not teach it to reach the right one. That is the gap E5 targets.

## 2. The question

At matched budget from the same starting checkpoints, does **student-prefix
recovery** supervision improve autonomous correctness more than **ordinary
teacher-prefix continuation**?

* **C — teacher-prefix continuation.** Continue P2-0.86M on the incremental
  0.86M→1.60M blocks exactly as they are.
* **R — student-prefix recovery.** Continue P2-0.86M on the same prompts, but
  each example is *the student's own partial trajectory* followed by the
  teacher's recovery continuation, with loss on the continuation only.

C is not a re-run of P2-1.60M and must not be described as one: P2-1.60M
interleaves all 1,174 blocks over three passes, while C sees 682 blocks three
times and then 492 blocks three times. Same total block presentations, different
order. C is the matched-budget control for R, and a curriculum-order datapoint
against P2-1.60M as a bonus.

## 3. Matched budget

Derived from the existing nested rungs, not chosen:

| quantity | value |
| --- | ---: |
| additional unique supervised tokens | **735,603** |
| additional blocks | **492** |
| optimizer steps (3 passes ÷ 2 blocks/step) | **738** |
| incremental sessions | **1,147** |
| mean supervised tokens/session | 641 |
| type mix | uniform, 16.4–16.9% across all six |

C and R match on: starting checkpoint (P2-0.86M-`sa`/`sb`), seeds, 738 optimizer
steps, supervised-token presentations, learning rate and scheduler continuation,
objective weights (ce 1.0 / kd 0.25, τ 1.0), full-rank Attention + FFN + Norm
training with embeddings and `lm_head` frozen, and the evaluation harness.

**The intended sole difference is the prefix/training-data distribution.**

**Leakage, verified offline:** the 1,147 incremental sessions have **zero overlap
with the 150-prompt evaluation battery** and are **absent from P2-0.86M's
training rung** — so R's prompts are new to the model that generates the prefixes.
All held-out and val splits remain excluded by the corpus's existing hashed
leakage filter.

## 4. Student-prefix construction — exact specification

**4.1 Prompt source.** The 1,147 incremental sessions. Excluded: the evaluation
battery, every reserved val/calib/holdout/behaviour split, and anything already
in the 0.86M rung.

**4.2 Prefix generation.** P2-0.86M-`<seed>` generates on each prompt under the
binding protocol — mandatory system message, `<think>` pre-opened by the template,
unrestricted allowance `context − prompt` at 8,192 (P18). Decoding uses the
official preset (`temp 0.6 / top_p 0.95 / top_k 20 / min_p 0`) with per-prompt
seeds, matching corpus v2, so prefixes reflect the deployment sampling
distribution rather than only the modal greedy failure. **Prefixes for seed `sa`
come from the `sa` checkpoint and likewise for `sb`** — mixing them would break
the matched design.

**4.3 Truncation.** For each prompt, take **two** independent truncation points
(so one prompt yields two training examples covering different student states):

* let `L` be the number of student reasoning tokens before `</think>`;
* if `L = 0` (the student emitted nothing — the single most common repaired
  failure, 50/88), then `k = 0`: the teacher writes the whole trajectory. This is
  a legitimate and important recovery case, not a degenerate one;
* otherwise draw `k ~ Uniform{1 … min(L, context − reserve)}`, `reserve = 2048`
  tokens kept for the teacher's continuation;
* **never cut inside a multi-token special sequence**; snap `k` back to the
  nearest token boundary that leaves `<think>` open and `</think>` unemitted.

Truncation is uniform over the reasoning span on purpose: the student's error
distribution is not concentrated at one depth, and a fixed cut point would teach
recovery from only one kind of state.

**4.4 Serialization.** Token-level concatenation, asserted exact:
`[system + user rendered by the chat template, `<think>` open] + [student tokens
1..k] + [teacher continuation]`. The chat template is **never** re-applied to a
multi-message list — doing so silently deletes earlier reasoning traces (the
finding that shaped the corpus builder, §10).

**4.5 Masking.** **Loss on the teacher continuation only.** The prompt and the
student prefix are context: masked out of CE, out of the supervised-token count,
and out of acceptance accounting. The student is never trained to reproduce its
own broken prefix.

**4.6 One open decision, flagged rather than silently taken.** `kd_scope: all`
applies KD at *every real position*, which in R includes the student-generated
prefix. Two defensible readings:

* **keep `all`** — preserves exact objective parity with C, and the teacher's
  distribution over student text is arguably the most informative signal in the
  example;
* **restrict KD to the continuation** — matches the CE mask, but makes R's
  objective differ from C's in a second way.

**Recommendation: keep `kd_scope: all`** for parity, and record that KD is
evaluated over student-generated context. The scope variant is the natural
follow-up, not part of this experiment.

## 5. Mixture: R should be 100% recovery data

Not a teacher/recovery blend, for three reasons:

1. **C already is the teacher-prefix arm.** A blend would put teacher-prefix data
   on both sides and shrink the contrast the experiment exists to measure.
2. **738 steps is a small budget.** A 50/50 blend halves the recovery dose and
   makes a null uninterpretable — was it the method or the dose?
3. **A collapse is cheap and informative.** If 100% recovery destabilizes, the
   training curve and the movement report say so within one arm.

Risk, stated up front: R never sees a clean teacher prefix during continuation,
which could drift the model off the teacher protocol. Mitigation is structural —
R starts from P2-0.86M, which was trained *entirely* on teacher prefixes, and the
protocol-validity component of `usable_rollout` will detect drift directly.

## 6. Projected budget — REQUIRES ADDITIONAL AUTHORIZATION

Every line from measured throughput: training 3.61 s/step (P2 at this geometry),
teacher generation 1,110 tok/s sustained (corpus v2 bulk), student generation
from E4's evaluation timings.

| phase | min |
| --- | ---: |
| setup + preflight | 12 |
| student prefix generation, 2 seeds × 1,147 prompts | 54 |
| teacher recovery generation, 4,588 examples | 98 |
| corpus build + pack + gate (CPU, on pod) | 12 |
| train C ×2 and R ×2, 738 steps each | 178 |
| evaluate 4 checkpoints on the pinned battery | 44 |
| movement, transfer, teardown | 25 |
| **TOTAL** | **423 min = 7.05 h → $6.98** |

| | |
| --- | ---: |
| projected (expected) | **$6.98** |
| proposed hard backstop, 480 min | **$7.92** |
| currently authorized remainder | **$2.11** |
| **additional authorization required** | **$6.00** |

Split: **training $2.94**, **teacher/student data generation $2.51**,
**infrastructure (setup, corpus build, evaluation, transfer, teardown) $1.53**.

**A cheaper variant exists and is offered, not assumed.** Dropping to one
truncation per prompt halves teacher generation (−49 min, −$0.81) and yields
~1,147 examples; whether that reaches 735,603 *continuation* supervised tokens
depends on the measured prefix/continuation split, which is unknown until the
prefixes exist. Reducing to one seed is **not** offered — E4's own conclusion
rests on both-seed consistency.

## 7. What this design cannot settle

* One truncation policy and one mixture. A null means "uniform-random truncation
  at 100% recovery under this budget did not help", not "student-prefix recovery
  does not work".
* The evaluation battery remains the 150 shared **training** prompts pinned to
  the 0.86M mask — recall-style autonomous behaviour, not held-out
  generalization. It is kept for comparability with every prior arm.
* C is a curriculum-ordered continuation, not a reproduction of P2-1.60M.
* Teacher recovery targets inherit corpus v2's known limitation: correctness is
  computed but not enforced, and `openmath` targets are ~38% correct.

## 8. Decision rules to register before launch

1. If **R** improves `correct_overall` over **C** on both seeds beyond the 0.0600
   floor, student-prefix recovery addresses the correctness gap that scale did
   not.
2. If R and C are tied on correctness but R improves `correct_given_usable`, the
   effect is on answer quality given a well-formed trajectory — report it, do not
   promote on it alone.
3. If R degrades `protocol_valid` or `usable_rollout` relative to C, recovery
   training costs protocol fidelity; report the Pareto tradeoff.
4. If neither arm moves correctness, the correctness gap is not addressable by
   *prefix distribution* at this budget, and the next lever is verification or
   reward, not data ordering.
5. No promotion on one seed alone.
6. Teacher-native CE and FineWeb NLL remain diagnostics and never select a
   winner (E4's standing lesson).
