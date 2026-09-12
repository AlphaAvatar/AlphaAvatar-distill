# Experiment 5 — proposal: matched-budget teacher-prefix continuation (C) vs student-prefix recovery (R)

> **STATUS: APPROVED AND REGISTERED 2026-08-06** — [`logs/e5_registration.json`](e5_registration.json).
> $6.00 additional authorized, $8.11 available, $7.92 backstop. Arm C is **built
> and validated offline at $0**. No paid teacher data generated, no GPU allocated.
>
> **Two design points changed at approval and supersede §4–§5 below:** C is now
> structurally symmetric with R (prefix + supervised continuation, not unsplit
> continuation training), and R's accurate name is **student-prefix on-policy KD
> plus teacher recovery continuation**.

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

* **C — teacher-prefix continuation, structurally symmetric with R.** For each
  matched prompt/truncation instance, C conditions on a **teacher-native prefix**
  and supervises the corresponding teacher continuation. It is *not* ordinary
  unsplit full-completion training: that would confound the state distribution
  with the prefix/continuation shape, which is the very thing R changes.
  C needs **no generation** — truncation only moves the loss mask over tokens
  already in the corpus.
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

**Superseded at approval — `k = 0` is now forbidden.** The registered guard is
*no empty assistant prefix*, so every example must carry at least one student
token. The rule is:

* let `L` be the student's generated span; `k ≥ 1` always;
* the cut fraction is drawn deterministically from the sample's identity hash and
  clamped into `[1, L − 8]`, so the continuation is always at least 8 tokens and
  the cut never lands at or past the end of the answer;
* the prefix never ends on a stop token — that would ask the model to continue
  past its own `<|im_end|>`;
* the two truncations of one rollout are never identical;
* a sample whose total length exceeds the context budget is **rejected
  deterministically before packing**, never silently cut by the packer.

**Consequence, recorded rather than hidden:** a rollout that generated *nothing*
cannot yield a legal prefix and is rejected as `no_supervised_tokens` /
`too_short_to_split`. Since `non_empty` was the failure scale repaired most often
(50 of 88), R's corpus will **under-represent the empty-output failure mode**
relative to its prevalence in P2-0.86M. The rejection census reports exactly how
many prompts this costs, by task and source seed.

Cutting at a deterministic fraction of the span is deliberate: the student's
error distribution is not concentrated at one depth, and a fixed cut point would
teach recovery from only one kind of state.

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

**LOCKED AT APPROVAL: keep `kd_scope: all` in both arms**, preserving the P2
objective semantics.

For R this means **KD is applied on the student-generated prefix states as well
as on the teacher recovery continuation**, while CE remains limited to the
continuation. R is therefore named exactly:

> **student-prefix on-policy KD plus teacher recovery continuation**

It must **not** be described as pure continuation-only recovery, and a result
must **not** later be attributed exclusively to the recovery continuation. The
treatment jointly tests learning on student-visited states and teacher-guided
recovery from those states.

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


---

## 9. Approval addendum (2026-08-06) — what changed, and what is now built

**C is structurally symmetric with R.** Both arms are prefix + supervised
continuation; CE applies to the continuation only in both; `kd_scope` is `all` in
both; both use two truncations per source prompt; and packed blocks, optimizer
steps, LR, scheduler, objective weights and trainable parameters are matched.
**The intended sole difference is the prefix state distribution** — teacher-native
(C) versus student-visited (R).

**Prefix lengths are matched on *fraction*, not on absolute tokens.** A teacher
target averages 641 supervised tokens while a student rollout at this stage runs
far longer, so equal token counts would force the two arms to cut at completely
different relative depths. Paired C and R examples share seed material and
therefore cut at the **same relative depth by construction**; the residual
difference in absolute prefix tokens is measured and reported.

**Arm C is built and validated, at $0** (`artifacts/stage3/e5_arm_c/`):

| | |
| --- | ---: |
| source sessions (incremental slice) | 1,147 |
| examples (2 truncations each) | **2,292** of 2,294 |
| acceptance | **99.9%** (1 rejection: `duplicate_fractions`) |
| supervised continuation tokens | **904,597** |
| prefix tokens p25/p50/p75/max | 245 / 386 / 822 / 7,453 |
| continuation tokens p25/p50/p75/max | 94 / 197 / 399 / 6,629 |

This is the profile R's corpus is matched against, and it is the reason C was
built first: if the split path could not produce a sound corpus from data that
already exists, generating paid teacher data would have been premature.

**Truncation guards, all implemented and tested** (`prefix_split.py`, 20 tests):
no empty assistant prefix; continuation ≥ 8 tokens so no cut at or past the
answer end; the prefix never ends on a stop token; the two truncations of one
rollout are never identical; prefix and continuation lengths recorded per
example; oversized samples rejected deterministically **before** the packer can
cut them; and packing never truncates a prefix or a supervised continuation.

**Interpretation boundaries** are registered: C vs R is the primary causal
comparison; C/R vs the from-scratch P2-1.60M checkpoint is a **recipe**
comparison because continuation schedule and data order differ; per-seed recovery
data is intentionally on-policy with source-checkpoint identity stored per
sample; and **no promotion on improved usable rollout alone** — the question is
whether student-prefix training converts more prompts into *correct and
naturally terminated* trajectories.
