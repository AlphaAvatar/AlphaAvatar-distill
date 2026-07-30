# 2026-07-30 — Stage 3 teacher-target SFT warm-up: 2×2 pilot (pre-registration)

**Status (amended 2026-07-30, second revision):** the design in §1–§9 was run
and is **superseded**. Its start point was invalid — see §10. The corrected
baseline is **§11**, which is what any future paid run implements.

The first execution ($4.87) is relabelled a *post-s2v1 continuation diagnostic*
([log](../../experiments/stage3/2026-07-30_stage3_post_s2v1_continuation_diagnostic.md));
its R2 "reject" outcome is void as evidence about teacher-native targets.

*Original status: pre-registered, nothing run, no spend committed.*

Rests on the CPU preflight
([log](../../experiments/stage3/2026-07-30_stage3_target_preflight.md)), which is complete.

## 1. Question

Does replacing public SFT targets with **teacher-native targets** improve the
student's protocol competence — the thing Stage 3's exit gate is blocked on —
at a fixed training-token budget?

This is the direct test of the standing hypothesis that *the teacher disagrees
with any target it did not write*, confirmed twice as a CE/KD conflict
(2026-07-28) at `</think>` and `<|im_end|>`. Teacher-native targets remove the
disagreement at its source instead of masking one span at a time.

## 2. Design — 2×2, two arms × two seeds

| | control | treatment |
|---|---|---|
| targets | **v1 public** | **teacher-native** (trace + answer) |
| seeds | 20260726, 20260728 | 20260726, 20260728 |

Held identical across all four runs: start checkpoint
(`stage3/s2v1_from_init/step_002700`), **prompt set** (the accepted subset, §3),
total training-token budget, packing, block length, optimizer, schedule, loss
weights, and evaluation.

**No prefill, no forced mode, no answer-length quality gate.** Targets are
accepted or rejected on *correctness* by `aadistill.verify` only; length plays
no part (P10).

## 3. Pilot corpus scope

* **Slices:** `rag_evidence`, `multihop_qa`, `gsm8k`, `openmath` — the in-scope
  four (scope frozen, preflight §1). No refusal data is generated.
* **Prompts:** **750**, slice-balanced (≈188 per slice), deterministic stride
  from `data/stage2_v1`, seed-free and reproducible from the manifest.
* **Candidates:** **n = 2**, both sampled, **no greedy candidate** — untruncated
  at temperature 1.0 / top_p 1.0 / top_k off (decision 2026-07-29).
* **Cap:** 4096 new tokens. Not raised: the 16,384 arm closed more traces but cut
  verified accuracy 0.750 → 0.294 and doubled cost per accepted target
  (decision 2026-07-30).
* **Engine:** **vLLM 0.26.0 provisionally**, in its official image on a CUDA-13
  host (`--min-cuda-version 13.0`). The final rollout-engine choice stays
  deferred to Stage 4/5; this is an offline Stage 3 build and its correctness is
  verified per candidate, so the engine's token identity is not load-bearing.
* **Snapshot:** exact token ids, per-token log-probs, policy/engine identity and
  sampling parameters, written to a **hashed artifact** before anything trains
  on it (P4/P5, `aadistill.rollout`).

**The 2×2 trains only on prompts where a teacher target was accepted.** Both
arms use that same subset — control takes those prompts' public targets,
treatment takes their teacher targets. This keeps the prompt set identical and
keeps the public-target fallback out of the treatment arm, which would otherwise
contaminate it with control data. At the pilot's measured per-slice accept@n,
750 prompts at n=2 should yield roughly **400–550** accepted; the exact number
is an output, and the run is sized from it.

## 4. Sequence-length handling — the preflight's binding constraint

**`best_fit` packing at `block_len` 8192, both arms.**

Proven on the real corpus: `truncated_samples = 0` and **41,276 / 41,276
supervised tokens preserved**. Bounded by construction: worst-case prompt 2,765
+ cap 4,096 + template overhead ≈ **6,925 < 8,192**.

The current recipe (`concat` @ 1024) is disqualified: 48.5% of teacher targets
exceed one block and the expected split rate is 79.9%, which trains the student
on continuations of premises it cannot see. The naive alternative, `best_fit` @
1024, is worse still — it silently discards **56%** of supervised tokens.

This supersedes the 2026-07-28 packing decision **for this experiment only**;
that decision measured public targets, where splitting is a rounding error.

**Prerequisite, not yet measured:** training throughput and memory at
`block_len` 8192 on the 0.6B student. Attention is quadratic in sequence length,
so batch size must fall and gradient accumulation rise to hold the token budget.
A short smoke test sizes this before the paid runs and is included in the budget.

## 5. Budget — identical across arms

**Total training tokens** (steps × batch × block_len) are held identical, which
is what a compute budget means. Passes over the prompt set will therefore differ
between arms, because teacher targets are 4.2× longer.

**Supervised tokens will not be equal and that is reported, not corrected**: the
supervised fraction is 0.687 for teacher targets against 0.547 for public ones,
so at equal total tokens the treatment arm receives ~26% more supervised tokens.
Engineering that away would require unequal compute, which is a worse confound.

## 6. Primary readouts

Protocol competence, exactly as the maintainer specified:

`format_ok` · `think_closed` · `terminated` · `empty_answer` ·
**p(`</think>`)** · **p(`<|im_end|>`)**

with **holdout NLL as a ±1% guard rail** (a large drop aborts).

Reported per arm as a **mean over two seeds with the spread**, never as a single
run: the measured seed-only noise floor on `behavior_score_v0` is **0.1290**,
wider than any inter-arm difference this project has reported, which is why two
seeds per arm is a standing rule and why the composite score is *not* a primary
readout here. The two probe metrics — `p(</think>)` and `p(<|im_end|>)` — are the
sharpest instruments available: the CE/KD intervention moved p(`</think>`) from
0.2995 to 0.9989 where NLL moved 0.36%.

## 7. Pre-registered decision rules

* **R1 — the treatment wins** if it improves **both** `p(</think>)` and
  `p(<|im_end|>)` beyond the seed spread of the control, **and** holdout NLL
  stays inside ±1%.
* **R2 — the treatment is rejected** if holdout NLL degrades more than 1%, or if
  `terminated` regresses beyond the seed spread. Termination is what Stage 3's
  exit gate is actually blocked on.
* **R3 — inconclusive** if the arms overlap within seed spread on the probes.
  Recorded as such; a larger corpus is then the lever, not a rerun.
* **R4 — abort** any run whose primary-val CE at the first eval exceeds its
  step-0 value, or on non-finite loss.

Effect sizes are read against the **seed spread**, not a p-value, and the
composite behavior score is reported only as context.

## 8. Cost

| item | estimate | cap |
|---|---|---|
| block_len 8192 throughput smoke (0.6B) | ~0.3 h | — |
| corpus generation, 750 prompts × n=2, vLLM 0.26.0 | **$1.0–3.0** | `--max-hours 2.5` |
| 2×2 training, 4 runs | **$2.0–3.0** | per-run step cap |
| **total** | **$3.0–6.0** | **hard ceiling $7.00** |

The generation range is wide for an honest reason: vLLM was measured at **247.5
tok/s at concurrency 8**, and 8 prompts do not saturate it. A 750-prompt build
submits far more concurrently, so real throughput should be materially higher
and the cost nearer the bottom of the range. The ceiling assumes it is not.

Training cost is the least certain line: no run has been done at `block_len`
8192, which is why the smoke test comes first and why the maintainer may prefer
to approve generation and training as two separate gates.

## 9. Out of scope

* No full corpus — this is a 750-prompt pilot.
* No Stage 4/5, no corrected-training pilot, no rollout-engine adoption.
* No refusal data and no alignment-oriented slice.
* The openmath cap stays 4096.

---

# 10. Why §1–§9 was invalid: the start point

The design above started both arms from
`stage3/s2v1_from_init/step_002700`. That checkpoint is **2,700 optimizer steps
of training on public targets** (mixture v1). Forking a public-target control
and a teacher-native treatment from it does not compare the two target sets: it
compares them *conditioned on 2,700 steps of one of them*.

Concretely, the control resumes inside its own target distribution — same answer
style, same empty-`<think>` convention, same length profile — while the
treatment must move away from it. Any measured difference confounds "which
target set is better" with "which target set this checkpoint was already trained
on". No re-scoring, cap change or metric substitution fixes a path-dependent
start; the arms have to be forked from a point neutral to both.

This is independent of, and more fundamental than, the instrument problems the
run also uncovered (invalid `p(</think>)` siting, saturated generation cap,
protocol metrics that reward terseness). Those remain real and are carried into
§11 as constraints on the readouts.

**Consequence:** the completed run is a post-s2v1 continuation diagnostic. Its
R2 outcome is not evidence for or against teacher-native training.

# 11. Corrected baseline (to be run; supersedes §2 and §4's start point)

## 11.1 What changes

| | superseded (§1–§9) | corrected |
|---|---|---|
| start point | `s2v1_from_init/step_002700` (2,700 public-target steps) | **Stage 1 structural init**, before any Stage 3 training |
| what it answers | short post-s2v1 continuation | **teacher-native vs public from the common student initialization** |
| `step_002700` | initialization | **external reference only, never an initialization** |

Everything else is deliberately unchanged: the accepted paired corpus, the
train/val split, `best_fit` packing at `block_len` 8192, the trainable parameter
set, block-ordering rules, loss weights, and per-arm parity of the total
training-token budget. **Only the assistant turn of each sample may differ
between arms.**

## 11.2 Start point, pinned

`artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, on the relay at
`stage1/qwen3_0p6b_init_v0/checkpoint/`. Verified identical local vs relay:

| file | sha256 |
|---|---|
| `model.safetensors` | `86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54` |
| `config.json` | `a7131bb092b38a078edc213961f0eb57eaead24f1396e25741f4887b1a694054` |
| `tokenizer.json` | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |
| `tokenizer_config.json` | `8fa82a4ba512c8bee7c1c5e82b9a71ddbef362e4665be5c8f7ce0afd78af129a` |
| `generation_config.json` | `0019fccc989feeebf6d72934e1f6b917b320cc61f294dca1e562bce4c9cf5f83` |
| `chat_template.jinja` | `3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7` |

This is the Stage 1 PCA/sandwich init ([log](../../experiments/stage1/2026-07-14_stage1_qwen3_0p6b_init_v0.md)),
**not** Stage 3 `s1@660` and **not** `step_002700`.

## 11.3 Optimizer and scheduler reset — verified, not assumed

Launching an arm from `student_path` **without** `--resume` already resets both:

* `AdamW` is constructed fresh in `Trainer.__init__` (`src/aadistill/training/train.py:426`);
* `restore()` loads optimizer state **only** on the `--resume` path (`:640`);
* the LR is a pure function `lr_factor(step, total_steps, warmup_steps, min_lr_frac)`
  (`:315`) — there is no scheduler state object, and `step` starts at 0.

So the reset requirement is satisfied by construction, provided **no arm is ever
launched with `--resume`** except to recover an interrupted arm of the same run.

## 11.4 Shared step-0 model

All four arms share one step-0 model, so it is evaluated **once** and recorded as
the common baseline, on the same GPU in the same session as the arms.

Known already (CPU, 2026-07-14): `holdout_v1` NLL **11.748**, PPL 126.5K,
top-1 0.0175, against random init 12.13 and teacher 2.63.

Not yet measured and required at step 0: `eval_behavior_v0` (`format_ok`,
`terminated`, `think_closed`, `empty_answer`, `truncated_at_cap`), the
`p(</think>)` / `p(<|im_end|>)` probes, and INT8 holdout. These are what make the
per-arm deltas readable from a common origin.

## 11.5 Arms

Four arms, one variable plus seed:

| arm | targets | seed |
|---|---|---|
| `ttb_ctrl_a` | public v1 | 20260726 |
| `ttb_ctrl_b` | public v1 | 20260728 |
| `ttb_treat_a` | teacher-native | 20260726 |
| `ttb_treat_b` | teacher-native | 20260728 |

Generated by `scripts/data/make_tt2x2_configs.py`, which asserts the four configs
differ only in `data_dir`, `seed`, `run_name`, `out_dir` and a purpose note.

## 11.6 Corpus, split and exposure — unchanged and shared

540 accepted prompts, **487 train / 53 val**, identical prompt sets and identical
split membership in both arms (`sha256(id)`-derived, seed-free, arm-independent).

| | control | treatment |
|---|---:|---:|
| train prompts | 487 | 487 |
| train blocks @ `best_fit` 8192 | **36** | **91** |
| real tokens / epoch | 263,840 | 714,941 |
| supervised tokens / epoch | **27,526** | **519,478** |
| supervised fraction | 0.0929 | 0.6591 |
| packing efficiency | 0.895 | 0.959 |

## 11.7 The confound this design still cannot remove

At equal total training tokens the treatment arm receives **18.9×** the
supervised tokens (519,478 vs 27,526 per epoch). A public target for these
slices is a few tokens after an empty think block; a teacher-native one is a
full trace. **The extra supervision is not a nuisance — it is inseparable from
the intervention**, because a public target cannot be given a reasoning trace.

From the corrected common initialization this matters *more* than it did from a
trained start: an arm with 18.9× the gradient signal should move further on
almost any axis, so a treatment win is close to guaranteed and would not by
itself establish that teacher-native targets are better *per supervised token*.

The corrected baseline is still worth running — it removes a confound that
currently makes the comparison unreadable in the *opposite* direction — but its
result must be reported as "teacher-native targets vs the best available public
alternative at equal compute", never as "traces beat short answers per token".
Separating those needs a third arm (e.g. public targets at matched supervised
tokens), which is explicitly **out of scope** until this baseline exists.

## 11.8 Readout constraints carried over from the diagnostic

* **`p(</think>)` is not a valid protocol readout here.** It is measured where
  the *public* render demands the token, so it scores "skip reasoning entirely".
  Report it as a descriptive statistic only, or re-site it on teacher-native
  renders; it must not gate a decision in its current form.
* **The generation cap must not saturate either arm.** At 512 the treatment arm
  truncated 84.2% of prompts. Score at a cap where `truncated_at_cap` is
  reported for both arms, and report every protocol metric **both** raw and
  conditioned on generations that finished.
* **Protocol metrics must be paired with an answer-quality/length axis.**
  `format_ok` and `terminated` are maximised by a 2-word answer, and the
  control's median finished answer was 2 words. P10 forbids reading that as a
  win.

Decision rules are **not** re-registered here: the maintainer has directed that
metric design is settled after this baseline exists, not before it.

## 11.9 Out of scope until this baseline exists

No LR sweep, no step-budget ablation, no metric redesign, no final-only or
trace-length variants, no new corpus, and no regeneration of the corpus.

---

# 12. Teacher-mode-preserving programme (proposed; nothing run)

Supersedes §11 as the standing plan. Grounded in the unrestricted pilot
([log](../../experiments/stage3/2026-07-30_unrestricted_pilot.md)), the
structural audit and the system-prompt audit. **Every item preserves the
thinking-only protocol** (P17): no no-think, empty-think, final-only or
shortened targets appear anywhere below.

## 12.1 What the evidence forces

`s2v1@2700`, the best checkpoint, degenerates on 8/8 uncapped prompts. So the
first question is **not** "which target set" but **"can this student reach
non-degenerate free generation at all, in teacher mode"**. Until one checkpoint
does, target-set comparisons are comparisons between degenerate models.

## 12.2 Ordered programme

**A. Convergence probe (highest value, cheapest).** Take the teacher-native
recipe and extend exposure until free generation stops degenerating or provably
plateaus. Measure the uncapped 8-prompt probe at every checkpoint — it costs
~$0.02 per checkpoint and is the only metric that has distinguished anything.
This directly tests "insufficient optimization", the top-ranked cause.

**B. Data coverage.** 487 prompts / 0.71M real tokens is the binding limit for A.
Required coverage is estimated in §12.4. Regeneration must be **uncapped** (or
capped far above the teacher's p99 of 3,854) so openmath's long-reasoning
instances stop being systematically rejected.

**C. Exposure-bias correction**, only after A shows whether it is needed:
sequence-level / on-policy GKD, teacher correction, or self-distillation on the
student's own prefixes. The teacher-forced-vs-free gap (NLL 11.76 → 6.23 while
generation degenerates) is the textbook indication.

**D. Representation-level distillation.** Add hidden-state / span KD to the
current CE + logit KD, which is the cheapest capacity-side lever that does not
touch the behavioural target.

**E. System-prompt capability — a NEW objective, not a repair.** Rechecked
2026-07-31: the teacher requires no system message, the template has no
default-system fallback, and all four in-repo paths share a byte-identical
prompt prefix, so the corpus is native-correct and nothing needs correcting
([contract](../../experiments/stage3/2026-07-31_system_prompt_contract.md)).

If system-conditioned deployment is wanted, it is an **added capability** with
its own objective: generate teacher targets **conditioned on** varied,
representative system prompts sampled from a pool (never one static string, never
retrofitted to existing targets), keep a **no-system control** to isolate the
effect, and add system-instruction adherence to evaluation. Deprioritised below
A-D: it cannot help a model that degenerates without a system prompt.

**F. Architecture / quantization / runtime** — the P10 route to realtime, kept
last because it is irrelevant until free generation is stable.

## 12.3 Convergence criteria and stopping rules

Primary (uncapped probe, teacher mode):

* `natural_termination_rate` — target **≥0.8**, matching the teacher's own 80.1%;
* `degeneration_rate` — target **≤0.05**;
* generated-length p50 within **0.5×–2×** of the teacher's 727.

Secondary: holdout NLL (guard rail only — it improved monotonically while
generation degenerated, so it must never be the adoption metric again);
answer correctness; system-instruction adherence.

**Stopping rules.** Stop a run when: the probe's `degeneration_rate` fails to
improve across 3 consecutive checkpoints; or `natural_termination_rate` ≥0.8 with
`degeneration_rate` ≤0.05; or the pre-registered token budget is exhausted; or
val CE rises above its step-0 value (existing R4). Report ≥2 seeds before any
behaviour claim — and ≥4 for holdout NLL, whose cold-start seed spread was 2.21
nats.

## 12.4 Required data and token coverage (estimate, to be approved)

The reference run needed 22.1M tokens to reach NLL 3.83 and is still degenerate.
A convergence test on teacher-native data therefore needs, at minimum, the same
order of token exposure **without** relying on repetition:

| | current | needed for a convergence test |
|---|---:|---:|
| accepted prompts | 487 | **~6,000–8,000** |
| real tokens (treatment) | 0.71M | **~10–15M** |
| passes at 22M-token budget | 31× | **~2–3×** |

Generation cost at the measured 5.8 s/prompt (n=1, uncapped ≈ same, since 80%
terminate naturally): ~8,000 prompts ≈ **13 h ≈ $13** on one L40S. This exceeds
the remaining authorization and is the main thing needing approval.

## 12.5 Cost, checkpoints, and what is NOT proposed

| item | estimate |
|---|---:|
| A. convergence probe on existing corpus (≈1,000 steps, 2 seeds) | ~$4–6 |
| uncapped probe per checkpoint | ~$0.02 |
| B. corpus expansion to ~8,000 prompts | ~$13 |
| C/D. exposure-bias + representation KD pilots | scoped after A |

Checkpoints every 100 steps, probe at each, keep the best two by
`degeneration_rate`.

**Not proposed, deliberately:** no bulk regeneration before A reports; no
76-prompt full evaluation before the probe is trusted; no LR/metric sweep; and
no target-behaviour change of any kind.
