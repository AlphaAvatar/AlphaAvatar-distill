# 2026-07-30 — Stage 3 teacher-target SFT warm-up: 2×2 pilot (pre-registration)

**Status:** pre-registered, **nothing run, no pod created, no spend committed.**
Awaiting approval of scope, sequence-length handling, runtime and budget.

Rests on the CPU preflight
([log](../experiments/2026-07-30_stage3_target_preflight.md)), which is complete.

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
