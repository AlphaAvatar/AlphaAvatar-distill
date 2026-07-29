# 2026-07-30 — Rollout engine comparison + rollout-correction experiment (pre-registration)

> **RETIRED UNRUN 2026-07-30 (maintainer).** This plan was built around **vLLM
> 0.11.0**, a build reached by pinning backwards until the engine fit this
> project's training image. That is not a measurement of vLLM: the gap to the
> current stable 0.26.0 spans scheduler, kernels, CUDA-graph behaviour,
> throughput, log-prob support and operational characteristics. The 0.26.0
> failure it rests on was **environment selection, not an engine result** — the
> host ran driver 570.124.06 / CUDA 12.8 against an engine targeting CUDA 13,
> and `runpodctl --min-cuda-version` was never passed.
>
> Superseded by
> [`2026-07-31_current_engine_benchmark.md`](2026-07-30_current_engine_benchmark.md).
> The **correction experiment** in §5 survives intact and is carried forward;
> only the engine/version/environment framing is retired.

**Status:** pre-registered, **not run, no spend committed.** Supersedes the
engine-selection half of
[`2026-07-30_isolated_engine_and_cap.md`](2026-07-30_isolated_engine_and_cap.md);
that proposal's openmath cap arm is complete and its conclusion stands.

Written after the maintainer retired exact-token-agreement as an adoption gate
and retired HF `model.generate` as the production rollout path (decision
2026-07-30, AGENTS.md §4.6).

## 1. What changed, and why this proposal exists

The 2026-07-30 session measured an isolated-venv vLLM 0.11.0 server at **5.29×**
the in-stack throughput and **0/8** greedy token agreement, and its
pre-registered rule R1 turned that into "HF wins, and owns Stage 4/5". That was
wrong twice over:

* **Token equality is not a prerequisite for on-policy training.** Production RL
  systems pair an inference-optimized rollout engine with a separate trainer,
  generate asynchronously, and correct the mismatch explicitly. The mismatch is
  a quantity to measure and bound, not a disqualification.
* **One measured alternative cannot select a standing backend.** SGLang was
  never actually tested — 2026-07-29 attributed its failure to a Python
  dependency conflict, but the real constraint turned out to be the host's
  CUDA-12.8 driver, which was never re-tested against SGLang.

So the engine question is **reopened**, and the gate is replaced.

## 2. Objective

Choose the project's **rollout service** — one efficient, isolated engine reused
across Stage 3 corpus builds, Stage 4 rollout collection and Stage 5 on-policy
training — by comparing at least two serious candidates and by measuring whether
each one's policy mismatch against the trainer can be **quantified and corrected
within a pre-registered stability bound**.

HF `model.generate` participates only as the **reference oracle** that the
mismatch is measured against, not as a candidate.

## 3. Candidates

| candidate | why | known constraint |
|---|---|---|
| **vLLM 0.11.0** (isolated venv, HTTP) | measured 5.29× / $2.27 per 1k; works on the current CUDA-12.8 driver | needs `transformers==4.57.1` pinned in its venv and `ninja` on PATH |
| **SGLang deterministic mode** | the only candidate offering batch-invariant kernels, and Qwen3 is named as supported; never actually tested | may need a CUDA-13 host; must not be installed into the project venv |
| *(HF in-stack)* | **reference oracle only** | not a candidate |

If SGLang cannot run on an available driver, that is a recorded result and a
third candidate is substituted rather than the comparison being skipped —
adopting on a single candidate is precisely the error being corrected.

## 4. Adoption criteria (replacing exact token agreement)

An engine qualifies only on all of these. They are ordered so a cheap
disqualification happens before an expensive one.

1. **Token-in / token-out transport** — token ids in, token ids out, no text
   round-trip anywhere. A text hop reintroduces retokenization drift.
2. **Exact recording of rollout token IDs**, snapshotted to a **hashed artifact**
   before anything trains on them (P4/P5).
3. **Rollout log probabilities available per token**, for the tokens actually
   emitted.
4. **Measured KL and importance-ratio distribution against the trainer policy**:
   recompute trainer log-probs on the recorded rollout tokens and report the
   per-token ratio distribution, its tail, and sequence-level KL.
5. **Bounded off-policy rate and staleness** — fraction of tokens outside a
   pre-set ratio band, and the policy/checkpoint version gap, both logged.
6. **Stable corrected training in a small Stage 4/5 pilot** — the load-bearing
   criterion, and the only one that cannot be inferred from the others.
7. **Throughput, cost, and operational reliability** under the intended
   workload, including restart behaviour and failure modes.

Exact token agreement is **retained as a diagnostic only**. It is informative —
median first divergence at token 260 is a useful prior on correction magnitude —
and it gates nothing.

## 5. The correction experiment

This is the part the project has never done and cannot skip.

**Setup.** Generate rollouts from a fixed student checkpoint on a fixed prompt
set with each candidate engine, recording token ids and rollout log-probs.
Recompute trainer log-probs on those exact token ids in the training stack.

**Measure.**

* per-token importance ratio `r = exp(logp_trainer − logp_rollout)`: median, p95,
  p99, and the fraction outside `[1/c, c]` for a pre-set `c`;
* sequence-level KL between rollout and trainer policy;
* how these degrade with **staleness** — the same rollouts scored against
  trainer checkpoints N optimizer steps later.

**Then train.** A short Stage 4/5 pilot with token-level importance-sampling
correction and clipping, against an uncorrected control on the same rollouts.

**Pre-registered stability bound.** The engine is adoptable if, with correction:

* training loss and the guard-rail holdout NLL stay within the band a
  same-seed in-stack pilot occupies (no divergence, no collapse);
* the clipped/rejected token fraction stays **below 5%** — above that the
  correction is doing so much work that the rollouts are effectively a different
  policy;
* the corrected run is **not worse** than the uncorrected control on the guard
  rail, which is the check that the correction machinery is not itself the
  problem.

These numbers are set now, before any data, and are the thing to argue with
later rather than reverse-engineer.

## 6. Architecture this commits to

Stage 4/5 is designed around **asynchronous generation with explicit
correction**, not synchronous in-process generation:

* rollout service runs separately, generating while the trainer trains;
* every rollout record carries: token ids, rollout log-probs, the policy
  identity and checkpoint version that produced it, sampling parameters, and the
  engine identity and version;
* bounded staleness with logged weight synchronization points;
* rollout snapshots are hashed artifacts, and the **snapshot is the ground
  truth** a correction term is computed against — not a re-derived policy, which
  this project has already measured to be irreproducible even greedily.

## 7. Prerequisite work, none of it yet built

* `aadistill.engines` returns tokens only. **A log-prob path is required** and
  is the gating item — criteria 3–6 are all blocked on it.
* A rollout-snapshot format (token ids + log-probs + policy version + engine
  identity), hashed.
* A trainer-side scorer that recomputes log-probs on recorded token ids.
* CPU-testable against a toy model before any GPU spend (P8).

## 8. Budget

Not yet costed, deliberately: the prerequisite work in §7 is CPU-side and must
land first, and its shape will determine the GPU session's size. **No spend is
committed by this document, and no bulk corpus is built** — the $2.27/1k figure
is real but the engine is unchosen, and a corpus is not worth building twice.

## 9. What this proposal does not decide

* It does not adopt vLLM. It records it as the first measured engine.
* It does not rank engines by throughput. Throughput is criterion 7 of 7, and a
  fast engine whose mismatch cannot be corrected is not adoptable at any speed.
* It does not revisit the openmath cap, which is settled at 4,096.
