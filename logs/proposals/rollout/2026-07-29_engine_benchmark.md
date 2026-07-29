# 2026-07-29 — Engine benchmark, then teacher-corpus pilot (pre-registration)

**Status:** pre-registered plan. Nothing has been measured. Written before the
spend so the selection rules cannot be chosen after seeing the numbers (P6).

Supersedes nothing; it is the execution arm of the 2026-07-29 survey
([`2026-07-29_inference_engine_survey.md`](2026-07-29_inference_engine_survey.md)),
which deliberately selected no engine and measured nothing.

## 1. Objective

Choose the decode backend for the teacher-generated corpus, weighing
**throughput against integration cost** (maintainer directive, 2026-07-29) —
not throughput alone. Then, in the same session and automatically, build a
bounded pilot corpus with the winner.

## 2. Hypotheses

* **H1.** The in-stack HF path is materially slower than a serving engine at
  this job shape. *Unmeasured:* the 55 s/prompt figure that motivated this whole
  line came from `batch_size` 1 in `eval_behavior.py` and is not an engine
  limit. `src/aadistill/generate.py` has never been benchmarked at all, so H1
  could fail simply because batching was the missing speedup.
* **H2.** A serving engine's greedy tokens will **not** exactly match the
  training stack's. If true, an engine's throughput advantage is bought with a
  train/inference mismatch, which is the property Stage 4/5 cannot tolerate.
* **H3.** Batch invariance does not hold for the real 4B in bf16 on at least one
  engine, including possibly the incumbent. Verified so far only on a toy model
  in fp32, which is the friendly case.

## 3. Fixed budget (P6)

| item | value |
|---|---|
| hardware | 1× L40S (48 GB), pod-local disk, `--volume-in-gb 0` |
| price basis | $0.86/h (the `--hourly-usd` input to the $/1k column) |
| setup | ≤ 1.0 h (includes pod-side vLLM + SGLang install) |
| benchmark | ≤ 0.7 h |
| pilot generation | **hard cap `--max-hours 3.0`** |
| pod backstop | `--terminate-after` +6 h absolute UTC |
| **expected total** | **~$3.5–5.2** |

The generation cap is enforced in-process at the batch boundary
(`generate_teacher_answers.py --max-hours`), so a budget stop still writes
complete, hashed artifacts for every prompt finished. The corpus manifest
records `complete: false` when that happens — a truncated corpus is a valid
artifact but must not be mistaken for a full one.

## 4. Job shape

Deliberately *our* shape, not a leaderboard reproduction: unique prompts, no
shared prefix, long thinking traces, offline batch.

* teacher `Qwen/Qwen3-4B-Thinking-2507@768f209d`, bf16, **native thinking mode,
  never prefilled** (decision 2026-07-28);
* 32 prompts, slice-balanced by deterministic stride across the five verifiable
  slices (`rag_evidence`, `multihop_qa`, `refusal_uncertainty`, `gsm8k`,
  `openmath`) — balanced because trace length varies far more by slice than by
  engine, and a skewed sample would measure the slice;
* cap 4096 new tokens; greedy for every comparison;
* in-stack arm sweeps `batch_size` 8/16/32 (its real knob); vLLM and SGLang get
  the whole set at once, since they schedule internally.

## 5. Pre-registered decision rules

Applied mechanically by `bench_engines.py::decide` and written to
`decision.json`; `generate_teacher_answers.py --engine-from` consumes that file,
so **no agent interprets the numbers mid-session**.

* **R1 — agreement is a gate, not a tiebreak.** A non-incumbent arm is eligible
  only if its greedy tokens match the in-stack reference at
  **exact-match ≥ 0.90**. Below that it is a different policy from the trainer's;
  cheap wrong data is not a bargain.
* **R2 — among eligible arms, lowest $ per 1k prompts wins.**
* **R3 — a second stack must earn its keep (P1).** The winner must beat the
  incumbent by **≥ 1.5×** on cost. Below that, the incumbent stands. Ties go to
  the incumbent. This is the rule that encodes the maintainer's
  simplicity/efficiency balance: a marginal speedup does not pay for a second
  runtime, a second set of numerics, and a heavier dependency tree.
* **R4 — no reference, no decision.** If the `hf` arm fails, the session picks
  nothing and generation does not start. Throughput alone cannot be judged.

Reported alongside, but **not** part of the automatic rule: batch invariance per
engine, SGLang's deterministic-mode tax, peak memory, and setup wall time. These
inform the write-up and any follow-up; folding them into an automatic rule would
be pre-registering a threshold on a quantity nobody has ever measured here.

## 6. Abort rules

* **A1.** Any arm that raises is recorded with its traceback and skipped; the
  session continues. A wrong adapter guess costs one arm, not the session.
* **A2.** If `hf` fails, stop after the benchmark and upload the report. No
  corpus is built (R4).
* **A3.** If the pilot's accept@n on the first completed slice is **0.00**, stop
  and report rather than spend the remaining budget — that indicates a
  prompt-rendering or verifier fault, not a teacher quality result.

## 7. Pilot corpus

Runs only if a winner exists. 200 prompts/slice × 5 slices = 1,000 prompts,
`n=4` candidates, cap 4096, with unfiltered top-n selection deferred to
Stage 4/5 per the 2026-07-28 direction.

**All four candidates are sampled untruncated — temperature 1.0, top_p 1.0,
top_k off — and none is greedy** (maintainer, 2026-07-29; decision record same
date). With a verifier downstream this is rejection sampling, where candidate
diversity is what makes accept@n exceed accept@1. Note that `accept_at_1`
consequently changes meaning, from "greedy was accepted" to "one sample was
accepted", and is not comparable to any earlier figure.

The benchmark itself still decodes **greedily**, and that is not an
inconsistency: agreement and batch-invariance are only meaningful between
deterministic decodes. The benchmark measures engines; the pilot builds a corpus.

Its purpose is threefold and all three are prerequisites the queue already
lists: a real accept@1/accept@n rate, the per-slice divergence profile that sets
adaptive `n`, and a measured $/1k that finally prices the bulk build (previously
a guessed $25–145).

## 8. What is already verified, and what is not

**Verified on CPU (dev box, 191 tests passing):**

* the shared post-processing every engine goes through — stop-cutting, cap
  flags, stop-token re-appending so lengths are comparable across engines that
  do and do not return the stop token;
* the whole-prompt-echo strip, and that partial overlaps are *not* guessed at —
  a regression test pins this after the first implementation silently truncated
  a completion that legitimately began with the prompt's last token;
* SGLang's `output_ids` overlap resolved exactly from `completion_tokens`
  (sgl-project/sglang#10896), tested against a stubbed engine;
* the selection rules R1–R4 including their boundaries;
* the candidate→prompt mapping for `n>1`, where an off-by-one would attach one
  prompt's candidates to another's gold key and silently corrupt every verifier
  verdict downstream;
* in-stack batch invariance and input-order preservation on a toy model.

**Not verified, and cannot be on this box:**

* the vLLM and SGLang adapters have **never executed** — both are CUDA-only.
  They are written against documented APIs, defended against the two known
  hazards (removed `prompt_token_ids=` keyword; `output_ids` overlap), and are
  smoke-tested pod-side at a 16-token cap before any timed run;
* every throughput, memory and agreement number in this plan is unmeasured;
* batch invariance on the real 4B in bf16, for any engine including the
  incumbent.

## 9. Risks

* **The adapters are the least-tested code in the session.** Mitigated by A1
  and by the 16-token smoke that precedes each timed pass; a failed arm still
  leaves a valid benchmark of the arms that ran.
* **Engine installs may fail or conflict on the pod image.** That is itself an
  integration-cost finding and is recorded as one, not retried indefinitely.
* **R3's 1.5× threshold is a judgment call**, set before seeing data precisely
  so it cannot be tuned to a preferred answer. It is recorded here so a future
  session can argue with the number rather than reverse-engineer it.
* **Chaining into generation is new.** Prior sessions ran one kind of work; this
  one hands a machine-written decision to a second script. The failure mode —
  generation starting on a bad decision — is closed by R4 plus
  `--engine-from` refusing to guess when `winner` is null.

## 10. Verdict template

To be filled after the run; H1/H2/H3 each get an explicit answer, and the
`$/1k prompts` figure replaces the guessed bulk-build cost in the Stage 3/4
planning.
