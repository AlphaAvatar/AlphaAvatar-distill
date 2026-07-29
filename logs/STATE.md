# Current project state

Updated: 2026-07-29 (UTC+8 dev box). **The engine benchmark is built, validated
on CPU, and pre-registered — and it found something before costing anything**
([log](experiments/2026-07-29_engine_adapter_and_bf16_invariance.md)).

**bf16 batched greedy generation is not batch-invariant.** With every prompt
truncated to a common length so the batch needs *no padding at all*, the Stage 1
student matched its own batch-1 output on **1 of 6** prompts; **fp32 matched
6 of 6** at the identical configuration. Only the dtype changed, so this is
reduction order, not padding — and padding-free batching is therefore not a
mitigation. The existing toy test missed it because it runs in fp32, which its
own docstring flags as the friendly case.

Two consequences. **SGLang's deterministic mode is now the most interesting arm
of the benchmark**, not a curiosity: it sells batch-invariant kernels, and this
project has now confirmed it lacks the property. And **the "corpus is the
artifact" stance is justified rather than assumed** — even *greedy* decoding is
not reproducible across batch compositions in bf16.

*Claim strength:* measured on **CPU, on the 0.6B student**. Not measured: GPU,
the 4B teacher, any serving engine, or whether divergent tokens change answer
*content*. An open hypothesis worth more than the engine ranking: if the 4B
teacher behaves the same, every batched-eval behavior scorecard carries an
unquantified batch-composition term, which would sit beside the 0.1290 seed
noise floor as a second source of behavior-metric instability. Eval batching has
never been varied in a completed run.

**GPU session complete** ([log](experiments/2026-07-29_engine_benchmark_gpu.md)).
Pod deleted after upload verification; **$2.60**, nothing billing, no pods or
volumes remain. Three results, in order of how much they change the plan:

1. **In-stack decoding does not scale with batch size, and the bulk corpus is
   unaffordable on it.** 37.5 / 43.9 / 39.3 tok/s at batch 2 / 4 / 8 — flat
   across a 4× range, at 9–13 GB of 44 GB. This **refutes the founding premise
   of `src/aadistill/generate.py`**, that batching was the missing speedup. At
   ~44 tok/s and ~1,600 tokens/candidate the pre-registered 1,000-prompt × n=4
   pilot needs **~40 GPU-hours (~$40)**; the guessed $25–145 bulk cost now has a
   measured basis and the answer is "not in-stack".
2. **Batch invariance fails on the 4B teacher**, as it did on the 0.6B student:
   7/8 identical at cap 64, one diverging at token 50. Corroborated at long
   context — the same ten prompts under *greedy* decoding average **1568 / 1760
   / 1514** new tokens at batch 2 / 4 / 8, so the logits themselves move with
   batch composition. "The corpus is the artifact" (P5) is now a measured
   requirement, not a preference.
3. **The pilot's two failing slices are the interesting part** — and a follow-up
   CPU analysis at $0 found that **one of them is not failing**
   ([log](experiments/2026-07-29_pilot_slice_analysis.md), two decision records
   2026-07-29). `rag_evidence` 1.000, `gsm8k` 1.000, `multihop_qa` 0.900.

   - **`refusal_uncertainty` (accept@n 0.100) is evaluation-only** — a
     **capability-scope / alignment-tax** decision (2026-07-30), *not* a
     judgement about answer length. It sits outside the recipe's declared target
     (reasoning, problem-solving, agent decision) and no product or safety
     requirement makes it mandatory, so the capacity, tokens and optimization
     pressure it would consume are unjustified for this baseline. The safety
     guard rail is preserved: refusal stays scored in `eval_behavior_v0`.
     *The earlier framing on this line — excluding it because the teacher's
     refusals are ~6× longer than the public targets — is **withdrawn**; P10
     now states length is not a reason to reject a target.* If refusal is ever
     trained it uses the teacher's native protocol and the same generic rules as
     every other slice: no refusal-specific word limit, no forced terseness, no
     public-target fallback.
   - **`openmath` (0.300) is genuinely cap-bound**: 28/40 truncated, but among
     the 12 that finished, accuracy is **0.750** and the longest used only 2,970
     of 4,096 tokens. The truncated ones are censored at exactly the cap, so
     **the yield of a higher cap is unmeasurable from this run** — it is a
     hypothesis with an unknown payoff, and at measured throughput a 16k cap
     costs ~4× per candidate. Not raised; queued as a measurement.

   **Standing consequence:** teacher targets are strongest where the corpus was
   already fine and weakest where the student is weakest. Separately, **the
   teacher answered 10/40 unanswerable questions** ("Hyrule", "GameCube and
   Wii.") — consistent with its measured grounding ceiling of 0.562. That is a
   fact about the teacher's grounding, recorded as such; it is not the reason
   refusal is out of the training mixture (scope is).

**Engine choice, decided by integration cost rather than throughput:** both serving engines
are **incompatible with this project's pinned stack in-process**, measured on
the pod. vLLM 0.26.0 installs but its compiled extension needs
`libcudart.so.13` (CUDA 13) while the image and this project's torch are
**cu128** — it does not import. SGLang 0.5.9 installs *and imports*, but only by
**downgrading torch 2.11.0 → 2.9.1 and transformers 5.13.1 → 4.57.1**, a major
transformers version this repo's code does not target; `uv sync` restored the
environment. So the benchmark runs `hf` alone, and rule R3 (a second stack must
earn its keep) is satisfied by integration cost rather than by tokens/second.
Untested follow-up: driving a serving engine from an isolated venv as a
subprocess, which is a real deployment pattern but more integration surface —
exactly the cost being weighed.

**This is a bound on what was tested, not a verdict on the engines.** The
untested path is running a serving engine from an **isolated venv** as a
subprocess or local HTTP server — a normal deployment pattern that sidesteps the
dependency conflict at the cost of a process boundary. Given result 1 above,
that path is now the highest-value next purchase: it is the only route that
could make a bulk corpus affordable, and the `Engine` interface already isolates
it to one new subclass.

Earlier context — the CE/KD conflict intervention —
**the conflict is confirmed causal, and it is the best-understood defect in the
project** ([log](experiments/2026-07-28_kd_conflict_intervention.md)). A 2x2
({`all`, `all_no_think`} x 2 seeds, 1000 steps, $4.48) moved p(`</think>`) from
**0.2995 ±0.0176 to 0.9989 ±0.0004** — a factor of 40 over the seed spread, with
the spread itself collapsing 44x. R1 and R2 fired; R3 (holdout +0.31%) is inside
the guard rail; **R4 did not fire** and is reported as registered.

| | ctrl (2 seeds) | treat (2 seeds) | baseline @2700 |
|---|---:|---:|---:|
| p(`</think>`) | 0.2995 | **0.9989** | — |
| think_closed | 0.2763 | **0.6776** | 0.6053 |
| format_ok | 0.0066 | **0.2894** | 0.2237 |
| empty_answer | 0.4342 | **0.0263** | 0.1711 |
| terminated | 0.3158 | 0.3750 | — |

The treatment beats the full-length baseline on every form metric at **37% of
the steps**. It is **not adopted as the recipe**: it is a confirmed diagnostic,
and the corpus contains an unknown number of similar disagreements — a second is
already measured at `<|im_end|>` (teacher p≈0.00003–0.004 on stopping for terse
slices, but **0.610 on code_math**, where targets resemble teacher output). That
is why `terminated` barely moved, and why **Stage 3's exit gate is blocked on
termination, not think-closing**.

General statement, now with two confirmed instances: **the teacher disagrees
with any target it did not write, and where it disagrees KD at 2x per-position
weight wins.** Masking is one span per disagreement; teacher-generated targets
remove all of them at once.

Earlier the same day, the packing control session —
**the project's first run-to-run variance measurement, and it is large enough
to unsettle earlier conclusions.** Two runs of the *same config* differing only
in seed scored `behavior_score_v0` **0.1380** and **0.2670**: a noise floor of
**0.1290**, which is *wider than the entire 0.1124 spread* of the 2026-07-27
start-point ablation's behavior ranking. Rule R5 fired.
[Log + interpretation](experiments/2026-07-28_stage3_packing_control.md).

Two things follow, and they point in opposite directions:

1. **The packing / `block_len` change is rejected, and that result is solid.**
   Both arms regressed on `holdout_v1` (+2.06% and +2.15% vs baseline 3.8285)
   and agree with each other to 0.09%. R2 fired. `concat`@1024 stands as the
   Stage 3 data path; the `packing` knob stays in the trainer, defaulted to the
   path every logged run used.
2. **Every behavior-based ranking in this project is now provisional.** The
   numbers are real; the *orderings* they were used to justify are not
   supported at one run per arm. Affected: the behavior half of the start-point
   ablation's conclusion, and "current best 20.2%" wherever it appears.
   The ablation's **NLL**-based half stands — holdout is the metric that turned
   out to be stable.

**New standing rule: behavior comparisons need ≥2 seeds per arm.** And more
eval prompts will *not* substitute — the largest seed spreads are on the n=76
axes (`format_ok` 0.2632, `fluency` 0.2193), not the n=12 ones, so this is
correlated variation in the *model's* protocol behavior, not sampling error
over prompts (§3 of the log). Variance budget goes to seeds, not prompts.

**And a follow-up CPU investigation found what is probably causing that
variance — at $0** ([log](experiments/2026-07-28_kd_ce_protocol_conflict.md)).
At the token the instability lives on, **CE and KD want opposite things**:

| signal | wants at `<think>\n\n` → | weight |
|---|---|---|
| CE (mask confirmed) | `</think>`, one-hot | 0.25 |
| KD (`scope: all`, τ=1) | `Okay`/`Hmm` ≈1.0; **p(`</think>`) = 0.000000** | 1.0 |

Per position **KD pulls 2.0× harder than CE** (`ce_targets`≈7,950 vs
`kd_positions`≈15,740 measured on both arms). The student equilibrates at
**p(`</think>`) ≈ 0.334**, uniform to ±0.015 across four unrelated groups —
a force balance, not noise — and greedy decoding turns that near-tie into
correlated flips across many prompts, which is exactly the observed signature.
The cause is that the teacher is **teacher-forced through an empty-think target
it would never produce**, so KD transmits a contradiction rather than knowledge.
This has been in every Stage 3 run since 2026-07-22.

*Claim strength:* the conflict, the 2.0× ratio and the 0.334 equilibrium are
**measured**; that they *cause* the instability is inferred and needs an
intervention to confirm (two mechanism claims were retracted earlier the same
day, so this one is held to that standard).

Nothing is running or billing; pod deleted by the orchestrator after upload
verification. Session cost **$7.17**, balance **$226.00**.

Earlier header, still true of its own subject — **the teacher has a ceiling: 0.7443 on
`eval_behavior_v0` vs the student's 0.2015**, so the figure shows both ends of
the gap for the first time ([log](experiments/2026-07-28_teacher_behavior_v0.md)).
The eval itself is validated by that number — a competent model scores 0.74 on
it, so the student's score is the student's problem, not the harness's. Nothing
is running or billing; the pod was deleted by its fetch driver on completion
(~$1.20 for the session).

Standing context, restated for accuracy after today:

- **The Stage 3 recovery recipe is single-stage** (2026-07-27). The FFN-first
  warm-up ladder is retired: a single run from the Stage 1 init reaches the
  chain's holdout quality at 33% fewer steps. This rests on **NLL**, which the
  noise measurement leaves intact. The companion claim that it was also
  "best-behaved" is **not supported** at one seed per arm.
- **The headline metric is `behavior_score_v0`**, six credited mechanical axes
  on 76 held-out prompts, with holdout NLL as a ±1% guard rail (maintainer,
  2026-07-28). Today's result does not change the choice of headline, but it
  does establish that a *single* run of it cannot rank two recipes. Current
  best measurement: `s2v1_from_init@2700` at **20.2% ± ~0.13**; teacher
  **74.4%** ([log](experiments/2026-07-28_teacher_behavior_v0.md)).
- **The teacher is never forced out of thinking mode** (2026-07-28). Evaluation
  and generation judge it at its actual capability: no prefill, 4096-token cap.
  Scope is cut by lowering n or dropping slices, never by suppressing reasoning.
- **Stage boundary, restated (maintainer, 2026-07-28, supersedes two earlier
  records the same day):** "on-policy" means the training **states come from the
  student**. So **Stage 3 owns all teacher-generated data** — offline, streamed
  or interleaved, at any scale — because those states come from a fixed prompt
  corpus. **Stage 4/5 begins only when the student's own generations become
  training states.** "Online" is retired as a staging word: it describes a
  schedule, not a distribution, and conflating the two is what made the earlier
  records inconsistent. Surviving constraint, independent of staging: any
  streamed generation must still be **snapshotted to a hashed artifact** before
  it trains anything, or the run is not reproducible (P4/P5).
- **Stage 3's next direction is an SFT warm-up on teacher-generated data**
  (maintainer, 2026-07-28, four decision records). Superseding the earlier
  plan: targets are the teacher's **unfiltered top-n** — correctness selection
  moves to Stage 4/5 — with `n` adaptive per prompt from measured divergence,
  and the target protocol observed from the teacher rather than configured.
  All four are recorded as **direction/hypothesis, not results**; no spend is
  committed and the quantitative claims are unmeasured.

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV) → student 0.6B-class
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb). BF16 training,
INT8 deployment target.

**This run is a baseline instance, not the target** (decisions 2026-07-28). Two
standing rules follow from that: (a) **no README Optim record entries during
baseline construction** — the first record point is written only after the
baseline reaches Stage 6 with satisfactory results, and it will be the first
entry rather than a backfill; (b) the algorithm core stays
**model-family-agnostic** — the methods are meant to carry to MoE, VLM and
Omni-models, so family specifics belong in recipes and new core code fails
loudly on an unsupported architecture. Neither is a claim of support: no
non-dense, non-text model has been attempted. The porting surface is inventoried
in the 2026-07-28 scope decision.

Pipeline position: **Stage 0 → Stage 1 → Stage 2 (v0 + v1 scale-up) → Stage 3
s1 → s2 sizing A/B → s2 quality gate → start-point ablation (2026-07-27)**, all
passed. Stage 4 not started.

Repo: branch `main`, clean at the 2026-07-28 documentation commits on top of
`30c066c`. Those rebuilt the README figure twice — first as a lineage view, then
(current) as the **leaderboard-style view the maintainer asked for**: one point
per student at its current best against its teacher, y = behavior score, x =
parameters. They also rewrote the teacher-target proposal for the directives
above and recorded the record/scope/metric policies. No experiment data changed;
the behavior scores are recomputed from scorecards that already existed.

## The recipe, as it now stands

**Stage 3 recovery is one run, not a ladder** (decision record 2026-07-27):
`configs/stage3_s2v1_from_init.json` — start from the Stage 1 init, 2700 steps
× 16 × 1024-token blocks on mixture v1, attention-unfrozen freeze set (440.5M
trainable, tied embedding frozen), CE 0.25 + full-vocab KD 1.0 at τ=1 scope
`all`, lr 2e-4 / warmup 60 / cosine to 0.1×, fp32 master + bf16 autocast,
**seed 20260726**, eval every 150 steps on 64 val blocks.
The `s1_ffn_norm` → `s2_blocks` warm-up legs are **retired** for this
architecture and data scale.

## Verified state (all on the real model/data)

- **Start-point ablation (2026-07-27, 1× L40S, $5.82, both arms verified
  16/16):**

  | | s1@660 | A0 `chain` | A1 `from_s1` | A2 `from_init` |
  |---|---|---|---|---|
  | total steps | 660 | 4020 | 3360 | **2700** |
  | holdout_v1 NLL | 4.2107 | **3.8003** | 3.8067 | 3.8285 |
  | INT8 decoder / full | +0.03/+0.21% | +0.08/+0.21% | +0.13/+0.29% | +0.20/+0.33% |
  | format_ok | 0.105 | 0.066 | 0.145 | **0.224** |
  | think_closed | 0.263 | 0.316 | 0.513 | **0.605** |
  | empty_answer | 0.605 | 0.382 | 0.211 | **0.171** |
  | tool_call_parsed | 0.083 | 0.000 | 0.000 | **0.250** |
  | rag evidence_hit *credited* | 0.167 | 0.000 | 0.083 | **0.333** |

  Log + review: `logs/experiments/2026-07-27_stage3_start_point_ablation.md`.
- **`eval_behavior_v0` (2026-07-27, CPU + GPU):** 76 held-out prompts over 7
  chat groups, mechanical scorers only, committed at `data/eval_behavior_v0/`.
  Run at every recovery gate next to holdout_v1 and the INT8 evals. Build log
  (including the echo-credit and truncation findings):
  `logs/experiments/2026-07-27_eval_behavior_v0.md`.
- **Stage 3 `s2_blocks_v1` (2026-07-26):** holdout 3.8003, the best
  language-modeling result; retains that title, but is **last** on every
  behavior axis. Log: `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md`.
- **Stage 2 mixture v1 (2026-07-26, CPU):** 64,484 train samples / 22,133,631
  train tokens / 21,610 blocks@1024; val_v1 1,916; calib 200.
- Tests: **191/191** on the dev box (torch 2.13.0+cpu), +33 from the engine
  adapter layer on 2026-07-29; the 2026-07-27 GPU pod ran
  the then-current suite at 85 passed / 6 skipped (torch 2.11.0+cu128,
  Python 3.12.3).

## Best checkpoint — the behavior ranking is inside the noise floor

**Read this section with ±0.13 on every behavior number** (measured 2026-07-28,
see the header). The scores below are real measurements of real checkpoints;
what is *not* supported is the ordering between any two of them.

- **Recommended branch point, on grounds that survive:**
  `stage3/s2v1_from_init/step_002700/model` — holdout 3.8285 (+0.74% vs the
  best NLL, inside the ±1% guard-rail band) at **33% fewer steps**. That is a
  cost argument backed by the stable metric, and it is why this remains the
  branch point. Its `behavior_score_v0` **20.2%** is a measurement, not a
  demonstrated win over the others.
- **Best held-out NLL:** `stage3/s2_blocks_v1/step_002700/model` (3.8003).
  Its 8.9% behavior score was previously called "worst on behavior" — with a
  0.1290 noise floor that claim is **withdrawn**; 8.9% and 20.2% are not
  separable at one run per arm.
- Other measured checkpoints: s1@660 **12.9%** / 4.2107 · `s2v1_from_s1`
  **9.5%** / 3.8067 · `s2v1_bl2048` **13.8%** / 3.9073 · `s2v1_bl2048_seedB`
  **26.7%** / 3.9109. The last two are the *same config* and bracket every
  other student on this list — which is the clearest possible statement of the
  problem.
- The A/B arms and the two Stage 1 inits were never scored on the behavior eval.

## Three comparability rules now in force

0b. **Stage 3 has an exit gate, and it is not currently met** (decision
   2026-07-28). Stage 4/5 makes the student the *data source*, so its format
   competence bounds everything downstream: below threshold the rollout corpus
   is mostly parse failures and every verifier label encodes "did this parse".
   Gate is on `format_ok` / `think_closed` / `empty_answer` at ≥2 seeds; the
   number is set from a Stage 4 pilot's measured accept rate, not in advance.
   For scale: at `format_ok` 0.22 only ~21% of prompts yield 2 parseable
   candidates of 4 — and that is optimistic, since format failures are measured
   to be *correlated* across a model's samples. Baseline stands at 0.2237, so
   **Stage 3 is not exitable today.**
0a. **Metrics are chosen by resolving power, not stage number** (decision
   2026-07-28). A metric applies iff it is off its floor, off its ceiling, and
   its noise is smaller than the effect being chased. Consequences in force:
   NLL is **not** retired after Stage 2 — it is the guard rail *and* the only
   metric measured to be stable, so it decides when behavior is noise-bound;
   behavior axes are reported individually and only once off the floor
   (`format_ok` was 0.0132 at 1000 steps); every stage also carries a targeted
   **probe** for the defect it is fixing (`p(</think>)` moved 0.2907 → 0.9990
   where NLL moved 0.36%); and real benchmarks are gated on measured format
   competence, since at `format_ok` 0.22 a benchmark scores the parser.
0. **Behavior comparisons need ≥2 seeds per arm** (measured 2026-07-28). One
   run per arm cannot support a behavior ranking: the seed-only noise floor on
   `behavior_score_v0` is **0.1290**, wider than any inter-arm difference this
   project has reported. This roughly doubles the cost of a behavior-based
   claim, which is an argument for making fewer and larger ones. Adding eval
   prompts is **not** a substitute — the spread is correlated across prompts
   (largest on the n=76 axes), so it is variance in the model, not the sample.
1. **Pin the seed across compared runs.** The 64-block val subset is a
   permutation of `cfg["seed"] + 777` (`src/aadistill/train.py:332`). Seed
   **20260726** is pinned for this whole family of runs.
2. **holdout_v1 is a guard rail, not the headline** (decision 2026-07-28). It
   is fineweb-edu web text, nearly blind to chat format, grounding, refusal and
   tool-call validity, and it ranked the ablation arms in the *opposite* order
   to the behavior eval. Band ±1%; a large drop is an abort signal.
3. **Behavior scorecards are only comparable within one device** (decision
   record 2026-07-27). The same checkpoint scored on CPU vs L40S moved
   `format_ok` by 1–3 prompts, though `terminated`/`truncated_at_cap` matched
   exactly. This is why `scripts/pod/score_refs.sh` re-scores reference
   checkpoints on the pod rather than reusing dev-box baselines.

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
  `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1.
  **Note:** a `VIRTUAL_ENV=/home/ecs-user/AlphaAvatar/.venv` (py3.11, torch
  2.9+cu128, transformers 4.57) may be active in the shell; it is a *different
  project's* env and is incompatible with this repo's transformers-v5 API.
  Always run through `uv run`.
- GPU runs: RunPod (runpodctl 2.7.1 authenticated; skill at
  `.agents/skills/runpodctl`). Balance **$233.17** (~$17.5 total project GPU
  spend), $80 spend limit. **No pods or volumes exist; nothing is billing.**
- Pod playbook: `scripts/pod/AGENTS.md`. The scripts are now **parameterized by
  `run_env.sh`** and drive multi-arm sessions unattended; arm-scoped markers,
  sha256 verification at every hop, pre-registered abort checks, and pod
  deletion only after upload verification. Setup was **1.2%** of the last
  session's spend (down from 19%) because two arms shared one pod.
- HF: dev box logged in as `AlphaAvatar`; private artifact repo
  `AlphaAvatar/aadistill-artifacts` holds s1@660, both A/B finals,
  `s2_blocks_v1`, both ablation arms, the Stage 1 init, and the reference
  behavior scorecards (see `logs/artifact_manifests.md`).
- GPU run-to-run variance at the holdout level remains **unmeasured**.

## What exists and why

- `src/aadistill/engines.py` (2026-07-29) — token-in/token-out `Engine`
  interface with `hf` / `vllm` / `sglang` adapters. Adapters do only "prompt ids
  in → new ids out"; stop-cutting and cap flags are shared, so a cross-engine
  token comparison measures engines rather than three copies of the trimming
  code. **The vLLM and SGLang adapters have never executed** (both CUDA-only);
  they are smoke-tested pod-side before any timed run.
- `src/aadistill/` — env, manifest, teacher (loading + `load_causal_lm`),
  collect (S0), project, sandwich, student (S1), data (S2 loader, now with
  **`best_fit_blocks`** — packing that never splits a sample), train (S3
  recovery trainer + extra_val), quant (INT8 fake-quant eval, P9), behavior
  (eval_behavior_v0 scorers + **`behavior_score`**, the headline metric),
  **verify (per-slice correctness rules for teacher targets)**.
- `scripts/` — stage scripts, `train_stage3.py`, `eval_ppl.py`,
  `build_eval_behavior_v0.py`, `eval_behavior.py`, `plot_perf_trend.py`,
  **`generate_teacher_answers.py`** (top-n verified teacher targets), and
  `scripts/pod/` (GPU session scripts + **`teacher_session.sh`** /
  **`teacher_session_fetch.sh`**, the nohup dev-box driver that fetches and
  deletes the pod without an agent session attached).
- `configs/` — Stage 0/1; stage3: `s1_ffn_norm`, `s1_ext`, `s2_blocks`,
  `s2_blocks_v1`, `s2v1_from_s1`, `s2v1_from_init` (all ran), smoke configs.
- `data/` — frozen v0 and v1 mixtures (jsonl gitignored, manifests committed);
  **`data/eval_behavior_v0/` (prompts.jsonl + manifest, both committed)**.
- `tests/` — 119 tests. `logs/` — decisions (25), experiments (14), proposals (3),
  supported_models, artifact_manifests, this file.
- `artifacts/` (gitignored) — stage0 stats, stage1 checkpoint, stage3 run
  artifacts and reference scorecards (small files local; final weights HF-only
  except s1).
- `artifacts/teacher/` (gitignored) — the teacher scorecard + generations.
- `assets/` — perf figure json + svg. The json now carries a `headline` metric
  block, `systems` (one entry per student: current best + previous best +
  params), `references` (teacher size, score `null` until measured), the `guard`
  block (NLL), and the 8-attempt run history that generates the README table.

## Latest known working commands

```
uv run pytest tests/ -q                                          # 119 passed
uv run python scripts/build_eval_behavior_v0.py                  # rebuild prompt set
uv run python scripts/eval_behavior.py \
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model \
  --out artifacts/stage3/s1_ffn_norm_v0/eval_behavior_v0.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model <ckpt> [--fake-quant int8 [--fake-quant-scope decoder]]
uv run python scripts/plot_perf_trend.py [--print-table]
POD_ID=<id> HOST=<ip> PORT=<port> bash scripts/pod/orchestrate.sh   # GPU session
```

## Capability scope (declared 2026-07-30)

The recipe's **primary capability target** is the teacher's reasoning,
problem-solving and agent-decision quality under the deployment budget. Broad
teacher imitation is explicitly **not** the objective (AGENTS.md P3, P10.1).
Every Stage 2 group is classified against that target:

| group | class |
|---|---|
| `code_math` (gsm8k, openmath), `multihop_qa` | **primary capability transfer** |
| `rag_evidence`, `tool_calling`, `instruction`, `long_context` | **supporting capability** |
| `short_realtime` | **evaluation-only (provisional — currently still in the trained mixture; moving it needs an ablation)** |
| `refusal_uncertainty` | **evaluation-only** |

Mixture weights are not uniform by default, and reasoning capability is reported
**separately** from refusal, safety, style and format axes. Adding or increasing
any alignment-oriented slice, loss or filter requires the P10.1 record: the
requirement it serves, whether it is non-negotiable, the reasoning metrics that
may regress, a fixed budget, and the ablation that will measure the tax.

**Known debt (decision 2026-07-30):** `src/aadistill/verify.py` still hard-codes
refusal-specific and generic word-count rules (`REFUSAL_MAX_WORDS = 60`,
`MAX_ANSWER_WORDS = 600`) and `generate_teacher_answers.py` still falls back to
the public target — all three are P3 violations. They are inert while refusal is
evaluation-only, but **anyone adding a slice to teacher generation inherits the
600-word answer ceiling and must fix it first.**

## Latest session: isolated-venv engine + openmath cap (2026-07-30, $1.85)

[Log](experiments/2026-07-30_isolated_engine_and_cap.md). Pods deleted, nothing
billing. No corpus built, no refusal data generated, no alignment slice added.

**1. An isolated-venv vLLM server is 5.3× faster and agrees with the trainer on
nothing.**

| engine | tok/s | $/1k prompts | batch-invariant | agreement vs in-stack |
|---|---:|---:|---:|---:|
| `hf` in-stack | 40.4 | $12.33 | 7/8 identical | — (reference) |
| `vllm_server` | **213.9** | **$2.27** | 4/8 identical | **0/8 = 0.000** |

R2 (≥3×) fires at 5.29×; **R1 (≥0.90 agreement) fails at 0.000**, so
`decision.json` records `winner: hf`. Divergence is not immediate — median first
divergence is token **260** — the stacks track each other for a few hundred
tokens then split. vLLM is also *less* batch-invariant than in-stack.

**Not adopted — the engine question is reopened** (maintainer correction,
2026-07-30). The session's *rule* R1 made exact token agreement an adoption gate
and produced `winner: hf`; **that gate is retired**, and with it the stage-scoped
split this line previously recorded. Two reasons, both independent of the
measurements:

- **Token equality is not a prerequisite for on-policy training.** Production RL
  systems pair an inference-optimized rollout engine with a separate trainer,
  generate asynchronously, and correct the mismatch explicitly — rollout and
  trainer log-probs, policy/checkpoint versioning, bounded staleness, token- or
  sequence-level importance sampling, clipping or rejection of far-off-policy
  samples, hashed rollout snapshots. A different engine creates a *measurable*
  mismatch, not an impossible one.
- **One measured alternative cannot select a standing backend.** SGLang was
  never actually tested: 2026-07-29 blamed a Python dependency conflict when the
  real constraint is the host's CUDA-12.8 driver.

The 0/8 figure survives as a **diagnostic** — median first divergence at token
260 is a useful prior on how large a correction term must be — and it gates
nothing. Note the gate was incoherent anyway: decoding is not batch-invariant
*within* one stack (7/8 in-stack, 4/8 vLLM here), so demanding cross-stack token
identity asks for a property the trainer lacks against itself.

**Standing position:** vLLM 0.11.0 is the **first measured engine, not the
choice**. **HF `model.generate` is retired as the planned production rollout
path** — reference implementation, debugging path, small-scale correctness
oracle, and fallback only. The production direction is one efficient, isolated
rollout service reused across Stages 3, 4 and 5, with Stage 4/5 designed around
**asynchronous generation plus explicit correction**. Adoption criteria are now
token-in/token-out transport, exact rollout token-ID recording, rollout log-prob
availability, measured KL / importance-ratio distribution, bounded off-policy
rate and staleness, stable corrected training in a pilot, and throughput/cost/
reliability. **This project meets the first two and none of the rest.**

**2. The 2026-07-29 "vLLM is incompatible" conclusion was version-specific.**
The real wall is the **host driver**: latest vLLM needs CUDA 13, this box runs
570.124.06 (CUDA 12.8), and no venv fixes that. vLLM **0.11.0** (torch
2.8.0+cu128) does run — with `transformers==4.57.1` pinned *inside its own venv*
and `/opt/vllm-venv/bin` on PATH for `ninja`. Bounded, documented, buys 5.4×.

**3. OpenMath cap stays at 4,096 — R3 does not fire.** Closure improved 0.300 →
0.850 (R3a fires; true median trace is **6,487** tokens, so the cap was really
binding), but **accuracy among closing candidates collapsed 0.750 → 0.294**
(R3b fails). The candidates needing >4,096 tokens are the ones the teacher gets
**wrong**: raising the cap converts `truncated_at_cap` into `answer_mismatch`
(3 → 10), and cost per accepted target **doubles** (14,931 → 29,707 tokens).
Long reasoning here is floundering, not care — the pre-registered risk, measured.

## Next action: build the log-prob path, then compare rollout engines

**No bulk corpus is built yet** (maintainer, 2026-07-30): the $2.27/1k figure is
real, but the engine is unchosen and a corpus is not worth building twice.

1. ~~Build the rollout log-prob path~~ — **done 2026-07-30, CPU, $0.**
   `Engine.generate(logprobs=True)` returns per-token rollout log-probs aligned
   1:1 with tokens, with the alignment *enforced* in `_finalize` rather than
   trusted; `aadistill.rollout` adds `score_tokens` (trainer-side teacher-forced
   recompute), `importance_stats` (ratio quantiles, off-policy rate against a
   band, k3 KL), `aggregate_stats` (pooled **by token**, not by sequence), and
   hashed `write_snapshot`/`read_snapshot` with mandatory policy and engine
   identity. **218 tests pass.** The load-bearing test generates with log-probs
   and recomputes them with the scorer on the same model — the only end-to-end
   evidence that both halves of a ratio refer to the same token.

   Two deliberate design points worth carrying forward: masked positions (a
   re-appended stop token has no rollout log-prob) are **dropped and counted,
   never imputed**, since a fabricated denominator would bias every statistic;
   and `HFEngine` log-probs cost ~10 GB at 4k tokens × batch 4 because HF keeps
   per-step vocab scores, which is fine for its oracle role and is another reason
   the production rollout path is a serving engine.

   Still missing for a real pilot: **`VLLMServerEngine` log-probs are written but
   unexecuted** (it requests `logprobs: 0` and refuses a response without
   `token_logprobs`), and SGLang raises `NotImplementedError` rather than
   pretending.
2. **Compare *current* rollout engines** per
   [`proposals/2026-07-31_current_engine_benchmark.md`](proposals/2026-07-31_current_engine_benchmark.md)
   — **awaiting approval, no pod created.** `vllm/vllm-openai:v0.26.0` vs
   `lmsysorg/sglang:v0.5.12`, each in its **own official image** on a
   CUDA-13 host, **one pod per engine**, ≤3.0 h ≈ **$3.00**.

   **vLLM 0.11.0's numbers do not carry forward.** That build was reached by
   pinning backwards until the engine fit the training image; it is a
   measurement of an obsolete compatibility build, not of vLLM. The 0.26.0
   failure was **environment selection, not an engine result** — the host ran
   driver 570.124.06 / CUDA 12.8 against an engine targeting CUDA 13. The fix is
   one flag that was never passed: **`runpodctl pod create --min-cuda-version
   13.0`** (RunPod's `allowedCudaVersions` includes 13.0). L40S hosts with driver
   580.95.05 / CUDA 13.0 exist, so the GPU was never the problem.

   An engine that cannot be scheduled has **not lost**; a third candidate
   substitutes rather than SGLang being recorded as a loss.
3. **Run the correction experiment**, with its pre-registered stability bound:
   corrected training stays inside a same-seed in-stack pilot's band, the
   clipped/rejected token fraction stays **below 5%**, and the corrected run is
   not worse than an uncorrected control on the guard rail.

## Superseded — the isolated-venv engine test (completed 2026-07-30)

The 2026-07-29 session closed three prerequisites (4B batch invariance, real
batched throughput, the per-slice accept/divergence profile) and opened one
question that now gates the whole teacher-target direction: **can a serving
engine be driven from an isolated venv?** In-stack decoding is measured at ~44
tok/s and flat in batch size, which prices a bulk corpus at ~$40/1k prompts —
too expensive to scale. A subprocess/HTTP-server engine with its own torch build
is the only known route around both that and the dependency conflict.

**Pre-registered and ready to run, awaiting approval on spend:**
[`proposals/2026-07-30_isolated_engine_and_cap.md`](proposals/2026-07-30_isolated_engine_and_cap.md)
— 1x L40S, ceiling **2.5 h ≈ $2.50**, no corpus built (sizing a bulk build is
the *output* of this test, not an activity to run beside it). It bundles the
openmath cap measurement so that question does not justify its own pod.

The code is already written and CPU-verified: `VLLMServerEngine` in
`src/aadistill/engines.py` drives a vLLM OpenAI-compatible server over HTTP,
sends **token ids** rather than text, requires `return_token_ids`, and orders
choices by the server's `index` rather than arrival — the OpenAI schema carries
an index because order is not contractual, and mispairing completions with
prompts is silently wrong. It adds **no dependency** (`urllib`, stdlib).
Rules: adopt only at **>=3x** in-stack throughput (stricter than the previous
1.5x, because a process boundary is real integration surface) **and** >=0.90
greedy agreement with the in-stack reference.

**Both slice questions are now closed on the CPU side** (decisions 2026-07-29,
refusal reframed 2026-07-30): `refusal_uncertainty` is **evaluation-only by
capability scope**, `openmath` keeps its 4,096 cap until a measurement justifies
raising it. The openmath cap test should
**ride along with the engine session** rather than justify its own pod — same 10
prompts at a raised cap, reporting both completion rate *and* accuracy, since
longer reasoning is not automatically better reasoning.

Both generation and benchmark scripts now **default** to the in-scope slices
`rag_evidence,multihop_qa,gsm8k,openmath`; pass `--slices` explicitly to
override.

## Next actions (ordered)

Revised 2026-07-28 **after** the packing control. The queue changed shape: the
control answered its question (no) and surfaced a bigger one (behavior
comparisons are not readable at one seed per arm), so the top of the queue is
now about making measurements trustworthy before buying more of them.

1. ~~Score the teacher~~ — **done 2026-07-28**, 0.7443 vs 0.2015
   ([log](experiments/2026-07-28_teacher_behavior_v0.md)). Biggest gaps: math
   +0.714, tool_call +0.667, format_ok +0.618. Grounding's ceiling is only
   0.562, so effort there has little headroom. **Caveat still open:** student
   rows were scored at cap 512 and the teacher at 4096, so form metrics are not
   like-for-like. The control session did **not** re-score references at 4096 —
   it kept 512 to stay comparable with existing scorecards.
2. ~~Packing / `block_len` control~~ — **done 2026-07-28, rejected.** Both arms
   +2.1% holdout vs baseline, agreeing to 0.09%. R2 fired; `concat`@1024
   stands ([log](experiments/2026-07-28_stage3_packing_control.md)).
3. ~~Run-to-run variance~~ — **done, and it is the session's real finding.**
   Noise floor **0.1290** on `behavior_score_v0`, wider than the 0.1124 spread
   of the whole start-point ablation. R5 fired.
4. **Re-state the conclusions R5 unsettled.** Not delete — the runs happened and
   the numbers are real; the *orderings* are unsupported. Touches
   `supported_models.md` (done), the README figure caption, and
   `logs/experiments/2026-07-27_stage3_start_point_ablation.md`, which should
   carry a pointer to the noise floor rather than be rewritten.
5. **Test the CE/KD protocol conflict — now the highest-value next experiment**
   ([log](experiments/2026-07-28_kd_ce_protocol_conflict.md)). Candidate fixes,
   cheapest first: (a) exclude the template-inserted think block from KD — the
   trainer already threads a content mask for padding, so it is the same
   mechanism on a different span; (b) drop the empty-think rendering;
   (c) teacher-generated targets, which removes the cause rather than the
   symptom and is already the maintainer's direction; (d) reweight CE/KD, which
   trades one arbitrary balance for another. Readout is `think_closed`,
   `format_ok`, `empty_answer` — not the composite. **≥2 seeds per arm.**
   (a) is a short run and would confirm or kill the mechanism cheaply.
6. **Decide how to buy trustworthy behavior numbers.** Options, none yet
   chosen: (a) ≥2 seeds per arm on every future comparison — simple, doubles
   cost; (b) report a seed-averaged score with its spread, which needs ≥3;
   (c) find a lower-variance behavior metric — e.g. score the *rate* of protocol
   compliance over many more generations per prompt rather than one greedy
   decode, which attacks the correlated-flip problem directly rather than
   averaging over it. **(c) is the interesting one and is CPU-cheap to
   prototype** against the four scorecards already on disk.
7. **Stage 3 SFT warm-up on teacher-generated data** — direction set by the
   maintainer 2026-07-28 (four decision records, all marked
   direction/hypothesis, not results). Unaffected by the packing result, and
   arguably strengthened by it: protocol behavior is exactly what is unstable,
   so supervising it directly is the obvious lever. Before any spend, three
   things must be measured, all cheap:
   - `assert_batch_invariant` on the real 4B teacher in bf16 (verified only on
     a toy model in fp32 so far);
   - real batched generation throughput, to replace the 55 s/prompt figure,
     which was measured at `batch_size` 1 and is not an engine limit;
   - the per-slice divergence profile that sets the adaptive-`n` thresholds.
8. **`eval_behavior_v1` prompt-set expansion** — **demoted.** Still worth doing
   for coverage, but it is explicitly *not* the fix for the noise floor: the
   spread is largest on the n=76 axes, so it is correlated variance in the
   model, not sampling error over prompts.
9. Optional backlog: Stage 1 ablations; a from-init-tuned lr/warmup sweep; KD
   objective and weights (CE 0.25 + KD 1.0, τ=1, scope `all` have never been
   varied); attribution of the packing result between packing and `block_len`,
   which is only worth buying if the data path is revisited for trace data.

## Open decisions for the user

- **Score the teacher (~$1–1.5, ≈1 h GPU).** Independent of everything else and
  the fix for the figure's missing ceiling. Recommended first.
- **Online teacher generation feeding the student is Stage 4/5, not Stage 3**
  (decision 2026-07-28). The deciding property is whose distribution the
  training states come from, not whether the teacher runs in real time — the
  teacher already runs inside Stage 3's loop doing full-vocab KD every step.
- **Nothing is pending for the next session except the control run itself.**
  The teacher-generation branch (pilot, bulk build, trace training) moved to
  Stage 4/5 on 2026-07-28, taking the mixture-change approval, the vLLM
  dependency approval and the $25–145 build with it — **no spend is committed**.
  Option B (full traces) stands as the design for that work, not as a
  cancellation.
*(Closed 2026-07-28: the README Optim-record question. The answer is a standing
rule — no records during baseline construction, first record point after Stage 6
— so it is no longer an open decision. See the Status section and the 2026-07-28
decision record.)*

## Links

- `logs/experiments/2026-07-27_stage3_start_point_ablation.md` (latest run + review)
- `logs/experiments/2026-07-27_eval_behavior_v0.md` (behavior gate: build + baselines)
- `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md` ·
  `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md` ·
  `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`
- `logs/experiments/2026-07-26_stage2_offline_v1.md` ·
  `logs/experiments/2026-07-26_int8_fakequant_eval.md`
- `logs/proposals/2026-07-27_stage2_teacher_generated_answers.md` (awaiting approval)
- `logs/decisions.md` · `logs/supported_models.md` · `logs/artifact_manifests.md`
- `scripts/pod/AGENTS.md` (GPU session playbook)
