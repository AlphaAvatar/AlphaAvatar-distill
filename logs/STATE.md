# Current project state

Updated: 2026-07-28 (UTC+8 dev box), after the packing control session —
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
- Tests: **119/119** on the dev box (torch 2.13.0+cpu); the 2026-07-27 GPU pod ran
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
5. **Decide how to buy trustworthy behavior numbers.** Options, none yet
   chosen: (a) ≥2 seeds per arm on every future comparison — simple, doubles
   cost; (b) report a seed-averaged score with its spread, which needs ≥3;
   (c) find a lower-variance behavior metric — e.g. score the *rate* of protocol
   compliance over many more generations per prompt rather than one greedy
   decode, which attacks the correlated-flip problem directly rather than
   averaging over it. **(c) is the interesting one and is CPU-cheap to
   prototype** against the four scorecards already on disk.
6. **Stage 3 SFT warm-up on teacher-generated data** — direction set by the
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
7. **`eval_behavior_v1` prompt-set expansion** — **demoted.** Still worth doing
   for coverage, but it is explicitly *not* the fix for the noise floor: the
   spread is largest on the n=76 axes, so it is correlated variance in the
   model, not sampling error over prompts.
8. Optional backlog: Stage 1 ablations; a from-init-tuned lr/warmup sweep; KD
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
