# Current project state

Updated: 2026-07-27 (UTC+8 dev box) — **Start-point ablation complete; the
Stage 3 recovery recipe is now single-stage.** Both pre-registered rules fired:
the A/B arm-B leg was neutral (+0.17%) and the **FFN-first warm-up ladder is
unnecessary** — a single-stage run from the Stage 1 init reaches the chain's
quality with 33% fewer steps. A new behavior gate, `eval_behavior_v0`, was built
first and **reversed the ranking the primary metric produced**: the cheapest
lineage is the best-behaved one. No GPU work is running or billing.

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV) → student 0.6B-class
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb). BF16 training,
INT8 deployment target.

Pipeline position: **Stage 0 → Stage 1 → Stage 2 (v0 + v1 scale-up) → Stage 3
s1 → s2 sizing A/B → s2 quality gate → start-point ablation (2026-07-27)**, all
passed. Stage 4 not started.

Repo: branch `main`, clean at the documentation commit on top of `aa93d91`
(auto-generated ablation write-up + review, decision records, manifests,
regenerated perf trend).

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
  Run at every recovery gate next to holdout_v1 and the INT8 evals.
- **Stage 3 `s2_blocks_v1` (2026-07-26):** holdout 3.8003, the best
  language-modeling result; retains that title, but is **last** on every
  behavior axis. Log: `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md`.
- **Stage 2 mixture v1 (2026-07-26, CPU):** 64,484 train samples / 22,133,631
  train tokens / 21,610 blocks@1024; val_v1 1,916; calib 200.
- Tests: **91/91** on the dev box (torch 2.13.0+cpu); 85 passed / 6 skipped on
  the 2026-07-27 GPU pod (torch 2.11.0+cu128, Python 3.12.3).

## Best checkpoint — the two metrics disagree, deliberately

- **Best holdout NLL:** `stage3/s2_blocks_v1/step_002700/model` (3.8003).
- **Recommended branch point for further work:**
  `stage3/s2v1_from_init/step_002700/model` (3.8285, +0.74% — inside the
  pre-registered 1% band) — best on every behavior axis and 33% cheaper to
  produce. For a realtime agent target (P10) that is the more relevant
  evidence, but choosing it is a **judgment, not a pre-registered rule**.

## Three comparability rules now in force

1. **Pin the seed across compared runs.** The 64-block val subset is a
   permutation of `cfg["seed"] + 777` (`src/aadistill/train.py:332`). Seed
   **20260726** is pinned for this whole family of runs.
2. **holdout_v1 is a language-modeling metric, not a behavior metric.** It is
   fineweb-edu web text and is nearly blind to chat format, grounding, refusal
   and tool-call validity. `eval_behavior_v0` covers those.
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

- `src/aadistill/` — env, manifest, teacher, collect (S0), project, sandwich,
  student (S1), data (S2 loader), train (S3 recovery trainer + extra_val),
  quant (INT8 fake-quant eval, P9), **behavior (eval_behavior_v0 scorers)**.
- `scripts/` — stage scripts, `train_stage3.py`, `eval_ppl.py`,
  **`build_eval_behavior_v0.py`**, **`eval_behavior.py`**, `plot_perf_trend.py`,
  and `scripts/pod/` (parameterized GPU session scripts).
- `configs/` — Stage 0/1; stage3: `s1_ffn_norm`, `s1_ext`, `s2_blocks`,
  `s2_blocks_v1`, `s2v1_from_s1`, `s2v1_from_init` (all ran), smoke configs.
- `data/` — frozen v0 and v1 mixtures (jsonl gitignored, manifests committed);
  **`data/eval_behavior_v0/` (prompts.jsonl + manifest, both committed)**.
- `tests/` — 91 tests. `logs/` — decisions (13), experiments (9), proposals (3),
  supported_models, artifact_manifests, this file.
- `artifacts/` (gitignored) — stage0 stats, stage1 checkpoint, stage3 run
  artifacts and reference scorecards (small files local; final weights HF-only
  except s1).
- `assets/` — perf trend json + svg (8 attempt points).

## Latest known working commands

```
uv run pytest tests/ -q                                          # 91 passed
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

1. **Teacher-generated answers** (`logs/proposals/2026-07-27_stage2_teacher_generated_answers.md`)
   — now unblocked (the behavior eval exists) and aimed squarely at the one
   defect every arm shares: verbose, repetitive, non-terminating answers
   (`rep_3gram` 0.35–0.41, `answer_words` 199–231, `truncated_at_cap` 0.58–0.67).
   On-the-fly KD cannot fix this by construction — it distills the teacher's
   distribution over *the dataset's own target tokens*. **Needs maintainer
   approval:** it changes the official data mixture (AGENTS.md 4.4). Cost:
   phase-1 rewrite of 18.3k targets in 3 groups **$2–4**, plus a **$4–5**
   comparison run. Branch it from `s2v1_from_init@2700`.
2. **Measure GPU run-to-run variance** (~$1–2: re-run one arm's final leg with a
   different seed, or the same seed twice). It would justify or shrink the 1%
   decision band that two verdicts now rest on, and is the cheapest way to firm
   up the ablation's conclusions.
3. Stage 4 online data collection design (unchanged, still after Stage 3).
4. Optional backlog: Stage 1 ablations; a from-init-tuned lr/warmup sweep (A2
   was run under the ladder's hyperparameters, so single-stage may have more
   headroom than measured).

## Open decisions for the user

- **Teacher-generated answers** — approval to change the official data mixture,
  plus the $6–9 budget for generation + comparison run. Recommended next.
- **README "Optim record history" entries.** Four Stage 3 runs now have
  reproducible records (s1, the A/B, `s2_blocks_v1`, the ablation). AGENTS.md
  3.8/P12 require maintainer approval before any appears as an official record;
  **nothing has been added** (user held on 2026-07-27). Say the word and the
  entries get written from the existing logs.

## Links

- `logs/experiments/2026-07-27_stage3_start_point_ablation.md` (latest run + review)
- `logs/experiments/2026-07-26_stage3_s2_blocks_v1_gpu_run.md` ·
  `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md` ·
  `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`
- `logs/experiments/2026-07-26_stage2_offline_v1.md` ·
  `logs/experiments/2026-07-26_int8_fakequant_eval.md`
- `logs/proposals/2026-07-27_stage2_teacher_generated_answers.md` (awaiting approval)
- `logs/decisions.md` · `logs/supported_models.md` · `logs/artifact_manifests.md`
- `scripts/pod/AGENTS.md` (GPU session playbook)
