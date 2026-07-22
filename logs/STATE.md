# Current project state

Updated: 2026-07-23 (UTC+8 dev box) — Stage 3 recovery sub-stage 1 **ran on
GPU and passed its gate**; pod torn down; all artifacts hash-verified local.

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV) → student 0.6B-class
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb). BF16 training,
INT8 deployment target.

Pipeline position: **Stage 0 passed → Stage 1 passed → Stage 2 passed →
Stage 3 s1 passed (2026-07-22)**; sub-stages 2+ pending a sizing decision.

Verified state (all on the real model):

- **Stage 3 s1 recovery run** (`s1_ffn_norm_v0`, 660 steps × 16×1024-token
  blocks ≈ 2 epochs of mixture v0, 1× RunPod L40S, 33.5 min train):
  - stage2-val: val_ce 12.009 → **2.1805** (ppl 8.85), val_kd 11.091 → 1.006,
    monotone at every eval, no collapse;
  - **holdout_v1 NLL 4.2107 (ppl 67.4)** vs teacher 2.63 / init 11.75 /
    random 12.13;
  - generation smoke passed (valid fluent tokens; chat template + termination
    correct: `"Okay, 2+2 = 4.<|im_end|>"`); factual/code quality still weak
    (expected before sub-stages 2+);
  - GPU resume check: exact state restore (first replayed step identical to
    all logged decimals); cross-process GPU drift ~1e-4…1e-3 relative/step
    (P5-logged variance scale);
  - full log: `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`.
- Stage 1 init checkpoint `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`
  (596.0M params, bf16); Stage 2 mixture `stage2_offline_v0` (18,484 train
  samples / 5.39M tokens, 771 val, 120 calib).
- Trainer: 43/43 tests pass locally (torch 2.13.0+cpu) **and on the GPU pod**
  (torch 2.11.0+cu128 — cu128 channel max; logged deviation).

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
  `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1.
- GPU runs: RunPod (runpodctl 2.7.1 authenticated; skill at
  `.agents/skills/runpodctl`). Balance ≈ $247 after this run (~$4.3 total),
  $80 spend limit. **No pods or volumes currently exist** (pod
  `zae6ba3we52vgu` deleted 2026-07-23 after hash-verified artifact download).
- Pod playbook (hard-won, see experiment log §infrastructure): venv on
  pod-local disk (`UV_PROJECT_ENVIRONMENT=/root/venv`, network volume fails
  on venvs and serves stale reads after write bursts — always sha256-verify);
  torch cu128 (2.11.0) is the max for driver 570; China→US upload needs
  parallel chunked ssh or croc relay at night; pod→dev download ~1.9 MB/s.
- Known CPU nondeterminism (oneDNN/AMX ULP-level, P5-logged) unchanged.
- HF cache ~12 GB (7.6 GB teacher + Stage 2 source datasets).

## What exists and why

- `src/aadistill/` — env, manifest, teacher, collect (S0), project, sandwich,
  student (S1), data (S2 loader), train (S3 recovery trainer).
- `scripts/` — stage scripts + `train_stage3.py`, `eval_ppl.py`,
  `plot_perf_trend.py`.
- `configs/` — Stage 0 v0/v1, Stage 1 init, `stage3_s1_ffn_norm.json` (ran),
  `stage3_s1_gpu_smoke.json` (10-step GPU smoke, ran), `stage3_smoke_cpu.json`.
- `data/warmup/`, `data/stage2/` — corpora manifests (jsonl gitignored).
- `tests/` — 43 tests.
- `artifacts/` (gitignored) — Stage 0 stats; Stage 1 checkpoint; Stage 3:
  `s1_ffn_norm_v0/` (train_log.jsonl, run_manifest.json,
  eval_holdout_v1.json, `checkpoints/step_000660/model/` **final fp32
  student, sha256 `dc64f244…e900`, bit-verified**, tokenizer files included),
  `s1_gpu_smoke_v0/` (jsonl + manifests), console logs. Not retained:
  optimizer state (2.1 GB), smoke checkpoints, rolling checkpoints 440/550.
- `logs/` — decisions (7), experiments (6), supported_models, this file.
- `assets/` — perf trend json + svg (now 3 attempt points incl. s1).

## Latest known working commands

```
uv run pytest tests/ -q                                          # 43 passed
uv run python scripts/train_stage3.py --config configs/stage3_s1_ffn_norm.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model
uv run python scripts/plot_perf_trend.py
```

Note: stage3 checkpoints are saved as `step_XXXXXX/model/` +
`trainer_state.pt`; copy tokenizer files into `model/` before `eval_ppl.py`
(done for the retained final checkpoint).

## Latest verification

- Stage 3 s1 gate passed 2026-07-22 (GPU run, evals, generation smoke,
  resume check) — `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`.
- Stage 2 gate 2026-07-21; Stage 1 gate 2026-07-14; Stage 0 v1 2026-07-13.

## Not done yet (next, in order)

1. **Decide sub-stage 2 sizing** (unfreeze attention, block-level recovery)
   under a fixed budget, with s1 as the baseline (holdout 4.21 / val_ce
   2.18). Also consider whether a longer s1 (val curve had not plateaued at
   660 steps) is worth one comparison run before unfreezing.
2. INT8/fake-quant eval path (deployment target INT8 — P9); calib set exists.
3. Stage 4 online data collection design.
4. Optional backlog: Stage 2 mixture scale-up (approval needed), Stage 1
   ablations (function-aware subspace, per-group P).

## Open decisions for the user

- Sub-stage 2 recovery run (GPU, similar cost ≈ $5): needs approval when
  the sizing proposal is ready (P12).
- Whether the s1 result should become an official README "Optim record
  history" entry (requires maintainer approval per AGENTS.md 3.8; the
  reproducible record exists in the experiment log).

## Links

- `logs/experiments/2026-07-22_stage3_s1_gpu_run.md` (this session)
- `logs/experiments/2026-07-22_stage3_trainer_toy.md`
- `logs/experiments/2026-07-21_stage2_offline_v0.md`
- `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md`
- `logs/experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md`
- `logs/decisions.md` · `logs/supported_models.md`
