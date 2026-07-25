# Current project state

Updated: 2026-07-26 (UTC+8 dev box) — **INT8 fake-quant eval path landed
and measured** (P9): INT8 weight quant is near-lossless on s1@660 (+0.21%
holdout NLL full-scope, +0.03% decoder-only; 51/51 tests). **Stage 2
mixture v1 scale-up proposal drafted — awaiting user approval**
(`logs/proposals/2026-07-26_stage2_mixture_v1_scaleup.md`): ~24M train
tokens (4.5×), public permissive sources only, licenses verified. Recovery
remains data-limited per the 2026-07-25 A/B (attention-unfrozen freeze set
adopted; epochs 3–4 of mixture v0 overfit).

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV) → student 0.6B-class
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb). BF16 training,
INT8 deployment target.

Pipeline position: **Stage 0 passed → Stage 1 passed → Stage 2 passed →
Stage 3 s1 passed (2026-07-22) → s2 sizing A/B done (2026-07-25)**.
Freeze-set recipe fixed; next recovery run blocked on Stage 2 mixture
scale-up (user approval required).

Sub-stage 2 A/B result (design: decision record 2026-07-25; full record:
`logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md`). Both arms from
s1@660, equal budget (660 steps × 16×1024 blocks, lr 2e-4, CE 0.25 +
KD 1.0, shared seed 20260725, 1× L40S):

- **arm A** (control, FFN+norm only, 264.3M trainable): val_ce flat
  2.602→2.633, holdout NLL **4.2747** — *regressed* 1.5% vs s1@660;
- **arm B** (+attention q/k/v/o + q/k norms, 440.5M): val_ce 2.602→**2.579**,
  val_kd 1.061→**0.987**, holdout NLL **4.2118** — flat vs s1@660 (4.2107);
  generation smoke shows mild chat-format regression (corpus artifacts:
  stray `</think>`, `####`, one `<|im_start|>` echo).

Pre-registered rule fired (B beats A by 1.47% ≥ 1%): **attention-unfrozen
freeze set adopted** for all further Stage 3 recovery. Both arms were on
mixture epochs 3–4; holdout-flat + artifact pickup = **data exhaustion —
the binding constraint is now Stage 2 data volume, not the recipe**.
Cost: B is only +3.3% s/step (2.974 vs 2.878), peak VRAM 36.97 GB.
s1@660 remains the reference checkpoint; `s2_blocks_v0` final is the
preferred start for the next recovery run once fresh data exists.

Verified state (all on the real model):

- **INT8 fake-quant eval (2026-07-26, CPU):** holdout_v1 on s1@660 —
  bf16 4.2111 (matches GPU 4.2107 within P5 variance), int8 decoder-only
  4.2122 (+0.03%), int8 full-scope incl. tied head 4.2198 (+0.21%).
  Deployment-numerics gap negligible at this stage; ~14 s per CPU eval, run
  it at every future recovery gate. `src/aadistill/quant.py`,
  `eval_ppl.py --fake-quant int8`;
  log: `logs/experiments/2026-07-26_int8_fakequant_eval.md`.
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
- Tests: 51/51 pass locally (torch 2.13.0+cpu; 43 of them also passed on
  the GPU pod 2026-07-25, torch 2.11.0+cu128 — cu128 channel max; the 8
  quant tests are newer and CPU-verified only).

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
  `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1.
- GPU runs: RunPod (runpodctl 2.7.1 authenticated; skill at
  `.agents/skills/runpodctl`). Balance **$244.32** after the A/B session
  ($2.03; ~$6.4 total project GPU spend), $80 spend limit. **No pods or
  volumes currently exist** (pod `simbeepnf8syuu` deleted 2026-07-25 after
  artifacts were verified on HF).
- HF: dev box logged in as `AlphaAvatar` (write token, added 2026-07-25);
  private artifact repo `AlphaAvatar/aadistill-artifacts` holds s1@660 and
  both A/B finals (see `logs/artifact_manifests.md`).
- Pod playbook (hard-won, see experiment logs §infrastructure): venv on
  pod-local disk (`UV_PROJECT_ENVIRONMENT=/root/venv`); torch cu128
  (2.11.0) is the max for driver 570; **preferred big-file transfer is the
  private HF repo relay** (dev→HF ~680 KB/s, pod↔HF fast; single-stream
  scp dev→pod only ~165 KB/s); always sha256-verify after transfer; use
  `--terminate-after` as a cost backstop. 2026-07-25 pod's /workspace was a
  local md array (no MooseFS stale reads); older network-volume caveats
  still apply when a network volume is attached.
- Known CPU nondeterminism (oneDNN/AMX ULP-level, P5-logged) unchanged.
- HF cache ~12 GB (7.6 GB teacher + Stage 2 source datasets).

## What exists and why

- `src/aadistill/` — env, manifest, teacher, collect (S0), project, sandwich,
  student (S1), data (S2 loader), train (S3 recovery trainer), quant
  (INT8 fake-quant eval, P9).
- `scripts/` — stage scripts + `train_stage3.py`, `eval_ppl.py`,
  `plot_perf_trend.py`.
- `configs/` — Stage 0 v0/v1, Stage 1 init, `stage3_s1_ffn_norm.json` (ran),
  `stage3_s1_gpu_smoke.json` (ran), `stage3_smoke_cpu.json`,
  `stage3_s1_ext.json` + `stage3_s2_blocks.json` (A/B arms, ran 2026-07-25),
  `stage3_s2_smoke_cpu.json` (3-step CPU smoke, ran 2026-07-25).
- `data/warmup/`, `data/stage2/` — corpora manifests (jsonl gitignored).
- `tests/` — 51 tests.
- `logs/proposals/` — pending-approval proposals (currently: Stage 2
  mixture v1 scale-up, 2026-07-26).
- `artifacts/` (gitignored) — Stage 0 stats; Stage 1 checkpoint; Stage 3:
  `s1_ffn_norm_v0/` (logs + `checkpoints/step_000660/model/` **final fp32
  student, sha256 `dc64f244…e900`, bit-verified**, also on HF),
  `s1_ext_v0/` + `s2_blocks_v0/` (A/B: train_log.jsonl, run_manifest.json,
  eval_holdout_v1.json, gen_smoke.json, console.log;
  **final weights HF-only**, hashes in
  `ab_artifact_hashes_2026-07-25.txt`), `s1_gpu_smoke_v0/`. Not retained:
  optimizer states, smoke checkpoints, rolling checkpoints.
- `logs/` — decisions (8), experiments (7), supported_models,
  artifact_manifests, this file.
- `assets/` — perf trend json + svg (now 5 attempt points incl. the A/B).

## Latest known working commands

```
uv run pytest tests/ -q                                          # 43 passed
uv run python scripts/train_stage3.py --config configs/stage3_s1_ffn_norm.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model \
  [--fake-quant int8 [--fake-quant-scope decoder]]
uv run python scripts/plot_perf_trend.py
```

Note: stage3 checkpoints are saved as `step_XXXXXX/model/` +
`trainer_state.pt`; copy tokenizer files into `model/` before `eval_ppl.py`
(done for the retained final checkpoint).

## Latest verification

- INT8 fake-quant path 2026-07-26: 51/51 tests on the dev box; three CPU
  holdout evals of s1@660 (bf16 reference reproduced the GPU number within
  P5 variance) — `logs/experiments/2026-07-26_int8_fakequant_eval.md`.
- Sub-stage 2 A/B GPU session 2026-07-25: 43/43 tests on pod, bit-verified
  checkpoint transfer, both arms trained without collapse, holdout evals +
  generation smoke run, artifacts hash-verified —
  `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md`.
- Stage 3 s1 gate passed 2026-07-22; Stage 2 gate 2026-07-21; Stage 1 gate
  2026-07-14; Stage 0 v1 2026-07-13.

## Not done yet (next, in order)

1. **User decision on the Stage 2 mixture v1 proposal**
   (`logs/proposals/2026-07-26_stage2_mixture_v1_scaleup.md`); on approval:
   `build_stage2_v1.py`, build + gate + experiment log (CPU session).
2. Next recovery run on the scaled mixture: adopted attention-unfrozen
   freeze set; start-point comparison s2_blocks_v0-final vs s1@660 (A/B
   verdict); separate GPU approval request (~$4–5 L40S).
3. Stage 4 online data collection design.
4. Optional backlog: Stage 1 ablations (function-aware subspace,
   per-group P); teacher-generated data proposal (v2 upgrade path for the
   QA groups + reasoning traces).

## Open decisions for the user

- **Stage 2 mixture v1 scale-up approval** — proposal drafted 2026-07-26,
  see `logs/proposals/2026-07-26_stage2_mixture_v1_scaleup.md` (~24M train
  tokens, public permissive sources, ≤3 GB downloads, no paid APIs; optional
  add-on question: accept the auto-gated xlam-function-calling-60k on the
  AlphaAvatar HF account for tool-calling variety?).
- Whether the s1 result (and/or the A/B) should become an official README
  "Optim record history" entry (requires maintainer approval per AGENTS.md
  3.8; reproducible records exist in the experiment logs).

## Links

- `logs/experiments/2026-07-26_int8_fakequant_eval.md` (this session)
- `logs/proposals/2026-07-26_stage2_mixture_v1_scaleup.md` (this session)
- `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md`
- `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`
- `logs/experiments/2026-07-22_stage3_trainer_toy.md`
- `logs/experiments/2026-07-21_stage2_offline_v0.md`
- `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md`
- `logs/experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md`
- `logs/decisions.md` · `logs/supported_models.md` ·
  `logs/artifact_manifests.md`
