# Current project state

Updated: 2026-07-25 late (UTC+8 dev box) — Stage 3 **sub-stage 2 A/B ran on
GPU and answered the sizing question**: attention-unfrozen freeze set
adopted (beats equal-budget FFN-only by 1.47% relative holdout NLL), but
**recovery is now data-limited** — neither arm improved holdout over s1@660
(mixture epochs 3–4 overfit). Pod torn down; artifacts on private HF repo,
hash-verified. Session cost $2.03.

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
  student (S1), data (S2 loader), train (S3 recovery trainer).
- `scripts/` — stage scripts + `train_stage3.py`, `eval_ppl.py`,
  `plot_perf_trend.py`.
- `configs/` — Stage 0 v0/v1, Stage 1 init, `stage3_s1_ffn_norm.json` (ran),
  `stage3_s1_gpu_smoke.json` (ran), `stage3_smoke_cpu.json`,
  `stage3_s1_ext.json` + `stage3_s2_blocks.json` (A/B arms, ran 2026-07-25),
  `stage3_s2_smoke_cpu.json` (3-step CPU smoke, ran 2026-07-25).
- `data/warmup/`, `data/stage2/` — corpora manifests (jsonl gitignored).
- `tests/` — 43 tests.
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
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model
uv run python scripts/plot_perf_trend.py
```

Note: stage3 checkpoints are saved as `step_XXXXXX/model/` +
`trainer_state.pt`; copy tokenizer files into `model/` before `eval_ppl.py`
(done for the retained final checkpoint).

## Latest verification

- Sub-stage 2 A/B GPU session 2026-07-25: 43/43 tests on pod, bit-verified
  checkpoint transfer, both arms trained without collapse, holdout evals +
  generation smoke run, artifacts hash-verified —
  `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md`.
- Stage 3 s1 gate passed 2026-07-22; Stage 2 gate 2026-07-21; Stage 1 gate
  2026-07-14; Stage 0 v1 2026-07-13.

## Not done yet (next, in order)

1. **Propose Stage 2 mixture scale-up** (the measured bottleneck; needs
   user approval — larger downloads, possibly teacher-generated data).
   Design the target size/composition against the observed overfit.
2. INT8/fake-quant eval path (deployment target INT8 — P9); calib set
   exists; CPU-suitable, should land before the next GPU run.
3. Next recovery run: from s1@660, adopted attention-unfrozen freeze set,
   on the scaled mixture (fresh data, not more epochs of v0).
4. Stage 4 online data collection design.
5. Optional backlog: Stage 1 ablations (function-aware subspace,
   per-group P).

## Open decisions for the user

- **Stage 2 mixture scale-up approval** (P12: larger downloads / possible
  teacher-generated data) — proposal to be drafted next session.
- Whether the s1 result (and/or the A/B) should become an official README
  "Optim record history" entry (requires maintainer approval per AGENTS.md
  3.8; reproducible records exist in the experiment logs).

## Links

- `logs/experiments/2026-07-25_stage3_s2_ab_gpu_run.md` (this session)
- `logs/experiments/2026-07-22_stage3_s1_gpu_run.md`
- `logs/experiments/2026-07-22_stage3_trainer_toy.md`
- `logs/experiments/2026-07-21_stage2_offline_v0.md`
- `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md`
- `logs/experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md`
- `logs/decisions.md` · `logs/supported_models.md` ·
  `logs/artifact_manifests.md`
