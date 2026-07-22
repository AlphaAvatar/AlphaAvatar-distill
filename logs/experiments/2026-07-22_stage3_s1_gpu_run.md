# 2026-07-22 — Stage 3 sub-stage 1: first real recovery run (GPU)

- **Agent:** Claude Code (Fable 5), executing the run approved by the user on
  2026-07-22 (see decisions.md 2026-07-22 and STATE.md).
- **Git commit:** `96c30ce` (adds `configs/stage3_s1_gpu_smoke.json`; trainer
  from `0a97701`). Working tree clean on the pod (verified `git log` after
  sync; pod-local `pyproject.toml` torch-index edit documented below).
- **Objective:** Recover the Stage 1-initialized 0.6B student with the
  config-driven trainer: masked CE 0.25 + on-the-fly full-vocab forward-KL
  KD 1.0 (τ=1), FFN+norm trainable, attention/embeddings frozen.
- **Hypothesis:** Structural init + plain KD materially closes the gap from
  init NLL 11.75 toward teacher 2.63 on holdout_v1 without collapse.
- **Teacher:** `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`, bf16.
- **Student:** `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`
  (sha256 `86fbba78…3e54`, bit-verified after transfer), fp32 master +
  bf16 autocast.
- **Stage:** Stage 3, recovery sub-stage 1 (FFN + norms).
- **Hardware:** RunPod secure-cloud pod `zae6ba3we52vgu`, 1× L40S 48 GB
  (driver 570.124.06), 16 vCPU, US-NC, $0.99/hr, 60 GB network volume.
- **Environment:** Python 3.14, **torch 2.11.0+cu128**, transformers 5.13.1,
  safetensors 0.8.0 (uv re-lock on pod). *Logged deviation:* local
  verification used torch 2.13.0+cpu; the cu128 wheel channel tops out at
  2.11.0 and driver 570 cannot run newer CUDA channels. Bridged by re-running
  the full test suite on the pod: **43/43 passed**. Venv on pod-local disk
  (`UV_PROJECT_ENVIRONMENT=/root/venv`); `HF_HOME=/workspace/hf`.
- **Budget (fixed before run):** 660 steps × 16 blocks × 1024 tokens
  (~10.8 M block-tokens ≈ 2 epochs of `stage2_offline_v0`), one L40S,
  single run, eval every 110 steps on 64 val blocks.

## Commands (pod, repo at `/workspace/AlphaAvatar-distill`)

```
uv run pytest tests/ -q                                              # 43 passed
uv run python scripts/train_stage3.py --config configs/stage3_s1_gpu_smoke.json
uv run python scripts/train_stage3.py --config configs/stage3_s1_gpu_smoke.json --resume step_000005
uv run python scripts/train_stage3.py --config configs/stage3_s1_ffn_norm.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model \
  --out artifacts/stage3/s1_ffn_norm_v0/eval_holdout_v1.json
```

## GPU smoke (10 steps, real batch shape)

- ~2.8–3.05 s/step at the full 16×1024-token batch; no OOM on 46 GB.
- val_ce 12.418621 → 6.096098, val_kd 11.339111 → 5.004668 (16 blocks).
- **Resume check (P5):** `--resume step_000005` reproduced step 6 identically
  to all logged decimals (exact state restore), then drifted ~1e-4…1e-3
  relative per step (cross-process GPU kernel nondeterminism; final eval
  val_ce 6.098121 vs 6.096098). Logged as the GPU-run variance scale.

## Full run result (`s1_ffn_norm_v0`)

- 660/660 steps, 2007 s train wall-clock (~2.83 s/step, ~5.8 k tok/s incl.
  teacher forward), rolling checkpoints at 440/550/660.
- Stage 2 val (64 blocks): step 0 **val_ce 12.00856 / val_kd 11.09149** →
  step 660 **val_ce 2.180522 / val_ppl 8.8509 / val_kd 1.006482**; monotone
  improvement at every eval (110/220/330/440/550/660), no spikes, no NaN.
- **holdout_v1 NLL (bf16 eval): 4.2107 (ppl 67.4)** vs Stage 1 baselines:
  teacher 2.63 / init 11.75 / random 12.13. Recovery removes ~7.5 nats of
  the ~9.1-nat init-teacher gap; remaining gap 1.58 nats.
- **Generation smoke:** fluent valid tokens, correct chat behavior
  (`"Okay, 2+2 = 4.<|im_end|>"` with proper template + termination).
  Factual/code quality still weak (expected at s1): e.g. fibonacci prompt
  yields a number list, "Water boils at 1000°C".
- Peak VRAM not sampled during the run (no OOM at full batch; smoke showed
  the same batch shape fits comfortably). Precise memory profiling deferred.

## Gate check (AGENTS.md 4.5)

- reproducible from logged command/config — **yes** (run_manifest.json:
  config hash, data manifest hashes, tokenizer hash, teacher revision, code state);
- checkpoint resume — **yes** (exact state restore verified on GPU);
- loss + val proxy logged — **yes** (append-only train_log.jsonl);
- no exploding activations / collapse — **yes** (monotone val curve);
- generation smoke valid tokens — **yes**;
- autoregressive behavior improves — **yes** (holdout 11.75 → 4.21);
- latency/memory — throughput logged; precise VRAM profiling deferred;
- quantized eval — **deferred by plan** (INT8 eval path is a later Stage 3+
  milestone; see STATE.md next actions);
- documented — this log.

**Verdict: Stage 3 sub-stage 1 gate PASSED.**

## Cost and infrastructure notes

- Pod time ≈ 4.5 h total ≈ $4.5 (of which training ~40 min; the rest was
  environment setup and slow China→US artifact transfer). Within the
  approved envelope; exact billing in RunPod console.
- Transfer lessons (for future runs): direct single-stream ssh upload
  collapses at night (~20 KB/s); parallel chunked ssh (24 streams, 16 MB
  chunks, per-chunk sha256) and croc relay (~285 KB/s) work; pod→dev
  download is much faster (~1.9 MB/s). The /workspace MooseFS volume serves
  stale reads shortly after write bursts — always hash-verify, assemble with
  a verifying reader + fsync. Venv must live on pod-local disk.

## Artifacts

- Local (gitignored): `artifacts/stage3/s1_ffn_norm_v0/`
  — `train_log.jsonl`, `run_manifest.json`, `eval_holdout_v1.json`,
  `checkpoints/step_000660/model/` (fp32, sha256 `dc64f244…e900`,
  bit-verified after download) + tokenizer files; console logs under
  `artifacts/stage3/`. GPU-smoke jsonl + manifests under
  `artifacts/stage3/s1_gpu_smoke_v0/`.
- Intentionally not retained: optimizer state `trainer_state.pt` (2.1 GB;
  run is complete, sub-stage 2 starts a fresh optimizer), smoke checkpoints,
  intermediate rolling checkpoints 440/550 (deleted with the pod).
- Pod terminated after hash-verified download (see STATE.md).

## Next action

1. Decide sub-stage 2 (unfreeze attention, block-level recovery) sizing
   under a fixed budget, using s1 as baseline (holdout 4.21 / val_ce 2.18).
2. INT8/fake-quant eval path (deployment target is INT8 — P9).
3. Stage 2 val + holdout tracking scripts comparison for future runs
   (stage3 checkpoints need tokenizer files copied in for eval_ppl).
