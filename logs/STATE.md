# Current project state

Updated: 2026-07-22 (UTC+8 dev box) — Stage 3 trainer implemented and
verified at toy + real-model smoke scale; first GPU recovery run awaiting
user approval.

## Status

First dense-model compression experiment, teacher **Qwen/Qwen3-4B-Thinking-2507**
@ `768f209d` (hidden 2560, 36 layers, FFN 9728, 32Q/8KV).

User decisions (2026-07-13, see decisions.md): student target **0.6B-class**
(hidden 1024, 28 layers, FFN 3072, 16Q/8KV, tied emb — Qwen3-0.6B geometry);
**BF16 training, INT8 deployment target**; warm-up v1 public-corpus download approved.

Pipeline position: **Stage 0 passed → Stage 1 passed → Stage 2 passed →
Stage 3 in progress** (trainer verified; no real recovery run yet). New
decision this session (2026-07-22, see decisions.md): one config-driven
trainer for all recovery sub-stages, on-the-fly full-vocab forward-KL KD
(τ=1, scope "all") + masked CE 0.25, fp32 master weights + bf16 autocast,
stateless-exact resume; sub-stages 2–3 (span losses) deferred until the
plain-KD baseline is measured.

Verified state:

- initialized student checkpoint `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`
  (596.0M params, bf16); holdout NLL: teacher 2.63 / init 11.75 / random 12.13.
- Stage 2 offline mixture `stage2_offline_v0`: 8 groups, 18,484 train
  samples / 5.39M tokens, 771 val, 120 calib (= INT8 calibration set).
- **Stage 3 trainer** (`aadistill.train` + `scripts/train_stage3.py`):
  43/43 tests pass (10 new: loss correctness, freeze policy, deterministic
  block stream, **bitwise-exact resume**, config-refusal on resume). Real
  CPU smoke (`configs/stage3_smoke_cpu.json`): full path ran — mixture →
  teacher+student → 3 KD+CE steps (loss 17.64 → 14.85) → checkpoints →
  evals → jsonl trail; process-restart resume reproduced step 3 and the
  final eval identically to all logged decimals.

## Environment

- CPU-only dev box: 16 threads (AMX/AVX-512 BF16), 30 GB RAM, no GPU.
- `uv sync`: Python 3.14, torch 2.13.0+cpu, transformers 5.13.1, safetensors 0.8.0,
  datasets, pytest.
- Known nondeterminism (logged per P5): two model instances with bitwise-identical
  bf16 weights can differ by a few ULPs in logits (oneDNN/AMX alignment-dependent);
  each instance is self-deterministic. (The Stage 3 resume smoke nevertheless
  reproduced identically at logged precision.)
- RunPod control plane verified read-only 2026-07-16: `runpodctl` 2.7.1
  authenticated, balance $250 / $80 spend limit, no pods or volumes, SSH keys
  present. Skill at `.agents/skills/runpodctl`.
- HF cache ~12 GB (7.6 GB teacher + Stage 2 source datasets).

## What exists and why

- `src/aadistill/` — `env.py`, `manifest.py`, `teacher.py`, `collect.py`
  (Stage 0), `project.py`, `sandwich.py`, `student.py` (Stage 1), `data.py`
  (Stage 2 loader), and new this session: `train.py` — Stage 3 recovery
  trainer (losses, freeze policy, LR schedule, deterministic block stream,
  rolling checkpoints with exact resume, jsonl logging).
- `scripts/` — Stage 0/1/2 scripts as before; new: `train_stage3.py`
  (config-driven CLI, `--resume [TAG]`, run manifests).
- `configs/` — Stage 0 v0/v1, Stage 1 init, and new: `stage3_s1_ffn_norm.json`
  (recovery sub-stage 1, GPU-sized draft — not yet run) and
  `stage3_smoke_cpu.json` (3-step smoke, already exercised).
- `data/warmup/`, `data/stage2/` — corpora manifests (jsonl gitignored).
- `tests/` — 43 tests total; new `tests/test_train_toy.py` (10).
- `artifacts/` (gitignored) — Stage 0 stats, Stage 1 checkpoint + eval,
  Stage 2 dry-run report, new: `stage3/smoke_cpu_v0/` (smoke run output:
  jsonl log, run + resume manifests, rolling checkpoints).
- `logs/` — decisions (7 records), experiments (5), supported_models, this file.

## Latest known working commands

```
uv run pytest tests/ -q                                                  # 43 passed
uv run python scripts/train_stage3.py --config configs/stage3_smoke_cpu.json
uv run python scripts/train_stage3.py --config configs/stage3_smoke_cpu.json --resume step_000002
uv run python scripts/train_stage3.py --config configs/stage3_s1_ffn_norm.json   # REAL RUN — GPU, needs approval
uv run python scripts/dry_run_stage2.py
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl --model <dir-or-hf-id> ...
```

## Latest verification

- Stage 3 trainer: verified on toy models (bitwise resume, loss math) and
  on the real teacher/student at smoke scale, 2026-07-22. See
  `logs/experiments/2026-07-22_stage3_trainer_toy.md`.
- Stage 2 gate passed 2026-07-21; Stage 1 gate 2026-07-14; Stage 0 v1 2026-07-13.

## Not done yet (next, in order)

1. **Stage 3 first real run (needs user approval — P12):** recovery
   sub-stage 1 via `configs/stage3_s1_ffn_norm.json`. Proposed execution
   (P8.2 request):
   - *Operation:* GPU smoke (~10 steps, same config, early stop) then the
     full 660-step run (~10.8M tokens ≈ 2 epochs of mixture v0) on one pod.
   - *Why not CPU:* ~30 s/step CPU ⇒ ~5.5 h/epoch of pure teacher+student
     compute at batch 1; the real config (16-block batches) is ~16× that.
   - *Hardware:* recommended **1× L40S 48 GB** (or A100 80 GB PCIe);
     minimum RTX 4090 24 GB with `micro_blocks: 2`. Memory estimate:
     teacher bf16 ~8 GB + student fp32 ~2.4 GB + grads ~1 GB + AdamW
     moments ~2.1 GB (FFN/norm only, 264M trainable) + logits/activations
     ~4–8 GB ⇒ ~18–22 GB peak.
   - *Runtime/cost:* est. 1–2 h on L40S (~$1–3 at current RunPod rates);
     well under the $80 spend limit. Disk: HF teacher download 8 GB +
     3 rolling checkpoints ~10 GB ⇒ 40 GB volume is enough.
   - *Precision:* teacher bf16, student fp32 master + bf16 autocast (per
     2026-07-22 decision), matching the smoke-verified path.
   - *Gate it must pass:* AGENTS.md 4.5 — reproducible from logged
     command/config, resumable, loss + val proxy logged, no collapse,
     generation smoke, improvement over init baseline (holdout NLL 11.75)
     or documented failure analysis.
2. After s1: holdout_v1 + stage2-val evaluation vs init/random baselines,
   generation smoke test, then decide sub-stage 2 (unfreeze attention) /
   sub-stage 4 (full offline KD) sizing under a fixed budget.
3. Later stages: INT8/fake-quant eval path (Stage 3+ per precision policy),
   Stage 4 online data collection design.
4. Optional backlog: Stage 2 mixture scale-up (approval needed),
   Stage 1 ablations (function-aware subspace, per-group P).

## Open decisions for the user

- None blocking. The Stage 3 GPU run (item 1 above) was **approved by the
  user on 2026-07-22** ("push code and do it tomorrow") with execution
  deferred to 2026-07-23: rent one RunPod L40S-class pod, GPU smoke
  (~10 steps), then the full `stage3_s1_ffn_norm.json` run. Re-confirm cost
  ceiling at rental time if prices differ materially from the ~$1–3 estimate.

## Links

- `logs/experiments/2026-07-22_stage3_trainer_toy.md` (this session)
- `logs/experiments/2026-07-21_stage2_offline_v0.md`
- `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md`
- `logs/experiments/2026-07-13_stage0_qwen3_4b_thinking_v1.md`
- `logs/decisions.md` (7 records; 1 added 2026-07-22)
- `logs/supported_models.md`
