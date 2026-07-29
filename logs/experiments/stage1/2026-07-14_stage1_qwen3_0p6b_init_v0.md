# Stage 1 — projection + sandwich init of the 0.6B student (qwen3_0p6b_init_v0)

- **Date:** 2026-07-14
- **Agent:** Claude Code session (student architecture user-approved 2026-07-13)
- **Git commit:** work delivered in this session's commit (recipe hash `40b565ea…` binds config `9c71022e…` + code state)
- **Objective:** Produce the first fully initialized 0.6B-class student checkpoint from Qwen3-4B-Thinking-2507 using Stage 0 v1 activation statistics, with a random-init baseline comparison.
- **Teacher:** Qwen/Qwen3-4B-Thinking-2507 @ `768f209d`, bf16, CPU.
- **Student:** hidden 1024, 28 layers, FFN 3072, 16Q/8KV @128, tied embeddings, vocab 151936 → **596,049,920 params** (Qwen3-0.6B geometry). Deployment target INT8 (BF16 for now).
- **Stage:** Stage 1 (projection and structural initialization).
- **Hardware:** CPU-only dev box (16 threads AMX/AVX-512 BF16, 30 GB RAM). Init itself takes ~12 s + model load/save.
- **Budget:** single CPU session; eval = 21,080 held-out tokens (40 fineweb-edu docs, `data/warmup/holdout_v1.manifest.json`).

## Method (recipe v0, as committed)

- Single global stream projection `P` (2560→1024): eigenvectors of the trace-normalized
  average of **uncentered** second moments over all 37 residual points, with end points
  upweighted 9×/8× (embedding output, post-final-norm) — see decision record 2026-07-14.
- Sandwich init `P^T W P` with exact RMSNorm folding into following linears,
  `sqrt(d_t/d_s)` = 1.581 scale compensation on stream-reading projections.
- GQA-preserving Q-head selection 32→16 (top-2 of 4 per KV group, weight-norm proxy).
- FFN top-k 9728→3072 by cached `E|a_j| · ‖down_col_j‖` per representative layer.
- Depth 36→28: **middle-band pairwise merge** (t4..t19 → 8 student layers,
  first-of-span representative; t0-3 and t20-35 map 1:1).
- Tied embedding `E·P`; final norm via data-aware least-squares diagonal.

## Commands

```
uv run python scripts/init_stage1.py --config configs/stage1_qwen3_0p6b_from_4b_thinking.json
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model Qwen/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea \
  --model artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
  --model artifacts/stage1/qwen3_0p6b_init_v0/random_baseline \
  --out artifacts/stage1/qwen3_0p6b_init_v0/eval_holdout_v1.json
```

## Result (held-out, 21,080 tokens)

| Model | NLL (nats/tok) | PPL | top-1 (10-doc probe) |
| --- | --- | --- | --- |
| Teacher 4B | 2.626 | 13.8 | — |
| **PCA/sandwich init (final recipe)** | **11.748** | 126.5K | 0.0175 |
| Random-init baseline (same geometry) | 12.129 | 185.1K | 0.0000 |
| First-attempt init (early-band depth merge, unweighted P) | 17.798 | 53.6M | 0.0012 |

Gate checks: checkpoint loads; forward finite; 596.0M params as designed; shape/algebra
correctness covered by `tests/test_stage1_toy.py` (identity-projection exactness);
weights round-trip bitwise on save/reload; init is bitwise reproducible across runs.
Measured nondeterminism: two model instances with identical bf16 weights differ by up
to ~0.32 in logits on this CPU (oneDNN/AMX alignment-dependent kernels); logged, not a
weight issue. Projection energy captured (weighted avg): 0.932.

## Diagnosis history (kept per P11 — the first attempt failed usefully)

First full init evaluated **worse than uniform** (NLL 17.80; uniform ≈ 11.93).
Temperature sweep ruled out logit scale. Single-axis ablation (10-doc probe, NLL / top-1):

| Axis alone | NLL | top-1 |
| --- | --- | --- |
| Width 2560→1024 (orig P) | 11.19 | 0.036 |
| Width 2560→1024 (end-weighted P) | 10.83 | 0.082 |
| Depth 36→28, early-band merge | 10.48 | 0.008 |
| Depth 36→28, middle-band, first-of-span rep | **3.88** | **0.317** |
| Depth 36→28, middle-band, last-of-span rep | 7.54 | 0.116 |
| Heads 32Q→16Q | 5.47 | 0.195 |
| FFN 9728→3072 | 4.64 | 0.278 |

Also measured: per-point energy capture of the unweighted P was ≥0.94 mid-stream but
only 0.74/0.75 at the embedding-output / post-final-norm points — the two interfaces
wired to the tied embedding and lm head. Both fixes were adopted (decision record
2026-07-14) and are what the final recipe above uses.

## Verdict

**Stage 1 gate: PASS** (all gate items satisfied; baseline comparison exists).
Quality assessment, honestly stated: the full-compression init beats random init
(ΔNLL −0.38 nats, top-1 1.75% vs 0%) but is far from the teacher zero-shot; the
**width cut is the dominant bottleneck** (width-only alone lands at NLL 10.8).
This matches Minitron-style pruning literature, where pruned-before-recovery students
are near-noise and quality is restored by distillation. The depth/FFN/head axes
transfer strong signal already.

## Next action

1. Stage 2: offline warm-up data collection (data groups + manifests).
2. Stage 3: student recovery (FFN/norm → block → span → full KD), which is the
   designed repair path for the width damage.
3. Optional Stage 1 ablation backlog: function-aware width subspace (logit-gradient
   weighted PCA), per-group P with Procrustes chaining, activation-based head importance.

## Resume instructions

Checkpoint + manifest + eval report live under `artifacts/stage1/qwen3_0p6b_init_v0/`
(gitignored; regenerate deterministically with the two commands above — init is
bitwise-reproducible given the same environment and Stage 0 cache).
