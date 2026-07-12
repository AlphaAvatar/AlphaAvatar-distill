# Current project state

Updated: 2026-07-12 — Stage 0 collection **run and passed** on the real teacher.

## Status

First dense-model compression experiment. Teacher (user-directed):
**Qwen/Qwen3-4B-Thinking-2507**, revision `768f209d9ea81521153ed38c47d515654e938aea`
(local HF cache, ~7.6 GB; hidden 2560, 36 layers, intermediate 9728, vocab 151936).

**Stage 0 (initialization warm-up data collection) is complete and passed its
validation gate** on the real teacher (CPU). Stage 1 is next but blocked on the
student-architecture decision.

## Environment

- CPU-only dev box: 16 threads (AMX-BF16 + AVX-512 BF16), 30 GB RAM, no GPU.
- `uv sync`: Python 3.14, torch 2.13.0+cpu (pinned CPU index), transformers 5.13.1,
  safetensors 0.8.0, pytest. huggingface.co reachable directly.

## What exists and why

- `src/aadistill/env.py` — env fingerprint, git code-state hash, determinism seeding.
- `src/aadistill/manifest.py` — sha256 + JSON manifest helpers.
- `src/aadistill/teacher.py` — teacher loading with pinned revision, tokenizer hash, identity record.
- `src/aadistill/collect.py` — `ActivationStatsCollector`: streaming sufficient statistics
  (residual sum + X^T X per point in f64, FFN per-neuron |a| and a^2, token counts).
  `residual_covariance()` is the Stage 1 consumption entry point.
- `scripts/collect_stage0.py` — Stage 0 CLI. Fixed this session: chat-format encode now
  extracts `input_ids` from the transformers 5.x BatchEncoding.
- `configs/stage0_qwen3_4b_thinking.json` — Stage 0 config (pinned revision, bf16, seed
  20260712, max_seq_len 1024, cache_budget_gb 2.5). config sha256 `1bdbafcb…`.
- `data/warmup/warmup_v0.jsonl` — 47 handcrafted, license-clean warm-up samples across
  12 categories. sha256 `2dcb4dee…`.
- `tests/test_collect_toy.py` — 8 CPU tests (stats vs direct compute, FFN stats, token
  counts, determinism, save/load roundtrip, PSD covariance, batch/shape rejection). All pass.
- `logs/` — this STATE, `decisions.md`, `supported_models.md`, `experiments/`.
- `artifacts/stage0/qwen3_4b_thinking_v0/` (gitignored) — `activation_stats.safetensors`
  (1.95 GB) + `manifest.json`.

## Verification done

- `uv run pytest tests/ -q` → 8 passed.
- Stage 0 dry run (`--limit 2`) then full run: 47 samples, 4068 tokens, 24 s.
  Determinism bitwise-identical. Projection dry run PSD (min eig +0.0014, 191 supported
  directions). Cache 1.95 GB < 2.5 GB budget. Full manifest written.

## Not done yet (next session, in order)

1. **Student architecture target decision** (params, hidden size, depth, latency/precision
   budget). Blocks Stage 1. Open user decision.
2. Stage 1: grouped activation-PCA hidden projection, sandwich attention init, activation-
   importance FFN top-k selection, depth-span mapping — consuming the Stage 0 cache.
3. Optional: approve downloading a small public corpus for a larger warm-up v1 (current
   v0 is statistically thin: 4068 tokens vs 2560-dim covariance).

## Generated artifacts

- Stage 0 cache + manifest under `artifacts/` — gitignored (confirmed via `git check-ignore`).
  Regenerate with the Stage 0 command; not stored in git.

## Open decisions for the user

- Student architecture target (blocks Stage 1).
- Whether to approve a public-corpus download for warm-up v1.

## Links

- Experiment log: `logs/experiments/2026-07-12_stage0_qwen3_4b_thinking_v0.md`
- Decisions: `logs/decisions.md`
- Model table: `logs/supported_models.md`
