# Current project state

Updated: 2026-07-12 (session ended early at user request; Stage 0 in progress, not yet run)

## Status

First dense-model compression experiment kicked off. Teacher chosen by the user:
**Qwen/Qwen3-4B-Thinking-2507** (downloaded to local HF cache, 7.6 GB, snapshot
revision `768f209d9ea81521153ed38c47d515654e938aea`).

Stage 0 (initialization warm-up data collection) implementation is partially
complete. Code exists and passes a CPU smoke test on a tiny random Qwen3 model;
the real collection run has NOT been executed yet.

## Environment

- CPU-only dev box: 16 cores (AMX-BF16 + AVX-512 BF16 capable), 30 GB RAM, no GPU.
- `uv sync` works: Python 3.14 venv, torch 2.13.0+cpu (pinned to the pytorch-cpu
  index in `pyproject.toml`), transformers 5.13.1, safetensors 0.8.0, pytest.
- huggingface.co is reachable directly from this box.

## What exists and why

- `pyproject.toml`, `.gitignore` — project env + hygiene (pre-existed this session).
- `src/aadistill/env.py` — env fingerprint, git code-state hash, determinism seeding.
- `src/aadistill/manifest.py` — sha256 + JSON manifest helpers.
- `src/aadistill/teacher.py` — teacher loading with pinned revision, tokenizer hash,
  identity record.
- `src/aadistill/collect.py` — `ActivationStatsCollector`: streaming sufficient
  statistics instead of raw activation dumps (residual-stream sum + X^T X per
  collection point in float64, FFN per-neuron |a| and a^2 sums, token counts).
  `residual_covariance()` is the Stage 1 consumption entry point.
- `scripts/collect_stage0.py` — Stage 0 CLI: loads a JSON config, forwards a
  warm-up JSONL dataset through the teacher (batch=1, unpadded), runs a
  determinism check, enforces a cache budget, does a projection dry run
  (mid-layer eigendecomposition), writes `activation_stats.safetensors` +
  a full reproducibility `manifest.json`. References `configs/` and `data/`
  paths that DO NOT EXIST YET.

## Verification done

- Import + tiny-model smoke test passed on CPU (33-token sequence through a
  2-layer random Qwen3; shapes correct, covariance PSD, determinism flags set).
- Bug found and fixed during smoke test: transformers 5.x `hidden_states` has
  `num_layers + 1` points (embedding output = layer-0 input, inputs of layers
  1..N-1, final-norm output) — NOT num_layers + 2. `collect.py` now documents this.
- A full pytest suite for the collector was drafted but NOT written to disk
  (session ended); no `tests/` directory exists yet.

## Not done yet (next session, in order)

1. `tests/test_collect_toy.py` — pytest suite: stats vs direct computation,
   determinism across runs, save/load roundtrip, PSD covariance, batch rejection.
2. `configs/stage0_qwen3_4b_thinking.json` — teacher_model_id, teacher_revision
   (pin `768f209d9ea81521153ed38c47d515654e938aea`), dtype `bfloat16`, seed,
   max_seq_len (~1024), dataset path, output_dir under gitignored `artifacts/`,
   cache_budget_gb (~2; expected actual ~0.2 GB for 36 layers × 2560²·f64 sums —
   recheck: residual_sqsum is 37 × 2560² × 8 B ≈ 1.9 GB, so budget 2.5).
3. `data/warmup/warmup_v0.jsonl` — small handcrafted, license-clean warm-up set
   (~40-60 samples; categories: general text, instruction, reasoning, RAG/evidence,
   multi-hop, tool-call format, code, math, refusal/uncertainty, short realtime
   chat; fields: id, category, format: text|chat, text or messages). Known
   limitation to log: no teacher-generated responses (CPU too slow to generate);
   authored assistant turns only. Downloading a real public dataset requires
   user approval per AGENTS.md.
4. Dry run: `uv run python scripts/collect_stage0.py --config ... --limit 2`,
   then the full small run (est. tens of minutes on CPU with AMX bf16).
5. Logs to create after the run: decision record (teacher selection was
   user-directed; streaming-stats-not-raw-cache design), experiment log with
   manifest reference, supported-models table (status: planned). Update README
   project structure section.
6. (done 2026-07-12) Session checkpoint committed and pushed to origin/main at
   user request, including README updates (kickoff instruction, real project
   structure, honest quick start).

## Open decisions for the user

- Student architecture target (params, hidden size, depth, latency budget) — needed
  before Stage 1; Stage 0 collection does not depend on it.
- Whether to approve downloading a small public corpus for a larger Stage 0 v1
  warm-up set (handcrafted v0 is statistically thin: ~10-25k tokens vs 2560-dim
  covariance).
