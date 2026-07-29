# Stage 0 — Qwen3-4B-Thinking-2507 initialization warm-up collection (v0)

- **Date:** 2026-07-12
- **Agent:** Claude Code (Fable 5)
- **Git commit at run:** b8f49ce (working tree dirty: uncommitted scripts/config/data/tests; full uncommitted state hash recorded in the run manifest)
- **Stage:** Stage 0 — initialization warm-up data collection
- **Objective:** Collect teacher activation sufficient statistics needed for Stage 1 teacher-aware student initialization of a compressed dense student.
- **Hypothesis:** Streaming residual-stream second moments (X^T X) and FFN per-neuron activation magnitudes over a small, diverse warm-up set are sufficient to drive grouped activation-PCA hidden-width projection and activation-importance FFN neuron selection, without caching raw activations.

## Setup

- **Teacher:** Qwen/Qwen3-4B-Thinking-2507, revision `768f209d9ea81521153ed38c47d515654e938aea`
  - hidden_size 2560, num_hidden_layers 36, intermediate_size 9728, vocab_size 151936
  - tokenizer sha256 `7781771acc3798ee…`
- **Student target:** not yet chosen (open decision; Stage 0 does not depend on it).
- **Hardware:** CPU-only dev box, 16 threads (AMX-BF16 / AVX-512 BF16), 30 GB RAM, no GPU.
- **dtype:** bfloat16. **Seed:** 20260712. **max_seq_len:** 1024.
- **Budget:** cache_budget_gb 2.5; single pass over the warm-up set.
- **Config:** `configs/stage0_qwen3_4b_thinking.json` (sha256 `1bdbafcb5603dcdc…`).
- **Data:** `data/warmup/warmup_v0.jsonl`, 47 samples, sha256 `2dcb4dee92e9c6b1…`.
  - Categories: general_text 5, instruction 5, reasoning 5, code 5, math 4, rag_evidence 4,
    refusal_uncertainty 4, short_realtime 4, tool_calling 4, multihop_qa 3, long_context 2,
    quant_calibration 2.

## Command

```bash
uv run python scripts/collect_stage0.py --config configs/stage0_qwen3_4b_thinking.json
```

(Dry run first with `--limit 2`.)

## Result

- 47 sequences processed, 4068 tokens total, 24 s wall clock.
- Determinism check: two forwards of the same sequence were **bitwise identical** (max abs logit diff 0.0).
- Cache: `artifacts/stage0/qwen3_4b_thinking_v0/activation_stats.safetensors`, 1.95 GB (1,947,442,680 B), under the 2.5 GB budget. Dominated by residual_sqsum (37 points × 2560² × f64).
- Projection dry run (mid layer, point 18): min eigenvalue +0.0014 (PSD), top eigenvalue 1.59e6, 191 eigenvalues above 1e-6·top. Covariance is well-formed and consumable by Stage 1.
- Manifest: `artifacts/stage0/qwen3_4b_thinking_v0/manifest.json` — records config, config hash, code state, hardware, teacher identity, tokenizer hash, dataset hash + per-sample tokens, determinism check, cache hash/size, projection dry run.

## Verdict

**Pass.** All Stage 0 validation-gate items are satisfied: dataset manifest, tokenizer identity+hash, teacher checkpoint ID+revision, activation-cache manifest, cache size+sampling policy, deterministic teacher validation (bitwise), projection dry run reading the collected signals, and full reproducibility from logged metadata.

## Known limitations

- Warm-up set is statistically thin for a 2560-dim covariance: 4068 tokens vs 2560 dimensions. The mid-layer covariance has ~191 well-supported directions, adequate for a projection dry run but likely too few for a final Stage 1 recipe. A larger v1 warm-up set (requires user approval to download a public corpus per AGENTS.md P12/Stage 0 policy) is the natural next step.
- Assistant turns are human-authored, not teacher-generated (CPU is too slow to generate from a 4B thinking model here). This is fine for collecting teacher *activations over given text*, but the text distribution is authored, not teacher-sampled.
- rank_proxy is a coarse eigenvalue-count proxy, not a validated effective-rank metric.

## Next action

- Stage 1: implement grouped activation-PCA hidden projection and activation-importance FFN neuron selection consuming this cache; requires the student architecture target decision first.
- Optionally expand the warm-up set (v1) with an approved public corpus before locking a Stage 1 recipe.

## Resume instructions

Fully reproducible by re-running the command above at commit b8f49ce + the committed scripts/config/data. The cache is gitignored; regenerate it rather than expecting it in git.
