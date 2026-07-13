# Stage 0 v1 — activation statistics on warm-up v1 (Qwen3-4B-Thinking-2507)

- **Date:** 2026-07-13 (UTC; run 17:54–18:56 local)
- **Agent:** Claude Code session (user-approved corpus download)
- **Git commit:** `dc643d0` + uncommitted Stage 1 work (code-state hash in manifest: `7de7be95…`)
- **Objective:** Replace the statistically thin v0 statistics (4,068 tokens) with a ~1M-token collection so Stage 1 PCA projections are well-conditioned.
- **Hypothesis:** ~950K tokens ≫ 2560 dims yields full-rank, stable second moments at every residual point.
- **Teacher:** Qwen/Qwen3-4B-Thinking-2507 @ `768f209d`, bf16, CPU.
- **Stage:** Stage 0 (initialization warm-up data collection), second iteration.
- **Hardware:** CPU-only dev box, 16 threads (AMX/AVX-512 BF16), 30 GB RAM.
- **Budget:** cache ≤ 2.5 GB, single CPU run (actual 62.3 min).

## Data

`data/warmup/warmup_v1.jsonl` (gitignored), built by `scripts/build_warmup_v1.py`;
manifest `data/warmup/warmup_v1.manifest.json` (committed) pins source revisions:

| Source | License | Samples | Tokens |
| --- | --- | --- | --- |
| fineweb-edu sample-10BT @ `87f09149` | ODC-By 1.0 | 848 | 511,810 |
| databricks-dolly-15k @ `bdd27f4d` | CC-BY-SA 3.0 | 1,048 | 192,634 |
| gsm8k main @ `740312ad` | MIT | 921 | 179,177 |
| mbpp full @ `4bb6404f` | CC-BY-4.0 | 338 | 62,613 |
| warmup_v0 handcrafted (all 47) | project-authored | 47 | 3,625 |

Dataset sha256 `ed5e6b9b…`, 3,202 samples, 949,859 tokens (max_seq_len 1024).

## Command

```
uv run python scripts/collect_stage0.py --config configs/stage0_qwen3_4b_thinking_v1.json
```

Config sha256 `3f55e69f…`, seed 20260713.

## Result

- Cache `artifacts/stage0/qwen3_4b_thinking_v1/activation_stats.safetensors`,
  1.95 GB (< 2.5 GB budget), sha256 `aaeb2e4c…`; full manifest alongside.
- Teacher determinism check: bitwise identical (max logit diff 0.0).
- Projection dry run (mid layer 18): min eigenvalue **+0.0165** (v0: +0.0014),
  554 eigenvalues above 1e-6 of top (v0: 191) — covariance now positive
  definite with far broader support.
- Wall clock 3,735 s ≈ 254 tokens/s.

## Verdict

**Pass.** Stage 0 gate items re-validated on v1; these statistics are the
input for the Stage 1 initialization run (`artifacts/stage1/qwen3_0p6b_init_v0`).

## Next action

Stage 1 projection + sandwich init of the 0.6B-class student (same session);
see `logs/experiments/2026-07-14_stage1_qwen3_0p6b_init_v0.md`.

## Resume instructions

Rebuild data with `scripts/build_warmup_v1.py` (deterministic, revision-pinned),
then re-run the command above; the cache regenerates bitwise-reproducibly for
the same environment.
