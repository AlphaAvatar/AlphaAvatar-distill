# 2026-07-26 — INT8 weight fake-quant eval path + first INT8 eval of s1@660

- **Agent:** Claude Code (Fable 5), continuing the first dense-model
  compression experiment (teacher `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`,
  student 0.6B-class).
- **Git commit:** implemented on top of `e273cba` (dirty during the run —
  this session's quant module; committed immediately after, see the commit
  containing this log).
- **Stage:** Stage 3 support milestone (P9 precision policy: BF16 training,
  INT8 deployment target). Queued since the s1 gate as "INT8 eval path
  before the next GPU run".
- **Objective:** Land a deployment-matching INT8 evaluation path and measure
  the INT8-weight degradation of the current reference checkpoint.
- **Hypothesis:** Per-channel symmetric INT8 weight quantization is
  near-lossless for a BF16-recovered 0.6B student (weight-only INT8 is
  usually benign at this scale; the risk P9 guards against is INT4/activation
  quant, deferred to Stage 6).
- **Hardware:** CPU-only dev box (16 threads, AMX/AVX-512 BF16, 30 GB RAM).
  CPU-suitable per P8.2 — each holdout eval takes ~14 s.
- **Budget:** trivial (3 CPU evals × ~14 s + unit tests).

## What was built

- `src/aadistill/quant.py` — `int8_fake_quantize_(model, scope)`:
  per-output-channel symmetric INT8 weight fake-quant
  (`scale_i = amax|W_i|/127`, round-clamp-dequant computed in fp32, cast
  back to model dtype), applied in-place. Scopes: `decoder` (every
  `nn.Linear` under `model.layers.`) and `all` (decoder + lm_head; the
  tied embedding/head matrix is quantized once through the shared tensor,
  matching runtimes that store one quantized copy). Tied/shared weights
  deduped by tensor identity; fails loudly on empty scope match.
- `scripts/eval_ppl.py` — `--fake-quant int8` + `--fake-quant-scope
  {all,decoder}`; the quant summary (n modules, n params, mean/max relative
  Frobenius error, max scale) is embedded in the per-model result and the
  report JSON.
- `tests/test_quant_toy.py` — 8 tests (grid + error bound, zero-row safety,
  dtype preservation, determinism, scope module sets on a tiny tied
  Qwen3, tie preservation after quant, output perturbation bound, loud
  failure on unknown scope). Full suite: **51/51 pass**.

## Commands

```
uv run pytest tests/ -q                                    # 51 passed
uv run python scripts/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage3/s1_ffn_norm_v0/checkpoints/step_000660/model \
  [--fake-quant int8 [--fake-quant-scope decoder]] --out <report.json>
```

## Results (holdout_v1, 21,080 tokens, bf16 eval graph, CPU)

| config | quantized params | mean rel Fro err | NLL (nats) | ppl | Δ vs bf16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 reference | — | — | 4.2111 | 67.43 | — |
| int8 scope=decoder | 440.4M (196 linears) | 0.0099 | 4.2122 | 67.51 | +0.03% |
| int8 scope=all (+tied head) | 596.0M (197 linears) | 0.0100 | 4.2198 | 68.02 | +0.21% |

Cross-device check: CPU bf16 4.2111 vs the GPU-logged s1@660 value 4.2107 —
+0.0004 nats, within the P5-logged cross-process/device variance scale.

Reports: `artifacts/stage3/s1_ffn_norm_v0/eval_holdout_v1_{bf16,int8_all,int8_decoder}_cpu.json`
(gitignored artifacts; hashes + code state embedded in each report).

## Verdict

**Hypothesis confirmed: INT8 weight quantization is near-lossless for
s1@660** (+0.21% NLL full-scope, +0.03% decoder-only). The deployment
numerics gap (P9) is currently negligible; no QAT/fake-quant-in-training
pressure at this stage. Most of the small full-scope delta comes from the
tied embedding/head matrix — worth re-measuring per checkpoint since the
head is 26% of student params.

Caveats: weight-only INT8 (activation quant deferred to Stage 6 per the
2026-07-13 precision policy — the stratified `calib` split remains reserved
for it); fake-quant runs matmuls in bf16 rather than int32 accumulation, so
kernel-level differences are not captured. Both are logged limitations, not
blockers for the recovery-stage gates.

## Next action

- Run `--fake-quant int8` (scope=all) as a standard side-eval at every
  future recovery gate (cheap: seconds on CPU per checkpoint).
- Stage 2 mixture v1 scale-up proposal (this session, separate doc) — the
  measured Stage 3 bottleneck.
