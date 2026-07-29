# 2026-07-29 — Engine adapter layer + a bf16 batch-invariance finding (CPU, $0)

- **Agent:** Claude (dev box, CPU-only)
- **Commit:** see `code_state` in `artifacts/bench/cpu_smoke/report.json`
- **Cost:** $0 — no GPU, no pod, nothing billed
- **Objective:** build and validate the engine-benchmark harness on CPU before
  buying GPU time for it (P8), per the maintainer's 2026-07-29 direction to
  weigh engine efficiency against its impact on the codebase (P1)
- **Status:** harness verified end to end on CPU. The GPU benchmark itself is
  **pre-registered but not run** —
  [`logs/proposals/2026-07-29_engine_benchmark.md`](../../proposals/rollout/2026-07-29_engine_benchmark.md)

## 1. What was built

* `src/aadistill/engines.py` — a token-in/token-out `Engine` interface with
  three adapters (`hf` in-stack, `vllm`, `sglang`). Adapters do only
  "prompt ids in → new ids out"; **all** post-processing (stop-cutting, cap
  flags, stop-token normalization) is shared, so a cross-engine token comparison
  measures the engines rather than three copies of the trimming code.
* `scripts/bench_engines.py` — throughput at real job shape, batch invariance,
  agreement vs the in-stack reference, setup cost, and a mechanical
  `decision.json`.
* `scripts/generate_teacher_answers.py` — refactored to drive an `Engine` rather
  than `model.generate`, plus `--engine-from` (consumes `decision.json`) and
  `--max-hours` (wall-clock backstop for unattended paid runs). Dead
  `_cut_at_stop` removed; its logic now lives once, in `engines._finalize`.
* `scripts/pod/bench_and_generate.sh` — unattended pod driver chaining
  benchmark → decision → pilot corpus → hashes → upload.
* Tests: **191 passing** (was 119 in the last STATE snapshot; +33 here).

## 2. The finding: bf16 batched generation is not batch-invariant

Found while smoke-testing the harness, not while looking for it.

**Setup.** Stage 1 student checkpoint
(`artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`), CPU, greedy decoding, every
prompt truncated to **exactly 128 tokens so the batch requires no padding at
all**, 6 prompts, cap 16 new tokens. Each prompt generated alone, then all six
as one batch, then compared token by token.

| dtype | prompts identical alone vs batched | first divergence |
|---|---:|---|
| **bfloat16** | **1 / 6** | tokens 6, 6, 6, 7, 12 |
| float32 | **6 / 6** | — |

Same prompts, same code path, same batch size. **Only the dtype changed.**

An earlier padded run (5 prompts, mean 370 prompt tokens, cap 12) diverged too —
2/3 identical, one prompt flipping at token 0.

**Two things follow.**

1. **It is not padding.** The equal-length control exists precisely because
   batch-1 vs batch-N otherwise conflates "left-padding changed the numerics"
   with "batching changed the reduction order". With padding removed entirely,
   divergence not only survives, it is the common case. So padding-free batching
   is **not** a mitigation.
2. **It is precision.** fp32 is invariant at the identical configuration, which
   is the signature of non-associative float addition under a different
   reduction split — the documented mechanism the survey cited, now observed
   here rather than quoted.

**Why the existing tests missed it.** `tests/test_generate_toy.py` checks this
property on a toy model in **fp32**, and its own docstring flags fp32 as the
friendly case that "does not prove the property for a 4B model in bf16". That
caveat is now a measurement — on a 0.6B student, at least.

*Claim strength.* **Measured:** the table above, on CPU, on the 0.6B student.
**Inferred:** that the mechanism is reduction order — dtype is the only variable,
but no kernel-level attribution was done. **Not measured:** GPU behavior, the 4B
teacher, any serving engine, and whether the divergent tokens change answer
*content* rather than just token identity. The GPU run measures all four.

## 3. Consequences

* **The "corpus is the artifact" stance is now justified rather than assumed.**
  `generate_teacher_answers.py` already refused to promise bitwise
  reproducibility for batched sampling and pinned the experiment on the corpus
  hash instead. That was the right call for a reason nobody had measured: even
  *greedy* decoding is not reproducible across batch compositions in bf16.
* **SGLang's deterministic mode is now the most interesting arm**, not a
  curiosity. It is the only candidate offering batch-invariant kernels, and the
  property it sells is one this project has now confirmed it lacks. Its cited
  ~34% throughput tax has something concrete to buy.
* **Open question for the GPU run, worth more than the ranking:** does the
  in-stack path lose batch invariance on the 4B teacher too? If so, every
  behavior scorecard produced by batched eval carries an unquantified
  batch-composition term — which would sit next to the 0.1290 seed noise floor
  as a second source of the behavior-metric instability. This is a hypothesis,
  not a claim: eval batching was not varied in any completed run.

## 4. A bug this caught before it reached a corpus

The first `_strip_prefix` defended against sglang#10896 (`output_ids` carrying a
prompt-tail overlap) by stripping the longest prompt-suffix that was also an
output-prefix. `test_hf_engine_respects_the_cap` failed against it: prompt
`[5, 6, 7]` with a genuine completion `[7, 7, 7, 7]` lost a real token, because a
completion may legitimately begin with the token the prompt ended on.

A heuristic that silently shortens targets is worse than the bug it defends
against — it corrupts data instead of crashing. The function now strips only the
unambiguous whole-prompt echo, and the SGLang case is resolved **exactly**, from
that engine's reported `completion_tokens`. Both behaviors are pinned by tests.

## 5. Reproduce

```bash
uv run pytest tests/ -q                                    # 191 passed
uv run python scripts/bench_engines.py \
  --model artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
  --engines hf --n-prompts 5 --max-new-tokens 24 \
  --hf-batch-sizes 2,5 --invariance-n 3 --invariance-cap 12 \
  --out artifacts/bench/cpu_smoke
```

The equal-length padding control is not committed — it is a six-line driver over
`engines.batch_invariance` with prompts truncated to a common length, described
fully in §2.

## 6. Verdict and next action

**Verdict:** harness ready; one unplanned finding worth more than the smoke test
it came from. No decision changed yet — the engine choice is still unmeasured.

**Next:** the GPU session in
[`logs/proposals/2026-07-29_engine_benchmark.md`](../../proposals/rollout/2026-07-29_engine_benchmark.md),
which is pre-registered and awaiting maintainer approval on spend. It should add
one arm to its readout that this log motivates: **in-stack batch invariance on
the 4B teacher in bf16**, which the harness already measures for free.
