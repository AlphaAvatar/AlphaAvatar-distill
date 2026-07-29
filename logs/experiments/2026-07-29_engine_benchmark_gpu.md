# 2026-07-29 — Engine benchmark on the 4B teacher (L40S) + corpus pilot

- **Agent:** Claude, driving pod `g8ajahpwirhrfx` (1× L40S, $0.99/h)
- **Pre-registration:** [`proposals/2026-07-29_engine_benchmark.md`](../proposals/2026-07-29_engine_benchmark.md)
- **Objective:** choose the decode backend by weighing throughput against
  integration cost (maintainer, 2026-07-29), then build a teacher-corpus pilot
- **Status of this file:** benchmark complete; pilot running at time of writing

## 1. Headline: the engine choice was decided by integration cost, not speed

Neither serving engine could be benchmarked at all, because neither can coexist
with this project's pinned stack **in the same process**:

| engine | installs? | imports? | why it fails |
|---|---|---|---|
| **vLLM 0.26.0** | yes | **no** | compiled extension `vllm._C_stable_libtorch` needs `libcudart.so.13` (CUDA 13). The image and this project's torch are **cu128**; only `libcudart.so.12` exists. |
| **SGLang 0.5.9** | yes | yes | but only by **downgrading torch 2.11.0 → 2.9.1 and transformers 5.13.1 → 4.57.1**. This repo targets the transformers **v5** API, so the training stack is broken by the install. `uv sync` restored it. |

Rule **R3** (a second stack must beat the incumbent by ≥1.5× to be worth owning)
never needed a throughput number: an engine that cannot share a process with the
trainer is not a drop-in, and one that silently downgrades transformers by a
major version is the opposite of the "impact on the overall code" the maintainer
asked to minimise.

**This is a bound on what was tested, not a verdict on the engines.** The
untested path is running a serving engine from an **isolated venv as a
subprocess or HTTP server** — a normal deployment pattern that sidesteps the
dependency conflict entirely, at the cost of a process boundary and a protocol.
Given §2, that path is now worth buying.

## 2. In-stack throughput: batching does not help, and the corpus is unaffordable

`Qwen3-4B-Thinking-2507@768f209d`, bf16, greedy, cap 4096, 10 slice-balanced
prompts (mean 374 prompt tokens), L40S, torch 2.11.0+cu128.

| batch | tok/s | peak mem | mean new tokens | hit cap | $/1k prompts | wall |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 37.5 | 9.12 GB | 1568 | 1/10 | $11.51 | 418 s |
| 4 | **43.9** | 10.11 GB | 1760 | 1/10 | $11.04 | 401 s |
| 8 | 39.3 | 12.84 GB | 1514 | 1/10 | $10.62 | 386 s |

**Batching buys essentially nothing** — 37.5 → 43.9 → 39.3 tok/s across a 4×
range of batch size, which is inside the run-to-run spread. This **refutes the
premise in `src/aadistill/generate.py`**, which was built on the reasoning that
"batching is most of the available speedup at this project's model sizes". It is
not. The 55 s/prompt figure that started this line of inquiry was indeed a
`batch_size` 1 artifact, but fixing it recovers ~14 s/prompt, not the order of
magnitude a serving engine would.

For scale: ~44 tok/s at ~1,600 tokens per candidate puts the pre-registered
1,000-prompt × n=4 pilot at roughly **40 GPU-hours (~$40)** — against a 2.5 h
budget. The previously *guessed* $25–145 bulk-build cost now has a measured
basis, and the answer is that the bulk build is **not affordable in-stack**.

Peak memory is modest (9–13 GB of 44 GB), so memory is not the binding
constraint at this batch size — throughput is. That combination (low memory
utilisation, flat scaling) is the signature of a decode loop bound by
per-step overhead rather than by hardware, which is exactly what paged-KV /
CUDA-graph engines exist to fix.

## 3. Batch invariance fails on the 4B teacher too

The CPU finding on the 0.6B student ([log](2026-07-29_engine_adapter_and_bf16_invariance.md))
replicates on the real teacher in bf16:

- **7 of 8 prompts identical** between batch-1 and batch-8 greedy decoding at
  cap 64; one diverged at **token 50**.

And the throughput table is independent corroboration at long context: `mean new
tokens` is **1568 / 1760 / 1514** for the *same ten prompts under greedy
decoding* at batch 2 / 4 / 8. Greedy decoding is a deterministic function of the
logits, so identical prompts producing different average completion lengths
means the logits themselves differ by batch composition. The effect is small at
64 tokens and compounds over a 4,096-token trace.

**Consequence:** a corpus built at batch size *b* is not the corpus the same
model would produce at batch size *b′*. "The corpus is the artifact" (P5) is now
a measured requirement rather than a stylistic preference, and any future
importance-weighted on-policy objective must treat the recorded corpus — not a
re-derived policy — as ground truth.

## 4. What the session cost, and two process failures worth recording (P11)

1. **A 45-minute reference arm was lost to one OOM.** `run_arm` wrapped an
   entire engine arm in a single `try/except`, so an `OutOfMemoryError` on the
   *first* sweep point discarded the arm — and rule R4 then correctly refused to
   build a corpus without a reference. One recoverable error cost the whole
   session's output. Each sweep point is now isolated and an OOM is recorded as
   a data point; batch invariance is guarded separately, since losing it must
   not invalidate throughput or block the corpus.
2. **`uv pip install` silently no-ops without `--python`.** It does not read
   `UV_PROJECT_ENVIRONMENT` (that governs `uv sync`/`uv run`), so both engine
   installs "failed" in the same second — the tell that no download was ever
   attempted. Had that not been caught, the session would have reported both
   engines as unavailable for entirely the wrong reason.
3. Minor but expensive-looking: `pkill -f generate_teacher_answers` matches the
   *remote shell's own command line*, killing the launcher before it can start
   the process. Launch and kill must be separate SSH invocations.

## 5. Verdict against the pre-registered hypotheses

- **H1 — "the in-stack path is materially slower than a serving engine at this
  job shape": UNRESOLVED, and not for the expected reason.** No serving engine
  could be run in-process. What *was* measured is that the in-stack path is
  slow in absolute terms and does not improve with batching, which makes H1
  worth re-testing via the isolated-venv route.
- **H2 — "a serving engine's greedy tokens will not match the training stack":
  UNTESTED.** No cross-engine agreement number exists.
- **H3 — "batch invariance does not hold for the real 4B in bf16": CONFIRMED**
  for the in-stack path (§3). This was pre-registered as possibly true of the
  incumbent, and it is.

## 6. Next actions

1. **Buy the isolated-venv engine test.** It is the one path that could make the
   corpus build affordable, and §2 shows the in-stack path cannot. Scope: a
   separate venv with vLLM's own torch, driven as a subprocess or local HTTP
   server, measured on the same 10-prompt job shape. The adapter interface
   already isolates this — only a new `Engine` subclass changes.
2. **Do not size the bulk corpus from the in-stack number** ($44 per 1k prompts
   at n=4). Re-price after (1).
3. Feed §3 into the eval story: behavior scorecards are produced by batched
   generation, so they carry a batch-composition term that has never been
   quantified. That sits beside the 0.1290 seed noise floor as a *second*
   unmeasured source of behavior-metric variance.

## 7. Pilot corpus (in progress)

Rescoped from the pre-registered 200 prompts/slice to **10 prompts/slice (50
total), n=4, batch 4, cap 4096, `--max-hours 2.2`**. Reason: at the measured
throughput the original scope would have covered only the *first* slice before
the budget stopped it, since the generator walks slices in order. A small
complete pilot across all five slices answers the actual questions — per-slice
accept@1/accept@n and the divergence profile that sets adaptive `n` — where a
truncated one would not.

Sampling follows the 2026-07-29 decision: **all candidates sampled, none greedy**,
temperature 1.0 / top_p 1.0 / top_k off. `accept_at_1` therefore means "one
sample was accepted" and is not comparable to pre-2026-07-29 figures.

*Results to be filled in on completion.*
