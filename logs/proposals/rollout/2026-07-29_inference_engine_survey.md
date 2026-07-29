# 2026-07-29 — Inference engine survey (research only; no engine chosen, nothing benchmarked)

**Status:** literature/landscape survey, requested before benchmarking. **No
engine is selected here and no measurement was taken.** Everything below is
either a cited third-party number or a property of this project; the two are
labelled. Third-party throughput figures are on *their* hardware and *their*
models and must not be quoted as ours (P7).

## 1. Why this survey exists

Teacher generation is the next spend, and its cost is dominated by decode
throughput. A second requirement carries over from the 2026-07-28 decision:
generation must be **token-in/token-out** and its numerics must not drift from
the trainer's, because Stage 4/5 trains on data the model produced — if the
rollout engine and the trainer disagree, "on-policy" updates are quietly
off-policy.

## 2. The landscape, by layer

| layer | engines | notes |
|---|---|---|
| Production serving | **vLLM**, **SGLang**, **LMDeploy**, **Aphrodite** | continuous batching, paged KV |
| Datacenter scale | **TensorRT-LLM** (+ Triton) | fastest absolute; ~28 min compile per model |
| Local / edge | **llama.cpp**, **Apple MLX**, **ExLlamaV3**, **MLC-LLM** | MLC-LLM is the only real browser/mobile option |
| Developer UX | **Ollama**, **LM Studio**, **KoboldCpp** | wrappers, mostly over llama.cpp |
| Minimal / readable | **nano-vLLM** (DeepSeek, ~1.2k lines, pure Python + Triton) | offline-focused; no dynamic batching or streaming |
| In-stack (current) | **HF transformers `generate`** | what `src/aadistill/generate.py` uses today |
| **Deprecated** | ~~TGI~~ | entered maintenance March 2026; officially redirects to vLLM / SGLang / llama.cpp / MLX. **Do not adopt.** |

## 3. Third-party throughput (their hardware — not ours)

Reported on H100 for a Llama-3.1-8B-class model:

| engine | tok/s (cited) |
|---|---:|
| SGLang | 16,215 |
| LMDeploy | 16,132 |
| vLLM | 12,553 |
| TensorRT-LLM | fastest absolute, with a compile step |

Two caveats that matter more than the ranking:

* SGLang and LMDeploy differ by **<0.6%** — inside anyone's noise.
* The ordering **reverses by workload**: on single-turn unique prompts one
  comparison had vLLM ahead of SGLang (60 vs 52.7 tok/s). Our teacher-generation
  job is exactly that shape — many unique prompts, no shared prefix — so the
  headline numbers, which are driven by prefix-cache reuse, are close to
  irrelevant to us. Sources also report the bottleneck has moved from kernels to
  engine orchestration overhead, which is workload-shaped.

**This is the reason to benchmark rather than to pick from a table.**

## 4. The decisive axis for this project: determinism

Batch-invariance is not a niche concern — it is a documented failure mode with a
named fix, and it is the exact property `assert_batch_invariant`
(`src/aadistill/generate.py`) was built to check.

**The cause:** batch size changes how GPU kernels split reductions; float
addition is non-associative, so identical prompts take different numerical paths
to logits and can diverge under greedy decoding. vLLM's own docs note the same
requests "might be batched differently", giving "slightly different
logit/logprob values at each step".

**SGLang ships an explicit deterministic mode** (built on Thinking Machines'
batch-invariant operators; vLLM has adopted the same approach):

* guarantees **batch-invariant** output — same prompt, same result, regardless
  of how it was batched;
* cost **~34% average slowdown** (25–45%), with CUDA graphs recovering ~2.8×;
* backends: FlashInfer (fixed split size), FlashAttention 3 (num-splits=1),
  Triton (AMD);
* built **for reproducible RL** — integrated with slime it reports identical
  rollout responses *and* loss values across repeated runs;
* **limitations:** radix cache support incomplete on some backends; the authors
  recommend production use "only for RL scenarios"; **dense models only —
  Qwen3 and LLaMA named**; TP>2 needs extra work.

**Qwen3 being explicitly named matters**: our teacher is
`Qwen3-4B-Thinking-2507`, a dense Qwen3, which is inside the supported set.

## 5. What this means for us, stated as project facts

* **Determinism is purchasable.** It is not a reason to avoid a serving engine
  any more — it is a ~34% throughput tax with a documented implementation. That
  materially weakens the 2026-07-28 argument for staying in-stack, which
  assumed the choice was between speed and reproducibility.
* **But our workload is the one where the published ordering does not hold.**
  Unique prompts, no shared prefix, long thinking traces, offline batch. The
  engines' headline advantages come largely from prefix caching and continuous
  batching under concurrent load — neither of which describes a corpus build.
* **The in-stack baseline has never been measured properly.** The 55 s/prompt
  figure that started this line of inquiry was `batch_size` 1 in
  `eval_behavior.py`, not an engine limit. `src/aadistill/generate.py` (batched,
  built 2026-07-28) has **not** been benchmarked at all.
* **nano-vLLM is the interesting middle.** Pure Python + Triton, no C++/CUDA
  extension, offline-inference focus, reported near-parity with vLLM on offline
  workloads — the profile that would give throughput without the dependency
  weight. Unverified by us, and its determinism properties are unknown.

## 6. What tomorrow's benchmark should measure

Fixed budget, our shape, our teacher — not a leaderboard reproduction:

1. **Throughput at our actual job shape**: `Qwen3-4B-Thinking-2507` @ bf16,
   native thinking mode, cap 4096, unique prompts drawn from `data/stage2_v1`,
   measured in **tokens/s and $ per 1k accepted samples** (the number that
   actually sizes the corpus build).
2. **Batch invariance**, using our own `assert_batch_invariant` on each
   candidate — including the in-stack path, which has never been checked at 4B
   in bf16.
3. **Agreement with the training stack**: do the engine's greedy tokens match
   HF `generate` on the same prompts? This is the property Stage 4/5 needs, and
   no vendor claims it across stacks — it has to be measured.
4. **Setup cost**: install time, dependency weight, whether it can live
   pod-side only (out of the dev-box lockfile), and compile time if any.
5. **Determinism tax**, where a deterministic mode exists: throughput with and
   without.

**Candidate shortlist to actually run** (not all of the above — the rest are
excluded by layer or by status): in-stack HF `generate` (the incumbent, and the
only one whose numerics are training-identical by construction), **vLLM**
(breadth, `--model-impl transformers` keeps one model implementation),
**SGLang** (deterministic mode, Qwen3 named), and **nano-vLLM** if the first
three leave a dependency-weight question open. TGI is excluded as deprecated;
TensorRT-LLM as premature at 4B with a per-model compile step; llama.cpp /
Ollama / MLX / ExLlamaV3 / MLC-LLM as wrong numerics or wrong deployment target.

## 7. Sources

- [LLM Inference Engines: vLLM vs LMDeploy vs SGLang](https://aimultiple.com/inference-engines)
- [Best LLM Inference Engines (2026) — Yotta Labs](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)
- [The Complete Guide to Local LLM Inference Tools, July 2026](https://dev.to/sreeraj-sreenivasan/the-complete-guide-to-local-llm-inference-tools-in-july-2026-llamacpp-ollama-vllm-sglang-and-4mh1)
- [Towards Deterministic Inference in SGLang and Reproducible RL Training](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [SGLang deterministic inference docs](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/deterministic_inference.md)
- [nano-vLLM (DeepSeek)](https://huggingface.co/blog/zamal/introduction-to-nano-vllm)
- [Native-speed vLLM transformers modeling backend](https://huggingface.co/blog/native-speed-vllm-transformers-backend)
- [No More Train-Inference Mismatch (vLLM + TorchTitan)](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html)
