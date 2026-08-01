# Experiment record — AlphaAvatar-distill

The single consolidated record of everything run. Replaces 25 per-run logs and 11
proposal files, which are preserved in git history at commit `866dac2`.

**Teacher** `Qwen/Qwen3-4B-Thinking-2507@768f209d` (2560 hidden, 36L, FFN 9728,
32Q/8KV) → **student** 0.6B-class (1024 hidden, 28L, FFN 3072, 16Q/8KV, tied
embeddings). BF16 training, INT8 deployment target.

**Total spend to date: $7.93.** Itemised in §6, including what was wasted.

---

## 1. Pipeline state

| stage | what exists | status |
|---|---|---|
| 0 — activation collection | 949,859 tokens, v1 | **done** |
| 1 — structural init | `stage1/qwen3_0p6b_init_v0/checkpoint`, holdout NLL 11.748 | **done, pinned start point** |
| 2 — offline mixture | `stage2_v1`, 22.13M train tokens | **done** |
| 3 — recovery | best checkpoint `s2v1_from_init@2700`, holdout **3.8285** | **open — no checkpoint generates usable output** |
| 4/5/6 | — | not started |

**The blocking fact:** under unrestricted generation, *every* checkpoint
including the best one degenerates into repetition. No model in this line yet
produces a complete answer in the teacher's thinking protocol.

---

## 2. Stages 0–2 (all CPU or cheap GPU, concluded)

* **Stage 0** — teacher activations, 949,859 tokens.
* **Stage 1** — PCA/sandwich structural init. Holdout NLL **11.748** vs random
  init 12.13 vs teacher 2.63. Gate passed. This checkpoint (`model.safetensors`
  sha256 `86fbba78…`) is the pinned fork point for all recovery work.
* **Stage 2** — offline mixture v0 (5.39M tokens) then v1 (22.13M).

## 3. Stage 3 recovery runs

| run | what it tested | result | verdict |
|---|---|---|---|
| `s1_ffn_norm` (660 steps) | FFN+norm recovery | holdout 4.21 | gate passed |
| `s2_ab` | freeze-set sizing | attention-unfrozen adopted; holdout flat, mixture v0 exhausted | informed v1 |
| `s2_blocks_v1` (2700) | mixture v1 | holdout **3.8003** | best NLL |
| start-point ablation | ladder vs single-stage | single-stage reaches 3.8285 with 33% fewer steps | **ladder retired** |
| packing / `block_len` control | best_fit@2048 vs concat@1024 | +2.1% regression on both seeds | **rejected** |
| CE/KD conflict | `kd_scope all_no_think` | p(`</think>`) **0.2995 → 0.9989** | conflict confirmed causal |

**Standing branch point:** `s2v1_from_init/step_002700`, holdout **3.8285**.

## 4. Measurements that constrain everything downstream

| finding | measurement |
|---|---|
| **Behavior-metric noise floor** | seed-only spread on `behavior_score_v0` = **0.1290**, wider than any inter-arm difference reported. ≥2 seeds required for any behaviour claim |
| **Cold-start NLL noise** | two seeds of one config differed by **2.21 nats** (6.62 vs 8.83) from the Stage 1 init. Holdout NLL is unresolvable at 2 seeds from cold start; ≥4 needed |
| **bf16 decoding is not batch-invariant** | student 1/6 identical batch-1 vs batch-6; **fp32 6/6** |
| **Both rollout engines ≈5.5× HF and scale** | vLLM 0.26.0 **247.5** tok/s, SGLang 0.5.12 **241.0**; wall 57.03 s vs 56.87 s. Policy mismatch negligible (off-policy rate **0.000**, KL ~1e-4) |
| **Teacher natural termination** | **80.1%** of 1,504 rollouts; lengths p25 466 / **p50 727** / p90 2233 / p99 3854 / max 4069 |
| **The 4096 generation cap censored 19.9%** | per slice: rag 0.5%, multihop 1.1%, gsm8k 8.5%, **openmath 69.7%** — which is what drove openmath's 0.261 accept rate |
| **Teacher targets are 4.2× longer than public** | `best_fit`@8192 is lossless; `concat`@1024 would split 79.9% |
| **`block_len` 8192 memory** | **44,983 / 46,068 MiB (97.6%)** with gradient checkpointing, ~4.3 s/step. `best_fit` pads every block so this peak is constant |
| **Supervised-token asymmetry** | teacher targets carry **18.9×** the supervised tokens of public ones on the same prompts (519,478 vs 27,526) |

## 5. The two four-arm runs — both diagnostics, neither a route decision

### 5.1 Post-s2v1 continuation (2026-07-30, $3.50 of the training pod)

Public-target control vs teacher-native treatment × 2 seeds, 137 steps, forked
from `s2v1_from_init@2700`. **Invalid as a target comparison:** that checkpoint
is 2,700 steps of public-target training, so the public arm started inside its
own target distribution. Relabelled a continuation diagnostic; its R2 "reject"
is void as evidence about teacher-native targets.

### 5.2 Corrected baseline from the Stage 1 init (2026-07-30, $2.30)

Same 2×2 forked from the pinned Stage 1 init, so neither arm had a path
advantage. Rule R2 fired again — but the run is **convergence-limited** (137
steps ≈ 5% of the reference budget, 487 prompts) and was **measurement-limited**
(99.3% of treatment generations censored at 512 tokens).

| | step-0 | control (public) | treatment (teacher) |
|---|---:|---:|---:|
| holdout NLL | 11.7565 | 7.7260 ±1.10 | **6.2255** ±0.48 |
| `format_ok` | 0.000 | 0.625 | 0.000 |
| `terminated` | 0.000 | 0.658 | 0.007 |

The teacher arm had the **larger** NLL improvement.

### 5.3 What the unrestricted pilot then showed ($0.79)

8 deterministic prompts × 6 checkpoints, full 262,144 context, concurrency 1, no
token cap:

| checkpoint | natural | degenerated | context-limited |
|---|---:|---:|---:|
| step-0 Stage 1 init | 0 | 8 | 0 |
| public arms (2 seeds) | 7 / 5 | 1 / 3 | 0 |
| teacher arms (2 seeds) | 0 / 0 | 8 / 8 | 0 |
| **`s2v1@2700` (best ckpt)** | **0** | **8** | **0** |

* **The best checkpoint degenerates too**, so degeneration is a property of the
  whole student line, not of teacher-native targets.
* **Zero context-limit hits.** The 512-token cap was hiding repetition loops,
  not long reasoning (e.g. a 17-token block repeated 15× from position 513).
* The public arm's apparent win was **protocol substitution**: its natural
  terminations are 5–18 token stubs after an **empty** `<think>`
  (`</think>\n\nArthur's Magazine<|im_end|>`), several incoherent.

**Conclusion carried forward: no route-level claim about teacher-native
supervision is supported by any run so far.**

## 6. Cost, including waste

| item | cost | assessment |
|---|---:|---|
| Stage 3 recovery runs (s1, A/B, v1, ablation, packing, CE/KD) | prior sessions | necessary |
| engine benchmark + isolated engine | prior sessions | necessary once; now concluded and deferred |
| teacher corpus generation (752 prompts) | **$1.37** | necessary; reusable |
| post-s2v1 continuation 4-arm run | **$3.50** | **wasted** — invalid start point, my error |
| corrected baseline 4-arm run | **$2.30** | diagnostic only; convergence- and measurement-limited |
| unrestricted pilot | **$0.79** | necessary; produced the finding that reframed everything |
| **this session total** | **$7.93** | of which **$3.50 was avoidable** |

**Wasted:** the 2026-07-30 continuation run. Forking both arms from a
public-trained checkpoint was an error that a five-minute check of the start
point would have caught before spending.

**Also avoidable in hindsight:** the 512-token evaluation cap, which censored
99.3% of one arm and made two runs uninterpretable. Now forbidden by AGENTS.md
P18.

## 7. Standing decisions

| decision | date |
|---|---|
| Stage 3 recovery is single-stage from the Stage 1 init | 07-27 |
| Behaviour comparisons need ≥2 seeds; NLL from cold start needs ≥4 | 07-28 / 07-31 |
| The teacher is never forced out of thinking mode | 07-28 |
| Exact token agreement is **not** an engine adoption gate; engines run in their own official images | 07-30 |
| `best_fit` @ `block_len` 8192 for teacher-native targets, gradient checkpointing required | 07-30 |
| **P17** — teacher-behaviour fidelity: no no-think / empty-think / final-only / shortened substitute targets | 07-30 |
| **P18** — no artificial generation cap in formal measurement | 07-30 |
| Corpus is **effectively n=1** (92.7% byte-identical candidates); per-candidate seeds fixed | 07-30 |
| Teacher template injects **no** default system message — a fact about the checkpoint | 07-31 |
| **Project protocol: an explicit system message is mandatory**, fixed requirement, not an experimental variable | 07-31 |

## 8. Reusable assets

| asset | where |
|---|---|
| Stage 1 init (pinned fork point) | `stage1/qwen3_0p6b_init_v0/checkpoint`, sha256 `86fbba78…` |
| teacher corpus, 752 prompts / 540 accepted | `stage3_teacher_corpus_20260730/`, targets sha256 `18028f0c…` |
| rollout snapshot, 1,504 rollouts / 2.46M tokens | same prefix, sha256 `0e5b20dd…` |
| all trained checkpoints | relay under `stage3/`, `tt2x2/`, `ttb/` |
| Stage 2 mixture v1 | `data/stage2_v1` (regenerable from its manifest) |

---

## 9. Data-scaling experiment — §6 validation gate (2026-07-30)

**Objective:** gate into bulk teacher generation for the recovery-data scaling
study. 10 prompts × 4 types × `n=4`, full production path, plus a HotpotQA-only
follow-up at n=70.

**Hardware/stack, pinned:** RunPod L40S 46,068 MiB, driver **580.159.03**;
vLLM **0.26.0** (offline `LLM.generate`, per-prompt `SamplingParams`), torch
2.11.0+cu130, transformers 5.14.1, Python 3.12, isolated venv.
Teacher `Qwen/Qwen3-4B-Thinking-2507@768f209d`, chat template sha `3802169b…`,
stop ids `[151643, 151645]` from `generation_config.json`.
Preset `temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0`, session limit 8,192
end-to-end, per-prompt completion budget `8192 − rendered_prompt − 8`.

**Result: gate PASSED** — every §9 check across all four types (generation,
packing, loader, prefix equivalence).

| type | prompt accept | cand accept | pre-pack sup | post-pack sup | reject |
|---|---:|---:|---:|---:|---|
| rag_evidence | 1.000 | 1.000 | 368 | 368 | — |
| multihop_qa | 1.000 | 1.000 | 998 | 740 | — |
| gsm8k | 1.000 | 1.000 | 1,037 | 1,037 | — |
| openmath | 0.700 | 0.600 | 3,382 | 2,267 | length_limited 16/40 |

HotpotQA n=70: accept 1.000, supervised mean **963** sd 850 (cv 0.88), 95% CI
[764, 1162]. Packing discard **21.2%**; efficiency 0.958. Throughput 339 tok/s
at 10 concurrent, **682 tok/s** at 70 concurrent.

**Verdict: blocked on corpus size, not on correctness.** Equal four-way balance
needs 1,474 prompts/type for 5.50M post-packing supervised tokens; `hotpot_qa`
has 1,074. Max reachable **4.01M** (conservative 3.84M). Stopped per §11 rather
than alter the balance, target or pool.

**Cost: $1.03**, all against the $50 generation cap. Of that, ~$0.65 was spent on
two infrastructure failures before the successful run.

**Infrastructure findings (reusable):**
* vLLM 0.26.0's wheel links `libcudart.so.13`; a 570.x driver host cannot run it
  and `--torch-backend=cu128` does not help, because it changes torch, not the
  vLLM extension. Create the pod with `--min-cuda-version 13.0`.
* FlashInfer JIT-builds the top-k sampling kernel during vLLM warmup, so `ninja`
  must be on `PATH` — `top_k=20` is in the official preset, so without it the
  engine dies *after* loading weights.
* Put the venv on the container disk: a torch install onto the network-mounted
  `/workspace` took >9 minutes.
* `tar` needs `--no-same-owner` on that mount, or it exits non-zero on chown.

---

## 10. Difficulty-aware recovery corpus v2 (2026-08-01)

**Objective:** build the maximal reusable teacher corpus for the data-scaling
study, under an 8,192-token end-to-end session limit, with a difficulty-aware
mixture rather than equal four-way balance.

**Stack, pinned:** L40S 46,068 MiB, driver 580.159.03; vLLM **0.26.0** offline
`LLM.generate` with per-example `SamplingParams`; torch 2.11.0+cu130,
transformers 5.14.1. Teacher `Qwen/Qwen3-4B-Thinking-2507@768f209d`, chat
template sha `3802169b…`, stop ids `[151643, 151645]` from
`generation_config.json`. Preset `0.6 / 0.95 / top_k 20 / min_p 0`, `n=4`,
per-example budget `8192 − rendered_prompt − 8`.

**Result:** 11,574 examples → **11,174 accepted (96.5%)**, 66.08M generated
tokens, 16.5 h, **$25.56**. Gate re-run on the full corpus: **PASSED**.

| type | examples | accepted | ex accept | tok/cand | supervised | sup/session |
|---|---:|---:|---:|---:|---:|---:|
| rag_evidence | 4,100 | 4,100 | 1.000 | 503 | 2,087,594 | 509 |
| multihop_qa | 1,074 | 1,074 | 1.000 | 1,061 | 1,134,028 | 1,056 |
| gsm8k | 1,700 | 1,698 | 0.999 | 1,190 | 1,998,183 | 1,177 |
| openmath | 900 | 579 | 0.643 | 5,196 | 1,977,473 | 3,415 |
| code | 1,200 | 1,123 | 0.936 | 4,609 | 4,773,086 | 4,250 |
| tool_calling | 2,600 | 2,600 | 1.000 | 419 | 1,073,688 | 413 |

**Ladder:** 3,720 blocks, 11,174 sessions, **10,805,451 post-packing supervised
tokens** — ~2× the 5.50M rung, leaving headroom for saturation rungs above it.
Every rung lands within **0.2 pp** of the declared mixture; nesting is exact and
monotonic.

**Sizing lesson.** Prompt counts were set from deliberately conservative
supervised-token estimates, and those were most wrong on the most expensive
types: `code` came in at 4,609 tok/candidate against a 1,300 estimate (3.5×),
while `tool_calling` came in at 419 against 900. Net effect was over-provisioning
of `code`/`openmath` and a corpus ~2× the target — useful as saturation headroom,
but ~$4 of it bought tokens the 5.50M rung cannot consume.

**Packing efficiency 0.34 at the 5.50M rung**, and this is the dominant cost of
the whole study:

| at the 5.50M rung | blocks | efficiency | sessions/block | supervised/block |
|---|---:|---:|---:|---:|
| tool blocks | 2,074 | 0.092 | 1.11 | 398 |
| non-tool blocks | 789 | 0.985 | 6.62 | 5,925 |

`tool_calling` supplies 15% of the supervision and consumes **72% of the
blocks**, because each conversation's tool schema renders into the system block
and the system prompt is a hard packing boundary. The rung needs 2,863 blocks
where a dense pack would need ~855 — **3.35× training compute**. Maintainer
accepted the cost rather than relax the packing rule (2026-08-01) and raised the
training budget to $60.

**Cost, including waste:** generation **$26.69** total ($1.13 gate + $25.56
corpus). Of the corpus run, **~$8.70 was idle pod time** — the job finished at
06:27 and the pod was deleted at 15:14, because polling was driven by user
prompts rather than a timer and `--terminate-after` was a 32 h backstop. Standing
instruction since: tear down a finished pod without being asked.
