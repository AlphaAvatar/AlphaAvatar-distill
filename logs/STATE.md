# Current project state

**Updated:** 2026-07-30 18:17 UTC · branch `main`, dirty (recovery-corpus
implementation uncommitted) · **nothing running, nothing billing, no pods or
volumes exist.** Session spend to date **$7.93**.

**Active work:** the data-scaling experiment (maintainer spec, 2026-07-31).
Implementation complete and CPU-verified; the paid §6 validation gate is next.
Budgets: teacher generation **$50**, student training + evaluation **$50**,
separate hard caps.

Canonical handoff. Companions:

* [`logs/EXPERIMENTS.md`](EXPERIMENTS.md) — everything run, results, cost
* [`logs/PROPOSAL.md`](PROPOSAL.md) — the one active plan
* [`decisions.md`](decisions.md) · [`supported_models.md`](supported_models.md) · [`artifact_manifests.md`](artifact_manifests.md)

---

## 1. Where the project is

Teacher **`Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`** → student **0.6B-class**
(1024 hidden, 28L, FFN 3072, 16Q/8KV, tied emb). BF16 training, INT8 deployment.

Stages 0 → 1 → 2 complete. **Stage 3 recovery is open.**

**The blocking fact:** under unrestricted generation *every* checkpoint,
including the best one (`s2v1_from_init@2700`, holdout 3.8285), degenerates into
repetition. No model in this line yet produces a complete answer in the
teacher's thinking protocol. **Zero context-limit hits** — the old 512-token
evaluation cap was hiding repetition loops, not long reasoning.

Neither 2026-07-30 four-arm run supports a route-level claim about
teacher-native supervision: one had an invalid start point, and both were
convergence- and measurement-limited (`EXPERIMENTS.md` §5).

## 2. Pinned assets

| asset | identity |
|---|---|
| **fork point** — Stage 1 structural init | `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`, `model.safetensors` sha256 `86fbba78…` |
| teacher corpus (752 prompts, 540 accepted) | relay `stage3_teacher_corpus_20260730/`, targets sha256 `18028f0c…` |
| best recovery checkpoint (reference only) | `s2v1_from_init@2700`, holdout 3.8285 |
| relay | `AlphaAvatar/aadistill-artifacts` (private) |

## 3. Protocol requirements (binding)

* **System message is mandatory** in teacher generation, student training,
  primary evaluation and inference. Fixed project requirement, **not** an
  experimental variable. Default: `You are a helpful Assistant.`
* Thinking mode is never suppressed; `<think>` is opened by the template
  unconditionally on `add_generation_prompt`.
* **No artificial generation cap in formal measurement** (P18). Allowance is
  `context − prompt`; context resolves to **262,144**.
* Stop ids come from the model's `generation_config` (teacher `[151645, 151643]`),
  not the tokenizer alone.
* No no-think / empty-think / final-only / shortened substitute targets (P17).

## 4. Measurement constraints

| quantity | value |
|---|---|
| behaviour-metric seed noise floor | **0.1290** → ≥2 seeds per arm |
| cold-start holdout-NLL seed spread | **2.21 nats** → ≥4 seeds from the Stage 1 init |
| teacher natural termination | **80.1%**; lengths p50 **727**, p99 3854 |
| `block_len` 8192 memory | 44,983/46,068 MiB with gradient checkpointing, ~4.3 s/step |
| generation throughput | 5.8 s/prompt at n=2; 566 tok/s aggregate |
| training throughput | ~21 min per 137-step arm including gate evals |

## 5. Known deviations to carry

* Teacher corpus sampled at temperature 1.0 / top_p 1.0 / top_k off, against the
  model card's 0.6 / 0.95 / 20 / min_p 0. Deliberate (2026-07-29) but a deviation.
* Existing corpus is **effectively n=1** — 92.7% byte-identical candidate pairs
  because a serving engine seeds per request. Per-candidate seeds fixed in code;
  the fix is untested against real acceptance.
* The 4096 generation cap censored 19.9% of teacher rollouts, **69.7% of
  openmath**, biasing corpus composition against long-reasoning instances.

## 6. Data-scaling experiment — implementation state

Corpus is bounded by an **8,192-token end-to-end session limit** (prompt +
template + completion). `n=4`, official preset `0.6 / 0.95 / top_k 20 / min_p 0`.
Acceptance is hygiene, not correctness. Six nested rungs 0.25M → 5.50M supervised
tokens × 2 seeds = 12 runs.

| piece | file | state |
|---|---|---|
| session rendering + system-grouped packing | `src/aadistill/data/sessions.py` | new, 25 unit tests |
| shared assistant-mask helper | `src/aadistill/data/dataset.py` | refactored out of `encode_sample` |
| `min_p` + per-prompt completion budgets | `src/aadistill/rollout/engines.py` | threaded through all 5 adapters |
| corpus builder | `scripts/rollout/build_recovery_corpus.py` | new |
| one-pass pack + nested ladder cut | `scripts/data/build_token_ladder.py` | new |
| §6/§9 gate validator | `scripts/data/validate_corpus_gate.py` | new |
| end-to-end CPU dress rehearsal | `tests/data/test_recovery_corpus_pipeline.py` | builder→ladder→gate with a stub engine |

**274 tests pass on CPU.** Chunked CE/KD was assessed and is **not** needed:
`block_len` stays 8192, which the canonical recipe already runs.

**Finding that shaped the design:** the official chat template renders
`<think>…</think>` only for the assistant turn after the *last* user message, so
applying it to a multi-session message list **silently deletes every earlier
trace**. Verified directly. Sessions are therefore rendered independently and
concatenated at token level (asserted exact), with the system block emitted once.

**Deliberate deviation:** `verify.hygiene_reason`'s `too_long` rule
(`MAX_ANSWER_WORDS = 600`) is not applied — a generic word-count gate is
forbidden by P3/P10 and would reimpose in word space the censoring the 4,096-token
cap caused in token space. Structural hygiene only. Recorded in the manifest.

## 7. Re-projected corpus sizing (censored-MLE fit, CPU)

Fitted per type from the measured 2026-07-30 corpus (temperature 1.0, 4,096 cap):

| type | fitted median | P(fits budget) | supervised/accepted | accept@n=4 | pool |
|---|---:|---:|---:|---:|---:|
| rag_evidence | 459 | 1.000 | 537 | 1.000 | 9,635 |
| multihop_qa | 871 | 1.000 | 1,053 | 1.000 | **1,074** |
| gsm8k | 1,236 | 0.995 | 1,562 | 1.000 | 7,149 |
| openmath | 8,103 | 0.500 | 3,456 | 0.938 | 4,344 |

* **861 prompts/type (3,444 total)** reach 5.50M supervised tokens; ~30.9M
  generated tokens; **$2.83–$15.00** at 3,000–566 tok/s on an L40S.
* **Every pool prompt is usable** — no prompt leaves a non-positive budget.
* **`hotpot_qa` is the binding constraint at 1,074**, headroom 24.7%. Equal
  four-way balance fails if the official preset yields generations **>20%
  shorter** than the temperature-1.0 data the fit came from. The gate measures
  this directly.
* **openmath is half-censored by the 8,192 limit** (median 8,103 ≈ the limit), so
  accepted openmath sessions skew shorter/easier. Consequence of the fixed
  session limit; recorded, not worked around.

## 8. §6 validation gate — RUN 2026-07-30, **PASSED**, $1.03

L40S, driver **580.159.03**, **vLLM 0.26.0**, torch 2.11.0+cu130, transformers
5.14.1. Preset `0.6 / 0.95 / 20 / min_p 0`, `n=4`, session limit 8192, stop ids
`[151643, 151645]`, chat template `3802169b…`. Every §9 check PASSED for all four
types (`artifacts/` not committed; results in the session scratchpad).

| type | prompt accept | cand accept | post-pack sup/session | sd | reject |
|---|---:|---:|---:|---:|---|
| rag_evidence | 1.000 | 1.000 | 368 | 132 | — |
| multihop_qa | 1.000 | 1.000 | 740 | — | — |
| gsm8k | 1.000 | 1.000 | 1,037 | 795 | — |
| openmath | 0.700 | 0.600 | 2,267 | 1,532 | length_limited 16/40 |

HotpotQA follow-up at **n=70**: accept 1.000, pre-packing supervised mean **963**
(sd 850, cv 0.88), 95% CI [764, 1162] — down from [526, 1470] at n=10.

Other measurements: **packing discards 21.2%** of pre-packing supervised tokens
as terminal-truncation suffix; packing efficiency 0.958; throughput **339 tok/s**
at 10 concurrent, **682 tok/s** at 70 concurrent (a floor — bulk runs thousands
of concurrent sequences).

## 9. BLOCKER — equal four-way balance cannot reach 5.50M

Post-packing supervised per prompt-of-each-type is **3,732**, so 5.50M needs
**1,474 prompts/type**. The `hotpot_qa` pool is **1,074** — short by 27%.

| | max reachable at equal balance |
|---|---:|
| point estimate | **4.01M** |
| conservative (hotpot at 95% lower bound) | **3.84M** |

The n=70 follow-up removed the sampling uncertainty; the shortfall is structural,
not statistical. Three drivers, all measured: openmath's 0.700 prompt-accept
under the 8,192 limit, the 21.2% packing discard, and the `hotpot_qa` pool
ceiling.

**Resolved 2026-07-31:** the maintainer lifted the equal-balance requirement in
favour of a difficulty-aware mixture, and added multi-turn data via turn
expansion. See §10.

## 10. Difficulty-aware corpus v2 (running)

**Mixture** (supervised-token share of 5.50M, fixed across all rungs and seeds):

| type | share | why |
|---|---:|---|
| gsm8k | 22% | largest capability gap — math EM 0.000 vs teacher 0.714 |
| openmath | 17% | hardest reasoning; half-censored by the 8,192 limit |
| code | 16% | README names code; magicoder_oss + mbpp |
| rag_evidence | 20% | grounding / RAG objective; cheap per session |
| tool_calling | 15% | largest behaviour-axis gap (+0.667), agentic objective |
| multihop_qa | 10% | capped by its 1,074-conversation pool |

**Excluded and why:** `long_context` is `format: "text"` — raw documents with no
question, so a teacher cannot answer it without synthesizing prompts (a new
data-construction experiment). `refusal_uncertainty`, `instruction`,
`short_realtime` stay out of scope per the 2026-07-30 alignment-tax decision;
multi-turn coverage comes from `tool_calling`, which is both multi-turn and
on-target.

**Turn expansion.** A multi-turn source becomes one example per eligible
assistant turn; only the newly generated teacher turn is supervised, and every
preceding *original* assistant turn is context, masked from loss and from
supervised-token accounting (`final_assistant_loss_mask`). This unlocked
`tool_calling`: 7,123 conversations → 10,855 examples, 9,353 eligible.

**Two packing constraints this forced:**

1. *Tool schemas render into the system block*, and the system prompt is a hard
   packing boundary — 5,068 unique schemas over 7,127 conversations, 4,394 of
   them singletons. Packing is therefore per system-prompt group, and the
   declared mixture is restored by ordering **blocks** rather than sessions.
   Cost: tool blocks are largely padding, which inflates training compute.
2. *Turn-expanded siblings may never share a block* — `#t1` is supervised on
   `a1ᵗ` while `#t3` carries `a1ᵒ` in context, so co-packing duplicates and leaks
   supervision inside one causal block. Colliding sessions are deferred to a
   later block, never dropped; prefix nesting is preserved.

**Leakage/dedup, recomputed rather than trusted:** a source conversation is
dropped whole if its content hash or first-user-message hash appears in any
reserved val/calib/holdout/behaviour-eval split. This removed 2,519 tool
conversations and 15 gsm8k / 2 openmath rows.

## 11. Initialization as a second scaling axis (maintainer, 2026-07-31)

The recovery relationship must also be measured against **Stage 1 initialization
quality**, since a different init may need a different amount of data. Both
checkpoints exist, same geometry (1024/28L/3072/16Q/8KV, tied):

| init | sha256 | holdout NLL |
|---|---|---:|
| PCA/sandwich `checkpoint` | `86fbba78e8a2a324…` | 11.748 |
| `random_baseline` | `0e2e2b28cfe5dc5b…` | 12.129 |

This doubles the training matrix to **6 rungs × 2 seeds × 2 inits = 24 runs**.

## 12. Corpus v2 BUILT 2026-08-01 — gate PASSED, $25.56

11,574 examples → **11,174 accepted (96.5%)**, 66.08M generated tokens, 16.5 h on
one L40S. Hashes verified against the manifest before the pod was released.

| type | examples | accepted | ex accept | tok/cand | supervised | sup/session |
|---|---:|---:|---:|---:|---:|---:|
| rag_evidence | 4,100 | 4,100 | 1.000 | 503 | 2,087,594 | 509 |
| multihop_qa | 1,074 | 1,074 | 1.000 | 1,061 | 1,134,028 | 1,056 |
| gsm8k | 1,700 | 1,698 | 0.999 | 1,190 | 1,998,183 | 1,177 |
| openmath | 900 | 579 | 0.643 | 5,196 | 1,977,473 | 3,415 |
| code | 1,200 | 1,123 | 0.936 | 4,609 | 4,773,086 | 4,250 |
| tool_calling | 2,600 | 2,600 | 1.000 | 419 | 1,073,688 | 413 |

**Ladder: 3,720 blocks, 11,174 sessions, 10,805,451 post-packing supervised
tokens** — ~2× the 5.50M rung, so saturation rungs above it are available from
the same pack. Every rung lands within **0.2 pp** of the declared mixture and
nesting is exact.

## 13. BLOCKER — packing efficiency 0.34 at the 5.50M rung ⇒ training overruns $50

`tool_calling` renders a unique schema into the system block, so with the system
prompt as a hard packing boundary its sessions cannot share blocks.

| at the 5.50M rung | blocks | efficiency | sessions/block | supervised/block |
|---|---:|---:|---:|---:|
| tool blocks | **2,074** | **0.092** | 1.11 | 398 |
| non-tool blocks | 789 | 0.985 | 6.62 | 5,925 |

**`tool_calling` supplies 15% of the supervision and consumes 72% of the
blocks.** The rung needs 2,863 blocks where a dense pack would need ~855 —
**3.35× the training compute**.

Projected: 6,907 blocks/epoch × 3 epochs × 2 seeds × 2 inits ÷ 2 blocks/step
= 41,442 steps × 4.3 s = **49.5 h ≈ $49.0 training alone**, before evals.
With gate + uncapped evals for 24 checkpoints this is **~$60 against the $50
cap. Stopped per §11 — awaiting a decision.**
