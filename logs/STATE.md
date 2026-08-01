# Current project state

**Updated:** 2026-08-01 08:45 UTC · branch `main` · **TWO L40S PODS ARE RUNNING
AND BILLING** — Experiment 1 is under way (§12). Spend before this session
**$34.52**; this session is capped at **$59.40** by pod deadlines.

**Active work:** the recovery-data scaling study. The teacher corpus and the
nested token ladder are **built and gate-passed**; the training matrix is
specified and costed but **not started** — it needs the maintainer's go-ahead
against the raised $60 training budget.

**Experiment order (maintainer, 2026-08-01):** (1) does behavioural recovery
scale with teacher-generated supervised tokens — *token count is the only
variable, so the mixture is uniform*; (2) data mixing; (3) difficulty curriculum.
The capability-gap weighting is deferred to (2), not withdrawn.

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
| **recovery corpus v2** (2026-08-01) | `sessions.jsonl` sha256 `2b4edc2e…`, `candidates.jsonl` sha256 `f7f5035e…` |
| **token ladder v2** | `blocks.npz` + `audit.jsonl` + `ladder.json`, 3,720 blocks |
| teacher corpus v1 (752 prompts, 540 accepted) | relay `stage3_teacher_corpus_20260730/`, targets sha256 `18028f0c…` |
| best recovery checkpoint (reference only) | `s2v1_from_init@2700`, holdout 3.8285 |
| relay | `AlphaAvatar/aadistill-artifacts` (private) |

> **Durability risk — act before the next paid step.** Corpus v2 and its ladder
> (≈950 MB) exist **only** in the dev-box session scratchpad
> `/tmp/claude-1000/…/2e9e81e1-…/scratchpad/bulk/`. They are **not** on the HF
> relay and not on durable disk. Both file hashes were re-verified against the
> manifest on 2026-08-01 and match. Losing `/tmp` costs $25.56 and 16.5 h of
> teacher generation. Upload to the relay under `stage3_recovery_corpus_v2/`
> before the training matrix runs.

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
| corpus generation throughput | 66.08M tokens in 16.5 h on one L40S (~1,110 tok/s sustained) |
| training throughput | ~21 min per 137-step arm including gate evals |

## 5. Known deviations to carry

* Corpus **v1** (2026-07-30) was sampled at temperature 1.0 / top_p 1.0 / top_k
  off, against the model card's preset. Corpus **v2** uses the official preset
  `0.6 / 0.95 / 20 / min_p 0`, so the two corpora are not interchangeable.
* Corpus v1 is **effectively n=1** — 92.7% byte-identical candidate pairs
  because a serving engine seeds per request. Fixed in v2 by per-candidate
  seeds (`seed + batch_index + candidate_index × 1000003`); the v2 gate
  confirms distinct seeds and non-identical candidates for every type.
* The 8,192-token **session limit** censors the long tail of the hardest types:
  `openmath` loses 1,541/3,600 candidates and `code` 702/4,800 to
  `length_limited`, so accepted sessions of those types skew shorter/easier.
  Consequence of the fixed session limit; recorded, not worked around.
* Corpus v2's `code_state` block records **no git commit** — the pod bundle was
  unpacked outside a git checkout, so `git rev-parse` failed and the manifest
  stored `code_state_error` instead of a commit. The corpus is pinned by data
  hashes, teacher revision, chat-template hash and the full command, but its
  code state is pinned only by the bundle that was shipped. A P4 gap; fix the
  bundle to carry the commit before the next paid generation.
* Corpus v2 **computes correctness but does not enforce it** (acceptance is
  hygiene only, by design): `rag_evidence` 0.978, `gsm8k` 0.890, `multihop_qa`
  0.861, **`openmath` 0.380**; `code` and `tool_calling` have no mechanical key
  and score `unverifiable_slice`. Roughly a third of `openmath` targets teach a
  wrong final answer.
* `verify.hygiene_reason`'s `too_long` rule (`MAX_ANSWER_WORDS = 600`) is
  deliberately not applied — a generic word-count gate is forbidden by P3/P10.
  Structural hygiene only. Recorded in the manifest.

## 6. Corpus v2 — built 2026-08-01, gate PASSED, $25.56

`n=4` at the official preset, 8,192-token end-to-end session limit, turn
expansion for multi-turn sources. Prompt counts per type were set from the
measured capability gaps. 11,574 examples → **11,174 accepted (96.5%)**,
**66.08M generated tokens**, 16.5 h on one L40S.

| type | prompt share | examples | accepted | accept | tok/cand | supervised | sup/session |
|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 22% | 1,700 | 1,698 | 0.999 | 1,190 | 1,998,183 | 1,177 |
| rag_evidence | 20% | 4,100 | 4,100 | 1.000 | 503 | 2,087,594 | 509 |
| openmath | 17% | 900 | 579 | 0.643 | 5,196 | 1,977,473 | 3,415 |
| code | 16% | 1,200 | 1,123 | 0.936 | 4,609 | 4,773,086 | 4,250 |
| tool_calling | 15% | 2,600 | 2,600 | 1.000 | 419 | 1,073,688 | 413 |
| multihop_qa | 10% | 1,074 | 1,074 | 1.000 | 1,061 | 1,134,028 | 1,056 |

Those shares are **supervised-token shares**, not session counts — supervised
tokens per session differ **10×** across types (413 for `tool_calling`, 4,250
for `code`). They shaped **generation**; they are **not** what Experiment 1
trains on. The training mixture is chosen when the ladder is cut (§7), and
Experiment 1 uses the uniform cut. Rationale for the weighting, and why it is
deferred: the [mixture](decisions.md) and [experiment-order](decisions.md)
decisions (2026-08-01).

Source pools drawn from, after leakage filtering: `rag_evidence` 9,635,
`gsm8k` 7,134, `openmath` 4,342, `code` 1,751, `multihop_qa` 1,074 (fully
consumed), `tool_calling` 9,353 eligible expanded examples.

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
   packing boundary — 5,068 unique schemas, 4,394 of them singletons. Packing is
   therefore per system-prompt group (1,861 groups in the built corpus), and the
   declared mixture is restored by ordering **blocks** rather than sessions.
   Cost: tool blocks are largely padding, which inflates training compute (§8).
2. *Turn-expanded siblings may never share a block* — `#t1` is supervised on
   `a1ᵗ` while `#t3` carries `a1ᵒ` in context, so co-packing duplicates and leaks
   supervision inside one causal block. Colliding sessions are deferred to a
   later block, never dropped; prefix nesting is preserved.

**Leakage/dedup, recomputed rather than trusted:** a source conversation is
dropped whole if its content hash or first-user-message hash appears in any
reserved val/calib/holdout/behaviour-eval split. This removed 2,519 tool
conversations and 15 gsm8k / 2 openmath rows.

**Sizing lesson.** Prompt counts came from deliberately conservative
supervised-token estimates, and those were most wrong on the most expensive
types: `code` returned 4,609 tok/candidate against a 1,300 estimate (3.5×),
`tool_calling` 419 against 900. The corpus overshot to ~2× the 5.50M target.
Under the uniform cut Experiment 1 uses, the overshoot is concentrated in
`code`, which contributes 16.7% from a 3.48M pool while `multihop_qa`
contributes the same share from 1.01M — so the binding types are the *cheap*
ones and much of the `code` spend is unusable at this rung.

## 7. The token ladder — two cuts, one corpus

The corpus is packed once and cut into six **nested** rungs; the type mixture is
a parameter of the cut, not of the data, so re-cutting is free CPU work. Both
cuts exist:

| cut | corpus supervised | blocks | efficiency | ceiling | used by |
|---|---:|---:|---:|---:|---|
| **uniform 16.67% × 6** | 10,753,933 | 3,715 | 0.4699 | **6.08M** | **Experiment 1 (scaling)** |
| capability-gap weighted | 10,805,451 | 3,720 | 0.4709 | 10.81M | Experiment 2 (mixing) |

**Experiment 1's rungs** (uniform, re-cut 2026-08-01, `artifacts/stage3/ladder_uniform_probe`):

| rung (supervised) | actual | blocks | sessions | real tokens | padding | terminal truncations |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25M | 252,985 | 216 | 479 | 449,307 | 1,320,165 | 33 |
| 0.46M | 460,088 | 380 | 848 | 797,951 | 2,315,009 | 62 |
| 0.86M | 864,750 | 682 | 1,502 | 1,472,149 | 4,114,795 | 109 |
| 1.60M | 1,600,353 | 1,174 | 2,649 | 2,661,299 | 6,956,109 | 190 |
| 2.96M | 2,960,507 | 1,944 | 4,524 | 4,730,748 | 11,194,500 | 352 |
| **5.50M** | 5,501,372 | 2,941 | 7,350 | 8,256,511 | 15,836,161 | 635 |

All six rungs reachable; each realizes uniform within **0.3 pp** at the smallest
rung and **0.03 pp** at the top; nesting exact and monotonic.

**Two consequences of choosing uniform, both measured:**

* **+6.2% training compute** — 7,337 blocks/epoch against the weighted cut's
  6,907, because uniform raises the share of the badly-packing `tool_calling`
  type from 15% to 16.7%. The 24-run matrix goes ~$49 → **~$52**.
* **Saturation headroom nearly disappears** — the corpus supports at most
  **6,076,356** uniform supervised tokens, bound by `multihop_qa`'s 1,012,726
  post-packing tokens. Rungs meaningfully above 5.50M need more
  `multihop_qa`/`tool_calling` generation, or a non-uniform mixture
  (Experiment 2). Per-type post-packing pools: `code` 3,482,416 ·
  `rag_evidence` 1,980,108 · `gsm8k` 1,774,385 · `openmath` 1,430,610 ·
  `tool_calling` 1,073,688 · `multihop_qa` 1,012,726.

## 8. Packing efficiency 0.34 at the top rung — cost accepted, budget raised

`tool_calling` renders a unique schema into the system block, so with the system
prompt as a hard packing boundary its sessions cannot share blocks.

| at the uniform 5.50M rung | blocks | efficiency | sessions/block | supervised/block |
|---|---:|---:|---:|---:|
| tool blocks | **2,125 (72.3%)** | **0.096** | 1.14 | 431 |
| non-tool blocks | 816 | 0.984 | 6.03 | 5,619 |

**`tool_calling` supplies 16.7% of the supervision and consumes 72.3% of the
blocks.** The rung needs 2,941 blocks where a dense pack would need ~880 —
**3.3× the training compute**, most of a ~$52 training bill spent on positions
masked out of loss, KD and accounting. (The weighted cut is the same story at
15% / 72%: 2,074 tool blocks of 2,863, efficiency 0.092.)

**Resolved 2026-08-01:** the maintainer kept the packing rule and **raised the
training budget to $60** rather than allow several system blocks per packed
sample ([decision](decisions.md)). Every rung above 5.50M is billed at the same
3.35×, which is the trigger to revisit.

## 9. Initialization as a second scaling axis (maintainer, 2026-07-31)

The recovery relationship must also be measured against **Stage 1 initialization
quality**, since a different init may need a different amount of data. Both
checkpoints exist, same geometry (1024/28L/3072/16Q/8KV, tied):

| init | sha256 | holdout NLL |
|---|---|---:|
| PCA/sandwich `checkpoint` | `86fbba78e8a2a324…` | 11.748 |
| `random_baseline` | `0e2e2b28cfe5dc5b…` | 12.129 |

This makes the training matrix **6 rungs × 2 seeds × 2 inits = 24 runs**.
Projected: 6,907 blocks/epoch × 3 epochs × 2 seeds × 2 inits ÷ 2 blocks/step
= 41,442 steps × 4.3 s ≈ **49.5 h ≈ $49 training alone**, plus gate and
uncapped evals for 24 checkpoints — against the raised **$60** cap.

## 10. Implementation state (CPU-verified)

**279 tests pass on CPU** (`uv run pytest tests/ -q`, 10.5 s, no downloads).

| piece | file | state |
|---|---|---|
| session rendering + system-grouped packing | `src/aadistill/data/sessions.py` | used in the built corpus |
| shared assistant-mask helper | `src/aadistill/data/dataset.py` | `final_assistant_loss_mask` for turn expansion |
| `min_p` + per-prompt completion budgets | `src/aadistill/rollout/engines.py` | threaded through all 5 adapters |
| corpus builder | `scripts/rollout/build_recovery_corpus.py` | ran the 2026-08-01 bulk build |
| one-pass pack + nested ladder cut | `scripts/data/build_token_ladder.py` | produced the 6-rung ladder |
| §6/§9 gate validator | `scripts/data/validate_corpus_gate.py` | PASS on the full corpus |
| end-to-end CPU dress rehearsal | `tests/data/test_recovery_corpus_pipeline.py` | builder→ladder→gate with a stub engine |

Chunked CE/KD was assessed and is **not** needed: `block_len` stays 8192, which
the canonical recipe already runs.

**Finding that shaped the design:** the official chat template renders
`<think>…</think>` only for the assistant turn after the *last* user message, so
applying it to a multi-session message list **silently deletes every earlier
trace**. Verified directly. Sessions are therefore rendered independently and
concatenated at token level (asserted exact), with the system block emitted once.

## 12. Experiment 1 — LAUNCHED 2026-08-01 08:42 UTC

**Corpus v2 and both ladder cuts are on the relay**, prefix
`stage3_recovery_corpus_v2/` — 9/9 files hash-verified against the local copies
(LFS oid for the large ones, download-and-hash for the small). The §2 durability
risk is closed.

**Two L40S pods, split by initialization** so a lost pod costs one init axis
rather than half of every curve:

| pod | id | role | arms | steps |
|---|---|---|---:|---:|
| `aadistill-e1-pca` | `1ligfkwnaous4u` | PCA/sandwich init | 12 | 22,012 |
| `aadistill-e1-rand` | `vjavemn7m2tw5a` | random-baseline init | 12 | 22,012 |

Both carry a hard `--terminate-after 2026-08-02T14:31:18Z` (30.0 h), so the
session cannot exceed **2 × 30.0 × $0.99 = $59.40** even if every software guard
fails. The orchestrators additionally refuse to *start* an arm they cannot
finish before that deadline, so a budget-cut session ends with completed
uploads rather than a pod killed mid-transfer.

Expected: ~26.3 h training + ~0.5 h setup + ~1.6 h post-run per pod ≈ 28.4 h,
against 30.0 h. **Slack is ~1.5 h**, and the pca pod already spent ~0.4 h on two
setup failures. If a pod runs out of deadline it will skip its last arm(s),
which by arm ordering is the tail of **seed b** — seed a's full rung series
completes on both inits first.

Drive/monitor from the dev box (both orchestrators are `nohup`'d and survive any
session ending):

```bash
tail -f artifacts/stage3/e1_scaling_pca_orchestrator.log
cat artifacts/stage3/e1_scaling_{pca,rand}_orchestrator.status
```

**Deliberately deferred: the P18 uncapped behavioural readout.** It is not run
inline because its cost is unbounded on this model line —
`eval_behavior.py --unrestricted` has **no degeneration stop**, so a checkpoint
in a repetition loop generates until the 262,144-token context is exhausted, and
one prompt can outlast a whole training arm. Every checkpoint is uploaded to
`e1_scaling_20260801/`, so the readouts (`natural_termination_rate`,
`degeneration_rate`, length p50) run afterwards from the relay on vLLM with the
tested semantic degeneration stop, costed separately. **Until that runs,
Experiment 1 has trained checkpoints and holdout NLL, but no scaling curve.**

Per-arm the pods run: holdout NLL (bf16), a greedy 80-token generation smoke
test, sha256 of every retained file, upload to the relay, and independent
dev-box verification; the pod is deleted only after verification passes.

## 11. Next actions

1. **Persist corpus v2 to the relay** under `stage3_recovery_corpus_v2/`, with
   both ladder cuts, and write its manifest entry (§2 risk). Cheap, CPU-only, no
   approval needed beyond the upload itself.
2. **Maintainer decision required:** start the 24-run Experiment-1 matrix
   (~$52 + ~$5 evals against the $60 cap), or cut it first — dropping the init
   axis or one seed halves it to ~$26, dropping the top rung gives ~$31. Nothing
   paid runs until this is answered.
3. Fix the pod bundle so `code_state` carries a git commit (§5), before the next
   paid generation run.
4. After Experiment 1: fit the convergence curve of natural-termination rate,
   degeneration rate, generated-length p50 and holdout NLL against supervised
   tokens — the deliverable of the study ([`PROPOSAL.md`](PROPOSAL.md)). Only
   then Experiment 2 (data mixing), then Experiment 3 (difficulty curriculum).
