**Updated:** 2026-08-04 09:10 UTC · branch `main` · **no pods running, nothing
billing.** Verified spend **$109.51** of the **$126.02** cap ($96.02 prior +
$12.97 Experiment 2 phase 1 + **$0.52** the 2026-08-04 diagnostic session).
**$16.51 of the $30 Experiment 2 allocation is unspent.**

**Active work:** Experiment 2 phase 1 complete; the 2026-08-04 diagnostic session
complete. Full records in [`EXPERIMENTS.md`](EXPERIMENTS.md) §13–§14.

**The three results that matter now:**

1. **Capacity and task difficulty are ruled out.** The released `Qwen3-0.6B`
   (near-geometry: **identical 595,984,384 parameters**, but `rope_theta` 1e6 vs
   5e6 and `max_position_embeddings` 40,960 vs 262,144) solves **~70% of GSM8K**
   and **~78% of RAG** on our frozen battery. A 0.6B model can do this task. The
   gap is the recipe.
2. **The overfitted control cannot reproduce its own training targets.** After
   ~41 passes over the same rung it reaches **0.7803 gold-prefix next-token
   top-1** and **0.0 correctness**, median prefix match **0 tokens**. Handing it
   more gold prefix makes protocol validity *fall* (0.647 → 0.158 at k=256), so
   this is not simply a failure to get started.
3. **Padding truncation is worth ~2.69×** on the realistic block mixture (7.96×
   on the most padded, 0.996× on dense — the control). The 4B teacher forward is
   41% of a full-width step.

**Decision taken (see [decisions.md](decisions.md)): rollout/on-policy training
is NOT enabled.** The pre-registered gate points to auditing template, EOS
supervision, loss masking and target construction first. Exact repetition loops
are consistent with exposure bias but do not prove it; EOS supervision, greedy
decoding, entropy collapse, initialization and objective imbalance stay live.

**Measurement caveat carried forward:** `protocol_valid` assumes a generation
begins inside an already-open `<think>`, so **`correct` is not comparable across
models** until the splitter is fixed. This affects 1.9% of our own generations
(vs 65% `not_terminated`), so **E1/E2 conclusions stand**. Raw generations are
saved; rescoring is free.

**`batch.truncate_padding` is adopted per config and the code default stays
`false`** — the paths are mathematically equivalent but not bitwise identical, and
flipping the default would silently change already-logged configs.

Phases 2/3 and L1/R1/R2 remain paused. No training experiment was started.

Canonical handoff. Companions:

* [`logs/EXPERIMENTS.md`](EXPERIMENTS.md) — everything run, results, cost
* [`logs/PROPOSAL.md`](PROPOSAL.md) — the Experiment 2 pre-registration
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

> **Resolved 2026-08-01:** corpus v2 and both ladder cuts are on the relay under
> `stage3_recovery_corpus_v2/`, 9/9 files hash-verified against the local copies.
>
> **Open storage constraint.** The relay hit its private-storage limit during the
> run. Deleting the superseded `tt2x2`/`ttb` weights (19.07 GB, approved) dropped
> the tree to 80.31 GB but reclaimed **nothing** — HF bills LFS storage including
> history. The maintainer approved squashing history to reclaim it (see
> [decisions](decisions.md), 2026-08-02); until that runs, **four arms are
> dev-box-only**: `e1_r2960k_sb_pca`, `e1_r5500k_sb_pca`, `e1_r2960k_sb_rand`,
> `e1_r5500k_sb_rand`, each hash-verified under
> `artifacts/stage3/rescued/`.

## 3. Protocol requirements (binding)

* **System message is mandatory** in teacher generation, student training,
  primary evaluation and inference. Fixed project requirement, **not** an
  experimental variable. Default: `You are a helpful Assistant.`
* Thinking mode is never suppressed; `<think>` is opened by the template
  unconditionally on `add_generation_prompt`.
* **No artificial generation cap in formal measurement** (P18). Allowance is
  `context − prompt`. Context resolves to the **trained** `block_len` = **8,192**,
  not the architectural 262,144 the geometry inherits — recorded per measurement
  as `context_resolution.context_source = "trained_block_len"`. Experiment 2
  phase 1 confirmed `context_limit_rate` **0.0000 at all 20 checkpoints**, so
  nothing has ever been engine-truncated under this rule; the only stop reasons
  observed are `eos` and the degeneration stop.
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

**473 tests pass on CPU** (`uv run pytest tests/ -q`, ~12 s, no downloads).

| piece | file | state |
|---|---|---|
| session rendering + system-grouped packing | `src/aadistill/data/sessions.py` | used in the built corpus |
| shared assistant-mask helper | `src/aadistill/data/dataset.py` | `final_assistant_loss_mask` for turn expansion |
| `min_p` + per-prompt completion budgets | `src/aadistill/rollout/engines.py` | threaded through all 5 adapters |
| corpus builder | `scripts/rollout/build_recovery_corpus.py` | ran the 2026-08-01 bulk build |
| one-pass pack + nested ladder cut | `scripts/data/build_token_ladder.py` | produced the 6-rung ladder |
| §6/§9 gate validator | `scripts/data/validate_corpus_gate.py` | PASS on the full corpus |
| end-to-end CPU dress rehearsal | `tests/data/test_recovery_corpus_pipeline.py` | builder→ladder→gate with a stub engine |
| candidate cleaning rules (`clean-v2`) | `src/aadistill/data/cleaning.py` | median-length survivor; built the D1 corpus; 30 rule tests |
| cleaned-corpus driver | `scripts/data/build_cleaned_corpus.py` | 11,174 sessions screened in 114 s |
| ladder session-order anchor | `scripts/data/build_token_ladder.py --session-order` | keeps a re-cut pack on the anchor's prompts |
| prompt-matched single-rung packer | `scripts/data/build_matched_rung.py` | 89.1% D0 overlap at exact compute; appends the control's validation blocks |
| selection-rule comparison | `scripts/data/audit_selection_rule.py` | median vs shortest on one corpus |
| checkpoint retention policy | `scripts/pod/retain_checkpoints.py` | trajectory-driven keep set; 13 tests |
| frozen capability battery | `scripts/data/build_capability_battery.py` | `capability-v2`, 846 prompts, 7 sets, 0 leakage collisions |
| deterministic capability scorers | `src/aadistill/evaluation/capability.py` | alias EM, symbolic math, evidence recall, paired answerability **and** paired safety; 112 tests |
| battery scoring driver | `scripts/evaluation/score_battery.py` | pair-accuracy headline; offline, re-runnable |
| checkpoint inventory + cleanup | `scripts/pod/checkpoint_inventory.py` | both stores, hash-matched duplicates, declared classification |
| D0↔D1 corpus audit | `scripts/data/audit_d1_corpus.py` | overlap, shares, budget, residual mismatch |
| strict final-answer rule | `src/aadistill/evaluation/strict_answer.py` | replaces last-number GSM8K scoring; 17 tests |
| offline GSM8K re-scoring | `scripts/evaluation/rescore_gsm8k.py` | re-scored all 25 E1 arms, $0 |

Chunked CE/KD was assessed and is **not** needed: `block_len` stays 8192, which
the canonical recipe already runs.

**Finding that shaped the design:** the official chat template renders
`<think>…</think>` only for the assistant turn after the *last* user message, so
applying it to a multi-session message list **silently deletes every earlier
trace**. Verified directly. Sessions are therefore rendered independently and
concatenated at token level (asserted exact), with the system block emitted once.

## 12. Experiment 1 — COMPLETE 2026-08-02, all 24 arms, $47.6

Two L40S pods split by initialization; both released after hash-verified
teardown (pca 08:33, rand 08:49). Training ran at **3.08–3.8 s/step** against a
4.3 s/step projection, so the matrix finished inside the $59.40 cap.

**Teacher-native held-out CE** (16 pack-tail blocks, disjoint from every rung,
identical across all arms):

| supervised tokens | PCA sa | PCA sb | rand sa | rand sb |
|---:|---:|---:|---:|---:|
| step 0 | 10.9199 | 10.9199 | 12.1615 | 12.1615 |
| 0.25M | 2.0938 | 2.1427 | 8.8291 | 8.8234 |
| 0.46M | 1.7477 | 1.7611 | 8.3346 | 8.3575 |
| 0.86M | 1.5101 | 1.5038 | 7.9403 | 7.9542 |
| 1.60M | 1.2952 | 1.3015 | 7.4068 | 7.4025 |
| 2.96M | 1.1468 | 1.1486 | 6.6812 | 6.6643 |
| **5.50M** | **1.0032** | **1.0052** | **5.9807** | **5.9789** |

Full table incl. holdout NLL: `artifacts/stage3/e1_results.json` and
[`EXPERIMENTS.md`](EXPERIMENTS.md) §11. Headlines: seed gaps ≤0.049 (usable
instrument); **neither init saturates at 5.50M**; initialization dominates data
over this range (1.0032 vs 5.9807 at the top rung).

**Holdout NLL rises on PCA arms while CE falls** (6.72 → 10.79) and drifts down
on random arms. NLL cannot say whether the loss is knowledge or reasoning — the
open question §13 exists to answer.

## 13. Evaluation — COMPLETE, all 25 checkpoints

Behaviour, GSM8K and holdout NLL measured for every arm under one protocol; the
sweep's 125 artifacts were hash-verified on the dev box and the pod released
automatically at 00:21. Full results and variance analysis:
[`EXPERIMENTS.md`](EXPERIMENTS.md) §11. Consolidated table:
`artifacts/stage3/e1_consolidated.json` (regenerate with
`scripts/evaluation/consolidate_e1.py`).

**Headline:** CE scales with data at 74–261x the between-seed noise and has not
saturated at 5.50M. Initialization dominates: PCA reaches CE 1.0042 / behaviour
0.378 at the top rung against random's 5.9798 / 0.110, and every random arm sits
at p50 = 768 generated tokens — the degeneration-stop signature. **No rung, seed
or initialization developed measurable reasoning:** GSM8K EM across all 25
checkpoints is min 0.000, max 0.050, mean 0.006.

**Instrument quality, measured:** val CE resolves the effect 74x over seed noise;
the behaviour composite only 3.3x, so its rung ordering is not claimable; holdout
NLL has a between-seed |Δ| of 0.66 and is the weakest of the four.

**Reviewable samples for human analysis:** `logs/e1_test_cases.md` (46 cases,
readable) and `logs/e1_test_cases.jsonl` (untruncated), stratified over stop
reason and GSM8K correctness across rungs, seeds and inits.

**Artifacts:** evaluation JSONs are on the relay under
`e1_scaling_20260801/_evaluation` (small files pass the LFS quota). Four
checkpoints remain relay-blocked and are held, hash-verified, on the dev box
under `artifacts/stage3/rescued/`.

## 14. Experiment 2 phase 1 — prepared, awaiting launch approval

Zero-GPU preparation complete and verified ([`PROPOSAL.md`](PROPOSAL.md),
[`EXPERIMENTS.md`](EXPERIMENTS.md) §12):

* **Rung: 0.86M**, chosen because held-out NLL bottoms at 0.46M and takes its
  largest jump 0.46M → 0.86M; 2.96M is post-deterioration plateau.
* **Exact D0 baseline parsed, not scaled**: 864,750 supervised · 682 blocks ·
  1,502 sessions · 5,586,944 packed tokens · 1,023 steps · η 5e-5 / warmup 51 ·
  seeds 20260726 / 20260801 · init `86fbba78…` · config sha `08264ef1…` /
  `9048173d…`, both recomputed and matching the run manifests. Per-arm
  `train_log.jsonl` and `run_manifest.json` recovered **from the relay**.
* **`clean-v2` median-length survivor selection.** On the 73 prompts where it
  disagrees with shortest-survivor it keeps **1.35× more reasoning trace**.
* **D1 built and matched**: 682 blocks / 1,023 steps / 5,586,944 packed tokens
  exact, 858,409 supervised (−0.733%), **89.1% prompt overlap**, ≤0.17 pp share
  drift, and **byte-identical validation blocks** verified through the real
  trainer path.
* **Retention resolved**: `scripts/pod/retain_checkpoints.py` — metrics at all 9
  eval points, weights only for final / best-val-CE / best-NLL / the
  deterioration bracket. ~73 GB for eight arms against 117 GB free on the dev box.

**What D0 cannot support:** Experiment 1 ran `keep_last: 1`, so best-checkpoint
and within-run-onset comparisons exist for new arms only. The fixed-step endpoint
is the fully matched comparison.

* **Capability battery frozen**: `artifacts/eval/battery_v2/`, manifest sha256
  `060bdd31…`, **846 prompts**. Safety refusal is now its **own XSTest set**,
  separate from the SQuAD-v2 pairs (renamed `answerability_paired`) which measure
  evidence abstention on benign prompts. Deterministic scorers only, **0 leakage
  collisions**, **112** evaluator-validation tests — all five required policies
  verified on the safety set (always-answer 0/50 pairs, always-refuse 0/50,
  correct selective refusal 50/50, malformed 0/50, degenerate 0/50).
* **Evaluation throughput audited and the execution path corrected.** Experiment
  1 evaluated at **255 output tok/s** on a 0.6B model — ~10x low. The submission
  pattern was already correct (all requests queued before stepping, so vLLM
  continuously batches), but the engine was re-initialized **per prompt set**
  (1.73 min each; 7 sets = 12.1 min/checkpoint) and every scheduler step copied
  every running request's full token list. Fixed: one engine for all seven sets,
  lazy token materialisation, `detokenize=False`. **No decoding or evaluation
  semantics changed**, proven by 20 equivalence tests. Structural saving $1.71;
  the generation speedup is measured by the D0 baseline before D1 battery money
  is spent, and phase 1 **stops and reports** if throughput is still ~255 tok/s.
* **Terminology corrected**: zero hash collisions proves item-level exclusion,
  not distributional novelty. Sets are labelled source-disjoint, split-held-out,
  or split-held-out near-domain item-disjoint. No out-of-domain claim is made.
* **Checkpoints inventoried and cleaned**: 4.19 GiB reclaimed on the dev box
  (117 → 121 GiB free); **0 reclaimed on the relay**, where ordinary deletion
  cannot free LFS quota and every option that could invalidates existing
  revisions — reported for a separate decision, not performed.

## 11. Next actions

Ordered by expected value per dollar. Items 1–3 came out of the 2026-08-04
organization pass; all three are cheap relative to what they decide.

1. **Truncate padded blocks before the forward — a free ~3× on every future
   training run.** Blocks are **66–73% padding** (0.86M rung: 1,349,125 real
   positions in 617 × 8192 = 5,054,464). The trainer calls `self.student(ids)`
   and `self.teacher(ids)` on the full 8192 width, so both models — including the
   **4B teacher, which runs online every step** — spend most of their FLOPs on
   pad positions whose logits are then masked out of CE, KD and accounting.

   This is safe to do exactly, not approximately: padding is right-aligned and
   attention is causal, so a real token's hidden state does not depend on the pad
   run. Verified numerically on 2026-08-04 (tiny Qwen3, sdpa): max |Δ| on real
   positions **2.8e-07**, `allclose` True. `micro_blocks` is already **1**, so no
   length bucketing is needed — slice `ids[:, :unpadded]` in `_micro_losses` and
   `_eval_blocks`, where `unpadded` comes from the content mask already in hand.

   Expected: **~3.0–3.75× fewer linear FLOPs** and up to ~9–14× fewer attention
   FLOPs; realized wall-clock gain must be measured, since short blocks may
   become launch-bound. **This also dissolves most of §8's accepted 3.35×
   `tool_calling` penalty** — tool blocks are 0.092 *fill*, i.e. **90.8%
   padding**, so truncation shrinks exactly the blocks that dominate the count.
   Revisiting §8 was already flagged as due; this is the cheaper answer than
   relaxing the packing rule, and it changes no rendered token.

   Validation gate: identical loss to 1e-5 on a fixed batch before/after, then a
   measured tokens/s comparison. CPU-testable; no GPU needed to prove the
   equivalence.

2. **Measure a same-geometry reference on `capability-v2` (~$0.50, one battery
   run).** Every capability number in the project is measured against the 4B
   teacher or against this project's own students. **`correct` is at the floor on
   every set, arm, rung and seed**, and nothing on record distinguishes "the
   student is bad" from "the target is not reachable at 0.6B under this
   protocol". Running the released **Qwen3-0.6B** — the student's exact geometry,
   properly trained — through the frozen battery settles it in one run:
   * if it also scores ~0, the battery or the protocol is misspecified for this
     size and the whole reasoning objective needs restating;
   * if it scores well, the gap is attributable to the distillation recipe and
     its size bounds what recovery can buy.

   This is the highest information-per-dollar measurement available and it
   gates how to read every result so far.

3. **Quantify the teacher-forcing gap, then use the rollout stack that already
   exists.** The defining unexplained fact: teacher-native held-out CE is
   **1.0042 nats (PPL 2.73)** at the 5.50M rung — the student predicts teacher
   tokens very well *given a teacher prefix* — while free generation solves ~0%
   and degenerates. Degeneration is **`cycle`-dominated at every checkpoint**
   (11–60 per 76 prompts, vs 0–15 `low_novelty` and 0–12 `rambling`), i.e. exact
   repetition loops, which is the signature of a model with good local statistics
   and no sustained plan — the classic exposure-bias picture.

   Meanwhile `src/aadistill/rollout/` holds **2,075 lines of tested
   infrastructure** (engines, snapshots, off-policy measurement) and **no
   training path consumes any of it**. Stage 3 sub-stage 3 (student-forced span
   recovery) and Stages 4–5 are exactly the remedy AGENTS.md already specifies.
   Step one is cheap and CPU-side: report CE under teacher forcing against CE on
   the student's *own* prefixes for the same prompts, so the gap is a number
   before anything is trained against it.

4. **Replace the headline metric.** `best_holdout_nll` is retired
   ([decision](decisions.md) 2026-08-04). `behavior_score_v0` resolves at only
   **3.3×** its seed spread and cannot rank rungs. Aggregate **protocol validity
   on `capability-v2`** moved consistently on both seeds in phase 1 (−0.0898,
   −0.0731) and is the best-resolving generation metric currently available;
   `correct` stays reported but is at floor and cannot rank anything yet.

5. **Phases 2–3 of Experiment 2 remain unauthorized**, and **phase 3 should not
   run as written** — it was built around the retired metric
   ([`PROPOSAL.md`](PROPOSAL.md) §12). $17.03 of the $30 allocation is unspent.

6. Still open, unchanged: the approved relay history squash (destructive, confirm
   separately) and the four dev-box-only Experiment 1 arms. Not a prerequisite —
   Experiment 2 stores weights on the dev box.
