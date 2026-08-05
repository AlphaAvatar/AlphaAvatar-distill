**Updated:** 2026-08-05 14:05 UTC · branch `main` · commit `81d668a` ·
⚠️ **ONE POD RUNNING AND BILLING: `94slla57nnqjqa`, 1× L40S, $0.99/h**, created
14:02 UTC with a RunPod-side backstop at **21:32 UTC (450 min, $7.43 max)**.
Verified spend before this session **$117.32** of the **$126.02** cap; Experiment
3 is projected at **$6.16** against the **$8.70** remainder.

**Experiment 3 is in flight** — does restricting attention updates improve
autonomous generation stability at the 0.86M rung? Registered before training in
[`logs/e3_registration.json`](e3_registration.json) and
[`EXPERIMENTS.md`](EXPERIMENTS.md) §20.

| arm | trainable attention | status |
|---|---|---|
| **A0** = `p2_ceheavy_{sa,sb}` | q/k/v/o + q_norm/k_norm full-rank | **baseline, not retrained** |
| **A1** = `e3_a1_frozen_attn_{sa,sb}` | q_norm/k_norm only, projections frozen | training |
| **A2** = `e3_a2_lora_attn_{sa,sb}` | A1 + LoRA r8 on q/k/v/o, base frozen | queued behind A1's freeze gate |

**If this session dies, nothing is stranded.** The launcher runs under `nohup`
(`scripts/pod/e3_launch.sh`), deletes the pod itself on `ALL_DONE`, and is backed
by an independent dev-box watchdog plus the RunPod-side deadline. To check:
`runpodctl get pod 94slla57nnqjqa`; the launcher log and all state live in this
session's scratch dir, and results land in `/home/ecs-user/aad-artifacts/e3/`.

**After the pod returns**, three commands finish the experiment:

```bash
tar xzf /home/ecs-user/aad-artifacts/e3/e3_side.tar.gz -C .    # audit artifacts
PYTHONPATH=src uv run python scripts/evaluation/eval_ppl.py \
  --data data/warmup/holdout_v1.jsonl \
  --model /home/ecs-user/aad-artifacts/p2_ceheavy/p2_ceheavy_sa \
  --model /home/ecs-user/aad-artifacts/p2_ceheavy/p2_ceheavy_sb \
  --model /home/ecs-user/aad-artifacts/e3/e3_a1_frozen_attn_sa/model \
  --model /home/ecs-user/aad-artifacts/e3/e3_a1_frozen_attn_sb/model \
  --model /home/ecs-user/aad-artifacts/e3/e3_a2_lora_attn_sa/model \
  --model /home/ecs-user/aad-artifacts/e3/e3_a2_lora_attn_sb/model \
  --max-seq-len 1024 --dtype bfloat16 --out artifacts/audit/e3_holdout_nll_bf16.json
#   ... repeat with --fake-quant int8 --fake-quant-scope {all,decoder}
PYTHONPATH=src uv run python scripts/evaluation/analyze_e3.py
```

Held-out NLL runs on **CPU on purpose**: 25 s per model, it reproduces the
recorded GPU value to 0.02%, and it is the one metric measured on a single
device for all six arms. A0's three precisions are already computed (§20.6).

## NO EXISTING MODEL HAS YET COMPLETED THE STAGE 2/3 OBJECTIVE

A post-hoc re-evaluation of every retained model under the clarified stage
objectives is in [`EXPERIMENTS.md`](EXPERIMENTS.md) §19 (CPU, $0, no generation).
Two decision records: [`decisions.md`](decisions.md), 2026-08-05.

**Stage 0/1 = initialization. Stage 2/3 = behaviour recovery.** Primary Stage 2/3
metric:

```
usable_rollout = non_empty AND natural_termination AND no_severe_repetition
                 AND no_context_limit AND protocol_valid
```

Primary = autonomous rollout behaviour. Secondary = correctness, per-task
correctness, correctness | usable rollout. **Diagnostic only** = teacher-forced
top-1 and rank, teacher-native CE, FineWeb NLL, training loss. Keep them separate;
never combine onto one scale.

### Stage 0/1 — the initialization works, and its downstream value is proven

| | step-0 NLL | perplexity |
|---|---:|---:|
| teacher Qwen3-4B-Thinking | 2.6264 | 13.8 |
| **Stage 1 PCA init** | **11.7482** | 126,532 |
| random init | 12.1286 | 185,090 |

−0.3804 nats: real but small (~4% of the gap to the teacher), n=1 per condition.
**Downstream it is decisive.** Across the 12 matched Experiment 1 pairs, PCA vs
random on `usable_rollout` is **12 wins / 0 ties / 0 losses** on behaviour prompts
and **11 wins / 1 tie / 0 losses** on gsm8k — the single tie is 0.25M `sb` where
*both* score 0.0000, a shared floor rather than a contest PCA failed to win. Mean
**+0.364** / **+0.494**. Random init produces **zero usable rollouts at every rung
through 2.96M**. FineWeb NLL even *reverses* by 5.50M while behaviour stays 8×
apart — lower held-out NLL is not recovered behaviour.

### Stage 2/3 — nothing separates; no prospectively defined gate has been passed

**No model has demonstrated passage of a prospectively defined behaviour-recovery
gate.** No such gate existed when any of these runs launched, and **no threshold
may be invented post hoc** — this is the absence of a registered criterion, not a
measured failure against one.

| family | usable sa / sb | mean | spread | overall correct sa / sb (mean) | correct \| usable sa / sb |
|---|---|---:|---:|---|---|
| P0-assistant | 0.6067 / 0.5667 | **0.5867** | 0.0400 | 0.1867 / 0.1067 (0.1467) | 0.2747 / 0.1765 |
| **P1 = P0-real** | 0.5133 / 0.5933 | 0.5533 | 0.0800 | 0.1533 / 0.2133 (0.1833) | 0.2727 / 0.2921 |
| P2-ceheavy | 0.5200 / 0.5467 | 0.5333 | 0.0267 | 0.2000 / 0.1800 (**0.1900**) | **0.3590 / 0.2927** |

**Every gap is smaller than P0-real's own 0.0800 seed spread**, and paired at the
prompt level both interventions gain on `sa` and lose on `sb`.

* **P1 = P0-real is the incumbent reference checkpoint** — retained for continuity
  of comparison. It is **not** the best checkpoint on any primary measure and is
  not confirmed by behaviour.
* **P0-assistant holds the highest observed mean usable-rollout rate, 0.5867.** It
  is **not seed-consistent** (0.6067 / 0.5667; paired +14 on `sa`, **−4** on `sb`)
  and **its weights no longer exist**, so it cannot be re-measured or built on.
* **P2-ceheavy holds the best correctness conditional on a usable rollout**
  (0.3590 / 0.2927, highest on both seeds). Its overall correctness is reported
  separately (0.2000 / 0.1800, mean 0.1900). **It is not promoted** — correctness
  may only break a tie between behaviour-comparable candidates.

### The dominant failure is that the model does not stop

**31.1% of all 900 rollouts run to the 8,192-token context limit**, accounting for
280 of 395 protocol failures. Delimiter errors are a distant third (44/900, 4.9%).
`openmath` correctness given a usable rollout is **0.000 on five of six arms**.

### Two caveats that bound every number above

* **`usable_rollout` is one measurement, not five.** `protocol_valid` implies
  `non_empty` and `natural_termination` by construction (505/505),
  `not natural_termination` ⟺ `context_limit` (900/900), and
  `usable_rollout == protocol_valid` on **897/900**. Always report the components.
* **Correctness must be re-scored, never read from stored `correct`.** The stored
  field puts P0-real-sa at 0.0067; the corrected scorer at 0.1533 — pre-fix and
  post-fix arms are not comparable on the stored field.

### Strongest untested lead: the token budget, not the loss

All three Stage 2/3 families trained at the **0.86M rung**. **Preliminary**
evidence that this is not the best rung:

> **n=76 behaviour prompts · E1 behaviour-wave harness · degeneration stop
> ACTIVE** (loops cut at ~768 tokens, max 1,536)

| rung | sa | sb |
|---|---:|---:|
| 0.86M *(the candidate rung)* | 0.3684 | 0.4342 |
| **1.60M** | **0.4868** | **0.5132** |
| **2.96M** | **0.5921** | 0.5395 |

**These values must not be compared with the 150-example usable-rollout rates
above** — different prompt population, different harness, and a stop policy that
changes the termination and context-limit components outright. The same weights
score 0.3684 here and 0.5133 there. The higher rungs have never been run through
the 150-example harness and are **not evaluable** on it without new generation.

**Recoverable and hash-valid (verified 2026-08-05).** 30/30 local files match
their pod-side manifests. Relay digests recorded in
`artifacts/audit/relay_e1_digests.json`; **two were downloaded and recomputed
byte-exact** (`e1_r1600k_sa_pca` and P1-sa). Every E1 rung is covered across local
+ relay.

> ✅ **P1 storage risk RESOLVED 2026-08-05.** Both P1 arms are now copied to
> `artifacts/stage3/rescued/` and **hash-verified against the relay LFS digests**
> (`18ee10a1…`, `f66de532…`), each with a full tokenizer, each loading on CPU.
> The relay's pending history squash can no longer destroy the reference weights.
> **Every Stage 2/3 candidate that still has weights now has a verified local
> copy;** P0-assistant remains unrecoverable.

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
teacher's thinking protocol.

**Corrected 2026-08-05 — "zero context-limit hits" was a stop-policy artifact.**
The behaviour wave ran with the degeneration stop active, which cuts a repetition
loop early and records `stop_reason: degeneration`, so `context_limit_rate` read
0.0000. The three-mode harness does **not** cut, and the *same weights*
(`e1_r0860k_sa_pca` = P0-real-sa) then run to the limit: median 8,099 tokens,
**31.1% of all 900 rollouts context-limited**. Same phenomenon, two accounting
policies. Do not read "zero context-limit hits" as evidence that generations
terminate — it means they were stopped before they could be counted.

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
  phase 1 reported `context_limit_rate` **0.0000 at all 20 checkpoints** — but
  only because the **degeneration stop was active**, cutting loops before the
  limit (`stop_reason: degeneration`). With the stop disabled, the same weights
  context-limit on 31.1% of rollouts. When comparing context-limit rates, the
  stop policy must match or the comparison is meaningless.
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

## 14. Experiment 2 phase 1 — COMPLETE 2026-08-04 (historical; see EXPERIMENTS.md §12)

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

Ordered by expected value per dollar. **$8.70 remained before Experiment 3, which
is projected to consume $6.16 of it. Nothing beyond E3 is authorized.**

**0. Finish Experiment 3.** Collect the pod's artifacts, run the six-model NLL
table and `analyze_e3.py`, and write the verdict into `EXPERIMENTS.md` §20 under
the rules registered in §20.4 — including the two that only ever *block* a
promotion (no single-seed winners; no promotion for terminating earlier if
correctness given a usable rollout falls). A null is a result: §20.5 lists what
the design cannot settle, and that list was written before the numbers.

**What the last three experiments settled.** D0 (§16) located the bottleneck;
P0-assistant (§17) and P2-ceheavy (§18) each tried to fix it by reducing KD's
influence — once by scope, once by magnitude — and both failed the same way:
reasoning top-1 fell in proportion to the dose while free-generation tidiness
rose slightly. **The two-term CE/KD objective has been reweighted in both
available directions and neither helps.** Further reweighting of these two terms
is not worth money.

**The capacity question is CLOSED — do not re-run the battery.** The
near-geometry reference `Qwen/Qwen3-0.6B @ c1899de2` (identical parameter-bearing
fields and the **same 596,049,920** parameters; differs only in `rope_theta`
1e6 vs 5e6 and `max_position_embeddings`) was measured on the frozen
846-prompt `capability-v2` under **both** the project and native protocols in
Diagnostic A (§14.2) and rescored with the template-aware validator (§15.1).
Rescored `correct`, project / native:

| knowledge | math_verified | gsm8k | multihop | rag | answerability | safety |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.173 / 0.173 | 0.62 / 0.58 | **0.70 / 0.69** | 0.60 / 0.52 | **0.74 / 0.71** | 0.333 / 0.367 | 0.08 / 0.02 |

**A model with our student's geometry solves ~70% of GSM8K and ~74% of RAG under
our own protocol.** The task is reachable at 0.6B and the battery is not too
hard. Artifacts: `artifacts/eval/e2diag_rescored_v2/`.

**What this does not establish.** The battery closes whether a model at
approximately this size *can* perform substantially better. It does **not**
isolate the remaining gap to any one component. **Attribute the gap to the
broader training stack and trajectory — initialization, data, token budget,
stages, curriculum and objectives — until further evidence separates them.**
Writing "the gap is the recipe" overstates it and is not supported by anything on
record.

**Binding scope rule for teacher-forced reasoning top-1.** It is a
**within-family controlled-comparison metric** — valid for P1 vs P0-assistant vs
P2 because those share teacher distribution, architecture, initialization and
evaluation set. It **must not be promoted into a cross-model capacity scale**.
Scoring any model not trained on this teacher's traces against them measures
compatibility with that teacher's reasoning style, not capability. A proposal to
anchor it against the official `Qwen3-0.6B` was withdrawn before running for
exactly this reason ([`PROPOSAL.md`](PROPOSAL.md) §15,
[`decisions.md`](decisions.md) 2026-08-05). The capacity question needs no such
anchor — the battery above already answers it.

1. **Quantify the teacher-forcing gap, then use the rollout stack that already
   exists.** The defining unexplained fact is sharper after §18: teacher-forced
   reasoning top-1 is ~0.57 and moves by only ~0.05 across every recipe tried,
   while free-rollout correctness sits at 0.15–0.21 and is dominated by
   `cycle` degeneration — exact repetition loops, the classic exposure-bias
   signature. Train/inference distribution mismatch is the **leading hypothesis**
   for what remains, and the cheapest one to test. It is not established: what
   §17/§18 rule out is *reweighting the two existing loss terms*, which is
   narrower than "offline objectives". Initialization, data, token budget, stages
   and curriculum remain equally unseparated.

   `src/aadistill/rollout/` holds **2,075 lines of tested infrastructure**
   (engines, snapshots, off-policy measurement) and **no training path consumes
   any of it**. Stage 3 sub-stage 3 (student-forced span recovery) and Stages 4–5
   are exactly the remedy AGENTS.md specifies. Step one is cheap and CPU-side:
   report CE under teacher forcing against CE on the student's *own* prefixes for
   the same prompts, so the gap is a number before anything is trained against it.

2. **Test the NLL-variance observation, if a seeded run is being paid for
   anyway.** P2's FineWeb NLL spread was 66× tighter than P0-real's (§18.7). At
   n=2 per condition this is one draw per group and no claim is made — but if any
   future experiment runs ≥3 seeds, recording FineWeb NLL costs ~8 s per
   checkpoint and would either confirm or kill it for free.

3. **Headline metric.** `best_holdout_nll` is retired
   ([decision](decisions.md) 2026-08-04). `behavior_score_v0` resolves at only
   **3.3×** its seed spread and cannot rank rungs. Aggregate **protocol validity
   on `capability-v2`** is the best-resolving generation metric currently
   available; `correct` stays reported but is at floor. Note that free-rollout
   correctness on the 150-example diagnostic set has a measured seed spread of
   **0.0600** — any future selector needs an effect larger than that.

4. **Phases 2–3 of Experiment 2 remain unauthorized**, and **phase 3 should not
   run as written** — it was built around the retired metric
   ([`PROPOSAL.md`](PROPOSAL.md) §12).

5. Still open, unchanged: the approved relay history squash (destructive, confirm
   separately) and the four dev-box-only Experiment 1 arms.

**Done since this list was last written:** padding-suffix truncation is
implemented (`truncate_padding`, default **off** for P4 reproducibility, §13),
and the same-geometry capability reference is **complete** (§14.2, rescored
§15.1) — it is a settled fact above, not an action.
