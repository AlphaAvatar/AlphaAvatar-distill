# Active proposal — Experiment 2: three sequential 2.96M diagnostics

**Status 2026-08-03: Experiment 1 is COMPLETE** (24 arms + a step-matched
compute control, four metrics each; [`EXPERIMENTS.md`](EXPERIMENTS.md) §11).
Experiment 2 replaces the old "data mixing" plan with **three sequential
diagnostic experiments, all at the Experiment 1 2.96M rung**, run one at a time
with a decision between each:

1. **Data cleaning** — did bad teacher targets cause the 2.96M deterioration?
2. **Loss (KL-only first)** — does removing CE fix it?
3. **Learning rate** — does a lower LR fix it, or only postpone it?

They are **not** a Cartesian sweep. Each phase reuses the winner of the previous
one as its control and trains only the new arm.

**Phase 1 is prepared and costed below. It is NOT launched: it does not fit the
remaining budget** (§7). Zero GPU time has been spent.

---

## 1. Why 2.96M — and the nuance the maintainer asked for

The rung was selected because Experiment 1 showed held-out NLL deteriorating
around this scale. Reconstructed per seed from the arms' own logs, the PCA
held-out NLL (FineWeb-Edu, `data/warmup/holdout_v1.jsonl`, 40 samples / 21,080
tokens) is:

| rung (unique supervised) | steps | sa | sb | mean | seed \|Δ\| |
|---:|---:|---:|---:|---:|---:|
| 252,985 | 324 | 6.7169 | 7.8699 | 7.2934 | 1.153 |
| 460,088 | 570 | 6.1616 | 6.2948 | **6.2282** | 0.133 |
| 864,750 | 1,023 | 8.8758 | 9.3649 | 9.1204 | 0.489 |
| 1,600,353 | 1,761 | 9.7145 | 9.4845 | 9.5995 | 0.230 |
| **2,960,507** | **2,916** | 10.4031 | 9.7864 | 10.0948 | 0.617 |
| 5,501,372 | 4,412 | 10.7875 | 9.4548 | 10.1212 | 1.333 |

**The nuance: the rise does not begin at 2.96M.** It begins at **0.86M**, where
the single largest jump occurs (+2.89 nats on the seed mean, 0.46M → 0.86M).
By 2.96M the curve is already flattening: 1.60M → 2.96M is +0.495, and
2.96M → 5.50M is **+0.026**, an order of magnitude below the 0.617–1.333 nat
between-seed spread at those rungs.

Two further facts constrain what phase 1 can plausibly find:

* **Held-out NLL tracks optimizer steps, not unique data.** The step-matched
  control (`e1_ctl_r0250k_sa_pca_stepmatched`: 0.25M of data, 4,412 steps)
  lands at **10.7082** against the 5.50M/4,412-step arm's **10.7875** — a
  0.079 gap where the unique-data ratio is 21.7×, and 1/8 of the between-seed
  |Δ| at that rung. Whatever this metric is measuring, 21.7× more data barely
  moves it while 13.6× more steps moves it 4 nats.
* **Teacher-native val CE falls monotonically the whole way** (2.1183 → 1.0042),
  at 74× the between-seed noise. The two metrics genuinely disagree.

So phase 1 is a well-posed question, but the prior from Experiment 1 is that
**phase 3 (learning rate / optimization duration) is where the mechanism most
likely lives.** That is an argument about ordering, not about validity, and the
maintainer's order is kept.

**Every arm's per-seed val-CE trajectory** (10 eval points each) was recovered
and is in `artifacts/stage3/e1_consolidated.json`; the pca arms' values came
from their console logs after `consolidate_e1.py` was taught to read them (their
`train_log.jsonl` were destroyed by the Experiment 1 `scp` basename collapse, so
six pca arms previously read `val_ce: null`).

**A limitation to carry into every phase:** Experiment 1 ran with
`save_every: 1458, keep_last: 1`, so **D0 has only its final checkpoint**. There
is no held-out-NLL trajectory *within* any Experiment 1 run and no intermediate
checkpoint to score. The fixed-step endpoint is therefore the only fully matched
D0↔D1 comparison; "best held-out-NLL checkpoint" and "best validation-CE
checkpoint" are computable for new arms only. Phase 1 fixes this going forward
by evaluating and retaining checkpoints at every eval point.

## 2. The verified Experiment 1 configuration (binding)

Resolved from the committed source, the arm configs and the runs' own manifests
— not from any coefficient quoted in conversation.

**Source state.** Experiment 1 ran at commit `69c3fe1` (dirty; uncommitted state
`2e04f683…`). `src/aadistill/training/train.py`, `scripts/training/train_stage3.py`
and `src/aadistill/data/ladder.py` are **byte-identical between `69c3fe1` and
HEAD**, and `sha256_json(configs/stage3/e1/e1_r2960k_sa_rand.json)` reproduces
the `config_sha256` `34d8a4c8…` recorded in that run's manifest. The committed
configuration is the one that ran.

**Effective loss** (`train.py::step_once`, `masked_ce`, `kd_forward_kl`):

```
L = 0.25 · (Σ_{p ∈ CE-mask}  CE(student_p, target_p)) / N_ce
  + 1.00 · (τ² · Σ_{p ∈ KD-mask} KL(teacher_τ,p ‖ student_τ,p)) / N_kd
```

| property | value | where |
|---|---|---|
| `ce_weight` | **0.25** | `configs/stage3/e1/e1_r2960k_s*_pca.json` |
| `kd_weight` | **1.0** | same |
| KD direction | **forward KL**, `KL(teacher ‖ student)` = `Σ p_t (log p_t − log p_s)` | `train.py:297` |
| temperature | **τ = 1.0**, applied to both, with **τ² scaling present** (a no-op at τ=1) | `train.py:295–299` |
| `kd_scope` | **`all`** | config; `prediction_mask` |
| CE positions | `loss_mask[:, 1:]` — **assistant-target tokens only** | `masked_ce` |
| KD positions | `content_mask[:, 1:]` — **every real (non-pad) position**, including prompt and context | `prediction_mask` |
| **CE and KD do NOT share positions** | KD is a strict superset of CE's mask | — |
| normalizers | `N_ce` = CE targets in the step, `N_kd` = KD positions in the step; **different denominators**, both computed per optimizer step over `blocks_per_step=2` blocks | `step_once:516–522` |
| padding | excluded from both (`content_mask`), and excluded from `N_kd` | 2026-07-28 decision |
| teacher-forced prefix | the packed block itself; teacher and student consume identical `input_ids` | `_micro_losses` |
| teacher distributions | **computed online**, full-vocab, every step — **no cached logits exist anywhere in the repo** | `train.py:493` |
| reduction | float32 sum, chunked over 512 positions | `kd_forward_kl` |

**Optimizer and schedule** (identical across all E1 arms):
AdamW, **η = 5e-5**, betas (0.9, 0.95), eps 1e-8, weight decay 0.01, grad clip
1.0; linear warmup 146 steps then cosine to `min_lr_frac` 0.1 (= 5e-6) at
`total_steps` 2,916. Verified against the console logs: step 10 shows
`lr 3.42e-06` = 5e-5 × 10/146, and the final steps show `5.00e-06`.

**Trainable set:** all attention (q/k/v/o + q/k norms), all MLP, all norms —
440,467,456 of 596,049,920 parameters; the tied embedding stays frozen.

**Seeds:** `sa` = **20260726**, `sb` = **20260801**. The seed feeds only
`epoch_permutation` (`g.manual_seed(seed·1000003 + epoch)`), so the two seeds
differ **only in block order**; weights at step 0 are identical.

**Start point:** `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`,
`model.safetensors` sha256
`86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54` —
**re-verified on the dev box today**, matches the pinned value.

## 3. Phase 1 — D1 feasibility audit (COMPLETE, zero GPU)

### 3.1 The corpus is intact and reproduces exactly on CPU

`candidates.jsonl` (897 MB, all `n=4` completions with token ids and verdicts)
and `sessions.jsonl` are on the dev box and **hash-verified**: `f7f5035e…` and
`2b4edc2e…`, matching the corpus manifest. Re-rendering 400 sessions with the
local tokenizer reproduces `n_supervised_tokens`, `n_rendered_tokens`,
`n_system_tokens` and `system_key` **400/400 exactly**, and the local tokenizer
reproduces the corpus's vocab hash `3ec3c124…` and chat-template hash
`3802169b…` despite running transformers 5.13.1 against the corpus's 5.14.1.

> **Records conflict, and STATE.md is the one that is right.**
> `artifact_manifests.md` still says corpus v2 is "NOT YET PERSISTED" while
> `STATE.md` §2 says it was uploaded and 9/9 hash-verified on 2026-08-01. The
> local copy is verified either way; the relay claim is corrected in the
> manifest as part of this session.

### 3.2 What cleaning does

`aadistill.data.cleaning` (`RULES_VERSION = "clean-v1"`), driven by
`scripts/data/build_cleaned_corpus.py`. **Nothing is generated** — every target
is a completion the teacher already produced under the recorded preset. Four
ordered gates then a deterministic choice:

1. **serialization** — non-empty answer, closed `</think>`, exactly one close and
   no re-open, no stray template markers.
2. **correctness** — the slice's mechanical key: `gsm8k` final number, `openmath`
   `\boxed{}`, `rag_evidence` / `multihop_qa` gold-span containment (yes/no golds
   must lead). Calls `verify_answer_key`, **not** `verify` — the latter also
   applies `hygiene_reason`'s `MAX_ANSWER_WORDS=600`, a generic length gate that
   P3/P10 forbid and that the corpus build already declined to apply.
3. **tool protocol** — a tool call with no schema offered is rejected; with a
   schema, every call must be JSON, name a declared tool, and supply an object
   of arguments containing the required keys.
4. **completion** — natural `<|im_end|>`, not context-limited / over-budget /
   unfinished, and not degenerate under the **current** detector. This is
   strictly stricter than the corpus build: the `rambling` novel-n-gram signal
   was added in `e13f8f9` and **did not exist** on 2026-08-01.
5. **selection** — keep the corpus's own candidate when it survives all four;
   otherwise the shortest surviving candidate by **supervised-token length**,
   tie-broken by candidate index.

**Two limitations, stated rather than papered over:**

* `code` (1,123 sessions) and `tool_calling` (2,600) have **no mechanical
  correctness key**. They pass gate 2 by exemption, recorded as
  `unverifiable_slice`. Their targets are checked for structure, protocol and
  degeneration only. Fluency is not evidence of correctness and is not treated
  as such.
* **The shortest-survivor tie-break contradicts a recorded project finding.**
  `verify.select`'s docstring records that shortest-correct on the math slices
  "systematically selects answers that skip the derivation". The maintainer's
  specification overrides it, and the exposure is bounded — the rule fires only
  on the 241 prompts where the original failed — but it is a live risk on
  `openmath`, whose replacement rate is the highest at 9.9%. **Median-length
  selection is one flag away if you want it.**

`refusal` is **not applicable**: the corpus has no unanswerable slice
(`refusal_uncertainty` was excluded from generation by the 2026-07-30
alignment-tax decision), so "unsupported questions must be refused" has nothing
to fire on here. It stays an evaluation guard rail.

### 3.3 Cleaning result — corpus wide

11,174 accepted sessions screened in 114 s of CPU. **10,778 kept (96.5%)**.

| type | examples | kept | keep | replaced | replacement | no valid candidate |
|---|---:|---:|---:|---:|---:|---:|
| code | 1,123 | 1,123 | 1.000 | 0 | 0.000 | 0 |
| gsm8k | 1,698 | 1,596 | 0.940 | 88 | 0.055 | 102 |
| multihop_qa | 1,074 | 980 | 0.912 | 58 | 0.059 | 94 |
| openmath | 579 | 413 | **0.713** | 41 | **0.099** | 166 |
| rag_evidence | 4,100 | 4,066 | 0.992 | 54 | 0.013 | 34 |
| tool_calling | 2,600 | 2,600 | 1.000 | 1 | 0.000 | 0 |

Candidate-level rejection reasons (first failing gate, 44,696 candidates):

| type | dominant reasons |
|---|---|
| gsm8k | `correctness:answer_mismatch` 727 · `empty_answer` 9 · `not_terminated` 3 |
| openmath | `answer_mismatch` 500 · `empty_answer` 198 · `no_boxed` 186 · `not_terminated` 59 |
| multihop_qa | `gold_span_missing` 586 · `empty_answer` 12 |
| rag_evidence | `gold_span_missing` 357 · `yesno_not_leading` 3 |
| code | `empty_answer` 231 · `context_limit_reached` 163 |
| tool_calling | `empty_answer` 1 · `degenerate:rambling` 1 |

`openmath` behaves exactly as the corpus's own unenforced verdicts predicted
(0.380 per-candidate correctness): a third of D0's `openmath` targets teach a
wrong final answer, and cleaning cannot repair 166 of 579 prompts because none
of their four candidates is right.

### 3.4 The D1 rung is matched, and the residual is reported

D1 is packed with the **same packer and the same uniform mixture**, with the
session order **anchored to Experiment 1's own pack** (`--session-order`, new)
so the cut holds D0's prompts wherever cleaning kept them. The rung is cut at
**1,944 blocks**, matching D0 block-for-block, because optimizer-step parity is
the binding constraint.

| quantity | D0 | D1 | Δ |
|---|---:|---:|---:|
| packed blocks | 1,944 | 1,944 | **0.0000%** |
| optimizer steps (3 epochs, 2 blocks/step) | 2,916 | 2,916 | **0.0000%** |
| packed tokens | 15,925,248 | 15,925,248 | **0.0000%** |
| **unique supervised tokens** | **2,960,507** | **2,968,828** | **+0.281%** |
| sessions | 4,524 | 4,540 | +0.354% |
| terminal truncations | 352 | 375 | — |

Per-type supervised-token share drift: `code` 0.00 · `gsm8k` −0.02 ·
`multihop_qa` −0.04 · `openmath` −0.04 · `rag_evidence` +0.03 ·
`tool_calling` +0.05 **percentage points**. Both cuts realize uniform within
0.07 pp.

**Pre-registered tolerance, met:** block count, optimizer steps, packed tokens
and effective compute must match **exactly**; per-type share drift ≤ 0.25 pp;
unique supervised tokens within ±1.0%. Achieved: exact / ≤0.05 pp / +0.281%.

For scale: +0.281% of data is ~0.4% of a doubling, against a measured per-rung
val-CE effect of 0.14–0.35 nats per ~1.8× of data. It cannot manufacture a
result.

**Prompt overlap is 79.0%** of D0's rung (Jaccard 65.1%), and this is the one
number that did not come out as well as intended. It decomposes cleanly:

* 4,524 D0 rung sessions
* **166 (3.7%) dropped by a cleaning rule** — the intended effect
  (gsm8k 29, multihop_qa 59, openmath 68, rag_evidence 10)
* **784 (17.3%) survived cleaning but fell outside the 1,944-block prefix** —
  a re-packing artifact, not a data decision
* 3,574 (79.0%) shared; **966 top-up prompts** entered from the clean corpus
  outside D0's rung, preserving the mixture

Anchoring the session order lifted overlap from 63.6% to 79.0%. Anchoring the
**block** order as well was tried and **rejected on measurement**: it reached
higher overlap but drifted the mixture to `code` 0.193 / `openmath` 0.130
against a declared 0.1667 — 5.9 pp, which would have turned a target-quality
experiment into a mixture experiment. The rejected path and its numbers are
recorded in `build_token_ladder.py` so it is not retried blind.

Target-length distributions barely move (supervised tokens/session, p50):
`code` 2044→2014 · `gsm8k` 805→806 · `multihop_qa` 618→635 ·
`openmath` 1749→1750 · `rag_evidence` 338→336 · `tool_calling` 256→258.
Cleaning did **not** systematically shorten the targets.

### 3.5 KD-target alignment: nothing to recompute

**Teacher distributions are computed online.** `Trainer._micro_losses` runs the
teacher forward on the same packed block as the student every step
(`train.py:493`); `grep` finds no cached, stored or top-k logits anywhere in the
repository. A replaced completion is therefore paired with the teacher
distribution of **its own** teacher-forced prefix, system message and
serialization, by construction. There is no stale logit cache that could belong
to a different candidate, and **teacher-logit recomputation cost is zero** — it
is already inside the measured step time.

### 3.6 Artifacts produced (all CPU, all git-ignored under `artifacts/`)

| artifact | sha256 | size |
|---|---|---|
| `stage3/corpus_v2_clean/sessions_clean.jsonl` | `7d22b3e0…` | 68 MB |
| `stage3/corpus_v2_clean/cleaning_audit.json` | `8d97e03a…` | small |
| `stage3/ladder_uniform_clean_anchored/blocks.npz` | `1c8792db…` | 14 MB |
| `stage3/ladder_uniform_clean_anchored/ladder.json` | `51047640…` | small |
| `stage3/ladder_uniform_clean_anchored/audit.jsonl` | `213a422d…` | 6 MB |
| `stage3/e2_d1_corpus_audit.json` | `a27ae3df…` | small |
| `eval/e1/gsm8k_strict_rescore.json` | `57fa297d…` | small |

## 4. The corrected GSM8K evaluator, and corrected D0 metrics

`aadistill.evaluation.strict_answer` replaces `behavior.final_number`
("last number anywhere in the answer"), which credited numbers inside tool
calls, answers that never stated a conclusion, and degenerate loops that
happened to contain the gold value. The rule: **`\boxed{}` first; otherwise an
explicit standalone `Final Answer:` / `Answer:`; tool-call payloads stripped
before reading; no valid final answer means incorrect; protocol-invalid or
degenerate generations are incorrect regardless of content.** Termination and
protocol validity are returned as separate fields and never folded into
correctness.

All 25 Experiment 1 arms were re-scored **offline from stored generations**
(`scripts/evaluation/rescore_gsm8k.py`, no GPU). **The correction changes two
arms by one sample each** (`0250k_sa_pca` 0.010→0.000, `1600k_sb_pca`
0.050→0.040) and nothing else. Experiment 1's finding — no reasoning at any
rung — now stands under a rule that degeneration cannot game.

**Corrected D0 metrics at the 2.96M PCA rung:**

| metric | sa (20260726) | sb (20260801) | mean |
|---|---:|---:|---:|
| teacher-native val CE | 1.1468 | 1.1486 | 1.1477 |
| held-out NLL | 10.4031 | 9.7864 | 10.0948 |
| behaviour composite | 0.4413 | 0.3946 | 0.4180 |
| natural termination | 0.934 | 0.908 | 0.921 |
| degeneration | 0.066 | 0.092 | 0.079 |
| generated tokens p50 | 343 | 320 | 332 |
| **GSM8K strict EM** | **0.020** | **0.010** | **0.015** |
| GSM8K protocol-valid | 0.880 | 0.900 | 0.890 |
| GSM8K final-answer present | 0.870 | 0.910 | 0.890 |
| tool_call axis | 1.000 | 0.917 | 0.958 |
| grounding axis | 0.0625 | 0.000 | 0.031 |
| math axis | 0.000 | 0.000 | 0.000 |

## 5. Phase 1 arm table

| arm | data | loss | LR | init | seeds | steps | trained? |
|---|---|---|---|---|---|---|---|
| **D0** (control) | E1 uniform 2.96M, 2,960,507 sup tok | CE 0.25 + fwd-KL 1.0, τ=1, scope `all` | 5e-5 | Stage 1 PCA `86fbba78…` | 20260726, 20260801 | 2,916 | **already trained — reuse, do not retrain** |
| **D1** (treatment) | cleaned `clean-v1`, 2,968,828 sup tok, 1,944 blocks | **identical** | **identical** | **identical** | **identical** | 2,916 | **not yet** |

Everything except the target text is byte-identical: same init hash, same
trainable set, same optimizer, packing, batch construction, scheduler shape,
warmup proportion, system-message policy, serialization contract, evaluation
prompts and decoding protocol.

## 6. Pre-registered D1 selection gate

Fixed before training, using Experiment 1's measured seed variation.

**Select D1** only if **both** hold:

* held-out NLL improves by **more than 0.617 nats** on the seed mean — the
  measured between-seed |Δ| at this exact rung — *and* the improvement has the
  same sign on both seeds; **and**
* no material degradation, defined per axis before the fact:
  strict GSM8K EM ≥ D0 − 0.02; protocol-valid rate ≥ D0 − 0.05; natural
  termination ≥ D0 − 0.05 (its seed |Δ| is 0.026); degeneration ≤ D0 + 0.05;
  RAG/grounding and tool-call axes ≥ D0 − 0.10 (both are small-n and noisy).

**Reject D1** if held-out NLL fails to improve beyond noise → report that
cleaning does not explain the 2.96M deterioration, keep D0 as phase 2's dataset.

**Stop and report** if D1 improves NLL while degrading any capability axis past
its threshold. No composite score will be improvised to break such a tie.

Paired bootstrap CIs on the shared evaluation prompts for every generation
metric; val CE and held-out NLL are reported per seed, as a mean, and as full
trajectories.

## 7. Cost, and why phase 1 is not launched

**Verified cumulative spend: $96.02** = $34.52 (through corpus v2) + $47.60
(E1 training) + $8.10 (control + first evaluation) + $5.80 (evaluation sweep).
Against the **$100 hard cap** that leaves **$3.98**.

Phase 1 costed from measured Experiment 1 rates — the 2.96M PCA arm's own
orchestrator timestamps (launch 12:49:20 → `TRAIN_DONE` 15:40:51 = **2.859 h**,
`post_run` 69 s) and the evaluation sweep's $5.80 / 25 checkpoints:

| item | hours | $ @ $0.99/h L40S |
|---|---:|---:|
| pod provisioning + setup (image, venv, teacher, init, pack) | 0.50 | 0.50 |
| D1 training, 2 seeds × 2.859 h | 5.72 | 5.66 |
| inline val-CE + held-out-NLL evals at 8 checkpoints/arm | 0.35 | 0.35 |
| capability battery on 4 checkpoints (D1 ×2 **and D0 ×2**) | 1.40 | 1.39 |
| upload, hash verification, teardown | 0.20 | 0.20 |
| **total** | **8.17** | **$8.09** |

Contingency: Experiment 1 ran 3.08–3.8 s/step; at the top of that range training
is 6.3 h and the total is **~$9.3**.

D0 must be re-run on any **new** capability set, because its stored generations
only cover Experiment 1's behaviour and GSM8K prompts — offline re-scoring
(already done, free) cannot produce knowledge-QA, verifier-backed math, multihop
or RAG numbers that do not exist yet. Both D0 checkpoints are available:
`e1_r2960k_sa_pca` on the relay, `e1_r2960k_sb_pca` on the dev box under
`artifacts/stage3/rescued/`, hash-verified.

### The complete matched two-seed phase does not fit $3.98. Not launching.

**Minimum valid alternative — $7.40 (needs +$3.42 over the cap).** Drop the new
capability sets and evaluate D1 only on Experiment 1's existing battery
(behaviour + strict GSM8K + held-out NLL), so D0 needs **no** GPU
re-evaluation and offline re-scoring suffices:

| item | hours | $ |
|---|---:|---:|
| setup | 0.50 | 0.50 |
| D1 training, 2 seeds | 5.72 | 5.66 |
| inline evals | 0.35 | 0.35 |
| battery on D1's 2 checkpoints only | 0.70 | 0.69 |
| upload + teardown | 0.20 | 0.20 |
| **total** | **7.47** | **$7.40** |

What that gives up: knowledge/factual QA, verifier-backed math accuracy,
multihop accuracy and RAG evidence-supported correctness as **separate** axes —
i.e. exactly the decomposition that answers "is the NLL change knowledge or
reasoning". It still answers the D1 gate.

**A one-seed run is not offered.** At this rung the between-seed |Δ| is 0.617
nats on held-out NLL and 0.047 on the behaviour composite against a 0.1290 noise
floor; one seed cannot support the gate in §6, and the maintainer's instruction
forbids substituting one.

**Recommendation: raise the cap to $110.** That funds the full phase 1 at $8.09
(~$9.3 worst case) and leaves headroom to start phase 2, whose L1 arm is the
same 2,916 steps and therefore the same ~$6 of training.

### 7.1 Operational constraints to settle at launch

* **Checkpoint retention.** Experiment 2 arms save at all 8 eval points
  (`save_every: 364, keep_last: 9`) so the held-out-NLL trajectory and the
  best-NLL checkpoint exist — the gap Experiment 1 left. Each checkpoint is
  **4.3 GB** (2.3 GB model + 2.0 GB optimizer state), so an arm holds ~39 GB and
  the pod needs **~100 GB of container disk** for two sequential arms plus the
  teacher. Cheap, but it must be provisioned at creation.
* **The relay is still at its private-LFS limit** and the approved history
  squash has not run ([`decisions.md`](decisions.md), 2026-08-02). Only the
  final and best-NLL **model** directories should be uploaded (2.3 GB each,
  optimizer state dropped — it is only needed for resume), and even that needs
  the squash first or four Experiment 1 arms stay dev-box-only alongside these.
  Settle this before launch, not during teardown.
* **One pod, created with `--min-cuda-version 13.0`** so the same host can train
  *and* run the vLLM battery. Experiment 1 needed two pods and a third for
  evaluation because the training pods' driver could not host vLLM 0.26; that
  cost a full provisioning cycle and is avoidable here.
* **Pods idle-bill.** Teardown is tied to job completion, not to a generous
  `--terminate-after` backstop, and status polling is set up at launch.

## 8. Phases 2 and 3 (not prepared, not costed in detail)

Prepared only after phase 1 reports.

**Phase 2 — KL-only.** `L1` = the selected dataset, CE removed, KL component
preserved exactly (coefficient 1.0, forward direction, τ=1 with τ² scaling,
scope `all`, same masks/prefixes/normalizer). No rescaling, no LR change. `L0`
is whichever run phase 1 selects — already trained either way. Requires a
gradient audit first, on fixed calibration batches: raw and weighted CE/KL,
gradient norms, CE↔KL gradient cosine and the fraction of batches where it is
negative, and the teacher probability on the CE hard target, broken down by task
type and by whether cleaning replaced the candidate. Note in advance that CE and
KL run on **different position sets** with **different normalizers**, so
"removing CE" is not a pure reweighting and the gradient-magnitude change is
part of the intervention. Training cost ≈ D1's (~$6 for two seeds).

**Phase 3 — learning rate.** `R1` at η/2, `R2` at η/4, whole schedule scaled
(peak, warmup shape and the `min_lr_frac` floor together), same fixed 2,916-step
budget, both seeds. `R0` is reused. ~$12 for two arms × two seeds, plus
evaluation. This phase needs intermediate checkpoints to locate the onset of
deterioration — which is why phase 1 already adds them.

## 9. What will not happen

No Cartesian sweep. No phase launched before the previous one reports and is
approved. No retraining of any valid historical control. No arm at any rung
other than 2.96M. No one-seed substitution.
