# Current project state

**Updated:** 2026-07-30 (UTC+8 dev box) · branch `main` ·
**nothing running, nothing billing, no pods or volumes exist.**
Last session: a four-arm run completed end to end (**$4.87** of a $7.00
ceiling) and was then **relabelled a post-s2v1 continuation diagnostic** — its
start point was invalid, so it is not a teacher-target baseline. The corrected
baseline is specified and **awaiting approval before any paid work** (§8).

This file is the canonical handoff. It is a *snapshot*, not an archive —
historical detail lives in `logs/experiments/`, `logs/proposals/` and
`logs/decisions.md`, indexed at [`logs/indexes/`](indexes/).

---

## 1. Where the project is

First dense-model compression experiment. Teacher
**`Qwen/Qwen3-4B-Thinking-2507` @ `768f209d`** (hidden 2560, 36L, FFN 9728,
32Q/8KV) → student **0.6B-class** (hidden 1024, 28L, FFN 3072, 16Q/8KV, tied
emb). BF16 training, INT8 deployment target.

**Pipeline position:** Stage 0 → 1 → 2 (v0+v1) → **Stage 3 (in progress)**.
Stage 4/5 not started.

**Stage 3 is not exitable.** Its exit gate is format competence — Stage 4/5 makes
the student the data source, so below threshold the rollout corpus is mostly
parse failures. Baseline `format_ok` is **0.2237**; at that rate only ~21% of
prompts yield 2 parseable candidates of 4.

**This run is a baseline instance, not the target.** No README Optim-record
entries during baseline construction; the first record point comes after Stage 6.

### Declared capability scope (frozen 2026-07-30)

Primary target: the teacher's **reasoning, problem-solving and agent-decision**
capability under the deployment budget. Broad teacher imitation is explicitly not
the objective (AGENTS.md P3, P10.1).

| group | class |
|---|---|
| `code_math` (gsm8k, openmath), `multihop_qa` | **primary capability transfer** |
| `rag_evidence`, `tool_calling`, `instruction`, `long_context` | supporting capability |
| `short_realtime` | evaluation-only *(provisional — still in the trained mixture; moving it needs an ablation)* |
| `refusal_uncertainty` | **evaluation-only** |

In-scope slices for generation: **`rag_evidence`, `multihop_qa`, `gsm8k`,
`openmath`**. Both generation scripts default to exactly this list.

---

## 2. Current recipe

**Stage 3 recovery is one run, not a ladder** (2026-07-27):
`configs/stage3/s2v1_from_init.json` — from the Stage 1 init, 2700 steps × 16 ×
1024-token blocks on mixture v1, attention-unfrozen freeze set (440.5M
trainable), CE 0.25 + full-vocab KD 1.0 at τ=1 scope `all`, lr 2e-4 / warmup 60 /
cosine to 0.1×, fp32 master + bf16 autocast, **seed 20260726**, eval every 150
steps. The FFN-first warm-up ladder is retired.

**The teacher-target question is still open.** A four-arm run completed on
2026-07-30 but started every arm from `s2v1_from_init/step_002700`, which is
already 2,700 steps of public-target training — a path-dependent advantage for
the public arm. It is relabelled a **post-s2v1 continuation diagnostic**
([log](experiments/stage3/2026-07-30_stage3_post_s2v1_continuation_diagnostic.md));
its R2 "reject" is **void as evidence about teacher-native targets**. Both arms
regressed holdout NLL, so neither replaces the branch point. The corrected
baseline forks from the **Stage 1 structural init**
([proposal §11](proposals/stage3/2026-07-30_stage3_teacher_target_2x2.md)).

**Best checkpoint / branch point:** `stage3/s2v1_from_init/step_002700/model` —
holdout NLL **3.8285**, `behavior_score_v0` 20.2%, at 33% fewer steps than the
best-NLL run (`s2_blocks_v1`, 3.8003). Teacher ceiling on the same eval:
**74.4%**.

---

## 3. Measured results that constrain everything downstream

| finding | measurement | date |
|---|---|---|
| **CE/KD conflict is causal** | p(`</think>`) **0.2995 → 0.9989** under `kd_scope all_no_think`; seed spread collapsed 44× | 07-28 |
| **Behavior metric noise floor** | seed-only spread on `behavior_score_v0` = **0.1290**, wider than any inter-arm difference reported | 07-28 |
| **bf16 decoding is not batch-invariant** | student 1/6 identical batch-1 vs batch-6 with padding eliminated; **fp32 6/6**. Teacher 7/8 at cap 64 | 07-29 |
| **In-stack HF generation does not scale** | 37.5 / 43.9 / 39.3 tok/s at batch 2/4/8 — flat | 07-29 |
| **Both current engines ≈5.5× HF and do scale** | vLLM 0.26.0 **247.5** tok/s, SGLang 0.5.12 **241.0**; wall 57.03 s vs 56.87 s on identical work | 07-29 |
| **Policy mismatch between engines is negligible** | importance ratios median **1.000**, max 1.083, **off-policy rate 0.000**, KL ~1e-4 | 07-29 |
| **SGLang deterministic tax** | **55%** throughput loss (241.0 → 108.6), vs ~34% documented | 07-29 |
| **openmath is cap-bound but raising it fails** | cap 16384 closure 0.300 → **0.850**, accuracy among closed **0.750 → 0.294**, cost/accepted doubled | 07-29 |
| **Teacher targets would be shredded by the data path** | 48.5% exceed `block_len` 1024; `best_fit`@1024 loses **56%** of supervision; **`best_fit`@8192 is lossless** | 07-30 |
| **`block_len` 8192 needs gradient checkpointing** | peak **44,983 / 46,068 MiB (97.6%)** with it on; `best_fit` pads every block, so this is constant, not sample-dependent. ~4.3 s/step | 07-30 |
| **Continuing s2v1@2700 for 137 steps moves protocol a lot** | `format_ok` 0.250 → **0.625** (public) / 0.354 (teacher-native). **Diagnostic only** — arms forked from a public-trained checkpoint | 07-30 |
| **...and costs the LM in both arms** | holdout NLL 3.8285 → 4.0622 (public) / **3.9653** (teacher-native). Same caveat | 07-30 |
| **`p(</think>)` is inverted for teacher-native work** | measured where the *public* render demands it, so it scores "skip reasoning entirely" | 07-30 |
| **The 512-token scorecard was saturated for one arm** | treatment hit the cap on **84.2%** of prompts vs 42%/24%; re-scored at 2048 | 07-30 |
| **Conditioned on finishing, the arms are near-identical** | treatment `terminated` **1.000**, `format_ok` 0.909; the gap is *finishing less often*, not malformed output | 07-30 |
| **The control wins protocol partly by being terse** | median finished answer **2 words** (control) vs 34 (teacher-native) | 07-30 |
| **The corpus is effectively n=1** | 92.7% byte-identical pairs; a serving engine seeds per *request*, so the draws were never independent. accept@n == accept@1 by construction. **Not** evidence about sampling diversity | 07-30 |
| **A start point trained on one arm's targets invalidates the fork** | `s2v1_from_init@2700` is 2,700 public-target steps; forking both arms there compares target sets *conditioned on one of them* | 07-30 |

**The two that most changed the plan:**

- **Token divergence is not policy divergence.** Engines agree on **0/8** greedy
  tokens yet have off-policy rate **0.000**. Greedy takes an argmax, so a
  one-in-a-thousand logit difference flips a near-tied token while the
  distributions stay nearly identical. Gating on token identity would have traded
  5.5× throughput for a KL of 0.0001.
- **Trace fragmentation is a live blocker** for the teacher-target warm-up, and
  the naive fix is worse than the problem (§6).

---

## 4. Current decisions in force

| decision | date |
|---|---|
| Stage 3 recovery is single-stage from the Stage 1 init | 07-27 |
| `behavior_score_v0` is the headline; holdout NLL is a ±1% guard rail | 07-28 |
| Behavior comparisons need **≥2 seeds per arm**; more prompts do not substitute | 07-28 |
| Metrics chosen by resolving power, not stage number; every stage carries a targeted probe | 07-28 |
| The teacher is never forced out of thinking mode — no prefill, no suppression | 07-28 |
| "On-policy" = training **states come from the student**; Stage 3 owns all teacher-generated data | 07-28 |
| Answer generation samples **untruncated** (temp 1.0 / top_p 1.0 / top_k off), **no greedy candidate** | 07-29 |
| `verify.select` takes the median-length accepted candidate; candidate 0 is not privileged | 07-29 |
| `refusal_uncertainty` is **evaluation-only** — capability scope, *not* answer length | 07-30 |
| openmath generation cap stays **4096** | 07-30 |
| Exact token agreement is **not** an engine adoption gate; HF `generate` retired as production rollout path | 07-30 |
| Engines are benchmarked in their **own official images** on supported hosts; one pod per engine | 07-30 |

---

## 5. Open hypotheses (not results)

- **Teacher-native targets fix protocol competence.** The direct test is the 2×2
  in §8. Grounded in the twice-confirmed CE/KD conflict, unmeasured as a training
  intervention.
- **A higher openmath cap could pay off with better prompt selection.** Rejected
  as a blanket setting; the mechanism (hard problems produce long wrong answers)
  argues against retrying it as-is.
- **Batched evaluation carries an unquantified batch-composition term.** If the
  4B teacher's non-invariance holds at eval batch sizes, every behavior scorecard
  has a second variance source beside the 0.1290 seed floor. Eval batching has
  never been varied.
- **`short_realtime` may not need to be a training slice.** Classified
  evaluation-only provisionally; it is still in the trained mixture.

---

## 6. Resolved: the packing blocker

**Resolved 2026-07-30.** `best_fit` @ `block_len` 8192 was used for both arms
and was lossless on the real 540-prompt corpus (`truncated_samples` 0, packing
efficiency 0.895 control / 0.959 treatment). It needs gradient checkpointing —
97.6% of an L40S. Original finding below.
([preflight](experiments/stage3/2026-07-30_stage3_target_preflight.md))

Teacher targets are **4.2× longer** than public ones (rendered p50 997 vs 245,
max 5193) with **5.2× the supervised span**. Against the current recipe
(`concat` @ `block_len` 1024):

- **48.5%** exceed a whole block; expected split rate **79.9%**;
- split traces train the student to continue premises it cannot see — no crash,
  no log;
- **the naive fix is worse**: `best_fit` @ 1024 silently discards **56%** of
  supervised tokens (18,161 of 41,276);
- **proven lossless: `best_fit` @ `block_len` 8192** — `truncated_samples = 0`,
  41,276/41,276 supervised tokens preserved;
- **bounded by construction**: worst prompt 2,765 + cap 4,096 + overhead ≈
  **6,925 < 8,192**.

Both arms of any comparison must share that packing, or the experiment measures
fragmentation rather than targets.

---

## 7. Technical debt

| item | why it matters | where |
|---|---|---|
| **P3 violations in the core** — `REFUSAL_MAX_WORDS = 60`, generic `MAX_ANSWER_WORDS = 600` applied to every slice, and the fallback-to-public-target rule | Any slice added to teacher generation silently inherits a 600-word answer ceiling, which P10 forbids as a quality gate. Inert while refusal is evaluation-only | `src/aadistill/data/verify.py`, `scripts/rollout/generate_teacher_answers.py` |
| **vLLM cap-8192 cells never measured** | SGLang's cap-8192 numbers stand alone and are **not** a comparison | [log](experiments/rollout/2026-07-30_current_engine_benchmark.md) |
| **In-process vLLM / SGLang log-probs not wired** | Both raise `NotImplementedError`; only the HTTP paths supply rollout log-probs | `src/aadistill/rollout/engines.py` |
| **Server adapters' error branches verified only against CPU stubs** | Live transport was exercised in the benchmark; the failure branches were not | `tests/rollout/` |
| **`verify_and_report.py` still applies the 2026-07-28 packing rules** | Its R1 means "adopt best-fit packing"; the 2×2 uses `report_tt2x2.py` instead. Pointing the old one at a new experiment auto-generates a confident write-up of the wrong hypothesis | `scripts/pod/verify_and_report.py` |
| **Per-candidate seeding is fixed but its effect is unmeasured** | accept@n was identical to accept@1 because candidates were clones; distinct seeds are tested, the acceptance gain is not | `scripts/rollout/generate_teacher_answers.py` |
| Date drift in two filenames | Written as `2026-07-31` from a UTC/local mix-up; renamed to `2026-07-30`. Contents carry exact UTC timestamps | logs |

---

## 8. Awaiting approval — corrected teacher-target baseline

**Nothing is running or billing. No paid work until the maintainer approves the
budget in §8.3.**

Spent 2026-07-30: **$4.87** of a $7.00 ceiling, on the corpus (still valid) and
the run now relabelled a diagnostic (start point invalid).

### 8.1 What is settled

* **Corpus stands** — 540 accepted targets, hashed, reusable as-is; recorded as
  **effectively n=1** ([log](experiments/stage3/2026-07-30_teacher_corpus_750.md)).
  Not regenerated (maintainer direction).
* **Arms stand** — 487 train / 53 val, identical prompt sets and split
  membership, lossless at `best_fit`@8192.
* **Start point pinned** — `artifacts/stage1/qwen3_0p6b_init_v0/checkpoint`,
  `model.safetensors` sha256 `86fbba78…`, verified identical local vs relay.
* **Optimizer/scheduler reset is automatic** when launching from `student_path`
  without `--resume` (verified in code; proposal §11.3).

### 8.2 Blocked, by maintainer direction

No LR sweep, no step-budget ablation, no metric redesign, no final-only or
trace-length variants, no new corpus, no corpus regeneration — **until the
corrected baseline exists**.

### 8.3 The one decision needed: step budget from a cold start

The corrected baseline forks from the Stage 1 init (holdout NLL **11.748**), not
from a checkpoint at 3.83. The registered per-arm parity rule is unchanged
(equal total training tokens); the absolute budget is not fixed by the
pre-registration and is the maintainer's call.

| option | steps | tokens | treatment passes | control passes |
|---|---:|---:|---:|---:|
| parity with the diagnostic | 137 | 2.24M | 3.0 | 7.6 |
| **1× corpus-limited** | 273 | 4.47M | 6.0 | 15.2 |
| deeper | 546 | 8.94M | 12.0 | 30.3 |

**The tension to settle:** the reference run needed 2,700 steps / 44.2M tokens to
take this init from 11.75 → 3.83 on a 22.1M-token mixture. This corpus holds
0.71M real tokens (treatment) / 0.26M (control), so *no* budget on it produces a
converged model — more steps buy repetition, not coverage. Both arms will be far
from convergence at any of the above, and the control repeats 2.5× more often
than the treatment at equal tokens.

### 8.4 The confound that survives the correction

At equal total tokens the treatment arm gets **18.9×** the supervised tokens
(519,478 vs 27,526 per epoch), because a public target for these slices is a few
tokens and a teacher-native one is a full trace. From a common cold init this
matters *more*, not less: an arm with 18.9× the gradient signal should win on
almost any axis. A treatment win therefore establishes "teacher-native beats the
best available public alternative at equal compute" and **not** "traces beat
short answers per supervised token". Separating those needs a third arm and is
out of scope until this baseline exists.

## 9. Superseded conclusions — do not act on these

Kept because the underlying measurements stand; the *conclusions* do not.

| retired conclusion | why | replacement |
|---|---|---|
| "vLLM is incompatible with this project's stack" | Environment-selection failure: host ran driver 570 / CUDA 12.8 against an engine targeting CUDA 13. **`--min-cuda-version 13.0` was the whole fix** | Both engines run on L40S |
| "vLLM 0.11.0 is the first measured engine" (5.29×, $2.27/1k) | A build reached by pinning *backwards* until the engine fit the training image — not a measurement of vLLM | vLLM **0.26.0** measured natively |
| "Stage 3 → vLLM, Stage 4/5 → HF in-stack only" | Rested on exact token agreement as a gate, and picked a permanent backend from one measured alternative | Selection reopened; both engines viable |
| "Refusal is excluded because the teacher's answers are longer" | Length is not a valid reason to reject a target (P10) | Excluded on **capability scope / alignment tax** |
| "`s2v1_from_init` is best-behaved" | One seed per arm, below the 0.1290 noise floor | Only the **NLL** half of that ablation stands |
| "`best_fit` packing is rejected" (07-28) | Measured on *public* targets where splitting is a rounding error | Superseded **for the teacher-target experiment only** |

---

## 10. Environment and operational lessons

- **Dev box:** CPU-only, 16 threads, 30 GB RAM. `uv sync` → Python 3.14, torch
  2.13.0+cpu, transformers 5.13.1. **Always run through `uv run`** — a
  `VIRTUAL_ENV` from a different project may be active and is incompatible.
- **Tests:** **222 passing** on the dev box.
- **GPU:** RunPod, runpodctl 2.7.1. **Always pass `--min-cuda-version`** for
  engine work — omitting it is what produced the retired "vLLM is incompatible"
  conclusion.
- For third-party engine images, override the entrypoint via a **template**
  (`--docker-entrypoint`); `--docker-args` alone is appended as *arguments to the
  image's entrypoint*, which crash-loops the container.
- **Never** combine a `pkill` and a launch in one SSH invocation — `pkill -f`
  matches the remote shell's own command line. Kill by PID. (Cost time three
  times.)
- **HF relay:** `AlphaAvatar/aadistill-artifacts` (private). See
  [`artifact_manifests.md`](artifact_manifests.md).

## 11. Layout and indexes

- [`docs/REPO_LAYOUT.md`](../docs/REPO_LAYOUT.md) — where everything belongs
- [`logs/indexes/EXPERIMENTS.md`](indexes/EXPERIMENTS.md) ·
  [`logs/indexes/PROPOSALS.md`](indexes/PROPOSALS.md)
- [`logs/decisions.md`](decisions.md) ·
  [`logs/supported_models.md`](supported_models.md) ·
  [`logs/artifact_manifests.md`](artifact_manifests.md)
