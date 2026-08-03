# AlphaAvatar-distill

## 📈 Performance Trend and Project Goal

AlphaAvatar-distill is an agent-guided model compression and distillation framework for turning large teacher models into small, real-time, edge-deployable students.

The goal is to make distillation reproducible and automated, producing efficient student models with long-context and multi-turn comprehension, strong reasoning and self-correction, and reliable accuracy across sustained interactions and long-running agentic workloads such as AlphaAvatar—including RAG, tool use, quantized inference, and low-latency deployment.

The methods are meant to be **model-family-agnostic**: the same activation-statistics initialization, recovery training and deployment-numerics gates should apply to dense LLMs, MoE, VLM and Omni-models alike. That is a design constraint on the algorithm core, not a claim — the run below is a dense text **baseline**, and no MoE, vision or audio model has been attempted or validated ([scope decision](./logs/decisions.md)).

[![Experiment 1 recovery-data scaling](./assets/e1_scaling.svg)](./assets/e1_scaling.svg)

**Where the project stands.** Teacher-native held-out cross-entropy falls
monotonically with recovery data on both initializations and has **not saturated**
at 5.50M supervised tokens — the top of what this corpus reaches under a uniform
mixture. Natural termination on uncapped generation rose from **0/8** on every
earlier checkpoint to **0.93**, so the degeneration that blocked this line since
2026-07-30 is substantially resolved. What has *not* moved is reasoning: GSM8K
exact match is ≤0.05 across all 25 checkpoints. Full numbers, variance analysis
and the data-vs-compute control are in [Experiment 1](#-experiment-1--recovery-data-scaling).

**Current experiment:** [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) → a 0.6B-class student (Qwen3-0.6B geometry, ~6.7× compression, INT8 deployment target).

<details>
<summary><b>Historical run table (superseded metric — click to expand)</b></summary>

The runs below were scored on `behavior_score_v0` under a **512-token generation
cap**, which AGENTS.md P18 now forbids for formal measurement — the cap was
hiding repetition loops rather than measuring behaviour. They are kept as the
record of how the recovery recipe was built, and are **not comparable
point-for-point** with the uncapped numbers above. The size-vs-behaviour figure
they feed is [`assets/performance_trend.svg`](./assets/performance_trend.svg).

**What is measured, and why it changed.** The headline metric is `behavior_score_v0`: the unweighted mean of six *credited* mechanical checks over 76 held-out prompts — chat-format validity, fluency (an empty or copied answer scores zero), evidence grounding, refusal on unanswerable questions, parseable tool calls, and gsm8k exact match. No LLM judge, so it is free and reproducible from the stored generations ([scorer](./src/aadistill/evaluation/behavior.py), [build log](./logs/EXPERIMENTS.md)). Held-out NLL used to be the headline; it is now a **guard rail** — it is fineweb-edu text and is nearly blind to the failures that actually matter here. Behavior score is itself a stopgap: **real-world test suites take over as the headline once the student is good enough to attempt them** ([decision](./logs/decisions.md)).

| # | date | run | starts from | what changed | total steps | behavior | held-out NLL |
| ---: | --- | --- | :---: | --- | ---: | ---: | ---: |
| 1 | 2026-07-14 | [init v0, recipe attempt 1](./logs/EXPERIMENTS.md) | — | early-band depth merge, unweighted projection | 0 | – | 17.7977 |
| 2 | 2026-07-14 | [init v0, fixed recipe](./logs/EXPERIMENTS.md) | — | middle-band merge, end-weighted projection | 0 | – | 11.7482 |
| 3 | 2026-07-22 | [s1 recovery](./logs/EXPERIMENTS.md) | #2 | FFN+norm, CE 0.25 + KD 1.0, 660 steps on mixture v0 | 660 | 12.9% | 4.2107 |
| 4 | 2026-07-25 | [s2 A/B arm A](./logs/EXPERIMENTS.md) | #3 | control: +660 FFN-only steps; regressed on mixture-v0 epochs 3–4 | 1320 | – | 4.2747 |
| 5 | 2026-07-25 | [s2 A/B arm B](./logs/EXPERIMENTS.md) | #3 | attention unfrozen; freeze set adopted, holdout flat (data-limited) | 1320 | – | 4.2118 |
| 6 | 2026-07-26 | [s2 on mixture v1](./logs/EXPERIMENTS.md) | #5 | same recipe, 4.11× data, 2700 steps; plateau broken | 4020 | 8.9% | **3.8003** |
| 7 | 2026-07-27 | [start-point ablation: from_s1](./logs/EXPERIMENTS.md) | #3 | same 2700-step leg from s1@660; the A/B arm-B leg was neutral (+0.17%) | 3360 | 9.5% | 3.8067 |
| 8 | 2026-07-27 | [start-point ablation: from_init](./logs/EXPERIMENTS.md) | #2 | single-stage from Stage 1 init, 2700 steps total; warm-up ladder unnecessary (+0.74%) | 2700 | **20.2%** | 3.8285 |

Behavior score is the headline metric. **Held-out NLL is now a guard rail (±1% band), not the target** — teacher Qwen3-4B-Thinking-2507 2.6264 · random-init 0.6B baseline 12.1286.

> **The behavior scores in this table were measured under a 512-token generation cap, and that cap was hiding a failure.** Re-measured without it, every checkpoint in *this* line degenerated into repetition. That blocker has since been **substantially resolved** by Experiment 1's teacher-native recovery data: the best arm now terminates naturally on **93.4%** of held-out prompts (`e1_r2960k_sa_pca`), against 0/8 for every checkpoint above. The table below is retained as the historical record on the superseded capped metric; the current results are in [Experiment 1](#-experiment-1--recovery-data-scaling) and use a different, uncapped protocol, so the two are **not comparable point-for-point**.

Attempts 7–8 are a fixed-budget ablation of the *start point*, not of the training leg: runs 6–8 are the three branches that ran the identical 2700-step leg at the same seed, from lineages costing 4020, 3360 and 2700 total steps. Both landed inside the pre-registered 1% band, so the recovery recipe dropped its two warm-up legs and became a single stage — a third less compute per iteration.

The behavior eval **reverses the ranking held-out NLL gives**: the cheapest lineage (attempt 8) is the best-behaved at 20.2%, while the best-NLL checkpoint (attempt 6) is the worst at 8.9% — it improves next-token prediction while getting *worse* at chat format, grounding and tool calls. That result is why the headline metric changed. Caveats a reader must carry: one run per arm, no variance estimate, and the per-axis rates rest on 7–76 prompts each, so only large moves are evidence — see the [run log](./logs/EXPERIMENTS.md). Current state and next actions: [`logs/STATE.md`](./logs/STATE.md); costed, unapproved work: [`logs/PROPOSAL.md`](./logs/PROPOSAL.md).

The figure regenerates from [`assets/perf_trend.json`](./assets/perf_trend.json) with `uv run python scripts/evaluation/plot_perf_trend.py`; the table above comes from the same file via `--print-table`, and every point is backed by the consolidated record in [`logs/EXPERIMENTS.md`](./logs/EXPERIMENTS.md).

</details>


---

## 🔬 Experiment 1 — recovery-data scaling

**Question:** how much teacher-generated recovery data does this student need, and does the answer depend on how it was initialized?

**Design.** Six nested data rungs (0.25M → 5.50M supervised tokens) × 2 seeds × 2 initializations = 24 arms, each trained for a fixed **3 passes** over its own rung, plus a **step-matched compute control** — the 0.25M rung run for the 5.50M arm's full 4,412 optimizer steps, so data quantity can be separated from optimizer updates. Every arm differs from the canonical recipe in exactly four fields (rung, seed, start checkpoint, derived schedule).

| supervised tokens | PCA CE | random CE | PCA behaviour | PCA natural-termination | PCA GSM8K EM |
|---:|---:|---:|---:|---:|---:|
| 0.25M | 2.1183 | 8.8263 | 0.3194 | 0.533 | 0.005 |
| 0.46M | 1.7544 | 8.3461 | 0.3626 | 0.671 | 0.000 |
| 0.86M | 1.5069 | 7.9472 | 0.3695 | 0.763 | 0.000 |
| 1.60M | 1.2983 | 7.4047 | 0.4076 | 0.835 | 0.040 |
| 2.96M | 1.1468 | 6.6727 | 0.4180 | **0.921** | 0.015 |
| **5.50M** | **1.0042** | **5.9798** | 0.3781 | 0.803 | 0.020 |

CE is cross-entropy on held-out **teacher-native** sessions (16 packed blocks disjoint from every rung). Behaviour, natural termination and GSM8K EM come from uncapped generation within the model's **effective context of 8,192**, derived from the trained `block_len` — not the architectural 262,144 the geometry inherits. Seed-averaged; per-arm values in [`logs/EXPERIMENTS.md`](./logs/EXPERIMENTS.md).

**What the variance analysis supports.** Between-seed spread is compared against the range each metric covers, so claims are graded rather than asserted:

- **CE scales with data decisively** — 74× (PCA) and 261× (random) the between-seed noise, monotone at every rung, and **not saturated at 5.50M**.
- **Initialization dominates data over this range.** At the top rung the PCA init reaches CE 1.0042 against the random init's 5.9798. Every random-init arm sits at p50 = 768 generated tokens, the signature of the degeneration stop firing.
- **The improvement is data, not compute.** The step-matched control — same init, seed, optimizer, LR schedule, step count and validation set, differing *only* in unique data — reached CE 2.2907, **worse** than the same rung at 324 steps (2.0938) and far from the 5.50M arm's 1.0032. More passes over a small corpus overfit; more unique data did not.
- **What passes *do* buy is protocol competence:** the control leads every arm on natural termination (0.961) and behaviour (0.468).
- **The behaviour composite barely resolves** — only 3.3× the seed spread, so its rung ordering is not claimable. Held-out NLL is weaker still (between-seed |Δ| 0.66).
- **No reasoning emerged anywhere.** GSM8K exact match across all 25 checkpoints: min 0.000, max 0.050, mean 0.006, with the random init at 0.000 at every rung and seed.

The figure above regenerates from `artifacts/stage3/e1_consolidated.json` with `uv run python scripts/evaluation/plot_e1_scaling.py`.

**Reviewable samples:** [`logs/e1_test_cases.md`](./logs/e1_test_cases.md) — 46 generations stratified over stop reason and answer correctness, with the untruncated copies in `logs/e1_test_cases.jsonl`.

**Cost:** $61.5 — 24 training arms $47.6, control + first evaluation $8.1, full sweep $5.8. Every pod was released after hash-verified teardown.
---

## 🧠 How it works

| Stage | What it produces | Status |
| --- | --- | --- |
| **0** — activation statistics | streaming float64 sufficient statistics from the teacher: per residual point count / sum / `XᵀX`, per-FFN-neuron `Σ\|a\|` and `Σa²`, token frequencies. Fixed 1.95 GB cache regardless of token count. | passed ([log](./logs/EXPERIMENTS.md)) |
| **1** — projection + sandwich init | a complete, runnable Qwen3-format 0.6B student (596M params) plus a same-geometry random baseline, both with reproducibility manifests. | passed ([log](./logs/EXPERIMENTS.md)) |
| **2** — offline warm-up mixture | eight training-use groups from permissive revision-pinned sources (instruction, RAG/evidence, multi-hop QA, tool calling, refusal/uncertainty, code/math, short realtime, long context) with global dedup, holdout exclusion, and train/val/calib splits. | v0 5.39M tokens ([log](./logs/EXPERIMENTS.md)), v1 22.13M ([log](./logs/EXPERIMENTS.md)) |
| **3** — student recovery | one config-driven trainer for all recovery sub-stages: regex freeze policy, masked CE + on-the-fly full-vocab teacher KD, exact resume, per-run manifests, gate evals. Plus the recovery corpus builder (teacher generation at the model's official preset, session rendering, system-grouped packing, nested token ladder) and an uncapped vLLM evaluation harness with a semantic degeneration detector. | **Experiment 1 complete**: 24-arm data-scaling matrix + a step-matched compute control, all 25 checkpoints measured on four metrics ([results](./logs/EXPERIMENTS.md)). Natural termination reaches 0.934; reasoning has not emerged at any rung. Not yet exitable — see the exit gate below |
| **4–6** — online data, on-policy distillation, deployment validation | specified in [`AGENTS.md`](./AGENTS.md) | not started |

Design choices worth knowing:

- **Sufficient statistics, not activation dumps.** Stage 0 caches exactly what Stage 1 consumes (second moments, neuron importances, token frequencies) in float64, so the cache is O(1) in token count and the centering step stays numerically sound.
- **One global projection, transplanted sandwich-style.** Every teacher linear becomes `Pᵀ·W·P` with the preceding RMSNorm folded in exactly; Q heads are subsampled per GQA group, FFN neurons kept by activation importance, depth compressed by merging middle-band layer pairs. Attempt 1 in the table above is what happened when the merge band was wrong — the recipe is evidence-driven, not assumed.
- **Loss masks are computed from character offsets.** The Thinking-2507 chat template is *not* prefix-stable (it injects an empty think block into the final assistant turn), so the usual per-turn prefix diffing miscounts spans. The loader renders the conversation once and maps assistant character spans to tokens.
- **KD runs on the fly over the full vocabulary.** The teacher forwards the same packed blocks each step — no cached logits, so the corpus is not welded to one teacher revision and top-k approximation is unnecessary.
- **Block order is a pure function of (seed, epoch).** An interrupted run resumes bitwise-exactly. The validation subset is seed-derived too, which is why runs meant to be compared must share a seed ([decision record](./logs/decisions.md)).
- **Deployment numerics are a gate, not an afterthought.** Every recovery gate re-evaluates under INT8 weight fake-quantization at two scopes, so quality that quantization would destroy never counts as progress.
- **Ablations are config diffs.** Compared arms differ from their reference run in exactly one meaningful field, verified by diff — the start-point ablation and the packing control were both run this way.
- **Teacher sessions are rendered one at a time, then concatenated as tokens.** The official Thinking-2507 template renders `<think>…</think>` only for the assistant turn after the *last* user message, so applying it to a multi-session message list silently deletes every earlier reasoning trace. Measured directly, then designed around: each session renders alone, the system block is emitted once, and the token-level concatenation is asserted exact.
- **The system prompt is a hard packing boundary, and it is expensive.** Tool schemas render into the system block, so tool sessions almost never share a packed block: at the top ladder rung they supply 15% of the supervision and consume 72% of the blocks (efficiency 0.092 against 0.985 elsewhere). The rule was kept and the padding paid for, rather than relaxed ([decision](./logs/decisions.md)).
- **The core is kept family-agnostic, and is explicit about where it currently isn't.** Stage 0 hooks `mlp.down_proj` and fails loudly on anything else; Stage 1 assumes a dense SwiGLU MLP, GQA attention and one residual stream; the loader and behavior scorers are text-chat specific. Those are precisely the places an MoE, VLM or Omni port would touch — [inventoried](./logs/decisions.md) rather than abstracted away before a second family exists (P1/P2: no plugin system built on speculation).

Every run records config hash, code state, dataset/tokenizer/teacher hashes, and gate-check results; heavy artifacts stay out of git. GPU sessions run under a durable OS-level orchestrator that trains, evaluates, uploads artifacts, verifies the upload against pod-side hashes, generates the write-up, and tears the pod down unattended.

---

## ⚡ Quick start

```bash
uv sync                    # CPU torch by default; see pyproject.toml for a CUDA index
uv run pytest tests/ -q    # 301 CPU tests, no downloads
```

The implemented pipeline runs end to end on CPU (GPU optional):

```bash
# corpora (revision-pinned public sources; the jsonl files stay gitignored)
uv run python scripts/data/build_warmup_v1.py       # Stage 0/1 warm-up (~1M tokens)
uv run python scripts/data/build_holdout_v1.py      # held-out eval set
uv run python scripts/data/build_stage2_v0.py       # offline mixture v0 (5.39M train tokens)
uv run python scripts/data/build_stage2_v1.py       # offline mixture v1 (22.13M train tokens)

# Stage 0 → 1: teacher statistics (~1 h CPU; dry run with --limit 2), then init (~5 min)
uv run python scripts/training/collect_stage0.py --config configs/stage0/qwen3_4b_thinking_v1.json
uv run python scripts/training/init_stage1.py --config configs/stage1/qwen3_0p6b_from_4b_thinking.json

# gate check
uv run python scripts/evaluation/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
  --model artifacts/stage1/qwen3_0p6b_init_v0/random_baseline

# Stage 3 recovery: the canonical config, resumable through the same code path
uv run python scripts/training/train_stage3.py --config configs/stage3/recovery.json
uv run python scripts/training/train_stage3.py --config configs/stage3/recovery.json --resume

# any checkpoint, at the deployment precision (bf16 baseline + INT8 weight fake-quant)
uv run python scripts/evaluation/eval_ppl.py --data data/warmup/holdout_v1.jsonl \
  --model artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
  --fake-quant int8 --fake-quant-scope decoder

# behaviour, generated WITHOUT an artificial token budget (AGENTS.md P18)
uv run python scripts/evaluation/eval_behavior.py --unrestricted \
  --model artifacts/stage1/qwen3_0p6b_init_v0/checkpoint \
  --out artifacts/eval/step0_behavior.json
```

The Stage 3 recovery corpus is generated by the teacher, so its first step needs
a GPU; the pack, the token ladder and the gate are CPU work on its output.

```bash
# teacher generation (GPU, vLLM) — n=4 at the model's official preset,
# 8192-token end-to-end sessions. --limits sets how many prompts per type.
uv run python scripts/rollout/build_recovery_corpus.py --engine vllm --n 4 \
  --limits gsm8k=1700,openmath=900,code=1200,tool_calling=2600,rag_evidence=4100,multihop_qa=1074 \
  --out artifacts/stage3_corpus_v2

# one pass of system-grouped packing, cut into six nested token rungs (CPU)
uv run python scripts/data/build_token_ladder.py \
  --sessions artifacts/stage3_corpus_v2/sessions.jsonl \
  --mixture gsm8k=0.22,openmath=0.17,code=0.16,tool_calling=0.15,rag_evidence=0.20,multihop_qa=0.10 \
  --out artifacts/stage3_ladder_v2

# gate: template, seeds, budgets, masks, packing, nesting, loader round-trip
uv run python scripts/data/validate_corpus_gate.py \
  --corpus artifacts/stage3_corpus_v2 --packed artifacts/stage3_ladder_v2 --skip-logits
```

`configs/stage3/recovery.json` is the single recovery recipe; a run differs from
it only in `data_dir` and `schedule.total_steps` — hardware never changes the
experiment definition. Each step writes gitignored artifacts plus a full reproducibility manifest under `artifacts/` or `data/`.

---

## 🤖 Running the agent

This project is developed by autonomous coding agents (e.g. Claude Code, Codex, Cursor). [`AGENTS.md`](./AGENTS.md) is the single source of truth for agent instructions and must be read before making any change to this repository.

The first dense-model compression experiment was kicked off with this instruction:

> Hi, have a look at the AlphaAvatar-distill repo and start from the teacher model https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507. Let's kick off the first dense-model compression experiment.

Everything under `src/`, `scripts/`, and `logs/` grew from that instruction, following the staged workflow in `AGENTS.md`. The working split is: agents act directly on local, reversible, CPU-scale work, and write a **costed proposal with pre-registered decision rules** for anything paid, long-running, or public-facing — the maintainer approves per session. The active proposal lives in [`logs/PROPOSAL.md`](./logs/PROPOSAL.md); current state and next actions in [`logs/STATE.md`](./logs/STATE.md).

---

## 🗂️ Project structure

Code is grouped by **responsibility**; configs, logs and tests by **training stage**.
[`docs/REPO_LAYOUT.md`](./docs/REPO_LAYOUT.md) is the full map and the rule for where new files go.

```text
AlphaAvatar-distill/
├── AGENTS.md               # agent working contract (single source of truth)
├── docs/REPO_LAYOUT.md     # where code, configs, logs and artifacts belong
├── pyproject.toml          # uv-managed env; CPU torch index by default
├── src/aadistill/          # algorithm core — model-agnostic, config-driven
│   ├── models/             #   teacher/student loading, INT8 fake-quant
│   ├── init/               #   Stage 0/1: activation stats, projection, sandwich transplant
│   ├── data/               #   mixture loader (schema, chat render, loss masks, packing),
│   │                       #   session rendering + system-grouped packing (sessions.py),
│   │                       #   diversity, per-slice correctness rules
│   ├── training/           #   Stage 3 recovery trainer (CE+KD, freeze policy, resume)
│   ├── rollout/            #   engine adapters, in-stack generation, hashed rollout
│   │                       #   snapshots + importance-ratio diagnostics
│   ├── evaluation/         #   eval_behavior_v0 scorers and the headline metric
│   └── infrastructure/     #   env fingerprint, code-state hash, sha256 manifests
├── scripts/                # entry points, one per responsibility
│   ├── data/               #   mixture + eval-set builders · build_token_ladder ·
│   │                       #   validate_corpus_gate
│   ├── training/           #   collect_stage0 · init_stage1 · train_stage3
│   ├── evaluation/         #   eval_ppl · eval_behavior · uncapped_eval (P18, vLLM) ·
│   │                       #   degeneration · audit_prompt_rendering · exposure_report ·
│   │                       #   consolidate_e1 · build_test_cases · plot_perf_trend
│   ├── rollout/            #   teacher generation · build_recovery_corpus
│   └── pod/                #   GPU session scripts + durable orchestrator (run_env.sh)
├── configs/                # stage recipes: stage0/ · stage1/ · stage3/recovery.json
├── data/                   # corpus manifests (jsonl gitignored, rebuildable)
│   └── eval_behavior_v0/   #   76-prompt behavior set + manifest (both committed)
├── tests/                  # 301 CPU tests, mirroring the source areas
├── logs/                   # project memory — read STATE.md first
│   ├── STATE.md            #   canonical handoff: a snapshot, not an archive
│   ├── EXPERIMENTS.md      #   the consolidated record: what ran, results, cost
│   ├── PROPOSAL.md         #   the single active plan, costed, with stopping rules
│   ├── decisions.md        #   decision records (why, alternatives, risks)
│   ├── supported_models.md #   model status table
│   └── artifact_manifests.md  # artifacts stored outside git (HF), with hashes
└── assets/                 # trend data + rendered figure
```

The tree is abridged to the parts worth knowing about. New directories appear only when an implemented, verified milestone needs them — no empty placeholders. Model weights, activation caches and experiment artifacts are kept out of git (`.gitignore`); large checkpoints live in a private Hugging Face repo with hashes recorded in `logs/artifact_manifests.md`.

On 2026-07-31 the 25 per-run experiment logs and 11 per-experiment proposals were consolidated into [`logs/EXPERIMENTS.md`](./logs/EXPERIMENTS.md) and [`logs/PROPOSAL.md`](./logs/PROPOSAL.md), and ~26 GB of superseded artifacts were removed. The originals remain in git history at commit `866dac2`, and every deleted checkpoint is on the private Hugging Face relay.

---

## 🏆 Optim record history

Official records are stricter than ordinary experiments (AGENTS.md 3.8): exact commit, command, hardware, data and tokenizer hashes, budget, metric log, and maintainer approval.

**No records are being kept during baseline construction** (maintainer decision, 2026-07-28). Everything run so far — including the Stage 3 runs recorded in `logs/EXPERIMENTS.md` — is baseline work. The **first record point** will be written once the baseline is carried through Stage 6 deployment validation with satisfactory results; it will be the first entry, not a backfill. Until then every section below is intentionally empty, and the numbers in the run table above are attempts, not records.

### 🧪 Stage 0 — Initialization warm-up data collection

_No records yet._

### 🧩 Stage 1 — Projection and structural initialization

_No records yet._

### 📚 Stage 2 — Offline warm-up data collection

_No records yet._

### 🛠️ Stage 3 — Student recovery

_No records yet._

### 🔁 Stage 4 — Online data collection

_No records yet._

### 🎯 Stage 5 — On-policy distillation

_No records yet._

### 🚀 Stage 6 — Deployment validation

_No records yet._

---

## 🔎 References

| Reference | Topic | Status | Why it matters here |
| --- | --- | --- | --- |
| Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation* (Minitron), NVIDIA, 2024. [arXiv:2407.14679](https://arxiv.org/abs/2407.14679) | ffn-pruning, distillation | used | Activation-magnitude neuron/head importance for structured width pruning; establishes that pruned-before-recovery students score near-noise zero-shot and rely on distillation recovery. Informed Stage 1 FFN top-k selection and the interpretation of the init-checkpoint eval ([log](./logs/EXPERIMENTS.md)). |
| Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers*, 2024. [arXiv:2403.17887](https://arxiv.org/abs/2403.17887) | depth-compression | used | Layer-drop studies show early layers are critical and middle/late-middle layers are most redundant. Motivated moving Stage 1 depth merging from the early band to the middle band after the early-merge ablation collapsed. |
| Xia et al., *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning*, 2023. [arXiv:2310.06694](https://arxiv.org/abs/2310.06694) | svd-compression, distillation | queued | Structured pruning with mask learning + continued pre-training; candidate comparison recipe for Stage 3 recovery design. |
| Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016. [arXiv:1606.07947](https://arxiv.org/abs/1606.07947) | distillation, offline-data | queued | Training the student on the teacher's *generated* targets rather than on gold targets reweighted by the teacher. Basis of the pending teacher-generated-answer proposal, which targets the answer-style defects that survived the mixture-v1 recovery run ([proposal](./logs/EXPERIMENTS.md)). |
| Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning*, NeurIPS 2022. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) | offline-data, distillation | queued | Keep only generations whose final answer matches the reference. The correctness gate the same proposal now requires (2026-07-28 directive) is this filter applied to the *teacher*: a generated target is trained on only when it verifies against the public gold key it replaces. |
| Karpathy, *nanochat* — minimal full-stack LLM training/inference repo, 2025. [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) | kernel, distributed-training | queued | A dependency-minimal reference for how an efficient training loop is actually assembled (~8k LOC covering tokenizer → pretrain → midtrain → SFT → RL → serve). To be read **before** adding any kernel dependency here: the question it answers is which kernels earn their place in a small codebase and how they are called, which is the cheaper first move than importing a framework ([kernel plan](./logs/decisions.md)). |
| Ding et al., *Fewer Truncations Improve Language Modeling*, ICML 2024. [arXiv:2404.10830](https://arxiv.org/abs/2404.10830) | offline-data, distillation | used | Concatenate-then-cut packing silently truncates documents at every block boundary and measurably hurts grounded generation; best-fit-decreasing bin packing removes the truncations at the same token efficiency. Adopted as `best_fit_blocks`, and now **measured**: teacher-native targets exceed the current 1024-token block 48.5% of the time, and the naive fix (`best_fit` at 1024) silently discards 56% of the supervised tokens — only `best_fit` at 8192 is lossless ([preflight](./logs/EXPERIMENTS.md)). |
| Krell et al., *Efficient Sequence Packing without Cross-contamination*, Graphcore, 2021. [arXiv:2107.02027](https://arxiv.org/abs/2107.02027) | offline-data, kernel | partially-used | Formalizes packing as bin packing and pairs it with block-diagonal attention so packed samples cannot attend across each other. The packing half is adopted; the **masking half is deliberately rejected** — a deployed assistant reads a window holding several unrelated things and must attend across it, so training it to ignore irrelevant neighbours is the job rather than an artifact to mask ([decision](./logs/decisions.md)). |
| LMSYS, *Towards Deterministic Inference in SGLang and Reproducible RL Training*, 2025. [lmsys.org](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) | on-policy, rollout, kernel | partially-used | Batch-invariant kernels give **batch-invariant output** — same prompt, same tokens, regardless of how it was batched — at ~34% average slowdown (CUDA graphs recover ~2.8×), and report identical rollout responses *and* loss values across repeated RL runs. Dense models only, with **Qwen3 named**, which is this project's teacher family. Changes the 2026-07-28 in-stack argument: determinism is now a purchasable tax rather than a reason to avoid a serving engine ([survey](./logs/EXPERIMENTS.md)). **Measured here on SGLang 0.5.12: 241.0 → 108.6 tok/s, a 55% throughput loss — well above the ~34% cited**, which is why deterministic mode is treated as a per-run option rather than a default ([benchmark](./logs/EXPERIMENTS.md)). |
| DeepSeek, *nano-vLLM*, 2025. [HF blog](https://huggingface.co/blog/zamal/introduction-to-nano-vllm) | runtime-deployment | queued | ~1.2k lines, pure Python + Triton, no C++/CUDA extension, offline-inference focus with prefix caching and CUDA graphs; reported near-parity with vLLM on offline workloads. The profile that would give throughput without the dependency weight — determinism properties unknown and unverified by us. |
| Liu et al., *Defeating the Training-Inference Mismatch via FP16*, 2025. [arXiv:2510.26788](https://arxiv.org/abs/2510.26788) | on-policy, rollout, quantization | queued | Argues the RL training/inference gap is mostly a *numerics* problem, and that BF16's 7-bit mantissa is the culprit: reverting the rollout+training path to FP16 removes the mismatch and stabilizes optimization. Directly in tension with this project's BF16 training policy ([decision 2026-07-13](./logs/decisions.md)) — recorded as a **revisit trigger for Stage 4/5**, not acted on, since changing precision now would break comparability with every logged run. |
| vLLM team, *No More Train-Inference Mismatch: Bitwise Consistent On-Policy RL with vLLM and TorchTitan*, 2025. [blog.vllm.ai](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) | on-policy, rollout, kernel | queued | Achieves bitwise-identical sampler and trainer numerics by matching kernels, reporting faster convergence and higher reward. The strongest form of the guarantee this project wants, at the cost of the heaviest stack; the reference point for what "consistent" can mean. |
| Hugging Face, *Native-speed vLLM transformers modeling backend*, 2026. [huggingface.co/blog](https://huggingface.co/blog/native-speed-vllm-transformers-backend) | on-policy, rollout, runtime-deployment | queued | `vllm serve --model-impl transformers` runs vLLM's serving machinery over transformers modeling code, at or above native throughput on Qwen3 4B, so training and rollouts share one model implementation. The **upgrade path** for this project if in-stack generation stops scaling — noted with the caveat that the post makes *no* numerical-consistency claim, so equivalence would have to be measured, not assumed ([decision 2026-07-28](./logs/decisions.md)). |
| Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, ByteDance Seed, 2025. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) | on-policy, rollout, offline-data | partially-used | Rollouts are sampled **untruncated** because truncating the tail suppresses low-probability tokens at exactly the high-entropy positions where branching happens, biasing the sampled distribution away from the policy. The **no-greedy-candidate** half is standing — every candidate is sampled, with per-candidate seeds, because with a verifier downstream this is rejection sampling and diversity is what makes accept@n exceed accept@1. The **untruncated** half was reversed in practice: corpus v2 (2026-08-01) generates at the teacher's own published preset, `0.6 / 0.95 / top_k 20 / min_p 0`, so this paper's truncation-bias argument now applies to our corpus and is carried as a known deviation ([decision](./logs/decisions.md)). |
| Yuan et al., *Scaling Relationship on Learning Mathematical Reasoning with Large Language Models*, 2023. [arXiv:2308.01825](https://arxiv.org/abs/2308.01825) | offline-data | queued | Rejection-sampling fine-tuning with k samples per prompt — the source of the top-n recipe the proposal now uses (sample n candidates, keep a verified-correct one) and of its selection-bias and diversity caveats (accepted answers skew toward items the teacher finds easy). |

---

## 📚 Citation

If you use AlphaAvatar-distill in your research or projects, please cite it as:

```bibtex
@misc{alphaavatar_distill_2026,
  author       = {Licheng Wang and AlphaAvatar Contributors},
  title        = {AlphaAvatar-distill: Agentic Model Compression for Realtime and Edge AI Assistants},
  year         = {2026},
  url          = {https://github.com/AlphaAvatar/AlphaAvatar-distill}
}
```
