# AlphaAvatar-distill

## 🧠 Overview and Goals

AlphaAvatar-distill is an agent-guided framework for compressing large teacher
models into smaller, efficient student models. Its purpose is reproducible,
configurable model compression and distillation for real-time and edge
deployment: every experiment is driven by a logged configuration, command, code
state and data identity, so another person or agent can reproduce or continue it.

The primary capability objective is **fluid reasoning and adaptive
problem-solving**:

> Given unfamiliar problems, a student should use information supplied by the
> task or acquired through interaction to discover rules, form abstractions,
> reason deductively and compositionally, and generalize to produce correct
> answers or complete tasks.

Knowledge retention is a supporting capability, not a replacement for
reasoning ([`AGENTS.md`](./AGENTS.md) P10.2).

The framework is designed to extend across model sizes, architectures,
compression ratios and deployment environments: the reusable core in
`src/aadistill/` takes models, targets and budgets as inputs rather than
hard-coding them. The work so far is a dense text baseline, distilling
[Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
into a sub-1B student. Other model families, MoE and multimodal models are
design goals and have not been validated.

---

## 📈 Performance Trend

[![Historical Stage-3 recovery-data scaling](./assets/e1_scaling.svg)](./assets/e1_scaling.svg)

Performance is judged from recorded experimental evidence. Reasoning
correctness on unseen tasks is reported separately from rollout reliability
(whether the student produces well-formed, terminating output) and from runtime
efficiency, and none of them stands in for another.

Detailed results: the [experiment index](./logs/state/experiment_index.md), the
[current state](./logs/state/current.md) and the
[documentation](./docs/README.md).

---

## 🤖 Running the agent

This project is developed by autonomous coding agents (e.g. Claude Code, Codex, Cursor). [`AGENTS.md`](./AGENTS.md) is the single source of truth for agent instructions and must be read before making any change to this repository.

The first dense-model compression experiment was kicked off with this instruction:

> Hi, have a look at the AlphaAvatar-distill repo and start from the teacher model https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507. Let's kick off the first dense-model compression experiment.

Everything under `src/`, `scripts/`, and `logs/` grew from that instruction, following the staged workflow in `AGENTS.md`. The working split is: agents act directly on local, reversible, CPU-scale work, and write a **costed proposal with pre-registered decision rules** for anything paid, long-running, or public-facing — the maintainer approves per session. The active proposal lives in [`logs/stages/stage-3/e2/plans/PROPOSAL.md`](./logs/stages/stage-3/e2/plans/PROPOSAL.md); current state and next actions in [`logs/state/current.md`](./logs/state/current.md).

**Example prompts.** Open the repository with your agent and ask in plain language; it reads `AGENTS.md` and the experiment records before it changes anything.

**Introduce AlphaAvatar-distill**

> Read this repository and introduce AlphaAvatar-distill, including its goals, key features, architecture, and current capabilities.

**Explain how AlphaAvatar-distill works**

> Explore the repository and explain how AlphaAvatar-distill works, from teacher model compression and student initialization to distillation, recovery, evaluation, and deployment.

**Start a compression experiment**

> Hi, have a look at the AlphaAvatar-distill repo and start from the teacher model https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507. Let's kick off the first dense-model compression experiment.

**Distill another model**

> Help me start a new model distillation experiment using my own teacher model. Review the available architecture and compression options, then propose a reproducible workflow.

**Continue an existing experiment**

> Review the current project state and experiment records, then continue the authorized work using existing checkpoints and artifacts wherever possible.

---

## 🗂️ Project structure

All six trees — scripts, logs, artifacts, configs, data, docs — share one owner-first namespace: `stages/stage-<n>/<experiment>` (with `families/<family>` for experiment families), `shared/`, `maintenance/`.
[`docs/maintenance/REPO_LAYOUT.md`](./docs/maintenance/REPO_LAYOUT.md) is the full map and the rule for where new files go.

```text
AlphaAvatar-distill/
├── AGENTS.md               # agent working contract (single source of truth)
├── docs/maintenance/REPO_LAYOUT.md     # where code, configs, logs and artifacts belong
├── pyproject.toml          # uv-managed env; CPU torch index by default
├── src/aadistill/          # algorithm core — model-agnostic, config-driven
│   ├── models/             #   teacher/student loading, INT8 fake-quant
│   ├── initialization/     #   Stage 0/1 + the Teacher-Adaptive AutoInitializer:
│   │                       #   adapters, operator registry, versioned search state,
│   │                       #   Pareto beam search, calibration, recovery planning
│   ├── data/               #   mixture loader, session rendering, loss masks, packing
│   ├── training/           #   Stage-3 recovery trainer (CE+KD, freeze policy, resume)
│   ├── rollout/            #   engine adapters, hashed rollout snapshots
│   ├── evaluation/         #   usable_rollout, strict answer/protocol scorers,
│   │                       #   degeneration, general-text diagnostics
│   ├── governance/         #   authorization types, derived executable closures
│   ├── runtime/            #   run layout mechanism, pod environment, staging contract
│   └── infrastructure/     #   session spec/runner, provider control plane, budget,
│                           #   watchdog, log relay, artifact gate
├── scripts/                # executables, ONE OWNER EACH (2026-10-08 architecture)
│   ├── stages/             #   experiment-owned code: stages/stage-<n>/<experiment>/
│   │   └── stage-1/families/d_series/   # material owned by D1/D2/D3 together
│   ├── shared/             #   stage-neutral: the application layer (calibration,
│   │                       #   run_layout, recipes, …) and capabilities
│   │                       #   data/ evaluation/ training/ rollout/ validation/ pod/
│   └── maintenance/        #   repository tooling: architecture/ consolidation/
│                           #   migration/ (the declarative move map lives here)
├── configs/                # owner-first: stages/stage-<n>/<experiment>/ for
│                           # experiment-owned specs and authorizations; stage-level
│                           # registries at stages/stage-<n>/; shared/ and maintenance/
│                           # (shared/baselines/ = the selected forward baseline)
├── data/                   # owner-first: data/stages/stage-<n>/<dataset>/ — manifests
│                           # tracked, heavy files gitignored, rebuildable
│   └── eval-set/           #   permanent cross-stage evaluation sets: registry.json,
│                           #   eval-set-<n>/ items tracked and hash-verified (eval-set-1 frozen)
├── tests/                  # the CORE suite; experiment suites live with their
│                           # experiments under scripts/stages/.../tests/
├── logs/                   # project memory — read logs/README.md first
│   ├── state/              #   current.json + current.md (the live snapshot),
│   │                       #   ownership.md, artifact_manifests.md
│   ├── stages/             #   stage → experiment → run evidence; families/ for
│   │                       #   experiment-family records; index.json is the
│   │                       #   ownership index (one leg per tree, all six)
│   ├── budget/             #   ledger, decisions, consumed approvals
│   └── maintenance/        #   inventories, registries, source-relocations
├── artifacts/              # durable local products, by owner (gitignored except
│                           # README.md and the generated index.json)
├── docs/                   # durable reference, by owner — start at docs/README.md
└── assets/                 # trend data + rendered figure
```

The tree is abridged to the parts worth knowing about. New directories appear only when an implemented, verified milestone needs them — no empty placeholders. Model weights, activation caches and experiment artifacts are kept out of git (`.gitignore`); large checkpoints live in a private Hugging Face repo with hashes recorded in `logs/state/artifact_manifests.md`.

On 2026-07-31 the 25 per-run experiment logs and 11 per-experiment proposals were consolidated into [`logs/stages/stage-3/history/EXPERIMENTS.md`](./logs/stages/stage-3/history/EXPERIMENTS.md) and [`logs/stages/stage-3/e2/plans/PROPOSAL.md`](./logs/stages/stage-3/e2/plans/PROPOSAL.md), and ~26 GB of superseded artifacts were removed. The originals remain in git history at commit `866dac2`.

**Checkpoint retention is uneven and tracked.** Weights live outside git, on the
private relay or the dev box, with hashes in
[`logs/state/artifact_manifests.md`](./logs/state/artifact_manifests.md). One arm's weights
(`P0-assistant`) were discarded and are unrecoverable — recorded rather than
quietly dropped, because the loss bounds what can be re-measured.

---

## 🏆 Optim record history

Official records are stricter than ordinary experiments (AGENTS.md 3.8): exact commit, command, hardware, data and tokenizer hashes, budget, metric log, and maintainer approval.

**No records are being kept during baseline construction** (maintainer decision, 2026-07-28). Everything run so far — including every Stage 3 run in `logs/stages/stage-3/history/EXPERIMENTS.md` — is baseline work. The **first record point** will be written once the baseline is carried through Stage 6 deployment validation with satisfactory results; it will be the first entry, not a backfill. Until then every section below is intentionally empty, and every number on this page is an attempt, not a record.

This applies to the Stage 0/1 initialization result above as well: it is a strong,
reproducible finding, but it has not been through the record procedure and is not
claimed as one.

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

- Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation* (Minitron), 2024. [arXiv:2407.14679](https://arxiv.org/abs/2407.14679)
- Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers*, 2024. [arXiv:2403.17887](https://arxiv.org/abs/2403.17887)
- Xia et al., *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning*, 2023. [arXiv:2310.06694](https://arxiv.org/abs/2310.06694)
- Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016. [arXiv:1606.07947](https://arxiv.org/abs/1606.07947)
- Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning*, NeurIPS 2022. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Yuan et al., *Scaling Relationship on Learning Mathematical Reasoning with Large Language Models*, 2023. [arXiv:2308.01825](https://arxiv.org/abs/2308.01825)
- Ding et al., *Fewer Truncations Improve Language Modeling*, ICML 2024. [arXiv:2404.10830](https://arxiv.org/abs/2404.10830)
- Krell et al., *Efficient Sequence Packing without Cross-contamination*, 2021. [arXiv:2107.02027](https://arxiv.org/abs/2107.02027)
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- Liu et al., *Defeating the Training-Inference Mismatch via FP16*, 2025. [arXiv:2510.26788](https://arxiv.org/abs/2510.26788)
- LMSYS, *Towards Deterministic Inference in SGLang and Reproducible RL Training*, 2025. [lmsys.org](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- vLLM team, *No More Train-Inference Mismatch: Bitwise Consistent On-Policy RL with vLLM and TorchTitan*, 2025. [blog.vllm.ai](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html)
- Hugging Face, *Native-speed vLLM transformers modeling backend*, 2026. [huggingface.co/blog](https://huggingface.co/blog/native-speed-vllm-transformers-backend)
- DeepSeek, *nano-vLLM*, 2025. [huggingface.co/blog](https://huggingface.co/blog/zamal/introduction-to-nano-vllm)
- Karpathy, *nanochat*, 2025. [github.com/karpathy/nanochat](https://github.com/karpathy/nanochat)

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
