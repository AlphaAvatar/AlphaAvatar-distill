# AlphaAvatar-distill

## 📈 Performance Trend and Project Goal

AlphaAvatar-distill aims to build an agent-guided model compression and distillation framework for transforming large teacher models into small, real-time, edge-deployable student models.

The project goal is to make distillation reproducible, automated, and useful for realtime AI assistant runtimes, including RAG, tool use, reasoning, self-correction, quantized inference, and low-latency deployment.

_No experiments have been run yet. A performance trend chart will be added here once reproducible experiment logs exist, with every point linked to its experiment record._

---

## 🧠 How it works

_Not available yet. This section will describe the actual implemented pipeline once implementation exists._

---

## ⚡ Quick start

```bash
uv sync   # CPU-only torch by default; see pyproject.toml to switch to a CUDA index
uv run pytest tests/ -q
```

Stage 0 (teacher activation-statistics collection) runs end to end on CPU:

```bash
uv run python scripts/collect_stage0.py --config configs/stage0_qwen3_4b_thinking.json
# dry run: add --limit 2
```

It writes a gitignored activation-stats cache and a full reproducibility manifest under `artifacts/stage0/`. Later stages are not implemented yet. See [`logs/STATE.md`](./logs/STATE.md) for current state and next actions.

---

## 🤖 Running the agent

This project is developed by autonomous coding agents (e.g. Claude Code, Codex, Cursor). [`AGENTS.md`](./AGENTS.md) is the single source of truth for agent instructions and must be read before making any change to this repository.

The first dense-model compression experiment was kicked off with this instruction to the agent:

> Hi, have a look at the AlphaAvatar-distill repo and start from the teacher model https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507. Let's kick off the first dense-model compression experiment.

Everything under `src/`, `scripts/`, and `logs/` grew from that instruction, following the staged workflow in `AGENTS.md`. Current session state and the next recommended actions live in [`logs/STATE.md`](./logs/STATE.md).

---

## 🗂️ Project structure

```text
AlphaAvatar-distill/
├── AGENTS.md                   # agent working contract (single source of truth)
├── CLAUDE.md                   # Claude Code entrypoint (points to AGENTS.md)
├── LICENSE
├── README.md
├── pyproject.toml              # uv-managed env; CPU torch index by default
├── uv.lock
├── configs/
│   └── stage0_qwen3_4b_thinking.json   # Stage 0 run config (pinned teacher revision)
├── data/
│   └── warmup/warmup_v0.jsonl          # 47 handcrafted warm-up samples, 12 categories
├── logs/
│   ├── STATE.md                # current project state and next actions
│   ├── decisions.md            # decision records
│   ├── supported_models.md     # model status table
│   └── experiments/            # per-run experiment logs
├── scripts/
│   └── collect_stage0.py       # Stage 0 CLI: teacher activation-stats collection
├── tests/
│   └── test_collect_toy.py     # CPU toy-model tests for the collector
└── src/aadistill/              # algorithm core
    ├── collect.py              # streaming activation-statistics collector
    ├── env.py                  # env fingerprint, code-state hash, determinism
    ├── manifest.py             # sha256 + JSON manifest helpers
    └── teacher.py              # teacher loading with pinned revision + identity record
```

New directories are added only when required by an implemented and verified milestone, per `AGENTS.md`. Model weights, activation caches, and experiment artifacts are kept out of git (`.gitignore`).

---

## 🏆 Optim record history

Only add records backed by reproducible experiment logs. Do not add placeholder results.

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
| Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation* (Minitron), NVIDIA, 2024. [arXiv:2407.14679](https://arxiv.org/abs/2407.14679) | ffn-pruning, distillation | used | Activation-magnitude neuron/head importance for structured width pruning; establishes that pruned-before-recovery students score near-noise zero-shot and rely on distillation recovery. Informed Stage 1 FFN top-k selection and the interpretation of the init-checkpoint eval (see 2026-07-14 Stage 1 experiment log). |
| Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers*, 2024. [arXiv:2403.17887](https://arxiv.org/abs/2403.17887) | depth-compression | used | Layer-drop studies show early layers are critical and middle/late-middle layers are most redundant. Motivated moving Stage 1 depth merging from the early band to the middle band after the early-merge ablation collapsed (single-axis ablation, 2026-07-14). |
| Xia et al., *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning*, 2023. [arXiv:2310.06694](https://arxiv.org/abs/2310.06694) | svd-compression, distillation | queued | Structured pruning with mask learning + continued pre-training; candidate comparison recipe for Stage 3 recovery design. |

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
