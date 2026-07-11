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

Only environment setup works so far — no end-to-end experiment command exists yet:

```bash
uv sync   # CPU-only torch by default; see pyproject.toml to switch to a CUDA index
```

The Stage 0 collection script (`scripts/collect_stage0.py`) exists but is not runnable yet: its config and warm-up dataset have not been created. See [`logs/STATE.md`](./logs/STATE.md) for what remains.

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
├── logs/
│   └── STATE.md                # current project state and next actions
├── scripts/
│   └── collect_stage0.py       # Stage 0 CLI (not yet runnable: needs config + data)
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

_No references have been used yet. Entries will be added here only when a paper, repository, or document actually informs project design decisions._

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
