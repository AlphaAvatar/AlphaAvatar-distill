# AlphaAvatar-distill

## 📈 Performance Trend and Project Goal

<!--
Agent notes:
- Add a model performance trend chart here only when there is real experiment data
  or a clearly labeled target curve approved by the maintainer.
- The chart should follow the spirit of autoresearch-style progress visualization.
- x-axis: experiment stage, technique stage, or optimization record.
- y-axis: agreed optimization metric.
- Every point must link to a reproducible experiment log.
- Do not add fake metrics, fake records, fake supported models, or fake commands.
- Keep this section honest: if no experiment has been run, leave the chart empty.
-->

AlphaAvatar-distill aims to build an agent-guided model compression and distillation framework for transforming large teacher models into small, real-time, edge-deployable student models.

The project goal is to make distillation reproducible, automated, and useful for realtime AI assistant runtimes, including RAG, tool use, reasoning, self-correction, quantized inference, and low-latency deployment.

---

## 🧠 How it works

<!--
Agent notes:
- Explain the actual implemented pipeline here only after implementation exists.
- Do not describe features as completed before they exist.
- Prefer a short, high-level explanation over a large fake architecture description.
- When implementation begins, update this section to reflect the real workflow.
- Link to reproducible design notes, logs, or experiment records when available.
-->

---

## ⚡ Quick start

<!--
Agent notes:
- Add commands only after they actually work.
- Do not add fake install commands, fake CLI examples, or fake training commands.
- If the project only contains README.md and AGENTS.md, this section may stay empty.
- When the first working command exists, include the minimal command needed to verify it.
- Prefer a tiny reproducible smoke test before any expensive training command.
-->

---

## 🤖 Running the agent

<!--
Agent notes:
- Explain how Codex, Claude Code, Cursor, or other agents should use AGENTS.md.
- Do not duplicate the full AGENTS.md content here.
- Keep this section focused on how to start an agent safely in this repository.
- Mention that agents should read AGENTS.md before making changes.
- Add tool-specific notes only when they are actually used by the project.
-->

---

## 🗂️ Project structure

<!--
Agent notes:
- At bootstrap, do not add a fake repository tree.
- The initial valid project structure is only README.md and AGENTS.md.
- Add real directories here only after they are actually created.
- Each listed directory should have a real purpose and should exist in the repo.
- Do not predefine future folders that have not been implemented yet.
-->

---

## 🏆 Optim record history

<!--
Agent notes:
- Only add records backed by reproducible experiment logs.
- Do not add placeholder results.
- Do not claim "best", "record", "SOTA", "faster", or "better" without logs.
- Each record should include date, method, budget, metric, result, commit, and log link.
- Keep stage histories independent so agents can track where each optimization helped.
-->

Only add records backed by reproducible experiment logs. Do not add placeholder results.

### 🧪 Stage 0 — Initialization warm-up data collection

<!--
Agent notes:
- Record optimizations related to initialization warm-up data only.
- Examples: calibration data selection, activation sampling, token-frequency stats,
  teacher hidden-state collection, cache format, deterministic teacher validation.
- Do not include offline KD data or online rollout data here.
-->

### 🧩 Stage 1 — Projection and structural initialization

<!--
Agent notes:
- Record optimizations related to teacher-to-student initialization.
- Examples: embedding PCA, logit-preserving lm-head projection, grouped hidden PCA,
  sandwich initialization, FFN activation top-k selection, layer-span mapping,
  random/global/SVD baselines.
- Every entry should link to shape checks, initial eval, and baseline comparison when available.
-->

### 📚 Stage 2 — Offline warm-up data collection

<!--
Agent notes:
- Record optimizations related to offline training data preparation.
- Examples: SFT data mixture, teacher-generated responses, top-k logits,
  RAG/evidence-grounded data, tool-use formatting data, refusal/uncertainty data,
  code/math subsets, filtering, deduplication, and data manifests.
- Do not include student rollout or online preference data here.
-->

### 🛠️ Stage 3 — Student recovery

<!--
Agent notes:
- Record optimizations related to recovering the initialized student.
- Examples: FFN/norm recovery, attention+block recovery, student-forced span recovery,
  offline KD/SFT warm-up, loss weighting, checkpoint/resume, quantization-aware recovery.
- Do not move to later stages unless recovery has passed its validation gate or plateaued.
-->

### 🔁 Stage 4 — Online data collection

<!--
Agent notes:
- Record optimizations related to collecting data from the student's own distribution.
- Examples: student rollouts, teacher corrections, verifier feedback, preference pairs,
  RAG faithfulness checks, tool-use success/failure traces, self-correction traces.
- Record the student checkpoint used to generate the online data.
-->

### 🎯 Stage 5 — On-policy distillation

<!--
Agent notes:
- Record optimizations related to on-policy training.
- Candidate baseline/reference methods may include GKD, SDPO, TIP, rejection fine-tuning,
  teacher correction distillation, verifier-guided preference optimization,
  GRPO-style objectives, and objective mixing.
- These methods are not considered supported until they are implemented, tested,
  logged, and compared under a fixed budget.
- Before adding a record, link to the experiment log, objective config, rollout policy,
  data manifest, checkpoint, evaluation report, and baseline comparison.
- Every entry should report whether tool format, refusal behavior, latency,
  and RAG faithfulness regressed.
-->

### 🚀 Stage 6 — Deployment validation

<!--
Agent notes:
- Record optimizations related to real deployment behavior.
- Examples: quantized inference, streaming latency, memory footprint, target device validation,
  runtime compatibility, checkpoint loading, model card, artifact manifest.
- A model is not released until deployment validation passes and the maintainer approves it.
-->

---

## 🔎 References

<!--
Agent notes:
- Agents may add references here when they use papers, repos, docs, blog posts,
  benchmarks, or implementation notes to support project design decisions.
- Each reference should include:
  - title
  - authors or organization
  - year
  - link
  - short note on why it is relevant
- Do not add references that were not actually used.
- Prefer primary sources when possible, such as papers, official docs, or original repos.
-->

---

## 📚 Citation

<!--
Agent notes:
- Keep this citation stable unless the project name, URL, or authorship policy changes.
- Do not add papers or unrelated references here; use the References section above.
- If a formal paper is later published, add the paper citation here while keeping the software citation if useful.
-->

If you use AlphaAvatar-distill in your research or projects, please cite it as:

```bibtex
@misc{alphaavatar_distill_2026,
  author       = {Licheng Wang and AlphaAvatar Contributors},
  title        = {AlphaAvatar-distill: Agentic Model Compression for Realtime and Edge AI Assistants},
  year         = {2026},
  url          = {https://github.com/AlphaAvatar/AlphaAvatar-distill}
}
```
