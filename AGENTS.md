# AlphaAvatar-distill Agent Instructions

This file is the root working contract for autonomous coding agents working on AlphaAvatar-distill.

At project bootstrap, the repository should contain only this file and `README.md`. Agents must create all code directories, logs, configs, scripts, tests, and model recipe folders only when they are needed for the next concrete milestone. Do not pre-generate empty project structure.

`AGENTS.md` is the single source of truth for agent instructions. Do not maintain a separate `agent.md`.

---

## 1. Development Principles

### P1. Keep the codebase simple and efficient

Prefer small, explicit, readable modules over clever abstractions. The project must remain understandable to future agents while still being efficient enough for serious training and inference.

Simplicity does not mean slow code. Keep the implementation easy to inspect, but design performance-critical paths carefully, including data loading, activation collection, rollout, loss computation, checkpointing, distributed execution, quantization, inference, and custom kernels.

Avoid unnecessary frameworks, hidden global state, implicit registries, magic config mutation, deeply nested inheritance, and premature plugin systems. When plain Python is enough, use plain Python. When performance matters, use well-scoped optimized implementations, fused operations, efficient tensor layouts, and custom kernels only when they are justified, tested, and documented.

Do not add abstraction layers that hide performance bottlenecks. Any optimization that changes numerics, memory layout, kernel behavior, rollout behavior, or training throughput must be documented and validated.

### P2. Let the repository grow from working milestones

Do not create directories, packages, configs, or placeholder files only because a future design may need them. Create structure only when it is required by an implemented and verified milestone.

The first minimal repository state is:

```text
AlphaAvatar-distill/
├── AGENTS.md
└── README.md
```

Everything else should be generated step by step by agents or humans as the project becomes real.

### P3. Separate algorithm core from distilled model recipes

The codebase should eventually have two conceptual areas:

1. **Algorithm core**: reusable code for model loading, teacher adapters, student initialization, activation collection, training pipeline, rollout, loss computation, evaluation, checkpointing, distributed execution, quantization, logging, and reproducibility. This may live under a source directory such as `src/` when implementation begins.
2. **Student model recipes and model cards**: model-specific code, metadata, reproducibility files, and public release documentation. Each serious student model should eventually have its own folder containing model architecture, model config, initialization recipe, stage data list, inference script or engine adapter, evaluation entry, training notes, model card draft, and checkpoint metadata.

The algorithm core must not hard-code one model recipe. Model recipes may depend on the algorithm core, but the core should stay reusable.

### P4. Reproducibility is part of the result

Every experiment must be reproducible from a logged command, logged config, logged environment, and logged implementation state.

Reproducibility does not only mean reproducing model hyperparameters. It also means reproducing the exact algorithm implementation used to produce the result, including model code, kernel code, rollout logic, loss computation, training loop design, distributed execution, quantization behavior, checkpointing, and evaluation code.

A valid experiment should record:

* git commit hash;
* uncommitted diff or patch hash;
* command line;
* config hash;
* dataset manifest and data revision;
* tokenizer identity and hash;
* teacher checkpoint ID and revision;
* student config hash;
* initialization recipe hash;
* model architecture implementation version;
* kernel implementation version or source file hash;
* rollout code version and rollout sampling parameters;
* loss function implementation version and loss weights;
* training loop implementation version;
* distributed strategy and device mapping;
* quantization or fake-quantization implementation details;
* random seeds;
* deterministic flags;
* hardware and distributed topology;
* dtype and quantization mode;
* checkpoint and resume state;
* evaluation code version;
* metric logs;
* final report.

Any custom kernel, optimization trick, rollout strategy, loss function, trainer behavior, or distributed execution path must be documented well enough for another agent to reproduce, inspect, and debug it.

If an experiment cannot be reproduced from its logged code state, config, command, data, and environment, it is not a valid result.

### P5. Deterministic validation before analysis

Before claiming an optimization helped, validate that the evaluation path is stable.

For a fixed checkpoint, prompt set, decoding config, seed, and environment, the project should pursue token-in/token-out identical validation whenever possible. If exact determinism is impossible because of kernels, distributed execution, sampling, or hardware behavior, log the nondeterministic source and report variance across repeated runs.

### P6. Fixed budgets before comparisons

A comparison is only meaningful when the budget is fixed before the run.

Always record:

- wall-clock budget;
- optimizer steps;
- tokens processed;
- teacher calls allowed;
- activation cache budget;
- GPU/CPU type and count;
- batch size;
- sequence length;
- evaluation suite;
- target metric.

Do not compare methods using different compute budgets unless cost is explicitly part of the metric.

### P7. No public claims without records

Do not write that something is “better”, “faster”, “best”, “record”, “SOTA”, or “improved” unless a reproducible experiment report exists.

The README may contain empty sections and placeholders, but it must not contain invented results, fake commands, fake benchmark tables, fake project structure, or fake supported model lists.

### P8. Start cheap, scale late

Use CPU, tiny models, toy data, dry runs, static checks, and cheap GPUs before expensive training.

Expensive GPU runs are allowed only after the relevant code path, logging path, resume path, and evaluation path have been tested at small scale.

### P8.1. Hardware-portable environment setup

This repo may be run on CPU-only development servers, local GPU machines, or rented cloud GPU instances. Agents must not assume CUDA or GPU availability.

At the start of a task, the agent should inspect the current environment, infer or create the minimal Python environment needed for the next milestone, and run CPU-compatible checks, static validation, and smoke tests whenever possible.

GPU-dependent training, quantization, benchmarking, kernel validation, and large teacher inference must be gated behind explicit hardware detection such as `nvidia-smi` and `torch.cuda.is_available()`. The implementation should allow the same experiment definition to be prepared on a CPU-only machine and later resumed or executed on a GPU-enabled machine without changing the experiment logic.

If no `pyproject.toml` or dependency lockfile exists yet, the agent may create the smallest reasonable project environment only when it is required by the current milestone. Do not add heavy dependencies unless they are justified, documented, and approved when necessary.

### P9. Match training and deployment numerics when possible

If the target deployment is INT8, INT4, FP8, MXFP4, or another low-precision mode, prefer training, recovery, and evaluation paths that simulate or match deployment numerics as closely as practical.

When the target inference architecture only supports specific precisions or kernels, the student should be trained or recovered with those deployment constraints in mind. Do not assume that a model trained in BF16/FP32 will preserve the same behavior after aggressive quantization or architecture-specific inference.

Efficient training methods are encouraged when they improve the quality/cost/latency tradeoff. Candidate baseline or reference methods may include Muon-style optimizers, FP8 training, MXFP4 or other low-precision training paths, quantization-aware training, fake-quantized recovery, fused optimizers, fused kernels, activation checkpointing, efficient attention kernels, and architecture-specific inference-compatible kernels.

These methods are not considered supported until they are implemented, tested, logged, and compared under a fixed budget. Before using a precision mode, optimizer, or kernel path in an official run, the agent must record:

- target inference hardware or runtime;
- supported inference precision and kernel constraints;
- training precision and optimizer;
- quantization or fake-quantization behavior;
- expected quality/cost/latency benefit;
- numerical risks;
- baseline comparison plan;
- validation and deployment test plan.

Avoid training a student in a regime that will be heavily mismatched at inference.

### P10. Optimize for realtime agent usefulness

AlphaAvatar-distill is not only about closed-book benchmark scores.

Evaluation should eventually cover:

- RAG reading;
- evidence-grounded reasoning;
- multi-hop QA;
- tool calling;
- refusal and uncertainty;
- code and math subsets;
- self-correction;
- short realtime responses;
- quantized inference;
- streaming latency;
- memory footprint;
- integration with realtime assistant runtimes such as AlphaAvatar.

### P11. Failed experiments must remain useful

A failed run should still help the next agent. Record what was tried, what failed, why it likely failed, what logs exist, and whether to retry, modify, or discard the approach.

### P12. Human review is required for expensive or irreversible actions

Agents may autonomously code, refactor, test, create logs, run small experiments, and update documentation.

Agents must request human confirmation before:

- starting paid, long-running, or multi-GPU training;
- uploading checkpoints or model cards;
- deleting logs or artifacts;
- replacing official evaluation data;
- changing record rules;
- adding heavy dependencies;
- changing the public project mission;
- declaring an official optimization record.

### P13. Actively search for better techniques

AlphaAvatar-distill is a research-driven engineering project. Agents should actively search for relevant new techniques, papers, repositories, kernels, training recipes, and evaluation methods when they may improve the project.

Do not assume the current implementation or current plan is optimal. Before implementing a nontrivial algorithm, compression method, initialization recipe, training objective, rollout strategy, quantization path, distributed strategy, or custom kernel, the agent should consider whether there are recent or well-known references worth checking.

Prefer primary and high-signal sources:

* original papers;
* official repositories;
* technical reports;
* benchmark code;
* framework documentation;
* kernel implementation notes;
* strong open-source baselines;
* reproducible experiment writeups.

However, searching is not a substitute for engineering judgment. Do not blindly add complexity because a method is new or popular. Any borrowed technique must be adapted to this project’s goals: simplicity, reproducibility, training/inference efficiency, quantization compatibility, realtime agent usefulness, and low-cost experimentation.

When a searched or borrowed technique influences implementation, record the reference and explain:

* what was used;
* why it is relevant;
* what was changed for AlphaAvatar-distill;
* what assumptions or risks remain;
* how it will be validated against baselines.

### P14. Separate plans, hypotheses, and implemented facts

Agents must clearly distinguish between planned ideas, active hypotheses, implemented features, and validated results.

Do not describe a method as supported until code, tests, logs, and reproducible usage exist. Do not describe a result as improved until it is backed by an experiment record.

README may contain empty sections and future-facing notes, but public-facing claims must be grounded in implemented and validated project state.

### P15. Protect secrets, licenses, and sensitive data

Agents must never commit API keys, tokens, credentials, private datasets, user data, paid service credentials, or sensitive rollout traces.

Before adding a dataset, checkpoint, teacher output corpus, or external artifact, agents must check and record license constraints, redistribution rules, and privacy risks.

If a task may expose secrets, personal data, or restricted data, ask the user before proceeding.

### P16. Maintain repository hygiene before generating artifacts

Agents must protect the repository before creating code, environments, caches, logs, datasets, checkpoints, or experiment artifacts.

At bootstrap, `.gitignore` is an allowed safety file even if the repository otherwise starts with only `AGENTS.md` and `README.md`. Creating `.gitignore` does not count as pre-generating project structure.

Before running commands that may generate local files, the agent should ensure `.gitignore` covers common local and heavy artifacts, including Python environments, caches, build outputs, editor files, logs when appropriate, datasets, model checkpoints, optimizer states, activation caches, wandb or similar tracking directories, Hugging Face caches, downloaded model weights, temporary experiment outputs, secrets, and local environment files.

Do not commit generated artifacts unless they are intentionally small, reviewable, and useful for reproducibility, such as code, configs, manifests, small metadata, small plots, documentation, model card drafts, or reproducibility records.

If the agent introduces a new artifact location, cache directory, training output path, logging backend, dataset path, or model download path, it must also check whether `.gitignore` and artifact documentation need to be updated.

Repository hygiene is part of the definition of done: the final summary should mention whether new generated files were created, whether they are tracked or ignored, and whether any large or sensitive artifacts were avoided.

---

## 2. Project Writing Requirements

### 2.1 Agent autonomy

All code in this project may be written step by step by coding agents.

For each nontrivial task, the agent should:

1. read this `AGENTS.md`;
2. inspect the current repository state;
3. inspect existing logs if logs already exist;
4. create a short implementation plan;
5. make the smallest coherent change;
6. run the smallest relevant verification;
7. update logs or README when needed;
8. summarize what changed, what passed, what failed, and what should happen next.

Do not stop after writing code. Verification and documentation are part of the implementation.

### 2.2 Directory creation policy

At bootstrap, do not create any directory except when needed by the current task.

When a new directory is needed, the agent should create the minimal structure required by the implemented feature. Avoid empty placeholder trees.

The future project may naturally grow into areas such as:

- algorithm core code;
- model recipe folders;
- scripts;
- tests;
- logs;
- docs;
- local artifacts.

However, these should appear only after the agent has a concrete reason to create them.

### 2.3 Algorithm core writing requirements

The algorithm core should contain reusable implementation for the distillation system.

It may include:

- teacher model loading;
- student model loading;
- tokenizer handling;
- activation collection;
- teacher activation PCA;
- embedding/head projection;
- sandwich initialization;
- activation-aware SVD utilities;
- FFN activation-importance neuron selection;
- depth and layer-span mapping;
- loss computation;
- training loops;
- rollout;
- on-policy preference data generation;
- distributed helpers;
- quantization-aware or fake-quantized paths;
- evaluation harness;
- checkpoint and resume;
- structured logging.
- efficient optimizer experiments such as Muon-style optimizers when justified;
- FP8, MXFP4, or other low-precision training paths when supported by the target hardware/runtime;
- quantization-aware training and fake-quantized recovery;
- fused optimizers, fused kernels, and efficient tensor layouts;
- inference-compatible precision and kernel validation;

Requirements:

- keep model-specific constants out of the core;
- make execution config-driven;
- fail loudly when a requested model, dtype, dataset, kernel, or checkpoint is missing;
- support dry runs before expensive runs;
- support checkpoint resume;
- prefer explicit function boundaries;
- add tests for shape correctness, loss correctness, deterministic behavior, and resume behavior when those components are implemented.
- do not add a new optimizer, low-precision mode, or custom kernel without a baseline comparison plan;
- do not use a precision mode that the target inference architecture cannot support unless it is only an explicitly logged ablation;
- document numerical behavior, hardware support, fallback behavior, and validation coverage for every low-precision or custom-kernel path;
- prefer efficient training methods only when they preserve reproducibility, debuggability, and deployment compatibility.

### 2.4 Student model recipe and model card requirements

Each serious distilled student model should eventually have its own model recipe folder.

A model recipe is the internal reproducibility package for building, training, evaluating, and releasing a student model. It is not the same as a model card.

A model recipe may contain:

- model architecture file, if custom;
- model config;
- teacher and student metadata;
- tokenizer compatibility notes;
- initialization recipe;
- dataset list for each training stage;
- training stage config;
- inference script or engine adapter;
- evaluation script;
- local model README;
- local agent notes if useful;
- Hugging Face model card draft;
- checkpoint manifest.

The model card is the public-facing release document derived from the recipe. It should summarize what the model is, how it was trained, what data and checkpoints were used, evaluation results, intended use, limitations, license constraints, and citation information.

A model recipe must be understandable without reading chat history. A model card must not claim a released model exists until deployment validation and artifact publication are complete.

### 2.5 Model checkpoint policy

Never commit model checkpoints, optimizer states, activation caches, or large datasets to GitHub.

GitHub may contain:

- code;
- configs;
- manifests;
- small metadata;
- logs;
- small plots;
- model cards;
- reproducibility records.

Large artifacts should be stored outside GitHub, such as Hugging Face or another artifact store.

Every external artifact should eventually have a manifest containing:

- artifact ID or URL;
- revision;
- size;
- hash;
- license;
- creation command;
- related experiment log.

### 2.6 Local agent instruction files

The root `AGENTS.md` is the only required instruction file at bootstrap.

After the repository grows, agents may create additional local `AGENTS.md` files only when they reduce ambiguity for future work. Good places include a nontrivial algorithm module, a serious model recipe folder, a complex experiment folder, or a safety-sensitive scripts folder.

A local `AGENTS.md` may include:

- directory purpose;
- owned files;
- allowed changes;
- forbidden changes;
- local commands;
- current status;
- notes for future agents;
- links to detailed logs.

Do not duplicate the full root `AGENTS.md` inside local files.

### 2.7 README writing requirements

`README.md` is agent-editable, but it must not contain fake content.

The README should strictly keep the project-level structure requested by the maintainer:

1. model performance trend and project goal;
2. How it works;
3. Quick start;
4. Running the agent;
5. Project structure;
6. stage-wise Optim record history;
7. References;
8. Citation.

Empty sections are allowed. Placeholder text is allowed only when it clearly says the content is not available yet.

Do not add fake installation commands, fake repository structure, fake supported models, fake metrics, or fake records.

### 2.8 Verification requirements

Agents should run the smallest available check first.

Before code exists, verification may be limited to:

- Markdown formatting review;
- link review;
- consistency review;
- ensuring no fake results or generated structure were added.

After code exists, agents should add and run appropriate checks, such as:

- linting;
- unit tests;
- smoke tests;
- deterministic output tests;
- shape tests;
- loss tests;
- checkpoint resume tests.

If a check cannot run because the code, dependency, data, or hardware does not exist yet, document that explicitly.

### 2.9 Definition of done

A task is done only when:

- the requested change is implemented;
- no unnecessary files or directories were created;
- no fake benchmark, fake command, fake structure, or fake supported model was added;
- relevant verification was run or explicitly documented as unavailable;
- logs are updated if logs exist and the change affects project state;
- README is updated if the change affects public project information;
- the next useful action is clear.

---

## 3. Project Development and Maintenance Log Requirements

### 3.1 Logs are project memory

Logs are the shared memory for humans and agents. Once logs exist, agents should use them instead of relying on chat history.

The log system should let a new agent quickly answer:

- What is the current project status?
- What files exist and why?
- What models are supported?
- What experiments have been run?
- What stage is each model in?
- What failed recently?
- What is the current best reproducible result?
- What should be tried next?

### 3.2 Log directory creation policy

Do not create a log directory at bootstrap unless the current task needs it.

When the first meaningful code or experiment session begins, the agent should create a minimal log area.

The log area may eventually contain:

- current project state;
- supported model list;
- decision records;
- experiment logs;
- training logs;
- optimization record logs;
- artifact manifests.

Create only the logs needed by the current milestone.

### 3.3 Current state log

When implemented, the current state log should answer:

- current project status;
- latest working command, if any;
- existing directories and their purpose;
- active model recipe, if any;
- active training stage, if any;
- latest successful verification;
- latest failure;
- next recommended task.

Update it after meaningful coding, experiment, or structure changes.

### 3.4 Supported models log

When model recipes exist, maintain a supported model log.

It should include:

| Model | Teacher | Student target | Status | Stages passed | Best checkpoint | Notes |
| --- | --- | --- | --- | --- | --- | --- |

Allowed status values should include:

- planned;
- scaffolded;
- init-ready;
- stage0-running;
- stage0-passed;
- stage1-passed;
- stage2-passed;
- stage3-passed;
- stage4-passed;
- stage5-passed;
- stage6-passed;
- released;
- paused;
- failed.

Do not list unsupported or imaginary models as supported.

### 3.5 Decision records

Important technical decisions should be logged.

Decision records should include:

```markdown
## YYYY-MM-DD — Decision title

- Context:
- Decision:
- Alternatives considered:
- Expected upside:
- Risks:
- Revisit when:
```

Use decision records for:

- architecture changes;
- benchmark rules;
- data mixture changes;
- initialization recipe changes;
- training stage changes;
- quantization policy changes;
- infrastructure choices;
- public release decisions.

### 3.6 Experiment logs

Every nontrivial training, initialization, evaluation, or ablation run should have an experiment log.

A useful experiment log should include:

- date;
- human or agent;
- git commit;
- objective;
- hypothesis;
- teacher;
- student;
- stage;
- hardware;
- budget;
- command;
- config;
- data manifest;
- result;
- verdict;
- next action;
- resume instructions if interrupted.

The log should be enough for another agent to reproduce, resume, or intentionally discard the run.

### 3.7 Training logs

Training logs should be append-only and machine-readable when possible.

A training event may include:

```json
{"time":"2026-07-06T12:00:00Z","step":100,"stage":"stage2_ffn_recovery","loss":1.234,"lr":0.0003,"tokens":1048576,"gpu_mem_gb":21.4}
```

Useful event categories include:

- run_start;
- config_loaded;
- dataset_loaded;
- teacher_loaded;
- student_loaded;
- init_complete;
- train_step;
- eval_start;
- eval_result;
- checkpoint_saved;
- resume_loaded;
- run_end;
- error.

Training logs should support recovery and later analysis. Do not overwrite logs from previous runs.

### 3.8 Optimization record logs

Official optimization records should be stricter than ordinary experiments.

An official record should include:

- exact commit;
- exact command;
- hardware;
- environment;
- dataset and tokenizer hashes;
- teacher checkpoint revision;
- student config hash;
- initialization recipe hash;
- training budget;
- metrics log;
- final evaluation report;
- artifact manifest;
- maintainer approval.

A result should not enter README Optim record history unless the corresponding reproducible record exists.

### 3.9 Research references and reading queue

The project should maintain a place where humans and agents can add papers, repositories, docs, blog posts, benchmark reports, kernel notes, and implementation references that may be useful for future work.

At bootstrap, this can live in the `README.md` References section. Once the project grows, agents may create a dedicated references log only when needed.

A reference entry should include:

* title;
* authors or organization;
* year or date;
* link;
* topic area;
* why it is relevant;
* whether it has been used, rejected, or is only queued for future reading;
* related project stage, if any;
* related experiment log or decision record, if any.

Suggested status values:

* `queued`;
* `reading`;
* `used`;
* `partially-used`;
* `rejected`;
* `superseded`.

Suggested topic values:

* `initialization`;
* `activation-pca`;
* `svd-compression`;
* `ffn-pruning`;
* `depth-compression`;
* `distillation`;
* `offline-data`;
* `on-policy`;
* `dpo-sdpo`;
* `rollout`;
* `quantization`;
* `kernel`;
* `distributed-training`;
* `evaluation`;
* `runtime-deployment`;
* `agent-research`.

Agents may add references when they actually use them or when a human explicitly provides them as useful reading material. Do not add large fake bibliographies. Do not cite sources that were not read or used.

When a reference directly affects code, training design, evaluation rules, or public claims, the agent should also update the relevant decision record or experiment log.

---

## 4. Training Experiment Workflow

### 4.1 Stage advancement rule

The training workflow is staged. Agents must follow the stage order below.

Do not skip stages unless there is a documented reason. A later stage may start only when one of these is true:

1. the current stage passes its validation gate;
2. the current stage has plateaued under a predefined budget;
3. the user explicitly approves moving forward with a logged reason.

A plateau means the current stage no longer improves meaningfully under the agreed budget. By default, use one of:

* no meaningful improvement across three comparable runs;
* less than 1% relative improvement in the stage target metric;
* improvement smaller than measured run-to-run variance;
* the current bottleneck clearly belongs to the next stage rather than the current stage.

Do not cherry-pick lucky runs. Official claims should report the number of runs and variance when possible.

Before starting any stage, the agent must decide and record one of:

* **Ask user first**: required for expensive, irreversible, public-facing, or ambiguous actions.
* **Act directly**: allowed for small, local, reversible, clearly scoped implementation, verification, logging, or documentation work.

Every stage must be reproducible, recorded, and resumable.

---

### 4.2 Stage 0 — Initialization warm-up data collection

Goal: collect the teacher signals needed for teacher-aware student initialization.

This stage is specifically for initialization, not for full offline distillation.

Inputs may include:

* teacher model;
* tokenizer;
* small initialization warm-up dataset;
* target student architecture draft;
* target device or latency budget;
* target dtype or quantization mode.

Collect only what is needed under a logged cache budget:

* selected residual stream hidden states;
* final hidden states;
* embedding/token frequency statistics;
* FFN intermediate activations;
* attention outputs when needed;
* top-k logits or sampled logits when needed for lm-head projection;
* prompt metadata for coverage analysis.

The initialization warm-up data should be small but diverse enough to support stable projection and structural initialization. It may include general text, instruction samples, reasoning snippets, RAG/evidence-reading examples, tool-format examples, code/math subsets, and refusal/uncertainty examples.

Default action policy:

* **Act directly** for toy data collection, fake/tiny teacher dry runs, cache schema implementation, and deterministic validation scripts.
* **Ask user first** before downloading large datasets, using paid APIs, running large teacher inference, or storing large activation caches.

Validation gate:

* dataset manifest exists;
* tokenizer identity and hash are logged;
* teacher checkpoint ID and revision are logged;
* activation cache manifest exists if a cache is created;
* cache size and sampling policy are logged;
* teacher deterministic validation passes or variance is measured;
* projection dry run can read the collected signals;
* the run can be reproduced or resumed from logged metadata.

---

### 4.3 Stage 1 — Projection and structural initialization

Goal: create a complete student checkpoint initialized from teacher structure and teacher activations instead of random weights.

Default initialization direction:

* embedding uses embedding PCA or token-frequency weighted embedding PCA;
* lm head uses a logit-preserving projection when practical;
* hidden width uses grouped teacher activation PCA projections;
* attention uses sandwich initialization of the form `P_out^T W_t P_in`;
* FFN intermediate width uses activation-importance top-k neuron selection instead of a full dense intermediate projection;
* depth compression uses teacher-span-to-student-layer mapping, with late teacher layers compressed less aggressively;
* projection matrices are used for initialization and early alignment, then removed from the final student graph unless explicitly justified.

Baselines to compare when practical:

* random student initialization;
* random projection;
* global PCA projection;
* grouped PCA projection;
* ordinary SVD for local low-rank cases;
* activation-aware SVD for local linear compression cases.

Default action policy:

* **Act directly** for implementing initialization code, toy-model shape checks, small CPU/GPU dry runs, local baseline scaffolding, and metadata logging.
* **Ask user first** before selecting an official teacher, committing to a serious student architecture, running expensive initialization, or declaring an initialization recipe as the official baseline.

Validation gate:

* initialized student checkpoint loads;
* forward pass works;
* parameter count matches the target;
* shape checks pass for embedding, lm head, attention, FFN, and norm layers;
* initialization recipe hash is logged;
* initial evaluation report exists;
* at least one baseline comparison exists, or the reason for deferring it is logged;
* checkpoint metadata exists;
* resume or reload path works.

---

### 4.4 Stage 2 — Offline warm-up data collection

Goal: collect or prepare the offline training data used after structural initialization.

This stage is separate from Stage 0. Stage 0 collects data for initialization; Stage 2 prepares the broader offline warm-up and distillation data used to teach the initialized student.

Inputs may include:

* SFT data;
* teacher-generated answers;
* teacher top-k logits or sampled logits;
* hidden/span distillation targets when needed;
* RAG/evidence-grounded reasoning data;
* tool-use formatting data;
* refusal and uncertainty data;
* code/math subsets;
* short realtime conversation data;
* long-context samples.

The data should be organized by intended training use, not mixed into an opaque blob.

Recommended data groups:

* general instruction following;
* RAG reading and evidence-grounded reasoning;
* multi-hop QA;
* tool calling and structured outputs;
* refusal, uncertainty, and safety behavior;
* code and math subsets;
* short realtime interaction;
* long-context understanding;
* quantization-sensitive calibration/eval samples.

Default action policy:

* **Act directly** for creating data manifest schemas, tiny sample manifests, validation scripts, local format converters, and deduplication checks.
* **Ask user first** before downloading large datasets, generating large teacher-output datasets, using paid teacher APIs, adding license-restricted data, or changing the official data mixture.

Validation gate:

* data manifests exist;
* dataset names, revisions, licenses, and hashes are logged where available;
* filtering and deduplication rules are logged;
* teacher-generated data includes teacher ID, revision, decoding config, and prompt template;
* data can be loaded by the intended training pipeline;
* small-batch dry run passes;
* data mixture is reproducible;
* known data risks or license constraints are recorded.

---

### 4.5 Stage 3 — Student recovery

Goal: recover the initialized student before full on-policy training.

This stage may include multiple recovery sub-stages, but they belong to the same high-level stage:

1. FFN and norm recovery;
2. attention + FFN block recovery;
3. student-forced span recovery;
4. optional full-model offline KD/SFT warm-up.

The purpose is to repair structural compression damage and reduce mismatch between teacher-forced training and real student execution.

Recovery losses may include:

* CE/SFT loss;
* top-k logit KD;
* hidden or span distillation;
* projected teacher span output reconstruction;
* cosine alignment;
* RAG/evidence-grounded reasoning loss;
* tool-use formatting loss;
* refusal and uncertainty examples;
* quantization-aware loss when target deployment is quantized.

Recommended recovery order:

1. freeze attention and recover FFN/norm when useful;
2. unfreeze attention and train compressed blocks as functional units;
3. use student-generated intermediate hidden states for student-forced span recovery;
4. train all final student parameters with offline KD/SFT once local recovery is stable.

Do not force attention-map equality when head count, hidden width, or GQA grouping differs. Prefer block output loss, projected hidden/span loss, and generation behavior.

Default action policy:

* **Act directly** for toy recovery loops, unit tests, loss implementation, checkpoint/resume tests, and small local dry runs.
* **Ask user first** before starting long-running training, using paid GPUs, changing the official recovery recipe, or declaring a recovery result as an optimization record.

Validation gate:

* recovery run is reproducible from logged command and config;
* checkpoint can resume;
* training loss and validation proxy are logged;
* no exploding activations or severe generation collapse;
* generation smoke test produces valid tokens;
* autoregressive behavior improves or failure is explained;
* latency and memory are measured when relevant;
* quantized evaluation is run if the target deployment is quantized;
* stage result is documented in experiment logs.

---

### 4.6 Stage 4 — Online data collection

Goal: collect data from the student’s own generation distribution for on-policy distillation and preference optimization.

This stage addresses the mismatch between offline teacher-forced training and real student rollouts.

Process:

1. student generates rollouts;
2. teacher, verifier, rule-based checker, reward model, or human feedback evaluates outputs;
3. positive, negative, corrected, or ranked examples are stored;
4. data is filtered, deduplicated, and assigned to an on-policy training mixture.

Online data may include:

* student answers and teacher corrections;
* preferred vs rejected responses;
* tool-use success/failure traces;
* RAG answer faithfulness checks;
* self-correction traces;
* refusal/uncertainty comparisons;
* short realtime interaction rollouts;
* code/math verifier outcomes.

Default action policy:

* **Act directly** for implementing rollout schemas, small local rollout tests, verifier interfaces, and toy preference data.
* **Ask user first** before generating large-scale rollouts, using paid teacher/verifier APIs, collecting human feedback, storing sensitive data, or changing the official online data policy.

Validation gate:

* rollout generation is reproducible or randomness is logged;
* student checkpoint ID is logged;
* teacher/verifier/reward model identity is logged;
* preference or correction schema is documented;
* filtering rules are logged;
* data quality checks pass;
* data can be loaded by the on-policy trainer;
* privacy, license, and safety risks are recorded;
* the online dataset can be traced back to source prompts and student checkpoints.

---

### 4.7 Stage 5 — On-policy Distillation

Goal: improve the student on its own generation distribution using preference, correction, or reinforcement-style objectives.

The following methods are initial baseline or reference candidates for this stage. They are not considered supported until they are implemented, tested, logged, and compared under a fixed budget:

* GKD: Generalized Knowledge Distillation;
* SDPO: Reinforcement Learning via Self-Distillation;
* TIP: Token Importance in On-Policy Distillation;
* rejection fine-tuning;
* teacher correction distillation;
* verifier-guided preference optimization;
* GRPO-style methods;
* other logged on-policy objectives.

Before implementing any candidate method, the agent should record why it is relevant, what baseline it will be compared against, what data it needs, what risks it introduces, and how it will be validated.

The objective must be explicitly recorded. Do not silently change preference loss, beta, reward normalization, sampling policy, rollout policy, data filtering, or data mixture.

Default action policy:

* **Act directly** for implementing small-scale objective tests, toy preference training, toy rollout checks, data loader checks, and reproducibility checks.
* **Ask user first** before running serious on-policy training, changing the official objective mix, using expensive GPUs/APIs, or claiming an optimization record.

Validation gate:

* on-policy training config is logged;
* objective mix is logged;
* baseline/reference method is recorded;
* rollout policy is logged;
* preference or correction dataset manifest is logged;
* checkpoint can resume;
* on-policy model improves over student recovery or gives a clearly documented tradeoff;
* tool-use format is not broken;
* refusal and uncertainty behavior is not degraded;
* RAG faithfulness does not regress severely;
* latency remains within target;
* final evaluation is reproducible;
* result is documented before any README Optim record entry is added.

---

### 4.8 Stage 6 — Deployment validation

Goal: confirm that the student works in the intended runtime, target precision, and deployment environment.

Validate:

* local inference;
* quantized inference;
* streaming token latency;
* memory footprint;
* target device behavior;
* compatibility with RAG/tool/persona/memory assumptions;
* failure behavior under uncertainty;
* checkpoint loading from external artifact storage;
* model card and license constraints.

Default action policy:

* **Act directly** for local inference tests, toy export tests, small quantization checks, and model card draft updates.
* **Ask user first** before uploading checkpoints, publishing model cards, changing release notes, declaring a release, or marking an optimization record as official.

Gate to release:

* deployment validation passes;
* model card is complete;
* checkpoint is uploaded outside GitHub;
* artifact manifest includes URL or ID, revision, size, hash, license, and creation command;
* README Optim record entry is backed by a reproducible record;
* release lineage is documented;
* user or maintainer approves the release.
