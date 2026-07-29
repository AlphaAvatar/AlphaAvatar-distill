# Repository layout

Where things belong, and why. Kept short on purpose — if a rule here is not
load-bearing it should be deleted rather than elaborated (P1).

The organising principle is **responsibility first, training stage second**. Code
is grouped by what it does; logs and configs are grouped by the stage they serve,
because that is how anyone looks for them.

## Source — `src/aadistill/`

The reusable algorithm core. It must not hard-code a model recipe (P3).

| package | holds | notes |
|---|---|---|
| `models/` | `teacher.py`, `student.py`, `quant.py` | loading, construction, precision |
| `init/` | `collect.py`, `project.py`, `sandwich.py` | Stage 0/1 initialization: activation stats, PCA projections, sandwich init |
| `data/` | `dataset.py`, `diversity.py`, `verify.py` | mixture loading, loss masks, block packing, correctness rules |
| `training/` | `train.py` | the Stage 3 recovery trainer |
| `rollout/` | `engines.py`, `generate.py`, `snapshots.py` | engine adapters, in-stack generation, rollout snapshots + importance diagnostics |
| `evaluation/` | `behavior.py` | `eval_behavior_v0` scorers and the headline metric |
| `infrastructure/` | `env.py`, `manifest.py` | environment fingerprinting, hashing |

Import the concrete module, not a package re-export:
`from aadistill.data.dataset import encode_sample`. There are no `__init__`
re-export shims — the import path says where the code lives.

## Scripts — `scripts/`

Entry points. Every one is runnable as `uv run python scripts/<area>/<name>.py`.

| directory | holds |
|---|---|
| `data/` | mixture and eval-set builders, dry runs, corpus preflight and analysis |
| `training/` | stage execution: `collect_stage0.py`, `init_stage1.py`, `train_stage3.py` |
| `evaluation/` | `eval_ppl.py`, `eval_behavior.py`, `probe_think_close.py`, `plot_perf_trend.py` |
| `rollout/` | teacher generation, engine benchmarks, rollout scoring |
| `pod/` | GPU session orchestration — see `scripts/pod/AGENTS.md` |

## Configs — `configs/<stage>/`

One directory per stage that has configs: `stage0/`, `stage1/`, `stage3/`.
Filenames drop the redundant stage prefix (`stage3/s2v1_from_init.json`).
No directory is created for a stage that has no configs.

## Logs — `logs/`

| path | role |
|---|---|
| `STATE.md` | **canonical current handoff.** A snapshot, not an archive. Stale text is removed, not appended to |
| `decisions.md` | append-only decision records. Deliberately one file: splitting it would fragment the chronology |
| `experiments/<area>/` | what was run and what it measured. Never rewritten — superseded conclusions get a banner |
| `proposals/<area>/` | pre-registrations. Written *before* spend so rules cannot be chosen after seeing numbers (P6) |
| `indexes/` | `EXPERIMENTS.md`, `PROPOSALS.md` — find work by stage, date, status, topic |
| `supported_models.md`, `artifact_manifests.md` | model status table; external artifact manifests |

Experiment and proposal areas: `stage0`, `stage1`, `stage2`, `stage3`,
`rollout`, `evaluation`, `infrastructure`. An `evaluation` area exists because
harness work (behavior eval, INT8 eval) serves every stage and belongs to none.

## Tests — `tests/`

Mirrors the source areas where that helps navigation (`tests/data/`,
`tests/rollout/`, …). Tests that exercise a *script* rather than a module live
next to the area the script belongs to.

## Artifacts and data — gitignored

`artifacts/` and `data/*.jsonl` are local or on the HF relay, never in git
(AGENTS.md 2.5). Manifests, small metadata and plots are committed; checkpoints,
caches and datasets are not. Check `.gitignore` after adding an output path.
