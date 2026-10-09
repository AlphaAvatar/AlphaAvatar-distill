# scripts/shared/ — the stage-neutral layer

Two kinds of content, one bar for admission: **stage-neutral is the test,
cross-experiment is not** (AGENTS.md P3). A script two experiments call still
belongs to the stage that owns the question it answers.

**Flat modules** are the shared application layer, importable as `shared.<m>`:
the calibration and dataset registries, the recipes, the recovery policy, the
preflight harness, the deployment relay, the cost model, the run-layout
convention (`shared.run_layout`, which also owns `resolve_historical` — the
old-path → current-path lookup over `logs/index.json :: historical_paths.map`).

**Subpackages** are the CLI capabilities:

| dir | holds |
| --- | --- |
| `data/` | corpus/battery builders and audits that serve no single experiment |
| `evaluation/` | scorers and evaluation tooling (`score_recovery_search.py` is the frozen recovery-search scorer) |
| `training/` | the stage pipelines (`collect_stage0`, `init_stage1`, `train_stage3`) and generic training audits |
| `rollout/` | teacher generation, recovery-corpus building, leaf transport |
| `validation/` | CPU/CUDA engineering validation kits and `$0` gates |
| `pod/` | the session infrastructure every paid session shares — setup script, watchdog, collectors, readiness recorder — catalogued in `docs/POD_SCRIPTS.md` |

`tests/` is the shared application layer's own suite:
`pytest scripts/shared/tests`.
