# scripts/ — executables, one owner each

Three trees, by ownership. The machine-readable owner map is
[`index.json`](index.json) (generated; a projection of
`logs/stages/index.json`, the ownership index).

| tree | owner |
| --- | --- |
| [`stages/`](stages/README.md) | experiment instances: `stages/stage-<n>/<experiment>/` holds that experiment's design writers, issuers, pricing, bundles, drivers, launchers and `tests/`; `stages/stage-<n>/families/<family>/` holds material owned by a group of sibling experiments (today: `stage-1/families/d_series`) |
| [`shared/`](shared/README.md) | genuinely stage-neutral: the application layer (`shared.calibration`, `shared.run_layout`, …) flat at the top, and the CLI capabilities `data/ evaluation/ training/ rollout/ validation/ pod/` |
| [`maintenance/`](maintenance/migration/README.md) | repository tooling, not science: `architecture/` (run index, closures, relocation machinery), `consolidation/` (stage attribution, navigation, budget, inventories, convergence), `migration/` (the 2026-10-08 move map) |

The same namespace maps across the three canonical trees — an owner's code,
evidence and bytes are at the same relative address under `scripts/`, `logs/`
and `artifacts/`. A frozen record naming a pre-2026-10-08 path resolves
forward through `logs/index.json :: historical_paths.map`
(`shared.run_layout.resolve_historical`).

`conftest.py` here is the bootstrap for every suite under `scripts/`
(experiment suites and `shared/tests/`); AGENTS.md §2.8a names the three
suites and how each is invoked.
