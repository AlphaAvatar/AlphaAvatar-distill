# scripts/maintenance/ — repository tooling, not science

| dir | holds |
| --- | --- |
| `architecture/` | the run-index builder (`record_run_index.py`, which carries `logs/index.json :: historical_paths.map` forward), executable-closure snapshots (`derive_closure.py`), the ownership-view projections (`render_ownership_views.py` → `scripts/index.json`, `artifacts/index.json`), core-ownership/semantic-hardcode gates, and the source-relocation record machinery |
| `consolidation/` | the stage attribution (`stage_attribution.py` — the ownership index source), navigation rendering, budget derivation, log/checkpoint inventories, artifact retirement, and `converge_before_sweep.py` — the ordered generator chain (NOTE: it regenerates derived records IN PLACE on every run) |
| `migration/` | the 2026-10-08 information-architecture migration: design note and the declarative move map every old→new pair derives from |

Nothing here authorizes anything, and none of it is required for an
experiment to execute — it protects reproducibility and navigation
(AGENTS.md P8.2.1).
