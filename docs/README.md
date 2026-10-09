# docs/ — where documentation lives

Owner first, type second — the same namespace as `scripts/`, `logs/`,
`artifacts/`, `configs/` and `data/`. A document lives with what it
describes; this page only says where to look.

| you want | go to |
| --- | --- |
| stage/experiment/family technical references | `docs/stages/stage-<n>/` — e.g. the Stage-1 autoinit program's [`AUTOINIT_REFERENCE.md`](stages/stage-1/AUTOINIT_REFERENCE.md), [`OPERATOR_PROMOTION_CYCLE.md`](stages/stage-1/OPERATOR_PROMOTION_CYCLE.md), [`POST_PHASE_B_GENERALIZATION.md`](stages/stage-1/POST_PHASE_B_GENERALIZATION.md), and the historical handoff under [`stages/stage-1/archive/`](stages/stage-1/archive/) |
| shared framework design: provider/session machinery, pod scripts | [`shared/SESSION_ARCHITECTURE.md`](shared/SESSION_ARCHITECTURE.md), [`shared/POD_SCRIPTS.md`](shared/POD_SCRIPTS.md) |
| repository layout and maintenance | [`maintenance/REPO_LAYOUT.md`](maintenance/REPO_LAYOUT.md), [`maintenance/core-provenance.md`](maintenance/core-provenance.md) |
| which owner holds which documents (machine-readable) | [`../docs/index.json`](index.json) — a generated projection of `logs/stages/index.json`; edit `stage_attribution.py`, never the projection |

Scientific facts are not duplicated here. Experiment state, results,
decisions and manifests live under `logs/` (start at `logs/state/current.md`
and `logs/stages/index.json`); docs describe architecture and link to those
canonical records. Historical paths cited by frozen records resolve through
`logs/index.json :: historical_paths.map`.
