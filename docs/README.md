# docs/ — where documentation lives

Owner first, type second — the same namespace as `scripts/`, `logs/`,
`artifacts/`, `configs/` and `data/`. A document lives with what it
describes; this page only says where to look.

| you want | go to |
| --- | --- |
| stage/experiment/family technical references | `docs/stages/stage-<n>/` — e.g. the Stage-1 autoinit program's [`AUTOINIT_REFERENCE.md`](stages/stage-1/AUTOINIT_REFERENCE.md), [`OPERATOR_PROMOTION_CYCLE.md`](stages/stage-1/OPERATOR_PROMOTION_CYCLE.md), [`POST_PHASE_B_GENERALIZATION.md`](stages/stage-1/POST_PHASE_B_GENERALIZATION.md), and the historical handoff under [`stages/stage-1/archive/`](stages/stage-1/archive/) |
| shared framework design: provider/session machinery, pod scripts | [`shared/SESSION_ARCHITECTURE.md`](shared/SESSION_ARCHITECTURE.md), [`shared/POD_SCRIPTS.md`](shared/POD_SCRIPTS.md) |
| the forward research design for reasoning transfer (Stage 0, 2, 4 data and capability-aligned evaluation) | [`shared/REASONING_TRANSFER_DESIGN.md`](shared/REASONING_TRANSFER_DESIGN.md) |
| the next round's facts (Stage-2 data, then Stage-3 recovery): reusable initializations, data assets, verified mechanisms, how the current students fail, open risks | [`shared/STAGE2_3_HANDOFF.md`](shared/STAGE2_3_HANDOFF.md) |
| the Stage-2/3 round's design (PROPOSED, $0): data audit, target quality contract, isolation rules, the 2.96M old-vs-new comparison, the nested ladder, cost, durability | [`stages/stage-2/reasoning_data/DESIGN.md`](stages/stage-2/reasoning_data/DESIGN.md) |
| the Stage-0 programme's starting point (2026-10-17; the programme closed 2026-10-21) | [`shared/STAGE0_HANDOFF.md`](shared/STAGE0_HANDOFF.md) |
| repository layout and maintenance | [`maintenance/REPO_LAYOUT.md`](maintenance/REPO_LAYOUT.md), [`maintenance/core-provenance.md`](maintenance/core-provenance.md) |
| which owner holds which documents (machine-readable) | [`../docs/index.json`](index.json) — a generated projection of `logs/stages/index.json`; edit `stage_attribution.py`, never the projection |

Scientific facts are not duplicated here. Experiment state, results,
decisions and manifests live under `logs/` (start at `logs/state/current.md`
and `logs/stages/index.json`); docs describe architecture and link to those
canonical records. Historical paths cited by frozen records resolve through
`logs/index.json :: historical_paths.map`.
