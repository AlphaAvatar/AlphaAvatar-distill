# artifacts/ — durable local scientific products, by owner

Everything under this tree except this README and `index.json` is local-only
bytes (gitignored). Identity does not live beside the bytes: every artifact's
digests, retention status and tombstones are in
`logs/maintenance/inventories/checkpoint_registry.json` /
`checkpoint_tombstones.json`, out-of-GitHub copies in
`logs/state/artifact_manifests.md`, and the owner-to-tree navigation in
[`index.json`](index.json) (generated; one row per logical owner).

```text
artifacts/
├── stages/stage-<n>/<experiment>/...      experiment-owned products
├── stages/stage-<n>/families/<family>/... family-owned products
│   └── stage-1/families/d_series/batteries/d_series_behavioural_v1/
├── stages/stage-<n>/...                   stage pipeline assets (init
│                                          checkpoints, calibration suites)
├── shared/                                genuinely stage-neutral instruments
│                                          and validation outputs
└── audit/                                 the runtime session-audit namespace
                                           (session_runner writes here; it is
                                           operational, not experiment-owned)
```

Adjacent namespaces, by lifecycle — an object lives in exactly one:

| namespace | meaning | loss means |
|---|---|---|
| `artifacts/` (here) | durable scientific products | reproducibility damage |
| `/home/ecs-user/aad-artifacts/` | out-of-repo durable store (big bytes, e.g. the D1 finalists under `phase_d1/`) | reproducibility damage |
| HF relay `AlphaAvatar/aadistill-artifacts` | transport/second copies | nothing by itself — a relay copy is never the durability claim |
| `.scratch/`, `/home/ecs-user/aad-scratch/` | disposable working state | nothing scientific |
| `hf_cache/`, `torch_cache/` | re-fetchable caches | re-download time |

A frozen record naming a pre-2026-10-08 path (`artifacts/stage3/...`,
`artifacts/eval/...`) stays true as written; resolve it forward through
`logs/index.json :: historical_paths.map`
(`shared.run_layout.resolve_historical`).
