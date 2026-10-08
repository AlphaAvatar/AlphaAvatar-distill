"""The shared application layer, and the stage-neutral script capabilities.

Modules at this level (`shared.calibration`, `shared.run_layout`, …) are the
application layer the experiment packages build on — the calibration and
dataset registries, the recipes, the recovery policy, the preflight harness,
the deployment relay, the cost model, the run-layout convention. They are used
across stages, so they belong to no stage, and they are not reusable core
either: by P3 they are experiment-instance code, which is why they live here
rather than under `src/aadistill/`.

Subpackages are the stage-neutral CLI capabilities (`shared/data`,
`shared/evaluation`, `shared/training`, `shared/rollout`, `shared/validation`,
`shared/pod`, `shared/preflight`). A script that serves one experiment does
not live here — it lives with its experiment under `scripts/stages/`.
Cross-experiment is not the bar; stage-neutral is (AGENTS.md P3, and the
migration note at `scripts/maintenance/migration/README.md`).
"""
