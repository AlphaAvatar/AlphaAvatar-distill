# Stage-0 calibration volume

The first Stage-0 calibration-data experiment (maintainer decision 2026-10-18,
`logs/budget/decisions.md`): does more independent, capability-aligned
calibration data stabilize the forward baseline's structural selection, and does
it improve the recovered student on unseen rule families?

The method is fixed: the forward baseline `stage1_init.d1_target_aware.scored_v1@v1`
(`configs/shared/baselines/selection.json`), with D1's full search, operator
order and calibration profile per operator free. Only the calibration data
varies.

## Layout

| path | what it holds |
| --- | --- |
| `audit/data_audit.json` | the $0 audit of `corpus_v2`, the D1 profiles, `state_eval@v1`, the 0.86M recovery rung and the historical batteries (independence, effective signal, length selection, quality, contamination) |
| `design/` | development analyses that size the formal design (dev battery solvability, floor, seed noise, resolution) |
| `plans/` | the frozen preregistration, once it exists |
| `runs/<run_id>/` | one directory per paid session: `governance/` (one-use authorization), `runtime/` (session record, launcher log), `evidence/` (small fetched records), `closeout/outcome.json` (money and provider state) |

Heavy bytes (raw generations, profile item files, checkpoints) live under
`artifacts/stages/stage-0/calibration_volume/` outside git; each run's closeout
names them with their sha256.

## Executables and data

- `scripts/stages/stage-0/calibration_volume/`: the audit, the task
  generators (`tasks/`), set builders, the battery runner and scorer, the
  governance module, launcher, drivers and closeout. Experiment tests:
  `pytest scripts/stages/stage-0/calibration_volume/tests`.
- `data/stages/stage-0/calibration_volume/tasks/`: the procedural task sets
  (dev, calibration prompts, final) and their manifests, with enforced
  family/content/prompt isolation.
- `configs/stages/stage-0/calibration_volume/`: audit inputs, session plans,
  artifact specs, the volume design and the profile document.
- Money: `logs/budget/approvals/stage0_calibration_volume_programme_grant.json`,
  booked from each run's closeout by `derive_budget.py`.
