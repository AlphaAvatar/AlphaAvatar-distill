# Repository layout

```
AGENTS.md                  root working contract (P1-P18)
README.md                  public-facing; performance table is generated
logs/
  STATE.md                 canonical handoff — read this first
  EXPERIMENTS.md           the consolidated experiment record (what ran, results, cost)
  PROPOSAL.md              the single active plan
  decisions.md             decision records
  supported_models.md      model status table
  artifact_manifests.md    external artifact manifests
configs/
  stage0/ stage1/          collection and init configs
  stage3/recovery.json     canonical recovery config for the scaling study
src/aadistill/             algorithm core (data, init, models, rollout, training, evaluation)
scripts/
  data/                    dataset builders
  training/                stage entry points
  evaluation/              eval + unrestricted (P18) generation
  rollout/                 teacher generation
  pod/                     GPU session scripts (run_env.sh is the only per-session edit)
tests/                     mirrors src/ and scripts/
artifacts/                 gitignored; large outputs live on the HF relay
data/                      gitignored splits; manifests are tracked
```

Per-run logs and per-experiment proposals were consolidated into
`logs/EXPERIMENTS.md` on 2026-07-31; the originals are in git history at commit
`866dac2`.
