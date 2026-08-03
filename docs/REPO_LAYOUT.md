# Repository layout

```
AGENTS.md                  root working contract (P1-P18)
README.md                  public-facing; performance table is generated
docs/REPO_LAYOUT.md        this file — where new files belong
logs/
  STATE.md                 canonical handoff — read this first
  EXPERIMENTS.md           the consolidated experiment record (what ran, results, cost)
  PROPOSAL.md              the single active plan
  decisions.md             decision records
  supported_models.md      model status table
  artifact_manifests.md    external artifact manifests
  e1_test_cases.md         reviewable sample generations (jsonl alongside)
configs/
  stage0/ stage1/          collection and init configs
  stage3/recovery.json     canonical recovery config for the scaling study
  stage3/s2v1_from_init.json   the logged config of the current branch-point run
src/aadistill/             algorithm core (data, init, models, rollout, training,
                           evaluation, infrastructure)
  data/sessions.py         session rendering, system-grouped packing, ladder cuts
scripts/
  data/                    dataset builders · build_token_ladder · validate_corpus_gate
  training/                stage entry points
  evaluation/              eval + uncapped (P18) vLLM generation, degeneration
                           detection, prompt-rendering audit, exposure and
                           result consolidation, reviewable test-case extraction
  rollout/                 teacher generation · build_recovery_corpus
  pod/                     GPU session scripts (run_env.sh is the only per-session edit)
tests/                     mirrors src/ and scripts/ — 301 CPU tests
assets/                    perf_trend.json + the rendered trend figure
artifacts/                 gitignored; large outputs live on the HF relay
data/                      gitignored splits; manifests are tracked
```

Where new files go:

* **Reusable algorithm code** → `src/aadistill/<responsibility>/`, never a script.
* **Entry points** → `scripts/<responsibility>/`, one concern each.
* **Stage recipes** → `configs/stage<N>/`; a run differs from the canonical
  config only in data and schedule fields.
* **Anything heavy** (checkpoints, corpora, activation caches) → `artifacts/` or
  `data/`, both gitignored, with a manifest entry in
  `logs/artifact_manifests.md` when it lives outside git.

Per-run logs and per-experiment proposals were consolidated into
`logs/EXPERIMENTS.md` on 2026-07-31; the originals are in git history at commit
`866dac2`. Source, scripts, configs, logs and tests were grouped by
responsibility and stage on 2026-07-30 (`633dc6b`), so commands recorded in
older logs use pre-reorganization paths.
