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
  stage3/e1/               Experiment 1's 25 generated arm configs
  stage3/e2/               Experiment 2's generated arm configs
src/aadistill/             algorithm core (data, init, models, rollout, training,
                           evaluation, infrastructure)
  data/sessions.py         session rendering, system-grouped packing, ladder cuts
  data/mixture.py          type-mixture ordering so every ladder prefix keeps the mix
  data/ladder.py           reads a pre-packed ladder at training time
  data/cleaning.py         the five ordered corpus-hygiene gates (clean-v2)
  evaluation/degeneration.py   when a generation has stopped producing information
  evaluation/capability.py     the frozen battery's deterministic scorers
  evaluation/strict_answer.py  boxed-first numeric answer extraction
scripts/
  data/                    dataset builders · build_token_ladder · validate_corpus_gate
  training/                stage entry points
  evaluation/              eval + uncapped (P18) vLLM generation, degeneration
                           detection, prompt-rendering audit, exposure and
                           result consolidation, reviewable test-case extraction
  rollout/                 teacher generation · build_recovery_corpus
  pod/                     GPU session scripts (run_env.sh is the only per-session edit)
tests/                     mirrors src/ and scripts/ — 503 CPU tests, ~6 s
assets/                    perf_trend.json + the rendered trend figure
artifacts/                 gitignored; large outputs live on the HF relay
data/                      gitignored splits; manifests are tracked
```

Where new files go:

* **Reusable algorithm code** → `src/aadistill/<responsibility>/`, never a script.
  The test: if a second script would import it, or a test exercises it directly,
  it belongs in the core. Two modules were promoted on 2026-08-04 (`degeneration`,
  `mixture`) after being reached by `sys.path` hacks from across the tree —
  **a path hack into `scripts/` is the smell that something is misplaced.**
* **Entry points** → `scripts/<responsibility>/`, one concern each. A script may
  import a sibling script when it genuinely extends it (`build_stage2_v1` over
  `v0`); it may not host logic a third module needs.
* **Pod orchestration** → `scripts/pod/`. Two generations coexist and both are
  kept for reproducibility: `orchestrate.sh`/`setup.sh`/`run_env.sh` reproduce
  Experiment 1, `e2p1_*.sh`/`e2p1_driver.py` reproduce Experiment 2 phase 1.
  Retiring either would break P4 reproducibility of a completed experiment.
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
