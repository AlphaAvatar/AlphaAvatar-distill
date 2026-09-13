# logs/stages

Pipeline stages. **Stage → Experiment → Run.**

A stage is here because the repository holds evidence that it
happened — configs, builders, datasets, artifacts, decisions or
experiments — not because AGENTS.md numbers stages 0–6. Stages 4, 5
and 6 will appear the same way, and are not pre-created.

| stage | what it is | its own work | experiments | runs |
| --- | --- | --- | --- | --- |
| [`stage-0/`](stage-0/) | Initialization warm-up data collection | 1 pipeline activity · 2 configs · 1 data manifest | **none** — its output is data and artifacts | 0 |
| [`stage-1/`](stage-1/) | Projection and structural initialization | 3 pipeline activities · 4 configs · 1 data manifest | 5, all with logs | 44 |
| [`stage-2/`](stage-2/) | Offline warm-up data collection | 1 pipeline activity · 1 config · 3 data manifests | **none** — its output is data and artifacts | 0 |
| [`stage-3/`](stage-3/) | Student recovery | 3 pipeline activities · 13 configs · 1 data manifest | 15, of which 5 produced no logs of their own | 0 |

**20 experiments — 20 in one stage, 0 genuinely cross-stage, 0 unresolved.** 2 stage-neutral infrastructure entries are counted separately, in [`shared/`](../shared/).

Generated from [`index.json`](index.json), which carries the
evidence for every row.

A stage with no experiment-run logs is still a real stage. Its
README says so and points at where its material lives; no run,
experiment or manifest is invented to fill it.
