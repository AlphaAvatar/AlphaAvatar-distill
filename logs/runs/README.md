# logs/runs

Every managed run in this repository, in two layouts that one index reads.

```
logs/runs/stage-<stage_id>/<experiment_id>/<run_id>/   current
logs/runs/<experiment_id>/<run_id>/                    historical
```

**Canonical index: `index.json`.** It lists every run in both layouts, and
reports under `unrecorded` any run directory that exists without a manifest.
Read it rather than this file; nothing here is authoritative.

## Three dimensions, kept separate

| dimension | where it lives | example |
| --- | --- | --- |
| pipeline stage | the `stage-<id>` directory | `stage-1` — projection and structural initialization |
| experiment (carries the phase) | `<experiment_id>` | `phase_c1` |
| attempt | `<run_id>` | `attempt13` |

The stage is **declared by the experiment's configuration** and passed to the
launcher — never inferred from a directory name or from the string `c1`. Stages
`0`–`6` are the pipeline stages of `AGENTS.md` §4; `stage-shared` is for work
that is not a pipeline stage at all, such as CUDA engineering validation or
storage maintenance, which would otherwise be filed under whichever stage it
happened to touch.

## Why some runs are not under a stage

Runs that predate the stage grouping stay where they are. Their grants,
readiness records, closeouts and — for several — recorded directory digests all
name their existing paths, and moving a run whose identity was issued against
its location breaks the lineage that makes it evidence. They are **found, not
relocated**. `logs/runs/cuda_stage_f/` in particular carries per-directory
digests in `index.json`, and an appended file would change them.

## What a run directory contains

`manifest.json` (the run's own index), an optional `README.md`, and the five
areas `governance/ runtime/ evidence/ artifacts/ closeout/`. Nothing else: every
file in a run belongs to a declared role, so that a stray file cannot accumulate
without an owner.
