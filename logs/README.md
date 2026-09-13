# logs

The project's records. **Stage → Experiment → Run.**

| I want to know… | read |
| --- | --- |
| what is true right now | [`state/`](state/) |
| what has been spent | [`budget/`](budget/) |
| what each pipeline stage contains | [`stages/`](stages/) |
| shared infrastructure | [`shared/`](shared/) |
| looking after the repository itself | [`maintenance/`](maintenance/) |
| every run, anywhere | [`index.json`](index.json) |

Nothing here authorizes anything.

## The hierarchy

```
logs/
├── README.md
├── index.json          every run, across every stage
├── state/              what is true now
├── budget/             ledger, decisions, approvals/
├── stages/
│   ├── index.json      which stage each experiment belongs to, and why
│   └── stage-<id>/
│       ├── history/    records spanning the stage's experiments
│       └── <experiment>/
│           ├── plans/  analyses/  results/  history/  validations/
│           └── runs/<run_id>/{governance,runtime,evidence,artifacts,closeout}/
├── shared/             infrastructure owned by no stage
└── maintenance/        storage/, inventories/, cleanup/, source-relocations/
```

**One experiment, one directory.** Why the work was done, its protocol, its
analyses, its aggregate results, the engineering validation that supports it,
its history and every run — all in one place. You do not have to decide whether
something is "an experiment thing" or "a run thing" before you can find it.

## Stages

<!-- stages:begin -->
| stage | what it is | its own work | experiments | runs |
| --- | --- | --- | --- | --- |
| [`stage-0/`](stages/stage-0/) | Initialization warm-up data collection | 1 pipeline activity · 2 configs · 1 data manifest | **none** — its output is data and artifacts | 0 |
| [`stage-1/`](stages/stage-1/) | Projection and structural initialization | 3 pipeline activities · 4 configs · 1 data manifest | 5, all with logs | 44 |
| [`stage-2/`](stages/stage-2/) | Offline warm-up data collection | 1 pipeline activity · 1 config · 3 data manifests | **none** — its output is data and artifacts | 0 |
| [`stage-3/`](stages/stage-3/) | Student recovery | 3 pipeline activities · 13 configs · 1 data manifest | 15, of which 5 produced no logs of their own | 0 |

**20 experiments — 20 in one stage, 0 genuinely cross-stage, 0 unresolved.** 2 stage-neutral infrastructure entries are counted separately, in [`shared/`](shared/).

Generated from [`stages/index.json`](stages/index.json), which carries the
evidence for every row.
<!-- stages:end -->

### Stage is read from evidence, never guessed

An experiment's stage comes from repository facts, in this order: an explicit
`stage_id`; the `configs/stage<N>/` directory its config lives in; a stage
config that names it; then its own subject and product. An identifier like
`phase_a` names the *experiment* and says nothing about the stage. Watch two
false friends: a preregistration's `session_plan` stages are **driver** stages,
a different dimension entirely, and an operator called
`composite.stage1_sandwich_v0` names an operator.

A historical experiment does not need a modern `stage_id` field to be placed.
Config location, lineage and subject reconstruct it, and
[`stages/index.json`](stages/index.json) records the evidence for every
assignment along with the rules that decided it — including why a recovery probe
is how an initialization experiment *measures* rather than the stage it belongs
to.

### An experiment spanning several stages would say so

`cross-stage` means several pipeline stages are genuinely the **subject** of one
experiment. It does not mean "no stage was found" — an experiment whose stage is
not yet established is reported **unresolved**, which is a finding to settle.
Every experiment in the repository resolved to exactly one stage on evidence, so
**there is no cross-stage directory**. The first experiment that genuinely earns
one gets it then, with the same shape as any other.

`experiments.run_layout` can still compose that path for an undeclared stage,
and the run index still scans for it. That is deliberate: the module is a member
of the frozen C1 harness digest, so removing the fallback would move the digest
and invalidate an authorized preregistration in order to tidy a directory that
is already gone. The directory is what mattered, and the directory is not here.

### Validations belong to what they serve

The CUDA stage-F validation exists to prove C1's execution path, so it is at
`stages/stage-1/phase_c1/validations/cuda-stage-f/`. Infrastructure validation
whose subject is the machine, the provider or the transport — not any pipeline
stage — is in [`shared/validations/`](shared/validations/). `validation` is not
a top-level category.

### Superseded is not a category either

A document that is only *out of date* is deleted, because git history already
holds it and a second copy in the working tree is one more thing to keep
consistent. A historical document stays only when it is still part of the
scientific record — a preregistration, an immutable result, a consumed
authorization, required provenance — and then it lives under the experiment or
stage that owns it, not on a shelf.

### Old paths

An old path inside a consumed authorization or a closed manifest is **not
stale**: it states where that object was when the payload was written, which is
still true. Those payloads are never rewritten. When current tooling has to
follow such a name to the object's present address,
[`index.json`](index.json)`.historical_paths` maps old path → current path and
nothing else. Everything beyond that pairing — the old bytes, the old tree, the
old layout — is git's, and is not copied here.

## The rule

**One fact, one owner — physically.** A run's result is in its own
`evidence/`, `artifacts/`, `closeout/`; a cross-run aggregate in the
experiment's `results/`; engineering evidence in its `validations/`; money in
`budget/`; current state in `state/`. Never three editable copies.

Configuration is **not** here — `configs/` is the source of truth, and a run's
manifest records the config path and hash it ran under. The same rule keeps a
dataset manifest beside its data in `data/`, which is why stages 0 and 2 have
READMEs here and their material does not move.
