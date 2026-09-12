# logs

The project's records. **Stage → Experiment → Run.**

| I want to know… | read |
| --- | --- |
| what is true right now | [`state/current.md`](state/current.md) · [`state/current.json`](state/current.json) |
| what has been spent | [`budget/ledger.md`](budget/ledger.md), and run [`derive_budget.py`](../scripts/consolidate/derive_budget.py) for live balances |
| what a stage contains | [`stages/`](stages/) |
| work with no established stage | [`cross-stage/`](cross-stage/) |
| shared infrastructure | [`shared/`](shared/) |
| every run, anywhere | [`index.json`](index.json) |
| where an old path went | [`migrations/`](migrations/) |
| a superseded document | [`archive/`](archive/) |

Nothing here authorizes anything.

## The hierarchy

```
logs/
├── README.md
├── index.json          every run, across every stage
├── state/              what is true now
├── budget/             ledger, decisions, approvals/
├── stages/
│   └── stage-<id>/
│       └── <experiment>/
│           ├── plans/  analyses/  results/  history/  validations/
│           └── runs/<run_id>/{governance,runtime,evidence,artifacts,closeout}/
├── cross-stage/<experiment>/    same shape; stage not established
├── shared/             infrastructure analyses and validations
├── maintenance/        storage/, inventories/, cleanup/
├── migrations/         how old paths map forward
└── archive/            stages/, cross-stage/, repository/
```

**One experiment, one directory.** Why the work was done, its protocol, its
analyses, its aggregate results, the engineering validation that supports it,
its history and every run — all in one place. You do not have to decide whether
something is "an experiment thing" or "a run thing" before you can find it.

### Stage is read, never guessed

An experiment's stage comes from `configs/experiments/<id>/authorization.json`
→ `stage_id`. An identifier like `phase_a` or `phase_c1` names the *experiment*;
it says nothing about the pipeline stage. A preregistration's `session_plan`
stages are **driver** stages, a different dimension, and an operator called
`composite.stage1_sandwich_v0` names an operator.

Today only `phase_c1` declares one, so it is the only occupant of
[`stages/stage-1/`](stages/stage-1/). Everything else is in
[`cross-stage/`](cross-stage/) — a statement that no frozen record establishes
its stage, not a holding pen. Stages are never pre-created: a stage exists when
an experiment declares it.

### Validations belong to what they serve

The CUDA stage-F validation exists to prove C1's execution path, so it is at
`stages/stage-1/phase_c1/validations/cuda-stage-f/`. Infrastructure validation
that serves no single experiment is in [`shared/validations/`](shared/validations/).
`validation` is not a top-level category.

### Old paths

Every relocation is recorded in [`migrations/`](migrations/) as old → new, with
the identity each object had before it moved. An old path inside a consumed
authorization or a closed manifest is **not stale**: it states where that object
was when the payload was written, which is still true, and git history holds the
tree. Those payloads are never rewritten.

## The rule

**One fact, one owner — physically.** A run's result is in its own
`evidence/`, `artifacts/`, `closeout/`; a cross-run aggregate in the
experiment's `results/`; engineering evidence in its `validations/`; money in
`budget/`; current state in `state/`. Never three editable copies.

Configuration is **not** here — `configs/` is the source of truth, and a run's
manifest records the config path and hash it ran under.
