# logs

The project's records, in one canonical layout. **Start here.**

| I want to know… | read |
| --- | --- |
| what is true right now | [`state/current.md`](state/current.md) — human view · [`state/current.json`](state/current.json) — machine view |
| what has been spent | [`budget/ledger.md`](budget/ledger.md), and run [`derive_budget.py`](../scripts/consolidate/derive_budget.py) for live balances |
| every run | [`runs/index.json`](runs/index.json) |
| why something was decided | [`budget/decisions.md`](budget/decisions.md) |
| which file owns which fact | [`state/ownership.md`](state/ownership.md) |
| what happened in a phase | [`state/phase_index.md`](state/phase_index.md) |
| where an old path went | [`migrations/log-layout-v1/manifest.json`](migrations/log-layout-v1/manifest.json) |
| a superseded document | [`archive/`](archive/) |

Nothing here authorizes anything. A launch needs a grant, a launch-bound
readiness record, a one-use authorization and a bundle — see
[`state/current.md`](state/current.md).

## The layout

```
logs/
├── README.md            this file, the only thing at the root
├── state/               what is true now: current.json, current.md, ownership
├── budget/              ledger.md, decisions.md, approvals/ (grants, authorizations)
├── experiments/         <experiment_id>/{plans,analyses,results,history}/
├── runs/                index.json, stage-<stage_id>/<experiment>/<run>/, unscoped/
├── validations/         <validation_id>/ — engineering evidence, not science
├── maintenance/         storage/, inventories/, cleanup/
├── migrations/          log-layout-v1/ — how the old paths map forward
└── archive/             superseded documents, kept verbatim
```

### Runs

```
logs/runs/stage-<stage_id>/<experiment_id>/<run_id>/
logs/runs/unscoped/<experiment_id>/<run_id>/
```

Each run holds `README.md`, `manifest.json` and the five areas `governance/
runtime/ evidence/ artifacts/ closeout/`.

**Stage is declared, never inferred.** It comes from the experiment's
configuration — `configs/experiments/<id>/authorization.json:stage_id` — and
never from a directory name or from the string `c1`. Stages are not enumerated
in advance: a stage the pipeline grows later needs a config entry and no code
change.

**`unscoped/` is for a run whose stage no frozen record determines.** Today that
is every experiment except `phase_c1`: their frozen records describe *driver*
stages (`stage 0` attestation, `stage 1` build), which is a different dimension,
and an operator named `composite.stage1_sandwich_v0` names an operator. Filing
those under a guessed stage would be inventing provenance. When a frozen record
does determine one, the run belongs under that stage.

**Three dimensions, kept apart:** the pipeline *stage*, the *experiment* (which
carries the phase), and the *run* (one execution attempt). A provider *draw* is a
fourth thing and lives inside a run's evidence, not in its path.

### Validations

Engineering validation — CUDA integration, device placement, dtype, a canary —
lives in `validations/<id>/`, never in `runs/`. It answers a hardware question
and produces no scientific measurement.

### Finding an old path

Every object that moved is in
[`migrations/log-layout-v1/manifest.json`](migrations/log-layout-v1/manifest.json)
as old path → new path, with the identity it had before the move.

An old path written inside a frozen payload — a consumed authorization, a closed
run's manifest, an archived document — is **not stale**. It states where the
object was when that payload was written, which is still true, and git history
holds the tree that proves it. Those strings are not rewritten; the manifest is
how you follow one forward.

## The rule this directory is maintained by

**One fact, one owner — as a physical file, not only as a principle.** A result
belongs to its run (`evidence/`, `artifacts/`, `closeout/`); a cross-run summary
to `experiments/<id>/results/`; engineering evidence to `validations/`. The same
result does not exist as three editable copies. A number that can be derived
from a manifest, a closeout, an approval or the ledger is derived, not copied.

Configuration is **not** here: `configs/` is the source of truth, and a run's
manifest records the config path and hash it ran under.
