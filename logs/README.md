# logs

The project's records. **Start here.**

| I want to know… | read |
| --- | --- |
| what is true right now | [`STATE.md`](STATE.md) — human view · [`current_state.json`](current_state.json) — machine view |
| what has been spent | [`BUDGET_LEDGER.md`](BUDGET_LEDGER.md), and run [`derive_budget.py`](../scripts/consolidate/derive_budget.py) for the live balances |
| every run, in either layout | [`runs/index.json`](runs/index.json) |
| why something was decided | [`decisions.md`](decisions.md) |
| which file owns which fact | [`CATALOG.md`](CATALOG.md) |
| what happened in a phase | [`PHASE_INDEX.md`](PHASE_INDEX.md) |

Nothing in this directory authorizes anything. A launch needs a grant, a
launch-bound readiness record, a one-use authorization and a bundle — see
[`STATE.md`](STATE.md).

## How it is arranged

```
logs/
├── README.md STATE.md current_state.json    the entry points
├── CATALOG.md PHASE_INDEX.md decisions.md
├── BUDGET_LEDGER.md supported_models.md
├── runs/            every run, by stage → experiment → run
├── experiments/     experiment-level material and history
├── maintenance/     storage, scratch and inventory records
├── validations/     engineering validations, by subject and version
├── migrations/      source relocations, by subject and version
└── archive/         superseded documents, kept verbatim
```

### Runs

```
logs/runs/stage-<stage_id>/<experiment_id>/<run_id>/   current
logs/runs/<experiment_id>/<run_id>/                    historical
```

**Three separate dimensions.** The pipeline *stage* is declared by the
experiment's configuration and never inferred from a name; the *experiment*
carries the phase; the *run* is one execution attempt. A provider *draw* is a
fourth thing again and lives inside a run's evidence, not in its path.

[`runs/index.json`](runs/index.json) is the canonical list and covers **both**
layouts. Its `kinds` block says what each entry is — a recorded run, a prepared
run that never executed, historical evidence with no manifest, a legacy
aggregate spanning several directories, or a directory that only describes
itself.

### Why some runs are not under a stage

They predate the grouping, and their grants, readiness records and closeouts
name their existing paths. A run whose identity was issued against its location
cannot be moved without breaking the lineage that makes it evidence. They are
**found, not relocated** — the index lists them where they are.

### Experiments

`experiments/<experiment_id>/` holds what belongs to an experiment rather than
to one of its runs: its history, its analyses, its superseded plans. Historical
material moved here on 2026-09-12; every move is recorded old-path → new-path in
[`maintenance/log_relocation.json`](maintenance/log_relocation.json), so a
citation in an older document can be followed forward.

Material that did **not** move is listed there too, each with what holds it: an
executable that names the exact path, a record that stores the path beside a
digest, or a top-level entry point. Those are reasons, checkable against the
tree.

## The rule this directory is maintained by

**One fact, one owner.** A number that can be derived from a manifest, a
closeout, an approval or the ledger is derived — not copied into a second
document that then has to be remembered. Where you see a figure here, it names
the thing that owns it.
