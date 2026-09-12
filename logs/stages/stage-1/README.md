# stage-1

Pipeline stage **1**. Every experiment here DECLARES this
stage in `configs/experiments/<id>/authorization.json`; the stage is
never inferred from an experiment's name.

| experiment | material |
| --- | --- |
| [`continuation_b/`](continuation_b/) | analyses, plans, runs |
| [`measurement/`](measurement/) | analyses, runs |
| [`phase_a/`](phase_a/) | analyses, history, plans, results, runs |
| [`phase_b/`](phase_b/) | analyses, plans, runs |
| [`phase_c1/`](phase_c1/) | analyses, history, plans, results, runs, validations |
| [`recovery_continuation/`](recovery_continuation/) | analyses, history, runs |

An experiment's plans, analyses, results, history, validations and
runs are all inside its own directory. Canonical run list, across
every stage: [`../../index.json`](../../index.json).
