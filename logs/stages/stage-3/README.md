# stage-3

Pipeline stage **3**. Every experiment here DECLARES this
stage in `configs/experiments/<id>/authorization.json`; the stage is
never inferred from an experiment's name.

| experiment | material |
| --- | --- |
| [`e1/`](e1/) | analyses |
| [`e3/`](e3/) | analyses |
| [`e4/`](e4/) | analyses |
| [`e5/`](e5/) | analyses |
| [`e6/`](e6/) | analyses |
| [`e6b/`](e6b/) | analyses |
| [`e7/`](e7/) | analyses, history |
| [`e8/`](e8/) | analyses |
| [`e8b/`](e8b/) | analyses, history |

An experiment's plans, analyses, results, history, validations and
runs are all inside its own directory. Canonical run list, across
every stage: [`../../index.json`](../../index.json).
