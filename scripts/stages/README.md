# scripts/stages/ — experiment-owned executables

`stage-<n>/<experiment>/` mirrors `logs/stages/stage-<n>/<experiment>/` name
for name: the code here is the executable whose evidence lives there, and
`artifacts/stages/stage-<n>/<experiment>/` holds its durable bytes. Stage
ownership comes from `logs/stages/index.json` — the ownership index, whose
rows also carry each owner's `canonical_scripts` / `canonical_artifacts` /
`live_state` legs.

`stage-<n>/families/<family>/` is an experiment FAMILY: material owned by a
group of sibling experiments rather than any single one.
`stage-1/families/d_series/` carries the D-series battery family, allocation
rule, identity and scoring protocol shared by D1/D2/D3.

Python: this directory is package `stages`; `__init__.py` extends `__path__`
over the stage and family grouping directories, so `stages.phase_d1` and
`stages.d_series` resolve while the physical tree keeps the stage and family
levels. An experiment's tests live in its own `tests/` and are invoked
explicitly: `pytest scripts/stages/stage-1/phase_d1/tests`.
