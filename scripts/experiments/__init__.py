"""Experiment instances, organised by STAGE. Nothing here is a mechanism.

    scripts/experiments/stage-1/phase_d1/        <->  logs/stages/stage-1/phase_d1/
    scripts/experiments/stage-3/tests/           <->  logs/stages/stage-3/

The stage directories mirror `logs/stages/` exactly, name for name, so executable
experiment code and its evidence tree have one obvious mapping. Stage ownership
is taken from `logs/stages/index.json`, which is the repository's own answer:
Stage 1 holds `phase_a`, `phase_a3`, `phase_b`, `phase_c1`, `phase_c2`,
`phase_c3`, `phase_d1`, `measurement` and `recovery_continuation`; Stage 3 holds
the E-series ladder (`e1`-`e8b`), whose code lives under `scripts/training/` and
`scripts/data/` but whose tests belong to the experiments that own them.

**Modules at THIS level are the shared application layer** — the calibration and
dataset registries, the recipes, the recovery policy, the preflight harness, the
deployment relay, the cost model. They are used across stages, so they belong to
no stage, and they are not reusable core either: by P3 they are
experiment-instance code, which is why they live here rather than under
`src/aadistill/`.

**Why `__path__` is extended below.** `stage-1` is not a Python identifier, so
`import experiments.stage-1.phase_c1` cannot be spelled at all. Extending this
package's search path means `experiments.phase_c1` keeps resolving — to
`stage-1/phase_c1/` — so the physical tree can mirror the evidence tree without
every import in the repository growing a stage it does not care about.

That is deliberate, and it is not a compatibility shim for the old flat layout:
an experiment package's import name says WHICH experiment, and the stage is a
property of where its evidence lives. A caller that needs the stage reads the
index, which owns that fact. Adding a stage directory needs no code change here.
"""

from __future__ import annotations

import pathlib as _pathlib

#: Stage directories first, so a stage-owned package wins over a same-named
#: module at this level -- there is none today, and a silent shadow would be
#: worse than an import error if one appeared.
__path__ = [str(p) for p in sorted(_pathlib.Path(__file__).parent.glob("stage-*"))
            if p.is_dir()] + list(__path__)
