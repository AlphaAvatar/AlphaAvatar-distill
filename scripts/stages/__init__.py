"""Experiment instances, organised by STAGE. Nothing here is a mechanism.

    scripts/stages/stage-1/phase_d1/             <->  logs/stages/stage-1/phase_d1/
    scripts/stages/stage-1/families/d_series/    <->  logs/stages/stage-1/families/d_series/
    scripts/stages/stage-3/e8b/                  <->  logs/stages/stage-3/e8b/

The stage directories mirror `logs/stages/` exactly, name for name, so executable
experiment code and its evidence tree have one obvious mapping. Stage ownership
is taken from `logs/stages/index.json`, which is the repository's own answer.
The shared application layer that used to sit at this level lives in
`scripts/shared/` (package `shared`): it belongs to no stage, and it is not
reusable core either — by P3 it is experiment-instance code.

**Why `__path__` is extended below.** `stage-1` is not a Python identifier, so
`import stages.stage-1.phase_c1` cannot be spelled at all. Extending this
package's search path means `stages.phase_c1` keeps resolving — to
`stage-1/phase_c1/` — so the physical tree can mirror the evidence tree without
every import in the repository growing a stage it does not care about. A
`families/` directory under a stage groups material owned by an experiment
FAMILY rather than one experiment (`stages.d_series` resolves to
`stage-1/families/d_series/`); like the stage level, the family level is a
grouping directory, not a package, so it never appears in an import name.

That is deliberate, and it is not a compatibility shim for the old flat layout:
an experiment package's import name says WHICH experiment or family, and the
stage is a property of where its evidence lives. A caller that needs the stage
reads the index, which owns that fact. Adding a stage or family directory needs
no code change here.
"""

from __future__ import annotations

import pathlib as _pathlib

#: Stage directories first, then family groupings, so a stage- or
#: family-owned package wins over a same-named module at this level -- there
#: is none today, and a silent shadow would be worse than an import error if
#: one appeared.
_here = _pathlib.Path(__file__).parent
__path__ = (
    [str(p) for p in sorted(_here.glob("stage-*")) if p.is_dir()]
    + [str(p) for p in sorted(_here.glob("stage-*/families")) if p.is_dir()]
    + list(__path__)
)
