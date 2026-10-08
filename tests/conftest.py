"""The CORE suite's fixtures. Reusable framework behaviour only.

Everything shared with the experiment suites — the source roots, the shipped
adapter and operator registries, the cross-module registry isolation — lives in
the repository-root `conftest.py`, because an experiment suite needs it too.

What is left here is the toy-model fixture set, imported from
`support.toy` so that an experiment suite can import exactly the same objects by
name rather than inheriting them from a directory it no longer sits under.

**What this file no longer does, and why.** It used to
`import shared.datasets` and `import shared.calibration` at module
scope, so collecting any core test registered one campaign's frozen dataset
assets as a side effect. That made `tests/` unable to run without the
application layer, and made the application layer's registration order part of
core test behaviour. An experiment suite that needs those registries now imports
them in its own conftest.
"""

from __future__ import annotations

#: Re-exported so every core test sees them, exactly as it did while they lived
#: in `tests/autoinit/conftest.py`. Explicit names rather than a star import:
#: a fixture that disappears should break collection here, in one place, rather
#: than in whichever test used it.
from support.toy import (  # noqa: F401
    calibration_items,
    eval_suite,
    profile,
    suite_items,
    target_spec,
    teacher,
    teacher_spec,
    two_profiles,
)
