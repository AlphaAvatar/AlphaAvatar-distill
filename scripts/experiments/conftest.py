"""The bootstrap every EXPERIMENT suite needs, and the core suite must not.

One file rather than ten. pytest loads every `conftest.py` from the rootdir down
to the collected test, so this applies to each
`scripts/experiments/<phase>/tests/` suite and to none of `tests/` — which is
precisely the boundary the 2026-10-03 refactor drew.

**Why the registry imports live HERE.** `experiments.datasets` and
`experiments.calibration` are the application bootstrap: importing them
registers the frozen dataset assets and calibration profiles of this campaign.
They used to be imported by `tests/conftest.py`, so collecting a core test
pulled in one campaign's frozen assets and the application layer's registration
order became part of core test behaviour. An experiment test legitimately needs
them; a core test does not, and now does not get them.

**The toy fixtures are re-exported, not redefined.** `support.toy` holds the
tiny real Qwen3 teacher and the toy calibration items, and both suites import
the same objects. An experiment test that was moved out of `tests/` keeps
working with the fixtures it was written against, by name.
"""

from __future__ import annotations

#: The application registries. Import side effects are the point: each module
#: registers this campaign's assets at import, which is the application layer's
#: job and not the core's.
import experiments.calibration  # noqa: F401
import experiments.datasets  # noqa: F401

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
