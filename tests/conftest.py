"""Make the two repository source roots importable for every test.

`src/` holds the library; `scripts/` holds the experiment instances, which the
initialization cutover moved out of `src/aadistill` because a Phase/C1 plan with
its arms, seeds, replay digests, battery and budget is not a reusable mechanism.

Individual modules already insert `src` themselves, and that stays — a test file
that can be run directly should keep working. This adds `scripts` once, in the
one place every test goes through, rather than as a shim repeated in each of the
sixty-odd files that reach an experiment package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for root in (REPO / "src", REPO / "scripts", REPO / "tests"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


#: Register the shipped architecture adapters once, for the whole suite.
#:
#: This used to happen as a side effect of importing the old `aadistill.autoinit`
#: package. Nothing imports the new one, so whether `get_adapter("qwen3")`
#: resolved depended on whether an earlier test had pulled the adapter module
#: in -- and with `pytest-randomly` active, that varied per run. A module-scope
#: `get_adapter` call in a test file then failed at COLLECTION, taking its whole
#: file with it, in some orders and not others.
from aadistill.initialization.adapters import register_builtin_adapters  # noqa: E402
from aadistill.initialization.operators.register import (  # noqa: E402
    register_builtin_operators)

register_builtin_adapters()
register_builtin_operators()

#: The frozen dataset assets, likewise: `experiments.datasets` is the
#: application bootstrap and registering is its import-time job, not the core's.
import experiments.datasets  # noqa: E402,F401
import experiments.calibration  # noqa: E402,F401


@pytest.fixture
def repo_root() -> Path:
    """The repository, for the few assertions that are about the real tree.

    Most tests should build what they need under `tmp_path` instead: a property
    pinned by construction keeps holding, whereas one read off the repository
    only describes whatever it contains today.
    """
    return REPO


@pytest.fixture(autouse=True, scope="module")
def _operator_registry_is_module_local():
    """No test module may leave an implementation registered for the next one.

    `BeamSearch._allowed_impl_ids` falls back to **every registered
    implementation** when `allowed_impls` is None, so a registration that
    outlives its module silently adds a branch to an unrelated search. That is
    not hypothetical: `tests/pod/test_c3_session_contract.py` loads the C3
    launcher, whose `spec()` calls `register_experimental_operators()`, and
    nothing unregistered it. C2's joint-space enumeration then met
    `attention.causal_kl_v1`, which its cost model has never measured, and
    raised `CostModelError`.

    The visible damage was a suite total nobody could trust: 23 failed / 15
    errors where 11 were real, with the rest appearing and disappearing by
    collection order. Measured: the affected file passed 17/17 alone and failed
    2 when a C3 module ran first.

    **Module scope, not function scope.** Many modules register once for the
    whole file through a module-scoped fixture, and restoring after every test
    would undo that registration before the second test in the module ran.
    Leaks *within* a module are that module's own business; leaks *across*
    modules are what broke an unrelated experiment's cost model. This restores
    exactly at that boundary.
    """
    from aadistill.initialization.operators import base as _ops
    from aadistill.initialization.operators.register import (
        register_builtin_operators,
    )

    before = set(_ops._IMPLEMENTATIONS)
    try:
        yield
    finally:
        #: REMOVE WHAT THIS MODULE ADDED. Nothing more.
        #:
        #: The first version of this fixture snapshotted the whole mapping and
        #: restored it, which looked safer and was worse: a module that
        #: legitimately UNREGISTERED an implementation leaked in from earlier
        #: had it put back, so the fixture propagated leaks forward instead of
        #: clearing them. The full suite went from 23 failures to 29.
        #:
        #: Removals are not restored either -- except the builtins, whose
        #: registrar is documented idempotent. A module that drops an
        #: experimental implementation and does not put it back is cleaning up
        #: after itself, which is the behaviour this fixture wants; re-adding it
        #: would be re-creating the leak.
        for impl_id in sorted(set(_ops._IMPLEMENTATIONS) - before):
            _ops.unregister_implementation(impl_id)
        register_builtin_operators()
