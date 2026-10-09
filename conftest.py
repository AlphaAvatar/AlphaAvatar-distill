"""What every suite in this repository needs, and nothing that belongs to one.

There are three suites and they are different things (AGENTS.md §2.8a):

* the **core suite** — `pytest` — reusable framework behaviour under `tests/`;
* a **current experiment suite** —
  `pytest scripts/stages/stage-1/phase_d1/tests`;
* **historical verification** — an explicit run of a closed experiment's tests.

All three need the same two source roots importable and the same shipped
registries populated. That is all this file does.

**What it deliberately does NOT do.** It does not import
`shared.datasets`, `shared.calibration` or any other experiment
registry. The root bootstrap used to, which meant collecting a core test
required importing the application layer that registers one campaign's frozen
assets — the coupling the 2026-10-03 test-boundary refactor removed. An
experiment suite that needs its registries populated does that in its own
`tests/conftest.py`, where the dependency is visible to whoever reads it.

`scripts/` stays on the path for every suite, because core tests legitimately
exercise live scripts: the closure deriver, the log-inventory builder, the
pytest-outcome summarizer. Having the directory importable is not the same as
importing an experiment's registry from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent

#: `tests` is on the path so `support.<module>` resolves identically from the
#: core suite and from an experiment suite four directories away.
for _root in (REPO / "src", REPO / "scripts", REPO / "tests"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

#: Register the shipped architecture adapters and operators once, for whichever
#: suite is running.
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

    Here rather than in `tests/conftest.py` because it protects a CORE
    invariant from any suite: `BeamSearch._allowed_impl_ids` falls back to
    **every registered implementation** when `allowed_impls` is None, so a
    registration that outlives its module silently adds a branch to an
    unrelated search. The leak that proved it came from an experiment suite —
    the C3 launcher's `spec()` calls `register_experimental_operators()` — and
    landed in C2's joint-space enumeration, which met `attention.causal_kl_v1`,
    an implementation its cost model has never measured, and raised
    `CostModelError`.

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
        #: AND NO IMPLEMENTATION KEEPS A PATCHED METHOD. The registry mapping
        #: being unchanged is not the same as the implementations being unchanged,
        #: which this fixture originally assumed.
        #:
        #: `monkeypatch.setattr(impl, "execute", spy)` on an implementation whose
        #: `execute` is INHERITED records the bound method as the old value and,
        #: on undo, writes it back as an INSTANCE attribute. The registry holds
        #: singletons, so that attribute outlives the module and permanently
        #: shadows the class -- and any later class-level patch of the same name
        #: is silently ignored. Measured: a search test that passed alone observed
        #: nothing at all when one earlier module had spied on an operator, and
        #: the symptom was "no expansion ran" rather than anything about patching.
        for impl in list(_ops._IMPLEMENTATIONS.values()):
            for name in ("plan", "apply", "execute"):
                if name in vars(impl):
                    del vars(impl)[name]
