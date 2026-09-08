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

register_builtin_adapters()


@pytest.fixture
def repo_root() -> Path:
    """The repository, for the few assertions that are about the real tree.

    Most tests should build what they need under `tmp_path` instead: a property
    pinned by construction keeps holding, whereas one read off the repository
    only describes whatever it contains today.
    """
    return REPO
