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
for root in (REPO / "src", REPO / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


@pytest.fixture
def repo_root() -> Path:
    """The repository, for the few assertions that are about the real tree.

    Most tests should build what they need under `tmp_path` instead: a property
    pinned by construction keeps holding, whereas one read off the repository
    only describes whatever it contains today.
    """
    return REPO
