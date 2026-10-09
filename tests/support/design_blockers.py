"""Reach a LATER refusal in an issuer whose first refusal is an open blocker.

`d1_authorization.build_payload` refuses an open design blocker before it checks
the grant, the funded list, the price or the harness. That order is correct — an
open blocker is the cheapest and most categorical refusal — and it means a test
whose subject is one of the later refusals cannot reach it while any blocker is
open.

The temptation is to assert on whichever message comes out. That would make
every one of those tests a second copy of the blocker test and retire the checks
they were written for: **41 tests across four files** failed exactly this way
the first time a fourth blocker opened, each reporting "did not match" against
the blocker's message.

So the design is redirected to a blocker-free COPY for the duration. The copy is
byte-identical to the live document except for `open_blockers`, so every
identity the issuer binds — `design_hash` included — is the real one. Nothing is
relaxed and nothing is stubbed: the blocker check still runs, against a document
that reports none.

`DESIGN_REL` is replaced with an ABSOLUTE path, which is what makes this
surgical. `build_payload` resolves the design as `Path(repo_root) / DESIGN_REL`,
and `/` with an absolute right operand returns the right operand — so the design
moves and every other path the issuer reads still resolves against the real
repository.

**MODULE-SCOPED by default**, because the fixtures that issue an artifact are:
a function-scoped `monkeypatch` is torn down before a module-scoped fixture
runs, which would leave the redirect inert and the tests failing exactly as
before. `pytest.MonkeyPatch` is the explicit form for that scope and the undo
stays inside the fixture that applied it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def design_without_blockers(repo_root: Path | str, tmp_path: Path, *,
                            module: Any, attribute: str = "DESIGN_REL") -> Path:
    """Write a blocker-free copy of `module`'s design and return its path.

    Does not patch anything: the caller owns the patch so the undo stays with
    the fixture that applied it.
    """
    root = Path(repo_root)
    rel = getattr(module, attribute)
    live = Path(rel)
    live = live if live.is_absolute() else root / live
    doc = json.loads(live.read_text())
    doc["open_blockers"] = []
    copy = Path(tmp_path) / Path(rel).name
    copy.write_text(json.dumps(doc, indent=1) + "\n")
    return copy


def autouse_blocker_free_design(module: Any, *, repo_root: Path | str | None = None,
                                attribute: str = "DESIGN_REL",
                                scope: str = "module"):
    """The autouse fixture, built once here and named in each file that wants it.

        reach_past_the_blocker_refusal = autouse_blocker_free_design(A)

    One line per file, and visible in the file rather than hidden in a
    directory-wide `conftest` — a test file's own text should say that its
    design is redirected, because the redirect changes which refusal it is
    actually exercising.
    """
    root = Path(repo_root) if repo_root is not None else getattr(module, "REPO")

    @pytest.fixture(scope=scope, autouse=True)
    def _blocker_free_design(tmp_path_factory):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, attribute, str(design_without_blockers(
                root, tmp_path_factory.mktemp("design"), module=module,
                attribute=attribute)))
            yield

    return _blocker_free_design


__all__ = ["autouse_blocker_free_design", "design_without_blockers"]
