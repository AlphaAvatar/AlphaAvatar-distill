"""Historical executable declarations, and what a current test may ask of them.

A completed experiment's declared source set names the paths that experiment
actually executed. When a later refactor moves those files, the declaration does
NOT become wrong — it keeps describing what ran. What changes is that it can no
longer be resolved against the live tree, and the digest helpers say so by
raising, which is the fail-closed behaviour that stops a launch from proceeding
over a smaller executable than the one it would run.

So there are two different questions, and a test must pick one:

    historical: what did that experiment declare?     -> always answerable
    current:    does that declaration resolve here?   -> may be NO, on purpose

The wrong repair is to edit the tuple so today's packages resolve. That rewrites
what a completed run executed, and every frozen digest built on it. The right
one is for a current test to assert the refusal — which is what this module
makes cheap, and what keeps the gate at full strength rather than skipping past
it silently.

`unresolved_declaration` is deliberately computed from the filesystem rather
than from a marker someone sets: a skip predicate keyed on an environment flag
is a predicate that can be inverted by the environment, and this project has
already shipped one of those.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def missing_from_tree(declared: Sequence[str],
                      repo_root: Path | str = REPO) -> list[str]:
    """Declared paths the live tree no longer has, in declaration order."""
    root = Path(repo_root)
    return [rel for rel in declared if not (root / rel).is_file()]


def declaration_resolves(declared: Sequence[str],
                         repo_root: Path | str = REPO) -> bool:
    return not missing_from_tree(declared, repo_root)


def requires_resolvable(declared: Sequence[str], what: str):
    """Skip a test that can only run while a historical declaration resolves.

    Use for tests that exercise machinery *through* a historical digest. The
    machinery is not under suspicion and the declaration is not edited; the
    input simply is not available in this tree any more. Pair every use with
    `assert_declaration_refuses`, so the refusal is asserted somewhere rather
    than merely skipped around.
    """
    gone = missing_from_tree(declared)
    return pytest.mark.skipif(
        bool(gone),
        reason=(f"{what} names {len(gone)} path(s) the 2026-09-25 topology "
                f"migration moved (first: {gone[0] if gone else '-'}). The "
                "declaration is preserved exactly and the digest helper refuses "
                "by design; see the refusal test in this module's suite."))


def assert_declaration_refuses(digest_fn, declared: Sequence[str], *,
                               error_type: type[BaseException],
                               repo_root: Path | str = REPO) -> list[str]:
    """The positive form: the gate REFUSES, and for the stated reason.

    Returns the missing paths so a caller can report them. Asserting this is the
    point — a declaration that quietly produced a digest over whatever files
    happened to survive is the failure the helpers were written to prevent.
    """
    gone = missing_from_tree(declared, repo_root)
    assert gone, (
        "this declaration still resolves; assert the digest instead of the "
        "refusal")
    with pytest.raises(error_type) as caught:
        digest_fn(repo_root)
    message = str(caught.value)
    assert "missing" in message, message
    assert any(rel in message for rel in gone), (
        f"the refusal does not name any of the missing paths: {message}")
    return gone


def digest_pinned_replay_is_buildable(repo_root: Path | str = REPO) -> bool:
    """Whether a tree can still reproduce attempt 3's pinned artifact digests.

    False once the operator bytes those digests were produced by have moved.
    That is not a defect: `assert_operators_unmoved` refusing is the correct
    outcome, because rebuilding from different bytes is not a replay. Tests
    that exercise the replay *chain* use this to skip; the refusal itself is
    asserted in `tests/autoinit/test_c2_replay_specs.py`.
    """
    import sys
    sys.path.insert(0, str(Path(repo_root) / "scripts"))
    try:
        from experiments.phase_c2 import replay_specs as R
    except Exception:
        return False
    try:
        R.assert_operators_unmoved(Path(repo_root))
    except R.ReplaySourceError:
        return False
    except Exception:
        return False
    return True
