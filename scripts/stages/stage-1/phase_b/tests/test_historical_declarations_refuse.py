"""The completed declarations no longer resolve, and that is the correct state.

Phase A/B, Continuation-B and C1 each froze the executable source set their runs
executed. The 2026-09-25 topology migration moved several of those files. The
tuples are preserved byte-for-byte — they describe what ran — and every digest
helper built on them now refuses rather than hashing whatever survived.

This file asserts that refusal explicitly, so it is a checked property rather
than an absence someone might later "fix" by editing history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from shared.historical_declarations import (  # noqa: E402
    assert_declaration_refuses,
    missing_from_tree,
)
from aadistill.governance.authorization import AuthorizationError  # noqa: E402
from stages.phase_b.plan import (  # noqa: E402
    PHASE_B_EXECUTABLE_SOURCE_FILES_V1,
    phase_b_source_digest,
)
from stages.phase_b.continuation import (  # noqa: E402
    CONTINUATION_SOURCE_FILES_V2,
    continuation_source_digest,
)

#: Exactly what the 2026-09-25 topology migration moved — the five flat
#: operator modules. Later relocations are explained differently: an absent
#: member whose current address the historical-path table knows
#: (`shared.run_layout.resolve_historical`) is a RECORDED move, and an absence
#: with neither explanation is real drift and still fails.
EXPECTED_MOVED = {
    "src/aadistill/initialization/operators/attention.py",
    "src/aadistill/initialization/operators/composite.py",
    "src/aadistill/initialization/operators/depth.py",
    "src/aadistill/initialization/operators/ffn.py",
    "src/aadistill/initialization/operators/width.py",
}


def _explained(path: str) -> bool:
    from shared.run_layout import resolve_historical

    if path in EXPECTED_MOVED:
        return True
    moved_to = resolve_historical(path, REPO)
    return moved_to != path and (REPO / moved_to).is_file()

CASES = (
    ("PHASE_B_EXECUTABLE_SOURCE_FILES_V1", PHASE_B_EXECUTABLE_SOURCE_FILES_V1,
     phase_b_source_digest),
    ("CONTINUATION_SOURCE_FILES_V2", CONTINUATION_SOURCE_FILES_V2,
     continuation_source_digest),
)


@pytest.mark.parametrize("name,declared,digest_fn", CASES,
                         ids=[c[0] for c in CASES])
def test_the_declaration_refuses_rather_than_hashing_what_survived(
        name, declared, digest_fn):
    gone = assert_declaration_refuses(digest_fn, declared,
                                      error_type=AuthorizationError)
    unexplained = sorted(p for p in gone if not _explained(p))
    assert not unexplained, (
        f"{name} is missing paths no recorded migration explains: "
        f"{unexplained}")


@pytest.mark.parametrize("name,declared,digest_fn", CASES,
                         ids=[c[0] for c in CASES])
def test_the_declaration_itself_is_unedited(name, declared, digest_fn):
    """The tuple still names the flat paths. Editing it would rewrite history.

    Stated as an assertion because the tempting repair — point the tuple at the
    new packages so the digest works again — would make a frozen record claim
    that a completed experiment executed code written weeks later.
    """
    assert EXPECTED_MOVED <= set(declared), (
        f"{name} no longer names the paths its runs executed; a historical "
        "declaration was edited to satisfy today's tree")


def test_the_successors_exist_so_the_absence_is_a_move_not_a_loss():
    """An unexplained absence and a recorded move are different failures."""
    successors = (
        "src/aadistill/initialization/operators/attention/gqa/weight_proxy.py",
        "src/aadistill/initialization/operators/composite/stage1_sandwich.py",
        "src/aadistill/initialization/operators/depth/causal_kl_greedy.py",
        "src/aadistill/initialization/operators/depth/positional.py",
        "src/aadistill/initialization/operators/ffn/dense/activation_importance.py",
        "src/aadistill/initialization/operators/width/residual/global_pca.py",
    )
    assert missing_from_tree(successors) == []
