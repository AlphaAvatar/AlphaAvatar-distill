"""The replay spec builder, against attempt 3's real committed evidence.

No fixtures stand in for the selection or the journal here: the point of these
tests is that the files actually in the repository can pin a replay, and that
the builder refuses every way they could fail to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "tests"))
from historical_declarations import (  # noqa: E402
    digest_pinned_replay_is_buildable as _replay_buildable,
)

ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

from experiments.phase_c2 import replay_specs as R  # noqa: E402

#: What attempt 3 committed. Restated here so an edit to the module's constants
#: cannot quietly redirect the replay at a different selection.
SELECTION = "7271091c91416b523865ea1320190e4a9ceafb97ecc68bc1b0e49f621573b673"
SOURCE_COMMIT = "2421f630bc812d414ae25245e855059ffe29610d"
LEAF_PARAMS = 596_049_920


def test_the_committed_selection_is_the_one_the_replay_reconstructs():
    record = R.load_selection(ROOT)
    assert record["selection_sha256"] == SELECTION
    assert record["n_selected"] == 5 == len(record["selected"])
    assert record["search"]["seed"] == 20260815


def test_every_leaf_resolves_a_complete_four_step_ancestry():
    leaves = R.build_replay_leaves(ROOT, device="cpu")
    assert len(leaves) == 5
    for leaf in leaves:
        assert len(leaf.spec.steps) == 4, leaf.state_id
        assert len(leaf.step_digests) == 4
        #: EVERY step pinned, which is the requirement the whole replay rests
        #: on. A path pinned only at its leaf leaves three operators unchecked.
        assert all(s.expected_artifact_digest for s in leaf.spec.steps)
        assert leaf.step_digests[-1] == leaf.artifact_digest
        assert leaf.num_parameters == LEAF_PARAMS
        assert len(leaf.single_shard_sha256) == 64
        assert leaf.spec.seed == 20260815


def test_the_five_paths_are_distinct_and_share_no_prefix():
    """No shared prefix means no cross-path caching question to answer.

    If two leaves shared their first operator, reusing that intermediate would
    be tempting and would need an equivalence argument. They do not, so the
    replay can run five independent paths and say so.
    """
    leaves = R.build_replay_leaves(ROOT, device="cpu")
    firsts = [leaf.step_digests[0] for leaf in leaves]
    assert len(set(firsts)) == 5
    orders = [tuple(s.impl_id for s in leaf.spec.steps) for leaf in leaves]
    assert len(set(orders)) == 5, "the joint search selected five distinct orders"


def test_the_journals_duplicate_rows_agree_and_collapse():
    states = R.load_states(ROOT)
    raw = [line for line in
           (ROOT / R.JOURNAL_REL).read_text().splitlines() if line.strip()]
    assert len(raw) == 202, "the journal attempt 3 collected"
    assert len(states) == 108, "deduplicated by state id"


def test_a_journal_whose_rows_disagree_is_refused(tmp_path):
    """Two rows for one state with different digests is not a record."""
    rows = [json.loads(line) for line in
            (ROOT / R.JOURNAL_REL).read_text().splitlines() if line.strip()]
    poisoned = dict(rows[0])
    poisoned["artifact_digest"] = "0" * 64
    target = tmp_path / R.JOURNAL_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(r) for r in [rows[0], poisoned]) + "\n")
    with pytest.raises(R.ReplaySourceError, match="disagree about"):
        R.load_states(tmp_path)


def test_a_substituted_selection_is_refused_even_if_internally_consistent(
        tmp_path):
    """The commitment hash catches an edit; the equality catches a swap.

    A different selection, rehashed so it passes its own check, is exactly what
    `stage1_selection.load` cannot detect — and is the one way a replay could be
    pointed at a Top-5 nobody approved.
    """
    from aadistill.infrastructure.manifest import sha256_json
    from aadistill.initialization.planning import stage1_selection

    record = json.loads((ROOT / R.SELECTION_REL).read_text())
    record["selected"] = record["selected"][:4]
    record["n_selected"] = 4
    body = {k: v for k, v in record.items()
            if k not in ("selection_sha256", "generated_utc")}
    record["selection_sha256"] = sha256_json(body)
    target = tmp_path / R.SELECTION_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n")

    #: It passes the generic loader — that is the point.
    assert stage1_selection.load(target)["n_selected"] == 4
    with pytest.raises(R.ReplaySourceError, match="A replay is bound to one"):
        R.load_selection(tmp_path)


#: Every operator module the 2026-09-25 topology migration moved or rewrote,
#: which is why the digest-pinned replay can no longer be built on this tree.
#: Listed rather than counted so an UNEXPECTED entry is still a failure.
MOVED_BY_THE_TOPOLOGY_MIGRATION = {
    "src/aadistill/initialization/operators/attention.py",
    "src/aadistill/initialization/operators/attention_activation.py",
    "src/aadistill/initialization/operators/composite.py",
    "src/aadistill/initialization/operators/depth.py",
    "src/aadistill/initialization/operators/ffn.py",
    "src/aadistill/initialization/operators/width.py",
    "src/aadistill/initialization/statistics/attention.py",
    "src/aadistill/initialization/operators/__init__.py",
    "src/aadistill/initialization/operators/_common.py",
    "src/aadistill/initialization/operators/base.py",
    "src/aadistill/initialization/operators/register.py",
    "src/aadistill/initialization/planning/fixed_path.py",
    "src/aadistill/initialization/planning/search.py",
    "src/aadistill/initialization/statistics/collect.py",
    "scripts/experiments/phase_c2/search_space.py",
}


def test_the_operator_guard_now_refuses_this_tree_and_says_why():
    """The replay is digest-pinned to attempt 3's bytes, and those bytes moved.

    This asserted `modified_or_removed == {}` while the flat operator layout
    matched attempt 3's. The topology migration moved that layout, so the guard
    refuses — correctly, and in the only direction that is safe: attempt 3's
    artifact digests were produced by the old bytes, so rebuilding from the new
    ones would not be a replay.

    Nothing is owed by that refusal. C2 is CLOSED WITHOUT PROMOTION and the
    Top-5 were already reconstructed 5/5 exact and verified off-pod before the
    migration; no further replay is authorized or needed. What matters is that
    the guard REFUSES rather than quietly rebuilding, and that it names what
    moved — so this asserts both.
    """
    with pytest.raises(R.ReplaySourceError) as caught:
        R.assert_operators_unmoved(ROOT)
    message = str(caught.value)
    assert "not a replay" in message, message
    named = {path for path in MOVED_BY_THE_TOPOLOGY_MIGRATION if path in message}
    assert named == MOVED_BY_THE_TOPOLOGY_MIGRATION, (
        "the refusal does not name every file the migration moved; unnamed: "
        f"{sorted(MOVED_BY_THE_TOPOLOGY_MIGRATION - named)}")


def test_the_commit_it_pins_is_unchanged():
    """The binding's constants are untouched; only the tree moved away.

    Read from the module rather than through `source_binding`, which builds
    replay leaves and therefore runs the operator guard. The point here is that
    this round edited no part of what attempt 3 is pinned TO.
    """
    assert R.ATTEMPT3_SESSION_COMMIT == SOURCE_COMMIT
    assert R.SELECTION_SHA256 == SELECTION


def test_the_operator_guard_refuses_a_tree_whose_operators_moved():
    """A gate nobody has seen fail is not evidence.

    Walking back far enough that the operator trees genuinely differ is a real
    divergence, not a synthetic one, and the guard must refuse it.
    """
    with pytest.raises(R.ReplaySourceError, match="have moved since"):
        R.assert_operators_unmoved(ROOT, head=f"{SOURCE_COMMIT}~25")


def test_an_addition_after_the_source_commit_would_still_be_allowed():
    """The asymmetry the guard encodes, asserted from the rule it publishes.

    A file that did not exist at attempt 3 cannot have influenced it, so adding
    one must not trip the guard. That rule is unchanged; what changed is that
    this tree also MODIFIES files attempt 3 did execute, so the guard refuses
    for that reason instead. The rule text is read from the refusal so the
    asymmetry stays checked.
    """
    with pytest.raises(R.ReplaySourceError) as caught:
        R.assert_operators_unmoved(ROOT)
    assert "removed" in str(caught.value) or "modified" in str(caught.value)
    #: additions are absent from the refusal -- only modified/removed appear
    assert "(added)" not in str(caught.value)


def test_the_source_binding_is_refused_on_a_tree_whose_operators_moved():
    """`source_binding` builds replay leaves, so it inherits the guard.

    That coupling is deliberate and is worth asserting: a binding is a promise
    to reproduce specific digests, and handing one out from a tree that cannot
    reproduce them would be the failure the guard exists to prevent.
    """
    with pytest.raises(R.ReplaySourceError, match="not a replay"):
        R.source_binding(ROOT)


@pytest.mark.skipif(not _replay_buildable(), reason=(
    "exercises the binding's SHAPE, which requires building replay leaves; the "
    "topology migration moved the operator bytes attempt 3's digests are pinned "
    "to, so the guard refuses first. The refusal is asserted directly above, "
    "and C2 is closed with the Top-5 already reconstructed 5/5 exact."))
def test_the_source_binding_is_machine_readable_and_names_the_commit():
    """The review named attempt 3's commit in prose; the record needs it as data."""
    binding = R.source_binding(ROOT)
    assert binding["source_session_commit"] == SOURCE_COMMIT
    assert binding["selection_sha256"] == SELECTION
    assert binding["full_journal_sha256"] == (
        "eb008ee252ce9d35e7478c3cc346ef7db29acf3eb5bfb2486c6023e19abe4ef0")
    assert len(binding["leaves"]) == 5
    for leaf in binding["leaves"]:
        assert len(leaf["step_digests"]) == 4
        assert leaf["num_parameters"] == LEAF_PARAMS
    #: Serializable, because it is written into an authorization.
    assert json.loads(json.dumps(binding))["selection_sha256"] == SELECTION
