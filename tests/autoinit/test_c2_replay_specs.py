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


def test_the_operator_guard_accepts_the_current_tree():
    report = R.assert_operators_unmoved(ROOT)
    assert report["source_commit"] == SOURCE_COMMIT
    assert report["modified_or_removed"] == {}


def test_the_operator_guard_refuses_a_tree_whose_operators_moved():
    """A gate nobody has seen fail is not evidence.

    Walking back far enough that the operator trees genuinely differ is a real
    divergence, not a synthetic one, and the guard must refuse it.
    """
    with pytest.raises(R.ReplaySourceError, match="have moved since"):
        R.assert_operators_unmoved(ROOT, head=f"{SOURCE_COMMIT}~25")


def test_an_addition_after_the_source_commit_is_allowed():
    """This module did not exist at attempt 3 and could not have influenced it.

    The asymmetry is what lets the replay's own code live in a checked tree.
    """
    report = R.assert_operators_unmoved(ROOT)
    assert "rule" in report
    assert "added after the source commit cannot have influenced it" \
        in report["rule"]


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
