"""Importing a Stage-1 result must be stricter than trusting five paths.

Attempts 11 and 12 produced the same search, and attempt 12 preserved the five
selected checkpoints off-pod. A continuation should start at Stage 2 rather than
spend 203 minutes recomputing them — but only if the import verifies what it
claims, from bytes, and refuses everything else.

These run against the **real attempt-12 artifacts and the real preserved
checkpoints**, so what is tested is what a continuation would actually load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))

from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.stage1_import import (  # noqa: E402
    Stage1ImportError, import_stage1_result,
)

D = REPO / "logs/autoinit_phase_a_attempt12"
STORE = Path("/home/ecs-user/aad-artifacts/autoinit/phase_a")

pytestmark = pytest.mark.skipif(
    not (D / "search_result.json").is_file() or not STORE.is_dir(),
    reason="the attempt-12 evidence or its preserved leaves are not present here")


@pytest.fixture(scope="module")
def frozen():
    from phase_a_search import (
        CANONICAL_INIT, CANONICAL_INIT_SHA256, TARGET_GEOMETRY,
    )
    return {"target": TARGET_GEOMETRY, "control_dir": CANONICAL_INIT,
            "control_sha": CANONICAL_INIT_SHA256}


@pytest.fixture(scope="module")
def evidence():
    return (json.loads((D / "search_result.json").read_text()),
            json.loads((D / "selected_leaf_durability.json").read_text()))


def run_import(frozen, result, durability, **over):
    kw = dict(
        search_result=result, states_path=D / "search_states_reduced.jsonl",
        checkpoint_store=STORE, durability=durability,
        adapter=get_adapter("qwen3"),
        expected_config_hash=result["config_hash"],
        target_geometry=frozen["target"], control_dir=frozen["control_dir"],
        control_sha256=frozen["control_sha"])
    kw.update(over)
    return import_stage1_result(**kw)


# --- it works on the real thing ---------------------------------------------

def test_the_real_attempt12_result_imports(frozen, evidence):
    result, dur = evidence
    out = run_import(frozen, result, dur)
    assert len(out.leaves) == 5
    selected = [s["state_id"] for s in result["top_n"]["selected"]]
    assert [s.state_id for s in out.leaves] == selected, "order is the ranking"
    for s in out.leaves:
        assert s.is_complete_leaf(), f"{s.state_id} is not target geometry"
        assert s.validity.value == "measured"
        assert s.num_parameters == 596_049_920
        s.require_recovery_admissible()


def test_the_control_is_rebuilt_from_its_frozen_checkpoint(frozen, evidence):
    """Never from the journal: a re-executed composite is not the incumbent."""
    out = run_import(frozen, *evidence)
    assert out.control.provenance == "retained_canonical"
    assert out.control.artifact.single_shard_sha256 == (
        "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54")


def test_the_imported_leaves_pass_the_live_admission_gate(frozen, evidence):
    """Importing does not bypass `admit_leaves` — it feeds it.

    The five leaves arrive MEASURED, because their Stage-1 evaluations are in
    the evidence and bound to their artifact digests.
    """
    out = run_import(frozen, *evidence)
    for state in out.leaves:
        state.require_recovery_admissible()      # the same call admit_leaves makes


def test_the_control_arrives_unmeasured_and_the_gate_says_so(frozen, evidence):
    """Not an oversight: the control's Stage-1 evaluation is NOT in the
    persisted evidence — the journal has no retained_canonical row — so a
    continuation must measure it on the suite rather than invent it.

    The gate refusing here is the guarantee that it cannot be skipped."""
    from aadistill.autoinit.state import StateError

    out = run_import(frozen, *evidence)
    assert out.verification["control_is_unmeasured"] is True
    with pytest.raises(StateError, match="hash-bound measurements"):
        out.control.require_recovery_admissible()


def test_the_controls_evaluation_really_is_absent_from_the_evidence(evidence):
    """The premise of the test above, checked rather than assumed."""
    result, _ = evidence
    assert set(result["control"]) == {
        "state_id", "provenance", "artifact_digest", "single_shard_sha256",
        "frozen_sha256_verified"}, "the control record gained a measurement"
    rows = [json.loads(l) for l in
            (D / "search_states_reduced.jsonl").read_text().splitlines() if l.strip()]
    assert not [r for r in rows if r.get("provenance") == "retained_canonical"]


# --- and refuses everything else --------------------------------------------

def test_a_different_search_is_refused(frozen, evidence):
    result, dur = evidence
    with pytest.raises(Stage1ImportError, match="config_hash"):
        run_import(frozen, result, dur, expected_config_hash="0" * 64)


def test_a_reordered_selection_is_refused(frozen, evidence):
    """The order IS the ranking."""
    result, dur = evidence
    swapped = {**dur, "leaves": [dur["leaves"][1], dur["leaves"][0],
                                 *dur["leaves"][2:]]}
    with pytest.raises(Stage1ImportError, match="order matters"):
        run_import(frozen, result, swapped)


def test_a_missing_leaf_is_refused(frozen, evidence, tmp_path):
    result, dur = evidence
    empty = tmp_path / "store"
    (empty / dur["leaves"][0]["state_id"]).mkdir(parents=True)
    with pytest.raises(Stage1ImportError):
        run_import(frozen, result, dur, checkpoint_store=empty)


def test_a_substituted_leaf_is_refused(frozen, evidence, tmp_path):
    """Right id, wrong bytes: the digest must come from the file, not the name."""
    result, dur = evidence
    fake = tmp_path / "store"
    fake.mkdir(parents=True)
    ids = [r["state_id"] for r in dur["leaves"]]
    # A REAL substitution: leaf 1's bytes wearing leaf 0's id.
    #
    # SYMLINKS, not copies. Copying five 1.11 GiB checkpoints into pytest
    # scratch filled the disk and failed the run on ENOSPC — a test that needs
    # 5.55 GiB to prove one substitution is a test that will take the box down.
    # And copying another leaf's config.json would not work either: all five
    # share the target geometry, so the config is identical and the digest would
    # not move.
    (fake / ids[0]).symlink_to(STORE / ids[1])
    for sid in ids[1:]:
        (fake / sid).symlink_to(STORE / sid)
    with pytest.raises(Stage1ImportError, match="identifies as|shard"):
        run_import(frozen, result, dur, checkpoint_store=fake)


def test_a_digest_claim_that_the_bytes_contradict_is_refused(frozen, evidence):
    result, dur = evidence
    lying = {**dur, "leaves": [{**dur["leaves"][0], "artifact_digest": "0" * 64},
                               *dur["leaves"][1:]]}
    with pytest.raises(Stage1ImportError, match="identifies as"):
        run_import(frozen, result, lying)


def test_a_wrong_target_geometry_is_refused(frozen, evidence):
    """A leaf that is not the target size would rank well and deploy never."""
    result, dur = evidence
    # NOT 1024: the real target IS 1024, so mutating it to that changed
    # nothing and the first version of this test passed vacuously.
    assert frozen["target"]["hidden_size"] == 1024
    other = {**dict(frozen["target"]), "hidden_size": 2048}
    with pytest.raises(Stage1ImportError, match="is not the target"):
        run_import(frozen, result, dur, target_geometry=other)


def test_a_wrong_control_hash_is_refused(frozen, evidence):
    result, dur = evidence
    with pytest.raises(Exception):
        run_import(frozen, result, dur, control_sha256="0" * 64)


def test_a_journal_without_the_states_is_refused(frozen, evidence, tmp_path):
    result, dur = evidence
    empty = tmp_path / "states.jsonl"; empty.write_text("")
    with pytest.raises(Stage1ImportError, match="no record for"):
        run_import(frozen, result, dur, states_path=empty)


def test_there_is_no_permissive_state_deserializer():
    """The journal is evidence, not a trusted serialization format.

    A generic `InitializationState.from_dict` would let any recorded line become
    a live candidate. The import reconstructs field by field from values it has
    re-derived instead.
    """
    from aadistill.autoinit.state import InitializationState

    assert not hasattr(InitializationState, "from_dict"), (
        "a permissive deserializer appeared; the strict import exists so the "
        "journal never becomes a trusted input")
    src = (REPO / "src/aadistill/autoinit/stage1_import.py").read_text()
    assert "from_dict" not in src.split('"""')[2], (
        "the import gained a from_dict path")


def test_an_evaluation_measured_on_another_artifact_is_refused(frozen, evidence,
                                                               tmp_path):
    """Metrics bind to artifacts, and importing does not relax that.

    The import checks this explicitly AND `attach_evaluation` checks it again —
    deliberate redundancy, which is why removing either alone changes no
    behaviour. This asserts the property rather than one of its two guards.
    """
    result, dur = evidence
    rows = [json.loads(l) for l in
            (D / "search_states_reduced.jsonl").read_text().splitlines() if l.strip()]
    ids = [r["state_id"] for r in dur["leaves"]]
    swapped = tmp_path / "states.jsonl"
    with swapped.open("w") as f:
        for r in rows:
            if r.get("state_id") == ids[0]:
                # leaf 0's record wearing leaf 1's measurement
                other = next(x for x in rows if x.get("state_id") == ids[1])
                r = {**r, "evaluation": other["evaluation"]}
            f.write(json.dumps(r) + "\n")

    with pytest.raises(Stage1ImportError, match="measured on"):
        run_import(frozen, result, dur, states_path=swapped)
