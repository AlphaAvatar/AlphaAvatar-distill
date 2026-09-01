"""The C1 session contract: stage order, the two gates, and register-before-use."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.autoinit import c1_session as C  # noqa: E402
from aadistill.autoinit.operators import attention_activation  # noqa: E402
from aadistill.autoinit.operators import registered_implementations  # noqa: E402


@pytest.fixture
def registered():
    attention_activation.register(replace=True)
    yield
    attention_activation.unregister()


# --- stage order ------------------------------------------------------------

def test_the_ten_stages_are_declared_in_order():
    assert [s.letter for s in C.C1_STAGES] == list("ABCDEFGHIJ")
    assert [s.stage_id for s in C.C1_STAGES][:5] == [
        "session_setup", "teacher_fetch_verify", "register_operator",
        "replay_parent", "replay_incumbent"]


def test_a_prefix_is_legal_but_a_permutation_is_not():
    ids = [s.stage_id for s in C.C1_STAGES]
    C.assert_stage_order([])                      # stopped at once
    C.assert_stage_order(ids[:5])                 # stopped at a gate
    C.assert_stage_order(ids)                     # ran to completion
    with pytest.raises(C.C1SessionError, match="out of order"):
        C.assert_stage_order([ids[1], ids[0]])
    with pytest.raises(C.C1SessionError, match="out of order"):
        C.assert_stage_order([ids[0], ids[2]])    # skipped the teacher check


def test_evaluation_cannot_precede_the_probes_and_the_decision_cannot_precede_both():
    ids = [s.stage_id for s in C.C1_STAGES]
    assert ids.index("recovery_probes") < ids.index("evaluate") < ids.index("decide")
    with pytest.raises(C.C1SessionError):
        C.assert_stage_order(ids[:7] + ["decide"])       # decide before evaluate


# --- the gates --------------------------------------------------------------

def test_every_stage_before_training_blocks_training_on_failure():
    ids = [s.stage_id for s in C.C1_STAGES]
    training_at = ids.index("recovery_probes")
    for s in C.C1_STAGES[:training_at]:
        if s.stage_id == "session_setup":
            continue                               # setup fails by not existing
        assert s.blocks_training, f"{s.stage_id} does not block training"
    assert "replay_parent" in C.GATE_STAGES
    assert "replay_incumbent" in C.GATE_STAGES


def test_the_two_replay_gates_name_the_frozen_digests():
    assert C.EXPECTED_PARENT_DIGEST == (
        "eea90c91346a0745b8b1b847503b48fe73c33bb9d75d92c196dc43598e91e722")
    assert C.EXPECTED_INCUMBENT_DIGEST == (
        "c313d1b4081b9a3b410dddf7a29ebcaad8dd0759179d51e1d761238c1743a2a6")
    assert C.EXPECTED_PARENT_DIGEST[:12] in C.stage("D").fail_closed_on
    assert C.EXPECTED_INCUMBENT_DIGEST[:12] in C.stage("E").fail_closed_on


def test_a_mismatch_must_preserve_the_diagnostic_evidence():
    joined = " ".join(C.MISMATCH_EVIDENCE).lower()
    for needed in ("artifact_digest", "depth", "ffn", "width", "runtime"):
        assert needed in joined, needed


# --- register before use ----------------------------------------------------

def test_importing_the_operator_module_does_not_register_it():
    """Stage C exists because import is inert. If import registered the operator,
    `BeamSearch._allowed_impl_ids` would pick it up in any search in the process."""
    attention_activation.unregister()
    import importlib

    importlib.reload(attention_activation)
    assert "attention.activation_importance_v1" not in registered_implementations()


def test_building_the_arms_refuses_until_stage_c_has_registered():
    attention_activation.unregister()
    with pytest.raises(C.C1SessionError, match="not registered"):
        C.build_arm_specs()


def test_after_registration_the_arms_build(registered):
    arms = C.build_arm_specs()
    assert set(arms) == {"incumbent", "treatment"}
    assert arms["incumbent"].kinds == ("DEPTH", "FFN", "RESIDUAL_WIDTH", "ATTENTION")
    assert arms["treatment"].kinds == arms["incumbent"].kinds


# --- the arms differ by exactly one operator --------------------------------

def test_the_two_arms_share_their_prefix_and_differ_only_at_attention(registered):
    arms = C.build_arm_specs()
    assert C.arm_prefix_is_shared(arms)
    a, b = arms["incumbent"], arms["treatment"]
    assert a.steps[:-1] == b.steps[:-1]
    assert a.steps[-1].impl_id == "attention.weight_proxy_v0"
    assert b.steps[-1].impl_id == "attention.activation_importance_v1"
    assert b.steps[-1].profile_id == "calib.domain_balanced@v1"
    assert a.spec_hash != b.spec_hash


def test_both_arms_pin_the_parent_and_only_the_incumbent_pins_the_incumbent(
        registered):
    arms = C.build_arm_specs()
    for name, spec in arms.items():
        assert spec.steps[2].expected_artifact_digest == C.EXPECTED_PARENT_DIGEST, name
    assert arms["incumbent"].steps[3].expected_artifact_digest == (
        C.EXPECTED_INCUMBENT_DIGEST)
    # The treatment arm has no incumbent to reproduce — that is the experiment.
    assert arms["treatment"].steps[3].expected_artifact_digest is None


def test_the_incumbent_arm_reproduces_the_historical_path_label(registered):
    assert C.build_arm_specs()["incumbent"].path_label == (
        "DEPTH(calib.domain_balanced@v1)->FFN(calib.domain_balanced@v1)->"
        "RESIDUAL_WIDTH(calib.reasoning_heavy@v2)->ATTENTION(calib.none@v1)")


def test_the_target_geometry_is_the_frozen_student(registered):
    spec = C.build_arm_specs()["incumbent"].target_spec
    assert spec["hidden_size"] == 1024 and spec["intermediate_size"] == 3072
    assert spec["num_attention_heads"] == 16 and spec["num_key_value_heads"] == 8
    assert spec["num_hidden_layers"] == 28 and spec["head_dim"] == 128


# --- the contract -----------------------------------------------------------

def test_the_contract_is_two_arms_three_seeds_six_probes():
    c = C.C1_SESSION_CONTRACT
    assert (c.n_arms, c.n_seeds, c.n_probes) == (2, 3, 6)
    with pytest.raises(C.C1SessionError, match="every arm runs every seed"):
        C.C1SessionContract(n_arms=2, n_seeds=3, n_probes=5)


def test_the_contract_declares_that_it_contains_no_search():
    d = C.C1_SESSION_CONTRACT.as_dict()
    assert d["contains"] == {"search": False, "ranking": False,
                             "successive_halving": False, "tie_breaking": False,
                             "arm_elimination": False}


def test_the_contract_hash_moves_with_any_gate_or_shape_change():
    base = C.C1_SESSION_CONTRACT.contract_hash
    assert C.C1SessionContract().contract_hash == base
    assert C.C1SessionContract(session_id="other").contract_hash != base
    assert C.C1SessionContract(battery_asset_id="x").contract_hash != base


def test_the_module_carries_no_search_machinery():
    import ast

    path = (Path(__file__).resolve().parents[2]
            / "src/aadistill/autoinit/c1_session.py")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    for forbidden in ("search", "ranking", "phase_a", "phase_b"):
        assert not any(m == forbidden or m.endswith(f".{forbidden}")
                       for m in imported), f"c1_session imports {forbidden}"
