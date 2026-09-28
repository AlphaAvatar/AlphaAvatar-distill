"""The C3 executor executes the experiment the frozen plan describes.

WHY THIS EXISTS. At `b937aebb` the preregistration declared **three arms and
nine probes** while `session.py` built **two arms and six probes**, and
nothing in the repository compared them. Worse, the executor was internally
consistent about being wrong: `C1SessionContract` hard-coded `n_arms=2`,
`n_seeds=3`, `n_probes=6`, and its `__post_init__` check passed because
2 x 3 really is 6. A self-consistent statement of the wrong experiment.

So this file asserts the one thing that failure could not survive: **plan and
executor name the same arms, the same configs, the same seeds and the same
probe count.** The executor now derives all of it from the plan, which makes
the equality structural rather than a coincidence maintained by hand -- but
"derived" is a property of today's implementation, and these tests are what
stop a future edit from reintroducing a literal.

They also cover the registration ordering (section 11): every non-builtin
implementation the arms name must be registered explicitly before any spec is
built, and the set registered must be derived from the arms rather than
listed, so a fourth arm cannot arrive unregistered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from experiments.phase_c3 import session as S  # noqa: E402

PREREG = json.loads(
    (REPO / "logs/stages/stage-1/phase_c3/plans/c3_preregistration.json").read_text())


@pytest.fixture
def registered():
    """Stage C, scoped to one test, then undone.

    Explicit rather than ambient for two reasons. Relying on a sibling test
    having registered makes these pass or fail by collection ORDER, which is
    randomized in this suite. And leaving the two experimental ATTENTION
    implementations in the process-global registry is exactly what
    `register.py` warns about: `BeamSearch._allowed_impl_ids` falls back to
    every registered implementation when `allowed_impls` is None, so a
    leaked registration silently adds a branch to an unrelated search.
    """
    from aadistill.initialization.operators.attention.gqa import (
        activation_importance, causal_kl)

    S.register_experimental_operators()
    try:
        yield
    finally:
        for mod in (activation_importance, causal_kl):
            try:
                mod.unregister()
            except Exception:                        # noqa: BLE001 - idempotent
                pass


# ---------------------------------------------------------------------------
# Plan == executor
# ---------------------------------------------------------------------------

def test_the_executor_builds_exactly_the_arms_the_plan_declares(registered):
    """The b937aebb defect, asked directly."""
    planned = [k for k in PREREG["arms"] if not k.startswith("_")]
    assert list(S.arm_ids()) == planned
    specs = S.build_arm_specs(workdir_device="cpu")
    assert sorted(specs) == sorted(planned), (
        f"the plan declares {sorted(planned)} but the executor builds "
        f"{sorted(specs)}")
    assert len(specs) == 3, "C3 is a three-arm experiment"


def test_each_arm_carries_the_plan_s_implementation_profile_and_config(registered):
    """Not just the right NUMBER of arms -- the right arms."""
    specs = S.build_arm_specs(workdir_device="cpu")
    for arm_id, spec in specs.items():
        planned = PREREG["arms"][arm_id]
        tail = spec.steps[-1]
        assert [tail.impl_id, tail.profile_id] == planned["attention"], (
            f"{arm_id}: executor runs {tail.impl_id} @ "
            f"{tail.profile_id}, plan says {planned['attention']}")
        planned_cfg = planned.get("config") or {}
        built_cfg = dict(tail.config) if tail.config else {}
        assert built_cfg == planned_cfg, (
            f"{arm_id}: executor config {built_cfg} != plan config {planned_cfg}")


def test_the_identity_bearing_protocol_actually_distinguishes_B1_from_B3(registered):
    """B1 and B3 share an implementation; only the hashed config separates them.

    If `calibration_forward_batch_size` / `calibration_batch_packing` stopped
    being hashed into the step, these two arms would collapse to the same
    state id and the experiment would silently run one treatment twice.
    """
    specs = S.build_arm_specs(workdir_device="cpu")
    b1, b3 = specs["B_causal_b1"], specs["C_causal_b3"]
    assert b1.steps[-1].impl_id == b3.steps[-1].impl_id == "attention.causal_kl_v1"
    assert dict(b1.steps[-1].config) == {
        "calibration_forward_batch_size": 1,
        "calibration_batch_packing": "original_order_v1"}
    assert dict(b3.steps[-1].config) == {
        "calibration_forward_batch_size": 3,
        "calibration_batch_packing": "length_sorted_v1"}
    #: The whole point: different hashed steps, therefore different states.
    assert b1.steps[-1].as_dict() != b3.steps[-1].as_dict()
    assert b1.path_id != b3.path_id


def test_the_seeds_and_probe_count_come_from_the_plan():
    contract = S.C3SessionContract()
    assert list(S.recovery_seeds()) == PREREG["seeds"]["recovery"]
    assert S.bootstrap_seed() == PREREG["seeds"]["bootstrap"]
    assert contract.n_seeds == 3
    assert contract.n_arms == 3
    assert contract.n_probes == 9, "3 arms x 3 seeds = 9 probes, not 6"
    #: And they are DERIVED, not stored -- a literal cannot be assigned.
    with pytest.raises(AttributeError):
        contract.n_probes = 6                        # type: ignore[misc]


def test_the_shared_prefix_is_the_plan_s_and_is_not_restated():
    """One owner. A second copy of the prefix is a second thing to drift."""
    assert [list(p) for p in S.prefix_steps()] == PREREG["shared_parent"]["prefix"]
    src = (REPO / "scripts/experiments/phase_c3/session.py").read_text()
    assert "depth.causal_kl_greedy_v1" not in src.split('"""', 2)[2], (
        "the prefix implementations are literals in session.py again; they "
        "belong to the plan's shared_parent.prefix")


def test_the_replay_execution_config_is_pinned_by_the_plan():
    """No formal result may depend on whatever DEFAULT_MICRO_BATCH_SIZE is."""
    assert S.prefix_execution() == PREREG["shared_parent"]["execution"]
    assert S.prefix_execution()["micro_batch_size"] == 1


def test_the_digest_gates_come_from_the_plan(registered):
    specs = S.build_arm_specs(workdir_device="cpu")
    assert S.expected_parent_digest() == PREREG["shared_parent"]["artifact_digest"]
    assert S.expected_incumbent_digest() == PREREG["arms"]["A_incumbent"]["artifact_digest"]
    #: Every arm pins the parent on its last prefix step.
    for arm_id, spec in specs.items():
        assert spec.steps[len(S.prefix_steps()) - 1].expected_artifact_digest == \
            S.expected_parent_digest(), f"{arm_id} does not pin the shared parent"
    #: Only the incumbent pins an ATTENTION digest -- the causal arms are new
    #: states with no prior identity to gate against.
    assert specs["A_incumbent"].steps[-1].expected_artifact_digest == \
        S.expected_incumbent_digest()
    for arm_id in ("B_causal_b1", "C_causal_b3"):
        assert specs[arm_id].steps[-1].expected_artifact_digest is None


def test_the_contract_states_the_plan_it_was_built_from():
    """A contract that cannot name its plan cannot be checked against one."""
    d = S.C3SessionContract().as_dict()
    assert d["preregistration_sha256"] == PREREG["preregistration_sha256"]
    assert d["primary_contrast"] == PREREG["claim_boundary"]["primary_contrast"]
    assert "causal-B1" in d["primary_contrast"]
    assert d["arm_ids"] == list(S.arm_ids())
    assert d["n_probes"] == 9
    assert d["contains"]["arm_elimination"] is False


# ---------------------------------------------------------------------------
# Section 11: registration must match the arms
# ---------------------------------------------------------------------------

def test_every_implementation_the_arms_name_is_registered_by_stage_C(registered):
    from aadistill.initialization.operators.base import get_implementation

    S.register_experimental_operators()
    for impl_id in S.required_implementations():
        get_implementation(impl_id)                  # raises if absent
    #: Both experimental ATTENTION implementations, not just one.
    assert {"attention.activation_importance_v1",
            "attention.causal_kl_v1"} <= set(S.required_implementations())


def test_the_registered_set_is_derived_from_the_arms_not_listed():
    """A fourth arm must not be able to arrive unregistered.

    `required_implementations()` is built from `arm(...)["attention"][0]`
    across `arm_ids()`, so adding an arm to the plan adds its implementation
    here automatically. This asserts the derivation rather than the current
    contents: the current contents are what a stale literal would also match.
    """
    required = set(S.required_implementations())
    from_plan = {PREREG["arms"][a]["attention"][0]
                 for a in PREREG["arms"] if not a.startswith("_")}
    from_prefix = {impl for impl, _ in S.prefix_steps()}
    assert required == from_plan | from_prefix


def test_an_unknown_implementation_is_refused_rather_than_assumed(monkeypatch):
    """The registrar map is a whitelist, not a fallback."""
    monkeypatch.setattr(S, "required_implementations",
                        lambda: ("attention.not_a_real_operator_v9",))
    with pytest.raises(S.C3SessionError, match="no known registrar"):
        S.register_experimental_operators()


def test_build_arm_specs_refuses_before_registration(monkeypatch):
    """Ordering as a property of the code, not a line in a runbook."""
    monkeypatch.setattr(S, "required_implementations",
                        lambda: ("attention.definitely_unregistered_v0",))
    with pytest.raises(S.C3SessionError, match="not registered"):
        S.build_arm_specs(workdir_device="cpu")


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_the_arms_share_everything_but_their_last_step(registered):
    specs = S.build_arm_specs(workdir_device="cpu")
    assert S.arms_share_the_prefix(specs) is True
    n = len(S.prefix_steps())
    first = specs["A_incumbent"]
    for arm_id, spec in specs.items():
        assert spec.steps[:n] == first.steps[:n], f"{arm_id} diverges early"
        assert len(spec.steps) == n + 1


def test_arms_share_the_prefix_handles_three_not_just_two(registered):
    """The old form unpacked two keys by name and would have raised here."""
    specs = S.build_arm_specs(workdir_device="cpu")
    assert len(specs) == 3
    assert S.arms_share_the_prefix(specs) is True
    #: Two identical arms are not two arms.
    assert S.arms_share_the_prefix(
        {"x": specs["B_causal_b1"], "y": specs["B_causal_b1"]}) is False
    #: A single arm is not a contrast.
    assert S.arms_share_the_prefix({"x": specs["A_incumbent"]}) is False


def test_a_diverging_arm_is_detected(registered):
    specs = S.build_arm_specs(workdir_device="cpu")
    import dataclasses

    bad = dataclasses.replace(
        specs["C_causal_b3"],
        steps=(specs["C_causal_b3"].steps[0],) + specs["C_causal_b3"].steps[1:-1]
              + (specs["C_causal_b3"].steps[-1],))
    #: same object shape -- still shares. Now actually diverge the prefix.
    assert S.arms_share_the_prefix({**specs, "C_causal_b3": bad}) is True
    diverged = dataclasses.replace(
        specs["C_causal_b3"], steps=specs["C_causal_b3"].steps[1:])
    assert S.arms_share_the_prefix({**specs, "C_causal_b3": diverged}) is False


def test_the_stages_describe_nine_probes_and_the_right_incumbent():
    """Stage prose is read by humans deciding what the session does."""
    assert "9 probes" in S.stage("G").description
    assert "3 arms x 3 fresh seeds" in S.stage("G").description
    assert "attention.activation_importance_v1" in S.stage("E").description
    assert "weight_proxy" not in json.dumps(
        [s.as_dict() for s in S.C3_STAGES]), (
        "a stage still names weight_proxy_v0; that was C1's incumbent, not C3's")
    assert "all 9 results" in S.stage("I").description
    #: The gates that must stop the session before any paid training.
    assert set(S.GATE_STAGES) >= {"teacher_fetch_verify", "register_operator",
                                  "replay_parent", "replay_incumbent",
                                  "materialize_arms"}


def test_the_stage_order_refuses_a_gap():
    S.assert_stage_order(["session_setup", "teacher_fetch_verify"])
    with pytest.raises(S.C3SessionError, match="out of order"):
        S.assert_stage_order(["session_setup", "replay_parent"])


def test_a_plan_whose_stamp_does_not_bind_is_refused(tmp_path):
    """Building a formal experiment from an unverified plan is the failure
    this module exists to prevent; it must be an error, not a warning."""
    doc = dict(PREREG)
    doc["preregistration_sha256"] = "0" * 64
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(S.C3SessionError, match="does not bind itself"):
        S.load_preregistration(p)
