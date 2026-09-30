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


# ---------------------------------------------------------------------------
# The LAUNCHER's statement of the same experiment
#
# The tests above bind the plan to the EXECUTOR. The launcher makes its own
# statement, into the session record, and nothing bound that: its
# `evidence_fields` read a literal `arms: 2, seeds: 3, probes: 6` until
# 2026-09-30, so attempt66's session.json says it ran two arms and six probes
# for a session that trained NINE across three. Same defect, one file over.
# ---------------------------------------------------------------------------


def _launcher():
    import importlib.util

    for p in ("scripts/pod", "scripts/autoinit"):
        if str(REPO / p) not in sys.path:
            sys.path.insert(0, str(REPO / p))
    spec = importlib.util.spec_from_file_location(
        "c3launch_evidence", REPO / "scripts/pod/autoinit_c3_launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["c3launch_evidence"] = mod
    spec.loader.exec_module(mod)
    return mod


def _real_args(mod):
    """The namespace the launcher's REAL parser produces.

    Never a hand-written stub: device-canary attempt 1 died at `$0.0603` on an
    attribute a hand-written namespace happened to have and the real parser did
    not, and a stub here would hide exactly that class of defect again.
    """
    parser = mod.build_parser()
    return parser.parse_args(["--scr", "/tmp/c3-spec-check",
                              "--session-commit", "0" * 40,
                              "--bundle", "aad_test.bundle",
                              "--run-id", "spec_check"])


def test_the_launcher_reports_the_probe_count_the_plan_declares():
    """A run record that contradicts its own measurement is not evidence."""
    mod = _launcher()
    fields = mod.spec(_real_args(mod)).evidence_fields
    assert fields["arms"] == len(S.arm_ids()) == 3
    assert fields["seeds"] == len(S.recovery_seeds()) == 3
    assert fields["probes"] == 9, (
        "the launcher states a probe count the plan does not; attempt66's "
        "session.json says 6 for a run that trained 9")


def test_the_capacity_gate_refuses_when_the_backend_would_refuse(monkeypatch):
    """attempt66 trained nine probes for $13.67 and preserved none.

    Three outcomes must be distinguishable, because they need different
    responses: room, no room, and could-not-ask. Only the first may launch.
    """
    mod = _launcher()
    from aadistill.runtime.hub_capacity import CapacityVerdict

    def verdicts(*outcomes):
        seq = list(outcomes)

        def fake(repo, sizes, token, **kw):
            ok = seq.pop(0)
            return CapacityVerdict(ok, 200 if ok else 403,
                                   "fits" if ok else "Private repository "
                                   "storage limit reached",
                                   len(sizes), sum(sizes))
        return fake

    import aadistill.runtime.hub_capacity as HC

    monkeypatch.setattr(HC, "would_accept", verdicts(True, True))
    assert mod.durable_capacity_gate(None)[0] is True

    #: One probe fits, nine do not — the exact shape that would train every
    #: probe and lose the ones after the quota ran out.
    monkeypatch.setattr(HC, "would_accept", verdicts(True, False))
    ok, why = mod.durable_capacity_gate(None)
    assert ok is False and "refuse all 9" in why

    monkeypatch.setattr(HC, "would_accept", verdicts(False, True))
    ok, why = mod.durable_capacity_gate(None)
    assert ok is False and "ONE probe" in why

    #: Could not ask is NOT permission to spend.
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(HC, "would_accept", boom)
    ok, why = mod.durable_capacity_gate(None)
    assert ok is False and "could not ask" in why


def test_the_capacity_gate_asks_at_the_real_per_probe_size():
    """A gate threshold that is a guess is a gate that passes by luck.

    2,384,236,592 bytes is `bytes` from attempt66's own preservation payload,
    reported identically by all nine probes.
    """
    mod = _launcher()
    measured = json.loads((
        REPO / "logs/stages/stage-1/phase_c3/runs/attempt66/evidence/probes"
        / "autoinit.v1.phase_c3.A_incumbent.217230555.training.json").read_text())
    assert mod.PROBE_DURABLE_BYTES == measured["preserved"]["bytes"]
    #: And that record is the one that FAILED, which is the point: the bytes
    #: were measured, the upload was refused, and nothing asked beforehand.
    assert measured["preserved"]["preserved"] is False
    assert "storage limit reached" in measured["preserved"]["why_not"]


def test_the_capacity_gate_is_wired_into_the_prechecks():
    """A gate nothing calls is a gate that did not run. attempt66's lesson was
    not a missing check; it was a check that existed and was never asked."""
    mod = _launcher()
    names = {getattr(g, "__name__", "") for g in mod.spec(_real_args(mod)).precheck}
    assert "durable_capacity_gate" in names


# ---------------------------------------------------------------------------
# Inference parameters are PASSED, never inherited
#
# `isolation.py` carries C1's values as defaults. Four of C3's five coincide
# exactly -- sesoi 0.010, 20,000 iterations, 2-of-3 seed robustness, the two
# usable-rollout guardrails -- and the fifth, the bootstrap seed, does not:
# `phase-c1:bootstrap` is 816109261 and `phase-c3:bootstrap` is 654678655.
# The coinciding four are what made the fifth invisible, so these tests assert
# the values are CARRIED rather than merely equal.
# ---------------------------------------------------------------------------


def test_the_frozen_plan_carries_c3s_thresholds_not_c1s_defaults():
    """What `decide` reads is the PLAN, so the plan must carry C3's numbers.

    EQUALITY IS NOT ENOUGH and asserting it is the trap. C1's defaults equal
    C3's values for every one of these, so a plan that stopped passing them
    would still satisfy `plan.sesoi == 0.01`. Verified: deleting `sesoi=`
    from the constructor left an equality-only version of this test green.
    So each threshold is MOVED in the preregistration and the plan is
    required to follow it.
    """
    plan = S.frozen_isolation_plan()
    inf, guard = S.inference(), S.guardrails()
    assert plan.sesoi == inf["sesoi"] == 0.01
    assert plan.usable_pooled_min_delta == guard["pooled_usable_delta_min"]
    assert plan.usable_per_seed_min_delta == guard["per_seed_usable_delta_min"]
    assert tuple(plan.seeds) == S.recovery_seeds()
    assert [a.role for a in plan.arms] == ["incumbent", "treatment"]

    moved = json.loads(json.dumps(PREREG))
    moved["inference"]["sesoi"] = 0.012
    moved["guardrails"]["pooled_usable_delta_min"] = -0.011
    moved["guardrails"]["per_seed_usable_delta_min"] = -0.022
    saved = S._PREREG
    try:
        S._PREREG = moved
        followed = S.frozen_isolation_plan()
        assert followed.sesoi == 0.012, (
            "the plan did not follow a moved SESOI, so it is inheriting "
            "phase-C1's default and only looked right because the two agree")
        assert followed.usable_pooled_min_delta == -0.011
        assert followed.usable_per_seed_min_delta == -0.022
    finally:
        S._PREREG = saved


def test_the_bootstrap_seed_is_c3s_and_differs_from_c1s():
    """The one parameter that does NOT coincide, asserted as not coinciding."""
    from experiments.phase_c1.isolation import bootstrap_seed as c1_seed

    assert S.inference()["seed"] == S.bootstrap_seed() == 654678655
    assert c1_seed() == 816109261
    assert S.inference()["seed"] != c1_seed(), (
        "the two phases' bootstrap seeds now coincide, so this test can no "
        "longer show that C3's is passed rather than inherited; assert the "
        "call site instead")


def test_the_plan_refuses_to_disagree_with_itself_about_the_seed():
    """The preregistration states the seed twice. Both must be read."""
    doc = json.loads(json.dumps(PREREG))
    doc["inference"]["seed"] = doc["seeds"]["bootstrap"] + 1
    saved = S._PREREG
    try:
        S._PREREG = doc
        with pytest.raises(S.C3SessionError, match="disagrees with itself"):
            S.inference()
    finally:
        S._PREREG = saved


@pytest.mark.parametrize("missing", ["method", "iterations", "seed", "sesoi"])
def test_a_missing_inference_parameter_refuses_rather_than_defaulting(missing):
    """Falling back to another phase's default is the defect, not the fix."""
    doc = json.loads(json.dumps(PREREG))
    doc["inference"].pop(missing)
    saved = S._PREREG
    try:
        S._PREREG = doc
        with pytest.raises(S.C3SessionError, match="missing"):
            S.inference()
    finally:
        S._PREREG = saved


def test_the_aggregation_passes_both_seed_and_iterations():
    """A defaulted argument is invisible to every digest and import check.

    Asked by AST of both call sites -- the pod driver and the off-pod
    aggregation -- because this is exactly the class of defect that reached a
    paid pod twice.
    """
    import ast as _ast

    for rel in ("scripts/pod/autoinit_c3_driver.py",
                "scripts/autoinit/aggregate_c3_stage_i.py"):
        tree = _ast.parse((REPO / rel).read_text())
        calls = [n for n in _ast.walk(tree)
                 if isinstance(n, _ast.Call)
                 and getattr(n.func, "id", "") == "stratified_cluster_bootstrap"]
        assert calls, f"{rel} never calls the bootstrap"
        for call in calls:
            kw = {k.arg for k in call.keywords}
            assert {"seed", "iterations"} <= kw, (
                f"{rel} calls stratified_cluster_bootstrap without "
                f"{sorted({'seed', 'iterations'} - kw)}; it would inherit "
                "phase-C1's default")
