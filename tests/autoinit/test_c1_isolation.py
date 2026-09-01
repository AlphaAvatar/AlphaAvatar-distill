"""The C1 isolation plan, its seed derivation, and the frozen decision rule."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aadistill.autoinit.c1_isolation import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    C0_PREREGISTRATION_SHA256,
    HISTORICAL_SEEDS,
    C1Arm,
    C1IsolationPlan,
    C1PlanError,
    assert_preregistered,
    bootstrap_seed,
    decide,
    derive_recovery_seeds,
    paired_differences,
    stratified_cluster_bootstrap,
)

INCUMBENT = C1Arm("c1.incumbent", "incumbent", "attention.weight_proxy_v0",
                  "calib.none@v1")
TREATMENT = C1Arm("c1.treatment", "treatment", "attention.activation_importance_v1",
                  "calib.domain_balanced@v1")


def plan(**kw) -> C1IsolationPlan:
    base = dict(plan_id="autoinit.v1.phase_c1", arms=(INCUMBENT, TREATMENT),
                seeds=tuple(derive_recovery_seeds()),
                battery_asset_id="c1_confirmation_v1",
                battery_content_sha256="0" * 64)
    base.update(kw)
    return C1IsolationPlan(**base)


# --- structure: illegal states are unrepresentable --------------------------

def test_it_refuses_anything_other_than_two_arms():
    with pytest.raises(C1PlanError, match="two-arm"):
        plan(arms=(INCUMBENT,))
    third = C1Arm("c1.other", "treatment", "attention.causal_kl_v1", "calib.none@v1")
    with pytest.raises(C1PlanError, match="two-arm"):
        plan(arms=(INCUMBENT, TREATMENT, third))


def test_it_requires_one_incumbent_and_one_treatment():
    two_treatments = (TREATMENT, C1Arm("c1.b", "treatment", "x", "calib.none@v1"))
    with pytest.raises(C1PlanError, match="exactly one incumbent"):
        plan(arms=two_treatments)


def test_it_refuses_two_arms_that_isolate_nothing():
    same = C1Arm("c1.copy", "treatment", "attention.weight_proxy_v0", "calib.none@v1")
    with pytest.raises(C1PlanError, match="nothing is being isolated"):
        plan(arms=(INCUMBENT, same))


def test_it_refuses_any_seed_count_but_three():
    with pytest.raises(C1PlanError, match="exactly 3"):
        plan(seeds=(1, 2))
    with pytest.raises(C1PlanError, match="exactly 3"):
        plan(seeds=(1, 2, 3, 4))
    with pytest.raises(C1PlanError, match="distinct"):
        plan(seeds=(7, 7, 9))


@pytest.mark.parametrize("historical", HISTORICAL_SEEDS)
def test_it_refuses_the_phase_a_b_selection_seeds(historical):
    fresh = derive_recovery_seeds()
    with pytest.raises(C1PlanError, match="winner's-curse"):
        plan(seeds=(historical, fresh[1], fresh[2]))


def test_elimination_is_absent_from_the_type_not_merely_rejected():
    """No `survivors`, `rungs` or `tie_break_seed` field exists to be set."""
    fields = set(C1IsolationPlan.__dataclass_fields__)
    for forbidden in ("survivors", "rungs", "tie_break_seed", "equivalence",
                      "searched_leaves", "feasibility"):
        assert forbidden not in fields, f"{forbidden} is representable in C1"
    p = plan()
    assert p.probe_count == 6                     # 2 arms x 3 seeds, always
    d = p.as_dict()["structure"]
    assert d == {"successive_halving": False, "elimination_rung": False,
                 "tie_break_rung": False, "search_ranking": False,
                 "both_arms_run_every_seed": True}


def test_it_does_not_subclass_the_phase_a_b_plan():
    from aadistill.autoinit.recovery import SuccessiveHalvingPlan
    assert not issubclass(C1IsolationPlan, SuccessiveHalvingPlan)


def test_the_plan_hash_covers_every_frozen_threshold(tmp_path):
    p = plan()
    frozen = p.freeze(tmp_path / "plan.json")
    assert assert_preregistered(p, frozen)["plan_hash"] == p.plan_hash
    # Each of these is a *legal* plan that differs only in a frozen threshold,
    # so the hash — not the constructor — is what has to catch it.
    for changed in (dict(sesoi=0.012), dict(alpha=0.10),
                    dict(usable_pooled_min_delta=-0.20),
                    dict(seed_robustness_min_positive=1),
                    dict(catastrophic_candidate_max=0.5)):
        with pytest.raises(C1PlanError, match="does not match"):
            assert_preregistered(plan(**changed), frozen)


# --- seed derivation --------------------------------------------------------

def test_seed_derivation_is_deterministic_and_fresh():
    a, b = derive_recovery_seeds(), derive_recovery_seeds()
    assert a == b and len(a) == 3 == len(set(a))
    assert not set(a) & set(HISTORICAL_SEEDS)
    assert all(0 <= s < 2 ** 31 for s in a)


def test_seed_derivation_is_domain_separated_from_the_bootstrap_seed():
    assert bootstrap_seed() not in derive_recovery_seeds()


def test_seed_derivation_skips_a_collision_deterministically():
    a = derive_recovery_seeds()
    without = derive_recovery_seeds(exclude=(a[0],))
    assert a[0] not in without and len(without) == 3


def test_it_is_bound_to_the_pushed_c0_digest():
    assert C0_PREREGISTRATION_SHA256 == (
        "fb2eeea531f9f0d11f84b77cd47dff30697122de90a072a7a80c3a7535e89280")


# --- the estimand -----------------------------------------------------------

def test_d_j_is_the_seed_mean_of_the_paired_difference():
    inc = {1: {"a": False, "b": True}, 2: {"a": False, "b": True},
           3: {"a": True, "b": True}}
    trt = {1: {"a": True, "b": True}, 2: {"a": True, "b": False},
           3: {"a": True, "b": True}}
    d = paired_differences(inc, trt)
    assert d["a"] == pytest.approx(2 / 3)         # gained on 2 of 3 seeds
    assert d["b"] == pytest.approx(-1 / 3)        # lost on 1 of 3


def test_an_incomplete_design_is_refused_rather_than_averaged():
    inc = {1: {"a": True, "b": False}}
    with pytest.raises(C1PlanError, match="prompt set differs"):
        paired_differences(inc, {1: {"a": True}})
    with pytest.raises(C1PlanError, match="different seeds"):
        paired_differences(inc, {2: {"a": True, "b": False}})


def test_the_bootstrap_is_reproducible_and_states_its_claim_boundary():
    d = {f"p{i}": (1.0 if i % 5 == 0 else 0.0) for i in range(200)}
    strata = {k: ("A" if int(k[1:]) < 100 else "B") for k in d}
    a = stratified_cluster_bootstrap(d, strata, iterations=2000)
    b = stratified_cluster_bootstrap(d, strata, iterations=2000)
    assert a == b
    assert a["n_strata"] == 2 and a["n_prompts"] == 200
    assert "NOT a CI over hypothetical future recovery seeds" in a["claim_boundary"]
    assert a["lcb_one_sided"] <= a["delta"] <= a["ucb_one_sided"]
    assert a["seed"] == bootstrap_seed()


def test_the_bootstrap_refuses_a_prompt_with_no_stratum():
    with pytest.raises(C1PlanError, match="no stratum"):
        stratified_cluster_bootstrap({"a": 1.0}, {}, iterations=10)


def test_bootstrap_details_are_bound_before_any_c1_data_exist():
    assert BOOTSTRAP_ITERATIONS == 20_000
    d = {f"p{i}": 0.0 for i in range(10)}
    out = stratified_cluster_bootstrap(d, {k: "A" for k in d}, iterations=100)
    for key in ("algorithm", "quantile_convention", "stratum_convention",
                "seed", "iterations"):
        assert out[key], key


# --- the decision rule ------------------------------------------------------

def boot(delta, lcb, ucb):
    return {"delta": delta, "lcb_one_sided": lcb, "ucb_one_sided": ucb,
            "ci_two_sided_low": lcb, "ci_two_sided_high": ucb,
            "claim_boundary": "conditional"}


def call(p, b, per_seed=(0.02, 0.01, -0.001), pooled_u=0.0,
         per_seed_u=(0.0, 0.0, 0.0), violations=()):
    return decide(p, boot=b, per_seed_delta=per_seed,
                  usable_pooled_delta=pooled_u,
                  usable_per_seed_delta=per_seed_u,
                  catastrophic_violations=violations)


def test_go_requires_every_clause():
    p = plan()
    assert call(p, boot(0.015, 0.004, 0.026))["verdict"] == "GO"
    # LCB not above zero
    assert call(p, boot(0.015, -0.001, 0.031))["verdict"] == "INCONCLUSIVE"
    # point below SESOI
    assert call(p, boot(0.009, 0.002, 0.016))["verdict"] == "INCONCLUSIVE"
    # only one of three seeds positive
    assert call(p, boot(0.015, 0.004, 0.026),
                per_seed=(0.05, -0.01, -0.02))["verdict"] == "INCONCLUSIVE"


def test_no_go_on_an_excluded_sesoi_or_any_veto():
    p = plan()
    assert call(p, boot(0.001, -0.004, 0.006))["verdict"] == "NO-GO"
    # a veto overrides an otherwise passing GO
    r = call(p, boot(0.015, 0.004, 0.026), pooled_u=-0.06)
    assert r["verdict"] == "NO-GO" and r["vetoes"]
    r = call(p, boot(0.015, 0.004, 0.026), per_seed_u=(0.0, -0.11, 0.0))
    assert r["verdict"] == "NO-GO"
    r = call(p, boot(0.015, 0.004, 0.026),
             violations=({"reason": "tool collapsed"},))
    assert r["verdict"] == "NO-GO" and "tool collapsed" in r["vetoes"][0]


def test_inconclusive_is_the_default_and_no_winner_is_forced():
    p = plan()
    r = call(p, boot(0.005, -0.002, 0.012))
    assert r["verdict"] == "INCONCLUSIVE"
    assert r["no_forced_winner"] is True


def test_the_boundary_case_is_reported_exactly():
    """Delta exactly at the SESOI with an LCB just above zero is a GO; a hair
    below is not. The rule is an inequality, and it is applied as written."""
    p = plan()
    assert call(p, boot(0.010, 0.0001, 0.020))["verdict"] == "GO"
    assert call(p, boot(0.00999, 0.0001, 0.020))["verdict"] == "INCONCLUSIVE"


def test_it_refuses_a_delta_vector_that_does_not_cover_every_seed():
    p = plan()
    with pytest.raises(C1PlanError, match="seed-specific deltas"):
        call(p, boot(0.015, 0.004, 0.026), per_seed=(0.01, 0.01))
