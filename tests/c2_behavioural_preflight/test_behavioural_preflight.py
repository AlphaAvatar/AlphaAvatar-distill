"""Everything the paid behavioural session can fail before its first arm builds.

This is the pod selection: the blocking gate the session runs on the pod, and
the sweep a launch-bound readiness record describes. It is deliberately written
with NO conditional skips — a `skipif` keyed on a simulator-set marker is
INVERTED on the pod, and a readiness gate can check THAT a test skipped but
never WHY.

Everything here is `$0` and takes seconds. The point is that a protocol that
does not parse, a seed that falls back to C1's, an arm whose root is not the
verified teacher, a battery that does not match its identity record or a
schedule that would rank a partial field all fail HERE, rather than after six
arms and nine probes have been paid for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts", "scripts/autoinit"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from experiments.phase_c2 import behavioural as BH
from experiments.phase_c2 import behavioural_decision as BD
from experiments.phase_c2 import behavioural_governance as BG
from experiments.phase_c2 import behavioural_schedule as SCH
from experiments.phase_c2 import scoring as C2S


# -- the protocol is the plan ------------------------------------------------

def test_protocol_matches_its_own_hash():
    """`BH.protocol` refuses a protocol edited since it was frozen."""
    doc = BH.protocol(REPO)
    assert doc["protocol_sha256"]
    assert "behavioural_selection" in doc


def test_schedule_is_twelve_probes_in_two_rungs():
    sched = BH.schedule(REPO)
    assert sched["total_probes"] == 12
    assert sched["screening"]["probes"] == 6
    assert sched["confirmation"]["probes"] == 6
    assert sched["advanced_candidates"] == 1
    assert sched["conditional_rung"] is None


# -- the decision rule -------------------------------------------------------

def test_decision_rule_reads_the_three_frozen_confirmation_seeds():
    rule = BD.decision_rule(REPO)
    assert rule.seeds == (1936324010, 1916380711, 1523147638)
    assert len(set(rule.seeds)) == 3


def test_decision_rule_thresholds_come_from_the_protocol():
    rule = BD.decision_rule(REPO)
    assert rule.sesoi == 0.01
    assert rule.seed_robustness_min_positive == 2
    assert rule.usable_pooled_min_delta == -0.05
    assert rule.usable_per_seed_min_delta == -0.10
    assert rule.bootstrap_iterations == 20000


def test_c2_bootstrap_seed_is_not_c1s_default():
    """The fallback the confirmation must never take.

    `stratified_cluster_bootstrap` defaults to `bootstrap_seed()`, which
    domain-separates under `:phase-c1:bootstrap`. C2 froze its own under
    `:phase-c2:bootstrap`. If these ever coincided, passing the seed explicitly
    would stop being a check on anything.
    """
    from experiments.phase_c1.isolation import bootstrap_seed

    rule = BD.decision_rule(REPO)
    assert rule.bootstrap_seed == 834816710
    assert rule.bootstrap_seed != bootstrap_seed()


def test_screening_seed_is_disjoint_from_the_confirmation_seeds():
    proto = BH.protocol(REPO)["behavioural_selection"]
    screening = set(proto["seeds"]["screening"])
    confirmation = set(proto["seeds"]["confirmation"])
    assert screening and confirmation
    assert not (screening & confirmation)


# -- the six arms ------------------------------------------------------------

def test_all_six_arms_build_and_pin_every_step():
    specs = BG.arm_specs(REPO)
    assert len(specs) == 6, "five candidates and the incumbent B"
    for spec in specs:
        assert len(spec.steps) == 4


def test_every_arm_root_is_the_verified_teacher_binding():
    """A path built from a different root diverges at its first step."""
    pinned = json.loads(
        (REPO / "logs/stages/stage-1/phase_c1/plans/teacher_binding.json"
         ).read_text())
    for spec in BG.arm_specs(REPO):
        assert spec.root_repo_id == pinned["repo_id"]
        assert spec.root_revision == pinned["revision"]


def test_candidate_order_matches_the_frozen_selection():
    """The screening tie-break IS this ordering."""
    from experiments.phase_c2.replay_specs import load_selection

    order = [s["state_id"] for s in load_selection(REPO)["selected"]]
    assert [leaf.state_id for leaf in BG.candidate_leaves(REPO)] == order


def test_incumbent_b_construction_matches_c1s_frozen_spec_hash():
    binding = BH.b_binding(REPO, device="cuda")
    c = binding["construction"]
    assert c["spec_hash"] == c["expected_spec_hash"]


def test_every_candidate_carries_the_identity_its_gate_needs():
    for leaf in BG.candidate_leaves(REPO):
        assert leaf.artifact_digest
        assert leaf.weights_digest
        assert leaf.arch_signature
        assert leaf.num_parameters > 0


# -- the batteries -----------------------------------------------------------

def test_screening_battery_validates_against_its_identity_record():
    manifest = json.loads(
        (REPO / C2S.BATTERY_PATH / "manifest.json").read_text())
    identity = C2S.validate_screening_battery(manifest, repo_root=REPO)
    assert identity["role"] == "C2_SCREENING"
    assert identity["n_prompts"] == 950
    assert identity["n_scorable_prompts"] == 850


def test_confirmation_battery_validates_against_c1s_pins():
    from experiments.phase_c1.scoring import validate_c1_battery
    from aadistill.infrastructure.manifest import sha256_json

    path = REPO / "artifacts/stage3/c1_confirmation_v1/manifest.json"
    manifest = json.loads(path.read_text())
    sha = sha256_json({k: v for k, v in manifest.items()
                       if k != "manifest_sha256"})
    identity = validate_c1_battery(manifest, manifest_sha256=sha)
    assert identity["content_sha256"]


def test_the_two_batteries_are_different_assets():
    """Same mixture, disjoint prompts. Sharing content would leak selection."""
    screening = json.loads(
        (REPO / C2S.BATTERY_PATH / "manifest.json").read_text())
    confirmation = json.loads(
        (REPO / "artifacts/stage3/c1_confirmation_v1/manifest.json").read_text())
    assert screening["content_sha256"] != confirmation["content_sha256"]
    assert screening["n_prompts"] == confirmation["n_prompts"]
    assert screening["n_scorable_prompts"] == confirmation["n_scorable_prompts"]


def test_screening_validator_refuses_the_confirmation_battery():
    """The pins are real, not decorative."""
    manifest = json.loads(
        (REPO / "artifacts/stage3/c1_confirmation_v1/manifest.json").read_text())
    with pytest.raises(C2S.C2ScoringError):
        C2S.validate_screening_battery(manifest, repo_root=REPO)


def test_c1_scorer_refuses_the_screening_battery():
    """C1's production path cannot be aimed at the screening rung."""
    from experiments.phase_c1.scoring import C1ScoringError, validate_c1_battery
    from aadistill.infrastructure.manifest import sha256_json

    manifest = json.loads(
        (REPO / C2S.BATTERY_PATH / "manifest.json").read_text())
    sha = sha256_json({k: v for k, v in manifest.items()
                       if k != "manifest_sha256"})
    with pytest.raises(C1ScoringError):
        validate_c1_battery(manifest, manifest_sha256=sha)


# -- the schedule ------------------------------------------------------------

def _arm(state_id, rank=0):
    return {"state_id": state_id, "artifact_digest": f"d{state_id}",
            "durable_path": f"/p/{state_id}", "rank_in_frozen_selection": rank}


def test_screening_refuses_more_than_one_seed():
    cands = [_arm(f"c{i}", i) for i in range(5)]
    with pytest.raises(SCH.ScheduleError):
        SCH.screening_probes(cands, _arm("B"), [1, 2])


def test_confirmation_refuses_the_anchor_as_the_advanced_candidate():
    with pytest.raises(SCH.ScheduleError):
        SCH.confirmation_probes(_arm(SCH.ANCHOR), _arm(SCH.ANCHOR), [1, 2, 3])


def test_ranking_refuses_a_field_missing_the_anchor():
    cands = [_arm(f"c{i}", i) for i in range(5)]
    scores = {f"c{i}": 0.5 for i in range(5)}
    with pytest.raises(SCH.ScheduleError):
        SCH.rank_screening(scores, cands)


def test_ranking_refuses_a_partial_field():
    cands = [_arm(f"c{i}", i) for i in range(5)]
    scores = {SCH.ANCHOR: 0.5, "c0": 0.6, "c1": 0.55}
    with pytest.raises(SCH.ScheduleError):
        SCH.rank_screening(scores, cands)


def test_confirmation_may_not_begin_before_every_screening_probe_is_scored():
    cands = [_arm(f"c{i}", i) for i in range(5)]
    probes = SCH.screening_probes(cands, _arm("B"), [616738081])
    trained = {p.probe_id: {} for p in probes}
    scored = {p.probe_id: {} for p in probes[:-1]}
    ok, why = SCH.screening_is_complete(probes, trained, scored)
    assert not ok and "not scored" in why


def test_screening_may_not_emit_a_verdict():
    for verdict in ("GO", "NO_GO", "INCONCLUSIVE"):
        with pytest.raises(SCH.ScheduleError):
            SCH.assert_screening_emits_no_verdict({"verdict": verdict})


# -- the money ---------------------------------------------------------------

def test_the_plan_fits_its_derived_ceiling():
    c = BG.ceiling(REPO)
    assert c["expected"]["all_in_usd"] < c["hard_ceiling"]["all_in_usd"]
    assert c["materialization"]["n_candidates"] == 5
    #: The preparation term covers all six arms, not only B.
    assert (c["materialization"]["total_minutes"]
            > c["materialization"]["incumbent_b_minutes"])


def test_the_window_is_derived_from_the_rate_not_fixed():
    """A deadline set without reference to price can outlive the budget.

    Both bounds come from the AUTHORIZATION's own amounts rather than from a
    constant rate; `test_behavioural_launch_corrections.py` covers the runtime
    cap and the refusals. Here it is only the monotonicity: a dearer card must
    buy fewer minutes.
    """
    terms = BG.authorization_terms(
        REPO, rate_usd_per_hour=BG.QUOTED_RATE_USD_PER_HOUR)
    kw = {"gpu_hard_usd": terms["gpu_hard_usd"],
          "hard_runtime_minutes": terms["hard_runtime_minutes"]}
    assert BG.window_minutes(1.5, **kw) > BG.window_minutes(2.0, **kw)


# -- the resume policy is registered, and its rules name real enforcement ----

RESUME = ("logs/stages/stage-1/phase_c2_behavioural/plans/"
          "c2_behavioural_resume_preregistration.json")


def test_the_resume_policy_is_registered_and_matches_its_own_hash():
    """Registered BEFORE the run it governs. A reuse rule written after seeing
    a result is a rule chosen by the result."""
    from aadistill.infrastructure.manifest import sha256_json

    doc = json.loads((REPO / RESUME).read_text())
    assert doc["authorizes"] == "nothing"
    recomputed = sha256_json({k: v for k, v in doc.items()
                              if k != "record_sha256"})
    assert recomputed == doc["record_sha256"], (
        "the resume preregistration has been edited since it was registered")


def test_every_resume_rule_names_enforcement_that_exists():
    """A rule whose enforcement moved is a rule nobody applies."""
    doc = json.loads((REPO / RESUME).read_text())
    assert len(doc["rules"]) >= 6
    for rule in doc["rules"]:
        assert rule["why"], rule["id"]
        target = rule["enforced_by"]
        path = target.split("::")[0].split("—")[0].strip()
        found = list(REPO.rglob(path)) if "/" not in path else [REPO / path]
        assert any(p.exists() for p in found), (
            f"{rule['id']} says it is enforced by {path}, which does not exist")
        if "::" in target:
            symbol = target.split("::")[1].split("—")[0].strip()
            source = next(p for p in found if p.exists()).read_text()
            assert f"def {symbol}" in source, (
                f"{rule['id']} names {symbol}, which {path} does not define")


def test_the_catastrophic_veto_thresholds_are_read_not_repeated():
    """One owner. The protocol records the veto verbatim from its implementation."""
    proto = BH.protocol(REPO)["behavioural_selection"]
    veto = (proto["frozen_science"]["behavioural_guardrails"]
            ["catastrophic_capability_veto"])
    rule = BD.decision_rule(REPO)
    assert rule.catastrophic_candidate_max == float(veto["candidate_max"])
    assert rule.catastrophic_control_min == float(veto["control_min"])
    #: And the operands are NAMED rather than inherited, as the protocol requires.
    assert veto["control_operand"].startswith("the INCUMBENT")
