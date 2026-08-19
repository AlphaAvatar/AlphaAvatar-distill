"""The bounded measurement session: what it is, and what it cannot become.

A new paid session, so it gets the same structural scrutiny as the other four
plus the properties specific to it: it must not be able to start Phase A, must
run no search, select nothing and write no checkpoint, and its authorization must
be a fresh one-use artifact rather than a reinterpreted Phase-A grant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from session_specs import load_session_launcher, session_args  # noqa: E402

NAME = "autoinit_measurement_launch"


@pytest.fixture(scope="module")
def spec():
    mod = load_session_launcher(NAME)
    return mod.spec(session_args(mod))


def test_it_validates_and_names_itself(spec):
    spec.validate()
    assert spec.session_id == "autoinit-measurement"
    assert spec.schema == "aadistill.autoinit.measurement_session/v1"


def test_it_cannot_start_phase_a(spec):
    """A property of the type, not a promise. `SpendAuthorization.allows_phase_a`
    is a hard False, so this session refuses a Phase-A artifact rather than
    running it."""
    # By property, not by object identity: the launcher and this test resolve
    # `SpendAuthorization` through different sys.path entries, so `is` compares
    # two class objects that are the same class by every meaning that matters.
    loader = spec.authorization_loader
    assert loader.__qualname__ == "SpendAuthorization.load"
    owner = loader.__self__                       # the class the loader belongs to
    assert owner.__name__ == "SpendAuthorization"
    # `allows_phase_a` is a property, so it must be read off an INSTANCE; on the
    # class it is the property object and `is False` would be vacuously wrong.
    from aadistill.autoinit.measurement import MEASUREMENT_AUTHORIZATION
    assert MEASUREMENT_AUTHORIZATION.allows_phase_a is False, (
        "the measurement's authorization can authorize Phase A")
    assert spec.evidence_fields["phase_a_reachable_from_this_launcher"] is False
    assert spec.evidence_fields["phase_a_launched"] is False


def test_it_uses_its_own_authorization_and_plan_not_phase_as(spec):
    from aadistill.autoinit.measurement import MEASUREMENT_PLAN_V1
    from aadistill.autoinit.phase_a import PHASE_A_PLAN_V1

    assert spec.plan_hash == MEASUREMENT_PLAN_V1.plan_hash
    assert spec.plan_hash != PHASE_A_PLAN_V1.plan_hash
    assert spec.authorization_path == "logs/autoinit_measurement_authorization.json"
    assert "phase_a" not in spec.authorization_path


def test_it_declares_that_it_produces_nothing_scientific(spec):
    e = spec.evidence_fields
    for field in ("trains_anything", "runs_greedy_search", "selects_a_depth_map",
                  "writes_a_checkpoint", "retrains_permanent_controls",
                  "followon_started", "scientific_use"):
        assert e[field] is False, field
    assert spec.artifacts.fetch_products(None) == [], (
        "the measurement fetches products; it produces none")


def test_the_driver_runs_the_reviewed_job_with_its_reviewed_defaults(spec):
    cmd = spec.driver_command(None, None)
    assert "measure_causal_depth_runtime.py" in cmd
    assert "audit/autoinit_measurement/result.json" in cmd
    # The defaults ARE the reviewed design; overriding them here would mean the
    # reviewed job and the executed job are different jobs.
    for override in ("--samples-per-cardinality", "--e8a-pairs",
                     "--teacher-revision", "--teacher"):
        assert override not in cmd, f"the driver overrides {override}"
    assert "greedy" not in cmd and "search" not in cmd.replace(
        "measure_causal_depth_runtime.py", "")


def test_its_budget_is_under_the_authorized_ceiling(spec):
    plan = spec.budget.plan(price_per_hour=0.99, authorized_usd=1.6294)
    assert plan.hard_terminate_usd <= 1.6294, "the plan exceeds the ceiling"
    assert plan.expected_usd < plan.soft_stop_usd < plan.hard_terminate_usd
    from aadistill.autoinit.measurement import MEASUREMENT_AUTHORIZATION as A
    assert A.hard_cap_usd == 1.6294 and A.per_launch_hard_usd == 1.6294
    assert A.authorized_stages == (0,)


def test_it_stages_the_same_science_inputs_as_every_other_session(spec):
    staged = {r.path for r in spec.setup.staged_relay_inputs()}
    assert len(staged) == 10
    assert any("calibration_v1" in p for p in staged), (
        "the measurement reads the frozen calibration and must declare it")
    assert spec.setup.teacher_revision == "768f209d9ea81521153ed38c47d515654e938aea"


def test_both_failure_markers_bring_the_report_home(spec):
    """The report is written before either stop, and it is the whole artifact —
    so both failures are `incomplete`, meaning what they produced must still be
    collected."""
    assert set(spec.markers.failure) == {"MEASUREMENT_FAILED",
                                         "MEASUREMENT_FALLBACK"}
    assert set(spec.markers.incomplete) == set(spec.markers.failure)


def test_the_job_emits_those_exact_markers():
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    for m in ("MEASUREMENT_START", "MEASUREMENT_FAILED", "MEASUREMENT_FALLBACK",
              "ALL_DONE"):
        assert f'mark("{m}")' in src, f"the job never emits {m}"


def test_the_stop_conditions_are_the_ones_the_grant_names():
    src = (REPO / "scripts/autoinit/measure_causal_depth_runtime.py").read_text()
    assert "is non-zero: the repaired port" in src, "no backend-delta stop"
    assert "fell back to recompute" in src, "no cache-fallback stop"
    # The artifact is written BEFORE either stop: a run that found a real
    # difference is the one whose evidence matters most.
    assert src.index("out.write_text") < src.index("worst != 0.0")


def test_the_artifact_spec_exists_and_requires_the_report():
    import json
    spec_path = REPO / "configs/autoinit/measurement_artifacts.json"
    doc = json.loads(spec_path.read_text())
    required = [e for e in doc["entries"] if e.get("required")]
    assert len(required) == 1
    assert required[0]["pattern"] == "audit/autoinit_measurement/result.json"


def test_the_issuer_requires_a_grant_and_refuses_phase_a():
    src = (REPO / "scripts/autoinit/issue_measurement_authorization.py").read_text()
    assert '"--grant", required=True' in src
    assert "no grant document at" in src
    assert "a measurement may not authorize Phase A" in src
    # It digests ITS OWN harness, not Phase A's.
    assert "MEASUREMENT_HARNESS_FILES" in src
    assert "autoinit_measurement_launch.py" in src
    assert "measure_causal_depth_runtime.py" in src
