"""The C1 paid session: its type, its ceiling, and what it structurally cannot do.

Every assertion here is about a property the session has by construction rather
than by intention — the authorization type that refuses other phases, the ceiling
that exists in exactly one place, and the three driver methods that raise.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from aadistill.autoinit.c1_authorization import (  # noqa: E402
    C1_HARNESS_SOURCE_FILES_V1,
    SCHEMA as C1_SCHEMA,
    C1Authorization,
    c1_budget_spec,
    c1_hard_ceiling_usd,
    c1_harness_digest,
)
from aadistill.autoinit.authorization import AuthorizationError  # noqa: E402
from session_specs import load_session_launcher, session_args  # noqa: E402

CANDIDATE = Path("/home/ecs-user/aad-scratch/sessions/c1-candidate"
                 "/candidate_authorization.json")


@pytest.fixture(scope="module")
def launcher():
    return load_session_launcher("autoinit_c1_launch")


@pytest.fixture(scope="module")
def spec(launcher):
    return launcher.spec(session_args(launcher))


# --- the type refuses everything else --------------------------------------

def test_the_session_loads_the_c1_type_and_nothing_else(spec):
    # By qualified name, not object identity: `load_session_launcher` execs the
    # launcher under its own module name, so its `C1Authorization` is a distinct
    # object from this test's import even though both are the same source.
    loader = spec.authorization_loader
    assert loader.__qualname__ == "C1Authorization.load"
    assert loader.__module__.endswith("c1_authorization")
    assert loader.__self__.__name__ == C1Authorization.__name__
    assert spec.setup.env["SESSION_KIND"] == "c1"


def test_a_foreign_authorization_is_refused_by_schema(tmp_path):
    """A Phase-A/B or continuation grant measures a different harness and carries
    a ceiling derived for different work."""
    fake = {"schema": "aadistill.autoinit.phase_a_authorization/v1",
            "authorization_id": "x"}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(fake))
    with pytest.raises(AuthorizationError):
        C1Authorization.load(p)


def test_it_cannot_claim_phase_a_or_a_search():
    from dataclasses import FrozenInstanceError

    a = C1Authorization(
        authorization_id="t", granted_utc="u", granted_by="t", plan_id="p",
        plan_hash="h", science_plan_hash="s", expected_usd=1.0, hard_cap_usd=2.0,
        authorized_stages=(0,), stage_conditions={}, scope_note="t")
    assert a.allows_phase_a is False
    assert a.allows_beam_search is False
    assert a.authorizes_c1_isolation is True
    # Properties, not fields: there is nothing to set.
    with pytest.raises((FrozenInstanceError, AttributeError)):
        a.allows_beam_search = True


def test_the_setup_dispatcher_has_a_c1_branch():
    """A missing branch is not a type error — SESSION_KIND falls through to
    `spend` and loads a SpendAuthorization. Phase-B attempt 2 proved what that
    costs: $0.2300, one step after its test gate passed."""
    setup = (REPO / "scripts/pod/autoinit_preflight_setup.sh").read_text()
    assert 'elif [ "$SESSION_KIND" = "c1" ]; then' in setup
    branch = setup.split('"$SESSION_KIND" = "c1"')[1].split("elif")[0]
    assert "C1Authorization" in branch
    assert "authorizes_c1_isolation" in branch
    assert "allows_beam_search is False" in branch


# --- the ceiling exists in exactly one place -------------------------------

def test_the_ceiling_is_derived_from_the_accepted_pricing_record():
    pricing = json.loads((REPO / "logs/phase_c1_pricing.json").read_text())
    assert c1_hard_ceiling_usd(REPO) == pricing["totals"]["hard_ceiling_usd"]


def test_the_budget_plan_fits_that_ceiling_exactly():
    ceiling = c1_hard_ceiling_usd(REPO)
    plan = c1_budget_spec(REPO).plan(price_per_hour=0.99, authorized_usd=ceiling)
    assert plan.hard_terminate_usd <= ceiling
    assert plan.expected_usd == pytest.approx(12.2070, abs=5e-4)
    assert plan.soft_stop_usd == pytest.approx(13.4277, abs=5e-4)


def test_a_ceiling_that_rounds_down_would_fail_closed():
    """The 4-dp ceiling is rounded UP on purpose: the exact plan is $13.757733,
    and a grant written at a rounded-DOWN $13.7577 under-authorizes it."""
    from aadistill.infrastructure.budget import BudgetError

    with pytest.raises(BudgetError):
        c1_budget_spec(REPO).plan(price_per_hour=0.99, authorized_usd=13.7577)


def test_the_pricing_record_authorizes_nothing():
    pricing = json.loads((REPO / "logs/phase_c1_pricing.json").read_text())
    assert pricing["authorizes"] == "nothing"


def test_no_real_c1_grant_exists_in_the_repository():
    assert not (REPO / "logs/autoinit_c1_authorization.json").exists(), (
        "a C1 authorization is committed; none has been issued")


# --- the harness the grant measures ----------------------------------------

def test_the_harness_set_covers_the_launcher_driver_and_c1_science():
    for required in ("scripts/pod/autoinit_c1_launch.py",
                     "scripts/pod/autoinit_c1_driver.py",
                     "scripts/pod/autoinit_preflight_setup.sh"
                     if "scripts/pod/autoinit_preflight_setup.sh"
                     in C1_HARNESS_SOURCE_FILES_V1 else "scripts/pod/setup.sh",
                     "src/aadistill/autoinit/c1_session.py",
                     "src/aadistill/autoinit/c1_isolation.py",
                     "src/aadistill/autoinit/fixed_path.py",
                     "src/aadistill/autoinit/operators/attention_activation.py",
                     "src/aadistill/init/attention_stats.py",
                     "src/aadistill/autoinit/recovery.py",
                     "scripts/autoinit/score_recovery_search.py"):
        assert required in C1_HARNESS_SOURCE_FILES_V1, required


def test_a_missing_harness_file_raises_rather_than_shrinking_the_digest():
    with pytest.raises(AuthorizationError, match="missing"):
        c1_harness_digest(REPO, files=("src/aadistill/does_not_exist.py",))


def test_the_preregistration_records_the_live_harness_digest():
    doc = json.loads(
        (REPO / "logs/phase_c1_execution_preregistration.json").read_text())
    assert doc["c1_harness"]["digest"] == c1_harness_digest(REPO)["digest"]
    assert doc["authorizes"] == "nothing"
    assert doc["authorization"]["schema"] == C1_SCHEMA
    assert "NO GRANT EXISTS" in doc["authorization"]["status"]


# --- what the session cannot do --------------------------------------------

def test_the_driver_cannot_search_rank_or_eliminate():
    import inspect

    spec_ = importlib.util.spec_from_file_location(
        "c1_driver_probe", REPO / "scripts/pod/autoinit_c1_driver.py")
    mod = importlib.util.module_from_spec(spec_)
    sys.modules["c1_driver_probe"] = mod
    spec_.loader.exec_module(mod)
    for method in ("stage1", "run_rung", "selection_row"):
        src = inspect.getsource(getattr(mod.C1Driver, method))
        assert "raise NotImplementedError" in src, method


def test_neither_launcher_nor_driver_imports_the_search():
    import ast

    for rel in ("scripts/pod/autoinit_c1_launch.py",
                "scripts/pod/autoinit_c1_driver.py"):
        mods: set[str] = set()
        for node in ast.walk(ast.parse((REPO / rel).read_text())):
            if isinstance(node, ast.ImportFrom):
                mods.add(node.module or "")
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
        bad = [m for m in mods
               if "phase_a_search" in m or m.endswith(".search") or m == "search"]
        assert not bad, f"{rel} imports {bad}"


def test_the_probe_schedule_is_six_and_names_the_frozen_seeds():
    spec_ = importlib.util.spec_from_file_location(
        "c1_driver_sched", REPO / "scripts/pod/autoinit_c1_driver.py")
    mod = importlib.util.module_from_spec(spec_)
    sys.modules["c1_driver_sched"] = mod
    spec_.loader.exec_module(mod)
    probes = mod.C1Driver.probe_descriptors(object.__new__(mod.C1Driver))
    assert len(probes) == 6
    assert sorted({p["arm"] for p in probes}) == ["incumbent", "treatment"]
    assert sorted({p["seed"] for p in probes}) == [696460635, 1635674081, 1656475568]
    for arm in ("incumbent", "treatment"):
        assert sum(1 for p in probes if p["arm"] == arm) == 3


def test_the_session_declares_that_it_neither_searches_nor_eliminates(spec):
    assert spec.evidence_fields["runs_a_search"] is False
    assert spec.evidence_fields["eliminates_arms"] is False
    assert spec.evidence_fields["probes"] == 6
    assert spec.evidence_fields["formal_recovery_evidence"] == "OUT OF SCOPE"


# --- the pre-provider gates -------------------------------------------------

@pytest.mark.skipif(not CANDIDATE.is_file(),
                    reason="no candidate authorization on this machine")
def test_every_gate_but_the_commit_binding_passes_against_the_candidate(launcher,
                                                                        spec):
    """The candidate is structurally valid and bound to the live identities, so
    every `$0` gate must pass. `session_commit_and_lineage` cannot: it binds a
    real issued commit, which does not exist before issuance."""
    import types

    auth = C1Authorization.load(CANDIDATE)
    ctx = types.SimpleNamespace(scr=Path("/tmp/c1gate"), args=session_args(launcher),
                                auth=auth, evidence={}, image_digest="candidate",
                                price=0.99, spent_usd=0.0)
    failures = []
    for gate in spec.precheck:
        name = getattr(gate, "__name__", "session_commit_and_lineage")
        ok, msg = gate(ctx)
        if not ok and name != "session_commit_and_lineage":
            failures.append(f"{name}: {msg}")
    assert not failures, failures
