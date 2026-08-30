"""What each driver's REAL constructor loads, before any test supplies an auth.

Continuation attempt 2 died at driver stage 0 for `$0.3146`:

    AttributeError: 'PhaseAAuthorization' object has no attribute 'require_evidence'

`ContinuationDriver` overrode neither `AUTHORIZATION_TYPE` nor
`AUTHORIZATION_PATH`, so on the pod it loaded the committed **Phase-A** grant —
a real file, whose plan hash the parent's own `require_plan` accepts. It was the
only `PhaseADriver` subclass that set neither.

**Why the existing whole-function test could not catch it.** It builds the
driver with the real constructor and then discards what the constructor loaded,
on the very next line:

    driver = ContinuationDriver.__new__(ContinuationDriver)
    ContinuationDriver.__init__(driver, Args())
    driver.auth = make_auth(...)          # <- overwrites

Everything after that ran against a correctly-typed authorization the test
supplied. The wrong default was loaded and masked in the same three lines. That
is the general hazard of an injectable seam: extracting it makes the injected
value the only unexecuted code left.

So these tests assert on `driver.auth` **as the constructor left it**, and never
substitute one. They are deliberately narrow — authorization type, path, plan and
the evidence envelope — and are not a session simulator.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

#: A real, valid continuation authorization to construct against. The live one
#: is retired after every attempt, so the fixture is a retired artifact — which
#: is the point: it is a genuine `ContinuationAuthorization` on disk, not a stub.
FIXTURE_AUTH = (REPO / "logs/superseded"
                / "autoinit_continuation_b_authorization_20260829T194657Z_CONSUMED.json")


def load(name: str):
    path = REPO / f"scripts/pod/{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_wiring", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_wiring"] = mod
    spec.loader.exec_module(mod)
    return mod


class Args:
    """Only what the constructor itself touches."""

    stage = "all"
    image_digest = "sha256:wiring"
    rate = 0.99
    spent_usd = 0.0
    soft_stop_usd = 6.5
    authorized_usd = 8.0691
    probe_train_minutes = 61.55
    probe_battery_minutes = 9.82


@pytest.fixture
def constructed(tmp_path, monkeypatch):
    """The continuation driver, built by its own `__init__`, auth untouched."""
    mod = load("autoinit_continuation_b_driver")
    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True)
    shutil.copy(FIXTURE_AUTH, repo / mod.ContinuationDriver.AUTHORIZATION_PATH)
    monkeypatch.setattr(mod, "REPO", repo)
    monkeypatch.setattr(mod, "AUDIT", tmp_path / "audit")
    return mod, mod.ContinuationDriver(Args())


# --- the continuation's own wiring ------------------------------------------

def test_the_constructor_loads_a_continuation_authorization(constructed):
    from aadistill.autoinit.phase_b_continuation import ContinuationAuthorization

    mod, driver = constructed
    assert isinstance(driver.auth, ContinuationAuthorization), (
        f"the constructor produced {type(driver.auth).__name__}; on the pod that "
        "was PhaseAAuthorization and stage 0 raised on require_evidence")
    # The method whose absence ended attempt 2.
    assert hasattr(driver.auth, "require_evidence")
    assert driver.auth.runs_search is False


def test_it_came_from_the_continuation_authorization_path(constructed):
    mod, driver = constructed
    assert (mod.ContinuationDriver.AUTHORIZATION_PATH
            == "logs/autoinit_continuation_b_authorization.json")
    # Not merely declared — the object carries the fixture's own identity, so the
    # constructor demonstrably read THAT file.
    fixture = json.loads(FIXTURE_AUTH.read_text())
    assert driver.auth.authorization_id == fixture["authorization_id"]
    assert driver.auth.hard_cap_usd == fixture["hard_cap_usd"]


def test_it_validates_the_continuation_plan_and_rejects_phase_a(constructed):
    from aadistill.autoinit.authorization import AuthorizationError
    from aadistill.autoinit.phase_b_continuation import CONTINUATION_PLAN_V1

    mod, driver = constructed
    assert mod.ContinuationDriver.PLAN is CONTINUATION_PLAN_V1
    # Accepts its own — the constructor already ran this, so a repeat must pass.
    driver.auth.require_plan(CONTINUATION_PLAN_V1.plan_hash)

    import autoinit_phase_a_driver as parent
    with pytest.raises(AuthorizationError):
        driver.auth.require_plan(parent.PHASE_A_PLAN_V1.plan_hash)


def test_the_evidence_envelope_is_the_continuations_not_phase_as(constructed):
    """The artifact PATHNAME is unchanged — the collection contract names it —
    but what the file claims must describe this session."""
    _, driver = constructed
    ev = driver.ev
    assert ev["schema"] == "aadistill.autoinit.continuation_b_evidence/v1"
    assert ev["phase"] == "B-continuation"
    assert ev["runs_search"] is False
    assert ev["stage1_imported_not_recomputed"] is True
    assert ev["retrains_permanent_controls"] is False
    assert ev["followon_started"] is False
    assert ev["followon_reachable_from_this_driver"] is False
    # Inherited Phase-A framing must not ride along.
    assert "phase_a" not in ev, "the evidence still claims the Phase-A manifest"
    assert "scope" not in ev, "the evidence still claims the Phase-A scope"
    assert ev["authorization"]["authorization_id"].startswith(
        "autoinit.continuation_b.")


def test_the_constructor_leaves_nothing_the_inherited_stages_need_unset(constructed):
    """Not calling `super().__init__` means every attribute is this
    constructor's responsibility. Stage 3/4/5 are the inherited implementations."""
    _, driver = constructed
    for attr in ("a", "t0", "results", "evaluation_protocol", "plan",
                 "search_result", "leaves", "control_state", "rung1", "rung2",
                 "ev", "auth", "plan_spec", "evidence_universe", "finalists",
                 "imported_probe_ids", "evidence_observed"):
        assert hasattr(driver, attr), (
            f"{attr} is unset; an inherited stage would raise on it mid-session")

    # Every attribute the PARENT constructor establishes must exist here too,
    # derived from its bytecode rather than transcribed — so a future addition
    # to `PhaseADriver.__init__` cannot silently leave this driver short of an
    # attribute the inherited stages will then read.
    import dis

    import autoinit_phase_a_driver as parent
    parent_attrs = {i.argval
                    for i in dis.get_instructions(parent.PhaseADriver.__init__)
                    if i.opname == "STORE_ATTR"}
    assert parent_attrs, "the parent constructor sets nothing; the probe broke"
    missing = sorted(a for a in parent_attrs if not hasattr(driver, a))
    assert not missing, (
        f"{missing} are set by PhaseADriver.__init__ and not by this one, which "
        "does not call it; an inherited stage would raise on them mid-session")


def test_the_mutation_that_caused_attempt_2_is_caught(tmp_path, monkeypatch):
    """Put the defect back and require this file to notice.

    A guard that has only ever seen the fixed code is not known to be able to
    fail — which is precisely how the original slipped through.
    """
    from aadistill.autoinit.phase_b_continuation import ContinuationAuthorization

    mod = load("autoinit_continuation_b_driver")
    import autoinit_phase_a_driver as parent

    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True)
    shutil.copy(FIXTURE_AUTH, repo / mod.ContinuationDriver.AUTHORIZATION_PATH)
    shutil.copy(REPO / "logs/autoinit_phase_a_authorization.json",
                repo / "logs/autoinit_phase_a_authorization.json")
    monkeypatch.setattr(mod, "REPO", repo)
    monkeypatch.setattr(mod, "AUDIT", tmp_path / "audit")

    # Exactly attempt 2's tree: the subclass overrides neither attribute.
    monkeypatch.setattr(mod.ContinuationDriver, "AUTHORIZATION_TYPE",
                        parent.PhaseAAuthorization)
    monkeypatch.setattr(mod.ContinuationDriver, "AUTHORIZATION_PATH",
                        parent.PhaseADriver.AUTHORIZATION_PATH)
    monkeypatch.setattr(mod.ContinuationDriver, "PLAN", parent.PHASE_A_PLAN_V1)

    driver = mod.ContinuationDriver(Args())
    assert not isinstance(driver.auth, ContinuationAuthorization)
    assert not hasattr(driver.auth, "require_evidence"), (
        "the mutation no longer reproduces attempt 2; this guard proves nothing")


# --- and the same seam, pinned for every sibling -----------------------------

#: `(module, class, expected authorization type, expected path)`. Every
#: `PhaseADriver` subclass must NAME its own grant. Inheriting Phase A's is what
#: attempt 2 did, and the parent's own comment warns that the quiet version of
#: that bug is a wrong ceiling rather than a crash.
DRIVER_WIRING = (
    ("autoinit_phase_a_driver", "PhaseADriver",
     "PhaseAAuthorization", "logs/autoinit_phase_a_authorization.json"),
    ("autoinit_phase_b_driver", "PhaseBDriver",
     "PhaseBAuthorization", "logs/autoinit_phase_b_authorization.json"),
    ("autoinit_recovery_continuation_driver", "RecoveryContinuationDriver",
     "RecoveryContinuationAuthorization",
     "logs/autoinit_recovery_continuation_authorization.json"),
    ("autoinit_continuation_b_driver", "ContinuationDriver",
     "ContinuationAuthorization",
     "logs/autoinit_continuation_b_authorization.json"),
)


@pytest.mark.parametrize("module,cls,auth_type,auth_path", DRIVER_WIRING,
                         ids=[d[1] for d in DRIVER_WIRING])
def test_every_driver_names_its_own_authorization(module, cls, auth_type, auth_path):
    driver = getattr(load(module), cls)
    assert driver.AUTHORIZATION_TYPE.__name__ == auth_type, (
        f"{cls} would load a {driver.AUTHORIZATION_TYPE.__name__}")
    assert driver.AUTHORIZATION_PATH == auth_path, (
        f"{cls} would load {driver.AUTHORIZATION_PATH}")


def test_no_subclass_silently_inherits_phase_as_grant():
    """The property that actually failed, stated once rather than per-class."""
    parent = load("autoinit_phase_a_driver").PhaseADriver
    for module, cls, _, _ in DRIVER_WIRING:
        driver = getattr(load(module), cls)
        if driver is parent or driver.__name__ == "PhaseADriver":
            continue
        assert driver.AUTHORIZATION_TYPE is not parent.AUTHORIZATION_TYPE, (
            f"{cls} inherits Phase A's authorization TYPE")
        assert driver.AUTHORIZATION_PATH != parent.AUTHORIZATION_PATH, (
            f"{cls} inherits Phase A's authorization PATH — the attempt-2 defect")
