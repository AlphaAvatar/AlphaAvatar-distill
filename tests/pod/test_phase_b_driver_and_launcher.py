"""The Phase-B executables: what they inherit, what they override, what they cost.

A Phase-B pod runs Phase A's machinery under Phase B's contract. Both halves of
that need proving: that the overrides do what Phase B needs, and that the
inheritance still satisfies what the parent REQUIRES. Subclassing proven
machinery and reasoning only from what the new session wanted has cost this
project two pods.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

import autoinit_phase_b_driver as pbd  # noqa: E402
import autoinit_phase_b_launch as pbl  # noqa: E402
from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V2,
)
from aadistill.autoinit.phase_a import PHASE_A_PLAN_V1  # noqa: E402
from aadistill.autoinit.phase_b import (  # noqa: E402
    PHASE_B_PLAN_V1, PhaseBAuthorization, phase_b_source_digest,
)
from autoinit_phase_a_driver import PhaseADriver  # noqa: E402

HISTORICAL = REPO / "logs/autoinit_recovery_continuation_attempt7/probes"


def _args(**over):
    base = dict(image_digest="img@sha256:x", rate=0.99, spent_usd=0.0,
                soft_stop_usd=20.0, authorized_usd=26.0, search_minutes=360.0,
                search_deadline_minutes=400.0, probe_train_minutes=62.0,
                probe_battery_minutes=10.0, stage="all")
    return types.SimpleNamespace(**{**base, **over})


def _auth_file(tmp_path, **over) -> Path:
    auth = PhaseBAuthorization(
        authorization_id="test", granted_utc="2026-08-26T00:00:00Z",
        granted_by="test", plan_id=PHASE_B_PLAN_V1.plan_id,
        plan_hash=PHASE_B_PLAN_V1.plan_hash, science_plan_hash="s" * 64,
        calibration_profile_hashes={
            DOMAIN_BALANCED_V1.qualified_id: DOMAIN_BALANCED_V1.profile_hash,
            REASONING_HEAVY_V2.qualified_id: REASONING_HEAVY_V2.profile_hash},
        calibration_content_hashes={
            DOMAIN_BALANCED_V1.qualified_id: DOMAIN_BALANCED_V1.content_sha256,
            REASONING_HEAVY_V2.qualified_id: REASONING_HEAVY_V2.content_sha256},
        planning_floor_usd=13.08, hard_cap_usd=26.81,
        authorized_stages=(0, 1, 2, 3, 4, 5), stage_conditions={},
        scope_note="test", **over)
    path = tmp_path / "phase_b_auth.json"
    path.write_text(json.dumps(auth.as_dict()))
    return path


@pytest.fixture
def driver(tmp_path, monkeypatch):
    monkeypatch.setattr(pbd, "AUDIT", tmp_path / "audit")
    monkeypatch.setattr(pbd.PhaseBDriver, "AUTHORIZATION_PATH",
                        str(_auth_file(tmp_path)))
    return pbd.PhaseBDriver(_args())


# --- the governing artifacts ------------------------------------------------


def test_it_is_governed_by_the_phase_b_plan_and_grant(driver):
    assert driver.PLAN is PHASE_B_PLAN_V1
    assert driver.PLAN.plan_hash != PHASE_A_PLAN_V1.plan_hash
    assert driver.auth.allows_phase_b is True
    assert driver.auth.allows_phase_a is False
    assert driver.ev["schema"] == "aadistill.autoinit.phase_b_evidence/v1"
    assert driver.ev["phase"] == "B"


def test_a_phase_a_grant_cannot_govern_this_driver(tmp_path, monkeypatch):
    from aadistill.autoinit.authorization import AuthorizationError
    from aadistill.autoinit.phase_a import PHASE_A_AUTHORIZATION

    path = tmp_path / "phase_a.json"
    path.write_text(json.dumps(PHASE_A_AUTHORIZATION.as_dict()))
    monkeypatch.setattr(pbd, "AUDIT", tmp_path / "audit")
    monkeypatch.setattr(pbd.PhaseBDriver, "AUTHORIZATION_PATH", str(path))
    with pytest.raises(AuthorizationError, match="phase_b"):
        pbd.PhaseBDriver(_args())


def test_the_constructor_leaves_no_inherited_contract_unset(driver, tmp_path,
                                                            monkeypatch):
    """It deliberately does not call `super().__init__`, so it must still set
    everything the inherited methods read. Reasoning from what Phase B *needed*
    rather than what the parent *requires* is what cost two pods."""
    monkeypatch.setattr("autoinit_phase_a_driver.AUDIT", tmp_path / "audit_a")
    monkeypatch.setattr(PhaseADriver, "AUTHORIZATION_PATH",
                        str(_auth_file(tmp_path / "a" if False else tmp_path)))
    from aadistill.autoinit.phase_a import PhaseAAuthorization  # noqa: F401

    parent_attrs = {"a", "t0", "results", "evaluation_protocol", "plan",
                    "search_result", "leaves", "control_state", "rung1",
                    "rung2", "ev", "auth"}
    missing = parent_attrs - set(vars(driver))
    assert not missing, f"the Phase-B constructor never set {sorted(missing)}"


def test_stage_ordering_advances_through_the_phase_b_plan(driver, tmp_path,
                                                          monkeypatch):
    """`enter` must consult Phase B's plan, not Phase A's."""
    import autoinit_phase_a_driver as pad
    from aadistill.autoinit.recovery import RecoveryAdmissionError

    monkeypatch.setattr(pad, "STATUS", tmp_path / "phase_b.status")
    with pytest.raises(RecoveryAdmissionError):
        driver.enter(2)                      # stage 0 and 1 have not passed
    driver.results[0] = {"passed": True}
    driver.results[1] = {"passed": True}
    driver.enter(2)                          # now allowed
    assert "STAGE_START:2" in (tmp_path / "phase_b.status").read_text()


def test_the_driver_writes_its_markers_where_the_LAUNCHER_polls():
    """`mark()` is a module function appending to a module global, and every
    inherited stage calls it. A Phase-B run writing to `autoinit_phase_a.status`
    while the launcher polls `autoinit_phase_b.status` would look like a hung
    session for its whole duration and be killed."""
    import autoinit_phase_a_driver as pad

    assert str(pad.STATUS) == pbd.STATUS.as_posix(), (
        "the inherited mark() does not write to the Phase-B status file")
    assert pbl.STATUS == str(pbd.STATUS), (
        "the launcher polls a file the driver never writes")
    assert "phase_b" in pbl.STATUS


# --- citing verified history ------------------------------------------------


def test_it_imports_only_the_admitted_and_reusable_probes(driver):
    imported = driver.import_historical_probes()
    ids = imported["probe_ids"]
    assert len(ids) == 8, ids
    # The three excluded Phase-A leaves have verified sa probes that cost
    # nothing. They are STILL not imported.
    for excluded in ("158b96cf651f", "281a02c3ac18", "4e429f7ed722"):
        assert not any(excluded in p for p in ids), excluded
        assert any(excluded in s["probe_id"] for s in imported["skipped"]), excluded
    for finalist in ("cca699c93f34", "85bde4ded2c3", "control-qwen"):
        assert any(finalist in p for p in ids), finalist


def test_an_unverified_reuse_record_stops_the_import(driver, tmp_path, monkeypatch):
    bad = tmp_path / "reuse.json"
    bad.write_text(json.dumps({"reuse_verified": False, "failures": ["x"]}))
    monkeypatch.setattr(pbd, "REUSE_RECORD", bad)
    with pytest.raises(RuntimeError, match="not verified"):
        driver.import_historical_probes()


def test_importing_nothing_is_a_defect_not_a_saving(driver, tmp_path, monkeypatch):
    empty = tmp_path / "reuse.json"
    empty.write_text(json.dumps({"reuse_verified": True,
                                 "admitted_reusable_probes": ["nobody/sa"]}))
    monkeypatch.setattr(pbd, "REUSE_RECORD", empty)
    with pytest.raises(RuntimeError, match="defect not a saving"):
        driver.import_historical_probes()


def test_an_imported_probe_is_cited_under_a_COMPARABLE_protocol(driver):
    """The inherited check demands an identical protocol hash. For an imported
    Phase-A probe that is the wrong predicate and would silently re-buy it."""
    driver.import_historical_probes()
    probe_id = "autoinit.v1.phase_a.rung1.cca699c93f34.sa"
    record = json.loads((HISTORICAL / f"{probe_id}.json").read_text())

    driver.evaluation_protocol = types.SimpleNamespace(
        evaluation_protocol_hash="a different hash entirely")
    restored = driver.restore_probe({
        "probe_id": probe_id,
        "student_artifact_digest": record["student_artifact_digest"],
        "seed": record["seed"]})
    assert restored is not None, "a verified citation was refused and would be re-bought"
    assert restored["resumed"] is True
    assert restored["imported_from_phase_a"] is True


def test_a_citation_is_still_refused_on_a_different_student_or_seed(driver):
    driver.import_historical_probes()
    probe_id = "autoinit.v1.phase_a.rung1.cca699c93f34.sa"
    record = json.loads((HISTORICAL / f"{probe_id}.json").read_text())
    driver.evaluation_protocol = types.SimpleNamespace(
        evaluation_protocol_hash=record["evaluation_protocol_hash"])

    assert driver.restore_probe({
        "probe_id": probe_id, "student_artifact_digest": "wrong",
        "seed": record["seed"]}) is None
    assert driver.restore_probe({
        "probe_id": probe_id,
        "student_artifact_digest": record["student_artifact_digest"],
        "seed": 999}) is None


def test_a_probe_this_session_ran_still_binds_STRICTLY(driver, tmp_path):
    """Only imported ids relax the protocol check."""
    driver.evaluation_protocol = types.SimpleNamespace(
        evaluation_protocol_hash="live-hash")
    (pbd.AUDIT / "probes").mkdir(parents=True, exist_ok=True)
    own = {"probe_id": "own", "student_artifact_digest": "d", "seed": 1,
           "evaluation_protocol_hash": "a-different-hash", "complete": True}
    (pbd.AUDIT / "probes" / "own.json").write_text(json.dumps(own))
    assert driver.restore_probe(
        {"probe_id": "own", "student_artifact_digest": "d", "seed": 1}) is None


# --- the joint search -------------------------------------------------------


def test_run_search_passes_BOTH_profiles_and_a_dispatching_loader(driver):
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return "result"

    driver.plan = types.SimpleNamespace(searched_leaves=5)
    assert driver.run_search(fake_search) == "result"
    assert captured["profiles"] == (DOMAIN_BALANCED_V1, REASONING_HEAVY_V2)
    items = captured["calibration_items"]
    assert set(items) == {DOMAIN_BALANCED_V1.qualified_id,
                          REASONING_HEAVY_V2.qualified_id}
    # Genuinely different mixtures, not the same list under two labels.
    a = [i["item_id"] for i in items[DOMAIN_BALANCED_V1.qualified_id]]
    b = [i["item_id"] for i in items[REASONING_HEAVY_V2.qualified_id]]
    assert a != b and len(b) == 62 and len(a) == 67
    assert captured["top_n"] == 5
    assert "phase_b_search" in str(captured["workdir"])


def test_the_parents_search_seam_is_still_single_profile():
    """Phase A must not acquire a second profile by inheritance."""
    captured = {}
    parent = object.__new__(PhaseADriver)
    parent.plan = types.SimpleNamespace(searched_leaves=5)
    parent.a = _args()
    parent.run_search(lambda **kw: captured.update(kw))
    assert "profiles" not in captured and "calibration_items" not in captured


# --- the launcher -----------------------------------------------------------


@pytest.fixture(scope="module")
def spec():
    args = pbl.build_parser().parse_args(
        ["--scr", "/tmp/scr", "--session-commit", "d" * 40, "--bundle", "b.tgz"])
    return pbl.spec(args), args


def test_the_launcher_prices_TEN_probes_not_twelve(spec):
    session, args = spec
    assert session.budget.arms == 10, (
        "pricing twelve would quietly fund re-buying the three cited probes")
    assert pbl.RUNG1_PROBES_P2 == 5 and pbl.RUNG2_PROBES_P2 == 2
    assert pbl.TIE_BREAK_PROBES_P2 == 3


def test_the_launcher_widens_the_search_phase_for_P2(spec):
    session, _ = spec
    search = [p for p in session.budget.other_phases
              if p.name == pbl.STAGE1_SEARCH_PHASE]
    assert len(search) == 1 and search[0].minutes == pbl.SEARCH_MINUTES_P2
    assert pbl.SEARCH_MINUTES_P2 > 180.0, "P=2 searches more states than P=1"


def test_the_launcher_binds_the_phase_b_plan_and_grant_type(spec):
    session, _ = spec
    assert session.plan_hash == PHASE_B_PLAN_V1.plan_hash
    assert session.authorization_loader.__func__ is PhaseBAuthorization.load.__func__
    assert session.driver_job_id == "autoinit_phase_b"


def test_the_v2_mixture_travels_a_path_the_pod_can_actually_read(spec):
    """It was built on the dev box and never uploaded, so a RelayInput naming it
    would be an object the pod cannot fetch — attempt 5's failure exactly."""
    session, _ = spec
    names = [a.dest_name for a in session.setup.local_assets]
    assert "reasoning_heavy_v2" in names
    relay = [r.path for r in session.setup.relay_inputs]
    assert not any("reasoning_heavy" in p for p in relay)
    # And the file it points at exists to be copied.
    assert (REPO / "artifacts/stage1/reasoning_heavy_v2/items.jsonl").is_file()


def test_the_driver_command_launches_the_PHASE_B_driver(spec):
    session, args = spec
    ctx = types.SimpleNamespace(image_digest="img@sha256:x", price=0.99,
                                args=args, spent_usd=0.0,
                                auth=types.SimpleNamespace(hard_cap_usd=26.81))
    plan = types.SimpleNamespace(
        soft_stop_usd=20.0,
        breakdown=[types.SimpleNamespace(name=pbl.STAGE1_SEARCH_PHASE,
                                         minutes=pbl.SEARCH_MINUTES_P2)],
        soft_stop_reserves=[])
    command = session.driver_command(ctx, plan)
    assert "autoinit_phase_b_driver.py" in command
    assert "autoinit_phase_a_driver.py" not in command


def test_the_prechecks_refuse_before_a_pod_exists(spec):
    session, _ = spec
    names = [getattr(g, "__name__", type(g).__name__) for g in session.precheck]
    for gate in ("phase_b_source_gate", "preregistration_gate", "reuse_record_gate"):
        assert gate in names, gate

    ctx = types.SimpleNamespace(auth=types.SimpleNamespace(source_digest=None))
    ok, why = pbl.phase_b_source_gate(ctx)
    assert not ok and "authorizes no executable" in why

    ctx = types.SimpleNamespace(auth=types.SimpleNamespace(source_digest="0" * 64))
    ok, why = pbl.phase_b_source_gate(ctx)
    assert not ok and "digests to" in why

    live = phase_b_source_digest(REPO)["digest"]
    ctx = types.SimpleNamespace(auth=types.SimpleNamespace(source_digest=live))
    ok, why = pbl.phase_b_source_gate(ctx)
    assert ok, why


def test_the_preregistration_gate_binds_the_executable_and_both_mixtures():
    ok, why = pbl.preregistration_gate(types.SimpleNamespace())
    assert ok, why
    assert "binds this executable and both mixtures" in why


def test_the_reuse_gate_proves_the_ten_probe_budget():
    ok, why = pbl.reuse_record_gate(types.SimpleNamespace())
    assert ok, why
    assert "pricing ten, not twelve" in why


def test_the_candidate_filter_holds_even_if_the_RECORD_admits_an_excluded_leaf(
        driver, tmp_path, monkeypatch):
    """Isolates the driver's own guard from the reuse record's.

    Today the exclusion is defended twice — the record only admits the three
    candidates, and the driver filters on the candidate set — so removing the
    driver's filter changes nothing observable. That is defence in depth, and it
    also means a test that only checks the outcome cannot tell whether the
    driver's guard still exists. This feeds a record that *does* admit an
    excluded leaf and requires the driver to refuse it anyway.
    """
    record = json.loads(
        (REPO / "logs/autoinit_historical_probe_reuse.json").read_text())
    permissive = tmp_path / "reuse.json"
    permissive.write_text(json.dumps({
        **record,
        "admitted_reusable_probes": sorted(
            [*record["admitted_reusable_probes"], "158b96cf651f/sa"]),
    }))
    monkeypatch.setattr(pbd, "REUSE_RECORD", permissive)

    imported = driver.import_historical_probes()
    assert not any("158b96cf651f" in p for p in imported["probe_ids"]), (
        "an excluded Phase-A leaf was imported because the record admitted it; "
        "the candidate set is the driver's decision, not the record's")
    assert any(s["probe_id"].endswith("158b96cf651f.sa")
               and "not in the Phase-B candidate set" in s["reason"]
               for s in imported["skipped"])
