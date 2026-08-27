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

    driver.plan = types.SimpleNamespace(searched_leaves=5)
    imported = [types.SimpleNamespace(state_id="cca699c93f34dad7e94a5d13a25b2bc2"),
                types.SimpleNamespace(state_id="85bde4ded2c31953f802e39cf2252c87")]

    def fake_search_with_imports(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(imported=imported)

    found = driver.run_search(fake_search_with_imports)
    assert found.imported == imported
    assert driver.imported_finalists == imported, (
        "the imported finalists never reached the driver, so the universe would "
        "be six rather than eight")
    assert captured["retained_candidates"], "no retained candidate was injected"
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


# --- repair 1: the cross-phase candidate universe ---------------------------


class _FakeState:
    """Enough of an InitializationState for `probe_configs` and the universe."""

    def __init__(self, state_id, provenance="search"):
        self.state_id = state_id
        self.provenance = provenance
        self.path_label = "ROOT"
        self.checkpoint_path = f"/pod/{state_id}"
        self.artifact_digest = f"digest-{state_id}"
        self.checkpoint_sha256 = f"shard-{state_id}"
        self.num_parameters = 596_049_920


def test_stage2_actually_receives_EIGHT_candidate_descriptors(driver):
    """Not that a constant says eight — that the descriptors are eight.

    The inherited stage 2 builds `leaves + control` = six. Journal seeding does
    not fix that: a citation is only consulted if a DESCRIPTOR exists for its
    candidate, so without this the run would seed eight citations, use three, and
    compare a different set from the one the preregistration froze.
    """
    from aadistill.autoinit.recovery import probe_configs

    driver.leaves = [_FakeState(f"phaseb{i:026d}") for i in range(5)]
    driver.imported_finalists = [
        _FakeState("cca699c93f34dad7e94a5d13a25b2bc2", "retained_phase_a_finalist"),
        _FakeState("85bde4ded2c31953f802e39cf2252c87", "retained_phase_a_finalist")]
    driver.control_state = _FakeState("control-qwen3_0p6b_init_v0",
                                      "retained_canonical")

    universe = driver.candidate_universe()
    assert len(universe) == 8, [s.state_id[:12] for s in universe]

    plan = types.SimpleNamespace(
        plan_id="autoinit.v1.phase_a", seeds=(20260726, 20260801),
        recipe=types.SimpleNamespace(recipe_id="e1_p1_kd_heavy@0.86M"),
        feasibility_metric="usable_rollout_rate", primary_metric="correct_overall")
    descriptors = probe_configs(universe, plan, rung=1)
    assert len(descriptors) == 8, "stage 2 would not have compared eight candidates"

    ids = [d["probe_id"] for d in descriptors]
    # The imported ids must be EXACTLY the historical ones, or their citations
    # are never found and they are silently retrained.
    assert "autoinit.v1.phase_a.rung1.cca699c93f34.sa" in ids
    assert "autoinit.v1.phase_a.rung1.85bde4ded2c3.sa" in ids
    assert "autoinit.v1.phase_a.rung1.control-qwen.sa" in ids
    # Seven searched candidates compete for the two sb slots; the control does
    # not consume one.
    searched = [s for s in universe if s.provenance != "retained_canonical"]
    assert len(searched) == 7


def test_only_the_FIVE_new_leaves_owe_a_new_sa_probe(driver):
    """The eight-candidate universe must not become an eight-probe bill."""
    driver.import_historical_probes()
    driver.evaluation_protocol = types.SimpleNamespace(
        evaluation_protocol_hash="live-and-comparable")

    owed, cited = [], []
    for probe_id, digest, seed in (
            ("autoinit.v1.phase_a.rung1.cca699c93f34.sa", None, None),
            ("autoinit.v1.phase_a.rung1.85bde4ded2c3.sa", None, None),
            ("autoinit.v1.phase_a.rung1.control-qwen.sa", None, None)):
        record = json.loads((HISTORICAL / f"{probe_id}.json").read_text())
        restored = driver.restore_probe({
            "probe_id": probe_id,
            "student_artifact_digest": record["student_artifact_digest"],
            "seed": record["seed"]})
        (cited if restored else owed).append(probe_id)
    assert len(cited) == 3 and not owed, owed

    # A brand-new Phase-B leaf has nothing to restore and is genuinely owed.
    assert driver.restore_probe({
        "probe_id": "autoinit.v1.phase_a.rung1.phaseb000000.sa",
        "student_artifact_digest": "d", "seed": 20260726}) is None


def test_it_fails_closed_when_an_imported_finalist_lacks_its_evidence(
        driver, tmp_path, monkeypatch):
    """Do not fall through into training a checkpoint staged read-only."""
    record = json.loads(
        (REPO / "logs/autoinit_historical_probe_reuse.json").read_text())
    thin = tmp_path / "reuse.json"
    thin.write_text(json.dumps({
        **record,
        "admitted_reusable_probes": [p for p in record["admitted_reusable_probes"]
                                     if not p.startswith("cca699c93f34/sb")]}))
    monkeypatch.setattr(pbd, "REUSE_RECORD", thin)
    with pytest.raises(RuntimeError, match="lack verified evidence"):
        driver.require_citable(("sa", "sb"))


def test_an_imported_finalist_with_contradicting_bytes_is_refused():
    """The digest decides, not the record."""
    from aadistill.autoinit.state import StateError, make_retained_state

    artifact = types.SimpleNamespace(artifact_digest="actual", path="/pod/x",
                                     single_shard_sha256=None, is_sharded=False)
    with pytest.raises(StateError, match="contradict its record"):
        make_retained_state(
            state_id="cca699c93f34dad7e94a5d13a25b2bc2", artifact=artifact,
            spec=None, target_spec=None, num_parameters=1,
            root_teacher_id="t", root_teacher_sha256="a" * 64, description="d",
            expected_artifact_digest="recorded")


def test_an_imported_candidate_may_not_masquerade_as_the_control():
    from aadistill.autoinit.state import StateError, make_retained_state

    artifact = types.SimpleNamespace(artifact_digest="d", path="/pod/x")
    with pytest.raises(StateError, match="use make_control_state"):
        make_retained_state(
            state_id="x", artifact=artifact, spec=None, target_spec=None,
            num_parameters=1, root_teacher_id="t", root_teacher_sha256="a" * 64,
            description="d", provenance="retained_canonical")


def test_the_parents_universe_is_still_leaves_plus_control():
    """Phase A must not acquire cross-phase candidates by inheritance."""
    parent = object.__new__(PhaseADriver)
    parent.leaves = [_FakeState("a"), _FakeState("b")]
    parent.control_state = _FakeState("c", "retained_canonical")
    assert len(parent.candidate_universe()) == 3


# --- repair 2: disk provisioning --------------------------------------------


def test_the_disk_default_meets_the_frozen_provision():
    args = pbl.build_parser().parse_args(
        ["--scr", "/tmp/scr", "--session-commit", "d" * 40, "--bundle", "b.tgz"])
    assert args.disk_gb == pbl.PHASE_B_PROVISION_GIB == 300
    assert pbl.PHASE_B_PEAK_WORKING_GIB > 240


def test_a_pod_below_the_frozen_provision_is_refused_before_it_exists():
    for requested, expected in ((200, False), (299, False), (300, True), (400, True)):
        ctx = types.SimpleNamespace(args=types.SimpleNamespace(disk_gb=requested))
        ok, why = pbl.disk_provision_gate(ctx)
        assert ok is expected, (requested, why)
        if not ok:
            assert "below the frozen Phase-B provision" in why


# --- repair 3: the successful-run transfer path -----------------------------


def test_fetch_and_secure_speak_the_SAME_contract(monkeypatch, tmp_path):
    """Successful Stage 1 → five leaves fetched → five transfer records → secured.

    The failure this closes is asymmetric: it can only bite AFTER a
    scientifically successful run, which is the most expensive moment to lose
    the bytes.
    """
    import autoinit_phase_a_launch as pal

    store = tmp_path / "store"
    store.mkdir()
    leaves = [{"state_id": f"leaf{i}", "artifact_digest": f"d{i}"} for i in range(5)]
    (store / pal.SELECTED_LEAF_REPORT).write_text(json.dumps({"leaves": leaves}))
    ctx = types.SimpleNamespace(scr=tmp_path, args=types.SimpleNamespace(
        fetch_finalists=True, ckpt_store=str(tmp_path / "ck")))

    # The real transfer needs a pod; the contract under test is the SHAPE both
    # halves agree on, so the transfer itself is the one thing stood in for.
    fetched = [{"artifact": "stage1_selected_leaf", "state_id": r["state_id"],
                "rc": 0, "matched": True} for r in leaves]
    monkeypatch.setattr(pal, "fetch_selected_leaves", lambda c: fetched)

    ok, why = pal.selected_leaves_secured(ctx, fetched)
    assert ok, why
    assert "all 5 stage-1 selected leaves verified off-pod" in why

    # And the shape the OLD wiring produced — bare id strings — is REFUSED
    # rather than raising. It used to call `.get` on a str and die with
    # AttributeError, after a successful run, at teardown.
    ok, why = pal.selected_leaves_secured(ctx, [r["state_id"] for r in leaves])
    assert not ok
    assert "only 0 are" in why and "attempt 11" in why


def test_the_phase_b_policy_uses_the_record_returning_fetcher():
    """By NAME, not by identity.

    `test_phase_a_rehearsal` loads the launcher a second time through
    `importlib.util.spec_from_file_location`, so two live module objects hold two
    distinct function objects for the same source. An `is` comparison then fails
    for a reason that has nothing to do with the wiring under test — it passed
    alone and failed in the full suite.
    """
    args = pbl.build_parser().parse_args(
        ["--scr", "/tmp/scr", "--session-commit", "d" * 40, "--bundle", "b.tgz"])
    policy = pbl.spec(args).artifacts
    assert policy.fetch_products.__name__ == "fetch_selected_leaves"
    assert policy.products_secured.__name__ == "selected_leaves_secured"
    # The defect being closed: the id-returning fetcher paired with a
    # record-consuming gate.
    assert policy.fetch_products.__name__ != "finalists_to_fetch"


def test_the_checkpoint_store_capacity_gate_is_wired():
    """A pod may not be created unless the destination can hold the leaves."""
    args = pbl.build_parser().parse_args(
        ["--scr", "/tmp/scr", "--session-commit", "d" * 40, "--bundle", "b.tgz"])
    names = {getattr(g, "__name__", "") for g in pbl.spec(args).precheck}
    assert "ckpt_store_capacity_gate" in names, sorted(names)


def test_only_the_two_admitted_finalists_are_staged():
    inputs = pbl.imported_finalist_inputs()
    dests = {i.dest for i in inputs}
    assert len(dests) == 2, dests
    for excluded in ("158b96cf651f", "281a02c3ac18", "4e429f7ed722"):
        assert not any(excluded in d for d in dests), excluded
    for admitted in pbl.IMPORTED_FINALIST_PREFIXES:
        assert any(admitted in d for d in dests), admitted


def test_stage2_ITSELF_reads_the_universe_seam(driver, monkeypatch, tmp_path):
    """Asserting `candidate_universe()` returns eight proves nothing if stage 2
    does not call it. Reverting stage 2 to the hardcoded `leaves + control` was
    caught only by the digest cross-check, which is not a test of behaviour."""
    import autoinit_phase_a_driver as pad

    monkeypatch.setattr(pad, "STATUS", tmp_path / "s.status")
    driver.results[0] = {"passed": True}
    driver.results[1] = {"passed": True}
    driver.plan = types.SimpleNamespace(searched_leaves=5)

    called = []

    def universe():
        called.append(True)
        raise _Sentinel("stage 2 reached the seam")

    monkeypatch.setattr(driver, "candidate_universe", universe)
    with pytest.raises(_Sentinel):
        driver.stage2()
    assert called, "stage 2 built its candidates without going through the seam"


class _Sentinel(Exception):
    pass


def test_the_secured_gate_refuses_a_FAILED_or_UNMATCHED_transfer(tmp_path):
    """It must check the outcome, not merely the shape. A truncated leaf that
    reports `matched: False` is exactly what the digest re-check is for."""
    import autoinit_phase_a_launch as pal

    store = tmp_path / "store"
    store.mkdir()
    leaves = [{"state_id": f"leaf{i}"} for i in range(3)]
    (store / pal.SELECTED_LEAF_REPORT).write_text(json.dumps({"leaves": leaves}))
    ctx = types.SimpleNamespace(scr=tmp_path, args=types.SimpleNamespace(
        fetch_finalists=True, ckpt_store=str(tmp_path / "ck")))

    def record(state_id, **over):
        return {"artifact": "stage1_selected_leaf", "state_id": state_id,
                "rc": 0, "matched": True, **over}

    ok, why = pal.selected_leaves_secured(
        ctx, [record("leaf0"), record("leaf1", rc=1), record("leaf2")])
    assert not ok and "leaf1" in why, why

    ok, why = pal.selected_leaves_secured(
        ctx, [record("leaf0"), record("leaf1"), record("leaf2", matched=False)])
    assert not ok and "leaf2" in why, why

    ok, why = pal.selected_leaves_secured(
        ctx, [record("leaf0"), record("leaf1"),
              record("leaf2", artifact="something_else")])
    assert not ok and "leaf2" in why, why

    ok, _ = pal.selected_leaves_secured(
        ctx, [record("leaf0"), record("leaf1"), record("leaf2")])
    assert ok


# --- the shared-machinery seam ----------------------------------------------
#
# The defect this covers reached the final pre-provider gate: PhaseBAuthorization
# named three things `source_*` while the SHARED session machinery reads
# `harness_*`, so `session_commit_gate` raised AttributeError instead of running
# and the commit-lineage check never executed for Phase B. Property equality
# would not have caught it -- nothing was comparing the two names. What catches
# it is driving the real inherited path.


def _issued(tmp_path, **over):
    """A real, self-verifying authorization artifact, loaded back off disk."""
    from aadistill.autoinit.phase_b import phase_b_source_digest

    fields = dict(
        authorization_id="seam-test", granted_utc="2026-08-27T00:00:00Z",
        granted_by="test", plan_id=PHASE_B_PLAN_V1.plan_id,
        plan_hash=PHASE_B_PLAN_V1.plan_hash, science_plan_hash="s" * 64,
        calibration_profile_hashes={
            DOMAIN_BALANCED_V1.qualified_id: DOMAIN_BALANCED_V1.profile_hash,
            REASONING_HEAVY_V2.qualified_id: REASONING_HEAVY_V2.profile_hash},
        calibration_content_hashes={
            DOMAIN_BALANCED_V1.qualified_id: DOMAIN_BALANCED_V1.content_sha256,
            REASONING_HEAVY_V2.qualified_id: REASONING_HEAVY_V2.content_sha256},
        planning_floor_usd=13.08, hard_cap_usd=26.8049,
        authorized_stages=(0, 1, 2, 3, 4, 5), stage_conditions={},
        scope_note="seam test",
        source_digest=phase_b_source_digest(REPO)["digest"],
        authorized_session_commit="0" * 40)
    fields.update(over)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(PhaseBAuthorization(**fields).as_dict()))
    return PhaseBAuthorization.load(path), path


def test_require_harness_actually_RE_DERIVES_the_phase_b_digest(tmp_path):
    """Not an alias returning a stored string: the real derivation, over the real
    58 files, failing closed when it disagrees."""
    from aadistill.autoinit.authorization import AuthorizationError
    from aadistill.autoinit.phase_b import phase_b_source_digest

    auth, _ = _issued(tmp_path)
    observed = auth.require_harness(REPO)
    assert observed["digest"] == phase_b_source_digest(REPO)["digest"]
    assert len(observed["files"]) == 58
    assert observed["not_yet_covered"] == []
    # It is the Phase-B set, not Phase A's.
    paths = {e["path"] for e in observed["files"]}
    assert "scripts/pod/autoinit_phase_b_driver.py" in paths
    assert "src/aadistill/autoinit/phase_b.py" in paths

    stale, _ = _issued(tmp_path / "stale", source_digest="0" * 64)
    with pytest.raises(AuthorizationError, match="Re-rehearse and re-issue"):
        stale.require_harness(REPO)


def test_the_shared_session_runner_line_works_against_this_type(tmp_path):
    """`session_runner` does exactly `self.auth.require_harness(self.repo_root)`.
    That line is what raised for Phase B."""
    auth, _ = _issued(tmp_path)
    harness = auth.require_harness(REPO)          # the runner's line, verbatim
    assert harness["digest"] == auth.harness_source_digest
    assert tuple(e["path"] for e in harness["files"]) == tuple(sorted(auth.harness_source_files))


def test_session_commit_gate_runs_against_a_phase_b_authorization(tmp_path):
    """The gate that never executed. It reads `harness_source_files` and
    `harness_source_digest` off the authorization and re-derives both from git."""
    import subprocess

    from aadistill.infrastructure.session_prechecks import session_commit_gate

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=REPO).stdout.strip()
    auth, path = _issued(tmp_path)
    gate = session_commit_gate(REPO, str(path), check_lineage=False)
    ctx = types.SimpleNamespace(
        auth=auth, args=types.SimpleNamespace(session_commit=commit), evidence={})
    ok, why = gate(ctx)                      # must not raise, whatever it decides
    assert isinstance(ok, bool) and isinstance(why, str)

    # It ran, and it consulted the ALIAS fields. Before the repair this line was
    # never reached: the gate raised AttributeError on `harness_source_files`.
    record = ctx.evidence["session_commit_check"]
    assert record["authorized_harness_digest"] == auth.source_digest, (
        "the gate did not read the authorization's digest through the alias")
    assert len(record["harness_digest_at_commit"]) == 64, (
        "the gate never re-derived a digest from the commit's blobs, so it did "
        "not iterate harness_source_files")
    assert record["harness_matches"] is (
        record["harness_digest_at_commit"] == auth.source_digest)

    # And it fails closed on a digest it was not granted against, by that name.
    stale, stale_path = _issued(tmp_path / "stale", source_digest="0" * 64)
    ctx2 = types.SimpleNamespace(
        auth=stale, args=types.SimpleNamespace(session_commit=commit), evidence={})
    ok2, why2 = session_commit_gate(REPO, str(stale_path), check_lineage=False)(ctx2)
    assert not ok2 and "authorization was not granted against" in why2


def test_the_canonical_schema_still_has_ONE_name_per_identity(tmp_path):
    """`source_*` stays canonical in the JSON. Duplicating `harness_*` into the
    artifact would create a second thing to keep in step, which is the defect
    class the aliases exist to avoid."""
    auth, path = _issued(tmp_path)
    raw = json.loads(path.read_text())
    assert "source_digest" in raw and "source_files" in raw
    assert not any(k.startswith("harness_") for k in raw), sorted(raw)
    # The aliases exist on the object only, and agree with the canonical fields.
    assert auth.harness_source_digest == auth.source_digest
    assert auth.harness_source_files == auth.source_files


# --- repair 4: the pod is not asked for bytes only the dev box has -----------
#
# Attempt 1 died at $0.15 in the pod's setup gate, on two tests that reconstruct
# Phase-A citations from `/home/ecs-user/aad-artifacts` — a dev-box artifact
# store that is intentionally not transported. The repair moves the question to
# the machine that can answer it. These tests hold that boundary from both
# sides: nothing pod-run may reach for the store, and the check must still
# happen before a pod exists.


def _live_like(**over):
    """A `verify()`-shaped result taken from the real record, then mutated.

    Derived from the producer's own output rather than hand-drawn: a fixture
    invented to match what the consumer wants is how a stub ends up certifying
    the defect it was supposed to catch.
    """
    base = json.loads((REPO / "logs/autoinit_historical_probe_reuse.json").read_text())
    return {**base, **over}


def test_no_pod_run_test_reconstructs_citations_from_the_dev_box_store():
    """The rule the abort taught, enforced against every test in the tree.

    Importing `verify` means calling into the retained checkpoint store. Any
    module that does so must be excluded from the pod's setup gate, or it will
    fail there for a reason that has nothing to do with the session.
    """
    import ast

    offenders = []
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text())
        pulls_verify = any(
            isinstance(n, ast.ImportFrom)
            and n.module == "verify_historical_probe_reuse"
            and any(a.name == "verify" for a in n.names)
            for n in ast.walk(tree))
        if pulls_verify:
            rel = path.relative_to(REPO).as_posix()
            if rel not in pbl.PHASE_B_TEST_IGNORES:
                offenders.append(rel)
    assert not offenders, (
        f"{offenders} reconstruct Phase-A citations from the dev-box checkpoint "
        "store but still run inside the pod's setup gate; that is the defect "
        "that aborted Phase-B attempt 1")


def test_the_host_local_module_exists_and_is_the_one_excluded():
    excluded = set(pbl.PHASE_B_TEST_IGNORES) - set(pbl.TEST_IGNORES)
    assert excluded == {"tests/autoinit/test_phase_b_reuse_hostlocal.py"}
    assert (REPO / "tests/autoinit/test_phase_b_reuse_hostlocal.py").is_file()
    # The exclusion is Phase-B-specific: Phase A's historical contract, which
    # completed sessions ran under, is not rewritten by this repair.
    assert pbl.TEST_IGNORES == ("tests/data/test_recovery_corpus_pipeline.py",
                               "tests/pod/test_phase_a_stages1_5_execute.py")
    assert set(pbl.PHASE_B_TEST_IGNORES) > set(pbl.TEST_IGNORES)


def test_the_session_actually_ships_the_phase_b_ignore_list(spec):
    session, _ = spec
    assert session.setup.test_ignores == pbl.PHASE_B_TEST_IGNORES
    assert "tests/autoinit/test_phase_b_reuse_hostlocal.py" in \
        session.setup.test_ignores_env()


def test_the_reconstruction_gate_runs_before_a_pod_exists(spec):
    session, _ = spec
    names = [getattr(g, "__name__", type(g).__name__) for g in session.precheck]
    assert "historical_reuse_reconstruction_gate" in names
    # Record first, then the bytes the record claims to describe.
    assert names.index("reuse_record_gate") < \
        names.index("historical_reuse_reconstruction_gate")


@pytest.mark.parametrize("mutation,expected", [
    ({"n_probes": 10}, "not the 11"),
    ({"reuse_verified": False, "failures": [{"probe_id": "x"}]}, "reconstruction failed"),
    ({"probes_dir_digest": "0" * 64}, "drifted"),
    ({"admitted_reusable_probes": ["cca699c93f34/sa"]}, "not all"),
])
def test_the_reconstruction_gate_refuses_a_broken_evidence_set(
        monkeypatch, mutation, expected):
    monkeypatch.setattr(pbl, "verify_historical_reuse",
                        lambda: _live_like(**mutation))
    ok, why = pbl.historical_reuse_reconstruction_gate(types.SimpleNamespace())
    assert not ok and expected in why, why


def test_the_gate_refuses_a_probe_that_no_longer_derives_from_BYTES(monkeypatch):
    """The load-bearing check, asserted per probe rather than via the verdict.

    A verifier that flipped only its summary field — or a record edited to say
    `reuse_verified: true` — must still be caught, because the gate reads the
    per-probe checks itself.
    """
    live = _live_like()
    live["probes"] = [dict(p) for p in live["probes"]]
    live["probes"][0] = {**live["probes"][0], "checks": {
        **live["probes"][0]["checks"], "artifact_digest_re_derives_from_bytes": False}}
    monkeypatch.setattr(pbl, "verify_historical_reuse", lambda: live)
    ok, why = pbl.historical_reuse_reconstruction_gate(types.SimpleNamespace())
    assert not ok and "no longer re-derive" in why


def test_the_gate_refuses_rather_than_crashing_when_the_verifier_raises(monkeypatch):
    def boom():
        raise FileNotFoundError("the retained store is gone")

    monkeypatch.setattr(pbl, "verify_historical_reuse", boom)
    ok, why = pbl.historical_reuse_reconstruction_gate(types.SimpleNamespace())
    assert not ok and "FileNotFoundError" in why


def test_the_gate_refuses_a_verifier_that_admits_a_DIFFERENT_candidate_set(
        monkeypatch):
    """The two halves must agree on who the priors are, not merely on the count."""
    monkeypatch.setattr(pbl, "REUSE_ADMITTED",
                        ("cca699c93f34", "158b96cf651f", "control-qwen"))
    ok, why = pbl.historical_reuse_reconstruction_gate(types.SimpleNamespace())
    assert not ok and "frozen Phase-B rule" in why


def test_the_required_citations_are_the_eight_the_budget_assumes():
    assert pbl.REQUIRED_CITATIONS == frozenset({
        "cca699c93f34/sa", "cca699c93f34/sb", "cca699c93f34/sc",
        "85bde4ded2c3/sa", "85bde4ded2c3/sb", "85bde4ded2c3/sc",
        "control-qwen/sa", "control-qwen/sb"})
    # The control has no sc on record; asserting sc for it would demand evidence
    # that never existed and block every launch.
    assert "control-qwen/sc" not in pbl.REQUIRED_CITATIONS


def test_the_verifier_is_inside_the_digest_the_grant_is_issued_against():
    """It decides whether a pod is created, so it is executable, not provenance."""
    from aadistill.autoinit.phase_b import (
        PHASE_B_EXECUTABLE_SOURCE_FILES_V1, PHASE_B_SOURCE_SET_VERSION,
    )
    assert "scripts/autoinit/verify_historical_probe_reuse.py" in \
        PHASE_B_EXECUTABLE_SOURCE_FILES_V1
    # And the pricing module, for the same reason: the launcher imports it to
    # derive the Stage-1 deadline, so it decides how long a paid search may run.
    assert "scripts/autoinit/price_phase_b.py" in PHASE_B_EXECUTABLE_SOURCE_FILES_V1
    assert PHASE_B_SOURCE_SET_VERSION == 4
