"""Phase B's session plan, its executable-source identity and its grant type.

The three things a future paid Phase B depends on being right before it starts:
what may execute, what terminates it, and what it is allowed to compare.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

from aadistill.autoinit.authorization import AuthorizationError  # noqa: E402
from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1, REASONING_HEAVY_V2,
)
from aadistill.autoinit.phase_a import (  # noqa: E402
    PHASE_A_AUTHORIZATION, PHASE_A_HARNESS_SOURCE_FILES_V1,
)
from aadistill.autoinit.phase_b import (  # noqa: E402
    CANONICAL_CONTROL, PHASE_A_EXCLUDED_LEAVES, PHASE_A_IMPORTED_FINALISTS,
    PHASE_B_DELEGATED_IDENTITIES, PHASE_B_EXECUTABLE_SOURCE_FILES_V1,
    PHASE_B_PLAN_V1, PHASE_B_SEARCHED_LEAVES, PHASE_B_UNCOVERED,
    PhaseBAuthorization, phase_b_source_digest,
)

PREREG = REPO / "logs/autoinit_phase_b_preregistration.json"


def _authorization(**overrides) -> PhaseBAuthorization:
    base = dict(
        authorization_id="test", granted_utc="2026-08-25T00:00:00Z",
        granted_by="test", plan_id=PHASE_B_PLAN_V1.plan_id,
        plan_hash=PHASE_B_PLAN_V1.plan_hash, science_plan_hash="s" * 64,
        calibration_profile_hashes={
            DOMAIN_BALANCED_V1.qualified_id: DOMAIN_BALANCED_V1.profile_hash,
            REASONING_HEAVY_V2.qualified_id: REASONING_HEAVY_V2.profile_hash},
        calibration_content_hashes={
            DOMAIN_BALANCED_V1.qualified_id: DOMAIN_BALANCED_V1.content_sha256,
            REASONING_HEAVY_V2.qualified_id: REASONING_HEAVY_V2.content_sha256},
        planning_floor_usd=1.0, hard_cap_usd=2.0, authorized_stages=(0, 1, 2, 3, 4, 5),
        stage_conditions={}, scope_note="test")
    return PhaseBAuthorization(**{**base, **overrides})


# --- what may execute -------------------------------------------------------


def test_the_phase_a_harness_is_not_widened_and_phase_b_has_its_own():
    """Phase A is complete; its harness identity is historical fact."""
    assert len(PHASE_A_HARNESS_SOURCE_FILES_V1) == 16
    assert PHASE_A_AUTHORIZATION.harness_source_files == PHASE_A_HARNESS_SOURCE_FILES_V1
    assert set(PHASE_B_EXECUTABLE_SOURCE_FILES_V1) != set(PHASE_A_HARNESS_SOURCE_FILES_V1)


def test_the_source_set_covers_what_a_paid_P2_SEARCH_actually_executes():
    covered = set(PHASE_B_EXECUTABLE_SOURCE_FILES_V1)
    for required in (
            "scripts/autoinit/phase_a_search.py",          # the entrypoint
            "src/aadistill/autoinit/search.py",            # the beam engine
            "src/aadistill/autoinit/ranking.py",           # objectives and schedule
            "src/aadistill/autoinit/calibration.py",       # profiles and resolve()
            "src/aadistill/autoinit/operators/base.py",    # operator contracts
            "src/aadistill/autoinit/arch.py",              # ArchSpec, adapter registry
            "src/aadistill/autoinit/adapters/qwen3.py",    # the family boundary
            "src/aadistill/autoinit/metrics.py",           # state evaluation
            "src/aadistill/autoinit/state.py",             # state identity
            "src/aadistill/autoinit/phase_b.py"):          # the plan itself
        assert required in covered, required
    # Every concrete operator that can execute, not just the base class.
    for operator in ("attention", "composite", "depth", "ffn", "width"):
        assert f"src/aadistill/autoinit/operators/{operator}.py" in covered, operator
    # The adapter package __init__ registers the adapter; the AST closure of
    # search.py alone never reaches it.
    assert "src/aadistill/autoinit/adapters/__init__.py" in covered
    assert "src/aadistill/autoinit/__init__.py" in covered


def test_the_set_is_provenance_closure_not_a_maximal_file_list():
    """What is absent is absent for a stated reason, not by oversight."""
    covered = set(PHASE_B_EXECUTABLE_SOURCE_FILES_V1)
    # The pod consumes a materialized mixture; it never runs the builder.
    assert "src/aadistill/autoinit/reweight.py" not in covered
    assert "scripts/data/build_reasoning_heavy_calibration.py" not in covered
    # The probe path is bound elsewhere, and the record says by what.
    # `recovery.py` is inside RECOVERY_SCORING_FILES_V2, which the driver binds
    # at stage 0, so a change to the selection rules it holds is still detected.
    assert "src/aadistill/autoinit/recovery.py" not in covered
    # `autoinit/generation.py` IS covered: it is not in GENERATION_SOURCE_FILES_V1
    # (that set is the evaluator), and it decides the protocol hash and the
    # comparability verdict the whole Stage-0 gate turns on.
    assert "src/aadistill/autoinit/generation.py" in covered
    assert "src/aadistill/autoinit/generation_compat.py" in covered
    for topic in ("probe training", "probe generation", "probe scoring",
                  "the calibration mixtures"):
        assert topic in PHASE_B_DELEGATED_IDENTITIES, topic
    assert "trainer_source_digest" in PHASE_B_DELEGATED_IDENTITIES["probe training"]
    assert "recovery.py" in PHASE_B_DELEGATED_IDENTITIES["probe scoring"]
    assert "provenance" in PHASE_B_DELEGATED_IDENTITIES["the calibration mixtures"]


def test_the_driver_and_launcher_now_EXIST_and_are_covered():
    """The blocker is closed. The field stays so a future gap fails closed."""
    covered = set(PHASE_B_EXECUTABLE_SOURCE_FILES_V1)
    for executable in ("scripts/pod/autoinit_phase_b_driver.py",
                       "scripts/pod/autoinit_phase_b_launch.py",
                       # The parent class: every inherited stage a Phase-B run
                       # executes lives here, so it is Phase-B runtime too.
                       "scripts/pod/autoinit_phase_a_driver.py",
                       "scripts/pod/autoinit_phase_a_launch.py",
                       # Shelled out to, so no import closure reaches them.
                       "scripts/pod/autoinit_preflight_setup.sh",
                       "scripts/pod/watchdog.py",
                       "scripts/pod/collect_artifacts.py",
                       "scripts/pod/autoinit_engine_probe.py",
                       "scripts/autoinit/verify_frozen_assets.py",
                       # Stage 0 imports build_frozen_plan from it.
                       "scripts/autoinit/write_preregistration.py",
                       # The session machinery the launcher runs.
                       "src/aadistill/infrastructure/session_runner.py",
                       "src/aadistill/infrastructure/session_prechecks.py"):
        assert executable in covered, executable
    assert PHASE_B_UNCOVERED == (), PHASE_B_UNCOVERED
    assert phase_b_source_digest(REPO)["not_yet_covered"] == []


def test_the_digest_fails_closed_on_a_missing_declared_file():
    with pytest.raises(AuthorizationError, match="missing"):
        phase_b_source_digest(REPO, files=("src/aadistill/autoinit/nope.py",))


def test_the_digest_moves_when_a_covered_file_moves():
    a = phase_b_source_digest(REPO)["digest"]
    b = phase_b_source_digest(
        REPO, files=tuple(f for f in PHASE_B_EXECUTABLE_SOURCE_FILES_V1
                          if f != "src/aadistill/autoinit/search.py"))["digest"]
    assert a != b


# --- what terminates it -----------------------------------------------------


def test_stage_zero_terminates_on_comparability_failure_rather_than_rerunning():
    """The correction: comparability is a precondition of the FROZEN thresholds,
    not merely a reuse convenience."""
    stage0 = PHASE_B_PLAN_V1.stages[0]
    assert stage0.blocking is True
    conditions = " ".join(stage0.stop_conditions)
    assert "TERMINATE before any search or probe" in conditions
    assert "re-running all eight candidates" in conditions
    assert "NOT rematerialized" in conditions and "NOT redefined" in conditions
    assert "feasibility floor and equivalence interval" in conditions


def test_the_plan_binds_both_calibration_identities_at_stage_zero():
    produces = " ".join(PHASE_B_PLAN_V1.stages[0].produces)
    assert "spec identity AND materialized content identity" in produces
    conditions = " ".join(PHASE_B_PLAN_V1.stages[0].stop_conditions)
    assert "profile_hash identifies the spec, not the bytes" in conditions


def test_the_search_stage_is_a_fresh_JOINT_p2_search():
    stage1 = PHASE_B_PLAN_V1.stages[1]
    assert "JOINT" in stage1.name or "JOINT" in stage1.purpose
    conditions = " ".join(stage1.stop_conditions)
    assert "Phase-A leaves are NOT injected" in conditions
    assert f"fewer than {PHASE_B_SEARCHED_LEAVES} admissible" in conditions


def test_no_tie_break_authority_leaks_into_the_terminal_stage():
    conditions = " ".join(PHASE_B_PLAN_V1.stages[5].stop_conditions)
    assert "search-side KL and NLL may NOT break a behavioural tie" in conditions
    assert "canonical Stage-1 NLL diagnostic may NOT break a tie" in conditions
    assert "unresolved_equivalence" in conditions
    assert "no follow-on experiment" in conditions


# --- what it may compare ----------------------------------------------------


def test_the_candidate_set_is_closed_and_the_exclusions_carry_their_reason():
    assert PHASE_B_SEARCHED_LEAVES == 5
    assert PHASE_A_IMPORTED_FINALISTS == ("cca699c93f34", "85bde4ded2c3")
    assert CANONICAL_CONTROL == "control-qwen"
    assert set(PHASE_A_EXCLUDED_LEAVES) == {
        "158b96cf651f", "281a02c3ac18", "4e429f7ed722"}
    for reason in PHASE_A_EXCLUDED_LEAVES.values():
        assert "behavioural admission" in reason
    conditions = " ".join(PHASE_B_PLAN_V1.stages[2].stop_conditions)
    assert "three excluded Phase-A leaves are NOT admitted" in conditions
    assert "whatever their retained sa evidence would have cost" in conditions


# --- the grant type ---------------------------------------------------------


def test_a_phase_b_grant_can_never_reopen_phase_a():
    auth = _authorization()
    assert auth.allows_phase_b is True
    assert auth.allows_phase_a is False
    assert auth.automatic_followon_start is False
    assert not issubclass(PhaseBAuthorization, type(PHASE_A_AUTHORIZATION))


def test_a_phase_a_authorization_cannot_be_loaded_as_a_phase_b_one(tmp_path):
    path = tmp_path / "phase_a.json"
    path.write_text(json.dumps(PHASE_A_AUTHORIZATION.as_dict()))
    with pytest.raises(AuthorizationError, match="not 'aadistill.autoinit.phase_b"):
        PhaseBAuthorization.load(path)


def test_an_edited_authorization_is_refused(tmp_path):
    raw = _authorization(source_digest="x" * 64).as_dict()
    raw["hard_cap_usd"] = 999.0
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(AuthorizationError, match="edited since it was granted"):
        PhaseBAuthorization.load(path)


def test_a_matching_spec_hash_is_NOT_enough_to_admit_a_mixture():
    """The whole point of binding two identities.

    A mixture rebuilt under a changed rule, or truncated, can still satisfy its
    spec hash. `profile_hash` says what was intended; `content_sha256` says what
    the operators will be fed.
    """
    from dataclasses import replace

    auth = _authorization()
    auth.require_calibration(REASONING_HEAVY_V2)          # both match -> fine
    drifted = replace(REASONING_HEAVY_V2, content_sha256="d" * 64)
    assert drifted.profile_hash == REASONING_HEAVY_V2.profile_hash
    with pytest.raises(AuthorizationError, match="not identify the sampled bytes"):
        auth.require_calibration(drifted)


def test_an_unlisted_profile_is_refused_outright():
    auth = _authorization(
        calibration_profile_hashes={DOMAIN_BALANCED_V1.qualified_id:
                                    DOMAIN_BALANCED_V1.profile_hash},
        calibration_content_hashes={DOMAIN_BALANCED_V1.qualified_id:
                                    DOMAIN_BALANCED_V1.content_sha256})
    with pytest.raises(AuthorizationError, match="not one of the authorized"):
        auth.require_calibration(REASONING_HEAVY_V2)


def test_an_authorization_without_a_source_digest_authorizes_nothing():
    with pytest.raises(AuthorizationError, match="declares no source_digest"):
        _authorization().require_source(REPO)


def test_a_drifted_executable_is_refused():
    with pytest.raises(AuthorizationError, match="Re-rehearse and re-issue"):
        _authorization(source_digest="0" * 64).require_source(REPO)


def test_the_live_executable_satisfies_its_own_digest():
    observed = phase_b_source_digest(REPO)
    _authorization(source_digest=observed["digest"]).require_source(REPO)


# --- the preregistration ----------------------------------------------------


@pytest.mark.skipif(not PREREG.is_file(), reason="preregistration not emitted")
def test_the_preregistration_binds_spec_AND_content_for_both_profiles():
    prereg = json.loads(PREREG.read_text())
    by_id = {p["qualified_id"]: p for p in prereg["calibration_profiles"]}
    assert set(by_id) == {"calib.domain_balanced@v1", "calib.reasoning_heavy@v2"}
    for qid, profile in by_id.items():
        assert len(profile["spec_identity"]["profile_hash"]) == 64
        assert len(profile["materialized_identity"]["content_sha256"]) == 64
        assert profile["materialized_identity"]["items_file_sha256"]
    assert by_id["calib.reasoning_heavy@v2"]["spec_identity"]["profile_hash"] == \
        REASONING_HEAVY_V2.profile_hash
    assert by_id["calib.reasoning_heavy@v2"]["materialized_identity"]["content_sha256"] == \
        REASONING_HEAVY_V2.content_sha256


@pytest.mark.skipif(not PREREG.is_file(), reason="preregistration not emitted")
def test_the_preregistration_reuses_the_frozen_science_plan_unchanged():
    prereg = json.loads(PREREG.read_text())
    science = prereg["science_plan"]
    assert science["plan_hash"] == (
        "02be33b9a7a8e26bc8bfb75795351e8cdc9ffd441b47066cc81887cfc511b55c")
    assert science["reused_unchanged"] is True
    assert science["equivalence_interval"] == 0.011695296982299022
    assert science["feasibility_floor"] == 0.3
    assert "rematerialize" in science["phase_b_does_not"]


@pytest.mark.skipif(not PREREG.is_file(), reason="preregistration not emitted")
def test_the_preregistration_freezes_the_whole_procedure_before_any_result():
    prereg = json.loads(PREREG.read_text())
    assert "BEFORE ANY PHASE-B RESULT EXISTS" in prereg["status"]
    assert prereg["search"]["profiles"] == 2 and prereg["search"]["joint"] is True
    assert prereg["search"]["beam_width"] == 6
    assert prereg["search"]["decomposed_paths"] == 288
    assert prereg["procedure"]["seeds"]["fourth_seed"] == "never"
    assert prereg["procedure"]["unresolved_is_a_result"] is True
    assert prereg["candidate_set"]["total_at_sa"] == 8
    # The live digest is checked separately, below. Phase-B STAGE 1 IS COMPLETE,
    # so the digest frozen here describes the tree that produced attempt 5 rather
    # than a launch-readiness property of the current tree.
    assert prereg["session_plan"]["plan_hash"] == PHASE_B_PLAN_V1.plan_hash
    forbidden = " ".join(prereg["procedure"]["tie_break_authority"]["may_NOT_break_a_tie"])
    for phrase in ("search-side KL", "search-side NLL", "canonical Stage-1 NLL",
                   "fourth seed"):
        assert phrase in forbidden, phrase
    assert "TERMINATE" in prereg["runtime_comparability_gate"]["on_fail"]
    assert "NOT an executable scientific fallback" in \
        prereg["runtime_comparability_gate"]["explicitly_not_a_fallback"]
    assert "No Phase-B grant or authorization exists" in prereg["not_authorized"]


@pytest.mark.skipif(not PREREG.is_file(), reason="preregistration not emitted")
def test_the_preregistration_is_self_verifying_AND_stable():
    """The identity excludes `generated_utc`.

    It used to include it, so every regeneration produced a new id even when not
    one byte of the commitment had moved — an identity that churns cannot be
    cited as "the frozen preregistration". The timestamp stays in the document as
    provenance.
    """
    from aadistill.infrastructure.manifest import sha256_json

    prereg = json.loads(PREREG.read_text())
    stated = prereg["preregistration_sha256"]
    material = {k: v for k, v in prereg.items()
                if k not in ("preregistration_sha256", "generated_utc")}
    assert sha256_json(material) == stated, "the preregistration has been edited"
    assert "generated_utc" in prereg, "provenance was dropped rather than excluded"
    # Hashing the timestamp in would give a different answer — that is the bug.
    with_time = {k: v for k, v in prereg.items() if k != "preregistration_sha256"}
    assert sha256_json(with_time) != stated


@pytest.mark.skipif(not PREREG.is_file(), reason="preregistration not emitted")
def test_the_operator_branching_rule_is_frozen_with_the_reasons():
    branching = json.loads(PREREG.read_text())["search"]["operator_branching"]
    assert branching["depth.positional_v0"]["profiles_branched_over"] == 1
    assert branching["attention.weight_proxy_v0"]["profiles_branched_over"] == 1
    for impl in ("width.global_pca_v0", "ffn.activation_importance_v0",
                 "depth.causal_kl_greedy_v1"):
        assert branching[impl]["profiles_branched_over"] == 2, impl
    assert "byte-identical" in branching["depth.positional_v0"]["note"]


@pytest.mark.skipif(not PREREG.is_file(), reason="preregistration not emitted")
def test_any_post_freeze_change_to_the_phase_b_executable_is_recorded_and_additive():
    """The frozen digest may move, but only for a recorded, reviewed reason.

    `autoinit_preflight_setup.sh` is in the Phase-B source set AND is the single
    `SESSION_KIND` dispatcher for every session this repository can launch, so
    adding the behavioural continuation necessarily touched it. Three responses
    were available and two are wrong: rewriting the preregistration destroys the
    evidence of what attempt 5 ran, and deleting this assertion destroys its
    meaning.

    The rule itself lives in `aadistill.autoinit.post_freeze` because the paid
    launcher's `preregistration_gate` enforces exactly the same one. A test that
    reimplemented it would be free to drift from the gate it claims to describe.
    """
    from aadistill.autoinit.post_freeze import accounted_for

    prereg = json.loads(PREREG.read_text())
    ok, why = accounted_for(prereg["executable_source"]["digest"],
                            phase_b_source_digest(REPO)["digest"], REPO)
    assert ok, why


def test_the_drift_rule_refuses_everything_it_should():
    """Guards the guard: an allowance that allows everything is not a gate."""
    import json as _json

    from aadistill.autoinit.post_freeze import NOTE_PATH, accounted_for

    note_path = REPO / NOTE_PATH
    note = _json.loads(note_path.read_text())
    frozen, live = note["frozen_digest"], note["post_freeze_digest"]
    original = note_path.read_text()

    def with_note(mutate) -> tuple[bool, str]:
        broken = _json.loads(original)
        mutate(broken)
        note_path.write_text(_json.dumps(broken, indent=2) + "\n")
        try:
            return accounted_for(frozen, live, REPO)
        finally:
            note_path.write_text(original)

    def stale(n):        n["post_freeze_digest"] = "0" * 64
    def not_additive(n): n["change"]["additive_only"] = False
    def removals(n):     n["change"]["lines_removed"] = 3
    def touched(n):      n["dispatch_branches"]["pre_existing_changed"] = ["phase_b"]
    def lied(n):
        n["dispatch_branches"]["pre_existing_unchanged"]["phase_b"] = "f" * 64
    def wrong_freeze(n): n["frozen_digest"] = "1" * 64

    for name, mutate in (("stale digest", stale), ("non-additive", not_additive),
                         ("lines removed", removals), ("branch touched", touched),
                         ("branch hash lied about", lied),
                         ("different freeze", wrong_freeze)):
        ok, why = with_note(mutate)
        assert not ok, f"the drift rule accepted a note with a {name}"
        assert why

    # An undeclared change fails even with no note at all.
    moved = note_path.read_text()
    note_path.unlink()
    try:
        ok, _ = accounted_for(frozen, live, REPO)
        assert not ok, "undeclared drift was accepted"
    finally:
        note_path.write_text(moved)

    # And the identity case still passes without any note involvement.
    ok, _ = accounted_for(frozen, frozen, REPO)
    assert ok
