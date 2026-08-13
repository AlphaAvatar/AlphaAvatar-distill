"""Dataset-role isolation and calibration-profile discipline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.calibration import (  # noqa: E402
    DOMAIN_BALANCED_V1,
    REASONING_HEAVY_V1,
    STAGE0_CURRENT_V1,
    CalibrationError,
    CalibrationProfile,
    CalibrationSource,
    buildable_profiles,
    register_profile,
    unregister_profile,
)
from aadistill.autoinit.datasets import (  # noqa: E402
    DatasetAsset,
    DatasetRole,
    DatasetRoleViolation,
    assert_search_visible,
    assert_usable_for,
    check_role_isolation,
    protected_assets,
    register_asset,
    unregister_asset,
)


#: The E8a calibration mixture is a gitignored artifact. A pod session that does
#: not stage it (the micro-preflight does not) must skip these rather than fail
#: its setup test gate on inputs it was never given.
CALIB_ITEMS = REPO / "artifacts/stage1/e8_calibration_v1/items.jsonl"
needs_calibration = pytest.mark.skipif(
    not CALIB_ITEMS.is_file(),
    reason="E8a calibration mixture is a local artifact, not tracked in git")


def source(domain="general", n=1):
    return CalibrationSource("d", "rev", domain, n)


# --- role isolation ---------------------------------------------------------


def test_the_final_promotion_battery_is_not_reachable_from_the_search():
    """The mechanical form of 'the final battery is isolated from the search'."""
    assert DatasetRole.FINAL_PROMOTION not in aadistill.autoinit.datasets.SEARCH_VISIBLE_ROLES
    with pytest.raises(DatasetRoleViolation, match="may not"):
        assert_search_visible("battery.frozen_promotion_150")
    with pytest.raises(DatasetRoleViolation, match="may not"):
        assert_search_visible("battery.capability_v2_846")
    assert_search_visible("calib.e8a_domain_balanced_67")


def test_the_frozen_batteries_are_registered_as_protected():
    ids = {a.asset_id for a in protected_assets()}
    assert ids == {"battery.frozen_promotion_150", "battery.capability_v2_846"}
    battery = aadistill.autoinit.datasets.get_asset("battery.frozen_promotion_150")
    assert battery.metadata["inclusion_mask_sha256"] == (
        "d6e24e0b09da1bcc692b1dc96d8236808d29551a9fc94a47d1d968fd3f73d6ba")


def test_an_asset_cannot_be_used_outside_its_declared_role():
    with pytest.raises(DatasetRoleViolation, match="cannot be used as"):
        assert_usable_for("calib.e8a_domain_balanced_67", DatasetRole.FINAL_PROMOTION)
    assert assert_usable_for("calib.e8a_domain_balanced_67",
                             DatasetRole.OPERATOR_CALIBRATION)


def test_moving_an_asset_between_roles_is_refused():
    original = aadistill.autoinit.datasets.get_asset("calib.e8a_domain_balanced_67")
    moved = DatasetAsset(asset_id=original.asset_id, role=DatasetRole.FINAL_PROMOTION,
                         path=original.path, description=original.description)
    with pytest.raises(DatasetRoleViolation, match="already registered in role"):
        register_asset(moved)


def test_a_shared_prompt_across_roles_is_caught(tmp_path):
    """The leak the check exists for: the same question, two ids, two roles."""
    calib = tmp_path / "calib.jsonl"
    battery = tmp_path / "battery.jsonl"
    shared = {"prompt": "What is the capital of France?"}
    calib.write_text("\n".join(json.dumps(r) for r in
                     [{"id": "c1", **shared}, {"id": "c2", "prompt": "unique to calib"}]))
    battery.write_text("\n".join(json.dumps(r) for r in
                       [{"id": "b1", **shared}, {"id": "b2", "prompt": "unique to battery"}]))

    register_asset(DatasetAsset("t.calib", DatasetRole.OPERATOR_CALIBRATION,
                                "calib.jsonl", "test"), replace=True)
    register_asset(DatasetAsset("t.battery", DatasetRole.FINAL_PROMOTION,
                                "battery.jsonl", "test"), replace=True)
    try:
        with pytest.raises(DatasetRoleViolation, match="in-sample"):
            check_role_isolation(["t.calib", "t.battery"], repo_root=tmp_path)
        # Disjoint content passes, and the report says what it checked.
        battery.write_text(json.dumps({"id": "b1", "prompt": "something else entirely"}))
        report = check_role_isolation(["t.calib", "t.battery"], repo_root=tmp_path)
        assert report["passed"] and not report["overlaps"]
        assert {c["asset_id"] for c in report["checked_assets"]} == {"t.calib", "t.battery"}
    finally:
        unregister_asset("t.calib")
        unregister_asset("t.battery")


def test_an_unloadable_asset_fails_the_check_rather_than_being_skipped(tmp_path):
    register_asset(DatasetAsset("t.missing", DatasetRole.STATE_EVALUATION,
                                "nope.jsonl", "test"), replace=True)
    try:
        with pytest.raises(DatasetRoleViolation, match="proves nothing"):
            check_role_isolation(["t.missing"], repo_root=tmp_path)
        report = check_role_isolation(["t.missing"], repo_root=tmp_path,
                                      require_loadable=False)
        assert report["unloadable"] and not report["complete"]
    finally:
        unregister_asset("t.missing")


def test_an_unreadable_item_shape_raises_instead_of_hashing_the_empty_string(tmp_path):
    path = tmp_path / "weird.jsonl"
    path.write_text(json.dumps({"id": "x", "payload": "no prompt field"}))
    register_asset(DatasetAsset("t.weird", DatasetRole.OPERATOR_CALIBRATION,
                                "weird.jsonl", "test"), replace=True)
    try:
        with pytest.raises(DatasetRoleViolation, match="cannot derive any comparable"):
            aadistill.autoinit.datasets.get_asset("t.weird").identity_sets(tmp_path)
    finally:
        unregister_asset("t.weird")


@needs_calibration
def test_the_real_e8a_calibration_mixture_is_still_where_the_record_says():
    asset = aadistill.autoinit.datasets.get_asset("calib.e8a_domain_balanced_67")
    items = asset.load_items(REPO)
    assert len(items) == 67
    # It is pre-tokenized, so its comparable identity is the token sequence and
    # the recorded document hash — not prompt text, which it does not carry.
    identities = asset.identity_sets(REPO)
    assert identities["token_ids"] and len(identities["token_ids"]) == 67
    assert "prompt_content" not in identities
    assert asset.prompt_hashes(REPO) == set()


def test_two_roles_stored_in_different_forms_are_reported_not_passed(tmp_path):
    """The failure mode a naive prompt-hash check would never report.

    A tokenized calibration mixture and a text battery share no identity kind, so
    an intersection of their hash sets is empty for *every* input — including an
    input that leaks. That must read as 'could not compare', never as 'clean'.
    """
    (tmp_path / "tok.jsonl").write_text(
        json.dumps({"item_id": "a", "ids": [1, 2, 3]}))
    (tmp_path / "text.jsonl").write_text(
        json.dumps({"id": "b", "prompt": "some question"}))
    register_asset(DatasetAsset("t.tok", DatasetRole.OPERATOR_CALIBRATION,
                                "tok.jsonl", "test"), replace=True)
    register_asset(DatasetAsset("t.text", DatasetRole.FINAL_PROMOTION,
                                "text.jsonl", "test"), replace=True)
    try:
        with pytest.raises(DatasetRoleViolation, match="share no identity kind"):
            check_role_isolation(["t.tok", "t.text"], repo_root=tmp_path)
        report = check_role_isolation(["t.tok", "t.text"], repo_root=tmp_path,
                                      require_loadable=False)
        assert report["uncomparable_role_pairs"]
        assert not report["complete"]
    finally:
        unregister_asset("t.tok")
        unregister_asset("t.text")


def test_a_token_level_leak_is_caught_between_tokenized_assets(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"item_id": "a", "ids": [7, 8, 9]}))
    (tmp_path / "b.jsonl").write_text(json.dumps({"item_id": "b", "ids": [7, 8, 9]}))
    register_asset(DatasetAsset("t.a", DatasetRole.OPERATOR_CALIBRATION,
                                "a.jsonl", "test"), replace=True)
    register_asset(DatasetAsset("t.b", DatasetRole.STATE_EVALUATION,
                                "b.jsonl", "test"), replace=True)
    try:
        with pytest.raises(DatasetRoleViolation, match="in-sample"):
            check_role_isolation(["t.a", "t.b"], repo_root=tmp_path)
    finally:
        unregister_asset("t.a")
        unregister_asset("t.b")


# --- calibration profiles ---------------------------------------------------


def test_the_three_v1_profiles_are_representable_and_only_one_is_built():
    assert [p.profile_id for p in (STAGE0_CURRENT_V1, DOMAIN_BALANCED_V1,
                                   REASONING_HEAVY_V1)] == [
        "calib.stage0_current", "calib.domain_balanced", "calib.reasoning_heavy"]
    assert [p.qualified_id for p in buildable_profiles()] == ["calib.domain_balanced@v1"]


def test_an_unbuilt_profile_refuses_to_resolve():
    """Representable is not the same as usable. No invented hashes."""
    assert REASONING_HEAVY_V1.content_sha256 is None
    with pytest.raises(CalibrationError, match="not built"):
        REASONING_HEAVY_V1.resolve(REPO)


@needs_calibration
def test_the_built_profile_resolves_and_re_derives_the_frozen_mixture_hash():
    """The E8a mixture identity is recomputed from the tokens, not trusted."""
    from aadistill.autoinit.calibration import mixture_content_sha256

    items = DOMAIN_BALANCED_V1.resolve(REPO)
    assert len(items) == 67
    assert mixture_content_sha256(items) == DOMAIN_BALANCED_V1.content_sha256 == (
        "d65c1f40e4837ea1bd5bcc33c68041a13b797c68f5be3c0686e0142ed761028f")
    # Reformatting the file would move its byte hash; it must not move the
    # mixture's identity, because the operator sees tokens.
    assert DOMAIN_BALANCED_V1.items_file_sha256 != DOMAIN_BALANCED_V1.content_sha256


@needs_calibration
def test_a_tampered_mixture_is_rejected_even_at_the_right_file_hash():
    from aadistill.autoinit.calibration import CalibrationError, mixture_content_sha256

    items = DOMAIN_BALANCED_V1.resolve(REPO)
    swapped = [{**items[0], "ids": items[1]["ids"]}, *items[1:]]
    assert mixture_content_sha256(swapped) != DOMAIN_BALANCED_V1.content_sha256
    with pytest.raises(CalibrationError, match="cannot be hashed"):
        mixture_content_sha256([{"item_id": "x"}])


def test_a_changed_mixture_needs_a_new_version_not_a_redefinition():
    original = CalibrationProfile(
        profile_id="test.mix", version=1, description="",
        sources=(source(),), domain_weights={"general": 1.0}, token_budget=10,
        sample_rule="fixed", seed=1)
    register_profile(original)
    try:
        changed = CalibrationProfile(
            profile_id="test.mix", version=1, description="",
            sources=(source(),), domain_weights={"general": 1.0}, token_budget=999,
            sample_rule="fixed", seed=1)
        with pytest.raises(CalibrationError, match="new version"):
            register_profile(changed)
        assert original.profile_hash != changed.profile_hash
    finally:
        unregister_profile("test.mix@v1")


def test_profile_validation_catches_incoherent_mixtures():
    with pytest.raises(CalibrationError, match="sum to"):
        CalibrationProfile(profile_id="x", version=1, description="",
                           sources=(source(),), domain_weights={"general": 0.9},
                           token_budget=1, sample_rule="f", seed=1)
    with pytest.raises(CalibrationError, match="do not match"):
        CalibrationProfile(profile_id="x", version=1, description="",
                           sources=(source("general"),),
                           domain_weights={"math": 1.0},
                           token_budget=1, sample_rule="f", seed=1)
    with pytest.raises(CalibrationError, match="not materialized"):
        CalibrationProfile(profile_id="x", version=1, description="",
                           sources=(source(),), domain_weights={"general": 1.0},
                           token_budget=1, sample_rule="f", seed=1,
                           content_sha256="deadbeef")


def test_a_profile_hash_covers_the_specification():
    a = CalibrationProfile(profile_id="x", version=1, description="one",
                           sources=(source(),), domain_weights={"general": 1.0},
                           token_budget=1, sample_rule="f", seed=1)
    b = CalibrationProfile(profile_id="x", version=1, description="a different note",
                           sources=(source(),), domain_weights={"general": 1.0},
                           token_budget=1, sample_rule="f", seed=1)
    assert a.profile_hash == b.profile_hash, "prose must not change the identity"
    c = CalibrationProfile(profile_id="x", version=1, description="one",
                           sources=(source(),), domain_weights={"general": 1.0},
                           token_budget=1, sample_rule="f", seed=2)
    assert a.profile_hash != c.profile_hash, "the sampling seed must change it"
