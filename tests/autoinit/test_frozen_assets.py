"""The two new frozen search assets, and the isolation between all five roles.

These tests read the built artifacts rather than rebuilding them: the builders
need the teacher tokenizer and the recovery corpus, and the point here is that
what was frozen stays frozen and stays consistent with what the search will
assume about it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import aadistill.autoinit  # noqa: F401,E402
from aadistill.autoinit.calibration import mixture_content_sha256  # noqa: E402
from aadistill.autoinit.metrics import StateEvalSuite  # noqa: E402

STATE_EVAL = REPO / "artifacts/stage1/state_eval_v1"
#: The battery that is actually staged to a pod and actually evaluated. It was
#: still `recovery_search_v1` after the v2 migration — v1 is INVALID before first
#: use and is no longer staged anywhere, so these tests passed on the dev box,
#: where v1 remains on disk, and died on the pod's test gate, which is a blocking
#: setup step. That cost $0.63 and a full setup cycle on 2026-08-14.
RECOVERY_SEARCH = REPO / "artifacts/stage3/recovery_search_v2"
ISOLATION = REPO / "logs/autoinit_role_isolation.json"

#: Both assets are gitignored, so a checkout that staged neither must SKIP rather
#: than error. Guarding only on STATE_EVAL let a missing battery raise instead.
pytestmark = pytest.mark.skipif(
    not (STATE_EVAL / "manifest.json").is_file()
    or not (RECOVERY_SEARCH / "manifest.json").is_file(),
    reason="frozen search assets not built in this checkout")


def load(path):
    return json.loads(path.read_text())


def items(directory, name="items.jsonl"):
    return [json.loads(l) for l in (directory / name).open() if l.strip()]


# --- STATE_EVALUATION -------------------------------------------------------


def test_the_state_eval_suite_matches_its_frozen_content_hash():
    manifest = load(STATE_EVAL / "manifest.json")
    assert mixture_content_sha256(items(STATE_EVAL)) == manifest["content_sha256"]
    assert manifest["role"] == "INITIALIZER_STATE_EVAL"


def test_every_declared_sub_type_has_items():
    """A silently absent sub-type reweights its domain."""
    manifest = load(STATE_EVAL / "manifest.json")
    rows = items(STATE_EVAL)
    for domain, subtypes in manifest["domains"].items():
        for subtype in subtypes:
            present = [r for r in rows if r["subtype"] == subtype]
            assert present, f"{domain}/{subtype} has no items"
            assert {r["domain"] for r in present} == {domain}


def test_the_suite_declaration_matches_what_the_evaluator_will_require():
    """The built artifact and the runtime suite object must agree."""
    manifest = load(STATE_EVAL / "manifest.json")
    suite = StateEvalSuite(
        suite_id=manifest["suite_id"], version=manifest["version"],
        domains=tuple(manifest["domains"]),
        subtypes={k: tuple(v) for k, v in manifest["domains"].items()},
        critical_tags=tuple(manifest["critical_tags"]),
        content_sha256=manifest["content_sha256"],
        n_items=manifest["counts"]["n_items"],
        general_domain="general")
    required = suite.required_metrics()
    for domain in manifest["domains"]:
        assert f"state.teacher_kl.{domain}" in required
    assert "state.teacher_kl.worst_domain" in required
    assert not any(k.startswith("state.nll") for k in required)
    assert "general" in manifest["domains"]


def test_every_critical_class_clears_its_position_floor():
    """The classes are averaged unweighted, so a thin one is pure noise."""
    manifest = load(STATE_EVAL / "manifest.json")
    tags = manifest["counts"]["tag_positions"]
    floor = manifest["sampling_rule"]["min_critical_positions"]
    for name in manifest["critical_tags"]:
        assert tags[name] >= floor, f"{name} has {tags[name]} positions, floor {floor}"
    # The broad diagnostic classes are recorded but excluded from the aggregate.
    assert set(manifest["diagnostic_tags"]) == {"assistant", "reasoning"}
    assert not set(manifest["critical_tags"]) & set(manifest["diagnostic_tags"])


def test_the_suite_is_small_enough_that_recompute_stays_cheap():
    """The reference is recomputed per candidate; size is a real cost."""
    manifest = load(STATE_EVAL / "manifest.json")
    positions = manifest["counts"]["total_prediction_positions"]
    assert positions < 150_000, (
        "a suite this large makes the per-candidate teacher forward the dominant "
        "search cost; shrink it or justify the change")
    # What caching it instead would have cost, at the real vocabulary.
    assert positions * 151_936 * 4 / 2**30 > 30, (
        "sanity: caching full-vocabulary reference logits is still tens of GiB")


def test_the_reference_is_the_original_teacher():
    manifest = load(STATE_EVAL / "manifest.json")
    assert manifest["reference_model"]["role"] == "original_teacher"
    assert manifest["reference_model"]["id"] == "Qwen/Qwen3-4B-Thinking-2507"
    assert "never the parent state" in manifest["reference_model"]["note"]


def test_the_suite_pins_its_tokenizer_and_template():
    manifest = load(STATE_EVAL / "manifest.json")
    tok = manifest["tokenizer"]
    for field in ("tokenizer_sha256", "tokenizer_config_sha256",
                  "chat_template_sha256"):
        assert len(tok[field]) == 64
    assert set(tok["special_token_ids"]) == {"think_open", "think_close", "im_end",
                                             "tool_call_close"}


# --- RECOVERY_SEARCH --------------------------------------------------------


def test_the_recovery_battery_matches_its_frozen_hashes():
    from aadistill.infrastructure.manifest import sha256_file

    manifest = load(RECOVERY_SEARCH / "manifest.json")
    assert manifest["role"] == "RECOVERY_SEARCH"
    for name, entry in manifest["sets"].items():
        path = REPO / entry["path"]
        assert sha256_file(path) == entry["sha256"], name
        assert len(items(RECOVERY_SEARCH, f"{name}.jsonl")) == entry["n"]


def test_correctness_is_computed_only_over_sets_with_a_frozen_scorer():
    """An unvalidated scorer must not reach the selection path."""
    from aadistill.evaluation.capability import SCORERS

    manifest = load(RECOVERY_SEARCH / "manifest.json")
    scorable = set(manifest["scorable_sets"])
    behaviour_only = set(manifest["behaviour_only_sets"])
    assert scorable and behaviour_only
    assert not scorable & behaviour_only
    for name in scorable:
        assert name in SCORERS or name in ("gsm8k", "tool"), (
            f"{name} is marked scorable but has no frozen scorer")
    # Code stays behaviour-only; tool became scorable after the audit.
    assert behaviour_only == {"code"}
    assert "trades code capability for math" in \
        manifest["scorers"]["behaviour_only_note"]
    assert manifest["n_scorable_prompts"] == 170


def test_tool_items_carry_what_the_frozen_scorer_consumes():
    """The battery stores the scorer's shape, not a shape it has to guess."""
    from aadistill.evaluation.behavior import score_tool_call

    rows = items(RECOVERY_SEARCH, "tool.jsonl")
    assert rows and all(r["scorable"] for r in rows)
    for row in rows:
        assert row["correctness_field"] == "tool_call_exact_match"
        assert row["scorer_tools"] and row["gold_tool_calls"]
        for tool in row["scorer_tools"]:
            assert "function" in tool and "name" in tool["function"]
        for call in row["gold_tool_calls"]:
            assert set(call["function"]) == {"name", "arguments"}
    # The gold calls, replayed, score as an exact match through the real scorer.
    import json as _json
    row = rows[0]
    answer = "".join(
        f"<tool_call>{_json.dumps({'name': c['function']['name'], 'arguments': c['function']['arguments']})}</tool_call>"
        for c in row["gold_tool_calls"])
    verdict = score_tool_call(answer, row["scorer_tools"], row["gold_tool_calls"])
    assert verdict["tool_call_exact_match"] is True
    assert verdict["tool_name_valid"] is True


def test_the_tool_scoring_audit_backs_the_scorable_decision():
    audit = load(REPO / "logs/autoinit_tool_scoring_audit.json")
    assert audit["all_cases_behave_as_required"]
    assert audit["connection_is_mechanical"]
    assert audit["correctness_field_if_scorable"] == "tool_call_exact_match"
    # The one interpretive field is excluded from correctness.
    assert any("tool_args_schema_ok" in f
               for f in audit["fields_not_used_for_correctness"])
    assert set(audit["cases_run"]) == {
        "known_good", "malformed_json", "wrong_tool_name",
        "missing_required_args", "wrong_argument_values",
        "protocol_invalid_no_wrapper"}


# --- canonical control availability -----------------------------------------


def test_both_historical_checkpoints_are_available_and_lineage_valid():
    """Lineage only. Whether they may serve as controls is decided elsewhere.

    An entry whose bytes no longer exist is allowed — one relay copy was deleted
    on 2026-08-15 by maintainer instruction to reclaim space — but only if it
    SAYS so. A record that quietly reports availability for an artifact that is
    gone is the failure this guards against, so a deleted entry must carry its
    deletion provenance and keep the hashes that remain its identity.
    """
    report = load(REPO / "logs/autoinit_control_availability.json")
    assert report["relay_reachable"]
    assert report["both_pass_legacy_lineage_subset"], report["consequence"]
    for name, entry in report["controls"].items():
        if not entry["present_on_relay"]:
            assert entry["artifact_available"] is False, name
            assert entry["hash_verified"] is False, name
            assert entry["relay_copy_deleted_utc"], name
            assert "DELETED" in entry["relay_path_status"], name
            assert entry["expected_weights_sha256"], name
        else:
            assert entry["artifact_available"] and entry["hash_verified"], name
        assert entry["weights_hash_match"], name
        assert entry["config_hash_match"], name
        assert entry["lineage_valid"], (name, entry.get("lineage_checks"))
        # Lineage means it descends from the frozen 0.86M protocol's inputs.
        checks = entry["lineage_checks"]
        assert checks["descends_from_canonical_init"]
        assert checks["rung_is_0860k"] and checks["seed_matches"]
        assert checks["ce_weight"] and checks["kd_weight"] and checks["kd_scope"]


def test_the_control_record_does_not_claim_they_are_phase_a_controls():
    report = load(REPO / "logs/autoinit_control_availability.json")
    assert report["any_recipe_matched_control"] is False
    consequence = report["consequence"]
    assert "NOT Phase-A matched controls" in consequence
    assert "Rerun canonical sa/sb" in consequence
    # The superseded claim must not come back.
    assert "profiling/evaluation job" not in consequence
    assert "no recovery retraining is needed" not in consequence
    assert report["controls"]["e1_r0860k_sa_pca"][
        "recipe_matched_control_decided_by"].endswith(
            "autoinit_recovery_fingerprint_audit.json")


def test_the_battery_declares_no_weighted_scalar():
    manifest = load(RECOVERY_SEARCH / "manifest.json")
    metrics = manifest["metrics"]
    assert "usable_rollout_rate" in metrics["feasibility"]
    assert "correct_overall" in metrics["primary"]
    assert "SCORABLE" in metrics["primary"]
    assert "never combined" in metrics["no_weighted_scalar"]


def test_battery_prompts_are_unique_and_hash_indexed():
    manifest = load(RECOVERY_SEARCH / "manifest.json")
    index = manifest["prompt_sha256_index"]
    assert len(set(index.values())) == len(index)
    assert len(index) == manifest["n_prompts"]


def test_battery_ids_are_reproducible_not_process_dependent():
    """`hash()` is randomized per process and must not reach a frozen id."""
    ids = [r["id"] for name in load(RECOVERY_SEARCH / "manifest.json")["sets"]
           for r in items(RECOVERY_SEARCH, f"{name}.jsonl")]
    assert len(set(ids)) == len(ids)
    gsm = sorted(i for i in ids if i.startswith("gsm8k-"))
    assert all(i.split("-")[-1].isdigit() for i in gsm)


# --- isolation --------------------------------------------------------------


def test_the_equivalence_denominator_matches_the_battery():
    """The interval's n_pooled must track the scorable count, not a stale constant."""
    manifest = load(RECOVERY_SEARCH / "manifest.json")
    prereg = load(REPO / "logs/autoinit_phase_a_preregistration.json")
    rule = prereg["recovery"]["selection_rules"]["equivalence_rule"]
    assert rule["n_pooled"] == manifest["n_scorable_prompts"] * 2
    assert rule["status"] == "PENDING_CONTROL_CHARACTERIZATION"
    assert rule["value"] is None, (
        "the interval must not be pre-filled from a prior; that would be the "
        "second definition again")


def test_the_five_roles_are_isolated_and_the_check_was_complete():
    report = load(ISOLATION)
    assert report["passed"], report["exact_overlaps"]
    assert report["complete"], report["uncomparable_role_pairs"]
    assert report["exact_overlaps"] == []
    assert report["uncomparable_role_pairs"] == [], (
        "a role pair sharing no identity type cannot be checked for leakage in "
        "either direction")


def test_every_role_pair_was_compared_on_a_real_identity():
    report = load(ISOLATION)
    roles = set(report["roles"])
    expected_pairs = {(a, b) for i, a in enumerate(sorted(roles))
                      for b in sorted(roles)[i + 1:]}
    compared = {(p["role_a"], p["role_b"]) for p in report["pairs_compared"]}
    assert {tuple(sorted(p)) for p in compared} == {tuple(sorted(p))
                                                    for p in expected_pairs}
    for pair in report["pairs_compared"]:
        assert pair["compared_on"], pair
        assert "item_id" not in pair["compared_on"], (
            "an id collision without a content collision is a naming coincidence")


def test_residual_near_duplicates_are_recorded_not_hidden():
    report = load(ISOLATION)
    counts = report["near_duplicate_counts"]
    assert counts["flagged_by_enforced_rule"] <= 5, (
        "more residual near-duplicates than the record accounts for")
    # Both readings are present, so a reader can see what was enforced and what a
    # stricter rule would have said.
    assert counts["strict"] >= counts["flagged_by_enforced_rule"]
    assert report["near_duplicate_rule"]["min_words_for_enforced_rule"] > 0


def test_the_promotion_battery_is_never_a_search_input():
    report = load(ISOLATION)
    for pair in report["pairs_compared"]:
        if "FINAL_PROMOTION" not in (pair["role_a"], pair["role_b"]):
            continue
        assert not pair["exact_overlaps"], (
            f"{pair['role_a']} and {pair['role_b']} share content; the final "
            "number would be in-sample")
