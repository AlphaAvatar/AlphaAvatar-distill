"""The SELECTED forward baseline: it resolves, and it is the method D1 ran.

Maintainer decision 2026-10-17 adopted D1's initialization methodology with
`prediction_rows=scored_v1`. These tests hold the record to the records it
claims to describe -- D1's own search wiring and q2's recorded identities --
so the baseline cannot drift from the method it names, and they hold the one
distinction the decision insists on: q2 was produced under `all_v1`, and the
forward profile has no checkpoint of its own.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

D1_SEARCH = "logs/stages/stage-1/phase_d1/runs/d1_search_20261006_210210"
Q2_STATE = "afd50c9ff0e3d20542bb5473ea745b55"


@pytest.fixture(scope="module")
def baseline():
    from shared.forward_baseline import register_operators, selected

    b = selected(REPO)
    register_operators(b)
    return b


@pytest.fixture(scope="module")
def record():
    return json.loads((REPO / "configs/shared/baselines/"
                       "initialization_forward_baseline_v1.json").read_text())


def _d1_wiring() -> dict:
    return json.loads((REPO / D1_SEARCH / "evidence/d1_search.json")
                      .read_text())["wiring"]


def _q2_leaf() -> dict:
    for line in (REPO / D1_SEARCH / "evidence/complete_leaves.jsonl").open():
        leaf = json.loads(line)
        if leaf["state_id"] == Q2_STATE:
            return leaf
    raise AssertionError("q2 is not among D1's complete leaves")


class TestItIsTheMethodD1Ran:
    def test_the_operators_and_their_signatures(self, baseline):
        leaf = _q2_leaf()
        recorded = {s["kind"]: (s["impl_id"], s["impl_signature_hash"])
                    for s in leaf["steps"]}
        named = {k: (v["impl_id"], v["signature_hash"])
                 for k, v in baseline.operators.items()}
        assert named == recorded
        assert sorted(baseline.implementation_ids().values()) == \
            sorted(_d1_wiring()["allowed_impls"])

    def test_the_scoring_positions_support_and_beam(self, baseline):
        w = _d1_wiring()
        assert baseline.position_policy.qualified_id == \
            w["position_policy"]["position_policy"]
        assert baseline.position_policy.policy_hash == \
            w["position_policy"]["position_policy_hash"]
        assert baseline.distribution_support.as_dict() == \
            w["distribution_support"]
        assert {"width": baseline.schedule.width,
                "warmup_levels": baseline.schedule.warmup_levels} == w["beam"]

    def test_the_ranking_policy_is_the_one_the_search_recorded(self, baseline):
        doc = json.loads((REPO / D1_SEARCH / "evidence/d1_search.json")
                         .read_text())
        rankings = {lv["ranking"]["policy_hash"]
                    for lv in doc["search"]["levels"] if lv.get("ranking")}
        assert rankings == {baseline.ranking_policy.policy_hash}

    def test_the_d_series_protocol_constants_agree(self, baseline):
        """Corroboration, not a second owner: the D-series protocol module is
        historical and keeps its constants; the baseline must agree with it."""
        from stages.d_series import scoring_protocol as SP

        assert baseline.distribution_support == SP.D_SERIES_SUPPORT
        assert baseline.execution.micro_batch_size == SP.D_SERIES_MICRO_BATCH_SIZE
        assert baseline.execution.calibration_batch_packing == \
            SP.D_SERIES_BATCH_PACKING

    def test_depth_scores_against_the_parent_and_state_eval_the_teacher(
            self, baseline):
        assert baseline.kl_reference_config("DEPTH") == {}
        assert baseline.state_evaluation_reference == "lineage_root"


class TestScoredV1IsTheForwardProfile:
    def test_the_forward_profile_selects_rows(self, baseline):
        assert baseline.execution.prediction_rows == "scored_v1"
        assert baseline.execution.selects_prediction_rows

    def test_the_core_default_is_still_the_historical_path(self, baseline):
        """Forward work reads its profile from the record; the core default
        stays `all_v1` so every historical identity stays byte-identical."""
        from aadistill.initialization.execution import DEFAULT_EXECUTION

        assert DEFAULT_EXECUTION.prediction_rows == "all_v1"
        assert baseline.execution != DEFAULT_EXECUTION

    def test_session_terms_carry_the_forward_profile(self, baseline):
        from shared.forward_baseline import session_terms

        terms = session_terms(baseline)
        assert terms["execution"].prediction_rows == "scored_v1"
        assert terms["baseline"]["numerical_execution_fingerprint"] == \
            baseline.execution_fingerprint


class TestQ2IsTheHistoricalReferenceAndNotTheForwardArtifact:
    def test_q2_identities_are_the_recorded_ones(self, baseline):
        leaf = _q2_leaf()
        (q2,) = baseline.reference_artifacts
        assert q2.semantic_state_id == Q2_STATE
        assert q2.identities["artifact_digest"] == \
            leaf["materialization"]["artifact_digest"]
        assert q2.numerical_execution_fingerprint == \
            leaf["materialization"]["numerical_execution_fingerprint"]
        reg = json.loads((REPO / "logs/maintenance/inventories/"
                          "checkpoint_registry.json").read_text())
        ident = reg["scientific_artifacts"]["d1_finalist_q2"]["identity"]
        for field in ("artifact_digest", "weights_digest",
                      "single_shard_sha256", "arch_signature"):
            assert q2.identities[field] == ident[field], field

    def test_q2_was_materialized_under_all_v1(self, baseline):
        (q2,) = baseline.reference_artifacts
        assert q2.execution.prediction_rows == "all_v1"
        assert q2.numerical_execution_fingerprint != \
            baseline.execution_fingerprint

    def test_the_forward_profile_has_no_artifact_yet(self, baseline, record):
        assert record["forward_materializations"] == []
        assert baseline.forward_materialization() is None
        assert baseline.reference_artifact_for(baseline.execution) is None

    def test_the_registry_protects_q2_as_the_baselines_reference(
            self, baseline):
        reg = json.loads((REPO / "logs/maintenance/inventories/"
                          "checkpoint_registry.json").read_text())
        row = reg["scientific_artifacts"]["d1_finalist_q2"]
        assert row["retention"] == "canonical"
        assert row["forward_baseline"]["baseline"] == baseline.qualified_id
        assert row["forward_baseline"]["is_the_forward_profile_artifact"] is False
        protected = [c for c in reg["checkpoints"]
                     if c["path_local"].endswith(Q2_STATE)]
        assert protected and all(c["protected"] for c in protected)


class TestTheHistoricalComparatorIsAnotherQuestion:
    def test_b_is_still_the_standing_incumbent(self, baseline):
        from stages.d_series.incumbent import standing_incumbent

        b = standing_incumbent(REPO)
        assert b["artifact_digest"].startswith("53e30566")
        (q2,) = baseline.reference_artifacts
        assert b["artifact_digest"] != q2.identities["artifact_digest"]


class TestProvenanceIsCheckable:
    def test_every_cited_hash_matches_its_file(self, record):
        cited = []

        def walk(node):
            if isinstance(node, dict):
                if {"path", "sha256"} <= set(node):
                    cited.append((node["path"], node["sha256"]))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(record)
        assert len(cited) >= 12
        for rel, digest in cited:
            got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
            assert got == digest, rel

    def test_the_snapshot_names_the_resolved_baseline(self, baseline):
        snap = json.loads((REPO / "logs/state/current.json").read_text())
        block = snap["forward_baseline"]
        assert block["baseline"] == baseline.qualified_id
        assert "prediction_rows=scored_v1" in block["execution"]
        assert block["forward_materialization"] is None
