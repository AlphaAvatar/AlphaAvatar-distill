"""The six built batteries, and the verifier that proves them.

This experiment family's own tests: they read the built roles and the committed
manifest. The items are 22 MB of gitignored artifacts, so the whole class skips
where they are absent and says so — a checkout without them cannot answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.phase_d_series import battery_family as family  # noqa: E402
from experiments.phase_d_series import build_batteries as build  # noqa: E402
from experiments.phase_d_series import verify_batteries as verify  # noqa: E402

ROOT = REPO / build.OUT
MANIFEST = REPO / build.MANIFEST

requires_the_built_family = pytest.mark.skipif(
    not (ROOT / "family.json").is_file(),
    reason=("the six roles are 22 MB of gitignored artifacts; build them with "
            "`build_batteries.py --write` first"))


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.is_file():
        pytest.skip("the family manifest has not been written")
    return json.loads(MANIFEST.read_text())


class TestTheManifestIsACommittedRecord:
    """The items are artifacts; the manifest is the record that describes them."""

    def test_it_names_the_rule_version_and_both_ids(self, manifest):
        assert manifest["allocation_rule_version"] == 3
        assert manifest["allocation_rule_id"] == family.allocation_rule_id()
        assert len(manifest["family_content_id"]) == 64

    def test_it_carries_every_role_in_the_frozen_order(self, manifest):
        assert manifest["role_order"] == [r[0] for r in family.ROLES]
        assert sorted(manifest["roles"]) == sorted(r[0] for r in family.ROLES)

    def test_every_role_has_the_frozen_counts(self, manifest):
        spec = family.strata()
        want_total = sum(take for _d, take, _s in spec.values())
        want_scorable = sum(take for _d, take, s in spec.values() if s)
        for role, got in manifest["roles"].items():
            assert got["n_prompts"] == want_total == 950, role
            assert got["n_scorable"] == want_scorable == 850, role
            assert got["per_stratum"] == {
                g: take for g, (_d, take, _s) in spec.items()}, role

    def test_it_carries_a_digest_per_source_file(self, manifest):
        sources = manifest["sources"]
        assert sorted(sources) == sorted(family.strata())
        for group, spec in sources.items():
            files = spec.get("files", {})
            assert files, group
            for rel, got in files.items():
                assert len(got["sha256"]) == 64, (group, rel)

    def test_the_family_content_id_binds_what_was_drawn(self, manifest):
        """It must move when any role's realized items move, and it must NOT be
        the rule id — the rule says how, this says what."""
        assert manifest["family_content_id"] != manifest["allocation_rule_id"]
        assert build.family_content_id(manifest) == manifest["family_content_id"]

        moved = json.loads(json.dumps(manifest))
        role = manifest["role_order"][0]
        moved["roles"][role]["item_ids_sha256"] = "0" * 64
        assert build.family_content_id(moved) != manifest["family_content_id"]

    def test_it_says_the_family_predates_any_outcome(self, manifest):
        #: the CLAIM, not a phrase: all six drawn in one pass before any search
        #: ran, so no later choice about a confirmation rung's prompts exists.
        claim = manifest["_built_before_any_outcome"]
        assert "before any D1 search has run" in claim
        assert "No later choice" in claim


@requires_the_built_family
class TestTheVerifierPasses:
    """Every requirement, checked by the independent verifier."""

    @pytest.fixture(scope="class")
    def results(self):
        return verify.verify(ROOT)

    @pytest.mark.parametrize("check", [
        "counts_and_denominators", "renderer_scorer_parity",
        "pairwise_disjointness", "isolation_from_history",
        "recovery_training", "manifest",
    ])
    def test_check_passes(self, results, check):
        assert results[check] == [], results[check]


@requires_the_built_family
class TestTheVerifierWouldCatchAFailure:
    """A verifier that cannot fail proves nothing.

    Each case corrupts a loaded copy and asserts the relevant check reports it.
    The build on disk is never touched.
    """

    @pytest.fixture(scope="class")
    def loaded(self):
        roles, doc = verify.load(ROOT)
        return roles, doc

    def _copy(self, roles):
        return {r: {g: [dict(i) for i in items] for g, items in per.items()}
                for r, per in roles.items()}

    def test_a_short_stratum_is_caught(self, loaded):
        roles, _ = loaded
        bad = self._copy(roles)
        bad["d1_screening"]["gsm8k"].pop()
        assert verify.check_counts(bad), "a missing item must fail the counts"

    def test_a_shared_item_between_two_roles_is_caught(self, loaded):
        roles, _ = loaded
        bad = self._copy(roles)
        #: give d2_screening one of d1_screening's items
        bad["d2_screening"]["gsm8k"][0] = dict(bad["d1_screening"]["gsm8k"][0])
        problems = verify.check_pairwise_disjointness(bad)
        assert problems and "d1_screening" in problems[0]

    def test_a_historically_consumed_row_is_caught(self, loaded):
        """The failure the verifier actually found: a row whose historical render
        id is already reserved, which neither the D-series id nor the
        rendered-prompt hash could see."""
        roles, _ = loaded
        bad = self._copy(roles)
        consumed = json.loads(
            (REPO / "artifacts/stage3/c1_confirmation_v1/gsm8k.jsonl")
            .read_text().splitlines()[0])
        bad["d1_screening"]["gsm8k"][0]["historical_render_id"] = consumed["id"]
        problems = verify.check_isolation_from_history(bad)
        assert any("historical" in p for p in problems), problems

    def test_a_drifted_prompt_is_caught(self, loaded):
        roles, _ = loaded
        bad = self._copy(roles)
        bad["d1_screening"]["gsm8k"][0]["prompt_text"] += " (edited)"
        assert verify.check_renderer_parity(bad), (
            "a prompt that is not the historical rendering must fail parity")

    def test_a_manifest_that_describes_other_items_is_caught(self, loaded):
        roles, doc = loaded
        bad_doc = json.loads(json.dumps(doc))
        bad_doc["roles"][doc["role_order"][0]]["item_ids_sha256"] = "0" * 64
        assert verify.check_manifest(roles, bad_doc)


@requires_the_built_family
class TestTheIdentityCoordinatesOnDisk:
    def test_no_item_carries_a_historical_split_naming_id(self):
        """`mbpp-test-*` must not name a row from another split, and no D-series
        id may be a historical one."""
        roles, _ = verify.load(ROOT)
        for role, per in roles.items():
            for group, items in per.items():
                for item in items:
                    assert not str(item["id"]).startswith("mbpp-test-"), item["id"]
                    assert not str(item["id"]).startswith("gsm8k-test-"), item["id"]
                    if item.get("historical_render_id"):
                        assert item["id"] != item["historical_render_id"]

    def test_the_id_names_the_config_and_split_it_came_from(self):
        roles, _ = verify.load(ROOT)
        for role, per in roles.items():
            for group, items in per.items():
                for item in items:
                    assert str(item["_config"]) in str(item["id"]), item["id"]
                    assert str(item["_split"]) in str(item["id"]), item["id"]

    def test_the_three_payload_strata_all_carry_a_content_id(self):
        from experiments.phase_d_series.identity import PROBLEM_FIELD

        roles, _ = verify.load(ROOT)
        for role, per in roles.items():
            for group in PROBLEM_FIELD:
                for item in per[group]:
                    assert item.get("problem_content_id"), (role, group, item["id"])
