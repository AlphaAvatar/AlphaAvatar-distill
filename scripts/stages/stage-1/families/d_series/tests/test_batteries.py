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

REPO = Path(__file__).resolve().parents[6]
for extra in ("src", "scripts", "scripts/data"):
    path = str(REPO / extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from stages.d_series import battery_family as family  # noqa: E402
from stages.d_series import build_batteries as build  # noqa: E402
from stages.d_series import verify_batteries as verify  # noqa: E402

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

    def test_the_family_content_id_is_not_the_rule_id(self, manifest):
        """The rule says HOW the family is drawn; this says WHAT was drawn.

        This test used to assert that moving a role's `item_ids_sha256` moved the
        family id. It did then, and that was the defect: the id digest binds only
        which SOURCE ROWS the ids refer to. What moves the family id now is a
        changed output-file digest -- see `TestTheArtifactDigestIsWhatBinds`.
        """
        assert manifest["family_content_id"] != manifest["allocation_rule_id"]
        assert build.family_content_id(manifest) == manifest["family_content_id"]

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
            (REPO / "artifacts/stages/stage-1/phase_c1/batteries/c1_confirmation_v1/gsm8k.jsonl")
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
        from stages.d_series.identity import PROBLEM_FIELD

        roles, _ = verify.load(ROOT)
        for role, per in roles.items():
            for group in PROBLEM_FIELD:
                for item in per[group]:
                    assert item.get("problem_content_id"), (role, group, item["id"])


@requires_the_built_family
class TestTheArtifactDigestIsWhatBinds:
    """`family_content_id` must identify the actual bytes, not only the ids.

    The first version hashed `item id : problem_content_id` per role, which says
    which SOURCE ROWS the ids refer to and nothing about the behavioural item.
    `prompt_text`, `gold`, `messages`, aliases, a tool schema and every scorer
    field could all change without moving it -- and this round had already found
    `battery_v2`'s RAG prompts drifting from the current renderer with every id
    intact.
    """

    @pytest.fixture
    def mutated(self, tmp_path):
        """A full copy of the family, safe to corrupt."""
        import shutil

        root = tmp_path / "family"
        shutil.copytree(ROOT, root)
        return root

    def _rewrite(self, path, mutate):
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        mutate(rows)
        path.write_bytes(build.serialize(rows))
        return rows

    def test_the_manifest_names_one_file_per_role_and_stratum(self, manifest):
        expected = len(family.ROLES) * len(family.strata())
        assert len(manifest["output_files"]) == expected == 42
        for rel, got in manifest["output_files"].items():
            assert len(got["sha256"]) == 64, rel
            assert got["size_bytes"] > 0 and got["n_items"] > 0, rel

    def test_the_family_id_follows_from_the_output_digests(self, manifest):
        """And a changed digest moves it."""
        assert build.family_content_id(manifest) == manifest["family_content_id"]
        moved = json.loads(json.dumps(manifest))
        rel = sorted(moved["output_files"])[0]
        moved["output_files"][rel]["sha256"] = "0" * 64
        assert build.family_content_id(moved) != manifest["family_content_id"]

    def test_the_id_digest_alone_no_longer_binds_the_family(self, manifest):
        """The readable secondary identity must not be what the family id uses.

        If `family_content_id` still followed from `item_ids_sha256`, changing a
        gold would leave it unchanged -- which is the whole defect.
        """
        moved = json.loads(json.dumps(manifest))
        for role in moved["roles"]:
            moved["roles"][role]["item_ids_sha256"] = "0" * 64
        assert build.family_content_id(moved) == manifest["family_content_id"], (
            "family_content_id still depends on the id digest; it must bind the "
            "output bytes")

    def test_changing_a_gold_is_caught_by_the_artifact_digest_alone(self, mutated):
        """Id and problem_content_id untouched; only the behavioural content."""
        path = mutated / "d1_screening" / "gsm8k.jsonl"
        before = json.loads(path.read_text().splitlines()[0])
        rows = self._rewrite(path, lambda rs: rs[0].__setitem__("gold", "999999"))
        assert rows[0]["id"] == before["id"]
        assert rows[0]["problem_content_id"] == before["problem_content_id"]

        roles, doc = verify.load(mutated)
        assert verify.check_output_digests(mutated, doc), (
            "a changed gold must fail the artifact digest")
        #: and `check_manifest` must fail from the artifact digest ALONE
        assert verify.check_manifest(roles, doc, mutated)

    def test_changing_a_prompt_is_caught_by_the_artifact_digest(self, mutated):
        path = mutated / "d1_screening" / "rag.jsonl"
        self._rewrite(path, lambda rs: rs[0].__setitem__(
            "prompt_text", rs[0]["prompt_text"] + " (drifted)"))
        _roles, doc = verify.load(mutated)
        assert verify.check_output_digests(mutated, doc)

    def test_an_extra_file_on_disk_is_caught(self, mutated):
        (mutated / "d1_screening" / "extra.jsonl").write_text("{}\n")
        _roles, doc = verify.load(mutated)
        problems = verify.check_output_digests(mutated, doc)
        assert any("not named by the manifest" in p for p in problems), problems


@requires_the_built_family
class TestEveryItemIsCheckedAgainstTheWholeContract:
    """The native-identity coordinate, independently verified for every stratum.

    `rag`, `multihop`, `knowledge` and `tool` declare no problem payload, so the
    native/source-id coordinate is the only one with teeth there — and it was the
    one resting on the builder's word.
    """

    def test_the_contract_covers_the_complete_baseline(self):
        """Recovery training, calibration and state evaluation contribute source
        ids, and the contract must include them."""
        reserved = verify.reserved_contract("rag")
        assert len(reserved["ids"]) > 10000, (
            "the reserved id set looks too small to include the training corpus")
        #: a known recovery-training source id must be in it
        first = json.loads(
            (REPO / "artifacts/stages/stage-3/corpus_v2/sessions.jsonl")
            .read_text(errors="ignore").split("\n", 1)[0])
        assert str(first["source_id"]) in reserved["ids"]

    def test_a_reserved_source_key_is_caught_on_a_no_payload_stratum(self, tmp_path):
        """RAG: no `problem_content_id`, so only the id coordinate can catch this."""
        import shutil

        root = tmp_path / "family"
        shutil.copytree(ROOT, root)
        first = json.loads(
            (REPO / "artifacts/stages/stage-3/corpus_v2/sessions.jsonl")
            .read_text(errors="ignore").split("\n", 1)[0])
        reserved_id = str(first["source_id"])

        path = root / "d1_screening" / "rag.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        assert rows[0].get("problem_content_id") is None, (
            "this test needs a stratum with no problem payload")
        rows[0]["source_key"] = reserved_id
        path.write_bytes(build.serialize(rows))

        roles, _doc = verify.load(root)
        problems = verify.check_every_item_against_the_contract(roles)
        assert any("source_key" in p and "reserved id" in p for p in problems), (
            problems[:3])

    def test_a_reserved_historical_render_id_is_caught(self, tmp_path):
        import shutil

        root = tmp_path / "family"
        shutil.copytree(ROOT, root)
        consumed = json.loads(
            (REPO / "artifacts/stages/stage-1/phase_c1/batteries/c1_confirmation_v1/multihop.jsonl")
            .read_text().splitlines()[0])
        path = root / "d2_screening" / "multihop.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        rows[0]["historical_render_id"] = consumed["id"]
        path.write_bytes(build.serialize(rows))

        roles, _doc = verify.load(root)
        problems = verify.check_every_item_against_the_contract(roles)
        assert any("historical_render_id" in p for p in problems), problems[:3]
