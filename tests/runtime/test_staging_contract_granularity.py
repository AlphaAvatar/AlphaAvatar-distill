"""What the pod can see, at the granularity the pod actually receives it.

THE MODULE HAD NO TESTS. That is how this bug lived: `staged_files` collected a
local asset's contents

    if tree.is_dir():

and dropped anything else, matching a docstring that said "local assets are
whole trees". True until a session staged a resolved plan as ONE FILE — at
which point the file contributed nothing to the staged set, fell into the
hidden complement, and the simulator hid a file the pod demonstrably had. The
pod's driver had loaded that plan and reached its first operator; the sweep
still reported the test that reads it as an unexpected environment skip, and
the obvious reading of that was "my test's skip guard is wrong" rather than
"the simulator is wrong". Removing the fix again leaves every one of the 182
tests in this directory passing, which is the measurement that says these were
needed.

A simulation that hides what the pod receives is not conservative. It is wrong
in the direction that matters: a launch-bound readiness record would describe a
pod that does not exist, and the failure appears after setup has been paid for.

So these assert the granularity rule in both directions, for both staging
mechanisms, because the two are deliberately different:

    RelayInput  stages ONE NAMED FILE into a destination directory
                -> its siblings in that directory stay HIDDEN
    LocalAsset  stages whatever its source IS
                -> a directory contributes its whole tree, a file contributes
                   itself
"""
from __future__ import annotations

import pytest

from aadistill.infrastructure.session import LocalAsset, RelayInput, SetupManifest
from aadistill.runtime.staging_contract import (
    contract_digest, derive_contract, staged_files,
)


@pytest.fixture
def repo(tmp_path):
    """A repository-shaped tree with a directory asset, a file asset, and a
    destination holding both a staged and an unstaged sibling."""
    (tmp_path / "artifacts/stages/stage-1/corpus_v1").mkdir(parents=True)
    (tmp_path / "artifacts/stages/stage-1/corpus_v1/items.jsonl").write_text("{}\n")
    (tmp_path / "artifacts/stages/stage-1/corpus_v1/nested").mkdir()
    (tmp_path / "artifacts/stages/stage-1/corpus_v1/nested/more.json").write_text("{}")
    (tmp_path / "artifacts/stages/stage-1/plan.json").write_text('{"leaves": []}')
    (tmp_path / "artifacts/stages/stage-2/v0/ckpt").mkdir(parents=True)
    (tmp_path / "artifacts/stages/stage-2/v0/ckpt/model.safetensors").write_bytes(b"w")
    (tmp_path / "artifacts/stages/stage-2/v0/ckpt/config.json").write_text("{}")
    return tmp_path


def _contract(**over):
    setup = SetupManifest(**over)
    return derive_contract(setup, session_id="test-session")


class TestALocalAssetStagesWhateverItsSourceIs:

    def test_a_directory_asset_contributes_its_whole_tree(self, repo):
        contract = _contract(local_assets=(
            LocalAsset("artifacts/stages/stage-1/corpus_v1", "corpus_v1",
                       "artifacts/stages/stage-1"),))
        assert staged_files(contract, repo) == {
            "artifacts/stages/stage-1/corpus_v1/items.jsonl",
            "artifacts/stages/stage-1/corpus_v1/nested/more.json",
        }

    def test_a_FILE_asset_contributes_itself(self, repo):
        """The regression. This returned the empty set."""
        contract = _contract(local_assets=(
            LocalAsset("artifacts/stages/stage-1/plan.json", "plan.json",
                       "artifacts/stages/stage-1"),))
        assert staged_files(contract, repo) == {"artifacts/stages/stage-1/plan.json"}

    def test_a_file_asset_does_not_drag_in_its_directory(self, repo):
        """`artifacts/stages/stage-1/` also holds `corpus_v1/`, which this session does
        not stage. Staging a file must not make its parent present."""
        contract = _contract(local_assets=(
            LocalAsset("artifacts/stages/stage-1/plan.json", "plan.json",
                       "artifacts/stages/stage-1"),))
        staged = staged_files(contract, repo)
        assert not any(p.startswith("artifacts/stages/stage-1/corpus_v1")
                       for p in staged)

    def test_both_kinds_together(self, repo):
        contract = _contract(local_assets=(
            LocalAsset("artifacts/stages/stage-1/corpus_v1", "corpus_v1",
                       "artifacts/stages/stage-1"),
            LocalAsset("artifacts/stages/stage-1/plan.json", "plan.json",
                       "artifacts/stages/stage-1"),))
        assert staged_files(contract, repo) == {
            "artifacts/stages/stage-1/corpus_v1/items.jsonl",
            "artifacts/stages/stage-1/corpus_v1/nested/more.json",
            "artifacts/stages/stage-1/plan.json",
        }

    def test_an_asset_whose_source_is_absent_stages_nothing(self, repo):
        """Not an error here. A session may legitimately be inspected on a host
        that does not hold its assets, and the launcher's own precheck is what
        refuses a missing one before anything is priced."""
        contract = _contract(local_assets=(
            LocalAsset("artifacts/stages/stage-1/not_here.json", "not_here.json",
                       "artifacts/stages/stage-1"),))
        assert staged_files(contract, repo) == set()


class TestARelayInputStagesOneFileAndNotItsDirectory:
    """Deliberately NOT symmetric with a local asset, and that asymmetry is the
    original reason this module exists: modelling a relay destination as wholly
    present hid the class of error that cost four paid aborts."""

    def test_only_the_named_file_is_staged(self, repo):
        contract = _contract(relay_inputs=(
            RelayInput("transfer/model.safetensors", "artifacts/stages/stage-2/v0/ckpt"),))
        staged = staged_files(contract, repo)
        assert "artifacts/stages/stage-2/v0/ckpt/config.json" not in staged, (
            "a relay destination is being modelled as wholly present; its "
            "unstaged siblings must stay hidden")

    def test_the_sibling_in_a_relay_destination_stays_hidden(self, repo):
        contract = _contract(relay_inputs=(
            RelayInput("transfer/model.safetensors", "artifacts/stages/stage-2/v0/ckpt"),))
        staged = staged_files(contract, repo)
        assert len([p for p in staged
                    if p.startswith("artifacts/stages/stage-2/v0/ckpt/")]) <= 1


class TestTheDigestCoversTheDeclarationAndNotTheFilesystem:
    """`staged_files` is DERIVED; the digest is over the declaration. So adding
    a file inside a staged tree must not move a readiness record's digest —
    otherwise every sweep would be invalidated by an unrelated local file."""

    def test_adding_a_file_under_a_staged_tree_does_not_move_the_digest(
            self, repo):
        assets = (LocalAsset("artifacts/stages/stage-1/corpus_v1", "corpus_v1",
                             "artifacts/stages/stage-1"),)
        before = contract_digest(_contract(local_assets=assets))
        (repo / "artifacts/stages/stage-1/corpus_v1/extra.jsonl").write_text("{}\n")
        after = contract_digest(_contract(local_assets=assets))
        assert before == after

    def test_declaring_a_different_asset_does_move_it(self, repo):
        a = contract_digest(_contract(local_assets=(
            LocalAsset("artifacts/stages/stage-1/corpus_v1", "corpus_v1",
                       "artifacts/stages/stage-1"),)))
        b = contract_digest(_contract(local_assets=(
            LocalAsset("artifacts/stages/stage-1/plan.json", "plan.json",
                       "artifacts/stages/stage-1"),)))
        assert a != b

    def test_the_declared_assets_appear_in_the_contract_body(self):
        contract = _contract(local_assets=(
            LocalAsset("artifacts/stages/stage-1/plan.json", "plan.json",
                       "artifacts/stages/stage-1"),))
        assert contract["local_assets"] == [{
            "repo_path": "artifacts/stages/stage-1/plan.json",
            "dest_name": "plan.json",
            "install_to": "artifacts/stages/stage-1",
            "staged_tree": "artifacts/stages/stage-1/plan.json",
        }]


class TestTheContractSaysWhatItNowDoes:
    """The docstring that said "local assets are whole trees" is what the
    implementation matched, so a reader checking one against the other would
    have found them in agreement and both wrong."""

    def test_the_granularity_note_covers_the_file_case(self):
        contract = _contract()
        note = contract["granularity"]
        assert "whatever its source IS" in note
        assert "one file when it names a file" in note

    def test_the_module_docstring_no_longer_claims_trees_only(self):
        from aadistill.runtime import staging_contract

        assert "Local assets are whole trees" not in staging_contract.__doc__
        assert "stages whatever its source IS" in staging_contract.__doc__
