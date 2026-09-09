"""Changing what a paid session does must move the identity that authorizes it.

A closure is only worth computing if it actually covers the things that spend
money and decide behaviour. The previous C1 set was hand-written and did not:
it omitted `provider.py`, `remote.py` and `log_relay.py` -- the modules that
create a pod, reach it and relay its logs -- so all three could have changed
under an authorization that claimed to pin the executable.

These tests mutate a real file in a scratch copy of the repository and require
the derived digest to move. A closure that does not move here is not protecting
whatever was edited, whatever its docstring says.

The copy is a `git archive` of HEAD plus the working tree's version of every
declared file, so the mutation is the only difference. Nothing here touches the
repository.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


from aadistill.governance.closure import derive  # noqa: E402
from experiments.phase_c1.authorization import (  # noqa: E402
    C1_DECLARED_INPUTS, C1_ENTRY_POINTS, C1_SOURCE_ROOTS)


@pytest.fixture(scope="module")
def scratch_repo(tmp_path_factory) -> Path:
    """A copy of the working tree's declared closure, mutable and disposable."""
    dest = tmp_path_factory.mktemp("closure_repo")
    live = derive(REPO, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
           roots=C1_SOURCE_ROOTS)
    for row in live["files"]:
        target = dest / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / row["path"], target)
    return dest


def digest_of(repo: Path) -> str:
    return derive(repo, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS)["digest"]


def mutate(repo: Path, rel: str, find: str, replace: str) -> None:
    path = repo / rel
    text = path.read_text()
    assert find in text, f"{rel} no longer contains {find!r}; update this test"
    path.write_text(text.replace(find, replace, 1))


class TestTheClosureCoversWhatSpendsMoney:
    """§4: provider creation, remote execution, log relay, setup, config."""

    @pytest.mark.parametrize("rel,find,replace,what", [
        ("src/aadistill/infrastructure/provider.py",
         "def read_api_key", "def read_api_key_MUTATED",
         "provider creation"),
        ("src/aadistill/infrastructure/remote.py",
         "def probe", "def probe_MUTATED",
         "remote execution"),
        ("src/aadistill/infrastructure/log_relay.py",
         "class LogRelay", "class LogRelay_MUTATED",
         "log relay"),
        ("src/aadistill/infrastructure/session_runner.py",
         "def fetch_result_ok", "def fetch_result_ok_MUTATED",
         "the session runner"),
        ("src/aadistill/infrastructure/watchdog.py",
         "class WatchdogPolicy", "class WatchdogPolicy_MUTATED",
         "the watchdog"),
        ("scripts/pod/watchdog.py",
         "import argparse", "import argparse  # MUTATED",
         "the watchdog script the runner shells out to"),
        ("scripts/pod/autoinit_preflight_setup.sh",
         "#!/", "#!/ # MUTATED\n",
         "the setup shell"),
        ("configs/experiments/phase_c1/authorization.json",
         "{", '{"_mutated": true,',
         "the execution config"),
        ("configs/autoinit/c1_artifacts.json",
         "{", '{"_mutated": true,',
         "the artifact contract"),
    ], ids=lambda v: v if isinstance(v, str) and " " in v else None)
    def test_editing_it_moves_the_future_identity(self, scratch_repo, tmp_path,
                                                  rel, find, replace, what):
        work = tmp_path / "work"
        shutil.copytree(scratch_repo, work)
        before = digest_of(work)
        mutate(work, rel, find, replace)
        after = digest_of(work)
        assert before != after, (
            f"editing {what} ({rel}) left the C1 executable identity unchanged; "
            "an authorization binding that digest would not notice the change")


class TestTheClosureIsDerivedNotListed:
    def test_a_newly_imported_module_joins_the_closure(self, scratch_repo, tmp_path):
        """Nobody has to remember to add it -- that is the point."""
        work = tmp_path / "work"
        shutil.copytree(scratch_repo, work)
        before = derive(work, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
                      roots=C1_SOURCE_ROOTS)
        new_module = work / "src/aadistill/infrastructure/newly_added.py"
        new_module.write_text("VALUE = 1\n")
        mutate(work, "scripts/pod/autoinit_c1_driver.py",
               "import json",
               "import json\nfrom aadistill.infrastructure.newly_added import VALUE")
        after = derive(work, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
                      roots=C1_SOURCE_ROOTS)
        paths = {r["path"] for r in after["files"]}
        assert "src/aadistill/infrastructure/newly_added.py" in paths
        assert after["digest"] != before["digest"]

    def test_a_newly_shelled_out_script_joins_the_closure(self, scratch_repo,
                                                          tmp_path):
        """A subprocess target is part of what runs, though no import reaches it."""
        work = tmp_path / "work"
        shutil.copytree(scratch_repo, work)
        target = work / "scripts/pod/newly_shelled.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("print('hi')\n")
        before = derive(work, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
                      roots=C1_SOURCE_ROOTS)
        mutate(work, "scripts/pod/autoinit_c1_driver.py",
               "import json",
               'import json\n_NEW = REPO / "scripts/pod/newly_shelled.py"')
        after = derive(work, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
                      roots=C1_SOURCE_ROOTS)
        paths = {r["path"] for r in after["files"]}
        assert "scripts/pod/newly_shelled.py" in paths, (
            "a script the driver composes a path to is a subprocess target")
        assert after["digest"] != before["digest"]

    def test_prose_naming_a_script_does_not_pull_it_in(self, scratch_repo, tmp_path):
        """A docstring is documentation, not execution."""
        work = tmp_path / "work"
        shutil.copytree(scratch_repo, work)
        target = work / "scripts/pod/only_mentioned.py"
        target.write_text("print('hi')\n")
        before = derive(work, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
                      roots=C1_SOURCE_ROOTS)
        mutate(work, "scripts/pod/autoinit_c1_driver.py",
               "import json",
               'import json\n\n"""See scripts/pod/only_mentioned.py for context."""')
        after = derive(work, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
                      roots=C1_SOURCE_ROOTS)
        paths = {r["path"] for r in after["files"]}
        assert "scripts/pod/only_mentioned.py" not in paths, (
            "counting prose would put half the repository in the closure")

    def test_a_declaration_list_member_is_not_a_subprocess_target(self):
        """`pod_environment` names the tools ITS OWN digest covers, not C1's."""
        live = derive(REPO, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
           roots=C1_SOURCE_ROOTS)
        paths = {r["path"] for r in live["files"]}
        for declared_elsewhere in ("scripts/autoinit/publish_selected_leaves.py",
                                   "scripts/pod/simulate_pod_env.sh"):
            assert declared_elsewhere not in paths


class TestTheRealClosure:
    def test_it_contains_everything_that_creates_or_ends_a_billed_pod(self):
        live = derive(REPO, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
           roots=C1_SOURCE_ROOTS)
        paths = {r["path"] for r in live["files"]}
        for rel in ("src/aadistill/infrastructure/provider.py",
                    "src/aadistill/infrastructure/remote.py",
                    "src/aadistill/infrastructure/log_relay.py",
                    "src/aadistill/infrastructure/session_runner.py",
                    "src/aadistill/infrastructure/watchdog.py",
                    "scripts/pod/watchdog.py",
                    "scripts/pod/autoinit_preflight_setup.sh",
                    "configs/experiments/phase_c1/authorization.json"):
            assert rel in paths, f"{rel} decides what a paid session does"

    def test_the_preregistration_is_bound_by_its_own_hash_not_by_the_closure(self):
        """The one runtime input that CANNOT be in the digest.

        It records the harness digest, so including its bytes in that digest is
        a fixed point with no solution: writing the document changes the value
        it has to contain. Two other checks bind it instead, and both are
        exercised here rather than asserted.
        """
        import json

        from experiments.phase_c1.authorization import c1_harness_digest

        live = derive(REPO, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
                      roots=C1_SOURCE_ROOTS)
        rel = "logs/phase_c1_execution_preregistration.json"
        assert rel not in {r["path"] for r in live["files"]}, (
            "including it would make the digest unreachable")

        doc = json.loads((REPO / rel).read_text())
        # 1. it carries its own self-hash
        assert doc["preregistration_sha256"], "nothing would pin its bytes"
        # 2. and it must record the LIVE harness, which is what makes a stale
        #    preregistration refusable -- the payload gate compares these.
        assert doc["c1_harness"]["digest"] == c1_harness_digest(REPO)["digest"], (
            "re-emit it: scripts/autoinit/write_c1_execution_preregistration.py")

    def test_it_does_not_reach_another_phase_driver(self):
        """C1 has its own launcher and driver; reaching Phase A's would be wrong."""
        live = derive(REPO, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
           roots=C1_SOURCE_ROOTS)
        paths = {r["path"] for r in live["files"]}
        for rel in ("scripts/pod/autoinit_phase_a_driver.py",
                    "scripts/pod/autoinit_phase_a_launch.py",
                    "scripts/pod/autoinit_preflight_driver.py",
                    "scripts/autoinit/phase_a_search.py"):
            assert rel not in paths, f"{rel} is not on the C1 path"

    def test_the_committed_snapshot_lists_the_same_files(self):
        import json

        from experiments.phase_c1.authorization import CURRENT_CLOSURE_SNAPSHOT
        recorded = json.loads((REPO / CURRENT_CLOSURE_SNAPSHOT).read_text())
        live = derive(REPO, "phase_c1", C1_ENTRY_POINTS, C1_DECLARED_INPUTS,
           roots=C1_SOURCE_ROOTS)
        assert {r["path"] for r in recorded["files"]} == \
               {r["path"] for r in live["files"]}, (
            "re-run scripts/architecture/derive_closure.py --write")
