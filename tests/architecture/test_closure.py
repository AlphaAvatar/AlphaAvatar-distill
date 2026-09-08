"""The derived executable closure.

Built against a synthetic repository under `tmp_path` rather than the real tree,
so each property is pinned by construction instead of by whatever the repository
happens to contain today. A handful of real-tree assertions follow, for the
facts that are only meaningful about the real tree.

The relative-import cases exist because the first version of this walk read only
`node.module` and silently skipped every relative import. It reported a complete
62-file closure that was missing four modules — among them the provider and SSH
transport that create and terminate paid pods. A silent under-count is the exact
failure this module is supposed to make impossible, so it gets direct coverage.
"""
from __future__ import annotations

import json

import pytest

from aadistill.governance.closure import (
    ClosureError, _absolutize, compare, derive, walk)


def write(root, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def repo(tmp_path):
    """A synthetic repo: entry -> mid -> leaf, plus an unreachable module."""
    write(tmp_path, "src/pkg/__init__.py", "")
    write(tmp_path, "src/pkg/leaf.py", "VALUE = 1\n")
    write(tmp_path, "src/pkg/mid.py", "from pkg.leaf import VALUE\n")
    write(tmp_path, "scripts/entry.py", "from pkg.mid import VALUE\n")
    write(tmp_path, "src/pkg/unreachable.py", "raise AssertionError('never')\n")
    write(tmp_path, "configs/thing.json", '{"a": 1}\n')
    return tmp_path


class TestAbsolutize:
    """`from ..x import y` -> an absolute module name."""

    @pytest.mark.parametrize("rel,level,module,expected", [
        ("src/a/b/mod.py", 1, "sib", "a.b.sib"),
        ("src/a/b/mod.py", 2, "other", "a.other"),
        ("src/a/b/c/mod.py", 3, "top", "a.top"),
        # An __init__.py's own package is its directory, so level 1 is itself.
        ("src/a/b/__init__.py", 1, "sib", "a.b.sib"),
        ("src/a/b/__init__.py", 2, "other", "a.other"),
        # `from . import x` carries no module part.
        ("src/a/b/mod.py", 1, "", "a.b"),
        # scripts/ is a root too, and must not survive into the name.
        ("scripts/e/f/mod.py", 2, "g", "e.g"),
    ])
    def test_resolves(self, rel, level, module, expected):
        assert _absolutize(rel, level, module, ("src", "scripts")) == expected

    def test_past_top_of_package_raises(self):
        with pytest.raises(ClosureError, match="past the top"):
            _absolutize("src/a/mod.py", 3, "x", ("src", "scripts"))


class TestWalk:
    def test_follows_transitively(self, repo):
        files, unresolved = walk(repo, ("scripts/entry.py",))
        assert files == ["scripts/entry.py", "src/pkg/leaf.py", "src/pkg/mid.py"]
        assert unresolved == []

    def test_excludes_unreachable(self, repo):
        files, _ = walk(repo, ("scripts/entry.py",))
        assert "src/pkg/unreachable.py" not in files

    def test_follows_relative_imports(self, repo):
        """The defect that shipped: a relative edge must not be skipped."""
        write(repo, "src/pkg/mid.py", "from .leaf import VALUE\n")
        files, _ = walk(repo, ("scripts/entry.py",))
        assert "src/pkg/leaf.py" in files

    def test_follows_parent_relative_imports(self, repo):
        write(repo, "src/pkg/deep/__init__.py", "")
        write(repo, "src/pkg/deep/inner.py", "from ..leaf import VALUE\n")
        write(repo, "src/pkg/mid.py", "from pkg.deep.inner import VALUE\n")
        files, _ = walk(repo, ("scripts/entry.py",))
        assert "src/pkg/leaf.py" in files
        assert "src/pkg/deep/inner.py" in files

    def test_follows_bare_relative_package_import(self, repo):
        write(repo, "src/pkg/mid.py", "from . import leaf\n")
        files, _ = walk(repo, ("scripts/entry.py",))
        assert "src/pkg/leaf.py" in files

    def test_newly_reachable_module_is_included(self, repo):
        """The point of deriving: nobody has to remember to add it."""
        before, _ = walk(repo, ("scripts/entry.py",))
        assert "src/pkg/unreachable.py" not in before
        write(repo, "src/pkg/mid.py",
              "from pkg.leaf import VALUE\nimport pkg.unreachable\n")
        after, _ = walk(repo, ("scripts/entry.py",))
        assert "src/pkg/unreachable.py" in after

    def test_from_import_of_a_name_is_not_unresolved(self, repo):
        """`from x import SomeClass` must not be reported as a missing module."""
        write(repo, "src/pkg/mid.py", "from pkg.leaf import VALUE\n")
        _, unresolved = walk(repo, ("scripts/entry.py",))
        assert unresolved == []

    def test_third_party_import_is_not_unresolved(self, repo):
        write(repo, "src/pkg/mid.py", "import torch\nfrom pkg.leaf import VALUE\n")
        _, unresolved = walk(repo, ("scripts/entry.py",))
        assert unresolved == []

    def test_missing_internal_module_is_unresolved(self, repo):
        write(repo, "src/pkg/mid.py", "import aadistill.gone\n")
        _, unresolved = walk(repo, ("scripts/entry.py",))
        assert unresolved == ["aadistill.gone"]

    def test_missing_entry_point_raises(self, repo):
        with pytest.raises(ClosureError, match="entry point"):
            walk(repo, ("scripts/nope.py",))


class TestDerive:
    def test_digest_covers_declared_inputs(self, repo):
        a = derive(repo, "x", ("scripts/entry.py",), ("configs/thing.json",))
        (repo / "configs/thing.json").write_text('{"a": 2}\n')
        b = derive(repo, "x", ("scripts/entry.py",), ("configs/thing.json",))
        assert a["digest"] != b["digest"], "a non-python input must bind"

    def test_digest_changes_with_source_bytes(self, repo):
        a = derive(repo, "x", ("scripts/entry.py",))
        (repo / "src/pkg/leaf.py").write_text("VALUE = 2\n")
        b = derive(repo, "x", ("scripts/entry.py",))
        assert a["digest"] != b["digest"]

    def test_deterministic(self, repo):
        a = derive(repo, "x", ("scripts/entry.py",), ("configs/thing.json",))
        b = derive(repo, "x", ("scripts/entry.py",), ("configs/thing.json",))
        assert a["digest"] == b["digest"]
        assert [r["path"] for r in a["files"]] == [r["path"] for r in b["files"]]

    def test_refuses_missing_declared_input(self, repo):
        with pytest.raises(ClosureError, match="non-python input"):
            derive(repo, "x", ("scripts/entry.py",), ("configs/gone.json",))

    def test_refuses_unresolved_internal_import(self, repo):
        write(repo, "src/pkg/mid.py", "import aadistill.gone\n")
        with pytest.raises(ClosureError, match="unresolved internal import"):
            derive(repo, "x", ("scripts/entry.py",))

    def test_authorizes_nothing(self, repo):
        assert derive(repo, "x", ("scripts/entry.py",))["authorizes"] == "nothing"


class TestCompare:
    def test_reports_added_removed_and_changed(self, repo):
        before = derive(repo, "x", ("scripts/entry.py",))
        write(repo, "src/pkg/mid.py",
              "from pkg.leaf import VALUE\nimport pkg.unreachable\n")
        after = derive(repo, "x", ("scripts/entry.py",))
        drift = compare(after, before)
        assert drift["added_files"] == ["src/pkg/unreachable.py"]
        assert drift["removed_files"] == []
        assert drift["changed_files"] == ["src/pkg/mid.py"]
        assert drift["digest_matches"] is False

    def test_identical_matches(self, repo):
        doc = derive(repo, "x", ("scripts/entry.py",))
        drift = compare(doc, doc)
        assert drift == {"digest_matches": True, "added_files": [],
                         "removed_files": [], "changed_files": []}


class TestRealTree:
    """Facts that are only meaningful about the actual repository."""

    def test_current_digest_computes(self, repo_root):
        from experiments.phase_c1.authorization import c1_current_executable
        doc = c1_current_executable(repo_root)
        assert doc["n_files"] > 40
        assert len(doc["digest"]) == 64

    def test_historical_declaration_fails_closed(self, repo_root):
        """An old authorization must not be revalidatable on the migrated tree."""
        from aadistill.governance.authorization import AuthorizationError
        from experiments.phase_c1.authorization import c1_historical_harness_digest
        with pytest.raises(AuthorizationError, match="is missing"):
            c1_historical_harness_digest(repo_root)

    def test_closure_covers_the_paid_resource_path(self, repo_root):
        """Whatever can create or terminate a billed pod is part of identity.

        These three were the modules the silent walk missed.
        """
        from experiments.phase_c1.authorization import c1_current_executable
        paths = {r["path"] for r in c1_current_executable(repo_root)["files"]}
        for rel in ("src/aadistill/infrastructure/provider.py",
                    "src/aadistill/infrastructure/remote.py",
                    "src/aadistill/infrastructure/log_relay.py"):
            assert rel in paths, f"{rel} can spend money and must bind"

    def test_snapshot_matches_live(self, repo_root):
        """The committed snapshot is kept current, so drift is reviewable."""
        from experiments.phase_c1.authorization import (
            CURRENT_CLOSURE_SNAPSHOT, c1_current_executable)
        recorded = json.loads((repo_root / CURRENT_CLOSURE_SNAPSHOT).read_text())
        drift = compare(c1_current_executable(repo_root), recorded)
        assert drift["added_files"] == [] and drift["removed_files"] == [], (
            "re-run scripts/architecture/derive_closure.py --write")
