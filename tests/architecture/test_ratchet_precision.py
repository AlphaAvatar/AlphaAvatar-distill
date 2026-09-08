"""The ratchet's own failure modes, exercised against synthetic inventories.

The ratchet in `test_core_boundaries.py` reports on the real tree, so it can
only ever say what today's tree happens to contain. These tests feed it
inventories built for the purpose, and each one targets a way the previous
identity let a real violation through:

* a second offending literal inside an owner the baseline already lists, which
  `set(f"{file}::{owner}")` de-duplicated into an existing entry;
* changing a baselined digest to a different one without removing the
  dependency, which kept the same file and owner and so kept the same identity;
* widening the source and the allow list together, which two mutually-compared
  editable files cannot detect.

A rule that has never been seen to fail is not known to work, so each test
asserts the ratchet REFUSES, and the fixtures below are the minimum that makes
it refuse.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "core_boundaries_under_test", REPO / "tests/architecture/test_core_boundaries.py")
RATCHET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RATCHET)


def module(rel: str, *, literals=(), family=(), import_calls=(), imports=(),
           classification="generic_planning_execution") -> dict:
    return {
        "classification": classification,
        "literals": [{"owner": o, "value": v, "kinds": k} for o, v, k in literals],
        "family_attribute_access": [{"owner": o, "chain": c} for o, c in family],
        "import_time_calls": [{"call": c, "line": 1} for c in import_calls],
        "imports": [{"target": t} for t in imports],
    }


def inventory(mods: dict, cycles=()) -> dict:
    return {"modules": mods, "graph": {"package_cycles": [list(c) for c in cycles]}}


LOG_PATH = ("write_report", "logs/autoinit_run.json", ["path"])
OTHER_LOG = ("write_report", "logs/autoinit_other.json", ["path"])


class TestASecondViolationInAListedOwner:
    """The defect the content discriminator exists to fix."""

    def test_the_first_literal_is_the_baselined_one(self):
        inv = inventory({"src/aadistill/x.py": module("x", literals=[LOG_PATH])})
        sites = RATCHET._violations(inv)["path_literals"]
        assert len(sites) == 1
        assert sites[0].startswith("src/aadistill/x.py::write_report::")

    def test_a_second_literal_in_the_same_owner_is_a_distinct_site(self):
        inv = inventory({"src/aadistill/x.py": module(
            "x", literals=[LOG_PATH, OTHER_LOG])})
        sites = RATCHET._violations(inv)["path_literals"]
        assert len(sites) == 2, "two literals in one function are two violations"
        assert len(set(sites)) == 2, "and they must not share an identity"

    def test_the_ratchet_refuses_the_second_literal(self):
        """The regression proper: baseline the first, add the second, fail."""
        baselined = RATCHET._violations(
            inventory({"src/aadistill/x.py": module("x", literals=[LOG_PATH])})
        )["path_literals"]
        after = RATCHET._violations(
            inventory({"src/aadistill/x.py": module(
                "x", literals=[LOG_PATH, OTHER_LOG])})
        )["path_literals"]
        novel = sorted((Counter(after) - Counter(baselined)).elements())
        assert len(novel) == 1, (
            "a second hard-coded log path in an already-listed owner must be "
            f"NEW, got {novel}")

    def test_two_identical_literals_are_two_violations(self):
        """Multiplicity, not just distinctness: the same path twice is twice."""
        inv = inventory({"src/aadistill/x.py": module(
            "x", literals=[LOG_PATH, LOG_PATH])})
        after = Counter(RATCHET._violations(inv)["path_literals"])
        before = Counter(RATCHET._violations(
            inventory({"src/aadistill/x.py": module("x", literals=[LOG_PATH])})
        )["path_literals"])
        assert sorted((after - before).elements()), (
            "the same literal appearing twice must not collapse to one")


class TestChangingABaselinedDigest:
    """Same file, same owner, different value: previously the same identity."""

    OLD = ("<module>", "a" * 64, ["sha256"])
    NEW = ("<module>", "b" * 64, ["sha256"])

    def test_the_identity_follows_the_value(self):
        a = RATCHET._violations(
            inventory({"src/aadistill/y.py": module("y", literals=[self.OLD])})
        )["sha256_literals"]
        b = RATCHET._violations(
            inventory({"src/aadistill/y.py": module("y", literals=[self.NEW])})
        )["sha256_literals"]
        assert a != b, ("changing the digest must change the identity; keying on "
                        "file+owner alone could not see this")

    def test_the_ratchet_refuses_the_changed_digest(self):
        before = Counter(RATCHET._violations(
            inventory({"src/aadistill/y.py": module("y", literals=[self.OLD])})
        )["sha256_literals"])
        after = Counter(RATCHET._violations(
            inventory({"src/aadistill/y.py": module("y", literals=[self.NEW])})
        )["sha256_literals"])
        assert sorted((after - before).elements()), (
            "swapping a baselined digest for a different one, without removing "
            "the dependency, must register as a new violation")
        assert sorted((before - after).elements()), (
            "and the old site must register as fixed, so the baseline is forced "
            "down rather than left to absorb the next change")

    def test_an_unchanged_digest_is_not_novel(self):
        """The complement: identity must be stable, or every edit churns."""
        sites = RATCHET._violations(
            inventory({"src/aadistill/y.py": module("y", literals=[self.OLD])})
        )["sha256_literals"]
        again = RATCHET._violations(
            inventory({"src/aadistill/y.py": module("y", literals=[self.OLD])})
        )["sha256_literals"]
        assert sites == again


class TestWideningSourceAndAllowanceTogether:
    """Two editable files compared only to each other cannot catch this."""

    def test_the_pair_is_self_consistent_and_still_wrong(self):
        """Adding a violation AND its allow entry satisfies the other rules."""
        inv = inventory({"src/aadistill/z.py": module(
            "z", literals=[LOG_PATH, OTHER_LOG])})
        observed = RATCHET._violations(inv)["path_literals"]
        widened = {"allow": {"path_literals": list(observed)}}
        novel = sorted((Counter(observed)
                        - Counter(widened["allow"]["path_literals"])).elements())
        assert not novel, ("this is the hole: the tree and the allow list agree, "
                           "so the new-violation rule reports nothing")

    def test_the_accepted_revision_rule_catches_it(self, tmp_path):
        """Growth is measured against a revision, which a working tree cannot edit."""
        accepted = {"allow": {"path_literals": [
            "src/aadistill/z.py::write_report::aaaaaaaaaa"]}}
        widened = {"allow": {"path_literals": [
            "src/aadistill/z.py::write_report::aaaaaaaaaa",
            "src/aadistill/z.py::write_report::bbbbbbbbbb"]}}
        grew = sorted((Counter(widened["allow"]["path_literals"])
                       - Counter(accepted["allow"]["path_literals"])).elements())
        assert grew == ["src/aadistill/z.py::write_report::bbbbbbbbbb"], (
            "widening must be visible against the accepted revision")

    def test_the_real_baseline_names_a_revision_that_resolves(self):
        """The rule is inert unless the field actually points somewhere."""
        baseline = json.loads(
            (REPO / "configs/architecture/core_boundary_baseline.json").read_text())
        rev = baseline.get("accepted_revision")
        assert rev, "the baseline must name the revision its allowance was accepted at"
        out = subprocess.run(
            ["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=REPO,
            capture_output=True)
        assert out.returncode == 0, f"accepted_revision {rev} is not a commit"


class TestTheDetectorsDoNotFireOnOrdinaryCode:
    """§4: a zero count is not sufficient, and neither is a large one.

    Keyword-driven detectors turn ordinary code into architecture debt, which
    makes the baseline noise and trains reviewers to skim it.
    """

    @pytest.mark.parametrize("value", [
        3.141592653589793,          # a mathematical constant
        "application/json",         # a MIME type
        "content-type",             # a generic protocol field
        "https://api.example.com",  # not a repo id
        1e-8,                       # an epsilon
    ])
    def test_ordinary_values_are_not_experiment_hardcode(self, value):
        inv = inventory({"src/aadistill/ok.py": module(
            "ok", literals=[("f", value, [])])})
        v = RATCHET._violations(inv)
        for rule in ("path_literals", "sha256_literals", "repo_id_literals"):
            assert not v[rule], f"{value!r} was reported as {rule}"

    def test_a_retry_helper_is_not_import_time_registration(self):
        inv = inventory({"src/aadistill/ok.py": module(
            "ok", import_calls=["retry", "backoff", "configure_logging"])})
        assert not RATCHET._violations(inv)["import_time_registration"]

    def test_a_real_registration_still_fires(self):
        """The complement, so the rule above is not just switched off."""
        inv = inventory({"src/aadistill/ok.py": module(
            "ok", import_calls=["register_adapter"])})
        assert RATCHET._violations(inv)["import_time_registration"]
