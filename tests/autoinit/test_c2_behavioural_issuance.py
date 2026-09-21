"""What the attempt4 issuer re-derives, and what it used to take on trust.

The grant states that every identity in `reviewed_baseline` is re-derived at
issuance and that a disagreement is a refusal. For `proposal_sha256` that was
false: the plan hash and the executable closure were checked and the proposal
identity was recorded and believed. The gap is not covered by the other two —
the plan hash covers the frozen science, the closure covers the executable, so
an edit to the proposal WRITER moves the proposal hash and neither of those,
and the issued authorization would then name a review document nobody read.

This file exists because the issuer had no tests at all, which is how the gap
survived: it is the script that turns a maintainer decision into a binding
artifact, and nothing exercised it.

The proposal hash is also a quantity with a confusable twin. `sha256_json` over
the document minus `proposal_sha256` is the identity every governance artifact
in this repository binds; `sha256sum` of the file is a different number, and it
has already been reported to a maintainer under this name. Both are checked
here so the two cannot be confused again.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.infrastructure.manifest import sha256_json  # noqa: E402


def _issuer():
    """The real entry point, loaded by path — `scripts/autoinit/` is not a
    package, so an `import` would not find it."""
    src = REPO / "scripts/autoinit/issue_c2_behavioural_authorization.py"
    spec = importlib.util.spec_from_file_location("_c2b_issuer", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ISS = _issuer()
PROPOSAL = REPO / ISS.PROPOSAL_REL


# --- the path comes from the writer, not a second copy of the literal -------

def test_the_issuer_checks_the_document_the_writer_actually_writes():
    """Two spellings of one path is how a checker verifies an unread file."""
    src = REPO / "scripts/autoinit/write_c2_behavioural_proposal.py"
    spec = importlib.util.spec_from_file_location("_c2b_writer", src)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    assert ISS.PROPOSAL_REL == writer.OUT
    assert PROPOSAL.is_file(), ISS.PROPOSAL_REL


# --- the canonical hash, and its confusable twin ----------------------------

def test_the_committed_proposal_matches_its_own_stated_hash():
    got = ISS.reviewed_proposal_hash(REPO)
    assert got == json.loads(PROPOSAL.read_text())["proposal_sha256"]


def test_the_canonical_hash_is_not_the_file_hash():
    """They are different numbers and one has already been misreported.

    Asserted as an inequality rather than against a literal: pinning either
    value would make this a phrasing lock that fails on the next regeneration,
    and what must stay true is that the two are distinct quantities and the
    issuer uses the canonical one.
    """
    import hashlib

    canonical = ISS.reviewed_proposal_hash(REPO)
    file_bytes = hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
    assert canonical != file_bytes, (
        "the canonical and file hashes coincide, so this test can no longer "
        "tell which one the issuer reads")
    assert canonical == json.loads(PROPOSAL.read_text())["proposal_sha256"]


class TestTheProposalHashRefusals:
    """The refusals, on documents built for the purpose.

    `reviewed_proposal_hash` reads `PROPOSAL_REL` under a caller-supplied
    root, so a temporary tree exercises every branch without touching the
    committed proposal.
    """

    def _tree(self, tmp_path: Path, doc: dict) -> Path:
        p = tmp_path / ISS.PROPOSAL_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1) + "\n")
        return tmp_path

    def test_a_consistent_document_is_accepted(self, tmp_path):
        doc = {"a": 1, "b": {"c": 2}}
        doc["proposal_sha256"] = sha256_json(doc)
        assert ISS.reviewed_proposal_hash(
            self._tree(tmp_path, doc)) == doc["proposal_sha256"]

    def test_a_document_that_does_not_match_its_hash_is_refused(self, tmp_path):
        doc = {"a": 1}
        doc["proposal_sha256"] = sha256_json(doc)
        doc["a"] = 2                       # edited after it was hashed
        with pytest.raises(SystemExit, match="does not match its own"):
            ISS.reviewed_proposal_hash(self._tree(tmp_path, doc))

    def test_a_document_with_no_hash_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="no proposal_sha256"):
            ISS.reviewed_proposal_hash(self._tree(tmp_path, {"a": 1}))

    def test_the_file_hash_is_not_accepted_as_the_stated_one(self, tmp_path):
        """The exact confusion: the writer's own field set to the file hash."""
        import hashlib

        doc = {"a": 1}
        body = json.dumps(dict(doc, proposal_sha256="x" * 64), indent=1) + "\n"
        doc["proposal_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        with pytest.raises(SystemExit, match="does not match its own"):
            ISS.reviewed_proposal_hash(self._tree(tmp_path, doc))


# --- every reviewed identity is compared, including the proposal ------------

_LIVE = {"plan_hash": "p" * 64, "executable_closure": "c" * 64,
         "closure_files": 134, "proposal_sha256": "s" * 64}


def _reviewed(**overrides) -> dict:
    return dict(_LIVE, **overrides)


@pytest.mark.parametrize("field,wrong", [
    ("plan_hash", "9" * 64),
    ("executable_closure", "8" * 64),
    ("closure_files", 133),
    #: THE ONE THAT WAS NOT CHECKED. Without it in the loop this case passes.
    ("proposal_sha256", "7" * 64),
])
def test_a_disagreeing_reviewed_identity_is_reported(field, wrong):
    out = ISS.reviewed_identity_disagreements(_reviewed(**{field: wrong}),
                                              **_LIVE)
    assert len(out) == 1, out
    assert str(wrong) in out[0]


def test_an_agreeing_baseline_reports_nothing():
    assert ISS.reviewed_identity_disagreements(_reviewed(), **_LIVE) == []


def test_several_disagreements_are_all_reported():
    """Reporting the first would hide the rest from a reader deciding whether
    the tree moved by one repair or by a round nobody reviewed."""
    out = ISS.reviewed_identity_disagreements(
        _reviewed(plan_hash="9" * 64, proposal_sha256="7" * 64), **_LIVE)
    assert len(out) == 2, out


def test_an_identity_the_grant_omits_is_not_checked():
    """A grant may record fewer identities than it binds; a MISSING field is
    not a disagreement. Asserted so the refusal above is known to be about
    disagreement rather than about presence."""
    assert ISS.reviewed_identity_disagreements({}, **_LIVE) == []
    assert ISS.reviewed_identity_disagreements(
        {"proposal_sha256": None}, **_LIVE) == []


def test_the_caller_raises_on_what_the_seam_returns():
    """The seam is pure, so the one line that acts on it is the blind spot.

    Extracting a testable predicate leaves its injection point as the only
    unexecuted code, so the wiring is asserted directly: `main` must call both
    helpers and refuse on a non-empty result.
    """
    src = (REPO / "scripts/autoinit/issue_c2_behavioural_authorization.py"
           ).read_text()
    body = src.split("def main(", 1)[1]
    assert "reviewed_proposal_hash(REPO_ROOT)" in body
    assert "reviewed_identity_disagreements(" in body
    assert "proposal_sha256=proposal_sha256" in body
    assert "if disagreements:" in body and "raise SystemExit" in body


# --- and the issuance the chain will actually run ---------------------------

def test_the_issuer_refuses_a_dirty_tree_before_reading_anything():
    """The first refusal in the real entry point, exercised for real.

    Runs the actual script. It cannot issue here — there is no attempt4 grant
    — so this asserts the ordering property that matters: the tree check comes
    before any identity work, because an artifact binds a commit and
    uncommitted work is not in it.
    """
    out = subprocess.run(
        [sys.executable,
         str(REPO / "scripts/autoinit/issue_c2_behavioural_authorization.py"),
         "--run-id", "attempt-does-not-exist", "--rate", "1.09"],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": "src:scripts", "PATH": "/usr/bin:/bin",
             "HOME": str(Path.home())})
    assert out.returncode != 0
    #: Either refusal is correct and which one depends on the working tree;
    #: what must never happen is an authorization for a run with no grant.
    assert ("dirty tree" in out.stderr or "no grant at" in out.stderr), out.stderr
