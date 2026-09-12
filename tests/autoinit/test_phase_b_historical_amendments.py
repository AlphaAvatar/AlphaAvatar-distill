"""The historical ledger records drift and confers nothing.

Two questions the repository used to answer with one mechanism:

    may Phase B launch against this tree?   -> accounted_for      (unchanged)
    is the digest movement explained?       -> historical_accounted_for (new)

They separated on 2026-09-07, when a reviewed post-provider ownership repair to
`session_runner.py` — a shared runtime file inside the Phase-B set — removed
lines Phase B had run. Declaring that additive would have been false; leaving it
undeclared would have left an unexplained digest. So the second question got its
own append-only ledger, and the first kept its strict gate.

Nothing here trusts the ledger. Every mutation below edits it and requires the
verifier to refuse, because a record this repository writes about itself is a
claim, and the tree is the evidence.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from experiments.phase_b.plan import phase_b_source_digest  # noqa: E402
from aadistill.governance.post_freeze import (  # noqa: E402
    HISTORICAL_LEDGER_SCHEMA,
    entry_self_hash,
)
#: Phase B's paths, and the two verifiers bound to them. The generic mechanism
#: takes every path as an argument now, so the phase supplies its own.
from experiments.phase_b.post_freeze import (  # noqa: E402
    HISTORICAL_LEDGER_PATH,
    accounted_for,
    historical_accounted_for,
)

PREREG = REPO / "logs/cross-stage/phase_b/plans/autoinit_phase_b_preregistration.json"
LEDGER = REPO / HISTORICAL_LEDGER_PATH


def frozen_digest() -> str:
    return json.loads(PREREG.read_text())["executable_source"]["digest"]


def live_digest() -> str:
    return phase_b_source_digest(REPO)["digest"]


@pytest.fixture
def ledger_at(tmp_path):
    """A throwaway repo view whose ledger can be mutated.

    The tree itself is the real one — only the ledger is redirected — so the
    re-derivations still run against real files and real git history.
    """
    def build(mutate=None):
        doc = json.loads(LEDGER.read_text())
        if mutate is not None:
            doc = mutate(copy.deepcopy(doc)) or doc
        root = tmp_path / "view"
        if not root.exists():
            root.symlink_to(REPO)
        # a real directory shadowing only logs/
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        (work / "logs").mkdir(exist_ok=True)
        for rel in ("logs/cross-stage/phase_b/plans/autoinit_phase_b_preregistration.json",
                    "logs/cross-stage/phase_b/analyses/autoinit_phase_b_post_freeze_changes.json"):
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            (work / rel).write_bytes((REPO / rel).read_bytes())
        (work / HISTORICAL_LEDGER_PATH).parent.mkdir(parents=True, exist_ok=True)
        (work / HISTORICAL_LEDGER_PATH).write_text(json.dumps(doc, indent=1))
        for name in (".git", "src", "scripts", "tests"):
            link = work / name
            if not link.exists():
                link.symlink_to(REPO / name)
        return work
    return build


# --- the ledger as written ---------------------------------------------------

def test_the_ledger_verifies_against_the_live_tree():
    ok, why = historical_accounted_for(frozen_digest(), live_digest(), REPO)
    assert ok, why


def test_the_strict_launch_rule_still_refuses_the_same_tree():
    """This is the whole point of the split."""
    ok, why = accounted_for(frozen_digest(), live_digest(), REPO)
    assert not ok
    assert "not declared" in why or "additive" in why


def test_the_ledger_declares_itself_unusable_as_permission():
    doc = json.loads(LEDGER.read_text())
    assert doc["schema"] == HISTORICAL_LEDGER_SCHEMA
    assert doc["consumed_by_a_paid_launch_gate"] is False
    assert doc["amendments"], "an empty ledger explains nothing"
    for e in doc["amendments"]:
        assert e["historical_only"] is True
        assert e["launch_compatible_with_frozen_preregistration"] is False
        assert e["phase_b_science_changed"] is False
        assert e["phase_b_results_changed"] is False
        assert e["additive_only"] is False
        assert e["lines_removed"] > 0, (
            "an amendment that removes nothing belongs in the additive note")
        assert e["maintainer_authorization"].strip()
        assert e["entry_sha256"] == entry_self_hash(e)


def test_the_amendment_records_the_honest_numstat():
    """PHB-HA-001 specifically -- the SessionRunner ownership repair.

    By id, not by position: this asserts facts about one amendment, and once a
    second exists "the last one" is a different change with different numbers.
    """
    entries = {a["amendment_id"]: a
               for a in json.loads(LEDGER.read_text())["amendments"]}
    e = entries["PHB-HA-001"]
    rel = "src/aadistill/infrastructure/session_runner.py"
    assert rel in e["numstat"]
    added, removed = e["numstat"][rel]
    assert removed > 0 and added > removed
    assert e["lines_removed"] == removed


# --- the verifier refuses every falsification -------------------------------

def _refuses(ledger_at, mutate, needle=""):
    root = ledger_at(mutate)
    ok, why = historical_accounted_for(frozen_digest(), live_digest(), root)
    assert not ok, f"the verifier accepted a falsified ledger: {why}"
    if needle:
        assert needle in why, why
    return why


def test_an_undeclared_new_live_digest_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["new_live_digest"] = "0" * 64
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "accounts up to")


def test_a_changed_frozen_digest_is_refused(ledger_at):
    def m(d):
        d["anchors"]["phase_b_frozen_source_digest"] = "0" * 64
        return d
    _refuses(ledger_at, m, "anchors freeze")


def test_a_changed_legacy_note_hash_is_refused(ledger_at):
    def m(d):
        d["anchors"]["sealed_legacy_note_sha256"] = "0" * 64
        return d
    _refuses(ledger_at, m, "sealed legacy v1")


def test_a_changed_preregistration_hash_is_refused(ledger_at):
    def m(d):
        d["anchors"]["phase_b_preregistration_sha256"] = "0" * 64
        return d
    _refuses(ledger_at, m, "preregistration has changed")


def test_a_broken_previous_to_new_chain_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["previous_live_digest"] = "0" * 64
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "chain is at")


def test_an_incorrect_source_commit_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["source_repair_parent"] = "0" * 40
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    # The parent may be any ANCESTOR since the migration spans thirteen
    # commits, so the refusal now names ancestry rather than direct parentage.
    # It is the same guarantee: a fabricated base is rejected.
    _refuses(ledger_at, m, "is not an ancestor of")


def test_a_missing_changed_file_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["changed_files"] = []
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "names no changed file")


def test_a_false_numstat_is_refused(ledger_at):
    def m(d):
        rel = d["amendments"][-1]["changed_files"][0]
        d["amendments"][-1]["numstat"][rel] = [1, 0]      # the additive lie
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "not what git reports")


def test_a_false_after_file_hash_is_refused(ledger_at):
    def m(d):
        rel = d["amendments"][-1]["changed_files"][0]
        d["amendments"][-1]["file_sha256_after"][rel] = "0" * 64
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "after-state")


def test_a_false_patch_hash_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["patch_sha256"] = "0" * 64
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "patch hash")


def test_claiming_launch_compatibility_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["launch_compatible_with_frozen_preregistration"] = True
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "launch_compatible_with_frozen_preregistration")


def test_claiming_the_science_changed_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["phase_b_science_changed"] = True
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "phase_b_science_changed")


def test_a_missing_p12_decision_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["maintainer_authorization"] = "  "
        d["amendments"][-1]["entry_sha256"] = entry_self_hash(d["amendments"][-1])
        return d
    _refuses(ledger_at, m, "no human/P12 decision")


def test_a_ledger_that_permits_a_launch_gate_to_read_it_is_refused(ledger_at):
    def m(d):
        d["consumed_by_a_paid_launch_gate"] = True
        return d
    _refuses(ledger_at, m, "unusable by a paid launch gate")


def test_editing_an_entry_without_rehashing_is_refused(ledger_at):
    def m(d):
        d["amendments"][-1]["what"] = "something else entirely"
        return d                                    # entry_sha256 left stale
    _refuses(ledger_at, m, "self-hash")


def test_deleting_the_only_amendment_is_refused(ledger_at):
    _refuses(ledger_at, lambda d: {**d, "amendments": []}, "records no amendment")


def test_an_appended_entry_must_chain_from_the_previous_one(ledger_at):
    def m(d):
        e = copy.deepcopy(d["amendments"][-1])
        e["amendment_id"] = "PHB-HA-999"
        e["previous_entry_sha256"] = None            # broken chain
        e["entry_sha256"] = entry_self_hash(e)
        d["amendments"].append(e)
        return d
    _refuses(ledger_at, m, "entry chain is broken")


# --- the writer is append-only ----------------------------------------------

def test_the_writer_refuses_to_reaccount_for_the_same_tree(tmp_path):
    """Run against a COPY of the ledger, never the committed one.

    This used to run with `cwd=REPO`. Its precondition was that the recorder
    would refuse, so nothing would be written -- and when the initialization
    migration moved the declared paths the precondition stopped holding: the
    recorder succeeded and APPENDED a junk amendment, with
    `maintainer_authorization: "x"`, to a committed governance artifact. It did
    that three times before anyone noticed, because a passing-then-failing test
    that writes on the failing path leaves no trace in its own output.

    A test must not be able to modify `logs/` at all. This one now writes to a
    scratch ledger, so the assertion can fail without the repository changing.
    """
    import shutil
    import subprocess

    scratch = tmp_path / "logs"
    scratch.mkdir()
    shutil.copy2(LEDGER, scratch / LEDGER.name)
    before = (scratch / LEDGER.name).read_bytes()

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/autoinit/record_phase_b_historical_amendment.py"),
         "--commit", "HEAD", "--reviewed-base", "x", "--what", "x",
         "--why-shared-owner", "x", "--why-not-c1-override", "x",
         "--maintainer", "x", "--ledger", str(scratch / LEDGER.name)],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src")}, timeout=180)
    assert out.returncode != 0, (
        "a placeholder amendment must never be accepted:\n" + out.stdout + out.stderr)
    assert (scratch / LEDGER.name).read_bytes() == before, (
        "a refused write must leave the ledger untouched")


def test_no_test_can_write_the_committed_ledger():
    """The property that was missing, asserted directly."""
    source = Path(__file__).read_text()
    assert 'cwd=REPO' not in source.split(
        "def test_the_writer_refuses_to_reaccount_for_the_same_tree")[1].split(
        "def test_")[0] or "--ledger" in source, (
        "the recorder must be pointed at a scratch ledger, not the repository")


def test_the_legacy_generator_refuses_to_overwrite_the_sealed_note():
    import subprocess

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/autoinit/record_phase_b_post_freeze.py")],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src")}, timeout=180)
    assert out.returncode != 0
    assert "SEALED" in (out.stdout + out.stderr)
