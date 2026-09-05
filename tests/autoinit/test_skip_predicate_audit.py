"""Every skip predicate in the C1-selected suite is accounted for.

C1 attempt 5 died at the pod CPU test gate for `$0.3150` on a predicate keyed to
`AAD_SYNTHETIC_HF_TOKEN` — a flag the SIMULATOR sets and the pod does not. The
individual defect is repaired in `test_staging_contract.py`; this closes the
CLASS, by requiring every predicate whose premise can differ between the dev box
and the pod to carry an explicit classification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/autoinit"))

import audit_skip_predicates as A  # noqa: E402


@pytest.fixture(scope="module")
def rec():
    return A.audit(REPO)


def test_no_predicate_is_unaccounted_for(rec):
    assert rec["unaccounted"] == [], (
        "a skip predicate can decide differently on a pod and nothing says so. "
        "Resolve it, or register it in " + A.REGISTRY)
    assert rec["verdict"] == "PASS"


def test_the_registry_holds_no_stale_entry(rec):
    """Otherwise it rots into a list of excuses for predicates that moved."""
    assert rec["stale_registry_entries"] == [], (
        "these registry lines no longer hold a predicate that needs one")


def test_the_audit_covers_the_session_s_own_selection(rec):
    """Not `tests/` — the modules C1 actually runs, from its own manifest."""
    from aadistill.autoinit import staging_contract as sc  # noqa: F401
    assert rec["modules_scanned"] > 100
    assert rec["test_ignores"] == [
        "tests/data/test_recovery_corpus_pipeline.py",
        "tests/pod/test_phase_a_stages1_5_execute.py",
        "tests/autoinit/test_phase_b_reuse_hostlocal.py",
        "tests/autoinit/test_stage1_import.py",
    ]
    for ignored in rec["test_ignores"]:
        assert not any(p["file"] == ignored for p in rec["predicates"]), (
            f"{ignored} is ignored by the session but audited anyway")


def test_the_simulator_marker_is_not_a_premise_anywhere(rec):
    """The attempt-5 defect, as a standing prohibition.

    A simulator flag is a legitimate premise for exactly one thing — whether the
    credential is synthetic — and never for whether an artifact is present.
    """
    offenders = [p["nodeid"] for p in rec["predicates"]
                 if p["signal"] == "simulator_marker"]
    assert offenders == [], (
        "a predicate is keyed on the simulator marker again. The pod does not "
        f"set it, so the guard is inverted there: {offenders}")


def test_a_new_unclassified_predicate_is_refused(tmp_path, monkeypatch):
    """Mutation: the audit must REFUSE an unregistered off-pod premise.

    Without this, `unaccounted == []` proves only that nothing was found — which
    is exactly how a gate that cannot fail looks from the outside.
    """
    real = A.c1_selected_modules

    module = tmp_path / "test_smuggled_in.py"
    module.write_text(
        "import pytest\n"
        "from pathlib import Path\n"
        "STORE = Path('/home/ecs-user/aad-artifacts/nowhere')\n"
        "def test_needs_a_devbox_artifact():\n"
        "    if not STORE.is_dir():\n"
        "        pytest.skip('not here')\n")

    def with_extra(repo):
        files, ignores = real(repo)
        return files + [module], ignores

    monkeypatch.setattr(A, "c1_selected_modules", with_extra)
    mutated = A.audit(REPO)
    assert mutated["verdict"] == "REVIEW"
    assert any("test_smuggled_in.py" in nodeid
               for nodeid, _sig in mutated["unaccounted"]), mutated["unaccounted"]


def test_a_registered_line_that_moved_is_reported_stale(monkeypatch):
    """The other half: registration must not survive the predicate it excused."""
    real = json.loads((REPO / A.REGISTRY).read_text())
    poisoned = dict(real)
    poisoned["predicates"] = dict(real["predicates"])
    poisoned["predicates"]["tests/does_not_exist.py:1"] = {"class": "manifest_derived"}

    original = Path.read_text

    def fake(self, *a, **k):
        if self.name == Path(A.REGISTRY).name:
            return json.dumps(poisoned)
        return original(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake)
    mutated = A.audit(REPO)
    assert "tests/does_not_exist.py:1" in mutated["stale_registry_entries"]
    assert mutated["verdict"] == "REVIEW"


def test_the_committed_audit_record_matches_the_live_one(rec):
    """A record whose subject moved is worse than no record."""
    path = REPO / A.RECORD
    if not path.is_file():
        pytest.skip(f"{A.RECORD} has not been produced yet")
    committed = json.loads(path.read_text())
    assert committed["digest"] == rec["digest"], (
        "the committed skip-predicate audit no longer describes this tree; "
        "re-run scripts/autoinit/audit_skip_predicates.py --write")
