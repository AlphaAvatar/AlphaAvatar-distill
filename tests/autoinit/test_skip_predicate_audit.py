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


# --- strict CPU-test parity ---------------------------------------------------
#
# "Classified" is not "strictly comparable". Attempt 5's `--strict` comparison
# was correct machinery pointed at two different machines, and would have
# refused a healthy L40S for being one.

def test_strict_parity_is_ready(rec):
    assert rec["parity_unresolved"] == [], rec["parity_unresolved"]
    assert rec["strict_cpu_test_parity_ready"] == "PASS"


def test_every_predicate_has_a_parity_resolution(rec):
    assert "UNRESOLVED" not in rec["parity_by_resolution"]
    assert "REFUSED" not in rec["parity_by_resolution"]
    assert sum(rec["parity_by_resolution"].values()) == rec["n_predicates"]


def test_the_audit_binds_the_cpu_test_environment_it_resolved_against(rec):
    """A parity verdict against an environment that has since moved is not a
    verdict. The contract's digest travels with the record."""
    from aadistill.autoinit import cpu_test_env as cte
    assert rec["cpu_test_environment"]["digest"] == cte.digest()
    assert rec["cpu_test_environment"]["set"]["CUDA_VISIBLE_DEVICES"] == ""


def test_an_image_difference_is_not_evidence_of_dependency_equality(rec):
    """Every optional dependency must NAME what guarantees it on both machines."""
    deps = [p for p in rec["predicates"] if p["signal"] == "optional_dependency"]
    assert len(deps) == 9
    for p in deps:
        assert p["parity"] == "guaranteed_dependency", p["nodeid"]
        why = p["parity_why"]
        assert "not the dev box's venv" not in why
        assert any(k in why for k in ("committed at", "by construction")), why


def test_an_unresolvable_dependency_refuses_rather_than_waiving(monkeypatch):
    """Mutation: an optional dependency with no named guarantee must NOT pass."""
    monkeypatch.setattr(A, "DEPENDENCY_EVIDENCE", {})
    mutated = A.audit(REPO)
    assert mutated["strict_cpu_test_parity_ready"] == "REVIEW"
    assert any(sig == "optional_dependency"
               for _n, sig, _w in mutated["parity_unresolved"])


def test_removing_the_gpu_normalization_breaks_parity(monkeypatch):
    """Mutation: without `CUDA_VISIBLE_DEVICES=""` the GPU predicate decides the
    opposite way on the pod, and the audit must say so rather than pass."""
    rules = {k: v for k, v in A.PARITY_BY_SIGNAL.items() if k != "gpu"}
    monkeypatch.setattr(A, "PARITY_BY_SIGNAL", rules)
    mutated = A.audit(REPO)
    assert mutated["strict_cpu_test_parity_ready"] == "REVIEW"
    assert any(sig == "gpu" for _n, sig, _w in mutated["parity_unresolved"])


def test_a_simulator_marker_can_never_satisfy_parity(monkeypatch):
    """It is refused explicitly, not merely absent."""
    fake = dict(nodeid="tests/x.py::t", signal="simulator_marker", verdict="differs_on_pod",
                expanded="os.environ.get('AAD_SYNTHETIC_HF_TOKEN')", why="", file="tests/x.py",
                line=1)
    res, why = A.parity_of(fake, {}, set(), set())
    assert res == "REFUSED" and "never" in why


def test_the_audit_digest_does_not_depend_on_this_machines_artifacts(monkeypatch):
    """It once did, and the diagnostic caught it.

    `path_parity` asked the LIVE hidden set, so inside the simulation — where the
    artifacts have already been moved aside and the hidden set is empty — the
    audit produced a different digest than on the bare dev box, and the committed
    record failed its own equality check. Parity is derived from the git index
    and the SetupManifest, both of which are the same in either place.
    """
    src = (REPO / "scripts/autoinit/audit_skip_predicates.py").read_text()
    assert "hidden_files(" not in src, (
        "the audit reads the live hidden set again; its digest would depend on "
        "whether the simulator has run")
    assert "def path_parity(p: dict, tracked: set[str], staged: set[str])" in src
