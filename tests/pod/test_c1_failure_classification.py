"""A generic C1 failure must not be reported as a replay mismatch.

Attempt 8's driver crashed in stage D with `KeyError: 'input_ids'`, before a
single artifact digest had been computed. The launcher printed:

    C1_REPLAY_MISMATCH is the scientific stop: the frozen path did not reproduce
    its recorded digest …

It had not. Nothing had been compared. `SessionRunner` prints `failure_note` for
**any** marker in `MarkerPolicy.failure` — it is one constant string, chosen
before the run and blind to which marker fired — so a note that describes one
specific failure describes every failure incorrectly.

That matters more here than it would elsewhere. C1 exists to test whether a
frozen path still reproduces its recorded identity, so a spurious "did not
reproduce" is a false claim about exactly the property under measurement, in the
session's own operator-facing output.

The fix is that the shared note asserts nothing. The **explicit** path is
untouched and is re-checked here: `C1Driver.replay_mismatch` writes the evidence,
reads it back, and only then emits `MARKER:C1_REPLAY_MISMATCH`.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "pod"))
sys.path.insert(0, str(REPO / "scripts" / "autoinit"))

import autoinit_c1_driver as D  # noqa: E402

from session_specs import all_specs  # noqa: E402

RUNNER = REPO / "src/aadistill/infrastructure/session_runner.py"
LAUNCHER = REPO / "scripts/pod/autoinit_c1_launch.py"

#: Claims a marker-blind note is not entitled to make. Each is a statement about
#: WHICH stage failed or WHY, and the note is printed before either is known.
FORBIDDEN = (
    "did not reproduce",
    "recorded digest",
    "C1_REPLAY_MISMATCH",
    "mismatch evidence",
    "no recovery training was",
)


@pytest.fixture(scope="module")
def c1_spec():
    for _name, module, _args, spec in all_specs():
        if getattr(module, "__name__", "") == "autoinit_c1_launch":
            return spec
    pytest.fail("the C1 session spec was not produced by its own launcher")


def test_the_generic_failure_note_claims_nothing_about_which_stage_failed(c1_spec):
    note = c1_spec.markers.failure_note
    assert note, "C1 still needs a failure note; a silent teardown is worse"
    lowered = note.lower()
    for phrase in FORBIDDEN:
        assert phrase.lower() not in lowered, (
            f"the C1 failure note claims {phrase!r}. It is printed for every "
            f"marker in {list(c1_spec.markers.failure)}, including the generic "
            "C1_FAILED that attempt 8 actually produced.")


def test_the_note_directs_the_reader_to_the_evidence_instead(c1_spec):
    """Neutral is not the same as useless: it must still say what to do."""
    note = c1_spec.markers.failure_note.lower()
    assert "terminal marker" in note
    assert "evidence" in note


def test_every_failure_marker_shares_the_one_note(c1_spec):
    """Why neutrality is the only available fix.

    `MarkerPolicy` carries a single `failure_note`, so the note cannot be
    conditioned on the marker. If that ever changes — a per-marker mapping —
    this fails and the neutrality rule can be revisited on purpose.
    """
    from aadistill.infrastructure.session import MarkerPolicy

    assert MarkerPolicy.__dataclass_fields__["failure_note"].type in ("str", str)
    assert len(c1_spec.markers.failure) > 1
    assert "C1_FAILED" in c1_spec.markers.failure
    assert "C1_REPLAY_MISMATCH" in c1_spec.markers.failure


def test_the_runner_prints_the_note_without_consulting_the_marker():
    """Executed against the real source: the note's guard is the generic hit.

    If a future edit made the note conditional, the premise above would be false
    and the neutrality requirement would need restating rather than silently
    holding.
    """
    tree = ast.parse(RUNNER.read_text())
    uses = [n for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "failure_note"]
    assert len(uses) == 1, f"{len(uses)} uses of failure_note; expected exactly 1"

    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and any(
                  u is d for u in uses for d in ast.walk(n.test))]
    assert not guards, "failure_note is now inside its own marker-specific test"

    enclosing = [n for n in ast.walk(tree)
                 if isinstance(n, ast.If)
                 and any(u in ast.walk(n) for u in uses)]
    assert enclosing, "failure_note is no longer emitted from the poll loop"
    innermost = min(enclosing, key=lambda n: len(list(ast.walk(n))))
    printed = ast.unparse(innermost.test)
    assert "REPLAY" not in printed.upper(), (
        f"the note is now guarded by {printed!r}, a marker-specific test")


def test_attempt_8s_exact_sentence_is_gone():
    """The literal string that was printed against a KeyError."""
    text = LAUNCHER.read_text()
    banned = ("the frozen path did not reproduce its recorded digest",
              "the frozen path \"\n                \"did not reproduce")
    for phrase in banned:
        assert phrase not in text.replace("\n", " ").replace("  ", " "), phrase


def test_the_launcher_names_the_mismatch_marker_only_as_a_marker():
    """`C1_REPLAY_MISMATCH` may be polled for. It may not be narrated."""
    tree = ast.parse(LAUNCHER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "C1_REPLAY_MISMATCH" in node.value:
                assert node.value == "C1_REPLAY_MISMATCH", (
                    "the launcher embeds C1_REPLAY_MISMATCH inside prose: "
                    f"{node.value[:90]!r}")


# --- the explicit path, unchanged -------------------------------------------

def test_only_the_driver_that_observed_a_mismatch_emits_its_marker():
    src = (REPO / "scripts/pod/autoinit_c1_driver.py").read_text()
    tree = ast.parse(src)
    emitters = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(fn):
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name) and call.func.id == "mark"
                    and call.args and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "C1_REPLAY_MISMATCH"):
                emitters.append(fn.name)
    assert emitters == ["replay_mismatch"], emitters


def test_a_real_mismatch_still_records_that_the_path_did_not_reproduce(tmp_path,
                                                                       monkeypatch):
    """The claim is preserved exactly where it is earned — in the record.

    Removing it from the launcher's blanket note must not remove it from the
    artifact a reviewer actually reads.
    """
    monkeypatch.setattr(D, "AUDIT", tmp_path)
    monkeypatch.setattr(D, "STATUS", tmp_path / "c1.status")

    driver = D.C1Driver.__new__(D.C1Driver)
    driver.ev = {"stages": {}}
    driver.save = lambda: None
    driver.usd = lambda: 0.0

    exc = D.FixedPathDigestMismatch(2, "parent", "a" * 64, "b" * 64,
                                    {"steps": []})
    driver.replay_mismatch(exc, {"gpu": "none"}, [])

    record = json.loads((tmp_path / "c1_replay_record.json").read_text())
    assert record["stage"] == "D"
    assert record["training_started"] is False
    assert "did not reproduce its recorded digest" in record["meaning"]
    assert record["expected"] == "a" * 64 and record["actual"] == "b" * 64
    assert "MARKER:C1_REPLAY_MISMATCH" in (tmp_path / "c1.status").read_text()


def test_a_generic_failure_writes_no_mismatch_record(tmp_path, monkeypatch):
    """`fail()` is the generic path. It must leave no mismatch artifact behind.

    This is attempt 8's shape: stage D raised an ordinary exception, so the
    session had a failed stage and no comparison — and therefore nothing that
    entitles any reader, or any log line, to the word "mismatch".
    """
    monkeypatch.setattr(D, "AUDIT", tmp_path)
    monkeypatch.setattr(D, "STATUS", tmp_path / "c1.status")

    driver = D.C1Driver.__new__(D.C1Driver)
    driver.ev = {"stages": {}}
    driver.save = lambda: None
    driver.usd = lambda: 0.0

    driver.fail("D", "KeyError: 'input_ids'")

    assert not (tmp_path / "c1_replay_record.json").exists()
    status = (tmp_path / "c1.status").read_text()
    assert "MARKER:STAGE_FAILED:D" in status
    assert "C1_REPLAY_MISMATCH" not in status
    stage = driver.ev["stages"]["D"]
    assert stage["passed"] is False
    assert "mismatch" not in stage["reason"].lower()
    assert "mismatch_record" not in stage
