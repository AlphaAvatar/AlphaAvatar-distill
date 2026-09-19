"""The product path, driven against the layout the runner actually produces.

Attempt 8 reconstructed two leaves byte-identically and lost both. Every check
was green: `reconstructed_leaves` read a path nothing writes, returned `[]`,
`leaves_secured` reported "no leaf was reconstructed, so none is owed off-pod",
the teardown gate allowed, and the pod was deleted. 56 minutes of GPU, and the
exact failure this session exists to repair.

Two defects, and the second is the dangerous one:

* the evidence path was invented rather than derived from the artifact spec;
* an unreadable evidence file was reported as "nothing was reconstructed",
  collapsing "I could not tell" into "there is nothing there".

These tests build the real directory layout — `<scr>/store/extracted/<spec
pattern>` — and assert both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in ("src", "scripts", "scripts/autoinit", "scripts/pod"):
    if str(ROOT / extra) not in sys.path:
        sys.path.insert(0, str(ROOT / extra))

import autoinit_c2_replay_launch as L  # noqa: E402


class _Ctx:
    def __init__(self, scr: Path):
        self.args = type("A", (), {"scr": str(scr), "ckpt_store": str(scr / "dest"),
                                   "ckpt_fetch_limit_min": 1})()
        self.evidence: dict = {}
        self.said: list[str] = []
        self.say = self.said.append
        self.products_eligible = True
        self.host = "127.0.0.1"
        self.target = type("T", (), {"port": 22})()


def _write_evidence(scr: Path, leaves: list[dict]) -> Path:
    """At the location the runner really extracts to."""
    from collect_artifacts import load_specs

    spec = [e for e in load_specs(
        str(ROOT / "configs/autoinit/c2_replay_artifacts.json"))
        if e.artifact_class == "session_evidence"][0]
    path = scr / "store" / "extracted" / spec.pattern
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"leaves": leaves}) + "\n")
    return path


def _leaf(state_id: str, matched: bool = True) -> dict:
    return {"state_id": state_id, "identity_matches_attempt3": matched}


def test_the_reader_finds_evidence_where_the_runner_extracts_it():
    """The layout that actually exists, not the one the first version guessed."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        scr = Path(tmp)
        _write_evidence(scr, [_leaf("aaa"), _leaf("bbb"),
                              _leaf("ccc", matched=False)])
        ctx = _Ctx(scr)
        found = L.reconstructed_leaves(ctx)
    assert found is not None
    assert {f["state_id"] for f in found} == {"aaa", "bbb"}, found
    assert "store/extracted" in ctx.evidence["leaf_evidence_read_from"]


def test_attempt_eights_evidence_yields_its_two_leaves():
    """The real file from the run that lost them. If the reader had worked,
    these are the two state ids it would have fetched."""
    import tempfile

    real = Path("/home/ecs-user/aad-scratch-c2replay-a8/store/"
                "c2_replay_evidence.json")
    if not real.is_file():                      # the run's scratch may be gone
        return
    record = json.loads(real.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        scr = Path(tmp)
        _write_evidence(scr, record["leaves"])
        found = L.reconstructed_leaves(_Ctx(scr))
    assert found is not None
    assert {f["state_id"] for f in found} == {
        "d005dfb27e7bb5adc46eb988ae79e2a9",
        "7da1e4e2254e59766581e777b8066b7e"}, found


def test_unreadable_evidence_is_UNKNOWN_not_empty():
    """The distinction that cost two leaves."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        assert L.reconstructed_leaves(_Ctx(Path(tmp))) is None


def test_teardown_is_refused_when_the_evidence_cannot_be_read():
    """'I found no evidence' and 'nothing was reconstructed' are different
    findings. Reporting the first as the second is what allowed the delete."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ok, why = L.leaves_secured(_Ctx(Path(tmp)), [])
    assert ok is False
    assert "UNKNOWN" in why
    assert "Looked in" in why


def test_teardown_is_allowed_when_the_driver_really_reconstructed_nothing():
    """Guards the guard: a run that legitimately produced no leaf must not be
    blocked, or an early failure could never tear down."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        scr = Path(tmp)
        _write_evidence(scr, [])
        ok, why = L.leaves_secured(_Ctx(scr), [])
    assert ok is True
    assert "reconstructed no leaf" in why


def test_teardown_is_refused_while_a_reconstructed_leaf_is_unfetched():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        scr = Path(tmp)
        _write_evidence(scr, [_leaf("aaa"), _leaf("bbb")])
        ctx = _Ctx(scr)
        ok, why = L.leaves_secured(ctx, [
            {"artifact": "c2_replay_leaf", "state_id": "aaa", "rc": 0,
             "matched": True}])
    assert ok is False
    assert "bbb" in why


def test_teardown_is_allowed_once_every_reconstructed_leaf_is_verified():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        scr = Path(tmp)
        _write_evidence(scr, [_leaf("aaa")])
        ok, why = L.leaves_secured(_Ctx(scr), [
            {"artifact": "c2_replay_leaf", "state_id": "aaa", "rc": 0,
             "matched": True}])
    assert ok is True, why


def test_a_fetch_that_did_not_verify_does_not_count_as_secured():
    """rc=0 is not verification; `matched` is."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        scr = Path(tmp)
        _write_evidence(scr, [_leaf("aaa")])
        ok, why = L.leaves_secured(_Ctx(scr), [
            {"artifact": "c2_replay_leaf", "state_id": "aaa", "rc": 0,
             "matched": False}])
    assert ok is False
    assert "aaa" in why
