"""Writer path == success collector path == failure collector path.

Phase-B attempt 3 hit its Stage-1 search deadline after 544.7 min. The one
artifact that would say *how close it came* — per-state timings, beam contents,
how many admissible leaves existed when the clock ran out — is the search
journal, and it was deleted with the pod.

The driver wrote `artifacts/autoinit/phase_b_search/states.jsonl`. Both artifact
specs collected `autoinit/phase_a_search/states.jsonl`. One directory name apart.
Because the failed-run entry is `required: false, min_matches: 0`, the collector
matched nothing, the manifest reported `missing: 0`, the teardown gate passed,
and nothing anywhere said the journal had not come home.

So this file does not check that a path *looks* right. It reads the path the
driver will actually write to, derives what a collector must therefore match, and
requires both specs to match exactly that — success and failure. Mutating
`phase_b_search` back to `phase_a_search`, in the driver or in either spec, must
turn this red.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/pod"))

import autoinit_phase_b_driver as pbd  # noqa: E402
import autoinit_phase_b_launch as pbl  # noqa: E402

#: The archive root the collector's patterns are relative to. `manifest.json` from
#: attempt 3 records `root: /workspace/aad/artifacts`, i.e. `<repo>/artifacts`.
ARCHIVE_ROOT = pbd.REPO / "artifacts"


def _args():
    return pbl.build_parser().parse_args(
        ["--scr", "/tmp/does-not-matter", "--session-commit", "d" * 40,
         "--bundle", "b.bundle"])


def spec_paths() -> dict[str, str]:
    """The two spec files this session's `ArtifactPolicy` actually names."""
    policy = pbl.spec(_args()).artifacts
    return {"success": policy.spec_success, "failed": policy.spec_failed}


def journal_entry(spec_path: str) -> dict:
    entries = json.loads((REPO / spec_path).read_text())["entries"]
    matching = [e for e in entries if e["artifact_class"] == "search_journal"]
    assert len(matching) == 1, f"{spec_path} declares {len(matching)} search_journal entries"
    return matching[0]


def writer_pattern() -> str:
    """What a collector must match, derived from where the driver writes."""
    journal = pbd.SEARCH_WORKDIR / "states.jsonl"
    return journal.relative_to(ARCHIVE_ROOT).as_posix()


def test_the_driver_writes_inside_the_archive_root():
    """A journal written outside the archive root could never be collected."""
    assert pbd.SEARCH_WORKDIR.is_relative_to(ARCHIVE_ROOT), (
        f"{pbd.SEARCH_WORKDIR} is not under {ARCHIVE_ROOT}, so no collector "
        "pattern could reach it")
    assert writer_pattern() == "autoinit/phase_b_search/states.jsonl"


def test_phase_b_does_not_write_into_the_phase_a_search_workdir():
    """Two sessions retaining different journals must not share a path."""
    import autoinit_phase_a_driver as pad

    assert pbd.SEARCH_WORKDIR != pad.SEARCH_WORKDIR


@pytest.mark.parametrize("which", ["success", "failed"])
def test_both_collectors_match_the_path_the_driver_writes(which):
    entry = journal_entry(spec_paths()[which])
    assert entry["pattern"] == writer_pattern(), (
        f"the {which} spec collects {entry['pattern']!r} but the Phase-B driver "
        f"writes {writer_pattern()!r}. This exact mismatch lost attempt 3's "
        "search journal at the one moment it mattered.")


def test_the_session_uses_phase_b_specs_not_phase_a_ones():
    paths = spec_paths()
    assert paths["success"] != paths["failed"], (
        "a single spec for both outcomes would force a failed run to produce a "
        "successful run's artifacts, which keeps a dead pod billing")
    for which, path in paths.items():
        assert "phase_b" in Path(path).name, (which, path)


def test_a_failed_run_is_not_held_open_by_a_journal_that_was_never_written():
    """The reason the failure entry is optional, asserted rather than assumed.

    A search that died before writing its first state has no journal to bring
    home. Requiring one there would block teardown on the most expensive pod in
    the project. Attempt 3's loss was the PATH being wrong, not this being
    optional — so the fix must not make the failure path strict.
    """
    entry = journal_entry(spec_paths()["failed"])
    assert entry["required"] is False
    assert entry.get("min_matches", 0) == 0


def test_the_success_run_MUST_bring_the_journal_home():
    entry = journal_entry(spec_paths()["success"])
    assert entry["required"] is True
    assert entry.get("min_bytes", 0) >= 1, "an empty journal is not a journal"


def test_the_probe_minimums_are_phase_Bs_arithmetic_not_phase_As():
    """Phase A requires 9 probes. Phase B can legitimately finish with 7.

    The sc rung is conditional: a run that resolved at sb never produces it.
    Carrying Phase A's 9 into Phase B would fail a SUCCESSFUL run at teardown,
    after the science was done and paid for.
    """
    entries = {e["artifact_class"]: e
               for e in json.loads((REPO / spec_paths()["success"]).read_text())["entries"]}
    new_probe_minimum = pbl.RUNG1_PROBES_P2 + pbl.RUNG2_PROBES_P2
    assert new_probe_minimum == 7
    for cls in ("probe_config", "recovery_search_result", "per_sample", "generations"):
        assert entries[cls]["min_matches"] == new_probe_minimum, cls
    # The journal additionally holds the 8 imported citations, which stage 0
    # fails closed without.
    assert entries["probe_journal"]["min_matches"] == 8 + new_probe_minimum


def test_the_telemetry_stream_travels_with_both_outcomes():
    """Operational timings are most needed when the run did NOT finish."""
    for which, path in spec_paths().items():
        entries = json.loads((REPO / path).read_text())["entries"]
        tele = [e for e in entries if e["artifact_class"] == "search_telemetry"]
        assert len(tele) == 1, which
        assert tele[0]["pattern"] == (
            pbd.SEARCH_WORKDIR / "telemetry.jsonl").relative_to(ARCHIVE_ROOT).as_posix()
        assert tele[0]["required"] is False, (
            "telemetry is diagnostic; a missing timing file must never hold a pod open")
