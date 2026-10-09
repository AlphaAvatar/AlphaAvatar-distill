"""Phase C1's own log records, in today's tree.

Split from `tests/docs/test_log_organisation.py` by the 2026-10-03 convergence
round. That file keeps the MECHANISM -- the renderers, the stage index, the
ownership and path resolution, derived-navigation consistency, closeout parsing
semantics -- all of which hold for any experiment and most of which are pinned
against synthetic fixtures.

These ten asked about C1's concrete instances: that every C1 run sits in the
canonical layout, that attempt14's readiness and authorization paths are its own,
that C1's committed verification record names its own raw output, that STATE.md
agrees with it, that superseded verdicts are preserved in C1's readiness history,
and that C1's global paths are pointers rather than records.

The core suite should not go red because an old run record is archived, retired or
reorganised later. These should, which is why they live here.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest  # noqa: F401  (several of the moved tests parametrise)

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def load(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


#: The one tracked `logs/` file the SWEEP ITSELF removes while it runs: the
#: readiness record `record_pod_environment.py` stashes before replacing it,
#: because the suite it runs contains the tests that verify that record.
STASHED_BY_THE_SWEEP = "/governance/readiness.json"


def _missing_tracked_logs() -> list[str]:
    """Tracked `logs/` files not on disk right now, EXCLUDING that one.

    A pod receives a STAGED subset of the repository and the simulator models
    that by moving the rest aside, so anything reading the `logs/` tree as a
    whole describes a different tree there. The condition is OBSERVED rather than
    flagged: git says what should be present and the filesystem says what is. A
    `skipif` keyed on a simulator variable would be inverted on the pod, which
    does not set it -- this repository has already lost a session to that.
    """
    out = subprocess.run(["git", "ls-files", "logs"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.split()
    return [f for f in out
            if not (REPO / f).exists() and not f.endswith(STASHED_BY_THE_SWEEP)]


#: The reason NAMES the files: a fixed string told the attempt-14 sweep record
#: that 23 tests skipped for a partial tree and nothing said which file.
_MISSING_NOW = _missing_tracked_logs()
needs_whole_tree = pytest.mark.skipif(
    bool(_MISSING_NOW),
    reason=("the logs/ tree is partially staged, so a check that reads it as a "
            f"whole would describe a different tree: {len(_MISSING_NOW)} tracked "
            f"file(s) absent, e.g. {_MISSING_NOW[:3]}"))


def _L():
    """The C1 launcher, loaded by path.

    Carried over with the tests that moved: three of them read the launcher's
    own constants to check that a run's readiness, authorization and grant paths
    are the run's own rather than a global one the next attempt overwrites.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "L", REPO / "scripts/stages/stage-1/phase_c1/autoinit_c1_launch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m



def test_every_phase_c1_run_is_in_one_canonical_place():
    """It said attempts 1-12 were under `runs/phase_c1/`; three were, and
    the rest were in two other layouts. They are now in one."""
    index = load("logs/index.json")
    roots = {r["run_id"]: (r.get("root")
                           or (r.get("components") or {}).get("root", ""))
             for r in [*index["runs"], *index["unrecorded"]]
             if r["experiment_id"] == "phase_c1"}
    assert roots, "no phase_c1 runs found at all"
    stray = {k: v for k, v in roots.items()
             if not v.startswith("logs/stages/stage-1/phase_c1/runs/")}
    assert not stray, f"phase_c1 runs outside the canonical layout: {stray}"
    assert len(roots) >= 13, roots


def test_a_committed_record_still_names_its_own_raw_output():
    """The record's `evidence` block must point at paths that belong to the
    sweep it describes, so the two cannot drift apart again.

    Read THROUGH the pointer: that path stopped being the record when a run
    began owning its own readiness evidence, and the pointer carries the
    verdict but not the raw-output paths. The record lives under `runs/`,
    which git ignores, so a tracked-only checkout does not have it — and a
    check about a document that is not in this view is not a failure of it.
    """
    ptr = load("logs/stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json")
    named = ptr.get("record")
    if not named:
        rec = ptr                      # pre-pointer record, still inline
    elif not (REPO / named).is_file():
        pytest.skip(f"{named} is not in this checkout; it is run-owned "
                    "and gitignored")
    else:
        rec = load(named)
    ev = rec.get("evidence") or {}
    assert ev.get("junit") and ev.get("pytest_log"), ev


def test_the_file_the_sweep_stashes_is_not_read_as_a_partial_tree(monkeypatch):
    """The excluded path must be the one the driver actually stashes."""
    src = (REPO / "scripts/shared/pod/record_pod_environment.py").read_text()
    assert "shutil.move(str(live), str(stash))" in src, (
        "the sweep no longer stashes the previous record; if it stopped, "
        "this exclusion is now hiding a real partial tree")
    #: `record_rel = record_path_for(...)` until 2026-09-15: the recorder
    #: called C1's resolver directly. It now stashes the path the
    #: invocation's own `SweepContract` names, which is the same path for
    #: C1 and the right one for anybody else.
    assert "record_rel = sweep.record.record_path" in src

    #: BOTH experiments with a run-owned readiness record, because the
    #: predicate exempts one filename and a second experiment using a
    #: different one would be read as a partial tree.
    from stages.phase_c1.pod_environment import record_path_for as c1_path
    from stages.phase_c2.pod_environment import record_path_for as c2_path

    for stashed in (c1_path("attempt99", "1"), c2_path("attempt99", "1")):
        assert stashed.endswith(STASHED_BY_THE_SWEEP), (
            f"the sweep stashes {stashed}, which the predicate does not "
            "exempt")


def test_state_md_agrees_with_the_record_it_links_to():
    #: The rendered block follows the pointer to the run-owned record, so
    #: in a checkout without that record it renders the pointer's summary
    #: instead and cannot equal a block written where the record was there.
    #: Agreement is only checkable where both documents are.
    ptr = load("logs/stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json")
    named = ptr.get("record")
    if named and not (REPO / named).is_file():
        pytest.skip(f"{named} is not in this checkout; it is run-owned "
                    "and gitignored")
    from maintenance.consolidation.render_log_navigation import (R_BEGIN, R_END,
                                                   render_readiness)
    text = (REPO / "logs/state/current.md").read_text()
    i, j = text.index(R_BEGIN), text.index(R_END) + len(R_END)
    assert text[i:j] == render_readiness(REPO), (
        "STATE.md's readiness block is stale; re-run "
        "scripts/maintenance/consolidation/render_log_navigation.py --write")


@needs_whole_tree


def test_every_superseded_verdict_is_preserved():
    """The live record holds ONE sweep and the next replaces it, so a
    launch-bound FAILURE survived only in git history."""
    hist = load("logs/stages/stage-1/phase_c1/history/readiness_history.json")
    kinds = {(e["record_kind"], e["verdict"]) for e in hist["entries"]}
    assert ("launch_bound", "FAIL") in kinds, (
        "the failed launch-bound sweep is not preserved anywhere outside "
        "git history")
    assert len(hist["entries"]) > 10, len(hist["entries"])


def test_each_run_gets_its_own_readiness_path():
    from stages.phase_c1.pod_environment import record_path_for
    a = record_path_for("attempt13", "1")
    b = record_path_for("attempt14", "1")
    assert a != b
    assert a.startswith("logs/stages/stage-1/phase_c1/runs/attempt13/")
    assert a.endswith("governance/readiness.json")


def test_each_run_gets_its_own_authorization_path():
    L = _L()
    a, b = L.auth_path_for("attempt13"), L.auth_path_for("attempt14")
    assert a != b
    assert a.endswith("governance/authorization.json")


def test_the_lineage_exemption_names_one_file_not_the_directory():
    """The narrowness IS the protection: a grant committed after the sweep
    must still invalidate it, which is what forces grant-then-sweep."""
    from stages.phase_c1.pod_environment import (
        permitted_post_sweep_paths)
    permitted = permitted_post_sweep_paths("attempt13", "1")
    assert len(permitted) == 1, permitted
    assert permitted[0].endswith("governance/readiness.json")
    for p in permitted:
        assert not p.endswith("governance")
        assert not p.endswith("governance/")


def test_a_grant_in_the_same_directory_is_not_exempt():
    """Same directory, different file: the exemption is per path."""
    from stages.phase_c1.pod_environment import (
        permitted_post_sweep_paths)
    L = _L()
    grant = (f"logs/stages/stage-1/phase_c1/runs/attempt13/"
             f"{L.C1_RUN_ROLES['grant']}")
    assert grant not in permitted_post_sweep_paths("attempt13", "1")


def test_the_global_paths_are_pointers_not_records():
    from stages.phase_c1 import pod_environment as pe
    assert pe.RECORD_POINTER == "logs/stages/stage-1/phase_c1/analyses/c1_pod_environment_verification.json"
    #: The alias stays: every pre-2026-09-12 record is at that path.
    assert pe.RECORD_PATH == pe.RECORD_POINTER
    L = _L()
    assert L.auth_path_for(None) == L.AUTH_POINTER
