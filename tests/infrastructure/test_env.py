"""`code_state` must degrade, never raise.

It is called when a manifest is written — the last step of work that may have
cost hours of paid GPU time. On 2026-07-30 the corpus build generated all 752
prompts and then lost its manifest because `git` is absent from vLLM's official
image, so the failure mode this pins is one that has actually happened.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.env import code_state  # noqa: E402


def test_code_state_reports_the_commit_when_git_works():
    state = code_state(str(REPO_ROOT))
    assert state["git_commit_source"] == "git"
    assert len(state["git_commit"]) == 40
    assert isinstance(state["dirty"], bool)


def test_code_state_survives_a_missing_git(monkeypatch, tmp_path):
    """No git binary: record that fact rather than taking down the manifest."""
    monkeypatch.setenv("PATH", str(tmp_path))       # nothing executable here
    monkeypatch.delenv("AADISTILL_CODE_COMMIT", raising=False)
    state = code_state(str(REPO_ROOT))
    assert state["git_commit"] is None
    assert state["git_commit_source"] == "unavailable"
    assert state["code_state_error"]


def test_code_state_takes_the_commit_from_the_environment_as_a_fallback():
    """A git-less image can still say which commit it was built from, so the
    manifest stays reproducible where the caller actually knows the answer."""
    import os

    commit = "a" * 40
    old_path, old_commit = os.environ.get("PATH"), os.environ.get("AADISTILL_CODE_COMMIT")
    try:
        os.environ["PATH"] = "/nonexistent-for-this-test"
        os.environ["AADISTILL_CODE_COMMIT"] = commit
        state = code_state(str(REPO_ROOT))
    finally:
        if old_path is not None:
            os.environ["PATH"] = old_path
        if old_commit is None:
            os.environ.pop("AADISTILL_CODE_COMMIT", None)
        else:
            os.environ["AADISTILL_CODE_COMMIT"] = old_commit
    assert state["git_commit"] == commit
    assert state["git_commit_source"] == "env:AADISTILL_CODE_COMMIT"


def test_code_state_never_guesses_a_commit(monkeypatch, tmp_path):
    """A manifest that invents its code state is worse than one that admits it
    does not know (P4/P14)."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("AADISTILL_CODE_COMMIT", raising=False)
    state = code_state(str(REPO_ROOT))
    assert state["git_commit"] is None
    assert state["uncommitted_state_sha256"] is None
    assert state["dirty"] is None
