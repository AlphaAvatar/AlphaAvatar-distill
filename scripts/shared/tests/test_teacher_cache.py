"""The offline teacher lookup asks for the files setup fetched (`shared.pod.teacher_cache`)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import huggingface_hub

from shared.pod import teacher_cache as TC

REPO = Path(__file__).resolve().parents[3]


def test_the_lookup_patterns_are_the_ones_setup_downloads():
    setup = (REPO / "scripts/shared/pod/autoinit_preflight_setup.sh").read_text()
    teacher_step = setup[setup.index("if step_declared TEACHER_READY"):setup.index("mark TEACHER_READY")]
    got = re.search(r"allow_patterns=(\[[^\]]*\])", teacher_step)
    assert got and tuple(ast.literal_eval(got.group(1))) == TC.TEACHER_FILES


def test_the_lookup_is_offline_and_restricted_to_those_patterns(monkeypatch):
    seen = {}
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda model, **kw: seen.update(model=model, **kw) or "/cache/snapshots/rev")
    assert TC.local_snapshot("org/teacher", "rev") == "/cache/snapshots/rev"
    assert seen == {"model": "org/teacher", "revision": "rev", "local_files_only": True,
                    "allow_patterns": list(TC.TEACHER_FILES)}


def test_the_trainers_lineage_lookup_asks_for_setups_files():
    """2026-11-02: train_stage3 looked the teacher up as the whole repository offline, which huggingface_hub 1.x
    refuses on a pod whose setup fetched only TEACHER_FILES; caught at $0 before a_train_a1's training reached it."""
    tree = ast.parse((REPO / "scripts/shared/training/train_stage3.py").read_text())
    called = {getattr(n.func, "id", getattr(n.func, "attr", None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "local_snapshot" in called and "snapshot_download" not in called
