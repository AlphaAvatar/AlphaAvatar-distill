"""Where `battery_render` looks for the pinned snapshots, and when it decides.

Until 2026-09-04 the answer was "`~/.cache/huggingface/hub`, decided at import".
A pod exports `HF_HOME` and has nothing under `$HOME`, so the seven renderer-parity
cases passed on the dev box and could never pass on a pod. C1 attempt 3R paid
`$0.3482` to find that out at the setup test gate, having already cleared
`VLLM_READY → TEACHER_READY → ROPE_OK`.

These tests pin the two properties that fix cost: the precedence, and that it is
evaluated per call rather than frozen at import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/data"))

import battery_render as br  # noqa: E402


# --- cache resolution -------------------------------------------------------

def test_hf_hub_cache_wins_outright(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert br.hub_cache() == tmp_path / "explicit"


def test_hf_home_hub_is_the_fallback(monkeypatch, tmp_path):
    """The pod's case: it exports `HF_HOME` and never `HF_HUB_CACHE`."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert br.hub_cache() == tmp_path / "hfhome" / "hub"


def test_home_is_the_last_resort(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert br.hub_cache() == tmp_path / "home" / ".cache/huggingface/hub"


def test_an_empty_variable_does_not_count_as_a_setting(monkeypatch, tmp_path):
    """`HF_HUB_CACHE=` exports an empty string. Treating that as a path would
    resolve every snapshot against the process's working directory."""
    monkeypatch.setenv("HF_HUB_CACHE", "   ")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert br.hub_cache() == tmp_path / "hfhome" / "hub"


def test_resolution_happens_at_call_time_not_import_time(monkeypatch, tmp_path):
    """The defect was a module-level constant. A resolver that cached its answer
    on first use would reintroduce it for any process that imports before it
    configures — which is every test in this suite."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "first"))
    first = br.hub_cache()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "second"))
    second = br.hub_cache()
    assert first == tmp_path / "first"
    assert second == tmp_path / "second"


def test_the_module_exposes_no_frozen_hub_constant():
    """A surviving `HUB` would be a second, unmonitored way to reach the cache."""
    assert not hasattr(br, "HUB")


def test_snapshot_path_reports_where_it_looked(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    p = br.snapshot_path("openai/gsm8k", "deadbeef")
    assert p == tmp_path / "hub/datasets--openai--gsm8k/snapshots/deadbeef"


def test_snapshot_names_the_resolved_path_when_it_is_absent(monkeypatch, tmp_path):
    """The error a pod sees must say where it looked, or the next diagnosis is
    another guess about somebody else's environment."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    with pytest.raises(FileNotFoundError) as exc:
        br.snapshot("openai/gsm8k", "deadbeef")
    assert str(tmp_path / "hub") in str(exc.value)
    assert "openai/gsm8k" in str(exc.value) and "deadbeef" in str(exc.value)


# --- the parity detector itself ---------------------------------------------
#
# `check_group_parity` is the shared implementation behind both the pytest cases
# and the pre-provider gate. If it cannot see a renderer change, everything built
# on it is decoration. Driven against a synthetic snapshot and a synthetic frozen
# battery so it runs anywhere, including on a pod with no datasets at all.

def _synthetic_source(monkeypatch, tmp_path, frozen_prompt: str, index: int = 0):
    """A one-row gsm8k snapshot plus a frozen battery that quotes it.

    `index` decides the frozen item's id — `gsm8k-test-{index:05d}` — so a value
    the one-row source cannot produce models a frozen item that no longer has any
    source row at all, which is a different defect from a changed rendering.
    """
    repo, rev, rel = br.FROZEN_SOURCES["gsm8k"]
    snap = (tmp_path / "hub" / f"datasets--{repo.replace('/', '--')}"
            / "snapshots" / rev / rel).parent
    snap.mkdir(parents=True)
    src = snap / Path(rel).name
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"question": ["what is 2+2?"],
                             "answer": ["two plus two #### 4"]}), src)

    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    item = br.make_gsm8k({"_index": index, "question": frozen_prompt,
                          "answer": "two plus two #### 4"})
    item["prompt_sha256"] = br.content_sha256(br.norm(item["prompt_text"]))
    (frozen_dir / "gsm8k.jsonl").write_text(json.dumps(item) + "\n")

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(br, "FROZEN_BATTERY", frozen_dir)


def test_the_detector_passes_when_the_rendering_agrees(monkeypatch, tmp_path):
    _synthetic_source(monkeypatch, tmp_path, "what is 2+2?")
    result = br.check_group_parity("gsm8k")
    assert result["status"] == "PASS", result
    assert result["n_checked"] == 1 and result["mismatches"] == []


def test_mutating_the_renderer_breaks_parity(monkeypatch, tmp_path):
    """The mutation test. A renderer that decorates the prompt must be caught."""
    _synthetic_source(monkeypatch, tmp_path, "what is 2+2?")
    mutated = dict(br.RENDERERS)
    mutated["gsm8k"] = lambda r: dict(br.make_gsm8k(r),
                                      prompt_text=r["question"] + " (please)")
    monkeypatch.setattr(br, "RENDERERS", mutated)
    result = br.check_group_parity("gsm8k")
    assert result["status"] == "FAIL"
    assert result["mismatches"][0]["field"] == "prompt_text"


def test_a_frozen_item_that_cannot_be_re_rendered_is_a_failure(monkeypatch,
                                                               tmp_path):
    """Not a silent pass over the ones that happened to match: the frozen item
    quotes a question the source no longer contains, so it is never re-rendered
    and the group must fail on coverage rather than on comparison."""
    _synthetic_source(monkeypatch, tmp_path, "what is 2+2?", index=7)
    result = br.check_group_parity("gsm8k")
    assert result["status"] == "FAIL"
    assert result["missing"] == ["gsm8k-test-00007"] and result["n_checked"] == 0
    assert result["mismatches"] == []       # coverage, not comparison


def test_an_absent_source_is_reported_as_absent_not_as_a_pass(monkeypatch,
                                                              tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    result = br.check_group_parity("gsm8k")
    assert result["status"] == "SOURCE_ABSENT"
    assert result["n_checked"] == 0
    assert str(tmp_path / "empty-hub") in result["resolved_snapshot"]
