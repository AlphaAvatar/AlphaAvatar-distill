"""The E7 streams: their identity, their disjointness, and their teardown gate.

Three things are checked here that the trainer cannot check for itself: that the
built streams really are what the preregistration says, that the leakage proof
fails closed rather than reporting a vacuous pass, and that a session which
loses its structured training log cannot be torn down as if it had succeeded.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
FINEWEB = REPO / "artifacts/stage3/e7_fineweb_kd"
CONTROL = REPO / "artifacts/stage3/e7_control_kd"
VAL = REPO / "artifacts/stage3/e7_fineweb_val"
built = pytest.mark.skipif(
    not (FINEWEB / "manifest.json").is_file(),
    reason="E7 streams are gitignored artifacts; rebuild with the builders")


def run(args, **kw):
    import os
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run([PY, *args], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=300, **kw)


# --------------------------------------------------------------------------
# stream identity
# --------------------------------------------------------------------------

@built
def test_the_two_training_streams_are_budget_matched():
    """Arm C exists to be identical to B except in content. Check the except."""
    b = json.loads((FINEWEB / "manifest.json").read_text())
    c = json.loads((CONTROL / "manifest.json").read_text())
    for key in ("n_blocks", "block_len", "kd_positions", "total_tokens",
                "padding_tokens"):
        assert b[key] == c[key], f"{key} differs: {b[key]} vs {c[key]}"
    assert b["kd_positions"] == 1_801_503
    assert b["padding_tokens"] == 0
    assert b["kind"] == "general_text_kd"
    assert c["kind"] == "in_domain_kd_control"
    assert b["outputs"]["blocks"] != c["outputs"]["blocks"], (
        "identical contents would make the control meaningless")


@built
def test_the_fineweb_stream_is_raw_text_with_no_chat_wrapping():
    m = json.loads((FINEWEB / "manifest.json").read_text())
    assert m["chat_template_applied"] is False
    assert m["assistant_ce_positions"] == 0
    assert m["source"]["dataset"] == "HuggingFaceFW/fineweb-edu"
    assert m["source"]["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert m["source"]["config"] == "sample-10BT"
    assert m["boundary_policy"]["separator_token"] == "<|endoftext|>"


@built
def test_the_fineweb_stream_avoids_every_reserved_index_range():
    """warmup_v1 read from index 0; holdout_v1 skipped 5,000 and took 40."""
    for d in (FINEWEB, VAL):
        rng = json.loads((d / "manifest.json").read_text())["source"]["index_range"]
        assert rng[0] >= 20000, f"{d.name} starts at {rng[0]}, inside the "\
                                "historical prefix"
    train = json.loads((FINEWEB / "manifest.json").read_text())["source"]
    val = json.loads((VAL / "manifest.json").read_text())["source"]
    assert train["index_range"][0] >= val["index_range"][1] or \
        val["index_range"][0] >= train["index_range"][1], \
        "train and validation index ranges overlap"


@built
def test_the_control_stream_avoids_the_trained_rung_and_the_validation_tail():
    src = json.loads((CONTROL / "manifest.json").read_text())["source"]
    used = src["block_range_used"]
    assert used[0] >= src["excluded_trained_blocks"][1], (
        "the control would replay blocks the 1.60M rung already trains on")
    assert used[1] <= src["excluded_validation_tail"][0], (
        "the control would train on blocks every rung validates on")
    assert src["trained_rung"] == 1600000


@built
def test_every_stream_block_is_full():
    import numpy as np
    for d in (FINEWEB, CONTROL, VAL):
        m = json.loads((d / "manifest.json").read_text())
        z = np.load(d / "blocks.npz")
        assert z["input_ids"].shape == (m["n_blocks"], m["block_len"])
        assert bool(z["content_mask"].all()), f"{d.name} has padding"
        assert m["kd_positions"] == m["n_blocks"] * (m["block_len"] - 1)


# --------------------------------------------------------------------------
# the leakage proof fails closed
# --------------------------------------------------------------------------

def test_disjointness_refuses_to_run_with_no_streams():
    r = run(["scripts/data/check_stream_disjointness.py"])
    assert r.returncode != 0
    assert "nothing would be checked" in (r.stdout + r.stderr)


def test_disjointness_errors_on_a_missing_reserved_file(tmp_path):
    """A proof that silently skips its inputs proves nothing."""
    stream = tmp_path / "s"
    stream.mkdir()
    (stream / "manifest.json").write_text(json.dumps({"n_blocks": 1}))
    (stream / "docs.jsonl").write_text(json.dumps({"sha256": "a" * 64}) + "\n")
    r = run(["scripts/data/check_stream_disjointness.py", "--stream", str(stream),
             "--reserved", str(tmp_path / "does_not_exist.jsonl")])
    assert r.returncode != 0
    assert "FileNotFoundError" in r.stderr or "missing" in r.stderr


def test_disjointness_errors_when_a_stream_has_no_docs(tmp_path):
    stream = tmp_path / "s"
    stream.mkdir()
    (stream / "manifest.json").write_text(json.dumps({"n_blocks": 1}))
    r = run(["scripts/data/check_stream_disjointness.py", "--stream", str(stream)])
    assert r.returncode != 0
    assert "docs.jsonl" in r.stderr


def test_disjointness_detects_a_planted_overlap(tmp_path):
    """Verified to bite: two streams sharing one document must fail."""
    shared = "b" * 64
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"n_blocks": 1}))
        (d / "docs.jsonl").write_text(json.dumps({"sha256": shared}) + "\n")
    r = run(["scripts/data/check_stream_disjointness.py",
             "--stream", str(tmp_path / "a"), "--stream", str(tmp_path / "b")])
    assert r.returncode == 6
    assert "LEAKAGE" in r.stderr
    report = json.loads(r.stdout)
    assert report["disjoint"] is False
    assert report["content_hash_overlaps"][0]["n_shared"] == 1


def test_an_overlap_between_reserved_artifacts_is_reported_but_not_fatal(tmp_path):
    """It is a fact about those artifacts, not something E7 can fix by stopping.

    This is not hypothetical: `rag.jsonl` and `answerability_paired.jsonl` in
    `capability-v2` share one SQuAD item whose ids differ only by a pair prefix.
    """
    stream = tmp_path / "s"
    stream.mkdir()
    (stream / "manifest.json").write_text(json.dumps({"n_blocks": 1}))
    (stream / "docs.jsonl").write_text(json.dumps({"sha256": "c" * 64}) + "\n")
    for name in ("r1.jsonl", "r2.jsonl"):
        (tmp_path / name).write_text(json.dumps({"text": "identical text"}) + "\n")
    r = run(["scripts/data/check_stream_disjointness.py",
             "--stream", str(stream),
             "--reserved", str(tmp_path / "r1.jsonl"),
             "--reserved", str(tmp_path / "r2.jsonl")])
    assert r.returncode == 0
    report = json.loads(r.stdout)
    assert report["disjoint"] is True
    assert len(report["reserved_vs_reserved_overlaps"]) == 1
    assert "not involving any E7 stream" in r.stderr


@built
def test_the_shipped_disjointness_proof_covers_every_stream_and_passes():
    proof = json.loads((REPO / "artifacts/stage3/e7_disjointness.json").read_text())
    assert proof["disjoint"] is True
    assert proof["content_hash_overlaps"] == []
    for name in ("e7_fineweb_kd", "e7_control_kd", "e7_fineweb_val"):
        assert name in proof["groups"]
    for reserved in ("reserved:holdout_v1.jsonl", "reserved:warmup_v1.jsonl",
                     "reserved:prompts.jsonl"):
        assert reserved in proof["groups"]


# --------------------------------------------------------------------------
# the teardown gate knows what E7 must bring home
# --------------------------------------------------------------------------

def test_the_e7_artifact_spec_requires_the_structured_training_stream():
    spec = json.loads((REPO / "configs/stage3/e7/artifacts.json").read_text())
    classes = {s["artifact_class"]: s for s in spec}
    assert classes["event_stream"]["required"] is True
    assert "train_log.jsonl" in classes["event_stream"]["pattern"]
    assert classes["run_manifest"]["required"] is True
    # The treatment's identity must come home too, or the run is unreproducible.
    assert classes["extra_stream_manifest"]["required"] is True
    assert classes["disjointness_proof"]["required"] is True


def test_a_missing_training_stream_blocks_teardown(tmp_path):
    """E6b's loss, expressed against E7's own spec."""
    from aadistill.infrastructure.artifact_gate import (
        ArtifactSpec, build_manifest, evaluate_teardown,
    )
    spec = json.loads((REPO / "configs/stage3/e7/artifacts.json").read_text())
    specs = tuple(ArtifactSpec(**s) for s in spec)

    root = tmp_path / "artifacts"
    arm = root / "stage3" / "e7_fineweb_r1600k_sa"
    arm.mkdir(parents=True)
    (arm / "run_manifest.json").write_text("x" * 600)      # present
    # train_log.jsonl deliberately absent, exactly as in E6b.

    manifest = build_manifest(root, specs)
    assert not manifest.ok
    missing = {m["artifact_class"] for m in manifest.missing}
    assert "event_stream" in missing

    decision = evaluate_teardown({
        "training_complete": True, "evaluation_complete": True,
        "artifact_manifest_created": True,
        "required_files_present": manifest.ok})
    assert decision.allowed is False
    assert decision.failed_check == "required_files_present"


def test_an_undersized_training_stream_counts_as_missing(tmp_path):
    from aadistill.infrastructure.artifact_gate import ArtifactSpec, build_manifest
    spec = json.loads((REPO / "configs/stage3/e7/artifacts.json").read_text())
    specs = tuple(ArtifactSpec(**s) for s in spec
                  if s["artifact_class"] == "event_stream")
    root = tmp_path / "artifacts"
    arm = root / "stage3" / "e7_control_r1600k_sb"
    arm.mkdir(parents=True)
    (arm / "train_log.jsonl").write_text("{}\n")           # far under min_bytes
    manifest = build_manifest(root, specs)
    assert not manifest.ok
    assert manifest.missing[0]["usable_matches"] == 0
