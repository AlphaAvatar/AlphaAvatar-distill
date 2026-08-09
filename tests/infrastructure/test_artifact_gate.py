"""Collection is driven by a declared manifest, and teardown is gated on it.

E6b's bundle listed the three paths E6 needed and never listed
`artifacts/stage3/e6b_*/train_log.jsonl`, because the launcher was derived from
E6, which did not train. tar exited 0, the digest matched, the transfer verified
and the pod was deleted: every check passed, and none of them asked whether
everything that had to survive was present.
"""

import json
import tarfile

import pytest

from aadistill.infrastructure.artifact_gate import (
    ArtifactError, ArtifactManifest, ArtifactSpec, GATE_ORDER, build_manifest,
    create_archive, evaluate_teardown, training_session_specs,
    verify_archive, verify_extracted,
)

SPECS = (
    ArtifactSpec("event_stream", "stage3/*/train_log.jsonl"),
    ArtifactSpec("run_manifest", "stage3/*/run_manifest.json"),
    ArtifactSpec("generations", "audit/*/free.generations.jsonl"),
    ArtifactSpec("checkpoint_hashes", "stage3/*/hashes.txt", required=False),
)


def make_pod(tmp_path, *, arms=("arm_sa", "arm_sb"), with_event_stream=True):
    root = tmp_path / "aad" / "artifacts"
    for arm in arms:
        d = root / "stage3" / arm
        d.mkdir(parents=True)
        if with_event_stream:
            (d / "train_log.jsonl").write_text(
                "\n".join(json.dumps({"event": "train_step", "step": i})
                          for i in range(5)) + "\n")
        (d / "run_manifest.json").write_text(json.dumps({"config_sha256": "x"}))
        a = root / "audit" / arm
        a.mkdir(parents=True)
        (a / "free.generations.jsonl").write_text('{"gen": "hello"}\n')
    return root


def test_a_complete_pod_produces_a_complete_manifest(tmp_path):
    root = make_pod(tmp_path)
    m = build_manifest(root, SPECS, created_utc="2026-08-09T12:00:00Z")
    assert m.ok and not m.missing
    assert len(m.by_class("event_stream")) == 2
    assert len(m.by_class("run_manifest")) == 2
    assert all(e.sha256 and e.size_bytes > 0 for e in m.entries)


def test_the_e6b_omission_is_caught_before_teardown(tmp_path):
    """The whole point: a declared class with no file is a manifest failure."""
    root = make_pod(tmp_path, with_event_stream=False)
    m = build_manifest(root, SPECS)
    assert not m.ok
    classes = {x["artifact_class"] for x in m.missing}
    assert classes == {"event_stream"}
    assert m.missing[0]["matches"] == 0


def test_an_optional_class_missing_is_not_a_failure(tmp_path):
    root = make_pod(tmp_path)
    m = build_manifest(root, SPECS)
    assert m.ok, "hashes.txt is declared optional"
    assert not m.by_class("checkpoint_hashes")


def test_an_empty_required_file_counts_as_missing(tmp_path):
    """A zero-byte train_log.jsonl is not a retrieved event stream."""
    root = make_pod(tmp_path)
    for p in root.glob("stage3/*/train_log.jsonl"):
        p.write_text("")
    m = build_manifest(root, SPECS)
    assert not m.ok
    assert m.missing[0]["usable_matches"] == 0


def test_build_manifest_does_not_raise_on_a_missing_artifact(tmp_path):
    """It runs mid-session on a live pod; the file may still be findable."""
    root = make_pod(tmp_path, with_event_stream=False)
    m = build_manifest(root, SPECS)          # no exception
    assert isinstance(m, ArtifactManifest)


def test_the_archive_is_built_from_the_manifest_not_a_glob(tmp_path):
    root = make_pod(tmp_path)
    m = build_manifest(root, SPECS)
    archive = create_archive(m, tmp_path / "bundle.tar.gz")
    with tarfile.open(archive) as tar:
        names = sorted(x.name for x in tar.getmembers() if x.isfile())
    assert names == sorted(m.paths())
    assert not verify_archive(archive, m)


def test_a_file_that_vanishes_between_manifest_and_archive_is_an_error(tmp_path):
    """Never a silently shorter tarball."""
    root = make_pod(tmp_path)
    m = build_manifest(root, SPECS)
    (root / m.entries[0].path).unlink()
    with pytest.raises(ArtifactError, match="refusing to write a short archive"):
        create_archive(m, tmp_path / "bundle.tar.gz")


def test_verify_archive_detects_a_short_bundle(tmp_path):
    """The E6b bundle, reproduced: it verified because nothing checked coverage."""
    root = make_pod(tmp_path)
    m = build_manifest(root, SPECS)
    short = ArtifactManifest(root=m.root, entries=[
        e for e in m.entries if e.artifact_class != "event_stream"])
    archive = create_archive(short, tmp_path / "short.tar.gz")

    assert not verify_archive(archive, short), "self-consistent, as E6b's was"
    problems = verify_archive(archive, m)
    assert problems and all("missing from archive" in p for p in problems)
    assert any("train_log.jsonl" in p for p in problems)


def test_local_hashes_are_checked_against_the_pod_manifest(tmp_path):
    root = make_pod(tmp_path)
    m = build_manifest(root, SPECS)
    archive = create_archive(m, tmp_path / "bundle.tar.gz")

    local = tmp_path / "store"
    with tarfile.open(archive) as tar:
        tar.extractall(local, filter="data")
    assert not verify_extracted(local, m)

    (local / m.entries[0].path).write_text("corrupted in transit")
    problems = verify_extracted(local, m)
    assert len(problems) == 1 and "hash mismatch" in problems[0]


def test_a_partially_transferred_bundle_is_detected(tmp_path):
    root = make_pod(tmp_path)
    m = build_manifest(root, SPECS)
    local = tmp_path / "store"
    for e in m.entries[:2]:
        p = local / e.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((root / e.path).read_bytes())

    problems = verify_extracted(local, m)
    assert problems and all("not retrieved" in p for p in problems)


def test_an_unreadable_archive_is_reported_not_raised(tmp_path):
    root = make_pod(tmp_path)
    m = build_manifest(root, SPECS)
    bad = tmp_path / "truncated.tar.gz"
    bad.write_bytes(b"\x1f\x8b\x08\x00 not really a tarball")
    assert verify_archive(bad, m)[0].startswith("archive unreadable")


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def all_passed():
    return {name: True for name in GATE_ORDER}


def test_teardown_needs_every_check(tmp_path):
    d = evaluate_teardown(all_passed())
    assert d.allowed and not d.emergency and d.failed_check is None


def test_the_first_failure_is_the_reported_one():
    state = all_passed()
    state["required_files_present"] = False
    state["local_hashes_verified"] = False
    d = evaluate_teardown(state)
    assert not d.allowed
    assert d.failed_check == "required_files_present", (
        "the gate reports the first thing that went wrong, not the last "
        "thing noticed")


def test_an_unreported_check_is_not_a_passed_check():
    d = evaluate_teardown({"training_complete": True})
    assert not d.allowed and d.failed_check == "evaluation_complete"


def test_a_missing_event_stream_blocks_normal_teardown():
    state = all_passed()
    state["required_files_present"] = False
    d = evaluate_teardown(state)
    assert not d.allowed
    assert "must not be deleted" in d.reason


def test_the_cost_watchdog_may_override_but_must_say_why():
    state = all_passed()
    state["archive_contents_verified"] = False
    d = evaluate_teardown(state, emergency_budget=True,
                          emergency_reason="hard limit reached at 545 min")
    assert d.allowed and d.emergency
    assert d.failed_check == "archive_contents_verified"
    assert "LOST" in d.reason and "545 min" in d.reason


def test_an_unexplained_override_is_refused():
    state = all_passed()
    state["transfer_complete"] = False
    with pytest.raises(ArtifactError, match="must record its reason"):
        evaluate_teardown(state, emergency_budget=True)


def test_the_gate_order_matches_the_documented_sequence():
    assert GATE_ORDER == (
        "training_complete", "evaluation_complete", "artifact_manifest_created",
        "required_files_present", "archive_created",
        "archive_contents_verified", "transfer_complete",
        "local_hashes_verified", "checkpoint_hashes_matched",
        "report_inputs_verified")


def test_training_session_specs_always_require_the_event_stream():
    specs = training_session_specs(("stage3/e7_*",))
    classes = {s.artifact_class: s for s in specs}
    assert classes["event_stream"].required is True
    assert classes["event_stream"].pattern == "stage3/e7_*/train_log.jsonl"


# --------------------------------------------------------------------------
# Live-canary regression: a file that is still being appended
# --------------------------------------------------------------------------

def test_a_log_that_grows_between_manifest_and_archive_still_verifies(tmp_path):
    """The 2026-08-09 canary failure, reproduced.

    `train_log.jsonl` was 2,166 bytes when the manifest hashed it and 2,230 when
    tar read it — the job wrote one more event in the gap. The archive then
    could not match the manifest, and the teardown gate blocked forever on
    `archive_contents_verified` with nothing actually wrong.
    """
    root = tmp_path / "artifacts"
    arm = root / "stage3" / "run"
    arm.mkdir(parents=True)
    log = arm / "train_log.jsonl"
    log.write_text("".join(json.dumps({"event": "train_step", "step": i}) + "\n"
                           for i in range(20)))
    specs = (ArtifactSpec("event_stream", "stage3/*/train_log.jsonl"),)

    manifest = build_manifest(root, specs)
    at_manifest = manifest.entries[0].size_bytes

    # The trainer keeps going, exactly as it does on a pod.
    with open(log, "a") as f:
        f.write(json.dumps({"event": "train_step", "step": 20}) + "\n")

    archive = create_archive(manifest, tmp_path / "bundle.tar.gz")
    assert not verify_archive(archive, manifest), (
        "the manifest must describe what was archived, not what was hashed "
        "a moment earlier")
    assert manifest.entries[0].size_bytes > at_manifest
    assert manifest.appended_during_archive == [
        {"path": "stage3/run/train_log.jsonl",
         "manifest_bytes": at_manifest,
         "archived_bytes": manifest.entries[0].size_bytes}], (
        "growth is normal for an append-only log and must be recorded, not "
        "silently absorbed")

    # And the whole gate now passes, which it could not before the fix.
    local = tmp_path / "store"
    with tarfile.open(archive) as tar:
        tar.extractall(local, filter="data")
    assert not verify_extracted(local, manifest)
    assert evaluate_teardown({
        "training_complete": True, "evaluation_complete": True,
        "artifact_manifest_created": True, "required_files_present": manifest.ok,
        "archive_created": True,
        "archive_contents_verified": not verify_archive(archive, manifest),
        "transfer_complete": True,
        "local_hashes_verified": not verify_extracted(local, manifest),
        "checkpoint_hashes_matched": True,
        "report_inputs_verified": True}).allowed


def test_the_archived_hash_is_of_the_bytes_actually_written(tmp_path):
    root = tmp_path / "a"
    root.mkdir()
    f = root / "x.jsonl"
    f.write_bytes(b"one\n")
    manifest = build_manifest(root, (ArtifactSpec("event_stream", "x.jsonl"),))
    with open(f, "ab") as fh:
        fh.write(b"two\n")
    create_archive(manifest, tmp_path / "b.tar.gz")
    import hashlib
    assert manifest.entries[0].sha256 == hashlib.sha256(b"one\ntwo\n").hexdigest()
    assert manifest.entries[0].size_bytes == 8


def test_a_truncated_file_is_an_error_not_a_shorter_entry(tmp_path):
    """Growth is appending; shrinkage is data loss and must stop the archive."""
    root = tmp_path / "a"
    root.mkdir()
    f = root / "x.jsonl"
    f.write_bytes(b"0123456789")
    manifest = build_manifest(root, (ArtifactSpec("event_stream", "x.jsonl"),))
    f.write_bytes(b"012")
    with pytest.raises(ArtifactError, match="shrank"):
        create_archive(manifest, tmp_path / "b.tar.gz")


def test_a_rewritten_manifest_round_trips(tmp_path):
    root = tmp_path / "a"
    root.mkdir()
    (root / "x.jsonl").write_bytes(b"one\n")
    manifest = build_manifest(root, (ArtifactSpec("event_stream", "x.jsonl"),))
    create_archive(manifest, tmp_path / "b.tar.gz")
    path = manifest.write(tmp_path / "manifest.json")
    reloaded = ArtifactManifest.load(path)
    assert reloaded.entries[0].sha256 == manifest.entries[0].sha256
    assert reloaded.appended_during_archive == manifest.appended_during_archive
