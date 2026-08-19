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
        "required_files_present", "final_streams_quiescent", "archive_created",
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
    """The 2026-08-09 canary failure, reproduced — as a **mutable snapshot**.

    `train_log.jsonl` was 2,166 bytes when the manifest hashed it and 2,230 when
    tar read it — the job wrote one more event in the gap — so the archive could
    not match the manifest and the gate blocked with nothing actually wrong.

    The bounded read fixes that *for a snapshot*, which is what this is: the
    writer is still active, the archive records the captured boundary, and the
    claim is "these bytes are durable" rather than "this file is finished". The
    same growth under `final_required` is refused
    (`test_a_growing_snapshot_archives_but_a_growing_final_does_not`).
    """
    root = tmp_path / "artifacts"
    arm = root / "stage3" / "run"
    arm.mkdir(parents=True)
    log = arm / "train_log.jsonl"
    log.write_text("".join(json.dumps({"event": "train_step", "step": i}) + "\n"
                           for i in range(20)))
    specs = (ArtifactSpec("event_stream", "stage3/*/train_log.jsonl",
                          lifecycle="mutable_snapshot"),)

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
        "final_streams_quiescent": manifest.final_streams_quiescent,
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
    manifest = build_manifest(root, (ArtifactSpec(
        "event_stream", "x.jsonl", lifecycle="mutable_snapshot"),))
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


# --------------------------------------------------------------------------
# mutable_snapshot vs final_required
# --------------------------------------------------------------------------

from aadistill.infrastructure.artifact_gate import (  # noqa: E402
    CompletionMarker, LIFECYCLES,
)

DONE = (CompletionMarker("run.status", "MARKER:ALL_DONE"),)


def growing_pod(tmp_path, *, finished: bool):
    root = tmp_path / "artifacts"
    arm = root / "stage3" / "run"
    arm.mkdir(parents=True)
    (arm / "train_log.jsonl").write_text(
        "".join(json.dumps({"event": "train_step", "step": i}) + "\n"
                for i in range(30)))
    (root / "run.status").write_text(
        "MARKER:TRAIN_DONE\nMARKER:ALL_DONE\n" if finished
        else "MARKER:TRAIN_DONE\n")
    return root


def test_final_required_is_the_default_so_a_thoughtless_spec_is_strict():
    assert ArtifactSpec("c", "p").lifecycle == "final_required"
    assert LIFECYCLES[0] == "final_required"


def test_an_unknown_lifecycle_is_refused():
    with pytest.raises(ArtifactError, match="lifecycle must be one of"):
        ArtifactSpec("c", "p", lifecycle="whenever")


def test_a_growing_final_required_stream_is_not_complete(tmp_path):
    """The correction: archivable is not the same as finished."""
    root = growing_pod(tmp_path, finished=True)
    log = root / "stage3" / "run" / "train_log.jsonl"
    specs = (ArtifactSpec("event_stream", "stage3/*/train_log.jsonl"),)

    # `settle_seconds` observes the file across a window; the writer appends
    # inside it, exactly as a trainer does.
    def writer(_seconds):
        with open(log, "a") as f:
            f.write(json.dumps({"event": "train_step", "step": 30}) + "\n")

    m = build_manifest(root, specs, settle_seconds=0.01,
                       completion_markers=DONE, sleep=writer)
    assert not m.ok
    assert m.final_streams_quiescent is False
    assert m.still_being_written == ["stage3/run/train_log.jsonl"]
    assert "still being written" in m.missing[0]["reason"]
    assert "bounded prefix" in m.missing[0]["reason"]


def test_a_quiescent_marker_backed_stream_is_complete(tmp_path):
    root = growing_pod(tmp_path, finished=True)
    specs = (ArtifactSpec("event_stream", "stage3/*/train_log.jsonl"),)
    m = build_manifest(root, specs, settle_seconds=0.01,
                       completion_markers=DONE)
    assert m.ok and m.final_streams_quiescent
    assert m.final_entries() and not m.snapshot_entries()
    assert m.entries[0].lifecycle == "final_required"


def test_a_missing_completion_marker_blocks_final_classes(tmp_path):
    """A quiescent file whose producer never said it finished is not final —
    a crashed trainer leaves exactly that."""
    root = growing_pod(tmp_path, finished=False)
    specs = (ArtifactSpec("event_stream", "stage3/*/train_log.jsonl"),)
    m = build_manifest(root, specs, settle_seconds=0.01,
                       completion_markers=DONE)
    assert not m.ok and m.final_streams_quiescent is False
    assert m.completion_marker_failures
    assert "not signalled completion" in m.missing[0]["reason"]


def test_a_mutable_snapshot_ignores_markers_and_growth(tmp_path):
    """Its claim is 'these bytes are durable', not 'this file is finished'."""
    root = growing_pod(tmp_path, finished=False)
    specs = (ArtifactSpec("event_stream", "stage3/*/train_log.jsonl",
                          lifecycle="mutable_snapshot"),)
    m = build_manifest(root, specs, settle_seconds=0.01,
                       completion_markers=DONE)
    assert m.ok
    assert m.snapshot_entries() and not m.final_entries()
    # It makes no completeness claim, so it cannot make the gate's claim either.
    assert m.final_streams_quiescent is True, (
        "no final_required class is declared, so there is nothing to be "
        "non-quiescent about")


def test_a_growing_snapshot_archives_but_a_growing_final_does_not(tmp_path):
    root = growing_pod(tmp_path, finished=True)
    log = root / "stage3" / "run" / "train_log.jsonl"

    snap = build_manifest(root, (ArtifactSpec(
        "event_stream", "stage3/*/train_log.jsonl",
        lifecycle="mutable_snapshot"),))
    with open(log, "a") as f:
        f.write(json.dumps({"event": "train_step", "step": 99}) + "\n")
    create_archive(snap, tmp_path / "snap.tar.gz")          # allowed
    assert snap.appended_during_archive

    final = build_manifest(root, (ArtifactSpec(
        "event_stream", "stage3/*/train_log.jsonl"),),
        completion_markers=DONE)
    with open(log, "a") as f:
        f.write(json.dumps({"event": "train_step", "step": 100}) + "\n")
    with pytest.raises(ArtifactError, match="declared final_required and grew"):
        create_archive(final, tmp_path / "final.tar.gz")


def test_normal_teardown_refuses_a_non_quiescent_stream():
    state = {name: True for name in GATE_ORDER}
    state["final_streams_quiescent"] = False
    d = evaluate_teardown(state)
    assert not d.allowed
    assert d.failed_check == "final_streams_quiescent"
    assert "mutable_snapshot" in d.reason and "bounded prefix" in d.reason
    assert d.incomplete_event_streams == ()


def test_emergency_teardown_may_keep_a_snapshot_but_must_name_the_loss():
    state = {name: True for name in GATE_ORDER}
    state["final_streams_quiescent"] = False
    d = evaluate_teardown(
        state, emergency_budget=True,
        emergency_reason="hard limit reached at 545 min",
        incomplete_event_streams=("stage3/e7_fineweb_r1600k_sa/train_log.jsonl",))
    assert d.allowed and d.emergency
    assert d.incomplete_event_streams == (
        "stage3/e7_fineweb_r1600k_sa/train_log.jsonl",)
    assert "THE FINAL EVENT STREAM IS INCOMPLETE" in d.reason
    assert "tail is lost" in d.reason


def test_an_emergency_that_truncates_a_stream_without_naming_it_is_refused():
    state = {name: True for name in GATE_ORDER}
    state["final_streams_quiescent"] = False
    with pytest.raises(ArtifactError, match="must name the streams"):
        evaluate_teardown(state, emergency_budget=True,
                          emergency_reason="hard limit")


def test_the_two_lifecycles_survive_a_manifest_round_trip(tmp_path):
    root = growing_pod(tmp_path, finished=True)
    m = build_manifest(root, (
        ArtifactSpec("event_stream", "stage3/*/train_log.jsonl"),
        ArtifactSpec("status", "run.status", lifecycle="mutable_snapshot"),
    ), completion_markers=DONE)
    path = m.write(tmp_path / "manifest.json")
    back = ArtifactManifest.load(path)
    assert {e.path: e.lifecycle for e in back.entries} == {
        "stage3/run/train_log.jsonl": "final_required",
        "run.status": "mutable_snapshot"}
    assert back.completion_markers == [
        {"path": "run.status", "contains": "MARKER:ALL_DONE"}]
    assert json.loads(path.read_text())["final_streams_quiescent"] is True


# --- the three quiescence failures are not one failure ----------------------
#
# `final_streams_quiescent` is false when a producer has not signalled
# completion, OR a file is still growing, OR a `final_required` class is simply
# missing. Only the first two truncate a stream. The bounded measurement of
# 2026-08-19 declares no streams at all; its driver crashed before writing its
# one report, quiescence failed for the third reason, and the emergency path
# demanded it name streams it did not have — so the session took the
# launcher-error route instead of a clean incomplete collection.

def _failing_state() -> dict:
    """Every check passed except quiescence, which is the case under test."""
    return {name: True for name in GATE_ORDER} | {"final_streams_quiescent": False}


def test_a_session_with_no_event_streams_tears_down_cleanly_when_its_report_is_missing():
    """Case 1. Nothing is truncated, so nothing has to be named."""
    d = evaluate_teardown(
        _failing_state(), emergency_budget=True,
        emergency_reason="the driver died before writing its only report",
        incomplete_event_streams=(),
        streams_at_risk=())                 # the manifest says: no stream at risk
    assert d.allowed and d.emergency
    assert d.failed_check == "final_streams_quiescent"
    assert d.incomplete_event_streams == ()
    assert "No event stream was truncated" in d.reason
    assert "THE FINAL EVENT STREAM IS INCOMPLETE" not in d.reason


def test_a_stream_producing_session_that_does_not_name_its_loss_is_refused():
    """Case 2, unchanged: the protection this whole rule exists for."""
    with pytest.raises(ArtifactError, match="must name the streams it is truncating"):
        evaluate_teardown(
            _failing_state(), emergency_budget=True,
            emergency_reason="the cost watchdog fired mid-training",
            incomplete_event_streams=(),
            streams_at_risk=("train_log.jsonl",))


def test_a_stream_producing_session_that_names_its_loss_is_allowed_and_records_it():
    """Case 3: permitted, and the loss is in the record rather than inferred."""
    d = evaluate_teardown(
        _failing_state(), emergency_budget=True,
        emergency_reason="the cost watchdog fired mid-training",
        incomplete_event_streams=("train_log.jsonl",),
        streams_at_risk=("train_log.jsonl",))
    assert d.allowed and d.emergency
    assert d.incomplete_event_streams == ("train_log.jsonl",)
    assert "THE FINAL EVENT STREAM IS INCOMPLETE" in d.reason
    assert "train_log.jsonl" in d.reason


def test_a_caller_supplying_no_evidence_still_gets_the_strict_rule():
    """`streams_at_risk=None` is "I do not know", and an uninformed caller does
    not get the weaker rule. Every pre-existing caller is unaffected."""
    with pytest.raises(ArtifactError, match="must name the streams it is truncating"):
        evaluate_teardown(
            _failing_state(), emergency_budget=True,
            emergency_reason="something went wrong",
            incomplete_event_streams=())


def test_the_normal_path_still_blocks_on_a_live_stream():
    """None of this weakens the non-emergency gate: a growing stream still stops
    a normal teardown, whatever the evidence says."""
    d = evaluate_teardown(_failing_state(), streams_at_risk=())
    assert not d.allowed and not d.emergency
    assert d.failed_check == "final_streams_quiescent"
    assert "must not be accepted as the final one" in d.reason
