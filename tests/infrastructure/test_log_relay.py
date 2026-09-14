"""Already-emitted events must survive the pod that emitted them.

E6b's `train_log.jsonl` was written correctly for nine hours and destroyed at
teardown, because it existed only on the pod and the one retrieval step did not
list it. The relay's guarantee is that whatever has already been synced is
already durable — so a pod that disappears mid-run takes nothing with it except
the events written since the last cycle.
"""

import json
import shutil

from aadistill.infrastructure.log_relay import LogRelay, RelaySpec
from aadistill.infrastructure.remote import CommandResult, LocalShellTarget


def emit(path, n, start=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for i in range(start, start + n):
            f.write(json.dumps({"event": "train_step", "step": i,
                                "loss": 1.0 / (i + 1)}) + "\n")


def make_relay(tmp_path, required=True):
    pod = tmp_path / "pod"
    pod.mkdir(exist_ok=True)
    target = LocalShellTarget(pod)
    spec = RelaySpec(remote_path=str(pod / "run" / "train_log.jsonl"),
                     local_name="train_log.jsonl", required=required)
    relay = LogRelay(target, (spec,), tmp_path / "durable")
    return pod, relay, spec


def test_events_sync_incrementally_without_duplicating(tmp_path):
    pod, relay, spec = make_relay(tmp_path)
    emit(pod / "run" / "train_log.jsonl", 10)

    first = relay.sync_once()
    assert first.ok and first.synced_bytes[spec.remote_path] > 0
    assert len(relay.recovered_events(spec)) == 10

    # Nothing new: a second cycle must be a no-op, not a re-copy.
    second = relay.sync_once()
    assert second.synced_bytes[spec.remote_path] == 0
    assert len(relay.recovered_events(spec)) == 10

    emit(pod / "run" / "train_log.jsonl", 5, start=10)
    third = relay.sync_once()
    assert third.synced_bytes[spec.remote_path] > 0
    events = relay.recovered_events(spec)
    assert [e["step"] for e in events] == list(range(15))


def test_the_pod_can_disappear_mid_run_and_the_events_remain(tmp_path):
    """The load-bearing property."""
    pod, relay, spec = make_relay(tmp_path)
    emit(pod / "run" / "train_log.jsonl", 291)      # one E6b arm at log_every=10
    assert relay.sync_once().ok

    shutil.rmtree(pod)                              # the pod is deleted

    events = relay.recovered_events(spec)
    assert len(events) == 291
    assert events[0]["step"] == 0 and events[-1]["step"] == 290
    assert all(e["event"] == "train_step" for e in events)


def test_a_vanished_pod_is_reported_not_raised(tmp_path):
    """The relay runs inside the loop that tears the pod down. It may not die."""
    pod, relay, spec = make_relay(tmp_path)
    emit(pod / "run" / "train_log.jsonl", 4)
    relay.sync_once()
    shutil.rmtree(pod)

    result = relay.sync_once()                      # must not raise
    assert not result.ok
    assert spec.remote_path in result.errors
    assert len(relay.recovered_events(spec)) == 4


def test_a_broken_transport_is_reported_not_raised(tmp_path):
    class ExplodingTarget:
        def run(self, command, *, timeout):
            raise OSError("ssh: connection reset by peer")

    spec = RelaySpec(remote_path="/workspace/train_log.jsonl",
                     local_name="train_log.jsonl")
    relay = LogRelay(ExplodingTarget(), (spec,), tmp_path / "durable")
    result = relay.sync_once()
    assert not result.ok
    assert "OSError" in result.errors[spec.remote_path]


def test_a_timed_out_read_is_reported_not_raised(tmp_path):
    class HangingTarget:
        def run(self, command, *, timeout):
            return CommandResult(124, "", "", timed_out=True)

    spec = RelaySpec(remote_path="/workspace/train_log.jsonl",
                     local_name="train_log.jsonl")
    relay = LogRelay(HangingTarget(), (spec,), tmp_path / "durable")
    result = relay.sync_once()
    assert "timed out" in result.errors[spec.remote_path]


def test_a_partial_trailing_line_does_not_discard_the_rest(tmp_path):
    """A cycle can land between `write` and newline. Keep the complete events."""
    pod, relay, spec = make_relay(tmp_path)
    log = pod / "run" / "train_log.jsonl"
    emit(log, 3)
    with open(log, "a") as f:
        f.write('{"event": "train_step", "step": 3, "los')   # torn mid-write
    relay.sync_once()

    events = relay.recovered_events(spec)
    assert [e["step"] for e in events] == [0, 1, 2]


def test_offsets_survive_a_new_relay_object(tmp_path):
    """The poller is restartable; a restart must not re-copy nine hours."""
    pod, relay, spec = make_relay(tmp_path)
    emit(pod / "run" / "train_log.jsonl", 6)
    relay.sync_once()

    fresh = LogRelay(LocalShellTarget(pod), (spec,), tmp_path / "durable")
    assert fresh.sync_once().synced_bytes[spec.remote_path] == 0
    assert len(fresh.recovered_events(spec)) == 6


def test_a_corrupt_offset_file_re_syncs_rather_than_wedging(tmp_path):
    pod, relay, spec = make_relay(tmp_path)
    emit(pod / "run" / "train_log.jsonl", 2)
    relay.sync_once()
    relay.state_path.write_text("{not json")

    result = relay.sync_once()
    assert result.ok, "a corrupt offset file must not stop the relay"


def test_an_optional_missing_file_is_not_an_error(tmp_path):
    pod, relay, _ = make_relay(tmp_path, required=False)
    result = relay.sync_once()
    assert result.ok


# --- a rewritten document is replaced, never appended to ---------------------
#
# C1 attempt 18's `c1_evidence.json` came home unparseable. The driver rewrites
# that file on every state change; the relay had synced 11,343 bytes of an early
# version, and after the rewrite `tail -c +11344` appended the NEW document's
# tail to the OLD document's head. The local copy was exactly the right size and
# was not JSON. The artifact-store copy was intact, so the verdict was never in
# doubt — but the primary evidence file was lost, and one copy is not two.

def test_a_rewritten_file_ends_up_as_exactly_the_new_bytes(tmp_path):
    """Old file LONGER than new: the result is the new document, and it parses."""
    pod = tmp_path / "pod"
    pod.mkdir()
    remote = pod / "run" / "evidence.json"
    remote.parent.mkdir(parents=True)

    spec = RelaySpec(remote_path=str(remote), local_name="evidence.json",
                     required=False, whole_file=True)
    relay = LogRelay(LocalShellTarget(pod), (spec,), tmp_path / "durable")
    local = relay.local_path(spec)

    long_first = json.dumps({"stage": "G", "probes": list(range(400))}, indent=1)
    remote.write_text(long_first)
    assert relay.sync_once().ok
    assert local.read_text() == long_first

    #: The rewrite the driver performs: a SHORTER document replacing a longer.
    short_after = json.dumps({"stage": "I", "verdict": "GO"}, indent=1)
    assert len(short_after) < len(long_first)
    remote.write_text(short_after)

    assert relay.sync_once().ok
    assert local.read_text() == short_after, "the old tail survived the rewrite"
    assert json.loads(local.read_text())["verdict"] == "GO"
    assert not list(local.parent.glob("*.partial")), "a temp file was left behind"


def test_a_stored_offset_cannot_make_a_whole_file_spec_read_from_the_middle(tmp_path):
    """The persisted offset is IGNORED for a whole-file spec, not just unset.

    Found by mutation: reverting `start` to `offsets.get(key, 0)` left all the
    other tests green, because a whole-file spec stores 0 and so reads 0 anyway.
    That makes the guard look redundant, and it is not — the offsets file
    outlives the code that wrote it. An offsets file written by the relay AS IT
    RAN IN ATTEMPT 18 carries `11343` for exactly this path, which is the number
    that produced the corrupt document; a spec that only becomes `whole_file`
    later reads that number back. `tail -c +11344` on the new document is then
    the original defect with a repaired writer behind it, and the result still
    does not parse.
    """
    pod = tmp_path / "pod"
    pod.mkdir()
    remote = pod / "run" / "evidence.json"
    remote.parent.mkdir(parents=True)
    document = json.dumps({"stage": "I", "verdict": "GO", "pad": "x" * 400})
    remote.write_text(document)

    spec = RelaySpec(remote_path=str(remote), local_name="evidence.json",
                     required=False, whole_file=True)
    relay = LogRelay(LocalShellTarget(pod), (spec,), tmp_path / "durable")
    #: Exactly what the attempt-18 relay left behind: a byte offset into a
    #: document that no longer has that history.
    relay._save_offsets({spec.remote_path: 137})

    assert relay.sync_once().ok
    assert relay.local_path(spec).read_text() == document, (
        "a stored offset truncated the head off a whole document")
    assert json.loads(relay.local_path(spec).read_text())["verdict"] == "GO"


def test_an_append_only_spec_is_still_appended(tmp_path):
    """The flag must not turn the event streams into whole-file copies.

    `train_log.jsonl` is the file this relay exists for: appended to for hours,
    and re-reading it whole every cycle would move megabytes per poll.
    """
    pod, relay, spec = make_relay(tmp_path)
    assert spec.whole_file is False
    emit(pod / "run" / "train_log.jsonl", 5)
    relay.sync_once()
    emit(pod / "run" / "train_log.jsonl", 5, start=5)
    second = relay.sync_once()
    #: Only the new bytes moved; the whole file would be roughly twice this.
    assert 0 < second.synced_bytes[spec.remote_path] < 400
    assert [e["step"] for e in relay.recovered_events(spec)] == list(range(10))


def test_the_evidence_document_is_declared_rewritten_by_the_runner():
    """The flag is a repair only where it is actually applied."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2]
           / "src/aadistill/infrastructure/session_runner.py").read_text()
    block = src[src.index("specs = ["):src.index("relay = LogRelay(")]
    assert "whole_file=True" in block, (
        "the evidence document is relayed through the append path again")
    assert block.count("whole_file=True") == 1, (
        "an append-only stream was marked rewritten; re-reading a growing "
        "train log every cycle is what the offset scheme exists to avoid")
