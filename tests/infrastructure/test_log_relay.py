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
