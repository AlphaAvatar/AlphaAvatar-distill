"""`SessionRunner.attached_volume` — the provider flags for a pre-existing volume.

A session may bring bytes that were put in place while nothing expensive was
billing, by attaching a network volume the provider already holds. The C2
behavioural campaign does exactly that with its ten completed probes: ~22 GiB
that used to be copied onto a billing L40S once per attempt, at ~0.5 MB/s.

Everything here is about the COMMAND LINE, because that is where this can go
wrong invisibly. A volume that is named but not mounted where the session
expects, or attached in a datacenter the pod was not drawn in, produces a pod
that boots perfectly and then cannot find the probes — after setup, after the
image pull, after the teacher is materialized, and unrecoverably, because the
protocol forbids retraining a completed probe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from aadistill.infrastructure.session import SessionSpecError  # noqa: E402
from aadistill.infrastructure.session_runner import SessionRunner  # noqa: E402


class _Args:
    """Only what `attached_volume` reads. Absent attributes are the point."""

    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _runner(**kw) -> SessionRunner:
    """A runner shell carrying only what `attached_volume` reads.

    `ws` is the deployment's declared workspace root — `self.spec.commands.
    workspace_root` on a real runner — and the refusal names it rather than a
    path literal, because a deployment fact written into reusable core is the
    thing `test_no_command_the_runner_builds_contains_a_foreign_root` forbids.
    """
    r = SessionRunner.__new__(SessionRunner)
    r.a = _Args(**kw)
    r.ws = "/some/declared/workspace"
    return r


def test_a_session_that_names_no_volume_gets_the_command_it_always_got():
    """The default must be byte-identical for every launcher that predates this.

    Every other experiment in the repository defines none of these flags. If
    attaching became something they had to opt out of, this change would have
    silently altered the provisioning of sessions it has nothing to do with.
    """
    assert _runner().attached_volume() == ("--volume-in-gb", "0")
    assert _runner(network_volume_id="").attached_volume() == (
        "--volume-in-gb", "0")
    assert _runner(network_volume_id=None).attached_volume() == (
        "--volume-in-gb", "0")


def test_an_attached_volume_replaces_the_pods_own_ephemeral_volume():
    """`--volume-in-gb` is not passed beside `--network-volume-id`.

    They are alternatives. Asking the provider for both invites it to arbitrate
    between them, and which one wins is not something this project should learn
    from a running pod.
    """
    flags = _runner(network_volume_id="vol123",
                    volume_mount_path="/durable",
                    data_center_ids="EU-NL-1").attached_volume()
    assert "--volume-in-gb" not in flags
    assert flags == ("--network-volume-id", "vol123",
                     "--volume-mount-path", "/durable",
                     "--data-center-ids", "EU-NL-1")


def test_a_volume_without_a_mount_path_is_refused_before_the_pod_exists():
    """Providers default the mount to the session's own workspace root.

    Silently accepting that would put a session's repository on shared network
    storage and let two sessions share it. The refusal is here, at `$0`, rather
    than as a surprise on a pod — and it names the workspace THIS deployment
    declares, because reusable core may not carry one deployment's paths.
    """
    with pytest.raises(SessionSpecError, match="volume-mount-path"):
        _runner(network_volume_id="vol123",
                data_center_ids="EU-NL-1").attached_volume()
    with pytest.raises(SessionSpecError, match="volume-mount-path"):
        _runner(network_volume_id="vol123", volume_mount_path="  ",
                data_center_ids="EU-NL-1").attached_volume()
    #: And the message names the declared workspace rather than a literal.
    try:
        _runner(network_volume_id="vol123").attached_volume()
    except SessionSpecError as exc:
        assert "/some/declared/workspace" in str(exc)


def test_a_volume_without_a_datacenter_still_attaches_but_says_so_in_the_flags():
    """A volume lives in ONE datacenter and a pod attaches it only from there.

    The runner does not invent one: it passes what it was given. A campaign
    that needs the draw constrained supplies the datacenter, and its own
    launch-bound gate is where "you must supply it" is enforced — this layer
    is generic and a future caller may legitimately let the provider choose.
    """
    flags = _runner(network_volume_id="vol123",
                    volume_mount_path="/durable").attached_volume()
    assert flags == ("--network-volume-id", "vol123",
                     "--volume-mount-path", "/durable")


def test_the_flags_reach_the_provider_command_line():
    """The unit above is only worth something if `create` actually splices it.

    Read from the source rather than by running `create`, which needs a plan, a
    provider and a filesystem. What matters is that the call site expands
    `attached_volume()` instead of carrying its own copy of the flags — a
    second copy is how the two would drift.
    """
    src = (REPO / "src/aadistill/infrastructure/session_runner.py").read_text()
    body = src[src.index("def create(self) -> bool:"):]
    head = body[:body.index("capture_output=True")]
    assert "*self.attached_volume()," in head
    assert '"--volume-in-gb"' not in head, (
        "create still hardcodes the ephemeral volume beside the helper that "
        "decides it; one of the two will be wrong")
