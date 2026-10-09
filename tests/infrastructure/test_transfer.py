"""Parallel byte-range transfer: the arithmetic, the assembly, the verdict.

The ssh hop is NOT what these test. It was proven separately, by a 3.6 GB
transfer whose reassembled sha256 matched the digest a search had recorded. What
these test is everything around it, because that is where the failures were:

* **BYTES, not blocks.** `dd skip=N` without `iflag=skip_bytes` rounds to the
  block size, and the misalignment lands in the final partial chunk -- found by
  the reassembly hash only after the whole transfer has been paid for.
* **Coverage.** Floor division leaves a tail no range reads. The file arrives
  the right SIZE, because it was pre-sized, with a hole in it.
* **In place.** Part files plus a concatenate need twice the file's size on
  disk, and the transient double is what refuses on a host already holding
  other products.
* **A verdict that is not a raise.** A caller fetching several products wants
  to report which one failed to verify alongside the ones that did, so a digest
  mismatch returns `verified: False`; a transport failure raises.

`remote_argv` is the seam: these substitute a local reader for the ssh hop and
exercise the real range loop, the real `pwrite`, and the real digest.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys

import pytest

from aadistill.infrastructure import transfer as T


# --- the seam ---------------------------------------------------------------

def _local_argv(host, port, command):
    """Serve `stat -c %s` and `dd ...` from the local filesystem.

    Parses the command the module really builds, so a change to the dd flags
    changes what these tests execute. A `dd` that dropped `iflag=skip_bytes`
    would be read here in BLOCKS and the assertions below would catch it.
    """
    parts = shlex.split(command)
    if parts[0] == "stat":
        return [sys.executable, "-c",
                "import os,sys; sys.stdout.write(str(os.path.getsize(sys.argv[1])))",
                parts[-1]]
    if parts[0] == "cat":
        return [sys.executable, "-c",
                "import sys; sys.stdout.write(open(sys.argv[1]).read())",
                parts[-1]]
    assert parts[0] == "dd", command
    flags = {}
    for token in parts[1:]:
        if "=" in token:
            key, _, value = token.partition("=")
            flags[key] = value
    assert flags.get("iflag") == "skip_bytes,count_bytes", (
        "the range is not expressed in bytes; a block-rounded skip misaligns "
        "the final partial chunk")
    return [sys.executable, "-c",
            "import sys\n"
            "with open(sys.argv[1],'rb') as f:\n"
            "    f.seek(int(sys.argv[2]))\n"
            "    sys.stdout.buffer.write(f.read(int(sys.argv[3])))",
            flags["if"].strip("'"), flags["skip"], flags["count"]]


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(T, "remote_argv", _local_argv)


@pytest.fixture
def source(tmp_path):
    """A file whose size is deliberately NOT a multiple of anything: the last
    range is short and the tail is where the off-by-a-block bugs live."""
    payload = bytes((i * 31 + 7) % 256 for i in range(1_000_003))
    path = tmp_path / "remote" / "model.safetensors"
    path.parent.mkdir()
    path.write_bytes(payload)
    return path, payload, hashlib.sha256(payload).hexdigest()


# --- arithmetic -------------------------------------------------------------

class TestTheRangesCoverTheFileExactly:

    @pytest.mark.parametrize("total,streams", [
        (1, 1), (1, 8), (7, 3), (8, 8), (9, 8), (1_000_003, 8),
        (1 << 30, 8), ((1 << 30) + 1, 7),
    ])
    def test_every_byte_is_in_exactly_one_range(self, total, streams):
        ranges = T.byte_ranges(total, streams)
        assert sum(count for _, count in ranges) == total
        assert ranges[0][0] == 0
        for (start, count), (next_start, _) in zip(ranges, ranges[1:]):
            assert start + count == next_start, "ranges overlap or leave a gap"
        assert ranges[-1][0] + ranges[-1][1] == total

    def test_the_short_range_is_the_last_one(self):
        """Ceiling division, so earlier ranges are full. Floor division would
        leave a tail beyond the last range and the file would arrive the right
        size with a hole in it."""
        ranges = T.byte_ranges(9, 8)
        counts = [count for _, count in ranges]
        assert counts[-1] <= counts[0]
        assert sum(counts) == 9

    def test_never_more_ranges_than_streams(self):
        for streams in (1, 2, 8, 64):
            assert len(T.byte_ranges(1_000_003, streams)) <= streams

    def test_an_empty_file_needs_no_ranges(self):
        assert T.byte_ranges(0, 8) == []

    @pytest.mark.parametrize("total,streams", [(-1, 8), (10, 0), (10, -3)])
    def test_nonsense_is_refused(self, total, streams):
        with pytest.raises(ValueError):
            T.byte_ranges(total, streams)


# --- assembly ---------------------------------------------------------------

class TestTheFileIsAssembledInPlaceAndVerified:

    def test_the_bytes_arrive_identical(self, local, source, tmp_path):
        path, payload, digest = source
        record = T.fetch_file("fake@host", 1234, str(path),
                              tmp_path / "out" / "model.safetensors",
                              streams=8, expect_sha256=digest)
        assert record["verified"] is True
        assert record["bytes"] == len(payload) == record["expected_bytes"]
        assert (tmp_path / "out" / "model.safetensors").read_bytes() == payload
        assert record["streams"] == 8

    @pytest.mark.parametrize("streams", [1, 2, 3, 8, 16])
    def test_any_stream_count_reassembles(self, local, source, tmp_path,
                                          streams):
        path, payload, digest = source
        record = T.fetch_file("fake@host", 1234, str(path),
                              tmp_path / f"out{streams}" / "w.safetensors",
                              streams=streams, expect_sha256=digest)
        assert record["verified"] is True
        assert record["sha256"] == digest

    def test_no_part_files_are_left_anywhere(self, local, source, tmp_path):
        """The in-place write is the whole reason the disk requirement is one
        file's size rather than two. A part file would prove it regressed."""
        path, _, digest = source
        out = tmp_path / "out"
        T.fetch_file("fake@host", 1234, str(path), out / "model.safetensors",
                     streams=8, expect_sha256=digest)
        assert [p.name for p in out.iterdir()] == ["model.safetensors"]

    def test_a_wrong_digest_is_a_verdict_and_not_an_exception(
            self, local, source, tmp_path):
        path, payload, _ = source
        record = T.fetch_file("fake@host", 1234, str(path),
                              tmp_path / "out" / "w.safetensors",
                              streams=4, expect_sha256="0" * 64)
        assert record["verified"] is False
        assert record["bytes"] == len(payload)
        assert record["expected_sha256"] == "0" * 64

    def test_no_expected_digest_still_reports_the_one_it_got(
            self, local, source, tmp_path):
        path, _, digest = source
        record = T.fetch_file("fake@host", 1234, str(path),
                              tmp_path / "out" / "w.safetensors", streams=4)
        assert record["verified"] is True
        assert record["sha256"] == digest
        assert record["expected_sha256"] is None


class TestItRefusesBeforeItCrowdsTheFilesystem:

    def test_a_fetch_that_does_not_fit_is_refused_up_front(
            self, local, source, tmp_path, monkeypatch):
        """Checked BEFORE the transfer, because discovering it at the last
        range means an hour of billing for a file that cannot land."""
        path, payload, digest = source
        monkeypatch.setattr(T, "free_bytes", lambda _p: len(payload) // 2)
        with pytest.raises(T.TransferError) as exc:
            T.fetch_file("fake@host", 1234, str(path),
                         tmp_path / "out" / "w.safetensors",
                         streams=8, expect_sha256=digest)
        assert "refusing" in str(exc.value)
        assert not (tmp_path / "out" / "w.safetensors").exists()

    def test_the_reserve_is_counted_against_the_requirement(
            self, local, source, tmp_path, monkeypatch):
        """Room for the file and not for the reserve must refuse: the reserve
        exists so an archival fetch cannot crowd the products already there."""
        path, payload, digest = source
        monkeypatch.setattr(T, "free_bytes", lambda _p: len(payload) + 1024)
        T.fetch_file("fake@host", 1234, str(path),
                     tmp_path / "fits" / "w.safetensors",
                     streams=4, expect_sha256=digest)
        with pytest.raises(T.TransferError):
            T.fetch_file("fake@host", 1234, str(path),
                         tmp_path / "nope" / "w.safetensors",
                         streams=4, expect_sha256=digest,
                         reserve_bytes=1 << 20)


class TestATransportFailureRaisesRatherThanReturningShort:

    def test_a_range_that_never_completes_raises(self, source, tmp_path,
                                                 monkeypatch):
        """A silent partial would be caught by the digest -- after the rest of
        the transfer was paid for. It raises at the range instead."""
        path, _, digest = source

        def truncating(host, port, command):
            argv = _local_argv(host, port, command)
            if argv[0] == sys.executable and "f.read" in argv[2]:
                #: Return one byte less than asked for, every time.
                argv[2] = argv[2].replace("f.read(int(sys.argv[3]))",
                                          "f.read(int(sys.argv[3]) - 1)")
            return argv

        monkeypatch.setattr(T, "remote_argv", truncating)
        with pytest.raises(T.TransferError) as exc:
            T.fetch_byte_range("fake@host", 1234, str(path), 0, 4096,
                               os.open(tmp_path / "x", os.O_CREAT | os.O_WRONLY),
                               attempts=1)
        assert "never completed" in str(exc.value)

    def test_an_unreadable_size_raises_naming_the_path(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(T, "remote_argv", lambda h, p, c: [
            sys.executable, "-c", "import sys; sys.exit(1)"])
        with pytest.raises(T.TransferError) as exc:
            T.remote_size("fake@host", 1234, "/nowhere/model.safetensors")
        assert "/nowhere/model.safetensors" in str(exc.value)

    def test_a_nonnumeric_size_raises_rather_than_chunking_garbage(
            self, monkeypatch):
        monkeypatch.setattr(T, "remote_argv", lambda h, p, c: [
            sys.executable, "-c", "print('No such file or directory')"])
        with pytest.raises(T.TransferError):
            T.remote_size("fake@host", 1234, "/nowhere/w")


class TestACheckpointDirectoryBringsItsSidecars:

    def test_the_weights_go_in_parallel_and_the_sidecars_serially(
            self, local, source, tmp_path):
        path, payload, digest = source
        (path.parent / "config.json").write_text('{"model_type": "toy"}')
        record = T.fetch_checkpoint_dir(
            "fake@host", 1234, str(path.parent), tmp_path / "ckpt",
            weights_name="model.safetensors",
            sidecars=("config.json", "generation_config.json"),
            streams=8, expect_sha256=digest)
        assert record["verified"] is True
        assert record["sidecars"] == ["config.json"], (
            "an absent sidecar must not fail the fetch; the identity check "
            "downstream decides whether a missing one matters")
        assert (tmp_path / "ckpt" / "model.safetensors").read_bytes() == payload
        assert (tmp_path / "ckpt" / "config.json").read_text() == \
            '{"model_type": "toy"}'


class TestTheTransportKnowsNothingAboutWhatItMoves:
    """P3. A checkpoint format, a model family, a run id or an experiment in
    here would make the next session's fetch a fork of this one."""

    def test_no_format_or_experiment_vocabulary_in_the_module(self):
        from pathlib import Path

        src = Path(T.__file__).read_text()
        body = "\n".join(line for line in src.splitlines()
                         if not line.lstrip().startswith("#"))
        head, _, rest = body.partition('"""')
        docstring, _, code = rest.partition('"""')
        for word in ("safetensors", "qwen", "phase_", "d1_", "c2_",
                     "generation_config", "state_id"):
            assert word not in code.lower(), (
                f"{word!r} is in the transport's code; what a checkpoint "
                "directory contains is the caller's knowledge")

    def test_the_stream_count_is_a_named_measured_constant(self):
        from pathlib import Path

        src = Path(T.__file__).read_text()
        assert "DEFAULT_STREAMS = 8" in src
        assert "0.486 MB/s" in src and "4.46 MB/s" in src, (
            "the default is a measurement; if the numbers behind it are gone "
            "it is a guess")


class TestTheSSHHopIsBuiltOnce:

    def test_remote_argv_is_the_only_place_ssh_is_spelled(self):
        from pathlib import Path

        src = Path(T.__file__).read_text()
        assert src.count('"ssh"') == 1, (
            "ssh is constructed in more than one place; the seam these tests "
            "substitute no longer covers every remote call")

    def test_it_carries_the_options_a_slow_pod_needs(self):
        argv = T.remote_argv("root@1.2.3.4", 19181, "stat -c %s /x")
        assert argv[:3] == ["ssh", "-p", "19181"]
        assert "ServerAliveInterval=15" in argv
        assert "ConnectTimeout=20" in argv
        assert argv[-2:] == ["root@1.2.3.4", "stat -c %s /x"]

    def test_the_port_may_be_an_int_or_a_string(self):
        assert T.remote_argv("h", 22, "x")[2] == "22"
        assert T.remote_argv("h", "22", "x")[2] == "22"


def test_subprocess_is_not_shelled_out_with_shell_true():
    """A remote path with a space or a quote in it would become two arguments
    under a shell. Every call here passes a list."""
    from pathlib import Path

    src = Path(T.__file__).read_text()
    assert "shell=True" not in src
    assert subprocess  # the import is real, not a stub
