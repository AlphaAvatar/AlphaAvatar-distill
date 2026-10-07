"""Move a large file off a remote host fast enough that its size is not a cost.

WHY THIS EXISTS. A single scp connection to a rented pod in this programme was
measured at **0.486 MB/s**, and the same host at eight concurrent byte-range
readers at **4.46 MB/s** -- whole-file **8.15 MB/s** once reassembly stopped
double-handling the bytes. That is not a tuning difference. At one stream a
1.2 GiB checkpoint takes 44 minutes of billing, so a session that produces two
of them spends more on moving them than on computing them, and a 45-minute
transfer cap loses the second one outright. The bottleneck is per-connection,
not the link, so the fix is connections.

The transport is `dd` over ssh with the range in BYTES, written straight into
the destination at its offset:

* `iflag=skip_bytes,count_bytes` -- a block-rounded `skip=N` misaligns the final
  partial chunk, and a reassembly hash catches that only after the whole
  transfer has been paid for.
* `os.pwrite` into a pre-sized file -- NO part files. Writing each range to its
  own file and concatenating needs **twice** the file's size on disk, and the
  transient double is what refuses on a dev box holding several products.
  Concurrent `pwrite` to one descriptor at disjoint offsets is safe: it neither
  uses nor moves the shared file position.

Verification is the caller's digest, re-derived here from the bytes that
ARRIVED. A product verified only where it was produced is a product whose
transfer was never checked.

Nothing here knows what it is moving. Paths, digests, stream counts and the
free-space reserve are all the caller's.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: Non-interactive, and tolerant of a pod whose sshd is slow to answer. A
#: transfer that dies on the first stalled connection would restart an hour of
#: billing.
SSH_OPTS = (
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",
)

#: Eight was measured, not guessed: 1 stream 0.486 MB/s, 8 streams 4.46 MB/s on
#: the same host and file. More streams did not help and each one is an ssh
#: process on a pod that is also computing.
DEFAULT_STREAMS = 8

#: Read size per `pwrite`. Large enough that syscall overhead is irrelevant,
#: small enough that a stalled stream is noticed in seconds.
_BLOCK = 1 << 22


class TransferError(RuntimeError):
    """A remote file could not be moved, or did not arrive as its digest says."""


def remote_argv(host: str, port: str | int, command: str) -> list[str]:
    """How a command is run on the remote host. ONE place.

    Isolated so a test can exercise the byte arithmetic and the in-place
    assembly against a local fake without an ssh hop it does not change.
    """
    return ["ssh", "-p", str(port), *SSH_OPTS, host, command]


def _run(host: str, port: str | int, command: str, *,
         timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(remote_argv(host, port, command),
                          capture_output=True, text=True, timeout=timeout)


def remote_size(host: str, port: str | int, path: str, *,
                timeout: float = 120.0) -> int:
    """The remote file's size in bytes. Read, never assumed.

    Sizes in this programme are not uniform -- a checkpoint's size depends on
    how much of an operator path has been applied -- so a caller that assumed
    one would mis-chunk every other file.
    """
    result = _run(host, port, f"stat -c %s '{path}'", timeout=timeout)
    if result.returncode != 0:
        raise TransferError(
            f"stat failed for {path}: {result.stderr.strip()[:200]}")
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise TransferError(
            f"stat for {path} returned {result.stdout.strip()[:80]!r}") from exc


def free_bytes(path: str | Path) -> int:
    stat = os.statvfs(str(path))
    return stat.f_bavail * stat.f_frsize


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def byte_ranges(total: int, streams: int) -> list[tuple[int, int]]:
    """`(start, count)` pairs covering exactly `total`, at most `streams` of them.

    Ceiling division, so the LAST range is the short one and every earlier one
    is full. A floor would leave a tail no range covers.
    """
    if total < 0:
        raise ValueError(f"total must not be negative: {total}")
    if streams < 1:
        raise ValueError(f"streams must be at least 1: {streams}")
    if total == 0:
        return []
    chunk = -(-total // streams)
    ranges = [(i * chunk, min(chunk, total - i * chunk))
              for i in range(streams) if i * chunk < total]
    covered = sum(count for _, count in ranges)
    if covered != total:
        raise TransferError(
            f"chunk arithmetic covers {covered} of {total} bytes")
    return ranges


def fetch_byte_range(host: str, port: str | int, path: str, start: int,
                     count: int, fd: int, *, attempts: int = 4,
                     on_log: Callable[[str], None] | None = None) -> int:
    """One byte range, written straight into `fd` at its offset.

    Retried on a short read because a dropped connection mid-range is the
    ordinary failure here, and the range is idempotent: it rewrites the same
    offsets. Raises rather than returning short -- a silent partial would be
    found by the digest, after the rest of the transfer had been paid for.
    """
    written = 0
    for attempt in range(1, attempts + 1):
        proc = subprocess.Popen(
            remote_argv(host, port,
                        f"dd if='{path}' bs=1M iflag=skip_bytes,count_bytes "
                        f"skip={start} count={count} 2>/dev/null"),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        written = 0
        try:
            while written < count:
                block = proc.stdout.read(min(_BLOCK, count - written))
                if not block:
                    break
                os.pwrite(fd, block, start + written)
                written += len(block)
        finally:
            proc.stdout.close()
            proc.wait()
        if written == count:
            return written
        if on_log:
            on_log(f"range {start}+{count}: got {written}, "
                   f"retry {attempt}/{attempts}")
        time.sleep(5)
    raise TransferError(
        f"range {start}+{count} of {path} never completed "
        f"({written} of {count} bytes)")


def fetch_file(host: str, port: str | int, remote_path: str,
               dest: str | Path, *, streams: int = DEFAULT_STREAMS,
               expect_sha256: str | None = None,
               reserve_bytes: int = 0,
               on_log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """One remote file, in parallel ranges, verified here against its digest.

    `reserve_bytes` lets a caller refuse before a fetch can crowd the
    filesystem that other products already live on. The requirement is ONE
    file's size plus the reserve, because the ranges are written in place.

    Returns a record rather than raising on a digest mismatch: a caller usually
    wants to report which product failed to verify alongside the ones that did,
    and `verified` is the field that decides. A transport failure -- a range
    that never completed, a file whose size cannot be read -- does raise.
    """
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(host, port, remote_path)
    need = total + int(reserve_bytes)
    have = free_bytes(destination.parent)
    if have < need:
        raise TransferError(
            f"refusing {remote_path}: {total / 2**30:.2f} GiB needs "
            f"{need / 2**30:.2f} GiB free (the file plus a "
            f"{reserve_bytes / 2**30:.2f} GiB reserve) and "
            f"{have / 2**30:.2f} GiB is available")
    ranges = byte_ranges(total, streams)
    if on_log:
        on_log(f"{destination.name}: {total} bytes over {len(ranges)} streams "
               f"({have / 2**30:.1f} GiB free)")

    started = time.time()
    fd = os.open(destination, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
    try:
        os.ftruncate(fd, total)
        with ThreadPoolExecutor(max_workers=max(len(ranges), 1)) as pool:
            futures = [pool.submit(fetch_byte_range, host, port, remote_path,
                                   start, count, fd, on_log=on_log)
                       for start, count in ranges]
            for future in futures:
                future.result()
    finally:
        os.close(fd)
    elapsed = max(time.time() - started, 1e-6)

    got = destination.stat().st_size
    digest = sha256_file(destination)
    verified = got == total and (expect_sha256 is None
                                or digest == expect_sha256)
    if on_log:
        on_log(f"{destination.name}: {got} bytes in {elapsed / 60:.1f} min "
               f"({got / 1048576 / elapsed:.2f} MB/s) sha256 {digest[:16]} "
               f"{'MATCHES' if verified else 'DOES NOT MATCH'}")
    return {
        "path": str(destination), "bytes": got, "expected_bytes": total,
        "sha256": digest, "expected_sha256": expect_sha256,
        "verified": bool(verified), "seconds": round(elapsed, 1),
        "mb_per_s": round(got / 1048576 / elapsed, 3),
        "streams": len(ranges),
    }


def fetch_small_files(host: str, port: str | int, remote_dir: str,
                      dest: str | Path, names: tuple[str, ...], *,
                      timeout: float = 120.0) -> list[str]:
    """Text sidecars next to a large file, one `cat` each.

    Separate from `fetch_file` because the parallel machinery is pure overhead
    for a few KB, and because a MISSING sidecar is not an error here: a caller
    asks for the ones a format may carry, and the identity check downstream
    decides whether an absent one matters.
    """
    destination = Path(dest)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name in names:
        result = _run(host, port, f"cat '{remote_dir}/{name}'", timeout=timeout)
        if result.returncode == 0:
            (destination / name).write_text(result.stdout)
            written.append(name)
    return written


def fetch_checkpoint_dir(host: str, port: str | int, remote_dir: str,
                         dest: str | Path, *, weights_name: str,
                         sidecars: tuple[str, ...] = (),
                         streams: int = DEFAULT_STREAMS,
                         expect_sha256: str | None = None,
                         reserve_bytes: int = 0,
                         on_log: Callable[[str], None] | None = None
                         ) -> Mapping[str, Any]:
    """A checkpoint directory: one large file in parallel, sidecars serially.

    `weights_name` and `sidecars` are the caller's, because which file holds the
    weights is a format question and this module does not resolve formats.
    """
    destination = Path(dest)
    record = dict(fetch_file(
        host, port, f"{remote_dir}/{weights_name}", destination / weights_name,
        streams=streams, expect_sha256=expect_sha256,
        reserve_bytes=reserve_bytes, on_log=on_log))
    record["dir"] = str(destination)
    record["sidecars"] = fetch_small_files(host, port, remote_dir, destination,
                                           sidecars)
    return record
