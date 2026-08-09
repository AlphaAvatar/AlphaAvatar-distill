"""Mirror structured events off an ephemeral pod while it is still running.

E6b's `train_log.jsonl` and `run_manifest.json` existed, were written correctly,
and were lost — because they existed **only** on a pod, and the one chance to
retrieve them was a bundling step at teardown that did not list them. Nine hours
of machine-readable training events were destroyed by a single missing path in a
single command that ran once.

A structured event stream must not live only inside a pod until final teardown.
This relay pulls whatever has already been written, at whatever cadence the
poller runs at, and appends it to a durable local file. Its guarantee is narrow
and is the one that matters: **everything already synced survives the pod's
disappearance**, whatever happens next.

Design notes:

* Byte offsets, persisted. The remote files are append-only (`JsonlLogger` never
  overwrites), so `tail -c +N` is exact and resumable, and a sync that runs
  twice appends nothing the second time.
* base64 on the wire. A jsonl line contains quotes, braces and non-ASCII; moving
  it through a nested shell as text invites the quoting failure this whole
  session is about.
* **`sync_once` never raises.** It is called from a polling loop that is also
  the cost-control loop. A relay that throws on a dropped connection would take
  the poller down with it, and the poller is what tears the pod down. Errors are
  returned as data and the already-durable bytes stay durable.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .remote import ShellTarget

# How much to pull per file per cycle. A training log emits a few hundred bytes
# per event and the relay runs every poll, so this is generous; it exists so a
# pathological file cannot make one sync unbounded.
MAX_CHUNK_BYTES = 8 << 20


@dataclass(frozen=True)
class RelaySpec:
    """One remote append-only file and where it lands locally."""

    remote_path: str
    local_name: str
    required: bool = True


@dataclass
class RelayResult:
    """What one sync cycle achieved, per file. Never an exception."""

    synced_bytes: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"synced_bytes": dict(self.synced_bytes),
                "errors": dict(self.errors)}


class LogRelay:
    """Incrementally copy remote append-only files into a durable local dir."""

    def __init__(self, target: ShellTarget, specs: tuple[RelaySpec, ...],
                 local_root: str | Path, *, timeout: float = 120.0) -> None:
        self.target = target
        self.specs = specs
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.local_root / "relay_offsets.json"
        self.timeout = timeout

    # -- offsets -----------------------------------------------------------
    def _offsets(self) -> dict[str, int]:
        if not self.state_path.is_file():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            # A corrupt offset file must not wedge the relay. Re-syncing from
            # zero duplicates bytes, which is recoverable; refusing to sync is
            # not.
            return {}

    def _save_offsets(self, offsets: dict[str, int]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(offsets, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.state_path)

    def local_path(self, spec: RelaySpec) -> Path:
        return self.local_root / spec.local_name

    # -- one cycle ---------------------------------------------------------
    def sync_once(self) -> RelayResult:
        """Pull new bytes for every spec. Returns errors rather than raising."""
        result = RelayResult()
        offsets = self._offsets()
        for spec in self.specs:
            key = spec.remote_path
            start = int(offsets.get(key, 0))
            try:
                chunk, err = self._read_remote(spec.remote_path, start)
            except Exception as exc:  # noqa: BLE001 - see module docstring
                result.errors[key] = f"{type(exc).__name__}: {exc}"
                continue
            if err:
                if spec.required or "No such file" not in err:
                    result.errors[key] = err
                continue
            if not chunk:
                result.synced_bytes[key] = 0
                continue
            try:
                self._append(self.local_path(spec), chunk)
            except OSError as exc:
                result.errors[key] = f"local write failed: {exc}"
                continue
            offsets[key] = start + len(chunk)
            result.synced_bytes[key] = len(chunk)
        try:
            self._save_offsets(offsets)
        except OSError as exc:
            result.errors["<offsets>"] = f"{type(exc).__name__}: {exc}"
        return result

    def _read_remote(self, path: str, offset: int) -> tuple[bytes, str]:
        q = shlex.quote(path)
        # `tail -c +N` is 1-indexed, so an offset of 0 means byte 1.
        cmd = (f"if [ ! -f {q} ]; then echo 'MISSING'; else "
               f"tail -c +{offset + 1} {q} | head -c {MAX_CHUNK_BYTES} | "
               f"base64 | tr -d '\\n'; fi")
        res = self.target.run(cmd, timeout=self.timeout)
        if res.timed_out:
            return b"", f"timed out reading {path}"
        if res.returncode != 0:
            return b"", f"rc={res.returncode} {res.stderr.strip()[:300]}".strip()
        out = res.stdout.strip()
        if out == "MISSING":
            return b"", f"No such file: {path}"
        if not out:
            return b"", ""
        try:
            return base64.b64decode(out, validate=True), ""
        except (binascii.Error, ValueError) as exc:
            return b"", f"undecodable chunk from {path}: {exc}"

    @staticmethod
    def _append(path: Path, chunk: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as f:
            f.write(chunk)
            f.flush()
            os.fsync(f.fileno())

    # -- recovery ----------------------------------------------------------
    def recovered_events(self, spec: RelaySpec) -> list[dict]:
        """Parse whatever survived, tolerating a trailing partial line.

        A relay cycle can land mid-line if the writer was between `write` and
        newline. That final fragment is dropped and everything before it is
        returned, because a truncated last event is not a reason to discard the
        nine hours in front of it.
        """
        path = self.local_path(spec)
        if not path.is_file():
            return []
        events = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
