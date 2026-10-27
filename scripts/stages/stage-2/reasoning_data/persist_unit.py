#!/usr/bin/env python3
"""Completion-time persistence of one H1 unit's outputs. Runs where the unit ran.

    # the moment training ends, before evaluation starts
    python -m stages.reasoning_data.persist_unit checkpoint --unit <id> --checkpoint <trainer model dir> \
        --durable-root <mounted backend> --init-record <assets/d4_top1_materialization.json> \
        [--capacity-bytes <provider's figure>]
    # the moment evaluation ends (logical names are paths relative to --base)
    python -m stages.reasoning_data.persist_unit files --unit <id> --durable-root <root> \
        --base <evaluation dir> --files <raw outputs...> [--label evaluation]

A unit's trained student and its raw evaluation outputs are copied to the
durable backend AS EACH IS COMPLETED, so a later failure -- evaluation,
closeout, the pod itself -- cannot take them (AGENTS.md P8.2.1; C1 attempt 17
lost six trained probes to a later stage's failure).

COMPLETED EVIDENCE IS NEVER ALTERED. Per unit and kind (``checkpoint``, or a
file-set label such as ``evaluation``) there are two records:

* ``<kind>.descriptor.json`` -- the COMPLETED record, and nothing else ever
  has that name: it is written to a temporary name and atomically renamed only
  after the arrived bytes re-identify, and it is never rewritten;
* ``<kind>.inflight.json`` -- the attempt in progress: the identity being
  protected, written BEFORE the first risky byte so a copy that dies halfway
  still leaves a record of what it was protecting; a failure adds its error;
  removed once the completed record is committed.

Before anything is written, the completed target is checked. The same identity
saved again returns ``status: already_persisted, consistent: true`` after the
landed bytes are re-checked, and writes nothing; a different identity is
refused and the old bytes and descriptor are left exactly as they were. So a
retry -- after no room, a dead copy, a crash -- can never turn a completed
record into ``persisted: false``. A completed target with no completed record
(an attempt that stopped between its final rename and its commit) is committed
only when its in-flight record names this identity and the landed bytes
re-identify; anything else is unknown provenance, refused and left in place.

Files keep their LOGICAL paths (relative to ``base``, or the keys of an
explicit mapping); two inputs with one logical name are refused before any
byte moves, so same-named outputs of different directories never overwrite
each other. A file set lands in a staging directory, is hash-checked, and only
then takes its final name.

* Room is checked before the first byte moves; refusal is the answer. On a
  network volume ``df`` is not the quota, so ``--capacity-bytes`` (the
  provider's figure, less what is already there) is used when given.
* A checkpoint lands under a ``.partial`` name, is RE-IDENTIFIED from the
  bytes that arrived (`aadistill.runtime.leaf_durability.verify_transferred_leaf`,
  the check the D4 restore used), and only then takes its final name.

A trained checkpoint has no architecture spec of its own; its arch signature
and parameter count are the initialization's (`--init-record`). Persistence
authorizes no reuse: whether a preserved unit may be resumed or rerun is the
protocol's retry rule, not this file's existence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from aadistill.runtime.leaf_durability import (
    LeafDurabilityError, free_bytes_at, identify_for_transfer, verify_transferred_leaf)

SCHEMA = "aadistill.stage2.reasoning_data.persisted_unit/v2"
MARGIN = 1.05
#: what makes two checkpoint saves "the same": the bytes' identity and the
#: inputs the unit was trained from
IDENTITY = ("artifact_digest", "weights_digest", "config_sha256", "arch_signature",
            "num_parameters", "tokenizer_sha256", "inputs")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _room(dest: Path, need: int, capacity: int | None) -> None:
    free = capacity if capacity is not None else free_bytes_at(dest)
    if free < need * MARGIN:
        raise LeafDurabilityError(f"{dest}: {free} bytes free for {need} (x{MARGIN}); nothing was copied")


def _write_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def completed(root: Path, unit: str, kind: str) -> dict | None:
    """The COMPLETED record of `unit`'s `kind`, or None. A file under that name
    that is not a completed record is refused: it is not this module's."""
    doc = _read(Path(root) / unit / f"{kind}.descriptor.json")
    if doc is not None and doc.get("persisted") is not True:
        raise LeafDurabilityError(f"{root}/{unit}/{kind}.descriptor.json is not a completed record; "
                                  "preserved, nothing changed")
    return doc


def _ok(verdict: dict) -> bool:
    return bool(verdict["matched"] and verdict["weights_digest_matched"] and verdict["config_matched"] is not False)


def _attempt(inflight: Path, record: dict, work) -> None:
    """Run `work` under the in-flight record; a failure is added to it, never
    to a completed record."""
    _write_atomic(inflight, record)
    try:
        work()
    except BaseException as exc:
        _write_atomic(inflight, {**record, "failed_utc": _now(), "error": f"{type(exc).__name__}: {exc}"[:800]})
        raise


def _commit(desc: Path, inflight: Path, record: dict) -> dict:
    _write_atomic(desc, record)
    inflight.unlink(missing_ok=True)
    return {**record, "status": "persisted"}


def persist_checkpoint(unit: str, checkpoint: Path, root: Path, *, adapter, arch_signature: str,
                       num_parameters: int, inputs: dict, capacity: int | None = None) -> dict:
    """Persist a trained checkpoint once. Returns the completed record with
    ``status`` ``persisted`` or ``already_persisted`` (``consistent: true``)."""
    src = identify_for_transfer(checkpoint, adapter=adapter, arch_signature=arch_signature,
                                num_parameters=num_parameters)
    ident = {"artifact_digest": src.artifact_digest, "weights_digest": src.weights_digest,
             "config_sha256": src.config_sha256, "arch_signature": arch_signature,
             "num_parameters": int(num_parameters), "tokenizer_sha256": src.tokenizer_sha256,
             "inputs": json.loads(json.dumps(inputs))}
    root = Path(root)
    unit_dir = root / unit
    final, desc, inflight = unit_dir / "checkpoint", unit_dir / "checkpoint.descriptor.json", \
        unit_dir / "checkpoint.inflight.json"
    done = completed(root, unit, "checkpoint")
    if done is not None:
        diff = [k for k in IDENTITY if done.get(k) != ident[k]]
        if diff:
            raise LeafDurabilityError(f"{final}: a different checkpoint is already persisted (differs in {diff}); "
                                      "a completed unit is never overwritten, nothing was changed")
        verdict = verify_transferred_leaf(final, done, adapter=adapter)
        if not _ok(verdict):
            raise LeafDurabilityError(f"{final}: the persisted bytes no longer re-identify ({verdict}); "
                                      "preserved, nothing was changed")
        return {**done, "status": "already_persisted", "consistent": True}
    if final.exists():
        prior = _read(inflight)
        if prior is None or any(prior.get(k) != ident[k] for k in IDENTITY):
            raise LeafDurabilityError(f"{final} exists with no completed record and no in-flight record of this "
                                      "identity: unknown provenance, preserved, nothing was changed")
        verdict = verify_transferred_leaf(final, prior, adapter=adapter)
        if not _ok(verdict):
            raise LeafDurabilityError(f"{final}: landed bytes do not re-identify ({verdict}); preserved")
        rec = {k: v for k, v in prior.items() if k not in ("failed_utc", "error")}
        return {**_commit(desc, inflight, {**rec, "persisted": True, "finished_utc": _now(), "verified": verdict,
                                           "completed_after_interruption": True})}
    unit_dir.mkdir(parents=True, exist_ok=True)
    record = {"schema": SCHEMA, "unit": unit, "kind": "checkpoint", "started_utc": _now(),
              "source": str(checkpoint), **ident, "destination": str(final), "persisted": False}
    need = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    landed = {}

    def work():
        _room(root, need, capacity)
        partial = unit_dir / "checkpoint.partial"
        if partial.exists():                  # a dead earlier attempt's copy, never a completed one
            shutil.rmtree(partial)
        shutil.copytree(checkpoint, partial)
        verdict = verify_transferred_leaf(partial, record, adapter=adapter)
        if not _ok(verdict):
            raise LeafDurabilityError(f"{partial}: arrived bytes do not re-identify: {verdict}")
        partial.rename(final)
        landed["verified"] = verdict
    _attempt(inflight, record, work)
    return _commit(desc, inflight, {**record, "persisted": True, "finished_utc": _now(), "bytes": need,
                                    "verified": landed["verified"]})


def logical_names(files: Mapping[str, Path] | Sequence[Path], base: Path | None = None) -> dict[str, Path]:
    """{logical name: source}. A sequence is named relative to `base` (by
    basename without one); two inputs with one name are refused."""
    if isinstance(files, Mapping):
        items = [(str(k), Path(v)) for k, v in files.items()]
    else:
        items = []
        for f in map(Path, files):
            if base is None:
                items.append((f.name, f))
                continue
            try:
                items.append((f.resolve().relative_to(Path(base).resolve()).as_posix(), f))
            except ValueError:
                raise LeafDurabilityError(f"{f} is not under {base}; nothing was written") from None
    out: dict[str, Path] = {}
    for name, path in items:
        p = PurePosixPath(name)
        if not name or p.is_absolute() or p.as_posix() != name or any(x in ("..", ".") for x in p.parts):
            raise LeafDurabilityError(f"{name!r} is not a normalized relative logical name; nothing was written")
        if name in out:
            raise LeafDurabilityError(f"two inputs have the logical name {name!r} ({out[name]}, {path}); "
                                      "nothing was written")
        if not path.is_file():
            raise LeafDurabilityError(f"{path}: not a file; nothing was written")
        out[name] = path
    return out


def _check_landed(out: Path, rows: list[dict]) -> list[str]:
    want = {r["name"]: r["sha256"] for r in rows}
    have = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    bad = sorted(n for n in want if n not in have or _sha256(out / n) != want[n])
    return bad + sorted(have - set(want))


def persist_files(unit: str, files: Mapping[str, Path] | Sequence[Path], root: Path, *,
                  label: str = "evaluation", base: Path | None = None, capacity: int | None = None) -> dict:
    """Persist one completed file set under its logical names, once. Returns
    the completed record with ``status`` ``persisted`` or ``already_persisted``."""
    logical = logical_names(files, base)
    rows = [{"name": n, "sha256": _sha256(p), "bytes": p.stat().st_size} for n, p in sorted(logical.items())]
    root = Path(root)
    unit_dir = root / unit
    out, desc, inflight = unit_dir / label, unit_dir / f"{label}.descriptor.json", unit_dir / f"{label}.inflight.json"
    done = completed(root, unit, label)
    if done is not None:
        if done["files"] != rows:
            raise LeafDurabilityError(f"{out}: a different {label} file set is already persisted; "
                                      "a completed set is never overwritten, nothing was changed")
        bad = _check_landed(out, rows)
        if bad:
            raise LeafDurabilityError(f"{out}: persisted files no longer match their record {bad[:5]}; preserved")
        return {**done, "status": "already_persisted", "consistent": True}
    if out.exists():
        prior = _read(inflight)
        if prior is None or prior.get("files") != rows or _check_landed(out, rows):
            raise LeafDurabilityError(f"{out} exists with no completed record matching this file set: "
                                      "unknown provenance, preserved, nothing was changed")
        rec = {k: v for k, v in prior.items() if k not in ("failed_utc", "error")}
        return _commit(desc, inflight, {**rec, "persisted": True, "finished_utc": _now(),
                                        "completed_after_interruption": True})
    unit_dir.mkdir(parents=True, exist_ok=True)
    record = {"schema": SCHEMA, "unit": unit, "kind": label, "started_utc": _now(),
              "files": rows, "destination": str(out), "persisted": False}

    def work():
        _room(root, sum(r["bytes"] for r in rows), capacity)
        staging = unit_dir / f"{label}.partial"
        if staging.exists():
            shutil.rmtree(staging)
        for name, src in logical.items():
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, staging / name)
        bad = _check_landed(staging, rows)
        if bad:
            raise LeafDurabilityError(f"{staging}: copied bytes differ from the sources {bad[:5]}")
        staging.rename(out)
    _attempt(inflight, record, work)
    return _commit(desc, inflight, {**record, "persisted": True, "finished_utc": _now()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkpoint")
    c.add_argument("--unit", required=True)
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--durable-root", required=True)
    c.add_argument("--init-record", required=True)
    c.add_argument("--capacity-bytes", type=int, default=None)
    f = sub.add_parser("files")
    f.add_argument("--unit", required=True)
    f.add_argument("--durable-root", required=True)
    f.add_argument("--files", nargs="+", required=True)
    f.add_argument("--base", default=None, help="logical names are paths relative to this directory")
    f.add_argument("--label", default="evaluation")
    f.add_argument("--capacity-bytes", type=int, default=None)
    args = ap.parse_args(argv)
    root = Path(args.durable_root)
    if args.cmd == "files":
        rec = persist_files(args.unit, [Path(p) for p in args.files], root, label=args.label,
                            base=Path(args.base) if args.base else None, capacity=args.capacity_bytes)
    else:
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.specs.arch import get_adapter
        register_builtin_adapters()
        ident = json.loads(Path(args.init_record).read_text())["materialization_identity"]
        rec = persist_checkpoint(args.unit, Path(args.checkpoint), root, adapter=get_adapter("qwen3"),
                                 arch_signature=ident["arch_signature"], num_parameters=ident["num_parameters"],
                                 inputs={"initialization_record_sha256": _sha256(Path(args.init_record))},
                                 capacity=args.capacity_bytes)
    print(json.dumps({k: rec[k] for k in ("unit", "kind", "status", "persisted", "destination")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
