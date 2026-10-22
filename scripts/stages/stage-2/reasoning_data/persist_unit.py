#!/usr/bin/env python3
"""Completion-time persistence of one H1 unit's outputs. Runs where the unit ran.

    # the moment training ends, before evaluation starts
    python -m stages.reasoning_data.persist_unit checkpoint --unit <id> --checkpoint <trainer model dir> \
        --durable-root <mounted backend> --init-record <assets/d4_top1_materialization.json> \
        [--capacity-bytes <provider's figure>]
    # the moment evaluation ends
    python -m stages.reasoning_data.persist_unit files --unit <id> --durable-root <root> --files <raw outputs...>

A unit's trained student and its raw evaluation outputs are copied to the
durable backend AS EACH IS COMPLETED, so a later failure -- evaluation,
closeout, the pod itself -- cannot take them (AGENTS.md P8.2.1; C1 attempt 17
lost six trained probes to a later stage's failure).

* The DESCRIPTOR (unit, the checkpoint's identity, its inputs' hashes, the
  destination) is written to the durable root BEFORE the copy, so a copy that
  dies halfway still leaves a record of what was being protected.
* Room is checked before the first byte moves; refusal is the answer. On a
  network volume ``df`` is not the quota, so ``--capacity-bytes`` (the
  provider's figure, less what is already there) is used when given.
* The copy lands under a ``.partial`` name, is RE-IDENTIFIED from the bytes
  that arrived (`aadistill.runtime.leaf_durability.verify_transferred_leaf`, the
  check the D4 restore used), and only then takes its final name.

A trained checkpoint has no architecture spec of its own; its arch signature
and parameter count are the initialization's (`--init-record`). Persistence
authorizes no reuse: whether a preserved unit may be resumed or rerun is the
protocol's retry rule, not this file's existence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from aadistill.runtime.leaf_durability import (
    LeafDurabilityError, free_bytes_at, identify_for_transfer, verify_transferred_leaf)

SCHEMA = "aadistill.stage2.reasoning_data.persisted_unit/v1"
MARGIN = 1.05


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


def persist_checkpoint(unit: str, checkpoint: Path, root: Path, *, adapter, arch_signature: str,
                       num_parameters: int, inputs: dict, capacity: int | None = None) -> dict:
    src = identify_for_transfer(checkpoint, adapter=adapter, arch_signature=arch_signature,
                                num_parameters=num_parameters)
    unit_dir = root / unit
    unit_dir.mkdir(parents=True, exist_ok=True)
    desc_path = unit_dir / "checkpoint.descriptor.json"
    record = {"schema": SCHEMA, "unit": unit, "kind": "checkpoint", "started_utc": _now(),
              "source": str(checkpoint), "artifact_digest": src.artifact_digest,
              "weights_digest": src.weights_digest, "config_sha256": src.config_sha256,
              "arch_signature": arch_signature, "num_parameters": int(num_parameters),
              "tokenizer_sha256": src.tokenizer_sha256, "inputs": inputs,
              "destination": str(unit_dir / "checkpoint"), "persisted": False}
    desc_path.write_text(json.dumps(record, indent=1) + "\n")          # before any risky byte
    need = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    _room(root, need, capacity)
    partial = unit_dir / "checkpoint.partial"
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(checkpoint, partial)
    verdict = verify_transferred_leaf(partial, record, adapter=adapter)
    if not (verdict["matched"] and verdict["weights_digest_matched"] and verdict["config_matched"] is not False):
        raise LeafDurabilityError(f"{partial}: arrived bytes do not re-identify: {verdict}")
    final = unit_dir / "checkpoint"
    if final.exists():
        raise LeafDurabilityError(f"{final} already exists; a completed unit is never overwritten")
    partial.rename(final)
    record.update(persisted=True, finished_utc=_now(), bytes=need, verified=verdict)
    desc_path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def persist_files(unit: str, files: list[Path], root: Path, *, label: str = "evaluation",
                  capacity: int | None = None) -> dict:
    out = root / unit / label
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"name": f.name, "sha256": _sha256(f), "bytes": f.stat().st_size} for f in files]
    desc_path = root / unit / f"{label}.descriptor.json"
    record = {"schema": SCHEMA, "unit": unit, "kind": label, "started_utc": _now(),
              "files": rows, "destination": str(out), "persisted": False}
    desc_path.write_text(json.dumps(record, indent=1) + "\n")
    _room(root, sum(r["bytes"] for r in rows), capacity)
    for f, r in zip(files, rows):
        dest = out / f.name
        shutil.copy2(f, out / (f.name + ".partial"))
        if _sha256(out / (f.name + ".partial")) != r["sha256"]:
            raise LeafDurabilityError(f"{dest}: copied bytes differ from the source")
        (out / (f.name + ".partial")).rename(dest)
    record.update(persisted=True, finished_utc=_now())
    desc_path.write_text(json.dumps(record, indent=1) + "\n")
    return record


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
    f.add_argument("--capacity-bytes", type=int, default=None)
    args = ap.parse_args(argv)
    root = Path(args.durable_root)
    if args.cmd == "files":
        rec = persist_files(args.unit, [Path(p) for p in args.files], root, capacity=args.capacity_bytes)
    else:
        from aadistill.initialization.adapters import register_builtin_adapters
        from aadistill.initialization.specs.arch import get_adapter
        register_builtin_adapters()
        ident = json.loads(Path(args.init_record).read_text())["materialization_identity"]
        rec = persist_checkpoint(args.unit, Path(args.checkpoint), root, adapter=get_adapter("qwen3"),
                                 arch_signature=ident["arch_signature"], num_parameters=ident["num_parameters"],
                                 inputs={"initialization_record_sha256": _sha256(Path(args.init_record))},
                                 capacity=args.capacity_bytes)
    print(json.dumps({k: rec[k] for k in ("unit", "kind", "persisted", "destination")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
