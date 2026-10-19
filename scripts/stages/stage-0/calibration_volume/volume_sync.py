#!/usr/bin/env python3
"""Mirror a working directory to the session's network volume, and back.

    python volume_sync.py push --src <dir> --dst <volume dir> [--loop 120 --until <file>]
    python volume_sync.py pull --src <volume dir> --dst <dir>

A long search keeps its journal and every materialized state on the pod's local
disk, where it runs at local-disk speed; this copies them to the persistent
volume as they appear, so a lost pod costs only the expansion in flight. A
replacement pod `pull`s the mirror back to the SAME path before the search
starts, and the engine's own resume re-identifies every restored checkpoint from
its bytes, so a copy that was cut short is declined and rebuilt, never adopted.

Files are only ever added or replaced (a newer size or mtime), never deleted,
and each lands under a temporary name first, so a reader never sees a partial
file under its real name. Plain Python: no rsync on the image is assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def sync(src: Path, dst: Path) -> dict:
    copied, skipped, nbytes = 0, 0, 0
    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial"):
            continue
        target = dst / path.relative_to(src)
        st = path.stat()
        if target.is_file():
            ts = target.stat()
            if ts.st_size == st.st_size and int(ts.st_mtime) == int(st.st_mtime):
                skipped += 1
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.partial")
        shutil.copyfile(path, tmp)
        os.utime(tmp, (st.st_atime, st.st_mtime))
        os.replace(tmp, target)
        copied += 1
        nbytes += st.st_size
    return {"copied": copied, "skipped": skipped, "bytes": nbytes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("push", "pull"))
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--loop", type=float, default=0.0,
                    help="seconds between passes; 0 = one pass")
    ap.add_argument("--until", type=Path, default=None,
                    help="with --loop: stop after the pass that follows this file appearing")
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args(argv)

    def log(entry: dict) -> None:
        line = json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry})
        print(line, flush=True)
        if args.log:
            with args.log.open("a") as f:
                f.write(line + "\n")

    if not args.src.is_dir():
        log({"mode": args.mode, "error": f"no source {args.src}"})
        return 0 if args.mode == "pull" else 1
    while True:
        stop = args.until is not None and args.until.exists()
        t0 = time.time()
        try:
            res = sync(args.src, args.dst)
            log({"mode": args.mode, **res, "seconds": round(time.time() - t0, 1)})
        except OSError as exc:      # a full or flaky volume must not kill the search
            log({"mode": args.mode, "error": repr(exc)[:300]})
        if not args.loop or stop:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    sys.exit(main())
