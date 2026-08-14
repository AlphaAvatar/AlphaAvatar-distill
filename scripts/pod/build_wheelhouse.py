#!/usr/bin/env python3
"""Materialize the pod's dependency set as wheels, from the exact lock.

    PYTHONPATH=src python scripts/pod/build_wheelhouse.py --out <dir>
    PYTHONPATH=src python scripts/pod/build_wheelhouse.py --out <dir> --upload

Why this exists. Four of five host draws on 2026-08-14 died in `uv sync`, and
every one of them was the same thing: the pod resolves and downloads ~3.8 GiB of
wheels from PyPI on **every run**, over whatever network the drawn host happens
to have. A cold host is indistinguishable from a slow one until 8-28 minutes of
billed setup has gone. The one healthy host did the same sync in 45 seconds — a
40x spread on identical image, command and GPU type. $1.2712 bought zero driver
stages.

So the wheels move to the relay, which the pods read fast and which is already an
input path, and the pod installs with `--offline --no-index --find-links`. The
paid critical path then contains no PyPI.

**Exactness.** The lock is the source of truth, not a resolver run here: each
wheel is chosen from `uv.lock` by PEP-425 compatibility with the pod's
interpreter (cp312 / manylinux x86_64, recorded as `python_version 3.12.3` in
the run evidence) and verified against the lock's own sha256 after download. A
wheel whose hash does not match is not written.

**The cu128 rewrite is done here, once, and its lock is committed.** The setup
script used to `sed` the pytorch index and re-run `uv lock` on the pod, which is
a network resolve on the critical path and a resolution that nothing reviewed.
`uv-cu128.lock` is that resolution, frozen: same versions the pods actually ran
(torch 2.11.0+cu128, triton 3.6.0, setuptools 81.0.0).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The pod's interpreter, read from a real run's recorded runtime fingerprint
#: rather than assumed: `logs/autoinit_preflight_run4` records 3.12.3.
POD_PYTHON = (3, 12)
RELAY_REPO = "AlphaAvatar/aadistill-artifacts"
RELAY_PATH = "transfer/wheelhouse_cu128_cp312"


def compatible(filename: str) -> bool:
    """PEP 425: is this wheel installable on cp312 / manylinux x86_64?

    Deliberately explicit rather than delegating to `pip download --platform`,
    which rejected the whole NVIDIA stack: those wheels are tagged
    `py3-none-manylinux_2_27_x86_64`, which pip's platform filter would not
    accept alongside a `manylinux_2_28` request.
    """
    stem = filename[:-4] if filename.endswith(".whl") else filename
    parts = stem.split("-")
    if len(parts) < 5:
        return False
    pytag, abitag, platform = parts[-3], parts[-2], parts[-1]
    if platform != "any":
        if "x86_64" not in platform or "musllinux" in platform:
            return False
        if "manylinux" not in platform and "linux" not in platform:
            return False
    if abitag not in ("none", "abi3", f"cp{POD_PYTHON[0]}{POD_PYTHON[1]}"):
        return False

    def py_ok(tag: str) -> bool:
        if tag == "py2":
            return False
        if tag in ("py3", f"cp{POD_PYTHON[0]}{POD_PYTHON[1]}",
                   f"py{POD_PYTHON[0]}{POD_PYTHON[1]}"):
            return True
        # abi3 wheels built for an older minor run on a newer one.
        if tag.startswith("cp3") and abitag == "abi3":
            try:
                return int(tag[3:]) <= POD_PYTHON[1]
            except ValueError:
                return False
        if tag.startswith("py3"):
            try:
                return int(tag[3:]) <= POD_PYTHON[1]
            except ValueError:
                return False
        return False

    return any(py_ok(t) for t in pytag.split("."))


def select_from_pins(requirements: Path) -> list[dict]:
    """One wheel per pin, resolved from PyPI. For the vLLM environment.

    The train environment comes from a uv lock, which records URLs and hashes.
    vLLM's set has no lock in this repo — it is `uv pip compile`'d to exact pins
    (`requirements-vllm.txt`) — so the URL and hash come from the PyPI API for
    the pinned version. Same guarantees either way: exact version, exact file,
    hash verified after download, and a missing wheel raises rather than leaving
    the pod to reach the network for the remainder.
    """
    import json
    import re

    pins = []
    for line in requirements.read_text().splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9._-]+)==([^\s;]+)", line)
        if m:
            pins.append((m.group(1), m.group(2)))
    chosen, missing = [], []
    for name, version in pins:
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        req = urllib.request.Request(
            url, headers={"User-Agent": "aadistill-wheelhouse/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            meta = json.load(r)
        cands = [f for f in meta["urls"]
                 if f["filename"].endswith(".whl") and compatible(f["filename"])]
        if not cands:
            missing.append(f"{name}=={version}")
            continue
        w = cands[0]
        chosen.append({"name": name, "version": version, "url": w["url"],
                       "hash": f"sha256:{w['digests']['sha256']}"})
    if missing:
        raise SystemExit(
            f"no cp{POD_PYTHON[0]}{POD_PYTHON[1]}/manylinux wheel on PyPI for "
            f"{missing}. Refusing to build a wheelhouse that would send the pod "
            "back to PyPI for the remainder.")
    if len(chosen) != len(pins):
        raise SystemExit(f"resolved {len(chosen)} of {len(pins)} pins")
    return sorted(chosen, key=lambda c: c["name"])


def select(lock_path: Path, requirements: Path) -> list[dict]:
    """One wheel per required package, chosen from the lock. Fails closed."""
    lock = tomllib.load(lock_path.open("rb"))
    required = {line.split("==")[0].strip()
                for line in requirements.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
                and "==" in line}
    chosen, missing = [], []
    for pkg in lock["package"]:
        if pkg["name"] not in required:
            continue
        cands = [w for w in pkg.get("wheels", [])
                 if compatible(urllib.parse.unquote(w["url"].rsplit("/", 1)[-1]))]
        if not cands:
            missing.append(pkg["name"])
            continue
        # Several manylinux variants can match; any is installable, so take the
        # first deterministically rather than by size.
        chosen.append({"name": pkg["name"], "version": pkg["version"], **cands[0]})
    if missing:
        raise SystemExit(
            f"no cp{POD_PYTHON[0]}{POD_PYTHON[1]}/manylinux wheel in the lock for "
            f"{missing}. Refusing to build a wheelhouse that would send the pod "
            "back to PyPI for the remainder.")
    found = {c["name"] for c in chosen}
    if found != required:
        raise SystemExit(f"lock does not cover {sorted(required - found)}")
    return sorted(chosen, key=lambda c: c["name"])


def fetch(entry: dict, out: Path) -> tuple[Path, int]:
    """Download one wheel and verify it against the lock's own sha256."""
    name = urllib.parse.unquote(entry["url"].rsplit("/", 1)[-1])
    dest = out / name
    expect = (entry.get("hash") or "").removeprefix("sha256:")
    if dest.is_file() and expect:
        if hashlib.sha256(dest.read_bytes()).hexdigest() == expect:
            return dest, dest.stat().st_size
    # A plain `urlopen` gets HTTP 403 from download.pytorch.org, which rejects
    # urllib's default User-Agent. Retried because a transient failure here
    # would otherwise leave a wheelhouse that is silently short.
    req = urllib.request.Request(
        entry["url"], headers={"User-Agent": "aadistill-wheelhouse/1.0"})
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                blob = r.read()
            break
        except Exception as exc:                                  # noqa: BLE001
            last = exc
            time.sleep(3 * (attempt + 1))
    else:
        raise SystemExit(f"{name}: download failed after 4 attempts: {last}")
    if expect:
        got = hashlib.sha256(blob).hexdigest()
        if got != expect:
            raise SystemExit(f"{name}: sha256 {got} != lock's {expect}")
    dest.write_bytes(blob)
    return dest, len(blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default="uv-cu128.lock",
                    help="the frozen cu128 resolution; ignored with --from-pins")
    ap.add_argument("--from-pins", action="store_true",
                    help="resolve exact pins from PyPI instead of a uv lock "
                         "(the vLLM environment, which is uv-pip-compile'd)")
    ap.add_argument("--requirements", default=None,
                    help="exported requirements; derived from the lock if absent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upload", action="store_true",
                    help="push the wheelhouse to the relay after building")
    ap.add_argument("--relay-path", default=RELAY_PATH)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = REPO_ROOT / args.lock
    if args.from_pins:
        reqs = Path(args.requirements)
    elif args.requirements:
        reqs = Path(args.requirements)
    else:
        reqs = out / "requirements.txt"
        subprocess.run(
            ["uv", "export", "--frozen", "--group", "dev", "--no-emit-project",
             "--no-hashes", "--project", str(REPO_ROOT), "--locked" if False else
             "--frozen", "-o", str(reqs)],
            check=True, cwd=REPO_ROOT,
            env={**__import__("os").environ, "UV_LOCKFILE": str(lock)})

    entries = select_from_pins(reqs) if args.from_pins else select(lock, reqs)
    total = 0
    for i, entry in enumerate(entries, 1):
        path, size = fetch(entry, out)
        total += size
        print(f"[{i:3d}/{len(entries)}] {entry['name']:34s} "
              f"{size / 2**20:8.1f} MiB  {path.name}", flush=True)
    print(f"\n{len(entries)} wheels, {total / 2**30:.2f} GiB in {out}")

    if args.upload:
        import os
        from huggingface_hub import HfApi
        api = HfApi(token=open(os.path.expanduser(
            "~/.cache/huggingface/token")).read().strip())
        api.upload_folder(folder_path=str(out), repo_id=RELAY_REPO,
                          repo_type="model", path_in_repo=args.relay_path,
                          allow_patterns=["*.whl"])
        print(f"uploaded to {RELAY_REPO}:{args.relay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
