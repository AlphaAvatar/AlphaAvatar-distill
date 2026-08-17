#!/usr/bin/env python3
"""Prove, file by file, that a local tree is mirrored on the relay before deleting it.

    PYTHONPATH=src python scripts/consolidate/verify_relay_mirror.py \
        --local /home/ecs-user/aad-artifacts/wheelhouse_vllm_cp312 \
        --relay-prefix transfer/wheelhouse_vllm_cp312 \
        --out logs/relay_mirror_verification.json

"The remote copy has been hash-verified" is the only clause that lets a local
cache be deleted, so it needs evidence rather than an assertion. This produces
that evidence, and it is deliberately awkward about one thing:

**A Hugging Face repository does not expose a sha256 for every file.** Objects
stored through LFS carry their content sha256 as the LFS oid, and that can be
compared without downloading anything. Files below the LFS threshold are ordinary
git blobs, and their `blob_id` is a git **sha1 over a different byte string** —
comparing it to a sha256 proves nothing. Those files are downloaded and hashed.
The report records which method covered which file, because a verification that
does not say how it verified is not one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

RELAY_REPO = "AlphaAvatar/aadistill-artifacts"


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", required=True)
    ap.add_argument("--relay-prefix", required=True)
    ap.add_argument("--repo", default=RELAY_REPO)
    ap.add_argument("--out", default="logs/relay_mirror_verification.json")
    ap.add_argument("--max-download-bytes", type=int, default=256 << 20,
                    help="refuse to download more than this to verify; a mirror "
                         "that needs a large download is not cheap evidence")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download     # noqa: PLC0415

    local_root = Path(args.local)
    if not local_root.is_dir():
        print(f"no local tree at {args.local}", file=sys.stderr)
        return 1

    api = HfApi()
    info = api.repo_info(args.repo, repo_type="model", files_metadata=True)
    remote = {f.rfilename: f for f in (info.siblings or [])}

    local_files = sorted(p for p in local_root.rglob("*") if p.is_file())
    by_lfs, to_download, missing, mismatched = [], [], [], []
    for p in local_files:
        rel = p.relative_to(local_root).as_posix()
        rpath = f"{args.relay_prefix}/{rel}"
        f = remote.get(rpath)
        if f is None:
            missing.append(rel)
            continue
        if f.lfs and f.lfs.sha256:
            by_lfs.append((rel, rpath, f.lfs.sha256, p))
        else:
            to_download.append((rel, rpath, f.size or 0, p))

    checked = []
    for rel, rpath, remote_sha, p in by_lfs:
        got = sha256_file(p)
        ok = got == remote_sha
        checked.append({"file": rel, "method": "lfs_oid", "local_sha256": got,
                        "remote_sha256": remote_sha, "match": ok})
        if not ok:
            mismatched.append(rel)

    download_bytes = sum(sz for _, _, sz, _ in to_download)
    downloaded = 0
    if download_bytes > args.max_download_bytes:
        print(f"REFUSED: verifying the non-LFS files needs "
              f"{download_bytes / 2**20:.1f} MiB of download, over the "
              f"{args.max_download_bytes / 2**20:.1f} MiB limit", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as td:
        for rel, rpath, sz, p in to_download:
            got_local = sha256_file(p)
            fetched = hf_hub_download(args.repo, rpath, repo_type="model",
                                      local_dir=td)
            got_remote = sha256_file(Path(fetched))
            downloaded += sz
            ok = got_local == got_remote
            checked.append({"file": rel, "method": "downloaded_and_hashed",
                            "local_sha256": got_local,
                            "remote_sha256": got_remote, "match": ok})
            if not ok:
                mismatched.append(rel)

    verified = all(c["match"] for c in checked) and not missing
    report = {
        "schema": "aadistill.relay_mirror_verification/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "local_tree": args.local,
        "relay": args.repo,
        "relay_prefix": args.relay_prefix,
        "n_local_files": len(local_files),
        "local_bytes": sum(p.stat().st_size for p in local_files),
        "verified_by_lfs_oid": sum(1 for c in checked if c["method"] == "lfs_oid"),
        "verified_by_download": sum(1 for c in checked
                                    if c["method"] == "downloaded_and_hashed"),
        "downloaded_bytes": downloaded,
        "missing_on_relay": missing,
        "mismatched": mismatched,
        "verified": verified,
        "method_note": ("an LFS oid IS the content sha256, so those files are "
                        "verified without transfer. A non-LFS file's blob_id is a "
                        "git sha1 over a different byte string and proves nothing, "
                        "so those are downloaded and hashed."),
        "meaning": ("verified=true means every local file has a byte-identical "
                    "copy at the relay prefix. It does NOT mean the local copy is "
                    "safe to delete: that also requires the retention rule in "
                    "logs/checkpoint_registry.json to permit it."),
        "files": sorted(checked, key=lambda c: c["file"]),
    }
    out = REPO_ROOT / args.out
    doc = json.loads(out.read_text()) if out.is_file() else {"verifications": []}
    doc["verifications"] = [v for v in doc.get("verifications", [])
                            if v.get("local_tree") != args.local] + [report]
    out.write_text(json.dumps(doc, indent=2) + "\n")

    print(json.dumps({k: report[k] for k in
                      ("n_local_files", "verified_by_lfs_oid",
                       "verified_by_download", "missing_on_relay", "mismatched",
                       "verified")}, indent=2))
    print(f"-> {args.out}")
    return 0 if verified else 3


if __name__ == "__main__":
    raise SystemExit(main())
