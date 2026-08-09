#!/usr/bin/env python3
"""Stage the E7 extra-KD streams to the relay and verify them from the relay side.

Before E7 spends anything, nothing on its critical path may be dev-box-only. The
three streams are currently single-copy local artifacts; a disk failure or a
mistaken `rm` would cost the FineWeb download, the tokenization and the packing,
and — worse — would make the built streams unreproducible byte-for-byte if
FineWeb ever re-tags.

This uploads them, then **downloads them back and re-hashes**. An upload that
reports success is not evidence: the check that matters is that the bytes on the
relay hash to what the local manifest says, verified by fetching them again.

    PYTHONPATH=src python scripts/data/stage_e7_streams.py --verify-roundtrip

Exit codes: 0 staged and verified; 9 a hash mismatch or a missing file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.infrastructure.manifest import sha256_file  # noqa: E402

RELAY = "AlphaAvatar/aadistill-artifacts"
REPO_TYPE = "model"
PREFIX = "e7_streams_20260809"
STREAMS = ("e7_fineweb_kd", "e7_control_kd", "e7_fineweb_val")
FILES = ("blocks.npz", "docs.jsonl", "manifest.json")


def local_hashes(stream_dir: Path) -> dict:
    return {f: sha256_file(stream_dir / f) for f in FILES}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="artifacts/stage3")
    ap.add_argument("--relay", default=RELAY)
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--verify-roundtrip", action="store_true",
                    help="re-download and re-hash; an upload that returns 200 "
                         "is not evidence")
    ap.add_argument("--out", default="logs/e7_relay_manifest.json")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    root = REPO_ROOT / args.root
    record: dict = {
        "staged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "relay": args.relay, "repo_type": REPO_TYPE, "prefix": args.prefix,
        "streams": {},
    }
    failures: list[str] = []

    for name in STREAMS:
        d = root / name
        if not d.is_dir():
            failures.append(f"{name}: not built locally at {d}")
            continue
        manifest = json.loads((d / "manifest.json").read_text())
        want = local_hashes(d)
        # The manifest's own declared hashes must agree with the files on disk
        # before anything is uploaded; otherwise the relay copy would be pinned
        # to a manifest that never described it.
        for key, fname in (("blocks", "blocks.npz"), ("docs", "docs.jsonl")):
            declared = manifest["outputs"][key]
            if declared != want[fname]:
                failures.append(
                    f"{name}/{fname}: on disk {want[fname][:16]}… but the "
                    f"manifest declares {declared[:16]}…")

        for fname in FILES:
            path_in_repo = f"{args.prefix}/{name}/{fname}"
            api.upload_file(
                path_or_fileobj=str(d / fname), path_in_repo=path_in_repo,
                repo_id=args.relay, repo_type=REPO_TYPE,
                commit_message=f"E7: stage {name}/{fname}")
            print(f"uploaded {path_in_repo}", flush=True)

        entry = {"local_dir": str(d.relative_to(REPO_ROOT)),
                 "relay_prefix": f"{args.prefix}/{name}",
                 "sha256": want,
                 "n_blocks": manifest["n_blocks"],
                 "block_len": manifest["block_len"],
                 "kd_positions": manifest["kd_positions"],
                 "padding_tokens": manifest["padding_tokens"],
                 "kind": manifest["kind"]}

        if args.verify_roundtrip:
            got = {}
            for fname in FILES:
                fetched = hf_hub_download(
                    repo_id=args.relay, repo_type=REPO_TYPE,
                    filename=f"{args.prefix}/{name}/{fname}",
                    force_download=True)
                got[fname] = sha256_file(fetched)
            entry["roundtrip_sha256"] = got
            entry["roundtrip_verified"] = got == want
            if got != want:
                mismatched = [f for f in FILES if got[f] != want[f]]
                failures.append(f"{name}: roundtrip mismatch on {mismatched}")
            print(f"  roundtrip {'OK' if got == want else 'MISMATCH'}", flush=True)

        record["streams"][name] = entry

    record["failures"] = failures
    record["staged"] = not failures
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "streams"}, indent=2))
    print(f"-> {out}")
    if failures:
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 9
    print("all three streams are on the relay and hash-verified from it; "
          "no E7 critical-path dependency is dev-box-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
