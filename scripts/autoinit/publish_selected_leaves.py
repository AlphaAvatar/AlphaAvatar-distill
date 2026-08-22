"""Publish the Attempt-12 selected leaves to a private TRANSPORT repo. $0.

    PYTHONPATH=src python scripts/autoinit/publish_selected_leaves.py --upload
    PYTHONPATH=src python scripts/autoinit/publish_selected_leaves.py --verify

**Why a second repo.** Continuation attempt 2 died staging the first leaf: the
launcher pushes every `LOCAL_ASSET` by scp under a hard-coded 600 s timeout, and
one 1.110 GiB leaf needs 1.99 MB/s to fit that window against a dev box observed
at 0.44-0.72 MB/s. The main relay cannot take them either — it reported 1.60 GiB
of headroom against 5.55 GiB of leaves.

A second private repo was **measured** to work: 2026-08-13 recorded that "a 1 MiB
write to a different private repo succeeded, so the limit binds per-repo, not
account-wide". That is the non-destructive path, and it inverts the transfer that
failed: instead of the dev box PUSHING 5.55 GiB while a pod bills, the pod PULLS
from the hub at hub speed, and the slow half happens here at $0.

**This repo is transport only.** The canonical checkpoints stay at
`/home/ecs-user/aad-artifacts/autoinit/phase_a/<state_id>`. Nothing here becomes
a scientific owner: every identity is taken from attempt 12's committed
durability record and merely *reproduced* at the far end.

Verification is deliberately belt-and-braces, because a transport that silently
corrupts a leaf would be discovered by a paid session:

1. **before** — every file's size and sha256, recomputed from the canonical local
   bytes, checked against the attempt-12 record;
2. **remote** — the hub's own LFS sha256 OID for each uploaded shard, compared to
   the local digest without downloading anything;
3. **round trip** — every file downloaded to a temporary directory and passed
   through `verify_transferred_leaf`, the same function the pod uses, which
   rebuilds `artifact_digest` from the bytes that actually arrived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.arch import get_adapter  # noqa: E402
from aadistill.autoinit.leaf_durability import verify_transferred_leaf  # noqa: E402

#: Transport only. Private. Not a scientific owner.
TRANSPORT_REPO = "AlphaAvatar/aadistill-transport"
#: Paths identify the attempt AND the state id, so a future reader can tell what
#: these bytes are without consulting anything else.
PREFIX = "phase_a_attempt12/selected_leaves"

CANONICAL_STORE = Path("/home/ecs-user/aad-artifacts/autoinit/phase_a")
EVIDENCE = REPO_ROOT / "logs/autoinit_phase_a_attempt12/selected_leaf_durability.json"
MANIFEST = REPO_ROOT / "logs/autoinit_selected_leaf_transport_manifest.json"

#: The configured scratch root, when the operator names one. The dev box keeps
#: session working directories under `/home/ecs-user/aad-scratch`, which is where
#: a multi-GiB round trip belongs on that host.
SCRATCH_ENV = "AAD_SCRATCH"
DEV_BOX_SCRATCH = Path("/home/ecs-user/aad-scratch")


def scratch_dir() -> str | None:
    """Where `verify()` puts its round-trip temporaries.

    `None` means "let `tempfile` choose", which is the whole point: this returns
    a *preference*, never a requirement.

    Until 2026-08-22 the parent was the literal `/home/ecs-user/aad-scratch`,
    passed straight to `mkdtemp`. That directory does not exist on a pod, so
    `mkdtemp` raised `FileNotFoundError` — and although this module is a dev-box
    publishing tool the paid session never runs, its **tests** are part of the
    suite the pod's setup gate executes. Five of them reach this line, and
    recovery continuation attempt 3 died there at $0.2011 with the transport
    itself already proven.

    So the host default is a preference used only when it is actually there.
    Order: the configured root if it names an existing directory, then the dev
    box's own root if present, then the platform temporary directory.
    """
    configured = os.environ.get(SCRATCH_ENV, "").strip()
    if configured and Path(configured).is_dir():
        return configured
    if DEV_BOX_SCRATCH.is_dir():
        return str(DEV_BOX_SCRATCH)
    return None


def token() -> str:
    """The Hugging Face token, environment first.

    The same portability defect as `scratch_dir()`, one line further down and
    found the same way. Every pod-side script in `autoinit_preflight_setup.sh`
    reads `HF_TOKEN`, which the setup exports before the test gate runs; a pod
    has no `~/.cache/huggingface/token`. This is evaluated as an *argument* to
    `hf_hub_download`, so it runs even in tests that patch the download away —
    which is exactly how it surfaced once `mkdtemp` stopped failing first.

    The file remains the dev-box path, so nothing about local use changes.
    """
    env = os.environ.get("HF_TOKEN", "").strip()
    if env:
        return env
    return Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def leaves() -> list[dict]:
    """The five, in the frozen selected order the record stores them in."""
    return json.loads(EVIDENCE.read_text())["leaves"]


def remote_path(state_id: str, filename: str) -> str:
    return f"{PREFIX}/{state_id}/{filename}"


def build_before() -> dict:
    """What we are about to send, measured from the canonical bytes."""
    records = []
    for order, rec in enumerate(leaves()):
        sid = rec["state_id"]
        src = CANONICAL_STORE / sid
        if not src.is_dir():
            raise SystemExit(f"canonical leaf missing: {src}")
        files = []
        for p in sorted(src.iterdir()):
            if not p.is_file():
                continue
            files.append({"filename": p.name, "size_bytes": p.stat().st_size,
                          "sha256": sha256_file(p),
                          "remote_path": remote_path(sid, p.name)})
        ident = rec["identity"]
        # The shard digest in the record must be reproduced by the local bytes,
        # or the canonical copy has drifted and nothing downstream is meaningful.
        shard = next(f for f in files if f["filename"].endswith(".safetensors"))
        if shard["sha256"] != ident["single_shard_sha256"]:
            raise SystemExit(
                f"{sid}: local shard {shard['sha256']} does not match the "
                f"attempt-12 record {ident['single_shard_sha256']}")
        records.append({
            "selected_order": order,
            "state_id": sid,
            "canonical_source": str(src),
            "artifact_digest": ident["artifact_digest"],
            "single_shard_sha256": ident["single_shard_sha256"],
            "config_sha256": ident["config_sha256"],
            "arch_signature": ident["arch_signature"],
            "total_bytes": ident["total_bytes"],
            "remote_repo": TRANSPORT_REPO,
            "files": files})
    return {"schema": "aadistill.autoinit.leaf_transport_manifest/v1",
            "role": ("TRANSPORT ONLY. The canonical checkpoints remain at "
                     f"{CANONICAL_STORE}; this repo is a delivery path and is "
                     "not a scientific owner."),
            "repo": TRANSPORT_REPO, "prefix": PREFIX,
            "source_record": str(EVIDENCE.relative_to(REPO_ROOT)),
            "n_leaves": len(records),
            "total_bytes": sum(f["size_bytes"] for r in records for f in r["files"]),
            "leaves": records}


def upload(man: dict) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    api.create_repo(TRANSPORT_REPO, repo_type="model", private=True,
                    exist_ok=True)
    print(f"repo ready: {TRANSPORT_REPO} (private)", flush=True)
    present = set(api.list_repo_files(TRANSPORT_REPO, repo_type="model"))
    for rec in man["leaves"]:
        for f in rec["files"]:
            if f["remote_path"] in present:
                print(f"  already there: {f['remote_path']}", flush=True)
                continue
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(Path(rec["canonical_source"]) / f["filename"]),
                path_in_repo=f["remote_path"], repo_id=TRANSPORT_REPO,
                repo_type="model")
            dt = time.time() - t0
            print(f"  {f['remote_path']}  {f['size_bytes'] / 2**20:.1f} MiB in "
                  f"{dt / 60:.1f} min ({f['size_bytes'] / max(dt, 1e-9) / 1e6:.2f} MB/s)",
                  flush=True)


def remote_oids(man: dict) -> dict:
    """The hub's own content hash per file, without downloading the bytes."""
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    out = {}
    for item in api.list_repo_tree(TRANSPORT_REPO, path_in_repo=PREFIX,
                                   repo_type="model", recursive=True,
                                   expand=True):
        if getattr(item, "size", None) is None:
            continue                                   # a directory entry
        lfs = getattr(item, "lfs", None)
        out[item.path] = {
            "size_bytes": item.size,
            # LFS objects carry a sha256 OID; small non-LFS files do not, and
            # are covered by the round trip below instead of being assumed.
            "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None}
    return out


def verify(man: dict, *, keep: bool = False) -> dict:
    from huggingface_hub import hf_hub_download

    problems: list[str] = []
    oids = remote_oids(man)

    # -- 1. every declared file is present, with the right size ------------
    for rec in man["leaves"]:
        for f in rec["files"]:
            got = oids.get(f["remote_path"])
            if got is None:
                problems.append(f"{f['remote_path']}: absent from the transport repo")
                continue
            if got["size_bytes"] != f["size_bytes"]:
                problems.append(
                    f"{f['remote_path']}: {got['size_bytes']} bytes remote vs "
                    f"{f['size_bytes']} local")
            if got["lfs_sha256"] and got["lfs_sha256"] != f["sha256"]:
                problems.append(
                    f"{f['remote_path']}: LFS oid {got['lfs_sha256']} != "
                    f"local {f['sha256']}")

    # -- 2. round trip, one leaf at a time, through the pod's own verifier --
    adapter = get_adapter("qwen3")
    by_id = {r["state_id"]: r for r in json.loads(EVIDENCE.read_text())["leaves"]}
    round_trip = []
    tmp = Path(tempfile.mkdtemp(prefix="leaf-roundtrip-", dir=scratch_dir()))
    try:
        for rec in man["leaves"]:
            sid = rec["state_id"]
            into = tmp / sid
            into.mkdir(parents=True, exist_ok=True)
            for f in rec["files"]:
                cached = hf_hub_download(TRANSPORT_REPO, f["remote_path"],
                                         repo_type="model", token=token(),
                                         cache_dir=str(tmp / "hf"))
                shutil.copy(cached, into / f["filename"])
                got = sha256_file(into / f["filename"])
                if got != f["sha256"]:
                    problems.append(f"{f['remote_path']}: round-trip sha256 {got}")
            try:
                v = verify_transferred_leaf(into, by_id[sid], adapter=adapter)
                ok = bool(v["matched"] and v["shard_matched"])
                round_trip.append({"state_id": sid, "matched": v["matched"],
                                   "shard_matched": v["shard_matched"],
                                   "artifact_digest": v.get("artifact_digest")})
                if not ok:
                    problems.append(f"{sid}: round-trip identity did not reproduce")
                print(f"  round trip {sid[:12]}: matched={v['matched']} "
                      f"shard={v['shard_matched']}", flush=True)
            except Exception as exc:                          # noqa: BLE001
                problems.append(f"{sid}: {type(exc).__name__}: {exc}")
            finally:
                if not keep:
                    shutil.rmtree(into, ignore_errors=True)
                    shutil.rmtree(tmp / "hf", ignore_errors=True)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)

    return {"problems": problems, "round_trip": round_trip,
            "remote_files": len(oids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--keep", action="store_true",
                    help="keep the round-trip download (debugging only)")
    args = ap.parse_args()

    man = build_before()
    print(f"{man['n_leaves']} leaves, "
          f"{man['total_bytes'] / 2**30:.2f} GiB, -> {TRANSPORT_REPO}", flush=True)

    if args.upload:
        upload(man)
    if args.verify:
        result = verify(man, keep=args.keep)
        man["verification"] = result
        man["verified"] = not result["problems"]
        MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
        print(json.dumps({"verified": man["verified"],
                          "problems": result["problems"],
                          "remote_files": result["remote_files"]}, indent=2))
        if result["problems"]:
            raise SystemExit(1)
    elif not args.upload:
        print(json.dumps(man, indent=2)[:4000])


if __name__ == "__main__":
    main()
