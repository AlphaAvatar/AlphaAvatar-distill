#!/usr/bin/env python3
"""The D2 search's committed Top-4: verified, persisted, restorable. `$0`.

    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/stages/stage-1/phase_d2/d2_product_durability.py verify-local
    ... upload          # the four checkpoints + the evidence archive -> relay
    ... verify-remote   # LFS oids, then a full download round trip
    ... record          # all three, written to the run's evidence

**Why.** The four initializations exist as one dev-box copy
(`<scr>/products/<state_id>/`). They are the starting points of every D2
behavioural probe, so until that evidence exists and is reproducible they have a
consumer and are not retirable (AGENTS.md 2.5). A single local copy is one disk
away from a re-run of the search, and the screening pod cannot reach them by
scp (0.72 MB/s uplink, 600 s per-asset cap) -- so they go to the private relay
once, at `$0`, and the pod PULLS them at hub speed.

**What is checked, from bytes rather than records:**

* the evidence archive hashes to the closeout's `archive_sha256`; re-expanded
  into an INDEPENDENT directory, every file the session manifest lists --
  `states.jsonl` included -- hashes to the manifest's sha256 and size;
* the committed selection verifies its own `selection_sha256`, and the
  archive's copy is byte-identical to the one in git;
* each product's four identities (`artifact_digest`, `weights_digest`,
  `single_shard_sha256`, `arch_signature`) are recomputed from its bytes by the
  same function the behavioural pods use, and equal the selection's;
* after upload, the hub's LFS oid and size per file equal the local ones, and a
  download of every file into another independent directory re-identifies to
  the same four identities (and the archive re-expands to the same manifest).

Preservation is not permission (P8.2.1): nothing here authorizes reusing these
bytes for anything; the screening grant does that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
for _extra in ("src", "scripts", "scripts/stages/stage-1"):
    _p = str(REPO_ROOT / _extra)
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: THIS search's identity. Instance facts, so they live here and not in core.
RUN_ID = "d2_search_20261015_051722"
RUN_REL = f"logs/stages/stage-1/phase_d2/runs/{RUN_ID}"
SELECTION_REL = f"{RUN_REL}/evidence/stage1_selection.json"
MANIFEST_REL = f"{RUN_REL}/evidence/manifest.json"
CLOSEOUT_REL = f"{RUN_REL}/closeout/outcome.json"
RECORD_REL = f"{RUN_REL}/evidence/product_durability.json"
#: The dev-box durable store the closeout names as `<scr>`.
DEFAULT_SCR = Path("/home/ecs-user/aad-artifacts/phase_d2") / RUN_ID
ARCHIVE_NAME = "d2_search_artifacts.tar.gz"
#: Where the archive's paths are rooted: the pod's `artifacts/` tree.
ARCHIVE_PREFIX = "audit/autoinit_d2"
#: The files a checkpoint directory carries. Weight-only by design: a searched
#: leaf has no tokenizer, and adding one would move its artifact digest.
CHECKPOINT_FILES = ("model.safetensors", "config.json", "generation_config.json")
#: The relay prefix: names the experiment and the run, so a reader of the
#: relay can tell what these bytes are without consulting anything else.
RELAY_PREFIX = f"stage1/phase_d2/{RUN_ID}"
SCHEMA = "aadistill.phase_d2.product_durability/v1"


class DurabilityError(RuntimeError):
    """A durability premise does not hold. Nothing is written over it."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(rel: str, repo_root: Path = REPO_ROOT) -> Any:
    return json.loads((repo_root / rel).read_text())


def selected_rows(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """The committed Top-4, self-hash verified, in its own order."""
    from aadistill.initialization.planning import stage1_selection

    rows = list(stage1_selection.load(repo_root / SELECTION_REL)["selected"])
    if len(rows) != 4:
        raise DurabilityError(f"the committed selection names {len(rows)} "
                              "leaves; the preregistration commits 4")
    return rows


def relay_files(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    """Every object this module puts on the relay: (local rel, remote path)."""
    out = [{"local": f"products/{row['state_id']}/{name}",
            "remote": f"{RELAY_PREFIX}/products/{row['state_id']}/{name}",
            "state_id": row["state_id"]}
           for row in selected_rows(repo_root) for name in CHECKPOINT_FILES]
    out.append({"local": f"store/{ARCHIVE_NAME}",
                "remote": f"{RELAY_PREFIX}/{ARCHIVE_NAME}", "state_id": ""})
    return out


def identify(directory: Path, row: dict[str, Any]) -> dict[str, Any]:
    """The four identities recomputed from bytes, compared EXACTLY.

    D1's behavioural pods verify their arms with this function; using it here
    means "verified at rest" and "verified on the pod" are one predicate.
    """
    from stages.phase_d1.behavioural_materialize import (
        D1MaterializeError, _verify_bytes_at,
    )

    want = {k: row[k] for k in ("artifact_digest", "weights_digest",
                                "single_shard_sha256", "arch_signature")}
    try:
        got = _verify_bytes_at(directory, want, what=row["state_id"])
    except D1MaterializeError as exc:
        raise DurabilityError(str(exc)) from exc
    return {"directory": str(directory), "matched": True,
            "identities": {k: got[k] for k in want}}


def verify_archive(archive: Path, *, workdir: Path,
                   repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Re-expand the archive somewhere NEW and check every manifest entry."""
    closeout = _read(CLOSEOUT_REL, repo_root)
    want_sha = closeout["evidence_in_tree"]["out_of_tree"]["archive_sha256"]
    if not archive.is_file():
        raise DurabilityError(f"{archive} is missing")
    got_sha = _sha256(archive)
    if got_sha != want_sha:
        raise DurabilityError(f"{archive} hashes to {got_sha}, the closeout "
                              f"records {want_sha}")
    dest = workdir / "archive"
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest.resolve()) or not (
                    member.isfile() or member.isdir()):
                raise DurabilityError(f"{member.name}: not a plain member "
                                      "inside the archive root; refusing")
        tar.extractall(dest)
    manifest = _read(MANIFEST_REL, repo_root)
    rows = []
    for entry in manifest["entries"]:
        path = dest / entry["path"]
        ok = (path.is_file() and path.stat().st_size == entry["size_bytes"]
              and _sha256(path) == entry["sha256"])
        rows.append({"path": entry["path"], "sha256": entry["sha256"],
                     "size_bytes": entry["size_bytes"], "matched": ok})
    bad = [r["path"] for r in rows if not r["matched"]]
    if bad:
        raise DurabilityError(f"re-expanded archive disagrees with the "
                              f"session manifest on {bad}")
    selection_bytes = (dest / ARCHIVE_PREFIX / "stage1_selection.json").read_bytes()
    if selection_bytes != (repo_root / SELECTION_REL).read_bytes():
        raise DurabilityError("the archive's stage1_selection.json is not "
                              "byte-identical to the committed one")
    states = next(r for r in rows if r["path"].endswith("search/states.jsonl"))
    return {"archive": str(archive), "archive_sha256": got_sha,
            "archive_bytes": archive.stat().st_size,
            "re_expanded_into": "an independent temporary directory",
            "manifest_entries": rows, "all_matched": True,
            "states_jsonl": {"sha256": states["sha256"],
                             "size_bytes": states["size_bytes"],
                             "matched": True},
            "selection_byte_identical_to_git": True}


def verify_local(scr: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="d2-durability-") as tmp:
        archive = verify_archive(scr / "store" / ARCHIVE_NAME,
                                 workdir=Path(tmp), repo_root=repo_root)
    products = []
    for row in selected_rows(repo_root):
        directory = scr / "products" / row["state_id"]
        missing = [n for n in CHECKPOINT_FILES if not (directory / n).is_file()]
        if missing:
            raise DurabilityError(f"{row['state_id']}: {missing} missing at "
                                  f"{directory}")
        products.append({
            "state_id": row["state_id"],
            "files": {n: {"sha256": _sha256(directory / n),
                          "size_bytes": (directory / n).stat().st_size}
                      for n in CHECKPOINT_FILES},
            **identify(directory, row)})
    return {"scr": str(scr), "archive": archive, "products": products,
            "checked_utc": _now()}


def _token() -> str:
    return Path("~/.cache/huggingface/token").expanduser().read_text().strip()


def _relay() -> str:
    from shared.deployment import MAIN_RELAY

    return MAIN_RELAY


def capacity(scr: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Would the relay accept these exact objects? Asked before uploading."""
    from aadistill.runtime.hub_capacity import would_accept

    sizes = [(scr / f["local"]).stat().st_size for f in relay_files(repo_root)]
    verdict = would_accept(_relay(), sizes, _token())
    return {"accepted": verdict.accepted, "detail": str(verdict),
            "n_objects": len(sizes), "total_bytes": sum(sizes)}


def remote_listing() -> dict[str, dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    out = {}
    try:
        items = list(api.list_repo_tree(_relay(), path_in_repo=RELAY_PREFIX,
                                        repo_type="model", recursive=True,
                                        expand=True))
    except Exception as exc:                                 # noqa: BLE001
        if "404" in str(exc) or "not found" in str(exc).lower():
            return {}
        raise
    for item in items:
        if getattr(item, "size", None) is None:
            continue
        lfs = getattr(item, "lfs", None)
        out[item.path] = {"size_bytes": item.size,
                          "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
                          "blob_id": getattr(item, "blob_id", None)}
    return out


def upload(scr: Path, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Each object once; an object already present with the same oid is kept."""
    from huggingface_hub import HfApi

    room = capacity(scr, repo_root)
    print(f"relay capacity: {room['detail']}", flush=True)
    present = remote_listing()
    api = HfApi(token=_token())
    done = []
    for f in relay_files(repo_root):
        local = scr / f["local"]
        sha = _sha256(local)
        have = present.get(f["remote"])
        if have and have["size_bytes"] == local.stat().st_size and (
                have["lfs_sha256"] in (None, sha)):
            print(f"  already there: {f['remote']}", flush=True)
            done.append({**f, "uploaded": False, "already_present": True})
            continue
        if not room["accepted"]:
            raise DurabilityError(f"the relay refuses these objects: "
                                  f"{room['detail']}")
        t0 = time.time()
        api.upload_file(path_or_fileobj=str(local), path_in_repo=f["remote"],
                        repo_id=_relay(), repo_type="model",
                        commit_message=f"{RUN_ID}: {f['local']} (durability)")
        dt = time.time() - t0
        size = local.stat().st_size
        print(f"  {f['remote']}  {size / 2**20:.1f} MiB in {dt / 60:.1f} min "
              f"({size / max(dt, 1e-9) / 1e6:.2f} MB/s)", flush=True)
        done.append({**f, "uploaded": True, "seconds": round(dt, 1)})
    return done


def verify_remote(scr: Path, repo_root: Path = REPO_ROOT, *,
                  scratch: Path | None = None) -> dict[str, Any]:
    """Hub oids, then EVERY file downloaded somewhere new and re-identified."""
    from huggingface_hub import hf_hub_download

    listing = remote_listing()
    files = relay_files(repo_root)
    problems, oids = [], []
    for f in files:
        local = scr / f["local"]
        have = listing.get(f["remote"])
        row = {"remote": f["remote"], "present": have is not None}
        if have is None:
            problems.append(f"{f['remote']}: absent from {_relay()}")
        else:
            sha = _sha256(local)
            row.update({"size_bytes": have["size_bytes"],
                        "lfs_sha256": have["lfs_sha256"],
                        "size_matches_local": have["size_bytes"]
                        == local.stat().st_size,
                        "lfs_sha256_matches_local": (
                            None if have["lfs_sha256"] is None
                            else have["lfs_sha256"] == sha)})
            if not row["size_matches_local"] or row["lfs_sha256_matches_local"] \
                    is False:
                problems.append(f"{f['remote']}: remote differs from local")
        oids.append(row)
    if problems:
        raise DurabilityError("; ".join(problems))
    tmp = Path(tempfile.mkdtemp(prefix="d2-restore-", dir=scratch))
    try:
        restored = []
        rows = {r["state_id"]: r for r in selected_rows(repo_root)}
        for state_id, row in rows.items():
            into = tmp / "products" / state_id
            into.mkdir(parents=True)
            for name in CHECKPOINT_FILES:
                got = hf_hub_download(
                    _relay(), f"{RELAY_PREFIX}/products/{state_id}/{name}",
                    repo_type="model", token=_token(),
                    cache_dir=str(tmp / "hf"))
                shutil.copy(got, into / name)
                Path(got).unlink(missing_ok=True)
            restored.append({"state_id": state_id, **identify(into, row)})
            shutil.rmtree(into)
        got = hf_hub_download(_relay(), f"{RELAY_PREFIX}/{ARCHIVE_NAME}",
                              repo_type="model", token=_token(),
                              cache_dir=str(tmp / "hf"))
        archive = verify_archive(Path(got), workdir=tmp / "expand",
                                 repo_root=repo_root)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"relay": _relay(), "prefix": RELAY_PREFIX, "objects": oids,
            "restored_checkpoints": restored,
            "restored_archive": {k: archive[k] for k in (
                "archive_sha256", "all_matched", "states_jsonl")},
            "restore_dir": "an independent temporary directory, deleted after",
            "checked_utc": _now()}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(scr: Path, repo_root: Path = REPO_ROOT, *,
           scratch: Path | None = None) -> dict[str, Any]:
    local = verify_local(scr, repo_root)
    remote = verify_remote(scr, repo_root, scratch=scratch)
    rows = selected_rows(repo_root)
    body = {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "recorded_utc": _now(),
        "selection_sha256": _read(SELECTION_REL, repo_root)["selection_sha256"],
        "locations": {
            "dev_host_store": {
                "path": str(scr),
                "holds": ["products/<state_id>/{model.safetensors,config.json,"
                          "generation_config.json}",
                          f"store/{ARCHIVE_NAME}"],
                "kind": "local disk, single copy"},
            "relay": {
                "repo": remote["relay"], "prefix": RELAY_PREFIX,
                "kind": "private Hugging Face LFS (the pods' relay)",
                "holds": [f["remote"] for f in relay_files(repo_root)]},
        },
        "local_verification": local,
        "relay_verification": remote,
        "checkpoints": [{
            "quality_order": i + 1, "state_id": r["state_id"],
            "artifact_digest": r["artifact_digest"],
            "weights_digest": r["weights_digest"],
            "single_shard_sha256": r["single_shard_sha256"],
            "arch_signature": r["arch_signature"],
            "relay_paths": [f"{RELAY_PREFIX}/products/{r['state_id']}/{n}"
                            for n in CHECKPOINT_FILES],
        } for i, r in enumerate(rows)],
        "durability": ("two independent copies: the dev-box store and the "
                       "private relay; each verified from its own bytes"),
        "recoverability": (
            "either copy restores a checkpoint byte-exactly; a restore is "
            "accepted only when all four identities recompute equal "
            "(stages.phase_d1.behavioural_materialize._verify_bytes_at)"),
        "consumers": [
            {"consumer": "D2 behavioural screening (8 probes: these 4 x seeds "
                         "516448251, 614923639)",
             "reads": "each checkpoint as a probe's student_path, from the "
                      "relay on the pod",
             "status": "declared; not authorized to run"},
            {"consumer": "D2 shared confirmation of the advancing candidate, "
                         "if screening advances one",
             "reads": "that one checkpoint",
             "status": "contingent; not authorized"},
        ],
        "retention": ("reproducibility_required: NOT retirable under the "
                      "standing weight-retirement rule (AGENTS.md 2.5) while "
                      "any consumer above is open"),
        "_preservation_is_not_permission": (
            "AGENTS.md P8.2.1: this record authorizes nothing about reusing "
            "the bytes; the screening grant does"),
    }
    body["record_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("verify-local", "capacity", "upload",
                                       "verify-remote", "record"))
    ap.add_argument("--scr", default=str(DEFAULT_SCR))
    ap.add_argument("--scratch", default=None,
                    help="where the restore round trip downloads to")
    ap.add_argument("--write", action="store_true",
                    help="with `record`: write it to the run's evidence")
    args = ap.parse_args(argv)
    scr = Path(args.scr)
    scratch = Path(args.scratch) if args.scratch else None
    if args.action == "verify-local":
        out = verify_local(scr)
    elif args.action == "capacity":
        out = capacity(scr)
    elif args.action == "upload":
        out = upload(scr)
    elif args.action == "verify-remote":
        out = verify_remote(scr, scratch=scratch)
    else:
        out = record(scr, scratch=scratch)
        if args.write:
            dest = REPO_ROOT / RECORD_REL
            dest.write_text(json.dumps(out, indent=1) + "\n")
            print(f"wrote {RECORD_REL}")
    print(json.dumps(out, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
