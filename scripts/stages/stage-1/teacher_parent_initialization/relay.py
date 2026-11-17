"""The round's durable copies on the private relay: put (quota-checked, re-downloaded and re-hashed), get (re-hashed),
and the permanent removal of a superseded save point. Every call is online and runs in its own process when the
caller's process has ``HF_HUB_OFFLINE=1`` set (huggingface_hub reads it at import): `run_online`.

A record is ``{"relay", "prefix", "files_sha256": {rel: sha}, "bytes": {rel: n}, "verified", ...}``; `get` refuses a
file whose bytes are not the record's. Nothing here names a unit or a run: the caller passes the prefix.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

POD_TOKEN = Path("/workspace/hf/token")
ROOT_PREFIX = "durable/stage1_teacher_parent_init"


class RelayError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _token(token_file: Path | None = None) -> str | None:
    for p in (token_file, POD_TOKEN, Path("~/.cache/huggingface/token").expanduser()):
        if p is not None and Path(p).is_file():
            return Path(p).read_text().strip()
    return None


def _repo(relay: str | None) -> str:
    if relay:
        return relay
    from shared.deployment import MAIN_RELAY
    return MAIN_RELAY


def put(files: dict[str, Path], prefix: str, *, message: str, relay: str | None = None,
        token_file: Path | None = None, scratch: Path | None = None) -> dict:
    """Upload ``files`` (relative name -> local path) under ``prefix``, then download each again into a fresh cache and
    re-hash it. Refused (never raised) when the relay's quota does not admit the bytes: ``verified`` is False."""
    from huggingface_hub import HfApi, hf_hub_download
    from aadistill.runtime.hub_capacity import would_accept
    repo, token = _repo(relay), _token(token_file)
    want = {k: sha256_file(Path(p)) for k, p in files.items()}
    sizes = {k: Path(p).stat().st_size for k, p in files.items()}
    out = {"relay": repo, "prefix": prefix, "files_sha256": want, "bytes": sizes}
    api = HfApi(token=token)
    there = {i.path: i for i in api.get_paths_info(repo, [f"{prefix}/{k}" for k in files], repo_type="model")}
    todo = [k for k in sorted(files) if f"{prefix}/{k}" not in there
            or (getattr(there[f"{prefix}/{k}"], "lfs", None) is not None
                and there[f"{prefix}/{k}"].lfs.sha256 != want[k])]
    if todo:
        verdict = would_accept(repo, [sizes[k] for k in todo], token)
        out["quota"] = str(verdict)
        if not verdict.accepted:
            return {**out, "verified": False, "refused": "the relay's storage quota does not admit the bytes"}
    t0 = time.time()
    for k in todo:
        api.upload_file(path_or_fileobj=str(files[k]), path_in_repo=f"{prefix}/{k}", repo_id=repo,
                        repo_type="model", commit_message=f"{message}: {k}")
    out["uploaded"], out["upload_seconds"] = todo, round(time.time() - t0, 1)
    t1, got = time.time(), {}
    with tempfile.TemporaryDirectory(dir=str(scratch) if scratch else None) as tmp:
        for k in sorted(files):
            got[k] = sha256_file(Path(hf_hub_download(repo, f"{prefix}/{k}", repo_type="model", token=token,
                                                      cache_dir=f"{tmp}/hf", force_download=True)))
            shutil.rmtree(f"{tmp}/hf", ignore_errors=True)
    return {**out, "downloaded_sha256": got, "verified": got == want, "verify_seconds": round(time.time() - t1, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rule": "uploaded, downloaded again into a fresh cache and re-hashed"}


def get(record: dict, dest: Path, *, only: tuple[str, ...] | None = None, token_file: Path | None = None) -> dict:
    """Every file of ``record`` (or ``only`` those) into ``dest/<rel>``, atomically, each re-hashed against it."""
    from huggingface_hub import hf_hub_download
    repo, base, token = record["relay"], record["prefix"], _token(token_file)
    if not record.get("verified"):
        raise RelayError(f"{base}: the record was never verified")
    done = {}
    for k, want in sorted(record["files_sha256"].items()):
        if only is not None and k not in only:
            continue
        target = Path(dest) / k
        if target.is_file() and sha256_file(target) == want:
            done[k] = "present"
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cache = Path(dest) / ".hf_relay"
        src = Path(hf_hub_download(repo, f"{base}/{k}", repo_type="model", token=token, cache_dir=str(cache)))
        tmp = target.with_name(f".{target.name}.partial")
        shutil.copyfile(src, tmp)
        got = sha256_file(tmp)
        if got != want:
            tmp.unlink(missing_ok=True)
            raise RelayError(f"{k} from {repo}:{base} hashes to {got[:12]}, the record says {want[:12]}")
        os.replace(tmp, target)
        shutil.rmtree(cache, ignore_errors=True)
        done[k] = "downloaded"
    return {"from": f"{repo}:{base}", "files": done, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def remove(record: dict, *, message: str, token_file: Path | None = None) -> dict:
    """Delete a superseded record's paths and permanently delete their LFS objects (only permanent deletion frees the
    quota), then read the listing back. Used for a unit's PREVIOUS save point after the next one verified."""
    from huggingface_hub import CommitOperationDelete, HfApi
    repo, base, token = record["relay"], record["prefix"], _token(token_file)
    api = HfApi(token=token)
    paths = [f"{base}/{k}" for k in sorted(record["files_sha256"])]
    there = [i.path for i in api.get_paths_info(repo, paths, repo_type="model")]
    if there:
        api.create_commit(repo, operations=[CommitOperationDelete(path_in_repo=p) for p in there],
                          commit_message=message, repo_type="model")
    oids = set(record["files_sha256"].values())
    objs = [o for o in api.list_lfs_files(repo) if o.file_oid in oids]
    if objs:
        api.permanently_delete_lfs_files(repo, objs, rewrite_history=True)
    left = [o.file_oid for o in api.list_lfs_files(repo) if o.file_oid in oids]
    return {"removed_paths": there, "lfs_objects_deleted": len(objs), "lfs_left": left,
            "verified_absent": not left, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run_online(op: str, payload: dict, timeout_s: float = 3600) -> dict:
    """`put` / `get` / `remove` in a child process with the hub online; one JSON line back. Never raises: a failure is
    ``{"verified": False, "error": ...}``."""
    try:
        res = subprocess.run([sys.executable, str(Path(__file__).resolve()), op, json.dumps(payload)],
                             env={**os.environ, "HF_HUB_OFFLINE": "0"}, capture_output=True, text=True,
                             timeout=timeout_s)
        lines = [x for x in (res.stdout or "").splitlines() if x.startswith("{")]
        return json.loads(lines[-1]) if lines else {"verified": False, "error": (res.stderr or "")[-2000:]}
    except Exception as exc:  # noqa: BLE001
        return {"verified": False, "error": f"{type(exc).__name__}: {str(exc)[:1000]}"}


def _main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[4]
    for p in ("src", "scripts", "scripts/stages/stage-1"):
        if str(repo_root / p) not in sys.path:
            sys.path.insert(0, str(repo_root / p))
    op, payload = argv[0], json.loads(argv[1])
    try:
        if op == "put":
            out = put({k: Path(v) for k, v in payload["files"].items()}, payload["prefix"],
                      message=payload["message"], scratch=Path(payload["scratch"]) if payload.get("scratch") else None)
        elif op == "get":
            out = {**get(payload["record"], Path(payload["dest"]), only=tuple(payload["only"]) if payload.get("only")
                         else None), "verified": True}
        elif op == "remove":
            out = remove(payload["record"], message=payload["message"])
            out["verified"] = out["verified_absent"]
        else:
            raise RelayError(f"unknown op {op!r}")
    except Exception as exc:  # noqa: BLE001
        out = {"verified": False, "error": f"{type(exc).__name__}: {str(exc)[:1500]}"}
    print(json.dumps(out, default=str))
    return 0 if out.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
