"""Retire the relay's session code bundles that nothing reads and git can rebuild. $0.

    PYTHONPATH=src:scripts .venv/bin/python scripts/maintenance/consolidation/retire_relay_bundles.py            # dry run
    PYTHONPATH=src:scripts .venv/bin/python scripts/maintenance/consolidation/retire_relay_bundles.py --execute

A session bundle (``transfer/<prefix>_<commit hex>.bundle``) is the transport of one session's code to its pod: a git
bundle of the session commit, staged before the launch and read once by that pod's setup. After the session it has no
consumer, and its content is the commit itself. The maintainer's brief of 2026-11-26: code packages, caches and
environment resources are kept by their real consumers and recoverability, not kept forever because they are not
checkpoints, and an environment still in use is never deleted.

A bundle is retired only when, re-derived here:

* its name ends in ``_<hex>.bundle`` and that hex prefixes EXACTLY ONE commit reachable from a pushed ref
  (``git rev-list --remotes``): the code state is on the remote and the bundle can be rebuilt
  (``bundle_transport.build_bundle``); a bundle whose commit is not there is KEPT -- it may be that state's only copy;
* no pod exists on the account (the live position): no setup is reading one;
* it is not under a protected prefix (the wheelhouses are the pods' environment, never touched here).

The tombstone record (``logs/maintenance/inventories/relay_bundle_tombstones_<date>.json``: path, size, LFS sha256,
commit, rule) is written BEFORE anything is deleted; the objects go by ``permanently_delete_lfs_files`` on their exact
``file_oid`` (ordinary deletion frees no quota), then the listing is read back.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

RELAY = "AlphaAvatar/aadistill-artifacts"
PROTECTED = ("transfer/wheelhouse_",)
NAME = re.compile(r"^transfer/[A-Za-z0-9_]+_([0-9a-f]{7,40})\.bundle$")


def token() -> str:
    return Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()


def pushed_commits() -> list[str]:
    out = subprocess.run(["git", "rev-list", "--remotes"], capture_output=True, text=True, cwd=REPO, check=True)
    return out.stdout.split()


def candidates(api) -> tuple[list[dict], list[dict]]:
    """(retire, keep): every bundle on the relay with its LFS object, commit and the reason it is kept."""
    commits = pushed_commits()
    files = [i for i in api.list_repo_tree(RELAY, path_in_repo="transfer", recursive=False, repo_type="model")
             if hasattr(i, "size") and i.path.endswith(".bundle")]
    retire, keep = [], []
    for f in files:
        lfs = getattr(f, "lfs", None)
        row = {"path": f.path, "size_bytes": int(f.size), "lfs_sha256": getattr(lfs, "sha256", None)}
        m = NAME.match(f.path)
        hits = [c for c in commits if m and c.startswith(m.group(1))]
        if any(f.path.startswith(p) for p in PROTECTED):
            keep.append({**row, "why": "protected prefix"})
        elif lfs is None:
            keep.append({**row, "why": "not an LFS object"})
        elif len(hits) != 1:
            keep.append({**row, "why": f"its commit is not exactly one pushed commit ({len(hits)} matches): it may "
                                       "be the only copy of that code state"})
        else:
            retire.append({**row, "commit": hits[0]})
    return retire, keep


def main(argv=None) -> int:
    from huggingface_hub import CommitOperationDelete, HfApi
    from shared import study_engineering as SE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    api = HfApi(token=token())
    retire, keep = candidates(api)
    gib = 2 ** 30
    print(f"{len(retire)} bundles, {sum(r['size_bytes'] for r in retire) / gib:.3f} GiB to retire; {len(keep)} kept "
          f"({sum(r['size_bytes'] for r in keep) / gib:.3f} GiB)")
    if not a.execute:
        return 0
    pos = SE.live_position(REPO)
    if pos.get("account_pods"):
        print(f"REFUSED: {len(pos['account_pods'])} pod(s) exist; a setup may be reading a bundle")
        return 2
    stamp = time.strftime("%Y%m%d", time.gmtime())
    rec_p = REPO / f"logs/maintenance/inventories/relay_bundle_tombstones_{stamp}.json"
    rec = {"schema": "aadistill.maintenance.relay_bundle_tombstones/v1", "relay": RELAY,
           "decided_by": "the maintainer's brief of 2026-11-26 (code packages kept by consumer and recoverability)",
           "rule": __doc__.split("A bundle is retired only when")[1].split("The tombstone")[0].strip(),
           "rebuild": "bundle_transport.build_bundle at the recorded commit (or: git bundle create <out>.bundle "
                      "<commit>); the commit is on the remote",
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "position_pods": 0,
           "retired": retire, "kept": keep, "status": "TOMBSTONED_BEFORE_DELETION"}
    rec_p.write_text(json.dumps(rec, indent=1) + "\n")
    oids = {r["lfs_sha256"] for r in retire}
    paths = [r["path"] for r in retire]
    for i in range(0, len(paths), 100):
        api.create_commit(RELAY, operations=[CommitOperationDelete(path_in_repo=p) for p in paths[i:i + 100]],
                          commit_message=f"retire {len(paths[i:i + 100])} session code bundles (tombstone "
                                         f"{rec_p.name})", repo_type="model")
    still_referenced = {getattr(getattr(i, "lfs", None), "sha256", None)
                        for i in api.list_repo_tree(RELAY, recursive=True, repo_type="model") if hasattr(i, "size")}
    objs = [o for o in api.list_lfs_files(RELAY) if o.file_oid in oids - still_referenced]
    for i in range(0, len(objs), 100):
        api.permanently_delete_lfs_files(RELAY, objs[i:i + 100], rewrite_history=True)
    left = [o.file_oid for o in api.list_lfs_files(RELAY) if o.file_oid in oids]
    present = [p for p in paths if p in {i.path for i in api.get_paths_info(RELAY, paths, repo_type="model")}]
    rec.update(status="RETIRED" if not left and not present else "INCOMPLETE",
               lfs_objects_deleted=len(objs), lfs_left=left, paths_left=present,
               shared_oids_kept=sorted(oids & still_referenced),
               freed_bytes=sum(int(o.size) for o in objs if o.file_oid not in left),
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rec_p.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("status", "lfs_objects_deleted", "freed_bytes")} | {"left": len(left)}))
    return 0 if rec["status"] == "RETIRED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
