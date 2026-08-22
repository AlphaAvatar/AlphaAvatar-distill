"""Retire REMOTE relay copies whose canonical local copy is verified. $0.

    PYTHONPATH=src python scripts/autoinit/retire_relay_copies.py            # dry run
    PYTHONPATH=src python scripts/autoinit/retire_relay_copies.py --execute

**This retires a remote COPY, not a checkpoint.** Every object removed here has a
byte-identical copy under `/home/ecs-user/aad-artifacts`, verified by content
hash at the moment of deletion. The scientific artifact continues to exist; what
changes is that it stops being mirrored on Hugging Face. Nothing in the
experiment records, frozen identities or accountability metadata is touched.

**Why.** The private-storage limit is account-wide (measured 2026-08-22:
93.279 GiB against ~93.13), so the five Attempt-12 selected leaves — 5.55 GiB —
cannot be mirrored for transport while obsolete copies hold the quota. The
maintainer decided explicitly that obsolete remote redundancy is not worth
blocking Phase A.

**How, and why not the obvious way.** Ordinary file deletion reclaims *nothing*
on this repo — measured twice (2026-08-02, deleting 19.07 GB freed nothing;
2026-08-14, a 2.23 GiB delete left billed `usedStorage` unchanged for 45 min).
Hugging Face bills LFS including history, so the only reclaim is
`permanently_delete_lfs_files(..., rewrite_history=True)` on an exact object.
Objects are therefore selected **by `file_oid` identity, never by directory
prefix**, and every safety property is re-asserted here rather than inherited
from the table that proposed them.

The gotcha that reads as "not found": the LFS content hash is the object's
`file_oid`, **not** its `oid`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

RELAY = "AlphaAvatar/aadistill-artifacts"
TRANSPORT = "AlphaAvatar/aadistill-transport"
CANON = Path("/home/ecs-user/aad-artifacts")

#: Never removable, whatever a table says. Re-derived here so a mistake in the
#: candidate list cannot reach the API.
PROTECTED_PREFIXES = (
    "transfer/wheelhouse_cu128_cp312/",   # offline uv sync, paid critical path
    "transfer/wheelhouse_vllm_cp312/",    # offline vLLM venv, paid critical path
    "permanent_controls/",                # the two permanent controls
    "stage1/qwen3_0p6b_init_v0/",         # canonical initialization
    "stage3_recovery_corpus_v2/ladder_uniform/",
    "e8_inputs_20260810/calibration_v1/",
)


def token() -> str:
    return Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_remote_paths() -> set[str]:
    """Every remote path any CURRENT session declares, from the real specs."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "pod"))
    sys.path.insert(0, str(REPO_ROOT / "tests" / "pod"))
    from session_specs import all_specs

    return {r.path for _n, _m, _a, spec in all_specs()
            for r in spec.setup.relay_inputs}


def check(obj, declared: set[str]) -> tuple[bool, str, str | None]:
    """Every condition, re-derived. Returns (ok, why, canonical_local_path)."""
    fn = obj.filename
    if fn in declared:
        return False, "a current session declares this remote path", None
    if any(fn.startswith(p) for p in PROTECTED_PREFIXES):
        return False, "protected prefix", None
    # The canonical copy must exist and match, under the canonical store only.
    matches = []
    for root in (CANON,):
        for p in root.rglob("*"):
            try:
                if p.is_file() and p.stat().st_size == obj.size:
                    if sha256(p) == obj.file_oid:
                        matches.append(str(p))
                        break
            except OSError:
                continue
        if matches:
            break
    if not matches:
        return False, "NO canonical local copy with this content hash", None
    return True, "canonical local copy verified by content hash", matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--table", default=None,
                    help="candidate table from the analysis step (filenames only; "
                         "every property is re-checked here)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=token())
    declared = declared_remote_paths()
    wanted = None
    if args.table:
        wanted = {r["filename"] for r in json.load(open(args.table))["rows"]
                  if r.get("deletable")}

    plan, refused = [], []
    for obj in api.list_lfs_files(RELAY, repo_type="model"):
        if wanted is not None and obj.filename not in wanted:
            continue
        ok, why, local = check(obj, declared)
        (plan if ok else refused).append((obj, why, local))

    print(f"declared remote paths in current sessions: {len(declared)}")
    for obj, why, _ in refused:
        print(f"  REFUSED {obj.filename}: {why}")
    total = 0
    for obj, why, local in plan:
        total += obj.size
        print(f"  RETIRE  {obj.size / 2**30:7.3f} GiB  {obj.filename}")
        print(f"          oid {obj.file_oid[:16]}…  local {local}")
    print(f"\n{len(plan)} objects, {total / 2**30:.3f} GiB")

    if not args.execute:
        print("\nDRY RUN — nothing deleted. Re-run with --execute.")
        return
    if not plan:
        print("nothing to do")
        return
    api.permanently_delete_lfs_files(RELAY, [o for o, _, _ in plan],
                                     rewrite_history=True, repo_type="model")
    print(f"retired {len(plan)} remote copies from {RELAY}")


if __name__ == "__main__":
    main()
