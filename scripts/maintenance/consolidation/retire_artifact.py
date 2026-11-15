#!/usr/bin/env python3
"""Delete a local artifact or one relay object, but only behind a tombstone that outlives it.

    PYTHONPATH=src python scripts/maintenance/consolidation/retire_artifact.py \
        --path artifacts/audit/ladder_uniform_rebuild \
        --id ladder_uniform_rebuild \
        --criterion byte_identical_duplicate \
        --experiment E1 --role "rebuilt ladder for the mixture audit" \
        --reason "..." --canonical "artifacts/shared/instruments/ladder_uniform_probe" \
        --frozen-results "artifacts/audit/e1_mixture_rebuild.json" \
        --reconstruction "scripts/stages/stage-3/e1/audit_e1_mixture_rebuild.py …" \
        --reconstruction-cost '$0 (CPU)' \
        --apply

Nothing is removed without `--apply`. Everything else about the run happens
either way, so a dry run produces the exact tombstone that would be written.

The six criteria are the only reasons a deletion is allowed. They are spelled
out here rather than left to judgement at the keyboard:

    unreferenced_failed_partial   an interrupted attempt nothing points at
    byte_identical_duplicate      identical bytes to a copy that survives
    reproducible_materialization  deterministically rebuildable at a stated cost
    rejected_search_leaf          a searched-and-dropped state whose lineage,
                                  hashes, probe results and selection evidence
                                  are already recorded elsewhere
    verified_stale_cache          the canonical remote copy has been
                                  hash-verified, file by file
    agents_2_5_standing_retirement
                                  AGENTS.md 2.5: terminal, evidence and hashes
                                  durable, no consumer, not an anchor

How the bytes could come back is three separate facts, never one flag:

    --reconstruction / --reconstruction-kind
                     a recipe and what it is: `retrain` (a new training run: new weights, not these bytes),
                     `rebuild` (a computation from preserved inputs), `copy` (fetch a surviving copy) or `none`
    --byte-identical whether the recipe gives back THESE bytes: `verified` (shown by an earlier rebuild whose hash
                     matched; `--byte-identical-evidence` says where), `not_verified`, or `not_expected` (a retrain)
    --same-hash-copy where a byte-identical copy survives (a local one is re-hashed)

`deterministic_reconstruction_possible` is derived: true only when `--byte-identical verified`. Until 2026-11-14 it
was `bool(--reconstruction)`, so any recipe -- a retrain included -- read as a deterministic rebuild.

A same-hash copy is a fact only when it is checked, and every declared one is (`verify_copies`):

  * it exists: a missing path or relay object is refused, never skipped;
  * it is complete: a file target is matched by size and sha256; a directory target member by member (relative
    path, size, sha256), so a copy missing or differing in one file is refused;
  * it is outside the deletion scope: the target itself, a path inside it, a directory that contains it, or a hard
    link to the same inode is not a second copy; for a relay target, the same repository's file with the same sha256
    IS the target (one LFS object; the permanent deletion removes it under every path);
  * a remote copy (``relay:<path>`` or ``hf:<owner>/<repo>:<path>``) is checked by the object the hub serves: its LFS
    sha256 and size (a small non-LFS file is downloaded and hashed).

Two targets: ``--path`` (local; the bytes are removed with ``rmtree``/``unlink``) or ``--relay-object`` (one LFS
file of the private relay, matched by ``file_oid`` == the listing's sha256; the path is deleted from HEAD and the
object permanently deleted with ``permanently_delete_lfs_files(rewrite_history=True)``, then the listing is read
again). A relay object whose sha256 another path of the repository shares is refused: the deletion would remove
both. ``--consumer-audit <json>`` attaches the object's entry of an `audit_relay_weights.py` record (collected
facts and reviewed disposition) and refuses unless that entry says ``retire`` and was not refused.

Refusals, which are the point of the tool:

  * the path is `protected` in logs/maintenance/inventories/checkpoint_registry.json, or carries a
    `never_delete_clause` -- unless `--maintainer-decision` quotes the recorded maintainer decision that
    retires it (logs/budget/decisions.md); the tombstone then carries the decision and the class and
    clause it lifted (`protection_lifted`);
  * `--canonical` names something that does not exist;
  * a criterion of `byte_identical_duplicate` without one verified same-hash copy (above);
  * a criterion of `verified_stale_cache` without `--relay-verified`;
  * a recipe without its kind, `--byte-identical verified` without evidence, without a recipe, for a recipe of kind
    `none` or for a retrain, a `reproducible_materialization` whose rebuild is not verified byte-identical, and any
    declared same-hash copy that fails `verify_copies`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts"):
    if str(REPO_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _p))

TOMBSTONES = REPO_ROOT / "logs/maintenance/inventories/checkpoint_tombstones.json"
REGISTRY = REPO_ROOT / "logs/maintenance/inventories/checkpoint_registry.json"

CRITERIA = {
    "unreferenced_failed_partial": "an interrupted or failed attempt that nothing "
                                   "references",
    "byte_identical_duplicate": "byte-identical to a copy that survives",
    "reproducible_materialization": "a temporary materialization that is "
                                    "deterministically reproducible",
    "rejected_search_leaf": "a rejected search leaf whose lineage, hashes, probe "
                            "results and selection evidence are already preserved",
    "verified_stale_cache": "a stale local cache whose canonical remote artifact "
                            "has been hash-verified",
    "agents_2_5_standing_retirement": "a terminal historical checkpoint whose evidence "
                                      "and hashes are durable and that no consumer or "
                                      "anchor needs (AGENTS.md 2.5)",
}


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def absolute(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def manifest(root: Path) -> tuple[list[dict], int]:
    """Every file under root with its size and hash, so the tombstone can say
    exactly what stopped existing."""
    files = sorted(f for f in ([root] if root.is_file() else root.rglob("*"))
                   if f.is_file())
    out, total = [], 0
    for f in files:
        size = f.stat().st_size
        total += size
        out.append({"path": str(f.relative_to(REPO_ROOT))
                    if str(f).startswith(str(REPO_ROOT)) else str(f),
                    "size_bytes": size, "sha256": sha256_file(f)})
    return out, total


RECONSTRUCTION_KINDS = ("retrain", "rebuild", "copy", "none")
BYTE_IDENTICAL = ("verified", "not_verified", "not_expected")


def reconstruction_fields(recipe: str, kind: str | None, byte_identical: str | None, evidence: str,
                          copies: list, criterion: str, sha256: str | None = None) -> dict:
    """The tombstone's reconstruction fields, kept apart: a recipe and its kind, whether it gives back these bytes,
    and where a same-hash copy survives (``copies``: `verify_copies`' records, or location strings when the caller has
    nothing to verify). Raises ValueError on an inconsistent combination. No I/O: the copies are checked by
    `verify_copies` before this is called."""
    if recipe and not kind:
        raise ValueError("--reconstruction needs --reconstruction-kind (retrain, rebuild, copy or none)")
    kind = kind or "none"
    if byte_identical is None:
        byte_identical = "not_expected" if kind == "retrain" else "not_verified"
    if byte_identical == "verified":
        if not evidence.strip():
            raise ValueError("--byte-identical verified needs --byte-identical-evidence: the rebuild whose hash matched")
        if kind == "retrain":
            raise ValueError("a retrain does not give back the same bytes; it is a new checkpoint")
        if not recipe.strip() or kind == "none":
            raise ValueError("--byte-identical verified needs the recipe that rebuilds the bytes (--reconstruction "
                             "with kind rebuild or copy): a verified rebuild of nothing is not a fact")
    if criterion == "reproducible_materialization" and byte_identical != "verified":
        raise ValueError("reproducible_materialization needs a rebuild verified byte-identical")
    if criterion == "byte_identical_duplicate" and not copies:
        raise ValueError("byte_identical_duplicate needs --same-hash-copy: one existing, matching copy outside the "
                         "deletion scope")
    return {"reconstruction_recipe": recipe or None, "reconstruction_kind": kind,
            "byte_identical_reconstruction": byte_identical,
            "byte_identical_evidence": evidence.strip() or None,
            "same_hash_copies": copies,
            #: derived, never set from the presence of a recipe
            "deterministic_reconstruction_possible": byte_identical == "verified"}


def parse_copy(spec: str) -> dict:
    """A copy's location: ``relay:<path>``, ``hf:<owner>/<repo>:<path>`` or a local path (anything after the first
    space is a note)."""
    loc, _, note = spec.strip().partition(" ")
    if loc.startswith("relay:"):
        from shared.deployment import MAIN_RELAY
        return {"kind": "hub", "repo": MAIN_RELAY, "path": loc[len("relay:"):].strip("/"), "spec": spec, "note": note}
    if loc.startswith("hf:"):
        repo, _, path = loc[len("hf:"):].partition(":")
        if not repo or not path:
            raise ValueError(f"--same-hash-copy {spec!r}: hf:<owner>/<repo>:<path>")
        return {"kind": "hub", "repo": repo, "path": path.strip("/"), "spec": spec, "note": note}
    return {"kind": "local", "path": absolute(loc), "spec": spec, "note": note}


def hub_files(repo: str, paths: list[str]) -> dict[str, dict]:
    """``{path: {"sha256", "size", "lfs"}}`` for the paths the hub serves (a missing path is absent). A non-LFS file is
    downloaded and hashed (small by construction)."""
    import os
    from huggingface_hub import HfApi, hf_hub_download
    tok_p = Path(os.path.expanduser("~/.cache/huggingface/token"))
    tok = tok_p.read_text().strip() if tok_p.is_file() else None
    out = {}
    for f in HfApi(token=tok).get_paths_info(repo, paths, expand=True):
        if not hasattr(f, "size"):
            continue                                   #: a folder is not a file copy
        lfs = getattr(f, "lfs", None)
        if lfs is not None:
            out[f.path] = {"sha256": lfs.sha256, "size": int(lfs.size), "lfs": True}
        else:
            local = Path(hf_hub_download(repo, f.path, token=tok))
            out[f.path] = {"sha256": sha256_file(local), "size": local.stat().st_size, "lfs": False}
    return out


def verify_copies(specs: list[str], members: list[dict], *, target: Path | None = None,
                  target_relay: tuple[str, str] | None = None, hub=hub_files) -> list[dict]:
    """Every declared same-hash copy, checked (module docstring). ``members``: the target's files as
    ``{"rel", "size_bytes", "sha256"}``, ``rel`` = "" for a single-file target. Returns one record per copy; raises
    ValueError naming every copy that fails."""
    out, bad = [], []
    single = len(members) == 1 and members[0]["rel"] == ""
    #: every file of the deletion scope, by (device, inode): a copy member that IS one of them -- a hard link, or a
    #: symbolic link (stat follows it) -- is the same physical bytes, whichever member it stands in for
    scope: set[tuple[int, int]] = set()
    if target is not None and target.exists():
        for x in ([target] if target.is_file() else [y for y in target.rglob("*") if y.is_file()]):
            st = x.stat()
            scope.add((st.st_dev, st.st_ino))
    for spec in specs:
        c = parse_copy(spec)
        why = []
        if c["kind"] == "local":
            p = c["path"]
            if not p.exists():
                bad.append(f"{spec}: nothing there")
                continue
            rp = p.resolve()
            tr = target.resolve() if target is not None else None
            if target is not None:
                if rp == tr or tr in rp.parents or rp in tr.parents:
                    bad.append(f"{spec}: inside the deletion scope of {target}")
                    continue
            for m in members:
                f = p if single else p / m["rel"]
                if not f.is_file():
                    why.append(f"{m['rel'] or f.name} missing")
                    continue
                if target is not None:
                    fr, st = f.resolve(), f.stat()
                    if fr == tr or tr in fr.parents:
                        why.append(f"{m['rel'] or f.name} resolves into the deletion scope ({fr}) through a "
                                   "symbolic link")
                        continue
                    if (st.st_dev, st.st_ino) in scope:
                        why.append(f"{m['rel'] or f.name} is a hard link to a file of the target (one physical copy)")
                        continue
                if f.stat().st_size != m["size_bytes"] or sha256_file(f) != m["sha256"]:
                    why.append(f"{m['rel'] or f.name} differs")
            rec = {"location": str(p), "kind": "local"}
        else:
            if target_relay is not None and c["repo"] == target_relay[0]:
                bad.append(f"{spec}: the same repository's LFS object as the target, which its permanent deletion "
                           "removes")
                continue
            want = {(c["path"] if single else f"{c['path']}/{m['rel']}"): m for m in members}
            got = hub(c["repo"], sorted(want))
            for path, m in want.items():
                g = got.get(path)
                if g is None:
                    why.append(f"{path} not on {c['repo']}")
                elif g["size"] != m["size_bytes"] or g["sha256"] != m["sha256"]:
                    why.append(f"{path} on {c['repo']} differs")
            rec = {"location": f"{c['repo']}:{c['path']}", "kind": "hub"}
        if why:
            bad.append(f"{spec}: " + "; ".join(why))
            continue
        out.append({**rec, "verified": "every member's size and sha256 match", "members": len(members),
                    **({"note": c["note"]} if c["note"] else {})})
    if bad:
        raise ValueError("--same-hash-copy refused: " + " | ".join(bad))
    return out


def registry_guard(rel: str, relay_path: str | None = None) -> tuple[bool, str, dict | None]:
    if not REGISTRY.is_file():
        return True, "no registry to check against", None
    reg = json.loads(REGISTRY.read_text())
    for e in reg.get("checkpoints", []):
        local_hit = rel and (e["path_local"] == rel or rel.startswith(e["path_local"] + "/")
                             or e["path_local"].startswith(rel.rstrip("/") + "/"))
        rp = e.get("relay_path") or ""
        relay_hit = relay_path and rp and (rp == relay_path or relay_path.startswith(rp.rstrip("/") + "/"))
        if (local_hit or relay_hit) and e.get("protected"):
            return False, (f"{e['path_local']} is protected in the registry "
                           f"({e['retention']}): {e.get('never_delete_clause') or e['retention_reason']}"), \
                {"path_local": e["path_local"], "retention": e["retention"],
                 "never_delete_clause": e.get("never_delete_clause"), "retention_reason": e.get("retention_reason")}
    return True, "not protected in the registry", None


GENERIC_NAMES = {"model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "model", "checkpoint"}


def distinguishing_name(rel: str) -> str:
    """The deepest path component that names this artifact rather than every checkpoint (``model.safetensors``,
    ``model`` and ``step_*`` name thousands)."""
    for part in reversed(Path(rel).parts):
        if part not in GENERIC_NAMES and not part.startswith("step_"):
            return part
    return Path(rel).name


def references(rel: str, name: str) -> list[str]:
    hits = []
    for d in ("logs", "configs", "tests", "scripts", "src", "docs", "README.md"):
        base = REPO_ROOT / d
        files = [base] if base.is_file() else [
            f for f in base.rglob("*") if f.is_file()
            and "__pycache__" not in f.parts
            and f.suffix in (".md", ".json", ".py", ".txt", ".sh")]
        for f in files:
            try:
                text = f.read_text(errors="ignore")
            except Exception:
                continue
            if rel in text or (name and name in text):
                hits.append(str(f.relative_to(REPO_ROOT)))
    return sorted(set(hits))[:16]


def consumer_audit_entry(path: str, obj: str) -> dict:
    """The object's entry of an `audit_relay_weights.py` record; refuses unless it was reviewed `retire` and nothing
    the audit collected contradicts that."""
    doc = json.loads(absolute(path).read_text())
    e = (doc.get("objects") or {}).get(obj)
    if e is None:
        raise ValueError(f"{path} does not audit {obj}")
    d = e.get("disposition") or {}
    if d.get("disposition") != "retire":
        raise ValueError(f"{path} does not decide `retire` for {obj}: {d.get('disposition')!r} {d.get('reason', '')}")
    if e.get("refused"):
        raise ValueError(f"{path} refused {obj}: {e['refused']}")
    return {"record": path, "collected_utc": doc.get("collected_utc"), "sha256": e.get("sha256"),
            "registry": e.get("registry"), "launchers": e.get("launchers"),
            "code_readers": d.get("readers"), "closed_launchers": d.get("closed_launchers"),
            "same_sha256_paths": e.get("same_sha256_paths"), "evidence": d.get("evidence")}


def relay_object(repo: str, path: str):
    """The relay file at ``path`` (its LFS listing entry) and every repository path that shares its sha256."""
    import os
    from huggingface_hub import HfApi
    tok_p = Path(os.path.expanduser("~/.cache/huggingface/token"))
    api = HfApi(token=tok_p.read_text().strip() if tok_p.is_file() else None)
    info = api.model_info(repo, files_metadata=True)
    sha = {s.rfilename: (s.lfs.sha256 if getattr(s, "lfs", None) else None) for s in info.siblings}
    if sha.get(path) is None:
        raise ValueError(f"{repo}:{path} is not an LFS file of the repository")
    objs = [o for o in api.list_lfs_files(repo) if o.file_oid == sha[path]]
    if len(objs) != 1:
        raise ValueError(f"{len(objs)} LFS objects carry {sha[path][:12]}; exactly one expected")
    shared = sorted(q for q, h in sha.items() if h == sha[path] and q != path)
    return api, objs[0], shared


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    target_g = ap.add_mutually_exclusive_group(required=True)
    target_g.add_argument("--path", help="a local artifact to retire")
    target_g.add_argument("--relay-object", help="one LFS file of the private relay to retire (its path in the repo)")
    ap.add_argument("--id", required=True, help="canonical_id for the tombstone")
    ap.add_argument("--criterion", required=True, choices=sorted(CRITERIA))
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--canonical", required=True,
                    help="the surviving evidence: a path, a relay path, or a "
                         "record that carries the same fact")
    ap.add_argument("--frozen-results", default="",
                    help="where the conclusion that depended on this lives now")
    ap.add_argument("--reconstruction", default="", help="the recipe")
    ap.add_argument("--reconstruction-kind", choices=RECONSTRUCTION_KINDS, default=None)
    ap.add_argument("--byte-identical", choices=BYTE_IDENTICAL, default=None,
                    help="whether the recipe gives back these bytes (default: not_expected for a retrain, "
                         "else not_verified)")
    ap.add_argument("--byte-identical-evidence", default="",
                    help="the earlier rebuild whose hash matched (required with --byte-identical verified)")
    ap.add_argument("--same-hash-copy", action="append", default=[],
                    help="where a byte-identical copy survives (repeatable): a local path, relay:<path> or "
                         "hf:<owner>/<repo>:<path>; each one is verified (verify_copies)")
    ap.add_argument("--reconstruction-cost", default="")
    ap.add_argument("--relay-path", default="")
    ap.add_argument("--relay-verified", action="store_true",
                    help="every file was compared to the relay copy by hash")
    ap.add_argument("--store", default=None)
    ap.add_argument("--consumer-audit", default="",
                    help="an audit_relay_weights.py record whose entry for this object decides `retire`")
    ap.add_argument("--maintainer-decision", default="",
                    help="the recorded maintainer decision (logs/budget/decisions.md, quoted with its date) that "
                         "retires an artifact the registry protects")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    relay_mode = bool(args.relay_object)
    if relay_mode:
        from shared.deployment import MAIN_RELAY
        rel = args.relay_object.strip("/")
        try:
            api, obj, shared = relay_object(MAIN_RELAY, rel)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        if shared:
            print(f"REFUSED: {rel}'s LFS object is also {shared}; a permanent deletion would remove them too",
                  file=sys.stderr)
            return 2
        target = None
        files, total = [{"path": rel, "size_bytes": int(obj.size), "sha256": obj.file_oid}], int(obj.size)
        members = [{"rel": "", "size_bytes": int(obj.size), "sha256": obj.file_oid}]
    else:
        target = absolute(args.path)
        if not target.exists():
            print(f"nothing at {args.path}", file=sys.stderr)
            return 1
        rel = args.path.rstrip("/")

    ok, why, guard = registry_guard("" if relay_mode else rel, rel if relay_mode else (args.relay_path or None))
    if not ok and not args.maintainer_decision.strip():
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    if args.criterion == "verified_stale_cache" and not args.relay_verified:
        print("REFUSED: verified_stale_cache requires --relay-verified",
              file=sys.stderr)
        return 2
    # `--canonical` may name several survivors, separated by `;`, and may include
    # prose or a relay prefix. Every token that looks like a repository path must
    # actually be there: the whole point is that the survivor really survives.
    missing_survivors = []
    for token in args.canonical.split(";"):
        token = token.strip().split(" ")[0].rstrip(",;")
        if not token or "/" not in token:
            continue
        if token.startswith(("http", "hf:", "relay:")) or token == args.relay_path:
            continue
        if not token.startswith(("artifacts/", "logs/", "docs/", "configs/",
                                 "scripts/", "src/", "tests/", "/")):
            continue                       # a relay prefix or free text
        if not absolute(token).exists():
            missing_survivors.append(token)
    if missing_survivors:
        print(f"REFUSED: --canonical names {missing_survivors}, which are not there",
              file=sys.stderr)
        return 2

    if not relay_mode:
        files, total = manifest(target)
        root = target if target.is_dir() else None
        members = [{"rel": "" if root is None else str((REPO_ROOT / f["path"] if not Path(f["path"]).is_absolute()
                                                         else Path(f["path"])).relative_to(root)),
                    "size_bytes": f["size_bytes"], "sha256": f["sha256"]} for f in files]
    try:
        copies = verify_copies(args.same_hash_copy, members, target=target,
                               target_relay=(MAIN_RELAY, rel) if relay_mode else None) if args.same_hash_copy else []
        recon = reconstruction_fields(args.reconstruction, args.reconstruction_kind, args.byte_identical,
                                      args.byte_identical_evidence, copies, args.criterion,
                                      files[0]["sha256"] if len(files) == 1 else None)
        audit = consumer_audit_entry(args.consumer_audit, rel) if args.consumer_audit else None
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if audit is not None and audit["sha256"] != files[0]["sha256"]:
        print(f"REFUSED: the audit saw sha256 {audit['sha256'][:12]}, the object is {files[0]['sha256'][:12]}",
              file=sys.stderr)
        return 2
    entry = {
        "canonical_id": args.id,
        "historical_paths": [rel],
        "store": args.store or ("relay" if relay_mode else "repo_artifacts"),
        "relay_path": (rel if relay_mode else args.relay_path) or None,
        "retention_tier": "TIER_4_DISPOSABLE",
        "experiment": args.experiment,
        "seed": None,
        "scientific_role": args.role,
        "weights_sha256": (files[0]["sha256"] if len(files) == 1 else None),
        "config_sha256": None,
        "architecture": None,
        "size_bytes": total,
        "size_human": f"{total / 2**20:.2f} MiB" if total < 2**30
                      else f"{total / 2**30:.2f} GiB",
        "n_files": len(files),
        "members": files,
        "references_at_deletion": references(rel, distinguishing_name(rel)),
        **({"consumer_audit": audit} if audit is not None else {}),
        "deletion_criterion": args.criterion,
        "deletion_criterion_meaning": CRITERIA[args.criterion],
        "reason_physical_weights_deleted": args.reason,
        "canonical_surviving_evidence": args.canonical,
        "relay_hash_verified": True if relay_mode else bool(args.relay_verified),
        **recon,
        "expected_reconstruction_cost": args.reconstruction_cost or None,
        "frozen_results_location": args.frozen_results or None,
        **({"protection_lifted": {**guard, "maintainer_decision": args.maintainer_decision.strip()}}
           if guard else {}),
        **({"maintainer_decision": args.maintainer_decision.strip()}
           if args.maintainer_decision.strip() and not guard else {}),
        "deleted_utc": None,
        "tombstone_written_utc": None,
    }

    doc = json.loads(TOMBSTONES.read_text())
    if any(t["canonical_id"] == args.id for t in doc["tombstones"]):
        print(f"REFUSED: a tombstone for {args.id} already exists", file=sys.stderr)
        return 2

    print(json.dumps({k: v for k, v in entry.items() if k not in ("members", "consumer_audit")}, indent=2))
    print(f"  {len(files)} files, {entry['size_human']}")
    if not args.apply:
        print("\nDRY RUN — nothing written, nothing deleted. Pass --apply.")
        return 0

    #: the tombstone first: it must outlive the bytes even if the deletion below fails half way
    entry["tombstone_written_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc["schema"] = "aadistill.tombstone/v2"
    doc["fields"] = sorted(set(doc["fields"]) | set(entry))
    doc.setdefault("schema_note", {})
    doc["schema_note"] = (
        "v2 adds store, size_bytes, n_files, members, references_at_deletion, "
        "deletion_criterion, canonical_surviving_evidence and "
        "relay_hash_verified. v1 entries are unchanged and remain valid: every "
        "field they carried still means what it meant.")
    doc["tombstones"].append(entry)
    #: the owner's totals are a running account (retire_eligible_checkpoints.py advances them the same way)
    if "totals" in doc:
        t = doc["totals"]
        t.update(tombstones_total=t["tombstones_total"] + 1, active=t["active"] + 1,
                 active_retired_bytes=t["active_retired_bytes"] + total)
        t["active_retired_human"] = f"{t['active_retired_bytes'] / 2**30:.3f} GiB"
    TOMBSTONES.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    if relay_mode:
        from huggingface_hub import CommitOperationDelete
        before = sum(o.size for o in api.list_lfs_files(MAIN_RELAY))
        commit = api.create_commit(MAIN_RELAY, operations=[CommitOperationDelete(path_in_repo=rel)],
                                   commit_message=f"retire {rel} (AGENTS.md 2.5; tombstone {args.id})")
        api.permanently_delete_lfs_files(MAIN_RELAY, [obj], rewrite_history=True)
        left = [o for o in api.list_lfs_files(MAIN_RELAY) if o.file_oid == obj.file_oid]
        in_head = api.file_exists(MAIN_RELAY, rel)
        after = sum(o.size for o in api.list_lfs_files(MAIN_RELAY))
        deletion = {"status": "DELETED" if not left and not in_head else "INCOMPLETE",
                    "relay_commit": getattr(commit, "oid", None),
                    "method": "CommitOperationDelete of the path, then permanently_delete_lfs_files"
                              "(rewrite_history=True) on the one LFS object with this oid",
                    "lfs_bytes_before": before, "lfs_bytes_after": after, "freed_bytes": before - after,
                    "verified": {"oid_left": [o.filename for o in left], "path_in_head": in_head}}
    else:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        deletion = {"status": "DELETED" if not target.exists() else "INCOMPLETE"}
    doc = json.loads(TOMBSTONES.read_text())
    e = next(t for t in doc["tombstones"] if t["canonical_id"] == args.id)
    e.update(deleted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), deletion=deletion)
    TOMBSTONES.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"\ntombstoned and deleted: {rel} ({deletion['status']})")
    return 0 if deletion["status"] == "DELETED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
