#!/usr/bin/env python3
"""Object-level consumer and anchor audit of the weight files on the private relay ($0, read-only).

    PYTHONPATH=src:scripts:tests .venv/bin/python scripts/maintenance/consolidation/audit_relay_weights.py \
        [--decisions configs/maintenance/relay_weight_dispositions_<date>.json] [--out <json>]

AGENTS.md 2.5 retires a historical checkpoint's weights only after a real consumer audit: launchers that declare it,
setups that fetch it, registries that protect it, and code that reads it. This collects those facts for EVERY weight
file on the relay (`*.safetensors` / `*.bin` / `*.pt`), object by object, from the live listing:

* identity: path, LFS ``file_oid`` (= sha256 of the bytes), size, and every other relay path whose file has the same
  sha256 (one LFS object: a permanent deletion removes it under every path);
* registry: rows of ``logs/maintenance/inventories/checkpoint_registry.json`` that name the path or the sha256, with
  their retention class, ``protected`` flag and never-delete clause;
* launchers: every current session launcher (``tests/support/session_specs.SESSION_LAUNCHERS``, built through the real
  parsers) that declares the path as a relay input, with its study's state where the launcher has one
  (``shared.study_engineering`` grant: granted / closed / only proposed);
* code: files under ``src``, ``configs`` and ``scripts`` (tests reported apart) naming the object's distinguishing
  path fragments or the first 16 hex of its sha256, excluding the inventory tooling that lists every object;
* local copies: registry rows on the dev box with the same sha256.

It decides nothing by itself. ``--decisions`` names the reviewed disposition of each object (``retire`` or ``keep``
with the reason and the evidence location); the output joins the two and refuses (exit 2) a ``retire`` that a
collected fact contradicts: a protected or never-delete registry row, a launcher whose study is open or only proposed,
an unclassified code reader, or a sha256 shared with a path that is not retired with it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts", "tests"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

RELAY = "AlphaAvatar/aadistill-artifacts"
REGISTRY = REPO / "logs/maintenance/inventories/checkpoint_registry.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
GENERIC = {"model", "checkpoint", "products", "finalists", "formal_v1", "mixed_v2", "selected_leaves"}
#: files that list every object by construction (inventories, their builders, this tool): never a consumer
INVENTORY_TOOLING = ("scripts/maintenance/consolidation/build_checkpoint_registry.py",
                     "scripts/maintenance/consolidation/audit_relay_weights.py",
                     "scripts/maintenance/consolidation/retire_artifact.py",
                     "scripts/maintenance/consolidation/retire_relay_copies.py",
                     "scripts/maintenance/consolidation/retire_eligible_checkpoints.py",
                     "configs/maintenance/")


def token() -> str | None:
    p = Path(os.path.expanduser("~/.cache/huggingface/token"))
    return p.read_text().strip() if p.is_file() else None


def relay_listing(repo: str = RELAY) -> dict[str, dict]:
    """Every file of the repo with its LFS sha256 and size (``None`` sha256 for a non-LFS file)."""
    from huggingface_hub import HfApi
    info = HfApi(token=token()).model_info(repo, files_metadata=True)
    out = {}
    for s in info.siblings:
        lfs = getattr(s, "lfs", None)
        out[s.rfilename] = {"sha256": (lfs.sha256 if lfs is not None else None),
                            "size": int(lfs.size if lfs is not None else (s.size or 0))}
    return out


def _distinguishing(part: str) -> bool:
    """A directory name that names ONE object: a hex state id, or a long underscored name. A short or generic name
    (``d1``, ``2x``, ``model``) or a step tag would match unrelated files."""
    if part in GENERIC or part.startswith("step_"):
        return False
    if len(part) >= 16 and all(c in "0123456789abcdef" for c in part):
        return True
    return "_" in part and len(part) >= 8


def fragments(path: str) -> list[str]:
    """The path's distinguishing fragments below its top-level prefix (which names a whole family of objects), and
    the whole path."""
    return sorted({*(p for p in Path(path).parts[1:-1] if _distinguishing(p)), path})


def git_grep(needle: str, pathspecs: tuple[str, ...]) -> list[str]:
    r = subprocess.run(["git", "grep", "-l", "-F", needle, "--", *pathspecs], cwd=REPO, capture_output=True, text=True)
    return [h for h in r.stdout.split() if h]


def code_readers(path: str, sha: str | None) -> dict:
    needles = fragments(path) + ([sha[:16]] if sha else [])
    hits, tests = {}, {}
    for n in needles:
        for h in git_grep(n, ("src", "configs", "scripts")):
            if any(h.startswith(t) for t in INVENTORY_TOOLING):
                continue
            (tests if "/tests/" in h else hits).setdefault(h, []).append(n)
        for h in git_grep(n, ("tests",)):
            tests.setdefault(h, []).append(n)
    return {"needles": needles, "code": {k: sorted(set(v)) for k, v in sorted(hits.items())},
            "tests": {k: sorted(set(v)) for k, v in sorted(tests.items())}}


def study_state(mod) -> dict | None:
    """The launcher's study (`shared.study_engineering.Study`) and its grant's state, if it has one."""
    st = getattr(mod, "STUDY", None)
    if st is None or not hasattr(st, "grant"):
        return None
    g = REPO / st.grant
    if g.is_file():
        doc = json.loads(g.read_text())
        closed = doc.get("closed")
        return {"study": st.key, "grant": str(st.grant), "status": "closed" if closed else doc.get("status"),
                "closed_utc": (closed or {}).get("utc") if isinstance(closed, dict) else None}
    p = REPO / st.proposal
    return {"study": st.key, "grant": None, "status": "proposed" if p.is_file() else "no grant, no proposal",
            "proposal": str(st.proposal) if p.is_file() else None}


def declaring_launchers() -> dict[str, list[dict]]:
    from support.session_specs import SESSION_LAUNCHERS, load_session_launcher, session_args
    out: dict[str, list[dict]] = {}
    for name, extra in SESSION_LAUNCHERS:
        mod = load_session_launcher(name)
        spec = mod.spec(session_args(mod, extra))
        state = study_state(mod)
        for r in spec.setup.relay_inputs:
            out.setdefault(r.path, []).append({"launcher": name, "file": str(Path(mod.__file__).relative_to(REPO)),
                                               "study": state})
    return out


def registry_rows(path: str, sha: str | None) -> list[dict]:
    reg = json.loads(REGISTRY.read_text())
    rows = []
    for r in reg["checkpoints"]:
        rp = r.get("relay_path") or ""
        if (rp and (rp == path or path.startswith(rp.rstrip("/") + "/"))) or (sha and r.get("weights_sha256") == sha):
            rows.append({k: r.get(k) for k in ("canonical_name", "experiment", "retention", "protected",
                                                "never_delete_clause", "status", "path_local", "relay_path",
                                                "weights_sha256")})
    return rows


def collect() -> dict:
    listing = relay_listing()
    by_sha: dict[str, list[str]] = {}
    for p, f in listing.items():
        if f["sha256"]:
            by_sha.setdefault(f["sha256"], []).append(p)
    declared = declaring_launchers()
    objects = {}
    for p in sorted(p for p in listing if p.endswith(WEIGHT_SUFFIXES)):
        f = listing[p]
        sha = f["sha256"]
        siblings = sorted(q for q in listing if str(Path(q).parent) == str(Path(p).parent) and q != p)
        rows = registry_rows(p, sha)
        objects[p] = {
            "sha256": sha, "size_bytes": f["size"], "size_gib": round(f["size"] / 2**30, 3),
            "same_sha256_paths": sorted(q for q in by_sha.get(sha, []) if q != p) if sha else [],
            "siblings": siblings,
            "registry": rows,
            "local_copies": sorted({r["path_local"] for r in rows if r.get("weights_sha256") == sha and r.get("path_local")}),
            "launchers": declared.get(p, []),
            "readers": code_readers(p, sha)}
    return {"schema": "aadistill.maintenance.relay_weight_audit/v1",
            "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "relay": RELAY,
            "n_files": len(listing), "n_weight_files": len(objects),
            "weight_bytes": sum(o["size_bytes"] for o in objects.values()), "objects": objects}


def contradictions(path: str, obj: dict, dec: dict, retired: set[str]) -> list[str]:
    """Why a reviewed `retire` cannot stand against the collected facts (empty when it can)."""
    if dec.get("disposition") != "retire":
        return []
    bad = []
    for r in obj["registry"]:
        if r.get("protected") or r.get("never_delete_clause"):
            bad.append(f"registry row {r['canonical_name']} ({r['retention']}) protects it")
    for l in obj["launchers"]:
        st = (l.get("study") or {}).get("status")
        if st != "closed" and l["launcher"] not in (dec.get("closed_launchers") or {}):
            bad.append(f"launcher {l['launcher']} declares it and its study is {st!r}")
    unclassified = sorted(set(obj["readers"]["code"]) - set(dec.get("readers") or {}))
    if unclassified:
        bad.append(f"code readers not classified in the decision: {unclassified}")
    shared = [q for q in obj["same_sha256_paths"] if q not in retired]
    if shared:
        bad.append(f"its LFS object is also {shared}, which is not retired with it")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decisions", default=None, help="the reviewed disposition per object (JSON)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    doc = collect()
    rc = 0
    if a.decisions:
        decs = json.loads((REPO / a.decisions).read_text())
        doc["decisions"] = {"path": a.decisions, "schema": decs.get("schema")}
        retired = {p for p, d in decs["objects"].items() if d.get("disposition") == "retire"}
        missing = sorted(set(doc["objects"]) - set(decs["objects"]))
        if missing:
            doc["undecided"] = missing
            rc = 2
        for p, obj in doc["objects"].items():
            d = decs["objects"].get(p)
            if d is None:
                continue
            obj["disposition"] = d
            bad = contradictions(p, obj, d, retired)
            if bad:
                obj["refused"] = bad
                rc = 2
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    summary = {k: len([o for o in doc["objects"].values() if (o.get("disposition") or {}).get("disposition") == k])
               for k in ("retire", "keep")}
    print(json.dumps({"weight_files": doc["n_weight_files"], "gib": round(doc["weight_bytes"] / 2**30, 3), **summary,
                      "refused": [p for p, o in doc["objects"].items() if o.get("refused")],
                      "undecided": doc.get("undecided", [])}), file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
