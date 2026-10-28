"""Retire the registry's retirement-eligible checkpoint weights under AGENTS.md 2.5. $0.

    PYTHONPATH=src:scripts python scripts/maintenance/consolidation/retire_eligible_checkpoints.py            # dry run
    PYTHONPATH=src:scripts python scripts/maintenance/consolidation/retire_eligible_checkpoints.py --execute

The STANDING authorization (AGENTS.md 2.5, restated by the maintainer
2026-10-28): a completed historical checkpoint's weights are retired when its
experiment is terminal, its measurement evidence is durable, its hashes are
preserved, no active or declared consumer reads the bytes, and it is not an
incumbent, a release or a retained anchor. The registry
(``logs/maintenance/inventories/checkpoint_registry.json``) records that audit
per checkpoint as ``disposition: retire`` with its reason; this script acts on
exactly those rows and RE-DERIVES every condition it can check before touching a
byte, rather than inheriting the table's word:

* the row is not protected and has no ``never_delete_clause``;
* the local weights re-hash to the registry's ``weights_sha256``;
* no file under ``src/``, ``configs/`` or ``scripts/`` (outside ``tests/``
  directories, where a state id is an identity string) names the checkpoint's
  state id or path, and no current session launcher declares its relay path as an
  input (`tests/support/session_specs.SESSION_LAUNCHERS`, built through the real
  parsers; a launcher that cannot be built fails the whole run closed);
* a relay copy is removed only by ``permanently_delete_lfs_files`` on the exact
  object whose ``file_oid`` is the weights' sha256 and whose filename is the
  registry's relay path (exactly one match, asserted).

A row any condition refuses is KEPT and reported; only the others are retired.
A tombstone per checkpoint is appended to
``logs/maintenance/inventories/checkpoint_tombstones.json`` BEFORE its bytes go.
Evidence, manifests, records and identities are never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _p in ("src", "scripts", "tests"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

REGISTRY = REPO / "logs/maintenance/inventories/checkpoint_registry.json"
TOMBSTONES = REPO / "logs/maintenance/inventories/checkpoint_tombstones.json"
RELAY = "AlphaAvatar/aadistill-artifacts"
CRITERION = "agents_2_5_standing_retirement"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_relay_paths() -> set[str]:
    """Every relay path a current session launcher declares, from its real spec."""
    from support.session_specs import SESSION_LAUNCHERS, load_session_launcher, session_args

    out: set[str] = set()
    for name, extra in SESSION_LAUNCHERS:
        mod = load_session_launcher(name)
        spec = mod.spec(session_args(mod, extra))
        out |= {r.path for r in spec.setup.relay_inputs}
    return out


def code_references(needle: str) -> list[str]:
    hits = subprocess.run(["git", "grep", "-l", needle, "--", "src", "configs", "scripts"], cwd=REPO,
                          capture_output=True, text=True).stdout.split()
    return [h for h in hits if "/tests/" not in h and not h.endswith(Path(__file__).name)]


def plan(rows: list[dict], declared: set[str]) -> list[dict]:
    out = []
    for r in rows:
        why = []
        if r.get("protected"):
            why.append("protected")
        if r.get("never_delete_clause"):
            why.append(f"never_delete: {r['never_delete_clause']}")
        files = [Path(f) for f in r.get("weight_files") or []]
        if not files or not all(f.is_file() for f in files):
            why.append("the local weights are not where the registry says")
        elif len(files) == 1 and _sha(files[0]) != r["weights_sha256"]:
            why.append("the local weights do not re-hash to the registry's sha256")
        refs = code_references(r["canonical_name"][:24])
        if refs:
            why.append(f"named by code or config: {refs[:4]}")
        if r.get("relay_path") and r["relay_path"] in declared:
            why.append("a current session declares its relay path")
        out.append({"row": r, "eligible": not why, "why": why})
    return out


def tombstone(r: dict, utc: str, relay_verified: bool | None) -> dict:
    members = [{"path": f, "size_bytes": Path(f).stat().st_size if Path(f).is_file() else None,
                "sha256": r["weights_sha256"]} for f in r["weight_files"]]
    return {
        "canonical_id": f"{r['experiment']}_{r['canonical_name']}" + ("_duplicate" if r["retention"] == "duplicate" else ""),
        "historical_paths": [r["path_local"]], "store": r["store"], "relay_path": r.get("relay_path"),
        "retention_tier": "TIER_3_RECORD_ONLY", "experiment": r["experiment"], "seed": None,
        "scientific_role": r.get("role"), "weights_sha256": r["weights_sha256"],
        "config_sha256": r.get("config_sha256"), "architecture": r.get("architecture"),
        "size_bytes": r.get("size_bytes"), "size_human": r.get("size_human"), "n_files": len(members),
        "members": members, "references_at_deletion": list(r.get("referenced_by") or []),
        "deletion_criterion": CRITERION,
        "deletion_criterion_meaning": ("AGENTS.md 2.5 (standing; restated by the maintainer 2026-10-28): the experiment "
                                       "is terminal, its evaluation evidence is durable, its hashes are preserved, no "
                                       "active or declared consumer reads the weights, and it is not an incumbent, a "
                                       "release or a retained anchor"),
        "reason_physical_weights_deleted": (r.get("retention_reason") or "") + " | re-derived 2026-10-28 by "
                                           "retire_eligible_checkpoints.py: local bytes re-hashed, no code/config "
                                           "reader, no session launcher declares it",
        "canonical_surviving_evidence": "the experiment's run records under logs/stages/ (closeouts, evidence, "
                                        "identities) and this tombstone",
        "relay_hash_verified": relay_verified, "deterministic_reconstruction_possible": bool(r.get("reconstructable")),
        "reconstruction_recipe": r.get("reconstruction_recipe"), "expected_reconstruction_cost": r.get("reconstruction_cost"),
        "frozen_results_location": "logs/stages/" + str(r["experiment"]).replace("phase_", "stage-1/phase_"),
        "deleted_utc": utc, "withdrawn": False, "withdrawn_reason": None, "withdrawn_utc": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    reg = json.loads(REGISTRY.read_text())
    rows = [r for r in reg["checkpoints"] if r.get("disposition") == "retire"]
    declared = declared_relay_paths()
    p = plan(rows, declared)
    for x in p:
        r = x["row"]
        print(("ELIGIBLE " if x["eligible"] else "REFUSED  ") + f"{r['experiment']}/{r['canonical_name'][:16]} "
              f"{r['size_human']} local={r['path_local']} relay={r.get('relay_path')} {x['why']}")
    kept = [x for x in p if not x["eligible"]]
    p = [x for x in p if x["eligible"]]
    if kept:
        print(f"{len(kept)} row(s) KEPT: a condition does not hold, so their bytes stay")
    print(f"{len(p)} rows, {sum(x['row']['size_bytes'] for x in p) / 2**30:.3f} GiB local, "
          f"{len({x['row']['relay_path'] for x in p if x['row'].get('relay_path')})} relay objects")
    if not a.execute:
        return 0
    from huggingface_hub import HfApi
    api = HfApi(token=Path(os.path.expanduser("~/.cache/huggingface/token")).read_text().strip())
    lfs = {o.filename: o for o in api.list_lfs_files(RELAY)}
    utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ts = json.loads(TOMBSTONES.read_text())
    relay_done, freed = set(), 0
    for x in p:
        r = x["row"]
        rel = r.get("relay_path")
        obj = lfs.get(rel) if rel else None
        if rel and rel not in relay_done:
            if obj is None or obj.file_oid != r["weights_sha256"]:
                raise SystemExit(f"{rel}: the relay object is not the registry's ({getattr(obj, 'file_oid', None)})")
        ts["tombstones"].append(tombstone(r, utc, relay_verified=(obj is not None) if rel else None))
        TOMBSTONES.write_text(json.dumps(ts, indent=1) + "\n")
        freed += int(r["size_bytes"])
        shutil.rmtree(r["path_local"])
        print(f"retired local {r['path_local']}")
        if rel and rel not in relay_done:
            api.permanently_delete_lfs_files(RELAY, [obj], rewrite_history=True)
            relay_done.add(rel)
            print(f"retired relay {rel} ({obj.size / 2**30:.3f} GiB)")
    #: the owner's totals are a running account (older entries carry no size), so they are advanced, not re-summed
    ts["totals"].update(tombstones_total=ts["totals"]["tombstones_total"] + len(p),
                        active=ts["totals"]["active"] + len(p),
                        active_retired_bytes=ts["totals"]["active_retired_bytes"] + freed)
    ts["totals"]["active_retired_human"] = f"{ts['totals']['active_retired_bytes'] / 2**30:.3f} GiB"
    TOMBSTONES.write_text(json.dumps(ts, indent=1) + "\n")
    print(f"freed {freed / 2**30:.3f} GiB locally and {len(relay_done)} relay objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
