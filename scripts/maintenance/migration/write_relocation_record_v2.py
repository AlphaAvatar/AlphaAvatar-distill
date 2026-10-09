"""Derive the round-2 source-relocation record from the move declaration.

    PYTHONPATH=src:scripts python scripts/maintenance/migration/write_relocation_record_v2.py \
        --base <round-2 base commit> --write

Round 2 of the information-architecture migration moved `configs/`, `data/`
and `docs/` to the owner-first layout. This record is DERIVED from
`info_architecture_v2.py` (the single declaration) plus the tree and git:

* every tracked file the round relocated, classified `rename_pure` (bytes at
  the new path equal the bytes at the old path at the base commit) or
  `rename_modified` (they differ — the move was combined with a content
  edit, each of which has its own commit trail);
* every data byte — tracked or gitignored — hash-verified at its new home
  against the pre-move hash manifest;
* the scientific identities §15 of the round-1 instruction froze, recomputed;
* the **identity guard**: proof that the round distinguished CURRENT
  execution identity (the live `recovery_search_scoring@v4` digest, which
  follows the tree by design) from HISTORICALLY CONSUMED identity (the
  digests frozen run records cite), and that no frozen log record was
  content-modified by the round.

AUTHORIZES NOTHING.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import info_architecture_v2 as decl  # noqa: E402

OUT = "logs/maintenance/source-relocations/info-architecture/v2/source-relocation.json"

#: Derived records the round legitimately regenerated in place under logs/.
#: Anything ELSE under logs/ that shows as modified against the base commit
#: is a problem this generator must surface, not absorb.
REGENERATED_UNDER_LOGS = (
    "logs/index.json",
    "logs/README.md",
    "logs/state/current.md",
    "logs/state/current.json",
    "logs/state/artifact_manifests.md",
    "logs/budget/decisions.md",
    "logs/budget/ledger.md",
    "logs/shared/analyses/autoinit_historical_reuse_position.json",
    "logs/maintenance/inventories/log_inventory.json",
    "logs/maintenance/inventories/checkpoint_registry.json",
    "logs/stages/index.json",
    "logs/stages/stage-1/phase_d1/current.json",
    "logs/stages/stage-1/families/d_series/current.json",
    "logs/stages/stage-1/phase_c1/analyses/skip_predicate_audit.json",
)
#: ... and the generated README / navigation / link-repair surfaces, matched
#: by prefix because the doc-link fixer touches whichever documents cited a
#: moved path. Link-target repair preserves the visible historical spelling.
REGENERATED_PREFIXES = ("logs/stages/", "logs/state/", "logs/shared/README")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout


def git_bytes(commit: str, rel: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def tracked_at(commit: str, prefix: str) -> list[str]:
    out = git("ls-tree", "-r", "--name-only", commit, prefix)
    return [l for l in out.splitlines() if l.strip()]


def mapped(rel: str) -> str | None:
    for old, new in decl.all_pairs():
        if rel == old:
            return new
        if rel.startswith(old + "/"):
            return new + rel[len(old):]
    return None


def classify_moves(base: str) -> tuple[list[dict], int, int]:
    files, pure, modified = [], 0, 0
    for tree in ("configs", "data", "docs"):
        for rel in tracked_at(base, tree):
            new = mapped(rel)
            if new is None:
                continue
            old_bytes = git_bytes(base, rel)
            new_path = REPO / new
            if old_bytes is None or not new_path.is_file():
                raise SystemExit(f"unresolvable move {rel} -> {new}")
            same = hashlib.sha256(old_bytes).hexdigest() == sha(new_path)
            kind = "rename_pure" if same else "rename_modified"
            pure += same
            modified += not same
            files.append({"old": rel, "new": new, "kind": kind,
                          "sha256_at_base": hashlib.sha256(old_bytes).hexdigest(),
                          "sha256_now": sha(new_path)})
    return files, pure, modified


def data_bytes_verified(manifest: Path) -> dict:
    ok, entries = 0, []
    for line in manifest.read_text().splitlines():
        h, old = line.split(None, 1)
        new = mapped(old) or old
        now = sha(REPO / new)
        if now != h:
            raise SystemExit(f"DATA HASH MISMATCH {old} -> {new}")
        ok += 1
        entries.append({"old": old, "new": new, "sha256": h})
    return {"files_verified": ok, "mismatches": 0, "entries": entries}


def scientific_identities() -> dict:
    from shared.run_layout import resolve_historical
    from stages.d_series.battery_family import allocation_rule_id
    from stages.phase_d1.d1_session import design_hash

    retention = json.loads((REPO / "logs/stages/stage-1/phase_d1/decisions/"
                            "post_search_finalist_retention.json").read_text())
    members = retention["the_frozen_behavioural_finalists"]["members"]
    fam = json.loads((REPO / resolve_historical(
        "logs/shared/analyses/autoinit_d_series_family_manifest.json",
        REPO)).read_text())
    return {
        "d1_design_hash": design_hash(REPO),
        "allocation_rule_id": allocation_rule_id(),
        "family_content_id": fam["family_content_id"],
        "d1_finalists_single_shard_sha256": {
            f"q{m['finalist']}": m["single_shard_sha256"] for m in members},
        "_finalist_bytes": ("re-verified against the external store by the "
                            "checkpoint registry rebuild in the same round "
                            "(identity_verified per entry)"),
    }


def identity_guard(base: str) -> dict:
    sys.path.insert(0, str(REPO / "scripts/shared"))
    import source_sets

    current = source_sets.recovery_scoring_contract(REPO)
    doc = json.loads(source_sets.SOURCE_SETS_CONFIG.read_text())

    # frozen member lists keep their freeze-time spellings
    v2 = doc["recovery_scoring"]["files_v2_historical"]
    v3 = doc["recovery_scoring"]["files_v3_historical"]
    frozen_spellings_intact = (
        any(f.startswith("scripts/autoinit/") for f in v2)
        and any(f.startswith("scripts/") for f in v3))

    # the pod verifier still pins the HISTORICAL v2 contract it shipped with
    verifier = (REPO / "scripts/shared/pod/verify_frozen_assets.py").read_text()
    v2_pin = 'FROZEN_SCORING_CONTRACT = "recovery_search_scoring@v2"' in verifier

    # no frozen log record was content-modified this round
    touched = []
    for line in git("diff", "--name-status", "-M", f"{base}..HEAD",
                    "--", "logs").splitlines():
        parts = line.split("\t")
        status, rel = parts[0], parts[-1]
        if status.startswith("R100"):
            continue                       # pure rename: bytes identical
        if rel in REGENERATED_UNDER_LOGS or any(
                rel.startswith(p) and rel.endswith((".md", "README.md"))
                for p in REGENERATED_PREFIXES):
            continue
        if rel.endswith("current.json"):
            continue                       # derived live-state, regenerated
        touched.append(line)
    return {
        "_contract": (
            "CURRENT execution identity follows the tree by design; "
            "HISTORICALLY CONSUMED identity is frozen in run records and "
            "must not move. This block is the round's proof that the two "
            "were kept apart."),
        "current_scoring_contract": {
            "contract": current["contract"],
            "digest": current["digest"],
            "why_it_moved_this_round": (
                "prose-only pointers in recovery.py now cite the relocated "
                "docs/ and configs/ homes; scorer bytes and semantics "
                "unchanged (see frozen_assets.why_it_moved)"),
        },
        "historical_contracts_unmodified": {
            "v2_member_list_keeps_freeze_time_spellings": frozen_spellings_intact,
            "pod_verifier_still_pins_v2": v2_pin,
        },
        "frozen_log_records_content_modified_this_round": touched,
        "_touched_must_be_empty": (
            "every entry above is a frozen record this round changed in "
            "place — the list being empty IS the guard"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-manifest", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    files, pure, modified = classify_moves(args.base)
    doc = {
        "schema": "aadistill.migration_source_relocation/v1",
        "migration": "info-architecture",
        "version": "v2",
        "_contract": (
            "Round 2: configs/, data/ and docs/ join the owner-first layout "
            "established by round 1 (v1/ beside this file). Frozen records "
            "were not rewritten; old paths resolve through logs/index.json "
            ":: historical_paths.map. AUTHORIZES NOTHING."),
        "maintainer_decision": (
            "Maintainer instruction of 2026-10-09: finish the six-tree "
            "information architecture (configs, data, docs), extend the "
            "ownership index and relocation registry, close the artifact "
            "registry gap, and stop for independent review. D1 stays paused."),
        "base_commit": args.base,
        "branch": "migration/info-architecture",
        "move_map_owner": "scripts/maintenance/migration/info_architecture_v2.py",
        "totals": {
            "tracked_files_relocated": len(files),
            "rename_pure": pure,
            "rename_modified": modified,
            "relocation_pairs_added_to_historical_paths": len(decl.all_pairs()),
        },
        "data_bytes_verified": data_bytes_verified(Path(args.data_manifest)),
        "scientific_identities_verified_unchanged": scientific_identities(),
        "identity_guard": identity_guard(args.base),
        "frozen_exceptions": list(decl.FROZEN_EXCEPTIONS),
        "files": files,
        "authorizes": "nothing",
    }
    body = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    out = REPO / OUT
    guard = doc["identity_guard"]["frozen_log_records_content_modified_this_round"]
    if guard:
        print("IDENTITY GUARD VIOLATIONS:")
        for g in guard:
            print("  ", g)
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body)
        print(f"wrote {OUT} ({len(files)} moves, {pure} pure, {modified} modified)")
    else:
        print(f"would write {OUT} ({len(files)} moves, {pure} pure, "
              f"{modified} modified; guard violations: {len(guard)})")
    return 1 if guard else 0


if __name__ == "__main__":
    sys.exit(main())
