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
import re
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


def is_sealed_document(rel: str) -> bool:
    """A preregistration or proposal: a commitment whose value is that its
    bytes have not moved. NEVER tolerated by the guard below, however it was
    changed — the 2026-10-09 review found three of these link-repaired, and
    the prefix tolerance above would have absorbed them silently. Same
    convention as `fix_doc_links.is_sealed_document`."""
    name = rel.rsplit("/", 1)[-1].lower()
    return ("/plans/" in rel and name.endswith(".md")
            and ("preregistration" in name or "proposal" in name))


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


#: The TRUE pre-migration base. Round 1 started here; round 2's own base is
#: the round-1 HEAD, which is why a round-1 collateral modification was
#: invisible to a round-2-scoped check. Lineage questions are asked from here.
TRUE_BASE = "35259b64e8a8813cb1cfad4c0bc456b53bc69dbc"

#: A sealed record: its bytes are the evidence. Matched by path shape rather
#: than enumerated, so a new experiment's registration is covered the day it
#: is written.
SEALED_RECORD = re.compile(
    r"(/runs/|/decisions/|/plans/|registration|preregistration|proposal"
    r"|provenance|closeout|authorization|grant)", re.I)

#: ... except these, which match the shape and are not sealed evidence:
#: live documentation and generated navigation.
#: Spelled at BOTH ends, because the lineage walk sees a record's base-commit
#: path while the tree holds its current one.
NOT_SEALED = (
    "docs/core-provenance.md",               # live core documentation
    "docs/maintenance/core-provenance.md",
    "logs/index.json", "logs/stages/index.json",
)

LINK_TARGET = re.compile(r"\]\([^)]*\)")


def _link_normalized(text: str) -> str:
    """The document with every markdown link TARGET blanked.

    A relative link is navigation, not a fact: when the document moves, the
    same string denotes a different file, so preserving the string destroys
    the reference it was written to carry. Visible labels and all other prose
    are evidence and must be byte-identical. Comparing normalized text is
    what separates "its links were repaired" from "its content was edited" —
    the distinction the 2026-10-09 lineage review turned on, where a config's
    `_purpose` and `out_dir` VALUES had changed and no link was involved.
    """
    return LINK_TARGET.sub("](~)", text)


def lineage_guard() -> dict:
    """Has any SEALED record changed since the true pre-migration base?

    Asked across the whole migration lineage and all four record trees, not
    just one round and not just `logs/`. Round 1 modified two E6b arm configs
    and their provenance as collateral of a path sweep; its own record covered
    only the trees it set out to move, and round 2's guard started at the
    round-1 HEAD, so nothing in either record could see it. The registered
    `config_sha256` in E6b's prospective registration stopped matching the
    configs it registered and no mechanism noticed for three rounds.
    """
    from shared.run_layout import resolve_historical

    def resolve_historical_cached(rel: str) -> str:
        return resolve_historical(rel, REPO)

    changed, link_only, violations = 0, [], []
    inspected = 0
    for tree in ("configs", "data", "docs", "logs"):
        for rel in tracked_at(TRUE_BASE, tree):
            if not SEALED_RECORD.search(rel) or rel in NOT_SEALED:
                continue
            if resolve_historical_cached(rel) in NOT_SEALED:
                continue
            inspected += 1
            new = resolve_historical_cached(rel)
            path = REPO / new
            old = git_bytes(TRUE_BASE, rel)
            if not path.is_file():
                violations.append({"record": rel, "state": "ABSENT",
                                   "resolved_to": new})
                continue
            now = path.read_bytes()
            if now == old:
                continue
            changed += 1
            entry = {"record": rel, "now_at": new}
            if rel.endswith(".md") and _link_normalized(
                    old.decode(errors="ignore")) == _link_normalized(
                    now.decode(errors="ignore")):
                link_only.append(entry)
            else:
                entry["why_it_is_a_violation"] = (
                    "content outside markdown link targets differs: this is "
                    "an edit to sealed evidence, not a repaired reference")
                violations.append(entry)
    return {
        "_contract": (
            "Every sealed record, compared against the TRUE pre-migration "
            "base across all four record trees. A sealed record may differ "
            "only in markdown link TARGETS — navigation that necessarily "
            "moves when a document does — with visible labels and all other "
            "content byte-identical. Any other difference is a violation."),
        "true_base": TRUE_BASE,
        "sealed_records_inspected": inspected,
        "byte_identical": inspected - changed,
        "link_targets_repaired_only": link_only,
        "violations": violations,
        "_violations_must_be_empty": (
            "a non-empty list means sealed scientific evidence was edited by "
            "the migration"),
    }


def registration_pin_audit() -> dict:
    """Every committed record that pins a config path and its hash.

    This is the sweep that found the E6b defect generalizable: a prospective
    registration's whole purpose is that a LIVE reader can check the config
    it registered still hashes to the registered value. Run records are
    excluded — their identities are anchored to their own commit, which
    `git` preserves and a later tree is not expected to reproduce.
    """
    from aadistill.infrastructure.manifest import sha256_json
    from shared.run_layout import resolve_historical

    rows, broken = [], []
    for p in sorted((REPO / "logs").rglob("*.json")):
        if "/runs/" in p.as_posix():
            continue
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stack = [doc]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                cfg = node.get("config") or node.get("config_path")
                sha = node.get("config_sha256")
                if isinstance(cfg, str) and isinstance(sha, str) \
                        and cfg.endswith(".json"):
                    rel = str(p.relative_to(REPO))
                    target = REPO / resolve_historical(cfg, REPO)
                    if not target.is_file():
                        broken.append({"record": rel, "config": cfg,
                                       "state": "unresolvable"})
                    else:
                        got = sha256_json(json.loads(target.read_text()))
                        rows.append((rel, cfg, got == sha))
                        if got != sha:
                            broken.append({"record": rel, "config": cfg,
                                           "pinned": sha, "loaded": got})
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return {
        "_contract": ("every (record, config, pinned hash) triple outside a "
                      "run directory, verified against the config at its "
                      "resolved current location"),
        "triples_checked": len(rows),
        "agree": sum(1 for _, _, ok in rows if ok),
        "broken": broken,
        "known_pre_existing": {
            "logs/stages/stage-1/phase_c2/validations/state-eval-certification/"
            "v1/authorization.json": (
                "its inputs.config_sha256 (e0864de5…) and inputs.check_sha256 "
                "(537b3be9…) match NO blob at any commit in either file's "
                "history, under either sha256_file or sha256_json. The "
                "mismatch therefore PREDATES the migration and is not "
                "migration damage. Left as found: the record is a "
                "maintainer-granted authorization for a CLOSED engineering "
                "validation, and editing its recorded hashes to agree with "
                "the tree would rewrite a granted authorization — which is "
                "precisely what must not happen. Reported for the maintainer."),
        },
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
    touched, added = [], []
    for line in git("diff", "--name-status", "-M", f"{base}..HEAD",
                    "--", "logs").splitlines():
        parts = line.split("\t")
        status, rel = parts[0], parts[-1]
        if status.startswith("R100"):
            continue                       # pure rename: bytes identical
        if status.startswith("A"):
            #: A NEW file cannot be a frozen record this round rewrote. These
            #: are the round's own evidence (its relocation record and data
            #: manifest); counted so they are visible, not silent.
            added.append(rel)
            continue
        if is_sealed_document(rel):
            touched.append(f"{line}\t[SEALED DOCUMENT]")
            continue
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
        "records_added_this_round": sorted(added),
        "_touched_must_be_empty": (
            "every entry above is a frozen record this round changed in "
            "place — the list being empty IS the guard"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    #: Both default from the record this regenerates, so the convergence
    #: chain can re-derive it with no arguments and a stale claim shows up
    #: as drift instead of surviving as a point-in-time assertion. (The
    #: 2026-10-09 review found exactly that: a `rename_pure` claim that was
    #: true when written and false two commits later.)
    prev = REPO / OUT
    doc = json.loads(prev.read_text()) if prev.is_file() else {}
    ap.add_argument("--base", default=doc.get("base_commit"))
    ap.add_argument("--data-manifest",
                    default=str(prev.parent / "data_pre_move.sha256"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not args.base:
        raise SystemExit("no base commit: pass --base (no prior record to read)")
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
        "_data_manifest": ("data_pre_move.sha256, beside this record: the "
                           "sha256 of every data/ file before the move, "
                           "including the gitignored payloads"),
        "data_bytes_verified": data_bytes_verified(Path(args.data_manifest)),
        "scientific_identities_verified_unchanged": scientific_identities(),
        "identity_guard": identity_guard(args.base),
        "lineage_guard": lineage_guard(),
        "registration_pin_audit": registration_pin_audit(),
        "frozen_exceptions": list(decl.FROZEN_EXCEPTIONS),
        "_restored_to_true_base_content": {
            "_why": (
                "These were modified by ROUND 1 and restored, in round 3, to "
                "their content at the true pre-migration base. They therefore "
                "appear under `files` as `rename_modified` — this record's own "
                "base is the round-1 HEAD, against which a restoration IS a "
                "change — while being byte-identical to 35259b64. "
                "`lineage_guard` is the statement that matters for them."),
            "records": [
                "configs/stage3/e6b/e6b_p2_r2960k_sa.json",
                "configs/stage3/e6b/e6b_p2_r2960k_sb.json",
                "configs/stage3/e6b/provenance.json",
            ],
            "verified_byte_identical_to": TRUE_BASE,
        },
        "files": files,
        "authorizes": "nothing",
    }
    body = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    out = REPO / OUT
    guard = (doc["identity_guard"]["frozen_log_records_content_modified_this_round"]
             + doc["lineage_guard"]["violations"])
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
