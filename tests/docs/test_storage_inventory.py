"""The inventories have to stay true, or a deletion pass acts on a fiction.

These are cheap structural checks over `logs/maintenance/inventories/checkpoint_registry.json`,
`logs/maintenance/inventories/log_inventory.json` and `logs/maintenance/inventories/checkpoint_tombstones.json`. They exist
because the failure mode of a cleanup is not "the script crashed" — it is a
registry that still describes the tree of three days ago, a tombstone pointing at
a survivor that did not survive, or a `delete` proposal against something the
retention rule protects. Each of those reads as fine and is not.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "logs/maintenance/inventories/checkpoint_registry.json"
LOG_INVENTORY = REPO / "logs/maintenance/inventories/log_inventory.json"
TOMBSTONES = REPO / "logs/maintenance/inventories/checkpoint_tombstones.json"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


# --- the checkpoint registry -----------------------------------------------

def test_nothing_protected_is_proposed_for_deletion():
    for e in load(REGISTRY)["checkpoints"]:
        if e["disposition"] == "delete":
            assert not e["protected"], (
                f"{e['path_local']} is proposed for deletion and is protected "
                f"({e['retention']}): {e['retention_reason']}")
            assert not e["never_delete_clause"], (
                f"{e['path_local']} is proposed for deletion under a never-delete "
                f"clause: {e['never_delete_clause']}")


def test_every_delete_proposal_states_how_to_get_it_back():
    for e in load(REGISTRY)["checkpoints"]:
        if e["disposition"] == "delete":
            assert e["reconstructable"] and e["reconstruction_recipe"], (
                f"{e['path_local']} is proposed for deletion with no "
                "reconstruction recipe")


def test_a_delete_that_leans_on_the_relay_carries_the_verification():
    """"There is a copy on the relay" is only a reason to delete if somebody
    checked the bytes. Single files are checked by LFS oid in the registry;
    whole trees are checked by verify_relay_mirror.py."""
    for e in load(REGISTRY)["checkpoints"]:
        if e["disposition"] != "delete" or e["canonical_location"] != "relay + local":
            continue
        mirror = e.get("relay_mirror_verification") or {}
        assert e["relay_verified"] or mirror.get("verified"), (
            f"{e['path_local']} is proposed for deletion on the strength of a "
            "relay copy that has not been hash-verified")


def test_no_weight_artifact_is_committed_to_git():
    """AGENTS.md 2.5. The registry checks it; this makes the check run."""
    visible = load(REGISTRY)["repository_visible"]
    assert visible["violations"] == 0, (
        f"weight artifacts are tracked by git: "
        f"{visible['tracked_weight_artifacts'][:5]}")


def test_the_registry_covers_the_out_of_tree_store():
    """The first version of the registry saw only `artifacts/` and reported 4.47
    GiB while 81 GiB sat in /home/ecs-user/aad-artifacts. If that store exists,
    the registry must be looking at it."""
    external = Path.home() / "aad-artifacts"   # through $HOME; see cpu_test_env
    if not external.is_dir():
        pytest.skip("no out-of-tree store on this machine")
    stores = load(REGISTRY)["local"]["by_store"]
    assert "external_store" in stores and stores["external_store"]["units"] > 0, (
        "the out-of-tree artifact store exists but the registry inventories "
        "nothing in it")


# --- the log inventory -----------------------------------------------------

def test_every_duplicate_names_a_survivor_that_exists():
    inv = load(LOG_INVENTORY)
    for r in inv["files"]:
        if r["disposition"] == "delete_duplicate":
            assert r["duplicate_of"], f"{r['path']} has no canonical copy named"
            assert (REPO / r["duplicate_of"]).is_file(), (
                f"{r['path']} defers to {r['duplicate_of']}, which is not there")


def _relocated(old: str) -> str:
    """Where an object recorded at `old` lives now.

    The inventory names a survivor by the path it had when the record was
    written, and that statement is not rewritten. `logs/index.json` carries the
    old -> current table; it is already flattened, so one lookup answers what
    used to need chaining across four migration manifests.
    """
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from maintenance.architecture.record_run_index import resolve_historical

    return resolve_historical(old, REPO)


def test_removed_copies_still_name_a_survivor_and_keep_their_hash():
    inv = load(LOG_INVENTORY)
    for r in inv.get("removed", []):
        assert len(r["sha256"]) == 64, f"{r['path']} lost its hash"
        survivor = r.get("canonical_survivor")
        if survivor:
            #: Resolved through the migration record. The inventory names the
            #: survivor's path AS IT WAS; log-layout-v1 moved it, and the
            #: inventory is historical evidence that is not rewritten to match.
            #: What must still hold is that the survivor EXISTS -- which is a
            #: question about the object, not about its address.
            here = REPO / survivor
            if not here.is_file():
                here = REPO / _relocated(survivor)
            assert here.is_file(), (
                f"{r['path']} was removed in favour of {survivor}, which is "
                "gone and is not in the relocation record either")


def test_each_duplicate_group_decided_its_canonical_copy():
    for g in load(LOG_INVENTORY)["duplicate_groups"]:
        assert g["decided_by"] != "none", (
            f"duplicate group {g['sha256'][:12]} has no rule or override deciding "
            f"which of {g['members']} is canonical")


def test_declared_living_state_snapshots_really_are_in_history():
    for s in load(LOG_INVENTORY)["living_state_snapshots"]:
        if s["present_on_disk"]:
            assert s["verified_identical"], (
                f"{s['path']} is declared a snapshot of a file in git history, "
                f"but the bytes differ from {s['git_reference']}")
        else:
            assert s["sha256_in_history"], (
                f"{s['path']} was removed as a snapshot, but "
                f"{s['git_reference']} does not resolve")


# --- tombstones ------------------------------------------------------------

def test_every_tombstone_carries_what_makes_it_a_tombstone():
    doc = load(TOMBSTONES)
    required = {"canonical_id", "historical_paths", "scientific_role",
                "reason_physical_weights_deleted", "deleted_utc"}
    for t in doc["tombstones"]:
        missing = required - set(t)
        assert not missing, f"{t.get('canonical_id')} is missing {sorted(missing)}"


def test_no_tombstoned_path_is_still_on_disk():
    """A tombstone asserts something stopped existing. This found a real defect
    on the day it was written: the pod simulator recreates its quarantine on
    every sweep by design, so a tombstone for it was wrong — withdrawn, with the
    reason kept."""
    for t in load(TOMBSTONES)["tombstones"]:
        if t.get("withdrawn"):
            continue
        for p in t["historical_paths"]:
            path = Path(p) if Path(p).is_absolute() else REPO / p
            assert not path.exists(), (
                f"{t['canonical_id']} has a tombstone but {p} still exists — "
                "either the deletion did not happen or the tombstone is wrong")


def test_a_withdrawn_tombstone_says_why_it_was_wrong():
    for t in load(TOMBSTONES)["tombstones"]:
        if t.get("withdrawn"):
            assert t.get("withdrawn_reason"), (
                f"{t['canonical_id']} is withdrawn with no reason; the reason a "
                "tombstone was wrong is the part worth keeping")


def test_tombstone_ids_are_unique():
    ids = [t["canonical_id"] for t in load(TOMBSTONES)["tombstones"]]
    assert len(ids) == len(set(ids)), "duplicate canonical_id in the tombstones"


# --- the failure class Phase-A attempt 8 paid for --------------------------

def session_staging_destinations() -> dict[str, str]:
    """Every directory a session's manifest declares setup will stage into.

    Read from the four real `SessionSpec`s — the same declaration the pod
    actually executes — so this cannot drift from what a pod does. Returns
    `{repo_relative_dir: which session and field declared it}`.
    """
    import importlib.util
    import sys

    helper = REPO / "tests/support/session_specs.py"
    spec = importlib.util.spec_from_file_location("support.session_specs", helper)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["support.session_specs"] = mod
    spec.loader.exec_module(mod)

    out: dict[str, str] = {}
    for name, _m, _a, s in mod.all_specs():
        for r in s.setup.relay_inputs:
            for field, dest in (("dest", r.dest),
                                ("also_stage_to", r.also_stage_to)):
                if dest:
                    out.setdefault(dest.strip("/"), f"{name}.{field} ({r.path})")
    return out


def test_no_active_tombstone_names_a_routine_staging_destination():
    """A tombstone asserts a path stopped existing. A staging destination is
    recreated by design, every session, so it can never be retired.

    **This is the assertion Phase-A attempt 8 paid $0.19 to discover.** The
    tombstone `stage3_ladder_uniform_local_cache` named
    `artifacts/stages/stage-3/ladder_uniform`; the pod's setup stages the frozen
    recovery pack into exactly that directory as the mirror the recovery-corpus
    loader reads. On the pod the path existed,
    `test_no_tombstoned_path_is_still_on_disk` fired, the blocking setup gate
    failed and the session died having run no stage.

    That test could not have caught this at $0: on the dev box the pod simulator
    *hides* the same gitignored directory, so it passed for the wrong reason.
    This one asks a question about **declarations** and touches the filesystem
    nowhere, so it gives the same answer on the dev box, in the simulator and on
    a pod.
    """
    staged = session_staging_destinations()
    assert staged, "no staging destinations found; the extractor is broken"

    problems = []
    for t in load(TOMBSTONES)["tombstones"]:
        if t.get("withdrawn"):
            continue
        for p in t["historical_paths"]:
            if Path(p).is_absolute():
                continue                     # not a path inside the checkout
            rel = p.strip("/")
            for dest, who in staged.items():
                if rel == dest or rel.startswith(dest + "/") or dest.startswith(rel + "/"):
                    problems.append(
                        f"{t['canonical_id']} tombstones {p!r}, which {who} "
                        f"stages into {dest!r}")
    assert not problems, (
        "an active tombstone names a routine session staging destination:\n  "
        + "\n  ".join(problems)
        + "\nA directory a session recreates by design has not been retired. "
        "Withdraw the tombstone, keeping the deletion record and the reason, as "
        "podsim_quarantine_residue and stage3_ladder_uniform_local_cache were.")


def test_the_tombstone_totals_match_the_tombstones():
    """`totals` is the owner of the active-tombstone accounting, so it has to be
    derived from the list rather than remembered alongside it."""
    doc = load(TOMBSTONES)
    ts, t = doc["tombstones"], doc["totals"]
    active = [x for x in ts if not x.get("withdrawn")]
    withdrawn = [x for x in ts if x.get("withdrawn")]
    assert t["tombstones_total"] == len(ts)
    assert t["active"] == len(active)
    assert t["withdrawn"] == len(withdrawn)
    assert t["withdrawn_ids"] == sorted(x["canonical_id"] for x in withdrawn)
    assert t["active_retired_bytes"] == sum(
        x.get("size_bytes") or 0 for x in active)
    assert t["withdrawn_bytes"] == sum(
        x.get("size_bytes") or 0 for x in withdrawn)


# --- reconstruction is three facts, not one flag ------------------------------

def _retire_artifact():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "retire_artifact", REPO / "scripts/maintenance/consolidation/retire_artifact.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_recipe_is_not_a_byte_identical_rebuild():
    """`deterministic_reconstruction_possible` was `bool(--reconstruction)`, so eight retrained students read as
    deterministically rebuildable. It is now derived from a verified byte identity, never from a recipe."""
    RA = _retire_artifact()
    crit = "agents_2_5_standing_retirement"
    retrain = RA.reconstruction_fields("retrain the unit from its config", "retrain", None, "", [], crit)
    assert retrain["deterministic_reconstruction_possible"] is False
    assert retrain["byte_identical_reconstruction"] == "not_expected"
    assert RA.reconstruction_fields("rebuild", "rebuild", None, "", [], crit)[
        "deterministic_reconstruction_possible"] is False
    assert RA.reconstruction_fields("rebuild", "rebuild", "verified", "the rebuild's hash matched", [], crit)[
        "deterministic_reconstruction_possible"] is True
    for bad in (("a recipe", None, None, ""),                      # a recipe without its kind
                ("retrain", "retrain", "verified", "evidence"),    # a retrain is a new checkpoint
                ("rebuild", "rebuild", "verified", "")):           # verified needs its evidence
        with pytest.raises(ValueError):
            RA.reconstruction_fields(*bad, [], crit)
    with pytest.raises(ValueError):
        RA.reconstruction_fields("rebuild", "rebuild", "not_verified", "", [], "reproducible_materialization")
    with pytest.raises(ValueError):
        RA.reconstruction_fields("", "copy", None, "", [], "byte_identical_duplicate")


def test_a_verified_rebuild_needs_the_recipe_that_rebuilds():
    """`--byte-identical verified` with no recipe, or a recipe of kind `none`, claimed a rebuild of nothing."""
    RA = _retire_artifact()
    crit = "agents_2_5_standing_retirement"
    for recipe, kind in (("", "rebuild"), ("", "copy"), ("a recipe", "none")):
        with pytest.raises(ValueError, match="recipe"):
            RA.reconstruction_fields(recipe, kind, "verified", "the rebuild's hash matched", [], crit)


def _members(root, files):
    return [{"rel": "" if root is None else rel, "size_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
            for rel, b in files]


def test_a_same_hash_copy_is_checked_never_skipped(tmp_path):
    """The check skipped a path that did not exist, compared nothing for a directory, and accepted the target
    itself (or a path inside it) as its own surviving copy; a directory copy's member linked (hard or symbolic) to a
    file of the target is the target's bytes too."""
    RA = _retire_artifact()
    target = tmp_path / "ckpt"
    target.mkdir()
    files = [("model.safetensors", b"weights"), ("config.json", b"{}")]
    for rel, b in files:
        (target / rel).write_bytes(b)
    members = _members(target, files)
    #: a missing copy is refused, not skipped
    with pytest.raises(ValueError, match="nothing there"):
        RA.verify_copies([str(tmp_path / "gone")], members, target=target)
    #: the target, a file inside it, and a directory containing it are inside the deletion scope
    for inside in (target, target / "model.safetensors", tmp_path):
        with pytest.raises(ValueError, match="deletion scope"):
            RA.verify_copies([str(inside)], members, target=target)
    #: a directory copy is complete only when every member matches
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="config.json missing"):
        RA.verify_copies([str(copy)], members, target=target)
    (copy / "config.json").write_bytes(b"{ }")
    with pytest.raises(ValueError, match="config.json differs"):
        RA.verify_copies([str(copy)], members, target=target)
    (copy / "config.json").write_bytes(b"{}")
    got = RA.verify_copies([str(copy) + " (dev box)"], members, target=target)
    assert got[0]["members"] == 2 and got[0]["note"] == "(dev box)"
    #: a hard link is one physical copy
    single = tmp_path / "single.safetensors"
    single.write_bytes(b"weights")
    link = tmp_path / "elsewhere" / "single.safetensors"
    link.parent.mkdir()
    os.link(single, link)
    with pytest.raises(ValueError, match="hard link"):
        RA.verify_copies([str(link)], _members(None, [("", b"weights")]), target=single)
    #: a directory copy whose member is a symbolic link into the target is the target's bytes
    sym = tmp_path / "symcopy"
    sym.mkdir()
    (sym / "config.json").write_bytes(b"{}")
    (sym / "model.safetensors").symlink_to(target / "model.safetensors")
    with pytest.raises(ValueError, match="symbolic link"):
        RA.verify_copies([str(sym)], members, target=target)
    #: ... and so is a member linked to ANOTHER file of the target that holds the same bytes
    dup = tmp_path / "dup"
    dup.mkdir()
    for rel in ("a.bin", "b.bin"):
        (dup / rel).write_bytes(b"same")
    dmem = _members(dup, [("a.bin", b"same"), ("b.bin", b"same")])
    other = tmp_path / "dupcopy"
    other.mkdir()
    os.link(dup / "b.bin", other / "a.bin")
    (other / "b.bin").write_bytes(b"same")
    with pytest.raises(ValueError, match="a.bin is a hard link"):
        RA.verify_copies([str(other)], dmem, target=dup)
    #: a symbolic link to bytes OUTSIDE the scope is a copy
    (sym / "model.safetensors").unlink()
    (sym / "model.safetensors").symlink_to(copy / "model.safetensors")
    assert RA.verify_copies([str(sym)], members, target=target)[0]["members"] == 2


def test_a_remote_copy_is_the_object_the_hub_serves(tmp_path):
    RA = _retire_artifact()
    members = _members(None, [("", b"weights")])
    sha = members[0]["sha256"]

    def hub(repo, paths):
        return {"a/model.safetensors": {"sha256": sha, "size": 7, "lfs": True},
                "b/model.safetensors": {"sha256": "0" * 64, "size": 7, "lfs": True}}
    assert RA.verify_copies(["hf:org/other:a/model.safetensors"], members, target=tmp_path / "t", hub=hub)
    with pytest.raises(ValueError, match="differs"):
        RA.verify_copies(["hf:org/other:b/model.safetensors"], members, target=tmp_path / "t", hub=hub)
    with pytest.raises(ValueError, match="not on"):
        RA.verify_copies(["hf:org/other:c/model.safetensors"], members, target=tmp_path / "t", hub=hub)
    #: retiring a relay object: the same repository's file with the same sha256 is the same LFS object
    with pytest.raises(ValueError, match="same repository"):
        RA.verify_copies(["hf:org/relay:a/model.safetensors"], members, target_relay=("org/relay", "x"), hub=hub)


def test_the_relay_audit_refuses_a_retirement_its_facts_contradict():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "audit_relay_weights", REPO / "scripts/maintenance/consolidation/audit_relay_weights.py")
    AR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AR)
    obj = {"registry": [], "launchers": [], "readers": {"code": {"configs/x.json": ["id"]}}, "same_sha256_paths": []}
    ok = {"disposition": "retire", "readers": {"configs/x.json": "closed experiment"}}
    assert AR.contradictions("p", obj, ok, {"p"}) == []
    assert AR.contradictions("p", obj, {"disposition": "retire"}, {"p"})          # an unclassified reader
    prot = {**obj, "registry": [{"canonical_name": "c", "retention": "control", "protected": True}]}
    assert AR.contradictions("p", prot, ok, {"p"})
    for status in ("granted", "proposed", None):
        l = {**obj, "launchers": [{"launcher": "L", "study": {"status": status} if status else None}]}
        assert AR.contradictions("p", l, ok, {"p"}), status
    closed = {**obj, "launchers": [{"launcher": "L", "study": {"status": "closed"}}]}
    assert AR.contradictions("p", closed, ok, {"p"}) == []
    shared = {**obj, "same_sha256_paths": ["q"]}
    assert AR.contradictions("p", shared, ok, {"p"}) and not AR.contradictions("p", shared, ok, {"p", "q"})
    assert AR.fragments("e1_scaling_20260801/e1_r0860k_sb_pca/step_001023/model/model.safetensors") == [
        "e1_r0860k_sb_pca", "e1_scaling_20260801/e1_r0860k_sb_pca/step_001023/model/model.safetensors"]
    assert "d1" not in AR.fragments("stage0_calvol/finalists/formal_v1/d1/4df438a839cd1ed057c2277dfcb8922f/m.bin")


def test_tombstones_that_carry_a_byte_identity_derive_the_flag_from_it():
    for e in load(TOMBSTONES)["tombstones"]:
        if "byte_identical_reconstruction" not in e:
            continue      # written before 2026-11-14: see the file's field_semantics_20261114
        assert e["deterministic_reconstruction_possible"] == (e["byte_identical_reconstruction"] == "verified"), (
            e["canonical_id"])
        if e.get("reconstruction_kind") == "retrain":
            assert e["byte_identical_reconstruction"] != "verified", e["canonical_id"]
