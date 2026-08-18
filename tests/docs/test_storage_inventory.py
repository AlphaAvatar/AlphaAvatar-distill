"""The inventories have to stay true, or a deletion pass acts on a fiction.

These are cheap structural checks over `logs/checkpoint_registry.json`,
`logs/log_inventory.json` and `logs/checkpoint_tombstones.json`. They exist
because the failure mode of a cleanup is not "the script crashed" — it is a
registry that still describes the tree of three days ago, a tombstone pointing at
a survivor that did not survive, or a `delete` proposal against something the
retention rule protects. Each of those reads as fine and is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "logs/checkpoint_registry.json"
LOG_INVENTORY = REPO / "logs/log_inventory.json"
TOMBSTONES = REPO / "logs/checkpoint_tombstones.json"


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
    external = Path("/home/ecs-user/aad-artifacts")
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


def test_removed_copies_still_name_a_survivor_and_keep_their_hash():
    inv = load(LOG_INVENTORY)
    for r in inv.get("removed", []):
        assert len(r["sha256"]) == 64, f"{r['path']} lost its hash"
        survivor = r.get("canonical_survivor")
        if survivor:
            assert (REPO / survivor).is_file(), (
                f"{r['path']} was removed in favour of {survivor}, which is gone")


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

    helper = REPO / "tests/pod/session_specs.py"
    spec = importlib.util.spec_from_file_location("session_specs", helper)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["session_specs"] = mod
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
    `artifacts/stage3/ladder_uniform`; the pod's setup stages the frozen
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
