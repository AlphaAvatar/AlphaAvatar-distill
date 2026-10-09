"""Round 2 of the information-architecture migration, held in place.

CPU-only. Each test pins one property the six-tree migration promised:
resolution of every declared old path, per-tree ownership projections that
agree with the index, data manifests loadable at their new homes, and the
identity guard's cheap invariants (the expensive byte checks live in the
committed v2 relocation record and the hashed checkpoint registry).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/maintenance/migration"))

import info_architecture_v2 as decl  # noqa: E402


def test_every_declared_old_path_resolves_to_an_existing_object():
    from shared.run_layout import resolve_historical

    missing = []
    for old, new in decl.all_pairs():
        r = resolve_historical(old, REPO)
        if r != new or not (REPO / r).exists():
            missing.append((old, r))
    assert not missing, missing


def test_no_old_tree_directory_survives():
    for old, _ in decl.CONFIG_DIR_MOVES + decl.DATA_DIR_MOVES:
        assert not (REPO / old).exists(), f"{old} still exists beside its successor"
    for old, _ in decl.CONFIG_FILE_MOVES + decl.DOC_FILE_MOVES:
        assert not (REPO / old).exists(), f"{old} still exists beside its successor"


def test_the_six_tree_projections_are_fresh_and_agree_with_the_index():
    idx = json.loads((REPO / "logs/stages/index.json").read_text())
    by_id = {r["experiment_id"]: r for r in idx["experiments"]}
    for view, leg in (("scripts/index.json", "canonical_scripts"),
                      ("artifacts/index.json", "canonical_artifacts"),
                      ("configs/index.json", "canonical_configs"),
                      ("data/index.json", "canonical_data"),
                      ("docs/index.json", "canonical_docs")):
        doc = json.loads((REPO / view).read_text())
        assert doc["derived_from"] == "logs/stages/index.json"
        for o in doc["owners"]:
            if o["kind"] == "stage":
                continue
            assert by_id[o["logical_id"]].get(leg) == o[view.split("/")[0]]
            tree_path = o[view.split("/")[0]]
            assert (REPO / tree_path).is_dir(), tree_path


def test_data_manifests_load_at_their_new_homes():
    for rel in ("data/stages/stage-0/warmup/warmup_v1.manifest.json",
                "data/stages/stage-0/warmup/holdout_v1.manifest.json",
                "data/stages/stage-2/stage2/stage2_offline_v0.manifest.json",
                "data/stages/stage-2/stage2_v1/stage2_offline_v1.manifest.json",
                "data/stages/stage-3/stage3_pilot/manifest.json",
                "data/stages/stage-3/eval_behavior_v0/manifest.json"):
        doc = json.loads((REPO / rel).read_text())
        assert doc, rel


def test_the_v2_record_exists_and_its_guard_is_clean():
    rec = json.loads((REPO / "logs/maintenance/source-relocations/"
                      "info-architecture/v2/source-relocation.json").read_text())
    assert rec["identity_guard"][
        "frozen_log_records_content_modified_this_round"] == []
    assert rec["data_bytes_verified"]["mismatches"] == 0
    assert rec["data_bytes_verified"]["files_verified"] == 70
    sci = rec["scientific_identities_verified_unchanged"]
    assert sci["d1_design_hash"].startswith("f9c6688f")
    assert sci["allocation_rule_id"] == "f6047343c1c1ad2172f500e979c704c1"
    assert sci["family_content_id"].startswith("1e3445f1")


def test_historical_source_sets_keep_their_freeze_time_spellings():
    """The v2/v3 member lists are what completed runs bound; rewriting them
    would re-spell a consumed identity."""
    doc = json.loads((REPO / "configs/stages/stage-1/source_sets.json").read_text())
    assert any(f.startswith("scripts/autoinit/")
               for f in doc["recovery_scoring"]["files_v2_historical"])
    # and the current list names only paths that exist
    for f in doc["recovery_scoring"]["files_v4"]:
        assert (REPO / f).is_file(), f


def test_the_pod_verifier_still_pins_the_consumed_v2_contract():
    src = (REPO / "scripts/shared/pod/verify_frozen_assets.py").read_text()
    assert 'FROZEN_SCORING_CONTRACT = "recovery_search_scoring@v2"' in src


def test_the_registry_binds_all_four_finalists_verified():
    reg = json.loads((REPO / "logs/maintenance/inventories/"
                      "checkpoint_registry.json").read_text())
    assert reg["hashed"] is True
    seen = {}
    for e in reg["checkpoints"]:
        si = e.get("scientific_identity")
        if si and e.get("identity_verified") is not None:
            # products/ and oob_products/ can both carry an identity; any
            # copy failing verification is a finding either way.
            assert e["identity_verified"] is True, e["path_local"]
            seen[si["artifact_id"]] = si["single_shard_sha256"]
    assert set(seen) == {"d1_finalist_q1", "d1_finalist_q2",
                         "d1_finalist_q3", "d1_finalist_q4"}
    battery = [e for e in reg["checkpoints"]
               if e["path_local"].endswith("batteries/d_series_behavioural_v1")]
    assert battery and battery[0]["protected"]
