"""Verify the historical 0.86M checkpoints exist and are the ones the log claims.

**Scope note.** This script answers a lineage question — do these artifacts exist,
are their bytes what the tombstone recorded, and do they descend from the
canonical init at the right rung, seed and objective. It does **not** decide
whether they may serve as Phase-A matched controls; that is the full protocol
comparison in `compare_recovery_fingerprints.py`, which established that they may
not. Every entry here therefore reports `recipe_matched_control: false`, and the
canonical sa/sb reruns are required regardless of what this script finds.


    PYTHONPATH=src .venv/bin/python scripts/autoinit/verify_control_checkpoints.py

Zero cost: metadata and two small JSON files, no LFS payload downloaded.

It does not trust the tombstone's "full copy on the relay" claim —
that exact claim has been wrong in this project before. The Stage 0 activation
cache was recorded as being on the relay and its 780 files contained no `stage0/`
path at all; recovering it cost 83 minutes of CPU and was only possible because
the pipeline was deterministic and the hash was logged.

Four things are checked per checkpoint:

* the file exists on the relay;
* its LFS sha256 equals the tombstone's ``weights_sha256`` (LFS OIDs *are*
  sha256, so this verifies the payload without fetching 2.4 GB);
* its ``config.json`` matches the tombstone's ``config_sha256``;
* its ``run_manifest.json`` describes the recovery protocol the control is
  supposed to be — the frozen initialization, rung, seed, loss weights, schedule
  and block length.

The config check uses **raw file bytes**, because that is what
`build_checkpoint_registry.py` hashed. Note that the repository has two
config-hash conventions in play: `nll_gate.checkpoint_fingerprint` and
`autoinit.artifact` canonicalize the JSON first. Comparing across conventions
produces a mismatch that looks like corruption and is not; this script states
which convention it used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.recovery import E1_KD_HEAVY_0860K  # noqa: E402
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

RELAY = "AlphaAvatar/aadistill-artifacts"
CONTROLS = {
    "e1_r0860k_sa_pca": {"seed": 20260726, "relay_dir":
                         "e1_scaling_20260801/e1_r0860k_sa_pca/step_001023"},
    "e1_r0860k_sb_pca": {"seed": 20260801, "relay_dir":
                         "e1_scaling_20260801/e1_r0860k_sb_pca/step_001023"},
}
EXPECTED_INIT = "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_control_availability.json")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    tombstones = {t["canonical_id"]: t for t in json.loads(
        (REPO_ROOT / "logs/checkpoint_tombstones.json").read_text())["tombstones"]}

    report = {"schema": "aadistill.autoinit.control_availability/v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "relay": RELAY, "controls": {},
              "config_hash_convention": "raw file bytes (build_checkpoint_registry.py)"}

    local_present = {}
    for name in CONTROLS:
        local = REPO_ROOT / "artifacts/stage3/rescued" / name
        local_present[name] = local.is_dir()

    try:
        files = set(api.list_repo_files(RELAY, repo_type="model"))
    except Exception as exc:
        report["relay_reachable"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        (REPO_ROOT / args.out).write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(f"relay unreachable: {exc}")
    report["relay_reachable"] = True

    all_ok = True
    for name, meta in CONTROLS.items():
        tomb = tombstones[name]
        weights = f"{meta['relay_dir']}/model/model.safetensors"
        entry = {
            "canonical_id": name,
            "seed": meta["seed"],
            "present_on_dev_box": local_present[name],
            "tombstoned": True,
            "relay_path": weights,
            "present_on_relay": weights in files,
            "expected_weights_sha256": tomb["weights_sha256"],
            "expected_config_sha256": tomb["config_sha256"],
            "architecture": tomb["architecture"],
        }
        if entry["present_on_relay"]:
            info = api.get_paths_info(RELAY, [weights], repo_type="model")[0]
            oid = getattr(info.lfs, "sha256", None) or getattr(info.lfs, "oid", None)
            entry["actual_weights_sha256"] = oid
            entry["weights_hash_match"] = oid == tomb["weights_sha256"]
            entry["size_bytes"] = getattr(info.lfs, "size", None)

            cfg_path = hf_hub_download(
                RELAY, f"{meta['relay_dir']}/model/config.json", repo_type="model")
            actual_cfg = hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()
            entry["actual_config_sha256"] = actual_cfg
            entry["config_hash_match"] = actual_cfg == tomb["config_sha256"]

            manifest_rel = f"{meta['relay_dir'].rsplit('/', 1)[0]}/run_manifest.json"
            manifest = json.loads(
                Path(hf_hub_download(RELAY, manifest_rel, repo_type="model")).read_text())
            cfg = manifest.get("config", manifest)
            lineage = {
                "student_path": cfg.get("student_path"),
                "rung": cfg.get("rung"),
                "seed": cfg.get("seed"),
                "loss": cfg.get("loss"),
                "total_steps": (cfg.get("schedule") or {}).get("total_steps"),
                "block_len": cfg.get("block_len"),
            }
            recipe = E1_KD_HEAVY_0860K
            checks = {
                "descends_from_canonical_init":
                    lineage["student_path"] == EXPECTED_INIT,
                "rung_is_0860k": lineage["rung"] == recipe.tokens,
                "seed_matches": lineage["seed"] == meta["seed"],
                "ce_weight": (lineage["loss"] or {}).get("ce_weight") == recipe.ce_weight,
                "kd_weight": (lineage["loss"] or {}).get("kd_weight") == recipe.kd_weight,
                "kd_temperature":
                    (lineage["loss"] or {}).get("kd_temperature") == recipe.temperature,
                "kd_scope": (lineage["loss"] or {}).get("kd_scope") == recipe.kd_scope,
                "block_len": lineage["block_len"] == recipe.block_len,
            }
            entry["lineage"] = lineage
            entry["lineage_checks"] = checks
            entry["lineage_valid"] = all(checks.values())
            entry["artifact_available"] = True
            entry["hash_verified"] = bool(
                entry["weights_hash_match"] and entry["config_hash_match"])
            entry["passes_legacy_lineage_subset"] = bool(
                entry["hash_verified"] and entry["lineage_valid"])
        else:
            entry["artifact_available"] = False
            entry["hash_verified"] = False
            entry["passes_legacy_lineage_subset"] = False
        # Never derived here, and never true. Whether a run may serve as a
        # Phase-A matched control is decided by the full protocol comparison in
        # compare_recovery_fingerprints.py, which established that these two
        # cannot: their trainer and runtime identity was never recorded. This
        # script checks lineage and bytes, which is a strict subset.
        entry["recipe_matched_control"] = False
        entry["recipe_matched_control_decided_by"] = (
            "logs/autoinit_recovery_fingerprint_audit.json")
        all_ok = all_ok and entry["passes_legacy_lineage_subset"]
        report["controls"][name] = entry

    report["both_pass_legacy_lineage_subset"] = all_ok
    report["any_recipe_matched_control"] = False
    report["retrieval_bytes"] = sum(
        e.get("size_bytes") or 0 for e in report["controls"].values())
    report["consequence"] = (
        "Both historical 0.86M checkpoints are retrievable and their bytes and "
        "lineage verify, so they remain useful LINEAGE REFERENCES. They are NOT "
        "Phase-A matched controls: the protocol audit found their trainer and "
        "runtime identity was never recorded, so it cannot be compared. Rerun "
        "canonical sa/sb from qwen3_0p6b_init_v0 at 0.86M under the attested "
        "runtime and frozen trainer; those runs become the permanent Phase-A "
        "control probes and are retained, not disposable setup work."
        if all_ok else
        "At least one historical checkpoint is missing or does not verify. Do NOT "
        "redefine the threshold to one seed. The canonical sa/sb reruns are "
        "required either way; this only removes the lineage reference.")
    report["status_field_meanings"] = {
        "artifact_available": "the file exists on the relay",
        "hash_verified": "its payload and config match the tombstone hashes",
        "passes_legacy_lineage_subset": (
            "it descends from the canonical init at the right rung, seed and "
            "objective — a lineage reference, not a protocol match"),
        "recipe_matched_control": (
            "always false here; a matched control requires the full protocol "
            "comparison, which these runs fail on unrecorded trainer/runtime "
            "identity"),
    }
    report["report_sha256"] = sha256_json(report)

    (REPO_ROOT / args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "both_pass_legacy_lineage_subset": all_ok,
        "any_recipe_matched_control": False,
        "controls": {k: {"present_on_relay": v["present_on_relay"],
                         "weights_hash_match": v.get("weights_hash_match"),
                         "config_hash_match": v.get("config_hash_match"),
                         "lineage_valid": v.get("lineage_valid")}
                     for k, v in report["controls"].items()},
        "retrieval_gib": round(report["retrieval_bytes"] / 2**30, 2),
        "consequence": report["consequence"],
    }, indent=2))


if __name__ == "__main__":
    main()
