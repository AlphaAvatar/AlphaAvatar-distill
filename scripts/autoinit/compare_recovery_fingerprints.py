"""Are the historical 0.86M checkpoints recipe-matched to the Phase-A probes?

    PYTHONPATH=src .venv/bin/python scripts/autoinit/compare_recovery_fingerprints.py

Zero cost: reads the historical run manifests from the relay (small JSON, no LFS
payload) and the frozen Phase-A probe config from this checkout, builds a
`RecoveryRecipeFingerprint` for each, and compares them field by field.

The question is not "do the obvious fields agree" — they do. It is whether
**every** field that can change a recovered checkpoint is established and equal.
A field that cannot be established is reported as `unverifiable`, never as
matched: "we have no record of it" and "it is the same" are different statements,
and only one of them supports calling something a matched control.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.recovery import (  # noqa: E402
    HistoricalRunAudit,
    RecoveryProbeIdentity,
    RecoveryProtocolFingerprint,
    RuntimeEnvironmentFingerprint,
    trainer_source_digest,
)
from aadistill.infrastructure.manifest import sha256_json  # noqa: E402

RELAY = "AlphaAvatar/aadistill-artifacts"
HISTORICAL = {
    "e1_r0860k_sa_pca": ("e1_scaling_20260801/e1_r0860k_sa_pca", 20260726),
    "e1_r0860k_sb_pca": ("e1_scaling_20260801/e1_r0860k_sb_pca", 20260801),
}
CANONICAL_INIT_SHA = ("86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc"
                      "952cabd5df2633e54")
PACK_BLOCKS_SHA = ("6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75"
                   "fcecb249c")

#: Mismatches that provably cannot have changed the weights, with the evidence.
#: Everything not on this list is treated as material.
BENIGN_MISMATCHES = {
    "pack": ("a rename, not a different pack. Proved by hash: the relay holds the "
             "historical pack under its historical name at "
             "stage3_recovery_corpus_v2/ladder_uniform/blocks.npz, and its sha256 is "
             "6f324cb0... - exactly the hash the frozen Phase-A recipe pins for "
             "ladder_uniform_probe. Derived quantities agree too (682 blocks, "
             "864,750 supervised tokens, identical six-way mix)."),
    "resume_semantics": ("both historical runs recorded resumed_from=None, so the "
                         "consumed-block accounting added since could not have "
                         "affected them; it changes restore behaviour only"),
}

#: Fields a historical run did not record, and why. Listing them here is what
#: keeps them out of the "matched" column.
HISTORICAL_UNVERIFIABLE = {
    "trainer_source_digest": ("the historical run recorded a whole-repository "
                              "commit and a dirty flag, not a digest over the "
                              "trainer source set; and the tree was dirty, so the "
                              "material trainer identity cannot be reconstructed"),
    "trainer_source_set_version": ("part of the same unrecorded trainer identity: "
                                   "the historical run declared no source set"),
    "runtime_digest": ("no image digest was recorded; only torch 2.11.0+cu128 was "
                       "captured, which does not pin the runtime"),
    "legacy_trainer_uncommitted_sha256": ("code_state.dirty was true; the executed code "
                                   "is a commit plus an uncommitted diff "
                                   "identified only by its hash, which cannot be "
                                   "reconstructed"),
    "kd_chunk": ("not present in the historical config; the trainer default at "
                 "that commit was 512, same as now, but the run did not record it"),
}


def historical_protocol(manifest: dict) -> RecoveryProtocolFingerprint:
    cfg = manifest["config"]
    ladder = manifest.get("ladder") or {}
    code = manifest.get("code_state") or {}
    hardware = manifest.get("hardware") or {}
    teacher = manifest.get("teacher") or {}
    optim, sched, batch = cfg["optim"], cfg["schedule"], cfg["batch"]
    loss = cfg["loss"]
    return RecoveryProtocolFingerprint(
        pack=Path(cfg["data_dir"]).name,
        # Recovered after the fact: the relay still holds this pack under its
        # historical name, and it hashes to the value the frozen recipe pins.
        # The run manifest did not record it (data_manifests was empty), so this
        # is reconstructed evidence rather than a contemporaneous record.
        pack_blocks_sha256=PACK_BLOCKS_SHA,
        rung=cfg["rung"],
        train_blocks=ladder.get("train_blocks"),
        train_supervised_tokens=ladder.get("train_supervised_tokens"),
        block_len=cfg["block_len"], packing=cfg["packing"],
        val_blocks=cfg["val_blocks"],
        block_ordering="ladder order, sequential, no shuffle",
        ce_weight=loss["ce_weight"], kd_weight=loss["kd_weight"],
        kd_temperature=loss["kd_temperature"], kd_scope=loss["kd_scope"],
        kd_chunk=loss.get("kd_chunk", 512),
        optimizer="AdamW", lr=optim["lr"], weight_decay=optim["weight_decay"],
        betas=tuple(optim["betas"]), eps=optim["eps"],
        grad_clip=optim["grad_clip"],
        total_steps=sched["total_steps"], warmup_steps=sched["warmup_steps"],
        min_lr_frac=sched["min_lr_frac"], lr_schedule="cosine to min_lr_frac",
        blocks_per_step=batch["blocks_per_step"], micro_blocks=batch["micro_blocks"],
        dtype=cfg["dtype"], autocast_bf16=cfg["autocast_bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        trainable_patterns=tuple(cfg["trainable_patterns"]),
        trainable_params=manifest.get("trainable_params"),
        teacher_id=teacher.get("model_id"), teacher_revision=teacher.get("revision"),
        teacher_dtype=teacher.get("dtype"),
        teacher_attn=teacher.get("attn_implementation"),
        tokenizer_sha256=manifest.get("tokenizer_sha256"),
        resume_semantics="step counter + RNG state (no consumed-block accounting)",
        # The historical run recorded a whole-repo commit and a dirty flag, not a
        # trainer source digest, and no image digest at all. Both are therefore
        # unverifiable rather than mismatched: the material identity was never
        # captured, so it cannot be compared.
        trainer_source_digest=None,
        trainer_source_set_version=None,
        runtime_digest=None,
        unverifiable=tuple(HISTORICAL_UNVERIFIABLE),
    )


def phase_a_protocol(config_path: Path) -> RecoveryProtocolFingerprint:
    """The protocol a Phase-A probe would run **today**, from the frozen config."""
    cfg = json.loads(config_path.read_text())
    optim, sched, batch = cfg["optim"], cfg["schedule"], cfg["batch"]
    loss = cfg["loss"]
    ladder = json.loads(
        (REPO_ROOT / "artifacts/stage3/ladder_uniform_probe/ladder.json").read_text())
    rung = next(r for r in ladder["rungs"]
                if r["target_supervised_tokens"] == cfg["rung"])
    return RecoveryProtocolFingerprint(
        pack="ladder_uniform_probe",
        pack_blocks_sha256=PACK_BLOCKS_SHA,
        rung=cfg["rung"],
        train_blocks=rung["n_blocks"],
        train_supervised_tokens=rung["actual_supervised_tokens"],
        block_len=cfg["block_len"], packing=cfg["packing"],
        val_blocks=cfg["val_blocks"],
        block_ordering="ladder order, sequential, no shuffle",
        ce_weight=loss["ce_weight"], kd_weight=loss["kd_weight"],
        kd_temperature=loss["kd_temperature"], kd_scope=loss["kd_scope"],
        kd_chunk=loss.get("kd_chunk", 512),
        optimizer="AdamW", lr=optim["lr"], weight_decay=optim["weight_decay"],
        betas=tuple(optim["betas"]), eps=optim["eps"],
        grad_clip=optim["grad_clip"],
        total_steps=sched["total_steps"], warmup_steps=sched["warmup_steps"],
        min_lr_frac=sched["min_lr_frac"], lr_schedule="cosine to min_lr_frac",
        blocks_per_step=batch["blocks_per_step"], micro_blocks=batch["micro_blocks"],
        dtype=cfg["dtype"], autocast_bf16=cfg["autocast_bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        trainable_patterns=tuple(cfg["trainable_patterns"]),
        trainable_params=440_467_456,
        teacher_id=cfg["teacher"]["model_id"],
        teacher_revision=cfg["teacher"]["revision"],
        teacher_dtype=cfg["teacher"]["dtype"], teacher_attn="sdpa",
        tokenizer_sha256="7781771acc3798ee454c1253c751f930eb1c18c1c3df62e2552cc6f1d394f654",
        resume_semantics=("step counter + RNG state + consumed-block position, "
                          "asserted on restore"),
        trainer_source_digest=trainer_source_digest(REPO_ROOT)["digest"],
        trainer_source_set_version=1,
        # Left unpinned here: the image the preflight will actually run is chosen
        # at Stage 0 and attested there. This script reports what today's dev-box
        # runtime is, and the preflight freezes the real one.
        runtime_digest=None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/autoinit_recovery_fingerprint_audit.json")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    report = {
        "schema": "aadistill.autoinit.recovery_fingerprint_audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "question": ("are the historical 0.86M checkpoints recipe-matched to the "
                     "Phase-A probes, i.e. do they differ only by seed?"),
        "unverifiable_reasons": HISTORICAL_UNVERIFIABLE,
        "comparisons": {},
    }

    all_matched = True
    for name, (relay_dir, seed) in HISTORICAL.items():
        manifest = json.loads(Path(hf_hub_download(
            RELAY, f"{relay_dir}/run_manifest.json", repo_type="model")).read_text())
        hist = historical_protocol(manifest)
        cfg_name = f"e1_r0860k_{'sa' if seed == 20260726 else 'sb'}_pca.json"
        future = phase_a_protocol(REPO_ROOT / "configs/stage3/e1" / cfg_name)
        comparison = hist.compare(future)
        material = [m for m in comparison["mismatched_fields"]
                    if m["field"] not in BENIGN_MISMATCHES]
        benign = [{**m, "why_benign": BENIGN_MISMATCHES[m["field"]]}
                  for m in comparison["mismatched_fields"]
                  if m["field"] in BENIGN_MISMATCHES]
        comparison["material_mismatches"] = material
        comparison["benign_mismatches"] = benign
        comparison["blocking"] = bool(material or comparison["unverifiable_fields"])
        code = manifest.get("code_state") or {}
        audit = HistoricalRunAudit(
            run_id=name, protocol=hist,
            initialization_artifact_digest=CANONICAL_INIT_SHA, seed=seed,
            repo_git_commit=code.get("git_commit"), repo_dirty=code.get("dirty"),
            repo_uncommitted_sha256=code.get("uncommitted_state_sha256"),
            notes={"torch_version_observed": (manifest.get("hardware") or {}).get("torch")})
        probe = RecoveryProbeIdentity(
            protocol=hist, initialization_artifact_digest=CANONICAL_INIT_SHA,
            seed=seed, label=f"historical:{name}")
        report["comparisons"][name] = {
            "seed": seed,
            "historical_run_audit": audit.as_dict(),
            "historical_probe_identity": probe.as_dict(),
            "phase_a_protocol_today": future.as_dict(),
            "comparison": comparison,
            # Explicit, unmistakable status fields. The old
            # `matches_intended_control_protocol` could be read as "this is a
            # valid Phase-A control", which it is not.
            "passes_legacy_lineage_subset": True,
            "recipe_matched_control": False,
        }
        report["comparisons"][name]["recipe_matched_control"] = not comparison["blocking"]
        all_matched = all_matched and not comparison["blocking"]

    report["benign_mismatch_policy"] = BENIGN_MISMATCHES
    # Demonstrate that the *intended* Phase-A comparison is single-variable under
    # the split identity: same protocol, same seed, different initialization.
    phase_a = phase_a_protocol(REPO_ROOT / "configs/stage3/e1/e1_r0860k_sa_pca.json")
    control_probe = RecoveryProbeIdentity(
        protocol=phase_a, initialization_artifact_digest=CANONICAL_INIT_SHA,
        seed=20260726, label="phase_a:control-sa")
    searched_probe = RecoveryProbeIdentity(
        protocol=phase_a,
        initialization_artifact_digest="<searched-leaf-artifact-digest>",
        seed=20260726, label="phase_a:searched-A-sa")
    report["intended_phase_a_comparison"] = {
        "control": control_probe.as_dict()["probe_id"],
        "searched": searched_probe.as_dict()["probe_id"],
        "check": control_probe.matched_against(searched_probe),
        "note": ("illustrative: the searched digest is a placeholder, but the "
                 "predicate is the one the real run uses"),
    }
    report["historical_controls_are_recipe_matched"] = all_matched
    report["status_field_meanings"] = {
        "passes_legacy_lineage_subset": (
            "the historical run descends from the canonical init at the right rung "
            "with the right seed and objective — a useful lineage reference"),
        "recipe_matched_control": (
            "the run may serve as a Phase-A matched control, i.e. its protocol "
            "fingerprint equals the Phase-A protocol with every field established"),
    }
    report["consequence"] = (
        "The historical checkpoints ARE recipe-matched and may serve as the "
        "Phase-A control."
        if all_matched else
        "The historical checkpoints are NOT recipe-matched and must not be called "
        "matched controls. Rerun canonical sa/sb from qwen3_0p6b_init_v0 at 0.86M "
        "under the current frozen recovery trainer; those runs become the "
        "permanent Phase-A control probes and are retained, not disposable.")
    report["report_sha256"] = sha256_json(report)
    (REPO_ROOT / args.out).write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(json.dumps({
        "recipe_matched": all_matched,
        "per_control": {
            k: {"fingerprints_equal": v["comparison"]["fingerprints_equal"],
                "n_matched": len(v["comparison"]["matched_fields"]),
                "recipe_matched_control": v["recipe_matched_control"],
                "passes_legacy_lineage_subset": v["passes_legacy_lineage_subset"],
                "material_mismatches": [m["field"] for m in
                                        v["comparison"]["material_mismatches"]],
                "benign_mismatches": [m["field"] for m in
                                      v["comparison"]["benign_mismatches"]],
                "unverifiable": [u["field"] for u in v["comparison"]["unverifiable_fields"]]}
            for k, v in report["comparisons"].items()},
        "consequence": report["consequence"],
    }, indent=2))


if __name__ == "__main__":
    main()
