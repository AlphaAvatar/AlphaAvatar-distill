"""Emit the v2 generation-runtime compatibility artifact. Zero cost.

    PYTHONPATH=src python scripts/autoinit/emit_protocol_compat_v2.py

Reconciles the Stage-3 evaluation protocol with a currently-drawable runtime
**without touching either historical artifact**. It reconstructs the comparable
identity independently from two saved engine-probe evidence files — Stage 3's
(NVIDIA driver 580.159.03) and Phase-A attempt 4's (580.126.09) — and records
that they resolve to the same v2 identity while the exact driver patch is the
only formerly-material runtime difference.

`logs/autoinit_stage3_complete/{attested_evaluation_protocol,materialized_thresholds}.json`
are read and hashed, never written. `250f72ef…` remains the protocol those
controls actually attested; this artifact supersedes the *comparability rule*
applied to it, not the fact.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aadistill.autoinit.generation import (  # noqa: E402
    RecoveryEvaluationProtocol, declared_generation_protocol,
    generation_source_digest,
)
from aadistill.autoinit.generation_compat import (  # noqa: E402
    GENERATION_RUNTIME_COMPARABILITY_V2, comparable_generation_identity,
    require_comparable, split_image_identity,
)
from aadistill.autoinit.recovery import recovery_scoring_contract  # noqa: E402
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

STAGE3_DIR = "logs/autoinit_stage3_complete"
STAGE3_PROTOCOL_HASH = (
    "250f72efbd43b86a475e8dda293b45f07ee61a4d858e147f4a5bd7681c32c2e4")
OBSERVED_FIELDS = ("vllm_version", "transformers_version", "torch_version",
                   "runtime_digest", "dtype", "gpu_memory_utilization",
                   "max_num_seqs", "max_num_batched_tokens", "enforce_eager",
                   "tokenizer_sha256", "chat_template_sha256",
                   "resolved_context", "context_source", "stop_token_ids")


def protocol_from_probe(probe: dict, repo: Path) -> RecoveryEvaluationProtocol:
    """Rebuild an evaluation protocol from one saved engine probe."""
    gen = declared_generation_protocol().materialized(
        generation_source_digest=generation_source_digest(repo)["digest"],
        degeneration_source_digest=sha256_file(
            repo / "src/aadistill/evaluation/degeneration.py"))
    gen = gen.materialized(**{
        k: (tuple(probe[k]) if k == "stop_token_ids" else probe[k])
        for k in OBSERVED_FIELDS})
    gen.require_materialized(context="compat reconstruction")
    battery = json.loads(
        (repo / "artifacts/stage3/recovery_search_v2/manifest.json").read_text())
    scoring = recovery_scoring_contract(repo)
    return RecoveryEvaluationProtocol(
        generation=gen, scoring_contract=scoring["contract"],
        scoring_digest=scoring["digest"], battery_artifact=battery["artifact"],
        battery_manifest_sha256=battery["manifest_sha256"],
        battery_content_sha256=battery["content_sha256"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt4-probe", required=True,
                    help="the engine_probe.json collected from Phase-A attempt 4")
    ap.add_argument("--out", default="logs/autoinit_phase_a_protocol_compat_v2.json")
    args = ap.parse_args()

    s3_att_path = REPO_ROOT / STAGE3_DIR / "attested_evaluation_protocol.json"
    s3_thr_path = REPO_ROOT / STAGE3_DIR / "materialized_thresholds.json"
    s3_probe_path = REPO_ROOT / STAGE3_DIR / "engine_probe.json"

    s3_att = json.loads(s3_att_path.read_text())
    s3_probe = json.loads(s3_probe_path.read_text())
    a4_probe = json.loads(Path(args.attempt4_probe).read_text())

    if s3_att["evaluation_protocol_hash"] != STAGE3_PROTOCOL_HASH:
        raise SystemExit(
            f"the Stage-3 attestation declares "
            f"{s3_att['evaluation_protocol_hash']}, not the pinned "
            f"{STAGE3_PROTOCOL_HASH}; refusing to bind to a moved artifact")

    # Historical: from the attested protocol exactly as recorded, expanded with
    # the runtime block its own engine probe saved.
    historical = comparable_generation_identity(
        protocol=s3_att["evaluation_protocol"], runtime=s3_probe["runtime"])
    # Current: rebuilt INDEPENDENTLY from attempt 4's own probe.
    a4_protocol = protocol_from_probe(a4_probe, REPO_ROOT)
    current = comparable_generation_identity(
        protocol=a4_protocol.as_dict(), runtime=a4_probe["runtime"])

    comparison = require_comparable(current, historical, context="compat v2 migration")

    artifact = {
        "schema": "aadistill.autoinit.generation_runtime_compat/v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "supersede the comparability RULE applied to the Stage-3 evaluation "
            "protocol, without altering the protocol, the thresholds, or any "
            "historical artifact"),
        "rule": GENERATION_RUNTIME_COMPARABILITY_V2.as_dict(),
        "bound_to_historical_protocol": {
            "evaluation_protocol_hash": STAGE3_PROTOCOL_HASH,
            "attested_artifact": f"{STAGE3_DIR}/attested_evaluation_protocol.json",
            "attested_report_sha256": s3_att["report_sha256"],
            "engine_probe": f"{STAGE3_DIR}/engine_probe.json",
        },
        "historical_artifacts_unmodified": {
            "note": ("read and hashed here, never written. These hashes are the "
                     "evidence that the migration changed nothing historical."),
            "attested_evaluation_protocol.json": sha256_file(s3_att_path),
            "materialized_thresholds.json": sha256_file(s3_thr_path),
            "engine_probe.json": sha256_file(s3_probe_path),
        },
        "evidence": {
            "historical": {
                "source": f"{STAGE3_DIR}/engine_probe.json",
                "image": split_image_identity(s3_probe["runtime"]["image_digest"]),
                "v1_evaluation_protocol_hash": s3_att["evaluation_protocol_hash"],
                "v2_comparable_identity": historical["comparable_identity"],
            },
            "attempt_4": {
                "source": args.attempt4_probe,
                "image": split_image_identity(a4_probe["runtime"]["image_digest"]),
                "v1_evaluation_protocol_hash": a4_protocol.evaluation_protocol_hash,
                "v2_comparable_identity": current["comparable_identity"],
                "reconstructed_independently": True,
            },
        },
        "result": {
            "v1_hashes_differ": (s3_att["evaluation_protocol_hash"]
                                 != a4_protocol.evaluation_protocol_hash),
            "v2_identities_equal": comparison["identities_equal"],
            "comparable_identity": historical["comparable_identity"],
            "driver_branch_equal": comparison["driver_branch_equal"],
            "driver_patch_differs": comparison["driver_patch_differs"],
            "sole_formerly_material_difference": (
                "the exact NVIDIA driver patch, reached through "
                "runtime_digest -> runtime.image_digest, whose suffix is the "
                "host driver version rather than a container image digest"),
            "comparison": comparison,
        },
        "what_this_does_not_claim": [
            "that NVIDIA drivers are universally irrelevant to generation",
            "that a driver BRANCH change is inert -- it still fails closed",
            "that the Stage-3 thresholds, pooled_counts@v2, the search, the "
            "recovery design, the seeds or the science plan change in any way",
        ],
        "phase_a_stage0_requirement": (
            "BOTH: the thresholds artifact still declares the untouched "
            f"{STAGE3_PROTOCOL_HASH}, AND the live protocol is comparable to that "
            "historical protocol under generation_runtime_comparability@v2"),
    }
    artifact["report_sha256"] = sha256_json(artifact)
    (REPO_ROOT / args.out).write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "v2_comparable_identity": historical["comparable_identity"],
        "v2_identities_equal": comparison["identities_equal"],
        "historical_driver": comparison["historical_driver"],
        "live_driver": comparison["live_driver"],
        "driver_branch_equal": comparison["driver_branch_equal"],
        "report_sha256": artifact["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
