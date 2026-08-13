"""Preflight Stage 0: attest the runtime, materialize the protocol, freeze it.

    PYTHONPATH=src python scripts/autoinit/attest_protocol.py \
        --image-digest sha256:... --out logs/autoinit_phase_a_protocol_attested.json

Runs **on the pod**, first, before anything is measured or trained. It fills the
three environment fields the preregistration necessarily left unknown —
``trainer_source_digest``, ``trainer_source_set_version``, ``runtime_digest`` —
and emits the frozen artifact that Stage 2 compares every control run against.

Why this exists as a separate step rather than being folded into the control run:

* the preregistered protocol carries ``runtime_digest: null``, because the image
  is chosen when the pod is created, after preregistration is written. Comparing
  a control against that object would accept a control trained under any runtime,
  since ``None == None``;
* filling it later, from the control run itself, would mean the control defines
  its own protocol — which is not a check.

This is not adaptive modification of the experiment. Stage 0 fills preregistered
*environment* fields before any candidate behaviour or search result exists. It
cannot see a candidate; there is none yet. If the attested runtime contradicts a
value that *was* preregistered, ``materialized()`` raises rather than overwriting
it, and the session stops before Stage 2.

Verification, not just recording: every input artifact hash is checked against its
pin here, so a wrong pack or a wrong canonical init stops the session at $0.30
rather than after two control runs.
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
    PREFLIGHT_PLAN_V1,
    RecoveryAdmissionError,
    RuntimeEnvironmentFingerprint,
    trainer_source_digest,
)
from aadistill.infrastructure.manifest import sha256_file, sha256_json  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts/autoinit"))
from compare_recovery_fingerprints import phase_a_protocol  # noqa: E402

#: Inputs whose bytes must be what the preregistration says they are. Paths are
#: resolved on the pod; a missing file is a stop condition, not a skip.
PINNED_INPUTS = {
    "canonical_init_weights": (
        "artifacts/stage1/qwen3_0p6b_init_v0/checkpoint/model.safetensors",
        "86fbba78e8a2a32481ca77e5ac362ed1f17a39dbc30bcbc952cabd5df2633e54"),
    "recovery_pack_blocks": (
        "artifacts/stage3/ladder_uniform_probe/blocks.npz",
        "6f324cb0f37bc0f07128e554ce8c161879419537478950496534f75fcecb249c"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-digest", required=True,
                    help="the container image digest this pod is running")
    ap.add_argument("--attention-backend", default="sdpa")
    ap.add_argument("--config", default="configs/stage3/e1/e1_r0860k_sa_pca.json")
    ap.add_argument("--skip-input-hashes", action="store_true",
                    help="dry-run the handshake off-pod, where inputs are absent")
    ap.add_argument("--out", default="logs/autoinit_phase_a_protocol_attested.json")
    args = ap.parse_args()

    report = {
        "schema": "aadistill.autoinit.attested_protocol/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_stage": 0,
        "preflight_plan_hash": PREFLIGHT_PLAN_V1.plan_hash,
        "purpose": ("materialize and freeze the Phase-A recovery protocol before "
                    "any control is trained; Stage 2 compares against this hash, "
                    "not against the preregistration"),
    }

    runtime = RuntimeEnvironmentFingerprint.observe(
        image_digest=args.image_digest, attention_backend=args.attention_backend)
    runtime.require_pinned()
    trainer = trainer_source_digest(REPO_ROOT)

    inputs = {}
    ok = True
    for name, (rel, pinned) in PINNED_INPUTS.items():
        path = REPO_ROOT / rel
        if args.skip_input_hashes:
            inputs[name] = {"path": rel, "pinned_sha256": pinned,
                            "checked": False, "why": "--skip-input-hashes"}
            continue
        if not path.is_file():
            inputs[name] = {"path": rel, "pinned_sha256": pinned, "present": False,
                            "match": False}
            ok = False
            continue
        actual = sha256_file(path)
        inputs[name] = {"path": rel, "pinned_sha256": pinned,
                        "actual_sha256": actual, "present": True,
                        "match": actual == pinned}
        ok = ok and actual == pinned
    report["pinned_inputs"] = inputs
    report["pinned_inputs_ok"] = ok

    preregistered = phase_a_protocol(REPO_ROOT / args.config)
    report["preregistered_protocol"] = preregistered.as_dict()
    try:
        attested = preregistered.materialized(runtime=runtime, trainer_source=trainer)
    except RecoveryAdmissionError as exc:
        report["stage_0_passed"] = False
        report["error"] = str(exc)
        (REPO_ROOT / args.out).write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(f"STOP: protocol drift at attestation: {exc}")

    attested.require_materialized(context="Stage 0 attestation")
    report["runtime"] = runtime.as_dict()
    report["trainer_source"] = trainer
    report["attested_protocol"] = attested.as_dict()
    report["attested_protocol_fingerprint"] = attested.fingerprint
    report["dry_run"] = bool(args.skip_input_hashes)
    # A dry run exercises the handshake off-pod, where the inputs do not exist. It
    # must never be able to record a Stage-0 pass: that is what Stage 2 keys on.
    report["stage_0_passed"] = bool(ok and not args.skip_input_hashes)
    report["binding_invariant"] = (
        "every permanent control and every later searched probe must record this "
        "exact protocol fingerprint; RecoveryProbeIdentity.require_attested() is "
        "the check")
    report["next"] = ("Stage 1 cheap machine gates. Stage 2 must not start until "
                      "Stage 1 records a pass.")
    report["report_sha256"] = sha256_json(report)

    (REPO_ROOT / args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "stage_0_passed": report["stage_0_passed"],
        "attested_protocol_fingerprint": report["attested_protocol_fingerprint"],
        "runtime_digest": runtime.digest,
        "trainer_source_digest": trainer["digest"],
        "pinned_inputs_ok": ok,
    }, indent=2))
    if not ok:
        raise SystemExit("STOP: a pinned input does not match its hash")


if __name__ == "__main__":
    main()
