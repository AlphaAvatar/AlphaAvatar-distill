#!/usr/bin/env python3
"""Validate the external durable backend with ONE real probe. Correctness only.

    R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
    PYTHONPATH=src:scripts python \
        scripts/autoinit/validate_durable_backend.py \
        --endpoint https://<account>.r2.cloudflarestorage.com \
        --bucket <bucket> [--probe <name>] [--write]

WHAT IT PROVES, and it is deliberately not a benchmark:

    canonical probe -> source identity -> upload -> READBACK of the stored
    bytes -> presigned plan -> fetch to a NEW path -> re-identify -> exact
    match

Every step is the production path: `upload_checkpoint` and `restore_checkpoint`
from the generic interface, `S3CompatibleStore` and `PresignedHttpStore` as
backends, `verify_transferred_leaf` for identity. Nothing is stubbed and
nothing is re-saved -- the object at the far end must be the same scientific
artifact, merely transported.

WHAT IT DOES NOT PROVE. A throughput. The transfer produces a duration and the
record keeps it as DIAGNOSTIC evidence, because a network rate varies with time
of day, routing and provider load: one measurement is an observation, not a
constant, and re-running to refine it spends time manufacturing false
precision. The continuation's money comes from a named conservative reserve.

CREDENTIALS are read from the environment and never printed, logged, committed
or written into the record. A presigned plan is a secret while it is valid and
only its `redacted()` form is ever recorded.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in ("src", "scripts"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

from aadistill.initialization.adapters import register_builtin_adapters  # noqa: E402
from aadistill.initialization.specs.arch import get_adapter  # noqa: E402
from aadistill.runtime.durable_store import (DurableStoreError,  # noqa: E402
                                             restore_checkpoint,
                                             upload_checkpoint)

from experiments.phase_c2 import behavioural_governance as BG  # noqa: E402
from experiments.durable_stores import (PresignedHttpStore,  # noqa: E402
                                        S3CompatibleStore)

STORE = Path("/home/ecs-user/aad-artifacts/phase_c2_behavioural")
ANALYSIS = ("logs/stages/stage-1/phase_c2_behavioural/analyses/"
            "durable_backend_validation.json")


def validate(*, endpoint: str, bucket: str, probe: str, key_prefix: str,
             access_key_env: str, secret_key_env: str) -> dict:
    register_builtin_adapters()
    src = STORE / BG.CAMPAIGN_ID / "attempt5" / probe
    if not src.is_dir():
        raise SystemExit(f"no canonical probe at {src}")
    ack = json.loads((src / "durable_ack.json").read_text())["identity"]
    adapter = get_adapter("qwen3")
    store = S3CompatibleStore(endpoint_url=endpoint, bucket=bucket,
                              access_key_env=access_key_env,
                              secret_key_env=secret_key_env)
    key = f"{key_prefix}/{probe}"
    scratch = Path("/home/ecs-user/aad-scratch")

    #: 1. UPLOAD, verified by reading the stored bytes back. `verified=True`
    #: here is what a producer would release its only local copy on, so it may
    #: not mean "the PUT returned".
    up = upload_checkpoint(store, src, key, adapter=adapter,
                           arch_signature=ack["arch_signature"],
                           num_parameters=ack["num_parameters"],
                           scratch=scratch)

    #: 2. A PRESIGNED PLAN, which is what would cross to a pod. The credential
    #: stays here.
    plan = store.presign_fetch_plan(key)

    #: 3. FETCH IT BACK over plain HTTP, exactly as a pod would, into a path
    #: that did not exist.
    fetch = Path(tempfile.mkdtemp(dir=scratch)) / "arrived"
    t0 = time.time()
    down = restore_checkpoint(PresignedHttpStore(plan), key, fetch,
                              adapter=adapter, identity=up.identity)
    observed = time.time() - t0

    matched = {f: ack.get(f) == down.identity.get(f)
               for f in ("artifact_digest", "weights_digest",
                         "single_shard_sha256", "arch_signature",
                         "num_parameters", "config_sha256")}
    try:
        shutil.rmtree(fetch.parent, ignore_errors=True)
    finally:
        pass

    return {
        "schema": "aadistill.autoinit.durable_backend_validation/v1",
        "authorizes": "nothing",
        "probe": probe, "bytes": up.bytes, "gib": round(up.bytes / 2**30, 3),
        "key": key, "endpoint_host": endpoint.split("//")[-1].split("/")[0],
        "bucket": bucket,
        "upload": {"verified": up.verified,
                   "verified_by": up.as_dict()["verified_by"],
                   "source_identity_matched":
                       up.as_dict().get("source_identity_matched")},
        "presigned_plan": plan.redacted(),
        "restore": {"verified": down.verified, "n_files": down.n_files,
                    "to_a_new_path": True},
        "identity_fields_matched": matched,
        "all_fields_matched": all(matched.values()),
        "observed": {
            "upload_seconds": round(up.seconds, 1),
            "fetch_seconds": round(observed, 1),
            "_diagnostic_only": (
                "durations, not a transport rate. Network throughput varies "
                "with time of day, routing and provider load, so one "
                "measurement is an observation and not a constant. The "
                "continuation's money comes from a named conservative "
                "reserve, and this figure is never promoted to it."),
        },
        "_credentials": ("read from the environment by name and never "
                         "printed, logged or recorded"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--probe", default="confirmation.B.s1936324010")
    ap.add_argument("--key-prefix", default=BG.CAMPAIGN_ID)
    ap.add_argument("--access-key-env", default="R2_ACCESS_KEY_ID")
    ap.add_argument("--secret-key-env", default="R2_SECRET_ACCESS_KEY")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    try:
        out = validate(endpoint=a.endpoint, bucket=a.bucket, probe=a.probe,
                       key_prefix=a.key_prefix,
                       access_key_env=a.access_key_env,
                       secret_key_env=a.secret_key_env)
    except DurableStoreError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    if a.write:
        p = REPO / ANALYSIS
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {ANALYSIS}")
    return 0 if out["all_fields_matched"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
